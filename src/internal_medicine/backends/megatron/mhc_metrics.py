"""mHC (Manifold-Constrained Hyper-Connections) metric compute functions.

Pure, stateless tensor -> tensor helpers for the ``mhc_health`` monitor. They
operate on the three mappings a ``HyperConnectionModule`` produces per token:

- ``h_pre``  [s, b, n]      — stream aggregation gate (sigmoid)
- ``h_post`` [s, b, n]      — stream expansion gate (2*sigmoid)
- ``h_res``  [s, b, n, n]   — Sinkhorn doubly-stochastic residual-mixing matrix

``n = num_residual_streams``, ``s`` = sequence, ``b`` = micro-batch.

The ``amax_gain`` diagnostic follows the mHC paper: the max-abs **row** sum of a
mixing matrix bounds the worst-case forward-pass expansion, and the max-abs
**column** sum bounds the backward-pass expansion. For a single doubly-stochastic
``h_res`` both sit at ~1.0; on the *composite* mapping (cumulative product of
``h_res`` across layers) they drift away from 1.0 with depth, flagging
residual-stream amplification.

All functions return 0-dim GPU tensors (``stream_gram_stats`` additionally returns one ``[n]``
vector) and never sync the host (no ``.item()`` / ``.cpu()``), so they are safe to call from a
forward hot path. See ``.claude/skills/monitor-hook-perf-rules``.
"""

import math

import torch


def amax_gain(mat: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-token max-abs {row|col} sum of a batched ``[..., n, n]`` matrix, meaned over tokens.

    ``dim=-1`` sums over columns -> per-row sums (forward gain); ``dim=-2`` sums
    over rows -> per-column sums (backward gain). ``sum(dim)`` collapses one
    ``n``-axis to ``[..., n]``; ``abs().amax(-1)`` takes the worst stream per token
    (a 1-per-token scalar); ``mean()`` averages over all tokens.

    Returns a 0-dim tensor on ``mat``'s device/dtype (fp32 from ``compute_mappings``).
    """
    sums = mat.sum(dim=dim)  # [..., n]
    return sums.abs().amax(dim=-1).mean()  # 0-dim


def gate_stats(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and (unbiased) std of a gate tensor ``h`` [s, b, n] over all elements.

    Returns two 0-dim tensors ``(mean, std)``; no host sync.
    """
    return h.mean(), h.std()


def erase_beta_stats(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token ``n - tr(H_res)``: mean and (unbiased) std over tokens.

    This is exactly the erase strength ``beta`` of the rank-1 ablation
    ``H_res = I - beta * u u^T`` (``||u|| = 1``, so ``tr = n - beta``) — the quantity
    whose convergence (0 / (0,2) / 2) decides that experiment. On any other ``h_res``
    it reads as the trace deficit: 0 for the identity, ``n`` minus the diagonal mass
    for a doubly-stochastic matrix.

    Returns two 0-dim tensors ``(mean, std)``; no host sync.
    """
    n = mat.shape[-1]
    beta = n - mat.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [s, b]
    return beta.mean(), beta.std()


def outer_deviation(mat: torch.Tensor, h_pre: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of ``H_res - h_post (x) h_pre`` averaged over tokens.

    ``apply_h_res`` computes ``out_i = sum_j H_res[i,j] x_j``, and the sublayer path
    writes ``h_post_i * sum_j h_pre_j x_j``, so ``h_post (x) h_pre`` (entry ``[i, j] =
    h_post_i * h_pre_j``) is the rank-1 matrix that shares ``H_res``'s index order.
    The metric is therefore how much of the residual mix is NOT the read-write product
    — i.e. how far the mix is from re-doing the sublayer path's routing.

    Returns a 0-dim tensor; no host sync.
    """
    outer = h_post.unsqueeze(-1) * h_pre.unsqueeze(-2)  # [s, b, n, n]
    return (mat - outer).pow(2).sum(dim=(-2, -1)).sqrt().mean()


def orthogonality_deviation(mat: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token ``||H_res^T H_res - I||_F``: token mean, plus its max/median ratio.

    The mean measures how far ``h_res`` is from orthogonal: 0 for a perfectly
    orthogonal matrix, positive for doubly-stochastic (Sinkhorn baseline).

    The ratio is the tail detector the mean cannot be. An iterative
    orthogonalization (e.g. the R3 Schulz path) that drops a ~1e-3 fraction of
    tokens out of its convergence basin keeps the mean displaying 0.0000 while
    those tokens dominate the backward pass, so the mean alone cannot separate
    "exact everywhere" from "exact except for a tail". ``eps`` is the deviation
    level treated as indistinguishable from exact (fp32 rounding floor): an
    exactly-orthogonal ``h_res`` reads 1.0 rather than 0/0.

    Returns two 0-dim GPU tensors ``(mean, max_median_ratio)``; no host sync.
    """
    s, b, n, _ = mat.shape
    m = mat.reshape(s * b, n, n)
    gram = torch.bmm(m.transpose(-2, -1), m)
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype).unsqueeze(0)
    frob = (gram - eye).pow(2).sum(dim=(-2, -1)).sqrt()  # [s*b], one scalar per token
    return frob.mean(), (frob.amax() + eps) / (frob.median() + eps)


_COMPLEMENT_BASIS: dict = {}


def _complement_basis(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached ``[n, n-1]`` orthonormal basis of ``1^perp`` (truncated Helmert): ``Q^T 1 = 0``.

    Only the span matters here — the sigma spectrum below is invariant to which basis of
    ``1^perp`` is used, so this need not match the basis the H_res parameterization picked.
    """
    key = (n, device, dtype)
    q = _COMPLEMENT_BASIS.get(key)
    if q is None:
        i = torch.arange(n, device=device)[:, None]
        j = torch.arange(n - 1, device=device)[None, :]
        col = (j + 1).to(dtype)
        q = (i <= j).to(dtype) - col * (i == j + 1).to(dtype)
        q = q / (col * (col + 1)).sqrt()
        _COMPLEMENT_BASIS[key] = q
    return q


def _eigvalsh_sym3(a: torch.Tensor) -> torch.Tensor:
    """Ascending eigenvalues of a batched symmetric ``[..., 3, 3]``, in closed form.

    ``linalg.eigvalsh`` / ``svdvals`` round-trip cuSOLVER's info flag and sync the host, which a
    forward hook must not do; the trigonometric (Cardano) route is branch-free elementwise algebra.
    ``p2 = 0`` (the isotropic case, e.g. an exactly orthogonal ``H_res``) returns ``tr/3`` exactly:
    ``p = 0`` kills both cosine terms, and the clamped denominator only guards ``0/0``.
    """
    d = a.diagonal(dim1=-2, dim2=-1)
    q = d.mean(dim=-1)
    b00, b11, b22 = d[..., 0] - q, d[..., 1] - q, d[..., 2] - q
    b01, b02, b12 = a[..., 0, 1], a[..., 0, 2], a[..., 1, 2]
    off = b01.pow(2) + b02.pow(2) + b12.pow(2)
    p2 = (b00.pow(2) + b11.pow(2) + b22.pow(2) + 2 * off) / 6
    p = p2.sqrt()
    det = b00 * (b11 * b22 - b12.pow(2)) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)
    denom = (2 * p2 * p).clamp_min(torch.finfo(a.dtype).tiny)
    phi = torch.acos((det / denom).clamp(-1.0, 1.0)) / 3.0
    hi = q + 2 * p * torch.cos(phi)
    lo = q + 2 * p * torch.cos(phi + 2.0 * math.pi / 3.0)
    return torch.stack([lo, 3 * q - hi - lo, hi], dim=-1)


def sigma_stats(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``Sigma`` — the singular values of ``H_res`` on ``1^perp`` — as a token min and mean.

    Any mean-preserving mix (``H 1 = 1``, ``1^T H = 1^T``: the R8 sphere AND the Sinkhorn
    baseline) keeps ``1^perp`` invariant, so ``Q^T H^T H Q = Sigma^2`` with ``Q`` any orthonormal
    basis of it. Equivalently ``svdvals(H_res) = {1} u {|sigma_i|}``: the ``1`` direction is fixed
    and carries no information, and ``Sigma`` is the whole learnable part of the operator.

    This is R8a's only criterion. ``sigma -> 1`` is the model asking for a mean-preserving
    ISOMETRY (which is what R8b hard-codes, so R8b must read a flat 1.0 here); ``sigma -> 0`` is
    ``H_res -> J``, every stream replaced by the stream mean, i.e. a rank-1 collapse that is
    strictly weaker than a plain residual. ``min`` is reported alongside ``mean`` because one
    collapsed direction out of ``n-1`` is exactly what a mean hides (the ``amax_gain`` lesson).

    On a non-affine ``H_res`` (R1 identity, R3-Cayley, R4 erase) ``1^perp`` is not invariant and
    this reads the spectrum of ``H^T H`` compressed onto it — still bounded by ``||H_res||_2``,
    but no longer a factorization of the operator.

    Returns two 0-dim GPU tensors ``(sigma_min, sigma_mean)``; no host sync.
    """
    qb = _complement_basis(mat.shape[-1], mat.device, mat.dtype)
    g = qb.transpose(-2, -1) @ (mat.transpose(-2, -1) @ mat) @ qb  # [..., n-1, n-1], = Sigma^2
    ev = _eigvalsh_sym3(g) if g.shape[-1] == 3 else torch.linalg.eigvalsh(g)
    sig = ev.clamp_min(0).sqrt()
    return sig.amin(dim=-1).mean(), sig.mean()


def so4_angle_stats(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The TWO rotation angles of an ``n = 4`` ``H_res``, as token means, ascending.

    An element of ``SO(4)`` is not one rotation: it splits into independent rotations by ``alpha``
    and ``gamma`` on two invariant 2-planes, spectrum ``{e^{+-i alpha}, e^{+-i gamma}}``, and *that
    pair* is the conjugacy class. ``erase_beta_stats`` reads only ``4 - tr = 4 - 2(cos alpha +
    cos gamma)``, a sum — two runs at very different angle pairs can share a ``beta``. These two
    series are the sufficient statistic ``beta`` is not, so they measure how much of ``SO(4)`` the
    mix actually uses (R3-Cayley and R9 alike).

    Both invariants come from elementwise algebra, no matmul and no eigendecomposition
    (``eigvalsh`` / ``svdvals`` sync the host — see ``_eigvalsh_sym3``)::

        s     = tr(Q) / 2                    = cos alpha + cos gamma
        tr Q^2 = (Q * Q^T).sum((-2, -1))     = 4 (cos^2 alpha + cos^2 gamma) - 4
        cc    = (s^2 - (tr Q^2 + 4) / 4) / 2 = cos alpha cos gamma

    ``cos alpha, cos gamma`` are then the roots of ``z^2 - s z + cc``. The ``4``s are ``n``-specific:
    this is for ``n = 4`` only, and the caller gates on that.

    Only an ORTHOGONAL ``h_res`` (R3-Cayley, R9, R8b) has a conjugacy class for these to report; the
    clamps keep the formula finite on any other mix, but the numbers then describe no ``SO(4)``
    element.

    Splitting the pair goes through ``sqrt((cos alpha - cos gamma)^2)``, which halves the significant
    digits when the two angles nearly coincide, so from an fp32 ``h_res`` the SPREAD carries a noise
    floor of ~3e-2 rad and is biased outward (``sqrt`` of a squared quantity plus noise). Do not read
    a small ``theta_hi - theta_lo`` as real anisotropy. The floor shrinks as the angles separate
    (~9e-4 rad at the isoclinic ``theta ~ 0.36`` where the R3 run sits, ~2e-7 by ``(1.0, 2.0)``), and
    the mean of the two is always well conditioned — it is a reparameterization of ``tr(Q)``.
    Accumulating in fp64 does not help: the loss happens in ``h_res``'s own fp32 rounding.

    Returns two 0-dim GPU tensors ``(theta_lo_mean, theta_hi_mean)`` in radians; no host sync.
    """
    s = mat.diagonal(dim1=-2, dim2=-1).sum(dim=-1) * 0.5  # [s, b]
    tr_sq = (mat * mat.transpose(-2, -1)).sum(dim=(-2, -1))  # tr(Q^2), no matmul
    cc = 0.5 * (s.pow(2) - 0.25 * (tr_sq + 4.0))
    disc = (s.pow(2) - 4.0 * cc).clamp_min(0).sqrt()
    lo = torch.acos((0.5 * (s + disc)).clamp(-1.0, 1.0))  # larger cos = smaller angle
    hi = torch.acos((0.5 * (s - disc)).clamp(-1.0, 1.0))
    return lo.mean(), hi.mean()


def stream_gram_stats(
    x: torch.Tensor, n: int, eps: float = 1e-6, rel_floor: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Geometry of the ``n`` hidden STREAMS themselves (not the ``H_res`` operator).

    ``orthogonality_deviation`` measures the mixing MATRIX; this measures the frame it acts on.
    Every other metric here reduces over the stream axis (``h_pre.mean()``, per-token row sums),
    so a run where one stream carries the signal and the other ``n-1`` decay into noise — or
    where all ``n`` collapse onto a single direction — looks perfectly healthy in all of them.
    That is the blind spot this closes: stream-norm imbalance, inter-stream cosine and the
    cross-stream CV are the ways the ``n``-stream residual can stop being ``n`` streams.

    Args:
        x: pre-aggregation hidden state ``[s, b, n*C]``, exactly the tensor
            ``HyperConnectionModule.compute_mappings`` receives.
        n: ``num_residual_streams``; read from the module, not inferred from ``x``.
        eps: floor for the norm ratios / cosines (avoids 0/0 on a dead stream).
        rel_floor: floor on ``||m||^2`` for the CV, RELATIVE to the per-token mean stream
            energy. This is the one place raw CV is fragile (``||m|| -> 0`` on a token whose
            streams cancel); an absolute eps would let such a token contribute ~1e6 and
            dominate the mean. Mirrors ``residual_energy_split``'s ``rel_floor``.

    Returns, no host sync:
      * ``stream_norm``               — ``[n]`` per-stream mean L2 norm (per-stream, not meaned
        over the stream axis: a dominant stream is invisible in the mean).
      * ``stream_norm_max_min_ratio`` — per-token max/min stream-norm ratio, meaned over tokens.
      * ``gram_offdiag_signed_mean``  — mean SIGNED cosine between distinct streams, range
        ``[-1, 1]``. Signed, not ``|cos|``, because the absolute mean cannot tell "collapsed
        onto the common mean" (signed -> +1) from "anti-aligned rotation" (signed -> -1): both
        read 1.0. A mean-fixing orthogonal ``H_res`` leaves it at its input value; a general
        (mean-rotating) one can push it negative.
      * ``gram_offdiag_abs_max``      — worst ``|cosine|`` over the off-diagonal, meaned over
        tokens. Stays on ``|cos|``: it is a tail/collapse detector, and a signed max would miss
        an anti-aligned tail.
      * ``stream_cv``                 — cross-stream coefficient of variation, ``sqrt(Var)/||m||``
        with ``m`` the stream mean and ``Var = (1/n) sum_i ||x_i - m||^2``, per token then meaned.
        ``-> 0`` means the streams have collapsed onto their common mean; large means they carry
        independent content. Under a doubly-stochastic stack it shrinks with depth; under an
        orthogonal (mean-fixing) one it holds. Computed from ``gram`` by the parallel-axis
        identity — ``sum_i ||x_i||^2 = tr(gram)`` and ``||m||^2 = sum_ij gram_ij / n^2``, so
        ``Var = tr/n - ||m||^2`` — hence no new large tensor and no extra bmm.
    """
    xs = x.reshape(-1, n, x.shape[-1] // n)  # [T, n, C], view
    # Gram from raw streams, fp32 only on the [T,n,n] output: upcasting x first would cost a
    # transient [T,n,C] fp32 copy (~134MB at s=8192,n=4,C=1024) inside the forward.
    gram = torch.bmm(xs, xs.transpose(-2, -1)).float()  # [T, n, n]
    norms = gram.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()  # [T, n]
    cos = gram / (norms.unsqueeze(-1) * norms.unsqueeze(-2)).clamp_min(eps)

    off = ~torch.eye(n, device=xs.device, dtype=torch.bool)  # [n, n] off-diagonal mask
    off_vals = cos[:, off]  # [T, n*(n-1)], signed

    mean_stream_energy = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / n  # tr(gram)/n, [T]
    m_energy = gram.sum(dim=(-2, -1)) / (n * n)  # ||m||^2, [T]
    var = (mean_stream_energy - m_energy).clamp_min(0)
    cv = (var / m_energy.clamp_min(rel_floor * mean_stream_energy)).sqrt()

    return (
        norms.mean(dim=0),
        ((norms.amax(dim=-1) + eps) / (norms.amin(dim=-1) + eps)).mean(),
        off_vals.mean(),
        off_vals.abs().amax(dim=-1).mean(),
        cv.mean(),
    )


def residual_energy_split(
    residual: torch.Tensor,
    output: torch.Tensor,
    sublayer_out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    n: int,
    rel_floor: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    """Per-token energy split of one hyper-connection update ``out = Q x + w``.

    With ``x`` the ``n x C`` residual, ``Q = h_res``, ``o`` the sublayer output and
    ``w = h_post o^T`` the write (``w_i = h_post_i * o``), the Frobenius energy of the
    update is EXACTLY::

        ||out||^2 = ||Qx||^2 + W + X
            W = ||h_post||^2 ||o||^2
            X = 2 <Qx, w> = 2 (Q^T h_post)^T d,   d_j = <x_j, o>

    ``X`` is the term that neither an orthogonal ``h_res`` (R3-Cayley) nor a spherical
    ``h_post`` (R5) constrains, and it is the only way a per-token residual norm can
    *shrink* across a module. Measuring it is the whole point of these series: on R5 at
    iter~2250 the attn3->mlp3 module lost 2.08e4 of energy, which is impossible without it.

    ``d`` is the only ``C``-length work (``n`` inner products); the rest is ``[T, n]``
    algebra on the already-fp32 mappings. No ``n*C x n*C`` operator is ever formed.

    Two documented approximations:

    * ``R = ||x||^2`` stands in for ``||Qx||^2``. They are equal iff ``Q`` is orthogonal
      (exact under R3-Cayley to fp32 rounding; off by the Sinkhorn baseline's
      non-orthogonality otherwise). Computing ``||Qx||^2`` would need the ``[T, n, n]``
      stream Gram, 4x the flops of ``d``.
    * ``d`` is a bf16 ``bmm`` (~0.4% relative), to avoid a second fp32 copy of the
      residual inside the forward.

    Args:
        residual: pre-update residual ``x``, ``[s, b, n*C]`` (native contiguous layout,
            so ``reshape(T, n, C)`` is a zero-copy view).
        output: post-update residual ``out``, same shape.
        sublayer_out: sublayer output ``o``, ``[s, b, C]``, BEFORE dropout — with
            ``hidden_dropout > 0`` while training, ``resid_gain``'s bookkeeping check is
            biased by exactly the dropped mass.
        h_post: ``[s, b, n]`` write gate (fp32 from ``compute_mappings``).
        h_res: ``[s, b, n, n]`` residual mixing (fp32).
        n: ``num_residual_streams``, read from the module.
        rel_floor: floor on the write energy, RELATIVE to ``R``. A token whose write
            energy is negligible has no meaningful ``X/W`` or ``cos``; an absolute eps
            would let such tokens contribute ~1e17 and destroy the means.

    Returns 7 0-dim GPU tensors, no host sync, in the order the monitor declares them:
    ``(write_over_resid, cross_over_resid, cross_over_write, mix_write_cos,
    mix_write_cos_abs_max, resid_write_cos, resid_gain)``.
    """
    c = residual.shape[-1] // n
    xs = residual.reshape(-1, n, c)  # [T, n, C], view
    t = xs.shape[0]
    d = torch.bmm(xs, sublayer_out.reshape(t, c, 1)).reshape(t, n).float()  # d_j = <x_j, o>

    r = torch.linalg.vector_norm(residual, dim=-1, dtype=torch.float32).reshape(t).pow(2)
    out_e = torch.linalg.vector_norm(output, dim=-1, dtype=torch.float32).reshape(t).pow(2)
    o_e = torch.linalg.vector_norm(sublayer_out, dim=-1, dtype=torch.float32).reshape(t).pow(2)

    hp = h_post.reshape(t, n).float()
    w_e = hp.pow(2).sum(dim=-1) * o_e  # W
    # (Q^T h_post)_j = sum_i h_res[i, j] h_post_i
    qth = (h_res.reshape(t, n, n) * hp.unsqueeze(-1)).sum(dim=-2)
    cross = 2.0 * (qth * d).sum(dim=-1)  # X = 2<Qx, w>
    pre_cross = (hp * d).sum(dim=-1)  # <x, w>, pre-mix

    rc = r.clamp_min(torch.finfo(torch.float32).tiny)
    wc = w_e.clamp_min(rel_floor * rc)
    cos = cross / (2.0 * (rc * wc).sqrt())
    return (
        (w_e / rc).mean(),
        (cross / rc).mean(),
        (cross / wc).mean(),
        cos.mean(),
        cos.abs().amax(),
        (pre_cross / (rc * wc).sqrt()).mean(),
        (out_e / rc).mean(),
    )
