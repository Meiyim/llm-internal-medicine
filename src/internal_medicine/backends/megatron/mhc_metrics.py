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

All functions return 0-dim GPU tensors and never sync the host (no ``.item()`` /
``.cpu()``), so they are safe to call from a forward hot path. See
``.claude/skills/monitor-hook-perf-rules``.
"""

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
