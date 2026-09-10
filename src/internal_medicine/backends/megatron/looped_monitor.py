# -*- coding: utf-8 -*-
"""Recurrent-depth (looped transformer) health monitor — arXiv 2502.05171.

What it watches
---------------
A looped model runs its recurrent core ``R`` ``r`` times per forward, producing a
sequence of states ``s_1 .. s_r`` where ``s_i = R(A(s_{i-1}, e))`` (there is no
core-exit norm between iterations — the old ``state_norm`` was removed). ``s_i`` is
read as the **last core layer's forward output** (``block.layers[core_end - 1]``),
which runs exactly once per iteration and nowhere else.

The interesting signal is a *curve along the recurrence axis*, not along physical
layers, so this monitor reuses the base "layer index" slot to mean the **recurrence
iteration index** ``i`` (0-based). Every recurrence-axis metric key is therefore
``looped_health/layer_{i}/{metric}`` and encodes which iteration it came from.

That encoding is mandatory, not cosmetic: ``base_monitor._resolve_layer_idx``
assumes one observation per module per forward, so without a distinct key per
iteration the r firings of the single reused core layer / ``adapter`` module would
collapse into one averaged slot. ``r`` is fixed (no random sampling in v1), so the
full schema is known at registration time and declared up front, as the GPU-buffer
API requires.

Recurrence-axis metrics (per iteration ``i``)
---------------------------------------------
``token_cosine``   mean pairwise cosine similarity of the state's token
                   representations. ->1.0 is representation collapse — the paper's
                   Bad Run 1 failure mode (Fig 5).
``eff_rank``       participation-ratio (Renyi-2) effective rank of the state's
                   token cloud, ``(tr C)^2 / ||C||_F^2`` on the ``H x H`` second
                   moment ``C = X^T X``. Same primitive shape as
                   ``mhc_metrics.stream_gram_stats``; drops if the recurrence is
                   crushing the representation onto a low-dim subspace.
``state_delta``    RMS of ``s_i - s_{i-1}`` normalized by RMS of ``s_i`` (paper
                   Fig 11): does the recurrence converge to a fixed point or keep
                   moving? Only defined for ``i >= 1``.
``adapter_e_ratio``  ``||A_e e|| / ||A_s s||`` from the adapter's per-branch RMS
                   byproducts (concat injection only). ->0 means the model has
                   learned to ignore the injected input ``e``; -> large means it
                   ignores the recurrent state ``s`` (the paper's Bad Run 2).

Global weight-space metrics (concat injection only)
---------------------------------------------------
``global_spec_norm_A_s`` / ``global_spec_norm_A_e``   spectral norm ``sigma_max`` of
                   the adapter's ``linear_state`` (``A_s``) and ``linear_embed``
                   (``A_e``) weight matrices, via warm-started power iteration. Cast
                   the recurrence as a linear control system
                   ``h_{t+1} = A_s h_t + A_e e + F(...)``: ``sigma_max(A_s)`` is the
                   stability criterion (identity-init starts at 1.0; drift above 1 =
                   an expansive, potentially unstable recurrence). A weight-space
                   statistic, computed once per monitored step, max-aggregated over
                   the flush window (peak sigma is the stability-relevant value).
                   Caveats: only meaningful for ``step_injection == "e"`` +
                   ``input_injection == "concat"``; the clean ``A_s == A_bar``
                   identity assumes the residual skip is unbroken (looser under
                   ``looped_sandwich_norm``); ``nn.Linear`` weights are replicated
                   across TP, so per-rank sigma is identical (no reduction needed).

Hot-path discipline
-------------------
Every metric is computed from GPU tensors and recorded as 0-dim GPU tensors; no
``.item()`` / ``.cpu()`` / collective fires in a hook (power iteration is pure
on-GPU matmuls). See ``.claude/skills/monitor-hook-perf-rules``.

Ordering
--------
v1 forbids activation recompute (rejected in the provider's finalize()) and uses
``k == r`` full backprop, so the last core layer / ``adapter`` fire exactly ``r``
times per forward, in ascending iteration order, all with grad enabled. A simple
per-module ``count % r`` counter therefore recovers the iteration index and
self-resets at each forward boundary — no need for the stash/static-sort dance
``mhc_monitor`` needs for descending recompute-replay firing.
"""

import logging

import torch
import torch.nn as nn

from .base import TorchProbe
from .massive_activation_metrics import compute_post_norm_cosine_stability

logger = logging.getLogger(__name__)


def _covariance_effective_rank(state: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Participation-ratio effective rank of the token cloud, as a 0-dim tensor.

    ``state`` is ``[S, B, H]`` (or ``[B, S, H]``). Flatten tokens to ``X`` of
    ``[N, H]`` and form the ``H x H`` uncentered second moment ``C = X^T X``; its
    eigenvalues are the token-cloud spectrum. The Renyi-2 effective rank
    ``(sum lambda)^2 / sum lambda^2 = (tr C)^2 / ||C||_F^2`` needs no eigendecomp:
    ``tr C = sum_i ||col_i||^2`` and ``||C||_F^2 = sum_ij C_ij^2``. ``H`` is small
    (1024 for l12), so the ``[H, H]`` matmul and its square are cheap and only run
    on monitored steps. ``eps`` on both sides makes an all-zero state read 1.0
    rather than NaN. Matches ``mhc_metrics.py`` line ~317.
    """
    h = state.reshape(-1, state.shape[-1]).float()
    cov = h.transpose(0, 1) @ h  # [H, H]
    tr = cov.diagonal().sum()
    fro2 = cov.pow(2).sum()
    return (tr * tr + eps) / (fro2 + eps)


def _power_iteration_spec_norm(weight: torch.Tensor, v: torch.Tensor, n_iters: int, eps: float):
    """Estimate ``sigma_max(weight)`` by power iteration, GPU-only (no host sync).

    ``weight`` is ``[out, in]``; ``v`` is a unit vector in input space ``[in]``,
    warm-started across steps so 1-2 iterations suffice. Returns the 0-dim spectral
    norm and the updated ``v`` to persist for the next step. All ops stay on device.
    """
    w = weight.detach().float()
    for _ in range(n_iters):
        u = w @ v
        u = u / (u.norm() + eps)
        v = w.transpose(0, 1) @ u
        v = v / (v.norm() + eps)
    return (w @ v).norm(), v


class LoopedHealthMonitor(TorchProbe):
    """Recurrence-axis health metrics for a LoopedTransformerBlock.

    The base "layer index" is repurposed as the recurrence iteration index ``i``.
    All metrics are mean-aggregated (both class sets empty): the global reduction
    then averages a metric across iterations, and the per-iteration curve lives in
    ``looped_health/layer_{i}/{metric}``.
    """

    METRIC_PREFIX = "looped_health"
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()

    def __init__(
        self,
        cosine_sample_pairs: int = 256,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        hook_timing_enabled: bool = False,
        eps: float = 1e-6,
        spec_norm_power_iters: int = 2,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            hook_timing_enabled=hook_timing_enabled,
        )
        self.cosine_sample_pairs = cosine_sample_pairs
        self.eps = eps
        # Set once the looped block is discovered in _prepare_layers.
        self.r: int | None = None
        self._injection: str | None = None
        # Per-forward iteration counters (count % r == iteration index). Two
        # independent counters because the core-state hook and adapter each fire r times.
        self._state_iter = 0
        self._adapter_iter = 0
        # Previous iteration's state, for ||s_i - s_{i-1}||. Detached, one [S,B,H]
        # tensor held only across a monitored forward.
        self._prev_norm_state: torch.Tensor | None = None
        # Spectral-norm power iteration: warm-started unit vectors per matrix
        # (fp32, on the weight's device) and a once-per-step guard (the adapter
        # hook fires r times but the weight is constant within a forward).
        self._spec_norm_power_iters = spec_norm_power_iters
        self._piter_eps = 1e-12
        self._piter_v: dict[str, torch.Tensor] = {}
        self._spec_done = False

    # ------------------------------------------------------------------
    # registration (declare → allocate → attach)
    # ------------------------------------------------------------------
    def register_hooks(self, model: nn.Module):
        self._init_parallel_state()
        block = self._prepare_layers(model)
        if block is None:
            return
        self.allocate_buffers(next(model.parameters()).device)
        self._attach_hooks(block)

    def _init_parallel_state(self):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        except ImportError:
            pass

    def _find_looped_block(self, model: nn.Module):
        """Return the LoopedTransformerBlock if this chunk is a live looped core.

        A non-looped decoder, or a looped block in its degenerate ``l_R == 0``
        parity configuration (adapter is None there), returns None so the monitor
        is a clean no-op — it can sit in the ``all`` monitor list safely.
        """
        if hasattr(model, "module"):
            model = model.module
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            return None
        adapter = getattr(decoder, "adapter", None)
        r = getattr(decoder, "looped_num_recurrence", None)
        if adapter is None or not isinstance(r, int) or r < 1:
            return None
        return decoder

    def _prepare_layers(self, model: nn.Module):
        """Discover the looped block and declare the full recurrence-axis schema."""
        block = self._find_looped_block(model)
        if block is None:
            return None

        r = block.looped_num_recurrence
        injection = getattr(block.adapter, "injection", None)

        if self.r is not None:
            # PP > 1 is rejected for looped models, so there is exactly one core.
            # A second one would share these counters; refuse rather than mislabel.
            logger.warning("[LoopedMonitor] a looped block was already registered; ignoring a second one")
            return None

        self.r = r
        self._injection = injection

        for i in range(r):
            self.declare_layer_metric(i, "token_cosine")
            self.declare_layer_metric(i, "eff_rank")
        # state_delta compares against the previous iteration, so iteration 0 has
        # nothing to diff — declare it only for i >= 1.
        for i in range(1, r):
            self.declare_layer_metric(i, "state_delta")
        # The e/s branch ratio and the spectral norms only exist for the concat
        # adapter (none has no separable branches / no A_s, A_e matrices).
        if injection == "concat":
            for i in range(r):
                self.declare_layer_metric(i, "adapter_e_ratio")
            # Weight-space spectral norms are single scalars (one adapter), not
            # per-iteration, so declare them as explicit max-aggregated globals.
            self.declare_max(f"{self.METRIC_PREFIX}/global_spec_norm_A_s")
            self.declare_max(f"{self.METRIC_PREFIX}/global_spec_norm_A_e")

        if self.verbose:
            logger.info("[LoopedMonitor] looped block found: r=%d injection=%s", r, injection)
        return block

    def _attach_hooks(self, block):
        # s_i is the last core layer's output (runs once per iteration, only in
        # the loop; coda is layers[core_end:]). The old core-exit state_norm is gone.
        core_state_module = block.layers[block.core_end - 1]
        state_hook = core_state_module.register_forward_hook(
            self.timed_hook("core_state", self._make_state_hook())
        )
        self.hooks.append(state_hook)
        if self._injection == "concat":
            adapter_hook = block.adapter.register_forward_hook(
                self.timed_hook("adapter", self._make_adapter_hook())
            )
            self.hooks.append(adapter_hook)
        logger.info("[LoopedMonitor] Registered %d hooks (r=%d).", len(self.hooks), self.r)

    # ------------------------------------------------------------------
    # hooks (hot path)
    # ------------------------------------------------------------------
    def _make_state_hook(self):
        """Post-hook on the last core layer: token_cosine, eff_rank, state_delta of s_i."""

        def hook_fn(module, args, output):
            if not self._should_monitor():
                return
            state = output[0] if isinstance(output, tuple) else output
            if not isinstance(state, torch.Tensor):
                return
            i = self._state_iter % self.r
            self._state_iter += 1
            with torch.no_grad():
                state = state.detach()
                self.record_layer_metric(i, "token_cosine",
                                         compute_post_norm_cosine_stability(state, self.cosine_sample_pairs))
                self.record_layer_metric(i, "eff_rank", _covariance_effective_rank(state, self.eps))

                state_f = state.float()
                if i >= 1 and self._prev_norm_state is not None and self._prev_norm_state.shape == state_f.shape:
                    num = (state_f - self._prev_norm_state).pow(2).mean().sqrt()
                    den = state_f.pow(2).mean().sqrt().clamp_min(self.eps)
                    self.record_layer_metric(i, "state_delta", num / den)
                # Stash for the next iteration; drop it at the forward boundary so a
                # monitored forward holds at most one [S,B,H] fp32 copy.
                self._prev_norm_state = None if i == self.r - 1 else state_f

        return hook_fn

    def _make_adapter_hook(self):
        """Post-hook on the concat adapter: per-iteration ||A_e e|| / ||A_s s||, plus
        the once-per-step spectral norms of A_s / A_e."""

        def hook_fn(module, args, output):
            if not self._should_monitor():
                return
            with torch.no_grad():
                # Weight-space, constant within a forward: compute once per step.
                if not self._spec_done:
                    self._record_spec_norms(module)
                    self._spec_done = True
            embed_norm = module.last_embed_norm
            state_norm = module.last_state_norm
            i = self._adapter_iter % self.r
            self._adapter_iter += 1
            if embed_norm is None or state_norm is None:
                return
            with torch.no_grad():
                ratio = embed_norm.detach() / state_norm.detach().clamp_min(self.eps)
                self.record_layer_metric(i, "adapter_e_ratio", ratio)

        return hook_fn

    def _record_spec_norms(self, adapter):
        """sigma_max of A_s (linear_state) and A_e (linear_embed) via power iteration."""
        for name, linear in (("A_s", adapter.linear_state), ("A_e", adapter.linear_embed)):
            if linear is None:
                continue
            weight = linear.weight
            v = self._piter_v.get(name)
            in_dim = weight.shape[1]
            if v is None or v.numel() != in_dim:
                v = torch.randn(in_dim, device=weight.device, dtype=torch.float32)
                v = v / (v.norm() + self._piter_eps)
            sigma, v = _power_iteration_spec_norm(weight, v, self._spec_norm_power_iters, self._piter_eps)
            self._piter_v[name] = v
            self.record_max(f"{self.METRIC_PREFIX}/global_spec_norm_{name}", sigma)

    def step(self, global_step: int | None = None):
        # Flush first (base), then drop any state stash so it never survives across
        # steps (e.g. after a forward interrupted before its last iteration).
        super().step(global_step)
        self._prev_norm_state = None
        # Re-arm the once-per-step spectral-norm computation for the next step.
        self._spec_done = False


def setup_looped_monitor(
    model: nn.Module,
    cosine_sample_pairs: int = 256,
    verbose: bool = False,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    hook_timing_enabled: bool = False,
    eps: float = 1e-6,
    spec_norm_power_iters: int = 2,
    monitor_dict: dict | None = None,
) -> nn.Module:
    """VPP-aware factory mirroring setup_qk_monitor.

    Only the chunk holding the looped core declares any schema; all others return
    None from _prepare_layers, so this is a no-op on non-looped models and can live
    in the ``all`` monitor list.
    """
    monitor = LoopedHealthMonitor(
        cosine_sample_pairs=cosine_sample_pairs,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        hook_timing_enabled=hook_timing_enabled,
        eps=eps,
        spec_norm_power_iters=spec_norm_power_iters,
    )
    models = [model] if not isinstance(model, list) else model
    monitor._init_parallel_state()
    chunk_blocks = [(m, monitor._prepare_layers(m)) for m in models]
    if any(block is not None for _, block in chunk_blocks):
        device = next((p.device for m in models for p in m.parameters()), None)
        assert device is not None, "no parameters across model chunks; cannot pick a device"
        monitor.allocate_buffers(device)
        for _, block in chunk_blocks:
            if block is not None:
                monitor._attach_hooks(block)
    logger.info("[LoopedMonitor] Setup complete. Monitoring %d hooks.", len(monitor.hooks))
    if monitor_dict is not None:
        monitor_dict["looped_health"] = monitor
    return model
