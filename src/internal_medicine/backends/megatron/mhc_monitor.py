"""mHC Health Monitor for Megatron-Bridge.

Monitors the three per-token mappings of every ``HyperConnectionModule`` in an
mHC (Manifold-Constrained Hyper-Connections) model:

- ``h_pre`` / ``h_post`` — the aggregation / expansion gates: mean and std.
- ``h_res``              — the residual-mixing matrix: the paper's ``amax_gain``
  forward (max-abs row sum) and backward (max-abs column sum), computed both on
  this layer's own ``h_res`` and on two **composite mappings** over the layers local
  to this pipeline stage / VPP chunk — the ascending prefix product for ``fwd`` and
  the descending suffix product for ``bwd``, so each direction reads the chain it
  actually propagates through (see ``_flush_composites``); its deviation from
  orthogonality (token mean, plus a max/median ratio
  over tokens that catches a non-orthogonal tail the mean hides); its trace
  deficit ``n - tr(H_res)`` (= the rank-1 ablation's erase strength ``beta``);
  its distance from the read-write outer product ``h_post (x) h_pre``; and
  ``Sigma``, its singular values on ``1^perp`` (min and mean) — the learnable part
  of any mean-preserving mix, and the criterion for the R8 spectral-sphere runs. At
  ``n = 4`` also its two ``SO(4)`` rotation angles, the pair ``beta = n - tr`` cannot
  separate (it reads only their cosine sum).
- the ``n`` residual **streams** themselves, from the module's own input: per-stream
  L2 norm, the per-token max/min norm ratio, the inter-stream cosine off-diagonal of the
  stream Gram (SIGNED mean + per-token ``|cos|`` max), the cross-stream coefficient
  of variation ``stream_cv``, and the participation-ratio effective rank ``stream_eff_rank``
  (how many of the ``n`` stream directions actually carry energy). Everything above
  reduces over the stream axis, so a dominant stream or a collapse of all ``n``
  onto one direction is invisible in it; these series are what see that.
- the **energy split** of the update ``out = h_res x + w`` (``w = h_post o^T``):
  ``||out||^2 = ||h_res x||^2 + W + X`` with ``W`` the write energy and ``X = 2<h_res x, w>``
  the cross term. ``X`` is unconstrained by both an orthogonal ``h_res`` (R3) and a
  spherical ``h_post`` (R5), and is the only way a residual norm can shrink across a
  module. See ``conf/mai_ladder/mhc/R7_NORM_CONTROL.md`` in the training repo.

Per hyper-connection module (a layer has two: ``attn`` and ``mlp``) we emit
``35 + n`` series (``37 + n`` at ``n = 4``), name-prefixed by component — mean-aggregated
except the three orth_dev / cos ratios, the norm ratio and the Gram max, which are
max-aggregated:

    {attn,mlp}_h_pre_mean   {attn,mlp}_h_pre_std
    {attn,mlp}_h_post_mean  {attn,mlp}_h_post_std
    {attn,mlp}_amax_gain_fwd            {attn,mlp}_amax_gain_bwd
    {attn,mlp}_composite_amax_gain_fwd  {attn,mlp}_composite_amax_gain_bwd
    {attn,mlp}_h_res_orth_dev
    {attn,mlp}_composite_h_res_orth_dev_fwd  {attn,mlp}_composite_h_res_orth_dev_bwd
    {attn,mlp}_h_res_orth_dev_max_med_ratio
    {attn,mlp}_composite_h_res_orth_dev_fwd_max_med_ratio
    {attn,mlp}_composite_h_res_orth_dev_bwd_max_med_ratio
    {attn,mlp}_h_res_beta_mean          {attn,mlp}_h_res_beta_std
    {attn,mlp}_h_res_outer_dev
    {attn,mlp}_h_res_sigma_min          {attn,mlp}_h_res_sigma_mean
    {attn,mlp}_composite_h_res_sigma_min_fwd   {attn,mlp}_composite_h_res_sigma_mean_fwd
    {attn,mlp}_composite_h_res_sigma_min_bwd   {attn,mlp}_composite_h_res_sigma_mean_bwd
    {attn,mlp}_h_res_theta_lo           {attn,mlp}_h_res_theta_hi      (n = 4 only)
    {attn,mlp}_stream_norm_0 .. _{n-1}  {attn,mlp}_stream_norm_max_min_ratio
    {attn,mlp}_stream_gram_offdiag_mean {attn,mlp}_stream_gram_offdiag_max
    {attn,mlp}_stream_cv                {attn,mlp}_stream_eff_rank
    {attn,mlp}_write_over_resid         {attn,mlp}_cross_over_resid
    {attn,mlp}_cross_over_write         {attn,mlp}_mix_write_cos
    {attn,mlp}_mix_write_cos_abs_max    {attn,mlp}_resid_write_cos
    {attn,mlp}_resid_gain

``h_pre`` is not part of ``HyperConnectionModule.forward``'s return, so we cannot
use a forward hook. Instead we wrap the module's ``compute_mappings`` bound method
to capture its real ``(h_pre, h_post, h_res)`` — no recompute. The wrapper's own
argument is the pre-aggregation ``[s, b, n*C]`` hidden state, which is where the
stream-geometry series come from. The energy split needs ``o`` and the final output as
well, which only ``fused_h_res_h_post_bda`` holds, so that bound method is wrapped too.
Everything is detached and computed under ``no_grad`` (see the VRAM-safety notes on
the hook).

The monitor is a hard no-op unless the model actually uses the mHC layer: if the
mHC classes cannot be imported, or no ``HyperConnectionTransformerLayer`` is
found, ``setup_mhc_monitor`` attaches nothing and registers no metrics.

Hot-path discipline (no D2H sync, no hook-time collectives, schema fixed at
registration): see ``.claude/skills/monitor-hook-perf-rules``.
"""

import logging

import torch
import torch.nn as nn

from .base import TorchProbe
from .mhc_metrics import (
    amax_gain,
    erase_beta_stats,
    gate_stats,
    orthogonality_deviation,
    outer_deviation,
    residual_energy_split,
    sigma_stats,
    so4_angle_stats,
    stream_gram_stats,
)

logger = logging.getLogger(__name__)


# mHC classes are optional: this monitor must be a no-op when they are absent
# (non-mHC model, or a Megatron build without hyper-connections). Bind to None
# on import failure and gate every code path on that.
try:
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer
except Exception:  # pragma: no cover - environment dependent
    HyperConnectionModule = None
    HyperConnectionTransformerLayer = None


_METRIC_NAMES = (
    "h_pre_mean",
    "h_pre_std",
    "h_post_mean",
    "h_post_std",
    "amax_gain_fwd",
    "amax_gain_bwd",
    "composite_amax_gain_fwd",
    "composite_amax_gain_bwd",
    "h_res_orth_dev",
    "composite_h_res_orth_dev_fwd",
    "composite_h_res_orth_dev_bwd",
    "h_res_orth_dev_max_med_ratio",
    "composite_h_res_orth_dev_fwd_max_med_ratio",
    "composite_h_res_orth_dev_bwd_max_med_ratio",
    "h_res_beta_mean",
    "h_res_beta_std",
    "h_res_outer_dev",
    "h_res_sigma_min",
    "h_res_sigma_mean",
    "composite_h_res_sigma_min_fwd",
    "composite_h_res_sigma_mean_fwd",
    "composite_h_res_sigma_min_bwd",
    "composite_h_res_sigma_mean_bwd",
)

# Geometry of the n streams the mappings act on, from the module's own input. Fixed part;
# the n per-stream norm series are added per module by _stream_metric_names (n is a config
# constant, so the schema is still fully determined before allocate_buffers locks it).
_STREAM_METRIC_NAMES = (
    "stream_norm_max_min_ratio",
    "stream_gram_offdiag_mean",
    "stream_gram_offdiag_max",
    "stream_cv",
    "stream_eff_rank",
)


def _stream_metric_names(n: int) -> tuple[str, ...]:
    return tuple(f"stream_norm_{i}" for i in range(n)) + _STREAM_METRIC_NAMES


# The two rotation angles of an SO(4) h_res. Only declared at n = 4 (the formula's constants are
# n-specific); n is a config constant, so the schema is still fixed before allocate_buffers.
_SO4_METRIC_NAMES = ("h_res_theta_lo", "h_res_theta_hi")


def _so4_metric_names(n: int) -> tuple[str, ...]:
    return _SO4_METRIC_NAMES if n == 4 else ()


# Energy split of one update, from ``fused_h_res_h_post_bda``. All ratios are per-token,
# then meaned. R = ||x||^2, W = write energy, X = 2<h_res x, w>, Out = ||out||^2.
#
# X is computed from the module's INPUTS, so under R7c (cross-free write) these series
# still report the PRE-correction cross term — i.e. the alignment the model is asking
# for, not a monitor bug. What R7c changes is ``resid_gain``, which must move from
# 1 + W/R + X/R to 1 + (W/R) sin^2(theta).
_ENERGY_METRIC_NAMES = (
    "write_over_resid",  # W/R
    "cross_over_resid",  # X/R                            signed
    "cross_over_write",  # X/W'    W' = W floored at 1e-6 R
    "mix_write_cos",  # cos(theta) = X / (2 sqrt(RW'))    scale-free, comparable across layers
    "mix_write_cos_abs_max",  # max |cos(theta)|          tail detector
    "resid_write_cos",  # <x, w> / (||x|| ||w||)          pre-mix; what an anti-Hermitian write zeroes
    "resid_gain",  # Out/R                                growth law + bookkeeping self-check
)


# (component_name, layer attribute) — attn runs before mlp in the layer forward.
_COMPONENTS = (
    ("attn", "self_attention_hyper_connection"),
    ("mlp", "mlp_hyper_connection"),
)

# Execution rank within a layer, for the static ordering the composite chains walk.
_COMPONENT_RANK = {"attn": 0, "mlp": 1}

# The max/median and max/min ratios compose across microbatches / ranks with max, not mean:
# averaging a tail detector hides the tail it exists to catch.
_MAX_METRIC_NAMES = (
    "h_res_orth_dev_max_med_ratio",
    "composite_h_res_orth_dev_fwd_max_med_ratio",
    "composite_h_res_orth_dev_bwd_max_med_ratio",
    "stream_norm_max_min_ratio",
    "stream_gram_offdiag_max",
    "mix_write_cos_abs_max",
)
_MAX_AGGREGATED = {f"{comp}_{name}" for comp, _ in _COMPONENTS for name in _MAX_METRIC_NAMES}


class MHCHealthMonitor(TorchProbe):
    """Monitor the h_pre / h_post / h_res mappings of mHC hyper-connection layers."""

    METRIC_PREFIX = "mhc_health"
    # All series are means over tokens/batch (and over microbatches/ranks at flush)
    # except the ratios and the Gram off-diagonal max, which are max-aggregated. Those
    # names are also listed in training_logs.MAX_AGGREGATED_SUFFIXES so the suffix
    # classifier composes them across ranks with max instead of mean.
    MAX_AGGREGATED: set[str] = _MAX_AGGREGATED
    MIN_AGGREGATED: set[str] = set()

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        hook_timing_enabled: bool = False,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            hook_timing_enabled=hook_timing_enabled,
        )
        # chunk_id -> {(layer_idx, component): detached h_res [s, b, n, n]} for the
        # microbatch currently in flight. Drained by _flush_composites() as soon as the
        # chunk's full module set has fired, so the references live only inside one
        # forward — where the autograd graph holds those storages anyway, making the
        # residency cost ~free. The composite chains then walk the stash in STATIC
        # layer order, so neither chain depends on wrapper firing order.
        self._h_res: dict[int, dict[tuple[int, str], torch.Tensor]] = {}
        # chunk_id -> number of hc modules discovered, i.e. a complete stash.
        self._expected: dict[int, int] = {}
        # (module, attr_name, original_bound_method) triples, for remove_hooks() restoration.
        self._wrapped: list[tuple[nn.Module, str, object]] = []

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _find_layer_stack(self, model: nn.Module):
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return model.decoder.layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers
        if hasattr(model, "layers"):
            return model.layers
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "decoder") and hasattr(lm.decoder, "layers"):
                return lm.decoder.layers
        return None

    def _find_hc_modules(self, model: nn.Module, layer_offset: int = 0):
        """Return ``[(global_idx, component, hc_module)]`` for every mHC hc module.

        Empty (auto-skip) when the mHC classes are unavailable or the model has no
        ``HyperConnectionTransformerLayer``. Uses ``isinstance`` against the real
        classes rather than duck-typing, so a plain ``TransformerLayer`` (or an
        ``IdentityOp`` placeholder) is never matched.
        """
        if HyperConnectionTransformerLayer is None or HyperConnectionModule is None:
            return []
        layers = self._find_layer_stack(model)
        if layers is None:
            return []

        entries = []
        num_local = len(layers)
        for local_idx, layer in enumerate(layers):
            if not isinstance(layer, HyperConnectionTransformerLayer):
                continue
            global_idx = self._resolve_layer_idx(layer, local_idx, num_local, layer_offset)
            for comp, attr in _COMPONENTS:
                mod = getattr(layer, attr, None)
                if isinstance(mod, HyperConnectionModule):
                    entries.append((global_idx, comp, mod))
        return entries

    # ------------------------------------------------------------------
    # Three-phase setup: prepare (declare) -> allocate -> attach
    # ------------------------------------------------------------------

    def _prepare_layers(self, model: nn.Module, chunk_id: int, layer_offset: int = 0):
        entries = self._find_hc_modules(model, layer_offset=layer_offset)
        if not entries:
            return []
        for global_idx, comp, mod in entries:
            n = int(mod.n)
            for name in _METRIC_NAMES + _stream_metric_names(n) + _so4_metric_names(n) + _ENERGY_METRIC_NAMES:
                self.declare_layer_metric(global_idx, f"{comp}_{name}")
        # A stash of this size is a complete chunk, which is what triggers the
        # composite pass (see _make_capture). Accumulated across calls because VPP
        # discovery may add to the same chunk_id in several passes.
        self._expected[chunk_id] = self._expected.get(chunk_id, 0) + len(entries)
        return [(gi, comp, mod, chunk_id) for gi, comp, mod in entries]

    def _attach_hooks(self, targets):
        for layer_idx, comp, mod, chunk_id in targets:
            n = int(mod.n)
            # both originals are read before any setattr, so neither wrapper sees the other
            wrappers = {
                "compute_mappings": self._make_capture(mod.compute_mappings, layer_idx, comp, chunk_id, n),
                "fused_h_res_h_post_bda": self._make_bda_capture(mod.fused_h_res_h_post_bda, layer_idx, comp, n),
            }
            for attr, wrapper in wrappers.items():
                self._wrapped.append((mod, attr, getattr(mod, attr)))
                setattr(mod, attr, wrapper)
        logger.info(f"[MHCMonitor] Wrapped {len(self._wrapped)} bound methods on hyper-connection modules.")

    def register_hooks(self, model: nn.Module):
        """Single-chunk convenience path. Prefer ``setup_mhc_monitor`` for VPP."""
        self._init_parallel_state()
        targets = self._prepare_layers(model, chunk_id=0)
        if not targets:
            return
        self.allocate_buffers(next(model.parameters()).device)
        self._attach_hooks(targets)

    def _init_parallel_state(self):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        except ImportError:
            pass

    def remove_hooks(self):
        # Restore the original bound methods and drop all cross-call state so the
        # monitor holds no module references or h_res tensors after teardown.
        for mod, attr, orig in self._wrapped:
            try:
                delattr(mod, attr)  # fall back to the class method
            except AttributeError:
                setattr(mod, attr, orig)
        self._wrapped = []
        self._h_res.clear()
        self._expected.clear()
        super().remove_hooks()

    def step(self, global_step: int | None = None):
        # Drain any stash a partial forward left behind, so those composite values land
        # in this step's flush. A complete chunk already drained inside the forward.
        self._flush_composites()
        super().step(global_step=global_step)

    # ------------------------------------------------------------------
    # Deferred composite chains
    # ------------------------------------------------------------------

    def _flush_composites(self, chunk_id: int | None = None) -> None:
        """Build both composite chains from the stashed ``h_res``, in static layer order.

        For hc modules ``0..N`` in execution order (each applying ``r <- H_k r + w_k``):

            prefix   M_l = H_l @ ... @ H_0      d(output of l) / d(stage input)
            suffix   S_l = H_N @ ... @ H_l      d(stage output) / d(input of l)

        The gradient reaching module ``l``'s input is ``S_l^T g``, so the suffix chain is
        what answers "how much is a gradient amplified on its way from the loss down to
        layer l" — the prefix cannot, it looks the other way. ``amax_gain(S, dim=-2)``
        (max-abs column sum) is the row sum of ``S^T``, i.e. the backward bound.

        Deferred because ``S_l`` needs the layers above ``l``, which have not run when
        ``l``'s hook fires. Walking a static order also makes both chains independent of
        wrapper firing order, which matters: under ``recompute_granularity=full`` the
        only grad-enabled pass is the BACKWARD replay, and it fires modules in
        DESCENDING order.

        ``chunk_id=None`` drains every chunk (called from ``step()`` to catch a stash
        left partial by an incomplete forward).

        Cost: two bmm chains over ``[s*b, n, n]``, one live intermediate each.
        """
        targets = list(self._h_res) if chunk_id is None else [chunk_id]
        for cid in targets:
            stash = self._h_res.pop(cid, None)
            if not stash:
                continue
            # Static execution order: layer index, then attn before mlp within a layer.
            keys = sorted(stash, key=lambda k: (k[0], _COMPONENT_RANK.get(k[1], 0)))
            try:
                mats = [stash[k].reshape(-1, stash[k].shape[-1], stash[k].shape[-1]) for k in keys]
                shapes = {tuple(m.shape) for m in mats}
                if len(shapes) > 1:
                    # Variable s*b across modules (should not happen within one forward);
                    # a chain would be ill-defined, so skip rather than report garbage.
                    if self.verbose:
                        logger.warning(f"[MHCMonitor] chunk {cid}: mixed h_res shapes {shapes}; skipping composites")
                    continue
                self._record_chain(keys, mats, "fwd")
                self._record_chain(keys, mats, "bwd")
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MHCMonitor] Composite error on chunk {cid}: {e}")

    def _record_chain(self, keys, mats, direction: str) -> None:
        """Accumulate one chain and record its metrics.

        ``fwd``: ascending, ``M_l = H_l @ M_{l-1}`` — new factor on the LEFT.
        ``bwd``: descending, ``S_l = S_{l+1} @ H_l`` — new factor on the RIGHT.
        """
        ascending = direction == "fwd"
        order = range(len(mats)) if ascending else range(len(mats) - 1, -1, -1)
        gain_dim = -1 if ascending else -2

        acc = None
        with torch.no_grad():
            for i in order:
                h = mats[i]
                if acc is None:
                    acc = h.clone()  # noqa: SIM108 — own the buffer; never alias the model's h_res
                elif ascending:
                    acc = torch.bmm(h, acc)
                else:
                    acc = torch.bmm(acc, h)

                layer_idx, component = keys[i]
                self.record_layer_metric(
                    layer_idx, f"{component}_composite_amax_gain_{direction}", amax_gain(acc, dim=gain_dim)
                )
                orth_mean, orth_ratio = orthogonality_deviation(acc.unsqueeze(0))
                self.record_layer_metric(layer_idx, f"{component}_composite_h_res_orth_dev_{direction}", orth_mean)
                self.record_layer_metric(
                    layer_idx, f"{component}_composite_h_res_orth_dev_{direction}_max_med_ratio", orth_ratio
                )
                # Sigma on 1^perp of the COMPOSITE: the headline orthogonal-vs-DS curve. A
                # doubly-stochastic stack annihilates 1^perp with depth (sigma_min -> 0, measured
                # 0.14 -> 8e-4 -> 0 over 12 layers), a mean-fixing orthogonal one is an isometry
                # at any depth (sigma_min == 1). Pair with stream_cv, which sees the consequence.
                sig_min, sig_mean = sigma_stats(acc.unsqueeze(0))
                self.record_layer_metric(layer_idx, f"{component}_composite_h_res_sigma_min_{direction}", sig_min)
                self.record_layer_metric(layer_idx, f"{component}_composite_h_res_sigma_mean_{direction}", sig_mean)

    # ------------------------------------------------------------------
    # Capture wrapper (the hot path)
    # ------------------------------------------------------------------

    def _make_capture(self, orig, layer_idx: int, component: str, chunk_id: int, n_streams: int):
        """Wrap ``compute_mappings`` to record metrics from its real return value.

        VRAM safety: the mappings arrive attached to the training autograd graph;
        we ``.detach()`` them and do all metric math under ``no_grad`` so no stored
        tensor pins the graph through backward. ``h_res`` is stashed by (layer,
        component) for the deferred composite pass; the stash holds the detached view,
        which shares storage the model keeps alive until backward anyway.
        """

        def wrapped(x):
            out = orig(x)  # the real mappings the model consumes — returned unchanged
            # Gate the whole capture: metrics are only recorded on monitored steps.
            # _should_monitor() also requires grad enabled — under
            # recompute_granularity=full that is the BACKWARD replay, not the forward,
            # which is why the composite chains must not depend on firing order.
            if not self._should_monitor():
                return out
            try:
                h_pre, h_post, h_res = out
                with torch.no_grad():
                    h_pre = h_pre.detach()
                    h_post = h_post.detach()
                    h_res = h_res.detach()

                    # Stream geometry of this module's own input, before any mixing.
                    s_norms, norm_ratio, gram_mean, gram_max, stream_cv, eff_rank = stream_gram_stats(
                        x.detach(), n_streams
                    )
                    for i in range(n_streams):
                        self.record_layer_metric(layer_idx, f"{component}_stream_norm_{i}", s_norms[i])
                    self.record_layer_metric(layer_idx, f"{component}_stream_norm_max_min_ratio", norm_ratio)
                    self.record_layer_metric(layer_idx, f"{component}_stream_gram_offdiag_mean", gram_mean)
                    self.record_layer_metric(layer_idx, f"{component}_stream_gram_offdiag_max", gram_max)
                    self.record_layer_metric(layer_idx, f"{component}_stream_cv", stream_cv)
                    self.record_layer_metric(layer_idx, f"{component}_stream_eff_rank", eff_rank)

                    pre_mean, pre_std = gate_stats(h_pre)
                    post_mean, post_std = gate_stats(h_post)
                    self.record_layer_metric(layer_idx, f"{component}_h_pre_mean", pre_mean)
                    self.record_layer_metric(layer_idx, f"{component}_h_pre_std", pre_std)
                    self.record_layer_metric(layer_idx, f"{component}_h_post_mean", post_mean)
                    self.record_layer_metric(layer_idx, f"{component}_h_post_std", post_std)

                    self.record_layer_metric(layer_idx, f"{component}_amax_gain_fwd", amax_gain(h_res, dim=-1))
                    self.record_layer_metric(layer_idx, f"{component}_amax_gain_bwd", amax_gain(h_res, dim=-2))
                    orth_mean, orth_ratio = orthogonality_deviation(h_res)
                    self.record_layer_metric(layer_idx, f"{component}_h_res_orth_dev", orth_mean)
                    self.record_layer_metric(layer_idx, f"{component}_h_res_orth_dev_max_med_ratio", orth_ratio)

                    # beta = n - tr(H_res): the rank-1 erase strength (exact for
                    # H_res = I - beta*u u^T), else the trace deficit.
                    beta_mean, beta_std = erase_beta_stats(h_res)
                    self.record_layer_metric(layer_idx, f"{component}_h_res_beta_mean", beta_mean)
                    self.record_layer_metric(layer_idx, f"{component}_h_res_beta_std", beta_std)
                    self.record_layer_metric(
                        layer_idx, f"{component}_h_res_outer_dev", outer_deviation(h_res, h_pre, h_post)
                    )

                    # Sigma on 1^perp: the learnable part of any mean-preserving mix. R8's criterion.
                    sig_min, sig_mean = sigma_stats(h_res)
                    self.record_layer_metric(layer_idx, f"{component}_h_res_sigma_min", sig_min)
                    self.record_layer_metric(layer_idx, f"{component}_h_res_sigma_mean", sig_mean)

                    # The SO(4) conjugacy class needs BOTH angles; beta reads only their cosine sum.
                    if n_streams == 4:
                        theta_lo, theta_hi = so4_angle_stats(h_res)
                        self.record_layer_metric(layer_idx, f"{component}_h_res_theta_lo", theta_lo)
                        self.record_layer_metric(layer_idx, f"{component}_h_res_theta_hi", theta_hi)

                    # Composite mappings need the layers ABOVE this one, which have not
                    # run yet. Stash h_res; drain as soon as this chunk's full module set
                    # has fired, so the stash never spans microbatches (with GA that would
                    # pin ga_steps x modules tensors instead of one microbatch's worth).
                    stash = self._h_res.setdefault(chunk_id, {})
                    stash[(layer_idx, component)] = h_res
                    if len(stash) >= self._expected.get(chunk_id, 0):
                        self._flush_composites(chunk_id)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MHCMonitor] Error layer {layer_idx}/{component}: {e}")
            return out

        return wrapped

    def _make_bda_capture(self, orig, layer_idx: int, component: str, n_streams: int):
        """Wrap ``fused_h_res_h_post_bda`` for the energy split of the update it performs.

        This is the only method holding ``h_res``, the pre-update residual, ``h_post``, the
        sublayer output ``o`` and the final output at once. The signature is replicated from
        upstream ``HyperConnectionModule`` (all call sites pass positionally); the return
        value is passed through untouched.

        ``original_residual`` is bit-identically the ``x`` that ``compute_mappings`` saw —
        ``transformer_layer`` captures ``residual = hidden_states`` before aggregation.

        ``o`` is taken before dropout, so with ``hidden_dropout > 0`` while training the
        ``resid_gain`` bookkeeping check is biased by the dropped mass (every current mHC
        recipe runs ``hidden_dropout = 0.0``).
        """

        def wrapped(
            h_res,
            original_residual,
            h_post,
            layer_output_with_bias,
            dropout_prob,
            training,
            fused,
            manager=None,
        ):
            out = orig(h_res, original_residual, h_post, layer_output_with_bias, dropout_prob, training, fused, manager)
            if not self._should_monitor():
                return out
            try:
                with torch.no_grad():
                    o, bias = layer_output_with_bias
                    o = o.detach()
                    # the bda adds bias into the write, so it belongs to the write energy
                    if bias is not None:
                        o = o + bias.detach()
                    vals = residual_energy_split(
                        original_residual.detach(),
                        out.detach(),
                        o,
                        h_post.detach(),
                        h_res.detach(),
                        n_streams,
                    )
                    for name, val in zip(_ENERGY_METRIC_NAMES, vals, strict=True):
                        self.record_layer_metric(layer_idx, f"{component}_{name}", val)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MHCMonitor] Energy-split error layer {layer_idx}/{component}: {e}")
            return out

        return wrapped


def setup_mhc_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    hook_timing_enabled: bool = False,
    monitor_dict: dict | None = None,
):
    """Enable the mHC health monitor. No-op on any non-mHC model.

    Multi-chunk (VPP / interleaved 1F1B) safe: declares the schema across all
    chunks before ``allocate_buffers`` locks it, then attaches. Each model chunk
    gets its own composite slot (keyed by its enumerate index) so a later chunk's
    layers never contaminate an earlier chunk's running product.
    """
    # No-op guarantee #1: mHC classes unavailable -> touch nothing.
    if HyperConnectionTransformerLayer is None or HyperConnectionModule is None:
        logger.info("[MHCMonitor] Hyper-connection classes unavailable; skipping.")
        return model

    monitor = MHCHealthMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        hook_timing_enabled=hook_timing_enabled,
    )
    models = [model] if not isinstance(model, list) else model
    monitor._init_parallel_state()

    chunk_targets = []
    layer_offset = 0
    for chunk_id, m in enumerate(models):
        targets = monitor._prepare_layers(m, chunk_id=chunk_id, layer_offset=layer_offset)
        chunk_targets.append(targets)
        stack = monitor._find_layer_stack(m)
        layer_offset += len(stack) if stack is not None else 0

    if any(chunk_targets):
        device = next((p.device for m in models for p in m.parameters()), None)
        assert device is not None, "no parameters across model chunks; cannot pick a device"
        monitor.allocate_buffers(device)
        for targets in chunk_targets:
            monitor._attach_hooks(targets)
        if monitor_dict is not None:
            monitor_dict["mhc_health"] = monitor
    else:
        # No-op guarantee #2: no HyperConnectionTransformerLayer found.
        logger.info("[MHCMonitor] No hyper-connection layers found; skipping.")

    logger.info(f"[MHCMonitor] Setup complete. Monitoring {len(monitor._wrapped)} hc modules.")
    return model
