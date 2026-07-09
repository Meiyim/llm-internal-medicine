"""
Massive Activation Monitor for Megatron-Bridge.

Monitors massive activations in post-residual hidden states — extreme outlier
values that appear in a few channels and persist across intermediate layers
via the residual connection.
"""

import logging

import torch
import torch.nn as nn

from .base import TorchProbe
from .massive_activation_metrics import (
    DEFAULT_ABSOLUTE_THRESHOLDS,
    _threshold_key,
    compute_per_channel_max,
    compute_post_norm_cosine_stability,
    compute_post_norm_sparsity,
    compute_spectral_norm_bounds,
    summarize_per_channel_max,
)

logger = logging.getLogger(__name__)


class MassiveActivationMonitor(TorchProbe):
    """Monitor massive activations in the residual stream."""

    METRIC_PREFIX = "massive_act"
    MAX_AGGREGATED = {
        "channel_max",
        "channel_median",
        "channel_p95",
        "channel_p99",
        "channel_max_ratio",
        "topk_channel_norm",
        "activation_rms",
        "massive_act_channel_count",
        "spectral_norm_max",
    }
    MIN_AGGREGATED = {
        "spectral_norm_min",
    }

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        spike_threshold_multiplier: float = 100.0,
        topk_channels: int = 3,
        sparsity_epsilon: float = 0.01,
        cosine_sample_pairs: int = 256,
        sample_layers: list[int] | None = None,
        absolute_thresholds: tuple[float, ...] = DEFAULT_ABSOLUTE_THRESHOLDS,
        log_post_norm_metrics: bool = True,
        log_activation_rms: bool = True,
        hook_timing_enabled: bool = False,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            hook_timing_enabled=hook_timing_enabled,
        )
        self.spike_threshold_multiplier = spike_threshold_multiplier
        self.topk_channels = topk_channels
        self.sparsity_epsilon = sparsity_epsilon
        self.cosine_sample_pairs = cosine_sample_pairs
        self.sample_layers = set(sample_layers) if sample_layers else None
        self.absolute_thresholds = tuple(absolute_thresholds)
        self.log_post_norm_metrics = log_post_norm_metrics
        # Single knob for the residual-scale/gain group: activation_rms plus the
        # per-token spectral-norm bounds. activation_rms is derived from the same
        # per-token pre-RMS the spectral hook computes, so they share one flag.
        self.log_activation_rms = log_activation_rms
        self.MAX_AGGREGATED = self.MAX_AGGREGATED | {
            f"channel_count_gt_{_threshold_key(t)}" for t in self.absolute_thresholds
        }
        self.tp_size = 1
        self.tp_group = None
        self._warned_per_channel_aggregate = False
        self._post_norm_failed_layers: set[int] = set()

    def _layer_metric_names(self) -> tuple[str, ...]:
        names = [
            "channel_max",
            "channel_median",
            "channel_p95",
            "channel_p99",
            "channel_max_ratio",
            "topk_channel_norm",
            "massive_act_channel_count",
        ]
        if self.log_post_norm_metrics:
            names.extend(["post_norm_sparsity", "post_norm_cosine"])
        if self.log_activation_rms:
            names.extend(["activation_rms", "spectral_norm_max", "spectral_norm_min"])
        for t in self.absolute_thresholds:
            names.append(f"channel_count_gt_{_threshold_key(t)}")
        return tuple(names)

    def register_hooks(self, model: nn.Module, layer_offset: int = 0):
        """Register forward hooks. Single-chunk path.

        For multi-chunk models, prefer the two-phase setup in
        ``setup_massive_activation_monitor`` so all chunks declare keys before
        ``allocate_buffers`` locks the schema.
        """
        self._init_parallel_state()
        targets = self._prepare_layers(model, layer_offset=layer_offset)
        if not targets:
            return
        self.allocate_buffers(next(model.parameters()).device)
        self._attach_hooks(targets)

    def _init_parallel_state(self):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
                self.tp_group = parallel_state.get_tensor_model_parallel_group()
        except ImportError:
            pass

    def _prepare_layers(self, model: nn.Module, layer_offset: int = 0) -> list[tuple[int, nn.Module]]:
        layers = self._find_transformer_layers(model)
        if not layers:
            logger.warning("[MassiveActMonitor] No transformer layers found!")
            return []

        targets: list[tuple[int, nn.Module]] = []
        for local_idx, layer in layers:
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers), layer_offset)
            if self.sample_layers and global_idx not in self.sample_layers:
                continue
            targets.append((global_idx, layer))

        for global_idx, _ in targets:
            for name in self._layer_metric_names():
                self.declare_layer_metric(global_idx, name)
        return targets

    def _attach_hooks(self, targets: list[tuple[int, nn.Module]]):
        registered = 0
        for global_idx, layer in targets:
            norm_layer = getattr(layer, "input_layernorm", None)
            if norm_layer is not None:
                hook = norm_layer.register_forward_hook(
                    self.timed_hook("input_layernorm", self._make_input_layernorm_hook(global_idx)), with_kwargs=True
                )
            else:
                hook = layer.register_forward_pre_hook(
                    self.timed_hook("residual", self._make_residual_hook(global_idx)), with_kwargs=True
                )
            self.hooks.append(hook)
            # Spectral-norm bounds need BOTH the layer input and output, so they
            # ride a dedicated forward hook on the layer itself (the residual/
            # input_layernorm hooks above only see the input).
            if self.log_activation_rms:
                spec_hook = layer.register_forward_hook(
                    self.timed_hook("spectral_norm", self._make_spectral_norm_hook(global_idx)), with_kwargs=True
                )
                self.hooks.append(spec_hook)
            registered += 1
        logger.info(f"[MassiveActMonitor] Registered {registered} hooks.")

    def _find_transformer_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        if hasattr(model, "module"):
            model = model.module

        layers = None
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = model.decoder.layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "decoder") and hasattr(lm.decoder, "layers"):
                layers = lm.decoder.layers

        if layers is None:
            return []

        return list(enumerate(layers))

    def _extract_hidden_states(self, args, kwargs=None):
        if args:
            return args[0]
        if kwargs:
            for name in ("hidden_states", "input", "x"):
                if name in kwargs:
                    return kwargs[name]
        return None

    def _first_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, tuple | list):
            for item in value:
                tensor = self._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _make_input_layernorm_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs, output):
            if not self._should_monitor():
                return
            try:
                hidden_states = self._extract_hidden_states(args, kwargs)
                if hidden_states is None:
                    return
                normalized = self._first_tensor(output)

                with torch.no_grad():
                    self._compute_residual_metrics(layer_idx, hidden_states.detach())
                    if self.log_post_norm_metrics and normalized is not None:
                        self._compute_post_norm_metrics(layer_idx, normalized.detach())
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MassiveActMonitor] Error at layer {layer_idx}: {e}")

        return hook_fn

    def _make_residual_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs=None):
            if not self._should_monitor():
                return
            try:
                hidden_states = self._extract_hidden_states(args, kwargs)
                if hidden_states is None:
                    return

                with torch.no_grad():
                    self._compute_residual_metrics(layer_idx, hidden_states.detach())
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MassiveActMonitor] Error at layer {layer_idx}: {e}")

        return hook_fn

    def _make_spectral_norm_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs, output):
            if not self._should_monitor():
                return
            try:
                pre = self._extract_hidden_states(args, kwargs)
                post = self._first_tensor(output)
                if pre is None or post is None:
                    return

                with torch.no_grad():
                    self._compute_spectral_norm(layer_idx, pre.detach(), post.detach())
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MassiveActMonitor] Spectral-norm error at layer {layer_idx}: {e}")

        return hook_fn

    def _compute_spectral_norm(self, layer_idx: int, pre: torch.Tensor, post: torch.Tensor):
        if pre.shape[-1] == 0 or pre.numel() == 0 or post.numel() == 0:
            return
        # pre is the layer input residual — the same tensor the input_layernorm /
        # residual hook sees — so activation_rms is derived here for free from the
        # per-token pre-RMS instead of squaring the input a second time.
        tensor_metrics = compute_spectral_norm_bounds(pre, post, include_activation_rms=True)
        for name, val in tensor_metrics.items():
            self.record_layer_metric(layer_idx, name, val)

    def _compute_residual_metrics(self, layer_idx: int, hidden_states: torch.Tensor):
        per_channel_max = compute_per_channel_max(hidden_states)
        per_channel_max = self._aggregate_per_channel_max(per_channel_max)
        tensor_metrics = summarize_per_channel_max(
            per_channel_max,
            threshold_multiplier=self.spike_threshold_multiplier,
            k=self.topk_channels,
            absolute_thresholds=self.absolute_thresholds,
        )

        for name, val in tensor_metrics.items():
            self.record_layer_metric(layer_idx, name, val)

    def _compute_post_norm_metrics(self, layer_idx: int, normalized: torch.Tensor):
        try:
            sparsity = compute_post_norm_sparsity(normalized, epsilon=self.sparsity_epsilon)
            self.record_layer_metric(layer_idx, "post_norm_sparsity", sparsity)

            cosine = compute_post_norm_cosine_stability(normalized, num_sample_pairs=self.cosine_sample_pairs)
            self.record_layer_metric(layer_idx, "post_norm_cosine", cosine)
        except Exception as e:
            if self.verbose and layer_idx not in self._post_norm_failed_layers:
                logger.warning(f"[MassiveActMonitor] Post-norm metrics disabled at layer {layer_idx}: {e}")
                self._post_norm_failed_layers.add(layer_idx)

    def _aggregate_per_channel_max(self, per_channel_max: torch.Tensor) -> torch.Tensor:
        """TP all-reduce on the per-channel-max vector.

        This collective is unavoidable for correctness (TP shards the channel
        dim across ranks). It runs once per hook on a length-H vector, which
        is much smaller than the QK-monitor collectives we eliminated.
        """
        if self.tp_size <= 1 or self.tp_group is None:
            return per_channel_max
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.all_reduce(per_channel_max, op=dist.ReduceOp.MAX, group=self.tp_group)
        except Exception as e:
            if self.verbose and not self._warned_per_channel_aggregate:
                logger.warning(f"[MassiveActMonitor] TP per-channel aggregation failed; using local values: {e}")
                self._warned_per_channel_aggregate = True
        return per_channel_max


def setup_massive_activation_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    spike_threshold_multiplier: float = 100.0,
    topk_channels: int = 3,
    sparsity_epsilon: float = 0.01,
    cosine_sample_pairs: int = 256,
    sample_layers: list[int] | None = None,
    absolute_thresholds: tuple[float, ...] = DEFAULT_ABSOLUTE_THRESHOLDS,
    log_post_norm_metrics: bool = True,
    log_activation_rms: bool = True,
    hook_timing_enabled: bool = False,
    monitor_dict: dict | None = None,
):
    monitor = MassiveActivationMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        spike_threshold_multiplier=spike_threshold_multiplier,
        topk_channels=topk_channels,
        sparsity_epsilon=sparsity_epsilon,
        cosine_sample_pairs=cosine_sample_pairs,
        sample_layers=sample_layers,
        absolute_thresholds=absolute_thresholds,
        log_post_norm_metrics=log_post_norm_metrics,
        log_activation_rms=log_activation_rms,
        hook_timing_enabled=hook_timing_enabled,
    )

    models = [model] if not isinstance(model, list) else model
    monitor._init_parallel_state()
    chunk_targets = []
    layer_offset = 0
    for m in models:
        targets = monitor._prepare_layers(m, layer_offset=layer_offset)
        chunk_targets.append((m, targets))
        layer_offset += len(monitor._find_transformer_layers(m))
    if any(targets for _, targets in chunk_targets):
        device = next((p.device for m in models for p in m.parameters()), None)
        assert device is not None, "no parameters across model chunks; cannot pick a device"
        monitor.allocate_buffers(device)
        for _, targets in chunk_targets:
            monitor._attach_hooks(targets)
    logger.info(f"[MassiveActMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers.")

    if monitor_dict is not None:
        monitor_dict["massive_act"] = monitor

    return model
