"""Periodic on-disk activation (hidden-state) dumping for the Megatron backend.

Snapshots residual-stream hidden states ``[s, b, h]`` to disk every
``monitor_interval`` steps for offline structural analysis (effective rank,
anisotropy, clustering, massive-outlier detection — see ``spec_entropy_explorer.py``).

This is an ORTHOGONAL disk-dump hook, NOT a reuse of Megatron's fine-grained
activation offloading: that system is GPU<->pinned-CPU-RAM only, autograd-coupled,
with a same-step reload+free (pool-reuse) lifecycle and no disk path. We borrow only
its transfer *technique* — a non-blocking D2H copy into a pinned buffer on a side CUDA
stream, ordered by a CUDA event — and defer the blocking sync + disk write to flush
time (``_flush_buffers``, the cold path). This honors ``monitor-hook-perf-rules``:
no ``.item()/.cpu()`` sync and no collectives on the hot path.

Aggressive-sampling defaults keep long runs from filling disk: first microbatch only,
a subset of layers, one DP/TP rank, and RANDOM token positions (not the first K — the
leading tokens are biased by BOS / attention-sink massive activations, so a first-K
slice is unrepresentative of the residual stream). A rotation cap (``max_dump_steps``)
prunes the oldest ``step_*`` directories after each write so disk stays bounded no
matter how long the run is.

A ``min_channel_max_ratio`` gate targets massive-activation "ill-conditioned" states:
each captured tensor's per-channel ``max/median`` ratio is computed on the hot path
(0-dim GPU tensor, no sync) and, at flush, dumps below the threshold are discarded
before hitting disk. The ratio is recorded in the safetensors metadata regardless.
"""

import logging
import os
import shutil

import torch
import torch.nn as nn

from .base import TorchProbe

logger = logging.getLogger(__name__)


class ActivationDumpMonitor(TorchProbe):
    """Dump per-layer residual hidden states to disk on monitored steps.

    Records no scalar metrics: forward hooks stash GPU->pinned-CPU copies of a
    random-position token sample, and ``_flush_buffers`` writes them as safetensors
    files (with a metadata sidecar) once per step.
    """

    METRIC_PREFIX = "act_dump"

    def __init__(
        self,
        dump_dir: str = "./outputs/act_dumps",
        which: str = "output",
        sample_layers: list[int] | None = None,
        n_sample_tokens: int | None = 512,
        token_sample_seed: int = 0,
        first_microbatch_only: bool = True,
        dump_dp_ranks: list[int] | None = None,
        dump_tp_ranks: list[int] | None = None,
        max_dumps_per_step: int | None = None,
        max_dump_steps: int | None = 20,
        min_channel_max_ratio: float | None = None,
        monitor_interval: int = 1,
        verbose: bool = False,
        hook_timing_enabled: bool = False,
    ):
        super().__init__(
            monitor_interval=monitor_interval,
            verbose=verbose,
            hook_timing_enabled=hook_timing_enabled,
        )
        if which not in ("output", "input"):
            raise ValueError(f"which must be 'output' or 'input', got {which!r}")
        if max_dump_steps is not None and max_dump_steps < 1:
            raise ValueError(f"max_dump_steps must be >= 1 or None, got {max_dump_steps!r}")
        if min_channel_max_ratio is not None and min_channel_max_ratio < 0:
            raise ValueError(f"min_channel_max_ratio must be >= 0 or None, got {min_channel_max_ratio!r}")
        self.dump_dir = dump_dir
        self.which = which
        self.sample_layers = set(sample_layers) if sample_layers else None
        self.n_sample_tokens = n_sample_tokens
        self.token_sample_seed = token_sample_seed
        self.first_microbatch_only = first_microbatch_only
        self.dump_dp_ranks = set(dump_dp_ranks) if dump_dp_ranks is not None else {0}
        self.dump_tp_ranks = set(dump_tp_ranks) if dump_tp_ranks is not None else {0}
        self.max_dumps_per_step = max_dumps_per_step
        self.max_dump_steps = max_dump_steps
        self.min_channel_max_ratio = min_channel_max_ratio

        # Parallel state (resolved in _init_parallel_state).
        self.tp_rank = 0
        self.dp_rank = 0
        self.global_rank = 0
        self.sequence_parallel = False
        self._this_rank_dumps = True

        # Side stream for the non-blocking D2H copy (offload's technique). None on CPU.
        self._d2h_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        # Hot-path scratch, reset at flush.
        self._pending: list[dict] = []
        self._captured_this_step: set[int] = set()
        self._cached_idx_step: int | None = None
        self._cached_idx_cpu: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Lifecycle: register -> prepare -> allocate -> attach
    # ------------------------------------------------------------------

    def register_hooks(self, model: nn.Module, layer_offset: int = 0):
        """Register dump hooks. Single-chunk path.

        For multi-chunk models (VPP) prefer ``setup_activation_dump_monitor``,
        which resolves layers across every chunk before allocating buffers.
        """
        self._init_parallel_state()
        targets = self._prepare_layers(model, layer_offset=layer_offset)
        if not targets:
            return
        self.allocate_buffers(next(model.parameters()).device)
        self._attach_hooks(targets, model=model)

    def _init_parallel_state(self):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
                try:
                    self.dp_rank = parallel_state.get_data_parallel_rank()
                except Exception:
                    self.dp_rank = 0
        except ImportError:
            pass
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                self.global_rank = dist.get_rank()
        except Exception:
            pass
        # Precompute once: this rank only writes if it is in the DP/TP allow-lists.
        self._this_rank_dumps = self.dp_rank in self.dump_dp_ranks and self.tp_rank in self.dump_tp_ranks

    def _resolve_sequence_parallel(self, model: nn.Module) -> bool:
        m = model.module if hasattr(model, "module") else model
        config = getattr(m, "config", None)
        return bool(getattr(config, "sequence_parallel", False)) if config is not None else False

    def _prepare_layers(self, model: nn.Module, layer_offset: int = 0) -> list[tuple[int, nn.Module]]:
        layers = self._find_transformer_layers(model)
        if not layers:
            logger.warning("[ActivationDumpMonitor] No transformer layers found!")
            return []
        self.sequence_parallel = self._resolve_sequence_parallel(model) or self.sequence_parallel
        targets: list[tuple[int, nn.Module]] = []
        for local_idx, layer in layers:
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers), layer_offset)
            if self.sample_layers and global_idx not in self.sample_layers:
                continue
            targets.append((global_idx, layer))
        return targets

    def _attach_hooks(self, targets: list[tuple[int, nn.Module]], model: nn.Module | None = None):
        registered = 0
        for global_idx, layer in targets:
            hook = layer.register_forward_hook(
                self.timed_hook("act_dump", self._make_dump_hook(global_idx)), with_kwargs=True
            )
            self.hooks.append(hook)
            registered += 1
        logger.info(
            f"[ActivationDumpMonitor] Registered {registered} dump hooks (this_rank_dumps={self._this_rank_dumps})."
        )

    # ------------------------------------------------------------------
    # Model introspection (mirrors MassiveActivationMonitor)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Random token-position sampling
    # ------------------------------------------------------------------

    def _token_indices(self, n_tokens: int, device: torch.device):
        """Random token positions for this step, shared across sampled layers.

        Generated on CPU so the same indices double as file metadata without a D2H
        sync, then moved to ``device``. Cached per ``step_count`` so every sampled
        layer in a step gathers the SAME positions (cross-layer comparability). Sorted
        for readability; the exact positions are recorded in the dump's ``token_index``.
        """
        k = self.n_sample_tokens
        if k is None or k >= n_tokens:
            idx_cpu = torch.arange(n_tokens, dtype=torch.long)
        else:
            if (
                self._cached_idx_step != self.step_count
                or self._cached_idx_cpu is None
                or self._cached_idx_cpu.numel() != k
            ):
                g = torch.Generator()
                g.manual_seed(self.token_sample_seed + self.step_count)
                self._cached_idx_cpu = torch.randperm(n_tokens, generator=g)[:k].sort().values.contiguous()
                self._cached_idx_step = self.step_count
            idx_cpu = self._cached_idx_cpu
        idx_dev = idx_cpu if device.type == "cpu" else idx_cpu.to(device, non_blocking=True)
        return idx_cpu, idx_dev

    def _async_copy_to_cpu(self, t: torch.Tensor):
        """Non-blocking D2H into a pinned buffer on a side stream (offload's technique).

        Returns ``(cpu_tensor, event)``; ``event`` is None on CPU. The side stream first
        waits on the compute stream (so ``t`` is fully produced), copies into pinned host
        memory, and records an event; ``t.record_stream`` keeps its device memory alive
        until the copy finishes. The blocking wait happens in ``_flush_buffers`` (cold
        path), so the hot path never syncs.
        """
        t = t.contiguous()
        if self._d2h_stream is None or t.device.type != "cuda":
            return t.to("cpu").clone(), None
        stream = self._d2h_stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            cpu_tensor = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            cpu_tensor.copy_(t, non_blocking=True)
        t.record_stream(stream)
        event = torch.cuda.Event()
        event.record(stream)
        return cpu_tensor, event

    # ------------------------------------------------------------------
    # Hot path: capture a random-position sample (no D2H sync, no collectives)
    # ------------------------------------------------------------------

    def _make_dump_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs, output):
            if not self._this_rank_dumps or not self._should_monitor():
                return
            if self.first_microbatch_only and layer_idx in self._captured_this_step:
                return
            if self.max_dumps_per_step is not None and len(self._pending) >= self.max_dumps_per_step:
                return
            try:
                source = output if self.which == "output" else self._extract_hidden_states(args, kwargs)
                hidden = self._first_tensor(source)
                if hidden is None or hidden.dim() < 2:
                    return
                with torch.no_grad():
                    h = hidden.detach()
                    seq = h.shape[0]
                    batch = h.shape[1] if h.dim() >= 3 else 1
                    hdim = h.shape[-1]
                    flat = h.reshape(-1, hdim)
                    n_tokens = flat.shape[0]
                    ratio_gpu = self._channel_max_ratio(flat)
                    idx_cpu, idx_dev = self._token_indices(n_tokens, flat.device)
                    sample = flat.index_select(0, idx_dev)
                    cpu_tensor, event = self._async_copy_to_cpu(sample)
                    ratio_cpu, ratio_event = self._async_copy_to_cpu(ratio_gpu)
                meta = {
                    "step": str(self.step_count),
                    "layer_idx": str(layer_idx),
                    "which": self.which,
                    "global_rank": str(self.global_rank),
                    "pp_rank": str(self.pp_rank),
                    "tp_rank": str(self.tp_rank),
                    "dp_rank": str(self.dp_rank),
                    "seq": str(seq),
                    "batch": str(batch),
                    "hidden_size": str(hdim),
                    "n_tokens": str(n_tokens),
                    "n_sample_tokens": str(idx_cpu.numel()),
                    "token_sample_seed": str(self.token_sample_seed),
                    "src_dtype": str(h.dtype).replace("torch.", ""),
                    "sequence_parallel": str(self.sequence_parallel),
                    "token_layout": "seq_major_flattened_s_times_b",
                }
                self._pending.append(
                    {
                        "layer_idx": layer_idx,
                        "step": self.step_count,
                        "cpu": cpu_tensor,
                        "idx": idx_cpu,
                        "event": event,
                        "ratio_cpu": ratio_cpu,
                        "ratio_event": ratio_event,
                        "meta": meta,
                    }
                )
                self._captured_this_step.add(layer_idx)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[ActivationDumpMonitor] dump error at layer {layer_idx}: {e}")

        return hook_fn

    def _channel_max_ratio(self, flat: torch.Tensor) -> torch.Tensor:
        """0-dim GPU tensor: max / median of per-channel |activation| max. No host sync."""
        per_channel_max = flat.abs().amax(dim=0).float()
        return per_channel_max.max() / per_channel_max.median().clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Cold path: sync the side stream and write to disk (runs at step end)
    # ------------------------------------------------------------------

    def _flush_buffers(self) -> None:
        super()._flush_buffers()  # harmless: no scalar keys declared
        if not self._pending:
            self._captured_this_step.clear()
            return
        try:
            from safetensors.torch import save_file
        except Exception as e:
            logger.error(f"[ActivationDumpMonitor] safetensors unavailable, dropping {len(self._pending)} dumps: {e}")
            self._pending.clear()
            self._captured_this_step.clear()
            return
        written = 0
        skipped = 0
        for entry in self._pending:
            event = entry["event"]
            if event is not None:
                event.synchronize()  # cold path: safe to block here, off the hot path
            ratio_event = entry.get("ratio_event")
            if ratio_event is not None:
                ratio_event.synchronize()
            ratio_val = float(entry["ratio_cpu"].item())
            entry["meta"]["channel_max_ratio"] = f"{ratio_val:.6f}"
            if self.min_channel_max_ratio is not None and ratio_val < self.min_channel_max_ratio:
                skipped += 1
                continue
            hidden = entry["cpu"]
            # ``self.step_count`` reflects the trainer's global iteration during
            # this flush (synced by ``monitor.step(global_step=...)`` before the
            # flush runs; falls back to the hook-time counter when no global_step
            # is provided). Prefer it over ``entry["step"]`` so file layout stays
            # anchored to the trainer's iteration across checkpoint resume.
            step_id = int(self.step_count)
            entry["meta"]["step"] = str(step_id)
            step_dir = os.path.join(self.dump_dir, f"step_{step_id:07d}")
            os.makedirs(step_dir, exist_ok=True)
            fname = (
                f"rank{self.global_rank}_pp{self.pp_rank}_tp{self.tp_rank}_dp{self.dp_rank}"
                f"_layer{entry['layer_idx']}_{self.which}.safetensors"
            )
            path = os.path.join(step_dir, fname)
            tensors = {
                "hidden": hidden.contiguous(),
                "token_index": entry["idx"].to(torch.long).contiguous(),
            }
            try:
                save_file(tensors, path, metadata=entry["meta"])
                written += 1
            except Exception as e:
                logger.error(f"[ActivationDumpMonitor] failed writing {path}: {e}")
        if self.verbose and (written or skipped):
            logger.info(
                f"[ActivationDumpMonitor] step {self.step_count}: "
                f"wrote {written}, skipped {skipped} (channel_max_ratio filter)."
            )
        self._pending.clear()
        self._captured_this_step.clear()
        if written:
            self._rotate_dumps()

    def _rotate_dumps(self) -> None:
        """Retain only the ``max_dump_steps`` most recent ``step_*`` dirs (rotation).

        Keeps disk bounded on long runs: after each write we prune the oldest
        complete step directories beyond the retention count. Whole ``step_*`` dirs
        are the rotation unit (they hold every layer/rank for that step), and only
        steps far older than the current one are removed, so no rank is still
        writing into a pruned dir. ``ignore_errors=True`` tolerates a concurrent
        prune when several ranks dump to the same ``dump_dir``.
        """
        if self.max_dump_steps is None:
            return
        try:
            entries = os.listdir(self.dump_dir)
        except OSError:
            return
        step_dirs = []
        for name in entries:
            if not name.startswith("step_"):
                continue
            suffix = name[len("step_") :]
            if suffix.isdigit():
                step_dirs.append((int(suffix), name))
        if len(step_dirs) <= self.max_dump_steps:
            return
        step_dirs.sort()  # ascending by step number; oldest first
        for _, name in step_dirs[: len(step_dirs) - self.max_dump_steps]:
            shutil.rmtree(os.path.join(self.dump_dir, name), ignore_errors=True)
        if self.verbose:
            logger.info(
                f"[ActivationDumpMonitor] rotated dumps: kept {self.max_dump_steps} most-recent "
                f"step dirs, pruned {len(step_dirs) - self.max_dump_steps}."
            )


def setup_activation_dump_monitor(
    model,
    dump_dir: str = "./outputs/act_dumps",
    which: str = "output",
    sample_layers: list[int] | None = None,
    n_sample_tokens: int | None = 512,
    token_sample_seed: int = 0,
    first_microbatch_only: bool = True,
    dump_dp_ranks: list[int] | None = None,
    dump_tp_ranks: list[int] | None = None,
    max_dumps_per_step: int | None = None,
    max_dump_steps: int | None = 20,
    min_channel_max_ratio: float | None = None,
    monitor_interval: int = 1,
    verbose: bool = False,
    hook_timing_enabled: bool = False,
    monitor_dict: dict | None = None,
):
    """Build the monitor and attach dump hooks, handling VPP multi-chunk models.

    Mirrors ``setup_massive_activation_monitor``'s three-phase setup (prepare across
    all chunks -> allocate once -> attach per chunk) so the schema/lifecycle is locked
    before hooks fire, even under interleaved pipelining.
    """
    monitor = ActivationDumpMonitor(
        dump_dir=dump_dir,
        which=which,
        sample_layers=sample_layers,
        n_sample_tokens=n_sample_tokens,
        token_sample_seed=token_sample_seed,
        first_microbatch_only=first_microbatch_only,
        dump_dp_ranks=dump_dp_ranks,
        dump_tp_ranks=dump_tp_ranks,
        max_dumps_per_step=max_dumps_per_step,
        max_dump_steps=max_dump_steps,
        min_channel_max_ratio=min_channel_max_ratio,
        monitor_interval=monitor_interval,
        verbose=verbose,
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
        for m, targets in chunk_targets:
            monitor._attach_hooks(targets, model=m)
    logger.info(f"[ActivationDumpMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers.")

    if monitor_dict is not None:
        monitor_dict["act_dump"] = monitor

    return model
