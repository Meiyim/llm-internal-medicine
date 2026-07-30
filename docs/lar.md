# LAR Online Monitor (`lar`)

Log-Alignment Ratio (LAR) of a linear map, computed online with no SVD, as a
training-time generalization / overfitting diagnostic. See the outer spec at
`env_run/docs/lar_online_monitor.md` and arXiv:2605.28975 for theory.

## What it measures

For a `y = W x` linear on hidden inputs:

```
LAR = log_n( ||W X||_rms / (||W||_rms * ||X||_rms) ),   n = W.shape[1]  (input dim)
k   = n ** (2 * (1 - LAR))                              (effective dimension)
```

Hooked at two families of sites:

- **`lm_head`** — the output projection on the last PP stage. Hidden and logits
  are already in the forward pass; nothing recomputed. Works with tied
  input/output embeddings: under `share_embeddings_and_output_weights` Megatron
  builds `output_layer` with `skip_weight_param_allocation=True`, so
  `module.weight is None` and the tensor is only reachable via
  `shared_embedding_or_output_weight()`. The monitor resolves the weight through
  that accessor at attach time and closes over it, since reading `module.weight`
  inside the hook would yield `None` and silently drop the site.
- **`router_{L}`** — every MoE router. The router forward returns
  `(probs, routing_map)` (not raw gating logits), so the monitor recomputes
  `logits = F.linear(hidden.float(), weight.float())` locally in the hook.

Both sites use the SAME log base (`n = H`), so `lar/lm_head/lar` and
`lar/router_{L}/lar` are directly comparable.

## Loss-mask handling

A model-level `forward_pre_hook` captures `labels` (the standard Megatron GPT
kwarg). Tokens where `labels != label_ignore_index` (default `-100`) are used
for the `X` / logits sum-of-squares. The weight sum-of-squares is
token-independent and unaffected. When `labels` is not present on a forward
(eval / inference), the monitor falls back to using all tokens; `lar` still
emits.

There is **no `valid_frac` metric** reporting the mask's keep-rate. Computing it
would need a pre-mask token count, but the hooks index `x_flat[mask]` before
accumulating, so only post-mask counts reach flush time. Whether masking is live
is therefore not observable from the emitted metrics.

**Under `sequence_parallel=True`**, hidden entering routers is seq-sharded
across TP while `labels` are full-length. Router LAR falls back to unmasked
tokens on such runs (documented gotcha). `lm_head` is unaffected — it operates
post-SP-gather.

## Distributed reductions (`_flush_buffers`)

All-reduces are on `(sum_of_squares, count)` pairs — 2 fp64 scalars each,
one collective per stat. LAR is nonlinear in the sums, so per-rank averaging
of LARs is *incorrect* — pooling the sums globally is the point.

| site | ssW, nW | ssX, nX | ssZ, nZ |
|---|---|---|---|
| `lm_head` | TP-sum | DP-sum | TP-sum then DP-sum |
| `router_{L}` | (replicated) | DP-sum | DP-sum |

With TP=1/DP=1 no reduction fires. Cost is negligible either way.

## Metrics emitted per monitored step

Per site (`lm_head`, `router_0`, `router_1`, ...):
`lar/{site}/{lar, k}`.

The three RMS norms (`rms_w`, `rms_x`, `rms_z`) are computed at flush time but
**not logged** — only their combination `lar` carries the diagnostic signal, and
raw activation/weight scale is already covered by `massive_act`
(`activation_rms`, `spectral_norm_max/min`).

Globals:
- `lar/global_lm_head_lar`, `lar/global_lm_head_k` (equals the single lm_head site)
- `lar/global_router_lar`, `lar/global_router_k` (mean over routers, if any)

## Config

```yaml
internal_medicine_monitor_interval: 50
internal_medicine_monitors:
  lar: true
  # or explicit kwargs:
  # lar:
  #   hook_lm_head: true
  #   hook_moe_router: true
  #   apply_loss_mask: true
  #   label_ignore_index: -100
```

### Kwargs

| 参数 | 默认 | 说明 |
|---|---|---|
| `hook_lm_head` | `True` | 挂 output_layer（末 PP stage 有效） |
| `hook_moe_router` | `True` | 挂每个 MoE router；非 MoE 模型自动 no-op |
| `apply_loss_mask` | `True` | 用 `labels != ignore_index` 过滤 X/Z tokens |
| `label_ignore_index` | `-100` | Megatron 默认 |

## Reading the signal

- `lar ≈ 0.5` at random init.
- Healthy training: `lar` drifts in `(0.5, 0.75)` and stabilizes.
- **Declining `lar`, decline accelerating ⇒ overfitting onset** (paper's core
  signal — the *slope* is the strongest indicator). Consider logging a smoothed
  `d(lar)/dstep` downstream.
- `k = H ** (2*(1-lar))` — effective dimension used by the map. Relative trend
  is meaningful; treat absolute value cautiously at 200k-vocab scale.

## Perf notes

- Hooks: three sum-of-squares kernels + one `F.linear` (router only) per site
  per microbatch, on fp32. All 0-dim GPU tensors — no `.item()/.cpu()` on the
  hot path, no dist calls.
- Weight sum-of-squares recomputed once per step per site (constant within a
  step; guarded by `w_done` flag), not once per microbatch.
- SVD-based spectral metrics (spec §5, Appendix) are NOT computed online. Run
  offline at checkpoint boundaries if wanted.
