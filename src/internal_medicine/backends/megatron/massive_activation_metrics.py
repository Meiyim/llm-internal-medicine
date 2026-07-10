"""
Massive Activation Metrics Computation Functions.

Monitors massive activations (extreme outlier values in hidden state channels)
as characterized by Sun et al. (2026) "The Spike, the Sparse and the Sink:
Anatomy of Massive Activations and Attention Sinks" (arXiv:2603.05498).

Core metrics:
1. Channel Max — absolute maximum activation across all channels
2. Channel Median/P95/P99 — distribution of per-channel peak magnitudes
3. Channel Max Ratio — ratio of max channel to median channel (outlier severity)
4. Channel Counts over absolute thresholds — broad residual-scale growth
5. Massive Activation Channel Count — channels over median-relative threshold
6. Top-K Channel Norm — L2 norm of the top-K largest channels
7. Activation RMS — residual-stream scale
8. Post-Norm Sparsity — fraction of near-zero entries after RMSNorm (sparsification)
9. Post-Norm Cosine Stability — cosine similarity of normalized representations
   across tokens (near-constant vector detection)
10. Spectral Norm Bounds — per-token residual gain ratio (post_layer_rms /
    pre_layer_rms). The max ratio over the global batch lower-bounds the layer's
    spectral norm (largest singular value); the min ratio upper-bounds its
    smallest singular value.
11. Lipschitz / Gradient-Gain Bounds — per-token backward gradient-gain ratio
    (‖∂L/∂x‖ / ‖∂L/∂y‖). Since backprop gives ∂L/∂x = Jᵀ·∂L/∂y, this is the gain
    of the layer Jacobian's transpose on the observed gradient; the max over the
    global batch lower-bounds σ_max(J) (the layer's Lipschitz constant) and the
    min upper-bounds σ_min(J). Captured in the backward pass (see the monitor).

All metrics compute local values only; cross-rank aggregation is handled
by training_logs.gather_and_aggregate().

Reference:
    Sun, S., Canziani, A., LeCun, Y., & Zhu, J. (2026).
    The Spike, the Sparse and the Sink: Anatomy of Massive Activations
    and Attention Sinks. arXiv:2603.05498.
"""

import torch

DEFAULT_ABSOLUTE_THRESHOLDS = (10.0, 20.0, 30.0)


def _threshold_key(threshold: float) -> str:
    text = f"{threshold:g}"
    return text.replace("-", "neg").replace(".", "p")


def compute_per_channel_max(hidden_states: torch.Tensor) -> torch.Tensor:
    """Per-channel maximum absolute activation for [..., H] hidden states."""
    h = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    return h.abs().max(dim=0).values


def compute_activation_scale_stats(hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    """Absolute scale statistics over the full residual stream."""
    h = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    return {
        "activation_rms": h.square().mean().sqrt(),
    }


def compute_spectral_norm_bounds(
    pre_hidden: torch.Tensor,
    post_hidden: torch.Tensor,
    eps: float = 1e-8,
    include_activation_rms: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-token residual gain ratio (post_layer_rms / pre_layer_rms) reduced to
    max/min bounds on the layer's spectral norm.

    For each token, ``rms(y_t) / rms(x_t) == ||y_t|| / ||x_t||`` (the ``sqrt(H)``
    cancels), the gain of the residual block on that token. Over all tokens:

    - ``spectral_norm_max`` (max ratio) lower-bounds the layer's largest singular
      value (spectral norm): every observed gain is <= sup_x ||f(x)||/||x||.
    - ``spectral_norm_min`` (min ratio) upper-bounds the smallest singular value:
      every observed gain is >= inf_x ||f(x)||/||x||.

    Both inputs are the full-H residual ([S, B, H] or [B, S, H]); the RMS is a true
    full-vector RMS, so no TP channel reduction is needed. The max/min compose
    across token-partitioned ranks via ``training_logs.gather_and_aggregate``.

    When ``include_activation_rms`` is set, the input residual's global RMS
    (``activation_rms``) is derived from the same per-token ``pre_rms`` for free
    (``activation_rms == sqrt(mean_t(pre_rms_t**2))``), so callers that already run
    this function need not recompute it in a separate pass.

    Returns 0-dim GPU tensors (no host sync).
    """
    pre = pre_hidden.reshape(-1, pre_hidden.shape[-1]).float()
    post = post_hidden.reshape(-1, post_hidden.shape[-1]).float()
    pre_rms = pre.square().mean(dim=-1).sqrt()
    post_rms = post.square().mean(dim=-1).sqrt()
    ratio = post_rms / pre_rms.clamp(min=eps)
    metrics = {
        "spectral_norm_max": ratio.max(),
        "spectral_norm_min": ratio.min(),
    }
    if include_activation_rms:
        metrics["activation_rms"] = pre_rms.square().mean().sqrt()
    return metrics


def compute_grad_gain_bounds(
    grad_in: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Per-token backward gradient-gain ratio (‖∂L/∂x‖ / ‖∂L/∂y‖) reduced to
    max/min bounds on the layer's Jacobian singular values (its Lipschitz constant).

    ``grad_in`` is the gradient of the loss w.r.t. the layer INPUT hidden states
    (∂L/∂x); ``grad_out`` is the gradient w.r.t. the layer OUTPUT hidden states
    (∂L/∂y). Backprop through the layer map ``f: x -> y`` gives ``∂L/∂x = Jᵀ·∂L/∂y``
    with ``J = ∂y/∂x``, so per token

        ``rms(grad_in_t) / rms(grad_out_t) == ‖∂L/∂x_t‖ / ‖∂L/∂y_t‖ == ‖Jᵀ dy_t‖ / ‖dy_t‖``

    (the ``sqrt(H)`` cancels), the gain of ``Jᵀ`` on the observed gradient. Over all
    tokens, and since ``J`` and ``Jᵀ`` share singular values:

    - ``lipschitz_max`` (max ratio) lower-bounds ``σ_max(J)`` — the layer's spectral
      norm, i.e. its Lipschitz constant: every observed gain is <= sup_v ‖Jᵀv‖/‖v‖.
    - ``lipschitz_min`` (min ratio) upper-bounds ``σ_min(J)``: every observed gain is
      >= inf_v ‖Jᵀv‖/‖v‖.

    It also reads directly as the layer's gradient amplification in backprop:
    ``lipschitz_max > 1`` => gradients grow passing backward through the layer
    (explosion risk); ``<< 1`` => they shrink (vanishing risk).

    Both inputs are the full-H hidden-state gradient ([S, B, H] or [B, S, H]); the RMS
    is a true full-vector RMS, so no TP channel reduction is needed. The max/min compose
    across token-partitioned ranks via ``training_logs.gather_and_aggregate``.

    The per-token decomposition ignores the sequence coupling introduced by attention
    (the true Jacobian mixes tokens); it is the same simplification the forward
    ``compute_spectral_norm_bounds`` metric makes.

    Returns 0-dim GPU tensors (no host sync).
    """
    gin = grad_in.reshape(-1, grad_in.shape[-1]).float()
    gout = grad_out.reshape(-1, grad_out.shape[-1]).float()
    in_rms = gin.square().mean(dim=-1).sqrt()
    out_rms = gout.square().mean(dim=-1).sqrt()
    ratio = in_rms / out_rms.clamp(min=eps)
    return {
        "lipschitz_max": ratio.max(),
        "lipschitz_min": ratio.min(),
    }


def _nearest_quantile_from_sorted(sorted_values: torch.Tensor, q: float) -> torch.Tensor:
    idx = round((sorted_values.shape[0] - 1) * q)
    idx = min(max(idx, 0), sorted_values.shape[0] - 1)
    return sorted_values[idx]


def summarize_per_channel_max(
    per_channel_max: torch.Tensor,
    threshold_multiplier: float = 100.0,
    k: int = 3,
    absolute_thresholds: tuple[float, ...] = DEFAULT_ABSOLUTE_THRESHOLDS,
) -> dict[str, torch.Tensor]:
    """Derive scalar massive-activation metrics from per-channel maxima."""
    channel_max = per_channel_max.max()
    channel_median = per_channel_max.median()
    channel_max_ratio = channel_max / channel_median.clamp(min=1e-8)
    threshold = channel_median * threshold_multiplier
    topk_vals = per_channel_max.topk(min(k, per_channel_max.shape[0])).values
    sorted_channel_max = per_channel_max.sort().values

    metrics = {
        "channel_max": channel_max,
        "channel_median": channel_median,
        "channel_p95": _nearest_quantile_from_sorted(sorted_channel_max, 0.95),
        "channel_p99": _nearest_quantile_from_sorted(sorted_channel_max, 0.99),
        "channel_max_ratio": channel_max_ratio,
        "massive_act_channel_count": (per_channel_max > threshold).sum().float(),
        "topk_channel_norm": topk_vals.norm(),
    }
    for absolute_threshold in absolute_thresholds:
        metrics[f"channel_count_gt_{_threshold_key(absolute_threshold)}"] = (
            (per_channel_max > absolute_threshold).sum().float()
        )
    return metrics


def compute_channel_max(hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute per-channel maximum absolute activation statistics.

    Tracks the "rise–plateau–fall" lifecycle of massive activations across layers.
    A sudden spike in channel_max indicates a step-up block is injecting outliers.

    Args:
        hidden_states: [S, B, H] or [B, S, H] post-residual hidden states.
            (Flattened to [*, H] internally)

    Returns:
        Dict with:
            channel_max: max absolute value across all positions and channels (scalar)
            channel_median: median of per-channel max absolute values (scalar)
            channel_p95/channel_p99: high quantiles of per-channel max absolute values
            channel_max_ratio: channel_max / channel_median (outlier severity)
    """
    return summarize_per_channel_max(compute_per_channel_max(hidden_states))


def compute_massive_activation_channel_count(
    hidden_states: torch.Tensor,
    threshold_multiplier: float = 100.0,
) -> torch.Tensor:
    """Count channels with activations exceeding a dynamic threshold.

    The threshold is set relative to the median channel magnitude:
        threshold = median_channel_max × threshold_multiplier

    This captures Property (ii) from Sun et al. (2026): massive activations
    are confined to a small subset of channels.

    Args:
        hidden_states: [S, B, H] or [B, S, H] post-residual hidden states.
        threshold_multiplier: multiplier on median to define "spike" threshold.

    Returns:
        Scalar tensor: number of channels exceeding the threshold.
    """
    return summarize_per_channel_max(
        compute_per_channel_max(hidden_states),
        threshold_multiplier=threshold_multiplier,
    )["massive_act_channel_count"]


def compute_topk_channel_norm(
    hidden_states: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """L2 norm of the top-K largest channel activations.

    Tracks the magnitude of the most extreme channels. In models with massive
    activations, this should show the "rise–plateau–fall" pattern across layers
    (Figure 1 in Sun et al. 2026).

    Args:
        hidden_states: [S, B, H] or [B, S, H] post-residual hidden states.
        k: number of top channels to include.

    Returns:
        Scalar tensor: L2 norm of the top-K per-channel-max values.
    """
    return summarize_per_channel_max(compute_per_channel_max(hidden_states), k=k)["topk_channel_norm"]


def compute_post_norm_sparsity(
    normalized_states: torch.Tensor,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Fraction of near-zero entries in post-RMSNorm hidden states.

    After normalization, spike tokens become sparse vectors where non-spike
    channels are suppressed to near-zero (Equation 24, Sun et al. 2026).
    High sparsity indicates the model is creating "implicit parameters" via
    the normalization-spike interaction.

    Args:
        normalized_states: [S, B, H] or [B, S, H] hidden states AFTER RMSNorm.
        epsilon: threshold below which a value is considered "near-zero".

    Returns:
        Scalar tensor: fraction of entries with |x| < epsilon.
    """
    h = normalized_states.reshape(-1).float()
    return (h.abs() < epsilon).float().mean()


def compute_post_norm_cosine_stability(
    normalized_states: torch.Tensor,
    num_sample_pairs: int = 256,
) -> torch.Tensor:
    """Cosine similarity among token representations after normalization.

    Near-constant post-norm representations (cosine → 1.0) indicate that
    normalization has collapsed diverse spike tokens into identical vectors
    (Figure 5, Sun et al. 2026). This is a precondition for attention sinks.

    Args:
        normalized_states: [S, B, H] post-RMSNorm hidden states.
        num_sample_pairs: number of random pairs to sample for efficiency.

    Returns:
        Scalar tensor: mean pairwise cosine similarity (sampled).
    """
    # Flatten to [num_tokens, hidden_dim]
    h = normalized_states.reshape(-1, normalized_states.shape[-1]).float()
    num_tokens = h.shape[0]

    if num_tokens < 2:
        return torch.tensor(1.0, device=h.device)

    # Sample random pairs
    n_pairs = min(num_sample_pairs, num_tokens * (num_tokens - 1) // 2)
    idx_a = torch.randint(0, num_tokens, (n_pairs,), device=h.device)
    idx_b = torch.randint(0, num_tokens - 1, (n_pairs,), device=h.device)
    # Avoid same index
    idx_b = idx_b + (idx_b >= idx_a).long()

    vec_a = h[idx_a]
    vec_b = h[idx_b]

    cosine = torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=-1)
    return cosine.mean()
