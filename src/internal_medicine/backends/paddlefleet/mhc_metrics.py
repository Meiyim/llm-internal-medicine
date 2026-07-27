"""mHC (Manifold-Constrained Hyper-Connections) metric compute functions.

Paddle port of ``backends/megatron/mhc_metrics.py``. Pure, stateless tensor
helpers for the ``mhc_health`` monitor. They operate on the three mappings a
``HyperConnectionModule`` produces per token:

- ``h_pre``  [..., n]     — stream aggregation gate (sigmoid)
- ``h_post`` [..., n]     — stream expansion gate (2 * sigmoid)
- ``h_res``  [..., n, n]  — Sinkhorn doubly-stochastic residual-mixing matrix

``n = num_residual_streams``.

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

import paddle


def amax_gain(mat: paddle.Tensor, axis: int) -> paddle.Tensor:
    """Per-token max-abs {row|col} sum of a batched ``[..., n, n]`` matrix, meaned over tokens.

    ``axis=-1`` sums over columns -> per-row sums (forward gain); ``axis=-2``
    sums over rows -> per-column sums (backward gain). ``sum(axis)`` collapses
    one ``n``-axis to ``[..., n]``; ``abs().max(axis=-1)`` takes the worst
    stream per token; ``mean()`` averages over all tokens.

    Returns a 0-dim tensor on ``mat``'s device/dtype (fp32 from
    ``compute_mappings``).
    """
    sums = mat.sum(axis=axis)  # [..., n]
    return sums.abs().max(axis=-1).mean()  # 0-dim


def gate_stats(h: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Mean and (unbiased) std of a gate tensor ``h`` over all elements.

    Returns two 0-dim tensors ``(mean, std)``; no host sync.
    """
    return h.mean(), h.std()
