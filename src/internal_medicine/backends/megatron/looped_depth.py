"""Effective-depth (unrolled) layer indexing for looped/recurrent-depth models.

A looped model (arXiv 2502.05171) runs its recurrent core
``block.layers[l_P : l_P+l_R]`` ``r`` times per forward — the **same physical
module instances** reused each iteration. A per-layer monitor that keys metrics by
physical ``layer_number`` therefore folds the ``r`` core visits into a single slot
(losing the per-iteration trajectory) and biases every ``global_*`` reduction
``r``-fold toward core behavior (the flush-time mean is count-weighted).

Re-keying by **effective depth** ``d`` — a flat axis ``[0, effective_depth)`` that
unrolls the loop — fixes both: each unrolled position becomes its own slot, observed
once. It also lines the axis up 1:1 with an equal-FLOPs fixed-depth baseline.

Mapping (physical ``L``, core-iteration ``i in [0, r)``)::

    prelude  L in [0, l_P):              d = L
    core     L in [l_P, l_P+l_R):        d = l_P + i*l_R + (L - l_P)
    coda     L in [l_P+l_R, num_layers): d = l_P + r*l_R + (L - (l_P+l_R))

This is a bijection over unrolled positions, so ``(d, metric)`` keys never collide.
For a core layer ``depth(L, 0) == L``; coda shifts up onto the effective axis.

This module is pure geometry — no torch, no counters. Per-module iteration counters
live in the monitor hook closures.
"""


class LoopedDepthMap:
    """Stateless physical-layer -> effective-depth mapping for one looped core.

    ``l_R == core_end - prelude_end`` is the recurrent-core width; a live looped
    core always has ``l_R > 0`` (the degenerate ``l_R == 0`` parity config is
    filtered out by :func:`find_looped_depth_map`).
    """

    def __init__(self, prelude_end: int, core_end: int, num_layers: int, r: int):
        self.prelude_end = prelude_end
        self.core_end = core_end
        self.num_layers = num_layers
        self.r = r
        self.l_P = prelude_end
        self.l_R = core_end - prelude_end
        self.l_C = num_layers - core_end
        self.effective_depth = self.l_P + r * self.l_R + self.l_C

    def is_core(self, L: int) -> bool:
        return self.prelude_end <= L < self.core_end

    def depth(self, L: int, i: int = 0) -> int:
        """Effective depth of physical layer ``L`` at core-iteration ``i``.

        ``i`` is ignored for prelude/coda layers (they fire once)."""
        if self.prelude_end > L:
            return L
        if self.core_end > L:
            return self.l_P + i * self.l_R + (L - self.l_P)
        return self.l_P + self.r * self.l_R + (L - self.core_end)

    def unrolled_depths(self, L: int) -> list[int]:
        """All effective depths physical layer ``L`` maps onto across a forward:
        ``r`` slots for a core layer, a single slot for prelude/coda."""
        if self.is_core(L):
            return [self.depth(L, i) for i in range(self.r)]
        return [self.depth(L, 0)]


def find_looped_depth_map(model) -> "LoopedDepthMap | None":
    """Return a :class:`LoopedDepthMap` if ``model`` hosts a live looped core, else None.

    Unwraps ``.module`` and reads ``model.decoder``. Requires an int
    ``looped_num_recurrence >= 1`` and a readable ``prelude_end`` / ``core_end`` /
    ``layers`` with ``core_end > prelude_end`` (i.e. ``l_R > 0``). A non-looped
    decoder, or the degenerate ``l_R == 0`` parity config, returns None so the
    caller keeps its physical-layer indexing verbatim.

    Unlike ``looped_monitor._find_looped_block`` there is **no** ``adapter`` check:
    the adapter exists only for ``step_injection == "e"`` (variant A), but variants
    B (none) and C (timestep) fire the core ``r`` times and suffer the same bias.
    Gating on ``l_R > 0`` fixes all three.
    """
    if hasattr(model, "module"):
        model = model.module
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return None
    r = getattr(decoder, "looped_num_recurrence", None)
    if not isinstance(r, int) or r < 1:
        return None
    prelude_end = getattr(decoder, "prelude_end", None)
    core_end = getattr(decoder, "core_end", None)
    layers = getattr(decoder, "layers", None)
    if not isinstance(prelude_end, int) or not isinstance(core_end, int) or layers is None:
        return None
    if core_end <= prelude_end:
        return None
    return LoopedDepthMap(prelude_end, core_end, len(layers), r)
