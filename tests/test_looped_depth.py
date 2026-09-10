"""Pure-geometry unit tests for the effective-depth mapping (no torch model).

Pins the prelude/core/coda -> effective-depth bijection and the discovery gate
(``l_R > 0`` regardless of adapter; None for non-looped / degenerate blocks).

Run:
  python -m pytest tests/test_looped_depth.py
"""

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

importlib.import_module("_backend_env").skip_unless_backend("megatron")

looped_depth = importlib.import_module("internal_medicine.backends.megatron.looped_depth")
LoopedDepthMap = looped_depth.LoopedDepthMap
find_looped_depth_map = looped_depth.find_looped_depth_map


def _decoder(prelude_end, core_end, num_layers, r, **extra):
    return SimpleNamespace(
        decoder=SimpleNamespace(
            prelude_end=prelude_end,
            core_end=core_end,
            layers=list(range(num_layers)),
            looped_num_recurrence=r,
            **extra,
        )
    )


class LoopedDepthMapTest(unittest.TestCase):
    def setUp(self):
        # l_P=2, l_R=4, l_C=2, r=2 -> effective_depth 12; physical num_layers 8.
        self.m = LoopedDepthMap(prelude_end=2, core_end=6, num_layers=8, r=2)

    def test_geometry_fields(self):
        self.assertEqual((self.m.l_P, self.m.l_R, self.m.l_C, self.m.r), (2, 4, 2, 2))
        self.assertEqual(self.m.effective_depth, 12)

    def test_is_core(self):
        self.assertEqual([L for L in range(8) if self.m.is_core(L)], [2, 3, 4, 5])

    def test_prelude_maps_to_identity(self):
        self.assertEqual(self.m.depth(0), 0)
        self.assertEqual(self.m.depth(1), 1)

    def test_core_maps_per_iteration(self):
        # d = l_P + i*l_R + (L - l_P)
        self.assertEqual([self.m.depth(2, i) for i in range(2)], [2, 6])
        self.assertEqual([self.m.depth(3, i) for i in range(2)], [3, 7])
        self.assertEqual([self.m.depth(5, i) for i in range(2)], [5, 9])
        # i=0 is the physical index for a core layer.
        for L in (2, 3, 4, 5):
            self.assertEqual(self.m.depth(L, 0), L)

    def test_coda_shifts_onto_effective_axis(self):
        # d = l_P + r*l_R + (L - core_end)
        self.assertEqual(self.m.depth(6), 10)
        self.assertEqual(self.m.depth(7), 11)

    def test_unrolled_depths_shape(self):
        self.assertEqual(self.m.unrolled_depths(0), [0])  # prelude: once
        self.assertEqual(self.m.unrolled_depths(2), [2, 6])  # core: r slots
        self.assertEqual(self.m.unrolled_depths(6), [10])  # coda: once

    def test_full_axis_is_a_bijection(self):
        # Every physical layer's unrolled depths, unioned, tile [0, effective_depth)
        # exactly once.
        seen = []
        for L in range(8):
            seen.extend(self.m.unrolled_depths(L))
        self.assertEqual(sorted(seen), list(range(12)))

    def test_rep_depth_monotonic_in_L(self):
        reps = [self.m.depth(L, 0) for L in range(8)]
        self.assertEqual(reps, sorted(reps))


class FindLoopedDepthMapTest(unittest.TestCase):
    def test_returns_map_for_live_core_without_adapter(self):
        # No adapter attribute at all (variant B/C) -> still remapped on l_R > 0.
        m = find_looped_depth_map(_decoder(2, 6, 8, r=3))
        self.assertIsNotNone(m)
        self.assertEqual((m.l_P, m.l_R, m.l_C, m.r), (2, 4, 2, 3))
        self.assertEqual(m.effective_depth, 2 + 3 * 4 + 2)

    def test_unwraps_module(self):
        inner = _decoder(1, 3, 5, r=2)
        wrapped = SimpleNamespace(module=inner)
        self.assertIsNotNone(find_looped_depth_map(wrapped))

    def test_none_when_no_decoder(self):
        self.assertIsNone(find_looped_depth_map(SimpleNamespace()))

    def test_none_when_not_looped(self):
        model = SimpleNamespace(decoder=SimpleNamespace(layers=list(range(4))))
        self.assertIsNone(find_looped_depth_map(model))

    def test_none_when_recurrence_below_one(self):
        self.assertIsNone(find_looped_depth_map(_decoder(2, 6, 8, r=0)))

    def test_none_for_degenerate_l_R_zero(self):
        # Parity config: prelude_end == core_end -> l_R == 0 -> no remap.
        self.assertIsNone(find_looped_depth_map(_decoder(4, 4, 8, r=2)))

    def test_none_when_geometry_missing(self):
        model = SimpleNamespace(decoder=SimpleNamespace(looped_num_recurrence=2, layers=list(range(8))))
        self.assertIsNone(find_looped_depth_map(model))


if __name__ == "__main__":
    unittest.main()
