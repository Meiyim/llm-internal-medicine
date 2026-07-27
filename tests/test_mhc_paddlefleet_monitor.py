import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

importlib.import_module("_backend_env").skip_unless_backend("paddlefleet")

try:
    paddle = importlib.import_module("paddle")
    nn = importlib.import_module("paddle.nn")
except Exception as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

mhc_metrics = importlib.import_module("internal_medicine.backends.paddlefleet.mhc_metrics")
mhc_monitor = importlib.import_module("internal_medicine.backends.paddlefleet.mhc_monitor")
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs

PaddleMHCHealthMonitor = mhc_monitor.PaddleMHCHealthMonitor


class FakeHC(nn.Layer):
    """Stand-in for HyperConnectionModule exposing compute_mappings."""

    def __init__(self, n, h_pre, h_post, h_res):
        super().__init__()
        self.n = n
        self._h_pre = h_pre
        self._h_post = h_post
        self._h_res = h_res

    def compute_mappings(self, x):
        return self._h_pre, self._h_post, self._h_res

    def forward(self, hidden_states):  # pragma: no cover - unused
        return hidden_states, self._h_res, self._h_post


class FakeLayer(nn.Layer):
    """Stand-in for HyperConnectionTransformerLayer with the two hc modules."""

    def __init__(self, attn, mlp):
        super().__init__()
        self.self_attention_hyper_connection = attn
        self.mlp_hyper_connection = mlp


def _mhc_model(layers):
    return SimpleNamespace(decoder=SimpleNamespace(layers=nn.LayerList(layers)))


class MHCMetricsTest(unittest.TestCase):
    def test_amax_gain_row_vs_col(self):
        # [[1,2],[3,4]] -> row sums [3,7] -> max abs 7 (fwd); col sums [4,6] -> 6 (bwd).
        m = paddle.to_tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype="float32")
        self.assertAlmostEqual(float(mhc_metrics.amax_gain(m, axis=-1)), 7.0, places=5)
        self.assertAlmostEqual(float(mhc_metrics.amax_gain(m, axis=-2)), 6.0, places=5)

    def test_doubly_stochastic_gain_is_one(self):
        n = 4
        ident = paddle.eye(n).reshape([1, 1, n, n])
        self.assertAlmostEqual(float(mhc_metrics.amax_gain(ident, axis=-1)), 1.0, places=5)
        self.assertAlmostEqual(float(mhc_metrics.amax_gain(ident, axis=-2)), 1.0, places=5)

    def test_gate_stats(self):
        h = paddle.to_tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype="float32")
        mean, std = mhc_metrics.gate_stats(h)
        self.assertAlmostEqual(float(mean), 1.5, places=5)
        self.assertAlmostEqual(float(std), float(h.std()), places=5)


class MHCMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()
        # Discovery uses isinstance against the real classes; point them at the fakes.
        self._orig_layer_cls = mhc_monitor.HyperConnectionTransformerLayer
        self._orig_mod_cls = mhc_monitor.HyperConnectionModule
        mhc_monitor.HyperConnectionTransformerLayer = FakeLayer
        mhc_monitor.HyperConnectionModule = FakeHC

    def tearDown(self):
        mhc_monitor.HyperConnectionTransformerLayer = self._orig_layer_cls
        mhc_monitor.HyperConnectionModule = self._orig_mod_cls
        training_logs.reset()

    def _identity_layer(self, n, s, b):
        ident = paddle.eye(n).reshape([1, 1, n, n]).expand([s, b, n, n])

        def make_hc():
            return FakeHC(
                n=n,
                h_pre=paddle.full([s, b, n], 0.5),
                h_post=paddle.ones([s, b, n]),
                h_res=paddle.assign(ident),  # own storage so composite.clone can't alias
            )

        return FakeLayer(attn=make_hc(), mlp=make_hc())

    def _drive(self, targets, x_dim):
        # Fire each wrapped compute_mappings; grad is enabled by default in dygraph.
        for _, _, mod, _, _ in targets:
            mod.compute_mappings(paddle.randn([4, 2, x_dim]))

    def _prepare_and_attach(self, monitor, model):
        all_targets = monitor._prepare(model)
        monitor.allocate_buffers()
        monitor._attach(all_targets)
        # Return a single flat list of entries for convenience.
        return [entry for chunk_entries in all_targets for entry in chunk_entries]

    def test_identity_composite_stays_unit_gain(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])

        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self.assertTrue(targets)

        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for key in (
                "h_pre_mean",
                "h_pre_std",
                "h_post_mean",
                "h_post_std",
                "amax_gain_fwd",
                "amax_gain_bwd",
                "composite_amax_gain_fwd",
                "composite_amax_gain_bwd",
            ):
                self.assertIn(f"mhc_health/layer_0/{comp}_{key}", latest)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_amax_gain_fwd"], 1.0, places=4)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_composite_amax_gain_fwd"], 1.0, places=4)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_composite_amax_gain_bwd"], 1.0, places=4)
            # h_pre == 0.5 everywhere, h_post == 1.0 everywhere.
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_pre_mean"], 0.5, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_post_mean"], 1.0, places=5)
        # global aggregate is derived too.
        self.assertIn("mhc_health/global_attn_amax_gain_fwd", latest)

    def test_remove_hooks_restores_compute_mappings(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = PaddleMHCHealthMonitor()
        original = layer.self_attention_hyper_connection.compute_mappings
        self._prepare_and_attach(monitor, model)
        self.assertIsNot(layer.self_attention_hyper_connection.compute_mappings, original)
        monitor.remove_hooks()
        # falls back to the (bound) class method after deleting the instance attr
        self.assertEqual(
            layer.self_attention_hyper_connection.compute_mappings.__func__,
            FakeHC.compute_mappings,
        )
        self.assertEqual(monitor._wrapped, [])
        self.assertEqual(monitor._composite, {})

    def test_no_graph_retention(self):
        n, s, b = 4, 2, 3
        ident = paddle.eye(n).reshape([1, 1, n, n]).expand([s, b, n, n])
        # Outputs attached to a graph (require grad) — the wrapper must detach.
        leaf = paddle.zeros([s, b, n])
        leaf.stop_gradient = False
        h_pre = leaf + 0.5
        h_post = leaf + 1.0
        h_res = paddle.assign(ident) * (leaf.sum() + 1.0)  # grad-tracked
        hc = FakeHC(n=n, h_pre=h_pre, h_post=h_post, h_res=h_res)
        layer = FakeLayer(attn=hc, mlp=hc)
        model = _mhc_model([layer])

        monitor = PaddleMHCHealthMonitor()
        targets = self._prepare_and_attach(monitor, model)
        for _, _, mod, _, _ in targets:
            mod.compute_mappings(paddle.randn([4, 2, n * 8]))

        stored = monitor._composite[0]
        # detach() sets stop_gradient=True; the composite must not be graph-tracked.
        self.assertTrue(stored.stop_gradient)
        monitor.step()
        self.assertEqual(monitor._composite, {})  # cleared between steps


class MHCMonitorNoOpTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_auto_skip_no_hc(self):
        # A plain model (no HyperConnectionTransformerLayer) -> wraps nothing.
        plain_layer = nn.Linear(4, 4)
        model = _mhc_model([plain_layer])
        monitor_dict = {}
        mhc_monitor.setup_mhc_monitor(model, monitor_dict=monitor_dict)
        self.assertEqual(monitor_dict, {})

    def test_no_op_when_unimportable(self):
        # Simulate mHC classes not importable -> setup is a total no-op, no raise.
        orig_layer = mhc_monitor.HyperConnectionTransformerLayer
        orig_mod = mhc_monitor.HyperConnectionModule
        mhc_monitor.HyperConnectionTransformerLayer = None
        mhc_monitor.HyperConnectionModule = None
        try:
            model = _mhc_model([nn.Linear(4, 4)])
            monitor_dict = {}
            returned = mhc_monitor.setup_mhc_monitor(model, monitor_dict=monitor_dict)
            self.assertIs(returned, model)
            self.assertEqual(monitor_dict, {})
        finally:
            mhc_monitor.HyperConnectionTransformerLayer = orig_layer
            mhc_monitor.HyperConnectionModule = orig_mod


if __name__ == "__main__":
    unittest.main()
