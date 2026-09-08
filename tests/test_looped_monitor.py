# -*- coding: utf-8 -*-
"""LoopedHealthMonitor: recurrence-axis metrics for a looped transformer.

These pin the behaviours the plan (federated-wondering-nova §7) calls out:
  1. the schema is declared per recurrence iteration (r firings of one reused
     module must NOT collapse into a single averaged slot)
  2. token_cosine detects representation collapse (->1.0), eff_rank detects a
     low-rank state, state_delta detects a stationary vs moving recurrence,
     adapter_e_ratio reads ||A_e e|| / ||A_s s|| off the concat adapter
  3. state_delta is absent for iteration 0 (no previous state to diff)
  4. add / none injection declares no adapter_e_ratio; a non-looped or degenerate
     (l_R == 0) block is a clean no-op so the monitor is safe in the ``all`` list
  5. the monitor_interval gate suppresses recording on unmonitored steps

Uses fakes (no ernie / megatron model), so it runs anywhere torch is importable.

Run:
  python -m pytest tests/test_looped_monitor.py
"""

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

importlib.import_module("_backend_env").skip_unless_backend("megatron")

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
except Exception as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"torch backend unavailable: {exc}") from exc

looped_monitor = importlib.import_module("internal_medicine.backends.megatron.looped_monitor")
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs

LoopedHealthMonitor = looped_monitor.LoopedHealthMonitor
setup_looped_monitor = looped_monitor.setup_looped_monitor
_covariance_effective_rank = looped_monitor._covariance_effective_rank


class FakeStateNorm(nn.Module):
    """Identity stand-in for the paper's n_c: the hook measures its output, so
    passing the tensor we want to probe through it returns it unchanged."""

    def forward(self, x):
        return x


class FakeAdapter(nn.Module):
    """Stand-in for LoopedAdapter exposing injection + the branch-norm byproducts."""

    def __init__(self, injection="concat"):
        super().__init__()
        self.injection = injection
        self.last_state_norm = None
        self.last_embed_norm = None

    def forward(self, state, embed):
        # The real adapter records RMS per branch here; in tests we set the two
        # norms directly before calling, so just pass state through.
        return state


def _looped_model(r, injection="concat", with_block=True):
    """SimpleNamespace shaped like ErnieDevGPTModel.decoder for discovery."""
    if not with_block:
        # A plain decoder: no state_norm / adapter / looped_num_recurrence.
        return SimpleNamespace(decoder=SimpleNamespace(layers=nn.ModuleList([nn.Linear(4, 4)])))
    decoder = SimpleNamespace(
        state_norm=FakeStateNorm(),
        adapter=FakeAdapter(injection=injection),
        looped_num_recurrence=r,
    )
    # allocate_buffers picks a device off model.parameters(); give it one param.
    decoder._probe = nn.Linear(2, 2)
    model = SimpleNamespace(decoder=decoder)
    # setup uses next(model.parameters()); expose the decoder's param.
    model.parameters = decoder._probe.parameters
    return model


def _register(monitor, model):
    block = monitor._prepare_layers(model)
    assert block is not None
    monitor.allocate_buffers(torch.device("cpu"))
    monitor._attach_hooks(block)
    return block


def _drive_forward(block, states, embed_norms=None, state_norms=None):
    """Run one forward's worth of iterations: adapter then state_norm, r times."""
    r = block.looped_num_recurrence
    for i in range(r):
        if block.adapter.injection == "concat" and embed_norms is not None:
            block.adapter.last_embed_norm = torch.tensor(float(embed_norms[i]))
            block.adapter.last_state_norm = torch.tensor(float(state_norms[i]))
        block.adapter(states[i], states[i])
        block.state_norm(states[i])


class LoopedMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    # ------------------------------------------------------------------
    # schema / metric presence
    # ------------------------------------------------------------------
    def test_metric_prefix_matches_registry_key(self):
        # setup writes monitor_dict["looped_health"]; that key must equal the
        # prefix or the flush lands under the wrong tag.
        self.assertEqual(LoopedHealthMonitor.METRIC_PREFIX, "looped_health")
        self.assertEqual(LoopedHealthMonitor.MAX_AGGREGATED, set())
        self.assertEqual(LoopedHealthMonitor.MIN_AGGREGATED, set())

    def test_schema_declared_per_iteration(self):
        r = 4
        monitor = LoopedHealthMonitor()
        block = _register(monitor, _looped_model(r, injection="concat"))
        states = [torch.randn(8, 2, 16) for _ in range(r)]
        _drive_forward(block, states, embed_norms=[1.0] * r, state_norms=[1.0] * r)
        monitor.step()

        latest = training_logs.get_latest(prefix="looped_health")
        for i in range(r):
            self.assertIn(f"looped_health/layer_{i}/token_cosine", latest)
            self.assertIn(f"looped_health/layer_{i}/eff_rank", latest)
            self.assertIn(f"looped_health/layer_{i}/adapter_e_ratio", latest)
        # state_delta only for i >= 1 (iteration 0 has no previous state).
        self.assertNotIn("looped_health/layer_0/state_delta", latest)
        for i in range(1, r):
            self.assertIn(f"looped_health/layer_{i}/state_delta", latest)
        # cross-iteration global reductions are derived at flush.
        self.assertIn("looped_health/global_token_cosine", latest)
        self.assertIn("looped_health/global_eff_rank", latest)

    # ------------------------------------------------------------------
    # metric semantics
    # ------------------------------------------------------------------
    def test_token_cosine_flags_collapse(self):
        r = 2
        monitor = LoopedHealthMonitor()
        block = _register(monitor, _looped_model(r, injection="add"))
        # iter 0: identical tokens -> cosine 1.0 (collapse). iter 1: diverse.
        collapsed = torch.ones(64, 1, 16)
        diverse = torch.randn(64, 1, 16)
        _drive_forward(block, [collapsed, diverse])
        monitor.step()

        latest = training_logs.get_latest(prefix="looped_health")
        self.assertAlmostEqual(latest["looped_health/layer_0/token_cosine"], 1.0, places=4)
        self.assertLess(latest["looped_health/layer_1/token_cosine"], 0.5)

    def test_eff_rank_low_for_rank_one_state(self):
        r = 2
        monitor = LoopedHealthMonitor()
        block = _register(monitor, _looped_model(r, injection="add"))
        v = torch.randn(1, 16)
        coeffs = torch.randn(64, 1)
        rank_one = (coeffs @ v).reshape(64, 1, 16)  # every token is a multiple of v
        full = torch.randn(64, 1, 16)
        _drive_forward(block, [rank_one, full])
        monitor.step()

        latest = training_logs.get_latest(prefix="looped_health")
        self.assertAlmostEqual(latest["looped_health/layer_0/eff_rank"], 1.0, places=3)
        self.assertGreater(latest["looped_health/layer_1/eff_rank"], 5.0)

    def test_state_delta_zero_when_stationary(self):
        r = 3
        monitor = LoopedHealthMonitor()
        block = _register(monitor, _looped_model(r, injection="none"))
        s = torch.randn(8, 2, 16)
        # iter0 = s, iter1 = s (stationary -> delta 0), iter2 = 2s (delta 0.5).
        _drive_forward(block, [s, s.clone(), 2.0 * s])
        monitor.step()

        latest = training_logs.get_latest(prefix="looped_health")
        self.assertAlmostEqual(latest["looped_health/layer_1/state_delta"], 0.0, places=5)
        self.assertAlmostEqual(latest["looped_health/layer_2/state_delta"], 0.5, places=4)

    def test_adapter_e_ratio_is_embed_over_state(self):
        r = 2
        monitor = LoopedHealthMonitor()
        block = _register(monitor, _looped_model(r, injection="concat"))
        states = [torch.randn(8, 2, 16) for _ in range(r)]
        _drive_forward(block, states, embed_norms=[2.0, 3.0], state_norms=[0.5, 6.0])
        monitor.step()

        latest = training_logs.get_latest(prefix="looped_health")
        self.assertAlmostEqual(latest["looped_health/layer_0/adapter_e_ratio"], 4.0, places=4)  # 2/0.5
        self.assertAlmostEqual(latest["looped_health/layer_1/adapter_e_ratio"], 0.5, places=4)  # 3/6

    # ------------------------------------------------------------------
    # injection modes / no-op guards
    # ------------------------------------------------------------------
    def test_add_injection_declares_no_adapter_ratio(self):
        r = 3
        monitor = LoopedHealthMonitor()
        block = _register(monitor, _looped_model(r, injection="add"))
        # add adapter has no separable branches -> no adapter hook attached.
        self.assertEqual(len(monitor.hooks), 1)
        states = [torch.randn(8, 2, 16) for _ in range(r)]
        _drive_forward(block, states)
        monitor.step()
        latest = training_logs.get_latest(prefix="looped_health")
        self.assertFalse(any("adapter_e_ratio" in k for k in latest))
        # state-side metrics still present.
        self.assertIn("looped_health/layer_0/token_cosine", latest)

    def test_non_looped_model_is_noop(self):
        monitor = LoopedHealthMonitor()
        self.assertIsNone(monitor._prepare_layers(_looped_model(4, with_block=False)))

    def test_degenerate_block_is_noop(self):
        # l_R == 0 parity config: adapter / state_norm are None on the real block.
        decoder = SimpleNamespace(state_norm=None, adapter=None, looped_num_recurrence=4)
        model = SimpleNamespace(decoder=decoder)
        monitor = LoopedHealthMonitor()
        self.assertIsNone(monitor._prepare_layers(model))

    def test_second_looped_block_refused(self):
        monitor = LoopedHealthMonitor()
        self.assertIsNotNone(monitor._prepare_layers(_looped_model(4)))
        # A second core would share the iteration counters; discovery must refuse it.
        self.assertIsNone(monitor._prepare_layers(_looped_model(4)))

    # ------------------------------------------------------------------
    # interval gating + counter behaviour
    # ------------------------------------------------------------------
    def test_interval_gate_suppresses_unmonitored_step(self):
        r = 2
        monitor = LoopedHealthMonitor(monitor_interval=2)
        block = _register(monitor, _looped_model(r, injection="add"))
        monitor.step_count = 1  # 1 % 2 != 0 -> not a monitored step
        states = [torch.randn(8, 2, 16) for _ in range(r)]
        _drive_forward(block, states)
        monitor.step(global_step=1)
        self.assertEqual(training_logs.get_latest(prefix="looped_health"), {})

    def test_counter_wraps_across_microbatches(self):
        # Two forwards in one step: iteration index must wrap 0..r-1 each forward,
        # and state_delta at iter 0 must never be recorded (no cross-forward diff).
        r = 3
        monitor = LoopedHealthMonitor()
        block = _register(monitor, _looped_model(r, injection="add"))
        for _ in range(2):  # two microbatches
            _drive_forward(block, [torch.randn(8, 2, 16) for _ in range(r)])
        monitor.step()
        latest = training_logs.get_latest(prefix="looped_health")
        self.assertNotIn("looped_health/layer_0/state_delta", latest)
        for i in range(r):
            self.assertIn(f"looped_health/layer_{i}/token_cosine", latest)
        # exactly r iteration slots, not 2*r.
        cos_keys = [k for k in latest if k.endswith("/token_cosine") and "/layer_" in k]
        self.assertEqual(len(cos_keys), r)

    def test_step_clears_state_stash(self):
        r = 2
        monitor = LoopedHealthMonitor()
        block = _register(monitor, _looped_model(r, injection="add"))
        _drive_forward(block, [torch.randn(8, 2, 16) for _ in range(r)])
        monitor.step()
        self.assertIsNone(monitor._prev_norm_state)

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------
    def test_setup_factory_registers_and_attaches(self):
        monitor_dict = {}
        model = _looped_model(4, injection="concat")
        setup_looped_monitor(model, monitor_dict=monitor_dict, verbose=True)
        self.assertIn("looped_health", monitor_dict)
        self.assertEqual(len(monitor_dict["looped_health"].hooks), 2)  # state + adapter

    def test_setup_factory_noop_on_plain_model(self):
        monitor_dict = {}
        model = _looped_model(4, with_block=False)
        # give the plain model a parameters() for the device probe path (unused here)
        model.parameters = model.decoder.layers[0].parameters
        setup_looped_monitor(model, monitor_dict=monitor_dict)
        self.assertIn("looped_health", monitor_dict)
        self.assertEqual(len(monitor_dict["looped_health"].hooks), 0)

    # ------------------------------------------------------------------
    # primitive
    # ------------------------------------------------------------------
    def test_covariance_effective_rank_orthonormal_frame(self):
        # 8 orthonormal token directions in H=8 -> C == I -> eff_rank == 8.
        state = torch.eye(8).reshape(8, 1, 8)
        er = _covariance_effective_rank(state)
        self.assertAlmostEqual(er.item(), 8.0, places=4)

    def test_covariance_effective_rank_all_zero_is_one(self):
        # eps on both sides makes a dead state read 1.0, never NaN.
        state = torch.zeros(8, 1, 8)
        self.assertAlmostEqual(_covariance_effective_rank(state).item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
