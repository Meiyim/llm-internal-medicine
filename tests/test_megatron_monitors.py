import importlib
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

importlib.import_module("_backend_env").skip_unless_backend("megatron")

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    F = importlib.import_module("torch.nn.functional")
except Exception as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"torch backend unavailable: {exc}") from exc

MassiveActivationMonitor = importlib.import_module(
    "internal_medicine.backends.megatron.massive_activation_monitor"
).MassiveActivationMonitor
MoESpecialistMonitor = importlib.import_module("internal_medicine.backends.megatron.moe_monitor").MoESpecialistMonitor
moe_monitor_module = importlib.import_module("internal_medicine.backends.megatron.moe_monitor")
PLEHealthMonitor = importlib.import_module("internal_medicine.backends.megatron.ple_monitor").PLEHealthMonitor
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs
massive_activation_metrics = importlib.import_module("internal_medicine.backends.megatron.massive_activation_metrics")
compute_sink_head_classification = importlib.import_module(
    "internal_medicine.backends.megatron.sink_head_metrics"
).compute_sink_head_classification


class FakePLESublayer:
    act_fn = F.gelu


class FakeMoELayer(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts
        self.shared_experts = None


class MegatronMoEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_router_metrics_flush_from_gpu_buffer(self):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        for name in moe_monitor_module._ROUTER_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        router = SimpleNamespace(
            topk=2,
            _cached_scores_for_aux_loss=torch.tensor(
                [
                    [0.7, 0.2, 0.1],
                    [0.1, 0.6, 0.3],
                ],
                dtype=torch.float32,
            ),
        )
        monitor._compute_router_metrics(0, router, None, None)

        monitor.step()
        latest = training_logs.get_latest(prefix="moe_health")
        self.assertIn("moe_health/layer_0/router_entropy", latest)
        self.assertIn("moe_health/layer_0/score_sum_mean", latest)
        self.assertIn("moe_health/global_router_entropy", latest)
        self.assertIn("moe_health/global_score_sum_max", latest)

    def test_step_computes_expert_metrics_even_under_no_grad(self):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        for name in moe_monitor_module._EXPERT_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        hidden_size = 4
        ffn_hidden = 8
        num_experts = 2
        experts = SimpleNamespace(
            num_local_experts=num_experts,
            config=SimpleNamespace(hidden_size=hidden_size),
            weight1=torch.nn.Parameter(torch.ones(num_experts * hidden_size, ffn_hidden)),
            weight2=torch.nn.Parameter(torch.ones(num_experts * ffn_hidden, hidden_size)),
        )
        moe_layer = FakeMoELayer(experts)
        monitor._monitored_moe_layers = [(0, weakref.ref(moe_layer))]

        with torch.no_grad():
            monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertIn("moe_health/layer_0/expert_norm_mean", latest)
        self.assertIn("moe_health/global_expert_norm_mean", latest)

    def test_load_balance_metrics_from_reduced_tokens_per_expert(self):
        # Two bias-enabled layers, recorded in the order finalize_model_grads
        # stacks them. tokens_per_expert is the ALREADY-reduced global count.
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._expert_bias_enabled = True
        monitor._load_balance_layer_order = [0, 1]
        for layer_idx in (0, 1):
            for name in moe_monitor_module._LOAD_BALANCE_METRICS:
                monitor.declare_layer_metric(layer_idx, name)
        monitor.allocate_buffers(torch.device("cpu"))

        # layer 0: [5,3,2,2] -> max/min=5/2=2.5, max/median=5/2=2.5
        # layer 1: [10,0,4,6] -> max/min=10/1(clamp)=10, max/median=10/4=2.5
        reduced = torch.tensor([[5.0, 3.0, 2.0, 2.0], [10.0, 0.0, 4.0, 6.0]])
        monitor._record_load_balance_metrics(reduced)
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(latest["moe_health/layer_0/load_max_min_ratio"], 2.5, places=5)
        self.assertAlmostEqual(latest["moe_health/layer_1/load_max_min_ratio"], 10.0, places=5)
        self.assertAlmostEqual(latest["moe_health/layer_0/load_max_median_ratio"], 2.5, places=5)
        self.assertAlmostEqual(latest["moe_health/layer_1/load_max_median_ratio"], 2.5, places=5)
        # CV = population std / mean. layer 0 mean=3, std=sqrt(((2)+(0)+(1)+(1))/4)=sqrt(1.5)
        self.assertAlmostEqual(latest["moe_health/layer_0/load_cv"], (1.5**0.5) / 3.0, places=5)
        self.assertIn("moe_health/global_load_max_min_ratio", latest)
        self.assertIn("moe_health/global_load_cv", latest)

    def test_load_balance_metrics_absent_without_expert_bias(self):
        # When no router has expert-bias enabled, the metric must not be
        # declared (schema stays clean) — nothing to record or patch.
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        self.assertFalse(monitor._expert_bias_enabled)
        self.assertEqual(monitor._load_balance_layer_order, [])
        # Idempotent no-op when expert-bias is off.
        monitor._patch_expert_bias_update()
        self.assertIsNone(monitor._orig_get_updated_expert_bias)

    def test_expert_bias_patch_rebinds_caller_and_fires(self):
        # The wrapper must rebind the name in finalize_model_grads (the caller),
        # observe the reduced tokens_per_expert, and unpatch cleanly.
        fmg = moe_monitor_module._finalize_model_grads
        original = fmg.get_updated_expert_bias
        # Stub the underlying update so the test needs no distributed group:
        # the real fn all-reduces tokens_per_expert, which requires dist init.
        # Our wrapper only cares that the tensor it receives is the reduced one.
        fmg.get_updated_expert_bias = lambda tpe, bias, rate, *a, **k: bias

        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._expert_bias_enabled = True
        monitor._load_balance_layer_order = [0]
        for name in moe_monitor_module._LOAD_BALANCE_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        try:
            monitor._patch_expert_bias_update()
            self.assertTrue(getattr(fmg.get_updated_expert_bias, "_im_patched", False))

            # Caller hands the wrapper an ALREADY-reduced count row.
            tokens = torch.tensor([[6.0, 2.0, 3.0, 3.0]])  # max/min=6/2=3, max/median=6/3=2
            bias = torch.zeros_like(tokens)
            fmg.get_updated_expert_bias(tokens, bias, 0.0)
            monitor.step()

            latest = training_logs.get_latest(prefix="moe_health")
            self.assertAlmostEqual(latest["moe_health/layer_0/load_max_min_ratio"], 3.0, places=5)
            self.assertAlmostEqual(latest["moe_health/layer_0/load_max_median_ratio"], 2.0, places=5)
        finally:
            monitor.remove_hooks()
            fmg.get_updated_expert_bias = original


class MegatronPLEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_global_hooks_are_disabled_when_log_global_is_false(self):
        monitor = PLEHealthMonitor(log_global=False)
        monitor._num_layers = 2
        monitor._hidden_size = 6
        monitor._hidden_size_ple = 3

        monitor._make_token_ple_hook()(None, None, torch.randn(2, 4, 6))
        monitor._make_proj_ple_hook()(None, None, torch.randn(2, 4, 6))

        self.assertIsNone(monitor._token_ple_buf)
        self.assertIsNone(monitor._proj_ple_buf)
        self.assertEqual(training_logs.get_latest(prefix="ple_health"), {})

    def test_layer_hook_records_residual_and_gate_metrics_as_one_observation(self):
        monitor = PLEHealthMonitor(log_per_layer=True, log_global=True, gate_sparsity_threshold=0.01)
        hidden_states = torch.ones(2, 3, 4)
        for name in ("residual_ratio", "gate_activation_mean", "gate_sparsity"):
            monitor.declare_layer_metric(5, name)
        monitor.allocate_buffers(hidden_states.device)

        monitor._gate_out_buf[5] = torch.ones(2, 3, 4)
        output = hidden_states * 1.5

        hook = monitor._make_ple_layer_hook(5, FakePLESublayer())
        hook(None, (hidden_states,), output)
        monitor.step()

        latest = training_logs.get_latest(prefix="ple_health")
        self.assertIn("ple_health/layer_5/residual_ratio", latest)
        self.assertIn("ple_health/layer_5/gate_activation_mean", latest)
        self.assertIn("ple_health/global_residual_ratio", latest)
        self.assertEqual(monitor._gate_out_buf, {})


class MegatronMassiveActivationMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_compute_and_log_records_pre_norm_metrics(self):
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            cosine_sample_pairs=4,
            absolute_thresholds=(2.0, 3.0),
        )
        hidden_states = torch.tensor(
            [
                [[1.0, -2.0, 0.5, 4.0]],
                [[3.0, 1.0, -0.5, 2.0]],
            ]
        )
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(hidden_states.device)

        monitor._compute_residual_metrics(0, hidden_states)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for key in (
            "channel_max",
            "channel_median",
            "channel_p95",
            "channel_p99",
            "channel_max_ratio",
            "massive_act_channel_count",
            "channel_count_gt_2",
            "channel_count_gt_3",
            "topk_channel_norm",
        ):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_2"], 2.0)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_3"], 1.0)

    def test_spectral_norm_bounds_record_per_token_rms_ratio(self):
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            log_post_norm_metrics=False,
        )
        pre = torch.tensor(
            [
                [[1.0, -2.0, 0.5, 4.0]],
                [[3.0, 1.0, -0.5, 2.0]],
            ]
        )
        # post = 2 * pre => per-token RMS ratio is exactly 2.0 for every token,
        # so both the max and min bound collapse to 2.0.
        post = pre * 2.0
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(pre.device)

        monitor._compute_spectral_norm(0, pre, post)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for key in ("spectral_norm_max", "spectral_norm_min"):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertAlmostEqual(latest["massive_act/layer_0/spectral_norm_max"], 2.0, places=5)
        self.assertAlmostEqual(latest["massive_act/layer_0/spectral_norm_min"], 2.0, places=5)
        # activation_rms is derived from the shared per-token pre-RMS, not a second
        # square of the input: it must equal sqrt(mean(pre**2)) over the whole input.
        self.assertIn("massive_act/layer_0/activation_rms", latest)
        expected_rms = pre.reshape(-1, pre.shape[-1]).float().square().mean().sqrt().item()
        self.assertAlmostEqual(latest["massive_act/layer_0/activation_rms"], expected_rms, places=5)

    def test_derived_activation_rms_matches_original_formula(self):
        # Regression guard: the merged/derived activation_rms (from the spectral
        # hook's per-token pre-RMS) must be numerically identical to the original
        # standalone compute_activation_scale_stats(h) over the same tensor.
        torch.manual_seed(0)
        hidden_states = torch.randn(7, 3, 5) * 4.0 + 1.5  # [S, B, H], non-trivial scale

        original = massive_activation_metrics.compute_activation_scale_stats(hidden_states)["activation_rms"]
        # activation_rms depends only on the pre tensor; post is irrelevant to it.
        derived = massive_activation_metrics.compute_spectral_norm_bounds(
            hidden_states, torch.zeros_like(hidden_states), include_activation_rms=True
        )["activation_rms"]

        self.assertTrue(torch.allclose(original, derived, rtol=0, atol=1e-6), f"{original} != {derived}")

    def test_grad_gain_bounds_equal_scaled_gradients(self):
        # grad_in = 2 * grad_out per token => the per-token ratio ‖dx‖/‖dy‖ is exactly
        # 2.0 for every token, so both the Lipschitz max and min bound collapse to 2.0.
        torch.manual_seed(0)
        grad_out = torch.randn(5, 3, 4)  # [S, B, H]
        grad_in = grad_out * 2.0

        bounds = massive_activation_metrics.compute_grad_gain_bounds(grad_in, grad_out)

        self.assertAlmostEqual(bounds["lipschitz_max"].item(), 2.0, places=5)
        self.assertAlmostEqual(bounds["lipschitz_min"].item(), 2.0, places=5)

    def test_grad_gain_hook_records_lipschitz_from_backward(self):
        # End-to-end: a scalar-mul layer y = 2*x has ∂L/∂x = 2·∂L/∂y for any loss, so
        # the backward-captured gradient-gain ratio is exactly 2.0 regardless of the
        # gradient direction. This also proves the tensor grad-hook path fires and
        # records (a module full_backward_hook would see an empty grad_input here,
        # since the layer is called all-keyword like Megatron does).
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=True,
        )
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        class ScalarMul(nn.Module):
            def forward(self, hidden_states):
                return hidden_states * 2.0, None  # (output, context) like a Megatron layer

        layer = ScalarMul()
        hook = layer.register_forward_hook(monitor._make_grad_gain_hook(0), with_kwargs=True)

        x = torch.randn(3, 2, 4, requires_grad=True)  # [S, B, H]
        out, _ = layer(hidden_states=x)  # all-keyword call, as the Megatron block does
        out.sum().backward()
        hook.remove()

        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for key in ("lipschitz_max", "lipschitz_min"):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertAlmostEqual(latest["massive_act/layer_0/lipschitz_max"], 2.0, places=5)
        self.assertAlmostEqual(latest["massive_act/layer_0/lipschitz_min"], 2.0, places=5)


class SinkHeadClassificationTest(unittest.TestCase):
    """The gap computation is branchless to avoid a GPU->CPU sync on the hot
    path (Python comparisons on a tensor sink_count would .item()). These cases
    pin the branchless result against a readable branched reference.
    """

    THRESHOLD = 0.3

    def _reference_gap(self, sink_per_head):
        is_sink = sink_per_head > self.THRESHOLD
        num_heads = sink_per_head.numel()
        sink_count = int(is_sink.sum())
        if 0 < sink_count < num_heads:
            return (sink_per_head[is_sink].mean() - sink_per_head[~is_sink].mean()).item()
        if sink_count == num_heads:
            return sink_per_head.mean().item()
        return 0.0

    def _assert_gap(self, sink_per_head):
        result = compute_sink_head_classification(sink_per_head, threshold=self.THRESHOLD)
        self.assertAlmostEqual(result["sink_nonsink_gap"].item(), self._reference_gap(sink_per_head), places=5)

    def test_mixed_sink_and_nonsink(self):
        self._assert_gap(torch.tensor([0.5, 0.1, 0.8, 0.05]))

    def test_all_heads_are_sinks(self):
        self._assert_gap(torch.tensor([0.5, 0.6, 0.9]))

    def test_no_sinks(self):
        self._assert_gap(torch.tensor([0.1, 0.2, 0.05]))

    def test_single_head(self):
        self._assert_gap(torch.tensor([0.9]))
        self._assert_gap(torch.tensor([0.1]))

    def test_empty_input_is_zero(self):
        result = compute_sink_head_classification(torch.tensor([]), threshold=self.THRESHOLD)
        self.assertEqual(result["sink_nonsink_gap"].item(), 0.0)
        self.assertEqual(result["sink_head_ratio"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
