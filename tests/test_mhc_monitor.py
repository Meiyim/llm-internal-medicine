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

mhc_metrics = importlib.import_module("internal_medicine.backends.megatron.mhc_metrics")
mhc_monitor = importlib.import_module("internal_medicine.backends.megatron.mhc_monitor")
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs

MHCHealthMonitor = mhc_monitor.MHCHealthMonitor


class FakeHC(nn.Module):
    """Stand-in for HyperConnectionModule exposing compute_mappings / the bda + layer_number."""

    def __init__(self, n, layer_number, h_pre, h_post, h_res):
        super().__init__()
        self.n = n
        self.layer_number = layer_number
        self._h_pre = h_pre
        self._h_post = h_post
        self._h_res = h_res

    def compute_mappings(self, x):
        return self._h_pre, self._h_post, self._h_res

    def fused_h_res_h_post_bda(
        self, h_res, original_residual, h_post, layer_output_with_bias, dropout_prob, training, fused, manager=None
    ):
        """Reference ``out = h_res x + h_post (x) (o + bias)`` with the upstream signature."""
        s, b, nc = original_residual.shape
        x = original_residual.reshape(s, b, self.n, nc // self.n)
        o, bias = layer_output_with_bias
        if bias is not None:
            o = o + bias
        mixed = torch.einsum("sbij,sbjc->sbic", h_res, x)
        return (mixed + h_post.unsqueeze(-1) * o.unsqueeze(-2)).reshape(s, b, nc)

    def forward(self, hidden_states, mhc_recompute_manager=None):  # pragma: no cover - unused
        return hidden_states, self._h_res, self._h_post


class FakeLayer(nn.Module):
    def __init__(self, layer_number, attn, mlp):
        super().__init__()
        self.layer_number = layer_number
        self.self_attention_hyper_connection = attn
        self.mlp_hyper_connection = mlp


def _mhc_model(layers):
    return SimpleNamespace(decoder=SimpleNamespace(layers=nn.ModuleList(layers)))


ENERGY_NAMES = mhc_monitor._ENERGY_METRIC_NAMES


def _energy_split(h_res, x, o, h_post):
    """Build ``out = h_res x + h_post (x) o`` and return (metrics dict, per-token R/W/X)."""
    s, b, n, _ = x.shape
    mixed = torch.einsum("sbij,sbjc->sbic", h_res, x)
    w = h_post.unsqueeze(-1) * o.unsqueeze(-2)
    out = (mixed + w).reshape(s, b, -1)
    residual = x.reshape(s, b, -1)
    vals = mhc_metrics.residual_energy_split(residual, out, o, h_post, h_res, n)
    r = x.pow(2).sum(dim=(-2, -1))
    w_e = h_post.pow(2).sum(-1) * o.pow(2).sum(-1)
    cross = 2.0 * (mixed * w).sum(dim=(-2, -1))
    return dict(zip(ENERGY_NAMES, vals, strict=True)), (r, w_e, cross, out, mixed, w)


class MHCMetricsTest(unittest.TestCase):
    def test_amax_gain_row_vs_col(self):
        # [[1,2],[3,4]] -> row sums [3,7] -> max abs 7 (fwd); col sums [4,6] -> 6 (bwd).
        m = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.assertAlmostEqual(mhc_metrics.amax_gain(m, dim=-1).item(), 7.0, places=5)
        self.assertAlmostEqual(mhc_metrics.amax_gain(m, dim=-2).item(), 6.0, places=5)

    def test_doubly_stochastic_gain_is_one(self):
        n = 4
        ident = torch.eye(n).reshape(1, 1, n, n)
        self.assertAlmostEqual(mhc_metrics.amax_gain(ident, dim=-1).item(), 1.0, places=5)
        self.assertAlmostEqual(mhc_metrics.amax_gain(ident, dim=-2).item(), 1.0, places=5)

    def test_gate_stats(self):
        h = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
        mean, std = mhc_metrics.gate_stats(h)
        self.assertAlmostEqual(mean.item(), 1.5, places=5)
        self.assertAlmostEqual(std.item(), h.std().item(), places=6)

    def test_erase_beta_recovers_rank1_strength(self):
        # H_res = I - beta * u u^T with ||u|| = 1 -> tr = n - beta, per token.
        n = 4
        u = torch.tensor([0.9707, 0.1387, 0.1387, 0.1387])
        u = u / u.norm()
        betas = torch.tensor([0.25, 1.75])
        eye = torch.eye(n)
        mats = torch.stack([eye - b * torch.outer(u, u) for b in betas]).reshape(2, 1, n, n)
        mean, std = mhc_metrics.erase_beta_stats(mats)
        self.assertAlmostEqual(mean.item(), 1.0, places=5)
        self.assertAlmostEqual(std.item(), betas.std().item(), places=5)

    def test_erase_beta_zero_on_identity(self):
        n = 4
        ident = torch.eye(n).reshape(1, 1, n, n).expand(3, 2, n, n)
        mean, std = mhc_metrics.erase_beta_stats(ident)
        self.assertAlmostEqual(mean.item(), 0.0, places=6)
        self.assertAlmostEqual(std.item(), 0.0, places=6)

    def test_outer_deviation_zero_on_exact_outer(self):
        # h_res == h_post (x) h_pre (entry [i,j] = h_post_i * h_pre_j) -> 0.
        s, b, n = 3, 2, 4
        h_pre = torch.rand(s, b, n)
        h_post = torch.rand(s, b, n) * 2
        outer = h_post.unsqueeze(-1) * h_pre.unsqueeze(-2)
        self.assertAlmostEqual(mhc_metrics.outer_deviation(outer, h_pre, h_post).item(), 0.0, places=6)
        # Identity vs a zero outer product -> ||I||_F = sqrt(n).
        ident = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n)
        zeros = torch.zeros(s, b, n)
        self.assertAlmostEqual(mhc_metrics.outer_deviation(ident, zeros, zeros).item(), n**0.5, places=5)

    def test_outer_deviation_is_index_oriented(self):
        # Asymmetric gates: the transposed orientation must NOT read as 0.
        h_pre = torch.tensor([[[1.0, 0.0]]])
        h_post = torch.tensor([[[0.0, 2.0]]])
        outer = h_post.unsqueeze(-1) * h_pre.unsqueeze(-2)  # [[0,0],[2,0]]
        self.assertAlmostEqual(mhc_metrics.outer_deviation(outer, h_pre, h_post).item(), 0.0, places=6)
        self.assertAlmostEqual(
            mhc_metrics.outer_deviation(outer.transpose(-2, -1), h_pre, h_post).item(),
            (2 * 2.0**2) ** 0.5,
            places=5,
        )

    def test_orth_dev_mean_and_ratio_on_orthogonal(self):
        # Exactly orthogonal everywhere: mean 0 and the ratio floors at 1.0 (not 0/0).
        n, s, b = 4, 8, 4
        ident = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        mean, ratio = mhc_metrics.orthogonality_deviation(ident)
        self.assertAlmostEqual(mean.item(), 0.0, places=6)
        self.assertAlmostEqual(ratio.item(), 1.0, places=6)

    def test_orth_dev_ratio_exposes_tail_the_mean_hides(self):
        # 1 non-orthogonal token in 65536: the mean still displays as 0.0000 at the
        # log's 4 decimals while the ratio reports the tail. R3's Schulz failure mode.
        n, s, b = 4, 1 << 16, 1
        mats = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).clone()
        mats[0, 0] = torch.eye(n) * 2.0**0.5  # gram - I = I -> frob = sqrt(n) = 2
        mean, ratio = mhc_metrics.orthogonality_deviation(mats)
        self.assertLess(mean.item(), 5e-5, "token mean rounds to 0.0000 in the log")
        self.assertGreater(ratio.item(), 1e5)

    def test_orth_dev_ratio_recovers_the_true_ratio(self):
        # Deviations well above eps: the ratio is max/median, not eps-dominated.
        n, s, b = 2, 4, 1
        # gram - I = (c^2 - 1) I -> frob = sqrt(n) * |c^2 - 1|
        cs = [3.0, 3.0, 3.0, 5.0]  # devs ~ [11.3, 11.3, 11.3, 33.9]
        mats = torch.stack([torch.eye(n) * c for c in cs]).reshape(s, b, n, n)
        _, ratio = mhc_metrics.orthogonality_deviation(mats)
        self.assertAlmostEqual(ratio.item(), 24.0 / 8.0, places=3)

    def test_stream_gram_on_an_orthogonal_frame(self):
        # 4 orthogonal streams with norms 1..4: per-stream norms recovered exactly, the
        # imbalance shows up in the ratio (a stream-axis mean would report a flat 2.5).
        n, c = 4, 4
        streams = torch.eye(c) * torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(-1)
        x = streams.reshape(1, 1, n * c).expand(3, 2, n * c).contiguous()
        norms, ratio, off_mean, off_max = mhc_metrics.stream_gram_stats(x, n)
        self.assertEqual(tuple(norms.shape), (n,))
        for i, expected in enumerate((1.0, 2.0, 3.0, 4.0)):
            self.assertAlmostEqual(norms[i].item(), expected, places=5)
        self.assertAlmostEqual(ratio.item(), 4.0, places=5)
        self.assertAlmostEqual(off_mean.item(), 0.0, places=6)
        self.assertAlmostEqual(off_max.item(), 0.0, places=6)

    def test_stream_gram_collapse_is_unit_cosine(self):
        # All n streams on one direction: |cosine| == 1 off-diagonal, norms still balanced.
        n, c = 4, 8
        one = torch.randn(c)
        x = one.repeat(n).reshape(1, 1, n * c).expand(5, 2, n * c).contiguous()
        norms, ratio, off_mean, off_max = mhc_metrics.stream_gram_stats(x, n)
        self.assertAlmostEqual(ratio.item(), 1.0, places=5)
        self.assertAlmostEqual(off_mean.item(), 1.0, places=5)
        self.assertAlmostEqual(off_max.item(), 1.0, places=5)
        self.assertAlmostEqual(norms[0].item(), one.norm().item(), places=4)

    def test_stream_gram_offdiag_max_exposes_tail_the_mean_hides(self):
        # Token 0 orthogonal, token 1 has two parallel streams: mean 1/6, per-token max 1/2.
        n, c = 3, 3
        tok0 = torch.eye(n)
        tok1 = torch.stack([torch.eye(n)[0], torch.eye(n)[0], torch.eye(n)[2]])
        x = torch.stack([tok0.reshape(-1), tok1.reshape(-1)]).reshape(2, 1, n * c)
        _, ratio, off_mean, off_max = mhc_metrics.stream_gram_stats(x, n)
        self.assertAlmostEqual(off_mean.item(), 1.0 / 6.0, places=5)
        self.assertAlmostEqual(off_max.item(), 0.5, places=5)
        self.assertAlmostEqual(ratio.item(), 1.0, places=5)

    def test_stream_gram_survives_a_dead_stream(self):
        # A zero stream must not produce nan/inf through the norm or cosine denominators.
        n, c = 4, 6
        streams = torch.randn(n, c)
        streams[2] = 0.0
        x = streams.reshape(1, 1, n * c).expand(4, 2, n * c).contiguous()
        for t in mhc_metrics.stream_gram_stats(x, n):
            self.assertTrue(torch.isfinite(t).all(), t)

    def test_energy_split_is_the_exact_identity(self):
        # ||out||^2 = R + W + X for an orthogonal h_res; every series must match the
        # naive C-length computation it was algebraically folded out of.
        torch.manual_seed(0)
        n, c, s, b = 4, 6, 3, 2
        q, _ = torch.linalg.qr(torch.randn(n, n))
        h_res = q.reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        x = torch.randn(s, b, n, c)
        o = torch.randn(s, b, c)
        h_post = torch.rand(s, b, n) * 2.0

        m, (r, w_e, cross, out, _, w) = _energy_split(h_res, x, o, h_post)
        self.assertAlmostEqual(m["write_over_resid"].item(), (w_e / r).mean().item(), places=5)
        self.assertAlmostEqual(m["cross_over_resid"].item(), (cross / r).mean().item(), places=5)
        self.assertAlmostEqual(m["cross_over_write"].item(), (cross / w_e).mean().item(), places=4)
        # the accounting self-check: measured Out/R == 1 + W/R + X/R
        self.assertAlmostEqual(m["resid_gain"].item(), (out.pow(2).sum(-1) / r).mean().item(), places=5)
        self.assertAlmostEqual(m["resid_gain"].item(), ((r + w_e + cross) / r).mean().item(), places=5)

        cos = cross / (2.0 * (r * w_e).sqrt())
        self.assertAlmostEqual(m["mix_write_cos"].item(), cos.mean().item(), places=5)
        self.assertAlmostEqual(m["mix_write_cos_abs_max"].item(), cos.abs().amax().item(), places=5)
        pre = (x * w).sum(dim=(-2, -1)) / (r * w_e).sqrt()
        self.assertAlmostEqual(m["resid_write_cos"].item(), pre.mean().item(), places=5)

    def test_energy_split_write_orthogonal_to_every_stream(self):
        # o outside span{x_j} -> d = 0 -> X = 0 and the energy is purely additive.
        torch.manual_seed(1)
        n, c, s, b = 2, 4, 3, 2
        q, _ = torch.linalg.qr(torch.randn(n, n))
        h_res = q.reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        basis = torch.eye(c)
        x = torch.stack([basis[0], basis[1]]).reshape(1, 1, n, c).expand(s, b, n, c) * 3.0
        o = basis[2].reshape(1, 1, c).expand(s, b, c) * 2.0
        h_post = torch.ones(s, b, n)

        m, (r, w_e, _, _, _, _) = _energy_split(h_res, x.contiguous(), o.contiguous(), h_post)
        self.assertAlmostEqual(m["cross_over_resid"].item(), 0.0, places=6)
        self.assertAlmostEqual(m["mix_write_cos"].item(), 0.0, places=6)
        self.assertAlmostEqual(m["resid_write_cos"].item(), 0.0, places=6)
        self.assertAlmostEqual(m["resid_gain"].item(), (1.0 + w_e / r).mean().item(), places=5)

    def test_energy_split_catches_a_shrinking_residual(self):
        # The R5 attn3->mlp3 signature: an anti-aligned write makes Out < R, which is
        # impossible from W alone. h_res = I, o = -k x_i -> X = -2 k ||v||^2 sum(h_post).
        n, c, s, b = 2, 5, 2, 2
        k = 0.5
        v = torch.randn(c)
        x = v.reshape(1, 1, 1, c).expand(s, b, n, c).contiguous()
        o = (-k * v).reshape(1, 1, c).expand(s, b, c).contiguous()
        h_res = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        h_post = torch.ones(s, b, n)

        m, (r, _, _, out, _, _) = _energy_split(h_res, x, o, h_post)
        self.assertLess(out.pow(2).sum(-1).mean().item(), r.mean().item())
        self.assertLess(m["resid_gain"].item(), 1.0)
        self.assertAlmostEqual(m["cross_over_write"].item(), -2.0 / k, places=4)
        self.assertAlmostEqual(m["mix_write_cos"].item(), -1.0, places=4)

    def test_energy_split_non_orthogonal_mix_loses_energy(self):
        # R stands in for ||h_res x||^2; for a doubly-stochastic (non-orthogonal) mix the
        # measured gain must sit strictly below the 1 + W/R + X/R the identity predicts.
        torch.manual_seed(2)
        n, c, s, b = 4, 6, 3, 2
        h_res = torch.full((s, b, n, n), 1.0 / n)
        x = torch.randn(s, b, n, c)
        o = torch.randn(s, b, c)
        h_post = torch.rand(s, b, n) * 2.0

        m, _ = _energy_split(h_res, x, o, h_post)
        predicted = 1.0 + m["write_over_resid"].item() + m["cross_over_resid"].item()
        self.assertLess(m["resid_gain"].item(), predicted)

    def test_energy_split_survives_a_vanishing_write(self):
        # W -> 0 makes X/W and cos ill-posed; the relative floor must keep them finite.
        n, c, s, b = 4, 6, 2, 2
        x = torch.randn(s, b, n, c)
        o = torch.zeros(s, b, c)
        h_res = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        h_post = torch.ones(s, b, n)

        m, _ = _energy_split(h_res, x, o, h_post)
        for name, val in m.items():
            self.assertTrue(torch.isfinite(val).all(), f"{name} = {val}")
        self.assertAlmostEqual(m["resid_gain"].item(), 1.0, places=6)


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

    def _identity_layer(self, n, s, b, layer_number=1):
        ident = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()

        def make_hc():
            return FakeHC(
                n=n,
                layer_number=layer_number,
                h_pre=torch.full((s, b, n), 0.5),
                h_post=torch.ones(s, b, n),
                h_res=ident,
            )

        return FakeLayer(layer_number=layer_number, attn=make_hc(), mlp=make_hc())

    def _drive(self, targets, x_dim):
        # Fire each wrapped compute_mappings (grad enabled -> _should_monitor passes).
        for _, _, mod, _, _ in targets:
            mod.compute_mappings(torch.randn(4, 2, x_dim))

    def test_identity_composite_stays_unit_gain(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b, layer_number=1)
        model = _mhc_model([layer])

        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = monitor._prepare_layers(model, chunk_id=0)
        self.assertTrue(targets)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)

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
                "h_res_orth_dev",
                "composite_h_res_orth_dev",
                "h_res_orth_dev_max_med_ratio",
                "composite_h_res_orth_dev_max_med_ratio",
                "h_res_beta_mean",
                "h_res_beta_std",
                "h_res_outer_dev",
                "stream_norm_max_min_ratio",
                "stream_gram_offdiag_mean",
                "stream_gram_offdiag_max",
                *(f"stream_norm_{i}" for i in range(n)),
            ):
                self.assertIn(f"mhc_health/layer_0/{comp}_{key}", latest)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_amax_gain_fwd"], 1.0, places=4)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_composite_amax_gain_fwd"], 1.0, places=4)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_composite_amax_gain_bwd"], 1.0, places=4)
            # h_pre == 0.5 everywhere, h_post == 1.0 everywhere.
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_pre_mean"], 0.5, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_post_mean"], 1.0, places=5)
            # h_res == I -> orthogonal, zero trace deficit; ||I - 0.5*J||_F == 2.0.
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_orth_dev"], 0.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_orth_dev_max_med_ratio"], 1.0, places=5)
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_composite_h_res_orth_dev_max_med_ratio"], 1.0, places=5
            )
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_beta_mean"], 0.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_beta_std"], 0.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_outer_dev"], 2.0, places=5)
        # global aggregate is derived too.
        self.assertIn("mhc_health/global_attn_amax_gain_fwd", latest)
        # the ratios are max-aggregated end to end: monitor buffer + training_logs suffix.
        self.assertIn("attn_h_res_orth_dev_max_med_ratio", MHCHealthMonitor.MAX_AGGREGATED)
        self.assertTrue(training_logs._is_max_metric("mhc_health/layer_0/attn_h_res_orth_dev_max_med_ratio"))

    def test_stream_geometry_comes_from_the_hook_input(self):
        # The stream series are computed from compute_mappings' own argument, not from the
        # returned mappings — feed a known frame and read the exact values back out.
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b, layer_number=1)
        model = _mhc_model([layer])

        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)

        c = 4
        streams = torch.eye(c) * torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(-1)
        x = streams.reshape(1, 1, n * c).expand(4, 2, n * c).contiguous()
        for _, _, mod, _, _ in targets:
            mod.compute_mappings(x)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for i, expected in enumerate((1.0, 2.0, 3.0, 4.0)):
                self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_stream_norm_{i}"], expected, places=4)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_stream_norm_max_min_ratio"], 4.0, places=4)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_stream_gram_offdiag_mean"], 0.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_stream_gram_offdiag_max"], 0.0, places=5)
        # the imbalance ratio and the Gram tail are max-aggregated end to end.
        self.assertIn("attn_stream_norm_max_min_ratio", MHCHealthMonitor.MAX_AGGREGATED)
        self.assertIn("attn_stream_gram_offdiag_max", MHCHealthMonitor.MAX_AGGREGATED)
        self.assertTrue(training_logs._is_max_metric("mhc_health/layer_0/attn_stream_norm_max_min_ratio"))
        self.assertTrue(training_logs._is_max_metric("mhc_health/layer_0/attn_stream_gram_offdiag_max"))

    def _bda_case(self, targets, n, c, s, b, bias=None):
        """Drive every wrapped bda with a known update; return the expected mean Out/R."""
        x = torch.randn(s, b, n, c)
        o = torch.randn(s, b, c)
        h_res = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        h_post = torch.ones(s, b, n)
        for _, _, mod, _, _ in targets:
            mod.fused_h_res_h_post_bda(h_res, x.reshape(s, b, n * c), h_post, (o, bias), 0.0, True, False)
        o_eff = o if bias is None else o + bias
        r = x.pow(2).sum(dim=(-2, -1))
        w_e = h_post.pow(2).sum(-1) * o_eff.pow(2).sum(-1)
        cross = 2.0 * (x * (h_post.unsqueeze(-1) * o_eff.unsqueeze(-2))).sum(dim=(-2, -1))
        return ((r + w_e + cross) / r).mean().item()

    def test_energy_split_series_reach_the_logs(self):
        n, c, s, b = 4, 6, 3, 2
        torch.manual_seed(3)
        layer = self._identity_layer(n, s, b, layer_number=1)
        model = _mhc_model([layer])

        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        expected_gain = self._bda_case(targets, n, c, s, b)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for key in ENERGY_NAMES:
                self.assertIn(f"mhc_health/layer_0/{comp}_{key}", latest)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_resid_gain"], expected_gain, places=4)
        # the cos tail detector is max-aggregated end to end (the _max suffix is enough)
        self.assertIn("attn_mix_write_cos_abs_max", MHCHealthMonitor.MAX_AGGREGATED)
        self.assertTrue(training_logs._is_max_metric("mhc_health/layer_0/attn_mix_write_cos_abs_max"))

    def test_energy_split_counts_the_bias_in_the_write(self):
        n, c, s, b = 4, 6, 3, 2
        torch.manual_seed(4)
        layer = self._identity_layer(n, s, b, layer_number=1)
        model = _mhc_model([layer])

        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        expected_gain = self._bda_case(targets, n, c, s, b, bias=torch.randn(c))
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertAlmostEqual(latest["mhc_health/layer_0/attn_resid_gain"], expected_gain, places=4)

    def test_remove_hooks_restores_compute_mappings(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = MHCHealthMonitor()
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        hc = layer.self_attention_hyper_connection
        originals = {a: getattr(hc, a) for a in ("compute_mappings", "fused_h_res_h_post_bda")}
        monitor._attach_hooks(targets)
        for attr, original in originals.items():
            self.assertIsNot(getattr(hc, attr), original)
        monitor.remove_hooks()
        # both fall back to the (bound) class methods
        self.assertEqual(hc.compute_mappings.__func__, FakeHC.compute_mappings)
        self.assertEqual(hc.fused_h_res_h_post_bda.__func__, FakeHC.fused_h_res_h_post_bda)
        self.assertEqual(monitor._wrapped, [])
        self.assertEqual(monitor._composite, {})

    def test_no_graph_retention(self):
        n, s, b = 4, 2, 3
        ident = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        # Outputs attached to a graph (require grad) — the wrapper must detach.
        leaf = torch.zeros(s, b, n, requires_grad=True)
        h_pre = leaf + 0.5
        h_post = leaf + 1.0
        h_res = ident * (leaf.sum() + 1.0)  # requires_grad, has grad_fn
        hc = FakeHC(n=n, layer_number=1, h_pre=h_pre, h_post=h_post, h_res=h_res)
        layer = FakeLayer(layer_number=1, attn=hc, mlp=hc)
        model = _mhc_model([layer])

        monitor = MHCHealthMonitor()
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        for _, _, mod, _, _ in targets:
            mod.compute_mappings(torch.randn(4, 2, n * 8))

        stored = monitor._composite[0]
        self.assertFalse(stored.requires_grad)
        self.assertIsNone(stored.grad_fn)
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
