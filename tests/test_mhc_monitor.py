import importlib
import math
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


def _canonical_so4(alpha, gamma, dtype=torch.float64):
    """Canonical SO(4): rotation by ``alpha`` on span(e0, e1), by ``gamma`` on span(e2, e3)."""

    def rot(t):
        c, s = math.cos(t), math.sin(t)
        return torch.tensor([[c, -s], [s, c]], dtype=dtype)

    m = torch.zeros(4, 4, dtype=dtype)
    m[:2, :2] = rot(alpha)
    m[2:, 2:] = rot(gamma)
    return m


def _conjugate_so4(m, seed):
    """Random ``P m P^T``, ``P`` in SO(4) — same conjugacy class, so the same angle pair."""
    torch.manual_seed(seed)
    p, _ = torch.linalg.qr(torch.randn(4, 4, dtype=m.dtype))
    if torch.linalg.det(p) < 0:
        p[:, 0] = -p[:, 0]
    return p @ m @ p.T


def _quat_pair_so4(q, r):
    """``L_q R_rbar``, i.e. ``x -> q x rbar`` — the R9 parameterization, built the explicit way.

    The training repo's patch never forms these two factors (that is the point of its constant
    GEMM); here they keep the fixture independent of it.
    """
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    left = torch.tensor([[qw, -qx, -qy, -qz], [qx, qw, -qz, qy], [qy, qz, qw, -qx], [qz, -qy, qx, qw]], dtype=q.dtype)
    right = torch.tensor([[rw, rx, ry, rz], [-rx, rw, -rz, ry], [-ry, rz, rw, -rx], [-rz, -ry, rx, rw]], dtype=r.dtype)
    return left @ right


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

    def test_sym3_eigvals_match_linalg(self):
        # The closed form replaces eigvalsh only to avoid its host sync; it must agree with it.
        torch.manual_seed(0)
        a = torch.randn(4096, 3, 3, dtype=torch.float64)
        a = a + a.transpose(-2, -1)
        got = mhc_metrics._eigvalsh_sym3(a)
        want = torch.linalg.eigvalsh(a)  # ascending
        self.assertLess((got - want).abs().amax().item(), 1e-10)

    def test_sigma_stats_recovers_a_planted_spectrum(self):
        # H = I + Q (diag(sigma) - I) Q^T with Q an orthonormal basis of 1^perp: the singular
        # values on 1^perp ARE sigma, and this is how R8's H_res is built.
        n, s, b = 4, 3, 2
        q = mhc_metrics._complement_basis(n, torch.device("cpu"), torch.float64)
        sig = torch.tensor([0.2, 0.55, 0.9], dtype=torch.float64)
        eye_m = torch.eye(n - 1, dtype=torch.float64)
        h = torch.eye(n, dtype=torch.float64) + q @ (torch.diag(sig) - eye_m) @ q.T
        mats = h.reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        s_min, s_mean = mhc_metrics.sigma_stats(mats)
        self.assertAlmostEqual(s_min.item(), 0.2, places=8)
        self.assertAlmostEqual(s_mean.item(), sig.mean().item(), places=8)

    def test_sigma_stats_is_flat_one_on_a_mean_preserving_isometry(self):
        # R8b's contract, and the isotropic (p2 == 0) branch of the closed form: exactly 1.0,
        # not 1 +- the eigensolver's degeneracy floor.
        n, s, b = 4, 8, 4
        ident = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        s_min, s_mean = mhc_metrics.sigma_stats(ident)
        self.assertAlmostEqual(s_min.item(), 1.0, places=6)
        self.assertAlmostEqual(s_mean.item(), 1.0, places=6)

    def test_sigma_stats_reads_zero_on_the_stream_mean_collapse(self):
        # H_res = J replaces every stream by the stream mean: sigma == 0, the R8a alarm.
        n, s, b = 4, 2, 3
        j = torch.full((n, n), 1.0 / n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        s_min, s_mean = mhc_metrics.sigma_stats(j)
        self.assertAlmostEqual(s_min.item(), 0.0, places=6)
        self.assertAlmostEqual(s_mean.item(), 0.0, places=6)

    def test_sigma_min_sees_one_collapsed_direction_the_mean_hides(self):
        # The amax_gain lesson: 1 of n-1 directions gone still reads a healthy 0.67 mean.
        n = 4
        q = mhc_metrics._complement_basis(n, torch.device("cpu"), torch.float32)
        sig = torch.tensor([0.0, 1.0, 1.0])
        h = torch.eye(n) + q @ (torch.diag(sig) - torch.eye(n - 1)) @ q.T
        s_min, s_mean = mhc_metrics.sigma_stats(h.reshape(1, 1, n, n))
        self.assertAlmostEqual(s_min.item(), 0.0, places=5)
        self.assertAlmostEqual(s_mean.item(), 2.0 / 3.0, places=5)

    def test_so4_angles_recover_a_planted_conjugated_pair(self):
        # The angle pair IS the conjugacy class, so a random SO(4) conjugation must not move it.
        for alpha, gamma in ((0.4, 1.9), (1.0, 2.0), (0.15, 0.9), (2.5, 3.0)):
            m = _conjugate_so4(_canonical_so4(alpha, gamma), seed=7)
            lo, hi = mhc_metrics.so4_angle_stats(m.reshape(1, 1, 4, 4))
            self.assertAlmostEqual(lo.item(), min(alpha, gamma), places=7)
            self.assertAlmostEqual(hi.item(), max(alpha, gamma), places=7)

    def test_so4_angles_match_the_spectrum_on_haar_samples(self):
        # Cross-check the closed form against eigvals (which a hook may not call) over the group.
        torch.manual_seed(3)
        for _ in range(20):
            p, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
            if torch.linalg.det(p) < 0:
                p[:, 0] = -p[:, 0]
            want = sorted({round(abs(torch.angle(e).item()), 9) for e in torch.linalg.eigvals(p)})
            lo, hi = mhc_metrics.so4_angle_stats(p.reshape(1, 1, 4, 4))
            self.assertAlmostEqual(lo.item(), want[0], places=7)
            self.assertAlmostEqual(hi.item(), want[-1], places=7)

    def test_so4_angles_on_the_three_boundary_points(self):
        eye = torch.eye(4, dtype=torch.float64)
        for m, want in (
            (eye, (0.0, 0.0)),
            (torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64)), (0.0, math.pi)),
            (-eye, (math.pi, math.pi)),
        ):
            lo, hi = mhc_metrics.so4_angle_stats(m.reshape(1, 1, 4, 4))
            self.assertAlmostEqual(lo.item(), want[0], places=3)
            self.assertAlmostEqual(hi.item(), want[1], places=3)

    def test_so4_angles_separate_a_pair_beta_cannot(self):
        # Why these series exist: beta = 4 - tr reads only cos alpha + cos gamma, so it is blind to
        # any move along that level set. Two very different mixes, one identical beta.
        alpha, gamma = 1.0, 2.0
        other_alpha = 0.2
        other_gamma = math.acos(math.cos(alpha) + math.cos(gamma) - math.cos(other_alpha))
        a = _canonical_so4(alpha, gamma).reshape(1, 1, 4, 4).expand(2, 1, 4, 4)
        b = _canonical_so4(other_alpha, other_gamma).reshape(1, 1, 4, 4).expand(2, 1, 4, 4)
        self.assertAlmostEqual(
            mhc_metrics.erase_beta_stats(a)[0].item(), mhc_metrics.erase_beta_stats(b)[0].item(), places=9
        )
        a_lo, a_hi = mhc_metrics.so4_angle_stats(a)
        b_lo, b_hi = mhc_metrics.so4_angle_stats(b)
        self.assertGreater(abs(a_lo.item() - b_lo.item()), 0.5)
        self.assertGreater(abs(a_hi.item() - b_hi.item()), 0.5)

    def test_so4_isoclinic_slice_is_where_beta_is_one_angle(self):
        # On alpha == gamma the two series collapse onto beta's single angle: beta = 4(1 - cos theta).
        for theta in (0.1, 0.36, 1.2):
            m = _conjugate_so4(_canonical_so4(theta, theta), seed=11).reshape(1, 1, 4, 4).expand(2, 1, 4, 4)
            lo, hi = mhc_metrics.so4_angle_stats(m)
            self.assertAlmostEqual(lo.item(), theta, places=4)
            self.assertAlmostEqual(hi.item(), theta, places=4)
            beta = mhc_metrics.erase_beta_stats(m)[0].item()
            self.assertAlmostEqual(beta, 4.0 * (1.0 - math.cos(theta)), places=7)

    def test_so4_angles_on_a_quaternion_pair_h_res(self):
        # R9's H_res: x -> q x rbar rotates by theta_q + theta_r and by |theta_q - theta_r|
        # (cos theta = the w component), which is why one quaternion could not cover SO(4).
        tq, tr = 0.5, 1.1
        axis = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        axis = axis / axis.norm()
        q = torch.cat([torch.tensor([math.cos(tq)], dtype=torch.float64), math.sin(tq) * axis])
        r = torch.cat([torch.tensor([math.cos(tr)], dtype=torch.float64), math.sin(tr) * axis])
        m = _quat_pair_so4(q, r)
        self.assertLess((m.T @ m - torch.eye(4, dtype=torch.float64)).abs().amax().item(), 1e-12)
        lo, hi = mhc_metrics.so4_angle_stats(m.reshape(1, 1, 4, 4))
        self.assertAlmostEqual(lo.item(), tr - tq, places=7)
        self.assertAlmostEqual(hi.item(), tq + tr, places=7)

    def test_so4_angles_stay_finite_on_a_non_orthogonal_mix(self):
        # The Sinkhorn baseline has no conjugacy class; the clamps must still keep the read finite.
        torch.manual_seed(5)
        m = torch.rand(6, 2, 4, 4)
        for _ in range(20):  # crude Sinkhorn to doubly stochastic
            m = m / m.sum(-1, keepdim=True)
            m = m / m.sum(-2, keepdim=True)
        for t in mhc_metrics.so4_angle_stats(m):
            self.assertTrue(torch.isfinite(t).all(), t)

    def test_stream_gram_on_an_orthogonal_frame(self):
        # 4 orthogonal streams with norms 1..4: per-stream norms recovered exactly, the
        # imbalance shows up in the ratio (a stream-axis mean would report a flat 2.5).
        n, c = 4, 4
        streams = torch.eye(c) * torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(-1)
        x = streams.reshape(1, 1, n * c).expand(3, 2, n * c).contiguous()
        norms, ratio, off_mean, off_max, _, _ = mhc_metrics.stream_gram_stats(x, n)
        self.assertEqual(tuple(norms.shape), (n,))
        for i, expected in enumerate((1.0, 2.0, 3.0, 4.0)):
            self.assertAlmostEqual(norms[i].item(), expected, places=5)
        self.assertAlmostEqual(ratio.item(), 4.0, places=5)
        self.assertAlmostEqual(off_mean.item(), 0.0, places=6)
        self.assertAlmostEqual(off_max.item(), 0.0, places=6)

    def test_stream_gram_collapse_is_unit_cosine(self):
        # All n streams on one direction: cosine == 1 off-diagonal, norms still balanced.
        n, c = 4, 8
        one = torch.randn(c)
        x = one.repeat(n).reshape(1, 1, n * c).expand(5, 2, n * c).contiguous()
        norms, ratio, off_mean, off_max, cv, eff_rank = mhc_metrics.stream_gram_stats(x, n)
        self.assertAlmostEqual(ratio.item(), 1.0, places=5)
        self.assertAlmostEqual(off_mean.item(), 1.0, places=5)
        self.assertAlmostEqual(off_max.item(), 1.0, places=5)
        self.assertAlmostEqual(norms[0].item(), one.norm().item(), places=4)
        # Collapsed onto the common mean -> zero cross-stream spread.
        self.assertAlmostEqual(cv.item(), 0.0, places=3)
        # ... and a rank-1 stream frame: only one direction carries energy.
        self.assertAlmostEqual(eff_rank.item(), 1.0, places=4)

    def test_stream_gram_offdiag_mean_is_signed(self):
        """The off-diagonal MEAN is the signed cosine; only the MAX stays on ``|cos|``.

        ``|cos|`` reads 1.0 both for "all streams collapsed onto the common mean" and for
        "alternating +v/-v", which are opposite conditions. The signed mean separates them
        (+1 vs negative); the max must stay absolute or it would miss an anti-aligned tail.
        """
        n, c = 4, 8
        v = torch.randn(c)

        _, _, same_mean, same_max, _, _ = mhc_metrics.stream_gram_stats(v.repeat(n).reshape(1, 1, n * c), n)
        self.assertAlmostEqual(same_mean.item(), 1.0, places=5)
        self.assertAlmostEqual(same_max.item(), 1.0, places=5)

        # +v, -v, +v, -v: 4 aligned and 8 anti-aligned off-diagonal pairs -> mean -1/3.
        alt = torch.stack([v, -v, v, -v]).reshape(1, 1, n * c)
        _, _, alt_mean, alt_max, _, _ = mhc_metrics.stream_gram_stats(alt, n)
        self.assertAlmostEqual(alt_mean.item(), -1.0 / 3.0, places=5)
        self.assertAlmostEqual(alt_max.item(), 1.0, places=5, msg="max must stay on |cos|")

        _, _, orth_mean, orth_max, _, _ = mhc_metrics.stream_gram_stats(torch.eye(n).reshape(1, 1, n * n), n)
        self.assertAlmostEqual(orth_mean.item(), 0.0, places=6)
        self.assertAlmostEqual(orth_max.item(), 0.0, places=6)

    def test_stream_cv_matches_the_closed_form(self):
        torch.manual_seed(4)
        n, c, t = 4, 7, 5
        xs = torch.randn(t, n, c)
        *_, cv, _ = mhc_metrics.stream_gram_stats(xs.reshape(t, 1, n * c), n)

        m = xs.mean(dim=1)  # [t, c]
        var = (xs - m.unsqueeze(1)).pow(2).sum(dim=(-2, -1)) / n
        expected = (var.sqrt() / m.norm(dim=-1)).mean()
        self.assertAlmostEqual(cv.item(), expected.item(), places=5)

    def test_stream_cv_survives_a_zero_common_mean(self):
        """Streams summing to zero make ``||m|| -> 0``; the relative floor must keep CV finite.

        An absolute eps here would let such a token contribute a huge value and destroy the
        token mean — this is the one place raw CV is fragile.
        """
        torch.manual_seed(5)
        n, c, t = 4, 6, 3
        xs = torch.randn(t, n, c)
        xs = xs - xs.mean(dim=1, keepdim=True)  # exact zero stream mean
        *_, cv, _ = mhc_metrics.stream_gram_stats(xs.reshape(t, 1, n * c), n)
        self.assertTrue(torch.isfinite(cv).all(), cv)
        self.assertGreater(cv.item(), 1.0, "a zero common mean is maximal cross-stream spread")

    def test_stream_stats_survive_an_all_zero_token(self):
        """An all-zero token must not poison any series with nan.

        ``stream_cv``'s relative floor is itself proportional to the token energy, so it also
        vanishes here — 0/0. ``stream_eff_rank`` reads 1.0 (no direction carries energy), never 0,
        which the metric can never legitimately take.
        """
        n, c = 4, 8
        for t in mhc_metrics.stream_gram_stats(torch.zeros(3, 2, n * c), n):
            self.assertTrue(torch.isfinite(t).all(), t)
        *_, cv, eff_rank = mhc_metrics.stream_gram_stats(torch.zeros(3, 2, n * c), n)
        self.assertAlmostEqual(cv.item(), 0.0, places=6)
        self.assertAlmostEqual(eff_rank.item(), 1.0, places=6)

    def test_stream_eff_rank_matches_the_singular_value_definition(self):
        """``(tr G)^2 / ||G||_F^2`` must equal ``(sum s^2)^2 / sum s^4`` on the stream matrix.

        The trace form is what the hot path computes (two Gram reductions, no eigvalsh); this
        pins it to the spectral definition it stands in for.
        """
        torch.manual_seed(11)
        n, c, t = 4, 9, 6
        xs = torch.randn(t, n, c)
        *_, eff_rank = mhc_metrics.stream_gram_stats(xs.reshape(t, 1, n * c), n)

        sv = torch.linalg.svdvals(xs)  # [t, n]
        s2 = sv.pow(2)
        expected = (s2.sum(dim=-1).pow(2) / s2.pow(2).sum(dim=-1)).mean()
        self.assertAlmostEqual(eff_rank.item(), expected.item(), places=4)

    def test_stream_eff_rank_spans_one_to_n(self):
        # Both ends of the range, exactly: rank-1 collapse -> 1, orthonormal frame -> n.
        n, c = 4, 8
        v = torch.randn(c)
        *_, collapsed = mhc_metrics.stream_gram_stats(v.repeat(n).reshape(1, 1, n * c), n)
        self.assertAlmostEqual(collapsed.item(), 1.0, places=4)

        q = torch.linalg.qr(torch.randn(c, n))[0].t().contiguous()  # [n, c], orthonormal rows
        *_, full = mhc_metrics.stream_gram_stats(q.reshape(1, 1, n * c), n)
        self.assertAlmostEqual(full.item(), float(n), places=4)

    def test_stream_eff_rank_counts_only_energised_directions(self):
        # 2 unit streams + 2 near-dead ones: eff rank ~= 2, NOT the algebraic rank 4.
        n, c = 4, 6
        e = torch.eye(c)
        streams = torch.stack([e[0], e[1], e[2] * 1e-4, e[3] * 1e-4])
        *_, eff_rank = mhc_metrics.stream_gram_stats(streams.reshape(1, 1, n * c), n)
        self.assertAlmostEqual(eff_rank.item(), 2.0, places=3)

    def test_stream_gram_offdiag_max_exposes_tail_the_mean_hides(self):
        # Token 0 orthogonal, token 1 has two parallel streams: mean 1/6, per-token max 1/2.
        n, c = 3, 3
        tok0 = torch.eye(n)
        tok1 = torch.stack([torch.eye(n)[0], torch.eye(n)[0], torch.eye(n)[2]])
        x = torch.stack([tok0.reshape(-1), tok1.reshape(-1)]).reshape(2, 1, n * c)
        _, ratio, off_mean, off_max, _, _ = mhc_metrics.stream_gram_stats(x, n)
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
        for _, _, mod, _ in targets:
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
                "composite_h_res_orth_dev_fwd",
                "composite_h_res_orth_dev_bwd",
                "h_res_orth_dev_max_med_ratio",
                "composite_h_res_orth_dev_fwd_max_med_ratio",
                "composite_h_res_orth_dev_bwd_max_med_ratio",
                "h_res_beta_mean",
                "h_res_beta_std",
                "h_res_outer_dev",
                "h_res_sigma_min",
                "h_res_sigma_mean",
                "composite_h_res_sigma_min_fwd",
                "composite_h_res_sigma_mean_fwd",
                "composite_h_res_sigma_min_bwd",
                "composite_h_res_sigma_mean_bwd",
                "h_res_theta_lo",
                "h_res_theta_hi",
                "stream_norm_max_min_ratio",
                "stream_gram_offdiag_mean",
                "stream_gram_offdiag_max",
                "stream_cv",
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
            for d in ("fwd", "bwd"):
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_0/{comp}_composite_h_res_orth_dev_{d}_max_med_ratio"], 1.0, places=5
                )
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_beta_mean"], 0.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_beta_std"], 0.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_outer_dev"], 2.0, places=5)
            # h_res == I is a mean-preserving isometry: sigma is flat 1 on 1^perp, and so is
            # every composite of it, in both directions.
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_sigma_min"], 1.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_sigma_mean"], 1.0, places=5)
            for d in ("fwd", "bwd"):
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_0/{comp}_composite_h_res_sigma_min_{d}"], 1.0, places=5
                )
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_0/{comp}_composite_h_res_sigma_mean_{d}"], 1.0, places=5
                )
            # ...and it rotates by nothing: both SO(4) angles are 0 (fp32 acos floor near cos = 1).
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_theta_lo"], 0.0, places=3)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_theta_hi"], 0.0, places=3)
        # global aggregate is derived too.
        self.assertIn("mhc_health/global_attn_amax_gain_fwd", latest)
        # the ratios are max-aggregated end to end: monitor buffer + training_logs suffix.
        self.assertIn("attn_h_res_orth_dev_max_med_ratio", MHCHealthMonitor.MAX_AGGREGATED)
        self.assertTrue(training_logs._is_max_metric("mhc_health/layer_0/attn_h_res_orth_dev_max_med_ratio"))
        # stream_cv and the composite sigmas are plain means; only sigma_min composes with min.
        for key in ("attn_stream_cv", "attn_composite_h_res_sigma_mean_fwd"):
            self.assertNotIn(key, MHCHealthMonitor.MAX_AGGREGATED)
            self.assertFalse(training_logs._is_max_metric(f"mhc_health/layer_0/{key}"))
            self.assertFalse(training_logs._is_min_metric(f"mhc_health/layer_0/{key}"))
        # No separate signed-cosine key was added: the existing _mean slot changed meaning.
        self.assertNotIn("mhc_health/layer_0/attn_stream_gram_offdiag_signed_mean", latest)

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
        for _, _, mod, _ in targets:
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

    def test_so4_angle_series_reach_the_logs_from_a_quaternion_pair_h_res(self):
        # End to end on an R9-shaped h_res: the two angles must survive the hook, the fp32 buffer
        # and the flush, and land at the pair the construction planted.
        n, s, b = 4, 3, 2
        tq, tr = 0.5, 1.1
        axis = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        axis = axis / axis.norm()
        q = torch.cat([torch.tensor([math.cos(tq)], dtype=torch.float64), math.sin(tq) * axis])
        r = torch.cat([torch.tensor([math.cos(tr)], dtype=torch.float64), math.sin(tr) * axis])
        h_res = _quat_pair_so4(q, r).float().reshape(1, 1, n, n).expand(s, b, n, n).contiguous()

        def make_hc():
            return FakeHC(
                n=n,
                layer_number=1,
                h_pre=torch.full((s, b, n), 0.5),
                h_post=torch.ones(s, b, n),
                h_res=h_res,
            )

        model = _mhc_model([FakeLayer(layer_number=1, attn=make_hc(), mlp=make_hc())])
        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_theta_lo"], tr - tq, places=4)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_theta_hi"], tq + tr, places=4)
            # constructively orthogonal, and beta only sees the cosine sum of the two angles
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_orth_dev"], 0.0, places=5)
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_res_beta_mean"],
                4.0 - 2.0 * (math.cos(tq + tr) + math.cos(tr - tq)),
                places=4,
            )
        # mean-aggregated: no _max suffix, so nothing to add to MAX_AGGREGATED
        self.assertNotIn("attn_h_res_theta_hi", MHCHealthMonitor.MAX_AGGREGATED)
        self.assertFalse(training_logs._is_max_metric("mhc_health/layer_0/attn_h_res_theta_hi"))

    def test_so4_angle_series_are_absent_when_n_is_not_four(self):
        # The formula's constants are n-specific, so the schema must not declare them at n != 4.
        n, s, b = 3, 2, 2
        self.assertEqual(mhc_monitor._so4_metric_names(n), ())
        model = _mhc_model([self._identity_layer(n, s, b, layer_number=1)])
        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertIn("mhc_health/layer_0/attn_h_res_sigma_min", latest)
        self.assertNotIn("mhc_health/layer_0/attn_h_res_theta_lo", latest)
        self.assertNotIn("mhc_health/layer_0/attn_h_res_theta_hi", latest)

    def _bda_case(self, targets, n, c, s, b, bias=None):
        """Drive every wrapped bda with a known update; return the expected mean Out/R."""
        x = torch.randn(s, b, n, c)
        o = torch.randn(s, b, c)
        h_res = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        h_post = torch.ones(s, b, n)
        for _, _, mod, _ in targets:
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
        self.assertEqual(monitor._h_res, {})
        self.assertEqual(monitor._expected, {})

    def test_no_graph_retention(self):
        n, s, b = 4, 2, 3
        ident = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        # Outputs attached to a graph (require grad) — the wrapper must detach.
        leaf = torch.zeros(s, b, n, requires_grad=True)

        def make_hc(layer_number):
            return FakeHC(
                n=n,
                layer_number=layer_number,
                h_pre=leaf + 0.5,
                h_post=leaf + 1.0,
                h_res=ident * (leaf.sum() + 1.0),  # requires_grad, has grad_fn
            )

        # Distinct hc objects per slot: sharing one would chain the wrappers onto the
        # same bound method, so a single call would fire all of them.
        layers = [FakeLayer(layer_number=i + 1, attn=make_hc(i + 1), mlp=make_hc(i + 1)) for i in range(2)]
        model = _mhc_model(layers)

        monitor = MHCHealthMonitor()
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        self.assertEqual(monitor._expected[0], 4)

        # Fire one of four: the stash stays mid-flight and inspectable.
        _, _, mod, _ = targets[0]
        mod.compute_mappings(torch.randn(4, 2, n * 8))

        stashed = list(monitor._h_res[0].values())
        self.assertEqual(len(stashed), 1)
        for t in stashed:
            self.assertFalse(t.requires_grad)
            self.assertIsNone(t.grad_fn)
        monitor.step()
        self.assertEqual(monitor._h_res, {})  # drained at step()

    def test_complete_chunk_drains_the_stash_inside_the_forward(self):
        """With gradient accumulation the stash must not span microbatches."""
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = MHCHealthMonitor()
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        self.assertEqual(monitor._expected[0], 2)

        self._drive(targets, x_dim=n * 8)
        self.assertEqual(monitor._h_res, {}, "a complete chunk should drain without waiting for step()")
        monitor.step()
        self.assertIn("mhc_health/layer_0/attn_composite_amax_gain_bwd", training_logs.get_latest(prefix="mhc_health"))


class MHCCompositeChainTest(unittest.TestCase):
    """Pin the two composite chains against explicit matrix products.

    ``h_res`` matrices here are deliberately NON-commuting and non-doubly-stochastic:
    under Sinkhorn every product is doubly stochastic and both gains sit at 1.0, so a
    wrong multiplication order is invisible. These are the signed charts (quat / cayley
    / orth) where it is not.
    """

    N = 3
    # Two tokens carrying the SAME matrix: the per-token mean stays exact against a
    # single [n, n] reference, while s*b > 1 keeps the unbiased std series off NaN.
    S, B = 2, 1

    def setUp(self):
        training_logs.reset()
        self._orig_layer_cls = mhc_monitor.HyperConnectionTransformerLayer
        self._orig_mod_cls = mhc_monitor.HyperConnectionModule
        mhc_monitor.HyperConnectionTransformerLayer = FakeLayer
        mhc_monitor.HyperConnectionModule = FakeHC

    def tearDown(self):
        mhc_monitor.HyperConnectionTransformerLayer = self._orig_layer_cls
        mhc_monitor.HyperConnectionModule = self._orig_mod_cls
        training_logs.reset()

    def _build(self, num_layers=3, seed=0):
        """One distinct h_res per hc module; returns (model, targets, monitor, mats).

        ``mats`` is the ``[n, n]`` per-module matrix in execution order (layer, then
        attn before mlp) — the order the chains must walk.
        """
        torch.manual_seed(seed)
        n = self.N
        mats, layers = [], []
        for li in range(num_layers):
            slots = {}
            for comp in ("attn", "mlp"):
                m = torch.randn(n, n)
                mats.append(m)
                slots[comp] = FakeHC(
                    n=n,
                    layer_number=li + 1,
                    h_pre=torch.full((self.S, self.B, n), 0.5),
                    h_post=torch.ones(self.S, self.B, n),
                    h_res=m.reshape(1, 1, n, n).expand(self.S, self.B, n, n).contiguous(),
                )
            layers.append(FakeLayer(layer_number=li + 1, attn=slots["attn"], mlp=slots["mlp"]))

        model = _mhc_model(layers)
        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        return model, targets, monitor, mats

    @staticmethod
    def _gain(mat, direction):
        dim = -1 if direction == "fwd" else -2
        return mat.sum(dim=dim).abs().amax().item()

    def _expected(self, mats):
        """Reference prefix / suffix products, built the long way."""
        prefix, suffix = [], [None] * len(mats)
        acc = None
        for m in mats:
            acc = m if acc is None else m @ acc  # M_l = H_l @ M_{l-1}
            prefix.append(acc)
        acc = None
        for i in range(len(mats) - 1, -1, -1):
            acc = mats[i] if acc is None else acc @ mats[i]  # S_l = S_{l+1} @ H_l
            suffix[i] = acc
        return prefix, suffix

    def _keys_in_order(self, targets):
        return sorted(
            ((li, comp) for li, comp, _, _ in targets),
            key=lambda k: (k[0], mhc_monitor._COMPONENT_RANK[k[1]]),
        )

    def _assert_chains(self, monitor, targets, mats, fire):
        prefix, suffix = self._expected(mats)
        fire()
        monitor.step()
        latest = training_logs.get_latest(prefix="mhc_health")

        for i, (layer_idx, comp) in enumerate(self._keys_in_order(targets)):
            got_fwd = latest[f"mhc_health/layer_{layer_idx}/{comp}_composite_amax_gain_fwd"]
            got_bwd = latest[f"mhc_health/layer_{layer_idx}/{comp}_composite_amax_gain_bwd"]
            self.assertAlmostEqual(got_fwd, self._gain(prefix[i], "fwd"), places=4, msg=f"fwd @ {i}")
            self.assertAlmostEqual(got_bwd, self._gain(suffix[i], "bwd"), places=4, msg=f"bwd @ {i}")
        return latest

    def test_chains_match_explicit_prefix_and_suffix_products(self):
        _, targets, monitor, mats = self._build()
        self._assert_chains(monitor, targets, mats, lambda: self._fire(targets, ascending=True))

    def test_chains_are_independent_of_firing_order(self):
        """Under recompute_granularity=full the only grad-enabled pass is the BACKWARD
        replay, which fires modules in DESCENDING order. The chains walk a static layer
        order, so both directions must come out identical either way."""
        _, targets, monitor, mats = self._build()
        ascending = self._assert_chains(monitor, targets, mats, lambda: self._fire(targets, ascending=True))

        training_logs.reset()
        _, targets2, monitor2, mats2 = self._build()  # same seed -> same matrices
        descending = self._assert_chains(monitor2, targets2, mats2, lambda: self._fire(targets2, ascending=False))

        for key, val in ascending.items():
            self.assertAlmostEqual(val, descending[key], places=5, msg=key)

    def test_suffix_gain_is_the_backward_jacobian_bound(self):
        """``S_l^T`` IS d(grad at l) / d(grad at output) — verified against autograd.

        Each hc module applies ``r <- H r``, so composing the stack and differentiating
        gives the gradient that actually reaches module ``l``. Its worst-case
        amplification under the paper's ``|row sum|`` bound is ``amax_gain(S_l, dim=-2)``,
        which is exactly what the bwd series reports.
        """
        n = self.N
        _, targets, monitor, mats = self._build()
        _, suffix = self._expected(mats)

        for i in range(len(mats)):
            r = torch.randn(n, 1, requires_grad=True)
            x = r
            for m in mats[i:]:
                x = m @ x
            g_out = torch.randn(n, 1)
            x.backward(g_out)
            self.assertTrue(
                torch.allclose(r.grad, suffix[i].T @ g_out, atol=1e-5),
                f"module {i}: grad != S_l^T g_out",
            )

    def _fire(self, targets, ascending: bool):
        order = self._keys_in_order(targets)
        if not ascending:
            order = list(reversed(order))
        by_key = {(li, comp): mod for li, comp, mod, _ in targets}
        # Same x per module regardless of firing order, so the input-derived series
        # (stream geometry) are comparable across the two orders too.
        xs = {key: torch.randn(self.S, self.B, self.N * 4) for key in self._keys_in_order(targets)}
        for key in order:
            by_key[key].compute_mappings(xs[key])


class MHCCompositeSigmaTest(unittest.TestCase):
    """The composite sigma spectrum on ``1^perp`` — the orthogonal-vs-DS headline.

    A doubly-stochastic ``H_res`` fixes the stream MEAN (sigma_max = 1) but can crush the
    stream-DIFFERENCE subspace ``1^perp``. Composed across depth that annihilates it, so the
    ``n`` streams collapse onto one direction. A mean-fixing orthogonal ``H_res`` is an
    isometry on ``1^perp``, so its composite holds sigma == 1 at any depth. These tests pin
    the mechanism; ``stream_cv`` measures the consequence.
    """

    N = 4

    def setUp(self):
        training_logs.reset()
        self._orig_layer_cls = mhc_monitor.HyperConnectionTransformerLayer
        self._orig_mod_cls = mhc_monitor.HyperConnectionModule
        mhc_monitor.HyperConnectionTransformerLayer = FakeLayer
        mhc_monitor.HyperConnectionModule = FakeHC

    def tearDown(self):
        mhc_monitor.HyperConnectionTransformerLayer = self._orig_layer_cls
        mhc_monitor.HyperConnectionModule = self._orig_mod_cls
        training_logs.reset()

    @staticmethod
    def _sinkhorn(a, iters=20, eps=1e-6):
        a = a.abs() + eps
        for _ in range(iters):
            a = a / a.sum(dim=-1, keepdim=True)
            a = a / a.sum(dim=-2, keepdim=True)
        return a

    @staticmethod
    def _cayley(s):
        """Mean-FIXING orthogonal: Cayley of a skew matrix that also kills ``1``.

        A plain ``Cayley(S - S^T)`` is orthogonal but rotates the mean direction into
        ``1^perp``, which changes the stream CV even though it is an isometry. Conjugating
        by ``P = I - J/n`` first gives ``A 1 = 0``, hence ``Q 1 = 1`` — the R3 / R9 property
        the DS baseline is being compared against.
        """
        n = s.shape[-1]
        eye = torch.eye(n, dtype=s.dtype)
        p = eye - torch.full((n, n), 1.0 / n, dtype=s.dtype)
        sp = p @ s @ p
        skew = sp - sp.transpose(-2, -1)
        return torch.linalg.solve(eye - skew, eye + skew)

    def _run(self, mats):
        """Drive one hc module per matrix (2 per layer) and return the flushed metrics."""
        n = self.N
        layers = []
        for li in range(0, len(mats), 2):
            slots = [
                FakeHC(
                    n=n,
                    layer_number=li // 2 + 1,
                    h_pre=torch.full((2, 1, n), 0.5),
                    h_post=torch.ones(2, 1, n),
                    h_res=m.reshape(1, 1, n, n).expand(2, 1, n, n).contiguous(),
                )
                for m in mats[li : li + 2]
            ]
            layers.append(FakeLayer(layer_number=li // 2 + 1, attn=slots[0], mlp=slots[1]))

        model = _mhc_model(layers)
        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        for _, _, mod, _ in targets:
            mod.compute_mappings(torch.randn(2, 1, n * 4))
        monitor.step()
        return training_logs.get_latest(prefix="mhc_health")

    def test_identity_stack_holds_unit_sigma(self):
        mats = [torch.eye(self.N) for _ in range(6)]
        latest = self._run(mats)
        for key, val in latest.items():
            if "composite_h_res_sigma" in key:
                self.assertAlmostEqual(val, 1.0, places=5, msg=key)

    def test_doubly_stochastic_composite_sigma_min_decays_with_depth(self):
        """The DS depth effect: the deepest composite must be far below a single layer."""
        torch.manual_seed(0)
        mats = [self._sinkhorn(torch.randn(self.N, self.N)) for _ in range(8)]
        latest = self._run(mats)

        single = latest["mhc_health/layer_0/attn_h_res_sigma_min"]
        # fwd prefix at the LAST module spans all 8 factors; bwd suffix at the FIRST does too.
        deep_fwd = latest["mhc_health/layer_3/mlp_composite_h_res_sigma_min_fwd"]
        deep_bwd = latest["mhc_health/layer_0/attn_composite_h_res_sigma_min_bwd"]
        for tag, deep in (("fwd", deep_fwd), ("bwd", deep_bwd)):
            self.assertLess(deep, single * 0.5, f"{tag}: composite {deep:.3e} vs single-layer {single:.3e}")
            self.assertLess(deep, 1e-2, f"{tag}: 1^perp should be nearly annihilated, got {deep:.3e}")

        # The 1-factor ends of each chain must still equal the single-layer value.
        self.assertAlmostEqual(latest["mhc_health/layer_0/attn_composite_h_res_sigma_min_fwd"], single, places=5)

    def test_orthogonal_composite_sigma_stays_unit_at_depth(self):
        """A mean-fixing orthogonal stack is an isometry on 1^perp at every depth."""
        torch.manual_seed(1)
        mats = [self._cayley(torch.randn(self.N, self.N) * 0.5) for _ in range(8)]
        latest = self._run(mats)
        for key, val in latest.items():
            if "composite_h_res_sigma" in key:
                self.assertAlmostEqual(val, 1.0, places=4, msg=key)

    def test_doubly_stochastic_collapse_shows_in_stream_cv(self):
        """The consequence side: pushing streams through a DS stack shrinks their CV.

        ``stream_cv`` is read off each module's INPUT, so this drives the streams through the
        matrices by hand and compares the CV of the frame before and after.
        """
        torch.manual_seed(2)
        n, c = self.N, 6
        xs = torch.randn(1, n, c)

        def cv_of(frame):
            *_, cv, _ = mhc_metrics.stream_gram_stats(frame.reshape(1, 1, n * c), n)
            return cv.item()

        before = cv_of(xs)
        ds = xs
        for _ in range(8):
            ds = self._sinkhorn(torch.randn(n, n)) @ ds
        orth = xs
        for _ in range(8):
            orth = self._cayley(torch.randn(n, n) * 0.5) @ orth

        self.assertLess(cv_of(ds), before * 0.1, "a DS stack should collapse the streams")
        self.assertAlmostEqual(cv_of(orth), before, places=3, msg="an isometry preserves the spread")

    def test_ds_stack_drops_stream_eff_rank_while_an_isometry_holds_it(self):
        """The same DS-vs-orthogonal contrast, read as effective rank instead of spread.

        A doubly-stochastic stack drives the stream frame to rank 1 (eff rank -> 1); a
        mean-fixing orthogonal stack is an isometry, so the frame's singular values — and
        therefore its eff rank — cannot change at all.
        """
        torch.manual_seed(12)
        n, c = self.N, 6
        xs = torch.randn(1, n, c)

        def eff_rank_of(frame):
            *_, er = mhc_metrics.stream_gram_stats(frame.reshape(1, 1, n * c), n)
            return er.item()

        before = eff_rank_of(xs)
        ds = xs
        for _ in range(8):
            ds = self._sinkhorn(torch.randn(n, n)) @ ds
        orth = xs
        for _ in range(8):
            orth = self._cayley(torch.randn(n, n) * 0.5) @ orth

        self.assertLess(eff_rank_of(ds), 1.05, "a DS stack should drive the frame to rank 1")
        self.assertAlmostEqual(eff_rank_of(orth), before, places=3, msg="an isometry preserves the spectrum")


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
