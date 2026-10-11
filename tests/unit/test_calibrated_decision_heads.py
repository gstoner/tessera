"""Calibrated decision heads — proper scoring rules, ordinal heads, calibration.

Covers ``tessera.losses.{brier_loss, binary_brier_loss, ordinal_logistic_loss,
ordinal_probs}`` and ``tessera.calibration``. Python reference only: these are
not ``op_catalog`` Graph ops and no lowering or device result is claimed here.

The VJP/JVP pairs are also swept by the autodiff law engine
(``test_autodiff_laws.py``); the tests below pin what the law sweep cannot —
the integer-target / ``ignore_index`` paths against finite differences,
propriety of the scoring rules, and the metric definitions on hand-built cases.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tessera as ts
from tessera import calibration as cal
from tessera import losses as L
from tessera.autodiff import grad
from tessera.autodiff.jvp import _JVPS
from tessera.autodiff.vjp import _VJPS

RNG = np.random.default_rng(20261011)


def _softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _fd_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


# ── brier_loss ───────────────────────────────────────────────────────────────


def test_brier_matches_definition_for_hard_and_soft_targets():
    z = RNG.standard_normal((5, 4))
    y = np.array([0, 3, 1, 1, 2])
    p = _softmax(z)
    onehot = np.eye(4)[y]
    np.testing.assert_allclose(L.brier_loss(z, y), np.mean(np.sum((p - onehot) ** 2, -1)))
    np.testing.assert_allclose(L.brier_loss(z, onehot), L.brier_loss(z, y))
    per = L.brier_loss(z, y, reduction="none")
    assert per.shape == (5,) and np.all((per >= 0) & (per <= 2))


def test_brier_is_strictly_proper():
    """E_{y~q}[Brier(p, y)] is minimized exactly at p = q (the property that
    makes it a calibration objective, not just an accuracy one)."""
    q = np.array([0.6, 0.25, 0.1, 0.05])

    def expected_score(logits):
        return sum(q[k] * L.brier_loss(logits[None], np.array([k])) for k in range(4))

    at_truth = expected_score(np.log(q))
    for _ in range(25):
        other = np.log(q) + 0.3 * RNG.standard_normal(4)
        assert expected_score(other) > at_truth - 1e-12
    # the gradient of the expected score vanishes at the truth
    g = sum(q[k] * _VJPS["brier_loss"](1.0, np.log(q)[None], np.array([k]))[0]
            for k in range(4))
    np.testing.assert_allclose(g, 0.0, atol=1e-12)


def test_brier_ignore_index_masks_loss_mean_and_gradient():
    z = RNG.standard_normal((4, 3))
    y = np.array([2, -100, 0, -100])
    keep = y != -100
    np.testing.assert_allclose(L.brier_loss(z, y), L.brier_loss(z[keep], y[keep]))
    np.testing.assert_allclose(L.brier_loss(z, y, reduction="sum"),
                               L.brier_loss(z[keep], y[keep], reduction="sum"))
    gz, gy = _VJPS["brier_loss"](1.0, z, y)
    assert gy is None
    np.testing.assert_allclose(gz[~keep], 0.0)
    np.testing.assert_allclose(gz, _fd_grad(lambda t: L.brier_loss(t, y), z), atol=1e-8)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_brier_integer_target_rules_match_finite_differences(reduction):
    z = RNG.standard_normal((3, 5))
    y = np.array([4, 0, 2])
    dout = np.ones(3) if reduction == "none" else 1.0
    f = lambda t: np.sum(np.asarray(L.brier_loss(t, y, reduction=reduction)) * dout)
    gz, _ = _VJPS["brier_loss"](dout, z, y, reduction=reduction)
    np.testing.assert_allclose(gz, _fd_grad(f, z), atol=1e-8)
    dz = RNG.standard_normal(z.shape)
    primal, tangent = _JVPS["brier_loss"]((z, y), (dz, None), reduction=reduction)
    np.testing.assert_allclose(primal, L.brier_loss(z, y, reduction=reduction))
    eps = 1e-6
    fd = (np.asarray(L.brier_loss(z + eps * dz, y, reduction=reduction))
          - np.asarray(L.brier_loss(z - eps * dz, y, reduction=reduction))) / (2 * eps)
    np.testing.assert_allclose(tangent, fd, atol=1e-8)


def test_brier_rejects_bad_targets():
    z = RNG.standard_normal((2, 3))
    with pytest.raises(ValueError, match="out of range"):
        L.brier_loss(z, np.array([0, 3]))
    with pytest.raises(ValueError, match="out of range"):
        L.brier_loss(z, np.array([0, -1]))          # no negative wraparound
    with pytest.raises(ValueError, match="must match logits"):
        L.brier_loss(z, np.ones((2, 4)) / 4)
    with pytest.raises(ValueError, match="do not match"):
        L.brier_loss(z, np.array([0, 1, 2]))


def test_brier_is_tape_differentiable_through_ops():
    z = RNG.standard_normal((4, 6))
    y = np.array([1, 5, 0, 3])
    g = grad(lambda t: ts.ops.brier_loss(t, y))(z)
    np.testing.assert_allclose(np.asarray(g), _fd_grad(lambda t: L.brier_loss(t, y), z), atol=1e-8)


# ── binary_brier_loss ────────────────────────────────────────────────────────


def test_binary_brier_is_half_the_two_class_brier():
    z = RNG.standard_normal(7)
    t = (RNG.random(7) > 0.5).astype(np.int64)
    two_class = np.stack([np.zeros(7), z], axis=-1)   # softmax([0, z]) = [1-σ, σ]
    np.testing.assert_allclose(L.binary_brier_loss(z, t), 0.5 * L.brier_loss(two_class, t))


def test_binary_brier_is_stable_and_proper():
    z = np.array([-800.0, -30.0, 0.0, 30.0, 800.0])
    out = L.binary_brier_loss(z, np.array([0.0, 0.0, 1.0, 1.0, 1.0]), reduction="none")
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [0.0, 0.0, 0.25, 0.0, 0.0], atol=1e-12)
    # expected score under Bernoulli(q) is minimized at sigmoid(z) = q
    q = 0.7
    e = lambda zz: q * L.binary_brier_loss(zz, 1.0) + (1 - q) * L.binary_brier_loss(zz, 0.0)
    zq = np.log(q / (1 - q))
    assert all(e(zq + d) > e(zq) for d in (-0.5, -0.05, 0.05, 0.5))


# ── ordinal heads ────────────────────────────────────────────────────────────


def test_ordinal_loss_is_summed_threshold_bce():
    z = RNG.standard_normal((5, 3))            # K = 4 levels
    y = np.array([0, 3, 1, 2, 3])
    b = (y[:, None] > np.arange(3)).astype(np.float64)
    ref = np.mean(np.sum(L.binary_cross_entropy_loss(z, b, reduction="none"), -1))
    np.testing.assert_allclose(L.ordinal_logistic_loss(z, y), ref)


def test_ordinal_rules_match_finite_differences_with_ignore_index():
    z = RNG.standard_normal((4, 3))
    y = np.array([2, -100, 0, 3])
    f = lambda t: L.ordinal_logistic_loss(t, y)
    gz, gy = _VJPS["ordinal_logistic_loss"](1.0, z, y)
    assert gy is None
    np.testing.assert_allclose(gz, _fd_grad(f, z), atol=1e-8)
    np.testing.assert_allclose(gz[1], 0.0)
    np.testing.assert_allclose(f(z), L.ordinal_logistic_loss(z[[0, 2, 3]], y[[0, 2, 3]]))


def test_ordinal_rejects_bad_levels():
    z = RNG.standard_normal((2, 3))
    with pytest.raises(ValueError, match="out of range"):
        L.ordinal_logistic_loss(z, np.array([0, 4]))
    with pytest.raises(ValueError, match="integer levels"):
        L.ordinal_logistic_loss(z, np.array([0.0, 1.0]))


def test_ordinal_probs_are_a_distribution_even_for_unordered_logits():
    z = np.array([[2.0, 0.0, -2.0],             # rank-consistent
                  [-1.0, 1.5, 0.5]])            # violates monotonicity
    p = L.ordinal_probs(z)
    assert p.shape == (2, 4)
    assert np.all(p >= 0.0)
    np.testing.assert_allclose(p.sum(-1), 1.0)
    s = 1 / (1 + np.exp(-z[0]))
    np.testing.assert_allclose(p[0], [1 - s[0], s[0] - s[1], s[1] - s[2], s[2]])


# ── calibration metrics ──────────────────────────────────────────────────────


def test_ece_zero_when_confidence_equals_accuracy_per_bin():
    # 10 predictions at confidence 0.8 (8 correct), 10 at 0.6 (6 correct)
    conf = np.r_[np.full(10, 0.8), np.full(10, 0.6)]
    probs = np.stack([conf, 1 - conf], -1)
    labels = np.r_[np.zeros(8), np.ones(2), np.zeros(6), np.ones(4)].astype(int)
    assert cal.expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0, abs=1e-12)
    assert cal.max_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0, abs=1e-12)


def test_ece_and_mce_on_a_known_miscalibration():
    # overconfident: 0.95 stated, 70% correct (bin 20); 0.55 stated, 50% correct
    conf = np.r_[np.full(100, 0.95), np.full(100, 0.55)]
    probs = np.stack([conf, 1 - conf], -1)
    labels = np.r_[np.zeros(70), np.ones(30), np.zeros(50), np.ones(50)].astype(int)
    assert cal.expected_calibration_error(probs, labels, n_bins=20) == pytest.approx(
        0.5 * 0.25 + 0.5 * 0.05)
    assert cal.max_calibration_error(probs, labels, n_bins=20) == pytest.approx(0.25)


def test_bins_are_right_closed_and_match_reported_edges():
    bins = cal.reliability_bins([0.0, 0.1, 0.3, 0.30000001, 1.0], [1, 1, 1, 1, 1], n_bins=10)
    assert list(np.nonzero(bins.counts)[0]) == [0, 2, 3, 9]
    assert bins.counts[0] == 2                     # 0.0 and 0.1 → (0, 0.1]
    assert np.isnan(bins.accuracy[5])              # empty bin reports NaN
    assert bins.total == 5


def test_binary_convention_uses_max_of_p_and_one_minus_p():
    p_true = np.array([0.9, 0.2, 0.5])
    labels = np.array([1, 0, 0])
    conf, correct = cal.top_label_confidence(p_true, labels)
    np.testing.assert_allclose(conf, [0.9, 0.8, 0.5])
    np.testing.assert_allclose(correct, [1.0, 1.0, 0.0])   # 0.5 predicts True


@pytest.mark.parametrize("probs,labels,match", [
    (np.array([[2.0, -1.0]]), np.array([0]), "logits"),
    (np.array([[0.5, 0.6]]), np.array([0]), "sum to 1"),
    (np.array([[0.5, 0.5]]), np.array([2]), "out of range"),
    (np.array([0.5, np.nan]), np.array([0, 1]), "finite"),
    (np.array([0.3]), np.array([2]), "0 or 1"),
    (np.zeros((2, 3)) + 1 / 3, np.zeros((3,), int), "convention"),
])
def test_metrics_reject_invalid_inputs(probs, labels, match):
    with pytest.raises(ValueError, match=match):
        cal.expected_calibration_error(probs, labels)


# ── soft_binned_ece ──────────────────────────────────────────────────────────


def test_soft_ece_converges_to_binned_ece():
    m = 10
    # confidences at bin centres → well away from edges
    conf = (RNG.integers(5, 10, size=200) + 0.5) / m
    correct = (RNG.random(200) < conf - 0.1).astype(np.float64)
    hard = cal.reliability_bins(conf, correct, n_bins=m)
    filled = hard.counts > 0
    hard_ece = np.sum(hard.counts[filled] / hard.total * hard.gaps()[filled])
    soft = cal.soft_binned_ece(conf, correct, n_bins=m, temperature=1e-6)
    assert soft == pytest.approx(hard_ece, abs=1e-9)


def test_soft_ece_smoothing_matters_only_across_mixed_sign_bins():
    """With one over- and one under-confident bin, leaking membership between
    them cancels residual, so a smoother temperature reads lower. (When every
    bin errs the same way, Σ|r_m| = |Σ_i d_i| for any memberships — the
    surrogate is then temperature-invariant, which is correct, not a bug.)"""
    conf = np.r_[np.full(50, 0.45), np.full(50, 0.55)]
    correct = np.r_[np.ones(50), np.zeros(50)]          # bin 4 under, bin 5 over
    sharp = cal.soft_binned_ece(conf, correct, n_bins=10, temperature=1e-6)
    smooth = cal.soft_binned_ece(conf, correct, n_bins=10, temperature=5e-2)
    assert sharp == pytest.approx(0.55)          # |1 - 0.45| and |0 - 0.55|
    assert smooth < sharp


def test_soft_ece_gradient_reduces_miscalibration():
    conf = 0.05 + 0.9 * RNG.random(64)
    correct = (RNG.random(64) < 0.5).astype(np.float64)
    before = cal.soft_binned_ece(conf, correct, n_bins=8, temperature=5e-3)
    g, _ = _VJPS["soft_binned_ece"](1.0, conf, correct, n_bins=8, temperature=5e-3)
    np.testing.assert_allclose(
        g, _fd_grad(lambda c: cal.soft_binned_ece(c, correct, n_bins=8, temperature=5e-3), conf),
        atol=1e-6)
    after = cal.soft_binned_ece(conf - 1e-3 * g / np.abs(g).max(), correct,
                                n_bins=8, temperature=5e-3)
    assert after < before


def test_soft_ece_rejects_bad_parameters():
    with pytest.raises(ValueError, match="temperature"):
        cal.soft_binned_ece([0.5], [1.0], temperature=0.0)
    with pytest.raises(ValueError, match="n_bins"):
        cal.soft_binned_ece([0.5], [1.0], n_bins=0)
    with pytest.raises(ValueError, match="differ in length"):
        cal.soft_binned_ece([0.5, 0.6], [1.0])


# ── temperature scaling ──────────────────────────────────────────────────────


def _nll(logits, labels):
    lp = logits - np.log(np.sum(np.exp(logits - logits.max(-1, keepdims=True)), -1,
                                keepdims=True)) - logits.max(-1, keepdims=True)
    return -np.mean(lp[np.arange(len(labels)), labels])


def test_fit_temperature_recovers_overconfidence_and_minimizes_nll():
    n, k, t_true = 4000, 5, 2.5
    base = RNG.standard_normal((n, k))
    labels = np.array([RNG.choice(k, p=_softmax(row)) for row in base])
    logits = base * t_true                       # an overconfident model
    t_hat = cal.fit_temperature(logits, labels)
    assert t_hat == pytest.approx(t_true, rel=0.08)
    best = _nll(logits / t_hat, labels)
    for t in (t_hat * 0.97, t_hat * 1.03):
        assert _nll(logits / t, labels) > best
    ece_before = cal.expected_calibration_error(_softmax(logits), labels)
    ece_after = cal.expected_calibration_error(_softmax(logits / t_hat), labels)
    assert ece_after < ece_before


def test_fit_temperature_warns_on_separable_data():
    logits = np.array([[5.0, 0.0], [0.0, 5.0], [6.0, -1.0]])
    labels = np.array([0, 1, 0])
    with pytest.warns(RuntimeWarning, match="separable"):
        t = cal.fit_temperature(logits, labels)
    assert t == pytest.approx(1e-2)


def test_fit_temperature_warns_when_unidentifiable():
    with pytest.warns(RuntimeWarning, match="unidentifiable"):
        assert cal.fit_temperature(np.ones((4, 3)), np.array([0, 1, 2, 0])) == 1.0


def test_fit_temperature_does_not_warn_on_an_interior_optimum():
    logits = RNG.standard_normal((200, 3))
    labels = RNG.integers(0, 3, size=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cal.fit_temperature(logits * 0.5 + np.eye(3)[labels], labels)


def test_fit_temperature_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="integer class indices"):
        cal.fit_temperature(np.zeros((2, 3)), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="out of range"):
        cal.fit_temperature(np.zeros((2, 3)), np.array([0, 3]))
    with pytest.raises(ValueError, match=">= 2 classes"):
        cal.fit_temperature(np.zeros((2, 1)), np.array([0, 0]))
    with pytest.raises(ValueError, match="bounds"):
        cal.fit_temperature(np.zeros((2, 3)), np.array([0, 1]), bounds=(2.0, 1.0))
