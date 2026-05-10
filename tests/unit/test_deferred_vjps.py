"""Coverage for the four deferred VJPs / JVPs:

- `ctc_loss` — forward-backward DP (VJP only; JVP through DP is impractical).
- `js_divergence` — clean closed-form `dJS/dp = ½ log(p/m)`.
- `wasserstein_distance` — sort-permutation routing.
- `nt_xent_loss` — normalize → Gram → masked log_softmax → positive mean.

Each VJP is checked numerically against a central finite-difference
gradient on a small input. Where a JVP exists, it's checked against the
finite-difference forward-mode tangent.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera.losses as losses
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp


# ── helpers ────────────────────────────────────────────────────────────────


def _numeric_grad(fn, x, eps=1e-4):
    g = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64).copy()
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = float(np.asarray(fn(x)).sum())
        x[idx] = orig - eps
        f_minus = float(np.asarray(fn(x)).sum())
        x[idx] = orig
        g[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return g


def _numeric_jvp(fn, x, dx, eps=1e-5):
    plus = np.asarray(fn(x + eps * dx), dtype=np.float64)
    minus = np.asarray(fn(x - eps * dx), dtype=np.float64)
    return (plus - minus) / (2 * eps)


# ── registration smoke tests ───────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "ctc_loss",
    "js_divergence",
    "wasserstein_distance",
    "nt_xent_loss",
])
def test_vjp_registered(name):
    assert get_vjp(name) is not None, f"VJP missing: {name}"


@pytest.mark.parametrize("name", [
    "ctc_loss",
    "js_divergence",
    "wasserstein_distance",
    "nt_xent_loss",
])
def test_jvp_registered(name):
    assert get_jvp(name) is not None, f"JVP missing: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# CTC
# ─────────────────────────────────────────────────────────────────────────────


def _make_log_probs(rng, T, B, V):
    """Random log-probabilities (already log-softmaxed along the V axis)."""
    raw = rng.normal(size=(T, B, V))
    return raw - np.log(np.exp(raw).sum(axis=-1, keepdims=True))


def test_ctc_loss_vjp_matches_numeric_simple():
    """B=1, T=4, V=3, target [1, 2] — fully exercised forward-backward."""
    rng = np.random.default_rng(0)
    T, B, V = 4, 1, 3
    log_probs = _make_log_probs(rng, T, B, V)
    targets = np.array([[1, 2]])
    input_lengths = np.array([T])
    target_lengths = np.array([2])

    grad_lp, *rest = get_vjp("ctc_loss")(
        1.0, log_probs, targets, input_lengths, target_lengths,
        blank=0, reduction="mean",
    )
    expected = _numeric_grad(
        lambda v: losses.ctc_loss(v, targets, input_lengths, target_lengths,
                                  blank=0, reduction="mean"),
        log_probs,
    )
    np.testing.assert_allclose(grad_lp, expected, atol=1e-3)
    assert all(g is None for g in rest)


def test_ctc_loss_vjp_handles_batch_and_padding():
    """B=2, varying input/target lengths — grad outside valid region must be 0."""
    rng = np.random.default_rng(1)
    T, B, V = 5, 2, 4
    log_probs = _make_log_probs(rng, T, B, V)
    # Pad targets to a uniform length; only `target_lengths[b]` chars are used.
    targets = np.array([[1, 2, 0], [3, 0, 0]])
    input_lengths = np.array([5, 4])
    target_lengths = np.array([2, 1])

    grad_lp, *_ = get_vjp("ctc_loss")(
        1.0, log_probs, targets, input_lengths, target_lengths,
        blank=0, reduction="mean",
    )
    # Beyond inp_len=4 for batch 1, the gradient stays zero — no DP touched it.
    np.testing.assert_array_equal(grad_lp[4:, 1, :], 0.0)

    expected = _numeric_grad(
        lambda v: losses.ctc_loss(v, targets, input_lengths, target_lengths,
                                  blank=0, reduction="mean"),
        log_probs,
    )
    np.testing.assert_allclose(grad_lp, expected, atol=1e-3)


def test_ctc_loss_vjp_obeys_reduction_modes():
    rng = np.random.default_rng(2)
    T, B, V = 4, 2, 3
    log_probs = _make_log_probs(rng, T, B, V)
    targets = np.array([[1, 2], [2, 1]])
    input_lengths = np.array([T, T])
    target_lengths = np.array([2, 2])

    g_mean, *_ = get_vjp("ctc_loss")(
        1.0, log_probs, targets, input_lengths, target_lengths, blank=0,
        reduction="mean",
    )
    g_sum, *_ = get_vjp("ctc_loss")(
        1.0, log_probs, targets, input_lengths, target_lengths, blank=0,
        reduction="sum",
    )
    # Mean is sum/B; flipping reduction multiplies the gradient by B.
    np.testing.assert_allclose(g_sum, g_mean * B, atol=1e-7)


# ── CTC JVP — via VJP-contraction ("double backward") ──────────────────────


def test_ctc_loss_jvp_matches_finite_difference_simple():
    """Single-batch JVP: tangent is just `∇L · v` for scalar reductions."""
    rng = np.random.default_rng(0)
    T, B, V = 4, 1, 3
    log_probs = _make_log_probs(rng, T, B, V)
    targets = np.array([[1, 2]])
    input_lengths = np.array([T])
    target_lengths = np.array([2])
    v = rng.normal(size=log_probs.shape) * 0.05

    primal, tangent = get_jvp("ctc_loss")(
        (log_probs, targets, input_lengths, target_lengths),
        (v, np.zeros_like(targets), np.zeros_like(input_lengths),
         np.zeros_like(target_lengths)),
        blank=0, reduction="mean",
    )
    np.testing.assert_allclose(
        primal,
        losses.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                        blank=0, reduction="mean"),
        atol=1e-9,
    )
    expected_tan = _numeric_jvp(
        lambda v_: losses.ctc_loss(v_, targets, input_lengths, target_lengths,
                                   blank=0, reduction="mean"),
        log_probs, v,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


def test_ctc_loss_jvp_handles_batch_and_padding():
    """Multi-batch JVP with varying input/target lengths."""
    rng = np.random.default_rng(1)
    T, B, V = 5, 2, 4
    log_probs = _make_log_probs(rng, T, B, V)
    targets = np.array([[1, 2, 0], [3, 0, 0]])
    input_lengths = np.array([5, 4])
    target_lengths = np.array([2, 1])
    v = rng.normal(size=log_probs.shape) * 0.05

    _, tangent = get_jvp("ctc_loss")(
        (log_probs, targets, input_lengths, target_lengths),
        (v, np.zeros_like(targets), np.zeros_like(input_lengths),
         np.zeros_like(target_lengths)),
        blank=0, reduction="mean",
    )
    expected_tan = _numeric_jvp(
        lambda v_: losses.ctc_loss(v_, targets, input_lengths, target_lengths,
                                   blank=0, reduction="mean"),
        log_probs, v,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


def test_ctc_loss_jvp_matches_vjp_contraction_for_scalar_reductions():
    """The JVP should be exactly `dot(VJP(1.0), v)` — verifies the contraction
    trick is doing what it claims and not silently double-counting."""
    rng = np.random.default_rng(2)
    T, B, V = 4, 2, 3
    log_probs = _make_log_probs(rng, T, B, V)
    targets = np.array([[1, 2], [2, 1]])
    input_lengths = np.array([T, T])
    target_lengths = np.array([2, 2])
    v = rng.normal(size=log_probs.shape) * 0.1

    # Reduction='mean':
    grad_mean, *_ = get_vjp("ctc_loss")(
        1.0, log_probs, targets, input_lengths, target_lengths,
        blank=0, reduction="mean",
    )
    _, tangent_mean = get_jvp("ctc_loss")(
        (log_probs, targets, input_lengths, target_lengths),
        (v, np.zeros_like(targets), np.zeros_like(input_lengths),
         np.zeros_like(target_lengths)),
        blank=0, reduction="mean",
    )
    np.testing.assert_allclose(
        tangent_mean, float(np.sum(grad_mean * v)), atol=1e-9,
    )


def test_ctc_loss_jvp_reduction_none_returns_per_batch_tangent():
    """For reduction='none' the primal is shape (B,); the JVP must match."""
    rng = np.random.default_rng(3)
    T, B, V = 4, 3, 3
    log_probs = _make_log_probs(rng, T, B, V)
    targets = np.array([[1, 2], [2, 1], [1, 1]])
    input_lengths = np.array([T, T, T])
    target_lengths = np.array([2, 2, 2])
    v = rng.normal(size=log_probs.shape) * 0.05

    primal, tangent = get_jvp("ctc_loss")(
        (log_probs, targets, input_lengths, target_lengths),
        (v, np.zeros_like(targets), np.zeros_like(input_lengths),
         np.zeros_like(target_lengths)),
        blank=0, reduction="none",
    )
    assert tangent.shape == (B,)
    # Each batch-element tangent should match a per-element FD reference.
    for b in range(B):
        primal_b = lambda v_, b=b: losses.ctc_loss(
            v_, targets, input_lengths, target_lengths,
            blank=0, reduction="none",
        )[b]
        expected_b = _numeric_jvp(primal_b, log_probs, v)
        np.testing.assert_allclose(tangent[b], expected_b, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# JS divergence
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_simplex(rng, shape):
    raw = rng.uniform(0.05, 1.0, size=shape)
    return raw / raw.sum(axis=-1, keepdims=True)


def test_js_divergence_vjp_matches_numeric():
    rng = np.random.default_rng(0)
    p = _normalize_simplex(rng, (3, 4))
    q = _normalize_simplex(rng, (3, 4))
    grad_p, grad_q = get_vjp("js_divergence")(1.0, p, q, reduction="mean")

    expected_p = _numeric_grad(lambda v: losses.js_divergence(v, q), p)
    expected_q = _numeric_grad(lambda v: losses.js_divergence(p, v), q)
    np.testing.assert_allclose(grad_p, expected_p, atol=1e-3)
    np.testing.assert_allclose(grad_q, expected_q, atol=1e-3)


def test_js_divergence_jvp_matches_numeric():
    rng = np.random.default_rng(1)
    p = _normalize_simplex(rng, (3, 4))
    q = _normalize_simplex(rng, (3, 4))
    dp = rng.normal(size=p.shape) * 0.01
    dq = rng.normal(size=q.shape) * 0.01

    primal, tangent = get_jvp("js_divergence")(
        (p, q), (dp, dq), reduction="mean",
    )
    np.testing.assert_allclose(primal, losses.js_divergence(p, q))
    expected_tan = _numeric_jvp(lambda v: losses.js_divergence(v, q), p, dp) + \
                   _numeric_jvp(lambda v: losses.js_divergence(p, v), q, dq)
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


def test_js_divergence_symmetric_in_arguments():
    """JS(p,q) == JS(q,p); gradients should mirror the swap."""
    rng = np.random.default_rng(2)
    p = _normalize_simplex(rng, (4,))
    q = _normalize_simplex(rng, (4,))
    g_p_first, g_q_first = get_vjp("js_divergence")(1.0, p, q)
    g_q_swap, g_p_swap = get_vjp("js_divergence")(1.0, q, p)
    np.testing.assert_allclose(g_p_first, g_p_swap, atol=1e-7)
    np.testing.assert_allclose(g_q_first, g_q_swap, atol=1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# Wasserstein
# ─────────────────────────────────────────────────────────────────────────────


def test_wasserstein_vjp_matches_numeric():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 5))
    y = rng.normal(size=(3, 5))
    grad_x, grad_y = get_vjp("wasserstein_distance")(
        1.0, x, y, reduction="mean",
    )
    expected_x = _numeric_grad(lambda v: losses.wasserstein_distance(v, y), x)
    expected_y = _numeric_grad(lambda v: losses.wasserstein_distance(x, v), y)
    np.testing.assert_allclose(grad_x, expected_x, atol=1e-3)
    np.testing.assert_allclose(grad_y, expected_y, atol=1e-3)


def test_wasserstein_jvp_matches_numeric():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 6))
    y = rng.normal(size=(2, 6))
    dx = rng.normal(size=x.shape) * 0.01
    dy = rng.normal(size=y.shape) * 0.01

    primal, tangent = get_jvp("wasserstein_distance")((x, y), (dx, dy))
    np.testing.assert_allclose(primal, losses.wasserstein_distance(x, y))
    expected_tan = _numeric_jvp(
        lambda v: losses.wasserstein_distance(v, y), x, dx,
    ) + _numeric_jvp(
        lambda v: losses.wasserstein_distance(x, v), y, dy,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


def test_wasserstein_grad_routes_through_sort_permutation():
    """If x is already sorted ascending and y is sorted ascending, the
    gradient at the sort positions should equal `(1/N) sign(x - y)`."""
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = np.array([[0.5, 2.5, 2.5, 5.0]])
    grad_x, grad_y = get_vjp("wasserstein_distance")(1.0, x, y)
    # Both already sorted, so the sort permutation is identity. N=4.
    expected = np.sign(x - y) / 4.0  # reduction='mean' over batch of 1 → /1.
    np.testing.assert_allclose(grad_x, expected)
    np.testing.assert_allclose(grad_y, -expected)


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent
# ─────────────────────────────────────────────────────────────────────────────


def _nt_xent_setup(rng, B=4, D=8):
    z = rng.normal(size=(B, D))
    # Three classes with at least one positive each — avoids the all-zero row case.
    labels = np.array([0, 0, 1, 1])
    return z[:B], labels[:B]


def test_nt_xent_vjp_matches_numeric():
    rng = np.random.default_rng(0)
    z, labels = _nt_xent_setup(rng)
    grad_z, grad_labels = get_vjp("nt_xent_loss")(
        1.0, z, labels, temperature=0.5, reduction="mean",
    )
    expected = _numeric_grad(
        lambda v: losses.nt_xent_loss(v, labels, temperature=0.5), z,
    )
    np.testing.assert_allclose(grad_z, expected, atol=5e-3, rtol=5e-3)
    assert grad_labels is None


def test_nt_xent_jvp_matches_numeric():
    rng = np.random.default_rng(1)
    z, labels = _nt_xent_setup(rng)
    dz = rng.normal(size=z.shape) * 0.01
    primal, tangent = get_jvp("nt_xent_loss")(
        (z, labels), (dz, np.zeros_like(labels)),
        temperature=0.5, reduction="mean",
    )
    np.testing.assert_allclose(
        primal, losses.nt_xent_loss(z, labels, temperature=0.5), atol=1e-7,
    )
    expected_tan = _numeric_jvp(
        lambda v: losses.nt_xent_loss(v, labels, temperature=0.5), z, dz,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=5e-3, rtol=5e-3)


def test_nt_xent_handles_no_positives_row_gracefully():
    """If some row has no positives (all unique labels), its loss term is 0
    and its gradient should be 0 — no division-by-zero from K_i=0."""
    rng = np.random.default_rng(2)
    z = rng.normal(size=(3, 4))
    labels = np.array([0, 1, 2])  # no positives anywhere.
    grad_z, _ = get_vjp("nt_xent_loss")(
        1.0, z, labels, temperature=0.5, reduction="sum",
    )
    np.testing.assert_array_equal(grad_z, np.zeros_like(z))


# ─────────────────────────────────────────────────────────────────────────────
# Registry promotion
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "ctc_loss", "js_divergence", "wasserstein_distance", "nt_xent_loss",
])
def test_registry_reports_vjp_complete(name):
    from tessera.compiler.primitive_coverage import coverage_for

    entry = coverage_for(name)
    assert entry.contract_status["vjp"] == "complete", (
        f"registering a VJP for {name} must auto-flip the dashboard"
    )


@pytest.mark.parametrize("name", [
    "js_divergence", "wasserstein_distance", "nt_xent_loss",
])
def test_registry_reports_jvp_complete(name):
    from tessera.compiler.primitive_coverage import coverage_for

    entry = coverage_for(name)
    assert entry.contract_status["jvp"] == "complete"


def test_registry_keeps_ctc_jvp_unregistered():
    """CTC JVP is intentionally not implemented — the registry should
    still surface that gap."""
    from tessera.compiler.primitive_coverage import coverage_for

    entry = coverage_for("ctc_loss")
    assert entry.contract_status["jvp"] != "complete"
