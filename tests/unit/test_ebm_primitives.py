"""EBM1 acceptance: energy primitive surface (Euclidean baseline).

Sprint: EBM1.
Roadmap: docs/audit/ga_ebm_roadmap.md
Spec: docs/spec/EBM_SPEC.md § 2

Covers the EBM1 acceptance criteria:
  - 5 primitives registered in primitive_coverage.py under category="ebm".
  - Langevin step on quadratic energy E(y) = ||y||^2 / 2 converges to
    the origin in 100 steps (zero-temperature) to fp32 tolerance.
  - self_verify returns argmin energy hard; soft-min is differentiable
    through the soft-min smoothing parameter beta.
  - inner_step / langevin_step are pure (no mutation of inputs).
  - decode_init returns correct (B, K, *event) shapes for all 3 strategies.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import ebm
from tessera.rng import RNGKey


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------

def test_five_ebm1_primitives_registered_in_coverage() -> None:
    """EBM1 introduces the five core energy primitives; later sprints add
    more under the same category. We assert the EBM1 subset is present
    and that each entry has been promoted out of ``planned`` now that
    Python + native Apple GPU paths both exist (Decision #25)."""
    from tessera.compiler import primitive_coverage as pc

    ebm_entries = {
        e.name: e for e in pc.all_primitive_coverages().values() if e.category == "ebm"
    }
    expected_ebm1 = {
        "ebm_decode_init",
        "ebm_energy",
        "ebm_inner_step",
        "ebm_langevin_step",
        "ebm_self_verify",
    }
    assert expected_ebm1.issubset(ebm_entries.keys())
    for name in expected_ebm1:
        # 2026-05-18: all five EBM1 primitives ship fused MSL kernels,
        # so they live at ``status="partial"`` per Decision #25 (Python
        # reference + native kernel + tests; contract-axis completeness
        # still the next gate).
        assert ebm_entries[name].status == "partial"
        assert ebm_entries[name].category == "ebm"


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------

def test_energy_calls_user_model_fn() -> None:
    def quad(x, y):
        return 0.5 * (y ** 2).sum()

    out = ebm.energy(quad, x=None, y=np.array([1.0, 2.0, 3.0]))
    assert pytest.approx(float(out), abs=1e-6) == 7.0


def test_energy_threads_params_when_provided() -> None:
    def linear(x, y, *, params):
        return float((params["w"] * y).sum())

    out = ebm.energy(linear, x=None, y=np.array([1.0, 1.0]),
                    params={"w": np.array([2.0, 3.0])})
    assert pytest.approx(float(out), abs=1e-6) == 5.0


# ---------------------------------------------------------------------------
# inner_step
# ---------------------------------------------------------------------------

def test_inner_step_pure_gradient_descent() -> None:
    y = np.array([1.0, 2.0, -1.0])
    grad = np.array([0.5, 0.25, 0.0])
    out = ebm.inner_step(y, grad, eta=0.1)
    expected = y - 0.1 * grad
    assert np.allclose(out, expected)


def test_inner_step_does_not_mutate_input() -> None:
    y = np.array([1.0, 2.0, 3.0])
    y_before = y.copy()
    _ = ebm.inner_step(y, np.zeros_like(y), eta=0.5)
    assert np.array_equal(y, y_before)


def test_inner_step_requires_matching_shapes() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        ebm.inner_step(np.zeros(3), np.zeros(4), eta=0.1)


def test_inner_step_with_noise_requires_rng_key() -> None:
    with pytest.raises(ValueError, match="rng_key when noise_scale"):
        ebm.inner_step(np.zeros(3), np.zeros(3), eta=0.1, noise_scale=0.1)


def test_inner_step_with_noise_is_deterministic_per_key() -> None:
    key = RNGKey.from_seed(7)
    y = np.zeros(8)
    grad = np.zeros(8)
    a = ebm.inner_step(y, grad, eta=0.1, rng_key=key, noise_scale=0.5)
    b = ebm.inner_step(y, grad, eta=0.1, rng_key=key, noise_scale=0.5)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# langevin_step — convergence on quadratic energy
# ---------------------------------------------------------------------------

def _quadratic_energy(y: np.ndarray) -> float:
    return float(0.5 * (y ** 2).sum())


def _quadratic_grad(y: np.ndarray) -> np.ndarray:
    return y.astype(np.float64, copy=False)


def test_langevin_step_converges_to_origin_zero_temperature() -> None:
    """E(y) = ||y||^2/2; pure gradient descent collapses to origin."""
    key = RNGKey.from_seed(0)
    y = np.array([3.0, -4.0, 2.5, 1.0, -2.0], dtype=np.float32)
    for _ in range(100):
        y, key = ebm.langevin_step(
            y,
            energy_fn=_quadratic_energy,
            eta=0.1,
            temperature=0.0,
            rng_key=key,
            grad_fn=_quadratic_grad,
        )
    # After 100 steps of y <- 0.9 * y the magnitude is 0.9^100 ~= 2.66e-5.
    assert np.allclose(y, 0.0, atol=1e-3)


def test_langevin_step_thermal_noise_has_correct_scale() -> None:
    """At equilibrium, Langevin samples are zero-mean with variance ~ T."""
    key = RNGKey.from_seed(123)
    y = np.zeros(2048, dtype=np.float32)
    samples = []
    for _ in range(400):
        y, key = ebm.langevin_step(
            y,
            energy_fn=_quadratic_energy,
            eta=0.05,
            temperature=1.0,
            rng_key=key,
            grad_fn=_quadratic_grad,
        )
        samples.append(y.copy())
    burned_in = np.stack(samples[200:])
    # E(y) = 0.5 * y^2 has stationary distribution N(0, T=1) under Langevin.
    # Sampled variance should be ~1 with reasonable tolerance.
    var = float(burned_in.var())
    assert 0.7 < var < 1.4, f"expected stationary variance ~1, got {var}"
    assert abs(float(burned_in.mean())) < 0.1


def test_langevin_step_returns_consumed_key() -> None:
    key = RNGKey.from_seed(11)
    y = np.array([1.0, 2.0])
    _, next_key = ebm.langevin_step(
        y, energy_fn=_quadratic_energy, eta=0.1, temperature=0.5,
        rng_key=key, grad_fn=_quadratic_grad,
    )
    # Per S4 functional-RNG conventions, next_key is fresh.
    assert isinstance(next_key, RNGKey)
    assert (next_key.seed_high, next_key.seed_low) != (key.seed_high, key.seed_low)


def test_langevin_step_rejects_bad_eta_and_temperature() -> None:
    key = RNGKey.from_seed(1)
    with pytest.raises(ValueError, match="eta > 0"):
        ebm.langevin_step(np.zeros(3), _quadratic_energy, eta=0.0,
                          temperature=0.0, rng_key=key, grad_fn=_quadratic_grad)
    with pytest.raises(ValueError, match="temperature >= 0"):
        ebm.langevin_step(np.zeros(3), _quadratic_energy, eta=0.1,
                          temperature=-0.1, rng_key=key, grad_fn=_quadratic_grad)


def test_langevin_step_numerical_grad_matches_analytic() -> None:
    """Without grad_fn, central differences should match the analytic gradient."""
    key = RNGKey.from_seed(5)
    y = np.array([1.0, 2.0, -1.5], dtype=np.float32)
    y_numerical, _ = ebm.langevin_step(
        y, energy_fn=_quadratic_energy, eta=0.05, temperature=0.0,
        rng_key=key,
    )
    y_analytic, _ = ebm.langevin_step(
        y, energy_fn=_quadratic_energy, eta=0.05, temperature=0.0,
        rng_key=key, grad_fn=_quadratic_grad,
    )
    # Central differences on a quadratic are exact up to f64 noise.
    assert np.allclose(y_numerical, y_analytic, atol=1e-4)


# ---------------------------------------------------------------------------
# self_verify
# ---------------------------------------------------------------------------

def test_self_verify_hard_argmin_returns_minimum_energy_candidate() -> None:
    energies = np.array([[2.0, 0.5, 3.0], [1.0, 1.5, 0.25]])
    # Candidates: (B=2, K=3, D=2)
    candidates = np.array([
        [[10, 11], [20, 21], [30, 31]],
        [[40, 41], [50, 51], [60, 61]],
    ], dtype=np.float32)
    out = ebm.self_verify(energies, candidates)
    assert out.shape == (2, 2)
    # B=0: min at k=1 -> (20, 21); B=1: min at k=2 -> (60, 61).
    assert np.array_equal(out, np.array([[20, 21], [60, 61]], dtype=np.float32))


def test_self_verify_soft_min_approaches_hard_argmin_as_beta_grows() -> None:
    energies = np.array([[0.0, 1.0, 2.0]])
    candidates = np.array([[[1.0], [2.0], [3.0]]])
    hard = ebm.self_verify(energies, candidates)
    soft_large_beta = ebm.self_verify(energies, candidates, beta=50.0)
    assert np.allclose(hard, np.array([[1.0]]))
    assert np.allclose(soft_large_beta, np.array([[1.0]]), atol=1e-3)


def test_self_verify_soft_min_is_differentiable_in_candidates() -> None:
    """Soft-min must produce a smooth weighted combination, not a hard pick."""
    energies = np.array([[1.0, 1.0]])  # equal energies => 50/50 weighting
    candidates = np.array([[[0.0], [4.0]]])
    out = ebm.self_verify(energies, candidates, beta=1.0)
    assert np.allclose(out, np.array([[2.0]]), atol=1e-6)


def test_self_verify_validates_shapes() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        ebm.self_verify(np.zeros(3), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="candidates.shape"):
        ebm.self_verify(np.zeros((2, 3)), np.zeros((2, 4, 5)))


def test_self_verify_rejects_non_positive_beta() -> None:
    with pytest.raises(ValueError, match="beta > 0"):
        ebm.self_verify(np.zeros((1, 2)), np.zeros((1, 2, 1)), beta=0.0)
    with pytest.raises(ValueError, match="beta > 0"):
        ebm.self_verify(np.zeros((1, 2)), np.zeros((1, 2, 1)), beta=-0.5)


# ---------------------------------------------------------------------------
# decode_init
# ---------------------------------------------------------------------------

def test_decode_init_noise_strategy() -> None:
    key = RNGKey.from_seed(0)
    x = np.zeros((3, 7))  # B=3 batch
    out = ebm.decode_init(x, K=4, init_strategy="noise", rng_key=key, shape=(5,))
    assert out.shape == (3, 4, 5)
    assert out.dtype == np.float32


def test_decode_init_noise_is_deterministic_per_key() -> None:
    key = RNGKey.from_seed(99)
    a = ebm.decode_init(np.zeros(3), K=2, init_strategy="noise",
                        rng_key=key, shape=(4,))
    b = ebm.decode_init(np.zeros(3), K=2, init_strategy="noise",
                        rng_key=key, shape=(4,))
    assert np.array_equal(a, b)


def test_decode_init_base_model_broadcasts_initialization() -> None:
    def base(x):
        return x + 1.0
    out = ebm.decode_init(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        K=3,
        init_strategy="base_model",
        base_model_fn=base,
    )
    assert out.shape == (2, 3, 2)
    # All K replicas should be identical.
    assert np.array_equal(out[:, 0, :], out[:, 1, :])
    assert np.array_equal(out[0, 0], np.array([2.0, 3.0], dtype=np.float32))


def test_decode_init_copy_strategy() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    out = ebm.decode_init(x, K=2, init_strategy="copy")
    assert out.shape == (2, 2, 3)
    assert np.array_equal(out[0, 0], x[0])
    assert np.array_equal(out[1, 1], x[1])


def test_decode_init_validates_strategy_and_requirements() -> None:
    with pytest.raises(ValueError, match="K > 0"):
        ebm.decode_init(np.zeros(3), K=0, init_strategy="copy")
    with pytest.raises(ValueError, match="requires rng_key"):
        ebm.decode_init(np.zeros(3), K=2, init_strategy="noise", shape=(4,))
    with pytest.raises(ValueError, match="explicit `shape`"):
        ebm.decode_init(np.zeros(3), K=2, init_strategy="noise",
                        rng_key=RNGKey.from_seed(0))
    with pytest.raises(ValueError, match="base_model_fn"):
        ebm.decode_init(np.zeros(3), K=2, init_strategy="base_model")
    with pytest.raises(ValueError, match="unknown init_strategy"):
        ebm.decode_init(np.zeros(3), K=2, init_strategy="garbage")


# ---------------------------------------------------------------------------
# Version stamp
# ---------------------------------------------------------------------------

def test_ebm_version_at_least_ebm1() -> None:
    """Energy primitives exist from EBM1 onwards; allow later sprint bumps."""
    assert ebm.__version__.startswith("0.0.0-ebm")
    sprint_str = ebm.__version__.split("-ebm", 1)[1]
    assert int(sprint_str) >= 1
