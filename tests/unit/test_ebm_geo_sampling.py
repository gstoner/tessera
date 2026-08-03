"""EBM7 acceptance: manifold-aware Langevin integrators.

Sprint: EBM7 (the GA + EBM merge point).
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md § EBM7
Scope lock: docs/audit/domain/DOMAIN_AUDIT.md § Q5

Covers the EBM7 acceptance criteria:
  - Bivector Langevin starting from a random rotor stays even-grade
    (specifically grade-2) to fp32 numerical noise over 1k steps.
  - Sphere Langevin (target distribution = vMF) recovers the
    concentration parameter to within 10% over 10k samples.
  - Sphere samples stay on the unit sphere (|x|=1 to 1e-7) across the
    entire chain.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import ebm
from tessera.ebm import (
    bivector_langevin_sample,
    bivector_langevin_step,
    sphere_langevin_sample,
    sphere_langevin_step,
    vmf_kappa_mle,
)
from tessera.ga import Cl, Multivector
from tessera.rng import RNGKey


# ---------------------------------------------------------------------------
# Registry — 4 EBM7 primitives present
# ---------------------------------------------------------------------------

def test_four_ebm7_primitives_registered() -> None:
    from tessera.compiler import primitive_coverage as pc

    names = {e.name for e in pc.all_primitive_coverages().values()}
    expected = {
        "ebm_bivector_langevin_step",
        "ebm_sphere_langevin_step",
        "ebm_bivector_langevin_sample",
        "ebm_sphere_langevin_sample",
    }
    assert expected.issubset(names)


# ---------------------------------------------------------------------------
# Bivector Langevin — headline grade-preservation test
# ---------------------------------------------------------------------------

def _quadratic_bivector_energy(bivector: Multivector) -> float:
    """E(B) = 0.5 * Σ B_i² over grade-2 coefficients only."""
    coeffs = bivector.coefficients
    return 0.5 * float(np.sum(coeffs ** 2))


def test_bivector_langevin_stays_in_grade_2_over_1000_steps() -> None:
    """The acceptance criterion: starting from a random grade-2 bivector,
    1000 steps of bivector Langevin must keep the state in the grade-2
    subspace. Odd-grade and even-but-non-grade-2 leakage should be
    bounded by float noise.
    """
    a = Cl(3, 0)
    rng_np = np.random.RandomState(7)
    # Random bivector: only e12, e13, e23 coefficients set.
    bivec_coeffs = np.zeros(a.dim, dtype=np.float64)
    bivec_coeffs[a.blade("e12").mask] = rng_np.randn()
    bivec_coeffs[a.blade("e13").mask] = rng_np.randn()
    bivec_coeffs[a.blade("e23").mask] = rng_np.randn()
    state = Multivector(bivec_coeffs, a, grades={2})

    key = RNGKey.from_seed(0)
    max_off_grade = 0.0
    for step in range(1000):
        state, key = bivector_langevin_step(
            state,
            _quadratic_bivector_energy,
            eta=0.005,
            temperature=1.0,
            rng_key=key,
            grade=2,
        )
        # Check active grades — only grade 2 should carry signal.
        for blade in a.blades():
            if blade.grade != 2:
                amp = abs(float(state.coefficients[blade.mask]))
                max_off_grade = max(max_off_grade, amp)

    assert max_off_grade < 1e-10, (
        f"bivector Langevin leaked into non-grade-2 components over 1000 steps; "
        f"max off-grade magnitude = {max_off_grade:.3e}"
    )


def test_bivector_langevin_converges_under_quadratic_energy() -> None:
    """Zero-temperature bivector Langevin on E(B) = ||B||²/2 should drive
    the state toward the zero bivector — same convergence as Euclidean
    Langevin, but restricted to the bivector subspace.
    """
    a = Cl(3, 0)
    init = Multivector(
        np.array([0.0, 0.0, 0.0, 3.0, 0.0, -2.0, 1.5, 0.0]),
        a,
        grades={2},
    )
    state = init
    key = RNGKey.from_seed(1)
    for _ in range(200):
        state, key = bivector_langevin_step(
            state,
            _quadratic_bivector_energy,
            eta=0.05,
            temperature=0.0,
            rng_key=key,
            grade=2,
        )
    # After 200 steps of y ← 0.95 y, magnitude ~3.5 * 0.95^200 ≈ 1.4e-5.
    assert np.linalg.norm(state.coefficients) < 1e-3


def test_bivector_langevin_chain_records_grade_2_only() -> None:
    a = Cl(3, 0)
    init = Multivector.from_blade(a.blade("e12"), a, dtype=np.float64)
    key = RNGKey.from_seed(2)
    samples, _, _ = bivector_langevin_sample(
        key, init=init,
        energy_fn=_quadratic_bivector_energy,
        eta=0.01, temperature=1.0,
        n_samples=200, burn_in=50,
        grade=2,
    )
    assert samples.shape == (200, a.dim)
    # No grade-{0, 1, 3} components in any sample.
    for blade in a.blades():
        if blade.grade != 2:
            assert np.all(np.abs(samples[:, blade.mask]) < 1e-10), (
                f"blade {blade.name} (grade {blade.grade}) has non-zero amplitude"
            )


def test_bivector_langevin_with_analytic_grad_fn() -> None:
    """Passing grad_fn explicitly should match the numerical-grad result
    to high precision."""
    a = Cl(3, 0)
    init = Multivector.from_blade(a.blade("e12"), a, dtype=np.float64) + \
           0.3 * Multivector.from_blade(a.blade("e23"), a, dtype=np.float64)
    # E = 0.5||B||² → grad E = B.
    def analytic_grad(B):
        return B

    key = RNGKey.from_seed(3)
    state_a, _ = bivector_langevin_step(
        init, _quadratic_bivector_energy,
        eta=0.05, temperature=0.0,
        rng_key=key, grade=2, grad_fn=analytic_grad,
    )
    state_n, _ = bivector_langevin_step(
        init, _quadratic_bivector_energy,
        eta=0.05, temperature=0.0,
        rng_key=key, grade=2,  # numerical
    )
    # Central differences on a quadratic match analytic to high precision.
    assert np.allclose(state_a.coefficients, state_n.coefficients, atol=1e-6)


def test_bivector_langevin_validates_params() -> None:
    a = Cl(3, 0)
    state = Multivector.from_blade(a.blade("e12"), a, dtype=np.float64)
    key = RNGKey.from_seed(0)
    with pytest.raises(ValueError, match="eta > 0"):
        bivector_langevin_step(state, _quadratic_bivector_energy,
                               eta=0.0, temperature=0.0, rng_key=key)
    with pytest.raises(ValueError, match="temperature >= 0"):
        bivector_langevin_step(state, _quadratic_bivector_energy,
                               eta=0.1, temperature=-0.1, rng_key=key)
    with pytest.raises(ValueError, match="out of range"):
        bivector_langevin_step(state, _quadratic_bivector_energy,
                               eta=0.1, temperature=1.0, rng_key=key, grade=5)


# ---------------------------------------------------------------------------
# Sphere Langevin — headline vMF recovery test
# ---------------------------------------------------------------------------

def _vmf_energy(kappa: float, mu: np.ndarray):
    """E(x) = -κ μ·x — the energy whose Boltzmann distribution is vMF(μ, κ)."""
    def fn(x):
        return -float(kappa) * float(np.dot(mu, x))
    return fn


def _vmf_grad(kappa: float, mu: np.ndarray):
    """Analytic gradient of E(x) = -κ μ·x: ∇E = -κ μ (constant)."""
    grad = -float(kappa) * np.asarray(mu, dtype=np.float64)
    return lambda x, g=grad: g


def test_sphere_langevin_stays_on_unit_sphere() -> None:
    """Every sampled point must lie on S² to numerical precision."""
    rng_np = np.random.RandomState(0)
    mu = np.array([0.0, 0.0, 1.0])
    init = rng_np.randn(3); init /= np.linalg.norm(init)
    key = RNGKey.from_seed(0)
    samples, _, _ = sphere_langevin_sample(
        key, init=init,
        energy_fn=_vmf_energy(2.0, mu),
        eta=0.01, temperature=1.0,
        n_samples=500, burn_in=100,
        grad_fn=_vmf_grad(2.0, mu),
    )
    norms = np.linalg.norm(samples, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-7), (
        f"sphere chain drifted off unit sphere: norm range "
        f"[{norms.min():.6f}, {norms.max():.6f}]"
    )


def test_sphere_langevin_recovers_vmf_concentration_within_10pct() -> None:
    """Target: vMF(μ = e3, κ = 5) on S². 10k samples; assert MLE within
    10% of 5.0.

    Step size matters: unadjusted Langevin has bias proportional to
    eta, and the bias direction over-concentrates samples near the
    mode (inflating κ̂). With eta=0.003 + burn_in=3000 the bias drops
    below the 10% target. EBM7+ work could add a MALA correction on
    the sphere to eliminate the bias entirely.
    """
    kappa_target = 5.0
    mu = np.array([0.0, 0.0, 1.0])
    init = np.array([1.0, 0.0, 0.0])
    key = RNGKey.from_seed(13)
    samples, _, _ = sphere_langevin_sample(
        key, init=init,
        energy_fn=_vmf_energy(kappa_target, mu),
        eta=0.003, temperature=1.0,
        n_samples=10_000, burn_in=3000, thin=1,
        grad_fn=_vmf_grad(kappa_target, mu),
    )
    kappa_est = vmf_kappa_mle(samples, dim=3)
    rel_err = abs(kappa_est - kappa_target) / kappa_target
    assert rel_err < 0.10, (
        f"vMF κ MLE = {kappa_est:.3f} vs target {kappa_target:.3f} "
        f"(rel err {rel_err:.4f})"
    )


def test_sphere_langevin_zero_temperature_climbs_to_mode() -> None:
    """At T=0, gradient descent on E(x) = -μ·x climbs to the mode μ.
    Equivalent: dot(x, μ) → 1 as the chain runs.
    """
    mu = np.array([0.0, 0.0, 1.0])
    init = np.array([1.0, 0.0, 0.0])
    key = RNGKey.from_seed(0)
    x = init
    for _ in range(2000):
        x, key = sphere_langevin_step(
            x, _vmf_energy(1.0, mu),
            eta=0.05, temperature=0.0,
            rng_key=key, grad_fn=_vmf_grad(1.0, mu),
        )
    # Should converge close to μ.
    assert float(np.dot(x, mu)) > 0.999


def test_sphere_langevin_validates_inputs() -> None:
    key = RNGKey.from_seed(0)
    # Rank-2 init is rejected.
    with pytest.raises(ValueError, match="rank-1"):
        sphere_langevin_step(np.zeros((2, 3)), lambda x: 0.0,
                             eta=0.1, temperature=1.0, rng_key=key)
    # Non-unit init is rejected.
    with pytest.raises(ValueError, match=r"\|x\| = 1"):
        sphere_langevin_step(np.array([2.0, 0.0, 0.0]),
                             lambda x: 0.0,
                             eta=0.1, temperature=1.0, rng_key=key)
    # eta <= 0.
    init = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="eta > 0"):
        sphere_langevin_step(init, lambda x: 0.0,
                             eta=0.0, temperature=1.0, rng_key=key)


def test_sphere_langevin_sample_normalizes_user_init() -> None:
    """User can pass any non-zero vector; it gets normalized to S^{d-1}."""
    key = RNGKey.from_seed(5)
    mu = np.array([0.0, 1.0, 0.0])
    samples, _, _ = sphere_langevin_sample(
        key,
        init=np.array([3.0, 0.0, 0.0]),  # not unit norm
        energy_fn=_vmf_energy(1.0, mu),
        eta=0.05, temperature=1.0,
        n_samples=8, burn_in=0,
        grad_fn=_vmf_grad(1.0, mu),
    )
    norms = np.linalg.norm(samples, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-7)


def test_sphere_langevin_sample_rejects_zero_init() -> None:
    key = RNGKey.from_seed(0)
    with pytest.raises(ValueError, match="non-zero"):
        sphere_langevin_sample(
            key, init=np.zeros(3),
            energy_fn=lambda x: 0.0,
            eta=0.1, n_samples=1,
        )


# ---------------------------------------------------------------------------
# vMF MLE helper
# ---------------------------------------------------------------------------

def test_vmf_kappa_mle_on_concentrated_data() -> None:
    """When all samples are exactly at μ, r̄ → 1 and κ → ∞."""
    samples = np.tile(np.array([0.0, 0.0, 1.0]), (100, 1))
    kappa = vmf_kappa_mle(samples, dim=3)
    assert kappa == float("inf")


def test_vmf_kappa_mle_on_uniform_data_is_near_zero() -> None:
    """Uniform-on-sphere data gives r̄ ≈ 0, hence κ ≈ 0."""
    rng = np.random.RandomState(0)
    pts = rng.randn(1000, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    kappa = vmf_kappa_mle(pts, dim=3)
    # Tiny non-zero from finite samples; should be near 0.
    assert kappa < 1.0


def test_vmf_kappa_mle_validates_shape() -> None:
    with pytest.raises(ValueError, match=r"\(N, d\)"):
        vmf_kappa_mle(np.zeros((5,)), dim=3)
    with pytest.raises(ValueError, match="dim=3"):
        vmf_kappa_mle(np.zeros((5, 4)), dim=3)


# ---------------------------------------------------------------------------
# Version stamp
# ---------------------------------------------------------------------------

def test_ebm_version_advanced_to_ebm7() -> None:
    assert ebm.__version__.startswith("0.0.0-ebm")
    sprint_str = ebm.__version__.split("-ebm", 1)[1]
    assert int(sprint_str) >= 7


# ─────────────────────────────────────────────────────────────────────────────
# W0.3 — traceable energies use reverse mode; NumPy callbacks fall back
# ─────────────────────────────────────────────────────────────────────────────


class TestTraceableEnergyGradient:
    """Both gradient paths, and the rule that picks between them.

    Central differences cost 2·D energy evaluations, and for a multivector in
    Cl(p,q,r) that is D = 2^n -- exponential in the algebra's dimension, and
    only first-order accurate. Reverse mode costs one forward plus one backward
    regardless of D. Before W0.3 the samplers always took the numerical path,
    because `Multivector.coefficients` hands out a fresh whole-array view on
    every access, so nothing the energy touched could be matched back to the
    state buffer.
    """

    @staticmethod
    def _traceable_energy(counter):
        """E(c) = sum(c^2), routed through tessera.ops so the tape records it."""
        from tessera import ops
        from tessera.ga.multivector import Multivector

        def energy(x):
            counter["n"] += 1
            c = x.coefficients if isinstance(x, Multivector) else x
            return ops.reduce(ops.mul(c, c), op="sum")

        return energy

    @staticmethod
    def _raw_numpy_energy(counter):
        """The same energy in raw NumPy — records nothing, must fall back."""
        from tessera.ga.multivector import Multivector

        def energy(x):
            counter["n"] += 1
            c = x.coefficients if isinstance(x, Multivector) else x
            return float(np.sum(np.asarray(c) ** 2))

        return energy

    def test_traceable_multivector_energy_uses_reverse_mode(self):
        from tessera.ebm.geo_sampling import _tape_grad_mv
        from tessera.ga.multivector import Multivector
        from tessera.ga.signature import Cl

        algebra = Cl(3, 0)
        state = Multivector(np.arange(1.0, algebra.dim + 1.0), algebra)

        counter = {"n": 0}
        grad = _tape_grad_mv(self._traceable_energy(counter), state)

        assert grad is not None, "a tessera.ops-based energy must be traceable"
        # E = sum(c^2) → dE/dc = 2c, exactly (not to finite-difference tolerance).
        np.testing.assert_allclose(grad, 2.0 * np.asarray(state.coefficients))
        assert counter["n"] == 1, (
            f"reverse mode must evaluate the energy once; got {counter['n']}"
        )

    def test_reverse_mode_is_cheaper_than_central_differences(self):
        from tessera.ebm.geo_sampling import _numerical_grad_mv, _tape_grad_mv
        from tessera.ga.multivector import Multivector
        from tessera.ga.signature import Cl

        algebra = Cl(3, 0)
        state = Multivector(np.arange(1.0, algebra.dim + 1.0), algebra)

        tape_calls = {"n": 0}
        _tape_grad_mv(self._traceable_energy(tape_calls), state)

        num_calls = {"n": 0}
        _numerical_grad_mv(self._traceable_energy(num_calls), state)

        # 2 evaluations per coefficient, and Cl(3,0) has 2^3 = 8 coefficients.
        assert num_calls["n"] == 2 * algebra.dim
        assert tape_calls["n"] == 1
        assert tape_calls["n"] < num_calls["n"]

    def test_raw_numpy_energy_is_untraceable_and_falls_back(self):
        from tessera.ebm.geo_sampling import _numerical_grad_mv, _tape_grad_mv
        from tessera.ga.multivector import Multivector
        from tessera.ga.signature import Cl

        algebra = Cl(3, 0)
        state = Multivector(np.arange(1.0, algebra.dim + 1.0), algebra)

        counter = {"n": 0}
        energy = self._raw_numpy_energy(counter)

        assert _tape_grad_mv(energy, state) is None, (
            "a raw-NumPy energy records no cotangent path and must report "
            "untraceable so the caller falls back, rather than returning a "
            "silently wrong gradient"
        )

        # The fallback still produces the right answer, to fd tolerance.
        counter["n"] = 0
        grad = _numerical_grad_mv(energy, state)
        np.testing.assert_allclose(
            grad, 2.0 * np.asarray(state.coefficients), rtol=1e-4, atol=1e-4
        )

    def test_both_paths_agree_on_a_langevin_step(self):
        """The sampler must produce the same step whichever path supplied the
        gradient -- the choice is an optimization, not a semantic change."""
        from tessera.ebm.geo_sampling import bivector_langevin_step
        from tessera.ga.multivector import Multivector
        from tessera.ga.signature import Cl
        from tessera.rng import RNGKey

        algebra = Cl(3, 0)
        coeffs = np.zeros(algebra.dim)
        # Populate the grade-2 (bivector) slots.
        for i, blade in enumerate(algebra.blades()):
            if blade.grade == 2:
                coeffs[i] = 0.5 * (i + 1)
        state = Multivector(coeffs, algebra)

        traced, _ = bivector_langevin_step(
            state, self._traceable_energy({"n": 0}),
            eta=0.01, temperature=0.0, rng_key=RNGKey(0),
        )
        fallback, _ = bivector_langevin_step(
            state, self._raw_numpy_energy({"n": 0}),
            eta=0.01, temperature=0.0, rng_key=RNGKey(0),
        )

        np.testing.assert_allclose(
            np.asarray(traced.coefficients),
            np.asarray(fallback.coefficients),
            rtol=1e-5, atol=1e-6,
        )

    def test_sphere_step_traces_a_tessera_ops_energy(self):
        from tessera.ebm.geo_sampling import _tape_grad

        x = np.array([0.6, 0.8, 0.0])
        counter = {"n": 0}
        grad = _tape_grad(self._traceable_energy(counter), x, lambda a: a)

        assert grad is not None
        np.testing.assert_allclose(grad, 2.0 * x)
        assert counter["n"] == 1

        raw_counter = {"n": 0}
        assert _tape_grad(
            self._raw_numpy_energy(raw_counter), x, lambda a: a
        ) is None


def test_tape_grad_sums_cotangents_from_every_alias():
    """PR #490 review (P1): aliased views must accumulate, not first-match-win.

    `Multivector.coefficients` returns a fresh whole-array view on every access,
    so an energy that reads the state twice registers two distinct tape ids,
    each holding its own half of dE/dx. Returning the first match gave a
    gradient of 1 where the true value is 2 — a silently wrong Langevin step,
    not an error, which is precisely the failure class this work exists to
    remove.
    """
    import numpy as np

    from tessera import ops
    from tessera.ebm.geo_sampling import _numerical_grad_mv, _tape_grad_mv
    from tessera.ga.multivector import Multivector
    from tessera.ga.signature import Cl

    algebra = Cl(3, 0)
    state = Multivector(np.arange(1.0, algebra.dim + 1.0), algebra)

    def aliased_energy(x):
        a = x.coefficients          # view #1
        b = x.coefficients          # view #2 — a different object
        return ops.reduce(ops.add(a, b), op="sum")   # E = sum(2c) -> dE/dc = 2

    grad = _tape_grad_mv(aliased_energy, state)
    assert grad is not None
    np.testing.assert_allclose(grad, np.full(algebra.dim, 2.0))

    # And it agrees with the numerical fallback, which never had this bug.
    np.testing.assert_allclose(
        grad, _numerical_grad_mv(aliased_energy, state), rtol=1e-4, atol=1e-4
    )
