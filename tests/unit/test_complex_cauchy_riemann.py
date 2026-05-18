"""M7 Step 6 — Cauchy-Riemann verifier.

Coverage:

  - ``z²``, ``e^z``, Möbius pass at every interior probe point.
  - ``z̄``, ``z·|z|`` fail with non-trivial residuals.
  - The ``@analytic`` decorator runs the check at the first call
    and raises :class:`NotHolomorphicError` on a non-holomorphic
    function.
  - The decorator caches the result (subsequent calls don't
    re-verify) and pins the offending probe point in the error.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import complex as tc


# ---------------------------------------------------------------------------
# Helpers — wrap each test function so it goes through tc.complex_*
# ---------------------------------------------------------------------------

def _f_z_squared(z: complex) -> complex:
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_mul(cz, cz)
    return complex(float(out.re), float(out.im))


def _f_exp(z: complex) -> complex:
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_exp(cz)
    return complex(float(out.re), float(out.im))


def _f_mobius(z: complex) -> complex:
    cz = tc.from_pair(z.real, z.imag)
    out = tc.mobius(cz, a=1.0, b=2.0, c=0.0, d=3.0)
    return complex(float(out.re), float(out.im))


def _f_conjugate(z: complex) -> complex:
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_conjugate(cz)
    return complex(float(out.re), float(out.im))


def _f_z_times_abs_z(z: complex) -> complex:
    """``f(z) = z · |z|`` — not analytic anywhere except z=0."""
    return z * abs(z)


# ---------------------------------------------------------------------------
# Positive: holomorphic functions pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z0", [0.5 + 0.3j, 1.0 - 0.7j, -1.2 + 0.4j, 2.1 + 1.9j])
def test_z_squared_passes_cauchy_riemann(z0) -> None:
    passes, residual = tc.check_cauchy_riemann(_f_z_squared, z0)
    assert passes, f"residual at z₀={z0}: {residual}"


@pytest.mark.parametrize("z0", [0 + 0j, 1 + 1j, -0.5 + 2j, 3 - 2.5j])
def test_exp_passes_cauchy_riemann(z0) -> None:
    passes, residual = tc.check_cauchy_riemann(_f_exp, z0)
    assert passes, f"residual at z₀={z0}: {residual}"


def test_mobius_passes_cauchy_riemann_away_from_pole() -> None:
    # Mobius (a=1, b=2, c=0, d=3) has no finite pole, so it's
    # entire — passes everywhere.
    for z0 in [0.5 + 0.3j, -1.0 + 0.5j, 2.0 + 1.0j]:
        passes, residual = tc.check_cauchy_riemann(_f_mobius, z0)
        assert passes, f"Mobius failed at z₀={z0}, residual={residual}"


# ---------------------------------------------------------------------------
# Negative: non-holomorphic functions fail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z0", [0.5 + 0.3j, 1.0 - 0.7j, -1.2 + 0.4j])
def test_conjugate_fails_cauchy_riemann(z0) -> None:
    """``z̄`` has residual ``|1 - 1| / 2 = 0``? No — ``∂z̄/∂x = 1``,
    ``∂z̄/∂y = -i``, so ``∂f/∂z̄ = (1 + i·(-i))/2 = 1``."""
    passes, residual = tc.check_cauchy_riemann(_f_conjugate, z0)
    assert not passes, f"z̄ should fail CR; got residual={residual}"
    assert residual > 0.9, f"z̄ residual should be ~1; got {residual}"


@pytest.mark.parametrize("z0", [0.5 + 0.3j, 1.0 - 0.7j, -1.2 + 0.4j])
def test_z_times_abs_z_fails_cauchy_riemann(z0) -> None:
    passes, residual = tc.check_cauchy_riemann(_f_z_times_abs_z, z0)
    assert not passes, f"z·|z| should fail CR; got residual={residual}"


def test_residual_is_nonzero_for_real_part_only() -> None:
    """``f(z) = Re(z)`` has ``∂f/∂x = 1``, ``∂f/∂y = 0`` so
    ``∂f/∂z̄ = 1/2`` — should fail."""
    def f_real(z: complex) -> complex:
        return complex(z.real, 0.0)
    passes, residual = tc.check_cauchy_riemann(f_real, 1 + 1j)
    assert not passes
    assert abs(residual - 0.5) < 1e-3


# ---------------------------------------------------------------------------
# @analytic decorator
# ---------------------------------------------------------------------------

def test_analytic_decorator_accepts_holomorphic_function() -> None:
    """``z²`` is holomorphic so the decorator is a no-op."""

    @tc.analytic
    def f(z):
        cz = tc.from_pair(z.real, z.imag)
        out = tc.complex_mul(cz, cz)
        return complex(float(out.re), float(out.im))

    # First call triggers verification + actual evaluation.
    result = f(1 + 1j)
    assert pytest.approx(result, abs=1e-9) == (1 + 1j) ** 2


def test_analytic_decorator_rejects_non_holomorphic_function() -> None:
    """``z̄`` must be rejected with a clear diagnostic."""

    @tc.analytic
    def f(z):
        cz = tc.from_pair(z.real, z.imag)
        out = tc.complex_conjugate(cz)
        return complex(float(out.re), float(out.im))

    with pytest.raises(tc.NotHolomorphicError, match="CR at z ="):
        f(1 + 1j)


def test_analytic_decorator_caches_after_first_call() -> None:
    """The verification runs once.  We confirm by mutating a
    counter inside the wrapped function and verifying it
    increments only on real calls, not on probe samples."""
    calls = [0]

    @tc.analytic(probes=3)
    def f(z):
        calls[0] += 1
        cz = tc.from_pair(z.real, z.imag)
        out = tc.complex_mul(cz, cz)
        return complex(float(out.re), float(out.im))

    # First call: probes count internally + 1 real call.
    f(0.5 + 0.5j)
    after_first = calls[0]
    # Subsequent call: no probes — just the real call.
    f(1 + 0j)
    f(2 + 1j)
    assert calls[0] - after_first == 2


def test_analytic_decorator_with_parameters_form() -> None:
    """``@analytic(probes=N, atol=T)`` parametrized usage."""

    @tc.analytic(probes=2, atol=1e-2)
    def f(z):
        cz = tc.from_pair(z.real, z.imag)
        return complex_to_native(tc.complex_exp(cz))

    def complex_to_native(out):
        return complex(float(out.re), float(out.im))

    out = f(0 + 0j)
    assert pytest.approx(out, abs=1e-9) == 1 + 0j


# ---------------------------------------------------------------------------
# Residual interpretation
# ---------------------------------------------------------------------------

def test_residual_matches_known_wirtinger_value_for_conjugate() -> None:
    """``∂z̄/∂z̄ = 1`` is exact — the residual should equal 1
    (up to finite-difference error)."""
    _, residual = tc.check_cauchy_riemann(_f_conjugate, 0.5 + 0.5j)
    assert abs(residual - 1.0) < 1e-3
