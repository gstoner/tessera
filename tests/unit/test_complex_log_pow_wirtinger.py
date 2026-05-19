"""Bundle A — log / arg / pow (Needham Ch. 2) + Wirtinger derivatives
(Ch. 4-5).

Coverage:

  - **Branch-cut policy** is locked: ``arg`` in ``(-π, π]``;
    ``log(-1) = i·π``.  Matches NumPy.
  - ``log`` round-trips with ``exp`` on the principal branch.
  - ``complex_pow`` matches ``numpy.power`` and falls back to
    ``complex_mul`` for integer powers.
  - **Wirtinger derivatives** ``dz`` and ``dbar`` satisfy the
    holomorphic / non-holomorphic boundary:

      - For ``z²`` and ``e^z``: ``dbar = 0``, ``dz = f'(z)``.
      - For ``z̄``: ``dz = 0``, ``dbar = 1``.
      - For ``|z|²``: ``dz = z̄₀``, ``dbar = z₀``.

  - ``dbar`` returns the same magnitude as :func:`check_cauchy_riemann`
    on the same inputs (the existing M7 numerical CR verifier is
    just ``|dbar|``, and now we expose ``dbar`` as the primitive).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import complex as tc


# ---------------------------------------------------------------------------
# complex_arg — principal branch (-π, π]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z, expected", [
    (1 + 0j, 0.0),
    (1j, math.pi / 2),
    (-1 + 0j, math.pi),       # The branch cut: arg(-1) = +π.
    (-1j, -math.pi / 2),
    (1 + 1j, math.pi / 4),
])
def test_complex_arg_principal_branch(z, expected) -> None:
    out = float(tc.complex_arg(tc.from_numpy(z)))
    assert abs(out - expected) < 1e-12


def test_complex_arg_matches_numpy_angle() -> None:
    rng = np.random.RandomState(0)
    z = (rng.randn(16) + 1j * rng.randn(16)).astype(np.complex128)
    out = tc.complex_arg(tc.from_numpy(z))
    np.testing.assert_allclose(out, np.angle(z), atol=1e-12)


# ---------------------------------------------------------------------------
# complex_log — branch cut along negative real axis
# ---------------------------------------------------------------------------

def test_log_of_one_is_zero() -> None:
    out = tc.complex_log(tc.from_pair(1.0, 0.0))
    assert pytest.approx(float(out.re)) == 0.0
    assert pytest.approx(float(out.im)) == 0.0


def test_log_of_e_is_one() -> None:
    out = tc.complex_log(tc.from_pair(math.e, 0.0))
    assert pytest.approx(float(out.re)) == 1.0
    assert pytest.approx(float(out.im), abs=1e-12) == 0.0


def test_log_of_i_is_i_pi_over_2() -> None:
    out = tc.complex_log(tc.from_pair(0.0, 1.0))
    assert pytest.approx(float(out.re), abs=1e-12) == 0.0
    assert pytest.approx(float(out.im)) == math.pi / 2


def test_log_of_minus_one_is_i_pi() -> None:
    """Branch cut along the negative real axis: log(-1) = +i·π
    (NumPy / IEEE-754 convention)."""
    out = tc.complex_log(tc.from_pair(-1.0, 0.0))
    assert pytest.approx(float(out.re), abs=1e-12) == 0.0
    assert pytest.approx(float(out.im)) == math.pi


def test_log_then_exp_recovers_z_away_from_origin() -> None:
    rng = np.random.RandomState(1)
    z = (rng.randn(16) + 1j * rng.randn(16)).astype(np.complex128)
    z = z[np.abs(z) > 0.1]
    round_trip = tc.complex_exp(tc.complex_log(tc.from_numpy(z))).to_numpy()
    np.testing.assert_allclose(round_trip, z, atol=1e-10)


def test_log_matches_numpy_on_sweep() -> None:
    """The whole point of pinning to NumPy's branch cut is so
    `tc.complex_log` and `np.log` agree on a complex array."""
    rng = np.random.RandomState(2)
    z = (rng.randn(16) + 1j * rng.randn(16)).astype(np.complex128)
    z = z[np.abs(z) > 1e-6]
    out = tc.complex_log(tc.from_numpy(z)).to_numpy()
    np.testing.assert_allclose(out, np.log(z), atol=1e-12)


def test_log_of_zero_is_minus_inf_real_part() -> None:
    """log(0) is -inf + 0·i — matches numpy."""
    out = tc.complex_log(tc.from_pair(0.0, 0.0))
    assert math.isinf(float(out.re)) and float(out.re) < 0
    assert pytest.approx(float(out.im), abs=1e-12) == 0.0


# ---------------------------------------------------------------------------
# complex_pow
# ---------------------------------------------------------------------------

def test_pow_one_is_z() -> None:
    z = tc.from_pair(2.0, 3.0)
    out = tc.complex_pow(z, tc.from_pair(1.0, 0.0))
    assert pytest.approx(float(out.re)) == 2.0
    assert pytest.approx(float(out.im)) == 3.0


def test_pow_two_equals_complex_mul_z_z() -> None:
    rng = np.random.RandomState(3)
    z = (rng.randn() + 1j * rng.randn())
    p2 = tc.complex_pow(
        tc.from_numpy(z), tc.from_pair(2.0, 0.0),
    ).to_numpy()
    direct = (z * z)
    assert abs(p2 - direct) < 1e-9


def test_pow_half_is_sqrt() -> None:
    """``z^0.5`` matches ``np.sqrt(z)`` on the principal branch."""
    rng = np.random.RandomState(4)
    z = (rng.randn(8) + 1j * rng.randn(8)).astype(np.complex128)
    # Stay away from the cut for stability.
    z = z[z.real > -0.5]
    out = tc.complex_pow(
        tc.from_numpy(z), tc.from_pair(0.5, 0.0),
    ).to_numpy()
    np.testing.assert_allclose(out, np.sqrt(z), atol=1e-9)


def test_pow_zero_exponent_is_one() -> None:
    rng = np.random.RandomState(5)
    z = (rng.randn() + 1j * rng.randn())
    out = tc.complex_pow(
        tc.from_numpy(z), tc.from_pair(0.0, 0.0),
    ).to_numpy()
    assert abs(out - 1.0) < 1e-9


def test_pow_matches_numpy_power_on_sweep() -> None:
    rng = np.random.RandomState(6)
    z = (rng.randn(8) + 1j * rng.randn(8)).astype(np.complex128)
    z = z[z.real > -0.5]
    w = (rng.randn(8) + 1j * rng.randn(8) * 0.5).astype(np.complex128)[: z.size]
    out = tc.complex_pow(tc.from_numpy(z), tc.from_numpy(w)).to_numpy()
    np.testing.assert_allclose(out, np.power(z, w), atol=1e-7)


# ---------------------------------------------------------------------------
# Wirtinger derivatives — dz and dbar
# ---------------------------------------------------------------------------

def _f_z_squared(z):
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_mul(cz, cz)
    return complex(float(out.re), float(out.im))


def _f_exp(z):
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_exp(cz)
    return complex(float(out.re), float(out.im))


def _f_conjugate(z):
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_conjugate(cz)
    return complex(float(out.re), float(out.im))


def _f_abs_squared(z):
    """``|z|² = z · z̄`` — real-valued, non-holomorphic."""
    cz = tc.from_pair(z.real, z.imag)
    return complex(float(tc.complex_abs(cz)) ** 2, 0.0)


@pytest.mark.parametrize("z0", [1 + 0.5j, 0.7 - 0.3j, -1.2 + 1.8j])
def test_dz_of_z_squared_equals_2z(z0) -> None:
    """For holomorphic ``f(z) = z²``, the complex derivative is
    ``f'(z) = 2z`` — and Wirtinger ``dz`` agrees."""
    out = tc.dz(_f_z_squared, z0)
    expected = 2 * z0
    assert abs(out - expected) < 1e-3


@pytest.mark.parametrize("z0", [1 + 0.5j, 0.7 - 0.3j, -1.2 + 1.8j])
def test_dbar_of_z_squared_is_zero(z0) -> None:
    """``z²`` is holomorphic ⇒ ``∂/∂z̄ ≈ 0``."""
    out = tc.dbar(_f_z_squared, z0)
    assert abs(out) < 1e-3


@pytest.mark.parametrize("z0", [0 + 0j, 1 + 1j, -0.5 + 2j])
def test_dz_of_exp_equals_exp(z0) -> None:
    """For ``f(z) = e^z``, ``f'(z) = e^z``."""
    out = tc.dz(_f_exp, z0)
    expected = complex(math.exp(z0.real) * math.cos(z0.imag),
                       math.exp(z0.real) * math.sin(z0.imag))
    assert abs(out - expected) < 1e-3


@pytest.mark.parametrize("z0", [0 + 0j, 1 + 1j, -0.5 + 2j])
def test_dbar_of_exp_is_zero(z0) -> None:
    """``e^z`` is entire ⇒ ``∂/∂z̄ ≈ 0`` everywhere."""
    out = tc.dbar(_f_exp, z0)
    assert abs(out) < 1e-3


def test_dz_of_conjugate_is_zero() -> None:
    """``∂z̄/∂z = 0`` exactly — the canonical Wirtinger fact."""
    out = tc.dz(_f_conjugate, 1 + 1j)
    assert abs(out) < 1e-3


def test_dbar_of_conjugate_is_one() -> None:
    """``∂z̄/∂z̄ = 1`` exactly."""
    out = tc.dbar(_f_conjugate, 1 + 1j)
    assert abs(out - 1.0) < 1e-3


@pytest.mark.parametrize("z0", [1 + 1j, 0.5 - 0.5j, 2 + 0j])
def test_dz_of_abs_squared_is_conjugate(z0) -> None:
    """For ``f(z) = |z|² = z · z̄``: ``∂f/∂z = z̄``."""
    out = tc.dz(_f_abs_squared, z0)
    expected = z0.conjugate()
    # Use a slightly looser tolerance — this op has more numerical noise.
    assert abs(out - expected) < 1e-2


@pytest.mark.parametrize("z0", [1 + 1j, 0.5 - 0.5j, 2 + 0j])
def test_dbar_of_abs_squared_is_z(z0) -> None:
    """For ``f(z) = |z|²``: ``∂f/∂z̄ = z``."""
    out = tc.dbar(_f_abs_squared, z0)
    assert abs(out - z0) < 1e-2


def test_dbar_magnitude_matches_check_cauchy_riemann_residual() -> None:
    """The numerical CR verifier in :func:`check_cauchy_riemann`
    computes the *magnitude* of ``dbar(f, z₀)``.  Locking the
    invariant: ``|dbar(f, z₀)| == residual(check_cauchy_riemann)``.
    Bundle A's Wirtinger primitive is the natural building block
    underneath the verifier."""
    z0 = 1 + 0.5j
    _, residual = tc.check_cauchy_riemann(_f_conjugate, z0)
    dbar_mag = abs(tc.dbar(_f_conjugate, z0))
    # Both use the same h; ought to be very close.
    assert abs(dbar_mag - residual) < 1e-4


# ---------------------------------------------------------------------------
# Module surface — every new symbol is exported
# ---------------------------------------------------------------------------

def test_all_new_bundle_a_symbols_are_exported() -> None:
    for name in (
        "complex_arg", "complex_log", "complex_pow", "dz", "dbar",
    ):
        assert hasattr(tc, name), f"missing export: {name}"
        assert callable(getattr(tc, name))


def test_all_new_bundle_b_symbols_are_exported() -> None:
    for name in (
        "cross_ratio", "is_concyclic", "mobius_from_three_points",
    ):
        assert hasattr(tc, name), f"missing export: {name}"
        assert callable(getattr(tc, name))
