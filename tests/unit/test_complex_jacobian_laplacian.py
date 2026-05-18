"""M7 Step 5 — ``conformal_jacobian`` + ``laplacian_2d``.

Coverage:

  - ``conformal_jacobian`` for ``z²`` at ``z₀`` returns
    ``(|2z₀|, arg(2z₀))``.
  - ``conformal_jacobian`` for ``e^z`` at ``z₀`` returns
    ``(|e^z₀|, Im(z₀))``.
  - ``laplacian_2d`` is ≈ 0 for harmonic samples (real / imaginary
    parts of an analytic function on a grid).
  - ``laplacian_2d`` is finite + non-zero for non-harmonic input
    (e.g., ``|z|²``, which is not harmonic — its Laplacian is 4).
  - 1-D / 3-D inputs are rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import complex as tc


# ---------------------------------------------------------------------------
# conformal_jacobian — analytic functions
# ---------------------------------------------------------------------------

def _f_z_squared(z: complex) -> complex:
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_mul(cz, cz)
    return complex(float(out.re), float(out.im))


def _f_exp(z: complex) -> complex:
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_exp(cz)
    return complex(float(out.re), float(out.im))


@pytest.mark.parametrize("z0", [1 + 0.5j, 0.7 - 0.3j, -1.2 + 1.8j, 2.0 + 0.0j])
def test_jacobian_of_z_squared_matches_2z(z0) -> None:
    """f'(z) = 2z, so |J| = |2z| and ∠J = arg(2z) = arg(z)."""
    scale, angle = tc.conformal_jacobian(_f_z_squared, z0)
    expected_scale = abs(2 * z0)
    expected_angle = float(np.angle(2 * z0))
    assert pytest.approx(scale, rel=1e-3) == expected_scale
    # Normalize angle to (-π, π].
    angle_norm = ((angle - expected_angle + np.pi) % (2 * np.pi)) - np.pi
    assert abs(angle_norm) < 1e-3


@pytest.mark.parametrize("z0", [0 + 0j, 1 + 1j, -0.5 + 2j])
def test_jacobian_of_exp_matches_exp(z0) -> None:
    """f'(z) = e^z, so |J| = e^(Re z), ∠J = Im z."""
    scale, angle = tc.conformal_jacobian(_f_exp, z0)
    expected_scale = float(np.exp(z0.real))
    expected_angle = float(z0.imag)
    assert pytest.approx(scale, rel=1e-3) == expected_scale
    # Wrap into (-π, π] for comparison.
    delta = ((angle - expected_angle + np.pi) % (2 * np.pi)) - np.pi
    assert abs(delta) < 1e-3


# ---------------------------------------------------------------------------
# laplacian_2d — harmonic vs non-harmonic
# ---------------------------------------------------------------------------

def _grid(n: int, half_width: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``n × n`` grid of (x, y) coordinates on [−half_width, +half_width]."""
    axis = np.linspace(-half_width, half_width, n)
    return np.meshgrid(axis, axis, indexing="xy")


def test_laplacian_of_harmonic_function_is_zero() -> None:
    """The real and imaginary parts of ``f(z) = z²`` are
    ``u(x,y) = x² − y²`` and ``v(x,y) = 2xy`` — both harmonic.

    Run ``laplacian_2d`` on each and check the interior is ≈ 0
    within the finite-difference tolerance."""
    x, y = _grid(32, half_width=2.0)
    u = x * x - y * y
    v = 2 * x * y
    lap_u = tc.laplacian_2d(u)
    lap_v = tc.laplacian_2d(v)
    # Interior (avoid boundary).
    inner = (slice(2, -2), slice(2, -2))
    assert float(np.abs(lap_u[inner]).max()) < 1e-9
    assert float(np.abs(lap_v[inner]).max()) < 1e-9


def test_laplacian_of_real_part_of_exp_is_zero() -> None:
    """``Re(e^z) = e^x · cos(y)``.  Harmonic everywhere."""
    x, y = _grid(64, half_width=1.0)
    u = np.exp(x) * np.cos(y)
    dx = 2.0 / 63
    lap = tc.laplacian_2d(u, dx=dx)
    inner = (slice(2, -2), slice(2, -2))
    assert float(np.abs(lap[inner]).max()) < 1e-2


def test_laplacian_of_modulus_squared_is_4() -> None:
    """``|z|² = x² + y²`` is NOT harmonic.  Its Laplacian is the
    constant ``4`` (= ∂²/∂x² + ∂²/∂y² of ``x² + y²``)."""
    x, y = _grid(64, half_width=1.0)
    field = x * x + y * y
    dx = 2.0 / 63
    lap = tc.laplacian_2d(field, dx=dx)
    inner = (slice(2, -2), slice(2, -2))
    np.testing.assert_allclose(lap[inner], 4.0, atol=1e-9)


def test_laplacian_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="expected 2-D"):
        tc.laplacian_2d(np.zeros(8))
    with pytest.raises(ValueError, match="expected 2-D"):
        tc.laplacian_2d(np.zeros((2, 3, 4)))


# ---------------------------------------------------------------------------
# Cross-product: a non-analytic function's "Jacobian" doesn't
# match the conformal decomposition we get from the analytic
# central-difference estimator
# ---------------------------------------------------------------------------

def test_conformal_jacobian_for_conjugate_is_path_dependent() -> None:
    """``z̄`` is not analytic; central differences along the real
    axis give a different ``df/dz`` than central differences along
    the imaginary axis.  We use this property in Step 6 to detect
    non-holomorphic functions; here we just verify the central
    difference is not a true derivative."""
    def f_conj(z: complex) -> complex:
        cz = tc.from_pair(z.real, z.imag)
        out = tc.complex_conjugate(cz)
        return complex(float(out.re), float(out.im))
    h = 1e-5
    z0 = 1 + 1j
    real_dir = (f_conj(z0 + h) - f_conj(z0 - h)) / (2 * h)
    imag_dir = (f_conj(z0 + h * 1j) - f_conj(z0 - h * 1j)) / (2j * h)
    # For a holomorphic function these would be equal.  For
    # conjugate they differ by a sign: real_dir = +1, imag_dir = -1.
    assert pytest.approx(real_dir, abs=1e-5) == 1 + 0j
    assert pytest.approx(imag_dir, abs=1e-5) == -1 + 0j
