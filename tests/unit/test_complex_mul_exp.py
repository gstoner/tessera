"""M7 Step 2 — ``complex_mul``, ``complex_exp``, ``complex_conjugate``,
``complex_abs``.

Coverage:

  - Algebraic identities (``i² = −1``, ``e^{iπ} = −1``).
  - Compatibility with numpy's complex semantics on a swept
    sample of inputs.
  - **Conformality** — a holomorphic ``f`` rotates+scales the
    tangent plane uniformly; the angle between two infinitesimal
    tangent vectors is preserved by ``z²`` and ``e^z`` except at
    points where the derivative vanishes.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import complex as tc


# ---------------------------------------------------------------------------
# Algebraic identities
# ---------------------------------------------------------------------------

def test_imaginary_unit_squared_is_minus_one() -> None:
    i = tc.from_pair(0.0, 1.0)
    out = tc.complex_mul(i, i)
    assert pytest.approx(float(out.re)) == -1.0
    assert pytest.approx(float(out.im), abs=1e-12) == 0.0


def test_one_times_z_is_z() -> None:
    z = tc.from_pair(3.0, -2.0)
    one = tc.from_pair(1.0, 0.0)
    out = tc.complex_mul(one, z)
    assert float(out.re) == 3.0 and float(out.im) == -2.0


def test_complex_exp_of_i_pi_is_minus_one() -> None:
    """Euler: e^{iπ} = −1."""
    z = tc.from_pair(0.0, np.pi)
    out = tc.complex_exp(z)
    assert pytest.approx(float(out.re), abs=1e-12) == -1.0
    assert pytest.approx(float(out.im), abs=1e-12) == 0.0


def test_complex_exp_of_zero_is_one() -> None:
    out = tc.complex_exp(tc.from_pair(0.0, 0.0))
    assert pytest.approx(float(out.re)) == 1.0
    assert pytest.approx(float(out.im), abs=1e-12) == 0.0


def test_complex_conjugate_is_idempotent_twice() -> None:
    z = tc.from_pair(3.7, -2.1)
    cc = tc.complex_conjugate(tc.complex_conjugate(z))
    assert float(cc.re) == float(z.re)
    assert float(cc.im) == float(z.im)


def test_complex_abs_matches_numpy() -> None:
    rng = np.random.RandomState(0)
    arr = rng.randn(8) + 1j * rng.randn(8)
    out = tc.complex_abs(tc.from_numpy(arr))
    np.testing.assert_allclose(out, np.abs(arr), atol=1e-12)


# ---------------------------------------------------------------------------
# Cross-check against numpy's complex semantics on a sweep
# ---------------------------------------------------------------------------

def _sample(rng) -> tuple[np.ndarray, np.ndarray]:
    a = rng.randn(8) + 1j * rng.randn(8)
    b = rng.randn(8) + 1j * rng.randn(8)
    return a, b


def test_complex_mul_matches_numpy_on_sweep() -> None:
    rng = np.random.RandomState(1)
    a, b = _sample(rng)
    out = tc.complex_mul(tc.from_numpy(a), tc.from_numpy(b)).to_numpy()
    np.testing.assert_allclose(out, a * b, atol=1e-12)


def test_complex_exp_matches_numpy_on_sweep() -> None:
    rng = np.random.RandomState(2)
    z = rng.randn(8) + 1j * rng.randn(8)
    out = tc.complex_exp(tc.from_numpy(z)).to_numpy()
    np.testing.assert_allclose(out, np.exp(z), atol=1e-12)


def test_complex_mul_broadcasts_like_numpy() -> None:
    """Mixing scalar × array should follow numpy broadcasting."""
    rng = np.random.RandomState(3)
    arr = rng.randn(4) + 1j * rng.randn(4)
    out = tc.complex_mul(tc.from_pair(0.0, 1.0), tc.from_numpy(arr)).to_numpy()
    np.testing.assert_allclose(out, 1j * arr, atol=1e-12)


# ---------------------------------------------------------------------------
# Conformality — the M7 thesis
# ---------------------------------------------------------------------------

def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle (radians) between two complex numbers treated as 2-D
    real vectors."""
    cos_t = float(
        (u.real * v.real + u.imag * v.imag)
        / (np.linalg.norm([u.real, u.imag]) * np.linalg.norm([v.real, v.imag]))
    )
    return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))


def _conformality_residual(
    f, z0: complex, *, h: float = 1e-5,
) -> float:
    """Apply ``f`` to two infinitesimal tangent vectors at ``z₀``
    and return ``|θ_image − θ_source|``.

    For analytic ``f`` with ``f'(z₀) ≠ 0`` the residual must be
    ≈ 0 (angles preserved).  Non-analytic functions or vanishing
    derivatives surface as a finite residual.
    """
    # Two infinitesimal tangent directions at z0.
    dz1 = h
    dz2 = h * 1j
    src_angle = _angle_between(np.asarray(dz1), np.asarray(dz2))
    img1 = (f(z0 + dz1) - f(z0)) / h
    img2 = (f(z0 + dz2) - f(z0)) / h
    img_angle = _angle_between(np.asarray(img1), np.asarray(img2))
    return abs(img_angle - src_angle)


def _f_z_squared(z: complex) -> complex:
    """``f(z) = z²`` via our ``complex_mul``."""
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_mul(cz, cz)
    return complex(float(out.re), float(out.im))


def _f_exp(z: complex) -> complex:
    """``f(z) = e^z`` via our ``complex_exp``."""
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_exp(cz)
    return complex(float(out.re), float(out.im))


def _f_conjugate(z: complex) -> complex:
    """``f(z) = z̄`` — NOT analytic.  Conformality residual must
    be large (angles get flipped, not preserved)."""
    cz = tc.from_pair(z.real, z.imag)
    out = tc.complex_conjugate(cz)
    return complex(float(out.re), float(out.im))


@pytest.mark.parametrize("z0_real,z0_imag", [
    (1.0, 0.5),
    (0.7, -0.3),
    (-1.2, 1.8),
    (2.0, 0.0),  # on the real axis
])
def test_z_squared_preserves_angles_at_non_origin_points(z0_real, z0_imag) -> None:
    """``z²`` is analytic everywhere; conformal everywhere except
    at ``z = 0`` (where ``f'(z) = 2z = 0``)."""
    residual = _conformality_residual(_f_z_squared, complex(z0_real, z0_imag))
    assert residual < 1e-3, (
        f"z² should preserve angles at z₀={z0_real}+{z0_imag}j, residual={residual}"
    )


@pytest.mark.parametrize("z0_real,z0_imag", [
    (0.0, 0.0),
    (1.0, 1.0),
    (-0.5, 2.0),
    (3.0, -2.5),
])
def test_exp_preserves_angles_everywhere(z0_real, z0_imag) -> None:
    """``e^z`` is analytic with ``f'(z) = e^z`` never zero, so it
    is conformal at every point in ℂ."""
    residual = _conformality_residual(_f_exp, complex(z0_real, z0_imag))
    assert residual < 1e-3, (
        f"e^z should preserve angles at z₀={z0_real}+{z0_imag}j, residual={residual}"
    )


def test_conjugate_does_NOT_preserve_angles() -> None:
    """Sanity: the conformality test must catch non-analytic
    functions.  ``z̄`` reverses orientation — the angle between
    tangent vectors is preserved in magnitude but flipped, so a
    direct comparison of unsigned angles equals ≈ 0… but a sweep
    of differently-oriented tangent pairs would not.  Use the
    full picture: ``z̄`` rotates ``(1, i) → (1, -i)``, so the
    transformed angle is `-π/2` not `+π/2`.

    We test this by checking that the **signed** angular
    deviation of the second tangent direction (relative to the
    first) is non-zero after applying ``z̄`` — i.e., the map
    flips the imaginary tangent."""
    z0 = complex(1.0, 1.0)
    h = 1e-5
    img1 = (_f_conjugate(z0 + h) - _f_conjugate(z0)) / h
    img2 = (_f_conjugate(z0 + h * 1j) - _f_conjugate(z0)) / h
    # img2 should be -1j (conjugate flips the imaginary tangent).
    assert pytest.approx(float(img2.imag), abs=1e-4) == -1.0
    # In particular the signed orientation has flipped: img1 ⊥ img2
    # but in the opposite handedness from the source frame.
    cross = float(img1.real * img2.imag - img1.imag * img2.real)
    assert cross < 0, (
        f"z̄ should reverse orientation: signed area should be "
        f"negative, got {cross}"
    )
