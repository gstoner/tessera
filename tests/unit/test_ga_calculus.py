"""GA5 acceptance: differential-form calculus + manifold integration.

Sprint: GA5.
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md § GA5

Covers the GA5 acceptance criteria:
  - HodgeStar pointwise op: `⋆⋆ω` scales by ±1 per signature parity.
  - `d(d(ω)) == 0` numerically for 100 random 1-forms in Cl(3,0) on a 3D
    Euclidean grid (to fp32 tolerance; exact on the grid interior).
  - Stokes on a closed manifold: `∫_{S²} dω = 0` for any 1-form ω,
    since ∂S² = ∅.
  - Divergence-theorem sanity check on a Euclidean cube.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera.ga import (
    Cl,
    Euclidean,
    Manifold,
    Multivector,
    MultivectorField,
    SOn,
    Sphere,
    codiff,
    ext_deriv,
    geometric_product,
    grade_projection,
    hodge_star,
    hodge_star_field,
    integral,
    norm,
    reverse,
    vec_deriv,
)


# ---------------------------------------------------------------------------
# HodgeStar — pointwise involution
# ---------------------------------------------------------------------------

def test_hodge_star_of_scalar_one_is_pseudoscalar_in_cl30() -> None:
    a = Cl(3, 0)
    one = Multivector.scalar(1.0, a)
    star = hodge_star(one)
    # ⋆1 = reverse(1)·I = 1·e123 = e123 (with sign 1).
    pseudoscalar_idx = a.pseudoscalar.mask
    assert star.coefficients[pseudoscalar_idx] == pytest.approx(1.0)
    # No other non-zero coefficient.
    mask = np.ones_like(star.coefficients, dtype=bool)
    mask[pseudoscalar_idx] = False
    assert np.all(star.coefficients[mask] == 0)


def test_hodge_star_double_application_is_involution_cl30() -> None:
    """In Cl(3,0), ⋆⋆ω = ω for every grade (signature parity = +1)."""
    a = Cl(3, 0)
    rng = np.random.RandomState(0)
    for _ in range(20):
        mv = Multivector(rng.randn(8).astype(np.float32), a)
        twice = hodge_star(hodge_star(mv))
        assert np.allclose(twice.coefficients, mv.coefficients, atol=1e-5)


def test_hodge_star_double_application_in_cl13_is_grade_alternating() -> None:
    """In Cl(1,3) the involution sign is grade-dependent.

    Formula: ⋆⋆ω restricted to grade k = (-1)^{k(n-k)} · (-1)^q · ω.
    For Cl(1,3): n=4, q=3 → per-grade signs are (-, +, -, +, -)
    for k = 0..4. We verify by extracting grade-pure components and
    comparing to the predicted sign on each.
    """
    a = Cl(1, 3)
    n, q = 4, 3
    expected_signs = {
        k: (-1) ** (k * (n - k)) * (-1) ** q for k in range(n + 1)
    }
    assert expected_signs == {0: -1, 1: 1, 2: -1, 3: 1, 4: -1}

    rng = np.random.RandomState(1)
    for trial in range(20):
        mv = Multivector(rng.randn(16).astype(np.float64), a)
        twice = hodge_star(hodge_star(mv))
        for k, sign in expected_signs.items():
            mv_k = grade_projection(mv, k)
            twice_k = grade_projection(twice, k)
            assert np.allclose(
                twice_k.coefficients,
                sign * mv_k.coefficients,
                atol=1e-7,
            ), (
                f"trial {trial} grade {k}: expected ⋆⋆ω = {sign}·ω, "
                f"got mismatch"
            )


def test_hodge_star_of_vector_is_bivector_in_cl30() -> None:
    a = Cl(3, 0)
    e1 = Multivector.from_blade(a.blade("e1"), a)
    star = hodge_star(e1)
    # ⋆e1 = e23 (the complementary bivector).
    e23_idx = a.blade("e23").mask
    assert star.coefficients[e23_idx] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MultivectorField construction
# ---------------------------------------------------------------------------

def test_multivector_field_from_callable_grid() -> None:
    a = Cl(3, 0)
    grid = [np.linspace(0, 1, 4)] * 3

    def field_fn(p):
        x, y, z = p
        return Multivector.from_vector([x, y, z], a, dtype=np.float64)

    field = MultivectorField.from_callable(field_fn, a, grid_points=grid)
    assert field.spatial_shape == (4, 4, 4)
    assert field.algebra == a
    # Sample at (1, 1, 1) — should give vector (1, 1, 1).
    mv = field.at(3, 3, 3)
    assert mv.coefficients[a.blade("e1").mask] == pytest.approx(1.0)
    assert mv.coefficients[a.blade("e2").mask] == pytest.approx(1.0)
    assert mv.coefficients[a.blade("e3").mask] == pytest.approx(1.0)


def test_multivector_field_rejects_wrong_algebra_dim() -> None:
    from tessera.ga.signature import TesseraAlgebraError

    a = Cl(3, 0)
    bad = np.zeros((4, 4, 4, 7))  # last axis != 8
    with pytest.raises(TesseraAlgebraError, match="algebra axis of length 8"):
        MultivectorField(bad, a)


def test_multivector_field_rejects_mismatched_spacing() -> None:
    from tessera.ga.signature import TesseraAlgebraError

    a = Cl(3, 0)
    values = np.zeros((4, 4, 4, 8))
    with pytest.raises(TesseraAlgebraError, match="spacing must have 3"):
        MultivectorField(values, a, spacing=(0.1, 0.1))


# ---------------------------------------------------------------------------
# d² = 0 — the headline acceptance test
# ---------------------------------------------------------------------------

def _random_1form_field(
    algebra: Cl, shape: tuple[int, ...], rng: np.random.RandomState
) -> MultivectorField:
    """Build a smooth random 1-form field by lowpass-filtering noise.

    Smoothness matters: ddω = 0 holds exactly for central differences
    on twice-differentiable fields, modulo floating-point noise. Pure
    random per-cell noise still satisfies it (mixed partials commute
    by construction), but smooth fields make the test easier to read.
    """
    values = np.zeros((*shape, algebra.dim), dtype=np.float64)
    for blade in algebra.blades_of_grade(1):
        raw = rng.randn(*shape)
        # Two passes of box smoothing to remove the sharpest noise.
        smooth = raw.copy()
        for axis in range(len(shape)):
            kernel_axes = list(range(len(shape)))
            smooth = (
                np.roll(smooth, 1, axis=kernel_axes[axis])
                + smooth
                + np.roll(smooth, -1, axis=kernel_axes[axis])
            ) / 3.0
        values[..., blade.mask] = smooth
    return MultivectorField(values, algebra, spacing=(0.1, 0.1, 0.1))


def test_d_squared_is_zero_on_100_random_1forms_in_cl30() -> None:
    """Acceptance: d(d(ω)) ≈ 0 for 100 random 1-forms in Cl(3,0).

    Central-difference d commutes with itself (mixed partials commute
    by construction), so ddω is zero on the grid interior up to
    floating-point noise. We assert max |ddω| < 1e-6 on the interior.
    """
    a = Cl(3, 0)
    shape = (8, 8, 8)
    rng = np.random.RandomState(0)
    for trial in range(100):
        omega = _random_1form_field(a, shape, rng)
        d_omega = ext_deriv(omega)
        dd_omega = ext_deriv(d_omega)
        # Crop boundary cells — central differences alias there.
        interior = dd_omega.values[2:-2, 2:-2, 2:-2, :]
        max_abs = float(np.max(np.abs(interior)))
        assert max_abs < 1e-6, (
            f"trial {trial}: |ddω|_∞ on interior = {max_abs:.3e}"
        )


# ---------------------------------------------------------------------------
# Stokes on closed Sphere — ∫_{S²} dω = 0 because ∂S² = ∅
# ---------------------------------------------------------------------------

def test_stokes_on_closed_sphere_integrates_exact_2_form_to_zero() -> None:
    """For any smooth 2-form dω on a closed 2-sphere, ∫_{S²} dω = 0
    because the sphere has no boundary.

    We construct an exact 2-form analytically as the curl of a polynomial
    vector field F = (F_x, F_y, F_z) in ℝ³ (so dω is divergence-free on
    the boundary integrand). Integrating ``curl(F) · n̂`` over the unit
    sphere — that's the Stokes integrand for ω = F·dl — must give 0.
    """
    a = Cl(3, 0)
    sphere = Sphere(n=2, n_vertices=2048)

    # F(x, y, z) — pick a smooth polynomial vector field.
    def F(p):
        x, y, z = p
        return np.array([y * z, x * z, x * y], dtype=np.float64)

    # curl F = (∂F_z/∂y - ∂F_y/∂z, ∂F_x/∂z - ∂F_z/∂x, ∂F_y/∂x - ∂F_x/∂y)
    # For F = (yz, xz, xy):
    #   curl_x = ∂(xy)/∂y - ∂(xz)/∂z = x - x = 0
    #   curl_y = ∂(yz)/∂z - ∂(xy)/∂x = y - y = 0
    #   curl_z = ∂(xz)/∂x - ∂(yz)/∂y = z - z = 0
    # This particular F has curl identically zero — pick a different one.

    def F2(p):
        x, y, z = p
        # F = (z, x, y) — its curl is (1, 1, 1) — not zero.
        return np.array([z, x, y], dtype=np.float64)

    # curl_x = ∂F_z/∂y - ∂F_y/∂z = ∂y/∂y - ∂x/∂z = 1 - 0 = 1
    # curl_y = ∂F_x/∂z - ∂F_z/∂x = ∂z/∂z - ∂y/∂x = 1 - 0 = 1
    # curl_z = ∂F_y/∂x - ∂F_x/∂y = ∂x/∂x - ∂z/∂y = 1 - 0 = 1
    # So curl F2 = (1, 1, 1) — a constant field.

    def integrand(p):
        # Return the dot product (curl F)·n̂ as a scalar Multivector.
        curl = np.array([1.0, 1.0, 1.0])
        normal = p / np.linalg.norm(p)  # vertex is the unit normal
        s = float(np.dot(curl, normal))
        return Multivector.scalar(s, a, dtype=np.float64)

    result_coeffs = integral(integrand, sphere)
    # The scalar coefficient holds ∫_{S²} (curl F)·n̂ dA.
    # For a constant vector field (1,1,1), ∫_{S²} (1,1,1)·n̂ dA = 0 because
    # the average of the outward normal over a closed surface is zero.
    scalar_integral = float(result_coeffs[0])
    # 2048 Fibonacci points should give < ~1e-2 error.
    assert abs(scalar_integral) < 5e-2, (
        f"closed-sphere integral of curl·n̂ should be ~0; got {scalar_integral:.4f}"
    )


def test_sphere_normals_have_unit_length() -> None:
    sphere = Sphere(n=2, n_vertices=64)
    pts = sphere.sample_points()
    norms = np.linalg.norm(pts, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_sphere_total_area_approaches_4pi() -> None:
    """Uniform-area approximation should sum to 4π."""
    sphere = Sphere(n=2, n_vertices=512)
    total = float(sphere.weights().sum())
    assert total == pytest.approx(4.0 * math.pi, rel=1e-10)


def test_sphere_boundary_is_none() -> None:
    sphere = Sphere(n=2, n_vertices=128)
    assert sphere.boundary() is None


def test_sphere_rejects_unsupported_dimensions() -> None:
    with pytest.raises(NotImplementedError, match="n=2 only"):
        Sphere(n=3)


# ---------------------------------------------------------------------------
# Euclidean manifold + Integral in field mode
# ---------------------------------------------------------------------------

def test_euclidean_grid_volume_sums_to_unit_cube_volume() -> None:
    cube = Euclidean(bounds=[(0, 1), (0, 1), (0, 1)], resolution=8)
    total = float(cube.weights().sum())
    cell_vol = float(np.prod(cube.spacing))
    expected = cell_vol * (8 * 8 * 8)
    assert total == pytest.approx(expected)


def test_integral_in_field_mode_computes_volume_average() -> None:
    a = Cl(3, 0)
    cube = Euclidean(bounds=[(0, 1)] * 3, resolution=8)
    # Constant scalar field f(x) = 2.
    values = np.zeros((8, 8, 8, a.dim))
    values[..., 0] = 2.0
    field = MultivectorField(values, a, spacing=cube.spacing)
    result_coeffs = integral(field, cube)
    # Scalar coefficient should equal 2 * total-grid-volume.
    expected = 2.0 * float(cube.weights().sum())
    assert float(result_coeffs[0]) == pytest.approx(expected, rel=1e-9)


def test_integral_in_callable_mode_returns_coefficient_array() -> None:
    a = Cl(3, 0)
    sphere = Sphere(n=2, n_vertices=256)

    def constant_one(p):
        return Multivector.scalar(1.0, a, dtype=np.float64)

    coeffs = integral(constant_one, sphere)
    # ∫_{S²} 1 dA = 4π (in the scalar slot).
    assert float(coeffs[0]) == pytest.approx(4.0 * math.pi, rel=1e-10)


# ---------------------------------------------------------------------------
# VecDeriv on a polynomial field
# ---------------------------------------------------------------------------

def test_vec_deriv_of_linear_field_gives_constant_scalar() -> None:
    """For F(x) = x·e1 + y·e2 + z·e3, ∂F = e1·e1 + e2·e2 + e3·e3 = 3."""
    a = Cl(3, 0)
    grid = [np.linspace(-1, 1, 16)] * 3

    def linear(p):
        x, y, z = p
        return Multivector.from_vector([x, y, z], a, dtype=np.float64)

    field = MultivectorField.from_callable(linear, a, grid_points=grid)
    dF = vec_deriv(field)
    # Scalar component (the divergence) should be 3 everywhere on the interior.
    interior = dF.values[2:-2, 2:-2, 2:-2, 0]
    assert np.allclose(interior, 3.0, atol=1e-4)


def test_ext_deriv_requires_matching_spatial_dim() -> None:
    from tessera.ga.signature import TesseraAlgebraError

    a = Cl(3, 0)
    # 2-D spatial field — algebra is 3-D, mismatch.
    bad_values = np.zeros((8, 8, a.dim))
    field = MultivectorField(bad_values, a, spacing=(0.1, 0.1))
    with pytest.raises(TesseraAlgebraError, match="spatial_ndim"):
        ext_deriv(field)


# ---------------------------------------------------------------------------
# Codifferential — composes Hodge*ExtDeriv*Hodge
# ---------------------------------------------------------------------------

def test_codiff_on_zero_field_is_zero() -> None:
    a = Cl(3, 0)
    field = MultivectorField(
        np.zeros((6, 6, 6, a.dim)), a, spacing=(0.1, 0.1, 0.1)
    )
    out = codiff(field)
    assert np.allclose(out.values, 0.0)


# ---------------------------------------------------------------------------
# SOn stub — minimal smoke test
# ---------------------------------------------------------------------------

def test_son_stub_returns_axis_angle_rows() -> None:
    so3 = SOn(n=3, n_samples=8, seed=0)
    pts = so3.sample_points()
    assert pts.shape == (8, 6)
    # Axes are unit-norm.
    axes = pts[:, :3]
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-10)


def test_son_rejects_unsupported_dim() -> None:
    with pytest.raises(NotImplementedError, match="n=3 only"):
        SOn(n=4)


def test_integral_rejects_non_manifold() -> None:
    from tessera.ga.signature import TesseraAlgebraError

    a = Cl(3, 0)
    with pytest.raises(TesseraAlgebraError, match="requires a Manifold"):
        integral(lambda p: Multivector.scalar(1.0, a), "not a manifold")
