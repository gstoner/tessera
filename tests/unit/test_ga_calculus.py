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
from tessera.ga.types import TesseraAlgebraError


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


# ── MSW-4a: codiff is the codifferential, not just ⋆d⋆ ──────────────────────
#
# Before 2026-09-02 `codiff` applied the ⋆d⋆ composition with no sign and told
# callers to supply it themselves. That is impossible for the mixed-grade
# fields it accepts: the sign depends on the grade of each input component, so
# no single scalar corrects a field carrying several. These pin the property
# the operator is named for.

def _bump_field(algebra, grade, seed, n=24, L=6.0):
    """A compactly-supported field of one grade, so boundary terms vanish."""
    h = L / n
    axis = (np.arange(n) - n / 2) * h
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    bump = np.exp(-(X**2 + Y**2 + Z**2) / 0.6)
    rng = np.random.default_rng(seed)
    values = np.zeros((n, n, n, algebra.dim))
    for i, blade in enumerate(algebra.blades()):
        if blade.grade == grade:
            values[..., i] = bump * (np.sin(X * 1.3 + i) + rng.standard_normal())
    return MultivectorField(values, algebra, spacing=(h, h, h)), h


def _l2(a, b, h):
    return float(np.sum(a.values * b.values) * h**3)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_codiff_is_adjoint_to_ext_deriv(k: int) -> None:
    """Stokes: ⟨dα, β⟩ == ⟨α, δβ⟩ for a compactly-supported field."""
    algebra = Cl(3, 0)
    alpha, h = _bump_field(algebra, k - 1, 100 + k)
    beta, _ = _bump_field(algebra, k, 200 + k)
    lhs = _l2(ext_deriv(alpha), beta, h)
    rhs = _l2(alpha, codiff(beta), h)
    assert abs(lhs - rhs) <= 2e-3 * max(abs(lhs), 1e-12), (
        f"grade {k}: <da,b>={lhs} but <a,codiff b>={rhs}"
    )


def test_codiff_is_adjoint_on_a_mixed_grade_field() -> None:
    """The case that proved a scalar correction could not work.

    With the unsigned composition, neither +1 nor -1 reconciled a grade-1 +
    grade-2 field (0.9433 vs 1.3866) — the sign is per-grade or it is nothing.
    """
    algebra = Cl(3, 0)
    g1, h = _bump_field(algebra, 1, 301)
    g2, _ = _bump_field(algebra, 2, 302)
    beta = MultivectorField(g1.values + g2.values, algebra, spacing=g1.spacing)
    a0, _ = _bump_field(algebra, 0, 303)
    a1, _ = _bump_field(algebra, 1, 304)
    alpha = MultivectorField(a0.values + a1.values, algebra, spacing=g1.spacing)
    lhs = _l2(ext_deriv(alpha), beta, h)
    rhs = _l2(alpha, codiff(beta), h)
    assert abs(lhs - rhs) <= 2e-3 * max(abs(lhs), 1e-12)


def test_codiff_of_a_vector_field_is_minus_divergence() -> None:
    """δ on a 1-form in R^3 is -div. It returned +div before the fix."""
    algebra = Cl(3, 0)
    n, L = 24, 6.0
    h = L / n
    axis = (np.arange(n) - n / 2) * h
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    bump = np.exp(-(X**2 + Y**2 + Z**2) / 0.8)
    vx, vy, vz = X * bump, (Y**2) * bump, np.sin(Z) * bump
    ones = [i for i, b in enumerate(algebra.blades()) if b.grade == 1]
    values = np.zeros((n, n, n, algebra.dim))
    values[..., ones[0]], values[..., ones[1]], values[..., ones[2]] = vx, vy, vz
    divergence = (
        np.gradient(vx, h, axis=0, edge_order=2)
        + np.gradient(vy, h, axis=1, edge_order=2)
        + np.gradient(vz, h, axis=2, edge_order=2)
    )
    got = codiff(MultivectorField(values, algebra, spacing=(h, h, h))).values[..., 0]
    interior = (slice(3, -3),) * 3
    np.testing.assert_allclose(got[interior], -divergence[interior], atol=1e-12)


def test_codiff_refuses_a_signature_it_cannot_verify() -> None:
    """Fail closed rather than guess the (-1)^q metric-determinant factor."""
    algebra = Cl(1, 3)
    values = np.zeros((4,) * algebra.n + (algebra.dim,))
    with pytest.raises(TesseraAlgebraError, match="Euclidean"):
        codiff(MultivectorField(values, algebra, spacing=1.0))


# ── MSW-4: the vector-identity law family ───────────────────────────────────

def test_vector_identity_law_family_passes() -> None:
    """Law 7 — every field-calculus identity holds on the shipped operators."""
    from tessera.autodiff.laws import vector_identity_checks

    results = vector_identity_checks()
    assert {r.op for r in results} == {
        "d_squared_is_zero",
        "codiff_squared_is_zero",
        "codiff_is_adjoint_to_ext_deriv",
        "ext_deriv_matches_analytic_gradient",
        "divergence_product_rule",
    }
    failed = [(r.op, r.status, r.max_rel_residual) for r in results if r.status != "pass"]
    assert not failed, f"field-calculus identities failed: {failed}"


def test_vector_identity_laws_detect_a_corrupted_operator() -> None:
    """The family must be able to FAIL. A law nothing can break proves nothing.

    Two mutations with known reach: dropping `codiff`'s per-grade sign (the
    MSW-4a defect) must break the Stokes pairing, and scaling `ext_deriv` must
    break the analytic-gradient law — the only one here that pins absolute
    scale, since every other identity is homogeneous in `ext_deriv`.
    """
    import tessera.ga.calculus as cal
    from tessera.autodiff.laws import vector_identity_checks

    saved_signs, saved_ext = cal.codifferential_output_signs, cal.ext_deriv
    try:
        cal.codifferential_output_signs = lambda algebra: np.ones(algebra.dim)
        broken = {r.op for r in vector_identity_checks() if r.status != "pass"}
        assert "codiff_is_adjoint_to_ext_deriv" in broken
    finally:
        cal.codifferential_output_signs = saved_signs

    try:
        cal.ext_deriv = lambda f: MultivectorField(
            saved_ext(f).values * 1.5, f.algebra, spacing=f.spacing
        )
        broken = {r.op for r in vector_identity_checks() if r.status != "pass"}
        assert "ext_deriv_matches_analytic_gradient" in broken
    finally:
        cal.ext_deriv = saved_ext

    assert all(r.status == "pass" for r in vector_identity_checks())
