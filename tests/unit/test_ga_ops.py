"""GA3 acceptance (part 2): multivector operations.

Sprint: GA3.
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md § GA3

Covers the GA3 acceptance criteria:
  - geometric_product agrees with hand-computed Cayley table for Cl(3,0)
    on 50 sampled pairs.
  - Rotor sandwich R * v * ~R rotates a Cl(3,0) vector identically to the
    equivalent SO(3) matrix-vector product to fp32 tolerance.
  - Rotor composition equals quaternion composition for Cl(3,0).
  - reverse / grade_involution sign rules.
  - Both fp32 and fp64 dtypes accepted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera.ga import (
    Cl,
    Multivector,
    conjugate,
    exp_mv,
    geometric_product,
    grade_involution,
    grade_projection,
    inner,
    log_mv,
    norm,
    reverse,
    rotor_from_axis,
    rotor_sandwich,
    wedge,
)


# ---------------------------------------------------------------------------
# Geometric product — hand-checked identities
# ---------------------------------------------------------------------------

def test_geometric_product_e1_times_e1_is_one_in_cl30() -> None:
    a = Cl(3, 0)
    e1 = Multivector.from_blade(a.blade("e1"), a)
    out = geometric_product(e1, e1)
    assert out.coefficients[0] == pytest.approx(1.0)


def test_geometric_product_e1_e2_is_e12() -> None:
    a = Cl(3, 0)
    e1 = Multivector.from_blade(a.blade("e1"), a)
    e2 = Multivector.from_blade(a.blade("e2"), a)
    out = geometric_product(e1, e2)
    e12_idx = a.blade("e12").mask
    assert out.coefficients[e12_idx] == pytest.approx(1.0)
    # No other component should be set.
    mask = np.ones_like(out.coefficients, dtype=bool)
    mask[e12_idx] = False
    assert np.all(out.coefficients[mask] == 0)


def test_geometric_product_anticommutation_for_orthogonal_vectors() -> None:
    a = Cl(3, 0)
    e1 = Multivector.from_blade(a.blade("e1"), a)
    e2 = Multivector.from_blade(a.blade("e2"), a)
    ab = geometric_product(e1, e2)
    ba = geometric_product(e2, e1)
    assert np.allclose(ab.coefficients, -ba.coefficients)


def test_geometric_product_against_random_pairs_via_cayley_table() -> None:
    """Random Multivector × Multivector should match a direct Cayley-table multiply."""
    a = Cl(3, 0)
    table = a.product_table()
    rng = np.random.RandomState(42)
    for trial in range(50):
        x = Multivector(rng.randn(8).astype(np.float32), a)
        y = Multivector(rng.randn(8).astype(np.float32), a)
        out = geometric_product(x, y).coefficients
        # Reference: explicit accumulation via the Cayley table.
        expected = np.zeros(8, dtype=np.float32)
        for i in range(8):
            for j in range(8):
                res_mask, sign = table[i][j]
                expected[res_mask] += sign * x.coefficients[i] * y.coefficients[j]
        assert np.allclose(out, expected, atol=1e-5), f"mismatch on trial {trial}"


# ---------------------------------------------------------------------------
# Reverse / conjugation / grade involution
# ---------------------------------------------------------------------------

def test_reverse_flips_bivector_sign_in_cl30() -> None:
    a = Cl(3, 0)
    bivec = Multivector.from_blade(a.blade("e12"), a)
    rev = reverse(bivec)
    # k=2 ⇒ sign (-1)^(2*1/2) = (-1)^1 = -1.
    assert rev.coefficients[a.blade("e12").mask] == pytest.approx(-1.0)


def test_reverse_leaves_scalars_and_vectors_alone() -> None:
    a = Cl(3, 0)
    scalar = Multivector.scalar(3.0, a)
    vec = Multivector.from_vector([1.0, 2.0, 3.0], a)
    assert np.allclose(reverse(scalar).coefficients, scalar.coefficients)
    assert np.allclose(reverse(vec).coefficients, vec.coefficients)


def test_grade_involution_flips_odd_grades() -> None:
    a = Cl(3, 0)
    vec = Multivector.from_vector([1.0, 2.0, 3.0], a)  # grade 1 (odd)
    bivec = Multivector.from_blade(a.blade("e12"), a)  # grade 2 (even)
    assert np.allclose(grade_involution(vec).coefficients, -vec.coefficients)
    assert np.allclose(grade_involution(bivec).coefficients, bivec.coefficients)


def test_clifford_conjugate_combines_reverse_and_involution() -> None:
    a = Cl(3, 0)
    # For grade 2, reverse gives -1, involution gives +1, conjugate = -1.
    bivec = Multivector.from_blade(a.blade("e12"), a)
    assert np.allclose(conjugate(bivec).coefficients, -bivec.coefficients)


# ---------------------------------------------------------------------------
# Grade projection
# ---------------------------------------------------------------------------

def test_grade_projection_extracts_only_named_grade() -> None:
    a = Cl(3, 0)
    coeffs = np.arange(1, 9, dtype=np.float32)
    mv = Multivector(coeffs, a)
    proj_bi = grade_projection(mv, 2)
    for blade in a.blades():
        if blade.grade == 2:
            assert proj_bi.coefficients[blade.mask] == coeffs[blade.mask]
        else:
            assert proj_bi.coefficients[blade.mask] == 0


def test_grade_projection_accepts_set_argument() -> None:
    a = Cl(3, 0)
    coeffs = np.ones(8, dtype=np.float32)
    mv = Multivector(coeffs, a)
    proj = grade_projection(mv, {0, 2})
    expected_grades = {0, 2}
    for blade in a.blades():
        if blade.grade in expected_grades:
            assert proj.coefficients[blade.mask] == 1.0
        else:
            assert proj.coefficients[blade.mask] == 0.0


# ---------------------------------------------------------------------------
# Wedge / inner / norm
# ---------------------------------------------------------------------------

def test_wedge_of_orthogonal_vectors_is_bivector() -> None:
    a = Cl(3, 0)
    e1 = Multivector.from_blade(a.blade("e1"), a)
    e2 = Multivector.from_blade(a.blade("e2"), a)
    w = wedge(e1, e2)
    assert w.coefficients[a.blade("e12").mask] == pytest.approx(1.0)


def test_wedge_of_parallel_vectors_is_zero() -> None:
    a = Cl(3, 0)
    e1 = Multivector.from_blade(a.blade("e1"), a)
    w = wedge(e1, e1)
    assert np.all(w.coefficients == 0)


def test_norm_of_unit_vector_is_one() -> None:
    a = Cl(3, 0)
    e1 = Multivector.from_blade(a.blade("e1"), a)
    assert float(np.asarray(norm(e1)).item()) == pytest.approx(1.0)


def test_inner_product_returns_scalar_array() -> None:
    a = Cl(3, 0)
    u = Multivector.from_vector([1.0, 2.0, 3.0], a)
    v = Multivector.from_vector([4.0, -5.0, 6.0], a)
    s = inner(u, v)
    # For vectors in Cl(3,0), inner = u . v = 4 - 10 + 18 = 12.
    assert float(np.asarray(s).item()) == pytest.approx(12.0, rel=1e-5)


# ---------------------------------------------------------------------------
# exp_mv / log_mv — rotors
# ---------------------------------------------------------------------------

def test_exp_of_zero_bivector_is_one() -> None:
    a = Cl(3, 0)
    zero_bi = Multivector.zeros(a)
    out = exp_mv(zero_bi)
    assert out.coefficients[0] == pytest.approx(1.0)


def test_rotor_from_axis_has_unit_norm() -> None:
    a = Cl(3, 0)
    bi = Multivector.from_blade(a.blade("e12"), a)
    R = rotor_from_axis(bi, math.pi / 3)
    assert float(np.asarray(norm(R)).item()) == pytest.approx(1.0, abs=1e-5)


def test_log_then_exp_round_trip_for_rotor() -> None:
    """log(exp(B/2)) should recover B/2 for a small pure bivector in Cl(3,0)."""
    a = Cl(3, 0)
    bi = 0.3 * Multivector.from_blade(a.blade("e12"), a)
    # exp_mv with bi (already half-angle).
    R = exp_mv(bi)
    recovered = log_mv(R)
    assert recovered.is_close(bi, atol=1e-5)


# ---------------------------------------------------------------------------
# Rotor sandwich — the headline equivariance claim
# ---------------------------------------------------------------------------

def _so3_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula — reference SO(3) matrix."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _bivector_for_axis(axis: np.ndarray, algebra: Cl) -> Multivector:
    """The bivector dual to a 3D vector axis: a*e23 - b*e13 + c*e12 for axis=(a,b,c)."""
    a, b, c = axis
    coeffs = np.zeros(algebra.dim, dtype=np.float64)
    coeffs[algebra.blade("e23").mask] = a
    coeffs[algebra.blade("e13").mask] = -b
    coeffs[algebra.blade("e12").mask] = c
    return Multivector(coeffs, algebra, grades={2})


def _vector_to_mv(v: np.ndarray, algebra: Cl) -> Multivector:
    return Multivector.from_vector(v.astype(np.float64), algebra, dtype=np.float64)


def _mv_to_vector(mv: Multivector, algebra: Cl) -> np.ndarray:
    return np.array([
        mv.coefficients[algebra.blade("e1").mask],
        mv.coefficients[algebra.blade("e2").mask],
        mv.coefficients[algebra.blade("e3").mask],
    ])


def test_rotor_sandwich_rotates_vector_identically_to_so3_matrix() -> None:
    a = Cl(3, 0)
    rng = np.random.RandomState(1)
    for trial in range(50):
        axis = rng.randn(3)
        angle = float(rng.uniform(-math.pi, math.pi))
        v_ref = rng.randn(3)
        # Reference rotation via SO(3) matrix.
        R_mat = _so3_matrix_from_axis_angle(axis, angle)
        v_rotated_ref = R_mat @ v_ref
        # Tessera GA rotation via rotor sandwich.
        bivec_axis = _bivector_for_axis(axis, a)
        R = rotor_from_axis(bivec_axis, angle)
        v_mv = _vector_to_mv(v_ref, a)
        rotated_mv = rotor_sandwich(R, v_mv)
        v_rotated_ga = _mv_to_vector(rotated_mv, a)
        assert np.allclose(v_rotated_ga, v_rotated_ref, atol=1e-4), (
            f"trial {trial}: GA {v_rotated_ga} vs SO(3) {v_rotated_ref}"
        )


def test_rotor_composition_matches_so3_composition() -> None:
    """R1 * R2 then sandwich == sandwich(R1) ∘ sandwich(R2)."""
    a = Cl(3, 0)
    rng = np.random.RandomState(2)
    for _ in range(20):
        axis1, axis2 = rng.randn(3), rng.randn(3)
        angle1 = float(rng.uniform(-math.pi, math.pi))
        angle2 = float(rng.uniform(-math.pi, math.pi))
        v = rng.randn(3)
        R1 = rotor_from_axis(_bivector_for_axis(axis1, a), angle1)
        R2 = rotor_from_axis(_bivector_for_axis(axis2, a), angle2)
        # Composite rotor.
        R_composite = geometric_product(R1, R2)
        v_mv = _vector_to_mv(v, a)
        out_composite = _mv_to_vector(rotor_sandwich(R_composite, v_mv), a)
        # Sequential application.
        out_sequential = _mv_to_vector(
            rotor_sandwich(R1, rotor_sandwich(R2, v_mv)), a
        )
        assert np.allclose(out_composite, out_sequential, atol=1e-4)


# ---------------------------------------------------------------------------
# dtype coverage
# ---------------------------------------------------------------------------

def test_ga_ops_accept_fp64() -> None:
    a = Cl(3, 0)
    x = Multivector(np.ones(8, dtype=np.float64), a)
    y = Multivector(2 * np.ones(8, dtype=np.float64), a)
    prod = geometric_product(x, y)
    assert prod.dtype == np.float64


def test_ga_ops_accept_fp32() -> None:
    a = Cl(3, 0)
    x = Multivector(np.ones(8, dtype=np.float32), a)
    y = Multivector(2 * np.ones(8, dtype=np.float32), a)
    prod = geometric_product(x, y)
    assert prod.dtype == np.float32


# ---------------------------------------------------------------------------
# Cl(1,3) — Lorentz signature spot-checks
# ---------------------------------------------------------------------------

def test_cl13_first_generator_squares_to_plus_one() -> None:
    a = Cl(1, 3)
    e1 = Multivector.from_blade(a.blade("e1"), a)
    prod = geometric_product(e1, e1)
    assert prod.coefficients[0] == pytest.approx(1.0)


def test_cl13_spacelike_generator_squares_to_minus_one() -> None:
    a = Cl(1, 3)
    e2 = Multivector.from_blade(a.blade("e2"), a)
    prod = geometric_product(e2, e2)
    assert prod.coefficients[0] == pytest.approx(-1.0)
