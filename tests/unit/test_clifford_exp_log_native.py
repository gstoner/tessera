"""Rotor sampling on the group: `exp`, `log` and `rotor_from_axis` natively.

Before this slice the Clifford dialect declared `exp` and `log` with no lowering
at all, so the only way to exponentiate a bivector was the numpy reference — the
gap the GA/EBM review names as "exp/log of multivectors (rotor sampling on the
group rather than the Lie algebra)". All three now lower to their closed forms on
Cl(3, 0) and execute through the MLIR/LLVM CPU JIT lane.

Agreement with the reference is a few ulp, not bit-exact, and the cause is
recorded rather than hidden: the reference's `norm` reduces the full 8x8 Cayley
table with numpy's own (vectorized, possibly pairwise) summation, while the
lowering emits an ordered scalar fold, and `cos`/`sin`/`atan2` come from libm on
both sides but not through the same call sequence. The metamorphic identities
below are the checks that do not depend on that: log(exp(B)) == B, |exp(B)| == 1,
exp(0) == 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera import _jit_boundary as jb
from tessera.ga import ops as ga
from tessera.ga.multivector import Multivector
from tessera.ga.signature import Cl

CL30 = Cl(3, 0, 0)
BIVECTOR_MASKS = (3, 5, 6)   # the grade-2 blades of Cl(3, 0): e12, e13, e23
TOLERANCE = 1e-6             # measured worst 5.4e-07 on the M1 Max


def _lane():
    if not jb.has_clifford():
        pytest.skip("libtessera_jit built without the Clifford lane")


def _bivector(rows: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((rows, 8), dtype=np.float32)
    for mask in BIVECTOR_MASKS:
        out[:, mask] = (rng.standard_normal(rows) * scale).astype(np.float32)
    return out


@pytest.mark.parametrize("scale", [0.01, 0.5, 1.0, 3.0])
def test_exp_of_a_bivector_matches_the_reference(scale):
    _lane()
    b = _bivector(6, scale, seed=int(scale * 100))
    got = jb.jit_clifford_op("exp", b)
    want = ga.exp_mv(Multivector(b.copy(), CL30)).to_numpy()
    assert np.max(np.abs(got - want)) < TOLERANCE


@pytest.mark.parametrize("scale", [0.01, 0.5, 1.0, 3.0])
def test_log_of_a_rotor_matches_the_reference(scale):
    _lane()
    rotor = np.ascontiguousarray(
        ga.exp_mv(Multivector(_bivector(6, scale, seed=7), CL30)).to_numpy(), dtype=np.float32)
    got = jb.jit_clifford_op("log", rotor)
    want = ga.log_mv(Multivector(rotor.copy(), CL30)).to_numpy()
    assert np.max(np.abs(got - want)) < TOLERANCE


def test_log_inverts_exp_on_the_group():
    """The identity that does not depend on either side's summation order."""
    _lane()
    b = _bivector(8, 0.7, seed=3)
    rotor = jb.jit_clifford_op("exp", b)
    assert np.max(np.abs(jb.jit_clifford_op("log", rotor) - b)) < TOLERANCE
    # exp(B) is a unit rotor: that is what makes it a group element.
    norms = np.asarray(ga.norm(Multivector(np.ascontiguousarray(rotor), CL30)))
    np.testing.assert_allclose(norms, 1.0, atol=TOLERANCE)


def test_the_zero_bivector_exponentiates_to_the_identity():
    """The reference guards the division by |B| below 1e-12 so exp(0) is 1, not
    NaN; the lowering carries the same guard."""
    _lane()
    out = jb.jit_clifford_op("exp", np.zeros((3, 8), dtype=np.float32))
    expect = np.zeros((3, 8), dtype=np.float32)
    expect[:, 0] = 1.0
    np.testing.assert_array_equal(out, expect)
    np.testing.assert_array_equal(jb.jit_clifford_op("log", np.zeros((2, 8), dtype=np.float32)),
                                  np.zeros((2, 8), dtype=np.float32))


def test_exp_refuses_an_operand_it_cannot_prove_is_a_bivector():
    """The reference switches to a 24-term power series when the operand is not a
    pure bivector, and that choice depends on the value. Rather than let the two
    take different branches for one input, the lane refuses."""
    _lane()
    mixed = _bivector(2, 1.0, seed=5)
    mixed[0, 0] = 0.25          # a scalar part makes it not a pure bivector
    with pytest.raises(jb.TesseraJitError, match="pure bivector"):
        jb.jit_clifford_op("exp", mixed)


@pytest.mark.parametrize("op", ["exp", "log"])
def test_the_closed_form_is_cl30_only(op):
    _lane()
    with pytest.raises(jb.TesseraJitError, match="Cl\\(3, 0\\)"):
        jb.jit_clifford_op(op, np.zeros((2, 4), dtype=np.float32), algebra=(2, 0, 0))


@pytest.mark.parametrize("angle", [0.0, 0.3, np.pi / 2, -2.2, 3.0])
def test_rotor_from_axis_matches_the_reference(angle):
    _lane()
    axis = _bivector(5, 1.0, seed=11)
    got = jb.jit_clifford_op("rotor_from_axis", axis, angle=float(angle))
    want = np.stack([ga.rotor_from_axis(Multivector(axis[i].copy(), CL30), float(angle)).to_numpy()
                     for i in range(axis.shape[0])])
    assert np.max(np.abs(got - want)) < TOLERANCE


def test_rotor_from_axis_refuses_a_degenerate_axis():
    """An axis with no grade-2 part has no rotor. Returning the identity for it
    would be a wrong answer, so neither the frontend nor the lowering invents
    one — the reference raises here too."""
    _lane()
    with pytest.raises(jb.TesseraJitError, match="non-zero grade-2 part"):
        jb.jit_clifford_op("rotor_from_axis", np.zeros((1, 8), dtype=np.float32), angle=1.0)


def test_rotor_from_axis_rotates_by_the_angle_it_was_given():
    """R x R† must turn a vector in the plane by exactly `angle`: the property
    the constructor exists for, checked without reference to its coefficients."""
    _lane()
    axis = np.zeros((1, 8), dtype=np.float32)
    axis[0, 3] = 1.0                       # the e12 plane
    vector = np.zeros((1, 8), dtype=np.float32)
    vector[0, 1] = 1.0                     # e1
    for angle in (0.5, np.pi / 3, 2.0):
        rotor = jb.jit_clifford_op("rotor_from_axis", axis, angle=float(angle))
        turned = jb.jit_clifford_op("rotor_sandwich", rotor, vector)
        # e1 -> cos(angle) e1 + sin(angle) e2 for a rotation in the e12 plane.
        np.testing.assert_allclose(turned[0, 1], np.cos(angle), atol=1e-5)
        np.testing.assert_allclose(abs(turned[0, 2]), abs(np.sin(angle)), atol=1e-5)


def test_the_angle_is_only_for_the_rotor_constructor():
    _lane()
    with pytest.raises(jb.TesseraJitError, match="angle applies to rotor_from_axis"):
        jb.jit_clifford_op("exp", _bivector(1, 1.0, seed=1), angle=1.0)


# --- ragged batches --------------------------------------------------------

def test_a_ragged_batch_compiles_once_and_runs_at_any_length():
    """The batched lowering used to need static leading extents, so a batch of a
    different length was a different program. With the leading axes dynamic the
    loop bound comes from `tensor.dim` and one cached module serves every length
    — which is the whole point of admitting them."""
    _lane()
    for rows in (3, 7, 3, 64):
        a = np.random.default_rng(rows).standard_normal((rows, 8)).astype(np.float32)
        b = np.random.default_rng(rows + 1).standard_normal((rows, 8)).astype(np.float32)
        got = jb.jit_clifford_op("geo_product", a, b, ragged=True)
        want = ga.geometric_product(Multivector(a.copy(), CL30),
                                    Multivector(b.copy(), CL30)).to_numpy()
        assert np.max(np.abs(got - want)) < 2e-6, rows
        # The static spelling must agree with the ragged one for the same input.
        np.testing.assert_allclose(got, jb.jit_clifford_op("geo_product", a, b), atol=1e-7)


@pytest.mark.parametrize("op", ["reverse", "grade", "exp", "norm"])
def test_the_unary_family_is_ragged_too(op):
    _lane()
    b = _bivector(5, 0.8, seed=2).reshape(5, 8)
    kw = {"grades": [2]} if op == "grade" else {}
    got = jb.jit_clifford_op(op, b, ragged=True, **kw)
    want = jb.jit_clifford_op(op, b, **kw)
    np.testing.assert_array_equal(got, want)


def test_ragged_needs_a_batch_axis_to_be_ragged_in():
    _lane()
    with pytest.raises(jb.TesseraJitError, match="ragged needs a batched operand"):
        jb.jit_clifford_op("norm", np.zeros((8,), dtype=np.float32), ragged=True)
