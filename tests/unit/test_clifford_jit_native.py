"""Batched geometric products through the MLIR/LLVM CPU lane (W6.4, 2026-09-16).

The Clifford dialect's GradeFusion + ExpandProductTable now run inside
libtessera_jit, so a `tessera_clifford.geo_product` on `[..., 8]` tensors is
compiled -- not interpreted, not routed to a Python-emitted kernel -- and
executed on the host CPU. The oracle is the standalone GA reference
(`tessera._clifford_ops`, Cl(3,0)); the unfakeable invocation counter proves
the compiled function ran. Skips only when libtessera_jit lacks the lane.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera import _jit_boundary as jb

pytestmark = pytest.mark.skipif(
    not jb.has_clifford(),
    reason="libtessera_jit built without the Clifford lane (TESSERA_BUILD_CLIFFORD_BACKEND=ON)",
)


def _reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from tessera._clifford_ops import clifford_geometric_product
    flat_a, flat_b = a.reshape(-1, 8), b.reshape(-1, 8)
    return np.stack([clifford_geometric_product(x, y) for x, y in zip(flat_a, flat_b)]).reshape(a.shape).astype(np.float32)


@pytest.mark.parametrize("shape", [(8,), (32, 8), (4, 5, 8), (1, 8)])
def test_batched_geo_product_matches_reference(shape):
    rng = np.random.default_rng(hash(shape) & 0xFFFF)
    a = rng.standard_normal(shape).astype(np.float32)
    b = rng.standard_normal(shape).astype(np.float32)
    before = jb.invocation_count()
    out = jb.jit_clifford_geo_product(a, b)
    assert jb.invocation_count() == before + 1  # the MLIR/LLVM function executed
    assert out.shape == shape and out.dtype == np.float32
    np.testing.assert_allclose(out, _reference(a, b), rtol=1e-5, atol=1e-5)


def test_grade_restricted_product_prunes_to_the_projection():
    """`grades=[2]` must equal the grade-2 projection of the full product and
    write zeros elsewhere -- the pruned table is observable, not just smaller."""
    from tessera._clifford_ops import clifford_grade_projection
    rng = np.random.default_rng(7)
    a = rng.standard_normal((6, 8)).astype(np.float32)
    b = rng.standard_normal((6, 8)).astype(np.float32)
    full = _reference(a, b)
    expect = np.stack([clifford_grade_projection(row, 2) for row in full]).astype(np.float32)
    out = jb.jit_clifford_geo_product(a, b, grades=[2])
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)
    grade2 = [3, 5, 6]  # blade masks with popcount 2 in Cl(3,0)
    assert np.all(out[:, [i for i in range(8) if i not in grade2]] == 0)
    assert np.any(out[:, grade2] != 0)


def test_noncommutativity_is_preserved():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((3, 8)).astype(np.float32)
    b = rng.standard_normal((3, 8)).astype(np.float32)
    ab, ba = jb.jit_clifford_geo_product(a, b), jb.jit_clifford_geo_product(b, a)
    assert not np.allclose(ab, ba)
    np.testing.assert_allclose(ab, _reference(a, b), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bad", [((4, 7), (4, 7)), ((4, 8), (5, 8)), ((), ())])
def test_out_of_envelope_raises_without_fallback(bad):
    a = np.zeros(bad[0], np.float32)
    b = np.zeros(bad[1], np.float32)
    with pytest.raises(jb.TesseraJitError):
        jb.jit_clifford_geo_product(a, b)


def test_runtime_launch_reports_the_native_cpu_lane():
    """The execution-matrix row has a consumer: `runtime.launch` on the packaged
    artifact executes the lane and labels the result native_cpu / mlir_llvm_jit."""
    from tessera import runtime as rt
    from tessera.compiler.clifford_jit import package_clifford_geo_product_cpu
    rng = np.random.default_rng(3)
    a = rng.standard_normal((5, 8)).astype(np.float32)
    b = rng.standard_normal((5, 8)).astype(np.float32)
    artifact = package_clifford_geo_product_cpu((5, 8))
    before = jb.invocation_count()
    result = rt.launch(artifact, (a, b))
    assert result["ok"] and result["execution_kind"] == "native_cpu"
    assert result["compiler_path"] == "cpu_clifford_llvm_jit"
    assert jb.invocation_count() == before + 1
    np.testing.assert_allclose(np.asarray(result["output"]), _reference(a, b), rtol=1e-5, atol=1e-5)
    # A shape outside the packaged envelope is a failed launch, never a fallback.
    refused = rt.launch(artifact, (a[:2], b[:2]))
    assert refused["ok"] is False and "shape" in str(refused.get("reason", refused))
    assert "output" not in refused or refused["output"] is None


_REF = {
    "wedge": "clifford_wedge", "left_contract": "clifford_left_contraction",
    "inner": "clifford_inner", "norm": "clifford_norm", "rotor_sandwich": "clifford_rotor_sandwich",
    "reverse": "clifford_reverse", "grade_involute": "clifford_grade_involution",
    "conjugate": "clifford_conjugate", "hodge_star": "clifford_hodge_star",
}


def _rows(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _reference_op(op, *arrays):
    """Apply the standalone GA reference row by row (it takes one multivector
    at a time for the unary/scalar forms) and re-batch."""
    import tessera._clifford_ops as ref
    fn = getattr(ref, _REF[op])
    lead = arrays[0].shape[:-1]
    flat = [x.reshape(-1, 8) for x in arrays]
    rows = [np.asarray(fn(*[f[i] for f in flat]), dtype=np.float32) for i in range(flat[0].shape[0])]
    return np.stack(rows).reshape(lead + rows[0].shape).astype(np.float32)


@pytest.mark.parametrize("op", sorted(_REF))
@pytest.mark.parametrize("shape", [(8,), (6, 8), (2, 3, 8)])
def test_family_matches_reference(op, shape):
    """Every op of the family the lane admits matches the standalone GA
    reference for single, batched and rank-3 inputs (Cl(3,0), f32)."""
    arity = jb.CLIFFORD_JIT_OPS[op][0]
    arrays = [_rows(shape, 100 * arity + i + len(shape)) for i in range(arity)]
    if op == "rotor_sandwich":  # a unit rotor: cos + sin e12
        rotor = np.zeros(shape, np.float32); rotor[..., 0] = np.cos(0.3); rotor[..., 3] = np.sin(0.3)
        arrays[0] = rotor
    before = jb.invocation_count()
    out = jb.jit_clifford_op(op, *arrays)
    assert jb.invocation_count() == before + 1
    expect = _reference_op(op, *arrays)
    assert out.shape == expect.shape, (out.shape, expect.shape)
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)


def test_grade_projection_keeps_only_the_listed_grades():
    from tessera._clifford_ops import clifford_grade_projection
    a = _rows((5, 8), 21)
    out = jb.jit_clifford_op("grade", a, grades=[1, 3])
    expect = np.stack([clifford_grade_projection(row, [1, 3]) if False else
                       np.where(np.isin(np.arange(8), [1, 2, 4, 7]), row, 0.0) for row in a]).astype(np.float32)
    np.testing.assert_allclose(out, expect, rtol=0, atol=0)
    with pytest.raises(jb.TesseraJitError, match="grades"):
        jb.jit_clifford_op("grade", a)


def test_rotor_sandwich_preserves_vector_norm():
    """A unit rotor rotates: |R v R~| == |v| through the native lane."""
    v = np.zeros((4, 8), np.float32); v[:, [1, 2, 4]] = _rows((4, 3), 5)
    rotor = np.zeros((4, 8), np.float32); rotor[:, 0] = np.cos(0.7); rotor[:, 5] = np.sin(0.7)
    rotated = jb.jit_clifford_op("rotor_sandwich", rotor, v)
    np.testing.assert_allclose(jb.jit_clifford_op("norm", rotated), jb.jit_clifford_op("norm", v), rtol=1e-5, atol=1e-5)
    assert not np.allclose(rotated, v)


def _pure_bivector(shape, seed):
    """Only grade 2 populated: what `exp`'s closed form admits."""
    out = np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    for mask in (3, 5, 6):
        out[..., mask] = rng.standard_normal(shape[:-1]).astype(np.float32)
    return out


def test_family_launch_consumer():
    from tessera import runtime as rt
    from tessera.compiler.clifford_jit import package_clifford_cpu
    a, b = _rows((3, 8), 1), _rows((3, 8), 2)
    result = rt.launch(package_clifford_cpu("wedge", (3, 8)), (a, b))
    assert result["ok"] and result["execution_kind"] == "native_cpu"
    np.testing.assert_allclose(np.asarray(result["output"]), _reference_op("wedge", a, b), rtol=1e-5, atol=1e-5)
    # `exp` used to be the negative case here: declared with no lowering. It has
    # one since 2026-09-16, so the negative case moves to a field op, which still
    # has none.
    exp_result = rt.launch(package_clifford_cpu("exp", (3, 8)), (_pure_bivector((3, 8), 4),))
    assert exp_result["ok"] and exp_result["execution_kind"] == "native_cpu"
    with pytest.raises(Exception, match="no lowering"):
        package_clifford_cpu("ext_deriv", (3, 8))

