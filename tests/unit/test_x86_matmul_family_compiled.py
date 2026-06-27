"""x86 GEMM-family lane — batched_gemm / linear_general / qkv_projection /
factorized_matmul / einsum, all on the AVX-512 f32 GEMM microkernel
(tessera_x86_avx512_gemm_f32) with reshape/batch/einsum in Python. The CPU
analog of the ROCm WMMA matmul-family lane.

Reachable through `runtime.launch()` via
`compiler_path="x86_matmul_family_compiled"`. f32; validated vs numpy at a
K-scaled tolerance (rtol 1e-3).

Skip-clean: libtessera_x86_elementwise.so absent.
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, operands, kwargs=None):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_matmul_family_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": list(operands), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(operands),
                 "kwargs": kwargs or {}}],
    })


_TOL = dict(atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("shape", [(2, 4, 8, 16), (3, 5, 7), (8, 8)])
def test_batched_gemm_matches_numpy(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(11 + len(shape))
    *batch, m, k = shape
    a = rng.standard_normal((*batch, m, k)).astype(np.float32)
    b = rng.standard_normal((*batch, k, m)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.batched_gemm", ("a", "b")), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_matmul_family_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               (a @ b).astype(np.float32), **_TOL)


def test_batched_gemm_shared_rank2_rhs():
    rt = _x86_or_skip()
    rng = np.random.default_rng(3)
    a = rng.standard_normal((4, 6, 16)).astype(np.float32)
    b = rng.standard_normal((16, 10)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.batched_gemm", ("a", "b")), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               (a @ b).astype(np.float32), **_TOL)


@pytest.mark.parametrize("ash,bsh", [
    ((2, 1, 5, 8), (1, 3, 8, 6)),    # mutual broadcast -> (2, 3, 5, 6)
    ((1, 4, 8), (3, 8, 6)),          # broadcast leading 1
    ((3, 5, 8), (8, 6)),             # rank-2 rhs shared across batch
    ((5, 8), (2, 8, 6)),             # rank-2 lhs shared across batch
])
def test_batched_gemm_broadcast(ash, bsh):
    rt = _x86_or_skip()
    rng = np.random.default_rng(hash((ash, bsh)) % 2**31)
    a = rng.standard_normal(ash).astype(np.float32)
    b = rng.standard_normal(bsh).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.batched_gemm", ("a", "b")), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               (a @ b).astype(np.float32), **_TOL)


def test_gemm_rejects_non_f32():
    rt = _x86_or_skip()
    a = np.zeros((4, 8), np.float64)
    b = np.zeros((8, 4), np.float64)
    res = rt.launch(_artifact(rt, "tessera.batched_gemm", ("a", "b")), (a, b))
    assert res["ok"] is False
    assert "f32 only" in str(res.get("reason"))


def test_linear_general_with_bias():
    rt = _x86_or_skip()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 5, 32)).astype(np.float32)
    w = rng.standard_normal((32, 12)).astype(np.float32)
    bias = rng.standard_normal((12,)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.linear_general", ("x", "w", "b")),
                    (x, w, bias))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               (x @ w + bias).astype(np.float32), **_TOL)


def test_qkv_projection():
    rt = _x86_or_skip()
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2, 7, 16)).astype(np.float32)
    w = rng.standard_normal((16, 48)).astype(np.float32)   # 3*16
    res = rt.launch(_artifact(rt, "tessera.qkv_projection", ("x", "w")), (x, w))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               (x @ w).astype(np.float32), **_TOL)


@pytest.mark.parametrize("spec,ls,rs", [
    ("bij,bjk->bik", (3, 4, 5), (3, 5, 6)),      # batched matmul
    ("ij,jk->ik", (8, 16), (16, 10)),            # plain matmul
    ("bhid,bhjd->bhij", (2, 3, 5, 7), (2, 3, 6, 7)),  # attention scores
])
def test_einsum_single_contraction(spec, ls, rs):
    rt = _x86_or_skip()
    rng = np.random.default_rng(hash(spec) % 2**31)
    a = rng.standard_normal(ls).astype(np.float32)
    b = rng.standard_normal(rs).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.einsum", ("a", "b"),
                              {"spec": spec}), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               np.einsum(spec, a, b).astype(np.float32), **_TOL)


def test_factorized_matmul_rank_truncation():
    rt = _x86_or_skip()
    rng = np.random.default_rng(13)
    a = rng.standard_normal((12, 20)).astype(np.float32)
    b = rng.standard_normal((20, 16)).astype(np.float32)
    rank = 4
    res = rt.launch(_artifact(rt, "tessera.factorized_matmul", ("a", "b"),
                              {"rank": rank}), (a, b))
    assert res["ok"] is True, res.get("reason")
    full = (a @ b).astype(np.float32)
    u, s, vh = np.linalg.svd(full, full_matrices=False)
    ref = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               ref.astype(np.float32), atol=1e-2, rtol=1e-2)


def test_einsum_multi_contraction_rejected():
    rt = _x86_or_skip()
    a = np.zeros((4, 4, 4), np.float32)
    with pytest.raises(ValueError, match="single contraction"):
        rt._execute_x86_compiled_matmul_family(
            _artifact(rt, "tessera.einsum", ("a", "b"),
                      {"spec": "ijk,ijk->i"}), (a, a))


def test_matmul_family_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_matmul_family_compiled executor"):
        rt._execute_x86_compiled_matmul_family(
            _artifact(rt, "tessera.softmax", ("a",)), (a,))
