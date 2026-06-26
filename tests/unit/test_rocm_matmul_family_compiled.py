"""Compiler-generated matmul-family ops on gfx1151 — batched_gemm /
linear_general / qkv_projection / factorized_matmul / einsum, all built on the
same WMMA GEMM kernel (the rocm_compiled spine) via reshaping/batching/splitting
in the runtime (the matmul analog of how flash_attn GQA/MQA reuse the FA kernel).

Reachable through `runtime.launch()` via
`compiler_path="rocm_matmul_family_compiled"`. f16/bf16 storage, f32 accumulate
(WMMA on gfx1151 has no f32 storage path). Validated vs numpy.

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _mm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, operands, kwargs=None):
    names = [f"x{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_matmul_family_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": names, "kwargs": kwargs or {}}],
    }), tuple(operands)


def _dtypes():
    out = [(np.float16, 3e-2)]
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    out.append((bf16, 2e-1))
    return out


@pytest.mark.parametrize("dtype,tol", _dtypes())
@pytest.mark.parametrize("batch", [(3,), (2, 4)])
def test_batched_gemm(dtype, tol, batch):
    rt = _mm_or_skip()
    rng = np.random.default_rng(1 + len(batch))
    m, k, n = 32, 48, 16
    a = (rng.standard_normal((*batch, m, k)) * 0.3).astype(dtype)
    b = (rng.standard_normal((*batch, k, n)) * 0.3).astype(dtype)
    art, ops = _artifact(rt, "tessera.batched_gemm", [a, b])
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    out = res["output"].astype(np.float32)
    ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
    np.testing.assert_allclose(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype,tol", _dtypes())
def test_linear_general(dtype, tol):
    rt = _mm_or_skip()
    rng = np.random.default_rng(7)
    bsz, seq, k, n = 2, 5, 64, 32
    x = (rng.standard_normal((bsz, seq, k)) * 0.3).astype(dtype)
    w = (rng.standard_normal((k, n)) * 0.3).astype(dtype)
    bias = (rng.standard_normal((n,)) * 0.1).astype(np.float32)
    art, ops = _artifact(rt, "tessera.linear_general", [x, w, bias])
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    out = res["output"].astype(np.float32)
    ref = np.matmul(x.astype(np.float32), w.astype(np.float32)) + bias
    np.testing.assert_allclose(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype,tol", _dtypes())
def test_qkv_projection(dtype, tol):
    rt = _mm_or_skip()
    rng = np.random.default_rng(9)
    bsz, seq, d, n = 2, 6, 48, 16
    x = (rng.standard_normal((bsz, seq, d)) * 0.3).astype(dtype)
    w = (rng.standard_normal((d, 3 * n)) * 0.3).astype(dtype)
    art, ops = _artifact(rt, "tessera.qkv_projection", [x, w])
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    packed = res["output"].astype(np.float32)
    ref = np.matmul(x.astype(np.float32), w.astype(np.float32))
    np.testing.assert_allclose(packed, ref, atol=tol, rtol=tol)
    # the op semantics: a 3-way last-axis split into (Q, K, V).
    q, kk, v = np.split(packed, 3, axis=-1)
    assert q.shape == kk.shape == v.shape == (bsz, seq, n)


@pytest.mark.parametrize("dtype,tol", _dtypes())
def test_factorized_matmul(dtype, tol):
    rt = _mm_or_skip()
    rng = np.random.default_rng(13)
    m, k, n, rank = 24, 32, 20, 4
    a = (rng.standard_normal((m, k)) * 0.3).astype(dtype)
    b = (rng.standard_normal((k, n)) * 0.3).astype(dtype)
    art, ops = _artifact(rt, "tessera.factorized_matmul", [a, b],
                         {"rank": rank})
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    out = res["output"].astype(np.float32)
    # reference: gemm in the storage dtype (matches the GPU product) then SVD.
    prod = np.matmul(a.astype(np.float32), b.astype(np.float32))
    u, s, vh = np.linalg.svd(prod, full_matrices=False)
    r = max(1, min(rank, s.shape[-1]))
    ref = (u[..., :r] * s[..., :r]) @ vh[..., :r, :]
    # SVD on a half-precision-rounded product amplifies error; compare loosely.
    np.testing.assert_allclose(out, ref, atol=max(tol, 5e-2), rtol=max(tol, 5e-2))


@pytest.mark.parametrize("dtype,tol", _dtypes())
@pytest.mark.parametrize("spec,shapes", [
    ("ik,kj->ij", ((16, 32), (32, 24))),               # plain matmul
    ("bik,bkj->bij", ((3, 16, 32), (3, 32, 24))),      # batched
    ("bhik,bhkj->bhij", ((2, 2, 16, 32), (2, 2, 32, 24))),  # multi-batch
    ("bsk,kn->bsn", ((2, 5, 48), (48, 16))),           # shared weight
])
def test_einsum_matmul_specs(dtype, tol, spec, shapes):
    rt = _mm_or_skip()
    rng = np.random.default_rng(21 + len(spec))
    a = (rng.standard_normal(shapes[0]) * 0.3).astype(dtype)
    b = (rng.standard_normal(shapes[1]) * 0.3).astype(dtype)
    art, ops = _artifact(rt, "tessera.einsum", [a, b], {"spec": spec})
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    out = res["output"].astype(np.float32)
    ref = np.einsum(spec, a.astype(np.float32), b.astype(np.float32))
    np.testing.assert_allclose(out, ref, atol=tol, rtol=tol)


def test_einsum_unsupported_spec_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 4), np.float16)
    # two contraction indices -> not a single (batched) matmul
    art, ops = _artifact(rt, "tessera.einsum", [a, a], {"spec": "ij,ij->"})
    with pytest.raises(ValueError, match="unsupported|single contraction"):
        rt._execute_rocm_compiled_matmul_family(art, ops)
