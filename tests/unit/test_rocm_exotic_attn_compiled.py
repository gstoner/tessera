"""Compiler-generated exotic-attention compositions on gfx1151 — gated_attention,
mla_decode, mla_decode_fused, built by composing the already-compiled WMMA
flash_attn kernel + the WMMA GEMM kernel (MLA latent projections) + an
elementwise gate (the attention analog of the matmul-family lane).

Reachable through `runtime.launch()` via
`compiler_path="rocm_exotic_attn_compiled"`. f16/bf16 storage, f32
softmax+accumulate. Validated vs the numpy references (flash-attn softmax math).

The recurrent DeltaNet variants (gated_deltanet / kimi / modified) and
block-sparse deepseek_sparse_attention need a sequential-scan / sparse-gather
kernel and are NOT covered here (they stay artifact_only).

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _attn_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, operands, kwargs=None):
    names = [f"x{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_exotic_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": names, "kwargs": kwargs or {}}],
    }), tuple(operands)


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _ref_attn(q, k, v, *, scale=None, causal=True):
    q = q.astype(np.float32); k = k.astype(np.float32); v = v.astype(np.float32)
    d = q.shape[-1]
    s = np.matmul(q, np.swapaxes(k, -1, -2)) * (scale or 1.0 / np.sqrt(d))
    if causal:
        sq, sk = s.shape[-2], s.shape[-1]
        mask = np.triu(np.ones((sq, sk), bool), k=1 + (sk - sq))
        s = np.where(mask, -np.inf, s)
    return np.matmul(_softmax(s), v)


def _dtypes():
    out = [(np.float16, 3e-2)]
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    out.append((bf16, 1e-1))
    return out


@pytest.mark.parametrize("dtype,tol", _dtypes())
def test_gated_attention(dtype, tol):
    rt = _attn_or_skip()
    rng = np.random.default_rng(2)
    b, h, s, d = 1, 2, 16, 32
    q = (rng.standard_normal((b, h, s, d)) * 0.3).astype(dtype)
    k = (rng.standard_normal((b, h, s, d)) * 0.3).astype(dtype)
    v = (rng.standard_normal((b, h, s, d)) * 0.3).astype(dtype)
    gate = (rng.standard_normal((b, h, s, d)) * 0.5).astype(np.float32)
    art, ops = _artifact(rt, "tessera.gated_attention", [q, k, v, gate],
                         {"causal": True})
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    out = res["output"].astype(np.float32)
    attn = _ref_attn(q, k, v, causal=True)
    ref = attn * (1.0 / (1.0 + np.exp(-gate)))
    np.testing.assert_allclose(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype,tol", _dtypes())
def test_mla_decode_with_projections(dtype, tol):
    rt = _attn_or_skip()
    rng = np.random.default_rng(4)
    b, h, s, dl, d = 1, 2, 16, 48, 32
    q = (rng.standard_normal((b, h, s, d)) * 0.3).astype(dtype)
    k_lat = (rng.standard_normal((b, h, s, dl)) * 0.3).astype(dtype)
    v_lat = (rng.standard_normal((b, h, s, dl)) * 0.3).astype(dtype)
    w_k = (rng.standard_normal((dl, d)) * 0.2).astype(dtype)
    w_v = (rng.standard_normal((dl, d)) * 0.2).astype(dtype)
    art, ops = _artifact(rt, "tessera.mla_decode",
                         [q, k_lat, v_lat, w_k, w_v], {"causal": False})
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    out = res["output"].astype(np.float32)
    k = np.matmul(k_lat.astype(np.float32), w_k.astype(np.float32))
    v = np.matmul(v_lat.astype(np.float32), w_v.astype(np.float32))
    ref = _ref_attn(q, k, v, causal=False)
    np.testing.assert_allclose(out, ref, atol=max(tol, 5e-2), rtol=max(tol, 5e-2))


@pytest.mark.parametrize("dtype,tol", _dtypes())
def test_mla_decode_fused(dtype, tol):
    rt = _attn_or_skip()
    rng = np.random.default_rng(6)
    b, s, dx, dc, d = 1, 16, 64, 48, 32
    x = (rng.standard_normal((b, s, dx)) * 0.2).astype(dtype)
    w_dkv = (rng.standard_normal((dx, dc)) * 0.2).astype(dtype)
    w_uk = (rng.standard_normal((dc, d)) * 0.2).astype(dtype)
    w_uv = (rng.standard_normal((dc, d)) * 0.2).astype(dtype)
    q = (rng.standard_normal((b, s, d)) * 0.3).astype(dtype)
    art, ops = _artifact(rt, "tessera.mla_decode_fused",
                         [x, w_dkv, w_uk, w_uv, q], {"causal": False})
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    out = res["output"].astype(np.float32)
    c = np.matmul(x.astype(np.float32), w_dkv.astype(np.float32))
    k = np.matmul(c, w_uk.astype(np.float32))
    v = np.matmul(c, w_uv.astype(np.float32))
    ref = _ref_attn(q, k, v, causal=False)
    np.testing.assert_allclose(out, ref, atol=max(tol, 6e-2), rtol=max(tol, 6e-2))


def test_hybrid_attention_mla_slot_reference_fallback_matches_ops():
    import tessera as ts
    from tessera import runtime as rt

    rng = np.random.default_rng(9)
    q = (rng.standard_normal((1, 2, 8, 32)) * 0.2).astype(np.float32)
    k = (rng.standard_normal((1, 2, 8, 32)) * 0.2).astype(np.float32)
    v = (rng.standard_normal((1, 2, 8, 32)) * 0.2).astype(np.float32)
    kw = {"pattern": "ling_1_7_mla_lightning", "layer_index": 7, "causal": True}
    art, ops = _artifact(rt, "tessera.hybrid_attention", [q, k, v], kw)
    res = rt.launch(art, ops)

    assert res["ok"], res.get("reason")
    assert res["execution_kind"] == "reference_cpu"
    ref = ts.ops.hybrid_attention(q, k, v, **kw)
    np.testing.assert_allclose(res["output"], ref, atol=1e-6, rtol=1e-6)


def test_hybrid_attention_fp32_lightning_reference_fallback_matches_ops():
    import tessera as ts
    from tessera import runtime as rt

    rng = np.random.default_rng(11)
    q = (rng.standard_normal((1, 2, 8, 32)) * 0.2).astype(np.float32)
    k = (rng.standard_normal((1, 2, 8, 32)) * 0.2).astype(np.float32)
    v = (rng.standard_normal((1, 2, 8, 32)) * 0.2).astype(np.float32)
    kw = {"pattern": "auto", "causal": True}
    art, ops = _artifact(rt, "tessera.hybrid_attention", [q, k, v], kw)
    res = rt.launch(art, ops)

    assert res["ok"], res.get("reason")
    assert res["execution_kind"] == "reference_cpu"
    ref = ts.ops.hybrid_attention(q, k, v, **kw)
    np.testing.assert_allclose(res["output"], ref, atol=1e-6, rtol=1e-6)


def test_hybrid_attention_auto_native_gpu_matches_ops_on_hardware():
    import tessera as ts

    rt = _attn_or_skip()
    rng = np.random.default_rng(10)
    q = (rng.standard_normal((1, 2, 8, 32)) * 0.2).astype(np.float16)
    k = (rng.standard_normal((1, 2, 8, 32)) * 0.2).astype(np.float16)
    v = (rng.standard_normal((1, 2, 8, 32)) * 0.2).astype(np.float16)
    kw = {"pattern": "auto", "causal": True}
    art, ops = _artifact(rt, "tessera.hybrid_attention", [q, k, v], kw)
    res = rt.launch(art, ops)

    assert res["ok"], res.get("reason")
    assert res["execution_kind"] == "native_gpu"
    ref = ts.ops.hybrid_attention(q, k, v, **kw)
    np.testing.assert_allclose(res["output"].astype(np.float32), ref.astype(np.float32),
                               atol=3e-2, rtol=3e-2)


def test_gated_attention_bad_arity_rejected():
    from tessera import runtime as rt
    q = np.zeros((1, 2, 16, 32), np.float16)
    art, ops = _artifact(rt, "tessera.gated_attention", [q, q, q])  # missing gate
    with pytest.raises(ValueError, match="gated_attention takes"):
        rt._execute_rocm_compiled_exotic_attention(art, ops)
