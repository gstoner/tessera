"""Live sm_120 proof for the CUDA base linear-attention contract."""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import require_nvidia_mma_runtime


def _runtime():
    return require_nvidia_mma_runtime()


def _artifact():
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_linear_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": "tessera.linear_attn", "result": "o",
                 "operands": ["q", "k", "v"], "kwargs": {"causal": True}}],
    })


def _variant_artifact(op, **kwargs):
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_linear_attn_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["q", "k", "v"], "output_name": "o", "ops": [{"op_name": op, "result": "o", "operands": ["q", "k", "v"], "kwargs": kwargs}]})


def _bwd_artifact():
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_linear_attn_bwd_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["go", "q", "k", "v"], "output_name": "grads", "ops": [{"op_name": "tessera.linear_attn", "result": "grads", "operands": ["go", "q", "k", "v"], "kwargs": {"causal": True}}]})


def _variant_bwd_artifact(op, **kwargs):
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_linear_attn_bwd_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["go", "q", "k", "v"], "output_name": "grads", "ops": [{"op_name": op, "result": "grads", "operands": ["go", "q", "k", "v"], "kwargs": kwargs}]})


def _reference(q, k, v):
    out = np.empty_like(q)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for m in range(q.shape[2]):
                for d in range(q.shape[3]):
                    out[b, h, m, d] = sum(
                        float(q[b, h, m] @ k[b, h, n]) * float(v[b, h, n, d])
                        for n in range(m + 1))
    return out


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_linear_attn_causal_identity_f32():
    rt = _runtime(); rng = np.random.default_rng(941)
    q = rng.standard_normal((2, 3, 5, 4), dtype=np.float32)
    k = rng.standard_normal((2, 3, 5, 4), dtype=np.float32)
    v = rng.standard_normal((2, 3, 5, 4), dtype=np.float32)
    got = rt.launch(_artifact(), (q, k, v))
    assert got["ok"], got.get("reason")
    np.testing.assert_allclose(got["output"], _reference(q, k, v), atol=3e-5, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_linear_attn_vjp_f32():
    rt = _runtime(); rng = np.random.default_rng(947)
    q = rng.standard_normal((1, 2, 4, 3), dtype=np.float32); k = rng.standard_normal(q.shape, dtype=np.float32); v = rng.standard_normal(q.shape, dtype=np.float32); go = rng.standard_normal(q.shape, dtype=np.float32)
    dq = np.zeros_like(q); dk = np.zeros_like(k); dv = np.zeros_like(v)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for m in range(q.shape[2]):
                for n in range(m + 1):
                    score = q[b,h,m] @ k[b,h,n]; ds = go[b,h,m] @ v[b,h,n]
                    dq[b,h,m] += ds * k[b,h,n]; dk[b,h,n] += ds * q[b,h,m]; dv[b,h,n] += score * go[b,h,m]
    got = rt.launch(_bwd_artifact(), (go, q, k, v))
    assert got["ok"], got.get("reason")
    for actual, expected in zip(got["output"], (dq, dk, dv)):
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("op,fmap", [("tessera.lightning_attention", "identity"), ("tessera.retention", "polynomial_2")])
def test_live_nvidia_linear_attn_decay_variants(op, fmap):
    rt = _runtime(); rng = np.random.default_rng(953)
    q = rng.standard_normal((1, 2, 4, 3), dtype=np.float32); k = rng.standard_normal(q.shape, dtype=np.float32); v = rng.standard_normal((1, 2, 4, 2), dtype=np.float32); decay = np.array([[[1,.9,.8,.7], [1,.8,.9,.6]]], np.float32)
    phi = (lambda x: x) if fmap == "identity" else (lambda x: x*x)
    ref = np.zeros_like(v)
    for b in range(1):
        for h in range(2):
            for m in range(4):
                for n in range(m + 1): ref[b,h,m] += (phi(q[b,h,m]) @ phi(k[b,h,n])) * np.prod(decay[b,h,n+1:m+1]) * v[b,h,n]
    got = rt.launch(_variant_artifact(op, decay=decay.tolist(), deg=2), (q, k, v))
    assert got["ok"], got.get("reason")
    np.testing.assert_allclose(got["output"], ref, atol=4e-5, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("op,square", [("tessera.lightning_attention", False), ("tessera.retention", True)])
def test_live_nvidia_linear_attn_variant_vjp(op, square):
    rt = _runtime(); rng = np.random.default_rng(961); q = rng.standard_normal((1,1,3,2), dtype=np.float32); k = rng.standard_normal(q.shape, dtype=np.float32); v = rng.standard_normal((1,1,3,3), dtype=np.float32); go = rng.standard_normal(v.shape, dtype=np.float32); de = np.array([[[1,.8,.6]]], np.float32); dq=np.zeros_like(q);dk=np.zeros_like(k);dv=np.zeros_like(v)
    for m in range(3):
        for n in range(m+1):
            fac=np.prod(de[0,0,n+1:m+1]); pq=q[0,0,m]**2 if square else q[0,0,m]; pk=k[0,0,n]**2 if square else k[0,0,n]; ds=fac*(go[0,0,m]@v[0,0,n]); dq[0,0,m]+=ds*pk*(2*q[0,0,m] if square else 1); dk[0,0,n]+=ds*pq*(2*k[0,0,n] if square else 1); dv[0,0,n]+=fac*(pq@pk)*go[0,0,m]
    got=rt.launch(_variant_bwd_artifact(op, decay=de.tolist(), deg=2), (go,q,k,v)); assert got["ok"],got.get("reason")
    for a,e in zip(got["output"],(dq,dk,dv)): np.testing.assert_allclose(a,e,atol=8e-5,rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_retention_vjp_honors_log_g():
    rt = _runtime(); rng = np.random.default_rng(967)
    q = rng.standard_normal((1, 1, 3, 2), dtype=np.float32)
    k = rng.standard_normal(q.shape, dtype=np.float32)
    v = rng.standard_normal((1, 1, 3, 3), dtype=np.float32)
    go = rng.standard_normal(v.shape, dtype=np.float32)
    decay = np.array([[[1.0, .8, .6]]], np.float32)
    expected = rt.launch(
        _variant_bwd_artifact("tessera.retention", decay=decay.tolist(), deg=2),
        (go, q, k, v))
    actual = rt.launch(
        _variant_bwd_artifact("tessera.retention", log_g=np.log(decay).tolist(), deg=2),
        (go, q, k, v))
    assert expected["ok"], expected.get("reason")
    assert actual["ok"], actual.get("reason")
    for got, want in zip(actual["output"], expected["output"]):
        np.testing.assert_allclose(got, want, atol=8e-5, rtol=0)
