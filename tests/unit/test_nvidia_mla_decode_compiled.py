import os, shutil
import numpy as np
import pytest
@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_mla_decode_composed():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")): pytest.skip("nvcc unavailable")
    from tessera import runtime as rt
    if not rt._nvidia_mma_runtime_available(): pytest.skip("CUDA unavailable")
    rng=np.random.default_rng(977); q=rng.standard_normal((1,2,4,4),dtype=np.float32); kl=rng.standard_normal((1,1,5,3),dtype=np.float32); vl=rng.standard_normal((1,1,5,2),dtype=np.float32); wk=rng.standard_normal((3,4),dtype=np.float32); wv=rng.standard_normal((2,3),dtype=np.float32)
    art=rt.RuntimeArtifact(metadata={"target":"nvidia_sm120","compiler_path":"nvidia_mla_decode_compiled","executable":True,"execution_kind":"native_gpu","arg_names":["q","k","v","wk","wv"],"output_name":"o","ops":[{"op_name":"tessera.mla_decode","result":"o","operands":["q","k","v","wk","wv"],"kwargs":{"causal":True,"scale":.5}}]})
    got=rt.launch(art,(q,kl,vl,wk,wv)); assert got["ok"],got.get("reason")
    k=kl@wk;v=vl@wv; s=np.einsum("bhid,bhjd->bhij",q,k)*.5; s=np.where(np.arange(5)[None,None,None,:]<=np.arange(4)[None,None,:,None],s,-np.inf); p=np.exp(s-s.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True); np.testing.assert_allclose(got["output"],p@v,atol=5e-5,rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_mla_decode_fused_entry_smoke():
    rt = _cuda_runtime()
    from tessera.compiler.emit.nvidia_cuda import run_mla_decode_fused
    rng = np.random.default_rng(981)
    x = rng.standard_normal((1, 1, 5, 3), dtype=np.float32)
    wd = rng.standard_normal((3, 2), dtype=np.float32)
    wk = rng.standard_normal((2, 4), dtype=np.float32)
    wv = rng.standard_normal((2, 3), dtype=np.float32)
    q = rng.standard_normal((1, 2, 4, 4), dtype=np.float32)
    out = run_mla_decode_fused(x, wd, wk, wv, q, scale=.5, causal=True)
    assert rt._nvidia_mma_runtime_available()
    assert out.shape == (1, 2, 4, 3)
    assert np.isfinite(out).all()


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_mla_decode_fused_three_way():
    rt = _cuda_runtime()
    from tessera.compiler.emit.nvidia_cuda import (
        run_flash_attention_forward, run_mla_decode_fused)
    rng = np.random.default_rng(983)
    x = rng.standard_normal((1, 1, 6, 3), dtype=np.float32)
    wd = rng.standard_normal((3, 2), dtype=np.float32)
    wk = rng.standard_normal((2, 4), dtype=np.float32)
    wv = rng.standard_normal((2, 3), dtype=np.float32)
    q = rng.standard_normal((1, 2, 5, 4), dtype=np.float32)
    c = x @ wd; k = c @ wk; v = c @ wv
    artifact = rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_mla_decode_fused_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x", "wd", "wk", "wv", "q"], "output_name": "o",
        "ops": [{"op_name": "tessera.mla_decode_fused", "result": "o",
                 "operands": ["x", "wd", "wk", "wv", "q"],
                 "kwargs": {"scale": .5, "causal": True}}]})
    launched = rt.launch(artifact, (x, wd, wk, wv, q))
    assert launched["ok"], launched.get("reason")
    fused = launched["output"]
    composed = run_flash_attention_forward(q, k, v, scale=.5, causal=True)
    scores = np.einsum("bhid,bhjd->bhij", q, k) * .5
    scores = np.where(np.arange(k.shape[2])[None,None,None,:] <=
                      np.arange(q.shape[2])[None,None,:,None], scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    oracle = (probs / probs.sum(axis=-1, keepdims=True)) @ v
    np.testing.assert_allclose(fused, composed, atol=8e-5, rtol=0)
    np.testing.assert_allclose(fused, oracle, atol=8e-5, rtol=0)


def _cuda_runtime():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):
        pytest.skip("nvcc unavailable")
    from tessera import runtime as rt
    if not rt._nvidia_mma_runtime_available():
        pytest.skip("CUDA unavailable")
    return rt
