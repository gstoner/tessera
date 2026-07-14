"""Native CUDA RoPE/ALiBi positional encoding contracts."""

from __future__ import annotations
import os, shutil
import numpy as np
import pytest


def _live():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):
        return False
    from tessera import runtime
    return runtime._nvidia_mma_runtime_available()


def _artifact(rt, op, operands, args, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_posenc_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": args, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": operands,
                 "kwargs": kwargs}]})


@pytest.mark.skipif(not _live(), reason="requires nvcc and a live NVIDIA GPU")
def test_rope_runtime_matches_pairwise_oracle():
    from tessera import runtime as rt
    rng = np.random.default_rng(11); x = rng.standard_normal((3, 5, 32)).astype(np.float32)
    theta = rng.uniform(-2, 2, x.shape).astype(np.float32)
    out = rt.launch(_artifact(rt, "tessera.rope", ["x", "theta"],
                             ["x", "theta"], {}), (x, theta))["output"]
    ref = np.empty_like(x); e=x[...,0::2]; o=x[...,1::2]; a=theta[...,0::2]
    ref[...,0::2]=e*np.cos(a)-o*np.sin(a); ref[...,1::2]=e*np.sin(a)+o*np.cos(a)
    np.testing.assert_allclose(out, ref, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not _live(), reason="requires nvcc and a live NVIDIA GPU")
def test_alibi_default_and_explicit_slopes():
    from tessera import runtime as rt
    h,s=4,17
    for slopes in (None, np.array([.5,.25,.125,.0625],np.float32)):
        operands=["slopes"] if slopes is not None else []; args=operands
        art=_artifact(rt,"tessera.alibi",operands,args,{"num_heads":h,"seq_len":s})
        out=rt.launch(art, (slopes,) if slopes is not None else tuple())["output"]
        sl=(2.0**(-8*np.arange(1,h+1,dtype=np.float32)/h) if slopes is None else slopes)
        pos=np.arange(s,dtype=np.float32); ref=sl[:,None,None]*(pos[None,:]-pos[:,None])[None]
        np.testing.assert_allclose(out,ref,rtol=1e-6,atol=1e-6)


def test_posenc_rejects_invalid_contract_without_cuda():
    from tessera.compiler.emit import nvidia_cuda as nv
    with pytest.raises(ValueError, match="even final"):
        nv.run_rope_f32(np.zeros((2,7),np.float32),np.zeros((2,7),np.float32))
    with pytest.raises(ValueError, match="length 3"):
        nv.run_alibi_f32(3,4,np.ones(2,np.float32))
