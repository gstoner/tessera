from __future__ import annotations
import os,shutil
import numpy as np
import pytest
from tessera.stdlib import moe

def _live():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):return False
    from tessera import runtime as rt
    return rt._nvidia_mma_runtime_available()

def _art(rt,op,names):
    return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120","compiler_path":"nvidia_moe_transport_compiled","executable":True,"execution_kind":"native_gpu","arg_names":names,"output_name":"o","ops":[{"op_name":op,"result":"o","operands":names,"kwargs":{}}]})

def _plan(seed=1):
    rng=np.random.default_rng(seed);ids=rng.integers(0,4,(12,2),dtype=np.int64);w=rng.random((12,2),dtype=np.float32);w/=w.sum(1,keepdims=True)
    return moe.plan_dispatch(ids,w,4,capacity=5)

@pytest.mark.skipif(not _live(),reason="requires nvcc and NVIDIA GPU")
def test_dispatch_and_combine_match_oracles():
    from tessera import runtime as rt
    rng=np.random.default_rng(3);x=rng.standard_normal((12,9)).astype(np.float32);plan=_plan(3)
    d=rt.launch(_art(rt,"tessera.moe_dispatch",["x","plan"]),(x,plan));assert d["execution_kind"]=="native_gpu";np.testing.assert_array_equal(d["output"],moe.dispatch(x,plan))
    partials=d["output"]*rng.uniform(.8,1.2,d["output"].shape).astype(np.float32)
    c=rt.launch(_art(rt,"tessera.moe_combine",["partials","plan"]),(partials,plan));assert c["execution_kind"]=="native_gpu";np.testing.assert_allclose(c["output"],moe.combine(partials,plan),rtol=1e-5,atol=1e-6)

@pytest.mark.skipif(not _live(),reason="requires nvcc and NVIDIA GPU")
def test_grouped_gemm_matches_per_expert_oracle():
    from tessera import runtime as rt
    from tessera.compiler.grouped_layout import reference_grouped_gemm
    rng=np.random.default_rng(8);sizes=np.array([2,0,3,1],np.int64);x=rng.standard_normal((6,7)).astype(np.float32);w=rng.standard_normal((4,7,5)).astype(np.float32)
    out=rt.launch(_art(rt,"tessera.grouped_gemm",["x","weights","group_sizes"]),{"x":x,"weights":w,"group_sizes":sizes})["output"]
    np.testing.assert_allclose(out,reference_grouped_gemm(x,w,sizes),rtol=2e-5,atol=2e-5)

def test_grouped_gemm_rejects_bad_partition_without_cuda():
    from tessera.compiler.emit.nvidia_cuda import run_grouped_gemm_f32
    with pytest.raises(ValueError,match="K/group_sizes"):
        run_grouped_gemm_f32(np.zeros((4,3),np.float32),np.zeros((2,3,2),np.float32),np.array([1,1]))
