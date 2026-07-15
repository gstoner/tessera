from __future__ import annotations
import numpy as np
import pytest
from _nvidia_testutil import nvidia_mma_runtime_available
from tessera.stdlib import quant

def _art(rt,op,names):
    return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120","compiler_path":"nvidia_dequant_gemm_compiled","executable":True,"execution_kind":"native_gpu","arg_names":names,"output_name":"o","ops":[{"op_name":op,"result":"o","operands":names,"kwargs":{}}]})

@pytest.mark.skipif(not nvidia_mma_runtime_available(),reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype",["int4","int8"])
def test_dequant_matmul_is_fused_and_matches_oracle(dtype):
    from tessera import runtime as rt
    rng=np.random.default_rng(20+len(dtype));x=rng.standard_normal((5,16)).astype(np.float32);p=quant.quantize_weight(rng.standard_normal((16,9)).astype(np.float32),dtype,group_size=4)
    res=rt.launch(_art(rt,"tessera.dequant_matmul",["x","packed_w"]),(x,p));assert res["execution_kind"]=="native_gpu"
    np.testing.assert_allclose(res["output"],quant.dequant_matmul(x,p,backend="reference"),rtol=2e-6,atol=2e-6)

@pytest.mark.skipif(not nvidia_mma_runtime_available(),reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype",["int4","int8"])
def test_dequant_grouped_gemm_single_launch_matches_oracle(dtype):
    from tessera import runtime as rt
    rng=np.random.default_rng(30+len(dtype));gs=np.array([2,0,3],np.int64);x=rng.standard_normal((5,12)).astype(np.float32);ps=[quant.quantize_weight(rng.standard_normal((12,7)).astype(np.float32),dtype,group_size=3) for _ in range(3)]
    res=rt.launch(_art(rt,"tessera.dequant_grouped_gemm",["x","packed_experts","group_sizes"]),{"x":x,"packed_experts":ps,"group_sizes":gs});assert res["execution_kind"]=="native_gpu"
    np.testing.assert_allclose(res["output"],quant.dequant_grouped_gemm(x,ps,gs),rtol=2e-6,atol=2e-6)
