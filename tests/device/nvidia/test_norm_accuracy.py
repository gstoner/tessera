"""Long-row BF16 regression with the original composed numerical gate."""
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs,runtime_artifact
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import rms_lhs,_storage,_oracle

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning SM120 host required")

@pytest.mark.parametrize("columns",[4096,8192])
@pytest.mark.parametrize("schedule",["serial","cooperative_128"])
def test_long_bf16_norm_matmul_preserves_original_oracle_gate(columns,schedule):
    rng=np.random.default_rng(120517)
    source=(rng.normal(size=(128,columns))*.2).astype(_storage("bf16"))
    rhs=(rng.normal(size=(columns,64))*.2).astype(_storage("bf16"))
    module=rms_lhs._traced_autodiff_module((source,rhs),{})
    program=package_traced_lhs(module,producer_schedule=schedule)
    receipt=rt.launch(runtime_artifact(program),(source,rhs))
    assert receipt["ok"],receipt
    np.testing.assert_allclose(receipt["output"],_oracle(source,rhs,"rmsnorm"),
        rtol=.015,atol=.015)
    assert "math.sqrt" in program.edge.producer.target_ir
    assert "nvvm.rsqrt" not in program.edge.producer.target_ir


@pytest.mark.parametrize("dtype",["fp16","bf16","fp32"])
@pytest.mark.parametrize("schedule",["serial","cooperative_128"])
def test_norm_compensation_preserves_nonfinite_rms_outputs(dtype,schedule):
    from tests.device.nvidia.test_cooperative_norm import package,launch
    storage=np.float32 if dtype=="fp32" else _storage(dtype)
    source=np.full((2,257),.5,storage)
    source[0,0]=np.inf
    source[1,0]=np.nan
    _,native=package("rmsnorm",source,schedule)
    actual=launch(native,source).astype(np.float32)
    expected=np.zeros(source.shape,np.float32)
    expected[0,0]=np.nan
    expected[1,:]=np.nan
    np.testing.assert_array_equal(actual,expected)
