"""Owning-device regression for checked broadcast copies and private AD state."""
import pytest
from tessera.compiler import nvidia_native
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support.nvidia import nvidia_cuda_host_ready
from benchmarks.nvidia.benchmark_checkpoint_bias_gradient import run_case as checked
from benchmarks.nvidia.benchmark_jit_attention_bias_vjp import run_case as public

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.skipif(
    not nvidia_cuda_host_ready() or not nvidia_native.tools_available() or find_tessera_opt() is None,
    reason="requires SM120 GPU and matching native compilers")]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("bias_gradient", [False, True])
def test_checked_broadcast_physical_outputs(causal, bias_gradient):
    result = checked((2,4,2,5,7,4,3),causal,samples=1,reps=20,bias_shape=(1,1,1,7),bias_gradient=bias_gradient)
    expected = (nvidia_native.SM120_ATTN_BWD_LSE_BCAST_GRAD_F32_ABI if bias_gradient
                else nvidia_native.SM120_ATTN_BWD_LSE_BCAST_F32_ABI)
    assert result["abi"] == expected


@pytest.mark.parametrize("wrt", [("bias","q","v"), ("q","v")])
def test_public_broadcast_private_capture(wrt):
    result = public((2,4,2,7,5,4,3),True,wrt,samples=1,bias_shape=(1,4,1,1))
    # Native paired AD computes all operand cotangents; public selection
    # returns only the requested roles without changing the saved-state ABI.
    assert result["backward_abi"] == nvidia_native.SM120_ATTN_BWD_LSE_BCAST_GRAD_F32_ABI
    assert len(result["active"]) == len(wrt)
