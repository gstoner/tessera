"""Owning SM120 long-row producer integration across compiler-selected routes."""
import numpy as np
import pytest
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs, layer_lhs, softmax_lhs, rms_lhs_fused, layer_lhs_fused,
    softmax_lhs_fused, _storage, _oracle,
)

pytestmark = pytest.mark.skipif(
    not nvidia_cuda_host_ready(), reason="owning SM120 host required")


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("kind", ["rmsnorm", "layernorm", "softmax"])
@pytest.mark.parametrize("epilogue", [False, True])
def test_long_producer_selected_route_and_output_lifetime(dtype, kind, epilogue):
    rng = np.random.default_rng(20261008)
    source = (rng.normal(size=(128, 4096)) * .2).astype(_storage(dtype))
    rhs = np.array(rng.normal(size=(4096, 64)) * .2,
                   dtype=_storage(dtype), order="F")
    bias = (rng.normal(size=64) * .02).astype(np.float32)
    residual = (rng.normal(size=(128, 64)) * .02).astype(np.float32)
    operands = (source, rhs, bias, residual) if epilogue else (source, rhs)
    plain = {"rmsnorm": rms_lhs, "layernorm": layer_lhs, "softmax": softmax_lhs}
    fused = {"rmsnorm": rms_lhs_fused, "layernorm": layer_lhs_fused,
             "softmax": softmax_lhs_fused}
    program = (fused if epilogue else plain)[kind].compile_native_lhs_matmul(*operands)
    descriptor = program.edge.consumer.descriptor
    route = ("typed_fragment_global" if epilogue else
             "macro_cta_cp_async_2stage_shared_ab_" +
             ("f16" if dtype == "fp16" else "bf16"))
    geometry = ("sm120_scheduled_typed_16x8_mn" if epilogue else
                "sm120_scheduled_macro_cta_32x32_mn")
    assert descriptor.provenance["physical_route"] == route
    assert descriptor.geometry.policy == geometry
    call = PreparedLhsCall(program)
    try:
        first, receipt = call(operands)
        expected = _oracle(source, rhs, kind,
                           bias if epilogue else None,
                           residual if epilogue else None)
        np.testing.assert_allclose(first, expected, rtol=.015, atol=.015)
        saved = first.copy()
        changed = (-source, *operands[1:])
        second, second_receipt = call(changed)
        expected = _oracle(changed[0], rhs, kind,
                           bias if epilogue else None,
                           residual if epilogue else None)
        np.testing.assert_allclose(second, expected, rtol=.015, atol=.015)
        np.testing.assert_array_equal(first, saved)
        assert receipt["execution_kind"] == second_receipt["execution_kind"] == "native_gpu"
    finally:
        call.close()
