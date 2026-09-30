from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler import nvidia_native
from tessera import runtime as rt


@pytest.mark.hardware_nvidia
@pytest.mark.skipif(
    rt._nvidia_device_name() != "sm_120",
    reason="requires exact SM120 device execution",
)
def test_sm120_public_frontend_rmsnorm_matmul_explicit_fp32_output():
    """The traced public output policy reaches the native resident ABI."""
    from tessera.compiler.from_text import from_text

    producer_jit = from_text("""
        def rmsnorm_frontend(x):
            return ts.ops.rmsnorm(x, eps=1e-5)
    """)
    consumer_jit = from_text("""
        def matmul_frontend(normalized, weights):
            return ts.ops.matmul(normalized, weights, output_dtype="fp32")
    """)
    rng = np.random.default_rng(18032)
    source = np.ascontiguousarray(rng.normal(0.0, 0.25, (16, 16)).astype(np.float16))
    producer_jit(source)
    weights = np.asfortranarray(rng.normal(0.0, 0.25, (16, 8)).astype(np.float16))
    consumer_jit(np.zeros_like(source), weights)
    assert consumer_jit.graph_ir.functions[0].result_types[0].dtype == "fp32"
    program = nvidia_native.package_scheduled_rmsnorm_matmul(
        producer_jit.graph_ir,
        consumer_jit.graph_ir,
        pipeline_name="tessera-lower-to-nvidia-sm120",
    )
    assert program.consumer.descriptor.provenance["epilogue"]["output"] == "f32"
    with program.execute_resident(source, weights) as result:
        intermediate = result.intermediate.numpy()
        output = result.output.numpy()
        x32 = source.astype(np.float32)
        expected_norm = (
            x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + 1e-5)
        ).astype(np.float16)
        np.testing.assert_allclose(intermediate, expected_norm, rtol=0.0, atol=2e-3)
        expected = intermediate.astype(np.float32) @ weights.astype(np.float32)
        assert output.dtype == np.float32
        np.testing.assert_allclose(output, expected, rtol=0.0, atol=2e-5)
        assert result.producer_receipt["execution_kind"] == "native_gpu"
        assert result.consumer_receipt["execution_kind"] == "native_gpu"
