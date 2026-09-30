from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tessera.compiler import nvidia_native, scheduled_kernel, scheduled_matmul
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera import runtime as rt


pytestmark = pytest.mark.skipif(
    scheduled_matmul.find_tessera_opt() is None or not nvidia_native.tools_available(),
    reason="requires the production SM120 Schedule/Target compiler tools",
)


def _program(dtype="fp16"):
    m, k, n = 16, 16, 8
    elem = "f16" if dtype == "fp16" else "bf16"
    a = IRType(f"tensor<{m}x{k}x{elem}>", (str(m), str(k)), dtype)
    b = IRType(f"tensor<{k}x{n}x{elem}>", (str(k), str(n)), dtype)
    out = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    producer_module = GraphIRModule(functions=[GraphIRFunction(
        name="sm120_rmsnorm_tensor_producer",
        args=[IRArg("x", a)],
        result_types=[a],
        body=[IROp(
            result="normalized", op_name="tessera.rmsnorm",
            operands=["%x"], operand_types=[str(a)], result_type=str(a),
            kwargs={"eps": 1e-5},
        )],
        return_values=["%normalized"],
    )])
    consumer_module = GraphIRModule(functions=[GraphIRFunction(
        name="sm120_rmsnorm_matmul_consumer",
        args=[IRArg("normalized", a), IRArg("weights", b)],
        result_types=[out],
        body=[IROp(
            result="result", op_name="tessera.matmul",
            operands=["%normalized", "%weights"],
            operand_types=[str(a), str(b)], result_type=str(out), kwargs={},
        )],
        return_values=["%result"],
    )])
    producer = scheduled_kernel.lower_scheduled_kernel(
        producer_module, target="nvidia_sm120",
    )
    consumer = scheduled_matmul.lower_scheduled_matmul(
        consumer_module, target="nvidia_sm120",
    )
    return nvidia_native.package_scheduled_rmsnorm_matmul(
        producer, consumer, pipeline_name="tessera-lower-to-nvidia-sm120",
    )


def test_sm120_rmsnorm_tensor_edge_keeps_both_canonical_packages():
    program = _program()
    program.validate()
    assert program.producer.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    assert program.consumer.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    assert program.producer.descriptor.provenance["schedule_digest"]
    assert program.consumer.descriptor.provenance["schedule_digest"]
    assert "tile.norm_kernel" in program.producer.tile_ir
    assert "tile.view" in program.consumer.tile_ir
    assert "tile.fragment_pack" in program.consumer.tile_ir
    assert program.intermediate_name == "normalized"
    assert program.consumer_input_name == "normalized"


def test_sm120_rmsnorm_tensor_edge_rejects_shape_drift_and_aliasing():
    program = _program()
    with pytest.raises(ValueError, match="shapes do not match"):
        replace(program, k=15).validate()

    source = np.ones((program.m, program.k), dtype=np.float16)
    weights = np.asfortranarray(np.ones((program.k, program.n), dtype=np.float16))
    with pytest.raises(ValueError, match="must not alias"):
        program.execute(source, weights, intermediate=source)


def test_sm120_rmsnorm_tensor_edge_uses_same_resident_device_buffer(monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires exact SM120 device execution")
    program = _program()
    rng = np.random.default_rng(17)
    source = np.ascontiguousarray(
        rng.normal(0.0, 0.25, (program.m, program.k)).astype(np.float16)
    )
    rhs = np.asfortranarray(
        rng.normal(0.0, 0.25, (program.k, program.n)).astype(np.float16)
    )
    original_launch = rt.launch
    launch_args = []

    def capture_launch(kernel, args, stream=None):
        launch_args.append((args, stream))
        return original_launch(kernel, args, stream=stream)

    monkeypatch.setattr(rt, "launch", capture_launch)
    with program.execute_resident(source, rhs) as result:
        producer_args, producer_stream = launch_args[0]
        consumer_args, consumer_stream = launch_args[1]
        producer_buffer = producer_args[program.intermediate_name]
        consumer_buffer = consumer_args[program.consumer_input_name]
        assert producer_buffer.ptr == consumer_buffer.ptr
        assert producer_stream == consumer_stream == result.device_session.stream
        intermediate = result.intermediate.numpy()
        output = result.output.numpy()
        reference = intermediate.astype(np.float32) @ rhs.astype(np.float32)
        assert np.max(np.abs(output - reference)) < 2e-4
        assert result.producer_receipt["execution_kind"] == "native_gpu"
        assert result.consumer_receipt["execution_kind"] == "native_gpu"


@pytest.mark.parametrize("producer_stream", [None, 1, 2, 0x5678])
def test_sm120_resident_launch_rejects_unordered_cuda_buffer_stream(producer_stream):
    interface = {"stream": producer_stream}
    with pytest.raises(RuntimeError, match="producer stream must match"):
        rt._validate_nvidia_cuda_buffer_streams([interface], 0x1234)


def test_sm120_resident_launch_requires_matching_cuda_buffer_stream():
    interface = {"stream": 0x1234}
    rt._validate_nvidia_cuda_buffer_streams([interface], 0x1234)


def test_sm120_resident_launch_rejects_missing_cuda_buffer_stream():
    with pytest.raises(RuntimeError, match="producer stream must match"):
        rt._validate_nvidia_cuda_buffer_streams([{}], 0x1234)


def test_sm120_bf16_rmsnorm_tensor_edge_executes_on_one_resident_allocation():
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires exact SM120 device execution")
    import ml_dtypes

    program = _program("bf16")
    program.validate()
    rng = np.random.default_rng(119)
    source = np.ascontiguousarray(
        rng.normal(0.0, 0.25, (program.m, program.k)).astype(ml_dtypes.bfloat16)
    )
    rhs = np.asfortranarray(
        rng.normal(0.0, 0.25, (program.k, program.n)).astype(ml_dtypes.bfloat16)
    )
    with program.execute_resident(source, rhs) as result:
        intermediate = result.intermediate.numpy()
        output = result.output.numpy()
        source_f32 = source.astype(np.float32)
        expected_norm = source_f32 / np.sqrt(
            np.mean(source_f32 * source_f32, axis=-1, keepdims=True) + 1e-5
        )
        np.testing.assert_allclose(
            intermediate.astype(np.float32), expected_norm,
            rtol=8e-3, atol=8e-3,
        )
        reference = intermediate.astype(np.float32) @ rhs.astype(np.float32)
        np.testing.assert_allclose(output, reference, rtol=0.0, atol=2e-5)
        assert result.producer_receipt["execution_kind"] == "native_gpu"
        assert result.consumer_receipt["execution_kind"] == "native_gpu"


def test_cuda_buffer_preserves_registered_bf16_dtype_metadata():
    import ml_dtypes

    class Buffer:
        dtype = np.dtype(ml_dtypes.bfloat16)
        shape = (2, 3)
        flags = type("Flags", (), {"c_contiguous": True, "f_contiguous": False})()

        @property
        def __cuda_array_interface__(self):
            return {
                "shape": self.shape,
                "strides": None,
                "typestr": self.dtype.str,
                "data": (0x1000, False),
                "version": 3,
                "stream": 0x2000,
            }

    _, argument = rt._native_buffer_value(Buffer())
    assert argument.dtype == "bf16"
    assert argument.shape == (2, 3)
    assert argument.layout == "row_major"
