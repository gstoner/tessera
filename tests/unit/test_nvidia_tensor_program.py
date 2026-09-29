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


def _program():
    m, k, n = 16, 16, 8
    a = IRType(f"tensor<{m}x{k}xf16>", (str(m), str(k)), "fp16")
    b = IRType(f"tensor<{k}x{n}xf16>", (str(k), str(n)), "fp16")
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
