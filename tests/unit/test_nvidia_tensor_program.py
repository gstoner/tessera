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


def _program(dtype="fp16", dynamic_n=False, output_dtype="fp32"):
    m, k, n = 16, 16, 16 if dynamic_n else 8
    elem = "f16" if dtype == "fp16" else "bf16"
    a = IRType(f"tensor<{m}x{k}x{elem}>", (str(m), str(k)), dtype)
    b = (
        IRType(f"tensor<{k}x?x{elem}>", (str(k), "?"), dtype)
        if dynamic_n
        else IRType(f"tensor<{k}x{n}x{elem}>", (str(k), str(n)), dtype)
    )
    output_elem = "f16" if output_dtype == "fp16" else "f32"
    out = (
        IRType(f"tensor<{m}x?x{output_elem}>", (str(m), "?"), output_dtype)
        if dynamic_n
        else IRType(
            f"tensor<{m}x{n}x{output_elem}>", (str(m), str(n)), output_dtype
        )
    )
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
            operand_types=[str(a), str(b)], result_type=str(out),
            kwargs={"shape_bounds": [m, n, k]} if dynamic_n else {},
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


def test_sm120_rmsnorm_tensor_edge_accepts_bounded_dynamic_consumer_n():
    program = _program(dynamic_n=True)
    program.validate()
    assert program.dynamic_n
    assert program.consumer.descriptor.provenance["dynamic_shape_bounds"] == [
        program.m, program.n, program.k
    ]
    assert program.consumer.descriptor.provenance["leading_dimension_abi"] == "runtime_i64"
    rhs_guards = [
        guard for guard in program.consumer.descriptor.shape_guards
        if guard.binding == program.consumer_rhs_name
    ]
    assert [(guard.dimension, guard.predicate, guard.value) for guard in rhs_guards] == [
        (0, "eq", program.k), (1, "max", program.n)
    ]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("output_dtype", ["fp32", "fp16"])
def test_sm120_rmsnorm_tensor_edge_reuses_dynamic_n_package_on_exact_device(
    dtype, output_dtype,
):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires exact SM120 device execution")
    import ml_dtypes

    storage_dtype = np.float16 if dtype == "fp16" else np.dtype(ml_dtypes.bfloat16)
    program = _program(dtype, dynamic_n=True, output_dtype=output_dtype)
    program.validate()
    rng = np.random.default_rng(17016)
    source = np.ascontiguousarray(
        rng.normal(0.0, 0.25, (program.m, program.k)).astype(storage_dtype)
    )
    consumer_digest = program.consumer.image.image_digest
    for active_n in (7, program.n):
        rhs = np.asfortranarray(
            rng.normal(0.0, 0.25, (program.k, active_n)).astype(storage_dtype)
        )
        with program.execute_resident(source, rhs) as result:
            assert result.output.shape == (program.m, active_n)
            assert program.consumer.image.image_digest == consumer_digest
            intermediate = result.intermediate.numpy()
            output = result.output.numpy()
            expected = intermediate.astype(np.float32) @ rhs.astype(np.float32)
            if output_dtype == "fp16":
                expected = expected.astype(np.float16)
            np.testing.assert_allclose(
                output, expected, rtol=0.0,
                atol=2e-3 if output_dtype == "fp16" else 5e-4,
            )
            assert result.producer_receipt["execution_kind"] == "native_gpu"
            assert result.consumer_receipt["execution_kind"] == "native_gpu"

def test_sm120_rmsnorm_tensor_edge_packages_public_frontend_graph_ir():
    from tessera.compiler.from_text import from_text

    producer_jit = from_text("""
        def rmsnorm_frontend(x):
            return ts.ops.rmsnorm(x, eps=1e-5)
    """)
    consumer_jit = from_text("""
        def matmul_frontend(normalized, weights):
            return ts.ops.matmul(normalized, weights)
    """)
    source = np.ascontiguousarray(np.ones((16, 16), dtype=np.float16))
    producer_result = producer_jit(source)
    weights = np.asfortranarray(np.ones((16, 8), dtype=np.float16))
    consumer_result = consumer_jit(producer_result, weights)
    assert producer_jit.frontend_authority == "tracer"
    assert consumer_jit.frontend_authority == "tracer"
    assert producer_jit.graph_ir.module_attrs["tessera.frontend.authority"] == '"tracer"'
    assert consumer_jit.graph_ir.module_attrs["tessera.frontend.authority"] == '"tracer"'
    program = nvidia_native.package_scheduled_rmsnorm_matmul(
        producer_jit.graph_ir, consumer_jit.graph_ir,
        pipeline_name="tessera-lower-to-nvidia-sm120",
    )
    program.validate()
    assert program.producer.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    assert program.consumer.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    assert program.consumer.descriptor.provenance["epilogue"]["output"] == "f16"
    assert consumer_result.dtype == np.float16


def test_sm120_rmsnorm_tensor_edge_executes_public_frontend_on_exact_device():
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires exact SM120 device execution")
    from tessera.compiler.from_text import from_text

    producer_jit = from_text("""
        def rmsnorm_frontend(x):
            return ts.ops.rmsnorm(x, eps=1e-5)
    """)
    consumer_jit = from_text("""
        def matmul_frontend(normalized, weights):
            return ts.ops.matmul(normalized, weights)
    """)
    rng = np.random.default_rng(18016)
    source = np.ascontiguousarray(rng.normal(0.0, 0.25, (16, 16)).astype(np.float16))
    producer_result = producer_jit(source)
    weights = np.asfortranarray(rng.normal(0.0, 0.25, (16, 8)).astype(np.float16))
    consumer_jit(producer_result, weights)
    program = nvidia_native.package_scheduled_rmsnorm_matmul(
        producer_jit.graph_ir, consumer_jit.graph_ir,
        pipeline_name="tessera-lower-to-nvidia-sm120",
    )
    with program.execute_resident(source, weights) as result:
        edge = result.intermediate.numpy()
        output = result.output.numpy()
        x32 = source.astype(np.float32)
        expected_norm = (
            x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + 1e-5)
        ).astype(np.float16)
        np.testing.assert_allclose(edge, expected_norm, rtol=0.0, atol=2e-3)
        expected = (edge.astype(np.float32) @ weights.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(output, expected, rtol=0.0, atol=2e-3)
        assert result.producer_receipt["execution_kind"] == "native_gpu"
        assert result.consumer_receipt["execution_kind"] == "native_gpu"
