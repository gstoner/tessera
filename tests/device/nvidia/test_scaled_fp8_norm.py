"""SM120-W11-FP8-RHS-20261026: RMSNorm/LayerNorm fused with quantize_scaled.

The fused kernel must store exactly quantize_scaled(norm(x), s): the codes are
compared bit for bit against the reference applied to the unfused native norm
output, which isolates the fused store from reduction-order differences."""
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type
from tessera.compiler.nvidia_native import package_scheduled_kernel
from tessera.compiler.scheduled_kernel import lower_scheduled_fp8_norm, lower_scheduled_kernel

pytestmark = pytest.mark.hardware_nvidia

DTYPES = {"fp16": np.dtype(np.float16), "bf16": np.dtype(ml_dtypes.bfloat16), "fp32": np.dtype(np.float32),
          "fp8_e4m3": np.dtype(ml_dtypes.float8_e4m3fn), "fp8_e5m2": np.dtype(ml_dtypes.float8_e5m2)}
PIPELINE = "tessera-nvidia-pipeline-sm120"


def modules(storage, fmt, rows, columns, kind):
    x = tensor_ir_type((rows, columns), storage)
    s = tensor_ir_type((1, 1), "fp32")
    q = tensor_ir_type((rows, columns), fmt)
    op_name = "tessera.rmsnorm" if kind == "rmsnorm" else "tessera.layer_norm"
    norm = IROp(result="y", op_name=op_name, operands=["%x"], operand_types=[str(x)],
                result_type=str(x), kwargs={"eps": 1e-5})
    quantize = IROp(result="q", op_name="tessera.quantize_scaled", operands=["%y", "%s"],
                    operand_types=[str(x), str(s)], result_type=str(q), kwargs={"format": fmt})
    fused = GraphIRModule(functions=[GraphIRFunction(
        name="fp8_norm", args=[IRArg("x", x), IRArg("s", s)], result_types=[q],
        body=[norm, quantize], return_values=["%q"])])
    plain = GraphIRModule(functions=[GraphIRFunction(
        name="plain_norm", args=[IRArg("x", x)], result_types=[x], body=[norm], return_values=["%y"])])
    return fused, plain


def artifact(package):
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"}, native_image=package.image,
                              launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
                              target_ir=package.target_ir)


def launch(package, arguments):
    receipt = rt.launch(artifact(package), arguments)
    assert receipt.get("ok") is True and receipt.get("execution_kind") == "native_gpu", receipt


@pytest.mark.parametrize("storage", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("kind", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("columns", [96, 512])  # serial and cooperative_128 schedules
@pytest.mark.parametrize("scale", [1.0, 0.0078125, 3.3e-3, 1e-6, 640.0])
def test_fused_codes_equal_quantize_of_the_native_norm(storage, fmt, kind, columns, scale):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    rows = 37
    fused, plain = modules(storage, fmt, rows, columns, kind)
    scaled_artifact = lower_scheduled_fp8_norm(fused, target="nvidia_sm120")
    assert scaled_artifact.schedule == ("serial" if columns < 256 else "cooperative_128")
    package = package_scheduled_kernel(scaled_artifact, pipeline_name=PIPELINE)
    assert package.descriptor.entry_symbol.startswith(f"tessera_tile_norm_{kind}_")
    assert f"_to_{fmt}" in package.descriptor.entry_symbol
    baseline = package_scheduled_kernel(lower_scheduled_kernel(plain, target="nvidia_sm120"),
                                        pipeline_name=PIPELINE)
    rng = np.random.default_rng(20261026)
    x = rng.normal(0, 2, (rows, columns)).astype(DTYPES[storage])
    x[0, :4] = 0  # rows with zeros and a large outlier exercise both ends
    x[1, 0] = 300
    s = np.full((1, 1), scale, np.float32)
    normalized = np.empty_like(x)
    launch(baseline, {"x": x, "y": normalized, "Rows": rows, "Columns": columns})
    expected = ts.ops.quantize_scaled(normalized, s, format=fmt)
    arena = np.full(rows * columns + 32, 0x5A, np.uint8)
    codes = arena[16:-16].view(DTYPES[fmt]).reshape(rows, columns)
    launch(package, {"x": x, "s": s, "q": codes, "Rows": rows, "Columns": columns})
    np.testing.assert_array_equal(codes.view(np.uint8), expected.view(np.uint8))
    np.testing.assert_array_equal(arena[:16], np.full(16, 0x5A, np.uint8))
    np.testing.assert_array_equal(arena[-16:], np.full(16, 0x5A, np.uint8))


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_fused_store_saturates_and_keeps_nan(fmt):
    """Rows whose normalized values overflow the format saturate to the
    largest finite code; a NaN source row stays NaN."""
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    rows, columns = 4, 128
    fused, plain = modules("fp32", fmt, rows, columns, "rmsnorm")
    package = package_scheduled_kernel(lower_scheduled_fp8_norm(fused, target="nvidia_sm120"),
                                       pipeline_name=PIPELINE)
    baseline = package_scheduled_kernel(lower_scheduled_kernel(plain, target="nvidia_sm120"),
                                        pipeline_name=PIPELINE)
    x = np.random.default_rng(3).normal(0, 1, (rows, columns)).astype(np.float32)
    x[3, 5] = np.nan
    s = np.full((1, 1), 1e-9, np.float32)  # every finite value saturates
    normalized = np.empty_like(x)
    launch(baseline, {"x": x, "y": normalized, "Rows": rows, "Columns": columns})
    codes = np.empty((rows, columns), DTYPES[fmt])
    launch(package, {"x": x, "s": s, "q": codes, "Rows": rows, "Columns": columns})
    expected = ts.ops.quantize_scaled(normalized, s, format=fmt)
    np.testing.assert_array_equal(codes.view(np.uint8), expected.view(np.uint8))
    largest = float(ml_dtypes.finfo(DTYPES[fmt]).max)
    decoded = codes.astype(np.float32)
    assert np.all(np.abs(decoded[:3]) == largest)
    assert np.isnan(decoded[3]).all()
