"""SM120-W11-FP8-RHS-20261026: matmul(A, dequantize_scaled(B_codes, s)).

The B gather decodes each FP8 code exactly to A's half storage and the
per-tensor scale multiplies the f32 accumulator once, so the fused result
must equal f32(native_half_matmul(A, decode(B)) * s) bit for bit."""
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type
from tessera.compiler.nvidia_native import package_scheduled_matmul
from tessera.compiler.scheduled_matmul import lower_scheduled_fp8_rhs_matmul, lower_scheduled_matmul

pytestmark = pytest.mark.hardware_nvidia

DTYPES = {"fp16": np.dtype(np.float16), "bf16": np.dtype(ml_dtypes.bfloat16),
          "fp8_e4m3": np.dtype(ml_dtypes.float8_e4m3fn), "fp8_e5m2": np.dtype(ml_dtypes.float8_e5m2)}
PIPELINE = "tessera-nvidia-pipeline-sm120"


def modules(storage, fmt, m, k, n):
    a = tensor_ir_type((m, k), storage)
    bq = tensor_ir_type((k, n), fmt)
    s = tensor_ir_type((1, 1), "fp32")
    b = tensor_ir_type((k, n), "fp32")
    half_b = tensor_ir_type((k, n), storage)
    d = tensor_ir_type((m, n), "fp32")
    dequant = IROp(result="b", op_name="tessera.dequantize_scaled", operands=["%bq", "%s"],
                   operand_types=[str(bq), str(s)], result_type=str(b), kwargs={})
    matmul = IROp(result="d", op_name="tessera.matmul", operands=["%a", "%b"],
                  operand_types=[str(a), str(b)], result_type=str(d), kwargs={"rhs_storage_order": "row_major"})
    fused = GraphIRModule(functions=[GraphIRFunction(
        name="fp8_rhs", args=[IRArg("a", a), IRArg("bq", bq), IRArg("s", s)], result_types=[d],
        body=[dequant, matmul], return_values=["%d"])])
    plain_matmul = IROp(result="d", op_name="tessera.matmul", operands=["%a", "%b"],
                        operand_types=[str(a), str(half_b)], result_type=str(d),
                        kwargs={"rhs_storage_order": "row_major"})
    plain = GraphIRModule(functions=[GraphIRFunction(
        name="half_rhs", args=[IRArg("a", a), IRArg("b", half_b)], result_types=[d],
        body=[plain_matmul], return_values=["%d"])])
    return fused, plain


def launch(package, arguments):
    artifact = rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"}, native_image=package.image,
                                  launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
                                  target_ir=package.target_ir)
    receipt = rt.launch(artifact, arguments)
    assert receipt.get("ok") is True and receipt.get("execution_kind") == "native_gpu", receipt


@pytest.mark.parametrize("storage", ["fp16"])
@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("shape", [(16, 16, 8), (37, 40, 29), (128, 256, 96), (1, 2048, 256)])
@pytest.mark.parametrize("scale", [1.0, 0.0625, 3.3e-3])
def test_fused_fp8_rhs_equals_scaled_half_matmul_of_decoded_codes(storage, fmt, shape, scale):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    m, k, n = shape
    fused, plain = modules(storage, fmt, m, k, n)
    package = package_scheduled_matmul(lower_scheduled_fp8_rhs_matmul(fused, target="nvidia_sm120"),
                                       pipeline_name=PIPELINE)
    assert package.descriptor.abi_id.endswith(f".{'f16' if storage == 'fp16' else 'bf16'}_{fmt}.v1")
    baseline = package_scheduled_matmul(lower_scheduled_matmul(plain, target="nvidia_sm120"),
                                        pipeline_name=PIPELINE)
    rng = np.random.default_rng(20261026)
    a = rng.normal(0, 1, (m, k)).astype(DTYPES[storage])
    codes = rng.normal(0, 4, (k, n)).astype(DTYPES[fmt])
    s = np.full((1, 1), scale, np.float32)
    out = np.full((m, n), np.nan, np.float32)
    launch(package, {"a": a, "bq": codes, "s": s, "d": out, "M": m, "N": n, "K": k})
    unscaled = np.full((m, n), np.nan, np.float32)
    launch(baseline, {"a": a, "b": codes.astype(np.float32).astype(DTYPES[storage]), "d": unscaled,
                      "M": m, "N": n, "K": k})
    np.testing.assert_array_equal(out, unscaled * np.float32(scale))
    # And against the declared Graph composition in fp64, within f32 accumulation.
    reference = a.astype(np.float64) @ ts.ops.dequantize_scaled(codes, s).astype(np.float64)
    bound = 1e-5 * (np.abs(a.astype(np.float64)) @ np.abs(ts.ops.dequantize_scaled(codes, s).astype(np.float64))) + 1e-30
    assert np.all(np.abs(out - reference) <= bound)


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_every_finite_code_decodes_exactly(fmt):
    """With A = I, D = s * decode(B): each finite code must come back exact."""
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    every = np.arange(256, dtype=np.uint8).view(DTYPES[fmt])
    finite = every[np.isfinite(every.astype(np.float32))]
    k = 16
    n = -(-finite.size // k) * 8
    codes = np.zeros((k, n), DTYPES[fmt])
    codes.reshape(-1)[:finite.size] = finite
    fused, _ = modules("fp16", fmt, k, k, n)
    package = package_scheduled_matmul(lower_scheduled_fp8_rhs_matmul(fused, target="nvidia_sm120"),
                                       pipeline_name=PIPELINE)
    out = np.empty((k, n), np.float32)
    launch(package, {"a": np.eye(k, dtype=np.float16), "bq": codes, "s": np.ones((1, 1), np.float32),
                     "d": out, "M": k, "N": n, "K": k})
    np.testing.assert_array_equal(out, codes.astype(np.float32))


def test_largest_f16_and_e5m2_magnitudes_stay_finite():
    """The factored scale is admitted because the unscaled f32 accumulator
    cannot overflow for f16 A: here every term is 65504 * 57344 over K = 2048."""
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    m, k, n = 16, 2048, 8
    fused, plain = modules("fp16", "fp8_e5m2", m, k, n)
    package = package_scheduled_matmul(lower_scheduled_fp8_rhs_matmul(fused, target="nvidia_sm120"),
                                       pipeline_name=PIPELINE)
    baseline = package_scheduled_matmul(lower_scheduled_matmul(plain, target="nvidia_sm120"),
                                        pipeline_name=PIPELINE)
    a = np.full((m, k), 65504, np.float16)
    codes = np.full((k, n), 57344, DTYPES["fp8_e5m2"])
    s = np.full((1, 1), 2.0 ** -20, np.float32)
    out = np.empty((m, n), np.float32)
    launch(package, {"a": a, "bq": codes, "s": s, "d": out, "M": m, "N": n, "K": k})
    unscaled = np.empty((m, n), np.float32)
    launch(baseline, {"a": a, "b": codes.astype(np.float32).astype(np.float16), "d": unscaled,
                      "M": m, "N": n, "K": k})
    assert np.isfinite(out).all() and np.isfinite(unscaled).all()
    np.testing.assert_array_equal(out, unscaled * np.float32(2.0 ** -20))
    np.testing.assert_allclose(out, 65504.0 * 57344.0 * k * 2.0 ** -20, rtol=1e-6)


def test_bf16_lhs_is_refused():
    """bf16 spans the f32 range: 2**120 x 448 would overflow the unscaled
    accumulator where the Graph's per-element product stays finite."""
    fused, _ = modules("bf16", "fp8_e4m3", 16, 16, 8)
    with pytest.raises(ValueError, match="f16 lhs"):
        lower_scheduled_fp8_rhs_matmul(fused, target="nvidia_sm120")

