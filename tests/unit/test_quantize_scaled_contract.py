"""SM120-W11-FP8-RHS-20261026: the quantize_scaled Graph op and its reference."""
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.scheduled_matmul import find_tessera_opt

FORMATS = {"fp8_e4m3": (ml_dtypes.float8_e4m3fn, "f8E4M3FN", 448.0),
           "fp8_e5m2": (ml_dtypes.float8_e5m2, "f8E5M2", 57344.0)}


@pytest.fixture
def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("production tessera-opt required")
    return tool


def module(values="bf16", scale="1x1", codes="f8E4M3FN", fmt="fp8_e4m3", shape="4x8"):
    v, s, c = f"tensor<{shape}x{values}>", f"tensor<{scale}xf32>", f"tensor<{shape}x{codes}>"
    return (f"func.func @f(%x: {v}, %s: {s}) -> {c} {{\n"
            f'  %q = tessera.quantize_scaled %x, %s {{format = "{fmt}"}} : ({v}, {s}) -> {c}\n'
            f"  return %q : {c}\n}}")


@pytest.mark.parametrize("values", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("scale", ["1x1", "4x1", "1x8", "4x8"])
def test_verifier_admits_declared_storage_and_broadcast_scale(compiler, values, fmt, scale):
    result = subprocess.run([str(compiler)], input=module(values, scale, FORMATS[fmt][1], fmt),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("edits,message", [
    ({"fmt": "fp8_e5m2"}, "name the result storage"),
    ({"fmt": "e4m3"}, "name the result storage"),
    ({"codes": "f16"}, "name the result storage"),
    ({"values": "i8"}, "f16, bf16 or f32 values"),
    ({"scale": "2x8"}, "1 or the values extent"),
    ({"scale": "8"}, "scale rank"),
])
def test_verifier_refuses_inconsistent_contracts(compiler, edits, message):
    result = subprocess.run([str(compiler)], input=module(**edits), capture_output=True, text=True)
    assert result.returncode != 0 and message in result.stderr


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_reference_saturates_and_round_trips_every_finite_code(fmt):
    target, _, largest = FORMATS[fmt]
    edge = np.array([[0.0, -0.0, 1.0, 1e9, -1e9, np.inf, -np.inf, np.nan]], np.float32)
    codes = ts.ops.quantize_scaled(edge, np.ones((1, 1), np.float32), format=fmt)
    assert codes.dtype == np.dtype(target)
    decoded = codes.astype(np.float32)
    np.testing.assert_array_equal(decoded[0, :7], [0, 0, 1, largest, -largest, largest, -largest])
    assert np.signbit(decoded[0, 1]) and np.isnan(decoded[0, 7])
    assert codes.view(np.uint8)[0, 7] == 0x7F  # the canonical NaN code
    negative_nan = ts.ops.quantize_scaled(np.array([[-np.nan]], np.float32), np.ones((1, 1), np.float32), format=fmt)
    assert negative_nan.view(np.uint8)[0, 0] == 0x7F
    every = np.arange(256, dtype=np.uint8).view(target)
    every = every[np.isfinite(every.astype(np.float32))].reshape(1, -1)
    scale = np.full((1, 1), 0.375, np.float32)
    again = ts.ops.quantize_scaled(ts.ops.dequantize_scaled(every, scale), scale, format=fmt)
    np.testing.assert_array_equal(again.view(np.uint8), every.view(np.uint8))


def test_reference_requires_format_and_f32_broadcast_scale():
    x = np.ones((2, 3), np.float16)
    with pytest.raises(TypeError):
        ts.ops.quantize_scaled(x, np.ones((1, 1), np.float32))
    with pytest.raises(ValueError, match="format"):
        ts.ops.quantize_scaled(x, np.ones((1, 1), np.float32), format="e4m3")
    with pytest.raises(ValueError, match="f32 scale"):
        ts.ops.quantize_scaled(x, np.ones((1, 1), np.float16), format="fp8_e4m3")
    with pytest.raises(ValueError, match="values extent"):
        ts.ops.quantize_scaled(x, np.ones((2, 2), np.float32), format="fp8_e4m3")


def test_shape_rule_names_fp8_storage_and_refuses_a_missing_format():
    from tessera.compiler.graph_ir import _shape_quantize_scaled, tensor_ir_type
    values = tensor_ir_type((4, 8), "bf16")
    scale = tensor_ir_type((1, 1), "fp32")
    assert _shape_quantize_scaled([values, scale], {"format": "fp8_e5m2"}).dtype == "fp8_e5m2"
    with pytest.raises(ValueError, match="format"):
        _shape_quantize_scaled([values, scale], {})
