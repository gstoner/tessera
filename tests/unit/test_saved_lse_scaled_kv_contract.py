"""SM120-SAVED-LSE-SCALED-FP8-KV-20261025: dequantize_scaled + saved-LSE fusion."""
import re
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.compiler_route

FP8 = {"fp8_e4m3": "f8E4M3FN", "fp8_e5m2": "f8E5M2"}


def module(q="f16", kv="f8E4M3FN", k_scale="1x1x5x1", v_scale="1x1x1x3", bias=None,
           output=None, target="nvidia_sm120", arch="sm_120", key_dequant=True, value_dequant=True,
           scale_storage="f32"):
    output = output or q
    args = [f"%q: tensor<1x2x3x4x{q}>", f"%k8: tensor<1x1x5x4x{kv}>", f"%v8: tensor<1x1x5x3x{kv}>",
            f"%ks: tensor<{k_scale}x{scale_storage}>", f"%vs: tensor<{v_scale}x{scale_storage}>"]
    names = ["q", "k", "v", "k_scale", "v_scale"]
    if bias:
        args.append(f"%bias: tensor<{bias}xf32>")
        names.append("bias")
    body = []
    key, value = "%k8", "%v8"
    key_type, value_type = f"tensor<1x1x5x4x{kv}>", f"tensor<1x1x5x3x{kv}>"
    if key_dequant:
        body.append(f"  %k = tessera.dequantize_scaled %k8, %ks : (tensor<1x1x5x4x{kv}>, "
                    f"tensor<{k_scale}x{scale_storage}>) -> tensor<1x1x5x4xf32>")
        key, key_type = "%k", "tensor<1x1x5x4xf32>"
    if value_dequant:
        body.append(f"  %v = tessera.dequantize_scaled %v8, %vs : (tensor<1x1x5x3x{kv}>, "
                    f"tensor<{v_scale}x{scale_storage}>) -> tensor<1x1x5x3xf32>")
        value, value_type = "%v", "tensor<1x1x5x3xf32>"
    operands = ["%q", key, value] + (["%bias"] if bias else [])
    types = [f"tensor<1x2x3x4x{q}>", key_type, value_type] + ([f"tensor<{bias}xf32>"] if bias else [])
    results = f"tensor<1x2x3x3x{output}>, tensor<1x2x3xf32>"
    body.append(f'  %o, %lse = "tessera.flash_attn"({", ".join(operands)}) {{lse_checkpoint = "saved", '
                f'head_dim = 4 : i64, causal = true, operandSegmentSizes = array<i32: 1, 1, 1, {int(bool(bias))}>}} '
                f': ({", ".join(types)}) -> ({results})')
    body.append(f"  return %o, %lse : {results}")
    bindings = ", ".join(f'"{n}"' for n in names)
    return (f'module attributes {{tessera.target = "{target}", tessera.arch = "{arch}"}} {{\n'
            f' func.func @saved({", ".join(args)}) -> ({results})\n'
            f' attributes {{tessera.argument_bindings = [{bindings}], tessera.result_bindings = ["o", "lse"]}} {{\n'
            + "\n".join(body) + "\n }\n}\n")


@pytest.fixture
def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("production tessera-opt required")
    return tool


def _tile(compiler, text):
    return run_tessera_opt(compiler, run_tessera_opt(compiler, text, "--tessera-graph-to-schedule"),
                           "--tessera-schedule-to-tile")


def test_dequantize_scaled_reference_rounds_the_product_once():
    codes = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).reshape(4, 64)
    scale = np.linspace(0.01, 3.0, 64, dtype=np.float32).reshape(1, 64)
    got = ts.ops.dequantize_scaled(codes, scale)
    assert got.dtype == np.float32 and got.shape == (4, 64)
    want = (codes.astype(np.float64) * scale.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("codes,scale,message", [
    (np.zeros((2, 3), np.float16), np.ones((1, 3), np.float32), "fp8_e4m3 or fp8_e5m2"),
    (np.zeros((2, 3), ml_dtypes.float8_e4m3fn), np.ones((1, 3), np.float64), "f32 scale"),
    (np.zeros((2, 3), ml_dtypes.float8_e4m3fn), np.ones((2, 2), np.float32), "each be 1"),
    (np.zeros((2, 3), ml_dtypes.float8_e4m3fn), np.ones((3,), np.float32), "each be 1"),
])
def test_dequantize_scaled_reference_refusals(codes, scale, message):
    with pytest.raises(ValueError, match=message):
        ts.ops.dequantize_scaled(codes, scale)


@pytest.mark.parametrize("kv", sorted(FP8))
@pytest.mark.parametrize("q", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("scales", [("1x1x1x1", "1x1x1x1"), ("1x1x5x1", "1x1x1x3"), ("1x1x5x4", "1x1x5x3")])
@pytest.mark.parametrize("bias", [None, "1x2x1x1", "1x2x3x5"])
def test_scaled_chain_fuses_into_one_checkpoint(compiler, kv, q, scales, bias):
    tile = _tile(compiler, module(q=q, kv=FP8[kv], k_scale=scales[0], v_scale=scales[1], bias=bias))
    assert "tessera.dequantize_scaled" not in tile and "tessera.flash_attn" not in tile
    contract = re.search(r"tessera.native_contract = \{([^\n]*)\}", tile)
    assert contract is not None
    shapes = ", ".join(scales[0].split("x") + scales[1].split("x"))
    assert f"kv_scale_shapes = array<i64: {shapes}>" in contract[1]
    assert 'kv_dequantize = "code_times_f32_scale_round_once_v1"' in contract[1]
    assert f'kv_storage = "{kv}"' in contract[1]
    entry = re.findall(r"llvm.func @([\w]+)\(", tile)
    assert len(entry) == 1 and f"_kv_{kv}_scaled_" in entry[0]
    kernel = [line for line in tile.splitlines() if "tile.attention_kernel" in line]
    assert len(kernel) == 1 and f"kv_scale_shapes = array<i64: {shapes}>" in kernel[0]


@pytest.mark.parametrize("edits,reason", [
    ({"value_dequant": False, "kv": "f8E4M3FN"}, "both K and V dequantize_scaled"),
    ({"key_dequant": False}, "both K and V dequantize_scaled"),
    ({"k_scale": "1x1x2x1"}, "each scale axis must be 1 or the codes extent"),
    ({"scale_storage": "f16"}, "f32 scale"),
    ({"kv": "f16"}, "f8E4M3FN or f8E5M2"),
    ({"output": "bf16"}, "output storage must be f32 or match Q"),
    # Off SM120 the importer never fuses; the generic attention rule refuses.
    ({"target": "rocm", "arch": "gfx1201"}, "Graph->Schedule requires"),
])
def test_scaled_chain_refusals(compiler, edits, reason):
    process = subprocess.run([str(compiler), "--tessera-graph-to-schedule"],
                             input=module(**edits), capture_output=True, text=True)
    assert process.returncode == 1, process.stdout + process.stderr
    assert reason in process.stderr


@pytest.mark.parametrize("bias", [None, "1x2x1x1"])
def test_scaled_descriptor_is_entry_ordered_with_scale_scalars(compiler, monkeypatch, bias):
    from tessera.compiler import nvidia_native as native, scheduled_checkpoint as checkpoint
    graph = module(bias=bias)
    schedule = run_tessera_opt(compiler, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    scheduled = checkpoint._decode_checkpoint(graph, schedule, tile, backward=False)
    assert scheduled.kv_scale_shapes == (1, 1, 5, 1, 1, 1, 1, 3)
    monkeypatch.setattr(native, "_compile_tile_ir",
        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    descriptor = native.package_scheduled_checkpoint(
        scheduled, pipeline_name="tessera-nvidia-pipeline-sm120").descriptor
    buffers = sorted(descriptor.buffers, key=lambda value: value.ordinal)
    expected = ["q", "k", "v"] + (["bias"] if bias else []) + ["k_scale", "v_scale", "o", "lse"]
    assert [b.name for b in buffers] == expected
    assert [b.dtype for b in buffers[:3]] == ["fp16", "fp8_e4m3", "fp8_e4m3"]
    scalars = [s.name for s in sorted(descriptor.scalars, key=lambda value: value.ordinal)]
    assert scalars[-8:] == ["KScaleB", "KScaleH", "KScaleS", "KScaleD",
                            "VScaleB", "VScaleH", "VScaleS", "VScaleD"]
    assert ("BiasB" in scalars) == bool(bias)
    assert descriptor.abi_id in native.SM120_ATTN_LSE_SCALED_KV_ABIS
    unscaled = native._checkpoint_identity(scheduled.dims, scheduled.scale, scheduled.causal,
        bias=bool(bias), storage="f16", output_storage="f16", kv_storage="fp8_e4m3")
    assert descriptor.provenance["checkpoint_contract"] != unscaled


def test_public_trace_emits_the_dequantize_chain():
    from tessera.compiler.nvidia_native import requests_scaled_kv_attention, supports_attention_scaled_kv
    def scaled(q, k8, v8, ks, vs):
        k = ts.ops.dequantize_scaled(k8, ks)
        v = ts.ops.dequantize_scaled(v8, vs)
        return ts.ops.flash_attn(q, k, v, causal=True, lse_checkpoint="saved")
    rng = np.random.default_rng(0)
    values = (rng.normal(0, .2, (1, 2, 3, 4)).astype(np.float16),
              rng.normal(0, 2, (1, 1, 5, 4)).astype(ml_dtypes.float8_e4m3fn),
              rng.normal(0, 2, (1, 1, 5, 3)).astype(ml_dtypes.float8_e4m3fn),
              np.full((1, 1, 5, 1), .05, np.float32), np.full((1, 1, 1, 3), .07, np.float32))
    fn = ts.jit(target="nvidia_sm120")(scaled)
    assert requests_scaled_kv_attention(fn.graph_ir)
    traced = fn._trace_frontend_capture(values, {})[0]
    assert supports_attention_scaled_kv(traced)


def test_single_result_scaled_chain_declines_instead_of_crashing():
    """Codex PR #1014 review: a chain returning only O is outside the
    two-result checkpoint envelope and must decline, not raise."""
    from tessera.compiler.nvidia_native import supports_attention_scaled_kv
    def only_output(q, k8, v8, ks, vs):
        k = ts.ops.dequantize_scaled(k8, ks)
        v = ts.ops.dequantize_scaled(v8, vs)
        o, lse = ts.ops.flash_attn(q, k, v, causal=True, lse_checkpoint="saved")
        return o
    rng = np.random.default_rng(0)
    values = (rng.normal(0, .2, (1, 2, 3, 4)).astype(np.float16),
              rng.normal(0, 2, (1, 1, 5, 4)).astype(ml_dtypes.float8_e4m3fn),
              rng.normal(0, 2, (1, 1, 5, 3)).astype(ml_dtypes.float8_e4m3fn),
              np.full((1, 1, 5, 1), .05, np.float32), np.full((1, 1, 1, 3), .07, np.float32))
    traced = ts.jit(target="nvidia_sm120")(only_output)._trace_frontend_capture(values, {})[0]
    assert supports_attention_scaled_kv(traced) is False
