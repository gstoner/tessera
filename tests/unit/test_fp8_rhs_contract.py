"""SM120-W11-FP8-RHS-20261026: fused FP8 norm output and FP8 row-major-B matmul."""
import re
import subprocess

import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.compiler_route

HEADER = 'module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {'
FP8 = {"fp8_e4m3": "f8E4M3FN", "fp8_e5m2": "f8E5M2"}


@pytest.fixture
def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("production tessera-opt required")
    return tool


def norm_graph(storage="bf16", fmt="fp8_e4m3", scale="1x1", rows=64, cols=256, kind="rmsnorm", extra_use=False):
    x = f"tensor<{rows}x{cols}x{storage}>"
    s = f"tensor<{scale}xf32>"
    q = f"tensor<{rows}x{cols}x{FP8[fmt]}>"
    results = f"({q}, {x})" if extra_use else q
    returns = f"%q, %y : {q}, {x}" if extra_use else f"%q : {q}"
    return f'''{HEADER}
func.func @f(%x: {x}, %s: {s}) -> {results} {{
  %y = tessera.{kind} %x {{eps = 1.0e-05 : f64}} : ({x}) -> {x}
  %q = tessera.quantize_scaled %y, %s {{format = "{fmt}"}} : ({x}, {s}) -> {q}
  return {returns}
}}}}'''


def matmul_graph(storage="f16", fmt="fp8_e4m3", scale="1x1", order="row_major", m=32, k=64, n=48):
    a, bq = f"tensor<{m}x{k}x{storage}>", f"tensor<{k}x{n}x{FP8[fmt]}>"
    s, b, d = f"tensor<{scale}xf32>", f"tensor<{k}x{n}xf32>", f"tensor<{m}x{n}xf32>"
    return f'''{HEADER}
func.func @f(%a: {a}, %bq: {bq}, %s: {s}) -> {d} {{
  %b = tessera.dequantize_scaled %bq, %s : ({bq}, {s}) -> {b}
  %d = tessera.matmul %a, %b {{rhs_storage_order = "{order}"}} : ({a}, {b}) -> {d}
  return %d : {d}
}}}}'''


def fails(compiler, text, option, message):
    result = subprocess.run([str(compiler), option], input=text, capture_output=True, text=True)
    assert result.returncode != 0 and message in result.stderr, result.stderr[-400:]


@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("fmt", sorted(FP8))
@pytest.mark.parametrize("kind", ["rmsnorm", "layer_norm"])
@pytest.mark.parametrize("cols", [96, 256])
def test_fused_norm_schedules_and_tiles_one_scaled_fp8_store(compiler, storage, fmt, kind, cols):
    schedule = run_tessera_opt(compiler, norm_graph(storage, fmt, kind=kind, cols=cols), "--tessera-graph-to-schedule")
    row = [line for line in schedule.splitlines() if "schedule.norm" in line]
    assert len(row) == 1 and f'output_storage = "{fmt}"' in row[0] and 'scale = "per_tensor_f32"' in row[0]
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    entry = re.findall(r"llvm.func @(\w+)\(([^)]*)\)", tile)
    norm_kind = "rmsnorm" if kind == "rmsnorm" else "layernorm"
    assert len(entry) == 1 and entry[0][0].startswith(f"tessera_tile_norm_{norm_kind}_{storage}_to_{fmt}_")
    assert entry[0][1].count("!llvm.ptr") == 3
    kernel = [line for line in tile.splitlines() if "tile.norm_kernel" in line]
    assert len(kernel) == 1 and f'output_storage = "{fmt}"' in kernel[0]


def test_unfused_norm_schedule_is_unchanged(compiler):
    text = f'''{HEADER}
func.func @f(%x: tensor<64x256xbf16>) -> tensor<64x256xbf16> {{
  %y = tessera.rmsnorm %x {{eps = 1.0e-05 : f64}} : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
  return %y : tensor<64x256xbf16>
}}}}'''
    schedule = run_tessera_opt(compiler, text, "--tessera-graph-to-schedule")
    assert "output_storage" not in schedule and 'scale = "' not in schedule


@pytest.mark.parametrize("edits", [{"scale": "64x1"}, {"scale": "1x256"}, {"extra_use": True}])
def test_fused_norm_refuses_non_per_tensor_scale_or_a_second_use(compiler, edits):
    fails(compiler, norm_graph(**edits), "--tessera-graph-to-schedule", "semantic-kernel contract")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("fmt", sorted(FP8))
def test_fused_matmul_schedules_fp8_b_with_a_per_tensor_scale(compiler, storage, fmt):
    schedule = run_tessera_opt(compiler, matmul_graph(storage, fmt), "--tessera-graph-to-schedule")
    row = [line for line in schedule.splitlines() if "schedule.matmul" in line]
    assert len(row) == 1
    for field in (f'storage = "{storage}"', f'storage_b = "{fmt.removeprefix("fp8_")}"',
                  'b_scale = "per_tensor_f32"', 'b_layout = "row_major"', 'output = "f32"'):
        assert field in row[0]


def test_fused_matmul_requires_row_major_b(compiler):
    fails(compiler, matmul_graph(order="col_major"), "--tessera-graph-to-schedule", "row_major")


@pytest.mark.parametrize("edits", [{"scale": "1x48"}, {"scale": "64x1"}, {"storage": "f32"}])
def test_fused_matmul_refuses_other_scales_and_storage(compiler, edits):
    fails(compiler, matmul_graph(**edits), "--tessera-graph-to-schedule", "E2E-REAL-2 Graph->Schedule")


def test_half_matmul_schedule_has_no_b_scale(compiler):
    text = f'''{HEADER}
func.func @f(%a: tensor<32x64xf16>, %b: tensor<64x48xf16>) -> tensor<32x48xf32> {{
  %d = tessera.matmul %a, %b {{rhs_storage_order = "row_major"}} : (tensor<32x64xf16>, tensor<64x48xf16>) -> tensor<32x48xf32>
  return %d : tensor<32x48xf32>
}}}}'''
    assert "b_scale" not in run_tessera_opt(compiler, text, "--tessera-graph-to-schedule")


def test_fused_matmul_tile_decodes_b_and_scales_the_store(compiler):
    schedule = run_tessera_opt(compiler, matmul_graph(), "--tessera-graph-to-schedule")
    result = subprocess.run([str(compiler), "--tessera-schedule-to-tile"], input=schedule,
                            capture_output=True, text=True)
    if result.returncode != 0 and "registered NVIDIA Target IR dialect" in result.stderr:
        pytest.skip("this tessera-opt has no NVIDIA Target IR dialect")
    assert result.returncode == 0, result.stderr[-400:]
    tile = result.stdout
    entry = re.findall(r"llvm.func @(\w+)\(([^)]*)\)", tile)
    assert len(entry) == 1 and entry[0][0].endswith("_rhs_fp8_e4m3_scaled_kernel")
    assert entry[0][1].count("!llvm.ptr") == 4
    assert 'decode = "fp8_e4m3"' in tile and 'tile.output_scale = "per_tensor_f32"' in tile
