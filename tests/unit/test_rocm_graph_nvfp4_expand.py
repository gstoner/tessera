"""Native NVFP4 source-preserving Graph/Schedule/Tile contract tests.

This is a compiler route test; numerical/device evidence belongs to the
gfx1201 owning-device lane, not to successful IR emission.
"""
import subprocess

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt

POLICY = '''execution_mode = "source_preserving_expansion",
 source_format = "nvfp4_e2m1_e4m3_k16",
 destination_format = "fp8_e4m3_k16_f32_scales",
 source_scale_application = "projection_global_times_e4m3",
 destination_code_selection = "exact_e2m1_signed_zero",
 destination_scale_order = "k_group_n",
 scale_rounding = "f64_product_to_f32_rne"'''


def graph(k=48):
    types = f"tensor<7x{k // 2}xi8>, tensor<7x{k // 16}xf8E4M3FN>, tensor<2xf64>"
    outputs = f"tensor<7x{k}xf8E4M3FN>, tensor<{k // 16}x7xf32>"
    return f'''module attributes {{tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201"}} {{
 func.func @expand(%codes: tensor<7x{k // 2}xi8>, %scales: tensor<7x{k // 16}xf8E4M3FN>, %globals: tensor<2xf64>) -> ({outputs})
 attributes {{tessera.bindings = ["codes", "scales", "globals", "values", "scale_plane"]}} {{
 %v, %s = "tessera.nvfp4_expand"(%codes, %scales, %globals) {{
 row_offsets = [0 : i64, 3 : i64, 7 : i64], numeric_policy = {{{POLICY}}}
 }} : ({types}) -> ({outputs})
 return %v, %s : {outputs}
 }}
}}'''


@pytest.fixture(scope="module")
def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("matching compiler required")
    return str(tool)


def run(tool, source, *passes):
    return subprocess.run([tool, *passes], input=source, text=True, capture_output=True)


@pytest.mark.parametrize("k", [16, 32, 48, 80])
def test_native_expansion_route(compiler, k):
    schedule = run(compiler, graph(k), "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    assert 'family=nvfp4_expand' in schedule.stdout
    tile = run(compiler, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    assert "tile.nvfp4_expand_kernel" in tile.stdout
    assert "memref<?xf32>" in tile.stdout
    target = run(compiler, tile.stdout, "--lower-tile-to-rocm=arch=gfx1201")
    assert target.returncode == 0, target.stderr
    assert "tessera_rocm.nvfp4_expand" in target.stdout
    native = run(compiler, target.stdout, "--generate-rocm-fpquant-kernel")
    assert native.returncode == 0, native.stderr
    assert "gpu.func" in native.stdout
    assert "arith.mulf" in native.stdout
    assert "arith.truncf" in native.stdout
    assert "tessera_rocm.nvfp4_expand" not in native.stdout


@pytest.mark.parametrize("old,new", [
    ('"source_preserving_expansion"', '"approximate"'),
    ('"f64_product_to_f32_rne"', '"truncate"'),
    ('"exact_e2m1_signed_zero"', '"absolute_values"'),
    ('"k_group_n"', '"n_k_group"'),
    ('tensor<7x48xf8E4M3FN>', 'tensor<7x48xf32>'),
    ('tensor<3x7xf32>', 'tensor<7x3xf32>'),
    ('[0 : i64, 3 : i64, 7 : i64]', '[0 : i64, 7 : i64, 3 : i64]'),
    ('"values", "scale_plane"', '"codes", "scale_plane"'),
    ('tessera.target = "rocm_gfx1201"', 'tessera.target = "rocm_gfx1151"'),
])
def test_rejects_changed_graph_contract(compiler, old, new):
    result = run(compiler, graph().replace(old, new), "--tessera-graph-to-schedule")
    assert result.returncode != 0


@pytest.mark.parametrize("old,new", [
    ("private_outputs_distinct_readonly_inputs", "aliased_outputs"),
    ("exact_e2m1_signed_zero", "other_code_policy"),
    ('"scale_plane"', '"values"'),
])
def test_schedule_replay_rejects_mutation(compiler, old, new):
    schedule = run(compiler, graph(), "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    assert old in schedule.stdout
    result = run(compiler, schedule.stdout.replace(old, new), "--tessera-schedule-to-tile")
    assert result.returncode != 0
