"""Native Graph/Schedule/Tile/Target contracts for positive-stride pages.

These are compiler proofs. Owning-device execution remains a separate gate.
"""
import os
from pathlib import Path
import re
import subprocess

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt


def graph(arch="gfx1151", layout='"strided"', table_layout="", reverse=False):
    page = f'%pages: tensor<4x3x2x5xf32> {{tessera.layout = {layout}}}' if layout else '%pages: tensor<4x3x2x5xf32>'
    table = '%table: tensor<4xi32>' + table_layout
    args = (table + ", " + page) if reverse else (page + ", " + table)
    target = "nvidia_sm120" if arch == "sm_120" else "rocm_" + arch
    return f"""module attributes {{tessera.target = "{target}", tessera.arch = "{arch}"}} {{
      func.func @read({args}) -> tensor<7x2x5xf32>
          attributes {{tessera.bindings = ["pages", "table", "out"]}} {{
        %out = tessera.paged_kv_read %pages, %table {{start = 2 : i64, end = 9 : i64}}
          : (tensor<4x3x2x5xf32>, tensor<4xi32>) -> tensor<7x2x5xf32>
        return %out : tensor<7x2x5xf32>
      }}
    }}"""


def run(tool, source, *args, ok=True):
    result = subprocess.run([str(tool), *args], input=source, text=True, capture_output=True)
    if ok:
        assert result.returncode == 0, result.stderr
        return result.stdout
    assert result.returncode != 0, result.stdout
    return result.stderr


@pytest.fixture
def tool():
    found = find_tessera_opt()
    if found is None:
        pytest.skip("requires production tessera-opt")
    return found


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("reverse", [False, True])
def test_strided_pages_preserve_graph_binding_and_lower_four_runtime_strides(tool, arch, reverse):
    schedule = run(tool, graph(arch, reverse=reverse), "--tessera-graph-to-schedule")
    assert 'page_stride_policy = "positive_runtime_element_strides"' in schedule
    assert 'source_extent_policy = "checked_physical_span"' in schedule
    tile = run(tool, schedule, "--tessera-schedule-to-tile")
    assert "llvm.func @tessera_tile_paged_kv_read_f32_strided" in tile
    assert 'page_layout = "strided"' in tile
    signature = re.search(r"llvm.func @[^\(]+\(([^)]*)\)", tile).group(1)
    assert signature.count("!llvm.ptr") == 3
    assert signature.count("i64") == 11


def test_explicit_row_major_keeps_the_existing_compact_contract(tool):
    implicit = run(tool, graph(layout=""), "--tessera-graph-to-schedule")
    explicit = run(tool, graph(layout='"row_major"'), "--tessera-graph-to-schedule")
    assert re.findall(r'artifact_hash = "([0-9a-f]{64})"', implicit) == re.findall(
        r'artifact_hash = "([0-9a-f]{64})"', explicit)
    tile = run(tool, explicit, "--tessera-schedule-to-tile")
    assert "llvm.func @tessera_tile_paged_kv_read_f32_direct" in tile
    assert "page_stride_policy" not in tile
    assert "page_layout" not in tile


@pytest.mark.parametrize("layout", ['"col_major"', "7 : i64"])
def test_unknown_or_untyped_source_layout_is_rejected(tool, layout):
    assert "row_major or strided pages" in run(
        tool, graph(layout=layout), "--tessera-graph-to-schedule", ok=False)


def test_table_layout_cannot_be_silently_treated_as_compact(tool):
    assert "compact table" in run(tool, graph(table_layout=' {tessera.layout = "strided"}'),
                                 "--tessera-graph-to-schedule", ok=False)


def test_sm120_requires_its_own_strided_target_consumer(tool):
    assert "owning ROCm Target consumer" in run(
        tool, graph("sm_120"), "--tessera-graph-to-schedule", ok=False)


def test_stride_policy_is_part_of_sealed_schedule_contract(tool):
    schedule = run(tool, graph(), "--tessera-graph-to-schedule")
    changed = schedule.replace("positive_runtime_element_strides", "unchecked_byte_strides")
    assert changed != schedule
    assert "contract changed after hashing" in run(
        tool, changed, "--tessera-schedule-to-tile", ok=False)


@pytest.mark.parametrize("layout", ["7 : i64", '"row_major"'])
def test_tile_marker_rejects_untyped_or_ambiguous_stride_policy(tool, layout):
    schedule = run(tool, graph(), "--tessera-graph-to-schedule")
    tile = run(tool, schedule, "--tessera-schedule-to-tile")
    changed = tile.replace('page_layout = "strided"', "page_layout = " + layout)
    assert changed != tile
    assert "four element strides" in run(tool, changed, ok=False)


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
def test_rocm_target_retains_layout_and_materializes_four_stride_arguments(tool, arch):
    target_tool = Path(os.environ.get('TESSERA_ROCM_OPT', 'missing-tessera-rocm-opt'))
    if not target_tool.is_file():
        pytest.skip("requires matching tessera-rocm-opt")
    schedule = run(tool, graph(arch), "--tessera-graph-to-schedule")
    tile = run(tool, schedule, "--tessera-schedule-to-tile")
    target = run(target_tool, tile, f"--pass-pipeline=builtin.module(lower-tile-to-rocm{{arch={arch}}})")
    assert 'page_layout = "strided"' in target
    kernel = run(target_tool, target, "--generate-rocm-paged-kv-read-kernel")
    signature = re.search(r"gpu.func @[^\(]+\(([^)]*)\)", kernel).group(1)
    assert signature.count("memref<") == 3
    assert signature.count(": index") == 11
    assert "arith.divui" in kernel and "arith.remui" in kernel
    assert "tile.paged_kv_read_kernel" not in kernel
    assert "tessera_rocm.paged_kv_read" not in kernel


def test_frontend_gather_result_does_not_inherit_the_page_storage_view():
    from tessera.compiler.graph_ir import _shape_kv_cache_read, tensor_ir_type

    pages = tensor_ir_type((4, 3, 2, 5), "fp32", layout="strided")
    table = tensor_ir_type((4,), "int32")
    output = _shape_kv_cache_read([pages, table], {"start": 2, "end": 9})
    assert output.shape == ("7", "2", "5")
    assert output.layout == "row_major"
