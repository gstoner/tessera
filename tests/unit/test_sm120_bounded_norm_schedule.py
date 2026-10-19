"""Bounded-query RMSNorm stays semantic until native Schedule/Tile lowering."""
import re

import pytest

from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.compiler_route


def source(storage="f16", capacity=(2, 3, 7, 35)):
    b, h, sq, d = capacity
    tensor = f"tensor<{b}x{h}x?x{d}x{storage}>"
    return f'''module attributes {{tessera.arch = "sm_120", tessera.target = "nvidia_sm120"}} {{
      func.func @norm(%x: {tensor}) -> {tensor} attributes {{
          tessera.native.query_sequence_bound = {sq} : i64}} {{
        %out = tessera.rmsnorm %x {{eps = 1.000000e-05 : f64,
            numeric_policy = {{accum = "fp32"}}}} : ({tensor}) -> {tensor}
        return %out : {tensor}
      }}
    }}'''


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("capacity", [(2, 3, 7, 35), (1, 2, 17, 257)])
def test_bounded_norm_reaches_native_with_capacity_guards(compiler_toolchain, storage, capacity):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    graph = source(storage, capacity)
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    assert f"tensor<{capacity[0]}x{capacity[1]}x?x{capacity[-1]}x{storage}>" in schedule.stdout
    assert 'shape_policy = "bounded_query_v1"' in schedule.stdout
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    for name, value in (("sequence_bound", capacity[2]),
                        ("row_capacity", capacity[0] * capacity[1] * capacity[2]),
                        ("row_multiple", capacity[0] * capacity[1]),
                        ("fixed_columns", capacity[-1])):
        assert f"{name} = {value} : i64" in tile.stdout
    assert 'shape_policy = "bounded_query_v1"' in tile.stdout
    target = invoke(tool, tile.stdout, "--lower-tile-to-nvidia=sm=120")
    assert target.returncode == 0, target.stderr
    assert "tile.norm_kernel" not in target.stdout
    assert "arith.remui" in target.stdout
    assert "arith.cmpi ule" in target.stdout
    native = invoke(tool, target.stdout, "--lower-tessera-nvidia-to-nvvm")
    assert native.returncode == 0, native.stderr
    assert "llvm.func" in native.stdout


@pytest.mark.parametrize("fault", ["missing_bound", "zero", "negative", "overflow", "wrong_type",
                                   "static_query", "dynamic_columns", "other_target", "f32", "layernorm", "arg_layout"])
def test_bounded_norm_rejects_unproved_graph_envelopes(compiler_toolchain, fault):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    graph = source()
    if fault == "missing_bound":
        graph = graph.replace("tessera.native.query_sequence_bound = 7 : i64", "")
    elif fault in ("zero", "negative", "overflow"):
        graph = graph.replace("= 7 : i64", f'= {dict(zero=0, negative=-1, overflow=2147483647)[fault]} : i64')
    elif fault == "wrong_type":
        graph = graph.replace("= 7 : i64", "= 7 : i32")
    elif fault == "static_query":
        graph = graph.replace("2x3x?x35", "2x3x7x35")
    elif fault == "dynamic_columns":
        graph = graph.replace("2x3x?x35", "2x3x?x?")
    elif fault == "other_target":
        graph = graph.replace("nvidia_sm120", "rocm_gfx1201").replace("sm_120", "gfx1201")
    elif fault == "f32":
        graph = graph.replace("xf16", "xf32")
    elif fault == "arg_layout":
        graph = graph.replace("%x: tensor<2x3x?x35xf16>",
                              '%x: tensor<2x3x?x35xf16> {tessera.layout = "strided"}')
    else:
        graph = graph.replace("tessera.rmsnorm", "tessera.layer_norm")
    result = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert result.returncode != 0


@pytest.mark.parametrize("fault", ["capacity", "prefix", "columns", "missing_policy", "static_policy"])
def test_bounded_norm_schedule_cannot_diverge_from_graph(compiler_toolchain, fault):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    schedule = invoke(tool, source(), "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    line = next(line for line in schedule.stdout.splitlines() if "= schedule.norm " in line)
    altered = line
    if fault == "capacity":
        altered = altered.replace("row_capacity = 42", "row_capacity = 48")
    elif fault == "prefix":
        altered = altered.replace("row_multiple = 6", "row_multiple = 7")
    elif fault == "columns":
        altered = altered.replace("fixed_columns = 35", "fixed_columns = 34")
    elif fault == "missing_policy":
        altered = re.sub(r'shape_policy = "bounded_query_v1",? ?', "", altered)
    else:
        altered = altered.replace("bounded_query_v1", "static")
    assert altered != line
    result = invoke(tool, schedule.stdout.replace(line, altered), "--tessera-schedule-to-tile")
    assert result.returncode != 0


@pytest.mark.parametrize("pipeline", [
    "--pass-pipeline=builtin.module(lower-tile-to-rocm{arch=gfx1201})",
    "--lower-tile-to-nvidia=sm=90",
    "--tessera-tile-to-x86"])
def test_bounded_norm_cannot_be_consumed_by_another_target(compiler_toolchain, pipeline):
    tool = compiler_toolchain.require_tessera_opt(pipeline)
    scheduled = invoke(tool, source(), "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    tile = invoke(tool, scheduled.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    refused = invoke(tool, tile.stdout, pipeline)
    assert refused.returncode != 0
