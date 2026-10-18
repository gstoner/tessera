"""Last-axis native reverse rows retain their logical tensor prefix.

Test-only rank projection of an actual paired-AD fixture. Production code
must clone native AD SSA; these tests do not claim composed execution.
"""
import hashlib
import math
from pathlib import Path

import pytest

from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.compiler_route


def ranked_source(tool, storage, shape, *, broadcast_axes=None):
    fixture = Path(__file__).resolve().parents[1] / "fixtures/native_sm120/reverse_row_ssa.mlir"
    text = fixture.read_text().replace("xf16", "x" + storage)
    prefix = shape[:-1]
    text = text.replace("17x35", "x".join(map(str, shape)))
    text = text.replace("tensor<17xf32>", "tensor<" + "x".join(map(str, prefix)) + "xf32>")
    text = text.replace("axis = 1 :", f"axis = {len(shape) - 1} :")
    text = text.replace("broadcast_dimensions = [0]",
                        "broadcast_dimensions = [" + ", ".join(map(str, range(len(prefix)) if broadcast_axes is None else broadcast_axes)) + "]")
    printed = invoke(tool, text)
    assert printed.returncode == 0, printed.stderr
    lines = printed.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if "func.func private" in line)
    function = "\n".join(line[2:] for line in lines[start:-1]).rstrip()
    digest = hashlib.sha256(function.encode()).hexdigest()
    contract = (f' attributes {{tessera.native.sm120_row_producer = {{accum = "f32", '
                f'body_hash = "{digest}", columns = {shape[-1]} : i64, '
                f'rows = {math.prod(prefix)} : i64, schema = 6 : i64, storage = "{storage}"}}}}')
    function = function.replace(" {\n", contract + " {\n", 1)
    return ('module attributes {tessera.arch = "sm_120", tessera.target = "nvidia_sm120"} {\n'
            + function + "\n}")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("shape", [(2, 3, 35), (2, 3, 5, 35), (2, 1, 3, 257), (2, 4, 17, 257)])
def test_ranked_reverse_rows_preserve_prefix_and_flatten_only_physical_views(
        compiler_toolchain, storage, shape):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    graph = ranked_source(tool, storage, shape)
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    assert "tensor<" + "x".join(map(str, shape)) + "x" + storage + ">" in schedule.stdout
    assert f"rows={math.prod(shape[:-1])};columns={shape[-1]}" in schedule.stdout
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    assert f"rows = {math.prod(shape[:-1])}" in tile.stdout
    assert f"columns = {shape[-1]}" in tile.stdout
    assert tile.stdout.count("= tile.buffer_view ") == 3
    assert tile.stdout.count("tile.store_element") == 1
    target = invoke(tool, tile.stdout, "--lower-tile-to-nvidia=sm=120")
    assert target.returncode == 0, target.stderr
    assert "tessera_nvidia.row_region_kernel" in target.stdout
    native = invoke(tool, target.stdout, "--lower-tessera-nvidia-to-nvvm")
    assert native.returncode == 0, native.stderr
    assert "tile.buffer_view" not in native.stdout and "tessera.rmsnorm" not in native.stdout
    assert "tessera_nvidia.row_region_kernel" not in native.stdout


@pytest.mark.parametrize("fault", ["row_count", "overflow", "rocm", "nonleading_broadcast"])
def test_ranked_reverse_rows_reject_wrong_geometry_and_unproved_sibling(
        compiler_toolchain, fault):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    shape = ((50000, 50000, 35) if fault == "overflow" else
             (2, 3, 5, 5) if fault == "nonleading_broadcast" else (2, 3, 5, 35))
    graph = ranked_source(tool, "f16", shape,
                          broadcast_axes=(0, 1, 3) if fault == "nonleading_broadcast" else None)
    if fault == "row_count":
        graph = graph.replace("rows = 30 : i64", "rows = 2 : i64")
    elif fault == "rocm":
        graph = graph.replace('tessera.target = "nvidia_sm120"', 'tessera.target = "rocm_gfx1201"')
        graph = graph.replace('tessera.arch = "sm_120"', 'tessera.arch = "gfx1201"')
        graph = graph.replace("sm120_row_producer", "rocm_row_producer").replace("schema = 6", "schema = 7")
    result = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert result.returncode != 0
