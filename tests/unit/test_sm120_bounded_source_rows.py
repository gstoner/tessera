"""Bounded logical-query rows for the native source-attention pullback."""
import hashlib
import math
import re

import pytest

from tests.unit.test_sm120_ranked_reverse_rows import ranked_source
from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.compiler_route


def bounded_source(tool, storage="f16", shape=(2, 3, 7, 35)):
    graph = ranked_source(tool, storage, shape)
    graph = re.sub(r" attributes \{tessera.native.sm120_row_producer = \{[^\n]+?\}\}", "", graph)
    graph = graph.replace("x".join(map(str, shape)), f"{shape[0]}x{shape[1]}x?x{shape[3]}")
    graph = graph.replace("x".join(map(str, shape[:-1])) + "xf32",
                          f"{shape[0]}x{shape[1]}x?xf32")
    canonical = invoke(tool, graph)
    assert canonical.returncode == 0, canonical.stderr
    lines = canonical.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if "func.func private" in line)
    function = "\n".join(line[2:] for line in lines[start:-1]).rstrip()
    digest = hashlib.sha256(function.encode()).hexdigest()
    contract = (f' attributes {{tessera.native.sm120_row_producer = {{accum = "f32", '
                f'body_hash = "{digest}", columns = {shape[-1]} : i64, '
                f'rows = {math.prod(shape[:-1])} : i64, schema = 8 : i64, '
                f'sequence_bound = {shape[2]} : i64, storage = "{storage}"}}}}')
    function = function.replace(" {\n", contract + " {\n", 1)
    return ('module attributes {tessera.arch = "sm_120", tessera.target = "nvidia_sm120"} {\n'
            + function + "\n}")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("shape", [(2, 3, 7, 35), (1, 2, 17, 257)])
def test_bounded_query_row_survives_native_stages(compiler_toolchain, storage, shape):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    graph = bounded_source(tool, storage, shape)
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    assert f"tensor<{shape[0]}x{shape[1]}x?x{shape[3]}x{storage}>" in schedule.stdout
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    assert "schema = 8" in tile.stdout
    assert f"row_multiple = {shape[0] * shape[1]}" in tile.stdout
    assert tile.stdout.count("= tile.buffer_view ") == 3
    assert "arith.remui" in tile.stdout
    assert "arith.cmpi ule" in tile.stdout
    target = invoke(tool, tile.stdout, "--lower-tile-to-nvidia=sm=120")
    assert target.returncode == 0, target.stderr
    # Exactly the kernel geometry guard and the per-thread active/capacity
    # guard remain. Enclosing conjunction + SCF induction prove each access.
    assert target.stdout.count("scf.if") == 2
    native = invoke(tool, target.stdout, "--lower-tessera-nvidia-to-nvvm")
    assert native.returncode == 0, native.stderr
    assert "tessera_nvidia.row_region_kernel" not in native.stdout


@pytest.mark.parametrize("fault", ["zero", "negative", "overflow", "rows", "schema",
                                   "missing", "extra_dynamic", "rocm"])
def test_bounded_query_row_rejects_wrong_contract(compiler_toolchain, fault):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    graph = bounded_source(tool)
    if fault in ("zero", "negative", "overflow"):
        bound = {"zero": 0, "negative": -1, "overflow": 2147483647}[fault]
        graph = graph.replace("sequence_bound = 7", f"sequence_bound = {bound}")
        graph = graph.replace("rows = 42", f"rows = {6 * bound}")
    elif fault == "rows":
        graph = graph.replace("rows = 42", "rows = 7")
    elif fault == "schema":
        graph = graph.replace("schema = 8", "schema = 6")
    elif fault == "missing":
        graph = graph.replace("sequence_bound = 7 : i64, ", "")
    elif fault == "extra_dynamic":
        graph = graph.replace("2x3x?", "?x3x?")
    else:
        graph = graph.replace("nvidia_sm120", "rocm_gfx1201").replace("sm_120", "gfx1201")
    result = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert result.returncode != 0


@pytest.mark.parametrize("fault", ["missing_prefix", "wrong_prefix", "static_schema"])
def test_bounded_row_target_rejects_invalid_physical_profile(compiler_toolchain, fault):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    schedule = invoke(tool, bounded_source(tool), "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    if fault == "missing_prefix":
        changed = tile.stdout.replace("row_multiple = 6 : i64, ", "")
    elif fault == "wrong_prefix":
        changed = tile.stdout.replace("row_multiple = 6", "row_multiple = 5")
    else:
        changed = tile.stdout.replace("schema = 8", "schema = 6")
    assert changed != tile.stdout
    result = invoke(tool, changed, "--lower-tile-to-nvidia=sm=120")
    assert result.returncode != 0
