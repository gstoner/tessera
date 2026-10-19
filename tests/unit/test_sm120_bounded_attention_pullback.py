"""Bounded row member from the actual native source-attention paired AD."""
import hashlib
import math
import re

import pytest

from tests.unit.test_native_source_attention_reverse_outline import outline, source
from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.compiler_route


def actual_bounded_pullback(tool, storage, capacity):
    _, plan = outline(tool, source(storage))
    graph = plan["member_graphs"][-1]
    graph = re.sub(r'tessera.native.sm120_row_producer = \{[^}]+\},? ?', '', graph)
    graph = re.sub(r'func.func @[^\s(]+', 'func.func private @__native_row_body', graph)
    graph = graph.replace("tensor<1x2x3x4x",
                          f"tensor<{capacity[0]}x{capacity[1]}x?x{capacity[3]}x")
    graph = graph.replace("tensor<1x2x3xf32>",
                          f"tensor<{capacity[0]}x{capacity[1]}x?xf32>")
    printed = invoke(tool, graph)
    assert printed.returncode == 0, printed.stderr
    lines = printed.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if "func.func private" in line)
    function = "\n".join(line[2:] for line in lines[start:-1]).rstrip()
    digest = hashlib.sha256(function.encode()).hexdigest()
    contract = (f'tessera.native.sm120_row_producer = {{accum = "f32", body_hash = "{digest}", '
                f'columns = {capacity[-1]} : i64, rows = {math.prod(capacity[:-1])} : i64, '
                f'schema = 8 : i64, sequence_bound = {capacity[2]} : i64, storage = "{storage}"}}, ')
    assert " attributes {" in function
    function = function.replace(" attributes {", " attributes {" + contract, 1)
    return ('module attributes {tessera.arch = "sm_120", tessera.target = "nvidia_sm120"} {\n'
            + function + "\n}")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("capacity", [(2, 3, 7, 35), (1, 2, 17, 257)])
def test_actual_source_attention_pullback_has_bounded_native_route(
        compiler_toolchain, storage, capacity):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    graph = actual_bounded_pullback(tool, storage, capacity)
    assert "tessera.normalization_stats" in graph
    assert "tessera.relu" not in graph and "tessera.masked_fill" not in graph
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    target = invoke(tool, tile.stdout, "--lower-tile-to-nvidia=sm=120")
    assert target.returncode == 0, target.stderr
    assert target.stdout.count("scf.if") == 2
    native = invoke(tool, target.stdout, "--lower-tessera-nvidia-to-nvvm")
    assert native.returncode == 0, native.stderr
