"""Additive one-output row ABI; retain the original paired-JVP schema."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.compiler_route


def invoke(tool, text, *flags):
    return subprocess.run([str(tool), *flags, "-"], input=text, text=True,
                          capture_output=True, timeout=60)


def source(tool, storage, outputs):
    fixture = Path(__file__).resolve().parents[1] / "fixtures/native_sm120/active_source_jvp_f16.json"
    data = json.loads(fixture.read_text())
    plan = json.loads(data["metadata"]["native_program"]["plan_json"])
    text = plan["member_graphs"][0].replace("xf16", "xbf16") if storage == "bf16" else plan["member_graphs"][0]
    # Test-only projection of an actual compiler-produced row SSA fixture.
    text = re.sub(r' attributes \{tessera.native.sm120_row_producer = \{[^\n]+?\}\}', "", text)
    text = text.replace("func.func @native_sm120_tensor_producer", "func.func private @__native_row_body")
    if outputs == 1:
        tensor = f"tensor<17x35x{storage}>"
        text = text.replace(f"-> ({tensor}, {tensor})", f"-> {tensor}", 1)
        text = re.sub(r"return %[A-Za-z0-9_]+, (%[A-Za-z0-9_]+) : [^\n]+",
                      lambda m: f"return {m[1]} : {tensor}", text)
    printed = invoke(tool, text)
    assert printed.returncode == 0, printed.stderr
    lines = printed.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if "func.func private" in line)
    function = "\n".join(line[2:] for line in lines[start:-1]).rstrip()
    digest = hashlib.sha256(function.encode()).hexdigest()
    schema = 6 if outputs == 1 else 1
    contract = (f' attributes {{tessera.native.sm120_row_producer = {{accum = "f32", '
                f'body_hash = "{digest}", columns = 35 : i64, rows = 17 : i64, '
                f'schema = {schema} : i64, storage = "{storage}"}}}}')
    function = function.replace(" {\n", contract + " {\n", 1)
    return ('module attributes {tessera.arch = "sm_120", tessera.target = "nvidia_sm120"} {\n'
            + function + "\n}")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("outputs", [1, 2])
def test_row_arity_survives_schedule_tile_and_target(compiler_toolchain, storage, outputs):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    graph = source(tool, storage, outputs)
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    assert f"outputs={outputs}" in schedule.stdout
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    assert f"schema = {6 if outputs == 1 else 1}" in tile.stdout
    assert tile.stdout.count("= tile.buffer_view ") == 2 + outputs
    assert tile.stdout.count("tile.store_element") == outputs
    target = invoke(tool, tile.stdout, "--lower-tile-to-nvidia=sm=120")
    assert target.returncode == 0, target.stderr
    assert "tessera_nvidia.row_region_kernel" in target.stdout
    assert "tile.buffer_view" not in target.stdout
    # The body seal cannot be reinterpreted under the other ABI.
    altered = tile.stdout.replace(f"schema = {6 if outputs == 1 else 1}",
                                  f"schema = {1 if outputs == 1 else 6}", 1)
    refused = invoke(tool, altered, "--lower-tile-to-nvidia=sm=120")
    assert refused.returncode != 0


@pytest.mark.parametrize("outputs", [1, 2])
def test_row_schedule_rejects_cross_profile_graph_seal(compiler_toolchain, outputs):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    graph = source(tool, "f16", outputs)
    changed = graph.replace(f"schema = {6 if outputs == 1 else 1}",
                            f"schema = {1 if outputs == 1 else 6}")
    result = invoke(tool, changed, "--tessera-graph-to-schedule")
    assert result.returncode != 0


def reverse_source(tool, storage):
    """Seal a test fixture extracted from the native paired reverse Graph."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures/native_sm120/reverse_row_ssa.mlir"
    text = fixture.read_text().replace("xf16", "x" + storage)
    printed = invoke(tool, text)
    assert printed.returncode == 0, printed.stderr
    lines = printed.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if "func.func private" in line)
    function = "\n".join(line[2:] for line in lines[start:-1]).rstrip()
    seal = hashlib.sha256(function.encode()).hexdigest()
    contract = (f' attributes {{tessera.native.sm120_row_producer = {{accum = "f32", '
                f'body_hash = "{seal}", columns = 35 : i64, rows = 17 : i64, '
                f'schema = 6 : i64, storage = "{storage}"}}}}')
    function = function.replace(" {\n", contract + " {\n", 1)
    return ('module attributes {tessera.arch = "sm_120", tessera.target = "nvidia_sm120"} {\n'
            + function + "\n}")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_actual_reverse_row_ssa_lowers(compiler_toolchain, storage):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    graph = reverse_source(tool, storage)
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    assert tile.stdout.count("tile.store_element") == 1
    target = invoke(tool, tile.stdout, "--lower-tile-to-nvidia=sm=120")
    assert target.returncode == 0, target.stderr
