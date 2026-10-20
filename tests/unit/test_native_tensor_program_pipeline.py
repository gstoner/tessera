"""General SM120 pipelines preserve actual tensor producers as native programs.

Compiler-route proof only; owning numerical execution belongs to device tests.
The positive contract requires a real multi-stage route, not a refusal.
"""
from dataclasses import replace
import re
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts

from tessera.compiler.scheduled_matmul import find_tessera_opt

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def relu_product(a, b):
    return ts.ops.matmul(ts.ops.relu(a), b, output_dtype="fp32")


def norm_product(a, b):
    return ts.ops.matmul(ts.ops.rmsnorm(a, eps=1e-5, numeric_policy={"accum": "fp32"}),
                         b, output_dtype="fp32")


def both_products(a, b):
    return ts.ops.matmul(ts.ops.relu(a), ts.ops.softmax(b, axis=-1), output_dtype="fp32")


def reversed_product(b, a):
    return ts.ops.matmul(ts.ops.relu(a), b, output_dtype="fp32")


def traced_source(function, storage):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    a, b = np.zeros((17, 32), dtype), np.zeros((32, 11), dtype)
    args = (b, a) if function is reversed_product else (a, b)
    jit = ts.jit(target="nvidia_sm120")(function)
    graph = jit._traced_autodiff_module(args, {})
    graph = replace(graph, module_attrs={**graph.module_attrs,
        "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
    return graph.to_mlir(target="nvidia_sm120", canonical=True)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("function,stages", [
    (relu_product, 2), (norm_product, 2), (both_products, 3), (reversed_product, 2)])
@pytest.mark.parametrize("route", ["standard", "explicit_schedule_tile"])
def test_general_pipeline_preserves_tensor_producer_program(function, stages, storage, route):
    source = traced_source(function, storage)
    flags = (["--tessera-nvidia-pipeline-sm120"] if route == "standard" else
             ["--tessera-graph-to-schedule", "--tessera-schedule-to-tile"])
    compiled = subprocess.run([str(find_tessera_opt()), "-", *flags],
                              input=source, text=True, capture_output=True, check=False)
    assert compiled.returncode == 0, compiled.stderr
    result = compiled.stdout
    entries = [line for line in result.splitlines() if
               (re.match(r"\s*llvm.func @", line) and "nvvm.kernel" in line)
               or (re.match(r"\s*gpu.func @", line) and " kernel" in line)]
    assert len(entries) == stages, (
        "general pipeline lost producer arithmetic: expected one native entry per actual "
        f"source stage ({stages}), received {entries}")
    # A real typed host/program dataflow must connect the entries; merely
    # emitting disconnected kernels does not repair the dropped producer.
    invokes = re.findall(r"(?:tile.program_invoke|gpu.launch_func)\b", result)
    allocations = re.findall(r"(?:tile.program_alloc|gpu.alloc|memref.alloc)\b", result)
    assert len(invokes) == stages
    assert len(allocations) >= stages - 1
    assert "memref<17x32x" + ("f16" if storage == "f16" else "bf16") + ">" in result
    assert "tile.view " in result and "tile.fragment_pack " in result
    assert "tile.fragment_zero " in result and "tile.fragment_unpack " in result
    assert "tile.async_copy " not in result


def _run(source, *flags):
    return subprocess.run([str(find_tessera_opt()), "-", *flags],
                          input=source, text=True, capture_output=True, check=False)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_native_frame_preserves_reversed_source_roles(storage):
    import base64
    import json

    compiled = _run(traced_source(reversed_product, storage),
                    "--tessera-graph-to-schedule", "--tessera-schedule-to-tile")
    assert compiled.returncode == 0, compiled.stderr
    frame = json.loads(base64.b64decode(re.search(
        r'tessera.native.tensor_program_frame_json = "([^"]+)"', compiled.stdout)[1]))
    assert frame["roots"] == [0, 1]
    assert frame["stages"][0]["inputs"] == [1]
    assert frame["stages"][1]["inputs"] == [2, 0]
    assert frame["stages"][1]["output"] == frame["output"] == 3
    for index, stage in enumerate(frame["stages"]):
        projected = _run(compiled.stdout, f"--tessera-schedule-to-tile=tensor-member={index}")
        assert projected.returncode == 0, projected.stderr
        assert "tile.program_invoke " not in projected.stdout
        assert "memref.alloc" not in projected.stdout
        assert projected.stdout.count("nvvm.kernel") == 1
        assert f"llvm.func @{stage['entry']}(" in projected.stdout
    refused = _run(compiled.stdout, "--tessera-schedule-to-tile=tensor-member=2")
    assert refused.returncode != 0
    assert "member index is outside the program" in refused.stderr


@pytest.mark.parametrize("edit,marker", [
    ("uninitialized", "input must retain initialized live storage"),
    ("early_retire", "retires before its final reader"),
    ("returned_retire", "returned allocation must remain live"),
    ("missing_retire", "exact ownership retirement"),
    ("geometry", "selected kernel ABI and geometry"),
    ("storage", "physical buffer shape or storage differs"),
    ("symbol", "callee is not a visible native program member"),
])
def test_invalid_memory_and_launch_edges(edit, marker):
    compiled = _run(traced_source(relu_product, "f16"),
                    "--tessera-graph-to-schedule", "--tessera-schedule-to-tile")
    assert compiled.returncode == 0, compiled.stderr
    text = compiled.stdout
    if edit == "uninitialized":
        text = text.replace("tile.program_invoke %arg0, %alloc ",
                            "tile.program_invoke %alloc, %alloc ", 1)
    elif edit == "early_retire":
        line = "    memref.dealloc %alloc : memref<17x32xf16>\n"
        assert line in text
        text = text.replace(line, "", 1).replace(
            "    %alloc_0 = memref.alloc()", line + "    %alloc_0 = memref.alloc()", 1)
    elif edit == "returned_retire":
        text = text.replace("    return %alloc_0 :", (
            "    memref.dealloc %alloc_0 : memref<17x11xf32>\n"
            "    return %alloc_0 :"), 1)
    elif edit == "missing_retire":
        text = text.replace("    memref.dealloc %alloc : memref<17x32xf16>\n", "", 1)
    elif edit == "geometry":
        line = next(line for line in text.splitlines() if "tile.program_invoke " in line)
        changed = line.replace("geometry = array<i64: 1,", "geometry = array<i64: 2,", 1)
        assert changed != line
        text = text.replace(line, changed, 1)
    elif edit == "storage":
        host, members = text.split("  module @tensor_stage_0", 1)
        text = host.replace("memref<17x32xf16>", "memref<17x32xbf16>") + (
            "  module @tensor_stage_0" + members)
    else:
        text = text.replace("callee = @tensor_stage_0::@", "callee = @missing_stage::@", 1)
    rejected = _run(text)
    assert rejected.returncode != 0, edit
    assert marker in rejected.stderr


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_graph_schedule_pass_loads_program_carrier_without_later_passes(storage):
    compiled = _run(traced_source(relu_product, storage), "--tessera-graph-to-schedule")
    assert compiled.returncode == 0, compiled.stderr
    assert compiled.stdout.count("tile.program_invoke ") == 2
    assert "schedule.matmul " in compiled.stdout
    assert "schedule.activation " in compiled.stdout
    assert "nvvm.kernel" not in compiled.stdout

@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("function,stages", [
    (relu_product, 2), (norm_product, 2), (both_products, 3), (reversed_product, 2)])
def test_complete_target_controller_and_member_projection(function, stages, storage):
    import base64
    import json

    tile = _run(traced_source(function, storage),
                "--tessera-graph-to-schedule", "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    target = _run(tile.stdout, "--tessera-lower-to-nvidia-sm120")
    assert target.returncode == 0, target.stderr
    assert "tile." not in target.stdout
    assert target.stdout.count("tessera_nvidia.program_invoke ") == stages
    assert "nvvm.mma" in target.stdout
    def frame(text):
        return json.loads(base64.b64decode(re.search(
            r'tessera.native.tensor_program_frame_json = "([^"]+)"', text)[1]))
    expected = frame(tile.stdout)
    for index in range(stages):
        projected = _run(target.stdout,
                         f"--tessera-schedule-to-tile=tensor-member={index}")
        assert projected.returncode == 0, projected.stderr
        assert frame(projected.stdout) == expected
        assert "program_invoke " not in projected.stdout
        assert projected.stdout.count("nvvm.kernel") == 1
        assert f"llvm.func @{expected['stages'][index]['entry']}(" in projected.stdout


@pytest.mark.parametrize("edit", ["architecture", "target", "geometry", "early_retire"])
def test_target_controller_retains_checked_ownership(edit):
    tile = _run(traced_source(relu_product, "f16"),
                "--tessera-graph-to-schedule", "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    target = _run(tile.stdout, "--tessera-lower-to-nvidia-sm120")
    assert target.returncode == 0, target.stderr
    text = target.stdout
    if edit == "architecture":
        text = text.replace('tessera.arch = "sm_120"', 'tessera.arch = "sm_100"')
    elif edit == "target":
        text = text.replace('tessera.target = "nvidia_sm120"', 'tessera.target = "rocm_gfx1201"')
    elif edit == "geometry":
        line = next(line for line in text.splitlines() if "tessera_nvidia.program_invoke " in line)
        changed = line.replace("geometry = array<i64: 1,", "geometry = array<i64: 2,", 1)
        assert changed != line
        text = text.replace(line, changed, 1)
    else:
        line = "    memref.dealloc %alloc : memref<17x32xf16>\n"
        assert line in text
        text = text.replace(line, "", 1).replace(
            "    %alloc_0 = memref.alloc()", line + "    %alloc_0 = memref.alloc()", 1)
    assert text != target.stdout
    rejected = _run(text)
    assert rejected.returncode != 0
    assert "program_invoke" in rejected.stderr

@pytest.mark.parametrize("attribute", [
    'tessera.autodiff = "reverse"', 'tessera.autodiff.mode = "jvp"'])
@pytest.mark.parametrize("owner", ["module", "function"])
def test_pending_differentiation_is_not_projected_as_primal(attribute, owner):
    source = traced_source(norm_product, "f16")
    if owner == "module":
        source = source.replace("module attributes {", "module attributes {" + attribute + ", ", 1)
    else:
        source = source.replace(") attributes {", ") attributes {" + attribute + ", ", 1)
    assert attribute in source
    rejected = _run(source, "--tessera-graph-to-schedule")
    assert rejected.returncode != 0
    assert "pending differentiation must be transformed" in rejected.stderr


def test_target_invocation_cannot_bind_semantic_graph_member():
    scheduled = _run(traced_source(relu_product, "f16"), "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    text = scheduled.stdout.replace("tile.program_invoke ", "tessera_nvidia.program_invoke ")
    text = text.replace("{callee = @tensor_stage_", '{arch = "sm_120", geometry = array<i64: 1, 1, 1, 32, 1, 1>, scalars = array<i64>, callee = @tensor_stage_')
    rejected = _run(text)
    assert rejected.returncode != 0
    assert "Target invocation requires a lowered native kernel" in rejected.stderr
