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

@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("function", [relu_product, norm_product, both_products, reversed_product])
def test_native_admission_query_preserves_semantic_source(function, storage):
    source = traced_source(function, storage)
    queried = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert queried.returncode == 0, queried.stderr
    assert "tessera.native.tensor_program_admission = true" in queried.stdout
    assert "tile.program_invoke" not in queried.stdout
    assert "tensor_stage_" not in queried.stdout
    assert "memref.alloc" not in queried.stdout
    for op in ("tessera.relu", "tessera.rmsnorm", "tessera.softmax", "tessera.matmul"):
        assert source.count(op) == queried.stdout.count(op)


def test_native_admission_query_declines_fused_without_mutating_graph():
    source = traced_source(relu_product, "f16")
    source = source.replace('output_dtype = "fp32"', 'activation = "relu", output_dtype = "fp32"')
    assert 'activation = "relu"' in source
    queried = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert queried.returncode == 0, queried.stderr
    assert "tessera.native.tensor_program_admission = false" in queried.stdout
    assert 'activation = "relu"' in queried.stdout
    assert "tile.program_invoke" not in queried.stdout
    # Real compilation remains strict. A query decline cannot rewrite a fused
    # operation into an unfused program.
    rejected = _run(source, "--tessera-graph-to-schedule")
    assert rejected.returncode != 0


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("function", [relu_product, norm_product, both_products, reversed_product])
def test_native_target_replays_its_original_source(function, storage):
    source = _run(traced_source(function, storage))
    assert source.returncode == 0, source.stderr
    target = _run(source.stdout, "--tessera-graph-to-schedule", "--tessera-schedule-to-tile",
                  "--tessera-lower-to-nvidia-sm120")
    assert target.returncode == 0, target.stderr
    import hashlib
    seal = hashlib.sha256(source.stdout.strip().encode()).hexdigest()
    assert f'tessera.native.tensor_program_source_sha256 = "{seal}"' in target.stdout
    checked = _run(target.stdout, "--tessera-graph-to-schedule=tensor-program-replay")
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout == target.stdout


@pytest.mark.parametrize("edit", ["source_hash", "policy", "body", "entry"])
def test_native_replay_rejects_valid_typed_but_changed_program(edit):
    target = _run(traced_source(norm_product, "f16"), "--tessera-graph-to-schedule",
                  "--tessera-schedule-to-tile", "--tessera-lower-to-nvidia-sm120")
    assert target.returncode == 0, target.stderr
    text = target.stdout
    if edit == "source_hash":
        text = re.sub(r'(tessera.native.tensor_program_source_sha256 = ")[^"]+',
                      r'\g<1>' + "0" * 64, text)
    elif edit == "policy":
        # Only alter the controller's semantic policy, preserving valid native
        # pointer/scalar bindings and geometry. Ordinary type checking passes.
        lines = text.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if "tessera_nvidia.program_invoke " in line and 'accum = "fp32"' in line:
                lines[i] = line.replace('accum = "fp32"', 'accum = "fp16"')
                break
        else:
            raise AssertionError("expected explicit normalization policy")
        text = "".join(lines)
    elif edit == "body":
        assert "arith.mulf " in text
        text = text.replace("arith.mulf ", "arith.addf ", 1)
    else:
        entry = re.search(r"llvm.func @([A-Za-z0-9_]+)", text)[1]
        text = text.replace(entry, entry + "_different")
    assert text != target.stdout
    typed = _run(text)
    assert typed.returncode == 0, typed.stderr
    checked = _run(text, "--tessera-graph-to-schedule=tensor-program-replay")
    assert checked.returncode != 0
    assert "differs from original Graph replay" in checked.stderr


@pytest.mark.parametrize("flags", [(), ("--tessera-schedule-to-tile=tensor-member=0",)])
def test_flat_kernel_reference_is_rejected_before_member_projection(flags):
    # Previously this passed verification then ArrayRef::back asserted while
    # exporting the nonexistent nested member symbol.
    source = 'module attributes {tessera.arch = "sm_120", tessera.target = "nvidia_sm120",\ntessera.native.tensor_program = 1 : i64, tessera.native.tensor_program_target = "nvidia_sm120"} {\nfunc.func @host(%a: memref<1x1xf16>) -> memref<1x1xf32>\nattributes {tessera.native.tensor_program = 1 : i64} {\n %o = memref.alloc() : memref<1x1xf32>\n tessera_nvidia.program_invoke %a, %o {arch = "sm_120", callee = @kernel,\n input_count = 1 : i64, geometry = array<i64: 1, 1, 1, 32, 1, 1>, scalars = array<i64>} : memref<1x1xf16>, memref<1x1xf32>\n return %o : memref<1x1xf32>\n}\nllvm.func @kernel(%a: !llvm.ptr, %o: !llvm.ptr)\nattributes {nvvm.kernel, tessera.native.program_buffer_types = [memref<1x1xf16>, memref<1x1xf32>],\ntessera.native.program_geometry = array<i64: 1, 1, 1, 32, 1, 1>, tessera.native.program_scalars = array<i64>} {\n llvm.return\n}\n}\n'
    result = _run(source, *flags)
    assert result.returncode > 0, result.stderr  # diagnostic, never SIGABRT
    assert "requires one isolated member::kernel symbol reference" in result.stderr


@pytest.mark.parametrize("edit,marker", [
    ("member", "owned native program member module"),
    ("schema", "requires native program module schema 1")])
def test_native_controller_requires_owned_members_and_known_module_schema(edit, marker):
    compiled = _run(traced_source(relu_product, "f16"), "--tessera-graph-to-schedule",
                    "--tessera-schedule-to-tile", "--tessera-lower-to-nvidia-sm120")
    assert compiled.returncode == 0, compiled.stderr
    if edit == "member":
        changed = compiled.stdout.replace("tessera.native.tensor_program_member, ", "")
    else:
        changed = compiled.stdout.replace("tessera.native.tensor_program = 1 : i64",
                                          "tessera.native.tensor_program = 2 : i64", 1)
    assert changed != compiled.stdout
    result = _run(changed)
    assert result.returncode > 0, result.stderr
    assert marker in result.stderr


@pytest.mark.parametrize("replacement,marker", [
    ("", "checked input/output arity"),
    ("input_count = 0 : i64, ", "checked input/output arity"),
    ("input_count = 3 : i64, ", "checked input/output arity"),
    ("input_count = 2 : i128, ", "64-bit signless integer attribute")])
def test_later_invocation_arity_is_checked_before_cross_operation_lifetimes(replacement, marker):
    compiled = _run(traced_source(relu_product, "f16"), "--tessera-graph-to-schedule",
                    "--tessera-schedule-to-tile", "--tessera-lower-to-nvidia-sm120")
    assert compiled.returncode == 0, compiled.stderr
    lines = compiled.stdout.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if "tessera_nvidia.program_invoke " in line and "input_count = 2 : i64, " in line:
            lines[index] = line.replace("input_count = 2 : i64, ", replacement)
            break
    else:
        raise AssertionError("expected matmul consumer invocation")
    result = _run("".join(lines))
    assert result.returncode > 0, result.stderr  # malformed IR must never SIGSEGV
    assert marker in result.stderr


@pytest.mark.parametrize("type_", [
    "i64", "memref<*xf16>", "memref<?x1xf16>", "memref<0x1xf16>",
    "memref<2147483648x1xf16>", "memref<2147483647x2147483647xf32>",
    "memref<1x1xcomplex<f32>>", "memref<1x1xi8>"])
def test_unused_root_cannot_escape_frame_physical_type_checks(type_):
    compiled = _run(traced_source(relu_product, "f16"), "--tessera-graph-to-schedule",
                    "--tessera-schedule-to-tile", "--tessera-lower-to-nvidia-sm120")
    assert compiled.returncode == 0, compiled.stderr
    changed = compiled.stdout.replace(") -> memref", ", %unused: " + type_ + ") -> memref", 1)
    assert changed != compiled.stdout
    result = _run(changed, "--tessera-schedule-to-tile")
    assert result.returncode > 0, result.stderr
    assert "program roots require checked static physical buffer types" in result.stderr


@pytest.mark.parametrize("external", [False, True])
def test_frame_export_requires_a_defined_host_with_native_invocations(external):
    head = ('module attributes {tessera.arch = "sm_120", tessera.target = "nvidia_sm120", '
            'tessera.native.tensor_program = 1 : i64, '
            'tessera.native.tensor_program_target = "nvidia_sm120"} {')
    if external:
        function = ('func.func private @host() -> memref<1x1xf32> '
                    'attributes {tessera.native.tensor_program = 1 : i64}')
        marker = "needs one defined single-block host program"
    else:
        function = ('func.func @host(%a: memref<1x1xf16>) -> memref<1x1xf16> '
                    'attributes {tessera.native.tensor_program = 1 : i64} '
                    '{ return %a : memref<1x1xf16> }')
        marker = "needs at least one verified native invocation"
    result = _run(head + function + "}", "--tessera-schedule-to-tile")
    assert result.returncode > 0, result.stderr
    assert marker in result.stderr
