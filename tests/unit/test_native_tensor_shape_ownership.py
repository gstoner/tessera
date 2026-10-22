"""Independent dimensions must remain native SSA through ownership and replay."""
import base64
import json
import re

import pytest
from tests.unit.test_native_tensor_program_pipeline import _run

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def shape_source(case, storage="f16", *, explicit=False, m=17, k=32, n=11):
    def axis(value):
        return "?" if explicit else str(value)
    a = f"tensor<{axis(m)}x{axis(k)}x{storage}>"
    b = f"tensor<{axis(k)}x{axis(n)}x{storage}>"
    out = f"tensor<{axis(m)}x{axis(n)}xf32>"
    lines = []

    def emit(name, op, operands, types, result, attrs=""):
        fields = 'tessera.effect_kind = "pure"' + (", " + attrs if attrs else "")
        lines.append(f'  %{name} = "tessera.{op}"({operands}) {{{fields}}} : ({types}) -> {result}')

    def matmul(name, left, right, lt, rt, result, bounds):
        attrs = 'output_dtype = "fp32", rhs_storage_order = "row_major"'
        if explicit:
            attrs += ", shape_bounds = [" + ", ".join(f"{v} : i64" for v in bounds) + "]"
        emit(name, "matmul", f"%{left},%{right}", f"{lt},{rt}", result, attrs)

    if case == "two_matmuls":
        a = f"tensor<{axis(m)}x32x{storage}>"
        w = f"tensor<32x{axis(k)}x{storage}>"
        h = f"tensor<{axis(m)}x{axis(k)}xf32>"
        args = f"%a: {a}, %w: {w}, %b: {b}"
        matmul("h", "a", "w", a, w, h, [65, 64, 32])
        emit("x", "relu", "%h", h, h)
        matmul("out", "x", "b", h, b, out, [65, 47, 64])
    else:
        args = f"%a: {a}, %b: {b}"
        rhs = "b"
        if case == "rhs_add":
            args += f", %r: {b}"
            emit("rhs", "add", "%b,%r", f"{b},{b}", b)
            rhs = "rhs"
        if case == "chain":
            args += f", %r: {a}"
            emit("s", "add", "%a,%r", f"{a},{a}", a)
            emit("x", "relu", "%s", a, a)
        elif case == "norm_lhs":
            emit("x", "rmsnorm", "%a", a, a)
        else:
            emit("x", "relu", "%a", a, a)
        matmul("out", "x", rhs, a, b, out, [65, 47, 64])
    return ('module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120", '
            'tessera.native.tensor_program_shape_bounds_request = {M = 65 : i64, N = 47 : i64, K = 64 : i64}} {\n'
            f" func.func @generic({args}) -> {out} {{\n" + "\n".join(lines)
            + f"\n  return %out : {out}\n }}\n}}\n")


def check_owned_shapes(source, case, route):
    flags = (["--pass-pipeline=builtin.module(tessera-nvidia-pipeline-sm120,tessera-lower-to-nvidia-sm120)"]
             if route == "registered" else ["--tessera-graph-to-schedule", "--tessera-schedule-to-tile",
                                             "--tessera-lower-to-nvidia-sm120"])
    result = _run(source, "--verify-each", *flags)
    assert result.returncode == 0, result.stderr
    match = re.search(r'tessera.native.tensor_program_frame_json = "([^"]+)"', result.stdout)
    assert match, "independent dimensions lost the compiler-owned program frame"
    frame = json.loads(base64.b64decode(match[1], validate=True))
    assert frame["schema"] == "tessera.native.tensor_program_frame.v3"
    roots = [frame["buffers"][i]["shape"] for i in frame["roots"]]
    expected = ([[65, 32], [32, 64], [64, 47]] if case == "two_matmuls" else
                [[65, 64], [64, 47], [64, 47]] if case == "rhs_add" else
                [[65, 64], [64, 47], [65, 64]] if case == "chain" else
                [[65, 64], [64, 47]])
    assert roots == expected
    assert len(frame["stages"]) == (2 if case == "norm_lhs" else 3)
    assert re.search(r"^\s*%.*=\s*memref.reinterpret_cast\b", result.stdout, re.M), (
        "active widths must recompute compact strides in typed memory SSA")
    assert re.search(r"^\s*%.*=\s*(?:arith.muli|llvm.mul)\b", result.stdout, re.M)
    replay = _run(result.stdout, "--tessera-graph-to-schedule=tensor-program-replay")
    assert replay.returncode == 0, replay.stderr
    assert replay.stdout == result.stdout


@pytest.mark.parametrize("case", ["norm_lhs", "rhs_add", "chain", "two_matmuls"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("route", ["explicit", "registered"])
def test_independent_mnk_uses_native_shape_and_memory_ownership(case, storage, explicit, route):
    check_owned_shapes(shape_source(case, storage, explicit=explicit), case, route)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_equal_trace_extents_do_not_unify_independent_dimensions(storage):
    check_owned_shapes(shape_source("norm_lhs", storage, m=17, k=17, n=17), "norm_lhs", "explicit")


@pytest.mark.parametrize("case", ["norm_lhs", "rhs_add", "chain", "two_matmuls"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("explicit", [False, True])
def test_shape_controller_verifies_before_native_frame_export(case, storage, explicit):
    result = _run(shape_source(case, storage, explicit=explicit),
                  "--verify-each", "--tessera-graph-to-schedule")
    assert result.returncode == 0, result.stderr
    assert "memref.reinterpret_cast" in result.stdout
    assert "tessera.native.tensor_program_shape_bindings" in result.stdout


@pytest.mark.parametrize("mutation", ["positive_guard", "capacity_guard", "equality_guard",
                                     "volume_guard", "volume_product", "compact_stride",
                                     "symbol_origin", "symbol_capacity"])
def test_shape_controller_rejects_corrupted_memory_proof(mutation):
    scheduled = _run(shape_source("norm_lhs"), "--verify-each",
                     "--mlir-print-op-generic", "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    source = scheduled.stdout
    if mutation in {"positive_guard", "equality_guard", "volume_guard"}:
        message = {"positive_guard": "shape extent violates positive capacity",
                   "equality_guard": "semantic dimension equality violated",
                   "volume_guard": "active view exceeds allocation capacity"}[mutation]
        pattern = r'(?m)^.*"cf.assert".*' + message + r'.*\n'
        source, count = re.subn(pattern, "", source, count=1)
        assert count == 1
    elif mutation == "capacity_guard":
        source = source.replace("predicate = 7 : i64", "predicate = 6 : i64", 1)
    elif mutation == "volume_product":
        source, count = re.subn(r'("arith.muli"\()(%[\w]+), (%[\w]+)', r'\1\2, \2', source, count=1)
        assert count == 1
    elif mutation == "compact_stride":
        source, count = re.subn(
            r'("memref.reinterpret_cast"\()(%[\w]+), (%[\w]+), (%[\w]+), (%[\w]+)',
            r'\1\2, \3, \4, \3', source, count=1)
        assert count == 1
    elif mutation == "symbol_origin":
        source = source.replace("shape_variables = array<i64: 0, 0, 65", "shape_variables = array<i64: 0, 1, 65", 1)
    elif mutation == "symbol_capacity":
        source = source.replace("shape_variables = array<i64: 0, 0, 65", "shape_variables = array<i64: 0, 0, 64", 1)
    assert source != scheduled.stdout
    result = _run(source, "--verify-each")
    assert result.returncode != 0
    assert "shape frame requires checked capacity" in result.stderr


@pytest.mark.parametrize("edit,marker", [
    ("missing_retirement", "exact ownership retirement"),
    ("early_retirement", "retires before its final reader"),
    ("borrowed_retirement", "directly owned private allocations"),
])
def test_shape_controller_preserves_allocation_lifetimes(edit, marker):
    scheduled = _run(shape_source("norm_lhs"), "--verify-each",
                     "--mlir-print-op-generic", "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    source = scheduled.stdout
    release = next(line for line in source.splitlines(keepends=True) if '"memref.dealloc"' in line)
    source = source.replace(release, "", 1)
    if edit == "early_retirement":
        invocation = [line for line in source.splitlines(keepends=True) if '"tile.program_invoke"' in line][1]
        source = source.replace(invocation, release + invocation, 1)
    elif edit == "borrowed_retirement":
        argument = re.search(r"\^bb0\((%[\w]+): memref", source).group(1)
        source = source.replace('    "func.return"', f'    "memref.dealloc"({argument}) : (memref<?x?xf16>) -> ()\n    "func.return"', 1)
        # Keep the legitimate private retirement as well; the invalid borrowed
        # release must fail independently of any missing private retirement.
        source = source.replace('    "func.return"', release + '    "func.return"', 1)
    result = _run(source, "--verify-each")
    assert result.returncode > 0, result.stderr
    assert marker in result.stderr



@pytest.mark.parametrize("fault", ["swapped_root", "wrong_member_identity", "missing_member_axis"])
def test_equal_capacities_do_not_erase_member_dimension_identity(fault):
    source = shape_source("norm_lhs", m=17, k=17, n=17)
    source = source.replace("M = 65 : i64, N = 47 : i64, K = 64 : i64",
                            "M = 64 : i64, N = 64 : i64, K = 64 : i64")
    scheduled = _run(source, "--verify-each", "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    changed = scheduled.stdout
    if fault == "swapped_root":
        match = re.search(r"func.func @generic\((%[\w]+): memref<[^>]+>, (%[\w]+): memref", changed)
        assert match
        line = next(line for line in changed.splitlines() if "tile.program_invoke " in line)
        swapped = line.replace(match[1] + ",", match[2] + ",", 1)
        assert swapped != line
        changed = changed.replace(line, swapped, 1)
    else:
        binding = "1, 2, 1, 2" if fault == "wrong_member_identity" else "0, 1"
        changed, count = re.subn(
            r"(tessera.native.tensor_program_dimension_bindings = array<i64: )0, 1, 0, 1>",
            lambda match: match[1] + binding + ">", changed, count=1)
        assert count == 1
    result = _run(changed, "--verify-each")
    assert result.returncode > 0, result.stderr
    assert "shape member" in result.stderr and "binding" in result.stderr
