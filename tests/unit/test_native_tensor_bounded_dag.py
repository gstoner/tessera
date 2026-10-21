"""Bounded producer rows follow SSA edges, never coincident trace extents."""
import pytest
from tessera.compiler.native_tensor_program import _frame
from tests.unit.test_native_tensor_program_pipeline import _run

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]
CASES = ("chain", "static_rhs", "fork_join", "two_matmuls", "fp32_middle", "rmsnorm", "softmax")


def dag_source(case, storage="f16", *, form="policy", m=17, k=32, bound=65):
    row = "?" if form == "dynamic" else str(m)
    a, b, h, out = f"tensor<{row}x{k}x{storage}>", f"tensor<{k}x11x{storage}>", f"tensor<{row}x16xf16>", f"tensor<{row}x11xf32>"
    if case == "fp32_middle":
        h = f"tensor<{row}x16xf32>"
    lines = []

    def op(name, operation, operands, types, result, extra=""):
        fields = 'tessera.effect_kind = "pure"' + (", " + extra if extra else "")
        lines.append(f'  %{name} = "tessera.{operation}"({operands}) {{{fields}}} : ({types}) -> {result}')

    def matmul(name, left, right, left_type, right_type, result, n, contraction, output="fp32"):
        attrs = f'output_dtype = "{output}", rhs_storage_order = "row_major"'
        if form == "dynamic":
            attrs += f", shape_bounds = [{bound} : i64, {n} : i64, {contraction} : i64]"
        op(name, "matmul", f"%{left},%{right}", f"{left_type},{right_type}", result, attrs)

    if case in ("two_matmuls", "fp32_middle"):
        w, down = f"tensor<{k}x16x{storage}>", f"tensor<16x11x{'f16' if case == 'two_matmuls' else storage}>"
        args = f"%a: {a}, %w: {w}, %b: {down}"
        matmul("h", "a", "w", a, w, h, 16, k, "fp16" if case == "two_matmuls" else "fp32")
        op("x", "relu", "%h", h, h)
        matmul("out", "x", "b", h, down, out, 11, 16)
    else:
        args = f"%a: {a}, %b: {b}"
        rhs = "b"
        if case == "static_rhs":
            args += f", %r: {b}"
            op("rhs", "add", "%b,%r", f"{b},{b}", b)
            rhs = "rhs"
        if case == "chain":
            args += f", %r: {a}"
            op("s", "add", "%a,%r", f"{a},{a}", a)
            op("x", "relu", "%s", a, a)
        elif case == "fork_join":
            op("u", "relu", "%a", a, a)
            op("y", "add", "%u,%a", f"{a},{a}", a)
            op("x", "add", "%y,%u", f"{a},{a}", a)
        elif case in ("rmsnorm", "softmax"):
            op("x", case, "%a", a, a, 'axis = -1 : i64')
        else:
            op("x", "relu", "%a", a, a)
        matmul("out", "x", rhs, a, b, out, 11, k)
    request = f", tessera.native.tensor_program_row_bound_request = {bound} : i64" if form == "policy" else ""
    return (f'module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"{request}}} {{\n'
            f" func.func @dag({args}) -> {out} {{\n" + "\n".join(lines) + f"\n  return %out : {out}\n }}\n}}\n")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("form", ["policy", "dynamic"])
def test_bounded_dag_keeps_dynamic_memory_and_complete_source_replay(case, storage, form):
    source = dag_source(case, storage, form=form)
    result = _run(source, "--tessera-graph-to-schedule", "--tessera-schedule-to-tile", "--tessera-lower-to-nvidia-sm120")
    assert result.returncode == 0, result.stderr
    frame = _frame(result.stdout)
    assert frame["schema"] == "tessera.native.tensor_program_frame.v2"
    assert frame["row_bound"] == 65
    assert "memref<?x" in result.stdout
    assert "memref.subview" in result.stdout
    assert len(frame["stages"]) == (4 if case == "fork_join" else 2 if case in ("rmsnorm", "softmax") else 3)
    if case == "static_rhs":
        assert frame["row_buffers"][:3] == [1, 0, 0]
        assert not any(frame["stages"][0]["scalar_row_multipliers"])
        assert not any(frame["stages"][0]["grid_row_multipliers"])
    replay = _run(result.stdout, "--tessera-graph-to-schedule=tensor-program-replay")
    assert replay.returncode == 0, replay.stderr
    assert replay.stdout == result.stdout


@pytest.mark.parametrize("m,k", [(1, 1), (17, 17)])
@pytest.mark.parametrize("case", ["chain", "static_rhs", "fork_join"])
def test_trace_extent_coincidence_never_projects_rhs_rows(case, m, k):
    result = _run(dag_source(case, m=m, k=k), "--tessera-graph-to-schedule", "--tessera-schedule-to-tile", "--tessera-lower-to-nvidia-sm120")
    assert result.returncode == 0, result.stderr
    frame = _frame(result.stdout)
    assert frame["row_buffers"][0] == 1
    assert frame["row_buffers"][1] == 0
    assert frame["buffers"][1]["shape"][0] == k


def test_same_ssa_value_cannot_be_both_runtime_rows_and_fixed_rhs():
    source = '''module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120", tessera.native.tensor_program_row_bound_request = 65 : i64} {
 func.func @conflict(%a: tensor<17x17xf16>, %unused: tensor<17x17xf16>) -> tensor<17x17xf32> {
  %x = "tessera.relu"(%a) {tessera.effect_kind = "pure"} : (tensor<17x17xf16>) -> tensor<17x17xf16>
  %out = "tessera.matmul"(%x,%a) {tessera.effect_kind = "pure", output_dtype = "fp32", rhs_storage_order = "row_major"} : (tensor<17x17xf16>,tensor<17x17xf16>) -> tensor<17x17xf32>
  return %out : tensor<17x17xf32>
 }
}'''
    result = _run(source, "--tessera-graph-to-schedule")
    assert result.returncode != 0
    assert "producer DAG" in result.stderr
    assert "PLEASE submit a bug report" not in result.stderr


@pytest.mark.parametrize("case,family", [("chain", "activation"), ("rmsnorm", "norm"), ("softmax", "softmax")])
@pytest.mark.parametrize("value", [None, "0 : i64", "-1 : i64", "64 : i64", "true", '"65"', "65 : i32"])
def test_unary_schedule_row_capacity_is_typed_and_bound_to_original_source(case, family, value):
    scheduled = _run(dag_source(case), "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    lines = scheduled.stdout.splitlines()
    index = next(i for i, line in enumerate(lines) if "schedule." + family + " " in line)
    original = "row_bound = 65 : i64"
    assert original in lines[index]
    lines[index] = (lines[index].replace(original + ", ", "") if value is None else
                    lines[index].replace(original, "row_bound = " + value))
    assert lines[index] != scheduled.stdout.splitlines()[index]
    altered = _run("\n".join(lines) + "\n", "--tessera-schedule-to-tile")
    assert altered.returncode != 0
    assert "PLEASE submit a bug report" not in altered.stderr
