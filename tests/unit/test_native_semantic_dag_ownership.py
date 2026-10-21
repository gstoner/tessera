"""Semantic DAG ownership is independent of a terminal matrix consumer."""
import pytest

from tessera.compiler.native_tensor_program import _frame
from tests.unit.test_native_tensor_program_pipeline import _run

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def semantic_source(case, storage, form):
    row = "?" if form == "dynamic" else "1"
    tensor = f"tensor<{row}x32x{storage}>"
    request = "" if form == "static" else ", tessera.native.tensor_program_row_bound_request = 65 : i64"
    args = f"%a: {tensor}"
    if case == "fork_join":
        operations = [("x", "relu", "%a", tensor),
                      ("out", "add", "%x,%a", f"{tensor},{tensor}")]
    else:
        first, second = case.split("_then_")
        operations = [("x", first, "%a", tensor), ("out", second, "%x", tensor)]
    lines = []
    for name, operation, operands, types in operations:
        attrs = 'tessera.effect_kind = "pure"'
        if operation in {"rmsnorm", "layer_norm", "softmax"}:
            attrs += ", axis = -1 : i64"
        lines.append(f'  %{name} = "tessera.{operation}"({operands}) {{{attrs}}} : ({types}) -> {tensor}')
    return (f'module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"{request}}} {{\n'
            f" func.func @semantic({args}) -> {tensor} {{\n" + "\n".join(lines)
            + f"\n  return %out : {tensor}\n }}\n}}\n")


@pytest.mark.parametrize("case", ["rmsnorm_then_relu", "softmax_then_rmsnorm", "layer_norm_then_softmax", "fork_join"])
@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("form", ["static", "policy", "dynamic"])
def test_semantic_dag_has_owned_frame_and_strict_source_replay(case, storage, form):
    result = _run(semantic_source(case, storage, form), "--verify-each",
                  "--tessera-graph-to-schedule", "--tessera-schedule-to-tile",
                  "--tessera-lower-to-nvidia-sm120")
    assert result.returncode == 0, result.stderr
    frame = _frame(result.stdout)
    assert frame["roots"] == [0]
    assert len(frame["stages"]) == 2
    assert frame["stages"][0]["inputs"] == [0]
    assert frame["stages"][1]["inputs"] == ([1, 0] if case == "fork_join" else [1])
    assert frame["output"] == 2
    if form != "static":
        assert frame["row_bound"] == 65
        assert frame["row_buffers"] == [1, 1, 1]
        assert "memref.subview" in result.stdout
    replay = _run(result.stdout, "--tessera-graph-to-schedule=tensor-program-replay")
    assert replay.returncode == 0, replay.stderr
    assert replay.stdout == result.stdout


@pytest.mark.parametrize("capacity", ["0 : i64", "-1 : i64", "true", "65 : i32", "2147483648 : i64"])
def test_semantic_dag_rejects_invalid_capacity_before_ownership(capacity):
    source = semantic_source("rmsnorm_then_relu", "f16", "policy")
    source = source.replace("65 : i64", capacity)
    result = _run(source, "--verify-each", "--tessera-graph-to-schedule")
    assert result.returncode != 0
    assert "PLEASE submit a bug report" not in result.stderr


def test_semantic_dag_requires_capacity_for_dynamic_rows():
    source = semantic_source("rmsnorm_then_relu", "f16", "dynamic")
    source = source.replace(", tessera.native.tensor_program_row_bound_request = 65 : i64", "")
    result = _run(source, "--verify-each", "--tessera-graph-to-schedule")
    assert result.returncode != 0
    assert "PLEASE submit a bug report" not in result.stderr


@pytest.mark.parametrize("epilogue", ["relu", "add"])
def test_matrix_epilogues_keep_existing_fused_native_owner(epilogue):
    import re

    tensor = "tensor<16x16xf16>"
    extra = f", %bias: {tensor}" if epilogue == "add" else ""
    operands = "%product,%bias" if epilogue == "add" else "%product"
    types = f"{tensor},{tensor}" if epilogue == "add" else tensor
    source = f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
 func.func @fused(%a: {tensor}, %b: {tensor}{extra}) -> {tensor} {{
  %product = "tessera.matmul"(%a,%b) {{output_dtype = "fp16", rhs_storage_order = "row_major", tessera.effect_kind = "pure"}} : ({tensor},{tensor}) -> {tensor}
  %out = "tessera.{epilogue}"({operands}) {{tessera.effect_kind = "pure"}} : ({types}) -> {tensor}
  return %out : {tensor}
 }}
}}'''
    query = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert query.returncode == 0, query.stderr
    assert "tessera.native.tensor_program_admission = false" in query.stdout
    compiled = _run(source, "--tessera-nvidia-pipeline-sm120", "--tessera-lower-to-nvidia-sm120")
    assert compiled.returncode == 0, compiled.stderr
    kernels = [line for line in compiled.stdout.splitlines()
               if re.match(r"\s*llvm.func @", line) and "nvvm.kernel" in line]
    assert len(kernels) == 1
    assert "nvvm.mma.sync" in compiled.stdout
