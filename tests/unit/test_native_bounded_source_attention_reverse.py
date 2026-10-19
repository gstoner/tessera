"""Native bounded source-attention projection retains actual paired SSA."""
import hashlib

import pytest

from tessera.compiler.attention_shape_contract import DYNAMIC_DIM

from tests.unit.test_native_source_attention_reverse_outline import outline, source
from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.compiler_route


def bounded_source(storage="f16", variant="normal", *, rows=False, trace_one=False,
                   bounds=(7, 11), requested=(0, 1, 2)):
    text = source(storage, variant, requested)
    if trace_one:
        text = text.replace("tensor<1x2x3x", "tensor<1x2x1x")
        text = text.replace("tensor<1x2x3xf32>", "tensor<1x2x1xf32>")
        text = text.replace("tensor<1x1x5x", "tensor<1x1x1x")
    marker = f"tessera.attention_sequence_bounds = array<i64: {bounds[0]}, {bounds[1]}>, "
    if rows:
        marker += "tessera.native.source_attention_reverse_rows_request = unit, "
    return text.replace("module attributes {", "module attributes {" + marker, 1)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("variant", ["normal", "permuted", "swapped"])
@pytest.mark.parametrize("trace_one", [False, True])
def test_native_bounded_source_reverse_projects_all_members(
        compiler_toolchain, storage, rows, variant, trace_one):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    _, plan = outline(tool, bounded_source(storage, variant, rows=rows, trace_one=trace_one,
                                          requested=(2, 0)))
    assert plan["schema"] == f"tessera.native.source_attention_reverse.v{4 if rows else 3}"
    assert plan["kind"] == ("norm_attention_saved_reverse_rows_bounded" if rows else
                            "norm_attention_saved_reverse_bounded")
    assert plan["shape_policy"] == "bounded_sequences_v1"
    assert plan["sequence_bounds"] == [7, 11]
    assert plan["shape_bounds"] == [1, 2, 1, 7, 11, 4, 3]
    roles = [1, 2, 0] if variant == "permuted" else [0, 1, 2]
    assert plan["role_indices"] == roles
    assert plan["requested_gradients"] == [2, 0]
    assert "tensor<1x2x?x4x" not in plan["paired_graph_ir"]
    assert "tessera.normalization_stats" in plan["paired_graph_ir"]
    assert "tessera.attention_shape_bounds" in plan["program_graph_ir"]
    verified = invoke(tool, plan["program_graph_ir"], "--verify-each")
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout.count("call @") == (5 if rows else 4)
    key_ids = {roles[1], roles[2], 10 if rows else 9, 11 if rows else 10}
    for buffer in plan["buffers"]:
        assert buffer["shape"][2] == DYNAMIC_DIM
        assert buffer["capacity_shape"][2] == (11 if buffer["id"] in key_ids else 7)
        assert buffer["trace_shape"][2] == (1 if trace_one else 5 if buffer["id"] in key_ids else 3)
        size = 2 if buffer["storage"] in ("f16", "bf16") else 4
        for extent in buffer["capacity_shape"]:
            size *= extent
        assert size == buffer["bytes"]
    for name in ("source", "paired", "program"):
        assert hashlib.sha256(plan[name + "_graph_ir"].encode()).hexdigest() == plan[name + "_graph_sha256"]
    for index, graph in enumerate(plan["member_graphs"]):
        assert "tensor<1x2x?x" in graph
        if index == 0:
            assert "tessera.native.query_sequence_bound = 7 : i64" in graph
        if index == len(plan["member_graphs"]) - 1:
            assert "schema = 8 : i64" in graph and "sequence_bound = 7 : i64" in graph
            assert "tessera.normalization_stats" in graph
        if 0 < index < len(plan["member_graphs"]) - 1:
            assert "tessera.attention_shape_bounds" in graph
        for flag in ("--tessera-graph-to-schedule", "--tessera-schedule-to-tile",
                     "--lower-tile-to-nvidia=sm=120", "--lower-tessera-nvidia-to-nvvm"):
            lowered = invoke(tool, graph, flag)
            assert lowered.returncode == 0, lowered.stderr
            graph = lowered.stdout
        assert "llvm.func" in graph


@pytest.mark.parametrize("fault", ["short_q", "short_k", "zero", "negative", "overflow",
                                   "length", "type", "conflicting_bounds"])
def test_native_bounded_source_reverse_rejects_invalid_capacities(compiler_toolchain, fault):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    text = bounded_source()
    if fault in ("short_q", "short_k", "zero", "negative", "overflow"):
        bound = {"short_q": (2, 11), "short_k": (7, 4), "zero": (0, 11),
                 "negative": (-1, 11), "overflow": (2147483647, 11)}[fault]
        text = text.replace("array<i64: 7, 11>", f"array<i64: {bound[0]}, {bound[1]}>")
    elif fault == "length":
        text = text.replace("array<i64: 7, 11>", "array<i64: 7>")
    elif fault == "type":
        text = text.replace("array<i64: 7, 11>", "array<i32: 7, 11>")
    else:
        text = text.replace("module attributes {", "module attributes {tessera.attention_shape_bounds = array<i64: 1, 2, 1, 7, 11, 4, 3>, ", 1)
    result = invoke(tool, text, "--tessera-autodiff-paired")
    assert result.returncode != 0


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("rows", [False, True])
def test_bounded_source_transport_replays_native_projection_without_graph_synthesis(
        storage, rows, monkeypatch):
    from tessera.compiler import graph_ir
    from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
    text = bounded_source(storage, "permuted", rows=rows, trace_one=True)

    def forbidden(*args, **kwargs):
        pytest.fail("bounded source transport reconstructed Graph IR")

    monkeypatch.setattr(graph_ir.GraphIRFunction, "__init__", forbidden)
    monkeypatch.setattr(graph_ir.IROp, "__init__", forbidden)
    program = export_native_source_attention_reverse(text)
    plan = program.manifest
    assert plan["execution_state"] == "artifact_only"
    assert plan["sequence_bounds"] == [7, 11]
    members = program.lower_members()
    assert len(members) == (5 if rows else 4)
    assert "schema = 8 : i64" in members[-1][0]
    assert all("llvm.func" in member[-1] for member in members)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("trace_one", [False, True])
def test_bounded_auto_row_policy_uses_capacity_not_trace(compiler_toolchain, storage, trace_one):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    text = bounded_source(storage, trace_one=trace_one, bounds=(17, 19))
    text = text.replace(f"x4x{storage}>", f"x35x{storage}>")
    text = text.replace("head_dim = 4 : i64", "head_dim = 35 : i64")
    text = text.replace(f"x3x{storage}>", f"x7x{storage}>")
    text = text.replace("module attributes {",
                        "module attributes {tessera.native.source_attention_reverse_auto_request = unit, ", 1)
    _, plan = outline(tool, text)
    assert plan["schema"] == "tessera.native.source_attention_reverse.v4"
    assert plan["shape_bounds"] == [1, 2, 1, 17, 19, 35, 7]
    assert len(plan["member_graphs"]) == 5


@pytest.mark.parametrize("target,arch", [
    ("rocm_gfx1151", "gfx1151"), ("rocm_gfx1201", "gfx1201"),
    ("apple_gpu", "apple7"), ("x86_avx512", "x86_avx512")])
def test_bounded_source_projection_has_no_sibling_target_admission(compiler_toolchain, target, arch):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    text = bounded_source().replace('tessera.target = "nvidia_sm120"',
                                    f'tessera.target = "{target}"')
    text = text.replace('tessera.arch = "sm_120"', f'tessera.arch = "{arch}"')
    assert invoke(tool, text, "--tessera-autodiff-paired").returncode != 0
