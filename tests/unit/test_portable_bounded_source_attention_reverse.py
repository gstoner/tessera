"""Offline bounded source-program contracts. PTX stubs here are never executed."""
from copy import deepcopy
import json
import subprocess

import pytest

from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.portable_source_attention_reverse import restore
from tests.unit.test_native_bounded_source_attention_reverse import bounded_source
from tests.unit.test_portable_source_attention_reverse import portable_program, reseal

pytestmark = pytest.mark.compiler_route


@pytest.fixture(scope="module", params=[("f16", False), ("bf16", False), ("f16", True), ("bf16", True)])
def bounded_portable(compiler_toolchain, request):
    storage, rows = request.param
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    return portable_program(export_native_source_attention_reverse(
        bounded_source(storage, rows=rows, trace_one=True), tool=tool))


def test_bounded_restore_needs_no_tools_or_provider(bounded_portable, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    def forbidden(*args, **kwargs):
        pytest.fail("offline bounded restore invoked tools/provider")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(scheduled_matmul, "find_tessera_opt", forbidden)
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    data, plan, members, dims = restore(json.loads(json.dumps(bounded_portable)))
    rows = plan["schema"].endswith(".v4")
    assert data["schema"].endswith(".v4" if rows else ".v3")
    assert dims == (1, 2, 1, 7, 11, 4, 3)
    assert len(members) == (5 if rows else 4)
    for member in data["members"]:
        abi = member["abi"]
        assert abi["shape_policy"] == "bounded_sequences_v1"
        assert abi["grid_policy"] == "actual_frame_v1"
        assert abi["shape_bounds"] == list(dims)
        assert "grid_capacity" in abi and "grid" not in abi


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("variant", ["normal", "permuted", "swapped"])
def test_bounded_restore_keeps_trace_and_logical_roles(compiler_toolchain, storage, rows, variant):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    data = portable_program(export_native_source_attention_reverse(
        bounded_source(storage, variant, rows=rows, trace_one=True, requested=(2, 0)), tool=tool))
    _, plan, _, dims = restore(data)
    assert dims[3:5] == (7, 11)
    assert plan["role_indices"] == ([1, 2, 0] if variant == "permuted" else [0, 1, 2])
    assert plan["cotangent_indices"] == ([4, 3] if variant == "swapped" else [3, 4])
    assert plan["requested_gradients"] == [2, 0]
    assert all(buffer["trace_shape"][2] == 1 for buffer in plan["buffers"])


@pytest.mark.parametrize("fault", [
    "outer_schema", "native_schema", "shape_policy", "sequence_bool", "sequence_source",
    "fixed_capacity", "logical_static", "trace_exceeds_capacity", "trace_bool", "capacity_bytes",
    "capacity_role", "program_bounds", "member_bounds", "norm_rows", "norm_prefix", "norm_columns",
    "norm_policy", "norm_missing", "norm_graph_bound", "pullback_schema", "pullback_sequence",
    "pullback_rows", "checkpoint_bounds", "checkpoint_contract_bounds", "checkpoint_policy", "checkpoint_static_shape",
    "step_capacity", "step_symbols", "step_grid", "step_policy", "lifetime", "source_policy",
    "private_storage", "root_call", "paired_digest",
])
def test_rehashed_bounded_conflicts_rejected_before_provider(bounded_portable, fault, monkeypatch):
    from tessera import runtime as rt
    data = deepcopy(bounded_portable)
    plan, records = data["plan"], data["members"]
    if fault == "outer_schema":
        data["schema"] = "tessera.nvidia.source_attention_reverse_program.v1"
    elif fault == "native_schema":
        plan["schema"] = "tessera.native.source_attention_reverse.v1"
    elif fault == "shape_policy":
        plan["shape_policy"] = "static"
    elif fault == "sequence_bool":
        plan["sequence_bounds"][0] = True
    elif fault == "sequence_source":
        plan["source_graph_ir"] = plan["source_graph_ir"].replace("array<i64: 7, 11>", "array<i64: 8, 11>", 1)
    elif fault == "fixed_capacity":
        plan["shape_bounds"][0] = 2
    elif fault == "logical_static":
        plan["buffers"][0]["shape"][2] = 7
    elif fault == "trace_exceeds_capacity":
        plan["buffers"][0]["trace_shape"][2] = 8
    elif fault == "trace_bool":
        plan["buffers"][0]["trace_shape"][2] = True
    elif fault == "capacity_bytes":
        plan["buffers"][0]["bytes"] -= 2
    elif fault == "capacity_role":
        plan["buffers"][plan["role_indices"][1]]["capacity_shape"][2] = 7
    elif fault == "program_bounds":
        plan["program_graph_ir"] = plan["program_graph_ir"].replace("array<i64: 1, 2, 1, 7, 11, 4, 3>",
                                                                  "array<i64: 1, 2, 1, 8, 11, 4, 3>", 1)
    elif fault == "member_bounds":
        records[1]["graph_ir"] = records[1]["graph_ir"].replace(
            "tessera.attention_shape_bounds = array<i64: 1, 2, 1, 7, 11, 4, 3>, ", "")
        plan["member_graphs"][1] = records[1]["graph_ir"]
    elif fault.startswith("norm_"):
        field, old, new = {
            "norm_rows": ("tile_ir", "row_capacity = 14 : i64", "row_capacity = 16 : i64"),
            "norm_prefix": ("tile_ir", "row_multiple = 2 : i64", "row_multiple = 1 : i64"),
            "norm_columns": ("tile_ir", "fixed_columns = 4 : i64", "fixed_columns = 5 : i64"),
            "norm_policy": ("tile_ir", 'shape_policy = "bounded_query_v1"', 'shape_policy = "static"'),
            "norm_missing": ("tile_ir", "sequence_bound = 7 : i64, ", ""),
            "norm_graph_bound": ("graph_ir", "tessera.native.query_sequence_bound = 7 : i64",
                                 "tessera.native.query_sequence_bound = 8 : i64"),
        }[fault]
        assert old in records[0][field]
        records[0][field] = records[0][field].replace(old, new, 1)
        if field == "graph_ir":
            plan["member_graphs"][0] = records[0][field]
            plan["program_graph_ir"] = plan["program_graph_ir"].replace(old, new, 1)
    elif fault.startswith("pullback_"):
        old, new = {
            "pullback_schema": ("schema = 8 : i64", "schema = 6 : i64"),
            "pullback_sequence": ("sequence_bound = 7 : i64", "sequence_bound = 8 : i64"),
            "pullback_rows": ("rows = 14 : i64", "rows = 16 : i64"),
        }[fault]
        field = "graph_ir" if fault == "pullback_sequence" else "tile_ir"
        assert old in records[-1][field]
        records[-1][field] = records[-1][field].replace(old, new, 1)
        if field == "graph_ir":
            plan["member_graphs"][-1] = records[-1][field]
            plan["program_graph_ir"] = plan["program_graph_ir"].replace(old, new)
    elif fault.startswith("checkpoint_"):
        old, new = {
            "checkpoint_bounds": ("array<i64: 1, 2, 1, 7, 11, 4, 3>", "array<i64: 1, 2, 1, 8, 11, 4, 3>"),
            "checkpoint_contract_bounds": (", shape_bounds = array<i64: 1, 2, 1, 7, 11, 4, 3>",
                                           ", shape_bounds = array<i64: 1, 2, 1, 8, 11, 4, 3>"),
            "checkpoint_policy": ('shape_policy = "bounded_sequences_v1"', 'shape_policy = "static"'),
            "checkpoint_static_shape": ("shape = array<i64: 1, 2, 1, -9223372036854775808, -9223372036854775808, 4, 3>",
                                        "shape = array<i64: 1, 2, 1, 7, 11, 4, 3>"),
        }[fault]
        assert old in records[1]["tile_ir"]
        records[1]["tile_ir"] = records[1]["tile_ir"].replace(old, new, 1)
    elif fault.startswith("step_"):
        abi = records[1]["abi"]
        if fault == "step_capacity":
            abi["shape_bounds"][3] += 1
        elif fault == "step_symbols":
            abi["dimensions"][3] = 7
        elif fault == "step_grid":
            abi["grid_capacity"][0] += 1
        else:
            abi["grid_policy"] = "static"
    elif fault == "lifetime":
        plan["buffers"][5]["last_read"] = 0
    elif fault == "source_policy":
        plan["source_graph_ir"] = plan["source_graph_ir"].replace('accum = "fp32"', 'accum = "fp16"', 1)
    elif fault == "private_storage":
        plan["buffers"][7]["storage"] = "f16"
    elif fault == "root_call":
        plan["program_graph_ir"] = plan["program_graph_ir"].replace("__member_0(%arg0)", "__member_0(%arg1)", 1)
    else:
        plan["paired_graph_ir"] += "\n"
        # Keep the old digest so corruption must be detected.
        def forbidden(*args, **kwargs):
            pytest.fail("bad paired digest reached provider")
        monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
        data["contract_digest"] = __import__("tessera.compiler.portable_source_attention_reverse", fromlist=["digest"]).digest(
            {k: v for k, v in data.items() if k != "contract_digest"})
        with pytest.raises(ValueError):
            restore(data)
        return
    def forbidden(*args, **kwargs):
        pytest.fail("conflicting bounded program invoked tools/provider")
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        restore(reseal(data))
