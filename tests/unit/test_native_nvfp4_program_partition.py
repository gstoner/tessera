"""Whole NVFP4 Graph partition retains compiler-owned SSA and buffer lifetimes."""
import base64
import json
import re
import pytest
from tessera.compiler.rocm_nvfp4_program import _full_graph
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def export(roles=(0,1,2,3,4), stage=None):
    graph = _full_graph(16,16,64,(0,16),roles).to_mlir(
        target="rocm_gfx1201", canonical=True)
    flags = "export-scaled-primal=true"
    if stage is not None:
        flags += f" select-scaled-member={stage}"
    return run_tessera_opt(find_tessera_opt(), graph,
                          f"--tessera-autodiff-forward={flags}")


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("roles", [(0,1,2,3,4), (2,0,4,1,3)])
def test_native_nvfp4_plan_preserves_original_roles_and_lifetimes(roles):
    text = export(roles)
    value = re.search(r'tessera.native.nvfp4_program_json = "([^"]+)"', text)
    assert value is not None
    plan = json.loads(base64.b64decode(value.group(1)))
    assert plan["schema"] == "tessera.native.nvfp4_program.v1"
    assert plan["role_indices"] == list(roles)
    assert len(plan["buffers"]) == 11
    assert [step["inputs"] for step in plan["steps"]] == [
        list(roles[:3]), [5,6], [roles[3],8,roles[4],9]]
    assert [step["outputs"] for step in plan["steps"]] == [[5,6,7],[8,9],[10]]
    assert plan["output"] == 10
    assert plan["buffers"][10]["ownership"] == "returned_output"
    assert plan["buffers"][10]["last_read"] == 3
    assert plan["buffers"][5]["last_read"] == 1
    assert plan["buffers"][8]["last_read"] == 2
    assert all(name in plan["root_ir"] for name in (
        "tessera.nvfp4_requantize", "tessera.mxfp4_folded_storage", "tessera.scaled_matmul"))


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("stage,carrier", [
    (0,"tile.nvfp4_requantize_kernel"),
    (1,"tile.mxfp4_folded_storage_kernel"),
    (2,"tile.scaled_matmul_kernel")])
def test_native_projected_member_reaches_schedule_and_tile(stage,carrier):
    graph = export(stage=stage)
    tool = find_tessera_opt()
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    assert "tessera.native.nvfp4_program_json" not in graph
    assert "tessera.native.nvfp4_program_json" not in tile
    assert carrier in tile
    target = run_tessera_opt(tool, tile, "--lower-tile-to-rocm=arch=gfx1201")
    assert ("tessera_rocm.nvfp4_requantize",
            "tessera_rocm.mxfp4_folded_storage",
            "tessera_rocm.scaled_wmma_gemm")[stage] in target


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
def test_native_export_preserves_producer_edge_instead_of_retargeting():
    module = _full_graph(16,16,64,(0,16),(0,1,2,3,4))
    module.functions[0].body[1].operands[0] = "%arg0"
    graph = module.to_mlir(target="rocm_gfx1201", canonical=True)
    tool = find_tessera_opt()
    # This is valid semantic IR; the named program must preserve its actual SSA.
    run_tessera_opt(tool, graph, "--verify-each")
    with pytest.raises(RuntimeError, match="SSA edges"):
        run_tessera_opt(tool, graph,
                       "--tessera-autodiff-forward=export-scaled-primal=true")


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
def test_native_export_rejects_existing_member_symbol():
    graph = _full_graph(16,16,64,(0,16),(0,1,2,3,4)).to_mlir(
        target="rocm_gfx1201", canonical=True)
    insertion = graph.rfind("}")
    graph = graph[:insertion] + "  func.func private @nvfp4_resident__nvfp4_member_1()\n" + graph[insertion:]
    with pytest.raises(RuntimeError, match="symbol already exists"):
        run_tessera_opt(find_tessera_opt(), graph,
                       "--tessera-autodiff-forward=export-scaled-primal=true")


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
def test_adapter_never_constructs_graph_below_frontend(monkeypatch):
    from tessera.compiler import graph_ir
    from tessera.compiler.native_nvfp4_program import export_native_nvfp4_program
    source = _full_graph(16,16,64,(0,16),(0,1,2,3,4)).to_mlir(
        target="rocm_gfx1201",canonical=True)
    def refused(*args, **kwargs):
        raise AssertionError("backend adapter reconstructed Graph IR")
    monkeypatch.setattr(graph_ir.GraphIRFunction, "__init__", refused)
    monkeypatch.setattr(graph_ir.IROp, "__init__", refused)
    native = export_native_nvfp4_program(source)
    assert native.manifest["role_indices"] == [0,1,2,3,4]
    changed = native.manifest
    changed["role_indices"][0] = 99
    assert native.manifest["role_indices"] == [0,1,2,3,4]
    for index in range(3):
        assert native.project_member(index) == native.manifest["member_graphs"][index]


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
def test_native_member_records_roundtrip_and_reject_changed_projection(monkeypatch):
    from tessera.compiler import native_nvfp4_program as adapter
    source = _full_graph(16,16,64,(0,16),(2,0,4,1,3)).to_mlir(
        target="rocm_gfx1201", canonical=True)
    program = adapter.export_native_nvfp4_program(source)
    plan = program.manifest
    tool = find_tessera_opt()
    # The stored source is compiler-printed semantic IR, not Python normalization.
    run_tessera_opt(tool, plan["source_graph_ir"], "--verify-each")
    replay = adapter.export_native_nvfp4_program(plan["source_graph_ir"])
    assert replay.plan_json == program.plan_json
    for index, member in enumerate(plan["member_graphs"]):
        run_tessera_opt(tool, member, "--verify-each")
        assert program.project_member(index) == member
    original = adapter.run_tessera_opt
    def altered(tool, source, option):
        result = original(tool, source, option)
        return result.replace("@storage", "@retargeted_storage") if "select-scaled-member=1" in option else result
    monkeypatch.setattr(adapter, "run_tessera_opt", altered)
    with pytest.raises(ValueError, match="original compiler program"):
        program.project_member(1)


@pytest.mark.skipif(__import__("os").getenv("TESSERA_GFX1201_DEVICE_PROOF") != "1",
                    reason="exact gfx1201 and matching native runtime")
@pytest.mark.parametrize("shape,roles", [
    ((128,32,256),(0,1,2,3,4)),
    ((257,80,1024),(2,0,4,1,3)),
    ((256,64,64),(0,1,2,3,4)),
])
def test_public_native_partition_replays_without_graph_constructors(shape,roles,monkeypatch):
    import numpy as np
    import subprocess
    from tessera.compiler import graph_ir, rocm_nvfp4_program as frontend
    from tessera.compiler import rocm_nvfp4_resident as resident
    from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle
    m,n,k = shape
    args,offsets,converted,stored,expected = inputs_and_oracle(m,n,k)
    module = _full_graph(m,n,k,offsets,roles)
    def forbidden(*args, **kwargs):
        pytest.fail("native public packaging/replay reconstructed Graph IR")
    monkeypatch.setattr(graph_ir.GraphIRFunction, "__init__", forbidden)
    monkeypatch.setattr(graph_ir.IROp, "__init__", forbidden)
    for owner,names in [
        (frontend, ("_full_graph","build_nvfp4_ingest_graph","build_mxfp4_storage_graph","build_packed_consumer_module")),
        (resident, ("build_nvfp4_ingest_graph","build_mxfp4_storage_graph","author_packed_folded_shape_graph")),
    ]:
        for name in names:
            monkeypatch.setattr(owner,name,forbidden)
    program = frontend.package_traced_resident(module)
    assert program.native.native_plan_json is not None
    ordered = [None]*5
    for role,index in enumerate(roles):
        ordered[index] = args[role]
    output,receipts = program.execute(*ordered)
    np.testing.assert_allclose(output.astype(np.float32),expected,rtol=.008,atol=.015625)
    assert all(r["execution_kind"] == "native_gpu" for r in receipts)
    data = program.manifest()
    assert data["native_program"]["schema"] == "tessera.rocm.nvfp4_resident_program.v2"
    monkeypatch.setattr(subprocess,"run",forbidden)
    restored = frontend.program_from_manifest(data)
    output,_ = restored.execute(*ordered)
    np.testing.assert_allclose(output.astype(np.float32),expected,rtol=.008,atol=.015625)
    with restored.native.native_session(*args) as session:
        with pytest.raises(RuntimeError,match="invocation"):
            session.launch_matmul()
        session.run_combined()
        diagnostics = session.diagnostics()
        np.testing.assert_array_equal(diagnostics["packed"],converted[0])
        np.testing.assert_array_equal(diagnostics["plane"],stored[1])
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("field,value", [
    ("bytes",1), ("shape",[0,64]), ("ownership","readonly_input"),
    ("first_write",0), ("last_read",1),
])
def test_native_portable_record_rejects_changed_buffer_contract(field,value):
    from tessera.compiler.native_nvfp4_program import export_native_nvfp4_program
    from tessera.compiler.rocm_nvfp4_resident import _native_nvfp4_plan
    source = _full_graph(16,16,64,(0,16),(0,1,2,3,4)).to_mlir(
        target="rocm_gfx1201",canonical=True)
    program = export_native_nvfp4_program(source)
    plan = program.manifest
    _native_nvfp4_plan(program.plan_json)
    plan["buffers"][8][field] = value
    with pytest.raises(ValueError,match="record differs"):
        _native_nvfp4_plan(json.dumps(plan))
