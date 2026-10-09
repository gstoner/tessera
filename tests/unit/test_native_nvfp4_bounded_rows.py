"""Host-free native compiler contracts for bounded NVFP4 activation rows.

No HIP execution is claimed here; the native compiler owns capacity projection.
"""
import copy
import json

import pytest

from tessera.compiler.native_nvfp4_program import export_native_nvfp4_program
from tessera.compiler.rocm_nvfp4_program import _full_graph
from tessera.compiler.rocm_nvfp4_resident import _native_nvfp4_plan
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = [pytest.mark.compiler_route,pytest.mark.usefixtures("production_compiler")]


def source(rows=17,bound=256,roles=(0,1,2,3,4),bound_text=None):
    module=_full_graph(rows,16,64,(0,16),roles)
    module.module_attrs["tessera.native.nvfp4_m_bound"] = bound_text or f"{bound} : i64"
    return module.to_mlir(target="rocm_gfx1201",canonical=True)


@pytest.mark.parametrize("rows",[1,17,200,256])
@pytest.mark.parametrize("roles",[(0,1,2,3,4),(2,0,4,1,3)])
def test_native_capacity_preserves_ingest_edges_and_original_witness(rows,roles):
    graph=source(rows=rows,roles=roles)
    native=export_native_nvfp4_program(graph)
    plan=_native_nvfp4_plan(native.plan_json)
    assert plan["schema"]=="tessera.native.nvfp4_program.v2"
    assert (plan["active_m"],plan["m_bound"])==(rows,256)
    assert "tessera.native.nvfp4_m_bound" in plan["original_graph_ir"]
    assert "tessera.native.nvfp4_m_bound" not in plan["source_graph_ir"]
    assert plan["buffers"][roles[3]]["shape"]==[256,64]
    assert plan["buffers"][roles[4]]["shape"]==[256]
    assert plan["buffers"][10]["shape"]==[256,16]
    assert plan["buffers"][10]["bytes"]==256*16*2
    assert plan["buffers"][5]["last_read"]==1
    assert plan["buffers"][8]["last_read"]==2
    assert plan["buffers"][10]["ownership"]=="returned_output"
    assert export_native_nvfp4_program(graph).manifest==plan
    assert export_native_nvfp4_program(plan["original_graph_ir"]).manifest==plan
    static=export_native_nvfp4_program(plan["source_graph_ir"]).manifest
    assert static["schema"]=="tessera.native.nvfp4_program.v1"
    assert static["buffers"]==plan["buffers"]
    assert native.project_member(0)==plan["member_graphs"][0]
    assert native.project_member(1)==plan["member_graphs"][1]


@pytest.mark.parametrize("stage,carrier",[
    (0,"tile.nvfp4_requantize_kernel"),
    (1,"tile.mxfp4_folded_storage_kernel"),
    (2,"tile.scaled_matmul_kernel")])
def test_bounded_member_uses_actual_native_schedule_tile_and_target(stage,carrier):
    native=export_native_nvfp4_program(source())
    member=native.project_member(stage)
    tool=find_tessera_opt()
    schedule=run_tessera_opt(tool,member,"--tessera-graph-to-schedule")
    tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    target=run_tessera_opt(tool,tile,"--lower-tile-to-rocm=arch=gfx1201")
    assert carrier in tile
    assert ("tessera_rocm.nvfp4_requantize","tessera_rocm.mxfp4_folded_storage",
            "tessera_rocm.scaled_wmma_gemm")[stage] in target
    if stage==2:
        assert "m = 256 : i64" in target


@pytest.mark.parametrize("bound",["0 : i64","-1 : i64","16 : i64","256 : i32",
                                  "4611686018427387904 : i64","true"])
def test_invalid_capacity_is_rejected_by_native_compiler(bound):
    with pytest.raises(RuntimeError,match="NVFP4 bounded rows"):
        export_native_nvfp4_program(source(bound_text=bound))


@pytest.mark.parametrize("change",["active","bound","scale","output","witness"])
def test_portable_plan_rejects_inconsistent_bounded_capacity(change):
    plan=copy.deepcopy(export_native_nvfp4_program(source()).manifest)
    if change=="active":plan["active_m"]=257
    elif change=="bound":plan["m_bound"]=128
    elif change=="scale":plan["buffers"][plan["role_indices"][4]]["shape"]=[128]
    elif change=="output":plan["buffers"][10]["shape"]=[128,16]
    else:plan["original_graph_ir"]=""
    with pytest.raises(ValueError,match="capacity"):
        _native_nvfp4_plan(json.dumps(plan))
