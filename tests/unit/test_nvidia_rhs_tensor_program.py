from dataclasses import replace
import numpy as np
import pytest
from tessera.compiler.graph_ir import GraphIRModule,GraphIRFunction,IRArg,IROp,IRType
from tessera.compiler.nvidia_tensor_rhs import package_rmsnorm_rhs_matmul
from tessera.compiler import scheduled_matmul,nvidia_native
from tests.unit.test_scheduled_matmul_consumers import _module

pytestmark=pytest.mark.skipif(scheduled_matmul.find_tessera_opt() is None or not nvidia_native.tools_available(),reason="native SM120 toolchain required")


def graphs(shape=(17,35,19),dtype="fp16"):
    _,k,n=shape
    element="f16" if dtype=="fp16" else "bf16"
    tensor=IRType(f"tensor<{k}x{n}x{element}>",(str(k),str(n)),dtype)
    producer=GraphIRModule(functions=[GraphIRFunction(name="rhs_rmsnorm",args=[IRArg("x",tensor)],
        result_types=[tensor],body=[IROp(result="normalized_rhs",op_name="tessera.rmsnorm",operands=["%x"],
            operand_types=[str(tensor)],result_type=str(tensor),kwargs={"eps":1e-5})],return_values=["%normalized_rhs"])])
    return producer,_module(target="nvidia_sm120",shape=shape,dtype=dtype)


def program(shape=(17,35,19),dtype="fp16"):
    return package_rmsnorm_rhs_matmul(*graphs(shape,dtype),pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_rhs_edge_native_projection_preserves_caller(dtype):
    p,c=graphs(dtype=dtype)
    edge=package_rmsnorm_rhs_matmul(p,c,pipeline_name="tessera-nvidia-pipeline-sm120")
    edge.validate()
    assert "rhs_storage_order" not in c.functions[0].body[0].kwargs
    assert "tile.view" in edge.consumer.tile_ir
    assert 'role = "b", transpose' in edge.consumer.tile_ir
    with pytest.raises(ValueError,match="shape drift"):
        replace(edge,n=edge.n+1).validate()
    with pytest.raises(ValueError,match="static shapes"):
        edge.execute_resident(np.zeros((edge.k,edge.n+1),np.float16),np.zeros((edge.m,edge.k),np.float16))


def test_rhs_conflicting_storage_not_replaced():
    p,c=graphs()
    c.functions[0].body[0].kwargs["rhs_storage_order"]="col_major"
    with pytest.raises(ValueError,match="row-major"):
        package_rmsnorm_rhs_matmul(p,c,pipeline_name="tessera-nvidia-pipeline-sm120")
    assert c.functions[0].body[0].kwargs["rhs_storage_order"]=="col_major"


def test_rhs_edge_rejects_tampered_package_identity_and_binding():
    edge=program()
    descriptor=edge.consumer.descriptor
    for change in (
        {"image_digest":"0"*64},
        {"provenance":{**descriptor.provenance,"b_layout":"col_major"}},
        {"provenance":{**descriptor.provenance,"tile_ir_digest":"0"*64}},
        {"buffers":tuple(replace(b,layout="col_major") if b.name==edge.rhs_name else b for b in descriptor.buffers)},
    ):
        consumer=replace(edge.consumer,descriptor=replace(descriptor,**change))
        with pytest.raises(ValueError):
            replace(edge,consumer=consumer).validate()
