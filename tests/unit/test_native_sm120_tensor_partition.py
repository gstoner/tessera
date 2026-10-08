"""Actual producer/matmul Graph outlining preserves SSA and typed fragments."""
import base64
import json
import re
import pytest
from tessera.compiler.nvidia_tensor_lhs import _semantic_graph
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt


def graph(producer="tessera.rmsnorm",dtype="fp16",order="row_major",roles=None):
    semantics={"producer":producer,"producer_attrs":{"axis":-1} if producer=="tessera.softmax" else {"eps":1e-5},
               "consumer":"tessera.matmul","consumer_attrs":{"output_dtype":"fp32","rhs_storage_order":order},
               "roles":roles or {"source":0,"rhs":1}}
    return _semantic_graph(17,32,11,dtype,semantics)


def export(source,index=None):
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip("matching native compiler required")
    option="export-scaled-primal=true"
    if index is not None:
        option+=f" select-scaled-member={index}"
    return run_tessera_opt(tool,source,"--tessera-autodiff-forward="+option)


def plan(source):
    text=export(source)
    match=re.search(r'tessera.native.sm120_tensor_program_json = "([^"]+)"',text)
    assert match is not None
    return json.loads(base64.b64decode(match[1],validate=True))


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("producer",["tessera.rmsnorm","tessera.layer_norm","tessera.softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("order",["row_major","col_major"])
def test_actual_tensor_members_reach_native_schedule_and_fragments(producer,dtype,order):
    source=graph(producer,dtype,order,{"source":1,"rhs":0}).to_mlir(
        target="nvidia_sm120",canonical=True)
    record=plan(source)
    assert record["role_indices"]==[1,0]
    assert [s["inputs"] for s in record["steps"]]==[[1],[2,0]]
    assert [s["outputs"] for s in record["steps"]]==[[2],[3]]
    assert record["buffers"][2]["ownership"]=="private_scratch"
    assert record["buffers"][2]["bytes"]==17*32*2
    assert record["buffers"][2]["first_write"]==0
    assert record["buffers"][2]["last_read"]==1
    assert record["buffers"][3]["ownership"]=="returned_output"
    assert record["buffers"][3]["last_read"]==2
    assert record["buffers"][3]["bytes"]==17*11*4
    replay=plan(record["source_graph_ir"])
    assert replay==record
    tool=find_tessera_opt()
    for index in range(2):
        member=export(source,index)
        assert member==record["member_graphs"][index]
        schedule=run_tessera_opt(tool,member,"--tessera-graph-to-schedule")
        tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
        assert ("tile.softmax_kernel" if producer=="tessera.softmax" else "tile.norm_kernel") in tile if index==0 else "tile.mma " in tile
        assert producer not in tile
    consumer=run_tessera_opt(tool,record["member_graphs"][1],"--tessera-nvidia-pipeline-sm120")
    assert consumer.count("tile.view ")==2
    assert consumer.count("tile.fragment_pack ")==2
    assert consumer.count("tile.fragment_zero ")==1
    assert consumer.count("tile.fragment_unpack ")==1
    assert "tile.mma " in consumer
    assert " : (!tile.fragment<" in next(line for line in consumer.splitlines() if " = tile.mma " in line)


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
def test_actual_tensor_export_refuses_retargeted_edge():
    module=graph()
    module.functions[0].body[1].operands[0]="%arg0"
    source=module.to_mlir(target="nvidia_sm120",canonical=True)
    run_tessera_opt(find_tessera_opt(),source,"--verify-each")
    with pytest.raises(RuntimeError,match="SSA edge"):
        export(source)


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
def test_actual_tensor_export_refuses_symbol_collision_before_outlining():
    source=graph().to_mlir(target="nvidia_sm120",canonical=True)
    index=source.rfind("}")
    source=source[:index]+"  func.func private @native_lhs__tensor_member_1()\n"+source[index:]
    with pytest.raises(RuntimeError,match="symbol already exists"):
        export(source)


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("producer",["tessera.rmsnorm","tessera.layer_norm","tessera.softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_native_schedule_adapter_never_constructs_graph(producer,dtype,monkeypatch):
    from tessera.compiler import graph_ir
    from tessera.compiler.native_sm120_tensor_program import export_native_sm120_tensor_graph
    source=graph(producer,dtype).to_mlir(target="nvidia_sm120",canonical=True)
    def forbidden(*args,**kwargs):
        pytest.fail("native tensor Schedule adapter reconstructed Graph")
    monkeypatch.setattr(graph_ir.GraphIRFunction,"__init__",forbidden)
    monkeypatch.setattr(graph_ir.IROp,"__init__",forbidden)
    native=export_native_sm120_tensor_graph(source)
    producer_artifact,consumer_artifact=native.scheduled_members()
    assert producer_artifact.input_shape==(17,32)
    assert producer_artifact.output_name==consumer_artifact.a_name=="edge"
    assert consumer_artifact.output_name=="out"
    assert (consumer_artifact.m,consumer_artifact.n,consumer_artifact.k)==(17,11,32)














@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_native_static_producer_chain_outlining(dtype):
    from copy import deepcopy
    module = graph(dtype=dtype)
    fn = module.functions[0]
    second = deepcopy(fn.body[0])
    second.op_name = "tessera.softmax"
    second.kwargs = {"axis": -1}
    second.operands = ["%" + fn.body[0].result]
    second.result = "chain_softmax"
    fn.body[-1].operands[0] = "%" + second.result
    fn.body.insert(1, second)
    source = module.to_mlir(target="nvidia_sm120", canonical=True)
    record = plan(source)
    assert record["schema"].endswith(".v3")
    assert [step["inputs"] for step in record["steps"]] == [[0], [2], [3, 1]]
    assert [step["outputs"] for step in record["steps"]] == [[2], [3], [4]]
    assert record["buffers"][2]["last_read"] == 1
    assert record["buffers"][3]["last_read"] == 2
    assert record["buffers"][4]["last_read"] == 3
    from tessera.compiler.native_sm120_tensor_program import export_native_sm120_tensor_graph
    members = export_native_sm120_tensor_graph(source).scheduled_members()
    assert len(members) == 3
    assert members[0].kind == "rmsnorm"
    assert members[1].kind == "softmax"




@pytest.mark.parametrize("retained", [False, True])
def test_legacy_certificate_cannot_drop_chain_members(retained):
    from types import SimpleNamespace
    from tessera.compiler.nvidia_tensor_lhs import TracedLhsProgram
    edge=SimpleNamespace(validate=lambda: None,m=1,k=1,n=1)
    semantics={} if retained else {"producer_chain":[]}
    program=TracedLhsProgram(edge=edge,graph_ir="module {}",argument_names=("source","rhs"),
                             semantics=semantics,producer_chain=(object(),) if retained else ())
    with pytest.raises(ValueError,match="native Graph member and lifetime"):
        program.validate()
