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


def test_actual_tensor_export_refuses_retargeted_edge():
    module=graph()
    module.functions[0].body[1].operands[0]="%arg0"
    source=module.to_mlir(target="nvidia_sm120",canonical=True)
    run_tessera_opt(find_tessera_opt(),source,"--verify-each")
    with pytest.raises(RuntimeError,match="SSA edge"):
        export(source)


def test_actual_tensor_export_refuses_symbol_collision_before_outlining():
    source=graph().to_mlir(target="nvidia_sm120",canonical=True)
    index=source.rfind("}")
    source=source[:index]+"  func.func private @native_lhs__tensor_member_1()\n"+source[index:]
    with pytest.raises(RuntimeError,match="symbol already exists"):
        export(source)


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


@pytest.mark.parametrize("producer",["tessera.rmsnorm","tessera.layer_norm","tessera.softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("order",["row_major","col_major"])
def test_native_partition_executes_resident_edge_on_sm120(producer,dtype,order,monkeypatch):
    import numpy as np
    from tessera import runtime as rt
    from tessera.compiler import graph_ir
    from tessera.compiler.native_sm120_tensor_program import export_native_sm120_tensor_graph
    from tessera.compiler.nvidia_native import package_scheduled_tensor_matmul
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning SM120 GPU required")
    source_ir=graph(producer,dtype,order,{"source":1,"rhs":0}).to_mlir(
        target="nvidia_sm120",canonical=True)
    def forbidden(*args,**kwargs):
        pytest.fail("native partition packaging reconstructed Python Graph")
    monkeypatch.setattr(graph_ir.GraphIRFunction,"__init__",forbidden)
    monkeypatch.setattr(graph_ir.IROp,"__init__",forbidden)
    native=export_native_sm120_tensor_graph(source_ir)
    program=package_scheduled_tensor_matmul(*native.scheduled_members(),
        pipeline_name="tessera-nvidia-pipeline-sm120")
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    rng=np.random.default_rng(120081)
    x=rng.normal(0,.25,(17,32)).astype(storage)
    rhs=np.array(rng.normal(0,.25,(32,11)),dtype=storage,
                 order="F" if order=="col_major" else "C")
    xf=x.astype(np.float64)
    if producer=="tessera.softmax":
        exp=np.exp(xf-xf.max(axis=-1,keepdims=True))
        expected_edge=(exp/exp.sum(axis=-1,keepdims=True)).astype(storage)
    else:
        centered=xf-xf.mean(axis=-1,keepdims=True) if producer=="tessera.layer_norm" else xf
        expected_edge=(centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)).astype(storage)
    expected=expected_edge.astype(np.float64)@rhs.astype(np.float64)
    with program.execute_resident(x,rhs) as result:
        assert result.producer_receipt["execution_kind"]=="native_gpu"
        assert result.consumer_receipt["execution_kind"]=="native_gpu"
        actual_edge=result.intermediate.numpy().astype(np.float64)
        actual=result.output.numpy()
        np.testing.assert_allclose(actual_edge,expected_edge.astype(np.float64),
                                   rtol=8e-3 if dtype=="bf16" else 2e-3,atol=2e-3)
        np.testing.assert_allclose(actual,expected,
                                   rtol=2e-2 if dtype=="bf16" else 3e-3,
                                   atol=1e-2 if dtype=="bf16" else 2e-3)



@pytest.mark.parametrize("producer",["tessera.rmsnorm","tessera.layer_norm","tessera.softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("fused",[False,True])
def test_public_native_partition_and_portable_replay_without_graph_authors(producer,dtype,fused,monkeypatch):
    import numpy as np
    import subprocess
    from tessera import runtime as rt
    from tessera.compiler import graph_ir,nvidia_tensor_lhs as lhs
    from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
    if rt._nvidia_device_name()!="sm_120":
        pytest.skip("owning SM120 GPU required")
    semantics={"producer":producer,
        "producer_attrs":{"axis":-1} if producer=="tessera.softmax" else {"eps":1e-5},
        "consumer":"tessera.matmul","consumer_attrs":{"output_dtype":"fp16" if fused else "fp32",
            "rhs_storage_order":"row_major"},
        "roles":{"source":1,"rhs":0}}
    if fused:
        semantics["consumer_attrs"].update(bias="bias",residual="residual",activation="relu")
        semantics["roles"].update(bias=3,residual=2)
    from tests.device.nvidia.test_lhs_tensor_jit import _oracle
    module=lhs._semantic_graph(17,35,19,dtype,semantics)
    before=module.to_mlir(target="nvidia_sm120",canonical=True)
    def forbidden(*args,**kwargs):
        pytest.fail("public native route reconstructed Python Graph or recompiled portable replay")
    monkeypatch.setattr(graph_ir.GraphIRFunction,"__init__",forbidden)
    monkeypatch.setattr(graph_ir.IROp,"__init__",forbidden)
    monkeypatch.setattr(lhs,"_semantic_graph",forbidden)
    monkeypatch.setattr(lhs,"_partitions",forbidden)
    program=lhs.package_traced_lhs(module)
    assert module.to_mlir(target="nvidia_sm120",canonical=True)==before
    assert program.native_plan_json is not None
    manifest=program.manifest()
    assert manifest["schema"]=="tessera.nvidia.lhs_tensor_program.v2"
    monkeypatch.setattr(subprocess,"run",forbidden)
    restored=lhs.from_manifest(manifest)
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    rng=np.random.default_rng(120519)
    x=rng.normal(0,.2,(17,35)).astype(storage)
    rhs=rng.normal(0,.2,(35,19)).astype(storage)
    bias=rng.normal(0,.2,19).astype(np.float32)
    residual=rng.normal(0,.2,(17,19)).astype(np.float32)
    expected=_oracle(x,rhs,{"tessera.rmsnorm":"rmsnorm","tessera.layer_norm":"layernorm","tessera.softmax":"softmax"}[producer],
        bias if fused else None,residual if fused else None)
    args=[rhs,x,residual,bias] if fused else [rhs,x]
    from contextlib import closing
    with closing(PreparedLhsCall(restored)) as prepared:
        actual,receipt=prepared(args)
        assert receipt["execution_kind"]=="native_gpu"
        np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)



@pytest.mark.parametrize("field,value",[
    ("bytes",1),("ownership","readonly_input"),("first_write",-1),("last_read",0),
    ("shape",[17,36]),("source_graph_ir","module {}"),("member_graph","module {}"),
])
def test_portable_native_lifetime_or_graph_drift_refused_before_cuda(field,value,monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import nvidia_tensor_lhs as lhs
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    import subprocess
    if rt._nvidia_device_name()!="sm_120":
        pytest.skip("owning SM120 packaging GPU required")
    program=lhs.package_traced_lhs(graph())
    manifest=program.manifest()
    record=json.loads(manifest["native_plan_json"])
    if field=="source_graph_ir":
        record[field]=value
        manifest["graph_ir"]=value
    elif field=="member_graph":
        record["member_graphs"][0]=value
    else:
        record["buffers"][2][field]=value
    manifest["native_plan_json"]=json.dumps(record,sort_keys=True)
    manifest["contract_digest"]=lhs._digest({k:v for k,v in manifest.items() if k!="contract_digest"})
    def forbidden(*args,**kwargs):
        pytest.fail("invalid portable native program reached CUDA or compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",forbidden)
    with pytest.raises(ValueError,match="native tensor|native LHS"):
        lhs.from_manifest(manifest)



@pytest.mark.parametrize("axes",[("M",),("N",),("K",),("M","N"),("M","K"),("N","K"),("M","N","K")])
@pytest.mark.parametrize("producer",["tessera.rmsnorm","tessera.layer_norm","tessera.softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_native_bounded_projection_replays_without_python_graph_or_compiler(axes,producer,dtype,monkeypatch):
    import numpy as np
    import subprocess
    from contextlib import closing
    from tessera import runtime as rt
    from tessera.compiler import graph_ir,nvidia_tensor_lhs as lhs
    from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
    from tests.device.nvidia.test_lhs_tensor_jit import _oracle,_storage
    if rt._nvidia_device_name()!="sm_120":
        pytest.skip("owning SM120 GPU required")
    order="row_major" if "N" in axes else "col_major"
    sem={"producer":producer,"producer_attrs":{"axis":-1} if producer=="tessera.softmax" else {"eps":1e-5},
        "consumer":"tessera.matmul","consumer_attrs":{"output_dtype":"fp32","rhs_storage_order":order},
        "roles":{"source":1,"rhs":0}}
    module=lhs._semantic_graph(5,7,3,dtype,sem)
    original=module.to_mlir(target="nvidia_sm120",canonical=True)
    def forbidden(*args,**kwargs):
        pytest.fail("native bounded route reconstructed Graph or compiled portable replay")
    monkeypatch.setattr(graph_ir.GraphIRFunction,"__init__",forbidden)
    monkeypatch.setattr(graph_ir.IROp,"__init__",forbidden)
    monkeypatch.setattr(lhs,"_semantic_graph",forbidden)
    monkeypatch.setattr(lhs,"_partitions",forbidden)
    bounds={axis:{"M":16,"N":8,"K":32}[axis] for axis in axes}
    program=lhs.package_traced_lhs(module,shape_bounds=bounds)
    assert module.to_mlir(target="nvidia_sm120",canonical=True)==original
    plan=json.loads(program.native_plan_json)
    assert plan["schema"]=="tessera.native.sm120_tensor_program.v2"
    assert plan["active_shape"]==[5,3,7]
    assert plan["dynamic_axes"]==list(axes)
    assert "?" in plan["source_graph_ir"]
    assert "shape_bounds" in plan["source_graph_ir"]
    assert "?" not in plan["original_graph_ir"]
    capacities=tuple(bounds.get(axis,size) for axis,size in zip(("M","N","K"),(5,3,7),strict=True))
    assert plan["shape_bounds"]==list(capacities)
    assert plan["buffers"][2]["bytes"]==capacities[0]*capacities[2]*2
    manifest=program.manifest()
    monkeypatch.setattr(subprocess,"run",forbidden)
    restored=lhs.from_manifest(manifest)
    storage=_storage(dtype)
    rng=np.random.default_rng(120811)
    images=(restored.edge.producer.image.image_digest,restored.edge.consumer.image.image_digest)
    with closing(PreparedLhsCall(restored)) as prepared:
        capacity=None
        for extent in (capacities,(5,3,7),tuple(1 if axis in axes else size for axis,size in zip(("M","N","K"),(5,3,7),strict=True))):
            m,n,k=extent
            x=rng.normal(0,.2,(m,k)).astype(storage)
            rhs=np.array(rng.normal(0,.2,(k,n)),dtype=storage,order="F" if order=="col_major" else "C")
            actual,receipt=prepared([rhs,x])
            expected=_oracle(x,rhs,{"tessera.rmsnorm":"rmsnorm","tessera.layer_norm":"layernorm","tessera.softmax":"softmax"}[producer])
            np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)
            assert receipt["native_call_binding"]=="prepared_cpp_dynamic_tensor_matmul"
            current=prepared.scratch_stats()
            if capacity is None:capacity=current
            assert current==capacity
    assert (restored.edge.producer.image.image_digest,restored.edge.consumer.image.image_digest)==images



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


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_native_producer_chain_prepared_and_resident(dtype, monkeypatch):
    from copy import deepcopy
    from contextlib import closing
    import numpy as np
    import subprocess
    from tessera import runtime as rt
    from tessera.compiler import nvidia_tensor_lhs as lhs
    from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning SM120 GPU required")
    module = graph(dtype=dtype)
    fn = module.functions[0]
    second = deepcopy(fn.body[0])
    second.op_name = "tessera.softmax"
    second.kwargs = {"axis": -1}
    second.operands = ["%" + fn.body[0].result]
    second.result = "chain_softmax"
    fn.body[-1].operands[0] = "%" + second.result
    fn.body.insert(1, second)
    program = lhs.package_traced_lhs(module)
    manifest = program.manifest()
    assert manifest["schema"] == "tessera.nvidia.lhs_tensor_program.v3"
    def forbidden(*args, **kwargs):
        pytest.fail("portable chain execution recompiled")
    monkeypatch.setattr(subprocess, "run", forbidden)
    restored = lhs.from_manifest(manifest)
    storage = np.float16
    if dtype == "bf16":
        import ml_dtypes
        storage = ml_dtypes.bfloat16
    rng = np.random.default_rng(125)
    x = rng.normal(0, .2, (17, 32)).astype(storage)
    rhs = rng.normal(0, .2, (32, 11)).astype(storage)
    xf = x.astype(np.float64)
    normalized = (xf / np.sqrt(np.mean(xf*xf, axis=-1, keepdims=True)+1e-5)).astype(storage)
    nf = normalized.astype(np.float64)
    exponential = np.exp(nf-np.max(nf, axis=-1, keepdims=True))
    edge = (exponential / exponential.sum(axis=-1, keepdims=True)).astype(storage)
    expected = edge.astype(np.float64) @ rhs.astype(np.float64)
    with closing(PreparedLhsCall(restored)) as prepared:
        for _ in range(2):
            actual, receipt = prepared([x, rhs])
            assert len(receipt["component_receipts"]) == 3
            np.testing.assert_allclose(actual, expected, rtol=.015, atol=.002)
    with restored.execute_resident(x, rhs) as result:
        assert len(result.producer_receipt["component_receipts"]) == 2
        np.testing.assert_allclose(result.device_session.download(result.output), expected, rtol=.015, atol=.002)


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
