"""Ordinary JIT LHS producer fixtures and exact-device proof."""
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests._support.nvidia import nvidia_cuda_host_ready

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning NVIDIA host required")


@ts.jit(target="nvidia_sm120")
def rms_lhs(source,rhs):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs,output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def layer_lhs(source,rhs):
    return ts.ops.matmul(ts.ops.layer_norm(source,eps=1e-5),rhs,output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def softmax_lhs(source,rhs):
    return ts.ops.matmul(ts.ops.softmax(source,axis=-1),rhs,output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def rms_lhs_fused(source,rhs,bias,residual):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs,bias=bias,
                         activation="relu",residual=residual,output_dtype="fp16")


@ts.jit(target="nvidia_sm120")
def layer_lhs_fused(source,rhs,bias,residual):
    return ts.ops.matmul(ts.ops.layer_norm(source,eps=1e-5),rhs,bias=bias,
                         activation="relu",residual=residual,output_dtype="fp16")


@ts.jit(target="nvidia_sm120")
def softmax_lhs_fused(source,rhs,bias,residual):
    return ts.ops.matmul(ts.ops.softmax(source,axis=-1),rhs,bias=bias,
                         activation="relu",residual=residual,output_dtype="fp16")


@ts.jit(target="nvidia_sm120")
def layer_lhs_reordered(residual,rhs,source,bias):
    return ts.ops.matmul(ts.ops.layer_norm(source,eps=1e-5),rhs,bias=bias,
                         activation="relu",residual=residual,output_dtype="fp16")


def _storage(dtype):
    if dtype=="fp16":
        return np.float16
    import ml_dtypes
    return ml_dtypes.bfloat16


def _oracle(source,rhs,kind,bias=None,residual=None):
    x=source.astype(np.float64)
    if kind=="rmsnorm":
        y=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)
    elif kind=="layernorm":
        centered=x-np.mean(x,axis=-1,keepdims=True)
        y=centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)
    else:
        exp=np.exp(x-np.max(x,axis=-1,keepdims=True))
        y=exp/np.sum(exp,axis=-1,keepdims=True)
    out=y.astype(source.dtype).astype(np.float64)@rhs.astype(np.float64)
    if bias is not None:
        out=np.maximum(out+bias,0)+residual
        out=out.astype(np.float16)
    return out


@pytest.mark.parametrize("kind",["rmsnorm","layernorm","softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("fused",[False,True])
@pytest.mark.parametrize("shape",[(17,35,19),(64,256,64)])
@pytest.mark.parametrize("rhs_order",["C","F"])
def test_lhs_native_cached_portable(kind,dtype,fused,shape,rhs_order,monkeypatch):
    m,k,n=shape
    storage=_storage(dtype)
    rng=np.random.default_rng(120517)
    padded=np.zeros((m,k+7),storage)
    padded[:,:k]=(rng.normal(size=(m,k))*.2).astype(storage)
    source=padded[:,:k]
    rhs=np.array((rng.normal(size=(k,n))*.2),dtype=storage,order=rhs_order)
    bias=(rng.normal(size=n)*.2).astype(np.float32)
    residual=(rng.normal(size=(m,n))*.2).astype(np.float32)
    plain={"rmsnorm":rms_lhs,"layernorm":layer_lhs,"softmax":softmax_lhs}
    epilogue={"rmsnorm":rms_lhs_fused,"layernorm":layer_lhs_fused,"softmax":softmax_lhs_fused}
    function=(epilogue if fused else plain)[kind]
    args=(source,rhs,bias,residual) if fused else (source,rhs)
    expected=_oracle(source,rhs,kind,bias if fused else None,residual if fused else None)
    actual=function(*args)
    np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)
    assert function.execution_kind=="native_gpu"
    packages=function.native_lhs_packages()
    assert len(packages)==2 and packages[0].descriptor.provenance["kind"]==kind
    layout = "row_major" if rhs_order == "C" else "col_major"
    assert packages[1].descriptor.provenance["b_layout"] == layout
    assert (".a_row_b_" in packages[1].descriptor.abi_id) == (rhs_order == "C")
    artifact=rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
    assert artifact.metadata["compiler_path"]=="canonical_nvidia_lhs_program"
    receipt=rt.launch(artifact,args)
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu", receipt
    assert len(receipt["component_receipts"])==2
    np.testing.assert_array_equal(actual,receipt["output"])
    def unavailable(*args,**kwargs):
        raise AssertionError("eager execution or recompilation")
    monkeypatch.setattr(function,"_fn",unavailable)
    monkeypatch.setattr(function,"compile_native_lhs_matmul",unavailable)
    np.testing.assert_array_equal(actual,function(*args))
    assert function.native_lhs_packages()==packages


def test_lhs_reordered_and_caller_graph_unchanged():
    from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    bias=np.ones(19,np.float32)
    residual=np.ones((17,19),np.float32)
    args=(residual,rhs,source,bias)
    graph=layer_lhs_reordered._traced_autodiff_module(args,{})
    before=graph.to_mlir(canonical=True,target="nvidia_sm120")
    package_traced_lhs(graph)
    assert graph.to_mlir(canonical=True,target="nvidia_sm120")==before
    actual=layer_lhs_reordered(residual=residual,rhs=rhs,source=source,bias=bias)
    np.testing.assert_array_equal(actual,np.full((17,19),2,np.float16))
    receipt=rt.launch(layer_lhs_reordered.runtime_artifact(),dict(
        residual=residual,rhs=rhs,source=source,bias=bias))
    assert receipt["ok"]
    np.testing.assert_array_equal(actual,receipt["output"])


@pytest.mark.parametrize("field,value",[("axis",0),("gamma",2.),("beta",.5)])
def test_lhs_rejects_unimplemented_semantics_before_compile(field,value,monkeypatch):
    from copy import deepcopy
    from tessera.compiler import nvidia_tensor_lhs as lhs
    graph=deepcopy(layer_lhs._traced_autodiff_module((
        np.ones((17,35),np.float16),np.ones((35,19),np.float16)),{}))
    graph.functions[0].body[0].kwargs[field]=value
    def unavailable(*args,**kwargs):
        raise AssertionError("invalid semantics reached compiler")
    monkeypatch.setattr(lhs,"find_tessera_opt",unavailable)
    with pytest.raises(ValueError,match="affine/axis"):
        lhs.package_traced_lhs(graph)


def test_lhs_manifest_semantics_checked_before_cuda(monkeypatch):
    from copy import deepcopy
    from tessera.compiler import nvidia_tensor_lhs as lhs
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    args=(np.ones((17,35),np.float16),np.ones((35,19),np.float16))
    layer_lhs(*args)
    artifact=layer_lhs.runtime_artifact()
    metadata=deepcopy(artifact.metadata)
    manifest=metadata["native_program"]
    manifest["semantics"]["producer"]="tessera.rmsnorm"
    manifest["contract_digest"]=lhs._digest({k:v for k,v in manifest.items() if k!="contract_digest"})
    def unavailable(*args,**kwargs):
        raise AssertionError("invalid semantics reached CUDA")
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",unavailable)
    receipt=rt.launch(rt.RuntimeArtifact(graph_ir=artifact.graph_ir,metadata=metadata),args)
    assert not receipt["ok"] and "semantic certificate" in receipt["reason"]


@ts.jit(target="nvidia_sm120")
def rms_lhs_gelu(source,rhs):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs,activation="gelu",output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def rms_lhs_silu(source,rhs):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs,activation="silu",output_dtype="fp32")


@pytest.mark.parametrize("activation,function",[("gelu",rms_lhs_gelu),("silu",rms_lhs_silu)])
def test_lhs_activation_only_native(activation,function):
    rng=np.random.default_rng(12)
    source=(rng.normal(size=(17,35))*.2).astype(np.float16)
    rhs=(rng.normal(size=(35,19))*.2).astype(np.float16)
    accum=_oracle(source,rhs,"rmsnorm")
    expected=(.5*accum*(1+np.tanh(np.sqrt(2/np.pi)*(accum+.044715*accum**3)))
              if activation=="gelu" else accum/(1+np.exp(-accum)))
    np.testing.assert_allclose(function(source,rhs),expected,rtol=.015,atol=.015)
    assert function.execution_kind=="native_gpu"
    replay=rt.launch(rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json()),(source,rhs))
    assert replay["ok"],replay
    np.testing.assert_allclose(replay["output"],expected,rtol=.015,atol=.015)


@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_lhs_fresh_process_replay_without_compiler(dtype,tmp_path):
    import os
    import subprocess
    import sys
    source=np.ones((17,35),_storage(dtype))
    rhs=np.ones((35,19),_storage(dtype))
    bias=np.ones(19,np.float32)
    residual=np.ones((17,19),np.float32)
    layer_lhs_fused(source,rhs,bias,residual)
    path=tmp_path/"program.json"
    path.write_text(layer_lhs_fused.runtime_artifact().to_json())
    script = """
import sys
import numpy as np
from tessera import runtime as rt
from tessera.compiler import nvidia_tensor_lhs as lhs
dtype=np.float16
if sys.argv[2]=='bf16':
 import ml_dtypes
 dtype=ml_dtypes.bfloat16
def unavailable(*args,**kwargs):
 raise AssertionError('portable replay called compiler')
lhs.find_tessera_opt=unavailable
artifact=rt.RuntimeArtifact.from_json(open(sys.argv[1]).read())
values=(np.ones((17,35),dtype),np.ones((35,19),dtype),np.ones(19,np.float32),np.ones((17,19),np.float32))
receipt=rt.launch(artifact,values)
assert receipt['ok'],receipt
assert receipt['execution_kind']=='native_gpu'
assert len(receipt['component_receipts'])==2
np.testing.assert_array_equal(receipt['output'],np.full((17,19),2,np.float16))
print('fresh replay native passed')
"""
    env=dict(os.environ,TESSERA_OPT="/nonexistent/tessera-opt")
    result=subprocess.run([sys.executable,"-c",script,str(path),dtype],env=env,
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr


def test_lhs_storage_switch_keeps_distinct_sealed_packages(monkeypatch):
    source = np.arange(17*35,dtype=np.float16).reshape(17,35)/1000
    rhs = np.arange(35*19,dtype=np.float16).reshape(35,19)/1000
    snapshots = {}
    for order in ("C","F"):
        right = np.array(rhs,copy=True,order=order)
        actual = rms_lhs(source,right)
        snapshots[order] = (actual.copy(),rms_lhs.native_lhs_packages())
    assert snapshots["C"][1][1].image.image_digest != snapshots["F"][1][1].image.image_digest
    np.testing.assert_array_equal(snapshots["C"][0],snapshots["F"][0])
    def unavailable(*args,**kwargs):
        raise AssertionError("warmed layout switch recompiled")
    monkeypatch.setattr(rms_lhs,"compile_native_lhs_matmul",unavailable)
    for order in ("F","C","F","C"):
        np.testing.assert_array_equal(rms_lhs(source,np.array(rhs,order=order)),snapshots[order][0])
        assert rms_lhs.native_lhs_packages() == snapshots[order][1]


@pytest.mark.parametrize("order",["C","F"])
def test_lhs_native_layout_drift_refused_before_allocation(order,monkeypatch):
    from dataclasses import replace
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16,order=order)
    program=rms_lhs.compile_native_lhs_matmul(source,rhs)
    consumer=program.edge.consumer
    provenance=dict(consumer.descriptor.provenance)
    provenance["b_layout"]="col_major" if order=="C" else "row_major"
    bad=replace(program.edge,consumer=replace(consumer,descriptor=replace(
        consumer.descriptor,provenance=provenance)))
    def unavailable(*args,**kwargs):
        raise AssertionError("layout drift reached device allocation")
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",unavailable)
    with pytest.raises(ValueError,match="RHS layout differs"):
        bad.execute_resident(source,rhs)


@ts.jit(target="nvidia_sm120")
def rms_softmax_lhs(source, rhs):
    return ts.ops.matmul(ts.ops.softmax(ts.ops.rmsnorm(source, eps=1e-5), axis=-1),
                         rhs, output_dtype="fp32")


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("rhs_order", ["C", "F"])
def test_public_native_producer_chain(dtype, rhs_order, monkeypatch):
    import subprocess
    storage = _storage(dtype)
    rng = np.random.default_rng(120518)
    source = rng.normal(0, .2, (17, 35)).astype(storage)
    rhs = np.array(rng.normal(0, .2, (35, 19)), dtype=storage, order=rhs_order)
    x = source.astype(np.float64)
    normalized = (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+1e-5)).astype(storage)
    y = normalized.astype(np.float64)
    exponential = np.exp(y-np.max(y, axis=-1, keepdims=True))
    edge = (exponential/exponential.sum(axis=-1, keepdims=True)).astype(storage)
    expected = edge.astype(np.float64) @ rhs.astype(np.float64)
    actual = rms_softmax_lhs(source, rhs)
    assert rms_softmax_lhs.execution_kind == "native_gpu"
    assert len(rms_softmax_lhs.native_lhs_packages()) == 3
    np.testing.assert_allclose(actual, expected, rtol=.015, atol=.002)
    artifact = rt.RuntimeArtifact.from_json(rms_softmax_lhs.runtime_artifact().to_json())
    def forbidden(*args, **kwargs):
        raise AssertionError("chain replay used eager execution or compiler")
    monkeypatch.setattr(rms_softmax_lhs, "_fn", forbidden)
    monkeypatch.setattr(rms_softmax_lhs, "compile_native_lhs_matmul", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    np.testing.assert_array_equal(actual, rms_softmax_lhs(source, rhs))
    receipt = rt.launch(artifact, (source, rhs))
    assert receipt["ok"] and len(receipt["component_receipts"]) == 3, receipt
    np.testing.assert_array_equal(actual, receipt["output"])

@ts.jit(target="nvidia_sm120")
def layer_rms_softmax_lhs(source, rhs):
    normalized=ts.ops.layer_norm(source, eps=1e-5)
    normalized=ts.ops.rmsnorm(normalized, eps=1e-5)
    return ts.ops.matmul(ts.ops.softmax(normalized, axis=-1), rhs, output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def layer_rms_softmax_lhs_fused(source, rhs, bias, residual):
    normalized=ts.ops.layer_norm(source, eps=1e-5)
    normalized=ts.ops.rmsnorm(normalized, eps=1e-5)
    return ts.ops.matmul(ts.ops.softmax(normalized, axis=-1), rhs, bias=bias,
                         residual=residual, activation="relu", output_dtype="fp16")


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("rhs_order", ["C", "F"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("shape", [(17,35,19), (64,256,64)])
def test_public_three_producer_scratch_alternation(dtype,rhs_order,fused,shape,monkeypatch):
    import subprocess
    m,k,n=shape
    storage=_storage(dtype)
    rng=np.random.default_rng(120521)
    source=rng.normal(0,.2,(m,k)).astype(storage)
    rhs=np.array(rng.normal(0,.2,(k,n)),dtype=storage,order=rhs_order)
    bias=rng.normal(0,.2,n).astype(np.float32)
    residual=rng.normal(0,.2,(m,n)).astype(np.float32)
    xf=source.astype(np.float64)
    centered=xf-xf.mean(axis=-1,keepdims=True)
    layer=(centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)).astype(storage)
    lf=layer.astype(np.float64)
    rms=(lf/np.sqrt(np.mean(lf*lf,axis=-1,keepdims=True)+1e-5)).astype(storage)
    rf=rms.astype(np.float64)
    exponential=np.exp(rf-rf.max(axis=-1,keepdims=True))
    edge=(exponential/exponential.sum(axis=-1,keepdims=True)).astype(storage)
    expected=edge.astype(np.float64)@rhs.astype(np.float64)
    if fused:
        expected=(np.maximum(expected+bias,0)+residual).astype(np.float16)
    function=layer_rms_softmax_lhs_fused if fused else layer_rms_softmax_lhs
    args=(source,rhs,bias,residual) if fused else (source,rhs)
    actual=function(*args)
    assert function.execution_kind=="native_gpu"
    assert len(function.native_lhs_packages())==4
    np.testing.assert_allclose(actual,expected,rtol=.015,atol=.002)
    program=function._nvidia_lhs_last_program
    artifact=rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
    def forbidden(*args,**kwargs):
        raise AssertionError("three-producer execution recompiled or ran eagerly")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(function,"_fn",forbidden)
    monkeypatch.setattr(function,"compile_native_lhs_matmul",forbidden)
    for _ in range(2):
        np.testing.assert_array_equal(actual,function(*args))
    receipt=rt.launch(artifact,args)
    assert receipt["ok"] and len(receipt["component_receipts"])==4,receipt
    np.testing.assert_array_equal(actual,receipt["output"])
    with program.execute_resident(*args) as result:
        assert len(result.producer_receipt["component_receipts"])==3
        np.testing.assert_allclose(result.device_session.download(result.output),
                                   expected,rtol=.015,atol=.002)
