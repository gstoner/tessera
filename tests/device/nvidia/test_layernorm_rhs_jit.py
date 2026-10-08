"""LayerNorm RHS frontend/native/portable numerical and semantic proof."""
from copy import deepcopy
import hashlib
import json
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests._support.nvidia import nvidia_cuda_host_ready
from tessera.compiler.nvidia_tensor_rhs import package_traced_norm_rhs

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning NVIDIA host required")


@ts.jit(target="nvidia_sm120")
def layer_rhs(lhs,source):
    return ts.ops.matmul(lhs,ts.ops.layer_norm(source,eps=1e-5),output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def layer_rhs_reordered(source,lhs):
    return ts.ops.matmul(lhs,ts.ops.layer_norm(source,eps=1e-5),output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def layer_rhs_default(lhs,source):
    return ts.ops.matmul(lhs,ts.ops.layer_norm(source),output_dtype="fp32")


def storage_dtype(dtype):
    if dtype=="fp16":
        return np.float16
    import ml_dtypes
    return ml_dtypes.bfloat16


def oracle(a,x):
    xf=x.astype(np.float64)
    centered=xf-np.mean(xf,axis=-1,keepdims=True)
    normalized=(centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)).astype(x.dtype)
    return a.astype(np.float64)@normalized.astype(np.float64)


@pytest.mark.parametrize("function",[layer_rhs,layer_rhs_reordered])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("shape",[(16,16,8),(17,35,19),(64,256,64)])
def test_layer_rhs_native_cached_portable(function,dtype,shape,monkeypatch):
    m,k,n=shape
    storage=storage_dtype(dtype)
    rng=np.random.default_rng(120416)
    a=(rng.normal(size=(m,k))*.2).astype(storage)
    padded=np.full((k,n+5),17,storage)
    padded[:,:n]=(rng.normal(size=(k,n))*.2).astype(storage)
    x=padded[:,:n]
    expected=oracle(a,x)
    actual=function(lhs=a,source=x)
    np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)
    assert function.execution_kind=="native_gpu"
    packages=function.native_rhs_packages()
    assert packages[0].descriptor.provenance["kind"]=="layernorm"
    assert packages[1].descriptor.provenance["b_layout"]=="row_major"
    artifact=rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
    assert artifact.metadata["native_program"]["schema"]=="tessera.nvidia.norm_rhs_program.v2"
    receipt=rt.launch(artifact,{"lhs":a,"source":x})
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu"
    assert len(receipt["component_receipts"])==2
    np.testing.assert_array_equal(actual,receipt["output"])
    def unavailable(*args,**kwargs):
        raise AssertionError("eager execution or recompilation")
    monkeypatch.setattr(function,"_fn",unavailable)
    monkeypatch.setattr(function,"compile_native_rhs_matmul",unavailable)
    np.testing.assert_array_equal(actual,function(lhs=a,source=x))
    assert function.native_rhs_packages()==packages


@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_layer_rhs_stable_variance_default_epsilon(dtype):
    storage=storage_dtype(dtype)
    a=np.full((17,35),.01,storage)
    levels=[9992,10000,10008] if dtype=="fp16" else [9920,9984,10048]
    x=np.tile(np.asarray(levels*6+[levels[1]],storage),(35,1))
    assert np.ptp(x.astype(np.float64),axis=-1).min()>0
    np.testing.assert_allclose(layer_rhs_default(a,x),oracle(a,x),rtol=.015,atol=.015)
    constant=np.full((35,19),10000,storage)
    np.testing.assert_array_equal(layer_rhs_default(a,constant),np.zeros((17,19),np.float32))


@pytest.mark.parametrize("attribute,value",[("gamma",2.0),("beta",.5),("axis",0)])
def test_layer_rhs_does_not_drop_affine_axis_semantics(attribute,value):
    a=np.zeros((17,35),np.float16)
    x=np.ones((35,19),np.float16)
    graph=deepcopy(layer_rhs._traced_autodiff_module((a,x),{}))
    graph.functions[0].body[0].kwargs[attribute]=value
    before=graph.to_mlir(verify=False)
    with pytest.raises(ValueError,match="gamma|beta|axis"):
        package_traced_norm_rhs(graph,pipeline_name="tessera-nvidia-pipeline-sm120")
    assert graph.to_mlir(verify=False)==before


def test_layer_rhs_portable_cannot_relabel_normalization(monkeypatch):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    a=np.zeros((17,35),np.float16)
    x=np.ones((35,19),np.float16)
    layer_rhs(a,x)
    artifact=layer_rhs.runtime_artifact()
    bad=deepcopy(artifact.metadata)
    manifest=bad["native_program"]
    manifest["graph_ir"]=manifest["graph_ir"].replace("tessera.layer_norm","tessera.rmsnorm")
    manifest["graph_digest"]=hashlib.sha256(manifest["graph_ir"].encode()).hexdigest()
    body={key:value for key,value in manifest.items() if key!="contract_digest"}
    manifest["contract_digest"]=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    def unavailable(*args,**kwargs):
        raise AssertionError("invalid norm semantics reached CUDA allocation")
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",unavailable)
    receipt=rt.launch(rt.RuntimeArtifact(graph_ir=manifest["graph_ir"],metadata=bad),{"lhs":a,"source":x})
    assert not receipt["ok"] and "kind" in receipt["reason"]


@pytest.mark.parametrize("function,accepted",[(layer_rhs,False)])
def test_legacy_rmsnorm_schema_cannot_admit_layernorm(function,accepted,monkeypatch):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    a=np.zeros((17,35),np.float16)
    x=np.ones((35,19),np.float16)
    function(a,x)
    artifact=function.runtime_artifact()
    metadata=deepcopy(artifact.metadata)
    manifest=metadata["native_program"]
    manifest["schema"]="tessera.nvidia.rmsnorm_rhs_program.v1"
    body={key:value for key,value in manifest.items() if key!="contract_digest"}
    manifest["contract_digest"]=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    def unavailable(*args,**kwargs):
        raise AssertionError("legacy mismatch reached CUDA allocation")
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",unavailable)
    receipt=rt.launch(rt.RuntimeArtifact(graph_ir=artifact.graph_ir,metadata=metadata),{"lhs":a,"source":x})
    assert receipt["ok"] is accepted and "legacy" in receipt["reason"]


def test_legacy_rmsnorm_schema_remains_executable():
    from tests.unit.test_nvidia_rhs_jit import rhs_product
    a=np.full((17,35),.01,np.float16)
    x=np.ones((35,19),np.float16)
    expected=rhs_product(a,x)
    artifact=rhs_product.runtime_artifact()
    metadata=deepcopy(artifact.metadata)
    manifest=metadata["native_program"]
    manifest["schema"]="tessera.nvidia.rmsnorm_rhs_program.v1"
    body={key:value for key,value in manifest.items() if key!="contract_digest"}
    manifest["contract_digest"]=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    receipt=rt.launch(rt.RuntimeArtifact(graph_ir=artifact.graph_ir,metadata=metadata),{"lhs":a,"source":x})
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu"
    np.testing.assert_array_equal(expected,receipt["output"])
