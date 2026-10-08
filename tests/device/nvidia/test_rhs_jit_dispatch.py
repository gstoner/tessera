"""Exact SM120 ordinary JIT calls must execute both native RHS packages."""
import numpy as np
import pytest
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.unit.test_nvidia_rhs_jit import rhs_product,rhs_reordered

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning NVIDIA host required")


@pytest.mark.parametrize("function",[rhs_product,rhs_reordered])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("shape",[(16,16,8),(17,35,19),(64,256,64)])
def test_ordinary_jit_rhs_call_native_and_cached(function,dtype,shape,monkeypatch):
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    m,k,n=shape
    rng=np.random.default_rng(120413)
    a=(rng.normal(size=(m,k))*.2).astype(storage)
    x=(rng.normal(size=(k,n))*.2).astype(storage)
    args=(a,x) if function is rhs_product else (x,a)
    actual=function(*args)
    xf=x.astype(np.float64)
    norm=(xf/np.sqrt(np.mean(xf*xf,axis=1,keepdims=True)+1e-5)).astype(storage)
    expected=a.astype(np.float64)@norm.astype(np.float64)
    np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)
    assert function.execution_kind=="native_gpu"
    assert all(r["execution_kind"]=="native_gpu" and r["ok"] for r in function._nvidia_rhs_last_receipts)
    packages=function.native_rhs_packages()
    assert len(packages)==2
    artifact=function.runtime_artifact()
    assert artifact.metadata["compiler_path"]=="canonical_nvidia_rhs_program"
    assert artifact.metadata["executable"] is True
    assert [artifact.metadata["native_program"][role]["native_image"]["image_digest"]
            for role in ("producer","consumer")]==[p.image.image_digest for p in packages]
    # Tracing has already captured this signature. Subsequent native calls
    # must use its cache even if eager execution and recompilation are unavailable.
    def unavailable(*args,**kwargs):
        raise AssertionError("eager body or recompilation was invoked")
    monkeypatch.setattr(function,"_fn",unavailable)
    monkeypatch.setattr(function,"compile_native_rhs_matmul",unavailable)
    again=function(*args)
    np.testing.assert_array_equal(actual,again)
    assert function.native_rhs_packages()==packages


def test_failed_native_rhs_launch_does_not_fall_back(monkeypatch):
    a=np.zeros((17,35),np.float16)
    x=np.ones((35,19),np.float16)
    rhs_product(a,x)
    compiled=rhs_product._nvidia_rhs_last_program
    def failed(*args,**kwargs):
        raise RuntimeError("injected native launch failure")
    monkeypatch.setattr(type(compiled),"execute_resident",failed)
    with pytest.raises(RuntimeError,match="native launch failure"):
        rhs_product(a,x)
    assert rhs_product.native_rhs_packages()==()
    assert rhs_product.runtime_artifact().metadata.get("compiler_path")!="canonical_nvidia_rhs_program"


def test_ordinary_rhs_jit_checks_keyword_binding():
    a=np.zeros((17,35),np.float16)
    x=np.ones((35,19),np.float16)
    np.testing.assert_array_equal(rhs_product(lhs=a,source=x),np.zeros((17,19),np.float32))
    with pytest.raises(TypeError):
        rhs_product(a,x,source=x)
    with pytest.raises(TypeError):
        rhs_product(lhs=a,source=x,unknown=x)
    assert rhs_product.native_rhs_packages()==()


@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_rhs_normalization_uses_f32_intermediates(dtype):
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    a=np.full((17,35),.01,storage)
    x=np.full((35,19),10000,storage)
    expected=a.astype(np.float32)@np.ones((35,19),np.float32)
    np.testing.assert_allclose(rhs_product(a,x),expected,rtol=4e-5,atol=4e-5)
    assert rhs_product.execution_kind=="native_gpu"
    report=rhs_product.compile_report()
    assert report.target_decision["nvidia_sm120"].startswith("canonical_nvidia_rhs_program")
    assert set(report.ir_hashes)=={"graph_ir"}
    assert rhs_product.runtime_artifact().metadata["native_program"]["producer"]["native_image"]["image_digest"]==rhs_product.native_rhs_packages()[0].image.image_digest


@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_portable_rhs_program_roundtrip_and_owned_stream(dtype):
    import json
    from copy import deepcopy
    from tessera import runtime as rt
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    rng=np.random.default_rng(120415)
    a=(rng.normal(size=(17,35))*.2).astype(storage)
    x=(rng.normal(size=(35,19))*.2).astype(storage)
    expected=rhs_reordered(x,a)
    artifact=rt.RuntimeArtifact.from_json(rhs_reordered.runtime_artifact().to_json())
    receipt=rt.launch(artifact,{"lhs":a,"source":x})
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu"
    assert len(receipt["component_receipts"])==2
    np.testing.assert_array_equal(receipt["output"],expected)
    refused=rt.launch(artifact,{"lhs":a,"source":x},stream=123)
    assert not refused["ok"] and "owns" in refused["reason"]
    # Rehashed metadata must still satisfy the parent Graph's operand lineage.
    bad=deepcopy(artifact.metadata)
    manifest=bad["native_program"]
    manifest["source_index"],manifest["lhs_index"]=manifest["lhs_index"],manifest["source_index"]
    import hashlib
    body={key:value for key,value in manifest.items() if key!="contract_digest"}
    manifest["contract_digest"]=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    invalid=rt.RuntimeArtifact(graph_ir=artifact.graph_ir,metadata=bad)
    refused=rt.launch(invalid,{"lhs":a,"source":x})
    assert not refused["ok"] and "lineage" in refused["reason"]


def test_portable_rhs_rejects_changed_parent_and_component_before_cuda(monkeypatch):
    from copy import deepcopy
    import hashlib
    import json
    from tessera import runtime as rt
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    a=np.zeros((17,35),np.float16)
    x=np.ones((35,19),np.float16)
    rhs_product(a,x)
    artifact=rhs_product.runtime_artifact()
    def unavailable(*args,**kwargs):
        raise AssertionError("invalid contract reached CUDA allocation")
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",unavailable)
    changed=rt.RuntimeArtifact(graph_ir=artifact.graph_ir+"\n// changed",metadata=artifact.metadata)
    assert not rt.launch(changed,{"lhs":a,"source":x})["ok"]
    bad=deepcopy(artifact.metadata)
    manifest=bad["native_program"]
    manifest["producer"]["artifact_hash"]="0"*64
    body={key:value for key,value in manifest.items() if key!="contract_digest"}
    manifest["contract_digest"]=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    refused=rt.launch(rt.RuntimeArtifact(graph_ir=artifact.graph_ir,metadata=bad),{"lhs":a,"source":x})
    assert not refused["ok"] and "hash" in refused["reason"]
