"""Owning SM120 cooperative softmax packages and native producer chains."""
import math
from dataclasses import replace
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import nvidia_native
from tessera.compiler.native_artifact import LaunchGeometry
from tessera.compiler.scheduled_kernel import lower_scheduled_kernel
from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs, project_rhs_storage
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests._support.nvidia import nvidia_cuda_host_ready
from benchmarks.nvidia.benchmark_public_softmax_alias import ordinary
from benchmarks.nvidia.benchmark_native_producer_chain import chain, three_chain
from tests.device.nvidia.test_lhs_tensor_jit import softmax_lhs

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.skipif(
    not nvidia_cuda_host_ready() or not nvidia_native.tools_available(),
    reason="owning SM120 GPU, matching compiler and native runtime required")]

def storage(dtype):
    return np.float16 if dtype == "fp16" else np.float32 if dtype == "fp32" else pytest.importorskip("ml_dtypes").bfloat16

def oracle(x):
    with np.errstate(invalid="ignore"):
        value = x.astype(np.float64)
        exp = np.exp(value-value.max(axis=-1,keepdims=True))
        return exp/exp.sum(axis=-1,keepdims=True)

@pytest.mark.parametrize("dtype", ["fp16","bf16","fp32"])
@pytest.mark.parametrize("shape", [(3,1),(3,17),(2,3,257),(3,4096),(3,4097)])
def test_cooperative_softmax_host_resident_and_nonfinite(dtype,shape):
    dtype_storage=storage(dtype)
    source=np.random.default_rng(5070128).uniform(-20,20,shape).astype(dtype_storage)
    source.reshape(-1,shape[-1])[0]=dtype_storage(1000)
    graph=ordinary._traced_autodiff_module((source,),{})
    before=graph.to_mlir()
    scheduled=lower_scheduled_kernel(graph,target="nvidia_sm120",schedule="cooperative_128")
    assert graph.to_mlir()==before
    package=nvidia_native.package_scheduled_kernel(scheduled,pipeline_name="tessera-nvidia-pipeline-sm120")
    assert package.descriptor.entry_symbol.endswith("_cooperative_128")
    assert package.descriptor.geometry==LaunchGeometry(policy="sm120_softmax_cooperative_128_rows")
    artifact=rt.RuntimeArtifact(metadata={"target":"nvidia_sm120"},native_image=package.image,
        launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
    bindings=sorted(package.descriptor.buffers,key=lambda b:b.ordinal)
    scalars={"Rows":math.prod(shape[:-1]),"K":shape[-1]}
    probes=[source]
    if shape[-1]>1:
        for value in (np.nan,np.inf,-np.inf):
            probe=source.copy();probe.reshape(-1,shape[-1])[1]=value;probes.append(probe)
    for probe in probes:
        expected=oracle(probe)
        output=np.empty_like(source)
        result=rt.launch(artifact,{bindings[0].name:probe,bindings[1].name:output,**scalars})
        assert result.get("ok"),result
        tolerance=dict(rtol=.01 if dtype!="fp32" else 3e-5,atol=2e-4 if dtype!="fp32" else 2e-6,equal_nan=True)
        np.testing.assert_allclose(output.astype(np.float64),expected,**tolerance)
        session=NvidiaDeviceSession()
        try:
            resident={bindings[0].name:session.upload(probe),bindings[1].name:session.empty(shape,dtype_storage),**scalars}
            result=rt.launch(artifact,resident,stream=session.stream)
            assert result.get("ok"),result
            np.testing.assert_allclose(session.download(resident[bindings[1].name]).astype(np.float64),expected,**tolerance)
        finally:session.close()

@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("shape",[(17,35,19),(128,4096,64)])
@pytest.mark.parametrize("count",[1,2,3])
def test_cooperative_softmax_native_chain_lifetime(dtype,shape,count):
    m,k,n=shape;s=storage(dtype)
    rng=np.random.default_rng(5070128);x=rng.normal(0,.2,(m,k)).astype(s)
    rhs=np.array(rng.normal(0,.2,(k,n)),dtype=s,order="F")
    function={1:softmax_lhs,2:chain,3:three_chain}[count]
    graph=project_rhs_storage(function._traced_autodiff_module((x,rhs),{}),[x,rhs])
    before=graph.to_mlir()
    program=package_traced_lhs(graph,softmax_schedule="cooperative_128")
    assert graph.to_mlir()==before
    producers=program.producer_chain or (program.edge.producer,)
    assert producers[-1].descriptor.provenance["schedule"]=="cooperative_128"
    def expected(source):
        value=source.astype(np.float64)
        if count==3:
            centered=value-value.mean(axis=-1,keepdims=True)
            value=(centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)).astype(s).astype(np.float64)
        if count>=2:
            value=(value/np.sqrt(np.mean(value*value,axis=-1,keepdims=True)+1e-5)).astype(s).astype(np.float64)
        return oracle(value).astype(s).astype(np.float64)@rhs.astype(np.float64)
    owner=PreparedLhsCall(program)
    try:
        first,receipt=owner([x,rhs])
        np.testing.assert_allclose(first,expected(x),rtol=.015,atol=.002)
        saved=first.copy()
        second,second_receipt=owner([-x,rhs])
        np.testing.assert_allclose(second,expected(-x),rtol=.015,atol=.002)
        np.testing.assert_array_equal(first,saved)
        assert receipt["execution_kind"]==second_receipt["execution_kind"]=="native_gpu"
    finally:owner.close()

@pytest.mark.parametrize("field",["geometry","workgroup","schedule","storage"])
def test_cooperative_softmax_refuses_forged_runtime_policy(field,monkeypatch):
    x=np.ones((3,17),np.float32)
    graph=ordinary._traced_autodiff_module((x,),{})
    scheduled=lower_scheduled_kernel(graph,target="nvidia_sm120",schedule="cooperative_128")
    package=nvidia_native.package_scheduled_kernel(scheduled,pipeline_name="tessera-nvidia-pipeline-sm120")
    d=package.descriptor
    changes={"geometry":dict(geometry=LaunchGeometry(policy="sm120_softmax_thread_per_row_128")),
        "workgroup":dict(geometry=LaunchGeometry(grid=(3,1,1),workgroup=(32,1,1))),
        "schedule":dict(provenance={**d.provenance,"schedule":"serial"}),
        "storage":dict(provenance={**d.provenance,"storage":"f16"})}
    def forbidden(*args,**kwargs):
        raise AssertionError("forged descriptor reached CUDA registration")
    monkeypatch.setattr(rt,"_register_nvidia_ptx",forbidden)
    bindings=sorted(d.buffers,key=lambda b:b.ordinal)
    with pytest.raises(RuntimeError,match="schedule/entry/geometry ABI mismatch"):
        rt._submit_nvidia_sm120_native(package.image,replace(d,**changes[field]),
            {bindings[0].name:x,bindings[1].name:np.empty_like(x)},{"Rows":3,"K":17},None)
