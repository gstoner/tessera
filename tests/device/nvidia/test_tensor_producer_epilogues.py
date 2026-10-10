"""Exact SM120 tensor producer and complete-epilogue execution proofs."""
import numpy as np
import pytest
from tessera.compiler import nvidia_native, scheduled_matmul
from tessera import runtime as rt
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.unit.test_nvidia_tensor_program import _program
from tests.unit.test_sm120_legacy_scheduled_producer import legacy_artifact, canonical_loop_artifact
from tests.unit.test_scheduled_matmul_consumers import _module, _dynamic_module

pytestmark = pytest.mark.skipif(
    scheduled_matmul.find_tessera_opt() is None or not nvidia_native.tools_available(),
    reason="requires the production SM120 Schedule/Target compiler tools",
)

@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype", ["fp16","bf16"])
@pytest.mark.parametrize("output_dtype", ["fp16","fp32"])
@pytest.mark.parametrize("dynamic_m", [False,True])
@pytest.mark.parametrize("epilogue", [(False,False,"relu"),(True,False,"none"),
                                     (False,True,"relu"),(True,True,"relu")])
def test_resident_tensor_edge_executes_complete_epilogue(dtype,output_dtype,dynamic_m,epilogue):
    from tests._support.nvidia import nvidia_cuda_host_ready
    if not nvidia_cuda_host_ready():
        pytest.skip("requires the owning SM120 host")
    bias_on,residual_on,activation = epilogue
    program = _program(dtype=dtype,output_dtype=output_dtype,dynamic_m=dynamic_m,
                       bias=bias_on,residual=residual_on,activation=activation)
    if dtype == "bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    else:
        storage=np.float16
    rng=np.random.default_rng(120_404)
    for m in ((16,7) if dynamic_m else (16,)):
        source=(rng.normal(size=(m,program.k))*.2).astype(storage)
        rhs=np.asfortranarray((rng.normal(size=(program.k,program.n))*.2).astype(storage))
        bias=(rng.normal(size=(program.n,))*.2).astype(np.float32) if bias_on else None
        residual=(rng.normal(size=(m,program.n))*.2).astype(np.float32) if residual_on else None
        with program.execute_resident(source,rhs,bias=bias,residual=residual) as result:
            edge=result.device_session.download(result.intermediate)[:m,:program.k]
            actual=result.device_session.download(result.output)
            x=source.astype(np.float32)
            normalized=(x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-5)).astype(storage)
            np.testing.assert_allclose(edge.astype(np.float32),normalized.astype(np.float32),
                                       rtol=.012,atol=.012)
            expected=edge.astype(np.float32)@rhs.astype(np.float32)
            if bias_on: expected += bias
            if activation == "relu": expected=np.maximum(expected,0)
            if residual_on: expected += residual
            if output_dtype == "fp16": expected=expected.astype(np.float16)
            np.testing.assert_allclose(actual,expected,rtol=.002,atol=.002)
            assert result.consumer_receipt["execution_kind"] == "native_gpu"
            assert result.producer_receipt["execution_kind"] == "native_gpu"
        assert result.device_session.stream == 0
        assert result.device_session._buffers == []


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("output_dtype",["fp16","fp32"])
def test_host_tensor_edge_uses_the_complete_epilogue_contract(output_dtype):
    from tests._support.nvidia import nvidia_cuda_host_ready
    if not nvidia_cuda_host_ready():
        pytest.skip("requires owning SM120 host")
    program=_program(output_dtype=output_dtype,bias=True,residual=True,activation="relu")
    rng=np.random.default_rng(120406)
    source=(rng.normal(size=(16,16))*.2).astype(np.float16)
    rhs=np.asfortranarray((rng.normal(size=(16,8))*.2).astype(np.float16))
    bias=(rng.normal(size=8)*.2).astype(np.float32)
    residual=(rng.normal(size=(16,8))*.2).astype(np.float32)
    result=program.execute(source,rhs,bias=bias,residual=residual)
    expected=np.maximum(result.intermediate.astype(np.float32)@rhs.astype(np.float32)+bias,0)+residual
    if output_dtype=="fp16":expected=expected.astype(np.float16)
    np.testing.assert_allclose(result.output,expected,rtol=.002,atol=.002)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("shape", [(17,19,23), (48,67,17), (64,256,64)])
@pytest.mark.parametrize("via_tiling", [False, True])
def test_legacy_sm120_producer_executes_native_schedule_package(shape,dtype,via_tiling):
    if not nvidia_cuda_host_ready():
        pytest.skip("requires exact host SM120 GPU and toolchain")
    artifact = legacy_artifact(shape,dtype,via_tiling=via_tiling)
    package = nvidia_native.package_scheduled_matmul(
        artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
    m,k,n = shape
    rng=np.random.default_rng(120_301)
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    else:
        storage=np.float16
    a=(rng.normal(size=(m,k))*.2).astype(storage)
    b=np.asfortranarray((rng.normal(size=(k,n))*.2).astype(storage))
    output=np.empty((m,n),np.float32)
    args={artifact.a_name:a, artifact.b_name:b, artifact.output_name:output,
          "M":m,"N":n,"K":k}
    runtime = rt.RuntimeArtifact(
        metadata={"target": package.image.target}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
        target_ir=package.target_ir)
    result=rt.launch(runtime, args)
    assert result["ok"], result.get("reason", result)
    assert result["execution_kind"]=="native_gpu"
    np.testing.assert_allclose(output,a.astype(np.float32)@b.astype(np.float32),
                               rtol=4e-5,atol=4e-5)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("via_tiling", [False, True])
def test_legacy_dynamic_fused_package_reuses_checked_runtime_shapes(dtype,via_tiling):
    if not nvidia_cuda_host_ready():
        pytest.skip("requires exact host SM120 GPU and toolchain")
    projected = legacy_artifact(module=_dynamic_module(
        dtype=dtype, bias=True, residual=True, activation="relu"),via_tiling=via_tiling)
    package = nvidia_native.package_scheduled_matmul(
        projected, pipeline_name="tessera-nvidia-pipeline-sm120")
    runtime = rt.RuntimeArtifact(metadata={"target":package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir)
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    else:
        storage=np.float16
    rng=np.random.default_rng(120_302)
    for m,k,n in ((17,19,13),(9,5,7)):
        lda,ldb,ldd=k+3,k+5,n+7
        a_storage=np.zeros((m,lda),storage)
        a=a_storage[:,:k]
        a[:]=(rng.normal(size=(m,k))*.2).astype(storage)
        b_storage=np.zeros((ldb,n),storage,order="F")
        b=b_storage[:k,:]
        b[:]=(rng.normal(size=(k,n))*.2).astype(storage)
        bias=(rng.normal(size=(n,))*.1).astype(np.float32)
        residual_storage=np.zeros((m,ldd),np.float32)
        residual=residual_storage[:,:n]
        residual[:]=(rng.normal(size=(m,n))*.05).astype(np.float32)
        output_storage=np.full((m,ldd),-123.,np.float32)
        output=output_storage[:,:n]
        result=rt.launch(runtime, {"a":a,"b":b,"bias":bias,"residual":residual,"o":output,
                                  "M":m,"N":n,"K":k,"LDA":lda,"LDB":ldb,"LDD":ldd})
        assert result["ok"], result.get("reason",result)
        assert result["execution_kind"]=="native_gpu"
        expected=np.maximum(a.astype(np.float32)@b.astype(np.float32)+bias,0)+residual
        np.testing.assert_allclose(output,expected,rtol=4e-5,atol=4e-5)

        np.testing.assert_array_equal(output_storage[:,n:],-123.)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("fused",[False,True])
def test_sm120_canonical_tensor_loop_package_executes(dtype,fused):
    if not nvidia_cuda_host_ready():
        pytest.skip("requires exact host SM120 GPU and toolchain")
    shape=(17,35,19)
    artifact=canonical_loop_artifact(_module(target="nvidia_sm120",dtype=dtype,
        shape=shape,bias=fused,residual=fused,activation="relu" if fused else "none"))
    package=nvidia_native.package_scheduled_matmul(artifact,
        pipeline_name="tessera-nvidia-pipeline-sm120")
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    m,k,n=shape
    rng=np.random.default_rng(120_401)
    a=(rng.normal(size=(m,k))*.2).astype(storage)
    b=np.asfortranarray((rng.normal(size=(k,n))*.2).astype(storage))
    output=np.full((m,n),np.nan,np.float32)
    args={artifact.a_name:a,artifact.b_name:b,artifact.output_name:output,
          "M":m,"N":n,"K":k}
    expected=a.astype(np.float32)@b.astype(np.float32)
    if fused:
        bias=(rng.normal(size=n)*.1).astype(np.float32)
        residual=(rng.normal(size=(m,n))*.05).astype(np.float32)
        args[artifact.bias_name]=bias;args[artifact.residual_name]=residual
        expected=np.maximum(expected+bias,0)+residual
    runtime=rt.RuntimeArtifact(metadata={"target":package.image.target},
        native_image=package.image,launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,target_ir=package.target_ir)
    result=rt.launch(runtime,args)
    assert result["ok"],result
    assert result["execution_kind"]=="native_gpu"
    np.testing.assert_allclose(output,expected,rtol=4e-5,atol=4e-5)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("order", [(1,0), (3,1,0,2), (2,3,1,0)])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_permuted_tensor_producer_executes_native_package(order,dtype):
    if not nvidia_cuda_host_ready():
        pytest.skip("requires exact host SM120 GPU and toolchain")
    fused=len(order)==4
    module=_module(target="nvidia_sm120",shape=(17,35,19),dtype=dtype,
        bias=fused,residual=fused,activation="relu" if fused else "none")
    function=module.functions[0]
    function.args=[function.args[i] for i in order]
    artifact=canonical_loop_artifact(module)
    package=nvidia_native.package_scheduled_matmul(
        artifact,pipeline_name="tessera-nvidia-pipeline-sm120")
    import ml_dtypes
    storage=ml_dtypes.bfloat16 if dtype=="bf16" else np.float16
    rng=np.random.default_rng(120_706)
    a=(rng.normal(size=(17,35))*.2).astype(storage)
    b=np.asfortranarray((rng.normal(size=(35,19))*.2).astype(storage))
    out=np.full((17,19),np.nan,np.float32)
    bindings={artifact.a_name:a,artifact.b_name:b,artifact.output_name:out,
              "M":17,"N":19,"K":35}
    want=a.astype(np.float64)@b.astype(np.float64)
    if fused:
        bias=(rng.normal(size=19)*.1).astype(np.float32)
        residual=(rng.normal(size=(17,19))*.05).astype(np.float32)
        bindings[artifact.bias_name]=bias
        bindings[artifact.residual_name]=residual
        want=np.maximum(want+bias.astype(np.float64),0)+residual.astype(np.float64)
    executable=rt.RuntimeArtifact(metadata={"target":package.image.target},
        native_image=package.image,launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,target_ir=package.target_ir)
    receipt=rt.launch(executable,bindings)
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
    np.testing.assert_allclose(out,want,rtol=4e-5,atol=4e-5)
