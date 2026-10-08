"""Exact SM120 native serial/cooperative normalization parity."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import nvidia_native, scheduled_kernel
from tessera.compiler.graph_ir import GraphIRModule
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import rms_lhs,layer_lhs,_storage

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning SM120 host required")


def graph(kind,source):
    function=layer_lhs if kind=="layernorm" else rms_lhs
    rhs=np.ones((source.shape[-1],1),source.dtype)
    module=deepcopy(function._traced_autodiff_module((source,rhs),{}))
    fn=module.functions[0]
    producer=fn.body[0]
    fn=replace(fn,name="cooperative_norm",args=[fn.args[0]],body=[producer],
        result_types=[fn.args[0].ir_type],return_values=["%"+producer.result],
        structured_cfg=None,fn_attrs={})
    return GraphIRModule([fn])


def oracle(source,kind):
    x=source.astype(np.float64)
    if kind=="layernorm":x=x-np.mean(x,axis=-1,keepdims=True)
    return (x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)).astype(source.dtype)


def package(kind,source,strategy):
    artifact=scheduled_kernel.lower_scheduled_kernel(graph(kind,source),target="nvidia_sm120",schedule=strategy)
    native=nvidia_native.package_scheduled_kernel(artifact,pipeline_name="tessera-nvidia-pipeline-sm120")
    return artifact,native


def launch(native,source):
    output=np.empty_like(source)
    d=native.descriptor
    values={d.buffers[0].name:source,d.buffers[1].name:output,
            "Rows":source.shape[0],"Columns":source.shape[1]}
    artifact=rt.RuntimeArtifact(metadata={"target":"nvidia_sm120"},
        tile_ir=native.tile_ir,target_ir=native.target_ir,
        native_image=native.image,launch_descriptor=d)
    receipt=rt.launch(artifact,values)
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
    return output


@pytest.mark.parametrize("kind",["rmsnorm","layernorm"])
@pytest.mark.parametrize("dtype",["fp16","bf16","fp32"])
@pytest.mark.parametrize("shape",[(1,1),(3,17),(17,35),(129,257),(128,1024),(2,4097)])
def test_cooperative_norm_parity(kind,dtype,shape):
    storage=np.float32 if dtype=="fp32" else _storage(dtype)
    source=(np.random.default_rng(128).normal(size=shape)*.2).astype(storage)
    expected=oracle(source,kind)
    packages=[]
    for strategy in ("serial","cooperative_128"):
        artifact,native=package(kind,source,strategy)
        assert artifact.schedule==strategy
        assert native.descriptor.provenance["schedule"]==strategy
        actual=launch(native,source)
        np.testing.assert_allclose(actual.astype(np.float32),expected.astype(np.float32),
                                   rtol=.015 if dtype=="bf16" else .003,
                                   atol=.015 if dtype=="bf16" else .003)
        packages.append(native)
    assert packages[0].image.image_digest!=packages[1].image.image_digest
    assert "nvvm.barrier" in packages[1].target_ir
    assert "cooperative_128" in packages[1].descriptor.entry_symbol


@pytest.mark.parametrize("dtype",["fp16","bf16","fp32"])
def test_cooperative_layernorm_centered_variance(dtype):
    storage=np.float32 if dtype=="fp32" else _storage(dtype)
    levels=[9992,10000,10008] if dtype!="bf16" else [9920,9984,10048]
    source=np.tile(np.asarray(levels*85+[10000],storage),(3,1))
    _,native=package("layernorm",source,"cooperative_128")
    np.testing.assert_allclose(launch(native,source).astype(np.float32),
        oracle(source,"layernorm").astype(np.float32),rtol=.015,atol=.015)
    constant=np.full(source.shape,10000,storage)
    np.testing.assert_array_equal(launch(native,constant),np.zeros(source.shape,storage))


def test_cooperative_schedule_hash_and_policy_are_checked():
    source=np.ones((3,257),np.float16)
    artifact,_=package("rmsnorm",source,"cooperative_128")
    bad=replace(artifact,schedule="serial")
    with pytest.raises(ValueError,match="disagrees"):
        nvidia_native.package_scheduled_kernel(bad,pipeline_name="tessera-nvidia-pipeline-sm120")
    bad=replace(artifact,schedule_ir=artifact.schedule_ir.replace('schedule = "cooperative_128"','schedule = "serial"'))
    with pytest.raises(ValueError):
        nvidia_native.package_scheduled_kernel(bad,pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.parametrize("arch",["gfx1201","gfx1151","apple7","zen5-avx512"])
def test_cooperative_tile_requires_owning_architecture(arch):
    from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
    source=np.ones((3,257),np.float16)
    artifact,_=package("rmsnorm",source,"cooperative_128")
    bad=artifact.tile_ir.replace('tessera.arch = "sm_120"',f'tessera.arch = "{arch}"')
    with pytest.raises(RuntimeError,match="cooperative norm requires"):
        run_tessera_opt(find_tessera_opt(),bad,"--verify-each")


def test_cooperative_tile_requires_typed_schedule():
    from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
    artifact,_=package("rmsnorm",np.ones((3,257),np.float16),"cooperative_128")
    bad=artifact.tile_ir.replace('schedule = "cooperative_128"','schedule = 128 : i64')
    with pytest.raises(RuntimeError,match="schedule must be a string"):
        run_tessera_opt(find_tessera_opt(),bad,"--verify-each")
