"""Native normalization selection and ordinary frontend package proof."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import scheduled_kernel
from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs,runtime_artifact
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_cooperative_norm import graph,oracle,launch
from tests.device.nvidia.test_lhs_tensor_jit import rms_lhs,layer_lhs,rms_lhs_fused,layer_lhs_fused,_storage,_oracle

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning SM120 host required")


@pytest.mark.parametrize("kind",["rmsnorm","layernorm"])
@pytest.mark.parametrize("dtype",["fp16","bf16","fp32"])
@pytest.mark.parametrize("columns,expected",[(255,"serial"),(256,"cooperative_128"),(257,"cooperative_128")])
def test_native_selector_boundaries_and_explicit_override(kind,dtype,columns,expected):
    from tessera.compiler import nvidia_native
    storage=np.float32 if dtype=="fp32" else _storage(dtype)
    source=(np.random.default_rng(256).normal(size=(3,columns))*.2).astype(storage)
    module=graph(kind,source)
    before=module.to_mlir(canonical=True,target="nvidia_sm120")
    artifact=scheduled_kernel.lower_scheduled_kernel(module,target="nvidia_sm120")
    assert artifact.schedule==expected
    native=nvidia_native.package_scheduled_kernel(artifact,pipeline_name="tessera-nvidia-pipeline-sm120")
    np.testing.assert_allclose(launch(native,source).astype(np.float32),
        oracle(source,kind).astype(np.float32),rtol=.015,atol=.015)
    assert module.to_mlir(canonical=True,target="nvidia_sm120")==before
    forced=scheduled_kernel.lower_scheduled_kernel(module,target="nvidia_sm120",schedule="serial")
    assert forced.schedule=="serial"
    assert 'schedule = "serial"' in forced.graph_ir
    forced_native=nvidia_native.package_scheduled_kernel(forced,pipeline_name="tessera-nvidia-pipeline-sm120")
    np.testing.assert_allclose(launch(forced_native,source).astype(np.float32),
        oracle(source,kind).astype(np.float32),rtol=.015,atol=.015)


def test_python_reads_native_decision_instead_of_choosing_by_shape(monkeypatch):
    source=np.ones((3,1024),np.float16)
    original=scheduled_kernel.run_tessera_opt
    def compiler_selects_serial(tool,text,flags):
        if flags=="--tessera-graph-to-schedule":
            text=text.replace("{eps =",'{schedule = "serial", eps =')
        return original(tool,text,flags)
    monkeypatch.setattr(scheduled_kernel,"run_tessera_opt",compiler_selects_serial)
    artifact=scheduled_kernel.lower_scheduled_kernel(graph("rmsnorm",source),target="nvidia_sm120")
    assert artifact.columns==1024 and artifact.schedule=="serial"


@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("kind",["rmsnorm","layernorm"])
@pytest.mark.parametrize("fused",[False,True])
def test_selected_lhs_program_ordinary_cached_portable_and_serial_pair(dtype,kind,fused,monkeypatch):
    source=(np.random.default_rng(512).normal(size=(17,1024))*.2).astype(_storage(dtype))
    rhs=(np.random.default_rng(513).normal(size=(1024,19))*.2).astype(_storage(dtype))
    bias=np.full(19,.01,np.float32)
    residual=np.full((17,19),.02,np.float32)
    function=({"rmsnorm":rms_lhs_fused,"layernorm":layer_lhs_fused} if fused else
              {"rmsnorm":rms_lhs,"layernorm":layer_lhs})[kind]
    args=(source,rhs,bias,residual) if fused else (source,rhs)
    expected=_oracle(source,rhs,kind,bias if fused else None,residual if fused else None)
    selected=function(*args)
    np.testing.assert_allclose(selected,expected,rtol=.015,atol=.015)
    packages=function.native_lhs_packages()
    assert packages[0].descriptor.provenance["schedule"]=="cooperative_128"
    module=deepcopy(function._traced_autodiff_module(args,{}))
    baseline=package_traced_lhs(module,producer_schedule="serial")
    assert baseline.edge.producer.descriptor.provenance["schedule"]=="serial"
    assert baseline.edge.consumer.image.image_digest==packages[1].image.image_digest
    for artifact in (runtime_artifact(baseline),rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())):
        receipt=rt.launch(artifact,args)
        assert receipt["ok"],receipt
        np.testing.assert_allclose(receipt["output"],expected,rtol=.015,atol=.015)
    def unavailable(*args,**kwargs):raise AssertionError("cached call recompiled or ran eager")
    monkeypatch.setattr(function,"compile_native_lhs_matmul",unavailable)
    monkeypatch.setattr(function,"_fn",unavailable)
    np.testing.assert_array_equal(selected,function(*args))
    assert function.native_lhs_packages()==packages


def test_explicit_physical_policy_cannot_be_ignored():
    from tests.device.nvidia.test_lhs_tensor_jit import softmax_lhs
    args=(np.ones((3,257),np.float16),np.ones((257,19),np.float16))
    with pytest.raises(ValueError,match="normalization or reduction"):
        package_traced_lhs(softmax_lhs._traced_autodiff_module(args,{}),producer_schedule="cooperative_128")
