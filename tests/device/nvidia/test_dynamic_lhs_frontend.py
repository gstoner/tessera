"""Bounded frontend tensor programs over checked native Schedule/Tile packages."""
import numpy as np
import pytest
from tessera import runtime as rt
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs,layer_lhs,softmax_lhs,rms_lhs_fused,layer_lhs_fused,softmax_lhs_fused,_storage,_oracle,
)

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning NVIDIA host required")


@pytest.mark.parametrize("fused",[False,True])
@pytest.mark.parametrize("kind",["rmsnorm","layernorm","softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("axes",[("M",),("N",),("K",),("M","K"),("M","N"),("N","K"),("M","N","K")])
def test_dynamic_lhs_portable_reuses_images(kind,dtype,axes,fused,monkeypatch):
    from tessera.compiler import nvidia_tensor_lhs as lhs
    from tessera.compiler import nvidia_native
    function=({"rmsnorm":rms_lhs_fused,"layernorm":layer_lhs_fused,"softmax":softmax_lhs_fused}
              if fused else {"rmsnorm":rms_lhs,"layernorm":layer_lhs,"softmax":softmax_lhs})[kind]
    storage=_storage(dtype)
    m,k,n=33,65,23
    rng=np.random.default_rng(120606)
    x=(rng.normal(size=(m,k))*.2).astype(storage)
    b=np.array(rng.normal(size=(k,n))*.2,dtype=storage,order="F")
    bias=(rng.normal(size=n)*.2).astype(np.float32)
    residual=(rng.normal(size=(m,n))*.2).astype(np.float32)
    args=(x,b,bias,residual) if fused else (x,b)
    graph=function._traced_autodiff_module(args,{})
    before=graph.to_mlir(canonical=True,target="nvidia_sm120")
    program=lhs.package_traced_lhs(graph,dynamic_axes=axes)
    assert graph.to_mlir(canonical=True,target="nvidia_sm120")==before
    program.validate()
    assert "?" in program.graph_ir
    artifact=rt.RuntimeArtifact.from_json(lhs.runtime_artifact(program).to_json())
    images=(program.edge.producer.image.image_digest,program.edge.consumer.image.image_digest)
    def forbidden(*args,**kwargs):
        raise AssertionError("warm dynamic replay attempted compiler work")
    monkeypatch.setattr(lhs,"find_tessera_opt",forbidden)
    monkeypatch.setattr(nvidia_native,"package_scheduled_tensor_matmul",forbidden)
    for row,width,columns in ((m,k,n),(1,1,1),(17,35,19),(m,k,n)):
        am=row if "M" in axes else m
        ak=width if "K" in axes else k
        an=columns if "N" in axes else n
        # Independent padded/sliced host operands are packed by the checked ABI.
        source=x[:am,:ak]*np.asarray(.75,dtype=storage)
        rhs=np.array(b[:ak,:an],order="F")
        active_bias=bias[:an] if fused else None
        active_residual=residual[:am,:an] if fused else None
        args=(source,rhs,active_bias,active_residual) if fused else (source,rhs)
        receipt=rt.launch(artifact,args)
        assert receipt["ok"],receipt
        np.testing.assert_allclose(receipt["output"],_oracle(source,rhs,kind,active_bias,active_residual),rtol=.015,atol=.015)
        assert tuple(r["image_digest"] for r in receipt["component_receipts"])==images
        assert receipt["execution_kind"]=="native_gpu"
        assert receipt["output"].shape==(am,an)


def test_public_dynamic_lhs_compile_method():
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16,order="F")
    program=rms_lhs.compile_native_lhs_matmul(source,rhs,dynamic_axes=("M","K","N"))
    assert program.edge.dynamic_m and program.edge.dynamic_k and program.edge.dynamic_n
    result=program.execute_resident(source[:3,:11],rhs[:11,:7])
    try:
        np.testing.assert_allclose(result.device_session.download(result.output),np.full((3,7),11/np.sqrt(1+1e-5)),rtol=.002)
    finally:
        result.close()


@pytest.mark.parametrize("axis",["M","N","K"])
@pytest.mark.parametrize("extent",[0,34])
def test_dynamic_capacity_guard_precedes_device_allocation(axis,extent,monkeypatch):
    from tessera.compiler import nvidia_tensor_lhs as lhs
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    source=np.ones((33,33),np.float16)
    rhs=np.ones((33,33),np.float16,order="F")
    program=rms_lhs.compile_native_lhs_matmul(source,rhs,dynamic_axes=("M","N","K"))
    artifact=lhs.runtime_artifact(program)
    m,k,n=(extent if axis=="M" else 33,extent if axis=="K" else 33,extent if axis=="N" else 33)
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid capacity reached device allocation")
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",forbidden)
    with pytest.raises(ValueError,match="bound"):
        rt._execute_nvidia_lhs_program_artifact(artifact,(
            np.ones((m,k),np.float16),np.ones((k,n),np.float16,order="F")))


def test_dynamic_parent_graph_rejected_before_cuda(monkeypatch):
    from dataclasses import replace
    from tessera.compiler import nvidia_tensor_lhs as lhs
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16,order="F")
    program=rms_lhs.compile_native_lhs_matmul(source,rhs,dynamic_axes=("M","N","K"))
    artifact=replace(lhs.runtime_artifact(program),graph_ir="module {}")
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid Graph reached CUDA")
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",forbidden)
    with pytest.raises(rt.ArtifactContractError,match="Graph"):
        rt._execute_nvidia_lhs_program_artifact(artifact,(source,rhs))


def test_dynamic_permuted_fused_argument_mapping():
    from tessera.compiler.nvidia_tensor_lhs import runtime_artifact
    from tests.device.nvidia.test_lhs_tensor_jit import layer_lhs_reordered
    rng=np.random.default_rng(120607)
    source=(rng.normal(size=(33,65))*.2).astype(np.float16)
    rhs=np.array(rng.normal(size=(65,23))*.2,dtype=np.float16,order="F")
    bias=(rng.normal(size=23)*.2).astype(np.float32)
    residual=(rng.normal(size=(33,23))*.2).astype(np.float32)
    program=layer_lhs_reordered.compile_native_lhs_matmul(
        residual,rhs,source,bias,dynamic_axes=("M","N","K"))
    artifact=rt.RuntimeArtifact.from_json(runtime_artifact(program).to_json())
    args={"source":source[:17,:35],"rhs":rhs[:35,:19],"bias":bias[:19],"residual":residual[:17,:19]}
    receipt=rt.launch(artifact,args)
    assert receipt["ok"],receipt
    np.testing.assert_allclose(receipt["output"],_oracle(
        args["source"],args["rhs"],"layernorm",args["bias"],args["residual"]),rtol=.015,atol=.015)


@pytest.mark.parametrize("fused",[False,True])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_dynamic_fresh_process_replay_without_compiler(tmp_path,fused,dtype):
    import os
    import subprocess
    import sys
    from tessera.compiler.nvidia_tensor_lhs import runtime_artifact
    function=layer_lhs_fused if fused else layer_lhs
    source=np.ones((17,35),_storage(dtype))
    rhs=np.ones((35,19),_storage(dtype),order="F")
    bias=np.ones(19,np.float32)
    residual=np.ones((17,19),np.float32)
    args=(source,rhs,bias,residual) if fused else (source,rhs)
    program=function.compile_native_lhs_matmul(*args,dynamic_axes=("M","N","K"))
    path=tmp_path/"program.json"
    path.write_text(runtime_artifact(program).to_json())
    code="""
import sys
import numpy as np
from pathlib import Path
from tessera import runtime as rt
from tests.device.nvidia.test_lhs_tensor_jit import _storage
from tessera.compiler import nvidia_tensor_lhs as lhs,nvidia_native
def forbidden(*args,**kwargs):raise AssertionError("fresh replay attempted compilation")
lhs.find_tessera_opt=forbidden
nvidia_native.package_scheduled_tensor_matmul=forbidden
artifact=rt.RuntimeArtifact.from_json(Path(sys.argv[1]).read_text())
source=np.ones((3,11),_storage(sys.argv[2]))
rhs=np.ones((11,7),_storage(sys.argv[2]),order="F")
fused=sys.argv[3]=="True"
args=(source,rhs,np.ones(7,np.float32),np.ones((3,7),np.float32)) if fused else (source,rhs)
receipt=rt.launch(artifact,args)
assert receipt["ok"],receipt
np.testing.assert_array_equal(receipt["output"],np.full((3,7),2 if fused else 0))
print("fresh bounded native replay passed")
"""
    env={**os.environ,"TESSERA_OPT":"/missing/tessera-opt","TESSERA_NVIDIA_OPT":"/missing/tessera-nvidia-opt"}
    completed=subprocess.run([sys.executable,"-c",code,str(path),dtype,str(fused)],
                             env=env,text=True,capture_output=True,timeout=60)
    assert completed.returncode==0,completed.stdout+completed.stderr
