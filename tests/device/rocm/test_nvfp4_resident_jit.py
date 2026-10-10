"""Exact gfx1201 composed frontend/native/portable checkpoint proof."""
from copy import deepcopy
import os
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_ingest import nvfp4_requantization_policy
from tessera.compiler.rocm_mxfp4_storage import MXFP4_STORAGE_CONTRACT
from tessera.compiler.rocm_nvfp4_program import packed_consumer_attrs,program_from_manifest
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle

pytestmark=pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",
                             reason="matching exact gfx1201 host required")


def make_function(n,k,reordered=False):
    offsets=[0,n//2,n]
    policy=nvfp4_requantization_policy()
    attrs=packed_consumer_attrs(k)
    if reordered:
        @ts.jit(target="rocm_gfx1201")
        def chain(a_scale,a,codes,projection_globals,scales):
            packed,exponents,stats=ts.ops.nvfp4_requantize(codes,scales,projection_globals,
                row_offsets=offsets,numeric_policy=policy)
            fragment,plane=ts.ops.mxfp4_folded_storage(packed,exponents,
                storage_contract=MXFP4_STORAGE_CONTRACT)
            return ts.ops.scaled_matmul(a,fragment,a_scale,plane,
                physical_contract=attrs["physical_contract"],numeric_policy=attrs["numeric_policy"],
                scale_layout=attrs["scale_layout"])
    else:
        @ts.jit(target="rocm_gfx1201")
        def chain(codes,scales,projection_globals,a,a_scale):
            packed,exponents,stats=ts.ops.nvfp4_requantize(codes,scales,projection_globals,
                row_offsets=offsets,numeric_policy=policy)
            fragment,plane=ts.ops.mxfp4_folded_storage(packed,exponents,
                storage_contract=MXFP4_STORAGE_CONTRACT)
            return ts.ops.scaled_matmul(a,fragment,a_scale,plane,
                physical_contract=attrs["physical_contract"],numeric_policy=attrs["numeric_policy"],
                scale_layout=attrs["scale_layout"])
    return chain


@pytest.mark.parametrize("shape",[(128,32,256),(257,80,1024),(256,64,64)])
@pytest.mark.parametrize("reordered",[False,True])
def test_composed_jit_cached_and_portable(shape,reordered,monkeypatch):
    from tessera.compiler import rocm_nvfp4_ingest,rocm_mxfp4_storage,rocm_nvfp4_program,rocm_nvfp4_resident
    m,n,k=shape
    args,offsets,converted,stored,expected=inputs_and_oracle(m,n,k)
    function=make_function(n,k,reordered)
    named=dict(zip(("codes","scales","projection_globals","a","a_scale"),args))
    def forbidden(*a,**kw):
        pytest.fail("compiled checkpoint JIT invoked eager weight arithmetic")
    monkeypatch.setattr(rocm_nvfp4_ingest,"reference_nvfp4_requantize",forbidden)
    monkeypatch.setattr(rocm_mxfp4_storage,"reference_mxfp4_folded_storage",forbidden)
    monkeypatch.setattr(rocm_nvfp4_program,"reference_scaled_matmul",forbidden)
    monkeypatch.setattr(rocm_nvfp4_resident,"ResidentNVFP4Matmul",forbidden)
    actual=function(**named)
    np.testing.assert_allclose(actual.astype(np.float32),expected,rtol=.008,atol=.015625)
    assert function.execution_kind=="native_gpu"
    packages=function.native_nvfp4_packages()
    assert len(packages)==3
    assert all(p.image.architecture=="gfx1201" for p in packages)
    artifact=rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
    receipt=rt.launch(artifact,named)
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu"
    assert len(receipt["component_receipts"])==3
    assert all(r["native_call_binding"]=="native_cpp_nvfp4" for r in receipt["component_receipts"])
    np.testing.assert_array_equal(actual,receipt["output"])
    monkeypatch.setattr(function,"_fn",forbidden)
    monkeypatch.setattr(function,"compile_native_nvfp4_program",forbidden)
    monkeypatch.setattr(rocm_nvfp4_program,"runtime_artifact",forbidden)
    np.testing.assert_array_equal(function(**named),actual)
    assert function.native_nvfp4_packages()==packages


def test_malformed_frontend_manifest_refused_before_hip(monkeypatch):
    from tessera.compiler.rocm_nvfp4_resident import _program_digest
    args,*_=inputs_and_oracle(128,32,256)
    function=make_function(32,256)
    function(*args)
    artifact=function.runtime_artifact()
    original=artifact.metadata["native_program"]
    monkeypatch.setattr(rt,"_load_hip_for_launch",lambda:pytest.fail("bad program reached HIP"))
    for key,value in [
        ("role_indices",[0,1,2,3,3]),("role_indices",[0,1,2,4,3]),
        ("graph_ir",original["graph_ir"].replace("folded_row_reference_explicit_approximate","exact_per_block")),
    ]:
        bad=deepcopy(original);bad[key]=value
        bad["contract_digest"]=_program_digest({k:v for k,v in bad.items() if k!="contract_digest"})
        with pytest.raises(ValueError):
            program_from_manifest(bad)
    parent=rt.RuntimeArtifact(graph_ir=artifact.graph_ir+" ",metadata=artifact.metadata)
    receipt=rt.launch(parent,args)
    assert not receipt["ok"] and "parent Graph" in receipt["reason"]
    receipt=rt.launch(artifact,args,stream="borrowed")
    assert not receipt["ok"] and "owns its execution stream" in receipt["reason"]


def test_frontend_runtime_artifact_fresh_process_without_compiler(tmp_path):
    import subprocess
    import sys
    args,*_,expected=inputs_and_oracle(128,32,256)
    function=make_function(32,256,True)
    named=dict(zip(("codes","scales","projection_globals","a","a_scale"),args))
    function(**named)
    saved=tmp_path/"program.json";saved.write_text(function.runtime_artifact().to_json())
    inputs=tmp_path/"inputs.npz"
    np.savez(inputs,codes=args[0],scales=args[1].view(np.uint8),
             projection_globals=args[2],a=args[3],a_scale=args[4])
    output=tmp_path/"output.npy"
    script="""
from pathlib import Path
import sys
import numpy as np
import ml_dtypes
from tessera import runtime as rt
from tessera.compiler import scheduled_matmul,rocm_native
def forbidden(*a,**kw):
    raise AssertionError("portable frontend artifact invoked compiler")
scheduled_matmul.run_tessera_opt=forbidden
rocm_native._compile_native_tile_ir=forbidden
from tessera.compiler import rocm_nvfp4_resident
rocm_nvfp4_resident.ResidentNVFP4Matmul=forbidden
artifact=rt.RuntimeArtifact.from_json(Path(sys.argv[1]).read_text())
with np.load(sys.argv[2],allow_pickle=False) as x:
    args={name:x[name] for name in x.files}
    args["scales"]=args["scales"].view(ml_dtypes.float8_e4m3fn)
    result=rt.launch(artifact,args)
assert result["ok"] and result["execution_kind"]=="native_gpu",result
assert len(result["component_receipts"])==3
np.save(sys.argv[3],result["output"].astype(np.float32))
"""
    environment={**os.environ,"TESSERA_OPT":"/nonexistent/portable-frontend",
        "TESSERA_ROCM_NATIVE_MOVEMENT_LIB":str(rt._load_rocm_native_movement_runtime()._name),
        "TESSERA_ROCM_NATIVE_IMAGE_LIB":str(rt._load_rocm_native_image_runtime()._name)}
    result=subprocess.run([sys.executable,"-c",script,str(saved),str(inputs),str(output)],
                          env=environment,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    np.testing.assert_allclose(np.load(output,allow_pickle=False),expected,rtol=.008,atol=.015625)


def test_invalid_jit_policy_never_runs_eager_or_reaches_gpu(monkeypatch):
    import inspect
    from tessera.compiler import native_nvfp4_program,rocm_nvfp4_ingest
    args,*_=inputs_and_oracle(128,32,256)
    function=make_function(32,256)
    attributes=inspect.getclosurevars(function._fn).nonlocals["attrs"]
    attributes["numeric_policy"]["execution_mode"]="exact_per_block"
    def forbidden(*a,**kw):
        pytest.fail("invalid JIT policy reached eager conversion/compiler/GPU")
    monkeypatch.setattr(rocm_nvfp4_ingest,"reference_nvfp4_requantize",forbidden)
    monkeypatch.setattr(native_nvfp4_program,"export_native_nvfp4_program",forbidden)
    monkeypatch.setattr(rt,"_load_hip_for_launch",forbidden)
    with pytest.raises(ValueError,match="named contract"):
        function(*args)

def test_cached_manifest_is_independent_of_inspection_and_input_values(monkeypatch):
    from tessera.compiler import rocm_nvfp4_program
    args,*_,expected=inputs_and_oracle(128,32,256)
    function=make_function(32,256)
    function(*args)
    images=tuple(p.image.image_digest for p in function.native_nvfp4_packages())
    inspection=function.runtime_artifact()
    inspection.metadata["native_program"]["role_indices"]=[0,0,2,3,4]
    def forbidden(*a,**kw):
        pytest.fail("warm native launch reconstructed its compiler product")
    monkeypatch.setattr(rocm_nvfp4_program,"runtime_artifact",forbidden)
    args=(*args[:4],args[4]*np.float32(.5))
    output=function(*args)
    np.testing.assert_allclose(output.astype(np.float32),expected*.5,rtol=.008,atol=.015625)
    assert tuple(p.image.image_digest for p in function.native_nvfp4_packages())==images
    receipt=rt.launch(inspection,args)
    assert not receipt["ok"]
