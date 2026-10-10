"""Authored folded Graph -> Schedule/Tile -> native MLIR image proof."""
import os
from pathlib import Path
from dataclasses import replace
import ml_dtypes
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import (
    prepare_folded_weights, package_mxfp4_folded_prefill, FoldedPrefillSchedule)
from tessera.compiler.rocm_mxfp4_folded_frontend import compile_folded_scaled_matmul

pytestmark = [pytest.mark.hardware_rocm,pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 gate")]

def _inputs(m,n,k):
    rng=np.random.default_rng(m+n+k)
    a=rng.integers(-2,3,(m,k)).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    scales=np.exp2(rng.integers(-2,3,m)).astype(np.float32)
    codes=rng.integers(0,16,(n,k),dtype=np.uint8)
    references=rng.integers(125,130,(k//32,n),dtype=np.uint8)
    folded=prepare_folded_weights(mx.pack_e2m1_codes(codes),references,allow_approximate=True)
    return a,scales,folded

def _launch(package,a,scales,folded):
    m,k=a.shape;n=folded.weight_bytes.shape[0]
    output=np.full((m,n),-101,ml_dtypes.bfloat16)
    artifact=rt.RuntimeArtifact(metadata={"target":package.image.target},
        native_image=package.image,launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,target_ir=package.target_ir)
    result=rt.launch(artifact,{"buffers":{
        "a":a,"b_folded":folded.weight_bytes,"a_scale":scales,
        "row_reference":folded.row_reference,"output":output},
        "scalars":{"M":m,"N":n,"K":k}})
    assert result.get("ok") and result.get("execution_kind")=="native_gpu",result
    return output

@pytest.mark.parametrize("shape",[(65,48,64),(127,80,128),(256,128,192),(257,80,256),(513,129,128),(127,64,128),(256,129,128)])
@pytest.mark.parametrize("runtime_k",[False,True])
def test_native_folded_graph_matches_oracle_and_hip_control(shape,runtime_k):
    assert rt._rocm_live_arch()=="gfx1201"
    m,n,k=shape;a,scales,folded=_inputs(m,n,k)
    package=compile_folded_scaled_matmul(a,scales,folded,
        tessera_opt=Path(os.environ["TESSERA_OPT"]),allow_approximate=True,runtime_k=runtime_k).package
    prov=package.descriptor.provenance
    assert prov["native_compiler_owned"] and prov["kernel_argument_layout"]=="expanded_memref"
    assert package.image.pipeline_name=="tessera-lower-to-rocm"
    assert "gpu.binary" in package.backend_ir
    got=_launch(package,a,scales,folded)
    reference=(a.view(ml_dtypes.float8_e4m3fn).astype(np.float64)
        @ folded.weight_bytes.view(ml_dtypes.float8_e4m3fn).astype(np.float64).T)
    reference*=np.exp2(folded.row_reference.astype(np.int64)-127)[None,:]
    reference*=scales.astype(np.float64)[:,None]
    want=reference.astype(np.float32).astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(got.view(np.uint16),want.view(np.uint16))
    schedule=FoldedPrefillSchedule(raster_group_m=prov["raster_group_m"],
        workgroup_mode=prov["workgroup_mode"],staging_prefetch=prov["staging_prefetch"],
        epilogue=prov["epilogue_schedule"],row_guard=prov["row_guard"])
    control=package_mxfp4_folded_prefill(m,n,k,folded,
        entry="folded_hip_control",allow_approximate=True,schedule=schedule)
    np.testing.assert_array_equal(_launch(control,a,scales,folded).view(np.uint16),got.view(np.uint16))

@pytest.mark.parametrize("layout",[None,"raw_pointer","unknown"])
def test_native_folded_refuses_wrong_argument_layout_before_hip(layout,monkeypatch):
    a,scales,folded=_inputs(65,48,64)
    package=compile_folded_scaled_matmul(a,scales,folded,
        tessera_opt=Path(os.environ["TESSERA_OPT"]),allow_approximate=True).package
    prov=dict(package.descriptor.provenance)
    if layout is None:prov.pop("kernel_argument_layout")
    else:prov["kernel_argument_layout"]=layout
    broken=replace(package.descriptor,provenance=prov)
    monkeypatch.setattr(rt,"_load_hip_for_launch",lambda:pytest.fail("invalid argument layout reached HIP"))
    with pytest.raises(RuntimeError,match="expanded memref"):
        rt._submit_rocm_mxfp4_w4a8(package.image,broken,{
            "a":a,"b_folded":folded.weight_bytes,"a_scale":scales,
            "row_reference":folded.row_reference,"output":np.zeros((65,48),ml_dtypes.bfloat16)},
            {"M":65,"N":48,"K":64})

@pytest.mark.parametrize("case",["overflow_scale","underflow_scale","zero_partial"])
@pytest.mark.parametrize("runtime_k",[False,True])
def test_native_folded_package_preserves_wide_scale_fallback(case,runtime_k):
    assert rt._rocm_live_arch()=="gfx1201"
    m,n,k=65,48,64
    value=2.0**-9 if case=="overflow_scale" else 448.0 if case=="underflow_scale" else 0.0
    a=np.full((m,k),value,ml_dtypes.float8_e4m3fn).view(np.uint8)
    scales=np.full(m,2.0**-25 if case=="underflow_scale" else 2.0**20,np.float32)
    references=np.full((k//32,n),1 if case=="underflow_scale" else 240,np.uint8)
    codes=np.full((n,k),7 if case=="underflow_scale" else 2,np.uint8)
    if case=="overflow_scale":
        # Row reference stays 240, but only the delta-nine block contributes:
        # tiny partial * overflowing combined scale has a finite wide result.
        codes[:,:32]=0
        references[1,:]=231
    packed=mx.pack_e2m1_codes(codes)
    folded=prepare_folded_weights(packed,references,allow_approximate=True)
    package=compile_folded_scaled_matmul(a,scales,folded,
        tessera_opt=Path(os.environ["TESSERA_OPT"]),allow_approximate=True,runtime_k=runtime_k).package
    got=_launch(package,a,scales,folded)
    partial=a.view(ml_dtypes.float8_e4m3fn).astype(np.float64) @ folded.weight_bytes.view(ml_dtypes.float8_e4m3fn).astype(np.float64).T
    want=(partial*np.exp2(folded.row_reference.astype(np.int64)-127)[None,:]
          *scales.astype(np.float64)[:,None]).astype(np.float32).astype(ml_dtypes.bfloat16)
    assert np.isfinite(want.astype(np.float32)).all()
    if case!="zero_partial":
        assert np.all(want.view(np.uint16)!=0)
    np.testing.assert_array_equal(got.view(np.uint16),want.view(np.uint16))
    prov=package.descriptor.provenance
    schedule=FoldedPrefillSchedule(raster_group_m=prov["raster_group_m"],
        workgroup_mode=prov["workgroup_mode"],staging_prefetch=prov["staging_prefetch"],
        epilogue=prov["epilogue_schedule"],row_guard=prov["row_guard"])
    control=package_mxfp4_folded_prefill(m,n,k,folded,
        entry="folded_scale_fallback_control",allow_approximate=True,schedule=schedule)
    np.testing.assert_array_equal(_launch(control,a,scales,folded).view(np.uint16),got.view(np.uint16))

def test_native_folded_reuses_image_across_checked_launch_shapes():
    assert rt._rocm_live_arch()=="gfx1201"
    packages=[]
    for m,n in [(65,48),(127,80),(191,112)]:
        a,scales,folded=_inputs(m,n,64)
        package=compile_folded_scaled_matmul(a,scales,folded,
            tessera_opt=Path(os.environ["TESSERA_OPT"]),allow_approximate=True,runtime_k=False).package
        got=_launch(package,a,scales,folded)
        want=(a.view(ml_dtypes.float8_e4m3fn).astype(np.float64)
            @ folded.weight_bytes.view(ml_dtypes.float8_e4m3fn).astype(np.float64).T)
        want*=np.exp2(folded.row_reference.astype(np.int64)-127)[None,:]
        want*=scales.astype(np.float64)[:,None]
        np.testing.assert_array_equal(got.view(np.uint16),
            want.astype(np.float32).astype(ml_dtypes.bfloat16).view(np.uint16))
        packages.append(package)
    assert len({p.image.image_digest for p in packages})==1
    assert len({p.image.payload_digest for p in packages})==1
    assert len({p.descriptor.entry_symbol for p in packages})==1
    assert len({p.descriptor.provenance["schedule_hash"] for p in packages})==3
    assert len({p.descriptor.provenance["weight_sha256"] for p in packages})==3
    assert all(p.descriptor.provenance["image_shape_policy"]=="runtime_mn_fixed_k" for p in packages)
    assert all(p.image.compile_state=="warm_cache" for p in packages[1:])
    a,scales,folded=_inputs(65,48,128)
    different_k=compile_folded_scaled_matmul(a,scales,folded,
        tessera_opt=Path(os.environ["TESSERA_OPT"]),allow_approximate=True,runtime_k=False).package
    assert different_k.image.image_digest!=packages[0].image.image_digest
    # Public frontend retains a static native control for parity/timing.
    a,scales,folded=_inputs(65,48,64)
    static=compile_folded_scaled_matmul(a,scales,folded,
        tessera_opt=Path(os.environ["TESSERA_OPT"]),allow_approximate=True,runtime_mn=False,runtime_k=False).package
    assert static.descriptor.provenance["image_shape_policy"]=="static_mnk"
    assert static.image.image_digest!=packages[0].image.image_digest
    np.testing.assert_array_equal(_launch(static,a,scales,folded).view(np.uint16),
                                 _launch(packages[0],a,scales,folded).view(np.uint16))

@pytest.mark.parametrize("key,value",[
    ("image_whole_m",True),("image_whole_n",True),("image_whole_m","false"),
    ("image_k",128),("image_m",65),("image_shape_policy","unknown"),
])
def test_native_folded_refuses_invalid_image_shape_before_hip(key,value,monkeypatch):
    a,scales,folded=_inputs(65,48,64)
    package=compile_folded_scaled_matmul(a,scales,folded,
        tessera_opt=Path(os.environ["TESSERA_OPT"]),allow_approximate=True).package
    prov=dict(package.descriptor.provenance);prov[key]=value
    broken=replace(package.descriptor,provenance=prov)
    monkeypatch.setattr(rt,"_load_hip_for_launch",lambda:pytest.fail("invalid image shape reached HIP"))
    with pytest.raises(RuntimeError,match="native folded MXFP4"):
        rt._submit_rocm_mxfp4_w4a8(package.image,broken,{
            "a":a,"b_folded":folded.weight_bytes,"a_scale":scales,
            "row_reference":folded.row_reference,"output":np.zeros((65,48),ml_dtypes.bfloat16)},
            {"M":65,"N":48,"K":64})

@pytest.mark.parametrize("field,value",[("grid",(2,1,1)),("workgroup",(128,1,1))])
def test_native_folded_refuses_wrong_runtime_geometry_before_hip(field,value,monkeypatch):
    a,scales,folded=_inputs(65,48,64)
    package=compile_folded_scaled_matmul(a,scales,folded,
        tessera_opt=Path(os.environ["TESSERA_OPT"]),allow_approximate=True).package
    geometry=replace(package.descriptor.geometry,**{field:value})
    broken=replace(package.descriptor,geometry=geometry)
    monkeypatch.setattr(rt,"_load_hip_for_launch",lambda:pytest.fail("invalid geometry reached HIP"))
    with pytest.raises(RuntimeError,match="launch geometry"):
        rt._submit_rocm_mxfp4_w4a8(package.image,broken,{
            "a":a,"b_folded":folded.weight_bytes,"a_scale":scales,
            "row_reference":folded.row_reference,"output":np.zeros((65,48),ml_dtypes.bfloat16)},
            {"M":65,"N":48,"K":64})

def test_native_static_folded_refuses_wrong_shape_before_hip(monkeypatch):
    a,scales,folded=_inputs(65,48,64)
    package=compile_folded_scaled_matmul(a,scales,folded,
        tessera_opt=Path(os.environ["TESSERA_OPT"]),allow_approximate=True,runtime_mn=False,runtime_k=False).package
    monkeypatch.setattr(rt,"_load_hip_for_launch",lambda:pytest.fail("invalid static shape reached HIP"))
    with pytest.raises(RuntimeError,match="static image dimensions"):
        rt._submit_rocm_mxfp4_w4a8(package.image,package.descriptor,{
            "a":a,"b_folded":folded.weight_bytes,"a_scale":scales,
            "row_reference":folded.row_reference,"output":np.zeros((65,48),ml_dtypes.bfloat16)},
            {"M":66,"N":48,"K":64})

def test_native_folded_runtime_k_reuses_image_across_shapes_and_k():
    assert rt._rocm_live_arch()=="gfx1201"
    packages=[]
    for shape in [(65,48,64),(127,80,128),(191,112,192),(65,48,256)]:
        m,n,k=shape
        a,scales,folded=_inputs(m,n,k)
        native=compile_folded_scaled_matmul(a,scales,folded,
            tessera_opt=Path(os.environ["TESSERA_OPT"]),
            allow_approximate=True).package
        static=compile_folded_scaled_matmul(a,scales,folded,
            tessera_opt=Path(os.environ["TESSERA_OPT"]),
            allow_approximate=True,runtime_mn=False,runtime_k=False).package
        np.testing.assert_array_equal(_launch(native,a,scales,folded).view(np.uint16),
                                     _launch(static,a,scales,folded).view(np.uint16))
        assert native.descriptor.provenance["image_shape_policy"]=="runtime_mnk"
        assert native.descriptor.provenance["image_k"]==0
        packages.append(native)
    assert len({p.image.image_digest for p in packages})==1
    assert len({p.image.payload_digest for p in packages})==1
    assert len({p.descriptor.entry_symbol for p in packages})==1
    assert len({p.descriptor.provenance["schedule_hash"] for p in packages})==4
    assert len({p.descriptor.provenance["weight_sha256"] for p in packages})==4
    assert all(p.image.compile_state=="warm_cache" for p in packages[1:])

@pytest.mark.parametrize("k",[0,32,96])
def test_native_folded_runtime_k_refuses_invalid_extent_before_hip(k,monkeypatch):
    a,scales,folded=_inputs(65,48,64)
    package=compile_folded_scaled_matmul(a,scales,folded,
        tessera_opt=Path(os.environ["TESSERA_OPT"]),
        allow_approximate=True,runtime_k=True).package
    monkeypatch.setattr(rt,"_load_hip_for_launch",
        lambda:pytest.fail("invalid runtime K reached HIP"))
    with pytest.raises(RuntimeError,match="native folded MXFP4 runtime dimensions"):
        rt._submit_rocm_mxfp4_w4a8(package.image,package.descriptor,{
            "a":a,"b_folded":folded.weight_bytes,"a_scale":scales,
            "row_reference":folded.row_reference,
            "output":np.zeros((65,48),ml_dtypes.bfloat16)},
            {"M":65,"N":48,"K":k})

def test_native_folded_runtime_k_refuses_insufficient_capacity_before_hip(monkeypatch):
    a,scales,folded=_inputs(65,48,64)
    package=compile_folded_scaled_matmul(a,scales,folded,
        tessera_opt=Path(os.environ["TESSERA_OPT"]),
        allow_approximate=True,runtime_k=True).package
    monkeypatch.setattr(rt,"_load_hip_for_launch",
        lambda:pytest.fail("insufficient K capacity reached HIP"))
    with pytest.raises(RuntimeError):
        rt._submit_rocm_mxfp4_w4a8(package.image,package.descriptor,{
            "a":a,"b_folded":folded.weight_bytes,"a_scale":scales,
            "row_reference":folded.row_reference,
            "output":np.zeros((65,48),ml_dtypes.bfloat16)},
            {"M":65,"N":48,"K":128})
