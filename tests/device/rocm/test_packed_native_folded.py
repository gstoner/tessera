"""Exact gfx1201 packed Graph/Tile/native compiler proof."""
from dataclasses import replace
import ctypes as C
import os
from pathlib import Path
import ml_dtypes
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx,rocm_mxfp4_packed_folded as packed

pytestmark=pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",
    reason="exact gfx1201 owning host required")


def inputs(m,n,k):
    rng=np.random.default_rng(0x1201260+m+n+k)
    codes=rng.integers(0,16,(n,k),dtype=np.uint8)
    codes[:, :32] = np.tile(np.arange(16, dtype=np.uint8), 2)
    scales=np.full((k//32,n),127,np.uint8)
    scales[0]=127-np.arange(n,dtype=np.uint8)%15
    scales[0,::17]=0
    a=rng.integers(-4,5,(m,k)).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    sa=np.linspace(.5,1.5,m,dtype=np.float32)
    payload=packed.prepare_packed_folded_payload(mx.pack_e2m1_codes(codes),scales,allow_approximate=True)
    levels=np.asarray([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float64)
    reference=scales.max(axis=0)
    powers=np.where(scales==0,0.,np.exp2(scales.astype(np.int16)-reference[None,:])).T.repeat(32,axis=1)
    folded=(levels[codes]*powers).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    weight=(folded*np.exp2(reference.astype(np.int16)-127)[:,None]).T
    activation=a.view(ml_dtypes.float8_e4m3fn).astype(np.float64)*sa[:,None]
    expected=(activation@weight).astype(ml_dtypes.bfloat16)
    return a,sa,payload,expected


def compile_case(m,n,k,*,runtime_mn=False):
    assert rt._rocm_live_arch()=="gfx1201"
    a,sa,payload,expected=inputs(m,n,k)
    program=packed.compile_packed_folded_scaled_matmul(
        a,sa,payload,tessera_opt=Path(os.environ["TESSERA_OPT"]),runtime_mn=runtime_mn)
    return program,a,sa,payload,expected


@pytest.mark.parametrize("shape",[(83,32,64),(128,64,128),(256,80,192),(200,128,256)])
@pytest.mark.parametrize("runtime_mn",[False,True])
def test_native_packed_matches_independent_folded_oracle(shape,monkeypatch,runtime_mn):
    def legacy(*args,**kwargs):
        raise AssertionError("hand-emitted HIP materializer was used")
    monkeypatch.setattr(packed,"package_mxfp4_packed_folded_prefill",legacy)
    program,a,sa,payload,expected=compile_case(*shape,runtime_mn=runtime_mn)
    package=program.package
    assert package.image.pipeline_name=="tessera-lower-to-rocm"
    assert package.descriptor.provenance["native_compiler_owned"] is True
    assert package.descriptor.provenance["producer_kind"]=="native_mlir_packed_lds"
    output=np.full(expected.shape,np.nan,ml_dtypes.bfloat16)
    artifact=rt.RuntimeArtifact(tile_ir=package.tile_ir,target_ir=package.target_ir,
        metadata={"target":"rocm_gfx1201"},native_image=package.image,launch_descriptor=package.descriptor)
    buffers=dict(a=a,b_packed=payload.weight_bytes,a_scale=sa,scale_plane=payload.scale_plane,output=output)
    receipt=rt.launch(artifact,dict(buffers=buffers,scalars=dict(zip(("M","N","K"),shape,strict=True))))
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
    np.testing.assert_array_equal(output.view(np.uint16),expected.view(np.uint16))
    # Wrong physical scale bytes cannot reach the native image.
    changed=np.array(payload.scale_plane,copy=True)
    changed[0,0]^=np.uint8(1)
    refused=rt.launch(artifact,dict(buffers={**buffers,"scale_plane":changed},
        scalars=dict(zip(("M","N","K"),shape,strict=True))))
    assert not refused["ok"]
    invalid = [("native_compiler_owned", False),
               ("kernel_argument_layout", "raw_pointers"),
               ("decode_policy", "unchecked"),
               ("image_n", shape[1] + 16),
               ("image_k", shape[2] + 64),
               ("image_shape_policy", "runtime_mnk")]
    if runtime_mn:
        invalid += [("image_whole_m", not (shape[0] % 256 == 0)),
                    ("image_whole_n", not (shape[1] % 64 == 0)),
                    ("image_whole_m", int(shape[0] % 256 == 0))]
    for field, value in invalid:
        descriptor = replace(package.descriptor,
                             provenance={**package.descriptor.provenance, field: value})
        bad_artifact = replace(artifact, launch_descriptor=descriptor)
        refused = rt.launch(bad_artifact, dict(buffers=buffers,
            scalars=dict(zip(("M", "N", "K"), shape, strict=True))))
        assert not refused["ok"], (field, refused)


def test_native_packed_resident_matches_checked_launch():
    from benchmarks.rocm.benchmark_gfx1201_mxfp8_package import Resident
    from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape
    m,n,k=128,80,128
    program,a,sa,payload,expected=compile_case(m,n,k)
    hip=rt._load_hip_for_launch()
    assert hip is not None and hip.hipInit(0)==0
    output=np.full(expected.shape,np.nan,ml_dtypes.bfloat16)
    buffers=dict(a=a,b_packed=payload.weight_bytes,a_scale=sa,scale_plane=payload.scale_plane,output=output)
    engine=Resident(hip,program.package,buffers,BlockScaleShape(m,n,k,32,1,"nk","bf16"),
        grid=((n+63)//64,(m+255)//256,1),block=(256,1,1))
    try:
        engine.launch_on_stream(C.c_void_p())
        np.testing.assert_array_equal(engine.result(output).view(np.uint16),expected.view(np.uint16))
    finally:
        engine.close()

@pytest.mark.parametrize("pair",[
    ((128,32,256),(200,80,256)),
    ((256,64,256),(512,128,256)),
])
def test_projected_packages_share_image_with_distinct_shape_certificates(pair):
    left,right=(compile_case(*shape,runtime_mn=True)[0].package for shape in pair)
    assert left.target_ir==right.target_ir
    assert left.image.payload==right.image.payload
    assert left.image.image_digest==right.image.image_digest
    assert left.descriptor.entry_symbol==right.descriptor.entry_symbol
    assert left.descriptor.shape_guards!=right.descriptor.shape_guards
    assert left.descriptor.provenance["authored_target_ir_sha256"]!=right.descriptor.provenance["authored_target_ir_sha256"]
