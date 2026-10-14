"""Owning gfx1201 proof for exact MXFP4 BF16 residuals and native widening."""
import json
import os
import subprocess
import numpy as np
import pytest
from tessera import runtime
from tessera.compiler.native_scaled_program import (
    PreparedScaledProgram,package_native_scaled_primal,package_native_scaled_vjp)
from tests.unit.test_native_mxfp4_residual_gate import typed_case
from tests._support import rocm_isa

pytestmark=pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")


def frame(shape):
    import ml_dtypes
    m,k,n=shape
    rng=np.random.default_rng(946+k+n)
    a=rng.choice([-3,-1,-.5,0,.5,1,3],(m,k)).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    codes=rng.integers(0,16,(k,n),dtype=np.uint8)
    b=np.ascontiguousarray(codes[::2]|(codes[1::2]<<np.uint8(4)))
    sa=rng.choice([.25,.5,1,2],m).astype(np.float32)
    sb=rng.integers(124,131,(k//32,n),dtype=np.uint8)
    sb[0,::7]=0
    gate=rng.uniform(-.75,.75,(m,n)).astype(np.float32)
    cot=rng.uniform(-.75,.75,(m,n)).astype(np.float32)
    return (a,b,sa,sb,gate),cot


def coefficient(values):
    import ml_dtypes
    a,b,sa,sb,_=values
    decoded_a=a.view(ml_dtypes.float8_e4m3fn).astype(np.float64)
    codes=np.empty((a.shape[1],b.shape[1]),np.uint8)
    codes[::2]=b&np.uint8(15)
    codes[1::2]=b>>np.uint8(4)
    magnitudes=np.array([0,.5,1,1.5,2,3,4,6],np.float64)
    decoded_b=magnitudes[codes&np.uint8(7)]*np.where(codes&np.uint8(8),-1.,1.)
    result=np.zeros(values[4].shape,np.float32)
    for group in range(a.shape[1]//32):
        block=slice(group*32,(group+1)*32)
        factor=np.where(sb[group]==0,0.,np.ldexp(np.ones(b.shape[1]),sb[group].astype(np.int32)-127))
        partial=(decoded_a[:,block]@decoded_b[block,:])*sa.astype(np.float64)[:,None]*factor
        result+=partial.astype(np.float32)
    return result.astype(ml_dtypes.bfloat16).astype(np.float32)


@pytest.mark.parametrize("shape",[(7,32,5),(17,64,19),(256,128,80)])
@pytest.mark.parametrize("mode",["primal","reverse"])
def test_native_exact_mxfp4_coefficient_gate(shape,mode,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    owner,_=typed_case(shape)
    values,cot=frame(shape)
    graph=owner._specialized_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    builder=package_native_scaled_primal if mode=="primal" else package_native_scaled_vjp
    package=builder(graph.to_mlir(target="rocm_gfx1201",canonical=True))
    manifest=json.loads(package.program_json)
    product=next(step for step in manifest["steps"] if step["operation"]=="tessera.scaled_matmul")
    rocm_isa.assert_selected(package.images[product["step"]],chip="gfx1201",
        pattern=r"v_wmma_f32_16x16x16_\w+",require="v_wmma_f32_16x16x16_fp8_fp8",
        what="native exact MXFP4 residual")
    expected=coefficient(values)*(values[4] if mode=="primal" else cot)
    inputs=values if mode=="primal" else (*values,cot)
    with PreparedScaledProgram(package,inputs,
            runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as prepared:
        generation,_=prepared.invoke()
        actual=prepared.read(generation)
        np.testing.assert_array_equal(actual[0],expected)
        retained=actual[0].copy()
        original_run=subprocess.run
        def forbidden(*args,**kwargs):
            raise AssertionError("warm native MXFP4 residual invoked compiler")
        monkeypatch.setattr(subprocess,"run",forbidden)
        changed=list(values)
        changed[0]=np.bitwise_xor(values[0],np.uint8(128))
        changed[1]=np.bitwise_xor(values[1],np.uint8(136))
        changed[2]=values[2]*np.float32(.5)
        changed[4]=values[4]*np.float32(-.5)
        changed_cot=cot*np.float32(-.5)
        prepared.update(changed if mode=="primal" else (*changed,changed_cot))
        repeated_generation,_=prepared.invoke()
        repeated=prepared.read(repeated_generation)
        np.testing.assert_array_equal(repeated[0],
            coefficient(changed)*(changed[4] if mode=="primal" else changed_cot))
        np.testing.assert_array_equal(actual[0],retained)
        with pytest.raises(RuntimeError):
            prepared.read(generation)
        monkeypatch.setattr(subprocess,"run",original_run)

