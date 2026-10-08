"""Public typed FP8 capture retains native scale-JVP semantics."""
from pathlib import Path
import numpy as np
import ml_dtypes
import pytest
import tessera as ts
from tessera.compiler.native_scaled_program import package_native_scaled_jvp

def public_scaled(a:ts.Tensor["M","K","fp8_e4m3"],
                  b:ts.Tensor["K","N","fp8_e4m3"],
                  sa:ts.Tensor["M","G","fp32"],
                  sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def _inputs():
    rng=np.random.default_rng(27)
    a=rng.choice([-.5,0,.25,1],(17,256)).astype(ml_dtypes.float8_e4m3fn)
    b=rng.choice([-1,0,.5,2],(256,19)).astype(ml_dtypes.float8_e4m3fn)
    sa=rng.uniform(.2,1,(17,2)).astype(np.float32)
    sb=rng.uniform(.2,1,(2,1)).astype(np.float32)
    return a,b,sa,sb

def test_typed_scaled_eager_preserves_block_scale_semantics():
    a,b,sa,sb=_inputs()
    want=sum((a[:,g*128:(g+1)*128].astype(np.float64)@
              b[g*128:(g+1)*128].astype(np.float64))*
             sa[:,g,None]*sb[g,0] for g in range(2))
    from tessera.compiler.reference_typed_scaled_matmul import reference_typed_scaled_matmul
    actual=reference_typed_scaled_matmul(a,b,sa,sb,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})
    np.testing.assert_allclose(actual,want,rtol=2e-6,atol=2e-5)

def test_typed_scaled_public_capture_packages_native_graph(monkeypatch):
    import os,json
    if not os.environ.get("TESSERA_OPT"):pytest.skip("matching native compiler required")
    monkeypatch.setenv("TESSERA_ROCM_CHIP","gfx1201")
    from dataclasses import replace
    compiled=ts.jit(target="rocm",autodiff="forward",wrt=("sa","sb"))(public_scaled)
    module=compiled._specialized_autodiff_module(_inputs(),{})
    module=replace(module,module_attrs={**module.module_attrs,
        "tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    graph=module.to_mlir(target="rocm")
    assert "f8E4M3FN" in graph
    assert 'physical_contract = none' not in graph
    package=package_native_scaled_jvp(graph)
    p=json.loads(package.program_json)
    assert p["argument_count"]==6
    assert len(p["steps"])==4
    assert p["steps"][-1]["operation"]=="tessera.add"
    assert all(b["storage"]=="f32" for b in p["buffers"][2:])

def test_scaled_program_never_admits_gfx1151_fp8_wmma():
    from tessera.compiler.native_jvp import architecture_admits
    assert not architecture_admits("rocm","gfx1151","scaled_product_program")
    assert architecture_admits("rocm","gfx1201","scaled_product_program")

@pytest.mark.parametrize("transpose_a,transpose_b",[(False,False),(False,True),(True,False),(True,True)])
def test_typed_e8m0_eager_uses_encoded_exponents(transpose_a,transpose_b):
    rng=np.random.default_rng(831)
    a=rng.choice([-.5,0,.25,1],(3,64)).astype(ml_dtypes.float8_e4m3fn)
    b=rng.choice([-1,0,.5,2],(64,5)).astype(ml_dtypes.float8_e4m3fn)
    sa=np.array([[125,129],[128,127],[126,130]],dtype=np.uint8)
    sb=rng.integers(125,130,(2,5),dtype=np.uint8)
    # Independent ml_dtypes E8M0 interpretation, not the implementation decoder.
    decoded_a=sa.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    decoded_b=sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    want=sum((a[:,g*32:(g+1)*32].astype(np.float64)@
              b[g*32:(g+1)*32].astype(np.float64))*
             decoded_a[:,g,None]*decoded_b[g,None,:] for g in range(2))
    actual=ts.ops.scaled_matmul(
        np.ascontiguousarray(a.T) if transpose_a else a,
        np.ascontiguousarray(b.T) if transpose_b else b,sa,sb,
        transposeA=transpose_a,transposeB=transpose_b,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})
    assert actual.dtype==np.float32 and actual.shape==(3,5)
    np.testing.assert_allclose(actual,want,rtol=2e-6,atol=2e-5)

def test_typed_e8m0_eager_preserves_zero_code_and_nan_code():
    a=np.full((2,32),.5,dtype=ml_dtypes.float8_e4m3fn)
    b=np.full((32,1),.5,dtype=ml_dtypes.float8_e4m3fn)
    actual=ts.ops.scaled_matmul(a,b,np.array([[0],[255]],np.uint8),
        np.array([[127]],np.uint8),
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})
    assert actual[0,0]==np.float32(8*np.exp2(-127.))
    assert np.isnan(actual[1,0])

@pytest.mark.parametrize("block,scale_dtype",[([2,32],np.uint8),([1,16],np.uint8),([1,32],np.int8),([1,32],np.float32)])
def test_typed_e8m0_eager_rejects_conflicting_layout_or_storage(block,scale_dtype):
    a=np.zeros((2,32),dtype=ml_dtypes.float8_e4m3fn)
    b=np.zeros((32,1),dtype=ml_dtypes.float8_e4m3fn)
    with pytest.raises(ValueError):
        ts.ops.scaled_matmul(a,b,np.ones((2,1),dtype=scale_dtype),
            np.ones((1,1),dtype=scale_dtype),
            numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
            scale_layout={"granularity":"block","block":block,"format":"e8m0"})
