"""Ordinary primal JIT reaches original typed Graph/Schedule/Tile packaging."""
import os
import subprocess
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tests.device.rocm.test_public_scaled_jvp import scaled,scaled_nk,oracle
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")

@pytest.mark.parametrize("nk",[False,True])
@pytest.mark.parametrize("shape",[(17,19,256),(200,129,1536)])
def test_public_fp8_primal(shape,nk,monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    m,n,k=shape;rng=np.random.default_rng(719)
    a=rng.choice([-.5,0,.25,1],(m,k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b=rng.choice([-1,0,.5,2],(k,n)).astype(ml_dtypes.float8_e4m3fn)
    b=np.ascontiguousarray(logical_b.T) if nk else logical_b
    sa=rng.uniform(.2,1,(m,k//128)).astype(np.float32)
    sb=rng.uniform(.2,1,(k//128,(n+127)//128)).astype(np.float32)
    fn=ts.jit(target="rocm_gfx1201")(scaled_nk if nk else scaled)
    got=fn(a,b,sa,sb)
    expected=oracle(a,logical_b,sa,sb,np.zeros_like(sa),np.zeros_like(sb))[0]
    np.testing.assert_allclose(got,expected,rtol=4e-5,atol=1e-4)
    assert fn.compile_result.executable
    import json
    manifest=fn.compile_result.launch_descriptor.provenance["native_scaled_primal_program"]
    assert json.loads(manifest["program_json"])["kind"]=="primal"
    assert fn._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    assert "tessera.scaled_matmul" in fn.compile_result.graph_ir
    assert "tile.scaled_matmul_kernel" in fn.compile_result.tile_ir
    def forbidden(*args,**kwargs):raise AssertionError("warm call attempted compiler subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated=fn(a,b,sa*.5,sb)
    np.testing.assert_allclose(repeated,expected*.5,rtol=4e-5,atol=1e-4)

from tessera.dtype import Dtype
encoded_byte = Dtype("uint8", allow_planned_gated=True)

def mxfp8(a:ts.Tensor["M","K","fp8_e4m3"], b:ts.Tensor["K","N","fp8_e4m3"],
          sa:ts.Tensor["M","G",encoded_byte], sb:ts.Tensor["G","N",encoded_byte]):
    import tessera as ts
    return ts.ops.scaled_matmul(a,b,sa,sb,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def mxfp8_nk(a:ts.Tensor["M","K","fp8_e4m3"], b:ts.Tensor["N","K","fp8_e4m3"],
             sa:ts.Tensor["M","G",encoded_byte], sb:ts.Tensor["G","N",encoded_byte]):
    import tessera as ts
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

@pytest.mark.parametrize("nk",[False,True])
@pytest.mark.parametrize("shape",[(17,19,256),(200,129,1536)])
def test_public_mxfp8_primal(shape,nk,monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    m,n,k=shape;rng=np.random.default_rng(811)
    a=rng.choice([-.5,0,.25,1],(m,k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b=rng.choice([-1,0,.5,2],(k,n)).astype(ml_dtypes.float8_e4m3fn)
    b=np.ascontiguousarray(logical_b.T) if nk else logical_b
    sa=rng.integers(125,130,(m,k//32),dtype=np.uint8)
    sb=rng.integers(125,130,(k//32,n),dtype=np.uint8)
    decoded_a=np.exp2(sa.astype(np.float64)-127)
    decoded_b=np.exp2(sb.astype(np.float64)-127)
    expected=np.zeros((m,n),np.float64)
    for g in range(k//32):
        expected+=(a[:,g*32:(g+1)*32].astype(np.float64) @
                   logical_b[g*32:(g+1)*32].astype(np.float64))*decoded_a[:,g,None]*decoded_b[g,None,:]
    source=mxfp8_nk if nk else mxfp8
    # Eager decoding is a frontend differential oracle, not a backend route.
    np.testing.assert_allclose(source(a,b,sa,sb),expected,rtol=4e-5,atol=1e-4)
    fn=ts.jit(target="rocm_gfx1201")(source)
    got=fn(a,b,sa,sb)
    np.testing.assert_allclose(got,expected,rtol=4e-5,atol=1e-4)
    assert fn.compile_result.executable
    import json
    manifest=fn.compile_result.launch_descriptor.provenance["native_scaled_primal_program"]
    assert json.loads(manifest["program_json"])["kind"]=="primal"
    assert fn._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    assert "ui8" in fn.compile_result.graph_ir
    assert 'tessera.dtype_status = "planned_gated"' in fn.compile_result.graph_ir
    assert "tile.scaled_matmul_kernel" in fn.compile_result.tile_ir
    def forbidden(*args,**kwargs):raise AssertionError("warm call attempted compiler subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    np.testing.assert_allclose(fn(a,b,sa-np.uint8(1),sb),expected*.5,rtol=4e-5,atol=1e-4)

@pytest.mark.parametrize("shape,selected_mode",[((17,19,256),"0"),((200,129,1536),"1")])
def test_native_auto_transfer_reuses_selected_mode(shape,selected_mode,monkeypatch):
    import ctypes as c,json
    from tessera import runtime
    from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
    assert runtime._rocm_live_arch()=="gfx1201"
    monkeypatch.delenv("TESSERA_ROCM_PROGRAM_PINNED",raising=False)
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_CACHE","1")
    m,n,k=shape;rng=np.random.default_rng(821)
    a=rng.choice([-.5,0,.25,1],(m,k)).astype(ml_dtypes.float8_e4m3fn)
    b=rng.choice([-1,0,.5,2],(n,k)).astype(ml_dtypes.float8_e4m3fn)
    sa=rng.integers(125,130,(m,k//32),dtype=np.uint8)
    sb=rng.integers(125,130,(k//32,n),dtype=np.uint8)
    fn=ts.jit(target="rocm_gfx1201")(mxfp8_nk)
    expected=fn(a,b,sa,sb)
    package=NativeScaledProgram.from_manifest(fn.compile_result.launch_descriptor.provenance["native_scaled_primal_program"])
    lib=runtime._load_rocm_native_movement_runtime()
    lib.tessera_rocm_program_cache_stats.argtypes=[c.POINTER(c.c_uint64)]*4
    def stats():
        values=[c.c_uint64() for _ in range(4)]
        assert lib.tessera_rocm_program_cache_stats(*(c.byref(v) for v in values))==0
        return [v.value for v in values]
    assert lib.tessera_rocm_program_cache_clear()==0
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_PINNED",selected_mode)
    with PreparedScaledProgram(package,[a,b,sa,sb],runtime_library=lib._name) as owner:
        generation,_=owner.invoke()
        np.testing.assert_array_equal(owner.read(generation)[0],expected)
    before=stats()
    monkeypatch.delenv("TESSERA_ROCM_PROGRAM_PINNED",raising=False)
    with PreparedScaledProgram(package,[a,b,sa-np.uint8(1),sb],runtime_library=lib._name) as owner:
        generation,_=owner.invoke()
        np.testing.assert_allclose(owner.read(generation)[0],expected*.5,rtol=4e-5,atol=1e-4)
    after=stats()
    assert after[0]==before[0]+1 and after[1]==before[1]
    assert after[2]==1 and after[3]<=128*1024*1024
    assert lib.tessera_rocm_program_cache_clear()==0

def test_native_idle_cache_evicts_oldest_and_reuses_new_program(monkeypatch):
    import ctypes as c
    from tessera import runtime
    from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
    assert runtime._rocm_live_arch()=="gfx1201"
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_CACHE","1")
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_PINNED","0")
    lib=runtime._load_rocm_native_movement_runtime()
    lib.tessera_rocm_program_cache_stats.argtypes=[c.POINTER(c.c_uint64)]*4
    def stats():
        values=[c.c_uint64() for _ in range(4)]
        assert lib.tessera_rocm_program_cache_stats(*(c.byref(v) for v in values))==0
        return [v.value for v in values]
    assert lib.tessera_rocm_program_cache_clear()==0
    programs=[]
    for m in range(17,23):
        a=np.full((m,256),.5,dtype=ml_dtypes.float8_e4m3fn)
        b=np.full((19,256),.25,dtype=ml_dtypes.float8_e4m3fn)
        sa=np.full((m,8),127,dtype=np.uint8)
        sb=np.full((8,19),127,dtype=np.uint8)
        fn=ts.jit(target="rocm_gfx1201")(mxfp8_nk)
        np.testing.assert_allclose(fn(a,b,sa,sb),32,rtol=0,atol=0)
        package=NativeScaledProgram.from_manifest(fn.compile_result.launch_descriptor.provenance["native_scaled_primal_program"])
        programs.append((package,[a,b,sa,sb]))
        assert stats()[2]==min(m-16,4)
        assert stats()[3]<=128*1024*1024
    before=stats()
    package,inputs=programs[-1]
    changed=[*inputs[:2],inputs[2]-np.uint8(1),inputs[3]]
    with PreparedScaledProgram(package,changed,runtime_library=lib._name) as owner:
        generation,_=owner.invoke()
        np.testing.assert_allclose(owner.read(generation)[0],16,rtol=0,atol=0)
    after=stats()
    assert after[0]==before[0]+1 and after[1]==before[1]
    package,inputs=programs[0]
    with PreparedScaledProgram(package,inputs,runtime_library=lib._name) as owner:
        generation,_=owner.invoke()
        np.testing.assert_allclose(owner.read(generation)[0],32,rtol=0,atol=0)
    final=stats()
    assert final[1]==after[1]+1 and final[2]==4
    assert lib.tessera_rocm_program_cache_clear()==0

def test_public_mxfp8_reserved_and_smallest_scale_codes():
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    a=np.full((17,32),.5,dtype=ml_dtypes.float8_e4m3fn)
    b=np.full((19,32),.5,dtype=ml_dtypes.float8_e4m3fn)
    sa=np.full((17,1),127,dtype=np.uint8)
    sa[0,0]=0
    sa[1,0]=255
    sb=np.full((1,19),127,dtype=np.uint8)
    sb[0,1]=255
    expected=(8*sa.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)*
              sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(mxfp8_nk(a,b,sa,sb),expected,rtol=0,atol=0,equal_nan=True)
    fn=ts.jit(target="rocm_gfx1201")(mxfp8_nk)
    np.testing.assert_allclose(fn(a,b,sa,sb),expected,rtol=0,atol=0,equal_nan=True)
    assert fn._native_descriptor_last_receipt["execution_kind"]=="native_gpu"

@pytest.mark.parametrize("shape",[(3,7,19,256),(2,100,129,1536)])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_public_shared_rhs_scaled_batch(shape,fmt,nk,monkeypatch):
    import json
    from tessera import runtime
    from tests.unit import test_rocm_shared_scaled_batch as shared
    assert runtime._rocm_live_arch()=="gfx1201"
    values,expected=shared.batch_inputs(shape,fmt,nk)
    source=getattr(shared,f"shared_{fmt}_{'nk' if nk else 'kn'}")
    fn=ts.jit(target="rocm_gfx1201")(source)
    actual=fn(*values)
    np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=1e-4)
    assert actual.shape==shape[:2]+(shape[2],)
    descriptor=fn.compile_result.launch_descriptor
    manifest=descriptor.provenance["native_scaled_primal_program"]
    program=json.loads(manifest["program_json"])
    assert len(program["steps"])==1 and program["argument_count"]==4
    assert program["buffers"][0]["shape"]==list(values[0].shape)
    assert program["buffers"][1]["shape"]==list(values[1].shape)
    assert program["buffers"][program["outputs"][0]]["shape"]==list(actual.shape)
    assert json.loads(manifest["members_json"][0])["scalars"]==[shape[0]*shape[1],shape[2],shape[3]]
    assert fn._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    assert 'batching = "shared_rhs_rows"' in fn.compile_result.graph_ir
    assert "tile.scaled_matmul_kernel" in fn.compile_result.tile_ir
    def forbidden(*args,**kwargs):raise AssertionError("warm batch attempted compiler subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    changed=list(values);changed[2]=changed[2]-np.uint8(1) if fmt=="e8m0" else changed[2]*.5
    np.testing.assert_allclose(fn(*changed),expected*.5,rtol=4e-5,atol=1e-4)

@pytest.mark.parametrize("shape",[(3,7,19,256),(2,100,129,1536),(2,128,4096,256)])
@pytest.mark.parametrize("policy",["independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_public_independent_scaled_batch(shape,policy,fmt,nk,monkeypatch):
    import json
    from tessera import runtime
    from tests.unit import test_rocm_independent_scaled_batch as batch
    assert runtime._rocm_live_arch()=="gfx1201"
    values,oracle=batch.batch_inputs(shape,fmt,nk,policy)
    prefix="independent" if policy=="independent_rhs" else "lhs_shared"
    source=getattr(batch,f"{prefix}_{fmt}_{'nk' if nk else 'kn'}")
    fn=ts.jit(target="rocm_gfx1201")(source)
    actual=fn(*values)
    np.testing.assert_allclose(actual,oracle,rtol=4e-5,atol=1e-4)
    if nk and fmt=="fp32" and shape==(2,128,4096,256):
        assert 'staging = "lds"' in fn.compile_result.tile_ir
    manifest=fn.compile_result.launch_descriptor.provenance["native_scaled_primal_program"]
    member=json.loads(manifest["members_json"][0])
    assert member["scalars"]==list(shape[1:])
    assert member["geometry"][2]==shape[0]
    assert fn._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    def forbidden(*args,**kwargs):raise AssertionError("warm batch attempted compiler subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    changed=list(values)
    # Only one RHS batch changes: exposes aliased scales and wrong z-plane offsets.
    changed[3]=changed[3].copy()
    changed[3][1]=changed[3][1]-np.uint8(1) if fmt=="e8m0" else changed[3][1]*.5
    expected=oracle.copy();expected[1]*=.5
    np.testing.assert_allclose(fn(*changed),expected,rtol=4e-5,atol=1e-4)

def test_immutable_host_binding_preserves_distinct_active_native_owners(monkeypatch):
    from tessera import runtime
    from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
    from tests.unit import test_rocm_independent_scaled_batch as batch
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_BINDING_CACHE","1")
    assert runtime._rocm_live_arch()=="gfx1201"
    values,oracle=batch.batch_inputs()
    fn=ts.jit(target="rocm_gfx1201")(batch.independent_fp32_kn)
    np.testing.assert_allclose(fn(*values),oracle,rtol=4e-5,atol=1e-4)
    package=NativeScaledProgram.from_manifest(fn.compile_result.launch_descriptor.provenance["native_scaled_primal_program"])
    lib=runtime._load_rocm_native_movement_runtime()
    with PreparedScaledProgram(package,values,runtime_library=lib._name) as a:
        with PreparedScaledProgram(package,values,runtime_library=lib._name) as b:
            assert a._binding is b._binding and a.handle.value!=b.handle.value
            # Metadata snapshots remain per owner, even though ABI bytes are shared.
            a.program["buffers"][0]["shape"][0]=1
            assert b.program["buffers"][0]["shape"][0]==3
            g,_=b.invoke()
            np.testing.assert_allclose(b.read(g)[0],oracle,rtol=4e-5,atol=1e-4)
