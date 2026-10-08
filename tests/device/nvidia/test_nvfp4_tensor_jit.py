"""Ordinary JIT NVFP4 logical storage binding on owning SM120."""
import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tessera.compiler.nvfp4_tensor import NVFP4Tensor
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_e2e_spine_native import _pack_nvfp4, _decode_e2m1, _decode_ue4m3

pytestmark=[pytest.mark.hardware_nvidia,pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning SM120 required")]

def rank_two(a,b,sa,sb):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        physical_contract="nvidia_sm120_nvfp4_blockscale_v1",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,16],"format":"ue4m3"})

def shared(a,b,sa,sb):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        physical_contract="nvidia_sm120_nvfp4_blockscale_v1",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,16],"format":"ue4m3"},
        batching="shared_rhs_rows")

def independent(a,b,sa,sb):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        physical_contract="nvidia_sm120_nvfp4_blockscale_v1",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,16],"format":"ue4m3"},
        batching="independent_rhs")

@pytest.mark.parametrize("mode",["rank_two","shared","independent"])
@pytest.mark.parametrize("use_vmap", [False, True])
@pytest.mark.parametrize("rows,n,k",[(7,5,31),(17,19,129)])
def test_ordinary_nvfp4_jit_matches_oracle_and_rebinds(mode,rows,n,k,use_vmap,monkeypatch):
    if use_vmap and mode == "rank_two":
        pytest.skip("rank-two has no batch transform")
    batch=3
    a_shape=(rows,k) if mode=="rank_two" else (batch,rows,k)
    b_shape=(batch,k,n) if mode=="independent" else (k,n)
    sk=k//16+(k%16!=0)
    sa_shape=(*a_shape[:-1],sk)
    sb_shape=(batch,sk,n) if mode=="independent" else (sk,n)
    rng=np.random.default_rng(120612+k)
    ac=rng.integers(0,16,a_shape,dtype=np.uint8)
    bc=rng.integers(0,16,b_shape,dtype=np.uint8)
    a=NVFP4Tensor(_pack_nvfp4(ac,len(a_shape)-1),a_shape,len(a_shape)-1)
    b=NVFP4Tensor(_pack_nvfp4(bc,len(b_shape)-2),b_shape,len(b_shape)-2)
    choices=np.array([0x30,0x33,0x35,0x38,0x3a,0x40],np.uint8)
    sa=np.ascontiguousarray(choices[rng.integers(0,len(choices),sa_shape)])
    sb=np.ascontiguousarray(choices[rng.integers(0,len(choices),sb_shape)])
    scalar = ts.jit(rank_two if use_vmap else globals()[mode], target="nvidia_sm120")
    before = scalar.graph_ir.to_mlir()
    call = vmap(scalar, in_axes=(0, None, 0, None) if mode == "shared" else 0) if use_vmap else scalar
    expected=(_decode_e2m1(ac)*np.repeat(_decode_ue4m3(sa),16,axis=len(a_shape)-1)[..., :k]).astype(np.float64)
    rhs_scale=np.repeat(_decode_ue4m3(sb),16,axis=len(b_shape)-2)
    rhs_scale=rhs_scale[:, :k, :] if mode=="independent" else rhs_scale[:k,:]
    expected=expected@(_decode_e2m1(bc)*rhs_scale).astype(np.float64)
    from tessera import runtime as runtime
    launches = []
    real_launch = runtime.launch
    def counted_launch(*args, **kwargs):
        launches.append(1)
        return real_launch(*args, **kwargs)
    monkeypatch.setattr(runtime, "launch", counted_launch)
    out=call(a,b,sa,sb)
    assert len(launches) == 1
    np.testing.assert_allclose(out,expected,rtol=0,atol=2e-3)
    if use_vmap:
        assert scalar.graph_ir.to_mlir() == before
        assert call.graph_ir.functions[0].body[0].kwargs["batching"] == ("shared_rhs_rows" if mode == "shared" else "independent_rhs")
    assert call.frontend_authority=="tracer"
    assert call._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    assert "!tessera.nvfp4" in call.compile_bundle.graph.text
    assert "tile.matmul_kernel" in call.compile_bundle.tile.text
    assert "mma.sync.aligned.m16n8k64" in call._cached_artifact.native_image.payload.decode()
    def forbidden(*args,**kwargs):
        raise AssertionError("warm JIT must not execute eager Python or compile")
    monkeypatch.setattr(call,"_fn",forbidden)
    import tessera.compiler.canonical_compile as compiler
    monkeypatch.setattr(compiler,"canonical_compile",forbidden)
    sa[:]=0x38
    expected=(_decode_e2m1(ac)*np.repeat(_decode_ue4m3(sa),16,axis=len(a_shape)-1)[..., :k]).astype(np.float64)@(_decode_e2m1(bc)*rhs_scale).astype(np.float64)
    np.testing.assert_allclose(call(a,b,sa,sb),expected,rtol=0,atol=2e-3)
    assert len(launches) == 2

def test_wrong_matrix_packing_axis_refused_before_compilation(monkeypatch):
    a=NVFP4Tensor(np.zeros((2,4),np.uint8),(4,4),0)
    b=NVFP4Tensor(np.zeros((2,5),np.uint8),(4,5),0)
    call=ts.jit(rank_two,target="nvidia_sm120")
    import importlib
    compiler=importlib.import_module("tessera.compiler.canonical_compile")
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid binding must not compile or launch")
    monkeypatch.setattr(compiler,"canonical_compile",forbidden)
    with pytest.raises(ValueError,match="packing axis differs"):
        call(a,b,np.ones((4,1),np.uint8),np.ones((1,5),np.uint8))

def test_invalid_scale_storage_never_falls_back_to_eager(monkeypatch):
    a=NVFP4Tensor(np.zeros((4,2),np.uint8),(4,4),1)
    b=NVFP4Tensor(np.zeros((2,5),np.uint8),(4,5),0)
    call=ts.jit(rank_two,target="nvidia_sm120")
    import importlib
    compiler=importlib.import_module("tessera.compiler.canonical_compile")
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid storage must not compile")
    monkeypatch.setattr(compiler,"canonical_compile",forbidden)
    with pytest.raises(ValueError,match="native NVFP4 JIT requires"):
        call(a,b,np.ones((4,1),np.float32),np.ones((1,5),np.uint8))


def test_native_vmap_equal_flat_rows_preserve_batch_geometry(monkeypatch):
    from tests.unit.test_native_nvfp4_vmap import typed_product
    scalar = ts.jit(typed_product, target="nvidia_sm120")
    owner = vmap(scalar, in_axes=0)
    rng = np.random.default_rng(120614)
    fingerprints = []
    for batch, rows in ((3, 7), (7, 3), (3, 7)):
        k, n = 31, 5
        ac = rng.integers(0, 16, (batch, rows, k), dtype=np.uint8)
        bc = rng.integers(0, 16, (batch, k, n), dtype=np.uint8)
        a = NVFP4Tensor(_pack_nvfp4(ac, 2), ac.shape, 2)
        b = NVFP4Tensor(_pack_nvfp4(bc, 1), bc.shape, 1)
        sa = np.full((batch, rows, 2), 0x38, np.uint8)
        sb = np.broadcast_to(np.arange(batch, dtype=np.uint8)[:, None, None] + 0x30,
                             (batch, 2, n)).copy()
        oracle = _decode_e2m1(ac).astype(np.float64) @ (
            _decode_e2m1(bc) * np.repeat(_decode_ue4m3(sb), 16, axis=1)[:, :k, :]).astype(np.float64)
        if len(fingerprints) == 2:
            import importlib
            compiler = importlib.import_module("tessera.compiler.canonical_compile")
            def forbidden(*args, **kwargs):
                raise AssertionError("return to an existing geometry must reuse its package")
            monkeypatch.setattr(compiler, "canonical_compile", forbidden)
        np.testing.assert_allclose(owner(a=a, b=b, sa=sa, sb=sb), oracle, rtol=0, atol=2e-3)
        fingerprints.append(owner._cached_artifact.launch_descriptor.provenance["schedule_digest"])
        assert owner._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
        assert owner.graph_ir.functions[0].result_types[0].shape == (str(batch), str(rows), str(n))
    assert fingerprints[0] != fingerprints[1]
    assert fingerprints[0] == fingerprints[2]
    assert len(owner._native_descriptor_artifacts) == 2


def test_typed_vmap_constraint_refuses_before_compile(monkeypatch):
    from tests.unit.test_native_nvfp4_vmap import typed_product, operands
    from tessera.compiler.constraints import Range, TesseraConstraintError
    scalar = ts.jit(typed_product, target="nvidia_sm120")
    scalar.constraints.add(Range("M", 1, 6))
    owner = vmap(scalar, in_axes=0)
    import importlib
    compiler = importlib.import_module("tessera.compiler.canonical_compile")
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid scalar dimension must not compile or launch")
    monkeypatch.setattr(compiler, "canonical_compile", forbidden)
    with pytest.raises(TesseraConstraintError, match="M"):
        owner(*operands(True))
