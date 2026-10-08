"""Exact SM120 nested NVFP4 native maps and compiler-free changed-input reuse."""
import pytest,numpy as np
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.unit.test_native_nested_nvfp4_vmap import case,AXES

pytestmark=[pytest.mark.hardware_nvidia,pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning SM120 required")]

@pytest.mark.parametrize("mode",tuple(AXES))
@pytest.mark.parametrize("ta,tb",[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize("prefix,shape",[((2,3),(7,5,31)),((1,2,3),(17,19,129))])
def test_nested_public_nvfp4_single_launch_native_replay(mode,ta,tb,prefix,shape,monkeypatch):
    from tessera import runtime as rt
    scalar,owner,values,wanted=case(mode,ta,tb,prefix,shape)
    before=scalar.graph_ir.to_mlir();launches=[]
    real=rt.launch
    def counted(*args,**kwargs):launches.append(1);return real(*args,**kwargs)
    monkeypatch.setattr(rt,"launch",counted)
    out=owner(*values)
    assert len(launches)==1
    np.testing.assert_allclose(out,wanted,rtol=0,atol=2e-3)
    assert owner._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    desc=owner._cached_artifact.launch_descriptor
    assert desc.provenance["logical_batch_shape"]==list(prefix)
    assert out.shape==(*prefix,shape[0],shape[1])
    assert "tile.matmul_kernel" in owner.compile_bundle.tile.text
    assert "mma.sync.aligned.m16n8k64" in owner._cached_artifact.native_image.payload.decode()
    import importlib,subprocess
    compiler=importlib.import_module("tessera.compiler.canonical_compile")
    def forbidden(*args,**kwargs):raise AssertionError("warm nested call escaped native package")
    monkeypatch.setattr(compiler,"canonical_compile",forbidden)
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(owner,"_fn",forbidden)
    # New codes and scale arrays retain the exact signature and image.
    _,_,fresh,expected=case(mode,ta,tb,prefix,shape,seed=50708)
    np.testing.assert_allclose(owner(*fresh),expected,rtol=0,atol=2e-3)
    assert len(launches)==2 and scalar.graph_ir.to_mlir()==before
    np.testing.assert_allclose(out,wanted,rtol=0,atol=2e-3)
