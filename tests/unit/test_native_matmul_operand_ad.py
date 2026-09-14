"""Logical operands survive physical specialization and AD at numeric zeros."""
from types import SimpleNamespace
import numpy as np
import pytest
from tessera.compiler.native_vjp_plugins import _execute_rocm_matmul_backward


def execute(monkeypatch,operands,a,b,**policies):
    from tessera import runtime
    def launch(artifact,inputs):
        return {'ok':True,'execution_mode':'hip_runtime','output':inputs[0]@inputs[1]}
    monkeypatch.setattr(runtime,'launch',launch)
    monkeypatch.setattr(runtime,'_rocm_live_arch',lambda: 'gfx1201')
    return _execute_rocm_matmul_backward(source=SimpleNamespace(operands=operands,kwargs=policies),
        target='rocm',ordered_inputs=(a,b),arg_names=('a','b'),source_arg_names=('x','y'),
        out_cotangents=np.ones((2,2),np.float32),wrt_names=('a','b'),
        declaration=SimpleNamespace(family='matmul_backward',schedule_consumer='test',tile_consumer='test',target_consumers={'rocm':'test'}),
        source_graph_ir=None,frontend_certificate=None).gradients


def test_reordered_operands_preserve_derivatives_at_zeros(monkeypatch):
    a=np.array([[0.,2.],[0.,0.]],np.float32)
    b=np.array([[3.,0.],[4.,5.]],np.float32)
    da,db=execute(monkeypatch,['%y','%x'],a,b)
    np.testing.assert_array_equal(da,b.T@np.ones((2,2)))
    np.testing.assert_array_equal(db,np.ones((2,2))@a.T)
    assert da[0,0] != 0


def test_repeated_operand_accumulates_both_edges(monkeypatch):
    a=np.array([[0.,2.],[3.,0.]],np.float32)
    da,db=execute(monkeypatch,['%x','%x'],a,np.ones_like(a))
    np.testing.assert_array_equal(da,np.ones((2,2))@a.T+a.T@np.ones((2,2)))
    np.testing.assert_array_equal(db,0)


def test_unconsumed_epilogue_refused(monkeypatch):
    from tessera.compiler.jit import TesseraJitError
    with pytest.raises(TesseraJitError,match='policies'):
        execute(monkeypatch,['%x','%y'],np.ones((2,2)),np.ones((2,2)),activation='relu')


@pytest.mark.skipif(__import__('os').environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1',reason='owning gfx1201 proof')
@pytest.mark.parametrize('operands',[['%y','%x'],['%x','%x']])
def test_device_logical_matmul_edges_at_zeros(operands):
    from tessera import runtime
    assert runtime._rocm_live_arch() == 'gfx1201'
    rng=np.random.default_rng(81)
    a=(rng.integers(-3,4,(16,16))*.125).astype(np.float16)
    a[:,::2]=0
    b=(rng.integers(-3,4,(16,16))*.125).astype(np.float16)
    result=_execute_rocm_matmul_backward(source=SimpleNamespace(operands=operands,kwargs={}),
        target='rocm',ordered_inputs=(a,b),arg_names=('a','b'),source_arg_names=('x','y'),
        out_cotangents=np.ones((16,16),np.float16),wrt_names=('a','b'),
        declaration=SimpleNamespace(family='matmul_backward',schedule_consumer='test',tile_consumer='test',target_consumers={'rocm':'test'}),
        source_graph_ir=None,frontend_certificate=None)
    ones=np.ones((16,16),np.float32)
    if operands[0]=='%y': expected=(b.T.astype(np.float32)@ones,ones@a.T.astype(np.float32))
    else: expected=(ones@a.T.astype(np.float32)+a.T.astype(np.float32)@ones,np.zeros_like(b))
    for actual,want in zip(result.gradients,expected,strict=True):
        np.testing.assert_allclose(actual,want,atol=1e-4,rtol=1e-4)
    assert result.execution['evidence_target']=='rocm_gfx1201'
