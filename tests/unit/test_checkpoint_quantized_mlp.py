"""Independent full-layer checkpoint reference and boundary contracts."""
from types import SimpleNamespace
import ml_dtypes
import numpy as np
import pytest
from benchmarks.rocm import benchmark_checkpoint_quantized_mlp as recorder


def test_decode_retains_nibble_order_globals_and_negative_zero():
    projection=SimpleNamespace(packed_codes=np.array([[0x81,0xF7,0,0,0,0,0,0]],np.uint8),
        e4m3_scales=np.array([[2]],dtype=ml_dtypes.float8_e4m3fn),global_scale=.25)
    result=recorder.decode_projection(projection)
    np.testing.assert_array_equal(result[:,:4],np.array([[.25,-0.,3.,-3.]]))
    assert np.signbit(result[0,1])


def test_layer_reference_uses_distinct_gate_up_and_down_orientations():
    x=np.array([[1.,2.],[-1.,.5]],np.float32)
    gate=np.array([[1.,0.],[0.,1.],[1.,-1.]],np.float32)
    up=np.array([[2.,0.],[0.,3.],[-1.,1.]],np.float32)
    down=np.array([[1.,2.,3.],[-2.,1.,.5]],np.float32)
    g=np.einsum('mk,hk->mh',x.astype(np.float64),gate).astype(ml_dtypes.bfloat16).astype(np.float32)
    u=np.einsum('mk,hk->mh',x.astype(np.float64),up).astype(ml_dtypes.bfloat16).astype(np.float32)
    a=(g/(1+np.exp(-g))*u).astype(ml_dtypes.bfloat16)
    expected=np.einsum('mh,nh->mn',a.astype(np.float64),down).astype(np.float32).astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(recorder.layer_reference(x,gate,up,down),expected)


def test_layer_reference_handles_extreme_negative_silu():
    with np.errstate(over='raise',invalid='raise'):
        result=recorder.layer_reference(np.array([[1.]]),np.array([[-1000.]]),np.array([[1.]]),np.array([[1.]]))
    assert result[0,0]==0


@pytest.mark.parametrize('rows,repeats',[([],1),([0],1),([True],1),([1],0)])
def test_bad_counts_rejected_before_checkpoint_or_device(monkeypatch,rows,repeats):
    monkeypatch.setattr(recorder,'load_capture',lambda *a:pytest.fail('invalid arguments reached capture'))
    with pytest.raises(ValueError,match='positive'):
        recorder.measure(None,rows,None,repeats)


def test_reference_rejects_transposed_down():
    with pytest.raises(ValueError,match='shapes'):
        recorder.layer_reference(np.ones((1,2)),np.ones((3,2)),np.ones((3,2)),np.ones((3,4)))
