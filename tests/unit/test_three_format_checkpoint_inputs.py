"""Bounded FP4 preparation must retain signed midpoint and actual-array semantics."""
import numpy as np
import pytest
from benchmarks.rocm import benchmark_gfx1201_three_formats as recorder


def _whole_array_oracle(b):
    k,n=b.shape
    blocks=b.T.reshape(n,k//32,32)
    exponents=np.maximum(np.ceil(np.log2(np.maximum(np.max(np.abs(blocks),axis=2),
                                                  np.finfo(np.float32).tiny)/6)).astype(np.int16),-126)
    normalized=blocks/np.exp2(exponents)[...,None]
    levels=np.asarray([0,.5,1,1.5,2,3,4,6],np.float32)
    distances=np.abs(np.abs(normalized)[...,None]-levels)
    ties=distances==distances.min(axis=-1,keepdims=True)
    priority=np.asarray([0,9,2,11,4,13,6,15])
    indices=np.argmin(np.where(ties,priority,100),axis=-1)
    return (indices+8*np.signbit(normalized)).astype(np.uint8).reshape(n,k),(exponents.T+127).astype(np.uint8)


@pytest.mark.parametrize("batch",[1,7,64,100])
def test_bounded_fp4_matches_whole_array_signed_midpoint_oracle(batch):
    rng=np.random.default_rng(0x1201256)
    b=rng.normal(size=(128,67)).astype(np.float32)
    # Fix max=6 to select scale=1; every midpoint tests nearest/even choice.
    tied=np.asarray([0,-0.,.25,.75,1.25,1.75,2.5,3.5,5,6],np.float32)
    b[:32,:2]=np.resize(tied,(32,2))
    b[31,:2]=6
    b[:31,1]*=-1
    expected_codes,expected_scales=_whole_array_oracle(b)
    codes,scales=recorder._quantize_fp4_rows(b,rows_per_batch=batch)
    np.testing.assert_array_equal(codes,expected_codes)
    np.testing.assert_array_equal(scales,expected_scales)


@pytest.mark.parametrize("shape,batch",[((31,2),64),((32,0),64),((32,2),0)])
def test_bounded_fp4_rejects_invalid_storage_envelopes(shape,batch):
    with pytest.raises(ValueError):
        recorder._quantize_fp4_rows(np.zeros(shape,np.float32),rows_per_batch=batch)


def test_actual_operand_evaluation_refuses_bad_shape_before_gpu():
    with pytest.raises(ValueError,match="compatible"):
        recorder.run_operands(np.zeros((128,128),np.float32),np.zeros((32,64),np.float32),None,None,None)


@pytest.mark.parametrize("e8m0",[False,True])
@pytest.mark.parametrize("group_k,group_n",[(128,128),(32,1)])
def test_checkpoint_column_major_source_materializes_identical_scale_planes(e8m0,group_k,group_n):
    rng=np.random.default_rng(0x1201257)
    a=rng.normal(size=(128,256)).astype(np.float32)
    b=rng.normal(size=(256,129)).astype(np.float32)
    row=recorder.quantize_fp8(a,b,group_k,group_n,e8m0=e8m0)
    column=recorder.quantize_fp8(a,np.asfortranarray(b),group_k,group_n,e8m0=e8m0)
    for actual,expected in zip(column[:2],row[:2],strict=True):
        np.testing.assert_array_equal(actual.view(np.uint8),expected.view(np.uint8))
    for actual,expected in zip(column[2],row[2],strict=True):
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual,expected)
    for actual,expected in zip(column[3:],row[3:],strict=True):
        np.testing.assert_array_equal(actual,expected)
