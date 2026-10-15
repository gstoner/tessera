"""Shared eager policy and canonical dtype inference, independent of GPU access."""
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.graph_ir import _infer_result_types,tensor_ir_type


@pytest.mark.parametrize('storage,accum',[(np.float16,'fp32'),(ml_dtypes.bfloat16,'fp32'),(np.float32,'fp64')])
def test_explicit_precision_and_storage_rounding(storage,accum):
    x=np.array([[64,-64,64,-64],[0,0,0,0]],dtype=storage)
    working=x.astype(np.float64 if accum=='fp64' else np.float32)
    expected=(working/np.sqrt(np.mean(working*working,axis=-1,keepdims=True)+1e-5)).astype(storage)
    result=ts.ops.rmsnorm(x,numeric_policy={'accum':accum})
    assert result.dtype==x.dtype
    np.testing.assert_array_equal(result,expected)
    gamma=np.array([1,2,3,4],dtype=storage)
    expected=((working/np.sqrt(np.mean(working*working,axis=-1,keepdims=True)+1e-5))*gamma.astype(working.dtype)).astype(storage)
    np.testing.assert_array_equal(ts.ops.rmsnorm(x,gamma,numeric_policy={'accum':accum}),expected)


@pytest.mark.parametrize('policy',[
    {'storage':'fp32','accum':'fp32'}, {'accum':'fp16'}, {'accum':'int32'},
    {'accum':'fp32','rounding':'stochastic'}, {'accum':'not_a_dtype'}, 'fp32',
])
def test_invalid_policy_does_not_silently_promote(policy):
    x=np.ones((2,8),dtype=ml_dtypes.bfloat16)
    with pytest.raises(ValueError):
        ts.ops.rmsnorm(x,numeric_policy=policy)


@pytest.mark.parametrize('dtype',['fp16','bf16','fp32'])
@pytest.mark.parametrize('output',['fp16','fp32'])
def test_matmul_canonical_result_type(dtype,output):
    args=[tensor_ir_type((2,32),dtype),tensor_ir_type((32,16),dtype)]
    result,=_infer_result_types('tessera.matmul',args,{'output_dtype':output})
    assert result.dtype==output and tuple(map(int,result.shape))==(2,16)


def test_apple_encode_does_not_silently_ignore_policy(monkeypatch):
    from tessera import apple_gpu_ops_interception as interception
    monkeypatch.setattr(interception,'_active_trace',lambda:object())
    x=np.ones((2,8),np.float16)
    with pytest.raises(ValueError,match='Apple encode RMSNorm does not support explicit numeric_policy'):
        interception._wrap_rmsnorm(lambda *a,**k:pytest.fail('unproved encode fallback'))(
            x,numeric_policy={'accum':'fp32'})


@pytest.mark.parametrize('storage',[np.float16,ml_dtypes.bfloat16,np.float32])
def test_omitted_policy_retains_existing_eager_precision(storage):
    x=np.array([[0.5,-1.0,2.0,-4.0]],dtype=storage)
    working=x.astype(np.float32)
    expected=(working/np.sqrt(np.mean(working*working,axis=-1,keepdims=True)+1e-5)).astype(storage)
    np.testing.assert_array_equal(ts.ops.rmsnorm(x),expected)
