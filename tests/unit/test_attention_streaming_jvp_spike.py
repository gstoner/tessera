"""Host reference algorithm validation, not a native tangent capability."""
import numpy as np
import pytest
from tools.attention_jvp_spike import streaming_row_jvp


@pytest.mark.parametrize('active',['q','k','qk','qkv'])
@pytest.mark.parametrize('block',[1,3,8,32])
def test_streamed_direction_matches_independent_dense_difference(active,block):
    rng=np.random.default_rng(709)
    q=rng.normal(size=4)
    k=rng.normal(size=(17,4))
    v=rng.normal(size=(17,3))
    dq=rng.normal(size=q.shape) if 'q' in active else np.zeros_like(q)
    dk=rng.normal(size=k.shape) if 'k' in active else np.zeros_like(k)
    dv=rng.normal(size=v.shape) if 'v' in active else np.zeros_like(v)
    mask=np.arange(17)<=9
    def dense(epsilon):
        scores=.5*((k+epsilon*dk)@(q+epsilon*dq))
        weights=np.exp(scores[mask]-scores[mask].max())
        return weights@(v+epsilon*dv)[mask]/weights.sum()
    primal,tangent,_=streaming_row_jvp(q,k,v,dq,dk,dv,scale=.5,mask=mask,block_size=block)
    np.testing.assert_allclose(primal,dense(0),atol=1e-12)
    np.testing.assert_allclose(tangent,(dense(1e-5)-dense(-1e-5))/2e-5,atol=2e-9,rtol=2e-8)


def test_empty_mask_and_max_rescaling():
    q=np.array([1.])
    k=np.array([[-1000.],[0.],[1000.]])
    v=np.array([[1.,2.],[3.,4.],[5.,6.]])
    dq=np.array([.1]); dk=k*.1; dv=v*.1
    for block in (1,2,3):
        p,t,lse=streaming_row_jvp(q,k,v,dq,dk,dv,scale=1.,mask=np.ones(3,bool),block_size=block)
        np.testing.assert_allclose(p,v[-1])
        np.testing.assert_allclose(t,dv[-1],atol=1e-12)
        assert lse==1000
        p,t,lse=streaming_row_jvp(q,k,v,dq,dk,dv,scale=1.,mask=np.zeros(3,bool),block_size=block)
        assert not p.any() and not t.any() and np.isneginf(lse)
