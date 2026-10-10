"""Independent four-operand broadcast oracle; no compiled-support claim."""
import itertools
import numpy as np
import ml_dtypes
import pytest
from tessera.compiler.reference_typed_scaled_matmul import reference_typed_scaled_matmul

POLICY = {"accum": "fp32", "execution_mode": "exact_per_block"}
LAYOUT = {"granularity": "block", "block": [3, 4], "format": "fp32"}

def reference(values, ta, tb):
    return reference_typed_scaled_matmul(*values, numeric_policy=POLICY,
        scale_layout=LAYOUT, transposeA=ta, transposeB=tb, batching="broadcast")

def scalar_oracle(values, ta, tb, prefix):
    # Deliberate scalar indexing independent of matmul/broadcast accumulation.
    a,b,sa,sb = (np.asarray(v).astype(np.float64) for v in values)
    if ta: a=a.swapaxes(-1,-2)
    if tb: b=b.swapaxes(-1,-2)
    def read(x, plane, i, j):
        own=x.shape[:-2]
        padded=(1,)*(len(prefix)-len(own))+own
        index=tuple(0 if extent==1 else axis for extent,axis in zip(padded,plane))
        index=index[len(prefix)-len(own):]
        return x[(*index,i,j)]
    out=np.zeros((*prefix,3,5),np.float64)
    for plane in np.ndindex(prefix):
        for i,j,k in itertools.product(range(3),range(5),range(7)):
            out[(*plane,i,j)] += (read(a,plane,i,k)*read(b,plane,k,j)*
                read(sa,plane,i,k//4)*read(sb,plane,k//4,j//3))
    return out

def inputs(prefixes, ta, tb):
    rng=np.random.default_rng(7107)
    shapes=((7,3) if ta else (3,7),(5,7) if tb else (7,5),(3,2),(2,2))
    return tuple((rng.uniform(-.5,.5,size=(*prefix,*shape)).astype(ml_dtypes.float8_e4m3fn)
                  if index<2 else rng.uniform(.3,1.3,size=(*prefix,*shape)).astype(np.float32))
                 for index,(prefix,shape) in enumerate(zip(prefixes,shapes)))

@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
def test_all_independent_mapped_shared_operands(mask,ta,tb):
    prefixes=tuple((2,3) if mask & (1<<i) else () for i in range(4))
    values=inputs(prefixes,ta,tb)
    expected=scalar_oracle(values,ta,tb,(2,3))
    np.testing.assert_allclose(reference(values,ta,tb),expected,rtol=2e-6,atol=2e-7)
    cot=np.random.default_rng(17).uniform(-1,1,size=expected.shape)
    # Every scale coordinate: scalar finite differences independently check
    # reduction over shared batch axes and ragged K/N scale boundaries.
    for operand in (2,3):
        for index in np.ndindex(values[operand].shape):
            plus=list(values);minus=list(values)
            plus[operand]=values[operand].copy();minus[operand]=values[operand].copy()
            plus[operand][index]+=.02;minus[operand][index]-=.02
            delta=float(plus[operand][index]-minus[operand][index])
            actual=np.sum((reference(plus,ta,tb).astype(np.float64)-
                           reference(minus,ta,tb).astype(np.float64))*cot)/delta
            wanted=np.sum((scalar_oracle(plus,ta,tb,(2,3))-
                           scalar_oracle(minus,ta,tb,(2,3)))*cot)/delta
            assert actual==pytest.approx(wanted,rel=4e-4,abs=2e-5)

@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
def test_independent_singleton_and_rank_broadcast(ta,tb):
    values=inputs(((2,1),(3,),(),(1,3)),ta,tb)
    np.testing.assert_allclose(reference(values,ta,tb),
        scalar_oracle(values,ta,tb,(2,3)),rtol=2e-6,atol=2e-7)

def test_incompatible_prefixes_are_not_flattened():
    values=inputs(((2,3),(3,2),(),()),False,False)
    with pytest.raises(ValueError,match="prefixes do not broadcast"):
        reference(values,False,False)
