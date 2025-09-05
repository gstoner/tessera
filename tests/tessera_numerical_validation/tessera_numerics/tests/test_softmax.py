import numpy as np
import pytest
from tessera_numerics.validators import check_allclose
from tessera_numerics import tessera_adapter as A

def ref_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x, dtype=np.float64)
    return (ex / np.sum(ex, axis=axis, keepdims=True)).astype(np.float64)

@pytest.mark.parametrize("shape", [(32,1024), (4,12,2048)])
@pytest.mark.parametrize("dtype", ["float32","float16","bfloat16"])
def test_softmax_stable(cfg, rec, shape, dtype):
    axis = cfg.get("softmax", {}).get("axis", -1)
    rng = A.rng(7)
    x = (rng.standard_normal(shape) * 5.0).astype(np.float64)  # wide dynamic range
    got = A.softmax(x, axis=axis, dtype=dtype)
    ref = ref_softmax(x, axis=axis)
    check_allclose("softmax", got, ref.astype(got.dtype), dtype=dtype)
    # Probability simple sanity: sums to ~1
    s = np.sum(got, axis=axis)
    assert np.allclose(s, 1.0, atol=1e-3 if dtype!="float32" else 1e-5)
    rec("softmax", {"shape": shape, "dtype": dtype, "sum_mean": float(s.mean())})
