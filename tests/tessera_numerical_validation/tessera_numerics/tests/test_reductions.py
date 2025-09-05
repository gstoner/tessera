import numpy as np
import pytest
from tessera_numerics.validators import check_allclose
from tessera_numerics import tessera_adapter as A

@pytest.mark.parametrize("dtype", ["float32","float16","bfloat16"])
def test_sum_reorder_invariance(dtype):
    rng = A.rng(99)
    x = (rng.standard_normal((1024,))).astype(np.float64) * 1e3
    ref = np.sum(x, dtype=np.float64)
    # emulate different tilings/reorders
    got1 = np.sum(x.astype(dtype))
    got2 = np.sum(x.reshape(32,32).astype(dtype), axis=0).sum()
    got3 = np.sum(np.sort(x).astype(dtype))
    ref = np.array(ref, dtype=got1.dtype)
    check_allclose("sum", np.array(got1), ref, dtype=dtype)
    check_allclose("sum", np.array(got2), ref, dtype=dtype)
    check_allclose("sum", np.array(got3), ref, dtype=dtype)

def test_nan_inf_propagation():
    x = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -1.0], dtype=np.float64)
    s = np.sum(x)  # will be nan
    assert np.isnan(s), "NaN should propagate in sum"
