import numpy as np
import pytest
from tessera_numerics.validators import check_allclose, Tol
from tessera_numerics import tessera_adapter as A

def ref_matmul(a, b):
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float64)

@pytest.mark.parametrize("shape", [(64,96,80), (128,128,128)])
@pytest.mark.parametrize("dtype", ["float32","float16","bfloat16"])
def test_matmul_basic(cfg, rec, shape, dtype):
    if "matmul" not in cfg: pytest.skip("matmul not configured")
    M,K,N = shape
    rng = A.rng(42)
    a = rng.standard_normal((M,K)).astype(np.float64) * 0.1
    b = rng.standard_normal((K,N)).astype(np.float64) * 0.1

    got = A.matmul(a, b, dtype=dtype)
    ref = ref_matmul(a, b)
    tol_cfg = cfg["matmul"]["tol"].get(dtype, {})
    tol = Tol(**tol_cfg) if tol_cfg else None
    check_allclose("matmul", got, ref.astype(got.dtype), dtype=dtype, tol=tol)
    rec("matmul", {"shape": shape, "dtype": dtype, "max": float(np.max(np.abs(got))) })
