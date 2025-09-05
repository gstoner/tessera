import math
import numpy as np
from dataclasses import dataclass

@dataclass
class Tol:
    atol: float
    rtol: float
    ulp: int | None = None

DEFAULT_TOLS = {
    "float64": Tol(1e-12, 1e-9, 4),
    "float32": Tol(1e-6,  1e-5, 8),
    "float16": Tol(5e-3,  1e-2, 64),
    "bfloat16": Tol(6e-3, 1.5e-2, 64),
}

def ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute ULP distance for float32/float64."""
    if a.dtype == np.float32 or b.dtype == np.float32:
        ai = a.astype(np.float32).view(np.int32)
        bi = b.astype(np.float32).view(np.int32)
    else:
        ai = a.astype(np.float64).view(np.int64)
        bi = b.astype(np.float64).view(np.int64)
    # Handle sign: map to twos-complement ordering
    ai = np.where(ai < 0, 0x80000000 + ai if ai.dtype==np.int32 else 0x8000000000000000 + ai, ai)
    bi = np.where(bi < 0, 0x80000000 + bi if bi.dtype==np.int32 else 0x8000000000000000 + bi, bi)
    return np.abs(ai - bi)

def check_allclose(name, got, ref, dtype="float32", tol: Tol | None = None):
    got = np.asarray(got)
    ref = np.asarray(ref)
    if tol is None:
        tol = DEFAULT_TOLS.get(dtype, Tol(1e-6, 1e-5, None))
    # Basic NaN/Inf propagation check
    if np.isnan(ref).any():
        assert np.isnan(got)[np.isnan(ref)].all(), f"{name}: NaNs not preserved where expected"
    if np.isinf(ref).any():
        assert np.isinf(got)[np.isinf(ref)].all(), f"{name}: Infs not preserved where expected"
    # Value closeness
    np.testing.assert_allclose(got, ref, atol=tol.atol, rtol=tol.rtol, err_msg=f"{name}: allclose failed")
    # Optional ULP bound
    if tol.ulp is not None and got.dtype.kind == "f" and ref.dtype.kind == "f":
        dd = ulp_diff(got.astype(ref.dtype, copy=False), ref)
        max_ulp = int(dd.max()) if dd.size else 0
        assert max_ulp <= tol.ulp, f"{name}: ULP {max_ulp} > {tol.ulp}"

def finite_difference_grad(f, x, eps=1e-3):
    x = x.astype(np.float64, copy=True)
    grad = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        x[idx] = orig
        grad[idx] = (fp - fm) / (2*eps)
    return grad
