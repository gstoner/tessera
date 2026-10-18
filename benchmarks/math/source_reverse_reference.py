"""Independent FP64 source-pullback oracle with explicit half edge stores.

Shared by owning NVIDIA/ROCm tests; no backend execution evidence transfers.
The finite Gaussian regimes avoid ReLU storage-underflow branch ambiguity.
"""
import numpy as np


def inputs(storage, permuted, seed=87, witness=False, shape=(17, 19, 35)):
    import ml_dtypes
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    m, n, k = shape
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .75, (m, k)).astype(dtype)
    b = rng.normal(0, .2, (k, n)).astype(dtype)
    dy = rng.normal(0, .2, (m, n)).astype(np.float32)
    if witness:
        if n < 2:
            raise ValueError("FP32 cotangent witness requires two seed columns")
        b.fill(0)
        b[:, :2] = 1
        dy.fill(0)
        dy[:, 0], dy[:, 1] = 1 + 2.0**(-12 if storage == "f16" else -9), -1
    return (b, x, dy) if permuted else (x, b, dy)


def oracle(roots, permuted):
    x, b, dy = (roots[1], roots[0], roots[2]) if permuted else roots
    xx, bb, seed = (v.astype(np.float64) for v in (x, b, dy))
    wide = seed @ bb.T
    # The native AD Graph stores the matmul adjoint to the primal half dtype.
    edge = wide.astype(x.dtype).astype(np.float64)
    g = np.where(xx > 0, edge, 0)
    r = 1 / np.sqrt(np.mean(xx * xx, axis=1, keepdims=True) + 1e-5)
    reference = g * r - xx * r**3 * np.mean(g * xx, axis=1, keepdims=True)
    rounded = reference.astype(x.dtype)
    ulp = np.abs(np.nextafter(rounded, x.dtype.type(np.inf)).astype(np.float64) -
                 rounded.astype(np.float64))
    products = seed.shape[1]
    gamma = products * np.finfo(np.float32).eps / (1 - products * np.finfo(np.float32).eps)
    edge_ulp = np.abs(np.nextafter(wide.astype(x.dtype), x.dtype.type(np.inf)).astype(np.float64) -
                      wide.astype(x.dtype).astype(np.float64))
    edge_budget = gamma * (np.abs(seed) @ np.abs(bb.T)) + edge_ulp
    propagated = r * edge_budget + np.abs(xx) * r**3 * np.mean(
        np.abs(xx) * edge_budget, axis=1, keepdims=True)
    budget = propagated + xx.shape[1] * 32 * np.finfo(np.float32).eps * (np.abs(reference) + .01) + ulp
    return reference, budget


def check(actual, roots, permuted):
    reference, budget = oracle(roots, permuted)
    assert len(actual) == 1 and actual[0].dtype == roots[int(permuted)].dtype
    assert np.isfinite(actual[0]).all()
    assert np.all(np.abs(actual[0].astype(np.float64) - reference) <= budget)
