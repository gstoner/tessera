import numpy as np

def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated points. points: [N, M]."""
    N, M = points.shape
    mask = np.ones(N, dtype=bool)
    for i in range(N):
        if not mask[i]:
            continue
        for j in range(N):
            if i == j: continue
            ge = np.all(points[j] >= points[i])
            gt = np.any(points[j] >  points[i])
            if ge and gt:
                mask[i] = False
                break
    return mask

def scalarize_linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x @ w

def scalarize_tchebycheff(x: np.ndarray, w: np.ndarray, z: np.ndarray, eps=1e-3) -> np.ndarray:
    d = np.abs(x - z)
    return -(np.max(w * d, axis=-1) + eps * np.sum(w * d, axis=-1))

def pcgrad_pairwise(grads: np.ndarray) -> np.ndarray:
    """grads: [M, D] -> combined [D] with PCGrad."""
    M, D = grads.shape
    g = np.zeros(D, dtype=grads.dtype)
    for m in range(M):
        g_m = grads[m].copy()
        for n in range(M):
            if n == m: continue
            dot = float(np.dot(g_m, grads[n]))
            if dot < 0:
                denom = float(np.dot(grads[n], grads[n])) + 1e-8
                g_m = g_m - (dot/denom) * grads[n]
        g += g_m
    return g / M
