"""GPU linear-algebra lane — Cholesky / LU / triangular solve via MPSMatrix.

This is the one capability MPSGraph cannot provide (no matrix-decomposition ops),
so these dense f32 factorizations/solves are the only GPU path for
``tessera.ops.{cholesky, solve, cholesky_solve, tri_solve}`` — previously
numpy/CPU only. Correctness holds on both the GPU path (Metal present) and the
numpy fallback; ``ran_on_gpu`` is asserted True only when a Metal device exists.
"""

import numpy as np
import pytest

import tessera.runtime as R

TOL = 2e-4  # f32 factorization/solve vs an f64 numpy reference


def _on_metal() -> bool:
    return R.DeviceTensor.is_metal()


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)).astype(np.float32)
    return (M @ M.T + n * np.eye(n)).astype(np.float32)


def _wellcond(n, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, n)).astype(np.float32)
            + n * np.eye(n, dtype=np.float32))


def _rel(x, ref):
    ref = np.asarray(ref, np.float64)
    return float(np.abs(np.asarray(x, np.float64) - ref).max()
                 / (np.abs(ref).max() + 1e-12))


# ── Cholesky factorization ────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [1, 2, 5, 16, 64])
def test_cholesky_matches_numpy(n):
    A = _spd(n, n)
    L, ran = R.apple_gpu_cholesky(A, np)
    assert L.shape == (n, n) and L.dtype == np.float32
    # lower-triangular (strict upper zeroed, matching numpy)
    assert np.allclose(np.triu(L, 1), 0.0)
    assert _rel(L, np.linalg.cholesky(A.astype(np.float64))) < TOL
    assert _rel(L @ L.T, A) < TOL
    if _on_metal():
        assert ran is True


def test_cholesky_non_pd_falls_back_and_raises():
    # A non-positive-definite matrix: GPU returns a failure code -> numpy
    # fallback, which raises LinAlgError (same as a pure-numpy call would).
    A = np.array([[1.0, 2.0], [2.0, 1.0]], np.float32)  # eigenvalues 3, -1
    with pytest.raises(np.linalg.LinAlgError):
        R.apple_gpu_cholesky(A, np)


# ── SPD solve via Cholesky ────────────────────────────────────────────────────
@pytest.mark.parametrize("n,nrhs", [(4, 1), (8, 3), (32, 5)])
def test_cholesky_solve_matches_numpy(n, nrhs):
    A = _spd(n, n + nrhs)
    B = np.random.default_rng(n).standard_normal((n, nrhs)).astype(np.float32)
    X, ran = R.apple_gpu_cholesky_solve(A, B, np)
    assert X.shape == (n, nrhs)
    assert _rel(X, np.linalg.solve(A.astype(np.float64), B.astype(np.float64))) < TOL
    assert _rel(A @ X, B) < TOL
    if _on_metal():
        assert ran is True


# ── General solve via LU ──────────────────────────────────────────────────────
@pytest.mark.parametrize("n,nrhs", [(4, 1), (8, 3), (48, 4)])
def test_solve_lu_matches_numpy(n, nrhs):
    A = _wellcond(n, n * 7 + nrhs)
    B = np.random.default_rng(n + 1).standard_normal((n, nrhs)).astype(np.float32)
    X, ran = R.apple_gpu_solve(A, B, np)
    assert X.shape == (n, nrhs)
    assert _rel(X, np.linalg.solve(A.astype(np.float64), B.astype(np.float64))) < TOL
    if _on_metal():
        assert ran is True


def test_solve_vector_rhs_returns_vector():
    n = 10
    A = _wellcond(n, 3)
    b = np.random.default_rng(9).standard_normal(n).astype(np.float32)
    x, ran = R.apple_gpu_solve(A, b, np)
    assert x.shape == (n,)
    assert _rel(x, np.linalg.solve(A.astype(np.float64), b.astype(np.float64))) < TOL


# ── Triangular solve ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("unit", [False, True])
def test_tri_solve_matches_numpy(lower, trans, unit):
    n, nrhs = 12, 3
    rng = np.random.default_rng(int(lower) * 4 + int(trans) * 2 + int(unit))
    base = rng.standard_normal((n, n)).astype(np.float32)
    A = (base + n * np.eye(n, dtype=np.float32))
    B = rng.standard_normal((n, nrhs)).astype(np.float32)
    X, ran = R.apple_gpu_tri_solve(A, B, np, lower=lower, trans=trans, unit=unit)
    tri = np.tril(A) if lower else np.triu(A)
    tri = tri.astype(np.float64)
    if unit:
        np.fill_diagonal(tri, 1.0)
    if trans:
        tri = tri.T
    assert _rel(X, np.linalg.solve(tri, B.astype(np.float64))) < TOL
    if _on_metal():
        assert ran is True


def test_tri_solve_vector_rhs():
    n = 9
    rng = np.random.default_rng(2)
    A = np.tril(rng.standard_normal((n, n)).astype(np.float32)) + n * np.eye(n, dtype=np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    x, _ = R.apple_gpu_tri_solve(A, b, np, lower=True)
    assert x.shape == (n,)
    assert _rel(x, np.linalg.solve(np.tril(A).astype(np.float64), b.astype(np.float64))) < TOL


# ── Batched inputs fall back to numpy (still correct) ─────────────────────────
def test_batched_cholesky_falls_back_correct():
    A = np.stack([_spd(5, k) for k in range(3)])  # [3, 5, 5]
    L, ran = R.apple_gpu_cholesky(A, np)
    assert L.shape == (3, 5, 5)
    assert ran is False  # rank-3: numpy fallback
    assert _rel(L, np.linalg.cholesky(A.astype(np.float64))) < TOL


def test_batched_solve_falls_back_correct():
    A = np.stack([_wellcond(6, k) for k in range(4)])  # [4, 6, 6]
    B = np.random.default_rng(0).standard_normal((4, 6, 2)).astype(np.float32)
    X, ran = R.apple_gpu_solve(A, B, np)
    assert X.shape == (4, 6, 2)
    assert ran is False
    assert _rel(X, np.linalg.solve(A.astype(np.float64), B.astype(np.float64))) < TOL
