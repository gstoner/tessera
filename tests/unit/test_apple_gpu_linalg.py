"""GPU linear-algebra lane — Cholesky / LU / triangular solve via MPSMatrix.

This is the one capability MPSGraph cannot provide (no matrix-decomposition ops),
so these dense f32 factorizations/solves are the only GPU path for
``tessera.ops.{cholesky, solve, cholesky_solve, tri_solve}`` — previously
numpy/CPU only. Correctness holds on both the GPU path (Metal present) and the
numpy fallback; ``ran_on_gpu`` is asserted True only when a Metal device exists.
"""

import numpy as np
import pytest

import tessera as ts
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


# ── Batched inputs (ndim>2) — per-matrix GPU loop, else numpy ─────────────────
def test_batched_cholesky():
    A = np.stack([_spd(5, k) for k in range(3)])  # [3, 5, 5]
    L, ran = R.apple_gpu_cholesky(A, np)
    assert L.shape == (3, 5, 5)
    assert _rel(L, np.linalg.cholesky(A.astype(np.float64))) < TOL
    if _on_metal():
        assert ran is True


def test_batched_solve_matrix_rhs():
    A = np.stack([_wellcond(6, k) for k in range(4)])  # [4, 6, 6]
    B = np.random.default_rng(0).standard_normal((4, 6, 2)).astype(np.float32)
    X, ran = R.apple_gpu_solve(A, B, np)
    assert X.shape == (4, 6, 2)
    ref = np.stack([np.linalg.solve(A[k].astype(np.float64), B[k].astype(np.float64))
                    for k in range(4)])
    assert _rel(X, ref) < TOL
    if _on_metal():
        assert ran is True


def test_batched_solve_vector_rhs():
    A = np.stack([_wellcond(6, k) for k in range(4)])      # [4, 6, 6]
    b = np.random.default_rng(1).standard_normal((4, 6)).astype(np.float32)  # [4, 6]
    x, ran = R.apple_gpu_solve(A, b, np)
    assert x.shape == (4, 6)
    ref = np.stack([np.linalg.solve(A[k].astype(np.float64), b[k].astype(np.float64))
                    for k in range(4)])
    assert _rel(x, ref) < TOL
    if _on_metal():
        assert ran is True


def test_batched_cholesky_solve():
    A = np.stack([_spd(7, k) for k in range(5)])           # [5, 7, 7] SPD
    B = np.random.default_rng(2).standard_normal((5, 7, 3)).astype(np.float32)
    X, ran = R.apple_gpu_cholesky_solve(A, B, np)
    assert X.shape == (5, 7, 3)
    ref = np.stack([np.linalg.solve(A[k].astype(np.float64), B[k].astype(np.float64))
                    for k in range(5)])
    assert _rel(X, ref) < TOL
    if _on_metal():
        assert ran is True


def test_batched_tri_solve():
    rng = np.random.default_rng(6)
    A = np.stack([np.tril(rng.standard_normal((8, 8)).astype(np.float32))
                  + 8 * np.eye(8, dtype=np.float32) for _ in range(3)])  # [3, 8, 8]
    B = rng.standard_normal((3, 8, 2)).astype(np.float32)
    X, ran = R.apple_gpu_tri_solve(A, B, np, lower=True)
    ref = np.stack([np.linalg.solve(np.tril(A[k]).astype(np.float64),
                                    B[k].astype(np.float64)) for k in range(3)])
    assert _rel(X, ref) < TOL
    if _on_metal():
        assert ran is True


# ── @jit(target="apple_gpu") dispatch — reachable from model code ─────────────
@ts.jit(target="apple_gpu")
def _jit_cholesky(A):
    return ts.ops.cholesky(A)


@ts.jit(target="apple_gpu")
def _jit_tri_solve(A, b):
    return ts.ops.tri_solve(A, b, lower=True)


def test_jit_linalg_runtime_executable():
    """The linalg ops are admitted to the apple_gpu runtime envelope (single-op
    metal_runtime on Darwin, metal_artifact otherwise)."""
    _jit_cholesky(_spd(6, 0))
    _jit_tri_solve(np.tril(_wellcond(6, 1)), np.zeros((6, 2), np.float32))
    for fn in (_jit_cholesky, _jit_tri_solve):
        meta = fn.runtime_artifact().metadata
        assert meta["execution_mode"] in ("metal_runtime", "metal_artifact")


def test_jit_cholesky_matches_numpy():
    A = _spd(8, 4)
    L = np.asarray(_jit_cholesky(A))
    assert _rel(L, np.linalg.cholesky(A.astype(np.float64))) < TOL
    assert _rel(L @ L.T, A) < TOL


def test_jit_tri_solve_matches_numpy():
    A = np.tril(_wellcond(8, 5)).astype(np.float32)
    b = np.random.default_rng(5).standard_normal((8, 3)).astype(np.float32)
    X = np.asarray(_jit_tri_solve(A, b))
    assert _rel(X, np.linalg.solve(np.tril(A).astype(np.float64), b.astype(np.float64))) < TOL


# ── f16 / bf16 / f64 dtype policy ─────────────────────────────────────────────
@pytest.mark.parametrize("npdt,tol", [(np.float16, 3e-3), (np.float32, TOL)])
def test_cholesky_dtype_preserved(npdt, tol):
    A = _spd(8, 1).astype(npdt)
    L, ran = R.apple_gpu_cholesky(A, np)
    assert L.dtype == np.dtype(npdt)              # input float dtype preserved
    ref = np.linalg.cholesky(_spd(8, 1).astype(np.float64))
    assert _rel(L.astype(np.float64), ref) < tol
    if _on_metal():
        assert ran is True


def test_cholesky_bf16_preserved():
    ml = pytest.importorskip("ml_dtypes")
    A = _spd(8, 1).astype(ml.bfloat16)
    L, ran = R.apple_gpu_cholesky(A, np)
    assert L.dtype == np.dtype(ml.bfloat16)
    ref = np.linalg.cholesky(_spd(8, 1).astype(np.float64))
    assert _rel(L.astype(np.float64), ref) < 5e-2


def test_f64_stays_on_numpy_full_precision():
    # f64 must NOT silently downcast to f32 on the GPU — it routes to numpy and
    # keeps full double precision.
    A = _spd(8, 1).astype(np.float64)
    L, ran = R.apple_gpu_cholesky(A, np)
    assert L.dtype == np.float64
    assert ran is False
    assert _rel(L, np.linalg.cholesky(A)) < 1e-12   # f64-accurate


def test_f16_solve_and_tri_solve():
    A = _wellcond(8, 2).astype(np.float16)
    b = np.random.default_rng(3).standard_normal((8, 2)).astype(np.float16)
    X, ran = R.apple_gpu_solve(A, b, np)
    assert X.dtype == np.float16
    ref = np.linalg.solve(_wellcond(8, 2).astype(np.float64),
                          b.astype(np.float64))
    assert _rel(X.astype(np.float64), ref) < 5e-3
    At = np.tril(_wellcond(8, 4)).astype(np.float16)
    Xt, _ = R.apple_gpu_tri_solve(At, b, np, lower=True)
    assert Xt.dtype == np.float16


# ── QR via Cholesky-QR (GPU) with verified Householder fallback ───────────────
@pytest.mark.parametrize("m,n", [(8, 8), (16, 4), (64, 32), (128, 16)])
def test_qr_well_conditioned_on_gpu(m, n):
    A = np.random.default_rng(m + n).standard_normal((m, n)).astype(np.float32)
    Q, Rm, ran = R.apple_gpu_qr(A, np)
    assert Q.shape == (m, n) and Rm.shape == (n, n)
    assert _rel(Q @ Rm, A) < 1e-4                       # reconstruction
    assert float(np.abs(Q.T @ Q - np.eye(n)).max()) < 1e-3   # orthonormal
    assert float(np.abs(np.tril(Rm, -1)).max()) == 0.0       # R upper-triangular
    if _on_metal():
        assert ran is True


def test_qr_ill_conditioned_falls_back_orthonormal():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((20, 5)).astype(np.float32)
    A[:, 4] = A[:, 0] + 1e-7 * A[:, 4]   # near rank-deficient -> Cholesky-QR fails
    Q, Rm, ran = R.apple_gpu_qr(A, np)
    assert ran is False                  # verification rejected the GPU result
    assert _rel(Q @ Rm, A) < 1e-4
    assert float(np.abs(Q.T @ Q - np.eye(5)).max()) < 1e-4   # still orthonormal


def test_qr_f16_dtype_preserved():
    A = np.random.default_rng(1).standard_normal((32, 8)).astype(np.float16)
    Q, Rm, ran = R.apple_gpu_qr(A, np)
    assert Q.dtype == np.float16 and Rm.dtype == np.float16
    assert _rel(Q.astype(np.float32) @ Rm.astype(np.float32),
                A.astype(np.float32)) < 5e-3


# ── SVD via one-sided Jacobi (custom MSL) ─────────────────────────────────────
@pytest.mark.parametrize("m,n", [(8, 8), (16, 4), (64, 32), (128, 16), (50, 50)])
def test_svd_matches_numpy(m, n):
    A = np.random.default_rng(m * 3 + n).standard_normal((m, n)).astype(np.float32)
    U, S, Vh, ran = R.apple_gpu_svd(A, np)
    assert U.shape == (m, n) and S.shape == (n,) and Vh.shape == (n, n)
    assert _rel((U * S) @ Vh, A) < 1e-4                       # reconstruction
    assert float(np.abs(U.T @ U - np.eye(n)).max()) < 1e-3    # U orthonormal cols
    assert float(np.abs(Vh @ Vh.T - np.eye(n)).max()) < 1e-3  # V orthonormal
    assert np.all(np.diff(S) <= 1e-4)                         # descending
    s_np = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    assert float(np.abs(S.astype(np.float64) - s_np).max() / s_np.max()) < 1e-3
    if _on_metal():
        assert ran is True


def test_svd_rank_deficient():
    rng = np.random.default_rng(1)
    A = (rng.standard_normal((20, 2)) @ rng.standard_normal((2, 6))).astype(np.float32)
    U, S, Vh, _ = R.apple_gpu_svd(A, np)
    assert _rel((U * S) @ Vh, A) < 1e-4
    assert float(np.abs(S[2:]).max()) < 1e-3                  # rank 2 -> tail σ ≈ 0


def test_svd_clustered_singular_values():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 4)))
    A = (Q * np.array([3.0, 3.0, 3.0, 1.0])).astype(np.float32)
    U, S, Vh, _ = R.apple_gpu_svd(A, np)
    assert _rel((U * S) @ Vh, A) < 1e-4
    np.testing.assert_allclose(np.sort(S)[::-1], [3, 3, 3, 1], atol=1e-3)


def test_svd_f16_dtype_preserved():
    A = np.random.default_rng(2).standard_normal((32, 8)).astype(np.float16)
    U, S, Vh, ran = R.apple_gpu_svd(A, np)
    assert U.dtype == np.float16 and S.dtype == np.float16 and Vh.dtype == np.float16
    assert _rel((U.astype(np.float32) * S.astype(np.float32)) @ Vh.astype(np.float32),
                A.astype(np.float32)) < 5e-3


def test_svd_wide_matrix_falls_back():
    # m < n is outside the GPU path (m >= n only) -> numpy, still correct.
    A = np.random.default_rng(3).standard_normal((4, 10)).astype(np.float32)
    U, S, Vh, ran = R.apple_gpu_svd(A, np)
    assert ran is False
    assert _rel((U * S) @ Vh, A) < TOL


def test_svd_full_matrices_falls_back():
    A = np.random.default_rng(4).standard_normal((8, 5)).astype(np.float32)
    U, S, Vh, ran = R.apple_gpu_svd(A, np, full_matrices=True)
    assert ran is False
    assert U.shape == (8, 8) and Vh.shape == (5, 5)
