"""L-series linalg family (LF1–LF5) — Apple CPU runtime conformance.

The 5 next linalg members (tri_solve, cholesky_solve, lu, qr, svd) each lower
through the table-driven Graph→Schedule→Tile→Target Apple spine and execute on
the Apple CPU runtime via Accelerate LAPACK.  This compiles the runtime shim
and ABI-tests every symbol against numpy — the executable foundation behind the
compiler lowering (the lowering itself is locked by the lit fixtures + the
seam-closure / drift tests).
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_SRC = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_cpu_runtime.cpp"

_F = ctypes.POINTER(ctypes.c_float)
_I = ctypes.c_int32


def _ptr(a):
    return a.ctypes.data_as(_F)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")
    lib = tmp_path_factory.mktemp("rt") / (
        "librt.dylib" if sys.platform == "darwin" else "librt.so")
    cmd = [cxx, "-std=c++17", "-O2", "-shared", "-fPIC", str(_SRC), "-o", str(lib)]
    if sys.platform == "darwin":
        cmd += ["-framework", "Accelerate"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    from tests._support.apple import require_apple_accelerate

    require_apple_accelerate()
    return ctypes.CDLL(str(lib))


def _spd(n, rng):
    m = rng.standard_normal((n, n)).astype(np.float32)
    return (m @ m.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)


def test_tri_solve(runtime):
    fn = runtime.tessera_apple_cpu_tri_solve_f32
    fn.argtypes = [_F, _F, _F, _I, _I, _I, _I, _I]
    fn.restype = _I
    rng = np.random.default_rng(1)
    for n, k in ((4, 1), (5, 3), (16, 4)):
        L = (np.tril(rng.standard_normal((n, n)).astype(np.float32))
             + n * np.eye(n, dtype=np.float32))
        B = rng.standard_normal((n, k)).astype(np.float32)
        X = np.zeros((n, k), np.float32)
        assert fn(_ptr(L), _ptr(B), _ptr(X), n, k, 1, 0, 0) == 0
        np.testing.assert_allclose(L @ X, B, rtol=1e-3, atol=1e-3)


def test_cholesky_solve(runtime):
    fn = runtime.tessera_apple_cpu_cholesky_solve_f32
    fn.argtypes = [_F, _F, _F, _I, _I, _I]
    fn.restype = _I
    rng = np.random.default_rng(2)
    for n, k in ((4, 1), (8, 3), (16, 5)):
        A = _spd(n, rng)
        B = rng.standard_normal((n, k)).astype(np.float32)
        X = np.zeros((n, k), np.float32)
        assert fn(_ptr(A), _ptr(B), _ptr(X), n, k, 1) == 0
        np.testing.assert_allclose(A @ X, B, rtol=1e-3, atol=1e-3)


def test_lu(runtime):
    fn = runtime.tessera_apple_cpu_lu_f32
    fn.argtypes = [_F, _F, ctypes.POINTER(_I), _I]
    fn.restype = _I
    rng = np.random.default_rng(3)
    for n in (4, 8, 16):
        A = _spd(n, rng)  # nonsingular
        LU = np.zeros((n, n), np.float32)
        piv = np.zeros(n, np.int32)
        assert fn(_ptr(A), _ptr(LU), piv.ctypes.data_as(ctypes.POINTER(_I)), n) == 0
        Lm = np.tril(LU, -1) + np.eye(n, dtype=np.float32)
        Um = np.triu(LU)
        Ap = A.copy()
        for i in range(n):  # apply 1-based ipiv sequential row swaps
            j = int(piv[i]) - 1
            Ap[[i, j]] = Ap[[j, i]]
        np.testing.assert_allclose(Ap, Lm @ Um, rtol=1e-3, atol=1e-3)


def test_qr(runtime):
    fn = runtime.tessera_apple_cpu_qr_f32
    fn.argtypes = [_F, _F, _F, _I, _I]
    fn.restype = _I
    rng = np.random.default_rng(4)
    for m, n in ((4, 4), (6, 4), (10, 3)):
        A = rng.standard_normal((m, n)).astype(np.float32)
        Q = np.zeros((m, n), np.float32)
        R = np.zeros((n, n), np.float32)
        assert fn(_ptr(A), _ptr(Q), _ptr(R), m, n) == 0
        np.testing.assert_allclose(Q @ R, A, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(Q.T @ Q, np.eye(n), rtol=1e-3, atol=1e-3)
        assert np.allclose(np.tril(R, -1), 0.0, atol=1e-4)  # R upper-triangular


def test_svd(runtime):
    fn = runtime.tessera_apple_cpu_svd_f32
    fn.argtypes = [_F, _F, _F, _F, _I, _I]
    fn.restype = _I
    rng = np.random.default_rng(5)
    for m, n in ((4, 4), (6, 4), (4, 6)):
        k = min(m, n)
        A = rng.standard_normal((m, n)).astype(np.float32)
        U = np.zeros((m, k), np.float32)
        S = np.zeros(k, np.float32)
        V = np.zeros((k, n), np.float32)
        assert fn(_ptr(A), _ptr(U), _ptr(S), _ptr(V), m, n) == 0
        np.testing.assert_allclose(U @ np.diag(S) @ V, A, rtol=1e-3, atol=1e-3)
        assert np.all(S[:-1] >= S[1:] - 1e-4)  # descending singular values
