"""waitUntilCompleted → Pattern-4 timeout-event migration (batch 1).

The audit (May 2026) flagged ~60 sites in ``apple_gpu_runtime.mm``
still using the legacy::

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return ...;

pattern. The replacement is the Apple-sample Pattern 4 helper
``commit_and_wait_with_timeout`` already in place for the canonical
MTL4 lane: it ENCODES a shared-event signal into the command buffer,
commits, then waits with a timeout. Migrated sites get:

* **30 second timeout protection** — a hung kernel no longer wedges
  the entire test process; the dispatcher returns ``-1`` / ``false``
  with a precise diagnostic.
* **Named diagnostic on timeout** — ``"[tessera_apple_gpu] <op_name>:
  GPU dispatch did not signal within 30000 ms ..."`` instead of
  ``MTLCommandBufferStatusNotEnqueued`` mystery silence.

Batch 1 covers the 5 simple MPS dispatchers:

* ``tessera_apple_gpu_cholesky_f32`` (MPSMatrixDecompositionCholesky)
* ``tessera_apple_gpu_solve_cholesky_f32``
* ``tessera_apple_gpu_solve_lu_f32`` (MPSMatrixDecompositionLU + SolveLU)
* ``tessera_apple_gpu_tri_solve_f32`` (MPSMatrixSolveTriangular)
* The internal ``dispatch_mps_random_f32`` namespace function
  (MPSMatrixRandomPhilox)

Tests pin:

* **Numerical correctness preserved** — each migrated dispatcher
  still produces output that agrees with a CPU reference at fp32
  tolerance. Functional regression catch.
* **Build hygiene** — the source file still mentions
  ``commit_and_wait_with_timeout`` at all the expected sites; the
  legacy ``waitUntilCompleted`` count for the 5 covered op symbols
  is zero in the migrated regions. This is a drift gate.

Subsequent batches will migrate the remaining ~55 sites.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest


_RUNTIME_SRC = (Path(__file__).resolve().parent.parent.parent
                / "src" / "compiler" / "codegen"
                / "Tessera_Apple_Backend" / "runtime"
                / "apple_gpu_runtime.mm")


def _runtime_loads() -> bool:
    try:
        from tessera._apple_gpu_dispatch import apple_gpu_runtime
    except ImportError:
        return False
    return apple_gpu_runtime() is not None


# ---- Numerical-correctness (migrated dispatchers still work) -----------

def test_cholesky_still_correct_after_migration():
    if not _runtime_loads():
        pytest.skip("Apple GPU runtime not buildable on this host")
    from tessera._apple_gpu_dispatch import bind_symbol
    import ctypes
    fn = bind_symbol(
        "tessera_apple_gpu_cholesky_f32",
        (ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32),
        restype=ctypes.c_int32)
    if fn is None:
        pytest.skip("symbol not available")

    rng = np.random.default_rng(0xC0FFEE)
    n = 8
    M = rng.standard_normal((n, n)).astype(np.float32)
    A = (M @ M.T + n * np.eye(n)).astype(np.float32)  # SPD
    L = np.zeros((n, n), dtype=np.float32)
    rc = fn(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            L.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(n))
    assert rc == 0, f"cholesky returned {rc}"
    # Verify L is lower-triangular and L @ L.T ≈ A.
    assert np.allclose(np.triu(L, k=1), 0.0)
    np.testing.assert_allclose(L @ L.T, A, rtol=1e-3, atol=1e-3)


def test_solve_cholesky_still_correct_after_migration():
    if not _runtime_loads():
        pytest.skip("Apple GPU runtime not buildable on this host")
    from tessera._apple_gpu_dispatch import bind_symbol
    import ctypes
    fn = bind_symbol(
        "tessera_apple_gpu_solve_cholesky_f32",
        (ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32, ctypes.c_int32),
        restype=ctypes.c_int32)
    if fn is None:
        pytest.skip("symbol not available")

    rng = np.random.default_rng(0xBEEF)
    n, nrhs = 6, 3
    M = rng.standard_normal((n, n)).astype(np.float32)
    A = (M @ M.T + n * np.eye(n)).astype(np.float32)  # SPD
    B = rng.standard_normal((n, nrhs)).astype(np.float32)
    X = np.zeros((n, nrhs), dtype=np.float32)
    rc = fn(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(n), ctypes.c_int32(nrhs))
    assert rc == 0
    np.testing.assert_allclose(A @ X, B, rtol=1e-3, atol=1e-3)


def test_solve_lu_still_correct_after_migration():
    if not _runtime_loads():
        pytest.skip("Apple GPU runtime not buildable on this host")
    from tessera._apple_gpu_dispatch import bind_symbol
    import ctypes
    fn = bind_symbol(
        "tessera_apple_gpu_solve_lu_f32",
        (ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32, ctypes.c_int32),
        restype=ctypes.c_int32)
    if fn is None:
        pytest.skip("symbol not available")

    rng = np.random.default_rng(0xDEAD)
    n, nrhs = 6, 2
    # Well-conditioned (diagonally dominant) so LU is stable in f32.
    A = (rng.standard_normal((n, n)) + n * np.eye(n)).astype(np.float32)
    B = rng.standard_normal((n, nrhs)).astype(np.float32)
    X = np.zeros((n, nrhs), dtype=np.float32)
    rc = fn(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(n), ctypes.c_int32(nrhs))
    assert rc == 0
    np.testing.assert_allclose(A @ X, B, rtol=1e-2, atol=1e-2)


def test_tri_solve_still_correct_after_migration():
    if not _runtime_loads():
        pytest.skip("Apple GPU runtime not buildable on this host")
    from tessera._apple_gpu_dispatch import bind_symbol
    import ctypes
    fn = bind_symbol(
        "tessera_apple_gpu_tri_solve_f32",
        (ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32),
        restype=ctypes.c_int32)
    if fn is None:
        pytest.skip("symbol not available")

    rng = np.random.default_rng(0xFACE)
    n, nrhs = 8, 2
    L = np.tril(rng.standard_normal((n, n))).astype(np.float32)
    L[np.arange(n), np.arange(n)] = 2.0 + np.abs(np.diag(L))  # stable
    B = rng.standard_normal((n, nrhs)).astype(np.float32)
    X = np.zeros((n, nrhs), dtype=np.float32)
    rc = fn(L.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(n), ctypes.c_int32(nrhs),
            ctypes.c_int32(1),   # lower
            ctypes.c_int32(0),   # no transpose
            ctypes.c_int32(0))   # not unit-diagonal
    assert rc == 0
    np.testing.assert_allclose(L @ X, B, rtol=1e-3, atol=1e-3)


# ---- Source-level drift gate (the migrated sites stay migrated) --------

def test_runtime_source_includes_pattern_4_helper():
    """Sanity check: the wrapper the migration depends on is still
    defined in the source. If a future refactor renames or removes
    it, this guard fires before behavior changes silently."""
    src = _RUNTIME_SRC.read_text()
    assert "commit_and_wait_with_timeout" in src, (
        "Pattern-4 timeout-event wrapper missing from runtime source")
    # And the helper is callable across the entire file (not just one
    # local block) — there should be at least 6 call sites total
    # (1 inside the helper itself for the fallback case + 5 batch-1
    # migrations).
    call_sites = src.count("commit_and_wait_with_timeout(")
    assert call_sites >= 6, (
        f"expected ≥6 call sites for commit_and_wait_with_timeout, "
        f"found {call_sites} — migration may have regressed")


def test_migrated_dispatchers_no_longer_call_waituntilcompleted():
    """The 5 batch-1 dispatchers must not call ``waitUntilCompleted``.
    Drift gate: a future edit that re-introduces the legacy pattern
    in any of these functions will fire this test."""
    src = _RUNTIME_SRC.read_text()
    targets = [
        "tessera_apple_gpu_cholesky_f32",
        "tessera_apple_gpu_solve_cholesky_f32",
        "tessera_apple_gpu_solve_lu_f32",
        "tessera_apple_gpu_tri_solve_f32",
        "dispatch_mps_random_f32",
    ]
    for fn_name in targets:
        # Locate the function's opening brace, then walk balanced
        # braces to its match. That gives us the exact body.
        m = re.search(rf"\b{re.escape(fn_name)}\b\s*\(", src)
        assert m is not None, f"could not locate {fn_name!r} in runtime source"
        # Find the next "{" after the signature.
        body_open = src.find("{", m.end())
        assert body_open != -1, f"no body start for {fn_name!r}"
        depth = 1
        i = body_open + 1
        while i < len(src) and depth > 0:
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        assert depth == 0, f"unbalanced braces in body of {fn_name!r}"
        body = src[body_open:i]
        # Check for the actual Objective-C method INVOCATION
        # ``[cb waitUntilCompleted]`` rather than the bare word — that
        # way a comment referring to the legacy pattern (e.g., in the
        # migration changelog comment we leave next to the wrapper
        # call) doesn't false-positive.
        assert "waitUntilCompleted]" not in body, (
            f"{fn_name}: legacy [cb waitUntilCompleted] still present "
            f"after batch-1 migration — body slice:\n{body[:400]}...")
        # And the migrated helper IS present in each body.
        assert "commit_and_wait_with_timeout" in body, (
            f"{fn_name}: Pattern-4 wrapper missing — migration "
            f"incomplete; body slice:\n{body[:400]}...")
