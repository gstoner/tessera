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

def test_only_documented_waituntilcompleted_sites_remain():
    """Only three ``waitUntilCompleted`` invocations should remain.

    * The fallback path INSIDE ``commit_and_wait_with_timeout`` itself
      (around the helper's lazy event init). Recursing into the
      wrapper there would deadlock; the legacy path is the correct
      fallback.
    * The equivalent fallback inside
      ``commit_mpsgraph_and_wait_with_timeout``. The MPSGraph wrapper must
      commit its live root itself because ``commitAndContinue`` may rotate
      the originally supplied command buffer.
    * The fallback INSIDE ``ts_enc_commit_wait`` for the case where
      shared-event init failed (the encode-session path also keeps a
      legacy synchronous wait as a no-crash fallback, mirroring the
      helper's own). This waits on ``[root waitUntilCompleted]`` — the
      *live* ``s->cb.rootCommandBuffer`` — NOT the ``s->mtlcb`` captured
      at ts_enc_begin: MPSGraph's encode may call ``commitAndContinue``
      and rotate the underlying buffer, so the captured handle can be
      stale. (See ``ts_enc_commit_wait`` + MPSCommandBuffer.h.)
    * The equivalent no-shared-event fallback inside
      ``ts_enc_wait_destroy``. ``ts_enc_commit_async`` replaces the captured
      handle with the live root used for its commit before ownership crosses
      the asynchronous boundary.

    Any OTHER site is a regression. This test fires loud."""
    src = _RUNTIME_SRC.read_text()
    import re
    cb_calls = [m for m in re.finditer(
        r"\[cb waitUntilCompleted\]", src)]
    session_calls = [m for m in re.finditer(
        r"\[root waitUntilCompleted\]", src)]
    # One generic-command-buffer fallback and three live-root fallbacks.
    assert len(cb_calls) == 1, (
        f"expected exactly 1 [cb waitUntilCompleted] (the wrapper's "
        f"own fallback), found {len(cb_calls)}")
    assert len(session_calls) == 3, (
        f"expected exactly 3 [root waitUntilCompleted] calls (the owned "
        f"MPSGraph, synchronous-session, and async-session fallbacks on live root command "
        f"buffers), found {len(session_calls)}")
    # The pre-fix stale-handle wait must be gone (commitAndContinue safety).
    assert "[s->mtlcb waitUntilCompleted]" not in src, (
        "stale [s->mtlcb waitUntilCompleted] reintroduced — must wait on "
        "the live rootCommandBuffer, not the handle captured at begin")


def test_runtime_source_includes_pattern_4_helper():
    """Sanity check: the wrapper the migration depends on is still
    defined in the source. If a future refactor renames or removes
    it, this guard fires before behavior changes silently."""
    src = _RUNTIME_SRC.read_text()
    assert "commit_and_wait_with_timeout" in src, (
        "Pattern-4 timeout-event wrapper missing from runtime source")
    # And the helper is callable across the entire file (not just one
    # local block) — there should be at least 50 call sites total
    # (1 inside the helper itself + 5 batch-1 + 6 batch-2 + 8 batch-3
    # + 30 batch-4 migrations).
    call_sites = src.count("commit_and_wait_with_timeout(")
    assert call_sites >= 50, (
        f"expected ≥50 call sites for commit_and_wait_with_timeout, "
        f"found {call_sites} — migration may have regressed")


def test_migrated_dispatchers_no_longer_call_waituntilcompleted():
    """The 5 batch-1 dispatchers must not call ``waitUntilCompleted``.
    Drift gate: a future edit that re-introduces the legacy pattern
    in any of these functions will fire this test."""
    src = _RUNTIME_SRC.read_text()
    targets = [
        # Batch 1 (2026-05-31)
        "tessera_apple_gpu_cholesky_f32",
        "tessera_apple_gpu_solve_cholesky_f32",
        "tessera_apple_gpu_solve_lu_f32",
        "tessera_apple_gpu_tri_solve_f32",
        "dispatch_mps_random_f32",
        # Batch 2 (2026-06-01)
        "dispatch_mps_gemm_f16",
        "dispatch_svd_jacobi_f32",
        "dispatch_svd_jacobi_bl_f32",
        "dispatch_cholesky_batched_f32",
        "dispatch_tri_solve_batched_f32",
        "dispatch_dev_cast",
        # Batch 3 (2026-06-01) — MSL custom kernels for attention primitives
        "dispatch_rope_msl",
        "dispatch_rope_msl_f16",
        "dispatch_flash_attn_msl",
        "dispatch_flash_attn_msl_f16",
        "dispatch_softmax_msl",
        "dispatch_softmax_msl_f16",
        "dispatch_gelu_msl",
        "dispatch_gelu_msl_f16",
        # Batch 4 (bulk-migrated via
        # tools/scripts/migrate_wait_until_completed.py) — spot-check the
        # dispatch_* helpers that hold the migrated call sites. Other
        # batch-4 dispatchers (EBM / complex / layer_norm / flash_attn_gqa
        # / matmul_softmax variants) inline the encode-and-wait logic
        # directly in the extern "C" entry; the global "only two
        # exceptions remain" gate (test_only_documented_waituntilcompleted_sites_remain)
        # already covers them.
        "dispatch_matmul_softmax_tiled_msl",
        "dispatch_clifford_geo_product_cl30_f32_msl",
        "dispatch_clifford_unary_8x8_f32_msl",
        "dispatch_clifford_unary_8x1_f32_msl",
        "dispatch_clifford_binary_8x8_f32_msl",
        "dispatch_clifford_binary_8x1_f32_msl",
        "dispatch_clifford_grade_projection_cl30_f32_msl",
    ]
    for fn_name in targets:
        # Locate the function's opening brace, then walk balanced
        # braces to its match. That gives us the exact body. The
        # ``\(`` is tight (no \s* before) so comments that mention
        # the identifier followed by a space-then-paren (e.g., "see
        # foo (legacy path)") don't match.
        m = re.search(rf"\b{re.escape(fn_name)}\b\(", src)
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
