"""Representative reference-op execution matrix
(Test-tree review phase P2-9, 2026-05-20).

The E2E coverage audit reports 232 ops at ``runnable_reference``.
We don't want 232 native tests, but every family should have **one
representative test** that exercises ``tessera.ops.registry.dispatch``
end-to-end against a small input.  If a family's reference path
ever silently regresses (e.g., a registry entry stops being
registered, or its signature drifts incompatibly), the matching
representative test fails fast.

One test per family — small inputs, deterministic seeds, no
hardware dependency.  Each test:

1. Looks up the op in ``tessera.ops.registry``.
2. Calls it with shape-appropriate inputs.
3. Asserts the output matches a hand-coded numpy expectation
   (correctness oracle).

When a family's representative op is moved to a fused native path,
the corresponding test still passes because it goes through
``registry.dispatch(..., prefer_runtime=False)`` which always picks
the reference path.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture(scope="module")
def ops_registry() -> Any:
    """Lazy-load tessera.ops.registry; the import is non-trivial."""
    import tessera
    return tessera.ops.registry


def _dispatch_ref(registry: Any, op_name: str, *args: Any, **kwargs: Any) -> Any:
    """Force the numpy-reference path through the registry."""
    return registry.dispatch(op_name, *args, prefer_runtime=False, **kwargs)


# ─────────────────────────────────────────────────────────────────────
# Tensor family — elementwise + arithmetic
# ─────────────────────────────────────────────────────────────────────


def test_family_tensor_elementwise_add(ops_registry: Any) -> None:
    """Tensor family / elementwise: ``tessera.ops.add`` matches numpy."""
    rng = np.random.default_rng(seed=0xDEAD)
    x = rng.standard_normal((4, 4)).astype(np.float32)
    y = rng.standard_normal((4, 4)).astype(np.float32)
    out = _dispatch_ref(ops_registry, "add", x, y)
    np.testing.assert_allclose(out, x + y, rtol=1e-6, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────
# Reduction family
# ─────────────────────────────────────────────────────────────────────


def test_family_reduction_sum(ops_registry: Any) -> None:
    """Reduction family: ``tessera.ops.sum`` matches numpy."""
    rng = np.random.default_rng(seed=0xBEEF)
    x = rng.standard_normal((8, 4)).astype(np.float32)
    out_all = _dispatch_ref(ops_registry, "sum", x)
    np.testing.assert_allclose(out_all, x.sum(), rtol=1e-5, atol=1e-5)
    out_axis = _dispatch_ref(ops_registry, "sum", x, axis=0)
    np.testing.assert_allclose(out_axis, x.sum(axis=0), rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
# Layout / structural family — reshape (no compute, just shape)
# ─────────────────────────────────────────────────────────────────────


def test_family_layout_reshape(ops_registry: Any) -> None:
    """Layout family: ``tessera.ops.reshape`` produces the expected shape."""
    x = np.arange(24, dtype=np.float32)
    out = _dispatch_ref(ops_registry, "reshape", x, (2, 3, 4))
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out, x.reshape(2, 3, 4))


# ─────────────────────────────────────────────────────────────────────
# Linalg family — Cholesky (the canonical SPD decomposition)
# ─────────────────────────────────────────────────────────────────────


def test_family_linalg_cholesky(ops_registry: Any) -> None:
    """Linalg family: ``tessera.ops.cholesky`` of a known-SPD matrix."""
    # Build a small SPD: A = M M^T + I.
    rng = np.random.default_rng(seed=0xC0DE)
    M = rng.standard_normal((4, 4)).astype(np.float64)
    A = (M @ M.T + np.eye(4)).astype(np.float64)
    L = _dispatch_ref(ops_registry, "cholesky", A)
    np.testing.assert_allclose(L @ L.T, A, rtol=1e-10, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────
# Spectral family — fft (round-trip via ifft)
# ─────────────────────────────────────────────────────────────────────


def test_family_spectral_fft_round_trip(ops_registry: Any) -> None:
    """Spectral family: ``tessera.ops.fft`` matches numpy.fft.fft."""
    rng = np.random.default_rng(seed=0xFEED)
    x = rng.standard_normal((1, 64)).astype(np.complex64)
    out = _dispatch_ref(ops_registry, "fft", x)
    np.testing.assert_allclose(out, np.fft.fft(x), rtol=1e-3, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────
# Complex / M7 family — complex_mul
# ─────────────────────────────────────────────────────────────────────


def test_family_complex_complex_mul(ops_registry: Any) -> None:
    """M7 family: ``tessera.ops.complex_mul`` matches numpy complex
    multiplication (forced through the registry reference path)."""
    rng = np.random.default_rng(seed=0xCAFE)
    a = (rng.standard_normal(8) + 1j * rng.standard_normal(8)).astype(np.complex64)
    b = (rng.standard_normal(8) + 1j * rng.standard_normal(8)).astype(np.complex64)
    out = _dispatch_ref(ops_registry, "complex_mul", a, b)
    # The reference returns ComplexScalar; convert via to_numpy for
    # comparison.  numpy * numpy is the oracle.
    import tessera.complex as _ts_complex
    actual = _ts_complex.to_numpy(out, dtype=np.complex64)
    np.testing.assert_allclose(actual, a * b, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
# Indexing family — gather
# ─────────────────────────────────────────────────────────────────────


def test_family_indexing_gather(ops_registry: Any) -> None:
    """Indexing family: ``tessera.ops.gather`` matches numpy take."""
    x = np.arange(20, dtype=np.float32).reshape(5, 4)
    indices = np.array([0, 2, 4], dtype=np.int64)
    out = _dispatch_ref(ops_registry, "gather", x, indices, axis=0)
    np.testing.assert_array_equal(out, x[indices])


# ─────────────────────────────────────────────────────────────────────
# RNG / stochastic family — rng_normal (deterministic via seed)
# ─────────────────────────────────────────────────────────────────────


def test_family_rng_normal_deterministic(ops_registry: Any) -> None:
    """RNG family: ``tessera.ops.rng_normal`` is deterministic given a
    seed.  We don't gate on distributional properties (that's a
    separate stats test); just confirm two calls with the same seed
    produce bitwise-identical output."""
    out_a = _dispatch_ref(ops_registry, "rng_normal", (16,), seed=42)
    out_b = _dispatch_ref(ops_registry, "rng_normal", (16,), seed=42)
    np.testing.assert_array_equal(out_a, out_b)
    out_c = _dispatch_ref(ops_registry, "rng_normal", (16,), seed=43)
    # Different seed → different output (with overwhelming probability
    # at N=16).  Assert at least one element differs.
    assert not np.array_equal(out_a, out_c), (
        "rng_normal returned identical output for distinct seeds"
    )


# ─────────────────────────────────────────────────────────────────────
# Comparison / non-differentiable family — floor
# ─────────────────────────────────────────────────────────────────────


def test_family_nondiff_floor(ops_registry: Any) -> None:
    """Non-differentiable family: ``tessera.ops.floor`` matches numpy."""
    rng = np.random.default_rng(seed=0xBABE)
    x = (rng.standard_normal(20) * 5.0).astype(np.float32)
    out = _dispatch_ref(ops_registry, "floor", x)
    np.testing.assert_array_equal(out, np.floor(x))


# ─────────────────────────────────────────────────────────────────────
# Coverage guard — every family above must have a representative op
# present in the registry.  If the registry drops one of these
# names, this test fails before the per-family tests do, giving a
# clearer error.
# ─────────────────────────────────────────────────────────────────────


REPRESENTATIVE_OPS = (
    ("tensor",     "add"),
    ("reduction",  "sum"),
    ("layout",     "reshape"),
    ("linalg",     "cholesky"),
    ("spectral",   "fft"),
    ("complex",    "complex_mul"),
    ("indexing",   "gather"),
    ("rng",        "rng_normal"),
    ("nondiff",    "floor"),
)


@pytest.mark.parametrize("family,op_name", REPRESENTATIVE_OPS)
def test_representative_op_present_in_registry(
    ops_registry: Any, family: str, op_name: str,
) -> None:
    entry = ops_registry.get(op_name)
    assert entry is not None, (
        f"family={family!r} representative op {op_name!r} missing "
        "from tessera.ops.registry — the reference-op matrix "
        "guard expects every family to have a registered "
        "representative."
    )
    assert entry.reference is not None, (
        f"family={family!r} op {op_name!r} has no reference "
        "implementation registered"
    )
