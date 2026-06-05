"""Phase 1 production-lane tests (docs/spec/PRODUCTION_COMPILER_PLAN.md).

Sprint 1.1 — generalized JIT harness + first non-elementwise op (matmul) + the
binary elementwise family expansion (sub, mul).

Every op MUST:
* match numpy within tolerance (numpy = oracle, not executor),
* advance the unfakeable invocation counter by exactly 1 per call (proof the
  MLIR/LLVM lane executed; a silent numpy fallback would fail this).

Skips only when libtessera_jit is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


# ── Matmul (the Phase 1 headline: first non-elementwise op) ────────────────


@pytest.mark.parametrize(
    "M,K,N", [(4, 8, 3), (1, 1, 1), (16, 16, 16), (7, 5, 9)]
)
def test_jit_matmul_matches_numpy_oracle(M, K, N):
    rng = np.random.default_rng(M * 1000 + K * 31 + N)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    out = jb.jit_matmul(a, b)
    # Loose tolerance: matmul accumulates K multiplies in f32; the lane uses
    # default linalg.matmul lowering, not Kahan / fp64 accumulation.
    np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-4)


def test_jit_matmul_executed_the_compiled_function():
    a = np.eye(5, dtype=np.float32)
    b = np.arange(25, dtype=np.float32).reshape(5, 5)
    before = jb.invocation_count()
    out = jb.jit_matmul(a, b)
    assert jb.invocation_count() == before + 1
    # Identity @ B == B (sanity that the result actually came from the JIT).
    np.testing.assert_array_equal(out, b)


@pytest.mark.parametrize(
    "a,b",
    [
        # Shape mismatch: inner dims don't agree.
        (np.ones((4, 8), np.float32), np.ones((7, 3), np.float32)),
        # Rank-3: matmul is rank-2 only in Phase 1 (batched matmul is a later slice).
        (np.ones((2, 4, 8), np.float32), np.ones((2, 8, 3), np.float32)),
        # f16: outside the Phase 1 f32 envelope.
        (np.ones((4, 4), np.float16), np.ones((4, 4), np.float16)),
    ],
)
def test_jit_matmul_rejects_out_of_envelope(a, b):
    with pytest.raises(jb.TesseraJitError):
        jb.jit_matmul(a, b)


# ── Binary elementwise family expansion ────────────────────────────────────


@pytest.mark.parametrize(
    "fn, ref",
    [
        (jb.jit_sub, np.subtract),
        (jb.jit_mul, np.multiply),
    ],
)
def test_jit_binary_eltwise_matches_numpy_oracle(fn, ref):
    rng = np.random.default_rng(42)
    a = rng.standard_normal((3, 5)).astype(np.float32)
    b = rng.standard_normal((3, 5)).astype(np.float32)
    out = fn(a, b)
    np.testing.assert_allclose(out, ref(a, b), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("fn", [jb.jit_sub, jb.jit_mul])
def test_jit_binary_eltwise_executed(fn):
    a = np.full((2, 4), 3.0, dtype=np.float32)
    b = np.full((2, 4), 1.0, dtype=np.float32)
    before = jb.invocation_count()
    fn(a, b)
    assert jb.invocation_count() == before + 1


# ── Harness generalization: descriptor packing handles distinct ranks ──────


def test_descriptor_packing_handles_distinct_ranks_in_one_session():
    """The Phase 1 harness packs rank-N descriptors generically; running rank-1,
    rank-2, rank-3 invocations in the same process exercises the (rank, ctype)
    cache and the double-indirection packed_args path.
    """
    rng = np.random.default_rng(0)
    cases = [(7,), (4, 5), (2, 3, 4)]
    for shape in cases:
        a = rng.standard_normal(shape).astype(np.float32)
        b = rng.standard_normal(shape).astype(np.float32)
        before = jb.invocation_count()
        out = jb.jit_add(a, b)
        assert jb.invocation_count() == before + 1
        np.testing.assert_allclose(out, a + b, rtol=1e-6, atol=1e-6)
