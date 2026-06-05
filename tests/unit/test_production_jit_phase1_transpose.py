"""Phase 1 Sprint 1.7 — transpose + transposed matmul
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

`tessera.transpose` (rank-2, via linalg.transpose) and `tessera.matmul` with
transposeA/transposeB (operand transposed before a plain matmul). The
transpose_b path is the attention `Q @ Kᵀ` shape.

numpy oracle + unfakeable invocation-counter advance. Skips when libtessera_jit
is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


# ── transpose ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(2, 3), (5, 1), (16, 7)])
def test_jit_transpose_matches_numpy(shape):
    rng = np.random.default_rng(abs(hash(shape)) & 0xFFFF)
    a = rng.standard_normal(shape).astype(np.float32)
    out = jb.jit_transpose(a)
    assert out.shape == (shape[1], shape[0])
    np.testing.assert_array_equal(out, a.T)


def test_jit_transpose_executed():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    before = jb.invocation_count()
    jb.jit_transpose(a)
    assert jb.invocation_count() == before + 1


# ── transposed matmul ───────────────────────────────────────────────────────


def test_matmul_transpose_b_is_a_at_b_transposed():
    # The Q @ Kᵀ pattern: a=(M,K)=(2,4), b stored (N,K)=(3,4) -> (M,N)=(2,3).
    rng = np.random.default_rng(1)
    a = rng.standard_normal((2, 4)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    out = jb.jit_matmul(a, b, transpose_b=True)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, a @ b.T, rtol=1e-4, atol=1e-4)


def test_matmul_transpose_a():
    # a stored (K,M)=(4,2), b=(K,N)=(4,3) -> (M,N)=(2,3): out = aᵀ @ b.
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, 2)).astype(np.float32)
    b = rng.standard_normal((4, 3)).astype(np.float32)
    out = jb.jit_matmul(a, b, transpose_a=True)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, a.T @ b, rtol=1e-4, atol=1e-4)


def test_matmul_transpose_both():
    # aᵀ @ bᵀ: a stored (K,M)=(4,2), b stored (N,K)=(3,4) -> (2,3).
    rng = np.random.default_rng(3)
    a = rng.standard_normal((4, 2)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    out = jb.jit_matmul(a, b, transpose_a=True, transpose_b=True)
    np.testing.assert_allclose(out, a.T @ b.T, rtol=1e-4, atol=1e-4)


def test_matmul_transpose_executed_and_no_regression_on_plain():
    a = np.eye(3, dtype=np.float32)
    b = np.arange(9, dtype=np.float32).reshape(3, 3)
    before = jb.invocation_count()
    plain = jb.jit_matmul(a, b)
    tb = jb.jit_matmul(a, b, transpose_b=True)
    assert jb.invocation_count() == before + 2
    np.testing.assert_allclose(plain, b, rtol=1e-5)
    np.testing.assert_allclose(tb, b.T, rtol=1e-5)


def test_matmul_transpose_b_contracting_mismatch_rejected():
    a = np.ones((2, 4), np.float32)
    b = np.ones((3, 5), np.float32)  # stored (N,K)=(3,5): K=5 != a's K=4
    with pytest.raises(jb.TesseraJitError):
        jb.jit_matmul(a, b, transpose_b=True)


# ── attention-shaped composition: softmax(Q Kᵀ / √d) V ─────────────────────


def test_single_head_attention_composes():
    rng = np.random.default_rng(9)
    T, d = 5, 8
    q = rng.standard_normal((T, d)).astype(np.float32)
    k = rng.standard_normal((T, d)).astype(np.float32)
    v = rng.standard_normal((T, d)).astype(np.float32)

    scores = jb.jit_matmul(q, k, transpose_b=True)          # (T,T) = Q Kᵀ
    scale = np.float32(1.0 / np.sqrt(d))
    scaled = jb.jit_mul(scores, np.full_like(scores, scale))  # / √d
    probs = jb.jit_softmax(scaled, axis=-1)                  # row-softmax
    out = jb.jit_matmul(probs, v)                           # (T,d)

    # numpy oracle for the whole block
    ref_scores = (q @ k.T) * scale
    ref_probs = np.exp(ref_scores - ref_scores.max(-1, keepdims=True))
    ref_probs /= ref_probs.sum(-1, keepdims=True)
    ref = ref_probs @ v
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
