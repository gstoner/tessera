"""Phase 1 Sprint 1.9 — batched matmul (docs/spec/PRODUCTION_COMPILER_PLAN.md).

`tessera.batched_gemm` (rank-3, C[i] = A[i] @ B[i]) → linalg.batch_matmul. The
batch dimension unblocks batched inference and (with reshape, later) multi-head
attention.

numpy oracle + unfakeable invocation-counter advance. Skips when libtessera_jit
is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


@pytest.mark.parametrize("B,M,K,N", [(2, 3, 4, 5), (1, 1, 1, 1), (8, 16, 16, 16)])
def test_jit_bmm_matches_numpy_oracle(B, M, K, N):
    rng = np.random.default_rng(B * 100 + M * 10 + K + N)
    a = rng.standard_normal((B, M, K)).astype(np.float32)
    b = rng.standard_normal((B, K, N)).astype(np.float32)
    out = jb.jit_bmm(a, b)
    assert out.shape == (B, M, N)
    np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-4)


def test_jit_bmm_each_batch_is_independent():
    # Distinct per-batch matrices: batch 0 is identity, batch 1 is 2*identity.
    a = np.stack([np.eye(3), 2 * np.eye(3)]).astype(np.float32)
    b = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 3.0)]).astype(np.float32)
    out = jb.jit_bmm(a, b)
    np.testing.assert_allclose(out[0], b[0], rtol=1e-5)        # I @ ones = ones
    np.testing.assert_allclose(out[1], 2 * b[1], rtol=1e-5)    # 2I @ 3 = 6


def test_jit_bmm_executed():
    a = np.ones((2, 2, 2), np.float32)
    b = np.ones((2, 2, 2), np.float32)
    before = jb.invocation_count()
    jb.jit_bmm(a, b)
    assert jb.invocation_count() == before + 1


def test_jit_bmm_bf16_f32_accumulate():
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 4, 8)).astype(bf16)
    b = rng.standard_normal((2, 8, 4)).astype(bf16)
    out = jb.jit_bmm(a, b)
    assert out.dtype == bf16
    ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(bf16)
    np.testing.assert_allclose(
        out.astype(np.float32), ref.astype(np.float32), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    "a,b",
    [
        (np.ones((2, 3, 4), np.float32), np.ones((3, 4, 5), np.float32)),  # batch mismatch
        (np.ones((2, 3, 4), np.float32), np.ones((2, 5, 6), np.float32)),  # K mismatch
        (np.ones((3, 4), np.float32), np.ones((4, 5), np.float32)),        # rank-2
    ],
)
def test_jit_bmm_rejects_out_of_envelope(a, b):
    with pytest.raises(jb.TesseraJitError):
        jb.jit_bmm(a, b)


def test_bmm_batched_attention_in_one_graph():
    """Batched (per-head, K pre-transposed) attention via bmm, ONE device_verified_jit fn.

    H heads laid out as the batch dim: scores[h] = Q[h] @ Kt[h], softmax, @ V[h].
    K is pre-transposed by the caller (batched transpose is a later slice); the
    point here is that the whole batched block compiles to one function.
    """
    rng = np.random.default_rng(13)
    H, T, d = 3, 5, 8
    q = (rng.standard_normal((H, T, d)) / np.sqrt(d)).astype(np.float32)  # scale folded
    k = rng.standard_normal((H, T, d)).astype(np.float32)
    v = rng.standard_normal((H, T, d)).astype(np.float32)
    kt = np.ascontiguousarray(np.transpose(k, (0, 2, 1)))  # (H, d, T)

    g = GraphFn()
    gq, gkt, gv = g.arg((H, T, d)), g.arg((H, d, T)), g.arg((H, T, d))
    scores = g.bmm(gq, gkt)        # (H, T, T)
    probs = g.softmax(scores)      # row-softmax over last axis
    out_v = g.bmm(probs, gv)       # (H, T, d)
    g.ret(out_v)

    before = jb.invocation_count()
    out = g.run(q, kt, v)
    assert jb.invocation_count() == before + 1  # one device_verified_jit function

    s = q @ kt
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    ref = p @ v
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
