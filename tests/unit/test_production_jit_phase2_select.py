"""Phase 2 Sprint 2.1 — data-parallel conditional / attention masking
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

`tessera.select` (cond != 0 ? a : b) and `tessera.masked_fill` (mask != 0 ? x :
value). The masked_fill path is the causal-attention masking primitive. These
are the data-parallel form of a conditional (no divergent control flow yet —
that's scf.if in Sprint 2.3).

numpy oracle + invocation-counter. Skips when libtessera_jit is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def test_select_matches_numpy_where():
    rng = np.random.default_rng(0)
    cond = (rng.standard_normal((3, 4)) > 0).astype(np.float32)  # 0/1 mask
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    out = jb.jit_select(cond, a, b)
    np.testing.assert_array_equal(out, np.where(cond != 0, a, b))


def test_masked_fill_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 4)).astype(np.float32)
    mask = np.tril(np.ones((4, 4))).astype(np.float32)  # causal: lower-tri keep
    out = jb.jit_masked_fill(x, mask, value=-1e9)
    expect = np.where(mask != 0, x, np.float32(-1e9))
    np.testing.assert_array_equal(out, expect)


def test_masked_fill_executed():
    x = np.ones((2, 2), np.float32)
    m = np.array([[1, 0], [1, 1]], np.float32)
    before = jb.invocation_count()
    jb.jit_masked_fill(x, m, -5.0)
    assert jb.invocation_count() == before + 1


def test_causal_attention_with_mask_in_one_graph():
    """softmax(masked_fill(Q Kᵀ, causal_mask, -inf)) V — ONE compiled function.

    The causal mask zeroes attention to future positions. This is the first
    Phase-2 capability composed into a real model pattern.
    """
    rng = np.random.default_rng(7)
    T, d = 5, 8
    q = (rng.standard_normal((T, d)) / np.sqrt(d)).astype(np.float32)  # scale folded
    k = rng.standard_normal((T, d)).astype(np.float32)
    v = rng.standard_normal((T, d)).astype(np.float32)
    causal = np.tril(np.ones((T, T))).astype(np.float32)

    g = GraphFn()
    gq, gk, gv, gm = g.arg((T, d)), g.arg((T, d)), g.arg((T, d)), g.arg((T, T))
    scores = g.matmul(gq, gk, transpose_b=True)
    masked = g.masked_fill(scores, gm, value=-1e9)
    probs = g.softmax(masked)
    out_v = g.matmul(probs, gv)
    g.ret(out_v)

    before = jb.invocation_count()
    out = g.run(q, k, v, causal)
    assert jb.invocation_count() == before + 1  # one compiled function

    # numpy causal-attention oracle
    s = q @ k.T
    s = np.where(causal != 0, s, -1e9)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    ref = p @ v
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    # Sanity: row 0 attends only to position 0 (causal), so out[0] == v[0].
    np.testing.assert_allclose(out[0], v[0], rtol=1e-4, atol=1e-4)
