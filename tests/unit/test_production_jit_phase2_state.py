"""Phase 2 completion — state (docs/spec/PRODUCTION_COMPILER_PLAN.md).

The Phase-2 DoD: a *stateful decode step runs end-to-end on CPU through the
production lane*. Built from:
  * tessera.write_row  — functional KV-cache update (tensor.insert_slice),
  * multi-result functions — a decode step returns out + updated caches as ONE
    compiled function (DPS out-params per result).

The capstone threads the KV cache across decode steps and matches a numpy
incremental-decoding reference. Skips when libtessera_jit is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


# ── write_row (functional KV update) ────────────────────────────────────────


def test_write_row_matches_numpy():
    buf = np.zeros((4, 3), np.float32)
    val = np.array([[1.0, 2.0, 3.0]], np.float32)
    out = jb.jit_write_row(buf, val, row=2)
    expect = buf.copy()
    expect[2] = val[0]
    np.testing.assert_array_equal(out, expect)
    np.testing.assert_array_equal(buf, np.zeros((4, 3), np.float32))  # functional


@pytest.mark.parametrize("row", [0, 1, 3])
def test_write_row_each_position(row):
    buf = np.arange(12, dtype=np.float32).reshape(4, 3)
    val = np.full((1, 3), -1.0, np.float32)
    out = jb.jit_write_row(buf, val, row)
    expect = buf.copy()
    expect[row] = -1.0
    np.testing.assert_array_equal(out, expect)


# ── multi-result function ───────────────────────────────────────────────────


def test_graph_multi_result_returns_tuple():
    g = GraphFn()
    a, b = g.arg((4,)), g.arg((4,))
    g.ret(g.add(a, b), g.mul(a, b))  # two results
    x = np.array([1, 2, 3, 4], np.float32)
    y = np.array([10, 10, 10, 10], np.float32)
    s, p = g.run(x, y)
    np.testing.assert_array_equal(s, x + y)
    np.testing.assert_array_equal(p, x * y)


def test_graph_multi_result_one_compiled_function():
    g = GraphFn()
    a = g.arg((3,))
    g.ret(g.relu(a), g.tanh(a), g.add(a, a))  # three results
    before = jb.invocation_count()
    g.run(np.array([-1.0, 0.5, 2.0], np.float32))
    assert jb.invocation_count() == before + 1  # one fn, regardless of #results


# ── stateful decode step (the Phase-2 DoD) ─────────────────────────────────


def _decode_step_graph(T, D, t):
    """One incremental-decode step at position `t`, as ONE compiled function.

    Inputs: cache_k, cache_v (T,D), new_k, new_v (1,D), q (1,D), mask (1,T).
    Writes new_k/new_v into row t, attends q over the (now-updated) cache with
    the causal mask, returns (out (1,D), cache_k', cache_v').
    """
    g = GraphFn()
    ck, cv = g.arg((T, D)), g.arg((T, D))
    nk, nv = g.arg((1, D)), g.arg((1, D))
    q = g.arg((1, D))
    mask = g.arg((1, T))

    ck2 = g.write_row(ck, nk, t)
    cv2 = g.write_row(cv, nv, t)
    scores = g.matmul(q, ck2, transpose_b=True)   # (1,T) = q · cache_kᵀ
    masked = g.masked_fill(scores, mask, value=-1e9)
    probs = g.softmax(masked)                      # (1,T)
    out = g.matmul(probs, cv2)                     # (1,D)
    g.ret(out, ck2, cv2)                           # multi-result: out + new caches
    return g


def test_stateful_decode_loop_matches_full_attention():
    """Thread the KV cache across T decode steps; match a full-attention oracle.

    Each step writes the new K/V into the cache and attends over the active
    prefix (mask). The output at step t must equal full causal attention of the
    t-th query over keys/values 0..t. State flows through the production lane.
    """
    rng = np.random.default_rng(0)
    T, D = 6, 8
    scale = np.float32(1.0 / np.sqrt(D))
    K = (rng.standard_normal((T, D))).astype(np.float32)
    V = (rng.standard_normal((T, D))).astype(np.float32)
    Q = (rng.standard_normal((T, D)) * scale).astype(np.float32)  # scale folded

    cache_k = np.zeros((T, D), np.float32)
    cache_v = np.zeros((T, D), np.float32)
    outs = []
    for t in range(T):
        mask = np.zeros((1, T), np.float32)
        mask[0, : t + 1] = 1.0  # attend to positions 0..t
        g = _decode_step_graph(T, D, t)
        out_t, cache_k, cache_v = g.run(
            cache_k, cache_v, K[t : t + 1], V[t : t + 1], Q[t : t + 1], mask
        )
        outs.append(out_t[0])
    got = np.stack(outs)

    # full causal-attention oracle
    scores = (Q @ K.T)
    causal = np.tril(np.ones((T, T)))
    scores = np.where(causal != 0, scores, -1e9)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    ref = p @ V
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    # state actually accumulated: the final cache holds all of K and V.
    np.testing.assert_allclose(cache_k, K, rtol=1e-5)
    np.testing.assert_allclose(cache_v, V, rtol=1e-5)
