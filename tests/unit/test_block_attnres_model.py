"""Executable mathematical contract for Block AttnRes (arXiv 2603.15031).

Verifies the derived results in docs/audit/compiler/BLOCK_ATTNRES_ROCM_PLAN.md:
the depth-attention VJP (eqs. 11-13, derived there — the paper gives no
backward), the softmax-merge exactness lemma (I.6), operator properties P2-P4,
and the two-phase schedule's equivalence to the naive forward (A4 == A2).

These numpy reference functions are the F4 oracles for the later ROCm/Apple
kernels (plan §III.6); keep them dependency-free (no torch on the fleet).
"""

from __future__ import annotations

import numpy as np
import pytest

EPS = 1e-6


# ---------------------------------------------------------------------------
# Reference implementations (plan Part II)
# ---------------------------------------------------------------------------


def _rms(v: np.ndarray) -> np.ndarray:
    """RMS(v) = sqrt(mean(v^2) + eps) along the last axis, kept-dims."""
    return np.sqrt((v * v).mean(axis=-1, keepdims=True) + EPS)


def _khat(v: np.ndarray) -> np.ndarray:
    """Weightless RMSNorm: k̂(v) = v / RMS(v)."""
    return v / _rms(v)


def depth_attn(w: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A1: depth-attention operator over sources V[m, d] with query w[d].

    Returns (h, alpha, K) so tests and the VJP can share intermediates.
    """
    if V.ndim != 2:
        raise ValueError(f"expected V[m, d], got shape {V.shape}")
    K = _khat(V)
    s = K @ w
    s = s - s.max()  # shift-invariant; matches kernel practice
    a = np.exp(s)
    a = a / a.sum()
    return a @ V, a, K


def depth_attn_vjp(
    w: np.ndarray, V: np.ndarray, g: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A5: analytic VJP (plan eqs. 11-13) of h = depth_attn(w, V) w.r.t. (w, V)."""
    _, a, K = depth_attn(w, V)
    r = _rms(V)
    u = V @ g
    ubar = a @ u
    ds = a * (u - ubar)  # softmax VJP
    dw = ds @ K  # (11)
    dV = a[:, None] * g[None, :]  # value path (12)
    dV = dV + ds[:, None] * (w[None, :] - K * (K @ w)[:, None] / V.shape[-1]) / r  # (13)
    return dw, dV


def attn_with_stats(w: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, float, float]:
    """A3: unnormalized max-shifted attention state (o, m, l) with o/l == depth_attn."""
    K = _khat(V)
    s = K @ w
    m = float(s.max())
    p = np.exp(s - m)
    return p @ V, m, float(p.sum())


def softmax_merge(
    a: tuple[np.ndarray, float, float], b: tuple[np.ndarray, float, float]
) -> tuple[np.ndarray, float, float]:
    """A3: exact, associative, commutative merge of two attention states."""
    o1, m1, l1 = a
    o2, m2, l2 = b
    m = max(m1, m2)
    c1, c2 = np.exp(m1 - m), np.exp(m2 - m)
    return c1 * o1 + c2 * o2, m, c1 * l1 + c2 * l2


def block_attnres_forward(
    x: np.ndarray,
    sublayers: list,
    queries: list[np.ndarray],
    block_sizes: list[int],
) -> np.ndarray:
    """A2: naive Block AttnRes forward over per-token state (single token).

    ``queries`` holds one pseudo-query per sub-layer plus the final
    aggregation query w_{L+1} (GAP-3). ``sublayers`` are callables f_l acting
    on R^d WITHOUT internal residuals.
    """
    if len(queries) != len(sublayers) + 1:
        raise ValueError(
            f"need {len(sublayers) + 1} queries (one per sub-layer + final), got {len(queries)}"
        )
    if sum(block_sizes) != len(sublayers):
        raise ValueError(f"block sizes {block_sizes} do not partition {len(sublayers)} sub-layers")
    blocks = [x]
    li = 0
    for size in block_sizes:
        p = None
        for _ in range(size):
            V = np.stack(blocks if p is None else blocks + [p])  # GAP-2: no zero slot
            h, _, _ = depth_attn(queries[li], V)
            out = sublayers[li](h)
            p = out if p is None else p + out
            li += 1
        blocks.append(p)
    h_out, _, _ = depth_attn(queries[-1], np.stack(blocks))
    return h_out


def block_attnres_two_phase(
    x: np.ndarray,
    sublayers: list,
    queries: list[np.ndarray],
    block_sizes: list[int],
) -> np.ndarray:
    """A4: two-phase schedule — Phase-1 batched inter-block stats, Phase-2 merge."""
    blocks = [x]
    li = 0
    for size in block_sizes:
        base = li
        phase1 = [attn_with_stats(queries[base + i], np.stack(blocks)) for i in range(size)]
        p = None
        for i in range(size):
            o1, m1, l1 = phase1[i]
            if p is None:
                h = o1 / l1
            else:
                o, _, l = softmax_merge((o1, m1, l1), attn_with_stats(queries[base + i], p[None, :]))
                h = o / l
            out = sublayers[li](h)
            p = out if p is None else p + out
            li += 1
        blocks.append(p)
    h_out, _, _ = depth_attn(queries[-1], np.stack(blocks))
    return h_out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D, M = 16, 5


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture()
def sources(rng: np.random.Generator) -> np.ndarray:
    scales = np.array([1.0, 3.0, 0.5, 2.0, 1.0])[:, None]
    return rng.normal(size=(M, D)) * scales


# ---------------------------------------------------------------------------
# Operator properties (plan I.2)
# ---------------------------------------------------------------------------


def test_p2_zero_query_gives_uniform_average(sources: np.ndarray) -> None:
    h, a, _ = depth_attn(np.zeros(D), sources)
    np.testing.assert_allclose(a, np.full(M, 1.0 / M))
    np.testing.assert_allclose(h, sources.mean(axis=0), atol=1e-12)


def test_p3_weights_invariant_under_source_rescale(rng: np.random.Generator, sources: np.ndarray) -> None:
    w = rng.normal(size=D)
    _, a1, _ = depth_attn(w, sources)
    scaled = sources.copy()
    scaled[2] *= 7.0
    _, a2, _ = depth_attn(w, scaled)
    np.testing.assert_allclose(a1, a2, atol=1e-6)


def test_p4_key_norm_gain_absorbed_into_query(rng: np.random.Generator, sources: np.ndarray) -> None:
    w = rng.normal(size=D)
    gain = rng.uniform(0.5, 2.0, size=D)
    K = _khat(sources) * gain  # gained key-norm
    s = K @ w
    s = s - s.max()
    a = np.exp(s)
    h_gained = (a / a.sum()) @ sources
    h_absorbed, _, _ = depth_attn(gain * w, sources)
    np.testing.assert_allclose(h_gained, h_absorbed, atol=1e-12)


def test_p1_convexity_bound(rng: np.random.Generator, sources: np.ndarray) -> None:
    for _ in range(10):
        h, _, _ = depth_attn(rng.normal(size=D), sources)
        assert np.linalg.norm(h) <= np.linalg.norm(sources, axis=-1).max() + 1e-9


# ---------------------------------------------------------------------------
# VJP (plan I.5, eqs. 11-13) vs central finite differences
# ---------------------------------------------------------------------------


def test_vjp_matches_finite_differences(rng: np.random.Generator, sources: np.ndarray) -> None:
    w = rng.normal(size=D)
    g = rng.normal(size=D)

    def loss_w(wv: np.ndarray) -> float:
        return float(g @ depth_attn(wv, sources)[0])

    def loss_v(Vv: np.ndarray) -> float:
        return float(g @ depth_attn(w, Vv)[0])

    dw, dV = depth_attn_vjp(w, sources, g)

    step = 1e-6
    for i in range(D):
        e = np.zeros(D)
        e[i] = step
        fd = (loss_w(w + e) - loss_w(w - e)) / (2 * step)
        assert abs(dw[i] - fd) < 1e-7, f"dw[{i}]: analytic {dw[i]} vs fd {fd}"
    for j in range(M):
        for i in range(D):
            E = np.zeros((M, D))
            E[j, i] = step
            fd = (loss_v(sources + E) - loss_v(sources - E)) / (2 * step)
            assert abs(dV[j, i] - fd) < 1e-7, f"dV[{j},{i}]: analytic {dV[j, i]} vs fd {fd}"


# ---------------------------------------------------------------------------
# Merge lemma (plan I.6) and the two-phase schedule (A4 == A2)
# ---------------------------------------------------------------------------


def test_merge_lemma_exact_over_random_partition_trees(rng: np.random.Generator, sources: np.ndarray) -> None:
    w = rng.normal(size=D)
    h_ref, _, _ = depth_attn(w, sources)
    for trial in range(20):
        idx = rng.permutation(M)
        parts = np.array_split(idx, rng.integers(2, M + 1))
        acc = None
        for part in parts:
            st = attn_with_stats(w, sources[np.sort(part)])
            acc = st if acc is None else softmax_merge(acc, st)
        assert acc is not None
        np.testing.assert_allclose(acc[0] / acc[2], h_ref, atol=1e-12, err_msg=f"trial {trial}")


def test_merge_is_commutative(rng: np.random.Generator, sources: np.ndarray) -> None:
    w = rng.normal(size=D)
    a = attn_with_stats(w, sources[:2])
    b = attn_with_stats(w, sources[2:])
    ab, ba = softmax_merge(a, b), softmax_merge(b, a)
    np.testing.assert_allclose(ab[0], ba[0], atol=1e-12)
    assert ab[1] == ba[1]
    np.testing.assert_allclose(ab[2], ba[2], atol=1e-12)


def _make_net(rng: np.random.Generator, n_sublayers: int) -> list:
    """Random linear+tanh sub-layers WITHOUT internal residuals (plan I.1)."""
    mats = [rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(n_sublayers)]
    return [lambda h, A=A: np.tanh(A @ h) for A in mats]


@pytest.mark.parametrize("block_sizes", [[2, 2, 2], [3, 3], [1, 2, 3], [6]])
def test_two_phase_equals_naive_forward(rng: np.random.Generator, block_sizes: list[int]) -> None:
    n = sum(block_sizes)
    sublayers = _make_net(rng, n)
    queries = [rng.normal(size=D) * 0.5 for _ in range(n + 1)]
    x = rng.normal(size=D)
    naive = block_attnres_forward(x, sublayers, queries, block_sizes)
    two_phase = block_attnres_two_phase(x, sublayers, queries, block_sizes)
    np.testing.assert_allclose(two_phase, naive, atol=1e-12)


def test_full_attnres_is_the_blocksize_one_limit(rng: np.random.Generator) -> None:
    """N = L, S = 1: every source is an individual sub-layer output (plan I.3)."""
    n = 4
    sublayers = _make_net(rng, n)
    queries = [rng.normal(size=D) * 0.5 for _ in range(n + 1)]
    x = rng.normal(size=D)
    h_block = block_attnres_forward(x, sublayers, queries, [1] * n)

    # Explicit Full AttnRes: sigma(l) = (v_0..v_{l-1}); partial always absent.
    outputs = [x]
    for li in range(n):
        h, _, _ = depth_attn(queries[li], np.stack(outputs))
        outputs.append(sublayers[li](h))
    h_full, _, _ = depth_attn(queries[-1], np.stack(outputs))
    np.testing.assert_allclose(h_block, h_full, atol=1e-12)


def test_first_sublayer_of_block_excludes_partial(rng: np.random.Generator) -> None:
    """GAP-2: source count is n (blocks only) at i=1, n+1 afterwards — never a zero slot."""
    def probe(h: np.ndarray) -> np.ndarray:
        return h  # identity sub-layer

    orig = depth_attn
    counts_by_call: list[int] = []

    def counting_depth_attn(w: np.ndarray, V: np.ndarray):
        counts_by_call.append(V.shape[0])
        return orig(w, V)

    sublayers = [probe] * 4
    queries = [rng.normal(size=D) for _ in range(5)]
    globals()["depth_attn"] = counting_depth_attn
    try:
        block_attnres_forward(rng.normal(size=D), sublayers, queries, [2, 2])
    finally:
        globals()["depth_attn"] = orig
    # Block 1: [b0] -> 1 source, then [b0, p] -> 2. Block 2: [b0, b1] -> 2, then 3.
    # Final aggregation: [b0, b1, b2] -> 3.
    assert counts_by_call == [1, 2, 2, 3, 3], counts_by_call


def test_input_validation_errors() -> None:
    with pytest.raises(ValueError, match="expected V"):
        depth_attn(np.zeros(D), np.zeros((2, 3, D)))
    with pytest.raises(ValueError, match="queries"):
        block_attnres_forward(np.zeros(D), [lambda h: h], [np.zeros(D)], [1])
    with pytest.raises(ValueError, match="partition"):
        block_attnres_forward(
            np.zeros(D), [lambda h: h], [np.zeros(D), np.zeros(D)], [2]
        )
