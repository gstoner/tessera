"""AD-JET-STRUCT-1 acceptance — structured jets + the estimator mode.

Four proof obligations, per AUTODIFF_NEXTGEN_PLAN §7:

1. **Anchoring to production** — order 0 of every structured jet equals
   the CANONICAL forward (`ops.*`), and order 1 equals the registered
   hand JVP along the same direction. The jets never self-certify.
2. **Law 4, structured** — the jet-vs-nested differential proof per
   family: order-k coefficients agree with the k-nested dual tower on
   the diagonal seed. The nested reference here is a genuinely
   independent implementation (a recursive dual-number array algebra,
   2ᵏ-cost — the very path §3.1 exists to retire), and the §3.1
   factorial bookkeeping is spelled by the AD-WEIL-1
   ``coefficient_scaling="derivative"`` key: the tower's top mixed term
   IS the derivative-scaled extract. Mutation-tested: a corrupted jet
   fails the comparison.
3. **`control_at_order = 0` + Law 5 extended** — max selection by the
   primal only; at exact ties `jet_reduce_max` implements the declared
   ``SUBGRAD_SPLIT`` share, consistent with the first-order rules; and
   the flash_attn jet is EXACT at score ties (softmax's shift
   invariance), verified against the nested tower at a manufactured tie.
4. **Estimator (§3.7 / Law 6)** — unbiasedness with exactness for the
   diagonal-quadratic/Rademacher pair, mean convergence under fixed
   Philox keys, bit reproducibility, split-key independence, and
   fail-closed unknown distributions / non-scalar programs.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops
from tessera.autodiff import jet as J
from tessera.autodiff.algebra import TruncatedJet
from tessera.autodiff.errors import TesseraAutodiffError
from tessera.autodiff.derivative_contract import RETIRED_HAND_RULES
from tessera.autodiff.jvp import _JVPS


def _oracle_jvp(name):
    """The hand JVP as the anchor. Since AD-RETIRE-2 the registered rules
    for softmax/logsumexp/rmsnorm ARE jet-derived, so anchoring against
    the registry would be circular; the displaced oracle (#31) is the
    independent reference. flash_attn is not retired — its registry entry
    is still the hand rule."""
    pair = RETIRED_HAND_RULES.get(name)
    return pair[0] if pair is not None else _JVPS[name]
from tessera.rng import RNGKey


# ── the independent nested-dual array tower (the 2ᵏ reference) ───────────────


class ND:
    """One dual level; payloads are ndarrays or deeper NDs."""

    __slots__ = ("re", "du")

    def __init__(self, re, du):
        self.re, self.du = re, du


def nd_lift_diagonal(x: np.ndarray, v: np.ndarray, depth: int):
    """x + Σᵢ εᵢ·v — the §3.1 diagonal seed, εᵢ² = 0 per level."""
    if depth == 0:
        return np.asarray(x, dtype=np.float64)
    return ND(nd_lift_diagonal(x, v, depth - 1),
              nd_lift_diagonal(v, np.zeros_like(np.asarray(v)), depth - 1))


def nd_const(x, depth: int):
    if depth == 0:
        return np.asarray(x, dtype=np.float64)
    z = np.zeros_like(np.asarray(x, dtype=np.float64))
    return ND(nd_const(x, depth - 1), nd_const(z, depth - 1))


def _is_nd(a):
    return isinstance(a, ND)


def nd_add(a, b):
    if _is_nd(a):
        return ND(nd_add(a.re, b.re), nd_add(a.du, b.du))
    return a + b


def nd_sub(a, b):
    if _is_nd(a):
        return ND(nd_sub(a.re, b.re), nd_sub(a.du, b.du))
    return a - b


def nd_mul(a, b):
    if _is_nd(a):
        return ND(nd_mul(a.re, b.re),
                  nd_add(nd_mul(a.re, b.du), nd_mul(a.du, b.re)))
    return a * b


def nd_matmul(a, b):
    if _is_nd(a):
        return ND(nd_matmul(a.re, b.re),
                  nd_add(nd_matmul(a.re, b.du), nd_matmul(a.du, b.re)))
    return np.matmul(a, b)


def nd_scale(a, s: float):
    if _is_nd(a):
        return ND(nd_scale(a.re, s), nd_scale(a.du, s))
    return a * s


def nd_exp(a):
    if _is_nd(a):
        e = nd_exp(a.re)
        return ND(e, nd_mul(a.du, e))
    return np.exp(a)


def nd_log(a):
    if _is_nd(a):
        return ND(nd_log(a.re), nd_mul(a.du, nd_reciprocal(a.re)))
    return np.log(a)


def nd_reciprocal(a):
    if _is_nd(a):
        r = nd_reciprocal(a.re)
        return ND(r, nd_scale(nd_mul(a.du, nd_mul(r, r)), -1.0))
    return 1.0 / a


def nd_sqrt(a):
    if _is_nd(a):
        s = nd_sqrt(a.re)
        return ND(s, nd_scale(nd_mul(a.du, nd_reciprocal(s)), 0.5))
    return np.sqrt(a)


def nd_sum(a, axis=None, keepdims=False):
    if _is_nd(a):
        return ND(nd_sum(a.re, axis, keepdims), nd_sum(a.du, axis, keepdims))
    return np.sum(a, axis=axis, keepdims=keepdims)


def nd_mean(a, axis=None, keepdims=False):
    if _is_nd(a):
        return ND(nd_mean(a.re, axis, keepdims), nd_mean(a.du, axis, keepdims))
    return np.mean(a, axis=axis, keepdims=keepdims)


def nd_swap(a):
    if _is_nd(a):
        return ND(nd_swap(a.re), nd_swap(a.du))
    return np.swapaxes(a, -1, -2)


def nd_sub_const(a, m):
    """Subtract an order-0 array from the ∅ component only —
    `control_at_order = 0` in tower form."""
    if _is_nd(a):
        return ND(nd_sub_const(a.re, m), a.du)
    return a - m


def nd_where0(mask, fill: float, a):
    if _is_nd(a):
        return ND(nd_where0(mask, fill, a.re), nd_where0(mask, 0.0, a.du))
    return np.where(mask, fill, a)


def nd_primal(a):
    while _is_nd(a):
        a = a.re
    return a


def nd_top(a, depth: int):
    """The ε₁⋯ε_k coefficient: du taken at every level. Carries k! times
    the Taylor coefficient (§3.1) — i.e. the derivative-scaled value."""
    for _ in range(depth):
        a = a.du
    return np.asarray(a, dtype=np.float64)


def nd_softmax(z, axis=-1):
    m = np.max(nd_primal(z), axis=axis, keepdims=True)
    e = nd_exp(nd_sub_const(z, m))
    s = nd_sum(e, axis=axis, keepdims=True)
    return nd_mul(e, nd_reciprocal(s))


def nd_rmsnorm(x, gamma, eps):
    ms = nd_mean(nd_mul(x, x), axis=-1, keepdims=True)
    ms = nd_add(ms, nd_const(np.full_like(nd_primal(ms), eps),
                             _depth_of(ms)))
    inv = nd_reciprocal(nd_sqrt(ms))
    out = nd_mul(x, inv)
    return nd_mul(out, nd_const(np.broadcast_to(gamma, nd_primal(out).shape),
                                _depth_of(out)))


def _depth_of(a):
    d = 0
    while _is_nd(a):
        d, a = d + 1, a.re
    return d


def nd_flash_attn(Q, K, V, *, scale, causal, block_size):
    depth = _depth_of(Q)
    q_len = nd_primal(Q).shape[-2]
    k_len = nd_primal(K).shape[-2]
    d_v = nd_primal(V).shape[-1]
    kT = nd_swap(K)
    m_run = np.full(nd_primal(Q).shape[:-1] + (1,), -np.inf)
    ell = nd_const(np.zeros(nd_primal(Q).shape[:-1] + (1,)), depth)
    out = nd_const(np.zeros(nd_primal(Q).shape[:-2] + (q_len, d_v)), depth)

    def nd_slice_cols(a, start, stop):
        if _is_nd(a):
            return ND(nd_slice_cols(a.re, start, stop),
                      nd_slice_cols(a.du, start, stop))
        return a[..., :, start:stop]

    def nd_slice_rows(a, start, stop):
        if _is_nd(a):
            return ND(nd_slice_rows(a.re, start, stop),
                      nd_slice_rows(a.du, start, stop))
        return a[..., start:stop, :]

    for start in range(0, k_len, block_size):
        stop = min(start + block_size, k_len)
        scores = nd_scale(nd_matmul(Q, nd_slice_cols(kT, start, stop)), scale)
        if causal:
            cols = np.arange(start, stop)[None, :]
            rows = np.arange(q_len)[:, None]
            mask = cols > rows + max(k_len - q_len, 0)
            scores = nd_where0(mask, -np.inf, scores)
        blk_max = np.max(nd_primal(scores), axis=-1, keepdims=True)
        m_new = np.maximum(m_run, blk_max)
        with np.errstate(invalid="ignore"):
            alpha = np.where(np.isneginf(m_run), 0.0, np.exp(m_run - m_new))
        p = nd_exp(nd_sub_const(scores, m_new))
        alpha_c = nd_const(alpha, depth)
        ell = nd_add(nd_mul(ell, alpha_c),
                     nd_sum(p, axis=-1, keepdims=True))
        out = nd_add(nd_mul(out, alpha_c),
                     nd_matmul(p, nd_slice_rows(V, start, stop)))
        m_run = m_new
    return nd_mul(out, nd_reciprocal(ell))


# ── 1. anchoring: order 0 = canonical forward, order 1 = hand JVP ────────────


def test_structured_jets_anchor_to_canonical_forward_and_hand_jvp():
    rng = np.random.default_rng(3)
    W = TruncatedJet(3)
    z = rng.standard_normal((2, 5))
    dz = rng.standard_normal((2, 5))
    g = rng.standard_normal(5)
    b = rng.standard_normal(5)

    c = J.jet_softmax(W, J.jet_lift(W, z, dz))
    np.testing.assert_allclose(c[0], np.asarray(ops.softmax(z)), atol=1e-14)
    _, t = _oracle_jvp("softmax")((z,), (dz,))
    np.testing.assert_allclose(c[1], t, atol=1e-13)

    c = J.jet_logsumexp(W, J.jet_lift(W, z, dz), axis=-1)
    np.testing.assert_allclose(c[0], np.asarray(ops.logsumexp(z, axis=-1)),
                               atol=1e-14)
    _, t = _oracle_jvp("logsumexp")((z,), (dz,), axis=-1)
    np.testing.assert_allclose(c[1], t, atol=1e-13)

    c = J.jet_rmsnorm(W, J.jet_lift(W, z, dz), gamma=g, eps=1e-5)
    np.testing.assert_allclose(c[0], np.asarray(ops.rmsnorm(z, g, eps=1e-5)),
                               atol=1e-13)
    # The registered hand JVP is the gamma-less core (x-only primal);
    # compare against the jet of the same function.
    c_plain = J.jet_rmsnorm(W, J.jet_lift(W, z, dz), gamma=None, eps=1e-5)
    _, t = _oracle_jvp("rmsnorm")((z,), (dz,), eps=1e-5)
    np.testing.assert_allclose(c_plain[1], t, atol=1e-10)

    c = J.jet_layer_norm(W, J.jet_lift(W, z, dz), gamma=g, beta=b, eps=1e-5)
    np.testing.assert_allclose(
        c[0], np.asarray(ops.layer_norm(z, g, b, eps=1e-5)), atol=1e-13)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block_size", [1, 3, 4])
def test_flash_attn_jet_anchors_across_block_sizes(causal, block_size):
    """The ONLINE recurrence itself is what's proven: every blocking of
    the key axis produces the identical jet (order 0 = canonical, order 1
    = hand JVP), so the running (m, ℓ, o) rescale logic carries no
    block-size dependence."""
    rng = np.random.default_rng(11)
    W = TruncatedJet(2)
    Q = rng.standard_normal((1, 2, 4, 3))
    K = rng.standard_normal((1, 2, 4, 3))
    V = rng.standard_normal((1, 2, 4, 3))
    dQ = rng.standard_normal(Q.shape)
    dK = rng.standard_normal(K.shape)
    dV = rng.standard_normal(V.shape)

    c = J.jet_flash_attn(
        W, J.jet_lift(W, Q, dQ), J.jet_lift(W, K, dK), J.jet_lift(W, V, dV),
        causal=causal, block_size=block_size,
    )
    np.testing.assert_allclose(
        c[0], np.asarray(ops.flash_attn(Q, K, V, causal=causal)), atol=1e-13)
    _, t = _JVPS["flash_attn"]((Q, K, V), (dQ, dK, dV), causal=causal)
    np.testing.assert_allclose(c[1], np.asarray(t), atol=1e-7)


def test_flash_attn_jet_carries_attn_bias():
    rng = np.random.default_rng(13)
    W = TruncatedJet(2)
    Q = rng.standard_normal((1, 2, 3, 3))
    K = rng.standard_normal((1, 2, 5, 3))
    V = rng.standard_normal((1, 2, 5, 3))
    bias = rng.standard_normal((1, 3, 5))[:, None, :, :] * 0 + \
        rng.standard_normal((1, 1, 3, 5))
    c = J.jet_flash_attn(
        W, J.jet_const(W, Q), J.jet_const(W, K), J.jet_const(W, V),
        attn_bias=bias[0, 0], block_size=2,
    )
    ref = np.asarray(ops.flash_attn(Q, K, V, attn_bias=bias[0, 0]))
    np.testing.assert_allclose(c[0], ref, atol=1e-13)


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_flash_attn_jet_survives_a_leading_padding_mask(block_size):
    """A row masked only in its LEADING key blocks (ordinary left padding)
    is finite in the canonical forward. The online recurrence used to shift
    by a running max of −inf, and −inf − (−inf) = NaN contaminated ℓ and o
    for every later block through the alpha rescale — a unilateral
    divergence from `ops.flash_attn`, reachable through the documented
    attn_bias substrate (2026-08-29 review, P2)."""
    rng = np.random.default_rng(23)
    W = TruncatedJet(2)
    Q = rng.standard_normal((1, 4, 3))
    K = rng.standard_normal((1, 4, 3))
    V = rng.standard_normal((1, 4, 3))
    dQ = rng.standard_normal(Q.shape)
    bias = np.zeros((1, 4, 4))
    bias[0, 1, 0:2] = -np.inf

    c = J.jet_flash_attn(
        W, J.jet_lift(W, Q, dQ), J.jet_const(W, K), J.jet_const(W, V),
        attn_bias=bias, block_size=block_size,
    )
    ref = np.asarray(ops.flash_attn(Q, K, V, attn_bias=bias))
    assert np.all(np.isfinite(ref)), "the canonical forward is finite here"
    assert np.all(np.isfinite(c[0])) and np.all(np.isfinite(c[1]))
    np.testing.assert_allclose(c[0], ref, atol=1e-13)


# ── 2. Law 4, structured: jet ≡ k-nested duals on the diagonal seed ──────────


def _deriv_extract(order: int, coeffs, k: int) -> np.ndarray:
    """k-th derivative from Taylor coefficients — the AD-WEIL-1
    ``coefficient_scaling="derivative"`` key doing §3.1's factorial
    bookkeeping as API."""
    return np.asarray(
        TruncatedJet(order, coefficient_scaling="derivative").extract(
            coeffs, k)
    )


@pytest.mark.parametrize("k", [2, 3])
def test_law4_softmax_jet_matches_nested_tower(k):
    rng = np.random.default_rng(17)
    z = rng.standard_normal((2, 4))
    v = rng.standard_normal((2, 4))
    W = TruncatedJet(k)
    coeffs = J.jet_softmax(W, J.jet_lift(W, z, v))
    tower = nd_softmax(nd_lift_diagonal(z, v, k))
    np.testing.assert_allclose(
        _deriv_extract(k, coeffs, k), nd_top(tower, k), atol=1e-11)


@pytest.mark.parametrize("k", [2, 3])
def test_law4_rmsnorm_jet_matches_nested_tower(k):
    rng = np.random.default_rng(19)
    x = rng.standard_normal((2, 4))
    v = rng.standard_normal((2, 4))
    g = rng.standard_normal(4)
    W = TruncatedJet(k)
    coeffs = J.jet_rmsnorm(W, J.jet_lift(W, x, v), gamma=g, eps=1e-5)
    tower = nd_rmsnorm(nd_lift_diagonal(x, v, k), g, 1e-5)
    np.testing.assert_allclose(
        _deriv_extract(k, coeffs, k), nd_top(tower, k), atol=1e-11)


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("causal", [False, True])
def test_law4_flash_attn_jet_matches_nested_tower(k, causal):
    """The centerpiece: order-k directional derivatives of the ONLINE
    attention recurrence agree with the k-nested (2ᵏ-cost) dual tower —
    the differential proof that would gate any hand-rule retirement."""
    rng = np.random.default_rng(23)
    Q = rng.standard_normal((1, 3, 3))
    K = rng.standard_normal((1, 4, 3))
    V = rng.standard_normal((1, 4, 3))
    dQ = rng.standard_normal(Q.shape)
    dK = rng.standard_normal(K.shape)
    dV = rng.standard_normal(V.shape)
    scale = 1.0 / np.sqrt(Q.shape[-1])

    W = TruncatedJet(k)
    coeffs = J.jet_flash_attn(
        W, J.jet_lift(W, Q, dQ), J.jet_lift(W, K, dK), J.jet_lift(W, V, dV),
        causal=causal, block_size=2,
    )
    tower = nd_flash_attn(
        nd_lift_diagonal(Q, dQ, k), nd_lift_diagonal(K, dK, k),
        nd_lift_diagonal(V, dV, k), scale=scale, causal=causal, block_size=2,
    )
    np.testing.assert_allclose(
        _deriv_extract(k, coeffs, k), nd_top(tower, k), atol=1e-10)


def test_law4_matmul_jet_is_exact_by_nilpotency():
    """Bilinear ⇒ the jet-matmul convolution is EXACT (tolerance at fp
    rounding, not truncation) against the nested tower — Law 2's
    polynomial-exactness claim in structured form."""
    rng = np.random.default_rng(29)
    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((4, 2))
    dA = rng.standard_normal(A.shape)
    dB = rng.standard_normal(B.shape)
    k = 3
    W = TruncatedJet(k)
    coeffs = J.jet_matmul(W, J.jet_lift(W, A, dA), J.jet_lift(W, B, dB))
    tower = nd_matmul(nd_lift_diagonal(A, dA, k), nd_lift_diagonal(B, dB, k))
    np.testing.assert_allclose(
        _deriv_extract(k, coeffs, k), nd_top(tower, k), atol=1e-12)
    # order ≥ 3 of a bilinear map along one direction is identically zero
    assert not np.any(coeffs[3])


def test_law4_proof_has_teeth():
    """Mutation control: corrupting one jet coefficient must fail the
    nested comparison — the proof can fail."""
    rng = np.random.default_rng(31)
    z = rng.standard_normal((2, 4))
    v = rng.standard_normal((2, 4))
    k = 2
    W = TruncatedJet(k)
    coeffs = J.jet_softmax(W, J.jet_lift(W, z, v))
    coeffs = list(coeffs)
    coeffs[k] = coeffs[k] * 1.01  # planted 1% error
    tower = nd_softmax(nd_lift_diagonal(z, v, k))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            _deriv_extract(k, coeffs, k), nd_top(tower, k), atol=1e-11)


# ── 3. control_at_order = 0 + Law 5 extended ─────────────────────────────────


def test_jet_reduce_max_splits_at_exact_ties():
    """At an exact tie the declared SUBGRAD_SPLIT policy governs the
    higher-order coefficients: the equal-share average, mass conserved —
    the first-order selection Law 5 pins, extended upward."""
    W = TruncatedJet(2)
    z = np.array([[1.0, 3.0, 3.0, 0.0]])
    v = np.array([[10.0, 2.0, 6.0, 5.0]])
    c = J.jet_reduce_max(W, W.lift(z, v), axis=-1)
    np.testing.assert_allclose(c[0], [3.0])
    np.testing.assert_allclose(c[1], [(2.0 + 6.0) / 2.0])

    # Off-tie: hard selection of the argmax slot.
    z2 = np.array([[1.0, 7.0, 3.0, 0.0]])
    c2 = J.jet_reduce_max(W, W.lift(z2, v), axis=-1)
    np.testing.assert_allclose(c2[1], [2.0])


def test_flash_attn_jet_is_exact_at_score_ties():
    """`control_at_order = 0` for the running max is EXACT even at ties:
    softmax's shift invariance cancels the m-dependence identically, so
    the jet agrees with the nested tower at a manufactured score tie."""
    rng = np.random.default_rng(37)
    Q = np.ones((1, 2, 2))          # constant Q rows ⇒ tied scores when
    K = np.ones((1, 3, 2))          # K rows coincide
    V = rng.standard_normal((1, 3, 2))
    dQ = rng.standard_normal(Q.shape)
    dK = rng.standard_normal(K.shape)
    dV = rng.standard_normal(V.shape)
    scale = 1.0 / np.sqrt(2.0)
    k = 2
    W = TruncatedJet(k)
    coeffs = J.jet_flash_attn(
        W, J.jet_lift(W, Q, dQ), J.jet_lift(W, K, dK), J.jet_lift(W, V, dV),
        block_size=2,
    )
    tower = nd_flash_attn(
        nd_lift_diagonal(Q, dQ, k), nd_lift_diagonal(K, dK, k),
        nd_lift_diagonal(V, dV, k), scale=scale, causal=False, block_size=2,
    )
    np.testing.assert_allclose(
        _deriv_extract(k, coeffs, k), nd_top(tower, k), atol=1e-10)


# ── 4. §3.7 estimator: unbiased, deterministic, fail-closed ──────────────────


def _quadratic_jet_program(A: np.ndarray):
    """f(x) = ½ xᵀ A x as a jet program (A symmetric, order-0)."""
    def program(W, x):
        ax = [A @ c for c in x]                     # linear: per-coefficient
        quad = J.jet_mul(W, x, ax)                  # elementwise x ⊙ (Ax)
        half = J.jet_sum(W, quad, axis=None)
        return [0.5 * c for c in half]
    return program


def test_hessian_trace_estimator_is_exact_for_diagonal_rademacher():
    """For diagonal A and Rademacher probes, vᵀAv = tr A on EVERY sample
    (vᵢ² = 1) — the estimator must be exact, not merely converging."""
    D = np.diag([1.5, -2.0, 3.0, 0.5])
    est = J.hessian_trace_estimate(
        _quadratic_jet_program(D), np.zeros(4), RNGKey(5), samples=3,
        distribution="rademacher",
    )
    np.testing.assert_allclose(est, np.trace(D), rtol=1e-12)


def test_hessian_trace_estimator_converges_and_is_deterministic():
    rng = np.random.default_rng(41)
    M = rng.standard_normal((5, 5))
    A = 0.5 * (M + M.T)
    prog = _quadratic_jet_program(A)
    x0 = np.zeros(5)

    est1 = J.hessian_trace_estimate(prog, x0, RNGKey(7), samples=400,
                                    distribution="rademacher")
    est2 = J.hessian_trace_estimate(prog, x0, RNGKey(7), samples=400,
                                    distribution="rademacher")
    assert est1 == est2, "same Philox key must be bit-reproducible"
    np.testing.assert_allclose(est1, np.trace(A), rtol=0.2)

    k1, k2 = RNGKey(7).split(2)
    alt = J.hessian_trace_estimate(prog, x0, k2, samples=400,
                                   distribution="rademacher")
    assert alt != est1, "split keys must give independent draws"

    gauss = J.hessian_trace_estimate(prog, x0, RNGKey(11), samples=4000,
                                     distribution="normal")
    np.testing.assert_allclose(gauss, np.trace(A), rtol=0.25)


def test_estimator_fails_closed():
    prog = _quadratic_jet_program(np.eye(3))
    with pytest.raises(TesseraAutodiffError, match="semantic key"):
        J.hessian_trace_estimate(prog, np.zeros(3), RNGKey(1),
                                 distribution="uniform")
    with pytest.raises(TesseraAutodiffError, match="samples"):
        J.hessian_trace_estimate(prog, np.zeros(3), RNGKey(1), samples=0)

    def vector_prog(W, x):
        return x  # not scalar-output
    with pytest.raises(TesseraAutodiffError, match="scalar-output"):
        J.hessian_trace_estimate(vector_prog, np.zeros(3), RNGKey(1),
                                 samples=1)


def test_jet_map_fails_closed_on_unregistered_scalar():
    W = TruncatedJet(2)
    with pytest.raises(KeyError, match="holonomic"):
        J.jet_map(W, "erfinv", W.lift(np.zeros(2), np.ones(2)))
