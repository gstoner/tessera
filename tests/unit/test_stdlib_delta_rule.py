"""Track L (L0/L1/L2) — the *true* gated delta rule, oracle-gated.

L0 (discovery lock): the shipped ``tessera.ops.gated_deltanet`` is gated *linear*
attention (no erase) — it equals our ``erase=False`` path and DIFFERS from the
genuine delta rule when keys correlate.

L1 (recurrence): ``gated_delta_rule_recurrent`` matches an independent
brute-force delta recurrence written in the paper's ``(I − β k kᵀ)`` layout;
``erase=False`` reduces to the existing reference; state carry is consistent.

L2 (chunk UT-transform, the keystone): ``gated_delta_rule_chunked ≡
gated_delta_rule_recurrent`` across ungated / β / fully-gated / output-gated
cases and across chunk sizes — the chunk≡recurrent DESIL proof.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops
from tessera.stdlib import delta_rule as dr


# ── helpers ──────────────────────────────────────────────────────────────────
def _rng(seed):
    return np.random.default_rng(seed)


def _qkv(rng, B=2, H=3, S=12, d_k=5, d_v=4, scale=1.0):
    Q = rng.standard_normal((B, H, S, d_k)) * scale
    K = rng.standard_normal((B, H, S, d_k)) * scale
    V = rng.standard_normal((B, H, S, d_v)) * scale
    return Q, K, V


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _bruteforce_delta(Q, K, V, *, beta=None, decay=None, gate=None):
    """Independent transcription of the gated delta rule in the paper's layout:
    state S ∈ [d_v, d_k], S_t = α_t S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ,
    O_t = S_t q_t.  Deliberately a different layout + explicit Householder matrix
    so it is NOT the same code path as the module under test.
    """
    Q = np.asarray(Q, np.float64); K = np.asarray(K, np.float64); V = np.asarray(V, np.float64)
    B, H, S, d_k = Q.shape
    d_v = V.shape[-1]
    O = np.zeros((B, H, S, d_v))
    I = np.eye(d_k)
    for b in range(B):
        for h in range(H):
            St = np.zeros((d_v, d_k))
            for t in range(S):
                k = K[b, h, t]; v = V[b, h, t]; q = Q[b, h, t]
                a = float(decay[b, h, t]) if decay is not None else 1.0
                bt = float(beta[b, h, t]) if beta is not None else 1.0
                St = a * St @ (I - bt * np.outer(k, k)) + bt * np.outer(v, k)
                O[b, h, t] = St @ q
    if gate is not None:
        O = O * _sig(np.asarray(gate, np.float64))
    return O


# ── L0 — discovery lock ──────────────────────────────────────────────────────
def test_existing_gated_deltanet_is_linear_attention_not_delta():
    """The shipped op == erase-off (linear) path, and DIFFERS from the true
    delta rule once keys carry overlapping directions."""
    rng = _rng(0)
    Q, K, V = _qkv(rng)
    beta = _sig(rng.standard_normal(Q.shape[:3]))
    existing = np.asarray(ops.gated_deltanet(Q, K, V, beta=beta), np.float64)
    linear = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, erase=False)
    delta = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, erase=True)
    # Existing reference IS the linear (no-erase) path.
    np.testing.assert_allclose(existing, linear, rtol=1e-10, atol=1e-10)
    # ...and the genuine delta rule is materially different.
    assert not np.allclose(delta, linear, rtol=1e-3, atol=1e-3)


# ── L1 — the genuine recurrence ──────────────────────────────────────────────
def test_recurrent_matches_independent_bruteforce_ungated():
    rng = _rng(1)
    Q, K, V = _qkv(rng)
    ours = dr.gated_delta_rule_recurrent(Q, K, V)
    ref = _bruteforce_delta(Q, K, V)
    np.testing.assert_allclose(ours, ref, rtol=1e-9, atol=1e-9)


def test_recurrent_matches_bruteforce_fully_gated():
    rng = _rng(2)
    Q, K, V = _qkv(rng)
    beta = _sig(rng.standard_normal(Q.shape[:3]))
    decay = _sig(rng.standard_normal(Q.shape[:3]) + 2.0)   # decay near 1
    ours = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay)
    ref = _bruteforce_delta(Q, K, V, beta=beta, decay=decay)
    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-8)


def test_recurrent_erase_off_equals_existing_reference_with_decay():
    rng = _rng(3)
    Q, K, V = _qkv(rng)
    beta = _sig(rng.standard_normal(Q.shape[:3]))
    decay = _sig(rng.standard_normal(Q.shape[:3]) + 1.5)
    existing = np.asarray(ops.gated_deltanet(Q, K, V, beta=beta, decay=decay), np.float64)
    ours = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, erase=False)
    np.testing.assert_allclose(existing, ours, rtol=1e-9, atol=1e-9)


def test_output_gate_matches_bruteforce():
    rng = _rng(4)
    Q, K, V = _qkv(rng, d_v=4)
    gate = rng.standard_normal((2, 3, 12, 4))
    ours = dr.gated_delta_rule_recurrent(Q, K, V, gate=gate)
    ref = _bruteforce_delta(Q, K, V, gate=gate)
    np.testing.assert_allclose(ours, ref, rtol=1e-9, atol=1e-9)


def test_return_state_shape_and_cross_call_carry():
    rng = _rng(5)
    Q, K, V = _qkv(rng, S=10, d_k=5, d_v=4)
    beta = _sig(rng.standard_normal(Q.shape[:3]))
    # One pass over the whole sequence...
    O_full, S_full = dr.gated_delta_rule_recurrent(
        Q, K, V, beta=beta, return_state=True, state_dtype="fp64")
    assert S_full.shape == (2, 3, 5, 4)
    # ...equals two passes carrying state across the split.
    O1, S1 = dr.gated_delta_rule_recurrent(
        Q[:, :, :6], K[:, :, :6], V[:, :, :6], beta=beta[:, :, :6],
        return_state=True, state_dtype="fp64")
    O2 = dr.gated_delta_rule_recurrent(
        Q[:, :, 6:], K[:, :, 6:], V[:, :, 6:], beta=beta[:, :, 6:], state=S1)
    np.testing.assert_allclose(O_full[:, :, :6], O1, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(O_full[:, :, 6:], O2, rtol=1e-9, atol=1e-9)


# ── L2 — the chunk UT-transform keystone ─────────────────────────────────────
def test_forward_substitution_solves_unit_lower_triangular():
    rng = _rng(6)
    C, d = 7, 3
    A = np.tril(rng.standard_normal((C, C)), k=-1)        # strictly lower
    W = rng.standard_normal((C, d))
    U = dr._forward_substitution(A, W)
    np.testing.assert_allclose((np.eye(C) + A) @ U, W, rtol=1e-10, atol=1e-10)


def test_chunk_equals_recurrent_ungated():
    rng = _rng(7)
    Q, K, V = _qkv(rng, S=20)
    rec = dr.gated_delta_rule_recurrent(Q, K, V)
    ch = dr.gated_delta_rule_chunked(Q, K, V, chunk_size=8)
    np.testing.assert_allclose(ch, rec, rtol=1e-9, atol=1e-9)


def test_chunk_equals_recurrent_with_beta():
    rng = _rng(8)
    Q, K, V = _qkv(rng, S=20)
    beta = _sig(rng.standard_normal(Q.shape[:3]))
    rec = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta)
    ch = dr.gated_delta_rule_chunked(Q, K, V, beta=beta, chunk_size=8)
    np.testing.assert_allclose(ch, rec, rtol=1e-9, atol=1e-9)


def test_chunk_equals_recurrent_fully_gated():
    """The decay-folding proof: γ_t/γ_j ratios in the chunk form must reproduce
    the per-token decay of the recurrence."""
    rng = _rng(9)
    Q, K, V = _qkv(rng, S=24)
    beta = _sig(rng.standard_normal(Q.shape[:3]))
    decay = _sig(rng.standard_normal(Q.shape[:3]) + 2.0)
    rec = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay)
    ch = dr.gated_delta_rule_chunked(Q, K, V, beta=beta, decay=decay, chunk_size=8)
    np.testing.assert_allclose(ch, rec, rtol=1e-8, atol=1e-8)


def test_chunk_equals_recurrent_with_output_gate():
    rng = _rng(10)
    Q, K, V = _qkv(rng, S=16, d_v=4)
    gate = rng.standard_normal((2, 3, 16, 4))
    rec = dr.gated_delta_rule_recurrent(Q, K, V, gate=gate)
    ch = dr.gated_delta_rule_chunked(Q, K, V, gate=gate, chunk_size=8)
    np.testing.assert_allclose(ch, rec, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("chunk_size", [1, 4, 8, 16, 64])
def test_chunk_size_invariance(chunk_size):
    rng = _rng(11)
    Q, K, V = _qkv(rng, S=16)
    beta = _sig(rng.standard_normal(Q.shape[:3]))
    decay = _sig(rng.standard_normal(Q.shape[:3]) + 2.0)
    rec = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay)
    ch = dr.gated_delta_rule_chunked(Q, K, V, beta=beta, decay=decay, chunk_size=chunk_size)
    np.testing.assert_allclose(ch, rec, rtol=1e-8, atol=1e-8)


def test_chunk_state_carry_matches_recurrent_state():
    rng = _rng(12)
    Q, K, V = _qkv(rng, S=18, d_k=5, d_v=4)
    beta = _sig(rng.standard_normal(Q.shape[:3]))
    _, S_rec = dr.gated_delta_rule_recurrent(
        Q, K, V, beta=beta, return_state=True, state_dtype="fp64")
    _, S_ch = dr.gated_delta_rule_chunked(
        Q, K, V, beta=beta, chunk_size=7, return_state=True, state_dtype="fp64")
    np.testing.assert_allclose(S_ch, S_rec, rtol=1e-8, atol=1e-8)


def test_chunk_erase_off_equals_linear_reference():
    rng = _rng(13)
    Q, K, V = _qkv(rng, S=16)
    beta = _sig(rng.standard_normal(Q.shape[:3]))
    existing = np.asarray(ops.gated_deltanet(Q, K, V, beta=beta), np.float64)
    ch = dr.gated_delta_rule_chunked(Q, K, V, beta=beta, chunk_size=8, erase=False)
    np.testing.assert_allclose(ch, existing, rtol=1e-9, atol=1e-9)
