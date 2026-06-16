"""Track-R (ReplaySSM) Phase 6 — ``DeltaNetStateHandle`` contract tests.

Headline oracle: **replay ≡ eager gated delta rule** — decoding token-by-token
through the handle (checkpoint + replay ring buffer, flushing when full)
reproduces ``stdlib.delta_rule.gated_delta_rule_recurrent`` per token.  Plus the
ring-buffer ABI (flush/rollback/checkpoint/clone) and a greedy-spec ≡ greedy-AR
proof for a DeltaNet LM using the shared ``speculative.advance_ssm``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import DeltaNetStateHandle
from tessera.stdlib.delta_rule import gated_delta_rule_recurrent
from tessera.speculative import advance_ssm


def _inputs(rng, B, H, dk, dv, S):
    Q = rng.standard_normal((B, H, S, dk))
    K = rng.standard_normal((B, H, S, dk))
    V = rng.standard_normal((B, H, S, dv))
    beta = np.abs(rng.standard_normal((B, H, S))) * 0.5
    decay = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, S))))   # (0, 1)
    return Q, K, V, beta, decay


def _decode(h, Q, K, V, beta, decay, gate=None):
    B, H, S, dv = V.shape
    O = np.zeros((B, H, S, dv))
    for t in range(S):
        gt = None if gate is None else gate[:, :, t, :]
        O[:, :, t, :] = h.step(Q[:, :, t, :], K[:, :, t, :], V[:, :, t, :],
                               beta_t=beta[:, :, t], decay_t=decay[:, :, t], gate_t=gt)
    return O


# ── The headline oracle ─────────────────────────────────────────────────

@pytest.mark.parametrize("B,H,dk,dv,S,cap,spec,erase,gate", [
    (1, 2, 4, 3, 16, 64, 0, True, False),    # no flush
    (2, 2, 4, 4, 20, 5, 0, True, False),     # forced flushes
    (1, 3, 6, 4, 18, 7, 2, True, False),     # spec window
    (2, 2, 4, 3, 16, 6, 0, False, False),    # erase=False → gated linear attn
    (1, 2, 5, 5, 14, 4, 0, True, True),      # output gate
])
def test_replay_equals_eager(B, H, dk, dv, S, cap, spec, erase, gate):
    rng = np.random.default_rng(B * 50 + H + dk + dv + S + cap)
    Q, K, V, beta, decay = _inputs(rng, B, H, dk, dv, S)
    g = rng.standard_normal((B, H, S, dv)) if gate else None
    O_eager = np.asarray(gated_delta_rule_recurrent(
        Q, K, V, beta=beta, decay=decay, gate=g, erase=erase))
    h = DeltaNetStateHandle(batch=B, num_heads=H, key_dim=dk, value_dim=dv,
                            capacity=cap, spec_window=spec, erase=erase)
    O = _decode(h, Q, K, V, beta, decay, gate=g)
    assert np.max(np.abs(O - O_eager)) < 1e-9


def test_materialize_state_equals_eager_state():
    rng = np.random.default_rng(7)
    B, H, dk, dv, S = 2, 2, 4, 3, 12
    Q, K, V, beta, decay = _inputs(rng, B, H, dk, dv, S)
    _, S_eager = gated_delta_rule_recurrent(
        Q, K, V, beta=beta, decay=decay, return_state=True, state_dtype="fp64")
    h = DeltaNetStateHandle(batch=B, num_heads=H, key_dim=dk, value_dim=dv, capacity=5)
    _decode(h, Q, K, V, beta, decay)
    assert np.max(np.abs(h.materialize_state() - np.asarray(S_eager))) < 1e-9


# ── Ring-buffer ABI ─────────────────────────────────────────────────────

def test_route_delegates_to_contract():
    from tessera.compiler import ssm_replay as R
    h = DeltaNetStateHandle(batch=1, num_heads=1, key_dim=2, value_dim=2,
                            capacity=8, spec_window=2)
    for _ in range(4):
        h.append(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
    assert h.should_flush(1) == R.should_flush(4, 8, 2, 1)
    assert h.route_for(1) == R.select_route(4, 8, 2, 1)


def test_flush_preserves_state():
    rng = np.random.default_rng(2)
    B, H, dk, dv = 1, 2, 4, 3
    h = DeltaNetStateHandle(batch=B, num_heads=H, key_dim=dk, value_dim=dv, capacity=8)
    for _ in range(5):
        h.append(rng.standard_normal((B, H, dk)), rng.standard_normal((B, H, dv)),
                 beta_t=np.abs(rng.standard_normal((B, H))) * 0.5)
    before = h.materialize_state()
    h.flush()
    assert h.count == 0
    assert np.max(np.abs(h.materialize_state() - before)) < 1e-12


def test_rollback_full_noop_on_outputs():
    rng = np.random.default_rng(3)
    B, H, dk, dv, S = 1, 1, 4, 3, 6
    Q, K, V, beta, decay = _inputs(rng, B, H, dk, dv, S)
    base = DeltaNetStateHandle(batch=B, num_heads=H, key_dim=dk, value_dim=dv,
                               capacity=64, spec_window=4)
    for t in range(3):
        base.step(Q[:, :, t, :], K[:, :, t, :], V[:, :, t, :],
                  beta_t=beta[:, :, t], decay_t=decay[:, :, t])
    spec = base.clone()
    for _ in range(3):  # 3 drafts then reject all
        spec.step(rng.standard_normal((B, H, dk)), rng.standard_normal((B, H, dk)),
                  rng.standard_normal((B, H, dv)))
    advance_ssm(spec, 0, num_drafts=3)
    assert spec.count == base.count
    for t in range(3, S):
        ob = base.step(Q[:, :, t, :], K[:, :, t, :], V[:, :, t, :],
                       beta_t=beta[:, :, t], decay_t=decay[:, :, t])
        os = spec.step(Q[:, :, t, :], K[:, :, t, :], V[:, :, t, :],
                       beta_t=beta[:, :, t], decay_t=decay[:, :, t])
        assert np.max(np.abs(ob - os)) < 1e-12


def test_checkpoint_restore_roundtrip():
    rng = np.random.default_rng(9)
    B, H, dk, dv, S = 2, 2, 5, 4, 13
    Q, K, V, beta, decay = _inputs(rng, B, H, dk, dv, S)
    h = DeltaNetStateHandle(batch=B, num_heads=H, key_dim=dk, value_dim=dv,
                            capacity=6, spec_window=1)
    _decode(h, Q, K, V, beta, decay)
    h2 = DeltaNetStateHandle.restore(h.checkpoint())
    assert h2.count == h.count
    assert np.max(np.abs(h2.materialize_state() - h.materialize_state())) < 1e-12


def test_clone_isolated():
    h = DeltaNetStateHandle(batch=1, num_heads=1, key_dim=2, value_dim=2, capacity=8)
    h.append(np.ones((1, 1, 2)), np.ones((1, 1, 2)))
    c = h.clone()
    c.append(np.ones((1, 1, 2)) * 9, np.ones((1, 1, 2)))
    assert h.count == 1 and c.count == 2


@pytest.mark.parametrize("kwargs", [
    dict(batch=0, num_heads=1, key_dim=2, value_dim=2),
    dict(batch=1, num_heads=1, key_dim=2, value_dim=2, capacity=0),
    dict(batch=1, num_heads=1, key_dim=2, value_dim=2, spec_window=-1),
])
def test_construction_guards(kwargs):
    with pytest.raises(ValueError):
        DeltaNetStateHandle(**kwargs)


# ── greedy-spec ≡ greedy-AR for a DeltaNet LM ───────────────────────────

class _TinyDeltaLM:
    def __init__(self, vocab, dim, *, capacity, spec_window, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.d = vocab, dim
        self.E = rng.standard_normal((vocab, dim)) * 0.5
        self.Wq = rng.standard_normal((dim, dim)) * 0.3
        self.Wk = rng.standard_normal((dim, dim)) * 0.3
        self.Wv = rng.standard_normal((dim, dim)) * 0.3
        self.wb = rng.standard_normal(dim) * 0.3
        self.wd = rng.standard_normal(dim) * 0.3
        self.Wo = rng.standard_normal((dim, vocab)) * 0.5
        # Draft head = a noisy copy of the target head — a cheap predictor that
        # matches often (→ acceptances) but not always (→ rejections + rollback).
        self.Wo_draft = self.Wo + rng.standard_normal((dim, vocab)) * 0.25
        self.capacity, self.spec_window = capacity, spec_window

    def new_state(self):
        return DeltaNetStateHandle(batch=1, num_heads=1, key_dim=self.d,
                                   value_dim=self.d, capacity=self.capacity,
                                   spec_window=self.spec_window)

    def _feat(self, tok):
        x = self.E[tok]
        q = (x @ self.Wq)[None, None, :]
        k = (x @ self.Wk)[None, None, :]
        v = (x @ self.Wv)[None, None, :]
        beta = np.array([[1.0 / (1.0 + np.exp(-x @ self.wb))]])
        decay = np.array([[1.0 / (1.0 + np.exp(-x @ self.wd))]])
        return q, k, v, beta, decay

    def feed(self, h, tok, *, auto_flush=True):
        q, k, v, beta, decay = self._feat(tok)
        h.append(k, v, beta_t=beta, decay_t=decay, auto_flush=auto_flush)
        o = h.read_output(q)[0, 0]                          # (d,)
        return int(np.argmax(o @ self.Wo))

    def draft_propose(self, h, tok, k):
        scratch = h.clone()
        cur, out = tok, []
        for _ in range(k):
            q, kk, v, beta, decay = self._feat(cur)
            scratch.append(kk, v, beta_t=beta, decay_t=decay, auto_flush=False)
            o = scratch.read_output(q)[0, 0]
            cur = int(np.argmax(o @ self.Wo_draft))
            out.append(cur)
        return out

    def decode_ar(self, start, steps):
        h = self.new_state()
        seq = [start]
        pending = self.feed(h, start)
        for _ in range(steps):
            seq.append(pending)
            pending = self.feed(h, pending)
        return seq

    def decode_spec(self, start, steps, k):
        h = self.new_state()
        seq = [start]
        pending = self.feed(h, start)
        stats = {"accepted": 0, "rejected": 0}
        while len(seq) < steps + 1:
            if h.should_flush(k + 1):
                h.flush()
            drafts = self.draft_propose(h, seq[-1], k)
            preds = [self.feed(h, d, auto_flush=False) for d in drafts]
            accepted, expected = 0, pending
            for j in range(k):
                if drafts[j] == expected:
                    accepted += 1
                    expected = preds[j]
                else:
                    break
            bonus = expected
            advance_ssm(h, accepted, num_drafts=k)
            seq.extend(drafts[:accepted])
            stats["accepted"] += accepted
            stats["rejected"] += k - accepted
            if len(seq) >= steps + 1:
                break
            pending = self.feed(h, bonus, auto_flush=False)
            seq.append(bonus)
        return seq[: steps + 1], stats


@pytest.mark.parametrize("k,capacity,spec_window", [(3, 64, 4), (4, 12, 4), (2, 7, 2)])
def test_greedy_spec_equals_greedy_ar(k, capacity, spec_window):
    lm = _TinyDeltaLM(vocab=6, dim=8, capacity=capacity, spec_window=spec_window, seed=3)
    steps = 36
    ar = lm.decode_ar(start=1, steps=steps)
    spec, stats = lm.decode_spec(start=1, steps=steps, k=k)
    assert spec == ar, f"spec diverged:\n ar  ={ar}\n spec={spec}"
    assert stats["accepted"] > 0 and stats["rejected"] > 0
