"""Workstream B — prefill/decode as different device_verified_jit programs.

A tiny single-layer causal-attention LM exercises the phase contract end to end:
``prefill`` fills a KVCacheHandle, ``decode`` advances one token by consuming that
cache through Workstream A's ``paged_attention``, and the oracle proves the split
equals a monolithic ``forward`` — the schedule separation is semantics-preserving.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream B).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import KVCacheHandle, paged_attention
from tessera.compiler.phase_specialization import (
    Phase, SLO, SchedulePolicy, CacheHandoff, PhaseSpecializedProgram,
    specialize, verify_phase_split)


# ── A tiny causal-attention LM (numpy, float64) ──────────────────────────────


class ToyLM:
    def __init__(self, vocab=7, dim=4, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.D = vocab, dim
        self.E = rng.standard_normal((vocab, dim))
        self.Wq = rng.standard_normal((dim, dim))
        self.Wk = rng.standard_normal((dim, dim))
        self.Wv = rng.standard_normal((dim, dim))
        self.Wout = rng.standard_normal((dim, vocab))

    def _qkv(self, tokens):
        x = self.E[np.asarray(tokens, dtype=int)]          # (n, D)
        return x @ self.Wq, x @ self.Wk, x @ self.Wv

    def forward(self, tokens):
        """Monolithic reference: logits per position, causal attention."""
        Q, K, V = self._qkv(tokens)
        n = Q.shape[0]
        scale = 1.0 / np.sqrt(self.D)
        scores = (Q @ K.T) * scale
        mask = np.triu(np.ones((n, n), bool), k=1)
        scores = np.where(mask, -np.inf, scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        w = np.exp(scores); w /= w.sum(axis=-1, keepdims=True)
        h = w @ V                                          # (n, D)
        return h @ self.Wout                               # (n, V)

    # ── phase-split programs ──
    def prefill(self, tokens):
        tokens = list(np.asarray(tokens).reshape(-1))
        Q, K, V = self._qkv(tokens)
        n = len(tokens)
        cache = KVCacheHandle(num_heads=1, head_dim=self.D, max_seq=n + 64,
                              page_size=8)
        cache.append(K.reshape(n, 1, self.D), V.reshape(n, 1, self.D))
        logits_all = self.forward(tokens)
        return logits_all[-1], cache

    def decode(self, cache: KVCacheHandle, token: int):
        q, k, v = self._qkv([token])
        cache.append(k.reshape(1, 1, self.D), v.reshape(1, 1, self.D))
        Q = q.reshape(1, 1, self.D)                        # (heads, q_len, hd)
        O = paged_attention(Q, cache, causal=False)        # attends all cached
        h = np.asarray(O).reshape(1, self.D)
        return (h @ self.Wout)[0], cache


# ── Contract: policies differ by phase ───────────────────────────────────────


def test_schedule_policies_differ_by_phase():
    p = SchedulePolicy.for_phase(Phase.PREFILL)
    d = SchedulePolicy.for_phase(Phase.DECODE, SLO(max_latency_ms=5.0))
    assert p.tile_strategy == "bulk_throughput" and p.materialize_scores
    assert not p.prefer_resident_kv
    assert d.tile_strategy == "low_latency" and not d.materialize_scores
    assert d.prefer_resident_kv
    assert d.slo.max_latency_ms == 5.0


def test_for_phase_accepts_string():
    assert SchedulePolicy.for_phase("prefill").phase is Phase.PREFILL


def test_cache_handoff_advances_position():
    h = CacheHandoff(state="s0", position=10)
    h2 = h.advanced("s1", 1)
    assert h2.position == 11 and h2.step == 1 and h2.state == "s1"


# ── jit attaches the policy ───────────────────────────────────────────────────


def test_jit_phase_attaches_schedule_policy():
    import tessera

    @tessera.jit(phase="decode", slo=SLO(max_latency_ms=3.0))
    def step(x):
        return tessera.ops.softmax(x)

    assert step.phase is Phase.DECODE
    assert step.schedule_policy.tile_strategy == "low_latency"
    assert step.slo.max_latency_ms == 3.0


def test_jit_without_phase_has_none():
    import tessera

    @tessera.jit
    def step(x):
        return tessera.ops.softmax(x)

    assert step.phase is None and step.schedule_policy is None


# ── The oracle: prefill ▸ decode == monolithic forward ───────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("prompt_len,n_new", [(3, 3), (5, 4), (1, 5)])
def test_phase_split_equals_forward(seed, prompt_len, n_new):
    lm = ToyLM(vocab=7, dim=4, seed=seed)
    prog = specialize(lm.prefill, lm.decode,
                      prefill_slo=SLO(min_throughput_tok_s=1e4),
                      decode_slo=SLO(max_latency_ms=2.0))
    rng = np.random.default_rng(100 + seed)
    prompt = rng.integers(0, 7, size=prompt_len).tolist()

    verdict = verify_phase_split(lm.forward, prog, prompt, n_new)
    assert verdict.is_equivalent, verdict.detail
    assert verdict.tokens_match


def test_generate_matches_manual_kv_loop():
    lm = ToyLM(seed=3)
    prog = specialize(lm.prefill, lm.decode)
    prompt = [1, 2, 3]
    gen = prog.generate(prompt, max_new_tokens=4)

    # Manual reference: recompute full forward at each step (no cache).
    stream = list(prompt)
    ref = []
    for _ in range(4):
        tok = int(np.argmax(lm.forward(stream)[-1]))
        ref.append(tok); stream.append(tok)
    assert gen == ref


def test_handoff_carries_paged_kv_state():
    lm = ToyLM(seed=4)
    prog = specialize(lm.prefill, lm.decode)
    _, handoff = prog.run_prefill([1, 2, 3], prompt_len=3)
    assert handoff.position == 3
    # The handoff state is a real KV cache that conforms to Workstream A's ABI.
    from tessera.cache import as_paged_kv_state, PagedKVState
    assert isinstance(as_paged_kv_state(handoff.state), PagedKVState)
