"""Track-R (ReplaySSM) Phase 3 — speculative SSM decode via cursor rollback.

Two things are proven:

1. :func:`tessera.speculative.advance_ssm` commits an accepted speculative
   prefix on an ``SSMStateHandle`` by rewinding the ring-buffer cursor (the SSM
   sibling of ``advance_kv``), with guards.

2. **greedy-spec ≡ greedy-AR** — the DFlash invariant for SSMs: a real
   argmax-feedback selective-SSM language model decoded with speculation +
   rollback produces the *identical token sequence* to pure autoregressive
   greedy decode, regardless of draft quality. Speculation changes speed, never
   output.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import SSMStateHandle
from tessera.speculative import advance_ssm


# ─────────────────────────────────────────────────────────────────────────
# advance_ssm contract
# ─────────────────────────────────────────────────────────────────────────

def _filled(count, *, cap=64, spec=4):
    h = SSMStateHandle(batch=1, num_channels=3, state_dim=2,
                       a=np.array([-1.0, -0.5, -0.2]), capacity=cap, spec_window=spec)
    rng = np.random.default_rng(count)
    for _ in range(count):
        h.append(rng.standard_normal((1, 3)), rng.standard_normal((1, 3)),
                 rng.standard_normal((1, 2)))
    return h


def test_advance_ssm_rolls_back_rejected():
    h = _filled(2)            # 2 committed
    for _ in range(3):        # 3 drafts
        h.append(np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 2)))
    assert h.count == 5
    advance_ssm(h, 1, num_drafts=3)   # accept 1 of 3 → roll back 2
    assert h.count == 3


def test_advance_ssm_accept_all_and_none():
    h = _filled(2)
    for _ in range(3):
        h.append(np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 2)))
    advance_ssm(h, 3, num_drafts=3)   # accept all → no rollback
    assert h.count == 5
    h2 = _filled(2)
    for _ in range(3):
        h2.append(np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 2)))
    advance_ssm(h2, 0, num_drafts=3)  # reject all → back to committed
    assert h2.count == 2


@pytest.mark.parametrize("bad", [
    dict(num_accepted=-1, num_drafts=3),
    dict(num_accepted=4, num_drafts=3),     # accepted > drafts
])
def test_advance_ssm_guards_value(bad):
    h = _filled(2)
    for _ in range(3):
        h.append(np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 2)))
    with pytest.raises(ValueError):
        advance_ssm(h, bad["num_accepted"], num_drafts=bad["num_drafts"])


def test_advance_ssm_guards_drafts_exceed_count():
    h = _filled(2)            # only 2 live, no drafts appended
    with pytest.raises(ValueError):
        advance_ssm(h, 1, num_drafts=3)


def test_advance_ssm_rejects_wrong_handle():
    with pytest.raises(TypeError):
        advance_ssm(object(), 1, num_drafts=2)


# ─────────────────────────────────────────────────────────────────────────
# greedy-spec ≡ greedy-AR  (a real argmax-feedback SSM LM)
# ─────────────────────────────────────────────────────────────────────────

class _TinySSMLM:
    """A minimal selective-SSM language model with argmax token feedback.

    Token -> embedding -> (delta, b, c) -> SSMStateHandle.step -> readout ->
    argmax next token.  Deterministic; the same weights serve as draft and
    target (the draft just reads through a *different* head so it guesses
    wrong often enough to exercise rejection)."""

    def __init__(self, vocab, dim, state, *, capacity, spec_window, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.D, self.N = vocab, dim, state
        self.E = rng.standard_normal((vocab, dim)) * 0.5
        self.Wd = rng.standard_normal((dim, dim)) * 0.3
        self.Wb = rng.standard_normal((dim, state)) * 0.3
        self.Wc = rng.standard_normal((dim, state)) * 0.3
        self.Wo = rng.standard_normal((dim, vocab)) * 0.5
        self.Wo_draft = rng.standard_normal((dim, vocab)) * 0.5  # cheap wrong head
        self.a = -np.abs(rng.standard_normal(dim)) - 0.05
        self.capacity, self.spec_window = capacity, spec_window

    def new_state(self):
        return SSMStateHandle(batch=1, num_channels=self.D, state_dim=self.N,
                              a=self.a, capacity=self.capacity,
                              spec_window=self.spec_window)

    def _feat(self, tok):
        x = self.E[tok]                                  # (D,)
        delta = np.logaddexp(0.0, x @ self.Wd)           # softplus > 0  (D,)
        b = x @ self.Wb                                  # (N,)
        c = x @ self.Wc                                  # (N,)
        return x, delta, b, c

    def feed(self, handle, tok, *, auto_flush=True):
        """Consume ``tok``; return the target's greedy next-token prediction.

        ``auto_flush`` is left on for committed (autoregressive) tokens and
        turned off while speculatively feeding a draft burst, so the drafts
        stay live in the ring buffer and remain rollback-able (a mid-burst
        flush would fold them into the checkpoint irreversibly)."""
        x, delta, b, c = self._feat(tok)
        handle.append(delta[None], x[None], b[None], auto_flush=auto_flush)
        y = handle.read_output(c[None])                       # (1, D)
        return int(np.argmax(y[0] @ self.Wo))

    def draft_propose(self, handle, tok, k):
        """Cheap draft: read the *draft* head off a cloned state, greedily, k
        steps ahead.  Wrong often (different head) → exercises rollback."""
        scratch = handle.clone()
        cur, out = tok, []
        for _ in range(k):
            x, delta, b, c = self._feat(cur)
            y = scratch.step(delta[None], x[None], b[None], c[None])
            cur = int(np.argmax(y[0] @ self.Wo_draft))
            out.append(cur)
        return out

    # -- pure autoregressive greedy --
    def decode_ar(self, start, steps):
        h = self.new_state()
        seq = [start]
        pending = self.feed(h, start)        # greedy next after consuming start
        for _ in range(steps):
            seq.append(pending)
            pending = self.feed(h, pending)
        return seq

    # -- speculative greedy via the ring buffer + advance_ssm --
    def decode_spec(self, start, steps, k):
        h = self.new_state()
        seq = [start]
        pending = self.feed(h, start)        # target greedy next (P1)
        stats = {"accepted": 0, "rejected": 0, "blocks": 0}
        while len(seq) < steps + 1:
            # Flush only at the block boundary, reserving room for the whole
            # k-draft burst + the bonus token, so no flush fires mid-burst.
            if h.should_flush(k + 1):
                h.flush()
            drafts = self.draft_propose(h, seq[-1], k)
            # Speculatively feed all k drafts (no mid-burst flush); record the
            # target's prediction after each.
            preds = [self.feed(h, d, auto_flush=False) for d in drafts]
            # Acceptance: longest prefix where draft == the target's expected
            # token (P1, then each accepted draft's successor prediction).
            accepted, expected = 0, pending
            for j in range(k):
                if drafts[j] == expected:
                    accepted += 1
                    expected = preds[j]
                else:
                    break
            bonus = expected                 # target's correction / continuation
            advance_ssm(h, accepted, num_drafts=k)   # roll back rejected drafts
            seq.extend(drafts[:accepted])
            stats["accepted"] += accepted
            stats["rejected"] += k - accepted
            stats["blocks"] += 1
            if len(seq) >= steps + 1:
                break
            # Feed the bonus token (the actual next sequence token) and update
            # the target's pending prediction.  Committed, but kept in-buffer
            # (no flush) until the next block boundary.
            pending = self.feed(h, bonus, auto_flush=False)
            seq.append(bonus)
        return seq[: steps + 1], stats


@pytest.mark.parametrize("k,capacity,spec_window", [
    (3, 64, 4),     # no flush during a block
    (4, 12, 4),     # small ring + spec reservation → flushes between blocks
    (2, 7, 2),      # tight ring
])
def test_greedy_spec_equals_greedy_ar(k, capacity, spec_window):
    lm = _TinySSMLM(vocab=6, dim=8, state=4, capacity=capacity,
                    spec_window=spec_window, seed=3)
    steps = 40
    ar = lm.decode_ar(start=1, steps=steps)
    spec, stats = lm.decode_spec(start=1, steps=steps, k=k)
    assert spec == ar, f"spec diverged from AR:\n  ar  ={ar}\n  spec={spec}"
    # The proof is only meaningful if BOTH paths actually fired.
    assert stats["accepted"] > 0, "no draft was ever accepted — spec path untested"
    assert stats["rejected"] > 0, "no draft was ever rejected — rollback untested"
