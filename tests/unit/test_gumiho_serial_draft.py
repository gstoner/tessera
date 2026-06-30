"""SD1 — the Gumiho serial-draft speculative-decode loop composes the SD1
primitives (draft → target_verify → spec_accept → cache_commit) into ONE loop, and
its greedy invariant holds: for ANY draft model the emitted sequence is identical
to plain autoregressive decode with the target — speculation changes only the
number of target calls, never the output. The loop wrapper is the bounded
recurrence that lowers to one control_scan device dispatch (the "one backend
loop").
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.speculative import autoregressive_decode, gumiho_serial_draft


def _target(V):
    # A deterministic greedy "target model": next token is a fixed transition of
    # the last token. Any deterministic next-token function works for the invariant.
    return lambda seq: (int(seq[-1]) * 7 + 3) % V


def test_gumiho_equals_ar_with_a_perfect_draft():
    V = 32
    target = _target(V)
    # perfect draft == target → every draft token accepts (max speedup).
    out = gumiho_serial_draft(prompt=[1], draft_next=target, target_next=target,
                              max_new=20, draft_len=4)
    assert out == autoregressive_decode(prompt=[1], target_next=target, max_new=20)


def test_gumiho_equals_ar_with_an_always_wrong_draft():
    V = 32
    target = _target(V)
    wrong = lambda seq: (target(seq) + 1) % V  # noqa: E731 — every draft rejected
    out = gumiho_serial_draft(prompt=[1], draft_next=wrong, target_next=target,
                              max_new=20, draft_len=4)
    # even with zero accepts, the bonus correction per step reproduces AR exactly.
    assert out == autoregressive_decode(prompt=[1], target_next=target, max_new=20)


@pytest.mark.parametrize("draft_len,seed", [(1, 0), (2, 1), (4, 2), (8, 3)])
def test_gumiho_equals_ar_with_a_random_draft(draft_len, seed):
    V = 64
    target = _target(V)
    rng = np.random.default_rng(seed)
    draft = lambda seq: int(rng.integers(0, V))  # noqa: E731 — random draft model
    out = gumiho_serial_draft(prompt=[3, 5], draft_next=draft, target_next=target,
                              max_new=30, draft_len=draft_len)
    assert out == autoregressive_decode(prompt=[3, 5], target_next=target,
                                        max_new=30)


def test_gumiho_emits_exactly_max_new_tokens():
    V = 16
    target = _target(V)
    out = gumiho_serial_draft(prompt=[2], draft_next=target, target_next=target,
                              max_new=13, draft_len=5)
    assert len(out) == 1 + 13  # prompt + max_new (truncated, never overshoots)


def test_gumiho_rejects_bad_draft_len():
    with pytest.raises(ValueError):
        gumiho_serial_draft(prompt=[0], draft_next=lambda s: 0,
                            target_next=lambda s: 0, max_new=4, draft_len=0)


@pytest.mark.parametrize("max_new,draft_len", [(1, 4), (2, 8), (3, 5), (7, 3)])
def test_gumiho_clamps_draft_verify_to_the_budget(max_new, draft_len):
    # Regression (review): a production target_next that enforces the requested
    # generation/context limit must never be called on an over-long speculative
    # context. The loop clamps the per-iteration depth to the remaining budget, so
    # neither model ever sees a context longer than prompt + max_new - 1.
    V = 16
    target = _target(V)
    prompt = [2]
    budget_ctx = len(prompt) + max_new - 1

    def guarded(model, key):
        def fn(seq):
            assert len(seq) <= budget_ctx, (
                f"{key}_next called on context len {len(seq)} > budget "
                f"{budget_ctx}")
            return model(seq)
        return fn

    out = gumiho_serial_draft(
        prompt=prompt,
        draft_next=guarded(lambda s: (int(s[-1]) + 1) % V, "draft"),
        target_next=guarded(target, "target"),
        max_new=max_new, draft_len=draft_len)
    assert len(out) == len(prompt) + max_new
    # still AR-equivalent under the clamp.
    assert out == autoregressive_decode(prompt=prompt, target_next=target,
                                        max_new=max_new)
