"""SD1-4 — target_verify is the speculative-decode target-verification I/O
contract: given the verified-position context tokens (current + draft prefix, S =
prefix_len+1 positions) and the composed target model's raw logits at those
positions (S × V), it returns the contract-shaped per-position target log-probs
(S × V, a log_softmax over the vocab) — exactly the shape spec_accept_sample
consumes. A composed model call (pure), not a fused kernel; the op pins the S × V
batching contract.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera.compiler.op_catalog import OP_SPECS


def test_target_verify_registered_and_pure():
    assert "target_verify" in tessera.ops.registry.list()
    assert hasattr(tessera.ops, "target_verify")
    assert OP_SPECS["target_verify"].effect == "pure"


def test_target_verify_is_log_softmax_over_vocab():
    rng = np.random.default_rng(0)
    S, V = 5, 11  # S = prefix_len + 1 verified positions
    tokens = rng.integers(0, V, size=S, dtype=np.int32)
    logits = rng.standard_normal((S, V)).astype(np.float32)
    pr = np.asarray(tessera.ops.target_verify(tokens, logits))
    assert pr.shape == (S, V)
    # each row is a valid PROBABILITY distribution: non-negative and sums to 1.
    assert (pr >= 0).all()
    np.testing.assert_allclose(pr.sum(axis=1), np.ones(S), atol=1e-5)
    # matches a reference softmax.
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(pr, ref, rtol=1e-5, atol=1e-6)


def test_target_verify_output_feeds_spec_accept_sample_directly():
    # Regression (review): target_verify returns PROBABILITIES, so its output is
    # the `target_probs` operand of spec_accept_sample DIRECTLY — no exp() — which
    # reads target_probs as probabilities (accept ratio + residual). Feeding
    # log-probs here would treat valid distributions as negative weights.
    rng = np.random.default_rng(1)
    D, V = 4, 8
    S = D + 1
    tokens = rng.integers(0, V, size=S, dtype=np.int32)
    logits = rng.standard_normal((S, V)).astype(np.float32)
    target_probs = np.asarray(tessera.ops.target_verify(tokens, logits),
                              dtype=np.float32)  # (D+1) × V probabilities
    assert target_probs.shape == (D + 1, V)
    assert (target_probs >= 0).all()  # never negative → no spurious rejection
    draft = rng.integers(0, V, size=D, dtype=np.int32)
    draft_probs = np.full((D, V), 1.0 / V, np.float32)
    accept_u = rng.random(D).astype(np.float32)
    resid_u = rng.random(1).astype(np.float32)
    out = np.asarray(tessera.ops.spec_accept_sample(
        draft, target_probs, draft_probs, accept_u, resid_u))
    assert out.shape == (D + 2,)            # [accepted, t0..t_D]
    assert 0 <= int(out[0]) <= D            # a sane accepted count, not degenerate


def test_target_verify_rejects_mismatched_tokens():
    with pytest.raises(ValueError):
        tessera.ops.target_verify(np.zeros(3, np.int32),
                                  np.zeros((2, 4), np.float32))
