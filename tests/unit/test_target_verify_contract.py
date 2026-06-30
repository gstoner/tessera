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
    lp = np.asarray(tessera.ops.target_verify(tokens, logits))
    assert lp.shape == (S, V)
    # each row is a valid log-prob distribution: exp sums to 1.
    np.testing.assert_allclose(np.exp(lp).sum(axis=1), np.ones(S), atol=1e-5)
    # matches a reference log_softmax.
    m = logits.max(axis=1, keepdims=True)
    ref = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(lp, ref, rtol=1e-5, atol=1e-6)


def test_target_verify_output_feeds_spec_accept_sample_shape():
    # The (S × V) contract is exactly what spec_accept_sample's target_probs wants
    # (S = D+1): the verify output for a D-token draft has the bonus row built in.
    rng = np.random.default_rng(1)
    D, V = 4, 8
    S = D + 1
    tokens = rng.integers(0, V, size=S, dtype=np.int32)
    logits = rng.standard_normal((S, V)).astype(np.float32)
    target_logprobs = np.asarray(tessera.ops.target_verify(tokens, logits))
    target_probs = np.exp(target_logprobs).astype(np.float32)  # (D+1) × V
    assert target_probs.shape == (D + 1, V)
    draft = rng.integers(0, V, size=D, dtype=np.int32)
    draft_probs = np.full((D, V), 1.0 / V, np.float32)
    accept_u = rng.random(D).astype(np.float32)
    resid_u = rng.random(1).astype(np.float32)
    out = tessera.ops.spec_accept_sample(draft, target_probs, draft_probs,
                                         accept_u, resid_u)
    assert np.asarray(out).shape == (D + 2,)  # [accepted, t0..t_D]


def test_target_verify_rejects_mismatched_tokens():
    with pytest.raises(ValueError):
        tessera.ops.target_verify(np.zeros(3, np.int32),
                                  np.zeros((2, 4), np.float32))
