"""DiffusionGemma Phase C — entropy-bound sampler.

Covers the work plan's sampler test group: reference parity on small vocab,
large-vocab (262144 × 256) shape/throughput smoke, determinism by RNG key, the
temperature schedule, the re-noise mask, and the stop conditions
(entropy/stability/EOS/all-accepted/max-steps).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.models.sampler import (
    SamplerConfig,
    SamplerResult,
    entropy_bound_sample,
    temperature_schedule,
)


def _peaked(P, V, peak=60.0, seed=0):
    """Logits with one dominant class per position → near-zero entropy."""
    rng = np.random.default_rng(seed)
    lg = rng.standard_normal((P, V)) * 0.01
    lg[np.arange(P), rng.integers(0, V, size=P)] = peak
    return lg


# ── Reference parity ─────────────────────────────────────────────────────────

def test_entropy_and_softcap_logsoftmax_parity():
    cfg = SamplerConfig(vocab_size=8, num_steps=10)
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((5, 8))
    r = entropy_bound_sample(logits, step=0, config=cfg, rng_key=7)
    # independent reference
    capped = 30.0 * np.tanh(logits / 30.0)
    scaled = capped / temperature_schedule(0, 10, 0.8, 0.4)
    lp = scaled - np.log(np.exp(scaled).sum(-1, keepdims=True))
    ent = -(np.exp(lp) * lp).sum(-1)
    np.testing.assert_allclose(r.entropy, ent, atol=1e-9)


def test_softcap_bounds_logits():
    cfg = SamplerConfig(vocab_size=4, final_logit_softcap=30.0, num_steps=4)
    # huge logits must not blow up entropy / log-softmax (softcap bounds them).
    logits = np.array([[1e6, -1e6, 5e5, -5e5]])
    r = entropy_bound_sample(logits, step=0, config=cfg, rng_key=0)
    assert np.isfinite(r.entropy).all()
    assert np.isfinite(r.entropy_summary["mean"])


# ── Temperature schedule ─────────────────────────────────────────────────────

def test_temperature_schedule_anneals_0_8_to_0_4():
    assert temperature_schedule(0, 48, 0.8, 0.4) == pytest.approx(0.8)
    assert temperature_schedule(47, 48, 0.8, 0.4) == pytest.approx(0.4)
    mid = temperature_schedule(24, 48, 0.8, 0.4)
    assert 0.4 < mid < 0.8
    # monotonically decreasing
    temps = [temperature_schedule(s, 48, 0.8, 0.4) for s in range(48)]
    assert all(b <= a for a, b in zip(temps, temps[1:]))


# ── Determinism by RNG key ───────────────────────────────────────────────────

def test_same_key_is_deterministic():
    cfg = SamplerConfig(vocab_size=32, num_steps=10, entropy_threshold=100.0)
    logits = np.random.default_rng(2).standard_normal((16, 32))
    a = entropy_bound_sample(logits, step=0, config=cfg, rng_key=5)
    b = entropy_bound_sample(logits, step=0, config=cfg, rng_key=5)
    np.testing.assert_array_equal(a.tokens, b.tokens)


def test_different_key_changes_samples():
    # threshold=100 → every position accepted, so tokens are real samples that
    # depend on the key (Gumbel-max).
    cfg = SamplerConfig(vocab_size=32, num_steps=10, entropy_threshold=100.0)
    logits = np.random.default_rng(3).standard_normal((16, 32))
    a = entropy_bound_sample(logits, step=0, config=cfg, rng_key=5)
    b = entropy_bound_sample(logits, step=0, config=cfg, rng_key=6)
    assert bool(a.accepted_mask.all())
    assert not np.array_equal(a.tokens, b.tokens)


# ── Re-noise mask ────────────────────────────────────────────────────────────

def test_renoise_mask_is_complement_and_masks_tokens():
    cfg = SamplerConfig(vocab_size=8, num_steps=10, entropy_threshold=1.0, mask_id=0)
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((10, 8))  # high entropy → mostly re-noised
    r = entropy_bound_sample(logits, step=0, config=cfg, rng_key=1)
    np.testing.assert_array_equal(r.renoise_mask, ~r.accepted_mask)
    assert np.all(r.tokens[r.renoise_mask] == cfg.mask_id)


def test_confident_positions_accept():
    cfg = SamplerConfig(vocab_size=64, num_steps=10, entropy_threshold=0.5)
    r = entropy_bound_sample(_peaked(12, 64), step=0, config=cfg, rng_key=0)
    assert bool(r.accepted_mask.all())
    assert r.entropy_summary["mean"] < 0.5


# ── Stop conditions ──────────────────────────────────────────────────────────

def test_stop_all_accepted():
    cfg = SamplerConfig(vocab_size=64, num_steps=48, entropy_threshold=0.5, eos_id=-1)
    r = entropy_bound_sample(_peaked(8, 64), step=0, config=cfg, rng_key=0)
    assert r.stop_reason == "all_accepted"


def test_stop_max_steps():
    cfg = SamplerConfig(vocab_size=32, num_steps=4, entropy_threshold=0.01,
                        stability_entropy=0.001)
    logits = np.random.default_rng(7).standard_normal((10, 32))  # high entropy
    r = entropy_bound_sample(logits, step=3, config=cfg, rng_key=0)
    assert r.stop_reason == "max_steps"


def test_stop_stability():
    # Most positions ultra-confident (entropy≈0) + a couple two-way ties
    # (entropy≈ln2 > threshold). Mean entropy is tiny (< stability) but not all
    # positions are accepted → "stability".
    cfg = SamplerConfig(vocab_size=8, num_steps=48, entropy_threshold=0.05,
                        stability_entropy=0.2, eos_id=-1)
    lg = _peaked(20, 8, peak=80.0, seed=1)
    lg[18] = 0.0; lg[18, 0] = 40.0; lg[18, 1] = 40.0   # two-way tie
    lg[19] = 0.0; lg[19, 2] = 40.0; lg[19, 3] = 40.0   # two-way tie
    r = entropy_bound_sample(lg, step=2, config=cfg, rng_key=0)
    assert not r.accepted_mask.all()
    assert r.entropy_summary["mean"] < cfg.stability_entropy
    assert r.stop_reason == "stability"


def test_stop_eos():
    # A confident position whose dominant class is the EOS id → "eos".
    cfg = SamplerConfig(vocab_size=16, num_steps=48, entropy_threshold=0.5, eos_id=3)
    lg = _peaked(6, 16, peak=80.0, seed=2)
    lg[2] = 0.0; lg[2, 3] = 80.0   # position 2 → eos id 3
    r = entropy_bound_sample(lg, step=1, config=cfg, rng_key=0)
    assert r.stop_reason == "eos"


# ── Large-vocab smoke ────────────────────────────────────────────────────────

def test_large_vocab_smoke():
    cfg = SamplerConfig(vocab_size=262144, num_steps=48)
    logits = np.random.default_rng(0).standard_normal((256, 262144)).astype(np.float32)
    r = entropy_bound_sample(logits, step=5, config=cfg, rng_key=1)
    assert r.tokens.shape == (256,)
    assert r.accepted_mask.shape == (256,)
    assert np.isfinite(r.entropy).all()
    assert (r.tokens >= 0).all() and (r.tokens < 262144).all()
    # API contract: the four documented return fields are present.
    assert set(r.entropy_summary) == {"mean", "max", "min", "frac_accepted"}
    assert r.stop_reason in {"continue", "all_accepted", "stability", "eos", "max_steps"}
