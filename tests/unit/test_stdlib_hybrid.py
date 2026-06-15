"""Track L (L3) — hybrid linear/attention schedule + dual-cache contract.

Schedule: `layer_types` is a first-class object (Qwen3.6 = `[lin,lin,lin,full]·N`).
Dual cache: linear layers carry a constant-size recurrent state Ŝ; full-attention
layers carry a growing KV cache.  The headline oracle is **streaming dual-cache
decode ≡ full recompute** — it only holds if both caches are threaded correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.stdlib import hybrid as hy


# ── schedule (first-class layer_types) ───────────────────────────────────────
def test_qwen36_schedule_pattern():
    s = hy.qwen3_6_schedule(num_layers=40)
    types = s.layer_types()
    assert len(types) == 40
    # [linear, linear, linear, full] repeated.
    assert types[:4] == [hy.LINEAR, hy.LINEAR, hy.LINEAR, hy.FULL]
    assert s.counts() == {hy.LINEAR: 30, hy.FULL: 10}
    assert s.full_indices() == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]


def test_nemotron_schedule_is_mostly_linear():
    s = hy.nemotron_schedule(num_layers=32, attn_period=8)
    c = s.counts()
    assert c[hy.LINEAR] == 28 and c[hy.FULL] == 4  # sparse anchors


def test_schedule_validates():
    with pytest.raises(ValueError):
        hy.HybridSchedule(num_layers=0)
    with pytest.raises(ValueError):
        hy.HybridSchedule(num_layers=4, period=0)


def test_full_config_qwen36_dims_no_execution():
    """Full-config artifact check: the schedule lowers at production depth."""
    s = hy.qwen3_6_schedule(num_layers=40)
    assert s.counts() == {hy.LINEAR: 30, hy.FULL: 10}
    # 75% of token-mixers are linear (Gated DeltaNet), every 4th is full attention.
    assert len(s.linear_indices()) == 30


# ── the dual-cache oracle ────────────────────────────────────────────────────
def _cfg(num_layers=8, H=2, Dh=8, Dm=16):
    return hy.HybridConfig(d_model=Dm, num_heads=H, head_dim=Dh,
                           schedule=hy.HybridSchedule(num_layers=num_layers,
                                                      period=4, full_offset=1))


@pytest.mark.parametrize("prefill", [1, 3, 7])
def test_streaming_dualcache_decode_equals_full_recompute(prefill):
    """Headline L3 oracle: token-by-token decode carrying Ŝ (linear layers) +
    KV (full layers) reproduces the full parallel forward."""
    rng = np.random.default_rng(0)
    cfg = _cfg(num_layers=8)
    w = hy.synth_weights(cfg, rng)
    x = rng.standard_normal((2, 12, cfg.d_model))
    full = hy.hybrid_forward(x, w, cfg)
    stream = hy.hybrid_decode(x, w, cfg, prefill=prefill)
    np.testing.assert_allclose(stream, full, rtol=1e-9, atol=1e-9)


def test_decode_equals_forward_pure_linear_stack():
    """All-linear stack (period huge → no full layers): pure recurrent-state
    carry ≡ full forward."""
    rng = np.random.default_rng(1)
    cfg = hy.HybridConfig(d_model=16, num_heads=2, head_dim=8,
                          schedule=hy.HybridSchedule(num_layers=6, period=999))
    assert cfg.schedule.counts()[hy.FULL] == 0
    w = hy.synth_weights(cfg, rng)
    x = rng.standard_normal((1, 10, 16))
    np.testing.assert_allclose(hy.hybrid_decode(x, w, cfg, prefill=2),
                               hy.hybrid_forward(x, w, cfg), rtol=1e-9, atol=1e-9)


def test_decode_equals_forward_with_attention_anchors():
    """Every-other-layer full attention — both caches exercised heavily."""
    rng = np.random.default_rng(2)
    cfg = hy.HybridConfig(d_model=24, num_heads=3, head_dim=8,
                          schedule=hy.HybridSchedule(num_layers=6, period=2, full_offset=1))
    assert cfg.schedule.counts() == {hy.LINEAR: 3, hy.FULL: 3}
    w = hy.synth_weights(cfg, rng)
    x = rng.standard_normal((2, 9, 24))
    np.testing.assert_allclose(hy.hybrid_decode(x, w, cfg, prefill=1),
                               hy.hybrid_forward(x, w, cfg), rtol=1e-9, atol=1e-9)
