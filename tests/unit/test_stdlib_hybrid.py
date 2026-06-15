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


# ── L4.1 — SSM (Mamba) mixer (Nemotron) ──────────────────────────────────────
def test_ssm_scan_matches_shipped_selective_ssm_reference():
    """_ssm_scan (which returns the carried state) must reproduce the shipped
    tessera.ops.selective_ssm — the L4 op's reference — so the SSM mixer is
    anchored to ground truth, not self-consistent."""
    from tessera import ops
    rng = np.random.default_rng(30)
    B, S, D, N = 2, 14, 6, 4
    x = rng.standard_normal((B, S, D))
    A = -np.exp(rng.standard_normal(D) * 0.1)          # negative, scalar-state
    Bp = rng.standard_normal((B, S, N))
    Cp = rng.standard_normal((B, S, N))
    dt = np.log1p(np.exp(rng.standard_normal((B, S, D))))   # softplus > 0
    y, h = hy._ssm_scan(x, A, Bp, Cp, dt)
    ref = np.asarray(ops.selective_ssm(x, A, Bp, Cp, dt))
    np.testing.assert_allclose(y, ref, rtol=1e-9, atol=1e-9)
    assert h.shape == (B, D, N)


def _nemotron_cfg(num_layers=8):
    return hy.HybridConfig(d_model=16, num_heads=2, head_dim=8, ssm_state=4,
                           linear_mixer=hy.SSM,
                           schedule=hy.nemotron_schedule(num_layers, attn_period=4))


@pytest.mark.parametrize("prefill", [1, 2, 5])
def test_nemotron_ssm_dualcache_decode_equals_recompute(prefill):
    """Nemotron-shaped: Mamba SSM linear layers + sparse attention anchors.
    Streaming carries SSM state h (linear) + KV (anchors) ≡ full recompute."""
    rng = np.random.default_rng(31)
    cfg = _nemotron_cfg(num_layers=8)
    assert cfg.mixer_for(0) == hy.SSM and cfg.mixer_for(3) == hy.FULL
    w = hy.synth_weights(cfg, rng)
    x = rng.standard_normal((2, 11, cfg.d_model))
    np.testing.assert_allclose(hy.hybrid_decode(x, w, cfg, prefill=prefill),
                               hy.hybrid_forward(x, w, cfg), rtol=1e-9, atol=1e-9)


def test_all_ssm_stack_decode_equals_recompute():
    """Pure Mamba stack (no anchors): SSM state carry alone ≡ full forward."""
    rng = np.random.default_rng(32)
    cfg = hy.HybridConfig(d_model=16, num_heads=2, head_dim=8, ssm_state=4,
                          linear_mixer=hy.SSM,
                          schedule=hy.HybridSchedule(num_layers=5, period=999))
    assert cfg.schedule.counts()[hy.FULL] == 0
    w = hy.synth_weights(cfg, rng)
    x = rng.standard_normal((1, 10, 16))
    np.testing.assert_allclose(hy.hybrid_decode(x, w, cfg, prefill=3),
                               hy.hybrid_forward(x, w, cfg), rtol=1e-9, atol=1e-9)


# ── L5 — LIV (LFM2.5) gated short-conv mixer ─────────────────────────────────
def _lfm_cfg(num_layers=8):
    return hy.HybridConfig(d_model=16, num_heads=2, head_dim=8, linear_mixer=hy.LIV,
                           schedule=hy.HybridSchedule(num_layers, period=4, full_offset=1))


@pytest.mark.parametrize("prefill", [1, 2, 5])
def test_lfm_liv_dualcache_decode_equals_recompute(prefill):
    """LFM2.5-shaped: LIV gated-conv layers + attention anchors.  The conv state
    (last k-1 inputs) carried in the dual cache ≡ full recompute."""
    rng = np.random.default_rng(40)
    cfg = _lfm_cfg(8)
    assert cfg.mixer_for(0) == hy.LIV and cfg.mixer_for(3) == hy.FULL
    w = hy.synth_weights(cfg, rng)
    x = rng.standard_normal((2, 11, cfg.d_model))
    np.testing.assert_allclose(hy.hybrid_decode(x, w, cfg, prefill=prefill),
                               hy.hybrid_forward(x, w, cfg), rtol=1e-9, atol=1e-9)


def test_liv_conv_is_causal():
    """A future input must not change a past LIV output (causality of the conv)."""
    rng = np.random.default_rng(41)
    cfg = hy.HybridConfig(d_model=16, num_heads=2, head_dim=8, linear_mixer=hy.LIV,
                          schedule=hy.HybridSchedule(3, period=999))   # all-LIV
    w = hy.synth_weights(cfg, rng)
    x = rng.standard_normal((1, 8, 16))
    y = hy.hybrid_forward(x, w, cfg)
    x2 = x.copy(); x2[:, 5:] = rng.standard_normal((1, 3, 16))   # perturb tail
    y2 = hy.hybrid_forward(x2, w, cfg)
    np.testing.assert_allclose(y[:, :5], y2[:, :5], rtol=1e-9, atol=1e-9)


# ── MoE FFN composition (full model block) ───────────────────────────────────
@pytest.mark.parametrize("prefill", [1, 4])
@pytest.mark.parametrize("shared", [True, False])
def test_moe_ffn_dualcache_decode_equals_recompute(prefill, shared):
    """Routed-MoE FFN (exact, no-drop routing) is per-token → decode ≡ recompute,
    across delta mixers + attention anchors."""
    rng = np.random.default_rng(50)
    cfg = hy.HybridConfig(d_model=16, num_heads=2, head_dim=8,
                          schedule=hy.HybridSchedule(6, period=3, full_offset=1),
                          ffn="moe", num_experts=8, top_k=2, shared_expert=shared)
    w = hy.synth_weights(cfg, rng)
    x = rng.standard_normal((2, 9, cfg.d_model))
    np.testing.assert_allclose(hy.hybrid_decode(x, w, cfg, prefill=prefill),
                               hy.hybrid_forward(x, w, cfg), rtol=1e-9, atol=1e-9)


def test_moe_routing_is_load_bearing():
    """top_k changes the output — proves experts are actually routed, not summed."""
    rng = np.random.default_rng(51)
    base = dict(d_model=16, num_heads=2, head_dim=8,
                schedule=hy.HybridSchedule(2, period=999), ffn="moe",
                num_experts=8, shared_expert=False)
    cfg1 = hy.HybridConfig(top_k=1, **base)
    cfg4 = hy.HybridConfig(top_k=4, **base)
    w = hy.synth_weights(cfg1, np.random.default_rng(52))   # same weights both
    x = rng.standard_normal((1, 6, 16))
    y1 = hy.hybrid_forward(x, w, cfg1)
    y4 = hy.hybrid_forward(x, w, cfg4)
    assert not np.allclose(y1, y4, rtol=1e-3, atol=1e-3)


def test_nemotron_full_block_ssm_plus_moe():
    """Nemotron = SSM mixer + attention anchors + MoE FFN, all at once."""
    rng = np.random.default_rng(53)
    cfg = hy.HybridConfig(d_model=16, num_heads=2, head_dim=8, ssm_state=4,
                          linear_mixer=hy.SSM, schedule=hy.nemotron_schedule(8, 4),
                          ffn="moe", num_experts=8, top_k=2, shared_expert=True)
    w = hy.synth_weights(cfg, rng)
    x = rng.standard_normal((2, 10, cfg.d_model))
    np.testing.assert_allclose(hy.hybrid_decode(x, w, cfg, prefill=2),
                               hy.hybrid_forward(x, w, cfg), rtol=1e-9, atol=1e-9)


# ── MTP — multi-token-prediction draft head (graph-level) ────────────────────
def _lm(num_layers=6, vocab=20, seed=60):
    cfg = hy.HybridConfig(d_model=16, num_heads=2, head_dim=8,
                          schedule=hy.qwen3_6_schedule(num_layers))
    return hy.synth_lm_weights(cfg, vocab, np.random.default_rng(seed))


def test_mtp_head_shapes_and_determinism():
    lm = _lm()
    rng = np.random.default_rng(61)
    ids = rng.integers(0, 20, size=(2, 5))
    h = hy._backbone_hidden(ids, lm)
    nxt = rng.integers(0, 20, size=2)
    d1 = hy.mtp_draft_logits(h[:, -1], nxt, lm)
    d2 = hy.mtp_draft_logits(h[:, -1], nxt, lm)
    assert d1.shape == (2, 20)
    np.testing.assert_array_equal(d1, d2)      # deterministic graph object


def test_mtp_speculative_decode_is_lossless():
    """Headline MTP contract: greedy self-speculation == greedy autoregressive."""
    lm = _lm(seed=62)
    prompt = np.array([[3, 7, 1]])
    ar = hy.greedy_generate(lm, prompt, n=8)
    spec, accepted = hy.mtp_speculative_generate(lm, prompt, n=8)
    np.testing.assert_array_equal(spec, ar)
    assert accepted >= 0 and spec.shape == ar.shape


def test_mtp_accept_path_exercised_on_predictable_model():
    """Construct a degenerate (constant-prediction) LM so the MTP draft always
    matches the verified token — exercises the accept branch + proves the
    speculation is still lossless."""
    lm = _lm(seed=63)
    lm.embed[:] = lm.embed[0]                   # identical rows → argmax always 0
    prompt = np.array([[5, 2, 9]])
    ar = hy.greedy_generate(lm, prompt, n=6)
    spec, accepted = hy.mtp_speculative_generate(lm, prompt, n=6)
    np.testing.assert_array_equal(spec, ar)
    assert (ar[:, 3:] == 0).all()               # constant-prediction model
    assert accepted >= 1                         # MTP hits land
