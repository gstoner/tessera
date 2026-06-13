"""P1 — DFlash cached drafting (#1) + sampling / rejection acceptance (#2)."""
import numpy as np
import pytest

from tessera import dflash as D


def _cfg(**kw):
    base = dict(hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=4, intermediate_size=32,
                vocab_size=37, block_size=5, target_layer_ids=(0, 1, 2))
    base.update(kw)
    return D.DFlashConfig(**base)


def _weights(rng, cfg):
    Dm, Hq, Hkv, Dh = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    I, V, nL = cfg.intermediate_size, cfg.vocab_size, cfg.num_target_layers
    s = lambda *sh: rng.standard_normal(sh).astype(np.float32) * 0.1
    layers = [D.DFlashLayerWeights(
        q_proj=s(Dm, Hq * Dh), k_proj=s(Dm, Hkv * Dh), v_proj=s(Dm, Hkv * Dh),
        o_proj=s(Hq * Dh, Dm), q_norm=s(Dh) + 1.0, k_norm=s(Dh) + 1.0,
        input_layernorm=s(Dm) + 1.0, post_attention_layernorm=s(Dm) + 1.0,
        mlp_gate=s(Dm, I), mlp_up=s(Dm, I), mlp_down=s(I, Dm),
    ) for _ in range(cfg.num_hidden_layers)]
    return D.DFlashWeights(embed_tokens=s(V, Dm), fc=s(nL * Dm, Dm),
                           hidden_norm=s(Dm) + 1.0, layers=layers,
                           final_norm=s(Dm) + 1.0, lm_head=s(Dm, V))


# ── #1 cached drafting: cached(accumulated) == non-cached(full context) ─────

def test_cached_draft_forward_matches_full_context():
    rng = np.random.default_rng(0)
    cfg = _cfg()
    w = _weights(rng, cfg)
    rope = D.make_rope(cfg.head_dim, cfg.rope_theta)
    nL, Dm = cfg.num_target_layers, cfg.hidden_size
    block = rng.integers(0, cfg.vocab_size, (1, cfg.block_size))
    th0 = rng.standard_normal((1, 4, nL * Dm)).astype(np.float32)   # step-1 context
    th1 = rng.standard_normal((1, 3, nL * Dm)).astype(np.float32)   # step-2 context

    cache = D.DraftKVCache(cfg.num_hidden_layers)
    D.dflash_draft_forward_cached(block, th0, w, cfg, cache, logits_start=1, rope_fn=rope)
    logits_cached = D.dflash_draft_forward_cached(block, th1, w, cfg, cache, logits_start=1, rope_fn=rope)

    full_ctx = np.concatenate([th0, th1], axis=1)
    logits_full = D.dflash_draft_forward(block, full_ctx, w, cfg, logits_start=1, rope_fn=rope)
    assert logits_cached.shape == logits_full.shape
    assert np.abs(logits_cached - logits_full).max() < 1e-3


def test_draft_cache_accumulates_and_advances():
    rng = np.random.default_rng(1)
    cfg = _cfg()
    w = _weights(rng, cfg)
    cache = D.DraftKVCache(cfg.num_hidden_layers)
    block = rng.integers(0, cfg.vocab_size, (1, cfg.block_size))
    for s_len in (4, 3, 5):
        th = rng.standard_normal((1, s_len, cfg.num_target_layers * cfg.hidden_size)).astype(np.float32)
        D.dflash_draft_forward_cached(block, th, w, cfg, cache, logits_start=1)
    assert cache.offset == 4 + 3 + 5
    # each layer cached (B, total_ctx, Hkv, Dh)
    for k in cache.keys:
        assert k.shape == (1, 12, cfg.num_key_value_heads, cfg.head_dim)


# ── #2 samplers ─────────────────────────────────────────────────────────────

def test_sampler_greedy_is_argmax():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((3, 8))
    s = D.make_sampler(temperature=0.0)
    assert np.array_equal(s(logits), np.argmax(logits, axis=-1))


def test_sampler_temperature_is_reproducible_and_in_range():
    logits = np.random.default_rng(3).standard_normal((4, 10))
    s1 = D.make_sampler(temperature=0.8, rng=np.random.default_rng(7))
    s2 = D.make_sampler(temperature=0.8, rng=np.random.default_rng(7))
    a, b = s1(logits), s2(logits)
    assert np.array_equal(a, b)              # same seed -> same draws
    assert a.shape == (4,) and a.min() >= 0 and a.max() < 10


def test_sampler_top_k_restricts_support():
    # A logit vector with one dominant + a long tail; top_k=1 must always pick argmax.
    logits = np.array([[5.0, 0.1, 0.2, 0.0, -1.0]])
    s = D.make_sampler(temperature=1.0, top_k=1, rng=np.random.default_rng(0))
    for _ in range(20):
        assert int(s(logits)[0]) == 0


# ── #2 rejection acceptance (distribution-preserving) ───────────────────────

def test_speculative_verify_accepts_when_draft_equals_target():
    rng = np.random.default_rng(4)
    V = 6
    p = D._softmax_lastaxis(rng.standard_normal((1, V)))
    target = np.concatenate([p, D._softmax_lastaxis(rng.standard_normal((1, V)))], 0)  # (2,V)
    tok = int(np.argmax(p[0]))
    res = D.dflash_speculative_verify([tok], p, target, rng)
    # draft prob == target prob on row 0 -> min(1, pt/pd)=1 -> accepted + bonus
    assert res.accepted == 1 and len(res.new_tokens) == 2


def test_speculative_verify_marginal_matches_target():
    """The speculative-sampling theorem: if the draft token is drawn from the
    draft distribution, the first emitted token's marginal equals the target's."""
    rng = np.random.default_rng(5)
    V = 5
    draft_probs = D._softmax_lastaxis(rng.standard_normal((1, V)) * 1.5)   # (1,V)
    target_row = D._softmax_lastaxis(rng.standard_normal((1, V)) * 1.5)
    bonus_row = D._softmax_lastaxis(rng.standard_normal((1, V)))
    target_probs = np.concatenate([target_row, bonus_row], 0)             # (2,V)

    N = 40000
    counts = np.zeros(V)
    for _ in range(N):
        dt = int(rng.choice(V, p=draft_probs[0]))
        res = D.dflash_speculative_verify([dt], draft_probs, target_probs, rng)
        counts[res.new_tokens[0]] += 1
    emp = counts / N
    assert np.abs(emp - target_probs[0]).max() < 0.02
