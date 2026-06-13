"""P1 — DFlash stateful target (#3) + real reference target model (#4).

Validates the KV-cached decoder with rollback against its own stateless forward,
and the efficient cached+stateful DFlash generation loop against greedy AR.
"""
import numpy as np
import pytest

from tessera import dflash as D
from tessera import dflash_reference as R


def _lm_cfg(**kw):
    base = dict(vocab_size=31, hidden_size=16, num_layers=2, num_heads=4,
                head_dim=4, intermediate_size=32, target_layer_ids=(0, 1))
    base.update(kw)
    return R.DecoderLMConfig(**base)


def _draft_weights(rng, lm_cfg, block_size=5):
    cfg = D.DFlashConfig(
        hidden_size=lm_cfg.hidden_size, num_hidden_layers=2,
        num_attention_heads=lm_cfg.num_heads, num_key_value_heads=2,
        head_dim=lm_cfg.head_dim, intermediate_size=lm_cfg.intermediate_size,
        vocab_size=lm_cfg.vocab_size, block_size=block_size,
        target_layer_ids=lm_cfg.target_layer_ids)
    Dm, Hq, Hkv, Dh = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    I, V, nL = cfg.intermediate_size, cfg.vocab_size, cfg.num_target_layers
    s = lambda *sh: rng.standard_normal(sh).astype(np.float32) * 0.1
    layers = [D.DFlashLayerWeights(
        q_proj=s(Dm, Hq * Dh), k_proj=s(Dm, Hkv * Dh), v_proj=s(Dm, Hkv * Dh),
        o_proj=s(Hq * Dh, Dm), q_norm=s(Dh) + 1.0, k_norm=s(Dh) + 1.0,
        input_layernorm=s(Dm) + 1.0, post_attention_layernorm=s(Dm) + 1.0,
        mlp_gate=s(Dm, I), mlp_up=s(Dm, I), mlp_down=s(I, Dm),
    ) for _ in range(cfg.num_hidden_layers)]
    w = D.DFlashWeights(embed_tokens=s(V, Dm), fc=s(nL * Dm, Dm),
                        hidden_norm=s(Dm) + 1.0, layers=layers,
                        final_norm=s(Dm) + 1.0, lm_head=s(Dm, V))
    return cfg, w


# ── #3: stateful KV cache == stateless forward, incl. rollback ──────────────

def test_stateful_step_matches_stateless_forward():
    rng = np.random.default_rng(0)
    lm = R.random_decoder_lm(_lm_cfg(), rng)
    toks = rng.integers(0, 31, (1, 9))
    full_logits, full_hidden = lm.forward(toks)
    # feed in three incremental chunks
    lm.reset()
    pieces = [toks[:, :4], toks[:, 4:7], toks[:, 7:]]
    log_parts, hid_parts = [], []
    for p in pieces:
        lg, hd = lm.step(p)
        log_parts.append(lg); hid_parts.append(hd)
    inc_logits = np.concatenate(log_parts, axis=1)
    inc_hidden = np.concatenate(hid_parts, axis=1)
    assert np.abs(inc_logits - full_logits).max() < 1e-3
    assert np.abs(inc_hidden - full_hidden).max() < 1e-3
    assert lm.cache_len == 9


def test_rollback_restores_cache_state():
    rng = np.random.default_rng(1)
    lm = R.random_decoder_lm(_lm_cfg(), rng)
    toks = rng.integers(0, 31, (1, 6))
    extra = rng.integers(0, 31, (1, 3))
    lm.reset(); lm.step(toks)
    # speculate 3 tokens then roll them back
    lm.step(extra)
    assert lm.cache_len == 9
    lm.rollback(3)
    assert lm.cache_len == 6
    # a fresh token now must match the no-speculation continuation
    nxt = rng.integers(0, 31, (1, 1))
    rolled_logits, _ = lm.step(nxt)
    lm.reset()
    lm.step(toks)
    clean_logits, _ = lm.step(nxt)
    assert np.abs(rolled_logits - clean_logits).max() < 1e-4


# ── #4 + #3: cached+stateful generate == greedy AR ──────────────────────────

def _greedy_ar(lm, prompt, max_new, eos_id=None):
    tokens = list(prompt)
    for _ in range(max_new):
        lg, _ = lm.forward(np.asarray(tokens, dtype=np.int64)[None, :])
        nxt = int(np.argmax(lg[:, -1]))
        tokens.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
    return tokens


def test_cached_stateful_generate_matches_greedy_ar():
    rng = np.random.default_rng(2)
    lm_cfg = _lm_cfg()
    lm = R.random_decoder_lm(lm_cfg, rng)
    cfg, w = _draft_weights(rng, lm_cfg)
    rope = D.make_rope(cfg.head_dim, cfg.rope_theta)
    prompt = [3, 1, 4, 1, 5]
    max_new = 12
    ar = _greedy_ar(lm, prompt, max_new)
    spec = D.dflash_generate_cached(prompt, w, cfg, lm, max_new_tokens=max_new, rope_fn=rope)
    assert len(spec) > len(prompt)
    assert spec == ar[: len(spec)]


def test_cached_generate_block_size_independent():
    rng = np.random.default_rng(3)
    lm_cfg = _lm_cfg()
    lm = R.random_decoder_lm(lm_cfg, rng)
    cfg, w = _draft_weights(rng, lm_cfg, block_size=8)
    prompt = [2, 7, 1]
    out2 = D.dflash_generate_cached(prompt, w, cfg, lm, max_new_tokens=10, block_size=2)
    out6 = D.dflash_generate_cached(prompt, w, cfg, lm, max_new_tokens=10, block_size=6)
    nmin = min(len(out2), len(out6))
    assert out2[:nmin] == out6[:nmin] and nmin > len(prompt)


def test_cached_sampling_generate_runs_in_vocab_and_reproducible():
    rng_lm = np.random.default_rng(4)
    lm_cfg = _lm_cfg()
    lm = R.random_decoder_lm(lm_cfg, rng_lm)
    cfg, w = _draft_weights(rng_lm, lm_cfg)
    prompt = [1, 2, 3]
    a = D.dflash_generate_cached(prompt, w, cfg, lm, max_new_tokens=10,
                                 temperature=0.9, top_k=8, rng=np.random.default_rng(11))
    b = D.dflash_generate_cached(prompt, w, cfg, lm, max_new_tokens=10,
                                 temperature=0.9, top_k=8, rng=np.random.default_rng(11))
    assert a == b                                   # same seed -> same output
    assert all(0 <= t < cfg.vocab_size for t in a)
    assert len(a) > len(prompt)
