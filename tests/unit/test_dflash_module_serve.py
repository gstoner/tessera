"""P1 — DFlash nn.Module (#6), rotating cache (#9b), tokenizer + scheduler (#9c/#9d)."""
import numpy as np
import pytest

from tessera import dflash as D
from tessera import dflash_reference as R
from tessera import dflash_serve as S
from tessera.data import VocabTokenizer


def _lm_cfg(V=31):
    return R.DecoderLMConfig(vocab_size=V, hidden_size=16, num_layers=2, num_heads=4,
                             head_dim=4, intermediate_size=32, target_layer_ids=(0, 1))


def _draft(rng, lm_cfg, block_size=5):
    cfg = D.DFlashConfig(hidden_size=lm_cfg.hidden_size, num_hidden_layers=2,
                         num_attention_heads=lm_cfg.num_heads, num_key_value_heads=2,
                         head_dim=lm_cfg.head_dim, intermediate_size=lm_cfg.intermediate_size,
                         vocab_size=lm_cfg.vocab_size, block_size=block_size,
                         target_layer_ids=lm_cfg.target_layer_ids)
    Dm, Hq, Hkv, Dh, I, V, nL = (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                                 cfg.head_dim, cfg.intermediate_size, cfg.vocab_size, cfg.num_target_layers)
    s = lambda *sh: rng.standard_normal(sh).astype(np.float32) * 0.1
    layers = [D.DFlashLayerWeights(
        q_proj=s(Dm, Hq * Dh), k_proj=s(Dm, Hkv * Dh), v_proj=s(Dm, Hkv * Dh),
        o_proj=s(Hq * Dh, Dm), q_norm=s(Dh) + 1.0, k_norm=s(Dh) + 1.0,
        input_layernorm=s(Dm) + 1.0, post_attention_layernorm=s(Dm) + 1.0,
        mlp_gate=s(Dm, I), mlp_up=s(Dm, I), mlp_down=s(I, Dm)) for _ in range(2)]
    w = D.DFlashWeights(embed_tokens=s(V, Dm), fc=s(nL * Dm, Dm), hidden_norm=s(Dm) + 1.0,
                        layers=layers, final_norm=s(Dm) + 1.0, lm_head=s(Dm, V))
    return cfg, w


# ── #6 nn.Module ────────────────────────────────────────────────────────────

def test_module_forward_matches_functional():
    rng = np.random.default_rng(0)
    lm_cfg = _lm_cfg()
    cfg, w = _draft(rng, lm_cfg)
    mod = D.DFlashDraft.from_weights(w, cfg)
    block = rng.integers(0, cfg.vocab_size, (1, cfg.block_size))
    th = rng.standard_normal((1, 6, cfg.num_target_layers * cfg.hidden_size)).astype(np.float32)
    rope = D.make_rope(cfg.head_dim)
    ref = D.dflash_draft_forward(block, th, w, cfg, logits_start=1, rope_fn=rope)
    got = mod(block, th, logits_start=1)
    assert np.abs(np.asarray(got) - ref).max() < 1e-5


def test_module_registers_parameters_and_roundtrips_weights():
    rng = np.random.default_rng(1)
    lm_cfg = _lm_cfg()
    cfg, w = _draft(rng, lm_cfg)
    mod = D.DFlashDraft(cfg, w)
    n_params = sum(1 for _ in mod.parameters())
    # top-level (embed, fc, hidden_norm, final_norm, lm_head) + 11 per layer * 2
    assert n_params == 5 + 11 * 2
    w2 = mod.to_weights()
    assert np.allclose(w2.fc, w.fc)
    assert np.allclose(w2.layers[1].mlp_down, w.layers[1].mlp_down)


# ── #9b rotating draft cache ────────────────────────────────────────────────

def test_rotating_cache_caps_length_and_matches_when_unbounded():
    rng = np.random.default_rng(2)
    lm_cfg = _lm_cfg()
    cfg, w = _draft(rng, lm_cfg)
    block = rng.integers(0, cfg.vocab_size, (1, cfg.block_size))
    ths = [rng.standard_normal((1, 4, cfg.num_target_layers * cfg.hidden_size)).astype(np.float32)
           for _ in range(3)]
    # unbounded rotating cache (max_size huge) == plain DraftKVCache
    plain = D.DraftKVCache(cfg.num_hidden_layers)
    rot = D.RotatingDraftKVCache(cfg.num_hidden_layers, max_size=10_000)
    for th in ths:
        lp = D.dflash_draft_forward_cached(block, th, w, cfg, plain, logits_start=1)
        lr = D.dflash_draft_forward_cached(block, th, w, cfg, rot, logits_start=1)
        assert np.abs(lp - lr).max() < 1e-6
    # bounded cache caps the per-layer context length
    capped = D.RotatingDraftKVCache(cfg.num_hidden_layers, max_size=5)
    for th in ths:
        D.dflash_draft_forward_cached(block, th, w, cfg, capped, logits_start=1)
    assert all(k.shape[1] <= 5 for k in capped.keys)


# ── #9c tokenizer text generation + #9d scheduler ───────────────────────────

def _greedy_ar(lm, prompt, max_new):
    tokens = list(prompt)
    for _ in range(max_new):
        lg, _ = lm.forward(np.asarray(tokens, dtype=np.int64)[None, :])
        tokens.append(int(np.argmax(lg[:, -1])))
    return tokens


def test_scheduler_generate_matches_greedy_ar():
    rng = np.random.default_rng(3)
    lm_cfg = _lm_cfg()
    lm = R.random_decoder_lm(lm_cfg, rng)
    cfg, w = _draft(rng, lm_cfg)
    sched = S.DFlashScheduler(w, cfg, lm)
    prompt = [3, 1, 4, 1, 5]
    out = sched.generate(prompt, max_new_tokens=10)
    ar = _greedy_ar(lm, prompt, 10)
    assert out == ar[: len(out)] and len(out) > len(prompt)


def test_generate_text_roundtrips_through_tokenizer():
    rng = np.random.default_rng(4)
    V = 20
    tok = VocabTokenizer({f"t{i}": i for i in range(V)})
    lm_cfg = _lm_cfg(V=V)
    lm = R.random_decoder_lm(lm_cfg, rng)
    cfg, w = _draft(rng, lm_cfg)
    sched = S.DFlashScheduler(w, cfg, lm)
    prompt = "t3 t1 t4"
    text = sched.generate_text(prompt, tok, max_new_tokens=8)
    assert isinstance(text, str) and len(text) > 0
    # decoded continuation must match the greedy-AR ids decoded
    ar = _greedy_ar(lm, tok.encode(prompt), 8)
    expected = tok.decode(ar[len(tok.encode(prompt)):])
    # scheduler greedy output ids are a prefix of AR -> decoded tokens are a prefix.
    got_toks, exp_toks = text.split(), expected.split()
    assert got_toks == exp_toks[: len(got_toks)] and len(got_toks) > 0
    assert all(t in {f"t{i}" for i in range(V)} for t in got_toks)
