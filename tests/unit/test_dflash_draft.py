"""P1 — DFlash draft model + orchestration.

Validates tessera.dflash (target-feature projection, decoder layer, draft
forward, linear verify, step cycle) against an independent numpy port of the
z-lab/dflash model_mlx semantics.
"""
import numpy as np
import pytest

from tessera import dflash as D


# --- independent numpy reference -------------------------------------------

def _rms(t, w, eps=1e-6):
    y = t / np.sqrt((t * t).mean(-1, keepdims=True) + eps)
    return y * w if w is not None else y


def _ref_attn(x, x_ctx, lw, Hq, Hkv, Dh, rope, cache_offset, sliding):
    B, L, _ = x.shape
    S = x_ctx.shape[1]

    def heads(t, W, h):
        y = t @ W
        return y.reshape(B, y.shape[1], h, Dh).transpose(0, 2, 1, 3)

    q = _rms(heads(x, lw.q_proj, Hq), lw.q_norm)
    ck = _rms(heads(x_ctx, lw.k_proj, Hkv), lw.k_norm)
    cv = heads(x_ctx, lw.v_proj, Hkv)
    pk = _rms(heads(x, lw.k_proj, Hkv), lw.k_norm)
    pv = heads(x, lw.v_proj, Hkv)
    if rope is not None:
        q = rope(q, cache_offset + S); ck = rope(ck, cache_offset); pk = rope(pk, cache_offset + S)
    ctx_len = ck.shape[2]
    K = np.concatenate([ck, pk], 2); V = np.concatenate([cv, pv], 2); Sk = K.shape[2]
    if Hkv != Hq:
        K = np.repeat(K, Hq // Hkv, 1); V = np.repeat(V, Hq // Hkv, 1)
    s = np.einsum("bhqd,bhkd->bhqk", q, K) * (Dh ** -0.5)
    if sliding is not None:
        qpos = ctx_len + np.arange(L)[:, None]; kpos = np.arange(Sk)[None, :]
        s = np.where((kpos <= qpos) & (kpos > qpos - sliding), s, -1e30)
    s = s - s.max(-1, keepdims=True); a = np.exp(s); a /= a.sum(-1, keepdims=True)
    o = np.einsum("bhqk,bhkd->bhqd", a, V).transpose(0, 2, 1, 3).reshape(B, L, Hq * Dh)
    return o @ lw.o_proj


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _ref_forward(block, target_hidden, w, cfg, rope, logits_start=0):
    h = w.embed_tokens[block] * cfg.embed_scale
    x_ctx = _rms(target_hidden @ w.fc, w.hidden_norm)
    for i, lw in enumerate(w.layers):
        sliding = cfg.sliding_window if cfg.layer_types[i] == "sliding_attention" else None
        xn = _rms(h, lw.input_layernorm)
        h = h + _ref_attn(xn, x_ctx, lw, cfg.num_attention_heads,
                          cfg.num_key_value_heads, cfg.head_dim, rope, 0, sliding)
        hn = _rms(h, lw.post_attention_layernorm)
        mlp = (_silu(hn @ lw.mlp_gate) * (hn @ lw.mlp_up)) @ lw.mlp_down
        h = h + mlp
    if logits_start:
        h = h[:, logits_start:]
    h = _rms(h, w.final_norm)
    lm = w.embed_tokens.T if w.lm_head is None else w.lm_head
    return h @ lm


# --- weight factory --------------------------------------------------------

def _make_weights(rng, cfg, tied=False):
    D_, Hq, Hkv, Dh = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    I, V = cfg.intermediate_size, cfg.vocab_size
    nL = cfg.num_target_layers
    s = lambda *sh: (rng.standard_normal(sh).astype(np.float32) * 0.1)
    layers = [D.DFlashLayerWeights(
        q_proj=s(D_, Hq * Dh), k_proj=s(D_, Hkv * Dh), v_proj=s(D_, Hkv * Dh),
        o_proj=s(Hq * Dh, D_), q_norm=s(Dh) + 1.0, k_norm=s(Dh) + 1.0,
        input_layernorm=s(D_) + 1.0, post_attention_layernorm=s(D_) + 1.0,
        mlp_gate=s(D_, I), mlp_up=s(D_, I), mlp_down=s(I, D_),
    ) for _ in range(cfg.num_hidden_layers)]
    return D.DFlashWeights(
        embed_tokens=s(V, D_), fc=s(nL * D_, D_), hidden_norm=s(D_) + 1.0,
        layers=layers, final_norm=s(D_) + 1.0,
        lm_head=None if tied else s(D_, V))


def _cfg(**kw):
    base = dict(hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=4, intermediate_size=32,
                vocab_size=37, block_size=5, target_layer_ids=(0, 1, 2))
    base.update(kw)
    return D.DFlashConfig(**base)


# --- tests -----------------------------------------------------------------

def test_target_feature_projection():
    rng = np.random.default_rng(0)
    B, S, D_, nL = 2, 6, 16, 3
    th = rng.standard_normal((B, S, nL * D_)).astype(np.float32)
    fc = rng.standard_normal((nL * D_, D_)).astype(np.float32) * 0.1
    hn = rng.standard_normal(D_).astype(np.float32) * 0.1 + 1.0
    got = D.target_feature_projection(th, fc, hn)
    ref = _rms(th @ fc, hn)
    assert got.shape == (B, S, D_)
    assert np.abs(np.asarray(got) - ref).max() < 1e-4


@pytest.mark.parametrize("tied", [False, True])
def test_draft_forward_matches_numpy(tied):
    rng = np.random.default_rng(1)
    cfg = _cfg()
    w = _make_weights(rng, cfg, tied=tied)
    rope = D.make_rope(cfg.head_dim, cfg.rope_theta)
    B, S = 2, 6
    block = rng.integers(0, cfg.vocab_size, (B, cfg.block_size))
    th = rng.standard_normal((B, S, cfg.num_target_layers * cfg.hidden_size)).astype(np.float32)
    got = D.dflash_draft_forward(block, th, w, cfg, logits_start=1, rope_fn=rope)
    ref = _ref_forward(block, th, w, cfg, rope, logits_start=1)
    assert got.shape == (B, cfg.block_size - 1, cfg.vocab_size)
    assert np.abs(got - ref).max() < 1e-3


def test_draft_forward_sliding_layer():
    rng = np.random.default_rng(2)
    cfg = _cfg(layer_types=("full_attention", "sliding_attention"), sliding_window=3)
    w = _make_weights(rng, cfg)
    rope = D.make_rope(cfg.head_dim)
    block = rng.integers(0, cfg.vocab_size, (1, cfg.block_size))
    th = rng.standard_normal((1, 5, cfg.num_target_layers * cfg.hidden_size)).astype(np.float32)
    got = D.dflash_draft_forward(block, th, w, cfg, logits_start=0, rope_fn=rope)
    ref = _ref_forward(block, th, w, cfg, rope, logits_start=0)
    assert np.abs(got - ref).max() < 1e-3


def test_linear_verify():
    # full match (greedy, draft==target) → all accepted + 1 bonus
    v = D.dflash_linear_verify([5, 6, 7], [5, 6, 7, 8])
    assert v.accepted == 3 and v.new_tokens == [5, 6, 7, 8]
    # divergence at index 1 → accept prefix [5] + corrected token 99
    v = D.dflash_linear_verify([5, 6, 7], [5, 99, 7, 8])
    assert v.accepted == 1 and v.new_tokens == [5, 99]
    # immediate divergence → 0 accepted + 1 corrected
    v = D.dflash_linear_verify([1, 2], [9, 2, 3])
    assert v.accepted == 0 and v.new_tokens == [9]


def test_step_perfect_draft_accepts_all():
    """If the target's greedy sample equals the draft on every position, the
    whole block is accepted and the fresh hidden is trimmed to accepted+1."""
    rng = np.random.default_rng(3)
    cfg = _cfg()
    w = _make_weights(rng, cfg)
    rope = D.make_rope(cfg.head_dim)
    nL, D_ = cfg.num_target_layers, cfg.hidden_size
    prev = 4
    target_hidden = rng.standard_normal((1, 6, nL * D_)).astype(np.float32)

    # target_fn that always agrees with the draft: greedily decode the draft's
    # own tokens for the block positions, and a fixed bonus at the end.
    def target_fn(verify_input):
        T = verify_input.shape[1]
        V = cfg.vocab_size
        logits = np.full((1, T, V), -10.0, np.float32)
        # position i predicts verify_input[i+1] (the next draft token); last
        # position predicts a bonus token (here: 0).
        for i in range(T - 1):
            logits[0, i, int(verify_input[0, i + 1])] = 10.0
        logits[0, T - 1, 0] = 10.0
        hidden = rng.standard_normal((1, T, nL * D_)).astype(np.float32)
        return logits, hidden

    result, fresh = D.dflash_step(prev, target_hidden, w, cfg, target_fn, rope_fn=rope)
    assert result.accepted == cfg.block_size - 1            # every draft token accepted
    assert len(result.new_tokens) == cfg.block_size         # accepted + 1 bonus
    assert fresh.shape == (1, result.accepted + 1, nL * D_)


def test_step_early_divergence_trims():
    """A target that disagrees at position 0 accepts nothing and emits one
    corrected token; fresh hidden trims to length 1."""
    rng = np.random.default_rng(4)
    cfg = _cfg()
    w = _make_weights(rng, cfg)
    nL, D_ = cfg.num_target_layers, cfg.hidden_size
    target_hidden = rng.standard_normal((1, 6, nL * D_)).astype(np.float32)

    def target_fn(verify_input):
        T = verify_input.shape[1]
        logits = np.zeros((1, T, cfg.vocab_size), np.float32)
        logits[0, :, 13] = 10.0          # always predicts token 13 → diverges at pos 0
        hidden = rng.standard_normal((1, T, nL * D_)).astype(np.float32)
        return logits, hidden

    result, fresh = D.dflash_step(7, target_hidden, w, cfg, target_fn)
    assert result.accepted == 0
    assert result.new_tokens == [13]
    assert fresh.shape == (1, 1, nL * D_)
