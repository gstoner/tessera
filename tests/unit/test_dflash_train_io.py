"""P1 — DFlash training loss (#9a) + checkpoint I/O (#7)."""
import numpy as np
import pytest

from tessera import dflash as D
from tessera import dflash_io as IO


# ── #9a training loss ───────────────────────────────────────────────────────

def test_position_weights_decay_and_normalize():
    w = D.dflash_position_weights(8, gamma=4.0)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.all(np.diff(w) < 0)               # strictly decreasing (early emphasis)
    assert w[0] > w[-1]


def test_block_loss_grad_matches_finite_diff():
    rng = np.random.default_rng(0)
    B, L, V = 2, 5, 7
    logits = rng.standard_normal((B, L, V))
    targets = rng.integers(0, V, (B, L))
    grad = D.dflash_block_loss_grad(logits, targets, gamma=3.0)
    eps = 1e-6
    fd = np.zeros_like(logits)
    for idx in np.ndindex(logits.shape):
        lp = logits.copy(); lp[idx] += eps
        lm = logits.copy(); lm[idx] -= eps
        fd[idx] = (D.dflash_block_loss(lp, targets, gamma=3.0)
                   - D.dflash_block_loss(lm, targets, gamma=3.0)) / (2 * eps)
    assert np.abs(grad - fd).max() < 1e-7


def test_block_loss_decreases_with_a_grad_step():
    rng = np.random.default_rng(1)
    B, L, V = 3, 4, 9
    logits = rng.standard_normal((B, L, V))
    targets = rng.integers(0, V, (B, L))
    loss0 = D.dflash_block_loss(logits, targets)
    g = D.dflash_block_loss_grad(logits, targets)
    loss1 = D.dflash_block_loss(logits - 0.5 * g * B, targets)   # undo /B scaling for a real step
    assert loss1 < loss0


def test_block_loss_reductions():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((4, 3, 6))
    targets = rng.integers(0, 6, (4, 3))
    none = D.dflash_block_loss(logits, targets, reduction="none")
    assert none.shape == (4,)
    assert abs(D.dflash_block_loss(logits, targets, reduction="mean") - none.mean()) < 1e-9
    assert abs(D.dflash_block_loss(logits, targets, reduction="sum") - none.sum()) < 1e-9


# ── #7 checkpoint I/O ───────────────────────────────────────────────────────

def _cfg():
    return D.DFlashConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4, intermediate_size=32,
                          vocab_size=31, block_size=5, target_layer_ids=(0, 1, 2))


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


def test_safetensors_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    d = {"a": rng.standard_normal((3, 4)).astype(np.float32),
         "b": rng.integers(0, 10, (5,)).astype(np.int64),
         "c": rng.standard_normal((2, 2, 2)).astype(np.float16)}
    p = tmp_path / "t.safetensors"
    IO.save_safetensors(p, d)
    loaded = IO.load_safetensors(p)
    assert set(loaded) == set(d)
    for k in d:
        assert loaded[k].dtype == d[k].dtype
        assert np.array_equal(loaded[k], d[k])


def test_weights_state_dict_roundtrip_and_transpose(tmp_path):
    rng = np.random.default_rng(4)
    cfg = _cfg()
    w = _weights(rng, cfg)
    sd = IO.dflash_weights_to_state_dict(w)
    # HF nn.Linear weight is (out, in): q_proj stored as (Hq*Dh, D) = transpose of mine (D, Hq*Dh)
    Dm, Hq, Dh = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
    assert sd["model.layers.0.self_attn.q_proj.weight"].shape == (Hq * Dh, Dm)
    # round-trip through a real safetensors file
    p = tmp_path / "draft.safetensors"
    IO.save_safetensors(p, sd)
    w2 = IO.load_dflash_weights(p, cfg, embed_tokens=w.embed_tokens, lm_head=w.lm_head)
    assert np.allclose(w2.fc, w.fc) and np.allclose(w2.layers[1].mlp_down, w.layers[1].mlp_down)
    assert np.allclose(w2.layers[0].q_proj, w.layers[0].q_proj)
    assert np.allclose(w2.final_norm, w.final_norm)


def test_loaded_weights_produce_identical_logits(tmp_path):
    """The end-to-end proof: weights round-tripped through safetensors give
    bit-comparable draft logits."""
    rng = np.random.default_rng(5)
    cfg = _cfg()
    w = _weights(rng, cfg)
    p = tmp_path / "draft.safetensors"
    IO.save_dflash_weights(p, w)
    w2 = IO.load_dflash_weights(p, cfg, embed_tokens=w.embed_tokens, lm_head=w.lm_head)
    block = rng.integers(0, cfg.vocab_size, (1, cfg.block_size))
    th = rng.standard_normal((1, 6, cfg.num_target_layers * cfg.hidden_size)).astype(np.float32)
    rope = D.make_rope(cfg.head_dim)
    l1 = D.dflash_draft_forward(block, th, w, cfg, logits_start=1, rope_fn=rope)
    l2 = D.dflash_draft_forward(block, th, w2, cfg, logits_start=1, rope_fn=rope)
    assert np.abs(l1 - l2).max() < 1e-5
