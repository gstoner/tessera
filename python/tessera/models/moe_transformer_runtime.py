"""M5 — executable MoE-transformer: full decoder stack + autoregressive decode.

The capstone that ties M1–M4 into a whole-model forward and a KV-cached greedy
decode loop on a *scaled-faithful* config (Mac-executable; the full-scale graph
is the artifact target verified in :mod:`moe_transformer`).  One forward composes
every layer — RMSNorm → attention (MLA via :mod:`stdlib.attention`, or GQA dense)
→ residual → RMSNorm → FFN (capacity-aware MoE via :mod:`stdlib.moe`, or dense
SwiGLU) → residual — then a final norm + LM head.

The headline M5 oracle is **KV-cached greedy decode ≡ full recompute**: the
incremental decode loop (per-layer attention caches — MLA latent cache / GQA
K/V cache) must produce the same tokens as recomputing the full forward on the
growing prefix.  A non-circular, whole-model cache-consistency proof.

Scope honesty (M5.1): the runtime attention uses dense MLA (absorbed, the M3
primitive) / dense GQA. Wiring *DSA block-sparsity into the decode loop* (an
offset-aware indexer) is the M5.1 extension — DSA is proven at the primitive
level in M4 and at the graph level in :mod:`moe_transformer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..cache.latent import LatentKVCacheHandle
from ..stdlib import attention as _attn
from ..stdlib import moe as _moe
from .moe_transformer import MoETransformerConfig, verify_config


# ── weights ──────────────────────────────────────────────────────────────────
@dataclass
class LayerWeights:
    norm1: np.ndarray
    norm2: np.ndarray
    attn: dict       # MLA: {"mla": MLAWeights, "w_o": ...}; GQA: {"w_q","w_k","w_v","w_o"}
    ffn: dict        # MoE: {"router","gate","up","down",[shared...]}; dense: {"w_gate","w_up","w_down"}
    is_moe: bool


@dataclass
class ModelWeights:
    embed: np.ndarray        # (vocab, H)
    layers: list[LayerWeights]
    final_norm: np.ndarray   # (H,)
    lm_head: np.ndarray      # (H, vocab)


def synthetic_weights(config: MoETransformerConfig, *, seed: int = 0) -> ModelWeights:
    """Small-scale synthetic weights for a scaled config (tests/benchmarks)."""
    verify_config(config)
    rng = np.random.default_rng(seed)
    H = config.hidden_size
    s = 1.0 / np.sqrt(H)

    def n(*shape, sc=s):
        return (rng.standard_normal(shape) * sc).astype(np.float64)

    layers: list[LayerWeights] = []
    Hq, Hkv, D = config.num_attention_heads, config.num_kv_heads, config.head_dim
    for li in range(config.num_layers):
        if config.attn_kind == "mla":
            d_c, d_rope = config.kv_lora_rank, config.rope_head_dim
            d_nope = D - d_rope
            mlaw = _attn.MLAWeights(
                w_dkv=n(H, d_c), w_uk=n(d_c, Hq * d_nope, sc=1.0 / np.sqrt(d_c)),
                w_uv=n(d_c, Hq * D, sc=1.0 / np.sqrt(d_c)),
                w_q=n(H, Hq * (d_nope + d_rope)), w_kr=n(H, d_rope),
                num_heads=Hq, d_nope=d_nope, d_rope=d_rope, d_v=D)
            attn = {"kind": "mla", "mla": mlaw, "w_o": n(Hq * D, H, sc=1.0 / np.sqrt(Hq * D))}
        else:
            attn = {"kind": "gqa", "w_q": n(H, Hq * D), "w_k": n(H, Hkv * D),
                    "w_v": n(H, Hkv * D), "w_o": n(Hq * D, H, sc=1.0 / np.sqrt(Hq * D))}

        is_moe = config.is_moe_layer(li)
        if is_moe:
            E, F = config.num_experts, config.moe_intermediate_size
            ffn = {"router": n(H, E), "gate": n(E, H, F), "up": n(E, H, F),
                   "down": n(E, F, H, sc=1.0 / np.sqrt(F))}
            if config.num_shared_experts > 0:
                Fs = config.shared_expert_intermediate_size
                ffn["shared"] = (n(H, Fs), n(H, Fs), n(Fs, H, sc=1.0 / np.sqrt(Fs)))
        else:
            Fd = config.dense_intermediate_size or config.shared_expert_intermediate_size
            ffn = {"w_gate": n(H, Fd), "w_up": n(H, Fd), "w_down": n(Fd, H, sc=1.0 / np.sqrt(Fd))}

        layers.append(LayerWeights(norm1=np.ones(H), norm2=np.ones(H),
                                   attn=attn, ffn=ffn, is_moe=is_moe))
    return ModelWeights(embed=n(config.vocab_size, H, sc=1.0),
                        layers=layers, final_norm=np.ones(H),
                        lm_head=n(H, config.vocab_size, sc=s))


# ── primitive helpers ────────────────────────────────────────────────────────
def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * w


def _swiglu(x, wg, wu, wd):
    g = x @ wg
    return ((g * (1.0 / (1.0 + np.exp(-g)))) * (x @ wu)) @ wd


def _ffn(h: np.ndarray, lw: LayerWeights, config: MoETransformerConfig) -> np.ndarray:
    if not lw.is_moe:
        return _swiglu(h, lw.ffn["w_gate"], lw.ffn["w_up"], lw.ffn["w_down"])
    shared = lw.ffn.get("shared")
    res = _moe.moe_forward(
        h.astype(np.float32), lw.ffn["router"].astype(np.float32),
        lw.ffn["gate"].astype(np.float32), lw.ffn["up"].astype(np.float32),
        lw.ffn["down"].astype(np.float32), top_k=config.num_experts_per_tok,
        shared=tuple(w.astype(np.float32) for w in shared) if shared else None,
        capacity_factor=None)
    return np.asarray(res.y, dtype=np.float64)


# ── attention (prefill + decode) ─────────────────────────────────────────────
def _gqa_project(x, lw, config):
    Hq, Hkv, D = config.num_attention_heads, config.num_kv_heads, config.head_dim
    S = x.shape[0]
    q = (x @ lw.attn["w_q"]).reshape(S, Hq, D)
    k = (x @ lw.attn["w_k"]).reshape(S, Hkv, D)
    v = (x @ lw.attn["w_v"]).reshape(S, Hkv, D)
    return q, k, v


def _project_qkv_materialized(x, lw, config, positions):
    """Materialized, rope'd per-head Q/K/V for the DSA path (the indexer needs
    real K). Works for both attention kinds: MLA expands the latent to per-head
    K/V (numerically the MLA primitive — absorb ≡ materialized, proven in M3),
    GQA projects directly. Returns Q (S,Hq,D), K (S,Hkv,D), V (S,Hkv,Dv)."""
    if config.attn_kind == "mla":
        mlaw = lw.attn["mla"]
        Hh, d_nope, d_rope, d_v = mlaw.num_heads, mlaw.d_nope, mlaw.d_rope, mlaw.d_v
        S = x.shape[0]
        c = x @ mlaw.w_dkv                                          # (S,d_c)
        k_nope = (c @ mlaw.w_uk).reshape(S, Hh, d_nope)
        k_rope = _attn.apply_rope(x @ mlaw.w_kr, positions)         # (S,d_rope) shared
        K = np.concatenate([k_nope, np.broadcast_to(k_rope[:, None, :], (S, Hh, d_rope))],
                           axis=-1)                                 # (S,Hh,D)
        q = (x @ mlaw.w_q).reshape(S, Hh, d_nope + d_rope)
        q_rope = _attn.apply_rope(q[..., d_nope:].transpose(1, 0, 2), positions).transpose(1, 0, 2)
        Q = np.concatenate([q[..., :d_nope], q_rope], axis=-1)     # (S,Hh,D)
        V = (c @ mlaw.w_uv).reshape(S, Hh, d_v)
        return Q, K, V
    Hq, Hkv = config.num_attention_heads, config.num_kv_heads
    q, k, v = _gqa_project(x, lw, config)
    qr = _attn.apply_rope(q.transpose(1, 0, 2), positions).transpose(1, 0, 2)
    kr = _attn.apply_rope(k.transpose(1, 0, 2), positions).transpose(1, 0, 2)
    return qr, kr, v


def _dsa_attend(Q, K, V, config, q_positions):
    """Block-sparse attention over materialized per-head Q/K/V → (Sq, Hq*Dv)."""
    Sq, Hq, D = Q.shape
    Hkv, Dv = K.shape[1], V.shape[-1]
    Q4 = Q.transpose(1, 0, 2)[None]                                 # (1,Hq,Sq,D)
    K4 = K.transpose(1, 0, 2)[None]                                 # (1,Hkv,Sk,D)
    V4 = V.transpose(1, 0, 2)[None]
    out = _attn.dsa_block_sparse_attention(
        Q4, K4, V4, top_k_blocks=config.dsa_top_k_blocks,
        block_size=config.dsa_block_size, causal=True,
        q_positions=np.asarray(q_positions))[0]                    # (Hq,Sq,Dv)
    return out.transpose(1, 0, 2).reshape(Sq, Hq * Dv)


def _attn_prefill(x, lw, config, max_seq):
    """Full causal attention over a prompt; returns (out (S,H), cache).

    ``max_seq`` sizes the MLA latent cache for the whole decode (prompt + new
    tokens). For the recompute ``forward`` path the cache is unused, so
    ``max_seq = S`` is fine."""
    if config.sparse == "dsa":
        # DSA layers materialize K/V (the indexer needs real keys) and cache
        # them for an offset-aware decode step.
        S = x.shape[0]
        Q, K, V = _project_qkv_materialized(x, lw, config, np.arange(S))
        out = _dsa_attend(Q, K, V, config, np.arange(S))
        return out @ lw.attn["w_o"], ("dsa", K, V)
    if lw.attn["kind"] == "mla":
        mlaw = lw.attn["mla"]
        o, c, kr = _attn.mla_prefill(x, mlaw)
        lat = LatentKVCacheHandle(latent_dim=mlaw.d_c, max_seq=max_seq, dtype="fp64")
        rope = LatentKVCacheHandle(latent_dim=mlaw.d_rope, max_seq=max_seq, dtype="fp64")
        lat.append(c); rope.append(kr)
        return o @ lw.attn["w_o"], ("mla", lat, rope)
    Hq, Hkv, D = config.num_attention_heads, config.num_kv_heads, config.head_dim
    S = x.shape[0]
    q, k, v = _gqa_project(x, lw, config)
    pos = np.arange(S)
    qr = _attn.apply_rope(q.transpose(1, 0, 2), pos).transpose(1, 0, 2)
    kr = _attn.apply_rope(k.transpose(1, 0, 2), pos).transpose(1, 0, 2)
    Q = qr.transpose(1, 0, 2)[None]                    # (1,Hq,S,D)
    K = kr.transpose(1, 0, 2)[None]                    # (1,Hkv,S,D)
    V = v.transpose(1, 0, 2)[None]
    out = _attn.dense_causal_attention(Q, K, V)[0]     # (Hq,S,D)
    out = out.transpose(1, 0, 2).reshape(S, Hq * D)
    return out @ lw.attn["w_o"], ("gqa", kr, v)        # cache rope'd K + V


def _attn_decode(x_t, lw, cache, config, position):
    """Single-token attention against the cache; returns (out (1,H), cache)."""
    Hq, Hkv, D = config.num_attention_heads, config.num_kv_heads, config.head_dim
    if cache[0] == "dsa":
        _, K_prev, V_prev = cache                      # materialized (P,Hkv,D)/(P,Hkv,Dv)
        Qn, Kn, Vn = _project_qkv_materialized(x_t, lw, config, np.array([position]))
        K = np.concatenate([K_prev, Kn], axis=0)       # (P+1,Hkv,D)
        V = np.concatenate([V_prev, Vn], axis=0)
        out = _dsa_attend(Qn, K, V, config, np.array([position]))
        return out @ lw.attn["w_o"], ("dsa", K, V)
    if cache[0] == "mla":
        _, c_cache, kr_cache = cache
        lat = c_cache; rope = kr_cache
        o = _attn.mla_decode_step(x_t, lat, rope, lw.attn["mla"])
        return o @ lw.attn["w_o"], ("mla", lat, rope)
    _, K_prev, V_prev = cache                          # K_prev rope'd (P,Hkv,D), V_prev (P,Hkv,D)
    q, k, v = _gqa_project(x_t, lw, config)            # (1,Hq/Hkv,D)
    qr = _attn.apply_rope(q.transpose(1, 0, 2), np.array([position])).transpose(1, 0, 2)
    krt = _attn.apply_rope(k.transpose(1, 0, 2), np.array([position])).transpose(1, 0, 2)
    K = np.concatenate([K_prev, krt], axis=0)          # (P+1,Hkv,D)
    V = np.concatenate([V_prev, v], axis=0)
    g = Hq // Hkv
    scale = 1.0 / np.sqrt(D)
    out = np.zeros((Hq, D), dtype=np.float64)
    for h in range(Hq):
        kv = h // g
        s = (qr[0, h] @ K[:, kv].T) * scale
        w = np.exp(s - s.max()); w = w / w.sum()
        out[h] = w @ V[:, kv]
    out = out.reshape(1, Hq * D)
    return out @ lw.attn["w_o"], ("gqa", K, V)


# ── model forward + decode ───────────────────────────────────────────────────
def embed_tokens(weights: ModelWeights, token_ids) -> np.ndarray:
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    return weights.embed[ids].astype(np.float64)


def forward(config: MoETransformerConfig, weights: ModelWeights, token_ids) -> np.ndarray:
    """Full causal forward → logits ``(S, vocab)`` (the recompute reference)."""
    x = embed_tokens(weights, token_ids)
    eps = config.rms_norm_eps
    S = x.shape[0]
    for lw in weights.layers:
        a, _ = _attn_prefill(rmsnorm(x, lw.norm1, eps), lw, config, S)
        x = x + a
        x = x + _ffn(rmsnorm(x, lw.norm2, eps), lw, config)
    x = rmsnorm(x, weights.final_norm, eps)
    return x @ weights.lm_head


@dataclass
class DecodeState:
    caches: list[Any]
    position: int


def prefill(config: MoETransformerConfig, weights: ModelWeights, token_ids,
            *, max_seq: int | None = None):
    """Run the prompt and return ``(last_logits (vocab,), DecodeState)``.

    ``max_seq`` (≥ prompt + planned new tokens) sizes the per-layer MLA caches.
    """
    x = embed_tokens(weights, token_ids)
    eps = config.rms_norm_eps
    S = x.shape[0]
    cap = max_seq if max_seq is not None else S
    caches = []
    for lw in weights.layers:
        a, cache = _attn_prefill(rmsnorm(x, lw.norm1, eps), lw, config, cap)
        x = x + a
        x = x + _ffn(rmsnorm(x, lw.norm2, eps), lw, config)
        caches.append(cache)
    logits = rmsnorm(x, weights.final_norm, eps) @ weights.lm_head
    return logits[-1], DecodeState(caches=caches, position=S)


def decode_step(config: MoETransformerConfig, weights: ModelWeights,
                state: DecodeState, token_id: int):
    """Advance decode by one token → ``(logits (vocab,), state)``."""
    eps = config.rms_norm_eps
    x = embed_tokens(weights, [token_id])              # (1,H)
    new_caches = []
    for lw, cache in zip(weights.layers, state.caches):
        a, cache = _attn_decode(rmsnorm(x, lw.norm1, eps), lw, cache, config, state.position)
        x = x + a
        x = x + _ffn(rmsnorm(x, lw.norm2, eps), lw, config)
        new_caches.append(cache)
    logits = (rmsnorm(x, weights.final_norm, eps) @ weights.lm_head)[0]
    return logits, DecodeState(caches=new_caches, position=state.position + 1)


def greedy_generate(config: MoETransformerConfig, weights: ModelWeights,
                    prompt_ids, max_new_tokens: int) -> list[int]:
    """KV-cached greedy autoregressive decode → list of generated token ids."""
    total = len(list(np.asarray(prompt_ids).reshape(-1))) + max_new_tokens
    logits, state = prefill(config, weights, prompt_ids, max_seq=total)
    out: list[int] = []
    for _ in range(max_new_tokens):
        tok = int(np.argmax(logits))
        out.append(tok)
        logits, state = decode_step(config, weights, state, tok)
    return out


__all__ = [
    "LayerWeights", "ModelWeights", "synthetic_weights",
    "rmsnorm", "forward", "prefill", "decode_step", "greedy_generate",
    "embed_tokens", "DecodeState",
]
