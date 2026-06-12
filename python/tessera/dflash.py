"""DFlash — block-diffusion draft model for flash speculative decoding.

Numpy-reference composition of the DFlash draft (z-lab/dflash,
arXiv:2602.06036) on top of Tessera primitives. The draft predicts a whole
block of ``block_size - 1`` tokens in a single parallel forward (no per-block
denoising loop), conditioned on injected target hidden features, then the
target model verifies the block linearly.

This module owns the *model + orchestration* surface; the per-layer attention
keystone lives in :func:`tessera.nn.functional.block_diffusion_attention`.

Faithful to ``model_mlx.py``:
  * ``DFlashDraftModel.__call__`` (lines 181-198) → :func:`dflash_draft_forward`
  * ``DFlashDecoderLayer.__call__`` (lines 127-129) → :func:`dflash_decoder_layer`
  * draft loop accept rule (line 519)              → :func:`dflash_linear_verify`
  * draft → verify → accept cycle (lines 491-567)  → :func:`dflash_step`

All compute composes through ``tessera.ops`` / ``tessera.nn.functional`` so an
active autodiff tape sees every primitive and the Apple GPU lane picks up the
attention core (heads folded into batch → rank-3 ``ops.flash_attn``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from . import ops
from .nn import functional as F


# ---------------------------------------------------------------------------
# Config + weight containers
# ---------------------------------------------------------------------------

@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    block_size: int
    target_layer_ids: Tuple[int, ...]
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sliding_window: Optional[int] = None
    layer_types: Tuple[str, ...] = ()
    final_logit_softcapping: Optional[float] = None
    embed_scale: float = 1.0
    mask_token_id: int = 0

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = ("full_attention",) * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")
        bad = set(self.layer_types) - {"full_attention", "sliding_attention"}
        if bad:
            raise ValueError(f"unsupported layer_types: {sorted(bad)}")
        if "sliding_attention" in self.layer_types and self.sliding_window is None:
            raise ValueError("sliding_attention layers require sliding_window")

    @property
    def num_target_layers(self) -> int:
        return len(self.target_layer_ids)


@dataclass
class DFlashLayerWeights:
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray
    input_layernorm: np.ndarray
    post_attention_layernorm: np.ndarray
    mlp_gate: np.ndarray
    mlp_up: np.ndarray
    mlp_down: np.ndarray


@dataclass
class DFlashWeights:
    embed_tokens: np.ndarray            # (vocab, D) — shared + frozen with target
    fc: np.ndarray                      # (num_target_layers * D, D)
    hidden_norm: np.ndarray             # (D,)
    layers: List[DFlashLayerWeights]
    final_norm: np.ndarray              # (D,)
    lm_head: Optional[np.ndarray] = None  # (D, vocab); None → tied (embed_tokens.T)


# ---------------------------------------------------------------------------
# Rope
# ---------------------------------------------------------------------------

def make_rope(head_dim: int, theta: float = 10000.0) -> Callable[[np.ndarray, int], np.ndarray]:
    """Standard rotary embedding callable ``rope_fn(t_BHSD, offset)``.

    Rotates even/odd pairs of the last (``head_dim``) axis by
    ``(offset + pos) * inv_freq``; ``offset`` shifts the absolute positions so
    block tokens sit after the injected context (DFlash ``model_mlx`` 102-104).
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))

    def rope_fn(t: np.ndarray, offset: int) -> np.ndarray:
        t = np.asarray(t)
        pos = (offset + np.arange(t.shape[2]))[None, None, :, None]
        ang = pos * inv_freq
        cos, sin = np.cos(ang), np.sin(ang)
        e, o = t[..., 0::2], t[..., 1::2]
        out = np.empty_like(t)
        out[..., 0::2] = e * cos - o * sin
        out[..., 1::2] = e * sin + o * cos
        return out

    return rope_fn


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def target_feature_projection(target_hidden, fc, hidden_norm, *, eps: float = 1e-6):
    """Inject target features: ``x_ctx = RMSNorm(target_hidden @ fc)``.

    ``target_hidden`` is the concatenation of a fixed set of target layers'
    hidden states ``(B, S, num_target_layers * D)``; ``fc`` projects it down to
    ``D`` and a final RMSNorm conditions it (DFlash ``model_mlx`` line 189). The
    result is fed unchanged to every draft layer as the attention context.
    """
    x_ctx = F.linear_general(target_hidden, fc)
    return F.rms_norm(x_ctx, hidden_norm, eps=eps)


def dflash_decoder_layer(x, x_ctx, lw: DFlashLayerWeights, cfg: DFlashConfig,
                         layer_idx: int, *, rope_fn=None,
                         cache_keys=None, cache_values=None, cache_offset: int = 0):
    """One DFlash decoder layer: pre-norm attention + pre-norm SwiGLU, residual.

    Mirrors ``DFlashDecoderLayer.__call__``: ``input_layernorm`` is applied to
    the query/proposal source ``x`` only (the injected ``x_ctx`` is projected
    raw inside attention).
    """
    sliding = cfg.sliding_window if cfg.layer_types[layer_idx] == "sliding_attention" else None
    x_normed = F.rms_norm(x, lw.input_layernorm, eps=cfg.rms_norm_eps)
    attn = F.block_diffusion_attention(
        x_normed, x_ctx,
        q_proj=lw.q_proj, k_proj=lw.k_proj, v_proj=lw.v_proj, o_proj=lw.o_proj,
        q_norm=lw.q_norm, k_norm=lw.k_norm,
        num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, rope_fn=rope_fn, cache_offset=cache_offset,
        cache_keys=cache_keys, cache_values=cache_values,
        sliding_window=sliding, eps=cfg.rms_norm_eps)
    h = np.asarray(x) + np.asarray(attn)
    h_normed = F.rms_norm(h, lw.post_attention_layernorm, eps=cfg.rms_norm_eps)
    mlp = F.swiglu(h_normed, lw.mlp_gate, lw.mlp_up, lw.mlp_down)
    return np.asarray(h) + np.asarray(mlp)


def dflash_draft_forward(block_tokens, target_hidden, w: DFlashWeights,
                         cfg: DFlashConfig, *, logits_start: int = 0,
                         rope_fn=None, caches: Optional[List[Tuple[Any, Any]]] = None):
    """Run the draft over a token block in a single parallel forward → logits.

    ``block_tokens`` ``(B, L)`` is ``[prev_token, MASK, ...]``; ``target_hidden``
    ``(B, S, num_target_layers * D)`` is the tapped target context. ``caches`` is
    an optional per-layer ``(keys, values)`` of accumulated context KV (shape
    ``(B, Sc, num_kv_heads, head_dim)``). ``logits_start`` drops leading
    positions before the LM head (DFlash uses ``1`` so the bonus token's logits
    are skipped). Returns logits ``(B, L - logits_start, vocab)``.
    """
    tokens = np.asarray(block_tokens, dtype=np.int64)
    h = w.embed_tokens[tokens] * cfg.embed_scale          # (B, L, D)
    x_ctx = target_feature_projection(target_hidden, w.fc, w.hidden_norm, eps=cfg.rms_norm_eps)
    S = np.asarray(x_ctx).shape[1]
    for i, lw in enumerate(w.layers):
        ck = cv = None
        if caches is not None and caches[i] is not None:
            ck, cv = caches[i]
        cache_offset = 0 if ck is None else np.asarray(ck).shape[1]
        h = dflash_decoder_layer(h, x_ctx, lw, cfg, i, rope_fn=rope_fn,
                                 cache_keys=ck, cache_values=cv, cache_offset=cache_offset)
    if logits_start:
        h = np.asarray(h)[:, logits_start:]
    h = F.rms_norm(h, w.final_norm, eps=cfg.rms_norm_eps)
    lm_head = w.embed_tokens.T if w.lm_head is None else w.lm_head
    logits = F.linear_general(h, lm_head)
    if cfg.final_logit_softcapping is not None:
        cap = cfg.final_logit_softcapping
        logits = np.tanh(np.asarray(logits) / cap) * cap
    return np.asarray(logits)


# ---------------------------------------------------------------------------
# Verification + step
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DFlashVerification:
    accepted: int               # number of draft tokens accepted
    new_tokens: List[int]       # accepted draft prefix + 1 bonus target token


def dflash_linear_verify(draft_tokens, target_tokens) -> DFlashVerification:
    """Linear block acceptance (DFlash ``model_mlx`` line 519).

    Accept the longest prefix where the draft token equals the target's sampled
    token, then append the target's token at the first divergence (the bonus
    token). With ``draft_tokens`` of length ``b`` and ``target_tokens`` of
    length ``b`` (the target sampled one extra position via the prepended bonus
    token), this yields ``accepted`` accepted drafts plus one corrected token.
    """
    d = list(np.asarray(draft_tokens).reshape(-1).tolist())
    t = list(np.asarray(target_tokens).reshape(-1).tolist())
    accepted = next((i for i in range(len(d)) if d[i] != t[i]), len(d))
    new_tokens = d[:accepted] + [t[accepted]] if accepted < len(t) else d[:accepted]
    return DFlashVerification(accepted=accepted, new_tokens=new_tokens)


def dflash_step(prev_token: int, target_hidden, w: DFlashWeights, cfg: DFlashConfig,
                target_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                *, block_size: Optional[int] = None, rope_fn=None,
                sampler: Optional[Callable[[np.ndarray], np.ndarray]] = None
                ) -> Tuple[DFlashVerification, np.ndarray]:
    """One DFlash draft → verify → accept cycle (``model_mlx`` 491-567).

    1. Build the block ``[prev_token, MASK, ...]`` and draft ``block_size - 1``
       tokens in parallel (``logits_start=1`` drops the bonus position).
    2. Run ``target_fn`` over ``[prev_token, *draft_tokens]`` — one forward — to
       get target logits and the fresh tapped hidden states.
    3. Accept the longest matching prefix + one bonus token.

    ``target_fn(token_ids_1xT) -> (logits_1xTxV, hidden_1xTx(nL*D))`` is the
    target model + multi-layer tap. ``sampler`` defaults to greedy argmax.
    Returns ``(verification, fresh_target_hidden_for_accepted_prefix)``.
    """
    bs = int(block_size if block_size is not None else cfg.block_size)
    sample = sampler if sampler is not None else (lambda lg: np.argmax(lg, axis=-1))

    block = F.mask_token_block(np.asarray([prev_token]), bs, cfg.mask_token_id)  # (1, bs)
    draft_logits = dflash_draft_forward(block, target_hidden, w, cfg,
                                        logits_start=1, rope_fn=rope_fn)
    draft_tokens = np.asarray(sample(draft_logits)).reshape(-1)                  # (bs-1,)

    verify_input = np.concatenate([[prev_token], draft_tokens])[None, :]         # (1, bs)
    target_logits, fresh_hidden = target_fn(verify_input)
    target_tokens = np.asarray(sample(target_logits)).reshape(-1)                # (bs,)

    result = dflash_linear_verify(draft_tokens, target_tokens)
    fresh_hidden = np.asarray(fresh_hidden)[:, : result.accepted + 1, :]
    return result, fresh_hidden


def dflash_generate(prompt, w: DFlashWeights, cfg: DFlashConfig,
                    target_step: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                    *, max_new_tokens: int, block_size: Optional[int] = None,
                    rope_fn=None, sampler: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                    eos_id: Optional[int] = None) -> List[int]:
    """Full DFlash speculative generation loop (``model_mlx`` stream_generate).

    ``target_step(token_ids_1xT) -> (logits_1xTxV, target_hidden_1xTx(nL*D))`` is
    a *causal* target forward over the running sequence (this numpy reference
    recomputes statelessly; a production target keeps a KV cache and rolls back
    rejected tokens via ``advance_kv``). The multi-layer tapped ``target_hidden``
    conditions the draft.

    Each cycle drafts ``block_size - 1`` tokens in one parallel forward, verifies
    them against the target in one forward, and commits the accepted prefix plus
    one bonus token. With greedy sampling the output is **identical** to greedy
    autoregressive decode from ``target_step`` — speculation changes only speed,
    never the tokens. Returns the prompt followed by the generated tokens.
    """
    bs = int(block_size if block_size is not None else cfg.block_size)
    sample = sampler if sampler is not None else (lambda lg: np.argmax(lg, axis=-1))
    tokens = list(np.asarray(prompt, dtype=np.int64).reshape(-1).tolist())
    prompt_len = len(tokens)

    # Prefill: first token + tapped hidden over the prompt.
    logits, hidden = target_step(np.asarray(tokens, dtype=np.int64)[None, :])
    first = int(np.asarray(sample(logits[:, -1:])).reshape(-1)[0])
    tokens.append(first)
    target_hidden = np.asarray(hidden)                       # (1, prompt_len, nL*D)
    if eos_id is not None and first == eos_id:
        return tokens
    n = 1

    while n < max_new_tokens:
        b = min(bs, max_new_tokens - n + 1)
        if b <= 1:
            break
        prev = int(tokens[-1])
        block = F.mask_token_block(np.asarray([prev]), b, cfg.mask_token_id)      # (1, b)
        draft_logits = dflash_draft_forward(block, target_hidden, w, cfg,
                                            logits_start=1, rope_fn=rope_fn)
        draft_tokens = np.asarray(sample(draft_logits)).reshape(-1)               # (b-1,)

        # Verify: causal target over [running prefix .. prev, draft...]. The last
        # ``b`` positions are verify_input = [prev, *draft_tokens].
        full = np.asarray(tokens + draft_tokens.tolist(), dtype=np.int64)[None, :]
        v_logits, v_hidden = target_step(full)
        target_tokens = np.asarray(sample(v_logits[:, -b:])).reshape(-1)          # (b,)

        result = dflash_linear_verify(draft_tokens, target_tokens)
        new_tokens = result.new_tokens[: max_new_tokens - n]
        eos_cut = next((i for i, t in enumerate(new_tokens) if t == eos_id), None)
        if eos_cut is not None:
            new_tokens = new_tokens[: eos_cut + 1]
        tokens.extend(new_tokens)
        n += len(new_tokens)
        if eos_cut is not None:
            break
        # Carry the target hidden for the accepted context into the next draft
        # (verify positions 0..accepted), mirroring hidden[:, :accepted+1].
        target_hidden = np.asarray(v_hidden)[:, -b:][:, : result.accepted + 1, :]

    return tokens


__all__ = [
    "DFlashConfig",
    "DFlashLayerWeights",
    "DFlashWeights",
    "DFlashVerification",
    "make_rope",
    "target_feature_projection",
    "dflash_decoder_layer",
    "dflash_draft_forward",
    "dflash_linear_verify",
    "dflash_step",
    "dflash_generate",
]
