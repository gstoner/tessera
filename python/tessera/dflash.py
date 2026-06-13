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

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

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
                         cache_keys=None, cache_values=None, cache_offset: int = 0,
                         attention_fn=None):
    """One DFlash decoder layer: pre-norm attention + pre-norm SwiGLU, residual.

    Mirrors ``DFlashDecoderLayer.__call__``: ``input_layernorm`` is applied to
    the query/proposal source ``x`` only (the injected ``x_ctx`` is projected
    raw inside attention). ``attention_fn`` (e.g.
    :func:`apple_gpu_attention_fn`) routes the attention core onto a backend.
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
        sliding_window=sliding, eps=cfg.rms_norm_eps, attention_fn=attention_fn)
    h = np.asarray(x) + np.asarray(attn)
    h_normed = F.rms_norm(h, lw.post_attention_layernorm, eps=cfg.rms_norm_eps)
    mlp = F.swiglu(h_normed, lw.mlp_gate, lw.mlp_up, lw.mlp_down)
    return np.asarray(h) + np.asarray(mlp)


def dflash_draft_forward(block_tokens, target_hidden, w: DFlashWeights,
                         cfg: DFlashConfig, *, logits_start: int = 0,
                         rope_fn=None, caches: Optional[List[Tuple[Any, Any]]] = None,
                         attention_fn=None):
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
                                 cache_keys=ck, cache_values=cv, cache_offset=cache_offset,
                                 attention_fn=attention_fn)
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


def dflash_speculative_verify(draft_tokens, draft_probs, target_probs,
                              rng) -> DFlashVerification:
    """Distribution-preserving speculative-sampling acceptance (Leviathan rule).

    Generalizes :func:`dflash_linear_verify` to non-greedy sampling: draft token
    ``d_i`` is accepted with probability ``min(1, p_target/p_draft)``; on the
    first rejection a corrected token is drawn from the residual
    ``normalize(relu(p_target - p_draft))``; if the whole block is accepted a
    bonus token is drawn from the target's distribution at the next position.
    The marginal of every emitted token equals the target's distribution — the
    output is distributionally identical to plain target sampling.

    ``draft_probs`` is ``(b-1, V)`` (one row per drafted position), ``target_probs``
    is ``(b, V)`` (one extra row for the bonus). ``rng`` is a numpy Generator.
    With one-hot (temperature-0) distributions this reduces to exact-match.
    """
    d = np.asarray(draft_tokens).reshape(-1)
    dp = np.asarray(draft_probs, dtype=np.float64)
    tp = np.asarray(target_probs, dtype=np.float64)
    V = tp.shape[1]
    new: List[int] = []
    accepted = 0
    for i in range(len(d)):
        pd = dp[i, d[i]]
        pt = tp[i, d[i]]
        if pd > 0.0 and rng.random() <= min(1.0, pt / pd):
            new.append(int(d[i]))
            accepted += 1
        else:
            resid = np.maximum(tp[i] - dp[i], 0.0)
            s = resid.sum()
            tok = int(rng.choice(V, p=resid / s)) if s > 0 else int(np.argmax(tp[i]))
            new.append(tok)
            return DFlashVerification(accepted=accepted, new_tokens=new)
    # whole block accepted → bonus from the extra target row.
    bonus = int(rng.choice(V, p=tp[len(d)])) if tp.shape[0] > len(d) else int(np.argmax(tp[-1]))
    new.append(bonus)
    return DFlashVerification(accepted=accepted, new_tokens=new)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def _softmax_lastaxis(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def make_sampler(temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0,
                 rng=None) -> Callable[[np.ndarray], np.ndarray]:
    """Build a token sampler ``sampler(logits) -> token_ids``.

    ``temperature == 0`` is greedy ``argmax``. Otherwise logits are scaled by
    ``1/temperature``, optionally restricted to the ``top_k`` highest and/or the
    ``top_p`` nucleus, then sampled. ``rng`` (a numpy Generator) makes sampling
    reproducible; if ``None`` a fresh default generator is used per call.
    """
    def sampler(logits: np.ndarray) -> np.ndarray:
        lg = np.asarray(logits, dtype=np.float64)
        if temperature <= 0.0:
            return np.argmax(lg, axis=-1)
        flat = (lg / float(temperature)).reshape(-1, lg.shape[-1])
        V = flat.shape[-1]
        gen = rng if rng is not None else np.random.default_rng()
        out = np.empty(flat.shape[0], dtype=np.int64)
        for r in range(flat.shape[0]):
            row = flat[r].copy()
            if top_k and 0 < top_k < V:
                kth = np.partition(row, -top_k)[-top_k]
                row = np.where(row < kth, -np.inf, row)
            p = _softmax_lastaxis(row)
            if top_p and 0.0 < top_p < 1.0:
                order = np.argsort(-p)
                csum = np.cumsum(p[order])
                cut = int(np.searchsorted(csum, top_p)) + 1
                drop = order[cut:]
                p[drop] = 0.0
                p = p / p.sum()
            out[r] = gen.choice(V, p=p)
        return out.reshape(lg.shape[:-1])

    return sampler


def sampler_probs(logits, temperature: float = 1.0, top_k: int = 0,
                  top_p: float = 0.0) -> np.ndarray:
    """Return the (renormalized) sampling distribution for ``logits`` under the
    same temperature/top-k/top-p truncation a :func:`make_sampler` would apply —
    used by :func:`dflash_speculative_verify` to score draft vs target."""
    lg = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-8)
    flat = lg.reshape(-1, lg.shape[-1])
    V = flat.shape[-1]
    out = np.empty_like(flat)
    for r in range(flat.shape[0]):
        row = flat[r].copy()
        if top_k and 0 < top_k < V:
            kth = np.partition(row, -top_k)[-top_k]
            row = np.where(row < kth, -np.inf, row)
        p = _softmax_lastaxis(row)
        if top_p and 0.0 < top_p < 1.0:
            order = np.argsort(-p)
            csum = np.cumsum(p[order])
            cut = int(np.searchsorted(csum, top_p)) + 1
            p[order[cut:]] = 0.0
            p = p / p.sum()
        out[r] = p
    return out.reshape(lg.shape)


# ---------------------------------------------------------------------------
# Training loss (position-weighted block cross-entropy)
# ---------------------------------------------------------------------------

def dflash_position_weights(block_len: int, gamma: Optional[float] = None) -> np.ndarray:
    """Per-position loss weights ``wₖ = exp(-k/γ)`` (normalized), emphasizing the
    early positions of the drafted block (DFlash ``model_mlx`` loss decay,
    ``wₖ = exp(-(k-1)/γ)`` in 1-indexed form). ``γ`` defaults to ``block_len``."""
    g = float(gamma if gamma is not None else block_len)
    w = np.exp(-np.arange(block_len, dtype=np.float64) / max(g, 1e-8))
    return w / w.sum()


def dflash_block_loss(logits, targets, *, gamma: Optional[float] = None,
                      reduction: str = "mean"):
    """Position-weighted cross-entropy for training a DFlash draft block.

    ``logits`` ``(B, L, V)``, ``targets`` ``(B, L)`` int. Each block position is
    weighted by :func:`dflash_position_weights`. Returns a scalar (``mean``/
    ``sum`` over the batch) or per-example ``(B,)`` when ``reduction == "none"``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    B, L, V = logits.shape
    w = dflash_position_weights(L, gamma)
    logp = logits - (logits.max(-1, keepdims=True)
                     + np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)))
    ce = -np.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]  # (B, L)
    wce = (ce * w[None, :]).sum(-1)                                       # (B,)
    if reduction == "mean":
        return float(wce.mean())
    if reduction == "sum":
        return float(wce.sum())
    if reduction == "none":
        return wce
    raise ValueError(f"reduction must be mean/sum/none; got {reduction!r}")


def dflash_block_loss_grad(logits, targets, *, gamma: Optional[float] = None):
    """``dLoss/dlogits`` for the mean-reduced :func:`dflash_block_loss` — the
    explicit gradient for draft training (composes with any optimizer in
    ``tessera.optim``)."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    B, L, V = logits.shape
    w = dflash_position_weights(L, gamma)
    p = _softmax_lastaxis(logits)
    onehot = np.zeros_like(p)
    np.put_along_axis(onehot, targets[..., None], 1.0, axis=-1)
    return (p - onehot) * w[None, :, None] / float(B)


# ---------------------------------------------------------------------------
# Draft KV cache (efficient cached drafting)
# ---------------------------------------------------------------------------

class DraftKVCache:
    """Per-layer accumulated context KV for the DFlash draft.

    Each drafting step appends this step's projected+roped context KV (the
    injected target features) per layer; the draft attends to the full
    accumulation plus the current block's proposal. The proposal KV is never
    cached, and the context is always already-accepted, so no rollback is
    needed (cf. ``model_mlx`` ``make_cache`` + ``cache.update_and_fetch``).
    """

    def __init__(self, num_layers: int):
        self.keys: List[Optional[np.ndarray]] = [None] * num_layers
        self.values: List[Optional[np.ndarray]] = [None] * num_layers
        self.offset = 0

    def append(self, i: int, k, v) -> None:
        k = np.asarray(k)
        v = np.asarray(v)
        self.keys[i] = k if self.keys[i] is None else np.concatenate([self.keys[i], k], axis=1)
        self.values[i] = v if self.values[i] is None else np.concatenate([self.values[i], v], axis=1)

    def advance(self, s: int) -> None:
        self.offset += int(s)


def dflash_decoder_layer_cached(x, x_ctx, lw: DFlashLayerWeights, cfg: DFlashConfig,
                                layer_idx: int, cache: DraftKVCache, *, rope_fn=None,
                                attention_fn=None):
    """Cached DFlash decoder layer — attends to ``cache`` ++ this step's context
    and appends this step's projected context KV to ``cache``. Equivalent to
    :func:`dflash_decoder_layer` with the prior context supplied via the cache."""
    sliding = cfg.sliding_window if cfg.layer_types[layer_idx] == "sliding_attention" else None
    x_normed = F.rms_norm(x, lw.input_layernorm, eps=cfg.rms_norm_eps)
    attn, ck_new, cv_new = F.block_diffusion_attention(
        x_normed, x_ctx,
        q_proj=lw.q_proj, k_proj=lw.k_proj, v_proj=lw.v_proj, o_proj=lw.o_proj,
        q_norm=lw.q_norm, k_norm=lw.k_norm,
        num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, rope_fn=rope_fn, cache_offset=cache.offset,
        cache_keys=cache.keys[layer_idx], cache_values=cache.values[layer_idx],
        sliding_window=sliding, eps=cfg.rms_norm_eps, return_ctx_kv=True,
        attention_fn=attention_fn)
    cache.append(layer_idx, ck_new, cv_new)
    h = np.asarray(x) + np.asarray(attn)
    h_normed = F.rms_norm(h, lw.post_attention_layernorm, eps=cfg.rms_norm_eps)
    mlp = F.swiglu(h_normed, lw.mlp_gate, lw.mlp_up, lw.mlp_down)
    return np.asarray(h) + np.asarray(mlp)


def dflash_draft_forward_cached(block_tokens, target_hidden, w: DFlashWeights,
                                cfg: DFlashConfig, cache: DraftKVCache, *,
                                logits_start: int = 0, rope_fn=None, attention_fn=None):
    """Cached draft forward — threads ``cache`` through every layer so the draft
    attends to the full accumulated context (not just this step's). Output is
    identical to :func:`dflash_draft_forward` fed the same accumulated context;
    the cache is advanced by this step's context length."""
    tokens = np.asarray(block_tokens, dtype=np.int64)
    h = w.embed_tokens[tokens] * cfg.embed_scale
    x_ctx = target_feature_projection(target_hidden, w.fc, w.hidden_norm, eps=cfg.rms_norm_eps)
    S = np.asarray(x_ctx).shape[1]
    for i, lw in enumerate(w.layers):
        h = dflash_decoder_layer_cached(h, x_ctx, lw, cfg, i, cache, rope_fn=rope_fn,
                                        attention_fn=attention_fn)
    cache.advance(S)
    if logits_start:
        h = np.asarray(h)[:, logits_start:]
    h = F.rms_norm(h, w.final_norm, eps=cfg.rms_norm_eps)
    lm_head = w.embed_tokens.T if w.lm_head is None else w.lm_head
    logits = F.linear_general(h, lm_head)
    if cfg.final_logit_softcapping is not None:
        cap = cfg.final_logit_softcapping
        logits = np.tanh(np.asarray(logits) / cap) * cap
    return np.asarray(logits)


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


# ---------------------------------------------------------------------------
# Target-model hidden-state tap (multi-layer feature injection)
# ---------------------------------------------------------------------------

class HiddenStateTap:
    """Capture the outputs of a fixed set of target-model layers.

    The DFlash draft is conditioned on hidden states tapped from a fixed set of
    target layers (``model_mlx`` ``_patch_model`` / ``_LayerHook``). This wraps
    each selected layer's ``forward`` to record its output; after the target
    forward runs, :attr:`hidden_states` concatenates the per-layer captures along
    the last axis into the ``(B, S, num_target_layers * D)`` tensor that
    :func:`target_feature_projection` consumes.

    Works with any ``nn.Module`` layer container that is index-addressable
    (``nn.ModuleList`` or a plain list). Use as a context manager or call
    :meth:`install` / :meth:`remove` explicitly. ``object.__setattr__`` bypasses
    ``Module``'s attribute machinery so the wrap is transparent and reversible.
    """

    def __init__(self, layers, layer_ids):
        self.layers = layers
        self.layer_ids = list(layer_ids)
        self._captured: dict = {}
        self._installed = False

    def install(self) -> "HiddenStateTap":
        if self._installed:
            return self
        for i in self.layer_ids:
            layer = self.layers[i]
            orig = layer.forward

            def make(idx, orig_fwd):
                def hook(*a, **k):
                    out = orig_fwd(*a, **k)
                    self._captured[idx] = out[0] if isinstance(out, tuple) else out
                    return out
                return hook

            object.__setattr__(layer, "forward", make(i, orig))
        self._installed = True
        return self

    def remove(self) -> None:
        for i in self.layer_ids:
            try:
                object.__delattr__(self.layers[i], "forward")
            except AttributeError:
                pass
        self._installed = False

    def reset(self) -> None:
        self._captured.clear()

    @property
    def hidden_states(self) -> np.ndarray:
        missing = [i for i in self.layer_ids if i not in self._captured]
        if missing:
            raise RuntimeError(
                f"hidden-state tap has not captured layers {missing}; "
                "run the target forward while the tap is installed")
        return np.concatenate(
            [np.asarray(self._captured[i]) for i in self.layer_ids], axis=-1)

    def __enter__(self) -> "HiddenStateTap":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()


def capture_target_hidden(layers, layer_ids) -> HiddenStateTap:
    """Context manager that taps ``layers[i]`` for ``i in layer_ids`` (DFlash
    multi-layer feature injection). See :class:`HiddenStateTap`."""
    return HiddenStateTap(layers, layer_ids)


# ---------------------------------------------------------------------------
# Apple GPU attention core
# ---------------------------------------------------------------------------

def apple_gpu_attention_fn(q, k, v, *, scale=None, causal=False, attn_bias=None):
    """``attention_fn`` for :func:`tessera.nn.functional.block_diffusion_attention`
    that runs the rank-3 attention core on the Apple GPU ``metal_runtime`` lane.

    Routes through the P0 ``flash_attn`` / ``flash_attn_bias`` runtime symbols.
    Pass as ``block_diffusion_attention(..., attention_fn=apple_gpu_attention_fn)``
    (or via :func:`dflash_decoder_layer`'s composition) to execute the DFlash
    draft attention on Metal. Falls back to the numpy reference off-Darwin / when
    a tensor is outside the kernel envelope (the dispatcher handles that).
    """
    from . import runtime as _rt
    operands = [q, k, v] if attn_bias is None else [q, k, v, attn_bias]
    return _rt._apple_gpu_dispatch_flash_attn(
        "tessera.flash_attn", operands, {"scale": scale, "causal": causal}, np)


def dflash_generate_cached(prompt, draft_w: DFlashWeights, cfg: DFlashConfig,
                           target, *, max_new_tokens: int,
                           block_size: Optional[int] = None, rope_fn=None,
                           temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0,
                           rng=None, eos_id: Optional[int] = None) -> List[int]:
    """Efficient DFlash generation: cached draft (#1) + stateful target (#3) +
    optional sampling (#2).

    ``target`` is a stateful object exposing ``reset()``, ``step(tokens) ->
    (logits, hidden)`` (causal, KV-cached, appends to its cache), and
    ``rollback(n)`` (discard the last ``n`` cached tokens) — e.g.
    :class:`tessera.dflash_reference.ReferenceDecoderLM`. The draft accumulates
    its own per-layer context cache so it attends to the full history.

    With ``temperature == 0`` (greedy) the output is identical to greedy
    autoregressive decode from ``target``; with sampling it uses the
    distribution-preserving rejection rule (:func:`dflash_speculative_verify`),
    so the output distribution matches plain target sampling. Returns the prompt
    followed by the generated tokens.
    """
    bs = int(block_size if block_size is not None else cfg.block_size)
    greedy = temperature <= 0.0
    if rng is None:
        rng = np.random.default_rng()
    sample = make_sampler(temperature, top_k, top_p, rng)

    target.reset()
    tokens = list(np.asarray(prompt, dtype=np.int64).reshape(-1).tolist())

    logits, hidden = target.step(np.asarray(tokens, dtype=np.int64)[None, :])
    first = int(np.asarray(sample(logits[:, -1:])).reshape(-1)[0])
    tokens.append(first)
    target_hidden = np.asarray(hidden)
    if eos_id is not None and first == eos_id:
        return tokens
    n = 1
    draft_cache = DraftKVCache(cfg.num_hidden_layers)

    while n < max_new_tokens:
        b = min(bs, max_new_tokens - n + 1)
        if b <= 1:
            break
        prev = int(tokens[-1])
        block = F.mask_token_block(np.asarray([prev]), b, cfg.mask_token_id)
        draft_logits = dflash_draft_forward_cached(block, target_hidden, draft_w, cfg,
                                                   draft_cache, logits_start=1, rope_fn=rope_fn)
        draft_tokens = np.asarray(sample(draft_logits)).reshape(-1)               # (b-1,)

        verify_input = np.concatenate([[prev], draft_tokens])[None, :]            # (1, b)
        v_logits, v_hidden = target.step(verify_input)
        if greedy:
            target_tokens = np.asarray(np.argmax(v_logits, axis=-1)).reshape(-1)
            result = dflash_linear_verify(draft_tokens, target_tokens)
        else:
            dprob = sampler_probs(draft_logits, temperature, top_k, top_p)[0]     # (b-1, V)
            tprob = sampler_probs(v_logits, temperature, top_k, top_p)[0]         # (b, V)
            result = dflash_speculative_verify(draft_tokens, dprob, tprob, rng)

        # Discard the over-speculated tail from the target cache: step appended b
        # tokens; keep prev (1) + accepted, drop the rest.
        trim = b - 1 - result.accepted
        if trim > 0:
            target.rollback(trim)

        new_tokens = result.new_tokens[: max_new_tokens - n]
        eos_cut = (next((i for i, t in enumerate(new_tokens) if t == eos_id), None)
                   if eos_id is not None else None)
        if eos_cut is not None:
            new_tokens = new_tokens[: eos_cut + 1]
        tokens.extend(new_tokens)
        n += len(new_tokens)
        if eos_cut is not None:
            break
        # Next draft context = this step's target hidden for the accepted prefix
        # ([prev, accepted drafts]) = v_hidden[:, :accepted+1].
        target_hidden = np.asarray(v_hidden)[:, : result.accepted + 1, :]

    return tokens


__all__ = [
    "DFlashConfig",
    "DFlashLayerWeights",
    "DFlashWeights",
    "DFlashVerification",
    "HiddenStateTap",
    "DraftKVCache",
    "make_rope",
    "make_sampler",
    "sampler_probs",
    "dflash_position_weights",
    "dflash_block_loss",
    "dflash_block_loss_grad",
    "target_feature_projection",
    "capture_target_hidden",
    "apple_gpu_attention_fn",
    "dflash_decoder_layer",
    "dflash_decoder_layer_cached",
    "dflash_draft_forward",
    "dflash_draft_forward_cached",
    "dflash_linear_verify",
    "dflash_speculative_verify",
    "dflash_step",
    "dflash_generate",
    "dflash_generate_cached",
]
