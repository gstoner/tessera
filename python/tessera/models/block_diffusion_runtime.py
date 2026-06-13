"""DiffusionGemma block-diffusion region — native execution path.

The shape-only graph + verifier in :mod:`block_diffusion` is the *contract*;
this module is the **executable lowering** of the canvas-denoiser region. The
denoiser layer is multi-head GQA attention with the canvas queries attending
bidirectionally over ``[encoder_context_KV ++ canvas_KV]`` (the same KV-injection
shape as DFlash's ``block_diffusion_attention``), followed by the grouped-SwiGLU
MoE FFN. Both compose through ``tessera.ops`` so the attention runs on the Apple
GPU ``metal_runtime`` lane via the ``attn_bias`` / ``flash_attn`` substrate and
the MoE through ``moe_swiglu_block``.

``attention_fn`` selects the attention backend: ``None`` → ``ops.flash_attn``
(eager / numpy reference / ``@jit`` traced); pass
``tessera.dflash.apple_gpu_attention_fn`` to execute the per-head attention on
Metal. The result is identical across backends (that equivalence is the test).
"""
from __future__ import annotations

import numpy as np

from .. import ops
from ..nn import functional as F
from . import moe_routing as _mr


def synthetic_layer_weights(config, *, seed: int = 0) -> dict:
    """Synthetic per-layer weights for the native denoiser: attention
    projections (Q/K/V/O), the two RMSNorm scales, and the MoE block."""
    rng = np.random.default_rng(seed)
    H = config.hidden_size
    Hq, Hkv, Dh = config.num_attention_heads, config.num_kv_heads, config.head_dim
    s = 1.0 / np.sqrt(H)
    return {
        "q_proj": (rng.standard_normal((H, Hq * Dh)) * s).astype(np.float32),
        "k_proj": (rng.standard_normal((H, Hkv * Dh)) * s).astype(np.float32),
        "v_proj": (rng.standard_normal((H, Hkv * Dh)) * s).astype(np.float32),
        "o_proj": (rng.standard_normal((Hq * Dh, H)) / np.sqrt(Hq * Dh)).astype(np.float32),
        "input_norm": (rng.standard_normal(H) * 0.02 + 1.0).astype(np.float32),
        "post_norm": (rng.standard_normal(H) * 0.02 + 1.0).astype(np.float32),
        "moe": _mr.synthetic_moe_weights(config, seed=seed + 1),
    }


def synthetic_encoder_kv(config, *, context_len: int, seed: int = 0):
    """Synthetic committed encoder/context KV ``(Hkv, S, Dh)`` per head."""
    rng = np.random.default_rng(seed)
    Hkv, Dh = config.num_kv_heads, config.head_dim
    s = 1.0 / np.sqrt(Dh)
    k = (rng.standard_normal((Hkv, context_len, Dh)) * s).astype(np.float32)
    v = (rng.standard_normal((Hkv, context_len, Dh)) * s).astype(np.float32)
    return k, v


def denoise_layer(h, ctx_k, ctx_v, w: dict, config, *, top_k: int,
                  attention_fn=None):
    """One native block-diffusion canvas-denoiser layer.

    ``h`` ``(T, H)`` is the canvas hidden; ``ctx_k`` / ``ctx_v`` ``(Hkv, S, Dh)``
    are the committed encoder/context KV. Pre-norm GQA attention (canvas queries
    over ``[ctx_KV ++ canvas_KV]``, bidirectional) + residual, then pre-norm MoE
    FFN + residual. ``attention_fn`` routes the per-head attention core onto a
    backend (e.g. Apple GPU ``metal_runtime``).
    """
    H = config.hidden_size
    Hq, Hkv, Dh = config.num_attention_heads, config.num_kv_heads, config.head_dim
    h = np.asarray(h)
    T = h.shape[0]

    xn = np.asarray(F.rms_norm(h, w["input_norm"], eps=config.rms_norm_eps))

    def to_heads(x, Wp, nh):
        return (x @ np.asarray(Wp)).reshape(T, nh, Dh).transpose(1, 0, 2)  # (nh, T, Dh)

    q = to_heads(xn, w["q_proj"], Hq)            # (Hq, T, Dh)
    canvas_k = to_heads(xn, w["k_proj"], Hkv)    # (Hkv, T, Dh)
    canvas_v = to_heads(xn, w["v_proj"], Hkv)
    # KV injection: committed encoder context KV ++ this step's canvas KV.
    K = np.concatenate([np.asarray(ctx_k), canvas_k], axis=1)   # (Hkv, S+T, Dh)
    V = np.concatenate([np.asarray(ctx_v), canvas_v], axis=1)
    if Hkv != Hq:
        rep = Hq // Hkv
        K = np.repeat(K, rep, axis=0)            # (Hq, S+T, Dh)
        V = np.repeat(V, rep, axis=0)

    attn_core = attention_fn if attention_fn is not None else ops.flash_attn
    scale = Dh ** -0.5
    # Per-head batched, bidirectional (canvas denoising sees the whole canvas).
    out = attn_core(q, K, V, scale=scale, causal=False)          # (Hq, T, Dh)
    out = np.asarray(out).transpose(1, 0, 2).reshape(T, Hq * Dh)
    h = h + (out @ np.asarray(w["o_proj"]))                       # attn residual

    pn = np.asarray(F.rms_norm(h, w["post_norm"], eps=config.rms_norm_eps))
    moe_out, _ = _mr.moe_forward(pn, **w["moe"], top_k=top_k)
    return h + np.asarray(moe_out)                               # MoE residual


def run_denoise(canvas_embed, encoder_kv, layer_weights, config, *,
                num_layers: int, top_k: int, attention_fn=None):
    """Run ``num_layers`` native denoiser layers over the canvas. ``encoder_kv``
    is ``(ctx_k, ctx_v)``; ``layer_weights`` is a list of per-layer weight dicts
    (or one dict, reused across layers). Returns the denoised canvas ``(T, H)``."""
    h = np.asarray(canvas_embed, dtype=np.float64)
    ctx_k, ctx_v = encoder_kv
    for i in range(num_layers):
        w = layer_weights[i] if isinstance(layer_weights, (list, tuple)) else layer_weights
        h = denoise_layer(h, ctx_k, ctx_v, w, config, top_k=top_k,
                          attention_fn=attention_fn)
    return h


__all__ = [
    "synthetic_layer_weights",
    "synthetic_encoder_kv",
    "denoise_layer",
    "run_denoise",
]
