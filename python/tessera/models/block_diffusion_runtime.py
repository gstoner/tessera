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
from .block_diffusion import BlockDiffusionStepResult
from .sampler import SamplerConfig, SamplerResult, entropy_bound_sample


# ── Apple GPU dispatch helpers (eager metal_runtime routing) ────────────────

def _apple_gpu_matmul(a, b):
    """Rank-2/3 matmul on the Apple GPU MPS lane (f32). Falls back to numpy
    off-Darwin / outside the envelope (the dispatcher handles that)."""
    from .. import runtime as _rt
    return _rt._apple_gpu_dispatch_matmul(
        "tessera.matmul",
        [np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)], np)


def _apple_gpu_moe_swiglu(x, w_gate, w_up, w_down, group_sizes, *, kind="contiguous"):
    """Grouped-SwiGLU MoE expert FFN on the Apple GPU lane (f32). Same signature
    as ``ops.moe_swiglu_block`` so it slots into ``moe_forward(swiglu_fn=...)``."""
    from .. import runtime as _rt
    ops_ = [np.asarray(x, dtype=np.float32), np.asarray(w_gate, dtype=np.float32),
            np.asarray(w_up, dtype=np.float32), np.asarray(w_down, dtype=np.float32),
            np.asarray(group_sizes)]
    return _rt._apple_gpu_dispatch_moe_swiglu_block(ops_, {"kind": kind}, np)


def _attention_backend(backend, attention_fn):
    if attention_fn is not None:
        return attention_fn
    if backend == "apple_gpu":
        from ..dflash import apple_gpu_attention_fn
        return apple_gpu_attention_fn
    return None  # ops.flash_attn (numpy / eager)


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
                  attention_fn=None, backend: str = "numpy"):
    """One native block-diffusion canvas-denoiser layer.

    ``h`` ``(T, H)`` is the canvas hidden; ``ctx_k`` / ``ctx_v`` ``(Hkv, S, Dh)``
    are the committed encoder/context KV. Pre-norm GQA attention (canvas queries
    over ``[ctx_KV ++ canvas_KV]``, bidirectional) + residual, then pre-norm MoE
    FFN + residual.

    ``backend="apple_gpu"`` runs the *whole* layer on the Apple GPU lane: the
    per-head attention via the ``attn_bias``/``flash_attn`` symbols and the
    grouped-SwiGLU MoE via ``moe_swiglu_block`` (both ``metal_runtime``).
    ``attention_fn`` overrides just the attention backend.
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

    attn_core = _attention_backend(backend, attention_fn) or ops.flash_attn
    scale = Dh ** -0.5
    # Per-head batched, bidirectional (canvas denoising sees the whole canvas).
    out = attn_core(q, K, V, scale=scale, causal=False)          # (Hq, T, Dh)
    out = np.asarray(out).transpose(1, 0, 2).reshape(T, Hq * Dh)
    h = h + (out @ np.asarray(w["o_proj"]))                       # attn residual

    pn = np.asarray(F.rms_norm(h, w["post_norm"], eps=config.rms_norm_eps))
    swiglu_fn = _apple_gpu_moe_swiglu if backend == "apple_gpu" else None
    moe_out, _ = _mr.moe_forward(pn, **w["moe"], top_k=top_k, swiglu_fn=swiglu_fn)
    return h + np.asarray(moe_out)                               # MoE residual


def run_denoise(canvas_embed, encoder_kv, layer_weights, config, *,
                num_layers: int, top_k: int, attention_fn=None, backend: str = "numpy"):
    """Run ``num_layers`` native denoiser layers over the canvas. ``encoder_kv``
    is ``(ctx_k, ctx_v)``; ``layer_weights`` is a list of per-layer weight dicts
    (or one dict, reused across layers). Returns the denoised canvas ``(T, H)``.
    ``backend="apple_gpu"`` runs attention + MoE on the Metal lane."""
    h = np.asarray(canvas_embed, dtype=np.float64)
    ctx_k, ctx_v = encoder_kv
    for i in range(num_layers):
        w = layer_weights[i] if isinstance(layer_weights, (list, tuple)) else layer_weights
        h = denoise_layer(h, ctx_k, ctx_v, w, config, top_k=top_k,
                          attention_fn=attention_fn, backend=backend)
    return h


def execute_block_diffusion_step(canvas_embed, encoder_kv, layer_weights, w_lm,
                                 config, *, step: int, sampler_config: SamplerConfig,
                                 num_denoise_layers: int, rng_key: int, top_k: int,
                                 attention_fn=None, backend: str = "numpy"
                                 ) -> BlockDiffusionStepResult:
    """Native execution of one block-diffusion step (the faithful multi-head
    counterpart to :func:`block_diffusion.run_block_diffusion_step`).

    Runs ``num_denoise_layers`` native denoiser layers over the canvas, projects
    the LM head, then the Phase-C entropy-bound sampler. With
    ``backend="apple_gpu"`` the *whole* step's heavy compute — per-head attention,
    grouped MoE, and the LM-head matmul — lands on the Apple GPU ``metal_runtime``
    lane. ``attention_fn`` overrides just the attention backend.
    """
    h = run_denoise(canvas_embed, encoder_kv, layer_weights, config,
                    num_layers=num_denoise_layers, top_k=top_k,
                    attention_fn=attention_fn, backend=backend)
    lm_matmul = _apple_gpu_matmul if backend == "apple_gpu" else ops.gemm
    logits = np.asarray(lm_matmul(np.asarray(h), np.asarray(w_lm)))   # LM head (T, vocab)
    res: SamplerResult = entropy_bound_sample(
        logits, step=step, config=sampler_config, rng_key=rng_key)
    return BlockDiffusionStepResult(
        tokens=res.tokens, accepted_mask=res.accepted_mask,
        renoise_mask=res.renoise_mask, sampled=res.sampled, entropy=res.entropy,
        entropy_summary=res.entropy_summary, stop_reason=res.stop_reason,
        committed=int(res.accepted_mask.sum()))


# ── End-to-end native decode loop ───────────────────────────────────────────

def synthetic_decoder_weights(config, *, num_denoise_layers: int, seed: int = 0) -> dict:
    """Faithful weights for the native decoder: per-layer denoiser weights, the
    LM head, an embedding table, a mask embedding, and the committed-context K/V
    projections (``H -> num_kv_heads * head_dim``) used for KV promotion."""
    rng = np.random.default_rng(seed)
    H, V = config.hidden_size, config.vocab_size
    Hkv, Dh = config.num_kv_heads, config.head_dim
    s = 1.0 / np.sqrt(H)
    return {
        "layers": [synthetic_layer_weights(config, seed=seed + 1 + i)
                   for i in range(num_denoise_layers)],
        "w_lm": (rng.standard_normal((H, V)) / np.sqrt(H)).astype(np.float32),
        "embed_table": (rng.standard_normal((V, H)) * s).astype(np.float32),
        "mask_embedding": (rng.standard_normal(H) * s).astype(np.float32),
        "ctx_k_proj": (rng.standard_normal((H, Hkv * Dh)) * s).astype(np.float32),
        "ctx_v_proj": (rng.standard_normal((H, Hkv * Dh)) * s).astype(np.float32),
    }


class NativeBlockDiffusionDecoder:
    """End-to-end native block-diffusion decoder.

    Orchestrates the commit / freeze / re-noise / KV-promotion loop over the
    *native* :func:`execute_block_diffusion_step` (multi-head GQA denoiser + MoE
    + LM head + entropy sampler). ``backend="apple_gpu"`` runs every step on the
    Metal lane. Committed blocks are projected (``ctx_k_proj`` / ``ctx_v_proj``)
    into per-head encoder KV and promoted into the context for later blocks —
    the canvas denoiser attends bidirectionally over the canvas and causally over
    the committed context.
    """

    def __init__(self, config, weights: dict, *, num_denoise_layers: int,
                 max_steps: int, sampler_config, top_k: int,
                 backend: str = "numpy", max_context_blocks: int = 4):
        if max_steps > 48:
            raise ValueError("max_steps must be <= 48 (block-diffusion budget)")
        self.cfg = config
        self.w = weights
        self.num_layers = num_denoise_layers
        self.max_steps = max_steps
        self.sampler = sampler_config
        self.top_k = top_k
        self.backend = backend
        self.embed = np.asarray(weights["embed_table"], dtype=np.float64)
        self.mask_emb = np.asarray(weights["mask_embedding"], dtype=np.float64)
        # Per-head committed-context KV cache (num_kv_heads).
        from ..cache import KVCacheHandle
        self.kv = KVCacheHandle(
            num_heads=config.num_kv_heads, head_dim=config.head_dim,
            max_seq=config.canvas_size * max_context_blocks, dtype="fp32")
        self.blocks: list = []

    @property
    def context_len(self) -> int:
        return self.kv.current_seq

    def _embed(self, committed, frozen):
        e = self.embed[committed].copy()
        e[~frozen] = self.mask_emb
        return e

    def _context_kv(self, *, sliding: bool):
        """Committed context KV as ``(Hkv, S, Dh)`` (empty when no context yet)."""
        Hkv, Dh = self.cfg.num_kv_heads, self.cfg.head_dim
        n = self.kv.current_seq
        if n == 0:
            return np.zeros((Hkv, 0, Dh), np.float64), np.zeros((Hkv, 0, Dh), np.float64)
        start = max(0, n - self.cfg.sliding_window) if sliding else 0
        k, v = self.kv.read(start, n)              # (s, Hkv, Dh)
        return (np.asarray(k, np.float64).transpose(1, 0, 2),
                np.asarray(v, np.float64).transpose(1, 0, 2))

    def decode_block(self, *, rng_key: int, sliding: bool = False):
        from .decode import BlockDecodeResult
        cfg = self.cfg
        canvas = cfg.canvas_size
        committed = np.full(canvas, self.sampler.mask_id, dtype=np.int64)
        frozen = np.zeros(canvas, dtype=bool)
        progress: list = []
        stop, steps = "max_steps", 0

        for step in range(self.max_steps):
            steps = step + 1
            embeds = self._embed(committed, frozen)
            enc = self._context_kv(sliding=sliding)
            res = execute_block_diffusion_step(
                embeds, enc, self.w["layers"], self.w["w_lm"], cfg, step=step,
                sampler_config=self.sampler, num_denoise_layers=self.num_layers,
                rng_key=rng_key + step, top_k=self.top_k, backend=self.backend)
            newly = res.accepted_mask & ~frozen
            if newly.any():
                committed[newly] = res.tokens[newly]
            else:
                unfrozen = np.flatnonzero(~frozen)
                pick = int(unfrozen[int(np.argmin(res.entropy[unfrozen]))])
                committed[pick] = res.sampled[pick]
                newly = np.zeros(canvas, dtype=bool)
                newly[pick] = True
            frozen |= newly
            progress.append(int(frozen.sum()))
            if frozen.all():
                stop = "all_committed"
                break

        # KV promotion: project the committed block to per-head encoder KV.
        final = self._embed(committed, np.ones(canvas, dtype=bool))   # (canvas, H)
        Hkv, Dh = cfg.num_kv_heads, cfg.head_dim
        ck = (final @ np.asarray(self.w["ctx_k_proj"], np.float64)).reshape(canvas, Hkv, Dh)
        cv = (final @ np.asarray(self.w["ctx_v_proj"], np.float64)).reshape(canvas, Hkv, Dh)
        self.kv.append(ck.astype(np.float32), cv.astype(np.float32))
        self.blocks.append(committed.copy())

        return BlockDecodeResult(
            tokens=committed, steps=steps, stop_reason=stop,
            committed=int(frozen.sum()), progress=tuple(progress))


__all__ = [
    "synthetic_layer_weights",
    "synthetic_encoder_kv",
    "synthetic_decoder_weights",
    "denoise_layer",
    "run_denoise",
    "execute_block_diffusion_step",
    "NativeBlockDiffusionDecoder",
]
