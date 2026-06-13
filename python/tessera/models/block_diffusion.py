"""DiffusionGemma Phase D — block-diffusion step (compiler-visible region).

A first-class block-diffusion *step* contract plus a reference runner. One step:

  1. encoder KV read     — read the committed (causal) encoder/context KV;
  2. canvas denoise      — a bidirectional (causal=False) denoiser over the
                           256-token canvas, N transformer layers;
  3. logits (lm_head);
  4. entropy stats + entropy-bound sample (Phase C);
  5. selected token updates + re-noise mask;
  6. stop flags.

``build_block_diffusion_step`` returns the compiler-visible region — an ordered
list of sub-regions, embedding the canvas-denoiser text-block graphs (Phase A,
causal=False) so lowering sees the step as one structured op, distinct from the
causal encoder read. ``run_block_diffusion_step`` is a **reference runner only**
(NumPy); production lowering executes the region natively.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diffusion_gemma import (
    DiffusionGemmaConfig,
    GraphNode,
    TextBlockGraph,
    build_lm_head,
    build_text_block,
)
from .sampler import SamplerConfig, SamplerResult, entropy_bound_sample
from . import moe_routing as _mr


# ── Compiler-visible step region ─────────────────────────────────────────────

@dataclass(frozen=True)
class BlockDiffusionStepGraph:
    """The block-diffusion step as an ordered region list.

    ``regions`` is the step's sub-region sequence (name + attrs). ``denoiser``
    holds the per-layer canvas text-block graphs (causal=False = bidirectional);
    ``encoder_read`` records the causal encoder-KV read; ``lm_head`` is the vocab
    projection node.
    """

    regions: tuple[tuple, ...]              # (name, attrs) in execution order
    denoiser: tuple[TextBlockGraph, ...]    # bidirectional canvas layers
    encoder_read: GraphNode
    lm_head: GraphNode
    config: DiffusionGemmaConfig
    num_denoise_layers: int

    def region_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.regions)


def build_block_diffusion_step(
    config: DiffusionGemmaConfig,
    *,
    num_denoise_layers: int | None = None,
) -> BlockDiffusionStepGraph:
    """Build the compiler-visible block-diffusion step region for ``config``."""
    L = num_denoise_layers if num_denoise_layers is not None else config.num_layers
    if L <= 0:
        raise ValueError("num_denoise_layers must be positive")
    H = config.hidden_size

    # Encoder KV read — causal (committed context), single read per step.
    encoder_read = GraphNode(
        op="encoder_kv_read",
        inputs=((("Ctx"), H),),
        output=((("Ctx"), H),),
        attrs={"causal": True, "source": "committed_canvases",
               "sliding_window": config.sliding_window},
    )
    # Canvas denoiser — N bidirectional text-block layers over the canvas.
    denoiser = tuple(
        build_text_block(config, layer_index=i, causal=False) for i in range(L)
    )
    lm_head = build_lm_head(config)

    regions: tuple[tuple, ...] = (
        ("encoder_kv_read", {"causal": True}),
        ("canvas_denoise", {"layers": L, "causal": False,
                            "canvas_size": config.canvas_size}),
        ("lm_head", {"vocab_size": config.vocab_size}),
        ("entropy_stats", {}),
        ("entropy_sample", {"final_logit_softcap": config.final_logit_softcap}),
        ("token_update", {}),
        ("renoise_mask", {}),
        ("stop_flags", {"eos": True, "stability": True, "all_accepted": True}),
    )
    return BlockDiffusionStepGraph(
        regions=regions, denoiser=denoiser, encoder_read=encoder_read,
        lm_head=lm_head, config=config, num_denoise_layers=L,
    )


def verify_block_diffusion_step(graph: BlockDiffusionStepGraph) -> None:
    """Contract checks: encoder read is causal; every canvas-denoiser layer is
    bidirectional; lm_head projects to vocab."""
    from .diffusion_gemma import DiffusionGemmaDimError, verify_lm_head
    if graph.encoder_read.attrs.get("causal") is not True:
        raise DiffusionGemmaDimError("encoder KV read must be causal")
    for g in graph.denoiser:
        if g.causal is not False:
            raise DiffusionGemmaDimError("canvas denoiser layers must be bidirectional (causal=False)")
        if g.find("attention").attrs["causal"] is not False:
            raise DiffusionGemmaDimError("canvas attention must be bidirectional")
    verify_lm_head(graph.lm_head, graph.config)


# ── Reference runner (NumPy, reference-only) ─────────────────────────────────

@dataclass(frozen=True)
class BlockDiffusionStepResult:
    tokens: np.ndarray          # (canvas,) committed token per position
    accepted_mask: np.ndarray   # (canvas,) bool
    renoise_mask: np.ndarray    # (canvas,) bool
    sampled: np.ndarray         # (canvas,) raw sampled token per position (pre-accept)
    entropy: np.ndarray         # (canvas,) per-position entropy (nats)
    entropy_summary: dict
    stop_reason: str
    committed: int              # number of positions committed this step


def _rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)


def _attention(q, k, v, *, causal):
    # q (P, D), k/v (S, D). Bidirectional unless causal (only for square self-attn).
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = (q @ k.T) * scale
    if causal:
        P, S = s.shape
        mask = np.triu(np.ones((P, S), dtype=bool), k=1)
        s = np.where(mask, -1e30, s)
    s = s - s.max(axis=-1, keepdims=True)
    w = np.exp(s)
    w /= w.sum(axis=-1, keepdims=True)
    return w @ v


def run_block_diffusion_step(
    canvas_embed,
    encoder_kv,
    weights: dict,
    *,
    step: int,
    sampler_config: SamplerConfig,
    num_denoise_layers: int,
    rng_key: int,
    top_k: int,
    causal_canvas: bool = False,
):
    """Execute ONE block-diffusion step (reference). ``canvas_embed`` (canvas, H);
    ``encoder_kv`` = (K_ctx, V_ctx) committed context; ``weights`` carries the
    per-layer attention/MoE weights + ``w_lm`` (H, vocab). Returns a
    :class:`BlockDiffusionStepResult`."""
    h = np.asarray(canvas_embed, dtype=np.float64)
    K_ctx, V_ctx = (np.asarray(t, dtype=np.float64) for t in encoder_kv)

    for _ in range(num_denoise_layers):
        # bidirectional self-attention over the canvas + cross-read of encoder KV
        normed = _rmsnorm(h)
        kv = np.concatenate([K_ctx, normed], axis=0)
        vv = np.concatenate([V_ctx, normed], axis=0)
        attn = _attention(normed, kv, vv, causal=causal_canvas)
        h = h + attn
        # MoE FFN (Phase B pipeline)
        normed2 = _rmsnorm(h)
        moe_out, _ = _mr.moe_forward(normed2, **weights["moe"], top_k=top_k)
        h = h + moe_out

    logits = h @ np.asarray(weights["w_lm"], dtype=np.float64)   # (canvas, vocab)
    res: SamplerResult = entropy_bound_sample(
        logits, step=step, config=sampler_config, rng_key=rng_key)
    return BlockDiffusionStepResult(
        tokens=res.tokens, accepted_mask=res.accepted_mask,
        renoise_mask=res.renoise_mask, sampled=res.sampled, entropy=res.entropy,
        entropy_summary=res.entropy_summary,
        stop_reason=res.stop_reason, committed=int(res.accepted_mask.sum()),
    )


__all__ = [
    "BlockDiffusionStepGraph",
    "BlockDiffusionStepResult",
    "build_block_diffusion_step",
    "verify_block_diffusion_step",
    "run_block_diffusion_step",
]
