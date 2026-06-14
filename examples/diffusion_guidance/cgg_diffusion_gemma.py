"""Functional CGG example over Tessera's DiffusionGemma denoiser path.

This example is intentionally checkpoint-free: it uses small, valid
DiffusionGemma configs and synthetic weights so the whole flow is reproducible
and runnable from a source checkout:

    PYTHONPATH=python python examples/diffusion_guidance/cgg_diffusion_gemma.py

It demonstrates the CGG mechanics, not human-preference quality. The favored and
unfavored denoisers are structurally identical denoiser stacks with small
adapter-style weight delta packs applied to the same synthetic base weights.
That mirrors PO/DPO-style model variants more closely than adding a direct
preference bias to the predicted clean canvas, while keeping the example
deterministic and checkpoint-free.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from tessera.diffusion_guidance import (  # noqa: E402
    ContrastivePair,
    ContrastiveScoreGuidance,
    DenoiseOutput,
    DiffusionSchedule,
    GuidanceSafety,
    GuidedDiffusionSampler,
)
from tessera.models import DiffusionGemmaConfig, SamplerConfig, entropy_bound_sample  # noqa: E402
from tessera.models import block_diffusion_runtime as BR  # noqa: E402


@dataclass(frozen=True)
class BlockDenoiser:
    cfg: DiffusionGemmaConfig
    layer_weights: list[dict]
    encoder_kv: tuple[np.ndarray, np.ndarray]
    adapter_name: str
    adapter_scale: float
    model_id: str

    def __call__(
        self,
        x_t: np.ndarray,
        t: int,
        condition: Any,
        schedule: DiffusionSchedule,
    ) -> DenoiseOutput:
        h = BR.run_denoise(
            x_t,
            self.encoder_kv,
            self.layer_weights,
            self.cfg,
            num_layers=self.cfg.num_layers,
            top_k=self.cfg.num_experts_per_tok,
        )
        return DenoiseOutput(
            x0_pred=h,
            prediction_type="x0",
            timestep=t,
            model_id=self.model_id,
            metadata={"adapter": self.adapter_name, "adapter_scale": self.adapter_scale},
        )


def tiny_config() -> DiffusionGemmaConfig:
    return DiffusionGemmaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=48,
        num_layers=2,
        canvas_size=12,
        vocab_size=50,
    )


def _final_tokens(final_canvas: np.ndarray, w_lm: np.ndarray, *, cfg: DiffusionGemmaConfig):
    logits = final_canvas @ w_lm
    sample_cfg = SamplerConfig(
        vocab_size=cfg.vocab_size,
        num_steps=6,
        entropy_threshold=100.0,
        mask_id=0,
    )
    return entropy_bound_sample(logits, step=0, config=sample_cfg, rng_key=17)


def _unit_axis(rng: np.random.Generator, hidden: int) -> np.ndarray:
    axis = rng.standard_normal(hidden).astype(np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    return axis


def _fit_axis(axis: np.ndarray, size: int) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(-1)
    if axis.size == size:
        out = axis
    else:
        reps = int(np.ceil(size / axis.size))
        out = np.tile(axis, reps)[:size]
    out = out / max(float(np.linalg.norm(out)), 1.0e-12)
    return out.astype(np.float32)


def make_adapter_layers(
    base_layers: list[dict],
    *,
    axis: np.ndarray,
    scale: float,
    name: str,
) -> list[dict]:
    """Create a tiny synthetic adapter delta pack over base denoiser weights.

    The perturbations are rank-1 deltas, not a direct output bias:
    - attention projection/output paths receive axis-aligned low-rank deltas;
    - MoE router logits receive an axis × expert-pattern delta.

    This keeps all model variants structurally identical to the reference
    denoiser and makes CGG compose model-score differences instead of a fake
    preference head.
    """
    deltas: list[dict] = []
    for layer_idx, layer in enumerate(base_layers):
        w = copy.deepcopy(layer)
        layer_scale = float(scale) * (1.0 + 0.1 * layer_idx)
        for key, weight_scale in (("q_proj", 0.35), ("o_proj", 1.0)):
            weight = np.asarray(w[key], dtype=np.float32)
            left = _fit_axis(axis, weight.shape[0])
            right = _fit_axis(axis, weight.shape[1])
            w[key] = (weight + weight_scale * layer_scale * np.outer(left, right)).astype(np.float32)

        router = np.asarray(w["moe"]["w_router"], dtype=np.float32)
        expert_axis = np.linspace(-1.0, 1.0, router.shape[1], dtype=np.float64)
        expert_axis /= max(float(np.linalg.norm(expert_axis)), 1.0e-12)
        w["moe"]["w_router"] = (
            router + layer_scale * np.outer(_fit_axis(axis, router.shape[0]), expert_axis)
        ).astype(np.float32)
        w["_adapter"] = {"name": name, "scale": layer_scale}
        deltas.append(w)
    return deltas


def _tokens_for_step(canvas: np.ndarray, w_lm: np.ndarray, *, cfg: DiffusionGemmaConfig, step: int):
    logits = canvas @ w_lm
    sample_cfg = SamplerConfig(
        vocab_size=cfg.vocab_size,
        num_steps=6,
        entropy_threshold=100.0,
        mask_id=0,
    )
    return entropy_bound_sample(logits, step=step, config=sample_cfg, rng_key=17 + step)


def _proxy(canvas: np.ndarray, axis: np.ndarray) -> float:
    return float(np.mean(np.asarray(canvas) @ np.asarray(axis).reshape(-1)))


def run_demo(
    *,
    quality_gamma: float = 0.75,
    safety_gamma: float = 0.25,
    quality_scale: float = 0.08,
    unfavored_scale: float = -0.06,
    safety_scale: float = 0.045,
    seed: int = 123,
) -> dict[str, Any]:
    cfg = tiny_config()
    rng = np.random.default_rng(seed)
    schedule = DiffusionSchedule(alpha_bar=[0.99, 0.82, 0.64, 0.45, 0.28, 0.14])
    sampler = GuidedDiffusionSampler(schedule)

    base_layers = [BR.synthetic_layer_weights(cfg, seed=10 + i) for i in range(cfg.num_layers)]
    encoder_kv = BR.synthetic_encoder_kv(cfg, context_len=8, seed=70)
    w_lm = (rng.standard_normal((cfg.hidden_size, cfg.vocab_size)) / np.sqrt(cfg.hidden_size)).astype(np.float32)

    init_canvas = (rng.standard_normal((cfg.canvas_size, cfg.hidden_size)) * 0.2).astype(np.float32)
    axis_quality = _unit_axis(rng, cfg.hidden_size)
    axis_safety = _unit_axis(rng, cfg.hidden_size)

    quality_favored_layers = make_adapter_layers(
        base_layers, axis=axis_quality, scale=quality_scale, name="quality_favored_lora_delta"
    )
    quality_unfavored_layers = make_adapter_layers(
        base_layers, axis=axis_quality, scale=unfavored_scale, name="quality_unfavored_lora_delta"
    )
    safety_layers = make_adapter_layers(
        base_layers, axis=axis_safety, scale=safety_scale, name="safety_favored_lora_delta"
    )

    ref = BlockDenoiser(cfg, base_layers, encoder_kv, "reference", 0.0, "ref")
    favored = BlockDenoiser(
        cfg, quality_favored_layers, encoder_kv, "quality_favored_lora_delta",
        quality_scale, "favored_quality"
    )
    unfavored = BlockDenoiser(
        cfg, quality_unfavored_layers, encoder_kv, "quality_unfavored_lora_delta",
        unfavored_scale, "unfavored_quality"
    )
    safety = BlockDenoiser(
        cfg, safety_layers, encoder_kv, "safety_favored_lora_delta", safety_scale,
        "favored_safety"
    )

    guidance = ContrastiveScoreGuidance(
        (
            ContrastivePair(favored=favored, unfavored=unfavored, gamma=quality_gamma, name="quality"),
            ContrastivePair(favored=safety, unfavored="base", gamma=safety_gamma, name="safety"),
        ),
        safety=GuidanceSafety(max_delta_ratio=1.5),
    )

    unguided = sampler.sample(init_canvas, ref)
    guided = sampler.sample(init_canvas, ref, guidance=guidance)
    unguided_tokens = _final_tokens(unguided.final, w_lm, cfg=cfg)
    guided_tokens = _final_tokens(guided.final, w_lm, cfg=cfg)

    first_guidance = next(g for g in guided.guidance if g is not None)
    guided_quality_proxy = _proxy(guided.final, axis_quality)
    guided_safety_proxy = _proxy(guided.final, axis_safety)
    unguided_quality_proxy = _proxy(unguided.final, axis_quality)
    unguided_safety_proxy = _proxy(unguided.final, axis_safety)
    step_report = []
    for i, (g_step, u_step, gr) in enumerate(zip(guided.trajectory, unguided.trajectory, guided.guidance)):
        tokens = _tokens_for_step(g_step.state, w_lm, cfg=cfg, step=i)
        u_tokens = _tokens_for_step(u_step.state, w_lm, cfg=cfg, step=i)
        delta_norms = {} if gr is None else {
            name: round(float(np.linalg.norm(delta.reshape(-1))), 6)
            for name, delta in gr.deltas.items()
        }
        step_report.append({
            "timestep": int(g_step.timestep),
            "gamma": {} if gr is None else {k: round(v, 6) for k, v in gr.scales.items()},
            "delta_norms": delta_norms,
            "unguided_score_norm": round(u_step.score_norm, 6),
            "guided_score_norm": round(g_step.score_norm, 6),
            "score_norm_delta": round(g_step.score_norm - u_step.score_norm, 6),
            "canvas_l2_delta": round(float(np.linalg.norm(g_step.state - u_step.state)), 6),
            "entropy_mean": round(float(tokens.entropy_summary["mean"]), 6),
            "accepted_tokens": int(tokens.accepted_mask.sum()),
            "token_delta_count": int(np.count_nonzero(tokens.tokens != u_tokens.tokens)),
            "quality_proxy": round(_proxy(g_step.state, axis_quality), 6),
            "safety_proxy": round(_proxy(g_step.state, axis_safety), 6),
        })

    token_delta_count = int(np.count_nonzero(guided_tokens.tokens != unguided_tokens.tokens))
    unguided_entropy = float(unguided_tokens.entropy_summary["mean"])
    guided_entropy = float(guided_tokens.entropy_summary["mean"])
    return {
        "adapter_style": "rank1_attention_output_and_moe_router_deltas",
        "config": {
            "quality_gamma": quality_gamma,
            "safety_gamma": safety_gamma,
            "quality_scale": quality_scale,
            "unfavored_scale": unfavored_scale,
            "safety_scale": safety_scale,
            "seed": seed,
        },
        "canvas_shape": list(init_canvas.shape),
        "steps": len(guided.trajectory),
        "guidance_scales": dict(first_guidance.scales),
        "unguided_score_norm_last": round(unguided.trajectory[-1].score_norm, 6),
        "guided_score_norm_last": round(guided.trajectory[-1].score_norm, 6),
        "canvas_l2_delta": round(float(np.linalg.norm(guided.final - unguided.final)), 6),
        "quality_proxy": round(guided_quality_proxy, 6),
        "safety_proxy": round(guided_safety_proxy, 6),
        "guided_vs_unguided_deltas": {
            "accepted_tokens": int(guided_tokens.accepted_mask.sum() - unguided_tokens.accepted_mask.sum()),
            "canvas_l2_delta": round(float(np.linalg.norm(guided.final - unguided.final)), 6),
            "entropy_mean": round(guided_entropy - unguided_entropy, 6),
            "quality_proxy": round(guided_quality_proxy - unguided_quality_proxy, 6),
            "safety_proxy": round(guided_safety_proxy - unguided_safety_proxy, 6),
            "score_norm_last": round(guided.trajectory[-1].score_norm - unguided.trajectory[-1].score_norm, 6),
            "token_delta_count": token_delta_count,
        },
        "per_step": step_report,
        "unguided_tokens": unguided_tokens.tokens.astype(int).tolist(),
        "guided_tokens": guided_tokens.tokens.astype(int).tolist(),
        "token_delta_count": token_delta_count,
        "guided_accepted": int(guided_tokens.accepted_mask.sum()),
        "guided_entropy_mean": round(float(guided_tokens.entropy_summary["mean"]), 6),
        "unguided_entropy_mean": round(float(unguided_tokens.entropy_summary["mean"]), 6),
        "clipped": dict(first_guidance.metadata["clipped"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, help="Optional path for the CGG JSON report artifact.")
    parser.add_argument("--quality-gamma", type=float, default=0.75)
    parser.add_argument("--safety-gamma", type=float, default=0.25)
    parser.add_argument("--quality-scale", type=float, default=0.08)
    parser.add_argument("--unfavored-scale", type=float, default=-0.06)
    parser.add_argument("--safety-scale", type=float, default=0.045)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    report = run_demo(
        quality_gamma=args.quality_gamma,
        safety_gamma=args.safety_gamma,
        quality_scale=args.quality_scale,
        unfavored_scale=args.unfavored_scale,
        safety_scale=args.safety_scale,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
