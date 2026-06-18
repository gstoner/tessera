"""Scaled GLM-5.2 serving-pressure benchmark.

This is a CPU reference benchmark for the combined system contract: DSA
IndexShare, MLA/KV cache sizing, MTP rejection sampling, and long-context
scheduling.  The full 1M-context plan is shape-only; the scaled path is runnable
in unit tests and local CI.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from tessera.losses import mtp_e2e_tv_loss
from tessera.models import glm5
from tessera.models.moe_transformer import shared_topk_index_groups
from tessera.speculative import (
    SpeculativeSamplingConfig,
    rejection_verify_chain,
    sample_draft_chain,
)


def _kv_bytes(cfg, *, context_length: int) -> int:
    elem = 1 if cfg.rollout_kv_dtype == "fp8" else 2
    if cfg.attn_kind == "mla":
        per_token = cfg.kv_lora_rank + cfg.rope_head_dim
    else:
        per_token = cfg.num_kv_heads * (cfg.qk_per_head_dim + cfg.value_per_head_dim)
    return int(cfg.num_layers * context_length * per_token * elem)


def _materialized_verify_bytes(cfg, *, context_length: int) -> int:
    elem = 2
    per_token = cfg.num_attention_heads * (cfg.qk_per_head_dim + cfg.value_per_head_dim)
    return int(cfg.num_layers * context_length * per_token * elem)


def plan_full_glm52_serving_pressure() -> dict[str, Any]:
    """Shape-only full-scale 1M-context planning numbers."""
    cfg = glm5.glm52_config()
    groups = shared_topk_index_groups(cfg)
    return {
        "model": cfg.name,
        "context_length": cfg.context_length,
        "layers": cfg.num_layers,
        "index_groups": len(groups),
        "indexer_calls_saved_per_token": sum(len(g.consumer_layers) for g in groups),
        "index_topk": cfg.index_topk,
        "mtp_steps": cfg.mtp_num_steps,
        "kv_bytes_per_request": _kv_bytes(cfg, context_length=cfg.context_length),
        "materialized_verify_bytes_per_request": _materialized_verify_bytes(
            cfg, context_length=cfg.context_length),
        "storage_policy": groups[0].storage_policy if groups else "",
    }


def run_scaled_glm52_serving_pressure(*, tokens: int = 128, seed: int = 0) -> dict[str, Any]:
    """Run the scaled benchmark and return JSON-serializable metrics."""
    cfg = glm5.scaled_config()
    groups = shared_topk_index_groups(cfg)
    rng = np.random.default_rng(seed)
    gamma = cfg.mtp_num_steps
    vocab = min(cfg.vocab_size, 128)
    start = time.perf_counter()

    accepted_lengths: list[int] = []
    tv_values: list[float] = []
    entropy_values: list[float] = []
    for _ in range(tokens):
        target_logits = rng.standard_normal((gamma + 1, vocab))
        draft_logits = target_logits[:gamma] + 0.35 * rng.standard_normal((gamma, vocab))
        sample = sample_draft_chain(
            draft_logits,
            config=SpeculativeSamplingConfig(temperature=0.8),
            rng=rng,
        )
        target_probs = np.exp(target_logits - target_logits.max(axis=-1, keepdims=True))
        target_probs = target_probs / target_probs.sum(axis=-1, keepdims=True)
        res = rejection_verify_chain(sample.tokens, sample.probs, target_probs, rng=rng)
        accepted_lengths.append(res.accepted)

        loss, metrics = mtp_e2e_tv_loss(
            target_logits[None, None, :gamma, :],
            draft_logits[None, None, :, :],
            return_metrics=True,
        )
        tv_values.append(float(np.mean(metrics["per_step_tv"])))
        entropy_values.append(float(np.mean(metrics["target_entropy"])))
        del loss

    elapsed = max(time.perf_counter() - start, 1e-9)
    return {
        "model": cfg.name,
        "tokens": tokens,
        "mtp_steps": gamma,
        "mean_accepted_length": float(np.mean(accepted_lengths)) if accepted_lengths else 0.0,
        "accepted_length_by_step": [
            float(np.mean([a > step for a in accepted_lengths])) for step in range(gamma)
        ],
        "mean_tv": float(np.mean(tv_values)) if tv_values else 0.0,
        "mean_target_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "index_groups": len(groups),
        "indexer_calls_saved_per_token": sum(len(g.consumer_layers) for g in groups),
        "kv_bytes_per_request": _kv_bytes(cfg, context_length=cfg.context_length),
        "materialized_verify_bytes_per_request": _materialized_verify_bytes(
            cfg, context_length=cfg.context_length),
        "cache_hit_ratio": (
            sum(len(g.consumer_layers) for g in groups)
            / max(cfg.num_layers, 1)
        ),
        "tokens_per_second": float(tokens / elapsed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-plan", action="store_true")
    args = ap.parse_args()
    out = plan_full_glm52_serving_pressure() if args.full_plan else run_scaled_glm52_serving_pressure(
        tokens=args.tokens, seed=args.seed)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
