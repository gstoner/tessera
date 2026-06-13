"""DiffusionGemma Phase C — entropy-bound block-diffusion sampler (reference).

One denoise-step sampler over logits ``(P, vocab)`` for the P positions of a
block-diffusion canvas. The fused path is:

    softcap → temperature(step) → log-softmax → entropy → threshold-accept →
    sample → re-noise the rest → stop check

and it returns ``(tokens, accepted_mask, entropy_summary, stop_reason)`` (plus
the per-position entropy and re-noise mask). Confident (low-entropy) positions
commit their sampled token this step; the rest are re-noised (set to the mask
id) for a later step. Temperature follows a 0.8 → 0.4 schedule across steps.

Reference semantics; native kernels (a fused softcap→log-softmax→entropy→sample
MSL/CUDA kernel over the 262144 vocab) specialize later. Sampling is Gumbel-max,
deterministic in ``rng_key``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplerConfig:
    vocab_size: int = 262144
    final_logit_softcap: float = 30.0
    entropy_threshold: float = 1.0      # nats; positions below this commit
    temp_start: float = 0.8
    temp_end: float = 0.4
    num_steps: int = 48
    stability_entropy: float = 0.1      # mean entropy below this → stability stop
    eos_id: int = -1                    # < 0 disables EOS stop
    mask_id: int = 0                    # token id written to re-noised positions


@dataclass(frozen=True)
class SamplerResult:
    tokens: np.ndarray          # (P,) committed token per position (mask_id if re-noised)
    accepted_mask: np.ndarray   # (P,) bool — committed this step
    renoise_mask: np.ndarray    # (P,) bool — re-noised (== ~accepted)
    sampled: np.ndarray         # (P,) raw sampled token per position (pre-accept)
    entropy: np.ndarray         # (P,) per-position entropy (nats)
    entropy_summary: dict       # mean / max / min / frac_accepted
    stop_reason: str            # continue | all_accepted | stability | eos | max_steps
    temperature: float


def temperature_schedule(step: int, num_steps: int, t0: float, t1: float) -> float:
    """Linear temperature anneal ``t0 → t1`` over ``num_steps`` (clamped)."""
    if num_steps <= 1:
        return t1
    frac = min(max(step / (num_steps - 1), 0.0), 1.0)
    return t0 + (t1 - t0) * frac


def entropy_bound_sample(logits, *, step: int, config: SamplerConfig,
                         rng_key: int) -> SamplerResult:
    """Run one entropy-bound denoise step over ``logits`` ``(P, vocab)``."""
    from tessera import ops as _ops

    lg = np.asarray(logits._data if hasattr(logits, "_data") else logits, dtype=np.float64)
    if lg.ndim != 2:
        raise ValueError("logits must be (positions, vocab)")
    if lg.shape[1] != config.vocab_size:
        raise ValueError(f"logits vocab {lg.shape[1]} != config.vocab_size {config.vocab_size}")
    P = lg.shape[0]

    # 1. soft-cap → 2. temperature → 3. log-softmax → 4. entropy
    capped = np.asarray(_ops.softcap(lg, cap=config.final_logit_softcap), dtype=np.float64)
    temp = temperature_schedule(step, config.num_steps, config.temp_start, config.temp_end)
    scaled = capped / temp
    logp = np.asarray(_ops.log_softmax(scaled, axis=-1), dtype=np.float64)
    p = np.exp(logp)
    entropy = -(p * logp).sum(axis=-1)               # (P,) nats

    # 5. sample (Gumbel-max, deterministic in rng_key)
    gen = np.random.default_rng(rng_key)
    gumbel = gen.gumbel(size=scaled.shape)
    sampled = np.argmax(scaled + gumbel, axis=-1).astype(np.int64)

    # 6. threshold-accept confident positions; re-noise the rest
    accepted = entropy < config.entropy_threshold
    renoise = ~accepted
    tokens = np.where(accepted, sampled, np.int64(config.mask_id)).astype(np.int64)

    summary = {
        "mean": float(entropy.mean()) if P else 0.0,
        "max": float(entropy.max()) if P else 0.0,
        "min": float(entropy.min()) if P else 0.0,
        "frac_accepted": float(accepted.mean()) if P else 0.0,
    }

    # 7. stop check (priority: eos > all_accepted > stability > max_steps)
    stop = "continue"
    if config.eos_id >= 0 and bool(np.any(accepted & (sampled == config.eos_id))):
        stop = "eos"
    elif P and bool(accepted.all()):
        stop = "all_accepted"
    elif P and summary["mean"] < config.stability_entropy:
        stop = "stability"
    elif step >= config.num_steps - 1:
        stop = "max_steps"

    return SamplerResult(
        tokens=tokens, accepted_mask=accepted, renoise_mask=renoise, sampled=sampled,
        entropy=entropy, entropy_summary=summary, stop_reason=stop, temperature=temp,
    )


__all__ = ["SamplerConfig", "SamplerResult", "temperature_schedule", "entropy_bound_sample"]
