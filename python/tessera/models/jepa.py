"""JEPA multimodal latent-prediction contract.

JEPA is not a decoder-only VLM variant. Its compiler contract is context/target
masking, stop-gradient target encoders, EMA state updates, and prediction in a
continuous latent space. This module provides a small executable reference
contract for those pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


class JEPAContractError(ValueError):
    """Raised when a JEPA shape/state contract is invalid."""


@dataclass(frozen=True)
class JEPAConfig:
    input_dim: int
    latent_dim: int
    predictor_hidden_size: int
    mask_block_size: int = 2
    tube_size: int = 2
    mask_ratio: float = 0.4
    modalities: tuple[str, ...] = ("image", "video", "text")
    decoder_vocab_size: int | None = None


@dataclass(frozen=True)
class JEPAWeights:
    context_proj: np.ndarray
    target_proj: np.ndarray
    predictor_w1: np.ndarray
    predictor_b1: np.ndarray
    predictor_w2: np.ndarray
    predictor_b2: np.ndarray


@dataclass(frozen=True)
class JEPAStepResult:
    context_latents: np.ndarray
    target_latents: np.ndarray
    predictions: np.ndarray
    loss: float
    mask: np.ndarray


@dataclass(frozen=True)
class SelectiveDecodeResult:
    logits: np.ndarray
    decode_mask: np.ndarray
    decoded_token_ids: np.ndarray


def verify_config(cfg: JEPAConfig) -> None:
    if cfg.input_dim <= 0 or cfg.latent_dim <= 0:
        raise JEPAContractError("input_dim and latent_dim must be positive")
    if cfg.predictor_hidden_size <= 0:
        raise JEPAContractError("predictor_hidden_size must be positive")
    if cfg.mask_block_size <= 0 or cfg.tube_size <= 0:
        raise JEPAContractError("mask_block_size and tube_size must be positive")
    if not 0.0 < cfg.mask_ratio < 1.0:
        raise JEPAContractError("mask_ratio must be in (0, 1)")
    if not cfg.modalities:
        raise JEPAContractError("modalities must be non-empty")


def synthetic_weights(cfg: JEPAConfig, *, seed: int = 0) -> JEPAWeights:
    verify_config(cfg)
    rng = np.random.default_rng(seed)

    def n(*shape, sc=1.0):
        return (rng.standard_normal(shape) * sc).astype(np.float64)

    return JEPAWeights(
        context_proj=n(cfg.input_dim, cfg.latent_dim, sc=1.0 / np.sqrt(cfg.input_dim)),
        target_proj=n(cfg.input_dim, cfg.latent_dim, sc=1.0 / np.sqrt(cfg.input_dim)),
        predictor_w1=n(cfg.latent_dim, cfg.predictor_hidden_size, sc=1.0 / np.sqrt(cfg.latent_dim)),
        predictor_b1=np.zeros(cfg.predictor_hidden_size, dtype=np.float64),
        predictor_w2=n(cfg.predictor_hidden_size, cfg.latent_dim, sc=1.0 / np.sqrt(cfg.predictor_hidden_size)),
        predictor_b2=np.zeros(cfg.latent_dim, dtype=np.float64),
    )


def mask_blocks_2d(shape, *, block_size: int, mask_ratio: float, seed: int = 0) -> np.ndarray:
    h, w = (int(v) for v in tuple(shape)[-2:])
    if h <= 0 or w <= 0 or block_size <= 0:
        raise JEPAContractError("2D mask shape and block_size must be positive")
    bh = max(1, (h + block_size - 1) // block_size)
    bw = max(1, (w + block_size - 1) // block_size)
    return _ratio_mask((bh, bw), mask_ratio=mask_ratio, seed=seed)


def mask_tubes_3d(shape, *, tube_size: int, mask_ratio: float, seed: int = 0) -> np.ndarray:
    t, h, w = (int(v) for v in tuple(shape)[-3:])
    if t <= 0 or h <= 0 or w <= 0 or tube_size <= 0:
        raise JEPAContractError("3D mask shape and tube_size must be positive")
    tt = max(1, (t + tube_size - 1) // tube_size)
    return _ratio_mask((tt, h, w), mask_ratio=mask_ratio, seed=seed)


def gather_context(tokens, mask) -> np.ndarray:
    x, m = _align_tokens_and_mask(tokens, mask)
    return x[~m]


def gather_targets(tokens, mask) -> np.ndarray:
    x, m = _align_tokens_and_mask(tokens, mask)
    return x[m]


def encode_context(tokens, weights: JEPAWeights) -> np.ndarray:
    x = np.asarray(tokens, dtype=np.float64)
    if x.shape[-1] != weights.context_proj.shape[0]:
        raise JEPAContractError("context token dim does not match context_proj")
    return x @ weights.context_proj


def encode_target(tokens, weights: JEPAWeights) -> np.ndarray:
    x = np.asarray(tokens, dtype=np.float64)
    if x.shape[-1] != weights.target_proj.shape[0]:
        raise JEPAContractError("target token dim does not match target_proj")
    return stop_gradient(x @ weights.target_proj)


def latent_predict(context_latents, target_count: int, weights: JEPAWeights) -> np.ndarray:
    ctx = np.asarray(context_latents, dtype=np.float64)
    if ctx.ndim != 2 or ctx.shape[-1] != weights.predictor_w1.shape[0]:
        raise JEPAContractError("context_latents must be rank-2 with latent_dim columns")
    if target_count <= 0:
        raise JEPAContractError("target_count must be positive")
    summary = ctx.mean(axis=0, keepdims=True)
    repeated = np.repeat(summary, target_count, axis=0)
    hidden = np.tanh(repeated @ weights.predictor_w1 + weights.predictor_b1)
    return hidden @ weights.predictor_w2 + weights.predictor_b2


def jepa_l2_loss(predictions, targets) -> float:
    pred = np.asarray(predictions, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise JEPAContractError(f"prediction shape {pred.shape} != target shape {tgt.shape}")
    diff = pred - tgt
    return float(np.mean(diff * diff))


def run_jepa_step(tokens, mask, weights: JEPAWeights) -> JEPAStepResult:
    context_tokens = gather_context(tokens, mask)
    target_tokens = gather_targets(tokens, mask)
    if context_tokens.shape[0] == 0 or target_tokens.shape[0] == 0:
        raise JEPAContractError("JEPA step requires at least one context and one target token")
    context_latents = encode_context(context_tokens, weights)
    target_latents = encode_target(target_tokens, weights)
    predictions = latent_predict(context_latents, target_latents.shape[0], weights)
    return JEPAStepResult(
        context_latents=context_latents,
        target_latents=target_latents,
        predictions=predictions,
        loss=jepa_l2_loss(predictions, target_latents),
        mask=np.asarray(mask, dtype=bool),
    )


def encode_multimodal_latents(
    modality_tokens: Mapping[str, np.ndarray],
    *,
    cfg: JEPAConfig,
    weights: JEPAWeights,
) -> dict[str, np.ndarray]:
    verify_config(cfg)
    out: dict[str, np.ndarray] = {}
    for modality, tokens in modality_tokens.items():
        if modality not in cfg.modalities:
            raise JEPAContractError(f"unsupported modality {modality!r}")
        out[modality] = encode_context(tokens, weights)
    return out


def stop_gradient(x) -> np.ndarray:
    return np.asarray(x).copy()


def ema_update(target, source, *, decay: float):
    if not 0.0 <= decay <= 1.0:
        raise JEPAContractError("decay must be in [0, 1]")
    if isinstance(target, Mapping) and isinstance(source, Mapping):
        if set(target) != set(source):
            raise JEPAContractError("EMA mapping keys must match")
        return {key: ema_update(target[key], source[key], decay=decay) for key in target}
    tgt = np.asarray(target, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    if tgt.shape != src.shape:
        raise JEPAContractError(f"EMA shape mismatch {tgt.shape} != {src.shape}")
    return decay * tgt + (1.0 - decay) * src


def selective_decode(
    latents,
    decoder_weight,
    decoder_bias=None,
    *,
    scores=None,
    threshold: float = 0.0,
) -> SelectiveDecodeResult:
    """Decode only latent rows whose score crosses a caller-owned threshold."""
    z = np.asarray(latents, dtype=np.float64)
    weight = np.asarray(decoder_weight, dtype=np.float64)
    if z.ndim != 2:
        raise JEPAContractError("latents must be rank-2")
    if weight.shape[0] != z.shape[-1]:
        raise JEPAContractError("decoder_weight first dimension must equal latent_dim")
    if scores is None:
        row_scores = np.linalg.norm(z, axis=-1)
    else:
        row_scores = np.asarray(scores, dtype=np.float64)
        if row_scores.shape != (z.shape[0],):
            raise JEPAContractError("scores must have one value per latent row")
    mask = row_scores >= threshold
    logits = z[mask] @ weight
    if decoder_bias is not None:
        logits = logits + np.asarray(decoder_bias, dtype=np.float64)
    decoded = np.argmax(logits, axis=-1).astype(np.int64) if logits.size else np.zeros((0,), dtype=np.int64)
    return SelectiveDecodeResult(logits=logits, decode_mask=mask, decoded_token_ids=decoded)


def _ratio_mask(shape: tuple[int, ...], *, mask_ratio: float, seed: int) -> np.ndarray:
    if not 0.0 < mask_ratio < 1.0:
        raise JEPAContractError("mask_ratio must be in (0, 1)")
    total = int(np.prod(shape))
    target = min(total - 1, max(1, int(round(total * mask_ratio))))
    rng = np.random.default_rng(seed)
    flat = np.zeros(total, dtype=bool)
    flat[rng.permutation(total)[:target]] = True
    return flat.reshape(shape)


def _align_tokens_and_mask(tokens, mask) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(tokens, dtype=np.float64)
    if x.ndim != 2:
        raise JEPAContractError("tokens must be rank-2 (tokens, dim)")
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size != x.shape[0]:
        raise JEPAContractError(f"mask length {m.size} != token count {x.shape[0]}")
    return x, m


__all__ = [
    "JEPAConfig",
    "JEPAContractError",
    "JEPAWeights",
    "JEPAStepResult",
    "SelectiveDecodeResult",
    "ema_update",
    "encode_context",
    "encode_multimodal_latents",
    "encode_target",
    "gather_context",
    "gather_targets",
    "jepa_l2_loss",
    "latent_predict",
    "mask_blocks_2d",
    "mask_tubes_3d",
    "run_jepa_step",
    "selective_decode",
    "stop_gradient",
    "synthetic_weights",
    "verify_config",
]
