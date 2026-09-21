"""
tessera_diffusion_llm/utils/__init__.py

Miscellaneous utilities shared across model variants.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return total (or trainable) parameter count."""
    params: Iterable = (
        (p for p in model.parameters() if p.requires_grad)
        if trainable_only else model.parameters()
    )
    return sum(p.numel() for p in params)


def param_summary(model: nn.Module) -> Dict[str, int]:
    """Return a dict with total, trainable, and frozen param counts."""
    total     = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
    }


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad_(True)


def tokens_to_human(n: int) -> str:
    """Format a large integer as a human-readable string (e.g. 1.2B)."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    elif n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def log_linear_schedule(step: int, warmup: int, total: int) -> float:
    """Log-linear warm-up + linear decay LR multiplier."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 1.0 - progress)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out logits outside the top-k values."""
    if k <= 0:
        return logits
    v, _ = logits.topk(k, dim=-1)
    return logits.masked_fill(logits < v[..., -1:], float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering of logits."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative prob above threshold (shift by 1 to keep first)
    remove = cumprobs - sorted_logits.softmax(dim=-1) > p
    sorted_logits[remove] = float("-inf")
    return sorted_logits.scatter(-1, sorted_idx, sorted_logits)


__all__ = [
    "count_parameters",
    "param_summary",
    "freeze_module",
    "unfreeze_module",
    "tokens_to_human",
    "log_linear_schedule",
    "top_k_filter",
    "top_p_filter",
]
