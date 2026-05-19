"""Helpers in the spirit of ``torch.nn.utils``.

Operates on iterables of :class:`tessera.nn.Parameter` — typically
``module.parameters()``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .module import Parameter


def clip_grad_norm_(
    parameters: Iterable[Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clip ``.grad`` of every parameter in-place so the total norm ≤ ``max_norm``.

    Returns the *pre-clip* total norm so callers can log it.

    - Parameters with ``.grad is None`` are skipped (matches torch).
    - ``norm_type=2`` (default) uses L2; ``norm_type=float('inf')`` uses
      max-abs.
    - Multiplier is computed once and applied to every grad — preserves the
      relative direction of the gradient vector, only scales magnitude.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive; got {max_norm}")

    # Capture each non-None grad directly so mypy doesn't have to
    # re-narrow ``p.grad`` inside every loop body below.
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    if norm_type == float("inf"):
        total_norm = max(np.abs(g.numpy()).max() for g in grads)
    else:
        total_norm = float(
            np.power(
                sum(np.power(np.abs(g.numpy()), norm_type).sum() for g in grads),
                1.0 / norm_type,
            )
        )

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for g in grads:
            # Write through the DistributedArray buffer in place
            g._data[...] = g._data * scale

    return total_norm


__all__ = ["clip_grad_norm_"]
