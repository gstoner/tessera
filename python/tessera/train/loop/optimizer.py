"""Tier-2 optimizer wiring: real gradients (autodiff tape) + real AdamW update.

``adamw_step`` is the model-agnostic training primitive the RL loop builds on.
It runs the reverse-mode tape over a traced loss closure, collects per-parameter
gradients, applies ``tessera.optim.adamw``, and writes the updated values back
into the module's Parameters. No torch, no facade — gradients come from
``tessera.autodiff`` and the update from ``tessera.optim``.

Requirement (the v1-tape contract): ``loss_fn()`` must build its scalar loss
from tape-traced ``tessera.ops.*`` calls with the module's Parameters flowing
through them, or no gradient is recorded. Numpy-only sublayers (integer
embedding lookup, the data-dependent top-k MoE routing in the reference models)
are tape-invisible in v1 — this mirrors PithTrain excluding the data-dependent
MoE forward/backward from ``torch.compile(fullgraph=True)``. Use a fully-traced
model (e.g. ``tessera.train.models.TracedMoEPolicy``) for the differentiable
path.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from tessera import optim
from tessera.autodiff import tape


def _np(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _param_set(p, arr: np.ndarray) -> None:
    """Write ``arr`` into a Parameter's underlying numpy buffer in place."""
    buf = p
    while hasattr(buf, "_data"):
        buf = buf._data
    buf[...] = np.asarray(arr, dtype=buf.dtype)


def adamw_step(
    module,
    loss_fn: Callable[[], Any],
    opt_state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[float, dict[str, Any]]:
    """One reverse-mode + AdamW update over ``module``'s parameters.

    Args:
        module:   an ``nn.Module`` whose ``named_parameters()`` are trainable.
        loss_fn:  zero-arg closure returning a *traced* scalar loss.
        opt_state: AdamW moment state from a prior step (``None`` to init).
    Returns:
        ``(loss_value, opt_state)``.
    """
    for p in module.parameters():
        p._grad = None

    with tape() as t:
        loss = loss_fn()
        t.backward(loss)

    named = list(module.named_parameters())
    params = {n: p.numpy() for n, p in named}
    grads = {
        n: (p.grad.numpy() if p.grad is not None else np.zeros_like(p.numpy()))
        for n, p in named
    }

    new_params, opt_state = optim.adamw(
        params, grads, opt_state,
        lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
    )
    for n, p in named:
        _param_set(p, new_params[n])

    return float(_np(loss)), opt_state
