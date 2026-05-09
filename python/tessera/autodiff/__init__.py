"""Tessera Autodiff — v1 first slice (numpy-reference, tape-based reverse-mode).

Public surface:
    tessera.autodiff.tape()              — context manager
    tessera.autodiff.reverse(fn)         — function transform → (loss, grads_dict)
    tessera.autodiff.custom_rule(name)   — register/override a VJP
    tessera.autodiff.TesseraAutodiffError — raised on misuse / missing VJP

See `docs/spec/AUTODIFF_SPEC.md` for the design and the explicit non-goals
(Graph/Tile IR adjoints, distributed grad collectives, rematerialization,
mixed-precision master-copy — all deferred to later slices).
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np

from .tape import (
    Tape,
    TapeEntry,
    InputDesc,
    TesseraAutodiffError,
    tape,
    install_op_wrappers,
)
from .vjp import register_vjp, get_vjp, _VJPS


# Wrap every op in `_VJPS` so it's tape-aware.
install_op_wrappers()


def custom_rule(op_name: str) -> Callable[[Callable], Callable]:
    """Decorator: register or override the VJP for a `tessera.ops.<op_name>`.

    The op's `tessera.ops` callable is automatically tape-wrapped if it wasn't
    already. The VJP signature is `(dout, *forward_inputs, **kwargs) -> tuple[dinput,...]`.

    Example:
        @tessera.autodiff.custom_rule("flash_attn")
        def _vjp_flash_attn(dout, Q, K, V, **kwargs):
            ...
            return (dQ, dK, dV)
    """
    def deco(fn: Callable) -> Callable:
        register_vjp(op_name, fn)
        # Re-install wrappers so a newly-registered op gets wrapped.
        install_op_wrappers()
        return fn
    return deco


def reverse(fn: Callable) -> Callable:
    """Convert `fn(*args, **kwargs) -> scalar_loss` into a function that
    returns `(loss, grads_dict)` and populates `.grad` on every Parameter
    encountered in the forward pass.

    `grads_dict` keys are constructed from any `Module` arguments in the input:
    each module contributes `{f"<arg_index>.<param_name>": ndarray}` entries.
    Modules without parameters or non-Module args produce no entries.
    """
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any):
        with tape() as t:
            loss = fn(*args, **kwargs)
            t.backward(loss)
        grads: dict[str, np.ndarray] = {}
        # Local import to avoid a cycle on module load
        from ..nn.module import Module
        for i, arg in enumerate(args):
            if isinstance(arg, Module):
                for name, p in arg.named_parameters():
                    if p.grad is not None:
                        grads[f"{i}.{name}"] = p.grad.numpy()
        return loss, grads
    return wrapped


__all__ = [
    "tape",
    "reverse",
    "custom_rule",
    "register_vjp",
    "get_vjp",
    "TesseraAutodiffError",
    "Tape",
    "TapeEntry",
    "InputDesc",
]
