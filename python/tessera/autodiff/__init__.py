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
    count_primitive_executions,
)
from .vjp import register_vjp, get_vjp, _VJPS
from . import degeneracy
from .degeneracy import (
    TesseraDegeneracyError,
    degeneracy_policy,
    FACTORIZATION_DEGENERACY,
)
from .mixed_precision import autocast, autocast_dtype, GradScaler
from .rematerialize import rematerialize, checkpoint
from .grad import grad, hvp, elementwise_grad
from .jvp import register_jvp, get_jvp, jvp
from .transforms import vmap, jacrev, jacfwd
from .linear import (
    make_linear_jvp,
    register_derived_linear_jvps,
    MULTILINEAR_PRIMITIVES,
)
from .implicit import (
    TesseraImplicitDiffError,
    cg_solve,
    ihvp,
    root_vjp,
    root_jvp,
    custom_root,
    adjoint_state_grad,
)
from .operator import (
    OperatorTangent,
    RootConditionCertificate,
    certify_root,
)


# AD-RETIRE-1: switch the ODE-family production rules to the
# datum-derived pair (the displaced hand rules become declared oracles in
# `derivative_contract.RETIRED_HAND_RULES` — see that module's retirement
# section). Runs before the op wrappers so the wrapped ops see the derived
# registry from the first call.
from .derivative_contract import register_datum_derived_rules
register_datum_derived_rules()

# AD-RETIRE-2: the structured family (softmax / logsumexp / rmsnorm-core)
# switches to jet-derived first-order rules — see jet.py's retirement
# section for the envelope audit and the symmetric-transpose derivations.
# MSW-2: the jet surface is part of the public autodiff API, not an internal
# of the rule-derivation hook. Exact higher-order derivatives were reachable
# only by importing `tessera.autodiff.jet` directly and hand-writing a jet
# program in its `jet_*` vocabulary; `jet_trace` removes the second half of
# that and this export removes the first.
from .jet import (
    Jet,
    hessian_trace_estimate,
    jet_add,
    jet_const,
    jet_flash_attn,
    jet_layer_norm,
    jet_lift,
    jet_logsumexp,
    jet_map,
    jet_matmul,
    jet_mean,
    jet_mul,
    jet_reduce_max,
    jet_rmsnorm,
    jet_scale,
    jet_softmax,
    jet_sub,
    jet_sum,
    jet_trace,
    laplacian_estimate,
    laplacian_exact,
    register_jet_derived_structured_rules,
)
from .algebra import TruncatedJet
register_jet_derived_structured_rules()

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
    # MSW-2 — exact higher-order derivatives
    "TruncatedJet",
    "Jet",
    "jet_trace",
    "laplacian_exact",
    "laplacian_estimate",
    "hessian_trace_estimate",
    "jet_lift",
    "jet_const",
    "jet_add",
    "jet_sub",
    "jet_mul",
    "jet_scale",
    "jet_map",
    "jet_matmul",
    "jet_sum",
    "jet_mean",
    "jet_softmax",
    "jet_logsumexp",
    "jet_rmsnorm",
    "jet_layer_norm",
    "jet_flash_attn",
    "jet_reduce_max",
    "tape",
    "reverse",
    "custom_rule",
    "register_vjp",
    "get_vjp",
    "TesseraAutodiffError",
    "Tape",
    "TapeEntry",
    "InputDesc",
    # Phase F1 — mixed-precision
    "autocast",
    "autocast_dtype",
    "GradScaler",
    # Phase F2 — activation checkpointing
    "rematerialize",
    "checkpoint",
    # Deferred-items plan, Item 4 — higher-order derivative helpers
    "grad",
    "hvp",
    "elementwise_grad",
    # Deferred-items plan, Item 5 — JAX-style transforms + forward-mode JVP
    "vmap",
    "jacrev",
    "jacfwd",
    "jvp",
    "register_jvp",
    "get_jvp",
]
