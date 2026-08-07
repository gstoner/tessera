"""Automatic differentiation of linear / multilinear primitives (C1).

Motivation (from the *Elements of Differentiable Programming* review). For a
linear map ``l`` the directional derivative is the map itself applied to the
tangent — ``∂l(w)[v] = l(v)`` — and the VJP is its adjoint ``l*`` (Blondel &
Roulet §4.5.4). More generally, for a *multilinear* map (linear in each argument
separately, e.g. ``matmul``) the JVP is the product-rule sum

    ∂f(w₁,…,wₙ)[v₁,…,vₙ] = Σᵢ f(w₁,…,vᵢ,…,wₙ),

i.e. the forward primitive itself, evaluated with one argument replaced by its
tangent, summed over the linearly-entered arguments. Nothing new needs to be
hand-written: the JVP of a multilinear op is *derived* from its own forward.

The review's point was that ``vjp.py`` (≈5.4k lines) and ``jvp.py`` (≈3.8k)
hand-maintain the same linear-primitive derivatives twice, and that the declared
``transpose_rule`` axis (``custom.py``) had **no consumer** — a Decision #29
violation. This module is that consumer:

  * ``make_linear_jvp`` derives a JVP for any multilinear op from its forward.
  * ``register_derived_linear_jvps`` fills JVP-registry gaps for the declared
    multilinear built-ins (additive: never overrides a hand-written JVP).
  * ``custom.py`` uses ``make_linear_jvp`` and treats a linear primitive's
    declared ``transpose_rule`` as its VJP (the adjoint of the forward map).

``tests/unit/test_linear_transposition.py`` cross-checks the derived JVP against
the hand-written one for every op in the overlap, proving the duplication is
mechanical (and that the surviving path is faithful before anything is removed —
Decision #31 ordering).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

__all__ = [
    "MULTILINEAR_PRIMITIVES",
    "is_multilinear",
    "linear_arg_positions",
    "make_linear_jvp",
    "register_derived_linear_jvps",
]

# op name → positional argument indices that enter (multi)linearly. Structural
# arguments (axes, shape, indices) travel as kwargs and are held fixed. Every op
# here satisfies ∂f[v] = Σ over these indices of f(with vᵢ swapped in).
MULTILINEAR_PRIMITIVES: Mapping[str, tuple[int, ...]] = {
    # Unary linear reshaping / reindexing (linear in arg 0).
    "transpose": (0,),
    "permute": (0,),
    "reshape": (0,),
    "view": (0,),
    "flatten": (0,),
    "squeeze": (0,),
    "unsqueeze": (0,),
    "broadcast": (0,),
    "expand": (0,),
    "flip": (0,),
    "roll": (0,),
    "tile": (0,),
    "repeat": (0,),
    # Bilinear contraction (linear in each factor separately).
    "matmul": (0, 1),
    "gemm": (0, 1),
}


def is_multilinear(name: str) -> bool:
    return name in MULTILINEAR_PRIMITIVES


def linear_arg_positions(name: str) -> tuple[int, ...]:
    """Positional args ``name`` is linear in. Raises ``KeyError`` (fail closed)
    for an op not declared multilinear — callers must not guess."""
    return MULTILINEAR_PRIMITIVES[name]


def make_linear_jvp(
    forward: Callable[..., Any],
    linear_args: tuple[int, ...],
) -> Callable[[tuple, tuple], tuple]:
    """Derive a JVP from linearity: ``∂f(w)[v] = Σᵢ f(…, vᵢ, …)``.

    ``forward`` is the primitive's own forward implementation (numpy reference).
    ``linear_args`` are the positional indices it is linear in. The returned
    callable has the project's JVP signature ``(primals, tangents, **kwargs) ->
    (primal_out, tangent_out)``. Tangents that are ``None`` (an argument with no
    incoming perturbation) contribute no term.
    """

    def _jvp(primals: tuple, tangents: tuple, **kwargs: Any) -> tuple:
        primal_out = forward(*primals, **kwargs)
        tangent_out = None
        for i in linear_args:
            if i >= len(tangents):
                continue
            ti = tangents[i]
            if ti is None:
                continue
            swapped = list(primals)
            swapped[i] = ti
            term = forward(*swapped, **kwargs)
            tangent_out = term if tangent_out is None else tangent_out + term
        if tangent_out is None:
            tangent_out = np.zeros_like(np.asarray(primal_out))
        return primal_out, tangent_out

    return _jvp


def register_derived_linear_jvps() -> list[str]:
    """Register linearity-derived JVPs for declared multilinear built-ins that
    have no JVP yet. Additive only — a hand-written JVP is never overridden
    (Decision #31: the surviving path must carry what it replaces before any
    duplicate is removed; this fills gaps, it does not delete). Returns the list
    of op names that received a derived JVP.
    """
    # Import the submodule members directly: the autodiff package __init__
    # binds the name ``jvp`` to the *transform function*, which shadows the
    # ``autodiff.jvp`` submodule for ``from . import jvp``.
    from .jvp import get_jvp as _get_jvp, register_jvp as _register_jvp

    filled: list[str] = []
    for name, linear_args in MULTILINEAR_PRIMITIVES.items():
        if _get_jvp(name) is not None:
            continue  # a hand-written JVP already exists; leave it.
        forward = _resolve_forward(name)
        if forward is None:
            continue
        _register_jvp(name, make_linear_jvp(forward, linear_args))
        filled.append(name)
    return filled


def _resolve_forward(name: str) -> Callable[..., Any] | None:
    """The unwrapped forward implementation for a built-in op, or ``None``."""
    try:
        from .. import ops
    except Exception:  # pragma: no cover - ops must import for anything to work
        return None
    fn = getattr(ops, name, None)
    if fn is None:
        return None
    # ops.* are tape-wrapped; unwrap to the raw numpy impl so the derived JVP
    # does not itself get recorded on a tape.
    return getattr(fn, "__wrapped__", fn)
