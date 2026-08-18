"""AD-LAW-1 — declarative per-op input specs for the law sweep.

Each spec builds a deterministic, kink-avoiding sample point for one
registered op so ``laws.py`` can evaluate the adjoint and chain laws on it.
Growing coverage is one entry here per op — no engine changes.

Conventions:

- ``make(rng) -> (primals, kwargs)``; arrays float64 unless the op requires
  otherwise (integer labels etc.).
- **Kink avoidance is part of the spec's contract**: nonsmooth ops must be
  sampled strictly inside a smooth piece (``relu`` away from 0, ``maximum``
  away from ties), because the chain law compares against finite
  differences, which straddle the kink otherwise. Behaviour *at* the kink
  is Law 5's job (``nonsmooth.py``'s declared policies and
  ``test_nonsmooth_selection.py``), not this file's.
- ``diff_args`` lists the positional args the laws probe; ``None`` = all.
- ``zero_tangent_ok`` marks ops whose true derivative is 0 almost
  everywhere (``sign``): an identically-zero tangent stream is correct
  there, not a matched-zero symptom.

Ops in the registries with no entry here are reported by the sweep as
``no_spec`` — visible debt, never a silent skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

__all__ = ["InputSpec", "LAW_INPUT_SPECS"]


@dataclass(frozen=True)
class InputSpec:
    make: Callable[[np.random.Generator], tuple[tuple, dict]]
    diff_args: Optional[tuple[int, ...]] = None
    probes: int = 4
    chain: bool = True
    zero_tangent_ok: bool = False
    note: str = ""


def _away_from(x: np.ndarray, kink: float = 0.0, margin: float = 0.25) -> np.ndarray:
    """Push samples at least `margin` away from a kink point."""
    d = x - kink
    return kink + np.where(np.abs(d) < margin, np.sign(d) * margin + d, d)


def _unary(shape=(3, 4), lo=None, hi=None, kink=None):
    def make(rng):
        x = rng.standard_normal(shape)
        if kink is not None:
            x = _away_from(x, kink)
        if lo is not None or hi is not None:
            span = (hi - lo)
            x = lo + (span / 2) * (1 + np.tanh(x))  # smooth squash into (lo, hi)
        return (x,), {}
    return make


def _positive(shape=(3, 4), floor=0.3):
    def make(rng):
        return (floor + np.abs(rng.standard_normal(shape)),), {}
    return make


def _binary(shape=(3, 4), tie_gap=None):
    def make(rng):
        a = rng.standard_normal(shape)
        b = rng.standard_normal(shape)
        if tie_gap is not None:
            b = np.where(np.abs(a - b) < tie_gap, b + np.sign(b - a + 1e-3) * tie_gap, b)
        return (a, b), {}
    return make


def _norm_input(shape=(4, 8)):
    def make(rng):
        return (rng.standard_normal(shape),), {}
    return make


S = InputSpec  # local shorthand


LAW_INPUT_SPECS: dict[str, InputSpec] = {
    # ── smooth unary pointwise ───────────────────────────────────────────────
    "exp": S(_unary()),
    "log": S(_positive()),
    "log1p": S(_positive(floor=-0.5)),
    "expm1": S(_unary()),
    "sqrt": S(_positive()),
    "sin": S(_unary()),
    "cos": S(_unary()),
    "tan": S(_unary(lo=-1.2, hi=1.2)),
    "sinh": S(_unary()),
    "cosh": S(_unary()),
    "asin": S(_unary(lo=-0.9, hi=0.9)),
    "acos": S(_unary(lo=-0.9, hi=0.9)),
    "atan": S(_unary()),
    "erf": S(_unary()),
    "erfc": S(_unary()),
    "reciprocal": S(_positive()),
    "tanh": S(_unary()),
    "sigmoid": S(_unary()),
    "sigmoid_safe": S(_unary()),
    "softplus": S(_unary()),
    "silu": S(_unary()),
    "gelu": S(_unary()),
    "softcap": S(lambda rng: ((rng.standard_normal((3, 4)),), {"cap": 5.0})),

    # ── nonsmooth unary, sampled inside a smooth piece ───────────────────────
    "relu": S(_unary(kink=0.0)),
    "abs": S(_unary(kink=0.0)),
    "absolute": S(_unary(kink=0.0)),
    "sign": S(_unary(kink=0.0), zero_tangent_ok=True, chain=False,
              note="derivative 0 a.e. — zero tangents are correct"),
    "clamp": S(lambda rng: ((rng.standard_normal((3, 4)) * 0.3,),
                            {"min": -1.0, "max": 1.0}),
               note="interior samples (canonical `min`/`max` kwargs); "
                    "bound behaviour is Law 5"),

    # ── binary pointwise ─────────────────────────────────────────────────────
    "add": S(_binary()),
    "mul": S(_binary()),
    "maximum": S(_binary(tie_gap=0.2)),
    "minimum": S(_binary(tie_gap=0.2)),
    "atan2": S(_binary(tie_gap=0.3)),

    # ── reductions ───────────────────────────────────────────────────────────
    "sum": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axis": -1})),
    "mean": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axis": -1})),
    "prod": S(lambda rng: ((0.5 + np.abs(rng.standard_normal((3, 4))),), {"axis": -1})),
    "logsumexp": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axis": -1})),
    "cumsum": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axis": -1})),
    "var": S(_unary(shape=(3, 5))),
    "std": S(_unary(shape=(3, 5))),
    "amax": S(lambda rng: ((np.cumsum(0.3 + np.abs(rng.standard_normal((3, 4))), axis=-1),),
                           {"axis": -1}),
              note="strictly increasing rows — unique argmax, no ties"),
    "amin": S(lambda rng: ((np.cumsum(0.3 + np.abs(rng.standard_normal((3, 4))), axis=-1),),
                           {"axis": -1}),
              note="unique argmin by construction"),

    # ── linear / multilinear ─────────────────────────────────────────────────
    "matmul": S(lambda rng: ((rng.standard_normal((3, 4)),
                              rng.standard_normal((4, 2))), {})),
    "gemm": S(lambda rng: ((rng.standard_normal((3, 4)),
                            rng.standard_normal((4, 2))), {})),
    "transpose": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axes": (1, 0)})),
    "reshape": S(lambda rng: ((rng.standard_normal((3, 4)),), {"shape": (12,)})),

    # ── softmax family / normalization ───────────────────────────────────────
    "softmax": S(lambda rng: ((rng.standard_normal((3, 5)),), {"axis": -1})),
    "softmax_safe": S(lambda rng: ((rng.standard_normal((3, 5)),), {"axis": -1})),
    "log_softmax": S(lambda rng: ((rng.standard_normal((3, 5)),), {"axis": -1})),
    "layer_norm": S(_norm_input()),
    "rmsnorm": S(_norm_input()),

    # ── losses ───────────────────────────────────────────────────────────────
    "mse_loss": S(_binary(), note="both operands differentiable"),
    "cross_entropy_loss": S(
        lambda rng: ((rng.standard_normal((4, 6)),
                      rng.integers(0, 6, size=(4,))), {}),
        diff_args=(0,), chain=False,
        note="integer targets are non-differentiable"),
    "kl_divergence": S(
        lambda rng: ((
            np.log((lambda z: z / z.sum(axis=-1, keepdims=True))(
                0.2 + np.abs(rng.standard_normal((3, 5))))),
            (lambda z: z / z.sum(axis=-1, keepdims=True))(
                0.2 + np.abs(rng.standard_normal((3, 5)))),
        ), {}),
        note="arg0 is LOG-probs (p_log_probs), arg1 is probs — per the rule signature"),
}
