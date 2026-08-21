"""AD-WEIL-1 §2.1 — one derivative datum per primitive.

`DerivativeContract` is the registration surface the AUTODIFF_NEXTGEN_PLAN
§2.1 specifies: the *minimal* mathematical data from which every AD mode
derives. The registry is **derived from the existing single authorities**
(Decision #30 — derive, don't ask): the holonomic ODE table
(`algebra.SCALAR_RECURRENCES`), the multilinear structure declarations
(`linear.MULTILINEAR_PRIMITIVES`), and the declared kink policies
(`nonsmooth.NONSMOOTH_SELECTION`). Nothing here restates a rule; a
primitive appearing in one of those tables *is* its registration.

Consumers (Decision #29 — a declaration must have a consumer):
  * the Law-2/Law-4 substrate programs evaluate the `ode` datum through
    `algebra.TruncatedJet.scalar_fn` / `Dual.scalar_fn`;
  * the Law-5 kink sweep asserts each `kink_policy` at its tie points;
  * `linear.py` derives JVPs and transposes from `linear_args`;
  * the `pd_witness` certificate is validated structurally here (a
    primitive cannot claim `smooth` while declaring a kink policy) and
    consumed by `tests/unit/test_autodiff_laws.py`'s contract tests.

The §3.6 semantics of `pd_witness`: `"smooth"` means C¹ (the conservative
field is the singleton gradient — no definability claim needed);
`"definable:semialgebraic"` is claimed exactly for the piecewise-polynomial
nonsmooth primitives, where the declared kink policy selects the element of
the Clarke subdifferential. Compositions inherit path-differentiability by
the Bolte–Pauwels chain rule, which is what tape composition implements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .algebra import SCALAR_RECURRENCES, ScalarRecurrence

__all__ = [
    "DERIVATIVE_CONTRACTS",
    "DerivativeContract",
    "RETIRED_HAND_RULES",
    "derivative_contract",
    "register_datum_derived_rules",
]

_PD_WITNESSES = ("smooth", "definable:semialgebraic")


@dataclass(frozen=True)
class DerivativeContract:
    """One primitive's derivative datum (AUTODIFF_NEXTGEN_PLAN §2.1).

    Exactly one of the structure fields is normally set; a primitive with
    none is registered only for its semantic keys. Structural conflicts
    (an ODE entry claiming a kink policy, an unknown witness) are rejected
    at construction — fail closed, never a silently-wrong default (#21a).
    """

    # (1) Structure declaration — often the WHOLE rule:
    linear_args: Optional[tuple[int, ...]] = None
    # (2) Pointwise/scalar primitives — the one datum all orders derive from:
    ode: Optional[ScalarRecurrence] = None
    # (3) Structured primitives — linearization as a graph of OTHER
    #     primitives (unpopulated until AD-JET-STRUCT-1 lifts a fused
    #     family; the field exists so the interface does not break then):
    linearize: Optional[Callable[..., Any]] = None
    transpose: Optional[Callable[..., Any]] = None
    # (4) Nonsmooth semantic key (#21a) — from nonsmooth.py:
    kink_policy: Optional[str] = None
    # (5) Path-differentiability certificate for the §3.6 guarantee:
    pd_witness: Optional[str] = None

    def __post_init__(self) -> None:
        if self.pd_witness is not None and self.pd_witness not in _PD_WITNESSES:
            raise ValueError(
                f"pd_witness must be one of {_PD_WITNESSES}; got "
                f"{self.pd_witness!r}. A bare boolean cannot carry the "
                f"hypothesis (AUTODIFF_NEXTGEN_PLAN §3.6)."
            )
        if self.ode is not None and self.kink_policy is not None:
            raise ValueError(
                "a primitive cannot declare both a holonomic ODE (C¹ "
                "smooth) and a kink policy — conflicting smoothness claims"
            )
        if self.ode is not None and self.pd_witness == "definable:semialgebraic":
            raise ValueError(
                "an ODE-family primitive carries the 'smooth' witness; "
                "'definable:semialgebraic' is claimed only for the "
                "piecewise-polynomial nonsmooth primitives (§3.6)"
            )
        if self.kink_policy is not None and self.pd_witness == "smooth":
            raise ValueError(
                "a kink-family primitive cannot claim the 'smooth' witness"
            )


def _build_registry() -> dict[str, DerivativeContract]:
    from .linear import MULTILINEAR_PRIMITIVES
    from .nonsmooth import NONSMOOTH_SELECTION

    contracts: dict[str, DerivativeContract] = {}

    def _register(name: str, contract: DerivativeContract) -> None:
        if name in contracts:
            raise ValueError(
                f"duplicate DerivativeContract registration for {name!r} — "
                f"the same duplicate guard the (V/J)VP registries apply"
            )
        contracts[name] = contract

    for name, rec in SCALAR_RECURRENCES.items():
        _register(name, DerivativeContract(ode=rec, pd_witness="smooth"))

    for name, policy in NONSMOOTH_SELECTION.items():
        _register(name, DerivativeContract(
            kink_policy=policy, pd_witness="definable:semialgebraic"))

    for name, linear_args in MULTILINEAR_PRIMITIVES.items():
        if name in contracts:
            raise ValueError(
                f"{name!r} declared both multilinear and "
                f"{'smooth-ODE' if contracts[name].ode else 'kinked'} — "
                f"reconcile the source registries before registering"
            )
        # Multilinear maps are polynomial, hence semialgebraic AND smooth;
        # the stronger claim (smooth) is the honest witness.
        _register(name, DerivativeContract(
            linear_args=tuple(linear_args), pd_witness="smooth"))

    return contracts


DERIVATIVE_CONTRACTS: dict[str, DerivativeContract] = _build_registry()


def derivative_contract(name: str) -> Optional[DerivativeContract]:
    """The registered contract for ``name``, or ``None`` — never a default
    contract: absence of a datum is a fact the caller must handle, not
    paper over (#21a)."""
    return DERIVATIVE_CONTRACTS.get(name)


# ── Hand-rule retirement: the datum becomes production (AD-RETIRE-1) ─────────
#
# With the Law-4 differential proofs green (AD-WEIL-1 for the scalar family,
# AD-JET-STRUCT-1 for the structured ones), retirement is an evidence-backed
# call. This is the first family to take it: the 13 holonomic-ODE primitives'
# production JVP/VJP pairs are now DERIVED from the one datum
# (`ScalarRecurrence.pointwise`), and the displaced hand rules become the
# declared oracles (#31) held in `RETIRED_HAND_RULES` for the differential
# test — deleted only in a follow-up once the oracle has soaked.
#
# What the derived path carries (§8's bar — everything the deleted one did):
#
# * **dtype behavior** — float64 casts, identical to the displaced factory.
# * **evaluation guards** — the displaced rules' domain clamps, carried as
#   EXPLICIT per-op declarations below instead of buried lambdas. One guard
#   per op for BOTH modes. This *fixes a measured inconsistency*: the
#   displaced pair disagreed at the boundary (``jvp_sqrt`` clamped ``√x`` at
#   1e-12 — slope cap 5e11 — while ``vjp_sqrt`` clamped ``x`` — slope cap
#   5e5; ``jvp_log`` had NO guard while ``vjp_log`` clamped). J and Jᵀ of
#   one function cannot disagree at the same point; the VJP's convention
#   survives (reverse mode is the historically production-critical path),
#   and the boundary change to forward mode is pinned in
#   ``test_retired_pointwise.py``.

_DERIVATIVE_EVAL_GUARDS: dict[str, Callable[[Any], Any]] = {
    "log": lambda x: np.where(
        np.abs(x) < 1e-12, np.copysign(1e-12, x + 0.0), x),
    "sqrt": lambda x: np.maximum(x, 1e-12),
}

#: The displaced hand rules, kept as declared oracles (#31) for the
#: differential test. Populated by `register_datum_derived_rules`.
RETIRED_HAND_RULES: dict[str, tuple[Callable, Callable]] = {}


def _make_derived_pair(name: str, rec: ScalarRecurrence):
    value, derivative = rec.pointwise
    guard = _DERIVATIVE_EVAL_GUARDS.get(name)

    def derived_jvp(primals, tangents, **_):
        x = np.asarray(primals[0], dtype=np.float64)
        dx = np.asarray(tangents[0], dtype=np.float64)
        xg = guard(x) if guard is not None else x
        return np.asarray(value(x)), derivative(xg) * dx

    def derived_vjp(dout, x, **_):
        a = np.asarray(x, dtype=np.float64)
        ag = guard(a) if guard is not None else a
        return (derivative(ag) * np.asarray(dout, dtype=np.float64),)

    derived_jvp._derived_from_datum = name  # type: ignore[attr-defined]
    derived_vjp._derived_from_datum = name  # type: ignore[attr-defined]
    derived_jvp.__name__ = f"jvp_{name}__datum"
    derived_vjp.__name__ = f"vjp_{name}__datum"
    return derived_jvp, derived_vjp


def register_datum_derived_rules() -> list[str]:
    """Switch the ODE-family production rules to the datum-derived pair.

    Idempotent; returns the switched names. Fails closed rather than
    switching a family whose hand pair is incomplete (a datum without a
    displaced oracle would leave the differential test with nothing to
    differ against)."""
    from .jvp import get_jvp, register_jvp
    from .vjp import get_vjp, register_vjp

    switched: list[str] = []
    for name, contract in DERIVATIVE_CONTRACTS.items():
        if contract.ode is None:
            continue
        current_jvp = get_jvp(name)
        if current_jvp is not None and getattr(
                current_jvp, "_derived_from_datum", None) == name:
            switched.append(name)  # already switched (idempotent re-entry)
            continue
        current_vjp = get_vjp(name)
        if current_jvp is None or current_vjp is None:
            raise ValueError(
                f"refusing to retire {name!r}: the hand pair is incomplete "
                f"(jvp={current_jvp is not None}, "
                f"vjp={current_vjp is not None}) — the displaced oracle "
                f"must exist before the switch (#31)"
            )
        RETIRED_HAND_RULES[name] = (current_jvp, current_vjp)
        derived_jvp, derived_vjp = _make_derived_pair(name, contract.ode)
        register_jvp(name, derived_jvp)
        register_vjp(name, derived_vjp)
        switched.append(name)
    return switched
