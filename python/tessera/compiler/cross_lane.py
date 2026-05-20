"""Cross-lane composition rules (Issue 1, 2026-05-20).

The five frontend lanes
(``tessera_jit`` / ``textual_dsl`` / ``clifford_jit`` /
``complex_jit`` / ``energy_jit``) verify different invariants at
decoration time.  When a function in one lane calls into a function
in another lane, we have to decide: is the call legal?

**The rule, with a one-line proof.**

A lane verifies that *every op in the function* satisfies the
lane's whitelist.  When an outer lane calls an inner lane, the
inner's invariants either *strengthen* the outer's claim (allowed)
or *break* it (forbidden).

  * ``@tessera.jit → @clifford_jit``: outer claims nothing
    op-specific, inner claims GA-only.  Allowed.
  * ``@clifford_jit → @tessera.jit``: outer claims GA-only, inner
    claims nothing op-specific.  The inner could contain ``np.dot``
    or any tensor op, breaking the outer's GA-only guarantee.
    **Forbidden**.
  * ``@complex_jit → @clifford_jit``: outer claims holomorphic,
    inner claims GA-only.  The two invariants are about different
    value kinds (``complex`` vs ``multivector``), so neither
    strengthens the other.  Today: **forbidden**.  Future: maybe
    allowed when a "mixed" composition story exists.

The general principle: a call is allowed iff the inner lane's
invariants are a **superset** of the outer's.  In practice today
this collapses to a small table since we have only 5 lanes.

This module ships the rule as data + a detector + a typed
exception.  It does **not** implement cross-lane execution
infrastructure — that's a future build-out gated by an actual
use case.  The detector exists so when a user accidentally writes
the forbidden form they get a typed diagnostic with the rule
explained, not a confusing op-not-whitelisted error from inside
the inner lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .diagnostics import Diagnostic
from .frontend_lanes import FrontendLane


# ─────────────────────────────────────────────────────────────────────
# The legality matrix.  Each entry is (outer, inner) → allowed.
#
# Default is "forbidden" for any pair not in the allowed set.  The
# allowed set is intentionally small — only one-way nesting from
# the general lanes (``tessera_jit`` / ``textual_dsl``) into the
# constrained lanes.
# ─────────────────────────────────────────────────────────────────────

_ALLOWED_NESTINGS: frozenset[tuple[FrontendLane, FrontendLane]] = frozenset({
    # General → constrained: the inner's invariant is stronger.
    (FrontendLane.TESSERA_JIT, FrontendLane.CLIFFORD_JIT),
    (FrontendLane.TESSERA_JIT, FrontendLane.COMPLEX_JIT),
    (FrontendLane.TESSERA_JIT, FrontendLane.ENERGY_JIT),
    (FrontendLane.TEXTUAL_DSL, FrontendLane.CLIFFORD_JIT),
    (FrontendLane.TEXTUAL_DSL, FrontendLane.COMPLEX_JIT),
    (FrontendLane.TEXTUAL_DSL, FrontendLane.ENERGY_JIT),
    # Same-lane is always allowed (trivially).  Listed explicitly so
    # the detector handles it without a special case.
    (FrontendLane.TESSERA_JIT, FrontendLane.TESSERA_JIT),
    (FrontendLane.TEXTUAL_DSL, FrontendLane.TEXTUAL_DSL),
    (FrontendLane.CLIFFORD_JIT, FrontendLane.CLIFFORD_JIT),
    (FrontendLane.COMPLEX_JIT, FrontendLane.COMPLEX_JIT),
    (FrontendLane.ENERGY_JIT, FrontendLane.ENERGY_JIT),
})


@dataclass(frozen=True)
class CrossLaneViolation:
    """A typed record of a forbidden cross-lane call."""

    outer_lane: FrontendLane
    inner_lane: FrontendLane
    reason: str
    """One-line explanation of why this nesting violates lane
    semantics.  Surfaced to the user verbatim — keep it actionable."""

    def to_diagnostic(self) -> Diagnostic:
        """Lift this violation into a unified ``Diagnostic`` for
        ``.explain()`` consumption.  Code is the constrained-lane
        code matching the *outer* lane (whose invariant is being
        broken)."""

        from .diagnostics import ConstrainedDiagnosticCode

        code_by_outer = {
            FrontendLane.CLIFFORD_JIT: ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED.value,
            FrontendLane.COMPLEX_JIT: ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC.value,
            FrontendLane.ENERGY_JIT: ConstrainedDiagnosticCode.ENERGY_FORBIDDEN_OP.value,
        }
        code = code_by_outer.get(
            self.outer_lane,
            ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED.value,
        )
        return Diagnostic.from_constrained(
            code=code,
            message=(
                f"cross-lane composition: outer "
                f"{self.outer_lane.value!r} cannot call inner "
                f"{self.inner_lane.value!r}.  {self.reason}"
            ),
            lane=self.outer_lane.value,
            detail={
                "outer_lane": self.outer_lane.value,
                "inner_lane": self.inner_lane.value,
            },
        )


def detect_violation(
    outer_lane: FrontendLane | str,
    inner_lane: FrontendLane | str,
) -> Optional[CrossLaneViolation]:
    """Return a :class:`CrossLaneViolation` when the pairing is
    forbidden, ``None`` when the pairing is legal.

    Accepts either :class:`FrontendLane` enum values or their
    string ``.value`` (``"tessera_jit"`` / ``"clifford_jit"`` / ...)
    so callers can pass strings without enum imports.
    """

    outer = _to_lane(outer_lane)
    inner = _to_lane(inner_lane)
    if (outer, inner) in _ALLOWED_NESTINGS:
        return None
    return CrossLaneViolation(
        outer_lane=outer,
        inner_lane=inner,
        reason=_reason_for(outer, inner),
    )


def is_legal(
    outer_lane: FrontendLane | str,
    inner_lane: FrontendLane | str,
) -> bool:
    """Convenience predicate over :func:`detect_violation`."""

    return detect_violation(outer_lane, inner_lane) is None


def allowed_nestings() -> tuple[tuple[FrontendLane, FrontendLane], ...]:
    """Return the full set of (outer, inner) pairs that are legal.

    Stable ordering: outer enum value alphabetical, inner enum
    value alphabetical.  Useful for docs + tests that want to
    enumerate the matrix without depending on the underlying set's
    iteration order.
    """

    return tuple(
        sorted(
            _ALLOWED_NESTINGS,
            key=lambda pair: (pair[0].value, pair[1].value),
        )
    )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _to_lane(value: FrontendLane | str) -> FrontendLane:
    if isinstance(value, FrontendLane):
        return value
    for lane in FrontendLane:
        if lane.value == value:
            return lane
    raise ValueError(
        f"unknown frontend lane: {value!r}; expected one of "
        f"{[lane.value for lane in FrontendLane]}"
    )


def _reason_for(outer: FrontendLane, inner: FrontendLane) -> str:
    """Build the human-readable explanation for a forbidden pair."""

    if outer == inner:
        # Should never happen — same-lane is always allowed.
        return "same-lane composition (should be allowed; bug in detector)"

    # Constrained → general (or different constrained) is the
    # common failure mode.  Walk the constrained-lane invariants
    # and explain why the inner doesn't satisfy them.
    constrained_invariants = {
        FrontendLane.CLIFFORD_JIT: (
            "the outer @clifford_jit lane requires every op in the "
            "function (including helpers) to be in the GA whitelist"
        ),
        FrontendLane.COMPLEX_JIT: (
            "the outer @complex_jit lane requires every op in the "
            "function (including helpers) to be holomorphic"
        ),
        FrontendLane.ENERGY_JIT: (
            "the outer @energy_jit lane requires every op in the "
            "function (including helpers) to be in the energy whitelist"
        ),
    }
    if outer in constrained_invariants:
        return (
            f"{constrained_invariants[outer]}; calling a "
            f"{inner.value!r}-decorated helper would import its "
            f"broader op vocabulary into the outer's verification "
            f"scope.  Inline the call or switch the outer to "
            f"@tessera.jit."
        )
    return (
        f"no allowed nesting from {outer.value!r} to {inner.value!r}.  "
        f"See tessera.compiler.cross_lane.allowed_nestings() for the "
        f"full legality matrix."
    )


__all__ = [
    "CrossLaneViolation",
    "allowed_nestings",
    "detect_violation",
    "is_legal",
]
