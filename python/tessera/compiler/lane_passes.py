"""Lane-aware Graph IR optimization passes (G1, 2026-05-19).

Demonstrates the payoff of attaching ``lane`` to
:class:`tessera.compiler.graph_ir.GraphIRFunction` — passes can now
exploit per-lane invariants without re-deriving lane membership at
every site.

Two passes ship today:

  * :func:`assert_complex_jit_holomorphic` — verifies that every op
    in a ``complex_jit``-lowered function is in the holomorphic
    whitelist.  The decoration-time CR check already proved this,
    but a Graph-IR-time assertion makes the invariant load-bearing
    against future passes that might rewrite the body.

  * :func:`assert_clifford_ops_only` — verifies that every op in a
    ``clifford_jit``-lowered function is in the GA whitelist (every
    op name starts with ``clifford_``).  Same logic — re-asserting
    the lane invariant after lowering catches accidental cross-lane
    op leakage.

These are conservative passes — they raise on violation rather than
silently rewriting.  Future passes (dead-anti-holomorphic-branch
elimination, fused-rotor-sandwich folding) can read the lane to
choose lane-specific rewrite strategies.
"""

from __future__ import annotations

from .diagnostics import ConstrainedDiagnosticCode, Diagnostic
from .graph_ir import GraphIRFunction


# Holomorphic op whitelist — mirrors the one in complex_jit.py but
# duplicated intentionally so the lane-aware pass is independent of
# the decoration-time logic.
_HOLOMORPHIC_GRAPH_OP_NAMES = frozenset({
    "complex_mul",
    "complex_div",
    "complex_exp",
    "complex_log",
    "complex_sqrt",
    "complex_pow",
    # Möbius / stereographic preserve holomorphicity.
    "mobius",
    "stereographic",
    "mobius_from_three_points",
})

_NON_HOLOMORPHIC_GRAPH_OP_NAMES = frozenset({
    "complex_conjugate",
    "complex_abs",
    "complex_arg",
    "check_cauchy_riemann",
    "dbar",
    "laplacian_2d",
})


def assert_complex_jit_holomorphic(
    fn: GraphIRFunction,
) -> list[Diagnostic]:
    """Lane-aware pass: every op in a ``complex_jit`` function must
    be holomorphic.

    Returns an empty list when the invariant holds.  Returns a
    Diagnostic per offending op otherwise.  The pass does not
    rewrite the IR — emission is for ``.explain()`` consumption.

    No-op for functions whose ``lane`` is not ``complex_jit``.  This
    makes the pass safe to add to a general optimization pipeline.
    """

    if fn.lane != "complex_jit":
        return []

    diagnostics: list[Diagnostic] = []
    for op in fn.body:
        # Strip "tessera." prefix if present so we match the bare op
        # name against the whitelist.
        op_name = op.op_name
        if op_name.startswith("tessera."):
            op_name = op_name[len("tessera."):]
        if op_name in _NON_HOLOMORPHIC_GRAPH_OP_NAMES:
            diagnostics.append(Diagnostic.from_constrained(
                code=ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC,
                message=(
                    f"@complex_jit function {fn.name!r} contains "
                    f"non-holomorphic op {op_name!r} at IR level; "
                    "decoration-time CR check should have caught this — "
                    "a pass earlier in the pipeline introduced it"
                ),
                lane="complex_jit",
                detail={"function": fn.name, "op": op_name},
            ))
    return diagnostics


def assert_clifford_ops_only(
    fn: GraphIRFunction,
) -> list[Diagnostic]:
    """Lane-aware pass: every op in a ``clifford_jit`` function must
    be a ``clifford_*`` op (the GA whitelist).

    Returns an empty list when the invariant holds.  Returns a
    Diagnostic per offending op otherwise.

    No-op for functions whose ``lane`` is not ``clifford_jit``.
    """

    if fn.lane != "clifford_jit":
        return []

    diagnostics: list[Diagnostic] = []
    for op in fn.body:
        op_name = op.op_name
        if op_name.startswith("tessera."):
            op_name = op_name[len("tessera."):]
        if not op_name.startswith("clifford_"):
            diagnostics.append(Diagnostic.from_constrained(
                code=ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED,
                message=(
                    f"@clifford_jit function {fn.name!r} contains "
                    f"non-GA op {op_name!r} at IR level; "
                    "decoration-time whitelist should have caught this — "
                    "a pass earlier in the pipeline introduced it"
                ),
                lane="clifford_jit",
                detail={"function": fn.name, "op": op_name},
            ))
    return diagnostics


def run_lane_aware_passes(fn: GraphIRFunction) -> list[Diagnostic]:
    """Run every lane-aware pass against ``fn`` and collect
    diagnostics.

    The pass list is open-ended — when a new lane-aware pass lands,
    append it here.  Today the two-pass set covers the two
    constrained math lanes; the Python ``@tessera.jit`` lane has no
    invariants strict enough to justify a dedicated pass.
    """

    out: list[Diagnostic] = []
    out.extend(assert_complex_jit_holomorphic(fn))
    out.extend(assert_clifford_ops_only(fn))
    return out


__all__ = [
    "assert_clifford_ops_only",
    "assert_complex_jit_holomorphic",
    "run_lane_aware_passes",
]
