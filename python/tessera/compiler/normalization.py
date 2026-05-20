"""Phase C — Graph IR normalization pipeline (skeleton, 2026-05-20).

This module defines the **canonical pass ordering** for safe Graph IR
normalization.  Per
``docs/architecture/frontend_substrate_plan.md`` § 3 (Normalization
Pipeline Ordering), normalization passes run in a documented,
stable order — implicit ordering breaks when a new pass lands
between two existing passes that happened to compose correctly by
accident.

**This commit is intentionally skeleton-only.**  Every pass below
is a typed stub with an empty body.  Pass bodies land in later
commits — each one paired with its own focused tests.  The
skeleton + order test establish:

  * The tuple :data:`NORMALIZATION_PIPELINE` exists with the
    documented ordering.
  * Every pass has a stable name + signature.
  * The order test in ``tests/unit/test_normalization_pipeline_order.py``
    refuses silent reorderings.

Pass-body PRs that follow this commit may **only** fill in the
function body — they may not reorder, rename, or remove passes
without updating the architecture doc and the order test in lock
step.

Ordering rationale (per the architecture doc):

1. ``canonicalize_op_names``         — strip ``tessera.`` prefix
   so subsequent passes don't need to.
2. ``propagate_source_positions``    — happens once, all later
   passes preserve it.
3. ``set_lane_provenance``           — once ``GraphIRFunction.lane``
   is set, every other pass can read it.
4. ``propagate_value_kinds``         — depends on op naming.
5. ``propagate_numeric_policy``      — keys on canonical names
   (G3, already shipped as a standalone function).
6. ``propagate_verification_facts``  — consumes everything above.

All passes:

  * **Idempotent** — running twice produces the same result.
  * **Lane-aware** — no-op when the lane invariant prevents the
    change (e.g., a ``complex_jit`` view's ops should not be
    renamed by ``canonicalize_op_names`` since they're already
    canonical).
  * **In-place** — mutate the ``GraphIRFunction`` directly.
    Callers that want the original intact deep-copy first.
"""

from __future__ import annotations

from typing import Callable

from .graph_ir import GraphIRFunction, IROp


# ─────────────────────────────────────────────────────────────────────
# Pass stubs.  Each function below is intentionally no-op pending its
# implementation commit.  Replacing a body is a one-line change; the
# pass name + signature is the stable contract.
# ─────────────────────────────────────────────────────────────────────


def canonicalize_op_names(fn: GraphIRFunction) -> None:
    """Strip the canonical ``tessera.`` prefix from every op name.

    ``tessera.matmul`` → ``matmul``.  Aliases that backend lookup
    re-adds (``complex_mobius``) are NOT introduced here — this pass
    only strips, never aliases.  The audit walker handles backend
    aliasing separately via ``_M7_BACKEND_ALIASES``.

    Idempotent: stripping ``"matmul"`` of its (absent) prefix
    leaves it ``"matmul"``.
    """

    for op in fn.body:
        if op.op_name.startswith("tessera."):
            op.op_name = op.op_name[len("tessera."):]


def propagate_source_positions(fn: GraphIRFunction) -> None:
    """Plumb source spans onto downstream ops that the AST visitor
    couldn't tag.

    The Python ``@tessera.jit`` AST visitor already populates
    ``source_span`` for ops it lowered directly.  Synthetic ops
    introduced by later passes (or by constrained-view adapters
    that don't track source positions) end up with
    ``source_span=None``.  This pass fills the gap by inheriting
    from the **unique producer** of one of the op's operands:

      * If the op has at least one operand whose producer carries
        a span, use the first such producer's span.
      * If no producer carries a span (or the op has no operands),
        leave ``source_span=None``.

    Idempotent: ops that already carry a span are skipped.
    """

    # Build a result-name → producing op map so we can resolve
    # ``%t0`` references back to the op that emitted them.  Both
    # ``%t0`` and ``t0`` forms are inserted so consumers can use
    # either notation.
    producer: dict[str, IROp] = {}
    for op in fn.body:
        if op.result is not None:
            producer[op.result] = op
            producer[f"%{op.result}"] = op

    for op in fn.body:
        if op.source_span is not None:
            continue
        for operand in op.operands:
            upstream = producer.get(operand)
            if upstream is None or upstream.source_span is None:
                continue
            op.source_span = upstream.source_span
            break


def set_lane_provenance(fn: GraphIRFunction) -> None:
    """Ensure ``fn.lane`` is set to a real lane value.

    The dataclass default is already ``"tessera_jit"``, so this
    pass is a safety net for producers that explicitly clear the
    field (e.g., assign ``fn.lane = ""``) or accidentally set it
    to ``None``.  Real lane values are left alone.

    Idempotent: an already-set lane is preserved.
    """

    valid_lanes = (
        "tessera_jit",
        "textual_dsl",
        "clifford_jit",
        "complex_jit",
        "energy_jit",
    )
    if fn.lane not in valid_lanes:
        fn.lane = "tessera_jit"


def propagate_value_kinds(fn: GraphIRFunction) -> None:
    """Stamp ``IROp.value_kind`` based on the op's catalog entry.

    Routing rules (first match wins):

      * ``clifford_*``                      → ``multivector``
      * ``complex_*`` / ``mobius`` /
        ``stereographic`` / ``mobius_from_three_points``
        / ``cross_ratio`` / ``is_concyclic``
        / ``conformal_jacobian`` /
        ``conformal_energy_on_sphere`` /
        ``dz`` / ``dbar`` / ``laplacian_2d`` /
        ``check_cauchy_riemann``            → ``complex``
      * ``energy_*`` / ``ebm_*``            → ``energy``
      * op-name appears in ``OP_SPECS``     → ``tensor``
      * otherwise                           → leave ``None``

    Producers that already set ``value_kind`` (e.g., the
    constrained view adapters) keep their choice.

    Idempotent: ops with a non-None ``value_kind`` are skipped.
    """

    # Late import — primitive_coverage is heavyweight; only pull
    # it in when this pass actually runs against a non-trivial fn.
    from .op_catalog import OP_SPECS

    # M7 op-name set lifted from audit._M7_INVENTORY (the
    # registry's source of truth for the Visual Complex surface).
    # Keeping the set local avoids a normalize→audit→normalize
    # import cycle.
    _M7_NAMES = frozenset({
        "complex_mul", "complex_div", "complex_exp", "complex_log",
        "complex_sqrt", "complex_pow", "complex_conjugate",
        "complex_abs", "complex_arg",
        "mobius", "mobius_from_three_points",
        "cross_ratio", "is_concyclic", "stereographic",
        "check_cauchy_riemann",
        "conformal_jacobian", "conformal_energy_on_sphere",
        "dz", "dbar", "laplacian_2d",
        "complex_jit",
    })

    for op in fn.body:
        if op.value_kind is not None:
            continue
        name = op.op_name
        if name.startswith("clifford_"):
            op.value_kind = "multivector"
        elif name.startswith("complex_") or name in _M7_NAMES:
            op.value_kind = "complex"
        elif name.startswith("energy_") or name.startswith("ebm_"):
            op.value_kind = "energy"
        elif name in OP_SPECS:
            op.value_kind = "tensor"
        # else: leave None — per the architecture-doc rule that
        # producers can't accidentally lie via a default like "tensor".


def propagate_numeric_policy(fn: GraphIRFunction) -> None:
    """Phase C wrapper around the existing G3 helper.

    G3 shipped ``tessera.compiler.numeric_policy_pass.propagate_numeric_policy(fn)``
    as a standalone function.  This pass wraps it so the
    normalization pipeline has a uniform call shape.

    Already idempotent + lane-aware.  This wrapper just delegates.
    """
    from .numeric_policy_pass import (
        propagate_numeric_policy as _propagate,
    )
    _propagate(fn)


def propagate_verification_facts(fn: GraphIRFunction) -> None:
    """Derive ``IROp.verification_facts`` from the function's lane
    + the op's catalog membership.

    Rules:

      * ``fn.lane == "clifford_jit"`` AND ``op.op_name`` starts with
        ``clifford_``                  → ``{"ga_only"}``
      * ``fn.lane == "complex_jit"`` AND ``op.op_name`` is in the
        holomorphic whitelist          → ``{"holomorphic"}``
      * ``fn.lane == "energy_jit"`` AND ``op.op_name`` starts with
        ``energy_`` or ``ebm_``        → ``{"energy_whitelisted"}``
      * Otherwise                       → leave empty

    The constrained-lane view adapters already stamp these on
    every op — but a ``@tessera.jit`` function that accidentally
    routes through a Clifford / Complex / Energy op would not.
    This pass closes that gap.

    Producers that already populated ``verification_facts`` keep
    their choice.  Idempotent: a non-empty set is left alone.
    """

    # Holomorphic whitelist — mirrors complex_jit.HOLOMORPHIC_OPS.
    # Inlined here so this module doesn't pull in complex_jit
    # (which imports ast_ir + dataclasses — keep normalization
    # cheap to import).
    _HOLOMORPHIC_OPS = frozenset({
        "complex_mul",
        "complex_exp",
        "complex_div",
        "mobius",
    })

    for op in fn.body:
        if op.verification_facts:
            continue
        name = op.op_name
        if fn.lane == "clifford_jit" and name.startswith("clifford_"):
            op.verification_facts = frozenset({"ga_only"})
        elif fn.lane == "complex_jit" and name in _HOLOMORPHIC_OPS:
            op.verification_facts = frozenset({"holomorphic"})
        elif fn.lane == "energy_jit" and (
            name.startswith("energy_") or name.startswith("ebm_")
        ):
            op.verification_facts = frozenset({"energy_whitelisted"})


# ─────────────────────────────────────────────────────────────────────
# The canonical pipeline.  Reordering, renaming, or removing entries
# requires updating BOTH the architecture doc § 3 AND the order test.
# ─────────────────────────────────────────────────────────────────────


NORMALIZATION_PIPELINE: tuple[Callable[[GraphIRFunction], None], ...] = (
    canonicalize_op_names,
    propagate_source_positions,
    set_lane_provenance,
    propagate_value_kinds,
    propagate_numeric_policy,
    propagate_verification_facts,
)


def run_normalization_pipeline(fn: GraphIRFunction) -> None:
    """Apply every pass in :data:`NORMALIZATION_PIPELINE` to ``fn``
    in declared order.

    No-op today because every pass body is still a stub.  Once
    bodies land, this becomes the canonical entry point for any
    consumer that wants a normalized Graph IR.
    """

    for pass_fn in NORMALIZATION_PIPELINE:
        pass_fn(fn)


__all__ = [
    "NORMALIZATION_PIPELINE",
    "canonicalize_op_names",
    "propagate_numeric_policy",
    "propagate_source_positions",
    "propagate_value_kinds",
    "propagate_verification_facts",
    "run_normalization_pipeline",
    "set_lane_provenance",
]
