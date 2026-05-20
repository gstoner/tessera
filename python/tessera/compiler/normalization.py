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

from .graph_ir import GraphIRFunction


# ─────────────────────────────────────────────────────────────────────
# Pass stubs.  Each function below is intentionally no-op pending its
# implementation commit.  Replacing a body is a one-line change; the
# pass name + signature is the stable contract.
# ─────────────────────────────────────────────────────────────────────


def canonicalize_op_names(fn: GraphIRFunction) -> None:
    """Strip the canonical ``tessera.`` prefix from every op name.

    ``tessera.matmul`` → ``matmul``.  Aliases that backend lookup
    re-adds (``complex_mobius``) are NOT introduced here — this pass
    only strips, never aliases.

    **Phase C body — TODO.**  Today this is a no-op stub.  The
    real implementation iterates ``fn.body`` and rewrites
    ``op.op_name`` in place.

    Idempotent: stripping ``"matmul"`` of its (absent) prefix
    leaves it ``"matmul"``.
    """
    # TODO(phase-c-body-1): implement.
    _ = fn  # silence unused-arg lint until the body lands


def propagate_source_positions(fn: GraphIRFunction) -> None:
    """Plumb AST ``lineno``/``col_offset`` into ``IROp.source_span``
    when the producer left it unset.

    The Python ``@tessera.jit`` AST visitor already populates this
    for most ops; this pass fills the gaps (e.g., synthetic ops
    introduced by canonicalization).

    **Phase C body — TODO.**

    Idempotent: ops that already carry a span are left untouched.
    """
    # TODO(phase-c-body-2): implement.
    _ = fn


def set_lane_provenance(fn: GraphIRFunction) -> None:
    """Ensure ``fn.lane`` is set to a real lane value.

    Default is already ``"tessera_jit"`` in the dataclass; this
    pass exists so future producers that forget the field still
    get a sensible lane.

    **Phase C body — TODO.**

    Idempotent: a function with a real lane is left alone.
    """
    # TODO(phase-c-body-3): implement.
    _ = fn


def propagate_value_kinds(fn: GraphIRFunction) -> None:
    """Stamp ``IROp.value_kind`` based on the op's catalog entry.

    ``matmul`` → ``tensor``, ``clifford_*`` → ``multivector``,
    ``complex_*`` → ``complex``, ``energy_*`` → ``energy``,
    ``ebm_*`` → ``energy``.

    Producers that already set ``value_kind`` (e.g., the constrained
    view adapters) keep their choice.  ``None`` is upgraded to a
    derived kind when the op is in the catalog; otherwise stays
    ``None``.

    **Phase C body — TODO.**

    Idempotent: ops with a non-None ``value_kind`` are skipped.
    """
    # TODO(phase-c-body-4): implement.
    _ = fn


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
    + the op's catalog category.

    A ``complex_jit`` function's ops inherit ``{"holomorphic"}``
    when the op is in ``HOLOMORPHIC_OPS``; otherwise they get
    no fact.  Same idea for Clifford / Energy.

    The constrained-lane view adapters already stamp this on
    every op — but a ``@tessera.jit`` function that accidentally
    calls a Clifford op would not.  This pass closes that gap.

    **Phase C body — TODO.**

    Idempotent: ops with non-empty ``verification_facts`` are
    skipped.
    """
    # TODO(phase-c-body-5): implement.
    _ = fn


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
