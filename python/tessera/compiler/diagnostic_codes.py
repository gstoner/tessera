"""Arch-1 (2026-05-22) — Central registry of MLIR verifier / pass diagnostic codes.

Before this sprint, diagnostic codes (e.g., ``SYMDIM_BINDING_VIOLATION``,
``QUEUE_PUSH_QUEUE_PROVENANCE``) were defined only at the C++ ``emitOpError``
site.  Discovering them required ``grep`` across ``src/``; their meaning lived
in the surrounding code comments and in sprint-specific lit fixtures.

This module is the single Python-side source of truth that:

  * Names every code Tessera emits, with severity / pass-origin /
    human summary / fix-hint / spec back-link.
  * Lets a drift gate cross-check: every ``"FOO_BAR_BAZ:`` substring in
    a C++ ``emitOpError`` is in the registry, and every registered code
    appears in at least one C++ file.

The registry is consulted by:

  * ``tests/unit/test_diagnostic_codes.py`` (drift gate).
  * ``docs/audit/diagnostic_codes.md`` (generated dashboard).
  * Future ``JitFn.explain()`` extensions that translate raw MLIR
    diagnostic strings to actionable Python guidance.

The format of a diagnostic code in C++ is:

    op->emitOpError("CODE_NAME: human-readable detail...")

The drift gate matches the all-caps prefix before the first ``:``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticCode:
    """One MLIR verifier / pass diagnostic code.

    Fields
    ------
    code
        The all-caps token before the ``:`` in ``emitOpError`` calls.
    pass_origin
        Symbolic name of the pass or verifier that emits the code.
        Use the C++ class name (``SymbolicDimEquality``) or a
        ``Dialect.OpName`` form (``Queue.PushOp::verify``).
    severity
        ``"error"`` (default — failure of ``verify()`` / pass) or
        ``"warning"`` (advisory; rarely used today).
    summary
        One-sentence human-readable explanation of what the code means.
    fix_hint
        Concrete action the user can take to silence the diagnostic.
    spec
        Optional path + section into the spec corpus that documents the
        invariant the code enforces (e.g.,
        ``"docs/spec/SHAPE_SYSTEM.md §11.2"``).
    sprint
        Which sprint introduced the code, for archaeological context.
    """

    code: str
    pass_origin: str
    severity: str
    summary: str
    fix_hint: str
    spec: str | None
    sprint: str


# ─────────────────────────────────────────────────────────────────────────
# Registry — keep alphabetised by code for easy review.
# ─────────────────────────────────────────────────────────────────────────

REGISTERED_CODES: tuple[DiagnosticCode, ...] = (
    # ── LayoutLegalityPass (V2 + V4a) ────────────────────────────────────
    DiagnosticCode(
        code="LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH",
        pass_origin="LayoutLegalityPass",
        severity="error",
        summary=(
            "A `tessera.matmul` operand's producer carries a `tessera.layout` "
            "attribute outside matmul's accept-set {row_major, col_major}, "
            "and no intervening cast converts it."
        ),
        fix_hint=(
            "Insert a `tessera.cast` that converts the producer's layout to "
            "either row_major or col_major before the matmul."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V4a",
    ),
    DiagnosticCode(
        code="LAYOUT_LEGALITY_UNKNOWN_LAYOUT",
        pass_origin="LayoutLegalityPass",
        severity="error",
        summary=(
            "A `tessera.cast` op carries a `tessera.layout` string attribute "
            "that is not in the canonical 8-name accept-set "
            "{row_major, col_major, nhwc, nchw, bhsd, tile, bsr, packed}."
        ),
        fix_hint=(
            "Use one of the canonical layout names listed in "
            "SHAPE_SYSTEM.md §2.1, or update the accept-set if a new "
            "canonical layout is needed."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §2.1",
        sprint="V2",
    ),

    # ── Queue dialect verifiers (V8) ─────────────────────────────────────
    DiagnosticCode(
        code="QUEUE_CREATE_OPERAND_COUNT",
        pass_origin="Queue.CreateOp::verify",
        severity="error",
        summary=(
            "`tessera.queue.create` must have zero operands; future TD "
            "revisions that accidentally add one are caught at the IR layer."
        ),
        fix_hint=(
            "Remove the extra operand or align Queue.td with the canonical "
            "zero-operand contract."
        ),
        spec=None,
        sprint="V8",
    ),
    DiagnosticCode(
        code="QUEUE_POP_QUEUE_PROVENANCE",
        pass_origin="Queue.PopOp::verify",
        severity="error",
        summary=(
            "The queue handle operand of a `tessera.queue.pop` is not "
            "defined by a `tessera.queue.create` — data-flow malformed."
        ),
        fix_hint=(
            "Ensure the queue handle traces back to a `tessera.queue.create` "
            "op (not a function argument, not a block argument)."
        ),
        spec=None,
        sprint="V8",
    ),
    DiagnosticCode(
        code="QUEUE_POP_TILE_TYPE",
        pass_origin="Queue.PopOp::verify",
        severity="error",
        summary=(
            "The result of a `tessera.queue.pop` is neither a ranked tensor "
            "nor a memref — the TD's `AnyType` was too permissive."
        ),
        fix_hint=(
            "Constrain the result type to a ranked tensor or memref shape."
        ),
        spec=None,
        sprint="V8",
    ),
    DiagnosticCode(
        code="QUEUE_POP_TOKEN_PROVENANCE",
        pass_origin="Queue.PopOp::verify",
        severity="error",
        summary=(
            "The dependency token operand of a `tessera.queue.pop` is not "
            "defined by a `tessera.queue.push` — the token must come from a "
            "matching push."
        ),
        fix_hint=(
            "Wire the dep token from a preceding `tessera.queue.push`; "
            "function-argument tokens are not legal in FA-4 IR."
        ),
        spec=None,
        sprint="V8",
    ),
    DiagnosticCode(
        code="QUEUE_PUSH_QUEUE_PROVENANCE",
        pass_origin="Queue.PushOp::verify",
        severity="error",
        summary=(
            "The queue handle operand of a `tessera.queue.push` is not "
            "defined by a `tessera.queue.create`."
        ),
        fix_hint=(
            "Trace the queue handle back to a `tessera.queue.create` op."
        ),
        spec=None,
        sprint="V8",
    ),
    DiagnosticCode(
        code="QUEUE_PUSH_TILE_TYPE",
        pass_origin="Queue.PushOp::verify",
        severity="error",
        summary=(
            "The tile operand of `tessera.queue.push` is neither a ranked "
            "tensor nor a memref."
        ),
        fix_hint=(
            "Pass a tile-shaped value (ranked tensor / memref) — scalars "
            "and opaque tokens are not legal queue payloads."
        ),
        spec=None,
        sprint="V8",
    ),

    # ── SymbolicDimEqualityPass family (V5 + V2-flow + V3a + V3b + V3c) ──
    DiagnosticCode(
        code="SYMDIM_BINDING_VIOLATION",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "A function-level `tessera.dim_bindings` equation (e.g., "
            "`D = H * Dh + K`) is contradicted by the function's "
            "`tessera.dim_sizes` (the concrete sizes evaluate to a "
            "different value than the LHS claims)."
        ),
        fix_hint=(
            "Either correct the concrete sizes in `tessera.dim_sizes` to "
            "match the binding, or update the binding equation to reflect "
            "the actual shape relationship."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V5",
    ),
    DiagnosticCode(
        code="SYMDIM_CALL_ARG_MISMATCH",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "A `func.call` site passes operands whose propagated dim-names "
            "disagree with the callee's declared `tessera.arg_dim_names`."
        ),
        fix_hint=(
            "Update the caller to pass values with matching dim-names, or "
            "update the callee's `tessera.arg_dim_names` declaration."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V3b",
    ),
    DiagnosticCode(
        code="SYMDIM_FLOW_INCONSISTENCY",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "SSA-value flow-propagated dim-names disagree with an explicit "
            "per-op `tessera.dim_names_in` / `tessera.dim_names_out` / "
            "`tessera.dim_names_lhs` / `tessera.dim_names_rhs` annotation."
        ),
        fix_hint=(
            "Either remove the explicit annotation (let propagation infer) "
            "or correct it to match the propagated names."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V2-flow",
    ),
    DiagnosticCode(
        code="SYMDIM_IF_BRANCH_MISMATCH",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "An `scf.if` op's then-branch and else-branch yield values "
            "with different propagated dim-names for the same result "
            "position."
        ),
        fix_hint=(
            "Make both branches yield values that share the same dim-name "
            "structure (transpose / reshape in the branch as needed)."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V3c",
    ),
    DiagnosticCode(
        code="SYMDIM_LOOP_YIELD_MISMATCH",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "An `scf.for` op's `scf.yield` operand carries dim-names that "
            "differ from the corresponding iter_arg's dim-names — the loop "
            "is not name-invariant."
        ),
        fix_hint=(
            "Restructure the body so the yielded value preserves the "
            "iter_arg's dim-name ordering (no transpose, or undo the "
            "transpose before yielding)."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V3c",
    ),
    DiagnosticCode(
        code="SYMDIM_MATMUL_CONTRACT_VIOLATION",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "A `tessera.matmul` op declares `tessera.dim_names_lhs` and "
            "`tessera.dim_names_rhs` whose contracting symbols disagree "
            "(lhs.back() != rhs.front())."
        ),
        fix_hint=(
            "Rename one side's contracting dim so both ends agree on the "
            "K symbol, or fix the per-op annotation."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V5",
    ),
    DiagnosticCode(
        code="SYMDIM_RESHAPE_VIOLATION",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "A `tessera.reshape` op's `tessera.dim_names_in` and "
            "`tessera.dim_names_out` resolve to different element counts "
            "given the function's `tessera.dim_sizes` + bindings — the "
            "reshape cannot hold."
        ),
        fix_hint=(
            "Fix the dim_names list so the product of resolved sizes "
            "matches on both sides, or correct dim_sizes if the symbolic "
            "model is wrong."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V5",
    ),
    DiagnosticCode(
        code="SYMDIM_TRANSPOSE_VIOLATION",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "A `tessera.transpose` op's `tessera.dim_names_in` and "
            "`tessera.dim_names_out` are not a permutation of each other."
        ),
        fix_hint=(
            "Adjust the output names so they're a reordering of the input "
            "names (same multiset)."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V5",
    ),
)


# ─────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────


def all_codes() -> tuple[str, ...]:
    """Return all registered code names, sorted."""
    return tuple(sorted(c.code for c in REGISTERED_CODES))


def code_lookup(code: str) -> DiagnosticCode | None:
    """Look up a single code by name. Returns None if not registered."""
    for entry in REGISTERED_CODES:
        if entry.code == code:
            return entry
    return None


def codes_by_pass(pass_origin: str) -> tuple[DiagnosticCode, ...]:
    """Return all codes emitted by a given pass / verifier."""
    return tuple(c for c in REGISTERED_CODES if c.pass_origin == pass_origin)


def codes_by_sprint(sprint: str) -> tuple[DiagnosticCode, ...]:
    """Return all codes introduced by a given sprint label."""
    return tuple(c for c in REGISTERED_CODES if c.sprint == sprint)


__all__ = [
    "DiagnosticCode",
    "REGISTERED_CODES",
    "all_codes",
    "code_lookup",
    "codes_by_pass",
    "codes_by_sprint",
]
