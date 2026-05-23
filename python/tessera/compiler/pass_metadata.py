"""Arch-6 (2026-05-22) — pass metadata layer (Layer B).

The companion to Arch-1 (diagnostic codes) and Arch-5 (pipelines).
Where Arch-1 catalogues the *errors* a pass can emit and Arch-5
catalogues the *named pipelines* a pass appears in, Arch-6 captures
metadata about each *individual pass*:

  * Input / output dialect requirements (what must be loaded
    before/after).
  * Required / preserved op attributes (e.g.,
    ``tessera.dim_bindings`` for SymbolicDimEquality).
  * Diagnostic codes the pass emits (cross-referenced into Arch-1).
  * Ordering constraints (``must_run_after`` / ``can_run_after``).

This is intentionally lighter than Arch-5: only the ~15 passes that
appear in named pipelines need entries here.  Most one-off
transformation passes don't need this metadata — their behavior is
captured by lit fixtures + the pipeline they're part of.

The drift gate at ``tests/unit/test_pass_metadata.py`` cross-checks:

  * Every diagnostic code referenced is in Arch-1's REGISTERED_CODES.
  * Every must_run_after / can_run_after target is itself a Layer-B
    pass.
  * Every input_dialect / output_dialect is in REGISTERED_DIALECTS
    (or a standard MLIR dialect).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PassMetadata:
    """Layer-B metadata for a single named MLIR pass.

    Fields
    ------
    name
        Pass name as registered via ``OPT_PASS_REGISTRATION`` (the
        string used in ``--pass-pipeline='builtin.module(name)'``).
    cpp_class
        The C++ class name (e.g., ``SymbolicDimEquality``).  Used by
        the drift gate to find the implementation.
    summary
        One-sentence description of what the pass does.
    input_dialects
        Dialect names that must be loaded before the pass runs.
    output_dialects
        Dialect names the pass produces.  Often the same as input
        for verifier-style passes.
    required_attrs
        Op-level attribute names the pass reads (e.g.,
        ``tessera.dim_bindings`` on ``func.func``).
    preserved_attrs
        Op-level attribute names the pass preserves (it doesn't
        rewrite or drop them).
    diagnostic_codes
        Diagnostic codes the pass can emit.  Each must be in
        :data:`tessera.compiler.diagnostic_codes.REGISTERED_CODES`.
    can_run_after
        Passes whose output is compatible input.  Empty tuple = no
        ordering constraint.
    must_run_after
        Passes that MUST have already run.  E.g.,
        ``DistributionLowering`` must precede ``SymbolicDimEquality``
        because the latter reads ``tessera.dim_sizes`` that the
        former injects.
    pass_kind
        ``"verifier"`` (read-only, emits diagnostics) /
        ``"transform"`` (mutates IR) / ``"lowering"`` (translates
        between dialects).
    sprint
        Sprint label for archaeology.
    """

    name: str
    cpp_class: str
    summary: str
    input_dialects: tuple[str, ...]
    output_dialects: tuple[str, ...]
    required_attrs: tuple[str, ...] = ()
    preserved_attrs: tuple[str, ...] = ()
    diagnostic_codes: tuple[str, ...] = ()
    can_run_after: tuple[str, ...] = ()
    must_run_after: tuple[str, ...] = ()
    pass_kind: str = "transform"
    sprint: str = ""


# ─────────────────────────────────────────────────────────────────────────
# Registry — keep alphabetised by pass name.
# ─────────────────────────────────────────────────────────────────────────


REGISTERED_PASSES: tuple[PassMetadata, ...] = (
    PassMetadata(
        name="tessera-distribution-lower",
        cpp_class="DistributionLoweringPass",
        summary=(
            "Lowers `tessera.shard` into `schedule.mesh.define` + "
            "`schedule.mesh.region` ops and injects `tessera.dim_sizes` "
            "on func.func from the mesh dimensions."
        ),
        input_dialects=("tessera",),
        output_dialects=("tessera", "schedule.mesh"),
        required_attrs=(),
        preserved_attrs=("tessera.dim_bindings", "tessera.arg_dim_names"),
        diagnostic_codes=(),
        must_run_after=("tessera-effect-annotate",),
        pass_kind="transform",
        sprint="Phase 2",
    ),
    PassMetadata(
        name="tessera-effect-annotate",
        cpp_class="EffectAnnotationPass",
        summary=(
            "Walks the IR + annotates each func.func with "
            "`tessera.effect = pure|random|memory|io|top` using the "
            "EffectLattice."
        ),
        input_dialects=("tessera",),
        output_dialects=("tessera",),
        preserved_attrs=("tessera.dim_bindings", "tessera.arg_dim_names",
                         "tessera.dim_sizes"),
        diagnostic_codes=(),
        pass_kind="transform",
        sprint="Phase 2",
    ),
    PassMetadata(
        name="tessera-layout-legality",
        cpp_class="LayoutLegalityPass",
        summary=(
            "Verifies `tessera.layout` string attributes are in the "
            "canonical 8-name accept-set and that matmul operand "
            "layouts are within matmul's stricter {row_major, col_major} "
            "accept-set."
        ),
        input_dialects=("tessera",),
        output_dialects=("tessera",),
        diagnostic_codes=(
            "LAYOUT_LEGALITY_UNKNOWN_LAYOUT",
            "LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH",
        ),
        pass_kind="verifier",
        sprint="V2 + V4a",
    ),
    PassMetadata(
        name="tessera-symdim-equality",
        cpp_class="SymbolicDimEquality",
        summary=(
            "Verifies function-level `tessera.dim_bindings` equations + "
            "per-op dim-name contracts (reshape / transpose / matmul), "
            "with SSA-value propagation, sum-of-products affine "
            "reasoning, interprocedural cross-checks via func.call, "
            "and scf.for/scf.if region recursion."
        ),
        input_dialects=("tessera", "func", "scf"),
        output_dialects=("tessera", "func", "scf"),
        required_attrs=(
            "tessera.dim_bindings",
            "tessera.dim_sizes",
            "tessera.arg_dim_names",
        ),
        preserved_attrs=(
            "tessera.dim_bindings",
            "tessera.dim_sizes",
            "tessera.arg_dim_names",
        ),
        diagnostic_codes=(
            "SYMDIM_BINDING_VIOLATION",
            "SYMDIM_CALL_ARG_MISMATCH",
            "SYMDIM_FLOW_INCONSISTENCY",
            "SYMDIM_IF_BRANCH_MISMATCH",
            "SYMDIM_LOOP_YIELD_MISMATCH",
            "SYMDIM_MATMUL_CONTRACT_VIOLATION",
            "SYMDIM_RESHAPE_VIOLATION",
            "SYMDIM_TRANSPOSE_VIOLATION",
        ),
        # V6b: inserted after DistributionLowering in the named
        # pipelines because the latter injects tessera.dim_sizes.
        must_run_after=("tessera-distribution-lower",),
        pass_kind="verifier",
        sprint="V5 + V2-flow + V3a + V3b + V3c",
    ),
)


# ─────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────


def all_pass_names() -> tuple[str, ...]:
    return tuple(sorted(p.name for p in REGISTERED_PASSES))


def pass_lookup(name: str) -> PassMetadata | None:
    for spec in REGISTERED_PASSES:
        if spec.name == name:
            return spec
    return None


def passes_emitting_code(code: str) -> tuple[PassMetadata, ...]:
    """Return passes that emit a given diagnostic code.  Cross-ref
    convenience for the diagnostic-code dashboard."""
    return tuple(p for p in REGISTERED_PASSES if code in p.diagnostic_codes)


__all__ = [
    "PassMetadata",
    "REGISTERED_PASSES",
    "all_pass_names",
    "pass_lookup",
    "passes_emitting_code",
]
