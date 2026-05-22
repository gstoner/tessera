"""Sprint S5 (2026-05-22) — structural guard for SHAPE_SYSTEM.md §11
MLIR Verifier Gap Enumeration.

The spec's §11.1 per-contract enforcement matrix names specific files
as the canonical evidence for each contract. If a file moves, is
renamed, or loses the API the spec cites, this test fails — making
the doc drift mechanical to catch rather than discovering it during a
spec audit months later.

This is a *thin* contract test: it verifies file presence + class /
function presence for the surfaces the spec promises. It does NOT
re-verify the contracts themselves (those have their own dedicated
test files; this guard exists so the spec's evidence pointers don't
silently rot).
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Evidence file presence (§11.3)
# ─────────────────────────────────────────────────────────────────────────────


_EVIDENCE_FILES = [
    "python/tessera/compiler/constraints.py",
    "python/tessera/shape.py",
    "src/compiler/diagnostics/ShapeInferencePass.cpp",
    "python/tessera/compiler/jit.py",
    "python/tessera/compiler/memory_verifier.py",
    "tests/unit/test_constraints.py",
    "tests/unit/test_shape_system_foundation.py",
    "tests/unit/test_shape_inference.py",
    "tests/unit/test_dynamic_shapes.py",
    "tests/unit/test_memory_verifier.py",
]


@pytest.mark.parametrize("relpath", _EVIDENCE_FILES)
def test_shape_system_spec_evidence_file_exists(relpath: str) -> None:
    """Every file SHAPE_SYSTEM.md §11.3 cites as canonical evidence
    must exist."""
    path = REPO_ROOT / relpath
    assert path.exists(), (
        f"SHAPE_SYSTEM.md §11.3 cites {relpath!r} as evidence, but "
        f"the file is missing. Either restore the file or update the "
        f"spec's §11.3 evidence pointers."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Python surface promised by §11.1
# ─────────────────────────────────────────────────────────────────────────────


def test_constraint_solver_predicates_present() -> None:
    """§11.1 names Equal / Divisible / Range / Derived as the
    decoration-time predicates."""
    from tessera.compiler.constraints import (
        ConstraintSolver, Divisible, Range, Equal,
    )
    assert ConstraintSolver is not None
    assert Divisible is not None
    assert Range is not None
    assert Equal is not None


def test_shape_module_classes_present() -> None:
    """§11.1 / §6 / §9 reference these public classes."""
    import tessera.shape as shape_mod
    for name in (
        "Dim", "DimProduct", "Layout", "Shape", "ShapeShard",
        "ScheduleFeasibility", "ShapeConstraintGraph",
        "RuntimeShapeWitness",
    ):
        assert hasattr(shape_mod, name), (
            f"tessera.shape.{name} missing — SHAPE_SYSTEM.md references "
            f"it as a public class"
        )


def test_shape_module_helpers_present() -> None:
    """§6 / §11.1 enumerate these helpers as the Python shape API."""
    import tessera.shape as shape_mod
    for name in (
        "sym", "dim", "broadcast_shape", "matmul_shape",
        "reshape_shape", "check_shard", "check_schedule_tile",
        "check_shapes",
    ):
        assert hasattr(shape_mod, name), (
            f"tessera.shape.{name} missing — SHAPE_SYSTEM.md §6 / §11.1 "
            f"references it"
        )


def test_jit_enforces_call_time_constraints() -> None:
    """§11.1 says `JitFn._enforce_call_time_constraints` is the
    call-time enforcement entry point."""
    from tessera.compiler.jit import JitFn
    assert hasattr(JitFn, "_enforce_call_time_constraints"), (
        "JitFn._enforce_call_time_constraints is missing — "
        "SHAPE_SYSTEM.md §11.1 names it as the canonical call-time "
        "enforcement entry"
    )


def test_memory_verifier_canonical_surface_present() -> None:
    """§11.1 says the Tile-IR memory-model verifier covers async
    copy / mbarrier / atomic order-scope / fence scope / determinism."""
    from tessera.compiler.memory_verifier import (
        verify_memory_model,
        VALID_ATOMIC_ORDERS, VALID_SYNC_SCOPES, VALID_ATOMIC_OPS,
        NONDETERMINISTIC_FLOAT_DTYPES, REDUCTION_ATOMIC_OPS,
    )
    assert callable(verify_memory_model)
    assert "acq_rel" in VALID_ATOMIC_ORDERS
    assert "device" in VALID_SYNC_SCOPES
    assert "cas" in VALID_ATOMIC_OPS
    assert "fp32" in NONDETERMINISTIC_FLOAT_DTYPES
    assert "add" in REDUCTION_ATOMIC_OPS


# ─────────────────────────────────────────────────────────────────────────────
# MLIR-side surface promised by §11.1
# ─────────────────────────────────────────────────────────────────────────────


def test_shape_inference_pass_has_per_op_rules() -> None:
    """§11.1 names ShapeInferencePass.cpp as the MLIR-PASS evidence
    for per-op shape inference (matmul, elementwise, flash_attn,
    reshape, transpose, concat, slice, reduce)."""
    pass_cpp = REPO_ROOT / "src" / "compiler" / "diagnostics" / "ShapeInferencePass.cpp"
    text = pass_cpp.read_text()
    for rule_name in (
        "inferMatmul", "inferElementwise", "inferFlashAttn",
        "inferReshape", "inferTranspose", "inferConcat",
        "inferSlice", "inferReduce",
    ):
        assert rule_name in text, (
            f"ShapeInferencePass.cpp is missing {rule_name!r} — "
            "SHAPE_SYSTEM.md §11.1 cites the 8 per-op inference rules"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gap list — pin the canonical list of named gaps so silent "fixed but
# spec not updated" drift gets caught
# ─────────────────────────────────────────────────────────────────────────────


# Sprint V1+V2+V3 (2026-05-22) re-shaped §11.2.  Three items are
# closed and one remains planned.  The phrases below pin both the
# closures (so a regression on the verifier code triggers here) and
# the remaining planned item.
_CANONICAL_GAP_PHRASES_CLOSURE = (
    # Sprint V1 — partial closure of the per-op verifier coverage gap.
    # Test against tokens that always appear on a single line in the
    # spec (avoid multi-word phrases that may straddle a markdown
    # word-wrap).
    "Sprint V1 closure",
    "TransposeOp",
    "MoeDispatchOp",
    # Sprint V2 — LayoutLegalityPass skeleton + first rule.
    "tessera-layout-legality",
    "LAYOUT_LEGALITY_UNKNOWN_LAYOUT",
    # Sprint V3 — target-aware FlashAttnOp head_dim ceiling.
    "Sprint V3 closure",
    "tessera.target_sm",
)

_CANONICAL_GAP_PHRASES_PLANNED: tuple[str, ...] = (
    # Sprint V5 (2026-05-22) closed the symbolic-dim-equality gap.
    # All four originally-named MLIR-verifier gaps from the
    # 2026-05-22 SHAPE_SYSTEM.md §11.2 set are now closed at V1.
    # If a new gap is introduced or a partial closure regresses,
    # add a phrase here AND record the open work in §11.2.
)


# Sprint V5 (2026-05-22) — V5 closure must be recorded in §11.2.
_CANONICAL_V5_CLOSURE_PHRASES = (
    "Sprint V5 closure",
    "SYMDIM_BINDING_VIOLATION",
    "SYMDIM_MATMUL_CONTRACT_VIOLATION",
    "tessera-symdim-equality",
)


def test_spec_records_sprint_v1_v2_v3_closures() -> None:
    """SHAPE_SYSTEM.md §11.2 must record the 3 closures Sprint V1+V2+V3
    landed.  If a closure regresses (e.g. someone reverts the
    `LayoutLegalityPass.cpp` source), this test fails — pointing the
    fixer at the right spec section."""
    spec = (REPO_ROOT / "docs" / "spec" / "SHAPE_SYSTEM.md").read_text()
    missing = [g for g in _CANONICAL_GAP_PHRASES_CLOSURE if g not in spec]
    assert not missing, (
        f"SHAPE_SYSTEM.md §11.2 is missing Sprint V1+V2+V3 closure "
        f"phrases: {missing}.  If a closure was reverted, restore the "
        f"verifier code AND the spec; do not edit one without the "
        f"other."
    )


def test_no_remaining_planned_gaps() -> None:
    """After Sprint V5 closed the 4th and final MLIR-verifier gap,
    the planned-gap list is empty.  If a regression re-opens a gap,
    add a phrase to `_CANONICAL_GAP_PHRASES_PLANNED` AND record the
    open work in §11.2 — keep doc + test in lockstep."""
    if not _CANONICAL_GAP_PHRASES_PLANNED:
        return  # All four originally-named gaps closed at V5.
    spec = (REPO_ROOT / "docs" / "spec" / "SHAPE_SYSTEM.md").read_text()
    missing = [g for g in _CANONICAL_GAP_PHRASES_PLANNED if g not in spec]
    assert not missing, (
        f"SHAPE_SYSTEM.md §11.2 is missing remaining planned gaps: "
        f"{missing}.  If the gap was closed, update §11.1 with the "
        f"closing evidence and update this test."
    )


def test_spec_records_sprint_v5_closure() -> None:
    """Sprint V5 (2026-05-22) closed the 4th MLIR-verifier gap.
    The spec must reference the closure with its specific evidence
    (pass name + 2 stable diagnostic codes lit-exercised)."""
    spec = (REPO_ROOT / "docs" / "spec" / "SHAPE_SYSTEM.md").read_text()
    missing = [g for g in _CANONICAL_V5_CLOSURE_PHRASES if g not in spec]
    assert not missing, (
        f"SHAPE_SYSTEM.md §11.2 is missing Sprint V5 closure phrases: "
        f"{missing}.  If V5 was reverted, restore the verifier code "
        f"(SymbolicDimEqualityPass.cpp) AND the spec; do not edit one "
        f"without the other."
    )
