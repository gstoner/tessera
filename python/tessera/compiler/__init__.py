"""
tessera.compiler — frontend compiler passes for the Python API layer.

Components:
    constraints.py  — ConstraintSolver: Divisible, Range, Equal predicates
                       checked at @jit decoration time
    effects.py      — EffectLattice: infers random/io/memory/pure through
                       the call graph; enforces @deterministic contracts
    graph_ir.py     — Python → Graph IR lowering (emits object-backed MLIR text)
    schedule_ir.py  — Graph IR → Schedule IR lowering and verifier
    tile_ir.py      — Schedule IR → Tile IR lowering and verifier
    target_ir.py    — Tile IR → CPU/NVIDIA/Apple/ROCm Target IR lowering
    capabilities.py — shared target/op/runtime capability registry
    schedule_planner.py — legality/cost/search schedule planning contracts
    jit.py          — @jit and @kernel decorators that drive the pipeline

Build order for Phase 1:
    1. constraints.py (no deps)
    2. effects.py (no deps)
    3. graph_ir.py (depends on constraints, effects, distributed.*)
    4. jit.py (depends on graph_ir)
"""

from .constraints import ConstraintSolver, Divisible, Range, Equal, TesseraConstraintError
from .effects import EffectLattice, Effect, TesseraEffectError
from .graph_ir import GraphIRConstructionContext, KVCacheSpec, NumericPolicy, construct_mlir_module
from .schedule_ir import ScheduleIRModule, ScheduleFunction, ScheduleOp, lower_graph_to_schedule_ir
from .tile_ir import TileIRModule, TileFunction, TileOp, lower_schedule_to_tile_ir
from .target_ir import TargetIRModule, TargetFunction, TargetOp, lower_tile_to_target_ir
from .capabilities import (
    CAPABILITY_REGISTRY_VERSION,
    CapabilityResult,
    TargetCapability,
    get_target_capability,
    normalize_target,
    runtime_status,
    supports_op,
)
from .primitive_coverage import (
    CONTRACT_FIELDS,
    PrimitiveCoverage,
    all_primitive_coverages,
    coverage_for,
    coverage_summary,
    primitives_for_model_family,
    render_markdown,
)
from .legality import LegalityDiagnostic, LegalityResult, TensorContract, check_op_legality
from .schedule_planner import ScheduleCandidate, SchedulePlanner, SelectedSchedule, schedule_cache_key
from .gpu_smoke import SmokeResult, run_matmul_smoke
from .jit import jit, TesseraJitError
from .driver import CompileArtifactBundle, CompileRequest, CompileTraceEvent, compile_graph_module
from .frontend import FrontendSemanticError, FrontendSyntaxError, lower_text_to_graph_ir, parse_text
from .support import (
    OpSupport,
    TargetSupport,
    Tier,
    is_native_supported,
    known_targets,
    support,
    tier,
)
from .diagnostics import (
    ConstrainedDiagnosticCode,
    Diagnostic,
    DiagnosticCode,
    FallbackDecision,
    FallbackReason,
    FrontendDiagnosticCode,
    JitDiagnosticCode,
    SourceLocation,
    TesseraNativeRequiredError,
    classify_host,
)
from .symbol_table import SymbolEntry, SymbolTable
from . import frontend_lanes as lanes  # noqa: F401 — re-export as ts.compiler.lanes
from .frontend_lanes import FrontendLane, FrontendLaneSpec

__all__ = [
    "ConstraintSolver",
    "Divisible",
    "Range",
    "Equal",
    "TesseraConstraintError",
    "EffectLattice",
    "Effect",
    "TesseraEffectError",
    "NumericPolicy",
    "GraphIRConstructionContext",
    "construct_mlir_module",
    "KVCacheSpec",
    "ScheduleIRModule",
    "ScheduleFunction",
    "ScheduleOp",
    "lower_graph_to_schedule_ir",
    "TileIRModule",
    "TileFunction",
    "TileOp",
    "lower_schedule_to_tile_ir",
    "TargetIRModule",
    "TargetFunction",
    "TargetOp",
    "lower_tile_to_target_ir",
    "CAPABILITY_REGISTRY_VERSION",
    "CapabilityResult",
    "TargetCapability",
    "get_target_capability",
    "normalize_target",
    "runtime_status",
    "supports_op",
    "CONTRACT_FIELDS",
    "PrimitiveCoverage",
    "all_primitive_coverages",
    "coverage_for",
    "coverage_summary",
    "primitives_for_model_family",
    "render_markdown",
    "LegalityDiagnostic",
    "LegalityResult",
    "TensorContract",
    "check_op_legality",
    "ScheduleCandidate",
    "SchedulePlanner",
    "SelectedSchedule",
    "schedule_cache_key",
    "SmokeResult",
    "run_matmul_smoke",
    "jit",
    "TesseraJitError",
    "CompileArtifactBundle",
    "CompileRequest",
    "CompileTraceEvent",
    "compile_graph_module",
    "FrontendSemanticError",
    "FrontendSyntaxError",
    "lower_text_to_graph_ir",
    "parse_text",
    # Public support / tier query API (P0-1, 2026-05-19).  Thin
    # wrapper over audit.support_row_for + backend_manifest +
    # capabilities.  No parallel registry — these always reflect
    # the same data the support_table dashboard renders.
    "OpSupport",
    "TargetSupport",
    "Tier",
    "is_native_supported",
    "known_targets",
    "support",
    "tier",
    # Public diagnostic-code taxonomy (P0-2 / F1+G2, 2026-05-19).
    # Five vocabularies: JIT lane / textual DSL / constrained math /
    # fallback / source-location.
    "ConstrainedDiagnosticCode",
    "Diagnostic",
    "DiagnosticCode",
    "FallbackDecision",
    "FallbackReason",
    "FrontendDiagnosticCode",
    "JitDiagnosticCode",
    "SourceLocation",
    "TesseraNativeRequiredError",
    "classify_host",
    # F2 substrate (2026-05-19) — shared scoped symbol table for
    # every frontend lane.
    "SymbolEntry",
    "SymbolTable",
    # F3 + U1 + U2 (2026-05-19) — frontend-lane registry +
    # recommend() heuristic + .explain().lane surface.
    "FrontendLane",
    "FrontendLaneSpec",
    "lanes",
]
