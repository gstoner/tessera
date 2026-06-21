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
from .target_ir import (
    TargetIRModule,
    TargetFunction,
    TargetOp,
    annotate_target_ir_with_probes,
    lower_tile_to_target_ir,
)
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
from .numeric_policy_pass import (
    propagate_numeric_policy,
    propagate_numeric_policy_module,
    validate_numeric_policy_chain,
)
from .graph_ir_cache import (
    cache_stats as graph_ir_cache_stats,
    clear_graph_ir_cache,
)
from .dry_run import dry_run
from .normalization import (
    NORMALIZATION_PIPELINE,
    run_normalization_pipeline,
)
from . import frontend_lanes as lanes  # noqa: F401 — re-export as ts.compiler.lanes
from .frontend_lanes import FrontendLane, FrontendLaneSpec
from .cross_lane import (
    CrossLaneViolation,
    allowed_nestings,
    detect_violation as detect_cross_lane_violation,
    is_legal as is_cross_lane_legal,
)
from .ir_version import (
    GRAPH_IR_SCHEMA_VERSION,
    IR_VERSION_HISTORY,
    migrate as migrate_ir_module,
)
from .e2e_coverage import (
    E2ECoverageRow,
    E2EStatus,
    all_coverage_rows as all_e2e_coverage_rows,
    coverage_row_for as e2e_coverage_row_for,
    status_counts as e2e_status_counts,
)
from .profiling_plan import (
    IntraKernelProbe,
    ModelAnalyzerManifest,
    ModelAnalyzerSweep,
    ProfilerPlan,
    ProviderCapability,
    model_analyzer_manifest,
    normalize_profiler_target,
    plan_intra_kernel_probes,
    plan_profile,
    provider_capabilities,
    summarize_capabilities,
)
from .model_analyzer import (
    AnalyzerTrial,
    MODEL_ANALYZER_RESULT_SCHEMA_VERSION,
    load_model_analyzer_manifest,
    run_model_analyzer_manifest,
    write_model_analyzer_result,
)
from .apple_profiler_context import (
    APPLE_PROFILER_CONTEXT_SCHEMA_VERSION,
    AppleProfilerContext,
    apple_profiler_context_contract,
    apple_unified_memory_bandwidth_ceiling_gbs,
    classify_apple_profiler_context,
)
from .accelerator_profiler_context import (
    ACCELERATOR_PROFILER_CONTEXT_SCHEMA_VERSION,
    AcceleratorProfilerContext,
    accelerator_profiler_context_contract,
    classify_accelerator_profiler_context,
)
from .profiler_context import (
    PROFILER_CONTEXT_SCHEMA_VERSION,
    build_profiler_context_artifact,
    load_profiler_context_artifact,
    summarize_profiler_context,
    validate_profiler_context_artifact,
    write_profiler_context_artifact,
)
from .profiler_collectors import (
    PROFILER_COLLECTOR_STATUS_FILE,
    PROFILER_COLLECTOR_STATUS_HOST_METADATA,
    PROFILER_COLLECTOR_STATUS_MEASURED,
    PROFILER_COLLECTOR_STATUS_MOCK,
    PROFILER_COLLECTOR_STATUS_UNAVAILABLE,
    collect_profiler_context,
    load_context_samples_file,
    mock_profiler_context,
    sample_apple_system_context,
    sample_nvidia_nvml_context,
    sample_rocm_amdsmi_context,
)
from .profiler_provider_trace import (
    PROVIDER_TRACE_SCHEMA_VERSION,
    ProviderTraceRecord,
    build_provider_trace_artifact,
    load_provider_trace_input,
    normalize_cupti_activity_record,
    normalize_cupti_callback_record,
    normalize_metal_command_buffer_record,
    normalize_metal_counter_record,
    normalize_rocprofiler_activity_record,
    normalize_rocprofiler_api_record,
    normalize_rocprofiler_counter_record,
    normalize_rocprofiler_thread_trace_record,
    records_from_raw,
    summarize_provider_trace_records,
    validate_provider_trace_artifact,
    write_provider_trace_artifact,
)
from .profiler_provider_status import (
    PROFILER_PROVIDER_STATUS_SCHEMA_VERSION,
    collect_provider_status,
    provider_status_artifact,
    validate_provider_status_artifact,
)
from .profiler_trace_merge import (
    MERGED_PROFILER_TRACE_SCHEMA_VERSION,
    merge_profiler_traces,
    validate_merged_profiler_trace,
    write_merged_profiler_trace,
)

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
    "annotate_target_ir_with_probes",
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
    # G3 (2026-05-19) — numeric_policy propagation through Graph IR.
    "propagate_numeric_policy",
    "propagate_numeric_policy_module",
    "validate_numeric_policy_chain",
    # G4 (2026-05-19) — Graph IR memoization by source hash.
    "clear_graph_ir_cache",
    "graph_ir_cache_stats",
    # U4 (2026-05-19) — dry-run compile for static analysis.
    "dry_run",
    # Phase C skeleton (2026-05-20) — normalization pipeline order.
    # Pass bodies land in subsequent commits; the tuple ordering is
    # the load-bearing contract.
    "NORMALIZATION_PIPELINE",
    "run_normalization_pipeline",
    # Cross-lane composition rules (Issue 1, 2026-05-20).
    "CrossLaneViolation",
    "allowed_nestings",
    "detect_cross_lane_violation",
    "is_cross_lane_legal",
    # IR versioning (Issue 2, 2026-05-20).
    "GRAPH_IR_SCHEMA_VERSION",
    "IR_VERSION_HISTORY",
    "migrate_ir_module",
    # E2E coverage audit (Issue 4, 2026-05-20).
    "E2ECoverageRow",
    "E2EStatus",
    "all_e2e_coverage_rows",
    "e2e_coverage_row_for",
    "e2e_status_counts",
    "IntraKernelProbe",
    "ModelAnalyzerManifest",
    "ModelAnalyzerSweep",
    "ProfilerPlan",
    "ProviderCapability",
    "model_analyzer_manifest",
    "normalize_profiler_target",
    "plan_intra_kernel_probes",
    "plan_profile",
    "provider_capabilities",
    "summarize_capabilities",
    "AnalyzerTrial",
    "MODEL_ANALYZER_RESULT_SCHEMA_VERSION",
    "load_model_analyzer_manifest",
    "run_model_analyzer_manifest",
    "write_model_analyzer_result",
    "APPLE_PROFILER_CONTEXT_SCHEMA_VERSION",
    "AppleProfilerContext",
    "apple_profiler_context_contract",
    "apple_unified_memory_bandwidth_ceiling_gbs",
    "classify_apple_profiler_context",
    "ACCELERATOR_PROFILER_CONTEXT_SCHEMA_VERSION",
    "AcceleratorProfilerContext",
    "accelerator_profiler_context_contract",
    "classify_accelerator_profiler_context",
    "PROFILER_CONTEXT_SCHEMA_VERSION",
    "build_profiler_context_artifact",
    "load_profiler_context_artifact",
    "summarize_profiler_context",
    "validate_profiler_context_artifact",
    "write_profiler_context_artifact",
    "PROFILER_COLLECTOR_STATUS_FILE",
    "PROFILER_COLLECTOR_STATUS_HOST_METADATA",
    "PROFILER_COLLECTOR_STATUS_MEASURED",
    "PROFILER_COLLECTOR_STATUS_MOCK",
    "PROFILER_COLLECTOR_STATUS_UNAVAILABLE",
    "collect_profiler_context",
    "load_context_samples_file",
    "mock_profiler_context",
    "sample_apple_system_context",
    "sample_nvidia_nvml_context",
    "sample_rocm_amdsmi_context",
    "PROVIDER_TRACE_SCHEMA_VERSION",
    "ProviderTraceRecord",
    "build_provider_trace_artifact",
    "load_provider_trace_input",
    "normalize_cupti_activity_record",
    "normalize_cupti_callback_record",
    "normalize_metal_command_buffer_record",
    "normalize_metal_counter_record",
    "normalize_rocprofiler_activity_record",
    "normalize_rocprofiler_api_record",
    "normalize_rocprofiler_counter_record",
    "normalize_rocprofiler_thread_trace_record",
    "records_from_raw",
    "summarize_provider_trace_records",
    "validate_provider_trace_artifact",
    "write_provider_trace_artifact",
    "PROFILER_PROVIDER_STATUS_SCHEMA_VERSION",
    "collect_provider_status",
    "provider_status_artifact",
    "validate_provider_status_artifact",
    "MERGED_PROFILER_TRACE_SCHEMA_VERSION",
    "merge_profiler_traces",
    "validate_merged_profiler_trace",
    "write_merged_profiler_trace",
]
