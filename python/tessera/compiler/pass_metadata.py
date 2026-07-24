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
        name="rocm-materialize-dynamic-lds",
        cpp_class="ROCMDynamicLDS",
        summary=(
            "Colors runtime-sized LLVM addrspace(3) byte arenas into "
            "SSA-lifetime interference slots backed by one launch-sized "
            "external ROCm LDS symbol."
        ),
        input_dialects=("llvm",),
        output_dialects=("llvm",),
        diagnostic_codes=("ROCM_DYNAMIC_LDS_SIZE_NOT_KERNEL_ARGUMENT",),
        pass_kind="lowering",
        sprint="CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24",
    ),
    PassMetadata(
        name="rocm-wave-lds-legality",
        cpp_class="ROCMWaveLdsLegalityPass",
        summary=(
            "ROCm Tile-IR legality gate: rejects NVIDIA-only TMA/TMEM/mbarrier "
            "semantics, missing waitcnt(vmcnt) before LDS-dependent matrix ops, "
            "and overlapping LDS writes without an intervening wait/barrier."
        ),
        input_dialects=("tile", "tessera_rocm", "func"),
        output_dialects=("tile", "tessera_rocm", "func"),
        required_attrs=("tile.buf", "tile.layout", "tile.barrier"),
        diagnostic_codes=(
            "ROCM_WAVE_LDS_MISSING_WAITCNT",
            "ROCM_WAVE_LDS_OVERLAPPING_WRITE",
            "ROCM_WAVE_LDS_UNSUPPORTED_BARRIER_KIND",
            "ROCM_WAVE_LDS_UNSUPPORTED_TMEM",
        ),
        must_run_after=("rocm-wave-lds-pipeline",),
        pass_kind="verifier",
        sprint="ROCm Tile-IR convergence",
    ),
    PassMetadata(
        name="rocm-wave-lds-pipeline",
        cpp_class="ROCMWaveLdsPipelinePass",
        summary=(
            "ROCm planner marker pass: annotates shared Tile IR with AMD-native "
            "LDS buffer refs, lds/wave layouts, waitcnt intent, and candidate "
            "pipeline-depth metadata before lower-tile-to-rocm."
        ),
        input_dialects=("tile", "func"),
        output_dialects=("tile", "func"),
        preserved_attrs=("numeric_policy", "tessera.storage_pack"),
        pass_kind="transform",
        sprint="ROCm Tile-IR convergence",
    ),
    PassMetadata(
        name="tessera-activation-rematerialization",
        cpp_class="ActivationRematerializationPass",
        summary=(
            "Selects and clones pure activation producers at backward "
            "consumers under an explicit or model/device-derived memory budget."
        ),
        input_dialects=("func",),
        output_dialects=("func",),
        required_attrs=(
            "tessera.autodiff.phase",
            "tessera.device_memory_capacity_bytes",
            "tessera.device_memory_reserve_basis_points",
            "tessera.model.parameter",
            "tessera.model.parameter_bytes_bound",
            "tessera.model_gradient_copies",
            "tessera.model_optimizer_state_copies",
            "tessera.model_persistent_bytes",
            "tessera.remat_budget_mb",
            "tessera.remat_cost_ns",
        ),
        preserved_attrs=("tessera.autodiff.phase",),
        diagnostic_codes=(
            "REMAT_EFFECTFUL",
            "REMAT_MODEL_BUDGET_INVALID",
            "REMAT_NON_CLONABLE",
        ),
        pass_kind="transform",
        sprint="CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24",
    ),
    PassMetadata(
        name="tessera-apple-materialize-layout-casts",
        cpp_class="MaterializeGraphLayoutToApplePass",
        summary=(
            "Consumes row-major/BHSD/NHWC Graph layout casts as indexed "
            "Apple runtime operand-binding contracts and rejects unsupported "
            "physical reinterpretation."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "func"),
        required_attrs=("tessera.layout",),
        preserved_attrs=("tessera.source_layout",),
        diagnostic_codes=(),
        pass_kind="lowering",
        sprint="CORE-COMPILER-FOLLOWON",
    ),
    PassMetadata(
        name="tessera-compute-legalize",
        cpp_class="ComputeLegalize",
        summary=(
            "C4 (TIRx): stamps `numeric_policy.accum` (fp32, or int32 for "
            "int4/int8) on any op whose storage is reduced-precision and lacks "
            "an accumulator — Decision #15a as an early rewrite. Default-on "
            "for x86/NVIDIA and forced on by ROCm's owned backend pipeline; "
            "runs before IRContractLegality so the stamped accum passes the "
            "contract."
        ),
        input_dialects=("tessera",),
        output_dialects=("tessera",),
        diagnostic_codes=(),
        pass_kind="transform",
        sprint="C4 (TIRx)",
    ),
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
            "canonical 8-name accept-set and that GEMM, convolution, "
            "attention, and last-axis reduction operands are within their "
            "consumer-specific accept-sets."
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
        name="tessera-nvidia-materialize-layout-casts",
        cpp_class="NVIDIAGraphLayoutMaterializationPass",
        summary=(
            "Consumes legal Graph layout casts as indexed NVIDIA binding "
            "contracts carried into Tile async-copy staging."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "func"),
        required_attrs=("tessera.layout",),
        preserved_attrs=("tessera.source_layout",),
        diagnostic_codes=(),
        must_run_after=("tessera-layout-legality",),
        pass_kind="lowering",
        sprint="CORE-COMPILER-FOLLOWON",
    ),
    PassMetadata(
        name="tessera-pipeline-partition",
        cpp_class="PipelineStagePartitionPass",
        summary=(
            "Cost-balanced, program-order-monotonic partition of each function "
            "into num_stages pipeline stages (emits tessera.pp_stage) — the real "
            "stage partitioning the insertion pass previously required an "
            "external tagger for."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "func"),
        required_attrs=("tessera.pipeline_plan", "tessera.pp_stage"),
        diagnostic_codes=(),
        pass_kind="transform",
        sprint="Pipeline-PP",
    ),
    PassMetadata(
        name="tessera-pipeline-schedule-legality",
        cpp_class="PipelineScheduleLegalityPass",
        summary=(
            "The 1F1B schedule proof — micro-batch fill (Decision #17), no empty "
            "stage, forward-adjacent send/recv pairing, and value-rewrite "
            "completeness (no direct cross-stage SSA edge)."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "func"),
        required_attrs=("tessera.pp_num_stages", "tessera.pp_num_micro_batches",
                        "tessera.pp_stage"),
        diagnostic_codes=(
            "PP_EMPTY_STAGE",
            "PP_MICRO_BATCHES_TOO_FEW",
            "PP_RECV_WITHOUT_SEND",
            "PP_SEND_WITHOUT_RECV",
            "PP_UNROUTED_CROSS_STAGE_VALUE",
        ),
        pass_kind="verifier",
        sprint="Pipeline-PP",
    ),
    PassMetadata(
        name="tessera-pipeline-stage-insertion",
        cpp_class="PipelineStageInsertionPass",
        summary=(
            "Inserts tessera.pipeline.send/recv at cross-stage boundaries and "
            "rewires the boundary uses to the recv (the real send/recv SSA "
            "rewrite), driven by the tessera.pp_stage partition."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "func"),
        required_attrs=("tessera.pp_stage", "tessera.layer"),
        preserved_attrs=("tessera.pp_stage",),
        diagnostic_codes=(),
        must_run_after=("tessera-pipeline-partition",),
        pass_kind="transform",
        sprint="Phase 5",
    ),
    PassMetadata(
        name="tessera-storage-legalize",
        cpp_class="StorageLegalize",
        summary=(
            "C4 (TIRx): terminal packing — stamps `tessera.storage_packed` + "
            "`tessera.storage_container` on sub-byte / block-scaled storage "
            "(fp4 / nvfp4 / fp6 / int4). Default-on only where a target owns a "
            "real packed-storage consumer (currently ROCm); runs last."
        ),
        input_dialects=("tessera",),
        output_dialects=("tessera",),
        diagnostic_codes=(),
        pass_kind="transform",
        sprint="C4 (TIRx)",
    ),
    PassMetadata(
        name="tessera-storage-pack-consume",
        cpp_class="StoragePackConsume",
        summary=(
            "C4 part 1 (TIRx): the first real consumer of the packing markers — "
            "reads tessera.storage_packed / storage_container + "
            "numeric_policy.storage and emits tessera.storage_pack = {logical, "
            "container, factor, signedness} (factor = container_bits / storage_bits) for a "
            "backend's packed load/store."
        ),
        input_dialects=("tessera",),
        output_dialects=("tessera",),
        required_attrs=("tessera.storage_packed", "tessera.storage_container"),
        diagnostic_codes=("DTYPE_PACK_BAD_WIDTHS",),
        must_run_after=("tessera-storage-legalize",),
        pass_kind="transform",
        sprint="C4 (TIRx)",
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
    PassMetadata(
        name="tessera-tile-barrier-reuse-legality",
        cpp_class="TileBarrierReuseLegality",
        summary=(
            "C2 (TIRx): barriers as a layout-reuse correctness property — two "
            "writes to overlapping storage-axis (m/tlane/tcol) footprints of one "
            "tile.buffer with no intervening barrier are a race. Runs after "
            "WarpSpecialization (which emits the #tile.layout + tile.buffer/"
            "tile.access markers) in the GPU / nvidia pipelines."
        ),
        input_dialects=("tessera", "tile", "func"),
        output_dialects=("tessera", "tile", "func"),
        required_attrs=("tile.buf", "tile.layout"),
        diagnostic_codes=("TILE_BARRIER_REUSE_MISSING_BARRIER",),
        pass_kind="verifier",
        sprint="C2 (TIRx)",
    ),
    PassMetadata(
        name="tessera-tile-pipeline-legality",
        cpp_class="TilePipelineLegality",
        summary=(
            "C3 (TIRx): cross-op pipeline legality — initial producer phase=1 / "
            "consumer phase=0 asymmetry (the off-by-one ring-deadlock fix) and "
            "per-tile.barrier_id #tile.barrier kind consistency. Runs after "
            "WarpSpecialization and again after NVTMADescriptor (typed barriers)."
        ),
        input_dialects=("tessera", "tile", "func"),
        output_dialects=("tessera", "tile", "func"),
        required_attrs=("tile.pipeline", "tile.pipeline_state",
                        "tile.barrier", "tile.barrier_id"),
        diagnostic_codes=(
            "TILE_PIPELINE_PHASE_ASYMMETRY",
            "TILE_PIPELINE_BARRIER_KIND_MISMATCH",
        ),
        pass_kind="verifier",
        sprint="C3 (TIRx)",
    ),
    PassMetadata(
        name="tessera-warpspec-legality",
        cpp_class="WarpSpecLegality",
        summary=(
            "C6 (TIRx): the 7 'Debugging Warp-Specialized Kernels' appendix "
            "invariants — init placement, collective-in-branch, loop-count "
            "agreement, TMA visibility fence, arrival-count==init-count, "
            "use-after-free. Runs after WarpSpecialization and again after "
            "NVTMADescriptor in the GPU / nvidia pipelines."
        ),
        input_dialects=("tessera", "tile", "func", "scf"),
        output_dialects=("tessera", "tile", "func", "scf"),
        required_attrs=("tile.warp_role", "tile.barrier", "tile.barrier_id",
                        "tile.pipeline", "tile.trip_count", "tile.buf"),
        diagnostic_codes=(
            "WARPSPEC_ARRIVAL_COUNT_MISMATCH",
            "WARPSPEC_COLLECTIVE_IN_DIVERGENT_BRANCH",
            "WARPSPEC_INIT_UNDER_GUARD",
            "WARPSPEC_LOOP_COUNT_DISAGREE",
            "WARPSPEC_MISSING_VISIBILITY_FENCE",
            "WARPSPEC_USE_AFTER_FREE",
        ),
        pass_kind="verifier",
        sprint="C6 (TIRx)",
    ),
    PassMetadata(
        name="tessera-x86-materialize-layout-casts",
        cpp_class="X86GraphLayoutMaterializationPass",
        summary=(
            "Consumes row-major/BHSD/NHWC Graph layout casts as indexed x86 "
            "binding contracts backed by the generic emitter's physical "
            "C-order materializer."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "func"),
        required_attrs=("tessera.layout",),
        preserved_attrs=("tessera.source_layout",),
        diagnostic_codes=(),
        must_run_after=("tessera-layout-legality",),
        pass_kind="lowering",
        sprint="CORE-COMPILER-TRAINING-SPINE",
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
