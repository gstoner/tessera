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
        name="declare-x86-pipeline-contract",
        cpp_class="DeclareX86PipelineContractPass",
        summary=(
            "Validates one closed x86 semantic-family plugin and stamps its "
            "Tile producer, tessera_x86 Target consumer, exact architecture, "
            "and prebuilt native-image boundary."
        ),
        input_dialects=("tile", "llvm", "arith"),
        output_dialects=("tile", "llvm", "arith"),
        required_attrs=("family",),
        preserved_attrs=("tessera.pipeline.family", "tessera.pipeline.arch"),
        pass_kind="verifier",
        sprint="X86-TYPED-FAMILY-PLUGIN-1",
    ),
    PassMetadata(
        name="lower-tile-to-rocm",
        cpp_class="LowerTileToROCMPass",
        summary=(
            "Lowers Tessera Tile IR matmul/attention movement contracts to "
            "ROCm Target IR. Typed `!tile.fragment` values go through a "
            "dialect conversion (fragment -> physical per-lane vector) so a "
            "K-loop accumulator, chained MMAs, and a non-zero accumulator all "
            "compose; the legacy bare `!tile.fragment` spelling still takes "
            "the single-shot whole-chain path."
        ),
        input_dialects=("tile", "tessera_rocm", "func", "scf", "vector",
                        "memref", "arith", "gpu"),
        output_dialects=("tessera_rocm", "func", "scf", "vector", "memref",
                         "arith", "gpu"),
        required_attrs=("tile.layout", "tile.memory"),
        diagnostic_codes=(
            "ROCM_FRAGMENT_ILLEGAL_ARCH_DESCRIPTOR",
            "ROCM_FRAGMENT_MATERIALIZATION_GATED",
            "ROCM_FRAGMENT_MISSING_CONTRACT",
            "ROCM_FRAGMENT_SOURCE_RANK",
            "ROCM_FRAGMENT_SOURCE_TYPE",
            "ROCM_FRAGMENT_STORE_LAYOUT",
            "ROCM_FRAGMENT_STORE_TYPE",
            "ROCM_FRAGMENT_TYPE_DISAGREES",
            "ROCM_FRAGMENT_UNPACK_UNCONSUMED",
            "ROCM_FRAGMENT_UNSUPPORTED_SOURCE_LAYOUT",
            "ROCM_LOWERING_LAYOUT_NOT_LDS",
            "ROCM_LOWERING_UNCONSUMED_STORAGE_PACK",
            "ROCM_TILE_UNSUPPORTED_DTYPE",
        ),
        pass_kind="lowering",
        sprint="ROCm Tile-IR convergence",
    ),
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
        required_attrs=("tile.layout", "tile.barrier"),
        diagnostic_codes=(
            "ROCM_WAVE_LDS_MISSING_WAITCNT",
            "ROCM_WAVE_LDS_OVERLAPPING_WRITE",
            "ROCM_WAVE_LDS_UNSUPPORTED_BARRIER_KIND",
            "ROCM_WAVE_LDS_ROLE_UNRESOLVED",
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
            "tessera.backward_work_ns",
            "tessera.residual.retained_bytes",
        ),
        preserved_attrs=("tessera.autodiff.phase",),
        diagnostic_codes=(
            "REMAT_EFFECTFUL",
            "REMAT_MODEL_BUDGET_INVALID",
            "REMAT_NON_CLONABLE",
            "REMAT_PLAN_CLONE_BOUND",
        ),
        pass_kind="transform",
        sprint="CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24",
    ),
    PassMetadata(
        name="tessera-adjoint-collective-insertion",
        cpp_class="AdjointCollectiveInsertionPass",
        summary=(
            "Wraps active sharded cotangent SSA values in registered async "
            "reduce-scatter/all-gather/all-reduce operations and awaits the "
            "payloads before returning them."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "tessera_collective", "func"),
        required_attrs=(
            "tessera.autodiff.arg_cotangents",
            "tessera.weight_sharding",
        ),
        preserved_attrs=("tessera.effect",),
        diagnostic_codes=(
            "ADJOINT_COLLECTIVE_COTANGENT_ARITY",
            "ADJOINT_COLLECTIVE_COTANGENT_SLOT_COUNT",
            "ADJOINT_COLLECTIVE_MULTIPLE_RETURNS",
            "ADJOINT_COLLECTIVE_NO_RETURN",
        ),
        must_run_after=("tessera-autodiff",),
        pass_kind="transform",
        sprint="COLLECTIVE-ASYNC-UNIFY-2026-08-09",
    ),
    PassMetadata(
        name="tessera-apple-canonical-gemm",
        cpp_class="CanonicalGemmToAppleGPUPass",
        summary=(
            "APPLE-TILE-2: recognizes the shared canonical M/N/K GEMM "
            "reduction (three-deep scf.for with an fp32 accumulator and staged "
            "!tile.pipeline_state) and re-forms it as one "
            "`tessera_apple.gpu.kernel_call` simdgroup_matrix dispatch, "
            "carrying the loop's own tile decision and ragged-zero-pad "
            "guarantee plus the compiler-owned Metal staging-byte contract. "
            "Recognition is not promotion: value-mode "
            "Accelerate/MPS remains the incumbent route."
        ),
        input_dialects=("tessera", "scf", "tensor", "func"),
        output_dialects=("tessera_apple", "func"),
        required_attrs=("tessera.canonical_k_step",),
        preserved_attrs=(
            "tessera_apple.canonical_k_loop",
            "tessera_apple.accumulate",
            "tessera_apple.ragged_zero_pad",
            "tessera_apple.staging_layout_owner",
            "tessera_apple.stage_depth",
            "tessera_apple.staged_a_bytes",
            "tessera_apple.staged_b_bytes",
            "tessera_apple.edge_scratch_bytes",
            "tessera_apple.threadgroup_arena_bytes",
            "tessera_apple.threadgroup_capacity_bytes",
        ),
        diagnostic_codes=(
            "APPLE_CANONICAL_GEMM_UNRECOGNIZED",
            "APPLE_CANONICAL_GEMM_SHAPE_UNSUPPORTED",
            "APPLE_CANONICAL_GEMM_DTYPE_UNSUPPORTED",
            "APPLE_CANONICAL_GEMM_ACCUM_UNSUPPORTED",
        ),
        pass_kind="lowering",
        sprint="APPLE-TILE-2",
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
        name="tessera-apple-streaming-attention",
        cpp_class="StreamingAttentionToAppleGPUPass",
        summary=(
            "APPLE-ATTN-STREAM-1/2: recognizes the shared KV-block "
            "streaming-attention recurrence and re-forms it as one Apple "
            "flash-attention dispatch, carrying boundary semantics read off "
            "tessera_attn.boundary_mask. For rank-4 distribution it replaces "
            "the enclosing batch/query-head loops with the static f32 GQA ABI, "
            "while retaining rank-2 and unsupported modifier/LSE cases as "
            "fail-closed boundaries."
        ),
        # The tessera_attn.* ops are matched generically by name, so the
        # FA-4 Attn dialect need not be loaded for this pass to run.
        input_dialects=("tile", "scf", "tensor", "func"),
        output_dialects=("tessera_apple", "func"),
        required_attrs=("tessera.streaming_attention",),
        preserved_attrs=(
            "tessera_apple.streaming_recurrence",
            "tessera_apple.rank4_distribution",
            "tessera_apple.batch",
            "tessera_apple.q_heads",
            "tessera_apple.kv_heads",
            "tessera_apple.gqa_group_size",
            "tessera_apple.sq",
            "tessera_apple.sk",
            "tessera_apple.scale",
            "tessera_apple.causal",
            "tessera_apple.logical_sk",
            "tessera_apple.kv_block",
        ),
        diagnostic_codes=(
            "APPLE_STREAMING_ATTN_UNRECOGNIZED",
            "APPLE_STREAMING_ATTN_LSE_UNSUPPORTED",
            "APPLE_STREAMING_ATTN_DROPOUT_UNSUPPORTED",
            "APPLE_STREAMING_ATTN_SHAPE_UNSUPPORTED",
            "APPLE_STREAMING_ATTN_HEAD_DIM_UNSUPPORTED",
            "APPLE_STREAMING_ATTN_DTYPE_UNSUPPORTED",
            "APPLE_STREAMING_ATTN_RANK4_MODIFIER_UNSUPPORTED",
        ),
        pass_kind="lowering",
        sprint="APPLE-ATTN-STREAM-1",
    ),
    PassMetadata(
        name="tessera-apple-threadgroup-pipeline",
        cpp_class="AppleThreadgroupPipelinePass",
        summary=(
            "APPLE-PIPE-1: Apple consumption of the shared Tile "
            "physical-allocation / staged-pipeline SSA contract. Places "
            "`!tile.buffer` allocations into one 16-byte-aligned, "
            "capacity-bounded Metal threadgroup arena and claims "
            "`!tile.pipeline_state` rings as ping-pong staging. Rejects "
            "NVIDIA-only TMA/mbarrier/TMEM vocabulary and name-based "
            "`#tile.buffer_ref` identity."
        ),
        input_dialects=("tile", "func"),
        output_dialects=("tile", "func"),
        required_attrs=("space", "bytes"),
        preserved_attrs=(
            "tessera_apple.address_space",
            "tessera_apple.threadgroup_offset",
            "tessera_apple.threadgroup_arena_bytes",
            "tessera_apple.stage_buffering",
        ),
        diagnostic_codes=(
            "APPLE_THREADGROUP_SPACE_UNSUPPORTED",
            "APPLE_THREADGROUP_MEMORY_EXCEEDED",
            "APPLE_THREADGROUP_MALFORMED_ALLOC",
            "APPLE_THREADGROUP_LEGACY_METADATA",
            "APPLE_THREADGROUP_INVALID_OPTION",
            "APPLE_STAGE_DEPTH_UNSUPPORTED",
            "APPLE_STAGE_MALFORMED_INIT",
            "APPLE_STAGE_UNROOTED_ADVANCE",
            "APPLE_TILE_UNSUPPORTED_VOCABULARY",
            "APPLE_MMA_STORAGE_UNSUPPORTED",
        ),
        pass_kind="lowering",
        sprint="APPLE-PIPE-1",
    ),
    PassMetadata(
        name="tessera-autodiff",
        cpp_class="AutodiffPass",
        summary=(
            "Builds an in-place reverse program using Graph adjoint and "
            "linear-transposition interfaces with SSA activity propagation."
        ),
        input_dialects=("tessera", "func", "arith"),
        output_dialects=("tessera", "func", "arith"),
        required_attrs=("tessera.autodiff",),
        preserved_attrs=("tessera.autodiff.activity",),
        diagnostic_codes=("AUTODIFF_STOCHASTIC_EFFECT",),
        pass_kind="transform",
        sprint="AD-CORE-EFFECT-CONTROL-1",
    ),
    PassMetadata(
        name="tessera-autodiff-forward",
        cpp_class="AutodiffForwardPass",
        summary=(
            "Emits a separate paired JVP function from compiler-owned Graph "
            "TangentInterface implementations."
        ),
        input_dialects=("tessera", "func", "arith"),
        output_dialects=("tessera", "func", "arith"),
        required_attrs=("tessera.autodiff",),
        preserved_attrs=("tessera.autodiff.jvp", "tessera.autodiff.role"),
        diagnostic_codes=(),
        pass_kind="transform",
        sprint="AD-FWD-CORE-1",
    ),
    PassMetadata(
        name="tessera-autodiff-hvp-prepare",
        cpp_class="AutodiffHvpPreparePass",
        summary=(
            "Marks the paired reverse Graph program for exact "
            "forward-over-reverse differentiation."
        ),
        input_dialects=("tessera", "func", "arith"),
        output_dialects=("tessera", "func", "arith"),
        required_attrs=("tessera.autodiff.role", "tessera.autodiff.forward"),
        preserved_attrs=(
            "tessera.autodiff.hvp",
            "tessera.autodiff.hvp_parent",
        ),
        diagnostic_codes=(),
        pass_kind="transform",
        sprint="AD-HIGHER-1",
    ),
    PassMetadata(
        name="tessera-autodiff-paired",
        cpp_class="AutodiffPairedPass",
        summary=(
            "Emits paired forward and backward functions under the explicit "
            "residual ABI: recompute-all by default, SAVE state tapes for "
            "control_scan and generic multi-state counted loops, plus saved "
            "branch/trip identity for scf.if and canonical bounded scf.while."
        ),
        input_dialects=("tessera", "func", "arith", "scf", "tensor"),
        output_dialects=("tessera", "func", "arith", "scf", "tensor"),
        required_attrs=("tessera.autodiff",),
        preserved_attrs=(
            "tessera.autodiff.activity",
            "tessera.autodiff.residual_policy",
            "tessera.autodiff.residual_sources",
        ),
        # W4-EFFECTS-1 slice E2 split the blanket stochastic refusal, so this
        # pass no longer emits AUTODIFF_STOCHASTIC_EFFECT — a keyed draw
        # carrying a verified product is now ADMITTED. Listing a code the
        # pass cannot emit is a declaration with no producer, the mirror of
        # #29, and it tells a reader the family is still refused wholesale.
        # (The in-place `tessera-autodiff` pass still emits the old code and
        # keeps it.)
        diagnostic_codes=(
            "AUTODIFF_CONTROL_SCAN_UNSUPPORTED",
            "AUTODIFF_STOCHASTIC_NO_PRODUCT",
            "AUTODIFF_STOCHASTIC_UNKEYED",
            "AUTODIFF_STOP_GRADIENT_RESIDUAL_REQUIRED",
        ),
        pass_kind="transform",
        sprint="W4-STRUCTURED-AD-2026-08-11",
    ),
    PassMetadata(
        name="tessera-await-sinking",
        cpp_class="AwaitSinkingPass",
        summary=(
            "Sinks registered collective awaits to their first SSA consumer "
            "only across operations proven memory-effect-free, with regions, "
            "mutation, RNG, aliases, and ordered collectives as barriers."
        ),
        input_dialects=("tessera_collective", "func"),
        output_dialects=("tessera_collective", "func"),
        preserved_attrs=("tessera.effect",),
        can_run_after=("tessera-adjoint-collective-insertion",),
        pass_kind="transform",
        sprint="COMP-SCHED-OVERLAP-1-R1-2026-08-10",
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
        preserved_attrs=(
            "tessera.presburger_constraints", "tessera.presburger_digest",
            "tessera.dim_bindings", "tessera.arg_dim_names",
        ),
        diagnostic_codes=(),
        must_run_after=("tessera-effect-annotate",),
        pass_kind="transform",
        sprint="Phase 2",
    ),
    PassMetadata(
        name="tessera-effect-annotate",
        cpp_class="EffectAnnotationPass",
        summary=(
            "Derives each func.func effect from registered Graph contracts "
            "and MLIR effect interfaces, propagating internal-call summaries "
            "to a fixed point and failing closed for unknown behavior."
        ),
        input_dialects=("tessera",),
        output_dialects=("tessera",),
        preserved_attrs=(
            "tessera.presburger_constraints", "tessera.presburger_digest",
            "tessera.dim_bindings", "tessera.arg_dim_names",
            "tessera.dim_sizes",
        ),
        diagnostic_codes=(),
        pass_kind="transform",
        sprint="Phase 2",
    ),
    PassMetadata(
        name="tessera-gpu-collective-insertion",
        cpp_class="GPUCollectiveInsertionPass",
        summary=(
            "Inserts registered asynchronous reduce-scatter/all-gather Target "
            "operations at DP/TP boundaries and rewires downstream SSA users "
            "through explicit awaits."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "tessera_collective", "func"),
        required_attrs=("tessera.weight_sharding",),
        preserved_attrs=(
            "tessera.presburger_constraints", "tessera.presburger_digest",
            "tessera.dim_bindings", "tessera.arg_dim_names",
        ),
        must_run_after=("tessera-effect-annotate",),
        pass_kind="transform",
        sprint="COLLECTIVE-ASYNC-UNIFY-2026-08-09",
    ),
    PassMetadata(
        name="tessera-graph-dataflow",
        cpp_class="GraphDataflowAnnotationPass",
        summary=(
            "Materializes the W2.1 fail-closed Graph shape, alias, liveness, "
            "and activity product snapshot for inspection and Python parity."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "func"),
        preserved_attrs=(
            "tessera.dataflow.shape",
            "tessera.dataflow.alias_roots",
            "tessera.dataflow.aliases_operands",
            "tessera.dataflow.live",
            "tessera.dataflow.activity",
            "tessera.dataflow.schema_version",
        ),
        diagnostic_codes=(),
        pass_kind="transform",
        sprint="COMP-GRAPH-DATAFLOW-W2.1-2026-08-11",
    ),
    PassMetadata(
        name="tessera-ir-contracts",
        cpp_class="IRContractLegality",
        summary=(
            "IR contract legality — numeric_policy SCHEMA (legal key set, "
            "string-valued entries, known dtypes/modes, accumulator not "
            "narrower in significand bits than storage, math_mode actually "
            "reducing), the Decision #15a storage/accum coupling, aliasing, "
            "and buffer-binding contracts."
        ),
        input_dialects=("tessera",),
        output_dialects=("tessera",),
        # Registering these makes them cross-checked against the diagnostic
        # registry by test_diagnostic_codes_are_registered; the pass had no
        # metadata entry before NUMPOL-CARRIER-1, so its codes were unchecked.
        diagnostic_codes=(
            "NUMERIC_POLICY_ACCUM_UNREALIZABLE",
            "NUMERIC_POLICY_MATH_MODE_NOT_REDUCING",
            "NUMERIC_POLICY_NARROWING_ACCUM",
            "NUMERIC_POLICY_NOT_A_DICTIONARY",
            "NUMERIC_POLICY_NON_STRING_VALUE",
            "NUMERIC_POLICY_UNKNOWN_ACCUM",
            "NUMERIC_POLICY_UNKNOWN_KEY",
            "NUMERIC_POLICY_UNKNOWN_MATH_MODE",
            "NUMERIC_POLICY_UNKNOWN_ROUNDING_MODE",
        ),
        must_run_after=("tessera-compute-legalize",),
        pass_kind="verifier",
        sprint="NUMPOL-CARRIER-1",
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
        name="tessera-lower-tile-collectives",
        cpp_class="LowerTileCollectivesPass",
        summary=(
            "Lowers the four typed Tile collectives to asynchronous portable "
            "tessera_collective Target IR plus explicit await dependencies."
        ),
        input_dialects=("tile", "func"),
        output_dialects=("tessera_collective", "func"),
        required_attrs=("mesh_axis", "tensor_axis", "reduction"),
        preserved_attrs=("world_size", "dtype", "chunk_bytes"),
        pass_kind="lowering",
        sprint="COLLECTIVE-TARGET-FUNCTIONAL-1",
    ),
    PassMetadata(
        name="tessera-newton-autodiff",
        cpp_class="NewtonAutodiffPass",
        summary=(
            "Validates the implicit residual function ABI and emits private "
            "value-producing IFT VJP/JVP functions over registered residual, "
            "matrix-free iterative GMRES/CG solve, and residual-adjoint "
            "operations with explicit convergence policy."
        ),
        input_dialects=("tessera_solver", "func"),
        output_dialects=("tessera_solver", "func"),
        required_attrs=("residual",),
        preserved_attrs=("residual",),
        pass_kind="transform",
        sprint="AD-SOLVER-IFT-1-2026-08-08",
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
        name="tessera-nvwgmma-lowering",
        cpp_class="NVWGMMALoweringPass",
        summary=(
            "Lowers tile.mma to a tessera_nvidia_wgmma_mma_async runtime call "
            "(SM>=90) or the WMMA path below it. Threads an optional third "
            "accumulator data operand and refuses extended data-operand forms "
            "that this compatibility boundary cannot represent."
        ),
        input_dialects=("tile", "func"),
        output_dialects=("func", "tile"),
        diagnostic_codes=("NVWGMMA_ACCUMULATOR_DROPPED",),
        pass_kind="lowering",
        sprint="NVWGMMA-ACCUMULATOR-GUARD-2026-08-03",
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
        required_attrs=("tessera.schedule_digest", "tessera.pipeline_steps",
                        "tessera.pp_num_stages", "tessera.pp_stage"),
        diagnostic_codes=(),
        pass_kind="transform",
        sprint="Pipeline-PP",
    ),
    PassMetadata(
        name="tessera-pipeline-schedule-legality",
        cpp_class="PipelineScheduleLegalityPass",
        summary=(
            "Proves 1F1B micro-batch fill, stage occupancy, send/recv pairing, "
            "value-rewrite completeness, and the producer-materialized "
            "Schedule Object dependency carrier without scalar reconstruction."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "func"),
        required_attrs=("tessera.schedule_digest", "tessera.pipeline_steps",
                        "tessera.pp_num_stages", "tessera.pp_num_micro_batches",
                        "tessera.pp_stage"),
        diagnostic_codes=(
            "PP_EMPTY_STAGE",
            "PP_MICRO_BATCHES_TOO_FEW",
            "PP_RECV_WITHOUT_SEND",
            "PP_SEND_WITHOUT_RECV",
            "PP_UNROUTED_CROSS_STAGE_VALUE",
        ),
        pass_kind="transform",
        sprint="Pipeline-PP",
    ),
    PassMetadata(
        name="tessera-pipeline-stage-insertion",
        cpp_class="PipelineStageInsertionPass",
        summary=(
            "Consumes the digest-bound Schedule Object carrier, inserts "
            "tessera.pipeline.send/recv at cross-stage boundaries, stamps the "
            "digest on emitted IR, and rewires boundary uses to recv results."
        ),
        input_dialects=("tessera", "func"),
        output_dialects=("tessera", "func"),
        required_attrs=("tessera.schedule_digest", "tessera.pipeline_steps",
                        "tessera.pp_num_stages", "tessera.pp_num_micro_batches",
                        "tessera.pp_stage", "tessera.layer"),
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
            "(fp4 / nvfp4 / fp6 / int4). Named targets gate this on the actual "
            "operation, structured physical descriptor, and complete def-use "
            "consumer (packed load/unpack, supported load/store round trip, "
            "packed matmul, or explicit conversion); runs last. An empty "
            "target remains an explicit inspection transform."
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
            "numeric_policy.storage and emits the structured "
            "#tile.packed_format logical/container/bits/factor/signedness/"
            "encoding/lane-order contract for a backend's packed load/store."
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
            "Consumes typed coefficient-vector `tessera.presburger_constraints` "
            "through MLIR integer Presburger analysis; verifies compatibility "
            "`tessera.dim_bindings` equations + "
            "per-op dim-name contracts (reshape / transpose / matmul), "
            "with SSA-value propagation seeded by frontend argument-local "
            "tessera.dim_names or legacy tessera.arg_dim_names, concrete sum-of-products witness "
            "checking, interprocedural cross-checks via func.call, and "
            "scf.for/scf.if/scf.while region recursion."
        ),
        input_dialects=("tessera", "func", "scf"),
        output_dialects=("tessera", "func", "scf"),
        required_attrs=(
            "tessera.dim_bindings",
            "tessera.dim_sizes",
            "tessera.arg_dim_names",
        ),
        preserved_attrs=(
            "tessera.presburger_constraints",
            "tessera.presburger_digest",
            "tessera.nonlinear_shape_guards",
            "tessera.nonlinear_shape_guard_digest",
            "tessera.dim_bindings",
            "tessera.dim_sizes",
            "tessera.arg_dim_names",
        ),
        diagnostic_codes=(
            "SYMDIM_BINDING_MALFORMED",
            "SYMDIM_BINDING_VIOLATION",
            "SYMDIM_CALL_ARG_MISMATCH",
            "SYMDIM_DIM_SIZES_MALFORMED",
            "SYMDIM_FLOW_INCONSISTENCY",
            "SYMDIM_IF_BRANCH_MISMATCH",
            "SYMDIM_LOOP_YIELD_MISMATCH",
            "SYMDIM_MATMUL_CONTRACT_VIOLATION",
            "SYMDIM_PRESBURGER_MALFORMED",
            "SYMDIM_PRESBURGER_UNSATISFIABLE",
            "SYMDIM_NONLINEAR_GUARD_MALFORMED",
            "SYMDIM_NONLINEAR_GUARD_INCOMPLETE",
            "SYMDIM_NONLINEAR_GUARD_VIOLATION",
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
            "!tile.buffer SSA allocation root with no intervening barrier are "
            "a race. Allocation identity is exclusively SSA-owned across "
            "NVIDIA, Apple/shared fixtures, and ROCm."
        ),
        input_dialects=("tessera", "tile", "func"),
        output_dialects=("tessera", "tile", "func"),
        required_attrs=("tile.layout",),
        diagnostic_codes=("TILE_BARRIER_REUSE_MISSING_BARRIER",),
        pass_kind="verifier",
        sprint="C2 (TIRx)",
    ),
    PassMetadata(
        name="tessera-tile-dataflow-legality",
        cpp_class="TileDataflowLegality",
        summary=(
            "P1a / CAKE §5.3 (W2.4's TileDataflowLegalityPass): derives — "
            "never name-matches — arrive→wait token pairing with slot "
            "identity, barrier origin, pipeline-ring advancement, and "
            "SSA-keyed TMA expect agreement, resolving across scf.for "
            "block-argument edges via the shared TileValueProvenance "
            "loop-carry resolver and failing closed on the underivable. "
            "Runs after the post-NVTMADescriptor C3/C6 blocks in both "
            "NVIDIA pipeline builders."
        ),
        input_dialects=("tessera", "tile", "func", "scf"),
        output_dialects=("tessera", "tile", "func", "scf"),
        required_attrs=("slot", "expect_tx", "tile.retire_all"),
        diagnostic_codes=(
            "TILE_WAIT_TOKEN_UNPAIRED",
            "TILE_WAIT_SLOT_MISMATCH",
            "TILE_WAIT_BARRIER_DISAGREES",
            "TILE_BARRIER_ORIGIN_UNRESOLVED",
            "TILE_PIPELINE_RING_STALE",
            "TILE_PIPELINE_RING_UNDERIVED",
            "TILE_TMA_EXPECT_MISMATCH",
            "TILE_TMA_DESC_ORIGIN_UNRESOLVED",
        ),
        pass_kind="verifier",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    PassMetadata(
        name="tessera-tile-pipeline-legality",
        cpp_class="TilePipelineLegality",
        summary=(
            "C3 (TIRx): SSA pipeline legality — tile.pipeline_init producer "
            "phase=1 / consumer phase=0 asymmetry, rejection of annotation-only "
            "#tile.pipeline_state metadata, and per-tile.barrier_id barrier-kind "
            "consistency."
        ),
        input_dialects=("tessera", "tile", "func"),
        output_dialects=("tessera", "tile", "func"),
        required_attrs=("tile.barrier", "tile.barrier_id"),
        diagnostic_codes=(
            "TILE_PIPELINE_PHASE_ASYMMETRY",
            "TILE_PIPELINE_LEGACY_METADATA",
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
    PassMetadata(
        name="verify-x86-executable",
        cpp_class="VerifyX86ExecutablePass",
        summary=(
            "Rejects surviving Tile family carriers and x86 packages that "
            "lack both a native ABI call and a registered tessera_x86 Target marker."
        ),
        input_dialects=("tessera_x86", "func", "llvm"),
        output_dialects=("tessera_x86", "func", "llvm"),
        required_attrs=("tessera.pipeline.family", "tessera.pipeline.arch"),
        pass_kind="verifier",
        sprint="X86-TYPED-FAMILY-PLUGIN-1",
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
