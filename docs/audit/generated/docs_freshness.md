# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-11-05**.

## Headline

- **190** docs catalogued across the canonical doc tree.
- **189** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **66** updated within the last 30 days.
- **62** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `APPLE_ARENA_NUMERICAL_POLICY.md` | - | 2026-09-12 | 54 | ✓ |
| `AUTODIFF_SPEC.md` | - | 2026-10-14 | 22 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-08-06 | 91 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-10-24 | 12 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 133 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 147 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-09-06 | 60 | ✓ |
| `EBM_SPEC.md` | - | 2026-10-24 | 12 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-07-18 | 110 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-09-27 | 39 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-10-24 | 12 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-07-14 | 114 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-07-13 | 115 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-10-24 | 12 | ✓ |
| `NATIVE_ARTIFACT_SPEC.md` | Normative | 2026-07-19 | 109 | ✓ |
| `NATIVE_BOUNDED_QUERY_NORM.md` | - | 2026-10-19 | 17 | ✓ |
| `NATIVE_BOUNDED_SOURCE_ATTENTION_REVERSE.md` | - | 2026-10-19 | 17 | ✓ |
| `NATIVE_BOUNDED_SOURCE_OWNER.md` | - | 2026-10-19 | 17 | ✓ |
| `NATIVE_BOUNDED_SOURCE_PUBLIC.md` | - | 2026-10-19 | 17 | ✓ |
| `NATIVE_FP8_BLOCK_QUANT_FOUNDATION.md` | - | 2026-10-17 | 19 | ✓ |
| `NATIVE_GRAPH_SLICE_CONTRACT.md` | - | 2026-10-16 | 20 | ✓ |
| `NATIVE_NVFP4_BOUNDED_SOURCE_CONTRACT.md` | - | 2026-10-16 | 20 | ✓ |
| `NATIVE_PARENT_VIEW_CONTRACT.md` | - | 2026-10-17 | 19 | ✓ |
| `NATIVE_ROCM_SILU_MATH_CONTRACT.md` | - | 2026-10-16 | 20 | ✓ |
| `NATIVE_ROW_LOGICAL_PREFIX.md` | - | 2026-10-18 | 18 | ✓ |
| `NATIVE_SOURCE_ATTENTION_PROGRAM.md` | - | 2026-10-18 | 18 | ✓ |
| `NATIVE_TENSOR_FRAME_OWNER.md` | - | 2026-10-20 | 16 | ✓ |
| `NATIVE_TENSOR_JIT.md` | - | 2026-10-20 | 16 | ✓ |
| `NATIVE_TENSOR_PROGRAM_BOUNDED.md` | - | 2026-10-21 | 15 | ✓ |
| `NATIVE_TENSOR_PROGRAM_PACKAGE.md` | - | 2026-10-20 | 16 | ✓ |
| `NATIVE_TENSOR_PROGRAM_PIPELINE.md` | - | 2026-10-20 | 16 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 153 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-10-15 | 21 | ✓ |
| `ROCM_FOLDED_INPUT_VIEWS.md` | - | 2026-10-16 | 20 | ✓ |
| `ROCM_MIXED_SCALAR_CONTRACTION.md` | - | 2026-10-18 | 18 | ✓ |
| `ROCM_SOURCE_PULLBACK.md` | - | 2026-10-18 | 18 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-07-18 | 110 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-10-24 | 12 | ✓ |
| `SM120_ATTENTION_ROW_OWNER_CONTRACT.md` | - | 2026-10-17 | 19 | ✓ |
| `SM120_FP8_RHS_EDGE_CONTRACT.md` | - | 2026-10-26 | 10 | ✓ |
| `SM120_SAVED_LSE_STORAGE_CONTRACT.md` | - | 2026-10-26 | 10 | ✓ |
| `SM120_TYPED_ATTENTION_BACKWARD_CONTRACT.md` | - | 2026-10-17 | 19 | ✓ |
| `SM120_TYPED_ATTENTION_JVP_CONTRACT.md` | - | 2026-10-17 | 19 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-08-24 | 73 | ✓ |
| `TILE_IR.md` | Normative | 2026-10-02 | 34 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-08-02 | 95 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 154 | ✓ |
| `ir/gfx1201_folded_fragment_schedule.md` | - | 2026-10-15 | 21 | ✓ |
| `ir/normalization_ad_accumulator_policy.md` | - | 2026-10-16 | 20 | ✓ |
| `ir/rocm_nvfp4_source_preserving_program.md` | - | 2026-10-16 | 20 | ✓ |
| `ir/sm120_activation_schedule_tile_contract.md` | - | 2026-10-15 | 21 | ✓ |
| `ir/sm120_native_tensor_result_contract.md` | - | 2026-10-18 | 18 | ✓ |
| `ir/sm120_paired_tensor_program_contract.md` | - | 2026-10-15 | 21 | ✓ |
| `ir/sm120_scalar_contraction_contract.md` | - | 2026-10-18 | 18 | ✓ |
| `ir/source_active_row_jvp.md` | - | 2026-10-16 | 20 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-10-24 | 12 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-10-24 | 12 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-10-24 | 12 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-10-24 | 12 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-10-24 | 12 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 147 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-10-24 | 12 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-08-06 | 91 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-08-06 | 91 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-10-24 | 12 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-07-14 | 114 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-07-14 | 114 | ✓ |
| `porting_advanced_examples.md` | Informative | 2026-09-21 | 45 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-07-14 | 114 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 147 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-07-13 | 115 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-07-14 | 114 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 147 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 147 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 147 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 147 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 147 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 147 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 147 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 147 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 147 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-07-13 | 115 | ✓ |
| `backend_local_proofs.md` | - | 2026-07-15 | 113 | ✓ |
| `release_gates.md` | Normative | 2026-07-13 | 115 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-07-14 | 114 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-07-14 | 114 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-07-14 | 114 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-07-14 | 114 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-07-14 | 114 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-07-13 | 115 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-07-14 | 114 | ✓ |
| `README.md` | Informative | 2026-10-24 | 12 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-10-24 | 12 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | - | 2026-07-14 | 114 | ✓ |
| `compiler_test_architecture.md` | Normative | 2026-08-02 | 95 | ✓ |
| `distributed/megamoe.md` | - | 2026-06-09 | 149 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-10-24 | 12 | ✓ |
| `inference/serving.md` | - | 2026-07-13 | 115 | ✓ |
| `proposals/tile_fragment_abi.md` | Proposal | 2026-09-13 | 53 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-07-14 | 114 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-10-24 | 12 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 147 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-10-24 | 12 | ✓ |
| `workloads/attention-family.md` | Planning | 2026-07-14 | 114 | ✓ |
| `workloads/dflash.md` | - | 2026-07-14 | 114 | ✓ |
| `workloads/msa-cuda-phase3.md` | - | 2026-07-13 | 115 | ✓ |
| `workloads/msa.md` | - | 2026-07-13 | 115 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-07-13 | 115 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_tensor_attributes.md` | Normative | 2026-10-24 | 12 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-09-27 | 39 | ✓ |
| `README.md` | - | 2026-09-06 | 60 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-09-05 | 61 | ✓ |
| `backend/E2E_COMPILATION_AUDIT.md` | - | 2026-09-05 | 61 | ✓ |
| `backend/README.md` | - | 2026-09-05 | 61 | ✓ |
| `backend/X86_AVX512_ABI_INVENTORY.md` | - | 2026-07-22 | 106 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-09-25 | 41 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-07-13 | 115 | ✓ |
| `backend/apple/README.md` | - | 2026-09-05 | 61 | ✓ |
| `backend/apple/todo.md` | - | 2026-10-21 | 15 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-09-05 | 61 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-09-05 | 61 | ✓ |
| `backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md` | - | 2026-09-27 | 39 | ✓ |
| `backend/nvidia/VERIFY_TARGET_IR_TAIL.md` | - | 2026-07-13 | 115 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 134 | ✓ |
| `backend/nvidia/todo.md` | - | 2026-11-04 | 1 | ✓ |
| `backend/rocm/GEMM_PERF_LADDER.md` | - | 2026-08-04 | 93 | ✓ |
| `backend/rocm/GFX125X_CDNA5_COMPILER_REFERENCE.md` | - | 2026-08-14 | 83 | ✓ |
| `backend/rocm/GIN_EXACT_DEVICE_RUNBOOK.md` | - | 2026-08-09 | 88 | ✓ |
| `backend/rocm/NATIVE_RDNA4_COMMISSIONING.md` | - | 2026-09-15 | 51 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-09-21 | 45 | ✓ |
| `backend/rocm/ROCM_LANE_MAP.md` | - | 2026-09-26 | 40 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-07-28 | 100 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-09-05 | 61 | ✓ |
| `backend/rocm/todo.md` | - | 2026-11-04 | 1 | ✓ |
| `backend/x86/todo.md` | - | 2026-10-21 | 15 | ✓ |
| `compiler/AMD_KERNEL_COMPILER_SURVEY.md` | - | 2026-09-22 | 44 | ✓ |
| `compiler/ANN_CALCULUS_DESIGN_SPIKE.md` | - | 2026-09-04 | 62 | ✓ |
| `compiler/AUTODIFF_ARCHITECTURE_REVIEW.md` | - | 2026-09-06 | 60 | ✓ |
| `compiler/AUTODIFF_EXECUTION_PLAN.md` | - | 2026-10-23 | 13 | ✓ |
| `compiler/AUTODIFF_NEXTGEN_PLAN.md` | - | 2026-09-06 | 60 | ✓ |
| `compiler/AUTODIFF_UNIFICATION_PLAN.md` | - | 2026-09-06 | 60 | ✓ |
| `compiler/BLOCK_ATTNRES_ROCM_PLAN.md` | - | 2026-09-09 | 57 | ✓ |
| `compiler/COMPILER_ARCHITECTURE_SWEEP.md` | - | 2026-09-09 | 57 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-09-16 | 50 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-09-08 | 58 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-28 | 100 | ✓ |
| `compiler/CORE_SUBSTRATE_VIEW.md` | - | 2026-09-07 | 59 | ✓ |
| `compiler/CUTE_IR_ASSESSMENT.md` | - | 2026-08-24 | 73 | ✓ |
| `compiler/DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | - | 2026-09-07 | 59 | ✓ |
| `compiler/EGGROLL_SUPPORT_PLAN.md` | - | 2026-09-07 | 59 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-10-06 | 30 | ✓ |
| `compiler/FIVE_SLICE_STATUS_20261007.md` | - | 2026-10-21 | 15 | ✓ |
| `compiler/FORGE_ASSESSMENT.md` | - | 2026-09-07 | 59 | ✓ |
| `compiler/FRONTEND_GRAPH_SCHEDULE_REVIEW.md` | - | 2026-08-02 | 95 | ✓ |
| `compiler/FRONT_END_LOWERING_ASSESSMENT.md` | - | 2026-09-03 | 63 | ✓ |
| `compiler/FUNCTIONAL_ANALYSIS_TSOL_PLAN.md` | - | 2026-09-06 | 60 | ✓ |
| `compiler/GAME_THEORY_PLAN.md` | - | 2026-09-07 | 59 | ✓ |
| `compiler/HEAP_BARRIER_ARCHITECTURE_REVIEW.md` | - | 2026-09-11 | 55 | ✓ |
| `compiler/INTEGRATED_COMPILER_LOG.md` | - | 2026-11-05 | 0 | ✓ |
| `compiler/INTEGRATED_COMPILER_PLAN.md` | - | 2026-11-03 | 2 | ✓ |
| `compiler/INTRA_KERNEL_FEEDBACK_PLAN.md` | - | 2026-09-23 | 43 | ✓ |
| `compiler/IR_STACK_INTEGRATION_REVIEW.md` | - | 2026-08-02 | 95 | ✓ |
| `compiler/LSE_CHECKPOINT_CONTRACT.md` | - | 2026-07-27 | 101 | ✓ |
| `compiler/MATH_SOURCE_WORKSTREAM.md` | - | 2026-09-07 | 59 | ✓ |
| `compiler/MATRIX_CALCULUS_REVIEW.md` | - | 2026-09-07 | 59 | ✓ |
| `compiler/MLIR_NATIVE_FOUNDATION_SURVEY.md` | - | 2026-11-03 | 2 | ✓ |
| `compiler/ODS_OP_CONNECTION_TRIAGE.md` | - | 2026-09-28 | 38 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-08-08 | 89 | ✓ |
| `compiler/PDE_STENCIL_CAPABILITY_PLAN.md` | - | 2026-09-07 | 59 | ✓ |
| `compiler/README.md` | - | 2026-10-07 | 29 | ✓ |
| `compiler/RIEMANNIAN_OT_PLAN.md` | - | 2026-09-07 | 59 | ✓ |
| `compiler/SCHEDULE_OBJECT_DESIGN.md` | - | 2026-08-16 | 81 | ✓ |
| `compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md` | - | 2026-09-10 | 56 | ✓ |
| `compiler/SEQUENCE_MIXER_THEORY.md` | - | 2026-09-08 | 58 | ✓ |
| `compiler/SPARDA_REVIEW.md` | - | 2026-08-12 | 85 | ✓ |
| `compiler/TARGET_IR_REVIEW.md` | - | 2026-09-06 | 60 | ✓ |
| `compiler/TILERT_ASSESSMENT.md` | - | 2026-09-05 | 61 | ✓ |
| `compiler/TILESIGHT_ASSESSMENT.md` | - | 2026-07-30 | 98 | ✓ |
| `compiler/W1_1_TYPING_DESIGN.md` | - | 2026-10-11 | 25 | ✓ |
| `compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md` | - | 2026-08-25 | 72 | ✓ |
| `compiler/compiler_enhancement.md` | - | 2026-09-08 | 58 | ✓ |
| `compiler/native_public_jvp_integration_plan.md` | - | 2026-10-14 | 22 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-10-11 | 25 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-09-16 | 50 | ✓ |
| `domain/EBM_NATIVE_LOOP_ARCHITECTURE.md` | - | 2026-09-16 | 50 | ✓ |
| `domain/GA_EBM_ARCHITECTURE_REVIEW.md` | - | 2026-09-16 | 50 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 128 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-08-12 | 85 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-08-11 | 86 | ✓ |
