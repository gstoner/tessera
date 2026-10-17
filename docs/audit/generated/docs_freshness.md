# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-10-17**.

## Headline

- **176** docs catalogued across the canonical doc tree.
- **175** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **41** updated within the last 30 days.
- **69** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `APPLE_ARENA_NUMERICAL_POLICY.md` | - | 2026-09-12 | 35 | ✓ |
| `AUTODIFF_SPEC.md` | - | 2026-10-14 | 3 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-08-06 | 72 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 153 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 114 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 128 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-09-06 | 41 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 154 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-07-18 | 91 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-09-27 | 20 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 164 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-07-14 | 95 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-07-13 | 96 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 148 | ✓ |
| `NATIVE_ARTIFACT_SPEC.md` | Normative | 2026-07-19 | 90 | ✓ |
| `NATIVE_FP8_BLOCK_QUANT_FOUNDATION.md` | - | 2026-10-17 | 0 | ✓ |
| `NATIVE_GRAPH_SLICE_CONTRACT.md` | - | 2026-10-16 | 1 | ✓ |
| `NATIVE_NVFP4_BOUNDED_SOURCE_CONTRACT.md` | - | 2026-10-16 | 1 | ✓ |
| `NATIVE_PARENT_VIEW_CONTRACT.md` | - | 2026-10-17 | 0 | ✓ |
| `NATIVE_ROCM_SILU_MATH_CONTRACT.md` | - | 2026-10-16 | 1 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 134 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-10-15 | 2 | ✓ |
| `ROCM_FOLDED_INPUT_VIEWS.md` | - | 2026-10-16 | 1 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-07-18 | 91 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 148 | ✓ |
| `SM120_ATTENTION_ROW_OWNER_CONTRACT.md` | - | 2026-10-17 | 0 | ✓ |
| `SM120_SAVED_LSE_STORAGE_CONTRACT.md` | - | 2026-10-16 | 1 | ✓ |
| `SM120_TYPED_ATTENTION_BACKWARD_CONTRACT.md` | - | 2026-10-17 | 0 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-08-24 | 54 | ✓ |
| `TILE_IR.md` | Normative | 2026-10-02 | 15 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-08-02 | 76 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 135 | ✓ |
| `ir/gfx1201_folded_fragment_schedule.md` | - | 2026-10-15 | 2 | ✓ |
| `ir/normalization_ad_accumulator_policy.md` | - | 2026-10-16 | 1 | ✓ |
| `ir/rocm_nvfp4_source_preserving_program.md` | - | 2026-10-16 | 1 | ✓ |
| `ir/sm120_activation_schedule_tile_contract.md` | - | 2026-10-15 | 2 | ✓ |
| `ir/sm120_paired_tensor_program_contract.md` | - | 2026-10-15 | 2 | ✓ |
| `ir/source_active_row_jvp.md` | - | 2026-10-16 | 1 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 164 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 164 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 172 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 172 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 172 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 128 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 172 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-08-06 | 72 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-08-06 | 72 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 172 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-07-14 | 95 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-07-14 | 95 | ✓ |
| `porting_advanced_examples.md` | Informative | 2026-09-21 | 26 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-07-14 | 95 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 128 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-07-13 | 96 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-07-14 | 95 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 128 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 128 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 128 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 128 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 128 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 128 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 128 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 128 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 128 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-07-13 | 96 | ✓ |
| `backend_local_proofs.md` | - | 2026-07-15 | 94 | ✓ |
| `release_gates.md` | Normative | 2026-07-13 | 96 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-07-14 | 95 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-07-14 | 95 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-07-14 | 95 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-07-14 | 95 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-07-14 | 95 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-07-13 | 96 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-07-14 | 95 | ✓ |
| `README.md` | Informative | 2026-05-20 | 150 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 164 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | - | 2026-07-14 | 95 | ✓ |
| `compiler_test_architecture.md` | Normative | 2026-08-02 | 76 | ✓ |
| `distributed/megamoe.md` | - | 2026-06-09 | 130 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 150 | ✓ |
| `inference/serving.md` | - | 2026-07-13 | 96 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 174 | ✓ |
| `proposals/tile_fragment_abi.md` | Proposal | 2026-09-13 | 34 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-07-14 | 95 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 150 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 128 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 170 | ✓ |
| `workloads/attention-family.md` | Planning | 2026-07-14 | 95 | ✓ |
| `workloads/dflash.md` | - | 2026-07-14 | 95 | ✓ |
| `workloads/msa-cuda-phase3.md` | - | 2026-07-13 | 96 | ✓ |
| `workloads/msa.md` | - | 2026-07-13 | 96 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-07-13 | 96 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 150 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 150 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 159 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-09-27 | 20 | ✓ |
| `README.md` | - | 2026-09-06 | 41 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-09-05 | 42 | ✓ |
| `backend/E2E_COMPILATION_AUDIT.md` | - | 2026-09-05 | 42 | ✓ |
| `backend/README.md` | - | 2026-09-05 | 42 | ✓ |
| `backend/X86_AVX512_ABI_INVENTORY.md` | - | 2026-07-22 | 87 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-09-25 | 22 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-07-13 | 96 | ✓ |
| `backend/apple/README.md` | - | 2026-09-05 | 42 | ✓ |
| `backend/apple/todo.md` | - | 2026-10-17 | 0 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-09-05 | 42 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-09-05 | 42 | ✓ |
| `backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md` | - | 2026-09-27 | 20 | ✓ |
| `backend/nvidia/VERIFY_TARGET_IR_TAIL.md` | - | 2026-07-13 | 96 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 115 | ✓ |
| `backend/nvidia/todo.md` | - | 2026-10-17 | 0 | ✓ |
| `backend/rocm/GEMM_PERF_LADDER.md` | - | 2026-08-04 | 74 | ✓ |
| `backend/rocm/GFX125X_CDNA5_COMPILER_REFERENCE.md` | - | 2026-08-14 | 64 | ✓ |
| `backend/rocm/GIN_EXACT_DEVICE_RUNBOOK.md` | - | 2026-08-09 | 69 | ✓ |
| `backend/rocm/NATIVE_RDNA4_COMMISSIONING.md` | - | 2026-09-15 | 32 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-09-21 | 26 | ✓ |
| `backend/rocm/ROCM_LANE_MAP.md` | - | 2026-09-26 | 21 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-07-28 | 81 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-09-05 | 42 | ✓ |
| `backend/rocm/todo.md` | - | 2026-10-17 | 0 | ✓ |
| `backend/x86/todo.md` | - | 2026-10-17 | 0 | ✓ |
| `compiler/AMD_KERNEL_COMPILER_SURVEY.md` | - | 2026-09-22 | 25 | ✓ |
| `compiler/ANN_CALCULUS_DESIGN_SPIKE.md` | - | 2026-09-04 | 43 | ✓ |
| `compiler/AUTODIFF_ARCHITECTURE_REVIEW.md` | - | 2026-09-06 | 41 | ✓ |
| `compiler/AUTODIFF_EXECUTION_PLAN.md` | - | 2026-09-14 | 33 | ✓ |
| `compiler/AUTODIFF_NEXTGEN_PLAN.md` | - | 2026-09-06 | 41 | ✓ |
| `compiler/AUTODIFF_UNIFICATION_PLAN.md` | - | 2026-09-06 | 41 | ✓ |
| `compiler/BLOCK_ATTNRES_ROCM_PLAN.md` | - | 2026-09-09 | 38 | ✓ |
| `compiler/COMPILER_ARCHITECTURE_SWEEP.md` | - | 2026-09-09 | 38 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-09-16 | 31 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-09-08 | 39 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-28 | 81 | ✓ |
| `compiler/CORE_SUBSTRATE_VIEW.md` | - | 2026-09-07 | 40 | ✓ |
| `compiler/CUTE_IR_ASSESSMENT.md` | - | 2026-08-24 | 54 | ✓ |
| `compiler/DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | - | 2026-09-07 | 40 | ✓ |
| `compiler/EGGROLL_SUPPORT_PLAN.md` | - | 2026-09-07 | 40 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-10-06 | 11 | ✓ |
| `compiler/FIVE_SLICE_STATUS_20261007.md` | - | 2026-10-17 | 0 | ✓ |
| `compiler/FORGE_ASSESSMENT.md` | - | 2026-09-07 | 40 | ✓ |
| `compiler/FRONTEND_GRAPH_SCHEDULE_REVIEW.md` | - | 2026-08-02 | 76 | ✓ |
| `compiler/FRONT_END_LOWERING_ASSESSMENT.md` | - | 2026-09-03 | 44 | ✓ |
| `compiler/FUNCTIONAL_ANALYSIS_TSOL_PLAN.md` | - | 2026-09-06 | 41 | ✓ |
| `compiler/GAME_THEORY_PLAN.md` | - | 2026-09-07 | 40 | ✓ |
| `compiler/HEAP_BARRIER_ARCHITECTURE_REVIEW.md` | - | 2026-09-11 | 36 | ✓ |
| `compiler/INTEGRATED_COMPILER_LOG.md` | - | 2026-10-17 | 0 | ✓ |
| `compiler/INTEGRATED_COMPILER_PLAN.md` | - | 2026-10-17 | 0 | ✓ |
| `compiler/INTRA_KERNEL_FEEDBACK_PLAN.md` | - | 2026-09-23 | 24 | ✓ |
| `compiler/IR_STACK_INTEGRATION_REVIEW.md` | - | 2026-08-02 | 76 | ✓ |
| `compiler/LSE_CHECKPOINT_CONTRACT.md` | - | 2026-07-27 | 82 | ✓ |
| `compiler/MATH_SOURCE_WORKSTREAM.md` | - | 2026-09-07 | 40 | ✓ |
| `compiler/MATRIX_CALCULUS_REVIEW.md` | - | 2026-09-07 | 40 | ✓ |
| `compiler/MLIR_NATIVE_FOUNDATION_SURVEY.md` | - | 2026-10-11 | 6 | ✓ |
| `compiler/ODS_OP_CONNECTION_TRIAGE.md` | - | 2026-09-28 | 19 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-08-08 | 70 | ✓ |
| `compiler/PDE_STENCIL_CAPABILITY_PLAN.md` | - | 2026-09-07 | 40 | ✓ |
| `compiler/README.md` | - | 2026-10-07 | 10 | ✓ |
| `compiler/RIEMANNIAN_OT_PLAN.md` | - | 2026-09-07 | 40 | ✓ |
| `compiler/SCHEDULE_OBJECT_DESIGN.md` | - | 2026-08-16 | 62 | ✓ |
| `compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md` | - | 2026-09-10 | 37 | ✓ |
| `compiler/SEQUENCE_MIXER_THEORY.md` | - | 2026-09-08 | 39 | ✓ |
| `compiler/SPARDA_REVIEW.md` | - | 2026-08-12 | 66 | ✓ |
| `compiler/TARGET_IR_REVIEW.md` | - | 2026-09-06 | 41 | ✓ |
| `compiler/TILERT_ASSESSMENT.md` | - | 2026-09-05 | 42 | ✓ |
| `compiler/TILESIGHT_ASSESSMENT.md` | - | 2026-07-30 | 79 | ✓ |
| `compiler/W1_1_TYPING_DESIGN.md` | - | 2026-10-11 | 6 | ✓ |
| `compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md` | - | 2026-08-25 | 53 | ✓ |
| `compiler/compiler_enhancement.md` | - | 2026-09-08 | 39 | ✓ |
| `compiler/native_public_jvp_integration_plan.md` | - | 2026-10-14 | 3 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-10-11 | 6 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-09-16 | 31 | ✓ |
| `domain/EBM_NATIVE_LOOP_ARCHITECTURE.md` | - | 2026-09-16 | 31 | ✓ |
| `domain/GA_EBM_ARCHITECTURE_REVIEW.md` | - | 2026-09-16 | 31 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 109 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-08-12 | 66 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-08-11 | 67 | ✓ |
