# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-09-02**.

## Headline

- **151** docs catalogued across the canonical doc tree.
- **150** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **51** updated within the last 30 days.
- **22** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `AUTODIFF_SPEC.md` | - | 2026-07-14 | 50 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-08-06 | 27 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 108 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 69 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 83 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-08-11 | 22 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 109 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-07-18 | 46 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-07-14 | 50 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 119 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-07-14 | 50 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-07-13 | 51 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 103 | ✓ |
| `NATIVE_ARTIFACT_SPEC.md` | Normative | 2026-07-19 | 45 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 89 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-07-23 | 41 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-07-18 | 46 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 103 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-08-24 | 9 | ✓ |
| `TILE_IR.md` | Normative | 2026-08-10 | 23 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-08-02 | 31 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 90 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 119 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 119 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 127 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 127 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 127 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 83 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 127 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-08-06 | 27 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-08-06 | 27 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 127 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-07-14 | 50 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-07-14 | 50 | ✓ |
| `porting_advanced_examples.md` | Informative | 2026-05-09 | 116 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-07-14 | 50 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 83 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-07-13 | 51 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-07-14 | 50 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 83 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 83 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 83 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 83 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 83 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 83 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 83 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 83 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 83 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-07-13 | 51 | ✓ |
| `backend_local_proofs.md` | - | 2026-07-15 | 49 | ✓ |
| `release_gates.md` | Normative | 2026-07-13 | 51 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-07-14 | 50 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-07-14 | 50 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-07-14 | 50 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-07-14 | 50 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-07-14 | 50 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-07-13 | 51 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-07-14 | 50 | ✓ |
| `README.md` | Informative | 2026-05-20 | 105 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 119 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | - | 2026-07-14 | 50 | ✓ |
| `compiler_test_architecture.md` | Normative | 2026-08-02 | 31 | ✓ |
| `distributed/megamoe.md` | - | 2026-06-09 | 85 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 105 | ✓ |
| `inference/serving.md` | - | 2026-07-13 | 51 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 129 | ✓ |
| `proposals/tile_fragment_abi.md` | Proposal | 2026-07-19 | 45 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-07-14 | 50 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 105 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 83 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 125 | ✓ |
| `workloads/attention-family.md` | Planning | 2026-07-14 | 50 | ✓ |
| `workloads/dflash.md` | - | 2026-07-14 | 50 | ✓ |
| `workloads/msa-cuda-phase3.md` | - | 2026-07-13 | 51 | ✓ |
| `workloads/msa.md` | - | 2026-07-13 | 51 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-07-13 | 51 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 105 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 105 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 114 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-08-11 | 22 | ✓ |
| `README.md` | - | 2026-08-08 | 25 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-07-31 | 33 | ✓ |
| `backend/E2E_COMPILATION_AUDIT.md` | - | 2026-07-27 | 37 | ✓ |
| `backend/X86_AVX512_ABI_INVENTORY.md` | - | 2026-07-22 | 42 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-07-28 | 36 | ✓ |
| `backend/apple/APPLE_GPU_CODEGEN_PLAN.md` | - | 2026-07-13 | 51 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-07-13 | 51 | ✓ |
| `backend/apple/README.md` | - | 2026-07-13 | 51 | ✓ |
| `backend/apple/todo.md` | - | 2026-08-29 | 4 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-06-24 | 70 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-07-18 | 46 | ✓ |
| `backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md` | - | 2026-07-19 | 45 | ✓ |
| `backend/nvidia/VERIFY_TARGET_IR_TAIL.md` | - | 2026-07-13 | 51 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 70 | ✓ |
| `backend/nvidia/todo.md` | - | 2026-09-02 | 0 | ✓ |
| `backend/rocm/GEMM_PERF_LADDER.md` | - | 2026-08-04 | 29 | ✓ |
| `backend/rocm/GFX125X_CDNA5_COMPILER_REFERENCE.md` | - | 2026-08-14 | 19 | ✓ |
| `backend/rocm/GIN_EXACT_DEVICE_RUNBOOK.md` | - | 2026-08-09 | 24 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-08-14 | 19 | ✓ |
| `backend/rocm/ROCM_LANE_MAP.md` | - | 2026-08-05 | 28 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-07-28 | 36 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-06-23 | 71 | ✓ |
| `backend/rocm/todo.md` | - | 2026-08-29 | 4 | ✓ |
| `backend/x86/todo.md` | - | 2026-08-29 | 4 | ✓ |
| `compiler/AMD_KERNEL_COMPILER_SURVEY.md` | - | 2026-07-28 | 36 | ✓ |
| `compiler/AUTODIFF_ARCHITECTURE_REVIEW.md` | - | 2026-08-18 | 15 | ✓ |
| `compiler/AUTODIFF_NEXTGEN_PLAN.md` | - | 2026-08-20 | 13 | ✓ |
| `compiler/AUTODIFF_UNIFICATION_PLAN.md` | - | 2026-08-18 | 15 | ✓ |
| `compiler/BLOCK_ATTNRES_ROCM_PLAN.md` | - | 2026-08-13 | 20 | ✓ |
| `compiler/CODE_REVIEW_2026-08-29.md` | - | 2026-08-29 | 4 | ✓ |
| `compiler/COMPILER_ARCHITECTURE_SWEEP.md` | - | 2026-08-11 | 22 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-08-10 | 23 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-08-08 | 25 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-28 | 36 | ✓ |
| `compiler/CORE_SUBSTRATE_VIEW.md` | - | 2026-08-24 | 9 | ✓ |
| `compiler/CUTE_IR_ASSESSMENT.md` | - | 2026-08-24 | 9 | ✓ |
| `compiler/DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | - | 2026-08-08 | 25 | ✓ |
| `compiler/EGGROLL_SUPPORT_PLAN.md` | - | 2026-08-09 | 24 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-08-08 | 25 | ✓ |
| `compiler/FORGE_ASSESSMENT.md` | - | 2026-08-15 | 18 | ✓ |
| `compiler/FRONTEND_GRAPH_SCHEDULE_REVIEW.md` | - | 2026-08-02 | 31 | ✓ |
| `compiler/FUNCTIONAL_ANALYSIS_TSOL_PLAN.md` | - | 2026-08-22 | 11 | ✓ |
| `compiler/GAME_THEORY_PLAN.md` | - | 2026-08-15 | 18 | ✓ |
| `compiler/INTEGRATED_COMPILER_PLAN.md` | - | 2026-08-24 | 9 | ✓ |
| `compiler/INTRA_KERNEL_FEEDBACK_PLAN.md` | - | 2026-08-27 | 6 | ✓ |
| `compiler/IR_STACK_INTEGRATION_REVIEW.md` | - | 2026-08-02 | 31 | ✓ |
| `compiler/LSE_CHECKPOINT_CONTRACT.md` | - | 2026-07-27 | 37 | ✓ |
| `compiler/MATH_SOURCE_WORKSTREAM.md` | - | 2026-09-01 | 1 | ✓ |
| `compiler/MATRIX_CALCULUS_REVIEW.md` | - | 2026-08-20 | 13 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-08-08 | 25 | ✓ |
| `compiler/PDE_STENCIL_CAPABILITY_PLAN.md` | - | 2026-08-14 | 19 | ✓ |
| `compiler/README.md` | - | 2026-08-18 | 15 | ✓ |
| `compiler/RIEMANNIAN_OT_PLAN.md` | - | 2026-08-08 | 25 | ✓ |
| `compiler/SCHEDULE_OBJECT_DESIGN.md` | - | 2026-08-16 | 17 | ✓ |
| `compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md` | - | 2026-08-08 | 25 | ✓ |
| `compiler/SEQUENCE_MIXER_THEORY.md` | - | 2026-07-17 | 47 | ✓ |
| `compiler/SPARDA_REVIEW.md` | - | 2026-08-12 | 21 | ✓ |
| `compiler/TARGET_IR_REVIEW.md` | - | 2026-08-02 | 31 | ✓ |
| `compiler/TILERT_ASSESSMENT.md` | - | 2026-08-14 | 19 | ✓ |
| `compiler/TILESIGHT_ASSESSMENT.md` | - | 2026-07-30 | 34 | ✓ |
| `compiler/W1_1_TYPING_DESIGN.md` | - | 2026-08-18 | 15 | ✓ |
| `compiler/W1_1_TYPING_INVENTORY.md` | - | 2026-08-02 | 31 | ✓ |
| `compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md` | - | 2026-08-25 | 8 | ✓ |
| `compiler/compiler_enhancement.md` | - | 2026-08-18 | 15 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-08-11 | 22 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-06-11 | 83 | ✓ |
| `domain/GA_EBM_ARCHITECTURE_REVIEW.md` | - | 2026-08-02 | 31 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 64 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-08-12 | 21 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-08-11 | 22 | ✓ |
