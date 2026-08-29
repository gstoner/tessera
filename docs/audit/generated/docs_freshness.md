# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-08-29**.

## Headline

- **150** docs catalogued across the canonical doc tree.
- **149** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **59** updated within the last 30 days.
- **22** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `AUTODIFF_SPEC.md` | - | 2026-07-14 | 46 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-08-06 | 23 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 104 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 65 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 79 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-08-11 | 18 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 105 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-07-18 | 42 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-07-14 | 46 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 115 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-07-14 | 46 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-07-13 | 47 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 99 | ✓ |
| `NATIVE_ARTIFACT_SPEC.md` | Normative | 2026-07-19 | 41 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 85 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-07-23 | 37 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-07-18 | 42 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 99 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-08-24 | 5 | ✓ |
| `TILE_IR.md` | Normative | 2026-08-10 | 19 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-08-02 | 27 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 86 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 115 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 115 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 123 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 123 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 123 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 79 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 123 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-08-06 | 23 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-08-06 | 23 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 123 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-07-14 | 46 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-07-14 | 46 | ✓ |
| `porting_advanced_examples.md` | Informative | 2026-05-09 | 112 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-07-14 | 46 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 79 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-07-13 | 47 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-07-14 | 46 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 79 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 79 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 79 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 79 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 79 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 79 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 79 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 79 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 79 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-07-13 | 47 | ✓ |
| `backend_local_proofs.md` | - | 2026-07-15 | 45 | ✓ |
| `release_gates.md` | Normative | 2026-07-13 | 47 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-07-14 | 46 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-07-14 | 46 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-07-14 | 46 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-07-14 | 46 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-07-14 | 46 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-07-13 | 47 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-07-14 | 46 | ✓ |
| `README.md` | Informative | 2026-05-20 | 101 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 115 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | - | 2026-07-14 | 46 | ✓ |
| `compiler_test_architecture.md` | Normative | 2026-08-02 | 27 | ✓ |
| `distributed/megamoe.md` | - | 2026-06-09 | 81 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 101 | ✓ |
| `inference/serving.md` | - | 2026-07-13 | 47 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 125 | ✓ |
| `proposals/tile_fragment_abi.md` | Proposal | 2026-07-19 | 41 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-07-14 | 46 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 101 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 79 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 121 | ✓ |
| `workloads/attention-family.md` | Planning | 2026-07-14 | 46 | ✓ |
| `workloads/dflash.md` | - | 2026-07-14 | 46 | ✓ |
| `workloads/msa-cuda-phase3.md` | - | 2026-07-13 | 47 | ✓ |
| `workloads/msa.md` | - | 2026-07-13 | 47 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-07-13 | 47 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 101 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 101 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 110 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-08-11 | 18 | ✓ |
| `README.md` | - | 2026-08-08 | 21 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-07-31 | 29 | ✓ |
| `backend/E2E_COMPILATION_AUDIT.md` | - | 2026-07-27 | 33 | ✓ |
| `backend/X86_AVX512_ABI_INVENTORY.md` | - | 2026-07-22 | 38 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-07-28 | 32 | ✓ |
| `backend/apple/APPLE_GPU_CODEGEN_PLAN.md` | - | 2026-07-13 | 47 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-07-13 | 47 | ✓ |
| `backend/apple/README.md` | - | 2026-07-13 | 47 | ✓ |
| `backend/apple/todo.md` | - | 2026-08-29 | 0 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-06-24 | 66 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-07-18 | 42 | ✓ |
| `backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md` | - | 2026-07-19 | 41 | ✓ |
| `backend/nvidia/VERIFY_TARGET_IR_TAIL.md` | - | 2026-07-13 | 47 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 66 | ✓ |
| `backend/nvidia/todo.md` | - | 2026-08-29 | 0 | ✓ |
| `backend/rocm/GEMM_PERF_LADDER.md` | - | 2026-08-04 | 25 | ✓ |
| `backend/rocm/GFX125X_CDNA5_COMPILER_REFERENCE.md` | - | 2026-08-14 | 15 | ✓ |
| `backend/rocm/GIN_EXACT_DEVICE_RUNBOOK.md` | - | 2026-08-09 | 20 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-08-14 | 15 | ✓ |
| `backend/rocm/ROCM_LANE_MAP.md` | - | 2026-08-05 | 24 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-07-28 | 32 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-06-23 | 67 | ✓ |
| `backend/rocm/todo.md` | - | 2026-08-29 | 0 | ✓ |
| `backend/x86/todo.md` | - | 2026-08-29 | 0 | ✓ |
| `compiler/AMD_KERNEL_COMPILER_SURVEY.md` | - | 2026-07-28 | 32 | ✓ |
| `compiler/AUTODIFF_ARCHITECTURE_REVIEW.md` | - | 2026-08-18 | 11 | ✓ |
| `compiler/AUTODIFF_NEXTGEN_PLAN.md` | - | 2026-08-20 | 9 | ✓ |
| `compiler/AUTODIFF_UNIFICATION_PLAN.md` | - | 2026-08-18 | 11 | ✓ |
| `compiler/BLOCK_ATTNRES_ROCM_PLAN.md` | - | 2026-08-13 | 16 | ✓ |
| `compiler/CODE_REVIEW_2026-08-29.md` | - | 2026-08-29 | 0 | ✓ |
| `compiler/COMPILER_ARCHITECTURE_SWEEP.md` | - | 2026-08-11 | 18 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-08-10 | 19 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-08-08 | 21 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-28 | 32 | ✓ |
| `compiler/CORE_SUBSTRATE_VIEW.md` | - | 2026-08-24 | 5 | ✓ |
| `compiler/CUTE_IR_ASSESSMENT.md` | - | 2026-08-24 | 5 | ✓ |
| `compiler/DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | - | 2026-08-08 | 21 | ✓ |
| `compiler/EGGROLL_SUPPORT_PLAN.md` | - | 2026-08-09 | 20 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-08-08 | 21 | ✓ |
| `compiler/FORGE_ASSESSMENT.md` | - | 2026-08-15 | 14 | ✓ |
| `compiler/FRONTEND_GRAPH_SCHEDULE_REVIEW.md` | - | 2026-08-02 | 27 | ✓ |
| `compiler/FUNCTIONAL_ANALYSIS_TSOL_PLAN.md` | - | 2026-08-22 | 7 | ✓ |
| `compiler/GAME_THEORY_PLAN.md` | - | 2026-08-15 | 14 | ✓ |
| `compiler/INTEGRATED_COMPILER_PLAN.md` | - | 2026-08-24 | 5 | ✓ |
| `compiler/INTRA_KERNEL_FEEDBACK_PLAN.md` | - | 2026-08-27 | 2 | ✓ |
| `compiler/IR_STACK_INTEGRATION_REVIEW.md` | - | 2026-08-02 | 27 | ✓ |
| `compiler/LSE_CHECKPOINT_CONTRACT.md` | - | 2026-07-27 | 33 | ✓ |
| `compiler/MATRIX_CALCULUS_REVIEW.md` | - | 2026-08-20 | 9 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-08-08 | 21 | ✓ |
| `compiler/PDE_STENCIL_CAPABILITY_PLAN.md` | - | 2026-08-14 | 15 | ✓ |
| `compiler/README.md` | - | 2026-08-18 | 11 | ✓ |
| `compiler/RIEMANNIAN_OT_PLAN.md` | - | 2026-08-08 | 21 | ✓ |
| `compiler/SCHEDULE_OBJECT_DESIGN.md` | - | 2026-08-16 | 13 | ✓ |
| `compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md` | - | 2026-08-08 | 21 | ✓ |
| `compiler/SEQUENCE_MIXER_THEORY.md` | - | 2026-07-17 | 43 | ✓ |
| `compiler/SPARDA_REVIEW.md` | - | 2026-08-12 | 17 | ✓ |
| `compiler/TARGET_IR_REVIEW.md` | - | 2026-08-02 | 27 | ✓ |
| `compiler/TILERT_ASSESSMENT.md` | - | 2026-08-14 | 15 | ✓ |
| `compiler/TILESIGHT_ASSESSMENT.md` | - | 2026-07-30 | 30 | ✓ |
| `compiler/W1_1_TYPING_DESIGN.md` | - | 2026-08-18 | 11 | ✓ |
| `compiler/W1_1_TYPING_INVENTORY.md` | - | 2026-08-02 | 27 | ✓ |
| `compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md` | - | 2026-08-25 | 4 | ✓ |
| `compiler/compiler_enhancement.md` | - | 2026-08-18 | 11 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-08-11 | 18 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-06-11 | 79 | ✓ |
| `domain/GA_EBM_ARCHITECTURE_REVIEW.md` | - | 2026-08-02 | 27 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 60 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-08-12 | 17 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-08-11 | 18 | ✓ |
