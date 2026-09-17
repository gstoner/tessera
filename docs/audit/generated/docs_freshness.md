# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-09-17**.

## Headline

- **158** docs catalogued across the canonical doc tree.
- **157** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **58** updated within the last 30 days.
- **38** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `APPLE_ARENA_NUMERICAL_POLICY.md` | - | 2026-09-12 | 5 | ✓ |
| `AUTODIFF_SPEC.md` | - | 2026-07-14 | 65 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-08-06 | 42 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 123 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 84 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 98 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-09-06 | 11 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 124 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-07-18 | 61 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-07-14 | 65 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 134 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-07-14 | 65 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-07-13 | 66 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 118 | ✓ |
| `NATIVE_ARTIFACT_SPEC.md` | Normative | 2026-07-19 | 60 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 104 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-07-23 | 56 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-07-18 | 61 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 118 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-08-24 | 24 | ✓ |
| `TILE_IR.md` | Normative | 2026-08-10 | 38 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-08-02 | 46 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 105 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 134 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 134 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 142 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 142 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 142 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 98 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 142 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-08-06 | 42 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-08-06 | 42 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 142 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-07-14 | 65 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-07-14 | 65 | ✓ |
| `porting_advanced_examples.md` | Informative | 2026-05-09 | 131 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-07-14 | 65 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 98 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-07-13 | 66 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-07-14 | 65 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 98 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 98 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 98 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 98 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 98 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 98 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 98 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 98 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 98 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-07-13 | 66 | ✓ |
| `backend_local_proofs.md` | - | 2026-07-15 | 64 | ✓ |
| `release_gates.md` | Normative | 2026-07-13 | 66 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-07-14 | 65 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-07-14 | 65 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-07-14 | 65 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-07-14 | 65 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-07-14 | 65 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-07-13 | 66 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-07-14 | 65 | ✓ |
| `README.md` | Informative | 2026-05-20 | 120 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 134 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | - | 2026-07-14 | 65 | ✓ |
| `compiler_test_architecture.md` | Normative | 2026-08-02 | 46 | ✓ |
| `distributed/megamoe.md` | - | 2026-06-09 | 100 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 120 | ✓ |
| `inference/serving.md` | - | 2026-07-13 | 66 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 144 | ✓ |
| `proposals/tile_fragment_abi.md` | Proposal | 2026-09-13 | 4 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-07-14 | 65 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 120 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 98 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 140 | ✓ |
| `workloads/attention-family.md` | Planning | 2026-07-14 | 65 | ✓ |
| `workloads/dflash.md` | - | 2026-07-14 | 65 | ✓ |
| `workloads/msa-cuda-phase3.md` | - | 2026-07-13 | 66 | ✓ |
| `workloads/msa.md` | - | 2026-07-13 | 66 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-07-13 | 66 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 120 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 120 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 129 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-09-12 | 5 | ✓ |
| `README.md` | - | 2026-09-06 | 11 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-09-05 | 12 | ✓ |
| `backend/E2E_COMPILATION_AUDIT.md` | - | 2026-09-05 | 12 | ✓ |
| `backend/README.md` | - | 2026-09-05 | 12 | ✓ |
| `backend/X86_AVX512_ABI_INVENTORY.md` | - | 2026-07-22 | 57 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-09-05 | 12 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-07-13 | 66 | ✓ |
| `backend/apple/README.md` | - | 2026-09-05 | 12 | ✓ |
| `backend/apple/todo.md` | - | 2026-09-15 | 2 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-09-05 | 12 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-09-05 | 12 | ✓ |
| `backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md` | - | 2026-07-19 | 60 | ✓ |
| `backend/nvidia/VERIFY_TARGET_IR_TAIL.md` | - | 2026-07-13 | 66 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 85 | ✓ |
| `backend/nvidia/todo.md` | - | 2026-09-14 | 3 | ✓ |
| `backend/rocm/GEMM_PERF_LADDER.md` | - | 2026-08-04 | 44 | ✓ |
| `backend/rocm/GFX125X_CDNA5_COMPILER_REFERENCE.md` | - | 2026-08-14 | 34 | ✓ |
| `backend/rocm/GIN_EXACT_DEVICE_RUNBOOK.md` | - | 2026-08-09 | 39 | ✓ |
| `backend/rocm/NATIVE_RDNA4_COMMISSIONING.md` | - | 2026-09-15 | 2 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-09-05 | 12 | ✓ |
| `backend/rocm/ROCM_LANE_MAP.md` | - | 2026-08-05 | 43 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-07-28 | 51 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-09-05 | 12 | ✓ |
| `backend/rocm/todo.md` | - | 2026-09-14 | 3 | ✓ |
| `backend/x86/todo.md` | - | 2026-09-14 | 3 | ✓ |
| `compiler/AMD_KERNEL_COMPILER_SURVEY.md` | - | 2026-07-28 | 51 | ✓ |
| `compiler/ANN_CALCULUS_DESIGN_SPIKE.md` | - | 2026-09-04 | 13 | ✓ |
| `compiler/AUTODIFF_ARCHITECTURE_REVIEW.md` | - | 2026-09-06 | 11 | ✓ |
| `compiler/AUTODIFF_EXECUTION_PLAN.md` | - | 2026-09-14 | 3 | ✓ |
| `compiler/AUTODIFF_NEXTGEN_PLAN.md` | - | 2026-09-06 | 11 | ✓ |
| `compiler/AUTODIFF_UNIFICATION_PLAN.md` | - | 2026-09-06 | 11 | ✓ |
| `compiler/BLOCK_ATTNRES_ROCM_PLAN.md` | - | 2026-09-09 | 8 | ✓ |
| `compiler/COMPILER_ARCHITECTURE_SWEEP.md` | - | 2026-09-09 | 8 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-09-16 | 1 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-09-08 | 9 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-28 | 51 | ✓ |
| `compiler/CORE_SUBSTRATE_VIEW.md` | - | 2026-09-07 | 10 | ✓ |
| `compiler/CUTE_IR_ASSESSMENT.md` | - | 2026-08-24 | 24 | ✓ |
| `compiler/DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | - | 2026-09-07 | 10 | ✓ |
| `compiler/EGGROLL_SUPPORT_PLAN.md` | - | 2026-09-07 | 10 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-08-08 | 40 | ✓ |
| `compiler/FORGE_ASSESSMENT.md` | - | 2026-09-07 | 10 | ✓ |
| `compiler/FRONTEND_GRAPH_SCHEDULE_REVIEW.md` | - | 2026-08-02 | 46 | ✓ |
| `compiler/FRONT_END_LOWERING_ASSESSMENT.md` | - | 2026-09-03 | 14 | ✓ |
| `compiler/FUNCTIONAL_ANALYSIS_TSOL_PLAN.md` | - | 2026-09-06 | 11 | ✓ |
| `compiler/GAME_THEORY_PLAN.md` | - | 2026-09-07 | 10 | ✓ |
| `compiler/HEAP_BARRIER_ARCHITECTURE_REVIEW.md` | - | 2026-09-11 | 6 | ✓ |
| `compiler/INTEGRATED_COMPILER_LOG.md` | - | 2026-09-17 | 0 | ✓ |
| `compiler/INTEGRATED_COMPILER_PLAN.md` | - | 2026-09-15 | 2 | ✓ |
| `compiler/INTRA_KERNEL_FEEDBACK_PLAN.md` | - | 2026-09-04 | 13 | ✓ |
| `compiler/IR_STACK_INTEGRATION_REVIEW.md` | - | 2026-08-02 | 46 | ✓ |
| `compiler/LSE_CHECKPOINT_CONTRACT.md` | - | 2026-07-27 | 52 | ✓ |
| `compiler/MATH_SOURCE_WORKSTREAM.md` | - | 2026-09-07 | 10 | ✓ |
| `compiler/MATRIX_CALCULUS_REVIEW.md` | - | 2026-09-07 | 10 | ✓ |
| `compiler/MLIR_NATIVE_FOUNDATION_SURVEY.md` | - | 2026-09-08 | 9 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-08-08 | 40 | ✓ |
| `compiler/PDE_STENCIL_CAPABILITY_PLAN.md` | - | 2026-09-07 | 10 | ✓ |
| `compiler/README.md` | - | 2026-09-11 | 6 | ✓ |
| `compiler/RIEMANNIAN_OT_PLAN.md` | - | 2026-09-07 | 10 | ✓ |
| `compiler/SCHEDULE_OBJECT_DESIGN.md` | - | 2026-08-16 | 32 | ✓ |
| `compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md` | - | 2026-09-10 | 7 | ✓ |
| `compiler/SEQUENCE_MIXER_THEORY.md` | - | 2026-09-08 | 9 | ✓ |
| `compiler/SPARDA_REVIEW.md` | - | 2026-08-12 | 36 | ✓ |
| `compiler/TARGET_IR_REVIEW.md` | - | 2026-09-06 | 11 | ✓ |
| `compiler/TILERT_ASSESSMENT.md` | - | 2026-09-05 | 12 | ✓ |
| `compiler/TILESIGHT_ASSESSMENT.md` | - | 2026-07-30 | 49 | ✓ |
| `compiler/W1_1_TYPING_DESIGN.md` | - | 2026-09-04 | 13 | ✓ |
| `compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md` | - | 2026-08-25 | 23 | ✓ |
| `compiler/compiler_enhancement.md` | - | 2026-09-08 | 9 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-09-04 | 13 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-09-16 | 1 | ✓ |
| `domain/EBM_NATIVE_LOOP_ARCHITECTURE.md` | - | 2026-09-16 | 1 | ✓ |
| `domain/GA_EBM_ARCHITECTURE_REVIEW.md` | - | 2026-09-16 | 1 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 79 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-08-12 | 36 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-08-11 | 37 | ✓ |
