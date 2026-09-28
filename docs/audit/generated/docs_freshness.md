# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-09-28**.

## Headline

- **159** docs catalogued across the canonical doc tree.
- **158** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **61** updated within the last 30 days.
- **39** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `APPLE_ARENA_NUMERICAL_POLICY.md` | - | 2026-09-12 | 16 | ✓ |
| `AUTODIFF_SPEC.md` | - | 2026-07-14 | 76 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-08-06 | 53 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 134 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 95 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 109 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-09-06 | 22 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 135 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-07-18 | 72 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-09-27 | 1 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 145 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-07-14 | 76 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-07-13 | 77 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 129 | ✓ |
| `NATIVE_ARTIFACT_SPEC.md` | Normative | 2026-07-19 | 71 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 115 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-07-23 | 67 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-07-18 | 72 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 129 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-08-24 | 35 | ✓ |
| `TILE_IR.md` | Normative | 2026-08-10 | 49 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-08-02 | 57 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 116 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 145 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 145 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 153 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 153 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 153 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 109 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 153 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-08-06 | 53 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-08-06 | 53 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 153 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-07-14 | 76 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-07-14 | 76 | ✓ |
| `porting_advanced_examples.md` | Informative | 2026-09-21 | 7 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-07-14 | 76 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 109 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-07-13 | 77 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-07-14 | 76 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 109 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 109 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 109 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 109 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 109 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 109 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 109 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 109 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 109 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-07-13 | 77 | ✓ |
| `backend_local_proofs.md` | - | 2026-07-15 | 75 | ✓ |
| `release_gates.md` | Normative | 2026-07-13 | 77 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-07-14 | 76 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-07-14 | 76 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-07-14 | 76 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-07-14 | 76 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-07-14 | 76 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-07-13 | 77 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-07-14 | 76 | ✓ |
| `README.md` | Informative | 2026-05-20 | 131 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 145 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | - | 2026-07-14 | 76 | ✓ |
| `compiler_test_architecture.md` | Normative | 2026-08-02 | 57 | ✓ |
| `distributed/megamoe.md` | - | 2026-06-09 | 111 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 131 | ✓ |
| `inference/serving.md` | - | 2026-07-13 | 77 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 155 | ✓ |
| `proposals/tile_fragment_abi.md` | Proposal | 2026-09-13 | 15 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-07-14 | 76 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 131 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 109 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 151 | ✓ |
| `workloads/attention-family.md` | Planning | 2026-07-14 | 76 | ✓ |
| `workloads/dflash.md` | - | 2026-07-14 | 76 | ✓ |
| `workloads/msa-cuda-phase3.md` | - | 2026-07-13 | 77 | ✓ |
| `workloads/msa.md` | - | 2026-07-13 | 77 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-07-13 | 77 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 131 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 131 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 140 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-09-27 | 1 | ✓ |
| `README.md` | - | 2026-09-06 | 22 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-09-05 | 23 | ✓ |
| `backend/E2E_COMPILATION_AUDIT.md` | - | 2026-09-05 | 23 | ✓ |
| `backend/README.md` | - | 2026-09-05 | 23 | ✓ |
| `backend/X86_AVX512_ABI_INVENTORY.md` | - | 2026-07-22 | 68 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-09-25 | 3 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-07-13 | 77 | ✓ |
| `backend/apple/README.md` | - | 2026-09-05 | 23 | ✓ |
| `backend/apple/todo.md` | - | 2026-09-27 | 1 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-09-05 | 23 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-09-05 | 23 | ✓ |
| `backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md` | - | 2026-09-27 | 1 | ✓ |
| `backend/nvidia/VERIFY_TARGET_IR_TAIL.md` | - | 2026-07-13 | 77 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 96 | ✓ |
| `backend/nvidia/todo.md` | - | 2026-09-27 | 1 | ✓ |
| `backend/rocm/GEMM_PERF_LADDER.md` | - | 2026-08-04 | 55 | ✓ |
| `backend/rocm/GFX125X_CDNA5_COMPILER_REFERENCE.md` | - | 2026-08-14 | 45 | ✓ |
| `backend/rocm/GIN_EXACT_DEVICE_RUNBOOK.md` | - | 2026-08-09 | 50 | ✓ |
| `backend/rocm/NATIVE_RDNA4_COMMISSIONING.md` | - | 2026-09-15 | 13 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-09-21 | 7 | ✓ |
| `backend/rocm/ROCM_LANE_MAP.md` | - | 2026-09-26 | 2 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-07-28 | 62 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-09-05 | 23 | ✓ |
| `backend/rocm/todo.md` | - | 2026-09-27 | 1 | ✓ |
| `backend/x86/todo.md` | - | 2026-09-27 | 1 | ✓ |
| `compiler/AMD_KERNEL_COMPILER_SURVEY.md` | - | 2026-09-22 | 6 | ✓ |
| `compiler/ANN_CALCULUS_DESIGN_SPIKE.md` | - | 2026-09-04 | 24 | ✓ |
| `compiler/AUTODIFF_ARCHITECTURE_REVIEW.md` | - | 2026-09-06 | 22 | ✓ |
| `compiler/AUTODIFF_EXECUTION_PLAN.md` | - | 2026-09-14 | 14 | ✓ |
| `compiler/AUTODIFF_NEXTGEN_PLAN.md` | - | 2026-09-06 | 22 | ✓ |
| `compiler/AUTODIFF_UNIFICATION_PLAN.md` | - | 2026-09-06 | 22 | ✓ |
| `compiler/BLOCK_ATTNRES_ROCM_PLAN.md` | - | 2026-09-09 | 19 | ✓ |
| `compiler/COMPILER_ARCHITECTURE_SWEEP.md` | - | 2026-09-09 | 19 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-09-16 | 12 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-09-08 | 20 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-28 | 62 | ✓ |
| `compiler/CORE_SUBSTRATE_VIEW.md` | - | 2026-09-07 | 21 | ✓ |
| `compiler/CUTE_IR_ASSESSMENT.md` | - | 2026-08-24 | 35 | ✓ |
| `compiler/DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | - | 2026-09-07 | 21 | ✓ |
| `compiler/EGGROLL_SUPPORT_PLAN.md` | - | 2026-09-07 | 21 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-08-08 | 51 | ✓ |
| `compiler/FORGE_ASSESSMENT.md` | - | 2026-09-07 | 21 | ✓ |
| `compiler/FRONTEND_GRAPH_SCHEDULE_REVIEW.md` | - | 2026-08-02 | 57 | ✓ |
| `compiler/FRONT_END_LOWERING_ASSESSMENT.md` | - | 2026-09-03 | 25 | ✓ |
| `compiler/FUNCTIONAL_ANALYSIS_TSOL_PLAN.md` | - | 2026-09-06 | 22 | ✓ |
| `compiler/GAME_THEORY_PLAN.md` | - | 2026-09-07 | 21 | ✓ |
| `compiler/HEAP_BARRIER_ARCHITECTURE_REVIEW.md` | - | 2026-09-11 | 17 | ✓ |
| `compiler/INTEGRATED_COMPILER_LOG.md` | - | 2026-09-27 | 1 | ✓ |
| `compiler/INTEGRATED_COMPILER_PLAN.md` | - | 2026-09-27 | 1 | ✓ |
| `compiler/INTRA_KERNEL_FEEDBACK_PLAN.md` | - | 2026-09-23 | 5 | ✓ |
| `compiler/IR_STACK_INTEGRATION_REVIEW.md` | - | 2026-08-02 | 57 | ✓ |
| `compiler/LSE_CHECKPOINT_CONTRACT.md` | - | 2026-07-27 | 63 | ✓ |
| `compiler/MATH_SOURCE_WORKSTREAM.md` | - | 2026-09-07 | 21 | ✓ |
| `compiler/MATRIX_CALCULUS_REVIEW.md` | - | 2026-09-07 | 21 | ✓ |
| `compiler/MLIR_NATIVE_FOUNDATION_SURVEY.md` | - | 2026-09-27 | 1 | ✓ |
| `compiler/ODS_OP_CONNECTION_TRIAGE.md` | - | 2026-09-27 | 1 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-08-08 | 51 | ✓ |
| `compiler/PDE_STENCIL_CAPABILITY_PLAN.md` | - | 2026-09-07 | 21 | ✓ |
| `compiler/README.md` | - | 2026-09-11 | 17 | ✓ |
| `compiler/RIEMANNIAN_OT_PLAN.md` | - | 2026-09-07 | 21 | ✓ |
| `compiler/SCHEDULE_OBJECT_DESIGN.md` | - | 2026-08-16 | 43 | ✓ |
| `compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md` | - | 2026-09-10 | 18 | ✓ |
| `compiler/SEQUENCE_MIXER_THEORY.md` | - | 2026-09-08 | 20 | ✓ |
| `compiler/SPARDA_REVIEW.md` | - | 2026-08-12 | 47 | ✓ |
| `compiler/TARGET_IR_REVIEW.md` | - | 2026-09-06 | 22 | ✓ |
| `compiler/TILERT_ASSESSMENT.md` | - | 2026-09-05 | 23 | ✓ |
| `compiler/TILESIGHT_ASSESSMENT.md` | - | 2026-07-30 | 60 | ✓ |
| `compiler/W1_1_TYPING_DESIGN.md` | - | 2026-09-04 | 24 | ✓ |
| `compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md` | - | 2026-08-25 | 34 | ✓ |
| `compiler/compiler_enhancement.md` | - | 2026-09-08 | 20 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-09-04 | 24 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-09-16 | 12 | ✓ |
| `domain/EBM_NATIVE_LOOP_ARCHITECTURE.md` | - | 2026-09-16 | 12 | ✓ |
| `domain/GA_EBM_ARCHITECTURE_REVIEW.md` | - | 2026-09-16 | 12 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 90 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-08-12 | 47 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-08-11 | 48 | ✓ |
