# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-09-29**.

## Headline

- **159** docs catalogued across the canonical doc tree.
- **158** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **61** updated within the last 30 days.
- **40** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `APPLE_ARENA_NUMERICAL_POLICY.md` | - | 2026-09-12 | 17 | ✓ |
| `AUTODIFF_SPEC.md` | - | 2026-07-14 | 77 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-08-06 | 54 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 135 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 96 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 110 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-09-06 | 23 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 136 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-07-18 | 73 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-09-27 | 2 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 146 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-07-14 | 77 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-07-13 | 78 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 130 | ✓ |
| `NATIVE_ARTIFACT_SPEC.md` | Normative | 2026-07-19 | 72 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 116 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-07-23 | 68 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-07-18 | 73 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 130 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-08-24 | 36 | ✓ |
| `TILE_IR.md` | Normative | 2026-08-10 | 50 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-08-02 | 58 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 117 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 146 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 146 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 154 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 154 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 154 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 110 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 154 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-08-06 | 54 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-08-06 | 54 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 154 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-07-14 | 77 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-07-14 | 77 | ✓ |
| `porting_advanced_examples.md` | Informative | 2026-09-21 | 8 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-07-14 | 77 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 110 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-07-13 | 78 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-07-14 | 77 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 110 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 110 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 110 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 110 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 110 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 110 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 110 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 110 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 110 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-07-13 | 78 | ✓ |
| `backend_local_proofs.md` | - | 2026-07-15 | 76 | ✓ |
| `release_gates.md` | Normative | 2026-07-13 | 78 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-07-14 | 77 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-07-14 | 77 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-07-14 | 77 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-07-14 | 77 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-07-14 | 77 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-07-13 | 78 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-07-14 | 77 | ✓ |
| `README.md` | Informative | 2026-05-20 | 132 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 146 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | - | 2026-07-14 | 77 | ✓ |
| `compiler_test_architecture.md` | Normative | 2026-08-02 | 58 | ✓ |
| `distributed/megamoe.md` | - | 2026-06-09 | 112 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 132 | ✓ |
| `inference/serving.md` | - | 2026-07-13 | 78 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 156 | ✓ |
| `proposals/tile_fragment_abi.md` | Proposal | 2026-09-13 | 16 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-07-14 | 77 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 132 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 110 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 152 | ✓ |
| `workloads/attention-family.md` | Planning | 2026-07-14 | 77 | ✓ |
| `workloads/dflash.md` | - | 2026-07-14 | 77 | ✓ |
| `workloads/msa-cuda-phase3.md` | - | 2026-07-13 | 78 | ✓ |
| `workloads/msa.md` | - | 2026-07-13 | 78 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-07-13 | 78 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 132 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 132 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 141 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-09-27 | 2 | ✓ |
| `README.md` | - | 2026-09-06 | 23 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-09-05 | 24 | ✓ |
| `backend/E2E_COMPILATION_AUDIT.md` | - | 2026-09-05 | 24 | ✓ |
| `backend/README.md` | - | 2026-09-05 | 24 | ✓ |
| `backend/X86_AVX512_ABI_INVENTORY.md` | - | 2026-07-22 | 69 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-09-25 | 4 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-07-13 | 78 | ✓ |
| `backend/apple/README.md` | - | 2026-09-05 | 24 | ✓ |
| `backend/apple/todo.md` | - | 2026-09-29 | 0 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-09-05 | 24 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-09-05 | 24 | ✓ |
| `backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md` | - | 2026-09-27 | 2 | ✓ |
| `backend/nvidia/VERIFY_TARGET_IR_TAIL.md` | - | 2026-07-13 | 78 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 97 | ✓ |
| `backend/nvidia/todo.md` | - | 2026-09-29 | 0 | ✓ |
| `backend/rocm/GEMM_PERF_LADDER.md` | - | 2026-08-04 | 56 | ✓ |
| `backend/rocm/GFX125X_CDNA5_COMPILER_REFERENCE.md` | - | 2026-08-14 | 46 | ✓ |
| `backend/rocm/GIN_EXACT_DEVICE_RUNBOOK.md` | - | 2026-08-09 | 51 | ✓ |
| `backend/rocm/NATIVE_RDNA4_COMMISSIONING.md` | - | 2026-09-15 | 14 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-09-21 | 8 | ✓ |
| `backend/rocm/ROCM_LANE_MAP.md` | - | 2026-09-26 | 3 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-07-28 | 63 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-09-05 | 24 | ✓ |
| `backend/rocm/todo.md` | - | 2026-09-29 | 0 | ✓ |
| `backend/x86/todo.md` | - | 2026-09-29 | 0 | ✓ |
| `compiler/AMD_KERNEL_COMPILER_SURVEY.md` | - | 2026-09-22 | 7 | ✓ |
| `compiler/ANN_CALCULUS_DESIGN_SPIKE.md` | - | 2026-09-04 | 25 | ✓ |
| `compiler/AUTODIFF_ARCHITECTURE_REVIEW.md` | - | 2026-09-06 | 23 | ✓ |
| `compiler/AUTODIFF_EXECUTION_PLAN.md` | - | 2026-09-14 | 15 | ✓ |
| `compiler/AUTODIFF_NEXTGEN_PLAN.md` | - | 2026-09-06 | 23 | ✓ |
| `compiler/AUTODIFF_UNIFICATION_PLAN.md` | - | 2026-09-06 | 23 | ✓ |
| `compiler/BLOCK_ATTNRES_ROCM_PLAN.md` | - | 2026-09-09 | 20 | ✓ |
| `compiler/COMPILER_ARCHITECTURE_SWEEP.md` | - | 2026-09-09 | 20 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-09-16 | 13 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-09-08 | 21 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-28 | 63 | ✓ |
| `compiler/CORE_SUBSTRATE_VIEW.md` | - | 2026-09-07 | 22 | ✓ |
| `compiler/CUTE_IR_ASSESSMENT.md` | - | 2026-08-24 | 36 | ✓ |
| `compiler/DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | - | 2026-09-07 | 22 | ✓ |
| `compiler/EGGROLL_SUPPORT_PLAN.md` | - | 2026-09-07 | 22 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-08-08 | 52 | ✓ |
| `compiler/FORGE_ASSESSMENT.md` | - | 2026-09-07 | 22 | ✓ |
| `compiler/FRONTEND_GRAPH_SCHEDULE_REVIEW.md` | - | 2026-08-02 | 58 | ✓ |
| `compiler/FRONT_END_LOWERING_ASSESSMENT.md` | - | 2026-09-03 | 26 | ✓ |
| `compiler/FUNCTIONAL_ANALYSIS_TSOL_PLAN.md` | - | 2026-09-06 | 23 | ✓ |
| `compiler/GAME_THEORY_PLAN.md` | - | 2026-09-07 | 22 | ✓ |
| `compiler/HEAP_BARRIER_ARCHITECTURE_REVIEW.md` | - | 2026-09-11 | 18 | ✓ |
| `compiler/INTEGRATED_COMPILER_LOG.md` | - | 2026-09-29 | 0 | ✓ |
| `compiler/INTEGRATED_COMPILER_PLAN.md` | - | 2026-09-28 | 1 | ✓ |
| `compiler/INTRA_KERNEL_FEEDBACK_PLAN.md` | - | 2026-09-23 | 6 | ✓ |
| `compiler/IR_STACK_INTEGRATION_REVIEW.md` | - | 2026-08-02 | 58 | ✓ |
| `compiler/LSE_CHECKPOINT_CONTRACT.md` | - | 2026-07-27 | 64 | ✓ |
| `compiler/MATH_SOURCE_WORKSTREAM.md` | - | 2026-09-07 | 22 | ✓ |
| `compiler/MATRIX_CALCULUS_REVIEW.md` | - | 2026-09-07 | 22 | ✓ |
| `compiler/MLIR_NATIVE_FOUNDATION_SURVEY.md` | - | 2026-09-27 | 2 | ✓ |
| `compiler/ODS_OP_CONNECTION_TRIAGE.md` | - | 2026-09-28 | 1 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-08-08 | 52 | ✓ |
| `compiler/PDE_STENCIL_CAPABILITY_PLAN.md` | - | 2026-09-07 | 22 | ✓ |
| `compiler/README.md` | - | 2026-09-28 | 1 | ✓ |
| `compiler/RIEMANNIAN_OT_PLAN.md` | - | 2026-09-07 | 22 | ✓ |
| `compiler/SCHEDULE_OBJECT_DESIGN.md` | - | 2026-08-16 | 44 | ✓ |
| `compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md` | - | 2026-09-10 | 19 | ✓ |
| `compiler/SEQUENCE_MIXER_THEORY.md` | - | 2026-09-08 | 21 | ✓ |
| `compiler/SPARDA_REVIEW.md` | - | 2026-08-12 | 48 | ✓ |
| `compiler/TARGET_IR_REVIEW.md` | - | 2026-09-06 | 23 | ✓ |
| `compiler/TILERT_ASSESSMENT.md` | - | 2026-09-05 | 24 | ✓ |
| `compiler/TILESIGHT_ASSESSMENT.md` | - | 2026-07-30 | 61 | ✓ |
| `compiler/W1_1_TYPING_DESIGN.md` | - | 2026-09-04 | 25 | ✓ |
| `compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md` | - | 2026-08-25 | 35 | ✓ |
| `compiler/compiler_enhancement.md` | - | 2026-09-08 | 21 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-09-04 | 25 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-09-16 | 13 | ✓ |
| `domain/EBM_NATIVE_LOOP_ARCHITECTURE.md` | - | 2026-09-16 | 13 | ✓ |
| `domain/GA_EBM_ARCHITECTURE_REVIEW.md` | - | 2026-09-16 | 13 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 91 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-08-12 | 48 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-08-11 | 49 | ✓ |
