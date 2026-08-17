# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-08-17**.

## Headline

- **144** docs catalogued across the canonical doc tree.
- **143** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **66** updated within the last 30 days.
- **15** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `AUTODIFF_SPEC.md` | - | 2026-07-14 | 34 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-08-06 | 11 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 92 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 53 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 67 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-08-11 | 6 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 93 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-07-18 | 30 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-07-14 | 34 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 103 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-07-14 | 34 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-07-13 | 35 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 87 | ✓ |
| `NATIVE_ARTIFACT_SPEC.md` | Normative | 2026-07-19 | 29 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 73 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-07-23 | 25 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-07-18 | 30 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 87 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-07-13 | 35 | ✓ |
| `TILE_IR.md` | Normative | 2026-08-10 | 7 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-08-02 | 15 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 74 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 103 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 103 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 111 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 111 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 111 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 67 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 111 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-08-06 | 11 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-08-06 | 11 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 111 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-07-14 | 34 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-07-14 | 34 | ✓ |
| `porting_advanced_examples.md` | Informative | 2026-05-09 | 100 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-07-14 | 34 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 67 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-07-13 | 35 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-07-14 | 34 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 67 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 67 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 67 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 67 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 67 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 67 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 67 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 67 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 67 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-07-13 | 35 | ✓ |
| `backend_local_proofs.md` | - | 2026-07-15 | 33 | ✓ |
| `release_gates.md` | Normative | 2026-07-13 | 35 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-07-14 | 34 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-07-14 | 34 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-07-14 | 34 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-07-14 | 34 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-07-14 | 34 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-07-13 | 35 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-07-14 | 34 | ✓ |
| `README.md` | Informative | 2026-05-20 | 89 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 103 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | - | 2026-07-14 | 34 | ✓ |
| `compiler_test_architecture.md` | Normative | 2026-08-02 | 15 | ✓ |
| `distributed/megamoe.md` | - | 2026-06-09 | 69 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 89 | ✓ |
| `inference/serving.md` | - | 2026-07-13 | 35 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 113 | ✓ |
| `proposals/tile_fragment_abi.md` | Proposal | 2026-07-19 | 29 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-07-14 | 34 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 89 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 67 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 109 | ✓ |
| `workloads/attention-family.md` | Planning | 2026-07-14 | 34 | ✓ |
| `workloads/dflash.md` | - | 2026-07-14 | 34 | ✓ |
| `workloads/msa-cuda-phase3.md` | - | 2026-07-13 | 35 | ✓ |
| `workloads/msa.md` | - | 2026-07-13 | 35 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-07-13 | 35 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 89 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 89 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 98 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-08-11 | 6 | ✓ |
| `README.md` | - | 2026-08-08 | 9 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-07-31 | 17 | ✓ |
| `backend/E2E_COMPILATION_AUDIT.md` | - | 2026-07-27 | 21 | ✓ |
| `backend/X86_AVX512_ABI_INVENTORY.md` | - | 2026-07-22 | 26 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-07-28 | 20 | ✓ |
| `backend/apple/APPLE_GPU_CODEGEN_PLAN.md` | - | 2026-07-13 | 35 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-07-13 | 35 | ✓ |
| `backend/apple/README.md` | - | 2026-07-13 | 35 | ✓ |
| `backend/apple/todo.md` | - | 2026-08-16 | 1 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-06-24 | 54 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-07-18 | 30 | ✓ |
| `backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md` | - | 2026-07-19 | 29 | ✓ |
| `backend/nvidia/VERIFY_TARGET_IR_TAIL.md` | - | 2026-07-13 | 35 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 54 | ✓ |
| `backend/nvidia/todo.md` | - | 2026-08-16 | 1 | ✓ |
| `backend/rocm/GEMM_PERF_LADDER.md` | - | 2026-08-04 | 13 | ✓ |
| `backend/rocm/GFX125X_CDNA5_COMPILER_REFERENCE.md` | - | 2026-08-14 | 3 | ✓ |
| `backend/rocm/GIN_EXACT_DEVICE_RUNBOOK.md` | - | 2026-08-09 | 8 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-08-14 | 3 | ✓ |
| `backend/rocm/ROCM_LANE_MAP.md` | - | 2026-08-05 | 12 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-07-28 | 20 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-06-23 | 55 | ✓ |
| `backend/rocm/todo.md` | - | 2026-08-16 | 1 | ✓ |
| `backend/x86/todo.md` | - | 2026-08-16 | 1 | ✓ |
| `compiler/AMD_KERNEL_COMPILER_SURVEY.md` | - | 2026-07-28 | 20 | ✓ |
| `compiler/AUTODIFF_ARCHITECTURE_REVIEW.md` | - | 2026-08-09 | 8 | ✓ |
| `compiler/AUTODIFF_UNIFICATION_PLAN.md` | - | 2026-08-10 | 7 | ✓ |
| `compiler/BLOCK_ATTNRES_ROCM_PLAN.md` | - | 2026-08-13 | 4 | ✓ |
| `compiler/COMPILER_ARCHITECTURE_SWEEP.md` | - | 2026-08-11 | 6 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-08-10 | 7 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-08-08 | 9 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-28 | 20 | ✓ |
| `compiler/CORE_SUBSTRATE_VIEW.md` | - | 2026-08-15 | 2 | ✓ |
| `compiler/CUTE_IR_ASSESSMENT.md` | - | 2026-08-16 | 1 | ✓ |
| `compiler/DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | - | 2026-08-08 | 9 | ✓ |
| `compiler/EGGROLL_SUPPORT_PLAN.md` | - | 2026-08-09 | 8 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-08-08 | 9 | ✓ |
| `compiler/FORGE_ASSESSMENT.md` | - | 2026-08-15 | 2 | ✓ |
| `compiler/FRONTEND_GRAPH_SCHEDULE_REVIEW.md` | - | 2026-08-02 | 15 | ✓ |
| `compiler/GAME_THEORY_PLAN.md` | - | 2026-08-15 | 2 | ✓ |
| `compiler/INTEGRATED_COMPILER_PLAN.md` | - | 2026-08-16 | 1 | ✓ |
| `compiler/IR_STACK_INTEGRATION_REVIEW.md` | - | 2026-08-02 | 15 | ✓ |
| `compiler/LSE_CHECKPOINT_CONTRACT.md` | - | 2026-07-27 | 21 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-08-08 | 9 | ✓ |
| `compiler/PDE_STENCIL_CAPABILITY_PLAN.md` | - | 2026-08-14 | 3 | ✓ |
| `compiler/README.md` | - | 2026-08-14 | 3 | ✓ |
| `compiler/RIEMANNIAN_OT_PLAN.md` | - | 2026-08-08 | 9 | ✓ |
| `compiler/SCHEDULE_OBJECT_DESIGN.md` | - | 2026-08-16 | 1 | ✓ |
| `compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md` | - | 2026-08-08 | 9 | ✓ |
| `compiler/SEQUENCE_MIXER_THEORY.md` | - | 2026-07-17 | 31 | ✓ |
| `compiler/SPARDA_REVIEW.md` | - | 2026-08-12 | 5 | ✓ |
| `compiler/TARGET_IR_REVIEW.md` | - | 2026-08-02 | 15 | ✓ |
| `compiler/TILERT_ASSESSMENT.md` | - | 2026-08-14 | 3 | ✓ |
| `compiler/TILESIGHT_ASSESSMENT.md` | - | 2026-07-30 | 18 | ✓ |
| `compiler/W1_1_TYPING_DESIGN.md` | - | 2026-08-04 | 13 | ✓ |
| `compiler/W1_1_TYPING_INVENTORY.md` | - | 2026-08-02 | 15 | ✓ |
| `compiler/compiler_enhancement.md` | - | 2026-08-15 | 2 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-08-11 | 6 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-06-11 | 67 | ✓ |
| `domain/GA_EBM_ARCHITECTURE_REVIEW.md` | - | 2026-08-02 | 15 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 48 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-08-12 | 5 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-08-11 | 6 | ✓ |
