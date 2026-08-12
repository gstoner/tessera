# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-08-12**.

## Headline

- **135** docs catalogued across the canonical doc tree.
- **134** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **90** updated within the last 30 days.
- **13** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `AUTODIFF_SPEC.md` | - | 2026-07-14 | 29 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-08-06 | 6 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 87 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 48 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 62 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-08-11 | 1 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 88 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-07-18 | 25 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-07-14 | 29 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 98 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-07-14 | 29 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-07-13 | 30 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 82 | ✓ |
| `NATIVE_ARTIFACT_SPEC.md` | Normative | 2026-07-19 | 24 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 68 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-07-23 | 20 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-07-18 | 25 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 82 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-07-13 | 30 | ✓ |
| `TILE_IR.md` | Normative | 2026-08-10 | 2 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-08-02 | 10 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 69 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 98 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 98 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 106 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 106 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 106 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 62 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 106 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-08-06 | 6 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-08-06 | 6 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 106 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-07-14 | 29 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-07-14 | 29 | ✓ |
| `porting_advanced_examples.md` | Informative | 2026-05-09 | 95 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-07-14 | 29 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 62 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-07-13 | 30 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-07-14 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 62 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 62 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 62 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 62 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 62 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 62 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 62 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 62 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 62 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-07-13 | 30 | ✓ |
| `backend_local_proofs.md` | - | 2026-07-15 | 28 | ✓ |
| `release_gates.md` | Normative | 2026-07-13 | 30 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-07-14 | 29 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-07-14 | 29 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-07-14 | 29 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-07-14 | 29 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-07-14 | 29 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-07-13 | 30 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-07-14 | 29 | ✓ |
| `README.md` | Informative | 2026-05-20 | 84 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 98 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | - | 2026-07-14 | 29 | ✓ |
| `compiler_test_architecture.md` | Normative | 2026-08-02 | 10 | ✓ |
| `distributed/megamoe.md` | - | 2026-06-09 | 64 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 84 | ✓ |
| `inference/serving.md` | - | 2026-07-13 | 30 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 108 | ✓ |
| `proposals/tile_fragment_abi.md` | Proposal | 2026-07-19 | 24 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-07-14 | 29 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 84 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 62 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 104 | ✓ |
| `workloads/attention-family.md` | Planning | 2026-07-14 | 29 | ✓ |
| `workloads/dflash.md` | - | 2026-07-14 | 29 | ✓ |
| `workloads/msa-cuda-phase3.md` | - | 2026-07-13 | 30 | ✓ |
| `workloads/msa.md` | - | 2026-07-13 | 30 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-07-13 | 30 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 84 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 84 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 93 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-08-11 | 1 | ✓ |
| `README.md` | - | 2026-08-08 | 4 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-07-31 | 12 | ✓ |
| `backend/E2E_COMPILATION_AUDIT.md` | - | 2026-07-27 | 16 | ✓ |
| `backend/X86_AVX512_ABI_INVENTORY.md` | - | 2026-07-22 | 21 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-07-28 | 15 | ✓ |
| `backend/apple/APPLE_GPU_CODEGEN_PLAN.md` | - | 2026-07-13 | 30 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-07-13 | 30 | ✓ |
| `backend/apple/README.md` | - | 2026-07-13 | 30 | ✓ |
| `backend/apple/todo.md` | - | 2026-08-12 | 0 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-06-24 | 49 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-07-18 | 25 | ✓ |
| `backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md` | - | 2026-07-19 | 24 | ✓ |
| `backend/nvidia/VERIFY_TARGET_IR_TAIL.md` | - | 2026-07-13 | 30 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 49 | ✓ |
| `backend/nvidia/todo.md` | - | 2026-08-12 | 0 | ✓ |
| `backend/rocm/GEMM_PERF_LADDER.md` | - | 2026-08-04 | 8 | ✓ |
| `backend/rocm/GFX1250_MI450_COMPILER_REFERENCE.md` | - | 2026-07-28 | 15 | ✓ |
| `backend/rocm/GIN_EXACT_DEVICE_RUNBOOK.md` | - | 2026-08-09 | 3 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-08-06 | 6 | ✓ |
| `backend/rocm/ROCM_LANE_MAP.md` | - | 2026-08-05 | 7 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-07-28 | 15 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-06-23 | 50 | ✓ |
| `backend/rocm/todo.md` | - | 2026-08-12 | 0 | ✓ |
| `backend/x86/todo.md` | - | 2026-08-12 | 0 | ✓ |
| `compiler/AMD_KERNEL_COMPILER_SURVEY.md` | - | 2026-07-28 | 15 | ✓ |
| `compiler/AUTODIFF_ARCHITECTURE_REVIEW.md` | - | 2026-08-09 | 3 | ✓ |
| `compiler/AUTODIFF_UNIFICATION_PLAN.md` | - | 2026-08-10 | 2 | ✓ |
| `compiler/COMPILER_ARCHITECTURE_SWEEP.md` | - | 2026-08-11 | 1 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-08-10 | 2 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-08-08 | 4 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-28 | 15 | ✓ |
| `compiler/DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | - | 2026-08-08 | 4 | ✓ |
| `compiler/EGGROLL_SUPPORT_PLAN.md` | - | 2026-08-09 | 3 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-08-08 | 4 | ✓ |
| `compiler/FRONTEND_GRAPH_SCHEDULE_REVIEW.md` | - | 2026-08-02 | 10 | ✓ |
| `compiler/INTEGRATED_COMPILER_PLAN.md` | - | 2026-08-11 | 1 | ✓ |
| `compiler/IR_STACK_INTEGRATION_REVIEW.md` | - | 2026-08-02 | 10 | ✓ |
| `compiler/LSE_CHECKPOINT_CONTRACT.md` | - | 2026-07-27 | 16 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-08-08 | 4 | ✓ |
| `compiler/README.md` | - | 2026-08-11 | 1 | ✓ |
| `compiler/RIEMANNIAN_OT_PLAN.md` | - | 2026-08-08 | 4 | ✓ |
| `compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md` | - | 2026-08-08 | 4 | ✓ |
| `compiler/SEQUENCE_MIXER_THEORY.md` | - | 2026-07-17 | 26 | ✓ |
| `compiler/TARGET_IR_REVIEW.md` | - | 2026-08-02 | 10 | ✓ |
| `compiler/TILERT_ASSESSMENT.md` | - | 2026-08-11 | 1 | ✓ |
| `compiler/TILESIGHT_ASSESSMENT.md` | - | 2026-07-30 | 13 | ✓ |
| `compiler/W1_1_TYPING_DESIGN.md` | - | 2026-08-04 | 8 | ✓ |
| `compiler/W1_1_TYPING_INVENTORY.md` | - | 2026-08-02 | 10 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-08-11 | 1 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-06-11 | 62 | ✓ |
| `domain/GA_EBM_ARCHITECTURE_REVIEW.md` | - | 2026-08-02 | 10 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 43 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-08-12 | 0 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-08-11 | 1 | ✓ |
