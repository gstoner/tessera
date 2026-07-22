# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-07-22**.

## Headline

- **121** docs catalogued across the canonical doc tree.
- **120** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **76** updated within the last 30 days.
- **0** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `AUTODIFF_SPEC.md` | - | 2026-07-14 | 8 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-05-01 | 82 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 66 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 27 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 41 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-06-30 | 22 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 67 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-07-18 | 4 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-07-14 | 8 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 77 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-07-14 | 8 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-07-13 | 9 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 61 | ✓ |
| `NATIVE_ARTIFACT_SPEC.md` | Normative | 2026-07-19 | 3 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 47 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-07-14 | 8 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-07-18 | 4 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 61 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-07-13 | 9 | ✓ |
| `TILE_IR.md` | Normative | 2026-05-22 | 61 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-07-13 | 9 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 48 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 77 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 77 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 85 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 85 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 85 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 41 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 85 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-06-21 | 31 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-07-13 | 9 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 85 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-07-14 | 8 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-07-14 | 8 | ✓ |
| `porting_advanced_examples.md` | Informative | 2026-05-09 | 74 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-07-14 | 8 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 41 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-07-13 | 9 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-07-14 | 8 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 41 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 41 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 41 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 41 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 41 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 41 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 41 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 41 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 41 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-07-13 | 9 | ✓ |
| `backend_local_proofs.md` | - | 2026-07-15 | 7 | ✓ |
| `release_gates.md` | Normative | 2026-07-13 | 9 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-07-14 | 8 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-07-14 | 8 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-07-14 | 8 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-07-14 | 8 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-07-14 | 8 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-07-13 | 9 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-07-14 | 8 | ✓ |
| `README.md` | Informative | 2026-05-20 | 63 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 77 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | - | 2026-07-14 | 8 | ✓ |
| `compiler_test_architecture.md` | Normative | 2026-07-15 | 7 | ✓ |
| `distributed/megamoe.md` | - | 2026-06-09 | 43 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 63 | ✓ |
| `inference/serving.md` | - | 2026-07-13 | 9 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 87 | ✓ |
| `proposals/tile_fragment_abi.md` | Proposal | 2026-07-19 | 3 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-07-14 | 8 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 63 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 41 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 83 | ✓ |
| `workloads/attention-family.md` | Planning | 2026-07-14 | 8 | ✓ |
| `workloads/dflash.md` | - | 2026-07-14 | 8 | ✓ |
| `workloads/msa-cuda-phase3.md` | - | 2026-07-13 | 9 | ✓ |
| `workloads/msa.md` | - | 2026-07-13 | 9 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-07-13 | 9 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 63 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 63 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 72 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-07-22 | 0 | ✓ |
| `README.md` | - | 2026-07-11 | 11 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-07-12 | 10 | ✓ |
| `backend/E2E_COMPILATION_AUDIT.md` | - | 2026-07-20 | 2 | ✓ |
| `backend/X86_AVX512_ABI_INVENTORY.md` | - | 2026-07-20 | 2 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-07-13 | 9 | ✓ |
| `backend/apple/APPLE_GPU_CODEGEN_PLAN.md` | - | 2026-07-13 | 9 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-07-13 | 9 | ✓ |
| `backend/apple/README.md` | - | 2026-07-13 | 9 | ✓ |
| `backend/apple/todo.md` | - | 2026-07-22 | 0 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-06-24 | 28 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-07-18 | 4 | ✓ |
| `backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md` | - | 2026-07-19 | 3 | ✓ |
| `backend/nvidia/VERIFY_TARGET_IR_TAIL.md` | - | 2026-07-13 | 9 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 28 | ✓ |
| `backend/nvidia/todo.md` | - | 2026-07-22 | 0 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-07-16 | 6 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-07-13 | 9 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-06-23 | 29 | ✓ |
| `backend/rocm/todo.md` | - | 2026-07-22 | 0 | ✓ |
| `compiler/AUTODIFF_UNIFICATION_PLAN.md` | - | 2026-07-14 | 8 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-07-22 | 0 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-07-22 | 0 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-02 | 20 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-07-22 | 0 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-07-14 | 8 | ✓ |
| `compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md` | - | 2026-07-17 | 5 | ✓ |
| `compiler/SEQUENCE_MIXER_THEORY.md` | - | 2026-07-17 | 5 | ✓ |
| `compiler/STAGE_A_EMIT_PLAN.md` | - | 2026-07-11 | 11 | ✓ |
| `compiler/WORKSTREAM_C_HANDOFF.md` | - | 2026-07-06 | 16 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-06-21 | 31 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-06-11 | 41 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 22 | ✓ |
| `roadmap/CONTRACT_PASS_PLAN.md` | - | 2026-06-20 | 32 | ✓ |
| `roadmap/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md` | - | 2026-06-30 | 22 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-06-26 | 26 | ✓ |
| `roadmap/REPLAYSSM_PLAN.md` | - | 2026-07-14 | 8 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-07-11 | 11 | ✓ |
| `roadmap/SINGLE_GPU_CLOSEOUT_PLAN.md` | - | 2026-06-30 | 22 | ✓ |
| `roadmap/S_SERIES_ENABLEMENT_MAP.md` | - | 2026-06-27 | 25 | ✓ |
| `roadmap/S_SERIES_GAP_CLOSURE_PLAN.md` | - | 2026-07-14 | 8 | ✓ |
