# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-07-10**.

## Headline

- **98** docs catalogued across the canonical doc tree.
- **97** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **56** updated within the last 30 days.
- **0** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `AUTODIFF_SPEC.md` | - | 2026-05-09 | 62 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-05-01 | 70 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 54 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 15 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 29 | ✓ |
| `CONTROL_FLOW_CONTRACT.md` | - | 2026-06-30 | 10 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 55 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-05-17 | 54 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-05-22 | 49 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 65 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-04-26 | 75 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-06-11 | 29 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 49 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 35 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-06-16 | 24 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-06-20 | 20 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 49 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-06-11 | 29 | ✓ |
| `TILE_IR.md` | Normative | 2026-05-22 | 49 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-05-18 | 53 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 36 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 65 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 65 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 73 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 73 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 73 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 29 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 73 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-06-21 | 19 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-06-20 | 20 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 73 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-06-20 | 20 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-04-28 | 73 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 29 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 29 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-05-22 | 49 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-04-26 | 75 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-04-26 | 75 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-05-06 | 65 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-06-11 | 29 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-05-06 | 65 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-04-30 | 71 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-04-26 | 75 | ✓ |
| `README.md` | Informative | 2026-05-20 | 51 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 65 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | Deferred | 2026-05-20 | 51 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 51 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 75 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-06-07 | 33 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 51 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 29 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 71 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-06-11 | 29 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 51 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 51 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 60 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-07-10 | 0 | ✓ |
| `README.md` | - | 2026-07-10 | 0 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-06-26 | 14 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-07-10 | 0 | ✓ |
| `backend/apple/APPLE_GPU_CODEGEN_PLAN.md` | - | 2026-06-15 | 25 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-06-04 | 36 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-06-24 | 16 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-07-07 | 3 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 16 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-07-10 | 0 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-06-18 | 22 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-06-23 | 17 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-07-10 | 0 | ✓ |
| `compiler/COMPILER_REFACTOR_PLAN.md` | - | 2026-07-11 | -1 | ✓ |
| `compiler/COMPILER_THEORY_OF_OPERATION.md` | - | 2026-07-02 | 8 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-06-12 | 28 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-07-02 | 8 | ✓ |
| `compiler/STAGE_A_EMIT_PLAN.md` | - | 2026-06-18 | 22 | ✓ |
| `compiler/WORKSTREAM_C_HANDOFF.md` | - | 2026-07-06 | 4 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-06-21 | 19 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-06-11 | 29 | ✓ |
| `roadmap/CF_CROSS_ELEMENT_PLAN.md` | - | 2026-06-30 | 10 | ✓ |
| `roadmap/CONTRACT_PASS_PLAN.md` | - | 2026-06-20 | 20 | ✓ |
| `roadmap/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md` | - | 2026-06-30 | 10 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-06-26 | 14 | ✓ |
| `roadmap/REPLAYSSM_PLAN.md` | - | 2026-06-15 | 25 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-06-26 | 14 | ✓ |
| `roadmap/SINGLE_GPU_CLOSEOUT_PLAN.md` | - | 2026-06-30 | 10 | ✓ |
| `roadmap/S_SERIES_ENABLEMENT_MAP.md` | - | 2026-06-27 | 13 | ✓ |
| `roadmap/S_SERIES_GAP_CLOSURE_PLAN.md` | - | 2026-06-28 | 12 | ✓ |
