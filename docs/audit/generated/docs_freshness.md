# Documentation Freshness Dashboard

Generated from `python/tessera/compiler/docs_manifest.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.docs_manifest import render_dashboard; open('docs/audit/generated/docs_freshness.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_docs_freshness.py`.

Reference date for staleness: **2026-06-29**.

## Headline

- **92** docs catalogued across the canonical doc tree.
- **91** carry a `last_updated:` marker; **1** are undated (invisible to the freshness audit until tagged).
- **54** updated within the last 30 days.
- **0** older than 90 days; **0** older than 180 days.

## Undated docs (no parseable `last_updated`)

These docs need either YAML frontmatter (`last_updated: YYYY-MM-DD`) or a body-form `Last updated:` line to participate in the audit.  Until tagged, the freshness signal is unavailable.

- `docs/reference/tessera_frontend_lanes.md`

## Per-root inventory

### `docs/spec/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `AUTODIFF_SPEC.md` | - | 2026-05-09 | 51 | ✓ |
| `CITL_ROCM_TRACE_PROFILER_SPEC.md` | Draft | 2026-05-01 | 59 | ✓ |
| `CLIFFORD_SPEC.md` | - | 2026-05-17 | 43 | ✓ |
| `COMPILER_REFERENCE.md` | Normative | 2026-06-25 | 4 | ✓ |
| `CONFORMANCE.md` | Normative | 2026-06-11 | 18 | ✓ |
| `EBM_SPEC.md` | - | 2026-05-16 | 44 | ✓ |
| `GA_EBM_EXECUTION_STATUS.md` | - | 2026-05-17 | 43 | ✓ |
| `GRAPH_IR_SPEC.md` | Normative | 2026-05-22 | 38 | ✓ |
| `LANGUAGE_AND_IR_SPEC.md` | Normative | 2026-05-06 | 54 | ✓ |
| `LANGUAGE_SPEC.md` | Normative | 2026-04-26 | 64 | ✓ |
| `LOWERING_PIPELINE_SPEC.md` | Normative | 2026-06-11 | 18 | ✓ |
| `MEMORY_MODEL_SPEC.md` | Normative | 2026-05-22 | 38 | ✓ |
| `PRODUCTION_COMPILER_PLAN.md` | Ratified | 2026-06-05 | 24 | ✓ |
| `PYTHON_API_SPEC.md` | Normative | 2026-06-16 | 13 | ✓ |
| `RUNTIME_ABI_SPEC.md` | Normative | 2026-06-20 | 9 | ✓ |
| `SHAPE_SYSTEM.md` | Normative | 2026-05-22 | 38 | ✓ |
| `TARGET_IR_SPEC.md` | Normative | 2026-06-11 | 18 | ✓ |
| `TILE_IR.md` | Normative | 2026-05-22 | 38 | ✓ |
| `VALIDATION_SPINE.md` | Normative | 2026-05-18 | 42 | ✓ |
| `VALUE_TARGET_IR_CONTRACT.md` | Normative | 2026-06-04 | 25 | ✓ |

### `docs/guides/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Debugging_Tools_Guide.md` | Informative | 2026-05-06 | 54 | ✓ |
| `Tessera_Developer_Frontend_End_To_End.md` | Informative | 2026-05-06 | 54 | ✓ |
| `Tessera_Differentiable_NAS_Guide.md` | Draft | 2026-04-28 | 62 | ✓ |
| `Tessera_Error_Handling_And_Diagnostics_Guide.md` | Normative | 2026-04-28 | 62 | ✓ |
| `Tessera_Fault_Tolerance_And_Elasticity_Guide.md` | Informative | 2026-04-28 | 62 | ✓ |
| `Tessera_Inference_Server_Guide.md` | Informative | 2026-06-11 | 18 | ✓ |
| `Tessera_Production_Reliability_And_Chaos_Guide.md` | Informative | 2026-04-28 | 62 | ✓ |
| `Tessera_Profiler_Release_Gates.md` | Informative | 2026-06-21 | 8 | ✓ |
| `Tessera_Profiling_And_Autotuning_Guide.md` | Informative | 2026-06-20 | 9 | ✓ |
| `Tessera_QA_Reliability_Guide.md` | Informative | 2026-04-28 | 62 | ✓ |
| `Tessera_Runtime_ABI_Guide.md` | Tutorial | 2026-06-20 | 9 | ✓ |
| `Tessera_Tensor_Layout_And_Data_Movement_Guide.md` | Normative | 2026-04-28 | 62 | ✓ |

### `docs/programming_guide/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Goals.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Appendix_NVL72.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter10_Portability.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter11_Conclusion.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter1_Introduction_Overview.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter2_Programming_Model.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter3_Memory_Model.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter4_Execution_Model.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter5_Kernel_Programming.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter6_Numerics_Model.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter7_Autodiff.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md` | Tutorial | 2026-06-11 | 18 | ✓ |
| `Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md` | Tutorial | 2026-06-11 | 18 | ✓ |

### `docs/operations/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Tessera_Standard_Operations.md` | Normative | 2026-05-22 | 38 | ✓ |

### `docs/architecture/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `Compiler/Tessera_Compiler_Architecture_Overview.md` | Informative | 2026-04-26 | 64 | ✓ |
| `Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md` | Informative | 2026-04-26 | 64 | ✓ |
| `Compiler/Tessera_Compiler_ScheduleIR_Design.md` | Informative | 2026-05-06 | 54 | ✓ |
| `Compiler/Tessera_Compiler_TargetIR_Design.md` | Informative | 2026-06-11 | 18 | ✓ |
| `Compiler/Tessera_Compiler_TileIR_Design.md` | Informative | 2026-05-06 | 54 | ✓ |
| `Compiler/tessera_ir_layers.md` | Informative | 2026-04-30 | 60 | ✓ |
| `Compiler/tessera_tile_ir_documentation.md` | Informative | 2026-04-26 | 64 | ✓ |
| `README.md` | Informative | 2026-05-20 | 40 | ✓ |
| `Tessera_Kernel_Compilation_Stages_Overview.md` | Informative | 2026-05-06 | 54 | ✓ |
| `compiler_gaps_1_3_5_plan.md` | Deferred | 2026-05-20 | 40 | ✓ |
| `frontend_substrate_plan.md` | Active | 2026-05-20 | 40 | ✓ |
| `proposals/cute_tessera_enhancement.md` | Proposal | 2026-04-26 | 64 | ✓ |
| `proposals/tiled_ssd_tile_ir_schedule.md` | - | 2026-06-07 | 22 | ✓ |
| `stencil_materialize_and_window_lowering.md` | Informative | 2026-05-20 | 40 | ✓ |
| `system_overview.md` | Informative | 2026-06-11 | 18 | ✓ |
| `tessera_target_ir_usage_guide.md` | Informative | 2026-04-30 | 60 | ✓ |

### `docs/reference/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `tessera-api-reference.md` | Informative | 2026-06-11 | 18 | ✓ |
| `tessera_frontend_lanes.md` | - | _undated_ | - | _body_ |
| `tessera_migration_guide_part1.md` | Pre-canonical | 2026-05-20 | 40 | ✓ |
| `tessera_migration_guide_part2.md` | Informative | 2026-05-20 | 40 | ✓ |
| `tessera_tensor_attributes.md` | Normative | 2026-05-11 | 49 | ✓ |

### `docs/audit/`

| Path | status | last_updated | days stale | frontmatter |
|------|--------|--------------|-----------:|--|
| `MASTER_AUDIT.md` | - | 2026-06-25 | 4 | ✓ |
| `README.md` | - | 2026-06-26 | 3 | ✓ |
| `backend/BACKEND_AUDIT.md` | - | 2026-06-26 | 3 | ✓ |
| `backend/apple/APPLE_AUDIT.md` | - | 2026-06-17 | 12 | ✓ |
| `backend/apple/APPLE_GPU_CODEGEN_PLAN.md` | - | 2026-06-15 | 14 | ✓ |
| `backend/apple/MPSGRAPH_RUNTIME_GLASS_JAWS.md` | - | 2026-06-04 | 25 | ✓ |
| `backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md` | - | 2026-06-24 | 5 | ✓ |
| `backend/nvidia/NVIDIA_AUDIT.md` | - | 2026-06-25 | 4 | ✓ |
| `backend/nvidia/spikes/sm120_mma_sync/README.md` | - | 2026-06-24 | 5 | ✓ |
| `backend/rocm/ROCM_AUDIT.md` | - | 2026-06-25 | 4 | ✓ |
| `backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` | - | 2026-06-18 | 11 | ✓ |
| `backend/rocm/STRIX_HALO_EXECUTION_PLAN.md` | - | 2026-06-23 | 6 | ✓ |
| `compiler/COMPILER_AUDIT.md` | - | 2026-06-24 | 5 | ✓ |
| `compiler/EVALUATOR_PLAN.md` | - | 2026-06-12 | 17 | ✓ |
| `compiler/OPTIMIZING_COMPILER_PLAN.md` | - | 2026-06-15 | 14 | ✓ |
| `compiler/STAGE_A_EMIT_PLAN.md` | - | 2026-06-18 | 11 | ✓ |
| `coverage/COVERAGE_AUDIT.md` | - | 2026-06-21 | 8 | ✓ |
| `domain/DOMAIN_AUDIT.md` | - | 2026-06-11 | 18 | ✓ |
| `roadmap/CONTRACT_PASS_PLAN.md` | - | 2026-06-20 | 9 | ✓ |
| `roadmap/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md` | - | 2026-06-28 | 1 | ✓ |
| `roadmap/MODEL_CLASS_ROADMAP.md` | - | 2026-06-26 | 3 | ✓ |
| `roadmap/REPLAYSSM_PLAN.md` | - | 2026-06-15 | 14 | ✓ |
| `roadmap/ROADMAP_AUDIT.md` | - | 2026-06-26 | 3 | ✓ |
| `roadmap/S_SERIES_ENABLEMENT_MAP.md` | - | 2026-06-27 | 2 | ✓ |
| `roadmap/S_SERIES_GAP_CLOSURE_PLAN.md` | - | 2026-06-28 | 1 | ✓ |
