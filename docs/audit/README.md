# Audit Documentation Index

This directory is organized around canonical audit themes. Generated dashboards
remain in `generated/` and are owned by scripts/tests; do not hand-edit them.

## Authority Rules

- Start with the canonical README for a theme.
- Treat generated dashboards as count/status truth.
- Treat `archive/` content as provenance and historical context.
- Root-level redirect files exist only to preserve old links.

## Themes

| Theme | Canonical doc | Scope |
|---|---|---|
| Compiler | [compiler/README.md](compiler/README.md) | Compiler architecture, IR handoffs, lowering, correctness, and spec gaps. |
| Shared backend | [backend/README.md](backend/README.md) | Cross-target runtime, ABI, proof rules, dtype policy, and hardware frontier. |
| Apple | [backend/apple/README.md](backend/apple/README.md) | Apple CPU/GPU, Metal 4, packaged kernels, command-buffer discipline, and Apple performance. |
| NVIDIA | [backend/nvidia/README.md](backend/nvidia/README.md) | CUDA/NVIDIA execution, target map, and execute-and-compare work. |
| ROCm | [backend/rocm/README.md](backend/rocm/README.md) | HIP/ROCm execution, target map, and MFMA proof work. |
| Coverage | [coverage/README.md](coverage/README.md) | Primitive, op, examples, KV-cache, and support coverage. |
| Domain roadmaps | [domain/README.md](domain/README.md) | GA/EBM, attention variants, CorrDiff/SciML, sharding, and autodiff crosscuts. |
| Roadmap | [roadmap/README.md](roadmap/README.md) | Execution roadmap, deferred items, and sprint plans. |
| Generated dashboards | [generated/](generated/) | Script/test-owned dashboards. |

## Classification Table

| Former root doc | Theme | Disposition |
|---|---|---|
| `compiler_apple_backend_end_to_end_audit_2026_06_02.md` | Compiler + Apple | Archived under `compiler/archive/`; current tracking split across compiler and Apple canonical docs. |
| `compiler_correctness_testing_audit.md` | Compiler | Archived under `compiler/archive/`. |
| `compiler_improvement_milestone_plan_2026_05_18.md` | Compiler | Archived under `compiler/archive/`. |
| `compiler_layer_gap_remediation.md` | Compiler | Archived under `compiler/archive/`; root redirect preserved. |
| `compiler_spec_gap_audit.md` | Compiler | Archived under `compiler/archive/`. |
| `compiler_spec_gap_matrix.md` | Compiler | Archived under `compiler/archive/`. |
| `2026_06_01_apple_gpu_chain_audit.md` | Apple | Archived under `backend/apple/archive/`. |
| `apple_ga_ebm_native_execution_gap.md` | Apple | Archived under `backend/apple/archive/`. |
| `single_command_buffer_decode_plan.md` | Apple | Archived under `backend/apple/archive/`; root redirect preserved. |
| `nvidia_execution_audit.md` | NVIDIA | Archived under `backend/nvidia/archive/`. |
| `nvidia_rocm_execute_and_compare_plan.md` | Shared backend / NVIDIA / ROCm | Archived under `backend/archive/`; linked from NVIDIA and ROCm canonical docs. |
| `phase_ghi_hardware_frontier.md` | Shared backend | Archived under `backend/archive/`; root redirect preserved. |
| `hardware_dtype_support_matrix.md` | Shared backend | Archived under `backend/archive/`. |
| `advanced_examples_capability_gap.md` | Coverage | Archived under `coverage/archive/`. |
| `kv_cache_coverage_matrix.md` | Coverage | Archived under `coverage/archive/`. |
| `partial_ops_uplift_plan.md` | Coverage | Archived under `coverage/archive/`. |
| `primitive_coverage_state.md` | Coverage | Archived under `coverage/archive/`; historical narrative snapshot. |
| `standalone_primitive_coverage.md` | Coverage | Kept at root because it is drift-gated by tests. |
| `op_target_conformance.md` | Generated-style conformance | Kept at root because the CLI/test drift gate owns this path. |
| `attention_variants_plan.md` | Domain | Archived under `domain/archive/`. |
| `corrdiff_compiler_split_evaluation.md` | Domain | Archived under `domain/archive/`. |
| `ebm_scope_lock.md` | Domain | Archived under `domain/archive/`. |
| `ga6_autodiff_plan.md` | Domain | Archived under `domain/archive/`. |
| `ga_ebm_roadmap.md` | Domain | Archived under `domain/archive/`; root redirect preserved. |
| `ga_scope_lock.md` | Domain | Archived under `domain/archive/`. |
| `sharding_partial_audit.md` | Domain | Archived under `domain/archive/`. |
| `source_base_review_2026_05_17.md` | Domain | Archived under `domain/archive/`. |
| `deferred_items_plan.md` | Roadmap | Archived under `roadmap/archive/`. |
| `execution_roadmap.md` | Roadmap | Archived under `roadmap/archive/`; root redirect preserved. |
| `sprint_plan_task4_and_crosscuts.md` | Roadmap | Archived under `roadmap/archive/`. |
