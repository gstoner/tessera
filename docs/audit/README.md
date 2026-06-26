---
last_updated: 2026-06-26
audit_role: index
---

# Audit Documentation Index

This directory is organized around one root audit and canonical theme folders.
Generated dashboards remain in `generated/` and are owned by scripts/tests; do
not hand-edit them.

For the consolidated view of what is finished and what still needs work, start
with [MASTER_AUDIT.md](MASTER_AUDIT.md).

## Authority Rules

- Start with [MASTER_AUDIT.md](MASTER_AUDIT.md) for all-up status.
- Start with the theme audit doc for focused status.
- Treat generated dashboards as count/status truth.
- Treat `archive/` content as provenance and historical context, not as the
  current status surface.
- Root-level hand-written redirect stubs are not used. Old source docs live in
  theme-local archives and are summarized by the theme audit.
- Each theme folder's single entry point is its `*_AUDIT.md` — there are no
  per-theme `README.md` redirect stubs (they only restated "start with the
  audit" + the archive map carried in this file's Archive Map below). The
  Themes table above links straight to each audit.

## Authored-doc contract

Every **authored** audit doc (this folder + theme subtrees `compiler/`,
`backend/` and its `apple`/`nvidia`/`rocm` sub-audits, `coverage/`, `domain/`,
`roadmap/`) carries YAML frontmatter so staleness and lifecycle are
machine-visible instead of vibes. This is gated by
[`tests/unit/test_audit_docs.py`](../../tests/unit/test_audit_docs.py) and the
dates surface in [`generated/docs_freshness.md`](generated/docs_freshness.md)
(the audit tree is now one of the freshness manifest's roots).

```yaml
---
last_updated: YYYY-MM-DD     # honest last-touch / review date
audit_role: <role>           # taxonomy below
plan_state: open|landing|closed   # REQUIRED iff audit_role: plan
---
```

| `audit_role` | Meaning | Lifecycle rule |
|---|---|---|
| `root` | The single all-up entry point (`MASTER_AUDIT.md`). | Living. |
| `index` | A folder navigation `README.md`. | Living. |
| `theme` | A living theme audit (`compiler`/`backend`/`coverage`/`domain`/`roadmap`). | Living. |
| `sub_audit` | A per-target audit under `backend/` (apple/nvidia/rocm). | Living. |
| `plan` | A work plan; must declare `plan_state`. | When `closed`, move to a theme `archive/` and summarize in the theme audit. |
| `reference` | Supporting reference/survey material, not a status surface. | Living. |
| `snapshot` | A dated point-in-time audit. | Archive once superseded; stale snapshots are flagged after 45 days. |

**Claim-anchoring.** Every `root`/`theme`/`sub_audit` doc must link at least
one generated dashboard, so its status claims trace back to count authority
instead of restating numbers that silently drift (Decision #26). The gate also
fails any `generated/<x>.md` reference that no longer resolves — a dangling
claim anchor left behind when a dashboard is renamed.

The generated dashboards (`generated/`), theme-local `archive/`, and the
root drift-gated dashboards ([op_target_conformance.md](op_target_conformance.md),
[standalone_primitive_coverage.md](standalone_primitive_coverage.md),
`stub_surface.md`) are **not** authored prose — they own their own generators
and gates and are excluded from this contract.

## Root Files

| File | Purpose |
|---|---|---|
| [MASTER_AUDIT.md](MASTER_AUDIT.md) | Current all-up audit: finished work, open work, and priority queue. |
| [README.md](README.md) | Folder map and authority rules. |
| [op_target_conformance.md](op_target_conformance.md) | Drift-gated op-by-target conformance dashboard. |
| [standalone_primitive_coverage.md](standalone_primitive_coverage.md) | Drift-gated standalone primitive coverage dashboard. |
| [generated/](generated/) | Script/test-owned generated dashboards. |

## Themes

| Theme | Start here | Scope |
|---|---|---|
| Compiler | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) | Compiler architecture, IR handoffs, lowering, correctness, and spec gaps. |
| Shared backend | [backend/BACKEND_AUDIT.md](backend/BACKEND_AUDIT.md) | Cross-target runtime, ABI, proof rules, dtype policy, and hardware frontier. |
| Apple | [backend/apple/APPLE_AUDIT.md](backend/apple/APPLE_AUDIT.md) | Apple CPU/GPU, Metal 4, packaged kernels, command-buffer discipline, and Apple performance. |
| NVIDIA | [backend/nvidia/NVIDIA_AUDIT.md](backend/nvidia/NVIDIA_AUDIT.md) | CUDA/NVIDIA execution, target map, and execute-and-compare work. |
| ROCm | [backend/rocm/ROCM_AUDIT.md](backend/rocm/ROCM_AUDIT.md) | HIP/ROCm execution, target map, and MFMA proof work. |
| Coverage | [coverage/COVERAGE_AUDIT.md](coverage/COVERAGE_AUDIT.md) | Primitive, op, examples, KV-cache, and support coverage. |
| Domain roadmaps | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) | GA/EBM, attention variants, CorrDiff/SciML, sharding, and autodiff crosscuts. |
| Roadmap | [roadmap/ROADMAP_AUDIT.md](roadmap/ROADMAP_AUDIT.md) | Execution roadmap, deferred items, and sprint plans. |
| Generated dashboards | [generated/](generated/) | Script/test-owned dashboards. |

## Archive Map

Archives preserve source material that has been consolidated into the current
theme audits. Read them only when you need the original narrative or acceptance
criteria behind a current item.

| Former root doc | Current audit | Archived source |
|---|---|---|
| `compiler_apple_backend_end_to_end_audit_2026_06_02.md` | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md), [backend/apple/APPLE_AUDIT.md](backend/apple/APPLE_AUDIT.md) | `compiler/archive/` |
| `compiler_correctness_testing_audit.md` | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) | `compiler/archive/` |
| `compiler_improvement_milestone_plan_2026_05_18.md` | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) | `compiler/archive/` |
| `compiler_layer_gap_remediation.md` | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) | `compiler/archive/` |
| `compiler_spec_gap_audit.md` | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) | `compiler/archive/` |
| `compiler_spec_gap_matrix.md` | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) | `compiler/archive/` |
| `CODE_AUDIT_2026_06_10.md` (snapshot) | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) | `compiler/archive/` |
| `DEEP_COMPILER_AUDIT_2026_06_10.md` (snapshot) | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) | `compiler/archive/` |
| `2026_06_01_apple_gpu_chain_audit.md` | [backend/apple/APPLE_AUDIT.md](backend/apple/APPLE_AUDIT.md) | `backend/apple/archive/` |
| `apple_ga_ebm_native_execution_gap.md` | [backend/apple/APPLE_AUDIT.md](backend/apple/APPLE_AUDIT.md) | `backend/apple/archive/` |
| `single_command_buffer_decode_plan.md` | [backend/apple/APPLE_AUDIT.md](backend/apple/APPLE_AUDIT.md) | `backend/apple/archive/` |
| `nvidia_execution_audit.md` | [backend/nvidia/NVIDIA_AUDIT.md](backend/nvidia/NVIDIA_AUDIT.md) | `backend/nvidia/archive/` |
| `nvidia_rocm_execute_and_compare_plan.md` | [backend/BACKEND_AUDIT.md](backend/BACKEND_AUDIT.md), [backend/nvidia/NVIDIA_AUDIT.md](backend/nvidia/NVIDIA_AUDIT.md), [backend/rocm/ROCM_AUDIT.md](backend/rocm/ROCM_AUDIT.md) | `backend/archive/` |
| `phase_ghi_hardware_frontier.md` | [backend/BACKEND_AUDIT.md](backend/BACKEND_AUDIT.md) | `backend/archive/` |
| `hardware_dtype_support_matrix.md` | [backend/BACKEND_AUDIT.md](backend/BACKEND_AUDIT.md) | `backend/archive/` |
| `advanced_examples_capability_gap.md` | [coverage/COVERAGE_AUDIT.md](coverage/COVERAGE_AUDIT.md) | `coverage/archive/` |
| `kv_cache_coverage_matrix.md` | [coverage/COVERAGE_AUDIT.md](coverage/COVERAGE_AUDIT.md) | `coverage/archive/` |
| `partial_ops_uplift_plan.md` | [coverage/COVERAGE_AUDIT.md](coverage/COVERAGE_AUDIT.md) | `coverage/archive/` |
| `primitive_coverage_state.md` | [coverage/COVERAGE_AUDIT.md](coverage/COVERAGE_AUDIT.md) | `coverage/archive/` |
| `attention_variants_plan.md` | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) | `domain/archive/` |
| `corrdiff_compiler_split_evaluation.md` | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) | `domain/archive/` |
| `ebm_scope_lock.md` | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) | `domain/archive/` |
| `ga6_autodiff_plan.md` | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) | `domain/archive/` |
| `ga_ebm_roadmap.md` | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) | `domain/archive/` |
| `ga_scope_lock.md` | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) | `domain/archive/` |
| `sharding_partial_audit.md` | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) | `domain/archive/` |
| `source_base_review_2026_05_17.md` | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) | `domain/archive/` |
| `deferred_items_plan.md` | [roadmap/ROADMAP_AUDIT.md](roadmap/ROADMAP_AUDIT.md) | `roadmap/archive/` |
| `execution_roadmap.md` | [roadmap/ROADMAP_AUDIT.md](roadmap/ROADMAP_AUDIT.md) | `roadmap/archive/` |
| `sprint_plan_task4_and_crosscuts.md` | [roadmap/ROADMAP_AUDIT.md](roadmap/ROADMAP_AUDIT.md) | `roadmap/archive/` |
