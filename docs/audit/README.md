---
last_updated: 2026-07-11
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
- Start with [generated/compiler_progress.md](generated/compiler_progress.md)
  for the generated compiler-progress dashboard and open-work summary.
- Start with the theme audit doc for focused status.
- Treat generated dashboards as count/status truth.
- Treat `archive/` content as provenance and historical context, not as the
  current status surface.
- Reader-facing backend documentation lives in [`docs/backends/`](../backends/).
  Audit subtrees own evidence, decisions, and history rather than the primary
  architecture walkthrough.
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
| [generated/compiler_progress.md](generated/compiler_progress.md) | Generated all-up compiler-progress dashboard: phase/IR state, primitives, integration, backend/codegen pathways, and open work. |
| [generated/](generated/) | Script/test-owned generated dashboards. |

## Forward Plans — the compiler north star

The go-forward direction for compiler development (the **new north star**) is a
paired plan + theory set under `compiler/`. Read these before starting backend or
middle-end work; they supersede the pre-2026-07 "op-library" framing.

| Doc | Role |
|---|---|
| [compiler/COMPILER_THEORY_OF_OPERATION.md](compiler/COMPILER_THEORY_OF_OPERATION.md) | **Read first.** Durable conceptual model — three-tier kernel model (generic framework / per-arch plugin / hand-tuned library), the accuracy-budgeted measured arbiter, the three-system fleet, and the W1–W8 world-class scope register. |
| [compiler/COMPILER_REFACTOR_PLAN.md](compiler/COMPILER_REFACTOR_PLAN.md) | Execution plan — Workstreams A–E (kernel spine) + F–K (world-class dimensions), sequencing, and the Mac/Strix-Halo/NR2-Pro coordination + routing matrix. |
| [compiler/OPTIMIZING_COMPILER_PLAN.md](compiler/OPTIMIZING_COMPILER_PLAN.md) | Middle-end synthesis (F0–F5 landed on Apple); **F6 = the backend-build seam** (reassessed 2026-07-02). |
| [compiler/EVALUATOR_PLAN.md](compiler/EVALUATOR_PLAN.md) | The scoring engine that gates every promotion in the plans above. |
| [compiler/STAGE_A_EMIT_PLAN.md](compiler/STAGE_A_EMIT_PLAN.md) | Cross-vendor emit-ladder grounding. |
| [compiler/AUTODIFF_UNIFICATION_PLAN.md](compiler/AUTODIFF_UNIFICATION_PLAN.md) | Front-end / IR / autodiff unification — make differentiation a compiler request with a native fwd+bwd execution path and a per-family × per-target proof ledger, replacing implicit-tape-reported-as-compiled. |

Governing rule across all of them: **ROCm/CUDA are the lead performance targets;
the generic framework raises the floor and must never cap their ceiling** (Theory
§1). Status of what has landed vs. open still flows through
[MASTER_AUDIT.md](MASTER_AUDIT.md) and the generated dashboards — the plans are
*direction*, not status truth.

## Themes

| Theme | Start here | Scope |
|---|---|---|
| Compiler | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) | Compiler architecture, IR handoffs, lowering, correctness, and spec gaps. |
| Shared backend | [backend/BACKEND_AUDIT.md](backend/BACKEND_AUDIT.md) | Cross-target runtime, ABI, proof rules, dtype policy, and hardware frontier. |
| Apple | [backend/apple/APPLE_AUDIT.md](backend/apple/APPLE_AUDIT.md) | Evidence and decisions; reader guide: [docs/backends/apple/](../backends/apple/). |
| NVIDIA | [backend/nvidia/NVIDIA_AUDIT.md](backend/nvidia/NVIDIA_AUDIT.md) | Evidence and decisions; reader guide: [docs/backends/nvidia/](../backends/nvidia/). |
| ROCm | [backend/rocm/ROCM_AUDIT.md](backend/rocm/ROCM_AUDIT.md) | Evidence and decisions; reader guide: [docs/backends/rocm/](../backends/rocm/). |
| x86 | [backend/BACKEND_AUDIT.md](backend/BACKEND_AUDIT.md) | Shared evidence; reader guide: [docs/backends/x86/](../backends/x86/). |
| Coverage | [coverage/COVERAGE_AUDIT.md](coverage/COVERAGE_AUDIT.md) | Primitive, op, examples, KV-cache, and support coverage. |
| Domain roadmaps | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) | GA/EBM, attention variants, CorrDiff/SciML, sharding, and autodiff crosscuts. |
| Roadmap | [roadmap/ROADMAP_AUDIT.md](roadmap/ROADMAP_AUDIT.md) | Execution roadmap, deferred items, and sprint plans. |
| Generated compiler progress | [generated/compiler_progress.md](generated/compiler_progress.md) | All-up compiler-progress rollup and open-work summary. |
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
