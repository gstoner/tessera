---
last_updated: 2026-08-11
audit_role: root
---

# Tessera Audit Master

This is the root routing document for compiler status and open engineering
work. It deliberately owns no copied totals. Generated dashboards own counts
and row-level states; plans own sequencing; backend audits own exact-device
claims.

## Start here

1. Read [`generated/compiler_progress.md`](generated/compiler_progress.md) for
   the live phase rollup.
2. Use [`generated/support_table.md`](generated/support_table.md) to locate the
   first incomplete compiler layer for an operation.
3. Use [`generated/s_series_status.md`](generated/s_series_status.md) for
   primitive transform, sharding, and backend-contract state.
4. Use [`generated/runtime_execution_matrix.md`](generated/runtime_execution_matrix.md)
   and the applicable backend plan before claiming native execution.
5. Use [`roadmap/ROADMAP_AUDIT.md`](roadmap/ROADMAP_AUDIT.md) for active
   ownership, ordering, and archived-plan provenance.

The curated matrix in [`op_target_conformance.md`](op_target_conformance.md) is
an exact-target representative suite. It is not the all-up compiler denominator.

## Current interpretation

The public API, frontend-capture inventory, Graph registration, Schedule IR,
Tile IR, runtime-readiness, verifier, batching, transpose, and lowering axes are
closed in the generated rollup. Those surfaces remain regression gates.

The compiler is not finished. Its active work is concentrated in the following
programs.

### 1. E2E-REAL-6 — one compiler authority

Promote the tracer to the sole general frontend, move family selection and
package construction out of `JitFn`, and require native packages to consume
their exact content-addressed Schedule→Tile parent. Delete the AST
`_OpExtractor` and Graph-to-backend reconstruction only after differential
execution and architecture-owned proof cover each migrated family.

Owner: [`compiler/INTEGRATED_COMPILER_PLAN.md`](compiler/INTEGRATED_COMPILER_PLAN.md).

### 2. General structured programs

Bounded `if`, counted `for`, canonical bounded `while`, and forward
`control_scan` exist. Remaining work is general source-CFG recovery, multi-block
regions, typed affine/Presburger constraints, scan JVP/VJP, and lowering a
selected SAVE/RECOMPUTE/HYBRID checkpoint plan into the generated region
product.

Owners: [`compiler/INTEGRATED_COMPILER_PLAN.md`](compiler/INTEGRATED_COMPILER_PLAN.md)
and [`../spec/CONTROL_FLOW_CONTRACT.md`](../spec/CONTROL_FLOW_CONTRACT.md).

### 3. Measured scheduling

The shared dataflow analysis and prune-only action-DAG ranker are real. The next
boundary is automatic dependence-edge generation from value, alias, effect,
memory-dependence, and ordered-collective facts. Ranked candidates then need
clean target calibration and selector-grade packets before a schedule can be
promoted. Analytical or WSL-only timing remains candidate-pruning evidence.

Owner: W2.1 and W5.2 in
[`compiler/INTEGRATED_COMPILER_PLAN.md`](compiler/INTEGRATED_COMPILER_PLAN.md).

### 4. Native distributed execution and sharding

The core collective Schedule/Tile contracts exist. Native NCCL/RCCL and
MPI/OFI/SHMEM launchers, subgroup propagation, reshard insertion, and real
multi-rank correctness/performance packets remain open. Sharding propagation
must use a typed, fail-closed placement lattice and explicit incompatibilities;
it must not infer safety across unknown effects, aliases, or regions.

Owners: W5.4 in the integrated plan, the generated sharding queue, and the four
backend plans.

### 5. Tiled SSD

ReplaySSM established a correct resident decode-state and replay ABI, but it is
not the shared SSD compiler family. The remaining project is a first-class
Schedule→Tile SSD program whose chunked GEMM/reduction/recurrent actions,
residency, checkpoint state, and mutation lineage are target-independent.
Backend WMMA/MFMA/WGMMA/AVX-512 selection and performance evidence remain
architecture-owned. Existing ReplaySSM kernels are candidates and oracles, not
the semantic compiler authority.

Owner: the tiled-SSD section of
[`roadmap/ROADMAP_AUDIT.md`](roadmap/ROADMAP_AUDIT.md), graduating into the
integrated compiler plan when implementation begins.

### 6. Model-level physical closure

The frontier-model graph vocabulary and scaled reference execution exist. The
active physical boundary is packed INT4/FP8 weight ingestion without full-weight
materialization, architecture-owned DeepSeek and MiniMax fused paths, and
full-scale distributed execution with real routing and transport. Model
configuration, artifact compilation, scaled correctness, fused execution, and
full-scale performance are distinct proof rungs.

Owner: [`roadmap/MODEL_CLASS_ROADMAP.md`](roadmap/MODEL_CLASS_ROADMAP.md) plus
the applicable backend plan.

### 7. Architecture promotion

Exact-device evidence never transfers between architectures. x86/AVX-512,
Apple, gfx1151, gfx1200/gfx1250, and individual NVIDIA SM generations retain
separate correctness and performance gates. A fused or packaged implementation
is not selector authority without valid target timing provenance.

Owners:
[`backend/apple/todo.md`](backend/apple/todo.md),
[`backend/nvidia/todo.md`](backend/nvidia/todo.md),
[`backend/rocm/todo.md`](backend/rocm/todo.md), and
[`backend/x86/todo.md`](backend/x86/todo.md).

## Proof vocabulary

| Term | Meaning |
|---|---|
| `complete` | Every required rung in the stated scope is proven. |
| `reference` | Correct execution exists without a native target implementation. |
| `device_verified_jit` | A compiler-generated binary launched and matched its oracle on the exact target. |
| `device_verified_abi` | A shipped stable ABI launched and matched its oracle on the exact target. |
| `fused` / `packaged` | An owned implementation exists; execution or promotion proof may still be absent. |
| `artifact_only` | Compilation evidence exists without link/launch proof. |
| `partial` / `planned` | An explicit contract or evidence obligation remains. |
| explicit terminal status | The axis is closed by design with a specific reason. |

## Dashboard map

| Question | Authority |
|---|---|
| What phase is open? | [`generated/compiler_progress.md`](generated/compiler_progress.md) |
| Which operation is affected? | [`generated/support_table.md`](generated/support_table.md) |
| Which primitive contracts remain? | [`generated/s_series_status.md`](generated/s_series_status.md) |
| Which target paths launch? | [`generated/runtime_execution_matrix.md`](generated/runtime_execution_matrix.md) |
| Which ABI symbols are real? | [`generated/runtime_abi.md`](generated/runtime_abi.md) |
| Which tests are direct or structural? | [`generated/test_coverage.md`](generated/test_coverage.md) |
| Which verifiers are registered? | [`generated/verifier_coverage.md`](generated/verifier_coverage.md) |
| Which target rows are exact, packaged, or reference? | Generated target maps plus the backend plan |
| What work is software-actionable? | [`generated/single_gpu_closeout.md`](generated/single_gpu_closeout.md) and [`stub_surface.md`](stub_surface.md) |

## Lifecycle rule

Generated dashboards own status. The integrated compiler plan and active
backend/model plans own work. Completed sprint plans and point-in-time enablement
maps live under `archive/` and may be cited only for provenance. A historical
plan must not remain an active owner merely because source comments still use
its old work-item labels.
