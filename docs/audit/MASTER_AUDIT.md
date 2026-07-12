---
last_updated: 2026-07-12
audit_role: root
---

# Tessera Audit Master

This document is the current, reader-facing audit summary. Generated dashboards
own counts and row-level status; this page explains what those results mean,
separates closed work from open work, and routes each action to its evidence.

## Technical Summary

- **The curated matrix now uses exact target grain and real IR verification.**
  Its 63 cells separate portable CPU reference, native x86, Apple CPU/GPU,
  ROCm, and four NVIDIA architectures. Current results are 17 complete,
  11 reference, and 35 missing.
- **The redesign exposed real gaps hidden by the former proxies.** Stateful
  `kv_cache_read` emits Graph IR but no Schedule IR; several native x86 rows lack
  exact-target fixtures; only `nvidia_sm120/matmul` completes the NVIDIA ladder.
- **Curated conformance closure is not whole-compiler closure.** The broader
  primitive registry still has backend-kernel promotion work, 43 open sharding
  contracts, a small Tile/Target IR tail, verifier triage, and structural-only
  test evidence.
- **Most compiler foundations are closed.** Public API capture, Graph IR,
  Schedule IR, runtime dispatch readiness, batching, transpose, and lowering
  contracts are closed in the generated rollups.
- **The next software priority is NVIDIA execution breadth.** Hardware-specific
  expansion for Hopper, datacenter Blackwell, and CDNA remains a separate proof
  program and must not be represented as ordinary software stubs.

Evidence snapshot: 2026-07-12, sourced from
[`op_target_conformance.csv`](op_target_conformance.csv),
[`generated/compiler_progress.md`](generated/compiler_progress.md), and
[`generated/s_series_status.md`](generated/s_series_status.md).

## Status Definitions

### Curated op×target conformance

The conformance ladder is:

`graph_emitted` → `schedule_legal` → `tile_legal` → `target_legal` →
`backend_compile` → `runtime_execute` → `numerical_check`

A cell is `complete` only when every column is complete. Its
`first_failing_gate` must then be empty. For an open cell,
`first_failing_gate` names the earliest blocking capability; it is not a record
of the machine that regenerated the dashboard.

The matrix's `cpu` target is the host x86/CPU reference conformance path. The
separate x86 native-kernel inventory is tracked in the execution matrix and
primitive target evidence.

### Broader compiler completion

The compiler-progress rollup measures a larger surface than the curated matrix:
315 compiler ops, 480 primitive-contract rows, runtime/ABI integration,
benchmark evidence, and repository proof surfaces. A green curated conformance
matrix therefore does not close primitive-wide `backend_kernel`, sharding,
benchmark, or ABI work.

### Proof vocabulary

| Term | Meaning |
|---|---|
| `complete` | The required proof exists for every rung in the measured scope. |
| `reference` | Correct execution exists without a native target kernel. |
| `compiled` / `fused` / `hardware_verified` | Increasingly concrete native target evidence; exact semantics are dashboard-defined. |
| `artifact_only` | Target text or an artifact emits, but link/launch proof is absent. |
| `partial` | A path exists with an explicit unresolved qualification. |
| `missing` | Required evidence or target support is absent. |
| explicit terminal status | The axis is closed by design, with a reason such as `no_kernel_required`; generic N/A is not used as an action bucket. |

## Current Truth Snapshot

| Area | Current result | Remaining frontier | Authority |
|---|---|---|---|
| Curated conformance | 63 exact-target cells: 17 complete, 11 reference, 35 missing | Stateful Schedule IR, x86 fixture gaps, and NVIDIA architecture breadth | [`op_target_conformance.md`](op_target_conformance.md) |
| Compiler phases | API, frontend, Graph IR, Schedule IR, and runtime readiness closed | 13 Tile IR rows and 14 Target IR rows remain mixed | [`generated/compiler_progress.md`](generated/compiler_progress.md) |
| Primitive contracts | Batching, transpose, and lowering closed across 480 primitives | Sharding has 43 open rows; registry-level backend promotion remains broad | [`generated/s_series_status.md`](generated/s_series_status.md) |
| Runtime execution | Checked-in executable rows are explicit and drift-gated | Add rows only after a real launch path exists | [`generated/runtime_execution_matrix.md`](generated/runtime_execution_matrix.md) |
| Verifiers | No trivial verifier stubs | Manually triage 11 no-verifier ops; add verifiers only where invariants exist | [`generated/verifier_coverage.md`](generated/verifier_coverage.md) |
| Test evidence | No `needs_direct_test` debt | Convert high-value structural-only evidence to direct or differential proof | [`generated/test_coverage.md`](generated/test_coverage.md) |
| Apple | Curated CPU/GPU conformance closed; Apple GPU proof is provenance-gated | Performance, precision, and the small target-map tail | [`backend/apple/APPLE_AUDIT.md`](backend/apple/APPLE_AUDIT.md) |
| ROCm | Curated conformance and current target-map promotion closed on the proven RDNA lane | CDNA and additional architecture breadth require real hardware | [`backend/rocm/ROCM_AUDIT.md`](backend/rocm/ROCM_AUDIT.md) |
| NVIDIA | One sm_120 runtime row exists; target artifacts cover a wider surface | Promote CUDA artifacts into compile/link/launch/numerical proof | [`backend/nvidia/NVIDIA_AUDIT.md`](backend/nvidia/NVIDIA_AUDIT.md) |
| Distributed | Single-device and mock-collective development paths exist | Real multi-rank NCCL/RCCL or equivalent execution | [`backend/BACKEND_AUDIT.md`](backend/BACKEND_AUDIT.md) |

## Conformance Result After The 2026-07-12 Evidence Redesign

The conformance cleanup established one consistent completion rule across the
CSV and Markdown outputs:

| Target family | Exact-target cells | Result |
|---|---:|---|
| Portable CPU reference | 7 | 6 reference, 1 missing |
| Native x86 | 7 | 3 complete, 4 missing |
| Apple CPU + GPU | 14 | 7 complete, 5 reference, 2 missing |
| ROCm | 7 | 6 complete, 1 missing |
| NVIDIA sm80/sm90/sm100/sm120 | 28 | 1 complete, 27 missing |

The closeout also hardened proof quality:

- `backend_compile`, `runtime_execute`, and `numerical_check` must all complete
  before the overall cell completes.
- Complete cells require a declared execute-and-compare fixture.
- The independent conformance Evaluator must have a program builder for every
  complete executable cell.
- Stateful `kv_cache_read` uses a state-aware Evaluator path rather than being
  forced through a pure-tensor JIT builder.
- The Apple GPU KV-cache fixture requires `metal_runtime` provenance and skips
  when the Metal DeviceTensor ABI is unavailable; a reference fallback cannot
  earn GPU proof.

## Closed Work Ledger

This section records durable outcomes only. Detailed chronology belongs in the
linked platform audits and [`roadmap/ROADMAP_AUDIT.md`](roadmap/ROADMAP_AUDIT.md).

### Compiler and IR foundations

- Canonical compilation and `CompileResult` metadata are the shared compiler
  contract for `@jit` and `runtime.launch()`.
- Multi-op artifacts carry component ops, blockers, effects, shape envelopes,
  layout contracts, fusion groups, and outputs.
- Named pipeline gates report precise blockers.
- Public API, frontend capture, Graph IR registration, and Schedule IR are
  closed in the generated progress dashboard.
- Effect-aware compiler interfaces and opt-in layout assignment are wired.
- IR parser/printer round-trip fuzzing and differential program generation are
  established regression tools.
- Generated audit documents use one registry and one drift-gated regeneration
  workflow.

### Runtime and backend foundations

- Runtime execution and ABI surfaces are generated and drift-gated.
- Host CPU, Apple CPU/GPU, x86 native lanes, ROCm, and the proven NVIDIA sm_120
  lane have explicit execution evidence.
- The backend-neutral C-ABI GPU launcher hook is implemented.
- Artifact-only, reference, compiled, executable, numerical, and
  hardware-verified states remain distinct.
- Execute-and-compare fixtures are the promotion requirement for complete
  curated cells.

### Apple

- Apple CPU execution through Accelerate/BNNS and correct reference composition
  is established.
- Apple GPU execution spans MPS, MPSGraph, custom MSL, synthesized kernels,
  packaged kernels, encode sessions, and command-buffer chains.
- Descriptor-driven dispatch, feature-limit selection, `auto_batch`, benchmark
  ratchets, and packaged-kernel lifecycle proofs are in place.
- Apple native and reference rows are now distinguished; complete executable
  rows retain independent Evaluator coverage.

### x86 and ROCm

- Portable CPU correctness is labeled `reference`; native x86 is a separate
  target with three currently complete curated rows.
- Native x86 compiled lanes cover a broad primitive surface; their inventory is
  tracked separately from the host reference target.
- ROCm gfx1151 has real compiler-generated and hardware-verified execution,
  including matrix, attention, recurrent/state-space, sparse, and EBM families.
- Six ROCm curated programs complete; stateful KV-cache lowering stops honestly
  at the missing Schedule IR rung.

### Coverage and audit discipline

- Trivial verifier stubs are zero.
- `needs_direct_test` debt is zero; remaining thin-test work is structural,
  family-covered, or hardware-gated.
- Batching, transpose, and lowering primitive contracts are closed.
- Generic `not_applicable` output was replaced on contract axes by explicit
  terminal reasons such as `non_differentiable`, `pure_no_effect`, and
  `no_kernel_required`.
- Generated dashboards, not historical roadmap prose, own live counts.

## Open Action Register

### P0 — Close the newly surfaced software proof gaps

Close gaps in ladder order:

1. Lower stateful `kv_cache_read` beyond Graph IR into verified Schedule, Tile,
   and Target IR.
2. Add exact-target execute-and-compare evidence for the native x86 matrix and
   composition rows that currently lack it.
3. Provide concrete compile/link paths for the open NVIDIA programs.
4. Register executable target paths and attach architecture-aligned numerical
   fixtures with honest provenance.
5. Clear `first_failing_gate` only after every proof rung is complete.

This is primarily an sm_120 adjacency/breadth program today. Hopper sm_90 and
datacenter sm_100 require architecture-specific proof; they are not aliases for
consumer Blackwell.

### P1 — Promote the broader compiler surface

| Workstream | Current open signal | Required closure evidence |
|---|---|---|
| Primitive backend kernels | Registry-level `backend_kernel` remains the largest open axis | Per-target compiled/fused/hardware evidence; do not require every target for all-up compiler readiness |
| Sharding | 43 primitive rows remain open | Mock-mesh equivalence or real distributed execution for the specific rule |
| Tile IR | 13 rows remain mixed | Real lowering or an explicit by-design terminal classification |
| Target IR | 14 rows remain mixed | Native/fused promotion or an intentional reference-only classification |
| Verifiers | 11 ops have no verifier | Add a verifier only where the op has structural or semantic invariants |
| Direct proof | Structural-only and family-covered rows remain | Prioritize high-use/native rows for direct compare; use differential generation for the long tail |
| Benchmarks | Evidence is intentionally sparse | Attach benchmarks first to native and hardware-promoted hot paths |

Live counts and row lists belong to
[`generated/compiler_progress.md`](generated/compiler_progress.md), not this
action table.

### P2 — Hardware and distributed breadth

- Prove Hopper sm_90 and datacenter sm_100 paths on their actual toolchains and
  silicon.
- Prove ROCm CDNA/MI300-class MFMA and low-precision paths independently from
  the RDNA gfx1151 lane.
- Replace mock multi-rank collectives with real NCCL/RCCL or equivalent target
  execution before promoting distributed claims.
- Expand Apple low-precision and Metal-specific performance lanes when the
  required SDK/toolchain is available.

These items are expected hardware/toolchain gates, not evidence that the
compiler is stub-riddled.

## Deferred Or By-Design Work

- A reference execution path may remain intentional when a native kernel has no
  product or performance justification.
- Scalar/configuration primitives may close an axis with a specific terminal
  reason; they do not need meaningless VJP, batching, sharding, or kernel rows.
- Fusion is a performance property. A correct sequential composition can be
  conformance-complete when every component compiles, executes, and matches its
  oracle.
- Platform target maps remain separate because they answer architecture-specific
  questions; merging them solely to reduce document count is deferred.
- Domain roadmaps are planning/history inputs. They are not status authorities.

## Method And Limitations

This audit is a synthesis over checked-in generated dashboards and their source
registries. It does not infer hardware execution from API presence, Target IR
text, or a keyword-only test. Hardware-specific claims remain limited to the
architectures and fixtures recorded in the backend manifests and execution
matrix.

Important interpretation limits:

- The curated matrix contains seven representative programs, not every
  primitive.
- `backend_kernel` is deliberately conservative and is not an all-up compiler
  veto.
- `reference` proves correctness, not target-native performance.
- A skipped hardware fixture preserves honesty but does not replace execution on
  the corresponding hardware lane.
- Numbers copied into this snapshot can age; the linked generated dashboard is
  always authoritative.

## Dashboard And Audit Map

| Question | Read |
|---|---|
| What is the all-up compiler status? | [`generated/compiler_progress.md`](generated/compiler_progress.md) |
| Which curated op×target cells are complete? | [`op_target_conformance.md`](op_target_conformance.md) and [`op_target_conformance.csv`](op_target_conformance.csv) |
| Which compiler phase is open for an op? | [`generated/support_table.md`](generated/support_table.md) |
| Which primitive contract axes remain open? | [`generated/s_series_status.md`](generated/s_series_status.md) |
| Which target paths actually execute? | [`generated/runtime_execution_matrix.md`](generated/runtime_execution_matrix.md) |
| Which backend rows are native, reference, or artifact-only? | [`generated/apple_target_map.md`](generated/apple_target_map.md), [`generated/rocm_target_map.md`](generated/rocm_target_map.md), [`generated/nvidia_sm90_target_map.md`](generated/nvidia_sm90_target_map.md) |
| Which verifiers are real or absent? | [`generated/verifier_coverage.md`](generated/verifier_coverage.md) |
| Which ops have direct test evidence? | [`generated/test_coverage.md`](generated/test_coverage.md) |
| Which ABI symbols are implemented or stubbed? | [`generated/runtime_abi.md`](generated/runtime_abi.md) |
| What is the software-actionable stub surface? | [`stub_surface.md`](stub_surface.md) |
| What are the platform-specific conclusions? | [`backend/BACKEND_AUDIT.md`](backend/BACKEND_AUDIT.md), [`backend/apple/APPLE_AUDIT.md`](backend/apple/APPLE_AUDIT.md), [`backend/nvidia/NVIDIA_AUDIT.md`](backend/nvidia/NVIDIA_AUDIT.md), [`backend/rocm/ROCM_AUDIT.md`](backend/rocm/ROCM_AUDIT.md) |
| What is the planning history? | [`roadmap/ROADMAP_AUDIT.md`](roadmap/ROADMAP_AUDIT.md) |

## Further Questions

- Which NVIDIA curated program should be the next complete end-to-end path after
  the existing sm_120 matrix lane?
- Which of the 43 sharding rows has the highest model-facing value and a
  tractable mock-mesh oracle?
- Which Tile/Target IR mixed rows are real lowering debt versus candidates for
  explicit by-design terminal classification?
- Which structural-only tests cover native hot paths and should be promoted to
  direct execute-and-compare fixtures first?
