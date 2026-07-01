---
last_updated: 2026-06-30
audit_role: plan
plan_state: open
scope: single_gpu
---

# Single-GPU Compiler Closeout Plan

This plan is the operating queue for closing every item that can be honestly
closed on one GPU, without letting multi-rank, unavailable-silicon, or
reference-only decisions hide inside the compiler-progress dashboards.

Status truth remains in generated dashboards:

- `docs/audit/generated/compiler_progress.csv`
- `docs/audit/generated/support_table.csv`
- `docs/audit/generated/s_series_status.md`
- `docs/audit/generated/verifier_coverage.csv`
- `docs/audit/generated/test_coverage.csv`
- `docs/audit/generated/runtime_abi.csv`
- `docs/audit/generated/surface_status.csv`

This document owns sequencing, acceptance criteria, and triage policy only.
The generated companion queue is `docs/audit/generated/single_gpu_closeout.csv`
with a human-readable `docs/audit/generated/single_gpu_closeout.md`; it classifies
the open rows below into single-GPU-closeable, fused reclassification,
multi-GPU-deferred, benchmark-required, and backend-pathway ownership buckets.

## Closeout Boundary

Single-GPU closeout includes:

- One-device Graph IR -> Schedule IR -> Tile IR -> Target IR lowering.
- Native or fused single-device Target IR/codegen claims.
- Runtime ABI symbols needed by a backend that claims native single-device
  execution.
- Verifier, direct-test, and benchmark evidence for promoted rows.
- Mesh/sharding rules that collapse to local placement or one-device identity.

Single-GPU closeout excludes:

- Multi-rank collectives that require real distributed launch.
- Cross-device mesh execution and sharded-state persistence.
- CUDA/ROCm hardware verification on unavailable silicon, except compile/artifact
  evidence explicitly classified as hardware-gated.
- Backend-kernel rows for targets that are not part of the selected one-GPU
  proof path.

## Current Open Queue

Snapshot from the generated dashboards on 2026-06-30:

| Area | Open | Single-GPU action |
|---|---:|---|
| Tile IR | 276 / 315 | Reclassify every partial row as `tile_lowered`, `fused`, `not_applicable`, or `multi_gpu_deferred`. |
| Target IR native/fused codegen | 83 / 315 | Promote high-use reference rows, or mark intentional reference-only lanes. |
| Backend kernel axis | 473 / 480 | Close only by backend/pathway; do not use this as an all-up compiler veto. |
| Benchmark evidence | 276 / 315 | Attach benchmark rows to promoted native/fused paths first. |
| Verifier coverage | 49 / 174 | Add real verifiers for promoted IR lanes before codegen promotion. |
| Direct test evidence | 133 / 480 | Convert structural-only rows that are single-GPU-visible into direct compare fixtures. |
| Runtime ABI symbols | 253 / 640 | Reduce stub-only symbols where a backend claims native execution. |
| Audited repo surfaces | 27 / 58 | Graduate compile-only/scaffold surfaces that exercise this pipeline; archive dead surfaces. |
| Sharding rules | 43 / 480 | Split one-device identity rules from multi-device-deferred rules. |

The highest-volume Tile IR partial families are `elementwise`, `attention`,
`layout_transform`, `loss`, `indexing`, `visual_complex`, `numeric_helper`,
`reduction`, `loop_nest`, and `spectral`. The highest-volume Target IR
reference families are `layout_transform`, `attention`, `indexing`,
`visual_complex`, `acceptance_verification`, `collective`, and `state_update`.

## Promotion Contract

No row should move directly from reference/partial to closed. The required path
is:

1. **Classify:** decide whether the row is single-GPU-closeable,
   fused/not-applicable, intentional reference, or deferred for multi-GPU/hardware.
2. **Schedule IR proof:** confirm Schedule IR is complete and carries the
   attributes needed by Tile IR. This layer is already closed, so failures here
   are regressions.
3. **Tile IR proof:** emit a real Tile IR artifact or explicitly classify the op
   as `fused`/`not_applicable`. `partial` is not an allowed terminal state.
4. **Target IR proof:** emit native/fused Target IR for the chosen backend, or
   mark the lane intentional reference-only with rationale.
5. **Verifier proof:** add an ODS/C++ verifier for promoted operations and
   support ops.
6. **Direct correctness proof:** add direct reference-vs-compiled comparison,
   with hardware-gated labels only when execution genuinely needs unavailable
   hardware.
7. **Benchmark proof:** add a smoke benchmark for each native/fused promotion.
8. **Runtime ABI proof:** ensure the C ABI has a real implementation for every
   native backend claim.
9. **Dashboard refresh:** regenerate generated audit outputs and run
   `graphify update .`.

## Group 1: Core Compiler

### 1. Tile IR Closeout

Goal: reduce `Tile IR partial=276` to zero terminal partials.

Work order:

1. Create a Tile IR triage table from `support_table.csv` with four terminal
   buckets: `tile_lowered`, `fused`, `not_applicable`, `deferred_multi_gpu`.
2. Start with families that unlock the most downstream proof:
   `elementwise`, `reduction`, `layout_transform`, `indexing`, `attention`.
3. For each promoted family, add or update lowering fixtures in the active
   `tests/tessera-ir` lane and keep the Schedule IR -> Tile IR pass ordering
   aligned with `docs/spec/LOWERING_PIPELINE_SPEC.md`.
4. Treat fused Apple/ROCm/x86 paths as valid Tile IR closure only when the row
   does not claim a separate hardware tile lowering.
5. Leave no bare `partial`: every non-promoted row needs a reason attached in
   the source table generator.

Acceptance:

- `compiler_progress.csv` reports Tile IR open count at zero, with remaining
  rows classified as fused/not-applicable/deferred rather than partial.
- At least one lit/direct fixture proves Schedule IR -> Tile IR -> Target IR
  continuity for each promoted family.

### 2. Target IR Native/Fused Codegen

Goal: reduce `Target IR reference=83` by promoting high-use single-GPU lanes.

Work order:

1. Promote in this order: layout/indexing primitives needed by generated code,
   attention kernels, visual_complex/GA/EBM lanes, then state/update support.
2. Keep intentional reference-only rows explicit. CPU numpy reference is allowed;
   a row that claims native GPU execution is not.
3. For NVIDIA/ROCm rows without local hardware, record compile/artifact proof
   separately from execute-and-compare proof.
4. For Apple/x86 lanes, require runtime execution evidence before calling the
   Target IR lane native/fused.

Acceptance:

- No native/fused backend claim depends on a reference Target IR row.
- Remaining references are marked as intentional reference-only or
  hardware-deferred with a backend owner.

### 3. Backend Kernel Axis

Goal: make `backend_kernel` actionable by backend instead of treating 473 open
rows as one compiler veto.

Work order:

1. For each promoted Target IR row, add or update `BackendKernelEntry` evidence
   for the selected backend.
2. Close backend rows only for backends that have a real pathway:
   Apple GPU/CPU, x86, ROCm artifact/execute lanes, NVIDIA artifact/execute
   lanes.
3. Keep missing target rows out of the single-GPU closure denominator unless
   the row participates in the selected proof path.

Acceptance:

- `s_series_status.md` backend proof by target explains native proven,
  reference, open artifact/planned, and missing-target counts without ambiguity.

## Group 2: Verification And Test

### 4. Verifier Coverage

Goal: reduce `no_verifier=49` for promoted compiler lanes.

Priority verifier queue:

- Basic promoted ops: `AddOp`, `DivOp`, `MulOp`, `ReduceOp`, `ReluOp`,
  `SigmoidOp`, `SiluOp`, `SiluMulOp`, `SinOp`, `SoftplusOp`, `SubOp`, `TanhOp`,
  `GeluOp`, `SelectOp`, `MaskedFillOp`.
- Runtime/state ops: `CacheCommitOp`, `CachePageLookupOp`, `CacheRollbackOp`,
  `WriteRowOp`.
- Optimizer ops: `AdamOp`, `AdamWOp`, `AdafactorOp`, `LionOp`, `MomentumOp`.
- Mesh/neighbors ops: all `Neighbors*` ops, but close one-device identity first
  and defer true halo exchange to multi-GPU validation.
- Tile/attention support ops: `CausalMaskOp`, `DropoutMaskOp`,
  `LseAccumulateOp`, `LseLoadOp`, `RopeSplitOp`, `RopeMergeOp`, `NTKRopeOp`,
  `ALiBiOp`.

Acceptance:

- Every op promoted through Tile IR or Target IR has a real verifier.
- Verifier tests include positive and negative cases for shape, dtype, layout,
  effect, and region constraints where applicable.

### 5. Direct Test Evidence

Goal: reduce single-GPU `structural_only`/thin rows into direct fixtures.

Work order:

1. Close compiler-visible structural rows first:
   `associative_scan`, `dynamic_slice`, `dynamic_update_slice`, `index_select`,
   `index_update`, `select`, `split`, `take`, `vmap`, `value_and_grad`, `vjp`,
   `jvp`, `remat`, `checkpoint`.
2. Close one-device sharding identity rows:
   `named_sharding`, `partition_spec`, `shard_map`, `pmap`.
3. Keep dataset/tokenizer/serialization rows separate; they are single-process
   closeable but not compiler-critical.
4. For every native/fused codegen promotion, add a direct compare fixture before
   adding benchmark evidence.

Acceptance:

- `test_coverage.csv` distinguishes direct correctness, family coverage,
  hardware-gated execution, and intentionally structural metadata rows.

### 6. Benchmark Evidence

Goal: attach benchmark evidence to promoted native/hardware paths, not to every
reference row.

Work order:

1. Benchmark the same rows promoted in Target IR: layout/indexing, attention,
   reductions/elementwise chains, visual_complex, GA/EBM.
2. Use smoke shapes for CI and preserve larger sweeps as opt-in.
3. Record fused-vs-reference and compiled-vs-reference evidence in the existing
   benchmark JSON schemas.

Acceptance:

- Every single-GPU native/fused row has either benchmark evidence or a written
  reason why benchmarking is not applicable.

## Group 3: Runtime, Surfaces, Mesh/Sharding

### 7. Runtime ABI Symbols

Goal: eliminate stub-only ABI rows that back native execution claims.

Work order:

1. Audit ABI rows by backend and symbol. Current counts are Apple 557, x86 68,
   ROCm 10, NVIDIA 5.
2. For Apple GPU, separate real `.mm` implementations from matching stub
   symbols used on non-Darwin hosts. Stubs are acceptable only as host-portable
   compile fallbacks.
3. For ROCm/NVIDIA, keep compile/artifact symbols separate from execute symbols
   until hardware proof lands.
4. Add ABI smoke tests for each new native single-GPU symbol.

Acceptance:

- No dashboard row can claim native backend execution when the corresponding ABI
  symbol resolves only to a stub on the target host.

### 8. Audited Repo Surfaces

Goal: graduate surfaces that exercise compiler pathways; archive or mark dead
the rest.

Work order:

1. Promote `tests/tessera-ir` from compile-only once the lit lane is the active
   proof for Schedule -> Tile -> Target lowering.
2. Promote benchmark library workloads from compile-only only when they execute
   a real smoke command, not just import.
3. Graduate examples only if they use current public APIs and exercise a real
   compiler pathway.
4. Archive empty scaffold directories unless an owner and next proof command are
   recorded.

Acceptance:

- `surface_status.csv` has no scaffold/compile-only entry without either a
  concrete promotion command or an archive decision.

### 9. Mesh/Sharding Rules

Goal: close all sharding rules that have a one-device meaning and defer only
real multi-device behavior.

Work order:

1. Classify each of the 43 open sharding rules as:
   `single_device_identity`, `local_layout_transform`, `requires_collective`,
   or `multi_rank_deferred`.
2. Close one-device identity rules with direct tests proving identical values and
   stable metadata.
3. Close local layout transforms when they lower through Schedule IR and either
   Tile IR or an explicit fused/not-applicable path.
4. Keep `requires_collective` and `multi_rank_deferred` out of the single-GPU
   closeout denominator; link them to distributed validation instead.

Acceptance:

- `sharding_rule` open count only reflects true multi-device or unimplemented
  local-layout work, never one-device identity semantics.

## Execution Order

1. **Bookkeeping PR:** add terminal classifications for Tile IR, Target IR
   references, sharding rules, and backend-kernel backend/pathway ownership.
   Initial generated queue: `docs/audit/generated/single_gpu_closeout.csv`.
2. **Compiler spine PRs:** promote `elementwise`, `reduction`,
   `layout_transform`, and `indexing` through Tile IR and Target IR with
   verifier and direct tests.
3. **Attention PRs:** promote `flash_attn`, `multi_head_attention`,
   `gqa/mqa/mla`, local/sliding sparse attention, and LSE/mask support ops.
4. **Domain PRs:** promote visual_complex, GA, EBM, spectral, and linalg rows
   that are single-GPU-closeable.
5. **Runtime PRs:** remove native-claim/stub mismatches and add ABI smokes.
6. **Surface PRs:** graduate lit/benchmark/example surfaces that now exercise
   the promoted compiler path.
7. **Final gate PR:** regenerate dashboards, run validation, and verify no
   single-GPU-closeable row remains as partial/reference/no-verifier/structural.

## Final Closeout Definition

This plan is complete when:

- Tile IR has zero unclassified `partial` rows.
- Target IR has zero unowned `reference` rows for single-GPU native/fused
  claims.
- Every promoted row has verifier, direct test, and benchmark evidence or an
  explicit not-applicable rationale.
- Runtime ABI contains no target-stub-only implementation behind a native
  execution claim.
- Mesh/sharding open work is split cleanly between one-device identity closure
  and multi-device deferred validation.
- `graphify update .` has refreshed the graph after the final code/doc change.
