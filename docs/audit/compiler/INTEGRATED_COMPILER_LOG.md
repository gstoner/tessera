---
last_updated: 2026-09-17
audit_role: reference
---

# Integrated compiler engineering log

Revision-bound provenance, not a sequencing or capability-status authority.
Historical “Next” lists are preserved after routing their obligations to the
[live plan](INTEGRATED_COMPILER_PLAN.md#routing-index). Entries may describe
uncommitted work; a date or test count does not establish merge or promotion.
`last_updated` records the latest append or substantive correction.

For new entries use a date-first heading and the five fields below. Name one
primary Owner; end the five-field block with `<!-- entry-fields:end -->` and
link additional owners in the body. Update that owner record in
the same PR. Evidence corrections are explicit; do not rewrite old results as
current proof. Current priorities live only in the plan.

### 2026-09-04 — Engineering follow-through

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: #721, #722, #723

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Merged snapshots: [PR #721](https://github.com/gstoner/tessera/pull/721),
[PR #722](https://github.com/gstoner/tessera/pull/722), and
[PR #723](https://github.com/gstoner/tessera/pull/723).
The following work remains independently gated; publishing the snapshot does
not close native/compiler evidence gaps.

| Owner | Current action and acceptance |
|---|---|
| APPLE-DISPATCH-WEDGE-1 / telemetry | Scoped telemetry capture restores prior process state, including nested/error paths; two device-clock tests adopted it. An opt-in pytest order tracer records transitions after teardown without resetting them. The original process latch and downstream MoE failures still require the triggering order and exact Metal reproduction. |
| MSW-9 | Program-pair native evaluator adapter and separate reference composition/identity law family implemented. Graph IR fragment inventory and the actual fusion consumer remain open. |
| FRONTEND-IR-MEDIUM-1 | PR #721 contains native pre-bucket CSE and narrow source-loop recognition in rank/prune mode. Native instantiation, arbiter consumption, tracer/MLIR raising and attention remain open. |
| APPLE-DISPATCH-WEDGE-1 | Metal 4 direct wait now stamps timeout kind/message. Branch-specific M1 Max E2E re-seals were recorded in commits `3e0ccb21` (#721) and `9773db8f` (#722); this is separate from timeout fault injection and telemetry-order reproduction. Lowp MoE remains opt-in without a measured ledger row; cooperative rewrite remains open. |
| DISPATCH-BREAKER cross-backend | CUDA and HIP waits remain unbounded. A replacement must poison the owning context on timeout and retain every outstanding buffer/module/event; adding a polling deadline followed by current synchronous cleanup would still hang or free live storage. First slices should target one owning bridge each, with injected not-ready/error cases before GPU proof. |
| IKF-1 | Bind `INTRA_KERNEL_FEEDBACK_PLAN.md` here. P0 extends the existing D2 gfx1151 timing probe; P2 waits for green clocks and consumes Schedule-Object region identity. P1 host schema/math is independent. Shared D2 cache insertion/loading and dispatch admission now reject L2/L3 or malformed instrumentation levels. No IR instrumentation is authorized by a scalar clock sample alone. |
| Toolchain evidence | Assertion-enabled LLVM remains an explicit fleet proof gap. Do not interpret release-build negative tests as assertion-enabled contract falsification. The sandbox/Metal mechanism remains unproven; no claim of its root cause is made. |
| Apple cross-run decision | Default remains mean ± t·sd/√n. The eight-process M1 Max cohort and predeclared synthetic 1.5x/3.0x slowdowns produced no policy disagreements across 24 decisions. Retain the mean default; the median remains opt-in. See `benchmarks/baselines/apple7_cross_run_policy_20260904/summary.json`. This comparison installs no ledger and does not establish robustness across workloads or physical stalls. |
| Generated coverage | Decision #26 now assigns coverage evidence to revision-bound CI artifacts: source commit/tree digest, output hashes, run and attempt. The owning generator and semantic tests remain required; missing/expired evidence must be regenerated from the cited revision. |

All numerical/backend promotion remains architecture-owned. The immediate
reproduction priority is telemetry attribution; a scoped cleanup proves a
state leak is fixed but does not establish the root cause of the reported
order-dependent failure.

#### Next acceptance steps after #723

This sequence refines the engineering follow-through above; it does not replace
the broader central queue or reopen its completed rows.

| Order | Owner and next bounded deliverable | Acceptance before expanding scope |
|---:|---|---|
| 1 | APPLE-DISPATCH-WEDGE-1: capture the ordered failing session with `scripts.trace_apple_telemetry`, then minimize the prefix that changes capture state or breaks downstream MoE. | Fresh-runtime reproduction, first transition attributed after teardown, and the minimized order passes after a cause-specific fix. A test passing alone or another scoped state cleanup is not closure. |
| 2 | FRONTEND-IR-MEDIUM-1: instantiate one optimized matmul recipe for two valid shape buckets through the native compiler. `ParametricRecipe.rank_buckets` currently emits only `BucketRank`. | Both instances bind the same recipe/compiler digest and complete shape witnesses; invalid witnesses fail before lowering; execute against the original oracle on the owning backend. Then connect the resulting native artifacts to arbiter admission. Broader attention recognition follows that working path. |
| 3 | MSW-9: enumerate the Graph IR envelope for affine composition and ReLU identity extension, then wire one eligible program pair into a fusion candidate's evaluator gate. | `program_pair_equivalence` must see native provenance on both sides; wrong-shape, unsupported-policy and reference-only cases refuse admission. Reference `ann_identity_checks` is the oracle, not production proof. |
| 4 | DISPATCH-BREAKER: implement one CUDA or HIP bridge's bounded wait with an explicit poisoned-context/resource-retention state. | Inject not-ready, error and timeout before exact-device tests; prove cleanup cannot free live storage or enter another unbounded wait. Migrate sibling bridges only after the owning state machine is sound. |
| 5 | IKF-P1 host schema/math can proceed independently; IKF-P0 continues cross-CU, monotonicity and read-cost checks on gfx1151. | Validate synthetic slot/clock edge cases for P1; retain WSL regression-only limits on P0. P2/P3 instrumentation waits for the specified clock evidence and Schedule-Object identity. |

The Apple policy experiment is complete for its declared cohort: retain the
existing default. Lowp MoE ledger admission/cooperative kernel work and an
assertions-enabled LLVM remain separately owned follow-ups; neither is closed
by a policy comparison or a fleet visibility probe.

### 2026-09-04 — Archive follow-through

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

The superseded code-review snapshot and typing inventory are archived with
[owning-audit summaries](COMPILER_AUDIT.md#archive-reconciliation--2026-09-04).
No new implementation queue is created. Surviving actions remain:

- **W1.1 / foundation F2:** migrate the two tensor-valued MMA constructors in
  `TileIRLoweringPass` and obtain NVIDIA Target/device proof before deleting
  the checked tensor-value lane. The old bare-fragment permissive branch is
  already gone; do not recreate that closed task.
- **DISPATCH-BREAKER / NVIDIA `P3-SOURCE-ONLY-2026-08-30`:** inject allocation
  failure in flash-backward cleanup and prove exactly-once release of acquired
  resources. Recorded normal-path device tests do not establish this case.
- **`P2-REVIEW-SHARED-PASSES-2026-08-29`:** preserve the existing backend
  queue obligations and their exact-host acceptance. Reconcile later receipts
  per row before claiming closure; the P3 closure note is not blanket P2 proof.

Older architecture reviews and overlapping scoped plans remain live during
reconciliation. Foundation F1 (NVIDIA scheduled matmul Graph re-entry) remains
the first code migration; archiving neither replaces nor completes it.

### 2026-09-04 — Foundation F1 implementation

Owner: [W5.5](INTEGRATED_COMPILER_PLAN.md#w55)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

`nvidia_native.package_scheduled_matmul(artifact)` now constructs the descriptor
from the scheduled launch contract and compiles only its Tile IR. The Graph
argument, dynamic Graph clone and discarded base-image build are removed.
Schedule fields and Tile entry ABI must agree with descriptor metadata before
compilation. The driver records the real adjacent scheduled ancestry.

This is the first bounded migration, not closure of E2E-REAL-3 or all native
packaging: F2 still owns the remaining Graph package roots and typed producers.
Validation and exact-device scope are recorded in the NVIDIA queue under
`IR-NATIVE-FOUNDATION-1`. Physical kernel optimization is outside this cut.

### 2026-09-05 — Foundation F2 unary driver migration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

The first E2E-REAL-5 NVIDIA subset now traverses native Graph → Schedule → Tile
before packaging: static f32 last-axis softmax and serial rank-reducing
sum/mean/max on arbitrary axes. The native passes own the launch wrapper;
Python validates/binds the artifact and does not synthesize a replacement body.
The established SM120 approximate-exp policy and runtime scalar ABI are retained.

Exact-device old/new numerical and ABI tests, independent Schedule replay,
policy tampering, and extra-work rejection cover this subset. A discovered
legacy canonical-reduce kind bug is fixed alongside the differential tests.
This closes the default-driver migration for that envelope, not F2: direct
Graph clients, narrow dtypes, min/keepdims/cooperative reductions and other
backends' remaining package families retain explicit retirement obligations.
See all four backend queues under `IR-NATIVE-FOUNDATION-1` for evidence boundaries.

### 2026-09-05 — F2 direct unary entry points

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

The migrated NVIDIA static f32 envelope now shares one native scheduling path
between driver and direct package clients. Supported requests require the
native scheduling compiler; they do not silently reconstruct Tile IR when it
is absent. Narrow dtypes and min/keepdims/cooperative reductions remain explicit
unmigrated envelopes, with the old implementation retained privately until their
own comparisons pass. Direct-client migration is closed for f32 softmax and
serial sum/mean/max; full constructor retirement remains open.

### 2026-09-05 — Next ten unary foundation actions

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

These are bounded cuts under F2 / E2E-REAL-5, in dependency order. Completion
requires native replay, correct ABI and owning RTX numerical evidence; constructor
retirement also requires all production callers to migrate.

| Action | Deliverable | State |
|---|---|---|
| F2-U1 | f16 scheduled softmax | implemented; RTX parity passed |
| F2-U2 | bf16 scheduled softmax | implemented; RTX parity passed |
| F2-U3 | scheduled min reduction | implemented; validated |
| F2-U4 | keepdims reduction carrier and scheduling | implemented; validated |
| F2-U5 | f16 input / f32 output reduction | implemented; validated |
| F2-U6 | bf16 input / f32 output reduction | implemented; validated |
| F2-U7 | explicit cooperative_128 reduction scheduling | implemented; validated |
| F2-U8 | retire production Graph softmax constructor | implemented; validated |
| F2-U9 | retire production Graph reduction constructor | implemented; validated |
| F2-U10 | close caller inventory, native replay and negative-policy gates | implemented; validated |

F2-U1–U10 implementation evidence: 184 RTX 5070 execute-and-compare cases
cover f32/f16/bf16 softmax and all 144 combinations of reduction storage, kind,
axis, keepdims and serial/cooperative policy. New packages match the retained
test-only baseline and NumPy. Another 24 cooperative cases cover 257-element
contiguous/strided axes; no performance promotion is inferred. The old
production unary constructors are removed. Native Schedule replay, policy
mutation refusal, missing-compiler refusal, one Tile compilation and explicit
unsupported-adjoint failure are regression gated. This closes the ten bounded
unary actions, not all of F2: norm, attention, packed matmul and sibling direct
package migrations remain open in the survey inventory.

### 2026-09-05 — F2 norm and forward-attention contracts

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owning item: E2E-REAL-5; synchronization key `IR-NATIVE-FOUNDATION-1`.
Implemented native SM120 unweighted RMSNorm/RMSNormSafe/LayerNorm scheduling and
forward attention packaging for f16/bf16/f32 storage. Native replay validates
the Tile artifact before one target compilation; the driver records real adjacent
Graph/Schedule/Tile lineage. The two Graph constructors are now test-only baselines.

Attention forward hashes now preserve exact f32 policy bits on all architectures.
The initial cut refused NVIDIA short-query causal/window masks; F2-A2 below
removes that restriction after aligning both physical kernels.
No new backward or saved-LSE support is implied. Numerical parity, not performance
promotion, is the acceptance criterion for this migration.

Next actions, in dependency order:

1. F2-A2: align NVIDIA ragged forward/backward masks and audit backward float hashes.
2. F2-A3: native paired saved-LSE migration implemented in the bounded f32 SM120 lane; integrate residual-policy selection separately.
3. F2-P1: signed INT4 migration implemented; extend scale-layout projections for NVFP4/MX.
4. F2-S1: bounded paged read migration implemented; replay-SSM and MoE state/workspace contracts remain.
5. F2-C1: retire sibling direct package constructors using their existing scheduled
   consumers only where storage, policy and ABI envelopes agree.

The updated survey lists every remaining NVIDIA constructor family and retains
all five backend package entry points in the broader census.

### 2026-09-05 — F2 aligned masks and package contracts

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owning item: E2E-REAL-5; synchronization key `IR-NATIVE-FOUNDATION-1`.

- **F2-A2 implemented:** NVIDIA forward/backward Tile kernels use
  `q + max(Sk - Sq, 0)` for causal and window masks, including saved-LSE variants.
  Short-query native scheduling is admitted. Backward Schedule identity now
  encodes exact f32 scale/softcap/dropout bits; regenerate older serialized
  backward artifacts. This changes no pointer ABI or physical schedule.
- **F2-A3 contract step implemented; native migration open:** explicit paired
  packaging checks shape, physical f32 policy, Q/K/V and saved-LSE bindings before
  compiling either half. Both descriptors carry a checkpoint digest. Unsupported
  score-modifier aliases and reordered gradient returns are refused. The current
  Graph forward op has a single native result; the historical Python saved-LSE
  API has two. Next add a native multi-result checkpoint producer and a consuming
  Schedule contract, then retire both Python Tile constructors together. A digest
  does not establish that a runtime LSE buffer came from the specified inputs.
- **F2-P1 reviewed, native migration open:** packed physical fields now reject
  lossy/noninteger values and negative offsets before target compilation. Next
  start with unscaled signed INT4 A/B/i32 output: make native storage packing,
  packing axis, container strides and output roles part of the Schedule identity;
  replay the serialized artifact and compare odd-K output on SM120. NVFP4/MX
  require scale-layout projections before admission. No packed route was promoted.
- **F2-S1 reviewed, native migration open:** paged-KV bounds no longer coerce
  strings/floats/bools, and unrelated function returns are refused. Shared replay
  geometry/span validation rejects noninteger values and overflowing allocation
  sizes. Next native paged read must retain table bounds and logical start/end;
  replay decode/flush must share an effects/lifetime contract; MoE must retain
  multi-entry capacities and workspace ownership. These are separate bounded
  producer migrations, not another Python wrapper around Graph reconstruction.

Validation includes 18 RTX 5070 attention cases, three forward-save/backward-load
pairs (both rectangular directions and unmasked), and 13 packed/paged/replay
regressions on the rebuilt NVIDIA compiler. This is correctness evidence, not
latency evidence. Princess-Luna runs shared contracts and the existing ROCm
backward lane (three gfx1151 device cases passed). The focused contract, registry
and dtype suite passed 510 tests with seven explicit skips; 11 audit tests, three
native IR fixtures, Ruff, the zero-error mypy ratchet and all 30 generated-doc
gates passed. Apple owning-host evidence is still separate.

### 2026-09-05 — Deleted functionality reassessment

Owner: [W3.3](INTEGRATED_COMPILER_PLAN.md#w33)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner: `IR-NATIVE-FOUNDATION-1`; this section routes reassessment into existing
work items rather than creating a parallel implementation queue. Status: open
architecture review, not authorization to restore deleted implementations or
claim new backend support.

Decisions #29/#31 remove unsupported production declarations and competing
implementations. They do not establish that the underlying semantics are
unhelpful. A bounded experiment may be retained with an owner, producer,
consumer and exit criteria, explicitly outside production capability claims.
The queue stays-deleted gate continues to prevent accidental restoration; change
it only with an accepted replacement contract and executable positive/negative
fixtures. No gate is weakened by this planning update.

| Reassessment | Existing owner and priority | Producer, consumer and native boundary | Acceptance evidence |
|---|---|---|---|
| Queue/pipeline ownership semantics | W2.4a / CAKE / SO-2, with W5.2 scheduling; next bounded spike | One native staged GEMM or attention producer; existing Tile pipeline/token, role and barrier legality consumers; architecture-owned NVVM/ROCDL lowering. First determine whether current operations already express capacity, acquire/publish/consume/release, slot generation and safe buffer reuse. A separate dialect requires a demonstrated representational gap. | Parse/serialize/replay; reject premature reuse, wrong slot/phase, missing release and illegal effects; execute against the current baseline. Measure device latency, barrier count, shared memory, registers and occupancy on NVIDIA and ROCm separately. No gain is presumed from adding vocabulary. |
| Native residual-policy consumption | W5.1 / AD D5 / foundation F3; existing high-priority work | Existing demand/retained-residual analysis and selected SAVE/RECOMPUTE/HYBRID plan feed the C++ region adjoint and native package. Do not re-enable the annotation-only EBM pass as a substitute. | Same forward/gradient results and stochastic identity; effectful replay refused; selected policy matches executed save/replay operations. Measure peak retained memory, complete forward/replay/backward work and device latency for each owning family. |
| Deleted-verifier invariant inventory | W2.4 legality consolidation; bounded correctness audit | Extract useful invariants from deleted ScheduleOps.cpp/Target helpers; map each to current registered ODS/type verification, IRContractLegality or TileDataflowLegality. Canonical producers and their actual lowered output are the inputs. Restore only a missing, still-valid invariant in its current authority. | Disposition for every reviewed invariant: covered, obsolete, intentionally extensible or missing. Missing checks require minimal malformed-IR rejection and valid-production acceptance fixtures. Check compiler cost if an analysis becomes nonlocal; no runtime speedup claim is required. |
| Native sharding / Shardy integration | W5.4 / W5.4-RESHARD-1 / foundation F3; architecture evaluation | Current typed placement lattice, mesh and explicit reshard SSA feed either native Tessera analysis or an evaluated typed Shardy bridge, then existing Schedule/Tile collectives and backend transport. Assess independently of TPU support. | Preserve partial reductions, local shapes, subgroup identity and nested-region constraints through round trips. Reject unknown metadata rather than silently treating it as replicated. Compare against existing mock-mesh semantics; require owning-host multi-rank packets before native transport or performance claims. Account for compile cost, communication bytes and device latency. |
| Neighbors/halo pipeline composition | Existing PDE/distributed work, W5.4 and COMP-SCHED-OVERLAP-1 | Existing halo inference, stencil materialization and transport passes produce native dataflow; overlap and target/runtime consumers must carry true-use dependencies through pack/exchange/unpack/compute. Deleted plugin pipeline names are not the deliverable. | One serialized end-to-end stencil workload, correct boundaries and deterministic exchanges; negative dependency/lifetime tests. Compare sequential and overlapped device/transport timelines on the actual multi-rank backend before claiming hidden communication. |
| StableHLO interoperability | Foundation F1/F3; deferred until a concrete external consumer is named | A bounded canonical Graph export/import boundary, with an identified framework/compiler or differential oracle as consumer. It is not a replacement for the native backend spine and does not reactivate TPU. | Preserve shapes, dtype, numerical policy and effects; explicitly refuse unsupported operations; round-trip and numerical checks with that consumer. Evaluate maintenance/compile cost; no performance benefit is assumed. |

Disposition ledger from the source/history survey:

- Queue MLIR implementation: remains deleted; evaluate ownership semantics above.
- EBM checkpoint pass: remains experimental and registered, removed only from
  the default pipeline. Its useful demand-aware behavior belongs to W5.1.
- Duplicate attention ODS: remains deleted. Canonical `tessera_attn.lse.save`
  and `lse.load` still exist; F2-A3 must evaluate those before defining another
  checkpoint vocabulary. Their existence alone does not close native producer
  or packaging gaps.
- Duplicate Tile/Schedule dialects, private registration scaffolds, permissive
  bare fragments and synchronization attribute escape hatches: keep removed.
  Preserve useful invariants through current typed authorities.
- Old TPP and scaling/resilience directories: implementations remain under
  `src/solvers/tpp` and `src/solvers/scaling_resilience`; these are relocation/
  consolidation cases, not lost capabilities. Likewise, deleting the collective
  generated-type implementation file is not deletion of async collective semantics.
- TPU, Metalium, Cerebras and Rubin CPX retirement remains a target-scope decision.
  Reuse portable ideas only through a named consumer; backend reactivation needs
  its own requirement, implementation owner and exact-device validation.

Backend assessment: NVIDIA and ROCm own separate physical pipeline/transport
experiments. Apple needs its own Metal synchronization and residency mapping;
x86 needs CPU ownership/effect validation, with explicit no-async behavior where
appropriate. Shared IR proofs do not transfer physical schedules or measurements.
All four backend queues link this reassessment; implementation remains ordered
by the existing owners above and the active F2 saved-LSE/packed/stateful sequence.

### 2026-09-05 — F2 unary artifact replay correction

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: #726

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

E2E-REAL-5 / `IR-NATIVE-FOUNDATION-1`, review follow-up to PR #726: native
Schedule-to-Tile replay is now mandatory in the shared NVIDIA unary package
consumer, covering softmax, serial/cooperative reductions and norm. Matching
attributes, entry ABI and a retained schedule hash cannot prove that serialized
Tile operands still implement the recorded schedule. Regressions mutate source/
destination operands without changing those markers and require rejection before
target compilation; missing production `tessera-opt` also fails closed. This
corrects the earlier F2-U10 replay-coverage claim rather than opening a new queue.

### 2026-09-05 — Five-action native ownership loop

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner: E2E-REAL-5 / W2.4 / W2.4a; synchronization key `IR-NATIVE-FOUNDATION-1`.
This increment supersedes the F2-A3/P1/S1 migration status above for the bounded
families listed here. It does not close all packed or stateful execution.

1. **F2-A3 implemented:** registered `tessera_attn.checkpoint_forward` yields
   output and row LSE; `checkpoint_backward` consumes dO/Q/K/V/LSE and yields
   dQ/dK/dV. Existing `lse.save/load` only transfer an already computed value, so
   they cannot substitute for the fused multi-result producer. Native
   `schedule.attention_checkpoint` retains shapes, binding roles, exact f32 scale,
   causal alignment and hash. Packaging replays Schedule-to-Tile and never
   reconstructs Graph. The old forward and saved-backward constructors are
   retired from production. Recompute backward remains a separate migration.
2. **W2.4 audit:** the deleted file's useful prefetch memory-space allowlist was
   missing in registered verification; restored with seven valid spaces and
   three invalid-input tests. Dispositions of the reviewed checks are below.
3. **W2.4a spike:** existing pipeline depth/stage/phase, roles, allocation roots
   and completion tokens express the bounded ownership vocabulary. A decorated
   `mbarrier.try_wait` falsely cleared reuse hazards; a native negative fixture
   reproduced it before the fix. Only registered completing operations now
   release that local hazard. No queue dialect is restored. **Still open:** the
   checker clears pending hazards broadly at waits and walks regions in order;
   allocation-specific release, control-flow joins and consumer-completion
   lifetimes need relational analysis. Do not call this full queue ownership.
4. **F2-P1 signed INT4 implemented:** builtin i4 Graph tensors enter the existing
   native matmul Schedule. C++ derives signed low-nibble-first byte packing,
   A-axis 1/B-axis 0, physical shapes, container strides, zero offsets and absent
   scales in typed packed views. Serialized Schedule replay rejects edited Tile
   operands before compilation. The fixed runtime entry ABI is preserved.
   NVFP4/MX scale layouts remain open; no performance promotion occurred.
5. **F2-S1 bounded paged read implemented:** registered physical `tessera.paged_kv_read`
   verifies static f32 pages, i32 table, logical interval and output shape.
   Native Schedule binds read-only borrowing, runtime physical-index validation,
   layout and names; the package consumes the replayed artifact. Runtime still
   validates actual table contents. Stateful mutation, replay decode/flush and
   MoE workspace/lifetime contracts remain separate open cuts.

The canonical driver retains adjacent native Graph/Schedule/Tile lineage for all
three package migrations. Descriptor identity is not proof that an external LSE
buffer contains values from the paired forward input. Python remains the
frontend, descriptor validation and oracle; native C++ owns these Tile programs.

#### Deleted-verifier disposition inventory

History inspected: `e34a59de^` ScheduleOps.cpp and Target/TesseraTargetIR.cpp.
The old files were unbuilt; comments claiming invariants were not enforcement.

| Historical checks | Disposition and current authority |
|---|---|
| Mesh dimensions, axis strings/count; mesh body/symbol; pipeline schedule/microbatches; stage devices | Covered by registered Schedule ODS and ScheduleDialect.cpp. Current checks additionally enforce result/yield roles. |
| Prefetch destination and overlap | Missing destination allowlist restored in ScheduleDialect.cpp; overlap/type preservation already covered. |
| Schedule async-copy spaces differ, nonnegative stage; await arity | Covered by Schedule ODS/verifiers. Historical prose claimed known-space validation but the implementation only checked presence/difference. Extending the space vocabulary is a separate contract decision. |
| Artifact hash/arch; knob name/choices/logit cardinality | Covered by registered Schedule verification; not a second verifier implementation. |
| Cache kv/pt/ring construction and page-lookup arity | Historical cache scaffold has no active native package producer. Its handle API is obsolete for these tensor package boundaries; do not resurrect it solely for arity checks. New paged-read ODS binds its actual tensor interface. |
| Tile alloc-shared memref operand; required async-copy/wait stage and two memrefs | Obsolete historical signature. Native tile.alloc returns an SSA buffer; copy/wait use typed tokens. Optional stage nonnegativity already lives in TileOps.cpp. Requiring historical operands would reject real producers. |
| Mbarrier count/scope, positive arrive bytes, try-wait arity | Active init/arrive/wait ODS and TileOps.cpp cover slots/phase bits/bytes/token shapes; TileDataflowLegality owns pairing and slot agreement. Historical alloc/scope spelling is not the current API. PMV11 compatibility checks must not become a competing native authority. |
| Atomic order/scope and divergent barrier | Old generic spellings retain PMV11 compatibility checks; current target effects and relational WarpSpec legality own executable semantics. No new atomic support implied. |
| Reduction op/order strings and unknown-op warning | Obsolete generic reduction helper. Registered reduction kernels have closed operation/storage contracts; the old advisory unknown-op warning is not a useful correctness gate. |
| Target warp_config positive count/power-of-two warning; smem_layout attr presence | Obsolete standalone target helper spellings. Current typed schedule, layout and backend launch/resource contracts are authoritative; a power-of-two warning cannot replace architecture limits. |
| Target TMA cp_size positive | Historical descriptor signature obsolete; current descriptor expected-byte and arrival agreement is checked by Tile types/ops and TileDataflowLegality. The old comment about global source memory was never enforced there. |

Validation in this loop: RTX 5070 passed nine native package device cases
(three checkpoint pairs, two INT4 cases and four paged intervals/invalid tables).
Princess-Luna gfx1151 passed 39 existing WMMA compiler/runtime tests, including
native SSA/LDS structure. These are correctness checks. There is no new
cross-backend latency/occupancy comparison, and the ownership spike's performance
exit criterion remains open. Apple/x86 physical migrations and owning-host proof
remain separate. Focused registry, native fixture, lint and generated-doc results
are recorded with the implementation rather than inferred from these device runs.

Final focused validation passed **438 tests**, including registry, dtype, artifact
replay, driver lineage and audit lifecycle checks. The separate canonical/consumer
cohort passed 90 tests with 12 explicit host/tool skips. Both native dataflow/reuse
fixtures passed, including strict registered parsing for the changed reuse
fixture. Ruff and the zero-error mypy ratchet passed. The ROCm SSA/LDS compiler
probe preserved allocation/token ownership through target lowering at one, two
and four stages; it measures host compiler cost only, not device latency.

### 2026-09-05 — Allocation lifetime analysis and device regression comparison

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`.
This increment supersedes the blanket-release and preorder-walk limitations in
the five-action loop for the bounded registered `!tile.buffer` vocabulary.

- Track all pending storage footprints and async accesses per allocation. A
  typed wait retires only its direct SSA copy producer; a keyed legacy wait
  retires matching copies. An arrival-only mbarrier wait does not prove buffer
  completion. Preserve the declared keyless legacy wait-all envelope.
- Thread barriers retire synchronous write hazards, not outstanding DMA.
  Check explicit copy destinations, source borrows, premature deallocation,
  use-after-free, and overlapping writes, including earlier disjoint writes.
- Merge `scf.if` branches and CFG predecessors by union of may-live facts.
  Iterate `scf.for` and CFG backedges to a finite fixed point; statically empty
  and single-trip loops do not invent backedge accesses. Follow supported
  allocation aliases across structured results and loop carries without
  recursive cycles. Preserve signed-stride footprint bounds.
- Unknown allocation origins, unsupported buffer-access operations and region
  forms fail closed. Loop-carried async tokens cannot release by static producer
  identity alone: dynamic slot/generation correlation remains open. Function/CFG
  buffer arguments need an alias/ownership contract before admission. This is
  not a universal interprocedural lifetime proof or a new queue dialect.

`benchmarks/record_allocation_lifetime_comparison.py` compares baseline and
candidate compilers on each owning GPU, preserves timing domains, validates
outputs and requires identical native images. It measures existing native attention/GEMM
routes, not a new queue schedule. This verifier does not optimize physical
schedules; selector promotion and occupancy gains are not implied. The older
memref reuse-group planner and backend consumption of its allocation groups
remain a separate integration task.

ROCm exact-host results remain in `benchmarks/baselines/allocation_lifetime_rocm.json`.
The NVIDIA allocation-lifetime comparison is withdrawn: the harness set the core
compiler variable while native lowering consumed a different compiler variable.
Its recorded binary hashes did not identify the native lowerers actually used.
The NVIDIA JSON is now an explicit withdrawal record; new measurements are required.

| Host / workload | Before → after median | Domain / proof |
|---|---|---|
| Princess-Luna gfx1151, 512³ f16 direct GEMM | 0.02278 → 0.02217 ms | Synchronized host wall time, not HIP events; relative error 2.28e-6. |
| Princess-Luna, same GEMM via Tile route | 0.02265 → 0.02234 ms | Same clock/oracle and identical HSACO. |

The ROCm comparison is regression evidence for existing executable paths, not
device proof of a new queue protocol. No CUDA timing or resource comparison from
the withdrawn packets is retained as compiler-change evidence.

Validation: 161 native fixtures passed (four feature-gated exclusions), including
real WarpSpecialization buffer markers and 22 positive/negative lifetime cases.
The inherited-access rule prevents role-local waits from clearing other roles.

### 2026-09-05 — Dynamic completion generations and memref arena proof

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`; **landing**.
This section supersedes the preceding increment's open direct-token, CFG-alias
and memref-planner boundaries within the following native envelope.

1. **Dynamic completion:** pending accesses carry SSA completion names. Structured
   branch results, loop initial/carry/backedge/exit edges and CFG forwarding rename
   those names. Backedges discard expired local names before a new dynamic copy
   executes; an unforwarded outstanding generation remains pending. Static
   producer reachability never releases an access. `tile.wait_async` and
   `tile.dealloc` share a declared-origin resolver with the lifetime verifier.
2. **Alias contracts:** allocation roots flow through supported structured and CFG
   identity edges. The memref planner follows cast and view interfaces transitively,
   retaining the entire backing allocation for unknown/subview offsets. Unknown
   users and non-linear control flow prevent reuse. Unannotated external Tile
   buffer ownership remains unproven; this does not add interprocedural noalias.
3. **Physical consumer:** `TileBufferReusePass` and `TileBufferArenaPass` use one
   native memref lifetime service. Missing/wrong waits retain in-flight copies
   through exit; stage and barrier keys both match. Synchronous loads/stores
   require a thread rendezvous before storage reassignment. The arena consumer
   rejects forged overlapping groups and pre-existing/escaping/nonidentity aliases,
   then propagates workgroup address space through supported view/cast chains.
   Static byte-size and arena-offset arithmetic are checked before materialization.

Native fixtures cover a legal rolling token pipeline, dropped generations,
branch-result and CFG token forwarding, opaque origins, transitive alias uses,
missing/wrong/conditional waits, physical arena alias types and forged reuse.
The real arena consumer is exercised; parser acceptance alone is not the evidence.

**Remaining:** path-sensitive memref group coalescing across branches/loops;
interprocedural ownership/escape summaries; offset-disjoint subview optimization;
physical ring slot/phase protocols and architecture-owned performance promotion.
The existing native GEMM/attention comparison remains a regression workload, not
measurement of a new ring schedule. Each backend's queue records its own device
boundary and follow-up; no cross-architecture performance transfer.


Validation for this increment: **167 native fixtures passed** (four feature-gated
exclusions), **252 focused unit/registry/audit tests passed**, Ruff passed and
mypy remained at zero errors. The ROCm owning-host packet
`benchmarks/baselines/token_memref_rocm.json` remains valid.
`benchmarks/baselines/token_memref_nvidia.json` is withdrawn for the same native
compiler-selection defect as the earlier NVIDIA comparison. No CUDA before/after
image or timing claim survives from those two records. The separate synchronous
ring and asynchronous GEMM experiments below are unaffected.

### 2026-09-05 — Structured-path coalescing, private borrowing and device ring experiment

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`; **landing**.

**Native compiler increment.** The shared memref lifetime service now proves
completion on every `scf.if` path, coalesces opposite arms only with a derived
workgroup-uniform predicate, and permits loop-local reuse only when accesses
complete before the backedge. Uniformity follows registered constants, GPU
block/grid dimensions and deterministic arithmetic; thread IDs and opaque
arguments do not establish it. The planner checks every member of a reuse group,
and the physical arena consumer independently uses the same proof.

Private direct calls can borrow memrefs when their bodies and transitive callees
prove no escape, free, returned alias or outstanding asynchronous access. External
symbols, recursion and opaque consumers remain unproven. The arena admits these
calls only with an existing workgroup-space ABI; it does not change a helper's
signature behind other callers. Native positive and negative fixtures exercise
both planning and physical `memref.view` materialization.

**Measured ring protocol.** `benchmarks/record_device_ring_protocol.py` emits a
native MLIR GPU kernel and lowers it through LLVM to NVPTX or AMDGPU. Direct
streaming and depths 1/2/4 implement identical neighbor-lane math. Shared slots
carry full generation tags, a publish barrier and a release barrier. Fill/drain
and wrap tests cover short and uneven trip counts through 257 rounds; an
intentionally stale generation must fail the output oracle. This is a standalone
cooperative protocol experiment, not a product Tile ring producer or proof of
asynchronous transfer/compute overlap.

Five unprofiled trials, 20 repetitions, 64 workgroups of 128 threads and 257
rounds produced the following medians. The fixed small grid is not an occupancy
saturation study. Each backend was compiled and measured on its own host.

| Backend / clock | Direct | Depth 1 | Depth 2 | Depth 4 |
|---|---:|---:|---:|---:|
| RTX 5070 / CUDA events | 0.037504 ms | 0.048861 ms | 0.049038 ms | 0.049131 ms |
| gfx1151 / HIP events | 0.078530 ms | 0.132742 ms | 0.120325 ms | 0.129554 ms |

Packets: `benchmarks/baselines/ring_protocol_{nvidia,rocm}.json`. Both numerical
and stale-generation negative controls passed. CUDA registers were 26 for direct
and 16 for the rings; API-reported active-block capacity stayed at 12 per SM.
ROCm reported 11/10/13/11 registers and capacity 16 blocks per compute unit.
Shared storage was 0/520/1040/2080 bytes; no local-memory allocation was reported.
API capacity is theoretical, not measured active occupancy.

**CUDA profiler evidence.** Isolated Nsight Compute direct and depth-2 captures
are exported as `ring_protocol_nvidia_ncu_{direct,depth2}.csv` in the same baseline
directory. The ring adds barrier stalls (2.017 stalled-barrier warps per issue
active versus zero); achieved active occupancy stays about 11.1% on this grid.
Profiler replay times are not the unprofiled event medians above. Nsight Systems
exports `ring_protocol_nvidia_nsys_{kernels,api}.csv`: the ring kernel median is
48.048 microseconds across four launches; context creation dominates the short
process API trace and must not be attributed to kernel execution. Reproduce
isolated profiling with `--profile-depth 0` or `2` after the normal run has emitted
its binaries under `--artifacts`; profile-only runs never write timing packets.

**Disposition.** Rings are correct but slower for this streaming experiment on
both devices. Do not promote a selector candidate or restore a queue dialect on
these results. Next: connect a real asynchronous producer/consumer to the proven
release protocol, then measure representative compute intensity and saturated
grids. General CFG coalescing, symbolic kernel-argument uniformity, recursive or
external ownership summaries and private-helper ABI specialization remain open.

### 2026-09-05 — Real asynchronous GEMM comparison

Owner: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2, `IR-NATIVE-FOUNDATION-1`: the next experiment now uses the
production SM120 macro-CTA F16 GEMM producer. Its next-panel `cp.async` overlaps
current-panel MMA; a benchmark-only immediate-wait control preserves the native
instruction body, two slots and launch geometry. See
[method, reproduction and evidence](../../../benchmarks/nvidia/ASYNC_PRODUCER_CONSUMER.md).

On RTX 5070, unprofiled resident CUDA-event medians improve 6.1% / 2.9% / 2.3%
for 512 / 1024 / 2048 square GEMMs. Six shapes pass the numerical oracle. Both
images have 56 registers/thread, 4 KiB static shared memory and zero spills.
Native SASS confirms MMA before the deferred wait; Nsight Compute reports fewer
long-scoreboard stalls. These are single-process observations, not promotion or
cross-run confidence evidence. Compute Sanitizer cannot initialize the WSL WDDM
debugger interface; synchronization sanitizer proof remains open.

This closes the useful-compute asynchronous benchmark gap on NVIDIA, but does
not connect the generic allocation/release-token pass to the macro kernel.
That integration, sanitizer validation and independent timing repetitions remain
open. ROCm needs a target-specific async producer; Apple and x86 have no physical
schedule parity claim. The previous synchronous ring remains a distinct negative
experiment and must not be relabeled as asynchronous.

### 2026-09-05 — Symbolic kernel uniformity for memref reuse

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`; **landing**.
Reuse assignment and static arena materialization now also visit registered
`gpu.func` bodies. Scalar entry arguments of actual GPU kernels provide a launch
uniformity proof; induction values are uniform when all three `scf.for` controls
are uniform. Arithmetic may propagate that proof. Ordinary function arguments,
GPU helper arguments, thread IDs and unproven loop-carried values remain unknown.
A `gpu.kernel` attribute on `func.func` does not override this boundary.

Native `gpu.barrier` releases synchronous memref accesses but never pending DMA.
The static arena consumer creates the shared global inside the owning GPU module
and replaces descriptors with address-space-3 views. The dynamic GPU extension
below supplies the native launch-size consumer; the existing dynamic `func.func`
path keeps its region-local allocation behavior.

The native fixture checks symbolic loops, induction-dependent exclusive arms,
real memref loads/stores and GPU barriers, divergent controls, missing release,
helper and forged-kernel annotations, and dynamic storage. These uniformity
fixtures establish compiler legality/materialization; the device experiment below
separately establishes its bounded dynamic-storage workload.
The remaining symbolic boundary is loop-carried scalar and general CFG uniformity.
Generic release-token integration with the production async macro kernel,
external ownership summaries and architecture-owned execution remain open.

### 2026-09-05 — Native dynamic GPU storage and owning-device proof

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`; **landing**.
`TileBufferArena` now materializes entry-block runtime-sized GPU scratch through
`gpu.dynamic_shared_memory`, with workgroup-address-space memref views. Its exact
layout expression also produces a native `func.func` sizing companion, referenced
by `tile.dynamic_shared_size`. Local `gpu.launch_func` sites call that companion
and use its i32 byte count; an explicit count must agree. No Python or generated
CUDA/HIP expression is the authority for those launch sizes.

Admission is deliberately bounded: actual registered GPU kernels, identity-layout
scalar-element memrefs, and a size expression derived from kernel arguments or
their memref dimensions using constants/add/multiply/max/positive-constant divide.
Every host intermediate must lie in `[0, INT32_MAX]`; the host index must be
64-bit so the checked arithmetic itself cannot overflow. The driver registers
upstream DLTI so a supplied data layout is interpreted. Existing independent
dynamic shared allocation, nested markers, GPU helpers and device-local size
provenance fail closed. This does not infer a device's physical shared-memory
capacity; the native runtime still rejects launches exceeding that capacity.

The [recorder](../../../benchmarks/record_dynamic_gpu_storage.py) compiles the
emitted host companion through LLVM into a native shared library, obtains the
launch count from that function, and separately lowers the emitted GPU module
to NVPTX or ROCDL. Negative and oversized extents must abort in fresh native
processes. Both owning-device packets cover four runtime widths, cross-lane
scratch reads, publish/release barriers, 17 iterations and 256 blocks, against
exact output oracles and independently compiled static arenas. Device-event
samples remain experiment evidence, with no selector promotion or speedup claim.

**Follow-on scope (bounded implementation below):** bind the companion and kernel
fingerprints together in a production native package; admit path-dependent/nested dynamic storage only with
a host-evaluable lifetime/size envelope; establish dynamic-size reuse equivalence
before coalescing unknown-size buffers; extend symbolic loop-carried/CFG uniformity;
and integrate release tokens with an actual asynchronous producer. Apple MSL
threadgroup arguments need their own materializer/ABI, and x86 retains its host
allocation route. See the [experiment report](../../../benchmarks/DYNAMIC_GPU_STORAGE.md)
for owning-host evidence and timing limits.

### 2026-09-05 — Bound native packages, nested dynamic lifetimes and NVGPU completion

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`; **landing**.
The [native storage package API](../../../python/tessera/compiler/native_gpu_storage.py)
now builds a single immutable pair from compiler-emitted IR. The binding digest
covers the GPU image, host sizing library, entry symbols, argument ABI, target,
compiler fingerprints and arena IR. Serialization requires the caller's pinned
digest on reload; mutations are rejected before native loading. Synchronous
launches use the loaded companion and the same argument tuple as the loaded
kernel. Invalid extents return `-1` from native sizing and raise before dispatch;
`gpu.launch_func` retains a checked assertion for this failure. This supersedes
the earlier companion's process-abort behavior.

The implemented runtime boundary is a raw device-pointer/index ABI on the
caller's current CUDA/HIP context. It does not allocate tensors, infer logical
shape guards, own scheduling, select a route, or regenerate lower-level IR from
Graph. Callers retain responsibility for pointer sizes and kernel launch geometry.
The loader owns target-image compatibility; this first build envelope is sm_120
and gfx1151, with x86 native host companions. It is not automatic JIT/arbiter
integration or a new general tensor operation.

Uniform `scf.if`/`scf.for` regions now admit dynamic markers when every nested
lifetime completes before region exit/backedge. Launch-derived size arithmetic
is hoisted without hoisting payload operations. Disjoint same-type dynamic GPU
buffers can share a group whose capacity is the maximum member size; unrelated
groups retain separate offsets. The envelope reserves even untaken branch sizes
conservatively. Iteration-dependent sizes, divergent control and escaping or
incomplete lifetimes remain rejected.

Native `nvgpu.device_async_copy` tokens now participate in the allocation proof.
Only the copy's direct commit-group token, a full drain (absent/zero numGroups),
and a following GPU barrier establish completion/publication. Missing, partial
and unrelated-group waits fail closed for nested arena reuse. This admits a real
NVGPU producer through NVVM `cp.async` into runtime-sized shared storage; it does
not claim overlap with useful compute or migration of the separate macro-GEMM
schedule. Loop-carried async token forwarding remains outside this direct-group
proof.

Owning-device evidence is in the [package report](../../../benchmarks/NATIVE_GPU_STORAGE_PACKAGE.md):
serialized/reloaded nested packages pass four exact cases on gfx1151 and RTX 5070;
RTX 5070 additionally passes four async-copy cases and native protocol-negative
checks. Native sizing rejects oversized requests before GPU dispatch. No timing
or sanitizer claim is made by this increment.

**Next:** typed tensor/package descriptors and JIT/arbiter consumer wiring;
path-dependent size selection and iteration-varying bounded envelopes; async
token forwarding and integration into the production macro-GEMM schedule; and
ROCm's own physical asynchronous producer. Apple MSL dynamic threadgroup argument
binding remains separate. Keep existing compiler-comparison tombstones withdrawn.

### 2026-09-05 — Explicit tensor/JIT and native producer integration

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2, synchronization `IR-NATIVE-FOUNDATION-1`.
The [native tensor/producer report](../../../benchmarks/NATIVE_TENSOR_PRODUCERS.md)
supersedes the preceding increment's tensor and direct-token limitations:
explicit tensor/index descriptors now bind native storage packages directly to
`JitFn`, preserving frontend checks while bypassing Graph regeneration. Native
allocation extent/device checks precede dispatch; Python equivalence is explicitly
caller-declared. Automatic lowering/arbiter selection and paired AD remain open.

NVGPU completion follows unanimous branch forwarding and identity loop carries;
changing backedges remain rejected by generic lifetime analysis. SM120 production
macro-GEMM emits typed copy/group/wait tokens and lowers through NVGPU-to-NVVM;
its static two-panel ownership is not inferred by the dynamic arena proof.

Exact-device validation: four tensor/JIT cases each on RTX 5070 and gfx1151;
six macro-GEMM shapes in both deferred/immediate-wait variants on RTX 5070.
ROCm uses an ISA-supported register-prefetch producer. Immediate VMEM drains at
several emitted barriers prevent an overlap claim; direct async global-to-LDS
is gfx1250-only. Next: ROCm barrier/scheduling ablation, automatic descriptor and
arbiter production, and changing-generation lifetime proofs. No performance
policy promotion or Apple runtime parity is implied.

### 2026-09-05 — ABI manifests, paired-program consumption and streams

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; `IR-NATIVE-FOUNDATION-1`.
The [integration matrix](../../../benchmarks/NATIVE_STORAGE_INTEGRATION.md)
records implementation separately from native and device proof.

- Compiler-preserved ABI manifests now generate tensor descriptors and optional
  Tier-2 arbiter candidates. The existing numerical oracle still gates selection;
  general Schedule producers must supply their ABI manifests and operation oracle.
- An existing compiler `NativeJVPArtifact` can bind pinned storage children with
  full ABI preflight. This is a host-tested consumer, not automatic derivative
  production or device-proven AD. The production AD planner remains the next
  producer integration; reverse mode stays unsupported by this adapter.
- Caller-owned CUDA/HIP streams use event dependencies and retained allocation
  owners; four exact generated/JIT/arbiter/stream cases pass on each owning GPU.
  Conflicting submissions through one binding are ordered. External writes still
  require published producer streams; no concurrency performance claim.
- Generic token provenance now handles loop-external generation replacement
  without dropping zero-trip seeds. Fresh loop-issued and rotating generations
  remain rejected; inter-iteration coalescing is not inferred from static origins.
- ROCm has a measured immediate-wait ablation: three exact workloads, seven
  alternating HIP-event samples. The control is 2–4.5% slower in this run, but
  added instructions and scheduling changes prevent an overlap/promotion claim.
- Apple MSL uses a shared slot declaration/preflight contract for its existing
  tiled runtime ABI, including static scratch. Generic dynamic-arena materialization,
  an Apple-native companion and exact Metal validation remain open. x86 continues
  to supply host sizing; GPU results do not establish CPU kernel performance.

**Next producer work:** Schedule-authored manifests and operation-specific arbiter
oracles; storage children from real compiler AD output; rotating-generation
ownership; independent ROCm profiling; Apple generic dynamic-arena materialization.

### 2026-09-06 — Generated AD children, ownership recurrence and Apple arenas

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. The
[follow-up report](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md) records the
bounded implementation and owning-host evidence, superseding the corresponding
blanket gaps in the preceding section.

- The actual C++ forward-AD SSA product can generate a native storage child for
  equal-shape rank-one f32 add/mul programs. Four exact primal/tangent cases pass
  on each GPU owner. General family/Schedule integration and reverse AD remain open.
- A fixed allocation's seed/iteration/final-drain token recurrence proves reuse
  after changing generations. Native adversarial tests reject stale or missing
  completion; five CUDA cases include zero trips and generation-varying input.
  Dynamic slot selection, permuted aliases and general CFG ownership remain open.
- The native arena pass emits bounded MSL with a dynamic threadgroup argument and
  a checked, 16-byte-aligned native sizing companion. Six M1 Max cases pass using
  native arm64 sizing. Production Metal package/JIT binding and broader operations
  remain open; this is a materialization/device probe, not route promotion.
- Five independent ROCm processes now compare an instruction-identical no-wait
  control: 348 instructions, only five wait thresholds differ, matching resources.
  Larger workloads favor deferred waits in 5/5 runs; the small case remains mixed.
  The WSL profiler exposes no PMC metrics/device trace, so stall attribution remains
  open. No selector threshold or incumbent policy changed.

### 2026-09-06 — Expanded AD and dynamic aliases / resident Apple packages

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. The expanded section of the
[follow-up report](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md) supersedes the
corresponding gaps above:

- AD adds subtraction, stop-gradient and compiler-owned splat constants. Forward
  tangent mapping now respects requested `wrt` order. Sixteen exact cases pass on
  each CUDA/ROCm owner. Transcendental, reduction/attention and reverse families
  still require family-owned lowering and validation.
- Dynamic memref aliases support a bijective two-slot swap with full copy
  completion/publication and collective release before every backedge. Seven
  CUDA cases cover zero/odd/even trips. Pending tokens across swaps, N-slot rings,
  nested permutations and general CFG ownership remain open.
- Apple has an immutable shader/native-companion package and bounded synchronous
  binding to resident Metal tensors; six M1 Max cases pass after serialization.
  This is an explicit raw ABI with caller-owned extents and geometry. Typed JIT
  descriptors, arbiter registration and cross-binding stream ownership remain open.
- ROCm hardware-counter attribution is blocked by the installed gfx1151 WSL
  profiler: no PMC/SPM counters or PC-sampling agents. Structured capability
  evidence distinguishes unavailable counters from zero-valued observations.

### 2026-09-06 — Nonlinear forward products, pending swaps and typed Apple JIT

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. The newest section of the
[follow-up report](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md) records:

- Native sigmoid/tanh/composed forward children, using overflow-safe tails and
  cancellation-safe small-input tanh. Nine numerical cases pass on each CUDA/ROCm
  owner, including signed zero and nonfinite inputs. Derivatives stay in C++ AD.
- Coupled pending-token/two-slot recurrences, with exact seed/backedge/final-drain
  checks. Prefetch may precede current-slot consumption. Seven CUDA cases cover
  zero/odd/even trips; no speedup is claimed from correctness evidence.
- Shared manifest validation with an Apple-specific resident tensor adapter and
  explicit JIT binding. Six M1 Max cases cover keyword calls, lazy rebinding and
  invalid shapes/scalars. Requested AD fails closed until Apple has a paired ABI.

Next: reduction/attention and reverse AD families; generalized N-slot/nested
recurrences; Apple paired AD and cross-queue ownership; supported ROCm counter
capture tied to actual dispatch images. Automatic family/arbiter selection still
requires semantic oracles. x86 retains host-companion evidence only, without
inherited GPU compute/AD proof.

### 2026-09-06 — N-slot ownership, reduction/reverse pairs and Metal queues

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. This section supersedes the
bounded status in the preceding loop; [device packets and scope](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md)
remain the evidence authority.

- Released bijective N-slot rings and uniformly nested two-slot pending ownership
  now participate in reuse and arena revalidation. CUDA proves 3/4/8-slot rings
  and nested pending swaps, including zero/odd/even inner trips.
- Compiler-generated sum/mean reduction JVPs and residual-free single-input
  elementwise reverse pairs share a typed, serialized primal/derivative ABI.
  CUDA, ROCm and Metal each pass 15 cases. Reverse inputs include an explicit
  output cotangent; the compiler's backward SSA owns the derivative.
- Apple explicit JIT binding accepts paired packages. Fresh retained shared-event
  fences order producer blits before consumer AD on separate queues; 12 Metal
  generations pass. This is correctness evidence, with no selector promotion or
  overlap/performance claim.
- Frontend scalar tensors retain rank zero; traced and AST reduce(op=...) emit
  the canonical kind attribute. This closes actual native parse failures.

Next ordered contracts:
1. General N-slot *pending* ownership: a bijection alone is insufficient. Track
   token-to-allocation generation mappings through every nested backedge, seed,
   zero-trip path and final drain; reject incomplete or stale mappings.
2. Reduction VJP: separate primal and cotangent shapes in the physical ABI,
   implement scalar-to-vector broadcast from compiler backward SSA, and prove
   both output extents. Current reverse native children require equal shapes.
3. Attention AD: Tessera_FlashAttnOp lacks TangentInterface and AdjointInterface.
   Start with dense deterministic Q/K/V and paired saved-LSE/mask contracts,
   then causal/bias variants. Keep dropout RNG and cache effects explicit;
   never infer support from the existing hand-written backward packages.
   Use row/tile cooperative schedules rather than extending lane scalarization
   into quadratic materialized score matrices. Validate primal, JVP and VJP
   against independent oracles on each owning backend before arbiter enrollment.
4. General reverse residual/tape ABI and automatic family/arbiter generation.
5. Cross-queue resource ownership beyond explicit Metal fences, CUDA/HIP event
   parity, and measured overlap with hardware attribution where available.
   x86 remains the host companion; no GPU execution proof transfers to it.

### 2026-09-06 — Pending rings, mixed-shape reverse and attention products

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. The
[latest follow-up evidence](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md)
supersedes the previous next-action list:

1. N-slot pending ownership is implemented for a single outstanding copy and
   a proven bijective slot/token destination mapping, including uniform nesting.
   CUDA validates 3/4/8 slots. Multiple outstanding generations and general CFG
   joins still need a set of generation-specific ownership facts.
2. Sum/mean reduction VJP and independent primal/cotangent tensor widths are
   implemented and measured for correctness on CUDA, ROCm and Metal.
3. Dense-f32 attention reverse and V-only forward interfaces are implemented
   using existing registered checkpoint operations; LSE is explicit and recomputed
   under one causal/scale policy. This is compiler-IR evidence, not device AD
   closure. Next: schedule compound products, persist forward LSE with generation
   identity, then add Q/K forward products and bias/dropout/cache policies.
4. Straight-line SAVE residuals now feed the fused native backward from forward
   SSA; multiple public primal outputs cannot masquerade as saved residuals.
   External persistent tapes and nested control-flow/state tapes remain open.
5. Five-process Metal queue measurements establish overlapping command timestamp
   intervals for matched independent work, with exact numerical results. They
   do not establish simultaneous instruction issue or a selector-worthy speedup.
   CUDA/HIP event parity and backend-specific counter attribution remain open.

Keep compiler builds separate from tests: pin the validated executable before
starting a suite, record its digest, and avoid re-linking under running tests.

### 2026-09-06 — Outstanding cohorts and saved attention LSE

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2; **IR-NATIVE-FOUNDATION-1**. This supersedes the
single-pending and recomputed-LSE boundaries above. See the
[implementation, measurements and next architecture boundaries](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

- Multiple matched pending cohorts now carry distinct generation ownership
  facts. RTX5070 validates 2/3/4 outstanding copies under uniform nesting;
  arbitrary CFG joins and head-only FIFO waits remain open.
- Paired attention forward now returns LSE and backward consumes that explicit
  residual. This closes compiler SSA persistence, not device-owned tape storage.
- CUDA/HIP matched queue experiments have exact numerical and event evidence;
  CUDA additionally has Nsight kernel intervals and an isolated counter sample.
  ROCm hardware attribution remains open, and no selection policy changes.
- Next ordered work: owned persistent tape frames and split native products;
  nested invocation/state ownership; cooperative Q/K JVP lowering with its native
  consumer; attention package/LSE generation binding; production queue and
  fresh-process hardware attribution. Do not register an unsupported attention
  product merely to advertise an interface.

MSW status rechecked: MSW-5/6/7/8 are implemented in the reference lane;
36 focused tests, both tutorials and all 15 ANN reference-spike checks pass.
MSW-9's design spike and native program-pair evaluator adapter are implemented;
fragment/parameter extraction and the fusion candidate consumer remain open.
These statuses do not imply native backend closure for the math examples.

### 2026-09-06 — Persistent snapshots and generated checkpoint packages

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

W2.4a / CAKE / SO-2 / MSW-9; **IR-NATIVE-FOUNDATION-1**.

- CUDA/HIP reverse packages now expose `NativeStoragePair.capture`: owned
  snapshots survive source mutation, repeated backward results have distinct
  allocations, and nested invocation frames release with their parent. Twelve
  cases per GPU validate square/tanh/sum/mean. This is synchronous recomputation
  from snapshots, not higher-order AD or a lowering of nested control-flow tapes.
- Paired AD can export an isolated forward/backward attention checkpoint with
  canonical physical argument order and full paired-IR provenance. It feeds the
  existing native Schedule/Tile and NVIDIA package paths without Graph rebuilding.
  Six RTX5070 cases validate forward, LSE and Q/K/V VJP for causal/noncausal and
  unequal-length inputs. The host-buffer bridge persists LSE between calls;
  resident attention tape ownership and Q/K JVP remain open.
- MSW-9 has bounded immutable fragment extraction and an explicit composition
  candidate gate. Automatic fusion discovery, executable-to-fragment identity
  and backend promotion remain open; reference evidence never supplies native
  provenance.
- Incoming native Ubuntu26.04 RX9070XT profiling uses **gfx1201**, not gfx1200.
  The [commissioning plan](../backend/rocm/NATIVE_RDNA4_COMMISSIONING.md) includes
  a read-only probe, assertions-enabled compiler build, rocprofv3 and Systems
  Profiler capture. Hardware/counter validation awaits the actual host.

Next: split native forward/backward packages to avoid recomputation; integrate
saved-state control-flow frames; implement cooperative attention Q/K JVP with a
registered native consumer; bind resident LSE generation ownership; connect ANN
inventories to automatic fusion candidates; commission gfx1201 counters.

Q/K JVP algorithm spike: `tools/attention_jvp_spike.py` maintains rescaled
normalization/value/directional-score moments per row. Seventeen reference
cases cover Q-only, K-only, combined directions, block-size variation, extreme
logits and empty masks. It avoids full score materialization. Native tangent
registration is deliberately unchanged until the cooperative consumer exists.
The saved-LSE variant should consume the same generation's output and LSE,
compute `sum(P*dS*V + P*dV) - O*sum(P*dS)`, and share the canonical mask policy.

### 2026-09-06 — native ANN composition and resident attention generation

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner: W2.4a / CAKE / SO-2; MSW-9. Sync: **IR-NATIVE-FOUNDATION-1**.

Implemented this increment:

- `AttentionCheckpointPair.capture(q,k,v)` binds the existing native forward and
  backward images directly to CUDA allocations. The frame copies Q/K/V into
  private device storage and retains the exact forward-produced natural-log LSE.
  Repeated backward calls use that generation and return independently owned
  results. Dtype, strides, full allocation bounds, CUDA context, image/descriptor
  ABI, shape guards and pair policy are checked. Close invalidates views and
  unloads images; failed backward allocations do not reclaim earlier results.
  Six RTX 5070 cases cover both causal modes, unequal sequence lengths, caller
  input mutation and repeated backward. This is synchronous correctness evidence,
  not stream overlap, kernel speedup, or sibling-backend execution proof.
- `tessera-canonicalize=ann-reassociate=true` now automatically recognizes a
  single-use two-affine chain in native MLIR and folds constant weights and
  constant matrix biases with deterministic fp32 arithmetic. No replacement
  GraphIRModule or hand-written backend source is generated. Runtime parameters,
  intervening activations, transposition, policy overrides, nonfinite folding and
  excessive folding work refuse. The option defaults off because association
  changes rounding. This establishes a native MSW-9 fusion consumer for frozen
  inference constants, not automatic JIT/arbiter promotion or trainable fusion.

The next architecture work remains open, with these concrete boundaries:

1. **General nested control-flow tapes:** replace NativeStorageJVP's hard-coded
   two-output scalarizer with a split forward/backward product ABI. The ABI must
   carry every residual's dtype, full shape, ownership and indexing; nested loop
   state requires an iteration path, valid extent and branch selection, not just
   a stack of invocation frames. Preserve zero-trip and untaken-region behavior.
   Lower compiler-produced tensor tape reads/writes before accepting a new family;
   never infer a backward from an independently replayed control path. Establish
   allocation-size overflow checks and fail-before-launch capacity checks.
2. **Native Q/K JVP:** retain the paired O/LSE generation and lower
   `sum(P*dS*V + P*dV) - O*sum(P*dS)`, with
   `dS = scale*(dQ*K + Q*dK)` and `P = exp(S-LSE)`. The row consumer must use the
   exact `end_aligned_v1` mask and grouped-head mapping of forward. A registered
   tangent producer must ship with a real bounded-storage native consumer and
   exact-device finite-difference cases. The streaming Python spike remains a
   numerical oracle; V-only is still the public native tangent interface.
3. **Resident concurrency and siblings:** use completion-owned allocation
   generations before adding stream submission; private allocations alone do
   not establish concurrent lifetime safety. Add HIP/Metal ownership on those
   hosts independently. x86 needs its own host tape execution path.
4. **ANN promotion:** connect frozen parameter materialization and explicit
   reassociation policy to pipeline selection, then require native original/fused
   program-pair comparison and performance evidence before arbiter promotion.
   The earlier Python fragment view is an inventory, not executable identity.

Evidence: [loop8 implementation and validation](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

### 2026-09-06 — typed nested exports and resident score tangents

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners **W2.4a / CAKE / SO-2**, **MSW-9**; synchronization key
**IR-NATIVE-FOUNDATION-1**. This section supersedes the loop8 next-step status.

- `tessera-autodiff-paired=export-product=forward|backward` exports native
  products with a compiler-produced, full-type residual ABI and common paired
  lineage. Saved if/while products and nested SAVE loops retain native regions.
  Nested pullbacks now receive their cloned region residuals; statically empty
  positive-step loops are removed before tape sizing. This closes the export
  boundary, not general device tape lowering: nested inner residuals may be
  reconstructed from the saved outer state. General persistent device tapes
  still need allocation ownership, tensor tape lowering, dynamic iteration-path
  indexing, capacity checks and backend execution proof.
- Explicit CUDA resident `prepare_jvp`/`jvp` products now implement Q, K and V
  directions through native GPU MLIR, shared-memory reduction and the saved
  forward O/LSE generation. The bounded consumer uses grouped heads and the
  same end-aligned causal mask. The automatic FlashAttnOp TangentInterface is
  still V-only; wiring this consumer into automatic AD remains open. This
  consumer recomputes scores for each output column; it is correctness evidence,
  not a hand-tuned performance candidate.
- Resident backward launch sizing now covers the concatenated dQ/dK/dV range.
  The 129-key case exposed the old maximum-of-ranges grid under-launch.
- Arbiter exact-cache reuse and incumbent retention now require admissible
  selection evidence, including explicit eligibility and timing separation.
  Malformed or ineligible cached records trigger a fresh comparison. This is a
  shared admission prerequisite; no ANN candidate was promoted. MSW-9 still
  needs frozen-parameter native candidate registration, policy-bound identity,
  original/fused backend comparison and measured selection evidence.

See [loop9 evidence and backend boundaries](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

### 2026-09-06 — control-flow contract correction and automatic score JVP

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners W4 / W2.4a / CAKE / SO-2; sync **IR-NATIVE-FOUNDATION-1**.
`docs/spec/CONTROL_FLOW_CONTRACT.md` now distinguishes frontend acceptance,
portable control-to-SCF lowering, native packaging and owning-device proof.
The previous Apple-only sentence contradicted ROCm support; the table also
omitted cooperative ROCm paths, bounded native x86 state-machine proof and narrow handwritten CUDA control helpers, and confused incomplete direct CUDA graph packaging
with absence of native CUDA control flow. Historical CF0–CF4 narrative is
replaced by the active ownership links; this changes no backend execution state.

The FlashAttnOp TangentInterface now emits a registered internal
`checkpoint_jvp` for Q/K directions, with a same-producer O/LSE verifier and
explicit inactive tangent slots. `export-attention-jvp` accepts one isolated
attention function with direct argument/return mapping. The resident CUDA
consumer lowers this native contract and binds its product digest into package
identity. Eight CUDA cases / 32 directions pass analytic and finite-difference
oracles. Generic JIT compositions and sibling backend consumers remain open.

Persistent snapshot allocations now check byte overflow before allocation;
partial backward allocation failure rolls back only the attempted outputs.
This closes a lifetime defect, not general persistent nested tensor tapes.
The next tape implementation must replace NativeStorageJVP's combined
forward/recomputed-backward scalarizer with two independently callable native
products: bufferize full typed residual outputs, preserve iteration-path and
valid-extent indexing, then retain those allocations until all backward users
complete. Start with static nested SAVE loops; reject dynamic capacities until
native size derivation and fail-before-launch checks exist. A host-only arena
or another residual inventory would not close this item.

Evidence: [loop10](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

### 2026-09-06 — split persistent tensor tapes and JIT-owned Q/K programs

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners **W4 / W2.4a / CAKE / SO-2**; sync **IR-NATIVE-FOUNDATION-1**.
The first physical split-tape consumer now runs fresh native forward and backward
products independently. Upstream one-shot bufferization preserves full f32 tensor
residual shapes; readonly product inputs prevent in-place mutation of retained
snapshots. The new `tessera-native-tape-to-gpu` pass materializes bounded static
for/if bodies with per-iteration-path temporary slots. Backend-owned allocation
addressing is explicit: generic allocas for NVVM and private address space 5 for
AMDGPU. Repeated backward calls consume retained forward residuals and produce
independent outputs. The owning frame retains all allocations until explicit
close. These synchronous, serial GPU entries establish correctness, not a
parallel schedule or performance improvement.

The public JIT object now exposes `compile_persistent_device_tape` for reverse
requests and `compile_native_attention_jvp` for isolated SM120 forward attention
requests. The latter traces the function, runs native AD, binds the same resident
forward O/LSE generation and accepts tangent arguments in `wrt` order. Dense
three-tensor, dropout-free attention now emits its required head dimension and
precise pure effect; cache, dropout and unrecognized forms remain conservative.

Remaining: dynamic/mixed-type residual capacities (including saved predicates),
persistent while/general control tapes, concurrent backward retirement, parallel
physical tape schedules, composed automatic attention AD and sibling native
attention consumers. Static f32 slots are limited to 1024 elements each and
logical temporary storage to 4096 bytes. Native for/if acceptance does not imply
that every frontend control-flow or saved-state form fits this envelope.
See [loop11 evidence](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

### 2026-09-06 — domain and autodiff documentation consolidation

Owner: [AD-HIGHER-1](INTEGRATED_COMPILER_PLAN.md#ad-higher-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

The scoped [AD execution plan](AUTODIFF_EXECUTION_PLAN.md) now owns remaining
P0–P6/A–B/D/AD-NEXTGEN acceptance gates. The three prior documents are archived
as superseded designs, with small routing files preserving source links. Their
uncompleted work is transferred, not marked complete. W4/W5.1/W6 sequencing and
IR-NATIVE-FOUNDATION-1 remain the active owners.

The [domain audit](../domain/DOMAIN_AUDIT.md) and
[GA/EBM review](../domain/GA_EBM_ARCHITECTURE_REVIEW.md) correct stale manifold,
grade-consumer and CPU-performance claims. The next domain sequence is W3.6
batched GA/native rotor consumption; W3.5/W4 typed energy/gradient programs;
W5.1/W2.4a resident state and measured policy; W6.4 native finite-algebra lowering.
No independent domain emitter or rematerialization stack is commissioned.

[Target IR review](TARGET_IR_REVIEW.md) now recognizes the x86 dialect, semantic
string constraints and mixed native/compatibility ownership. Standalone loose
matrix contracts and spelling-only smoke tests remain real scoped gaps.
`CORE_SUBSTRATE_VIEW.md` was reviewed without modification; the
[compiler audit](COMPILER_AUDIT.md#2026-09-06-substrate-review-and-ad-consolidation)
records its outdated status/ownership passages. This update changes no target
execution status and adds no new device/performance claim.

### 2026-09-06 — Functional-analysis contracts — consolidated ownership

Owner: [RIEMANNIAN-OT](INTEGRATED_COMPILER_PLAN.md#riemannian-ot)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Updated 2026-09-06. This section replaces the independent FA-1–FA-7 execution
sequence. The [archived design](archive/FUNCTIONAL_ANALYSIS_TSOL_PLAN.md)
retains the mathematics and original IDs. These are remaining tasks, not new
support claims. Numerical-policy transport is an existing foundation;
composable error bounds are an additional analysis obligation.

| Item and owner | Remaining work and dependencies | Acceptance gate |
|---|---|---|
| **FA-1 — numerical legality / evaluator and arbiter** | First deliver one bounded reduction or fused-epilogue consumer. Extend NUMPOL-CARRIER-1 semantics with explicit norm, admissible domain, shape/reduction length and absolute error budget; preserve analysis facts across native MLIR boundaries. No independent Python production lowering stack or unused coverage axis. | A real candidate is accepted/refused using the budget. Composition checks perturbed intermediate-domain containment and uses justified downstream Lipschitz constants. Reduction bounds enforce their accumulation-algorithm, Ku < 1 and under/overflow assumptions. Unknown bounds fail closed for budget-based promotion. Oracle evidence, analytic bounds and device measurements stay distinct; norm-to-tolerance conversions are explicit. |
| **FA-2 — AD-LAW / AD-CLOSEOUT-1** | Reuse the implemented adjoint and canonical-forward laws; carry only public debug adapter, norm-aware tolerance and derivative-coverage evidence integration into AUTODIFF_EXECUTION_PLAN.md. Does not wait on invention of FA-1 or a new harness. | Planted incorrect and matched-incorrect derivative pairs remain detected; coverage changes cite the actual law result and evidence tier. Public adapter tests exercise the existing engine. Amend the coverage contract explicitly before tightening automatic status transitions. |
| **FA-3 — spectral family / numerical legality** | Audit existing spectral laws, then fill normalization, window/domain and multiplier-bound gaps. Error-budget consumption depends on FA-1; independent oracle coverage does not. | FFT normalization and adjoints agree; STFT/ISTFT round trips declare window/overlap assumptions. A spectral transformation consumes a justified multiplier bound, with negative legality cases and native-boundary preservation. Existing adjoint tests alone do not close this consumer. |
| **FA-4 — sequence-mixer stability** | Sequence-mixer plan owns recurrence/discretization-specific certificates and eventual op wiring. No new generic certificate registry without a recurrence consumer. | Domain and timestep assumptions are checked; nonnormal transient amplification and discretization-specific stability are covered. Reference oracle and native-device execution remain separate evidence. |
| **FA-5 — functional-calculus admission, deferred** | Require a named workload and a specific missing operation before reopening broad admission. Reuse spectral/solver owners; not a prerequisite for FA-1/2/3/6. | An admission proposal names semantics, domain, derivative rules, native producer/consumer and measurable benefit. An abstract common interface is insufficient. |
| **FA-6 — approximation legality / arbiter, consumer-gated** | After FA-1, attach norm-specific truncation bounds to an actual low-rank substitution candidate. Preserve the distinction between exact factorization and approximation. | Candidate selection respects the caller's budget and composition domain; over-budget substitutions reject. Sampled estimates cannot masquerade as certified upper bounds, and no automatic tolerance relaxation is allowed. |
| **FA-7 — PDE/forms, deferred** | Reopen under PDE_STENCIL_CAPABILITY_PLAN.md only when a solver or rewrite needs coercivity. | Concrete discrete operator, boundary/domain hypotheses and a certificate-consuming solver or transformation. |

Order: FA-1's bounded consumer first; FA-3/FA-6 numerical promotion builds on
it. Existing AD and spectral law improvements proceed independently. FA-4 is
sequenced by its domain owner; FA-5/FA-7 are explicitly deferred. Introduce
registry/schema changes together with their first consumer and focused drift
gates. CUDA, ROCm, Apple and x86 each require their own lowering-preservation
and exact-device evidence before hardware promotion; this consolidation changes
no backend support state or physical schedule.

### 2026-09-06 — F0 census correction and current next steps

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: #721, #732

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

This update supersedes older summary counts, not their historical evidence.
The frontend-authority dashboard currently has nine missing NVIDIA family rows
and Apple normalization; older eight-row paragraphs are historical. Future
status reports should read the generated rows rather than duplicate this total.
A declared certification path is not an emitted exact-device certificate.

The package census now includes Apple GPU alongside Apple CPU and the three
other native package modules. Unannotated, `Any` and raw-IR inputs no longer
count automatically as compiled artifact consumers. Computed classifier returns
are exposed explicitly; Apple primitive membership remains unclassified rather
than being inferred from incomplete literals. Typed artifact inputs still need
semantic consumer/replay verification before migration closure. Missing source
modules fail the census instead of shrinking its denominator.

F0 remains landing: resolve computed Apple family membership from the live
producer, make family-to-compiled admission target/envelope aware, then join
actual driver/plugin call paths and artifact lineage. This bounded census fix
adds no backend execution support. After the relevant family's census is sound,
F2 migrates its remaining Graph constructor and F3 instantiates one optimized
matmul recipe for two witnessed buckets. PR #721's pre-bucket optimization and
narrow source-loop recognition are implemented foundations, not unstarted work.

Preserve PR #732's bounded tapes, resident Q/K JVP and ANN extraction/native
constant composition; their generalization and measured promotion remain open.
The recomputed-tape scratch-retirement follow-up is implemented with repeated
backward and failure-path allocation regressions. The active AD plan and FA-series
consolidation own the remaining native product and numerical-legality work.

### 2026-09-06 — F0 Apple family domains and native unary ancestry

Owner: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

The census resolves the known Apple CPU computed return from the producer's
value/low-precision symbol tables and Apple GPU `value_*` returns from the
ready-descriptor table minus dedicated-family exclusions. Unknown expressions
remain unresolved. These are family-domain candidates, not proof that every
shape/policy is admitted. Apple primitive-membership joins remain explicitly
unverified; resolving names does not authorize coverage promotion.

A family-to-scheduled mapping now requires the corresponding consumer to exist
in that target's package module. In particular, Apple CPU matmul does not inherit
Apple GPU or x86 scheduled coverage. This is a necessary structural gate;
full driver/plugin control-flow and envelope joins remain F0 work.

Apple scheduled softmax/reduction now replays the retained Schedule through
native `--tessera-schedule-to-tile`, requires exact equality with the supplied
Tile product, and checks the native module's apple7/apple_gpu identity before
runtime lookup. A changed operand pair or relabelled x86 parent is refused even
when the previous structural validator accepts it. Direct packaging therefore
requires the production `tessera-opt`; synthetic descriptor tests explicitly
stub replay and do not count as compiler proof. Native positive/negative replay
runs on the WSL compiler host, not as new Metal execution evidence.

Next: verify descriptor-field projection and extend the same ancestry obligation
to remaining Apple matmul/attention consumers before retiring their Graph
constructors. Reuse each native parent and preserve library-vs-generated-body
provenance. No other backend's runtime, capability or promotion state changes.

### 2026-09-06 — Descriptor projection and seven-program continuation

Owner: [LAYOUT-ALG-1](INTEGRATED_COMPILER_PLAN.md#layout-alg-1)

PRs: #732

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owning key: **F0 / IR-NATIVE-FOUNDATION-1**. Apple GPU unary and plain
rank-two matmul packages now check descriptor geometry, tensor types, supported
layout/arithmetic policy and tile decisions against native-printed Schedule IR.
Schedule-to-Tile replay now also gates matmul and forward/backward attention
before runtime lookup. Unary buffer names remain explicit host binding aliases;
they are not claimed as serialized SSA provenance. The projection reader accepts
the current bounded native printer grammar and rejects unsupported forms.

This closes a bounded package-boundary defect, not canonical ownership as a
whole. Attention descriptor projection, dynamic/epilogue matmul projection,
complete shape-policy route joins and remaining Graph constructors stay open.
Compiler replay evidence is separate from Metal execution and performance.
Princess-Luna's LLVM 23.1.1 reports assertion mode `OFF`; its passing tests do
not satisfy the assertions-enabled MLIR gate.

Continue in this dependency order, preserving each program's existing owner:

1. **Canonical ownership / descriptor projection:** finish attention and dynamic
   descriptor fields, then migrate the next Graph constructor with independent
   target ancestry and owning-device comparison. Retain library/generated-body
   provenance rather than equating a typed wrapper with compiled execution.
2. **General AD execution:** dynamic/mixed/while persistent tapes and asynchronous
   retirement, followed by composed attention, batching, sparse derivatives and
   native higher-order products. Bounded PR #732 products remain bounded.
3. **Optimization integration:** native optimized-recipe instantiation and native
   ANN evaluation before selector-grade measured promotion.
4. **Numerical legality:** carry FA-1–FA-7 error-budget composition into concrete
   spectral/approximation consumers; metadata alone cannot establish legality.
5. **Runtime resilience:** bounded CUDA/HIP waits must define context poisoning
   and retain resources until completion is known, including timeout tests.
6. **Evidence infrastructure:** obtain an assertions-enabled native compiler,
   finish route/envelope inventories and reconcile authored summaries against
   revision-bound generated evidence. Device proof remains architecture-owned.

These remain active programs; this implementation does not close their broader
execution or performance gates.

### 2026-09-07 — Attention projection and static softmax migration

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

**F0/F2 / IR-NATIVE-FOUNDATION-1:** Apple forward/backward attention packaging
now projects rank-four tensor signatures, dimensions, storage, numeric modifiers,
bias presence, LSE policy and scheduling fields from native-printed Schedule IR
after exact Schedule-to-Tile replay. Host buffer aliases remain explicit and
must be distinct. Backward's public function name is a source alias; its native
entry is synthesized and the library symbol is fixed by the verified storage ABI.
Floating attributes are compared at the native f32 precision.

Static f32 `package_softmax(GraphIRModule)` now acts as a compatibility frontend:
it invokes the shared native lowering and artifact consumer instead of building
a descriptor and placeholder Tile text from Graph fields. The f16/bf16 branch
remains historical. This is a bounded constructor migration, not deletion of
the Graph entry API or proof that every package is IR-owned. The scheduled
consumer now owns f32 provenance even when reached through the legacy entry.
Missing native compiler support fails rather than reconstructing a package.

WSL native compiler and descriptor tests validate the boundary; no new Metal
execution/performance evidence is claimed. A follow-up probe found that Apple
f16/bf16 backward's forward companion is rejected by the current native
Graph-to-Schedule pass despite the runtime variant table. That producer gap
must close before claiming low-precision paired package support. Remaining:
full operand/region semantic projection, low-precision softmax migration and
owning-device differential proof before removing historical constructors.

### 2026-09-07 — Low-precision Apple native closure slice

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

**F0/F2 / IR-NATIVE-FOUNDATION-1:** static f16/bf16 softmax now follows the same
native Schedule/Tile consumer as f32, with storage-derived ABI, alignment and
buffer types. The historical static softmax descriptor constructor is removed;
the public Graph entry remains a compatibility frontend to native lowering.
Apple low-precision attention's forward recompute companion is now admitted by
Graph-to-Schedule when its function carries the recompute checkpoint contract.
Standalone low-precision forward runtime packaging is still refused.

The new native compiler passed WSL projection tests. Exact-device differential
checks on the M1 Max passed for f16 and bf16 softmax and causal GQA backward,
including all three gradients, using a freshly compiled runtime and requiring
`native_gpu` for every launch. See
[the bounded evidence packet](../../../benchmarks/apple_gpu/lowp_native_differential.json)
and `tests/unit/test_apple_lowp_native_contract.py` for shapes, tolerances,
source/compiler/runtime hashes and reproduction assertions. Native compilation
ran on Princess-Luna over SSH; execution and numerical proof belong only to the
Mac. This is correctness evidence, not timing or selector promotion.

Remaining: biased/noncausal and broader-shape low-precision device coverage,
full semantic operand projection, and subsequent Graph-owned families. The
assertions-enabled LLVM gate remains open.

### 2026-09-07 — Broader Apple coverage and measured runtime promotion

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

**F0/F2 / APPLE-ATTN-BWD-1 / IR-NATIVE-FOUNDATION-1:** six additional M1 Max
package differential cases passed: MHA/GQA/MQA, B=1/2, Sq=7/9/17/19,
Sk=7/19/33/65, D=16/32/64/128, causal/noncausal, fp16/bf16 unbiased and fp32
biased. All dQ/dK/dV outputs were checked against a float64 analytic VJP and
every launch required native GPU placement. Low-precision compiled bias remains
explicitly refused: native IR specifies fp32 bias while the runtime variant
requires storage-typed bias. Silently quantizing it would change the program.

Five independent, paired/interleaved runtime benchmark processes (seven trials,
ten repetitions) validated all three routes on the existing six-case closure
matrix. The strict median-order-statistic gate promoted `split_reduced` over
`serial_recompute` for six exact shape/dtype keys in both timing domains.
Across the recorded per-run medians, device speedups were 5.25–22.55x and
host-input/output end-to-end speedups were 4.26–16.72x; these ranges describe
these fixtures only. Atomic was measured but was not the selected winner.

[Raw reports and coverage](../../../benchmarks/baselines/apple_backward_20260907)
and the [strict family ledger](../../../benchmarks/baselines/apple7_attention_backward_strict_v2_route_ledger.json)
retain distinct timing domains and exact context. The backward route-policy
resolver now defaults to this dedicated ledger, preserving explicit global and
per-call overrides. Live validation accepted 12 rows, refused a changed runtime
fingerprint, and retained serial when the split workspace exceeded the cap.
This promotes the measured **runtime-route policy**; compiled artifacts still
bind their declared fixed two-way split and these reports do not establish
package-subgraph performance. Biased BF16 direct-runtime evidence uses BF16
bias and is not evidence for compiled fp32-bias inputs.

Remaining: mixed-storage bias ABI/projection, package-level promotion evidence,
unmeasured shape envelopes, and assertions-enabled compiler validation.

### 2026-09-07 — Math audit / foundation reconciliation

Owner: [W6.4](INTEGRATED_COMPILER_PLAN.md#w64)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

This is a source-and-ownership review, not a fresh reproduction of the cited
papers or every historical numerical result. Math semantics and reference laws
feed one native compiler; they do not justify parallel Python lowering stacks.
Keep four evidence tiers distinct: reference law, serialized native contract,
executed target artifact, and measured candidate admission.

| Document | Disposition | Current integration / remaining gate |
|---|---|---|
| MLIR_NATIVE_FOUNDATION_SURVEY | Keep as architectural reference; this section supersedes dated census/absence claims. | F0 inventory plus F2 descriptor/ancestry migration; static Apple softmax now delegates through native IR for f32/f16/bf16. General program-body compilation remains distinct from library delegation. |
| MATH_SOURCE_WORKSTREAM | Archive the original proposal; live routing note retains MSW IDs. | MSW-1–8 bounded reference work stays landed. MSW-9 uses ANN/F3 for automatic native fragment discovery, executable identity and measured admission. Native higher-order products use the AD execution plan. |
| MATRIX_CALCULUS_REVIEW | Keep mathematical reference, label its historical findings. | MC1/2/5/8 remediation is not a new backlog. MC3 metric/manifold descent uses geometry + native AD; MC4 native Kronecker/vec rewrite uses F3 and a no-materialization gate; MC6 native higher-order composition uses AD-CLOSEOUT-1 and AD-WEIL-1; MC7/9 counting/law fixes stay landed and are reused as gates. Recheck deferred matrix operations before adding new public ops. |
| RIEMANNIAN_OT_PLAN | Keep scoped acceptance-workload plan. | Existing ga.manifold Euclidean/Sphere/SOn and metric reference methods contradict a blanket “whole layer missing” claim. R0/R1 require explicit native manifold/domain carriers and consumers; R2 reuses native AD/residual authority, R3–R5 are downstream workloads. No fresh OT numerical/backend proof is inferred. |
| SEQUENCE_MIXER_ENGINEERING_PLAN | Keep scoped implementation plan. | W1/W2 reference/state interfaces precede W3/W4 native recurrence descent, then W5/W8 target forward/backward ownership. W6 precision and W7 admission use NUMPOL/F3/FA-4. The common tiled SSD program remains distinct from ReplaySSM's bounded ABI. |
| SEQUENCE_MIXER_THEORY | Keep reference; dated capability table is not current status. | Transition/state/reassociation semantics constrain native IR. Check each transition family and current producer, not just the existence of scalar delta_rule or a runtime kernel. |
| PDE_STENCIL_CAPABILITY_PLAN | Keep scoped contract/consumer plan. | Python pde_operator classification and diagonal-diffusion certificates, coordinate fields and stencil materialization are bounded. PDE-STENCIL-FOUNDATION-1 / FA-7 require explicit spacing, coefficients, boundary/domain and discretization assumptions consumed by native passes and target code. |
| FORGE_ASSESSMENT | Archive proposal; preserve mathematical evidence and live routing note. | W1/W2 → LAYOUT-ALG-1/Schedule memory authority; W3/W4/W7/W8 → F3 stateful fusion; W5 → NUMPOL/FA-1; W6 → DIST-NATIVE-1. No new residency registry or unconditional “exact candidate” promotion. |

#### Dependency order and exit tests

1. **F0/F2 native ownership:** every selected family serializes ABI, layout,
   numeric policy, state and provenance; package fields are checked projections.
   Wrong dtype, operand ancestry, alias or policy rejects before launch.
2. **Native AD and numerical legality:** wire manifold/coordinate/recurrence
   consumers into existing AD and NUMPOL/FA-1 authority. State norm, domain,
   discretization and error-budget assumptions; unsupported cases refuse.
3. **F3 math optimizations:** native Kronecker/vec, ANN and stateful epilogues
   require original/transformed execution, alias/effect and no-materialization
   checks, then candidate identity binding. Reference identities remain oracles.
4. **Target workloads:** sequence mixers/SSD, PDE/OT and optimizer state updates
   consume that foundation. Paired package-level performance evidence is owned
   per architecture; direct-runtime measurements do not promote a compiled
   package or transfer to a sibling target.

The two archived proposals are consolidated, not completed programs. Remaining
work above stays active under existing IDs; the archive is not a second queue.

### 2026-09-07 — Mixed-storage bias and native-package admission

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

**F2 / APPLE-ATTN-BWD-1 / IR-NATIVE-FOUNDATION-1:** dedicated f16/bf16 + fp32-bias
status ABIs now preserve the native bias tensor without narrowing. MSL bias
loads and buffer allocation widths are independent of Q/K/V/dO storage; existing
storage-typed ABI signatures are unchanged. Python binding registration, native
launcher dispatch, descriptor types/alignment and non-Metal stubs agree.
Eight broader M1 Max package VJP cases passed, including fp16/bf16 biased cases.
This supersedes the earlier mixed-storage refusal as an implementation gap.

Five independent paired reports compare complete native-package dispatch with
the identical direct split-kernel ABI, excluding compilation from warm timings
and reporting it separately. The strict `package_subgraph` ledger retains the
direct incumbent: f16 package calls were slower; bf16 evidence was mixed and
failed its lower-bound gate. No package promotion is made. These are native
image/LaunchDescriptor packages, not `.mtlpackage` ML execution, and no device
clock claim is derived from the complete-call timer. See
[package reports and ledger](../../../benchmarks/baselines/apple_native_backward_package_20260907).
The instability reason no longer incorrectly says a mixed-win candidate won
every run; the admission thresholds themselves are unchanged.

Remaining: reduce measured package overhead with identity-preserving caching,
then remeasure before admission; extend mixed-bias shape/policy coverage. The
new runtime changes the exact fingerprint. Five fresh owning-device runtime
reports admit split_reduced for six shapes in both timing domains; live loading,
context mismatch refusal and workspace fallback passed. See
[refreshed runtime evidence](../../../benchmarks/baselines/apple_backward_mixed_runtime_20260907).
Fresh M1 Max fleet measurement is now sealed against committed source
`80504c8384e61f157a5fb7e772f3d6b2c22da56c`: matmul and softmax prove Metal
placement at fixture and timing shapes, with device-event and end-to-end rows.
The initial 15-sample/50-iteration attempt failed the unchanged 4% stability
gate; 21 samples with 200 amortized iterations passed. This is a fresh
measurement, not a fingerprint-only update. See
[the sealed fleet packet](../evidence/e2e_spine/apple_gpu/apple7/manifest.json).

### 2026-09-07 — PR #733 contract corrections

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

F2 / APPLE-ATTN-BWD-1: the initial low-precision softmax placement claim used
void ABIs and could not distinguish CPU fallback. Native packages now require
v2 status ABIs that return success only for Metal; failed or missing status
refuses. Ten fresh M1 Max differential cases passed using this boundary.
Low-precision attention recompute is admitted only with reciprocal primal/VJP
symbol links and verified matching types, zero/explicit bias and numerical
policy. A checkpoint attribute alone is insufficient. Shared graph producers
emit the relationship; standalone low-precision Apple forward remains refused.
The Apple tests use the central hardware capability inventory. Five fresh runtime reports retain the six-key admission in both timing domains.
The fresh fleet packet is sealed against e73cdfcee39c8265cdc329b407b05c834c5a5a70;
matmul and softmax passed placement and timing checks at the unchanged 4% gate.

### 2026-09-07 — Capability-plan reconciliation

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

This source/evidence reconciliation retains existing IDs and creates no parallel
compiler queue. It follows the F0–F4 foundation order above. Reference laws,
serialized native contracts, exact-target execution and measured admission
remain separate. The review's 111 passing focused reference/evidence tests
(one skipped) do not renew device measurements or external-paper surveys.

| Scoped document / residual | Existing owner | Required implementation and exit gate |
|---|---|---|
| CORE_SUBSTRATE_VIEW S1–S9 | W2.4a / W5.2 / F0–F4 / NUMPOL / LAYOUT-ALG-1 / AD | Keep one map to actual consumers. Retire the old blanket absence/unowned claims; derive alias/effect/lifetime facts and preserve serialized policy/provenance. The archived P0–P5 sequence is historical. |
| DIFFERENTIABLE_PROGRAMMING C1–C6/T1–T3/R1–R2 | AUTODIFF_EXECUTION_PLAN / AD-CLOSEOUT-1 / AD-HIGHER / AD-WEIL / NUMPOL | Preserve linear-transpose, effect, reference-loss and bounded solver foundations. Extend native products, checkpoint-plan execution, constrained/matrix-free solves and named estimators with appropriate laws, certificates and target evidence. These labels are provenance aliases, not new work IDs. |
| BLOCK-ATTNRES-1 phases 6–7 and broader envelopes | BLOCK-ATTNRES-ROCM-2026-08-12 / F2/F3 / W2.4a / AD / W5.2 | Consume the existing depth-statistics/merge contract. Prove native query hoisting and block-state retention/recompute legality; broaden storage/shapes with explicit numeric policy. Sibling packages need independent ancestry and device correctness; selector admission needs valid target clocks and matched baselines. TP uses DIST-NATIVE-1. |
| EGGROLL W2 reverse/breadth | EGGROLL-ES-LOWRANK-2026-08-09 / F2/F4 / AD / NUMPOL | Use shared linear transposition for the fixed-key correction, preserving member/RNG identity. Rank>1, quantized accumulation and other targets need explicit contracts and native comparison; rank-1 fp32 x86/gfx1151 correctness is already recorded. |
| EGGROLL W3/W4 | F3 / W5.2 / DIST-NATIVE-1 | Preserve optimizer-state order, aliases and numeric policy; prove no forbidden materialization, then measure the complete update. Mock scalar-gather reconstruction does not close native transport or multi-rank performance. |
| GAME G1b/G5 | REF-TIER-PHYS-2026-08-16 / LAYOUT-ALG-1 / F2/F3 | Preserve the shared coalition carrier. Replace FFT-only tiling/sharding only after layout and FFT bit-identity gates pass; require per-target package evidence. Coalition kernels do not imply equilibrium-solver support. |
| GAME G2–G4/G6 | AD execution / W4 / F4 / DIST-NATIVE-1 | Build a named certified solver or game workload using existing segmented reduction, scans and explicit RNG. Check constrained derivative assumptions, game-oracle correctness and real batching. Distributed/sampled variants require their own transport or estimator certificates. |

Workload selection is downstream of the required foundation slice, not a reason
to recreate it. Keep Block AttnRes, EGGROLL and game theory as scoped landing
plans; keep substrate and differentiable-programming documents as references.
Historical status/sequence excerpts live in `archive/` and carry no active queue.
The four backend plans retain architecture-specific validation obligations.

### 2026-09-07 — Status, native GELU and recipe instantiation

Owner: [DIST-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#dist-native-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

**F0/F2/F3 / IR-NATIVE-FOUNDATION-1:** this increment closes the native status
boundary for descriptor f32 softmax and f32/f16/bf16 GELU, including their dynamic
compatibility descriptors. ABI v2 uses native-only status symbols; missing or
failed dispatch refuses. The native Apple softmax/GELU lowering also consumes
status with `cf.assert`; it no longer drops the result of a fallible submission.
The legacy void APIs retain their explicit compatibility fallback. Positive i32
shape limits are checked before the runtime boundary.

Static GELU packaging now accepts a retained native artifact, replays its parent,
and projects its tensor contract from native-printed IR. Unsupported policies,
extra operations, edited output IR and wrong target identity refuse before
runtime access. This uses the existing Apple native library lowering rather
than adding a duplicate Schedule dialect operation. Dynamic GELU now delegates through the same native parent/replay artifact path.
Native dimension positivity and i32 element-count guards precede allocation.
CUDA floating matmul (including bounded dynamic and fused epilogues) and ROCm
plain/dynamic matmul now verify descriptor projections against native Schedule
and replayed Tile IR. Static axes retain equality guards even in a partly
dynamic CUDA package. Graph text remains historical provenance, not an input
to these package verifiers.

F3 now has explicit native instantiation of a straight-line matmul recipe using
`tessera-symdim-equality`'s opt-in dimension witness. Two buckets preserve one
optimized parent identity and produce distinct concrete IR identities. This
has no execution/promotion bit: broader shape-transfer rules, native ANN pair
execution and artifact-bound arbiter admission remain next consumers.

Package profiling identified buffer-contract preparation overhead. Only immutable
NumPy dtype spelling is cached. Mutable nested descriptor metadata invalidates
the cached-identity premise, so descriptor digests are recomputed. Every launch
still validates live buffers/scalars. Five new M1 Max complete-package/direct-ABI
comparisons do **not** justify promotion; see the
[reports](../../../benchmarks/baselines/apple_package_status_20260907/README.md).

Remaining dependency order:

1. Dynamic GELU and CUDA/ROCm floating matmul descriptor projection are implemented
   for their existing envelopes, with native replay and metadata-corruption
   regressions. Broader GELU shapes and ROCm fused matmul are not admitted;
   neither package replay nor differential execution establishes promotion.
2. Extend recipe shape transfers to the chosen ANN fragment; bind original and
   transformed executable artifacts, numerical policy and measured admission to
   each concrete instance. Matmul instantiation alone does not close ANN/F3.
3. Static mixed f32/f64 slots and proven counted whiles now have a native
   persistent consumer. Continue integer/predicate and dynamic slots, genuinely
   data-dependent while termination and completion-owned asynchronous retirement.
   Selected bounded SAVE/HYBRID/recompute-all plans execute on CUDA/HIP; automatic
   checkpoint-plan selection and performance promotion remain open.
4. FA-1 now has a bounded frozen-affine absolute-error consumer in native ANN
   admission. Extend beyond this fixed f32/linf envelope: serialized numerical
   carriers, general reductions, spectral/approximation consumers and justified
   domain composition. Sampled agreement alone remains insufficient.

Cross-cutting: all three current LLVM installations report assertions OFF.
A separate assertions-enabled LLVM/MLIR build remains required. The generated
route inventory must distinguish compatibility frontend adapters from
Graph-owned packaging, and a native library call from general program-body
compilation. The fleet packet
must be resealed after committing the changed runtime source, before publishing.

### 2026-09-07 — Dynamic package projection validation

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

F2 / IR-NATIVE-FOUNDATION-1: the dynamic GELU native producer handles dimensions
in IR; its compatibility frontend no longer constructs a separate descriptor.
Argument-local dimension names survive packaging. Matmul validation projects
shape bounds, individual dynamic axes, storage, accumulation, epilogues and
entry/tile identity. A modified Tile program refuses before compilation.

Focused WSL compiler/registry checks: 371 passed, 26 skipped for capabilities;
RTX 5070 scheduled-matmul device suite: 14 passed; M1 Max selected native
softmax/GELU suite: 6 passed; gfx1151 scheduled matmul: 3 passed. These are correctness results, not performance
promotion packets. Shared changes do not establish x86 or other GPU parity.

The next implementation boundary is F3's native ANN program-pair executable
adapter and artifact-bound admission. In parallel with that dependency chain,
AD-RESIDUAL-EVAL-1 still requires typed saved slots and bounded while retirement
before general checkpoint execution, and FA-1 still requires a real analytic
consumer. None of those is closed by this F2 increment.

### 2026-09-07 — Native ANN admission and persistent checkpoint execution

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners: **F3 / FA-1 / AD-RESIDUAL-EVAL-1 / W2.4a / IR-NATIVE-FOUNDATION-1**.

`native_ann.py` prepares one native two-affine-layer to one-layer rewrite and
replays its native producer before admission. The evaluator uses the existing
MLIR/LLVM JIT on x86, with no Graph reconstruction or reference fallback. Both
programs are checked against an independent rational-arithmetic affine oracle;
matching wrong native outputs refuse. `register_native_ann` registers original
and rewritten programs in the existing arbiter under its internal `ann_affine`
family. The field is bound to the program pair, input domain, budget and probe
snapshots. An over-budget rewrite is excluded even before timing. Equal-tier
selection retains the original; actual calls recheck the input domain.

The analytic contract is deliberately bounded: static rank-two f32 tensors with
dimensions at most 64, frozen normal-or-zero parameters, explicit reassociation
permission, finite inputs with `||X||inf <= R`, and absolute output linf difference from the original program.
Round-to-nearest is checked on the calling thread; other rounding modes refuse.
Exact rational analysis includes folded parameter error, each matrix's induced
column-sum norm, at most 2K arithmetic roundings per dot plus bias rounding, and
additive minimum-normal allowances for underflow/flush-to-zero. `2Ku < 1` and a
conservative intermediate-overflow exclusion are mandatory. Downstream domain
bounds include preceding rounding errors. This is numerical eligibility, not a
performance packet, automatic JIT routing, or GPU ANN proof. Broader recipe
instantiation, nonlinear ANN fragments, numerical-carrier integration and
measured promotion remain open.

Persistent storage now projects f32/f64 widths through compiler private arrays,
ABI manifests, owned allocations and backward outputs. Replay-loop capacity is
derived from bounded SSA arithmetic/selects over enclosing induction variables;
loaded bounds and unchecked maximum annotations do not constitute proofs. Every
nested allocation reserves a distinct slot for its maximum iteration path.
The optional paired-AD `normalize-counted-while` converts only pure zero-origin,
unit-step, constant-bound whiles with sufficient declared capacity to native
for loops, retaining SAVE/HYBRID policy. Other while forms remain refused by the
physical tape consumer.

The owning-device recorder exercises mixed storage, nested SAVE, HYBRID,
recompute-all and counted-while SAVE, with repeated backward calls and unchanged
residuals. See `benchmarks/baselines/tape_checkpoint_20260907/`. Physical retained
bytes are recorded separately from private temporary capacity. These results
do not establish latency, overlap, general mixed-state while execution, or
checkpoint autotuning. Apple still needs MSL-owned typed tape materialization;
x86 ANN execution does not supply CUDA/HIP ANN proof.

### 2026-09-07 — Data-dependent native tapes and nonlinear GPU ANN

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners: **F3 / FA-1 / AD-RESIDUAL-EVAL-1 / W2.4a / IR-NATIVE-FOUNDATION-1**.

The next physical increment preserves MLIR ownership throughout:

- `normalize-data-while` admits pure, single-block whiles with a zero-origin,
  unit-step index counter and an actual signed `counter < constant` conjunct.
  It freezes the complete carried state after a data-dependent exit, then uses
  the existing generic counted-loop checkpoint machinery. A `max_iters`
  annotation by itself is insufficient. Capacity remains at most 1024 steps;
  physical temporary storage remains at most 4096 logical bytes.
- `box-product-scalars` projects logical index/predicate residuals, including
  checkpoint tensors, into i64/i8 tensor storage in the native export. It does
  not change the differentiation domain: discrete region yields no longer
  receive cotangent seeds. Rank-zero LLVM memref descriptors now omit empty
  dimension/stride arrays. Dynamic tensor extents still refuse at this physical
  boundary; static integer checkpoint storage is not dynamic allocation proof.
- Persistent backward can submit distinct generations on caller-owned streams.
  Completion polling retires event owners, while each result advertises its
  producer stream. Explicit generation release waits for device completion and
  frees only that generation. Fully asynchronous allocation retirement requires
  tracking every downstream reader; no such general closure is claimed.
- The native frozen-affine rewrite can retain a terminal ReLU. ReLU is
  nonexpansive in the absolute infinity norm, so the existing analytic bound
  remains valid. Internal nonlinear activations cannot be commuted through an
  affine composition. CUDA/HIP original and transformed programs now consume
  native bufferized IR, preserve nonsplat constants, and replay source-to-device
  arena ancestry before binding. This is a bounded serial physical baseline,
  not a tuned tensor-core/WMMA ANN schedule.
- Nine independent processes compare each original/rewrite package, including
  the host buffer bridge, using randomized paired samples. Raw measurements are
  rechecked before applying the existing exact median order-statistic interval.
  Nine runs give a second-order bound, so one extreme run cannot alone set
  the lower endpoint. The lower bound must exceed a 2% margin. Package wall time is not a kernel
  clock, and numerical eligibility does not register a production GPU winner.

Six tape cases pass on each GPU. Nine independent ANN runs per target refuse
promotion: CUDA median 1.00447× (lower 0.98833×), ROCm median 1.00243×
(lower 0.99360×), against the 1.02× threshold. Evidence is in
[the native tape/ANN packet](../../../benchmarks/baselines/native_tape_ann_20260907/README.md).
The previous bounded checkpoint packet remains historical evidence for its own
source fingerprints. Neither GPU transfers proof to Metal or x86 tape execution.

Next concrete boundaries remain: shape-varying residual allocation with native
extent guards; broader while/CFG recovery beyond the proven counter envelope;
reader-complete asynchronous allocation retirement; GPU arbiter registration
and tuned ANN schedules; reduction/spectral/approximation error-budget consumers;
and measured promotion only where the independent bound passes. Assertions-enabled
MLIR validation remains missing on the fleet.

### 2026-09-07 — Shape-varying host tapes and GPU ANN arbitration

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: #734

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners: **F3 / FA-1 / AD-RESIDUAL-EVAL-1 / W2.4a / IR-NATIVE-FOUNDATION-1**.

Native Tessera-to-Linalg binary lowering now derives dynamic output extents
from the operand and checks unresolved operand/result equality before identity
indexing. Dynamic `zeros_like` adjoints use logical primal extents. The host JIT
accepts signature-checked i64/i8 residual buffers without widening its floating
high-level math API. Three shrinking-loop cases (input widths 4, 8, 16) execute
native x86 forward and repeated backward products with saved shape tapes and
unchanged persistent payloads. The host JIT now runs upstream ownership-based
deallocation after DPS result copies, including loop-carried temporary allocations.
Native extent violations currently lower through `cf.assert` to process abort,
not a recoverable Python exception. This is bounded shape-varying **host execution**;
CUDA/HIP persistent slot allocation still rejects dynamic extents.

Counter-bound recovery also recognizes the frontend's false-else short-circuit
`scf.if`, and equivalent `arith.select`. An arbitrary else value cannot prove
capacity. Arbitrary source CFG recovery, unbounded termination, dynamic device
slot allocation and generalized checkpoint selection remain open.

Terminal absolute value joins ReLU as a nonexpansive consumer of the affine
absolute-error bound. GPU candidates bind the logical domain and numerical
budget to both native artifact digests and the owning architecture. Scoped
registration retains the original under a zero rewrite budget and unregisters
its candidates only after successful binding close. Both GPUs execute all four
ReLU/absolute-value budget cases. The optional upstream elementwise-fusion
pipeline is serialized and replayed as part of artifact ancestry.

Nine fresh independent processes per GPU still refuse performance promotion:
CUDA median 0.99809x, lower bound 0.97773x; ROCm median 0.99728x, lower bound
0.98430x. Neither clears the 1.02x threshold. These are warm package wall times,
including transfers, for the small serial schedule; they neither measure kernel
clocks nor establish a tuned tensor-core/WMMA schedule. See
[raw reports and scope](../../../benchmarks/baselines/shape_tape_ann_20260907/README.md).

Fully asynchronous reclamation remains open: producer-event completion cannot
prove completion of readers of exported views. The next implementation needs
reader leases that prevent new acquisitions after retirement, stream-ordered
allocator/free support, and failure retention until all recorded readers finish.
The existing context barrier is retained. Further numerical consumers need
operator-specific induced-norm/error propagation; absolute value does not prove
reduction, spectral or approximation legality. Assertions-enabled MLIR validation
and Metal-owned tape storage remain required follow-ups.

**PR #734 review closure (F3):** `register_native_ann` returns a
`NativeANNRegistration`; use its `.region` inside a context manager or call
`.close()` when retiring a model/bucket. Close removes exact candidate instances,
invalidates retained candidates and releases owned probe references. Same-name
replacement ownership and registration rollback are tested without native tools.
Native-only tests now check compiler/JIT availability before preparing IR;
clean CI does not claim native execution. No new physical promotion is made.

### 2026-09-07 — bounded while recovery and row-parallel ANN

Owner: [NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners: **W2.4a / AD-RESIDUAL-EVAL-1 / F3 / FA-1**. This increment remains
landing; it is not general CFG, GPU dynamic-storage, or reclamation closure.

- Data-dependent while normalization proves the trip capacity for nonnegative
  constant starts and positive constant strides (each at most 1024), with signed
  `<` or `<=` guards. It retains the original carried counter and refuses an
  insufficient declared maximum. The checkpoint envelope still requires a
  proven capacity of 2 through 1024. Both owning GPUs validate early exits and
  repeated reverse evaluations for a counter starting at 2 with stride 2.
- The native x86 JIT executes shape-varying while residuals for widths
  3, 4, 5, 8 and 16, covering zero through three iterations and two reverse
  evaluations without changing saved state. This connects existing logical
  shape tapes with while recovery; it supplies no GPU dynamic-allocation proof.
- A terminal fp32 row sum is an ANN numerical consumer. Its bound multiplies
  input error by the row width and adds a reduction gamma bound plus the
  existing subnormal allowance. Folded-parameter error receives the same
  width factor. Native execution and binding project the rank-one result;
  nonlinear-plus-reduction and other reduction axes remain outside this slice.
- Optional `parallel-ann-rows` checks actual load/store row independence for
  batches 2 through 64 before assigning one row to each GPU thread. Cross-row
  reads, input writes, unknown memory operations/aliases, outer loop-carried
  values and unmatched outer loops refuse.
  The launch shape and pipeline choice are serialized and replayed. Private
  temporary capacity remains conservative; this is not a tensor-core schedule.
  Both SM120 and gfx1151 validate ReLU, absolute value and row-sum admission,
  including zero-budget rewrite exclusion and scoped registry cleanup.

Next ordered dependencies:

1. GPU shape-varying slots need an allocation-capacity proof separate from the
   logical shape, plus bounds-checked residual reads in backward. Never infer
   GPU support from the host memref descriptor or replace dynamic extents with
   their maximum in logical copies.
2. Extend recovery to typed source CFG edges only with dominance, condition
   effects and carried-state proofs. Arbitrary multi-block/while recovery is
   still open; the new constant-stride envelope does not establish it.
3. Reader-complete asynchronous retirement needs scoped reader acquisition,
   prohibition of new readers after retirement, completion events on every
   reader stream and allocator-owned stream-ordered frees. Existing raw exported
   views must retain context synchronization. Event/allocator failures must
   retain owners, including partially queued frees. Producer completion alone
   cannot authorize reclamation. No reclamation implementation changed here.
4. Extend numerical consumers through explicit induced norms and intermediate
   overflow checks; reduction support does not establish spectral legality.
5. Compare serial and row schedules for the same native program, then introduce
   tiled target-owned schedules. Rewrite-versus-original package timings under
   the row schedule do not prove that row parallelism beats serial execution.
   Performance promotion remains disabled pending selector-grade evidence.

Validation and raw device packets are recorded under
`benchmarks/baselines/row_ann_while_20260907/`. LLVM assertions and architecture-
owned Apple tape storage remain required follow-ups.

### 2026-09-07 — dynamic temporary capacity and tracked reader retirement

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owners: **W2.4a / AD-RESIDUAL-EVAL-1 / IR-NATIVE-FOUNDATION-1**. Landing.

`NativeTapeToGPU` now reserves each internal dynamic allocation from an SSA
interval proof, retaining its actual logical extents in the memref view and
copy loops. Bounded iteration paths still own separate capacity-sized slots,
within the 4096-byte reservation limit. Dynamic copies require identical shape
SSA on source and destination allocations; loaded extents, dynamic external
arguments and unknown dynamic aliases refuse. Both SM120 and gfx1151 execute
logical widths 1/2/3 in distinct three-element slots, without writing unused
capacity. This closes a bounded internal-storage slice, not the exported
dynamic-shape tape ABI or arbitrary allocation envelopes.

Bounded native CFG recovery accepts typed `cf.switch` case/default edges in
addition to branches. It validates edge ABIs and preserves selected successor
state via structured conditionals. Native x86 forward/reverse tests cover two
cases and the default; the existing positive replay bound and pure-body rules
remain. General Python source-CFG recovery and effectful/unbounded CFGs remain
open. Native edge handling is not evidence that the frontend recovers all source
control flow.

`frame.backward_async(stream, seed, tracked=True)` uses native stream-ordered
pool allocations. The resulting generation lends views through `read(stream)`
scopes. Every scope closes with a recorded reader event; `retire(stream)` forbids
new readers and orders frees after the producer and all readers. `poll()` retires
owners after completion. Native tensor submissions reject a borrowed view on a
different stream. A borrowed view may only be submitted on its declared stream
before scope exit; caching/exporting its raw address bypasses this ownership API
and is outside the contract.

Healthy tracked retirement uses no context wait. Event-record failures retain a
stream completion fallback; partial frees are never resubmitted. Failure paths
may require explicit blocking wait/close. The parent frame's unrestricted
primal/residual exports and legacy derivative views retain context-synchronized
close. This is reader-complete asynchronous derivative retirement, not arbitrary
external-pointer reclamation. Tests cover two readers, scope invalidation,
retirement exclusion, failed event records and partially queued frees. Both GPU
hosts validate four exit paths, two readers and a third retirement stream while
forbidding a context barrier in the tracked region.

Evidence: `benchmarks/baselines/dynamic_readers_20260907/`. No throughput,
overlap, allocator-latency bound or cross-architecture performance claim is made.
Remaining priorities are dynamic external descriptor/status projection and saved
logical-shape validation, general source-CFG recovery, and adoption of scoped
reader ownership by composed consumers. Apple needs its own MSL allocation and
completion binding; x86 native CFG execution supplies no CUDA/HIP physical proof.


**Allocator-failure boundary:** a failed free quarantines the frame and retains
its owners for device teardown; normal close must not retry an ambiguously freed
pointer. Completion-event failures alone retain the explicit stream-wait recovery
path. Quarantine is intentionally not reclaimed by garbage collection.


**Verified CFG device blocker:** exported multiway products retain
`cf.assert ... "bounded native CFG exhausted max_steps"`. The GPU packager
currently refuses this guard. A native device status/termination consumer is
required before those products become executable; removing the assertion is not
an admissible migration. The dynamic temporary device packets validate storage
lowering separately from this still-host-only multiway AD product.

Pool/event ownership follows the ordering contracts in the
[CUDA stream-ordered allocator guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
and [HIP stream-ordered allocator guide](https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html).
The measured native cases include zero logical extents (widths 0/1/2), while
retaining positive physical capacity.


Cross-block SSA values are also owned state: dominating operation results and
foreign block arguments used by a successor are mapped to distinct state slots,
not left pointing into the erased CFG region. Native forward/reverse regressions
cover direct shared definitions across all switch cases and the default. Dynamic
saved values still require explicit shape envelopes for their new state slots.


ROCm already has a separate per-element state-machine status consumer in
`GenerateROCMStateMachineKernel.cpp`, covered by the irreducible-CFG execution
lane. The pending work is integrating/extending that status contract for the
persistent-product route, plus a CUDA counterpart; it is not a claim that ROCm
has no bounded CFG device execution. Preserve each route's shape and control
scope when reusing the status mechanism.

### 2026-09-07 — Checked persistent products and composed readers

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Not specified in this historical entry; no merge inferred.

Outcome: Original recorded scope and validation follow below.

Remaining: Historical obligations below route through the Owner; they do not set current priority.

Evidence: Original source, tests and packet references preserved below.

<!-- entry-fields:end -->

Owner: **W2.4a / AD-RESIDUAL-EVAL-1**; synchronization key:
**IR-NATIVE-FOUNDATION-1**. This increment remains a bounded slice.

- Native function-body CFGs can enter the same bounded recovery as imported
  `scf.execute_region` CFGs. Function arguments remain external SSA values;
  returns become region yields. Positive step bounds, identity, purity and
  supported typed edge/state requirements remain enforced. This is native MLIR
  source recovery, not arbitrary Python syntax or effectful CFG recovery.
- `tessera-native-tape-to-gpu`'s opt-in `status-buffer` appends a one-element
  i64 status pointer (`tessera.autodiff.gpu_status = "guard-v1"`). It initializes
  success, replaces top-level assertions with guarded suffixes, and reports 1
  on a failed guard. Subsequent memory operations do not execute on failure.
  Nested assertions refuse; the serial CUDA/HIP product boundary is unchanged.
- `materialize_persistent_tape(..., checked_status=True)` projects and verifies
  that physical status ABI on both products. Capture and synchronous backward
  consume status before exposing outputs. Status-enabled asynchronous backward
  deliberately refuses pending an asynchronous checked-result protocol.
- Tracked generations provide `submit_to` for native tensor consumers and
  `backward_into` for persistent reverse composition. Both delimit reader
  ownership across validation/enqueue, including exceptions. Synchronous tensor
  calls reject borrowed views because they cannot honor the declared stream.

**Exported shape-varying ABI remains open.** Internal dynamic temporaries and
static capacity/shape-tape residuals are not a runtime-sized public result ABI.
The next implementation must return logical extents alongside capacity-backed
result storage, with dtype/rank/capacity projected from serialized compiler IR.
Caller-supplied extents must not stand in for device-computed output shapes.
Validate each loaded extent before reconstructing a view or replaying a slice;
carry status through both shape validation and CFG exhaustion. A failed product
must not expose data or reusable shape metadata. Forward-owned shape residuals
must remain immutable across repeated reverse calls. Tests must cover zero
logical extents, exact capacity, overflow, mismatched cotangent shape and failure
before any out-of-bounds access. The current GPU external-slot parser continues
to refuse dynamic slots rather than fabricate these contracts.

General source CFG recovery additionally needs frontend-produced typed edges and
SSA merge values for break/continue/early returns, effect-aware replay and
shape-envelope propagation. Do not infer those from native multi-block support.
For asynchronous checked outputs, status must gate every downstream reader or
complete a checked host ticket before exposure; merely recording a producer
event proves completion, not success. General allocator/reclamation performance
and assertions-enabled MLIR validation remain open.

Owning-device packets for this increment are in
`benchmarks/baselines/product_status_20260907/`: RTX 5070/SM120 and Radeon
8060S/gfx1151 each pass six cases covering switch/repeated reverse, imported
function CFG, exhaustion refusal and scoped native consumer composition.
Checked status is synchronous; the stream composition case uses ordinary static
products. No timing, overlap or performance promotion is inferred.

### 2026-09-07 — Compiler plan role migration

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted documentation and governance increment.

Outcome: One grouped live queue, ID-only anchors, a reference log and archived wave provenance replace competing delivery-order sections. Detailed FA gates remain in their scoped reference. Capability and device support are unchanged.

Remaining: Follow the live queue; assertions-enabled compiler validation and the existing tool-execution gates remain open. Historical numerical/device claims have not been remeasured by this move.

Evidence: `scripts/check_compiler_plan.py`, `tests/unit/test_compiler_plan_routing.py` and existing audit/governance tests; 42 focused tests on Princess-Luna and eight routing tests on Super-Bear, plus the 30-document generated drift gate. No performance claim.

<!-- entry-fields:end -->

The routing index preserves successors and archived dispositions without copying
support statuses. CI checks structured references and requires an owner-record
or disposition update when adding an entry; the pre-push hook performs the
structural check. Current-priority links were separated from historical evidence
links, including the earlier math/capability reconciliation anchors.


### 2026-09-08 — Native unary ancestry and direct x86 migration

Owner: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f)

PRs: Uncommitted follow-through after #735.

Outcome: x86 unary packages replay the native Schedule parent, verify native target attributes and project descriptor shapes, scalar extents and arithmetic policy before compilation; Apple reuses the bounded projection verifier.

Remaining: Other family ancestry and full route/envelope census; assertions-enabled LLVM/MLIR validation remains unavailable on the probed fleet builds.

Evidence: tests/unit/test_x86_unary_migration.py; tests/unit/test_scheduled_kernel_consumers.py; llvm-config --assertion-mode reports OFF on Princess-Luna and Super-Bear.

<!-- entry-fields:end -->

Forged dimensions/scalars, altered Tile dataflow and relabelled foreign parents
refuse before compilation. Host aliases remain explicit binding aliases rather
than claims about native SSA names. No device evidence transfers between targets.

### 2026-09-08 — Direct x86 unary caller migration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #735.

Outcome: Direct AVX-512 softmax and rank-reducing sum/mean/max route through the existing native Schedule/Tile producer and checked consumer.

Remaining: Baseline x86 and keepdims reductions retain their Graph constructors; other census families and frontend retirement are unchanged.

Evidence: tests/unit/test_x86_unary_migration.py direct runtime execution on Princess-Luna and native ancestry rejection tests.

<!-- entry-fields:end -->

The lower-level C ABI and native library implementation are unchanged. This is
package-authority migration, not new vectorization or a performance promotion.
The census still counts Graph-typed compatibility entry points; a migrated
branch does not erase its function's remaining fallback surface.

### 2026-09-08 — Checked asynchronous derivative tickets

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)

PRs: Uncommitted follow-through after #735.

Outcome: Static checked GPU products enqueue backward asynchronously with an independent status allocation per generation; successful wait/poll is mandatory before exposing outputs.

Remaining: Exported logical shapes, loaded extents, nested guards, fully asynchronous allocation/reclamation and device-gated tracked readers remain open under AD-RESIDUAL-EVAL-1 and W2.4a.

Evidence: benchmarks/baselines/checked_derivatives_20260908/; tests/unit/test_checked_derivative_ticket.py; tests/unit/test_native_product_status.py.

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) retains scoped-reader ownership.
Failed status never exposes output and can still drain/release safely; frame
close drains failed submissions without treating status failure as a reason to
leak allocations. Unrestricted successful exports keep their context-completion
release barrier. Two owning-device generations and explicitly injected nonzero
status pass on SM120 and gfx1151; no overlap or performance claim is made.


### 2026-09-08 — Native keepdims reduction

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #735.

Outcome: AVX-512 keepdims sum/mean/max now project output shapes and policy from native Schedule/Tile; descriptor keepdims forgery refuses before compilation.

Remaining: Baseline x86 and remaining census families still retain Graph-owned paths. No performance promotion.

Evidence: tests/unit/test_x86_unary_migration.py; direct native execution on Princess-Luna.

<!-- entry-fields:end -->


### 2026-09-08 — Logical results and nested device guards

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)

PRs: Uncommitted follow-through after #735.

Outcome: Native rank-one capacity-buffer programs return checked device-written logical lengths. Nested for/if guards suppress the enclosing suffix and later loop iterations, including scalar loop-carried exits.

Remaining: This is an explicit buffer-program producer, not automatic dynamic tensor/AD result generation. Multidimensional logical shapes, arbitrary while regions and asynchronous public-result reclamation remain open.

Evidence: benchmarks/baselines/deep_native_20260908/; tests/unit/test_native_public_results.py; tests/unit/test_native_product_status.py.

<!-- entry-fields:end -->


### 2026-09-08 — Device-gated derivative readers

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted follow-through after #735.

Outcome: CheckedDerivativeSubmission.backward_into enqueues a dependent backward with an event wait and native incoming-status gate before body effects; parent status need not return to the host first.

Remaining: Release still uses a context completion barrier, and ordinary checked outputs still require successful host-status consumption. This does not close fully asynchronous reader-aware reclamation or general device graphs.

Evidence: benchmarks/record_deep_native_contracts.py on SM120 and gfx1151; injected upstream failure refuses child output; no overlap claim.

<!-- entry-fields:end -->


### 2026-09-08 — Scoped measured ANN admission

Owner: [MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9)

PRs: Uncommitted follow-through after #735.

Outcome: Scoped native ANN registration binds nine independent reports to exact package images, source/recorder identity, target, domain and analytic budget before oracle-verified arbitration. Closing the registration retires its candidates.

Remaining: Package timing includes the host bridge and is not kernel timing. Broader nonlinear programs, tuning and global production promotion remain open.

Evidence: tests/unit/test_native_ann_measurement_owner.py; benchmarks/baselines/deep_ann_20260908/; benchmarks/select_native_ann_measurements.py.

<!-- entry-fields:end -->


### 2026-09-08 — Assertions-enabled native compiler

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted follow-through after #735.

Outcome: Pinned LLVM/MLIR 23.1.1 assertions-ON build and isolated Tessera consumer run on Super-Bear. An executable LLVM assertion probe aborts as expected. The new lane exposed and fixed NativeTapeToGPUPass's missing Tile dependent-dialect registration; the corrected compiler passes 257 focused checks plus 76 composed tape/ANN/storage/reader checks. The latter retain release downstream LLVM tools and host JIT.

Remaining: Full pass-corpus and installed-driver assertions lanes; other hosts continue using release toolchains. Upstream no-RTTI and Tessera compile flags must match. No GPU execution/performance claim follows from compiler validation.

Evidence: scripts/build_assertions_llvm.sh; scripts/probe_llvm_assertions.py; benchmarks/baselines/assertions_llvm_20260908/; tests/unit/test_native_public_results.py; tests/unit/test_native_product_status.py; native unary, pass metadata and diagnostic registry checks. Five target-tool tests skipped in the isolated build.

<!-- entry-fields:end -->


### 2026-09-08 — Baseline x86 unary ownership

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #735.

Outcome: Baseline softmax/reduction now project from native Schedule/Tile into their own scalar C ABI and image, with no AVX-512 feature requirement. Direct runtime tests cover both baseline families.

Remaining: Matmul/attention, cohort and breadth routes remain Graph-owned; frontend retirement is unchanged.

Evidence: tests/unit/test_x86_unary_migration.py; tests/unit/test_x86_e2e_spine.py on Princess-Luna.

<!-- entry-fields:end -->


### 2026-09-08 — Automatic AD public result generation

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)

PRs: Uncommitted follow-through after #735.

Outcome: NativeTapeToGPUPass generates a checked capacity copy and logical-length sidecar from a rank-one exported AD result. Integer select alternatives extend SSA allocation bounds. Python invokes native export/bufferization rather than synthesizing result-copy math.

Remaining: The dynamic forward slice uses static external inputs and one rank-one result; dynamic GPU backward inputs and saved multi-result/multidimensional products remain open.

Evidence: tests/unit/test_automatic_ad_public_results.py; benchmarks/baselines/automatic_ad_retirement_20260908/.

<!-- entry-fields:end -->


### 2026-09-08 — Checked generation asynchronous reclamation

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted follow-through after #735.

Outcome: Checked derivative data and status use stream-ordered pool allocation/free. Device-gated backward readers register completion before retirement; optional generic scoped reads require explicit successful host status. Successful polling cleans up events without a redundant wait, and failed destruction retains a monotonic completion proof for safe retry.

Remaining: Whole-frame capture/close and exceptional teardown remain synchronous. Uncertain free submissions quarantine ownership. This is not general external-reader tracking or measured overlap.

Evidence: tests/unit/test_native_reader_retirement.py; tests/unit/test_native_gpu_streams.py; benchmarks/record_checked_retirement.py on SM120 and gfx1151.

<!-- entry-fields:end -->


### 2026-09-08 — Independently tuned ANN rewrite

Owner: [MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9)

PRs: Uncommitted follow-through after #735.

Outcome: Native source replay now allows a fused row-parallel transformed candidate against an unchanged serial incumbent. Nine independent 16x8 runs yield scoped SM120 selection (median 1.0753x; interval [1.0374,1.0883]); gfx1151 retains the incumbent (median 0.9997x; interval [0.9657,1.0170]).

Remaining: Evidence is warm host-wall package timing for this frozen affine/ReLU workload, not kernel timing, arbitrary nonlinear closure or global promotion. Registrations retire after use.

Evidence: benchmarks/baselines/tuned_ann_20260908/; benchmarks/record_native_ann_execution.py; benchmarks/select_native_ann_measurements.py.

<!-- entry-fields:end -->


### 2026-09-08 — Runtime-shaped AD products

Owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)

PRs: Uncommitted follow-through after #735.

Outcome: Native export/bufferization now binds dynamic GPU backward inputs with checked physical capacities and per-axis shape sidecars. Multiple rank-one through rank-four floating outputs use independent flat storage; native guards validate each extent and total volume. GPU result submission exposes no logical view until event completion and successful status/shape consumption.

Remaining: Exact-device proof covers rank-one lengths 0/2/4, dynamic rank-two shapes, malformed/mismatched inputs and two matrix outputs. General saved mixed products, arbitrary layouts/aliases and automatic Python capture remain open.

Evidence: benchmarks/baselines/runtime_shape_frames_20260908/; tests/unit/test_automatic_ad_public_results.py; tests/unit/test_native_public_results.py.

<!-- entry-fields:end -->


### 2026-09-08 — Scoped whole-frame retirement

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted follow-through after #735.

Outcome: Opt-in scoped capture uses pool storage and reader leases for primals/residuals. Frame retirement enqueues every generation and frame free after registered readers; successful polling releases idle modules without an explicit context wait. Active readers refuse retirement and uncertain frees retain quarantine.

Remaining: Scoped persistent frames still use static tensor shapes; runtime-shaped public-result frames retain synchronous close. Capture and exceptional cleanup remain synchronous. Module unload has no bounded-latency claim. Existing unrestricted views still require synchronous close; general external reader adoption and measured overlap remain open.

Evidence: benchmarks/baselines/runtime_shape_frames_20260908/; tests/unit/test_native_reader_retirement.py; tests/unit/test_native_gpu_streams.py.

<!-- entry-fields:end -->


### 2026-09-08 — x86 f32 matmul native ownership

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #735.

Outcome: Direct f32 matmul now enters native Graph-to-Schedule-to-Tile lowering. Packaging replays the Schedule parent and validates projected fields before target compilation; descriptor shape forgery refuses.

Remaining: Non-f32 matmul, attention, cohort and breadth constructors remain scoped F2 work. This changes ownership, not the runtime kernel or performance promotion.

Evidence: tests/unit/test_x86_unary_migration.py; tests/unit/test_x86_e2e_spine.py.

<!-- entry-fields:end -->


### 2026-09-08 — Broader ANN workload evidence

Owner: [MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9)

PRs: Uncommitted follow-through after #735.

Outcome: The measurement workload generator and scoped replay admit terminal absolute value alongside ReLU. Fresh 8x4 absolute-value and 32x4 ReLU comparisons retain independently replayed incumbent/rewrite schedules, analytic domains and finite error budgets. Nine quiet runs per workload and target retain the incumbent in all four comparisons; no lower confidence bound clears the 2% margin.

Remaining: Use the revision-bound packets for selection outcomes. Host-wall package timing is not kernel timing or global promotion. Nonexpansive terminal consumers do not authorize moving nonlinearities across affine composition; arbitrary nonlinear graphs remain open.

Evidence: benchmarks/baselines/broad_ann_20260908/; benchmarks/record_native_ann_execution.py; benchmarks/select_native_ann_measurements.py.

<!-- entry-fields:end -->


### 2026-09-08 — Python CFG boundary review

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted follow-through after #735.

Outcome: Source review confirms TraceRef.__bool__ rejects raw value-dependent Python if/while; explicit tessera.control regions and imported native bounded CFGs are the current consumed paths. Nested native status guards now also protect dynamic input shapes and multiple output shape copies.

Remaining: Arbitrary Python source CFG remains open: tracer-owned typed merge values, break/continue/early return, effect legality and source-location-preserving recovery require a frontend producer. Legacy AST markers are not execution proof.

Evidence: python/tessera/compiler/trace.py; python/tessera/compiler/structured_cfg.py; tests/unit/test_trace_f4.py; tests/unit/test_native_public_results.py; docs/audit/compiler/AUTODIFF_EXECUTION_PLAN.md.

<!-- entry-fields:end -->

### 2026-09-08 — Source CFG and asynchronous ownership

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted engineering continuation after #736.

Outcome: Opt-in pure Python branch/early-return and single-carry bounded while recovery feeds typed tracer SSA and native SCF. Scoped dynamic public frames and asynchronous persistent capture use checked status, reader leases and event-ordered retirement. Module unload admission is bounded and polling does not wait for the driver. Dominating exact-product guards bound joint temporary volume. x86 forward attention consumes scheduled artifacts; terminal-square ANN has an analytic error consumer.

Remaining: Effectful/arbitrary Python CFG, multiple loop carries, automatic general JIT wiring, multi-status asynchronous composition, exceptional cleanup, driver cancellation, remaining Graph families and broader nonlinear kernels remain open. Bounded worker admission is not a bound on driver unload latency. GPU correctness is not performance promotion.

Evidence: `benchmarks/baselines/source_async_foundation_20260908/`; focused source, ownership, joint-volume, ancestry and nonlinear tests. NVIDIA's nine-run square comparison retained the incumbent. See the evidence README for independent host outcomes and measured scope.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1),
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a),
[E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6), and
[MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9).
Synchronization key: `SOURCE-ASYNC-FOUNDATION-2026-09-08`.

The source producer rejects effect-only calls, mutation, break/continue and
implicit numeric tensor truth. Native exhaustion asserts rather than silently
truncating. Logical dynamic shapes retain independent sidecars even when an
exact dominating product guard permits a smaller flat physical allocation.

Successful asynchronous retirement covers data, shape and status storage plus
registered readers. Generic capture readers require successful `poll_capture`;
the single incoming-status ABI cannot replace an unchecked capture status with
an external dependency. Unload workers retain their context and admission slot
on failure or stall. Explicit retirement/polling is the asynchronous API;
exceptional cleanup and unrestricted legacy views remain synchronous.

x86 attention now replays Schedule-to-Tile ancestry and projects native fields
through a shared contract. The extended route supplies the runtime's symmetric
`window` field. Non-f32 matmul, backward attention/cohorts and generic Graph
families are not retired by this migration. Square ANN propagates folded
parameter error through the nonlinear tail; route admission still requires
independent exact-device measurement and successful retirement of scoped candidates.

### 2026-09-08 — Effect-aware CFG and status composition

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #736.

Outcome: Bounded break/continue and multi-variable source loops merge iteration state, native assertions preserve effects, and dead-path unmodelled calls refuse. Two serialized native status inputs preserve capture plus upstream failure through checked asynchronous composition. The CAKE enhancement review now maps lessons to current consumers and owners.

Remaining: Loop-return payloads, external mutation/alias contracts, arbitrary Python CFG, wider status joins, exceptional recovery and general JIT integration. This bounded expansion is not general source recovery; host assertion failure still uses the LLVM abort ABI.

Evidence: `benchmarks/baselines/cfg_status_composition_20260908/`, source/compiler negative tests and independent SM120/gfx1151 four-case status truth tables.

<!-- entry-fields:end -->

Additional owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Sync key: `CFG-STATUS-COMPOSITION-2026-09-08`.
The first expansion prototype grew exponentially; per-iteration merges make
successive iterations linear in this fixed body. Native multi-result conditional
traces now retain the full inferred result types. Distinct status storage avoids
relaxing the native package's alias contract. `compiler_enhancement.md` preserves
its August assessment as historical provenance and corrects unsupported current
claims about phase readiness, pruning regret and fixed economic thresholds.

### 2026-09-08 — Completion state and status fan-in

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #736.

Outcome: Bounded nested loops carry tuple return payloads and statically resolved builtin exception completions. Explicit CPU state slots preserve exact full-tensor input aliases through serialized state results and post-completion copyback. Incoming checked statuses support counts one through eight, retaining reader leases for every prerequisite.

Remaining: Arbitrary objects, partial/strided mutable views, mutable return aliases, changing loop-state types, uncaught exception transport, general JIT integration, GPU mutation and heterogeneous/unbounded asynchronous effect joins. Eight-status device execution is not claimed by the four-status packet.

Evidence: `benchmarks/baselines/completion_state_fanin_20260908/`; native CPU differential tests and independent CUDA SM120/ROCm gfx1151 sixteen-case status truth tables.

<!-- entry-fields:end -->

Additional owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Sync key: `COMPLETION-STATE-FANIN-2026-09-08`.
The source producer handles explicit ValueError, RuntimeError and AssertionError
without creating Python exception objects. Finally continuations preserve or
override the pending completion. Native assertion exhaustion still uses the host
abort ABI. State execution requires exclusive host ownership; copyback is not an
atomic transaction against concurrent external readers. Device status-only
prerequisites add gates and reader lifetimes, not extra cotangent operands.

### 2026-09-08 — Source JIT error transport and device state

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #736.

Outcome: CPU source state accepts non-overlapping strided views and exact aliases. Explicit floating result specs transport builtin exception classes, preserving prior writes. Opt-in public JIT owns at most four native shape/alias specializations. ROCm executes two immutable source-state generations and all 256 eight-status combinations.

Remaining: Arbitrary object/overlapping-view mutation, implicit/dynamic exception transport and original messages, broader JIT/AD integration, in-place device state mutation and NVIDIA eight-status/device-state proof. Super-Bear SSH was unavailable for this increment; Apple needs a separate MSL consumer.

Evidence: `benchmarks/baselines/source_jit_state_20260908/`, focused native CPU tests and independent gfx1151 packets. No performance promotion.

<!-- entry-fields:end -->

Additional owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Synchronization key: `SOURCE-JIT-STATE-2026-09-08`.
The sequence mixer theory now distinguishes exact algebra, numerical legality,
state ownership and physical proof. NoPE alone does not justify MQA conversion;
scalar SSD is not generic DPLR, and zero/underflowing decay invalidates division.

### 2026-09-09 — Declared object state and owned GPU mutation

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #736.

Outcome: Declared dict/SimpleNamespace tensor fields project to native arguments and full-slice state writes. Read-only overlapping arrays use snapshots. Static builtin exception args, inherited/tuple handlers and bare re-raise, including pending exceptions in finally, survive native completion transport. Pure tensor source JIT exposes explicit compiler-exported CPU VJP. Exclusive owned GPU state reuses its allocation after synchronous checked computation; scoped readers prevent writes.

Remaining: Custom accessors and object identity/rebinding; overlapping writes; implicit/dynamic exception objects, chaining and traceback semantics; automatic effectful JIT/AD; asynchronous in-place writes and external borrowed storage. The GPU implementation computes into fresh output storage then copies back: this is allocation identity and correctness evidence, not a fused mutation kernel or performance promotion.

Evidence: `benchmarks/baselines/source_object_ownership_20260909/`, focused native CPU regressions and independent SM120/gfx1151 recorder packets.

<!-- entry-fields:end -->

Additional owners: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a), [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).
Synchronization key: `SOURCE-OBJECT-OWNERSHIP-2026-09-09`.
Writable overlapping views require one shared backing-storage SSA root, typed view maps and ordered updates. Independent input snapshots cannot preserve interleaved alias writes, so those writes still refuse before execution. VJP derivatives are per formal tensor input; this does not differentiate mutable object state.

### 2026-09-09 — Mixed aliases and asynchronous state

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted follow-through after #737.

Outcome: Read-only partial overlaps remain admissible beside disjoint mutable state at compilation, cache selection and execution. Plain custom instances project dictionary fields without custom accessors. Explicit CPU VJP differentiates functional results plus declared next-state outputs without copyback. Single-stream GPU submit/poll validates computation before event-ordered copyback and excludes readers until completion; failures poison the owner and retain pending storage.

Remaining: Arbitrary accessors/object rebinding, overlapping writes with common backing-storage SSA, dynamic/implicit exception payloads and chaining, object/exception AD, aliased state VJP, multi-writer mutation and asynchronous teardown of failed updates. Close may synchronize; no full exception or asynchronous lifecycle closure is claimed.

Evidence: `benchmarks/baselines/mixed_alias_async_state_20260909/`; native source regressions, injected pending/failure ownership tests and independent SM120/gfx1151 asynchronous state packets. No performance promotion.

<!-- entry-fields:end -->

Additional owners: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a), [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).
Synchronization key: `MIXED-ALIAS-ASYNC-STATE-2026-09-09`.

### 2026-09-09 — Writable views and exception value transport

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Contiguous rank-one writable aliases share an explicit containing-input SSA root; standard tensor.extract_slice/insert_slice preserve ordered writes and subsequent reads. Runtime revalidates view offsets and cache identity includes the view map. Single-element f32 exception-value payloads cross native completion, loop exits and finally; nested re-raise restores the outer payload. Explicit native VJP projects declared object fields and differentiates public/next-state values without mutation.

Remaining: Writable views without a supplied containing input, noncontiguous/multidimensional write maps, aliased state adjoints, dynamic strings/heterogeneous exception values, cause/context/traceback transport and implicit operation errors. Mutable exception aliases refuse rather than silently changing finally semantics. Exception AD and GPU view/exception execution remain open.

Evidence: `benchmarks/baselines/source_views_exception_20260909/`; native CPU execution regressions and independent SM120/gfx1151 regression packets for the existing single-state asynchronous consumer. Device packets do not prove writable view or exception execution. No performance promotion.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Synchronization key: `SOURCE-VIEW-EXCEPTION-2026-09-09`.

### 2026-09-09 — Strided roots static causes and device execution

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Positive-stride rank-one views and local static slices project onto an explicit contiguous root. Exact-alias state VJP accumulates at that root and returns zero for duplicate alias arguments. Explicit literal builtin causes and from-None suppression cross the native completion boundary. GPU bufferization copies before writes; the native ownership verifier resolves bounded known subview/cast chains to private allocations or output roots and continues rejecting input-rooted writes.

Remaining: Negative/multidimensional views, nontrivial slice adjoints, general caught exception identity/context/tracebacks, dynamic causes, exception AD and GPU exception transport. Copy-before-write can increase temporary storage and does not establish performance eligibility. Local one-root GPU slices do not admit arbitrary external aliased device pointers.

Evidence: `benchmarks/baselines/strided_source_gpu_20260909/`; focused source/ownership tests, compiler rebuilds, and independently measured strided asynchronous state updates. No performance promotion.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Synchronization key: `STRIDED-SOURCE-GPU-2026-09-09`.

### 2026-09-09 — Mapped adjoints and GPU exception completion

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Bounded injective negative/multidimensional views lower through standard tensor slices. Native slice adjoints accumulate overlapping reads at the containing root and mask overwritten destination gradients. Static caught bindings retain identity through re-raise; explicit causes and implicit contexts decode as an exception graph, with logical native source notes. SM120 and gfx1151 independently execute mapped forward/backward and checked synchronous/asynchronous static/dynamic exception cases. Failed public frames remain unreadable and repeated polls retain the same error object. Block AttnRes now routes these foundations into native package, lifetime and performance gates; its incorrect full-matrix-rank liveness claim is removed.

Remaining: Runtime-sized/rank-changing or large mapped views, noninjective writes, general object mutation, loop exception-context slots, retained dynamic context payloads, real Python traceback frames and exception AD. Owned in-place GPU mutation remains exception-free. Multi-state device aliases, tuned cooperative kernels and performance promotion are separate.

Evidence: `benchmarks/baselines/source_exception_gpu_20260909/`; native source/ownership regressions, finite-difference view adjoints, Block AttnRes rank counterexample, independent device packets.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a), [Block AttnRes](BLOCK_ATTNRES_ROCM_PLAN.md).
Synchronization key: `SOURCE-MAPPED-EXCEPTION-2026-09-09`.

### 2026-09-09 — Runtime maps context slots and cooperative experiment

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Positive rectangular source maps use compact native slices; CPU VJP no longer inherits a GPU slot cap. Per-site dynamic exception payload slots retain cause/context values; CPU VJP validates forward status before backward and zero-seeds completion metadata. Runtime-shaped native slice products execute across shapes on SM120/gfx1151, with exact cotangent-shape assertions, nonnegative unsigned-division allocation bounds and dominating equality proofs for logical copies. Block AttnRes has an opt-in cooperative width reduction, distinct cache/compiler identity and serialized pipeline policy; measured operation-total results do not establish promotion.

Remaining: Automatic runtime Python slice capture, large/general negative maps, generation-indexed loop context storage, full CPython frames/tracebacks and product-aware GPU exception AD. Cooperative Block AttnRes needs device-clock attribution, broader numerical/shape coverage and repeatable package wins; CUDA depth-attention packaging remains separate.

Evidence: `benchmarks/baselines/runtime_source_maps_20260909/`; CPU source/AD, artifact and registry tests; independent twenty-case GPU packets; default/cooperative gfx1151 package measurements.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a), [Block AttnRes](BLOCK_ATTNRES_ROCM_PLAN.md).
Synchronization key: `RUNTIME-SOURCE-MAPS-2026-09-09`.


### 2026-09-09 — Source index generations and checked GPU VJP

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Python source slices lower runtime int64 tensor bounds and positive steps to native arithmetic and dynamic slice results. Read-only view roots retain their SSA value across local rebinding. Bounded expanded loops carry one completion code and distinct per-generation exception payload slots; nested cause/context chains no longer depend on a frozen pre-loop edge table. Native public results preserve scalar i8/i64 residual types. A paired GPU VJP owns device input snapshots, checks forward exceptions before backward and zero-seeds completion metadata. Repeated failed-frame polling preserves exception identity without accumulating old host traceback chains.

Remaining: Python integer/index protocols, negative runtime strides, nested/runtime-shaped source roots and dynamic CPU JIT result allocation; cross-iteration arbitrary exception objects and unbounded storage; full CPython traceback frames/locals; asynchronous exception-aware AD. Block AttnRes promotion is unchanged.

Evidence: `benchmarks/baselines/source_generation_ad_20260909/`; native CPU and independent SM120/gfx1151 slice, generation and checked-VJP cases. No performance promotion.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Synchronization key: `SOURCE-GENERATION-AD-2026-09-09`.


### 2026-09-09 — Signed nested views and asynchronous source VJP

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #737.

Outcome: Signed runtime slice maps compose nested rank-preserving views against an immutable root and lower through tensor.generate, including INT64_MIN steps and empty results. The CPU JIT uses compiler-projected capacity/shape sidecars to allocate multiple multidimensional dynamic outputs; Python integer bounds remain runtime inputs. Identity memref arguments receive Python and C static-extent checks. Bounded loop-carried exception references retain their original generation payload and explicit cause. Same-stream GPU snapshots and checked asynchronous VJP stage backward only after a successful forward, retaining matching residuals until completion.

Remaining: Runtime-shaped original roots, custom index protocols, runtime gather adjoints, arbitrary custom exception heaps and unbounded generations; full CPython frames with live locals/closures and instruction identity; fully asynchronous retirement and cross-queue ownership. CPU automatic capacity is bounded to 1024 elements. Device source VJP remains single-input without projected state aliases/object fields. No performance promotion.

Evidence: `benchmarks/baselines/source_nested_async_20260909/`; 19 independently measured cases each on SM120 and gfx1151, host source/ABI tests and assertions-enabled compiler validation.

<!-- entry-fields:end -->

Additional owners: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Synchronization key: `SOURCE-NESTED-ASYNC-2026-09-09`.


### 2026-09-09 — Gather transposes and scoped source retirement

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #738.

Outcome: Index-only tensor.generate gathers transpose to serial nested scatter-add loops with exact seed-shape assertions, preserving repeated-index accumulation. CPU source JIT retains explicitly captured custom exception class bindings and invokes their constructors only on failed completion. Scoped source VJP records backward as a reader of forward snapshots/residuals; retirement enqueues frees after closed reader scopes and queries completion without a context wait on the healthy path.

Remaining: Arbitrary exception object heaps/attributes in native execution, GPU custom-class bindings, nonlinear generator adjoints, source-level dynamic VJP output allocation, fully asynchronous module unloading/failure recovery and full CPython frames. Native source notes remain distinct from interpreter frames and locals. No performance promotion.

Evidence: Native CPU gather/source regressions; independent 19-case SM120/gfx1151 packets in `benchmarks/baselines/source_scoped_ad_20260909/`.

<!-- entry-fields:end -->

Additional owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).
Synchronization key: `SOURCE-GATHER-SCOPED-2026-09-09`.


### 2026-09-09 — Exception class bindings and unload cleanup recovery

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #738.

Outcome: Expanded raise sites now carry occurrence/generation identities even for identical literal payloads. GPU source VJP accepts explicitly owned host exception-class bindings and validates missing bindings before compilation; failed completion constructs the host exception once, including repeated polling. Actual host constructor/completion traceback frames survive repeated polls without accumulating polling frames. Module retirement separates confirmed unload/context exit from filesystem cleanup. Only post-unload cleanup failures admit off-thread retry through native tensor, public-result and source-product owners; uncertain driver outcomes retain their quarantine and admission slot.

Remaining: Unbounded exception heaps, arbitrary object-field mutation/arguments in native execution, custom constructor effects inside handled source paths, true CPython frame reconstruction, cancellation of stalled driver unload and recovery from uncertain free/unload/context failures. No performance promotion.

Evidence: `benchmarks/baselines/source_exception_bindings_20260909/`; CPU identity and failure-injection tests plus independent SM120/gfx1151 custom-class completion and scoped-retirement packets.

<!-- entry-fields:end -->

Additional owner: [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).
Synchronization key: `SOURCE-EXCEPTION-BINDINGS-2026-09-09`.

### 2026-09-09 — Architecture sweep and failure-boundary reconciliation

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)

PRs: Uncommitted engineering increment.

Outcome: Reconciled the historical sweep into live owner routes; structural shape compatibility replaces string identity and pipeline validation rejects missing stages. Selected exception graphs validate before host constructors; unload failures survive a second context-exit failure with ownership retained.

Remaining: General nonlinear constraint solving, automatic distributed placement, arbitrary exception heaps, native CPython frames and uncertain driver recovery remain open. No new device or performance promotion.

Evidence: Super-Bear host WSL: 228 passed, 1 skipped across source/shape/distributed/lifetime/audit regressions; Ruff passed and mypy checked 305 files. Princess-Luna host WSL: 18 shared regression/fault-injection cases passed independently. This is not new GPU execution evidence.

<!-- entry-fields:end -->

Additional owners: [DIST-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#dist-native-1),
[W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1),
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Sync: `ARCH-SWEEP-FAILURE-2026-09-09`.
See [reconciliation](COMPILER_ARCHITECTURE_SWEEP.md#current-reconciliation-2026-09-09).
Previous source-exception packets retain their original fingerprints; they are
historical evidence and are not re-labelled as validation of this changed tree.

### 2026-09-09 — Indexed exception completion and one-shot unload

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted engineering increment.

Outcome: Source IR emits indexed exception objects, runtime validates and materializes shared/cyclic identities iteratively, and synchronous unknown unload/sync outcomes retain owners and refuse repeat driver actions.

Remaining: Arbitrary runtime native heap allocation, full CPython deoptimization/frame materialization and isolation-based recovery remain open. Current source generations remain bounded; no performance promotion.

Evidence: Super-Bear host WSL: 201 passed, 1 skipped; Princess-Luna: 18 heap/retirement tests passed. Ruff and mypy (306 files) pass. `benchmarks/baselines/source_exception_heap_20260909/` records 19 cases each on SM120 and gfx1151; this is completion-carrier correctness, not native heap allocation or destructive driver recovery.

<!-- entry-fields:end -->

Additional owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Sync: `SOURCE-HEAP-RETIRE-2026-09-09`.

### 2026-09-09 — Completion reclamation and broader assertions corpus

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted engineering increment.

Outcome: Broader lit execution found an undeclared Tile dialect dependency in AutodiffForwardPass; declaring it fixes the assertions-build abort. Fifteen x86/Apple fixtures now declare the optional backend they consume, so omitted components report unsupported rather than misleading compiler failures. Schedule-to-Tile declares its NVIDIA Target dialect dependency instead of loading it during pass execution. A hardware-free all-target build can explicitly retain the full compiler driver; ROCm translation registration follows the serialization libraries actually linked. The resulting assertions build passes every active fixture, and the opt-in CI lit lane requests the same portable matrix. Successful public completion retirement drops cached exception/traceback roots without mutating caller-owned errors. Unknown synchronous frees retain frames and refuse retry.

Remaining: Structural assertions coverage is closed. Add installed-driver smoke; keep native Apple, NVIDIA and ROCm device correctness and performance under their owning evidence gates. General native heap allocation, complete CPython frame state and isolation-based driver recovery remain open.

Evidence: 428 passed, 1 skipped in host WSL unit gates. Super-Bear assertions-enabled LLVM/MLIR 23.1.1 hardware-free all-target lane: 474 passed, 0 unsupported and 0 failed with CUDA/HIP runtime integration disabled. `scripts/check_lit_fleet_union.py` reports 474/474 active fixtures covered with no unexpected results. `benchmarks/baselines/source_completion_retirement_20260909/` retains the reports and exact compiler/source hashes. Ruff and mypy (306 files) pass.

<!-- entry-fields:end -->

Additional owners: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f),
[W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Sync: `COMPLETION-CORPUS-2026-09-09`.
F0 reconciliation: 72 package functions = 45 Graph/bootstrap inputs + 14 typed
scheduled inputs + 13 raw/unclassified inputs. This is an input inventory, not
artifact-consumption or device proof. Native unary family mappings still require
consumer-by-consumer reconciliation before any Graph constructor deletion.

### 2026-09-09 — Four canonicalization legality improvements

Owner: [W5.5](INTEGRATED_COMPILER_PLAN.md#w55)

PRs: Uncommitted engineering increment.

Outcome: Compose actual transpose permutations and eliminate only identities; fold only rank-two matrix swaps into matmul while preserving bias/residual operands; retain identity casts carrying policy/metadata; refuse epilogue fusion when the replacement cannot carry source attributes. Direct rank-two transpose lowering honors explicit identity permutations. Failed host exception materialization drops unpublished object roots while retaining constructor failure tracebacks.

Remaining: These are legality and data-movement improvements, not tuned GPU candidate promotion. Arbitrary native heap allocation/collection, full native CPython frames and isolation-based driver recovery remain open. The core assertions corpus has zero failures; owning-backend matrix coverage remains open.

Evidence: `canonicalize_permutation_policy.mlir`, existing fusion/transpose fixtures, `test_native_canonical_permutation.py` and `test_source_exception_heap.py`. Super-Bear: 236 focused unit/registry checks passed; Princess-Luna: 13 semantic/heap checks and three MLIR fixtures passed independently. CPU execution distinguishes an identity permutation from a matrix swap on non-symmetric square data. Broader source/operation/dtype checks: 179 passed, 1 skipped. The final all-target assertions corpus passes all 474 active fixtures.

<!-- entry-fields:end -->

Additional owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).
Sync: `CANONICAL-LEGALITY-2026-09-09`.
The generic and custom canonicalizers share transpose interpretation; unknown
metadata refuses simplification instead of silently losing an obligation.
Non-inverse permutations of equal-sized axes no longer cancel by type equality.
### 2026-09-09 — Native exception arena and isolation recovery boundary

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted engineering increment.

Outcome: Native exception allocation/collection and process-isolation recovery foundations implemented; typed native frame records cross the checked exception boundary.

Remaining: Exact CPython frames require interpreter deoptimization; F2 non-f32 matmul/cohort/breadth migrations and the registered W5.2f Schedule/Tile SSD producer remain open.

Evidence: Super-Bear passes 37 focused runtime/audit checks; Princess-Luna passes 26 focused runtime checks.

<!-- entry-fields:end -->

Additional owners: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6),
[W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f).

A C-compatible exception arena now exposes node and
payload pointers, grows without invalidating live handles, and collects
unrooted object graphs including cycles. Transported exceptions retain a typed
native deoptimization-frame record (file, line, function and instruction
slot); no synthetic Python traceback is claimed because CPython provides no
supported API for constructing an exact frame with native locals. A bounded
driver-isolation lease now permits recovery after an uncertain outcome only by
proving death of the owning process/context, and refuses recovery of a healthy
context. CUDA and ROCm host suites each pass the focused 15-case contract set.

The F2 x86 automatic package selector now routes backward attention through
the existing content-addressed Schedule/Tile artifact and saved-LSE descriptor;
its compiled consumer is no longer reachable only by a manual artifact call.
The remaining census identifies non-f32 matmul, cohort and breadth constructors.
The W5.2f source audit
found backend ReplaySSM kernels but no shared Schedule SSD producer or
Schedule-to-Tile consumer; the first correct SSD increment must add that typed
IR boundary before any backend package is promoted.


### 2026-09-10 — Installed drivers and owning-device measurements

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted engineering increment.

Outcome: The compiler-tools install component ships both compiler drivers and the layout shared library. Relative loader paths support prefix relocation. A smoke check copies the installed prefix, removes loader overrides, runs both drivers outside the checkout and translates MLIR to LLVM IR. The opt-in CI lane now consumes both the lit-union and installed-driver checks. Native Metal GELU correctness and attention-backward package timing are measured on M1 Max; native ANN correctness and scoped measurement admission run independently on CUDA SM120 and ROCm gfx1151.

Remaining: CI execution of this revision is pending publication. Measurements cover bounded workloads and complete-call timing; they do not close general backend execution, device-clock attribution or production performance promotion.

Evidence: `benchmarks/baselines/installed_device_gates_20260910/`, `scripts/check_installed_compiler.py`; installed-prefix smoke and all 474 assertions lit fixtures pass on Super-Bear. Two native Metal static/dynamic GELU tests pass on the Mac. Per-device measurement reports retain numerical checks and admission outcomes.

<!-- entry-fields:end -->

Additional owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1).
Sync: `INSTALLED-DEVICE-GATES-2026-09-10`.


### 2026-09-10 — Exception arena reader and payload lifetimes

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Uncommitted continuation after #739.

Outcome: Exported exception ABI views now pin storage through explicit reader leases. Allocation and collection refuse while any lease remains. Partial collection coalesces and reuses dead payload ranges without moving live payloads; repeated transient allocations alongside permanent roots remain bounded.

Remaining: Native arena producer integration, arbitrary object heaps and real CPython frame reconstruction. A caller must retain its ABI lease through native completion; raw addresses must not escape that lease.

Evidence: `test_native_exception_arena.py` pointer/owner retention and repeated partial-collection tests on Super-Bear and Princess-Luna. Addresses are host pointers, not a GPU heap claim.

<!-- entry-fields:end -->

### 2026-09-10 — Isolated native ANN worker and recovery

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted continuation after #739.

Outcome: Native ANN packages can bind in spawned CUDA/HIP workers with worker-owned contexts. One outstanding request carries only copied host tensors. Failed or timed-out completion exposes no result and quarantines ownership; explicit recovery confirms process death before replacement. Error paths exit without destructor-driven driver retries.

Remaining: General package families, heterogeneous dynamic frames, external device readers, concurrent writers and actual uncertain-driver fault testing. Worker IPC is opt-in and is not a performance candidate.

Evidence: `benchmarks/baselines/ownership_census_profile_20260910/`; independent SM120/gfx1151 numerical execution, stopped-idle-worker timeout, confirmed teardown and replacement. SIGSTOP injection is a process/transport fault, not a GPU driver failure.

<!-- entry-fields:end -->

### 2026-09-10 — x86 BF16 scheduled package migration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation after #739.

Outcome: x86 BF16 inputs with f32 accumulation/output now traverse the canonical Schedule/Tile producer. Native packaging projects dtype, byte widths, shape guards, CPU feature requirements and schedule identity, and rejects altered projection/Tile evidence before lowering.

Remaining: uint8/int8 and fp64 matmul, cohort/breadth constructors and target-specific promotion. Existing BF16 runtime kernels remain the physical consumer; no new speedup is claimed.

Evidence: `test_scheduled_matmul_consumers.py` BF16 producer, descriptor/replay tampering and owning-CPU differential cases; exact results recorded in the wave evidence directory.

<!-- entry-fields:end -->

### 2026-09-10 — Route census and device profile attribution

Owner: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f)

PRs: Uncommitted continuation after #739.

Outcome: A reproducible direct-call census keeps Graph-input wrappers separate from native scheduled consumers. The broader 32x8 terminal-square ANN workload is measured independently with Nsight Systems/Compute and rocprofv3.

Remaining: The 64x8 workload exceeds the current 4096-byte temporary bound. Row-parallel lowering still allocates full matrix temporaries per thread; per-row storage projection needs its own ownership proof. CUDA has one-block underutilization and substantial host launch/transfer overhead. ROCm WSL supplies HIP API traces but no kernel/copy timeline in this run; no counter/device-time parity or performance promotion follows.

Evidence: `benchmarks/baselines/ownership_census_profile_20260910/`. Profiler-instrumented timings are diagnostic, not independent promotion measurements.

<!-- entry-fields:end -->

Additional owners: [MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).
Sync for this wave: `OWNERSHIP-CENSUS-PROFILE-2026-09-10`.

### 2026-09-10 — Asynchronous isolation teardown

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted continuation after #739.

Outcome: A bounded off-thread process teardown ticket retains dependent owners and admission slots on uncertain termination. ANN workers and declared process-owned module retirements consume it; host cleanup follows confirmed death. Nonfinite recovery deadlines are refused.

Remaining: Real driver-hang recovery, fresh device-health admission, external device readers and in-process hung unloads are not closed. A failed ticket deliberately retains its slot; operator-level recovery of an unkillable process remains necessary. Module filesystem cleanup during completion polling is still synchronous.

Evidence: focused isolation/ANN/module ownership tests and independently measured SM120/gfx1151 stopped-worker replacement in `benchmarks/baselines/async_isolation_20260910/`. No GPU hang was injected and no performance promotion follows.

<!-- entry-fields:end -->

### 2026-09-10 — Device health admission and external readers

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted continuation after #739.

Outcome: Fresh isolated ANN workers must execute deterministic bounded numerical probes against the analytic oracle before announcing readiness. Replacement requires confirmed failed-worker teardown and repeats admission. A stuck health probe is bounded by the parent deadline. Multi-stream read scopes record every declared consumer, including exception paths. Eventless dependencies no longer silently synchronize the host; completion needs an explicit wait or isolated recovery.

Remaining: These probes establish workload/device-zero readiness at admission, not global driver health. Real wedged-driver recovery, unkillable kernel-mode processes, arbitrary device selection and unsupervised raw-pointer consumers remain open. External consumers must enqueue reads on declared streams and not retain pointers outside the scope. No automatic GPU reset or performance promotion is introduced.

Evidence: `benchmarks/baselines/health_reader_admission_20260910/` records independent SM120/gfx1151 health/replacement and two-stream native copy consumers across four tape generations. Host tests cover numerical rejection, a hung probe, partial reader failures and retained ownership.

<!-- entry-fields:end -->

### 2026-09-10 — Recovery retry and caller-error boundaries

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: #740 review follow-up.

Outcome: Missing dependency events refuse before retirement commits state, preserving retry after explicit completion. Invalid ANN shape/domain inputs refuse in the parent without poisoning a healthy worker. Leaving a scope with pending work poisons and tears down the worker; uncertain cleanup retains ownership and preserves the caller's original exception.

Remaining: Actual wedged-driver recovery and unrestricted external-reader lifetimes remain open; these fixes establish host lifecycle behavior.

Evidence: focused reader-retirement and isolated-ANN regressions cover retry after a missing event, repeated invalid inputs followed by successful execution, pending context exit and failed cleanup with exception preservation.

<!-- entry-fields:end -->

### 2026-09-10 — Expanded route callers and f64 ownership

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation after #740.

Outcome: FP64 x86 matmul gains a native Schedule producer/verifier, complete storage/accumulation/output projection and the existing native fp64 consumer. Floating Graph constructor branches are removed; mixed-signedness VNNI remains. The caller census now includes x86 breadth, import aliases and local helper-to-emitter paths.

Remaining: F0 certificate reconciliation and dynamic call resolution, mixed uint8/int8 matmul, cohort/elementwise/breadth migration. Tracks 3–10 retain their prior gates; no new driver-hang recovery, heap producer, CPython deoptimization, AD/F3/SSD closure or performance promotion follows from this increment.

Evidence: `benchmarks/baselines/f64_route_ownership_20260910/`; fp64 compiler fixture and owning-CPU projection/numerical tests in `test_scheduled_matmul_consumers.py`. Registry and ABI tests preserve compiler-free coverage.

<!-- entry-fields:end -->

Additional owner: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f).

### 2026-09-10 — Dynamic reader fanout and row-private ANN

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted continuation after #740.

Outcome: Checked dynamic public results and paired source-VJP frames expose multi-stream scoped readers. Host regressions preserve both products on external failure and close every acquired reader. The existing native row-independence proof now authorizes compacting entry-owned mutable ANN temporaries to one row per thread, preserving complete constants and nested generation slots. Internal SSA proof sets, not user attributes, authorize compaction.

Remaining: General package isolation, real wedged-driver recovery, native heap producers, CPython deoptimization, general heterogeneous saved-product generation and relational alias proofs, attention raising and shared tiled SSD remain open. This increment does not substitute host reader tests for owning-device frame proof. ROCm device counters and broader/multi-block tuning remain open.

Evidence: 64x8 square ANN numerical oracles pass independently on SM120/gfx1151; single-run rewrite speedups about 0.991x/0.994x retain incumbents. CUDA Nsight separates kernel, transfer and API costs. See `benchmarks/baselines/ann_row_private_20260910/`. No promotion or direct old/new optimization speedup is claimed.

<!-- entry-fields:end -->

Additional owners: [MSW-9](INTEGRATED_COMPILER_PLAN.md#msw-9), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — SSD recurrence and retryable completion

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #740.

Outcome: A native registered internal `schedule.ssd` and Python producer carry X, multiplicative decay, B/C, immutable initial carry, Y, final carry and chunk-end checkpoints. Schedule-to-Tile lowers the verified static f32 recurrence to structured tensor loops; native CPU tests cover partial chunks and input preservation. Artifact validation replays the incoming Schedule with the recorded compiler identity. Failed asynchronous recovery tickets can reconcile late process death without retrying termination; concurrent polls release owners/slots once. Paired VJP retirement resumes only unsubmitted children. Arena edge/root APIs permit actual cycles and preserve leased storage. Exception completion restores interpreter-owned fields without user assignment hooks.

Remaining: SSD public/frontend integration, tiled/cooperative kernels, checkpoint adjoints, ReplaySSM comparison and target packages; native exception allocation producers, general CPython deoptimization/frames, general saved products and attention raising remain open. Process exit proves resource isolation teardown, not driver health or recovery from a wedged driver. No new GPU or performance promotion claim.

Evidence: `tests/unit/test_scheduled_ssd.py`, `tests/tessera-ir/phase2/ssd_schedule.mlir`, `tests/unit/test_native_driver_isolation.py`, `tests/unit/test_native_public_results.py`, `tests/unit/test_native_exception_arena.py`, `tests/unit/test_source_exception_heap.py`. Assertions-enabled host compiler and CPU JIT; backend evidence boundaries remain independent.

<!-- entry-fields:end -->

Additional owners: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a), [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).

### 2026-09-10 — Native heap, attention recipes and SSD device proof

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #740.

Outcome: A bounded native C++ exception heap owns allocation, explicit roots, cause/context cycle collection and payload-hole reuse through generation-checked handles and copy-out reads. An exact dense f32 attention loop recognizer retains its source oracle and produces a native symbolic recipe; two buckets lower through Schedule and Tile with head-width witness validation. Serial replay-bound SSD packages execute on CUDA SM120 and ROCm gfx1151. Device tests exposed initial-carry mutation during bufferization; immutable input ownership is now projected before bufferization. The standalone runtime shared build also exposed duplicate disabled CUDA/HIP factories in the CPU translation unit; their owning backend files now supply them exclusively.

Remaining: Automatic source-IR heap producers, GPU heap allocation, general exception objects and native CPython deoptimization records/frames remain open. Attention needs broader patterns and exact-device candidate admission. SSD needs cooperative kernels, public frontend and checkpoint AD integration, ReplaySSM comparison and measured promotion. No performance promotion is claimed.

Evidence: `tests/unit/test_native_exception_producer.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_native_ssd.py`, the native runtime ABI smoke and `benchmarks/baselines/native_heap_attention_ssd_20260910/`. SSD records cover chunks 1, 2 and 5 on each owning device and verify all five inputs remain unchanged.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1).

### 2026-09-10 — Heap IR, raised attention and cooperative SSD

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: Static source exception tables emit checked native allocation/root/edge calls through MLIR/LLVM, with a native source-state decoder consumer and explicit generated-library lifetime. Native attention buckets project their Schedule descriptor and bind NVIDIA execution without Graph reconstruction. Cooperative SSD uses block-owned head/value columns, lane-owned state, shared-memory barriers and an ordered leader reduction. CUDA and ROCm correctness covers immutable inputs, outputs, final carry and chunk checkpoints; profiling separates resident event windows, checked-call preparation and API/kernel traces. The assertions build caught a missing Tessera dialect dependency in Schedule-to-Tile, now declared.

Remaining: Runtime-valued/iteration-site exception allocation, GPU heaps and full CPython frames; broader attention patterns, sibling-device bindings and arbiter promotion; SSD public integration, checkpoint AD, more efficient reduction/tiling, ReplaySSM comparison and selector-grade measurements. ROCProfiler exposed HIP API records but no GPU kernel/copy trace records on this host, so hardware-counter attribution remains open. No promotion claim.

Evidence: `tests/unit/test_native_exception_producer.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_native_ssd.py`, `benchmarks/record_ssd_gpu.py` and `benchmarks/baselines/heap_attention_cooperative_ssd_20260910/`. Backend queue historical increments are marked superseded rather than left as competing current obligations.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Dynamic heap payloads, checkpoint AD and paired measurements

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: Runtime numeric exception payloads are copied through native pointer/size operands into a fresh bounded heap; changing shapes and failed allocations preserve ownership and retryability. The native source-state exception decoder consumes the same path. Raised attention now projects x86 and Apple parents, with executable x86 proof on Princess-Luna and no Graph reconstruction. The SSD native CPU checkpoint VJP accepts cotangents for output, final carry and checkpoints, produces all five input gradients, and recomputes from chunk boundaries. Its scoped forward/VJP owner keeps checkpoints private to the corresponding forward call. Nine fixed independent process pairs on each GPU compare exact serial/cooperative artifacts in alternating order.

Remaining: In-kernel/iteration-site heap allocation, GPU exception heaps and general objects; broader attention recognition, Apple owning-device validation and ROCm low-precision binding; automatic public mixer AD, GPU checkpoint backward, persistent/asynchronous checkpoint ownership and ReplaySSM comparison. Paired resident event windows include submission gaps; they do not supply calibrated device clocks or selector admission. No route promotion.

Evidence: `tests/unit/test_native_exception_producer.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_scheduled_ssd.py`, `tests/unit/test_ssd_comparison.py`, `benchmarks/compare_ssd_variants.py`, and `benchmarks/baselines/ssd_paired_process_20260910/`. Numerical VJP checks use finite differences of every input with nonzero seeds for every result, including partial chunks.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — GPU payload frames, mixer AD and artifact selection

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: Native GPU exception-payload producers consume serialized numeric-site metadata and allocate bounded frames transactionally, publishing generation/offset/length records only on success. CUDA and ROCm prove overflow/negative-length refusal, unchanged storage on failure, empty payloads, subsequent success and stale-generation decoder rejection. The native SSD checkpoint VJP now packages for both GPUs, consumes actual cooperative-forward checkpoints and matches finite differences for all five inputs with nonzero output/carry/checkpoint cotangents. SSDCheckpointProgram also connects native CPU Y to automatic host-tape grad with private forward snapshots. Attention recognition accepts one positive f32-exact post-dot literal scale while retaining full AST matching; two CUDA buckets execute scaled/unscaled variants. Explicit measured SSD binding replays both packages, recomputes paired bounds, binds calibration to exact images/durations, and rebuilds the existing ROCm policy instead of trusting eligibility flags.

Remaining: GPU frames are preallocated single-writer numeric storage with caller-supplied generations, not arbitrary allocation or a concurrent object collector. Automatic source throw-site integration and reader-aware frame reclamation remain open. Automatic GPU-resident mixer AD, tuned backward/cooperative checkpoint kernels, higher-order products and public mixer family wiring remain open. Attention masks/GQA/broader recognition and sibling exact-device scaled proof remain open. The actual CUDA and ROCm selector calls retain incumbents: CUDA lacks a typed native calibration adapter, and ROCm lacks per-process eligible calibration (the current policy also requires bare metal). No promotion.

Evidence: `tests/unit/test_gpu_exception_heap.py`, `tests/unit/test_scheduled_ssd.py`, `tests/unit/test_native_ssd.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_ssd_comparison.py`, `benchmarks/record_gpu_heap_ssd_ad.py`, `benchmarks/check_ssd_admission.py`, and `benchmarks/baselines/gpu_heap_ssd_ad_20260910/`. Production package identities still match the prior nine-pair measurements.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Reusable GPU pools, resident AD and window calibration

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: Bounded native GPU pools allocate fixed payload slots and run stop-the-world mark/sweep with runtime root/edge graphs, generation checks, partial reuse and unrooted cycle collection. CUDA SM120 and ROCm gfx1151 execute the same semantic regressions with independent packages. ResidentSSDProgram generates paired forward/VJP packages, owns device snapshots and checkpoints, differentiates Y into all five resident gradients, and closes scoped buffers after synchronous completion. CUDA attention recognition now proves explicit GQA indexing and end-aligned causal masks for both longer-Q and longer-K buckets; native recipe instantiation enforces positive divisible head counts. CUDA SSD admission rebuilds a per-process Nsight/event-window calibration instead of requiring an absent adapter.

Remaining: Pools are bounded, fixed-width, single-writer numeric storage with two outgoing edges and quiescent readers; concurrent collection, automatic throw-site integration and arbitrary object/payload allocation remain open. Resident AD is synchronous first-order family integration, not general public tape composition, external-reader retirement or tuned GPU backward. General additive/padding masks, broader attention idioms and sibling device admission remain open. The measured CUDA calibration agrees within 2.26% but has 1.957x profiler overhead, dirty source and WSL execution, so promotion refuses. Eighteen eligible per-process calibrations and renewed exact-artifact comparisons remain required; prior packets retain their original compiler identities. ROCm bare-metal dispatch/counter calibration remains open.

Evidence: `tests/unit/test_gpu_heap_collection.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_cuda_window_calibration.py`, `benchmarks/record_pool_resident_ssd.py`, `benchmarks/calibrate_ssd_cuda.py`, and `benchmarks/baselines/pool_resident_gqa_calibration_20260910/`.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Stream-owned object graphs, asynchronous AD and window legality

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: ResidentObjectPool stores bounded opaque byte records with configurable generation-checked reference edges. NativeStreamEpoch permits scoped readers on multiple streams and serializes graph updates/allocation/collection after their events. Open reader scopes refuse mutation; failed event recording retains an eventless dependency until explicit completion. CUDA/HIP validate cyclic graphs, fourth-edge reachability, partial collection, byte payload preservation and generation reuse. SSD asynchronous VJPs now compose through projected scoped gradients and stream-ordered derivative retirement on both devices. Capture still snapshots synchronously. Windowed GQA recognizes bounded asymmetric slices and carries a mandatory Presburger nonempty-row condition through native recipe instantiation. SSD packaging no longer imports the 1024-element persistent-tape shape parser; its checked f32 storage bound is 64 MiB per buffer, validated with a 512x2x32x8 cooperative workload on both GPUs. CUDA calibration now binds process nonces and the Nsight PID, and a nine-pair collector preflights clean/bare-metal eligibility.

Remaining: Collection is reader-coordinated stop-the-world mutation, not simultaneous mutator/collector execution or arbitrary CPython graph discovery. Variable-size allocation, automatic exception producer integration and general object semantics remain open. AD is an explicit first-order SSD family API; general public tape integration, asynchronous capture/whole-frame teardown, higher-order products and optimized backward remain open. Additive/padding/general masks and sibling-device window admission remain open. The new CUDA capture passes clock agreement (0.51%) and overhead (0.24%) gates but remains dirty/WSL evidence. Eighteen eligible independent calibrations plus renewed exact-artifact comparison are still required for promotion; ROCm needs bare-metal profiler evidence. No promotion claimed.

Evidence: `tests/unit/test_native_stream_epoch.py`, `tests/unit/test_gpu_heap_collection.py`, `tests/unit/test_attention_loop_idiom.py`, `tests/unit/test_cuda_window_calibration.py`, `benchmarks/record_async_pool_ad.py`, `benchmarks/record_ssd_calibrated_pairs.py`, and `benchmarks/baselines/async_objects_windows_20260910/`.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Snapshot marking, public VJP and additive bias

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #741.

Outcome: Private GPU graph snapshots permit marking on a separate stream while active graph updates continue. Final seeded remark/sweep is exclusive and retains snapshot survivors conservatively; malformed seeds fail closed. Hook-free discovery handles exact builtin containers and plain instance dictionaries with cycles/aliases. Public vjp dispatches an explicitly owned resident SSD program; asynchronous capture retains inputs through copy completion, backward composes scoped gradients, and whole-frame retirement orders all forward/derivative readers before frees. Failed event dependencies keep retirement retryable. Full-shape finite additive attention bias now survives recognition, native recipe instantiation, Schedule projection and NVIDIA execution.

Remaining: Concurrent sweeping, lock-free producers, arbitrary extension heaps, concurrent Python discovery, variable-size storage and automatic exception-site fusion remain open. Public AD integration is first-order SSD protocol dispatch, not arbitrary traced GPU programs or higher-order AD. Program/module close may synchronize. General Boolean/padding/broadcast masks and fully masked-row semantics need further contracts. CUDA/HIP tests run on owning GPUs under WSL: they do not establish physical overlap or eligible performance. Clean bare-metal per-process evidence remains required; no promotion.

Evidence: `tests/unit/test_object_discovery.py`, `tests/unit/test_resident_ssd_ownership.py`, `tests/unit/test_gpu_heap_collection.py`, `tests/unit/test_attention_loop_idiom.py`, `benchmarks/record_snapshot_public_ad.py`, and `benchmarks/baselines/snapshot_public_ad_20260910/`.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1), [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Benchmark compiler alignment

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: Uncommitted continuation after #741.

Outcome: Reviewed math, linalg, energy/Clifford compositions, eight AD scripts and the operator harness against their actual callers. Consolidated navigation in benchmarks/COMPILER_ALIGNMENT.md; registered the missing math/autodiff manifest entries with syntax-only compile_only checks. New physical-math results require observed native execution and finite correctly shaped outputs; linalg enforces residual bounds and emits clean JSON stdout. GA/EBM rows now label unattributed library execution instead of claiming CPU reference placement despite optional Apple fast paths. New math and selected AD packets do not inherit performance eligibility from target names or historical packet paths. Historical evidence is unchanged; no executable suite is archived because each retains consumers or a distinct oracle.

Remaining: Shared evidence adapters, exact-package math ancestry, per-call GA/EBM route attribution, target-aware operator harness adapters, and public frontend counterparts for direct-IR AD probes. No fresh performance measurements or promotion. Clean bare-metal exact-artifact comparisons remain required.

Evidence: `benchmarks/COMPILER_ALIGNMENT.md`, `tests/unit/test_physical_math_evidence.py`, `tests/unit/test_benchmark_surface_repair.py`, `tests/unit/test_clifford_core_benchmark.py`, `tests/unit/test_solver_ift_evidence.py`, and `tests/unit/test_operator_benchmarks_contract.py`.

<!-- entry-fields:end -->

Additional owner: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Extended benchmark alignment

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: Uncommitted continuation after #741.

Outcome: Extended the benchmark alignment review across autodiff, DLOP, SuperBench, DeepScholar, grid-AI, lattice reasoning, visual-complex, RL and E2E spine. Retired the synthetic attention timer without removing compatibility wrappers. DLOP dispatch counts are explicitly static estimates; lattice Apple-call durations now populate host-wall rather than kernel timing. Grid/visual compositions report unattributed library execution. Every timed Apple policy submission now requires native execution and finite output. Four missing manifest entries now have syntax-only CI checks. E2E recorders, historical packets and distinct reference oracles remain intact.

Remaining: Profiler-backed dispatch receipts, native SuperBench package adapters, automatic public AD counterparts and full model/distributed execution proof. No device packet was re-sealed, no hardware speedup measured and no performance promoted.

Evidence: `benchmarks/COMPILER_ALIGNMENT.md`, `tests/unit/test_benchmark_alignment_extended.py`, `tests/unit/test_rl_policy_loss_benchmark.py`, and the affected core/schema/orchestration tests.

<!-- entry-fields:end -->

Additional owner: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).

### 2026-09-10 — Native benchmark adapters

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: Uncommitted continuation after #741.

Outcome: Added native CUDA/HIP SuperBench ANN adapters and a separate DLOP observed-dispatch lane sharing checked package execution. Three ANN workloads pass on RTX 5070 and gfx1151; original and transformed variants each produce one accepted driver launch per measured call. Added public resident SSD VJP comparisons for three shapes, two cotangents and all five inputs against independent float64 central differences; maximum gradient error is below 1.2e-8 on both hosts.

Remaining: Driver API receipts are not profiler kernel records. Broader GEMM/attention adapters, native mappings for the original DLOP catalog, general public frontend AD and clean bare-metal performance promotion remain open. Instrumented WSL host-wall timings are diagnostic only; no promotion or dispatch-reduction claim.

Evidence: `benchmarks/baselines/native_benchmark_adapters_20260910/`, `benchmarks/native_ann_adapter.py`, `benchmarks/autodiff/benchmark_public_ssd.py`, `tests/unit/test_native_benchmark_receipts.py`.

<!-- entry-fields:end -->

Additional owners: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1), [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f).

### 2026-09-10 — Broader adapters and kernel attribution

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: Uncommitted continuation after #741.

Outcome: Expanded native SuperBench configs to baseline/larger ANN and serial/cooperative SSD; all four execute correctly on SM120 and gfx1151. Preserved separate host-call and device-event timing domains. A dedicated CUDA SSD capture has 701 profiler kernels matched one-to-one to successful owning-process launch API records, tied to the packet's artifact/image identity. Attribution tests reject wrong processes, failed launches, wrong kernel names and missing/duplicate correlations.

Remaining: General mixed-artifact range attribution, native GEMM/attention adapters and clean performance promotion. ROCm's fresh kernel/copy-enabled trace still provides only HIP API and agent records on WSL, so owning-device kernel attribution remains open. No timing or execution evidence transfers to Apple/x86.

Evidence: `benchmarks/baselines/broader_adapters_20260910/`, `benchmarks/attribute_ssd_cuda.py`, `tests/unit/test_ssd_profiler_attribution.py`, and both native SuperBench suite results.

<!-- entry-fields:end -->

Additional owners: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1), [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f).

### 2026-09-10 — Matrix adapters and mixed attribution

Owner: [EVIDENCE-PACKET-1](INTEGRATED_COMPILER_PLAN.md#evidence-packet-1)

PRs: Uncommitted continuation after #741.

Outcome: Added scheduled native GEMM/attention adapters with descriptor-projected input layout and independent NumPy oracles. CUDA executes both. Mixed profiler ranges bind run, artifact and native image identities and correlate successful launch APIs to kernels. Tests reject wrong image identity, missing/duplicate correlations, overlapping ranges, failed APIs and escaping completion.

Remaining: ROCm GEMM/attention both abort at LLVM GPUFuncOpLowering's duplicate DictionaryAttr assertion with the installed assertions-enabled compiler; no ROCm execution result claimed. Asynchronous/multi-thread attribution, broader matrix envelopes and clean performance promotion remain open. Native ROCm config retains failing rows rather than silently demoting.

Evidence: `benchmarks/baselines/matrix_mixed_20260910/`, `benchmarks/native_matrix_adapter.py`, `benchmarks/attribute_mixed_cuda.py`, `tests/unit/test_mixed_profiler_attribution.py`.

<!-- entry-fields:end -->

Additional owner: [TPROF-NATIVE-1](INTEGRATED_COMPILER_PLAN.md#tprof-native-1).


### 2026-09-11 — Program retirement and declared-slot discovery

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: Uncommitted continuation after #742; preserves the SSD numerical-admission repair.

Outcome: Resident SSD now exposes program-level retire_async/poll_close: reader preflight precedes frame frees, partially queued retirement can resume, new captures refuse, and modules unload off-thread only after frame completion. CUDA SM120 and ROCm gfx1151 validate two public-VJP frames while context synchronization is forbidden on the program path. Opt-in declared-slot discovery preserves inherited aliases/cycles through native descriptors and refuses custom descriptors, undeclared dictionaries and opaque layouts; a slotted self-cycle survives GPU collection on both hosts.

Remaining: Arbitrary traced GPU tapes, extension heaps, variable-size device payloads, concurrent sweeping, asynchronous owner adoption and general masks remain open. A mutation publication barrier plus generation/reader-aware reclamation must precede concurrent sweep. Module workers bound admission and polling, not the driver's own latency or failure recovery.

Evidence: `benchmarks/baselines/program_retirement_20260911/`, `benchmarks/record_program_retirement.py`, `tests/unit/test_resident_ssd_ownership.py`, `tests/unit/test_object_discovery.py`; 54 focused host tests, Ruff and zero-error mypy on WSL.

<!-- entry-fields:end -->

Additional owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

The requested ten-track list is reconciled to existing owners, not a new status registry:

- F0/F2 retain caller/envelope certificate joins and the remaining mixed-integer/cohort/breadth constructors. Scheduled non-f32 GEMM already exists; ROCm matrix native lowering currently exposes an assertions-enabled LLVM failure. No constructor is removed by this increment.
- Isolation recovery already launches bounded native ANN packages; actual driver-hang/device-health recovery remains separate from stopped-worker teardown proof. Scoped dynamic/external readers exist; general heterogeneous capture and failure ownership still need work.
- Native exception allocation/root producers and bounded cycle collectors exist. Automatic throw-site integration, arbitrary heaps, concurrent reclamation and complete CPython deoptimization/frame reconstruction remain open.
- Persistent SSD AD/checkpoints and native attention recipes/buckets exist. General traced effectful AD, heterogeneous/aliased products, Boolean/padding/broadcast masks and fully masked-row behavior remain owner-specific acceptance gates.
- Larger ANN/SSD measurements and CUDA mixed-artifact attribution exist; they do not grant clean promotion or ROCm hardware-counter evidence. The shared registered Schedule SSD family is implemented; next work is frontend integration, tuning and selector-grade proof, not starting another SSD operation.


### 2026-09-11 — Incremental sweeping and traced SSD composition

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`); includes the slotted-module identity correction.

Outcome: Incremental GPU collection retires a complete unreachable cohort before range reclamation, preserving generation-safe reuse and avoiding dangling dead-cycle edges between batches. Explicit exact-type extension extractors copy schema-tagged bytes/referents under caller-owned quiescence. Linear ResidentSSDTrace composition builds its call graph before GPU submission, generates reverse traversal and projects scoped gradients back to all public inputs. NVIDIA additive-mask admission now accepts negative infinity but refuses fully masked rows after causal/window intersection.

Remaining: Sweep kernels still exclude writers/readers; simultaneous sweeping needs atomic publication and read barriers. Extension callbacks are explicitly trusted, not automatic arbitrary-heap discovery. GPU tracing is host orchestration of independently replayed packages; canonical composition IR/fusion remains open. It admits uniquely consumed linear SSD calls only: branches, accumulation, reused inputs, effects and additional families remain open. Boolean/broadcast masks and defined empty-row outputs need compiler/ABI work. No promotion or Apple/x86 device proof is claimed.

Evidence: `benchmarks/baselines/incremental_heap_trace_20260911/`, `benchmarks/record_incremental_heap_trace.py`, `tests/unit/test_resident_trace.py`, `tests/unit/test_resident_pool_sweep.py`, `tests/unit/test_object_discovery.py`, `tests/unit/test_attention_loop_idiom.py`.

<!-- entry-fields:end -->

Additional owners: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1).

The ten-track reconciliation above remains the ownership map. This increment
advances the heap, persistent-product/AD and attention tracks; it does not close
F0/F2 census/migrations, isolated driver recovery, CPython reconstruction or
performance admission. The shared tiled-SSD family already exists.


### 2026-09-11 — Resident DAG accumulation and snapshot readers

Owner: [W5.2f](INTEGRATED_COMPILER_PLAN.md#w52f)

PRs: This change set (`codex/gated-heaps-resident-ad`), extending the preceding incremental sweep work.

Outcome: Traced SSD DAGs now admit fan-out and repeated public inputs across calls; replay-validated native f32 addition accumulates every cotangent path in a fixed order. Addition outputs and modules share frame/program retirement. CUDA/HIP validate a three-call DAG against independent float64 finite differences. Immutable pool snapshots hold copied epochs for readers while the live pool sweeps; parent close retains all snapshots through reader completion. Discovery adds exact general-key maps, sets, frozensets and bytearrays, and refuses undeclared builtin-subclass payloads and custom metaclasses before invoking user hooks. CUDA validates irregular additive masks composed with ragged GQA, causal and window masks in both Q/K size directions.

Remaining: Snapshot isolation does not permit racing arbitrary readers/writers over the same storage. Unrestricted sweep still needs publication/read barriers and generation-safe reclamation. Arbitrary extension discovery still requires a declared trusted layout; opaque native storage is not guessed. Tracing remains host orchestration of native SSD packages, not canonical whole-program IR, effectful control-flow AD, general operation families, or same-call alias admission. Boolean/broadcast masks and defined empty-row outputs remain open. No promotion.

Evidence: `benchmarks/baselines/dag_snapshot_20260911/`, `benchmarks/record_dag_snapshot.py`, `tests/unit/test_resident_gradient_sum.py`, `tests/unit/test_resident_trace.py`, `tests/unit/test_resident_pool_snapshot.py`, `tests/unit/test_object_discovery.py`, `tests/unit/test_attention_loop_idiom.py`.

<!-- entry-fields:end -->

Additional owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1).

### 2026-09-11 — Heap publication and reader barrier exploration

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: Inventoried exclusive epochs, immutable snapshots, ordinary heap loads/stores and Tile completion domains; selected a staged nonmoving design for modeling.

Remaining: Executable interleaving model, consumed artifact contract, epoch retirement, barrier-aware updates and independent device/performance gates remain unimplemented.

Evidence: [Source-linked architecture review](HEAP_BARRIER_ARCHITECTURE_REVIEW.md); source inspection only, no new device result.

<!-- entry-fields:end -->

Reader ownership routes to [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a);
uncertain completion routes to [DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker).
Sync: `HEAP-BARRIERS-2026-09-11`. Final remark remains exclusive; memory ordering,
reachability and reclamation are independent obligations.

### 2026-09-11 — Bounded heap protocol and device comparison

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: Added a bounded interleaving model, consumed v1 artifact protocol, split retirement/reclamation and validated graph publication. CUDA/HIP independently execute the new boundaries.

Remaining: Per-object epochs, dirty-work barriers, concurrent final retirement, atomic scope lowering and selector-grade performance remain open. Final remark and whole-pool writes remain exclusive.

Evidence: [Independent packets and comparison](../../../benchmarks/baselines/heap_barriers_20260911/README.md); 69 focused host-WSL tests passed, Ruff passed and mypy ratchet remains zero.

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns reader-protected reuse;
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owns failure
retention. The model finds a counterexample when reclamation dependencies are
removed. Device packets check stale publication, newly rooted cycles and
multi-stream payload copies before reuse. Five single-process samples show
higher cost for split retirement; no promotion. Sync: `HEAP-BARRIERS-2026-09-11`.

### 2026-09-11 — Per-object readers and incremental marking

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: V2 native generation-checked pins, consumed tricolor marking barriers and final retirement during immutable payload-reader scopes execute on CUDA SM120 and ROCm gfx1151. Unrelated unpinned slots reuse while another reader remains admitted.

Remaining: Metadata writers are still exclusive. Asynchronous admission/unpin, allocation during marking, cooperative marking and an atomic multi-writer handshake remain open; no measured overlap or promotion.

Evidence: [Independent packets and model](../../../benchmarks/baselines/incremental_object_heap_20260911/README.md); model 6,690 states / 64,536 transitions; 77 focused host-WSL tests passed.

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns pin/completion lifetime;
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owns uncertainty.
Closed reader scopes remain pinned until completion; unknown decrements retain
the pool and cannot be retried. Metadata-scope refusal before submission stays
retryable. Sync: `HEAP-INCREMENTAL-2026-09-11`.


### 2026-09-11 — Asynchronous heap receipts and marking allocation

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: Optional private polled admission/unpin receipts and grey allocation publication execute independently on CUDA SM120 and ROCm gfx1151. Cancelled admission releases its pin after completion; stale admission does not poison the pool.

Remaining: Metadata writers remain serialized. The reservation model exposes the split validate/write race; native atomic reservation, retirement handshake, cooperative marking and asynchronous finalization/teardown remain open. No measured overlap or promotion.

Evidence: [Device packets and limitations](../../../benchmarks/baselines/async_heap_20260911/README.md).

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns receipt and reader lifetime;
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owns retention
after unknown copy/decrement outcomes. Sync: `HEAP-ASYNC-2026-09-11`.


### 2026-09-11 — Native heap handshake and deferred destruction

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: Schema-3 native graph/retirement try-lock kernels, private polled mark finalization and bounded context-owning teardown workers pass independent CUDA SM120 and ROCm gfx1151 checks.

Remaining: All metadata users must join the gate before removing resident-pool epoch ordering. Cooperative marking, finer transactions and isolated recovery after uncertain teardown remain open. Worker driver calls may block; no measured overlap or promotion.

Evidence: [Device packets, model and limitations](../../../benchmarks/baselines/heap_handshake_20260911/README.md); 77 focused host-WSL tests passed.

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns reader and receipt lifetimes;
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owns retained
teardown failures. Sync: `HEAP-HANDSHAKE-2026-09-11`.


### 2026-09-11 — Gated metadata owner and isolated recovery

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: This change set (`codex/gated-heaps-resident-ad`).

Outcome: Opt-in gated owner routes all admitted metadata operations through its owned gate and exposes copied inspection. Spawned CUDA/HIP heap workers support death-confirmed recovery with bounded retained capacity.

Remaining: Legacy snapshot/import migration, broader isolated commands, health-checked replacement, actual driver-hang validation and epoch relaxation remain open. The existing incremental owner is not silently migrated. No measured overlap or promotion.

Evidence: [Independent CUDA/HIP packets and limitations](../../../benchmarks/baselines/gated_heap_20260911/README.md).

<!-- entry-fields:end -->

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns pinned readers and copied metadata;
[DISPATCH-BREAKER](INTEGRATED_COMPILER_PLAN.md#dispatch-breaker) owns confirmed
process death and retained failures. Sync: `HEAP-GATED-ISOLATION-2026-09-11`.

### 2026-09-11 — Mixed integer ownership and dtype arithmetic

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted engineering increment after #744.

Outcome: Mixed x86 u8/s8 matmul now consumes a replay-verified Schedule/Tile artifact. Scalar reference accumulation uses defined modulo arithmetic. CUDA and HIP each execute 24 basic scalar/vector dtype rows with instruction witnesses.

Remaining: Cohort/breadth constructors; FP8 generic conversion legalization; packed/scaled, complex, bool and Apple arithmetic probes; broader layout and numerical consumers; selector-grade performance evidence.

Evidence: [dtype arithmetic and matrix packets](../../../benchmarks/baselines/dtype_arithmetic_20260911/README.md), `test_scheduled_matmul_consumers.py`, `test_x86_integer_wraparound.py`, `test_dtype_arithmetic_probe.py`.

<!-- entry-fields:end -->

Related owners: [E2E-REAL-6F](INTEGRATED_COMPILER_PLAN.md#e2e-real-6f),
[NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1), and
[LAYOUT-ALG-1](INTEGRATED_COMPILER_PLAN.md#layout-alg-1). The F0 caller census
separates lexical scopes, parameter/rebinding shadowing and relative package
imports. The dtype inventory retains all 26 canonical/planned storage names;
missing layered declarations do not mean no operation-specific implementation.

### 2026-09-11 — FP8 conversions and numerical boundaries

Owner: [NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1)

PRs: Uncommitted engineering increment after #744.

Outcome: Byte-backed scalar/vector FP8 conversions legalize to integer/f32 operations. Exhaustive CUDA/HIP arithmetic, bool logic and bounded complex probes pass. NVIDIA packed/scaled probes exercise nonzero codes and both axes. ROCm packed producers use inherent kernel properties and compiler serialization no longer depends on host runtime linkage.

Remaining: Apple strict subnormal arithmetic fails three comparisons; broader storage, numerical/layout consumers, matrix admission and clean performance evidence remain open.

Evidence: [independent dtype follow-through](../../../benchmarks/baselines/dtype_followthrough_20260911/README.md), `test_lowp_conversions.py`, `test_dtype_arithmetic_probe.py`, `test_packed_dtype_probe.py`.

<!-- entry-fields:end -->

### 2026-09-12 — Reconciliation and wave start

Owner: [NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1)

PRs: Uncommitted follow-through after #745.

Outcome: MASTER_AUDIT now routes sequencing to the live plan and distinguishes implemented SSD, MPI, checkpoint and assertions-enabled toolchain slices from their open boundaries. CUDA serialization now accepts an explicit toolkit path; the arithmetic recorder requires it for NVIDIA and fingerprints ptxas and the disassembler. All 34 scalar/vector rows pass on Super-Bear with CUDA SDK 13.4.1. Snapshot retirement closes admission and polls recorded readers without implicitly synchronizing or freeing.

Remaining: Apple gradual-underflow policy closure, the selected x86 absolute migration and serialized broadcast attention masks are scoped, not implemented in this increment. Snapshot frees and module unload remain synchronous; no new owning-device snapshot-retirement or performance proof is claimed.

Evidence: [CUDA arithmetic packet](../../../benchmarks/baselines/cuda1341_20260912/README.md); WSL tests cover toolkit selection, audit governance, reader admission, eventless retention and snapshot recovery.

<!-- entry-fields:end -->

Additional owners: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6),
[FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1) and
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a). The live records hold acceptance gates.

### 2026-09-12 — Native numerical and packaging slices

Owner: [NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1)

PRs: Uncommitted continuation after #745.

Outcome: Explicit Apple arena gradual/FTZ policy now consumes integer-significand f32 add/sub/mul/div. Both policies pass M1 Max checks over 133,376 boundary/random input pairs per operation. x86 absolute now uses a native Schedule record and Tile producer with replay-derived descriptors and owning-Zen-5 exceptional/ragged proof. Batch/head broadcast attention masks reach native SM120 indexing; the versioned host ABI copies the physical bias extent, and two ragged causal/GQA/window cases pass.

Remaining: Wider Apple floating operations/dtypes/vectors and optimized policy performance; other Graph-owned elementwise/cohort/breadth families; query/key broadcast, Boolean/padding masks and sibling attention backends. No general AD or performance closure is implied.

Evidence: [native slice packets](../../../benchmarks/baselines/native_slices_20260912/README.md), [Apple numerical contract](../../spec/APPLE_ARENA_NUMERICAL_POLICY.md), focused native/compiler and registry checks, and the new absolute lit fixture.

<!-- entry-fields:end -->

Additional owners: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6) and
[FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1).
The first native-hardware FTZ experiment failed 28 multiply/divide comparisons;
FTZ admission was corrected to use IEEE rounding before output flushing. The
passing packet is from the corrected implementation, not a tolerance change.

### 2026-09-13 — gfx1201 foundation and assertions

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted engineering increment.

Outcome: Assertions-enabled LLVM/MLIR and Tessera tools on Tajasarus; exact-chip storage binding, corrected RDNA4 fragment store and gfx12 HIPRTC GEMM/forward-attention specialization.

Remaining: General scheduled/Graph route and AD architecture admission, compiler-owned ragged/K-loop matrix proof, LDS/pipelined variants and native-Linux performance attribution. No constructor retirement or performance promotion.

Evidence: [gfx1201 commissioning](../../../benchmarks/baselines/gfx1201_foundation_20260913/README.md); [ROCm owner](../backend/rocm/todo.md#gfx1201-commissioning--2026-09-13).

<!-- entry-fields:end -->

### 2026-09-13 — gfx1201 scheduled integration and fleet validation

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: Uncommitted engineering increment.

Outcome: Replay-derived gfx1201 f32 unary packages execute; standalone FP16/BF16 backward passes ten cases on each of gfx1201 and gfx1151. Five generators use the LLVM 23 inherent kernel property. LDS/pipelined runtime variants pass correctness but lose the three-shape diagnostic comparison. Super-Bear and Princess-Luna retain clean main checkouts and working rebuilt compilers; candidate gfx1151 tests use an isolated compiler binary.

Remaining: General scheduled matrix/attention and paired public AD, resident saved-LSE, broader masks, emitted-kernel attribution, tuned schedules and native-Linux performance promotion. No Graph constructor retirement or cross-architecture performance transfer.

Evidence: [integration and fleet packet](../../../benchmarks/baselines/gfx1201_integration_20260913/README.md); [ROCm owner](../backend/rocm/todo.md#gfx1201-scheduled-integration--2026-09-13).

<!-- entry-fields:end -->

### 2026-09-13 — Attention ancestry and resident LSE proof

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: ROCm scheduled attention forward/backward now replay native parents and project the actual f32-output ABI. gfx1201 forward and backward compose using resident O/LSE; baseline/resident/optional half-load cases pass independently of the host-output oracle. The gfx1151 native composition remains validated. Static half-load evidence shows higher VGPR usage, not a promotion win.

Remaining: General gfx1201 matrix/attention package admission; automatic paired public AD, mixed-precision cotangent conversion and resident tape ownership; broader masks and runtime kernel attribution. WSL /dev/kfd blocks profiler capability enumeration. No Graph constructor deletion or performance promotion.

Evidence: [paired attention and loading packet](../../../benchmarks/baselines/gfx1201_attention_pair_20260913/README.md).

<!-- entry-fields:end -->

### 2026-09-13 — RDNA4 WMMA operand and accumulator audit

Owner: [NUMPOL-CARRIER-1](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1)

PRs: Uncommitted continuation.

Outcome: ISA-total RDNA operand signatures; all eleven dense gfx1201 WMMA forms lower with correct A/B provenance, signedness and accumulator storage. The 76-case device corpus includes finite range/subnormals and nonfinite classification; native low-precision accumulation has a distinct rounding oracle. Existing f32/i32 exact checks are preserved.

Remaining: Sparse SWMMAC metadata/packing and native producers, general matrix/Graph packaging, scaled-format consumers, error-budget integration and performance admission. gfx1200 has no inherited device proof; gfx1151 does not acquire FP8 or K32 forms.

Evidence: [WMMA dtype packet](../../../benchmarks/baselines/gfx1201_wmma_dtypes_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — GFX1201 scheduled package ancestry

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: Static f16 matmul and f16/bf16 forward attention have exact gfx1201 driver/Schedule/Tile/package lineage and twenty device/contract checks. Fixed architecture loss in the direct attention adapter, missing half-wave K partitions in precomputed addresses, and cold-start family registration.

Remaining: General dynamic/epilogue matrix envelopes, automatic public paired AD, resident LSE and reusable asynchronous tapes, backward package admission, sparse SWMMAC storage/index producers. rocprofv3 records API activity but zero dispatch/code-object rows; runtime kernel attribution and clean performance evidence remain open.

Evidence: [scheduled package packet](../../../benchmarks/baselines/gfx1201_scheduled_packages_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — GFX1201 public attention AD

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: gfx1201 scheduled backward programs retain architecture through direct Tile lowering and native packaging. Public fp16/bf16 GQA Q/K/V native_backward selects its owning architecture, executes the exact package and records artifact-bound physical certificates. The gfx1201 policy remains recompute-only; saved-LSE and unknown architecture requests fail closed. Device tests cover paired public AD and explicit bias/window/softcap/dropout programs.

Remaining: Reusable HIP resident-LSE tapes and asynchronous readers/teardown, sparse SWMMAC packing/index/native producers, broader matrix envelopes and runtime kernel attribution. No performance promotion or sibling device proof.

Evidence: [public AD packet](../../../benchmarks/baselines/gfx1201_public_attention_ad_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — GFX1201 resident ownership and sparse packing

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: Reusable HIP recompute workspace snapshots inputs, owns queued host calls through retirement and returns independent host results. fp16/bf16 device checks and host failure/interleaving tests pass. Dynamic f16 matmul reuses one image across three runtime shapes and refuses over-bound calls. Immutable fp16/bf16 2:4 packing/index bytes pass six native LLVM SWMMAC probe comparisons with exact disassembly checks. Compiler subprocess profiler injection is excluded after reproducing a linker crash.

Remaining: Saved-LSE admission remains closed; GPU-stream overlap, external device readers, isolated recovery, broader matrix dtypes/epilogues and production sparse Schedule/Tile producers remain open. Raw one-process host timing is diagnostic only. Profiler has API events but no kernel dispatch/code-object/counter rows; no performance promotion.

Evidence: [resident/sparse packet](../../../benchmarks/baselines/gfx1201_resident_sparse_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — GFX1201 streams readers and sparse Target IR

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: Resident attention uses private nonblocking HIP streams. Explicit same-device read-only output leases order publication, external reads, reuse and retirement; release failures retain retryable ownership. Two gfx1201 owners have intersecting ordered-program event windows in four of five trials with correct gradients. Registered internal f16/bf16 `tessera_rocm.swmmac` validates fragments and lowers through the production Target pass; six MLIR-to-HSACO numerical comparisons are exact. This is an internal physical primitive, not a new public Graph operation, dtype, batching or AD rule.

Remaining: Saved-LSE admission stays closed. General public sparse Schedule/Tile packaging, additional sparse forms, generic reader adoption, isolated uncertain-teardown recovery, calibrated clocks and per-kernel/counter attribution remain open. Event-window intersection is not proof of simultaneous individual kernels. No performance promotion or sibling-device proof.

Evidence: [stream/sparse IR packet](../../../benchmarks/baselines/gfx1201_stream_sparse_ir_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — Saved LSE and sparse Schedule handoff

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted continuation.

Outcome: Explicit gfx1201 saved-LSE retains forward O/LSE in the resident owner's immutable frame across repeated cotangents and external-reader retirement. Auto remains recompute. Internal packed f16/bf16 Schedule and Tile sparse MMA producers verify and lower to Target SWMMAC; six exact native comparisons pass. Valid HIP timing samples are distinguished from independently calibrated selector evidence.

Remaining: Public logical sparse Graph/package integration and additional sparse forms, generic reader adoption, uncertain teardown recovery and calibrated per-kernel attribution remain open. Fresh rocprofv3 records API regions but zero dispatch/code-object/symbol/counter samples. No performance promotion or sibling device proof.

Evidence: [saved/sparse Schedule packet](../../../benchmarks/baselines/gfx1201_saved_sparse_schedule_20260913/README.md).

<!-- entry-fields:end -->


### 2026-09-13 — Logical sparse ownership and floor migration

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: Logical sparse Schedule packing and tiled accumulation execute on gfx1201; asynchronous attention reader release retains retryability; isolated ANN replacement preserves its explicit device ordinal; x86 floor bypasses the historical Graph-owned Tile emitter through replayed native contracts.

Remaining: Public sparse Graph/package binding, additional sparse dtypes, general public attention composition and mixed cotangents, in-process uncertain teardown isolation, remaining cohort/elementwise/breadth constructors, and calibrated performance admission.

Evidence: [Compiler and owning-device packet](../../../benchmarks/baselines/sparse_ownership_migration_20260913/README.md), native artifact/ownership regressions and refreshed lexical route census.

<!-- entry-fields:end -->

The census counts Graph-annotated entry points, not remaining physical constructors
or device certificates. Floor is a migrated branch inside a function that retains
other Graph-owned routes. Sparse execution consumes ordinary matrix storage, but
its validity output is only checked by the device proof harness; no public runtime
admission is asserted. Explicit device ordinal tests use host IPC and do not prove
recovery from a hardware hang or recovery of resident attention allocations.

### 2026-09-14 — Sparse runtime and isolated attention

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: Explicit logical sparse compilation/runtime binding validates all device validity words before output exposure and confirms worker teardown. Matching f16/bf16 accumulation lowers through Schedule/Tile/Target to RDNA4 SWMMAC; eight bounded device cases cover two shapes and both accumulator policies. Resident attention composes pending host futures and lossless fp32/fp64 cotangents. The isolated owner confirms process death before replacement and repeats a workload-scoped zero-VJP check. The new zero-first saved-LSE test exposed a missing forward carrier in direct Tile lowering; the fix preserves it only for a reciprocal saved backward companion. Static f32 x86 ceil now projects and replays the native unary contract.

Remaining: Automatic sparse Graph/JIT admission, integer/FP8 sparse packing, general mixed-precision AD, external device pointers across isolation, recovery of an actually hung driver, cohort/breadth and remaining elementwise migrations. No calibrated timing or performance promotion. Zero VJP is a bounded liveness/numerical check, not a global device-health certificate.

Evidence: [Owning-device and host validation](../../../benchmarks/baselines/sparse_runtime_20260914/README.md).

<!-- entry-fields:end -->

Correction to earlier saved-LSE evidence: reciprocal source attributes and successful reused random-cotangent tests were insufficient to prove the direct forward ABI. The new constant cotangent after a zero-first launch exposed uninitialized checkpoint storage. Both checkpoint modes and repeated reuse are now tested against the oracle with the corrected producer.

### 2026-09-14 — Sparse capture, byte formats and scan

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: Opt-in JIT tracing captures one matmul under an explicit checked 2:4 precondition. Native Graph verification precedes the existing sparse Schedule producer; output storage conversion is emitted on the GPU. Signed i8/i32 and same-format E4M3FN/E5M2/f32 now lower through Schedule/Tile/Target with six gfx1201 instruction/numerical cases. Public attention reverse capture and execution apply exact wider-cotangent checks. Isolated attention admission now compares a nonzero VJP against the shared oracle as well as checking zero VJP. Trailing-axis f32 cumsum moves out of the x86 cohort emitter into replay-verified native Schedule/Tile contracts.

Remaining: Automatic sparse dispatch/arbiter selection, native Graph-to-sparse recipe lowering, mixed FP8/unsigned/INT4 packing, arbitrary public AD composition, actual driver-hang recovery, and remaining cohort/breadth routes. The explicit sparse adapter is frontend code; it does not establish complete MLIR ownership of sparse source lowering. Forced worker death is not a hung-driver or GPU-reset experiment.

Evidence: [Validation packet](../../../benchmarks/baselines/sparse_capture_20260914/README.md). No performance promotion.

<!-- entry-fields:end -->

### 2026-09-14 — Mixed sparse operand contracts

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: Sparse packages carry an independently typed B operand in their replay identity and input preflight. Schedule/Tile/Target accept both mixed FP8 orders; integer operands carry independent signedness flags consumed by LLVM intrinsic lowering. Noninteger signedness overrides are rejected. No automatic sparsification or performance admission is introduced.

Remaining: Native Graph-to-sparse producer and automatic selection; INT4 packing; arbitrary public AD; actual driver-hang recovery; remaining cohort/breadth migrations. Worker-death proof remains distinct from driver recovery, and replacement numerical checks remain workload-scoped.

Evidence: `tests/unit/test_rocm_sparse_byte_formats.py` owns emitted-instruction and gfx1201 numerical checks, including unsigned values above 127, all sparse index pairs and invalid-pattern refusal. The assertions-enabled gfx1201 build passed 407 focused device/registry tests and 33 additional sparse-contract/audit tests. [Validation packet](../../../benchmarks/baselines/sparse_mixed_20260914/README.md).

<!-- entry-fields:end -->

### 2026-09-14 — INT4 packing and sparse AD boundary

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: K=32 sparse INT4 retains byte-addressable logical operands with independently signed range contracts. Schedule/Tile/Target carry integer_bits; native lowering packs i32 words and uses the scalar-A/vector-B LLVM intrinsic ABI. Host and compiled GPU guards reject out-of-range values before exposing output. Four gfx1201 cases cover signed/unsigned pairs, sparse indices, emitted instructions and numerical agreement. The AD scoped plan now explicitly requires differentiation of logical matmul before physical packing, preserving derivatives at zero-valued entries.

Remaining: Automatic sparse selection/native Graph lowering, K=64 and other packed storage envelopes, arbitrary public AD and composed derivative ownership. The explicit sparse API remains forward-only; this is not general AD closure or performance promotion.

Evidence: `tests/unit/test_rocm_sparse_byte_formats.py::test_int4_logical_packing_device`; assertions-enabled LLVM on gfx1201. Sub-byte vector truncation and a vector-valued scalar intrinsic operand were rejected during development; final packing uses explicit i32 operations.

<!-- entry-fields:end -->

### 2026-09-14 — Native sparse Graph and logical AD

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: Declared checked-2:4 half matmul now lowers from Graph to a GPU Schedule kernel in C++, including logical packing, indices, accumulator/output conversion and validity status. JIT projects shape/storage from emitted attributes; it does not invoke the Python recipe producer. The pass declares every emitted dialect and rejects unconsumed module/function/argument/op policies. AD-configured JIT parents retain their logical native_backward path. ROCm matmul adjoints now follow operand order, sum repeated-operand contributions and identify the selected device rather than device zero/gfx1151 by assumption.

Remaining: Automatic default/arbiter sparse-versus-dense selection, larger/nonisolated Graph envelopes, arbitrary public AD composition and higher-order/control-flow closure. The selected forward remains an explicit checked contract; no uncalibrated promotion is introduced. Backward matmul still uses the existing composed runtime GEMM route, not a new general region AD package.

Evidence: `tests/unit/test_sparse_capture.py`, `tests/unit/test_native_matmul_operand_ad.py` and `tests/unit/test_autodiff_rocm_matmul_composed.py`; final results recorded with this wave. No performance promotion.

<!-- entry-fields:end -->

### 2026-09-14 — Native automatic sparse selection

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #747.

Outcome: JitFn.compile_sparse_auto lowers isolated half matmul through the native Graph producer. Each K tile uses a wave-wide 2:4 agreement: eligible tiles execute SWMMAC, others execute native dense accumulation. One image accepts changing density without pruning or CPU fallback. The original logical JIT source still owns native_backward. The emitted selection policy is checked against its source policy during package validation.

Remaining: Default-dispatch/performance promotion, broader shapes and composed native regions, arbitrary AD/control-flow/higher-order closure. This explicit automatic policy does not replace incumbents globally and claims no speedup. General AD remains separate from sparse selection.

Evidence: `tests/unit/test_sparse_capture.py::test_native_automatic_sparse_dense_selection` checks dense, sparse, one-invalid-lane mixed tiles and density changes with one artifact on gfx1201; fp16 also checks logical backward. f16 and bf16 cases use the owning device.

<!-- entry-fields:end -->


### 2026-09-14 — Native composed HVP execution

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Current follow-through after #747.

Outcome: Adds static CPU HVP execution through compiler-produced derivatives and signature-owned output allocation. Tracer nested regions use SCF; scalar extraction tangents, passive predicates and ownership-clone lowering repair nested branch/loop products.

Remaining: Effectful CFG, dynamic outputs, higher derivative orders and general asynchronous HVP frames remain open in [AD execution](AUTODIFF_EXECUTION_PLAN.md). No performance promotion.

Evidence: Native CPU HVP tests on Princess-Luna; implementation and exact-device evidence remain separate.

<!-- entry-fields:end -->


### 2026-09-14 — Saved-product HVP and GPU export

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Current follow-through after #747.

Outcome: Continuous saved products carry captured tangents into backward JVPs. Typed export feeds bufferized GPU tape lowering. Unsupported effectful while refuses rather than using an empty AST candidate.

Remaining: Effectful CFG, dynamic outputs, higher derivative orders and general asynchronous HVP frames remain open in [AD execution](AUTODIFF_EXECUTION_PLAN.md). No performance promotion.

Evidence: Native CPU nested SAVE curvature and gfx1201 composed/counting-loop HVP tests.

<!-- entry-fields:end -->


### 2026-09-14 — HVP product identity and CUDA execution

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: Current follow-through after #747.

Outcome: Export accepts only products generated by the pass invocation. Vector/matrix composed and counted-loop HVPs execute independently on SM120 and gfx1201.

Remaining: Effectful CFG, dynamic outputs, higher derivative orders and general asynchronous HVP frames remain open in [AD execution](AUTODIFF_EXECUTION_PLAN.md). No performance promotion.

Evidence: `tests/unit/test_native_hvp_execution.py`: 7 passed, 10 skipped per owning GPU host, including four device cases each.

<!-- entry-fields:end -->


### 2026-09-15 — Apple Metal 4 matmul2d family on the canonical GEMM

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: Uncommitted follow-through after #751 (APPLE-MATMUL2D-1).

Outcome: The first Apple family enters F2. `tessera_apple.gpu.tensor_view` (a strided rank-2 MTLTensor view over a real f16/bf16/f32/f8E4M3FN/f8E5M2/f4E2M1FN element type) and `tessera_apple.gpu.matmul2d` (MetalPerformancePrimitives matmul2d, operand pair verified against the header table incl. f16 × {f8E4M3FN, f8E5M2, f4E2M1FN}, fp32 accumulator required, K/M/N agreement) are declared with verifiers; `tessera-apple-canonical-gemm-matmul2d` re-forms the shared TilingPass reduction into them and refuses packed 8/4-bit operands whose row stride is off Apple's 128-byte quantum; `tessera-apple-matmul2d-to-call` lowers to a value-producing `gpu.kernel_call` on the runtime's `mtl4_matmul2d_{f16,bf16,lowp}` symbols with the view ABI (inner/outer/stride/byte_offset per operand, storage pair, format code, accumulator) as attributes; the value-lane dispatcher `mtl4_matmul2d` consumes those attributes and re-derives nothing. Both passes are registered standalone and are not in the default `tessera-lower-to-apple_gpu` pipeline; the incumbent MPS/Accelerate route is unchanged (APPLE-TILE-2 rule).

Remaining: Ragged M is zero-padded by the shared tiling pass before the view is taken (the IR states it; the runtime's own ragged support is not yet used); nonzero view origins and padded strides are verified in IR but not yet projected through the dispatcher; default-pipeline admission needs a paired corpus (measured 2026-09-14: FP8 at 0.77–0.93× fp16, fused epilogue not faster); the Python `apple_native` GEMM packager is bypassed only for this canonical-reduction route and is not deleted; the fused bias/act epilogue and the coopmat MSL emitter remain outside this op family.

Evidence: `tests/tessera-ir/phase8/apple_matmul2d{,_invalid,_lowering_invalid}.mlir` (positive, six verifier negatives, one lowering refusal); `tests/unit/test_apple_matmul2d_lane.py` — host-free lowering rows for seven operand pairs and three refusals, plus seven owning-Mac execution rows (M1 Max, macOS 27.0) comparing the lowered call's value-lane result against a float64 oracle built from the exact quantized bytes; `docs/audit/generated/target_ir_membership.md` scores both ops as requiring their contracts. No performance or promotion claim.

<!-- entry-fields:end -->

### 2026-09-15 — matmul2d route: ragged tails, view origins, fused epilogue, paired admission

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: #753 (same branch as APPLE-MATMUL2D-1; follow-through commits).

Outcome: The four items the first slice left open are closed on the compiled route. (1) Ragged M/N: `tessera-apple-canonical-gemm-matmul2d` looks through the shared TilingPass zero-pad (`tensor.insert_slice` into a zero constant) and binds the ORIGINAL operand at its true extents; the product is the true [M, N], the trailing slice disappears, and the op states the runtime contract it relies on (`tessera_apple.ragged_tail`, MPP's partial-tile store); the padded form is kept only when the true extents are not a legal MTLTensor view. (2) Nonzero origins and padded strides reach the dispatcher: a static unit-stride `tensor.extract_slice` operand is bound as a view INTO its parent (byte offset, parent row stride, nothing copied, misaligned FP8 origins refused with the verifier's wording); the runtime gained one strided-view entry for every pair (`tessera_apple_gpu_mtl4_matmul2d_view[_epilogue]`, pair codes 0–5 low precision, 10 f16, 11 bf16, 16-bit kernels added to the MSL 4.1 source), the call lowering emits that one symbol with `tessera_apple.pair`, and `runtime._dispatch_gpu_mtl4_matmul2d` passes element origins and strides through after checking the storage row equals the stated stride. The driver's value-call extractor never captured dialect-prefixed keys (`\w+` took only the suffix), so the view ABI had not actually reached the value lane through the driver before; keys now capture whole with the bare suffix kept for older consumers, and the GPU value executor admits the op kinds it used to reject before its own branch. (3) The fused epilogue is an op: `tessera_apple.gpu.matmul2d_epilogue` (optional rank-1 f32 [N] bias, `act` in {none, relu, gelu, silu}; contract: bias added to the fp32 accumulator, activation in fp32, one store; an op fusing nothing is refused) produced by `tessera-apple-matmul2d-fuse-epilogue` from the tracer's broadcast-add + activation chain (single-use only; Tessera's gelu IS the tanh form the kernel evaluates) and lowered to the `_view_epilogue` symbol with the bias as a third operand. (4) Paired admission measured: two independent 20-rep interleaved processes on the M1 Max against what production dispatches today. f16 retained at every shape (0.53–0.96×); bf16 within ±10% of the runtime's own contiguous bf16 entry (admitted only at 512/1024 square, split at ragged 1000, retained at 2048/MLP/decode); the low-precision pairs lose at every GEMM shape (0.72–0.88× fp16 on identical values) and win only at M == 1 decode (1.6–1.7×, lb ≥ 1.40 both runs) with a one-time ~28 ms pack of a 4096² weight. Decision: the default `tessera-lower-to-apple_gpu` pipeline admits the family for the 8/4-bit storage pairs only (`admit=lowp`), because those pairs had no executable route — the pipeline emitted an MPSGraph `matmul_contract` claim for an FP8 GEMM that MPSGraph cannot run — while f16 and bf16 keep the APPLE-TILE-2 incumbent. The bf16 ≤ 1024 and FP8-decode wins are arbiter-bucket candidates (Decision #28), not pipeline admissions.

Remaining: An arbiter bucket entry for bf16 ≤ 1024 square and weight-only FP8 decode (needs the toolchain-versioned cache key of Decision #11); the JIT front door for 8/4-bit storage dtypes (the pipeline now lowers them, `@jit` tracing of FP8 tensors is a separate gate); the bias-operand matmul form (`tessera.matmul` with `bias = "..."`) is dropped by the shared TilingPass before any backend sees it (pre-existing, sibling-visible — the fusion recognizes the broadcast-add form that survives); amortised or fused operand packing before any FP8 arbiter candidacy; the `apple_native` GEMM packager is still not deleted.

Evidence: `tests/tessera-ir/phase8/apple_matmul2d.mlir` (ragged look-through, sub-block origins, epilogue fusion, no fusion with two consumers), `apple_matmul2d_invalid.mlir` (nine verifier negatives), `apple_matmul2d_lowering_invalid.mlir` (two lowering refusals), `apple_matmul2d_pipeline.mlir` (default-pipeline admission: f16/bf16 on the incumbent, FP8/FP4 on the view call, closed admit set); `tests/unit/test_apple_matmul2d_lane.py` — 21 host-free rows and 19 owning-Mac rows (macOS 27.0, M1 Max): seven pairs at true ragged M = 100, ragged N = 100 f16 at a 200-byte row stride, four sub-block origin rows through the dispatcher, four fused-epilogue rows against the decomposed oracle, one `runtime.launch` end-to-end through the value artifact, two refusals; corpus packet `benchmarks/baselines/apple_matmul2d_route_corpus_20260915/` (two processes, README records the decision); `docs/audit/generated/target_ir_membership.md` scores the epilogue op as requiring its contracts. Full lit 442 passed / 40 unsupported on the Mac; the ROCm backend lit suite cannot run on this host. No general speedup claimed.

<!-- entry-fields:end -->

### 2026-09-15 — shared TilingPass preserves the matmul bias/residual epilogue

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: follow-up to #753 (sync `TILING-MATMUL-EPILOGUE-2026-09-15`).

Outcome: The shared `tessera-tiling` matmul pattern accepted the tracer's keyword-operand form (`tessera.matmul %a, %b, %bias {bias = "..."}` / `residual`) but copied only the two product operands into the inner tile op while keeping the marker attrs, so the pass failed its own verifier and, had it not, the epilogue would have been silently dropped (Decision #32). The inner K step is now the plain product with the markers stripped, and the epilogue is re-applied once on the logical [M, N] result as ordinary Graph IR — bias broadcast per output column then added, residual added after — the same form `nn.functional.linear` already emits, so every backend's canonical-GEMM recognizer sees the unchanged nest and the Apple `matmul2d_epilogue` fusion consumes the bias-operand matmul end to end.

Remaining: No backend executes a multi-op value program, so a biased matmul still runs the epilogue outside the kernel except on Apple's fused route; the arbiter bucket, the `@jit` front door for 8/4-bit storage tensors and the strict route ledger re-seal stay open from the previous entry.

Evidence: `tests/tessera-ir/phase2/tiling_matmul_epilogue.mlir` (bias, bias + residual on ragged M, residual alone; inner op carries no marker), `tests/tessera-ir/phase8/apple_matmul2d_bias_operand.mlir` (bias-operand matmul → `gpu.matmul2d_epilogue`); full lit 445 passed / 40 unsupported on the Mac (host-free fixtures); registry, tiling and lane unit gates green. No device or performance claim.

<!-- entry-fields:end -->

### 2026-09-15 — the @jit front door for 8/4-bit storage tensors on Apple GPU

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: follow-up to #753 (sync `LOWP-FRONT-DOOR-2026-09-15`).

Outcome: Probing `@jit(target="apple_gpu")` with FP8 / FP4 operands found a hollow green, not a missing feature: the tracer named every numpy dtype it did not know "f32", so an FP8 program traced as an f32 program; the Graph IR spellings of the low-precision types were unparseable (`xxf8E4M3FN`, `!tessera.fp4_e2m1` — a type no dialect defines); and the MPS matmul dispatcher, finding no lane for the dtype, computed the product on the host with numpy while the artifact reported `native_gpu` (the result matched a float64 oracle exactly). Fixed end to end: the tracer names `float8_e4m3fn` / `float8_e5m2` / `float4_e2m1fn` by their canonical storage dtypes; Graph IR spells them as the MLIR 23 builtins `f8E4M3FN` / `f8E5M2` / `f4E2M1FN` (reverse table too); a matmul over storage-only dtypes types its result fp32 (Decision #15a, matching `_promote_two`); the dispatcher routes every low-precision pair — fp8×fp8, fp4×fp4 and the weight-only f16×{fp8, fp4} — through the strided-view MPP matmul2d lane the compiled route also uses, and a dtype with no lane now goes through the fallback funnel (strict dispatch raises) instead of standing in silently; the apple_gpu matmul capability and manifest rows declare the three dtypes. Two shared-frontend fixes rode along: `to_graph_ir_module` verifies legality against the target the caller compiles for instead of always the CPU table, and the jit's Graph IR renders verify against the jit's own target — before this the tracer authority silently lost every GPU-only-dtype program to the AST fallback.

Remaining: The compiled (value-mode) route for a traced FP8 program is proven host-free through the default pipeline (`op_kind = "mtl4_matmul2d"`); the `@jit` default mode still dispatches through the MPS envelope, which now reaches the same Metal kernel — moving the default mode onto the compiled artifact is E2E-REAL-6's general migration, not this item. The arbiter bucket for bf16 ≤ 1024 stays open; the FP8-decode "bucket" is a quantization-policy decision (a different program), not an arbiter choice, and is recorded as such. The bias-operand matmul form dropped by the shared TilingPass is addressed separately (sync `TILING-MATMUL-EPILOGUE-2026-09-15`).

Evidence: `tests/unit/test_apple_jit_lowp_front_door.py` — 13 host-free rows (tracer naming, builtin spellings with round trip, fp32 matmul result, a traced FP8 program lowering to the matmul2d view call through the default pipeline) and 7 owning-Mac rows (five pairs through `@jit` on the Metal view lane against an exact-bytes float64 oracle, f16 unchanged, strict-dispatch refusal of a lane-less dtype); jit, frontend-authority, Apple value-lane, manifest and capability suites 507 passed on the Mac; `dtype_flow`, `apple_target_map` and `support_table` regenerated. No performance claim.

<!-- entry-fields:end -->

### 2026-09-15 — bf16 matmul2d arbiter bucket: measured, retained

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: follow-up to #753 and #755 (sync `APPLE-MATMUL2D-BUCKET-2026-09-15`).

Outcome: The bf16 ≤ 1024 "win" the paired admission corpus reported (5–12% median, view entry over the production bf16 route) is now an arbiter bucket in the strict route ledger — `retune_matmul2d_bf16` at 512, 1024 and 2048 square, the runtime's contiguous MPP `matmul2d` entry against the strided-view entry the compiled route dispatches, five independent runs of nine interleaved trials each — and the production bf16 route (`_mtl4_route_matmul2d_bf16`) consults that ledger per exact shape and dispatches the view entry only when it is promoted. On the first seal **every row retained the contiguous incumbent** (six rows, both timing domains): on device time the two entries are within ±3% (the same kernel), and end to end the view entry is 6–14% slower at 512 and 20–42% slower at 1024. The corpus's earlier win was not the kernel: its incumbent went through the production wrapper (a bf16 result cast and its buffer path) while its candidate did not, so the comparison measured wrapper overhead. The bucket exists so a later runtime change can promote it with evidence; nothing is promoted today.

Remaining: The view-entry wrapper zero-fills its fp32 output on the host, which is most of its end-to-end penalty at 1024; that is a wrapper cost to remove before re-measuring, not a kernel finding. The FP8-decode "bucket" from the corpus is a quantization-policy decision (a different program), recorded as such and not an arbiter row.

Evidence: `benchmarks/baselines/apple_strict_route_ledger.json` (24 decisions, 6 ineligible, macOS 27.0 / SDK 27.0, zero routes moved among the previous 18) and `apple7_legacy_retune_multi_run.json`; `tests/unit/test_apple_legacy_retune_benchmark.py` (14 passed on the Mac incl. the live-host admission row and the host-free consumer test that the bf16 route dispatches the view entry only on a promotion). No performance claim.

<!-- entry-fields:end -->

### 2026-09-15 — matmul epilogue markers from both frontends, activation after the reduction

Owner: [E2E-REAL-6](INTEGRATED_COMPILER_PLAN.md#e2e-real-6)

PRs: follow-up to #754 (review), sync `MATMUL-EPILOGUE-MARKERS-2026-09-15`.

Outcome: The #754 rewrite was unreachable from Python: `ops.matmul(a, w, bias=b)` traced as a three-operand `tessera.matmul` with no `bias` marker (the tracer moves keyword tensors into the operand list and drops the keyword), so `MatmulOp::verify` rejected it before tiling. The shared `apply_presence_flags` — one implementation both frontends call, held to parity by the differential certificate — now also emits the string markers the C++ verifier reads (`bias = "bias"`, `residual = "residual"`) for the bound epilogue operands of `tessera.matmul`. The `activation` attribute was left on every inner K step, where a consumer honoring it would apply it per partial product; the tiling pass now strips it and materializes `tessera.<act>` once on the completed product, between the bias and the residual (the public `gemm` contract), and leaves the nest untouched for an unknown activation name rather than guessing.

Remaining: Outside Apple's fused `gpu.matmul2d_epilogue` no backend executes the re-applied epilogue inside the kernel; the value lane still executes single-call programs only.

Evidence: `tests/unit/test_tiling_matmul_epilogue.py` (tracer and AST markers, contract-order tiling of a traced matmul through tessera-opt, host-free); `tests/tessera-ir/phase2/tiling_matmul_epilogue.mlir` (bias + gelu + residual order, activation alone, unknown activation left alone); full lit and the frontend/trace/jit suites green on the Mac. No device or performance claim.

<!-- entry-fields:end -->

### 2026-09-15 — query/key-axis broadcast attention masks reach native SM120 indexing

Owner: [FRONTEND-IR-MEDIUM-1](INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1)

PRs: continuation of the 2026-09-12 batch/head increment (sync `ATTN-QK-BROADCAST-2026-09-15`).

Outcome: A raised attention bias may now broadcast on any of its four axes: a key-padding row (`[1,1,1,K]`) and a per-(batch, head, query) column (`[B,Hq,Q,1]`) pass source recognition (`loop_idioms` accepts extent 1 and index 0 on the query and key axes), symbolic binding (`SymbolicDimEqualityPass`), Schedule selection and `bias_shape` carriage (`PMPasses`), Tile lowering that keeps the physical block (`TileIRLoweringPass` slices `[1,tkv]` / `[tq,1]` blocks and `ScoreBiasOp` verifies each axis as 1 or the scores extent), and SM120 kernel indexing (`NVIDIALowering` reads index 0 with stride 1 on every broadcast axis). The f32 broadcast host ABI is versioned to v3 and carries all four physical extents (`BiasB`, `BiasH`, `BiasQ`, `BiasK`); the launcher validates each against 1 or the logical extent and copies only the physical storage. Empty-row refusal now judges rows on the logical broadcast view, so a fully masked key-padding row is refused before launch. Six B=2 ragged causal/GQA/window buckets (batch/head, key-padding and per-query forms at Q/K = 3/5 and 5/3) match the numpy oracle on the owning RTX 5070 within 6e-8. ROCm's canonical streaming attention refuses a broadcast `score_bias` block instead of indexing it at the scores' shape.

Remaining: Boolean/padding masks as a first-class operand (today a padding mask is an additive `-inf` row), sibling-target consumers (Apple, ROCm, x86 raised attention have no broadcast proof; ROCm fails closed), f16/bf16 storage for broadcast bias, and any performance admission. The 2026-09-12 batch/head evidence was re-recorded here under the v3 ABI; the v2 packets are historical.

Evidence: [attention broadcast packets](../../../benchmarks/baselines/attention_broadcast_20260915/README.md) (six device rows, source fingerprints), `tests/unit/test_attention_broadcast.py` (recognition, physical-shape guards, tampered-schedule refusal, device rows behind `TESSERA_TEST_RAISED_ATTENTION=1`), `tests/tessera-ir/phase3/flash_attn_broadcast_bias_tile_lowering.mlir` (host-free Tile lowering of both forms), `ninja -C build-nvidia-cuda check-tessera-nvidia` 61/61 on Super-Bear.

<!-- entry-fields:end -->

### 2026-09-15 — probed admission and health-checked replacement of isolated heap workers

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: continuation of the 2026-09-11 gated owner (sync `HEAP-REPLACEMENT-HEALTH-2026-09-15`).

Outcome: A spawned `IsolatedHeapPool` worker is admitted only after its own in-process device probe verifies the admitted producers on the ordinal it will own (int8 allocation → one live slot of the payload width → bitwise readback through a generation-checked pin → empty graph published, marked and reclaimed); the ready message carries the probe tag and anything else is a failed admission, torn down with confirmed death. `replacement()` refuses until the predecessor's death is confirmed and then admits a fresh worker whose own probe is the health evidence. Admission has its own bound because it compiles the metadata kernels in-process. Both owning devices (RTX 5070, gfx1151) pass all nine recorder proofs. Prerequisite repair: since 100a2980 the storage packager expanded low-precision conversions while six validators (heap pool, SSD, ANN, exception heap, public result, gradient sum) replayed a hand-copied pipeline, so every such package was refused on device with "disagrees with native replay"; the pipeline is now spelled once (`native_gpu_storage.ARENA_PIPELINE`) with a drift test.

Remaining: Legacy snapshot/import callers still need an explicit gated producer; the stopped-worker fault is injected, not a reproduced driver hang; the probe proves the selected workload on one ordinal now, not global driver health; automatic replacement policy, broader isolated commands and epoch relaxation stay open. No measured overlap or promotion.

Evidence: [CUDA/HIP replacement packets](../../../benchmarks/baselines/gated_heap_replacement_20260915/README.md) (nine proofs each, source fingerprints), `tests/unit/test_gated_heap.py` (admission refusals and the probe's three failure modes, host-free), `tests/unit/test_arena_replay_pipeline.py`.

<!-- entry-fields:end -->

### 2026-09-16 — batched geometric products execute through the MLIR/LLVM backbone

Owner: [W6.4](INTEGRATED_COMPILER_PLAN.md#w64)

PRs: domain-support stream, first slice (sync `GA-NATIVE-BATCHED-2026-09-16`); also carries the `FLEET-REDS-2026-09-15` NVIDIA/x86 sibling entries Codex asked for on #761.

Outcome: `ExpandProductTable` lowers any static `[..., dim]` operand pair — the compile-time Cayley table emitted once inside an scf.for nest over the leading axes, result tensor as iter_arg, grade pruning unchanged, dynamic extents refused. `libtessera_jit` registers the Clifford dialect and runs GradeFusion + ExpandProductTable ahead of tessera→linalg, so `tessera_clifford.geo_product` compiles and executes through one-shot bufferization and LLVM; `_jit_boundary.jit_clifford_geo_product` and `package_clifford_geo_product_cpu` + `runtime.launch` are the consumers, and the execution matrix carries the `cpu` row the domain proof ladder now counts. Parity with the standalone GA reference for rank 1/2/3, grade-restricted products equal the projection with untouched coefficients zero, non-commutativity preserved, no fallback; verified on the M1 Max, Princess-Luna (Zen 5) and Super-Bear (Zen 2), with the Clifford lit suite 17/17 on all three. Princess-Luna and Super-Bear now configure the Clifford backend ON — until now only the Mac could build the domain dialects.

Remaining: rotor sandwich fold and the remaining Clifford ops have no lowering behind the JIT; ragged batches (static extents only); the GPU package route through the arena pipeline; the acceptance's separate dispatch/allocation/memory/kernel-time measurements (no performance claim); the Python-emitted x86/ROCm/Apple GA kernels stay the device lanes until displaced by measured evidence. EBM's traceable quadratic energy loop is the next domain slice.

Evidence: `tests/unit/test_clifford_jit_native.py` (three CPU hosts), `src/solvers/clifford/test/ir/passes/expand_batched.mlir` + `expand_rejects_dynamic.mlir`, `docs/audit/generated/domain_proof_ladder.md` (`cpu=1` under GA), `docs/audit/generated/runtime_execution_matrix.md`.

<!-- entry-fields:end -->

### 2026-09-16 — the Clifford product family executes behind the MLIR/LLVM JIT

Owner: [W6.4](INTEGRATED_COMPILER_PLAN.md#w64)

PRs: domain-support stream, second slice, stacked on the batched-product slice (sync `GA-NATIVE-FAMILY-2026-09-16`).

Outcome: One compile-time table now carries every linear/bilinear op the standalone GA reference defines: `ExpandProductTable` lowers wedge (disjoint blades only), left contraction (result grade = grade(b) − grade(a)), inner and norm (one scalar per multivector, typed `[..., 1]`, norm clipped before sqrt), reverse / grade involution / conjugate (per-grade signs), Hodge star (`reverse(a)·I` as a signed blade permutation), standalone grade projection, and rotor sandwich expanded to `gp(gp(R,x), reverse(R))` behind an `expand-rotor-sandwich` pass option — the fused marker still survives the standalone pipeline for backends with a sandwich kernel, and the JIT enables the expansion. All batched through the shared loop nest. `jit_clifford_op` / `package_clifford_cpu` are the consumers; the execution-matrix row is renamed `cpu` / `cpu_clifford_llvm_jit` (op family `clifford`). Nine ops × single/batched/rank-3 shapes match the reference on the M1 Max, Princess-Luna and Super-Bear; grade projection keeps only the listed grades; a unit rotor preserves the vector norm; Clifford lit 19/19 on all three.

Remaining: `exp`/`log` (series with branch handling) and the field ops (ext_deriv, codiff, vec_deriv, integral) have no native lowering; ragged batches; the GPU package route through the arena pipeline; the acceptance's separate overhead/traffic/kernel-time measurements (no performance claim); the Python-emitted device kernels stay the x86/ROCm/Apple GA lanes until measured against this path.

Evidence: `tests/unit/test_clifford_jit_native.py` (three CPU hosts), `src/solvers/clifford/test/ir/passes/expand_family.mlir` + `expand_family_rejects.mlir`, `docs/audit/generated/runtime_execution_matrix.md`.

<!-- entry-fields:end -->

### 2026-09-16 — the Clifford family reaches ROCm and sm_120 through the arena pipeline

Owner: [W6.4](INTEGRATED_COMPILER_PLAN.md#w64)

PRs: domain-support stream, third slice, on the family branch (sync `GA-NATIVE-GPU-2026-09-16`).

Outcome: A domain op now reaches a GPU package without a Python-emitted kernel. `native_clifford_gpu.py` writes only the kernel skeleton (one thread per multivector: loads, a rank-1 `tessera_clifford` op on tensors, stores); `ts-clifford-opt` — which now registers gpu/llvm/memref to parse kernels — expands it through the same GradeFusion + ExpandProductTable lowering the CPU JIT runs; the arena pipeline's canonicalization folds the tensors away, leaving scalar arithmetic the compiler emitted (64 products for the full Cl(3,0) product, 24 under a grade-2 restriction, counted in the arena IR); `build_native_gpu_storage` packages it and the native storage binding launches it. `runtime.launch` rows `rocm` / `rocm_clifford_native_compiled` and `nvidia_sm120` / `nvidia_clifford_native_compiled` share one executor; `package_clifford_native` builds the artifacts. Ten ops × single/batched/rank-3 shapes match the standalone GA reference on gfx1151 (Princess-Luna), gfx1201 (Tajasarus, Clifford backend now ON there too) and sm_120 (Super-Bear), 49/49 tests each; the domain proof ladder's GA row gains `nvidia_sm120=1` and a second `rocm` row from the matrix.

Remaining: the Python-emitted `rocm_clifford_compiled` / `x86_clifford_compiled` / Apple kernels are untouched and remain the device lanes until this route is measured against them (per-call host transfers here are correctness-only); `exp`/`log` and the field ops; ragged batches; an Apple package route (the Apple arena lane is MSL, not this pipeline); the acceptance's separate overhead/traffic/kernel-time measurements. EBM's traceable quadratic energy loop is the next domain slice.

Evidence: [three device packets](../../../benchmarks/baselines/clifford_native_gpu_20260916/README.md) with source fingerprints, `tests/unit/test_clifford_native_gpu.py` (host-free expansion/pruning half on the Mac; device half on the three boxes), `benchmarks/record_clifford_native_gpu.py`, `docs/audit/generated/domain_proof_ladder.md`.

<!-- entry-fields:end -->

### 2026-09-16 — the EBM quadratic energy loop executes through the MLIR/LLVM backbone

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: domain-support stream, fourth slice (sync `EBM-NATIVE-QUADRATIC-2026-09-16`); co-owner [AD-SOLVER-IFT-1](INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1).

Outcome: The GA/EBM review's first acceptance clause for "an energy is a typed program" is met on the CPU lane. `E(y, x) = 0.5·Σ(x − y)²` is a Graph IR function marked for reverse-mode; inside `libtessera_jit` the paired autodiff pass derives `@E__bwd`, the EBM dialect's first lowering pass (`tessera-ebm-lower-langevin`) turns `energy` / `inner_step` / `langevin_step` into arith + linalg over that gradient with Philox-4x32-10 / Box-Muller noise generated in a `linalg.generic`, and the K-step `scf.for` compiles as one function — no per-step host gradient or noise transfers. `langevin_step` gained a variadic `captures` operand for the energy's context. The declared RNG policy is stated in the pass and mirrored bit-for-bit by `native_langevin.reference_langevin_loop`; forward energy and the T = 0 step match the independent formulas; fixed-key samples match for 1, 5 and 12 steps; `runtime.launch` row `cpu` / `cpu_ebm_langevin_llvm_jit`. Verified on the M1 Max, Princess-Luna (Zen 5) and Super-Bear (Zen 2); EBM lit 14/14 on the Mac, both Zen hosts and Tajasarus under its assertions LLVM. Three compiler gaps found and closed on the way: `tessera.sub` had no adjoint (reverse-mode stopped on any subtraction); `tessera.unsqueeze` / `tessera.broadcast`, which the sum-reduce adjoint emits, had no linalg lowering (no reduce-sum gradient could reach the JIT); and the JIT's DPS out-param rewrite did not follow intra-module call sites.

Remaining: a GPU package for the loop (the tensor-level gradient needs the tile pipeline, not the arena skeleton); nonlinear and manifold energies (sphere / bivector integrators fail closed); opaque-callback energies keep the reported reference path; no performance measurement. The engineering plan for all of these — with the measured stop of each existing device route on this loop (the serial native-tape route rejects it for exactly two reasons; the cooperative route has no EBM Schedule op) — is [EBM_NATIVE_LOOP_ARCHITECTURE.md](../domain/EBM_NATIVE_LOOP_ARCHITECTURE.md). Pre-existing and unrelated: every bf16 JIT test fails on Super-Bear (Zen 2, no AVX512-BF16) with unresolved `_mlir_ciface_*` symbols — bisected by building the JIT from main's sources on that box; bf16 JIT proof belongs on the Zen 5 hosts.

Evidence: `tests/unit/test_ebm_native_langevin.py` (three CPU hosts), `src/solvers/ebm/test/ir/passes/lower_langevin_quadratic.mlir` + `lower_langevin_rejects.mlir`, `tests/tessera-ir/phase2_autodiff/autodiff_paired_sub.mlir`, `docs/audit/generated/domain_proof_ladder.md` (`cpu=1` under EBM).

<!-- entry-fields:end -->

### 2026-09-16 — the EBM Langevin loop runs as one cooperative kernel on gfx1151, gfx1201 and sm_120

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: domain-support stream, fifth slice, on the EBM branch (sync `EBM-NATIVE-GPU-2026-09-16`); co-owner [AD-SOLVER-IFT-1](INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1).

Outcome: The tensor-level gradient is lowered *inside* a device kernel, through the compiler alone. A new core pass, `tessera-row-program-to-gpu` (`RowProgramToGPUPass.cpp`), consumes the loop after paired autodiff → EBM lowering → `tessera-to-linalg` → inlining and emits one `gpu.func`: one block per row, one lane per feature (≤ 1024), the K-step `scf.for` carried in registers, Philox-4x32-10 / Box–Muller inside the loop, ordered shared-memory reductions (sequential over the feature index, so reduced quantities are bit-exact with a host f32 fold, not tolerance-based), guarded `!llvm.ptr<1>` loads/stores and the arena marker — then `build_native_gpu_storage` packages it and the native tensor binding launches it. `tessera-opt` now registers the EBM and Clifford dialects and passes when built with them, so the whole chain is one driver invocation (T1). Rows `rocm` / `rocm_ebm_langevin_native_compiled` and `nvidia_sm120` / `nvidia_ebm_langevin_native_compiled`; `package_ebm_langevin_native`; `compiler/native_row_program.py` is the packaging seam any `[rows, features]` domain program reuses. Measured on gfx1151 (Princess-Luna), gfx1201 (Tajasarus, assertions-ON driver) and sm_120 (Super-Bear): one launch per loop, bit-exact with the declared policy for K ∈ {1,4,5,8,12}, F ∈ {5,…,1024}, T ∈ {0,…,0.7} — worst abs error 0 in every packet row, the device f64 `log`/`cos` included; the row-normalization reduction program bit-exact with the sequential fold. The scoped G2 Tile contract was not needed: the lowered linalg loop already *is* the scalar body, and one pass over upstream dialects maps it (Decision #31 — no second scalar emitter); G1 serial residency stays scoped only as the fallback outside the envelope. The assertions-enabled driver falsified three promises every NDEBUG driver ran green through — `--inline` needs the LLVM dialect's promised inliner interface (now registered in `tessera-opt`), the pass emitted `tile.alloc_shared` without declaring the tile dialect, and its `tessera.*` provenance attributes load the Tessera dialect an op-free input never loaded (both now declared) — Decision #19's standing lesson, third instance. sm_120 exposed a fourth defect: `math.sqrt` on the NVVM route lowers to libdevice's `__nv_sqrtf`, whose precise branch is gated on a reflect flag MLIR never sets, so the kernel ran `MUFU.SQRT` and one row of the reduction proof was 1 ulp off; the emitter now calls libdevice's rounding-explicit `__nv_fsqrt_rn` on NVIDIA (`convert-gpu-to-nvvm` outlaws the LLVM math intrinsics) and pins `llvm.intr.sqrt` on ROCm and every other libdevice f32 path on that route is recorded as unmeasured.

Remaining: no performance claim — the Python-emitted `rocm_ebm_langevin_compiled` / `x86_ebm_langevin_compiled` / Apple kernels remain the lanes until the dispatch/allocation/traffic/kernel-time comparison is recorded with `route` and `latency_source`; rows wider than 1024 features, data-dependent control flow and cross-row-coupled energies fail closed; nonlinear (N1) and manifold (M1 sphere, M2 bivector) energies are the next slices and are row programs the same emitter maps; no Apple package route (the Apple arena lane is MSL). Super-Bear's `build-nvidia-cuda/` tree was configured without the EBM/Clifford backends and is now reconfigured with both ON (the runtime resolves that tree's driver).

Evidence: [three device packets](../../../benchmarks/baselines/ebm_langevin_native_gpu_20260916/README.md) with source fingerprints, `tests/unit/test_ebm_native_langevin_gpu.py` (host-free chain/replay/envelope half on the Mac; device half on the three boxes), `benchmarks/record_ebm_langevin_native_gpu.py`, `tests/tessera-ir/phase2_autodiff/row_program_to_gpu_langevin.mlir`, `tests/tessera-ir/phase_f5/row_program_to_gpu_reduce.mlir`, `docs/audit/generated/domain_proof_ladder.md` (EBM row gains the two device columns from the matrix).

<!-- entry-fields:end -->

### 2026-09-16 — nonlinear energies and the sphere integrator reach the same kernel

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: domain-support stream, sixth slice (sync `EBM-NONLINEAR-MANIFOLD-2026-09-16`); co-owner [AD-SOLVER-IFT-1](INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1).

Outcome: The EBM acceptance's remaining two clauses close on the CPU lane and on all three GPUs. **N1:** the integrator is energy-agnostic, so admitting an energy is a differentiation question — `native_langevin` now carries quadratic, Huber and softplus, each with its own numpy oracle, and the one gap this found was `softplus`, whose adjoint was a `custom_adjoint_call` placeholder (a host VJP, i.e. the per-step transfer the acceptance forbids). It gained a native adjoint, `dy · sigmoid(x)`, the stable form the architecture doc specified, plus a stable linalg lowering `max(x,0) + log1p(exp(-|x|))`; no kernel on any energy now contains a placeholder. **M1:** `manifold = "sphere"` has a native integrator — per row the gradient and the noise are projected to the tangent plane and the step is retracted by normalization, the dot products and norms being sequential f32 row sums. Both singularities fail closed per Decision #21a instead of being repaired silently: `langevin_step` gained an optional per-row i32 status word (bit 0 entry ‖x‖ ≠ 1, bit 1 retraction underflow with the previous state kept) that the loop ORs across steps and returns, and the JIT boundary admits i32 as a storage dtype beside its existing i64/i8 residual types. All six energy × manifold programs compile through one `tessera-opt` invocation and replay; the sphere kernels carry (state, key, key, status) in registers with four ordered row reductions per step, so the device reproduces the host's sequential fold rather than merely matching within a tolerance. Verified on the Mac (CPU lane 35/35), gfx1151, gfx1201 (assertions-ON driver) and sm_120 (device 32/32 each).

Three defects in the row-program emitter surfaced doing it, all fail-open or unsafe: the pass could fail with **no diagnostic at all**; a failed slot lookup returned an empty slot that every caller then **indexed out of bounds**; and a reshape copied a slot *reference* into the same `DenseMap`, so the insertion could rehash and the reshaped row arrived empty — the actual cause of the quadratic-sphere refusal. The pass now refuses to fail without naming an op, every refusal names the value and its producer, and multi-result `linalg.generic` is admitted (the Huber backward yields dstate and dcontext from one body). Separately, registering the EBM and Clifford dialects in `tessera-opt` (T1, last slice) had turned three `tests/tessera-ir/phase7` fixtures red on main: they predated their dialects and named placeholder ops with f32 attributes the real ops reject, parsing only under `--allow-unregistered-dialect`, which stops covering an op once its dialect is registered. CI does not run that suite. All three now use the real spellings and verify with no escape flag; rotor construction and an annealing schedule are genuinely absent and are named as gaps rather than faked (Decision #29).

Remaining: `manifold = "bivector"` still refuses (it needs the Clifford `grade` op inside the EBM lowering and a build dependency); no performance measurement, so the Python-emitted kernels remain their lanes; the device packets for this slice are the test suite rather than a recorder packet.

Evidence: `tests/unit/test_ebm_native_langevin.py` (35 on the Mac, Princess-Luna and Super-Bear), `tests/unit/test_ebm_native_langevin_gpu.py` (32 on each of gfx1151, gfx1201, sm_120), `src/solvers/ebm/test/ir/passes/lower_langevin_sphere.mlir`, `tests/tessera-ir/phase_f5/row_program_sphere_status.mlir`, `tests/tessera-ir/phase7/` (19/19).

<!-- entry-fields:end -->

### 2026-09-16 — the bivector integrator and the overhead measurement close the EBM stream

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: domain-support stream, seventh slice (sync `EBM-BIVECTOR-OVERHEAD-2026-09-16`); co-owner [AD-SOLVER-IFT-1](INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1).

Outcome: **M2 and the measurement.** `manifold = "bivector"` projects the gradient and the noise onto a grade, takes the Euclidean affine step and projects once more to clear float leakage — the reference's own structure — using `tessera_clifford.grade` rather than a second grade projection in the EBM pass (Decision #31). TesseraEBM therefore depends on TesseraClifford for this integrator only, gated on that backend: built without it the integrator fails closed naming the missing option instead of silently taking a Euclidean step. `grade` and `algebra` are semantic keys and are never defaulted; the entry grade is reported per row, never repaired. The state stays exactly in the subspace over a 100-step chain, and grade 1 is admitted beside grade 2, so the integrator is parameterized rather than a hard-wired so(3).

Getting it onto the device took two compiler changes. The **Clifford expansion** gained an elementwise path for DIAGONAL blade maps (grade, reverse, grade involution, conjugation): the per-multivector extract/insert loop nest expressed a diagonal scale, which was both slower and opaque to any consumer that maps the trailing axis to lanes. It is a select against per-blade keep/negate masks, **not** a multiply by a {0, ±1} mask, because `0 * NaN` is NaN while this map *drops* a blade — the drop stays exact and the family's "sign and permutation maps, never multiplies" contract holds; rank 1 keeps the `from_elements` path the Clifford GPU skeleton route folds to scalars. The **row-program emitter** learned per-feature constant tables: a rank-1 constant read along the feature axis is a compile-time table the lane indexes, emitted as a private constant global. That carries the grade masks into a cooperative kernel and generalizes to any per-feature weight vector. Separately, the EBM lowering now **fails when one of its ops survives the pass**: a pattern that refuses an op emits a diagnostic and declines to rewrite, and the greedy driver still reported success, so the pass exited 0 with an error printed and an unlowered op in the output.

**The overhead measurement** the GA/EBM review asks for, at temperature 0 where both routes compute the same iterated descent, with the agreement checked before any timing is kept and both dispatched at the same wrapper depth. The native route is **flat in K** — about 1.1 ms whether the loop runs 1 step or 32, and whether the state is 32 or 16384 elements — because the whole loop is one launch and one host round trip; the Python-emitted route costs about 1.8 ms **per step**, because its kernel takes `(y, grad)` and the loop cannot live inside it. On gfx1151 that is 1.6x at K = 1 and 53x at K = 32. It is a dispatch result, not a kernel-speed claim: wall clock only, since neither WSL2 ROCm box exposes `/dev/kfd` and `rocprofv3` returns no records. **No lane is promoted.** The more useful finding is that on two of the three devices the cooperative kernel is the *only* compiled Langevin lane — gfx1201 has no promoted Python-emitted EBM family and sm_120 never had one — so only gfx1151 can run the comparison at all.

Remaining: promotion still waits on kernel-time attribution, which no WSL2 ROCm box can produce, and on bare-metal calibration for the NVIDIA rows; `exp`/`log` of multivectors (rotor sampling on the group) stays closed; opaque-callback energies keep the reference path; no Apple package route.

Evidence: [three overhead packets](../../../benchmarks/baselines/ebm_langevin_overhead_20260916/README.md) with source fingerprints, `tests/unit/test_ebm_native_langevin.py` (44 on the Mac, Princess-Luna and Super-Bear), `tests/unit/test_ebm_native_langevin_gpu.py` (46 on each of gfx1151, gfx1201, sm_120), `src/solvers/ebm/test/ir/passes/lower_langevin_bivector.mlir` (+ its refusals), EBM lit 16/16 and Clifford lit 19/19 on the Mac and under Tajasarus's assertions driver.

<!-- entry-fields:end -->

### 2026-09-16 — the math a kernel is allowed to contain, and four closed domain gaps

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1)

PRs: domain-support stream, eighth slice (sync `EBM-GA-GAPCLOSE-2026-09-16`); co-owner [AD-SOLVER-IFT-1](INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1).

Outcome: **The math audit found a silent wrong answer, and closing it changed what a row-program kernel may contain.** `math.sqrt` had been pinned to libdevice's rounding-explicit `__nv_fsqrt_rn` after one sm_120 row came back 1 ulp off; every *other* `math.*` op on both device routes was subject to the same vendor default and passed through unmeasured. The emitter now carries a **closed admission table**, each op declaring why its result can be trusted — `rounding_explicit`, `bit_exact`, or `measured` — and refuses anything else by name, pointing at the recorder. The declaration has a consumer: each kernel carries `tessera.row_program.math`, the recorder refuses a kernel whose declaration disagrees with the table, and a unit test holds the Python mirror against the C++ table. Measured at 16384 points per input domain on all three devices: `sqrt` and `absf` exact everywhere (the sm_120 zero **is** the pin working), `cos` 1 ulp, `exp` 2 ulp on both RDNA parts and 3 on sm_120, `log` 3 ulp. These bounds do not weaken the earlier EBM packets, which recorded zero error for *their* inputs; this is the wider statement those rows never made.

The first run of that sweep produced the finding. **`math.tanh` returned zero for every input on gfx1151**, because it lowers to `__ocml_tanh_f32` and the packager's binary serialization shipped a kernel whose entire body was one `s_endpgm`: the launch succeeded, wrote nothing, and the caller read back a plausible-looking answer. The identical module serialized with `format=isa` contains the correct 40-instruction implementation, so the body is lost in the *binary* path, not the lowering. Nothing in the packager noticed and nothing about it is specific to tanh, so `build_native_gpu_storage` now disassembles the image it is about to ship and refuses one whose kernel contains no store — sound rather than heuristic, since every kernel in this ABI writes at least one output buffer. The guard covers the AMDGPU image only; the NVIDIA cubin needs `nvdisasm`, not a matched-LLVM tool, so an equivalent silent loss on the NVVM route would still ship. `math.log1p` left the table beside tanh: the device computes `log(1 + x)`, so `log1p(-1e-6)` returned -1.0132795e-06 against the host's -1.0000005e-06, and accuracy near zero is the only reason log1p exists.

**Four domain gaps closed.** `exp` and `log` were declared with no lowering, so rotor sampling happened on the Lie algebra with the reference doing the group step; both now lower to their closed forms on Cl(3, 0), and `rotor_from_axis` lands with them (its angle is an attribute, so both transcendentals fold and the kernel is one reciprocal and a scale). The power series is deliberately not emitted — 24 geometric products is ~1500 unrolled mul-adds — so `exp` admits an operand it can *prove* is a pure bivector; the reference picks its branch from the value, and guessing would make the two disagree on the same input. **Ragged batches**: leading axes may be dynamic, the bound comes from `tensor.dim`, and one cached module serves every batch length, with the shape agreement every bilinear op requires emitted as a `cf.assert` per dynamic axis rather than assumed. The geometric product's own static-only copy of the loop nest is now the shared one. **An annealing schedule**: `langevin_step` takes its temperature either as the constant attribute or as a runtime operand, exactly one, so a cooling chain is one loop and one cooperative kernel with the temperature carried in registers — as a carried multiply, not `powf` of the index, which is also the only form the math admission table admits. A ratio of 1.0 reproduces the constant chain bit for bit on both lanes. **An opaque energy adjoint** is now refused by name: documented as refused from the first slice and never checked, it previously surfaced as a JIT pipeline error naming nothing.

Two fail-open exit statuses closed: the Clifford expansion printed its refusals and exited 0 (the same defect the EBM lowering was fixed for), and the row-program emitter's no-diagnostic fallback fired on top of a real refusal.

**Four reds that were on main and that CI cannot see, found by sweeping and fixed rather than labelled.** (a) A *product* request — native VJP storage child, typed export, checkpoint — on an already-paired module exited 0 having emitted no product: the per-function pairing guard added two slices earlier left an existing `@f__bwd` alone, which is right for a plain re-pairing run and wrong for a request that must derive something from a pairing it is about to perform. It refuses now. (b) `test_f16_cond_native_selects_branch` had been recorded as a macOS 27 MPSGraph regression, "f16 cond off 10%". It is ours: `mpsg_build_branch` built every authored-branch op at the *boundary* dtype, so an f16 branch computed internals in f16, against the ABI's own f32-internal policy — which the authored-package path in the same file states in its comment. In an f16 cond branch `matmul` is accurate to 2e-3 and `sigmoid` to 3e-4, while `silu(x @ w)`, just those composed, was **34% off**; an explicit `mul(m, sigmoid(m))` reproduced it exactly and the bf16 boundary, which already upcast, was fine. Casting once at the branch edges fixes it. Attributing a numeric red to the vendor before bisecting our own lanes is what kept this open. (c) Changing the runtime invalidated the strict route ledger and the e2e packet, both **re-measured, not re-stamped** — all 24 ledger decisions and 6 ineligible rows reproduce with identical key sets, routes and incumbents. (d) Five tests hard-coding `nvidia`/`sm_120` failed inside NVVM serialization on a host with no CUDA toolchain instead of skipping; `test_native_ssd` guarded its `rocm` lane and left `nvidia` bare. `require_native_storage_lane` is the guard, and `native_storage_target`'s own docstring had already recorded this class on 2026-09-15.

Remaining: **26 further pre-existing failures on Princess-Luna**, bisected against main built from its own sources on that box and spun out as their own work — 12 are `complex64` on `tessera.istft` failing the ROCm capability check, and the rest span the profiler callback trace, the runtime artifact ABI, the ROCm backend lit suite and four native-storage files; the Clifford **field ops** (`ext_deriv`, `codiff`, `vec_deriv`, `integral`) still have no lowering; promotion still waits on kernel-time attribution no WSL2 ROCm box can produce; no Apple package route for the loop or the products; the row-program emitter's 1024-feature ceiling still needs a second reduction level, which would change the declared reduction order and so requires re-deriving the reference; G1 (serial native-tape residency) stays deliberately unbuilt, scoped only for programs outside the row-program envelope.

Evidence: [three math-precision packets](../../../benchmarks/baselines/row_program_math_precision_20260916/README.md) with source fingerprints; `tests/unit/test_row_program_math_admission.py` (16), `tests/unit/test_clifford_exp_log_native.py` (27), `tests/unit/test_ebm_native_langevin.py` (56), `tests/unit/test_ebm_native_langevin_gpu.py` (46 + 30 device); `src/solvers/clifford/test/ir/passes/expand_exp_log_rotor.mlir`, `expand_ragged_batch.mlir` and their refusals; `src/solvers/ebm/test/ir/passes/lower_langevin_annealed.mlir`, `src/solvers/ebm/test/ir/langevin_temperature_rejects.mlir`, `tests/tessera-ir/phase2_autodiff/langevin_opaque_adjoint_rejects.mlir`; lit 491/491 on the Mac and, in **both** trees, on Tajasarus including its assertions-enabled driver; EBM 18/18 and Clifford 22/22 on the Mac, Princess-Luna and both Tajasarus trees; 302 unit on Princess-Luna, 149 on Super-Bear, 65 on Tajasarus (68 skipped: that box has no `libtessera_jit`, so every CPU-lane case skips there). Collecting the Tajasarus figures exposed that `check-ebm` / `check-clifford` print nothing and exit 0 on that host — the documented venv-only-`lit` trap — so those numbers come from running lit directly; recorded in the ROCm queue.

<!-- entry-fields:end -->

### 2026-09-17 — the ROCm host's red zone, and four causes that were not what the errors said

Owner: [COMPILER-DEVEX-1](INTEGRATED_COMPILER_PLAN.md#compiler-devex-1)

PRs: ROCm-host red zone (sync `ROCM-HOST-RED-ZONE-2026-09-17`); co-owner [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1).

Outcome: **A full unit sweep on Princess-Luna reported 31 failures, all present on main and none visible to CI** — its unit lane has neither a CUDA nor a ROCm toolchain, so every one of these tests skips there. Five were test-gating bugs fixed earlier; **23 of the remaining 26 are fixed and verified on the owning box**, and not one cause was what its error message suggested.

**Twelve looked like a missing capability and were a routing bug of ours.** Every spectral test failed with "dtype 'complex64' is not supported for tessera.istft on rocm". The capability registry was right: `rocm_gfx1151` declares `tessera.stft`/`tessera.istft` and no other ROCm entry does, because `GenerateROCMSpectralBackwardKernel.cpp` is gfx1151-only and says so in its own diagnostics. But the rest of the stack already treats a `rocm` request as a gfx1151 compilation — `native_vjp_plugins` registers its ROCm consumers under the family name and resolves them to gfx1151 internally, `jit.py` emits Graph IR for `rocm_gfx1151` when the request said `rocm`. Only the legality gate asked about the generic name, and so refused at trace time a capability the lane would then have provided. It now routes the question to the chip the request compiles for — and **routes without widening**: `supports_op("rocm", "tessera.istft")` is still False, so the generic name still inherits no proof anywhere a dashboard or audit row reads it. The `stft` half of each pair had been passing only because the check reads the *operand* dtype and stft takes a real input, which is what made this read as a complex-storage problem.

**Six were a link requirement nobody published.** `libtessera_runtime.a` carries the HIP backend objects whenever the build enabled them; a harness compiled with its own command line gets none of the CMake target's usage requirements, so it failed with a page of `undefined reference to hipMalloc` — a missing library reported as a broken ABI. The build now writes `tessera_runtime.consumer-link.txt` beside the archive, `tests/_support/runtime_link.py` reads it, and a drift test holds writer and readers together; that test caught a link line I had missed while writing it.

**One failed `check-tessera-rocm` for the whole repository** — this backend's only automated fixture coverage, which no PR check runs. A Python-driven data file inside the lit suite was Unresolved ("Test has no RUN line"). It carried `UNSUPPORTED: true` meant to prevent exactly that; on LLVM 23's lit a test with no RUN line is Unresolved *before* the marker is consulted. Moved to `tests/fixtures/` beside the x86 twin, and `test_lit_fixture_keywords.py` now fails any lit fixture with no RUN line. 68/68.

**Four were the reverse-mode gate refusing what it should not.** `arith.select` had no transpose, so reverse mode over any lowered branch — a recovered switch edge, a Huber kink, a masked manifold step — stopped dead; it is linear in its two value operands, so the taken branch receives the cotangent and the other receives zero. `arith.index_cast` was refused as non-differentiable when index arithmetic is not a differentiable variable at all; that is now a property of the *types* rather than a list of op names that would need extending every time a shape computation used one more integer op. A first cut of that predicate omitted `complex<f32>` and silently dropped six FFT transposes — recorded, because dropping a gradient is the failure the rule exists to prevent.

Remaining: **`AUTODIFF-SHAPE-WHILE-FORWARD-2026-09-17`** — three shape-varying `scf.while` tests that compiled once the gate was fixed and then crashed inside JIT-compiled code. **The mechanism recorded here on 2026-09-17 ("a shrinking dynamic iter_arg does not survive bufferization") was wrong, and is corrected in the ROCm queue under the same key:** the exported forward product still carried the `tessera.autodiff = "reverse"` *request* marker, so the JIT's unconditional paired pass differentiated it again and re-materialized its tapes, and the invoke wrote past the ABI the caller had read. Ten hand-written modules established it; the envelope-carry change written against the wrong mechanism was reverted. Separately, the x86 JIT AD lane those tests exercise has **no host in any automated check**, so defects in it surface only in a manual sweep.

Evidence: `tests/unit/test_runtime_link_requirements.py`, `tests/unit/test_lit_fixture_keywords.py`; `check-tessera-rocm` 68/68 and `tests/unit/test_autodiff_spectral_target_binding.py` 40/40 on Princess-Luna, whose full sweep is **19455 passed / 0 failed**; the Mac at **18380 passed / 0 failed**; the bisect against `4898812c` built from its own sources in worktrees on Princess-Luna and Super-Bear.

<!-- entry-fields:end -->
