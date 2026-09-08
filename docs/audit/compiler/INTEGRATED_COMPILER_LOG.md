---
last_updated: 2026-09-07
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
