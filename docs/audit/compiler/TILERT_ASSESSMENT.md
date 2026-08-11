---
last_updated: 2026-08-11
audit_role: reference
---

# TileRT assessment — tile-granular composition scheduling, with proofs

**Date:** 2026-08-10 · **Status:** assessment + design note (direction, not status
truth — Decision #26) · **Charter:** the composition layer in
[`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) §8
(not the integrated plan's autodiff W6),
TileSight T3/T4 ([`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) §3.1), and
[`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) §I ("promote comm/compute
overlap from runtime machinery to a scheduled pass").

---

## 0. What TileRT is, and how much of it we can trust

[TileRT](https://github.com/tile-ai/TileRT) (tile-ai, the TileLang group) is a
**closed-source, binary-wheel** inference runtime for ultra-low-latency LLM
decode: DeepSeek-V3.2 / GLM-5 on 8× NVIDIA B200, millisecond-level TPOT. Its one
architectural idea, per its own README: operators are compiler-decomposed into
fine-grained **tile-level tasks**, and a runtime scheduler dynamically overlaps
compute, I/O, and communication across devices — the tile as a *scheduling*
unit, not just a codegen unit.

**Claim hygiene before anything else.** No code, no IR, no scheduler description
is published ("compiler techniques will be gradually shared... as they are
integrated into TileLang and TileScale"). Benchmarks are partly synthetic with
unstated baselines. Nothing here is usable as an evidence row by our own
standards. What TileRT provides is *external validation of a direction we
already track* (W6, T3/T4), plus numbers we can cross-check analytically — §1
does that, and finds the mechanism claim coherent and the headline speedup
claim necessarily composite.

Related work already in our references: **TileLink** (arXiv:2503.20313, tile-
centric overlap *codegen* — TILESIGHT_ASSESSMENT §5 row 2 calls it "the missing
half" of our `comm_overlap.py`) and NVIDIA's tile-VM direction (the
`nvidia-tile-ir` note). TileRT is the *deployed-system* data point on the same
thesis.

---

## 1. The models

Formalism first, so the design that follows is derived, not vibes
(Decision #30 applied to ourselves).

### 1.1 M1 — the bubble decomposition and the overlap ceiling

A decode step is a DAG of ops `i = 1..n`, each with a work vector over resource
classes `R = {compute, memory, network}`:

```
w_i = (C_i, M_i, N_i)        W_r = Σ_i w_i,r
```

Lower bound over any schedule: `T* ≥ max(W_comp, W_mem, W_net, CP(G))` where CP
is the critical path.

Kernel-per-op execution (each op completes before the next starts, even with
perfect intra-op overlap): `T_barrier = Σ_i max_r w_i,r`.

Tile-granular scheduling with true data deps approaches `T_tile → max_r W_r`
(+ fill/drain). This is exactly TileSight **T3**'s objective — "min over legal
topological orders of max over resources" (TILESIGHT_ASSESSMENT.md:133) — so
the objective is already formalized in-tree; this section just names its two
consequences.

**The bubble** (what a composition layer can reclaim):

```
B = Σ_i max_r(w_i,r) − max_r(Σ_i w_i,r) ≥ 0
```

Non-negative because sum-of-maxes ≥ max-of-sums. `B = 0` iff one resource
dominates every op; **B is maximized when consecutive ops alternate dominant
resources** — which is the MoE pattern (mem-dominant expert GEMM alternating
with net-dominant all-to-all). MoE is provably the max-bubble first target, not
just intuitively.

**The ceiling.** From `Σ_i max_r w_i,r ≤ Σ_r W_r ≤ |R| · max_r W_r`:

```
T_barrier / T_tile ≤ |R| = 3
```

Pure overlap of three resource classes can never exceed 3×. TileRT's "3–4×
over baseline" therefore **cannot be pure scheduling** — it bundles kernel
quality, MTP, and/or a weak baseline.

**Batch-1 refinement.** At decode batch 1 the tensor cores are idle (~74 GFLOP
per token ≈ 2 µs of compute on 8×B200 vs ~0.6 ms of weight streaming), so
compute collapses into memory, `R` effectively = {mem, net}, and the ceiling
tightens to 2×. The layer chain forbids cross-layer compute↔compute overlap
(full dependency through the hidden state); what legally slides across layer
boundaries is **weight prefetch and communication only**. A scheduler should
not even search for overlaps the dependence structure forbids.

### 1.2 M2 — roofline cross-check of TileRT's absolute numbers

Assumptions (stated; V3.2 internals partly undisclosed): ~37 GB active FP8
weights/token, MLA-compressed KV ≈ 35 KB/token (negligible), B200 ≈ 8 TB/s HBM
→ 64 TB/s aggregate.

```
TPOT floor = 37 GB / 64 TB/s ≈ 0.58 ms   →  ~1,730 tok/s ceiling
TileRT observed: 600 tok/s ≈ 1.67 ms     →  ~35% of roofline
```

Serialized comm estimate: ~58 MoE layers × (dispatch + combine ≈ 15–20 µs of
latency-bound small messages) ≈ **0.9–1.2 ms** — same order as the entire
memory floor. Sanity: 0.58 ms floor + partially-hidden ~1 ms comm ≈ the
observed 1.67 ms. The model reproduces their operating point: **at batch 1,
comm latency is the dominant bubble**, which is the quantitative version of the
W6 row's "overlap scheduling is a top differentiator." It also shows TileRT
itself leaves ~2.9× on the table.

### 1.3 M3 — selection and composition do not commute (the new result)

The Decision #28 arbiter ranks candidates by scalar measured latency per
`(op, shape-bucket, dtype, target)`. Claim: **per-op argmin is not the argmin
of overlapped makespan.**

Counterexample. One op, two kernels: `k_A = (C=10, N=0)`, `k_B = (C=4, N=4)`.
Standalone, `k_B` wins (4 < 10) — the arbiter picks it. The surrounding step
carries 8 units of network work per instance. Steady-state overlapped makespan:

```
pick k_B:  net = 8 + 4 = 12, comp = 4    →  T = 12
pick k_A:  net = 8,          comp = 10   →  T = 10
```

The standalone-slower kernel is 20% faster end-to-end, because candidate
ranking must be against the *step's bottleneck resource*, not standalone time.

Consequences:

1. **Once any composition layer exists, scalar-latency arbitration is wrong**,
   not merely incomplete. The §4 caveat ("no T3 action-DAG ordering or T4
   overlap model") already flags this for *pruning*; M3 upgrades it to a
   *selection-correctness* statement.
2. **Record per-candidate resource vectors now**, so the measured corpus does
   not need re-measuring when composition arrives. §4.2 below shows this is a
   zero-schema-change edit.
3. Today, with no composition layer, scalar latency is correct — no change to
   current arbitration semantics is proposed, only to what gets *recorded*.

### 1.4 M4 — static vs. dynamic scheduling, and determinism

If task durations are deterministic, an optimal **static** schedule attains the
M1 bound — dynamism buys nothing. Dynamism pays only under data-dependent
variance: MoE routing imbalance and MTP's variable acceptance length. And
Graham (1966): greedy list scheduling from a work queue is within `(2 − 1/m)`
of optimal makespan — a plain queue is already near-optimal; no exotic
scheduler is required.

Two design rules follow:

- **Static-first.** Compiler-planned overlap (a planner pass, per REFACTOR_PLAN
  §I) captures the full bound for dense models and fits the existing
  planner/pass architecture. A dynamic work-queue lane is a *targeted add-on*
  for MoE/MTP variance only.
- **Determinism constraint.** Dynamic rescheduling reorders reductions →
  floating-point non-determinism, colliding with `@jit(deterministic=True)` and
  Decision #18. Rule: dynamic reordering is restricted to non-reducing tasks,
  or reduction tree order is fixed independently of arrival order. TileRT
  (inference-only, closed) never answers this; a training compiler must, up
  front.

### 1.5 M5 — MTP, for completeness

With per-position acceptance `p` and draft depth `k`, expected tokens/step
`α = (1 − p^{k+1})/(1 − p)`. TileRT's reported mean 2.77 at `k=3` implies
`p ≈ 0.76` — internally consistent. In the memory-bound regime
`T(k) ≈ T(1)(1 + εk)` with small ε (weights are read once regardless of k), so
MTP composes multiplicatively with overlap — independent levers. Their MTP
throughput claim itself is uncheckable (baseline unstated). Not a Tessera
work item; recorded so nobody re-derives it.

---

## 2. What exists in-tree (traced 2026-08-10)

The surprising finding: **the repo has independently built four pieces of this
layer, and none of them are connected.** Traced at file:line depth; the
inventory below is the ground the design stands on.

| # | Artifact | State | Evidence |
|---|---|---|---|
| E1 | **T3/T4 cost-model slots** | Named, unbuilt ("Only T1 v1 is built") | TILESIGHT_ASSESSMENT.md:133-134, :227-228 |
| E2 | **`comm_overlap.py`** — SC-HRF scopes, release/acquire `SignalOp`, `OverlapPlan` over the three Iris strategies (sequential-fused / workgroup-specialized / unfused producer-consumer, with CU partitioning) | Built, tested, **zero production consumers** | `python/tessera/compiler/comm_overlap.py`; sole consumer `tests/unit/test_comm_overlap.py`; provenance `ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` row C3 |
| E3 | **Typed async collectives** — futures + explicit awaits + SSA lineage, QoS ops, `chunk_bytes` attr | **Landed 2026-08-09** (`COLLECTIVE-ASYNC-UNIFY`) — but the await is created *adjacent to the dispatch* (`CollectiveLowering.cpp:88-92`), so the overlap window is zero; `chunk_bytes` is forwarded verbatim, never sliced | `CollectiveOps.td:13-34,138-157`; COMPILER_AUDIT.md 2026-08-09 entry |
| E4 | **Runtime chunk machinery** — `ChunkDesc`, `TokenLimiter` inflight cap, `Policy::chunkBytesForPath` (NVLink 512K / PCIe 128K / RDMA 256K), Perfetto `CommChunk` spans with `memory_bytes` | Built; chunked transport adapters are **mocks** (`Adapters.h:351-355`, :645-649) | `Execution.h:11-86`, `Policy.h:54-62` |
| E5 | **`pipeline_planner.ScheduleStep`** — `(clock, rank, stage, micro_batch, phase)`; the **only real schedule data structure in the repo** | Built — and **discarded at the IR boundary**: `to_mlir_attrs()` emits five scalars; `PipelineStageInsertionPass` re-derives | `pipeline_planner.py:51-66,313-322` |
| E6 | **MoE overlap models** — `OverlapSchedule` (chunk-event record; loop actually runs sequentially) and `megamoe_forward_pipelined` (**genuinely threaded**: dispatch(c+1) ∥ compute(c), combine(c−1) ∥ compute(c), fixed cross-rank a2a order) | Built; the pipelined path is the one *live* overlap execution in the tree | `moe.py:560-680, 710-866` |
| E7 | **Resource records** — `autotune_v2.cost_measurements()` emits `bytes_moved`; `tile_ir.py` MMA records carry `async_copy_bytes` / `queue_depth` / `barrier_count`; comm bytes exist **only as Perfetto trace events**, never as a benchmark field | Partial | `autotune_v2.py:785-812`, `tile_ir.py:440-457`, `Execution.h:57-59`, `roofline/cli_v2.py:40-46` |
| E8 | **`target_perf.py`** calibrated device parameters (the T3 time-per-action input) | Built; measured overlay empty — calibration sweeps not yet run | TILESIGHT_ASSESSMENT.md §4 |
| E9 | **Collectives overlap design draft** — ComputeQ/CommQ, credit-based link scheduler, "penalize frames where ComputeQ idles while CommQ is busy", §8 "chunk slicing + await insertion only at true use sites" | Draft; §4/§5/§8 planner unimplemented | `src/collectives/docs/Tessera_Collectives_Overlap_Design.md` |

### 2.1 Negative findings (things the tree claims that do not exist)

These matter because a reader of CLAUDE.md or the pass comments would scope
this work against phantom infrastructure.

- **`tessera.queue` is dead IR vocabulary.** Three ops with real verifiers and
  six diagnostic codes — and **zero producers, zero consumers, zero passing
  fixtures**. `WarpSpecializationPass.cpp:9,19-24,201` *claims* in comments to
  emit queue triples; the pass body never touches a queue op — the actual
  mechanism is `!tile.pipeline_state` + `!tile.async_token` SSA chains
  (`WarpSpecializationPass.cpp:95-122`). The type syntax is unparseable (MLIR
  splits `!tessera.queue.tile_queue` at the first dot —
  `QueueVerifiers.cpp:9-22`); registration in the MLIR plugin is commented out.
  A parallel Python model (`compiler/queue_dialect.py`, with the `depth`/
  `queue_id`/warp-count attributes the .td lacks, plus a fourth `barrier` op)
  is imported by nothing. **Decision #29/#31 disposition required** — see §5.
- **`CollectiveScheduler` and `ChunkPlanner` do not exist as code.** They
  appear only in comments (`Adapters.h:7,239`, `moe.py:218`) and in docs
  including CLAUDE.md's `src/collectives/` row. The named component is E9's
  unimplemented draft.
- **`src/runtime/src/scheduler/tile_scheduler.h` is not a tile scheduler** — it
  is a 34-line generic CPU `ThreadPool`.
- **Two contradictory `tile.async_copy` contracts ship simultaneously**: ODS
  `Variadic<AnyType>`, no `stage` attr, dual token/legacy form
  (`TileOps.td:513-551`) vs. a `stage`-required memref-required verifier
  (`ScheduleOps.cpp:308-333`) — while the actual C++ emitter
  (`TileIRLoweringPass.cpp:71-93`) matches neither and the Python spine
  (`tile_ir.py:204-208`) matches the latter. Any layer reasoning about async
  copies must first pick one.

### 2.2 Quantified: how much of the static-overlap half exists?

Objective **formalized** (T3, unbuilt) · correctness contract **built,
unconsumed** (E2) · IR substrate **landed 2026-08-09, window-zero** (E3) ·
runtime admission control **built, transport mocked** (E4) · schedule datum
**built, lossy at IR boundary** (E5) · one **live threaded overlap** execution
(E6) · resource records **partial** (E7) · device parameters **built,
uncalibrated** (E8). What is absent is precisely the *connective tissue*: the
await-sinker, the chunk slicer, the T3 module that reads E7+E8, and a schedule
representation that survives into IR.

---

## 3. Verdict — take, adapt, skip

### Take

| # | Idea | Why | Lands in |
|---|---|---|---|
| **R1** | **Await-sinking as the first overlap pass** — sink E3's adjacent awaits to true SSA use sites | The 2026-08-09 unification made this a small, legal dataflow transform (typed futures, explicit lineage). It converts the overlap window from zero to "whatever independent work sits between dispatch and first use" — the cheapest possible B-reclaim, dense-model-safe, fully static (M4) | new pass after `tessera-lower-tile-collectives`; E9 §8 already names it |
| **R2 — landed 2026-08-10** | **Record resource vectors in the measured corpus** (M3 consequence) | Successful measured autotune rows carry validated `tessera.measured_resource_vector.v1` metadata in the declared `hot_path_metadata` slot. The vector binds compute time, dtype-correct bytes moved, communication bytes, queue/resource identity, timing provenance, and a content digest. Provenance round-trips through the tuning cache; analytical rows cannot claim measured vectors. `composition_analysis_only` and `selector_authority = latency_ms` make the non-selection contract machine-checkable | `autotune_v2.py` + `benchmark_row.py` |
| **R3 — landed 2026-08-10** | **T3 cost module over Tile IR** — resource vectors (E7) × calibrated per-action times (E8) × legal-topological-order search | `composition_cost.py` owns validated actions/DAGs, calibration provenance, bounded deterministic order search, and compute/memory/communication lane simulation. It reproduces M3's scalar-order reversal. Only exhaustive clear losers may be pruned; bounded searches retain the candidate. Every estimate is promotion-ineligible and names scalar `latency_ms` as selector authority | `composition_cost.py` |
| **R4 — bounded functional consumer landed 2026-08-10** | **MoE dispatch/combine chunk-granular overlap** as the worked dynamic example | `megamoe_overlap.py` emits a content-addressed action DAG with contiguous slices, capacity/workspace bounds, true-use edges, ordered collectives, and deterministic combines. `megamoe_forward_pipelined` consumes that exact plan and reports its digest and execution trace. R3 only prunes; measured scalar latency selects. Mock multi-rank execution proves numerical and bitwise-repeatable behavior; native transports and exact-device packets remain target-owned | `megamoe_overlap.py` + `distributed/moe.py` |

### Adapt

- **Schedule-to-IR boundary (E5).** A tile-task schedule must survive into IR
  rather than being re-derived from scalars. Adapt the `ScheduleStep` shape
  (immutable tuple + explicit clock, three builders behind one interface) —
  do not invent a second schedule datum (Decision #31).
- **Decision #28 §4 amendment (landed by R2):** the record carries the
  candidate's resource vector alongside measured latency. R3 consumes that
  vector for prune-only composition analysis; scalar measured latency remains
  the selector, and no analytical estimate can promote a candidate.

### Skip

- **TileRT's serving stack** (disaggregated prefill/decode, vLLM KVConnector,
  NIXL/Mooncake, OpenAI endpoints) — inference-serving infrastructure;
  wrapping it violates Decision #23 and it is off-mission regardless.
- **Dynamic-first scheduling** — M4: static planner passes capture the bound
  for dense models; dynamism only for MoE/MTP variance, under the determinism
  rule.
- **Anything claimed about TileRT's internals beyond its README** — including
  persistent-kernel/megakernel structure, which is *plausible but unevidenced*
  (an earlier internal analysis stated it as fact; retracted).

---

## 4. Prerequisites and gates (honest scoping)

1. **W2.1 and W2.2 are closed.** Graph records carry registered effect,
   alias, mutation, and stochastic-identity semantics; unknown operations fail
   closed, internal-call summaries reach a fixed point, and await sinking uses
   the shared fail-closed shape/alias/liveness/memory-dependence/activity
   analysis. This removes the former aliased-RNG/name-match hole and supplies
   the legality query substrate. R3/R4 continue to consume explicit DAG edges;
   automatic edge synthesis in future Graph-derived Tile DAG producers remains
   a client integration, not missing analysis infrastructure.
2. **Multi-device overlap is unmeasurable on the current fleet** — no box has
   two GPUs; chunked transport adapters are mocks (E4). Cross-device rows are
   Phase G/H-gated. Measurable today: R1's window widening under the mock
   adapter's two-rank fixtures (software contract), R2's records on gfx1151 /
   sm_120 / Apple, R4's threaded pipeline on any box. Per M1's batch-1
   refinement, single-device overlap gains are structurally small — do not
   oversell the near-term win; the payoff is having the layer *correct and
   measured* before Phase G/H silicon arrives.
3. **This note is direction.** MASTER_AUDIT and `docs/audit/generated/` remain
   status truth (Decision #26).

---

## 5. Cleanups this trace surfaced (independent of the design)

Disposition after PR #544:

1. **`tessera.queue` MLIR dialect — deleted.** The unparseable dead dialect and
   orphan Python dialect module are gone. Internal Python pipeline markers
   remain a compatibility carrier and must migrate to explicit Schedule/action
   DAG plus token lineage under `COMP-SCHED-OVERLAP-1`; they are not evidence
   that the deleted dialect is live.
2. **Phantom `CollectiveScheduler`/`ChunkPlanner` names — corrected.** Live
   documentation now points to the real runtime/planner surfaces.
3. **Dual `tile.async_copy` contracts — reconciled.** ODS is authoritative:
   typed `!tile.async_token` is production, while stage/barrier grouping keys
   are optional compatibility inputs. Remaining work is producer migration,
   not another contract decision.
4. **`tile_scheduler.h` misnomer — corrected/documented.** It no longer serves
   as evidence of a production tile scheduler.

The executable follow-ons R1–R4 are bound to the integrated plan as
`COMP-SCHED-OVERLAP-1`, rows W5.2a–W5.2d. This assessment remains rationale,
not a competing work queue.
