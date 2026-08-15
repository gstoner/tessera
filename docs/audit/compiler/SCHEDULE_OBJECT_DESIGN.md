---
last_updated: 2026-08-15
audit_role: plan
plan_state: open
---

# The Schedule Object — one representation for roles, actions, and residency

> **Routing:** start at [`README.md`](README.md). This design owns the shared
> schedule representation's contract; global ordering lives only in
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md). It is P3 of
> [`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md)'s build sequence (S2),
> written *before* any of its three consumers lands separately — that is the
> point.
>
> **Reads against:** [`compiler_enhancement.md`](compiler_enhancement.md)
> Phases 2–3 (roles + stated entry point),
> [`TILERT_ASSESSMENT.md`](TILERT_ASSESSMENT.md) E5/M3/M4 (the IR-surviving
> schedule datum, resource vectors, determinism), W5.2c/e/g (the action DAG
> that already runs host-side), and
> [`FORGE_ASSESSMENT.md`](FORGE_ASSESSMENT.md) W2 (residency as a schedule
> property).

---

## 0. Why one object, and why a design doc first

Three landed-or-planned workstreams each need "the schedule" as data, and each
currently has (or plans) its own shape:

| Workstream | Its schedule datum today |
|---|---|
| CAKE Phase 2/3 | Roles + barrier producer/consumer sets as new Tile IR; a Python builder that *states* a schedule (does not exist yet) |
| TileRT E5 / W5.2 | `pipeline_planner.ScheduleStep` — "the only real schedule data structure in the repo," **discarded at `to_mlir_attrs()`** and re-derived from scalars downstream; separately, W5.2c/e's `CompositionCandidate` action DAG with R2 resource vectors, host-side only |
| FORGE W2 | `tessera.residency` as a value property a boundary verifier checks — a schedule-visible fact with no schedule to live on |

Decision #31 says one boundary gets one representation. Landing any of the
three first and "converging later" is the documented way this fails (the
ordering caveat in the integrated plan). So the object is specified once,
here, and each consumer becomes an entry point or a view of it.

## 1. The object

A **ScheduleObject** is a content-addressed value with four components:

```
ScheduleObject = {
  actions:   [Action],       # nodes: op ref, tile/loop scope, resource vector
  edges:     [Edge],         # data | sync | resource, each with a WHY
  roles:     [Role],         # named warp/wave sets; producers/consumers per barrier
  residency: {value: tier},  # tile | layer | full, per intermediate (FORGE W2)
  digest:    sha256(canonical serialization)
}
```

* **Action** — one schedulable unit: a Tile-level op (or fused region) plus
  its measured-or-analytical resource vector in the W5.2b
  `tessera.measured_resource_vector.v1` schema. Actions are what the W5.2c
  cost model orders and what the arbiter's composition analysis prunes.
* **Edge** — `data` (SSA def-use, synthesized by W5.2e from W2.1 facts),
  `sync` (the arrive→wait / token / barrier edges the tile-dataflow legality
  pass derives — the S1 vocabulary), or `resource` (same-queue serialization).
  Every edge carries a machine-readable *reason* (which analysis emitted it),
  so a conservative edge is distinguishable from a proven one.
* **Role** — CAKE Phase 2's vocabulary: a named set of warps/waves; each
  barrier names producer and consumer role sets. Roles carry **no physical
  warp ids** (lowering derives identity); the CDNA ping-pong rotation and the
  Hopper producer/consumer split must both be expressible as role data, not
  as branches (Phase 2 exit gate 2 is the acceptance test of this design).
* **Residency** — the FORGE W2 tier per intermediate value. It lives here
  because residency is a *scheduling decision* (which tile loop owns the
  value's lifetime), and the boundary verifier that proves it needs the
  schedule to name the tile scope.

## 2. Two entry points, one IR carrier

**Derived** (the default `@jit` path): lowering constructs the ScheduleObject
as it makes decisions — today those decisions exist only as side effects
(barrier counts from op choices, stage depths from attrs). The construction
points already exist: `TileIRLoweringPass`, `WarpSpecializationPass`,
`NVTMADescriptorPass`, and the ROCm wave-LDS planner.

**Stated** (CAKE Phase 3): a Python builder (`@tessera.schedule`) constructs
the same object directly — same validation, same digest, same lowering. Not a
second IR: a constructor for this one.

**The IR carrier** — the E5 lesson is that a schedule must survive into IR
rather than be re-derived from scalars. The carrier is:

1. the **digest** stamped on the owning func/module
   (`tessera.schedule_digest`), binding IR to the full object out-of-band
   (the content-addressed pattern W5.2d already ships for MoE plans); plus
2. the **sync and role components materialized as IR** — barriers with
   producer/consumer role operands (CAKE Phase 2's ops), tokens, pipeline
   state — because those are what verifiers derive against; plus
3. **residency as a value attribute** the boundary verifier consumes.

What deliberately does **not** get IR ops: actions' resource vectors and the
edge list (they live in the object, keyed by digest — measurement data is not
IR), and physical warp identity (derived at lowering, per CAKE's own rule).

## 3. Contracts (each is a gate, none is optional)

1. **One-rule roles.** The CDNA ping-pong schedule and the Hopper
   producer/consumer split verify through the same role→barrier→role
   reachability rule with no target branch (CAKE Phase 2 gate 2). A negative
   verdict re-scopes this design before anything depends on it.
2. **Loop-carried roles and barriers survive block arguments** — resolved by
   the same `TileValueProvenance` discipline the tile-dataflow legality pass
   ships (P1a); a role that rotates per phase is the acceptance fixture.
3. **No scalar re-derivation.** A consumer that needs the schedule reads the
   object by digest; `to_mlir_attrs()`-then-reconstruct is deleted, not
   wrapped (E5).
4. **Determinism rule (TileRT M4).** Any dynamic reordering is restricted to
   non-reducing actions, or the reduction tree order is pinned independently
   of arrival order — `@jit(deterministic=True)` and Decision #18 are not
   negotiable inputs.
5. **Residency proofs ship with fusions.** A fusion that claims
   `residency = tile` carries the generalized `LOWER-COUNT-1` check (FORGE
   W2); a claim without a proof is a Decision #29 violation.
6. **Expectation ceiling.** Single-box overlap reclaim is bounded by the
   verified 2×/3× ceilings (`research/core_substrate/`); the object's payoff
   is correctness, attribution, and Phase G/H readiness — not near-term
   throughput. Written here so nobody re-sells it otherwise.

## 4. Consumers (Decision #29 — named, or the object is not built)

| Consumer | Reads |
|---|---|
| W5.2c composition cost + W5.2g scheduler | actions + edges (already does, host-side — this design gives its input a name and a digest) |
| Tile-dataflow legality (P1a, W2.4) | sync edges + roles, derived from IR against the object's claims |
| FORGE W2 residency verifier | residency tiers per value |
| CAKE Phase 3 builder | constructs the object |
| Schedule-level autodiff (S8 ceiling) | transposes the object: reverse edges, swap producer/consumer role sets, re-derive sync |
| Causal bottleneck attribution (CAKE capability #2) | maps a measured stall to the action/edge that caused it |

## 5. Non-goals

* **No dynamic scheduler.** Static-first (M4); the MoE/MTP variance lane is a
  later, bounded add-on under contract 4.
* **No physical warp ids in the object**, ever.
* **No second schedule datum.** `ScheduleStep` migrates into (or is deleted
  in favor of) this object; the same applies to any per-plan "plan" dicts
  that grow schedule-shaped fields.
* **No new serialization format** — the canonical form is the existing
  content-addressed JSON the W5.2d plans use, extended with the role and
  residency components.

## 6. Build order (proposal — the integrated plan owns the slot)

1. **SO-1**: the Python-side object + digest + validation, with
   `CompositionCandidate`/`infer_action_dag` re-based onto it (pure
   refactor-with-tests; no IR change).
2. **SO-2**: CAKE Phase 2's role/barrier IR (producer/consumer role operands
   on `tile.mbarrier.init`), verified by extending the P1a derivation pass;
   the ping-pong/Hopper one-rule gate lives here.
3. **SO-3**: digest stamping + the E5 migration (delete the scalar
   re-derivation in `PipelineStageInsertionPass`).
4. **SO-4**: residency attribute + the generalized materialization-proof pass
   (FORGE W2), consuming the object's tiers.
5. **SO-5**: the stated entry point (CAKE Phase 3's builder), last — it
   constructs an object every verifier already knows how to judge.

Every step is host-free except SO-2's final numerical fixture (gfx1151, per
the fleet routing).
