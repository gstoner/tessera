---
last_updated: 2026-09-04
audit_role: plan
plan_state: open
---

# IKF-1 — intra-kernel feedback: per-instance measurement as compiler training data

Scoped plan for an intra-kernel measurement foundation whose primary consumer
is the **compiler's own cost models and arbiter**, not a human with a timeline
viewer. Ordering authority stays with
[`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) — **this plan is
bound to integrated-plan item IKF-1**. P0 clock validation follows the
existing D2 timing contract; P2 consumes the Schedule-Object region contract
and remains gated on green P0 evidence. The compiler map and authority chain are in
[`README.md`](README.md). This document owns the design and acceptance detail
only.

**Provenance.** Derived from a source-level review (2026-08-27) of NVIDIA
CUTLASS's experimental **IKET** (In-Kernel Event Tracing) for the CuTe DSL —
the `iket.*` MLIR ops stripped by default, token/stack range pairing, the
`sentinel_token` loop-carried design, `run-iket`'s two-pass buffer sizing, the
Perfetto/JSON trace schema with interned string/location tables, and the
undocumented pure-metadata `iket.dag()` producer/consumer graph. Reviewed
sources: `docs.nvidia.com/cutlass/.../iket_profiling.html` and
`python/CuTeDSL/cutlass/cute/experimental/iket/{iket,dag}.py` in
NVIDIA/cutlass. IKET is the strongest public prior art for
compiler-integrated in-kernel instrumentation; §9 records what we deliberately
do **not** copy and why.

---

## 1. Problem statement

Tessera's optimizer makes load-bearing decisions from analytic proxies that
contain no measured time:

| Decision site | Proxy | Gap |
|---|---|---|
| `fusion_core.FusionCost.score` | `dispatch_saved + bytes_saved/MB`, unitless; `should_fuse_region` gates on `score > 0` | no coefficient a measurement could correct |
| `fusion_core.AttnLoweringCost` | total off-chip bytes | same |
| `emit/candidate.arbitrate` | tier priority + `_mma_footprint_cost` tie-break | D2's `measure` seam exists but the measurement is a scalar |
| `emit/autotune.MeasureRecord` | one `latency_ms` per candidate | no credit assignment: *which region* got faster is invisible |

Meanwhile the existing intra-kernel surface is a Decision #29 violation in
waiting: `profiling_plan.IntraKernelProbe` +
`target_ir.annotate_target_ir_with_probes` emit six per-target
`*.profiler_probe` ops (declared in ODS after the PR #490 review) that **no
pass consumes**, with a hardcoded `("prologue","mainloop","epilogue")` phase
list prepended to the *front of the function body* — placement that would make
the timings meaningless even with a consumer. `tools/profiler/` is strong on
the host/provider side (CUPTI, ROCprofiler-SDK, Metal, Perfetto export,
claim-disciplined timing artifacts); intra-kernel is the hole.

**Thesis.** The highest-value form of intra-kernel measurement for Tessera is
not a trace — it is the **label set that makes the analytic cost models
fittable** and the arbiter's verdicts explainable. The human-facing timeline
is a view over that record, never the record.

## 2. Design principles (settled 2026-08-27)

- **C1 — the region is a compiler object, not a user string.** Region
  identity derives from what the compiler already has: the `fusion_core`
  region classes (`MatmulRegion`, `AttentionRegion`, `NormChainRegion`,
  `PointwiseGraphRegion`, `GatedMatmulRegion`), Schedule IR pipeline stages,
  and `tessera.schedule.warp` roles. Derived identity is stable across runs,
  boxes, and source edits — which is what makes records joinable and
  diffable. The record key **extends** the existing D2 autotune key
  `(device, target, op, bucket, dtype, timing)` with a region axis and an
  `instr_level` field, rather than inventing a parallel taxonomy.
- **C2 — index by schedule coordinates, attribute hardware coordinates.**
  The index is `(role, wave-in-role, stage, tile-coord,
  iteration-stride-index)` — the coordinates the schedule is written in and
  the levers the tuner turns. `wave-in-role` is required for uniqueness, not
  decoration: a role is a *group* (`wave_specialization.py::
  sm90_attention_plan` places one producer and **four** consumer waves in the
  same role), so a key without it has four writers per slot and the
  write-once guarantee of §3 is false. Single-writer election per role group
  was considered and rejected: intra-role imbalance is signal, and per-wave
  slots are what make it visible. Waves-per-role is compile-time known, so
  the size formula stays closed-form.
  Physical placement (CU/SE/SM ids) is an attribute for the imbalance lens,
  not the key. This is the inverse of IKET's `locationTable`-primary schema,
  and it is what makes cross-run aggregation portable across vendors.
- **C3 — attribute to the decision.** Every region record carries the
  provenance of the choice that produced the code: fusion verdict + its
  `FusionCost` inputs, attention lowering variant + IO score, arbiter tier +
  candidate name, tile/stage config, selected MMA shape. No external profiler
  can have this (it does not survive to a binary); it is the argument for
  building in-tree. It also gives **Decision #32 a paying consumer**:
  carrying `numeric_policy`/`layout` down to codegen is what lets a slow
  region be explained in the terms the contract was stated in. The carrier
  itself is integrated-plan order 3b (**NUMPOL-CARRIER-1**, steps 1–2 landed
  2026-08-25) — IKF's decision provenance reads that carrier rather than
  inventing a parallel channel.
- **C4 — vector-valued measurement.** Per-region measurements de-confound
  credit assignment: a scalar says config B beat A; the vector says which
  region changed and by how much. This is what makes the fleet corpus
  transfer as *coefficients* rather than winner names.
- **C5 — stalls are classified, not eyeballed** (§5). The dependency DAG is
  derivable (W2.1 dependence + the pipeline/collective passes' sync
  mechanism = IKET's hand-written `via=` field, derived); classification is
  an offline join over per-instance endpoints.
- **C6 — instrumented runs explain; uninstrumented runs decide.**
  Instrumentation is a code-motion barrier (IKET says so explicitly; it is
  inherent). The arbiter and `MeasureCache` may never select on a record with
  `instr_level ≥ 2` (§7). Instrumented and provider-sampled records may share
  a **viewer**, never a provenance field.

**Resolved forks (owner, 2026-08-27):** accumulate (not event-log), per
instance (not aggregate-only), ROCm gfx1151 first lane.

## 3. Record model — indexed per-instance slots

The reconciliation of "accumulate" + "per-instance": **every dynamic region
instance owns a pre-addressed fixed-size slot, written exactly once.**
Sub-events within an instance accumulate in registers and flush in the single
store at instance end.

- The instance space `(region, wave-in-role, tile-coord,
  iteration-stride-index)` is enumerable **statically from launch geometry +
  loop bounds + the schedule's waves-per-role** — which CuTe DSL cannot do
  and Tessera can. Buffer size is closed-form at compile/launch time:
  `bytes = Σ_r waves(r) × instances(r) × slot_bytes`.
- This deletes IKET's worst hazards: no two-pass sizing run, no
  illegal-memory-access when per-warp event counts drift between passes, no
  append atomics, no contention — and slot *i* means the same instance in
  every run (the pairing property, §6).
- Under persistent-CTA / work-stealing scheduling, tile-coord is dynamic but
  the scheduler-issued work index is still unique per instance. Guard anyway:
  sentinel-init the buffer; host validates written-slot count and detects
  double-writes. A colliding index corrupts silently; the validator makes it
  fail closed **in reporting**.
- Slot contents are **level-dependent, and the level determines which §5
  analyses the capture can feed** (register accumulation destroys individual
  endpoints, so an aggregate can never be joined after the fact):
  - **L2 slots** (phase instances): start/end timestamps on the global clock
    (§4) plus *aggregate* stall-cycle and stall-count accumulators. A phase
    contains many waits and publishes; L2 keeps only their sums, so L2
    supports aggregate stall *fraction* per phase — it cannot support the §5
    producer-attribution join or the realized critical path.
  - **L3 slots** (dependency-edge instances): regions one-to-one with the
    events §5 joins — each consumer wait interval `[w₀,w₁]` is its own
    instance, and each producer **publish is a recorded endpoint** (a
    publish-timestamp field on the producer's iteration instance, or its own
    instance for multi-publish iterations). §5's stall classification and
    realized critical path are **L3 analyses by definition**; the stride/
    window restriction above keeps them affordable.
  - All levels: validity/flag bits (preemption-suspect, sub-threshold). u64
    pairs first; u32-delta-vs-wave-base compression is a recorded later
    optimization (wrap ≈ 40 s of kernel time at a 100 MHz clock — flag,
    don't assume).

**Instrumentation levels.** L0 none · L1 kernel boundary (provider-owned,
unperturbed) · L2 phase-level per-instance (default when profiling is on) ·
L3 per-iteration (opt-in). Size and cost anchors at ~32 B/slot:
L2 ≈ 6 phase regions × 10⁴ tiles ≈ **2 MB** (always affordable);
L3 ×64 k-iterations ≈ **123 MB** (not). L3 therefore takes a
**deterministic iteration stride** and/or a slot-space window — IKET's
`--enabled-cluster` idea reborn as index restriction rather than dump
filtering; deterministic stride preserves pairing. Per-instance cost (two
counter reads + register accumulate + one 32 B store ≈ tens of cycles against
10⁴–10⁶-cycle instances) is naturally sub-1 % at L2; L3 puts reads inside the
mainloop, which is where the code-motion perturbation lives — that, not data
volume, is why it is opt-in. The compiler computes the size formula and
**degrades with a diagnostic** above budget (performance key, Decision #21a).

## 4. Clock contract — the mathematical constraints

1. **One domain.** AMD exposes a variable-frequency shader-cycle counter and
   a constant-rate realtime counter (`wall_clock64()`, rate queryable —
   already proven on gfx1151 by `tprof_rocm_timing`). Cycles→ns via any
   single frequency is wrong under DVFS; cross-wave subtraction on the shader
   clock is meaningless across clock domains. **All timestamps come from the
   constant-rate domain.** Granularity is the wall tick (~10–40 ns at
   queried rates; comparable to IKET's 32 ns).
2. **Minimum-duration rule.** With tick *q* and read-pair cost *c*, a region
   with duration ≲ k·(q+c) is quantization noise. Records carry a validity
   flag; analysis **refuses** sub-threshold regions rather than reporting
   them (same posture as `tprof_timing.py`'s clock-domain rejection).
3. **Overhead subtraction.** Calibrate read-pair cost as a distribution;
   subtract the **minimum**, never the mean. Mean-subtraction yields negative
   durations on short regions; min-subtraction is conservatively biased and
   the residual bias is a *reported* uncertainty.
4. **Cross-CU validity is measured, not assumed.** Stall classification and
   critical-path math rest on the realtime counter being device-globally
   consistent. Gate: a ping-pong experiment (wave A stores timestamp+flag;
   wave B on another CU observes and timestamps; observed orderings must
   respect the clock) on the Strix Halo box — phase IKF-P0, before any IR
   work.
5. **Preemption.** CWSR (and WSL2 scheduler noise) appears as real elapsed
   time inside a region on the wall domain: detectable as tail outliers,
   flaggable, not subtractable.

## 5. Offline analysis — stall classification and critical path

**Input requirement: these analyses consume L3 records** — slots whose
regions are one-to-one with wait intervals and publish endpoints (§3). An L2
capture supports only per-phase aggregate stall fractions; it cannot name the
producer, and a critical-path claim from L2 data is refused, not
approximated.

Per-instance endpoints make stall classification an **offline join**, not
in-kernel logic. Consumer instance records wait interval [w₀,w₁]; the matching
producer instance records publish time *p* on the same clock:

- *p* ≤ w₀ → healthy (producer ahead; no lever);
- w₀ < *p* ≤ w₁ → waited on producer; recurse into the producer's records for
  that window to split *producer busy* (lever: producer's lowering /
  bandwidth) from *producer itself blocked* (back-pressure; follow the
  chain). Insufficient-stages shows as blocked-with-idle-producer (lever:
  stage depth / issue order).

**Async-edge caveat:** for hardware-async copies the producer records *issue*
completion; data arrival is observed only at the consumer's wait return, so
transfer completion is **bounded** in [issue_end, wait_return]. The classifier
reports the interval, never a point estimate.

**Realized critical path** per launch is *computed, not modeled*: longest path
over the instance graph with actual timestamps. Derived per-region
**criticality** (fraction of instances on the realized critical path) is the
sensitivity signal the tuner wants.

## 6. Cost-model fitting contract — the three fallacies designed out

1. **Additivity fallacy.** Kernel time is a max over DAG paths, not a sum of
   region times; additive regression of latency on region durations is
   misspecified and over-credits shortening non-critical regions. Rule: fit
   per-region models; compose predictions through the DAG longest path.
2. **Occupancy cliffs.** Latency vs (M,N,K) is piecewise (tile quantization,
   occupancy steps). Fit **within** the existing D2 shape bucket, and model
   in tile counts (`ceil(M/tile_m)`), not raw dims.
3. **Numerics.** Slots hold raw integer cycle pairs; **all** moments are
   computed on the host in f64/arbitrary-precision int. Never the one-pass
   `Σx²−(Σx)²/n` form on device in f32 (cancellation); never device-side
   `Σx²` at all (u64 sits near overflow at realistic counts).

Guarded identities in the tooling: Σ instance durations ≠ wall time
(parallelism); utilization requires occupancy weighting — printed as derived
fields with correct denominators.

**Prior-band sanity gates.** The first fitted coefficient — ns per byte of
avoided intermediate traffic, which `FusionCost` currently prices as
"1 MB ≡ 1 dispatch" by fiat — has a physical prior: ≈ 2/BW_effective (a write
plus a re-read), with device HBM GB/s already in the tprof peaks YAML. A fit
outside a sane band around the roofline value means the **experiment** is
broken. Every fitted coefficient ships with such a band where one exists.

**Pairing.** Deterministic slots mean instance *i* is the same tile in two
builds, so regression detection uses **paired per-instance differences** —
cancelling grid-position and scheduling variance for far higher statistical
power than comparing two latency distributions. This is a direct consequence
of indexed slots and is unavailable to any event-log design.

## 7. Perturbation and claim-discipline gates

- **Acceptance is statistical.** N uninstrumented + N instrumented launches,
  same seeds; accept only if the median delta is within the level's budget
  with a bootstrap CI excluding gross violation. Point comparisons are not a
  gate.
- **Amortization is verified, not assumed.** "Per-point overhead = Δ/points"
  is only claimable after measuring at two instrumentation densities and
  checking linearity; otherwise the report says *total*, not *per-event*.
  (Generalizes IKET's own overhead methodology into a testable gate.)
- **Arbiter guard.** Records carry `instr_level`; `emit/autotune.MeasureCache`
  refuses entries with level ≥ 2. Decide on L0/L1, explain on L2/L3, never
  mix. DVFS/thermal drift between decide-runs and explain-runs is handled by
  gating both on the existing SMI/context capture artifacts.
- Instrumented in-kernel records and provider-sampled records are different
  epistemic classes: shared **viewer** (`tprof-merge-trace` display), never a
  shared provenance field.

## 8. User surface

- **Zero annotation by default.** `@jit(profile="phases")` yields correct,
  auto-placed L2 instrumentation with names derived from Graph IR ops and
  schedule structure — the structure IKET's tutorial author transcribes by
  hand in 60+ `range_push` calls. The manual API exists for
  `@custom_primitive` and hand-written kernels; **the annotation API is the
  fallback, not the interface.**
- **`why`.** The arbiter already computes candidate lists, per-candidate
  latency, and F4 verdicts. "The tier-1 synthesized lane ran because the
  hand-tuned lane failed the accuracy budget at this bucket" is a decision
  fact, surfaced from provenance (C3) without any timing run.
- Divergent-payload reads and mispaired ranges are **diagnostics**, not
  IKET's "undefined profiling results" / first-active-thread silences.

## 9. Deliberately not copied from IKET

| IKET choice | Why not |
|---|---|
| Free-text region names, hand placement | C1/C2: identity must be derived to be joinable; Tessera has the IR to derive it |
| Event log + two-pass buffer sizing | §3: indexed slots are deterministic-size, collision-checked, pairable; the sizing-pass IMA hazard disappears |
| `locationTable`-primary indexing | C2 inversion: schedule coordinates are the portable key |
| Perfetto as primary artifact | View over the record, never the record |
| Warp-uniform-only + first-active-thread payloads under divergence | Silent wrong answer where Tessera wants a diagnostic |
| 32-char names, 30-unique-name encoding cliff | Encoding artifacts; intern names, record encoding width in the artifact |
| Hand-written `dag.edge(..., via=...)` | Derivable from W2.1 dependence + pass-known sync mechanism |
| "Use Nsight in a separate run" for overlap | Tessera merges views in tprof; claims stay separated by provenance (§7) |

Kept from IKET, adapted: strip-by-default IR ops with opt-in lowering; the
sentinel-token idea for loop-carried range identity (as typed token +
validity flag if/where a manual range API is exposed); the
overhead-comparison methodology (§7, hardened); single-window capture
restriction (§3, as index restriction).

## 10. Delivery phases

| Phase | What | Box | Acceptance gate |
|---|---|---|---|
| **IKF-P0** | Clock validation: monotonicity, cross-CU ping-pong, read-cost distribution, queried rate — extends the gfx1151 `tprof_rocm_timing` probe | Strix Halo | Recorded evidence packet; **no IR work until green** |
| **IKF-P1** | Artifact schema `tessera.profiler_intra_kernel.v1` (slot layout, `instr_level`, validity flags, decision provenance) + host math library (f64 stats, paired diff, perturbation gate, min-subtraction calibration) | any (host-independent; say which) | Unit tests incl. the §4/§6 numerical edge cases on synthetic traces |
| **IKF-P2** | Region contract in the compiler: identity from `fusion_core` region classes + Schedule IR roles/stages; typed trace ops at Tile IR; strip-pass **on by default**; static size formula + budget diagnostic | Mac or Strix | Each ODS op names its consumer (Decision #29); **negative lit fixture: default pipeline emits nothing** (`CHECK-NOT`, Decision #10a) |
| **IKF-P3** | ROCm lowering (`wall_clock64` pairs → indexed slot stores) + runtime buffer alloc/memset/readback; L2 on the compiled WMMA GEMM lane | Strix Halo | Perturbation gate passes at L2; slot validator clean; `check-tessera-rocm` run locally |
| **IKF-P4** | Offline analysis: realized critical path, stall join with async bounds (requires a strided **L3** capture per §3/§5 — L2 aggregates cannot feed the join), paired regression diff; first β fit for `FusionCost` with roofline prior band | Strix (capture) + any (analysis) | β in prior band across ≥ 3 shape buckets; join runs only on records whose level admits it |
| **IKF-P5** | Feedback: fitted coefficients as **versioned, device-keyed measured overrides** (analytic defaults remain the fallback — a box without a corpus is never worse than today); `MeasureCache` level-guard; `why` surface | — | Drift test: an override without provenance (device, level, fit version) is rejected |
| **IKF-P6** | Later: Apple lane (forces the Python-synthesizer / MLIR seam the Apple audit names), NVIDIA `%globaltimer`, regime segmentation (prologue/steady/drain), L3 sampling policy | per hardware | per-lane exact-device proof; no parity claims transfer |

**Cross-backend synchronization key: `IKF-INTRA-KERNEL-CONTRACT-2026-08-27`.**
This plan proposes a shared artifact schema, Tile IR trace ops, and a runtime
buffer contract, so per `AGENTS.md` all four architecture queues carry a
disposition under that key: **ROCm** follow-up required (owning lane, P0/P3);
**NVIDIA** follow-up required at P6 (`%globaltimer` lowering; sm_120 role
structure differs — no gfx1151 evidence transfers); **Apple** follow-up
required at P6 (the lowering must cross the Python-synthesizer/MLIR seam, and
an in-kernel constant-rate timestamp primitive is **unverified** on Metal —
ground it in SDK headers per Decision #27 before scoping); **x86** deferred
with reason (CPU intra-kernel visibility is already owned by the TPROF-X86
rdtscp/perf/IBS lanes; indexed-slot instrumentation of tiled CPU kernels —
invariant TSC as the constant-rate clock — waits on a measured need). No
device claim is made for any backend by this plan; each promotion needs that
backend's exact-device packet.

Ordering rationale: P0 first because every downstream claim rests on the
clock assumption; P1 before P2/P3 so the schema is fixed by the math
requirements, not by what the first lowering found convenient; P4 is
deliberately the *customer* of P2/P3 so the region contract is shaped by what
the fit needs (global-clock endpoints, stable identity, pairing), not by what
is easy to emit.

## 11. Risks and fallbacks

- **WSL2 scheduling noise** inflates tails → paired stats + preemption flag;
  tail claims quote quantiles with instance counts.
- **DVFS/thermal drift** across sessions → context-gated runs (existing SMI
  artifacts); no cross-session comparison without matching context.
- **Cross-CU clock assumption fails at P0** → degrade, don't collapse:
  per-CU-relative analysis only; stall classification weakens to bounds;
  within-wave critical path survives.
- **Slot-index collision** under exotic schedulers → host validator fails the
  report closed (§3).

## 12. Decision alignment

#10a (negative fixture for the strip pass) · #19 (region contract above the
hardware-free Target IR line; per-backend timer lowering below it) · #21a
(budget overflow degrades with a diagnostic — performance key) · #26 (every
hardware number in this plan is produced on the box with that hardware; WSL
correctness-not-timing caveats carried where they apply) · #29 (every
declared trace op has a named consumer: the lowering pass, and P4's fit as
the customer) · #30 (regions, DAG edges, and `via` are derived from the
analysis layer, never asked of the user) · #32 (decision provenance is the
first consumer that pays for attribute carrying).

## 13. Open items for the owner

1. **Integrated-plan queue entry — bound 2026-09-04:** IKF-1 starts with
   the existing D2 timing contract's P0 probes. P2 consumes Schedule-Object
   region identity and must wait for P0 clock validation; it does not create
   a parallel region taxonomy. P1 host mathematics may proceed independently.
2. **Disposition of the existing `*.profiler_probe` ops** — IKF-P2 either
   subsumes them (they become the lowered form of the new contract) or they
   are deleted per Decision #29; keeping them unconsumed is the one
   non-option.
3. Whether `@jit(profile=...)` is the final user spelling (it must compose
   with `deterministic=True`: instrumentation is effect-free but
   perturbation-relevant).

## P0 baseline evidence — 2026-09-04

The existing native probe was rebuilt with ROCm for exact gfx1151 and run on
Princess-Luna WSL. The [normalized timing artifact](../artifacts/ikf/gfx1151-wsl-clock-baseline-20260904.json)
binds the probe binary digest and source state. For 32 repetitions the device
clock reports 2.655480 ms, HIP events 2.665968 ms, and host wall 2.709466 ms;
queried wall-clock rate is 100 MHz. All three validity gates pass. The artifact
explicitly remains promotion-ineligible and records that ROCprofiler activity
was not collected.

This is a baseline, not P0 closure: cross-CU ping-pong, monotonicity stress and
read-cost distribution still need dedicated probes. Consequently no P2 IR
instrumentation or fitted-cost-model promotion follows from this packet.
Princess-Luna LLVM 23 reports assertions OFF; it cannot supply assertions-on
MLIR contract evidence. Super-Bear has an RTX 5070 visible in WSL, but no
`llvm-config` on the non-interactive PATH used for this inspection.
