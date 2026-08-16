---
last_updated: 2026-08-15
audit_role: reference
verification: every previously prose-only mathematical claim this view inherits
        is machine-checked — 13/13 in research/core_substrate/verify_substrate_math.py
        (CAKE §2 statistics + §7 filter gate; TileRT M1–M5; two game-theory
        closed forms). Sources with their own harnesses are not re-verified.
---

# Core Substrate View — one integrated read across the seven capability papers

> **Routing:** start at [`README.md`](README.md). This is a `reference`
> synthesis: it maps what seven independently-reviewed papers/plans demand of the
> **core compiler**, identifies the shared substrate so one investment serves
> many capabilities, and names the owning integrated-plan row for each piece.
> It mints **no** work-item IDs and owns **no** ordering — global sequencing
> stays in [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md)
> (Decision #26; README authority chain). Where a substrate piece has no owning
> row, that fact is flagged as an input to the integrated plan, not resolved
> here.
>
> **Sources (all reviewed in-tree):**
> [`SPARDA_REVIEW.md`](SPARDA_REVIEW.md) ·
> [`TILERT_ASSESSMENT.md`](TILERT_ASSESSMENT.md) ·
> [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) ·
> [`PDE_STENCIL_CAPABILITY_PLAN.md`](PDE_STENCIL_CAPABILITY_PLAN.md) ·
> [`GAME_THEORY_PLAN.md`](GAME_THEORY_PLAN.md) ·
> [`compiler_enhancement.md`](compiler_enhancement.md) (CAKE) ·
> [`FORGE_ASSESSMENT.md`](FORGE_ASSESSMENT.md) (folded 2026-08-15 — it landed
> mid-arc as PR #565).

---

## 0. Why one view

Each of the seven documents was written as its own extraction: a paper or domain
reviewed, verified, and mapped to Tessera. Read together, they are not seven
capability requests — they are **repeated hits on a small set of core-compiler
seams**. Every paper independently lands on some subset of: derive facts the
verifier currently takes on faith; make the schedule a first-class object; make
the arbiter resource- and certificate-aware; make semantic choices explicit and
fail-closed; carry `numeric_policy` below Graph IR; and provide a handful of
generic structural ops instead of per-domain kernels.

That convergence is the point. A capability plan that builds its own private
version of one of these seams (its own scheduler datum, its own cost model, its
own sync convention) creates exactly the Decision #29/#31 debt the W0 sweep
spent a month deleting. The purpose of this view is that the *next* six papers
land on substrate instead.

### 0.1 Verification status of the inherited mathematics (pass of 2026-08-15)

The SparDA review's own lesson ("an earlier draft asserted 47; the tight test
caught it — which is the argument for executable contracts over prose") applies
to this view: three of its sources ship harnesses (game theory: 27 checks;
PDE: 78 assertions; SparDA: 4 machine-checked contracts) but the **CAKE
statistics and all five TileRT models were prose-only**. They are now
machine-checked — 13/13 in
[`research/core_substrate/verify_substrate_math.py`](../../../research/core_substrate/verify_substrate_math.py):
the exact permutation p-values (4/20 speedup, 1/20 evolve-time), the Fisher
plateau p, the Bonferroni/Holm family-wise floor, the Phase 4 filter-gate
algebra (`gain = 1/((c_s/c_g)+(1−p))`, net-positive iff `c_s/c_g < p`,
confirmed against direct simulation), the bubble decomposition `B ≥ 0` with its
dominance/rotation edge cases, the `|R|`-overlap ceiling and its asymptotic
tightness, the M2 roofline arithmetic, the M3 non-commutation counterexample,
Graham's `(2 − 1/m)` bound with its tight instance family, the MTP acceptance
closed form (`α = 2.77 ⇒ p ≈ 0.76`), and the game-theory memory/fp64 walls
(`n = 53` under the half-ulp model). **No claim required correction.**

Two precision notes the verification pass adds, both narrowing rather than
contradicting the sources:

* **Graham's bound is an identical-machines theorem.** The composition setting
  is heterogeneous resource *lanes* (compute/memory/network are not
  interchangeable), so "a plain queue is already near-optimal" is verified
  *motivation* for static-first list scheduling, not a guarantee about our
  scheduler. The binding design constraint remains M4's determinism rule, which
  is exact regardless.
* **The batch-1 overlap ceiling is 2×, and the sum-of-lanes bound is what the
  harness certifies.** `T_tile ≥ max_r W_r` is the lower bound the ratio is
  measured against; a real schedule also pays `CP(G)` and fill/drain, so
  measured single-box gains sit *below* the certified ceiling. This quantifies
  TILERT_ASSESSMENT §4.2's "do not oversell the near-term win" and is carried
  into P3/P5 expectations below.

---

## 1. The demand matrix

What each source actually asks of the core compiler, reduced to its load-bearing
demands (details and evidence live in the source docs):

| Source | Core-compiler demands |
|---|---|
| **CAKE** (`compiler_enhancement.md`) | Typed sync/memory Tile surface (§5.2 floor **landed**, W2.4a); verifier facts **derived across block-argument edges** (§5.3 — the measured fail-open gap); warp **roles + producer/consumer sets as IR** (Phase 2); a **stated** schedule entry point sharing one IR with the derived path (Phase 3, Decision #31); two-stage predict→prune→measure arbiter gated on a measured rejection fraction (Phase 4); leakage governance + frozen-source provenance (Phases 5–6); ceiling capabilities: schedule-level autodiff, causal bottleneck attribution, declared cross-target schedule loss (Decision #32) |
| **Game theory** (`GAME_THEORY_PLAN.md`) | Generic **`butterfly_transform`** op + `coalition` layout value + **one** shared butterfly tiling/sharding pass consolidating with the spectral FFT lane (G1b, the sanctioned #31 consolidation); linear ops with real `transpose_rule` consumers; **`batching_rule` made real** (G4 forcing function — today's vmap is a Python for-loop); implicit-diff seam hardening (H3 strict complementarity — shared with Riemannian-OT/S-series); `segment_sum` + `scan` as catalog ops; **fusion as a correctness feature driven by `numeric_policy`** (fp64 lattice intermediates must not round through fp32 — needs a carrier below Graph IR); certificate discipline (duality gaps, CI contracts on sampled estimators) |
| **TileSight** (`TILESIGHT_ASSESSMENT.md`) | Calibrated per-device performance parameters with provenance (`target_perf.py` **landed; the measured overlay is still empty — calibration sweeps never run**); reuse-distance pruning model (T1 v1 landed); **prune-don't-select** in front of the measured arbiter (T2); resource-vector + action-DAG cost module (T3, landed as `composition_cost.py` v2); prologue/steady/epilogue + resident-tiles overlap model (T4, open); block-rasterization knob (landed as `tile_rasterization.py`; ~~no emitter consumes it yet~~ — **corrected 2026-08-16: all four emitters now consume it** (`emit/nvidia_cuda.py`, `msl_gemm_emit.py`, `apple_gemm_schedules.py`, `rocm_schedule.py`), verified by executing the NVIDIA path. It remains **carried, not swept** — `row_major` is still the production choice everywhere and automatic enumeration awaits an architecture-owned correlation/retain verdict per ROCM-CALIB-1, so the lever is expressible and still unpulled. See [`CUTE_IR_ASSESSMENT.md`](CUTE_IR_ASSESSMENT.md) §3.2) |
| **TileRT** (`TILERT_ASSESSMENT.md`) | The composition layer: await-sinking (W5.2a **closed**), resource vectors in the measured corpus (W5.2b **closed**), scalable action-DAG scheduling (W5.2c/g **closed**), MoE chunk-overlap consumer (W5.2d **closed**), automatic dependence-edge synthesis (W5.2e landing); **a schedule datum that survives into IR** (E5 — `ScheduleStep` is still discarded at the IR boundary); M3: once composition exists, per-op scalar-latency argmin is *wrong*, so selection must eventually rank against the step's bottleneck resource; M4: static-first, dynamic only for MoE/MTP variance under a determinism rule |
| **SparDA** (`SPARDA_REVIEW.md`) | No new math ops — a **scheduling and memory-space problem**: cross-layer data-dependent **prefetch edge** in Schedule IR with a static overlap-feasibility legality check; host-DRAM as a first-class KV tier with explicit cache-state invariants (`KVCacheHandle` extension); **bitmask block-sparse iteration** as a Tile-IR representation choice the arbiter picks per shape; stats-emitting attention as **one primitive family with two result modes** (#31); `causal_convention` as a fail-closed semantic key (#21a); GQA-fold layout rewrite via the #15a `layout` attribute; index ops (`top-k`, `block_set_diff`) |
| **FORGE** (`FORGE_ASSESSMENT.md`) | Generalize "fuse a consumer into its producer's tiled epilogue" into declared substrate: a **read-locality lattice** on operands (`coordinate ⊏ … ⊏ global`, the A/U optimizer split — gives `TilingInterface` its first consumer, closing a live #29 gap); a **residency contract + static materialization proof** (`LOWER-COUNT-1` generalized into a pass — host-free memory claims, Decision #19's discipline applied to memory); the `matmul → optimizer` fusion instance with structural guards; **fail-closed semantic keys** `grad_clip_scope` (global clipping is *provably* unfusable — P2) and `routing` (the 2.24× MoE decay hazard — P5); a **precision-realizability oracle** (§1.3: whether the fp32-accumulator win is realizable is `numeric_policy.accum` × state dtype — the measured motivation for the #32 carrier, with a published error target); affine reduce-into-state for ZeRO-2 |
| **PDE/stencil** (`PDE_STENCIL_CAPABILITY_PLAN.md`) | **Semantic analysis as compiler passes** (classify → guards → legality → discretize → stability certificate), all fail-closed incl. attribute-absence; certificates emitted as **symbolic admissible regions** a constructive consumer reads (`-tpp-select-dt` maximizes `dt` inside the region — the model for certificate-driven autotuning); BC/scheme/derivative-convention as semantic keys; typed operator handles; the neighbors/TPP **stencil-stack unification** (#31, GAP-4); `numeric_policy` has **no carrier below Graph IR in either stencil tree** (#32 info-loss records mandated); `tridiagonal_solve`, Chebyshev/DST, jet AD |

---

## 2. The shared substrate — nine investments, many consumers

### S1. Derived facts across block-argument edges (the legality substrate)

**What:** extend the W2.1 derivation discipline down into Tile IR: def-use
reachability that crosses `scf.for` `iter_args` (loop back-edges), token
pairing (arrive→wait, copy→retire), and SSA-keyed rather than string-keyed
identity. This is CAKE §5.3, and it is *measured* as the live hole — the
canonical pipelined kernel shape passes every legality gate silently
(`compiler_enhancement.md` §5.1.1; probes in `research/tile_sync/`).

**Consumers:** CAKE Phase 1 exit gates 2–5 and Phase 2 (roles are useless
without derived verification); W2.4's `TileDataflowLegalityPass` (same work);
W5.2e's dependence-edge synthesis for composition (already consumes W2.1 at
Graph level — the Tile-level analog is this); SparDA's prefetch-edge legality;
PDE stability enforcement at lowering (fail-closed on `unknown`).

**Stands:** W2.4a floor landed (typed token slot, fail-closed shape rules);
the derivation half is open. **Owning rows: W2.4 / W2.4a.**

### S2. One schedule/action-DAG object that survives into IR, with two entry points

**What:** a single schedule representation (Decision #31) that (a) the compiler
*derives* on the `@jit` path, (b) an author can *state* (CAKE Phase 3's
builder), (c) **survives the IR boundary** instead of being re-derived from
scalars (TileRT E5: `pipeline_planner.ScheduleStep` is today discarded at
`to_mlir_attrs()`), and (d) is what the composition cost model (W5.2c) and the
overlap planner order. CAKE Phase 2's barrier producer/consumer role sets are
the *vocabulary* of this object's sync edges; SparDA's cross-layer prefetch is
one of its edges (an effect-carrying dependence edge that skips a layer body).

**Consumers:** CAKE Phases 2–3; TileRT composition (W5.2); SparDA Phase 4;
TileSight T3 (reads the action DAG from IR rather than asking a human);
schedule-level autodiff (CAKE capability #1 — transpose the stated schedule);
causal bottleneck attribution (needs a schedule decision to point at).

**Stands:** action-DAG cost + scheduling landed host-side (W5.2c/g); the
IR-surviving schedule datum and the stated entry point are open. **Owning
rows: W5.2 family; CAKE Phases 2–3 have no slot yet.**

### S3. Calibration + certificate/resource-aware arbitration

**What:** three connected moves. (1) **Run the calibration sweeps** —
`target_perf.py` has the mechanism and provenance discipline, but the measured
overlay is empty on every fleet box, so *every* analytical model above it
(reuse-distance T1, composition T3, PDE dt-selection, schedule planning)
currently runs on spec-sheet numbers or refuses. This is the cheapest unblock
in this entire view. (2) **Prune-don't-select** (TileSight T2): analytical
models in front of the `arbitrate()` seam, measured scalar latency stays the
only promotion authority (already the machine-checked contract in W5.2b/c —
`selector_authority = latency_ms`). CAKE Phase 4's gate applies: measure the
rejection fraction `p` before building more predictor. (3) **Certificates as
arbiter inputs**: the PDE plan's symbolic admissible region (`{s ≤ 1/2 ∧ a² ≤
2s}`) consumed constructively by `-tpp-select-dt`, and accuracy budget as a
search axis (CAKE capability #4) — the arbiter searches *within* what a
certificate proves safe instead of probing and being refused.

**Consumers:** every backend lane (candidate pruning during bring-up);
PDE `-tpp-select-dt`; game-theory `saddle_solve`/estimator certificates;
TileRT M3's eventual bottleneck-resource ranking (resource vectors already
recorded, W5.2b); quantization/low-precision (accuracy-budget search).

**Stands:** mechanisms landed (T1, T2 seam, W5.2b/c, rasterization knob);
**calibration sweeps unrun, rasterization knob unconsumed by any emitter** —
both flagged in TILESIGHT_ASSESSMENT §4 since 2026-07-30.

### S4. Semantic keys + emitted certificates as the safety architecture

**What:** the governance pattern every paper independently confirmed. Semantic
choices are declared keys that fail closed on absence (Decision #21a): SparDA's
`causal_convention` (two irreconciled causal rules shipping in one codebase is
the cautionary tale), PDE's `bc`/`scheme`/derivative-convention (GAP-3: a
defaulted `scheme = "central"` silently produced a divergent kernel),
game theory's `game_form`/`value_weighting`/`regret_kind`, CAKE's
`tile.retire_all` (landed — an indistinguishable bare form became a declared
semantics). The complement: analyses emit **certificates** (stability verdicts
with a `DEFECTIVE` lattice state, duality gaps, CI triples
`(estimate, stderr, n)`, dt brackets) rather than booleans, so downstream
consumers can act constructively and the evaluator can refuse un-certified
artifacts a correctness rung.

**Stands:** governance already exists (#21a/#29/#30 + drift gates); the
*certificate-emission* half is per-lane work carried inside each plan. No new
core mechanism needed — the discipline is the mechanism. This is also the
answer to SparDA §II.6: every defect found in that mature external codebase
fails open; ours must not. FORGE adds the sharpest key instances yet:
`grad_clip_scope` (where `global` is rejected because exact fused global
clipping is *provably impossible* — P2 — and every approximation measured as
bad as no clipping) and `routing` (the silent 2.24× MoE state-decay hazard).

### S5. `numeric_policy` carried below Graph IR (the Decision #32 carrier)

**What:** one carrier design for storage/accumulator/math-mode that survives
Schedule and Tile IR, plus the boundary verifier that fails on silent loss.
Three papers hit this independently: CAKE (Decision #32's original derivation —
the accumulator contract no longer exists when codegen picks an instruction),
game theory §6 (**fusion is a correctness feature**: the `2^n`-magnitude zeta
intermediate must never round through fp32, a fact the fusion planner must
learn from `numeric_policy`, not from a special case), and the PDE plan §III.4
(zero occurrences of `numeric_policy` in either stencil tree; interim
`tessera.info_loss` records mandated).

**Stands:** partially landed where W1.1 reached — `!tile.fragment<…, acc, …>`
carries the accumulator on the typed ROCm route. Generalizing beyond MMA
fragments (pointwise/reduction/butterfly chains) has **no owning row**.
**FORGE §1.3 supplies the measured target** this carrier was missing: whether
the fused-epilogue fp32-accumulator win is realizable flips 913× → 1.1× → 1.0×
purely as a function of `accum` × state dtype — a fact only a compiler carrying
the policy can decide and report (the W5 realizability oracle is S3's
certificate discipline applied to it).

### S6. The general structural-op tranche

**What:** a small set of generic ops that multiple domains reduce to, each with
the full Decision #24 registry treatment (catalog + coverage + transpose +
batching), instead of per-domain kernels:

| Op | Demanded by |
|---|---|
| `tessera.butterfly_transform` (kernel enum: zeta/Möbius/WHT/FFT-stage) + `coalition` layout value + **one** shared tiling/sharding pass | Game theory G1b **and** the existing spectral FFT lane (the #31 consolidation both justify) |
| `scan` | **Corrected 2026-08-15: not missing.** `cumsum`/`cummax`/`cumprod` (prefix scans) and `control.scan` (S5, general carried recurrence → `tessera.control_scan`) already exist; the tranche item is *consuming* them (PDE Bellman, game value iteration), not adding a duplicate (#31) |
| `segment_sum` | **Corrected 2026-08-15: not missing.** `segment_reduce(x, seg_ids, op="sum")` is in the catalog; CFR (G3) and MoE consume it. Remaining check: the ragged encoding carries reach-weight multiplies |
| `tridiagonal_solve` | Crank–Nicolson (PDE §III.1); "missing everywhere today" — still the genuine gap in this row group |
| Stats-emitting attention **result modes** (`(o,m,ℓ)` vs `(m,ℓ)`+replay) as attributes on one op | SparDA stage 1 + Block-AttnRes — two independent 2026 systems reduce to it (#31: one family, not two ops) |
| Index ops: `top_k` (index-producing), `block_set_diff` | SparDA selection/delta-fetch; DSA lane |
| Chebyshev/DST/DCT alignment to BC keys | PDE spectral path (rides the existing FFT lane) |

**Stands:** none owned. Game theory open question 3 already proposed a
"general-ops tranche" that its plan then consumes; SparDA and PDE land in the
same tranche. This is the clearest case in the view for **one new
integrated-plan row serving three capability plans at once**.

### S7. Memory tiers + data-dependent movement as scheduled, legal IR

**What:** SparDA's real demand — pinned host DRAM as a first-class KV tier,
a `prefetch(blocks, layer+1)` Schedule-IR edge whose dependence skips one layer
body, a **static overlap-feasibility check** per (batch, context, k, block)
bucket, and explicit cache-state-machine invariants for `KVCacheHandle`
(reserved tail slot, delta-fetch eviction safety — every invariant SparDA
leaves emergent, made assertable). Structurally this is the same
copy/wait/token machinery S1/S2 govern, at host↔device granularity with
data-dependent transfer sets.

**Consumers:** SparDA Phases 3–4; long-context inference generally; the W5.2
overlap layer (same action-DAG vocabulary); Phase G/H multi-device work
inherits the same edge semantics for cross-rank movement.

**Stands:** device-side sync floor landed (W2.4a); chunk machinery exists with
mocked transports (TileRT E4); the prefetch edge, feasibility check, and cache
invariants are **unowned** beyond SparDA's own phasing table.

### S8. The transform substrate (batching, transpose, implicit-diff, schedule AD)

**What:** the program-transform layer several plans quietly depend on:
**real `batching_rule`s** (game theory G4 calls Bayesian-game index expansion
"a good forcing function for making batching real" — the axis is currently
closed over a Python for-loop, a named #29 instance); **transpose rules with
genuine consumers** (butterfly adjoints, PDE operator adjoints under declared
BC — wrong at the boundary without S4's `bc` key); **implicit-diff hardening**
(H3 strict-complementarity checks at `custom_root` solutions — one fix serving
game theory, Riemannian-OT, and the S-series, per game-theory open question 4);
and, at the ceiling, **schedule-level autodiff** (CAKE capability #1: derive
the backward schedule from the forward one — transpose the S2 object, re-derive
S1's producer/consumer sets on the transposed graph; "every training kernel
needs a backward, and today every one is written twice").

**Stands:** VJP/transpose registry machinery exists and auto-flips coverage
axes; batching and implicit-diff hardening are open with no row; schedule-level
AD is gated on S1+S2 and is deliberately a *later* item — it is the payoff, not
the prerequisite.

### S9. Locality + residency: fusion legality as declared, provable metadata (from FORGE)

**What:** the two-piece substrate FORGE's whole workstream list reduces to.
(1) A **read-locality lattice** on operands
(`coordinate ⊏ row/column ⊏ block ⊏ tensor ⊏ layer ⊏ global`): fusion into a
tiled producer is legal iff the consumer's read-locality ⊑ the producer's tile
partition. This is the A/U optimizer decomposition generalized past
optimizers, it subsumes FORGE's four regimes as lattice positions, and its
legality query finally consumes `MatmulOp::getLoopIteratorTypes` +
`tessera.full_k` — closing `TilingInterface`'s Decision #29 gap. (2) A
**residency contract** (`tessera.residency ∈ {tile, layer, full}`) with a
boundary verifier that fails when a lowering materializes above it — the
in-tree `LOWER-COUNT-1` idiom generalized into a pass, making memory claims
**host-free provable** (no fleet machine can hold the 62 GB peak; the IR
proof makes that irrelevant).

**Consumers:** FORGE W3/W7/W8 (the epilogue-fusion family: optimizer, loss
VJP over the 128k-vocab logits, norm reductions, quant/dequant, EMA); the
S2 schedule object (residency is a schedule-visible property); W5.3's fusion
region discovery (the legality oracle it was waiting for); the S5 carrier
(the fused path must transport `numeric_policy` before #31 allows collapse).

**Stands:** the pieces exist as precedents (`TrainingStepFusionPass` with its
negative fixture + `LOWER-COUNT-1` check; `numeric_policy` consumers below
Graph IR on `MatmulOp`); the lattice, the residency pass, and the fusion
instance are **unowned** (FORGE's own build order W1→W2→W3→W4 is the sequence,
and it is deliberately host-free).

---

## 3. Ownership map

| Substrate | Owning row today | Unowned remainder |
|---|---|---|
| S1 derived Tile facts | **W2.4 / W2.4a** | — (scope is clear) |
| S2 schedule object | **W5.2** (c/e/g landed/landing) | IR-surviving schedule datum; stated entry point (CAKE Ph 3); roles vocabulary (CAKE Ph 2) |
| S3 calibration + arbiter | W5.2b/c landed; TileSight §4 items | **Calibration sweeps** (fleet-box task, small); rasterization-knob consumers; certificate-driven candidate rejection (PDE §III.2) |
| S4 keys + certificates | Governance (#21a/#29/#30, drift-gated) | Per-lane, carried inside each plan |
| S5 numeric_policy carrier | W1.1 (fragments only) | **Generalized carrier + boundary verifier** — no row |
| S6 structural-op tranche | — | **Entire tranche — no row** (G1b, scan, segment_sum, tridiagonal, attention modes, index ops) |
| S7 memory tiers + prefetch | W2.4a (sync floor); E4 chunk machinery | **Prefetch edge + feasibility check + KV-cache invariants** — no row beyond SparDA's own table |
| S8 transform substrate | AD-* rows (partial) | **Real batching; implicit-diff hardening** — no rows; schedule-AD deliberately deferred |
| S9 locality + residency (FORGE) | — (precedents: `TrainingStepFusionPass`, `LOWER-COUNT-1`, `numeric_policy` on `MatmulOp`) | **Entire pair — no rows** (FORGE W1–W4 in its own order); host-free |

Four flagged inputs for the integrated plan (recommendation only; ordering is
its call): a row for the **structural-op tranche** (S6 — serves three plans),
a row for the **generalized numeric_policy carrier** (S5 — three plans mandate
it via #32), and the **calibration sweeps** as an explicit small task rather
than a perpetual "still open" footnote (S3 — it silently gates four other
items). Fourth: **rows for FORGE W1→W2→W3→W4** (S9 — host-free, and W2's
residency proof is the cheapest honest answer to every future memory claim).

## 4. The updated build sequence (proposal — ordering authority stays with the integrated plan)

Phased so that every phase has a falsifiable gate, phases P1/P2 run in
parallel on different resources, and nothing later builds on an unverified
assumption. Justifications cite the machine-checked results (§0.1).

**P0 — landed (2026-08-15).** The W2.4a typed-sync floor (CAKE §5.2 rows 1–5,
fail-closed verifiers, `tile.retire_all`, negative fixtures) and this view's
verification harness (13/13). The measured facts P1 relies on — the block-arg
fail-open, the 3-fixture break count — are recorded in
`compiler_enhancement.md` §5.1.1/§5.2.1 and `research/tile_sync/`.

> **P1a landed same day, in two increments:** `TileDataflowLegalityPass` + the
> shared `TileValueProvenance` loop-carry resolver (every `research/tile_sync/`
> silent row fires), then typed predicates, hatch deletion, the §5.4 fixture
> port, and NVIDIA-pipeline wiring (`compiler_enhancement.md` §5.3.1–§5.3.2).
> The gfx1151 numerical gate closed for the executing lane: 1,569 compiled
> device tests green on the changed tree (§5.5 gate 5 note). Remaining:
> barrier-at-birth restructure; the mbarrier/TMA family stays lowering-verified
> until Phase G/H silicon.
>
> **P1b's gfx1151 sweep is DONE (same day):** `dram_bw_gbps = 186.8`,
> `fp16:matrix = 47.3`, `bf16:matrix = 50.2` TFLOP/s
> (`benchmarks/calibration/calibrate_gfx1151.py`, wall-clock method; corpus in
> `benchmarks/baselines/`); `SchedulePlanner.for_target("rocm_gfx1151")` no
> longer raises after `load_corpus()`. Remaining for P1b: sm_120 + Apple
> sweeps (their boxes), Zen5 peaks, corpus auto-load wiring, and the
> rasterization knob's first consuming emitter.

**P1 — the two force multipliers (start now, in parallel).**

* **P1a: S1, the §5.3 derivation** (owner W2.4/W2.4a; primary box). Derive
  arrive→wait pairing (including slot identity), pipeline-ring reachability,
  and SSA-keyed arrival counts across `scf.for` block-argument edges;
  unresolvable facts fail closed. *Gate:* the `research/tile_sync/` rows still
  marked "silent" flip to rejections; the CAKE Phase 1 exit gates 2–5 close
  (registered-path legality, lowering, gfx1151 numerics).
* **P1b: S3's calibration sweeps** (each fleet box; hours, not weeks). Populate
  `target_perf.py`'s measured overlay on gfx1151, sm_120, and Apple —
  bf16/fp8 matrix peaks, DRAM bandwidth, and the cache parameters T1 reads.
  *Gate:* `SchedulePlanner.for_target(..., "bf16")` stops raising on all three
  boxes; T1/T3 run on measured numbers; the rasterization knob gets its first
  consuming emitter and a measured L2 delta (TileSight's 35%→72% claim is
  exactly what a sweep confirms or kills). *Justification:* the Phase 4 filter
  gate is now verified algebra — `c_s/c_g < p` decides whether more predictor
  ever pays — but `p` and `c_s/c_g` are only measurable with calibrated
  baselines, and the §5.2.1 break count (3/322) is the first, deliberately
  weak, datum.

**P2 — S6, the structural-op tranche (parallel track, host-free, any box).**
First slice per the game-theory plan's own §10: `subset_zeta`/`subset_mobius` +
`semivalue` through `custom_primitive(linear=True)` with oracles 1–6 — it is
self-contained and gives `transpose_rule` its first genuine consumer.

> **First slice landed (2026-08-15):** `python/tessera/game/` — the four
> butterflies + `coalition_marginal` + weight-parameterized `semivalue`
> (Shapley/Banzhaf as weight vectors), linear primitives with declared
> transposes, fp64 by the §6 mandate, fail-closed lattice contracts; six
> catalog rows on the spectral/contraction lanes, dashboards regenerated
> through the drift gate; oracles 1–6 + adjoint FD checks green (17 tests,
> incl. the §1.1 factor-of-two pin).
>
> **Second slice landed (same day) — G1 is COMPLETE:** `boltzmann_value`
> (closed-form VJP, FD-checked; H5's ±T limits pinned on an asymmetric
> fixture), `coalition_excess` (jointly linear; the zeta-of-an-additive-game
> identity as its oracle), and segmented `mex` (Sprague–Grundy nim-XOR
> oracle; non-differentiable by construction). And the consume-and-correct
> finding: `scan`/`segment_sum` were **never missing** —
> `segment_reduce`/`cum*`/`control.scan` already exist, so those tranche rows
> reduce to consumption checks (#31), corrected above. Remaining in P2:
> `tridiagonal_solve`, stats-attention result modes, then G1b after P4. Then
`scan` + `segment_sum` (three plans want them; catalog + coverage rows per
Decision #24), `tridiagonal_solve`, and the stats-attention **result-mode
attribute** (one op, two modes — the #31 form both SparDA and Block-AttnRes
reduce to). G1b (the shared butterfly pass) deliberately waits for P4's
carrier and for the tranche's Python reference to exist first (the #31
ordering caveat: don't collapse a duplication before the survivor can carry
what the deleted path carried). *Gate:* registry rows green through the
12-axis registry; `subset_mobius ∘ subset_zeta = id` through the shared
lowering once G1b lands.

**P3 — S2, the schedule object, designed once.** CAKE Phase 2 (roles +
producer/consumer sets as SSA), TileRT E5 (a schedule datum that survives
`to_mlir_attrs()`), and the W5.2 action DAG are **one object** — specify it in
a single design doc before any of the three lands separately.

> **Design doc landed (2026-08-15):**
> [`SCHEDULE_OBJECT_DESIGN.md`](SCHEDULE_OBJECT_DESIGN.md) — the
> actions/edges/roles/residency object with a content-addressed digest, two
> entry points (derived + stated) over one IR carrier, six gates (incl. the
> one-rule CDNA/Hopper role test and the M4 determinism rule), named
> consumers, and the SO-1..SO-5 build order (host-free except SO-2's gfx1151
> fixture). FORGE W2's residency lives on this object by design. *Gates:* CAKE
Phase 2 exit gate 2 (the CDNA ping-pong schedule and the Hopper
producer/consumer split verify through the same rule, no target branch — a
negative verdict re-scopes Phase 3 before anything depends on it); a
`ScheduleStep`-shaped datum reaches Tile IR without scalar re-derivation;
W5.2e's inferred edges populate the same object. *Expectation-setting from the
verified math:* single-box overlap reclaim is bounded by the 2×/3× ceilings
and batch-1 collapses toward 2× — the payoff of P3 is having the layer
correct and measured before Phase G/H silicon, plus unlocking S8's ceiling
items; it is not a near-term throughput story.

**P4 — S5, the `numeric_policy` carrier.** One carrier design (attribute or
type parameter) below Graph IR + the Decision #32 boundary verifier, replacing
the PDE plan's interim `tessera.info_loss` records and generalizing W1.1's
fragment-only `acc`. Must land before G1b's fusion-as-correctness consumer
needs it (the verified fp64 wall: nonnegative-game zeta storage dies at
`n ≈ 16` in fp32 and the fused path is the correctness fix, which the fusion
planner can only learn from a carried policy). *Gate:* a lowering that drops
`numeric_policy` without a named reason fails the boundary verifier; G1b and
one stencil lowering consume the carrier.

**P5 — S7 + S8 ceiling items, after P1/P3.** SparDA's prefetch edge + static
overlap-feasibility check (an S2 edge species + an S1 legality query), the
`KVCacheHandle` invariant work, real `batching_rule`s (G4 as forcing
function), implicit-diff strict-complementarity (one fix, three consumers),
and — last, because it composes S1+S2 — schedule-level autodiff and causal
bottleneck attribution, the two capabilities nobody else ships.

**Standing recommendation to the integrated plan** (unchanged from §3): mint
rows for the S6 tranche, the S5 carrier, and P1b's sweeps; P1a already has
W2.4a; P3 belongs beside W5.2; P5 items bind to their existing owners
(SparDA's table, AD-* rows) once P1/P3 exist.

## 5. Honest limits

- This view synthesizes documents that were themselves point-in-time reviews;
  where a source doc and a generated dashboard disagree, the dashboard wins
  (Decision #26).
- The demand matrix compresses; each source doc's own take/adapt/skip verdicts
  remain authoritative for their domain (e.g. SparDA's §V non-extractions,
  TileSight's §3.3 skip list).
- §0.1 verifies the *mathematics this view inherits and reasons from*, not the
  external papers' empirical claims: TileSight's accuracy numbers (12.35%
  MAPE, ~1pp L2) remain their unreproduced claims — no artifact exists — and
  TileRT's absolute throughput figures are cross-checked for internal
  consistency (M2), not reproduced. CAKE's per-run data are fixed by the
  reported medians/ranges only because n=3; if per-run data are released, the
  §0.1 permutation results should be recomputed.
- The verified models are models: M1's ceiling certifies a bound against
  `max_r W_r`, while real schedules also pay `CP(G)` and fill/drain; Graham's
  bound assumes identical machines (§0.1). Neither limitation weakens the
  *planning* conclusions drawn here (static-first, prune-don't-select,
  record resource vectors), all of which survive under the weaker readings.
- "No owning row" claims were checked against `INTEGRATED_COMPILER_PLAN.md`
  at `98282bd7` (2026-08-15) and go stale the day rows are added — which is
  the intended outcome.
