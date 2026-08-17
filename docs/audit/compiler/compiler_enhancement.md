---
last_updated: 2026-08-15
audit_role: plan
plan_state: open
---

# Compiler enhancement — what CAKE says about our Tile IR, and the two phases it scopes

> **Source:** Ye et al., *CAKE: Compiler–Agent Co-Design for Frontier Kernel
> Evolution*, arXiv:2608.12629v1 (NVIDIA / CMU). Companion artifact:
> [FlashInfer PR #4262](https://github.com/flashinfer-ai/flashinfer/pull/4262)
> (frozen CAKE-generated SM100a KDA prefill kernels). Assessed 2026-08-15
> against the tree at `f96695f`.
>
> **This is a scoped `plan`, not a status surface.** Status truth stays in
> [`MASTER_AUDIT.md`](../MASTER_AUDIT.md) + `generated/` (Decision #26).
>
> **Global sequencing authority is
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md), not this
> document.** §§5–6 scope *what* Phase 1 and Phase 2 are and *what gates them*;
> they deliberately do not assert a position in the global order or a fleet
> allocation against other workstreams. Both phases need an owning work-item ID
> and a slot in the integrated plan's table before implementation starts, and
> §5's own dependency on the in-flight W1.1 chain (§3.5) is a sequencing
> constraint that only the integrated plan can resolve. The compiler map is
> [`README.md`](README.md).
>
> **Reads against:** [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md) (the
> in-flight typing workstream — §3.5 states precisely what here is *not* new),
> [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) §4
> (the arbiter), [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) (the
> cost model this doc declines to promote to a selector).

---

## 0. Why this doc exists

CAKE is the first controlled A/B on **IR design as agent ergonomics**: same
model, same scaffold, same oracle, same token budget, same GPU, same task — the
only variable is the representation the agent edits. That is a question Tessera
has been answering by assertion since Decision #28, so the paper is worth taking
seriously.

Two things came out of reading it, and the second is the durable one:

1. **The paper's headline number does not survive its own statistics** (§2). The
   supported claim is *convergence*, not *speedup*, and that changes what we
   should build and how we should justify it.
2. **Reading it forced a trace of our Tile dialect's sync/memory surface**, and
   that surface is untyped in exactly the places a schedule verifier would need
   (§3.2), verified by a pass that ignores the type system by construction
   (§3.3), and unreachable by any author (§3.4).

§2 and §3 are the content. §5–§6 scope the two phases those findings open; §7
sketches what those phases must *inform* before the later ones are committed.

---

## 1. What CAKE is

Agents author **CAKE IR** — a typed, hardware-explicit *schedule* representation
— instead of CUDA/PTX. The compiler lowers it deterministically to inspectable
CUDA. Three commitments:

| Commitment | Mechanism |
|---|---|
| Agents edit a typed IR, not raw CUDA | Fixed op vocabulary; declared resources (SMEM/TMEM regions, barriers, pipelines, warp roles); addresses, phase bits, TMEM offsets, and warp identity are **derived by lowering, not authored** |
| The environment returns localized diagnostics, not a pass/fail bit | Pre-compile gates for program safety, hardware conformance, data consistency, schedule semantics; each finding names the affected resource/role/stage. Reports and hints are non-blocking |
| The harness is itself evolved | Recurring failures become verifier rules, IR primitives, or cost-model calibrations, gated by corpus tests |

The schedule fragment they show is the whole idea in ten lines:

```python
pool     = lm.smem(98304)
smem_q   = pool.view(offset=0, shape=(128,128), dtype=lm.bf16, stage=3)
tmem_acc = lm.tmem(cols=0, width=128, shape=(128,128), dtype=lm.f32)
load = lm.role(warps=[0]);  mma = lm.role(warps=[1])
pipe = lm.pipeline(stages=3)
q_full = lm.barrier(count=3, prod=[load], cons=[mma], init_count=1, pipeline=pipe)
```

Note `prod=[load], cons=[mma]` **on the barrier declaration**. That single edge
is what turns a synchronization verifier from name-matching into graph
reachability. It is the smallest load-bearing idea in the paper.

Deliberate non-goal: **layout is not a first-class abstraction.** The agent
writes concrete commitments (an SMEM view offset, a TMEM column range, a swizzle
tag) and the compiler decides legality. They can afford this because they are
single-vendor, Ampere→Blackwell. We cannot (§4).

Scope they claim and we should not overstate: NVIDIA only; cost model calibrated
for B200 and H100 and declining to predict elsewhere; static analysis explicitly
incomplete in both directions ("both false positives and false negatives can
occur"); compiler evolution still human-gated at merge; no autodiff, no
distribution, no training.

---

## 2. Mathematical verification

Because n=3 and the paper reports median [min, max], **the reported triple is
the complete sample** for each arm. Every test below is therefore exact, not
asymptotic. Reproduced with an exact permutation enumeration over all
`C(6,3) = 20` assignments.

### 2.1 The headline result is not statistically significant

| Metric | CAKE IR | Direct CUDA/PTX | Exact one-sided *p* |
|---|---|---|---|
| Best speedup @ 80M tokens | {1.041, 1.144, **1.205**} | {0.852, 0.928, **1.151**} | **0.200** |
| Active evolve time (h) | {1.02, 1.89, 2.33} | {3.59, 3.73, 4.34} | **0.050** |
| Plateau by 80M | 3/3 | 0/3 | **0.050** (Fisher) |

The performance distributions **overlap**: the control arm's best run (1.151)
beats the treatment arm's median (1.144) and two of its three runs. Pooled
ordering is `0.852, 0.928, 1.041, 1.144, 1.151, 1.205`, giving the treatment arm
rank-sum 13 of a possible 15, so `U = 13 − 6 = 7` and `P(U ≥ 7) = 4/20 = 0.20`.

**So "1.144× versus 0.928×" is not a supported claim at this n.** It is a point
estimate whose arms are not separated.

The other two metrics *are* separated. Evolve time shows complete separation
(every treatment run faster than every control run), and the plateau count is a
clean 3/3 vs 0/3.

### 2.2 The design cannot produce a family-wise significant result

The minimum achievable one-sided *p* at `n = m = 3` is `1/C(6,3) = 0.05`. Three
metrics are reported, so the Bonferroni family-wise floor is **0.15**. No effect
size, however large, could have made this experiment family-wise significant.

This is not a criticism of the authors — the *controls* (fixed model, scaffold,
oracle, tolerance, hardware, budget) are unusually disciplined for this
literature, and running 6 × 80M-token clean starts is expensive. It is a
statement about what we may conclude: **treat it as a well-controlled pilot, not
a demonstrated 23% win.**

### 2.3 What is actually supported — and it is less than "convergence is proven"

An earlier draft of this section asserted that CAKE IR "converges faster and
more reliably" as *the statistically supported conclusion*. **That overclaims,
and §2.2 is the reason.** The evolve-time and plateau results are each `p = 0.05`
**unadjusted**; against the family-wise floor of 0.15 established one section
earlier, neither reaches the conventional 0.05 level after correction. Holm is no
kinder: the sorted p-values `(0.05, 0.05, 0.20)` are compared first against
`α/3 = 0.0167`, which the smallest already fails, so the procedure stops at the
first step and rejects nothing.

Two further facts prevent rescuing a single metric:

* **No primary endpoint is prespecified.** The paper does not designate one of
  the three metrics as primary, so selecting the two that separate is post-hoc.
  A prespecified primary endpoint would have made an unadjusted `p = 0.05` a
  legitimate single test; without one, it cannot.
* **Evolve time and plateau are not independent** — a run that plateaus early
  necessarily consumes less evolve time — so Bonferroni is *conservative* for
  that pair. Conservative does not rescue them: the unadjusted values are at the
  floor, so no less-conservative correction brings either below 0.05 either.

What survives is **descriptive, not inferential**, and it is worth stating
plainly because it is still the useful part:

> Across three matched runs per arm, the two representations are **not
> separated on attained speedup** (the control's best run beats the treatment's
> median), but are **completely separated on evolve time** (every treatment run
> faster than every control run) and split 3/3 versus 0/3 on the plateau
> criterion. Treat this as unadjusted exploratory evidence of a *convergence*
> difference, not as a demonstrated effect.

That ordering of evidence — convergence separates, ceiling does not — is a claim
about the *search*, not the *ceiling*, and it is what §2.4 builds on. §2.4's
planning consequence does not depend on any of these three tests clearing a
significance threshold; if anything the correct reading strengthens it, because
the paper establishes **no** effect at a family-wise 0.05 level and the only
defensible read of the whole experiment is directional.

### 2.4 Budget transfer — the constraint that reshapes the plan

The paper states the CAKE IR mean crosses the tuned FlashML Triton baseline **at
55M tokens**. Below that budget, the expected output of the CAKE arm is *worse
than an existing tuned Triton kernel*.

A realistic per-kernel budget for this project is 2–10M tokens — an order of
magnitude inside the regime where the paper's own curve sits under baseline.

**Consequence, and it is the single most important planning fact in this
document: do not justify an authoring surface on the performance number.**
Justify it on time-to-plateau and convergence rate — not because those are
*proven* (§2.3: they are not, after correction) but because they are the only
axis on which the experiment shows any separation at all, **and** because they
are the properties that matter at small budgets, where the question is whether a
search *terminates* rather than how high it climbs. A capability that makes the
search converge is worth building at 2M tokens; a capability that adds 20% at
80M tokens is not.

Stated as the gate it becomes: **Phase 3's success criterion is a reduction in
repair rounds to a correct verified kernel, measured on our own workload — not a
reproduction of anyone's speedup number.** That criterion is measurable inside
our budget; the speedup claim is not, and per §2.3 it was never established
anyway.

Two secondary notes on their methodology, both minor:

* The token-budget curves plot *best validated speedup so far*, a running
  maximum. A running max is monotone in sample count, so curve shape conflates
  "improves" with "samples more." Comparing arms at equal token budget is still
  fair; reading the slope as a learning rate is not.
* Token budget is equalized across arms but wall-clock is not (1.89h vs 3.73h
  median). This favors the treatment arm on GPU-hours per token, which the paper
  does not claim but which is real.

### 2.5 The non-clean-start numbers carry no variance at all

The KDA 2.05× geometric mean, TinyGEMM 18–23%, Alpha-MoE 6.204×/4.025×
API-level, and the eleven known-kernel entries are **single-run point
comparisons** with no replication reported. They are reasonable engineering
evidence and they shipped as upstream PRs, which is a stronger signal than the
numbers. Do not cite them as measured effect sizes.

One of them is worth reading carefully as a *methodological* warning: Alpha-MoE
reports 6.204× at API level and **1.215×** on a GPU-span remeasurement. The gap
is launch/scheduling fusion (five GPU activities → one megakernel), which is
real, but a 5× discrepancy between two honest measurements of the same change is
exactly why our benchmark schema (Decision #12) pins the measurement basis.

---

## 3. In-tree trace (measured 2026-08-15 against `f96695f`)

### 3.1 Where we already stand

Reading CAKE alongside the tree, the resource model is close to a match, and on
the evaluation half we are ahead:

| CAKE component | Tessera equivalent | Read |
|---|---|---|
| Declared resource model (SMEM/TMEM/barrier/pipeline) | `!tile.mbarrier`, `!tile.mbarrier_token`, `!tile.tmem`, `!tile.pipeline_state`, `!tile.tma_descriptor`, `!tile.buffer` in `TileOps.td` | **Present as types.** §3.2 is about whether the *ops* use them |
| Localized pre-compile diagnostics | `WARPSPEC_INIT_UNDER_GUARD`, `TILE_BARRIER_REUSE_MISSING_BARRIER` with `previous write` notes | Present, and genuinely in CAKE's "program safety, pre-compile gate" register |
| Numerical validation gate | F4 oracle (`fusion_core.verify_synthesized_*`) gating every arbiter candidate | **Ahead** — ours is per candidate, not per backend |
| Calibrated cost model | `target_perf.py` with per-**field** `MEASURED`/`DERIVED`/`SPEC` provenance and a `require()` that raises rather than fabricating | **Ahead** — CAKE calibrates B200+H100 and declines elsewhere; we decline *per field*, which is finer |
| Evidence ladder | `evaluator.Rung` (`ARTIFACT_ONLY` → `HARDWARE_VERIFIED`) with anti-silent-fallback | **Ahead** — CAKE has no equivalent to the fallback-detection invariant |
| Layout | `#tile.layout<shard = […] on ["tlane"] …>` | **We have the algebra CAKE refused.** See §4 |
| Portfolio / dispatch stage | `tuned_dispatch.py`, arbiter `(op, shape-bucket, dtype, target)` | Present in outline; the leakage discipline is not (§7) |

So this is not a "we should build CAKE" document. It is a document about three
specific holes.

### 3.2 F1 — the typing hole is on the sync/memory surface

Counted in `src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td`. **Explicit
inclusion rule**, so the scan is reproducible: comment text is stripped, then
every `def <Name> : <Base><…>` declaration (including the multi-line
`def X\n    : Tile_Op<…>` spelling) is collected, and a declaration counts as an
op iff `<Base>` is one of the six op bases this file declares —
`Tile_Op`, `Tile_CollectiveOp`, `Tile_CliffordBinaryOp`, `Tile_CliffordUnaryOp`,
`Tile_LinalgOp`, `Tile_ControlOp` — each of which resolves to `Tile_Op`.

> **Corrected 2026-08-15 (PR review).** An earlier draft reported 74 ops and "57
> direct `Tile_Op`". Both were wrong: the scan enumerated only four of the six op
> bases, dropping `Tile_LinalgOp` (6) and `Tile_ControlOp` (4), and the
> direct-`Tile_Op` count used a single-line regex that missed the multi-line
> `def X\n    : Tile_Op<…>` form. The corrected figures are below; **the ratio
> gets worse, not better.**

* **82** op definitions — 61 `Tile_Op`, 6 `Tile_LinalgOp`, 4 `Tile_CollectiveOp`,
  4 `Tile_CliffordBinaryOp`, 4 `Tile_ControlOp`, 3 `Tile_CliffordUnaryOp`.
* **63** declare their own `let arguments` list; the other **19** inherit from a
  base or take none. The 63 are the correct denominator — an op with no argument
  list of its own cannot be under- or over-constrained by one.
* **55 of 63 (87%)** have `AnyType` somewhere in their `arguments` list.
* **8 of 63** reference a declared `Tile_*` type at all.
* **5** mix both — and those five are precisely the hardware-explicit ops a
  schedule verifier would need to reason about.

Two notes the corrected scan makes possible. The four `Tile_ControlOp` ops
(`control_for`, `control_if`, `control_while`, `control_scan`) are **not** among
the 55: they declare no argument list of their own. Had they done so, `AnyType`
on their `iter_args` would have been *explained and correct* — loop-carried
values are polymorphic exactly as `scf.for`'s are — which is the distinction
W1.1 step 6 drew for `tessera_apple.gpu.control_{if,loop,while}`. And of the 55,
53 are plain `Tile_Op`: the hole is in the core vocabulary, not in a peripheral
family.

The specific defects, all read from ODS:

| Op | Defect | Why it matters |
|---|---|---|
| `tile.mbarrier.wait` | `Optional<Tile_MBarrierType> $barrier`, `Variadic<AnyType> $dependencies` | **A wait with no barrier at all verifies.** And it cannot consume the `!tile.mbarrier_token` that `tile.mbarrier.arrive_expect_tx` produces — so *the arrive→wait edge, the central edge of any warp-specialized kernel, is not expressible in the type system* |
| `tile.mbarrier.try_wait` | `Tile_MBarrierType $barrier, Tile_MBarrierTokenType $token` | **Not a defect — the target form.** The correct shape for `wait` already exists in the same file, four lines away |
| `tile.tma.copy_async` | `Optional<Tile_MBarrierType> $barrier`, `Variadic<AnyType> $dependencies`, `Variadic<AnyType> $outputs` | A TMA copy that gates on nothing verifies; and nothing in the type system connects the copy to the buffer it filled |
| `tile.tma.descriptor` | `AnyType $source` | No constraint that a descriptor's source is a buffer or memref |
| `tile.tmem.load` / `.store` | `AnyType $value`, `Variadic<AnyType> $indices` | Indices are unconstrained; the loaded value has no relationship to the TMEM region's contents |
| `tile.tcgen05.mma` | `AnyType $lhs`, `AnyType $rhs` | The one op in the dialect that *most* needs W1.1's `!tile.fragment<m,n,k,elem,acc,role,layout,family>` does not use it |
| `tile.pipeline_advance` | `Variadic<AnyType> $inputs` → `!tile.pipeline_state` | **A `pipeline_advance` need not consume a `pipeline_state`.** The ring's def-use chain is unenforced |
| `tile.alloc` | `AnyAttr:$layout` | `#tile.layout` exists as a real attribute; accepting `AnyAttr` where a typed attribute states the legal set is the Decision #21a anti-pattern |

The `pipeline_advance` entry deserves emphasis because the contradiction is
*inside the dialect's own documentation*. `Tile_PipelineStateType`'s summary
says the value "establishes ordering and state ownership through ordinary SSA
def-use" — and then the only op that advances it accepts anything. That is
Decision #29 (a declaration with no consumer) stated and violated in the same
file.

### 3.3 F2 — the synchronization verifier is type-blind by construction

My first reading of this was wrong and the correction matters. I initially
diagnosed "the verifiers run on a second, untyped vocabulary." They do not.
`src/transforms/lib/WarpSpecLegalityPass.cpp` matches like this:

```cpp
static bool isBarrierInit(Operation *op) {
  if (op->hasAttr("tile.barrier_init")) return true;
  StringRef n = op->getName().getStringRef();
  return n.contains("mbarrier") && n.contains("init");
}
```

Substring matching fires on *both* spellings, so the pass is vocabulary-agnostic
by design. The real finding is worse and more actionable: **the pass never reads
an operand type.** It reads names and attributes. Six predicates in
`WarpSpecLegalityPass.cpp:76–119` follow this shape.

Three consequences:

1. **Tightening the ODS types buys the verifier nothing on its own.** F1 and F2
   are not independent work items; a phase that fixes one without the other is
   half a change. This is why §5 is a single phase.
2. It is a Decision #30 violation ("derive, don't ask") one level below the
   `EffectLattice` instance CLAUDE.md already indicts — and it fails the same
   way, *open*: an op the substring misses contributes nothing, so an
   unrecognized barrier idiom passes the gate silently.
3. `WarpSpecLegalityPass.cpp:280` chases a defining op
   (`def->getName().getStringRef()`). That is structurally the same
   producer-chasing W1.1 §2 proved cannot cross a block-argument edge — which
   generates the hardest-case-first experiment in §5.1.

Coverage gap on top of this: the two legality fixtures
(`tests/tessera-ir/phase2/tile_warpspec_legality.mlir`,
`tile_barrier_reuse_legality.mlir`) run under `--allow-unregistered-dialect`,
while the fixtures that exercise the *registered* vocabulary
(`phase3/flash_attn_full.mlir`, `phase3/nvtma_barrier_emission.mlir`,
`phase2/tile_async_hardware_ops.mlir`, `phase3/warpspec_async_token.mlir`) do
not run the legality passes. **The registered path and the verified path are
disjoint sets.**

### 3.4 F3 — schedules are derived, never stated

There is no path by which a program author states a schedule.

* `tessera.kernel` resolves to `python/tessera/distributed/launch.py:53` and is a
  *shard* decorator for `index_launch`. It is not a tile-schedule surface.
* `compiler/tile_ir.py` is `_lower_schedule_ops` — a lowering, not a builder.
* Barriers and TMEM are **synthesized per-arch by lowering**:
  `compiler/target_ir.py:1618,1636` emit `tessera_nvidia.mbarrier` and
  `target_ir.py:1535` emits `tessera_nvidia.tmem_alloc`, derived from a
  `tile.matmul`. The barrier *count* is a consequence of an op choice, not a
  declaration.

This is a defensible architecture — deriving a schedule is strictly more
automatic than stating one — and it is the right default for the `@jit` path.
The CAKE result says something narrower: **the stated form is the one a search
process can operate on.** You cannot mutate a schedule decision that only exists
as a side effect of matmul lowering.

Both forms should exist. §7 keeps them one IR with two entry points, not two
IRs (Decision #31).

### 3.5 What here is *not* new — relationship to W1.1

[`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md) is a 5-week item on the
critical chain `W0 → W1.1 → W2.1 → W3.1`, with steps 1, 2, 4 landed and 3
partially landed as of 2026-08-04. **Nothing in §5 re-proposes it.** The
division:

| Concern | Owner |
|---|---|
| `!tile.fragment` parameterization; `tile.mma` / `MMAOp::verify()`; the K-loop accumulator; producer migration; Target IR dialect tightening (`tessera_apple`, `tessera_nvidia`) | **W1.1** — in flight, do not duplicate |
| `tile.mbarrier.*`, `tile.tma.*`, `tile.tmem.*`, `tile.pipeline_*`, `tile.tcgen05.mma` operand typing | **This doc, Phase 1** — untouched by W1.1's step list |
| Rewriting `WarpSpecLegalityPass` / barrier-reuse to derive from types | **This doc, Phase 1** |
| Warp roles and producer/consumer sets as first-class IR | **This doc, Phase 2** |

W1.1 also supplies three method lessons that Phase 1 should inherit rather than
rediscover, each of which cost real time there:

* **§2 — run the hardest case first.** Starting with the easy migrations
  produced four working ports and a stall on the fifth, with the type system
  already half-changed.
* **§4.1 — verifying is not lowering.** A passing verifier fixture is not a gate.
* **§4.2 — accepting is not lowering correctly.** Both backends materialized a
  zero constant for the MMA's C operand and never read the accumulator; a
  lowering fixture would have passed while the GEMM was silently wrong. **Only
  numerical execution catches that class**, which on this fleet means gfx1151.
* **§4.6 — the migration's real shape is a dialect conversion**, because region
  signatures need type conversion.

`tile.tcgen05.mma` is the seam between the two workstreams: it should consume
W1.1's `!tile.fragment<…, family = "tcgen05">`. That is *reuse* of a landed
design, not new design, and it should be sequenced after W1.1 step 5 so it does
not fork the fragment contract.

---

## 4. Verdict — take, adapt, skip

### Take

* **Producer/consumer role sets on the barrier declaration** (§1). The smallest
  load-bearing idea in the paper, and the enabler for a derived sync verifier.
* **The generalization stage as a separate stage with its own objective**, and
  its leakage discipline: declare the valid shape domain *before* tuning;
  dispatcher predicates may partition that domain but may not introduce
  evaluation rows; coverage grows only through deterministic unseen shards of
  the same source. This is free, drift-gateable governance and it guards the
  failure mode most kernel-agent papers do not.
* **Frozen generated source with provenance** (FlashInfer PR #4262): generated
  kernels checked in verbatim, provenance header, SHA256 integrity check,
  integration patches explicitly delimited. This is how agent-generated code
  ships without a dependency on the generator — directly applicable to our
  Tier-3 candidates under Decision #28.
* **"A finding names the affected resource, role, or stage."** Our diagnostics
  are stable (Decision #21) but coded at op granularity; CAKE's are anchored to a
  schedule decision. Cheap to adopt once Phase 2 gives roles an identity.

### Adapt

* **The pre-compile analysis gate.** Take the *contract shape* (blocking gate /
  report / hint, with explicit "coverage missing" as a distinct outcome from
  "passed"). Do not take it as a *ranking* mechanism until §7's `p` is measured.
* **The layout position, half of it.** Take "the agent writes concrete
  commitments and the compiler owns legality." Do not take "delete the layout
  algebra" — see Skip.

### Skip

* **Layout as a non-abstraction.** CAKE affords this because it is single-vendor
  across one descriptor and swizzle model, and the paper concedes non-NVIDIA
  transfer is unmeasured with the backend lowering to be rebuilt. Our four-target
  spread (simdgroup_matrix / WMMA / MFMA / AVX-512) is the harder problem, and
  Decision #32 requires the opposite: information loss across a boundary must be
  *declared*, which presumes the information exists in a comparable form.
* **The kernel-agent framing wholesale.** CAKE has no autodiff, no distribution,
  no training. It should not redirect the S-series.
* **Their cost model as a selector.** See the two-stage correction in §7.

---

## 5. Phase 1 — one typed, verified synchronization surface

**One phase, not two,** per §3.3: tightening types without rewriting the
verifier leaves the verifier reading names, and rewriting the verifier without
tightening types leaves it nothing to read.

**Owner boundary:** Tile dialect sync/memory ops only. Fragment and `tile.mma`
belong to W1.1 (§3.5).

**Fleet routing:** primary box (Strix Halo / Ubuntu). The exit gate requires
numerical execution, and gfx1151 is the fleet's executing lane.

### 5.1 Step 0 — the hardest case, before any ODS change

W1.1 §2's lesson, applied. The predicted failure is specific: `tile.mbarrier`
handles and `tile.pipeline_state` are **loop-carried** in a staged pipeline ring
(the phase bit flips per iteration), so they arrive at their use sites as
`scf.for` block arguments — and `WarpSpecLegalityPass.cpp:280` chases defining
ops, which block arguments do not have.

Run these two experiments against `build/tools/tessera-opt` and record the exact
diagnostics in this section before touching ODS:

| Experiment | Question |
|---|---|
| A staged pipeline whose `!tile.mbarrier` is carried as an `scf.for` `iter_args` | Does the legality pass silently *skip* the barrier (fail-open) or reject it? Fail-open is the dangerous answer and the likely one |
| The same with `!tile.pipeline_state` carried through `tile.pipeline_advance` across the loop back-edge | Does the ring's def-use chain survive the back-edge at all? |

**If either fails open, that outcome is a finding in its own right** and should
be recorded here regardless of whether the rest of Phase 1 proceeds: it means
the warp-spec gate does not fire on the canonical pipelined kernel shape, which
is every real FA-4 schedule.

**Do not proceed to 5.2 until both experiments have a written answer.**

#### 5.1.1 Results (measured 2026-08-15, `tessera-opt` at `2d05e823`, Strix Halo box)

Both experiments were run on the registered vocabulary with **no**
`--allow-unregistered-dialect`, through all three legality passes
(`--tessera-warpspec-legality --tessera-tile-pipeline-legality
--tessera-tile-barrier-reuse-legality`). **Both fail open.**

**Experiment 1 — loop-carried barrier/staged data: FAIL-OPEN, confirmed.**
The control (straight-line `tile.async_copy` → `tile.mma` with no token edge)
correctly fires
`'tile.mma' op WARPSPEC_MMA_NOT_TOKEN_SYNCED: tile.mma reads an async-staged
tile from a producer but has no !tile.async_token completion edge to it`.
The identical defect in the canonical pipelined shape — the staged tile arrives
at the `tile.mma` as an `scf.for` `iter_args` block argument — produces **zero
diagnostics, rc=0**: `WarpSpecLegalityPass.cpp`'s
`operand.getDefiningOp()` returns null on a block argument and the walk
`continue`s past it. Three further silent passes in the same run:

* an `!tile.mbarrier` carried as `iter_args` with a `tile.mbarrier.wait` inside
  the loop and **no arrive anywhere in the module** (a guaranteed device hang)
  verifies and passes all three passes;
* the wait's required "asynchronous dependency" was satisfied by the **loop
  induction variable** (an `index`) — and a separate probe confirmed an
  `arith.constant 42 : i32` also satisfies it: the check is shape-only;
* two `#tile.barrier` annotations on the **same SSA barrier value** with
  conflicting `expect` counts under different `tile.barrier_id` strings pass
  silently — `WARPSPEC_ARRIVAL_COUNT_MISMATCH` keys on the string, not the SSA
  value, so an aliased barrier escapes the arrival-count gate.

Also confirmed live: a `tile.mbarrier.wait` with **no barrier operand at all**
verifies and passes — `MBarrierWaitOp::verify()` guards its dependency
requirement behind `getBarrier()`, so absence of the barrier bypasses both
checks (`TileOps.cpp:2374`).

**Experiment 2 — pipeline ring across the back-edge: expressible, unverified,
FAIL-OPEN on the semantic defect.** The well-formed ring (`tile.pipeline_init`
→ `iter_args` → `tile.pipeline_advance` → `scf.yield %next`) parses, verifies,
and passes legality — the ring **is** expressible across the back-edge. But the
**stale ring** — the loop yields the *original* state instead of the advanced
one, so the pipeline never advances (the classic stalled-ring bug) — also
passes everything, **rc=0, zero diagnostics**. Nothing walks the ring's def-use
chain; `TilePipelineLegalityPass` checks only per-`pipeline_init` phase
asymmetry and string-keyed barrier-kind consistency.

**Correction to §3.2 the experiments forced — three rows are narrower than the
ODS scan suggested** (the §9 "upper bound, not proof" caveat, realized):

| §3.2 row | Measured status |
|---|---|
| `tile.pipeline_advance` "need not consume a `pipeline_state`" | **Wrong at behavior level.** `PipelineAdvanceOp::verify()` (`TileOps.cpp:1604`) rejects a non-`!tile.pipeline_state` first operand: `'tile.pipeline_advance' op first operand must be the prior !tile.pipeline_state`. The hole is ODS-only; §5.2 row 6 becomes an ODS *hoist* plus back-edge derivation, not a new constraint |
| `tile.tma.copy_async` "gates on nothing verifies" | **Partially mitigated.** An SSA-barrier-bound copy without `expect_tx` is rejected: `'tile.tma.copy_async' op an SSA mbarrier binding requires explicit expect_tx bytes` (`TileOps.cpp:2332`). A copy with *no barrier at all* still verifies |
| `tile.mbarrier.wait` optional barrier | **Confirmed, with sharper shape.** Barrier present ⇒ a dependency is required but its *type* is unchecked (any value passes); barrier absent ⇒ nothing is checked at all |

**Consequence for §5.2/§5.3 sequencing:** the ODS rows 1–5 stand as scoped, but
the center of gravity moves to §5.3 — the load-bearing defect is that **no pass
derives anything across a block-argument edge and the arrival/wait pairing has
no SSA form**, not that op verifiers are absent. The warp-spec gate does not
fire on the canonical pipelined kernel shape (every real FA-4 schedule), which
was the outcome §5.1 said would be a finding in its own right. Experiment
fixtures preserved in `research/` are the seeds of §5.4's ported fixtures.

### 5.2 ODS tightening

Ordered by risk, lowest first. Each row is independently landable.

| # | Change | Risk | Note |
|---|---|---|---|
| 1 | `tile.mbarrier.wait`: barrier operand becomes **mandatory**; add `Tile_MBarrierTokenType:$token` | Low | The target form is `tile.try_wait` in the same file. Zero design risk — copy it |
| 2 | `tile.tma.copy_async`: barrier operand becomes mandatory | Low | A TMA copy gating on nothing is never correct |
| 3 | `tile.tma.descriptor`: `AnyType $source` → buffer/memref constraint | Low | |
| 4 | `tile.tmem.load` / `.store`: `Variadic<AnyType> $indices` → `Variadic<Index>` | Low | |
| 5 | `tile.alloc`: `AnyAttr:$layout` → `TileLayoutAttr` | Low | Decision #21a — an unvalidated attribute where a typed one states the legal set |
| 6 | `tile.pipeline_advance`: require at least one `Tile_PipelineStateType` input | **Medium** | This is the ring closure. Depends on 5.1's second experiment |
| 7 | `tile.tma.copy_async`: `Variadic<AnyType> $outputs` → typed buffer results | **Medium** | Connects a copy to the buffer it filled — the precondition for a data-consistency check |
| 8 | `tile.tcgen05.mma`: `AnyType $lhs/$rhs` → `!tile.fragment<…, family="tcgen05">` | **Deferred** | Sequence *after* W1.1 step 5, or it forks the fragment contract (§3.5) |

Rows 1–5 are the phase's floor: they are mechanical, they each admit a two-line
negative fixture, and they are worth landing even if 6–8 stall.

#### 5.2.1 Rows 1–5 landed (2026-08-15) — with one scope correction the tree forced

All five rows are in, drift-gated by
`tests/tessera-ir/phase2/tile_sync_typed_invalid.mlir` (six rejected-now cases)
plus a positive arrive→wait token case in `phase2/tile_async_hardware_ops.mlir`.
Full lit suite green (323 passed / 0 failed); `check-tessera-rocm` green except
the pre-existing, unrelated `gfx1151_philox_distributions.mlir` failure (also
fails on the base tree — reported separately).

**Scope correction — rows 1–2 could not make the barrier ODS-mandatory, and the
reason is a §3.3-class finding in its own right.** The NV lowering assembles
synchronization in two stages: `AsyncCopyLoweringPass` emits `tile.mbarrier.wait`
and `tile.tma.copy_async` **without barrier operands** (segments `{0,0,N}` /
`{1,0,N}`), and `NVTMADescriptorPass` retrofits the barrier, slots, and
`#tile.barrier` attrs afterward. An ODS-mandatory barrier would make the
pipeline's own intermediate IR unverifiable between those passes. So the landed
form is:

* **Row 1** — `tile.mbarrier.wait` gains a typed
  `Optional<Tile_MBarrierTokenType>:$token` slot (**the arrive→wait edge is now
  expressible in the type system**), and `MBarrierWaitOp::verify` fails closed:
  every dependency must be `!tile.async_token` / `!tile.mbarrier_token`
  (`TILE_WAIT_UNTYPED_DEPENDENCY` — kills the measured index/i32 hole), and a
  wait with no token, no dependencies, and no declared semantics is rejected
  (`TILE_WAIT_GATES_ON_NOTHING`). The legacy keyless `tile.wait_async`
  ("retire everything outstanding") is now an **explicit** `tile.retire_all`
  marker stamped by the lowering and replaced with the concrete token set by
  `NVTMADescriptorPass` — Decision #21a applied to what was an
  indistinguishable bare form. Barrier-mandatory is deferred to the emission
  restructure (barriers assigned at birth, not retrofitted); tracked as the
  revised row 6/7 follow-on.
* **Row 2** — `TMACopyAsyncOp::verify` rejects a copy with no SSA barrier, no
  `!tile.async_token` result, and no legacy grouping key
  (`TILE_TMA_COPY_GATES_ON_NOTHING`): nothing could ever retire it.
* **Row 3** — descriptor `$source` is `AnyTypeOf<[Tile_BufferType, AnyMemRef,
  AnyRankedTensor]>`; ranked tensors stay legal because the value lane stages
  tensors by contract.
* **Row 4** — `tile.tmem.load/.store` indices are `Variadic<Index>`.
* **Row 5** — `tile.alloc` layout is a typed `#tile.layout` constraint in ODS.
  The verifier **already enforced this** (`AllocOp::verify`) — a fourth
  §3.2-row-narrower-than-the-ODS-scan case; the hoist moves rejection to parse
  time.

**§5.6 fixture-break count: 3.** `nvtma_barrier_emission.mlir` and
`warpspec_async_token.mlir` broke through the legacy keyless-wait lane (the
lowered bare wait now fails closed — fixed by the `tile.retire_all` design, not
by weakening the gate), and `tile_async_hardware_ops.mlir` needed the
three-segment migration. That count is the first datum for §7's `p`: on a
322-fixture suite, three sites were relying on under-constrained sync IR.

### 5.3 Verifier rewrite

#### 5.3.1 First increment landed (2026-08-15) — the block-argument edge is closed

`TileDataflowLegalityPass` (`--tessera-tile-dataflow-legality`, W2.4's named
pass, born here) plus a shared loop-carry resolver
(`src/transforms/lib/TileValueProvenance.h`) now derive across `scf.for`
`iter_args` — both the init operand and the back edge — failing closed on the
underivable. **Every §5.1.1 silent row has flipped**, verified against the
preserved probes and drift-gated by
`tests/tessera-ir/phase2/tile_dataflow_legality.mlir` (positive control: the
canonical well-formed loop pipeline produces no diagnostic):

| §5.1.1 silent case | Now fires |
|---|---|
| mma reading a loop-carried staged tile with no token edge | `WARPSPEC_MMA_NOT_TOKEN_SYNCED` (WarpSpecLegality's producer walk resolves through the shared resolver) |
| wrong-slot arrive→wait on a loop-carried barrier | `TILE_WAIT_SLOT_MISMATCH` (+ `TILE_WAIT_TOKEN_UNPAIRED`, `TILE_WAIT_BARRIER_DISAGREES` for the sibling defects) |
| barrier of unprovable origin (e.g. function argument) | `TILE_BARRIER_ORIGIN_UNRESOLVED` — exit gate 2's fail-closed derivation |
| stale / restarted pipeline ring across the back edge | `TILE_PIPELINE_RING_STALE` / `TILE_PIPELINE_RING_UNDERIVED` |
| same-SSA-barrier conflicting `expect` under two string ids | `TILE_TMA_EXPECT_MISMATCH` (SSA-keyed through the descriptor edge, superseding string-keyed identity for this case) |

Suite state: full lit 324/0; `check-tessera-rocm` green modulo the
pre-existing unrelated Philox failure; full `ninja` build clean.

#### 5.3.2 Second increment landed (same day) — predicates typed, hatches deleted, fixtures ported, pipelines wired

The rest of §5.3 and all of §5.4, in one change:

* **Registered sync vocabulary.** Four new ODS ops close the husk gap:
  `tile.cta_sync` (WarpSpecialization's by-name emission now resolves to the
  registered op — zero emitter changes), `tile.fence`, `tile.tma.store`, and
  `tile.buffer_write` (one op replacing the four smem/tmem/lds/reg husk write
  spellings — the space comes from the alloc, Decision #31). The unproduced
  ops' named consumer is the legality gate itself (Decision #29); their
  emission lands with the warp-spec lowering growth.
* **The six substring predicates are gone.** `isBarrierInit` / `isCollective`
  / `isTmaStore` / `isVisibilityFence` / `isBufferFree` and the barrier-reuse
  release predicate are typed `isa<>` queries. The **attribute escape hatches
  (`tile.barrier_init`, `tile.collective`, `tile.tma_store`, `tile.fence`,
  `tile.buffer_free`) are deleted** — measured first: no pass ever stamped
  them, so they were pure fail-open surface. Exact-name (not substring)
  matches remain for exactly three cases with real or pending producers
  (`tile.cp_async.commit_group` — the SM<90 trio AsyncCopyLowering emits —
  and `tile.cluster_sync` / `tile.tile_scheduler.next_tile`), each annotated
  to graduate to ODS with its producer.
* **§5.4 fixture port complete.** Both legality fixtures run **without**
  `--allow-unregistered-dialect` on the registered vocabulary (loop markers
  became `scf.for` with the C6 trip-count attributes). The §3.3
  "registered path and verified path are disjoint sets" finding is closed.
* **Pipeline wiring.** `--tessera-tile-dataflow-legality` runs inside both
  NVIDIA pipeline builders after the post-NVTMA C3/C6 blocks — and the
  pipelines' own lowered output (matmul → warpspec → TMA retrofit) **passes
  its own derivation gate**, the dogfood result that makes the wiring safe.
  One cosmetic fixture fix: registered `tile.cta_sync` prints unquoted.

**Remaining for Phase 1 exit:** the §5.5 gate 5 numerical run on gfx1151, the
barrier-at-birth emission restructure (revised rows 6–7), and row 8 after
W1.1 step 5. The verifier-derivation scope of §5.3 is **closed**.

The original scope, kept for the record:

Replace the six substring predicates in `WarpSpecLegalityPass.cpp:76–119` with
type- and def-use-derived queries. Concretely:

* `isBarrierInit` → "the op defines a value of type `!tile.mbarrier`."
* The arrive→wait pairing → SSA reachability on `!tile.mbarrier_token`, made
  possible by 5.2 row 1.
* `isAsyncDataProducer` → operand-type query, not `n == "tile.async_copy"`.
* The attribute escape hatches (`hasAttr("tile.barrier_init")` and friends)
  become **migration-only**, marked as such, and are deleted in the same change
  that ports the fixtures (5.4).

Per §3.3 consequence 2, any predicate that cannot be derived must **fail closed**
until the W2.1 dataflow framework exists — treat unprovable as unsafe, do not
assume the permissive answer.

### 5.4 Fixture migration

Port `tests/tessera-ir/phase2/tile_warpspec_legality.mlir` and
`tile_barrier_reuse_legality.mlir` off `--allow-unregistered-dialect` onto the
registered vocabulary, closing the disjoint-sets gap in §3.3. Add the legality
passes to at least one registered fixture (`phase3/flash_attn_full.mlir` is the
natural one — it already uses `tile.mbarrier.init`, `.arrive_expect_tx`,
`.try_wait`, `tile.tma.copy_async`, and `tile.tma.descriptor` without
`--allow-unregistered-dialect`).

Ship a `CHECK-NOT` negative fixture per Decision #10a.

### 5.5 Exit gates

Ordered, and each is necessary:

1. **Operand-shape gate — what the type system can actually decide.** A
   `tile.mbarrier.wait` with **no** barrier operand, or with an operand that is
   not a `!tile.mbarrier`, is rejected with **no pass running**. Same for
   `tile.tma.copy_async`. That is the whole of what ODS decides here.

   > **Scope correction (PR review, 2026-08-15).** An earlier draft wrote this
   > gate as "a wait whose barrier never came from a `tile.mbarrier.init` is
   > rejected by the type system." **That is not achievable and the draft
   > contradicted itself**: a mandatory operand type constrains the *shape* of
   > the value, never its *origin*, so a function argument or a loop-carried
   > `scf.for` block argument of type `!tile.mbarrier` satisfies the ODS
   > constraint completely. The loop-carried case is not hypothetical — it is
   > the exact shape §5.1 exists to investigate, which is what makes the error
   > worth recording rather than silently fixing. **Provenance is a derivation
   > property and belongs to gate 2.**

2. **Derivation gate — where provenance is actually decided.** A verifier
   establishes barrier origin by def-use reachability to a `tile.mbarrier.init`,
   **including across a block-argument edge** (an `scf.for` `iter_args` barrier
   must resolve to its initializer through the loop's init operand and
   back-edge, not fail to resolve and pass). `WARPSPEC_INIT_UNDER_GUARD` fires
   from role-set / def-use reachability, and the same fixture passes with every
   `tile.barrier_init` attribute escape hatch deleted. Per §5.3, a barrier whose
   origin cannot be resolved **fails closed** — the fail-open answer is the
   defect §5.1 is looking for, not an acceptable outcome.
3. **Registered-path gate.** Both legality fixtures run without
   `--allow-unregistered-dialect`.
4. **Lowering gate** (W1.1 §4.1). The registered pipelined fixture *lowers*, not
   merely verifies.
5. **Numerical gate** (W1.1 §4.2). A pipelined kernel through the tightened path
   is numerically verified **on gfx1151**. A lowering fixture cannot catch the
   wrong-answer failure mode; W1.1 has the scar to prove it.

   > **Closed for the executing lane (2026-08-15).** On the tree carrying every
   > §5.2/§5.3/§5.4 change: the staged global→LDS token round-trip and the
   > staged LDS+WMMA GEMM (typed `!tessera_rocm.token` — the Target-IR form of
   > the tightened `tile.async_copy`/`wait_async` contract) execute on gfx1151
   > and match numpy (`test_rocm_async_copy_runnable.py`,
   > `test_rocm_gemm_staged_async_copy.py`); the via-Tile seam gates are 10/10
   > (`test_rocm_pipeline_tile_lowering.py` — `tile.mma` flows through
   > `lower-tile-to-rocm`, not around it); and the **full compiled ROCm device
   > sweep — 1,569 tests (every `_compiled` lane + staged pipeline + canonical
   > GEMM matrix) — passes in 61 s**, all genuine launches (skip-gated on GPU
   > presence, none skipped). **Scope honesty:** the `tile.mbarrier.*` /
   > `tile.tma.*` family this phase retyped most directly has **no device lane
   > on this fleet** — consumer sm_120 lives on the other box and Hopper is
   > Phase G/H — so for that family the evidence remains lowering + pipeline
   > legality (gates 1–4); the gfx1151 numerics cover the shared token
   > contract, which is what this fleet can execute.

   > **SO-2 reconciliation (2026-08-16).** The gate was rerun after the shared
   > role carrier became executable rather than inferred from the older sweep.
   > `rocm-wave-lds-pipeline` now emits producer/consumer `tile.role` SSA and
   > binds it to `tile.pipeline_init`; `rocm-wave-lds-legality` consumes and
   > verifies that relationship. On the visible gfx1151, the exact named cohort
   > (`test_rocm_async_copy_runnable.py`,
   > `test_rocm_gemm_staged_async_copy.py`, and
   > `test_rocm_pipeline_tile_lowering.py`) passed **8/8**, with no skips. The
   > prior 1,569-test sweep remains broader parity evidence, but is no longer
   > used as a substitute for this particular gate.
6. **No-regression gate.** `ninja -C build` (all targets, not `tessera-opt`
   alone), `lit tests/tessera-ir/`, and `ninja -C build check-tessera-rocm` —
   the second lit suite CI runs and `check-tessera` does not.

### 5.6 What Phase 1 must produce for later phases

Beyond the gates, Phase 1 owes three artifacts downstream:

* **A written answer to 5.1**, which determines whether the pipeline ring is
  expressible at all — a precondition for Phase 2's barrier-role edge.
* **A count of how many existing fixtures break** under rows 1–5. That number is
  the first real estimate of how much IR in the tree is currently
  under-constrained, and it feeds §7's `p`.
* **A decision on row 8's sequencing** against W1.1 step 5, recorded here.

---

## 6. Phase 2 — roles and producer/consumer sets as first-class IR

**Precondition:** Phase 1 exit gates 1–3. Roles without derived verification are
another declaration with no consumer.

### 6.1 The problem being closed

Roles exist three times, in three incompatible forms, none of them load-bearing:

| Where | Form | Status |
|---|---|---|
| `python/tessera/compiler/wave_specialization.py` | A target-parametric descriptor (wave groups, waves each, role per group, CDNA ping-pong rotation, barrier count) | **Declaration with no consumer.** Its own docstring says it is "the design contract the warp-specialization lowering *should* consume" |
| `schedule.warp` op | `role` StrAttr, read by `isAsyncDataProducer` | Schedule-level only; does not survive into Tile IR |
| `tile.warp_role` | A StrAttr on an ancestor region, matched by `hasAttr` | Untyped, unenumerated, and invisible to the type system |

This is a Decision #29 instance (`wave_specialization.py`), a Decision #31
instance (three implementations of one concept), and a Decision #21a instance
(an unvalidated `StrAttr` where an `EnumAttr` states the legal set) in one place.

### 6.2 The design

Take CAKE's edge and nothing more:

```mlir
%load = tile.role {warps = [0]} : !tile.role
%mma  = tile.role {warps = [1]} : !tile.role
%pipe = tile.pipeline_init {depth = 3, ...} : !tile.pipeline_state
%q_full = tile.mbarrier.init {slots = 3, phase_bits = 1}
            producers(%load) consumers(%mma) pipeline(%pipe)
          : !tile.mbarrier
```

Three properties, in decreasing order of importance:

1. **The barrier carries its producer and consumer role sets as SSA operands.**
   This is the whole point: a synchronization verifier becomes reachability over
   `role → barrier → role`, rather than "is there an ancestor with a
   `tile.warp_role` attribute whose name contains `producer`."
2. **`!tile.role` is a type, not a string.** Role identity must survive a block
   argument (a role held across a loop is ordinary), which is exactly the
   constraint W1.1 §2 discovered the hard way for fragments. Apply the lesson:
   **run the loop-carried role case first.**
3. **The role descriptor is consumed, not re-derived.** `wave_specialization.py`
   already parameterizes both the Hopper producer/consumer split and the CDNA
   ping-pong rotation. Phase 2 makes it the source of the role sets rather than a
   parallel document. Its phase-rotation model is the harder case and should be
   the acceptance fixture, not an afterthought — a role that *rotates* per phase
   is the shape that breaks a naive "role is a static partition" design.

### 6.3 Non-goals for Phase 2

* **No warp-count inference.** Roles state which warps; lowering derives warp
  identity, exactly as CAKE does. Do not put physical warp IDs in the IR.
* **No new pass.** Phase 2 rewrites the Phase 1 verifier's role predicates to
  read role sets; it does not add a pass.
* **No authoring surface.** That is Phase 3, and it must not be entangled here —
  Phase 2 must be provable entirely from MLIR fixtures.

### 6.4 Exit gates

1. `WARPSPEC_INIT_UNDER_GUARD` and the barrier-reuse rule are derived from
   `role → barrier → role` reachability, with the `tile.warp_role` attribute
   path deleted.
2. **The CDNA ping-pong schedule from `wave_specialization.py` verifies through
   the same rule as the Hopper producer/consumer split**, with no target-specific
   branch in the verifier. If it needs a branch, the role model is wrong and
   should be re-scoped before Phase 3 depends on it.
3. A loop-carried `!tile.role` survives an `scf.for` block argument (6.2
   property 2).
4. A negative fixture where a barrier's consumer set is empty is rejected.
5. Lowering + gfx1151 numerical gate, per Phase 1 §5.5 gates 4–5.

### 6.5 What Phase 2 must produce for later phases

* **A verdict on gate 2**, which is the real test of whether one role model spans
  NVIDIA warp specialization and AMD ping-pong. A negative verdict is a finding,
  not a failure, and it would materially change Phase 3's surface.
* **The role-set vocabulary** that Phase 3's builder will construct, and that
  Phase 5's diagnostics will name.

---

## 7. Phases 3–6, sketched — and the gates that must precede them

Deliberately not scoped in detail: each depends on an answer Phase 1 or 2
produces. Recorded here so the direction is legible and so the gates are not
invented after the fact.

**Phase 3 — `@tessera.schedule` authoring surface.** A Python builder emitting
the registered Tile vocabulary. Not a new IR — a constructor for one we have
(§3.4 keeps it one IR with two entry points, per Decision #31). **Justified on
convergence, never on the speedup number (§2.4):** the success criterion is that
a schedule authored through the builder reaches a correct, verified kernel in
fewer repair rounds than editing MLIR text, which is measurable at 2M tokens.
*Gate:* Phase 2 exit gate 2. A builder that emits a role model which needs a
per-target branch would bake that branch into the user-facing surface.

**Phase 4 — two-stage arbiter.** `emit/candidate.py:365` currently does
`min(cands, key=measure)`, so `measure` is the **terminal selector**. Wiring an
analytical predictor into that hook makes selection exactly as good as the model
and leaves **regret unbounded** — an inverted prediction picks the worst
candidate. The correct form is `predict → prune to top-k → measure survivors →
select by measurement`, which bounds regret to zero whenever the true best
survives the prune, with the F4 oracle already guaranteeing correctness is never
traded.

Two constraints on the predictor, both from `target_perf.py`'s own TileSight
citation (88.6 of 117 spec TFLOP/s; 1.4 of 1.8 TB/s — 24% on both):

* A **uniform** multiplicative bias is rank-preserving and therefore harmless for
  pruning. Non-uniform bias is not.
* Tile-size and stage-depth sweeps share a code shape, so roofline bias is
  roughly constant across them — rank-safe. A Tier-3 hand-tuned MSL kernel and a
  Tier-1 synthesized kernel are different programs with different bias — **not**
  rank-safe. **Rule: the predictor may prune within a tier and must never rank
  across tiers.** This also preserves Decision #28 lead-safety for free.

*Gate — measure before building.* With rejection fraction `p`, static cost `c_s`,
and GPU cost `c_g`, throughput gain is `1 / ((c_s/c_g) + (1 − p))`, and the
filter is net-positive iff `c_s/c_g < p`. **CAKE never reports `p`.** Instrument
it first — it is a counter, not a project — using Phase 1 §5.6's fixture-break
count as the first data point. If `p < 0.2`, Phase 4 does not pay and should be
dropped rather than built on faith.

**Phase 5 — portfolio and leakage discipline.** CAKE §5 adopted as governance
(§4 Take). Also the natural home for a **distributional perf regression
detector**: `flywheel.py` already records `median_ms`, `p10_ms`, and `reps`, so
the data exists, and our "never assert a fixed number" discipline — which is
right — currently means there is no regression detector at all. Flag when a new
p50 falls outside the historical p10–p90 band for that `device_id`. Distribution
based, so the discipline is honored rather than broken.

**Phase 6 — frozen-source provenance** for Tier-3 candidates, per FlashInfer PR
#4262 (§4 Take). Independent of Phases 1–5; landable any time.

---

## 8. The capability layer beyond CAKE

CAKE's admitted limits are where differentiated work is. Ranked by (developer
demand × existing substrate here):

**1. Schedule-level autodiff — the largest, and nobody has it.** CAKE's
known-kernel table lists FA4 **BWD** as 514 lines of *hand-authored* CAKE IR. We
have `autodiff/{tape,vjp,jvp}`, [`LSE_CHECKPOINT_CONTRACT.md`](LSE_CHECKPOINT_CONTRACT.md),
and `compiler/scheduled_attention_backward.py` as a hand-written precedent. The
capability is to **derive the backward schedule from the forward schedule** —
transpose the dataflow, invert the pipeline direction, re-derive barrier
producer/consumer sets from the transposed graph, and reuse the LSE checkpoint
contract for recompute. Every training kernel (KDA, GDN, FA-4, MoE) needs a
backward, and today every one is written twice. Phase 2's role/barrier edges are
what make the re-derivation expressible.

**2. Causal bottleneck attribution.** CAKE returns "broad bottleneck classes."
We have the full profiler stack plus a roofline residual on the same flywheel
row. The missing link is *causal*: from a measured stall back to **the IR
decision that caused it** — this stage depth, this barrier, this tile size.
Phase 2 is the precondition, because attribution needs a schedule decision to
point at.

**3. Declared cross-target schedule loss.** Decision #32 already requires that
information loss across a boundary be declared. CAKE concedes non-NVIDIA
transfer is unmeasured. One authored schedule → four targets → **an explicit
report of what each target dropped and why** is a capability nobody ships, and
it is the honest version of "portable schedule."

**4. Accuracy budget as a search axis.** We are ahead — F4 oracle plus
`numeric_policy` plus `math_mode`; CAKE does bitwise/tolerance only. Exposing
accuracy budget as a dimension the arbiter *searches over* rather than only
*gates on* is directly what quantization and low-precision work asks for.

**5. Reproducible search.** Seedable, replayable evolution with a decision log.
`debug.replay` manifests and `check_determinism` exist. CAKE retains results in
an artifact but the search itself is not reproducible.

**6. Expert seeding.** Inject a known schedule as the search's starting point and
constrain the space around it. This is the highest-leverage answer to §2.4: it is
how a 55M-token budget becomes a 5M-token one.

---

## 9. Honest limits of this assessment

* **The statistics in §2 are exact but the samples are the paper's.** If the
  authors release per-run data, §2.1 should be recomputed; the medians and ranges
  fix the triples only because n=3.
* **§3's counts are ODS-derived, not behavior-derived.** An `AnyType` operand can
  still be constrained by a hand-written verifier. All 55 ops' `verify()` bodies
  were not audited, so the count is an upper bound on the hole, not a proof of
  it. The eight ops named in the §3.2 table were read individually. The §3.2
  inclusion rule is stated so the scan is reproducible — the first version of
  this count was wrong in both numerator base and denominator, and an
  unreproducible scan is how that survived to review.
* **The 5.1 experiments have not been run.** They are the phase's first work
  item precisely because their outcome could invalidate 5.2 rows 6–7.
* **Phase 2 gate 2 (one role model across NVIDIA and AMD) is a genuine open
  question**, not a formality. `wave_specialization.py` models both, but modeling
  and verifying through one rule are different claims.
* **No CAKE source exists to check against.** Unlike SparDA, this assessment is
  paper-plus-artifact only; the FlashInfer PR shows generated output, not the
  compiler. Claims about CAKE's internals are the paper's, not verified.
