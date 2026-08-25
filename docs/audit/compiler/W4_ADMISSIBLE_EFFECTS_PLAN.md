---
last_updated: 2026-08-25
audit_role: plan
plan_state: open
---

# W4-EFFECTS-1 — operation-owned recorded products for admissible effects

Scoped plan for the last open item of integrated-plan queue **order 2**
(W4-PRODUCT-1): *"one physical packet family with admissible effects."*
Ordering authority stays with
[`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md); the compiler map
and authority chain are in [`README.md`](README.md). This document owns the
design and acceptance detail only.

**Status.** Slice **E1 is implemented**
(`python/tessera/compiler/recorded_product.py`, tests in
`tests/unit/test_recorded_product.py`): the carrier, its per-class (R)
requirements, the (C) confinement checks, region-level totality, and the
content address. **E2's gate half is landed too**: the blanket
`AUTODIFF_STOCHASTIC_EFFECT` refusal is split into replayability
(`AUTODIFF_STOCHASTIC_NO_PRODUCT` / `AUTODIFF_STOCHASTIC_UNKEYED`) and
differentiability (`AUTODIFF_OP_NOT_DIFFERENTIABLE`), and
`stochastic_product_for_call` admits only call forms whose replay is a
function of recorded data — measured, not assumed: `dropout(x, p,
seed=N)` replays bit-identically while the ambient and
caller-generator forms do not. **E2b** (the dropout `AdjointInterface` that replays the mask from the
product) is landed too: the Jacobian is `diag(m/(1-p))`, diagonal and
hence its own transpose, so `dx = dropout(dout, same key)`. **E3 is
landed**: `dtype` is a real parameter of the lineage identity (the
measured precondition — the default keeps every existing lineage id
byte-stable), and a mutation product binds lineage id + version + a
**content digest**, with `verify_recorded_state` rejecting BOTH
directions: changed bytes under an unchanged identity, and — since the
PR #630 review — the right bytes under the wrong identity, because
zero-initialised optimizer state makes distinct lineages byte-identical
and a content-only check would call that replay faithful. The caller
must name the lineage and version it is replaying; there is no default
(#21a). **E4 is landed**: the product binds communicator + order + reduction tree + topology, the verifier rejects a permuted order AND a changed tree under an identical order, and the recorded order is taken from the real W5.4 mock-mesh executor rather than a synthetic list — with the scope boundary explicit in code and tests, that this establishes ORDER and that result bit-identity still requires native deterministic evidence. **E5 is landed**: the keyed Philox RNG family carries its recorded product to real hardware. Exact-device rows on BOTH hosts — replay from the product alone is bit-identical on gfx1151 and on AVX-512, and the SAME product gives identical bits on both targets and matches the algorithm's independent reference, so a recorded product is portable evidence rather than a per-target coincidence. Rows assert the exact `execution_kind` per target, so a ROCm row cannot pass by falling through to a CPU lane. Correctness only (WSL, Decision #26a); no timing claimed. **With E5, queue order 2's remaining item is closed.** The PR #630 review additionally moved recorded-product admission into one shared verifier (`recordedProductFailure` in `SemanticEffects.cpp`) called by the paired pass, the structured region replayability walk, and the structurized-CFG walk, so an admitted keyed draw stays admitted inside `scf.if`/`scf.for`/`scf.while` instead of failing `AUTODIFF_REGION_ADJOINT` — the family was previously admissible only in straight-line code (#31). Every claim below that could have been
assumed was measured instead; the measurements are in §5.

---

## 1. What is actually blocked today

`AutodiffPairedPass` admits a region into reverse mode only when every op on
the path is differentiable and effect-free, with one carve-out: *compiler
generated* replay-safe assertions (the `cf.assert` extent checks W4.3
landed). Everything else fails closed:

| Gate | Site | Diagnostic |
|---|---|---|
| Stochastic | `hasStochasticEffect` → `SemanticEffectLevel::Random` | `AUTODIFF_STOCHASTIC_EFFECT` |
| Nested region | unsupported nested-region op | `AUTODIFF_NESTED_REGION` |
| Non-differentiable | no registered adjoint | `AUTODIFF_OP_NOT_DIFFERENTIABLE` |
| Mutation / state / I/O / ordered collective | registered effect > pure | region rejected |

Failing closed here was correct: a region whose replay differs from its
record produces a **wrong gradient**, silently. The task is not to weaken
the gate but to give each effect class a *product* that makes replay
provably identical — and to keep the classes that cannot have one closed.

---

## 2. The admissibility criterion

An effectful operation `E` is **admissible** in a recorded product iff there
exists a recorded value `π(E)` — content-addressed, carried in the package —
such that:

> **(R) Reproducibility.** For all replays, `E(inputs, π)` equals the
> recorded execution **bit-for-bit**, not merely in distribution.
>
> **(C) Confinement.** `E`'s write-set is contained in values `π` names.
> Nothing outside the recorded frame observes `E`, and `E` observes nothing
> outside it.

(R) without (C) admits an op that reproduces its own value while mutating a
neighbour's state; (C) without (R) admits an op that touches nothing but
returns different numbers on replay. Both are required, and both must be
*checked by the verifier*, not asserted by the producer.

A class that cannot satisfy (R) — a genuine external read — is not made
admissible by better bookkeeping and stays closed. That is a conclusion, not
a limitation to route around.

---

## 3. Per-class verdicts

### 3.1 Keyed RNG — **ADMISSIBLE** (first slice)

The enabling fact is mathematical, not architectural: the S4 generator is
counter-based (Philox), so a draw is a **pure function of its key**. §5.1
measures purity, replay bit-identity, split independence, derivation
collision-freedom, per-rank disjointness, and distributional sanity.

* `π(E)` = the op's `RNGKey` (seed + counter + derivation path) plus shape
  and dtype — nothing else. This is already what
  `tessera.stochastic_identity` names; the product makes it *carried and
  verified* rather than merely declared.
* (R) holds by purity; (C) holds because a keyed draw writes only its result.
* **Unkeyed** RNG stays closed: no `π` exists, which is exactly the split the
  queue row already anticipates ("unkeyed RNG … remain fail closed").
* Adjoint note: admitting the op into the region is orthogonal to *how* it is
  differentiated. `AUTODIFF_STOCHASTIC_EFFECT` currently conflates the two.
  Splitting it is part of this slice: a keyed draw with a registered pathwise
  or score-function rule is admitted; one without a rule still fails, with a
  diagnostic that says which of the two is missing.

### 3.2 Mutation of recorded state — **ADMISSIBLE, reusing the existing ABI**

`tessera.state_buffer_lineage.v1` (`stateful_training.py`) already
content-addresses `(name, role, shape, dtype, version, access, parents)`.
That is precisely a mutation product: replay reads the recorded **version**.

* `π(E)` = for each buffer `E` writes, its lineage id, the version it
  advances to, **and a content digest of the bytes that version names**.
  (C) holds when the declared `access` covers the write-set.
* **(R) does NOT follow from the lineage id and version alone** — corrected
  2026-08-25 after review. `_buffer` hashes name, role, shape, dtype,
  version, access, and parent ids; it does **not** hash contents (§5.2). Two
  buffers with identical metadata and different bytes therefore share a
  lineage id, so a replay that binds "version N" can bind *different bytes*
  and produce a different gradient — silently, which is the exact failure
  mode this whole gate exists to prevent. The metadata lineage is an
  identity, not a value authority. The product must therefore carry either a
  content digest or an immutable snapshot, and the verifier must check it;
  a version-to-content authority (a buffer store keyed by
  `(lineage_id, version)` whose entries are immutable) is the alternative
  design if snapshot cost is prohibitive. Either way, **metadata identity is
  necessary and not sufficient**.
* **Do not invent a second lineage schema** (#31). Extend the existing one.
* **Named precondition, measured in §5.2:** the identity separates version,
  shape, access, parents, and role — but `dtype` is *hardcoded* `"f32"` with
  no caller override. Every lineage constructed today is f32, so nothing
  collides now; the moment recorded state becomes mixed precision (bf16
  master weights, fp8 optimizer state) two materially different buffers would
  share an id. Making `dtype` real is a **precondition of this slice**, not a
  follow-up, and it is exactly the defect class that produced the MegaMoE
  schedule-digest fix in #625.

### 3.3 Ordered collectives — **ADMISSIBLE, gated on the schedule authority**

An ordered collective's requirement is that every rank issues collectives in
the *same relative order*. That is a property of a total order, and the
schedule authority now derives exactly that: since #625 the inference emits
collective↔collective ordering edges (and nothing spurious), and since #626
the pipeline carrier's rows are validated for a topological order.

* `π(E)` = the communicator identity, the position of `E` in the recorded
  collective sequence (a digest over that sequence), **and the reduction
  tree / algorithm plus the topology parameters that select it**.
* **Issue order alone does NOT give (R)** — corrected 2026-08-25 after
  review. Floating-point addition is not associative, so the reduction tree
  is part of the result rather than an implementation detail. Measured
  (§5.3): the same 1024 f32 values reduced with identical inputs and
  identical issue order give **three different bit patterns** for
  sequential, pairwise-tree, and ring reductions, and the ring result
  changes again with the rank count. `LANGUAGE_AND_IR_SPEC.md` §11 already
  says this — *"Deterministic profiles require fixed collective ordering and
  reduction trees"* — and the first draft of this plan bound only the first
  half.
* (C) holds when the collective's write-set is its declared outputs.
* **Gate:** this slice may not claim multi-rank correctness from a mock mesh.
  W5.4's mock executor proves the SSA is consumable; DIST-NATIVE-1 owns the
  real transport. Recorded-product *identity* can land before that; a
  numerical multi-rank claim cannot.

### 3.4 I/O — **NOT ADMISSIBLE** (stays closed, by argument)

An external read is not a function of any recorded value, so no `π` satisfies
(R). The existing carve-out is narrower than it looks and should stay narrow:
a compiler-generated assertion is admissible because it is *observational* —
it writes nothing and its only effect is the abort decision, which the STATUS
/ trap contract already makes explicit. Recording a file read would record a
**value**, not the effect, and replay would silently diverge the first time
the file changed. This class is closed on principle; the plan says so rather
than leaving it looking unfinished.

### 3.5 Alias-sensitive work — **CONDITIONAL, already half-built**

Admissible exactly when the alias facts are *known*: W2.1 supplies roots, and
#625 made a registered op's declared `aliasing="none"` yield a fresh root even
when effectful. Unknown alias facts remain a fail-closed barrier — that is
(C) being enforced, and it must not be relaxed to make a family fit.

---

## 4. Delivery slices

Each slice is independently reviewable and carries its own evidence. The
first is deliberately the smallest one that exercises the whole ABI.

| # | Slice | Deliverable | Acceptance |
|---|---|---|---|
| E1 | **Product ABI + verifier — LANDED 2026-08-25** | One `tessera.recorded_product.v1` carrier: effect class, content-addressed `π`, declared write-set. A boundary verifier checks (R)-inputs are present and (C) write-set ⊆ declared, failing closed with a named diagnostic per class. | Positive and negative fixtures per class; a product whose write-set exceeds its declaration is rejected; **no** class is admitted without a `π`. |
| E2 | **Keyed RNG (dropout family)** — gate split LANDED 2026-08-25; the adjoint (E2b) remains | Split `AUTODIFF_STOCHASTIC_EFFECT` into *unkeyed* vs *no adjoint rule*; admit keyed draws with a registered rule; carry the key as `π`. | Replay of a recorded dropout region is **bit-identical** (not distributional); unkeyed still fails; the two diagnostics are distinguishable. Gradient checked against the analytic pathwise rule. |
| E3 | **Mutation, on the existing lineage** — LANDED 2026-08-25 | `dtype` becomes real in `state_buffer_lineage`; the product binds lineage id + version **+ content digest**; region replay reads the recorded version and the verifier checks the digest. | Bit-identical replay of a stateful step; a tampered version fails closed; **a buffer whose bytes changed under an unchanged lineage id + version is REJECTED** (the §3.2 correction, with a negative test); a mixed-precision lineage no longer aliases (§5.2). |
| E4 | **Ordered collectives (identity only)** — LANDED 2026-08-25 | Communicator + sequence digest **+ reduction tree/algorithm and topology** as `π`; replay issues the recorded order under the recorded tree. | Mock-mesh replay reproduces the recorded order exactly; a reordered sequence fails closed; **a changed reduction tree fails closed** even when order and inputs match. Bit-identity of a collective RESULT additionally requires native deterministic evidence on real transport (RCCL/NCCL) — the mock check cannot establish it, so E4 claims identity only and the numerical claim moves to E5/DIST-NATIVE-1. |
| E5 | **One physical packet family end to end** — LANDED 2026-08-25 | The queue row's actual ask: one family carrying an admissible effect through Schedule→Tile→target on x86 and gfx1151. | Exact-device rows on both hosts, bit-identical to the recorded execution; digests bound; no reference-lane fallback. |

Estimated shape: E1 is the load-bearing one; E2 is the cheapest real proof;
E5 is the row that lets queue order 2 close.

---

## 5. Measurements taken while scoping (2026-08-25)

Evidence, not assumption. Re-runnable from the commands in the PR that
introduced this file.

### 5.1 RNG facts the admissibility of §3.1 rests on

| Property | Result |
|---|---|
| Purity + bit-identical replay of `normal(key, …)` | identical, max abs diff `0.0` |
| Child streams pairwise distinct; distinct from parent | no duplicates |
| Max abs correlation between distinct child streams (n=4000) | `0.032` vs noise band `3/√n = 0.047` |
| Collisions over 800 distinct `split`/`fold_in` derivations | `0` |
| Per-rank stream duplication across 8 ranks | none |
| KS test vs `N(0,1)`, n=200000 | `D = 0.0018`, `p = 0.55` |

The first row is the one that matters: **(R) is satisfied by construction**,
so a recorded key is a sufficient product.

### 5.3 Why a collective needs its reduction tree bound (§3.3)

1024 f32 values, identical inputs, identical issue order:

| reduction | result |
|---|---|
| sequential (rank order) | `-50370.79` |
| pairwise binary tree | `-50370.76` |
| ring, 8 partials | `-50370.758` |

Pairwise-distinct bit patterns, max gap `3.1e-2`; the ring result changes
again with the rank count (p = 2, 4, 8). Floating-point addition is
non-associative, so the tree is part of the value.

### 5.2 What the mutation lineage identity separates

Distinguishes `version`, `shape`, `access`, `parents`, `role` — verified
individually. Does **not** carry a real `dtype` (hardcoded `"f32"`, not a
parameter). Hence the precondition in §3.2.

---

## 6. What this plan deliberately does not do

* It does not weaken any existing gate to make a family fit. Every class that
  stays closed does so with an argument, not a TODO.
* It does not claim multi-rank or native-transport correctness from mock
  execution (§3.3).
* It does not introduce a second lineage or a second product schema where one
  exists (#31).
* It does not treat "the tests pass" as evidence for a replay claim: the
  acceptance bar for E2–E4 is **bit-identity of replay**, which a
  distributional check cannot establish.
