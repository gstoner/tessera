---
last_updated: 2026-08-02
audit_role: reference
scope: effects, shape system, fusion region formation, distributed planning, canonicalization, autotuner
companions: AUTODIFF_ARCHITECTURE_REVIEW.md · ../domain/GA_EBM_ARCHITECTURE_REVIEW.md · RIEMANNIAN_OT_PLAN.md
---

# Compiler Architecture Sweep — Where Else to Re-architect

Three reviews (GA/EBM, autodiff, and the OT-plan hardening) surfaced the same
handful of failure shapes over and over. This document names those shapes as a
diagnostic lens, applies them to compiler areas none of the three touched, and
consolidates everything into one ranked queue.

**Status truth stays with the generated dashboards** (Decision #26). Nothing here
reclassifies a row.

**Scope honesty.** Areas examined here: effects, shape system, fusion region
formation, distributed planning, canonicalization, autotuner. Areas **not**
examined and therefore not assessed: the spectral and TPP solver families,
quantization numerics, the evaluator, the KV-cache/memory model, collective
scheduling internals, and layout assignment. Absence from this document is not a
clean bill of health.

---

## 1. The lens — five recurring defect shapes

Extracted from the three prior reviews. Each is a *diagnosable class*, not a
one-off.

| # | Shape | Where it was found before |
|---|---|---|
| **L1** | **Semantic metadata is carried but never consumed.** The information the compiler needs is computed, validated, attached — and then no pass reads it. | `manifold` attr reaches no backend; `MultivectorSpec.grades` discarded by `geometric_product`; `batching_rule` axis closed across 480 primitives while `vmap` is a Python loop |
| **L2** | **Two disconnected compilers.** A Python fast path with hand-written kernels beside an MLIR pass that marks or annotates, with no connection. | GA `rotor_sandwich` (Python symbol vs `RotorSandwichFold` marker); the Apple synthesizer vs the MLIR lane (CLAUDE.md's own framing); the Python AD tape vs `AutodiffPass` |
| **L3** | **Syntactic checks where a real analysis belongs.** A pass pattern-matches on op names or presence instead of computing a dataflow fact. | `CheckpointInnerLoop` (no liveness); `AutodiffPass` (no activity analysis) |
| **L4** | **Fails open on a semantic key.** Absence or ambiguity yields a plausible default instead of an error. | `manifold` → `"euclidean"` with a warning; the first-order-correct Euclidean fallback that converges and lies |
| **L5** | **A constant where a measured decision belongs.** A tradeoff whose right answer is target/shape/dtype-dependent is fixed in a comment. | residual policy `RECOMPUTE_ALL`; `checkpoint_budget = 4`; the GA v1 allow-list welding `ExpandProductTable`'s lowering strategy to `dim ≤ 16` |

---

## 2. Findings in new territory

Ranked by leverage. Each names its shape from §1.

### F1 — The effect lattice walks the **Python AST**, not the IR — and Decision #5 says otherwise  *(L1 + L2 + L4)*

**This is the highest-leverage finding in the sweep.**

[`effects.py:173`](../../../python/tessera/compiler/effects.py:173):

```python
class _EffectVisitor(ast.NodeVisitor):
    def visit_Call(self, node: ast.Call) -> None:
        op_name = self._resolve_call_name(node)      # dotted-name string
        bare = op_name.split(".")[-1]
        if bare in _OP_EFFECTS:
            self._record(op_name, _OP_EFFECTS[bare])
```

Effects are inferred by walking Python source text and matching dotted call names
against a dictionary. The class docstring is candid: *"Phase 1: AST-based
single-function analysis. Phase 2: full inter-procedural dataflow over the Graph
IR call graph."*

**Decision #5 in `CLAUDE.md` states: "Effects are inferred, not declared.
`EffectLattice` walks the IR."** It does not walk the IR. That is a load-bearing
architectural decision documented incorrectly.

It also **fails open**, in exactly the manifold-default shape (L4). Any op reached
through an alias, a local variable, a helper function, a `getattr`, a dict
dispatch, a comprehension over an op table, or any indirection at all is invisible
to a name-matching AST walk — and an invisible op contributes `Effect.pure`. A
function that calls RNG through a wrapper is inferred pure, and
`@jit(deterministic=True)` then *passes*.

And there are **two** effect systems: `EffectAnnotationPass.cpp` (186 lines)
computes effects again on the MLIR side. CLAUDE.md's own collective-insertion
contract — "`GPUCollectiveInsertionPass` must run **after** `EffectAnnotationPass`"
— depends on the MLIR one, while `@jit`'s determinism contract depends on the
Python one. Two mechanisms, no stated relationship, free to disagree (L2).

**Why this ranks first:** effects are the substrate under nearly everything else
recommended across these reviews. AD activity analysis (autodiff review D3) needs
purity. `ActivationRematerializationPass` already gates on
`mlir::isMemoryEffectFree` and treats unmodelled effects as effectful —
*correctly*, but that means its conservatism is doing the work the effect system
should. Fusion legality needs it. Collective ordering needs it. Fixing the layers
above a wrong substrate is building on sand.

### F2 — The shape system is name equality, not a symbolic algebra — and Decision #28 requires more  *(L1)*

[`shape.py:353`](../../../python/tessera/shape.py:353):

```python
def dims_compatible(lhs, rhs):
    if isinstance(lhs, Dim) and isinstance(rhs, Dim):
        return lhs.name == rhs.name or (...both concrete and equal...)
    return str(lhs) == str(rhs)          # ← final fallback
```

`Dim` is `(name, value)`. `DimProduct` flattens nested products but never
normalizes — no sorting, no canonical form. Consequences:

- `H * Dh` and `Dh * H` are **incompatible**, because the fallback compares
  `"H * Dh"` to `"Dh * H"`. Commutativity of multiplication is not modeled.
- No affine expressions: `S + 1`, `S - 1`, `2*S` cannot be represented.
- No divisibility reasoning: `S % tile == 0` — the central question of tiling
  legality — cannot be asked, let alone proved.
- No inequalities, no `min`/`max`, no floor/ceil division.

**This collides directly with Decision #28**, which states the synthesizer/plugin
interface is "symbolic-dim-aware from day one (`static | bucket | dynamic`
policy; first impls bucket-specialize) so dynamic shapes never force an API
break." The arbiter does key on shape buckets — `candidate.py` is built around
`(op, shape-bucket, dtype, target)`. But with a shape system that cannot reason
symbolically, bucket membership can only be decided on *concrete* shapes at
runtime. The API break Decision #28 was written to prevent is latent, not averted.

**The SOTA gap is unusually cheap to close here**, because MLIR ships the answer
in-tree and Tessera already links it: `AffineExpr` for symbolic dimension
arithmetic with a canonical form, and the `presburger` library
(`IntegerRelation` / `FlatAffineValueConstraints`) for divisibility, inequality,
and emptiness queries. PyTorch built `SymInt` on sympy for the same reason.
Tessera hand-rolled string comparison next to a Presburger solver it already
depends on.

Closing it also delivers tiling-legality proofs (`check_schedule_tile` becomes a
constraint query rather than a guess) and symbolic bucket predicates, for free.

### F3 — Fusion is a hand-enumerated region catalog, not a fusion algorithm  *(L2)*

[`fusion_core.py`](../../../python/tessera/compiler/fusion_core.py) defines seven
region classes — `FusedRegion`, `MatmulRegion`, `NormChainRegion`,
`PointwiseGraphRegion`, `PointwiseReduceRegion`, `AttentionRegion`,
`GatedMatmulRegion` — each with its own discovery function
(`discover_pointwise_graph`, …), its own cost function (`fusion_cost`,
`attention_cost`, `attention_lowering_costs`), its own predicate
(`should_fuse_region`, `should_fuse_attention`, `should_fuse_gated`), and its own
emitter in each of `apple_msl.py` / `nvidia_cuda.py` / `rocm_hip.py` / `x86_*.py`.

Adding one fusion shape costs a class, a discovery pass, a cost model, and N
backend emitters. The taxonomy reached seven this way and there is no mechanism
by which it stops growing.

Meanwhile [`TesseraTiling.cpp`](../../../src/compiler/ir/TesseraTiling.cpp)
implements MLIR's **generic** `TilingInterface` on `MatmulOp`, `Conv2DNHWCOp`,
and `FlashAttnOp` — the same ops — and `fusion_core.py` does not use it. Two
fusion/tiling systems, one generic and one enumerated, on the same operators
(L2 again).

**What to keep and what to change.** The cost models are the good part: they are
measured, they feed the arbiter, and they encode real target knowledge — keep
them. The *discovery* is the problem. Region formation should be a generic DAG
partitioner over a legality oracle (producer-consumer fusibility, which is
exactly what F1's effect analysis and F2's shape constraints would supply), with
the existing cost functions scoring the candidate partitions. Then a new fusion
opportunity is **discovered**, not enumerated, and the per-backend emitter is
written once against a region *grammar* rather than once per region *shape*.

### F4 — `DistributedPlan` is a validator named "planner"; there is no sharding propagation and no search  *(L3 + L4)*

[`distributed_planner.py:124`](../../../python/tessera/compiler/distributed_planner.py:124).
The user supplies a `LayerSpec` list with `dp_axis`, `tp_axis`, `weight_sharding`,
and `pp_stage` **already chosen**, per layer. `validate()` then checks that named
axes exist in `mesh_axes`. The pipeline-stage contiguity check is:

```python
if stages != expected:
    # Allow gaps — warn but don't raise
    pass  # could emit a warning here
```

A validation branch that does nothing and admits it in a comment (L4 again — the
third instance of fail-open across these reviews).

More consequentially: a grep for propagation across `sharding.py`,
`distributed/`, and the Distribution passes finds **none**. There is no mechanism
that takes a sharding annotation on a few tensors and propagates it through the
op graph. Every layer must be annotated by hand.

Decisions #16 (ZeRO stage 2) and #17 (1F1B default) are *policies*, and settled
policies are fine. The gap is that there is no **mechanism** to apply a policy
automatically — no propagation, no cost model, no search.

**SOTA reference points, in ascending ambition:** GSPMD/Shardy-style sharding
*propagation* (annotate a handful of tensors, infer the rest — this is the
minimum bar and by far the highest value-per-effort); Alpa's two-level
formulation (ILP for intra-operator sharding, dynamic programming for
inter-operator pipeline slicing); FlexFlow's MCMC search over the parallelism
space. Propagation alone would turn a 300-annotation model into a
3-annotation model.

### F5 — Canonicalization is six greedy patterns for 315 ops  *(L5, latent)*

[`CanonicalizeTesseraIR.cpp`](../../../src/transforms/lib/CanonicalizeTesseraIR.cpp)
is 206 lines with six `matchAndRewrite` implementations, applied by the greedy
pattern rewrite driver. Greedy rewriting is order-dependent by construction: a
rewrite that is locally worthwhile can block a globally better one, and the
outcome depends on pattern benefit numbers and visitation order.

This is the classic problem equality saturation solves — apply all rules to a
congruence-closed e-graph without choosing an order, then extract the best term
under a cost function (TASO/Tensat and the `egg` line established this for
tensor graphs specifically).

**Tessera has an unusual opportunity here, and I want to be precise about both
halves of it.** The standard weakness of equality saturation in a compiler is
that extraction needs a cost model, and analytic cost models are unreliable on
real hardware. Tessera already has a *measured* one — `emit/autotune.py`'s
`measure_latency` + `MeasureCache`, fleet-shared, keyed by target. Saturating a
rewrite space and extracting under **measured** costs is a genuinely novel
combination, not a catch-up.

**But at six rules it is not worth building.** E-graphs are a real engineering
investment whose payoff scales with rule count and rule interaction. The
actionable recommendation now is the shared prerequisite: **make canonicalization
rule-table-driven — rules as data (PDL/PDLL or a declarative table), not
hand-written C++ patterns.** That is worth doing on its own merits at any rule
count, it makes the rule set countable and auditable, and it is the precondition
for saturation later. Revisit extraction strategy when the table is large enough
that ordering demonstrably matters.

### F6 — The autotuner is sound; do not re-architect it

Stated so the list is credible rather than uniformly negative.
[`autotune_v2.py`](../../../python/tessera/compiler/autotune_v2.py) is a real
Optuna TPE sampler with Hyperband pruning, a deterministic grid-search fallback
when Optuna is absent, SQLite warm-start keyed per Decision #11, a legality-aware
candidate generator (`LegalGEMMCandidateGenerator`) with explicit
`CandidateRejection` reasons, and a hook for learned-surrogate training. That is
a well-constructed piece of the compiler and it is not on the list.

The one refinement worth noting, not a re-architecture: the objective is
single-target latency. Cross-target and cross-shape *transfer* — warm-starting
gfx1151 from sm_120 data, or a new shape bucket from adjacent buckets — is where
the next win is, and the surrogate hook is already the right place for it.

---

## 3. The root cause, and the single highest-leverage move

Read F1–F5 together with the three prior reviews and one thing is common to
almost all of them:

> **Tessera has excellent *contracts* and weak *analyses*.**

The contract machinery is genuinely strong — ODS interfaces, the primitive
coverage registry with twelve auto-flipping axes, the backend manifest, the
arbiter's candidate registry, drift-gated generated dashboards, the connection
ledger's separation of forward/backward proof. Very few compilers have this much
discipline about what is *claimed*.

What is thin is the machinery by which a pass **derives** a fact instead of being
**told** one. Look at what is actually missing across all four documents:

| Missing analysis | Consequence found | Document |
|---|---|---|
| Effect/purity on the IR | fails open on determinism; two disagreeing systems | F1 |
| Symbolic shape constraints | no divisibility proofs; latent Decision #28 API break | F2 |
| Differentiation activity | no Enzyme-class adjoint pruning | Autodiff D3 |
| Trajectory liveness / gradient demand | 2500 dead steps marked rematerializable | GA/EBM §1.5 |
| Fusion legality oracle | region discovery hand-enumerated | F3 |
| Sharding propagation | every layer annotated by hand | F4 |
| Grade/structure sparsity | dense `2^n` products on 4-nonzero rotors | GA/EBM §2.1 |

**Every one of these is the same kind of object**: a lattice, a transfer
function, and a fixpoint over the Graph IR. MLIR ships the framework for exactly
this — `DataFlowAnalysis` / `DataFlowSolver` with sparse and dense forward and
backward variants, `AnalysisManager` for caching and invalidation, and
`presburger` for the integer-set queries F2 needs. Tessera has instead grown
seven partial, hand-rolled, mutually-inconsistent substitutes, several of which
fail open.

**The single highest-leverage architectural move is to stand up one Graph IR
dataflow-analysis layer and re-home effects, shapes, activity, liveness, fusion
legality, and sharding onto it.** Not as a big-bang rewrite — as a framework plus
one migrated client (effects, F1), after which each subsequent analysis is a
transfer function rather than a subsystem.

Three properties it must have, each learned from a specific failure above:

1. **Fail closed.** An analysis that cannot prove a fact returns ⊤ (unknown), and
   consumers must treat ⊤ as "not safe," never as the optimistic value. F1's
   invisible-op-is-pure and the manifold default are the same bug.
2. **Recomputable and invalidated.** Effects computed once at decoration time
   from Python source cannot survive fusion, differentiation, or remat — all of
   which change the facts. Analyses must be queries against current IR.
3. **Queryable from both C++ and Python.** The Python layer is the oracle
   (autodiff review M1) and must be able to ask the same questions and get the
   same answers, or L2 regenerates itself.

---

## 4. Consolidated queue

One ranked list across all four documents, so there is a single place to plan
from. Effort figures are engineering estimates for a single track, not
commitments.

### Tier 0 — Live defects and cheap correctness  *(~3 weeks total)*

| # | Item | Source | Effort |
|---|---|---|---|
| 1 | Route EBM `grad_fn=None` through `autodiff.tape` instead of `O(2^n)` finite differences | GA/EBM §2.6 | 2d |
| 2 | `manifold` → required verified enum; delete the Euclidean default (copy `AnnotateAlgebra`) | GA/EBM §1.1 · OT §H1 | 3d |
| 3 | Demand-gate `CheckpointInnerLoop` + `CHECK-NOT` fixtures + annotated-count assertion | GA/EBM §1.5 · OT §H2 | 4d |
| 4 | **Correct Decision #5 in `CLAUDE.md`** — the effect lattice walks the AST, not the IR | F1 | 1h |
| 5 | Fix `jacrev`/`jacfwd` forward-pass-per-element; route `vmap` through the batching-rule registry | Autodiff §B1–B3 | 1w |
| 6 | `.td` summary drift; distinguish "stub" from "annotation-only"; remove the false "GA8 will refuse" promise | GA/EBM §1.4 | 1d |
| 7 | Adopt Decisions #21a (semantic keys never default) and #10a (eligibility passes ship a negative fixture) | OT §4a | — |

### Tier 1 — The analysis layer  *(~10 weeks)*

| # | Item | Source | Effort |
|---|---|---|---|
| 8 | Stand up the Graph IR dataflow-analysis framework (MLIR `DataFlowSolver`, fail-closed, invalidated, Python-queryable) | §3 | 3w |
| 9 | Re-home effects onto it; reconcile with `EffectAnnotationPass`; retire the AST walker | F1 | 2w |
| 10 | Replace `Dim`/`DimProduct` with `AffineExpr` + `presburger`; make `dims_compatible` and `check_schedule_tile` constraint queries | F2 | 3w |
| 11 | Activity analysis for AD, as a client of #8 | Autodiff D3 | 2w |

### Tier 2 — Differentiation capability  *(~13 weeks, critical path)*

| # | Item | Source | Effort |
|---|---|---|---|
| 12 | Forward mode in the compiler (`TangentInterface`) | Autodiff D2 | 3w |
| 13 | Structured control-flow adjoints (`scf.*`, `tessera.control_*`) | Autodiff D4 | 6w |
| 14 | Residual policy as a measured arbiter axis + Revolve for loops; delete `EBMCheckpointInnerLoop` | Autodiff D5 | 4w |
| 15 | Finish `NewtonAutodiff` IFT body — shared with OT R2, budget once | Autodiff §B8 · OT R2 | 2w |

### Tier 3 — Generic mechanisms replacing enumerations  *(~11 weeks)*

| # | Item | Source | Effort |
|---|---|---|---|
| 16 | Sharding **propagation** (GSPMD/Shardy-style) — minimum bar, highest value-per-effort in distribution | F4 | 4w |
| 17 | Generic fusion region discovery over a legality oracle; keep the measured cost models | F3 | 4w |
| 18 | Rule-table-driven canonicalization (PDL/PDLL); defer saturation until the table is large | F5 | 3w |
| 19 | Thread `MultivectorSpec.grades` into `geometric_product`; add `input_grades` to `GradeFusion` | GA/EBM §2.1 | 1w |

### Tier 4 — Exceed the state of the art  *(~9 weeks)*

| # | Item | Source | Effort |
|---|---|---|---|
| 20 | Sparse AD — sparsity detection + coloring (a client of #8/#10) | Autodiff D7 | 5w |
| 21 | Taylor/jet mode hosted on the GA multivector engine | Autodiff D6 | 4w |
| 22 | Table-driven GA kernel synthesis via `emit/`; then PGA `Cl(3,0,1)` | GA/EBM §2.3–2.4 | 5w |

**If only one tier happens: Tier 0 plus item 8.** Tier 0 closes three live
defects and one wrong architectural decision in ~3 weeks; item 8 is what makes
everything after it cheaper instead of being the eighth hand-rolled analysis.

---

## 5. What not to do

- **Do not re-architect the autotuner.** It is sound (F6). Invest in cross-target
  transfer via the existing surrogate hook, not in replacing it.
- **Do not build e-graphs yet.** Six rules do not justify equality saturation.
  Make rules data first (F5); revisit when ordering demonstrably costs something.
- **Do not build an eighth hand-rolled analysis.** If a new pass needs a derived
  fact, that is a signal to land item 8, not to add another bespoke walker. This
  is the same warning `AUTODIFF_UNIFICATION_PLAN.md` §2a gives about parallel
  systems, generalized past autodiff.
- **Do not confuse settled policy with missing mechanism.** ZeRO-2 (#16) and
  1F1B (#17) are decided and should stay decided. F4 is not an argument to
  revisit them — it is that there is no machinery to *apply* them without
  hand-annotation.
- **Do not treat this document as coverage.** Six areas were examined; the
  solvers, quantization, evaluator, memory model, collective scheduling, and
  layout assignment were not.
