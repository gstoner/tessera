---
last_updated: 2026-08-02
audit_role: reference
scope: python/tessera/autodiff, src/compiler/ir/AdjointInterface.*, src/transforms/lib/Autodiff*.cpp, ActivationRematerializationPass, AdjointCollectiveInsertionPass, solvers/core NewtonAutodiff
companions: AUTODIFF_UNIFICATION_PLAN.md (sequencing) · ../../spec/AUTODIFF_SPEC.md (normative surface) · RIEMANNIAN_OT_PLAN.md · ../domain/GA_EBM_ARCHITECTURE_REVIEW.md
---

# Autodiff Architecture and Algorithm Review

A capability review of Tessera's differentiation surface against (a) the
MLIR/LLVM spine the compiler is now committed to, and (b) the published state of
the art. It **complements** [`AUTODIFF_UNIFICATION_PLAN.md`](AUTODIFF_UNIFICATION_PLAN.md),
which is right about direction and sequencing and which this review does not
replace. That plan answers *"how do we stop reporting the Python tape as
compiler support?"* This one answers a different question: *"once differentiation
is a compiler request, what must the compiler be able to differentiate, and
where can Tessera be better than everyone else?"*

**Status truth stays with the generated dashboards** (Decision #26). Counts
quoted below are read from
[`generated/autodiff_connection_ledger.md`](../generated/autodiff_connection_ledger.md),
not asserted here.

---

## 0. What exists today

| Layer | Surface | Size |
|---|---|---|
| Python engine + reference | `autodiff/{tape,grad,transforms,vjp,jvp,mixed_precision,rematerialize}.py`, `autodiff/geometric/` | ~11.7k lines (`vjp.py` 5.4k, `jvp.py` 3.8k) |
| Graph IR adjoints | `AdjointInterface.{td,cpp}` | 927 lines |
| Reverse-mode passes | `AutodiffPass.cpp` (in-place), `AutodiffPairedPass.cpp` (paired ABI) | 316 + 285 |
| Remat | `ActivationRematerializationPass.cpp` | 624 |
| Distributed adjoints | `AdjointCollectiveInsertionPass.cpp` | 269 |
| Implicit-op derivatives | `solvers/core/passes/NewtonAutodiff.cpp` | 87 (annotation-only) |

From the generated ledger: **287** differentiable families tracked, **287** with
a Python VJP/JVP reference, **32** with `ir_adjoint = native`, **3**
`placeholder` (runtime round-trip into Python), **23** with backward IR
oracle-verified on CPU, **29** with a backward `device_verified_jit` lane on some
exact target.

Read those five numbers together and the shape is clear: the Python reference is
broad; the compiler differentiates a small, deliberately-chosen core; and most
device-verified backward lanes are **hand-written Tier-3 kernels satisfying the
paired ABI** (flash-attn, GQA/MQA, selective SSM, KDA), not `AutodiffPass`
output. The ledger says this explicitly and is right to.

---

## 1. Architectural findings

### A1. The Python tape is global monkey-patching, not a program transform

[`tape.py:497`](../../../python/tessera/autodiff/tape.py#L497):

```python
wrapped = _make_wrapper(name, original)
setattr(ops, name, wrapped)
ops.registry._entries[name].reference = wrapped
```

`install_op_wrappers()` rebinds every name in the `tessera.ops` module and
mutates the op registry in place. It is process-wide, irreversible, not
thread-safe, and not re-entrant. "Being differentiated" is a global process
state rather than a property of a value or a program.

This is not a style objection — it is the reason the Python layer *cannot* be
promoted into the compiler and must be demoted to oracle. The unification plan
already concludes "oracle only"; A1 is the mechanical reason that conclusion is
forced rather than optional.

**Contrast.** JAX traces to a jaxpr — a value-level IR — and every transform
(`grad`, `vmap`, `jit`) is a function on that IR. Enzyme is a transform on LLVM
or MLIR. Neither mutates a module namespace to enable differentiation.

### A2. The tape is identity-keyed, which structurally forbids higher-order AD

`TapeEntry.output_id = id(output)`; `Tape.cotangent: dict[int, np.ndarray]`.
Gradients are looked up by Python object identity.

Three consequences follow directly:

1. Only values produced by a recorded `ops.*` call can be differentiated —
   `backward()` raises "backward target is not a tape-recorded output" otherwise.
2. There is no value-level gradient composition: `grad(f)` returns raw NumPy,
   detached from any tape.
3. **Therefore `grad(grad(f))` cannot work.** Higher-order differentiation is
   blocked by the data structure, not by missing rules.

That is why `hvp` is finite differences (§B4) despite a 3.8k-line JVP engine
sitting next to it. The rules exist; the substrate cannot compose them.

### A3. The compiler's reverse pass rejects all control flow — by design

[`AutodiffPass.cpp:118`](../../../src/transforms/lib/AutodiffPass.cpp#L118):

```cpp
if (op->getNumRegions() != 0) {
  op->emitError() << "[AUTODIFF_NESTED_REGION] reverse-mode autodiff does "
                     "not yet support ops with nested regions ('" ...
  signalPassFailure();
```

The pass collects only `func.getBody().front()` top-level ops and hard-fails on
anything with a region. So `scf.for`, `scf.if`, `scf.while`, and Tessera's own
`tessera.control_{for,while,scan}` (which exist in `TesseraOps.td:2633+`) are
all undifferentiable in the compiler.

The comment is honest about why — reverse-iterating a flat nested walk would
interleave parent and child adjoints — and the restriction was correct as a
bootstrap. But loops are the *body* of nearly every workload this compiler
targets: SSM and linear-attention scans, diffusion samplers, solver iterations,
the EBM Langevin loop, the RNOT `c`-transform loop. **Straight-line-only reverse
mode differentiates the parts of a model that were never the problem.**

**Contrast.** LAGrad (CC 2023) is a reverse-mode MLIR AD system whose stated
contribution is exploiting "the sparsity and structured control flow" of
high-level dialects. Enzyme handles arbitrary LLVM control flow with its
cache-and-analysis approach. Structured control flow is where MLIR-level AD is
*supposed* to have the advantage, because the loop structure is still visible.

### A4. There is no forward mode in the compiler at all

Verified by grep across `src/` for genuine forward-mode markers (excluding the
word "cotangent"): the only hits are `NewtonAutodiff.cpp`'s annotation strings.
There is no `TangentInterface`, no forward-mode pass, no dual-number lowering,
no `--tessera-autodiff-forward`. JVP exists only as 3.8k lines of Python.

This blocks, in one stroke: compiler HVP, `jacfwd`, Taylor/jet mode, any
forward-over-reverse composition, and tangent-space ops for the manifold work in
the [OT plan](RIEMANNIAN_OT_PLAN.md).

It is also the cheapest large win available, because **forward mode is the mode a
tile compiler is best at**: no tape, no residual liveness pressure, no
forward/backward boundary, and tangent computations fuse into the primal loop
nest trivially. Reverse mode is the hard one and it is the one that got built
first.

### A5. No activity analysis

`AutodiffPass` walks every top-level op in reverse and builds an adjoint for
each, then discovers at the end which arguments received cotangents. Nothing
asks, up front, *which values are differentially active*.

Activity analysis is the core of Enzyme's performance story — differentiating
*optimized* IR with activity information yields a **4.5× geometric-mean speedup**
over differentiating unoptimized IR on ADBench, and Enzyme "allocates memory to
store only the values needed by the reverse pass."

The pointed observation: **Tessera has strictly more information available for
this analysis than Enzyme does.** The effect lattice (Decision #5), region
privileges with read/write/reduce modes (Decision #2), static shapes, and
declared dtypes are all present at Graph IR and all gone by the time LLVM IR
exists. An activity analysis here should be *better* than Enzyme's, not absent.

### A6. The residual policy is one global constant, and it is not measured

[`AutodiffPairedPass.cpp`](../../../src/transforms/lib/AutodiffPairedPass.cpp)
header:

> Residual policy — RECOMPUTE_ALL (first cut). The backward function takes the
> forward *inputs* as arguments and recomputes any forward intermediates it needs

The doc is honest that a SAVE policy is "a future optimization the same ABI
already accommodates" and correctly notes the shipped ROCm flash-attn backward
recomputes softmax rather than saving the logsumexp. But the choice is made once,
in a comment, for every op on every target.

**This is a missed fit with the compiler's own north star.** Decision #28 exists
to pick, by *measurement*, among candidates per `(op, shape-bucket, dtype,
target)`. Save-vs-recompute is exactly that kind of choice: it is a
memory/FLOP tradeoff whose right answer is target-, shape-, and
dtype-dependent (on a bandwidth-bound Apple GPU, recompute usually wins; on
sm_120 with 100 KB of shared memory and a compute-bound GEMM, saving usually
wins). The residual policy should be an **arbiter axis**, and today it is a
constant.

**Contrast.** JAX exposes `jax.checkpoint(policy=...)` with saveable-value
predicates (`dots_saveable`, `checkpoint_dots_with_no_batch_dims`, …). Enzyme
decides per-value from analysis. Tessera has better machinery than either for
making this decision empirically and does not use it.

### A7. `custom_adjoint_call` puts a Python round-trip inside compiled IR

Three families (`log_softmax`, `sin`, `softplus`) have `ir_adjoint = placeholder`:
`buildAdjoint` emits a `custom_adjoint_call` that resolves against the Python VJP
registry **at runtime**. The ledger classifies these correctly as not-native.

The concern is directional. Decision #23 makes Tessera a standalone compiler; a
compiled artifact whose backward pass calls into a Python registry is not a
standalone artifact. The escape hatch is fine as a development affordance and
must not survive into a `native_required` compile — the unification plan's P2/P4
already say "defined runtime ABI or rejected," and that gate should be enforced
sooner rather than later, while the count is three.

### A8. Two remat implementations with opposite discipline

`ActivationRematerializationPass` is **good work** and worth naming as such: it
is liveness-aware, budget-driven (`--memory-budget-mb` /
`tessera.remat_budget_mb`), cost-model-aware (`tessera.remat_cost_ns`), refuses
effectful ops with a named diagnostic (`REMAT_EFFECTFUL`) rather than skipping
silently, treats unmodelled effects as effectful (conservative and correct), and
carries an explicit dominance argument for clone placement.

`EBMCheckpointInnerLoop` — reviewed in
[`GA_EBM_ARCHITECTURE_REVIEW.md §1.5`](../domain/GA_EBM_ARCHITECTURE_REVIEW.md) —
is purely syntactic with a hardcoded budget and no liveness analysis at all.

Same repository, same concept, opposite rigor. The domain pass should be deleted
and its op knowledge registered into the general one. Naming this here because it
is an *autodiff* policy question, not an EBM one.

---

## 2. Algorithmic findings

### B1. `jacrev` re-runs the forward pass once per output element

[`transforms.py:132`](../../../python/tessera/autodiff/transforms.py#L132). The
docstring says it "uses `retain_graph=True` … so the inner tape can be
backward'd repeatedly." The code does not do that:

```python
for k in range(out_size):
    cotan = ...one-hot...
    def loss_fn(*inner_args):
        y = fn(*inner_args, **kwargs)        # ← full forward, every iteration
        return _ops.reduce(_ops.mul(y, cotan), op="sum")
    grad_fn = grad(loss_fn, argnums=argnums_tuple)
    grads = grad_fn(*args)                    # ← builds a fresh tape, every iteration
```

A fresh `grad` closure is constructed and called inside the loop, so `fn` is
re-executed for every output element. Cost is `O(out_size) × (forward + backward)`
where the documented algorithm is `1 × forward + O(out_size) × backward`, and the
vectorized algorithm is `1 × forward + 1 × batched-backward`.

Three levels, and the implementation is on the worst one. This also means the
docstring is wrong, which matters because `jacrev` is an oracle for other things.

### B2. `jacfwd` has the same defect in the input dimension

`O(in_size)` separate `jvp` calls, each re-running the forward through
`fn_of_one`. Same three-level analysis, same bottom rung.

### B3. `vmap` is a Python `for` loop, and it ignores the batching rules that exist

Its own docstring: *"Implementation: scan-then-stack."* The body slices each
batched argument with `np.take`, calls `fn` once per element, and `np.stack`s the
results.

Meanwhile `primitive_coverage.py` carries a `batching_rule` axis that
`MASTER_AUDIT.md` reports as **closed across 480 primitives**. The batching rules
exist as an audited contract and `vmap` does not consult a single one.

The practical cost: `vmap(grad(f))` — per-example gradients, the single most
common composed transform in modern training (differential privacy, influence
functions, per-sample clipping, meta-learning) — is a Python loop over examples.

**Contrast.** In JAX, `vmap` is a program transform on the jaxpr driven by
per-primitive batching rules, and `vmap(grad(f))` compiles to one batched kernel.

### B4. `hvp` is central finite differences of `grad`

[`grad.py:120`](../../../python/tessera/autodiff/grad.py#L120), and its docstring
says so plainly:

```
hvp(f, x, v) ≈ (∇f(x + ε v) - ∇f(x - ε v)) / (2 ε)
```

Two full gradient evaluations, `ε = 1e-4` hardcoded, `O(ε)` truncation error
compounding with roundoff. Exact forward-over-reverse HVP costs roughly 2× a
single gradient and is exact. Blocked by A2 and A4, not by missing rules.

This matters more than it looks: HVP is the primitive under L-BFGS, natural
gradient, K-FAC, Gauss–Newton, trust-region methods, GAN gradient penalties, and
sharpness-aware minimization. A finite-difference HVP quietly caps the quality of
every one of them.

### B5. No sparsity exploitation anywhere in the stack

No Jacobian sparsity detection, no matrix coloring, no structured-Jacobian
representation, in either the Python layer or the compiler.

This is the clearest place Tessera can *exceed* the state of the art rather than
catch up. The 2025 automatic-sparse-differentiation line
([ICLR 2025 blogpost](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/),
["Sparser, Better, Faster, Stronger"](https://arxiv.org/pdf/2501.17737))
observes that **PyTorch, TensorFlow, and JAX all lack sparsity detection and
coloring**; the one JAX library that exists (`sparsejac`) has no detection, uses
weaker graph encodings, and cannot handle symmetric Hessians.

Tessera has better raw material than any of them for this: static shapes at Graph
IR, an effect lattice, region privileges that already express read/write
disjointness, and a tile IR where a colored Jacobian's structure maps directly
onto tile scheduling. Sparsity detection is a dataflow analysis over exactly the
information Tessera already computes.

### B6. Checkpointing is greedy-interval, not Revolve — and cannot touch loops

`ActivationRematerializationPass` performs deterministic liveness-aware selection
of the largest long-lived pure activation intervals until the estimated peak fits
the budget. That is a sound greedy heuristic for a straight-line block.

It is not the algorithm the literature settles on for loops. Revolve/treeverse
binomial checkpointing gives **provably optimal** schedules with `O(log T)` memory
and `O(log T)` recompute for a `T`-step loop. And per A3, the pass cannot act
inside a loop at all (`REMAT_NON_CLONABLE` on nested regions).

So the workloads with the worst activation-memory profile — long scans, diffusion
trajectories, the `T = 2500` RNOT inner loop — get *no* checkpointing, while
straight-line blocks that were never the bottleneck get a good greedy schedule.

### B7. Adjoints of collectives are real, and are a genuine lead worth naming

[`AdjointInterface.cpp:42-70`](../../../src/compiler/ir/AdjointInterface.cpp#L42):
`AllReduce` is self-dual; `AllGather† = ReduceScatter`; `ReduceScatter† =
AllGather` — correct, and `AdjointCollectiveInsertionPass` places them
effect-aware, after `EffectAnnotationPass`, keyed on `tessera.effect = "memory"`.

Most frameworks bolt distribution on *outside* AD (DDP hooks, FSDP wrappers).
Having collective adjoints as first-class ops inside the differentiation
interface is ahead of the common practice, and it should be stated as a lead
rather than buried.

### B8. Implicit-function-theorem differentiation is scaffolded, not absent

`NewtonAutodiff.cpp` walks `tessera_solver.implicit` ops and annotates
`tessera_solver.{vjp,jvp} = "generated"`. Its header states the intended
decomposition — `dF/dx = -(dR/dx)⁻¹ · dR/du` — and admits the body only
annotates and structurally decomposes; values resolve "at runtime via the
registered vjp/jvp kernels."

**This corrects a claim in the [OT plan](RIEMANNIAN_OT_PLAN.md) §3.2.** The
implicit-diff seam is not missing; it is a stub in the solver dialect with the
right shape and the right formula in its header. R2's `custom_root` should
*finish that pass*, not introduce a parallel mechanism. Same correction applies
to the RNOT Jacobian requirement — App. F.3's
`J = −[D_yF]⁻¹[D_xF]` is literally the formula in the header comment.

---

## 3. Position against the state of the art

Honest, per-capability. "Partial" is used where a real but incomplete
implementation exists.

| Capability | Tessera | JAX | Enzyme / EnzymeMLIR | LAGrad |
|---|---|---|---|---|
| Reverse mode, straight-line | ✅ 32 native families | ✅ | ✅ | ✅ |
| Reverse mode through structured control flow | ❌ hard error (A3) | ✅ | ✅ | ✅ (its thesis) |
| Forward mode in the compiler | ❌ Python only (A4) | ✅ | ✅ | partial |
| Higher-order (`grad∘grad`) | ❌ structurally blocked (A2) | ✅ | ✅ | — |
| Exact HVP | ❌ finite differences (B4) | ✅ fwd-over-rev | ✅ | — |
| `vmap` as a transform | ❌ Python loop (B3) | ✅ | n/a | n/a |
| Activity analysis | ❌ (A5) | partial | ✅ (its edge) | ✅ |
| AD after optimization | partial — Graph IR only | via XLA | ✅ (4.5× geomean) | ✅ |
| Residual policy, per-target measured | ❌ one constant (A6) | ✅ policies | ✅ analysis | partial |
| Revolve / binomial checkpointing | ❌ greedy, no loops (B6) | partial (manual) | partial | — |
| Sparsity detection + coloring | ❌ (B5) | ❌ | ❌ | partial (static) |
| Collective adjoints as IR ops | ✅ **lead** (B7) | partial (outside AD) | ❌ | ❌ |
| Manifold / geometric AD | partial (`autodiff/geometric/`) | ❌ | ❌ | ❌ |
| Per-target device-verified backward proof | ✅ **lead** — ledger axes | ❌ | ❌ | ❌ |

Two real leads (collective adjoints; per-target backward *proof* discipline, which
no other system even tracks), one partial lead (geometric AD), and six genuine
gaps.

---

## 4. Three architectural moves

### M1 — Differentiation is a Graph IR → Graph IR transform; the Python tape is the oracle, permanently

This is the unification plan's existing direction. The contribution of this
review is the *mechanism* argument: A1 and A2 show the tape cannot be
incrementally upgraded into a compiler transform. Global monkey-patching and
identity-keyed cotangents are not implementation details that can be refactored
away — they are the architecture. Any effort spent making the tape more capable
(higher-order, real `vmap`) is spent on the wrong layer.

The corollary is a scope reduction, which is good news: the Python layer needs
only to be **correct and fast enough to be a trustworthy oracle**. §D1 is small
precisely because of this.

### M2 — Differentiate late, on *structured* IR — take Enzyme's lesson, not Enzyme's level

Enzyme's central empirical result is that differentiating **optimized** IR beats
differentiating unoptimized IR by 4.5× geomean, because AD-then-optimize destroys
the structure the optimizer needed.

The lesson generalizes; the level does not. By LLVM IR, everything that makes
Tessera Tessera — tiles, memory spaces, precision policy, layouts, region
privileges, collectives — has been erased into loads, stores, and scalar math. An
AD pass at that level would be differentiating a program that has already thrown
away the information needed to generate a good backward *kernel*. This is the
same trap Decision #26a documents for the AIR question: the instinct to go lower
paid less than expected once measured.

So: **differentiate at Graph IR, after canonicalization and fusion-region
formation, before Schedule/Tile lowering.** LAGrad is the precedent that this
level works. Two contracts follow:

- `AutodiffPass` must run *after* the semantics-preserving Graph IR
  optimizations, not before — which is a pipeline-ordering decision to make
  explicitly and test, not to inherit by accident.
- The generated backward must remain **one arbiter candidate among several**
  (Decision #28). A hand-written ROCm WMMA flash-attn backward and a
  compiler-generated one satisfy the same `@f__bwd` ABI; the arbiter picks by
  measurement. `AutodiffPairedPass` already frames it this way and is right to.

### M3 — Differentiation is an interface *family*, not one pass

Today there is one `AdjointInterface` and one reverse pass. The capabilities in
§1–2 each need their own op-level contract:

| Interface | Question it answers | Enables |
|---|---|---|
| `AdjointInterface` *(exists)* | what is the VJP? | reverse mode |
| `TangentInterface` | what is the JVP? | forward mode, HVP, Taylor (D2, D6) |
| `ActivityInterface` | is this operand differentially active? | activity analysis (D3) |
| `ResidualInterface` | what must be saved vs recomputed, per target? | measured residual policy (D5) |
| `SparsityInterface` | what is this op's Jacobian structure? | sparse AD (D7) |
| `RegionAdjointInterface` | how does this region-carrying op differentiate? | control flow (D4) |

This is EnzymeMLIR's own design point — "operations and types implement or
inherit general interfaces to specify their differentiable behavior" — and it is
how MLIR scales a cross-cutting concern across 287 op families without one pass
growing to know all of them. It also fits Decision #24 cleanly: each interface is
a new axis in `primitive_coverage.py`, auto-flipping the same way `vjp`/`jvp`
already do.

---

## 5. Plan — D1 … D7

These are **capability additions interleaved with** `AUTODIFF_UNIFICATION_PLAN.md`'s
P-phases, not a replacement. That plan's P1–P6 make differentiation a truthful,
proven compiler request; D1–D7 expand what can be requested. Mapping is given per
phase.

### D1 — Make the oracle trustworthy and cheap  *(~1 week)*

Fix the three algorithmic defects in the Python reference. Everything else in
this plan is verified against it, so it must be correct and not absurdly slow.

- `jacrev`: hoist the forward pass out of the loop, use the `retain_graph`
  contract the docstring already promises. Fix the docstring either way.
- `jacfwd`: same, in the input dimension.
- `hvp`: keep the finite-difference path as a fallback, add exact
  forward-over-reverse once D2 lands; until then, document the accuracy bound at
  the call site rather than only in the docstring.
- `vmap`: route through the `batching_rule` registry that already exists and is
  audited. If a primitive has no batching rule, fall back to the loop **with a
  diagnostic** rather than silently.

Maps to: unification-plan P0 (truthfulness). Independent of everything else.

### D2 — Forward mode in the compiler  *(~3 weeks)*

`TangentInterface` in ODS + `--tessera-autodiff-forward` + `buildTangent` for the
32 families that already have `buildAdjoint`. Forward mode needs no tape, no
residual policy, and no liveness analysis, so it is a fraction of the reverse-mode
work.

Immediately unlocks: compiler `jacfwd`, exact HVP via forward-over-reverse,
tangent-space ops for the [manifold work](RIEMANNIAN_OT_PLAN.md), and the
substrate for D6.

Maps to: P5 (family expansion), running in parallel.

### D3 — Activity analysis  *(~3 weeks)*

`ActivityInterface` + a dataflow analysis over the effect lattice and region
privileges. Compute the active set forward from the differentiated inputs and
backward from the seeded outputs; intersect; skip adjoint construction outside it.

This is where the Enzyme-class speedup lives, and where Tessera's extra
information (effects, privileges, static shapes) should let it do better than a
system working on LLVM IR.

Exit criterion worth stating: a fixture where an inactive branch's adjoint is
**not emitted**, checked with `CHECK-NOT` — the same negative-fixture discipline
proposed as Decision #10a in the [OT plan](RIEMANNIAN_OT_PLAN.md) §4a.

Maps to: P2 (contract hardening).

### D4 — Structured control-flow adjoints  *(~6 weeks — hardest, highest value)*

`RegionAdjointInterface` and reverse-mode over `scf.for` / `scf.if` / `scf.while`
and `tessera.control_{for,while,scan}`. The standard construction: reverse a
counted loop by running the adjoint loop backward over a saved or recomputed
trajectory; reverse `if` by taping the predicate; reverse `while` by taping the
trip count.

Everything with a loop is blocked on this: SSM/linear-attention scans, diffusion
samplers, solver iterations, the EBM Langevin loop, the RNOT `c`-transform. It is
also the precondition for D5's Revolve.

Maps to: P5, and it is the correct next big rock after P4.

### D5 — Residual policy as a measured arbiter axis, plus Revolve  *(~4 weeks, after D4)*

- Promote `tessera.autodiff.residual_policy` from a constant to a decision the
  arbiter makes per `(op, shape-bucket, dtype, target)`, with SAVE / RECOMPUTE /
  HYBRID candidates, measured — exactly the Decision #28 mechanism, applied to a
  choice it was built for.
- Add Revolve/treeverse binomial checkpointing for counted loops, replacing "no
  checkpointing at all" (B6) with a provably optimal `O(log T)` schedule.
- Delete `EBMCheckpointInnerLoop`; register its op knowledge into
  `ActivationRematerializationPass` (A8).

Maps to: P4/P6.

### D6 — Higher-order, hosted on the geometric-algebra engine  *(~4 weeks — the differentiating move)*

Two steps:

1. Exact forward-over-reverse HVP, once D2 exists.
2. **Taylor / jet mode over Weil algebras — implemented on the existing
   multivector engine.**

Step 2 deserves emphasis, because it is a synergy nobody else has. Taylor-mode AD
computes all mixed partials to order `k` in a single forward pass at cost linear
in the algebra dimension, by carrying values in a **truncated polynomial (Weil)
algebra** instead of ℝ. A Weil algebra is a finite-dimensional commutative
algebra with a compile-time-known multiplication table and a nilpotent grading.

Tessera already has *exactly that object*:
[`ga/signature.py`](../../../python/tessera/ga/signature.py) builds a
compile-time-cached, graded, bitmask-indexed product table from a signature and
caches it per algebra, and
[`ExpandProductTable.cpp`](../../../src/solvers/clifford/lib/Passes/ExpandProductTable.cpp)
lowers a product table to unrolled IR. The GA track and the higher-order-AD track
are the same machine with different structure constants.

If the GA review's item 7 (table-driven kernel synthesis via `emit/`) lands
first, Taylor mode arrives largely for free — and the grade-sparsity work
(GA review §2.1) is *literally* the truncation-order sparsity Taylor mode needs.
No other AD system has a production graded-algebra kernel generator to host this
on.

Maps to: new capability; no existing P-phase covers it.

### D7 — Sparse AD: detection and coloring  *(~5 weeks — the SOTA-exceeding move)*

- `SparsityInterface` giving each op's Jacobian sparsity pattern.
- A sparsity-propagation dataflow analysis over Graph IR (static shapes make this
  tractable in a way it is not for eager frameworks).
- Greedy distance-1 / star coloring for Jacobians and Hessians respectively, then
  compress the seed matrix so `jacrev`/`jacfwd` need `O(colors)` passes instead
  of `O(rows)` or `O(cols)`.
- Map colored compression onto tile scheduling — this is the part only a tile
  compiler can do well.

Because PyTorch, TensorFlow, and JAX all lack this, a working implementation is a
defensible "exceeds the state of the art" claim rather than a catch-up.

Maps to: new capability.

### Alongside — finish `NewtonAutodiff`

Implement the IFT body the header already specifies (`dF/dx = -(dR/dx)⁻¹ dR/du`),
emitting real `tessera_solver.residual` + `linear_solve` ops instead of
annotations. Shared deliverable with [OT plan](RIEMANNIAN_OT_PLAN.md) R2 — the
two tracks should not build this twice. ~2 weeks.

### Sequencing

```
D1 (1w) ──────────────────────────────────────────────────►  (independent)
        D2 (3w) ──┬─► D6 (4w)          [needs GA item 7 for the cheap path]
                  └─► D5 (4w)
D3 (3w) ──────────┐
D4 (6w) ──────────┴─► D5
                      D7 (5w)          [needs D3's dataflow substrate]
NewtonAutodiff (2w)   [shared with OT-plan R2]
```

Critical path D2 → D4 → D5 ≈ 13 weeks; full set ≈ 28 weeks single-track.

**If only three things happen: D1, D2, D4.** D1 makes the oracle honest, D2 is
the cheapest large capability, D4 unblocks every workload with a loop.

---

## 6. What would let Tessera honestly claim to exceed the state of the art

Four claims that would be true and defensible, with what each requires:

1. **"Sparsity-aware AD that PyTorch, TensorFlow, and JAX do not have."**
   Requires D7 plus a benchmark on a genuinely sparse Jacobian (a stencil, a
   graph network, a structured attention mask) showing `O(colors)` scaling.
2. **"Higher-order AD on a graded-algebra kernel generator."** Requires D6 plus
   the GA synthesizer. The claim is not "we have Taylor mode" — several systems
   do — it is that the Taylor algebra and the Clifford algebra are one code
   generator, so order-`k` derivatives get the same tuned kernels as
   `simdgroup_matrix` GA products.
3. **"Collective adjoints are IR ops with per-target device-verified proof."**
   Mostly already true (B7 + the ledger). Requires finishing the backward
   distributed lane and stating it plainly, which no other system tracks at all.
4. **"Residual policy chosen by measurement per target, not by convention."**
   Requires D5. Every other system either exposes a policy knob (JAX) or infers
   from static analysis (Enzyme); none *measures* on the actual target and picks.

Claims 1 and 4 are the strongest because they are things the incumbents have
structurally chosen not to do, not things they have done better.

---

## 7. Risks and explicit non-goals

- **Do not build a fourth AD system.** The unification plan's own warning
  (§2a) applies with full force to D1–D7: every item above extends an existing
  surface (`AdjointInterface`, `ActivationRematerializationPass`,
  `NewtonAutodiff`, `primitive_coverage`) rather than forking one.
- **Do not adopt LLVM-IR-level AD or integrate Enzyme as the engine.** Take the
  lesson (§M2), not the level. At LLVM IR the tile, layout, precision, and
  collective structure that Tessera exists to reason about is gone, and Decision
  #26a already documents what happened last time the "go lower" instinct was
  measured. Enzyme remains a valuable *reference and differential oracle*.
- **Do not let `custom_adjoint_call` normalize.** Three families today. Gate it
  out of `native_required` compiles before the number grows (A7).
- **Do not report `device_verified` backward lanes as compiler AD.** The ledger
  already keeps these axes separate and must keep doing so: 29 device-verified
  backward lanes and 32 native IR adjoints are different, partly disjoint facts.
- **Effort figures are engineering estimates** for a single track with no
  hardware-gated dependencies, not commitments. D4 is the one most likely to
  exceed its estimate.

---

## Sources

- [Enzyme: high-performance AD of LLVM and MLIR](https://github.com/EnzymeAD/Enzyme) · [enzyme.mit.edu](https://enzyme.mit.edu/) — AD on optimized IR, 4.5× geomean on ADBench, activity analysis, EnzymeMLIR op interfaces
- ["Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients"](https://arxiv.org/pdf/2010.01709) — the post-optimization AD result
- [LAGrad: Statically Optimized Differentiable Programming in MLIR (CC 2023)](https://dl.acm.org/doi/10.1145/3578360.3580259) — MLIR-level reverse mode exploiting high-level dialect semantics, sparsity, and structured control flow
- [An Illustrated Guide to Automatic Sparse Differentiation (ICLR 2025 blogpost)](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) and ["Sparser, Better, Faster, Stronger"](https://arxiv.org/pdf/2501.17737) — sparsity detection + coloring; the observation that PyTorch/TF/JAX lack it
- [Gradient checkpointing with `jax.checkpoint` / `jax.remat`](https://docs.jax.dev/en/latest/gradient-checkpointing.html) — saveable-value remat policies
- [Treeverse / Revolve checkpointing](https://www.researchgate.net/publication/2290186_Treeverse_An_Implementation_of_Checkpointing_for_the_Reverse_or_Adjoint_Mode_of_Computational_Differentiation) and [divide-and-conquer checkpointing with no user annotation](https://openreview.net/forum?id=BkYYXJ9i-) — provably optimal binomial schedules
- [Jet functors and Weil algebras in AD](https://arxiv.org/pdf/2510.14342) and [Collapsing Taylor Mode AD](https://arxiv.org/pdf/2505.13644) — higher-order AD over truncated polynomial algebras
- [Dynamic Tensor Rematerialization](https://arxiv.org/pdf/2006.09616) — online remat, relevant to D5's fallback
