---
last_updated: 2026-08-09
audit_role: reference
scope: python/tessera/autodiff, src/compiler/ir/AdjointInterface.*, src/transforms/lib/Autodiff*.cpp, ActivationRematerializationPass, AdjointCollectiveInsertionPass, solvers/core NewtonAutodiff
companions: AUTODIFF_UNIFICATION_PLAN.md (sequencing) · ../../spec/AUTODIFF_SPEC.md (normative surface) · RIEMANNIAN_OT_PLAN.md · ../domain/GA_EBM_ARCHITECTURE_REVIEW.md · DIFFERENTIABLE_PROGRAMMING_REVIEW.md (book delta)
---

# Autodiff Architecture and Algorithm Review

> **Routing:** start at [`README.md`](README.md). Findings here feed the scoped
> [`AUTODIFF_UNIFICATION_PLAN.md`](AUTODIFF_UNIFICATION_PLAN.md); global order
> is owned by [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md).
>
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

**Companion delta review.**
[`DIFFERENTIABLE_PROGRAMMING_REVIEW.md`](DIFFERENTIABLE_PROGRAMMING_REVIEW.md)
reads Blondel & Roulet's *Elements of Differentiable Programming* against the
same surface. It independently motivated the original A3/A4/A5/B1/B2/B4/B6/B8
findings; this review now records the landed A5/A6/B6 foundations and their
remaining execution gaps. The book delta also adds findings this review does
not cover: automatic linear transposition (one
registry, not two), nonsmooth/Clarke selection as an undeclared semantic key,
stochastic-computation-graph typing for the effect lattice, semirings, implicit
differentiation (`custom_root`/IHVP/adjoint-state), Fenchel-Young losses, the
smoothing/relaxation operator family, and a Baur–Strassen cost-ratio oracle.

---

## 0. What exists today

| Layer | Surface | Size |
|---|---|---|
| Python engine + reference | `autodiff/{tape,grad,transforms,vjp,jvp,mixed_precision,rematerialize}.py`, `autodiff/geometric/` | ~11.7k lines (`vjp.py` 5.4k, `jvp.py` 3.8k) |
| Graph IR adjoints | `AdjointInterface.{td,cpp}` | 927 lines |
| Reverse-mode passes | `AutodiffPass.cpp` (in-place), `AutodiffPairedPass.cpp` (paired ABI) | 316 + 285 |
| Remat | `ActivationRematerializationPass.cpp` | 624 |
| Distributed adjoints | `AdjointCollectiveInsertionPass.cpp` | 269 |
| Implicit-op derivatives | `solvers/core/passes/NewtonAutodiff.cpp`, registered `tessera_solver` ODS | value-producing shared VJP/JVP IR; physical solve/adjoint lowering open |

The generated ledger reports a broad Python VJP/JVP reference surface, a much
smaller native Graph-IR adjoint set, three placeholder families that round-trip
into Python, a bounded CPU backward-IR oracle set, and a separate set of exact
target backward packages. Read the live summary in
[`generated/autodiff_connection_ledger.md`](../generated/autodiff_connection_ledger.md)
for the current counts; this review deliberately does not duplicate them.

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
mutates the op registry in place. Installation is process-wide and there is no
uninstall path.  However, active-tape state is held in a `ContextVar`, so normal
recording is async/thread-local and nested tape contexts restore their token.
The architectural limitation is global namespace instrumentation plus
identity-keyed values—not that every tape operation is inherently non-reentrant
or thread-unsafe.

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

That is why `hvp` remains finite differences (§B4) even though public `jvp` and
`jacfwd` now execute registered tangent rules exactly. First-order forward mode
is no longer the blocker; the identity-keyed reverse tape still cannot expose a
differentiable gradient program for forward-over-reverse composition.

### A3. Bounded SCF reverse and forward mode exist; general regions fail closed

[`AutodiffPass.cpp`](../../../src/transforms/lib/AutodiffPass.cpp):

```cpp
if (op->getNumRegions() != 0) {
  op->emitError() << "[AUTODIFF_NESTED_REGION] active reverse-mode path "
                     "contains unsupported nested-region op ('" ...
  signalPassFailure();
```

Both reverse passes compute the backward-reachable active cone first. The paired
pass now delegates active `scf.if`, positive-step counted `scf.for`, and the
tracer's canonical bounded `scf.while` to `RegionAdjointInterface`. It returns
cotangents for implicit captures, reverses loop order, and replays pure primal
prefixes under the current recompute-all policy. Noncanonical while forms,
effectful replay, multi-block regions, and `tessera.control_scan` remain
fail-closed; inactive regions are still pruned.

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

### A4. Compiler forward mode exists; region and higher-order closure remain

`TangentInterface`, `--tessera-autodiff-forward`, and the public
`autodiff="forward"`/`"jvp"` request now emit a typed paired Graph function.
Thirty-five families have compiler-owned rules. Direct execution of emitted JVP
IR covers matmul/mul, tanh/sigmoid, sum/mean reduction, softmax,
LayerNorm/RMSNorm, and FFT/IFFT/RFFT/IRFFT; a separate property harness records
the algebraic identity, valid domain, boundary policy, directional derivative,
and forward/reverse duality obligations. Fixed-key EGGROLL and seeded Philox
dropout have replay/linearity proofs rather than invalid smooth finite
differences.

This is still partial, but target execution is no longer absent. A
content-addressed product binds the paired-IR digest to physical children, and
sum, RMSNorm, affine LayerNorm, packed RFFT, spectral-filter, and the diagonal
matrix-free solver product pass natively on AVX-512 and gfx1151.
The x86 result is WSL correctness rather than a clean timing packet;
gfx1200/gfx1250 remain fail-closed. Native collective product execution is
gated to a live multi-rank NCCL/RCCL adapter; its hardware packet is open.
Bounded `scf.if`/positive-step `scf.for`/canonical `scf.while` forward products
and exact public `jacfwd` are now live. Broader spectral/solver products,
general or effectful regions, Apple/CUDA packages, exact forward-over-reverse
HVP, and Taylor/jet composition remain open.

### A5. Backward SSA activity is implemented; region and memory activity remain

`AD-CORE-EFFECT-CONTROL-1` closed the original absence. Both
`AutodiffPass` and `AutodiffPairedPass` compute the backward-reachable SSA cone
from returned values, stamp each top-level operation active or inactive, skip
inactive adjoint construction, consume registered Graph effects, and reject an
active stochastic operation. Inactive stochastic and nested-region producers
are legal. Direct negative tests cover the fail-closed paths.

Activity analysis is the core of Enzyme's performance story — differentiating
*optimized* IR with activity information yields a **4.5× geometric-mean speedup**
over differentiating unoptimized IR on ADBench, and Enzyme "allocates memory to
store only the values needed by the reverse pass."

This is a real compiler activity analysis, but it is narrower than Enzyme's
whole-program memory/alias activity. Active structured operations now propagate
activity to explicit operands and implicit captures, while their interface owns
internal block activity. Tessera still does not exploit read/write/reduce
privileges for whole-program memory activity.

### A6. The measured residual-policy boundary exists; complete family packets remain open

[`AutodiffPairedPass.cpp`](../../../src/transforms/lib/AutodiffPairedPass.cpp)
header:

> Residual policy — RECOMPUTE_ALL (first cut). The backward function takes the
> forward *inputs* as arguments and recomputes any forward intermediates it needs

`recompute_all` remains the generic paired-pass default, which is honest for an
artifact without evidence. It is no longer the only compiler policy. The
execution-derived residual evaluator represents SAVE, RECOMPUTE, HYBRID, and
TREEVERSE candidates per `(op, shape-bucket, dtype, target)`, measures complete
backward work and unique retained bytes, and permits only exact-device evidence
to stamp selector attributes consumed by `ActivationRematerializationPass`.

The remaining gap is execution breadth, not the decision boundary: complete
backward packets are still required family by family, and estimated Treeverse
envelopes cannot promote a policy. An unstamped generic artifact therefore
continues to recompute rather than inventing a target verdict.

**Contrast.** JAX exposes `jax.checkpoint(policy=...)` with saveable-value
predicates (`dots_saveable`, `checkpoint_dots_with_no_batch_dims`, …), while
Enzyme decides per-value from static analysis. Tessera's distinct mechanism is
an exact-target measurement gate; the current open obligation is to populate
that gate rather than claim fleet-wide selection from the infrastructure alone.

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
Its attributes currently have no runtime/codegen consumer, so the immediate
problem is inert policy in the default pipeline rather than an active backend
checkpoint schedule.

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

### B6. Treeverse candidates exist, but counted-loop execution is still absent

`ActivationRematerializationPass` performs deterministic liveness-aware selection
of the largest long-lived pure activation intervals until the estimated peak fits
the budget. That is a sound greedy heuristic for a straight-line block.

The residual evaluator now generates balanced Treeverse candidate envelopes
from measured per-step work, explicit state size, and memory budgets. Every such
candidate is deliberately selector-ineligible until its complete backward
executes. This is planning and pruning, not a Revolve implementation in the
compiler. Per A3, `ActivationRematerializationPass` still cannot execute a
checkpoint schedule inside an active loop (`REMAT_NON_CLONABLE` on nested
regions).

Thus long scans, diffusion trajectories, and the `T = 2500` RNOT inner loop can
now be ranked as candidate envelopes, but they still receive no executable
binomial checkpoint schedule. Straight-line blocks retain the working greedy
implementation.

### B7. Adjoints of collectives are real; the cross-IR proof boundary is the differentiator

[`AdjointInterface.cpp:42-70`](../../../src/compiler/ir/AdjointInterface.cpp#L42):
`AllReduce` is self-dual; `AllGather† = ReduceScatter`; `ReduceScatter† =
AllGather` — correct, and `AdjointCollectiveInsertionPass` places them
effect-aware, after `EffectAnnotationPass`, keyed on `tessera.effect = "memory"`.

Most framework training stacks bolt distribution on outside AD through hooks or
wrappers. However, JAX also has primitive transposition rules for
replication-inducing collectives, so the existence of a collective transpose is
not a unique Tessera lead. Tessera's stronger claim is narrower: all four
collectives retain registered Graph adjoints and typed Schedule→Tile→Target
contracts with an exact-backend proof ledger. Native multi-rank NCCL/RCCL and
MPI/OFI/SHMEM packets remain open and must land before calling that boundary
complete.

### B8. Implicit-function-theorem differentiation is scaffolded, not absent

`NewtonAutodiff.cpp` now walks registered `tessera_solver.implicit` operations,
requires an explicit residual-function symbol and exact signature, and emits a
private value-producing VJP (and optional JVP) function. The VJP carries
`residual(parameters, solution)` → transposed matrix-free `linear_solve` →
negative `residual_adjoint` SSA values, implementing
`dF/dx = -(dR/dx)⁻¹ · dR/du` without a runtime annotation lookup. Missing and
mismatched residual ABIs fail closed. A bounded diagonal-sqrt pilot now lowers
through content-addressed solver artifacts and executes complete
residual/solve/adjoint packages on Zen 5 AVX-512 and gfx1151. General
matrix-free iterative/Krylov execution, Apple/NVIDIA consumers, and broader
shape/dtype envelopes remain open; the two pilot packets do not transfer proof
to another architecture.

**This corrects a claim in the [OT plan](RIEMANNIAN_OT_PLAN.md) §3.2.** The
implicit-diff seam now has a value-producing shared IR body. R2's `custom_root`
must lower through this function contract rather than introduce a parallel
mechanism. Same correction applies to the RNOT Jacobian requirement — App.
F.3's `J = −[D_yF]⁻¹[D_xF]` is the emitted residual/solve/adjoint chain.

---

## 3. Position against the state of the art

Honest, per-capability. "Partial" is used where a real but incomplete
implementation exists. Tessera counts are generated from the current ledger;
external cells describe documented core capabilities checked on 2026-08-09,
not ecosystem packages or a performance comparison.

| Capability | Tessera | JAX | Enzyme / EnzymeMLIR | LAGrad |
|---|---|---|---|---|
| Reverse mode, straight-line | ✅ bounded: 51 native IR adjoints; 36 CPU-oracle and 29 exact-target proven | ✅ | ✅ | ✅ |
| Reverse mode through structured control flow | partial — paired reverse supports single-block `scf.if`, counted `scf.for`, and canonical bounded `scf.while`; general/effectful regions fail closed (A3) | partial — `cond`/`scan` and static loops; `while_loop` is forward-only | ✅ | ✅ stated scope |
| Forward mode in the compiler | partial — public paired Graph JVP ABI, exact `jacfwd`, 35 native tangent families, bounded `if`/`for`/`while`, and native x86/gfx1151 products for normalization, spectral, and diagonal solver families; general regions, higher-order composition, broader packages, and native collective evidence remain open | ✅ | ✅ | partial |
| Higher-order (`grad∘grad`) | ❌ structurally blocked (A2) | ✅ | ✅ | — |
| Exact HVP | ❌ finite differences (B4) | ✅ fwd-over-rev | ✅ | — |
| `vmap` as a transform | ❌ Python loop (B3) | ✅ | n/a | n/a |
| Activity analysis | partial — backward SSA/effect activity complete; region/memory activity open (A5) | partial | ✅ core analysis | ✅ static analysis |
| AD after optimization | partial — optimized Graph IR only | partial — AD on staged JAXPR, XLA downstream | ✅ (4.5× geomean) | ✅ |
| Residual policy, per-target measured | partial — exact-device selector boundary landed; family packets open (A6) | policy-driven checkpointing, not measured selection | static analysis, not measured selection | partial static |
| Revolve / binomial checkpointing | partial — counted-region SAVE/RECOMPUTE/HYBRID candidates execute with measured work; selected plans are not yet lowered into MLIR (B6) | manual `checkpoint`/`remat`; no automatic Revolve selector | partial | — |
| Sparsity detection + coloring | ❌ (B5) | ❌ | ❌ | partial (static) |
| Collective adjoints as IR ops | partial — four typed cross-IR contracts; native multi-rank proof open (B7) | ✅ primitive transpose rules | not established in reviewed core | not established in reviewed core |
| Manifold / geometric AD | partial — Python oracle, no general compiler tangent transform | not a core JAX facility | not established in reviewed core | not established in reviewed core |
| Per-target device-verified backward proof | ✅ 29 exact-target oracle-proven families with independent ledger axes | no comparable public proof ledger | no comparable public proof ledger | no comparable public proof ledger |

The defensible Tessera distinction is its per-target backward proof discipline
and the full cross-IR identity carried by collective adjoints—not the mere
existence of collective transposes. Four capabilities remain absent:
higher-order composition, exact HVP, transformed `vmap`, and sparsity
detection/coloring. Structured-control differentiation, general compiler
forward mode, activity, residual selection, and Treeverse are partial
foundations with explicit coverage or execution gaps rather than absent
capabilities.

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
how MLIR scales a cross-cutting concern across the registered op families without one pass
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

### D2 — Forward mode in the compiler  *(landing)*

The AD-FWD-CORE-1 foundation, AD-FWD-PRODUCT-2 public boundary, and bounded
AD-FWD-NATIVE-1 product are
implemented: ODS `TangentInterface`,
`--tessera-autodiff-forward`, a paired JVP function contract, fail-closed active
operation/region legality, and 31 native rules spanning algebra, structural
views, sum/mean reduction, normalization, core spectral transforms,
collectives, deterministic dropout replay, and fixed-key EGGROLL. Compiler JVP
IR for algebra, reduction/normalization, and FFT/IFFT/RFFT/IRFFT executes in the
CPU oracle and agrees with analytic, duality, or directional references. Public
`autodiff="forward"` / `"jvp"` requests now select a
mode-neutral provenance facet and a `wrt`-indexed paired ABI through
`compiled_jvp_ir()`; reverse compatibility fields cannot accidentally report a
JVP as backward execution. Tanh and sigmoid have direct compiler JVP proofs.
The generated ledger reports `ir_tangent` separately from Python JVP
availability. The native parent is tamper-evident and executes compiler-fixed
child packages without Graph redispatch; exact gfx1151 and WSL Zen 5 numerical
packets cover sum, non-affine RMSNorm, and packed RFFT.

Still required for D2 closure: broader compound spectral and solver families,
native collective, loss, and optimizer products; direct oracle rows for every
advertised family; general/effectful region tangents; Apple/CUDA consumption;
and clean target performance evidence. Forward mode needs no reverse tape or
residual policy, but it does require activity and effect legality once it
crosses regions or memory.

Already unlocked: exact compiler-owned public `jacfwd`. The remaining closure
unlocks exact HVP via forward-over-reverse, tangent-space ops for the
[manifold work](RIEMANNIAN_OT_PLAN.md), and the substrate for D6.

Maps to: P5 (family expansion), running in parallel.

### D3 — Activity and structured-region propagation complete; memory extension open

The landed backward SSA analysis computes the active cone from seeded outputs,
stamps activity, skips inactive adjoints, and enforces registered stochastic
effects. Structured operations propagate activity through explicit operands
and implicit captures into `RegionAdjointInterface`. The remaining extension is
whole-program memory activity over aliasing and read/write/reduce privileges.

This is where the Enzyme-class speedup lives, and where Tessera's extra
information (effects, privileges, static shapes) should let it do better than a
system working on LLVM IR.

The original inactive-region criterion and the active bounded-region structural
criterion are complete. The next exit criterion is execution of the emitted
region backward with numerical oracle agreement and a selected residual policy.

Maps to: P2 (foundation complete) and D4 (region extension).

### D4 — Structured control-flow adjoints  *(~6 weeks — hardest, highest value)*

`RegionAdjointInterface` now implements reverse mode over single-block
`scf.for` / `scf.if` / canonical bounded `scf.while`. The current construction
recomputes pure prefixes, reuses the executed branch predicate, and carries the
actual while trip count. Remaining work is native/CPU numerical execution,
measured checkpoint-plan lowering, multi-block regions, and
`tessera.control_scan`.

Everything with a loop is blocked on this: SSM/linear-attention scans, diffusion
samplers, solver iterations, the EBM Langevin loop, the RNOT `c`-transform. It is
also the precondition for D5's Revolve.

Maps to: P5, and it is the correct next big rock after P4.

### D5 — Measurement boundary and Treeverse candidates landed; execution open

- **Landed:** execution-derived SAVE / RECOMPUTE / HYBRID selection per
  `(op, shape-bucket, dtype, target)`, gated on complete exact-device backward
  samples and retained-residual bytes.
- **Landed as pruning only:** measured-step Treeverse candidates for explicit
  memory budgets; estimates are selector-ineligible.
- **Open after D4:** lower a selected binomial schedule into counted-loop
  forward/backward execution and measure the complete backward before promotion.
- Delete `EBMCheckpointInnerLoop`; register its op knowledge into
  `ActivationRematerializationPass` (A8).

Maps to: P4/P6; tracked by `AD-RESIDUAL-EVAL-1`.

### D6 — Higher-order, hosted on the geometric-algebra engine  *(~4 weeks — the differentiating move)*

Two steps:

1. Exact forward-over-reverse HVP, once D2 exists.
2. **Taylor / jet mode over Weil algebras — potentially sharing a future
   generic finite-algebra lowering substrate with GA.**

Step 2 deserves emphasis, because it is a synergy nobody else has. Taylor-mode AD
computes all mixed partials to order `k` in a single forward pass at cost linear
in the algebra dimension, by carrying values in a **truncated polynomial (Weil)
algebra** instead of ℝ. A Weil algebra is a finite-dimensional commutative
algebra with a compile-time-known multiplication table and a nilpotent grading.

Tessera does not yet have that general object.  It has a useful implementation
pattern:
[`ga/signature.py`](../../../python/tessera/ga/signature.py) builds a
compile-time-cached, graded, bitmask-indexed product table from a signature and
caches it per algebra, and
[`ExpandProductTable.cpp`](../../../src/solvers/clifford/lib/Passes/ExpandProductTable.cpp)
lowers a **Clifford** product table to unrolled IR.  `ga/signature.py` hard-codes
blade XOR, metric signs, and Clifford anti-commutation; it cannot represent an
arbitrary commutative nilpotent Weil algebra by changing `(p,q,r)`.

If the GA review's table-driven synthesis is generalized to accept an explicit
finite multiplication table, both Clifford and Taylor/Weil lowering could use
that substrate.  That generalization is new design work; Taylor mode does not
arrive "largely for free" from the current Clifford signature engine.

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

### Alongside — consume the `NewtonAutodiff` IFT body

The shared IFT body landed on 2026-08-08: registered residual, matrix-free
linear-solve, residual-JVP, and residual-adjoint values replace annotations.
The diagonal-sqrt residual still has the first monolithic Schedule→Tile
specialization. A second content-addressed physical parent now binds arbitrary
compiled residual and solution/parameter JVP/VJP children and executes
restarted GMRES on AVX-512 and gfx1151. The compiler now generates all five
children from verified typed Graphs containing pointwise operations, sum/mean,
rank-2 matmul/transpose, distinct parameter and solution spaces, bounded
dynamic dimensions, explicit mixed-storage widening, and statically bounded
`control_for`. Pure scalar `if` and bounded `while` additionally become
explicit compare/select SSA; primal and product children recompute the same
digest-bound predicate. Reverse reduction products explicitly unbroadcast
cotangents. Thirty-sample AVX-512 and gfx1151 WSL correctness packets cover the
nonlinear baseline plus reduction, reduced-storage matmul, bounded-dynamic
mixed storage, both predicate regions, and ISTFT window products without a
dense Jacobian. The remaining shared deliverable with
[OT plan](RIEMANNIAN_OT_PLAN.md) R2 is broader/non-pure predicate legality and
iterative/Krylov selector-grade performance proof; the two tracks still must
not build that mechanism twice.

### Sequencing

```
D1 (1w) ──────────────────────────────────────────────────►  (independent)
        D2 (3w) ──► D6 (4w)            [needs a generic finite-algebra substrate]
D3 SSA foundation complete ──► D4 (6w) ──► bounded executable D5 Treeverse
                           └──────────────► D7 (5w)
D5 measurement boundary complete; exact family packets continue in parallel
NewtonAutodiff shared IR complete ──► bounded x86/gfx1151 pilot ──► general solvers
```

The historical estimates remain useful sizing only; global order is owned by
`INTEGRATED_COMPILER_PLAN.md`. The next shared capability rocks here are D2 and
D4, while architecture queues supply D5 family packets and physical solver
consumers.

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
3. **"Collective adjoints retain one typed cross-IR identity with per-target
   device-verified proof."** The first half is true (B7 + the ledger); JAX also
   has collective transpose rules. The defensible claim requires finishing the
   native multi-rank distributed lane on each target.
4. **"Residual policy chosen by measurement per target, not by convention."**
   The D5 selection boundary is implemented. The claim becomes production-wide
   only after exact-device complete-backward packets populate it across the
   promoted family envelope. JAX exposes a policy knob and Enzyme uses static
   analysis; neither is the same evidence contract.

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
  keeps native IR adjoints, CPU-oracle proof, runtime binding, and exact-target
  device proof as separate, partly disjoint axes. Read their live counts from
  the generated ledger rather than copying another snapshot here.
- **Effort figures are engineering estimates** for a single track with no
  hardware-gated dependencies, not commitments. D4 is the one most likely to
  exceed its estimate.

---

## Sources

- [Enzyme: high-performance AD of LLVM and MLIR](https://github.com/EnzymeAD/Enzyme) · [enzyme.mit.edu](https://enzyme.mit.edu/) — AD on optimized IR, 4.5× geomean on ADBench, activity analysis, EnzymeMLIR op interfaces
- ["Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients"](https://arxiv.org/pdf/2010.01709) — the post-optimization AD result
- [LAGrad: Statically Optimized Differentiable Programming in MLIR (CC 2023)](https://dl.acm.org/doi/10.1145/3578360.3580259) — MLIR-level reverse mode exploiting high-level dialect semantics, sparsity, and structured control flow
- [JAX structured control-flow documentation](https://docs.jax.dev/en/latest/control-flow.html) — reverse-mode support for `cond`, `scan`, and statically bounded loops; forward-only `lax.while_loop`
- [JAX JEP 17111: efficient transposition of replication-inducing collectives](https://docs.jax.dev/en/latest/jep/17111-shmap-transpose.html) — primitive transpose rules for `psum` and `all_gather`
- [An Illustrated Guide to Automatic Sparse Differentiation (ICLR 2025 blogpost)](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) and ["Sparser, Better, Faster, Stronger"](https://arxiv.org/pdf/2501.17737) — sparsity detection + coloring; the observation that PyTorch/TF/JAX lack it
- [Gradient checkpointing with `jax.checkpoint` / `jax.remat`](https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html) — saveable-value remat policies
- [Treeverse / Revolve checkpointing](https://www.researchgate.net/publication/2290186_Treeverse_An_Implementation_of_Checkpointing_for_the_Reverse_or_Adjoint_Mode_of_Computational_Differentiation) and [divide-and-conquer checkpointing with no user annotation](https://openreview.net/forum?id=BkYYXJ9i-) — provably optimal binomial schedules
- [Jet functors and Weil algebras in AD](https://arxiv.org/pdf/2510.14342) and [Collapsing Taylor Mode AD](https://arxiv.org/pdf/2505.13644) — higher-order AD over truncated polynomial algebras
- [Dynamic Tensor Rematerialization](https://arxiv.org/pdf/2006.09616) — online remat, relevant to D5's fallback
