---
last_updated: 2026-08-02
audit_role: reference
scope: python/tessera/ga, python/tessera/ebm, src/solvers/clifford, src/solvers/ebm
companion: ../compiler/RIEMANNIAN_OT_PLAN.md
---

# GA + EBM Architecture Review

A review of the geometric-algebra and energy-based-model surfaces against the
compiler as it stands in 2026-08, not as it stood when they were written. Both
tracks were built before the `emit/` synthesizer, before the arbiter model
(Decision #28), and before the Target IR contract hardened. Several decisions
that were correct then are load-bearing constraints now.

**This is a review, not a status document.** `MASTER_AUDIT.md` and
`docs/audit/generated/` remain status truth. Nothing here reclassifies a
generated row.

Surface reviewed: ~5.5k lines Python (`ga/` 2.6k, `ebm/` 2.0k, op bridges 0.3k),
~1.1k lines C++ across two dialects and seven passes, plus the x86/ROCm kernel
generators and the Apple GPU symbol set.

---

## Summary

The GA and EBM tracks are well-built for what they were scoped to be: correct
NumPy references with hand-written fast paths on three backends, plus MLIR
dialects with real pass bodies and lit fixtures. The algebra code in particular
is clean — `signature.py` computes Cayley tables from first principles with a
proper bitmask blade representation, and it is fully general in `(p, q, r)`.

The findings below are not about correctness of the math. They are about three
structural facts that the compiler's own progress has turned from reasonable
into costly:

1. **Semantic metadata is carried but never consumed.** The `manifold`
   attribute reaches no backend. Grade structure is declared in the type system
   and discarded at the op boundary. In both cases the information the compiler
   would need to specialize is present and thrown away.
2. **The scalar field is an opaque host callable.** EBM energies and (in the
   RNOT plan) OT potentials are Python closures. Every "native" sampler
   evaluates its gradient on the host and ships the result to the device as
   data. This caps every iterative loop in both tracks at host round-trip speed.
3. **Two disconnected compilers, again.** The GA lane reproduces the exact seam
   `CLAUDE.md` describes for Apple: MLIR passes that recognize and mark
   optimizations, and a parallel Python path that calls hand-written kernels,
   with no connection between them.

Two of the findings (§1.1, §1.5) are the failure modes flagged in the
[Riemannian OT plan](../compiler/RIEMANNIAN_OT_PLAN.md) as future risks. They
are present defects today.

---

## 1. Architectural findings

### 1.1 The `manifold` attribute silently defaults to Euclidean — and no backend reads it

Three separate facts compound here.

**It defaults.** [`Canonicalize.cpp:56`](../../../src/solvers/ebm/lib/Passes/Canonicalize.cpp#L56):

```cpp
op->emitWarning("tessera_ebm.langevin_step missing `manifold`; "
                "defaulting to 'euclidean'");
op->setAttr("tessera.ebm.manifold", StringAttr::get(ctx, "euclidean"));
```

A warning is not a failure. A `langevin_step` that loses its `manifold`
attribute anywhere upstream becomes a Euclidean sampler that converges, looks
numerically healthy, and produces wrong samples — because `exp_x(v) ≈ x + v` is
correct to first order on any manifold.

**It is unvalidated.** The ODS declares `StrAttr:$manifold`
([`EBMOps.td:120`](../../../src/solvers/ebm/lib/Dialect/EBM/EBMOps.td)), and
`EBMOps.cpp` is 11 lines with no custom verifier. `"Sphere"`, `"sphere2"`, and
`"spere"` all parse, canonicalize, and propagate.

**It is never consumed.** `grep -rn manifold src/compiler/codegen/` returns six
hits, **all of them comments**. No backend kernel generator branches on it. The
only thing keeping sphere sampling correct today is that Python calls a
different *function* (`sphere_langevin_step` vs `langevin_step`) which calls a
different C symbol. The IR attribute is decorative.

**The fix already exists in the sibling dialect.** `AnnotateAlgebra.cpp:70`
handles the identical situation correctly:

```cpp
if (!algebra) {
  op->emitError("tessera_clifford op missing required `algebra` attribute");
  anyError = true;
  return WalkResult::interrupt();
}
```

Hard error, no default, interrupt. Two dialects in the same `src/solvers/` tree,
opposite policies on the same class of attribute. EBM should copy Clifford.

Design and tripwires: [RIEMANNIAN_OT_PLAN.md §H1](../compiler/RIEMANNIAN_OT_PLAN.md).

### 1.2 The energy function cannot cross the device boundary

The ODS gets this right — `FlatSymbolRefAttr:$energy_fn` models the energy as a
symbol reference to a function. The frontend never produces one. Python passes a
closure:

```python
langevin_step(x, energy_fn=<python callable>, ...)
```

Consequently, on **every** native path, the gradient is evaluated on the host and
handed to the device as data.
[`geo_sampling.py:279`](../../../python/tessera/ebm/geo_sampling.py#L279), inside
the branch commented "the whole step … is one MSL kernel":

```python
grad_f32 = np.asarray(grad_fn(x_arr.astype(np.float32, copy=False)), dtype=np.float32)
gpu_out = _try_apple_gpu_sphere_langevin_step_f32(x_arr, grad_f32, noise, eta, noise_scale)
```

The kernel fuses tangent-projection, the Euler–Maruyama update, and the retract —
genuinely. But the expensive term, `∇E`, is computed in Python, one host↔device
round trip per step. The fusion claim in the status ledger is true about the step
arithmetic and misleading about the loop.

`FuseEnergyGrad` cannot help, and says so in its own header:

> Without an explicit `ebm.grad_y` op in the dialect, this v1 pass is
> annotation-only: it doesn't rewrite the IR, just marks pairs.

There is no gradient op in the EBM dialect. There is nothing to fuse.

This is the single largest structural limit in the EBM track, and it is
**identical in shape** to the RNOT inner-loop problem. One fix serves both: the
scalar field must become a traced Graph IR region, not a closure. Tessera has had
the machinery for this since `@jit` tracing and `custom.py::def_lowering`
landed — the EBM API predates it.

### 1.3 The GA lane is two disconnected compilers

`RotorSandwichFold.cpp` recognizes `R · x · R†` written as three primitives and
rewrites it to a single `clifford.rotor_sandwich` marker, so that (per its
header) "GA9 backends can pick up ... a fused kernel."

Independently, [`ga/ops.py:759`](../../../python/tessera/ga/ops.py#L759) hardcodes:

```python
gpu_out = _try_apple_gpu_rotor_sandwich_cl30_f32(rotor, x)
if gpu_out is not None:
    return gpu_out
return geometric_product(geometric_product(rotor, x), reverse(rotor))
```

The MLIR pass's marker has no consumer. The Python fast path has no IR. Neither
knows the other exists. This is the seam `CLAUDE.md` describes for the Apple
backend, reproduced inside the GA track.

It is worse than the Apple case in one respect: **`ExpandProductTable` refuses
batched operands.**

```cpp
if (lhsTy.getRank() != 1 || rhsTy.getRank() != 1) {
  op->emitWarning("ExpandProductTable: batched (rank > 1) operands are pending a "
                  "follow-on sprint; skipping");
  return failure();
}
```

The MLIR GA lane therefore handles exactly one unbatched multivector. Every real
GA workload — which is batched by definition — goes through Python to a
hand-written kernel. The dialect is lit-testable and production-inert.

### 1.4 Pass declarations drift from pass bodies, in both directions

`CliffordPasses.td` labels `ExpandProductTable`, `GradeFusion`, and
`RotorSandwichFold` as `[GA8 stub] ... no IR rewriting yet`. All three have real
bodies (190 / 115 / 136 lines) with lit fixtures. `ts-clifford-opt --help`
reports them as stubs.

`EBMPasses.td` labels `FuseEnergyGrad`, `CheckpointInnerLoop`, and
`PipelineCandidates` as `[EBM6 stub]`. All three exist — but are genuinely
annotation-only, which is a *different and more useful* claim than "stub". A
reader cannot currently tell "not written" from "written, marks only."

Both directions of drift in one sibling file pair. Cheap to fix; worth fixing
because these summaries are what `--help` prints.

One related false promise: `AnnotateAlgebra` warns that a non-allow-listed
signature means "GA8 lowering will refuse", but `ExpandProductTable` never checks
`tessera.clifford.canonical` or `allow_listed` — it reads `algebra` directly and
builds the table, which is fully general. A non-allow-listed signature would
lower silently. The diagnostic describes an enforcement that does not exist.

### 1.5 `CheckpointInnerLoop` annotates without any liveness or differentiability analysis

The whole body of the analysis is:

```cpp
mod.walk([&](scf::ForOp loop) {
  bool sawInnerStep = false;
  loop.getBody()->walk([&](Operation *op) {
    if (isInnerLoopStep(op->getName().getStringRef())) {
      sawInnerStep = true;
      op->setAttr(kRecomputeStepAttr, builder.getUnitAttr());
    }
  });
  if (sawInnerStep) { /* mark loop, set budget = 4 */ }
});
```

Purely syntactic: *is there a step op in this loop?* → mark every step
rematerializable, budget hardcoded to 4. It never asks whether anything
downstream consumes the trajectory.

This contradicts Decision #10, which specifies that recompute insertion is
"budget-guided" using "a greedy live-set scan" with "only pure ops qualifying."
The general `InsertRecomputePass` does that scan. This domain pass bypasses it.

The practical consequence, and the reason it matters now: on an
envelope-theorem loop — RNOT's `c`-transform, where the inner trajectory is
provably never differentiated through — this pass would annotate 2500 dead steps
as rematerializable and instruct the backend to keep four live states of a
trajectory nothing reads. The budget default is documented as "enough to fit a
typical T=16 chain"; the workloads in flight are T=2500.

Design and tripwires: [RIEMANNIAN_OT_PLAN.md §H2](../compiler/RIEMANNIAN_OT_PLAN.md).

---

## 2. Algorithmic findings

### 2.1 Grade sparsity is declared in the type system, then discarded at the op boundary

`MultivectorSpec` carries `grades`, `is_grade_pure`, `is_even`, `is_odd`,
`grade_value`. `constraints.py` ships `GradeIn`, `Even`, `Odd`, `IsRotor`,
`IsForm`, all checked at decoration time by the `ConstraintSolver` (Decision #4).
The compiler therefore *knows*, statically, that a rotor has support only on
grades {0, 2}.

`geometric_product` then does this
([`ops.py:93`](../../../python/tessera/ga/ops.py#L93)):

```python
for i in range(dim):
    ai = a_co[..., i]
    if not np.any(ai):     # ← runtime value check, not a type check
        continue
    for j in range(dim):
        ...
```

All `dim × dim` blade pairs, skipped only by a **data-dependent** test on the
actual coefficient values. The static grade information is not consulted.

Concretely, in Cl(3,0): a rotor is 4 nonzero coefficients of 8. Rotor × rotor is
16 blade products, not 64. `rotor_sandwich` is currently two dense products =
128 mul-adds where a grade-aware contraction is roughly 30. In Cl(1,3) the ratio
is worse (8 of 16 for even elements, 256 → 64).

The C++ side has the same blind spot from the other direction: `GradeFusion`
propagates only **output** grades (`tessera.clifford.output_grades`). There is no
`input_grades` attribute anywhere. So even the implemented, working fusion pass
emits terms that multiply coefficients known at compile time to be zero.

This is the clearest instance of the review's theme: the information exists,
the type system captures it, the constraint solver validates it, and the code
generator ignores it.

### 2.2 Dense `2^n` storage for structurally sparse values

`Multivector` always stores `[..., 2^n]` coefficients. A Cl(3,0) rotor occupies
8 floats where 4 carry information; a Cl(1,3) even element 8 of 16. Batched GA
kernels are memory-bandwidth-bound, so this is a direct 2× on traffic for the
most common values in the algebra (rotors and even elements dominate real usage).

Decision #15a already makes `layout` one of the six canonical tensor attributes.
A graded layout is squarely in scope and does not require a new concept — only
that `MultivectorSpec.grades` reach the layout instead of stopping at validation.

### 2.3 The Cayley table is a compile-time-known sparse pattern — textbook synthesizer input, currently hand-written per signature

There are 17 hand-written Apple GA kernels, all `cl30_f32`, with symbol names
spelled out literally at each Python call site
(`"tessera_apple_gpu_clifford_geo_product_cl30_f32"`, `..._exp_cl30_f32`,
`..._rotor_sandwich_cl30_f32`, …). Plus x86 AVX-512 and ROCm generators for the
Cl(3,0) bilinear lane.

The Cayley table **is** the kernel. It is exactly, precisely, the kind of
compile-time-known, signature-parametric sparse contraction that Decision #28's
tier-1 synthesizer exists to specialize — and `python/tessera/compiler/emit/`
now exists, with `apple_msl.py`, `nvidia_cuda.py`, `rocm_hip.py`, and
`x86_llvm.py` over arch-agnostic regions in `fusion_core.py`. It did not exist
when GA was written.

One table-driven emitter would replace the hand-written set *and* make Cl(1,3),
PGA, and CGA free rather than each being another 17 kernels.

`ExpandProductTable`'s own header shows the strategy is welded to the current
allow-list:

> the alternative (linalg.generic + sparse-tensor encoding) would add MLIR-pipeline
> complexity for marginal benefit at these algebra sizes — Q1 locks v1 to dim ≤ 16.

Sound reasoning under that premise. It stops being sound the moment the
allow-list moves (§2.4): CGA Cl(4,1) is dim 32, i.e. 1024 scalar mul-adds
unrolled.

### 2.4 PGA and CGA are blocked by a frozenset, not by missing capability

`_blade_product` fully implements degenerate generators — the `r_mask` path
returns `(0, 0)` for any blade whose product annihilates. `Cl` is parameterized
on `(p, q, r)` throughout. `_product_table`, `_basis_list`, and every downstream
op are signature-generic.

The entire gate is:

```python
V1_ALLOWED_SIGNATURES = frozenset({(3, 0, 0), (1, 3, 0)})
```

**PGA `Cl(3,0,1)`** — the standard algebra for rigid-body motion, screw theory,
lines, and points-at-infinity — is one tuple away from constructible. That is
directly the algebra a manifold-aware compiler wants: `SE(3)` motors are the
natural home for `exp_map`/`log_map` on the rigid-motion group, which is the
same primitive the [OT plan](../compiler/RIEMANNIAN_OT_PLAN.md) needs for
`manifold="se3"`. **CGA `Cl(4,1)`** adds spheres and circles as first-class
elements.

**Stated honestly: one line to unlock, one sprint to prove.** Adding a signature
is trivial; validating it is not. PGA rotors (motors) do not use the Cl(3,0)
Euler identity in `exp_mv` — the degenerate generator makes the closed form
dual-number-flavored, and the current `_is_pure_bivector_3d` guard would silently
route a motor to the 24-term power series. Per-signature product-table identity
tests, exp/log closed forms, and kernels are all real work. The finding is that
the *architecture* does not block it; only the allow-list and the proof burden do.

### 2.5 Scalar dependency chains defeat ILP in the expanded product

`ExpandProductTable` accumulates each output coefficient as a strict serial
chain:

```cpp
updated = rewriter.create<arith::AddFOp>(loc, outCoeffs[entry.result_mask], prod);
outCoeffs[entry.result_mask] = updated;
```

Individual `arith.mulf` + `arith.addf`, no `arith.fma`, no tree reduction, no
vector ops. For Cl(1,3), each of 16 output coefficients accumulates up to 16
terms in a 16-deep serial dependency chain. Reassociating into a balanced tree
and emitting FMA is a local change with a real payoff, and it becomes necessary
rather than nice at dim 32.

### 2.6 Numerical-gradient fallbacks cost `O(d)` and `O(2^n)` energy evaluations per step

[`geo_sampling.py:45`](../../../python/tessera/ebm/geo_sampling.py#L45)
(`_numerical_grad_mv`) central-differences every one of `2^n` multivector
coefficients: **16 host energy calls per Cl(3,0) step, 32 for Cl(1,3)**.
[`energy.py:60`](../../../python/tessera/ebm/energy.py#L60) (`_numerical_grad`)
does the same over `d` dimensions: `2d` calls per step.

This is the **default** path — it runs whenever the caller does not supply an
analytic `grad_fn`, which is the common case for anything but the hardcoded
quadratic energy.

Tessera has tape-based reverse-mode autodiff (`autodiff/vjp.py`,
`autodiff/tape.py`) covering a broad Tessera primitive set, but it does not trace
arbitrary NumPy callbacks. The EBM samplers therefore cannot replace finite
differences unconditionally: an ordinary NumPy `energy_fn` records no cotangent
path, and the public gradient transform currently materializes a zero cotangent
for that case. The safe route is an explicit traceable-energy contract: use one
forward + one backward only when the tape records a supported path, and retain
the numerical gradient for untraceable callbacks. Tests must cover both a
Tessera-op energy and the existing NumPy energies before the tape becomes a
default. This removes the `eps`-selection accuracy problem where reverse mode is
actually supported without turning valid gradients into zeros elsewhere.

This remains high leverage, but it is a guarded migration rather than a
drop-in replacement.

### 2.7 AIS and Monte-Carlo partition estimators are Python-only and structurally host-bound

`partition_function_exact` has Apple GPU, x86, and ROCm lanes (a stable
log-sum-exp — well suited to it). `partition_function_monte_carlo` and
`partition_function_ais` have none, on any target.

AIS is not incidentally slow: it is an outer loop over a temperature schedule,
each step invoking the sampler, i.e. the *same* nested host-bound iterative shape
as §1.2. It cannot be fixed by writing an AIS kernel; it is fixed by making the
sampler loop device-resident, which requires §1.2.

---

## 3. The unifying observation

Three families in this tree are the same compiler object:

| Family | Loop | Scalar field | Manifold | Trip count |
|---|---|---|---|---|
| EBM Langevin / AIS | Euler–Maruyama | energy `E` | euclidean / sphere / bivector | 10–10³ |
| GA rotor flows | rotor integration | GA-valued potential | even subalgebra | 10–10² |
| RNOT `c`-transform | Riemannian GD | dual potential `ψ_θ` | sphere / torus | up to 2500 |

Every one is a **fixed-trip-count, device-resident iterative refinement of a
point on a manifold under the gradient of a scalar field, with an explicit
differentiability boundary.** Every one is currently implemented separately.
Every one is host-bound at the gradient. Every one carries manifold information
that no backend reads.

The recommendation is not three parallel tracks. It is one substrate:

- **A manifold contract** (§1.1) that is typed, verified, required, and part of
  the backend dispatch key — serving EBM's `manifold`, GA's `algebra`, and
  RNOT's `manifold` under one rule.
- **A scalar-field-as-region contract** (§1.2) so the energy/potential is traced
  Graph IR, its gradient is an IR op, and `FuseEnergyGrad` has something to fuse.
- **A demand-gated remat policy** (§1.5) unified with `InsertRecomputePass`, so
  the three loops share one correct answer about what must stay live.

Given those three, EBM Langevin, GA rotor flows, and the RNOT `c`-transform
become three configurations of one fused-loop region — which is also precisely
the shape Decision #28's arbiter is built to choose kernels for.

---

## 4. Recommended work, ranked by value per unit effort

| # | Item | § | Effort | Why this rank |
|---|---|---|---|---|
| 1 | Add a traceable-energy contract for `grad_fn=None`; use `autodiff.tape` for supported Tessera-op energies and retain numerical differentiation for untraceable NumPy callbacks | 2.6 | ~1 week | Removes repeated energy evaluations where reverse mode is valid while preserving the existing correct fallback and NumPy regression coverage. |
| 2 | Make `manifold` a required, verified enum; delete the Euclidean default | 1.1 | ~3 days | Closes a live silent-wrong-answer path. Pattern already exists in the sibling dialect. |
| 3 | Demand-gate `CheckpointInnerLoop` + negative fixtures | 1.5 | ~4 days | Live defect; the fixture is the durable artifact. |
| 4 | Fix `.td` summaries; distinguish "stub" from "annotation-only"; remove the false "GA8 will refuse" promise | 1.4 | ~1 day | Trivial; these strings are what `--help` prints. |
| 5 | Thread `MultivectorSpec.grades` into `geometric_product` and add `input_grades` to `GradeFusion` | 2.1 | ~1 week | 2–4× on the dominant GA ops using information already computed. |
| 6 | Batched operands in `ExpandProductTable` | 1.3 | ~1 week | Without it the MLIR GA lane cannot run a real workload. |
| 7 | Table-driven GA kernel synthesis via `emit/` | 2.3 | ~3 weeks | Replaces 17 hand kernels; makes new signatures cheap. Do after 5 and 6 so it synthesizes the *graded* contraction. |
| 8 | Scalar-field-as-region + `ebm.grad_y` op | 1.2 | ~4 weeks | Largest payoff, largest scope; unblocks device-resident loops for all three families and gives `FuseEnergyGrad` a job. |
| 9 | Graded layout in the multivector type | 2.2 | ~2 weeks | 2× memory traffic. Depends on 5. |
| 10 | PGA `Cl(3,0,1)` — allow-list + exp/log closed forms + identity tests | 2.4 | ~2 weeks | Unlocks `SE(3)`; shares the `exp_map`/`log_map` contract with the OT plan. Do after 7 so kernels come from the synthesizer. |
| 11 | Tree-reassociate + FMA in the expanded product | 2.5 | ~3 days | Small, local; becomes necessary at dim ≥ 32. |
| 12 | Native AIS / Monte-Carlo partition lanes | 2.7 | — | Do not attempt before 8; it is the same problem. |

Items 1–4 are ~1.5 weeks combined and close two live defects plus a
documentation hazard. That is the block to do first regardless of what happens
with the OT plan.

---

## 5. What this review does not claim

- No generated dashboard row is reclassified. Where the status ledger says
  `hardware-runtime`, the kernels do run on hardware — §1.2 qualifies *what*
  runs there, not *whether*.
- The math is not in question. The Cayley table construction, the sign
  conventions, the Euler-identity `exp_mv` closed form, and the Langevin
  discretizations all read as correct, and the unit tests exercise real
  algebraic identities.
- Effort figures are engineering estimates for a single track with no
  hardware-gated dependencies, not commitments.
