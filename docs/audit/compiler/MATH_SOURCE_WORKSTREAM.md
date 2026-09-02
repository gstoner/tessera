---
last_updated: 2026-09-01
audit_role: plan
plan_state: open
status: MSW-1 LANDED (2026-09-01, PR #679) — reverse-over-reverse autodiff
        fails closed instead of returning a silent zero gradient.
        `Tape.shadowed_buffer_ids` + `consumed_buffer_ids()` + the
        `_forward_closed` gate; consumed per-argument by `grad` and `jacrev`;
        16 regression tests (tests/unit/test_autodiff_nested_tape_guard.py).
        mypy clean (484 files); generated-doc drift gate green.
        MSW-4a OPENED 2026-09-01 by the correctness audit (`codiff` is not the
        codifferential) and now blocks MSW-4. All other items open.
source: T. Sochi, "Principles of Tensor Calculus" (189pp) and "Introduction to
        Tensor Calculus" (its condensed sibling); A. Jentzen, B. Kuckuck,
        P. von Wurstemberger, "Mathematical Introduction to Deep Learning:
        Methods, Implementations, and Theory" (arXiv:2310.20360v3, ships
        runnable reference code)
---

# Math-source workstream — what three texts say the reference lane is missing

> **Routing:** start at [`README.md`](README.md). This document owns the design
> and acceptance criteria for a **host-free reference-lane workstream** derived
> from three supplied mathematical sources. Global ordering lives only in
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md);
> [`../MASTER_AUDIT.md`](../MASTER_AUDIT.md) plus the generated dashboards stay
> status truth (Decision #26). This is a design and build-sequence document,
> not a status claim.

## Why this is a separate plan and not a re-ordering

The integrated plan's 14-item ordered queue is **entirely physical-execution
work**: family/target certificates, multi-block regions, schedule authority,
layout, reshard, native distributed transport, spectral packets, profiler
clocks, evidence packets. Every item in *this* plan is host-free reference-lane
or contract work. There is no contention — not for queue position, not for a
box, and not for the same reviewer's attention. This plan therefore runs
**alongside** Orders 1–14 rather than displacing any of them.

Two exceptions where it must defer rather than proceed alone:

- **MSW-5** (coordinate-aware field calculus) belongs to the owner of
  `PDE-STENCIL-FOUNDATION-1` / [`PDE_STENCIL_CAPABILITY_PLAN.md`](PDE_STENCIL_CAPABILITY_PLAN.md).
  That project already set the precedent this item applies.
- **MSW-7** (contraction normal form) must not create an ODS op ahead of its
  consumer (Decision #29) or a second lowering authority (Decision #31).

## Baseline — what was measured, 2026-09-01

Measured on the Mac; every lane named here is pure Python/numpy, so the results
are host-independent and no device claim is made or implied.

| Fact | Where |
|---|---|
| `grad(grad(f))` returned zeros for a function with a nonzero second derivative | `autodiff/grad.py` zero branch — **fixed, MSW-1** |
| Taylor-mode jets exist (`TruncatedJet(k)`, structured jets) but `autodiff/__init__.py` exports only `register_jet_derived_structured_rules` | `autodiff/jet.py` |
| `laplacian_estimate` is a Hutchinson trace estimate; there is no exact form | `autodiff/jet.py:405` |
| Adagrad, RMSprop, Adadelta, Shampoo, midpoint SGD are absent | `optim.py::__all__` |
| `einsum` appears in **no** `.td` file; `op_catalog` registers `tessera.einsum` with `lowering="contraction"` and Graph IR defines no such op | `src/**/*.td`, `compiler/op_catalog.py:43` |
| `MultivectorField` samples "a uniform Euclidean grid"; `_partial` is `np.gradient(values, h, …)`; zero occurrences of `christoffel` / `covariant_deriv` / `metric_tensor` / `curvilinear` package-wide | `ga/calculus.py` |
| `ga/calculus.integral` correctly refuses a non-Euclidean manifold; `ext_deriv` / `codiff` accept a field carrying no manifold at all | `ga/calculus.py:449` |
| Variance (covariant/contravariant index position) is not among the six canonical tensor attributes | Decision #15a |

## Ordered queue

Order is by what unblocks what, not by size. Items may be taken in parallel
where "Depends on" is empty.

| Order | Work item | Deliverable | Acceptance gate | Depends on |
|---:|---|---|---|---|
| 1 | **MSW-1 — reverse-over-reverse fails closed** ✅ **landed 2026-09-01** | `Tape.shadowed_buffer_ids`: the set of buffer ids an inner `tape()` actually consumed, unioned onto the outer tape **on exit** from `Tape.consumed_buffer_ids()` (literal operands excluded; outputs included). `Tape._forward_closed` distinguishes a shadowing nested tape from the legitimate one a VJP rule opens during backward (`rematerialize`). `grad` and `jacrev` consult it **per argument** in their zero-gradient branches and raise `raise_nested_tape_shadowed`, naming the cause, AD-WEIL-1 and `hvp`. In `jacrev` the structural identity-Jacobian proof is decided BEFORE the shadow evidence. | Landed. 16 tests in `tests/unit/test_autodiff_nested_tape_guard.py`: both defect paths raise; the refused answer is proven wrong by analytic + finite difference; the diagnostic names cause/plan-item/alternative; forward and reverse nesting are symmetric; first-order `grad`/`jacrev`/`hvp` unchanged; a genuinely unused argument still returns zeros **even alongside an unrelated nested tape**; a function whose inner tape swallows `b` but not `a` yields `a`'s real gradient and refuses only `b`; `rematerialize`'s nested tape is not flagged. 1561 autodiff-surface tests pass; mypy clean; drift gate green. | — |

> **Why MSW-1's design changed during review (recorded, per the amendment protocol).** The first implementation set a single `shadowed_by_nested_tape` bool on context-manager ENTRY. That was too coarse: any nested tape poisoned the whole outer pass, so a function that opened an inner tape for an unrelated diagnostic **and** genuinely ignored one argument had that argument's legitimate zero gradient turned into a refusal — trading a false-negative class for a false-positive one. The shipped version asks the narrower question the zero branch actually needs — *was **this** value's path swallowed?* — by recording ids on EXIT from what the inner tape consumed. Both the coarse and precise versions catch the original defect; only the precise one leaves honest zeros alone.
| 2 | **MSW-2 — exact higher-order derivatives are reachable** | Export the jet surface from `tessera.autodiff`. Add `laplacian_exact(fn, x)` — *d* deterministic jet evaluations along the coordinate directions — beside the existing Hutchinson `laplacian_estimate`. Add `jet_trace(fn)` lifting an `ops.*` program into jet arithmetic so a caller need not hand-write `jet_fn`. | `laplacian_exact` matches an analytic Laplacian to fp64 tolerance on at least one closed-form field per supported rank, and matches `laplacian_estimate`'s mean under a fixed Philox key. `jet_trace` order-0 equals the canonical `ops.*` forward and order-1 equals the registered JVP — the anchoring obligation `test_jet_struct.py` already imposes. An op with no jet rule fails closed rather than silently dropping to order 0. | MSW-1 (the guard is what makes the absence of this path visible rather than silently zero). |
| 3 | **MSW-3 — optimizer breadth** | Adagrad, RMSprop (plain + bias-adjusted), Adadelta, Shampoo, midpoint SGD in `optim.py`, each docstring citing its source definition label. Audit the shipped `momentum`/`nesterov` against the source's four distinct momentum formulations and record which one Tessera implements. | Each optimizer reproduces the source's reference implementation trajectory to fp64 tolerance over a fixed number of steps on a fixed quadratic. Registered in `op_catalog` **and** `primitive_coverage` (Decision #24 requires both). `muon` gains a differential test against the source's `def:ideal_Muon` (exact orthogonal polar factor) as a declared oracle (#31). | — |
| 3b | **MSW-4a — `codiff` is not the codifferential** ✅ **landed 2026-09-02** | `ga.calculus.codiff` now applies the per-grade sign `(-1)^(n(k+1)+1)`, exposed as `codifferential_output_signs` so the `clifford_codiff` VJP uses the identical table — `δ = S ∘ ⋆d⋆` with `S` diagonal and self-adjoint, so `δ* = (⋆d⋆)* ∘ S`. The sign is applied in Python to whichever lane produced the result, so the Apple MSL kernel is deliberately left computing the unsigned `⋆d⋆` and the two lanes cannot drift; measured agreement on the interior is 0.000e+00. Non-degenerate Euclidean only: `q > 0` carries an unproven `(-1)^q` metric factor and `r > 0` has no invertible Hodge star, so both are refused (#21a) rather than guessed. | Landed. Adjointness `⟨dα,β⟩ = ⟨α,δβ⟩` holds at 5.5e-11 / 3.9e-10 / 5.1e-09 for k=1,2,3 and **2.4e-10 on a mixed-grade field** — the case no scalar correction could fix. `codiff(v) == -div v` exactly (0.00e+00). AD law sweep: `clifford_codiff adjoint` 3.4e-16, `chain` 1.8e-11, 577 pass / 0 fail. 964 passed on the Mac GA surface (Apple GPU included), 1454 on Princess-Luna; mypy clean. | — |

> **Cross-host verification (2026-09-01) — the finding is host- and dtype-independent, and the fix has two sites.** Measured identically on all three fleet boxes: Mac (numpy 2.5.2), Princess-Luna (numpy 2.1.3, HIP loadable), The-Super-Bear (numpy 2.1.3, RTX 5070). Every one reports ⋆⋆ = +1 per grade, d∘d = 2.22e-16, and the adjointness ratio (−1)^k at k = 1, 2, 3 — in fp64 **and** fp32.
>
> **There is no CUDA or HIP `codiff` to be wrong.** The only native kernels are Apple's (14 references in `apple_gpu_runtime.mm`), and `backend_manifest` declares `clifford_codiff` for `apple_gpu` alone; every other host falls through to the arch-agnostic numpy composition. So this is one defect in shared Python, not a per-backend divergence.
>
> **But Apple is a second site, and it currently AGREES.** Instrumenting the dispatch confirms f32 Cl(3,0) really does take `tessera_apple_gpu_clifford_codiff_cl30_f32`, and that kernel returns the same ⋆d⋆ convention as numpy. That agreement is the thing to preserve: fixing the Python composition without the MSL kernel would make the two paths disagree by (−1)^k, turning a consistent wrong answer into a host-dependent one. Fix both, and gate both.
| 4 | **MSW-4 — vector-identity law family** | `vector_identity_check` in `autodiff/laws.py`, alongside the existing `adjoint_check` / `chain_check` / `hessian_symmetry_check`. Covers ∇×(∇f) = 0, ∇·(∇×A) = 0, ∇·(fA) = f∇·A + A·∇f, and the GA-native forms **d∘d = 0** and the codifferential adjointness **⟨dα, β⟩ = ⟨α, δβ⟩** (Stokes). | Every law is reference-free (no external oracle) and runs host-free. Mutation-tested: a deliberately corrupted `ext_deriv` / `codiff` fails the law. Laws are declared in the law registry so an unswept rule is visible, matching the AD-LAW-1 sweep discipline. | MSW-4a (landed 2026-09-02) — the Stokes law is now statable; `test_codiff_is_adjoint_to_ext_deriv` is its first instance. |
| 5 | **MSW-5 — coordinate-aware field calculus** | A coordinate/metric parameter on `MultivectorField`, defaulting to Cartesian **with a recorded reason** rather than silently. Orthogonal-coordinate grad/div/curl/Laplacian (scale factors `h_i`, √g weights) with cylindrical and spherical as the first concrete systems. | A field transformed into cylindrical/spherical coordinates and differentiated there agrees with the Cartesian computation of the same physical field. A field carrying no coordinate declaration where one is required fails closed with a named diagnostic — the discipline `PDE-STENCIL-FOUNDATION-1` already states as "none is manufactured by legalization" (#21a). MSW-4's laws hold in the non-Cartesian systems too, not just Cartesian. | MSW-4; routed through the `PDE-STENCIL-FOUNDATION-1` owner. |
| 6 | **MSW-6 — `examples/tensor_calculus/`** | A runnable tutorial in the established `examples/matrix_calculus/` mould: checks every number it prints, no device, no build. Kronecker/Levi-Civita identities verified numerically; the ε-δ identity shown as a rewrite with a before/after operation count; div/grad/curl/Laplacian in three coordinate systems checked against each other; the vector identities as assertions. | Runs on any host from a clean checkout with `PYTHONPATH=python python3 …`. Every printed claim is computed, not asserted. Registered in the examples CMake surface like its sibling. | MSW-5 for the coordinate sections; the identity sections need only MSW-4. |
| 7 | **MSW-7 — contraction normal form** | A canonical form for contraction specs on the Python side (`op_catalog` / the einsum surface), with the δ/ε identities as its corpus: index-replacement (`δ_ij A_j = A_i`) as copy-elision, identity-chain collapse (`δ_ij δ_jk = δ_ik`, `δ_ii = n`), the ε-δ expansion that removes a rank-3 materialization, and antisymmetry as a canonical index ordering. | Two einsum specs denoting the same contraction but spelling indices differently normalize to one key — the prerequisite for CSE across differently-spelled einsums. Each rewrite is value-preserving under a randomized differential test. **No ODS op is added by this item**: a `tessera.einsum` op waits for a named consuming pass (#29), and must not become a second lowering authority (#31). | — |
| 8 | **MSW-8 — `examples/pde_learning/`** | A Tessera reimplementation of the source's PINN and Deep Kolmogorov chapters (its `code/pinn.py` solves `u_t = 0.005·Δu + u − u³` on a 2-D domain in 111 lines of PyTorch). Exercises `nn`, `optim`, `autodiff` and RNG end to end. Decision #23: a reimplementation, with the book as reference vocabulary — no runtime dependency on the source's framework. | The learned solution matches a numerical reference PDE solve to a stated tolerance at several times. The Laplacian comes from MSW-2's exact path, not finite differences. This example is the forcing function for MSW-1/MSW-2: it could not be written before them. | MSW-1, MSW-2, MSW-3. |
| 9 | **MSW-9 — ANN-calculus laws over the feed-forward fragment** | Encode the source's network algebra as executable semantics-preservation laws: functoriality of realization under composition, associativity of composition, the parallelization realization law, ReLU identity networks and extension/enlargement, sums of same-length networks, and the closed-form parameter/length arithmetic. Wire the value laws into `evaluator.metamorphic_equivalence` as a law family; wire the arithmetic as a **self-check on fusion bookkeeping** (Decision #30, "derive, don't ask"). | Each law is stated over the feed-forward fragment of Graph IR **where it is a theorem**, and the restriction is recorded — the source proves these for plain feed-forward ReLU/Swish stacks, not for attention or normalization, and the plan must not claim otherwise. A fusion pass whose post-transform parameter accounting disagrees with the closed form fails. Mutation-tested. | MSW-4 (law-registry pattern); wants its own design spike before estimation. |

## Fleet routing

Every item is host-free. MSW-8 optionally benefits from a device for training
time but requires none for correctness, and no device claim may be made from
it. Per Decision #26 and the Working Rules, a result produced on a host without
a given device is not evidence for that device — this plan produces no
architecture evidence at all, by construction.

## Risks

| Risk | Item | Mitigation |
|---|---|---|
| MSW-7 grows an ODS op ahead of a consumer, manufacturing exactly the unconsumed declaration #29 forbids | 7 | The acceptance gate names this explicitly: Python-side normal form only; ODS waits for a pass that reads it. |
| MSW-9's laws get claimed beyond the fragment where they are proved | 9 | The restriction is part of the deliverable, not a caveat added later. Attention/normalization forms would need their own proofs. |
| MSW-5 duplicates discretization contracts `PDE-STENCIL-FOUNDATION-1` already owns | 5 | Routed to that plan's owner rather than run standalone. |
| MSW-3 adds optimizers to `optim.py` without the matching registry rows | 3 | Decision #24 requires `op_catalog` **and** `primitive_coverage`; the gate names both. |

## Correctness audit — 2026-09-01

A numerical audit of the reference lane ran alongside MSW-1: every identity was
computed against an independent closed form or defining property, on the Mac,
on host-free lanes only (no device claim is made or implied). Four findings,
routed by owner:

| # | Finding | Routed to |
|---|---|---|
| M-1 | `codiff` is not the codifferential — off by a grade-dependent (−1)^k | **MSW-4a** above |
| M-2 | **Root cause corrected 2026-09-01 (same day).** The full sweep hung uninterruptibly in `tessera_apple_gpu_mlpkg_dispatch` → `waitUntilSignaledValue:timeoutMS:` → `iokit_user_client_trap` — but the cause was a **two-day-stale `libTesseraAppleRuntime.dylib`**, not the dispatch. Against a fresh dylib the same selection is 115 passed in 32 s, where it had hung twice for 16 and 14 minutes. What survives is that a stale runtime must fail cleanly rather than block unpreemptably. | [`../backend/apple/todo.md`](../backend/apple/todo.md) — backend-owned, reduced to P1 |
| M-3 | 20 of 34 loss functions carry no docstring, including `kl_divergence`, whose argument order is the **inverse** of PyTorch's `kl_div` (Tessera computes KL(p‖q) from `p_log_probs`, verified exact) | folded into MSW-3's docstring obligation |
| M-4 | Every optimizer raises a raw `KeyError` on `state={}`; `None` is the documented init | folded into MSW-3 |

**What the audit confirmed exact** (recorded so the next reader does not re-derive
it): d∘d = 0 at 8.7e-15 and ⋆⋆ = +1 per grade; every `metric.py` manifold
property including first-order retraction and the Riesz representative; the
Cauchy integral formula, winding numbers, residues, the argument principle and
the residue theorem; Weierstrass ℘ converging at exactly its documented
O(1/cutoff²) with g₃ = −1.2e-15 on a square lattice; Adam against an independent
12-step reference at 6.7e-9, AdamW's decoupling bit-exact, Muon's update
singular values [1,1,1,1]; the loss closed forms including Fenchel–Young's zero
at the optimum; every Philox sampler against closed-form moments; Cholesky/QR/SVD
at 1.8e-15; softmax stable at ±1000 logits; and `run_law_sweep()` at 577 pass /
0 fail.

## Provenance

The three sources were reviewed 2026-09-01 against the compiler, the reference
library, the law suite and the samples. The review that produced this queue
recorded eleven findings; the mapping is MSW-1 ← D-1, MSW-2 ← D-2, MSW-3 ← D-3,
MSW-4 ← X-1, MSW-5 ← T-3, MSW-6/MSW-8 ← X-2, MSW-7 ← T-1 + T-2, MSW-9 ← D-4.

Two findings are deliberately **not** queued as work:

- **Index variance** (covariant/contravariant index position) is absent from
  Decision #15a's six canonical tensor attributes. For orthonormal Cartesian
  work the distinction collapses (`δ_ij = δ^ij`), which is why no deep-learning
  workload has missed it. Building it before a pass consumes it is #29's error.
  Record it as a deliberate omission in
  [`../../reference/tessera_tensor_attributes.md`](../../reference/tessera_tensor_attributes.md)
  with the condition that reopens it: a lane operating in a genuinely
  non-orthonormal basis.
- The source's **backpropagation** and **batch-normalization** chapters are
  citable specifications rather than gaps — the first for `NONSMOOTH_SELECTION`
  and `laws.kink_check`'s ReLU-at-zero convention, the second for the
  train/inference running-statistics correspondence that `nn.BatchNorm1d` must
  honour. Cite them where those contracts are documented; no new work item.
