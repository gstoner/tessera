---
last_updated: 2026-08-22
audit_role: scoped_plan
---

# Functional-Analysis Operator Contracts for TSOL — Architected Plan

> **Status:** scoped plan (design + phased workstreams), not status truth.
> `MASTER_AUDIT.md` and the generated dashboards remain the current-status
> authority (Decision #26), and `INTEGRATED_COMPILER_PLAN.md` owns cross-plan
> ordering — this document proposes FA-series workstreams; it does not insert
> them into the queue.
>
> **Source text:** Jan van Neerven, *Functional Analysis* (arXiv 2112.11166v7).
> Chapter references below are to that book. Rounding-error constants come
> from the standard Higham-style model; the book supplies the operator/norm
> framework, not the floating-point constants.
>
> **Terminology:** TSOL = Tessera Standard Operator Library
> (`docs/operations/Tessera_Standard_Operations.md`,
> `python/tessera/compiler/tsol_coverage.py`).

---

## 0. Purpose And Scope

TSOL currently specifies operators *operationally*: canonical name, category,
`NumericPolicy` (storage/accum/rounding/scale/quant_axis/deterministic/
math_mode/scale_layout), an effect, and lowering ownership. What it does not
specify is **how much a lowering is allowed to change the operator** — the
accuracy budget of Decision #28's measured arbiter is a scalar tolerance with
no attached norm, no shape dependence, and no compositional rule.

This plan adds that layer: a small family of **mathematical models (M1–M7)**
that treat TSOL operations as operators between normed vector spaces, and a
set of **workstreams (FA-1 … FA-7)** that turn each model into a registry
contract, an evaluator oracle, or an IR-carried attribute — each with a named
consumer (Decision #29) and a drift gate.

### 0.1 The finite-dimensional honesty clause

Every tensor a TSOL op touches lives in a finite-dimensional space, and on a
finite-dimensional normed space **every linear operator is bounded** (book
Ch. 1). So the value of functional analysis here is *not* "boundedness" as a
qualitative fact. It is three quantitative things the finite-dimensional
setting does not give you for free:

1. **Uniformity over shape families.** A TSOL op is not one operator; it is a
   family `{T_s : s ∈ ShapeBucket}` indexed by shapes, and the family is
   infinite. Per-shape error constants can grow without bound in the shape
   parameter (e.g. naive-summation error grows linearly in the reduction
   length K). The Uniform Boundedness Theorem (Ch. 5) is the conceptual
   ancestor: pointwise-fine families can be family-level bad. Our contracts
   therefore always bound `sup` over the bucket, with **explicit shape
   dependence** in the bound.
2. **Conditioning and non-normality.** Norm equalities and stability
   statements that are exact for normal/self-adjoint operators (Ch. 8–9)
   degrade controllably — by `κ(V)` or pseudospectral factors — for
   non-normal ones. The contracts state which regime they are in.
3. **Structure theorems as contracts.** Adjoint = VJP, Plancherel = energy
   conservation, dissipativity = recurrence stability, Eckart–Young–Mirsky =
   optimal low-rank truncation. Each is an *exactly checkable identity or
   inequality*, which is a stronger conformance instrument than tolerance
   sampling.

### 0.2 Non-goals

- **No infinite-dimensional machinery for its own sake.** Closed Graph, Open
  Mapping, Baire category, weak topologies: conceptual background only; no
  workstream depends on them.
- **No claim that nonlinear TSOL ops become linear-operator theory.** `gelu`,
  `softmax`, `layer_norm` enter only through (a) Lipschitz bounds and (b)
  their autodiff linearizations, which *are* linear operators per input point.
- **No device claims.** Every deliverable below is host-independent
  (numpy-level reference + registry + oracle code). Native-execution status
  stays with the backend target maps.

---

## 1. The Formal Model

### 1.1 Spaces and norms

Fix a shape `s` and dtype; a tensor is a point in `X_s = ℝ^{n(s)}` for the
real-dtype core. **Scalar-field rule:** M3 (Fourier) and M4 (recurrences with
complex/rotational parameterizations, e.g. complex-diagonal SSM `A`) work
over ℂ; every transpose in those models is the conjugate transpose `A*`
(for real `A`, `A* = Aᵀ`, so the real statements are the special case).
Norms used, and where:

| Norm | Definition | Used by |
|---|---|---|
| `‖x‖₂` (Euclidean / Frobenius on matrices) | `(Σ xᵢ²)^{1/2}` | default input/output norm; Plancherel; HS norm = Schatten-2 (Ch. 14) |
| `‖T‖₂→₂` operator norm | `sup_{x≠0} ‖Tx‖₂/‖x‖₂` = `σ_max(T)` | lowering error (M1), adjoint law (M2), multiplier bound (M3) |
| Schatten-p, `‖A‖_{S_p} = (Σ σᵢᵖ)^{1/p}` | p ∈ {1 (nuclear/trace), 2 (HS/Frobenius), ∞ (operator)} | low-rank policy (M6); Ch. 14 |
| Logarithmic norm `μ₂(A) = λ_max((A+A*)/2)` | matrix version of dissipativity / numerical range in `{Re ≤ μ}` | recurrence stability (M4); Ch. 13 (Lumer–Phillips) |

Standing inequalities the contracts rely on (all standard; Ch. 14 supplies
the Hilbert–Schmidt (p=2) and trace-class (p=1) instances and their ideal
properties — the book does not treat general Schatten-p, which is standard
material outside it):

```
‖A‖_{S_∞} ≤ ‖A‖_{S_2} ≤ ‖A‖_{S_1}          (Schatten monotonicity)
‖AB‖_{S_p} ≤ ‖A‖_{S_∞} ‖B‖_{S_p}            (ideal property)
‖A‖_{S_2}² = tr(AᵀA)                        (HS inner product; trace duality)
```

### 1.2 Operators, families, and lowerings

- A TSOL op at fixed shape/dtype/attribute point is a map
  `T_s : X_s → Y_s` (linear for the Linear-Algebra/Spectral/Layout/Collective
  core with fixed weights and fixed second operand; nonlinear otherwise).
- A **lowering candidate** (Tier-1 synthesized, Tier-2 plugin, Tier-3
  hand-tuned; Decision #28) is a map `T̃_s : X_s → Y_s` on the same spaces.
- A **shape bucket** `B` is the arbiter's symbolic-dim bucket; the object the
  arbiter scores is the family `{T̃_s : s ∈ B}`.

### 1.3 Error functionals

For linear `T`:

```
ε_op(T̃, T; s)  =  ‖T̃_s − T_s‖₂→₂ / ‖T_s‖₂→₂          (relative operator-norm error)
ε_op(T̃, T; B)  =  sup_{s ∈ B} ε_op(T̃, T; s)           (bucket error — the contract object)
```

For nonlinear `T` on a stated admissible input domain `D_s` (bounded, e.g.
`‖x‖₂ ≤ R` or a normalization-implied set):

```
ε_dom(T̃, T; s) = sup_{x ∈ D_s} ‖T̃_s(x) − T_s(x)‖₂ / scale(x)
```

with `scale(x)` chosen per-op (absolute for softmax outputs, relative
`‖T_s(x)‖₂` when `T_s(x)` is bounded away from 0 on `D_s`). The domain `D_s`
is part of the contract — an unbounded-domain sup is meaningless for
floating-point nonlinearities and MUST NOT be claimed.

### 1.4 Composition law (fusion epilogues)

If `g, g̃ : X → Y` and `f, f̃ : Y → Z` with `f` Lipschitz constant `L_f` on a
domain containing both `g(D)` and `g̃(D)`, then for `x ∈ D`:

```
‖f̃(g̃(x)) − f(g(x))‖ ≤ ‖f̃(g̃(x)) − f(g̃(x))‖ + ‖f(g̃(x)) − f(g(x))‖
                     ≤ ε_f + L_f · ε_g                                  (C1)
```

Two hypotheses that (C1) consumes and that MUST be enforced when composing
contracts: (a) `ε_f` is the sup of `‖f̃ − f‖` over a domain **containing
`g̃(D)`** — if `f`'s contract domain `D_s(f)` does not contain the perturbed
intermediate values, the first term is uncontrolled and (C1) does not apply;
(b) (C1) is stated in **absolute** deviations — a relative `ε_dom`
(§1.3's `scale(x)` division) must be multiplied back to absolute scale
before entering (C1). FA-1 implements (C1) over absolute sups only.

(C1) is the budget-propagation rule for fused epilogues: a fused
matmul+bias+activation kernel's budget decomposes into per-stage budgets with
the downstream Lipschitz constants as weights. Lipschitz constants the plan
commits to (each with a one-line proof obligation in FA-1):

| Op | Lipschitz bound (ℓ₂→ℓ₂) | Note |
|---|---|---|
| `relu` | 1 | metric projection onto the nonnegative orthant; projections onto convex sets are nonexpansive in ℓ₂ |
| `softmax` | ≤ 1 | Jacobian `J = diag(σ) − σσᵀ` satisfies `0 ⪯ J ⪯ diag(σ) ⪯ I`; tighter constants exist in the literature but are not needed |
| `gelu` | ≤ 1.13 | `sup |gelu′|` bounded numerically with an interval-arithmetic check; do not hand-wave the constant |
| `silu` | ≤ 1.1 | same treatment as gelu |
| linear `T` | `‖T‖₂→₂` | by definition |

### 1.5 Floating-point instantiation (where the constants come from)

The norm framework does not produce numerical constants; the standard model
of floating-point does. For unit roundoff `u` and reduction length `K`
(recursive summation): the classical backward-error bound for an inner
product is

```
|fl(xᵀy) − xᵀy| ≤ γ_K |x|ᵀ|y|,   γ_K = Ku / (1 − Ku)     (Higham)      (F1)
```

valid under the standard model's hypotheses: **`Ku < 1`** and no
underflow/overflow. The `Ku < 1` condition is not decorative — for fp16
accumulation (`u ≈ 4.9e−4`) it fails at `K ≈ 2048`, squarely inside real
reduction lengths; when the model is void the contract MUST fail closed
(no `τ` is derivable — the op needs a wider accumulator or a tree
reduction, both expressible in `NumericPolicy`), never extrapolate. (F1)
lifts to a per-entry matmul bound, hence to an operator-norm bound of
the shape `ε_op ≲ c(K) · u · κ_rows` with `c(K) = K` for recursive
accumulation and `c(K) = O(log K)` for pairwise/tree accumulation. **This is
the concrete content of §0.1(1):** the `accum` and `deterministic` fields of
`NumericPolicy` select `c(K)`, and the bucket-level contract makes the `K`
dependence explicit rather than absorbing it into a flat tolerance.

---

## 2. Model M1 — Lowering Admissibility As Operator-Norm Budget

**Claim.** A lowering candidate is admissible for `(op, bucket B, dtype,
target)` iff `ε_op(T̃, T; B) ≤ τ(op, policy)` where `τ` is derived from the
op's `NumericPolicy` via (F1)-style bounds, not chosen ad hoc.

**Estimators (computable, host-independent):**

- *Linear ops, moderate size:* `‖T̃_s − T_s‖₂→₂` by power iteration on
  `Δ = T̃_s − T_s` using only matvec access (`Δx` computed by running both
  lowerings). Convergence certificate: report the Rayleigh quotient gap; a
  power-iteration estimate is a **lower bound** on `‖Δ‖`, so admissibility
  decided from it alone can be optimistic. Pair it with the inequality
  `‖Δ‖₂→₂ ≤ ‖Δ‖_F = ‖Δ‖_{S_2}`, where `‖Δ‖_F²` is estimated by Hutchinson
  probes (`E‖Δz‖₂² = ‖Δ‖_F²` whenever `E[zzᵀ] = I`, e.g. Rademacher or
  standard Gaussian `z`). **An unbiased estimate is not an upper bound**: a
  finite-probe average can undershoot, so the upper side of the bracket is
  statistical and must carry a one-sided concentration bound or a declared
  quantile inflation (Rademacher-probe tail bounds are the standard tool).
  The bracket is therefore: deterministic lower bound (power iteration) +
  confidence-qualified upper estimate (inflated Hutchinson) — never
  reported as two deterministic bounds.
- *Bucket sup:* evaluated on the bucket's **componentwise-max corner** (each
  monotone parameter of the model bound at its bucket maximum — for gemm
  that is max-M, max-N, max-K jointly) plus random interior samples. The
  analytic shape-dependence of (F1) implies the sup of the *model* bound
  sits at that corner; corner + interior sampling validates the model's
  constant, but cannot exclude a shape-localized implementation defect
  (e.g. a tile-boundary bug at an interior K) — which is exactly why the
  resulting status is `model_bounded, sample-validated` and not more.
- *Nonlinear ops:* `ε_dom` by quasi-Monte-Carlo over `D_s` plus the (C1)
  decomposition when the op is a composition of contract-carrying stages.

**Consumers (Decision #29):** the arbiter's accuracy gate
(`COMPILER_REFACTOR_PLAN.md`) and the conformance evaluator's vertical
oracle (`EVALUATOR_PLAN.md` §9.5). The FA-1 registry axis is the
*declaration* these consumers read — per Decision #29 it is not itself a
consumer, and it does not land before they do.

**Honest limitation.** Sampling + model-validated extrapolation is not proof.
The contract language is "model-bounded, sample-validated at bucket
extremes", recorded as such in the registry status — never "proven".

---

## 3. Model M2 — The Adjoint Law For Autodiff (Ch. 4, 5, 8)

For a (real) linear TSOL op `T`, the VJP is the adjoint: `VJP_T(y) = Tᵀy`.
Two exactly checkable laws follow:

```
⟨Tx, y⟩ = ⟨x, Tᵀy⟩      ∀x, y                    (A1 — bilinear identity)
‖T‖₂→₂ = ‖Tᵀ‖₂→₂                                  (A2 — norm equality, Ch. 4/8)
```

**Contract.** Every `_VJPS` entry for a linear op must pass a randomized (A1)
test: draw `(x, y)` pairs, check
`|⟨Tx,y⟩ − ⟨x, VJP_T(y)⟩| ≤ γ-model tolerance · ‖x‖‖y‖‖T‖`. This
strengthens the existing `check_grad` finite-difference machinery: (A1) is
exact in exact arithmetic (no `O(h)` truncation term), so its tolerance is
pure rounding, orders tighter than a finite-difference tolerance. (A2) is a
cheaper smoke check via power iteration on both `T` and `VJP_T`.

For nonlinear ops the same law applies to the *linearization*: at input `x₀`,
`VJP(x₀, ·)` must be the adjoint of `JVP(x₀, ·)` — this is a per-point (A1)
test relating the two registries `_VJPS` and `_JVPS`, catching
transpose-convention bugs (the `LinearTransposeInterface` wgrad path is
exactly this shape).

**Consumers:** `tessera.debug.check_grad` (new mode), the coverage registry's
`vjp`/`jvp` axes (a rule may only be auto-flipped to complete when the (A1)
harness covers it — tightening the Decision #26 caveat that registration
alone is not proof).

---

## 4. Model M3 — Isometry And Multiplier Contracts For The Spectral Family (Ch. 5)

**Plancherel contract.** The DFT with the convention Tessera's `fft` actually
implements satisfies an exact energy identity with a known constant:

```
‖F x‖₂² = c_N ‖x‖₂²          (c_N = N for unnormalized, 1 for unitary)   (P1)
F⁻¹ = (1/c_N) F*                                                          (P2)
```

The normalization is **not an implementation constant** — Tessera's `fft`
takes a per-call `norm` argument (`backward`/`ortho`/`forward` semantics),
so `c_N` is a per-op-instance attribute read from that argument, and the
(P1)/(P2) oracles branch on it per instance. With that, they become oracle
assertions on every `fft/ifft/rfft/irfft/stft/istft` lowering, with
tolerance from the (F1) model (`O(u log N)` for a radix FFT). An `istft(stft(x)) = x` round-trip inherits
the same treatment with window-COLA conditions stated as a precondition, not
assumed.

**Multiplier contract.** `spectral_conv` / `spectral_filter` are Fourier
multipliers `T_m = F⁻¹ diag(m) F`. Every standard DFT convention is
`F = c·U` with `U` unitary, and the scalar cancels in the similarity, so

```
‖T_m‖₂→₂ = ‖m‖_∞                                                          (P3)
```

holds **exactly for every normalization convention — no `c_N` factor enters
(P3), and none may be inserted**. The hypotheses that actually matter:
(a) the forward/inverse pair composed in the lowering must be exact mutual
inverses — a mismatched pair (e.g. `F*` without the `1/c_N`) is not
`F⁻¹ diag(m) F` and does pick up a spurious factor; (b) on the
`rfft/irfft` half-spectrum lane the operator is not scalar-times-unitary —
(P3) survives there only for a matched pair with **Hermitian-symmetric
`m`** (real-output multipliers), which the oracle checks.

So carrying the symbol `m` (or just `‖m‖_∞`) as IR metadata gives the
compiler the operator norm of the whole op *for free* — feeding M1's budget
and (C1)'s composition weights with an exact constant instead of an estimate.
Decision #32 applies: a lowering that drops the symbol must record why.

**Consumers:** conformance evaluator (energy oracle), fusion cost/error model
in `fusion_core.py` (multiplier norm as the Lipschitz weight).

---

## 5. Model M4 — Recurrence Stability For The Sequence-Mixer Track (Ch. 13)

The `linear_recurrence` op (SEQUENCE_MIXER_THEORY.md) computes
`h_{t+1} = A_t h_t + B_t x_t` (possibly gated/diagonal/structured `A_t`).
Continuous-time SSM parameterizations pass through `h' = A h + B x` and a
discretization. The book's semigroup chapter gives the clean stability
criteria; the discrete-time subtleties below are where naive spectral
reasoning fails and MUST be part of the contract.

**M4.a — Continuous generator (Lumer–Phillips, Ch. 13).** For matrices,
`A` generates a contraction semigroup iff `A` is dissipative iff the
logarithmic norm satisfies

```
μ₂(A) = λ_max((A + A*)/2) ≤ 0      ⟺      ‖e^{tA}‖₂→₂ ≤ 1  ∀t ≥ 0     (S1)
```

`μ₂` is computable (extreme eigenvalue of a symmetric matrix; for the
diagonal-`A` SSM case it is just `max Re aᵢᵢ`). (S1) is a *sufficient and
necessary* contraction certificate — strictly stronger than the eigenvalue
condition `Re λ ≤ 0`, which permits arbitrarily large transient growth for
non-normal `A`.

**M4.b — Exact-discretization transfer.** If (S1) holds, then the
zero-order-hold operator `e^{ΔtA}` is a contraction for every `Δt > 0` —
stability of the *exact* discretization is inherited with **no step-size
condition**. Approximate discretizations do not inherit it automatically:
bilinear/Tustin `(I − Δt/2 A)⁻¹(I + Δt/2 A)` does — dissipativity gives
both invertibility of the resolvent factor and
`‖(I+hA)x‖² − ‖(I−hA)x‖² = 4h⟨Ax,x⟩ ≤ 0`, hence `‖·‖₂→₂ ≤ 1` (the Cayley
transform; standard A-stability material, not in the book) — while
forward-Euler `I + ΔtA` does **not** (skew `A` gives
`‖I+ΔtA‖ = √(1+Δt²‖A‖²) > 1`; needs `Δt`-dependent conditions). The contract
therefore keys on the discretization method stored with the op.

**M4.c — Discrete-time operators directly.** When the op is parameterized
directly by the discrete `A`, the honest hierarchy is:

```
‖A‖₂→₂ ≤ 1                 ⟹ contraction (uniform, all t)         — strongest, checkable
ρ(A) < 1 with normal A     ⟹ ‖Aᵗ‖ = ρ(A)ᵗ                         — clean decay
ρ(A) < 1, non-normal A     ⟹ decay eventually; transients governed by the
                              Kreiss constant K(A):
                              K(A) ≤ sup_t ‖Aᵗ‖ ≤ e·N·K(A)         — the e·N
                              dimension factor makes any Kreiss/κ(V)
                              certificate SHAPE-DEPENDENT across a bucket
                              (the §0.1(1) uniformity concern applies)
ρ(A) = 1                   ⟹ power-bounded ⟺ every |λ| = 1 eigenvalue is
                              semisimple (defectiveness strictly below the
                              unit circle is harmless); diagonalizable with
                              bounded κ(V) is the checkable SUFFICIENT
                              certificate (‖Aᵗ‖ ≤ κ(V)); unitary A gives
                              exact norm preservation (Stone analogue)
```

**Contract (semantic key, Decision #21a).** `linear_recurrence` admission
requires a declared `stability_class ∈ {contraction, normal_stable,
kreiss_bounded(M), unitary, unchecked}` with the corresponding certificate
(`μ₂ ≤ 0` for ZOH-from-continuous, `‖A‖ ≤ 1`, `κ(V)·ρ`-bound, or
`A*A = I`). `unchecked` is legal but is a *declared* absence — it fails
closed in deterministic long-context conformance runs rather than silently
passing. Gating/selectivity (time-varying `A_t`) uses the product bound
`‖Π A_t‖ ≤ Π ‖A_t‖`, which is exactly why per-step `‖A_t‖ ≤ 1`
(contraction class) is the composable choice for gated mixers.

**Consumers:** sequence-mixer engineering plan (admission gate), evaluator
long-context metamorphic oracle, and the registry row for
`linear_recurrence` when it lands.

---

## 6. Model M5 — A `functional_calculus` Primitive (Ch. 6, 8, 9)

**Proposal.** Admit (TSOL-A1 path) a primitive
`functional_calculus(A, f, spec)` computing `f(A)`. Instances: a **new**
`matrix_exp` (M4's reference path for `e^{ΔtA}` — no op of that name exists
in the repo today, so this is an admission, not a re-expression),
`inverse_sqrt` (whitening/orthogonalization — the Newton–Schulz iterations
used by orthogonalizing optimizers are an existing reference in `optim.py`),
`matrix_inverse`, and the **existing** dense `spectral_filter` re-expressed
as an instance.

**The exact contract, stated with its hypotheses (this is where sloppiness
would be dangerous):**

```
A normal (AᵀA = AAᵀ; incl. symmetric/skew/unitary):
    ‖f(A)‖₂→₂ = max_{λ ∈ σ(A)} |f(λ)|                    (FC1 — Ch. 8/9, exact)
    hypotheses: f is defined on σ(A) ⊂ ℂ (for real normal A with non-real
    spectrum — skew, rotation blocks — this REQUIRES a complex extension of
    f; a real scalar function alone is not meaningful there), and f(A) is a
    real matrix iff f is conjugate-symmetric, f(λ̄) = conj f(λ) — which
    holds for real-coefficient polynomial/rational approximants and is a
    checked precondition when comparing against a real reference path

A diagonalizable, A = V Λ V⁻¹:
    ‖f(A)‖₂→₂ ≤ κ₂(V) · max_{λ ∈ σ(A)} |f(λ)|            (FC2 — κ-degraded)

A general, f holomorphic on a neighborhood of the ε-pseudospectrum Λ_ε(A):
    ‖f(A)‖₂→₂ ≤ (L_ε / 2πε) · sup_{z ∈ Λ_ε(A)} |f(z)|    (FC3 — Ch. 6 holomorphic
                                                            calculus + resolvent bound;
                                                            L_ε = contour length)
```

The `spec` argument declares which regime is claimed (`normal`,
`diagonalizable(κ_max)`, `holomorphic(ε)`), and the runtime/oracle checks the
hypothesis (normality residual `‖AᵀA − AAᵀ‖_F`, or κ estimate) **before**
using the corresponding bound. Regime is a semantic key: no silent default
to `normal`.

**Why this is a structural win:** one primitive + three graded norm bounds
replaces N hand-lowered matrix functions each carrying an ad-hoc tolerance;
the bound feeds M1's budget directly; and the polynomial/rational
approximation degree used by a lowering (Chebyshev for `f` on a real
spectral interval, scaling-and-squaring for `exp`) gets its error budget from
`sup |f − p|` on the spectral set — a scalar approximation problem, fully
decoupled from the matrix.

**Scope guard:** this unifies *matrix-argument* spectral ops. Elementwise
nonlinearities (`gelu` on tensor entries) are NOT functional calculus on a
matrix and stay in the Lipschitz lane (§1.4). Data-dependent normalizations
(`softmax`, `layer_norm`) are not `f(A)` instances either.

---

## 7. Model M6 — Schatten Policy For Low-Rank And Attention (Ch. 14)

**Eckart–Young–Mirsky (Mirsky 1960 — standard result, not in the book;
Ch. 14 supplies the singular-value/SVD framework it lives in):** valid in
every unitarily invariant norm, in particular all Schatten-p. A truncated
SVD `A_r` is **a** best rank-r approximation (unique iff `σ_r > σ_{r+1}`),
with error `‖A − A_r‖` = tail of the singular values in the corresponding
norm — so an oracle asserts the *tail error is achieved*, never that the
substituted factor equals `A_r`:

```
‖A − A_r‖_{S_∞} = σ_{r+1}         ‖A − A_r‖_{S_2}² = Σ_{i>r} σᵢ²
‖A − A_r‖_{S_1} = Σ_{i>r} σᵢ                                             (E1)
```

**Contract.** Ops whose lowering may substitute a low-rank/factorized form
(`factorized_matmul`, low-rank attention variants, MoE expert
approximations) declare a `schatten_policy = (p, budget)` extension of
`NumericPolicy`: *which* Schatten norm the substitution must respect and by
how much. The Schatten chain (§1.1) makes the policy ordered: an `S_1`
budget implies the same `S_2` and `S_∞` budgets, never conversely — a
Frobenius-close approximation can still be nuclear-far, which is the
silent-degradation mode of low-rank attention. Downstream error then flows
through (C1) using `‖·‖_{S_∞}` of the substituted factor as the Lipschitz
weight.

**Trace instruments (Ch. 14).** Trace duality `tr(AᵀB) = ⟨A, B⟩_{HS}` and
Lidskii's theorem back two cheap oracles: Hutchinson trace probes for
`‖Δ‖_{S_2}` (already used in M1) and exact-trace checks for lowerings that
preserve diagonals. The partial trace formalizes marginalizing `einsum`
reductions and is contractive `S_1 → S_1` — usable as a sanity bound on
segment/einsum reductions, recorded as a note on those rows rather than a
new mechanism.

---

## 8. Model M7 — Forms And Coercivity For The Stencil/PDE Lane (Ch. 11, 12) — DEFERRED

For the Phase-7 neighbors dialect and `PDE_STENCIL_CAPABILITY_PLAN.md`: a
discretized elliptic problem `a(u, v) = ⟨f, v⟩` with bounded (`|a(u,v)| ≤
M‖u‖‖v‖`) and coercive (`a(u,u) ≥ α‖u‖²`) form has a unique solution with
`‖u‖ ≤ ‖f‖/α` (Lax–Milgram, Ch. 11/12) — and `M/α` bounds the relevant
condition number, i.e. the iteration count/precision requirement of the
solve. The workstream (FA-7) is deferred until the neighbors dialect has an
executing consumer; recorded here so the door is architected, not ad hoc.
Unbounded operators (Ch. 10) enter Tessera **only** through this form-based
door; nothing else in the plan touches them.

---

## 9. Architecture — Where Each Model Lands

```
                 ┌────────────────────────────────────────────────┐
                 │  primitive_coverage.py  (+ tsol_coverage.py)   │
   FA-1 ───────► │  new axis: error_contract                      │
                 │  status ∈ {none, lipschitz_declared,           │
                 │            model_bounded, oracle_validated}    │
                 └───────────────┬────────────────────────────────┘
                                 │ consulted by
              ┌──────────────────┼──────────────────────┐
              ▼                  ▼                      ▼
   ┌────────────────┐  ┌──────────────────┐  ┌────────────────────┐
   │ conformance    │  │ arbiter accuracy │  │ autodiff (A1)/(A2) │
   │ evaluator      │  │ gate (Decision   │  │ harness in         │
   │ oracles:       │  │ #28): τ from     │  │ check_grad +       │
   │ (P1)(P3)(A1)   │  │ NumericPolicy    │  │ _VJPS/_JVPS gate   │
   │ (S1)(FC1-3)(E1)│  │ via (F1) model   │  │                    │
   └────────────────┘  └──────────────────┘  └────────────────────┘
              │                  │
              ▼                  ▼
   ┌─────────────────────────────────────────┐
   │ IR metadata (Decision #32 carried-or-   │
   │ declared-dropped): multiplier symbol/   │
   │ ‖m‖_∞, stability_class, schatten_policy,│
   │ functional-calculus regime              │
   └─────────────────────────────────────────┘
```

Design rules applied:

- **Every declaration has a consumer (Decision #29).** The `error_contract`
  axis is consumed by the evaluator and the arbiter gate; `stability_class`
  by the sequence-mixer admission gate; `schatten_policy` by the low-rank
  substitution legality check; the multiplier symbol by the fusion
  error model. No axis lands before its first consumer does (that ordering
  is inside each workstream's steps).
- **Semantic vs performance keys (Decision #21a).** `stability_class` and
  the functional-calculus regime are *semantic*: once their carrying op
  exists, absence of the key fails closed, period — no "only when needed"
  qualifier, matching #21a and §5's admission rule (the `unchecked` *value*
  is the legal way to declare no certificate; a *missing key* is an error). `schatten_policy` and
  `error_contract` are *additive*: absence means status-quo behavior, never
  a silently invented default; presence is binding.
- **Derive, don't ask (Decision #30).** Certificates (`μ₂(A)`, normality
  residual, `κ` estimates) are *computed* by the oracle from the operand,
  not accepted as caller-asserted booleans; the declared class states which
  certificate to compute.
- **Claim integrity.** All oracles are numpy-level and host-independent.
  Registry statuses introduced here top out at `oracle_validated` —
  deliberately not named `proven` and deliberately not implying any
  target's native kernel satisfies the same bound (that remains
  exact-target evidence, per the backend maps).

---

## 10. Workstreams

Dependency shape: **FA-1 → {FA-2, FA-3, FA-4, FA-6} (all parallel) →
FA-5**; FA-7 deferred. The only real inbound dependency for FA-2/3/4/6 is
FA-1's `operator_analysis.py` module and the axis; none of them consumes an
FA-2 deliverable, so they do not serialize behind it. FA-5 waits on FA-3
(it re-expresses `spectral_filter`, whose multiplier contract FA-3 owns)
and FA-4 (its `matrix_exp` is M4's reference path) — not on FA-6. FA-4
coordinates with (does not preempt) the sequence-mixer engineering plan.
Alongside FA-1, register this plan in the `docs/audit/compiler/README.md`
scoped-plan index (the README routes all scoped plans; an unrouted plan is
an unconsumed declaration in the Decision #29 sense).

### FA-1 — `error_contract` axis + Lipschitz/norm tables (foundation)

1. Add `error_contract` to `PrimitiveCoverage.contract_status` and to
   `TSOLRow`, with statuses `none → lipschitz_declared → model_bounded →
   oracle_validated` (monotone ratchet, like existing floors in
   `test_tsol_coverage.py`). **This collides with four pieces of existing
   machinery, each of which is an explicit sub-step, not a surprise:**
   (a) `_contracts()` default-fills every axis with `planned` and validates
   against the *global* `VALID_CONTRACT_STATUSES` — the new axis needs
   **axis-scoped** status validation (so `lipschitz_declared` does not
   silently become legal on `sharding_rule`) and a `none` default;
   (b) `TERMINAL_CONTRACT_STATUS_BY_AXIS` needs an `error_contract` entry;
   (c) the dashboard glyph/summary buckets and `_AXIS_COMPLETE_FLOORS`
   assume the shared vocabulary — extend both for the new axis;
   (d) the "12 contract axes" prose (tsol_coverage.py docstring, CLAUDE.md
   Decision #24 entry and source-location row) becomes wrong — reword those
   to a count-free phrasing per Decision #26 in the same change.
2. New module `python/tessera/compiler/operator_analysis.py`: norm/Lipschitz
   table (§1.4) with each constant backed by a unit test (interval-arithmetic
   check for gelu/silu derivative sup — the constants in §1.4 are *plan
   inputs to be verified*, not assumptions); (F1) accumulation-model
   functions `gamma(K, u)`, `c(K, accum_mode)`; power-iteration +
   Hutchinson two-sided `‖Δ‖` bracket (§2).
3. Composition rule (C1) as a function over contract-carrying stages —
   consumed by the fused-epilogue budget check.
4. Drift gates: dashboard regeneration; floor test that no op regresses in
   `error_contract` status.
   **Exit:** every TSOL Linear-Algebra + Spectral + Layout row carries an
   explicit `error_contract` disposition — `lipschitz_declared` or better
   for the ops with global constants, and a **recorded domain-restricted
   disposition for the factorization/solve ops** (`cholesky`, `qr`, `svd`,
   `tri_solve`), which have no global Lipschitz constants (unbounded near
   rank-deficiency / degenerate singular values; `tri_solve`'s norm is
   `‖A⁻¹‖`): their route is `ε_dom` over a conditioning-restricted domain
   `D_s = {κ(A) ≤ κ_max}` with Stewart–Sun-style perturbation constants, or
   an honest `none` with the reason recorded. Second exit clause: the
   arbiter gate reads `τ` from the (F1) model for `gemm`/`matmul` on the
   reference lane — **via a declared interface extension**: the existing
   hook is a scalar elementwise `accuracy_atol` (`kernel_emitter.py`
   `max(caller_atol, accuracy_atol)`), and a relative operator-norm budget
   must enter as a typed `(norm_kind, τ)` pair converted explicitly
   (`atol ≈ τ·‖T‖·input-scale`), not shoehorned into the scalar — otherwise
   FA-1 reproduces the exact "scalar tolerance with no attached norm"
   defect §0 exists to fix.

### FA-2 — Adjoint-law harness (A1)/(A2)

1. Randomized (A1) test generator over `_VJPS`-registered linear ops;
   per-point (A1) linking `_VJPS`/`_JVPS` for nonlinear ops.
2. Surface in `python/tessera/debug.py` alongside the existing checkers —
   the natural extension point is the `check_grad_directional` shape
   (basis-free, one random direction — exactly (A1)'s form), exposed as a
   new `check_adjoint(T, vjp, ...)` rather than overloading `check_grad`'s
   scalar-function signature, which takes a scalar `fn` and cannot express
   operator access. Tolerance from the (F1) model, not a hand-tuned
   epsilon.
3. Tighten the registry auto-flip: `vjp`/`jvp` may report `complete` only
   if the (A1) harness covers the rule. **This amends Decision #24's stated
   behavior** ("auto-flips (V/J)VP axes from registered `_VJPS`/`_JVPS`"),
   not merely the #26 caveat that observes it — so the change lands with a
   dated in-place amendment to #24 per the CLAUDE.md amendment protocol,
   scheduled as part of this step.
4. Drift gate: negative fixture (deliberately-transposed rule) stays in the
   suite permanently; auto-flip conditioning covered by a governance test.
   **Exit:** harness green over the TSOL linear core; at least one
   deliberately-transposed rule caught by a negative test (a harness that
   only ever accepts proves nothing — Decision #19's negative-fixture
   principle).

### FA-3 — Spectral isometry + multiplier oracle

1. (P1)/(P2) oracle for `fft/ifft/rfft/irfft`, **branching on the per-call
   `norm` argument** (`backward`/`ortho`/`forward` — the implementation
   delegates to a caller-selected convention; there is no single
   implementation constant to "read once"), with `O(u log N)` tolerance.
2. `stft/istft` round-trip oracle with explicit COLA precondition check.
3. Multiplier symbol as Graph IR attribute on `spectral_conv`/
   `spectral_filter`; (P3) norm feeds the M1 budget (normalization-free,
   §4; Hermitian-symmetry check on the rfft lane); Decision #32 drop-note
   where a lowering discards it.
4. Drift gate: symbol attribute presence/drop-note checked by a governance
   test in the `test_governance_declarations.py` family.
   **Exit:** spectral family rows at `model_bounded`; one fusion decision in
   `fusion_core.py` demonstrably consuming `‖m‖_∞`.

### FA-4 — Recurrence stability certificates

1. `operator_analysis.py`: `mu2(A)` (log norm), `‖A‖₂→₂`, normality residual,
   power-boundedness probe; diagonal fast paths for diagonal SSM `A`.
2. `stability_class` semantic key on the `linear_recurrence` design (lands
   with the op, via the sequence-mixer plan), with the M4.b
   discretization-transfer table (ZOH/bilinear inherit, forward-Euler needs
   step condition) encoded as data.
3. Long-context metamorphic oracle: certificate class predicts the norm
   envelope of `‖h_t‖` over t; violation fails the run. **Scope split,
   because `linear_recurrence` does not exist as an op yet** (today it is
   only a schedule string): FA-4 delivers the oracle *code*, testable
   against a numpy reference recurrence loop; wiring it to the real op is
   an explicit handoff item to the sequence-mixer engineering plan, not an
   FA-4 exit condition.
4. Drift gate: certificate functions covered by unit tests including a
   non-normal transient-growth negative case (ρ < 1, ‖Aᵗ‖ ≫ 1 transient).
   **Exit:** certificates + standalone oracle implemented and tested against
   the reference recurrence (they precede the op landing); sequence-mixer
   plan references this section for admission and owns the op integration.

### FA-5 — `functional_calculus` TSOL admission

1. TSOL admission checklist (spec §Admission): registry row, API owner,
   reference implementation, regime `spec` semantic key, (FC1)–(FC3)
   oracle. Reference paths per regime: eigh for the **symmetric** sub-case
   only; real normal with non-real spectrum (skew/rotation/unitary) and
   general matrices go through Schur–Parlett, where the real Schur form's
   2×2 blocks force the oracle to complexify (FC1's hypotheses, §6).
2. Admit `matrix_exp` as a **new** instance (no such op exists today —
   verified against the repo; its consumer is M4's `e^{ΔtA}` reference
   path). Re-express the **existing** `spectral_filter` dense path as an
   instance, keeping its existing spelling as an alias (no breaking
   rename).
   **Exit:** admission lands whole (catalog + registry + tests + regenerated
   dashboards, per spec rule); (FC2) path validated on a deliberately
   ill-conditioned `V` fixture.

### FA-6 — `schatten_policy` for low-rank substitution

1. `NumericPolicy` extension field `schatten_policy: (p, budget) | None`
   (additive; None = status quo; `scale_layout` is the structural
   precedent). Decision #15a makes
   `docs/reference/tessera_tensor_attributes.md` normative for
   `numeric_policy` — that doc updates in the same change (cross-registry
   rule, Working Rules).
2. Truncated-SVD reference + (E1) oracle (tail-error achieved, not
   factor-equality — §7); substitution legality check consuming the policy
   in the arbiter/fusion path.
   **Exit:** `factorized_matmul` row carries the policy; one negative test:
   an `S_2`-passing, `S_1`-violating substitution is rejected when the
   policy says `p=1`.

### FA-7 — Forms/coercivity for stencils — **deferred** (see §8).

---

## 11. Risks And Honest Limits

| Risk | Mitigation |
|---|---|
| Norm estimates mistaken for proofs | Status vocabulary caps at `oracle_validated`; power iteration always paired with the Frobenius upper bound (two-sided) |
| Constants in §1.4 wrong | Each constant is a test obligation in FA-1, not an assumption; interval-arithmetic derivative bounds for gelu/silu |
| Nonlinear ops over-claimed | Domain `D_s` mandatory for `ε_dom`; softmax/layer_norm explicitly excluded from M5 |
| Non-normal `A` treated as normal | Normality residual computed, never asserted (Decision #30); FC2/FC3 ladder exists precisely for this |
| Discrete stability inferred from eigenvalues | M4.c hierarchy makes `ρ(A) < 1` alone insufficient by construction; Kreiss/transient class explicit |
| Plan drifts from status truth | This doc carries no counts/status claims; dashboards own them (Decision #26) |
| Cross-plan ordering conflict | INTEGRATED_COMPILER_PLAN owns the queue; FA-series enters it by a separate edit with its own review |

## 12. Verification Log

| Pass | Scope | Result |
|---|---|---|
| Mathematical review (2026-08-22, independent adversarial pass) | Every theorem hypothesis, inequality, constant in §§1–8; attributions grepped against the book's .tex sources; Lipschitz constants recomputed numerically | 13 findings, all incorporated. Substantive: (P3) normalization-invariance (the original `c_N` hedge was inverted — §4 rewritten); power-boundedness at ρ=1 characterized by semisimple unit-circle eigenvalues, not diagonalizability (§5); Hutchinson upper side is statistical, not a bound (§2); (FC1)/(C1)/(F1) missing hypotheses added; real-vs-complex scalar-field rule added (§1.1); Kreiss `e·N` dimension factor added (§5). Attribution fixes: Cayley, Schatten-p, Eckart–Young–Mirsky are standard material not in the book. Verified correct independently: softmax PSD-sandwich argument, gelu 1.1289 ≤ 1.13 and silu 1.0998 ≤ 1.1 (double-computed: reviewer + main session), (S1) as a genuine iff, ZOH/bilinear transfer, (A1)/(A2), (P1)/(P2), (FC2)/(FC3) constants vs Trefethen–Embree, (E1), S₁-ordering, partial-trace contractivity, Lax–Milgram. |
| Logical/architecture review (2026-08-22, independent adversarial pass) | Internal logic, dependency order, exit-criterion testability, consistency vs `tsol_coverage.py` / `primitive_coverage.py` / TSOL spec / CLAUDE.md Decisions #15a–#32 / evaluator plan / `debug.py` / arbiter hook — all checked against the actual repo | 16 findings, all incorporated. Substantive: `matrix_exp` did not exist — FA-5 reframed as admission; `error_contract` status vocabulary collides with global `VALID_CONTRACT_STATUSES` — axis-scoped validation made an explicit FA-1 sub-step; auto-flip tightening amends Decision #24 (dated amendment scheduled); dependency graph corrected to FA-1 → {FA-2,3,4,6} ∥ → FA-5; FA-4 step 3 split (oracle code now, op wiring handed to sequence-mixer plan); factorization ops given a domain-restricted route in FA-1's exit; τ↔atol arbiter interface made explicit; `check_adjoint` replaces the `check_grad` mode overload. Explicitly cleared: no τ-derivation/validation circularity (validation compares *measured* error against the model); claim-integrity clean (no device promises, no prose counts, no generated-doc hand-edits); evaluator oracle names, TSOLRow axes, `NumericPolicy` fields, TSOL-A1..A5 references all verified real. |
| Reconciliation (2026-08-22, main session) | Cross-check of the two reviews against each other and independent numeric verification of §1.4 constants | No conflicts between the two passes (they touched disjoint defects except §2, where both landed on the same "sampling ≠ proof" tension, resolved once). All 29 findings applied; the two "verified correct" lists jointly cover every load-bearing claim. |
