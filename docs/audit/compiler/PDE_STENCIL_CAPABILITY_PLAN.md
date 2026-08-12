---
last_updated: 2026-08-12
audit_role: plan
plan_state: open
---

# PDE and Stencil Capability — Symbol Classification, Stability Certificates, and a Neighbors-First Execution Plan

> **Routing:** start at [`README.md`](README.md). This document owns PDE-operator
> semantics, discrete-stability contracts, and their acceptance criteria; global
> ordering lives only in
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md).
> `MASTER_AUDIT.md` + generated dashboards stay status truth (Decision #26);
> this plan is the build sequence, not a status claim.

**Status:** plan (2026-08-12). **Sources:** four texts reviewed 2026-08-12 —
Jakobsen, *An Introduction to PDEs* (arXiv 1901.03022, full read, 20+ executed
checks); Choudary/Parveen/Varsan, *Partial Differential Equations* (ASSMS Lahore
2010, full read, 14 executed checks); Renardy & Rogers, *An Introduction to PDEs*
(Springer — **the available PDF is a 6-page TOC-only excerpt**, so its
contributions here are concept-level and labelled as such); Grinberg, *An
Introduction to Graph Theory* (arXiv 2308.04512, full read, 17/17 checks). Full
review and per-source correctness verdicts in Appendix A.

**Why neighbors/TPP first:** both stencil stacks are already built, registered in
`tessera-opt`, and lit-covered, so this capability is a *contract* problem rather
than a greenfield one — and the review found three latent correctness defects in
those existing lanes (§III.3) that must be closed before any new op is layered on
top. Everything through Phase 2 is host-free and runs on any box; the first
hardware proof is routed to gfx1151 per
[`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) §6a.

**Verification:** every derived result below is numerically checked by
[`tests/unit/test_pde_stencil_model.py`](../../../tests/unit/test_pde_stencil_model.py)
(pure numpy + stdlib `fractions`, no new deps, 78 assertions, green): the
stability decider reproduces every closed-form threshold exactly, the
classification ladder returns the documented verdicts including `mixed` for
Tricomi, and the graph-scheduling oracles are brute-force confirmed. Those
reference functions are the declared oracles for the later C++ analysis
(Decision #31) and for the F4 kernel gates.

---

## Part I — Mathematical Model

### I.1 Setting and notation

A linear differential operator is `L u = Σ_α a_α(x) D^α u` over spatial axes
`x ∈ R^n`, multi-index `α ∈ N^n`, order `m = max|α|`, system size `S` (`S = 1`
scalar). `D^α ≡ ∂^α` — plain partials. That convention is a **semantic key**
under Decision #21a and is declared, never defaulted: with `D = -i∂` the symbol
picks up `(-1)^m` instead of `i^m`, which *inverts* the sign test separating
forward- from backward-parabolic (§I.2, P4).

Discretisation introduces grid spacing `h_i` per axis and timestep `k`, giving
the dimensionless groups `s = k·ν/h²` (diffusion number) and `a = c·k/h`
(Courant number). Every stability bound in §I.3 is a statement about those two.

### I.2 The principal symbol and type classification

**P1 — the reduction that removes complex arithmetic.** The principal symbol is
`p_m(x,ξ) = Σ_{|α|=m} a_α(x)(iξ)^α`. Since `|α| = m` for every principal term,
`(iξ)^α = i^m ξ^α`, so

```
p_m(x,ξ) = i^m · q_m(x,ξ),     q_m(x,ξ) := Σ_{|α|=m} a_α(x) ξ^α
```

`i^m` is a common unimodular factor, so `p_m ≠ 0 ∀ξ≠0 ⟺ q_m ≠ 0 ∀ξ≠0`, and
`q_m` is a **real homogeneous form of degree `m` in `n` real variables**. For
systems `det P_m = i^{mS} det Q_m`, so the same reduction holds on the
determinant. The entire classification problem is therefore: *does a real
homogeneous form have a nontrivial real zero?* Verified numerically
(`test_principal_symbol_factors_as_i_to_the_m_times_a_real_form`).

**P2 — the odd-degree theorem prunes half the search space.** `q_m` homogeneous
of odd degree satisfies `q(-ξ) = -q(ξ)`; for `n ≥ 2` the unit sphere is
connected, so by the intermediate value theorem `q` has a zero on it. **No
scalar operator of odd order in ≥2 variables is ever elliptic**, hence every
elliptic scalar operator of order ≥3 has even order. Verified
(`test_odd_degree_scalar_operators_are_never_elliptic`).

**P3 — second order is completely decidable, in exact arithmetic.** For `m = 2`
build the symmetric coefficient matrix `A` (`A_ii = a_{2e_i}`,
`A_ij = a_{e_i+e_j}/2`) and compute its exact inertia `(n₊, n₋, n₀)` by rational
LDLᵀ with symmetric pivoting; congruence preserves inertia (Sylvester). Then:
definite ⇒ elliptic; Lorentzian `min(n₊,n₋)=1` ⇒ hyperbolic *relative to a
declared covector*; `min(n₊,n₋)≥2` ⇒ ultrahyperbolic (no well-posed IVP in any
direction); `n₀ ≥ 1` ⇒ degenerate, go to P4.

**The declared covector must be validated, not merely present.** A Lorentzian
form is hyperbolic only with respect to a covector in the minority cone, so
`time_axis` is checked against the form: non-characteristic (`A[t][t] ≠ 0`) and
definite once that axis is deleted. Accepting any non-null `time_axis` is
wrong — for `diag(1,1,-1)` the covector `e₀` gives `q(ξ + τe₀) = 1 + τ²`, which
has no real roots, so the Cauchy problem is not well posed in that direction
even though the form is Lorentzian. Only `e₂` qualifies. Verified in
`test_hyperbolic_requires_a_covector_that_is_actually_timelike`. Coefficients arrive as IEEE
doubles, i.e. exact dyadic rationals, so **there is no epsilon anywhere in this
decision**. Verified across six matrices including the zero-diagonal `xy`-form
(`test_second_order_classification_is_exact`).

**P4 — parabolicity is NOT decidable from the principal part.** This is the
result that breaks the textbook one-liner. For `u_t − Δu`, the order-2 principal
part is `−Σ∂_i²`: `ξ_t does not appear at all`, so `q_2` vanishes on the whole
line `ξ_x = 0`. Parabolicity lives in the interaction between the degenerate
direction and the **sub-principal** `D_t` term. With `D = ∂`, Fourier in `x`
gives `∂_t û = (1/c_t)·q_x(ξ)·û`, so the forward problem is well posed iff the
definite spatial part carries the sign *opposite* to `c_t`. Forward heat
(`u_t − u_xx`) and backward heat (`u_t + u_xx`) have **identical principal
parts** and opposite well-posedness; only the sub-principal coefficient
separates them. Verified (`test_forward_and_backward_heat_are_distinguished`).

**P5 — type can change over the domain, and that is a definite answer.** The
Tricomi operator `y·u_xx + u_yy` has `A = diag(y,1)`: elliptic for `y>0`,
hyperbolic for `y<0`, degenerate on `y=0`. The classifier enumerates realizable
sign patterns of the variable principal coefficients and returns the **definite**
verdict `mixed` with the type-change locus named — not `elliptic` (which is what
sampling `y` at one point yields) and not `unknown`. Verified
(`test_tricomi_is_classified_mixed_not_guessed`).

**Decidability boundary — stated honestly.** Deciding positivity of a general
quartic form is NP-hard, so the implemented ladder is: exact and complete for
`m ≤ 2`; exact and complete for odd `m` in the *negative* direction; for even
`m ≥ 4` a set of exact **sufficient** certificates (diagonal even powers, AM–GM
dominance, a verified sum-of-squares witness, composition provenance) plus an
exact **refuter**; everything else `unknown`. Sampling may refute, never confirm.

### I.3 Discrete stability: the amplification factor

Substituting `u^n_j = ξ^n e^{ι(j·θ)}`, `θ = k_wave ⊙ h`, into the normalised
update `Σ_l Σ_j A[l][j] u^{n+1-l}_{i+δ_j} = 0` gives the characteristic pencil
`Π(ξ,θ) = Σ_l ξ^{L-l} Â_l(θ)`, `Â_l(θ) = Σ_j A[l][j] e^{ιδ_j·θ}`.

**P6 — the cosine reduction, stronger than the symmetric-stencil form.** For
*real* tap coefficients the autocorrelation `R_m = Σ_j c_j c_{j-m}` is real and
symmetric, so `|ξ(θ)|² = R_0 + 2Σ_{m≥1} R_m cos mθ = P(cos θ)` — a polynomial in
`c = cos θ` over the rationals. Stencil **symmetry is not required**, which is
why upwind advection (an asymmetric stencil) is decidable in closed form.
Consequently the entire 1-D real-coefficient one-level explicit family is
decidable by Sturm sequences on `[-1,1]`, with no exceptions.

**P7 — `|ξ| ≤ 1` is the wrong test for multi-level schemes.** Stability is the
**root condition**: all roots of `det Π(ξ,θ) = 0` in the closed unit disk *and*
every unimodular root simple. Leapfrog for the wave equation gives
`ξ² − 2(1+p)ξ + 1 = 0` with `p = s²(cos θ − 1)`; the closed-disk condition holds
iff `s ≤ 1` (CFL), but at exactly `s = 1, θ = π` the two roots coincide on the
unit circle and the solution grows **linearly in n**. Verified: peak amplitude
scales 10× per decade of step count (`4.0×10³ → 4.0×10⁴ → 4.0×10⁵`), while at
`s = 0.9` it is bounded (`test_leapfrog_is_defective_at_the_cfl_boundary`). A
verdict lattice that lacks `DEFECTIVE` reports this scheme stable.

**P8 — for systems, spectral radius is necessary and not sufficient.** The
leapfrog companion at `p = −2` has `ρ(G) = 1` exactly and `‖G^n‖₂ = 2n`,
verified to 5% at `n = 1000` (`test_spectral_radius_is_not_sufficient_for_systems`).
Power-boundedness needs more (normality, an energy symmetriser, or Kreiss). The
pass therefore concludes `PROVEN_STABLE` for a system only via normality, an
explicit `‖G‖₂ ≤ 1` minor test, or a *supplied and verified* symmetriser —
never from `ρ ≤ 1`.

**P9 — sampling can only ever refute.** Dense `θ`-sampling computes a lower
bound on `sup|ξ|`, so `sampled_max > 1` ⇒ definitively unstable (sound), while
`sampled_max ≤ 1` proves nothing. This is not hypothetical: centered explicit
advection has `|ξ|² = 1 + a²(1−c²)`, which equals exactly 1 at **both**
`θ = 0` and `θ = π` and exceeds 1 only in the interior (max at `θ = π/2`). An
endpoint sampler reports "stable" for a scheme that is unconditionally unstable.
Pinned by `test_endpoint_sampling_cannot_prove_stable`, which is the
load-bearing negative in the contract test.

**Closed-form goldens** (all verified exactly, `test_pde_stencil_model.py`):

| Scheme | Amplification | Stable iff |
|---|---|---|
| FTCS heat | `ξ = 1 − 2s(1−c)` | `s ≤ 1/2` (boundary stable; `51/100` refuted) |
| θ/Q-scheme | `D²−N² = 4w(1 + w(2Q−1))`, `w = s(1−c)` | `s ≤ 1/(2(1−2Q))` for `Q<1/2`; **unconditional for `Q ≥ 1/2`** |
| Leapfrog wave | `ξ² − 2(1+p)ξ + 1`, `p = s²(c−1)` | `s < 1`; `s = 1` is `DEFECTIVE`, not stable |
| Centered advection (explicit) | `|ξ|² = 1 + a²(1−c²)` | **never** — the negative golden |
| Upwind advection | `1−|ξ|² = 2a(1−a)(1−c)` | `0 ≤ a ≤ 1` (catches wrong-sign wind free) |
| RK4 + centered advection | `|R(ιy)|²−1 = y⁶(y²−8)/576` | `a ≤ 2√2` |
| Advection–diffusion | (2-parameter) | `s ≤ 1/2 ∧ a² ≤ 2s` — a **region**, not a scalar |

### I.4 Boundary conditions as semantic keys

BC type selects the *mathematics*, not the speed: sine transform ⟺ Dirichlet,
cosine ⟺ Neumann, Laplace ⟺ initial data. Choosing the wrong one yields a
converged, plausible, wrong answer. Under Decision #21a, BC is therefore a
semantic key that fails closed on absence. The same applies to `scheme`
(§III.3 [GAP-3]) and to the upwind biasing direction.

Ghost-cell closures from the source review: `u_{-1} = u_1 − 2hf` (Neumann),
mirror (reflect), wrap (periodic), and the value-substitution (Dirichlet). The
review also supplies the negative fixture: discretising a Neumann BC at first
order **drops the whole scheme to first order**, which is a measurable
convergence-order regression rather than an opinion.

### I.5 Analytic solution kernels — the oracle set

Verified conventions (numerically pinned in the review, Appendix A): heat kernel
`(4πc²t)^{-n/2} e^{-|x|²/4c²t}` with unit mass and the semigroup property
`K_t * K_s = K_{t+s}` (to 5.6e-17); d'Alembert (machine-exact, 2.2e-16);
Laplace fundamental solutions `−(1/2π)ln|x|` (2-D) and `1/(4π|x|)` (3-D),
inverting `−Δ`; Poisson kernel for the half-plane; the erf solution on the
half-line; Kirchhoff/Poisson-descent in 3-D/2-D; image-sum ⟺ sine-series duality
(equal by Poisson summation, agreement 1.2e-15 across four decades of `t`).

### I.6 What the symbol buys beyond gatekeeping

For a hyperbolic operator the classifier's characteristic-speed bound `c_max`
(a sound **upper** bound — the conservative direction shrinks `dt`, and a lower
bound would produce an unstable step) yields `dt_max = C·min_i h_i / c_max`.
This is the only place characteristic speeds exist in the compiler. Likewise the
stability analysis should emit the **admissible region** symbolically rather than
a boolean: an autotuner handed `{s ≤ 1/2 ∧ a² ≤ 2s}` maximises `dt` inside it,
whereas one handed a verdict can only probe and be refused.

---

## Part II — Algorithms

**A1 — Classify.** Resolve principal coefficients (constant-fold; declared sign;
straddling; unknown ⇒ fail closed) → dispatch by `(m, S)` → scalar ladder
(P2 odd-degree → P3 exact inertia → sufficient certificates for even `m ≥ 4`) or
system path (Douglis–Nirenberg weights for mixed-order systems; Friedrichs
symmetric-hyperbolic test; else exact symbolic determinant then the scalar
ladder on a degree-`mS` form). Variable coefficients with a declared sign
produce `assumptions` that a guard pass materialises as runtime preconditions —
a told fact becomes a checked one (Decision #30).

**A2 — Extract the update rule.** Walk the `tpp.time.step` region maintaining a
`LinearForm: (level, δ) → Q^{S×S}` per SSA value. `FAIL` is the default: a call,
an `scf.if`, a nonlinear product, or a data-dependent coefficient terminates
extraction with a *named* reason and yields `unknown`. There is no
assume-identity branch.

**A3 — Decide stability.** Reduce to `R(c) = 1 − |ξ|² ≥ 0` on `[-1,1]` (P6),
decide by Sturm sequence over exact rationals; multi-level via Jury table plus a
mandatory discriminant/simple-root refinement (P7); systems via normality or an
`I − G*G ⪰ 0` minor chain (P8); multi-dimensional separable stencils directly,
non-separable via Bernstein positivity with adaptive subdivision.

**Every path fails closed on budget exhaustion.** An interval that is still
unresolved when subdivision runs out is UNKNOWN, and UNKNOWN must be raised or
surfaced — never collapsed into "nonnegative". Subdivision is driven by root
*counts*, never by evaluating a fixed grid: a negative dip narrower than the
grid is invisible to sampling, and two roots `2⁻⁹⁹` apart are enough to produce
one. That is the same P9 failure mode wearing a different hat, and it is
regression-tested by
`test_exact_decider_finds_a_narrow_negative_dip_between_close_roots` and
`test_decider_refuses_rather_than_guessing_when_budget_is_exhausted`.

**A4 — Emit the bound.** Factor out the ubiquitous `(1−c)` roots, then: degree
≤1 in `c` ⇒ endpoints suffice (this closes advection–diffusion in closed form);
degree 2 ⇒ add the interior critical point; else discriminant root isolation;
else certified bisection on `dt` returning a *bracketing certificate*
(`dt_max_certified`, `dt_min_refuted`) where both endpoints are proven and only
the gap is unknown.

**A5 — Manufacture solutions.** Given exact `u`, compute `ρ = L[u]` by
**2nd-order truncated Taylor (jet) AD** — not finite differences. Measured: an
FD source bottoms out near 1e-8, and feeding a 2nd-order FD source to a
4th-order scheme collapses the fitted order from `p = 3.998` to `p = 1.066`. Jet
AD matches sympy to 7.1e-15 with no dependency; sympy remains an optional
*second path*, which makes the source term itself a cross-path oracle subject.

**A6 — Schedule collectives by matching decomposition.** Regularise the demand
to Δ-regular, then peel Δ perfect matchings (Frobenius guarantees one exists at
each step). Δ rounds is **optimal, not merely achievable**: rank `i` must issue
`Σ_j D[i][j]` chunks at one per contention-free round. Use König's line-colouring
bound (bipartite ⇒ chromatic index exactly Δ) — **not Vizing's Δ+1**, which is
the general-graph bound and over-provisions every schedule by a round.

**A7 — Diagnose routing defects.** Hall–König: max matching `= |X| − max_U(|U| −
|N(U)|)`, and the maximizing `U` names *which token group starved which expert
group*. That set is the deliverable; the placement count is not (§IV.2).

---

## Part III — Tessera Mapping

### III.1 Op-set

TSOL has no PDE/stencil category today; open action **TSOL-A1** pre-authorises
admissions. Proposed category **"PDE And Stencil Operators"**, each entering the
12-axis registry in `primitive_coverage.py` (Decision #24 — update
`op_catalog.py` and `primitive_coverage.py` together):

| Op | Signature | Role |
|---|---|---|
| `pde.operator` | `(coeff_fields…) → !tessera.pde_operator<m,n,S,elt>` | Carries `Σ a_α D^α` as a typed handle; classification is *derived*, never a type parameter |
| `pde.discretize` | `(operator, fd_scheme, order, bc) → stencil` | **The seam** to `neighbors.stencil.define` / `tpp.stencil.apply`; folds `1/Πh^α` into taps |
| `pde.laplacian` / `gradient` / `divergence` / `curl` | typed, N-d | Replace the untyped `tpp.grad` scaffold |
| `stencil_apply`, `halo_exchange` | existing IR, new TSOL rows | Admission forces the single-surface question (§III.3 [GAP-4]) |
| `tridiagonal_solve` | `(dl,d,du,b) → x` | Required for Crank–Nicolson; missing everywhere today |
| `heat_kernel`, `advect`, `dalembert` | closed-form | Contract-rich (semigroup, shift-composition, reversibility) |

Spectral additions: DST/DCT aligned to BC selection, and **Chebyshev** — the
largest clean gap in `src/solvers/spectral/`, which is FFT-only (mixed-radix,
Stockham radix-4, Bluestein) and reuses the existing FFT lane via DCT.

Adjoints are exact and mutually defining (`grad* = −div`, `div* = −grad`,
`laplacian* = laplacian`) **under homogeneous Dirichlet/periodic only** — a
Neumann face contributes a boundary term. That dependence is why `bc` must be an
attribute of the op rather than module metadata: without it, reverse-mode AD
through a PDE operator is silently wrong at the boundary.

### III.2 Passes

`-tessera-pde-classify` (analysis + annotation, rewrites nothing) →
`-tessera-pde-materialize-guards` → `-tessera-pde-legality` (capability table;
refuses hyperbolic-into-elliptic, backward-parabolic time-marching, mixed-type,
ultrahyperbolic, variable-coefficient-into-spectral) → `-tessera-pde-discretize`
→ `-tpp-stability-certificate` (after legalize + halo-infer, **before**
`fuse-stencil-time`, which destroys per-op tap structure) → existing TPP
pipeline. Enforcement is at `-lower-tpp-to-target-ir`: `proven_unstable`,
`defective`, `unknown`, **and attribute-absent** all fail closed. Absence is not
permission.

Named consumers (Decision #29): the lowering gate; a new
`LegalStencilCandidateGenerator` in `autotune_v2.py` contributing four
`CandidateRejection` reasons; a `-tpp-select-dt` that reads
`dt_max_certified` when the program says `dt = "auto"` — the constructive
consumer, since the pass then *supplies* the largest proven-safe step rather
than only refusing.

### III.3 Latent defects found by this review — all three verified in-tree

These predate the plan and must close first; layering new ops on them would
inherit the bugs.

**[GAP-1] Grid spacing does not exist anywhere in TPP.**
`src/solvers/tpp/lib/TargetHooks/CPU/Stencil.cpp:9` states the convention in a
*comment* — "unit grid spacing, periodic boundary" — and the kernel hardcodes
both. A repo-wide grep for `spacing` across `src/solvers/tpp` and
`src/compiler/tessera_neighbors` returns **exactly that one comment**. Every PDE
lowered through this lane on a grid with `h ≠ 1` is wrong by `h^{-|α|}`, with no
diagnostic. The Target-IR ABI has no parameter for it either
(`ts_stencil_grad_cpu(const float*, float*, int nx, int ny, int axis, int order)`).

**[GAP-2] `StencilDefineOp` has taps but no coefficients.** Its ODS summary
reads *"Define a stencil taps/coeffs object with BC"* while its arguments are
`(ins ArrayAttr:$taps, OptionalAttr<StrAttr>:$bc)` — the coefficients field the
summary advertises does not exist. `StencilLoopMaterializePass.cpp:333`
consequently accumulates with a bare `arith::AddFOp`, i.e. implicit coefficient
1.0, so the only expressible update is `u_out[i] = Σ_j u[i+δ_j]`. **FTCS heat is
not representable.** This is simultaneously the blocking gap for the stability
pass and a live correctness bug.

**[GAP-3] A semantic key is silently defaulted.**
`LegalizeSpaceTime.cpp:74` reads `StringRef scheme = "central";` *before*
checking whether the attribute exists. Combined with the verified result that
centered explicit advection is unconditionally unstable (§I.3), an unannotated
advection kernel is silently converted into a divergent one. This is the
`EBMCanonicalize` `manifold` failure that motivated Decision #21a, recurring.

**[GAP-4] Two production stencil/halo stacks, no declared oracle relationship.**
`tessera_neighbors` and `src/solvers/tpp` each define `stencil.apply`,
`halo.exchange`, and a halo-infer pass. Decision #31 permits a second
implementation only as a declared oracle with a differential test. The ordering
caveat applies: do not collapse before the survivor carries temporal halos
(neighbors) *and* space–time fusion + RK stepping (TPP).

**[GAP-5] Three of four named Target-IR callees have no definition.**
`LowerTPPToTargetIR.cpp` names `ts_stencil_grad_*`, `ts_stencil_apply_*`,
`ts_bc_enforce_*`, `ts_halo_exchange_*`; only `ts_stencil_grad_cpu` exists. That
is Decision #29 inverted — a consumer named with nothing behind it.

### III.4 Autodiff and coverage

Registering the linear operators' VJPs auto-flips the `vjp`/`transpose` axes via
`_vjp_registered_names()`. Honest landing state: **34 of 120 contract cells
close**, each for a nameable reason; `backend_kernel` is `planned` /
`artifact_only` / `reference` everywhere, unchanged as the long-pole gate.
`advect`'s VJP stays `planned` deliberately — an upwind switch is
non-differentiable at `v = 0`, so a naive rule would be wrong at exactly the
interesting points (Decision #30).

Decision #32 information-loss records are required at two boundaries:
`numeric_policy` has **no carrier below Graph IR** (zero occurrences in either
stencil tree), and the Target-IR descriptor cannot yet carry BC value or
accumulator width. Until the widened descriptor lands, the lowering must record
`tessera.info_loss = ["numeric_policy"]` with a reason and the boundary verifier
must accept the loss *only* when that record is present.

### III.5 Verification plan

| Oracle | Entry point | Status |
|---|---|---|
| Stability goldens (7 schemes) | contract test | **green now** |
| Classification ladder incl. Tricomi, fwd/bwd heat | contract test | **green now** |
| Sampling-only-refutes | contract test | **green now** |
| Matching rounds, Hall–König, matrix-tree, de Bruijn | contract test | **green now** |
| Manufactured solutions + convergence order | `testing/manufactured.py` | Phase 1 |
| Semigroup, conservation, reversibility, Burgers residual | `metamorphic_equivalence` | Phase 2 |
| Images ⟺ sine series; multi-backend heat | `cross_path_equivalence` | Phase 2 |
| Discrete maximum principle | new `MaximumPrincipleVerdict` | Phase 2 |
| Ill-posed rejection (backward heat, elliptic Cauchy) | `E_PDE_*` diagnostics | Phase 2 |

**Three oracle designs were empirically falsified during this review and are
corrected in the plan** — recording them because each would have shipped as a
green test proving nothing:

1. **Causality does not discriminate as scoped.** Perturbing outside the light
   cone leaves the far field bitwise unchanged for the wave equation *and for
   FTCS heat* (both measured `0.000e+00`), because both use the same 3-point
   stencil and every explicit stencil has a discrete cone of radius `nsteps`.
   The bitwise form is a *stencil-structure* oracle (still worth having — it
   catches a wide halo read). PDE-class discrimination needs either the
   semantic-solver form (spectral/implicit heat gives far-field `9.0e-08`) or
   the refinement form: the physical cone radius is invariant for the wave
   equation (`0.982, 0.982, 1.006, 0.994`) and doubles per refinement for heat
   (`0.196, 0.393, 0.810, 1.632`).
2. **The maximum-principle negative control needs sharp input.** A 4th-order
   Laplacian has negative outer weights and cannot satisfy a DMP, yet it *passes*
   on smooth random data (violation `0.0`). On a delta it violates by exactly
   `r/12`. The generator is part of the oracle; `safe_input` needs a `"sharp"`
   kind or the test is decorative.
3. **The semigroup negative control needs non-commuting operators.** Advection
   and diffusion are both Fourier multipliers, so they commute and Lie-split
   *exactly* (defect `2.2e-16`). A real violator pairs a Fourier multiplier with
   a real-space one (variable-coefficient reaction): Lie `7.96e-04`, Strang
   `4.81e-05`.

### III.6 Benchmarks

No stencil, halo, Poisson, heat, Jacobi, multigrid, or shallow-water benchmark
exists anywhere in `benchmarks/`, despite both stencil stacks being built and
lit-covered. New `benchmarks/pde/` emitting `BenchmarkRow`, honest triple
`REFERENCE / EXECUTABLE / REFERENCE` at landing:

- **`relaxation_bench.py`** — Jacobi/Gauss–Seidel, carrying a *performance-model*
  oracle: measured convergence rate must match `ρ(M_J) = 1 − sin²(πk/2N) −
  sin²(πl/2N)` and `ρ(GS) = ρ(J)²` (both confirmed to ≤2.6e-06). **Pin the
  convention: `N` is the number of cells.** Using interior points gives a
  1.6e-2 error that reads as a genuine performance regression.
- **`heat_step_bench.py`** — FTCS vs Crank–Nicolson vs FFT-spectral vs direct
  convolution; `metrics` carries `crossover_radius`, the number an
  algorithm-selection pass needs and nothing else in the tree produces.
- **`wave2d_bench.py`** — leapfrog; correctness field is the time-reversal
  residual, plus `energy_drift_relative` — a *secular* drift means the
  integrator lost its symplectic structure, which a fusion pass that
  reassociates the update would cause and no value-tolerance compare would
  notice.
- **`shallow_water_bench.py`** — promote `shallow_water_smoke.mlir` (today 6
  lines, one `tpp.grad`, no timestep) to a measured benchmark; correctness is
  mass conservation and energy drift, since SWE has no closed form. Starts
  `ARTIFACT_ONLY`, and saying so is the point: it makes visible that the TPP
  lane compiles and does not execute, which the passing lit fixture obscures.

### III.7 Phasing

| Phase | Deliverable | Where it runs |
|---|---|---|
| **0** | This plan + the executable contract test (**landed**) | any box |
| **1** | [GAP-1..3] fixes: `spacing` attribute + tap scaling, `coeffs` on `StencilDefineOp` (also fixes the materializer), `scheme` fails closed. Each with a negative fixture. | any box (host-free) |
| **2** | `pde.operator` ODS + exact-rational analysis + `-tessera-pde-classify` + `-tessera-pde-legality`; 24 lit fixtures incl. 14 negative | any box |
| **3** | `-tpp-stability-certificate` + bound emission + autotuner rejection reasons + `-tpp-select-dt` | any box |
| **4** | Manufactured-solutions harness, the metamorphic oracle family, `E_PDE_*` diagnostics, `benchmarks/pde/` | any box |
| **5** | First hardware proof: stencil + halo on gfx1151, `evaluate()` to `HARDWARE_VERIFIED`; Apple GPU lane second | **Strix Halo**, then Mac |
| **6** | [GAP-4] neighbors/TPP unification, once the survivor carries both feature sets | any box |

Graph-infrastructure items (Part IV) are independent of this ordering and land
in the sequence 4 → 3 → 2 → 1 given in §IV.5.

---

## Part IV — Infrastructure from Graph Theory

### IV.1 Conflict-free collective schedule generator

`python/tessera/compiler/collective_schedule.py`. Peel Δ perfect matchings
(A6); each round lowers to **`collective_permute`, which already exists
end-to-end** — MLIR op with `source_peers`/`target_peers`, C++ adapter with real
`ncclSend`/`ncclRecv` in-group for both NCCL and RCCL, Python export — and
already validates source/target uniqueness, so *the existing verifier is the
port-conflict check*. No new op, dialect, or adapter method. It also **derives**
`max_inflight` (one outbound chunk per rank per round ⇒ `inflight_rounds`)
rather than taking the hand-authored constant.

Verified: 288 regular instances peel into exactly `k` permutation rounds; 300
non-regular pad to Δ with **excess 0**; Birkhoff–von Neumann reconstructs to
<1e-7 in ≤26 permutations.

**Scope honesty:** the planner and its `MockRankGroup` tests land now; native
transport reports `backend_unavailable`, so no latency claim is available, and
the C++ round-driver should wait — adding a second unimplemented C++ scheduler
beside the existing draft would violate Decision #31's spirit.

### IV.2 Hall/König MoE reference router

`python/tessera/testing/flow_router.py` (test-side: a declared oracle, not a
second production router). **Measured finding that changes the framing:** a
max-flow router adds *zero placement value* to today's `route_tokens()` — 300/300
instances tied — because each `(token, slot)` has exactly one candidate expert,
so flow has no choice and greedy is already optimal. The deliverables are
therefore (a) the **defect certificate** replacing a bare `-1` fill, and (b)
unblocking overflow-redirect, where greedy falls short in **294/600 (49%)** of
instances with candidate sets of size 2–3. The plan ships a test that *pins* the
zero-gap result, so a future contributor reporting a flow-solver "win" without
changing the candidate policy has a bug, not an improvement.

New diagnostic `TS_ERR_ROUTE_CAPACITY_DEFECT` carrying the violating set `U` and
the exact drop count `|U| − |N(U)|` — verified to reproduce the drop count in all
1004 instances that had a real violator.

### IV.3 DAG scheduling upgrade

`composition_cost.py::_topological_orders` enumerates **every** topological order
by DFS (factorial in the width of an antichain, cut off at `max_orders=4096`).
When the cutoff trips, `prune_composition_candidates` retains every candidate —
which is *deliberate and tested* (`test_bounded_nonexhaustive_search_retains_candidate`;
the docstring states "Inexact searches are always retained"), and is the correct
conservative choice given the search is inexact. The problem is not that the
fallback is silent; it is that the fallback engages at around 8 independent
actions (`8! = 40320 > 4096`), so the pruner becomes a no-op at exactly the DAG
widths where pruning would pay. A secondary nit: `_validate_dag` calls
`_topological_orders` twice with identical arguments (lines 416 and 421), using
only `exhaustive` from the first and `orders` from the second — bounded at
`max_orders=1` so it is cheap, but redundant. Replace with critical-path list
scheduling plus a real lower bound
(`max(critical path, per-lane work)`), retaining the exhaustive enumerator as a
**declared oracle** with a mandatory differential test on DAGs of ≤8 actions.
Independent checks: longest path via nilpotency index of the adjacency matrix
(verified 600/600), and the "unique topological order ⟺ Hamiltonian path"
zero-slack detector (verified; 91/600 instances) that should emit a planner
diagnostic — today a fully serial DAG produces one order, `exhaustive=True`, and
is pruned as though the search explored something.

### IV.4 Test utilities

`python/tessera/testing/graph_oracles.py`: `laplacian_count()` (matrix-tree,
self-checking across minor / coefficient / eigenvalue forms — verified against
brute force, Cayley `n^{n-2}` for `n = 2..8`, Petersen 2000) and
`de_bruijn_sequence()` via Hierholzer for exhaustive small-FSM sweeps.
**Count correction:** the cyclic-class count is `(k!)^{k^{n-1}} / k^n`; the
unquotiented `(k!)^{k^{n-1}}` counts labelled sequences and is `k^n` times too
large — a silent off-by-a-factor in a test assertion.

### IV.5 Order and effort

`IV.4` (small, self-contained, supplies IV.3's oracle) → `IV.3` (medium; removes
an already-degrading exponential path) → `IV.2` (small-medium; clean diagnostic
win) → `IV.1` (medium; largest, and most transport-gated).

---

## Part V — Risks

| Risk | Mitigation |
|---|---|
| Fixing [GAP-2] changes stencil numerics for anything relying on implicit-1.0 taps | The materializer bug and the ODS gap close in one patch with a golden-value fixture at `h = 0.5`; grep for in-tree `stencil.define` users first |
| Classification proves too conservative — real operators land in `unknown` | Deliberate. Fixture 14 documents a known false negative so a future "improvement" that guesses must delete a test with written rationale |
| Stability certificate blocks a scheme a user knows is fine | `-tpp-stability=advisory` downgrades to a warning **and stamps `enforced = false` into the IR**, so the artifact self-describes and the evaluator refuses it a correctness rung |
| Exact rational arithmetic is slow | Sizes are tiny (`n ≤ 4`, `S ≤ 8`, stencil degree < 10); integer-scaled `APInt` throughout |
| Jet AD is a new numerical component | It is ~180 lines with an exact cross-check against sympy (7.1e-15) available under `importorskip` |
| Plan competes with the global queue | Scoped plan; binds to existing IDs, mints none; global ordering stays in `INTEGRATED_COMPILER_PLAN.md` |

---

## Appendix A — Source review and correctness verdicts

**Jakobsen (arXiv 1901.03022) — sound; one inverted conclusion.** Every
load-bearing derivation verified: stability bounds, heat-kernel normalisation and
semigroup, Green's function of `y″−k²y`, Jacobi/Gauss–Seidel spectra including
`ρ(GS) = ρ(J)²`, Burgers shock times, transform pairs, Bessel norms. **Error:**
line 3419 states Klein–Gordon has `ω″ = 0` hence is non-dispersive; its own
eq12-272 gives `ω″ = 0.5898 ≠ 0`. Plus ~10 transcription typos (eq11-43 missing a
minus, Burgers `t(τ) = 1/(2τ−1)` should be `1/(2τ)`, eq12-34/35 missing `i`,
et al.).

**Choudary/Parveen/Varsan — final formulas hold; intermediate lines do not.**
All 12 machine checks of stated results pass (C-K series solutions, d'Alembert
cross-path 1.7e-16, Kirchhoff, Poisson-kernel reproduction, Liouville
determinant, C-K majorant residual 9e-11, spectral-vs-FD 1.5e-6, Wong–Zakai OU
bound). Most consequential printed inconsistency: eq 3:13:3.12 is off by exactly
a factor `r` (numerically confirmed, ratio 2.0). **Rule adopted: nothing from
this source enters a Tessera oracle without numerical re-verification first.**
Unique content worth noting: a full SDE chapter developed *via* Wong–Zakai ODE
approximation, which makes every stochastic identity testable with only an ODE
integrator plus RNG; and Ch.1 states the JVP (`th:1.6`) and adjoint/VJP
(`re:1.2`) rules for an `ode_solve` primitive as theorems, with Liouville
`det C = exp∫Tr A` as a free correctness oracle.

**Renardy & Rogers — not gradeable.** The supplied PDF is a **6-page excerpt**
(title + TOC), verified via the PDF page tree (`/Count 6`), `mdls`, and the
reader. Concept extraction was done from the verified TOC with provenance
labelled; the standard kernel conventions were pinned independently by numerical
check so those oracles stand regardless. A ~9-read plan exists to complete
page-exact extraction if the full book becomes available; the highest-variance
item is the §5.4.3 wave fundamental-solution normalisation (p.158).

**Grinberg (arXiv 2308.04512) — 17/17 correct.** Matrix-Tree (all three forms),
BEST, Cayley and its degree-refined form, de Bruijn counts, Hard Rédei (odd
Hamiltonian-path count, all 64 four-vertex tournaments), Whitney chromatic
polynomial, Hall ⟺ matching, König equality — all brute-force confirmed. Zero
errors. That precision is what makes it a source of executable oracles.

## Appendix B — What is already in-tree

`src/solvers/` carries six families: core (`cg`, `gmres`, `root.newton`,
`PeriodicHalo`), linalg (LAPACK-shaped `getrf`/`potrf`/`trsm`/`linear_solve`; **no
eigensolver, no tridiagonal**), spectral (**FFT only** — mixed-radix, Stockham
radix-4, Bluestein; no Chebyshev/DST/Legendre), TPP (space–time fields,
`stencil.apply`, `halo.exchange`, `time.step` with RK schemes,
`shallow_water_smoke.mlir`), scaling-resilience, and option-gated Clifford
(which already carries `ext_deriv`/`codiff` — a discrete-exterior-calculus
proposal would overlap) and EBM. `tessera_neighbors` carries topology/halo/
stencil ops with periodic + reflect BC lowered and **dirichlet/neumann parsed but
not lowered**. `op_catalog.py` has `laplacian_2d` only inside the complex-analysis
group, and no `gradient`/`divergence`/`curl`/`diff`/`tridiagonal`/`eig`.
`_HALO_AWARE_OPS` exists as a registry hook with exactly one user. Graph
algorithms in the compiler: one brute-force toposort enumeration, greedy top-k
MoE routing, closed-form pipeline schedules, and no algorithmic collectives.
