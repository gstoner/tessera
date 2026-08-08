---
last_updated: 2026-08-08
audit_role: plan
plan_state: open
status: proposal — not started, no code landed
source: arXiv:2602.03566v1 "Riemannian Neural Optimal Transport" (Micheli, Cao, Monod, Bhatt)
---

# Riemannian Optimal Transport — Paper Review and Operator Plan

> **Routing:** start at [`README.md`](README.md). This document owns the
> geometry/implicit-differentiation consumer and its acceptance workload;
> global ordering lives only in
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md).
>
**Status vocabulary warning (Decision #25/#26):** everything below is *direction*.
No row here is proof of anything. `docs/audit/MASTER_AUDIT.md` and
`docs/audit/generated/` stay status truth.

---

## 1. What the paper actually does

RNOT learns an optimal-transport map between two distributions **on a Riemannian
manifold** without discretizing the manifold. Prior art (RCPM, Cohen et al. 2021)
represents the potential as a discrete max over landmark values, which the paper
proves (Thm 3.1) needs `m ≥ (C/δ)^p` parameters for accuracy `δ` on a
`p`-dimensional manifold — exponential in dimension. RNOT parameterizes the
potential continuously and enforces `c`-concavity *structurally*, via the
`c`-transform, rather than as a constraint. Thm 5.1 gives a polynomial parameter
bound `W_ε = O(ε^(−4p/3(kp+1)))`.

The transport map is
```
T(x) = exp_x(−∇φ(x))
```
and the potential class is `φ = (f ∘ ϕ)^c` for an MLP `f` over a
distance-to-landmarks embedding `ϕ`.

### 1.1 The algorithm, as a kernel schedule

Algorithm 1 (App. E.3), restated as the thing a compiler has to run:

```
landmarks {ℓ_m}_{m=1..L},  L = 128 (synthetic) / 1024 (real)
embedding   ϕ_m(x) = d(x, ℓ_m)                       # [B,D] × [L,D] → [B,L]
potential   ψ_θ(x) = MLP_θ(ϕ(x))                     # 2 layers × 128 units
cost        c(x,y) = ½ d(x,y)²

for t in 1..T:                                       # T = 1000 outer steps
  sample {x_i}_{i=1..B} ~ μ,  {y_j}_{j=1..B} ~ ν     # B = 256
  for i in 1..B:                     # ← independent per sample, embarrassingly parallel
    y ← y₀(x_i)                      # softmin/LSE warm start, gradients stopped
    for k in 1..K:                   # ← K up to 2500 inner steps
      g ← −log_y(x_i) − ∇ψ_θ(y)      # Riemannian gradient of F(x,y)=c(x,y)−ψ_θ(y)
      y ← exp_y(−α g)
    ψ^c_θ(x_i) ← c(x_i, y) − ψ_θ(y)  # y treated as CONSTANT (envelope theorem)
  L(θ) ← −mean_j ψ_θ(y_j) − mean_i ψ^c_θ(x_i)
  θ ← Update(θ, ∇_θ L)               # Riemannian Adam / heavy ball
```

Warm start (App. E.2), which the ablation (App. G.1) says is decisive on tori:
```
y₀(x) = Π_M( Σ_k softmax_k( (ψ_θ(y_k) − c(x,y_k)) / γ ) · y_k )
```

Differentiation (App. E.2, eq. 19): because `∇_y F(x, y*) = 0` at the inner
optimum, `∇_θ ψ^c_θ(x) = ∇_θ ψ_θ(y*(x))` — **the inner loop needs no tape at
all**, only a stop-gradient at its boundary. The paper monitors the stationarity
residual `‖−log_ỹ(x) − ∇ψ_θ(ỹ)‖₂` to bound the bias this introduces.

Evaluation (App. F.3) needs the intrinsic Jacobian via the implicit function
theorem, not by unrolling:
```
F(x,y) = −log_y(x) − ∇ψ_θ(y) = 0
dT(x)  = −(D_y F)⁻¹ ∘ (D_x F)
J(x)   = −[E_yᵀ (D_yF)_amb E_y]⁻¹ [E_yᵀ (D_xF)_amb E_x] ∈ R^{p×p}
log|det dT| = log|det J|                    # then KL and ESS
```

### 1.2 The number that matters to us

| Manifold | Method | KL ↓ | ESS ↑ | wall-clock |
|---|---|---|---|---|
| S² | RCPM | **0.0037** | **0.996** | **37.3 s** |
| S² | RNOT (FPS) | 0.03 | 0.97 | 986 s |
| T² | RCPM | 0.93 | 0.55 | **62.3 s** |
| T² | RNOT (FPS) | **0.13** | **0.93** | 1022 s |

Run on AMD MI300X (192 GB).

**RNOT wins on quality in the hard case and on dimension-stability everywhere,
and loses wall-clock by ~26×.** That gap is not an algorithmic tax — it is
`B × K = 256 × 2500 = 640k` tiny sequential steps per outer iteration, each a
log-map, a two-layer MLP input-gradient, and an exp-map on a `p`-dimensional
point. Framework-level (per-step dispatch) execution is the worst possible shape
for it. **The paper's own weakness is a compiler problem, and it is exactly the
class of problem Tessera exists to solve.**

---

## 2. Why this belongs in Tessera

Three independent reasons, in decreasing order of durability:

1. **It forces two seams we are missing anyway and that are not OT-specific.**
   A general `stop_gradient` primitive and an implicit-function-theorem
   differentiation seam (`custom_root`) are prerequisites for deep equilibrium
   models, bilevel/meta-learning, proximal and ADMM layers, differentiable
   physics, and differentiable convex solvers. The Python reference lane now
   has `custom_root`, IHVP, and adjoint-state helpers, but the compiler has
   neither seam: the only stop-gradient in the tree is `jepa_stop_gradient`, a
   model-specific op in `models/jepa.py`, and `NewtonAutodiff.cpp` remains an
   annotation-only scaffold rather than a value-producing implicit-root rule.

2. **It makes *geometry* a first-class IR object, which is our own thesis
   applied one axis further.** Tessera already treats tiles, memory spaces,
   precision, and parallelism as IR objects rather than runtime heuristics. The
   manifold is the same kind of thing: `src/solvers/ebm/lib/Dialect/EBM/EBMOps.td`
   *already* carries a `manifold` attribute (`"euclidean" | "sphere" |
   "bivector"`) on `ebm.langevin_step`, and `EBMCanonicalize` already mirrors it
   as `tessera.ebm.manifold`. That enum is currently trapped inside the EBM
   dialect. RNOT is the forcing function to lift it into a shared contract.

3. **The inner loop is the cleanest possible test case for the Decision #28
   arbiter.** It has a fixed trip count, no tape (envelope theorem), no
   cross-sample dependence, and a small working set — a perfect fusion region
   with an unambiguous tier-1/tier-2/tier-3 comparison. And the warm-start step
   is *attention-shaped*: `[B,D] × [K,D] → [B,K]` scores, softmax over
   `(ψ(y_k) − c(x,y_k))/γ`, weighted sum of `y_k`. It is attention with a
   geodesic-distance bias, so it rides the existing `attn_bias` substrate
   (Decision: the same substrate DFlash and varlen two-stream attention ride).

A fourth, softer reason: the paper's hardware is MI300X. Our ROCm lane is the
one with real compiler-generated, hardware-verified execution (gfx1151) and
already-precompiled hsaco as its dominant lane. Reproducing this workload is a
credible ROCm-lane story, not an Apple-only demo.

---

## 3. Operator gap analysis

Inventoried against `python/tessera/compiler/op_catalog.py` (the acceptor),
`primitive_coverage.py` (the audit registry), `TesseraOps.td`, and the existing
Python surface. **"Exists" below means present in the catalog — not that its
twelve contract axes are closed.**

### 3.1 Geometric primitives — the whole layer is missing

| Paper concept | Signature | Today |
|---|---|---|
| `d(x,y)` geodesic distance | `(x, y, manifold) → scalar` | **missing.** `hyperbolic.py` has `poincare_distance`/`upper_half_plane_distance` as scalar host helpers for a different purpose; `losses.py` uses ambient `np.linalg.norm` |
| `d(x, ℓ_m)` landmark embedding | `[B,D] × [L,D] → [B,L]` | **missing — and there is no `cdist`/pairwise-distance op of any kind in the catalog** |
| `exp_x(v)` exponential map | `(x, v, manifold) → point` | **missing as an op.** A sphere retraction (`y/‖y‖`) is inlined inside `ebm/geo_sampling.py::sphere_langevin_step` |
| `log_y(x)` logarithm map | `(y, x, manifold) → tangent` | **missing.** Nothing in the tree computes an inverse exponential map |
| `P_x v` tangent projection | `(v, x, manifold) → tangent` | **private.** `ebm/geo_sampling.py::_project_to_tangent_plane`, sphere-only, underscore-private, not an op |
| `Π_M(z)` manifold projection | `(z, manifold) → point` | **partially inlined** (the normalize inside the Langevin retract) |
| `E_x ∈ R^{D×p}` orthonormal tangent basis | `(x, manifold) → [D,p]` | **missing.** `qr` is in the catalog and is the natural builder |
| Riemannian gradient `∇ψ(y)` | autodiff rule, `= P_y(∂ψ/∂y)` | **missing as a rule.** Ambient reverse-mode exists in `autodiff/vjp.py`; nothing is manifold-aware |

### 3.2 Differentiation seams — missing and broadly needed

| Paper concept | Needed | Today |
|---|---|---|
| envelope theorem at the argmin | general `stop_gradient` primitive | **missing.** Only `jepa_stop_gradient` (model-specific, `models/jepa.py:183`) |
| IFT Jacobian `−(D_yF)⁻¹ D_xF` | `custom_root` / implicit-diff VJP | **scaffolded, not missing** — corrected 2026-08-02. `src/solvers/core/passes/NewtonAutodiff.cpp` walks `tessera_solver.implicit` ops and its header specifies exactly `dF/dx = -(dR/dx)⁻¹·dR/du`; the body only *annotates* `tessera_solver.{vjp,jvp} = "generated"` and defers values to runtime. R2 must **finish that pass**, not build a parallel mechanism. See [`AUTODIFF_ARCHITECTURE_REVIEW.md §B8`](AUTODIFF_ARCHITECTURE_REVIEW.md) |
| `log|det J|` | `slogdet` | **missing.** No `det`, no `logdet`, no `slogdet` in the catalog |
| `[D_yF]⁻¹ [D_xF]` (dense `p×p`) | general `linalg.solve` | **partial.** `tri_solve` and `cholesky_solve` exist; `D_yF` is neither triangular nor SPD in general |
| JVP/VJP against `p` basis vectors | batched JVP over a basis | `autodiff/jvp.py` exists; batched-over-basis is untested for this shape |

`qr`, `svd`, `cholesky`, `lu`, `tri_solve`, `cholesky_solve` are all in the
catalog with `linalg_decomposition` / `linalg_solver` lowerings, so the linear
algebra tail is a smaller lift than the geometry.

### 3.3 RNOT-level composite ops

| Paper concept | Signature | Today |
|---|---|---|
| `c`-transform `ψ^c(x) = min_y F(x,y)` | fused `(x, ψ, manifold, K, α) → (ψ^c, y*)` | **missing.** Nearest structural analog is `ebm_sphere_langevin_step`, which has the identical inner shape (tangent-project → step → retract) and already has an Apple MSL kernel plus x86/ROCm affine lanes |
| softmin/LSE warm start | `(x, {y_k}, ψ(y_k), γ) → y₀` | **missing.** `softmax`, `log_softmax`, `logsumexp` exist; the barycenter + `Π_M` composite does not |
| `T(x) = exp_x(−∇φ(x))` | `(x, φ, manifold) → y` | **missing** |
| semi-dual loss | `−mean ψ(y) − mean ψ^c(x)` | **missing.** `losses.py::wasserstein_distance` is a 1-D sorted-quantile L1, unrelated to manifold OT |
| FPS landmark selection | greedy max-min over `[N,D]` | **missing** |
| Riemannian Adam | exp-map parameter update | **missing.** `optim.py` has nine optimizers, all Euclidean |
| ESS diagnostic | `(Σw)²/Σw²` | **missing.** `kl_divergence` exists in `losses.py` |

**Total new surface: ~19 primitives.** Roughly 8 geometric, 5 differentiation
seams, 6 RNOT composites. About half are small; the concentrated work is the
`c_transform` fused loop and the implicit-diff seam.

### 3.4 The one correctness hazard to design against

Decision #21 says an unsupported lowering must emit a stable diagnostic and never
silently no-op. Manifold geometry has a nastier version of this failure: a
**silent Euclidean fallback produces plausible, well-conditioned, entirely wrong
numbers**. `exp_x(v) ≈ x + v` and `log_y(x) ≈ x − y` are correct to first order,
so a fallback converges, looks stable, and quietly reports the wrong transport
map. Every geometric op must therefore treat `(manifold, target)` as a hard
dispatch key with a named diagnostic on miss — no ambient default, ever.

Second hazard, numeric: `log_y(x)` is undefined at the cut locus (antipodes on
the sphere), and the naive `d(x,y) = acos(⟨x,y⟩)` loses catastrophic precision as
`⟨x,y⟩ → ±1`. The `atan2(‖x×y‖, ⟨x,y⟩)` form is mandatory. Per Decision #15a this
belongs in `numeric_policy`, and manifold **coordinates** should carry an fp32
storage floor — bf16 coordinates on a unit sphere are not usable, whatever the
accumulator says.

---

## 4. Plan

Six phases. R0–R2 are the durable compiler assets and are worth doing whether or
not RNOT itself ever ships; R3–R5 are the paper. Each phase follows the proven
five-seam pattern used for the EBM geometric ops (numpy reference → catalog +
coverage registration → Graph/Target IR → backend kernel → manifest + tests +
dashboards).

### R0 — Manifold contract (spec only, no kernels)

Write `docs/spec/MANIFOLD_SPEC.md` defining the manifold as a first-class IR
attribute, lifting the enum out of the EBM dialect:

```
manifold ::= "euclidean" | "sphere" | "torus" | "hyperbolic"
           | "spd" | "bivector" | "product"<manifold, ...>
```

Contract per manifold: `dim`, ambient `D`, injectivity radius, and the required
op set (`exp`, `log`, `dist`, `tangent_project`, `tangent_basis`, `project`).
`"bivector"` is carried forward from the existing EBM enum so the GA/Clifford
lane is not orphaned; `"product"` exists because `T^n = (S¹)^n` and it gives a
free cross-path oracle (§R5).

- Deliverable: spec + a decision entry in `CLAUDE.md` if the enum lift is
  accepted.
- Blocking question for the maintainer: extend `tessera_ebm`'s enum in place, or
  introduce a shared `tessera_geom` dialect that EBM then consumes? Recommend the
  latter — EBM's enum is already load-bearing and a shared dialect avoids a
  layering inversion — but this is an architecture call, not mine to make
  unilaterally.
- Effort: ~2 days.

### R1 — Geometric primitive layer

New `python/tessera/geom/`: `exp_map`, `log_map`, `geodesic_distance`,
`geodesic_cdist`, `tangent_project`, `tangent_basis`, `manifold_project`,
`manifold_random`. numpy reference for `euclidean`, `sphere`, `torus`, and
`product` first; `hyperbolic` and `spd` deferred.

- Register in **both** `op_catalog.py` and `primitive_coverage.py` (Decision #24).
- Register VJPs in `autodiff/vjp.py` so the coverage axes auto-flip.
- **Refactor, do not duplicate:** rewrite `ebm/geo_sampling.py`'s
  `_project_to_tangent_plane` and the inlined sphere retract to call the new ops,
  and route `hyperbolic.py` through `geodesic_distance(manifold="hyperbolic")`
  when that lands. Two implementations of the sphere is exactly the seam problem
  described in `CLAUDE.md` for the Apple lane; do not create a second one.
- `geodesic_cdist` is the shape that matters: `[B,D] × [L,D] → [B,L]`, matmul-like,
  fusable with the MLP that consumes it.
- Tests: `tests/unit/test_geom_primitives.py` — exp/log round-trip inside the
  injectivity radius, distance symmetry and triangle inequality, tangency
  (`⟨log_y(x), y⟩ = 0` on the sphere), `d` against `atan2` reference at
  near-antipodal and near-identical inputs.
- Effort: ~1.5 weeks.

### R2 — Differentiation seams (the reusable payoff)

1. **`stop_gradient`** as a real primitive with a tape barrier, not a model op.
   Migrate `jepa_stop_gradient` onto it.
2. **`custom_root(F, solver)`** — implicit-function-theorem VJP:
   given `F(x, y*) = 0`, `dy*/dx = −(∂F/∂y)⁻¹ (∂F/∂x)`. Register through the
   existing `@custom_primitive` machinery in `custom.py`, which already has
   `def_vjp` / `def_jvp` / `def_batching` / `def_transpose` hooks.
   **On the compiler side this means completing `NewtonAutodiff.cpp`** — emitting
   real `tessera_solver.residual` + `linear_solve` ops in place of its current
   annotations — *not* adding a second implicit-diff path. This is a shared
   deliverable with the autodiff track; see
   [`AUTODIFF_ARCHITECTURE_REVIEW.md §5`](AUTODIFF_ARCHITECTURE_REVIEW.md)
   ("Alongside — finish `NewtonAutodiff`"). Budget it once, not twice.
3. **`slogdet`** and a general **`linalg.solve`**, lowering alongside the existing
   `linalg_decomposition` / `linalg_solver` families.
4. Batched JVP against a tangent basis — the App. F.3 note that the projected
   Jacobians should come from `p` JVP/VJP products rather than a dense `D×D`
   matrix is a real efficiency contract; encode it as the lowering rule.

- Tests: finite-difference agreement for `custom_root` on a closed-form fixed
  point; `slogdet` against `lu`; and the **envelope-vs-unroll differential test**
  — `∇_θψ^c` computed by unrolling `K` steps must agree with the stop-grad form
  as the stationarity residual `‖g_θ‖ → 0`. That test is a compiler test, not an
  OT test, and it is the highest-value single fixture in this plan.
- Effort: ~2 weeks. **This phase is independently valuable and is the one to keep
  if the rest is cut.**

### R3 — RNOT composite ops

`python/tessera/ot/`: `c_transform`, `softmin_barycenter`, `transport_map_apply`,
`semi_dual_ot_loss`, `farthest_point_sample`, plus `riemannian_adam` in
`optim.py` and `effective_sample_size` in `losses.py`.

`c_transform` is the centerpiece and should be a **single fused op with an
explicit trip count**, not a Python loop:

```
c_transform(x, psi, manifold, *, inner_steps=K, step=α, warm_start="lse", γ)
  → (psi_c, y_star, residual)
```

It returns the stationarity residual as a third output so the App. E.2 diagnostic
is a first-class contract rather than a debugging afterthought, and it applies
`stop_gradient` to `y_star` internally so the envelope theorem is structural.

- Lower to `tessera.control_for` (already in `TesseraOps.td:2633`) with a
  stop-gradient region boundary.
- Tests: Dirac collapse (`ν = δ_{y₀} ⟹ T ≡ y₀`, which the paper proves in
  App. line 1486 — a free exact oracle), identity transport (`μ = ν ⟹ T = id`),
  and the residual monotonically decreasing in `K`.
- Effort: ~2.5 weeks.

### R4 — Backend lanes

Target IR before hardware (Decision #19). Order chosen by evidence value:

| Order | Target | Rationale | Precedent to copy |
|---|---|---|---|
| 1 | Apple GPU (MSL) | fastest iteration; local M1 Max | `tessera_apple_gpu_ebm_sphere_langevin_step_f32` — the tangent-project → step → retract fusion is the same shape |
| 2 | ROCm gfx1151 | **the paper's own hardware class**; our strongest hardware-verified lane | ROCm affine-Langevin lane in `ebm/energy.py` |
| 3 | x86 AVX-512 | CPU spine, cheap once ROCm exists | `_try_x86_ebm_affine_langevin_step_f32` |
| 4 | NVIDIA sm_120 | breadth per the P0 queue | sm_120 sealed packets |

The kernel to write is the **fused inner step**: one thread block per sample,
`K` iterations resident in registers/LDS, no host round-trip. This is where the
26× goes.

Three existing EBM passes apply almost unchanged and should be generalized rather
than cloned — `EBMFuseEnergyGrad` (fuse `ψ` forward with `∇ψ` so the inner loop
reuses activations), `EBMPipelineCandidates` (map the independent-sample
dimension across streams), and `EBMCheckpointInnerLoop`. Note the third one:
**under the envelope theorem there is nothing to checkpoint**, because the inner
trajectory is never differentiated through. Recording that the remat pass
correctly becomes a no-op here — rather than silently rematerializing 2500 dead
steps — is itself a compiler result worth a fixture.

- Effort: ~2 weeks Apple, ~2 weeks ROCm, ~1 week each x86/NVIDIA.

### R5 — Evidence and oracles

Beyond the standard `backend_manifest` + drift-gated dashboard work, this
workload admits unusually strong metamorphic oracles — good material for the
Evaluator program (`EVALUATOR_PLAN.md` §9.5), which scores exactly this kind of
cross-path invariant:

| Oracle | Invariant | Catches |
|---|---|---|
| Isometry equivariance | `T(Rx) = R·T(x)` for `R ∈ O(D)` stabilizing `M` | any ambient/Euclidean leakage |
| Dirac collapse | `ν = δ_{y₀} ⟹ T ≡ y₀` (exact, proved in the paper) | inner-solver convergence |
| Identity transport | `μ = ν ⟹ T = id` | potential-scaling and sign errors |
| Product factorization | `T^n` via `product<S¹,…>` vs. a monolithic torus kernel | the DESIL cross-path oracle, for free |
| exp/log round-trip | `log_x(exp_x(v)) = v` inside `inj(x)` | per-target kernel drift |
| Envelope vs. unroll | stop-grad `∇_θψ^c` ≡ unrolled, as `‖g_θ‖→0` | the R2 seam itself |
| Monge gap | `E[c(x,T(x))] + L(θ) → 0` | end-to-end optimality (paper's own metric) |

Benchmark against the paper's numbers on S²/T² — KL, ESS, wall-clock — using the
stable benchmark JSON schema (Decision #12). **The headline target is wall-clock
parity with RCPM at RNOT's quality**, i.e. closing the 986 s → ~40 s gap. Anything
better than ~5× over RCPM is a publishable compiler result on its own.

- Effort: ~1.5 weeks.

### Sequencing and total

```
R0 (2d) → R1 (1.5w) → R2 (2w) ─┬→ R3 (2.5w) → R4 (Apple 2w → ROCm 2w → x86/NV 2w) → R5 (1.5w)
                               └→ (R2 ships standalone value regardless)
```

~13–14 weeks single-track. R0–R2 is ~4 weeks and is the part I would defend
independently of the paper.

---

## 4a. Hardening the two failure modes

Both hazards named in §3.4 and §R4 turned out to be **present defects in shipped
code**, not future risks. Evidence and full context:
[`domain/GA_EBM_ARCHITECTURE_REVIEW.md`](../domain/GA_EBM_ARCHITECTURE_REVIEW.md)
§1.1 and §1.5. The designs below therefore land in R0/R2 as fixes, not as
guardrails against a hypothetical.

### H1 — Semantic keys never default

**The defect.** `src/solvers/ebm/lib/Passes/Canonicalize.cpp:56` contains a
warning-plus-Euclidean repair for a missing `manifold`, but the registered ODS
op already requires the `StrAttr`; normal verification rejects absence before
the repair.  The reachable defects are that the string's value is not closed by
an enum/verifier and **no backend reads it**—every hit for `manifold` under
`src/compiler/codegen/` is a comment. Manifold correctness is carried entirely
by Python function-name dispatch.

**The generalizing principle.** The bug is not "EBM got one default wrong." It is
that Tessera has no rule separating two kinds of attribute:

> **Decision #21a — semantic keys never default; performance keys may.**
> An attribute that selects *semantics* fails closed on absence: missing is an
> error, unknown is an error, and no lowering may substitute a value. An
> attribute that selects *performance* may fall back, with a diagnostic.
> A wrong tile size is slow. A wrong manifold, algebra signature, math mode,
> rounding mode, or distribution is **wrong**, and wrong in the worst way —
> first-order-correct, convergent, and plausible.
>
> Semantic: `manifold`, `algebra`, `math_mode`, `rounding_mode`, `distribution`,
> `dtype`. Performance: tile sizes, pipeline depth, `auto_batch`, stage counts,
> checkpoint budget.

This sharpens Decision #21 (stable diagnostics on unsupported lowering) with the
case Decision #21 does not cover: not *unsupported*, but *unspecified*.

**The design, five parts.**

1. **Enum, not string.** Replace `StrAttr:$manifold` with an ODS `EnumAttr`
   (`ManifoldKindAttr`). Unknown values then fail at *parse* time. This alone
   eliminates the `"Sphere"` / `"sphere2"` typo class, which no amount of
   downstream checking catches.
2. **Delete the default.** Remove the `emitWarning` + default branch in
   `Canonicalize.cpp`; make absence a verifier error on the op itself, so it
   fails even if the canonicalize pass never runs. **Copy the sibling dialect
   verbatim** — `AnnotateAlgebra.cpp:70` already does exactly this for
   `algebra` (`emitError` + `WalkResult::interrupt()`). One dialect in
   `src/solvers/` is already right; the fix is to stop the other from being
   different.
3. **Put the manifold in the type, not only the attribute.** A
   `!tessera.point<manifold, D>` type — or a `manifold` field on the tensor's
   `layout`, which Decision #15a already establishes as one of the six canonical
   attributes. An attribute cannot catch a *mismatch across an op boundary*; a
   type can. `exp_map` on a sphere point and `log_map` on a torus point must not
   typecheck against each other.
4. **Manifold joins the dispatch key.** Extend `backend_manifest.py`'s
   `BackendKernelEntry` key from `(op, target, dtype)` to
   `(op, manifold, target, dtype)`. A miss raises a named diagnostic —
   `TESSERA-GEOM-001: no kernel for (exp_map, torus, apple_gpu, f32)` — and
   never an ambient fallback. `manifold="euclidean"` stays a perfectly legal
   *value*; the rule is that nothing may *arrive* at it by default.
5. **Tripwires that would have caught the present bug.**
   - Lit fixture: `tessera-ebm-canonicalize` on a `langevin_step` with no
     `manifold`, FileCheck for an **error**.
   - Negative lit fixture: assert the literal string `"euclidean"` never appears
     in output IR that did not contain it in input. This is the specific
     regression, stated as a test.
   - Unit test over `backend_manifest`: every manifold-carrying op family has an
     entry for every allowed manifold, or an explicit terminal
     `no_kernel_required` reason. No silent holes in the key space.
   - Metamorphic (rides R5): isometry equivariance `T(Rx) = R·T(x)` fails loudly
     under a Euclidean fallback. It is the runtime backstop for whatever the
     static rules miss.

### H2 — A transform that can be a no-op must prove it is one

**The defect.** `EBMCheckpointInnerLoop` performs **no liveness and no
differentiability analysis**. Its entire body walks every `scf.for`, and if it
finds a step op, marks every step `recompute_step` and sets a hardcoded
`budget = 4`. It never asks whether anything downstream consumes the trajectory.

This bypasses Decision #10's own discipline — "budget-guided … greedy live-set
scan … only pure ops qualify" — which the general `InsertRecomputePass`
implements and this domain pass does not. On the `c`-transform loop it would
annotate 2500 dead steps. Repository-wide inspection finds no runtime or codegen
consumer for those EBM attributes, so it does not currently instruct a backend
to keep four states; it emits inert, misleading policy. The budget comment reads
"enough to fit a typical T=16 chain" while the workload is T=2500, which would
become hazardous if a consumer were added without first replacing the policy.

**The design, five parts.**

1. **First remove the inert pass from the default pipeline.** Retain it as a
   standalone pass only if it is explicitly classified experimental; otherwise
   delete it under Decision #29 because its attributes have no consumer.
2. **Supply the missing concept: differentiation demand in the shared remat
   system.** The pass
   over-annotates because nothing in the IR states "this trajectory is needed for
   a backward pass." Add an explicit gradient-boundary marker — and note this is
   **the same object as R2's `stop_gradient` region**, not a second mechanism.
   A value is remat-eligible only if it is live-in to a differentiation boundary.
3. **Gate shared loop rematerialization on it.** A remat decision exists only when some value defined in the
   loop body reaches such a boundary. A loop whose results are all consumed
   through `stop_gradient`, or which yields only its final state, gets **no**
   annotation and no `checkpoint_loop` attribute.
4. **Make the no-op observable and tested.** Preserve the existing non-EBM-loop
   negative fixture. When demand-aware loop rematerialization lands, add:
   - `checkpoint_inner_loop_basic.mlir` *(exists)* — demand present → annotated.
   - **new** `checkpoint_skips_envelope_loop.mlir` — loop result crosses
     `stop_gradient` → `CHECK-NOT: tessera.ebm.recompute_step` and
     `CHECK-NOT: tessera.ebm.checkpoint_loop`.
   - **new** `checkpoint_skips_forward_only_loop.mlir` — no-grad module → same.

   The `CHECK-NOT` fixture *is* the proof asked for, and it costs almost nothing.
5. **Count shared remat decisions, don't just assert absence.** Have the pass emit
   `tessera.remat.steps_annotated = N` and assert `N == 0` on the envelope
   fixture. Absence of an attribute is weak evidence — it also holds if the pass
   silently failed to run. A zero is a measurement.
6. **Do not recreate the domain pass.** The end state is that `EBMCheckpointInnerLoop`
   stops existing as a separate syntactic pass and becomes a *registration* of
   EBM/OT step ops into `InsertRecomputePass`'s existing live-set scan. Domain
   passes should contribute op knowledge, not reimplement analysis — one remat
   policy, one place to be correct. Budget comes from `--memory-budget-mb` per
   Decision #10, derived from live-set size × trip count, not a constant.

**The generalizing principle:**

> **Decision #10a — an eligibility-marking pass ships a negative fixture.**
> Any pass that annotates work as "eligible" (rematerializable, fusable,
> pipelineable) must gate on demand analysis rather than syntactic presence, and
> must ship at least one fixture in which the correct output is *no annotation*.
> A pass with only positive fixtures has never been tested for the case where
> doing nothing is right.

### Where these land

| Item | Phase | Effort |
|---|---|---|
| H1.1–H1.2 enum + delete default + verifier | R0 | ~3 days |
| H1.3 manifold in the type | R0/R1 | folded into R1 |
| H1.4 dispatch key + `TESSERA-GEOM-001` | R1 | ~2 days |
| H1.5 tripwires | R1, R5 | ~2 days |
| H2.1–H2.2 demand gating | R2 (shares `stop_gradient`) | ~2 days on top of R2 |
| H2.3–H2.4 fixtures + counter | R2 | ~2 days |
| H2.5 fold into `InsertRecomputePass` | R4 | ~1 week |

H1 and H2 add roughly **1.5 weeks** to R0–R2 and close two live defects on the
EBM path independently of whether R3–R5 proceed.

---

## 5. Constraints this plan must respect

- **Decision #23 (standalone).** `geomstats`, POT, `jax`, and the authors'
  reference implementation are *vocabulary only*. Nothing imports them.
- **Decision #19 (Target IR first).** No Tile IR → MSL/PTX/HIP shortcut for the
  fused inner loop, however tempting given that the kernel is small.
- **Decision #21 (stable diagnostics), sharpened.** `(manifold, target)` is a hard
  dispatch key. A missing pair raises a named diagnostic. There is **no** ambient
  Euclidean fallback — see §3.4.
- **Decision #24 (dual registration).** Every new primitive lands in
  `op_catalog.py` *and* `primitive_coverage.py`.
- **Decision #26 (audit flow).** Update `compiler/COMPILER_AUDIT.md` as phases
  land; let `docs/audit/generated/` carry the counts; never hand-edit generated
  docs.
- **Decision #28 (arbiter, ROCm/CUDA set the ceiling).** The `c_transform` fused
  loop is a tier-2 plugin candidate; a hand-written MSL/HIP version stays a
  first-class arbiter candidate and is displaced only by a measured win in
  accuracy budget.
- **Decision #15a (numerics).** Manifold coordinates carry an fp32 storage floor;
  the `atan2` distance form is normative; cut-locus behavior is specified, not
  incidental.

## 6. Non-goals

- Hyperbolic and SPD manifolds in the first pass (`euclidean`, `sphere`, `torus`,
  `product` only).
- Entropic/Sinkhorn OT. This is the unregularized Monge–Kantorovich semi-dual
  path; a Sinkhorn lane is a separate, larger op family.
- Reproducing the paper's continental-drift experiment. It needs GPlates data and
  proves nothing about the compiler.
- Beating RCPM on KL. RNOT does not, on S², and that is not the paper's claim.

## 6a. Relationship to the EBM and GA tracks

A review of those surfaces against the current compiler
([`domain/GA_EBM_ARCHITECTURE_REVIEW.md`](../domain/GA_EBM_ARCHITECTURE_REVIEW.md))
found that RNOT is not a new workload family. It is a third instance of one that
already exists twice:

| Family | Loop | Scalar field | Manifold | Trip count |
|---|---|---|---|---|
| EBM Langevin / AIS | Euler–Maruyama | energy `E` | euclidean / sphere / bivector | 10–10³ |
| GA rotor flows | rotor integration | GA-valued potential | even subalgebra | 10–10² |
| RNOT `c`-transform | Riemannian GD | dual potential `ψ_θ` | sphere / torus | up to 2500 |

All three are a fixed-trip-count, device-resident iterative refinement of a point
on a manifold under the gradient of a scalar field, with an explicit
differentiability boundary. All three are implemented separately. All three are
host-bound at the gradient — EBM's `energy_fn` is a Python closure, so even its
"fused" MSL Langevin kernel evaluates `∇E` on the host and ships the result as
data, one round trip per step. All three carry manifold information no backend
reads.

**This changes the plan's framing.** R0–R2 should be built as the *shared*
substrate — a manifold contract, a scalar-field-as-region contract, and a
demand-gated remat policy — with EBM, GA, and RNOT as three configurations of one
fused-loop region. That is also exactly the shape Decision #28's arbiter selects
kernels for. Scoping R0–R2 to RNOT alone would build the third copy of something
that should be built once.

Two consequences for sequencing:

- The `ga/` and `ebm/` call sites are the **first consumers** of R1's geometric
  ops, ahead of any RNOT code. `ebm/geo_sampling.py::_project_to_tangent_plane`
  and the inlined sphere retract are the ops R1 defines, currently private and
  sphere-only.
- The review's items 1–4 (~1.5 weeks: autodiff-based gradients instead of finite
  differences, the H1 manifold fix, the H2 remat fix, and `.td` summary drift)
  close two live defects and should proceed **regardless of this plan's fate**.

## 7. Open questions for the maintainer

1. **Enum lift:** extend `tessera_ebm`'s `manifold` attribute in place, or a new
   shared `tessera_geom` dialect? (§R0 — recommend the latter.)
2. **Priority against the live queue.** MASTER_AUDIT's P0 is NVIDIA exact-target
   breadth and P1 is sequence mixers. This is neither. R2 (the differentiation
   seams) has the strongest claim to jumping the queue on its own merits; R3–R5
   are honestly new scope.
3. **Is a 26× wall-clock gap on someone else's algorithm the demo we want?** It
   is an unusually clean one — the bottleneck is a single fused loop with a
   proven no-tape boundary. Per §6a it is also *not* a new workload family, which
   weakens the objection: it is the third instance of a loop shape already in the
   tree twice.
4. **Do Decisions #21a and #10a get adopted?** Both are proposed in §4a as
   general rules, not RNOT-specific ones, and both would apply retroactively to
   existing passes. #21a in particular would make the `manifold` default an
   error rather than a warning, which is a behavior change on the EBM path.
