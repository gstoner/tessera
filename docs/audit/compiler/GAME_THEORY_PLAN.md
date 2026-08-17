---
last_updated: 2026-08-15
audit_role: plan
plan_state: open
status: G1 COMPLETE (2026-08-15, two slices) — python/tessera/game/ ships the
        four zeta/Möbius butterflies, coalition_marginal, weight-parameterized
        semivalue, boltzmann_value (closed-form VJP, FD-checked; T<0 legal,
        T=0 fails closed per H5), coalition_excess (jointly linear, declared
        transpose; zeta-of-additive-game oracle), and segmented mex
        (non-differentiable by construction; Sprague–Grundy nim-XOR oracle).
        fp64 per §6; fail-closed lattice/weight contracts; nine catalog rows +
        spec rows + regenerated dashboards (Decision #24/#26); oracles 1–7 +
        11 green (tests/unit/test_game_lattice.py, 26 tests; mypy clean).
        Open question 3 RESOLVED: segment_reduce/cum*/control.scan already
        existed — G3 consumes them, no new ops (#31). Every algebraic claim
        numerically verified (27 checks, research/game_theory/)
source: U. Faigle, "Mathematical Game Theory: A New Approach" (lecture-note draft,
        `Mathematical_Game_Theory_new.tex`) + a maintainer-supplied list of refined
        equilibrium concepts (SPE, Bayesian, k-resilient, correlated, bounded
        rationality, Preference-CFR)
reassessment: 2026-08-15 — full mathematical verification pass
        (`research/game_theory/verify_game_theory_plan.py`, all 27 checks pass).
        Two claims were corrected by measurement: the §6 fp32 wall is
        sign-structure-dependent (nonneg games are the worst case, and fp32
        *storage* of the zeta — not just accumulation — is the failure); and the
        regret-dynamics hazard H2 was demonstrated live (two consecutive
        plausible-but-wrong Blum–Mansour implementations, no invariant caught
        either). §6.1 hazards and §9 developer-capability surface added.
---

# Game Theory — Source Review and Operator Plan

> **Routing:** start at [`README.md`](README.md). This document owns the
> coalition-lattice / equilibrium operator family and its acceptance workload;
> global ordering lives only in
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md).
>
> **Status vocabulary warning (Decision #25/#26):** everything below is
> *direction*. No row here is proof of anything. `docs/audit/MASTER_AUDIT.md`
> and `docs/audit/generated/` stay status truth.
>
> **Guardrail overrides (maintainer, 2026-08-15):** Decision #23 and the "no new
> IR" non-goal are guides that may be overridden when the system's health calls
> for it. This plan exercises that in exactly two places, both argued in
> **§4.6**: a test-only differential-oracle dependency group (§4.6.1) and a
> *generic* `tessera.butterfly_transform` op that consolidates tiling already
> duplicated by the spectral FFT lane (§4.6.2). It **declines** the override for
> a runtime LP dependency (§4.4) and for a `tessera_game` dialect (§8), because
> neither buys anything the structure does not already give. An override spent
> where it is not needed costs the same credibility as a rule followed where it
> does not fit.

---

## 1. What the source actually is

**Read this before scoping off the abstract.** `Mathematical_Game_Theory_new.tex`
is a **3376-line outline, not a finished text**. A large fraction of its section
headers carry no body:

| Chapter | Sections with substantive body | Sections that are bare headers |
|---|---|---|
| Combinatorial games | — | alternating players, recursiveness, winning strategies, algebra of games, impartial games, Grundy sums |
| Zero-sum games | LP standard form, KKT, complementary slackness, shadow prices | matrix games, equilibria, convex zero-sum, LP games |
| Potentials/Utilities/Equilibria | marginal-potential lemma, gain/cost equilibrium definition | utilities, existence of equilibria |
| n-person games | model, fuzzy games, cooperation | dynamics, equilibria, randomization, traffic flows |
| Potentials & Temperature | Boltzmann temperature, T→0± limits, temperature of matrix games | Metropolis process |
| Cooperative games | production/linear-production games, core, nuclea/nucleolus, Monge/supermodularity, Banzhaf, Boltzmann values | Möbius transform, marginal values, Shapley value, coalition formation |
| Interaction/Quantum | symmetry + Hermitian representation, spectral decomposition | interaction states/potentials, quantum model, evolutions |

So the paper is **a framing device, not a specification**. What it gives us that
is genuinely load-bearing:

1. **A game is a sequence of states of a system.** A *potential* is `v: 𝔖 → ℝ`;
   a *utility* is `U ∈ ℝ^{𝔖×𝔖}` (a function on state **transitions**); the
   marginal potential `∂v(σ,τ) = v(τ) − v(σ)` determines `v` up to a constant.
2. **Equilibrium is a local sign condition, not a solver.** `σ` is a *gain
   equilibrium* iff `U(σ,τ) ≤ 0` for every `τ` in a declared neighborhood `ℱ^σ`.
   Nash, Wardrop, and core stability are all instances — they differ only in
   what `ℱ^σ` is.
3. **Cooperative games are linear functionals on `ℝ^{2^N}`.** `(N,v)` *is* the
   map `g ↦ ⟨v|g⟩`. Values (Shapley, Banzhaf, …) are linear (or Boltzmann-
   weighted) functionals over that same space.
4. **Temperature unifies optimization and averaging.** Boltzmann `β^{(T)}_σ ∝
   e^{v_σ/T}`; `T→0⁺` recovers `max v`, `T→0⁻` recovers `min v`, `|T|→∞`
   recovers the uniform mean. Boltzmann *values* are `E^T_i(v) = Z_T^{-1} Σ_S
   ∂_i v(S) e^{v(S)/T}`.
5. **The Hermitian representation.** `Â = A⁺ + i·A⁻` (symmetric + i·skew) maps
   `ℝ^{X×X}` isometrically onto the Hermitian matrices, so *every* real
   interaction matrix acquires a real spectral decomposition.

That is a compact, coordinate-free restatement of game theory in exactly the
vocabulary Tessera already speaks: **potentials are tensors, utilities are
transition operators, equilibria are fixed points, temperature is a softmax, and
cooperative games are transforms on a `2^N` index space.**

### 1.1 One transcription error to design against

§*The value of Banzhaf* states the count `|{S ⊆ N∖{i} : T ⊆ S∪{i}}| = 2^{n−|T|−1}`
and concludes `E^{π^B}_i(v_T) = 1/2^{|T|}`. The condition `T ⊆ S∪{i}` for `i∈T`
is `T∖{i} ⊆ S`, and `|T∖{i}| = |T|−1`, so the count over `S ⊆ N∖{i}` is
`2^{(n−1)−(|T|−1)} = 2^{n−|T|}` and the value is `2^{n−|T|}/2^{n−1} = 2^{1−|T|}`
— the standard Banzhaf value of a unanimity game. The draft is off by a factor
of two.

**The lesson is the plan-relevant part, not the erratum.** A reference
implementation transcribed from this text would inherit the error silently and
still "pass" a self-consistent test. Every value operator here must be validated
against its **axioms** (efficiency, symmetry, dummy, additivity) rather than
against a transcribed closed form. That is §7.

---

## 2. The restatement: Faigle's objects → Tessera IR objects

| Faigle | Tessera today | Gap |
|---|---|---|
| System `𝔖`, states `σ` | tensor index space; `domain.Rect` | none |
| Potential `v: 𝔖 → ℝ` | a tensor | none |
| Marginal potential `∂v(σ,τ)` | pairwise difference / bit-flip difference | new op (cheap, linear) |
| Utility `U ∈ ℝ^{𝔖×𝔖}` | a transition matrix | none |
| Neighborhood `ℱ^σ` | **sparsity/adjacency structure** | this is the missing *type*, §4.1 |
| Gain equilibrium | fixed point of a projected map | `autodiff/implicit.py::custom_root`, `tessera_solver.implicit`, `NewtonAutodiff.cpp` — **already built** |
| Mixed strategy ∈ Δ | `relaxation.sparsemax` (exact support-set Jacobian) | **already built** |
| Boltzmann `β^{(T)}` | softmax with temperature + LSE | **already built** (`losses.logsumexp`, FA online-softmax) |
| Metropolis process | `ebm.langevin_step` / MCMC family, Philox streams | partially built |
| Cooperative `v ∈ ℝ^{2^N}` | a `[..., 2^n]` tensor | new *layout*, §4.1 |
| Möbius / zeta transform | — | **new op; a radix-2 butterfly, §4.2** |
| Shapley / Banzhaf / semivalues | — | new op (one op, weight-parameterized) |
| Boltzmann values | — | new op = **n-head online softmax over the lattice** |
| Core / nucleolus | — | LP-shaped; solved as a saddle point, §4.4 |
| Hermitian representation `Â` | `complex.py`, `solvers/spectral` | new op (trivial, exactly testable) |
| Grundy numbers, nim-sum | `bitwise_xor`, `popcount` in `op_catalog` | `mex` is new |

The headline: **almost every hard piece already exists.** What is missing is one
transform, one index-space convention, and a family of contractions over it.

---

## 3. Why this belongs in Tessera

Four arguments, ordered by how load-bearing they are.

### 3.1 The Möbius transform is an FFT that Tessera can already almost run

For `v ∈ ℝ^{2^N}`, the zeta and Möbius transforms

```
(ζv)(T) = Σ_{S⊆T} v(S)          (μv)(T) = Σ_{S⊆T} (−1)^{|T∖S|} v(S)
```

are mutually inverse and computable by the Yates / subset-sum recurrence:

```
for i in 0..n-1:                       # n stages
  for T with bit i set:                # 2^{n-1} butterflies per stage
    f[T] += f[T ^ (1<<i)]              # zeta   — butterfly [[1,0],[1,1]]
    f[T] -= f[T ^ (1<<i)]              # Möbius — butterfly [[1,0],[-1,1]]
```

`n·2^{n−1}` adds, `n` stages, stride `2^i` at stage `i`. **That is structurally
the same schedule as a radix-2 Stockham FFT with a constant real twiddle.** The
consequences are concrete, not analogical:

* `python/tessera/compiler/emit/spectral_candidates.py` already ships a
  `SpectralFFTRegion` + `register_op_kind(OP_SPECTRAL_FFT, _verify_fft)` arbiter
  lane with Stockham radix-4 CPU and **gfx1151 HIP** kernels and an F4
  reference-verify gate. A `SubsetZetaRegion` is a sibling class with the same
  three methods (`reference`, `probe_input`, verify hook) — it drops into the
  Decision #28 candidate registry with **no new infrastructure**.
* Tiling is the FFT tiling problem: the last `log2(tile)` stages are
  stride-local and fit in shared/threadgroup memory; earlier stages stream.
* **Sharding is the distributed-FFT binary-exchange pattern.** Shard the lattice
  on the top `k` bits: the first `n−k` stages are rank-local, the last `k`
  stages each pair rank `r` with rank `r ⊕ 2^j` and exchange half the local
  buffer. Total comm `k·(local/2)`. This is a real stress test for
  `distributed_planner.py` and the collectives layer, with a known-correct
  reference to check against.
* It is **linear**, so `custom_primitive(..., linear=True)` + `def_transpose`
  yields VJP *and* JVP for free (`custom.py::def_transpose` registers both) —
  and the adjoint is exact and closed-form: **the transpose of subset-zeta is
  superset-zeta** (butterfly `[[1,1],[0,1]]`). This gives Decision #29's
  `transpose_rule` axis a genuine, non-decorative consumer.

### 3.2 Boltzmann values are attention over the coalition lattice

```
E^T_i(v) = (1/Z_T) Σ_{S⊆N} ∂_i v(S) · e^{v(S)/T},   ∂_i v(S) = v(S) − v(S Δ {i})
```

Read it as: **softmax weights over `2^n` "keys" (`v/T`), contracted against `n`
different "value" channels (the bit-flip differences).** This is an `n`-head
softmax-weighted reduction with `O(n)` running state over a `2^n` stream — one
pass, running max + running `Z` + `n` running numerators, rescaled exactly like
flash-attention's online softmax. Tessera already owns that rescaling in
`emit/apple_msl.py` (attention with online softmax) and in the
[`LSE_CHECKPOINT_CONTRACT.md`](LSE_CHECKPOINT_CONTRACT.md) numerics work.

So the flagship cooperative-game operator is **a kernel we can already emit**,
on Apple GPU day one, by parameterizing an emitter we shipped.

### 3.3 Equilibria are exactly the implicit-differentiation seam we built

`autodiff/implicit.py` gives `custom_root`, adjoint state and IHVP;
`src/solvers/core/passes/NewtonAutodiff.cpp` lowers `tessera_solver.implicit`'s
differentiation contract through the implicit function theorem, producing a
value-carrying VJP function rather than an annotation. A Nash equilibrium of a
normal-form game is the root of

```
F(x, θ) = x − Π_Δ( x + η · g(x; θ) ),      g_i(x) = ∂u_i/∂x_i
```

where `Π_Δ` is the Euclidean simplex projection — i.e. **`relaxation.sparsemax`,
which we ship with an exact support-set Jacobian**. So `∂_x F` is assemblable in
closed form, the implicit-function-theorem preconditions are checkable (the
existing code already raises when `∂_x F` is singular), and `dθ` of the
equilibrium comes out **without unrolling the solver**.

That is differentiable game solving — differentiable mechanism design,
learned auctions, adversarially-robust training — on infrastructure that exists.
No new differentiation machinery is required.

### 3.4 It hands the evaluator a large family of exact oracles

Game theory is unusually rich in **closed algebraic identities**, which is
precisely what [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md)'s metamorphic and
cross-path (DESIL) oracles consume. §7 lists eleven. This is the cheapest
oracle-surface expansion available to us: no ground-truth data, no reference
framework, no hardware — just identities that must hold to round-off.

---

## 4. Operator gap analysis

### 4.1 The one genuinely new *type*: the coalition lattice

A cooperative game is a tensor over the subset lattice `2^N`. Physically it is
`[..., 2^n]`; semantically the trailing axis is **bit-indexed**, and which bit is
contiguous determines butterfly tiling and shard placement. That is precisely a
Decision #15a `layout` attribute, not a shape.

**Resolved in §4.6.2:** this wants a **`coalition` value on the existing
`layout` attribute**, not a `!tessera.lattice<n>` type — Decision #15a already
declares `layout` first-class, so no new type and no override are needed. Its
consumer is the shared butterfly tiling/sharding pass (**G1b**), which satisfies
Decision #29 by construction. Until G1b lands the convention lives in the spec
and in a Python-side `CoalitionTensor` wrapper, and the ops take `n_players` as
an operand-derived quantity (`2^n == shape[-1]`, verified).

Index convention to fix in G0: **little-endian, bit `i` ↔ player `i`**, so
`S ∈ [0, 2^n)` and membership is `(S >> i) & 1`.

### 4.2 Missing ops — lattice layer

| Op | Shape | Linear? | Notes |
|---|---|---|---|
| `subset_zeta` / `subset_mobius` | `[...,2^n] → [...,2^n]` | yes | butterfly; transpose = superset variant |
| `superset_zeta` / `superset_mobius` | same | yes | the adjoints; needed as first-class for the VJP |
| `coalition_marginal(v, i)` → `∂_i v` | `[...,2^n] → [...,n,2^n]` | yes | bit-flip difference; fuses into consumers |
| `semivalue(v̂, w)` | `[...,2^n] × [n+1] → [...,n]` | yes | `Φ_i = Σ_{T∋i} w(|T|)·v̂(T)` in the Möbius basis |
| `boltzmann_value(v, T)` | `[...,2^n] × scalar → [...,n]` | no | §3.2, online-softmax |
| `coalition_excess(v, x)` | `[...,2^n] × [...,n] → [...,2^n]` | yes | `e(S,x) = v(S) − Σ_{i∈S} x_i`; the `x` term is `subset_zeta` of an additive game |
| `mex` (segmented) | ragged → int | no | Grundy/`mex`; `bitwise_xor` reduce already exists for nim-sums |

**The `semivalue` design point is the operator-design win.** Shapley
(`w(t)=1/t`) and Banzhaf (`w(t)=2^{1−t}`) are the *same contraction* against
different cardinality weights — Faigle's own "random values" framing says a value
*is* an expectation over a distribution on coalitions. Ship one weight-
parameterized op, not two kernels. Every probabilistic value (Shapley, Banzhaf,
`p`-binomial semivalues, the whole Weber set) is then a weight vector.

### 4.3 Missing ops — strategic layer

| Op | Notes |
|---|---|
| `best_response_gradient(u, x)` | `g_i = ∂u_i/∂x_i` = a tensor-times-vector chain contracting the order-`n` payoff tensor against the other players' mixtures. **This is `einsum`** — the op exists; what is new is the `n`-fold leave-one-out contraction schedule, and in the potential-game case an `n`× rewrite the compiler must own (§4.6.3). |
| `nash_residual(u, x, η)` | `x − sparsemax(x + η·g(x))`; both halves exist |
| `equilibrium_solve(...)` | `custom_root` over `nash_residual` → differentiable |
| `saddle_solve(A, method=...)` | zero-sum / bilinear: mirror-prox, extragradient, optimistic MDA. Inner loop is `A@y`, `Aᵀ@x`, softmax — **matmul-bound, fusible into one persistent kernel** |
| `regret_matching(R, kind)` | `σ ∝ [R]⁺`, uniform on the all-nonpositive branch. `kind="external"` → coarse correlated equilibrium; `kind="swap"` → correlated equilibrium (an `|X_i|²` regret matrix per player) |
| `cfr_update(...)` | segment scatter-add of counterfactual values over information sets, weighted by opponent reach probability. `scatter_add` and `index_select` are in `op_catalog`; **`segment_sum` is not and should be added** |
| `coalition_deviation_gain(u, x, k)` | k-resilience: `max_{y_S} min_{i∈S} [u_i(y_S,x_{−S}) − u_i(x)]` over `|S| ≤ k`. Embarrassingly parallel over `C(n,≤k)` coalitions × deviations → `index_launch` |
| `backward_induction(tree, u)` | SPE: reverse level-order scan; each level is a batched segment-argmax over children. Same skeleton as a Bellman backup and as reverse-mode tape replay |
| `hermitian_embed(A)` | `(A+Aᵀ)/2 + i(A−Aᵀ)/2`; trivial, exact VJP, and an **isometry** (§7) |

### 4.4 The LP question — where the expensive part actually is

The Lagrange-games / core / nucleolus material is LP-shaped, so the reflex is to
reach for an external LP library and treat **Decision #23** as the only thing
standing in the way. That reading is wrong, and it is worth being precise about
why, because *the dependency policy is not what is load-bearing here* — the
problem structure is.

The core LP has `n` variables and `2^n` constraints (`x(S) ≥ v(S)`). Nobody
materializes those constraints. The natural algorithm is **cutting-plane / row
generation**:

```
active ← a few constraints
loop:
  x ← solve the master LP           # n variables, tens of rows — tiny
  S* ← argmin_{S ⊆ N} [ x(S) − v(S) ]   # separation: ONE reduction over 2^n
  if excess(S*) ≥ −tol: stop
  active ← active ∪ {S*}
```

The separation oracle — a min over the whole coalition lattice — **is** the
expensive part, and it is a single lattice reduction over the same `[...,2^n]`
tensor `subset_zeta` already produces. That is a Tessera kernel. The master LP is
a dense problem of `n` variables and a few dozen rows, which runs in microseconds
in any implementation whatsoever.

So the honest conclusion is: **the LP is not the bottleneck, and it does not
justify a dependency in either direction.** A ~400-line exact dense simplex with
Bland's rule on the tiny master problem is appropriate, in-house, and dependency-
free — not because #23 forbids the alternative, but because a general LP library
would be a large surface bought to solve a problem that isn't ours.

Two consequences worth writing down:

* The *lexicographic* nucleolus needs **exact basis / tight-set identification**,
  which a first-order method with `O(1/T)` convergence cannot deliver — it tells
  you the gap is small, not which constraints are active. That is the one place
  the exact simplex earns its keep, and the reason not to route everything
  through `saddle_solve`.
* Everything else — zero-sum value, least core, Monge/supermodular core, and
  **correlated equilibria (the limit of no-swap-regret play, no LP at all)** —
  goes through `saddle_solve`: `O(1/T)` on bilinear saddles under an entropic
  mirror map, a GEMV plus a softmax per iteration, differentiable, and it shards.

### 4.5 "Bounded rationality" — the honest reading

The maintainer's "computational complexity limits: players with limits on time and
memory" is a **modelling** notion, not an operator. The concrete, non-hand-wavy
deliverable is a `ConstraintSolver` hook (Decision #4, decoration-time): a
strategy function annotated with a `strategy_budget(memory=…, steps=…)` is
checked at decoration time and rejected with a `TesseraConstraintError` if its
declared state exceeds the budget. That is a small, real feature.

Automatic inference of a strategy's complexity from its body is **not** deferred
on policy grounds — Decision #30 would in fact *prefer* a derived fact to a
declared one. It is deferred because the derivation is unsound in general (it is
the halting problem wearing a hat), and #30's own fallback applies: what cannot
be derived must fail closed. So an undeclared budget means the check does not
run, never that the strategy is assumed cheap.

### 4.6 Where the guardrails should be overridden

The maintainer's standing position (2026-08-15) is that **Decision #23 and the
"no new IR" non-goal are guides, and may be overridden when the system's health
genuinely calls for it.** Taking that seriously means spending it where it buys
something and declining it where it does not. Three findings.

#### 4.6.1 Decision #23 does not need overriding at runtime — but its test boundary should be made explicit

§4.4 dissolves the only runtime case. Nothing else in this plan wants a
third-party runtime dependency, so `python/tessera/`, the C++ runtime and every
shipped artifact stay clean. **No override requested.**

What *is* worth making explicit is a boundary #23 already permits by its letter
("nothing in `python/tessera/`, the C++ runtime, or any shipped artifact may
import them") but which this repo's culture has treated as stricter: a
**test-only differential-oracle group**.

```toml
[project.optional-dependencies]
gametheory-oracles = ["nashpy", "pygambit", "open_spiel"]   # tests only
```

`pytest.importorskip`-gated, never imported from `python/tessera/`, never a
build or CI-blocking dependency. The reason this is worth the paperwork: **CFR
correctness is the one thing in this plan that axioms cannot catch.** Oracle
rows 1–11 (§7) will not detect a subtly wrong opponent-reach weighting in
`cfr_update` — that is an implementation detail with no algebraic invariant, and
the field-standard way to validate it is a differential run against a mature
implementation on Kuhn and Leduc poker. Refusing that leaves a real correctness
hole to protect a rule that does not actually cover the case.

#### 4.6.2 The "no new IR" non-goal *should* be overridden — for a generic butterfly op, not a game dialect

This is where the override earns its keep, and the reason is Decision #31, not
convenience.

The Yates zeta/Möbius recurrence, the Walsh–Hadamard transform, and the FFT's
Stockham stages are **one skeleton**: `n` radix-2 stages, stride `2^i` at stage
`i`, differing only in a per-stage constant 2×2 kernel —

| transform | kernel |
|---|---|
| subset zeta | `[[1,0],[1,1]]` |
| subset Möbius | `[[1,0],[−1,1]]` |
| superset zeta / Möbius | the transposes |
| Walsh–Hadamard | `[[1,1],[1,−1]]` |
| FFT (Stockham) | complex twiddle per stage |

Keeping game theory as a pure Python library means writing the butterfly
tiling/sharding logic a **second** time, next to the one the spectral FFT lane
already has. That is precisely the duplication Decision #31 exists to prevent,
and #29's "a declaration must have a consumer" is satisfied by construction —
the consumer is a shared pass that serves *both* lanes.

So the override is:

* **`tessera.butterfly_transform`** in the **core** dialect with
  `lowering="spectral"` — matching the existing `tessera.fft` pattern exactly
  (`op_catalog.py:210`) — carrying a `kernel` enum over the table above. It is
  linear, so its transpose rule is a kernel-table lookup, not a new derivation.
* **A `coalition` (bit-indexed) `layout` value** on the existing tensor layout
  attribute. This is *not* a new type and needs no override at all: Decision #15a
  already declares `layout` first-class, and the new value's consumer is the same
  shared pass. The earlier `!tessera.lattice<n>` sketch is withdrawn in its
  favour — a layout is what it always was.
* **One shared butterfly tiling + sharding pass**, consuming both, replacing the
  FFT-only logic rather than sitting beside it.

Note what is *not* being asked for: no `tessera_game` dialect. Equilibrium and
CFR ops have no pass that needs to reason about them, so declaring them would be
exactly the #29 failure mode. Revisit at G4, with a specific trigger: when a
fusion or scheduling pass needs to see game structure it cannot recover from the
generic ops.

#### 4.6.3 One real compiler optimization that needs IR — and the correction that sharpened it

The first draft of this plan claimed the `n` best-response gradients
`g_i = ∂u_i/∂x_i` share subexpressions and could be computed by a prefix/suffix
scan pair in `O(Π|X_j|)` instead of `O(n·Π|X_j|)`. **That is false in general**:
each player has their *own* payoff tensor `u_i`, so there is nothing to share.

It is true — and it is an `n`× asymptotic win — exactly when all `n` gradients
descend from **one** tensor:

* **potential games** (`u_i(x) − u_i(x'_i, x_{−i}) = Φ(x) − Φ(x'_i, x_{−i})`, so
  `g_i = ∂Φ/∂x_i` for every `i`),
* common-payoff games, and 2-player zero-sum (`u_2 = −u_1`).

That is not a corner case — it is *the paper's central object*. Faigle's
Chapter 5 is titled "Potentials, Utilities and Equilibria", and his traffic /
Wardrop section is the canonical potential game.

The rewrite: given one potential `Φ`, compute prefix contractions
`P_k = Φ ×_1 x_1 ⋯ ×_k x_k` and suffixes `S_k`, then read every `g_i` off
`(P_{i−1}, S_{i+1})` — the standard all-leave-one-out scan, `O(Π|X_j|)` total.

This needs IR because the pass must (a) know the `n` gradients come from a single
potential and (b) rewrite `n` independent TTV chains into a scan pair. Deriving
"is this a potential game?" from an arbitrary payoff tensor is a closedness check
we should not attempt; so `game_form="potential"` is a **declared semantic key**
(Decision #21a: fails closed, never defaults) with a verifier that checks the
consistency condition on small instances, and **the rewrite simply does not fire
when the key is absent** — correct, `n`× slower, never wrong.

That is the honest shape of the argument: one specific pass, one provable win,
one declared key with a verifier and a conservative fallback.

---

## 5. Plan

Seven phases. G0–G3 need no hardware and run on either box; G1b is C++/MLIR work
and belongs on the primary (Ubuntu/Strix Halo) box per the fleet routing in
[`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) §6a; G5–G6 are
hardware-gated. Sizes are relative, not calendar estimates.

### G0 — Contract and spec (no kernels) · S

`docs/spec/GAME_THEORY_SPEC.md`, plus `docs/audit/domain/` registration.

* Lattice index convention (little-endian, bit `i` ↔ player `i`).
* `CoalitionTensor` shape/layout convention; `2^n == shape[-1]` verified, `n`
  derived, never defaulted.
* **Semantic keys that fail closed** (Decision #21a): `value_weighting`
  (`shapley`/`banzhaf`/`semivalue`), `regret_kind` (`external`/`swap`),
  `equilibrium_kind` (`nash`/`wardrop`/`correlated`/`bayesian`), `game_form`
  (`normal`/`extensive`/`tu`/`potential`). Absence is a diagnostic, not a
  default — and for `potential` specifically, absence disables the §4.6.3
  rewrite rather than assuming the game is not a potential game.
  Performance keys (tile width, stages, sampling budget) may fall back **with a
  diagnostic**.
* **Numeric policy, and it is not optional** — see §6.
* Non-goals (§8) written down before any code.

### G1 — Lattice layer · M

`python/tessera/game/` mirroring `ga/` and `ebm/`: `lattice.py`, `values.py`,
`combinatorial.py`, `__init__.py`.

* `subset_zeta` / `subset_mobius` / `superset_*` registered through
  `custom_primitive(linear=True)` with `def_transpose` (VJP + JVP fall out).
* `coalition_marginal`, `semivalue`, `boltzmann_value`, `coalition_excess`.
* `semivalue`'s generalized (order-`k`) form also yields **interaction
  indices** (§9.1) — the Möbius coefficients are the interaction dividends, so
  Shapley–Taylor pairwise interaction is one more cardinality weighting, not a
  new kernel.
* `mex`; nim-sum via existing `bitwise_xor`.
* Flat-array shims onto `tessera.ops.game_*`, exactly as the GA/EBM lane
  unification did (see `DOMAIN_AUDIT.md`), so the ops are tape-aware and
  catalog-visible.
* `op_catalog.py` **and** `primitive_coverage.py` rows — both, per Decision #24.
* Tests: axiom-based (§7), plus finite-difference VJP checks in the
  `test_relaxation_ops.py` style.

**`relaxation.py` is the exact template** for this phase — a self-contained
operator family, `custom_primitive`-registered, VJPs numerically checked, catalog
metadata included. Follow it literally.

### G1b — Butterfly IR consolidation · M — *the sanctioned IR override (§4.6.2)*

**2026-08-16 checkpoint (`REF-TIER-PHYS-2026-08-16`):** the four coalition
transforms now share a parameterized, content-addressed
`schedule.coalition_butterfly` → `tile.coalition_butterfly_kernel` boundary and
independent AVX-512/fp64-workspace and gfx1151/fp64-LDS consumers. This closes
the coalition-side emitter duplication. G1b itself remains open: the carrier
has not yet replaced FFT-only tiling/sharding, gained the `coalition` layout
value, or passed the required FFT bit-identity gate.

C++/MLIR. **This is a Decision #31 consolidation, and it must not be started
before G1's Python reference exists** — the W0→W1 ordering caveat in
`INTEGRATED_COMPILER_PLAN.md` applies verbatim: do not collapse the duplication
before the surviving path can carry what the deleted one carried.

* `tessera.butterfly_transform` in the core dialect, `lowering="spectral"`,
  mirroring the `tessera.fft` OpSpec pattern (`op_catalog.py:210`). `kernel`
  enum over the §4.6.2 table — an `EnumAttr`, not an unvalidated `StrAttr`
  (Decision #21a corollary).
* `coalition` layout value on the existing tensor `layout` attribute
  (Decision #15a) — a new *value*, not a new type.
* **One** shared butterfly tiling + sharding pass consuming both, **replacing**
  the FFT-only tiling logic rather than sitting beside it. If it does not
  replace, the phase has failed its own justification.
* Lit fixtures including a negative case (Decision #10a): a butterfly whose
  layout forbids the shared-memory staging must produce **no** annotation.
* Differential gate: `tessera.fft` results must be bit-unchanged across the
  consolidation, and `subset_mobius ∘ subset_zeta = id` must hold through the
  new lowering (oracle 1).

### G2 — Equilibrium layer · M

`python/tessera/game/equilibrium.py`.

* `best_response_gradient` (leave-one-out TTV chain over the payoff tensor),
  with the `game_form="potential"` prefix/suffix scan rewrite of §4.6.3 —
  `n`× when declared, the plain `n`-fold chain when not.
* `nash_residual` on top of `sparsemax`.
* `equilibrium_solve` = `custom_root(nash_residual)`; surfaces the existing
  singular-Jacobian diagnostic rather than a silent non-convergence.
* `saddle_solve` (mirror-prox / extragradient / optimistic MDA) for the bilinear
  case, with an explicit duality-gap convergence certificate returned alongside
  the value — **never a bare iterate**. (Verified: extragradient + `sparsemax`
  reaches gap `~1e-16` on random 4×4 games.)
* **`exploitability` / NashConv as a named metric op** (§9.2 ★) — best-response
  value minus current value, with the same certificate discipline.
* **H3/H4 guards**: strict-complementarity check at the solution (distance of
  nearest `sparsemax` coordinate to its kink) surfaced as a diagnostic; strategy-
  derivatives gated on it, value-derivatives always available.
* Wardrop/traffic equilibria and congestion games ride the same projected-gradient
  flow on a potential (Faigle §traffic-flows) with no new op.

### G3 — Regret / learning layer · M

`python/tessera/game/regret.py`, wired to `rl.py`.

* `regret_matching(kind="external"|"swap")`, CFR⁺-style `[·]⁺` accumulation.
* `cfr_update`: segment scatter-add over information sets with reach weights.
  **Adds `segment_sum` to the catalog** — a broadly useful op we currently lack.
* **Preference-CFR**: regularized regret matching with a preference/entropy prior
  — i.e. mirror descent with a non-uniform reference measure. This is a
  temperature parameter on the same softmax, so it composes with `boltzmann_value`
  and with `relaxation.entmax15` rather than being a bolt-on.
* MCCFR external sampling on Philox `RNGKey` with the Decision #18 deterministic
  stream assignment, so self-play runs are bit-reproducible across ranks.
* **Differential validation against an external oracle (§4.6.1)** on Kuhn and
  Leduc poker, `importorskip`-gated. This is the one correctness claim in the
  plan that no algebraic invariant can defend, so it does not ship without it —
  a requirement now *evidenced*, not hypothesized: the plan's own verification
  harness produced two consecutive plausible-but-wrong regret dynamics (H2).
* **Per-H2: every dynamics op also lands with an equilibrium-violation runtime
  assertion** on the `tessera.debug` surface (off by default), so a user's
  miswired training loop is catchable without the external oracle.

### G4 — Sequential and incomplete information · M

* `backward_induction` → **subgame perfect equilibrium**: reverse level-order
  scan, batched segment-argmax per level.
* **Bayesian equilibrium needs no new solver.** A Bayesian game's BNE is the Nash
  equilibrium of its *agent-normal form*, where each `(player, type)` pair is a
  separate player. So it is an index expansion plus a prior-weighted contraction
  over the type axis, then G2. What it *does* require is a real `batching_rule`
  on the game ops — which is exactly the Decision #29 complaint that today's
  `batching_rule` axis is closed over a Python `for` loop. **Game theory is a
  good forcing function for making batching real.**
* `coalition_deviation_gain` → k-resilience check.
* Nucleolus / least core: Python reference, `host_api_contract` on the backend
  axis; the least core itself goes through `saddle_solve`.
* **Acceptance workload: a PSRO double-oracle loop** (§9.2) — simulate an
  empirical meta-game over a policy population, `saddle_solve` it, add a best
  response, repeat until `exploitability` plateaus. It exercises G2's solver,
  G4's batching, `rl.py`'s policies and Decision #18's streams in one program,
  which is exactly what an integration gate should do.
* Freeze the **`Game` protocol** (§9.4 ★) here — the Kuhn/Leduc CFR fixtures
  force the extensive-form shape anyway.

### G5 — Backend lanes · M (hardware-gated per target)

Register arbiter candidates with F4 verification, `SubsetZetaRegion` alongside
`SpectralFFTRegion`:

| Target | First kernel | Why it is first |
|---|---|---|
| x86 AVX-512 (Zen5 box) | butterfly + LSE reduce | executes natively today; no AMX needed |
| Apple GPU (MSL) | `boltzmann_value` | reuses the shipped online-softmax emitter in `emit/apple_msl.py` |
| ROCm gfx1151 | butterfly (Stockham sibling) + WMMA for `saddle_solve`'s GEMM | executing lane; RDNA3.5 WMMA F16/BF16 confirmed in `docs/reference/isa/rdna/` |
| NVIDIA sm_120 | `saddle_solve` GEMM via `mma.sync` | hardware-verified matmul already exists |

### G6 — Distributed lattice · M

* Shard `[...,2^n]` on the top `k` bits (`Block` over the high bits).
* Binary-exchange butterfly for the `k` cross-rank stages; validate against the
  single-rank reference through `testing/mock_collective.MockRankGroup` (threads,
  no NCCL/MPI — Decision #6).
* Monte-Carlo (permutation-sampling) Shapley for `n` beyond the exact regime,
  with antithetic/stratified sampling on Philox streams — under the §9.1 ★ CI
  contract: every sampled estimator returns `(estimate, stderr, n_samples)`,
  never a bare point. KernelSHAP lands here as a recipe (weighted least squares
  over sampled coalitions — no new op).

---

## 6. Numerics — the constraint that decides the design

**Reassessed 2026-08-15 by measurement** (`research/game_theory/
verify_game_theory_plan.py`). The first draft of this section claimed a single
fp32 wall at `n ≈ 24` from the worst-case `2^{|T|}·ε` amplification of the
alternating-sign Möbius row. The measurement sharpened that in both directions:

**The wall is sign-structure-dependent, and real cooperative games sit on the
bad side of it.** A characteristic function is typically **nonnegative and
monotone**, so its zeta grows like `2^n` with *no* cancellation; random-sign
data random-walks at `~2^{n/2}` and merely degrades. Measured fp32-stored-zeta
roundtrip error on O(1) data:

| n | nonneg (real games) | random-sign |
|---|---|---|
| 12 | 3.9e-4 | 1.7e-5 |
| 16 | 9.0e-3 | 2.2e-4 |
| 20 | **1.5e-1** | 2.9e-3 |
| 24 | **5.9e0 — pure noise** | 1.3e-2 |

Meanwhile memory is `2^n · 4 B`: `n=28 → 1 GiB`, `n=30 → 4 GiB`, `n=32 → 16 GiB`.
The 62 GB Strix Halo box tops out near `n ≈ 33`.

**The sharpened conclusion — this is stronger than the original claim:** the
failure is not in the *accumulator*; it is in **storing the transformed tensor
itself in fp32**. `(ζv)(N) ~ 2^n` while the differences a consumer needs are
O(1), so the intermediate write is where the digits die. Therefore:

* **`numeric_policy.accum = fp64` always**, for `subset_zeta`/`subset_mobius`/
  `semivalue`/`coalition_excess` — necessary but **not sufficient**;
* the **materialized zeta must also be fp64** whenever any consumer will
  difference it — for nonnegative games this is unconditional past `n ≈ 16`;
* equivalently and preferably: **fuse zeta → consumer so the `2^n`-magnitude
  intermediate never rounds through fp32**. Fusion is a *correctness* feature
  for this family, not a performance feature — a fact the fusion planner should
  learn from `numeric_policy` rather than from a special case (Decision #32:
  the contract must survive to where codegen picks the instruction);
* `bf16` is **rejected** for the lattice family — not a tuning knob;
* fp64 puts the digits-gone wall at `n ≈ 53` (nonneg worst case), far past the
  `n ≈ 33` memory wall — memory-bound-limited, which is the regime you want.

`boltzmann_value` is the exception that proves the rule: it is a softmax-weighted
*average*, not an alternating sum, so it is well-conditioned in fp32 storage —
provided the max-subtracted online LSE is used and not a naive `exp`/sum.

### 6.1 Correctness hazards to design against

Catalogued during the 2026-08-15 verification pass; each one either bit during
verification or is a known sharp edge of the mathematics. The style follows
`RIEMANNIAN_OT_PLAN.md` §3.4.

**H1 — fp32 lattice intermediates (bit during verification).** Covered above.
The G0 spec must make the fused path the default lowering and require a
diagnostic when a backend materializes the zeta in a narrow dtype.

**H2 — the regret-machinery invariant gap (bit during verification, twice).**
The verification harness's first Blum–Mansour implementation skipped the
stationary-distribution step (played the regret-matching row of the *last
action* instead of the stationary distribution of the swap matrix). It
converged without complaint to a non-CE — violation 0.226 on chicken, caught
only by the CE-violation check itself. The *repair* then failed a second way:
fully normalizing the swap-matrix rows (no Hart–Mas-Colell inertia mass
`pos/max(s, μ)`) oscillates and lands at violation 0.200. **Two consecutive
plausible implementations of a three-line update rule, both wrong, neither
detectable by any algebraic invariant.** This is the strongest concrete
justification for §4.6.1's differential-oracle requirement, and it upgrades
that requirement from "CFR does not ship without it" to: **every dynamics
operator in G3 lands with (a) an equilibrium-violation check as a *runtime
debug assertion* (`tessera.debug` surface, off by default) and (b) a
differential fixture.**

**H3 — sparsemax support changes break the implicit function theorem.**
`nash_residual` composes with `sparsemax`, which is piecewise linear; at a
solution where the support set is not strict (a coordinate exactly at the
kink), `∂_x F` jumps and `custom_root`'s IFT-based VJP differentiates the wrong
branch — silently, since the residual is still zero. The existing
singular-Jacobian diagnostic in `autodiff/implicit.py` does not catch this
(the Jacobian is nonsingular on *each* branch). G2 must add a **strict-
complementarity check** at the solution — distance of the nearest coordinate to
its kink reported alongside the gradient, with a diagnostic below a threshold.
The verification harness's envelope test passed only because the probe was
moved *into* the equilibrium support; that near-miss is the hazard in
miniature.

**H4 — degenerate-game gradients.** In zero-sum games with non-unique
equilibria (positive-dimensional solution faces), the equilibrium *selection*
is solver-dependent even though the *value* is unique. `equilibrium_solve`
must therefore document (and test) that only value-derivatives are
well-defined in general; strategy-derivatives require a uniqueness check
(same strict-complementarity machinery as H3).

**H5 — `T < 0` Boltzmann is legal and meaningful (min-seeking) but a sign slip
is undetectable on symmetric data.** Oracle 7 tests both limits `T→0⁺ → max`
and `T→0⁻ → min` on *asymmetric* fixtures for exactly this reason.

---

## 7. Evidence and oracles

**Status: every identity below was verified numerically on 2026-08-15** —
27 checks, all passing, in
[`research/game_theory/verify_game_theory_plan.py`](../../../research/game_theory/verify_game_theory_plan.py)
(plus `verify_fixups.py`, the failure-diagnosis run). That harness is the seed
of G1's `tests/unit/test_game_lattice.py`; the butterfly implementations in it
are the reference implementations G1 starts from. Verified along the way, beyond
the oracle rows: the semivalue Möbius-basis weights against the direct
permutation-average and subset-average definitions (`w(t)=1/t` ⇒ Shapley,
`w(t)=2^{1−t}` ⇒ Banzhaf), the §1.1 factor-of-two paper erratum (measured
`2^{1−|T|}`, not `2^{−|T|}`, at `|T| ∈ {2,3,5}`), the §4.6.3 prefix-scan
gradients against leave-one-out einsum with the `≤ 2·Π|X_j|` cost bound, the
extragradient+sparsemax saddle solver to duality gap `~1e-16`, the envelope
derivative of the game value against finite differences at an in-support entry,
the corrected Blum–Mansour dynamics to CE-violation 0.0000, and the §4.4
separation oracle finding no violated cut at the Shapley point of a supermodular
game.

Every operator lands with axiom-level metamorphic identities, not transcribed
formulas (§1.1). These are exact-to-round-off and need no ground truth, so they
feed [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md)'s metamorphic and cross-path
oracles directly.

| # | Identity | Guards |
|---|---|---|
| 1 | `subset_mobius ∘ subset_zeta = id` | butterfly correctness, both directions |
| 2 | `⟨ζv, w⟩ = ⟨v, ζᵀw⟩` (superset zeta is the adjoint) | the `transpose_rule` consumer |
| 3 | Efficiency: `Σ_i Φ_i(v) = v(N)` for Shapley weights | `semivalue` weighting |
| 4 | Symmetry, dummy, additivity axioms | `semivalue`, independent of any closed form |
| 5 | Banzhaf on unanimity `u_T` equals `2^{1−|T|}` | catches the §1.1 error class |
| 6 | Faigle Ex. `marginal-sum`: `Σ_S ∂_i v(S) = 0` ⟹ uniform-`π` value is 0 | `coalition_marginal` |
| 7 | `boltzmann_value`: `T→∞` → uniform-mean limit; `E(v,β^{(T)}) → max v` as `T→0⁺`, `→ min v` as `T→0⁻` | temperature limits + LSE stability |
| 8 | `⟨A\|B⟩ = ⟨Â\|B̂⟩` (Hermitian map is an isometry) | `hermitian_embed`; and `Â = Â*` |
| 9 | Strong duality: `saddle_solve` primal value = dual value; duality gap → 0 | the zero-sum solver, with a returned certificate |
| 10 | `‖nash_residual(x*)‖ ≈ 0`, and `∂θ x*` from `custom_root` matches finite differences on `θ` | the differentiable-equilibrium seam end to end |
| 11 | Nim: Grundy of a sum = XOR of Grundy values; `mex` associativity of `+` | combinatorial layer |

Cross-path (DESIL) rows come free: `semivalue` computed in the Möbius basis must
equal the direct permutation-average definition; `saddle_solve` must match a
small-instance exact LP value; sharded lattice results must match single-rank.

External libraries (nashpy, Gambit, OpenSpiel) may be used **offline, by a human,
to generate committed fixture constants**, and — per §4.6.1 — as an
`importorskip`-gated test-only differential oracle for `cfr_update`. They are
never imported by `python/tessera/`, the C++ runtime, or any shipped artifact.

---

## 8. Non-goals

* **No general LP library dependency, in either direction** (§4.4) — the
  separation oracle is the expensive part and it is ours; the master LP is small
  enough that a ~400-line in-house exact simplex is the proportionate answer.
* **No `tessera_game` dialect.** Equilibrium, regret and CFR ops have no pass
  that needs to reason about them; declaring them would be exactly the
  Decision #29 failure mode. The sanctioned IR additions are the *generic*
  `tessera.butterfly_transform` and a `coalition` layout value (§4.6.2) — both
  justified by a shared consumer that removes duplication rather than adding it.
  Revisit at G4 on a specific trigger, not on general enthusiasm.
* **No game-tree interpreter in C++.** Extensive-form structure stays a Python
  data structure feeding batched, level-ordered tensor ops.
* **No exact methods past the memory wall.** Beyond `n ≈ 30` the answer is
  sampling, and the API must say so rather than thrashing.
* **No claim that any of this "executes natively"** until a generated dashboard
  row says it does (Decisions #25/#26).

---

## 9. The developer capability surface — what this class of user actually reaches for

Everything above is operator-and-compiler shaped. This section is the
reassessment's second mandate: what do developers working in this optimization
class *actually want to call*, and does the plan produce it? Surveyed against
the workflows of the four real user populations for this machinery. Each row
names the capability, the underlying plan items it composes from, and what (if
anything) is genuinely new. **The pattern to notice: almost every headline
capability is a thin composition over G1–G4 — the plan's ops are the right
basis — but three genuinely new items surfaced, marked ★.**

### 9.1 ML explainability — the largest population

The single most-used piece of game theory in software today is **SHAP**: Shapley
values over feature coalitions. This is the mainstream draw for the whole
family, and it is worth being blunt: more developers will call
`tessera.game.shapley_values(model, x)` than every equilibrium op combined.

| Capability | Composes from | New? |
|---|---|---|
| Exact Shapley/Banzhaf attribution, `n ≲ 25` features | `subset_zeta` + `semivalue` (G1) | no |
| **Sampled Shapley with confidence intervals**, large `n` | permutation sampling on Philox streams (G6) | ★ the **CI contract**: every sampled estimator returns `(estimate, stderr, n_samples)` — never a bare point. Same discipline as `saddle_solve`'s certificate. |
| **KernelSHAP** | it is a *weighted least squares* over sampled coalitions — `matmul` + solve; no new op | no — a recipe over existing ops |
| **Interaction indices** (Shapley–Taylor, Faigle's own interaction-systems chapter) | the Möbius coefficients *are* the interaction dividends; pairwise index = another cardinality weighting over `subset_mobius` output | no — one more weight vector into `semivalue`'s generalized form |
| Coalition-restricted attribution (features grouped a priori) | `semivalue` over a sub-lattice mask | no |

The interaction-index row closes a loop with the source text: Faigle's
Chapter on interaction systems (the `A⁺ + iA⁻` Hermitian machinery) is
mathematically the *second-order* extension of the same Möbius analysis, and the
pairwise Shapley-interaction matrix is exactly the symmetric part of an
interaction state. The quantum-games chapter we set aside as an outline becomes,
in this reading, the `k=2` case of machinery G1 already ships.

### 9.2 Multi-agent RL and self-play — the population Tessera already courts

Tessera ships PPO/GRPO/CISPO (`rl.py`) for reasoning-model post-training. The
game-theory layer is what turns that single-agent surface into a *population*
surface:

| Capability | Composes from | New? |
|---|---|---|
| **`exploitability(policy)` / NashConv** — *the* standard MARL eval metric | best-response value minus current value: `best_response_gradient` + a max (G2) | ★ as a **named first-class metric op** with a convergence certificate — it is to game solving what a validation loss is to training, and it belongs in `tessera.game` not in user code |
| Fictitious play / replicator / best-response dynamics | pure composition: softmax, averaging, `best_response_gradient` | no — ship as documented recipes + fixtures, not ops |
| **PSRO / empirical game-theoretic analysis** — the double-oracle loop: simulate a meta-game payoff matrix over a policy population, solve it, add a best response | the meta-game solve is `saddle_solve`/`equilibrium_solve` on an *empirical* payoff matrix (G2); population batching is the G4 `batching_rule` work | no new op — but it is the **integration test** for G2×G4×`rl.py`, and the plan adopts it as G4's acceptance workload |
| League/tournament evaluation (round-robin win matrices → ratings) | batched games + `semivalue` (a rating *is* a value on the win-graph game) | no |
| Deterministic distributed self-play | Philox stream assignment (Decision #18), already specified in G3 | no |

### 9.3 Mechanism design, auctions, and markets

| Capability | Composes from | New? |
|---|---|---|
| Differentiable auctions (RegretNet-family): learn a mechanism subject to incentive constraints | `equilibrium_solve`'s implicit gradients (G2) + penalty terms from `coalition_deviation_gain` (G4) | no |
| First/second-price, VCG reference implementations | closed forms over `sort`/`argsort` (in catalog) | no — recipes |
| **Market/traffic equilibria at scale** (Wardrop, Cournot, Fisher markets) | potential-game path: `game_form="potential"` + the §4.6.3 scan + `custom_root` | no |
| Matching markets (deferred acceptance) | sequential, branchy — **out of scope**; document the boundary honestly | n/a |

### 9.4 Governance and power analysis

Weighted-voting analysis (DAO governance, committee design, blockchain
validator-set analysis) is a small but real population whose entire workload is
G1: power indices are `semivalue` calls on threshold games, and the fp64
mandate of §6 is *exactly* the difference between a right and wrong answer for
them, since voting games are 0/1-valued and monotone — the measured worst case.

One genuinely new item serves all four populations:

★ **The `Game` protocol.** Developers do not want to hand-build payoff tensors;
they want to declare a game and have the library derive the tensors, the
lattice, or the tree. A small Python protocol —
`num_players / action_space(i) / payoff(joint_action) / [chance nodes /
information sets]` — with adapters that *materialize* to the G1/G2/G4
representations, gives every capability above a common front door. This is
vocabulary-borrowing from OpenSpiel's API shape under Decision #23 (reimplement,
don't wrap), and it is deliberately a **Python-layer** contract: the compiler
never sees it, only its materializations. Lands across G2–G4; the CFR
differential fixtures (§4.6.1) already need Kuhn/Leduc definitions, which
become its first two instances.

### 9.5 What was deliberately left out of the capability surface

* **Stochastic-game / MDP value iteration as a named op** — it is a `scan` over
  Bellman backups; belongs to a general `scan` op discussion (`scan` is absent
  from the catalog today, a sharper version of open question 3).
* **Nash bargaining / cake-cutting** — thin demand, closed forms; examples, not
  ops.
* **Full extensive-form solver suites** (sequence-form LPs) — the sequence-form
  representation is a sparse-matrix encoding whose payoff would duplicate the
  CFR path; revisit only if a user shows up with a game CFR cannot handle.

---

## 10. What to build first

If only one thing lands: **G1's `subset_zeta`/`subset_mobius` +
`semivalue`**, with oracles 1–6. It is self-contained, needs no hardware, is
exactly testable, gives `transpose_rule` a real consumer, and produces the
butterfly region class that G5's arbiter lane and G6's sharding both build on.
`boltzmann_value` is the natural second, because it is the one that reuses the
online-softmax emitter and therefore reaches an executing GPU lane soonest.

**G1b is the highest-value item that outlives game theory**, and it is the one
piece here that would still be worth building if the game-theory surface were
cancelled tomorrow: it consolidates butterfly tiling for the spectral FFT lane
too. But it is deliberately *second*, not first — the Decision #31 ordering
caveat says do not collapse a duplication before the surviving path can carry
what the deleted one carried, and until G1 exists there is only one butterfly
consumer and therefore nothing to consolidate.

---

## 11. Open questions for the maintainer

1. **Scope of `n`.** Is the target regime cooperative games with `n ≲ 25`
   (feature attribution, coalition analysis, mechanism design), or large-`n`
   sampled Shapley (ML explainability)? The first justifies G6's exact
   distributed lattice; the second makes sampling + the §9.1 CI contract the
   priority and G6 optional. §9 suggests the explainability population is the
   larger one.
2. **Is differentiable equilibrium the actual goal?** If the intent is learned
   mechanisms / differentiable economics, G2 outranks G1 and should go first.
   If the intent is cooperative-value analysis, G1 first as written.
3. **Do `segment_sum` and `scan` want to land independently?**
   **RESOLVED by measurement (2026-08-15): the premise was stale.** The
   catalog already carries `segment_reduce(x, seg_ids, op=...)` (CFR's
   scatter-add is `op="sum"`), the prefix-scan family
   (`cumsum`/`cummax`/`cumprod`), and `control.scan` (S5) lowering to
   `tessera.control_scan` for general carried recurrences (Bellman backups).
   Adding `segment_sum`/`scan` beside them would be the Decision #31
   duplication this plan elsewhere consolidates. G3/G4 consume the existing
   ops; the general-ops tranche item reduces to *verifying they carry what
   CFR needs* (ragged encoding, reach-weight multiply), not to new ops.
4. **Where does this sit against the S-series and the Riemannian-OT track?** All
   three consume the same implicit-differentiation seam; if that seam needs
   hardening — H3's strict-complementarity check is now a concrete work item on
   it — doing it once for all three is cheaper than three times.
5. **Should the `Game` protocol (§9.4 ★) be scoped now or after G2?** It is the
   developer front door, but designing it before the tensor representations
   stabilize risks a second API revision. Recommendation: sketch it in G0's
   spec as *non-normative*, freeze it at G4 when the CFR fixtures force the
   extensive-form shape anyway.
