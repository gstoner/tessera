---
last_updated: 2026-08-20
audit_role: plan
plan_state: open
---

# Autodiff next generation — one derivative functor, algebra as a parameter

> **Routing:** start at [`README.md`](README.md). This document owns the design,
> mathematical verification obligations, and acceptance criteria for the
> next-generation autodiff program — the design content behind
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) W6.3 ("research
> estimate required"), plus the law-oracle evidence lane that precedes it.
> Global ordering lives only in `INTEGRATED_COMPILER_PLAN.md`;
> `MASTER_AUDIT.md` + generated dashboards stay status truth (Decision #26).
> This is a design and build-sequence document, not a status claim.
> **Binding status (2026-08-18):** W6.3 now names this document as its design
> owner and carries §7's estimates; the E2E-REAL-6 Law-3 gate is recorded as
> *pending AD-LAW-1*. §5 tracks which bindings are accepted and which remain
> proposed.
> **Execution status (updated 2026-08-19, after the ad-law series — PRs
> #584–#588):** AD-LAW-1 slices a–h are merged: the §3 math harness (incl.
> the #10a negative fixture), Laws 1+3+5 over the live registries with
> canonical-forward anchoring, the byte-gated `autodiff_law_audit`
> dashboard, the tape's positional-call contract (1d), doubled spec
> coverage (1c), and the swallow triage (1b/1f/1g/1h) — the rule-vs-rule
> swallowed-kwarg class is **closed**; the forward-vs-rule class keeps
> ~43 pinned open findings (`_OPEN_FORWARD_KEY_SWALLOWS` in
> `test_autodiff_laws.py`), each awaiting a body read. **AD-LAW-2 and the
> AD-WEIL-1 substrate landed ahead of the written sequence** (same series):
> `DifferentialAlgebra`, `Dual`, `TruncatedJet(k)`, the holonomic ODE table
> (`SCALAR_RECURRENCES`), the finite multiplication-table substrate with
> `clifford_table(p,q,r)` cross-checked against `ga` as oracle (the
> W6.3/W6.4 substrate hypothesis, proven), the measured §3.8 conditioning
> envelope, and Laws 2+4+6 — all six declared laws execute host-free.
> **AD-LAW-1 sweep debt closed (2026-08-20, AD-LAW-1i–1n):** the geometric
> registry is fully swept (16/16), tensor spec growth took `no_spec` from
> 173 to eight rule-capability gaps pinned with named reasons
> (`_OPEN_UNSWEEPABLE_RULES`), the forward-key swallow triage is complete
> (one real defect — conv layout — fixed; 41 named-benign), and the
> `vjp_only` class is extinct (17 JVPs registered and law-checked). The
> sweep surfaced and fixed fourteen real rule defects along the way (svd/qr
> tangents, weight_norm and spectral_filter differentiating different
> functions, sddmm operand order, gru bias drop, mor delegation pair,
> sequence-operand tape support, adafactor FD step, conv layout, and
> friends — each pinned with a negative fixture). The **E2E-REAL-6 Law-3
> gate is ACTIVE** as of 2026-08-20. **AD-WEIL-1 acceptance closed
> (2026-08-20):** `DerivativeContract` (§2.1) is registered — derived from
> the three single authorities (ODE table, `MULTILINEAR_PRIMITIVES`,
> `NONSMOOTH_SELECTION`) with structural rejection of conflicting
> smoothness claims and the §3.6 `pd_witness` certificate per family —
> and the `coefficient_scaling` / jet `numeric_policy` semantic keys
> (§2.3) are declared, fail-closed keys on `TruncatedJet`, consumed and
> negative-fixtured by the law tests. One recorded follow-up: the
> Decision #24 `primitive_coverage.py` axes for the new fields are not
> yet wired (a registry-wide integration deserving its own slice). §5
> tracks bindings.

**Date:** 2026-08-18 · **Sources read:** [`AUTODIFF_SPEC.md`](../../spec/AUTODIFF_SPEC.md)
(complete), [`vjp.py`](../../../python/tessera/autodiff/vjp.py) /
[`jvp.py`](../../../python/tessera/autodiff/jvp.py) (registry structure + rule
signatures; 5,652 + 4,012 lines at time of writing — live counts are
dashboard-owned), [`tape.py`](../../../python/tessera/autodiff/tape.py),
[`linear.py`](../../../python/tessera/autodiff/linear.py),
[`nonsmooth.py`](../../../python/tessera/autodiff/nonsmooth.py),
[`implicit.py`](../../../python/tessera/autodiff/implicit.py),
[`geometric/registry.py`](../../../python/tessera/autodiff/geometric/registry.py) +
[`geometric/tape.py`](../../../python/tessera/autodiff/geometric/tape.py),
`INTEGRATED_COMPILER_PLAN.md` (queue + AD-CORE items + W6),
[`CUTE_IR_ASSESSMENT.md`](CUTE_IR_ASSESSMENT.md) (complete),
[`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md) (complete — see §5a for the
substrate cross-check). Statements about
external literature (Weil-algebra AD, Taylor-mode, conservative fields,
stochastic derivative estimators) are from the published record, not from
running those systems.

**Bottom line:** every AD mode is the same functor evaluated in a different
codomain algebra; today the codomain ℝⁿ is hardwired into two ~5k-line
hand-paired registries, and five in-tree structures have each independently
paid for that hardwiring. The fix is one derivative datum per primitive plus an
explicit `DifferentialAlgebra` parameter, with correctness enforced as
executable algebraic laws rather than pointwise tolerance checks. This plan
supports current best practice exactly (the `Dual()` instance + transpose *is*
JAX/PyTorch semantics) and goes past it on axes neither framework can retrofit:
order-k derivatives at cost k+1 instead of 2ᵏ with the coefficient axis fused
at tile level, certified enclosures, declared nonsmooth semantics with a
citable convergence theorem, deterministic contract-tested stochastic
estimators, operator tangents, and multivectors on the same tape as tensors.

---

## 1. Diagnosis — five in-tree structures, one missing abstraction

Not theoretical. Each row is code that exists because the codomain is not a
parameter:

| Evidence | What it is | What it proves |
|---|---|---|
| `vjp.py` + `jvp.py` | The same derivative information hand-written twice, once per mode (~9.6k lines total) | The mode is welded into the rule; one mathematical datum per primitive is maintained as two functions |
| [`linear.py`](../../../python/tessera/autodiff/linear.py) | Derives JVPs from the forward op for multilinear primitives; treats `transpose_rule` as the VJP | Rules are **derivable, not authored** — already demonstrated in-tree for one family, with a differential test (`test_linear_transposition.py`) proving the hand-written duplicates mechanical |
| [`geometric/registry.py`](../../../python/tessera/autodiff/geometric/registry.py) | `_VJPS_GEO`/`_JVPS_GEO` + a parallel tape, forked because values became `Multivector` | Changing the value algebra forced duplicating the entire AD stack — the cost of the hardwired codomain, measured in this repo. A standing Decision #31 tension (two tapes, one boundary) |
| [`nonsmooth.py`](../../../python/tessera/autodiff/nonsmooth.py) | Declared, drift-tested Clarke-subgradient selection (`SUBGRAD_ZERO`/`SUBGRAD_SPLIT`) | "Which generalized derivative" is already treated as a semantic key (#21a) with a consumer (#29) — the entry ramp to §3.6 |
| [`implicit.py`](../../../python/tessera/autodiff/implicit.py) | IFT via matrix-free `A`/`Aᵀ` matvecs + CG/GMRES | Derivatives-as-operators already exist as a local pattern; §3.5 promotes them to a tangent type |

And [`AUTODIFF_SPEC.md`](../../spec/AUTODIFF_SPEC.md) marks the ceiling
explicitly: higher-order in Graph IR is deferred as "run AutodiffPass twice" —
the 2ᵏ-cost path §3.1 exists to retire.

---

## 2. The design

### 2.1 One derivative datum per primitive

A primitive registers the *minimal* mathematical data from which every mode
derives — usually one field, sometimes zero:

```python
@dataclass(frozen=True)
class DerivativeContract:
    # (1) Structure declaration — often the WHOLE rule:
    linear_args: tuple[int, ...] | None      # multilinear ⇒ JVP = forward-with-swap
                                             #   (linear.py's MULTILINEAR_PRIMITIVES, today)
    # (2) Pointwise/scalar primitives — the one datum all orders derive from:
    ode: HolonomicODE | None                 # e.g. tanh: w′ = (1 − w²)·u′  (§3.2)
    # (3) Structured primitives — the linearization as a graph of OTHER
    #     primitives, in vocabulary the tracer already speaks:
    linearize: Callable | None               # primals -> (out, jvp_graph)
    transpose: Callable | None               # adjoint of jvp_graph ⇒ VJP  (§3.5)
    # (4) Nonsmooth semantic key (#21a) — shipped today in nonsmooth.py:
    kink_policy: str | None                  # SUBGRAD_ZERO / SUBGRAD_SPLIT / ...
    # (5) Definability certificate for §3.6 (semialgebraic / o-minimal):
    pd_witness: str | None                   # "smooth" | "definable:<structure>";
                                             #   fail-closed for exotic customs (§3.6)
```

The registry rejects conflicting registrations structurally (the same
duplicate-registration guard `jvp.py` already applies), and
`primitive_coverage.py` gains axes for the new fields per Decision #24 —
each axis with a named consumer per Decision #29.

### 2.2 The algebra parameter

The mode is an object passed to one interpreter, not a registry:

```python
class DifferentialAlgebra(Protocol):
    """The codomain W = ℝ ⊕ m. One instance per AD mode."""
    def lift(self, primal, seed): ...        # x ↦ x + Σ seedᵢ·basisᵢ
    def add(self, a, b): ...
    def mul(self, a, b): ...                 # the load-bearing method
    def scalar_fn(self, ode, a): ...         # φ(a) via coefficient recurrence (§3.2)
    def extract(self, a, index): ...         # read off a derivative component
```

| Instance | `mul` | Yields | Gated on |
|---|---|---|---|
| `Dual()` | `a₀b₁ + a₁b₀` | today's JVP, semantics-identical | AD-WEIL-1 |
| `TruncatedJet(k)` — Weil ℝ[ε]/(ε^{k+1}) | truncated Cauchy product | order-k Taylor, dim k+1 | AD-WEIL-1 |
| `MixedPartial(ideal)` | quotient-ring product | exactly the requested multi-indices (hyperduals etc.) | consumer (#29) |
| `ChebJet(k, interval)` | Chebyshev product `2TₘTₙ = T_{m+n} + T_{|m−n|}` | high order that stays conditioned (§3.8) | consumer (#29 — candidate: the PDE Chebyshev/DST spectral lane, §5a) |
| `TaylorModel(k)` | poly product + outward-rounded interval remainder | certified enclosures | AD-CERT-1, consumer-gated |
| `CliffordTangent(sig)` | geometric product | `geometric/` stops being a fork | AD-WEIL-1 substrate proof |
| `OperatorTangent(X)` | operator composition | Fréchet derivatives; `implicit.py` matvecs made first-class | AD-OPERATOR-1 |

Reverse mode is **not** an instance — it is the transpose functor applied to
the linearization (`vjp = transpose ∘ linearize`), which is how `linear.py`
already treats `transpose_rule` and how AD-CORE-LINEAR-1 already works at
Graph IR. During migration the existing `_VJPS`/`_JVPS` stay in place as
declared oracles (Decision #31), retired family-by-family only behind the
Law-4 differential proof (§4), mirroring the E2E-REAL-6 `_OpExtractor`
protocol.

### 2.3 Semantic keys (#21a) introduced by this design

These select **meaning** and therefore fail closed on absence — never
defaulted silently:

| Key | Legal set | Why semantic |
|---|---|---|
| `coefficient_scaling` | `taylor` (÷k!) \| `derivative` | The same buffer means different numbers under each convention (JAX's `factorial_scaled` made this a boolean default; here it is a declared key on the jet type) |
| `kink_policy` | `nonsmooth.py`'s named policies | Already shipped; enters the *forward* of higher-order kernels (§3.4) |
| `pd_witness` | `smooth` \| `definable:<structure>` | Per-primitive witness for the §3.6 convergence guarantee — a bare boolean cannot carry the hypothesis (see §3.6's correction) |
| `control_at_order` | `0` (only legal value in v1) | Data-dependent control flow (branch predicates, `while` trip counts, `max` selections) evaluates on the **primal coefficient only**; coefficients follow the primal's trace. Matches W4-PRODUCT's predicate-replay identity, stated for jets |
| cotangent/coefficient `numeric_policy` | per Decision #15a | Higher coefficients shrink like 1/k!; accumulator and storage dtype per coefficient is a declared contract, not an accident. (Today the tape seeds backward at float64 regardless of model dtype per the spec's mechanism section — an implicitly chosen accumulation dtype this key makes explicit.) **Carrying this key below Graph IR depends on the S5 generalized `numeric_policy` carrier (`CORE_SUBSTRATE_VIEW.md` S5 — fragment-only today, no owning row); AD-JET-IR-1 is its fourth mandating consumer, §5a** |

Interleaved-vs-planar coefficient storage (§6) is deliberately **not** in this
table: both layouts compute the same jet, so it is a performance key —
fallback-with-diagnostic, arbiter-measured.

---

## 3. Mathematical basis — each claim is a theorem with a test

### 3.1 Weil evaluation: exact by nilpotency, k+1 not 2ᵏ

In `W = ℝ[ε]/(ε^{k+1})`, lift `x̂ = x + x₁ε + … + x_kε^k`. For smooth `f` the
Taylor remainder is annihilated **identically** by `ε^{k+1} = 0`:

```
f(x̂) = Σ_{j=0}^{k} f^{(j)}(x)/j! · (x̂ − x)^j        (exact in W)
```

and `f ↦ f(·on W-points)` is an ℝ-algebra homomorphism — the chain rule at
every order simultaneously. k-times-nested forward mode is evaluation in
`⊗ᵢ ℝ[εᵢ]/(εᵢ²)` (dim **2ᵏ**); jet mode is evaluation in `ℝ[ε]/(ε^{k+1})`
(dim **k+1**). The two are related by the **diagonal embedding**

```
Δ : ℝ[ε]/(ε^{k+1}) ↪ ⊗ᵢ ℝ[εᵢ]/(εᵢ²),    ε ↦ ε₁ + ε₂ + … + ε_k
```

which is well-defined because `(ε₁+…+ε_k)^{k+1} = 0` (any k+1 factors over k
square-zero generators repeat one, by pigeonhole) and injective because
`(Σεᵢ)^j = j!·e_j(ε₁,…,ε_k) ≠ 0` for `j ≤ k`. Evaluating the nested tower on
the diagonal seed and reading coefficients back through `Δ` (the top mixed
term `ε₁⋯ε_k` of the degree-j image carries `j!` times the jet coefficient)
recovers the jet exactly — that is what proves nesting computes the right
value while carrying `2ᵏ` dimensions for a `k+1`-dimensional answer. That
embedding, with its factorial bookkeeping, is Law 4's test statement and the
retirement argument for "run AutodiffPass twice."

> *Corrected in review (2026-08-18):* an earlier draft stated this as a
> surjection `εᵢ ↦ ε` out of the nested algebra. That map does not exist —
> `εᵢ² = 0` would have to map to `ε² = 0`, false in `ℝ[ε]/(ε^{k+1})` for
> `k ≥ 2`. The relationship runs the other way (small algebra embeds in
> big), and the AD-LAW-1 math harness pins the wrong map as a negative
> fixture (#10a) so it cannot be reintroduced.

### 3.2 Pointwise family: the holonomic ODE is the whole rule

Every op in the spec's pointwise/activation tables satisfies a low-order ODE
in its input; the order-k coefficient recurrence follows by Cauchy products
(Griewank & Walther Ch. 13), O(k²) per op. Writing `u(t) = Σ uⱼtʲ`, `w = φ(u)`:

- **exp** (`w′ = w·u′`): `w_k = (1/k) Σ_{j=1}^{k} j·u_j·w_{k−j}`
- **tanh** (`w′ = (1−w²)·u′`, `s ≔ w²` one auxiliary Cauchy square):
  `w_k = (1/k) Σ_{j=1}^{k} j·u_j·(δ_{0,k−j} − s_{k−j})`
- **sin/cos** (joint pair): `s_k = (1/k)Σ j·u_j·c_{k−j}`, `c_k = −(1/k)Σ j·u_j·s_{k−j}`

The `k=1` row of the tanh recurrence is exactly `jvp_tanh`
([`jvp.py:231`](../../../python/tessera/autodiff/jvp.py#L231)); the
hand-written first-order pointwise registry is the shadow of this ODE table.
Registering the ODE replaces the pair of hand rules and adds all higher
orders.

### 3.3 Multilinear family: the jet rule is a Cauchy convolution — zero new rules

For bilinear `f(A,B) = A·B` with jets `Â = Σ Aᵢεⁱ`, `B̂ = Σ Bⱼεʲ`:

```
(Â·B̂)_k = Σ_{i+j=k} Aᵢ·Bⱼ
```

All coefficients through order k cost `(k+1)(k+2)/2` matmuls — polynomial vs
2ᵏ matmul-shaped terms under nesting. `linear.py`'s `MULTILINEAR_PRIMITIVES`
table already declares which arguments enter linearly; multilinearity *is*
the jet rule. Structurally, a jet is the primal tensor with one extra
length-(k+1) axis, and jet-matmul is batched matmul with a triangular
convolution over that axis — see §6 for why that axis is a layout-algebra
object and a tile-fusable dimension. This is the structural advantage over
`jax.experimental.jet`, whose coefficients are a Python list of arrays above
the compiler.

### 3.4 `flash_attn`: the jet of an online softmax

`softmax(z) = exp(z − lse(z))` decomposes into jet-closed pieces (`exp`, `Σ`,
`log` — all §3.2), so no new mathematics is needed; what is needed is the
*fused* form. The online recurrence's running stats `(m, ℓ, o)` generalize
as: `m` stays order-0 (it is the nonsmooth `max` — its behavior at ties is
governed by the declared `SUBGRAD_SPLIT` policy, which is why the kink policy
had to be a semantic key: it now enters the forward of the higher-order
kernel); `ℓ` and `o` become jets updated by the same recurrence with scalar
ops replaced by W-ops; the rescale `exp(m_old − m_new)` is an exp-jet of a
shift. Consequence: backward, HVP, and order-k directional derivatives share
one kernel skeleton at different `W` — same tiling, same memory-traffic
pattern, coefficient axis fused. The natural carrier is a jet-parameterized
sibling of `schedule.attention_backward` under the same content-addressed
package rules (E2E-REAL-5B), including the LSE checkpoint identity.

### 3.5 Reverse mode as adjunction; operator tangents

For the linearization `J = ∂f(x)`, reverse mode is `Jᵀ`, characterized
completely by `⟨Jv, u⟩ = ⟨v, Jᵀu⟩`. Three engineering consequences:

1. The adjoint law is a **complete test of the transpose relationship** —
   that the VJP is the adjoint of the supplied JVP (Law 3) — runnable over
   the entire existing `_VJPS`×`_JVPS` overlap today, including
   `_VJPS_GEO`×`_JVPS_GEO` under the multivector inner product. It is **not**
   by itself a derivative-correctness test: a matched-wrong pair (e.g. an
   all-zero JVP with an all-zero VJP) satisfies the identity on every probe.
   Derivative correctness is carried by the independent oracles — Law 2
   against the registered ODE/analytic datum, Law 4's jet-vs-nested proof,
   Law 1 composition, and the existing per-op numerical-Jacobian tests,
   which remain in force. Law 3's value is that it localizes a
   *disagreement* between the mode pair with no reference implementation
   and completes the pointwise checks on the transpose axis.
2. `Jᵀ` need not be materialized. `OperatorTangent` (composition as product,
   `.T` as involution) turns `implicit.py`'s IFT, iHVP, Newton–Krylov, and
   Gauss–Newton into compositions consumed by matrix-free solves — the
   generalization of the **landed** AD-SOLVER-IFT-1/W3.5 parent/child
   package, which already executes residual + transposed matrix-free solve +
   adjoint on AVX-512/gfx1151 without materializing a Jacobian.
3. The materialized twin of `Jᵀ` already exists at IR level: the paired
   backward ABI `@f__bwd`. Decision #31: one production lowering
   (materialized), one declared relationship to the unmaterialized operator
   form — recorded, not drifting.

Well-posedness (solvability of `Aᵀr = u`) is a property checked at the solve
— `implicit.py`'s `LinearSolveInfo` + fail-closed non-convergence is already
the right shape — never assumed from smoothness of the residual.

### 3.6 Nonsmooth: from declared policy to theorem-backed guarantee

`nonsmooth.py` pins *which* Clarke subgradient each kink returns. The
conservative-field results (Bolte & Pauwels) supply the missing top plate:
for **path-differentiable** programs, tape AD with any fixed selection
yields a conservative field; subgradient descent on AD output converges to
the correct stationary set.

*Corrected in review (2026-08-18):* an earlier draft claimed every catalog
primitive is semialgebraic-definable, which is false — `exp` is not
semialgebraic, and unrestricted `sin`/`cos` are not definable in **any**
o-minimal structure (infinitely many zeros). A bare `definable: bool` cannot
carry the hypothesis. The honest certificate is a per-primitive
**path-differentiability witness** (`pd_witness`, §2.1/§2.3 — a semantic key,
fail-closed for exotic `custom_primitive` registrations):

- **`smooth`** — C¹ primitives (`exp`, `sin`, `cos`, `tanh`, `erf`, …) are
  path-differentiable trivially; the conservative field is the singleton
  gradient. No definability claim is made or needed.
- **`definable:<structure>`** — the nonsmooth primitives (`relu`, `max`/`min`
  ties, `clip`, `abs`, sort/top-k selections) are piecewise-polynomial, hence
  **semialgebraic** — the definability claim is made exactly where it is true,
  with the structure named. Here the declared kink policy selects the element
  of the conservative field.
- **Compositions inherit path-differentiability by the conservative-field
  chain rule** (the central Bolte–Pauwels result) — which is precisely what
  tape composition implements. The program-level guarantee therefore needs no
  single o-minimal structure containing every primitive; it needs each
  primitive to carry a valid witness and the chain rule to compose them.

This gives the spec a citable training-convergence contract that neither JAX
nor PyTorch states — PyTorch's historical `clamp`-vs-`clip` kink
inconsistency is the exact bug class the policy registry already fixed
in-tree.

### 3.7 Stochastic derivative estimators, deterministic by construction

For high-dimensional operators (Laplacian, Hessian-trace) even k+1 jets are
too many. Randomizing the jet seed gives unbiased estimators of the operator
(STDE, NeurIPS 2024; compositional unbiasedness contracts per ADEV, POPL
2023): `E[extract(f(lift(x, v_random)))] = (𝒜f)(x)`. The correctness claim
changes type — from *equals* to *unbiased estimator of* — and Tessera can
test it: Decision #18's Philox streams make estimator runs deterministic and
replayable (empirical mean vs exact jet on small probes, seeded). Estimator
modes carry the `random` effect in the fail-closed W2.2 `EffectLattice`, so a
stochastic gradient cannot masquerade as deterministic under
`@jit(deterministic=True)`, and they register the stochastic identity that
W5.2e's dependence-edge inference already models.

### 3.8 Conditioning: the honest limit of the monomial basis

The §3.2 recurrences are exact in exact arithmetic; in floats the monomial
basis loses accuracy as order grows (exponentially ill-conditioned basis).
That is a conditioning fact, not a truncation fact, and it is what `ChebJet`
(same `scalar_fn` interface, Chebyshev product underneath, validity on an
interval) and `TaylorModel` (rigorous remainder) exist for. **Neither is
built until a consumer exists** (#29); the jet `numeric_policy` key (§2.3)
carries the interim contract, and the observed conditioning envelope (max
trustworthy k per dtype) is measured by AD-WEIL-1's law harness and recorded
in its generated dashboard rather than guessed.

---

## 4. Correctness as executable law

Frameworks test derivatives pointwise (finite-difference `gradcheck` —
noisy, tolerance-tuned, incomplete). This program tests the algebraic laws
the functor must satisfy, as evaluator oracles (the existing
vertical/horizontal/metamorphic/DESIL harness), rendered to a generated
dashboard and drift-gated:

| # | Law | Statement | Character |
|---|---|---|---|
| 1 | Functoriality | `D(g∘f) = Dg ∘ Df` on randomly composed primitive pairs | metamorphic; exact on polynomial pieces |
| 2 | Homomorphism | evaluation in `W` commutes with `+`/`×`/composition | **exact (tolerance 0)** for polynomial primitives by nilpotency; tight-tolerance for transcendental |
| 3 | Adjoint | `⟨Jv,u⟩ = ⟨v,Jᵀu⟩` per primitive, dimension-scaled probe count | complete for the **transpose relationship** (not derivative correctness — a matched-wrong pair passes; Laws 1/2/4 + existing FD tests carry that, §3.5); runs over today's registries **before any refactor**, incl. geometric |
| 4 | Quotient consistency | jet order-k ≡ k-nested `Dual()` on the diagonal seed (§3.1 embedding, factorial bookkeeping) | the differential proof that gates every hand-rule retirement (Decision #31) |
| 5 | Kink policy | probe exactly at ties/bounds; assert declared policy + mass conservation for `SUBGRAD_SPLIT` | extends `nonsmooth.py`'s existing drift test |
| 6 | Enclosure / unbiasedness | `TaylorModel` interval contains the exact jet; estimator mean → exact operator under fixed seeds | certified / stochastic modes only |

Harness discipline is copied from
[`test_layout_algebra_contracts.py`](../../../tests/unit/test_layout_algebra_contracts.py)
(per `CUTE_IR_ASSESSMENT.md`): fixtures pinned exactly, both value and
structure asserted, mutation-tested against collapsing encodings, and any
defect found in existing rules carried as a negative fixture (#10a). Laws 1–4
are host-free — pure numpy over small shapes, the same species as the layout
algebra's ~40-line hardware-free validation of NVIDIA's algebra.

**Dual use of the jet mode as test infrastructure:** once AD-WEIL-1 lands,
exact jets become higher-order ground truth for gates that currently use
weaker oracles — e.g. W6.1's HVP product currently checks against an
independent quadratic oracle; a jet reference generalizes that to arbitrary
programs, and the eager `tessera.autodiff.hvp` helper (still
finite-difference per the plan) gets an exact replacement essentially for
free.

---

## 5. Relationship to INTEGRATED_COMPILER_PLAN

**This document is the design + acceptance detail for W6.3 and does not
create a queue.** Bindings 1, 5, and the `CUTE_IR_ASSESSMENT.md` §3 row
(§6) were **accepted and applied 2026-08-18**; 2–4 remain proposed.

1. **Accepted (2026-08-18) — W6.3's "research estimate required" is
   replaced** by §7's phased estimates, and the W6 exit criterion now carries
   this plan's falsifiable stop condition (§10). The W6.4-note hypothesis ("a generic algebra representation…
   treat reuse as a design hypothesis to prove") is resolved by a concrete
   proof obligation: implement `TruncatedJet(k)` and `Cl(3,0)` over the same
   finite-multiplication-table substrate and cross-check the Clifford
   instance against `ga/signature.py` as oracle (AD-WEIL-1 acceptance). The
   `DifferentialAlgebra` protocol is the generic representation W6.3 asks
   for; the table substrate is its implementation detail, not its interface.
2. **W6.1 relationship:** forward-over-reverse (`@f__bwd__jvp`) is correct
   and stays the production k=2 path until Law 4 proves the jet route; then
   the nested path becomes the declared oracle (Decision #31), not deleted.
3. **W6.2 composition:** coloring applies to jet seeds exactly as to first
   order (Griewank-style interpolation of higher derivative tensors from
   colored univariate jets) — a composition of two funded rows, no new scope.
4. **De-duplication ledger (§3 of the integrated plan) gains one row:**
   `geometric/` parallel tape + `_VJPS_GEO`/`_JVPS_GEO` → absorbed as
   `CliffordTangent` instance (W3-class collapse, gated on the substrate
   proof + Law 3/4 green on the geometric registry).
5. **Accepted (2026-08-18) — E2E-REAL-6 gate strengthening**, recorded in
   the integrated plan as a **pending** gate: the Law-3 adjoint check joins
   the family-migration checklist *once AD-LAW-1 lands*, completing the
   existing pointwise "numeric identity" binding on the transpose axis. It
   composes with, and does not replace, the per-family derivative oracles
   (§3.5's completeness caveat: Law 3 alone cannot certify the derivative).
   No current migration is gated on it, because the oracle does not exist
   yet.

**Funding table (OT-style re-scope).** Most of the program is already paid
for by landed or landing work:

| This plan needs | Already provided by |
|---|---|
| Multilinear structure declarations + transpose authority | AD-CORE-LINEAR-1 (`LinearTransposeInterface`, landed) + `linear.py` |
| Forward-mode IR carrier to generalize | W6.1 `TangentInterface`, paired functions, `--tessera-autodiff-hvp-pipeline` (landing) |
| Matrix-free operator execution substrate | W3.5 / AD-SOLVER-IFT-1 GMRES/CG parent+children on AVX-512/gfx1151 (landed) |
| Activity/effects/liveness analyses, fail-closed | W2.1 + W2.2 (closed) |
| Control-flow regions + residual ABI for jets through loops | W4.3 (landing) + W4-PRODUCT-1 (queue Order 2) |
| Coefficient-axis representation + index math + residency | LAYOUT-ALG-1 L1/L3/L4/L5 (§6) |
| Kink semantic keys + drift tests | `nonsmooth.py` (landed) |
| Registry/dashboard home for new axes | `primitive_coverage.py` (Decision #24) |
| Oracle harness + anti-cheat scoring | Evaluator program (`EVALUATOR_PLAN.md`) |
| Deterministic streams for estimator contracts | S4 RNG / Decision #18 |

**One row corrected against `CORE_SUBSTRATE_VIEW.md` (2026-08-18):** an earlier
draft listed "F6 `vmap` machinery" as the funded batched-axis entry point. The
substrate view's S8 (and the integrated plan's de-duplication table) record
that the `batching_rule` axis is closed over a **Python for-loop** — a named
Decision #29 instance with no owning row. The `vmap` *surface* exists; real
batching does not. The coefficient axis therefore cannot ride existing
batching: real `batching_rule`s are a **dependency** of AD-JET-STRUCT-1/
AD-JET-IR-1, and the jet mode is the second forcing function for making
batching real (after game theory G4). See §5a.

Net-new work: the algebra substrate + ODE table, the law oracles, the
structured-jet kernels, and the IR descent — §7.

### 5a. Cross-check against CORE_SUBSTRATE_VIEW (2026-08-18)

The plan was checked against the nine-substrate synthesis in
[`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md). Findings, incorporated
above:

| Substrate | Relationship to this plan |
|---|---|
| **S8 transform substrate** | Three touch points. (1) **Real `batching_rule`s are a dependency, not an asset** — the funding-table correction above; AD-JET-* joins game theory G4 as a forcing function. (2) **Implicit-diff strict-complementarity hardening (H3)** — "one fix, three consumers, no row" — is adopted into AD-OPERATOR-1's scope (§7), since that slice already refactors `implicit.py`. (3) **Schedule-level autodiff** (CAKE capability #1: transpose the S2 schedule object) is the *same adjunction as §3.5 applied at the schedule level* — related, deliberately **not** claimed by this plan; it stays gated on S1+S2 per the substrate view. The two must eventually agree on transpose vocabulary, which is an argument for landing §3.5's value-level `OperatorTangent` first |
| **S6 structural-op tranche** | The PDE demand row **already names "jet AD"** as a demanded capability — a second in-tree consumer for AD-WEIL-1 beyond the law harness, materially strengthening its #29 position. The PDE Chebyshev/DST lane is likewise the named candidate consumer that would un-gate `ChebJet` (§2.2) |
| **S5 `numeric_policy` carrier** | The coefficient/cotangent `numeric_policy` key (§2.3) requires the generalized below-Graph-IR carrier, which S5 records as fragment-only with **no owning row**. AD-JET-IR-1 is the **fourth mandating consumer** (after CAKE, game theory §6, PDE §III.4) — added weight behind the substrate view's flagged input to the integrated plan |
| **S4 keys + certificates** | Full alignment: §2.3's semantic keys are S4 instances; `LinearSolveInfo`, the Law dashboard, the measured conditioning envelope (§3.8), and `TaylorModel` enclosures follow the certificates-not-booleans discipline |
| **S3 calibration + arbiter** | `TaylorModel` enclosures are the strongest available form of the **accuracy certificate** the Decision #28 accuracy-budgeted arbiter needs (CAKE capability #4: accuracy budget as a search axis). Named as AD-CERT-1's candidate consumer (§7) |
| **S2 schedule object** | Jet kernel packages ride the Schedule Object digest (`SCHEDULE_OBJECT_DESIGN.md`) like every other content-addressed carrier; no new mechanism |
| **§0.1 verification discipline** | Adopted: the substrate view machine-checks every prose-only mathematical claim it inherits (13/13). This plan's §3 recurrences, cost counts, and the diagonal embedding get the same treatment — an executable math-verification harness is an AD-LAW-1 acceptance criterion (§7), so no slice builds on unverified prose |

---

## 6. Relationship to LAYOUT-ALG-1 — the sixth consumer

`CUTE_IR_ASSESSMENT.md` §3 counts five independent in-tree asks for layout
reasoning. The jet design is the sixth, identified before it was built:

| Jet need | What it is | Layout-algebra op | Rung |
|---|---|---|---|
| Attach the coefficient axis to a tile | product of the primal layout with a `(k+1)` mode | `logical_product` | L1 |
| Planar (SoA) vs interleaved (AoS) coefficients | mode regrouping — the operation §2 of the assessment proved the product variants are | `group_modes`/regroup | L1 |
| Jet footprint ceiling in LDS/TMEM/threadgroup memory | materialization extent | `cosize` | L3 |
| Jet-epilogue fusion legality | factorization of the coefficient-extended read layout through the producer's partition | `⊑` | L3 |
| Content-addressed jet-kernel digests | decidable layout equality → canonical form | `coalesce` | L1 |
| Emitting the triangular Cauchy loop once, not per backend | index math through the shared algebra instead of new string templates | `crd2idx` | L4 |

Consequences:

- **AD-JET-IR-1 depends on L1 + L5** so the coefficient axis lives in the
  shared nested `#tile.layout` carrier from day one. A private "jet axis"
  notion would be the Decision #30 bespoke-walker anti-pattern committed by
  the plan that just diagnosed it.
- **L0's one-home decision is the template for the jet kernel's eventual
  descent**: the ODE table + W-ring ops will be needed by both the Python
  synthesizer and the C++ MLIR pipeline — the same two-compiler seam. When
  AD-JET-IR-1 promotes the numpy reference, it adopts L0's shape verbatim
  (one C++ implementation, ctypes binding, A1 fail-closed diagnostic, A2
  declared build dependency, no fallback path). Until then the numpy lane is
  the reference implementation, which is the established repo pattern and is
  what the law oracles certify.
- Interleaved-vs-planar is a **performance key** (§2.3): expressible
  everywhere, chosen non-defaultly only behind architecture-owned
  measurement — the same blocker class as `raster_order`
  (ROCM-CALIB-1's lesson; assessment §3.2).

**Applied 2026-08-18:** `CUTE_IR_ASSESSMENT.md` §3 now carries this as its
sixth consumer row, and its counts (§0 bottom line, §3 lead-in, §4 "For")
were updated to match. That document notes the row is different in kind from
the first five: it was found during design, before any code existed, which
is the layout-algebra argument working prospectively rather than as
archaeology.

---

## 7. Build sequence — AD-NEXTGEN (proposed slices)

Sizing is an estimate, not a measurement. Ordering respects Decision #31:
derive → prove differentially → only then retire. Nothing here jumps the
integrated plan's Orders 1–11; the first slice is evidence infrastructure in
the same class as Orders 12–13.

### AD-LAW-1 — law oracles over today's registries (~1–1.5 weeks · no dependencies · host-free)

Laws 1 + 3 (+ 5, extending the existing kink drift test) as evaluator
oracles over the current `_VJPS`/`_JVPS` **and** `_VJPS_GEO`/`_JVPS_GEO`
registries, rendered to a generated law dashboard under
`docs/audit/generated/`.

*Acceptance:* every registered VJP/JVP pair swept by the adjoint law with
dimension-scaled probe counts; failures filed as pre-existing defects with
negative fixtures (#10a), never silently tolerated (a red cell is a real
finding, reported per claim-integrity rules); dashboard drift-gated via
`check_generated_docs.sh`; **this plan's §3 mathematics machine-checked** in a
`verify_autodiff_math.py`-style harness (the §0.1 discipline from
`CORE_SUBSTRATE_VIEW.md`): the exp/tanh/sin-cos recurrences against numpy
derivatives at small k, the `(k+1)(k+2)/2` jet-matmul count, and the §3.1
diagonal embedding on small instances — including the **negative fixture**
that the naive `εᵢ ↦ ε` "surjection" is not an algebra map (#10a, per the
§3.1 review correction) — no slice builds on unverified prose.
*Consumers:* E2E-REAL-6 migration gates (§5 item 5); every later slice. **Do
not land any AD-WEIL-1 code before this exists** — the oracles are what make
the migration provable.

**Triage note (2026-08-18) — the calling convention is part of the datum.**
Sweeping the registries for the `jvp_clamp` class turned up a second, larger
one that no rule-vs-rule signature scan can see: what the *tape* hands a rule
depends on how the canonical forward was *called*. `_describe` recorded
array-likes and python int/float only, so a configuration argument passed
positionally either vanished from the record entirely (`str`/`bool`/`None` —
a mean-reduction forward with a **sum gradient**, no error) or became a
float64 literal that was handed to the forward itself and then replayed as an
extra positional the keyword-only rule could not bind. Empirically: **0 of the
spec-covered ops disagreed silently once routed, 11 of 13 raised before it.**
The fix is one binding site, not ~100 rule signatures — operands are derived
from the registered rule's own positional slots (Decision #30), everything
else is configuration and is recorded under its canonical parameter name.
Forward mode makes the same split. Two rules were also differentiating a
different function than the forward *by vocabulary*: `tri_solve` declared
`upper` where the forward says `lower` (and solved the raw matrix instead of
the selected triangle), `segment_reduce` declared `reduce` where the forward
says `op` (so its max/mean handling was unreachable, and broken once
reached). Both are FD-pinned. The remaining 59 forward-key swallows are
pinned as open findings in `tests/unit/test_autodiff_laws.py`.

### AD-WEIL-1 — algebra substrate + `Dual()` + `TruncatedJet(k)` (~3 weeks · after AD-LAW-1 · host-free)

`DifferentialAlgebra` protocol; generic finite-multiplication-table
substrate; `Dual()` and `TruncatedJet(k)` instances; the holonomic ODE table
for the pointwise/activation families; `DerivativeContract` registration for
the pointwise + multilinear families (the latter imported from
`MULTILINEAR_PRIMITIVES`); Laws 2 + 4 in the harness; `coefficient_scaling`
+ jet `numeric_policy` semantic keys; measured conditioning envelope (§3.8)
in the dashboard.

*Acceptance:* Law 2 at tolerance 0 on polynomial primitives; Law 4 (jet ≡
nested `Dual()`) green per family; the W6.3/W6.4 substrate hypothesis proven
by running `Cl(3,0)` over the same table substrate against `ga/signature.py`
as oracle; **no production authority changes** — `_VJPS`/`_JVPS` untouched,
new code is reference + oracle only (#29 satisfied by the law harness +
eager `hvp` replacement as consumers).

### AD-JET-STRUCT-1 — structured jets + estimator mode (~4 weeks · after AD-WEIL-1)

Jet rules for the fused/structured families via §3.4: `flash_attn`
(online-softmax jets), norm chain, `logsumexp`/`softmax`; STDE-style
Laplacian/Hessian-trace estimator on Philox with the Law-6 unbiasedness
check; `control_at_order = 0` declared and tested at `max`-tie and branch
sites. Hand-rule retirement begins **only** for families with green Law-4
proof.

*Acceptance:* jet-vs-nested differential proof per family; kink-policy
consistency between first-order VJP and higher-order forward at the same tie
points (Law 5 extended); estimator mean-convergence under fixed seeds;
`random` effect carried per W2.2. Physical carriers, when they come, are
x86/gfx1151 first per the integrated plan's architecture-expansion rules.

### AD-JET-IR-1 — IR descent of the coefficient axis (~5–6 weeks, research-adjacent · after W4-PRODUCT-1, LAYOUT-ALG-1 L1/L5, AD-JET-STRUCT-1, and real `batching_rule`s per §5a/S8)

`TaylorLiftPass(W)` replacing the "run AutodiffPass twice" bullet; the
coefficient axis as a typed layout mode in the L5 nested `#tile.layout`
carrier; jet variants of the content-addressed Schedule/Tile packages (first
candidate: the `schedule.attention_backward` sibling); L0-shape C++ home for
the W-ring ops + ODE table with ctypes binding (A1/A2); #32 boundary
verifier for dropped coefficient/enclosure attributes; jet rematerialization
routed through W5.1's residual-policy machinery (a jet multiplies activation
footprint by k+1 — remat interaction is in-scope here, not an afterthought).

*Acceptance:* paired CPU oracle for the lifted product; per-family digest
identity; bit-level agreement with the Python reference on the migrated
families; fail-closed on unsupported (op, W) pairs — a missing jet rule is a
named diagnostic, never a silent fall-through (#21/#21a).

### AD-OPERATOR-1 — operator tangents (~2 weeks · after AD-LAW-1; independent of jets)

Promote `implicit.py`'s matvec pattern to an `OperatorTangent` type
(composition, `.T`, solve-consumption); refactor `ihvp`/`root_vjp`/
`root_jvp`/`adjoint_state_grad` onto it; record the declared relationship to
the materialized `@f__bwd` ABI (#31); **adopt S8's implicit-diff hardening**:
a strict-complementarity certificate at `custom_root` solutions (H3 —
`CORE_SUBSTRATE_VIEW.md` S8's "one fix, three consumers" item, previously
unowned; consumers: game theory, Riemannian-OT, S-series), emitted as a
certificate in the S4 discipline alongside `LinearSolveInfo`, fail-closed
when the check cannot be evaluated.

*Acceptance:* behavior-identical refactor of `implicit.py`'s public surface
(its existing tests are the regression net); adjoint law on the operator
level (`⟨Av,u⟩ = ⟨v,Aᵀu⟩` through the type's own transpose); fail-closed
non-convergent solves preserved; complementarity certificate present on a
degenerate-root negative fixture (rejects) and a clean-root positive one.

### AD-CERT-1 — certified mode (deferred · consumer-gated per #29)

`TaylorModel(k)` with outward rounding; `debug.check_grad` upgraded from
spot-check to enclosure proof; Law 6. **Not scheduled** until a consumer
asks (candidates: evaluator anti-cheat hardening, solver-lane verification,
and — per §5a — the Decision #28 **accuracy-budgeted arbiter**, whose
accuracy certificate an enclosure supplies in the strongest available form,
the S3 certificate-driven-arbitration pattern); recorded here so the
interface (`DifferentialAlgebra`) is not designed in a way that forecloses
it.

---

## 8. What not to do

- **Do not add jet ops to Graph IR.** `tessera.matmul` does not grow a jet
  sibling; the jet enters as a transform parameterized by `W` over existing
  primitives, exactly as the layout assessment ruled `logical_divide` out of
  Graph IR.
- **Do not build `ChebJet`/`TaylorModel`/`MixedPartial` speculatively** —
  each waits for a named consumer (#29). The protocol exists so adding one
  later is an instance, not a redesign.
- **Do not collapse `vjp.py`/`jvp.py`/`geometric/` ahead of the Law-4 proof
  for the family in question** (#31's documented failure mode). The
  surviving path must carry what the deleted one carried — including kink
  policies and dtype behavior, not just values on smooth interior points.
- **Do not let the numpy reference and the eventual C++ jet kernel coexist
  as two production authorities.** L0's one-home rule applies at AD-JET-IR-1
  time; before that, the numpy lane is reference + oracle by declaration.
- **Do not claim device evidence from host runs.** Laws 1–5 are host-free by
  design; any "jet kernel executes natively" claim waits for exact-device
  packets on the owning box, per the standing claim-integrity rules.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Monomial-basis conditioning caps useful k lower than expected | Measured envelope in AD-WEIL-1's dashboard before any IR investment; `ChebJet` is the designed escape hatch, gated on that measurement |
| Jet memory footprint ((k+1)× activations) breaks real workloads | W5.1 residual policy owns the remat decision; in-scope for AD-JET-IR-1 acceptance, not deferred |
| Law-3 sweep finds widespread red in existing registries | That is the sweep working — findings are pre-existing defects, triaged before migration; budget for triage is inside AD-LAW-1's estimate |
| Table substrate fails to serve Clifford cleanly (W6.4-note risk) | It is an acceptance criterion of AD-WEIL-1 with `ga/signature.py` as oracle, so the hypothesis is settled early and cheaply, not discovered late |
| Control-flow jets interact badly with W4's residual ABI | `control_at_order = 0` is declared v1 semantics; anything beyond (differentiating through trip counts) is explicitly out of scope |
| Estimator modes blur the deterministic/stochastic boundary | `random` effect is mandatory, fail-closed (W2.2); `@jit(deterministic=True)` rejects estimator-mode gradients by construction |

---

## 10. Exit criterion

One defensible, measured claim, in the spirit of the W6 exit line and the
RNOT acceptance test: **the k+1-vs-2ᵏ scaling curve on a real workload** —
order-k directional derivatives of an attention block via fused jets vs
nested forward-over-forward, same hardware, same numerics budget — plus the
law dashboard green across migrated families and zero remaining parallel
registries (`geometric/` absorbed). If the curve does not separate by k=3,
the IR descent was not worth it and the program stops at the Python
reference + law infrastructure, which are independently valuable.
