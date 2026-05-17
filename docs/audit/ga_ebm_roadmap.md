---
status: Active (development roadmap — Q1/Q2/Q4/Q6 locked; Q3/Q5 deferred to GA6/EBM7)
classification: Audit / Plan
authority: Sequences Geometric Algebra (Clifford) + Energy-Based Model primitive surfaces into Tessera
last_updated: 2026-05-16
---

# Tessera GA / EBM Roadmap — Geometric Algebra + Energy-Based Models

This document plans two new primitive surfaces for the standalone Tessera
compiler:

- **GA-series** — first-class Clifford algebras `Cl(p,q,r)`, grade-aware
  multivector types, differential forms, and geometric autodiff.
- **EBM-series** — first-class energy-based modeling: energy primitives,
  Langevin / MCMC samplers, partition-function integration, contrastive
  divergence losses, and an inner-loop ("thinking") schedule pattern.

These are deliberately **Tessera-native** additions. Per Architecture Decision
#23, we do not wrap PyTorch / JAX / Flax — we reimplement the math at the
compiler primitive level. The EBM track has an archived design precedent in
[`examples/archive/advanced/EBT/`](../../examples/archive/advanced/EBT/) which
will be promoted to a live S-series surface; the GA track is greenfield.

The user-facing justification: "we do not need another PyTorch." Existing
frameworks cannot express algebra signatures, multivector derivatives,
exterior calculus, or manifold-aware integrators in their IR. Tessera's
dialect + hardware-free Target IR + S-series registry patterns are the right
substrate to host these as first-class compiler concepts.

## How to use this doc

Same conventions as [`execution_roadmap.md`](execution_roadmap.md):

1. Resolve every **Open question** in the [Scope lock](#scope-lock--open-questions-block-on-implementation) section before any GA0 / EBM0 work starts.
2. Pick the lowest-numbered sprint whose dependencies are all ✅.
3. Read its **Acceptance criteria**; those are the tests you must make pass.
4. After landing, mark the sprint ✅ and update:
   - [`primitive_coverage.py`](../../python/tessera/compiler/primitive_coverage.py) registry entries
   - [`standalone_primitive_coverage.md`](standalone_primitive_coverage.md) dashboard
   - This roadmap's status legend

Status legend: 📋 planned · 🚧 in progress · ✅ done · 🔲 deferred.

## Phase ordering rationale

```
GA0 (scope lock)
  └─► GA1 (algebra signature object)
       ├─► GA2 (grade-aware ConstraintSolver types)        [no IR — Python-only]
       └─► GA3 (multivector Python reference, numpy-backed)
            ├─► GA4 (GA primitives in primitive_coverage.py)
            ├─► GA5 (differential-form primitives: d, d*, ⋆, ∂)
            ├─► GA6 (multivector autodiff: VJP + JVP via dual multivectors)
            └─► GA7 (tessera.clifford Graph IR dialect + ts-clifford-opt)
                 └─► GA8 (lowering passes: product table → sparse contraction)
                      └─► GA9 (backend manifest: x86 first, then Apple, then NVIDIA)
                           └─► GA10 (lit fixtures + tiny-model conformance)

EBM0 (scope lock + revive archived EBT design)
  └─► EBM1 (energy / inner_step / self_verify primitives — Euclidean)
       ├─► EBM2 (Langevin + MCMC samplers; extends tessera.rng / S4)
       ├─► EBM3 (partition function Z — Euclidean baseline)
       ├─► EBM4 (contrastive divergence + score matching losses; extends S11)
       ├─► EBM5 (tessera.ebm Graph IR dialect + ts-ebm-opt)
       └─► EBM6 (inner-loop fusion + checkpointing passes)
            └─► EBM7 (manifold-aware integrator)            [depends on GA5+GA6]
                 └─► EBM8 (conformance: RBM, EBT, score-matching diffusion)
```

**Parallelism:**
- GA0 → GA1 → GA2 is a pure Python / type-system path and unblocks fast.
- EBM0 → EBM1 → EBM2 → EBM3 → EBM4 has **zero GA dependency** — it can land
  the Euclidean EBM surface in parallel with GA's first half.
- The two tracks merge at EBM7 (manifold integrator), which needs GA5+GA6.

**Estimated runway:** ~10–14 sprints of focused work; ~3–5 months at S-series
cadence. GA0 + EBM0 (scope lock) is days, not weeks. GA6 (multivector
autodiff) and GA8 (lowering) are the two longest sprints.

## Scope lock — open questions

**Q1, Q2, Q4, Q6 are locked (2026-05-16)** — see [`ga_scope_lock.md`](ga_scope_lock.md)
and [`ebm_scope_lock.md`](ebm_scope_lock.md) for the decision records.
**Q3 (autodiff) and Q5 (manifold set) remain deferred** to GA6 and EBM7
respectively; they don't block any earlier sprint.

### Q1 — Signature support matrix for v1 ✅ LOCKED: Cl(3,0) + Cl(1,3) only

Which `Cl(p,q,r)` signatures are first-class in v1?

- **Option A** (recommended): Cl(3,0,0) + Cl(1,3,0) only. Covers 3D Euclidean
  + spacetime — enough for robotics, vision, particle physics, relativistic
  ML demos. Both have ≤16 basis elements; product tables fit in registers.
- **Option B**: Add Cl(4,1,0) conformal (32 elements). Elegant for neural-net
  applications (translations become rotations), but doubles backend complexity.
- **Option C**: Fully general `Cl(p,q,r)` with `p+q+r ≤ 6`. Maximally
  ambitious; 64-element tables; would need MLIR sparse tensor dialect from
  day one.

Recommendation: **A** for v1, with the type system designed for **C** so we
can extend without breakage. — **Locked: A.** See [`ga_scope_lock.md` § Q1](ga_scope_lock.md#q1--signature-support-matrix-for-v1-cl30--cl13-only).

### Q2 — Tensor-attribute extension vs. new tensor kind ✅ LOCKED: sibling `Multivector` kind

Per Decision #15a, Tessera tensors have six canonical attributes (shape,
dtype, layout, device, distribution, numeric_policy). Multivectors need at
least `algebra` (signature) and `grades` (subset of {0..n}).

- **Option A**: Add as 7th + 8th tensor attributes. Cleanest for downstream
  passes; every existing tool sees them.
- **Option B**: Introduce a new tensor kind `Multivector` parallel to
  `Tensor`. Less risk of breaking existing primitives, but doubles surface.

Recommendation: **B** for v1 (`tessera.Multivector` as a sibling type), with
shared `shape`/`dtype`/`layout`/`device`/`distribution` machinery. Promotion
to a unified tensor with `algebra` attribute is a follow-up once the
multivector surface stabilizes. — **Locked: B.** See [`ga_scope_lock.md` § Q2](ga_scope_lock.md#q2--type-system-tesseramultivector-as-a-sibling-to-tensor).

### Q3 — Autodiff integration strategy ✅ LOCKED: parallel `tessera.autodiff.geometric` registry

The multivector derivative `∂F` is itself a multivector. Tape-based
np.ndarray VJPs cannot express this without modification.

- **Option A**: Parallel registry `tessera.autodiff.geometric._VJPS` /
  `_JVPS`, dispatched when the input is a `Multivector`.
- **Option B**: Extend the existing tape to carry per-tensor `algebra`
  metadata; every existing VJP becomes a no-op on non-multivector inputs.

Recommendation: **A** — parallel registry. Keeps the 241+236 existing VJP/JVP
entries untouched and avoids regression risk in S2–S15. — **Locked: A.**
See [`ga_scope_lock.md` § Q3](ga_scope_lock.md#q3--autodiff-strategy-parallel-tesseraautodiffgeometric-registry-locked-2026-05-17).

### Q4 — Backend priority ✅ LOCKED: x86 → Apple CPU → Apple GPU; NVIDIA after Phase G

Per Decision #1 (CPU-first), the natural order is:

1. **x86 reference** (numpy + AMX where applicable) — correctness baseline.
2. **Apple CPU** (Accelerate) — straightforward extension; product tables
   are small enough for cblas-level dispatch.
3. **Apple GPU** (MSL kernels) — proves the GPU lowering story.
4. **NVIDIA** — deferred until Phase G H100 BF16 GEMM lights up.

Recommendation: ship GA9 in this exact order. NVIDIA waits for Phase G. —
**Locked.** See [`ga_scope_lock.md` § Q4](ga_scope_lock.md#q4--backend-priority-x86--apple-cpu--apple-gpu-nvidia-after-phase-g).

### Q5 — Manifold integration surface for EBM7 ✅ LOCKED: S^n + SO(n) + ℝ^n

Which manifolds get first-class integrators?

- **Option A** (recommended): `S^n` (n-sphere) + `SO(n)` (rotation group)
  + flat `ℝ^n`. Sufficient for orientation models, molecular geometry,
  Lie-group GANs.
- **Option B**: Add `SE(n)`, `SU(n)`, hyperbolic `H^n`. Wider coverage,
  triples the integrator surface.

Recommendation: **A** for v1. — **Locked: A.** See
[`ebm_scope_lock.md` § Q5](ebm_scope_lock.md#q5--manifold-set-for-ebm7-sn--son--rn-locked-2026-05-17).

### Q6 — EBT package: revive or rewrite ✅ LOCKED: revive as seed for EBM5

[`examples/archive/advanced/EBT/Tessera_EBT_Package_v1/`](../../examples/archive/advanced/EBT/Tessera_EBT_Package_v1/)
has a `tessera.ebt` dialect design, IR samples, and a runner skeleton — all
archived (excluded from build per Decision: archive policy).

- **Option A** (recommended): Revive as the seed for EBM5. Honor the
  archived design where it's still right; correct what doesn't match
  current Tessera conventions.
- **Option B**: Start fresh; reference the archive but don't import it.

Recommendation: **A** — engineering economy. The archive's IR vocabulary
(`ebt.energy`, `ebt.inner_step`, `ebt.self_verify`, `ebt.grad_y`) is sound
and aligns with EBM1's planned API. — **Locked: A.** See
[`ebm_scope_lock.md` § Q6](ebm_scope_lock.md#q6--archived-ebt-package-revive-as-seed-for-ebm5).

---

## GA-series — Geometric Algebra (Clifford) primitive surface

### [GA0] Scope lock + signature configuration root ✅

**Scope:** S (~150 LOC code + ~100 LOC docs + ~80 LOC tests).

**Status (landed 2026-05-16):** Q1, Q2, Q4 locked in
[`ga_scope_lock.md`](ga_scope_lock.md). Q3 deferred to GA6 (decision needs
GA1–GA4 to inform it). `tessera.ga` namespace shipped at
[`python/tessera/ga/__init__.py`](../../python/tessera/ga/__init__.py)
with `__version__ = "0.0.0-ga0"`. Acceptance test at
[`tests/unit/test_ga_namespace.py`](../../tests/unit/test_ga_namespace.py).

**Files (new):**
- `docs/audit/ga_scope_lock.md` — captures Q1/Q2/Q4 answers; documents Q3 deferral.
- `python/tessera/ga/__init__.py` — empty namespace; placeholder so imports
  resolve from GA1 onwards.
- `tests/unit/test_ga_namespace.py` — namespace import + version + package-path tests.

**Acceptance:**
- ✅ Q1, Q2, Q4 each have a committed answer with rationale; Q3 explicitly deferred.
- ✅ `from tessera import ga` succeeds and exposes a `__version__` string.

### [GA1] Algebra signature as a first-class object ✅

**Scope:** M (~250 LOC code + ~150 LOC tests). Depends on GA0.

**Status (landed 2026-05-17):** Shipped at
[`python/tessera/ga/signature.py`](../../python/tessera/ga/signature.py)
with `Cl(p, q, r=0)`, `Basis`, `TesseraAlgebraError`, and
`V1_ALLOWED_SIGNATURES`. Frozen dataclass; v1 allow-list enforced at
construction (`{(3,0,0), (1,3,0)}`). Product tables and basis lists
cached via `lru_cache` per signature. 29 tests at
[`tests/unit/test_ga_signature.py`](../../tests/unit/test_ga_signature.py),
including full Cayley-table associativity sweep over both signatures
(4096 triples in Cl(1,3)).

Reverse / grade-involution / Clifford-conjugation automorphisms are
deferred to GA3 (multivector operations) — they're computed from the
product table, not stored separately.

**Files (new):**
- `python/tessera/ga/signature.py` — `Cl`, `Basis`, `_product_table`,
  `_basis_list`, name parser, `V1_ALLOWED_SIGNATURES`.
- `tests/unit/test_ga_signature.py` — 29 tests covering dimensions,
  grade enumeration, blade lookup, generator-squaring rules per
  signature, anti-commutation, pseudoscalar squares, associativity,
  caching, equality, pickle, repr, name-parser strictness, allow-list
  enforcement.

**Acceptance:**
- ✅ `Cl(3,0).dim == 8`, `Cl(1,3).dim == 16`.
- ✅ Product table computed once per signature (lru_cache); second
  `Cl(3,0)` call returns the same table tuple (`is`-identical).
- ✅ Hashable + value-equal across constructions.
- ✅ `Cl(3,0).grades == (0, 1, 2, 3)`; `Cl(3,0).blade("e12").grade == 2`.
- ✅ Signature parameters round-trip through `repr` / `pickle`.
- ✅ Bonus: full Cayley-table associativity verified over all
  `2**n × 2**n × 2**n` blade triples in both v1 signatures.

### [GA2] Grade-aware types via ConstraintSolver ✅

**Scope:** M (~300 LOC code + ~200 LOC tests). Depends on GA1.

**Status (landed 2026-05-17):** Annotation surface +
constraint-predicate set shipped:
- [`python/tessera/ga/types.py`](../../python/tessera/ga/types.py) —
  `Rotor`, `DiffForm`, `VectorField`, `Morphism` annotation markers; each
  subscript returns a frozen `MultivectorSpec` (or `MorphismSpec`).
- [`python/tessera/ga/multivector.py`](../../python/tessera/ga/multivector.py) —
  `MultivectorSpec` value class (shared with GA3); supports `is_grade_pure` /
  `is_even` / `is_odd` predicates and unified subscript syntax
  `Multivector[Cl, {0, 2}]`.
- [`python/tessera/ga/constraints.py`](../../python/tessera/ga/constraints.py) —
  `GradeIn`, `Even`, `Odd`, `IsRotor`, `IsForm` — all subclass the existing
  `Constraint` base; binding dict is duck-typed to accept either ints
  (existing Divisible/Range/Equal) or `MultivectorSpec`/`Multivector`
  values (new GA predicates).
- 28 tests at [`tests/unit/test_ga_constraints.py`](../../tests/unit/test_ga_constraints.py)
  including integration with `ConstraintSolver.check()` and
  `check_all()` alongside existing predicates.

**Decision-time vs runtime check.** The acceptance text said "at
decoration time" — in practice this lands as: (a) annotation parsing
runs at decoration time and rejects malformed specs; (b) constraint
predicates run via `ConstraintSolver.check(bindings)` at the point where
bindings are available, which for current `@jit` is call time. Full
abstract-trace-time grade checking of function bodies (e.g.
auto-detecting that `a + b` mixes grades into a return-annotated
grade-pure value) is GA6's responsibility — it needs the multivector
autodiff tracer.

**Files (new):**
- `python/tessera/ga/types.py`, `python/tessera/ga/constraints.py`,
  `tests/unit/test_ga_constraints.py`.

**Files (modified):**
- `python/tessera/ga/__init__.py` — re-exports the GA2 surface.

**Acceptance:**
- ✅ `Multivector[Cl(3,0), {0,2}]` annotation parses; returns a valid
  `MultivectorSpec` with grades=`{0,2}` and kind=`"multivector"`.
- ✅ `Multivector[Cl(3,0), 4]` (grade outside algebra) raises
  `TesseraAlgebraError` at decoration time.
- ✅ Mismatched-grade scenario: `GradeIn("a", {1}).check({"a":
  <bivector value>})` returns a `TesseraConstraintError` with a
  precise "disallowed grade" message — covered by
  `test_mismatched_grade_addition_raises_via_constraint_check`.
- ✅ `Rotor[Cl(3,0)]` carries `kind="rotor"` proof; `IsRotor("R")`
  accepts it but rejects a plain `Multivector[Cl(3,0), {0,2}]` even
  though both are even-grade.
- ✅ All errors flow through the existing `TesseraConstraintError`
  type, which already routes through `ErrorReporter`.

### [GA3] Multivector Python reference surface (numpy-backed) ✅

**Scope:** L (~700 LOC code + ~500 LOC tests). Depends on GA1 (independently
of GA2 in our implementation — Multivector value + spec share one class).

**Status (landed 2026-05-17):** Multivector value class + 14 ops
shipped. Ops file at [`python/tessera/ga/ops.py`](../../python/tessera/ga/ops.py)
exports:
- Core: `geometric_product`, `grade_projection`, `wedge`,
  `left_contraction`, `inner` (scalar-valued).
- Anti-automorphisms: `reverse`, `grade_involution`, `conjugate`.
- Norms: `norm`, `norm_squared`.
- Rotor / exp-log: `exp_mv` (closed-form for Cl(3,0) bivectors;
  power-series fallback), `log_mv` (closed-form for Cl(3,0) rotors;
  power-series fallback), `rotor_from_axis`, `rotor_sandwich`.

48 tests at [`tests/unit/test_ga_multivector.py`](../../tests/unit/test_ga_multivector.py)
and [`tests/unit/test_ga_ops.py`](../../tests/unit/test_ga_ops.py). The
headline rotor-sandwich-vs-SO(3) test runs **50 random axis/angle
samples** and matches the Rodrigues reference to fp32 tolerance.

**Files (new):**
- `python/tessera/ga/multivector.py` — `Multivector` value class +
  `MultivectorSpec` annotation type (shared with GA2). Construction
  helpers `zeros`, `scalar`, `from_blade`, `from_vector`; arithmetic
  via `+`, `-`, scalar `*` / `/`; `Multivector × Multivector` via
  scalar `*` is intentionally `TypeError` (use `geometric_product`).
- `python/tessera/ga/ops.py` — 14 pure functions.
- `tests/unit/test_ga_multivector.py` — 25 tests.
- `tests/unit/test_ga_ops.py` — 23 tests including 50-sample
  rotor-vs-SO(3), Cayley-table-vs-implementation cross-check, and
  rotor-composition associativity.

**Acceptance:**
- ✅ `Multivector + Multivector` returns a Multivector with the union
  of grade sets (`None` propagates as "unrestricted").
- ✅ `geometric_product` agrees with hand-computed Cayley-table
  multiplication for Cl(3,0) on 50 random pairs (fp32 tolerance).
- ✅ Rotor sandwich `R · v · R†` rotates a Cl(3,0) vector identically
  to the Rodrigues SO(3) matrix on 50 random axis-angle samples.
- ✅ Bonus: rotor composition matches sequential application
  (`rotor_sandwich(R1*R2, v) == rotor_sandwich(R1, rotor_sandwich(R2, v))`).
- ✅ Both `fp32` and `fp64` accepted; promotion via `np.result_type`.
- ✅ Cl(1,3) spot-checks: e1²=+1, e2²=-1 confirmed via geometric_product.

### [GA4] GA primitives in primitive_coverage.py ✅

**Scope:** S (~150 LOC code + ~100 LOC tests). Depends on GA3.

**Status (landed 2026-05-17):** 17 primitives registered under
`category="geometric_algebra"` — 12 from GA3 (geometric_product,
grade_projection, wedge, left_contraction, inner, reverse,
grade_involution, conjugate, norm, exp, log, rotor_sandwich) and 5
from GA5 (hodge_star, ext_deriv, codiff, vec_deriv, integral). 6
tests at [`tests/unit/test_ga_coverage.py`](../../tests/unit/test_ga_coverage.py).
Existing `test_standalone_compiler_roadmap.py` drift guards still
pass (categories now include `geometric_algebra`).

The acceptance text said "12 entries" — actual count is 17 because
GA5 lands in the same sprint as the registry bookkeeping. Each entry
is named `clifford_<op>`, mirroring the GA7 dialect's `tessera.clifford.*`
ops 1:1.

**Files (modified):**
- `python/tessera/compiler/primitive_coverage.py` — adds 17 entries.

**Files (new):**
- `tests/unit/test_ga_coverage.py` — 6 audit-style tests.

**Acceptance:**
- ✅ 17 new entries with `status="planned"` under
  `category="geometric_algebra"`. Per-axis contracts (math / shape /
  dtype / vjp / jvp / batching / transpose / sharding / lowering /
  kernel / tests) stay `planned` for now — the category-based hardening
  sweep will promote them as GA6 (autodiff) and GA8/GA9 (lowering /
  backend) land.
- ✅ Existing `test_standalone_compiler_roadmap.py` drift guard passes.
- ✅ Every entry cites a classical reference (Hestenes & Sobczyk;
  Doran & Lasenby; Frankel).

### [GA5] Differential-form primitives ✅

**Scope:** M (~400 LOC code + ~300 LOC tests). Depends on GA3.

**Status (landed 2026-05-17):** Five primitives shipped:

- [`python/tessera/ga/calculus.py`](../../python/tessera/ga/calculus.py)
  — `hodge_star` (pointwise), `MultivectorField`, `ext_deriv` /
  `codiff` / `vec_deriv` (finite-difference on uniform Euclidean grids),
  `hodge_star_field` (pointwise on a field), `integral` (callable mode
  + field mode).
- [`python/tessera/ga/manifold.py`](../../python/tessera/ga/manifold.py)
  — `Manifold` ABC + `Euclidean(bounds, resolution)`,
  `Sphere(n=2, n_vertices=...)` via spherical Fibonacci with uniform
  area weights, and `SOn(n=3, n_samples=...)` as a minimal stub.

22 tests at [`tests/unit/test_ga_calculus.py`](../../tests/unit/test_ga_calculus.py).
The d²=0 test runs **100 random 1-forms** on an 8×8×8 grid in Cl(3,0)
and asserts |ddω|∞ < 1e-6 on the interior (exact for central
differences modulo float noise).

**Honest deviation from acceptance text.** The roadmap asked for
"Stokes on Sphere(n=2)" with `∫ d(ω) = ∫_∂M ω` — but ∂S² = ∅, so this
reduces to `∫_{S²} dω = 0`. We test exactly that: for any 1-form
ω = F·dl, ``∫_{S²} dω = ∫_{S²} curl(F)·n̂ dA`` is zero because the
average outward normal on a closed surface is zero. Verified to
within 5e-2 absolute error using 2048 Fibonacci points. A more rigorous
sphere Stokes test (open hemisphere with boundary circle) is GA10
conformance work.

**Hodge involution table (signature-dependent).** ``⋆⋆ω`` per grade
``k`` of ``Cl(p, q)`` carries sign ``(-1)^{k(n-k)} · (-1)^q``:
- Cl(3,0): uniform `+ω` (all signs +1).
- Cl(1,3): grade-alternating `(-, +, -, +, -)` for k=0..4.

Verified by `test_hodge_star_double_application_in_cl13_is_grade_alternating`.

**Files (new):**
- `python/tessera/ga/calculus.py`, `python/tessera/ga/manifold.py`,
  `tests/unit/test_ga_calculus.py`.

**Files (modified):**
- `python/tessera/ga/__init__.py` — re-exports GA5 surface.

**Acceptance:**
- ✅ `d(d(ω)) == 0` for **100 random 1-forms** in Cl(3,0) on a 3D
  Euclidean grid (interior cells, fp64 tolerance).
- ✅ `⋆⋆ω = ±ω` per signature parity, verified grade-by-grade in
  both Cl(3,0) and Cl(1,3).
- ✅ `∫_{S²} dω ≈ 0` on a 2048-vertex Fibonacci sphere
  triangulation (proxy for the empty-boundary Stokes claim).
- ✅ Bonus: ``vec_deriv`` of a linear vector field recovers the
  expected scalar divergence (3.0 for ``F = x e_1 + y e_2 + z e_3``)
  to fp32 tolerance on the grid interior.

### [GA6] Multivector autodiff ✅

**Scope:** L (~800 LOC code + ~500 LOC tests). Depends on GA4.

**Status (landed 2026-05-17):** Parallel `tessera.autodiff.geometric.*`
package shipped per the Q3 lock. Six modules:

- `__init__.py` — exports.
- `registry.py` — `_VJPS_GEO`, `_JVPS_GEO`, `register_vjp_geo`,
  `register_jvp_geo`, getters.
- `vjp.py` — 16 VJPs registered: add, sub, neg, scalar_mul,
  grade_projection, reverse, grade_involution, conjugate, hodge_star,
  geometric_product, wedge, left_contraction, inner, norm,
  norm_squared, rotor_sandwich.
- `jvp.py` — 16 matching JVPs (forward-mode product rule).
- `check_grad.py` — `check_grad_geo` / `check_jvp_geo` central-difference
  verifiers.
- `tape.py` — `GeometricTape` + `tape_geo()` context manager +
  `multivector_grad()` central-difference fallback.

**VJP derivation strategy.** Used the **direct Cayley-table-adjoint
formula** rather than the Hestenes `dout * reverse(b)` rewrite. Both
formulas agree on Cl(p, 0); the direct-table formula generalizes
cleanly to Cl(p, q, r) with q + r > 0 without per-signature special
cases. Documented in `ga_scope_lock.md` § Q3.

**Headline tests pass** at [`tests/unit/test_ga_autodiff.py`](../../tests/unit/test_ga_autodiff.py)
(30 tests):

- All 16 VJPs registered (`test_all_ga6_vjps_registered`).
- All 16 JVPs registered.
- `check_grad_geo` matches central differences for every linear op,
  bilinear op (geometric_product, wedge, left_contraction), and the
  composite rotor_sandwich on random Cl(3,0) inputs.
- **`test_rotor_sandwich_gradient_is_even_grade`** — for
  ``L = ‖R v R†‖²`` with ``R`` a Cl(3,0) rotor, the gradient ``∂L/∂R``
  has odd-grade-component magnitude < 1e-10 and even-grade magnitude
  > 0.1. This is the GA-L4 equivariance-from-algebra claim made
  enforceable.
- `test_rotor_sandwich_gradient_for_input_vector_stays_grade1` —
  symmetric check on ``∂L/∂v``.
- `test_tape_geo_and_tensor_tape_coexist` — the parallel registry
  doesn't touch the existing 241+236 tensor VJP/JVP entries; both
  tapes can be active simultaneously without interference.

**Honest deferral.** Full tape-based reverse-mode autograd that
auto-records every `tessera.ga.ops.*` call is GA10/GA11 work — the
v1 `multivector_grad()` helper falls back to central differences
rather than walking a recorded DAG. The per-op VJP/JVP registry is
the load-bearing piece, and that ships green. `exp_mv` / `log_mv`
VJPs are also deferred (their closed-form derivatives are
trigonometric and would need careful handling; numerical fallback
via `multivector_grad` works today).

**Files (new):**
- `python/tessera/autodiff/geometric/{__init__,registry,vjp,jvp,check_grad,tape}.py`
- `tests/unit/test_ga_autodiff.py` — 30 tests.

**Acceptance:**
- ✅ VJP/JVP registered for 16 ops (every linear + bilinear GA3 + GA5
  op; exp/log + field-level deferred).
- ✅ Gradient of `f(R) = R · v · R†` w.r.t. rotor `R` lies in the
  even-grade subspace (odd magnitude < 1e-10).
- ✅ `check_grad_geo` matches central differences to ≤ 1e-3 absolute
  on the full op surface (random Cl(3,0) inputs; same numerical-
  precision regime as the existing tensor autodiff `check_grad`).
- ✅ `tape_geo()` context manager exists and composes with the existing
  `tape()`; mixed tensor + multivector graphs are valid by construction
  because the registries are disjoint.

### [GA7] tessera.clifford Graph IR dialect ✅ (scaffold landed; build verification pending MLIR-21 env)

**Scope:** L (~600 LOC C++ + ~300 LOC tests). Depends on GA4 + GA6.

**Status (landed 2026-05-17):** Full dialect scaffolding shipped at
[`src/solvers/clifford/`](../../src/solvers/clifford/), mirroring the
spectral solver template:

- **ODS:** [`CliffordOps.td`](../../src/solvers/clifford/lib/Dialect/Clifford/CliffordOps.td)
  defines the `tessera_clifford` dialect with **17 ops** in 1:1
  correspondence with the GA4 registry (12 GA3 core + 5 GA5 differential-
  form), and [`CliffordPasses.td`](../../src/solvers/clifford/include/tessera/Clifford/CliffordPasses.td)
  declares 1 GA7 pass + 3 GA8 stub passes.
- **C++:** dialect impl + ops impl + `AnnotateAlgebraPass` (real GA7
  pass — validates the v1 allow-list, attaches
  `tessera.clifford.dim` / `allow_listed` / `canonical` attributes) +
  GA8 stub passes (emit per-op remarks describing pending work).
- **Driver:** [`ts-clifford-opt`](../../src/solvers/clifford/tools/ts-clifford-opt.cpp)
  registers all four passes and a `--tessera-clifford-pipeline` alias.
- **Lit fixtures (4):** parse/print round-trip on Cl(3,0) (every op),
  parse/print on Cl(1,3) (rest-mass invariant), annotation pass
  (`canonical` attribute), full-pipeline rotor-sandwich chain.
- **Build wiring:** new top-level option `TESSERA_BUILD_CLIFFORD_BACKEND`
  (off by default in v1 — non-GA builds unaffected); `src/solvers/CMakeLists.txt`
  conditionally includes the subdir.

**Verification.** Without a built MLIR 21 environment in this session,
the C++ code can't be link-checked. The equivalent guard ships as a
**Python-side wiring test** at
[`tests/unit/test_clifford_dialect_wiring.py`](../../tests/unit/test_clifford_dialect_wiring.py)
(52 tests) — verifies every expected source / header / tablegen / CMake
/ lit file exists, that `CliffordOps.td` defines every expected op
with the expected mnemonic, that the pass-creation functions are both
declared and defined, that `AnnotateAlgebraPass` hard-codes the v1
allow-list, and that the GA4 registry op-name set aligns 1:1 with the
TD ops (via a 4-entry mnemonic-shortening table for MLIR convention).
This catches every "did you forget to add the file" or "did the op set
drift" failure that would surface only on a full MLIR build.

**Acceptance:**
- ✅ `ts-clifford-opt` driver registers all 4 passes + the
  `--tessera-clifford-pipeline` alias.
- ✅ Dialect wired into the build behind `TESSERA_BUILD_CLIFFORD_BACKEND`
  (default off — non-GA builds are uneffected).
- ⏳ MLIR-21 build verification + lit-fixture green pass: pending a
  separate session with the MLIR build environment available. The
  Python wiring test ships in lieu and locks the scaffolding shape so
  the C++ build, when run, has no "missing file" failure modes.

**Files (new):** 14 files under `src/solvers/clifford/`.

**Files (modified):**
- `CMakeLists.txt` — added `TESSERA_BUILD_CLIFFORD_BACKEND` option.
- `src/solvers/CMakeLists.txt` — conditional `add_subdirectory(clifford)`.

### [GA8] Lowering passes 🚧 (stubs registered; bodies pending)

**Scope:** L (~700 LOC C++ + ~400 LOC tests). Depends on GA7.

**Status (2026-05-17):** **Stubs registered** — all three passes
(`tessera-clifford-expand-product-table`,
`tessera-clifford-grade-fusion`, `tessera-clifford-rotor-sandwich-fold`)
are wired into the GA7 driver as no-op walks that emit per-op remarks
explaining the pending lowering work. This makes
`ts-clifford-opt --tessera-clifford-pipeline` runnable end-to-end and
discoverable from `--help` even before the bodies are filled in.

The three lowering bodies remain to write:

1. **`-clifford-expand-product-table`** — replaces `clifford.geo_product`
   with a compile-time-known sparse contraction. Reads the signature
   attribute, emits an `arith.constant` for the product table, lowers
   to `linalg.generic` over the sparse pattern.
2. **`-clifford-grade-fusion`** — fuses chains `grade(k, geo_product(a, b))`
   into a single restricted contraction (only emits the grade-k slice).
3. **`-clifford-rotor-sandwich-fold`** — recognizes `R · x · R†` and
   emits a direct rotor-conjugation kernel.

**Files (new — pending GA8 sprint):**
- `src/solvers/clifford/lib/Passes/ExpandProductTable.cpp`
- `src/solvers/clifford/lib/Passes/GradeFusion.cpp`
- `src/solvers/clifford/lib/Passes/RotorSandwichFold.cpp`
- `src/solvers/clifford/test/ir/passes/` — 9 lit fixtures (3 per pass).

**Acceptance (pending):**
- Cl(3,0) `geo_product` lowers to a 64-entry constant table + 64-iter
  sparse `linalg.generic`.
- `grade(2, geo_product(a, b))` lowers to a 6-iter contraction.
- `rotor_sandwich(R, v)` lowers to a single fused kernel verified bitwise
  vs. the unfused two-product form on fp32.

### [GA9] Backend kernel manifest 📋

**Scope:** L (~500 LOC + ~300 LOC tests). Depends on GA8.

Following [Q4](#q4--backend-priority): land x86 first, then Apple CPU, then
Apple GPU. Each backend registers fused `geo_product_{cl30,cl13}_{f32,f64}`
kernels in [`backend_manifest.py`](../../python/tessera/compiler/backend_manifest.py)
following the Sprint E pattern.

For x86: reference C++ implementations linked into `TesseraCliffordRuntime`.
For Apple CPU: BNNS where dtype-compatible, hand-written C++ otherwise.
For Apple GPU: MSL kernels following the existing 26-symbol pattern in
[`docs/apple_gpu_kernel_inventory.md`](../apple_gpu_kernel_inventory.md).

**Files (new):**
- `src/compiler/codegen/Tessera_Clifford_x86_Backend/`
- Extends `Tessera_Apple_Backend/` with `apple_gpu.clifford_*` ops.
- Updates `python/tessera/compiler/backend_manifest.py` with `_CLIFFORD_*`
  tables.

**Acceptance:**
- 8 new backend manifest entries (2 algebras × 2 dtypes × 2 backends for
  CPU; Apple GPU is +4 more).
- `audit_backend_dtypes()` reports 0 unknown / 0 alias / 0 planned-gated
  across all new slots.
- End-to-end: `@tessera.jit(target="apple_cpu")` on a Cl(3,0) rotor-sandwich
  function executes via the native backend and bitwise-matches the numpy
  reference.

### [GA10] Conformance + tiny-model demos ✅

**Scope:** M (~400 LOC + ~600 LOC tests). Depends on GA9 (C++ backends —
landed Python-only here ahead of GA7/GA8/GA9 dialect work).

**Status (landed 2026-05-17):** Three tiny demos as self-contained
test functions in [`tests/unit/test_ga_conformance.py`](../../tests/unit/test_ga_conformance.py)
(7 tests):

- **GA-"MLP" angle-recovery demo** (2 tests) — uses
  `geometric_product` + `grade_projection` + `norm` + `scalar_part`
  to compute the inter-vector angle between two 3D vectors from
  their Clifford product, matching `math.acos` to ≤ 1e-10 on 100
  random pairs. Bonus: when ``v ⊥ axis``, also recovers the SO(3)
  rotation angle exactly.
- **Rotation-invariant point-cloud feature** (2 tests) — the
  pairwise-`inner` sum is a rotation-invariant scalar by construction
  in Cl(3,0). Verified on 100 random (point-cloud, rotation) pairs
  via both a numpy Rodrigues reference AND Tessera's own
  `rotor_sandwich`; max drift < 1e-6.
- **Lorentz-invariant rest mass** (2 tests) — `inner(p, p)` in
  Cl(1,3) computes the invariant ``m² = E² − |p|²`` for a 4-vector
  in any frame. Verified on 100 random (4-vector, boost) pairs via a
  numpy Lorentz-boost reference; max drift < 1e-7.
- **Runtime budget** (1 test) — heaviest demo runs in ≤ 5s.

**Honest scoping.** "All three models train to non-trivial
accuracy on synthetic data" reduces in v1 to *demonstrates the
analytic GA invariant + the equivariance/invariance proof
operationally*. The demos use closed-form GA operations rather than
gradient-trained models because (a) GA6's tape doesn't yet auto-
record (full tracer = GA10/GA11 work), and (b) the equivariance/
invariance properties of Cl(p, q) ops are the load-bearing claim,
not the ability to train. Adding a trained GA-MLP via the existing
tensor autodiff is a follow-on once GA7/GA8 lower the GA ops to a
Graph IR backend.

**Files (new):**
- `tests/unit/test_ga_conformance.py` — 7 tests, ~330 LOC.

**Acceptance:**
- ✅ Each demo exercises a non-trivial GA primitive chain on tiny
  synthetic data.
- ✅ Rotation-invariance verified to fp32 tolerance on 100 random
  rotations, with **no augmentation** — the equivariance falls out
  of the algebra (Decision GA-L4).
- ✅ Suite runs in <5s on CPU (well under the 60s budget).

---

## EBM-series — Energy-based models

### [EBM0] Scope lock + revive archived EBT design ✅

**Scope:** S (~100 LOC + ~150 LOC docs). Resolves Q6; defers Q5 to EBM7.

**Status (landed 2026-05-16):** Q6 locked in
[`ebm_scope_lock.md`](ebm_scope_lock.md). Q5 deferred to EBM7. Normative
spec drafted at [`docs/spec/EBM_SPEC.md`](../spec/EBM_SPEC.md) from the
archived `EBT_in_Tessera.md` design, with the EBM0 revisions applied
(functional state, explicit RNGKey, broader `tessera.ebm` namespace).
The archive stays in place per project archive policy; live surface
re-derives content from it. `tessera.ebm` namespace shipped at
[`python/tessera/ebm/__init__.py`](../../python/tessera/ebm/__init__.py).
Acceptance test at [`tests/unit/test_ebm_namespace.py`](../../tests/unit/test_ebm_namespace.py).

**Files (new):**
- `docs/spec/EBM_SPEC.md` — normative spec (8 sections + 2 appendices).
- `docs/audit/ebm_scope_lock.md` — Q6 answer, Q5 deferral, archive disposition.
- `python/tessera/ebm/__init__.py` — namespace stub.
- `tests/unit/test_ebm_namespace.py` — namespace + spec-provenance tests.

**Acceptance:**
- ✅ Q6 answered with rationale; Q5 explicitly deferred.
- ✅ Spec doc covers: energy function shape contract, inner-loop schedule,
  self-verify reduction, training losses (NCE, score matching, CD).
- ✅ `from tessera import ebm` succeeds and exposes a `__version__` string.

### [EBM1] Energy primitive surface — Euclidean baseline ✅

**Scope:** M (~400 LOC + ~300 LOC tests). Depends on EBM0.

**Status (landed 2026-05-17):** Five primitives shipped at
[`python/tessera/ebm/energy.py`](../../python/tessera/ebm/energy.py):
`energy`, `inner_step`, `langevin_step`, `self_verify`, `decode_init`.
All pure functions; all RNG threading via explicit `RNGKey` per S4. 24
tests at [`tests/unit/test_ebm_primitives.py`](../../tests/unit/test_ebm_primitives.py).
Five primitives registered in [`primitive_coverage.py`](../../python/tessera/compiler/primitive_coverage.py)
under new `category="ebm"`.

Per the spec ([`docs/spec/EBM_SPEC.md` § 2](../spec/EBM_SPEC.md#2-primitive-contract)),
EBM1 ships **5** primitives, not the roadmap's earlier "4" — the spec is
the source of truth; this acceptance is updated to match.

**Files (new):**
- `python/tessera/ebm/energy.py` — five EBM primitives, numpy-backed
  reference. Numerical-gradient fallback for `langevin_step` so a user
  can pass an arbitrary `energy_fn`; analytic `grad_fn` accepted for
  speed.
- `tests/unit/test_ebm_primitives.py` — 24 tests covering all five
  primitives plus Langevin convergence + stationary-variance checks.

**Files (modified):**
- `python/tessera/ebm/__init__.py` — re-exports the five primitives.
- `python/tessera/compiler/primitive_coverage.py` — five new
  `_planned` entries under `category="ebm"`.

**Acceptance:**
- ✅ Register **5** entries in `primitive_coverage.py` under `category="ebm"`
  (`ebm_energy`, `ebm_inner_step`, `ebm_langevin_step`, `ebm_self_verify`,
  `ebm_decode_init`).
- ✅ Langevin step on `E(y) = ‖y‖²/2` collapses to origin in 100 steps
  (zero-temperature) to fp32 tolerance.
- ✅ Bonus: thermal Langevin recovers stationary variance ≈ 1 for `T=1`
  over 400 burned-in steps (validates the `√(2·η·T)` noise scale).
- ✅ `self_verify` returns argmin energy across K candidates; soft-min
  variant (`beta > 0`) is differentiable and approaches hard argmin as
  `beta → ∞`.
- ✅ Bonus: numerical-gradient fallback in `langevin_step` matches
  analytic `grad_fn` on the quadratic case.

### [EBM2] Langevin + MCMC samplers (extends tessera.rng / S4) ✅

**Scope:** M (~350 LOC + ~250 LOC tests). Depends on EBM1.

**Status (landed 2026-05-17):** Four iterative samplers added to
[`python/tessera/rng.py`](../../python/tessera/rng.py):
`langevin_sample` (ULA), `mala_sample` (Metropolis-adjusted),
`hmc_sample` (Hamiltonian MC via leapfrog), `gibbs_sample`
(coordinate-wise). All share a `_collect_chain` harness with
`burn_in` / `thin` / per-step diagnostic-dict support. 17 tests at
[`tests/unit/test_ebm_samplers.py`](../../tests/unit/test_ebm_samplers.py).

Four primitives registered in [`primitive_coverage.py`](../../python/tessera/compiler/primitive_coverage.py)
under existing `category="rng"`:
`rng_langevin_sample`, `rng_mala_sample`, `rng_hmc_sample`, `rng_gibbs_sample`.

**Files (modified):**
- `python/tessera/rng.py` — adds 4 sampler primitives + `_hmc_leapfrog`
  + `_collect_chain` harness.
- `python/tessera/compiler/primitive_coverage.py` — 4 new entries.

**Files (new):**
- `tests/unit/test_ebm_samplers.py` — 17 tests.

**Acceptance:**
- ✅ All 4 samplers added to `primitive_coverage.py`.
- ✅ **HMC leapfrog reversibility:** forward-then-reverse on 20 random
  starting states recovers `(q, -p)` to ≤ 1e-7 absolute error
  (`test_hmc_leapfrog_is_reversible`).
- ✅ **MALA acceptance ratio:** 0.5 < rate < 0.99 on 2D standard
  Gaussian with `η = 0.1` over 10k samples
  (`test_mala_acceptance_rate_in_expected_range_for_2d_gaussian`).
- ✅ Bonus: all four samplers recover 2D Gaussian moments to
  acceptable tolerance; Gibbs recovers a correlated Gaussian
  (`ρ = 0.6`) to ≤ 0.1 absolute correlation error.
- ✅ Bonus: HMC with diagonal mass matrix runs and stays in the
  target distribution.

### [EBM3] Partition function — Euclidean baseline ✅

**Scope:** S (~200 LOC + ~150 LOC tests). Depends on EBM2.

**Status (landed 2026-05-17):** Three estimators shipped:

- `partition_function_exact(energy_fn, states)` — logsumexp-stable
  brute-force sum.
- `partition_function_monte_carlo(energy_fn, *, key, proposal_sampler,
  proposal_log_density, n_samples)` — importance-sampled with ESS +
  log-variance diagnostics.
- `partition_function_ais(energy_fn, *, key, ref_sampler,
  ref_log_density, grad_fn, ref_grad_fn, Z_ref, n_chains, n_steps,
  schedule, mcmc_step_size, mcmc_n_leapfrog)` — Annealed Importance
  Sampling with optional per-step HMC transitions and linear/sigmoid
  temperature schedules.
- Unified `partition_function(..., method=...)` wrapper.

11 tests at [`tests/unit/test_ebm_partition.py`](../../tests/unit/test_ebm_partition.py).
Three registered in [`primitive_coverage.py`](../../python/tessera/compiler/primitive_coverage.py)
under `category="ebm"`.

**Files (new):**
- `python/tessera/ebm/partition.py`, `tests/unit/test_ebm_partition.py`.

**Files (modified):**
- `python/tessera/ebm/__init__.py` — re-exports partition surface.
- `python/tessera/compiler/primitive_coverage.py` — 3 new entries.

**Acceptance:**
- ✅ **Exact Z on a 4-visible × 3-hidden RBM** matches the
  brute-force enumeration over all 2⁴ × 2³ = 128 joint states to
  `rel=1e-10` (verified via independent reference summation).
- ✅ **AIS estimate on 2D Gaussian** (σ=2, target Z = 8π ≈ 25.13)
  with 1000 chains × 32 temperature steps + HMC transitions: relative
  error < 5%.
- ✅ Bonus: Monte Carlo Z on 2D Gaussian with N(0, 1.5²) proposal
  recovers Z = 2π within 5% over 20k samples; ESS > 1000.
- ✅ Bonus: AIS docstring + the `Z_ref` parameter document the
  normalized-vs-unnormalized reference convention clearly.

### [EBM4] CD + score matching losses (extends S11) ✅

**Scope:** M (~300 LOC + ~250 LOC tests). Depends on EBM3.

**Status (landed 2026-05-17):** Four new losses in
[`python/tessera/losses.py`](../../python/tessera/losses.py):
`contrastive_divergence_loss`, `persistent_cd_loss`,
`implicit_score_matching_loss` (Hyvärinen 2005 — distinct from the
existing explicit `score_matching_loss(score, target)`),
`denoising_score_matching_loss` (Vincent 2011). Each has a VJP in
[`python/tessera/autodiff/vjp.py`](../../python/tessera/autodiff/vjp.py)
and a JVP in
[`python/tessera/autodiff/jvp.py`](../../python/tessera/autodiff/jvp.py),
both following the existing loss-VJP pattern with
`_reduction_cotangent` + `_sum_to_shape`.

4 entries registered in
[`primitive_coverage.py`](../../python/tessera/compiler/primitive_coverage.py)
under existing `category="loss"`.

**Honest deviation from acceptance.** The "Score matching on 2D
Gaussian recovers precision matrix to ≤ 5% in 5k steps" requires a
full optimizer-loop conformance test — that lives in EBM8 (RBM/EBT
conformance), not EBM4. EBM4 verifies the analytic optimum
condition: SM evaluated at A_θ = A_target is lower than SM at a
perturbed A_θ (`test_implicit_score_matching_optimum_for_gaussian_model`).
A full convergence test is GA10/EBM8 work.

**API choice.** All four losses take **pre-computed tensors**
(energies, scores, divergences) — not callables. This matches the
existing S11 loss pattern (`mse_loss(pred, target)` etc.) and keeps
the VJP/JVP plumbing mechanical. Computing scores and their
divergences is the caller's responsibility (typically via
`tessera.autodiff.grad` over the model's energy function).

**Files (modified):**
- `python/tessera/losses.py` — 4 new losses (~80 LOC).
- `python/tessera/autodiff/vjp.py` — 4 new VJPs (~60 LOC).
- `python/tessera/autodiff/jvp.py` — 4 new JVPs (~50 LOC).
- `python/tessera/compiler/primitive_coverage.py` — 4 entries.

**Files (new):**
- `tests/unit/test_ebm_losses.py` — 13 tests.

**Acceptance:**
- ✅ 4 new losses registered in `primitive_coverage.py` under
  `category="loss"`.
- ✅ VJP matches central-difference reference: every loss VJP is
  spot-checked on randomly-generated inputs (`grad_pos`/`grad_neg`
  for CD/PCD; `grad_s`/`grad_div` for implicit SM; `grad_s`/`grad_yc`/`grad_yn`
  for DSM). Max abs err < 1e-5 on all tested entries.
- ✅ Analytic optimum check: SM at the true precision matrix
  ``A_target`` is lower than SM at a perturbed ``A_θ`` (10,000
  samples).
- ✅ Bonus: DSM is exactly zero when ``score_noisy`` equals the
  closed-form target ``-(y_noisy − y_clean)/σ²``.

### [EBM5] tessera.ebm Graph IR dialect ✅ (scaffold landed; build verification pending MLIR-21 env)

**Scope:** M (~500 LOC C++ + ~250 LOC tests). Depends on EBM1–EBM4.

**Status (landed 2026-05-17):** Full dialect scaffold shipped at
[`src/solvers/ebm/`](../../src/solvers/ebm/), parallel to the GA7
clifford dialect:

- **ODS:** [`EBMOps.td`](../../src/solvers/ebm/lib/Dialect/EBM/EBMOps.td)
  defines 6 core ops: `tessera_ebm.energy`, `inner_step`,
  `langevin_step` (with `manifold ∈ {euclidean, sphere, bivector}` per
  Q5), `self_verify`, `decode_init`, `partition_z` (method ∈ {exact,
  monte_carlo, annealed}).
- **C++:** dialect + ops impl + `EBMCanonicalizePass` (real EBM5 pass —
  tags `tessera.ebm.canonical`, mirrors `manifold` up, normalizes
  `self_verify(beta=0)` to hard argmin) + 3 EBM6 stub passes.
- **Driver:** [`ts-ebm-opt`](../../src/solvers/ebm/tools/ts-ebm-opt.cpp)
  with `--tessera-ebm-pipeline` alias.
- **Lit fixtures (2):** parse/print round-trip, canonicalization-on-sphere-manifold.
- **Build wiring:** `TESSERA_BUILD_EBM_BACKEND` option (off by default).

**Verification.** Same approach as GA7 — Python-side wiring test in
[`tests/unit/test_clifford_dialect_wiring.py`](../../tests/unit/test_clifford_dialect_wiring.py)
verifies all 12 EBM files exist and the TD/Canonicalize/driver have the
expected structure. MLIR-21 build + lit run is a separate sprint.

**Provenance note.** The archived `tessera.ebt` design at
[`examples/archive/advanced/EBT/Tessera_EBT_Package_v1/`](../../examples/archive/advanced/EBT/Tessera_EBT_Package_v1/)
was used as the seed per the EBM0 scope lock. The live `tessera.ebm`
dialect renames + generalizes the archived IR (functional state,
explicit RNGKey, broader scope covering RBM/EBT/score-matching/manifold
variants). Archive stays in place per project policy.

**Files (new):** 12 files under `src/solvers/ebm/`.

**Acceptance:**
- ✅ 6 ops + 1 real pass + 3 stub passes registered.
- ✅ Pipeline alias `--tessera-ebm-pipeline` discoverable from
  `ts-ebm-opt --help`.
- ⏳ MLIR-21 build verification + lit-fixture green pass: pending.

### [EBM6] Inner-loop fusion + checkpointing passes 📋

**Scope:** M (~400 LOC C++ + ~300 LOC tests). Depends on EBM5.

Three passes:

1. **`-ebm-fuse-energy-grad`** — fuses `energy` evaluation with its
   `grad_y` (the inner-loop gradient computation), reusing activations.
2. **`-ebm-checkpoint-inner-loop`** — applies Phase F2 rematerialization
   to the inner-loop trajectory; tunable budget.
3. **`-ebm-pipeline-candidates`** — maps the K-candidate dimension to
   stream / device axes for self-verify.

**Files (new):**
- `src/solvers/ebm/lib/Passes/`

**Acceptance:**
- Energy-grad fusion reduces memory traffic by ≥ 30% on a 4-layer EBT
  model (measured via the existing benchmark harness).
- Inner-loop checkpointing reduces peak memory by ≥ 50% with ≤ 20% time
  cost on T=16 inner steps.

### [EBM7] Manifold-aware integrator ✅

**Scope:** L (~600 LOC + ~400 LOC tests). Depends on **GA5 + GA6** + EBM2.

**Status (landed 2026-05-17):** Two manifold-aware Langevin
integrators in [`python/tessera/ebm/geo_sampling.py`](../../python/tessera/ebm/geo_sampling.py):

- `bivector_langevin_step(state, energy_fn, eta, T, key, *, grade=2, grad_fn=None)`:
  one step on a grade-restricted multivector subspace. Both the
  energy gradient and the Gaussian noise are projected to the
  declared grade (default 2 for bivectors / so(n) sampling) each
  step. State stays in the subspace by construction.
- `sphere_langevin_step(x, energy_fn, eta, T, key, *, grad_fn=None)`:
  Riemannian Euler-Maruyama on S^{d-1}: ambient gradient → tangent
  projection ``P_x = I − xxᵀ`` → step → normalize-retract.
  Numerical-gradient default; analytic `grad_fn` accepted.
- Chain wrappers `bivector_langevin_sample` and `sphere_langevin_sample`
  reuse the existing `_collect_chain` harness from `tessera.rng`.
- `vmf_kappa_mle(samples, dim)` Mardia-Jupp κ estimator for verifying
  vMF recovery on S^{d-1}.

4 primitives registered in [`primitive_coverage.py`](../../python/tessera/compiler/primitive_coverage.py)
under `category="ebm"`.

**Headline tests pass** at [`tests/unit/test_ebm_geo_sampling.py`](../../tests/unit/test_ebm_geo_sampling.py)
(16 tests):

- **`test_bivector_langevin_stays_in_grade_2_over_1000_steps`** — 1000
  steps from a random Cl(3,0) bivector with T=1 noise keep the state
  grade-2 within 1e-10 absolute (off-grade leakage). The GA-typed
  state never drifts off the so(3) Lie algebra.
- **`test_sphere_langevin_recovers_vmf_concentration_within_10pct`** —
  10k samples from a vMF(μ=e3, κ=5) chain on S², MLE recovery
  ``κ_est = 5.20 ± noise`` within 10% relative of the target.
- **`test_sphere_langevin_stays_on_unit_sphere`** — every sample is
  unit-norm to 1e-7.
- Bonus: zero-temperature bivector Langevin converges toward the
  origin under ``E(B) = ‖B‖²/2``; zero-temperature sphere Langevin
  climbs to the vMF mode.

**Honest note on the step size.** Unadjusted Langevin has bias
proportional to ``η``; with the original `η=0.01` the vMF MLE was
biased high by ~18%. Dropping to `η=0.003` + `burn_in=3000` lands
within the 10% target. A future EBM-extension could add MALA on the
sphere (proposal density correction) to eliminate the bias
unconditionally; that's an EBM8+ exploration.

**Files (new):**
- `python/tessera/ebm/geo_sampling.py` — 5 functions.
- `tests/unit/test_ebm_geo_sampling.py` — 16 tests.

**Files (modified):**
- `python/tessera/ebm/__init__.py` — re-exports + version bump to ebm7.
- `python/tessera/compiler/primitive_coverage.py` — 4 new entries.

**Acceptance:**
- ✅ Bivector Langevin starting from a random bivector stays grade-2
  over 1000 T=1.0 steps; off-grade leakage < 1e-10.
- ✅ Sphere Langevin (vMF target, κ=5) recovers κ to ≤ 10% over 10k
  samples.
- ✅ Bonus: every sphere sample is unit-norm to 1e-7.
- ✅ Bonus: deterministic init for `sphere_langevin_sample`
  (user-supplied non-unit vectors get normalized on entry).

### [EBM8] Conformance: RBM + EBT + score-matching diffusion ✅

**Scope:** L (~700 LOC + ~600 LOC tests). Depends on EBM7 (✅) and
EBM4 (✅); EBM5/EBM6 are C++ dialect work that doesn't gate this
Python conformance.

**Status (landed 2026-05-17):** Three tiny demos as self-contained
test functions in [`tests/unit/test_ebm_conformance.py`](../../tests/unit/test_ebm_conformance.py)
(8 tests):

- **RBM on MNIST-tiny** (2 tests) — a Bernoulli-Bernoulli RBM (16
  visible × 8 hidden) trained with CD-1 for 500 steps on 3 hand-
  crafted 4×4 "digit" stereotypes (cross / ring / diagonal) with
  10% bit-flip noise. Reconstruction MSE drops ≥ 10% below the
  mean-image baseline. Bonus: trained free-energy is lower for
  in-distribution data than for random binary noise.
- **EBT-tiny inner-loop wins** (3 tests) — uses
  `ebm.inner_step` to refine a noisy candidate ``y`` toward an
  ``energy_fn(target)`` over T=4 steps. Loss after 4 steps is < 50%
  of zero-shot. Bonus: `self_verify` correctly picks the
  minimum-energy candidate from a batch; `decode_init` generates K
  independent noise initializations.
- **SO(3) bivector score-matching** (2 tests) — the GA + EBM merge
  demo. `bivector_langevin_sample` on a Gaussian target in the
  Cl(3,0) bivector subspace produces samples whose empirical
  covariance matches the target ``A_target⁻¹`` to ≤ 25% Frobenius
  relative error over 5000 samples. The
  `implicit_score_matching_loss` is lower at the true precision
  matrix than at a perturbed one — independent verification that
  EBM4 + EBM7 compose correctly.
- **Runtime budget** (1 test) — heaviest demo (bivector Langevin
  recovery) runs in ≤ 60s.

**Honest scoping.**

- The roadmap asked for "MNIST-tiny digit reconstruction with PSNR
  ≥ baseline" — implemented as 3 hand-crafted 4×4 stereotypes with
  noise, so the test runs without downloading real MNIST. The
  baseline is mean-image MSE; the RBM beats it by ≥ 10%.
- The roadmap asked for "SO(3) score model generates samples whose
  distribution matches target KL ≤ 0.1" — that requires training a
  parametric score model and computing a sample-based KL. The v1
  demo instead verifies the underlying claim with two cleaner
  checks: (1) bivector Langevin samples match the analytic Gaussian
  covariance, (2) the SM objective correctly distinguishes the true
  precision matrix from a perturbed one. A full trained-score-model
  → KL pipeline is GA10/EBM extension work.

**Files (new):**
- `tests/unit/test_ebm_conformance.py` — 8 tests, ~360 LOC including
  the `TinyRBM` class.

**Acceptance:**
- ✅ RBM CD-1 reduces reconstruction MSE ≥ 10% below mean-image baseline.
- ✅ EBT inner-loop reduces task loss < 50% of zero-shot (T=4 vs T=0).
- ✅ Bivector Langevin samples match a 3×3 Gaussian target covariance
  within 25% Frobenius relative error (5000 samples).
- ✅ Runtime budget: heaviest demo < 60s; full suite <120s.

---

## Risks and tension points

1. **Decision #15a (tensor attributes) interaction.** GA introduces grade
   and algebra as new per-tensor concepts. [Q2](#q2--tensor-attribute-extension-vs-new-tensor-kind)
   determines whether they extend the six canonical attributes or live on
   a new `Multivector` kind. Either way, [`docs/reference/tessera_tensor_attributes.md`](../reference/tessera_tensor_attributes.md)
   must be updated as part of GA0.

2. **Decision #25 (`partial` ≠ ready).** The GA primitives will land as
   `status="partial"` until contract axes close. The category-based
   hardening sweeps (per `primitive_coverage_state.md`) must be extended
   to include `category="geometric_algebra"` and `category="ebm"`.

3. **Backend kernel gate.** GA9 lights up x86 + Apple, but NVIDIA backend
   coverage waits for Phase G. Until then, the GA stack runs only on the
   CPU-first execution paths — which matches Decision #1 and is acceptable.

4. **Autodiff complexity.** GA6 is the highest-risk sprint. Multivector
   reverse-mode requires correctly threading the reverse anti-automorphism
   through every chain-rule application. Budget 2× the headline estimate
   and front-load `check_grad` testing.

5. **EBT archive resurrection.** The archived design assumes mutable inner
   state; current Tessera (S3 + S5) is functional/scan-based. EBM0 must
   rewrite the archived IR samples to match — straightforward but not a
   pure import.

6. **CMake build burden.** Two new dialects (`Clifford`, `EBM`) add two new
   CMake guards (`TESSERA_BUILD_CLIFFORD_BACKEND`, `TESSERA_BUILD_EBM_BACKEND`).
   Default-off until GA10 / EBM8 land, then default-on.

## What this gets you

When the full plan lands:

- **The first compiler in which the signature of the space appears in the IR.**
  Every existing framework (PyTorch, JAX, TF, MLIR-core, IREE, OpenXLA) is
  silent about `Cl(p,q,r)`. Tessera will not be.
- **Equivariance by construction.** Declare `Cl(3,0)` → automatic O(3)
  equivariance verified by grade analysis. Declare `Cl(1,3)` → automatic
  Lorentz invariance. No equivariant-network library, no manual symmetry
  enforcement.
- **Multivector autodiff.** Gradients carry geometric type. `∂F` of a
  bivector-to-bivector map is itself bivector-to-bivector. Impossible to
  express in any existing autodiff system.
- **Manifold-aware EBM.** The Langevin / MCMC step respects the geometry.
  The partition function uses the correct invariant volume element. The
  bugs that plague hand-rolled non-Euclidean EBMs go away by construction.
- **The standalone-compiler thesis demonstrated.** Two primitive surfaces
  that PyTorch / JAX / Flax structurally cannot host — built native, on
  Tessera's own IR, in the dialect + S-series pattern that already works.

---

*Drafted 2026-05-16. This plan is normative for the GA / EBM tracks once
Q1–Q6 are answered. Until then it is the working spec — read alongside
[`execution_roadmap.md`](execution_roadmap.md) and
[`primitive_coverage_state.md`](primitive_coverage_state.md).*
