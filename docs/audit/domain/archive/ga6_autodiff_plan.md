---
status: Planning (GA6 not yet started)
classification: Plan / autodiff
authority: Captures the failure modes + 2× budget rationale for the GA6 Clifford autodiff sprint
last_updated: 2026-05-17
---

# GA6 — Clifford autodiff planning + budget

> Companion to
> [`docs/audit/roadmap/ROADMAP_AUDIT.md`](sprint_plan_task4_and_crosscuts.md).
> GA6 is the **highest-risk sprint** on the GA roadmap.  This doc
> captures (a) why, (b) what infrastructure is in place to mitigate,
> (c) what we plan to ship when GA6 is scheduled.

## Why GA6 is high-risk

Multivector reverse-mode autodiff doesn't reuse the tensor-side
VJP table.  Three classes of failure mode are unique to Clifford:

### 1. Reverse anti-automorphism threading

The Clifford reverse `a†` (reversion) is an
**anti-automorphism**: `(ab)† = b† a†` (order reverses).  Concretely:

- **`geometric_product(a, b)` w.r.t. `a`**: the VJP is
  `out̄ · reverse(b)`, **not** `out̄ · b`.  Forgetting the reverse
  silently produces numerically-close-but-wrong gradients that pass
  loose finite-difference checks at low precision but break at
  high-precision verification.
- **`geometric_product(a, b)` w.r.t. `b`**: the VJP is
  `reverse(a) · out̄` (left-multiply, *then* reverse).
- **`rotor_sandwich(R, x) = R x R†`**: the rotor `R` appears on both
  sides, so the VJP w.r.t. `R` is `out̄ · (R x)† + (R x)† · reverse(out̄)`
  (up to sign — to be verified at implementation time).

### 2. Grade-dependent sign rules

The Hodge star `⋆` is a grade-dependent linear involution: `⋆⋆ω =
(±1) ω` where the sign depends on signature parity and grade.
Cl(3,0) gives a clean `⋆⋆ω = ω` (everything's +1).  Cl(1,3) has
sign flips per grade.

- **`hodge_star` VJP**: the dual map is `⋆` again (up to the
  same sign).  Easy to get wrong by losing a factor of `(-1)^k`.
- **`grade_projection(a, grades)` VJP**: zero-pads non-selected
  grades.  Linear, but the sparsity pattern interacts with the
  cotangent: only the selected grades carry gradient back.

### 3. Field-op finite-difference boundaries

`ext_deriv` / `vec_deriv` / `codiff` use central differences on
the interior, zero at the boundary.  Their VJPs need to **respect
the same boundary policy** — the cotangent on the input is
boundary-zero too.  Naively transposing the central-difference
stencil produces a non-zero cotangent at the boundary cells, which
finite-difference verification will catch as an off-by-one.

## Mitigation: front-load test infrastructure

The harness lands in this sprint (Decision: budget 2× headline,
front-load `check_grad`).  Per the sprint plan
[`docs/audit/roadmap/ROADMAP_AUDIT.md`](sprint_plan_task4_and_crosscuts.md):

- [`python/tessera/ga/check_grad.py`](../../python/tessera/ga/check_grad.py)
  ships `multivector_check_grad` + `multivector_check_grad_scalar`.
- They finite-difference the forward op against a candidate VJP and
  return `(ok, max_rel_err)` for `assert err < tol` tests.
- The harness is proven against `norm_squared` (closed-form VJP is
  `2 * a`, easy) — see
  [`tests/unit/test_ga_check_grad.py`](../../tests/unit/test_ga_check_grad.py).
  When GA6 lands, every new VJP gets a one-liner harness test.

The verifier is fp64 by default (eps=1e-3, rtol=5e-4) — accurate
enough to catch reverse-anti-automorphism sign errors that
fp32-tolerance checks would miss.

## 2× budget rationale

Headline estimate for "implement VJPs for the 17 GA primitives" =
**1 sprint.**  Realistic budget = **2 sprints**, split as:

- **Sprint 1** — the easy half: `norm`, `norm_squared`, `inner`,
  `reverse`, `grade_involution`, `conjugate`, `grade_projection`,
  `wedge`, `left_contraction`.  All linear or quadratic in the
  input; finite-difference verification catches sign errors fast.
- **Sprint 2** — the hard half: `geometric_product` (with the
  reverse-anti-automorphism dual), `rotor_sandwich` (rotor on both
  sides), `exp_mv` / `log_mv` (closed-form rotors plus power-series
  fallback ⇒ VJP has two branches), `hodge_star` (grade-dependent
  sign), `ext_deriv` / `vec_deriv` / `codiff` (boundary policy), and
  `integral` (manifold-weighted reduction).

Both sprints front-load `check_grad` tests so failures surface at
the harness level, not weeks later in a downstream training-loop
debug session.

## What ships in this prep sprint

This sprint (the Task 4 / cross-cuts sprint) ships **infrastructure
only** — no VJPs:

1. `multivector_check_grad` + `multivector_check_grad_scalar` —
   the verification harness.
2. A starter unit test (`test_ga_check_grad.py`) that proves the
   harness against `norm_squared`'s known analytic VJP.
3. This planning doc — captures the failure-mode taxonomy + the
   2× budget so future GA6 sprints don't surprise.

## Out of scope (deferred to GA6 proper)

- The actual VJP / JVP table for the 17 GA primitives.
- Registry promotion (`vjp` / `jvp` axes flip from `planned` to
  `complete` only once verified VJPs land).
- Differentiable `rotor_sandwich` integration with `tessera.autodiff`
  tape-mode autodiff.
- MultivectorField VJPs for the four field ops.

## Pointers

- [`tessera.ga.check_grad`](../../python/tessera/ga/check_grad.py) —
  the harness module.
- [`tessera.ga.signature`](../../python/tessera/ga/signature.py) —
  `Cl(p, q, r)` + the reverse / grade-involution sign tables that
  every GA VJP will reference.
- [`docs/spec/AUTODIFF_SPEC.md`](../spec/AUTODIFF_SPEC.md) — the
  tensor-side autodiff spec the GA VJPs need to remain consistent
  with.
- [`docs/audit/roadmap/ROADMAP_AUDIT.md`](sprint_plan_task4_and_crosscuts.md) —
  this sprint's full work plan.
