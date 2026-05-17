---
status: Normative (decision record)
classification: Audit / Scope lock
authority: Locks the architectural forks for the EBM-series (Energy-Based Models) primitive surface
last_updated: 2026-05-16
---

# EBM-series Scope Lock — Energy-Based Model Decisions

This doc captures the answer to the EBM-series fork raised in
[`ga_ebm_roadmap.md`](ga_ebm_roadmap.md), and identifies one remaining
deferred question (Q5 — manifold set for EBM7).

## Q6 — Archived EBT package: **revive as seed for EBM5**

Locked. [`examples/archive/advanced/EBT/Tessera_EBT_Package_v1/`](../../examples/archive/advanced/EBT/Tessera_EBT_Package_v1/)
is the design seed for the live EBM track:

- `docs/EBT_in_Tessera.md` → adapted into [`docs/spec/EBM_SPEC.md`](../spec/EBM_SPEC.md)
  (live, normative). Vocabulary updated to current Tessera conventions:
  `tessera.ebt` → `tessera.ebm` (broader scope), mutable inner state →
  functional `tessera.control.scan` (S5), opaque runner config → typed
  `EBMStepConfig` dataclass.
- `models/ebt/ir/ebt_ir_samples.mlir` → reference for EBM5 lit fixtures.
  IR ops keep their archived shapes (`energy`, `inner_step`, `self_verify`,
  `decode_init`, `grad_y`) with the namespace rename.
- `models/ebt/passes/` → reference for EBM6 pass-pipeline structure
  (`-ebm-canonicalize`, `-ebm-lower`).
- `models/ebt/runtime/ebt_runner.{h,cc}` → reference for EBM5 runtime
  shim signatures.

**Rationale.** Engineering economy. The archived IR vocabulary is sound,
the operational shape (encode → init K candidates → T inner-loop steps →
self-verify) is exactly what the live surface needs, and the algorithm-
level decisions (NCE / score-matching / contrastive divergence at train
time; gradient-step-with-optional-noise at inference) are well-grounded
in the EBT paper.

What gets rewritten on revival:

- **Functional state.** The archive's `EBTStepConfig` + mutable runner is
  out; EBM1 uses a pure `(state, RNGKey) -> (state', RNGKey')` shape per
  S4 / S5 conventions. The inner loop becomes `tessera.control.scan`
  rather than a hand-rolled `scf.for` runner.
- **RNG.** Archived runner takes a `noise` float; EBM1 takes an
  explicit `RNGKey` (S4) so determinism + replay work.
- **Namespace.** `tessera.ebt` is too narrow. `tessera.ebm` covers EBT,
  RBMs, score-matching diffusion, and the geometric-Langevin demo (EBM7)
  under one dialect.
- **Energy head shape.** Archive sketched bilinear + MLP; EBM1 ships
  the MLP form (energy is a callable closure, not a fixed op) so users
  can plug in any differentiable scalar function.

What stays:

- IR-op set (`energy`, `inner_step`, `self_verify`, `decode_init`, `grad_y`).
- Pass-pipeline structure (canonicalize → lower).
- K-candidates-across-streams parallelization policy.
- Self-verify reduction (`argmin` over K final energies).

## Q5 — Manifold set for EBM7: **S^n + SO(n) + ℝ^n** ✅ LOCKED (2026-05-17)

Locked options for v1 manifold-aware integrators (EBM7):

- **S^n** — the unit n-sphere in ambient ℝ^(n+1). Riemannian
  Langevin via ambient-gradient → tangent-plane projection
  (``P_x = I − x xᵀ``) → normalization retraction.
- **SO(n)** — the rotation group, represented intrinsically via its
  Lie algebra of bivectors in ``Cl(p, 0)``. State is a grade-2
  multivector; gradient and noise are projected to the bivector
  subspace by `grade_projection`. The acceptance test uses Cl(3,0)
  bivectors (so(3)).
- **ℝ^n** — flat Euclidean space. Falls back to the existing
  `tessera.rng.langevin_sample` / `mala_sample` from EBM2 without
  manifold-aware machinery.

**Rationale.** S^n unlocks orientation diffusion (the EBM8 SO(3)
score-matching demo). SO(n) via bivectors closes the GA + EBM merge
point cleanly — bivector state IS the Lie algebra, no chart-mapping
needed. ℝ^n is free (already shipped in EBM2).

**Deferred (v1 → vN, no breakage):**

- **SE(n)** (rigid-body motion) — needs the conformal algebra Cl(4,1)
  from Q1's extension path. Lit-clean v2.
- **SU(n)** (special unitary) — useful for complex/quantum-flavored
  models. Different algebra family (complex multivectors); v2+.
- **Hyperbolic H^n** — Lorentz-signature trick lifts to Cl(1, n);
  doable in v2 once Q1 lifts to general (p, q).

Extension path (post-v1, no breakage): each new manifold is a new
concrete class deriving from `tessera.ga.manifold.Manifold` plus a
new `manifold_langevin_step(manifold, ...)` dispatch branch.

EBM7 is now unblocked.

## What's locked, what's open

| Question | Status | Resolved at |
|---|---|---|
| Q5 — manifold set | ✅ S^n + SO(n) + ℝ^n (2026-05-17) | EBM7 (this sprint) |
| Q6 — archive disposition | ✅ revive as EBM5 seed | EBM0 |


EBM1 (energy primitive surface — Euclidean baseline) is now unblocked.

## Archive disposition (operational)

The archive is NOT moved or deleted in this sprint. It stays at
`examples/archive/advanced/EBT/Tessera_EBT_Package_v1/` per the
project's archive policy (Architecture: archive is excluded from build,
not deleted). The live surface re-derives content from it:

- `docs/spec/EBM_SPEC.md` is the new normative spec, drafted from
  `docs/EBT_in_Tessera.md`.
- `python/tessera/ebm/__init__.py` is the new live namespace.
- Future EBM5 lit fixtures will be re-derived from
  `models/ebt/ir/ebt_ir_samples.mlir`, not copied.

This preserves provenance — anyone tracing the design back can see the
archive — without resurrecting it as a build target.
