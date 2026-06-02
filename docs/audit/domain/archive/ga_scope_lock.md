---
status: Normative (decision record)
classification: Audit / Scope lock
authority: Locks the architectural forks for the GA-series (Geometric Algebra / Clifford) primitive surface
last_updated: 2026-05-16
---

# GA-series Scope Lock — Clifford Algebra Decisions

This doc captures the answers to the four GA-series architectural forks
raised in [`ga_ebm_roadmap.md`](ga_ebm_roadmap.md). It is the gate that
opens GA1 (algebra signature object).

Three forks are answered here. Two more (Q3 autodiff strategy, Q5 manifold
set) are deferred — they don't bite until GA6 / EBM7 respectively, and the
answers will benefit from what we learn in GA1–GA4.

## Q1 — Signature support matrix for v1: **Cl(3,0) + Cl(1,3) only**

Locked options:
- `Cl(3,0,0)` — 3D Euclidean. 8 basis elements. Target use cases:
  robotics, computer vision, molecular geometry, equivariant point clouds.
- `Cl(1,3,0)` — Minkowski spacetime. 16 basis elements. Target use cases:
  particle-physics ML, relativistic invariance demos.

**Rationale.** Both algebras have ≤ 16 basis elements; full product tables
fit in registers and can be inlined as compile-time constants. This is the
minimal set that proves the two contested claims (Euclidean equivariance
from algebra; non-Euclidean invariance from signature). Conformal Cl(4,1)
and fully general `Cl(p,q,r)` are excluded from v1 but will not require
breaking changes:

- The signature object (GA1) is parameterized by `(p, q, r)` from day one —
  v1 simply restricts `p+q+r ≤ 4` and `r == 0`.
- The product table builder is general; v1 only exercises two
  instantiations.
- All dialect attrs (GA7) carry `(p, q, r)` triples; v1 lit fixtures only
  cover the two locked signatures.

Extension path (post-v1, no breakage):
1. Lift the `(p+q+r ≤ 4, r == 0)` guard to `≤ 6` and allow `r > 0`.
2. Add Cl(4,1) backend tables (GA9 follow-up).
3. Add lit fixtures for the new signatures.

## Q2 — Type system: **`tessera.Multivector` as a sibling to `Tensor`**

Locked: introduce a new tensor kind `Multivector` parallel to the existing
`Tensor`. Both kinds share the underlying shape / dtype / layout / device /
distribution machinery from [`docs/reference/tessera_tensor_attributes.md`](../reference/tessera_tensor_attributes.md);
`Multivector` adds two kind-specific fields:

- `algebra: Cl(p, q, r)` — the signature (always present).
- `grades: frozenset[int]` — the subset of `{0..n}` actually populated.

**Rationale.** Decision #15a is one of the most load-bearing in the project
— it fixes the six canonical tensor attributes and the canonical dtype set,
and is the source of truth for promotion rules. Adding a 7th + 8th attribute
would touch every primitive's `OP_SPECS` entry, every backend manifest
record, and the `audit_canonical_dtypes()` walker. The sibling-kind
approach contains the blast radius to the new `tessera.ga` namespace
without disturbing the existing 374 primitives.

When a multivector is grade-pure (e.g., a `Rotor` or a `DiffForm<k>`), the
type system treats `grades` as a static set that propagates through ops.
Mixed-grade multivectors carry the full set and lose grade-purity proof
obligations.

Promotion path (post-v1, no breakage):
- If the sibling-kind surface stabilizes and downstream passes routinely
  need to dispatch on `algebra`, promote `algebra` + `grades` to canonical
  attributes 7–8. The migration is mechanical because every site that
  reads them today already goes through `Multivector` accessors.

## Q4 — Backend priority: **x86 → Apple CPU → Apple GPU; NVIDIA after Phase G**

Locked order for GA9:

1. **x86 reference** (numpy-equivalent C++ in `TesseraCliffordRuntime`) —
   correctness baseline; every GA op has a reference implementation here
   before any optimized backend.
2. **Apple CPU** via Accelerate — small product tables map cleanly to
   `cblas_sgemm` for the matmul-flavored ops; hand-written C++ for the
   rest. Follows Phase 8.2 patterns.
3. **Apple GPU** via MSL — extends the existing 26-symbol MSL kernel
   inventory ([`docs/apple_gpu_kernel_inventory.md`](../apple_gpu_kernel_inventory.md))
   with `geo_product_{cl30,cl13}_{f32,f16,bf16}` + grade-fused variants.
   Proves the dialect → Target IR → MSL pipeline for the GA stack.
4. **NVIDIA** — deferred until Phase G (NVIDIA execution) lights up. GA's
   first NVIDIA kernel will land as part of the Phase G sweep, not as a
   GA-track sprint.

**Rationale.** Matches Decision #1 (CPU-first). Apple GPU is mature
(Phase 8.4.7) and gives us a real GPU path before NVIDIA is ready. NVIDIA
deferral is honest about the runway: GA can fully demonstrate its
architectural claims on CPU + Apple GPU; H100 throughput numbers are a
follow-on, not a prerequisite.

## Q3 — Autodiff strategy: **parallel `tessera.autodiff.geometric` registry** ✅ LOCKED (2026-05-17)

The recommendation in [`ga_ebm_roadmap.md`](ga_ebm_roadmap.md) — and now
the locked decision — is a **parallel autodiff registry** dispatched on
`Multivector` inputs, parallel to the existing `tessera.autodiff.vjp._VJPS`
/ `_JVPS` dicts for tensor ops. Concretely:

- `python/tessera/autodiff/geometric/__init__.py` is the parallel
  namespace. It exposes its own `_VJPS_GEO` and `_JVPS_GEO` registries
  (frozensets keyed on op name) plus `register_vjp_geo` /
  `register_jvp_geo` decorators and `get_vjp_geo` / `get_jvp_geo`
  accessors.
- The existing 241+236 VJP/JVP entries are untouched. Mixing tensor +
  multivector graphs is supported by composing the two tapes (a
  combined `tape_geo()` context manager). The two registries are
  independent — there is no automatic conversion between
  `np.ndarray` cotangents and `Multivector` cotangents.
- Per-op VJP signature: `vjp_<op>(dout: Multivector, *args, **kwargs) -> tuple[Multivector, ...]`
  returning gradients for each multivector argument (and `None` for
  scalar / non-differentiable args).
- Per-op JVP signature: `jvp_<op>(tangents_in: tuple[Multivector, ...], primals: tuple[Multivector, ...]) -> Multivector`
  returning the forward-mode pushforward.
- The VJP derivation uses the **direct Cayley-table-adjoint formula**
  rather than the Hestenes anti-automorphism rewrite (`dout * reverse(b)`).
  Both formulas agree on Cl(p, 0, 0) (where the Frobenius and Hestenes
  inner products coincide); the direct-table formula extends cleanly
  to Cl(p, q, r) with q + r > 0 without per-signature special cases.

**Rationale.** Keeping the existing 241+236 VJP/JVP entries untouched
avoids regression risk in S2-S15. The parallel registry is consistent
with Q2 (sibling `Multivector` kind) — if multivectors are a parallel
tensor kind, their autodiff is naturally a parallel registry.

GA6 (multivector autodiff) is now unblocked.

## Deferred — Q5 (manifold set for EBM7)

Doesn't bite until EBM7, which depends on GA5 + GA6. Will be decided
during EBM6 → EBM7 transition.

## What's locked, what's open

| Question | Status | Bites at |
|---|---|---|
| Q1 — signature matrix | ✅ Cl(3,0) + Cl(1,3) | GA1 onwards |
| Q2 — type system | ✅ sibling `Multivector` kind | GA2 onwards |
| Q3 — autodiff strategy | ✅ parallel `autodiff.geometric` registry (2026-05-17) | GA6 |
| Q4 — backend priority | ✅ x86 → Apple CPU → Apple GPU → NVIDIA after Phase G | GA9 |
| Q5 — manifold set | ⏸ deferred (recommendation: S^n + SO(n) + ℝ^n) | EBM7 |

GA1 (algebra signature object) is now unblocked.
