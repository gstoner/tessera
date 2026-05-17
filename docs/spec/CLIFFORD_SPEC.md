---
status: Normative (live specification)
classification: Spec
authority: Defines the Clifford / geometric-algebra primitive surface across Tessera IR levels
last_updated: 2026-05-17
---

# Tessera Clifford / Geometric Algebra Specification

This document is the compact normative spec for Tessera's **Clifford /
geometric algebra** primitive surface. It covers algebra signatures,
multivector values and annotations, primitive contracts, geometric autodiff,
IR mapping, lowering, backend status, and out-of-scope items for v1.

The implementation is deliberately Tessera-native: Clifford signatures and
grades are compiler-visible concepts, not conventions encoded in plain tensor
dimensions. The active roadmap and scope-lock records are
[`docs/audit/ga_ebm_roadmap.md`](../audit/ga_ebm_roadmap.md) and
[`docs/audit/ga_scope_lock.md`](../audit/ga_scope_lock.md).

## 1. Concept

A Clifford algebra `Cl(p,q,r)` defines products over basis vectors whose
squares are positive, negative, or null. A **multivector** is a value with
coefficients over the algebra's basis blades:

- scalars are grade 0;
- vectors are grade 1;
- bivectors are grade 2;
- higher-grade blades represent oriented areas, volumes, and forms;
- mixed-grade multivectors carry a static grade set when known.

Tessera v1 supports two first-class signatures:

| Signature | Dimension | Primary use |
|-----------|-----------|-------------|
| `Cl(3,0)` | 8 basis blades | 3D Euclidean geometry, robotics, vision, rotation-equivariant features |
| `Cl(1,3)` | 16 basis blades | Minkowski spacetime, Lorentz-invariant model features |

The type system is designed around general `(p,q,r)` triples, but the v1
allow-list is intentionally small so product tables fit in registers and
backend kernels stay inspectable.

## 2. Primitive Contract

The live namespace is `tessera.ga`.

### 2.1 `tessera.ga.Cl(p, q, r=0) -> Cl`

Constructs an algebra signature. The signature is a compile-time-visible value
used by Python reference ops, type annotations, dialect attributes, product
table expansion, and backend manifests.

Contract:

- v1 accepts `Cl(3,0)` and `Cl(1,3)` only.
- `dim == 2 ** (p + q + r)`.
- `grades == (0, ..., p + q + r)`.
- Basis blades are indexed by bitmask order; coefficient index 0 is scalar.
- The Cayley product table is cached per signature.
- Invalid or unsupported signatures raise `TesseraAlgebraError`.

### 2.2 `tessera.ga.Multivector(coefficients, algebra, *, grades=None) -> Multivector`

Creates a multivector value. The final coefficient axis is the algebra basis
dimension; leading axes are batch, spatial, or field axes and broadcast through
pointwise GA ops.

Contract:

- `coefficients.shape[-1] == algebra.dim`.
- Coefficients are real floating values; integer inputs are promoted to
  floating reference storage.
- `grades`, when provided, must be a subset of `algebra.grades`.
- Coefficients outside a declared grade set are zeroed on construction.
- Values are immutable from the public API: `.coefficients` returns a read-only
  view.
- Shape or grade violations raise `TesseraAlgebraError`.

### 2.3 `tessera.ga.Multivector[Cl, grades?] -> MultivectorSpec`

Creates the annotation form used by `@tessera.jit` and constraint checks.

Contract:

- `Multivector[Cl(3,0)]` means any multivector in `Cl(3,0)`.
- `Multivector[Cl(3,0), {0,2}]` restricts the value to scalar + bivector
  grades.
- `Multivector[Cl(1,3), 2]` restricts the value to bivectors.
- The first argument must be a `Cl` signature.
- Invalid grade sets fail at annotation construction time.

### 2.4 `geometric_product(a, b) -> Multivector`

Computes the Clifford product `a * b`.

Contract:

- `a.algebra == b.algebra`.
- Leading axes broadcast by NumPy/Tessera broadcasting rules.
- Output coefficient axis has length `algebra.dim`.
- Output dtype follows `np.result_type(a.dtype, b.dtype)` in the Python
  reference.
- The product is evaluated by the algebra's Cayley table.
- Mismatched algebras raise `TesseraAlgebraError`.

### 2.5 `grade_projection(a, grades) -> Multivector`

Projects a multivector onto one or more grades.

Contract:

- `grades` may be an integer grade or iterable of integer grades.
- All requested grades must be valid for `a.algebra`.
- Output preserves shape, dtype, and algebra.
- Coefficients outside the requested grades are zero.
- Output carries the requested grade set as static metadata.

### 2.6 `wedge(a, b) -> Multivector`

Computes the exterior product `a ∧ b`.

Contract:

- Inputs must share an algebra.
- Basis-blade terms with overlapping generator sets contribute zero.
- Disjoint terms use the same orientation sign convention as the geometric
  product.
- Leading axes broadcast; dtype promotion follows the geometric product.

### 2.7 `left_contraction(a, b) -> Multivector`

Computes the left contraction `a ⌋ b`.

Contract:

- Inputs must share an algebra.
- For grade-pure terms of grades `r` and `s`, output keeps the grade
  `s - r` component of `a * b` when `s >= r`; otherwise the term is zero.
- Mixed-grade inputs are handled by summing grade-pure contributions.

### 2.8 `inner(a, b) -> ndarray`

Computes the scalar bilinear form `<a, b> = <a * reverse(b)>_0`.

Contract:

- Inputs must share an algebra.
- Output is the scalar coefficient array, not a `Multivector`.
- Leading axes broadcast.
- In indefinite signatures, callers must treat the form as signature-aware,
  not positive definite.

### 2.9 Anti-automorphisms

Operations:

- `reverse(a)`
- `grade_involution(a)`
- `conjugate(a)`

Contract:

- Output preserves shape, dtype, algebra, and declared grade set.
- `reverse` applies sign `(-1)^(k(k-1)/2)` to grade `k`.
- `grade_involution` applies sign `(-1)^k` to grade `k`.
- `conjugate` is `reverse(grade_involution(a))`.

### 2.10 Norm and Rotor Ops

Operations:

- `norm_squared(a)`
- `norm(a)`
- `exp_mv(a)`
- `log_mv(a)`
- `rotor_from_axis(axis, angle)`
- `rotor_sandwich(R, x)`

Contract:

- `norm_squared` and `norm` use the scalar part of `a * reverse(a)` in the
  Python reference.
- `exp_mv` uses a closed form for `Cl(3,0)` bivectors and a power-series
  fallback for other accepted inputs.
- `log_mv` uses a closed form for `Cl(3,0)` rotors and a power-series fallback
  where supported.
- `rotor_from_axis` creates a rotor from a bivector axis and angle.
- `rotor_sandwich(R, x)` computes `R * x * reverse(R)` and is the canonical
  rotation/equivariance primitive.

### 2.11 Differential and Manifold Ops

Operations:

- `hodge_star(a)`
- `ext_deriv(field)`
- `codiff(field)`
- `vec_deriv(field)`
- `integral(field_or_fn, manifold)`
- `Euclidean`, `Sphere`, `SOn`

Contract:

- `hodge_star` is signature-aware; double application follows the
  signature-dependent sign rule.
- `ext_deriv`, `codiff`, and `vec_deriv` are finite-difference reference
  operations on uniform Euclidean grids in the current Python surface.
- `integral` supports callable and field modes over supported manifolds.
- `Sphere` and `SOn` are the initial manifold helpers used by GA/EBM
  conformance and manifold-aware Langevin paths.

## 3. Type and Constraint Contract

`Multivector` is a sibling kind to `Tensor`, not an extension of the six
canonical tensor attributes. It shares the usual shape, dtype, layout, device,
and distribution machinery where applicable, and adds:

- `algebra: Cl(p,q,r)`
- `grades: frozenset[int] | None`

Annotation markers:

- `Rotor[Cl]`
- `DiffForm[Cl, k]`
- `VectorField[Cl]`
- `Morphism[...]`

Grade-aware predicates live in `tessera.ga.constraints`:

- `GradeIn(name, allowed)`
- `Even(name)`
- `Odd(name)`
- `IsRotor(name)`
- `IsForm(name)`

Malformed signatures or impossible grade annotations must fail through
`TesseraAlgebraError` or the existing `TesseraConstraintError` path.

## 4. Autodiff Contract

Geometric autodiff is isolated in `tessera.autodiff.geometric`. It uses a
parallel registry rather than extending the ordinary tensor VJP/JVP tables:

- `_VJPS_GEO`, `_JVPS_GEO`
- `register_vjp_geo`, `register_jvp_geo`
- `get_vjp_geo`, `get_jvp_geo`
- `check_grad_geo`, `check_jvp_geo`
- `tape_geo()`

Per-op VJPs use direct Cayley-table adjoint formulas so they work for
non-Euclidean signatures such as `Cl(1,3)` without relying on Euclidean-only
anti-automorphism shortcuts. Tensor autodiff and geometric autodiff may coexist
in one program, but no implicit conversion between tensor cotangents and
multivector cotangents is performed.

## 5. Mapping to Tessera IR

| Level | What lives here |
|-------|-----------------|
| **Graph IR** | `tessera_clifford` ops for geometric product, grade projection, anti-automorphisms, norms, exp/log, rotor sandwich, and differential-form operations. |
| **Schedule IR** | Grade-aware transformations, algebra annotations, and loop structure for field-level operations. |
| **Tile IR** | Product-table contractions, grade-filtered contractions, rotor kernels, field finite differences, and reduction kernels. |
| **Target IR** | Backend-specific kernels and artifacts, with v1 priority x86 reference, Apple CPU reference, Apple GPU fused kernels, then NVIDIA after the main NVIDIA execution path matures. |

## 6. Clifford Dialect and Passes

The active C++/MLIR surface is under `src/solvers/clifford/`.

Driver:

- `ts-clifford-opt`

Dialect:

- `tessera_clifford`

Passes:

- `tessera-clifford-annotate-algebra` — validates the v1 signature
  allow-list and attaches canonical algebra metadata.
- `tessera-clifford-rotor-sandwich-fold` — recognizes
  `geo_product(geo_product(R, x), reverse(R))` and folds it to a single
  rotor-sandwich op when the rotor matches.
- `tessera-clifford-grade-fusion` — fuses grade projections into producer
  metadata so later expansion emits only requested output grades.
- `tessera-clifford-expand-product-table` — lowers rank-1 v1
  `geo_product` ops through compile-time Cayley tables into tensor extract,
  arithmetic, and tensor construction operations.

The canonical pipeline order is:

```text
annotate-algebra -> rotor-sandwich-fold -> grade-fusion -> expand-product-table
```

`rotor-sandwich-fold` must run before grade fusion because grade metadata can
hide the simple three-op chain shape.

## 7. Backend and Execution Status

Layer-specific status for the Clifford and EBM tracks is centralized in
[`docs/spec/GA_EBM_EXECUTION_STATUS.md`](GA_EBM_EXECUTION_STATUS.md). The short
Clifford summary is:

| Layer | Status | Source |
|-------|--------|--------|
| Python reference | implemented | `python/tessera/ga/`, `python/tessera/autodiff/geometric/` |
| MLIR dialect and lit fixtures | implemented / lit-testable | `src/solvers/clifford/` |
| Backend manifest | implemented | `python/tessera/compiler/backend_manifest.py` |
| Apple GPU native path | hardware-runtime for all 17 registered GA primitives as fused MSL kernels | `src/compiler/codegen/Tessera_Apple_Backend/` and focused Apple GPU tests |
| x86 / Apple CPU native GA kernels | reference-first | Python reference path; native batched kernels are follow-up work |
| NVIDIA / ROCm GA kernels | planned | No v1 hardware-runtime claim |

Backend contract:

- A manifest entry is a coverage and dtype contract, not by itself a hardware
  execution claim.
- Hardware-runtime status requires a concrete backend build and test path.
- Quantitative performance claims require benchmarks in addition to correctness
  tests.
- For Apple GPU, the current Clifford manifest maps the full 17-op GA primitive
  set to fused MSL kernels. `clifford_geometric_product` and
  `clifford_rotor_sandwich` carry fp32/fp16/bf16; the other GA kernels are fp32
  in v1.

## 8. Conformance Expectations

The v1 conformance suite should prove algebraic and geometric behavior before
performance:

- Cayley-table product correctness for `Cl(3,0)` and `Cl(1,3)`.
- Rotor sandwich agrees with an SO(3) Rodrigues reference.
- Rotation-invariant point-cloud features remain stable under random rotations.
- Lorentz-invariant rest mass is preserved under reference boosts.
- Geometric VJP/JVP checks agree with central-difference references.
- Lowering fixtures preserve algebra and grade metadata.

## 9. Out of Scope for v1

- Fully general `Cl(p,q,r)` beyond the v1 allow-list.
- Conformal `Cl(4,1)` kernels.
- Automatic reverse-mode tape recording for arbitrary Python GA operation DAGs
  at parity with the tensor autodiff path.
- Native NVIDIA / ROCm GA kernels.
- Symbolic exterior calculus or exact manifold integration.

## Appendix A — References

- Hestenes and Sobczyk, *Clifford Algebra to Geometric Calculus*
- Doran and Lasenby, *Geometric Algebra for Physicists*
- Frankel, *The Geometry of Physics*
