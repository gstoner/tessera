---
status: Normative
classification: Reference
last_updated: 2026-05-11
---

# Tessera Tensor Attributes And Dtypes

This document is the canonical user-facing reference for Tessera tensor
attributes. It gives the PyTorch-style answer to "what attributes does a tensor
carry?" while preserving Tessera's compiler-oriented model: shape, dtype,
layout, target, distribution, and numeric policy are semantic inputs to
lowering, legality, and scheduling.

Lower-level syntax and verifier rules remain defined by the language and IR
specs. This document defines the public vocabulary those specs should use.

## Tensor Attributes

| Attribute | Meaning | Current representation |
| --- | --- | --- |
| `shape` | Logical tensor dimensions. Dimensions may be concrete integers, symbolic names, or unknown `?` markers before specialization. | `IRType.shape`, `DistributedArray.shape`, textual `tensor<Shape x DType>` |
| `dtype` | Storage element type. Tessera uses canonical dtype strings with a small alias set. | `IRType.dtype`, `DistributedArray.dtype`, `TensorContract.dtype` |
| `layout` | Logical or physical layout metadata, separate from shape. Layout may be unspecified until schedule or target lowering. | `IRType.layout`, `IRArg.layout`, `TensorContract.layout`, textual `; layout=...` |
| `device` / `target` | Execution target profile. Tessera currently models this at JIT/module/legality level, not as a PyTorch-style per-tensor device attribute. | `@tessera.jit(target=...)`, `GPUTargetProfile`, capability registry target names |
| `distribution` | Mesh and sharding placement for distributed arrays. This is separate from both shape and layout. | `ShardSpec`, `MeshSpec`, `DistributedArray.shard_spec` |
| `numeric_policy` | Tessera-specific numerics contract: storage type, accumulator type, rounding, quantization scale/axis, determinism, and optional math mode. | `NumericPolicy(storage, accum, rounding, scale, quant_axis, deterministic[, math_mode])` |

## Multivector — A Parallel Tensor Kind (GA0 Scope Lock)

Per Q2 of [`docs/audit/ga_scope_lock.md`](../audit/ga_scope_lock.md),
the Geometric Algebra surface introduces a **sibling tensor kind**
called `Multivector` rather than adding `grade` and `algebra` as a
seventh and eighth canonical tensor attribute.  This keeps the six
canonical tensor attributes above unchanged for the tensor kind,
while giving Multivector its own kind-specific schema.

A `Multivector` value carries:

| Attribute | Meaning | Current representation |
| --- | --- | --- |
| `algebra` | Clifford signature `Cl(p, q, r)`. v1 supports `Cl(3, 0)` (3D Euclidean) and `Cl(1, 3)` (Minkowski) only. | `tessera.ga.Cl(p, q, r)`, `Multivector.algebra`, `Multivector.algebra.signature` |
| `coefficients` | Coefficient array on the algebra's basis blades. Last axis size = `algebra.dim`; preceding axes are batch. | `Multivector.coefficients` (`np.ndarray` shape `(*batch, algebra.dim)`) |
| `grades` | Set of grades present (subset of `{0, 1, …, algebra.n}`). `None` ⇒ all grades active; `grade_projection` narrows it. | `Multivector.grades`, `Multivector.active_grades` |
| `dtype` | Coefficient element type. Same canonical dtype set as the tensor kind — `f32` / `f64` / `f16` / `bf16`. | `Multivector.dtype` (NumPy dtype, canonicalized at the public-API boundary) |
| `shape` | Leading (batch) shape, **excluding** the algebra axis. A scalar (rank-0) Multivector has `shape=()`. | `Multivector.shape` (= `coefficients.shape[:-1]`) |

A `MultivectorField` adds:

| Attribute | Meaning | Current representation |
| --- | --- | --- |
| `spatial_ndim` | Rank of the underlying grid (3 for Cl(3,0) field ops `ext_deriv` / `vec_deriv` / `codiff`). | `MultivectorField.spatial_ndim` |
| `spacing` | Per-axis grid step `h_i`. Read by every finite-difference op. | `MultivectorField.spacing` |
| `spatial_shape` | Grid shape `(D_0, …, D_{spatial_ndim - 1})`. The algebra axis is the trailing dimension of `values`. | `MultivectorField.spatial_shape` |

**Why a separate kind?**  The grade structure and Clifford-algebra
signature don't compose meaningfully with the tensor kind's
attributes (a tensor `layout` is orthogonal to a multivector
`grade`; a tensor `distribution` over a mesh has no clean
counterpart for the algebra axis).  Keeping Multivector parallel
means the canonical six attributes stay clean for tensor ops, and
Multivector-specific rules (grade-restricted subspaces, Cayley-table
products) live in `tessera.ga` without leaking into the tensor IR.

**Shared with the tensor kind**: dtype canonicalization (the same
canonical dtype names + alias normalization apply), the broader
`@jit(target=…)` lowering pipeline, and `tessera._apple_gpu_dispatch`
for Apple GPU dispatch.  Manifest entries for Clifford ops live in
the **parallel** `_CLIFFORD_APPLE_GPU_FUSED` table inside
[`backend_manifest.py`](../../python/tessera/compiler/backend_manifest.py),
not in the tensor `OP_SPECS` catalog.

## Dtype Names

Tessera stores dtype metadata as canonical strings. The canonical spelling is
the value that should be stored in IR metadata. Aliases and `Dtype` helper
objects are accepted at selected API boundaries and should normalize to the
canonical spelling.

### Current Canonical Surface

| Family | Canonical names | Aliases / notes |
| --- | --- | --- |
| FP64 | `fp64` | `f64` |
| FP32 | `fp32` | `f32`; default user-facing floating dtype for accelerator-friendly examples |
| FP16 | `fp16` | `f16` |
| BF16 | `bf16` | Preferred reduced-precision training/inference dtype where target support exists |
| FP8 | `fp8_e4m3`, `fp8_e5m2` | Low-precision storage/quantization families; backend support is target-gated. AMD GFX12 instruction spellings `FP8` / `F8` normalize to `fp8_e4m3`; `BF8` / `bfloat8` normalize to `fp8_e5m2`. |
| FP6 | `fp6_e2m3`, `fp6_e3m2` | Low-precision storage/quantization families; backend support is target-gated |
| FP4 | `fp4_e2m1` | Low-precision storage/quantization family; backend support is target-gated |
| NVFP4 | `nvfp4` | NVIDIA block-scaled FP4 policy name; do not alias to OCP FP4 or AMD MXFP4 |
| Signed integers | `int8`, `int16`, `int32`, `int64` | `i8`, `i16`, `i32`, `i64` may appear in MLIR spellings |
| Boolean | `bool` | Lowers to `i1` in MLIR-like inspection text |

### Planned Or Gated Dtypes

| Family | Intended names | Status |
| --- | --- | --- |
| Unsigned integers | `uint8`, `uint16`, `uint32`, `uint64` | Planned Graph IR mappings; storage legality should be separate from acceleration legality |
| Complex | `complex64`, `complex128`; possible future `complex32` | Planned; no canonical Graph IR complex dtype family today |
| Direct packed INT4 | `int4` | Planned/gated; current quantized paths should not imply a first-class packed int4 tensor type. AMD GFX12 `IU4` WMMA/SWMMAC instructions are tracked against this planned-gated dtype until Tessera grows a separate unsigned packed-4 policy. |
| AMD MX formats | `mxfp8`, `mxfp6`, `mxfp4` | Planned/gated; needs block-scale metadata and ROCm/CDNA target gates |
| Tenstorrent BFP/block formats | `bfp8`, `bfp4`, `blockfp8`, `blockfp4` | Planned/gated; do not alias to OCP FP8/FP4, AMD MXFP, or NVIDIA NVFP4 |
| TF32 | Not a storage dtype | Model as `math_mode="tf32"` on `fp32` tensors or numeric policy, not as `dtype="tf32"` |

## Canonical Dtype API

Tessera keeps string compatibility for existing APIs, and also exposes
`tessera.dtype` as the canonical dtype helper module:

| API | Purpose |
| --- | --- |
| `tessera.dtype.Dtype(value)` | Str-compatible wrapper around a canonical dtype name. Aliases normalize at construction. |
| `tessera.dtype.canonicalize_dtype(value, allow_planned_gated=False)` | Normalize aliases such as `f32` or `float32` to canonical storage spellings such as `fp32`. |
| `tessera.dtype.assert_canonical_dtype(value, context=None)` | Canonicalize or raise a contextual `TesseraDtypeError`. |
| `tessera.dtype.result_type(*dtypes, mode="standard")` | Compute the standard promotion result or reject mixed dtypes in `mode="strict"`. |
| `tessera.dtype.canonical_dtypes()` | Return the canonical dtype set. |
| `tessera.dtype.planned_gated_dtypes()` | Return the recognized-but-gated dtype set. |
| `tessera.dtype.dtype_aliases()` | Return accepted alias spellings. |

The helper module is the implementation source for canonical dtype membership.
Graph IR, distributed arrays, backend manifests, and primitive coverage audits
should use these helpers instead of open-coded dtype tables.

The public API direction remains JAX-like rather than NumPy-like:

- Dtype inputs may be strings, aliases, or `Dtype` objects.
- Public APIs canonicalize aliases before storing dtype metadata.
- Default floating-point examples prefer `fp32`, not `fp64`, unless an explicit
  accuracy-oriented mode is enabled.
- Weak scalar semantics are still a planned tensor-expression behavior:
  Python scalar literals should eventually follow the typed tensor they combine
  with rather than forcing 64-bit promotion.
- Strict promotion is available at the dtype helper level through
  `result_type(..., mode="strict")`; broader operator enforcement remains a
  compiler policy direction.

## Promotion And Casting Policy

Tessera exposes a dtype-helper promotion lattice through
`tessera.dtype.result_type`. Current operator lowering does not yet enforce one
global PyTorch- or JAX-compatible promotion table; operator implementations and
backend shims may still use local reference behavior. The canonical direction
is:

1. Storage dtype is explicit on tensors and annotations.
2. Accumulator dtype belongs in `numeric_policy`, not in the storage dtype.
3. Mixed precision is legal only when declared by an operator, autocast region,
   target lowering rule, or numeric policy.
4. Low-bit and block-scaled formats require explicit quantization or numeric
   policy metadata.
5. Backend support is capability-gated; a dtype can be canonical without being
   executable on every target.

## JAX Comparison

JAX's closest concept is `jax.ShapeDtypeStruct(shape, dtype, sharding,
weak_type=...)`: a static container for array shape, dtype, sharding, and weak
scalar behavior. Tessera's equivalent direction is the tensor-attribute layer
over `shape`, `dtype`, `distribution`, `layout`, and `numeric_policy`, plus the
`tessera.dtype` helper module for canonicalization and promotion inspection.

JAX also provides two dtype behaviors Tessera should mirror over time:

- `jax_enable_x64`: an explicit switch for allowing/defaulting wider 64-bit
  dtypes. Tessera should remain accelerator-friendly by default.
- `jax_numpy_dtype_promotion`: standard vs. strict promotion. Tessera should
  support strict compiler dtype checking without changing the existing string
  APIs. `tessera.dtype.result_type(..., mode="strict")` is the helper-level
  surface for this direction.

## Source Of Truth

- User-facing tensor attribute vocabulary: this document.
- Textual dtype terminals and type syntax: `docs/spec/LANGUAGE_AND_IR_SPEC.md`.
- Python public symbols: `docs/spec/PYTHON_API_SPEC.md` and
  `docs/CANONICAL_API.md`.
- Graph IR dtype normalization and MLIR-like spelling:
  `python/tessera/compiler/graph_ir.py`.
- Public dtype helpers: `python/tessera/dtype.py`.
- Target capability gates: `python/tessera/compiler/capabilities.py`.
- Primitive dtype/layout contract status:
  `python/tessera/compiler/primitive_coverage.py`.
