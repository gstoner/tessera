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
| `numeric_policy` | Tessera-specific numerics contract: storage type, accumulator type, rounding, quantization scale/axis, and determinism. | `NumericPolicy(storage, accum, rounding, scale, quant_axis, deterministic)` |

## Dtype Names

Tessera keeps dtype names as strings today. The canonical spelling is the value
that should be stored in IR metadata. Aliases are accepted at selected API
boundaries and should normalize to the canonical spelling.

### Current Canonical Surface

| Family | Canonical names | Aliases / notes |
| --- | --- | --- |
| FP64 | `fp64` | `f64` |
| FP32 | `fp32` | `f32`; default user-facing floating dtype for accelerator-friendly examples |
| FP16 | `fp16` | `f16` |
| BF16 | `bf16` | Preferred reduced-precision training/inference dtype where target support exists |
| FP8 | `fp8_e4m3`, `fp8_e5m2` | Low-precision storage/quantization families; backend support is target-gated |
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
| Direct packed INT4 | `int4` | Planned/gated; current quantized paths should not imply a first-class packed int4 tensor type |
| AMD MX formats | `mxfp8`, `mxfp6`, `mxfp4` | Planned/gated; needs block-scale metadata and ROCm/CDNA target gates |
| Tenstorrent BFP/block formats | `bfp8`, `bfp4`, `blockfp8`, `blockfp4` | Planned/gated; do not alias to OCP FP8/FP4, AMD MXFP, or NVIDIA NVFP4 |
| TF32 | Not a storage dtype | Model as `math_mode="tf32"` on `fp32` tensors or numeric policy, not as `dtype="tf32"` |

## Canonicalization Direction

Tessera's current implementation is string-based. The intended public API layer
is JAX-like rather than NumPy-like:

- Dtype inputs may be strings, aliases, or future dtype objects.
- Public APIs should canonicalize aliases before storing dtype metadata.
- Default floating-point examples should prefer `fp32`, not `fp64`, unless an
  explicit accuracy-oriented mode is enabled.
- Future weak scalar semantics should allow Python scalar literals to follow the
  typed tensor they are combined with, rather than forcing 64-bit promotion.
- Future strict promotion mode should reject implicit mixed-dtype tensor
  arithmetic unless an operator or numeric policy declares the conversion.

Tessera does not currently expose a public `tessera.dtype` class, a public
promotion lattice, or a full equivalent of `jax.dtypes.result_type`.

## Promotion And Casting Policy

Current Tessera lowering does not define one global PyTorch- or JAX-compatible
promotion table. Operator implementations and backend shims may use local
reference behavior. The canonical direction is:

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
scalar behavior. Tessera's equivalent direction is a tensor-attribute layer
over `shape`, `dtype`, `distribution`, `layout`, and `numeric_policy`.

JAX also provides two dtype behaviors Tessera should mirror over time:

- `jax_enable_x64`: an explicit switch for allowing/defaulting wider 64-bit
  dtypes. Tessera should remain accelerator-friendly by default.
- `jax_numpy_dtype_promotion`: standard vs. strict promotion. Tessera should
  eventually support strict compiler dtype checking without changing the
  existing string APIs.

## Source Of Truth

- User-facing tensor attribute vocabulary: this document.
- Textual dtype terminals and type syntax: `docs/spec/LANGUAGE_AND_IR_SPEC.md`.
- Python public symbols: `docs/spec/PYTHON_API_SPEC.md` and
  `docs/CANONICAL_API.md`.
- Graph IR dtype normalization and MLIR-like spelling:
  `python/tessera/compiler/graph_ir.py`.
- Target capability gates: `python/tessera/compiler/capabilities.py`.
- Primitive dtype/layout contract status:
  `python/tessera/compiler/primitive_coverage.py`.
