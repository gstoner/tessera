---
status: Normative
classification: Normative
last_updated: 2026-04-26
---

# Tessera Graph IR Specification
**Status:** Normative — grounded in `src/compiler/ir/TesseraOps.td` and `src/transforms/lib/CanonicalizeTesseraIR.cpp`
**Last updated:** April 26, 2026
**Dialect name:** `tessera`
**C++ namespace:** `::tessera`

---

## 1. Overview

The **Graph IR** is the first lowering target for `@tessera.jit` functions. It sits above the Schedule IR in the four-layer IR stack:

```
Python API  (@tessera.jit, Region[...])
     │  [GraphIRBuilder — python/tessera/compiler/graph_ir.py]
     ▼
Graph IR    (tessera dialect — mathematical ops, effects, shapes)
     │  [CanonicalizeTesseraIRPass, DistributionLoweringPass, EffectAnnotationPass]
     ▼
Schedule IR (schedule.* dialect)
```

Graph IR encodes the **mathematical intent** of a computation. It is backend-agnostic: the same Graph IR module can be lowered to x86 AMX (via `tessera-lower-to-x86`) or NVIDIA GPU (via `tessera-lower-to-gpu`).

---

## 2. Tessera Dialect Registration

```mlir
// Dialect header declaration
dialect "tessera" {
  cppNamespace = "::tessera"
}
```

All ops use the `tessera.` prefix in MLIR text.

---

## 3. Attributes

### 3.1 `EpilogueKind` Enum Attribute

**TableGen:** `Tessera_EpilogueKind` (I32EnumAttr)
**MLIR attr type:** `tessera.epilogue`

| Case | Value | Meaning |
|------|-------|---------|
| `None` | 0 | No post-op fused |
| `Relu` | 1 | Fuse ReLU activation |
| `Gelu` | 2 | Fuse GELU activation |
| `Silu` | 3 | Fuse SiLU activation |

Used by `tessera.conv2d_nhwc` (`epilogue` attr) and `tessera.fused_epilogue` (`epilogue` required attr).

### 3.2 `tessera.effect` Function Attribute

Attached to `func.func` operations by `EffectAnnotationPass`. String-valued.

| Value | Meaning |
|-------|---------|
| `"pure"` | No side effects; recompute-safe |
| `"random"` | Calls RNG; result varies |
| `"memory"` | Reads/writes mutable state |
| `"io"` | Collective communication or host I/O |

### 3.3 `tessera.shard` Argument Attribute

Attached to `func.func` argument tensors by `GraphIRBuilder` when the function receives a `DistributedArray`. Encodes partition metadata for `DistributionLoweringPass`.

```mlir
func.func @step(%a: tensor<128x256xbf16>
                    {tessera.shard = {axes = ["dp"], dims = [0]}})
```

---

## 4. Op Catalog

All ops have `hasVerifier = 1` unless noted. Verifiers check shape compatibility and attribute validity at IR construction time.

---

### 4.1 `tessera.matmul`

**TableGen:** `Tessera_MatmulOp`
**Traits:** `Pure`, `NoSideEffect`, `TilingInterface`

Matrix multiply: `result = lhs @ rhs`.

```
tessera.matmul %lhs, %rhs : (tensor<MxKxeT>, tensor<KxNxeT>) -> tensor<MxNxeT>
```

**Arguments:**

| Arg | Type | Required | Description |
|-----|------|----------|-------------|
| `$lhs` | `TensorType` | Yes | Left operand. 2D ranked tensor. |
| `$rhs` | `TensorType` | Yes | Right operand. 2D ranked tensor. |
| `$tile_k` | `OptionalAttr<I64Attr>` | No | K-dimension tile size hint. If set, `TilingPass` uses this value instead of `--tile-n`. |
| `$transposeA` | `BoolAttr` | No (default `false`) | If `true`, treat `lhs` as transposed (lhs^T @ rhs). |
| `$transposeB` | `BoolAttr` | No (default `false`) | If `true`, treat `rhs` as transposed (lhs @ rhs^T). |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$result` | `TensorType` | Output matrix. Shape `[M, N]`, element type determined by accumulation rules. |

**Verifier rules:**
- `lhs` must be a 2D ranked tensor.
- `rhs` must be a 2D ranked tensor.
- Inner dimensions must match: `lhs.shape[1] == rhs.shape[0]` (unless one is dynamic).
- If both `transposeA` and `transposeB` are set, the verifier warns (prefer explicit transpose ops before `matmul`).

**MLIR text examples:**
```mlir
// Standard matmul
%C = tessera.matmul %A, %B : (tensor<128x256xbf16>, tensor<256x512xbf16>) -> tensor<128x512xf32>

// With transpose flags (after TransposeIntoMatmul canonicalization)
%C = tessera.matmul %A, %B {transposeB = true} : (tensor<128x256xbf16>, tensor<512x256xbf16>) -> tensor<128x512xf32>

// With tile_k hint
%C = tessera.matmul %A, %B {tile_k = 64 : i64} : (tensor<512x256xbf16>, tensor<256x512xbf16>) -> tensor<512x512xf32>
```

---

### 4.2 `tessera.conv2d_nhwc`

**TableGen:** `Tessera_Conv2DNHWCOp`
**Traits:** `Pure`, `NoSideEffect`, `TilingInterface`

2D convolution in NHWC layout: `result = conv2d(input, filter)`.

```
tessera.conv2d_nhwc %input, %filter {strides, dilations, epilogue?}
    : (tensor<NxHxWxCxeT>, tensor<HfxWfxCxFxeT>) -> tensor<NxHoxWoxFxeT>
```

**Arguments:**

| Arg | Type | Required | Description |
|-----|------|----------|-------------|
| `$input` | `TensorType` | Yes | Input feature map. Layout: `[N, H, W, C_in]`. |
| `$filter` | `TensorType` | Yes | Filter kernel. Layout: `[H_f, W_f, C_in, C_out]`. |
| `$strides` | `ArrayAttr` | Yes | Stride values `[stride_h, stride_w]`. |
| `$dilations` | `ArrayAttr` | Yes | Dilation values `[dilation_h, dilation_w]`. |
| `$epilogue` | `OptionalAttr<Tessera_EpilogueKindAttr>` | No | If set, fuses a post-op (e.g. `Relu`, `Gelu`) into the convolution. Set by `FuseConvRelu` canonicalization. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$result` | `TensorType` | Output feature map. Layout: `[N, H_o, W_o, C_out]`. |

**Verifier rules:**
- `input` and `filter` must be 4D ranked tensors.
- `strides` and `dilations` must each have exactly 2 elements.
- `C_in` must match between `input` dim 3 and `filter` dim 2 (if static).

**MLIR text examples:**
```mlir
// Plain convolution
%out = tessera.conv2d_nhwc %in, %filt {strides = [1, 1], dilations = [1, 1]}
    : (tensor<1x224x224x3xf32>, tensor<3x3x3x64xf32>) -> tensor<1x222x222x64xf32>

// Fused with ReLU (after FuseConvRelu canonicalization)
%out = tessera.conv2d_nhwc %in, %filt {strides = [1, 1], dilations = [1, 1], epilogue = #tessera.epilogue<Relu>}
    : (tensor<1x224x224x3xf32>, tensor<3x3x3x64xf32>) -> tensor<1x222x222x64xf32>
```

---

### 4.3 `tessera.flash_attn`

**TableGen:** `Tessera_FlashAttnOp`
**Traits:** `Pure`, `NoSideEffect`

Fused FlashAttention operation. Computes multi-head attention output `O = softmax(QK^T / sqrt(d_k)) V`.

In Phase 1, lowered to naive O(S²) numpy implementation. In Phase 3, `TileIRLoweringPass` expands this into FA-4 Tile IR ops.

```
tessera.flash_attn %q, %k, %v {head_dim, dropout_p?, causal?}
    : (tensor<BxHxSxDxeT>, tensor<BxHxSxDxeT>, tensor<BxHxSxDxeT>) -> tensor<BxHxSxDxeT>
```

**Arguments:**

| Arg | Type | Required | Description |
|-----|------|----------|-------------|
| `$q` | `TensorType` | Yes | Query tensor. Shape: `[B, H, S, D]`. |
| `$k` | `TensorType` | Yes | Key tensor. Shape: `[B, H, S, D]`. |
| `$v` | `TensorType` | Yes | Value tensor. Shape: `[B, H, S, D]`. |
| `$head_dim` | `I64Attr` | Yes | Head dimension `D`. Used to compute the attention scale `1/sqrt(D)`. |
| `$dropout_p` | `OptionalAttr<F64Attr>` | No | Dropout probability. If `0.0` or absent, `DropoutZeroSimplify` canonicalization removes it. Sets effect to `random` when non-zero. |
| `$causal` | `BoolAttr` | No (default `false`) | If `true`, `TileIRLoweringPass` emits `tessera.attn.causal_mask` in the inner loop. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$o` | `TensorType` | Attention output. Same shape as `q`. |

**Verifier rules:**
- `q`, `k`, `v` must each be 4D ranked tensors.
- Shapes of `q`, `k`, `v` must all be equal (if static).
- `head_dim` must be > 0.

**Tile size attributes** (added by `FlashAttnLoweringConfig.to_mlir_attrs()`):

| Attribute | Type | Description |
|-----------|------|-------------|
| `tessera.tile_q` | `i32` | Q tile rows. Used by `TileIRLoweringPass` and Phase 5 autotuner. |
| `tessera.tile_kv` | `i32` | KV tile cols. Used by `TileIRLoweringPass` and Phase 5 autotuner. |
| `tessera.pipeline_stages` | `i32` | Software pipeline stages for double-buffering. |

**MLIR text examples:**
```mlir
// Causal attention with tile hints (after @jit with FlashAttnLoweringConfig)
%o = tessera.flash_attn %Q, %K, %V
    {head_dim = 64 : i64, causal = true,
     tessera.tile_q = 64 : i32, tessera.tile_kv = 64 : i32,
     tessera.pipeline_stages = 2 : i32}
    : (tensor<2x8x512x64xbf16>, tensor<2x8x512x64xbf16>, tensor<2x8x512x64xbf16>)
   -> tensor<2x8x512x64xbf16>

// After DropoutZeroSimplify canonicalization (dropout_p removed)
%o = tessera.flash_attn %Q, %K, %V {head_dim = 64 : i64}
    : (tensor<2x8x512x64xbf16>, ...) -> tensor<2x8x512x64xbf16>
```

---

### 4.4 `tessera.fused_epilogue`

**TableGen:** `Tessera_FusedEpilogueOp`
**Traits:** `Pure`, `NoSideEffect`

Fused matmul + optional bias + activation. Generated by `FuseMatmulBiasGELU` canonicalization from `tessera.matmul → tessera.add → tessera.gelu` chains.

```
tessera.fused_epilogue %lhs, %rhs, %bias {epilogue, has_bias?}
    : (tensor<MxKxeT>, tensor<KxNxeT>, tensor<NxeT>) -> tensor<MxNxeT>
```

**Arguments:**

| Arg | Type | Required | Description |
|-----|------|----------|-------------|
| `$lhs` | `TensorType` | Yes | Left operand (A matrix). |
| `$rhs` | `TensorType` | Yes | Right operand (B matrix). |
| `$bias` | `TensorType` | Yes | Bias vector. Must be broadcastable to result shape. |
| `$epilogue` | `Tessera_EpilogueKindAttr` | Yes | Post-op to fuse: `None`, `Relu`, `Gelu`, or `Silu`. |
| `$has_bias` | `BoolAttr` | No (default `false`) | If `true`, bias is added before the activation. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$result` | `TensorType` | `activation(A @ B + bias)` if `has_bias`, else `activation(A @ B)`. |

**MLIR text examples:**
```mlir
// Matmul + bias + GELU (generated by FuseMatmulBiasGELU)
%out = tessera.fused_epilogue %A, %B, %bias
    {epilogue = #tessera.epilogue<Gelu>, has_bias = true}
    : (tensor<512x256xbf16>, tensor<256x512xbf16>, tensor<512xf32>) -> tensor<512x512xf32>

// Matmul + ReLU, no bias
%out = tessera.fused_epilogue %A, %B, %empty
    {epilogue = #tessera.epilogue<Relu>}
    : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64xf32>) -> tensor<64x64xf32>
```

---

### 4.5 `tessera.transpose`

**TableGen:** `Tessera_TransposeOp`
**Traits:** `Pure`, `NoSideEffect`
**Has verifier:** No

Transposes a tensor. In most cases canonicalized away by `TransposeIntoMatmul` — the transpose flag is folded into the `tessera.matmul` op.

```
tessera.transpose %x : tensor<MxNxeT> -> tensor<NxMxeT>
```

**Arguments:** `$x : TensorType`
**Results:** `$y : TensorType`

**MLIR text example:**
```mlir
%xt = tessera.transpose %x : tensor<256x128xbf16> -> tensor<128x256xbf16>
// After TransposeIntoMatmul: the following matmul becomes transposeA=true
%C  = tessera.matmul %xt, %B : (tensor<128x256xbf16>, tensor<256x64xbf16>) -> tensor<128x64xf32>
// →
%C  = tessera.matmul %x, %B {transposeA = true} : ...
```

---

### 4.6 `tessera.cast`

**TableGen:** `Tessera_CastOp`
**Traits:** `Pure`, `NoSideEffect`
**Has verifier:** No

Casts a tensor element type. Used for precision transitions (e.g. `bf16 → f32`).

```
tessera.cast %x : tensor<MxNxbf16> -> tensor<MxNxf32>
```

**Arguments:** `$x : TensorType`
**Results:** `$y : TensorType`

---

### 4.7 Implicit ops referenced in canonicalization

The following op names appear in canonicalization patterns but are not yet defined in `TesseraOps.td`. They are used as string-based `OperationState` names only:

| Op name | Context |
|---------|---------|
| `tessera.gelu` | Source op for `FuseMatmulBiasGELU` (the `gelu` being matched) |
| `tessera.relu` | Source op for `FuseConvRelu` (the `relu` being matched) |
| `tessera.add` | Intermediate add op matched by `FuseMatmulBiasGELU` |

These ops are created by `GraphIRBuilder` when lowering Python ops that call `tessera.ops.gelu`, `tessera.ops.relu`, and arithmetic addition. They carry the `Pure` trait by convention. The canonicalization patterns immediately fuse them away when they follow a `matmul` or `conv2d_nhwc`.

---

## 5. Canonicalization Patterns

Registered by `CanonicalizeTesseraIRPass` (pass flag: `--tessera-canonicalize`). Runs via `applyPatternsAndFoldGreedily`.

### 5.1 `FuseMatmulBiasGELU`

**Pattern:** `tessera.gelu(tessera.add(tessera.matmul(A, B), bias))` → `tessera.fused_epilogue(A, B, bias, epilogue=Gelu, has_bias=true)`

**Benefit:** 2 (high priority — eliminates two ops)

**Match conditions:**
- `gelu` has exactly 1 operand.
- Defining op of the operand is `tessera.add` with exactly 2 operands.
- Defining op of `add`'s first operand is `tessera.matmul`.

**Result:** Single `tessera.fused_epilogue` with `epilogue = Gelu`, `has_bias = true`. The bias operand is `add`'s second operand.

---

### 5.2 `FuseConvRelu`

**Pattern:** `tessera.relu(tessera.conv2d_nhwc(...))` → `tessera.conv2d_nhwc(..., epilogue=Relu)`

**Benefit:** 2

**Match conditions:**
- `relu` has exactly 1 operand.
- Defining op of the operand is `tessera.conv2d_nhwc`.

**Result:** New `tessera.conv2d_nhwc` with all original attributes plus `epilogue = Relu`. All other attributes (strides, dilations) are preserved.

---

### 5.3 `DropoutZeroSimplify`

**Pattern:** `tessera.flash_attn {..., dropout_p = 0.0}` → `tessera.flash_attn {...}` (with `dropout_p` attr removed)

**Benefit:** 1

**Match conditions:**
- Op is `tessera.flash_attn`.
- `dropout_p` attribute exists and has value `0.0`.

**Result:** New `tessera.flash_attn` with all attributes except `dropout_p`. This prevents `EffectAnnotationPass` from tagging the function as `random` when dropout is disabled.

---

### 5.4 `TransposeIntoMatmul`

**Pattern:** `tessera.matmul(tessera.transpose(A), tessera.transpose(B))` → `tessera.matmul(A, B, transposeA=true, transposeB=true)`

Also handles partial cases: only `A` transposed, or only `B` transposed.

**Benefit:** 1

**Match conditions:**
- Op is `tessera.matmul` with 2 operands.
- At least one of `lhs`, `rhs` is defined by a `tessera.transpose` op.

**Result:** New `tessera.matmul` consuming the pre-transpose tensors directly, with `transposeA` and/or `transposeB` set to `true`. Existing transpose flags on the matmul are OR'd with the detected flags.

---

## 6. Effect Annotation

The `EffectAnnotationPass` (flag: `--tessera-effect-annotation`) attaches `tessera.effect` string attributes to `func.func` operations. This attribute drives `@jit(deterministic=True)` enforcement and `DistributionLoweringPass` collective-insertion decisions.

**Inference rules applied in order:**

| Condition | Effect assigned |
|-----------|----------------|
| Body contains `tessera.flash_attn` with `dropout_p != 0.0` | `random` |
| Body contains `tessera.copy` | `memory` |
| Any argument has `tessera.effect = "write"` or `"reduce_*"` attribute | `memory` |
| Body contains `func.call` to external non-tessera function | `io` |
| None of the above | `pure` |

**Validation:** If a function already has `tessera.effect = "pure"` and the inferred level is higher, the pass emits an error and signals failure (enforcing the `@jit(deterministic=True)` contract).

---

## 7. Module-Level Conventions

### 7.1 Module version attribute

`VerifyTesseraIR.cpp` checks for the `tessera.version` attribute on the `ModuleOp`. This is set by `GraphIRBuilder` at emission time.

```mlir
module @my_model attributes {tessera.version = "1.0"} {
  func.func @step(...) attributes {tessera.effect = "pure"} {
    ...
  }
}
```

### 7.2 Shard attributes on function arguments

```mlir
func.func @step(
  %W: tensor<256x256xbf16> {tessera.shard = {axes = ["tp"], dims = [1]}},
  %X: tensor<128x256xbf16> {tessera.shard = {axes = ["dp"], dims = [0]}},
  %Y: tensor<128x256xf32>  {tessera.effect = "write"}
) attributes {tessera.effect = "memory"} {
  %r = tessera.matmul %X, %W : (tensor<128x256xbf16>, tensor<256x256xbf16>) -> tensor<128x256xf32>
  ...
}
```

---

## 8. Complete Graph IR Example

End-to-end Graph IR for a sharded GEMM step:

```mlir
module @sharded_step attributes {tessera.version = "1.0"} {

  func.func @step(
      %W: tensor<256x256xbf16> {tessera.shard = {axes = ["tp"], dims = [0]}},
      %X: tensor<128x256xbf16> {tessera.shard = {axes = ["dp"], dims = [0]}},
      %Y: tensor<128x256xf32>  {tessera.effect = "write"}
  ) attributes {tessera.effect = "memory"} {
    // Graph IR encodes intent: Y = X @ W
    %r = tessera.matmul %X, %W
        {tile_k = 64 : i64}
        : (tensor<128x256xbf16>, tensor<256x256xbf16>) -> tensor<128x256xf32>
    return
  }

  func.func @attn_fwd(
      %Q: tensor<2x8x512x64xbf16>,
      %K: tensor<2x8x512x64xbf16>,
      %V: tensor<2x8x512x64xbf16>
  ) -> tensor<2x8x512x64xbf16> attributes {tessera.effect = "pure"} {
    %o = tessera.flash_attn %Q, %K, %V
        {head_dim = 64 : i64, causal = true,
         tessera.tile_q = 64 : i32, tessera.tile_kv = 64 : i32}
        : (tensor<2x8x512x64xbf16>, tensor<2x8x512x64xbf16>, tensor<2x8x512x64xbf16>)
       -> tensor<2x8x512x64xbf16>
    return %o : tensor<2x8x512x64xbf16>
  }

}
```
