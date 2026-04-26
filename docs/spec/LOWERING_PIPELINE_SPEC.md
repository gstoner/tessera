---
status: Normative
classification: Normative
last_updated: 2026-04-26
---

# Tessera Lowering Pipeline Specification
**Status:** Normative — grounded in `src/transforms/lib/` and `src/tile_opt_fa4/lib/` Phase 1–3 implementations  
**Last updated:** April 26, 2026  
**Cross-references:** `docs/spec/COMPILER_REFERENCE.md` §Pass Pipeline Registry, `docs/spec/GRAPH_IR_SPEC.md`, `docs/spec/TARGET_IR_SPEC.md`

---

## 1. Overview

Tessera provides two named lowering pipelines, each composed of a fixed sequence of MLIR passes:

| Pipeline | CLI flag | Target | Phase |
|----------|----------|--------|-------|
| `tessera-lower-to-x86` | `--tessera-lower-to-x86` | CPU (x86 AMX / AVX-512) | 2 |
| `tessera-lower-to-gpu` | `--tessera-lower-to-gpu` | NVIDIA GPU (SM_80+) | 3 |

Both pipelines start from the same Graph IR input (emitted by `@tessera.jit`) and produce different backend-specific IR. The IR stack at each stage is:

```
Graph IR (tessera dialect)
  → [EffectAnnotationPass]         tessera.effect attrs on func.func
  → [CanonicalizeTesseraIRPass]    fusion patterns applied
  → [DistributionLoweringPass]     tessera.shard → schedule.mesh.*
  → [TilingPass]                   tessera.matmul → scf.for + tensor slices   ← x86 only
  → [TileToX86Pass]                tiled matmul → func.call @tessera_x86_*    ← x86 only
  → [TileIRLoweringPass]           schedule.mesh.region → tile.* + attn.*     ← GPU only
  → [WarpSpecializationPass]       warp role assignment + queue barriers       ← GPU only
  → [AsyncCopyLoweringPass]        tile.async_copy → TMA / cp.async           ← GPU only
  → [NVWGMMALoweringPass]          tile.mma → wgmma.mma_async PTX             ← GPU only
  → [NVTMADescriptorPass]          TMA descriptor hoisting + mbarrier init     ← GPU only
  → [NVFlashAttnKernelEmitter]     FA-4 kernel finalisation                    ← GPU only
```

---

## 2. Named Pipelines

### 2.1 `tessera-lower-to-x86`

Registered pass sequence (executed in this order):

1. `tessera-effect-annotation`
2. `tessera-canonicalize`
3. `tessera-distribution-lowering`
4. `tessera-tiling`
5. `tessera-tile-to-x86`

### 2.2 `tessera-lower-to-gpu`

Registered pass sequence (executed in this order):

1. `tessera-effect-annotation`
2. `tessera-canonicalize`
3. `tessera-distribution-lowering`
4. `tessera-tile-ir-lowering`
5. `tessera-warp-specialization`
6. `tessera-async-copy-lowering`
7. `tessera-nvwgmma-lowering`
8. `tessera-nvtma-descriptor`
9. `tessera-nvflash-attn-emitter`

---

## 3. Pass Specifications

Each pass entry covers: purpose, CLI flag, input IR contract, output IR contract, invariants, pass options, and an IR before/after example.

---

### 3.1 `EffectAnnotationPass`

**File:** `src/transforms/lib/EffectAnnotationPass.cpp`  
**CLI flag:** `--tessera-effect-annotation`  
**Pipeline position:** 1 (both pipelines)

#### Purpose

Infers the side-effect class of each `func.func` in the module and attaches `tessera.effect` as a string function attribute. This annotation is consumed downstream by `DistributionLoweringPass` (collective insertion) and by Python-side `@jit(deterministic=True)` validation.

#### Input IR contract

- Valid `ModuleOp` containing `func.func` operations.
- `tessera.*` ops may appear in function bodies.
- Some functions may already have a `tessera.effect` attribute (set by `GraphIRBuilder` for `deterministic=True` functions).

#### Output IR contract

- Every `func.func` in the module has a `tessera.effect` string attribute.
- Attribute value is one of: `"pure"`, `"random"`, `"memory"`, `"io"`.
- No ops are modified or reordered.

#### Effect inference rules

Applied in order; the highest-level effect found wins (lattice join):

| Condition in function body | Effect level |
|---------------------------|--------------|
| `tessera.flash_attn` with `dropout_p` attr present and `!= 0.0` | `random` |
| `tessera.copy` op | `memory` |
| Any argument with `tessera.effect = "write"` or `"reduce_*"` attribute | `memory` |
| `func.call` to external non-tessera function | `io` |
| None of the above | `pure` |

#### Invariants

- **Pre-condition:** `tessera.effect` attribute, if already set, must equal `"pure"`. If it is set to any other value, the pass treats it as an override and skips inference for that function.
- **Post-condition:** If a function's body infers an effect level higher than `"pure"` but the function already carries `tessera.effect = "pure"`, the pass emits an error and signals pipeline failure. This enforces the `@jit(deterministic=True)` contract.
- The pass does not modify any ops — it is annotation-only.

#### IR example

**Before:**
```mlir
func.func @stable_fwd(%x: tensor<128x256xbf16>) -> tensor<128x256xbf16> {
  %r = tessera.matmul %x, %x : (tensor<128x256xbf16>, tensor<128x256xbf16>) -> tensor<128x256xf32>
  return %r : tensor<128x256xf32>
}
```

**After:**
```mlir
func.func @stable_fwd(%x: tensor<128x256xbf16>) -> tensor<128x256xbf16>
    attributes {tessera.effect = "pure"} {
  %r = tessera.matmul %x, %x : (tensor<128x256xbf16>, tensor<128x256xbf16>) -> tensor<128x256xf32>
  return %r : tensor<128x256xf32>
}
```

---

### 3.2 `CanonicalizeTesseraIRPass`

**File:** `src/transforms/lib/CanonicalizeTesseraIR.cpp`  
**CLI flag:** `--tessera-canonicalize`  
**Pipeline position:** 2 (both pipelines)

#### Purpose

Applies four greedy rewrite patterns to simplify and fuse Graph IR ops. Runs `applyPatternsAndFoldGreedily` — patterns may fire repeatedly until fixed point.

#### Input IR contract

- Valid tessera dialect ops in `func.func` bodies.
- `tessera.effect` attributes already set (by pass 3.1).

#### Output IR contract

- All `tessera.transpose → tessera.matmul` chains replaced by `tessera.matmul` with `transposeA`/`transposeB` flags.
- All `tessera.matmul → tessera.add → tessera.gelu` chains replaced by `tessera.fused_epilogue {Gelu}`.
- All `tessera.conv2d_nhwc → tessera.relu` chains replaced by `tessera.conv2d_nhwc {epilogue=Relu}`.
- All `tessera.flash_attn` ops with `dropout_p = 0.0` have the `dropout_p` attribute removed.
- No `tessera.transpose` ops remain whose consumers are `tessera.matmul`.

#### Patterns (see `GRAPH_IR_SPEC.md §5` for full details)

| Pattern | Benefit | Match | Result |
|---------|---------|-------|--------|
| `FuseMatmulBiasGELU` | 2 | `gelu(add(matmul(A,B), bias))` | `fused_epilogue(A,B,bias, Gelu)` |
| `FuseConvRelu` | 2 | `relu(conv2d_nhwc(...))` | `conv2d_nhwc(..., epilogue=Relu)` |
| `DropoutZeroSimplify` | 1 | `flash_attn {dropout_p=0.0}` | `flash_attn` without `dropout_p` |
| `TransposeIntoMatmul` | 1 | `matmul(transpose(A), B)` or `matmul(A, transpose(B))` | `matmul(A, B, transposeA/B=true)` |

#### Invariants

- Idempotent: running the pass twice produces the same result.
- Does not change the mathematical semantics of any op.

---

### 3.3 `DistributionLoweringPass`

**File:** `src/transforms/lib/DistributionLoweringPass.cpp`  
**CLI flag:** `--tessera-distribution-lowering`  
**Pass options:** `--mesh-axes=<str>`, `--mesh-sizes=<str>`  
**Pipeline position:** 3 (both pipelines)

#### Purpose

Converts `tessera.shard` argument attributes on `func.func` arguments into `schedule.mesh.define` + `schedule.mesh.region` ops that wrap the function body. Bridges from Graph IR (tessera dialect) to Schedule IR (schedule dialect).

#### Pass options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mesh-axes` | `string` | `""` | Comma-separated mesh axis names, e.g. `"dp,tp"`. Overrides per-arg shard attrs. |
| `--mesh-sizes` | `string` | `""` | Comma-separated axis sizes matching `--mesh-axes`, e.g. `"4,2"`. |

If a function has no `tessera.shard` attributes and no pass options are provided, the function is left unchanged.

#### Input IR contract

- `func.func` arguments may carry `tessera.shard = {axes = [...], dims = [...]}` attributes (set by `GraphIRBuilder`).

#### Output IR contract

- `tessera.shard` attributes removed from all function arguments.
- `schedule.mesh.define` emitted at the top of the function body.
- Original function body wrapped in `schedule.mesh.region` with a `schedule.yield` terminator.
- Function body ops unchanged.

#### Invariants

- Only processes functions with at least one `tessera.shard` argument attribute (or explicit pass options).
- Does not modify the ops inside the mesh region.
- The `schedule.mesh.define` dims and axis_names must reflect all unique axes found in the function's shard attributes.

#### IR example

**Before:**
```mlir
func.func @step(
    %a: tensor<128x256xbf16> {tessera.shard = {axes = ["dp"], dims = [0]}}
) attributes {tessera.effect = "memory"} {
  %0 = tessera.matmul %a, %a : (tensor<128x256xbf16>, tensor<128x256xbf16>) -> tensor<128x256xf32>
  return
}
```

**After:**
```mlir
func.func @step(%a: tensor<128x256xbf16>) attributes {tessera.effect = "memory"} {
  schedule.mesh.define {dims = [4], axis_names = ["dp"]}
  schedule.mesh.region {mesh = @dp, axis = "dp"} {
    %0 = tessera.matmul %a, %a : (tensor<128x256xbf16>, tensor<128x256xbf16>) -> tensor<128x256xf32>
    schedule.yield
  }
  return
}
```

---

### 3.4 `TilingPass`

**File:** `src/transforms/lib/TilingPass.cpp`  
**CLI flag:** `--tessera-tiling`  
**Pass options:** `--tile-m=<int>`, `--tile-n=<int>`  
**Pipeline position:** 4 (x86 pipeline only)

#### Purpose

Tiles `tessera.matmul` ops into `scf.for` loop nests over the M and N output dimensions. Prepares matmul ops for x86 backend lowering by `TileToX86Pass`. Only ops with statically-shaped ranked tensor operands are tiled; dynamic or unranked ops are left unchanged.

#### Pass options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tile-m` | `int` | `16` | M-dimension tile size (rows per outer loop step). |
| `--tile-n` | `int` | `16` | N-dimension tile size (cols per outer loop step). |

#### Input IR contract

- `tessera.matmul` ops in function bodies (inside or outside `schedule.mesh.region`).
- Operands must be statically-shaped ranked tensors to be tiled.

#### Output IR contract

- Each `tessera.matmul %A, %B : tensor<MxKxeT>, tensor<KxNxeT> -> tensor<MxNxeT>` replaced by:
  ```mlir
  %init = tensor.empty() : tensor<MxNxeT>
  %C = scf.for %i = 0 to M step tile_m iter_args(%acc0 = %init) {
    %C1 = scf.for %j = 0 to N step tile_n iter_args(%acc1 = %acc0) {
      %a_sl = tensor.extract_slice %A[%i, 0][tile_m, K][1, 1]
      %b_sl = tensor.extract_slice %B[0, %j][K, tile_n][1, 1]
      %c_sl = tessera.matmul %a_sl, %b_sl
      %acc2 = tensor.insert_slice %c_sl into %acc1[%i, %j][tile_m, tile_n][1,1]
      scf.yield %acc2
    }
    scf.yield %C1
  }
  ```
- `tessera.matmul` ops with dynamic shapes are left unchanged.
- Ops other than `tessera.matmul` (e.g. `tessera.flash_attn`, `tessera.fused_epilogue`) are untouched.

#### Invariants

- All statically-shaped `tessera.matmul` ops have been expanded into tiled loops.
- Tile sizes are always divisors of the static dimensions (if not, the last tile may be smaller — boundary handling is implicit in `tensor.extract_slice`).

---

### 3.5 `TileToX86Pass`

**File:** `src/transforms/lib/TileToX86Pass.cpp`  
**CLI flag:** `--tessera-tile-to-x86`  
**Pass options:** `--prefer-amx=<bool>`  
**Pipeline position:** 5 (x86 pipeline only)

#### Purpose

Replaces `tessera.matmul` and `tessera.fused_epilogue` ops (with static BF16/F16 input tensors) with `func.call` to tessera x86 backend C functions. This is the final x86 lowering step that produces callable code against the pre-built x86 AMX/AVX-512 GEMM kernels.

#### Pass option

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--prefer-amx` | `bool` | `true` | If `true`, emit `tessera_x86_amx_gemm_bf16`. If `false`, always emit `tessera_x86_avx512_gemm_bf16`. |

#### x86 backend C functions called

| Function | Signature | Description |
|----------|-----------|-------------|
| `tessera_x86_amx_gemm_bf16` | `(i64 aPtr, i64 bPtr, i64 cPtr, i64 M, i64 N, i64 K, f32 beta)` | AMX BF16 GEMM |
| `tessera_x86_avx512_gemm_bf16` | same | AVX-512 BF16 GEMM fallback |
| `tessera_x86_epilogue_bias_fp32` | `(i64 cPtr, i64 biasPtr, i64 M, i64 N)` | Bias add |
| `tessera_x86_epilogue_bias_gelu_fp32` | same | Bias add + GELU |

#### Input IR contract

- `tessera.matmul` ops with static ranked BF16/F16 input tensors (typically tiled by `TilingPass`).
- `tessera.fused_epilogue` ops with static shapes.

#### Output IR contract

For each `tessera.matmul %A, %B : tensor<MxKxbf16>, tensor<KxNxbf16> -> tensor<MxNxf32>`:

1. `bufferization.to_memref %A` → `memref<MxKxbf16>`
2. `bufferization.to_memref %B` → `memref<KxNxbf16>`
3. `memref.alloc()` → `memref<MxNxf32>`
4. External C function declaration added to the module (once per unique function name).
5. `memref.extract_aligned_pointer_as_index` + `arith.index_cast` to extract raw `i64` pointers.
6. `func.call @tessera_x86_amx_gemm_bf16(aPtr, bPtr, cPtr, M, N, K, beta)`
7. `bufferization.to_tensor %C_buf` → `tensor<MxNxf32>`

For `tessera.fused_epilogue`: same GEMM lowering, followed by the appropriate epilogue C function call.

#### Invariants

- After this pass, no `tessera.matmul` or `tessera.fused_epilogue` ops remain for static BF16/F16 types.
- All required external C function declarations are present exactly once in the module.

---

### 3.6 `TileIRLoweringPass`

**File:** `src/transforms/lib/TileIRLoweringPass.cpp`  
**CLI flag:** `--tessera-tile-ir-lowering`  
**Pass options:** `--tile-q=<int>`, `--tile-kv=<int>`, `--sm=<int>`  
**Pipeline position:** 4 (GPU pipeline only)

#### Purpose

Lowers `schedule.mesh.region` bodies containing `tessera.flash_attn` into FA-4 Tile IR ops. Also handles `tessera.matmul` inside `mesh.region` bodies by emitting the `tile.async_copy` + `tile.mma` + `tile.wait_async` GPU tiling sequence.

#### Pass options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tile-q` | `int` | `64` | Q tile rows. Must match the GPU WGMMA tile granularity. |
| `--tile-kv` | `int` | `64` | KV tile cols. |
| `--sm` | `int` | `90` | Target SM version as integer (e.g. `90` for SM_90). Controls whether `CausalMaskOp` and `DropoutMaskOp` are emitted. |

#### Input IR contract

- `schedule.mesh.region` bodies containing `tessera.flash_attn` ops.
- `tessera.flash_attn` ops may carry `tessera.tile_q`, `tessera.tile_kv`, and `causal` attributes.
- `tessera.matmul` ops inside mesh regions.

#### Output IR contract

For `tessera.flash_attn(Q, K, V)`:

```mlir
// Async copy of Q and K tiles
%q_tile = tile.async_copy %Q {tile_rows = 64, tile_cols = 64}
%k_tile = tile.async_copy %K {tile_rows = 64, tile_cols = 64}
tile.wait_async

// Scaled dot product
%scores = tessera.attn.scaled_dot_product %q_tile, %k_tile scale = 0.125 : ...

// Optional causal mask (when causal=true)
%masked = tessera.attn.causal_mask %scores q_off = %q_offset kv_off = %kv_offset : ...

// Online softmax (FA-2 algorithm)
%new_acc, %new_m, %new_l = tessera.attn.online_softmax %masked, %running_m, %running_l, %acc_out

// V tile async copy + wait
%v_tile = tile.async_copy %V {tile_rows = 64, tile_cols = 64}
tile.wait_async

// LSE accumulation (final step, outside inner loop)
%output, %lse = tessera.attn.lse_accumulate %acc, %final_m, %final_l
```

For `tessera.matmul` inside a mesh region:
```mlir
%a_tile = tile.async_copy %A {tile_rows = 64, tile_cols = 64}
%b_tile = tile.async_copy %B {tile_rows = 64, tile_cols = 64}
tile.wait_async
%c_tile = tile.mma %a_tile, %b_tile : ...
```

#### Invariants

- `tessera.flash_attn` ops inside `schedule.mesh.region` are fully replaced by the FA-4 op sequence.
- `tessera.matmul` ops inside `schedule.mesh.region` are fully replaced by `tile.async_copy` + `tile.mma`.
- `tessera.flash_attn` ops outside `schedule.mesh.region` are left unchanged (handled differently — should not exist after `DistributionLoweringPass`).

---

### 3.7 `WarpSpecializationPass`

**File:** `src/tile_opt_fa4/lib/WarpSpecializationPass.cpp`  
**CLI flag:** `--tessera-warp-specialization`  
**Pipeline position:** 5 (GPU pipeline only)

#### Purpose

Assigns producer/consumer warp roles to the FA-4 Tile IR ops. Inserts `tessera.queue` barriers between producer (async data mover) and consumer (MMA compute) roles. This is required for the WGMMA warp specialization model on SM_90+.

#### Input IR contract

- FA-4 Tile IR ops (`tile.async_copy`, `tile.wait_async`, `tessera.attn.*`, `tile.mma`) inside function bodies.

#### Output IR contract

- Function body split into `tessera.schedule.warp {role="producer"}` and `tessera.schedule.warp {role="consumer"}` regions.
- `tessera.queue.create`, `tessera.queue.push`, `tessera.queue.pop` ops inserted at producer/consumer boundaries.
- `tile.async_copy` and `tile.wait_async` ops enclosed in the `producer` region.
- `tessera.attn.*` compute ops and `tile.mma` ops enclosed in the `consumer` region.

#### Key design contract

Warp role separation is **structural, not advisory**. The backend allocates separate register files and barrier (mbarrier) slots per role. Producer warps are dedicated to TMA prefetch; consumer warps run WGMMA MMA ops. They never execute the other role's code.

#### IR example

**Before:**
```mlir
%q_tile = tile.async_copy %Q {tile_rows = 64, tile_cols = 64}
tile.wait_async
%scores = tessera.attn.scaled_dot_product %q_tile, %k_tile scale = 0.125 : ...
```

**After:**
```mlir
%q = tessera.queue.create : !tessera.queue.type
tessera.schedule.warp {role = "producer"} {
  %q_tile = tile.async_copy %Q {tile_rows = 64, tile_cols = 64}
  tile.wait_async
  %tok = tessera.queue.push %q, %q_tile : ...
}
tessera.schedule.warp {role = "consumer"} {
  %q_tile = tessera.queue.pop %q, %dep_tok : ...
  %scores = tessera.attn.scaled_dot_product %q_tile, %k_tile scale = 0.125 : ...
}
```

---

### 3.8 `AsyncCopyLoweringPass`

**File:** `src/tile_opt_fa4/lib/AsyncCopyLoweringPass.cpp`  
**CLI flag:** `--tessera-async-copy-lowering`  
**Pipeline position:** 6 (GPU pipeline only)

#### Purpose

Lowers `tile.async_copy` ops to either TMA descriptor-based async copies (SM_90+) or `tessera.cp_async.*` ops (SM_80/86/89). The target SM version is read from the `tessera.target_sm` module attribute set by `@jit(target=GPUTargetProfile(...))`.

#### Input IR contract

- `tile.async_copy` ops inside warp-specialized producer regions.

#### Output IR contract

**For SM_90+:**
```mlir
tessera.tma.async_copy %descriptor, %smem_buf, %mbarrier : ...
```

**For SM_80/86/89:**
```mlir
tessera.cp_async.cg %smem_buf, %gmem_ptr {size = 16} : ...
```

#### Invariants

- All `tile.async_copy` ops are replaced.
- `tile.wait_async` ops are replaced by appropriate barrier ops (`tessera.tma.wait_async` or `tessera.cp_async.wait_group`).

---

### 3.9 `NVWGMMALoweringPass`

**File:** `src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVWGMMALoweringPass.cpp`  
**CLI flag:** `--tessera-nvwgmma-lowering`  
**Pipeline position:** 7 (GPU pipeline only)

#### Purpose

Lowers `tile.mma` ops to WGMMA (Warpgroup Matrix Multiply Accumulate) inline PTX (`tessera.nvgpu.wgmma.mma_async`) for SM_90+, or falls back to legacy WMMA for SM_80/86/89.

#### Input IR contract

- `tile.mma` ops inside warp-specialized consumer regions.
- `tessera.target_sm` module attribute present.

#### Output IR contract

**SM_90+ (WGMMA):**
```mlir
tessera.nvgpu.wgmma.mma_async %a_desc, %b_desc, %c_acc
    {m = 64, n = 64, k = 16, dtype = "bf16"} : ...
```

**SM_80/86/89 (WMMA fallback):**
```mlir
tessera.nvgpu.wmma.mma %a_frag, %b_frag, %c_frag
    {m = 16, n = 16, k = 16, dtype = "bf16"} : ...
```

#### Invariants

- All `tile.mma` ops replaced by hardware-specific MMA intrinsics.
- SM version gating is strict: no WGMMA ops emitted when `tessera.target_sm < 90`.

---

### 3.10 `NVTMADescriptorPass`

**File:** `src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVTMADescriptorPass.cpp`  
**CLI flag:** `--tessera-nvtma-descriptor`  
**Pipeline position:** 8 (GPU pipeline only)

#### Purpose

Hoists TMA descriptor setup to the kernel preamble and assigns mbarrier slots. TMA descriptors describe how global memory tiles are staged into shared memory. They must be constructed once per kernel launch (not once per tile loop iteration).

#### Key design contract

TMA descriptors are generated **once per kernel**, not once per tile. `cp.async.bulk.tensor` calls in the tile loop reference the descriptor; they do not rebuild it.

#### Input IR contract

- `tessera.tma.async_copy` ops referencing tensor operands.

#### Output IR contract

- TMA descriptor setup ops hoisted to kernel preamble.
- `cp.async.bulk.tensor.1d` (or 2d/3d) emitted in tile loop.
- `mbarrier.init`, `mbarrier.arrive`, `mbarrier.wait` sequences inserted at correct points.

**Kernel preamble example:**
```mlir
// Preamble (hoisted by NVTMADescriptorPass)
%q_desc = tessera.tma.make_descriptor %Q_global {tile_shape = [64, 64]} : ...
%mbar_0 = tessera.mbarrier.init {count = 1} : ...

// In tile loop (after hoisting)
tessera.tma.bulk_copy %q_desc, %smem_q, %mbar_0 : ...
tessera.mbarrier.arrive %mbar_0 : ...
tessera.mbarrier.wait %mbar_0 {phase = 0} : ...
```

---

### 3.11 `NVFlashAttnKernelEmitter`

**File:** `src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVFlashAttnKernelEmitter.cpp`  
**CLI flag:** `--tessera-nvflash-attn-emitter`  
**Pipeline position:** 9 (GPU pipeline only)

#### Purpose

Finalises the FA-4 kernel. Resolves the attention scale sentinel (replaces the `1/sqrt(D)` placeholder with the concrete float value), emits the full mbarrier arrive/wait sequence for double-buffering, and attaches CUDA launch bounds as `nvvm.maxntidx` attributes.

#### Input IR contract

- Full warp-specialized, descriptor-hoisted FA-4 kernel with `tessera.attn.*` ops.
- `tessera.flash_attn` `scale = -1.0` sentinel value indicating "auto-compute from head_dim".

#### Output IR contract

- `scale` sentinel resolved to concrete `1 / sqrt(head_dim)` float constant.
- Complete mbarrier arrive/wait synchronisation sequence present for all double-buffer stages.
- `nvvm.maxntidx` annotation attached to the kernel function: `warps_per_cta * 32` threads.
- `nvvm.kernel` attribute set on the function to mark it as a CUDA kernel entry point.

#### Invariants

- No `tessera.attn.scaled_dot_product` ops remain with a sentinel scale value.
- All FA-4 attn ops are enclosed in a complete mbarrier synchronisation region.
- The emitted kernel is directly translatable to PTX by LLVM's NVPTX backend.

---

## 4. Pass Ordering Constraints

The following ordering constraints are hard requirements (violating them produces incorrect IR):

| Constraint | Reason |
|-----------|--------|
| `EffectAnnotation` before `DistributionLowering` | Distribution pass reads `tessera.effect` to identify gradient tensors for collective insertion. |
| `CanonicalizeTesseraIR` before `TilingPass` | Transpose flags must be folded before tiling to avoid tiling transposed ops incorrectly. |
| `DistributionLowering` before `TileIRLowering` | Tile IR lowering operates on `schedule.mesh.region` bodies — these must exist before Tile IR lowering. |
| `TileIRLowering` before `WarpSpecialization` | Warp specialization assigns roles to tile ops — these ops don't exist until after Tile IR lowering. |
| `WarpSpecialization` before `AsyncCopyLowering` | Async copy lowering converts `tile.async_copy` inside producer regions — the regions must exist first. |
| `AsyncCopyLowering` before `NVTMADescriptor` | TMA descriptor hoisting operates on `tessera.tma.async_copy` ops — these are emitted by async copy lowering. |
| `NVWGMMALowering` before `NVFlashAttnKernelEmitter` | Kernel emitter finalises mbarrier slots, which depend on the WGMMA op structure. |

---

## 5. IR Layer Transitions Summary

| Transition | From | To | Pass |
|-----------|------|----|------|
| Graph IR → Graph IR + effects | `tessera.*` | `func.func {tessera.effect}` | `EffectAnnotationPass` |
| Graph IR → canonicalised Graph IR | `tessera.*` chains | fused `tessera.*` | `CanonicalizeTesseraIRPass` |
| Graph IR → Schedule IR | `tessera.shard` attrs | `schedule.mesh.*` | `DistributionLoweringPass` |
| Schedule IR → Tiled Graph IR (x86) | `tessera.matmul` | `scf.for + tensor.*` | `TilingPass` |
| Tiled Graph IR → x86 calls | `tessera.matmul/fused_epilogue` | `func.call @tessera_x86_*` | `TileToX86Pass` |
| Schedule IR → Tile IR (GPU) | `schedule.mesh.region + tessera.flash_attn` | `tile.* + tessera.attn.*` | `TileIRLoweringPass` |
| Tile IR → Warp-specialised IR | `tile.*` | `tessera.schedule.warp + tessera.queue.*` | `WarpSpecializationPass` |
| Warp IR → TMA/cp.async IR | `tile.async_copy` | `tessera.tma.*` or `tessera.cp_async.*` | `AsyncCopyLoweringPass` |
| TMA IR → WGMMA PTX IR | `tile.mma` | `tessera.nvgpu.wgmma.*` | `NVWGMMALoweringPass` |
| WGMMA IR → kernel IR | `tessera.tma.*` scattered | hoisted descriptors + mbarriers | `NVTMADescriptorPass` |
| Kernel IR → final PTX-ready IR | sentinel scale, partial barriers | concrete values + full mbarrier seq | `NVFlashAttnKernelEmitter` |
