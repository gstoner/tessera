# Tessera Target IR Specification
**Status:** Normative — grounded in `src/tile_opt_fa4/`, `src/programming_model/ir/schedule/`, and `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` Phase 2–3 implementations  
**Last updated:** April 26, 2026  
**Cross-references:** `docs/spec/GRAPH_IR_SPEC.md`, `docs/spec/LOWERING_PIPELINE_SPEC.md`

---

## 1. Overview

The Target IR layer sits at the bottom of the four-layer IR stack, immediately above backend code generation:

```
Schedule IR  (schedule dialect — mesh regions, pipeline stages)
     │  [TileIRLoweringPass, WarpSpecializationPass]
     ▼
Tile IR      (tile.* ops — async copy, MMA, barriers; tessera.attn.* FA-4 ops)
     │  [AsyncCopyLoweringPass, NVWGMMALoweringPass, NVTMADescriptorPass]
     ▼
Target IR    (tessera.nvgpu.*, tessera.tma.*, tessera.cp_async.*, mbarrier ops)
     │  [NVFlashAttnKernelEmitter → LLVM NVPTX backend]
     ▼
PTX / binary
```

This document specifies four dialect layers that together constitute the Target IR:

1. **Schedule dialect** — mesh and pipeline region structure (Schedule IR layer)
2. **`tessera.attn` dialect** — FA-4 FlashAttention ops (Tile IR layer, Phase 3)
3. **`tessera.queue` dialect** — tile queue synchronisation (Tile IR layer, Phase 3)
4. **`tile.*` ops** — generic tile async copy and MMA primitives (Tile IR layer)

The x86 Target IR (AMX/AVX-512 C function calls) is documented separately in `docs/architecture/tessera_target_ir_usage_guide.md`.

---

## 2. Schedule Dialect

**TableGen:** `src/programming_model/ir/schedule/ScheduleMeshPipelineOps.td`  
**Dialect name:** `schedule`  
**C++ namespace:** `::tessera::schedule`

The Schedule dialect bridges Graph IR and Tile IR by expressing **where** computation runs (mesh placement) and **when** (pipeline staging) without committing to hardware-specific tile ops.

---

### 2.1 `schedule.mesh.define`

Declares a logical device mesh. Must appear before any `schedule.mesh.region` that references it.

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$dims` | `ArrayAttr` | Axis sizes, e.g. `[4, 2]` for a 4×2 mesh. |
| `$axis_names` | `ArrayAttr` | Axis name strings, e.g. `["dp", "tp"]`. Must have same length as `$dims`. |

**Has verifier:** Yes — checks that `dims` and `axis_names` have matching lengths and all sizes are positive.

**MLIR text:**
```mlir
schedule.mesh.define {dims = [4, 2], axis_names = ["dp", "tp"]}
```

---

### 2.2 `schedule.mesh.region`

Defines a region of computation that executes within one axis of a logical mesh. The region body is replicated across all ranks along the named axis.

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$mesh` | `SymbolRefAttr` | Reference to the mesh defined by `schedule.mesh.define`, e.g. `@dp`. |
| `$axis` | `StrAttr` | The axis name this region is distributed over, e.g. `"dp"`. |

**Regions:** `$body` — `AnyRegion`. Must be terminated by `schedule.yield`.

**Has verifier:** Yes — checks that the referenced mesh symbol exists and `axis` is a valid axis name in that mesh.

**MLIR text:**
```mlir
schedule.mesh.region {mesh = @dp, axis = "dp"} {
  %0 = tessera.matmul %A, %B : (tensor<128x256xbf16>, tensor<256x256xbf16>) -> tensor<128x256xf32>
  schedule.yield
}
```

---

### 2.3 `schedule.pipeline.region`

Defines a pipeline-parallel region. The region body contains multiple `schedule.stage` ops, each mapped to a device subset. Phase 4 introduces the pass that lowers this.

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$schedule` | `StrAttr` | Pipeline schedule strategy. Valid values: `"1f1b"` (one-forward-one-backward), `"interleaved"`. |
| `$micro_batches` | `I32Attr` | Number of micro-batches. For `"interleaved"`, must be `>= 2 * num_stages`. |

**Regions:** `$body` — contains `schedule.stage` ops.

**Has verifier:** Yes.

**MLIR text:**
```mlir
schedule.pipeline.region {schedule = "1f1b", micro_batches = 8 : i32} {
  schedule.stage {devices = [0, 1]} {
    %0 = tessera.matmul %x, %W0 : ...
    schedule.yield %0
  }
  schedule.stage {devices = [2, 3]} {
    %1 = tessera.matmul %prev, %W1 : ...
    schedule.yield %1
  }
}
```

---

### 2.4 `schedule.stage`

One stage in a pipeline-parallel region. Defines the devices responsible for this stage and the computation it performs.

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$devices` | `ArrayAttr` | Device (rank) indices assigned to this stage. |

**Regions:** `$body` — computation ops + `schedule.yield` terminator.

**Has verifier:** Yes.

---

### 2.5 `schedule.yield`

Terminator for all `schedule.*` region bodies. Returns zero or more values from the region.

**Traits:** `Pure`, `Terminator`, `ReturnLike`

**Arguments:** `$values : Variadic<AnyType>`

**MLIR text:**
```mlir
// Zero-value yield (most common in mesh regions)
schedule.yield

// Value-returning yield (pipeline stages)
schedule.yield %result : tensor<128x256xf32>
```

---

## 3. `tessera.attn` Dialect — FA-4 FlashAttention Ops

**TableGen:** `src/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td` (v2.0)  
**Dialect name:** `tessera.attn`  
**C++ namespace:** `::tessera::attn`  
**Phase:** 3

These ops implement the FA-2 online softmax algorithm at the tile level. They are emitted by `TileIRLoweringPass` when lowering `tessera.flash_attn` (Graph IR) to Tile IR.

The FA-2 algorithm structure in Tile IR:

```
Outer loop over Q tiles:
  init running_m = -inf, running_l = 0, acc_out = 0
  Inner loop over KV tiles:
    %scores = tessera.attn.scaled_dot_product Q_tile, K_tile
    [%masked = tessera.attn.causal_mask %scores ...]       ← if causal=true
    [%masked = tessera.attn.dropout_mask %masked ...]      ← if dropout_p > 0
    %new_acc, %new_m, %new_l = tessera.attn.online_softmax %masked, %running_m, %running_l, %acc_out
    update running_m, running_l, acc_out
  End inner loop
  %output, %lse = tessera.attn.lse_accumulate %acc_out, %running_m, %running_l
  tessera.attn.lse.save %scores → %lse_saved   ← for backward pass
```

---

### 3.1 `tessera.attn.lse.save`

Saves the per-row log-sum-exp (LSE) values from the forward attention pass. LSE values are consumed by the backward pass.

**Traits:** None (has side effect — writes to LSE buffer)  
**Has verifier:** Yes (shape-aware check in `AttnOps.cpp`)

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$scores` | `AnyType` | Score tensor. Shape: `[tile_q, tile_kv]`. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$lse` | `AnyType` | Per-row log-sum-exp values. Shape: `[tile_q]`. |

**MLIR text:**
```mlir
%lse = tessera.attn.lse.save %scores : tensor<64x64xf32> -> tensor<64xf32>
```

---

### 3.2 `tessera.attn.lse.load`

Loads saved LSE values for the backward pass.

**Traits:** `Pure`

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$lse` | `AnyType` | Per-row LSE values. Shape: `[tile_q]`. |

**MLIR text:**
```mlir
%lse = tessera.attn.lse.load : tensor<64xf32>
```

---

### 3.3 `tessera.attn.scaled_dot_product`

Computes `Q_tile · K_tile^T * scale` for one (Q-tile, K-tile) pair. This is the innermost operation of the flash attention inner loop.

The `scale` attribute is constant-folded at lowering time from `head_dim` on the parent `tessera.flash_attn` op. The Phase 5 autotuner stores `tile_q` and `tile_kv` as op attributes so it can retile without re-emitting Graph IR.

**Traits:** `Pure`  
**Has verifier:** Yes

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$query` | `AnyType` | Q tile. Shape: `[tile_q, d_k]`. |
| `$key` | `AnyType` | K tile. Shape: `[tile_kv, d_k]`. |
| `$scale` | `F32Attr` | Attention scale factor: `1/sqrt(d_k)`. Constant at lowering time. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$scores` | `AnyType` | Raw attention scores. Shape: `[tile_q, tile_kv]`. |

**Assembly format:**
```
$query , $key scale = $scale attr-dict : type($query) , type($key) -> type($scores)
```

**MLIR text:**
```mlir
%scores = tessera.attn.scaled_dot_product %q_tile, %k_tile scale = 0.125 : f32
    : tensor<64x64xbf16>, tensor<64x64xbf16> -> tensor<64x64xf32>
```

---

### 3.4 `tessera.attn.online_softmax`

Implements the FA-2 online (incremental) softmax. Processes one score tile and updates the running statistics needed to correctly normalise the accumulated output.

The two-pass online algorithm:
1. `new_m = max(running_m, rowmax(scores))`
2. Correction factor: `alpha = exp(running_m - new_m)`
3. `new_acc = alpha * acc_out + exp(scores - new_m) @ V_tile`
4. `new_l = alpha * running_l + rowsum(exp(scores - new_m))`

**Traits:** None (updates accumulator in-place conceptually — not `Pure`)  
**Has verifier:** Yes

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$scores` | `AnyType` | Current score tile (after optional causal/dropout masking). Shape: `[tile_q, tile_kv]`. |
| `$running_m` | `AnyType` | Per-row running maximum from previous iterations. Shape: `[tile_q]`. |
| `$running_l` | `AnyType` | Per-row running sum from previous iterations. Shape: `[tile_q]`. |
| `$acc_out` | `AnyType` | Accumulated output from previous iterations. Shape: `[tile_q, d_v]`. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$new_acc` | `AnyType` | Updated output accumulator. Shape: `[tile_q, d_v]`. |
| `$new_m` | `AnyType` | Updated per-row running maximum. Shape: `[tile_q]`. |
| `$new_l` | `AnyType` | Updated per-row running sum. Shape: `[tile_q]`. |

**MLIR text:**
```mlir
%new_acc, %new_m, %new_l = tessera.attn.online_softmax
    %scores, %running_m, %running_l, %acc_out
    : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>
```

---

### 3.5 `tessera.attn.lse_accumulate`

Finalises the FA-2 two-pass algorithm after all KV tiles have been processed. Divides the accumulated output by the final running sum and computes the true LSE value.

Operations: `output = acc / running_l`,  `lse = log(running_l) + running_m`

**Traits:** `Pure`

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$acc` | `AnyType` | Final accumulated output. Shape: `[tile_q, d_v]`. |
| `$running_m` | `AnyType` | Final per-row running maximum. Shape: `[tile_q]`. |
| `$running_l` | `AnyType` | Final per-row running sum. Shape: `[tile_q]`. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$output` | `AnyType` | Normalised attention output. Shape: `[tile_q, d_v]`. |
| `$lse` | `AnyType` | Log-sum-exp values for backward. Shape: `[tile_q]`. |

**MLIR text:**
```mlir
%output, %lse = tessera.attn.lse_accumulate %acc, %final_m, %final_l
    : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>
```

---

### 3.6 `tessera.attn.dropout_mask`

Generates a stochastic binary dropout mask for attention scores. Only emitted when `@jit` has `seed` set and the parent `tessera.flash_attn` has `dropout_p > 0`. On SM_90 WGMMA kernels, dropout is handled in the fused epilogue instead.

**Traits:** None (calls RNG — side-effecting)

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$scores` | `AnyType` | Score tile to mask. Shape: `[tile_q, tile_kv]`. |
| `$dropout_p` | `F32Attr` | Drop probability in `[0, 1)`. |
| `$seed` | `I64Attr` | RNG seed. Required — no default. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$masked_scores` | `AnyType` | Scores with dropout applied. Same shape as input. |

**Assembly format:**
```
$scores p = $dropout_p seed = $seed attr-dict
```

**MLIR text:**
```mlir
%masked = tessera.attn.dropout_mask %scores p = 0.100000 seed = 42 : tensor<64x64xf32>
```

---

### 3.7 `tessera.attn.causal_mask`

Applies a lower-triangular causal mask to the score tile. Emitted when `tessera.flash_attn(..., causal=True)`. The mask is structurally zero cost: it becomes a scalar comparison in the tiling loop and is folded into the WGMMA epilogue on SM_90.

**Traits:** `Pure`

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$scores` | `AnyType` | Score tile. Shape: `[tile_q, tile_kv]`. |
| `$q_offset` | `I64Attr` | Row offset of this Q tile in the full sequence (for correct causal boundary). |
| `$kv_offset` | `I64Attr` | Column offset of this KV tile in the full sequence. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$masked_scores` | `AnyType` | Causally-masked scores. Same shape. Positions where `kv_col > q_row` are set to `-inf`. |

**Assembly format:**
```
$scores q_off = $q_offset kv_off = $kv_offset attr-dict
```

**MLIR text:**
```mlir
%masked = tessera.attn.causal_mask %scores q_off = 0 kv_off = 64
    : tensor<64x64xf32>
```

---

## 4. `tessera.queue` Dialect — Tile Queue Synchronisation

**TableGen:** `src/tile_opt_fa4/dialects/tessera_queue/Queue.td` (v1.3)  
**Dialect name:** `tessera.queue`  
**C++ namespace:** `::tessera::queue`  
**Phase:** 3

The queue dialect provides the barrier primitives used by `WarpSpecializationPass` to synchronise producer warps (which prefetch tiles via TMA) and consumer warps (which run WGMMA compute). Queues are opaque handles; their physical implementation is SM-specific (mbarriers on SM_90).

---

### 4.1 Types

#### `!tessera.queue.type`

Opaque tile-queue handle. Created by `tessera.queue.create`, consumed by `tessera.queue.push` and `tessera.queue.pop`.

#### `!tessera.queue.token`

Opaque ordering token. Produced by `tessera.queue.push`, consumed by `tessera.queue.pop` to express ordering.

---

### 4.2 `tessera.queue.create`

Creates a new tile queue. Must appear before any `push` or `pop` that uses it.

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$q` | `!tessera.queue.type` | New queue handle. |

**MLIR text:**
```mlir
%q = tessera.queue.create : !tessera.queue.type
```

---

### 4.3 `tessera.queue.push`

Pushes a tile into the queue from the producer warp. Returns an ordering token consumed by the matching `pop`.

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$q` | `!tessera.queue.type` | Queue to push into. |
| `$tile` | `AnyType` | Tile value to enqueue. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$t` | `!tessera.queue.token` | Ordering token. |

**MLIR text:**
```mlir
%tok = tessera.queue.push %q, %q_tile : !tessera.queue.type, tensor<64x64xbf16>
```

---

### 4.4 `tessera.queue.pop`

Pops a tile from the queue in the consumer warp. Blocks until the producer has pushed the corresponding tile (via the token dependency).

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$q` | `!tessera.queue.type` | Queue to pop from. |
| `$dep` | `!tessera.queue.token` | Token from the matching `push`. Expresses producer→consumer ordering. |

**Results:**

| Result | Type | Description |
|--------|------|-------------|
| `$tile` | `AnyType` | The dequeued tile value. Same type as the pushed tile. |

**MLIR text:**
```mlir
%q_tile = tessera.queue.pop %q, %tok : !tessera.queue.type, !tessera.queue.token -> tensor<64x64xbf16>
```

---

## 5. `tile.*` Ops — Generic Tile Primitives

These ops are not bound to a single registered dialect in the current implementation — they are string-based `OperationState` ops created by `TileIRLoweringPass` using the `"tile.*"` op name convention. They are validated by convention rather than a verifier.

---

### 5.1 `tile.async_copy`

Initiates an asynchronous copy of a tile from global memory into shared/L1 memory. Does not block — `tile.wait_async` must be called before the result is used.

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `src` | `TensorType` | Source tensor (global memory). |
| `tile_rows` | `I64Attr` | Number of rows in the tile. |
| `tile_cols` | `I64Attr` | Number of columns in the tile. |

**Results:** Same tensor type as `src` (tile-shaped).

**MLIR text:**
```mlir
%q_tile = tile.async_copy %Q {tile_rows = 64 : i64, tile_cols = 64 : i64}
    : tensor<512x64xbf16> -> tensor<64x64xbf16>
```

---

### 5.2 `tile.wait_async`

Drains all in-flight `tile.async_copy` operations. All async copies issued before this op are guaranteed complete after it.

**Arguments:** None  
**Results:** None

**MLIR text:**
```mlir
tile.wait_async
```

---

### 5.3 `tile.mma`

Performs a tile-granularity matrix multiply accumulate. Lowered to WGMMA PTX (`wgmma.mma_async.sync.aligned`) on SM_90+ or WMMA on earlier targets by `NVWGMMALoweringPass`.

**Arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `$a` | `TensorType` | A tile. Shape: `[M_tile, K_tile]`. |
| `$b` | `TensorType` | B tile. Shape: `[K_tile, N_tile]`. |

**Results:** `$c : TensorType` — output accumulator. Shape: `[M_tile, N_tile]`.

**MLIR text:**
```mlir
%c_tile = tile.mma %a_tile, %b_tile
    : tensor<64x64xbf16>, tensor<64x64xbf16> -> tensor<64x64xf32>
```

---

## 6. Target IR: NVIDIA-Specific Ops

These ops are emitted by the NVIDIA-backend lowering passes (`AsyncCopyLoweringPass`, `NVWGMMALoweringPass`, `NVTMADescriptorPass`) and are specific to the NVIDIA PTX / NVGPU target. They do not have a formal TableGen ODS definition in the current implementation — they are emitted as NVGPU dialect ops or inline PTX using `nvvm.*` attributes.

| Op | Description | Phase |
|----|-------------|-------|
| `tessera.tma.make_descriptor` | Creates a TMA tensor descriptor from a global tensor. Hoisted to kernel preamble by `NVTMADescriptorPass`. | 3 |
| `tessera.tma.bulk_copy` | `cp.async.bulk.tensor` — copies a tile from global to shared memory using the TMA descriptor. | 3 |
| `tessera.tma.wait_async` | Waits for TMA async copies to complete (replaces `tile.wait_async` after `AsyncCopyLoweringPass`). | 3 |
| `tessera.cp_async.cg` | `cp.async.cg` — 16-byte async copy to shared memory for SM_80/86/89 fallback. | 3 |
| `tessera.cp_async.wait_group` | Barrier for `cp.async` groups on SM_80/86/89. | 3 |
| `tessera.mbarrier.init` | Initialises an mbarrier with a thread count. | 3 |
| `tessera.mbarrier.arrive` | Signals mbarrier arrival (from producer warp). | 3 |
| `tessera.mbarrier.wait` | Waits on mbarrier (from consumer warp). Includes phase bit. | 3 |
| `tessera.nvgpu.wgmma.mma_async` | `wgmma.mma_async.sync.aligned` PTX — SM_90+ WGMMA MMA. | 3 |
| `tessera.nvgpu.wmma.mma` | Legacy WMMA — SM_80/86/89 fallback path. | 3 |

---

## 7. Complete Tile IR Example

A minimal FA-4 attention kernel in Tile IR before NVIDIA-specific lowering:

```mlir
module @flash_attn_sm90 attributes {tessera.version = "1.0", tessera.target_sm = 90 : i32} {

  func.func @flash_attn_fwd(
      %Q: tensor<2x8x512x64xbf16>,
      %K: tensor<2x8x512x64xbf16>,
      %V: tensor<2x8x512x64xbf16>
  ) -> tensor<2x8x512x64xbf16> attributes {tessera.effect = "pure", nvvm.kernel} {

    // Queue for producer→consumer tile handoff
    %kv_queue = tessera.queue.create : !tessera.queue.type

    // Producer warp: async tile prefetch
    tessera.schedule.warp {role = "producer"} {
      %q_tile = tile.async_copy %Q {tile_rows = 64 : i64, tile_cols = 64 : i64}
                  : tensor<512x64xbf16> -> tensor<64x64xbf16>
      %k_tile = tile.async_copy %K {tile_rows = 64 : i64, tile_cols = 64 : i64}
                  : tensor<512x64xbf16> -> tensor<64x64xbf16>
      tile.wait_async
      %tok_k = tessera.queue.push %kv_queue, %k_tile
                  : !tessera.queue.type, tensor<64x64xbf16>
    }

    // Consumer warp: compute
    tessera.schedule.warp {role = "consumer"} {
      %k_tile = tessera.queue.pop %kv_queue, %dep_tok
                  : !tessera.queue.type, !tessera.queue.token -> tensor<64x64xbf16>

      // Scaled dot product
      %scores = tessera.attn.scaled_dot_product %q_tile, %k_tile scale = 0.125 : f32
                  : tensor<64x64xbf16>, tensor<64x64xbf16> -> tensor<64x64xf32>

      // Causal mask (causal=true)
      %masked = tessera.attn.causal_mask %scores q_off = 0 kv_off = 0 : tensor<64x64xf32>

      // Online softmax update
      %new_acc, %new_m, %new_l = tessera.attn.online_softmax
          %masked, %running_m, %running_l, %acc_out
          : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>

      // LSE finalisation (after inner loop)
      %output, %lse = tessera.attn.lse_accumulate %new_acc, %new_m, %new_l
                        : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>

      // Save LSE for backward
      %lse_saved = tessera.attn.lse.save %scores : tensor<64x64xf32> -> tensor<64xf32>
    }

    return %output : tensor<2x8x512x64xbf16>
  }

}
```

---

## 8. Phase Coverage

| Dialect / Op | Phase introduced | Status |
|--------------|-----------------|--------|
| `schedule.mesh.define` | 2 | ✅ Implemented |
| `schedule.mesh.region` | 2 | ✅ Implemented |
| `schedule.pipeline.region` | 4 (designed Phase 2) | 🔲 Lowering pass in Phase 4 |
| `schedule.stage` | 4 (designed Phase 2) | 🔲 Lowering pass in Phase 4 |
| `schedule.yield` | 2 | ✅ Implemented |
| `tessera.attn.lse.save` | 1 (v1.3) | ✅ Implemented |
| `tessera.attn.lse.load` | 1 (v1.3) | ✅ Implemented |
| `tessera.attn.scaled_dot_product` | 3 (v2.0) | ✅ Implemented |
| `tessera.attn.online_softmax` | 3 (v2.0) | ✅ Implemented |
| `tessera.attn.lse_accumulate` | 3 (v2.0) | ✅ Implemented |
| `tessera.attn.dropout_mask` | 3 (v2.0) | ✅ Implemented |
| `tessera.attn.causal_mask` | 3 (v2.0) | ✅ Implemented |
| `tessera.queue.create/push/pop` | 3 | ✅ Implemented |
| `tile.async_copy` | 3 | ✅ Implemented (string-based) |
| `tile.wait_async` | 3 | ✅ Implemented (string-based) |
| `tile.mma` | 3 | ✅ Implemented (string-based) |
| `tessera.tma.*` | 3 | ✅ Implemented |
| `tessera.nvgpu.wgmma.*` | 3 | ✅ Implemented |
| `tessera.nvgpu.wmma.*` | 3 | ✅ Implemented (SM < 90 fallback) |
| NCCL collective ops | 4 | 🔲 Phase 4 |
| TPU StableHLO ops | 4 | 🔲 Phase 4 |
| ROCm MFMA full coverage | 6 | 🔲 Phase 6 |
