---
status: Normative
classification: Normative
authority: Tile IR op set and dialect semantics; defers Schedule IR and Target IR details to docs/spec/TARGET_IR_SPEC.md
last_updated: 2026-04-26
---

# Tessera Tile IR Specification (Normative)

**Version:** 0.3.0  
**Authority:** This document specifies the Tile IR op set, dialect structure, and verifier rules. For the full Target IR dialect (WGMMA, TMA, mbarrier) see `docs/spec/TARGET_IR_SPEC.md`. For the lowering passes that produce Tile IR from Schedule IR, see `docs/spec/LOWERING_PIPELINE_SPEC.md`.

---

## 1. Scope and Role

Tile IR is the **third layer** of the Tessera four-layer IR stack. It is produced by
`TileIRLoweringPass` from Schedule IR and consumed by the target backends (x86, NVIDIA,
ROCm, TPU).

```
Schedule IR  (schedule.* dialect)
     │
     ▼  TileIRLoweringPass
Tile IR      (tile.* + tessera.attn.* + tessera.queue.* + tessera.tcgen05.*)
     │
     ▼  NVWGMMALoweringPass / TileToX86Pass / ...
Target IR    (tessera.nvgpu.wgmma.*, tessera.tma.*, x86 intrinsics)
```

Tile IR is the layer at which:
- Explicit shared memory allocation appears (`tshared.alloc`)
- Warp roles are assigned (`tessera.schedule.warp {role="producer/consumer"}`)
- Async copy stages are made explicit (`tile.async_copy {stage=N}`)
- MMA operations are expressed (`tile.mma`)
- Producer/consumer ordering tokens are introduced (`tessera.queue.*`)
- FlashAttention sub-operations appear (`tessera.attn.*`)

Tile IR is **backend-agnostic**. Target-specific intrinsics are in Target IR.

---

## 2. IR Structure

Tile IR follows standard MLIR structural conventions:

- **Module** → contains one or more `func.func` ops (kernel functions)
- **Function** → SSA; all values are defined before use; no implicit captures
- **Blocks** → basic blocks with explicit successors; tile regions may introduce structured control flow
- **Ops** → each op has typed operands, results, regions (for structured control), and an `attributes` dictionary

### 2.1 Tile IR Attributes

Tile IR ops carry attributes that encode hardware-relevant parameters:

| Attribute | Type | Example | Meaning |
|-----------|------|---------|---------|
| `stage` | `i64` | `{stage = 0}` | Pipeline stage index for async copy double-buffering |
| `vector` | `i64` | `{vector = 16}` | Vector width in elements for async copy |
| `swizzle` | `string` | `{swizzle = "xor"}` | Shared memory swizzle pattern for bank conflict elimination |
| `order` | `string` | `{order = "tree"}` | Reduction tree order for deterministic results |
| `role` | `string` | `{role = "producer"}` | Warp role within a CTA |
| `tile_q` | `i64` | `{tile_q = 64}` | Query tile size (FlashAttention) |
| `tile_kv` | `i64` | `{tile_kv = 64}` | Key/value tile size (FlashAttention) |

### 2.2 Memory Space Encoding

Tile IR uses MLIR memory space integers to distinguish memory tiers:

| Memory space | Integer | Hardware mapping |
|-------------|---------|-----------------|
| Global (HBM) | `0` | Device DRAM |
| Shared (SMEM) | `1` | SM-local SRAM (per-CTA) |
| Tensor Memory (TMEM) | `2` | SM_100+ MMA accumulator (Blackwell) |
| Register | `3` | Per-thread register file |

Example: `memref<128x64xf16, 1>` = 128×64 f16 array in shared memory.

---

## 3. Core `tile.*` Operations

### 3.1 `tshared.alloc`

Allocates a buffer in shared memory. Must appear in the kernel preamble.

```mlir
%smem = tshared.alloc[f16](128, 64) {swizzle = "xor"}
    : memref<128x64xf16, 1>
```

**Attributes:**
- `swizzle` (optional, default `"none"`): `"xor"` eliminates bank conflicts for 128-bit access patterns; `"none"` disables swizzling

**Verifier rules:**
- Result type must be `memref<...x..., 1>` (memory space 1)
- All dimensions must be static integers

### 3.2 `tile.async_copy`

Issues an asynchronous copy from global memory to shared memory. Semantics: the copy
is initiated but not complete until the matching `tile.wait_async` with the same `stage`
executes.

```mlir
tile.async_copy %global_src into %smem_dst {stage = 0, vector = 16}
    : memref<?x?xf16, 0> into memref<128x64xf16, 1>
```

**Attributes:**
- `stage` (required): pipeline stage index (0-based); used to interleave copies with compute
- `vector` (required for SM_90+): copy vector width in elements; must match TMA descriptor alignment

**Lowering targets:**
- SM_90+: `tessera.tma.async_copy` → `cp.async.bulk.tensor` PTX
- Below SM_90: `tessera.cp_async.shared.global` → `cp.async.ca.shared.global` PTX

**Verifier rules:**
- `stage` must be ≥ 0
- Source must be memory space 0 (global); destination must be memory space 1 (shared)
- A `tile.wait_async {stage = N}` must dominate every use of any value written by `tile.async_copy {stage = N}` within the same block scope

### 3.3 `tile.wait_async`

Waits for all in-flight `tile.async_copy` ops with the matching `stage` to complete.
Acts as a barrier for the specified pipeline stage.

```mlir
tile.wait_async {stage = 0}
```

**Verifier rules:**
- Every `tile.async_copy {stage = N}` in the enclosing function must have a corresponding `tile.wait_async {stage = N}`
- No `tile.wait_async` with a stage that has no corresponding `tile.async_copy` (dead barrier)

### 3.4 `tile.mma`

Matrix multiply-accumulate on tile-sized operands. Backend-agnostic at Tile IR level;
lowered to `tessera.nvgpu.wgmma.mma_async` (SM_90), `tessera.nvgpu.wmma.*` (SM_80), or
AMX `tile_dpbf16ps` (x86) by the appropriate target pass.

```mlir
%C_out = tile.mma %A, %B, %C_in
    : memref<64x64xf16, 1>, memref<64x64xf16, 1>, memref<64x64xf32, 1>
    -> memref<64x64xf32, 1>
```

**Verifier rules:**
- A and B must be in shared memory (space 1) or registers (space 3)
- C accumulator must match the output type; BF16/F16 inputs accumulate to F32
- Tile dimensions must satisfy hardware alignment (64×64 for BF16 WGMMA on SM_90)

### 3.5 `tile.reduce`

Performs a reduction within a tile. The `order` attribute determines whether the reduction
tree is canonicalized for determinism.

```mlir
%sum = tile.reduce<add> %input {order = "tree"}
    : memref<128xf32, 1> -> f32
```

**Supported reduction kinds:** `add`, `max`, `min`, `and`, `or`, `xor`

**`order` attribute:**
- `"tree"`: deterministic binary tree reduction (required for `@jit(deterministic=True)`)
- `"warp"`: warp-shuffle reduction (faster, may be non-associative for floats)

### 3.6 `tile.barrier`

CTA-wide barrier. All threads in the CTA must reach this op before any thread proceeds.
Corresponds to `__syncthreads()` in CUDA or `s_barrier` in ROCm.

```mlir
tile.barrier
```

**Verifier rules:** Must not appear inside a warp-role region tagged `role="producer"` or
`role="consumer"` in isolation — producer/consumer synchronization uses queue tokens instead.

---

## 4. `tessera.attn.*` — FA-4 Attention Dialect

The `tessera.attn` dialect implements the **FA-4 FlashAttention algorithm** at Tile IR
level. It is produced by `TileIRLoweringPass` when lowering `tessera.flash_attn` Graph IR
ops targeting SM_90+.

### 4.1 Dialect Definition

```
Dialect name: tessera.attn
C++ namespace: ::tessera::attn
Source: src/compiler/tile_opt_fa4/dialects/tessera_attn/Attn.td
```

### 4.2 `tessera.attn.scaled_dot_product`

Computes QK^T / sqrt(d) for a single tile of Q against a tile of K.

```mlir
%scores = tessera.attn.scaled_dot_product %Q_tile, %K_tile {tile_q = 64, tile_kv = 64}
    : memref<64x64xf16, 1>, memref<64x64xf16, 1> -> memref<64x64xf32, 1>
```

The scale `1/sqrt(d)` is applied as a sentinel attribute resolved by
`NVFlashAttnKernelEmitter`. The result is the raw (unmasked, unsoftmaxed) attention score.

### 4.3 `tessera.attn.online_softmax`

Applies the FA-2 online softmax update: running max correction + exponential rescaling.
Must follow `scaled_dot_product` in program order.

```mlir
%scores_out, %lse_out = tessera.attn.online_softmax %scores, %lse_prev
    : memref<64x64xf32, 1>, memref<64xf32, 3>
    -> memref<64x64xf32, 1>, memref<64xf32, 3>
```

**Algorithm (normative):** Implements Algorithm 1 from "FlashAttention-2" (Dao, 2023).
Running max `m_new = max(m_prev, rowmax(scores))`. Rescale factor
`alpha = exp(m_prev - m_new)`. Output scores are `exp(scores - m_new)`. LSE updated as
`lse_new = alpha * lse_prev + rowsum(exp(scores - m_new))`.

**Verifier rules:**
- `%lse_prev` must have shape `[seq_len]` matching the Q-tile row count
- Score tensor must be F32 (not F16 — online softmax requires F32 precision)

### 4.4 `tessera.attn.lse_accumulate`

Accumulates the output tile weighted by the running LSE correction factor. This is the
accumulation step of FlashAttention: `O_new = alpha * O_prev + softmax(scores) * V`.

```mlir
%O_out = tessera.attn.lse_accumulate %O_prev, %scores_norm, %V_tile, %lse_correction
    : memref<64x64xf32, 1>, memref<64x64xf32, 1>,
      memref<64x64xf16, 1>, memref<64xf32, 3>
    -> memref<64x64xf32, 1>
```

### 4.5 `tessera.attn.causal_mask`

Applies causal masking (upper triangular zeroing) to an attention score tile. Only tiles
that may contain both causal and non-causal entries need this op — fully causal or fully
non-causal tiles are handled statically by the emitter.

```mlir
%masked = tessera.attn.causal_mask %scores, %q_offset, %kv_offset
    : memref<64x64xf32, 1>, index, index -> memref<64x64xf32, 1>
```

### 4.6 `tessera.attn.dropout_mask`

Applies stochastic attention dropout. Only emitted when `FlashAttnLoweringConfig.dropout_p > 0`.

```mlir
%dropped = tessera.attn.dropout_mask %scores, %rng_state {dropout_p = 0.1}
    : memref<64x64xf32, 1>, i64 -> memref<64x64xf32, 1>
```

### 4.7 `tessera.attn.lse.save` / `tessera.attn.lse.load`

Save and load the per-row log-sum-exp tensor for use in backward passes (Phase 5+).

```mlir
%lse_saved = tessera.attn.lse.save %scores : memref<?xf32, 3> -> memref<?xf32, 0>
%lse       = tessera.attn.lse.load         : -> memref<?xf32, 0>
```

**`lse.save` verifier:** Input must be register-file tensor (space 3); output must be global (space 0).

---

## 5. `tessera.queue.*` — Tile Queue Dialect

The `tessera.queue` dialect implements **producer/consumer token ordering** for warp
specialization. It is inserted by `WarpSpecializationPass` to coordinate asynchronous
data movement between producer and consumer warps.

### 5.1 Dialect Definition

```
Dialect name: tessera.queue
C++ namespace: ::tessera::queue
Source: src/compiler/tile_opt_fa4/dialects/tessera_queue/Queue.td
```

### 5.2 Types

| Type | Description |
|------|-------------|
| `tessera.queue.TileQueueType` | Opaque handle to a tile queue (FIFO of tile-sized buffers) |
| `tessera.queue.TokenType` | Opaque ordering token produced by `push`, consumed by `pop` |

### 5.3 `tessera.queue.create`

Creates a tile queue. Exactly one `create` must appear per producer/consumer pair in the
warp specialization pattern.

```mlir
%q = tessera.queue.create : !tessera.queue.TileQueueType
```

### 5.4 `tessera.queue.push`

Pushes a tile into the queue and returns an ordering token. The producer warp calls this
after completing a `tile.async_copy` + `tile.wait_async` sequence.

```mlir
%token = tessera.queue.push %q, %tile
    : !tessera.queue.TileQueueType, memref<128x64xf16, 1>
    -> !tessera.queue.TokenType
```

### 5.5 `tessera.queue.pop`

Pops a tile from the queue. Blocks until the token produced by the matching `push` is
available. The consumer warp calls this before performing `tile.mma`.

```mlir
%tile_out = tessera.queue.pop %q, %dep_token
    : !tessera.queue.TileQueueType, !tessera.queue.TokenType
    -> memref<128x64xf16, 1>
```

### 5.6 Queue Semantics

- Queues are single-producer, single-consumer (SPSC)
- `push` and `pop` are matched by position: the Nth `push` token feeds the Nth `pop`
- A `pop` that is not fed by a reachable `push` is a verifier error
- Queue ops are only valid inside a `tessera.schedule.warp` region

---

## 6. `tessera.tcgen05.*` — Tensor Memory (TMEM) Ops

SM_100+ (Blackwell) introduces **Tensor Memory (TMEM)**, a compiler-managed accumulator
space for MMA operations. These ops are gated behind `target_profile.isa >= ISA.SM_100`.

### 6.1 `tessera.tcgen05.alloc`

Allocates a TMEM accumulator buffer. Called once per kernel preamble per accumulator.

```mlir
%tmem = tessera.tcgen05.alloc[f32](64, 64) : memref<64x64xf32, 2>
```

Result type must be memory space 2 (TMEM). All dimensions must be static.

### 6.2 `tessera.tcgen05.mma`

Performs a TMEM-backed MMA. Accumulates directly into TMEM without round-tripping
through registers or shared memory.

```mlir
%acc_out = tessera.tcgen05.mma %A_smem, %B_smem, %acc_tmem
    : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xf32, 2>
    -> memref<64x64xf32, 2>
```

**Verifier rules:**
- A and B must be shared memory (space 1) BF16 or F16
- Accumulator must be TMEM (space 2) F32
- Only valid when module attribute `tessera.isa = "SM_100"` or higher

### 6.3 `tessera.tcgen05.commit`

Commits the TMEM accumulator to shared memory (or global) for epilogue processing.
Required before any non-MMA use of the accumulator.

```mlir
tessera.tcgen05.commit %acc_tmem into %smem_out
    : memref<64x64xf32, 2>, memref<64x64xf32, 1>
```

---

## 7. Warp Specialization Regions

`WarpSpecializationPass` wraps warp-role-specific code in `tessera.schedule.warp` regions.

```mlir
tessera.schedule.warp {role = "producer"} {
  // Async copy logic — producer warps only
  tile.async_copy %src into %smem {stage = 0, vector = 16}
  tile.wait_async {stage = 0}
  %tok = tessera.queue.push %q, %smem
}

tessera.schedule.warp {role = "consumer"} {
  // MMA logic — consumer warps only
  %tile = tessera.queue.pop %q, %tok
  %acc  = tile.mma %tile, %weight, %acc_init
}
```

**Verifier rules:**
- `tessera.schedule.warp` regions must not be nested
- A function may contain at most one producer region and one consumer region per queue
- `tile.barrier` is not permitted inside a single-role warp region (use queue tokens instead)
- All `tessera.queue.*` ops must be inside a `tessera.schedule.warp` region

---

## 8. Verifier Summary

The Tile IR verifier enforces the following (normative):

| Rule | Checked at |
|------|-----------|
| Every `tile.async_copy {stage=N}` has a matching `tile.wait_async {stage=N}` | Function level |
| No `tile.wait_async` for a stage with no corresponding `tile.async_copy` | Function level |
| All `tshared.alloc` results are memory space 1 | Op level |
| All `tessera.tcgen05.*` ops only appear when module `tessera.isa >= SM_100` | Module level |
| `tessera.queue.*` ops only appear inside `tessera.schedule.warp` regions | Op level |
| `tile.mma` input dimensions match hardware alignment requirements | Op level |
| `tessera.attn.online_softmax` LSE shape matches Q-tile row count | Op level |
| Producer and consumer warp regions are not nested | Region level |

---

## 9. Phase Coverage

| Tile IR feature | Phase introduced | Status |
|----------------|-----------------|--------|
| `tshared.alloc` | Phase 3 | ✅ Complete |
| `tile.async_copy` / `tile.wait_async` | Phase 3 | ✅ Complete |
| `tile.mma` | Phase 3 | ✅ Complete |
| `tile.reduce` | Phase 3 | ✅ Complete |
| `tile.barrier` | Phase 3 | ✅ Complete |
| `tessera.attn.*` (FA-4 ops) | Phase 3 | ✅ Complete |
| `tessera.queue.*` (warp specialization) | Phase 3 | ✅ Complete |
| `tessera.tcgen05.*` (TMEM / SM_100) | Phase 3 (ODS defined) | 🔲 Lowering Phase 6 |
| Collective ops (`tile.comm`) | Phase 4 planned | 🔲 Planned |
| ROCm MFMA Tile IR path | Phase 6 planned | 🔲 Planned |

---

## 10. Relationship to Other Specs

| Question | Where to look |
|----------|--------------|
| What Schedule IR ops produce Tile IR? | `LOWERING_PIPELINE_SPEC.md §2.2` (TileIRLoweringPass) |
| What Target IR ops does Tile IR lower to? | `TARGET_IR_SPEC.md §4–5` |
| What Python ops trigger `tessera.attn.*` emission? | `PYTHON_API_SPEC.md §15` (`flash_attn`) |
| What are the FlashAttention tile size defaults? | `PYTHON_API_SPEC.md §14` (`FlashAttnLoweringConfig`) |
| What are warp role counts per SM target? | `TARGET_IR_SPEC.md §2.2` |
