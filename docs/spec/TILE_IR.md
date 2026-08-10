---
status: Normative
classification: Normative
authority: Tile IR op set and dialect semantics; defers Schedule IR and Target IR details to docs/spec/TARGET_IR_SPEC.md
last_updated: 2026-08-10
---

# Tessera Tile IR Specification (Normative)

**Version:** 0.3.0
**Authority:** This document specifies the Tile IR op set, dialect structure, and verifier rules. For the full Target IR dialect (WGMMA, TMA, mbarrier) see `docs/spec/TARGET_IR_SPEC.md`. For the lowering passes that produce Tile IR from Schedule IR, see `docs/spec/LOWERING_PIPELINE_SPEC.md`.

---

## Documentation refresh (2026-05-22)

The 2026-05-06 audit asked Tile IR to clarify the status of debug
markers and to settle the `tshared.alloc` vs `tile.alloc_shared` naming
question. Resolution:

- **`tile.debug_artifact` and `tile.debug_barrier` are metadata-only**,
  not normative Tile IR ops. They propagate through
  `python/tessera/compiler/tile_ir.py` so debug inspection works
  across Schedule + Tile lowering, but they are **elided** by
  `target_ir.py` before Target IR codegen. Their Python source is the
  authoritative contract; no Tile IR ODS op is created for them.
- **Canonical name for shared memory allocation is `tile.alloc_shared`**
  (matches the PM verifier and tests). Older spec text referencing
  `tshared.alloc` is a documentation alias; the implementation accepts
  the canonical form only.
- **TilingInterface methods on `MatmulOp` are real** as of Sprint B3-v2
  (2026-05): the matmul tiling interface follows MLIR 23 signatures
  with a lit fixture under `tests/tessera-ir/phase2/` + Python guard.
  `Conv2DNHWCOp` has an honest `failure()` scaffold and is explicitly
  marked scaffolded.
- **TMEM / tcgen05 path stays scaffolded**: the `LowerTileToPTX.cpp`
  body emits a schematic Blackwell PTX form. Typed Blackwell target selection
  and NVVM-contract gating live in the NVIDIA backend lit suite under
  `src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/`. Architecture Decision
  #21 (unsupported lowering must emit a stable diagnostic) covers the
  "lower with honest gating" contract for TMEM today.
- **The FA-4 Attn dialect** is a normative Tile IR layer (see
  `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/`)
  and remains lit-testable. The 4/4 FA-4 lit fixtures pass.
- **The `tessera.queue` MLIR dialect was deleted 2026-08-10** (Decisions
  #29/#31) — see §5 for the disposition. Producer/consumer synchronization
  is expressed through `!tile.pipeline_state` + `!tile.async_token` SSA
  chains instead (§5, §7).

---

## 1. Scope and Role

Tile IR is the **third layer** of the Tessera four-layer IR stack. It is produced by
`TileIRLoweringPass` from Schedule IR and consumed by the target backends (x86, NVIDIA,
ROCm, Apple).

```
Schedule IR  (schedule.* dialect)
     │
     ▼  TileIRLoweringPass
Tile IR      (tile.* + tessera_attn.* + tessera.tcgen05.*)
     │
     ▼  NVWGMMALoweringPass / TileToX86Pass / ...
Target IR    (tessera.nvgpu.wgmma.*, tessera.tma.*, x86 intrinsics)
```

Tile IR is the layer at which:
- Explicit shared memory allocation appears (`tile.alloc_shared`)
- Warp roles are assigned (`tessera.schedule.warp {role="producer/consumer"}`)
- Async copy stages are made explicit (`tile.async_copy {stage=N}`)
- MMA operations are expressed (`tile.mma`)
- Producer/consumer ordering is introduced as `!tile.pipeline_state` + `!tile.async_token` SSA chains (`tile.pipeline_init` / `tile.pipeline_advance`)
- FlashAttention sub-operations appear (`tessera_attn.*`)

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

### 3.1 `tile.alloc_shared`

Allocates a buffer in shared memory. Must appear in the kernel preamble.

```mlir
%smem = tile.alloc_shared : memref<128x64xf16, 1> {swizzle = "xor"}
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
`role="consumer"` in isolation — producer/consumer synchronization uses
`!tile.pipeline_state` SSA chains instead (§7).

---

## 4. `tessera_attn.*` — FA-4 Attention Dialect

The `tessera.attn` dialect implements the **FA-4 FlashAttention algorithm** at Tile IR
level. It is produced by `TileIRLoweringPass` when lowering `tessera.flash_attn` Graph IR
ops targeting SM_90+.

### 4.1 Dialect Definition

```
Dialect name: tessera.attn
C++ namespace: ::tessera::attn
Source: src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td
```

### 4.2 `tessera_attn.scaled_dot_product`

Computes QK^T / sqrt(d) for a single tile of Q against a tile of K.

```mlir
%scores = tessera_attn.scaled_dot_product %Q_tile, %K_tile {tile_q = 64, tile_kv = 64}
    : memref<64x64xf16, 1>, memref<64x64xf16, 1> -> memref<64x64xf32, 1>
```

The scale `1/sqrt(d)` is applied as a sentinel attribute resolved by
`NVFlashAttnKernelEmitter`. The result is the raw (unmasked, unsoftmaxed) attention score.

### 4.3 `tessera_attn.online_softmax`

Applies the FA-2 online softmax update: running max correction + exponential rescaling.
Must follow `scaled_dot_product` in program order.

```mlir
%scores_out, %lse_out = tessera_attn.online_softmax %scores, %lse_prev
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

### 4.4 `tessera_attn.lse_accumulate`

Accumulates the output tile weighted by the running LSE correction factor. This is the
accumulation step of FlashAttention: `O_new = alpha * O_prev + softmax(scores) * V`.

```mlir
%O_out = tessera_attn.lse_accumulate %O_prev, %scores_norm, %V_tile, %lse_correction
    : memref<64x64xf32, 1>, memref<64x64xf32, 1>,
      memref<64x64xf16, 1>, memref<64xf32, 3>
    -> memref<64x64xf32, 1>
```

### 4.5 `tessera_attn.causal_mask`

Applies causal masking (upper triangular zeroing) to an attention score tile. Only tiles
that may contain both causal and non-causal entries need this op — fully causal or fully
non-causal tiles are handled statically by the emitter.

```mlir
%masked = tessera_attn.causal_mask %scores, %q_offset, %kv_offset
    : memref<64x64xf32, 1>, index, index -> memref<64x64xf32, 1>
```

### 4.6 `tessera_attn.dropout_mask`

Applies stochastic attention dropout. Only emitted when `FlashAttnLoweringConfig.dropout_p > 0`.

```mlir
%dropped = tessera_attn.dropout_mask %scores, %rng_state {dropout_p = 0.1}
    : memref<64x64xf32, 1>, i64 -> memref<64x64xf32, 1>
```

### 4.7 `tessera_attn.lse.save` / `tessera_attn.lse.load`

Save and load the per-row log-sum-exp tensor for use in backward passes (Phase 5+).

```mlir
%lse_saved = tessera_attn.lse.save %scores : memref<?xf32, 3> -> memref<?xf32, 0>
%lse       = tessera_attn.lse.load         : -> memref<?xf32, 0>
```

**`lse.save` verifier:** Input must be register-file tensor (space 3); output must be global (space 0).

---

## 5. `tessera.queue` MLIR Dialect — DELETED (2026-08-10)

The `tessera.queue` MLIR dialect (ops `create`/`push`/`pop`, types
`!tessera.queue.type` / `!tessera.queue.token`) was deleted under Decisions
#29/#31: no pass ever produced or consumed its ops, and the dotted-name type
syntax could not be parsed from standalone lit IR. Earlier revisions of this
section specified it as the warp-specialization synchronization mechanism;
that was never what `WarpSpecializationPass` emitted.

The normative producer/consumer synchronization mechanism is **pipeline-state
SSA**: `WarpSpecializationPass` stamps `tessera.schedule.warp` regions with
`tile.warp_role` attributes and threads `!tile.pipeline_state` +
`!tile.async_token` SSA chains through them (`tile.pipeline_init` /
`tile.pipeline_advance`) — see §7 and `LOWERING_PIPELINE_SPEC.md` §3.7.

The queue *vocabulary* survives only in the **Python tile IR reference
spine**: `lower_schedule_to_tile_ir` (`python/tessera/compiler/tile_ir.py`)
emits textual `tessera.queue.{create,push,pop,barrier}` ops with
`queue_id`/`depth`/`stage`/`scope` attributes, verified by `tile_ir.py` and
`memory_verifier.py`. Those textual ops are a Python-side reference contract,
not an MLIR dialect. Any MLIR revival must use a parseable single-segment
dialect name and ship a real producer plus a passing fixture (see
`tests/unit/test_mlir_verifier_sprint.py::test_queue_mlir_dialect_stays_deleted`).

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

`WarpSpecializationPass` wraps warp-role-specific code in `tessera.schedule.warp`
regions stamped with `tile.warp_role` and a shared `tile.pipeline` id, and
synchronizes the roles through `!tile.pipeline_state` + `!tile.async_token`
SSA chains.

```mlir
tessera.schedule.warp {tile.warp_role = "producer", tile.pipeline = "pipe0"} {
  // Async copy logic — producer warps only
  %ps = tile.pipeline_init {depth = 2, stage = 0, phase = 0, role = "producer"} : !tile.pipeline_state
  %smem, %tok = tile.async_copy %src into %smem_buf {stage = 0, vector = 16}
  %ps1 = tile.pipeline_advance %ps, %tok : !tile.pipeline_state
}

tessera.schedule.warp {tile.warp_role = "consumer", tile.pipeline = "pipe0"} {
  // MMA logic — consumer warps only
  %ps = tile.pipeline_init {depth = 2, stage = 0, phase = 1, role = "consumer"} : !tile.pipeline_state
  %acc = tile.mma %smem, %weight, %acc_init
  %ps1 = tile.pipeline_advance %ps, %acc : !tile.pipeline_state
}
```

**Verifier rules:**
- `tessera.schedule.warp` regions must not be nested
- A function may contain at most one producer region and one consumer region
  per `tile.pipeline` id
- generic `tile.barrier` is legacy spelling; active code uses `tile.mbarrier.*`
  for transactional barriers and pipeline-state SSA for producer/consumer handoff
- `tile.pipeline_init` / `tile.pipeline_advance` must be inside a
  `tessera.schedule.warp` region

---

## 8. Verifier Summary

The Tile IR verifier enforces the following (normative):

| Rule | Checked at |
|------|-----------|
| Every `tile.async_copy {stage=N}` has a matching `tile.wait_async {stage=N}` | Function level |
| No `tile.wait_async` for a stage with no corresponding `tile.async_copy` | Function level |
| All `tile.alloc_shared` results are memory space 1 | Op level |
| All `tessera.tcgen05.*` ops only appear when module `tessera.isa >= SM_100` | Module level |
| `tile.pipeline_init` / `tile.pipeline_advance` only appear inside `tessera.schedule.warp` regions | Op level |
| `tile.mma` input dimensions match hardware alignment requirements | Op level |
| `tessera_attn.online_softmax` LSE shape matches Q-tile row count | Op level |
| Producer and consumer warp regions are not nested | Region level |

---

## 8.1 Debug Metadata Markers

The Python object-model lowering path may emit `tile.debug_artifact` and
`tile.debug_barrier` as metadata-only markers. They are developer-tool aids used
for replay manifests, barrier inspection, and schedule hash correlation. They
must not be treated as target execution ops, and Target IR lowering elides them.
Native ODS debug ops should only be promoted if diagnostics and artifact
metadata cannot represent the same information.

---

## 9. Phase Coverage

| Tile IR feature | Phase introduced | Status |
|----------------|-----------------|--------|
| `tile.alloc_shared` | Phase 3 | implemented / lit-testable |
| `tile.async_copy` / `tile.wait_async` | Phase 3 | ✅ Complete |
| `tile.mma` | Phase 3 | ✅ Complete |
| `tile.reduce` | Phase 3 | ✅ Complete |
| `tile.mbarrier.*` | Phase 3 | implemented / lit-testable |
| `tile.barrier` | legacy note | planned alias only; prefer `tile.mbarrier.*` or pipeline-state SSA |
| `tile.debug_artifact` / `tile.debug_barrier` | developer tooling | metadata-only markers; elided before Target IR |
| `tessera_attn.*` (FA-4 ops) | Phase 3 | ✅ Complete |
| `tessera.queue` MLIR dialect | Phase 3 | ❌ deleted 2026-08-10 (Decisions #29/#31); warp-spec sync is pipeline-state SSA; textual queue vocabulary lives in the Python tile IR spine only |
| `tessera.tcgen05.*` (TMEM / SM_100) | Phase 3 (ODS defined) | stubbed / lit-testable until real Blackwell PTX operands land |
| Collective ops (`tile.comm`) | distributed extension | planned / adapter-gated; see collective lowering and validation docs |
| ROCm MFMA Tile IR path | backend extension | planned / scaffolded in ROCm Target IR |

---

## 10. Relationship to Other Specs

| Question | Where to look |
|----------|--------------|
| What Schedule IR ops produce Tile IR? | `LOWERING_PIPELINE_SPEC.md §2.2` (TileIRLoweringPass) |
| What Target IR ops does Tile IR lower to? | `TARGET_IR_SPEC.md §4–5` |
| What Python ops trigger `tessera_attn.*` emission? | `PYTHON_API_SPEC.md §15` (`flash_attn`) |
| What are the FlashAttention tile size defaults? | `PYTHON_API_SPEC.md §14` (`FlashAttnLoweringConfig`) |
| What are warp role counts per SM target? | `TARGET_IR_SPEC.md §2.2` |
