---
status: Informative
classification: Informative
authority: IR layer narrative; normative op semantics defer to docs/spec/
last_updated: 2026-04-30
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.

# Tessera IR Layers

Tessera uses a **four-layer MLIR IR stack**. Each layer represents a well-defined
lowering contract. Passes transform IR from one layer to the next; no layer is
skipped. This document describes each layer, gives a representative MLIR snippet,
and maps the passes that cross each boundary.

For normative op semantics at each layer, see the specs in `docs/spec/`.

---

## The Four Layers at a Glance

```
Python API  (@tessera.jit, tessera.ops.*, tessera.domain, Region[...])
     │  [GraphIRBuilder — python/tessera/compiler/graph_ir.py]
     ▼
Layer 1 — Graph IR     (tessera.* dialect)
     │  [EffectAnnotationPass → CanonicalizeTesseraIR → DistributionLoweringPass]
     ▼
Layer 2 — Schedule IR  (schedule.mesh.* dialect)
     │  x86:  [TilingPass → TileToX86Pass]
     │  GPU:  [TileIRLoweringPass → WarpSpecializationPass → AsyncCopyLoweringPass]
     ▼
Layer 3 — Tile IR      (tile.* + tessera.attn.* + tessera.queue.*)
     │  GPU:  [NVWGMMALoweringPass → NVTMADescriptorPass → NVFlashAttnKernelEmitter]
     │  x86:  (produced directly by TileToX86Pass as func.call stubs)
     ▼
Layer 4 — Target IR    (tessera.nvgpu.wgmma.*, tessera.tma.*, LLVM NVPTX / x86 calls)
```

Two named pipelines wire these passes:

| Pipeline | Target | Layer boundary |
|----------|--------|----------------|
| `tessera-lower-to-x86` | x86 AMX/AVX-512 | Graph IR → Schedule IR → x86 calls |
| `tessera-lower-to-gpu` | NVIDIA SM_90+ | Graph IR → Schedule IR → Tile IR → Target IR |

---

## Layer 1 — Graph IR

**What it is:** The algebraic representation of the computation. Produced by
`@tessera.jit` via `GraphIRBuilder`. All tensor shapes may be symbolic at this
layer; unknown dimensions are allowed.

**Dialect:** `tessera.*`

**Key ops:**

```mlir
// Matrix multiply
%C = tessera.matmul %A, %B
   : (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>

// Flash attention (causal)
%O = tessera.flash_attn %Q, %K, %V
   {causal = true, dropout_p = 0.0, scale = 0.125}
   : (tensor<Bx8xSx64xbf16>, tensor<Bx8xSx64xbf16>, tensor<Bx8xSx64xbf16>)
  -> tensor<Bx8xSx64xbf16>

// Effect and shard annotations on function arguments
func.func @step(%W: tensor<256x256xbf16>
                    {tessera.effect = "read",
                     tessera.shard = {axes = ["tp"], dims = [1]}},
                %X: tensor<128x256xbf16>
                    {tessera.effect = "read"})
              -> tensor<128x256xbf16> {
  %Y = tessera.matmul %X, %W
     : (tensor<128x256xbf16>, tensor<256x256xbf16>) -> tensor<128x256xbf16>
  return %Y : tensor<128x256xbf16>
}
```

**Passes that produce or transform Graph IR:**

| Pass | Role |
|------|------|
| `GraphIRBuilder` | Python → Graph IR (Phase 1) |
| `EffectAnnotationPass` | Infers `tessera.effect` attrs (`pure/random/memory/io/top`) |
| `CanonicalizeTesseraIR` | 4 fusion/simplification patterns (FuseMatmulBiasGELU, FuseConvRelu, DropoutZeroSimplify, TransposeIntoMatmul) |
| `VerifyTesseraIR` | Checks `tessera.version` module attr |

**Normative reference:** [`docs/spec/GRAPH_IR_SPEC.md`](../../spec/GRAPH_IR_SPEC.md)

---

## Layer 2 — Schedule IR

**What it is:** Graph IR after distribution lowering. Tensor partitioning is now
explicit: `schedule.mesh.define` records the mesh, and `schedule.mesh.region`
wraps each parallelized body. Unknown dimensions must be resolved, padded, or
guarded by shape witnesses before Tile IR.

**Dialect:** `schedule.mesh.*`, `schedule.pipeline.*`

**Key ops:**

```mlir
func.func @step(%W: tensor<256x256xbf16>, %X: tensor<128x256xbf16>)
              -> tensor<128x256xbf16> {

  schedule.mesh.define {dims = [4], axis_names = ["tp"]}

  %Y = schedule.mesh.region {mesh = @tp, axis = "tp"} {
    %y_local = tessera.matmul %X, %W
             : (tensor<128x64xbf16>, tensor<64x256xbf16>)
            -> tensor<128x256xbf16>
    schedule.yield %y_local : tensor<128x256xbf16>
  } : tensor<128x256xbf16>

  return %Y : tensor<128x256xbf16>
}
```

For pipelined kernels (GPU path):

```mlir
%Ct = tessera.schedule.tile %C {m = 128, n = 128, k = 64}
%Cp = tessera.schedule.pipeline %Ct {double_buffer = true, depth = 3}
```

**Passes that produce Schedule IR:**

| Pass | Role |
|------|------|
| `DistributionLoweringPass` | `tessera.shard` attrs → `schedule.mesh.define` + `schedule.mesh.region` |

**Passes that consume Schedule IR (x86 path):**

| Pass | Role |
|------|------|
| `TilingPass` | `tessera.matmul` → `scf.for` M×N loops over `tensor.extract_slice` |
| `TileToX86Pass` | Tiled matmul → `func.call @tessera_x86_amx_gemm_bf16(...)` |

**Normative reference:** [`docs/spec/LOWERING_PIPELINE_SPEC.md`](../../spec/LOWERING_PIPELINE_SPEC.md)

---

## Layer 3 — Tile IR

**What it is:** Computation bound to accelerator execution primitives: blocks,
warps, fragments, shared memory, transaction barriers, and MMA instructions.
All tensor ranks must be known and all symbolic dimensions resolved. Produced
by `TileIRLoweringPass` for the GPU path.

**Dialects:** `tile.*`, `tessera.attn.*`, `tessera.queue.*`

**Key ops:**

```mlir
// Shared memory allocation
%sa = tile.alloc_shared %desc : memref<128x64xbf16, shared>

// Asynchronous global→shared copy (tracked by mbarrier)
tile.async_copy %A_global, %A_shared {stage = 0, vector = 16}

// Transaction barrier (SM_90+ required for TMA completion)
%bar = tile.mbarrier.alloc {count = 1, scope = "block"}
%tok = tile.mbarrier.arrive_expect_tx %bar
         {bytes = 16384, semantics = "release", scope = "block"}
%ok  = tile.mbarrier.try_wait %bar, %tok

// Warp-group MMA accumulate
%fragC = tile.mma %fragA, %fragB, %fragC
           {m = 64, n = 256, k = 32, accum = "fp32"}

// FA-4 online softmax ops (Flash Attention 2 algorithm)
%sdp  = tessera.attn.scaled_dot_product %Q_frag, %K_frag
          {scale = 0.125, causal = true}
%max  = tessera.attn.online_softmax %sdp {axis = -1}
%lse  = tessera.attn.lse_accumulate %max, %lse_prev
```

**Warp roles (produced by WarpSpecializationPass):**

```mlir
// Producer warps: issue async copies
tessera.schedule.warp {role = "producer", id = 0} {
    tile.async_copy %Q_global, %Q_shared {stage = 0, vector = 16}
    tessera.queue.push %q, %tok
}

// Consumer warps: wait, then compute
tessera.schedule.warp {role = "consumer", id = 1} {
    %tok = tessera.queue.pop %q
    tile.mbarrier.try_wait %bar, %tok
    tile.mma %fragA, %fragB, %fragC {m = 64, n = 256, k = 32, accum = "fp32"}
}
```

**Tile IR verifier rules:**

- `ldmatrix` and MMA operands must satisfy alignment and layout constraints
- MMA tile shapes must be supported for the dtype on the target ISA
- Shared allocations must fit per-block shared-memory limits
- mbarriers must be initialized before arrival or wait
- Transaction byte counts must match associated async movement
- Barriers must not appear in divergent control paths for their scope

**Passes that produce Tile IR (GPU path):**

| Pass | Role |
|------|------|
| `TileIRLoweringPass` | `schedule.mesh.region { tessera.flash_attn }` → `tile.*` + `tessera.attn.*` |
| `WarpSpecializationPass` | Assigns producer/consumer roles; inserts `tessera.queue.*` barriers |
| `AsyncCopyLoweringPass` | `tile.async_copy` → TMA descriptor ops (SM_90+) or `cp.async` |

**Normative reference:** [`docs/spec/TILE_IR.md`](../../spec/TILE_IR.md), [`docs/spec/MEMORY_MODEL_SPEC.md`](../../spec/MEMORY_MODEL_SPEC.md)

---

## Layer 4 — Target IR

**What it is:** Tile IR lowered to vendor-specific MLIR intrinsics that map
1-to-1 to hardware instructions. This is the final MLIR layer before LLVM
backend emission.

**Dialects (NVIDIA):** `tessera.nvgpu.wgmma.*`, `tessera.tma.*`, `tessera.tcgen05.*`

**Dialects (AMD):** `rocdl.*` MFMA ops, LDS intrinsics

**Dialects (x86):** `func.call` to `@tessera_x86_amx_gemm_bf16` etc.

### NVIDIA SM_90 — WGMMA + TMA

```mlir
// tile.mma → wgmma.mma_async (NVWGMMALoweringPass)
%c = tessera.nvgpu.wgmma.mma_async %desc_a, %desc_b, %c
       {shape = [64, 256, 32], typeA = bf16, typeB = bf16, typeC = f32,
        layoutA = "row", layoutB = "col"}
   : (!llvm.ptr<3>, !llvm.ptr<3>, !llvm.array<64xf32>) -> !llvm.array<64xf32>

tessera.nvgpu.wgmma.wait_group.sync.aligned {pendings = 0}

// tile.async_copy → TMA bulk copy (NVTMADescriptorPass)
%desc = tessera.tma.create_descriptor %global_ptr
          {dtype = bf16, rank = 2, swizzle = "128B", l2_promotion = "128B"}

tessera.tma.async_load %shared_ptr, %desc, %bar, [%m, %k]
  {multicast_mask = 0 : i16}

// mbarrier ops (NVFlashAttnKernelEmitter)
%bar  = tessera.nvgpu.mbarrier.create {count = 1}
%tok  = tessera.nvgpu.mbarrier.arrive_expect_tx %bar {bytes = 16384}
%done = tessera.nvgpu.mbarrier.try_wait %bar, %tok {phase = 0}
```

### NVIDIA SM_100 — tcgen05 (Blackwell TMEM)

```mlir
%tmem = tessera.tcgen05.alloc {cols = 128}
tessera.tcgen05.mma.async %tmem, %desc_a, %desc_b, %idesc
  {cta_group = 2, shape = [128, 256, 32], typeA = fp8_e4m3, typeC = f32}
tessera.tcgen05.dealloc %tmem
```

### Kernel Entry ABI

```mlir
kernel @flash_attn_fwd(
    %q   : !llvm.ptr<global>,
    %k   : !llvm.ptr<global>,
    %v   : !llvm.ptr<global>,
    %out : !llvm.ptr<global>,
    %B   : i64, %H : i64, %S : i64, %D : i64,
    %scale : f32
) attributes {
    grid  = (B*H, S/64, 1),
    block = (128, 1, 1),
    smem_bytes = 49152,
    stream = %stream
}
```

All pointers are address-space-qualified global pointers. Scalar uniforms are
32-bit or 64-bit. Grid, block, and smem_bytes must be explicit.

**Passes that produce Target IR:**

| Pass | Role |
|------|------|
| `NVWGMMALoweringPass` | `tile.mma` → `tessera.nvgpu.wgmma.mma_async` + wait |
| `NVTMADescriptorPass` | Hoists TMA descriptor setup; `tile.async_copy` → `tessera.tma.async_load` |
| `NVFlashAttnKernelEmitter` | Completes FA-4 kernel: mbarrier arrive/wait, scale sentinel, launch bounds |
| `TileToX86Pass` | Emits `func.call @tessera_x86_amx_gemm_bf16(...)` (x86 path) |

**Normative reference:** [`docs/spec/TARGET_IR_SPEC.md`](../../spec/TARGET_IR_SPEC.md), [`docs/spec/LANGUAGE_AND_IR_SPEC.md §10`](../../spec/LANGUAGE_AND_IR_SPEC.md)

---

## Boundary Contracts at a Glance

| Crossing | Pre-condition | Post-condition |
|----------|--------------|----------------|
| Python → Graph IR | Decorated function with type annotations | `tessera.*` ops with effect and shard attrs |
| Graph IR → Schedule IR | `tessera.shard` attrs present on all distributed args | `schedule.mesh.define` + `schedule.mesh.region` |
| Schedule IR → Tile IR | All shapes resolved; movement plans explicit | `tile.*` + `tessera.attn.*` + mbarrier ops |
| Tile IR → Target IR | Alignment/layout constraints satisfied | Vendor intrinsics; kernel ABI ready for launch |

---

## Phase Status of Each Layer

| Layer | Status |
|-------|--------|
| Graph IR emission + canonicalization + effect inference | ✅ Phase 1–2 complete |
| Schedule IR (x86 path: Distribution → Tiling → TileToX86) | ✅ Phase 2 complete |
| Tile IR + warp specialization (GPU path) | ✅ Phase 3 complete |
| Target IR — NVIDIA WGMMA / TMA / mbarrier (SM_90+) | ✅ Phase 3 complete |
| Target IR — AMD MFMA full coverage | 🔲 Phase 6 planned |
| Target IR — Runtime C ABI wired for launch | 🔲 Phase 6 planned |
| Distributed collectives in Schedule/Target IR | 🔲 Phase 4 planned |

---

## References

- [`docs/spec/GRAPH_IR_SPEC.md`](../../spec/GRAPH_IR_SPEC.md) — Graph IR op semantics and verifier
- [`docs/spec/LOWERING_PIPELINE_SPEC.md`](../../spec/LOWERING_PIPELINE_SPEC.md) — pass input/output contracts
- [`docs/spec/TILE_IR.md`](../../spec/TILE_IR.md) — Tile IR op set and verifier rules
- [`docs/spec/TARGET_IR_SPEC.md`](../../spec/TARGET_IR_SPEC.md) — Target IR dialect and ABI
- [`docs/spec/MEMORY_MODEL_SPEC.md`](../../spec/MEMORY_MODEL_SPEC.md) — mbarrier, visibility, synchronization
- [`docs/spec/COMPILER_REFERENCE.md`](../../spec/COMPILER_REFERENCE.md) — named pipelines and pass registry
- [`docs/architecture/tessera_target_ir_usage_guide.md`](../tessera_target_ir_usage_guide.md) — Target IR usage examples
