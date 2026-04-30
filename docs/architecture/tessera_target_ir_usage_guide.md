---
status: Informative
classification: Informative
authority: Target IR usage examples; normative op semantics and ABI in docs/spec/TARGET_IR_SPEC.md and docs/spec/RUNTIME_ABI_SPEC.md
last_updated: 2026-04-30
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.

# Tessera Target IR — Usage Guide

Target IR is the **fourth and final layer** of the Tessera compilation pipeline. It receives Tile IR from the backend-specific lowering passes and emits vendor intrinsics — WGMMA/TMA/mbarrier for NVIDIA, MFMA/LDS for AMD, and AMX/AVX-512 function calls for x86. Programmers do not write Target IR directly; they write Python using `@tessera.jit` and the compiler produces Target IR.

For normative op semantics, see [`docs/spec/TARGET_IR_SPEC.md`](../spec/TARGET_IR_SPEC.md).
For the runtime C ABI that executes the compiled kernels, see [`docs/spec/RUNTIME_ABI_SPEC.md`](../spec/RUNTIME_ABI_SPEC.md).

---

## Where Target IR Sits

```
Python API  (@tessera.jit, tessera.ops.flash_attn, ...)
     │  [GraphIRBuilder]
     ▼
Graph IR    (tessera.flash_attn, tessera.matmul, ...)
     │  [EffectAnnotationPass → CanonicalizeTesseraIR → DistributionLoweringPass]
     ▼
Schedule IR (schedule.mesh.region, schedule.pipeline, ...)
     │  [TileIRLoweringPass → WarpSpecializationPass → AsyncCopyLoweringPass]
     ▼
Tile IR     (tile.async_copy, tile.mma, tile.mbarrier.*, tessera.attn.*)
     │  [NVWGMMALoweringPass → NVTMADescriptorPass → NVFlashAttnKernelEmitter]  ← NVIDIA
     │  [TileToX86Pass]                                                           ← x86
     ▼
Target IR   (tessera.nvgpu.wgmma.*, tessera.tma.*, mbarrier.* → LLVM NVPTX)
             (func.call @tessera_x86_amx_gemm_bf16 → x86 binary)
```

The two named lowering pipelines are:

| Pipeline | Target | Passes |
|----------|--------|--------|
| `tessera-lower-to-x86` | x86 AMX/AVX-512 | EffectAnnotation → Canonicalize → Distribution → Tiling → TileToX86 |
| `tessera-lower-to-gpu` | NVIDIA SM_90+ | EffectAnnotation → Canonicalize → Distribution → TileIRLowering → WarpSpecialization → AsyncCopyLowering → NVWGMMALowering → NVTMADescriptor → NVFlashAttnKernelEmitter |

---

## Python Entry Point

All Target IR generation starts from `@tessera.jit`. The `target=` parameter
selects the lowering pipeline:

```python
import tessera
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
from tessera.compiler.attn_lower import FlashAttnLoweringConfig

# NVIDIA SM_90 (Hopper) — uses tessera-lower-to-gpu
@tessera.jit(
    target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4),
    attn_config=FlashAttnLoweringConfig(tile_q=64, tile_kv=64, pipeline_stages=2),
)
def flash_attn_fwd(
    Q: tessera.Tensor["B", "H", "S", "D"],
    K: tessera.Tensor["B", "H", "S", "D"],
    V: tessera.Tensor["B", "H", "S", "D"],
) -> tessera.Tensor["B", "H", "S", "D"]:
    tessera.require(tessera.constraint.Divisible("D", 64))
    return tessera.ops.flash_attn(Q, K, V, causal=True)

# x86 AMX/AVX-512 — uses tessera-lower-to-x86 (no target= → CPU path)
@tessera.jit
def cpu_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    return tessera.ops.gemm(A, B)

# Inspect emitted Graph IR (available Phase 1+)
print(flash_attn_fwd.graph_ir.to_mlir())
```

`GPUTargetProfile` key parameters:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `isa` | required | `ISA.SM_80` / `SM_90` / `SM_100` |
| `warps_per_cta` | 4 | must be power of 2 |
| `shared_mem_bytes` | SM default | override per-kernel |
| `pipeline_stages` | 2 | double-buffer depth |

`.supports_wgmma` → `isa >= SM_90`. `.supports_tma` → `isa >= SM_90`.

---

## What Target IR Looks Like

Target IR ops are MLIR operations in the `tessera.nvgpu.*`, `tessera.tma.*`,
and `tessera.tcgen05.*` namespaces. They are produced by the lowering passes
from Tile IR — not authored by hand.

### NVIDIA SM_90 (Hopper) — WGMMA + TMA

`NVWGMMALoweringPass` converts `tile.mma` → `tessera.nvgpu.wgmma.mma_async`:

```mlir
// tile.mma in Tile IR
%fragC = tile.mma %fragA, %fragB, %fragC
           {m = 64, n = 256, k = 32, accum = "fp32"}

// After NVWGMMALoweringPass — Target IR
%c = tessera.nvgpu.wgmma.mma_async %desc_a, %desc_b, %c
       {shape = [64, 256, 32], typeA = bf16, typeB = bf16, typeC = f32,
        layoutA = "row", layoutB = "col"}
     : (!llvm.ptr<shared>, !llvm.ptr<shared>,
        !llvm.array<64 x f32>) -> !llvm.array<64 x f32>

tessera.nvgpu.wgmma.wait_group.sync.aligned {pendings = 0}
```

`NVTMADescriptorPass` converts `tile.async_copy` → TMA descriptor setup + bulk copy:

```mlir
// tile.async_copy in Tile IR
tile.async_copy %A_global, %A_shared {stage = 0, vector = 16}

// After NVTMADescriptorPass — Target IR
%desc = tessera.tma.create_descriptor %A_global
          {dtype = bf16, rank = 2, swizzle = "128B",
           l2_promotion = "128B", oob_fill = "zero"}

tessera.tma.async_load %A_shared, %desc, %bar, [%coord_m, %coord_k]
  {multicast_mask = 0 : i16}
```

mbarrier tracking (emitted by `NVFlashAttnKernelEmitter`):

```mlir
// Allocate one transaction barrier per pipeline stage
%bar = tessera.nvgpu.mbarrier.create
         {count = 1, smem_size = @smem_bar}

// Producer side: arrive with expected byte count
%tok = tessera.nvgpu.mbarrier.arrive_expect_tx %bar
         {bytes = 16384, semantics = "release", scope = "block"}

// Consumer side: wait for completion
%ready = tessera.nvgpu.mbarrier.try_wait %bar, %tok
           {phase = 0}
```

### Kernel Entry ABI

The compiled kernel entry point follows this ABI:

```mlir
// Emitted by NVFlashAttnKernelEmitter
kernel @flash_attn_fwd(
    %q_ptr  : !llvm.ptr<global>,   // BF16 Q tensor
    %k_ptr  : !llvm.ptr<global>,   // BF16 K tensor
    %v_ptr  : !llvm.ptr<global>,   // BF16 V tensor
    %o_ptr  : !llvm.ptr<global>,   // BF16 output tensor
    %B      : i64,                  // batch size
    %H      : i64,                  // heads
    %S      : i64,                  // sequence length
    %D      : i64,                  // head dimension
    %scale  : f32                   // 1/sqrt(D)
) attributes {
    grid  = (B*H, S/64, 1),
    block = (128, 1, 1),            // 4 warps × 32 threads
    smem_bytes = 49152,             // 48 KB shared memory
    stream = %stream
}
```

Pointers are address-space-qualified global pointers. Grid, block, and dynamic
shared-memory size must be explicit. Scalar uniforms are 32-bit or 64-bit.

### NVIDIA SM_100 (Blackwell) — tcgen05

```mlir
// Blackwell TMEM MMA via tcgen05
%tmem = tessera.tcgen05.alloc {cols = 128}

tessera.tcgen05.mma.async %tmem, %desc_a, %desc_b, %instr_desc
  {cta_group = 2, shape = [128, 256, 32],
   typeA = fp8_e4m3, typeB = fp8_e4m3, typeC = f32}

tessera.tcgen05.dealloc %tmem
```

### x86 — AMX/AVX-512 Function Calls

`TileToX86Pass` converts tiled Schedule IR matmul ops to C function calls:

```mlir
// After TileToX86Pass — Target IR
func.call @tessera_x86_amx_gemm_bf16(%a_ptr, %b_ptr, %c_ptr, %M, %N, %K, %beta)
  : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32) -> ()

// With bias+GELU epilogue
func.call @tessera_x86_epilogue_bias_gelu_fp32(%c_ptr, %bias_ptr, %M, %N)
  : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()
```

The x86 backend (`src/compiler/codegen/tessera_x86_backend/`) implements these
functions with AMX BF16 tiles or AVX-512 fallback. This path is fully functional
in Phase 2 and is the recommended CPU validation path before deploying to GPU.

### AMD ROCm — MFMA (Phase 6)

```mlir
// MFMA lowering for gfx90a (MI210/250)
%result = rocdl.mfma.f32.32x32x8.f16 %a, %b, %c
            {cbsz = 0, abid = 0, blgp = 0}
          : (vector<4xf16>, vector<4xf16>, vector<32xf32>) -> vector<32xf32>

// LDS async copy
rocdl.ds.read.b128 %addr : (!llvm.ptr<3>) -> vector<4xi32>
```

Full MFMA coverage for gfx90a / gfx94x / gfx120x is Phase 6 planned.

---

## Warp Specialization in the Pipeline

`WarpSpecializationPass` (Phase 3) assigns warps to producer or consumer roles
before the WGMMA and TMA passes run. This creates the double-buffering structure:

```mlir
// After WarpSpecializationPass
tessera.schedule.warp {role = "producer", id = 0} {
    // Warps 0-1: issue TMA async loads
    tile.async_copy %Q_global, %Q_shared {stage = 0, vector = 16}
    tile.async_copy %K_global, %K_shared {stage = 0, vector = 16}
    tessera.queue.push %barrier_queue, %tok
}

tessera.schedule.warp {role = "consumer", id = 1} {
    // Warps 2-3: wait for data, run WGMMA
    %tok = tessera.queue.pop %barrier_queue
    tile.mbarrier.try_wait %bar, %tok
    tile.mma %fragA, %fragB, %fragC {m = 64, n = 256, k = 32, accum = "fp32"}
}
```

Producer and consumer roles use separate register files and barrier slots.
The `tessera.queue.*` ops are hard boundaries — not advisory annotations.

---

## ISA Feature Matrix

| Feature | SM_80 (A100) | SM_90 (H100) | SM_100 (B100) |
|---------|-------------|-------------|--------------|
| WGMMA | ❌ | ✅ | ✅ |
| TMA / cp.async.bulk | ❌ | ✅ | ✅ |
| mbarrier (transaction) | ❌ | ✅ | ✅ |
| tcgen05 / TMEM | ❌ | ❌ | ✅ |
| FP8 (e4m3 / e5m2) | ❌ | ✅ | ✅ |
| NVFP4 | ❌ | ❌ | ✅ |
| Fallback (WMMA + cp.async) | ✅ | ✅ | ✅ |

Below SM_90, `NVWGMMALoweringPass` falls back to `nvcuda::wmma` WMMA
intrinsics and `cp.async` (non-bulk) copies.

---

## Inspecting the Emitted IR

The `graph_ir.to_mlir()` method is available in Phase 1+:

```python
@tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
def my_kernel(Q, K, V):
    return tessera.ops.flash_attn(Q, K, V, causal=True)

# Graph IR text
print(my_kernel.graph_ir.to_mlir())
# tessera.flash_attn %Q, %K, %V {causal = true, dropout_p = 0.0}
# : (tensor<...xbf16>, tensor<...xbf16>, tensor<...xbf16>) -> tensor<...xbf16>
```

Tile IR and Target IR inspection helpers are Phase 4 planned. Until then, use
`tessera-opt` on the `.mlir` text output with the named pipeline flag:

```bash
tessera-opt my_module.mlir \
  --pass-pipeline="tessera-lower-to-gpu" \
  --mlir-print-ir-after-all \
  -o my_module_target.mlir
```

---

## References

- [`docs/spec/TARGET_IR_SPEC.md`](../spec/TARGET_IR_SPEC.md) — normative Target IR dialect and op semantics
- [`docs/spec/COMPILER_REFERENCE.md`](../spec/COMPILER_REFERENCE.md) — named pipelines and all pass definitions
- [`docs/spec/LOWERING_PIPELINE_SPEC.md`](../spec/LOWERING_PIPELINE_SPEC.md) — pass input/output IR contracts
- [`docs/spec/MEMORY_MODEL_SPEC.md`](../spec/MEMORY_MODEL_SPEC.md) — mbarrier, visibility, synchronization
- [`docs/spec/RUNTIME_ABI_SPEC.md`](../spec/RUNTIME_ABI_SPEC.md) — C ABI for kernel launch
- [`docs/guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md`](../guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md) — TMA and async copy patterns
