---
status: Informative
classification: Informative
authority: Target IR usage examples; normative op semantics and ABI in docs/spec/TARGET_IR_SPEC.md and docs/spec/RUNTIME_ABI_SPEC.md
last_updated: 2026-10-24
---

> **Current-state note (2026-05-20):** This is historical architecture guidance. Phase labels below are design lineage, not current support claims. For implementation status, use `docs/spec/COMPILER_REFERENCE.md`, `docs/audit/generated/support_table.md`, `docs/audit/generated/e2e_op_coverage.md`, and `docs/spec/VALIDATION_SPINE.md`.
>
> **Refreshed 2026-10-24:** op, pass, pipeline and `GPUTargetProfile` names below were re-checked against the code. Several MLIR snippets describe the original design rather than what the passes emit today; each such section now says so and names the real output. What actually executes on which device is recorded only in `docs/audit/generated/runtime_execution_matrix.md` — nothing in this guide is a device claim.

# Tessera Target IR — Usage Guide

Target IR is the **fourth and final layer** of the Tessera compilation pipeline. It receives Tile IR from the backend-specific lowering passes and is expressed in one toolchain-free, contract-carrying dialect per backend — `tessera_nvidia`, `tessera_rocm`, `tessera_apple` and `tessera_x86` (Decision #19) — which then lowers to upstream NVVM / ROCDL / LLVM or to a declared runtime ABI call. The fleet's machine primitives are WGMMA/`mma.sync`/TMA/mbarrier for NVIDIA, WMMA for RDNA AMD parts (MFMA is CDNA-only and hardware-gated), simdgroup / Metal 4 `matmul2d` for Apple, and AVX-512 for x86 (AMX is a retired target). Programmers do not write Target IR directly; they write Python using `@tessera.jit` and the compiler produces Target IR.

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
Tile IR     (tile.async_copy, tile.mma, tile.mbarrier.*, tessera_attn.*)
     │  [lower-tile-to-nvidia → lower-tessera-nvidia-to-nvvm]                    ← NVIDIA (exact per-arch)
     │  [TileToX86Pass]                                                           ← x86
     ▼
Target IR   (tessera_nvidia.{wgmma,mma_sync,tcgen05_mma,tma_async_copy,mbarrier,...} → NVVM → PTX)
             (func.call @tessera_x86_avx512_gemm_bf16 + tessera_x86.abi_call marker → x86 binary)
```

The main named lowering pipelines (authoritative pass lists live in
`python/tessera/compiler/pipeline_registry.py`, drift-gated against the C++
registrations; `tessera-opt --help` lists what a given build registered):

| Pipeline | Target | Passes |
|----------|--------|--------|
| `tessera-lower-to-x86` | x86 AVX-512 (AMX ops kept as IR contract only) | EffectAnnotation → Canonicalize → Distribution → layout / compute / IR-contract / symdim legality → Tiling → TileToX86 |
| `tessera-lower-to-gpu` | NVIDIA generic Tile-level chain (sm_90 / sm_100 / sm_120) | EffectAnnotation → Canonicalize → Distribution → legality gates → TileIRLowering → WarpSpecialization → pipeline / warp-spec / barrier-reuse legality → AsyncCopyLowering → TMADescriptor → legality + tile-dataflow legality |
| `tessera-lower-to-nvidia-sm{90,100,120}` | NVIDIA exact-arch Tile → `tessera_nvidia` → NVVM | `lower-tile-to-nvidia` → `lower-tessera-nvidia-to-nvvm` (sm_120 adds `convert-nvgpu-to-nvvm`) |
| `tessera-lower-to-rocm` | AMD ROCm family pipeline (typed GEMM subset) | see the registry |
| `tessera-lower-to-apple_{cpu,gpu}[-full,-runtime]` | Apple CPU (Accelerate) / GPU (Metal) | see the registry |

`NVWGMMALoweringPass` (`--tessera-nvwgmma-lowering`), `NVTMADescriptorPass`
(`--tessera-nvtma-descriptor`) and `NVFlashAttnKernelEmitter`
(`--tessera-nvflash-attn-emitter`) are still registered as standalone passes
and are what the examples below describe; they are no longer stages of
`tessera-lower-to-gpu`.

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

# CPU — no target= records the tessera-lower-to-x86 alias; execution may stay
# on the NumPy reference plan (see last_fallback_reason)
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
| `isa` | `ISA.SM_90` | `ISA.SM_80` / `SM_86` / `SM_89` / `SM_90` / `SM_100` / `SM_120` |
| `warps_per_cta` | 4 | must be power of 2 |
| `shared_mem_bytes` | `None` (SM default) | override per-kernel |
| `prefer_ptx` | `True` | |
| `pipeline_stages` | 2 | double-buffer depth |

Feature properties read the per-SM feature table (`cuda_feature_status`), not
an ISA ordering: `.supports_wgmma` is true on SM_90 / SM_100 only — consumer
Blackwell **sm_120 has no `wgmma`** and uses `mma.sync` — and `.supports_tma`
is true on SM_90 / SM_100 / SM_120.

---

## What Target IR Looks Like

NVIDIA Target IR ops live in the `tessera_nvidia` dialect
(`src/compiler/codegen/tessera_gpu_backend_NVIDIA/include/tessera/gpu/IR/TesseraNVIDIADialect.td`:
`wgmma`, `mma_sync`, `wmma`, `tcgen05_mma`, `tmem_{alloc,load,store}`,
`tma_async_copy`, `mbarrier`, `nvfp4_block_scale_mma`, `kernel_call`, …). They
are produced by the lowering passes from Tile IR — not authored by hand. There
are no `tessera.nvgpu.*`, `tessera.tma.*` or `tessera.tcgen05.*` dialects; the
snippets in this section that use those spellings show the original design, and
each says what the pass emits today.

### NVIDIA SM_90 (Hopper) — WGMMA + TMA

`NVWGMMALoweringPass` converts `tile.mma` into a WGMMA backend call:

```mlir
// tile.mma in Tile IR
%fragC = tile.mma %fragA, %fragB, %fragC
           {m = 64, n = 256, k = 32, accum = "fp32"}

// After NVWGMMALoweringPass (sm >= 90) — what the pass emits today
%c = func.call @tessera_nvidia_wgmma_mma_async_bf16_m64n64k16(...)
       {tessera.nvidia.shape = "m64n64k16", tessera.nvidia.dtype_ab = "bf16",
        tessera.nvidia.dtype_c = "f32"}
```

The pass replaces `tile.mma` with a `func.call` to a declared backend symbol
(`..._f16_...` for f16 operands) carrying the shape and dtype contract as
attributes; it does not emit `tessera_nvidia.wgmma`. Its gate is a numeric
`sm >= 90` comparison, which would also admit sm_120 (no `wgmma`; see
Decision #1's correction), so use it only for sm_90 / sm_100 — sm_120 kernels
go through `tessera-lower-to-nvidia-sm120` instead.

`NVTMADescriptorPass` stays in the Tile dialect: it deduplicates
`tile.tma.descriptor` ops by `(src_ptr, tile_rows, tile_cols)`, hoists one
descriptor per kernel to the preamble (Decision #9), and assigns mbarrier slots:

```mlir
// Before
%desc_q = tile.tma.descriptor(%Q) {tile_rows=64, tile_cols=64}
tile.tma.copy_async %desc_q {mbarrier_slot=0}

// After NVTMADescriptorPass — preamble mbarrier init + hoisted descriptor
tile.mbarrier.init {slots=2, phase_bits=2}
tile.tma.copy_async %Q_desc {mbarrier_slot=0, expect_tx=8192}
```

The `tessera_nvidia.tma_async_copy` / `tessera_nvidia.mbarrier` Target ops are
reached through the exact per-arch pipelines.

mbarrier tracking (original design sketch for `NVFlashAttnKernelEmitter`; the
pass inserts the arrive/wait sequence between producer and consumer, and the
`tessera.nvgpu.mbarrier.*` spellings below are not a registered dialect):

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
// Design sketch — NVFlashAttnKernelEmitter attaches launch bounds
// (nvvm.maxntidx, nvvm.minctasm) and tessera.nvidia.kernel /
// tessera.smem_bytes / tessera.tile_q / tessera.tile_kv / tessera.sm
// attributes to a func.func rather than emitting a `kernel` op.
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

Datacenter sm_100 only (sm_120 has no tcgen05/TMEM) and hardware-gated: no
fleet device. The registered ops are Tile `tile.tcgen05.mma` /
`tile.tmem.*` and Target `tessera_nvidia.tcgen05_mma` /
`tessera_nvidia.tmem_{alloc,load,store}`; the snippet below is the original
design spelling.

```mlir
// Blackwell TMEM MMA via tcgen05 (design sketch)
%tmem = tessera.tcgen05.alloc {cols = 128}

tessera.tcgen05.mma.async %tmem, %desc_a, %desc_b, %instr_desc
  {cta_group = 2, shape = [128, 256, 32],
   typeA = fp8_e4m3, typeB = fp8_e4m3, typeC = f32}

tessera.tcgen05.dealloc %tmem
```

### x86 — AVX-512 Function Calls

`TileToX86Pass` converts tiled matmul / `fused_epilogue` ops to C function
calls, and marks each one with a registered `tessera_x86.abi_call` op so the
delegated boundary stays visible in Target IR:

```mlir
// After TileToX86Pass — Target IR
func.call @tessera_x86_avx512_gemm_bf16(%a_ptr, %b_ptr, %c_ptr, %M, %N, %K, %beta)
  : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32) -> ()
tessera_x86.abi_call {symbol = "tessera_x86_avx512_gemm_bf16", abi = "tessera_x86_c"}

// With bias+GELU epilogue
func.call @tessera_x86_epilogue_bias_gelu_fp32(%c_ptr, %bias_ptr, %M, %N)
  : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()
```

The `tessera_x86` dialect
(`src/compiler/codegen/tessera_x86_backend/include/TesseraX86/IR/TesseraX86Ops.td`)
also declares `avx512_gemm_microkernel`, `pack_b_panel`, `elementwise`,
`kernel`, and the AMX `amx_tile_*` / `amx_dp*` ops. The pass can still select
`tessera_x86_amx_gemm_bf16`, but AMX is a retired target with no fleet
hardware; native x86 proof means AVX-512 on the Zen 5 hosts. See
`docs/audit/backend/x86/todo.md` for what is proven.

### AMD ROCm — MFMA / WMMA

ROCm Target IR is the `tessera_rocm` dialect (`TesseraROCMOps.td`: `mfma`,
`wmma`, `wmma_gemm`, `scaled_wmma_gemm`, `swmmac`, `async_copy`, `wait`, …).
The fleet's parts are RDNA and use **WMMA**: gfx1151 (RDNA 3.5, F16/BF16/IU8/IU4,
no FP8/BF8 WMMA) and gfx1201 (RDNA4, adds FP8/BF8 and 2:4 sparse `swmmac`).
Proof on one never transfers to the other. MFMA is CDNA-only and
hardware-gated; the snippet below is an upstream-ROCDL illustration for a part
the fleet does not have.

```mlir
// MFMA lowering for gfx90a (MI210/250) — CDNA, hardware-gated
%result = rocdl.mfma.f32.32x32x8.f16 %a, %b, %c
            {cbsz = 0, abid = 0, blgp = 0}
          : (vector<4xf16>, vector<4xf16>, vector<32xf32>) -> vector<32xf32>

// LDS async copy
rocdl.ds.read.b128 %addr : (!llvm.ptr<3>) -> vector<4xi32>
```

ROCm target support is split by architecture family. CDNA targets use MFMA
shapes where the inventory reports them; RDNA4 (`gfx1200` / `gfx1201`) is
modeled as WMMA-only, including FP8/BF8 and planned-gated INT4 coverage —
`gfx1201` has device proof on Tajasarus, `gfx1200` has none. See
`docs/backends/rocm/kernel-inventory.md` and
`docs/audit/backend/BACKEND_AUDIT.md` for the current dtype and
instruction matrix.

---

## Warp Specialization in the Pipeline

`WarpSpecializationPass` (Phase 3; the warp-role split targets sm_90 / sm_100) assigns warps to producer or consumer roles
before the WGMMA and TMA passes run. This creates the double-buffering structure:

```mlir
// After WarpSpecializationPass
tessera.schedule.warp {tile.warp_role = "producer", tile.pipeline = "pipe0"} {
    // Warps 0-1: issue TMA async loads
    %ps = tile.pipeline_init {depth = 2, stage = 0, phase = 0, role = "producer"} : !tile.pipeline_state
    %q_tile, %tok_q = tile.async_copy %Q_global, %Q_shared {stage = 0, vector = 16}
    %k_tile, %tok_k = tile.async_copy %K_global, %K_shared {stage = 0, vector = 16}
    %ps1 = tile.pipeline_advance %ps, %tok_q, %tok_k : !tile.pipeline_state
}

tessera.schedule.warp {tile.warp_role = "consumer", tile.pipeline = "pipe0"} {
    // Warps 2-3: wait for data, run WGMMA
    %ps = tile.pipeline_init {depth = 2, stage = 0, phase = 1, role = "consumer"} : !tile.pipeline_state
    %acc = tile.mma %fragA, %fragB, %fragC {m = 64, n = 256, k = 32, accum = "fp32"}
    %ps1 = tile.pipeline_advance %ps, %acc : !tile.pipeline_state
}
```

Producer and consumer roles use separate register files and barrier slots.
The `!tile.pipeline_state` + `!tile.async_token` SSA chains are hard
boundaries — not advisory annotations. (Earlier revisions of this guide showed
`tessera.queue.push/pop` here; the `tessera.queue` MLIR dialect was deleted
2026-08-10 under Decisions #29/#31 and was never what the pass emitted.)

---

## ISA Feature Matrix

| Feature | SM_80 (A100) | SM_90 (H100) | SM_100 (B100) | SM_120 (RTX 50) |
|---------|-------------|-------------|--------------|----------------|
| WGMMA | ❌ | ✅ | ✅ | ❌ (`mma.sync`) |
| TMA / cp.async.bulk | ❌ | ✅ | ✅ | ✅ |
| mbarrier (transaction) | ❌ | ✅ | ✅ | ✅ |
| tcgen05 / TMEM | ❌ | ❌ | ✅ | ❌ |
| FP8 (e4m3 / e5m2) | ❌ | ✅ | ✅ | ✅ |
| NVFP4 | ❌ | ❌ | ✅ | ✅ (`mma.sync` block-scale) |
| Fallback (WMMA + cp.async) | ✅ | ✅ | ✅ | ✅ |

This is the ISA capability table in `python/tessera/compiler/gpu_target.py`,
not an execution claim: sm_90 and sm_100 are hardware-gated (no fleet device),
and sm_120 evidence comes from the RTX 5070 only.

Below SM_90, `NVWGMMALoweringPass` falls back to a
`tessera_nvidia_wmma_mma_sync_bf16_m16n16k16` backend call.

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

Tile IR and Target IR are inspectable through `fn.explain()` and the lower-level
`schedule_ir`, `tile_ir`, `target_ir`, and `lowering_artifacts()` accessors.
For C++ pass debugging, use `tessera-opt` on the `.mlir` text output with the
named pipeline flag:

```bash
tessera-opt my_module.mlir \
  --tessera-lower-to-gpu \
  --mlir-print-ir-after-all \
  -o my_module_target.mlir
```

The exact per-arch `tessera-lower-to-nvidia-sm*` pipelines are only registered
in a build with the NVIDIA dialect enabled (it is off by default); use
`tessera-opt --help` to see what yours has.

---

## References

- [`docs/spec/TARGET_IR_SPEC.md`](../spec/TARGET_IR_SPEC.md) — normative Target IR dialect and op semantics
- [`docs/spec/COMPILER_REFERENCE.md`](../spec/COMPILER_REFERENCE.md) — named pipelines and all pass definitions
- [`docs/spec/LOWERING_PIPELINE_SPEC.md`](../spec/LOWERING_PIPELINE_SPEC.md) — pass input/output IR contracts
- [`docs/spec/MEMORY_MODEL_SPEC.md`](../spec/MEMORY_MODEL_SPEC.md) — mbarrier, visibility, synchronization
- [`docs/spec/RUNTIME_ABI_SPEC.md`](../spec/RUNTIME_ABI_SPEC.md) — C ABI for kernel launch
- [`docs/guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md`](../guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md) — TMA and async copy patterns
