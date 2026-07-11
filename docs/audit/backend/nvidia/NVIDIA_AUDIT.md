---
last_updated: 2026-07-07
audit_role: sub_audit
---

# NVIDIA Backend Audit

This document consolidates NVIDIA-specific audit material.

> **Real-hardware bring-up:** see [`BLACKWELL_SM120_EXECUTION_PLAN.md`](BLACKWELL_SM120_EXECUTION_PLAN.md)
> — the sm_120 target model is correctly grounded as **Blackwell consumer** (RTX 5070 Ti),
> not the old "Rubin placeholder" (no wgmma/tcgen05/TMEM; FP4 via `mma.sync.block_scale`).
> **Status (2026-06-24, #106):** the first real NVIDIA `backend_kernel` is **proven on
> silicon** — a sm_120 `mma.sync` bf16 matmul executes end-to-end on the RTX 5070 Ti
> (`emit_mma_sync_matmul_ptx` → PTX → assemble → CUDA launch bridge `tsrRegisterGpuLauncher`
> → execute-and-compare), under CUDA 13.3. The `mma.sync` emit path the old "Key gap" called
> for now exists. **Still open:** broaden sm_120 beyond matmul (flash-attn family), and the
> separate Hopper sm_90 (WGMMA) / datacenter sm_100 (tcgen05) emit paths — `mma.sync` ≠ WGMMA,
> so each arch needs its own proof.

## Finished

- NVIDIA target-map generation exists at
  `../../generated/nvidia_sm90_target_map.md`.
- CUDA/NVIDIA toolchain and execute-and-compare plans are documented.
- The repo distinguishes NVIDIA artifact generation from hardware execution.
- Compiler pass-order and lit-style structural work exists for NVIDIA-oriented
  paths.
- **First real NVIDIA `backend_kernel` hardware proof — LANDED 2026-06-25
  (RTX 5070 Ti, Blackwell consumer sm_120, CUDA 13.3, driver 610.62).** The
  full rung ladder closed end-to-end; see
  [`BLACKWELL_SM120_EXECUTION_PLAN.md`](BLACKWELL_SM120_EXECUTION_PLAN.md)
  §"Sequencing when the box lands" for the on-silicon record. Concretely:
  - **Execution row now exists** in
    `../../generated/runtime_execution_matrix.md` — the executable
    `(nvidia_sm120, nvidia_mma)` row (executor `nvidia_mma`, `native_gpu`,
    `cuda_runtime`). sm_120 was removed from `_UNIMPLEMENTED_TARGETS`.
  - **Manifest promoted** `artifact_only → hardware_verified` for
    `matmul` on `nvidia_sm120` (carries both `runtime_symbol`
    `tessera_nvidia_mma_gemm_bf16` and an `execute_compare_fixture`).
  - **Execute-and-compare proof** vs a bf16-rounded CPU reference passed on
    silicon (max abs err 4.8e-7 — f32 epsilon — 0/128 off).
  - **Shipped runtime symbol:** `libtessera_nvidia_gemm.so` (CMake target
    `tessera_nvidia_gemm`) exporting
    `tessera_nvidia_mma_gemm_{bf16,f16,tf32,e4m3,e5m2}`, NVRTC-compiled for the
    device arch (tiled/K-looped `mma.sync` GEMM).
  - **Emit path:** `ptx_emit.emit_mma_sync_matmul_ptx` (+ validators) emits the
    complete sm_120 `mma.sync.aligned.m16n8k16` kernel (parallel to the
    existing sm_90a WGMMA emitter — the WGMMA path does not run on sm_120).
  - **CUDA launcher** registered into the C-ABI bridge `tsrRegisterGpuLauncher`
    (loads the emitted PTX via the Driver API, runs through `tsrLaunchKernel`).
  - **Tests:** `tests/unit/test_nvidia_mma_runtime_symbol.py` (dlopen +
    numerical validation of all 5 dtypes vs numpy/ml_dtypes),
    `test_conformance_execute_compare_nvidia.py` (launch-bridge
    execute-and-compare), `test_nvidia_launch_execute.py` (matrix row +
    `@jit(target="nvidia_sm120")` default). All are skip-clean off-device.
  - **`@jit` lane:** `@jit(target="nvidia_sm120")` matmul dispatches through the
    shipped symbol on a capable host; stays `artifact_only` off-device (no
    behavior change when no GPU is present).
  - Spike artifacts (kernel + driver harness + smem device-query + reproduce
    steps) committed at `spikes/sm120_mma_sync/`.

## Still Open

The original "no execution row / not hardware-proven" gaps are **closed**
(above), and the **compiler-generated lane is now landed + hardware-proven** on
sm_120 (RTX 5070 Ti, PRs #290–#297):

- **Compiler-GENERATED NVIDIA lane — LANDED.** `emit/nvidia_cuda.py` is a full
  three-seam plugin (emitter + `nvcc` compile + ctypes runner) that synthesizes,
  compiles, and launches kernels in-process for **all four `fusion_core` region
  kinds** — fused matmul-epilogue, flash-attention (C4), SwiGLU gate + pointwise
  DAG (C5) — each F4-gated on-device. The emit-path `mma.sync` GEMM
  (`ptx_emit.py` → the shipped `tessera_nvidia_ptx_launch` bridge: driver-JIT +
  `cuLaunchKernel`) executes the *emitted* PTX, distinct from the hand-shipped
  `libtessera_nvidia_gemm` symbol. Both are first-class D1 arbiter candidates:
  the shipped GEMM is **Tier-3 hand-tuned**, the emitted GEMM **Tier-2 emitted**
  (B1), with D2 measured autotune + D3 fallback logging choosing/observing between
  them. So NVIDIA now has the `rocm_compiled` analog it lacked, plus the arbiter
  surface.
- **NVFP4 block-scale matmul (#9)** — **emit + ptxas-assemble landed** (#291,
  `emit_nvfp4_block_scale_mma_ptx`); on-device execution + non-unit-scale numerics
  stay gated on the PTX-ISA scale-distribution spec. The warp `mma.sync…block_scale`
  instruction already assembles + executes on `sm_120a` (see
  `spikes/sm120_mma_sync/`); productization is pending the PTX ISA
  scale-distribution spec for numerics.
- **Other NVIDIA SMs stay `artifact_only`** — sm_80/90/100 are proven only on
  sm_120 silicon; promoting them needs their own hardware (Hopper box for
  sm_90a WGMMA; datacenter Blackwell for sm_100 `tcgen05`/TMEM).
- **MLIR Target IR dialect — typed (Decision #19), NVVM lowering still marker-only.**
  The hardware-proven sm_120 lanes above run through the **Python** emit path
  (`emit/nvidia_cuda.py` / `ptx_emit.py`), *not* the MLIR `tessera_nvidia` Target
  IR. The `tessera_nvidia` dialect was `isExtensible` with **zero registered ops**
  (generic ops, no verifier, `--allow-unregistered-dialect` to print) — a
  Decision #19 violation vs. the typed ROCm/Apple dialects. **Increment 1 (landed):**
  `TesseraNVIDIADialect.td` now defines 10 typed ops (`mma_sync`, `wgmma`,
  `tcgen05_mma`, `wmma`, `tma_async_copy`, `mbarrier`, `tmem_{alloc,load,store}`,
  `cuda_kernel`); `LowerTileToNVIDIA` populates them via the unchanged generic
  builders (`usePropertiesForAttributes=0`), they round-trip/verify without
  `--allow-unregistered-dialect`, and `allowUnknownOperations()` is dropped so a
  malformed `tessera_nvidia.*` op is an error. Proof:
  `test/nvidia/nvidia_target_ir_typed.mlir` + all existing NVIDIA fixtures
  unregressed. **Still marker-only:** `LowerNVIDIAToNVVM` rewrites every typed op
  to a void `llvm.nvvm.*.contract` marker — no real `NVVM::MmaOp`/`WgmmaOp`/
  `tcgen05` intrinsic (see Next Work #6).
- **`flash_attn` on `nvidia_sm120`** — **proven on hardware 2026-07-07** (C4): the
  synthesized flash-attention CUDA lane (`emit/nvidia_cuda.py`
  `NvidiaFlashAttnCandidate` / `run_fused_attention`) computes
  `O = softmax(scale·Q·Kᵀ)·V` with a one-query-per-thread online softmax, executes
  on sm_120, and matches the numpy reference across scale/causal/shape
  (`test_nvidia_plugin.py::test_live_nvidia_flash_attention`), passing the same
  universal F4 oracle. An mma.sync tensor-core flash version is the perf follow-on.

## Next Work

Done (2026-07-07): the compiler-generated lane (#290–#297), the sm_120 `mma.sync`
flash-attention execute-compare (C4), and the sm_120 kernel-inventory doc
(`docs/nvidia_sm120_mma_sync_kernel_inventory.md`). Remaining:

1. **NVFP4 block-scale execution + numerics** — bind the fp4 fragment packing and
   flip the manifest row once execute-and-compare passes on `sm_120a` and the
   scale-distribution numerics are grounded (emit + ptxas-assemble already land).
2. **mma.sync tensor-core FUSED + FLASH-ATTENTION lanes — LANDED** (Tier-2):
   `NvidiaMmaFusedCandidate` — a warp-tiled `mma.sync.m16n8k16` f16 GEMM +
   bias/activation epilogue, ~6x faster than the scalar generic lane on 512³; and
   `NvidiaMmaAttnCandidate` — two `mma.sync` matmuls with a smem-staged row softmax
   (sidesteps the accumulator→operand fragment shuffle), ~2.7x faster on
   64×512×64×64. Both F4-gated within the f16 budget, arbiter-preferred over Tier-1.
   The attention lane **gates on softmax sharpness** (`scale·D·amax²`) + operand
   magnitude, delegating large-scale/large-magnitude f32 attention to the exact
   scalar lane so it never silently degrades f32 semantics. **Still open:** the
   **D2 fleet-shared autotune corpus** (persist `MeasureCache`, Theory §7.5).
3. **Dtypes beyond f32** for the fused / attention / gated lanes (16-bit storage
   is served by the B1 matmul lane today).
4. **`wgmma` sm_90a** — complete the instruction-encoding skeleton into a real
   Hopper WGMMA kernel (assemble-only until a Hopper box) — and **sm_100 tcgen05**.
5. Promote sm_80/90/100 manifest rows only when their own silicon is available
   and the generated dashboards agree.
6. **Native NVVM lowering for the typed Target IR (Tile IR / Target IR tail).**
   Increment 1 typed the `tessera_nvidia` dialect; increment 2 replaces the
   `LowerNVIDIAToNVVM` void-marker path with real NVVM emission, starting with
   `tessera_nvidia.mma_sync` → `NVVM::MmaOp` (the one op already hardware-proven on
   sm_120, so it is validatable on the RTX 5070 Ti), then `wgmma`/`tcgen05`/TMEM as
   their silicon lands. This converges the MLIR Target IR path with the Python emit
   path that currently carries execution.

## Source Material Consolidated

- `archive/nvidia_execution_audit.md`
- `../archive/nvidia_rocm_execute_and_compare_plan.md`

