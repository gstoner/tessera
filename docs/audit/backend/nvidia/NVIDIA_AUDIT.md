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
(above). What remains is breadth beyond the first proven kernel:

- **Compiler-GENERATED NVIDIA lane** — the analog of ROCm's `rocm_compiled`
  (a `tessera-opt` NVIDIA pipeline that generates + serializes the kernel
  in-process, rather than dispatching the hand-shipped `libtessera_nvidia_gemm`
  symbol). Today's `nvidia_mma` lane mirrors `rocm_wmma` (shipped symbol), not
  the compiler-generated path.
- **NVFP4 block-scale matmul (#9)** — the warp `mma.sync…block_scale`
  instruction already assembles + executes on `sm_120a` (see
  `spikes/sm120_mma_sync/`); productization is pending the PTX ISA
  scale-distribution spec for numerics.
- **Other NVIDIA SMs stay `artifact_only`** — sm_80/90/100 are proven only on
  sm_120 silicon; promoting them needs their own hardware (Hopper box for
  sm_90a WGMMA; datacenter Blackwell for sm_100 `tcgen05`/TMEM).
- **`flash_attn` on `nvidia_sm120`** remains unproven on hardware (the matmul
  lane is the first and only `hardware_verified` NVIDIA row so far).

## Next Work

1. Build the compiler-generated NVIDIA lane (the `rocm_compiled` analog) via a
   `tessera-opt` NVIDIA pipeline; add its executable matrix row.
2. Land NVFP4 block-scale matmul once the scale-distribution numerics are
   grounded; flip its manifest row when execute-and-compare passes on `sm_120a`.
3. Bring the sm_120 `mma.sync` flash-attention forward to the same
   execute-and-compare bar (attention analog of the GEMM proof).
4. Author an sm_120 `mma.sync` kernel inventory (sibling to
   `docs/nvidia_cuda13_kernel_inventory.md`).
5. Promote sm_80/90/100 manifest rows only when their own silicon is available
   and the generated dashboards agree.

## Source Material Consolidated

- `archive/nvidia_execution_audit.md`
- `../archive/nvidia_rocm_execute_and_compare_plan.md`

