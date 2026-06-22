# ROCm Backend Audit

This document consolidates ROCm-specific audit material.

> **Real-hardware bring-up:** see [`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md)
> — the gfx1151 (RDNA 3.5 / Ryzen AI Max+ 395) target model is now grounded in the
> RDNA3.5 ISA (WMMA 16×16×16, no FP8), and the doc lays out the rung ladder to the
> first real non-Apple `backend_kernel` execution proof (emit → assemble → HIP-launch →
> execute-and-compare). This is the unblock for the "Still Open" / "Next Work" items below.
>
> **Design patterns from the AMD ROCm ecosystem:** see
> [`ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md`](ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md) — a
> source-grounded survey of AITER, ATOM, hipBLASLt, rocWMMA, Mori, Iris, XIO, and the
> AMD Gluon GEMM tutorial, with ranked, Tessera-mapped patterns (hardware-free IR/dispatch
> wins to adopt now, the GEMM perf ladder for Strix Halo bring-up, and the GPU-initiated
> comm track).

## Finished

- ROCm target-map generation exists at `../../generated/rocm_target_map.md`.
- ROCm/gfx target handling and HIP toolchain gates are represented.
- The execute-and-compare plan covers ROCm alongside NVIDIA.
- ROCm sub-arch gating was corrected so missing HIP toolchain is reported on
  the right axis.

## Box landed (2026-06-22) — toolchain gates cleared

A Strix Halo box (Ryzen AI Max+ 395) is now available: Ubuntu 24.04 (WSL2),
ROCm **7.2.4**, LLVM/MLIR **22.1.8**. The iGPU enumerates as **`gfx1100`**
(RDNA3 profile; same 16×16×16 WMMA family as gfx1151, no FP8 WMMA) — see
[`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md) "Bring-up status".
Cleared: `rocminfo` enumerates without `HSA_OVERRIDE`; `hipcc` compiles WMMA for
gfx1100; ROCm lit suite 11/11; `tessera-opt`/`tessera-rocm-opt` build clean.

**Stage A increment landed (2026-06-22):** `lower-tile-to-rocm` was emitting
`tessera_rocm.mfma` for every arch — wrong for RDNA. Added a `tessera_rocm.wmma`
op + arch-keyed selection (`gfx11xx` → WMMA, CDNA → MFMA, no-FP8-on-RDNA gate
preserved) + a `llvm.amdgcn.wmma.contract` ROCDL marker, with lit fixtures.

**Stage B verified on the box (2026-06-22):** `rocdl_emit.py` (the AMD analog of
`ptx_emit.py`) already emits `llvm.amdgcn.wmma.*` LLVM IR and `llc`-assembles it
to real `v_wmma_*` AMDGCN; now runs for real on the box (LLVM 22.1.8 AMDGPU `llc`)
and is parametrized over **gfx1100** (the box target). Added `llc_object()` — the
GEMM lowers to a real AMD GPU ELF object (`EM_AMDGPU`); `_find_llc()` now finds the
apt.llvm.org `llc`. `test_rocdl_emit.py`: 96 passed, 0 skipped. **Note:** the MLIR
`--tessera-emit-rocdl` pipeline aborts here (`tessera-to-linalg` pass unregistered
in `tessera-opt`) — a separate follow-up; Stage B rides the direct LLVM-IR emitter.

**Stage C verified on the box (2026-06-22):** a real GEMM kernel **executes on
the gfx1100 device through the C-ABI launch bridge** (`tsrLaunchKernel` →
registered `tsrGpuLauncherFn` → HIP launch) and matches `A @ B`; unregistered
kernels still report `UNIMPLEMENTED`. First non-Apple kernel through the bridge.
Mirrors the Apple G7 proof; test `test_runtime_abi_rocm_launch_bridge.py`. Fixed
the runtime CMake HIP-include bug (`tessera_runtime` now links `hip::host`) and
handled the WSL `hipGetDeviceCount==0` quirk (probe-based skip).

**Stage D verified on the box (2026-06-22):** the real RDNA **WMMA** matrix op
(`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, the same `v_wmma_f32_16x16x16_f16`
`rocdl_emit.py` emits) executes on the device and produces a numerically correct
16×16×16 `f32←f16` GEMM, routed through the C-ABI bridge — maxerr ≈ 3e-8
standalone / < 1e-2 through the bridge. Test
`tests/unit/test_rocm_wmma_execute_compare.py`. This clears the *numerical-proof*
half of the `hardware_verified` contract.

## Still Open

- **The `backend_kernel` / `hardware_verified` flip is NOT done — and is gated,
  honestly.** `hardware_verified` requires a *shipped* `runtime_symbol` (an
  auto-registered ROCm runtime launcher) in addition to the now-satisfied
  `execute_compare_fixture`. The Stage C/D kernels + launcher live in the test
  harness (like Apple G7), so promoting the manifest row now would be Decision
  #25 inflation. The flip becomes mechanical once the launcher ships.
- No ROCm execution row in `../../generated/runtime_execution_matrix.md` yet
  (same shipped-launcher gate).
- Stage D proof is a single 16×16×16 tile + `f32←f16` only; general tiled/
  K-looped GEMM and the bf16 combo (documented gfx115x bugs) are follow-ups.

## Next Work

1. ✅ **Stage B — assemble (2026-06-22):** `rocdl_emit.py` emits the WMMA GEMM
   LLVM IR and `llc -mcpu=gfx1100` lowers it to real `v_wmma_*` AMDGCN + an
   AMD GPU ELF object; verified on the box, gfx1100 + gfx1151.
2. ✅ **Stage C — launch (2026-06-22):** a GEMM executes through the C-ABI
   bridge on gfx1100 and matches `A @ B`; runtime HIP build fixed.
3. ✅ **Stage D — prove (2026-06-22):** the WMMA `f32←f16` GEMM executes through
   the bridge and matches a host reference (`test_rocm_wmma_execute_compare.py`).
4. **Ship an auto-registered ROCm runtime launcher** (move the harness WMMA
   kernel + launcher into a hipcc-built backend lib that auto-registers via
   `tsrRegisterGpuLauncher` with a real shipped `runtime_symbol`). This is the
   one remaining gate to (a) promote the gfx1151 matmul manifest row to
   `hardware_verified`, (b) flip `backend_kernel`, and (c) add a
   `runtime_execution_matrix` ROCm row. Then extend to tiled/K-looped GEMM + bf16.
5. Register `tessera-to-linalg` into `tessera-opt` so the MLIR-graph
   `--tessera-emit-rocdl` route works (Stage B currently rides the direct emitter).
6. Promote manifest rows only after generated dashboards agree.

## Source Material Consolidated

- `../archive/nvidia_rocm_execute_and_compare_plan.md`

