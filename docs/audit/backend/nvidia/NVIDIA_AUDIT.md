---
last_updated: 2026-07-14
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
  - **Tests:** `tests/device/nvidia/test_mma_runtime_symbol.py` (dlopen +
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
- **NVFP4 block-scale matmul (#9)** — **execute-and-compare complete** on the
  RTX 5070 Ti `sm_120a`. Uniform 0.5/1/2 and distinct per-block UE4M3 scale
  matrices match exactly; the PTX 9.3 selector ABI and
  `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` SASS are regression-tested. The old
  128/128 failure was an unsigned-host-buffer conversion bug in the spike.
- **Other NVIDIA SMs stay `artifact_only`** — sm_80/90/100 are proven only on
  sm_120 silicon; promoting them needs their own hardware (Hopper box for
  sm_90a WGMMA; datacenter Blackwell for sm_100 `tcgen05`/TMEM).
- **MLIR Target IR dialect — typed (Decision #19), NVVM lowering still marker-only.**
  The hardware-proven sm_120 lanes above run through the **Python** emit path
  (`emit/nvidia_cuda.py` / `ptx_emit.py`), *not* the MLIR `tessera_nvidia` Target
  IR. The `tessera_nvidia` dialect was `isExtensible` with **zero registered ops**
  (generic ops, no verifier, `--allow-unregistered-dialect` to print) — a
  Decision #19 violation vs. the typed ROCm/Apple dialects. **Increment 1 (landed):**
  `TesseraNVIDIADialect.td` now defines typed ops: the inner contract ops
  (`mma_sync`, `wgmma`, `tcgen05_mma`, `wmma`, `tma_async_copy`, `mbarrier`,
  `tmem_{alloc,load,store}`, `cuda_kernel`) the C++ lowering emits, plus the
  Python-emitter wrapper/probe ops (`func` region wrapper, `kernel_call`,
  `profiler_probe`) so the *whole* emitted `tessera_nvidia` surface parses under
  the now-non-extensible dialect (Codex review, PR #371);
  `LowerTileToNVIDIA` populates them via the unchanged generic
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
(`docs/backends/nvidia/sm120-kernel-guide.md`). Remaining:

1. **NVFP4 block-scale productization** — execution, fragment packing, scale
   distribution, and numerics are now grounded on `sm_120a`; connect the proven
   oracle to general-shape runtime dispatch and the Tile fragment contract.
2. **mma.sync tensor-core FUSED + FLASH-ATTENTION + GATED lanes — LANDED**
   (Tier-2, f16 and bf16 storage): `NvidiaMmaFusedCandidate` — a warp-tiled
   `mma.sync.m16n8k16` GEMM +
   bias/activation epilogue, ~6x faster than the scalar generic lane on 512³; and
   `NvidiaMmaAttnCandidate` — two `mma.sync` matmuls with a smem-staged row softmax
   (sidesteps the accumulator→operand fragment shuffle), ~2.7x faster on
   64×512×64×64; and `NvidiaMmaGatedCandidate` — paired gate/up projections
   sharing A followed by the gate activation and multiply. Each has explicit
   f16/bf16 candidates and storage-policy applicability, with on-device ragged
   execute-and-compare coverage. They are arbiter-preferred over Tier-1.
   The attention lane **gates on softmax sharpness** (`scale·D·amax²`) + operand
   magnitude, delegating large-scale/large-magnitude f32 attention to the exact
   scalar lane so it never silently degrades f32 semantics. **D2 corpus landed
   and consumed:** measured, device-keyed `nvidia:sm_120` rows cover square and
   rectangular bf16 matmul, small/large fused GEMM+bias+GELU, and short/long
   causal attention; see `benchmarks/nvidia/record_autotune_corpus.py`. Normal
   `run_arbitrated()` dispatch consults the matching persisted
   device/op/shape-bucket/dtype verdict before tier priority, while retaining F4
   verification and explicit-force precedence. Expand this corpus when a new
   candidate, representative workload, or target device is introduced.
3. **Dtype breadth after f16/bf16 — LANDED for executable composition
   (2026-07-14).** The shipped GEMM ABI now dispatches all five implemented
   symbols: f16, bf16, TF32 math over f32 storage, E4M3, and E5M2. TF32/FP8
   `matmul_relu`, `matmul_softmax`, attention, and gated projections execute as
   explicit multi-launch compositions with generated CUDA softmax/ReLU/gate
   stages, one-stream device-resident intermediates, CUDA-event timing, and
   declared numerical budgets. The f16/bf16 candidates remain the
   native single-kernel fused performance lanes; TF32/FP8 are not mislabeled as
   fused until equivalent native kernels have their own execution oracle.
4. **`wgmma` sm_90a** — complete the instruction-encoding skeleton into a real
   Hopper WGMMA kernel (assemble-only until a Hopper box) — and **sm_100 tcgen05**.
5. Promote sm_80/90/100 manifest rows only when their own silicon is available
   and the generated dashboards agree.
6. **Native NVVM lowering for the typed Target IR (Tile IR / Target IR tail).**
   Increment 1 typed the `tessera_nvidia` dialect. **Increment 2 (landed):**
   `LowerNVIDIAToNVVM` now emits a **real `nvvm.mma.sync`** (not a void marker) for
   the canonical **m16n8k16 f16/bf16** fragment contract. f16 A/B registers are
   `vector<2xf16>`; bf16 uses NVVM's packed-i32 ABI, with the backend owning the
   `vector<2xbf16>` bitcast. Both are built via the dedicated `NVVM::MmaOp`
   builder (row/col layout) and **validated by
   the NVVM verifier** (a real structural correctness signal without a device). It is
   gated on the fragment types: the abstract tile→target form (scalar operands,
   `dtype_ab="bf16"`) carries no fragments and falls through to the honest marker
   (Decision #21). Proof: `test/nvidia/nvidia_mma_sync_to_nvvm.mlir`.
   **Fragment-form bridge landed:** canonical `tile.mma` with A×4/B×2/C×2
   `vector<2xf16>` operands now feeds the real intrinsic; proof:
   `test/nvidia/sm120_fragment_tile_to_nvvm.mlir`. **A/B pointer materialization
   landed:** canonical layout-bearing `tile.view` + `tile.fragment_pack` emits
   four A / two B `vector<2xf16>` loads and reaches real `nvvm.mma.sync`; proof:
   `test/nvidia/sm120_pointer_fragment_pack.mlir`, with wrong-layout rejection in
   `sm120_pointer_fragment_pack_invalid.mlir`. **Accumulator/output and execution
   landed:** `tile.fragment_unpack` + `tile.store` lower the four f32 accumulator
   registers per lane to the canonical row-major 16x8 output mapping. The LLVM
   kernel translates to PTX, assembles to an sm_120 cubin, launches through the
   CUDA Driver API, and matches NumPy on the RTX 5070 Ti; proof:
   `sm120_pointer_fragment_store.mlir` and
   `tests/unit/test_nvidia_tile_fragment_compiler_path.py`; explicit bf16 pack
   proof is `sm120_pointer_fragment_store_bf16.mlir`. **Dtype fragment breadth
   landed:** TF32 m16n8k8, E4M3/E5M2 m16n8k32, and s8 m16n8k32 layout-bearing
   packs lower, assemble, launch, and compare on the RTX 5070 Ti. LLVM 22's NVVM
   verifier handles TF32/int8; FP8 uses narrowly typed inline PTX until the
   dialect accepts those valid encodings. Still open are
   `wgmma`/`tcgen05`/TMEM on their silicon. This converges the MLIR Target IR
   path with the Python emit path that currently carries execution. The
   cross-vendor Tile view/fragment-pack/unpack contract and delivery order are in
   [`tile_fragment_abi.md`](../../../architecture/proposals/tile_fragment_abi.md).
   **Useful launch kernel landed:** portable `tile.matmul_kernel` and
   `#tile.epilogue` retain a direct-global one-warp baseline and add a shared
   four-warp 32x32 CTA macro-tile. A[32,16]/B[16,32] panels are cooperatively
   staged; each warp executes two adjacent m16n8 MMAs, amortizing each barrier
   interval over eight CTA MMAs. The path masks ragged M/N/K traffic, carries
   eight f32 accumulator registers per warp through the K-panel loop, and reuses
   the output mapping for bias, ReLU, f32/f16 conversion, and masked store. On
   the RTX 5070 Ti it is 1.20x faster than direct loads at 1024³ and 1.37x at
   2048³, but slower at 256/512, so shape-aware dispatch remains required. The
   generated sm_120
   cubin passes aligned and ragged execute-compare on the RTX 5070 Ti; structural
   proof is `sm120_matmul_kernel.mlir`, numerical proof is
   `test_nvidia_tile_fragment_compiler_path.py`.
   **Size-aware production dispatch landed:** the runtime compiles the direct and
   shared Tile schedules through the C++ Tile→NVVM pipeline, registers both PTX
   entries with the launch bridge, and exposes them as
   `nvidia_tile_matmul_{direct,shared}` candidates. D2 measures them alongside
   the shipped and legacy-emitted GEMMs and persists winners by
   `(device, shape bucket, dtype)`. The bridge uses the schedule-specific grid,
   block size, and i64 dimension ABI; ragged f16/bf16 execute-and-compare passes.
   The committed sm_120 corpus covers 64/256/512/1024/2048 square buckets for
   both storage dtypes, so ordinary dispatch consumes evidence while explicit
   forcing still isolates either compiler schedule.
   **Device-resident timing landed:** `timing="device"` uses CUDA driver events
   around repeated launches after one-time allocation/upload, stores a separate
   corpus row from `end_to_end`, and excludes candidates lacking a resident
   measurement hook. The committed f16/bf16 rows prove the direct→shared
   crossover between 512³ and 1024³ and select shared at 1024/2048.

7. **sm_120 op×target conformance backlog — initial queue CLOSED
   (2026-07-14).** The generated
   [`op_target_conformance.md`](../../op_target_conformance.md) dashboard is the
   authoritative completion check; do **not** edit it directly. All four
   formerly open `nvidia_sm120` cells below now have compile, live execution,
   and numerical fixtures:

   | Order | Operation | Status | Implementation / proof |
   |---:|---|---|---|
   | A | `matmul_relu` | Complete | Canonical Tile accumulator epilogue for f16/bf16; TF32/FP8 use shipped MMA + generated CUDA ReLU. Bias and ragged stores execute-and-compare on sm_120. |
   | B | `matmul_softmax` | Complete | Device-resident GEMM→row-softmax composition, including TF32/E4M3/E5M2 execution and event timing. |
   | C | `kv_cache_read` | Complete | Stable paged f32 ABI with logical→physical tables, arbitrary crossings, guards, and resident gather→attention decode with only final D2H. |
   | D | `conv2d` | Complete | Guarded direct, shared-weight tiled, and im2col→TF32 routes with numerical fixtures and event-timed size dispatch. D2 selects direct at 1×32×32×32/3×3×64 and im2col→TF32 at 1×64×64×64/3×3×64 on this RTX 5070 Ti. |

   `matmul`, `softmax`, and `flash_attn` were already complete on sm_120. Re-run
   `python -m tessera.cli.conformance_matrix --render` after each promotion; the
   dashboard and its drift test are the completion gate. Keep NVFP4 separate from
   this queue until its block-scale ABI/numerics are grounded.

8. **Serving evidence landed (2026-07-14).** ReplaySSM's ordered multi-slot
   ring now exposes start/completion event timing while retaining its opaque
   device-buffer lease. `benchmarks/baselines/nvidia_sm120_serving.json` records
   live async ReplaySSM and resident paged-KV decode. For the measured 16-token
   wave, analytical recurrent-state traffic falls 20.4× at B1/D128/N64 and 25.5×
   at B1/D256/N128; device time is 0.098 and 0.101 ms/token. Paged decode now
   compares a single-launch kernel that reads the page table directly against
   the staged resident plan. D2 selects fused at 128 tokens (0.107 vs 0.320 ms)
   and staged at 512 (0.374 vs 0.434 ms) and 2048 (0.469 vs 1.722 ms), then
   warm-starts runtime selection from those committed rows. The corpus also
   carries device-only TF32/E4M3/E5M2 fused, attention, gated, convolution, and
   ReplaySSM route measurements.

## Source Material Consolidated

- `archive/nvidia_execution_audit.md`
- `../archive/nvidia_rocm_execute_and_compare_plan.md`
