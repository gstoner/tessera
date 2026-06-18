# RTX 5070 Ti (Blackwell sm_120) — Tessera NVIDIA Execution Plan

> Second real box (with the Strix Halo machine — `../rocm/STRIX_HALO_EXECUTION_PLAN.md`).
> Authored 2026-06-17. **This is the Linux/CUDA runner with real silicon that the NVIDIA
> Evaluator track has been gated on** — today NVIDIA sits at **rung 2.5** (WGMMA PTX *emitted* +
> structurally validated by `compiler/ptx_emit.py`; rung-3 `ptxas` and rung-7 execute-and-compare
> were blocked because the arm64 dev Mac can't run ptxas/CUDA). NVIDIA is the **primary target** and
> the path the whole MLIR→LLVM→PTX strategy points at; this box makes it executable.

## The box

| Part | What | Tessera role |
|------|------|--------------|
| **NVIDIA RTX 5070 Ti 16 GB** | **Blackwell consumer, GB203, sm_120** (CC 12.0); 8,960 CUDA cores; 16 GB GDDR7, 256-bit, 28 Gbps → **896 GB/s**; ~1,406 AI TOPS | The NVIDIA execution target |
| **Intel Core Ultra 7 265F** | Arrow Lake-S, 20c; **AVX-512 disabled, no AMX** | CUDA **host only** — *not* an x86-backend target |
| 32 GB DDR5-6000 | | |

**Memory/roofline:** 896 GB/s — **~3.5× the Strix Halo's 256 GB/s**, so this box is *not*
bandwidth-starved the way the APU is; it's a healthy roofline target (but 16 GB caps model size,
where Strix Halo's 128 GB unified is the complement). `flywheel.py` per-chip calibration should get
an sm_120 entry.

### Grounded hardware facts (CUDA Programming Guide, compute-capabilities appendix, CC 12.0)

Authoritative source (per the user, 2026-06-17): the
[compute-capabilities appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
is the place to confirm what CC 12.0 supports — not marketing or forum posts. From it:
- **Tensor Core input types (Table 33), all "Yes" for 12.x:** FP64, TF32, BF16, FP16, FP8, **FP6**,
  **FP4**, INT8, **INT4**. (Marketing lists FP4/FP8/INT8/INT4/FP16 but *omits FP6* — the table
  confirms FP6 is in fact supported. `gpu_target.py`'s sm_120 dtype set now matches, incl. int4.)
- **Shared memory (Table 31): 100 KB/SM, 99 KB/block** (>48 KB needs dynamic opt-in). `gpu_target.py`
  `_SMEM_BYTES[SM_120]` set to 102400 accordingly.
- **vs datacenter sm_100 (Table 32):** same Tensor Core *input dtypes*, but CC 12.x has a **smaller
  unified data cache (128 KB vs 256 KB)** and fewer SMEM config options (5 vs 9). The deeper
  difference is the *instruction path* (sm_120 `mma.sync` vs sm_100 `tcgen05`/TMEM), which the
  programming-guide dtype table does not show — see the correction below.

## ⚠️ The load-bearing correction: sm_120 ≠ Hopper, ≠ datacenter Blackwell

The target model previously mislabelled sm_120 as a "Rubin placeholder / superset of Blackwell."
**Corrected (2026-06-17, grounded):** sm_120 is **Blackwell consumer (RTX 50-series, GB20x)** and is
**not** a superset of datacenter sm_100. Concretely, consumer sm_120:

- **Has no `wgmma`** (that is Hopper sm_90a-only). → **The existing `ptx_emit.py` emits sm_90a WGMMA
  PTX, which will NOT run on this card.** The first executable kernel on *this box* needs a new
  **sm_120 `mma.sync` emit path**, not the Hopper WGMMA we have today.
- **Has no `tcgen05` / `TMEM`** (datacenter sm_100a-only — NVIDIA/cutlass#2800, modular#5707).
- **Does have FP4 / block-scaled MMA** via warp-level `mma.sync.aligned…block_scale` (E2M1 + block
  scaling) — the consumer Blackwell headline (NVFP4). Compile target **`sm_120a`**.

`gpu_target.py` / `capabilities.py` now reflect this (wgmma/tcgen05/tmem → `not_supported` on sm_120,
block_scaled_mma → `ready`; shared mem → ~100 KB consumer class). Guards in
`tests/unit/test_target_toolchain_pins.py::test_sm120_is_blackwell_consumer_not_datacenter_superset`.

## Execution roadmap — rung ladder to first real sm_120 GEMM

Mirrors the NVIDIA PTX/Evaluator track (`compiler/ptx_emit.py`, `EVALUATOR_PLAN.md` §9.5). The key
difference from the existing rung-2.5 work: that targets **sm_90a WGMMA** (Hopper); this box is
**sm_120 mma.sync** (consumer Blackwell). Both are valuable, but only the latter executes here.

| Stage | Rung | What | Gate |
|-------|------|------|------|
| **A. Emit (sm_120)** | 2.5 | New emit path: bf16/fp16 `mma.sync.aligned.m16n8k16` PTX for `sm_120a` (parallel to the existing sm_90a `wgmma` emitter). Host-free structural validator. Then the FP4 headline: `mma.sync…block_scale` (m16n8k64, E2M1). | none — **can start now** |
| **B. Assemble** | 3 | `ptxas --gpu-name=sm_120a` (or `nvcc -arch=sm_120a`) compiles A to a real cubin. CUDA ≥ 12.8 supports sm_120; Tessera pins 13.2 U1. Skip-clean when toolkit absent (like the existing ptxas rung). | CUDA toolkit on the box |
| **C. Launch** | 6 | Register a CUDA launcher into the C-ABI bridge `tsrRegisterGpuLauncher` (landed G7, `../BACKEND_AUDIT.md`); HIPRTC-equivalent is NVRTC + `cuModuleLoadData` / `cuLaunchKernel`. | the box + R570+ driver |
| **D. Prove** | 7 | Execute-and-compare the sm_120 bf16 `mma.sync` GEMM vs numpy (Evaluator vertical oracle); flip `backend_kernel` for `tessera.matmul` on `nvidia_sm120` to a real-execution status. **First real NVIDIA `backend_kernel` proof.** Then NVFP4 via block_scale. | the box |

## ⭐ Preferred Stage-A path: lower to NVIDIA Tile IR (not hand-rolled PTX)

NVIDIA shipped **CUDA Tile IR** (https://docs.nvidia.com/cuda/tile-ir/latest/) — "a portable,
low-level tile virtual machine and instruction set that models the GPU as a tile-based processor",
i.e. **the tile-level analog of PTX**. Critically for us, it ships **an MLIR dialect "that existing
compilers can use to target Tile IR as a backend compiler target"** (plus textual syntax + portable
bytecode), with an optimizing Tile IR compiler "available as part of the CUDA driver and as a
standalone tool". OpenAI Triton already has a CUDA Tile IR backend — the integration precedent.

**Why this is the better Stage A:** Tile IR's explicit goal is to "abstract tensor-cores and their
programming model". That makes **the per-arch instruction selection NVIDIA's job, not Tessera's** —
exactly the gap this plan flags (sm_120 needs `mma.sync`, not the sm_90a `wgmma` we hand-emit today;
sm_100 needs `tcgen05`). Instead of maintaining three hand-rolled PTX emitters (wgmma / tcgen05 /
mma.sync), Tessera lowers its **own Tile IR → NVIDIA Tile IR (via the MLIR dialect)** and one program
runs across sm_90 / sm_100 / sm_120. This is the "MLIR/LLVM foundation for NVIDIA" realized at exactly
Tessera's abstraction level — and it validates Tessera's tile-centric thesis (NVIDIA now agrees the
GPU is a tile processor).

**So the revised Stage A is a fork:**
- **(A-tileir, preferred):** Tessera Tile IR → NVIDIA Tile IR MLIR dialect → driver/standalone Tile IR
  compiler → SASS. Portable across arches; no per-arch PTX. **Spike this first** once the spec
  details below are confirmed.
- **(A-ptx, fallback):** hand-emit `sm_120a` `mma.sync` PTX (the row above). Keep as a control /
  oracle and for environments without the Tile IR compiler.

**Verify against the full Tile IR spec (sections 1-8) before committing:** CUDA toolkit version +
GA-vs-preview status; whether the standalone Tile IR compiler ships in the pinned CUDA 13.2 U1; the
MLIR dialect name/op set; the SASS-vs-PTX consumption path; concrete sm_90/100/120 portability
guarantees. (Grounded so far from the spec landing + Introduction only.)

## Honest external gates

- **CUDA toolkit ≥ 12.8 is required for sm_120** (Tessera pins 13.2 U1 — fine). The widely-reported
  "RTX 5070 Ti not supported" noise is about **framework wheels** (PyTorch/TF prebuilt binaries
  lagging sm_120), **not** the CUDA toolkit — `nvcc`/`ptxas` 12.8+ assemble `sm_120a` cleanly, which
  is all Tessera needs. Driver **R570+**.
- **sm_120 uses `sm_120a` (arch-specific) for FP4 block-scale**; a `compute_120f` family variant also
  exists (CUDA 13.0+). Use `sm_120a` for the block-scaled mma path.
- **Do NOT reuse the sm_90a WGMMA artifact as the execution proof here** — it won't run on sm_120.
  The sm_90a WGMMA path stays valid for an actual Hopper box; it is not this card.
- **Intel 265F host:** no AVX-512/AMX — drive CUDA from it, but build Tessera's x86 backend on the
  **Zen 5 (Strix Halo) box**, never with `-mavx512*` here (would SIGILL).

## Sequencing when the box lands

1. **Toolkit/driver:** install CUDA ≥ 12.8 (13.2 U1 preferred) + R570+; confirm `nvidia-smi` and
   `nvcc --list-gpu-arch | grep sm_120`.
2. **Stage A spike (host-free, can begin now):** emit + structurally validate an sm_120 bf16
   `mma.sync` GEMM PTX (new path; do not reuse the sm_90a wgmma emitter).
3. **Stages B→D:** ptxas-assemble `sm_120a` → NVRTC/cuLaunch via `tsrRegisterGpuLauncher` →
   execute-and-compare → first real NVIDIA `backend_kernel` proof. Then NVFP4 block-scale.

## The two-box frontier

With this box + Strix Halo, Tessera has real silicon for **NVIDIA (sm_120) + AMD (gfx1151) GPUs +
AVX-512 CPU (Zen 5)** — Apple is already proven on the dev Mac. The entire hardware-gated
`backend_kernel` frontier (today **0 entries complete**, `../BACKEND_AUDIT.md`) is now openable
across all vendors.

## Cross-refs

- `../BACKEND_AUDIT.md` — the hardware-gated frontier + `tsrRegisterGpuLauncher`.
- `NVIDIA_AUDIT.md` — NVIDIA theme audit (Still Open / runtime_execution_matrix items this unblocks).
- `../rocm/STRIX_HALO_EXECUTION_PLAN.md` — the parallel AMD plan.
- `python/tessera/compiler/gpu_target.py`, `capabilities.py` — the corrected sm_120 (Blackwell consumer) model.
- `compiler/ptx_emit.py`, `EVALUATOR_PLAN.md` §9.5 — the rung ladder this mirrors (note: existing emitter is sm_90a wgmma).
- `docs/nvidia_cuda13_kernel_inventory.md` — NVIDIA kernel inventory (an sm_120 mma.sync inventory is a sibling TODO).
