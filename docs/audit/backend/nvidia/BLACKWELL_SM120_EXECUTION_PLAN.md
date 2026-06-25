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
| **B. Assemble** | 3 | `ptxas --gpu-name=sm_120a` (or `nvcc -arch=sm_120a`) compiles A to a real cubin. CUDA ≥ 12.8 supports sm_120; Tessera pins 13.3. Skip-clean when toolkit absent (like the existing ptxas rung). | CUDA toolkit on the box |
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

**CUDA Tile op vocabulary → Tessera mapping (grounded from the 13.3 Programming Guide §2.4, the
attached PDF).** The CUDA Tile model is a near-1:1 match for Tessera's own tile abstraction, which
makes the Stage-A lowering largely structural:

| Tessera concept | CUDA Tile C++ (`ct = cuda::tiles`) | notes |
|---|---|---|
| tile (first-class IR) | `ct::tile<T, ct::shape<…>>` | **dims must be powers of two**, shape known at compile time, value semantics |
| `tessera.matmul` (storage=bf16, **accum=fp32**) | `ct::mma(a, b, acc)` / `a @ b` | **mma mixes operand vs accumulator precision** — Tessera's `numeric_policy{storage,accum}` maps *exactly*; 2D **and** 3D-batched supported |
| tile loads / `partition_view` | `ct::tensor_span` + `ct::partition_view{…}.load(idx)` | **lowered to TMA automatically on supported HW** |
| causal / edge masking | `.load_masked()` / `.store_masked()` (OOB→0, `PaddingMode.ZERO`) | the canonical GEMM K-loop zero-pads partial K-tiles, store-side OOB-discard for M/N edges |
| gather/scatter (our data-movers) | tile-of-pointers `ct::load/store`, `ct.gather/scatter` | |
| **softmax / layernorm / attention reductions** | `ct::sum/max/...` reduce; `cumsum` scan (§2.4.9.2) | "softmax denominator, layernorm mean/var, attention max all are reductions"; C++ keeps rank, Python drops axis |
| **transpose data-mover** (Apple-GPU lane) | `ct::transpose(x)` / `ct::permute(x, dimension_map{…})` (§2.4.9.3) | transpose swaps first two axes; permute = arbitrary reorder — "materializing a matmul-operand transpose, attention row/col swap" |
| `where` / `select` | `ct::select(cond,lhs,rhs)` / `ct.where` (§2.4.9.4) | tile conditional |
| pointwise activations (gelu/silu decompose) | `ct::exp/log/sqrt/rsqrt/tanh/sin/cos/pow/...` (§2.4.9.5) | full elementwise math in `ct` namespace |
| cross-block reduce (split-K / MoE) | `ct::atomic_{add,max,min,and,or,xor,xchg,cas}` (§2.4.10) | per-element; `memory_order` + `thread_scope` tags |
| **autotuner knobs** (cf. `flywheel`/`BayesianAutotuner`) | `cutile::hint(arch, kind=val)` / `@ct.kernel(...=ByTarget(sm_90=…))` (§2.4.11) | semantics-preserving hints: `occupancy`/`latency`/`num_cta_in_cga`/`allow_tma`; **`.replace_hints()` returns a re-tuned kernel with its own JIT cache** — "the natural building block for autotuning loops" |
| block id / grid | `ct::bid()`, `ct::num_blocks()` | one logical thread/block; launch `<<<grid, 1>>>` |
| bounded loops | `ct::irange(0, n)` | single control-flow path per block — **no warp-divergence concerns** |
| `__tile_global__` / `__tile__` | tile-kernel entry / tile-callable | coexist with SIMT in one `.cu` |

The guide's canonical `gemm(const __half* A, const __half* B, float* C, …)` with `tm=32,tn=32,tk=16`,
an **fp32 accumulator**, `ct::mma(load_masked, load_masked, acc)` over a `ceil(K/tk)` K-loop, and
`store_masked` — **is the exact shape Tessera's matmul lowering would emit.** The fp32-accum/bf16-operand
pattern validates Decision #15a (storage on the tensor, accumulator in `numeric_policy`).
**CUDA Tile C++ requires CUDA Toolkit 13.3** ("available from 13.3 onward") — another reason to bump
the pin. Both Tile C++ and cuTile Python share one backend = **CUDA Tile IR**.

**Tile IR spec status (grounded from the Tile IR spec TOC + §8 Operations + §10 Stability).** The spec
has 13 sections (Introduction, Programming Model, Syntax, Binary Format, Type System, Semantics,
Memory Model, **Operations**, Debug Info, **Stability**, Optimization Guide, Appendix, Release Notes).
- **Op set (§8)** — grounded for §8.1–8.6: Core (`broadcast`, **`cat`**, `constant`, `extract`,
  `reduce`, `scan`, `reshape`, **`permute`**, `pack`, `unpack`), Control flow (`if`, `loop`, `for`,
  `break`, `continue`, `return`, `yield`, `assert`), Memory (`load_view_tko`/`store_view_tko` [the
  partition-view/TMA loads], `load_ptr_tko`/`store_ptr_tko`, `alloca`, `atomic_rmw_tko`,
  `atomic_cas_tko`), Conversions (`bitcast`, `exti`, `trunci`, `ftof`, `ftoi`, `itof`, ptr casts).
  **Notably, Tessera's four data-movers (cat / transpose-permute / reshape / broadcast) are all
  first-class Tile IR core ops**, and `reduce`/`scan` cover softmax/layernorm — so the data-mover work
  already shipped on Apple GPU maps directly to Tile IR ops on NVIDIA. *Still missing:* the matmul/MMA
  op mnemonic (in §8.7+, past the fetch truncation).
- **Stability (§10)** — Tile IR is **released and versioned** (tracks CUDA 13.1/13.2/**13.3**), with a
  forward-compat **bytecode** guarantee ("a program conforming to vX.Y is portable to vX.Y or newer"),
  respecting CUDA minor-version compat. **Caveat, quoted:** "*During the 13.x release cycle, we are
  bringing up existing hardware targets which may introduce new features on old targets. This 'cold
  start' period is an exception.*" → production-intent, not "preview", but actively maturing in 13.x.
- **SASS path** — the optimizing Tile IR compiler ships **in the CUDA driver and as a standalone
  tool**; the driver loads/interprets the bytecode ("interpreted and loaded by all conforming
  drivers"). So: emit Tile IR bytecode → driver/standalone compiler → SASS (no PTX in the middle).
- **MLIR dialect** — mentioned in the Introduction as a *producer surface* ("an MLIR dialect existing
  compilers can use to target Tile IR as a backend"), but it is **not a TOC section** of the spec (the
  spec documents the IR itself — syntax/bytecode/operations). So the dialect is a front-end onto Tile
  IR, separate from the IR spec; its exact op/dialect name still needs the MLIR/headers, not this spec.

*Remaining unknowns:* the MMA op mnemonic (§8.7+) and the MLIR dialect's concrete name/op set.

## Toolkit: the box runs CUDA 13.3 (pin bumped 13.2.1 → 13.3 — LANDED 2026-06-18)

The target system loads **CUDA 13.3** (release notes 27-May-2026). Tessera's pin **has been bumped
13.2.1 → 13.3** (this section's table). 13.3 is materially better for this work:
- **CUDA Tile C++** — tile programming in CUDA C++ with **NVCC *and* NVRTC**. This is the C++ sibling
  of the Tile IR / cuTile path above; the **NVRTC** half is a *runtime* tile-compile lane that drops
  straight onto Tessera's `tsrRegisterGpuLauncher` bridge (emit tile program → NVRTC → load → launch).
- **Full Blackwell compiler support** (SM_100/101/120); **libNVVM** Blackwell codegen on an
  **LLVM 18.1.8**-based NVVM IR dialect (relevant to a future MLIR→NVVM path).
- Perf: ~5% FP4 GEMM (Blackwell Ultra), ~27% TF32 GEMM (Blackwell/Ultra). CompileIQ (AI compiler
  autotuning — compare Tessera's flywheel/autotuner). CUDA Python 1.0.
- **Tile coverage is sm_80 and later** (CUDA Toolkit 13.3 release notes, "CUDA TILE Supported
  Architectures") — the Tile C++/Tile IR path is **not** Blackwell-only; it covers Ampere→Blackwell
  (sm_80/90/100/103/120/121). So one tile-lowering path serves *every* Tessera NVIDIA target, not just
  sm_120 — strengthening "lower to Tile IR" over per-arch PTX.
- **Pin values — landed:**

  | pin | was (13.2.1) | **now (13.3)** | source |
  |---|---|---|---|
  | CUDA toolkit | 13.2.1 | **13.3** | release notes |
  | min Linux driver | 555.85 | **610.43.02** | release notes Table 3 |
  | PTX ISA | 8.6 | **9.3** | release notes, CUDA Compiler features |
  | NCCL | 2.22 | **2.22** (floor kept; 13.3 bundles 2.30.7) | NCCL `2.30.7-1+cuda13.3` is the bundle, but the *minimum* floor stays 2.22 (backward-compatible; in sync with RCCL 2.22) |

  Tile C++ is presented as a shipped feature (nvcc + NVRTC), no "preview" label.
- **Done (2026-06-18):** pin bumped **13.2.1 → 13.3** across `gpu_target.py`
  (`TESSERA_TARGET_CUDA_TOOLKIT` / `_CUDA_DRIVER_MIN` / `_PTX_ISA`; `_NCCL_MIN` floor unchanged),
  `cmake/TesseraToolchainPins.cmake`, the C++ pin header (`AdapterVersionPin.h`), `Passes.cpp`,
  `ptx_emit.py` (PTX `.version 9.3`), `backend_manifest.py` (`nvcc_version_min`), `capabilities.py`
  provenance, `nvidia_cuda13_kernel_inventory.md`, and the byte-identical cross-language consistency
  tests. The per-SM `ready/tba` feature readiness was *not* re-evaluated for 13.3 (separate grounded
  task). Silicon execution (rungs 6-7) still gated on the sm_120 box.

### CUDA 13.3 toolchain → Tessera component map (grounded from NVIDIA docs, 2026-06-17)

| CUDA 13.3 surface | What it is | Tessera role |
|---|---|---|
| **CUDA Tile C++** ([api ref](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/index.html)) | Tile programming in C++; elementwise ops on tiles; compiler "leverages TMA and Tensor Cores" automatically (`cuda::tiles`, `ct::partition_view`, `load_masked`/`store_masked`) | **Stage-A frontend** — the concrete way to lower Tessera tiles to Tile IR without hand-rolling per-arch PTX. Confirms the tensor-core auto-abstraction. |
| **`cuda.tile`** (Python DSL) | Same tile model from Python | Python authoring / prototyping of the tile path |
| **CUDA Python** ([cuda-python](https://nvidia.github.io/cuda-python/latest/), 13.3.1) | Official ctypes-free bindings: `cuda.bindings` (CUDA C ABI), `cuda.core` (runtime) | **Replaces the hand-rolled ctypes CUDA wrapper** in `runtime.py` for the `tsrRegisterGpuLauncher` launch bridge (Stage C) |
| **CompileIQ** ([repo](https://github.com/nvidia/compileiq), OSS) | Measured ptxas/nvcc Advanced-Control autotuning → ACF; `PtxasSearchSpace(version="13.3")` | Compiler-flag layer beneath Tessera's kernel-config autotuner; a direct parallel + reference for `flywheel.py` / the Evaluator |

## Honest external gates

- The "RTX 5070 Ti not supported" noise is about **framework wheels** (PyTorch/TF prebuilt binaries
  lagging sm_120), **not** the CUDA toolkit — `nvcc`/`ptxas` 13.x assemble `sm_120a` cleanly, which is
  all Tessera needs.
- **Shared-memory discrepancy — RESOLVED on-silicon (2026-06-25, RTX 5070 Ti, driver 610.62 / CUDA
  UMD 13.3, nvcc 13.3.33, `-arch=sm_120`).** The 100 KB figure wins; the release-note 128 KB is the
  *unified data cache* (Table 32), not the shared-memory carve-out. Measured via
  `cudaDeviceGetAttribute` / `cudaGetDeviceProperties` on the box:
  - `sharedMemPerMultiprocessor` = **102400 B (100 KiB)** — exact match for `_SMEM_BYTES[SM_120] = 102400`.
  - `cudaDevAttrMaxSharedMemoryPerBlockOptin` = **101376 B (99 KiB)** — matches Table 31's "99 KB/block".
  - `cudaDevAttrMaxSharedMemoryPerBlock` (static, no opt-in) = **49152 B (48 KiB)**; SM count = 70.
  Conclusion: `gpu_target.py`'s 100 KB/SM pin is correct. **Lowering nuance:** a single block can opt
  into at most **99 KiB (101376 B)**, not the full 100 KiB/SM — any per-block dynamic-smem budget
  (and the `cudaFuncAttributeMaxDynamicSharedMemorySize` set-attr) must cap at 101376 on sm_120, not
  102400. (Same per-SM-vs-per-block-optin 1 KiB reservation pattern exists on Hopper; `_SMEM_BYTES`
  records the per-SM capacity by convention.)
- **`sm_120a` vs `sm_120f` — grounded from the 13.3 Programming Guide §5.1.2 + Table 28.** There are
  three compiler-target tiers: **baseline** (`compute_120`, no arch-specific features, runs CC 12.0+);
  **family-specific** (`compute_120f`, the arch-specific subset *common to the consumer-Blackwell
  family* — Table 28: **runs on CC 12.0 AND 12.1**); **architecture-specific** (`compute_120a`, the
  *full* arch-specific set incl. the FP4 `mma.sync.block_scale` Tensor-Core path — **runs ONLY on
  exactly CC 12.0**). So: use **`sm_120a`** for the FP4/block-scale matmul proof (needs the full
  arch-specific set); use **`sm_120f`** if/when Tessera wants one binary portable across the consumer
  family (sm_120 + sm_121). `gpu_target.py` keeps `sm_120a` as the default (correct for the FP4 path);
  adding an `sm_120f` family target is a clean follow-on for portable artifacts.
- **Do NOT reuse the sm_90a WGMMA artifact as the execution proof here** — it won't run on sm_120.
  The sm_90a WGMMA path stays valid for an actual Hopper box; it is not this card.
- **Intel 265F host:** no AVX-512/AMX — drive CUDA from it, but build Tessera's x86 backend on the
  **Zen 5 (Strix Halo) box**, never with `-mavx512*` here (would SIGILL).

## Sequencing when the box lands

1. **Toolkit/driver:** install CUDA ≥ 12.8 (13.3 preferred, ≥610.43.02 driver) + R570+; confirm `nvidia-smi` and
   `nvcc --list-gpu-arch | grep sm_120`.
2. **Stage A spike — PROVEN on-silicon (2026-06-25, RTX 5070 Ti / CC 12.0, CUDA 13.3, nvcc 13.3.33).**
   A hand-emitted, Tessera-style sm_120 bf16 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
   single-tile GEMM (D[16x8] = A[16x16]·B[16x8], f32 accumulate; new path, NOT the sm_90a wgmma
   emitter) cleared the full rung ladder end-to-end:
   - **Rung 2.5 (emit):** raw PTX `.version 9.3` / `.target sm_120a` — byte-matches the pin; nvcc emits
     the identical mnemonic for the inline-asm oracle.
   - **Rung 3 (assemble):** `ptxas --gpu-name=sm_120a` → 6168-byte cubin, clean.
   - **Rung 5 (execute-and-compare):** loaded via Driver API `cuModuleLoadDataEx` + `cuLaunchKernel`,
     output matches a bf16-rounded CPU reference to **max abs err 4.8e-7 (f32 epsilon), 0/128 off**.
   Spike artifacts (kernel + driver harness + smem device-query + reproduce steps) are committed at
   `spikes/sm120_mma_sync/`.
   Gotchas found: (a) PTX comments must be ASCII — the driver JIT ptxas rejects non-ASCII (em-dash)
   that standalone ptxas tolerates; (b) the f32 accumulator regs must be explicitly zero-initialized
   before the `mma` (they are the C operand); (c) CUDA 13.3 `cuCtxCreate` is v4 — use
   `cuDevicePrimaryCtxRetain` in host harnesses.
3. **Stages B→D (productization) — LANDED 2026-06-25.** The full chain is in-tree and green on the box:
   - `ptx_emit.emit_mma_sync_matmul_ptx` (+ validators) emits the complete sm_120 mma.sync kernel;
     `test_ptx_emit.py` includes a ptxas-gated rung-3 assemble.
   - C-ABI launch bridge: `test_conformance_execute_compare_nvidia.py` registers a CUDA launcher via
     `tsrRegisterGpuLauncher` that loads the **emitted PTX** through the Driver API and runs it via
     `tsrLaunchKernel` (execute-and-compare vs CPU ref).
   - **Shipped runtime symbol:** `libtessera_nvidia_gemm.so` (CMake `tessera_nvidia_gemm`) exports
     `tessera_nvidia_mma_gemm_{bf16,f16,tf32,e4m3,e5m2}` — a general tiled/K-looped mma.sync GEMM
     NVRTC-compiled for the device arch. `test_nvidia_mma_runtime_symbol.py` dlopens it and numerically
     validates all 5 dtypes vs numpy/ml_dtypes.
   - **Manifest:** `nvidia_sm120` matmul flipped `artifact_only → hardware_verified` (runtime_symbol +
     execute_compare_fixture) — the **first NVIDIA `backend_kernel` hardware-verified row** (mirrors the
     ROCm Strix Halo pattern). sm_80/90/100 stay artifact_only (proven only on sm_120).
   - Build fix: `src/runtime/CMakeLists.txt` + the NVIDIA backend wire CUDA includes/links so
     `TESSERA_ENABLE_CUDA=ON` builds (was enabled-but-unbuildable).

   **@jit execution lane — LANDED 2026-06-25.** `@jit(target="nvidia_sm120")` matmul now dispatches
   through the shipped symbol on a capable host: `execution_matrix` has an executable
   `("nvidia_sm120","nvidia_mma")` row (sm_120 removed from `_UNIMPLEMENTED_TARGETS`); `runtime.py`
   adds the `nvidia_mma` executor (loads libtessera_nvidia_gemm.so, picks the dtype symbol, runs +
   returns); `jit.py` stamps `executable=True, compiler_path="nvidia_mma", native_gpu` when the runtime
   probe passes (off-device it stays artifact_only — no behavior change). f16/bf16/fp32(tf32-math)
   storage. `test_nvidia_launch_execute.py` covers the matrix row + execute-and-compare + the @jit
   default. This mirrors ROCm's `rocm_wmma` lane; a compiler-GENERATED nvidia lane (the `rocm_compiled`
   analog, via a tessera-opt NVIDIA pipeline) remains a follow-up.

   **Still open:** the compiler-generated nvidia lane (above); NVFP4 block-scale (#9; the warp
   instruction is already confirmed to assemble+execute on sm_120a — see `spikes/sm120_mma_sync/` —
   pending the PTX ISA scale-distribution spec for numerics).

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
