# Cross-Vendor Grounding Summary + Stage-A Emit Plan

> Consolidates the 2026-06-17 spec-grounding pass (CUDA 13.3 Programming Guide + Tile IR/Tile C++ +
> RDNA 3.5 & RDNA 4 ISAs + Metal Feature Set Tables + MSL Spec + MPP guide + MLX source) and lays out
> the **host-free Stage-A (rung 2.5) emit** work. Per-vendor bring-up plans:
> `../backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md`, `../backend/rocm/STRIX_HALO_EXECUTION_PLAN.md`,
> `../backend/apple/APPLE_AUDIT.md`. Rung ladder: `EVALUATOR_PLAN.md` §2.

## 1. What we learned — the cross-vendor convergence

All three vendors expose a **tile-fragment matrix-multiply** primitive and an MLIR/IR lowering target
at exactly Tessera's abstraction level (tiles as first-class IR). Tessera's thesis is what the
ecosystems independently converged on.

| | tile-MMA primitive | IR / emit target | accumulator |
|---|---|---|---|
| **NVIDIA** | `ct::mma(a,b,acc)` (CUDA Tile C++) | **CUDA Tile IR** (bytecode → driver → SASS) / NVVM / PTX `wgmma`·`mma.sync` | separate (mixes operand vs acc precision) |
| **AMD** | `V_WMMA_*` / SWMMAC | **`amdgpu.wmma` → `rocdl.wmma` → LLVM AMDGPU → GCN** | F32 acc |
| **Apple** | `simdgroup_matrix` + `simdgroup_multiply_accumulate` | custom MSL (MLX "steel" is the production proof; no public LLVM→AIR) | F32 acc |

**Per-arch low-precision matrix tiers (all ISA-grounded):**

| arch | tile | low-precision matrix dtypes |
|---|---|---|
| Apple7 (M1 Max) | 8×8 simdgroup_matrix | f16/bf16; FP8/FP4/MX = **macOS-27.0 toolchain-gated**, not hardware |
| gfx1151 (RDNA 3.5) | 16×16×16 | f16/bf16/int8/int4 — **no FP8** |
| gfx1200 (RDNA 4) | 16×16×16 + 16×16×32 | + **FP8/BF8** + SWMMAC sparse — **no FP4** |
| sm_120 (Blackwell consumer) | 16×16×16 (mma.sync) | + **FP8 + FP4** (no tcgen05/TMEM/wgmma — those are datacenter sm_100) |

So NVIDIA consumer goes one precision tier lower (FP4) than AMD consumer (FP8-only); Apple has the
dtypes in the ISA but SDK-gated. **Decision #15a validated** — every vendor's matmul API separates
storage from accumulator, exactly Tessera's `numeric_policy{storage, accum}`.

**Toolchain (the box runs these):** NVIDIA **CUDA 13.3** (driver 610.43.02 / PTX ISA 9.3 / NCCL
2.30.7); AMD **ROCm 7.2.x** (gfx1151 officially supported on the Radeon/Ryzen track); Apple **MSL 4.0 /
macOS 26.5.1** (M1 Max = Apple7). CUDA pin bump 13.2.1 → 13.3 is fully grounded and unblocked.

**Process learning (this whole arc):** ground every hardware/ISA/API claim against the authoritative
spec, not marketing or memory (Decision #27). It repeatedly mattered — caught the sm_120-as-Rubin
mislabel, the FP6/INT4 dtype gaps, the gfx1151 no-FP8 gate, the FP8/FP4 asymmetry, and the
toolchain-vs-hardware distinction for Apple low precision.

## 2. The Stage-A emit ladder (host-free → silicon)

"Stage A" = **rung 2.5**: emit the documented instruction/kernel text + a **host-free structural
validator** (no GPU, no toolchain — runs in CI on the arm64 dev Mac). Then rung 3 = real toolchain
assemble (skip-clean when absent), rungs 6–7 = launch + execute-and-compare on owned silicon.

| vendor | emitter | structural validator (rung 2.5) | toolchain rung 3 | rung-3 here? | status |
|---|---|---|---|---|---|
| **NVIDIA** | `ptx_emit.py` (sm_90a WGMMA PTX) | `validate_ptx_structure` | `ptxas_assemble` | skip (no ptxas) | landed |
| **Apple** | `msl_gemm_emit.py` (simdgroup_matrix + steel) | `validate_{msl,steel}_gemm_structure` | `metal_compile` | skip (no metal) | landed |
| **AMD** | `rocdl_emit.py` (`llvm.amdgcn.wmma` LLVM-IR) | `validate_wmma_llvmir_structure` | `llc -mcpu=gfx1151` | **RUNS** (Homebrew LLVM 22) | **landed** |

**AMD is special: its rung 3 actually runs on the dev Mac.** Homebrew LLVM 22 carries the AMDGPU
backend, so `llc -mcpu=gfx1151` lowers the emitted `llvm.amdgcn.wmma.*` to a real `v_wmma_*` *here* —
3 of the AMD tests are genuine rung-3 (not skip-clean). This already paid off: the host-free validator
passed bf16 emitting `<16 x bfloat>`, but **rung-3 `llc` rejected it** — the RDNA bf16 wmma intrinsic
takes `<16 x i16>` (bf16 bit-patterns). The rung ladder caught a real ABI error the structural check
could not. (Scoped to RDNA3-class gfx1100/gfx1151; gfx1200/RDNA 4 uses a different gfx12 v2 wmma
intrinsic ABI — a follow-on.)

## 3. This spike — Apple `simdgroup_matrix` GEMM lane (landed)

`python/tessera/compiler/msl_gemm_emit.py` (+ `tests/unit/test_msl_gemm_emit.py`, 16 tests). The Apple
analog of `ptx_emit.py`, motivated by the MLX "steel" finding (the production framework uses custom
`simdgroup_matrix` MSL over MPS for the fusion control a compiler needs — the "clear-MPS" direction):

- `emit_simdgroup_gemm_msl(dtype, m, n, k, accum="f32")` — emits the documented MSL
  `simdgroup_matrix<{T},8,8>` GEMM: zero-filled fp32 accumulator → K-loop of
  `simdgroup_load` ×2 → `simdgroup_multiply_accumulate(acc, a, b, acc)` → `simdgroup_store`. f16/bf16/f32
  inputs, fp32 accumulator (the `numeric_policy{storage,accum}` + MLX-steel pattern). Tile dims must be
  multiples of the 8×8 fragment.
- `validate_msl_gemm_structure(...)` — host-free rung 2.5: checks the scaffolding + per-dtype fragment
  types + the load→mma→store sequence. Catches corrupted/missing-token MSL and dtype mismatch.
- `min_metal_std(dtype)` — bf16 → `metal3.1` (bfloat is an MSL 3.1 type), else `metal3.0`.
- `metal_compile(...)` — rung 3; **skip-cleans** when the offline `metal` compiler is absent (the case
  on this CommandLineTools-only arm64 Mac, exactly like `ptxas`).

**Steel-structured emit (landed — the production shape):** `emit_steel_gemm_msl(dtype, bm, bn, bk)` +
`validate_steel_gemm_structure` grow the lane from the single-fragment skeleton to the MLX-steel shape:
a threadgroup computes a `BM×BN` tile (an `MF×NF` grid of 8×8 output fragments) over `BK`-deep steps,
each step doing a **cooperative bounds-guarded load** of A/B into **threadgroup memory** (zero-padded
at edges = ragged-edge masking on the load side) → `threadgroup_barrier` → fragment `simdgroup_load`
from threadgroup memory → the `MF×NF` fragment `simdgroup_multiply_accumulate` inner product →
whole-fragment guarded store. The structural validator asserts all of these.

**Metal-CI rung-3 lane (landed):** `tests/unit/test_msl_gemm_emit.py` adds rung-3 tests
(`test_rung3_*_compiles_on_metal_host`) that invoke the offline `metal` compiler on **both** emitters'
output and assert it compiles to AIR — `@pytest.mark.skipif` when `metal` is absent (the arm64 dev Mac
/ Linux), so they run only on a Metal-capable runner. 23 host-free tests pass here; 6 rung-3 tests skip.

**Honesty ceiling (still in the docstrings + emitted kernel comments):** even the steel form is a
documented skeleton — single-buffered staging (no double-buffer / async copy), a naive cooperative
load, and **whole-fragment** store guards; *partial* output-fragment edge handling (M/N not a multiple
of 8) is the named next sub-step. The API is grounded from MLX `steel/gemm/mma.h` + MSL spec ch.6;
**not compile-verified on this host** (no Metal toolchain — that is exactly what the rung-3 CI lane is
for). The structural validators are what earn rung 2.5.

## 4. Next steps (in dependency order)

1. **CUDA pin bump 13.2.1 → 13.3** — fully grounded (driver 610.43.02 / PTX 9.3 / NCCL 2.30.7); a
   coordinated edit across `gpu_target.py` / `TesseraToolchainPins.cmake` / the C++ pin header / the
   byte-identical consistency test.
2. **AMD: grow `rocdl_emit` toward a real GEMM** — *in progress, four steps landed; the operand
   layout is now complete.* `emit_wmma_gemm_llvmir` added the **K-reduction GEMM tile** (`<8 x float>`
   accumulator PHI + global A/B loads + `wmma` in the loop); `emit_wmma_gemm_layout_llvmir` added the
   **ISA §7.9 lane replication** (lanes 0-15 → 16-31 via `lane & 15`) + nt contiguous loads; and
   `emit_wmma_gemm_store_llvmir` is now the **complete-layout** emit, carrying all three grounded
   operand rules:
   - **Column-major A** — `a_frag[ele] = a[16*lane + ele]`: lane selects the K-column, the contiguous
     16-run walks the A-tile rows. Generalized over K-tiles: column `= k0 + lane`, base `= col*16`
     (leading dim = the 16 A-tile rows). *Corrects* the earlier row-major A load.
   - **nt B** — B supplied pre-transposed (N×K) so the per-lane load is contiguous over K (the perf
     form of the blog's row-major `b[16*ele + lane]` strided gather; same operand values).
   - **D→C store** — wave32 fp32 lane `L` register `ele` (0-7) → `D[2*ele + L/16][L%16]`, 8 strided
     scalar stores (`col = lane & 15`, `row_base = lane >> 4`).
   All **grounded from the GPUOpen RDNA3 WMMA blog** — the RDNA 3.5 ISA §7.9 references the blog + the
   AMD Matrix Instruction Calculator rather than tabulating (the tabulated "Matrix Element Storage in
   VGPRs" is an RDNA *4* §7.12.2 addition). Verified via `llc -mcpu=gfx1151` — the AMDGCN carries a
   real `v_wmma_*`, `v_and_b32 _,15,_` (replication), the column-major A address math, and strided
   `global_store`s (the D mapping).
   Remaining: **threadgroup tiling** (BM×BN tile over a BK K-loop — the AMD analog of the Apple steel
   structure), the **§7.9.1 V_NOP scheduling hazard** (`s_nop`/`v_nop` between dependent WMMAs), and
   the **gfx12 v2 wmma intrinsic ABI** for gfx1200/RDNA 4 (gfx11 intrinsics "cannot select" on gfx12;
   unlocks FP8/BF8 + SWMMAC sparse + int4 16×16×32). The rung-3 `llc` lane runs here, so each step is
   immediately AMDGCN-verifiable on the dev Mac; *numerical* correctness waits for the AMD Matrix
   Instruction Calculator cross-check or real silicon (rungs 6-7).
3. **Apple GEMM remaining sub-steps** — partial-edge store handling + double-buffered staging /
   async copy (perf), then run the rung-3 lane on a Metal-capable runner to confirm real `.air`.
4. **Silicon rungs (6–7)** once the boxes land — launch via the C-ABI bridge (`tsrRegisterGpuLauncher`)
   + execute-and-compare, flipping `backend_kernel` to a real-execution status per vendor.
