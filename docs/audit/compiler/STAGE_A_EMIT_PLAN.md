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

| vendor | emitter | structural validator (rung 2.5) | toolchain rung 3 | status |
|---|---|---|---|---|
| **NVIDIA** | `ptx_emit.py` (sm_90a WGMMA PTX) | `validate_ptx_structure` | `ptxas_assemble` (Linux-CI) | landed |
| **Apple** | **`msl_gemm_emit.py`** (simdgroup_matrix GEMM) | `validate_msl_gemm_structure` | `metal_compile` (Metal toolchain) | **landed (this spike)** |
| **AMD** | `rocdl_emit.py` (`amdgpu.wmma`→`rocdl.wmma` text) | `validate_rocdl_structure` | `llc -mcpu=gfx1151` / hipcc | TODO |

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

**Honesty ceiling (in the module docstring + the emitted kernel comment):** a documented
single-output-fragment skeleton, **not** a perf-optimal or boundary-correct kernel — production needs
threadgroup staging, cooperative simdgroup loads, multi-fragment M/N tiling, and ragged-edge masking.
The API is grounded from MLX `steel/gemm/mma.h` + the MSL spec ch.6; **not compile-verified on this
host** (no Metal toolchain). The structural validator is what earns rung 2.5.

## 4. Next steps (in dependency order)

1. **AMD `rocdl_emit.py`** — the third emitter (`amdgpu.wmma`/`rocdl.wmma.f32.16x16x16.bf16` text +
   structural validator), completing the rung-2.5 set across all three vendors.
2. **CUDA pin bump 13.2.1 → 13.3** — fully grounded (driver 610.43.02 / PTX 9.3 / NCCL 2.30.7); a
   coordinated edit across `gpu_target.py` / `TesseraToolchainPins.cmake` / the C++ pin header / the
   byte-identical consistency test.
3. **Apple GEMM completeness** — extend `msl_gemm_emit` from the single-fragment skeleton toward the
   steel structure (multi-fragment tiling + threadgroup staging + edge masking), then wire the
   `metal_compile` rung in a Metal-capable CI lane (rung 3 → real `.air`).
4. **Silicon rungs (6–7)** once the boxes land — launch via the C-ABI bridge (`tsrRegisterGpuLauncher`)
   + execute-and-compare, flipping `backend_kernel` to a real-execution status per vendor.
