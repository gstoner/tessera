---
last_updated: 2026-06-18
audit_role: plan
plan_state: open
---

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
| gfx1250/1251 (v2 ABI) | 16×16×**32** + fp8 16×16×64/128 | f16/bf16 (**native `bfloat`**) + FP8/BF8; **K doubled**; mods/reuse immediate operands. `llc`-grounded (LLVM 22), distinct from RDNA 4. |
| sm_120 (Blackwell consumer) | 16×16×16 (mma.sync) | + **FP8 + FP4** (no tcgen05/TMEM/wgmma — those are datacenter sm_100) |

So NVIDIA consumer goes one precision tier lower (FP4) than AMD consumer (FP8-only); Apple has the
dtypes in the ISA but SDK-gated. **Decision #15a validated** — every vendor's matmul API separates
storage from accumulator, exactly Tessera's `numeric_policy{storage, accum}`.

**Toolchain (the box runs these):** NVIDIA **CUDA 13.3** (driver 610.43.02 / PTX ISA 9.3 / NCCL
2.30.7); AMD **ROCm 7.2.x** (gfx1151 officially supported on the Radeon/Ryzen track); Apple **MSL 4.0 /
macOS 26.5.1** (M1 Max = Apple7). CUDA pin bumped 13.2.1 → 13.3 (landed 2026-06-18).

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

**B1 partial-edge store + B2 double-buffer staging (landed — opt-in flags).** `emit_steel_gemm_msl`
gains `partial_edge=` and `double_buffer=` (default off → the original skeleton, byte-token-compatible):
- **B1** — `M`/`N` not a multiple of 8: full fragments take the direct `simdgroup_store` fast path;
  edge fragments stage their 8×8 to a **threadgroup scratch** (`Cs`) then **cooperatively copy only the
  valid `min(8,M-cr)×min(8,N-cc)` elements** to `C`. The full/edge branch is **threadgroup-uniform**
  (keyed on `tgid` + compile-time loop counters), so the scratch barriers are hit uniformly — never in
  divergent control flow (the one MSL-correctness trap here, avoided by construction).
- **B2** — **ping-pong** staging: two threadgroup slots (`As[2]`/`Bs[2]`), a prologue prefetch of tile
  0, then a steady-state loop that prefetches the next tile into the alternate slot while computing the
  current one — **one barrier per K-step instead of two**.
Both compose, with token validators (`validate_steel_gemm_structure(partial_edge=, double_buffer=)`)
earning rung 2.5.

**B3 Metal-CI rung-3 lane (extended).** `test_rung3_steel_refinements_compile_on_metal_host` adds the
B1/B2 variants to the rung-3 lane — `@pytest.mark.skipif` when `metal` is absent (this host), so they
**verify on a Metal-capable runner**.

**Honesty ceiling:** even with B1+B2 this is a structurally-grounded skeleton (cooperative load still
naive; no async-copy DMA). The Apple rung-3 toolchain (`metal`) is **absent on this host** — so, unlike
the AMD `llc` lane, these are **not compile-verified here**; B3 is the verification. API grounded from
MLX `steel/gemm/mma.h` + MSL spec ch.6.

## 4. Next steps (in dependency order)

1. **CUDA pin bump 13.2.1 → 13.3** — *landed 2026-06-18.* Coordinated edit across `gpu_target.py`
   (toolkit 13.3 / driver 610.43.02 / PTX ISA 9.3; NCCL floor kept at 2.22 — it's a minimum, 13.3
   bundles 2.30.7 but backward-compat keeps the floor in sync with RCCL 2.22),
   `TesseraToolchainPins.cmake`, the C++ pin header (`AdapterVersionPin.h`), `Passes.cpp`, `ptx_emit.py`
   (PTX `.version 9.3`), `backend_manifest.py` (`nvcc_version_min`), `capabilities.py` provenance, the
   `nvidia_cuda13_kernel_inventory.md` doc, and the byte-identical cross-language consistency tests.
   The per-SM `ready/tba` feature readiness was *not* re-evaluated for 13.3 (separate grounded task).
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
   **Threadgroup tiling landed** — `emit_wmma_gemm_threadgroup_llvmir(dtype, mf, nf)` lifts the
   single-wave fragment GEMM to a `BM×BN` (= `16mf × 16nf`) output tile: an `mf×nf` grid of WMMA
   fragments (one `<8 x float>` accumulator PHI + one `wmma` each), A/B staged through **LDS**
   (`addrspace(3)` globals) with a **double** `llvm.amdgcn.s.barrier` (stage-complete + LDS-reuse
   guard) over a BK-deep (16-wide) K-loop, reusing the A1 column-major-A / nt-B addressing and the
   grounded D→C store per fragment (with the `(16i, 16j)` block offset). Verified via `llc
   -mcpu=gfx1151`: the `mf×nf` `v_wmma_*` grid, a real `s_barrier`, and LDS `ds_store`/`ds_load` all
   land. The AMD analog of the Apple steel structure. *Honesty ceiling:* one wave owns the whole grid
   (cooperative multi-wave fragment distribution + coalesced/vectorized LDS loads + double-buffering
   are the perf follow-ons).
   **§7.9.1 WMMA scheduling hazard grounded + locked** — `emit_dependent_wmma_chain_llvmir(dtype,
   hazard=)` + `wmma_scheduling(asm)` pin both behaviors via `llc`: the **in-place accumulation
   pattern** (C/D feedback, independent SrcA/B — what every Tessera WMMA GEMM emits) is **hazard-free**,
   scheduled with `s_delay_alu` and **no** `v_nop`; a **forced** SrcA-reads-prior-WMMA-destination chain
   triggers the mandatory `v_nop` from `llc`'s `GCNHazardRecognizer`. The real threadgroup GEMM is
   asserted hazard-free by construction (`test_rung3_threadgroup_gemm_is_hazard_free`). Key grounding:
   the IR-level emit gets the (rare) mandatory hazard nop from the backend for free — there is nothing
   for the emitter to insert by hand; the deliverable is the grounding + the lock.
   **RDNA 4 (gfx1200/1201) ABI grounded + emitted** — `emit_wmma_rdna4_llvmir(dtype)` +
   `wmma_intrinsic_rdna4` + `validate_wmma_rdna4_structure`, all `llc`-verified on this host across
   f16/bf16/fp8_e4m3/fp8_e5m2. **Decision #27 correction:** the earlier "gfx12 v2 ABI = extra
   format/reuse operands" note was *wrong* — `llc` on this host shows the mods/reuse ABI (`i1
   A_mod`/`i16 C_mod`/reuse flags, `wmma.f32.16x16x32.f16`) is **gfx1250/1251**, a later arch, NOT
   RDNA 4. RDNA 4 keeps the **plain 3-arg `wmma(A,B,C)` ABI** but uses **denser `<8 x elem>`
   fragments** (gfx11 is `<16 x elem>` — RDNA 4 drops the wave32 lane 0-15 → 16-31 duplication) and
   adds native **FP8/BF8** (`fp8_e4m3`→`fp8.fp8`, `fp8_e5m2`→`bf8.bf8` → `v_wmma_f32_16x16x16_{fp8_fp8,
   bf8_bf8}`). Cross-checked: the FP8 intrinsic "Cannot select" on gfx1151, confirming the unlock is
   genuinely RDNA-4-only. 7 new tests (5 rung-3). Honesty ceiling: single-intrinsic ABI proof (the
   gfx11 path's starting point) — RDNA 4's GEMM/operand-layout/threadgroup generalizations are
   follow-ons, and its denser VGPR layout means the D→C mapping is its own grounding job (not a reuse
   of the gfx11 one).

   **gfx1250/1251 v2 ABI landed** — `emit_wmma_gfx1250_llvmir` + `validate_wmma_gfx1250_structure` +
   `wmma_intrinsic_gfx1250`: the **K-doubled 16×16×32** mods/reuse ABI (`wmma(i1 A_mod, A, i1 B_mod, B,
   i16 C_mod, C, i1 a_reuse, i1 b_reuse)`, **native `<16 x bfloat>`**), `llc`-verified on gfx1250/1251
   for f16/bf16 — cross-checked "Cannot select" on gfx1200, confirming it's a distinct arch class. FP8
   (16×16×64/128, the `ModsC` ABI — `(A, B, i16 C_mod, C, reuse, reuse)`, no A/B negate) scoped out as a
   follow-on. Also registered gfx1250/1251 in `rocm_target.py` (AMDArch enum + WMMA variants + wave32 +
   arch strings grounded; LDS/occupancy/non-WMMA features marked **provisional**/`tba` — no gfx1250 ISA
   consulted, per Decision #27). 9 new tests (3 rung-3 + a target-profile guard).

   The **AMD Stage-A emit track (A1–A4) is now complete**: the gfx11/RDNA3-class
   GEMM has the full operand layout + threadgroup tiling + the §7.9.1 hazard locked, and the RDNA 4 ABI
   path is grounded and emitting. *Numerical* correctness across both waits for the AMD Matrix
   Instruction Calculator cross-check or real silicon (rungs 6-7).
3. **Apple GEMM sub-steps** — *partial-edge store (B1) + double-buffered staging (B2) landed* (opt-in
   `emit_steel_gemm_msl(partial_edge=, double_buffer=)`; §3). Remaining: **async-copy DMA** staging
   (`simdgroup_async_copy`, a further perf step), and running the rung-3 Metal-CI lane (B3, now covering
   the refinements) on a **Metal-capable runner** to confirm real `.air` — the Apple emitter's
   verification gap, since `metal` is absent on this dev Mac (unlike AMD's on-host `llc`).
4. **Silicon rungs (6–7)** once the boxes land — launch via the C-ABI bridge (`tsrRegisterGpuLauncher`)
   + execute-and-compare, flipping `backend_kernel` to a real-execution status per vendor.
