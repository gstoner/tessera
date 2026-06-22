# Strix Halo (Ryzen AI Max+ 395) — Tessera Bring-Up & RDNA 3.5 Execution Plan

> First real **non-Apple** silicon Tessera will own. Authored 2026-06-17 against the
> "RDNA3.5" Instruction Set Architecture Reference Guide (AMD, 23-July-2024).
> Strategic context: this box is the **unblock for the ROCm `backend_kernel` axis**
> (today **0 entries `backend_kernel=complete`**, gated on real hardware —
> `../BACKEND_AUDIT.md`). Unlike Apple — which ships no public LLVM→AIR, so MSL-source
> synthesis is the only path — **AMD has a fully open LLVM AMDGPU backend + HIP/HIPRTC**,
> so the MLIR→LLVM→AMDGPU codegen path becomes real and testable on owned silicon, and
> it is the path that **generalizes to NVIDIA** where the Apple path does not.

## The hardware — three engines, three Tessera stories

| Engine | What | Tessera status | Action |
|--------|------|----------------|--------|
| **Zen 5 CPU** | 16 cores, AVX512 (VNNI/BF16); **no Intel AMX** | ✅ correct-by-construction (verified by source read) | Execute-verify on box |
| **Radeon 8060S iGPU** | RDNA 3.5 = **gfx1151**, 40 CUs / 2560 ALUs, WMMA 16×16×16 | 🟡 target model grounded (this work); execution is the roadmap below | Stages A→D below |
| **XDNA 2 NPU** | 50 TOPS, AIE array | ⛔ out of scope (separate MLIR-AIE / IRON-Peano toolchain) | Park |
| **Memory** | 128 GB unified LPDDR5x @ **256 GB/s** | n/a | Bandwidth-bound roofline; large-model / long-KV capacity is the win |

## Engine 1 — Zen 5 CPU (AVX512): verified correct-by-construction

The x86 backend's BF16 GEMM dispatch (`src/compiler/codegen/tessera_x86_backend/src/backend_x86.cpp`)
runtime-gates AMX via CPUID and **cleanly falls back to AVX512**:

```cpp
if (cfg_.preferAMX && perfectAMXTile && amxAvailable()) tessera_x86_amx_gemm_bf16(...);
else                                                     tessera_x86_avx512_gemm_bf16(...);
// amxAvailable() = tessera_x86_amx_supported() (CPUID) && tessera_x86_amx_enable_linux()
```

On Zen 5 `tessera_x86_amx_supported()` is false → the AVX512 path is **always** taken. The
AVX512 kernel further emulates BF16→FP32 when not built with native BF16, so it is correct
regardless of build flags. CMake already auto-adds `-mavx512bf16` / `-mavx512vnni`
(`tessera_x86_backend/CMakeLists.txt`), so a build on the box (or `-march=znver5`) gets the
**native** AVX512-BF16/VNNI path, not emulation.

- **Conclusion:** day-one good. AMX-targeting code does not break on Zen 5; AVX512 is the path.
- **Caveat:** verified by source read only — this dev machine is Apple Silicon (ARM64), so the
  x86 backend does not compile/run here. **Execution-verify on the box** (build with
  `-march=znver5`, run the GEMM/attention unit + benchmark smoke).

## Engine 2 — RDNA 3.5 iGPU (gfx1151): the frontier

### ISA-grounded facts (RDNA3.5 ISA §7.9 WMMA)

- **`V_WMMA_*` (VOP3P), tile 16×16×16** (M=N=K=16). One tile shape across all dtypes.
- Dtype combos: `F32←F16`, `F32←BF16`, `F16←F16`, `BF16←BF16`, `I32←IU8`, `I32←IU4`.
- **No FP8/FP4 WMMA on RDNA 3.5** (that is CDNA 4 / RDNA 4). Tessera FP8 matmul must
  *decompose* on this chip, not use a native instruction.
- RNE-only floats. **wave32.** A is column-major in VGPRs; B/C/D row-major; lanes 0-15
  replicated into 16-31. Back-to-back dependent WMMA needs a `V_NOP` if D overlaps next A/B.
- rocdl/LLVM intrinsic surface: `llvm.amdgcn.wmma.f32.16x16x16.f16` (and `.bf16`, `.f16.f16`,
  `.bf16.bf16`, `.i32.16x16x16.iu8`, `.iu4`). Present in LLVM 16+; fully in the box's LLVM 22.

### Target model — grounded (this work, 2026-06-17)

`python/tessera/compiler/rocm_target.py` now has `AMDArch.GFX_1151` + a new `_WMMA_VARIANTS`
table (`wmma_variants()`, `.wmma_shapes`). gfx1151: wave32, WMMA 16×16×16, **`wmma_f8`
`not_supported`** (the load-bearing ISA distinction from gfx1200), dtypes `{fp32,fp16,bf16,int8}`,
no MFMA. Capability entry `rocm_gfx1151` in `capabilities.py`. Guards in
`tests/unit/test_target_toolchain_pins.py` (`TestROCmWMMAShapeTable`, gfx1151 feature/registry
tests). This is hardware-free pre-work — the target model is correct *before* the box arrives.

### Execution roadmap — rung ladder to first real gfx1151 GEMM

Mirrors the NVIDIA PTX/Evaluator track (`compiler/ptx_emit.py`, `EVALUATOR_PLAN.md` §9.5):
emit → structurally validate (host-free) → assemble (toolchain) → execute-and-compare (silicon).

| Stage | Rung | What | Gate |
|-------|------|------|------|
| **A. Emit** | 2.5 | Lower a 16×16×16 WMMA bf16 GEMM to `rocdl.wmma.*` → LLVM IR text → AMDGCN `.s` for `gfx1151`. Structural validator (host-free): the `v_wmma_f32_16x16x16_f16` op + wave32 + correct VGPR layout. Parallel to `ptx_emit.py`. | none (host-free) — **can start now** |
| **B. Assemble** | 3 | `hipcc --offload-arch=gfx1151` (or `llc -mcpu=gfx1151 -filetype=obj`) compiles A to a real object. Skip-clean when ROCm/hipcc absent (like NVIDIA's `ptxas` rung). | ROCm toolchain installed |
| **C. Launch** | 6 | Register a HIP launcher into the existing C-ABI bridge `tsrRegisterGpuLauncher` (landed G7, see `../BACKEND_AUDIT.md`); HIPRTC-compile + `hipModuleLaunchKernel` the gfx1151 kernel. | the box + working ROCm runtime |
| **D. Prove** | 7 | Execute-and-compare the WMMA GEMM vs numpy (Evaluator vertical oracle); flip `backend_kernel` for `tessera.matmul` on `rocm_gfx1151` to a real-execution status. **First non-Apple `backend_kernel` proof.** | the box |

Two viable emit paths for Stage A (pick after a spike):
- **(i) MLIR-native (preferred):** the AMD analog of the NVIDIA NVVM/Tile-IR path, through the MLIR
  **`amdgpu` → `rocdl`** two-layer (grounded from the [ROCDL dialect docs](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/)).
  ROCDL is the low-level "wrappers around AMD-specific intrinsics" dialect (the AMD NVVM); the
  higher-level `amdgpu` dialect (`amdgpu.wmma`) lowers to it. Target `amdgpu.wmma` and let it lower to
  `rocdl.wmma.*` → LLVM AMDGPU → GCN. **gfx1151 (RDNA 3.5) op family — the 16×16×16 WMMA set, matching
  the ISA Table 33:** `rocdl.wmma.f32.16x16x16.{f16,bf16}`, `rocdl.wmma.f16.16x16x16.f16`,
  `rocdl.wmma.bf16.16x16x16.bf16`, `rocdl.wmma.i32.16x16x16.{iu8,iu4}`. Kernel scaffold uses
  `rocdl.workgroup.id.{x,y,z}` / `rocdl.workitem.id.*` / `rocdl.barrier`. **⚠️ Correctness gate:** the
  ROCDL dialect ALSO exposes RDNA4/gfx12 FP8 WMMA ops (e.g. `rocdl.wmma.f32.16x16x128.bf8_bf8`) — but
  **RDNA 3.5 has no FP8 WMMA**, so the gfx1151 lowering must NOT emit the `bf8`/larger-K ops (this is
  exactly the `wmma_f8 = not_supported` gate already in `rocm_target.py`). `tessera-emit-rocdl` is
  scaffolded in `tessera-opt`.
- **(ii) HIP-source (oracle):** synthesize a HIP C++ kernel using
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, compile with hipcc. Faster to first-light, less
  reusable; good as a correctness oracle for (i).

### Honest external gates

- **gfx1151 is officially supported via the "ROCm on Radeon and Ryzen" client track (≥ 7.2.0,
  with dedicated 7.2.1 release notes for Ryzen AI Max+).** AMD ships a "ROCm 7.2.1 on Radeon and
  Ryzen for Linux" release-notes page; gfx1151 support landed in 7.0/7.2.0 (counter-collection for
  gfx1150/1151 added in 7.2.0) and continues in 7.2.x. AMD's recommended stack for Ryzen AI Max+ is
  **Ubuntu 24.04.3 inbox graphics drivers + ROCm 7.2.1**. Note this is the *Radeon/Ryzen client*
  track, distinct from the Instinct "Supported GPUs" matrix — so cite the Radeon-Ryzen docs, not the
  data-center matrix. Tessera's **7.2.4 pin (≥ the user-confirmed 7.2.1/7.2.2)** is covered by
  official support; community guides (ollama, llama.cpp) corroborate it enumerates **without** an
  `HSA_OVERRIDE_GFX_VERSION` hack on 7.2. Still `rocminfo`-verify enumeration on the box before Stage B.
- **⚠️ Documented gfx1151 bf16 correctness bugs (ROCm/ROCm#6034: "5 critical bf16 bugs").** This is
  directly load-bearing for us: Stage D's first proof is a **bf16 WMMA GEMM**. Mitigation —
  bring up the **fp32←f16 and f16←f16 WMMA combos first** (Stage D), then bf16, and cross-check any
  bf16 mismatch against the upstream bug list rather than assuming a Tessera codegen bug. The
  Evaluator's execute-and-compare oracle is exactly the instrument to catch this.
- **Bandwidth-bound:** 256 GB/s. Roofline analysis and `flywheel.py` per-chip calibration should get
  a Strix Halo entry; perf expectations are capacity-led (128 GB unified) not bandwidth-led.

## Sequencing when the box lands

1. **CPU first (lowest risk):** build x86 backend `-march=znver5`, run GEMM/attention units +
   benchmark smoke → confirm the AVX512 path executes and matches.
2. **ROCm enumeration:** install ROCm, confirm `rocminfo` sees gfx1151 (apply `HSA_OVERRIDE` if needed).
3. **Stage A spike (can begin now, host-free):** emit + structurally validate a gfx1151 WMMA GEMM.
4. **Stages B→D:** assemble → HIP-launch → execute-and-compare → first real ROCm `backend_kernel` proof.

## Cross-refs

- `../BACKEND_AUDIT.md` — the hardware-gated frontier (0 `backend_kernel=complete`) + `tsrRegisterGpuLauncher`.
- `ROCM_AUDIT.md` — ROCm theme audit (Next Work aligns with Stages C/D here).
- `python/tessera/compiler/rocm_target.py`, `capabilities.py` — the grounded gfx1151 target model.
- `docs/rocm_mfma_kernel_inventory.md` — CDNA MFMA inventory (RDNA WMMA inventory is a sibling TODO).
- `compiler/ptx_emit.py`, `EVALUATOR_PLAN.md` §9.5 — the NVIDIA rung ladder this mirrors.
