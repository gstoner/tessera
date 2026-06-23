# Strix Halo (Ryzen AI Max+ 395) — Tessera Bring-Up & RDNA 3.5 Execution Plan

> First real **non-Apple** silicon Tessera will own. Authored 2026-06-17 against the
> "RDNA3.5" Instruction Set Architecture Reference Guide (AMD, 23-July-2024).
> Strategic context: this box is the **unblock for the ROCm `backend_kernel` axis**
> (today **0 entries `backend_kernel=complete`**, gated on real hardware —
> `../BACKEND_AUDIT.md`). Unlike Apple — which ships no public LLVM→AIR, so MSL-source
> synthesis is the only path — **AMD has a fully open LLVM AMDGPU backend + HIP/HIPRTC**,
> so the MLIR→LLVM→AMDGPU codegen path becomes real and testable on owned silicon, and
> it is the path that **generalizes to NVIDIA** where the Apple path does not.

## Bring-up status — box landed (2026-06-22)

The Strix Halo box is here (Ubuntu 24.04 LTS under **WSL2**, ROCm **7.2.4**,
LLVM/MLIR **22.1.8** from apt.llvm.org). Findings that update this plan:

- **The part is RDNA 3.5 = `gfx1151` (the true ISA). `gfx1100` is a *transient
  WSL enumeration*, not a permanent target.** The Radeon 8060S in Strix Halo
  (Ryzen AI MAX+ 395) is RDNA 3.5, ISA `gfx1151` (RDNA 3.5 ISA Ref Guide, doc
  70649, 23-Jul-2024). (`gfx1150` is the related Strix *Point* iGPU — Radeon
  890M — a distinct part; the 8060S is `gfx1151`.) The toolchain **fully
  supports `gfx1151`** — verified here: `hipcc --offload-arch=gfx1151` compiles
  and `llc -mcpu=gfx1151` emits a real `v_wmma_f32_16x16x16_f16`. So **`gfx1151`
  is the codegen target.** Today the WSL/ROCm 7.2.4 runtime *enumerates* the
  device as `Name: gfx1100` (RDNA 3 discrete profile) — a **temporary WSL
  limitation; AMD's WSL enablement will report the native RDNA 3.5 arch
  (`gfx1151`)**. A `gfx1100` binary also assembles, so it is the current-WSL
  transitional alias, but the bring-up targets `gfx1151` going forward. **Both
  are RDNA, same 16×16×16 WMMA op family, no FP8 WMMA** — the rung-3 emit tests
  assemble for both (gfx1151 primary + gfx1100), and the Stage C launcher uses
  hipcc's device-default arch, so it auto-adapts when WSL starts reporting
  `gfx1151`. Both archs are in `rocm_target.py`/`capabilities.py`.
- **External gates from "Honest external gates" below are now cleared:**
  `rocminfo` enumerates the GPU **without** any `HSA_OVERRIDE_GFX_VERSION`;
  `hipcc --offload-arch=gfx1100` compiles a WMMA kernel (374 lines of AMDGCN
  `.s`); the ROCm backend lit suite is **11/11**; `tessera-opt` +
  `tessera-rocm-opt` build clean against LLVM/MLIR 22.1.8. So **Stages A and B
  are fully unblocked**, and the box presence unblocks C/D.
- **Build-state findings (verified in-tree):**
  - The `--tessera-emit-rocdl` pipeline (linalg→scf.parallel→gpu→ROCDL) scaffold
    exists in `tessera-opt`; the `tessera_rocm`/`rocdl` dialects are registered.
  - **`lower-tile-to-rocm` had no WMMA path — it always emitted
    `tessera_rocm.mfma` regardless of arch**, which is wrong for RDNA (no MFMA).
    Fixed in the Stage A increment (2026-06-22): a `tessera_rocm.wmma` op +
    arch-keyed selection (`gfx11xx` → WMMA) + a `llvm.amdgcn.wmma.contract`
    ROCDL marker, with the no-FP8-on-RDNA gate preserved. Lit fixtures added.
  - The HIP runtime `loader.cpp` already has real
    `hipModuleLoad`/`hipModuleGetFunction`/`hipModuleLaunchKernel` (behind
    `TESSERA_HAS_HIP`), **but it is a standalone `tessera_rocm_runtime` lib and
    is NOT wired into the core C-ABI bridge `tsrRegisterGpuLauncher`**, and
    there is **no HIPRTC** path (it loads a prebuilt `.hsaco`). That wiring is
    the bulk of **Stage C**.
  - The arch knob is `lower-tile-to-rocm{arch=gfxNNNN}` (and the FP8 gate). The
    `--rocm-target=gfxNNNN` flag referenced by the `tests/tessera-ir/phase8/
    rocm_7_2/*.mlir` fixtures is **stale — no tool implements it**; those
    fixtures are not in the run suite. Canonical fixtures live in
    `src/compiler/codegen/Tessera_ROCM_Backend/test/rocm/`.
  - **The MLIR `--tessera-emit-rocdl` pipeline is broken on this build:** it
    references a pass `tessera-to-linalg` that is **not registered** in
    `tessera-opt` here, so the pipeline string fails to build and aborts
    (`-nvvm` too). That blocks the *MLIR-graph*→WMMA route. **Stage B does not
    depend on it** — `rocdl_emit.py` (the AMD analog of `ptx_emit.py`) is a
    direct LLVM-IR WMMA emitter, the established rung pattern. Registering
    `tessera-to-linalg` into `tessera-opt` is its own follow-up.

### Stage B — DONE, verified on the box (2026-06-22)

`python/tessera/compiler/rocdl_emit.py` already implements rung 2.5 (emit
`llvm.amdgcn.wmma.*` LLVM IR + host-free structural validators) **and** rung 3
(`llc -mcpu=<arch>` → real `v_wmma_*` AMDGCN), the AMD mirror of `ptx_emit.py`.
On the box this now runs for real (not skip-clean):

- **rung 3 asm:** the K-reduction / operand-layout / D-store WMMA GEMMs lower to
  AMDGCN with the documented `v_wmma_f32_16x16x16_{f16,bf16}`, lane-replication
  (`v_and_b32 _,15,_`), and strided `global_store`s — now parametrized over
  **gfx1100** (the box) as well as gfx1151.
- **rung 3 object:** new `llc_object()` lowers the GEMM to a real relocatable
  **AMD GPU ELF** (`EM_AMDGPU`) — the plan's "compiles A to a real object" gate,
  confirmed for gfx1100/gfx1151 (`test_rung3_gemm_assembles_to_amdgpu_elf_object`).
- `_find_llc()` now finds the apt.llvm.org `llc` (`/usr/lib/llvm-22/bin/llc`), so
  the rung-3 tests run by default on the box. `tests/unit/test_rocdl_emit.py`:
  **96 passed, 0 skipped.**

### Stage C — DONE: first non-Apple kernel through the C-ABI launch bridge (2026-06-22)

A real GPU kernel now **executes on the gfx1100 device through Tessera's C-ABI
launch bridge** — the first non-Apple backend to do so. Mirrors the Apple G7
proof (`test_runtime_abi_gpu_launch_bridge.py`): a hipcc-compiled harness
registers a `tsrGpuLauncherFn` for `(target="rocm", "tessera_rocm_gemm_f32")`
that runs a real `__global__` GEMM (hipMalloc / H2D / launch / sync / D2H) over
the params' buffers + dims; it compiles a `rocm` artifact, launches via
`tsrLaunchKernel`, and **verifies the GPU output equals `A @ B`**. An
unregistered kernel name still returns `UNIMPLEMENTED` (the bridge never
silently succeeds). Test: `tests/unit/test_runtime_abi_rocm_launch_bridge.py`.

Two real fixes landed with it:
- **Runtime CMake HIP-include bug:** with `-DTESSERA_ENABLE_HIP=ON`,
  `hip_backend.cpp` was compiled without the HIP include path
  (`fatal error: hip/hip_runtime.h`). `tessera_runtime` now links `hip::host`
  (or falls back to `$ROCM_PATH/include`) — `libtessera_runtime.a` builds with
  HIP enabled.
- **WSL device-enumeration quirk:** `hipGetDeviceCount` reports **0** under WSL
  even though kernels launch and compute correctly. The harness gates on a real
  HIP probe (malloc + sync round-trip), not the device count, and skip-cleans
  (`SKIP_NO_DEVICE`) when no usable GPU is present.

**Honesty ceiling.** Stage C proves the *launch bridge + execution mechanics*
with a correct (naive) GEMM kernel compiled by hipcc. It does **not** yet route
the Stage A/B **WMMA** kernel, nor is the launcher auto-registered by a shipped
ROCm runtime lib (it lives in the test harness, exactly as the Apple G7 proof
does). That WMMA execute-compare is Stage D, below.

### Stage D — WMMA matrix-core GEMM executes-and-compares through the bridge (2026-06-22)

The real RDNA WMMA matrix instruction now runs on the device and produces a
**numerically correct GEMM**, routed through the C-ABI launch bridge. The kernel
uses `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (the same
`v_wmma_f32_16x16x16_f16` `rocdl_emit.py` emits), with the operand/accumulator
fragment layout matching the grounded mapping in `rocdl_emit.py` (col = lane&15,
row = 2·e + lane>>4). A 16×16×16 `f32 ← f16` tile vs a host reference: **maxerr
≈ 3e-8 standalone, < 1e-2 through the bridge** (`f16` rounding). Test:
`tests/unit/test_rocm_wmma_execute_compare.py`. We bring up `f32←f16` first
(bf16 has documented gfx115x bugs).

**What this clears, and what it does NOT.** This is a genuine on-hardware
execute-and-compare of the WMMA op — the *numerical-proof* half of the
`backend_manifest` `hardware_verified` contract (`execute_compare_fixture`). It
is **deliberately NOT promoted to `hardware_verified` / `backend_kernel`
complete**, because that status also requires a **shipped `runtime_symbol`** — a
C-ABI kernel symbol that runs at dispatch from an auto-registered ROCm runtime
lib (cf. Apple's `tessera_apple_gpu_mps_matmul_f32` in the shipped runtime).
Today the WMMA kernel + launcher live in the *test harness* (exactly as the
Apple G7 bridge proof does), not a shipped, auto-registering backend lib.
Promoting the manifest row now — with a test-only symbol — would be the audit
inflation Decision #25 forbids. **The formal `backend_kernel` flip is gated on
the remaining "ship an auto-registered ROCm runtime launcher" item** (Next Work
in `ROCM_AUDIT.md`); the numerical proof is in hand, so that flip becomes
mechanical once the symbol ships. It is also a single 16×16×16 tile, not a
general tiled/K-looped GEMM (a separate scale item).

### Stage E — shipped runtime symbol + runtime.launch() lane + tiled/bf16 (2026-06-22)

The WMMA kernel is now a **shipped, auto-built symbol** wired into the runtime:
`libtessera_rocm_gemm.so` exports `tessera_rocm_wmma_gemm_{f16,bf16}` (HIPRTC-
compiled for the device arch at load), `runtime.launch()` dispatches
`target="rocm"` matmul to it via the `rocm_wmma` execution-matrix lane, and the
kernel is a **general tiled/K-looped GEMM** (any positive M/N/K, ragged edges
zero-padded). The `backend_manifest` matmul row is `hardware_verified` for
`{fp16, bf16}`. Closes the Stage D "ship an auto-registered launcher" gate.

### Stage F — GEMM perf ladder (2026-06-22, in progress)

Correctness done, now performance — grounded in the AMD **Gluon GEMM tutorial**
v0→v9 ladder (`ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md` §B1/§B2). Rungs are *measured*,
not asserted: `tessera_rocm_wmma_gemm_f16_bench` (in the shipped lib) hipEvent-
times kernel-only launches (buffers reused), and
`benchmarks/rocm/benchmark_rocm_wmma_gemm.py` emits the stable JSON schema
(Decision #12) + a `--ladder` sweep.

**Rung 1 — output-tile register blocking (DONE).** Each 32-lane wave computes an
MT×NT grid of 16×16 WMMA tiles; a loaded A fragment is reused across NT B-tiles
and a B fragment across MT A-tiles, cutting global-load traffic per output
element. Measured best-of-3 on **gfx1100/WSL (Ryzen AI Max+ 395 / Radeon 8060S)**,
f16, kernel-only TFLOP/s:

| MT×NT | 512³ | 1024³ | 2048³ |
|------|-----:|------:|------:|
| 1×1 (rung-0 naive) | 1.68 | 3.36 | 4.00 |
| 2×2 | 1.72 | 2.24 | 3.37 |
| **2×4 (production)** | 3.47 | **7.87** | **9.46** |
| 4×2 | 3.91 | 4.91 | 8.79 |
| 4×4 | 1.65 | 5.47 | 6.94 |

**~2.3× over the naive baseline** at the compute-bound sizes. The empirical
lesson mirrors Gluon exactly: the **tile shape is the lever, and the obvious
choice can regress** — `2×2` lands *below* `1×1` here (occupancy/register
pressure), while the non-square `2×4` wins. Shipped tiling = `kProdMT=2,
kProdNT=4` in `tessera_rocm_gemm.cpp`; correctness unchanged (the
execute-compare fixture passes at 2×4).

**Rung 2 — LDS staging, multi-wave workgroup (IMPLEMENTED; did NOT win — kept
behind the bench).** A WM×WN-wave workgroup cooperatively stages the A/B 16-wide
K-panels for its macro-tile into LDS once per K-step, then every wave reads its
WMMA fragments from LDS. Correct across shapes (fixture
`test_shipped_rocm_wmma_lds_matches_numpy`), shipped as
`tessera_rocm_wmma_gemm_f16_lds` + `..._bench_lds`. **Measured verdict on
gfx1100 (best-of-3 f16 TFLOP/s, `--lds`):**

| size | rung-1 reg 2×4 | best rung-2 LDS |
|------|---------------:|----------------:|
| 512³  | **3.47** | 3.20 (4×2w 1×2t) |
| 1024³ | **8.09** | 7.85 (4×1w 2×4t) |
| 2048³ | 8.88 | **9.38** (4×1w 2×4t) |
| 4096³ | **11.40** | 8.46 |

Single-buffer LDS staging is a **wash-to-regression** here: it loses at
512/1024, edges +6% at 2048, and loses decisively at 4096. This is the Strix
Halo unified-memory story — global bandwidth is shared with the CPU and is *not*
the bottleneck LDS staging targets, so the `__syncthreads` + occupancy cost
isn't repaid. So **production stays rung-1 register blocking (2×4).** This is the
Gluon v6 lesson generalized: the "obvious next optimization" must be measured,
not assumed. Rung 2 is kept as a shipped, correctness-guarded symbol because it
is the **substrate for rung 3** (software pipelining needs LDS buffering) and
should pay off on discrete RDNA / CDNA where global *is* the bottleneck.

**Open rungs (next):** 2-/3-stage K-loop software pipelining over the LDS buffers
(§B2) — the rung where LDS staging starts to earn its keep by overlapping global
loads with WMMA; arch-aware LDS layout (swizzle vs pad). Per Gluon's v6 lesson,
watch register budget — double-buffer can regress if it spills. Not wired yet.

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
- `python/tessera/compiler/rocdl_emit.py` + `tests/unit/test_rocdl_emit.py` — Stage B emitter (rung 2.5 + rung 3 `llc`), AMD analog of `ptx_emit.py`.
- **RDNA 3.5 ISA Reference Guide** (AMD doc 70649, 23-Jul-2024) — authoritative WMMA spec (§7.9 / Table 33): <https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture>. Note: docs.amd.com is a JS-rendered SPA — fetch the linked PDF, not the HTML.
