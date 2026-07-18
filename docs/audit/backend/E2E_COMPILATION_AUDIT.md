---
last_updated: 2026-07-18
audit_role: plan
plan_state: open
---

# Backend End-to-End Compilation Audit

**What this audits.** For each backend (Apple GPU, Apple CPU, x86, ROCm
gfx1151, NVIDIA sm_120/sm_90) and each frontend lane, *where does the compiler
pipeline actually end*, and what is missing to reach genuine
**compiler-native** end-to-end compilation.

**The bar (Def B — compiler-native).** A lane is E2E only if the registered
MLIR pipeline lowers frontend IR → Target IR → native code **without the Python
runtime orchestrating the Tile→Target seam**. This is deliberately stricter
than "the op runs on device by any path" (Def A), which the generated
dashboards already report — see
[`../generated/runtime_execution_matrix.md`](../generated/runtime_execution_matrix.md)
and [`../generated/e2e_op_coverage.md`](../generated/e2e_op_coverage.md). The
gap between Def A and Def B is this document's subject.

> Status truth for *counts* stays with the generated dashboards (Decision #26).
> This audit owns the *architectural* claim — which lanes are compiler-lowered
> vs. runtime-bridged — grounded in pass/source citations, not counts.

---

## TL;DR — the one structural finding

**On all four backends, the registered MLIR pipeline genuinely lowers only the
core GEMM/matmul lane (plus, on Apple GPU, a broad per-op fusion set). Every
other lane that executes at all does so through a Python `runtime.py`
side-channel, not through the compiler's Tile→Target lowering.** The
compiler-side *consumers* (generator passes / rewrite patterns) exist, but **no
in-tree pass emits the Target-IR directive from Tile IR** — Python bridges that
seam.

The bridge differs per backend but the shape is identical:

| Backend | Registered pipeline lowers to native… | Everything else reaches the device via… |
|---|---|---|
| **Apple GPU** | GEMM + ~16 per-op/fusion passes (attn, rope, softmax, gelu, unary, silu_mul, row-op, linear-attn, MLA-fusion, NSA-fusion, swiglu) | envelope `status`/`symbol` attrs → Python calls `extern "C" tessera_apple_gpu_*` in `apple_gpu_runtime.mm` |
| **Apple CPU** | **rank-2 f32 matmul → `cblas_sgemm` only** | everything else stops at artifact IR / numpy reference |
| **x86** | **matmul + fused-epilogue only** (`TileToX86Pass.cpp` has exactly 3 patterns) | `runtime.py` `KNOWN_EXECUTORS` keyed on `compiler_path` → ctypes into `avx512_*.cpp` |
| **ROCm** | **WMMA/MFMA GEMM only** (default pipeline wires 1 generator + `TileToROCM`) | `runtime.py` `_build_compiled_*_hsaco` hand-writes `tessera_rocm.*` directive text → shells `tessera-opt` single-generator → HSACO |
| **NVIDIA** | **two disconnected stacks**; MLIR pipeline is artifact-only except sm_120 `mma.sync`/NVFP4 | Python NVRTC/PTX (`nvidia_cuda.py` + `ptx_launch.cpp` bridge) — real on sm_120 |

**Why it matters beyond aesthetics.** Because the non-GEMM lanes bypass the
Target IR dialect, they are **not covered by the hardware-free Target IR
contract (Decision #19)** nor the golden-IR determinism tripwire — the
compiler has no typed, verifiable representation of what those kernels do. The
correctness burden sits entirely in hand-written kernels + Python glue.

**Is the side-channel wrong?** Not inherently. Decision #28 explicitly allows a
generic synth→compile→cache→launch loop to live outside the hand-written
lowering. The problem is *where the seam sits*: today Python hand-builds
directive **text** (ROCm) or ctypes-calls **ABI symbols** (x86) directly,
bypassing typed Tile IR. The fix is not to delete the side-channel but to make
it flow through a Tile→Target pass that emits the Target-IR directive from
typed IR, so Decision #19/#28 coverage extends past GEMM.

---

## Legend

| Code | Meaning |
|---|---|
| ✅ **PIPE** | Registered MLIR pipeline lowers this lane to native code |
| 🐍 **SC** | Executes natively, but **only** via the Python runtime side-channel (no in-pipeline Tile→Target) |
| 📄 **ART** | MLIR emits an artifact/marker only — no execution |
| 📚 **REF** | numpy / scalar reference only |
| ∅ | Nothing on this target |

Multiple codes = split behavior; see the per-backend notes for the nuance.

---

## Consolidated matrix (Def B — compiler-native)

| Frontend lane | Apple GPU | Apple CPU | x86 | ROCm gfx1151 | NVIDIA sm_120 | NVIDIA sm_90 |
|---|---|---|---|---|---|---|
| Core GEMM / matmul | ✅ | ✅¹ | ✅² | ✅³ | ✅/📄⁴ | 📄⁵ |
| Fused epilogue (bias/gelu) | ✅ | 📄 | ✅ | 🐍 | 🐍 | 🐍 |
| Flash-attention / SDPA | ✅⁶ | 📚 | 🐍 | 🐍⁷ | 🐍 | 🐍 |
| Reasoning attn (linear/MLA/NSA/local-win) | ✅⁶ | 📚 | 🐍 | 🐍 | 🐍 | 🐍 |
| Reasoning attn (lightning/kimi/hybrid/lookahead) | 🐍/📚 | 📚 | 📚⁸ | ∅⁹ | ∅ | ∅ |
| GA / Clifford | 🐍¹⁰ | 📚 | 🐍 | 🐍 | ∅ | ∅ |
| EBM | 🐍¹⁰ | 📚 | 🐍 | 🐍 | ∅ | ∅ |
| RL losses (ppo/grpo/cispo) | 🐍/📚¹¹ | 📚 | 🐍 | 🐍 | ∅ | ∅ |
| Elementwise / unary / binary | ✅ | 📚 | 🐍 | 🐍 | 🐍 | 🐍 |
| Norm / softmax / reduction | ✅ | 📚 | 🐍 | 🐍 | 🐍 | 🐍 |
| Structured (conv / sparse / moe) | 🐍 | 📚 | 🐍 | 🐍 | 🐍 | 🐍 |
| Spectral / FFT | 📚 | 📚 | 🐍 | 🐍 | 📚 | 📚 |
| Linalg (cholesky/lu/qr/svd) | 🐍¹² | 🐍¹² | 🐍 | 🐍 | 📚 | 📚 |
| Control flow | 📄¹³ | 📄 | ✅¹⁴ | 🐍 | 📄 | 📄 |
| KV cache | 🐍 | 🐍 | 📄¹⁵ | ∅¹⁶ | 📄 | 📄 |

**Footnotes**
1. `tessera-lower-to-apple_cpu-runtime` lowers **only** rank-2 f32 matmul → `cblas_sgemm` (`Tessera_Apple_Backend/lib/Target/Apple/Passes.cpp:65-71`).
2. `LowerMatmulToX86` requires bf16/f16, 2-D, static (`src/transforms/lib/TileToX86Pass.cpp:67-149`). f32 GEMM kernel exists (`tessera_x86_avx512_gemm_f32`) but the pass rejects it → 🐍.
3. `GenerateWMMAGemmKernelPass` + `TileToROCM` (`tile.mma`→`tessera_rocm.wmma`/`.mfma`) → ROCDL. f16/bf16 only; fp8 matrix is a hard, named error on gfx1151 (`Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp:391-398`).
4. sm_120 `mma.sync` lowers to real `NVVM::MmaOp` (`tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp:1492-1518`) — but the **named pipeline stops at NVVM/marker**; on-device execution is via the separate Python NVRTC launch bridge (`runtime/cuda/tessera_nvidia_ptx_launch.cpp:105-191`), proven on RTX 5070 Ti. Two stacks, not one.
5. sm_90 `wgmma` is a **void marker** (`NVIDIALowering.cpp:1334-1347`); `ptx_emit.py` emits wgmma PTX but it is **not wired into the launch-bridge ABI table** → not executed.
6. Registered `tessera-lower-to-apple_gpu-runtime` wires real per-op passes (`Passes.cpp:89-131`). Runtime **falls back to host-reference** for out-of-envelope inputs: flash-attn D>256, MLA absorb-K, NSA branches, linear-attn non-causal (`MLADecodeFusionToAppleGPU.cpp:152-153`, `NativeSparseAttnFusionToAppleGPU.cpp:13-14`, `LinearAttnToAppleGPU.cpp:15-18`).
7. `GenerateWMMAFlashAttnKernelPass` has a partial "via-tile" hook but is opt-only and its input `tessera_rocm.flash_attn` is produced only by `runtime.py`.
8. x86 kernels for `lightning`/`hybrid` do **not** exist by name — `x86_linear_attn_compiled` covers only linear_attn/power_attn/retention → reference fallback.
9. No generator file exists for kimi/lookahead/hybrid on ROCm (ODS comments only).
10. GA/EBM value lowerings exist in `TileToApple.cpp` **value-mode**, which **no registered pipeline enables** (`valueMode=true` reachable only by a caller flag); native path is the envelope side-channel + the 17 clifford / EBM MSL kernels in `apple_gpu_runtime.mm`.
11. PPO has a value-mode lowering (`TileToApple.cpp:921-943`); grpo/cispo are reference. No NVIDIA/ROCm-vs-Apple RL parity — RL native lanes are x86 + ROCm.
12. Apple linalg (`kLinalgSpecs`) is value-mode-only (unregistered) + `.mm`/`apple_cpu_runtime.cpp` LAPACK symbols.
13. Apple control-flow ops lower but carry `status="artifact"` (lit-verified; MLIR-driven execution deferred); the GraphFn route calls `run_graph_*` runtime symbols directly.
14. x86 pipeline runs `LowerControlFlowToSCFPass` + `ControlFlowTargetGuard("x86")` in-pipeline (`src/transforms/lib/Passes.cpp:114-118`).
15. `LowerKVCacheToX86` emits an attribute-only op and **bails if the result is consumed** — explicitly "not native-execution evidence" (`TileToX86Pass.cpp:293-334`).
16. `tile.kv_cache` and `tile.tmem.*` are explicit hard errors in `TileToROCM` (`TileToROCM.cpp:712,718`).

---

## Per-backend detail

### Apple GPU — the reference implementation of an in-pipeline backend

Apple GPU is the **only** backend where a *registered* pipeline lowers a broad
lane set to executable Target IR. `tessera-lower-to-apple_gpu-runtime`
(`Passes.cpp:89-131`) runs 17 passes in longest-fusion-first order: NSA-fusion,
MLA-decode-fusion, swiglu-fusion, 4 matmul-fusions, matmul, rope, flash-attn,
linear-attn, local-window-attn, softmax, gelu, unary, silu_mul, row-op — each
producing `tessera_apple.gpu.kernel_call` value ops with a concrete C-ABI
`symbol` + `status`, dispatched by `apple_gpu_runtime.mm` (112 inline MSL
kernels, MPS/MPSGraph lanes).

**Gaps to full compiler-native:**
- **GA/EBM/linalg/PPO are not in the registered pipeline.** Their value
  lowerings live in `TileToApple.cpp` *value-mode*, which no
  `PassPipelineRegistration` enables — native execution routes through the
  envelope side-channel instead. Registering value-mode (or adding
  clifford/ebm passes to the `-runtime` pipeline) closes this.
- **On-device kernels still fall back to host-reference** for MLA absorb-K, NSA
  branches, linear-attn non-causal, flash-attn D>256. The MSL kernels are
  deferred follow-ups (footnote 6).
- The **artifact** pipeline `tessera-lower-to-apple_gpu` emits `ub.poison` husks
  (`TileToApple.cpp:567-577`) — inspection only, never executable.

### Apple CPU — matmul and nothing else

`tessera-lower-to-apple_cpu-runtime` lowers **only rank-2 f32 matmul** to
`cblas_sgemm`. `tessera-lower-to-apple_cpu` is artifact-only. Value-mode covers
linalg + rank-2/3 matmul but is unregistered. Every other lane (softmax, gelu,
elementwise, norm, attention, EBM, clifford, RL) stops at artifact IR or numpy
reference. **Missing:** an executable CPU pipeline wiring per-op Accelerate/
LAPACK kernels analogous to the GPU `-runtime` pipeline.

### x86 — GEMM in-pipeline, everything else ctypes

`tessera-lower-to-x86` (`src/transforms/lib/Passes.cpp:106-148`) ends in
`TileToX86Pass`, which has **exactly 3 rewrite patterns**: `tessera.matmul`,
`tessera.fused_epilogue`, `tessera.kv_cache` (artifact-only). **There is no x86
Target IR dialect** — Decision #19 is unmet; `registerTesseraX86BackendDialects()`
is an empty stub (`backend_x86.cpp:45`), and Tile IR lowers straight to
`func.call` of C-ABI symbols. ~80 `avx512_*`/`amx_*` kernels exist (flash-attn,
clifford, ebm, deltanet, ssm, moe, loss, policy_loss, norm, softmax, reduce,
fft, svd, sparse), but all except GEMM/epilogue are reached only by
`runtime.py`'s `KNOWN_EXECUTORS` table keyed on `compiler_path`
(`runtime.py:18624-18731`). **Missing:** `TileToX86` patterns for the non-GEMM
lanes (kernels + ABIs already exist), wiring the standalone Graph-IR fusion
passes (`MLAFusion`, `NativeSparseAttnFusion`, `LightningAttnFusion`,
`RLLossDecompose`, …) into the pipeline body, and an f32/int8 GEMM path.

### ROCm gfx1151 — GEMM in-pipeline, ~70 generators stranded

`buildTesseraROCMBackendPipeline` wires 6 passes:
`GenerateWMMAGemmKernel` → WaveLds(+legality) → `TileToROCM` →
`LowerKernelABI` → `LowerTesseraTargetToROCDL`. `TileToROCM` matches **only 6
tile ops** (`tile.mma`, `tile.matmul_kernel`, `tile.async_copy`,
`tile.wait_async`, `tile.kv_cache`, `tile.tmem.*`). The **~70 directive
generators** (`GenerateROCM{Clifford,Ebm*,DeltaNet,FlashAttn,…}KernelPass`) are
registered **opt-only** and match `tessera_rocm.*` directive ops that **no
in-tree pass emits** — the sole producer is `runtime.py`, which hand-builds the
directive as MLIR **text** and shells `tessera-opt` with a single-generator
pipeline (`_build_compiled_*_hsaco`, `runtime.py:8975,12532`). **Missing:** a
Tile-matching front that recognizes the non-GEMM lanes, a pass that **emits**
the `tessera_rocm.*` directives from typed Tile IR, and wiring the generators
into the default pipeline. This is the **lead-target** gap (Decision #28: ROCm
sets the perf ceiling; today only its GEMM lane is compiler-native).

### NVIDIA — two stacks that never meet

- **Stack A (MLIR `tessera_nvidia` dialect):** the named
  `tessera-nvidia-pipeline-{sm90,sm100,sm120}` aliases share **one identical
  builder** `buildCUDA13Pipeline` — no per-SM branching, and it hardcodes
  `ControlFlowTargetGuard("nvidia_sm90")` even in the sm120 alias
  (`src/transforms/lib/Passes.cpp:403`). It ends at `NVWGMMALowering` (a WGMMA
  `func.call`) + marker passes. The real sm_120 `mma.sync`/NVFP4 lowering lives
  in `LowerTileToNVIDIA(sm=120)` — but **no registered pipeline passes sm=120**;
  it's reachable only via the raw pass option. Everything except
  `mma.sync`/NVFP4 is a void marker `llvm.nvvm.*.contract`.
- **Stack B (Python NVRTC/PTX):** the actual on-device path. `nvidia_cuda.py` +
  `ptx_emit.py` + `tessera_nvidia_ptx_launch.cpp` NVRTC/PTX-compile and launch
  GEMM, flash-attn, MLA, deltanet, linear-attn, DSA, softmax, norm, reduction,
  SwiGLU/MoE, conv2d on sm_120 silicon.

The two stacks are disconnected: the named MLIR pipeline never reaches the
NVRTC executors. **GA/EBM/RL have zero NVIDIA presence on any arch.**
**Missing:** unify the stacks (NVVM output → PTX → launch bridge), promote the
marker ops (wgmma/tcgen05/TMA/mbarrier/control) to real NVVM, wire sm_90 wgmma
PTX into the launch bridge, add per-SM pipeline branching, and — for GA/EBM/RL
— everything from Tile IR down (greenfield).

---

## Open-work queue (prioritized)

Priority inherits from Decision #28 (ROCm + CUDA are lead perf targets) and
`../MASTER_AUDIT.md` (NVIDIA breadth is P0).

**P0 — NVIDIA stack unification.** The single largest Def-A/Def-B gap: the whole
MLIR pipeline is artifact-only except sm_120 `mma.sync`/NVFP4, while real
execution rides a disconnected Python stack. Connect `LowerNVIDIAToNVVM` output
→ PTX → `tessera_nvidia_ptx_register/invoke`; register an sm=120 pipeline;
promote marker ops to NVVM; add per-SM branching.

**P0 — ROCm directive-emitting pass layer.** Lead target with only GEMM
compiler-native. Add (1) `TileToROCM` matchers for the non-GEMM tile lanes,
(2) a pass that emits `tessera_rocm.*` directives from typed Tile IR (today only
`runtime.py` text-synthesizes them), (3) the ~70 generators into
`buildTesseraROCMBackendPipeline`.

**P1 — x86 `TileToX86` breadth.** Mechanical: add rewrite patterns for the
non-GEMM lanes (kernels + stable C ABIs already exist), wire the standalone
fusion passes into `tessera-lower-to-x86`, add f32/int8 GEMM. Optionally realize
the missing hardware-free x86 Target dialect (Decision #19).

**P1 — Apple GA/EBM/linalg in-pipeline + deferred MSL kernels.** Closest to
done. Register `TileToApple` value-mode (or add clifford/ebm/linalg passes to
`-runtime`), and land the deferred MSL kernels (MLA absorb-K, NSA fused,
linear-attn non-causal) so on-device stops falling back to host-reference.

**P2 — Apple CPU executable pipeline.** Extend beyond rank-2 matmul to a per-op
Accelerate/LAPACK `-runtime` pipeline.

**P2 — GA/EBM/RL on NVIDIA.** Greenfield from Tile IR down; gated behind the P0
stack-unification work.

**Cross-cutting.** Wherever a lane moves from 🐍 to ✅, it gains Decision #19
(typed Target IR) + golden-IR tripwire coverage it does not have today — track
that as the acceptance signal, not just "still runs."

---

## Evidence routing

- Def-A native-execution truth: [`../generated/runtime_execution_matrix.md`](../generated/runtime_execution_matrix.md), [`../generated/e2e_op_coverage.md`](../generated/e2e_op_coverage.md), [`../generated/support_table.md`](../generated/support_table.md).
- Per-target maps: [`../generated/apple_target_map.csv`](../generated/apple_target_map.csv), [`../generated/rocm_target_map.csv`](../generated/rocm_target_map.csv), [`../generated/nvidia_sm90_target_map.md`](../generated/nvidia_sm90_target_map.md).
- Sibling audits: [`BACKEND_AUDIT.md`](BACKEND_AUDIT.md), [`apple/APPLE_AUDIT.md`](apple/APPLE_AUDIT.md), [`nvidia/NVIDIA_AUDIT.md`](nvidia/NVIDIA_AUDIT.md), [`rocm/ROCM_AUDIT.md`](rocm/ROCM_AUDIT.md).
- Forward direction: [`../compiler/COMPILER_THEORY_OF_OPERATION.md`](../compiler/COMPILER_THEORY_OF_OPERATION.md) (Decision #28 three-tier/arbiter model).
