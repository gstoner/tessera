---
status: Audit
classification: NVIDIA execution punch-list
authority: Phase G1 of `docs/audit/execution_roadmap.md`
last_updated: 2026-05-09
---

# NVIDIA Execution Path Audit (G1)

Goal: produce a concrete punch list of what's missing for one shape — SM_90
BF16 GEMM 128×128×128 — to actually launch on a real H100. The audit turns
G2–G7 from a vague "biggest single block" into well-scoped 1–3 day tasks.

## 1. Current State (per-component)

| Component | Status | Notes |
|-----------|--------|-------|
| Runtime backend (`src/runtime/src/backend/cuda_backend.cpp`) | 🟡 Wired, not exercised | `cudaMalloc`, `cudaMemcpy`, `cudaStream`, `cudaEventCreate` all implemented (lines 101–234). **Missing:** a `cuLaunchKernel` call site exposed via the C ABI. |
| NVIDIA codegen (`NVWGMMALoweringPass.cpp`) | ✅ Wired & tested | Lowers `tile.mma` → `wgmma.mma_async` PTX for SM_90+ (lines 88–93); fallback WMMA for SM<90. |
| PTX→cubin (`nvrtc_jit.cpp` + `cuda_driver.cpp`) | ✅ Wired | NVRTC compilation (`jitToPtx`) + `cuModuleLoadDataEx` (line 70). Extracts kernel via `cuModuleGetFunction`. |
| TMA descriptor pass (`NVTMADescriptorPass.cpp`) | ✅ Wired | Hoists descriptors to kernel preamble; deduplicates by `(src, tile_rows, tile_cols)`. |
| Inline-PTX kernel (`wgmma_bf16_inline_ptx.cu`) | 🟡 Wired, not orchestrated | `wgmma_bf16_ptx_kernel` defined; TMA + mbarrier inline asm at lines 71–77; `launch_wgmma_bf16_ptx` C wrapper present. **Missing:** Python-level orchestration. |
| Python JIT (`python/tessera/compiler/jit.py`) | ⛔ Missing for GPU | `@jit(target=GPUTargetProfile(...))` accepted; Graph IR emitted; `compile_bundle` created — but **execution falls through to `fallback_eager`**. Only CPU (`_native_cpu_fast_call`, `_apple_cpu_fast_call`, `_apple_gpu_fast_call`) wired. |
| GPU launch ABI (`runtime.py`) | ⛔ Missing | `launch()` dispatches `apple_cpu` and `apple_gpu` only. No `nvidia_sm90` / `nvidia_sm80` branches. |
| Lit tests (`tests/tessera-ir/phase3/`) | 🟡 Structural only | `nvwgmma_lowering.mlir` verifies IR shape (XFAIL). No execution on real hardware. |
| Build system (`CMakeLists.txt`) | ✅ Complete | `find_package(CUDAToolkit)` under `TESSERA_ENABLE_CUDA=ON`; `tessera_gpu_backend` library; kernel compilation; test binaries (`test_wmma_gemm`, `test_wgmma_tile_ir`). |

## 2. Concrete Punch List

| ID | Task | Acceptance | Effort | Files |
|----|------|------------|--------|-------|
| **G1-2** | Implement `cuLaunchKernel` dispatcher in C++ runtime ABI | `cuda_backend.cpp` gains `launchKernel(Stream*, FunctionHandle, dim3 grid, dim3 block, size_t smem, void** args)` exporting via `tessera_runtime.h`. | 1 day | `src/runtime/src/backend/cuda_backend.cpp:235+`, `src/runtime/include/tessera/tessera_runtime.h` |
| **G1-3** | Surface compiled WGMMA kernel handle to Python | `compile_bundle` returns a `CUmodule` + kernel metadata (grid/block, shared mem, PTX hash). Extracted in `jit.py:886` via `compile_graph_module()`. | 1–2 days | `python/tessera/compiler/driver.py:~140`, `python/tessera/compiler/jit.py:886` |
| **G1-1** | Wire Python GPU dispatcher to runtime ABI | `@jit(target=GPUTargetProfile(isa=ISA.SM_90))` on a 128×128×128 GEMM sets `execution_kind="native_gpu"` and calls a new `_execute_nvidia_gpu_artifact()` in `runtime.py`. | 1–2 days | `python/tessera/runtime.py:1070–1180`, `python/tessera/compiler/jit.py:288` |
| **G1-4** | Bind NVRTC→cubin pipeline to phase-3 Target IR | `tessera-lower-to-gpu` pipeline emits Target IR with inline PTX, invokes NVRTC (`nvrtc_jit.cpp`) → `cuModuleLoadDataEx` → kernel handle. Metadata includes kernel name + cubin binary. | 2–3 days | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/`, runtime driver |
| **G1-5** | Validate `cuLaunchKernel` + TMA descriptor flow on H100 | `test_wgmma_tile_ir` runs on H100; one 64×64×16 tile via `wgmma_bf16_ptx_kernel`. Output matches cuBLAS BF16-input/FP32-output reference within fp32 tolerance. | 1–2 days | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/tests/` |
| **G1-6** | 128×128×128 → 4× 64×64×16 tile-loop orchestration | Python dispatcher computes `grid=(2,2)`, `block=(128 threads per warpgroup)`. Either four-call K-slab loop or single PTX-emitted tiling loop. | 2–3 days | Python GPU dispatcher (G1-1) + C++ wrapper |
| **G1-7** | Replace `tessera.nvgpu` dialect placeholders with NVVM PTX | `tessera.nvgpu.wgmma.mma_async` lowered from `tile.mma` (line 10 in `NVWGMMALoweringPass`) becomes real NVVM inline PTX (or Hopper-specific GPU dialect). MLIR→PTX must not strip inline asm. | 1–2 days | NVIDIA backend IR + lowering |
| **G1-8** | End-to-end test: Python → H100 BF16 GEMM 128×128×128 | New `tests/unit/test_gpu_target.py` decorates a GEMM with `@jit(target=GPUTargetProfile(isa=ISA.SM_90))`, runs on H100, validates against NumPy reference + captures timing. | 1–2 days | `tests/unit/test_gpu_target.py` (new) |

## 3. Critical Path & Blockers

```
        ┌─ G1-2 (cuLaunchKernel) ──────┐
        │                              ├─► G1-1 (Python dispatcher) ─┐
        ├─ G1-3 (compile_bundle metadata) ─┘                          │
        │                                                              │
        └─ G1-4 (NVRTC→cubin) ─► G1-5 (H100 validation) ──────────────┤
                                                                       │
                                  G1-7 (dialect cleanup) ──► G1-6 ─────┤
                                                                       │
                                                              G1-8 ────┘
                                                              (E2E)
```

**Earliest real-hardware launch:** G1-2 + G1-4 + G1-5 in serial = **4–6 days**.

**Single-threaded critical path to first-cubin-on-H100:** G1-2 (1d) → G1-4
(2-3d) → G1-5 (1-2d) = **4–6 days**. G1-1 / G1-3 can run in parallel and
land by day 2–3.

**Hardware blocker:** G1-5, G1-6, G1-8 require a real H100. None of
G1-2/G1-3/G1-4/G1-7 do — they can land first on a CUDA-only-no-H100 dev
box (NVRTC + driver API are sufficient for the build steps).

## 4. Out of Scope for G1's Children

These are deliberately deferred to later G-phase / cross-phase tickets:

- Multi-GEMM fusion (`matmul → softmax → matmul`) — needs Schedule IR scheduler integration
- TMA coordinate transform for non-unit-stride matrices (`cuTensorMapEncode`)
- Warp-specialization (compute/MMA vs. TMA load imbalance) — Phase 4 work
- FA-4 KV-cache eviction — sparse TMA descriptors + cooperative shared-memory mgmt
- Heterogeneous tile sizes (`128×64×32` mixed with `64×64×16`) — dynamic tile IR
- FP8/BF8 quantization during WGMMA — custom tensor element types
- Persistent thread blocks for streaming K-slab input
- Multi-GPU NCCL collectives — Phase 4
- Autotuning kernel hyperparameters (block/grid, pipeline stages) — Phase 5
- All of FA-4 forward/backward verification on H100 — folds into G5/G6 once G1 lands

## Cross-references

- `docs/audit/execution_roadmap.md` Phase G — the umbrella tracking item
- `docs/audit/kv_cache_coverage_matrix.md` — KV-cache will follow the same
  pattern once GEMM lands
- `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` — backend tree
- `src/runtime/src/backend/cuda_backend.cpp` — runtime ABI gap
- `python/tessera/compiler/jit.py:288` — where the GPU path falls through
  to `fallback_eager`
- `python/tessera/runtime.py:1070+` — where the GPU launch dispatcher needs
  a new branch

## Summary

Tessera's NVIDIA path is **IR-ready** (Phases 1–3 compile successfully) but
**runtime-incomplete**. The gap is a small set of well-defined dispatch +
ABI plumbing tasks (G1-1 through G1-8). Critical path to "first BF16 GEMM
on H100" is **4–6 days** of focused work, of which only the validation
steps (G1-5, G1-6, G1-8) require a real H100 — the rest can land on a
CUDA-only dev box.
