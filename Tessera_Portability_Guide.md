# Tessera Portability Guide
*(How Tessera maps across backends: NVIDIA PTX, AMD ROCm, Intel Level Zero, CPU fallback — what is guaranteed cross‑platform vs vendor‑specific)*

**Status:** v1.0 (Informative + Normative sections marked accordingly)

---

## 1. Overview (Informative)

Tessera targets multiple execution backends while keeping a single programming model:
- **NVIDIA PTX** (CUDA/NVVM → PTX/SASS)
- **AMD ROCm** (ROCm LLVM → GCN/CDNA ISA)
- **Intel Level Zero** (L0 driver; DPAS/XMX where available)
- **CPU** fallback (LLVM CPU/JIT + vectorized libraries)

This guide explains **what is portable vs. vendor‑specific**, how the **multi‑level IRs** map to each backend, and how to **write portable, high‑performance code** using Tessera’s abstractions.

---

## 2. Guarantees & Non‑Goals (Normative)

**Tessera guarantees**:
- **Correctness** across all backends for Graph IR semantics (same results within documented numerical profiles).
- **Determinism options** via numerics profiles (`deterministic`, `strict`) with fixed reduction order.
- **Functional portability** of Schedule IR constructs (tiling, fusion, pipelining) and Tile IR ops (mma‑like, ldmatrix‑like) with **emulation fallbacks** when a native intrinsic is unavailable.
- **Distributed collectives** at the Graph IR level (`all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`) with backend‑specific libraries underneath.

**Tessera does not guarantee**:
- **Equal performance** across vendors or architectures.
- Availability of **vendor‑specific dtypes** (e.g., FP8 variants, TF32) or **async copy instructions** on all targets.

---

## 3. Backend Support Matrix (Informative)

| Capability                     | NVIDIA PTX          | AMD ROCm            | Intel Level Zero        | CPU Fallback            |
|-------------------------------|---------------------|---------------------|-------------------------|-------------------------|
| Vector ALUs (fp32/int32)      | ✓                   | ✓                   | ✓                       | ✓ (SIMD)                |
| BF16                           | ✓ (native)          | ✓ (native)          | ✓ (varies by arch)      | △ (emulated or AVX512)  |
| FP16                           | ✓                   | ✓                   | ✓                       | △ (emulated)            |
| TF32                           | ✓ (Ampere+)         | —                   | —                       | —                       |
| FP8 (E4M3/E5M2)               | ✓ (Hopper+)         | △ (availability varies)| △ (availability varies) | —                       |
| Tensor Core / MMA             | ✓ (`mma.sync`)      | ✓ (MFMA/WMMA)       | ✓ (DPAS/XMX)            | △ (tiled GEMM kernels)  |
| `ldmatrix`-class warp loads   | ✓                   | △ (LDS + DS ops)    | △ (SLM block loads)     | —                       |
| Async global→shared copy      | ✓ (`cp.async`)      | △ (emulation/DS)    | △ (block load hints)    | —                       |
| Warp size                     | 32                  | 64 (wavefront)      | 16/32 (subgroups)       | N/A                     |
| Shared/LDS/SLM memory         | ✓ (Shared)          | ✓ (LDS)             | ✓ (SLM)                 | —                       |
| Collectives lib               | NCCL                | RCCL                | oneCCL                  | MPI/UCX (fallback)      |

**Legend:** ✓ native, △ partial/emulated/varies, — not available.

---

## 4. IR → Backend Mapping (Informative)

### 4.1 Tile IR: Matrix Multiply‑Accumulate
- **NVIDIA**: `tessera.tile.mma.sync` → `mma.sync.aligned.m...n...k...` (Tensor Cores)
- **AMD**:    `tessera.tile.mma.sync` → MFMA/WMMA intrinsics (CDNA/GFX)
- **Intel**:  `tessera.tile.mma.sync` → DPAS/XMX intrinsics (Level Zero)
- **CPU**:    lowered to blocked/vectorized GEMM (e.g., `llvm.matrix.*`, BLAS kernels)

If a native intrinsic is unavailable, Tessera **emulates** using vector FMAs.

### 4.2 Tile IR: Warp/Block Loads
- **NVIDIA**: `ldmatrix.sync.aligned` from shared memory.
- **AMD**:    LDS DS‑read patterns + swizzles (bank‑conflict‑aware).
- **Intel**:  SLM block loads / subgroup block reads.
- **CPU**:    normal vector loads; no special warp op.

### 4.3 Tile IR: Async Copy
- **NVIDIA**: `cp.async` (global→shared) + barriers.
- **Others**: Emulated pipelining with prefetch + dual buffers; HW async used where present.

### 4.4 Graph IR Collectives
Resolved to:
- **NVIDIA** → NCCL
- **AMD** → RCCL
- **Intel** → oneCCL
- **CPU** → MPI/UCX (or local threaded reductions)

Tessera enforces **deterministic reduction trees** when requested.

---

## 5. Data Types & Numerics (Informative + Normative notes)

| DType        | Portability Notes                                                                 |
|--------------|------------------------------------------------------------------------------------|
| `fp32`       | Universal baseline; deterministic under `strict`.                                  |
| `bf16`       | Native on NVIDIA/AMD; Intel varies; CPU may emulate; math is portable.             |
| `fp16`       | GPU‑native; CPU emulation likely; use fp32 accum in GEMM for stability.            |
| `tf32`       | **NVIDIA‑only**; Tessera auto‑casts to `fp32` on other backends.                   |
| `fp8_e4m3`   | **Vendor‑specific**; fallback to bf16/fp16 with range scaling on unsupported HW.   |
| `fp8_e5m2`   | Same as above.                                                                     |

**Normative:** When a dtype is unsupported, the compiler **shall** select a documented fallback with equal or higher precision. Accumulation precision defaults to **fp32** for mixed‑precision GEMM/attention across all backends.

---

## 6. Memory Model Mapping (Informative)

- **Shared memory** ↔ **LDS** (AMD) ↔ **SLM** (Intel). Use Tile IR allocs; avoid vendor‑specific bank assumptions.
- **Warp size differences**: write **warp‑agnostic** code; use Tessera’s `tile.group` and reductions instead of hard‑coding 32/64.
- **Barriers/Fences** map to PTX `bar.sync`/`membar.*`, AMD LDS/GL barriers, Intel subgroup/workgroup barriers.
- Tessera’s **formal memory model** (shared visibility, fences, atomics orders/scopes) applies uniformly; backends must meet or exceed its guarantees.

---

## 7. Distributed Topologies (Informative)

Tessera `dist.Mesh` abstracts multi‑GPU:
- **NVIDIA**: NVLink/NVSwitch + NCCL
- **AMD**: XGMI/Infinity Fabric + RCCL
- **Intel**: Fabric + oneCCL
- **CPU**: sockets + MPI/UCX

**Best practice**: structure meshes as (TP × PP × DP) sub‑axes; let the autotuner overlap comm/compute.

---

## 8. Capability Detection & Specialization (Normative API)

Query backend and features at runtime:
```python
from tessera import runtime

caps = runtime.capabilities()
# caps example:
# { "backend":"ptx", "tensorcore":{"fp16":True,"bf16":True,"fp8_e4m3":True},
#   "warp_size":32, "cp_async":True, "shared_bytes_per_block": 227_328 }

# Specialize code paths safely
if caps["tensorcore"].get("fp8_e4m3", False):
    Y = op.flash_attention_fp8(Q, K, V, scale="amax_dynamic")
else:
    Y = op.flash_attention(Q, K, V)  # bf16/fp16 with fp32 accum
```

Compile‑time hinting:
```python
@op.specialize(when={"backend":"rocm"}, use={"tile_sizes":[128,128,32]})
def my_gemm(A,B): return op.matmul(A,B)
```

---

## 9. Performance Portability (Informative)

- **Use the autotuner** (cost models + on‑device measurements). Persist caches per `(arch, shape, dtype)`.
- **Parametrize tiles**; avoid hard‑coding MMA tile shapes.
- **Prefer fused ops** (`flash_attention`, fused optimizers) — Tessera lowers to the best pattern per backend.
- **Keep math stable** (fp32 accum, stable softmax) to avoid backend drift.

---

## 10. Vendor‑Specific Features (Informative)

- **NVIDIA**: TF32, FP8 (Hopper+), `cp.async`, `ldmatrix`.  
  Tessera exposes them via **capability‑guarded** APIs and **automatic fallback**.

- **AMD**: MFMA tile shapes, fast LDS swizzles, XGMI collectives.  
  Tessera schedules tiles to match MFMA footprints; LDS patterns use DS ops behind Tile IR.

- **Intel**: DPAS/XMX intrinsics, SLM.  
  Tessera emits subgroup‑aware tiles and uses Level Zero interfaces to select DPAS shapes.

---

## 11. Building & Selecting Backends (Informative)

Install extras:
```bash
# NVIDIA
pip install tessera[gpu-cuda]

# AMD ROCm
pip install tessera[gpu-rocm]

# Intel Level Zero
pip install tessera[gpu-levelzero]

# CPU only
pip install tessera[cpu]
```

Select backend at runtime:
```python
from tessera import runtime
runtime.select_backend("ptx")        # "rocm" | "level_zero" | "cpu"
```

Environment override:
```bash
export TESSERA_BACKEND=ptx
```

---

## 12. Testing & Validation (Informative)

- **Golden outputs** with `numerics.profile("strict")` across all backends.
- **Kernel fuzzing** for edge shapes/dtypes.
- **Tolerance sweeps** for mixed precision (BF16/FP16/FP8) vs. FP32 reference.
- **Distributed soak tests** (NCCL/RCCL/oneCCL) with forced topology changes.

---

## 13. Portability Checklist (Actionable)

- [ ] Avoid hard‑coded warp size; use `tile.group` reductions.  
- [ ] Query capabilities; guard vendor‑specific fast paths.  
- [ ] Parametrize tile sizes; rely on autotuner; cache results.  
- [ ] Use stable numerics (fp32 accum, stable softmax).  
- [ ] Keep collectives at Graph IR; let Tessera schedule overlap.  
- [ ] Validate under `deterministic` and `strict` profiles.  
- [ ] Ensure memory model correctness: barriers/fences per spec.  

---

## 14. Known Limitations (Informative)

- Some FP8/TF32 paths are **NVIDIA‑only**; automatic bf16/fp16 fallbacks may be slower.  
- Async copy and `ldmatrix` patterns may be emulated on non‑NVIDIA backends.  
- Subgroup sizes differ (32 vs 64 vs 16/32); avoid subgroup‑fixed algorithms.  
- CPU fallback is **correctness‑first**; performance depends on BLAS/LLVM vectorization.

---

## 15. Example: Portable FlashAttention (Informative)

```python
from tessera import runtime, op

caps = runtime.capabilities()
if caps["tensorcore"].get("fp8_e4m3", False):
    # NVIDIA Hopper path (or other FP8‑capable)
    Y = op.flash_attention_fp8(Q, K, V, scale="amax_dynamic")
else:
    # Portable path (bf16/fp16 → fp32 accum)
    Y = op.flash_attention(Q, K, V, causal=True)
```

---

## 16. Appendix A — Backend Intrinsic Mapping (Informative)

| Tile IR op             | NVIDIA PTX             | AMD ROCm (GCN/CDNA)    | Intel Level Zero         | CPU LLVM             |
|------------------------|------------------------|------------------------|--------------------------|---------------------|
| `tile.mma.sync`        | `mma.sync.aligned`     | `mfma.*`/WMMA          | `dpas.*`                 | `llvm.matrix.*`     |
| `tile.ldmatrix`        | `ldmatrix.sync`        | DS‑read swizzled       | SLM block read           | vector loads        |
| `tile.cp.async`        | `cp.async`             | emulated prefetch      | block load + barriers    | N/A                 |
| `tile.barrier`         | `bar.sync`             | `s_barrier`/LDS fence  | `workgroupBarrier`       | thread barrier      |

---

## 17. Change History
- v1.0 — Initial publication.
