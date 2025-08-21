# Tessera Hardware Mapping Guide
## Chapter 7: Appendix A — Reference Mapping Tables

Reference tables mapping Tessera IR constructs to CUDA/PTX intrinsics and distributed collectives.

---

### A.1 Overview

This appendix provides **reference tables** mapping Tessera IR constructs to **CUDA/PTX hardware intrinsics** and execution primitives.  
It serves as a quick reference for developers optimizing kernels or debugging codegen.

---

### A.2 Graph IR → Schedule IR

| Graph IR Operation     | Schedule IR Transformation     | Notes |
|------------------------|--------------------------------|-------|
| `op.matmul(A, B)`      | `schedule.tile(m,n,k)`         | Tiling parameters autotuned. |
| `op.fft(X)`            | `schedule.stage(scope="shared")`| FFT staged into shared memory. |
| `dist.all_reduce(X)`   | `schedule.collective(op="sum")` | Maps to NVLink/NCCL collective. |
| `op.recursive_mixture` | `schedule.pipeline(depth=N)`   | Recursion unrolled into pipeline stages. |

---

### A.3 Schedule IR → Tile IR

| Schedule IR Construct            | Tile IR Lowering         | CUDA/PTX Equivalent |
|----------------------------------|--------------------------|---------------------|
| `tile(m=128,n=128,k=64)`         | Warp-level fragments     | `mma.sync.aligned.m16n16k16` |
| `prefetch(scope="shared")`       | Shared mem double-buffer | `ldmatrix.sync.aligned` |
| `pipeline(double_buffer=true)`   | Async load + compute     | `cp.async`, barriers |
| `fuse(op1, op2)`                 | Inlined tile kernels     | Custom PTX fusion |

---

### A.4 Tile IR → Target IR

| Tile IR Operation     | PTX/LLVM Lowering      | Hardware Usage |
|-----------------------|------------------------|----------------|
| `mma_sync`            | `mma.sync.aligned`     | Tensor Cores |
| `ldmatrix`            | `ldmatrix.sync`        | Warp loads from shared |
| `barrier`             | `bar.sync`             | Block-wide sync |
| `shfl` (warp shuffle) | `shfl.sync`            | Intra-warp data exchange |
| `reduce` (warp sum)   | `__reduce_add_sync`    | Warp reductions |

---

### A.5 Distributed Collectives

| Tessera Collective   | NVLink/NCCL Mapping    | Notes |
|----------------------|------------------------|-------|
| `all_reduce`         | NCCL `ncclAllReduce`   | Deterministic reduction order enforced |
| `broadcast`          | NCCL `ncclBroadcast`   | Broadcast tensor shards |
| `all_gather`         | NCCL `ncclAllGather`   | Used in MoE + pipeline parallelism |
| `reduce_scatter`     | NCCL `ncclReduceScatter`| Balances comm + compute |

---

### A.6 Precision Mapping

| Tessera DType | PTX/Hardware Precision | Notes |
|---------------|-------------------------|-------|
| `fp32`        | F32 ALUs, Tensor Cores  | Full precision |
| `tf32`        | TensorFloat-32 (10-bit mantissa) | Default matmul on Ampere+ |
| `bf16`        | Brain Float 16          | Tensor Core optimized |
| `fp16`        | Half precision          | Widely supported |
| `fp8_e4m3`    | FP8 (E4M3)              | Hopper+ Tensor Core FP8 |
| `fp8_e5m2`    | FP8 (E5M2)              | Hopper+ variant |

---

### A.7 Example End-to-End Mapping

Matmul (`C = A × B`) flow:

Graph IR:      op.matmul(A,B)
↓
Schedule IR:   tile(m=128,n=128,k=64), prefetch(shared), pipeline(double_buffer)
↓
Tile IR:       warp-level mma_sync, ldmatrix loads
↓
Target IR:     PTX mma.sync.aligned.m16n16k16.row.col.f16.f16.f32
↓
Hardware:      Tensor Cores on Hopper/Blackwell SM

---

### A.8 Reference Notes

- Tessera ensures **deterministic lowering** for reproducibility.  
- Autotuning caches schedules per `(arch, shape, dtype)`.  
- Distributed collectives rely on **NVLink/NVSwitch fabrics**.  
- Debugging supported by **MLIR dumps** at each IR level.  

---

### A.9 Summary

- Tables provide a quick **IR → hardware intrinsic** mapping.  
- Useful for developers tuning performance or verifying kernel lowering.  
- Complements the detailed descriptions in Chapters 2–6.  

---
