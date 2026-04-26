---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.


# Tessera Programming Guide  
## Chapter 9: Libraries & Primitives (Updated)

Tessera provides a library of **primitives and standard operations** that form the building blocks for high-performance applications. Unlike traditional libraries that exist outside the compiler, Tessera integrates current primitives directly into the IR stack. Autodiff-aware primitive lowering is Phase 5 planned.

---

### 9.1 Principles

- **First-class IR nodes**: primitives live in Graph IR and lower through Schedule IR → Tile IR → Target IR.  
- **Autodiff-ready contracts**: current primitives expose stable forward semantics; generated backward definitions are Phase 5 planned.  
- **Composable**: primitives fuse with surrounding code (e.g., matmul + bias + norm).  
- **Distributed by construction**: work with `ShardSpec`, domains, and distributions automatically.  
- **Privilege-safe**: region privileges enforce correct use (read/write/reduce).  

---

### 9.2 Dense Linear Algebra

- **Matrix Multiplication**: `gemm`, `gemv`, `batched_gemm`.  
- **Factorizations**: `lu`, `qr`, `cholesky`.  
- **Eigen/SVD solvers**: `eig`, `svd`.  

#### Example: Distributed GEMM
```python
W = tessera.array.from_domain(
    tessera.domain.Rect((8192, 8192)),
    dtype="bf16",
    distribution=tessera.dist.Block(mesh_axes=("tp",)),
)
X = tessera.array.from_domain(
    tessera.domain.Rect((B, 8192)),
    dtype="bf16",
    distribution=tessera.dist.Block(mesh_axes=("dp",)),
)

Y = tessera.ops.gemm(X, W)
```

---

### 9.3 Deep Learning Primitives

- **Convolutions**: direct, Winograd, FFT-based.  
- **Pooling**: max, average.  
- **Activations**: relu, gelu_safe, swish.  
- **Normalizations**: layernorm_safe, rmsnorm_safe.  
- **RNN Primitives**: lstm_cell, gru_cell.  
- **Attention**: flash_attention (v1, v2).  

#### Example: Transformer Block
```python
@tessera.jit
def block(x, Wqkv, Wo):
    h = tessera.ops.rmsnorm_safe(x)
    qkv = tessera.ops.gemm(h, Wqkv)
    q, k, v = tessera.ops.split_qkv(qkv, heads=16)
    y = tessera.ops.flash_attn(q, k, v, causal=True)
    return tessera.ops.gemm(y, Wo)
```

Autodiff for this block is Phase 5 planned; distributed collectives inside the TP path are Phase 4 planned.

---

### 9.4 Spectral Library

- **FFT/IFFT**: 1D, 2D, 3D, batched.  
- **DCT / IDCT**.  
- **Wavelets**.  

#### Example: Distributed 2D FFT
```python
D = tessera.domain.Rect((H,W))
dist = tessera.dist.Block(mesh_axes=("dp","tp"))
X = tessera.array.from_domain(D, dtype="fp32", distribution=dist)

Y = tessera.ops.fft2d(X)
```

Distributed all-to-all lowering for FFTs is Phase 4 planned.

---

### 9.5 Sparse & Graph Primitives

- **Sparse formats**: CSR, COO, BSR.  
- **SpMV, SpMM, SpGEMM**.  
- **Graph neural net operators**: scatter, gather, message_passing.  

#### Example: Sparse MatVec
```python
S = tessera.domain.SparseCSR(rows=N, nnz=nnz, distribution=dist)
A = tessera.array.from_domain(S, dtype="fp16")
x = tessera.array((N,), dtype="fp16")

y = tessera.ops.spmv(A, x)
```

---

### 9.6 Random Number Generation

- **Distributions**: uniform, normal, lognormal, Poisson.  
- **Generators**: Philox, XORWOW, Sobol.  
- **Mesh-aware**: reproducible across distributed meshes.  

```python
eps = tessera.ops.randn((B, D), generator=tessera.ops.philox(seed=42))
```

---

### 9.7 Low-Level Building Blocks

- **Reductions**: block reduce, warp scan, segmented reduce.  
- **Broadcast, transpose, tile shuffle**.  
- **Collectives**: allreduce, all_gather, reduce_scatter, all_to_all.  

#### Example: Allreduce Gradient
```python
grad = compute_gradients(...)
# Phase 4 planned distributed collective lowering
grad = tessera.ops.allreduce(grad, axis="dp")
```

---

### 9.8 Privileges and Safety in Primitives

Every primitive uses **region privileges**:

```python
@tessera.jit
def step(X: tessera.Region["read"], W: tessera.Region["read"], G: tessera.Region["reduce_sum"]):
    G[:] += tessera.ops.gemm(X, W)
```

- **Read regions**: inputs to GEMM.  
- **Reduce regions**: accumulate gradients.  
- Compiler guarantees legality and overlap.

---

### 9.9 NVL72 and Distributed Libraries

NVL72 library scaling is Phase 4 planned:

- **Dense GEMM**: TP-sharded with reduce_scatter/all_gather over NVSwitch.  
- **FFT**: decomposed into local FFTs + all-to-all.  
- **FlashAttention**: fused with collectives for latency hiding.  
- **SpMM**: partitions CSR/COO rows across DP and TP axes.  
- **RNG**: reproducible streams across 72 ranks.

---

### 9.10 Summary

- Tessera libraries cover **dense, deep learning, spectral, sparse, RNG, and collectives**.  
- Current primitives are **privilege-safe** and designed for distributed/autodiff lowering.  
- Programmers use them as **drop-in ops** that fuse into kernels.  
- NVL72 lowering to optimized NCCL/Tile IR collectives is Phase 4 planned.  

This makes Tessera’s primitives **both high-level and high-performance**, bridging the gap between productivity and peak utilization.
