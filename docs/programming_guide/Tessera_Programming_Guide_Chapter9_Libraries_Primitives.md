# Tessera Programming Guide  
## Chapter 9: Libraries & Primitives (Updated)

Tessera provides a comprehensive library of **primitives and standard operations** that form the building blocks for high-performance applications. Unlike traditional libraries that exist outside the compiler, Tessera integrates primitives directly into the IR stack, making them **first-class, composable, and autodiff-aware**.

---

### 9.1 Principles

- **First-class IR nodes**: primitives live in Graph IR and lower through Schedule IR → Tile IR → Target IR.  
- **Autodiff-aware**: every primitive has forward and backward definitions.  
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
W = dist.tensor(shape=(8192,8192),
    layout=ShardSpec(partition=("col",), mesh_axes=("tp",)),
    mesh=mesh, dtype="bf16 @accum(fp32)")

X = dist.tensor(shape=(B,8192),
    layout=ShardSpec(partition=("row",), mesh_axes=("dp",)),
    mesh=mesh, dtype="bf16")

Y = gemm(X, W)   # lowers to TP-sharded GEMM with all_gather/reduce_scatter
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
@jit @autodiff
def block(x, Wqkv, Wo):
    h = rmsnorm_safe(x)
    qkv = gemm(h, Wqkv)                  # TP-sharded GEMM
    q,k,v = split_qkv(qkv, heads=16)
    y = flash_attention(q, k, v)         # distributed, safe softmax
    return gemm(y, Wo)
```

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

Y = fft2d(X)    # automatically wires up all-to-all across TP axis
```

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

y = spmv(A, x)
```

---

### 9.6 Random Number Generation

- **Distributions**: uniform, normal, lognormal, Poisson.  
- **Generators**: Philox, XORWOW, Sobol.  
- **Mesh-aware**: reproducible across distributed meshes.  

```python
eps = randn((B,D), generator=philox(seed=42), mesh=mesh)
```

---

### 9.7 Low-Level Building Blocks

- **Reductions**: block reduce, warp scan, segmented reduce.  
- **Broadcast, transpose, tile shuffle**.  
- **Collectives**: allreduce, all_gather, reduce_scatter, all_to_all.  

#### Example: Allreduce Gradient
```python
grad = compute_gradients(...)
grad = allreduce(grad, axis="dp")   # DP gradient sync
```

---

### 9.8 Privileges and Safety in Primitives

Every primitive uses **region privileges**:

```python
@jit
def step(X: Region[read], W: Region[read], G: Region[reduce_sum]):
    G[:] += gemm(X, W)
```

- **Read regions**: inputs to GEMM.  
- **Reduce regions**: accumulate gradients.  
- Compiler guarantees legality and overlap.

---

### 9.9 NVL72 and Distributed Libraries

On NVL72, primitives scale automatically:  

- **Dense GEMM**: TP-sharded with reduce_scatter/all_gather over NVSwitch.  
- **FFT**: decomposed into local FFTs + all-to-all.  
- **FlashAttention**: fused with collectives for latency hiding.  
- **SpMM**: partitions CSR/COO rows across DP and TP axes.  
- **RNG**: reproducible streams across 72 ranks.

---

### 9.10 Summary

- Tessera libraries cover **dense, deep learning, spectral, sparse, RNG, and collectives**.  
- All primitives are **autodiff-aware, privilege-safe, and distributed by construction**.  
- Programmers use them as **drop-in ops** that fuse into kernels.  
- On NVL72, Tessera automatically lowers them to optimized NCCL/Tile IR collectives.  

This makes Tessera’s primitives **both high-level and high-performance**, bridging the gap between productivity and peak utilization.
