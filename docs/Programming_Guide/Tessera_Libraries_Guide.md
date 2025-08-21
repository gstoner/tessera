# Tessera Libraries & Operator Primitives Guide

## 1. Introduction
Tessera consolidates the functionality traditionally split across CUDA libraries (cuBLAS, cuDNN, cuFFT, cuSPARSE, cuRAND, CUB, CUTLASS, CuTe) into **first-class operators**.  
- Instead of calling external libraries via C APIs, Tessera provides **graph-level operators** that lower through the IR stack:  
  - **Graph IR**: algebraic operators (`op.gemm`, `op.fft`, `op.conv`)  
  - **Schedule IR**: autotuned tiling, fusion, pipeline stages  
  - **Tile IR**: block/warp/tensor-core implementation  
  - **Target IR**: backend codegen (PTX, ROCm LLVM, Level Zero, CPU)  
- Cross-platform: semantics are consistent across vendors; performance depends on backend optimizations.

---

## 2. Linear Algebra Library (cuBLAS Equivalent)
### Capabilities
- Dense matrix operations: GEMM, GEMV, batched GEMM.  
- Factorizations: LU, QR, Cholesky.  
- Eigen/SVD solvers.  

### Tessera Operators
```python
C = op.gemm(A, B, alpha=1.0, beta=0.0, transA=False, transB=False)
Q, R = op.qr(A)
U, S, V = op.svd(A)
```

### Backend Mapping
- NVIDIA → cuBLASLt / CUTLASS  
- AMD → rocBLAS  
- Intel → oneMKL  
- CPU → OpenBLAS / LAPACK  

---

## 3. Deep Learning Primitives (cuDNN Equivalent)
### Capabilities
- Convolution: direct, Winograd, FFT-based.  
- Pooling, activations, normalization.  
- RNN primitives.  

### Tessera Operators
```python
Y = op.conv2d(X, W, stride=1, padding="same")
Y = op.batch_norm(Y, gamma, beta)
Z = op.relu(Y)
```

### Backend Mapping
- NVIDIA → cuDNN kernels  
- AMD → MIOpen  
- Intel → DNNL  

---

## 4. Spectral Library (cuFFT Equivalent)
### Capabilities
- FFT, IFFT, DCT, wavelets.  
- 1D/2D/3D, batched.  

### Tessera Operators
```python
Xf = op.fft(X, axes=[-1])
Y = op.ifft(Xf)
```

### Backend Mapping
- NVIDIA → cuFFT  
- AMD → rocFFT  
- Intel → oneMKL DFT  
- CPU → FFTW  

---

## 5. Sparse & Graph Operators (cuSPARSE Equivalent)
### Capabilities
- Sparse formats: CSR, COO, BSR.  
- SpMV, SpMM, SpGEMM.  
- Graph neural net primitives.  

### Tessera Operators
```python
Y = op.spmv(A_csr, X)
Z = op.spmm(A_csr, B)
```

### Backend Mapping
- NVIDIA → cuSPARSE  
- AMD → hipSPARSE  
- Intel → oneMKL Sparse  

---

## 6. Random Number Generation (cuRAND Equivalent)
### Capabilities
- Uniform, normal, lognormal, Poisson.  
- Generators: Philox, XORWOW, Sobol.  
- Reproducible across distributed meshes.  

### Tessera Operators
```python
rng = op.rng(seed=1234, generator="philox")
X = rng.normal(shape=(1024,), mean=0, std=1)
```

---

## 7. Primitives & Collectives (CUB / CUTLASS / CuTe Equivalent)
### Capabilities
- Block reductions, warp scans, segmented reductions.  
- CUTLASS-like tensor-core GEMMs.  
- CuTe-like tiling abstractions for portable scheduling.  

### Tessera Operators
```python
s = op.reduce(X, op="sum", axis=0)
y = op.scan(X, op="exclusive_sum", axis=0)
```

---

## 8. Cross-Library Composition
Tessera allows operators to fuse across “library boundaries”:
```python
# Spectral convolution in one graph
Xf = op.fft(X)
Yf = op.mul(Xf, Kf)
Y = op.ifft(Yf)
```
This fuses into a single kernel schedule when possible.

---

## 9. Backend Mapping Summary
| Category          | NVIDIA Backend   | AMD Backend  | Intel Backend  | CPU Backend |
|-------------------|-----------------|--------------|----------------|-------------|
| Linear Algebra    | cuBLASLt        | rocBLAS      | oneMKL         | OpenBLAS    |
| Deep Learning     | cuDNN           | MIOpen       | DNNL           | Eigen/DNNL  |
| Spectral          | cuFFT           | rocFFT       | oneMKL DFT     | FFTW        |
| Sparse/Graph      | cuSPARSE        | hipSPARSE    | oneMKL Sparse  | SuiteSparse |
| RNG               | cuRAND          | rocRAND      | oneMKL RNG     | stdlib RNG  |
| Collectives       | NCCL            | RCCL         | oneCCL         | MPI         |

---

## 10. Examples

### 10.1 Batched GEMM with Mixed Precision
```python
C = op.gemm(A, B, dtype="bf16", accumulate_dtype="fp32", batch=True)
```

### 10.2 CNN Forward
```python
Y = op.conv2d(X, W, stride=1, padding="same")
Y = op.batch_norm(Y, gamma, beta)
Z = op.relu(Y)
```

### 10.3 Spectral Convolution
```python
Xf = op.fft(X)
Yf = op.mul(Xf, Kf)
Y = op.ifft(Yf)
```

### 10.4 Sparse Graph Op
```python
Z = op.spmm(A_csr, X)
```

### 10.5 RNG for Dropout
```python
mask = rng.bernoulli(shape=X.shape, p=0.1)
Y = op.mul(X, mask)
```

---

⚡ **Key Point**: Unlike CUDA, Tessera does not split libraries — all primitives are **first-class operators** integrated into the same IR stack, making them composable and fusible.
