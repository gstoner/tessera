# Tessera Programming Guide
# Chapter 5. Operators and Numerical Model

---

## 5.1 Overview

Operators are the fundamental building blocks in Tessera.  
They represent algebraic computations such as matmul, FFT, convolution, softmax, PDE solvers, or custom-defined kernels.  

Unlike CUDA kernels or PyTorch ops, Tessera operators are **algebraic objects** with explicit:  
- **Definitions**: Mathematical form of the operator.  
- **Adjoints**: Exact, deterministic backward rules.  
- **Rewrite Rules**: Algebraic transformations for fusion or optimization.  

This chapter explains the operator model, adjoints, numerical stability rules, and operator libraries.  

---

## 5.2 Operator Algebra

Operators in Tessera form a composable algebra:  

- **Composition**: `op3 = op2 ∘ op1`  
- **Fusion**: Adjacent operators can be algebraically combined into a more efficient form.  
- **Adjoints**: Every operator has a mathematically defined backward pass.  
- **Spectral Forms**: Operators can be expressed in alternative bases (e.g., Fourier, wavelet).  

Example:
```python
from tessera import op

# Define attention as an operator composition
def attention(Q, K, V):
    return op.matmul(op.softmax(op.matmul(Q, K.T)), V)
```

Tessera internally represents this as an algebraic graph, allowing fusions like:  
- `softmax ∘ matmul` → fused tile kernel.  
- `(Q @ K.T) @ V` → reordered for optimal tile scheduling.  

---

## 5.3 Operator Adjoints

Every operator has a deterministic adjoint rule, ensuring reproducibility and correctness.  

Example: **Matrix Multiplication**
- Forward: `C = A @ B`  
- Backward:  
  - `dA = dC @ B.T`  
  - `dB = A.T @ dC`  

Tessera encodes these rules explicitly, unlike PyTorch’s autograd which infers them dynamically.  
This makes training **bitwise deterministic**.  

```python
C = op.matmul(A, B)
dA, dB = op.adjoint(op.matmul)(A, B, dC)
```

---

## 5.4 Numerical Stability

Tessera enforces numerical stability policies globally:  

1. **Stable Summation**: Reductions use deterministic accumulation (pairwise or Kahan).  
2. **Softmax with Max-Shift**: Prevents overflow/underflow.  
3. **Mixed Precision Rules**: FP16/BF16 accumulations are promoted to FP32 when needed.  
4. **Deterministic Randomness**: RNG streams are mesh-consistent and reproducible.  

Example:
```python
A = op.softmax(X, stable=True)
```

Guarantees that the softmax uses max-subtraction for numerical stability.  

---

## 5.5 Operator Libraries

Tessera includes standard operator libraries, grouped by domain:  

- **Linear Algebra**: `matmul`, `einsum`, `cholesky`, `qr`, `svd`.  
- **Transformations**: `fft`, `ifft`, `dct`, `wavelet`.  
- **Neural Networks**: `conv2d`, `layernorm`, `softmax`, `mlp`.  
- **PDE Solvers**: `laplacian`, `grad`, `div`, `helmholtz`.  
- **Control Flow**: `if_then_else`, `map`, `reduce`, `scan`.  
- **Collectives**: `all_reduce`, `all_gather`, `scatter`, `broadcast`.  

---

## 5.6 Operator Fusion

Tessera’s compiler applies algebraic fusion rules:  

- **Attention Fusion**: `softmax(Q @ K.T) @ V` → fused FlashAttention kernel.  
- **Norm + Matmul Fusion**: `layernorm + matmul` → fused tile kernel.  
- **FFT + Convolution Fusion**: `fft ∘ pointwise_mul ∘ ifft` → spectral convolution.  

Fusion reduces HBM traffic, improves locality, and minimizes collectives.  

---

## 5.7 Spectral Operators

Tessera natively supports **spectral operator forms**, enabling efficient PDEs and advanced ML models:  

Example: Convolution in Fourier Space
```python
Xf = op.fft(X)
Yf = op.pointwise_mul(Xf, Kf)
Y  = op.ifft(Yf)
```

This decomposition avoids explicit convolution and reduces O(N²) → O(N log N).  

---

## 5.8 Comparison with CUDA / PyTorch / JAX

| Framework  | Operator Model | Adjoint Rules | Fusion | Determinism |
|------------|----------------|---------------|--------|-------------|
| CUDA       | Imperative kernels | User-coded | Manual | No |
| PyTorch    | Dynamic ops | Autograd | Limited | No |
| JAX        | Functional ops | XLA AD | Fusion via XLA | Partial |
| Tessera    | Algebraic ops | Explicit adjoints | Algebraic fusion | Yes |

---

## 5.9 Summary

- Tessera operators are **algebraic**, with explicit adjoints and rewrite rules.  
- Numerical stability is guaranteed via deterministic reductions and stable policies.  
- Operator libraries cover linear algebra, ML, PDEs, and collectives.  
- Fusion and spectral rewrites enable highly efficient execution.  
- Tessera ensures determinism and reproducibility beyond CUDA, PyTorch, or JAX.  

