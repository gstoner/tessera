# Tessera Programming Guide
# Chapter 6. Numerical Stability and Determinism

---

## 6.1 Overview

Tessera enforces **numerical stability** and **determinism** across all executions.  
Whereas CUDA, PyTorch, or JAX may produce nondeterministic results due to race conditions, atomics, or floating-point order, Tessera guarantees **bitwise reproducibility** across runs, devices, and cluster scales.  

This chapter explains:  
- Why determinism is critical in ML and HPC.  
- Stability strategies for reductions, softmax, and mixed precision.  
- Deterministic collectives.  
- Worked examples showing Tessera’s guarantees in practice.  

---

## 6.2 Why Determinism Matters

In ML and scientific computing:  
- **Reproducibility**: Debugging requires identical runs.  
- **Validation**: Results must match across hardware.  
- **Research Integrity**: Small nondeterministic changes can alter published findings.  
- **Distributed Training**: Nondeterministic reductions can introduce silent divergence.  

CUDA/NCCL often exhibit nondeterminism:  
- Floating-point sums depend on thread/block order.  
- All-reduce order is not fixed.  
- Atomic operations yield inconsistent results across runs.  

Tessera eliminates these issues.  

---

## 6.3 Deterministic Reductions

Reductions (sum, mean, norm) are executed with deterministic accumulation.  

Strategies:  
- **Pairwise Summation**: Balanced binary tree ordering.  
- **Kahan Summation**: Corrects floating-point error accumulation.  
- **Stable Mixed Precision**: Accumulate FP16/BF16 into FP32 or FP64.  

Example:
```python
X = dist.tensor((1_000_000,), layout=dist.ShardSpec(partition=("row",), mesh_axes=("dp",)), mesh=mesh)
Y = op.reduce_sum(X, stable=True)
```

This guarantees identical results across runs and mesh sizes.  

---

## 6.4 Stable Softmax

Softmax is a common source of instability.  
Tessera applies **max-subtraction** by default:  

```python
A = op.softmax(X)          # Uses stable algorithm
B = op.softmax(X, stable=False)  # Explicitly disable stability policy
```

Formula:  
```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

This avoids overflow for large values and underflow for small ones.  

---

## 6.5 Deterministic Collectives

Collectives are first-class operators in Tessera.  
They execute in **fixed order**, ensuring reproducibility.  

- **All-reduce**: Balanced binary tree.  
- **Reduce-scatter**: Deterministic partition.  
- **All-gather**: Fixed concatenation order.  

Example:
```python
grads = op.all_reduce(local_grads, op="sum", axis="dp")
```

Even across 72 GPUs in an NVL72 cluster, Tessera ensures identical reduction results.  

---

## 6.6 Randomness

Random number generation is reproducible across the mesh:  

- **Stateless RNG**: Random values depend only on seed + indices.  
- **Mesh-consistent**: Sharded tensors produce consistent values regardless of partition.  

Example:
```python
X = op.random_normal((1024, 1024), seed=42)
```

Sharding `X` across GPUs yields identical results when reassembled.  

---

## 6.7 Worked Example: Attention

In CUDA/PyTorch, attention kernels are nondeterministic due to reductions.  
In Tessera, determinism is enforced:  

```python
def attention(Q, K, V):
    scores = op.matmul(Q, K.T)
    P = op.softmax(scores)   # stable softmax with deterministic reduction
    return op.matmul(P, V)
```

Guarantees:  
- Identical results across runs.  
- Identical results across different mesh partitions.  
- No race-condition noise in gradients.  

---

## 6.8 Worked Example: PINN with PDEs

In Physics-Informed Neural Networks (PINNs), nondeterministic gradients can cause divergence.  
Tessera’s deterministic adjoints guarantee stable PDE solves:  

```python
# Laplacian operator with deterministic adjoint
residual = op.laplacian(phi) - f
loss = op.reduce_sum(residual**2, stable=True)
```

Unlike CUDA atomics, residual aggregation is reproducible.  

---

## 6.9 Comparison

| Framework  | Deterministic? | Stable Reductions | RNG Reproducible? |
|------------|----------------|-------------------|-------------------|
| CUDA       | No             | No                | No |
| PyTorch    | Partial        | Optional           | Partial |
| JAX        | Partial        | Optional           | Yes |
| Tessera    | Yes (bitwise)  | Yes (default)     | Yes (mesh-consistent) |

---

## 6.10 Summary

- Tessera guarantees **bitwise reproducibility** across runs.  
- Reductions and softmax are stable by default.  
- Collectives execute in deterministic order.  
- RNG is stateless and mesh-consistent.  
- Determinism enables debugging, reproducibility, and reliable training of large models.  

