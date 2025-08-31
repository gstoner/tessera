# Tessera Programming Guide  
## Chapter 6: Numerics Model (Updated)

Numerical precision is central to both performance and stability on modern accelerators. Tessera makes **numerics first-class** by encoding precision and accumulation policies directly in the type system. This ensures correctness in single-GPU kernels and scalability across distributed meshes like NVL72.

---

### 6.1 Supported Data Types

Tessera supports a wide range of datatypes, aligned with NVIDIA Blackwell hardware:

- **FP4 (e2m1, e3m0)**: ultra-low precision, high throughput for LLM inference.  
- **FP6 (e3m2, e2m3)**: balance between FP4 and FP8 for mixed-precision training.  
- **FP8 (e4m3, e5m2)**: efficient for training and inference with Transformer Engine scaling.  
- **FP16 / BF16**: half-precision formats for training stability.  
- **TF32**: NVIDIA tensor core accelerated format for FP32 workloads.  
- **FP32**: full precision float, default accumulation type.  
- **INT8 / INT4**: quantized integer types for inference and embedding tables.  

---

### 6.2 Numerics as Types

Each tensor in Tessera carries a **numerical policy** in its type:

```python
x: Tensor["B","D", fp8_e4m3 @accum(fp32) @stochastic_round]
```

- **Storage dtype**: `fp8_e4m3`  
- **Accumulation policy**: `fp32`  
- **Rounding policy**: stochastic rounding  
- **Optional loss scaling**: `@loss_scale(2.0)`

---

### 6.3 Safe Numerical Primitives

Tessera provides **safe versions** of numerically sensitive operations:

- `softmax_safe` → stable softmax with log-sum-exp trick.  
- `layernorm_safe`, `rmsnorm_safe` → prevent overflow/underflow in normalization.  
- `logsumexp_safe` → robust to large exponents.  

Example:
```python
y = softmax_safe(x)      # guaranteed not to overflow
```

---

### 6.4 Mixed Precision Policies

Programmers can mix precisions declaratively:

```python
@jit
def proj(x: Tensor["B","D", fp8_e4m3 @accum(fp32)],
         W: Tensor["D","K", bf16 @accum(fp32)]):
    return gemm(x, W)
```

Here:
- **Inputs** in FP8/bf16.  
- **Accumulation** in FP32.  
- **Output** cast back to storage type per policy.

---

### 6.5 Distributed Numerics on NVL72

On NVL72, mixed precision must remain stable across 72 GPUs. Tessera enforces consistent policies across shards:

```python
W = dist.tensor(
    shape=(8192,8192),
    layout=ShardSpec(partition=("col",), mesh_axes=("tp",)),
    mesh=mesh,
    dtype="fp6 @accum(fp32)"
)
```

All TP shards use the same accumulation rules. Tessera ensures reductions (`reduce_scatter`, `all_gather`) are performed at accumulation precision before casting.

---

### 6.6 Region Privileges and Reductions

Numerical reductions are tracked via **region privileges**:

```python
@jit
def update_grad(A: Region[read], B: Region[read], G: Region[reduce_sum]):
    G[:] += gemm(A, B)   # reductions in FP32 before casting
```

Privileges guarantee safe accumulation even in distributed settings.

---

### 6.7 Autodiff and Numerics

Tessera’s autodiff respects numeric policies:

- Gradients accumulate in FP32 by default.  
- Loss scaling policies propagate through backward passes.  
- Collectives in AD (e.g., gradient allreduce) use accumulation precision.  

---

### 6.8 Example: Distributed FP6 GEMM on NVL72

```python
mesh = dist.mesh(devices=[f"cuda:{i}" for i in range(72)], axes=("dp","tp"), shape=(8,9))

X = dist.tensor(shape=(B, D),
    layout=ShardSpec(partition=("row",), mesh_axes=("dp",)),
    mesh=mesh, dtype="fp6 @accum(fp32)")

W = dist.tensor(shape=(D, K),
    layout=ShardSpec(partition=("col",), mesh_axes=("tp",)),
    mesh=mesh, dtype="fp6 @accum(fp32)")

Y = gemm(X, W)   # FP6 storage, FP32 accum, distributed across dp/tp
```

---

### 6.9 Debugging Numerical Issues

Programmers can enable numeric debug tools:

- **Overflow/underflow detection**: runtime flags when FP4/FP6 values exceed representable range.  
- **NaN/Inf sentinels**: insert checks for invalid values.  
- **Numerics verification decorators**:  
```python
@verify.numerics(stable=True, bounded=[-100,100])
def attention_scores(Q, K):
    return tile.dot(Q, K.T) / sqrt(d)
```

---

### 6.10 Summary

- Tessera supports modern datatypes: **FP4, FP6, FP8, BF16, FP16, FP32, INT8/4**.  
- **Numerics are part of the type system**, not implicit.  
- Provides **safe primitives** for stable training.  
- **Mixed precision** handled declaratively.  
- **Region privileges** ensure safe reductions.  
- On NVL72, Tessera ensures consistent policies across shards and collectives.  
- Debugging tools catch NaN/Inf and overflow errors.

This makes Tessera both **high-performance** and **numerically robust**, scaling from single GPU kernels to 72-GPU NVL72 systems.
