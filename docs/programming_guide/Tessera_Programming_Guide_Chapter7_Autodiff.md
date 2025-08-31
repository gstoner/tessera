# Tessera Programming Guide  
## Chapter 7: Autodiff (Updated)

Automatic differentiation (autodiff) is a first-class feature in Tessera. Unlike frameworks where AD is bolted on top, Tessera integrates AD into the compiler and IR stack. This enables effect-aware, distributed, and numerics-safe differentiation at every level of the programming model.

---

### 7.1 Built-In Forward and Reverse AD

Tessera provides both forward-mode and reverse-mode AD:

```python
@jit @autodiff
def f(x):
    return x * x + 2 * x

y, grad_fn = tessera.vjp(f, 3.0)
print(y, grad_fn(1.0))   # => (15.0, 8.0)
```

- **Forward-mode (JVP)** for small-output functions.  
- **Reverse-mode (VJP)** for large-output functions (deep nets).  

---

### 7.2 Custom VJP and JVP

Programmers can override differentiation rules for efficiency or stability.

```python
@tessera.custom_vjp
def gelu_safe(x): ...
def gelu_fwd(x):
    y = 0.5 * x * (1 + tanh(0.79788456*(x + 0.044715*x**3)))
    return y, y
def gelu_bwd(saved_y, bar_y):
    return bar_y * (0.5*(1+tanh(...)) + ...)
gelu_safe.defvjp(gelu_fwd, gelu_bwd)
```

Custom rules let programmers fuse backward passes or add safe numerics.

---

### 7.3 Effect-Aware Autodiff

Tessera’s AD respects **effects** such as:  

- **Randomness** (counter-based RNG).  
- **State** (optimizer updates, KV-caches).  
- **Collectives** (allreduce, reduce_scatter, all_gather, all_to_all).  

The compiler generates reverse-mode collectives automatically.

```python
def loss_fn(W):
    y = model(x, W)              # may include all_gather
    return mse(y, y_true)

gW = tessera.grad(loss_fn)(W)    # inserts reduce_scatter in backward pass
```

---

### 7.4 Region Privileges in AD

Region privileges extend naturally into autodiff:

```python
@jit
def step(X: Region[read], W: Region[read], G: Region[reduce_sum]):
    G[:] += gemm(X, W)   # backward also marked as reduce_sum
```

This ensures reductions are consistent across forward and backward passes, even in distributed execution.

---

### 7.5 AD Across Domains & Distributions

AD is distribution-aware. Gradients are computed and reduced according to the tensor’s distribution.

```python
mesh = dist.mesh(devices=[f"cuda:{i}" for i in range(72)], axes=("dp","tp"), shape=(8,9))

W = dist.tensor(shape=(8192,8192),
    layout=ShardSpec(partition=("col",), mesh_axes=("tp",)),
    mesh=mesh, dtype="fp8_e4m3 @accum(fp32)")

gW = tessera.grad(loss_fn)(W)   # reduce_scatter along "tp" axis
```

Tessera automatically lowers this to NCCL collectives across NVSwitch on NVL72.

---

### 7.6 Checkpointing and Rematerialization

To trade off memory vs compute, Tessera supports:  

- **Checkpointing**: re-compute intermediate values during backward.  
- **Rematerialization**: drop tensors and re-infer them.  

```python
@checkpoint
def transformer_block(x, W):
    ...
```

This is crucial for NVL72-scale models where memory per GPU is limited.

---

### 7.7 Debugging Autodiff

Programmers can debug gradients with:  

- **`gradcheck(fn)`** → finite difference check.  
- **NaN/Inf sentinels** during backward.  
- **IR inspection**: view lowered backward Tile IR.  

```python
tessera.gradcheck(loss_fn, W)
```

---

### 7.8 Example: Distributed Autodiff on NVL72

```python
mesh = dist.mesh(devices=[f"cuda:{i}" for i in range(72)], axes=("dp","tp","pp"), shape=(4,9,2))

@jit @autodiff
def block(x, Wqkv, Wo):
    h = rmsnorm_safe(x)
    qkv = gemm(h, Wqkv)                # TP-sharded GEMM
    q,k,v = split_qkv(qkv, heads=16)
    y = flash_attention(q, k, v)       # involves all_gather in forward
    return gemm(y, Wo)                 # reduce_scatter in backward
```

- Forward: all_gather across TP axis.  
- Backward: reduce_scatter along TP axis.  
- Compiler ensures collectives are placed correctly.  

---

### 7.9 Summary

- Tessera integrates **forward and reverse AD** into the compiler.  
- Programmers can define **custom VJP/JVP rules**.  
- AD is **effect-aware** (RNG, state, collectives).  
- **Region privileges** extend into AD for safe distributed gradients.  
- AD respects **domains/distributions**, lowering to collectives on NVL72.  
- Checkpointing and rematerialization help balance memory vs compute.  
- Debugging tools ensure gradient correctness.

With Tessera, autodiff is **built-in, distributed, and safe by design**.
