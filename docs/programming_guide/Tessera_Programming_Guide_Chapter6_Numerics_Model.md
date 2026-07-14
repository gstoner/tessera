---
status: Tutorial
classification: Tutorial
last_updated: 2026-06-11
---

> **Phase status note (updated 2026-06-11):** Phases 1–7 are complete and Phase 8 (Apple M-Series CPU via Accelerate, GPU via Metal/MPS/MPSGraph/custom MSL) is operational — on Apple Silicon this is the primary single-node execution path. Autodiff (forward/reverse transforms + activation checkpointing), ZeRO-2 optimizer sharding, the Bayesian autotuner, and the runtime Python wrapper (`tessera.runtime.TesseraRuntime`) are **shipped**. Genuinely still planned: **multi-GPU / multi-rank** execution of distributed collectives (NCCL/RCCL), `Cyclic` distribution lowering, and **NVL72** rack-scale execution (single-device collectives run over in-process mock ranks today). Canonical API names: `docs/CANONICAL_API.md`; phase table: root `CLAUDE.md`.


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
- **FP32**: full precision float, default accumulation type.  
- **INT8 / INT4**: quantized integer types for inference and embedding tables.  

The **canonical storage-dtype set** (15 names, enforced by
`tessera.dtype.canonicalize_dtype`) is: `fp64`, `fp32`, `fp16`, `bf16`,
`fp8_e4m3`, `fp8_e5m2`, `fp6_e2m3`, `fp6_e3m2`, `fp4_e2m1`, `nvfp4`,
`int8`, `int16`, `int32`, `int64`, `bool`. Aliases (`f32`, `bfloat16`,
`half`, `i8`, …) normalize to these. A further planned/gated set
(`uint*`, `complex*`, packed `int4`, AMD `mxfp*`) requires
`allow_planned_gated=True`.

> **TF32 is not a storage dtype.** `canonicalize_dtype("tf32")` is
> rejected; TF32 is modeled as a **compute mode** —
> `numeric_policy.math_mode="tf32"` on `fp32` storage — never as a tensor
> dtype. See [`docs/reference/tessera_tensor_attributes.md`](../reference/tessera_tensor_attributes.md).

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
- `rmsnorm_safe` → prevents overflow/underflow in normalization (companion to `layer_norm`).  
- `logsumexp` → robust to large exponents.  

Example:
```python
y = softmax_safe(x)      # guaranteed not to overflow
```

---

### 6.4 Mixed Precision Policies

Programmers can mix precisions declaratively:

```python
@tessera.jit
def proj(x: tessera.Tensor["B","D", fp8_e4m3 @accum(fp32)],
         W: tessera.Tensor["D","K", bf16 @accum(fp32)]):
    return tessera.ops.gemm(x, W)
```

Here:
- **Inputs** in FP8/bf16.  
- **Accumulation** in FP32.  
- **Output** cast back to storage type per policy.

---

### 6.5 Distributed Numerics on NVL72

On NVL72, mixed precision must remain stable across 72 GPUs. Tessera enforces consistent policies across shards:

```python
W = tessera.array.from_domain(
    tessera.domain.Rect((8192, 8192)),
    dtype="fp6 @accum(fp32)",
    distribution=tessera.dist.Block(mesh_axes=("tp",)),
)
```

All TP shards use the same accumulation rules. Phase 4 distributed lowering ensures reductions (`reduce_scatter`, `all_gather`) are performed at accumulation precision before casting.

---

### 6.6 Region Privileges and Reductions

Numerical reductions are tracked via **region privileges**:

```python
@tessera.jit
def update_grad(A: tessera.Region["read"], B: tessera.Region["read"], G: tessera.Region["reduce_sum"]):
    G[:] += tessera.ops.gemm(A, B)   # reductions in FP32 before casting
```

Privileges guarantee safe accumulation even in distributed settings.

---

### 6.7 Autodiff and Numerics

Autodiff is **shipped** (tape-based forward/reverse; see Ch.7). Its numeric contract is:

- Gradients accumulate in FP32 by default.  
- Loss scaling policies (`tessera.autodiff.GradScaler`) propagate through backward passes.  
- Collectives in AD (e.g., gradient allreduce) use accumulation precision; their adjoints are registered (multi-GPU execution remains Phase 4).  

---

### 6.8 Future Example: Distributed FP6 GEMM on NVL72

NVL72 execution is Phase 4 planned.

```python
X = tessera.array.from_domain(
    tessera.domain.Rect((B, D)),
    dtype="fp6 @accum(fp32)",
    distribution=tessera.dist.Block(mesh_axes=("dp",)),
)
W = tessera.array.from_domain(
    tessera.domain.Rect((D, K)),
    dtype="fp6 @accum(fp32)",
    distribution=tessera.dist.Block(mesh_axes=("tp",)),
)

Y = tessera.ops.gemm(X, W)   # FP6 storage, FP32 accumulation
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
