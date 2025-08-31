# Tessera Programming Guide  
## Chapter 2: Programming Model (Merged Final with NVL72 Examples)

The Tessera programming model is designed to make GPU programming both **productive and performant**. It replaces low-level thread math with a **tile-first abstraction**, unifies memory and distributed execution under one IR stack, and integrates numerics, autodiff, and collectives directly into the language.  

This chapter provides both **conceptual structure** and **practical examples** so programmers can immediately apply Tessera, including at NVL72 scale.

---

### 2.1 Tiles, Groups, and Meshes

At the core of Tessera is the **tile**:  
- A **tile** is a block of work mapped to hardware lanes (threads).  
- A **tile group** is a cooperating set of tiles that share memory and synchronize.  
- A **mesh** is a collection of devices (GPUs/accelerators) organized for parallelism.  

#### Example: Tile-first elementwise kernel
```tessera
@kernel
def saxpy(x: f32[n], y: mut f32[n], alpha: f32):
    i = tile.linear_id()
    if i < n:
        y[i] = alpha * x[i] + y[i]
```

#### Example: NVL72 mesh
```python
from tessera import dist

# Factorization: 4x9x2 = 72 GPUs
mesh = dist.mesh(
    devices=[f"cuda:{i}" for i in range(72)],
    axes=("dp","tp","pp"),
    shape=(4,9,2)
)
```

This mesh can drive **data parallel (dp)**, **tensor parallel (tp)**, and **pipeline parallel (pp)** dimensions across all 72 GPUs.

---

### 2.2 Memory in the Programming Model

Each tile accesses multiple memory tiers (see Ch.3):  
- **Registers**: lane-local scalars.  
- **Shared memory**: tile-group buffer (`tshared.alloc`).  
- **Global memory (HBM)**: large tensors, may be sharded across GPUs.  
- **Mesh memory**: distributed tensors via NVLink/NVSwitch.  

#### Example: Sharded weight matrix on NVL72
```python
W = dist.tensor(
    shape=(8192, 8192),
    layout=dist.ShardSpec(partition=("col",), mesh_axes=("tp",)), # sharded across tensor-parallel axis
    mesh=mesh,
    dtype="bf16"
)
```

The compiler wires up the **all_gather/reduce_scatter** collectives to move between sharded and replicated views, using NVLink/NVSwitch.

---

### 2.3 Execution Semantics

- **SPMD model**: each tile executes the same kernel function.  
- **Group sync**: `tbarrier()` makes shared memory writes visible.  
- **Pipelining**: async copies overlap memory and compute.  
- **Collectives**: synchronize across devices (`allreduce`, `reduce_scatter`).  

#### Example: Double-buffered pipeline
```tessera
@kernel.autotune(space=dict(BM=[128,256], stages=[2,3]))
def tp_axpy(X: f16[m], Y: mut f16[m], alpha: f16):
    s0 = tshared.alloc
    s1 = tshared.alloc
    for t in tile.range(m, step=128, prefetch=2):
        cp_async.shared.global(s1, X[t:t+128])
        tbarrier()
        if t > 0:
            for i in range(128):
                yv = alpha * s0[i] + tile.load(Y, index=t-128+i)
                tile.store(Y, index=t-128+i, value=yv)
        s0, s1 = s1, s0
    tbarrier()
```

---

### 2.4 Numerics as Types

Tessera types encode precision policies (see Ch.6):  
- Storage dtype (`fp8_e4m3`, `fp6`, `fp4`).  
- Compute/accumulation dtype (`fp16`, `fp32`).  
- Rounding and scaling policies.  

#### Example: Blackwell FP6 matmul with FP32 accumulation
```python
x: Tensor["B","D", fp6 @accum(fp32)]
w: Tensor["D","K", fp6 @accum(fp32)]

y = gemm(x, w)   # lowers to WGMMA + Transformer Engine configs
```

---

### 2.5 Autodiff & Effects

Autodiff is first-class:  
- Forward and reverse AD are integrated.  
- AD respects effects (randomness, collectives, state).  
- Collectives automatically generate gradient rules.  

#### Example: Gradient with TP collectives on NVL72
```python
def loss_fn(W):
    y = model(x, W)  # includes tensor-parallel GEMMs
    return mse(y, y_true)

gW = tessera.grad(loss_fn)(W)  # inserts reduce_scatter/all_gather in backward pass
```

---

### 2.6 Layouts & Data Movement

Layouts control how tensors are accessed (Ch.8):  
- **Row/col-major**, **blocked**, **swizzled**, **interleaved**.  
- Movement via `tile.load/store`, `cp_async`, `tbarrier()`.  

#### Example: GEMM with swizzled layout
```tessera
sA = tshared.alloc[f16](BM, BK, swizzle="xor")
sB = tshared.alloc[f16](BK, BN, swizzle="xor")
cp_async.shared.global(sA, A_tile)
cp_async.shared.global(sB, B_tile)
tbarrier()
C += tile.mma(sA, sB, accum=f32)
```

---

### 2.7 Parallelism & Collectives

Parallelism is expressed declaratively:  
- **Data parallel (dp)**: gradient allreduce.  
- **Tensor parallel (tp)**: reduce_scatter + all_gather fused into GEMM.  
- **Pipeline parallel (pp)**: send/recv at stage boundaries.  

#### Example: NVL72 TP GEMM
```python
y_local = x @ W_shard                  # partial on each TP shard
y = all_gather(y_local, axis="tp")     # full output assembled over NVLink/NVSwitch
```

---

### 2.8 Libraries & Primitives

Tessera provides optimized primitives (Ch.9):  
- **Linear algebra**: `gemm`, `cholesky`, `svd`.  
- **NN ops**: `flash_attention`, `layernorm_safe`, `conv2d`.  
- **Spectral ops**: `fft2d`.  
- **Sparse ops**: `spmm`.  
- **RNG**: `randn`, `poisson`.  

#### Example: Transformer block with TP mesh
```python
@jit @autodiff
def block(x, Wqkv, Wo):
    h = rmsnorm_safe(x)
    qkv = gemm(x, Wqkv)                 # TP-sharded GEMM
    q,k,v = split_qkv(qkv, heads=16)
    y = flash_attention(q, k, v)        # all_gather + reduce_scatter fused
    return gemm(y, Wo)
```

---

### 2.9 Portability

On NVIDIA GPUs (Ch.10):  
- `tile.load` → `ld.global` or `cp.async`.  
- `tile.mma` → WMMA/WGMMA.  
- Collectives → NCCL over NVSwitch (with SHARP).  

Inspect IR for debugging:  
```python
print(kernel.inspect_ir("tile"))
print(kernel.inspect_ir("target"))
```

---

### 2.10 Putting It Together

1. Define **mesh** (dp/tp/pp) → map to NVL72 hardware.  
2. Shard tensors with **ShardSpecs**.  
3. Use **primitives** (gemm, attention).  
4. Drop to **tile kernels** only where needed.  
5. Let Tessera’s compiler insert collectives and autotune schedules.  

This unified model scales from single-GPU to NVL72 seamlessly.

