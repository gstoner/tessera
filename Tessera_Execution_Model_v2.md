# Tessera Execution Model (v2)

This document describes Tessera’s execution model in detail, following the style of NVIDIA’s CUDA Programming Guide, and now includes deep, worked examples inlined within each section.

---

## 1. Kernel Launch Parameters

In Tessera, kernels are launched with explicit mesh and tile configurations, analogous to CUDA’s grid/block model. Developers specify:
- **Global size (N)**: the problem size.
- **Block size (threads per tile)**: number of threads grouped into a Tile IR fragment.
- **Warp-level decomposition**: explicit control over warp allocation to operators.

### Example: SAXPY (1D pointwise kernel)
```python
from tessera import kernel, launch

@kernel
def saxpy(a: float, X: tensor[float], Y: tensor[float], Out: tensor[float]):
    i = launch.thread_id()
    if i < X.shape[0]:
        Out[i] = a * X[i] + Y[i]

launch.grid(blocks=(ceil_div(N, 256),), threads=(256,))(saxpy)(a, X, Y, Out)
```

---

## 2. Mapping Operators to Warps and Tiles

Tessera provides Tile IR constructs to map computations to warps and tiles. Each tile maps to a thread block, and each warp executes a micro-kernel fragment.

### Example: Tiled GEMM with Tensor Cores
```python
from tessera import kernel, tile

@kernel
def gemm_tc(A: tensor[bfloat16], B: tensor[bfloat16], C: tensor[float]):
    # Each tile = 128x128, decomposed into 16x16 warp fragments
    for m, n, k in tile.mma(A, B, C, BM=128, BN=128, BK=32):
        # Use tensor core intrinsic (mma.sync equivalent)
        tile.mma_sync(m, n, k)
```

### Example: FlashAttention with Double Buffering
```python
@kernel
def flash_attention(Q, K, V, O):
    # Prefetch tiles of Q and K using cp.async equivalent
    for block in tile.range(Q, axis=0, step=128):
        q_tile = tile.cp_async(Q, block)
        k_tile = tile.cp_async(K, block)
        tile.barrier()
        scores = tile.dot(q_tile, k_tile.T)
        probs = tile.softmax(scores, axis=-1)
        v_tile = tile.cp_async(V, block)
        O[block] = tile.dot(probs, v_tile)
```

---

## 3. Scheduling Order Guarantees

Tessera’s scheduler ensures:
- **Fairness**: each stream receives scheduling priority proportional to its assigned weight.
- **Overlap**: compute kernels can overlap with asynchronous memory copies.
- **Determinism**: reductions follow explicitly defined associative trees.

### Example: Reduction via Warp Shuffles
```python
@kernel
def warp_reduce_sum(X, Out):
    tid = launch.thread_id()
    val = X[tid]
    for offset in [16,8,4,2,1]:
        val += tile.warp_shuffle_down(val, offset)
    if tile.lane_id() == 0:
        Out[tile.warp_id()] = val
```

### Example: Copy/Compute Overlap with Streams
```python
from tessera import stream

s1 = stream.new()
s2 = stream.new()

# Async H2D transfer on stream 1
stream.memcpy_async(dst=device_X, src=host_X, stream=s1)

# GEMM compute on stream 2
gemm_tc[grid, block, stream=s2](A, B, C)

stream.synchronize()
```

### Example: Double-buffered Pipeline
```python
@kernel
def pipeline(A, B, C):
    buf0, buf1 = tile.alloc_shared(), tile.alloc_shared()
    toggle = 0
    for k in range(0, A.shape[1], 128):
        buf = buf0 if toggle == 0 else buf1
        tile.cp_async(buf, A[:, k:k+128])
        tile.barrier()
        C += tile.dot(buf, B[k:k+128, :])
        toggle ^= 1
```

---

## 4. Deterministic Execution

### Example: Collective Reduction with Fixed Tree
```python
from tessera import dist, op

mesh = dist.Mesh(axes=["dp"], devices=range(8))

# Deterministic all-reduce
Y = op.all_reduce(X, mesh=mesh, algo="binary_tree", deterministic=True)
```

---

## 5. Advanced Features

### Cooperative Groups & Barriers
```python
@kernel
def coop_example(X):
    g = tile.cooperative_group(size=64)
    g.sync()
    X[g.thread_rank()] += 1
```

### Graph Capture & Replay
```python
from tessera import graph

with graph.capture() as g:
    gemm_tc(A, B, C)
    saxpy(alpha, X, Y, Out)

# Replay without CPU overhead
graph.launch(g)
```

### Autotuning Example
```python
from tessera import autotune

config = autotune.search(
    kernel=gemm_tc,
    params={"BM":[64,128], "BN":[64,128], "BK":[16,32]},
    objective="runtime"
)
```

### Fused MLP Walkthrough
```python
@kernel
def fused_mlp(X, W1, b1, W2, b2, Out):
    H = tile.dot(X, W1) + b1
    H = tile.relu(H)
    Out[:] = tile.dot(H, W2) + b2
```

---

## 6. Summary

- Tessera defines **kernel launch semantics** analogous to CUDA but extended to higher-level IR.  
- Operators map directly to **Tile IR** (tensor cores, cp.async, shared memory).  
- Scheduling ensures fairness, overlap, and determinism with **explicit stream semantics**.  
- Advanced features include **autotuning, cooperative groups, graph capture, and fused pipelines**.

This execution model provides a deterministic, tunable, and scalable foundation for Tessera applications.
