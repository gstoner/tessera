# Tessera Tutorials Volume
## Chapter 3 — Memory & Performance

### 3.1 Shared Memory Staging
Shared memory can be used to stage data for reuse across threads in a block, reducing global memory traffic.

```python
from tessera import op

@op.kernel
def tiled_matmul(A, B, C, tile_size: int = 32):
    # Allocate tile in shared memory
    tileA = op.shared((tile_size, tile_size), dtype=A.dtype)
    tileB = op.shared((tile_size, tile_size), dtype=B.dtype)

    for i, j, k in op.tile_range(A.shape[0], B.shape[1], A.shape[1], step=tile_size):
        tileA[:] = A[i:i+tile_size, k:k+tile_size]
        tileB[:] = B[k:k+tile_size, j:j+tile_size]
        C[i:i+tile_size, j:j+tile_size] += op.matmul(tileA, tileB)
```

---

### 3.2 Prefetching & Double-Buffer Pipelines
Tessera allows pipelining global memory loads with compute using double buffering.

```python
@op.kernel
def pipelined_attention(Q, K, V, block: int = 128):
    smem_Q = op.shared((block, Q.shape[-1]))
    smem_K = op.shared((block, K.shape[-1]))
    smem_V = op.shared((block, V.shape[-1]))

    for start in range(0, Q.shape[0], block):
        op.prefetch(smem_Q, Q[start:start+block])
        op.prefetch(smem_K, K[start:start+block])
        op.prefetch(smem_V, V[start:start+block])

        op.barrier()  # Ensure data ready
        scores = op.matmul(smem_Q, smem_K.T)
        weights = op.softmax(scores)
        out = op.matmul(weights, smem_V)
```

---

### 3.3 Autotuner Integration
Tessera integrates an **autotuner** that can optimize kernel schedules using both **cost models** and **on-device measurements**.  

```python
from tessera import autotune

@op.autotunable(params={"tile": [16, 32, 64], "unroll": [1, 2, 4]})
def conv2d(X, W):
    return op.conv2d(X, W)

best_config = autotune(conv2d, input_shapes={"X": (64, 3, 224, 224), "W": (64, 3, 7, 7)})
print(best_config)
```

The autotuner maintains a **persistent cache per shape/arch** so retraining is unnecessary once tuned.

---

### 3.4 Memory Hierarchy Considerations
- **Global Memory**: High capacity, high latency → minimize redundant reads.  
- **Shared Memory**: Low latency, block-local → stage tiles for reuse.  
- **Registers**: Fastest, but limited → keep inner-loop accumulators here.  
- **Prefetching**: Overlap compute with memory transfer.  

---

### 3.5 Performance Checklist
- Use **tiling** to increase data reuse.  
- Apply **double-buffering** to hide latency.  
- Use **autotuning** for architecture-specific optimization.  
- Prefer **fused operators** to reduce intermediate memory traffic.  
