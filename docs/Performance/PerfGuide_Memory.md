# Tessera Performance Best Practices Guide
## Chapter 3: Memory Hierarchy and Bandwidth

---

### 3.1 Overview

GPU performance is often **memory-bound** rather than compute-bound.  
Tessera provides explicit control over memory usage at the **Schedule IR** and **Tile IR** levels.

Key memory tiers:

1. **Global HBM**  
   - High capacity, high bandwidth (3–5 TB/s on GB200).  
   - Long latency (~400–800 cycles).

2. **Shared Memory / L1 Cache**  
   - On-chip SRAM, low latency.  
   - Critical for tiling, buffering, and warp communication.  

3. **Registers**  
   - Fastest tier, but limited per thread.  
   - Excessive usage → spilling to local memory (slow).  

4. **NVLink/NVSwitch Interconnect**  
   - High-bandwidth collectives across GPUs.  
   - Exposed via `dist.collective()` APIs.

---

### 3.2 Global Memory Best Practices

- **Coalesced Access**: Ensure threads in a warp access contiguous addresses.  
- **Vectorized Loads**: Use `float4`/`bf16x2` style packing when possible.  
- **Alignment**: Align to 128-bit boundaries for TensorCore efficiency.  
- **Avoid Stride > 1**: Strided loads reduce coalescing efficiency.  

**Example (vectorized load in Tessera):**
```python
from tessera import schedule

@op.kernel
def vec_add(A, B, C):
    schedule.vectorize(4)   # load 4 elements per thread
    C[:] = A + B
```
3.3 Shared Memory Best Practices
	•	Use shared memory to stage tiles from HBM.
	•	Avoid bank conflicts by padding arrays (+1 stride for 32-bank hardware).
	•	Use async copies (schedule.async_copy(True)) to overlap load+compute.

Example (matmul with smem staging):

```python
@op.kernel
def matmul(A, B, C):
    schedule.tile((128, 128, 32))
    schedule.use_shared_memory(True, padding=1)
    schedule.async_copy(True)
    return A @ B + C
```

3.4 Register Usage
	•	Each thread gets a fixed register budget.
	•	Too many registers → spills to local memory (slow).
	•	Too few → forces smaller tile sizes.

Best Practice: Let the autotuner sweep tile shapes while monitoring register count.

⸻

3.5 NVLink / NVSwitch Collectives
	•	Tessera lowers collectives (all_reduce, all_gather, reduce_scatter) to NVLink operations.
	•	Communication is overlapped with compute using asynchronous streams.
	•	Mesh-aware sharding ensures locality (minimizes hops on NVSwitch).

Example (sharded matmul with reduce-scatter):

```python 
from tessera import dist, op

mesh = dist.Mesh(axes=["tp", "dp"], devices=range(8))
A = dist.tensor((B, D), layout=dist.ShardSpec(("row",), ("dp",)), mesh=mesh)
B = dist.tensor((D, H), layout=dist.ShardSpec(("col",), ("tp",)), mesh=mesh)

C_partial = op.matmul(A, B)
C = dist.collective.reduce_scatter(C_partial, axis="tp")
```
3.6 Worked Example: FlashAttention Memory Optimization

FlashAttention is memory-bound due to softmax buffering.
	•	Naive Implementation:
Loads query, key, value from HBM repeatedly → bandwidth bottleneck.
	•	Optimized Tessera Schedule:

```python
schedule.tile((128, 128, 64))
schedule.use_shared_memory(True, padding=1)
schedule.async_copy(True)      # overlap load + compute
schedule.pipeline_stages(3)    # triple-buffering
```
	•	Outcome:
	•	Reduces HBM traffic ~3×.
	•	Latency hidden via pipelined shared-memory stages.
	•	Achieves ~80–90% of peak bandwidth.

⸻

3.7 Key Takeaways
	•	Always check for coalesced access and alignment.
	•	Use shared memory tiling with async copy + padding to avoid bank conflicts.
	•	Monitor register usage to prevent spills.
	•	Exploit collectives and shard-aware layouts for distributed efficiency.
	•	FlashAttention and similar bandwidth-bound ops benefit most from tiling + async pipelines.

