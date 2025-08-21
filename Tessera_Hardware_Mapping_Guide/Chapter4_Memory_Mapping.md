# Tessera Hardware Mapping Guide
## Chapter 4: Memory Mapping

Explains how Tessera handles registers, shared memory, global memory, and collectives.

---

### 4.1 Overview

Efficient GPU programming requires careful use of the **memory hierarchy**.  
Tessera provides abstractions in **Schedule IR** and **Tile IR** to control how tensors are placed, staged, and reused across:

- Registers  
- Shared memory  
- L2 cache  
- Global HBM memory  
- Multi-GPU interconnects (NVLink, NVSwitch)  

---

### 4.2 GPU Memory Hierarchy

| Level        | Latency (cycles) | Bandwidth        | Scope           | Tessera Usage                     |
|--------------|------------------|------------------|-----------------|-----------------------------------|
| Registers    | 1–2              | TB/s             | Per-thread      | Tile IR fragments                 |
| Shared Memory| ~20              | 20+ TB/s (per SM)| SM (per block)  | Schedule IR tile staging          |
| L2 Cache     | ~200             | ~10 TB/s         | All SMs         | Cross-SM tensor reuse             |
| HBM3e        | 500–1000         | 8+ TB/s          | Global device   | Graph IR tensors                  |
| NVLink/NVSw  | 1000–2000+       | 1.8 TB/s/rack    | Multi-GPU       | Collectives in Graph IR           |

---

### 4.3 Tessera Memory Placement

- **Graph IR**: high-level tensor placement (`local`, `sharded`, `replicated`).  
- **Schedule IR**: explicit tiling, prefetch into shared memory.  
- **Tile IR**: registers and warp-level fragments.  

Example:
```mlir
%smemA = "tessera.schedule.prefetch"(%A)
           {scope="shared", double_buffer=true}
```
4.4 Prefetch and Double Buffering

	•	Prefetching hides global memory latency by overlapping load and compute.
	•	Double buffering stages multiple tiles in shared memory to pipeline memory and compute.

Execution pattern:
```
Load tile N+1 → while → Compute tile N
```
Example (Python Tessera API):
```python
with schedule.pipeline(double_buffer=True):
    A_tile = schedule.prefetch(A, scope="shared")
    B_tile = schedule.prefetch(B, scope="shared")
    C_tile = tile.mma_sync(A_tile, B_tile)
```
4.5 Tensor Layouts

Tessera supports multiple layouts for tensors:

	•	Row-major / Column-major
	•	Block-sharded across GPUs
	•	Interleaved fragments for Tensor Cores

Example (sharded tensor):
```python
W = dist.tensor((1_000_000, 1_000_000),
                layout=dist.ShardSpec(partition=("row","col"), mesh_axes=("tp","pp")),
                mesh=mesh, dtype="bf16")
```
4.6 Global Synchronization

	•	Graph IR collectives (all_reduce, broadcast) synchronize data across GPUs.
	•	Schedule IR overlaps collectives with compute via pipelining.

Example:
```python
Y = dist.all_reduce(X, axis="dp", op="sum")
```
4.7 Example: GEMM Memory Mapping

	1.	Graph IR: C = A × B
	2.	Schedule IR: tile A and B into shared memory (128×128×64)
	3.	Tile IR: load tile fragments into registers
	4.	Tensor Cores: perform mma.sync
	5.	Registers → Shared → Global: accumulate results

 HBM → L2 → Shared → Registers → Tensor Cores

 4.8 Summary
 
	•	Tessera makes memory mapping explicit at Schedule IR and Tile IR.
	•	Prefetching and double buffering are first-class constructs.
	•	Tensor layouts and distributed shards are encoded at Graph IR.
	•	This unified view enables performance portability across hardware generations.

⸻
