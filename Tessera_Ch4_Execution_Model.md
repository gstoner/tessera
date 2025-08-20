# Tessera Programming Guide
# Chapter 4. Execution Model

---

## 4.1 Overview

The Tessera execution model defines how operator graphs are lowered into GPU-executable code and scheduled across devices.  
It builds on the **operator/tile/mesh abstraction** described earlier, providing deterministic scheduling, collective coordination, and reproducibility.  

This chapter explains:  
- How operators are lowered into **Tile IR**.  
- How tiles are scheduled across GPUs.  
- How collectives are executed deterministically.  
- How Tessera ensures reproducibility and performance.  

---

## 4.2 Operator Lowering

Each operator in Tessera (e.g., `matmul`, `fft`, `softmax`) is lowered into **Tile IR**, which specifies:  
- **Tile shape**: `(BM, BN, BK)` for matmuls, or dimension blocks for convolutions.  
- **Memory usage**: Buffers in shared memory, registers, and sharded tensors in HBM.  
- **Thread mapping**: Implicitly handled by the compiler, unlike CUDA.  

Example:
```python
Y = op.matmul(A, B, tile_shape=(128, 128, 64))
```

This produces a set of fused **tile kernels** optimized for Tensor Cores, register reuse, and shared memory double-buffering.  

---

## 4.3 Tile Scheduling

Tiles are scheduled hierarchically:  

1. **Within a GPU**  
   - Tiles are mapped to Streaming Multiprocessors (SMs).  
   - Scheduling aims to maximize occupancy while respecting memory constraints.  

2. **Across GPUs in a Mesh**  
   - Shards determine which GPU executes which tiles.  
   - Pipeline stages and tensor-parallel fragments are placed automatically.  

3. **Overlap**  
   - Compute and communication overlap is exploited.  
   - Example: During matmul, partial results can be reduced while other tiles compute.  

---

## 4.4 Collectives

Tessera treats **collectives** (e.g., all-reduce, all-gather, scatter) as first-class operators.  

- **Deterministic Order**: Collectives execute in a fixed sequence, eliminating nondeterminism common in CUDA/NCCL.  
- **Topology-Aware**: NVLink/NVSwitch topology is taken into account for scheduling.  
- **Composable**: Collectives appear in the operator graph, allowing algebraic rewrites (e.g., fusing an all-reduce into a backward pass).  

Example:
```python
from tessera import dist, op

# Gradient aggregation via all-reduce
grads = op.all_reduce(local_grads, op="sum", axis="dp")
```

---

## 4.5 Reproducibility

Unlike CUDA kernels, Tessera programs are guaranteed to be **bitwise reproducible** across runs.  

Mechanisms:  
- **Fixed reduction order** for collectives.  
- **Stable accumulation rules** for floating-point sums.  
- **Deterministic graph execution** enforced by the runtime.  

This makes Tessera especially suited for scientific ML, where reproducibility is critical.  

---

## 4.6 Execution Flow

1. **Program Build**: Operator graph defined.  
2. **Compilation**: Operators lowered → Tile IR → GPU kernels.  
3. **Sharding & Placement**: Mesh + ShardSpec determines where tensors live.  
4. **Scheduling**: Tiles assigned to SMs and GPUs.  
5. **Execution**: Kernels run with deterministic collectives.  
6. **Profiling**: Tile timing, bandwidth usage, and communication latency recorded.  

---

## 4.7 Performance Considerations

- **Tile shape tuning**: Adjust BM, BN, BK for Tensor Core efficiency.  
- **Communication overlap**: Shard to minimize cross-mesh data movement.  
- **Fused operators**: Prefer algebraic fusion (e.g., softmax + matmul).  
- **Pipeline stages**: Ensure balanced workload across PP axis.  

---

## 4.8 Comparison with CUDA Execution Model

| CUDA Concept | Tessera Equivalent | Notes |
|--------------|--------------------|-------|
| Kernel launch | Operator graph compile + schedule | Tessera abstracts launch |
| Thread block scheduling | Tile scheduling | Implicit, not exposed to user |
| Streams | Operator graph execution order | Deterministic |
| NCCL collectives | Tessera collectives | Deterministic, algebraic |
| Reproducibility | Not guaranteed | Bitwise guaranteed |

---

## 4.9 Summary

- Tessera lowers operators into **Tile IR** optimized for GPU hardware.  
- Tiles are scheduled across SMs and GPUs based on sharding rules.  
- Collectives are **deterministic and algebraically composable**.  
- Execution is **reproducible**, unlike traditional CUDA or NCCL programs.  

