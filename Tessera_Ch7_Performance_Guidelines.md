# Tessera Programming Guide
# Chapter 7. Performance Guidelines

---

## 7.1 Overview

Performance in Tessera is achieved by combining **operator algebra**, **tile-level optimizations**, and **mesh-aware scheduling**.  
Unlike CUDA, which exposes low-level thread and block primitives, Tessera focuses on **algebraic optimizations** and **hardware mapping rules**.  

This chapter provides guidelines for maximizing performance on single GPUs and multi-GPU clusters such as NVL72.  

---

## 7.2 Tile Shape Tuning

Tile shapes (`BM, BN, BK`) directly impact Tensor Core efficiency and shared memory usage.  

- **Guideline 1**: Choose `BM, BN` to match Tensor Core fragment sizes (e.g., multiples of 64 for BF16/FP16).  
- **Guideline 2**: Increase `BK` to maximize register reuse.  
- **Guideline 3**: Avoid exceeding shared memory limits per block.  

Example:
```python
Y = op.matmul(A, B, tile_shape=(128, 128, 64))
```

---

## 7.3 Memory Hierarchy Optimization

Tessera automatically manages GPU memory tiers but developers can guide layout.  

- **Registers**: Store temporaries inside tile kernels.  
- **Shared Memory**: Use double-buffering for matmul/conv tiles.  
- **HBM**: Place large tensors; shard to minimize traffic.  
- **NVLink/NVSwitch**: Used for collectives; minimize frequency of global ops.  

Guideline: **Prefer operator fusion** to reduce repeated HBM loads.  

---

## 7.4 Operator Fusion and Rewrites

Algebraic rewrites are the main mechanism for performance.  

Examples:  
- **FlashAttention**: Fuse `softmax ∘ matmul ∘ matmul` into a single kernel.  
- **Spectral Convolution**: Rewrite `conv2d` into `fft ∘ pointwise_mul ∘ ifft`.  
- **Pipeline Fusion**: Fuse gradient reductions into backward operators.  

Guideline: **Write operators in algebraic form and let compiler fuse**.  

---

## 7.5 Sharding and Mesh Mapping

Correct **ShardSpec + Mesh** design is crucial.  

- **Tensor Parallel (TP)**: Split large matrix multiplies across GPUs.  
- **Pipeline Parallel (PP)**: Assign operator sequences to stages.  
- **Data Parallel (DP)**: Replicate weights, split batches.  

Guideline: Balance TP, PP, DP axes to minimize communication cost.  

Example:
```python
mesh = dist.Mesh(axes=["tp","pp","dp"], devices=range(72))
W = dist.tensor(shape=(1_000_000, 1_000_000),
                layout=dist.ShardSpec(partition=("row","col"), mesh_axes=("tp","pp")),
                mesh=mesh)
```

---

## 7.6 Communication Overlap

Tessera automatically overlaps compute and communication when possible.  

- **Gradient Aggregation**: All-reduce overlapped with next forward step.  
- **Pipeline Parallelism**: Micro-batching enables compute/comm overlap.  
- **FFT-based Operators**: Tessera overlaps butterfly stages with local compute.  

Guideline: **Favor sharding that enables overlap** (e.g., local reductions before global).  

---

## 7.7 Profiling in Tessera

Tessera includes profiling tools similar to `nvprof` but at the operator/tile level.  

- **Tile Timing**: Measure per-tile execution latency.  
- **Memory Bandwidth**: Track HBM and NVLink utilization.  
- **Fusion Trace**: Show which algebraic rewrites occurred.  

Example:
```python
graph.profile(step_fn)
```

Output:
```
MatMul[128x128x64]  latency=2.3µs   occupancy=92%
AllReduce[dp]       latency=15µs    bandwidth=1.6TB/s
FusedOp[FlashAttn]  latency=4.1µs   tiles=256
```

---

## 7.8 NVL72-Specific Guidelines

On large-scale clusters like GB200 NVL72:  

- **Topology-Aware Sharding**: Align mesh axes with physical NVLink partitions.  
- **Spectral Operators**: FFT/IFFT distribute well across 72 GPUs.  
- **Recursive MoE**: Experts can be pinned to sub-mesh groups.  
- **Collective Scheduling**: Favor reduce-scatter + all-gather over naive all-reduce.  

Guideline: **Map compute-heavy operators locally; global comms only when required**.  

---

## 7.9 Comparison with CUDA Performance Tuning

| CUDA Practice        | Tessera Equivalent | Notes |
|----------------------|--------------------|-------|
| Tune thread/block size | Tune tile shape (BM, BN, BK) | Higher-level, hardware-aware |
| Shared memory tiling | Automatic tile buffering | User can annotate if needed |
| Stream overlap       | Communication overlap | Compiler/runtime handled |
| Kernel fusion        | Algebraic operator fusion | Declarative |
| NVLink/NCCL tuning   | Mesh + ShardSpec mapping | Topology-aware |

---

## 7.10 Summary

- Choose **tile shapes** aligned with Tensor Cores.  
- Use **fusion rules** to minimize memory traffic.  
- Design **ShardSpec + Mesh** layouts for balance.  
- Overlap communication and compute where possible.  
- Use Tessera’s **profiling tools** to identify bottlenecks.  
- On NVL72, leverage spectral and recursive operators for scalable efficiency.  

