# Tessera Collectives & Distributed Systems Guide

## 1. Introduction

Collective operations and distributed execution are central to scaling Tessera beyond a single GPU.  
This guide explains:

- **What it is**: Tessera’s unified abstraction for collective communication and distributed layouts.  
- **How it works**: Mapping high-level tensor/graph operators into efficient collectives (all-reduce, all-gather, reduce-scatter, broadcast) on large GPU clusters.  
- **How to use it**: Annotating distributed tensors with mesh layouts, configuring pipeline/tensor/data parallelism, and controlling determinism in reductions.  

---

## 2. Distributed Abstractions in Tessera

Tessera provides three levels of abstraction:

- **Distributed Meshes** – logical mapping of devices (single-node, multi-node, racks).  
- **Shard Specs** – how tensors are partitioned across axes (`row`, `col`, `depth`).  
- **Collectives** – semantic operations (e.g., `op.all_reduce`) that lower into backend-specific implementations.  

### Example: Defining a Mesh
```python
from tessera import dist

# Define a 3D mesh: tensor-parallel × pipeline × data-parallel
mesh = dist.Mesh(axes=["tp","pp","dp"], devices=range(72))
```

### Example: Sharding a Tensor
```python
W = dist.tensor(
    shape=(1_000_000, 1_000_000),
    layout=dist.ShardSpec(partition=("row","col"), mesh_axes=("tp","pp")),
    mesh=mesh,
    dtype="bf16"
)
```

---

## 3. Core Collectives

- **All-Reduce**: Sum across GPUs (used for gradient aggregation).  
- **Reduce-Scatter**: Reduction + partitioning (used in optimizer states).  
- **All-Gather**: Distribute partitioned tensors to all GPUs.  
- **Broadcast**: Send from one root to many GPUs.  

Each collective is **deterministic by construction**: Tessera fixes reduction order and numeric policy.

---

## 4. Integration with NVLink, NVSHMEM, and DeepEP

- **NVLink/NVSwitch**: Tessera lowers collectives to topology-aware kernels.  
- **NVSHMEM / DeepEP**: Used for fine-grained one-sided communication, enabling overlap of computation and comms.  
- **DeepSpeed ZeROFlow**: Tessera adopts partitioned optimizer states + activation checkpointing, integrated into Graph IR schedules.  

---

## 5. Parallelism Modes

Tessera supports hybrid parallelism:

- **Data Parallelism**: Batch split across GPUs.  
- **Tensor Parallelism**: Splits large matrix ops across rows/cols.  
- **Pipeline Parallelism**: Layers partitioned along pipeline stages.  
- **Expert Parallelism (MoE)**: Routing subsets of tokens to different GPUs.  

These modes can be composed through **multi-axis meshes**.

---

## 6. Best Practices

- Use **reduce-scatter** over all-reduce when possible (lower memory pressure).  
- Overlap collectives with compute using **asynchronous ops**.  
- Persist **autotuned schedules per mesh+shape** for re-use.  
- For determinism, set `dist.set_deterministic(True)` globally.  

---

## 7. Example: Training with Hybrid Parallelism

```python
from tessera import dist, op, graph

# 4-way tensor parallel, 8-way data parallel, 2-stage pipeline
mesh = dist.Mesh(axes=["tp","dp","pp"], devices=range(64))

# Shard weights across tensor-parallel
W = dist.tensor((100_000, 100_000),
                layout=dist.ShardSpec(partition=("row","col"), mesh_axes=("tp","pp")),
                mesh=mesh)

# Forward pass: tensor-parallel matmul
Y = op.matmul(X, W)

# Backward pass: gradients aggregated with all-reduce
grads = graph.backward(Y)
```

---

## 8. Debugging and Profiling Collectives

- Use `dist.debug_trace()` to log collective operations.  
- Profile comm/comp overlap with Tessera Profiler (like Nsight Systems).  
- Enable `dist.validate_determinism()` for reproducibility tests.  

---

## 9. Future Extensions

- **Elastic collectives**: Dynamic resizing when nodes fail.  
- **Topology-aware layouts**: Automatic mapping for NVL72/InfiniBand.  
- **Energy-aware scheduling**: Balance bandwidth vs. power draw.  
