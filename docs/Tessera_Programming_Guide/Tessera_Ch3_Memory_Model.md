# Tessera Programming Guide
# Chapter 3. Memory Model

---

## 3.1 Overview

Tessera’s memory model defines how data is stored, accessed, and synchronized across GPUs.  
It builds on concepts familiar from CUDA (registers, shared memory, global memory), but extends them to **sharded tensors** and **mesh-wide consistency**.  

Key goals of the Tessera memory model:  
- **Explicit layouts** via `ShardSpec`.  
- **Hierarchical memory awareness** (registers → SRAM → HBM → NVLink).  
- **Deterministic consistency rules** across devices.  
- **Performance-portable abstractions** that map naturally to modern GPU memory systems.  

---

## 3.2 Memory Hierarchy

Tessera maps naturally to NVIDIA GPU memory tiers:  

| Memory Tier    | Tessera Concept      | Scope               | Latency / Bandwidth |
|----------------|----------------------|---------------------|----------------------|
| Registers      | Tile-local scalars   | Single thread/TensorCore | 1 cycle, highest BW |
| Shared Memory  | Tile-local buffer    | Tile / SM           | ~20 cycles, TB/s |
| HBM3e (Global) | Tensor shards        | Single GPU          | ~200–300 cycles, 8–12 TB/s |
| NVLink / NVSwitch | Mesh-collective tensors | Multi-GPU domain | ~500ns–1µs latency, 1–1.8 TB/s |

Tessera developers do **not** manage registers or shared memory explicitly (unlike CUDA).  
Instead, they **declare shard layouts** and **tile shapes**, and the compiler maps these to optimal memory usage.  

---

## 3.3 Sharded Tensors

In Tessera, global memory is abstracted as **sharded tensors**.  
Each tensor has an associated `ShardSpec` describing:  
- Partition dimensions (row, col, batch, channel, etc.).  
- Mapping to mesh axes (`dp`, `tp`, `pp`).  
- Replication or reduction semantics.  

Example:
```python
from tessera import dist

X = dist.tensor(
    shape=(1024, 1024),
    layout=dist.ShardSpec(partition=("row",), mesh_axes=("tp",)),
    mesh=mesh,
    dtype="fp32"
)
```

This partitions the matrix rows across the **tensor-parallel axis** of the mesh.  
HBM usage per GPU is reduced by a factor of the partition size.  

---

## 3.4 Replication and Reduction

Some tensors are **replicated** across GPUs instead of sharded (e.g., small weight matrices).  
Others require **reductions** (e.g., gradient aggregation).  

Tessera specifies this with layout annotations:  

- `ShardSpec(partition=None, replicate=True)` → full replication.  
- `ShardSpec(partition=("row",), reduce="sum")` → reduce-scatter after accumulation.  

These semantics ensure deterministic collectives across the mesh.  

---

## 3.5 Memory Consistency

CUDA allows relaxed memory ordering; Tessera enforces **deterministic consistency** across operators:  

1. **Within a Tile**: Execution is sequentially consistent.  
2. **Within a GPU**: Operators respect dependency order in the graph.  
3. **Across Mesh**: Collectives (all-reduce, all-gather) execute in deterministic order.  

Unlike CUDA atomics (which may be non-deterministic across GPUs), Tessera guarantees that repeated runs produce **bit-identical results**.  

---

## 3.6 Memory Performance Guidelines

To achieve high performance in Tessera:  

- **Choose shard dimensions carefully**: Sharding along batch or sequence dims minimizes communication.  
- **Balance memory footprint**: Keep per-GPU HBM usage within 80% of available memory.  
- **Exploit locality**: Use tile shapes that fit in shared memory buffers to minimize HBM traffic.  
- **Overlap communication and compute**: Tessera schedules collectives asynchronously where safe.  

---

## 3.7 Comparison with CUDA

| CUDA Memory Model  | Tessera Memory Model |
|--------------------|----------------------|
| Global Memory      | Sharded Tensor in HBM |
| Shared Memory      | Tile-local Buffer |
| Registers          | Tile scalars |
| Unified Memory     | Mesh-wide Sharded Tensor |
| Atomics            | Deterministic collectives (reduce, scatter, gather) |

---

## 3.8 Summary

- Tessera abstracts memory as **sharded tensors** with explicit layouts.  
- Shards map to the GPU memory hierarchy and mesh collectives.  
- Consistency rules ensure deterministic results, unlike CUDA atomics.  
- Developers optimize by choosing shard dimensions, tile sizes, and communication overlap strategies.  

