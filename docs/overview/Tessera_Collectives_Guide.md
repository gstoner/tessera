# Tessera Collectives & Distributed Systems Guide

## 1. Introduction
Tessera integrates **collective communication** and **distributed tensor semantics** directly into the programming model. Unlike CUDA or MPI where collectives are external libraries (NCCL, RCCL, MPI), Tessera treats them as **first-class operators** lowering through the IR stack.  

- **Scope**: Multi-GPU within a node, multi-node across clusters.  
- **Abstraction**: Distributed mesh topologies (`dist.Mesh`) define layouts.  
- **Determinism**: Reductions and collectives guarantee reproducible order.  
- **Scalability**: Borrowing techniques from DeepSpeed (ZeRO partitioning, pipeline/tensor parallelism, communication overlap).  

---

## 2. Distributed Mesh Model

### 2.1 Mesh Definition
A mesh defines how devices (GPUs or nodes) are grouped along named axes.
```python
mesh = dist.Mesh(axes=["dp", "tp", "pp"], devices=range(72))  # 3D mesh
```

- **dp**: data parallel  
- **tp**: tensor parallel  
- **pp**: pipeline parallel  

### 2.2 Shard Specification
Tensors carry **ShardSpec** metadata mapping dimensions to mesh axes:
```python
W = dist.tensor((10000, 10000),
                layout=dist.ShardSpec(partition=("row", "col"), mesh_axes=("tp", "pp")),
                mesh=mesh)
```

---

## 3. Collective Operations

Tessera defines deterministic collectives:

- **AllReduce**: `op.all_reduce(X, op="sum", axis="dp")`  
- **ReduceScatter**: `op.reduce_scatter(X, axis="tp")`  
- **AllGather**: `op.all_gather(X, axis="tp")`  
- **Broadcast**: `op.broadcast(X, src=0, axis="pp")`  
- **AllToAll**: `op.all_to_all(X, axis="tp")`  

All collectives:
- **Normative requirement**: Reductions are performed in a deterministic order, bitwise reproducible.  
- **Scheduling**: Collectives may overlap with computation if marked `async=True`.  

---

## 4. Parallelism Models

### 4.1 Data Parallelism
- Replicated weights, split data.  
- Uses `all_reduce` for gradient synchronization.  

### 4.2 Tensor Parallelism
- Splits tensors across `tp` axis.  
- Requires `all_gather` and `reduce_scatter` around matmuls.  

### 4.3 Pipeline Parallelism
- Model layers split across `pp` axis.  
- **Pipeline stages** are scheduled with forward/backward microbatch overlap.  

### 4.4 Hybrid Parallelism
- Any combination of DP, TP, PP.  
- Tessera allows arbitrary mesh factorization.  

---

## 5. Deterministic Reductions

- **Requirement**: Tessera defines reductions as associative but **not re-ordered** arbitrarily.  
- **Implementation**: Uses fixed tree topology or ring order.  
- **Motivation**: Ensures reproducibility across training runs.  

---

## 6. Techniques Borrowed from DeepSpeed

Tessera integrates scaling strategies proven in DeepSpeed:  

- **ZeRO Partitioning**: Partition optimizer state, gradients, and activations across DP groups.  
  ```python
  optimizer = dist.zero_optim(model.parameters(), stage=2)
  ```
- **Communication/Computation Overlap**: Collectives can be launched asynchronously (`async=True`) and fused with kernel launches.  
- **Pipeline Engine**: Explicit microbatch scheduling built into Tessera Graph IR.  
- **Activation Checkpointing**: Integrated into Tessera’s autodiff graph to reduce memory footprint.  

---

## 7. Multi-Node Systems

- **Transport Backends**:  
  - NVIDIA → NCCL over NVLink/IB.  
  - AMD → RCCL.  
  - Intel → oneCCL.  
  - CPU clusters → MPI.  

- **Topology Awareness**: Tessera maps meshes onto node+device hierarchies.  
- **Fault Tolerance**: Collectives raise deterministic errors if rank failure detected.  

---

## 8. Example: Hybrid Parallel Training

```python
from tessera import dist, op, graph

# Define 3D mesh for 1024 GPUs: dp × tp × pp = 16 × 8 × 8
mesh = dist.Mesh(axes=["dp","tp","pp"], devices=range(1024))

# Shard large model weights
W = dist.tensor((1_000_000, 1_000_000),
                layout=dist.ShardSpec(partition=("row","col"), mesh_axes=("tp","pp")),
                mesh=mesh)

# Forward pass with distributed matmul
Y = op.gemm(W, X)

# Backward pass with deterministic allreduce
@graph.backward_step
def step(loss):
    grads = graph.backward(loss)
    grads_synced = op.all_reduce(grads, op="sum", axis="dp")
    return grads_synced
```

---

## 9. Performance Notes

- Use **persistent NCCL communicators** to avoid setup cost.  
- Fuse small collectives into a single group op.  
- Overlap compute/comm using `async=True` collectives.  
- For pipeline parallelism, balance stage compute time to minimize bubble overhead.  

---

## 10. Normative Rules

- **Collectives MUST** be invoked on all participating ranks synchronously.  
- **Reductions MUST** guarantee deterministic results.  
- **Mesh mapping MUST** be static during execution of a compiled graph.  
- **Asynchronous collectives MAY** overlap, but order must be explicitly synchronized.  

---

📌 **Key Difference vs CUDA**: In CUDA, collectives (NCCL) are external. In Tessera, collectives are **native operators** in the IR, with **determinism and reproducibility defined normatively**.
