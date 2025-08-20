# Tessera Programming Guide
# Chapter 2. Programming Model

---

## 2.1 Overview

The Tessera programming model defines how computation is expressed, lowered, and executed across GPUs.  
It parallels CUDA’s **thread/block/grid** model, but raises the abstraction level to **operator/tile/mesh**.  

- **Operators** represent algebraic computations (e.g., matmul, FFT).  
- **Tiles** represent fused fragments of computation executed on GPU cores.  
- **Meshes** represent distributed device topologies across which tensors and operators are sharded.  

This abstraction allows developers to focus on *what* should be computed and *how it is partitioned*, instead of manually managing threads, indexing, and synchronization.  

---

## 2.2 Operator Graphs

A **Tessera program** is built from an operator graph:  

- **Nodes** are operators (e.g., `matmul`, `softmax`, `fft`).  
- **Edges** are tensors flowing between operators.  
- **Adjoints** are automatically generated for differentiation.  

Unlike CUDA kernels or PyTorch eager ops, Tessera operators are **algebraic**.  
This means:  
- They can be **rewritten** into equivalent but more efficient forms (e.g., fusing softmax + matmul).  
- They have **explicit mathematical adjoints**, enabling deterministic backward passes.  

Example:
```python
from tessera import op, graph

@graph.module("attention")
def attention(Q, K, V):
    S = op.matmul(Q, K.T)
    A = op.softmax(S)
    return op.matmul(A, V)
```

Here, Tessera captures `attention` as an operator graph, not as three sequential kernels.  
The compiler can then fuse, tile, and distribute it efficiently.  

---

## 2.3 Tiles

Tiles are the Tessera equivalent of CUDA thread blocks.  
- A **tile** is the unit of fused computation (e.g., 128×128×64 matmul fragment).  
- Tessera compiles operators into tile IR, which schedules work to GPU cores.  
- Tiles map naturally to NVIDIA Tensor Cores and WMMA fragments.  

For example:
```python
Y = op.matmul(A, B, tile_shape=(128, 128, 64))
```

This indicates that the matmul will be lowered into tiles of size `(BM=128, BN=128, BK=64)`, optimized for Tensor Core execution.  

---

## 2.4 Meshes

Meshes abstract distributed topologies.  
A mesh defines how devices are arranged along logical axes:  

- **Data Parallel (`dp`)** – Batch dimension partitioning.  
- **Tensor Parallel (`tp`)** – Splitting along model dimensions (rows/columns).  
- **Pipeline Parallel (`pp`)** – Splitting by layers or stages.  

Example:
```python
from tessera import dist

# Define a 3D mesh over 72 GPUs: tensor, pipeline, and data parallel axes
mesh = dist.Mesh(axes=["tp", "pp", "dp"], devices=range(72))
```

---

## 2.5 Shard Specifications

A **ShardSpec** describes how tensors are partitioned across mesh axes.  

Example:
```python
W = dist.tensor(
    shape=(1_000_000, 1_000_000),
    layout=dist.ShardSpec(partition=("row", "col"), mesh_axes=("tp", "pp")),
    mesh=mesh,
    dtype="bf16"
)
```

Here, a massive weight matrix is:  
- Partitioned along its **row and column dimensions**.  
- Sharded across **tensor-parallel and pipeline axes** of the mesh.  

This enables trillion-parameter models to fit naturally across GPU clusters.  

---

## 2.6 Comparison with CUDA

| CUDA Concept   | Tessera Concept | Notes |
|----------------|-----------------|-------|
| Thread         | Scalar element  | Implicit in operator definition |
| Warp           | Tile sub-fragment | Hardware-managed, abstracted |
| Thread Block   | Tile            | Programmer specifies tile size |
| Grid           | Mesh            | Programmer specifies distribution |
| Global Memory  | Sharded Tensor  | Explicit layout via ShardSpec |

Tessera generalizes CUDA’s execution hierarchy into a **graph + algebra + distribution** model.  

---

## 2.7 Summary

- Tessera programs are expressed as **operator graphs**.  
- Operators are lowered into **tiles** that execute efficiently on GPU cores.  
- Distributed execution is expressed through **meshes** and **shard specifications**.  
- This abstraction retains CUDA’s performance but removes manual thread management.  

