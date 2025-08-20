# Tessera Hardware Mapping Guide
## Chapter 1: Overview

---

### 1.1 Purpose

The Tessera programming model is designed to **bridge high-level ML operator algebra** and **low-level GPU execution**.  
This guide describes how Tessera abstractions map onto GPU hardware:

- **Graph IR** → expresses ML computations as algebraic graphs.  
- **Schedule IR** → describes tiling, fusion, and pipeline strategies.  
- **Tile IR** → defines block/warp-level computations targeting Tensor Cores.  
- **Target IR** → lowers into PTX/LLVM for final codegen.  

---

### 1.2 Abstraction Stack

Tessera organizes its abstractions into four IR layers:
      
|  Graph IR           | ML Ops, Autodiff, Algebraic Rewrites       |
|---------------------|--------------------------------------------|
|  Schedule IR        | Tiling, Fusion, Pipeline, Autotuning       |
|  Tile IR            |  Warps, Tensor Cores, Memory Layouts       |
|  Target IR (PTX)    | PTX/SASS, ROCm LLVM, Hardware ABI          |

- **Graph IR** is hardware-agnostic and declarative.  
- **Schedule IR** decides *how* computation is partitioned.  
- **Tile IR** binds computations to **SMs, warps, and Tensor Cores**.  
- **Target IR** integrates with **CUDA PTX** or **ROCm LLVM** backends.  

---

### 1.3 Comparison with CUDA

CUDA’s model centers around:
- **Grids** → multi-block launch.  
- **Blocks** → cooperative thread groups.  
- **Threads** → SIMT execution.  

Tessera generalizes this into operator-based IR:

| CUDA Concept  | Tessera Equivalent  |
|---------------|----------------------|
| Grid          | Distributed Mesh (multi-GPU) |
| Block         | Tile IR block (e.g., 128×128) |
| Warp          | Tile fragment (MMA unit) |
| Thread        | Lane in warp fragment |

---

### 1.4 Execution Flow

A Tessera program executes as follows:

1. **Graph IR Construction**  
   User defines an operator graph (e.g., matmul, softmax).  

2. **Graph IR → Schedule IR**  
   Compiler applies tiling, fusion, pipeline strategies.  

3. **Schedule IR → Tile IR**  
   Kernels lowered to block/warp-level Tensor Core ops.  

4. **Tile IR → Target IR**  
   Emitted as PTX, passed through `nvcc` → GPU binary.  

---

### 1.5 Example Diagram

Execution mapping for a matrix multiply:

Graph IR:   C = A × B
|
v
Schedule IR: tile(m=128,n=128,k=64), fuse
|
v
Tile IR:  warps compute MMA fragments
|
v
Target IR: PTX -> SASS (Tensor Core ops)

---

### 1.6 Summary

- Tessera introduces a **layered IR stack** that generalizes CUDA’s grid-block-thread model.  
- Each layer progressively lowers from **algebraic ops → tiles → hardware PTX**.  
- This abstraction provides a **clean mapping** to GPU hardware while preserving **optimizable structure** for autotuning and portability.  

---
