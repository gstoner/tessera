# Tessera Programming Guide
# Chapter 1. Introduction

---

## 1.1 Purpose of This Guide

This guide introduces **Tessera**, a programming model for deep learning and scientific machine learning, in the style of NVIDIA’s CUDA C Programming Guide.  
It is intended for developers, researchers, and system architects who require **scalable, deterministic, and algebraically structured execution** across modern GPU clusters.  

The content is organized into chapters, moving from programming concepts and memory rules to execution, operators, numerical policies, performance guidelines, and worked examples.  
Each chapter builds on the previous, and the appendices provide ready-to-run examples and profiling tables.

---

## 1.2 What Is Tessera?

Tessera is a **graph-based operator programming model** designed for:  

- **Deep Learning**: Transformer models, diffusion models, mixture-of-experts.  
- **Physics-Informed Models**: PINNs, operator learning, PDE solvers.  
- **Hybrid AI/Science Workloads**: Combining symbolic, spectral, and neural components.  

At its core, Tessera introduces three abstraction layers:  

- **Operator Graph IR** – Defines computation as algebraic operators with explicit adjoints.  
- **Tile IR** – Lowers operators into GPU tile fragments (similar to CUDA thread blocks).  
- **Mesh Abstraction** – Defines distributed device topologies for parallel execution.  

Together, these abstractions provide a **deterministic execution model** that scales from a single GPU to large NVLink/NVL72 clusters.

---

## 1.3 Why Tessera?

CUDA and higher-level frameworks (PyTorch, JAX, TensorFlow) provide excellent support for GPU programming, but Tessera addresses three persistent gaps:

1. **Determinism at Scale**  
   - CUDA atomics and collectives may yield non-deterministic results across devices.  
   - Tessera enforces **fixed reduction orders** and **stable accumulation**.  

2. **Operator Algebra**  
   - PyTorch and JAX express operations imperatively.  
   - Tessera treats operators as algebraic entities with explicit adjoints, enabling optimizations like operator fusion and spectral rewrites.  

3. **Unified Multi-Scale Programming**  
   - Traditional frameworks separate ML kernels from PDE solvers.  
   - Tessera integrates both, allowing a PINN and a transformer to coexist in one operator graph.  

---

## 1.4 Tessera vs CUDA

- **CUDA**: Thread/block/grid model; explicit kernels; programmer manages indexing, memory movement.  
- **Tessera**: Operator/tile/mesh model; implicit scheduling; programmer declares shard layouts and operator graphs.  

Analogy:  
- CUDA developer decides *how to launch threads*.  
- Tessera developer decides *how to shard tensors and compose operators*.  

---

## 1.5 Tessera Workflow

A typical Tessera workflow looks like:  

1. **Define the Mesh**  
   - Choose topology (data, tensor, pipeline parallel).  

2. **Declare Distributed Tensors**  
   - Annotate tensors with `ShardSpec`.  

3. **Compose Operators**  
   - Build algebraic graphs (`matmul`, `fft`, `softmax`, PDE solvers).  

4. **Compile & Execute**  
   - Tessera lowers operators → tile IR → device kernels.  
   - Deterministic collectives scheduled on mesh.  

5. **Profile & Optimize**  
   - Use Tessera profiler for tile times, bandwidth, collective latency.  

This is similar to CUDA’s compile→launch→profile workflow, but **at a higher algebraic level**.  

---

## 1.6 Audience

This guide assumes readers have familiarity with:  

- CUDA or GPU programming concepts (threads, memory hierarchy, synchronization).  
- Machine learning frameworks (PyTorch, JAX, TensorFlow).  
- Basic linear algebra, PDEs, and operator-based reasoning.  

It is not necessary to be an expert in all three, but some background in each will help.

---

## 1.7 Summary

- Tessera introduces a **deterministic, operator-graph model** for GPU programming.  
- It extends CUDA by abstracting threads/blocks into tiles/meshes.  
- It unifies deep learning and scientific ML under one programming abstraction.  
- This guide provides the reference needed to **write, optimize, and deploy Tessera programs** from single-GPU to frontier-scale systems.  

