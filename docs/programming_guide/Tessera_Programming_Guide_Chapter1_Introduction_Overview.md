---
status: Tutorial
classification: Tutorial
last_updated: 2026-06-11
---

> **Phase status note (updated 2026-06-11):** Phases 1–7 are complete and Phase 8 (Apple M-Series CPU via Accelerate, GPU via Metal/MPS/MPSGraph/custom MSL) is operational — on Apple Silicon this is the primary single-node execution path. Autodiff (forward/reverse transforms + activation checkpointing), ZeRO-2 optimizer sharding, the Bayesian autotuner, and the runtime Python wrapper (`tessera.runtime.TesseraRuntime`) are **shipped**. Genuinely still planned: **multi-GPU / multi-rank** execution of distributed collectives (NCCL/RCCL), `Cyclic` distribution lowering, and **NVL72** rack-scale execution (single-device collectives run over in-process mock ranks today). Canonical API names: `docs/CANONICAL_API.md`; phase table: root `CLAUDE.md`.


# Tessera Programming Guide  
## Chapter 1: Introduction & Overview (Updated)

Tessera is a new programming model for modern accelerators. The current implemented stack covers the Python frontend; **executable backends** today are x86 (AMX/AVX-512), Apple M-Series CPU (Accelerate) and Apple M-Series GPU (Metal/MPS/MPSGraph/custom MSL), plus a production CPU MLIR→LLVM **JIT** lane. Non-Apple hardware is no longer purely gated: NVIDIA sm_120 (consumer Blackwell) has a hardware-verified `mma.sync` matmul and ROCm gfx1151 (Strix Halo) executes a compiler-generated matmul + flash-attention family via `runtime.launch()`. Broader op coverage and datacenter archs (NVIDIA Hopper sm_90 / sm_100, ROCm CDNA) still produce Target IR artifacts with real-hardware execution gated on Phase G/H. Autodiff (forward/reverse + checkpointing) is shipped. Multi-GPU distributed training and NVL72 rack-scale execution remain Phase 4 planned.

Unlike thread- and block-centric models (e.g., CUDA), Tessera is **tile-first**: programmers think in terms of tiles, groups, and meshes. This abstraction, combined with an expressive type system and a multi-level IR, makes Tessera code both **portable** and **high-performance**.

---

### 1.1 Why Tessera?

- **One stack for research → production**: no rewrites between Python prototyping and GPU kernel development.  
- **Tiles, not threads**: simpler reasoning about performance, memory, and synchronization.  
- **Numerics as types**: stability and performance with FP4/FP6/FP8/BF16/FP16/FP32, all declared in the type system.  
- **Autodiff-aware design**: tape-based forward/reverse transforms, custom VJP/JVP rules, and activation checkpointing are **shipped** (effect-aware and collective-aware adjoints registered); see Ch.7.  
- **Standalone compiler**: the runtime is independent of PyTorch / JAX / Flax — they are reference vocabularies only (Architecture Decision #23). The S-series Python reference surface (RNG, state, control, sharding, NN functional, quantization, optimizers, losses + RL, checkpointing, custom ops, AOT, data pipeline) has shipped.  
- **Distributed by construction**: domains/distributions exist today; multi-GPU/multi-rank collective execution is Phase 4 planned (single-device runs over in-process mock ranks).  
- **Portability**: x86 + Apple CPU/GPU execute today; NVIDIA sm_120 and ROCm gfx1151 now execute a compiler-generated matmul/flash-attention family on hardware; broader op coverage and datacenter archs (Hopper sm_90 / sm_100, ROCm CDNA) emit PTX/Tile IR + ROCm artifacts with hardware execution gated on Phase G/H.  
- **NVL72 design target**: treating a 72-GPU NVSwitch rack as a single programming domain is Phase 4 planned.  

---

### 1.2 Key Capabilities at a Glance

| Capability | Tessera Feature | Example |
|------------|----------------|---------|
| **Execution model** | Tile → Group → Mesh hierarchy | `tile.linear_id()` |
| **IR stack** | Graph IR → Schedule IR → Tile IR → Target IR | `fn.graph_ir.to_mlir()` |
| **Numerics** | FP4/FP6/FP8/BF16/FP16/FP32 policies (TF32 is a `math_mode`, not a storage dtype) | `Tensor["B","D", fp8_e4m3 @accum(fp32)]` |
| **Autodiff** | Forward/reverse transforms, custom rules, checkpointing — **shipped** | `tessera.autodiff.reverse(fn)`, `value_and_grad(fn)` |
| **Executable backends** | x86, Apple CPU (Accelerate), Apple GPU (Metal/MPS/MPSGraph), CPU JIT | `@tessera.jit(target="apple_gpu")` |
| **Distributed tensors** | `ShardSpec`, domains, distributions | `tessera.array.from_domain(...)` |
| **Region privileges** | Read/Write/Reduce semantics for safe scheduling | `def step(W: Region["read"], Y: Region["write"])` |
| **Collectives** | Declarative DP/TP/PP/EP parallelism (multi-GPU exec Phase 4 planned) | `tessera.index_launch(axis="tp")(gemm_tile)` |
| **Portability** | NVIDIA PTX + CUDA Tile IR (artifact; hardware Phase G) | `tile.mma → wgmma` |
| **Deployment** | Runtime ABI wiring, AOT bundles (`tessera.aot`), packaged Apple kernels | `aot.export(fn, *examples)` |

---

### 1.3 What’s New in This Specification

Since Tessera Programming Model V1, this spec now includes:  

- **Domains & Distributions**: declarative iteration spaces and tensor layouts.  
- **Region Privileges**: explicit read/write/reduce annotations for safety and overlap.  
- **Index Launches**: scalable task distribution over mesh partitions.  
- **NVL72 Appendix**: best practices for programming NVIDIA’s 72-GPU supernode.  

---

### 1.4 Teaser Example

Here is a short current example showing domains, distributions, and a compiled operation:

```python
import tessera

D = tessera.domain.Rect((4, 128, 256))
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

@tessera.jit
def block(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    return tessera.ops.gemm(A, B)
```

Future NVL72 examples should be marked Phase 4 planned.

---

### 1.5 Structure of This Guide

- **Ch.2 Programming Model**: tiles, groups, meshes, and distributed concepts.  
- **Ch.3 Memory Model**: registers, shared, HBM, NVLink/NVSwitch.  
- **Ch.4 Execution Model**: pipelines, async copies, barriers.  
- **Ch.5 Kernel Programming**: `@tessera.kernel`, dtype annotations, `index_launch`, tile DSL, FA-4 attention ops.  
- **Ch.6 Numerics Model**: FP4/6/8/BF16/FP16/FP32, safe ops, rounding.  
- **Ch.7 Autodiff**: shipped forward/reverse transforms, custom rules, checkpointing, and effect contracts.  
- **Ch.8 Layouts & Data Movement**: explicit layouts, async pipelines.  
- **Ch.9 Libraries & Primitives**: dense, sparse, spectral, RNG.  
- **Ch.10 Portability**: NVIDIA PTX/Tile IR, NCCL, CUTLASS interop.  
- **Ch.11 Conclusion**: programming workflow, deployment, best practices.  
- **Appendix A (NVL72)**: future-facing guide for Phase 4 planned NVL72 programming.  

---

### 1.6 Summary

Tessera provides:  
- **A unified programming model** across kernels, models, and distributed execution.  
- **Tiles, domains, and distributions** as first-class concepts.  
- **Safe parallelism** via region privileges and effect tracking.  
- **Performance portability** from current single-node paths toward planned NVL72 support.  

With Tessera, programmers use one model across current compiler paths while the distributed/runtime roadmap fills in later phases.
