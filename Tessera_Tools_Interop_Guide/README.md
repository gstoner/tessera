# Tessera Interop & Tooling Guide

This guide documents Tessera’s interoperability with other frameworks and its developer tooling.  
It is structured in modular chapters, similar to the NVIDIA CUDA Programming Guide.

## Contents

- **Chapter 1: Python & C++ APIs**
  - Using Tessera from Python
  - Embedding Tessera in C++ runtimes

- **Chapter 2: MLIR Integration**
  - Tessera MLIR dialects (Graph IR, Schedule IR, Tile IR)
  - Interop with upstream MLIR lowering

- **Chapter 3: External ML Framework Interop**
  - PyTorch integration (custom ops, TorchScript)
  - JAX integration (XLA bridge)
  - Hugging Face Transformers acceleration

- **Chapter 4: Debugging Tools**
  - Graph inspection and IR dumps
  - Numerical tracing and summaries
  - Autodiff validation (gradient checks)
  - Determinism and reproducibility checks

- **Chapter 5: Profiling & Autotuning**
  - Runtime profiler (latency, FLOPs, bandwidth)
  - Cost models and on-device autotuning
  - Persistent caches per shape/architecture

- **Appendix A: Reference Commands & Examples**
  - CLI tools (`tessera run`, `tessera-mlir`, `tessera-prof`)
  - C++ snippets for embedding Tessera
  - MLIR dialect cheatsheet
  - Debugging & profiling reference

---

## Usage

Each chapter is a standalone Markdown file.  
You can read them individually or in sequence for a full guide.

Example:

```bash
# Run a Tessera program with profiling enabled
tessera-prof my_model.py --metrics=flops,bandwidth,occupancy
```

---
