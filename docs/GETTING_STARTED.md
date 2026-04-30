---
status: Informative
classification: Informative
authority: Entry-point orientation; defers all API and spec claims to docs/CANONICAL_API.md and docs/spec/
last_updated: 2026-04-30
---

# Getting Started with Tessera

Tessera is a pre-alpha, tile-centric programming model and compiler for deep
learning and HPC. This page gets you from zero to a running compiled function
in five minutes.

---

## Prerequisites

- Python 3.10+
- pip

GPU execution requires CUDA 12+ (NVIDIA SM_90 / Hopper or newer for full
feature support). All examples here run on CPU so no GPU is needed to start.

---

## Install

```bash
git clone <tessera-repo>
cd tessera
pip install -e ".[dev]"
```

Verify the install:

```python
import tessera
print(tessera.__version__)
```

---

## Your First Tessera Function

```python
import tessera
import numpy as np

@tessera.jit
def add_one(x: tessera.Tensor["B", "D"]):
    return tessera.ops.gelu(x)

# Inspect the emitted Graph IR
print(add_one.graph_ir.to_mlir())
```

---

## Matrix Multiply with Shape Constraints

```python
import tessera

@tessera.jit
def gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    return tessera.ops.gemm(A, B)
```

`tessera.require` checks constraints at decoration time — before any data is
passed. A violation raises `TesseraConstraintError` immediately.

---

## Distributed Array

```python
import tessera

D    = tessera.domain.Rect((4, 128, 256))          # logical shape
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))  # placement strategy
X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

print(X.shape)                     # (4, 128, 256) — global shape
print(X.shard_spec.mesh_axes)      # ("dp", "tp")
print(X.dtype)                     # "bf16"
```

---

## Region Privileges

Region annotations let the compiler track read/write intent at compile time,
not at runtime.

```python
import tessera

@tessera.jit
def step(W: tessera.Region["read"],
         X: tessera.Region["read"],
         Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(X, W)
```

Valid modes: `"read"`, `"write"`, `"reduce_sum"`, `"reduce_max"`, `"reduce_min"`.

---

## Effect System

Effects are **inferred**, not declared. A function that calls `dropout` is
automatically tagged `random`. Use `@tessera.jit(deterministic=True)` to
forbid random effects:

```python
import tessera

@tessera.jit(deterministic=True, seed=42)
def stable_fwd(x: tessera.Tensor["B", "D"]):
    return tessera.ops.layer_norm(x)   # pure — OK under deterministic
```

---

## GPU Target (Phase 3, SM_90+)

```python
import tessera
from tessera.compiler.gpu_target import GPUTargetProfile, ISA

@tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4))
def flash_attn_fwd(Q, K, V):
    tessera.require(tessera.constraint.Divisible("D", 64))
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

---

## Run the Tests

```bash
# All phases
pytest tests/ -v

# Phase 1 only (Python frontend — no GPU needed)
pytest tests/unit/ -v

# Type check
mypy python/tessera/
```

---

## Where to Go Next

| Goal | Document |
|------|----------|
| Current API reference | [`docs/CANONICAL_API.md`](CANONICAL_API.md) |
| Compiler architecture | [`docs/architecture/README.md`](architecture/README.md) |
| All public Python symbols | [`docs/spec/PYTHON_API_SPEC.md`](spec/PYTHON_API_SPEC.md) |
| What is conformant today? | [`docs/spec/CONFORMANCE.md`](spec/CONFORMANCE.md) |
| Profiling and autotuning | [`docs/guides/Tessera_Profiling_And_Autotuning_Guide.md`](guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| Error codes and diagnostics | [`docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md`](guides/Tessera_Error_Handling_And_Diagnostics_Guide.md) |
| Memory model and layouts | [`docs/guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md`](guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md) |
| Full documentation map | [`docs/README.md`](README.md) |
| Glossary of terms | [`docs/GLOSSARY.md`](GLOSSARY.md) |
