---
status: Informative
classification: Informative
authority: Entry-point orientation; defers all API and spec claims to docs/CANONICAL_API.md and docs/spec/
last_updated: 2026-06-01
---

# Getting Started with Tessera

Tessera is a pre-alpha, tile-centric programming model and compiler for deep
learning and HPC. This page gets you from zero to a running compiled function
in five minutes.

---

## Prerequisites

- Python 3.10+
- pip

GPU artifact validation is pinned to CUDA 13.3 for NVIDIA and ROCm
7.2.4 for AMD. Native GPU execution is hardware-gated by target; all examples
here run on CPU so no accelerator is needed to start.

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

> Note: `.[dev]` pulls `torch`/`transformers` as *reference* test vocabularies
> (Decision #23 — Tessera imports neither at runtime). For a lean environment,
> install only `numpy scipy ml_dtypes pyyaml click rich tqdm` plus the dev tools
> (`pytest mypy ruff lit`), as the platform setup paths below do.

---

## Developer Environment & Building the Compiler

Tessera builds on **macOS (Apple backend)** and **Ubuntu 24.04 (x86/ROCm
backend)** from one source tree. The Python flow needs only the lean deps above;
the C++ compiler (`tessera-opt` and friends) additionally needs **LLVM/MLIR 22**.

### macOS (Homebrew) — Apple backend

LLVM/MLIR 22, ninja, cmake, and the Python tooling come from Homebrew (no venv):

```bash
brew install llvm ninja cmake        # LLVM/MLIR 22.x
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir \
  -DTESSERA_CPU_ONLY=ON -DTESSERA_BUILD_APPLE_BACKEND=ON
ninja -C build tessera-opt
```

### Ubuntu 24.04 — x86 + AMD ROCm 7.2.4 backend

One script provisions LLVM/MLIR 22 (from apt.llvm.org — ROCm's bundled LLVM has
no MLIR), the base build deps, and a project-local `.venv`. It needs `sudo` for
the apt steps and is idempotent:

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

Then configure + build the compiler with the ROCm Target IR backend (ROCm
**7.2.4** at `/opt/rocm`; kernel execution is hardware-gated on a GPU + `kfd`
driver, Phase H — the build itself needs no GPU):

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-22/lib/cmake/mlir \
  -DTESSERA_ENABLE_HIP=ON \
  -DTESSERA_BUILD_ROCM_BACKEND=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm
ninja -C build tessera-opt
```

For a CPU-only Linux build (no ROCm), drop the last three flags and add
`-DTESSERA_CPU_ONLY=ON`.

> The Ubuntu venv caps `numpy<2.2`: numpy ≥2.2 ships PEP 695 `type` statements +
> stricter reduction overloads in its bundled stubs that break the mypy ratchet
> under the project's `python_version=3.10` type-check target.

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
# Daily edit-loop sanity check (skips slow benchmark contracts)
pytest tests/unit/ -m "not slow" -q

# Full unit suite, including heavier benchmark contracts
pytest tests/unit/ -q

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
