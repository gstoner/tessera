# Tessera

**Pre-alpha. Breaking changes expected. Not production-ready.**

Tessera is a tile-centric programming model and compiler for deep learning and
HPC. It makes tiles, explicit memory spaces, numerical precision, and
distributed parallelism first-class compiler objects rather than runtime
heuristics.

Target work exists for NVIDIA, AMD ROCm, Google TPU, Apple Silicon, Tenstorrent
Metalium, Cerebras, Rubin CPX, and x86 AMX/AVX512. Backend maturity varies by
target: some paths execute through CPU/mock runtime support, while others are
currently artifact-only or lit-testable compiler paths. See
[`docs/README.md`](docs/README.md) for the status labels used across docs.

---

## What Tessera Is

Tessera replaces thread-level GPU programming with a tile-first abstraction.
Programmers express computation in terms of tiles, groups, domains, and meshes.
The compiler handles thread mapping, memory staging, pipeline scheduling, and
collective insertion where those lowering paths are implemented.

```python
import tessera

D = tessera.domain.Rect((4, 128, 256))
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

@tessera.jit
def step(W: tessera.Region["read"],
         X: tessera.Region["read"],
         Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(X, W)

@tessera.kernel
def tp_gemm(A: tessera.f16[..., ...],
            B: tessera.f16[..., ...],
            C: tessera.mut_f32[..., ...]):
    C[:] = tessera.ops.gemm(A, B)

tessera.index_launch(axis="tp")(tp_gemm)(
    X.parts("tp"), X.parts("tp"), X.parts("tp")
)
```

---

## Development Status

Use these status words consistently:

| Status | Meaning |
|--------|---------|
| implemented | Source exists in the active tree and has unit or lit coverage. |
| lit-testable | Compiler/dialect behavior is covered by MLIR lit or target-contract tests; native execution is not implied. |
| mock-runtime | Runtime API works through a deterministic mock or CPU fallback for development and tests. |
| hardware-runtime | Native runtime execution is wired for the backend and has a concrete build/test path. |
| scaffolded | Directory, API shape, or design skeleton exists, but behavior is incomplete or artifact-only. |
| planned | Design direction only. |

Current high-level status:

| Area | Status |
|------|--------|
| Python frontend, `@tessera.jit`, constraints, effects, Graph IR | implemented |
| CPU/x86 lowering artifacts and NumPy-backed execution path | implemented / mock-runtime |
| NVIDIA SM90+ FA-4 and target artifacts | implemented / lit-testable |
| Distributed APIs, cyclic sharding, collectives scaffolding | implemented / scaffolded |
| TPU target profile and StableHLO/Shardy artifacts | implemented / lit-testable |
| Solver, sparse/RNG, linalg, resilience, and autotuning foundations | implemented / lit-testable |
| Runtime C ABI and Python wrapper | mock-runtime; hardware-runtime when C runtime is built |
| ROCm, Metalium, Apple, Cerebras, Rubin CPX backend trees | scaffolded / lit-testable unless backend docs say otherwise |

---

## Architecture

Tessera compiles through a four-layer IR stack:

```text
Python API  (@tessera.jit, Region[...], tessera.domain, index_launch)
     |
     v
Graph IR    (tessera dialect: mathematical ops, effects, shard attrs)
     |
     v
Schedule IR (schedule.* dialect: mesh regions, pipeline stages)
     |
     v
Tile IR     (tile.*, tessera.attn.*, tessera.queue.*)
     |
     v
Target IR   (backend-specific artifacts: x86, NVIDIA, ROCm, TPU, Apple, ...)
```

Primary named pipelines and target paths are tracked in
[`docs/spec/COMPILER_REFERENCE.md`](docs/spec/COMPILER_REFERENCE.md). The most
common paths are:

| Path | Status |
|------|--------|
| `tessera-lower-to-x86` | implemented |
| `tessera-lower-to-gpu` | implemented / lit-testable |
| `tessera-lower-to-rocm` | lit-testable / artifact-only |
| `tessera-lower-to-apple_cpu` | lit-testable / artifact-only |
| `tessera-lower-to-apple_gpu` | lit-testable / artifact-only |
| `tessera-lower-to-metalium` | scaffolded / target-contract artifacts |

---

## Documentation

| Document | What it covers |
|----------|----------------|
| [`docs/README.md`](docs/README.md) | Documentation authority tree and status labels |
| [`docs/CANONICAL_API.md`](docs/CANONICAL_API.md) | Public API names and syntax |
| [`docs/spec/PYTHON_API_SPEC.md`](docs/spec/PYTHON_API_SPEC.md) | Public Python symbols and signatures |
| [`docs/spec/COMPILER_REFERENCE.md`](docs/spec/COMPILER_REFERENCE.md) | IR stack, pass registry, pipelines, compiler source map |
| [`docs/spec/LOWERING_PIPELINE_SPEC.md`](docs/spec/LOWERING_PIPELINE_SPEC.md) | Pass contracts and invariants |
| [`docs/spec/TARGET_IR_SPEC.md`](docs/spec/TARGET_IR_SPEC.md) | Schedule, Tile, and Target IR dialect details |
| [`docs/spec/RUNTIME_ABI_SPEC.md`](docs/spec/RUNTIME_ABI_SPEC.md) | Runtime C ABI |
| [`docs/architecture/README.md`](docs/architecture/README.md) | Architecture guide index |
| [`docs/guides/Tessera_Developer_Frontend_End_To_End.md`](docs/guides/Tessera_Developer_Frontend_End_To_End.md) | First executable frontend path and IR inspection |

---

## Build & Test

```bash
# Python development install
pip install -e ".[dev]"

# Python unit tests configured by pyproject.toml
pytest tests/unit -v

# Optional performance tests
TESSERA_RUN_PERFORMANCE_TESTS=1 scripts/test.sh

# CPU validation spine: versions, unit tests, runtime and benchmark smokes,
# standalone runtime/profiler builds, and collectives compile check
scripts/validate.sh

# Type check
mypy python/tessera/
```

C++/MLIR builds require LLVM/MLIR 21 or newer:

```bash
# Convenience build wrapper; defaults to CPU-only unless CUDA/HIP is enabled
scripts/build.sh

# Or configure explicitly, for example on macOS with Homebrew LLVM 21
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm@21 \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/mlir \
  -DTESSERA_CPU_ONLY=ON

cmake --build build --parallel

# MLIR lit tests require a built tessera-opt on PATH
python -m lit tests/tessera-ir/ -v

# If the Python lit package is not installed, run a focused Apple contract
# directly with the LLVM FileCheck installed alongside Homebrew LLVM:
build/tools/tessera-opt/tessera-opt tests/tessera-ir/phase8/apple_gpu_lowering.mlir \
  -tessera-lower-to-apple_gpu --allow-unregistered-dialect \
  | /opt/homebrew/opt/llvm@21/bin/FileCheck tests/tessera-ir/phase8/apple_gpu_lowering.mlir
```

Documentation checks:

```bash
scripts/lint_docs.sh
```

---

## Project Layout

```text
python/tessera/
  compiler/              @jit, constraints, effects, Graph IR, targets, pipelines
  distributed/           Region, domain, dist, array, shard, index_launch, MoE helpers
  testing/               Mock collectives, compiler and QA helpers
  cli/                   tessera-mlir, tessera-prof, tessera-runtime-smoke
  runtime.py             Python wrapper over the runtime C ABI with mock fallback
  profiler.py            Runtime profiler facade
  autotune.py            Public autotuning facade

src/
  compiler/ir/           Core Tessera Graph IR ODS and C++ dialect sources
  compiler/mlir/         MLIR plugin integration
  compiler/programming_model/  Schedule/programming-model IR
  compiler/tile_opt_fa4/ FA-4 attention and queue dialects/passes
  compiler/tessera_neighbors/  Neighbor/halo/stencil dialect and passes
  compiler/codegen/      x86, NVIDIA, ROCm, TPU, Apple, Metalium, Cerebras, Rubin CPX
  collectives/           Collective IR and runtime scaffolding
  runtime/               Runtime C ABI, CPU/CUDA/HIP backend code, scheduler tests
  solvers/               Core solver, linalg, scaling-resilience, spectral, TPP work
  transforms/            Canonicalization and lowering passes

tests/
  unit/                  Python unit and compiler-contract tests
  tessera-ir/            MLIR lit tests by phase/path
  kernel_tests/          System-level kernel and roofline scaffolds
  performance/           Optional performance sweeps
  tessera_numerical_validation/ Numerical reference validation

docs/
  spec/                  Normative specs
  architecture/          Design documents
  guides/                Developer, runtime, profiling, QA, reliability guides
  programming_guide/     User-facing programming guide chapters
  archive/               Historical/pre-canonical material
```

---

## Key Design Decisions

1. The Python frontend is permanent; MLIR/C++ handles performance-critical
   lowering.
2. Tessera uses static compiler analysis, not a tracing JIT tier.
3. `Region[...]` is a type annotation, not a runtime wrapper.
4. Domains describe shape; distributions describe placement. They remain
   separate objects.
5. Constraint checks run at decoration time when concrete bindings are
   available.

---

## License

Apache License 2.0 - see [LICENSE](LICENSE).
