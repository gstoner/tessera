# Tessera

**Pre-alpha. Breaking changes expected. Not production-ready.**

Tessera is a tile-centric programming model and compiler for deep learning and
HPC. It makes tiles, explicit memory spaces, numerical precision, and
distributed parallelism first-class compiler objects rather than runtime
heuristics.

Tessera also hosts compiler-native mathematical IR surfaces for structured
models. The active Clifford / geometric algebra (`tessera.ga`) and
energy-based model (`tessera.ebm`) tracks make algebra signatures,
multivector grades, energy minimization, sampling loops, and manifold-aware
integrators visible to the compiler rather than treating them as plain tensor
conventions.

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

The sketch below shows the canonical Python surface; runnable end-to-end
hardware examples are called out separately in backend-specific docs and tests.

```python
import tessera

dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
A = tessera.array.from_domain(
    tessera.domain.Rect((4, 128, 64)), dtype="fp16", distribution=dist
)
W = tessera.array.from_domain(
    tessera.domain.Rect((4, 64, 256)), dtype="fp16", distribution=dist
)
Y = tessera.array.from_domain(
    tessera.domain.Rect((4, 128, 256)), dtype="fp32", distribution=dist
)

@tessera.jit
def step(W: tessera.Region["read"],
         A: tessera.Region["read"],
         Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(A, W)

@tessera.kernel
def tp_gemm(A: tessera.f16[..., ...],
            B: tessera.f16[..., ...],
            C: tessera.mut_f32[..., ...]):
    C[:] = tessera.ops.gemm(A, B)

tessera.index_launch(axis="tp")(tp_gemm)(
    A.parts("tp"), W.parts("tp"), Y.parts("tp")
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
| Python frontend, textual DSL frontend, constraints, effects, Graph IR | implemented |
| Object-backed Graph/Schedule/Tile IR and CPU/NVIDIA/Apple/ROCm Target IR artifacts | implemented / lit-testable |
| CPU/x86 lowering artifacts and NumPy-backed execution path | implemented / mock-runtime |
| NVIDIA SM90+ FA-4, WGMMA/TMA, and Blackwell TCGEN05/TMEM target artifacts | implemented / lit-testable |
| Distributed APIs, cyclic sharding, collectives scaffolding | implemented / scaffolded |
| TPU target profile and StableHLO/Shardy artifacts | implemented / lit-testable |
| Solver, sparse/RNG, linalg, resilience, and autotuning foundations | implemented / lit-testable |
| Clifford / geometric algebra Python surface, autodiff registry, dialect, lowering passes, and Apple GPU fused kernels | implemented / lit-testable; 17/17 Apple GPU native primitives benchmarked |
| Energy-based model Python surface, samplers, losses, partition estimators, dialect, annotation passes, and Apple GPU kernels | implemented / lit-testable; 6 native Apple GPU EBM ops benchmarked, 3 core rows remain Python-only |
| Runtime C ABI and Python wrapper | mock-runtime; hardware-runtime when C runtime is built |
| ROCm and Apple Target IR artifact lowering | implemented / lit-testable / artifact-only |
| Metalium, Cerebras, Rubin CPX backend trees | scaffolded / lit-testable unless backend docs say otherwise |

---

## Architecture

Tessera compiles through a four-layer IR stack:

```text
Python API + textual DSL frontend
(@tessera.jit, module/func/kernel syntax, Region[...], tessera.domain)
     |
     v
Graph IR    (tessera dialect: math ops, shape/dtype/layout metadata, diagnostics)
     |
     v
Schedule IR (schedule.* dialect: mesh.define/region, pipeline.region, stage, yield)
     |
     v
Tile IR     (tile.* ops, tessera.attn.* FA-4 ops, tessera.queue.* barriers)
     |
     v
Target IR   (backend-specific artifacts: x86, NVIDIA, ROCm, TPU, Apple, ...)
```

The Python compiler now carries object models and verifier checks for Graph IR,
Schedule IR, Tile IR, and CPU/x86, NVIDIA/CUDA, Apple, and ROCm Target IR. The JIT artifact spine emits
textual MLIR-like inspection strings from those objects; native hardware
execution remains target-specific and is claimed only where backend docs say so.

Primary named pipelines and target paths are tracked in
[`docs/spec/COMPILER_REFERENCE.md`](docs/spec/COMPILER_REFERENCE.md). The most
common paths are:

| Path | Status |
|------|--------|
| `tessera-lower-to-x86` | implemented |
| `tessera-lower-to-gpu` | implemented / lit-testable |
| `tessera-lower-to-rocm` | implemented / lit-testable / artifact-only |
| `tessera-lower-to-apple_cpu` | implemented / lit-testable / artifact-only |
| `tessera-lower-to-apple_gpu` | implemented / lit-testable / artifact-only |
| `tessera-lower-to-metalium` | scaffolded / target-contract artifacts |

---

## Mathematical IR Surfaces

Tessera's GA + EBM work is the first compiler-native track where mathematical
structure is part of the IR contract:

- **Clifford / geometric algebra (`tessera.ga`)** — v1 supports `Cl(3,0)` and
  `Cl(1,3)`, grade-aware `Multivector` values, geometric products, rotor
  sandwich operations, differential-form primitives, and a parallel geometric
  autodiff registry.
- **Energy-based models (`tessera.ebm`)** — energy primitives, inner-loop
  refinement, self-verification, Langevin / MALA / HMC / Gibbs samplers,
  partition-function estimators, EBM losses, and manifold-aware sphere /
  bivector Langevin reference paths.

Key documents:

| Document | What it covers |
|----------|----------------|
| [`docs/spec/CLIFFORD_SPEC.md`](docs/spec/CLIFFORD_SPEC.md) | Clifford signatures, multivector type contract, GA ops, autodiff, dialect, and lowering |
| [`docs/spec/EBM_SPEC.md`](docs/spec/EBM_SPEC.md) | Energy primitive contract, inner-loop schedule, training losses, and EBM IR mapping |
| [`docs/spec/GA_EBM_EXECUTION_STATUS.md`](docs/spec/GA_EBM_EXECUTION_STATUS.md) | Layered status for Python reference, MLIR/lit, manifests, and native execution |
| [`docs/audit/ga_ebm_roadmap.md`](docs/audit/ga_ebm_roadmap.md) | Sprint roadmap and acceptance history for the GA + EBM tracks |

Native execution status is layer-specific:

| Component | Python reference | MLIR / lit | Backend manifest | Native execution |
|-----------|------------------|------------|------------------|------------------|
| GA signature, multivector values, ops, calculus, and autodiff | implemented | implemented / lit-testable for dialect and lowering fixtures | implemented for registered `clifford_*` primitives | 17/17 Apple GPU native primitives benchmarked; x86 and Apple CPU are reference-first; NVIDIA/ROCm planned |
| EBM energy primitives, samplers, partition estimators, losses, and manifold-aware sampling | implemented | implemented / lit-testable for dialect and annotation-pass fixtures | implemented for registered `ebm_*` primitives | 6 native Apple GPU ops benchmarked (`inner_step`, `refinement`, `langevin_step`, `decode_init`, `bivector_langevin`, `sphere_langevin`); 3 core rows remain Python-only (`energy`, `self_verify`, `partition_exact`) |
| GA/EBM composite workloads | implemented as deterministic benchmark workloads | n/a | uses GA/EBM manifest-resolved symbols where native | `ga_feature_pipeline` and `ebt_tiny_refinement` benchmarked with Apple GPU and Python-reference rows |

---

## Documentation

| Document | What it covers |
|----------|----------------|
| [`docs/README.md`](docs/README.md) | Documentation authority tree and status labels |
| [`docs/CANONICAL_API.md`](docs/CANONICAL_API.md) | Public API names and syntax |
| [`docs/spec/PYTHON_API_SPEC.md`](docs/spec/PYTHON_API_SPEC.md) | Public Python symbols and signatures |
| [`docs/spec/COMPILER_REFERENCE.md`](docs/spec/COMPILER_REFERENCE.md) | IR stack, pass registry, pipelines, compiler source map |
| [`docs/spec/CLIFFORD_SPEC.md`](docs/spec/CLIFFORD_SPEC.md) | Clifford / geometric algebra primitive surface |
| [`docs/spec/EBM_SPEC.md`](docs/spec/EBM_SPEC.md) | Energy-based model primitive surface |
| [`docs/spec/GA_EBM_EXECUTION_STATUS.md`](docs/spec/GA_EBM_EXECUTION_STATUS.md) | GA + EBM execution status by implementation layer |
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

# GA + EBM native Apple GPU health check; skip-recording on non-Darwin
python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci

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
  compiler/              @jit, textual frontend, Graph/Schedule/Tile/Target IR, pipelines
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
