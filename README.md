# Tessera

**Pre-alpha. Breaking changes expected. Not production-ready.**

Tessera is a **standalone, tile-centric programming model and compiler** for
deep learning and HPC. It makes tiles, explicit memory spaces, numerical
precision, and distributed parallelism first-class compiler objects rather than
runtime heuristics.

**Standalone means runtime-independent of PyTorch, JAX, and Flax.** Those are
reference vocabularies only — the Tessera runtime never imports them. The
Python frontend, training step, data pipeline, custom-op API, and AOT export
are all in-scope (S-series, see below); file-format compatibility (e.g.
SentencePiece protobufs, SafeTensors, GGUF) is the single permitted concession.

Tessera also hosts compiler-native mathematical IR surfaces for structured
models. The active Clifford / geometric algebra (`tessera.ga`) and
energy-based model (`tessera.ebm`) tracks make algebra signatures,
multivector grades, energy minimization, sampling loops, and manifold-aware
integrators visible to the compiler rather than treating them as plain tensor
conventions.

Target work exists for NVIDIA, AMD ROCm, Google TPU, Apple Silicon, Tenstorrent
Metalium, Cerebras, Rubin CPX, and x86 AMX/AVX512. Backend maturity varies by
target: x86 AMX and Apple Silicon (CPU + GPU) execute natively today; NVIDIA
and ROCm have toolchain-pinned Target IR + lit fixtures with native execution
gated on real hardware (Phase G/H — see
[`docs/audit/phase_ghi_hardware_frontier.md`](docs/audit/phase_ghi_hardware_frontier.md));
other paths are artifact-only or lit-testable. See
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

Current high-level status (as of May 31, 2026):

| Area | Status |
|------|--------|
| Python frontend, textual DSL frontend, constraints, effects, Graph IR | implemented |
| Object-backed Graph / Schedule / Tile IR + per-target Target IR artifacts | implemented / lit-testable |
| **x86 AMX BF16 + AVX512 lowering and execution** (Phase 2) | **implemented / hardware-runtime** |
| **Apple Silicon CPU** via Accelerate (cblas_sgemm rank-2/rank-3 + BNNS f16/bf16) (Phase 8.2) | **implemented / hardware-runtime** |
| **Apple Silicon GPU** via MPS, MPSGraph, custom MSL, and additive Metal 4 lanes — 159 Apple runtime C ABI symbols, 82 Apple GPU kernel families, GA/EBM/M7 fused kernels, MTL4 `matmul2d` bf16 default routing, MTL4 epilogue/session/archive paths, and batched linalg MSL kernels | **implemented / hardware-runtime (Darwin); non-Darwin stubs are CI fallbacks, not hardware proof** |
| NVIDIA SM_90+ FA-4, WGMMA/TMA, Blackwell TCGEN05/TMEM target artifacts; **CUDA 13.2 Update 1 toolchain pin** | implemented / lit-testable; execution gated on real hardware (Phase G) |
| ROCm MFMA gfx90a / gfx94x / gfx950 / gfx1100; **ROCm 7.2.3 toolchain pin** | implemented / lit-testable; execution gated on real hardware (Phase H) |
| TPU target profile and StableHLO / Shardy artifacts | implemented / lit-testable |
| Distributed APIs, cyclic sharding, NCCL/RCCL adapters (≥ 2.22 pin) | implemented / scaffolded |
| Solver, sparse/RNG, linalg, scaling-resilience, **spectral (all 6 passes shipped)**, TPP | implemented / lit-testable |
| **S-series standalone compiler track** (S0–S15): RNG, state/pytrees, control flow, sharding, NN functional, quantization, optimizers, losses, **`tessera.rl` PPO/GRPO/CISPO**, AOT export, custom-primitive API, dataset combinators + tokenizers | implemented (Python reference); 432 entries × 12 contract axes tracked in `primitive_coverage.py` |
| **Reasoning-model attention family** — DeepSeek sparse attention, MiniMax Lightning, Kimi-Delta, gated/hybrid/MLA decode + RL post-training losses, all with VJP+JVP | implemented / lit-testable |
| Clifford / geometric algebra Python surface, autodiff registry, dialect, lowering passes, and Apple GPU fused kernels | implemented / lit-testable; **17/17 Apple GPU GA primitives benchmarked** |
| Energy-based model Python surface, samplers, losses, partition estimators, dialect, annotation passes, and Apple GPU kernels | implemented / lit-testable; **9/9 native Apple GPU EBM ops benchmarked** (incl. `ebm_partition_exact` via stable-logsumexp MSL kernel) |
| Runtime C ABI and Python wrapper | mock-runtime; hardware-runtime when C runtime is built |
| Cerebras WSE-3, Tenstorrent Metalium, Rubin CPX backend trees | scaffolded / lit-testable |
| **Audit-as-data infrastructure** — 5 manifest families + 17 op-level audit dashboards drift-gated by `tests/unit/` | implemented |

The 5,750-test fast unit suite passes under `-m "not slow"` in ~4 minutes;
the full Python suite collects ~6,530 tests including heavy benchmark contracts.

### Current Source and Documentation Health

This repository is moving quickly; prefer generated audits and executable tests
over phase prose when they disagree. As of the May 31, 2026 source review:

| Surface | Health |
|---------|--------|
| Source and static checks | `mypy` ratchet is clean (`errors=0`), and the docs/runtime ABI/surface/Apple-target audit slice passes (`82 passed, 1 skipped`). |
| Runtime ABI inventory | Drift-gated and current: `docs/audit/generated/runtime_abi.md` reports 170 `tessera_*` C ABI symbols, 159 Apple symbols, and 82 Apple GPU kernel families. |
| Apple backend | The source has moved beyond the older Phase 8.4.7 overview: MPS/MPSGraph remain the default lanes, while Metal 4 is additive for bf16/f16 `matmul2d`, fused epilogues, resident MLP sessions, pipeline archives, opt-in conv2d, and control-flow experiments. See `docs/apple_gpu_metal4_adoption.md`, `docs/apple_backend_integration_review.md`, and `docs/apple_gpu_kernel_inventory.md`. |
| Known Apple test gaps | The local Apple slice still exposes unstable or platform-sensitive paths: non-Darwin f16 `bmm` and `conv2d` stubs return zeros, Metal 4 `DeviceTensor` session tests assume unavailable resident tensors, and bf16 P6 epilogue tests need the numerical contract/tolerance tightened. Treat these as active blockers before claiming green Apple CI. |
| Documentation | The freshness dashboard is healthy (63 docs catalogued, 59 dated within 30 days, 4 undated), but semantic freshness is uneven. Generated dashboards and `docs/README.md` are more reliable than older narrative pages such as `docs/apple_gpu_overview.md` until those are updated for Metal 4 and batched linalg. |

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

The Python compiler carries object models and verifier checks for Graph IR,
Schedule IR, Tile IR, and CPU/x86, NVIDIA/CUDA, Apple, and ROCm Target IR.
The JIT artifact spine emits textual MLIR-like inspection strings from those
objects; native hardware execution remains target-specific and is claimed only
where backend docs say so.

Primary named pipelines and target paths are tracked in
[`docs/spec/COMPILER_REFERENCE.md`](docs/spec/COMPILER_REFERENCE.md). The
canonical pipelines registered in `tessera-opt` today:

| Pipeline | Status |
|------|--------|
| `tessera-lower-to-x86` | implemented (hardware-runtime via AMX) |
| `tessera-lower-to-gpu` (NVIDIA SM_90 default) | implemented / lit-testable |
| `tessera-nvidia-pipeline-{sm90,sm100,sm120}` (per-SM aliases) | implemented / lit-testable |
| `tessera-lower-to-rocm` | implemented / lit-testable / artifact-only |
| `tessera-lower-to-apple_cpu` (artifact) / `tessera-lower-to-apple_cpu-runtime` (Accelerate) | implemented / hardware-runtime |
| `tessera-lower-to-apple_gpu` (artifact) / `tessera-lower-to-apple_gpu-runtime` (MPS + custom MSL) | implemented / hardware-runtime |
| `tessera-lower-to-metalium` | scaffolded / target-contract artifacts |
| `tpp-space-time` (Tensor Parallel Primitives) | implemented / lit-testable |
| `ts-spectral-pipeline` (Spectral / FFT) | implemented / lit-testable |
| `tessera-cpx-pipeline` / `tessera-cpx-context-pipeline` (NV Rubin CPX) | implemented (separate `tessera-cpx-opt` driver) |

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
- **Reasoning-model attention family + RL** — DeepSeek sparse attention, MiniMax
  Lightning, Kimi-Delta, gated/hybrid/MLA decode (each with VJP+JVP); `tessera.rl`
  ships PPO/GRPO/CISPO policy losses for post-training. Backed by
  `src/transforms/lib/AttentionFamilyPasses.cpp`.

Key documents:

| Document | What it covers |
|----------|----------------|
| [`docs/spec/CLIFFORD_SPEC.md`](docs/spec/CLIFFORD_SPEC.md) | Clifford signatures, multivector type contract, GA ops, autodiff, dialect, and lowering |
| [`docs/spec/EBM_SPEC.md`](docs/spec/EBM_SPEC.md) | Energy primitive contract, inner-loop schedule, training losses, and EBM IR mapping |
| [`docs/spec/GA_EBM_EXECUTION_STATUS.md`](docs/spec/GA_EBM_EXECUTION_STATUS.md) | Layered status for Python reference, MLIR/lit, manifests, and native execution |
| [`docs/status/ga_ebm_milestone.md`](docs/status/ga_ebm_milestone.md) | Canonical GA/EBM native milestone status, health check, non-claims, and next targets |
| [`docs/audit/ga_ebm_roadmap.md`](docs/audit/ga_ebm_roadmap.md) | Sprint roadmap and acceptance history for the GA + EBM tracks |

Native execution status is layer-specific:

| Component | Python reference | MLIR / lit | Backend manifest | Native execution |
|-----------|------------------|------------|------------------|------------------|
| GA signature, multivector values, ops, calculus, and autodiff | implemented | implemented / lit-testable for dialect and lowering fixtures | implemented for registered `clifford_*` primitives | 17/17 Apple GPU native primitives benchmarked; x86 and Apple CPU are reference-first; NVIDIA/ROCm planned |
| EBM energy primitives, samplers, partition estimators, losses, and manifold-aware sampling | implemented | implemented / lit-testable for dialect and annotation-pass fixtures | implemented for registered `ebm_*` primitives | **9/9 native Apple GPU ops benchmarked**: `inner_step`, `refinement`, `langevin_step`, `decode_init`, `bivector_langevin`, `sphere_langevin`, `self_verify`, quadratic `energy`, `partition_exact` (stable-logsumexp MSL kernel landed 2026-05-17) |
| GA/EBM composite workloads | implemented as deterministic benchmark workloads | n/a | uses GA/EBM manifest-resolved symbols where native | `ga_feature_pipeline`, `ebt_tiny_refinement`, and opt-in `--ebt-sweep` benchmarked with Apple GPU and Python-reference rows |

---

## Audit-as-Data

Status truth for many areas of the repo is rendered from machine-readable
registries, not prose. The 5 **surface manifests** under
`python/tessera/compiler/`:

- `examples_manifest.py` → [`docs/audit/generated/examples_status.md`](docs/audit/generated/examples_status.md)
- `benchmarks_manifest.py` → [`docs/audit/generated/benchmarks_status.md`](docs/audit/generated/benchmarks_status.md)
- `research_manifest.py` → [`docs/audit/generated/research_status.md`](docs/audit/generated/research_status.md)
- `tools_manifest.py` → [`docs/audit/generated/tools_status.md`](docs/audit/generated/tools_status.md)
- `tests_manifest.py` → [`docs/audit/generated/tests_status.md`](docs/audit/generated/tests_status.md)

Plus 13 op-level / compiler-level audit registries covering primitive
coverage (`primitive_coverage.py` — 432 entries × 12 contract axes), backend
kernel manifests, MLIR verifier coverage, dialect registration, named-pipeline
registry, diagnostic codes, docs freshness, effect lattice, runtime C ABI
surface, test coverage by op family, TSOL canonical-op coverage, and
Apple GPU/CPU target maps. Each dashboard is drift-gated by a test in
`tests/unit/`. When you wonder "is X tested / shipped / supported?", the
answer is one of these registries.

CLIs that consume them:

- `tessera-surface-audit --surface=<examples|benchmarks|research|tools|tests>`
- `tessera-claim-lint --surface=<…>`
- `tessera-apple-target-map`, `tessera-gpu-target-map`
- `tessera-e2e-coverage`, `tessera-examples-audit`
- `tessera-operator-benchmarks-coverage`

---

## Documentation

**Start here:** [`examples/getting_started/compile_and_explain.py`](examples/getting_started/compile_and_explain.py)
is the canonical compiler tour — `@tessera.jit` → `fn(...)` →
`fn.explain()` → `tessera.compiler.support(op)` → `tessera.from_text(...)`
in ~80 lines.  Runs on CPU, no accelerator required.

| Document | What it covers |
|----------|----------------|
| [`docs/README.md`](docs/README.md) | Documentation authority tree and status labels |
| [`docs/CANONICAL_API.md`](docs/CANONICAL_API.md) | Public API names and syntax |
| [`docs/spec/PYTHON_API_SPEC.md`](docs/spec/PYTHON_API_SPEC.md) | Public Python symbols and signatures |
| [`docs/spec/COMPILER_REFERENCE.md`](docs/spec/COMPILER_REFERENCE.md) | IR stack, pass registry, pipelines, compiler source map |
| [`docs/spec/AUTODIFF_SPEC.md`](docs/spec/AUTODIFF_SPEC.md) | Tape-based reverse-mode autodiff (Tier 2) design |
| [`docs/spec/CLIFFORD_SPEC.md`](docs/spec/CLIFFORD_SPEC.md) | Clifford / geometric algebra primitive surface |
| [`docs/spec/EBM_SPEC.md`](docs/spec/EBM_SPEC.md) | Energy-based model primitive surface |
| [`docs/spec/GA_EBM_EXECUTION_STATUS.md`](docs/spec/GA_EBM_EXECUTION_STATUS.md) | GA + EBM execution status by implementation layer |
| [`docs/spec/LOWERING_PIPELINE_SPEC.md`](docs/spec/LOWERING_PIPELINE_SPEC.md) | Pass contracts and invariants |
| [`docs/spec/TARGET_IR_SPEC.md`](docs/spec/TARGET_IR_SPEC.md) | Schedule, Tile, and Target IR dialect details |
| [`docs/spec/RUNTIME_ABI_SPEC.md`](docs/spec/RUNTIME_ABI_SPEC.md) | Runtime C ABI |
| [`docs/reference/tessera_tensor_attributes.md`](docs/reference/tessera_tensor_attributes.md) | Normative tensor attributes + dtype names (six attributes, canonical/alias/planned-gated dtype sets, promotion rules) |
| [`docs/audit/execution_roadmap.md`](docs/audit/execution_roadmap.md) | Phases A–I + S-series S0–S15 standalone compiler track with per-task acceptance criteria |
| [`docs/audit/phase_ghi_hardware_frontier.md`](docs/audit/phase_ghi_hardware_frontier.md) | Hardware-gated frontier — what's blocked on real NVIDIA / ROCm / Metalium |
| [`docs/architecture/README.md`](docs/architecture/README.md) | Architecture guide index |
| [`docs/guides/Tessera_Developer_Frontend_End_To_End.md`](docs/guides/Tessera_Developer_Frontend_End_To_End.md) | First executable frontend path and IR inspection |
| [`docs/apple_gpu_overview.md`](docs/apple_gpu_overview.md) | Apple GPU architecture story; useful background, but not fully current for Metal 4 |
| [`docs/apple_gpu_metal4_adoption.md`](docs/apple_gpu_metal4_adoption.md) | Current Metal 4 ladder and coexistence model |
| [`docs/apple_backend_integration_review.md`](docs/apple_backend_integration_review.md) | Apple backend health review, optimization gaps, and Metal 4 grounding |
| [`docs/apple_gpu_kernel_inventory.md`](docs/apple_gpu_kernel_inventory.md) | Current Apple GPU C ABI/kernel inventory |

---

## Build & Test

```bash
# Python development install
pip install -e ".[dev]"

# Daily edit-loop sanity check (~5,750 fast tests, ~4 min, < 512 MB RAM)
pytest tests/unit/ -m "not slow" -q

# Full Python suite including heavy benchmarks (~6,530 collected; ~30 min)
pytest tests/unit/ -q

# GA + EBM native Apple GPU health check; skip-recording on non-Darwin
python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci

# Optional performance tests
TESSERA_RUN_PERFORMANCE_TESTS=1 scripts/test.sh

# CPU validation spine: versions, unit tests, runtime and benchmark smokes,
# standalone runtime/profiler builds, collectives compile check, audit lane
scripts/validate.sh

# Per-target release gate (the canonical Apple release blocker)
scripts/release_gate.py --target=apple_gpu

# Type check
mypy python/tessera/
```

C++/MLIR builds require LLVM/MLIR 22 or newer:

```bash
# Convenience build wrapper; defaults to CPU-only unless CUDA/HIP is enabled
scripts/build.sh

# Or configure explicitly, for example on macOS with Homebrew LLVM 22
LLVM_PREFIX="$(brew --prefix llvm)"
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$LLVM_PREFIX" \
  -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir" \
  -DTESSERA_CPU_ONLY=ON

cmake --build build --parallel

# MLIR lit tests require a built tessera-opt on PATH
python -m lit tests/tessera-ir/ -v

# If the Python lit package is not installed, run a focused Apple contract
# directly with the LLVM FileCheck installed alongside Homebrew LLVM:
build/tools/tessera-opt/tessera-opt tests/tessera-ir/phase8/apple_gpu_lowering.mlir \
  -tessera-lower-to-apple_gpu --allow-unregistered-dialect \
  | "$LLVM_PREFIX/bin/FileCheck" tests/tessera-ir/phase8/apple_gpu_lowering.mlir
```

NVIDIA / ROCm toolchain checks (skip cleanly when toolchains absent):

```bash
# Validate CUDA 13.2 U1 PTX patterns against installed nvcc
python scripts/validate_nvcc_compile.py

# Validate ROCm 7.2.3 AMDGCN intrinsics against installed hipcc
python scripts/validate_hipcc_compile.py

# Probe NCCL/RCCL ≥ 2.22 symbols at runtime
python scripts/probe_collective_libs.py
```

Documentation checks:

```bash
scripts/lint_docs.sh
```

---

## Project Layout

For the full canonical layout see [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).
A compressed view of the active surface:

```text
python/tessera/
  __init__.py            Public surface — re-exports core, jit, dist, ops, dtype, …
  core/                  Tensor, Module, fundamental abstractions
  compiler/              @jit, textual frontend, IR objects, target maps, audit modules
  distributed/           Region, domain, dist, array, shard, index_launch, MoE helpers
  nn/                    Stateful module / layers / functional / utils
  autodiff/              Tape, VJPs, JVPs, mixed-precision, rematerialize (Tier 2)
  cache/                 KVCacheHandle + MemoryStateHandle persistent state ABI
  ebm/                   Energy-based model primitives (M6)
  ga/                    Geometric Algebra / Clifford primitives (M5/M7)
  state/                 Pytree primitives + state-collection taxonomy
  testing/               Mock collectives and Python test helpers
  cli/                   tessera-mlir, tessera-prof, tessera-translate,
                         tessera-runtime-smoke, tessera-surface-audit,
                         tessera-claim-lint, tessera-{apple,gpu}-target-map,
                         tessera-e2e-coverage, tessera-examples-audit,
                         tessera-operator-benchmarks-coverage, tessera-autotune
  runtime.py             ctypes wrapper over the runtime C ABI
  profiler.py            Runtime profiler facade
  autotune.py            Public autotuning facade
  dtype.py               Canonical dtype enforcement + Dtype + result_type
  diagnostics.py         ErrorReporter, ShapeInferenceEngine
  debug.py               DebugTrace, GraphTrace, check_grad, check_determinism
  telemetry.py           Shared telemetry event/report schema
  # S-series reference surface modules:
  aot.py · checkpoint.py · control.py · custom.py · data.py · losses.py ·
  memory.py · optim.py · quantization.py · rl.py · rng.py · sharding.py
  # Domain modules:
  complex.py · conformal_advanced.py · contour.py · distributions.py ·
  energy.py · flow.py · hyperbolic.py · riemann_surface.py · server.py …

src/
  compiler/ir/           Core Tessera Graph IR ODS + C++ dialect sources
  compiler/mlir/         MLIR plugin integration
  compiler/programming_model/  Schedule / programming-model IR
  compiler/tile_opt_fa4/ FA-4 attention and queue dialects/passes
  compiler/tessera_neighbors/  Neighbor / halo / stencil dialect and passes
  compiler/codegen/      x86, NVIDIA, ROCm, TPU, Apple, Metalium, Cerebras, Rubin CPX
  compiler/diagnostics/  ErrorReporter, ShapeInferencePass
  compiler/autotuning/   Autotuner v1 framework
  collectives/           Collective IR + NCCL/RCCL adapters + chunk planner
  runtime/               Runtime C ABI, CPU/CUDA/HIP backends, scheduler tests
  solvers/               core, linalg, scaling_resilience, spectral, tpp,
                         clifford (M5), ebm (M6)
  transforms/            Canonicalization, lowering, named pipelines

tests/
  unit/                  Python unit and compiler-contract tests
  tessera-ir/            MLIR lit tests by phase / path
  kernel_tests/          System-level kernel and roofline scaffolds
  performance/           Optional performance sweeps
  tessera_numerical_validation/ Numerical reference validation
  integration/, regression/, tessera_tests/

docs/
  spec/                  Normative specs (14 files)
  architecture/          Design documents
  guides/                11 developer guides (~3,400 LOC)
  programming_guide/     11-chapter user manual + Appendix NVL72
  reference/             Tensor attribute + dtype reference, migration guide
  operations/            Canonical operation catalog (TSOL)
  audit/                 Audit reports + generated/ dashboards (drift-gated)
  status/                Per-milestone status docs
  tutorials/             Flash Attention, performance tuning
  context/               Auto-generated context outputs
  benchmarks/            Benchmark-related documentation
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
6. **Tessera is a standalone compiler.** PyTorch, JAX, and Flax are reference
   vocabularies only — never imported by the runtime. File-format compatibility
   (SafeTensors, GGUF, SentencePiece protobufs) is the single permitted
   concession.
7. Each backend exposes a **hardware-free Target IR** layer between Tile IR
   and final emission (e.g. `tessera_rocm.mfma`, `tessera_apple.gpu.metal_kernel`).
   This is what makes backends lit-testable and what
   `tests/unit/test_target_ir_contract.py` validates.
8. **`primitive_coverage.py` is the standalone compiler's audit truth.**
   Adding a primitive means updating both the runtime catalog (op_catalog.py)
   and the audit registry; (V)JP / (J)VP registration auto-promotes the
   matching contract axes.
9. Generated dashboards under `docs/audit/generated/` are **not edited by
   hand** — every one is drift-gated by a test that compares the live
   registry to the on-disk snapshot.

For the full design rationale (22 numbered decisions), see
[`CLAUDE.md`](CLAUDE.md).

---

## License

Apache License 2.0 - see [LICENSE](LICENSE).
