# Tessera

**Pre-alpha. Breaking changes expected. Not production-ready.**

Tessera is a tile-centric programming model and compiler for deep learning and HPC. It makes
tiles, explicit memory spaces, numerical precision, and distributed parallelism **first-class
compiler objects** rather than runtime heuristics.

Target hardware: NVIDIA (SM90 Hopper, SM100 Blackwell), AMD ROCm, Google TPU, and x86
AMX/AVX512. The x86 backend is working today. GPU and distributed backends are in progress.

---

## What Tessera Is

Tessera replaces thread-level GPU programming with a **tile-first abstraction**. Programmers
express computation in terms of tiles, groups, and meshes. The compiler handles thread mapping,
memory staging, pipeline scheduling, and collective insertion automatically.

```python
import tessera

# Shard tensors across a distributed mesh
D    = tessera.domain.Rect((4, 128, 256))
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

# Annotate access privileges — the compiler enforces them
@tessera.jit
def step(W: tessera.Region["read"],
         X: tessera.Region["read"],
         Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(X, W)

# Dispatch a kernel across tensor-parallel shards
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

| Phase | Status | What it delivers |
|-------|--------|-----------------|
| **Phase 1** | ✅ Complete | Python frontend — `@tessera.jit`, `@tessera.kernel`, `Region`, `domain`, `dist`, `DistributedArray`, `index_launch`, `ConstraintSolver`, `EffectLattice`, `GraphIRBuilder` |
| **Phase 2** | ✅ Complete | C++ lowering chain — `DistributionLoweringPass`, `EffectAnnotationPass`, `TilingPass`, `TileToX86Pass`; `tessera-lower-to-x86` pipeline |
| **Phase 3** | ✅ Complete | NVIDIA GPU backend — `GPUTargetProfile`, `TileIRLoweringPass`, `WarpSpecializationPass`, FA-4 Attn dialect, FlashAttention Tile IR lowering |
| **Phase 4** | 🔲 Next | NCCL/RCCL collectives, TPU StableHLO backend, Cyclic MoE distribution |
| **Phase 5** | 🔲 Future | Checkpointing, ZeRO optimizer sharding, Bayesian autotuner, sparse/RNG solver passes |
| **Phase 6** | 🔲 Future | ROCm MFMA full coverage, runtime C ABI wiring, benchmarks, production diagnostics |

---

## Architecture

Tessera compiles through a four-layer IR stack:

```
Python API  (@tessera.jit, Region[...], tessera.domain, index_launch)
     │
     ▼
Graph IR    (tessera dialect — mathematical ops, effects, shard attrs)
     │
     ▼
Schedule IR (schedule.* dialect — mesh regions, pipeline stages)
     │
     ▼
Tile IR     (tile.* + tessera.attn.* + tessera.queue.* — warp roles, async copy, MMA)
     │
     ▼
Target IR   (tessera.nvgpu.wgmma.*, tessera.tma.*, mbarrier → PTX / x86 AMX)
```

Two named compilation pipelines are registered:

- **`tessera-lower-to-x86`** — 5 passes, CPU target (working today)
- **`tessera-lower-to-gpu`** — 9 passes, NVIDIA SM90+ target (Phase 3 complete)

---

## Documentation

### Architecture and Design

| Document | What it covers |
|----------|---------------|
| [`docs/architecture/system_overview.md`](docs/architecture/system_overview.md) | Phase status, what works today, what doesn't |
| [`docs/architecture/Compiler/Tessera_Compiler_Architecture_Overview.md`](docs/architecture/Compiler/Tessera_Compiler_Architecture_Overview.md) | Full pipeline: Python surface → PTX |
| [`docs/architecture/Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md`](docs/architecture/Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md) | `@tessera.jit` decoration sequence, Graph IR emission |

### Normative Specs

| Document | What it covers |
|----------|---------------|
| [`docs/CANONICAL_API.md`](docs/CANONICAL_API.md) | **Single naming authority** — wins all disputes |
| [`docs/spec/PYTHON_API_SPEC.md`](docs/spec/PYTHON_API_SPEC.md) | Every public symbol, Phases 1–3 |
| [`docs/spec/COMPILER_REFERENCE.md`](docs/spec/COMPILER_REFERENCE.md) | IR stack, pass pipeline registry, architecture decisions |
| [`docs/spec/GRAPH_IR_SPEC.md`](docs/spec/GRAPH_IR_SPEC.md) | All 6 Graph IR ops, 4 canonicalization patterns |
| [`docs/spec/LOWERING_PIPELINE_SPEC.md`](docs/spec/LOWERING_PIPELINE_SPEC.md) | Every pass: input/output IR contracts, invariants |
| [`docs/spec/TARGET_IR_SPEC.md`](docs/spec/TARGET_IR_SPEC.md) | Schedule, Attn, Queue, Tile, NVIDIA-specific dialects |
| [`docs/spec/RUNTIME_ABI_SPEC.md`](docs/spec/RUNTIME_ABI_SPEC.md) | C ABI functions, types, error model, backend architecture |

### Programming Guide (11 chapters)

| Chapter | Topic |
|---------|-------|
| [Ch 1](docs/programming_guide/Tessera_Programming_Guide_Chapter1_Introduction_Overview.md) | Introduction & Overview |
| [Ch 2](docs/programming_guide/Tessera_Programming_Guide_Chapter2_Programming_Model.md) | Programming Model — tiles, groups, meshes |
| [Ch 3](docs/programming_guide/Tessera_Programming_Guide_Chapter3_Memory_Model.md) | Memory Model |
| [Ch 4](docs/programming_guide/Tessera_Programming_Guide_Chapter4_Execution_Model.md) | Execution Model |
| [Ch 5](docs/programming_guide/Tessera_Programming_Guide_Chapter5_Kernel_Programming.md) | Kernel Programming — `@tessera.kernel`, `index_launch`, Tile IR |
| [Ch 6](docs/programming_guide/Tessera_Programming_Guide_Chapter6_Numerics_Model.md) | Numerics Model |
| [Ch 7](docs/programming_guide/Tessera_Programming_Guide_Chapter7_Autodiff.md) | Autodiff (Phase 5 planned) |
| [Ch 8](docs/programming_guide/Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md) | Layouts & Data Movement |
| [Ch 9](docs/programming_guide/Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md) | Libraries & Primitives |
| [Ch 10](docs/programming_guide/Tessera_Programming_Guide_Chapter10_Portability.md) | Portability |
| [Ch 11](docs/programming_guide/Tessera_Programming_Guide_Chapter11_Conclusion.md) | Conclusion |
| [Appendix A](docs/programming_guide/Tessera_Programming_Guide_Appendix_NVL72.md) | NVL72 Guide |

### Full Doc Index

[`docs/README.md`](docs/README.md) — complete hierarchical index of all documentation.

---

## Build & Test

```bash
# Python development install
pip install -e ".[dev]"

# Run Phase 1 tests (no GPU required)
pytest tests/phase1/ -v

# Run Phase 2 tests (no GPU required)
pytest tests/phase2/ -v

# Run Phase 3 tests (GPU or mock)
pytest tests/phase3/ -v

# Run all phases
pytest tests/ -v

# Type check
mypy python/tessera/

# C++ build — CPU only
mkdir -p build && cd build
cmake .. -DTESSERA_ENABLE_CUDA=OFF -DTESSERA_CPU_ONLY=ON
make -j$(nproc)

# C++ build — with CUDA (Phase 3+)
cmake .. -DTESSERA_ENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
make -j$(nproc)

# MLIR lit tests
python -m lit tests/tessera-ir/ -v
```

---

## Key Design Decisions

These are locked — see `CLAUDE.md` for full rationale and `docs/spec/COMPILER_REFERENCE.md`
for the complete list.

1. **CPU-first, then GPU.** Phase 1–2 target x86 AMX. GPU-specific IR is gated behind
   `target_profile.isa >= ISA.SM_90`.
2. **Static AOT compilation only.** No tracing JIT. `@tessera.jit` is a decoration-time
   compiler, not a trace-and-replay system.
3. **Region is a type annotation, not a runtime wrapper.** `Region["read"]` participates in
   Python's annotation protocol and lowers to a `tessera.effect` attribute on Graph IR args.
4. **Domains and distributions are always separate.** `Rect` describes shape; `Block/Cyclic/
   Replicated` describes placement. They never merge.
5. **Constraints run at decoration time.** `@tessera.jit` inspects annotations and runs
   `ConstraintSolver` before any IR is emitted.

---

## Project Layout

```
python/tessera/         Python frontend (Phases 1–3 complete)
  distributed/          Region, domain, dist, array, index_launch
  compiler/             @jit, @kernel, ConstraintSolver, EffectLattice, GraphIRBuilder,
                        GPUTargetProfile, FlashAttnLoweringConfig
  testing/              MockRankGroup for multi-rank tests without NCCL

src/
  ir/                   TesseraOps.td — Graph IR op definitions
  transforms/lib/       All lowering passes (Phases 1–3)
  tile_opt_fa4/         FA-4 Attn + Queue dialects
  collectives/          Collective IR (Phase 4)
  scaling_resilience/   SR passes (Phase 5 stubs)
  runtime/include/      C ABI headers (Phase 6)
  compiler/codegen/     x86, NVIDIA, ROCm backends

tests/
  phase1/ … phase3/     Python test suites (all passing)
  tessera-ir/           MLIR lit tests

docs/
  spec/                 Normative specs (canonical reference)
  architecture/         Design documents
  programming_guide/    11-chapter guide
  reference/            Older reference material (pre-canonical API names)
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
