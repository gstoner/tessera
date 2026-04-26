---
status: Normative
classification: Normative
last_updated: 2026-04-26
---

# Tessera Compiler Reference
**Status:** Normative — grounded in `src/` and `python/tessera/` Phases 1–3 implementation  
**Last updated:** April 26, 2026  
**Audience:** Claude Code sessions, compiler engineers, pass authors

This is the first document to read when working on the compiler. It defines canonical names, maps the full pass pipeline, records phase status, and locks in architecture decisions. See `docs/CANONICAL_API.md` for the Python surface API.

---

## 1. The Four-Layer IR Stack

One name per layer. No aliases.

```
Python API  (@tessera.jit, Region[...], tessera.domain, index_launch)
     │
     ▼  [GraphIRBuilder.lower() — python/tessera/compiler/graph_ir.py]
     │
Graph IR    (tessera dialect — TesseraOps.td, effect/shard/privilege attrs)
     │
     ▼  [DistributionLoweringPass, EffectAnnotationPass]
     │
Schedule IR (schedule.mesh.* dialect — mesh regions, pipeline stages)
     │
     ▼  [TilingPass, TileIRLoweringPass, WarpSpecializationPass, AsyncCopyLoweringPass]
     │
Tile IR     (tile.*, tessera.attn.*, tessera.queue.* — explicit tile ops)
     │
     ▼  [TileToX86Pass | NVWGMMALoweringPass + NVTMADescriptorPass + NVFlashAttnKernelEmitter]
     │
Target IR   (per-backend: LLVM+AMX/AVX512 | NVVM/PTX | StableHLO | ROCDL)
```

| Layer | Canonical name | Primary file(s) | Key dialect(s) |
|-------|---------------|-----------------|----------------|
| Layer 1 | **Graph IR** | `src/ir/TesseraOps.td`, `src/ir/TesseraTiling.cpp` | `tessera.*` |
| Layer 2 | **Schedule IR** | `src/programming_model/ir/schedule/ScheduleMeshPipelineOps.td` | `schedule.mesh.*` |
| Layer 3 | **Tile IR** | `src/tile_opt_fa4/dialects/tessera_attn/Attn.td`, `tessera_queue/Queue.td` | `tile.*`, `tessera.attn.*`, `tessera.queue.*` |
| Layer 4 | **Target IR** | `src/compiler/codegen/tessera_x86_backend/`, `tessera_gpu_backend_NVIDIA/` | LLVM, NVVM, StableHLO |

**Naming rule:** Always write the full two-word IR names: Graph IR, Schedule IR, Tile IR, and Target IR. Avoid older abbreviated or one-word aliases.

---

## 2. Pass Pipeline Registry

All passes in pipeline order, with their source files, named pipelines, and introduction phases.

### 2.1 Named Pipelines

| Pipeline name | Target | Passes (in order) |
|--------------|--------|-------------------|
| `tessera-lower-to-x86` | x86 AMX/AVX512 CPU | EffectAnnotation → Distribution → Canonicalize → Tiling → TileToX86 |
| `tessera-lower-to-gpu` | NVIDIA SM_90+ | EffectAnnotation → Distribution → TileIRLowering → WarpSpecialization → AsyncCopyLowering → NVWGMMALowering → NVTMADescriptor → NVFlashAttnKernelEmitter |

Registration: `src/transforms/lib/Passes.cpp` (x86); `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` (GPU)

---

### 2.2 Phase 1 — Graph IR

**`CanonicalizeTesseraIR`**
- File: `src/transforms/lib/CanonicalizeTesseraIR.cpp`
- Input: Graph IR with raw tessera.* ops
- Output: Graph IR with 4 fusion patterns applied
- Patterns: (1) matmul+bias fusion, (2) matmul+gelu fusion, (3) matmul+bias+gelu fusion, (4) conv+bias fusion
- Named pipeline: both (runs first)

**`VerifyTesseraIR`**
- File: `src/transforms/lib/VerifyTesseraIR.cpp`
- Input: Any Graph IR module
- Output: Same (verification only, no transforms)
- Checks: module version attribute (`tessera.version`) only (Phase 1 scope)
- Note: Full shape/privilege verification is planned for Phase 4+

---

### 2.3 Phase 2 — Graph IR → Schedule IR (x86 path)

**`EffectAnnotationPass`**
- File: `src/transforms/lib/EffectAnnotationPass.cpp`
- Input: Graph IR — `func.func` ops with tessera.* body ops
- Output: Same, with `tessera.effect = "pure"|"random"|"memory"|"io"` on each function
- Effect inference rules:
  - `tessera.flash_attn` with `dropout_p != 0.0` → `random`
  - `tessera.copy` → `memory`
  - Any arg with `tessera.effect = "write"` or `"reduce_*"` → `memory`
  - `func.call` to non-tessera external function → `io`
  - All else → `pure`
- Validation: if a function already carries `tessera.effect = "pure"` but body infers higher, emits error
- Named pipeline: both (runs before Distribution)

**`DistributionLoweringPass`**
- File: `src/transforms/lib/DistributionLoweringPass.cpp`
- Input: Graph IR — `func.func` args with `tessera.shard` attributes
- Output: Schedule IR — `schedule.mesh.define` + `schedule.mesh.region` wrapping function bodies
- Pass options: `--mesh-axes` (comma-separated names), `--mesh-sizes` (comma-separated sizes)
- Before (Graph IR):
  ```mlir
  func.func @step(%a: tensor<128x256xbf16>
                  {tessera.shard = {axes = ["dp"], dims = [0]}}) { ... }
  ```
- After (Schedule IR):
  ```mlir
  func.func @step(%a: tensor<128x256xbf16>) {
    schedule.mesh.define {dims = [4], axis_names = ["dp"]}
    schedule.mesh.region {mesh = @dp, axis = "dp"} {
      ...
      schedule.yield
    }
  }
  ```
- Note: `schedule.yield` added as region terminator in Phase 2
- Named pipeline: both

**`TilingPass`**
- File: `src/transforms/lib/TilingPass.cpp`
- Input: Schedule IR — `tessera.matmul` ops with static-shape ranked tensor operands
- Output: Schedule IR — `scf.for` M×N loop nests over `tensor.extract_slice` / `tensor.insert_slice`
- Pass options: `--tile-m` (default 16), `--tile-n` (default 16)
- Only tiles ops with statically-shaped operands; dynamic shapes pass through unchanged
- Named pipeline: `tessera-lower-to-x86`

**`TileToX86Pass`**
- File: `src/transforms/lib/TileToX86Pass.cpp`
- Input: Tiled Schedule IR — `tessera.matmul` + `tessera.fused_epilogue` with static BF16/F16 tensors
- Output: Target IR — `func.call` to x86 backend C functions
- Lowering targets:
  - `tessera_x86_amx_gemm_bf16(aPtr, bPtr, cPtr, M, N, K, beta)` — AMX BF16
  - `tessera_x86_avx512_gemm_bf16` — AVX512 fallback
  - `tessera_x86_epilogue_bias_fp32` — bias epilogue
  - `tessera_x86_epilogue_bias_gelu_fp32` — bias+GELU epilogue
- Uses: `bufferization.to_memref`, `memref.extract_aligned_pointer_as_index`, `arith.index_cast`
- Named pipeline: `tessera-lower-to-x86`

---

### 2.4 Phase 3 — Schedule IR → Tile IR → Target IR (GPU path)

**`TileIRLoweringPass`**
- File: `src/transforms/lib/TileIRLoweringPass.cpp`
- Input: Schedule IR — `schedule.mesh.region` bodies containing `tessera.flash_attn` or `tessera.matmul`
- Output: Tile IR — `tile.async_copy`, `tessera.attn.*`, `tile.mma`, `tile.wait_async`
- For flash_attn: emits `tessera.attn.scaled_dot_product` + `tessera.attn.causal_mask?` + `tessera.attn.online_softmax` + `tessera.attn.lse_accumulate`
- Pass options: `--tile-q` (default 64), `--tile-kv` (default 64), `--sm` (target SM version int)
- Named pipeline: `tessera-lower-to-gpu`

**`WarpSpecializationPass`**
- File: `src/compiler/tile_opt_fa4/lib/WarpSpecializationPass.cpp`
- Input: Tile IR — ops in mesh.region bodies
- Output: Tile IR — `tessera.schedule.warp {role="producer"}` and `{role="consumer"}` regions with `tessera.queue` barriers between them
- Warp role separation is structural (hard boundary), not advisory
- Registration: `--tessera-warp-specialization`
- Named pipeline: `tessera-lower-to-gpu`

**`AsyncCopyLoweringPass`**
- File: `src/compiler/tile_opt_fa4/lib/AsyncCopyLoweringPass.cpp`
- Input: Tile IR — `tile.async_copy` ops
- Output: Tile IR — `tessera.tma.*` (SM_90+) or `tessera.cp_async.*` (SM < 90)
- SM_90+ path uses TMA bulk-tensor copy; pre-SM_90 uses cp.async.cg
- Registration: `--tessera-async-copy-lowering`
- Named pipeline: `tessera-lower-to-gpu`

**`NVWGMMALoweringPass`**
- File: `src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVWGMMALoweringPass.cpp`
- Input: Tile IR — `tile.mma` ops
- Output: Target IR — `tessera.nvgpu.wgmma.mma_async` inline PTX (SM_90+) or WMMA (SM < 90)
- SM_90+ emits: `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16` PTX
- Fallback: WMMA path for SM < 90
- Registration: `--tessera-nvwgmma-lowering`
- Named pipeline: `tessera-lower-to-gpu`

**`NVTMADescriptorPass`**
- File: `src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVTMADescriptorPass.cpp`
- Input: Target IR with `tessera.tma.*` ops scattered throughout loop bodies
- Output: Target IR — TMA descriptors hoisted to kernel preamble; `cp.async.bulk.tensor` referencing descriptors; mbarrier slot assignments
- TMA descriptors are generated once per kernel, not per tile
- Registration: `--tessera-nvtma-descriptor`
- Named pipeline: `tessera-lower-to-gpu`

**`NVFlashAttnKernelEmitter`**
- File: `src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVFlashAttnKernelEmitter.cpp` (conceptually; may be split)
- Input: Target IR after TMA descriptor setup
- Output: Target IR ready for PTX lowering — full FA fwd kernel with QKV tiles, online softmax, LSE accumulation, epilogue; mbarrier arrive/wait; launch bounds annotation
- Resolves scale sentinel for flash attention; attaches `nvvm.reqntid` launch bounds
- Registration: `--tessera-nvflashattn-emitter`
- Named pipeline: `tessera-lower-to-gpu`

---

### 2.5 Phase 4 Passes (Planned — Not Yet Implemented)

These passes are designed in CLAUDE.md §Phase 4 but not yet built:

| Pass | File (planned) | Purpose |
|------|---------------|---------|
| `GPUCollectiveInsertionPass` | `src/transforms/lib/GPUCollectiveInsertionPass.cpp` | Insert `collective.reduce_scatter` at DP boundaries post-backward |
| `PipelineStageInsertionPass` | `src/transforms/lib/PipelineStageInsertionPass.cpp` | 1F1B micro-batch schedule across PP stages |
| `TPUAttentionLoweringPass` | `src/compiler/codegen/Tessera_TPU_Backend/` | `tessera.flash_attn` → `stablehlo.composite` |
| `TPUShardyExportPass` | same | Shardy mesh + sharding rules export |

---

## 3. Phase Completion Status

| Phase | Status | Key deliverables |
|-------|--------|-----------------|
| **Phase 1** | ✅ Complete | Python frontend: `Region`, `domain`, `dist`, `array`, `@jit`, `@kernel`, `ConstraintSolver`, `EffectLattice`, `GraphIRBuilder` |
| **Phase 2** | ✅ Complete | C++ lowering chain: `DistributionLoweringPass`, `EffectAnnotationPass`, `TilingPass`, `TileToX86Pass`; `tessera-lower-to-x86` pipeline |
| **Phase 3** | ✅ Complete | NVIDIA GPU backend: `GPUTargetProfile`, `ISA`, `TileIRLoweringPass`, `WarpSpecializationPass`, `AsyncCopyLoweringPass`, `NVWGMMALoweringPass`, `NVTMADescriptorPass`, `NVFlashAttnKernelEmitter`; `tessera-lower-to-gpu` pipeline; FA-4 Attn dialect (`ScaledDotProduct`, `OnlineSoftmax`, `LseAccumulate`, `DropoutMask`, `CausalMask`) |
| **Phase 4** | 🔲 Next | Distributed training: NCCL/RCCL collectives, TPU StableHLO backend, Cyclic MoE, DP/TP/PP collective insertion, 1F1B pipeline |
| **Phase 5** | 🔲 Future | Scaling/Resilience passes, Bayesian autotuner v2, Solver suite (sparse, RNG, Newton autodiff) |
| **Phase 6** | 🔲 Future | ROCm full MFMA, Runtime C ABI Python wrapper, production diagnostics, benchmark suite |

---

## 4. Architecture Decisions — Do Not Revisit

These decisions are closed. Do not reopen them in new sessions.

**1. CPU-first, then GPU.**  
Phase 1 targets x86 AMX via `tessera_x86_backend`. All GPU-specific IR ops are gated behind `target_profile.isa >= ISA.SM_90`. This is deliberate and permanent.

**2. Region is a type annotation, not a runtime wrapper.**  
`Region["read"]` returns a `RegionType` object via `__class_getitem__`. It does NOT wrap tensors at runtime. Changing this would break the IR emission model.

**3. Domains and distributions are always separate.**  
`Rect` describes shape; `Block`/`Cyclic`/`Replicated` describes placement. Never merge them into a single constructor.

**4. ConstraintSolver runs at decoration time.**  
`@jit` parses the function body AST via `_ConstraintExtractor`, extracts `tessera.require()` calls, and runs `ConstraintSolver.check()` before any IR is emitted. This is deliberate for early error reporting.

**5. Effects are inferred, not declared.**  
`EffectLattice` walks the IR and infers effects. Programmers only declare `@jit(deterministic=True)` and `@jit(seed=N)` at the top level.

**6. Mock collectives use threads, not processes.**  
Phase 1 multi-rank tests run in-process with Python threads as fake ranks. Avoids NCCL/MPI dependency in the Python frontend test suite.

**7. `tessera.array` is not `numpy.ndarray`.**  
`DistributedArray` carries a `ShardSpec` and a logical shape. Physical storage is backend-dependent. In Phase 1 it is an eagerly-evaluated numpy array on CPU. Do not subclass or alias `np.ndarray`.

**8. Frontend is pure Python, permanently.**  
The Rust frontend architecture proposal (`docs/old_concepts/Rust_Frontend_Research/`) was evaluated and rejected (April 2026). The frontend is Python. The MLIR C++ pass pipeline handles performance-critical stages.

**9. Tracing JIT is not on the roadmap.**  
The multi-tier meta-tracing JIT design (`docs/old_concepts/Tracing_JIT_Research/`) is research-only. The implemented compiler is static AOT. Constraint checking at decoration time depends on static analysis.

---

## 5. GPU-Only Gated Features

The following features must be gated behind `target_profile.isa >= ISA.SM_90`. Do not emit these ops for CPU or pre-SM_90 GPU targets.

```python
if target_profile.isa >= ISA.SM_90:
    # emit warp specialization ops
    # emit TMA (cp.async.bulk.tensor) ops
    # emit WGMMA (wgmma.mma_async.sync.aligned) PTX
    # emit tessera.queue.{create, push, pop} ops
    # emit tessera.attn.* FA-4 ops (ScaledDotProduct, etc.)
```

| Feature | IR op(s) | Gate |
|---------|---------|------|
| WGMMA | `tessera.nvgpu.wgmma.mma_async` | `isa >= SM_90` |
| TMA bulk copy | `tessera.tma.*` | `isa >= SM_90` |
| Warp specialization | `tessera.schedule.warp {role=...}` | `isa >= SM_90` |
| Tile queue | `tessera.queue.{create,push,pop}` | `isa >= SM_90` |
| FA-4 Attn ops | `tessera.attn.*` | `isa >= SM_90` |
| TMEM MMA (Blackwell) | `tile.mma.tcgen05` | `isa >= SM_100` |

For SM < 90: fall back to WMMA via `NVWGMMALoweringPass` WMMA path. For SM < 90 + TMA: use `cp.async.cg` via `AsyncCopyLoweringPass` pre-SM_90 path.

---

## 6. Test Suite Map

| Phase | Python tests | MLIR lit tests |
|-------|-------------|----------------|
| Phase 1 | `tests/phase1/` (179+ tests) | `tests/tessera-ir/` (basic) |
| Phase 2 | `tests/phase2/test_lowering_chain.py` | `tests/tessera-ir/phase2/` |
| Phase 3 | `tests/phase3/` (56 tests) | `tests/tessera-ir/phase3/` |
| Phase 4 | `tests/phase4/` (planned) | `tests/tessera-ir/phase4/` (planned) |
| Phase 5 | `tests/phase5/` (planned) | `tests/tessera-ir/phase5/` (planned) |
| Phase 6 | `tests/phase6/` (planned) | `tests/tessera-ir/phase6/` (planned) |

### Running tests

```bash
# All implemented phases
pytest tests/phase1/ tests/phase2/ tests/phase3/ -v

# MLIR lit tests (requires tessera-opt built)
python -m lit tests/tessera-ir/ -v

# Coverage
pytest tests/phase1/ --cov=tessera.distributed --cov=tessera.compiler -v

# Type check
mypy python/tessera/distributed/ python/tessera/compiler/
```

### Build commands

```bash
# CPU only (Phases 1–2)
mkdir -p build && cd build
cmake .. -DTESSERA_ENABLE_CUDA=OFF -DTESSERA_CPU_ONLY=ON
make -j$(nproc)

# With CUDA (Phase 3+)
cmake .. -DTESSERA_ENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
make -j$(nproc)
```

---

## 7. Key Source File Index

| What you need | Where to look |
|---------------|--------------|
| Graph IR op definitions | `src/ir/TesseraOps.td` |
| Graph IR canonicalization (4 patterns) | `src/transforms/lib/CanonicalizeTesseraIR.cpp` |
| Graph IR verifier | `src/transforms/lib/VerifyTesseraIR.cpp` |
| Schedule IR op definitions | `src/programming_model/ir/schedule/ScheduleMeshPipelineOps.td` |
| x86 backend headers | `src/compiler/codegen/tessera_x86_backend/include/tessera/x86/target.h` |
| FA-4 Attn dialect ODS | `src/tile_opt_fa4/dialects/tessera_attn/Attn.td` |
| FA-4 Queue dialect ODS | `src/tile_opt_fa4/dialects/tessera_queue/Queue.td` |
| Collective IR ODS | `src/collectives/include/tessera/Dialect/Collective/IR/CollectiveOps.td` |
| Collective adapters (Phase 4) | `src/collectives/include/tessera/Dialect/Collective/Runtime/Adapters.h` |
| Scaling/Resilience dialect ODS | `src/scaling_resilience/lib/sr/dialect/SROps.td` |
| Solver pass declarations | `src/solvers/passes/SolversPasses.h` |
| Runtime C ABI header | `src/runtime/include/tessera/tessera_runtime.h` |
| Python `@jit` implementation | `python/tessera/compiler/jit.py` |
| Python Graph IR builder | `python/tessera/compiler/graph_ir.py` |
| Python effect lattice | `python/tessera/compiler/effects.py` |
| Python constraint solver | `python/tessera/compiler/constraints.py` |
| Python GPU target | `python/tessera/compiler/gpu_target.py` |
| Python distributed API | `python/tessera/distributed/` |
| Mock collective (testing) | `python/tessera/testing/mock_collective.py` |
