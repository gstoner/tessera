---
status: Informative
classification: Audit Matrix
last_updated: 2026-05-04
---

# Compiler Spec Gap Matrix

Remediation status: the gaps in this matrix are addressed by the current
spec-status alignment pass. Rows classified as `spec-stale` or
`needs-clarification` were fixed by updating the affected specs to use current
status labels and by adding regression coverage in
`tests/unit/test_compiler_spec_gap_remediation.py`. Rows classified as
`scaffolded`, `stubbed`, or `missing` remain valid implementation backlog items
and are explicitly documented as incomplete rather than current behavior.

Statuses used here:

- `implemented`: source exists and has direct Python/unit or C++/lit coverage.
- `lit-testable`: MLIR/dialect/backend behavior has target-contract tests, but
  native hardware execution is not the claimed surface.
- `scaffolded`: API, directory, ODS, or pass shape exists with incomplete
  behavior.
- `stubbed`: behavior intentionally returns no-op, placeholder, or diagnostic
  output.
- `missing`: no active implementation evidence found.
- `spec-stale`: implementation moved ahead of or away from the spec text.
- `needs-clarification`: source and spec can be reconciled, but the current
  claim is too broad.

| Spec area | Requirement or claim | Expected implementation | Evidence | Status | Notes / recommended action |
|-----------|----------------------|-------------------------|----------|--------|----------------------------|
| Language/API | `@tessera.jit` bare and keyword forms | Decorator collects constraints/effects and emits Graph IR artifacts | `python/tessera/compiler/jit.py`; `tests/unit/test_constraints.py`; `tests/unit/test_effects.py` | implemented | Current call path executes Python/reference CPU unless compiled artifact path is available. |
| Language/API | `@tessera.kernel` and `index_launch(axis=...)` | Kernel wrapper plus shard dispatcher | `python/tessera/distributed/launch.py`; distributed tests | implemented | Runtime is sequential/mock in Python; native multi-rank dispatch remains outside current behavior. |
| Language/API | `Region[...]` privilege annotations | Type annotation sentinel and exported namespace | `python/tessera/distributed/region.py`; `python/tessera/__init__.py` | implemented | Compiler lowering of privileges should be mapped explicitly in spec. |
| Language/API | Constraint predicates `Divisible`, `Range`, `Equal` | Decoration-time collection/checking | `python/tessera/compiler/constraints.py`; `tests/unit/test_constraints.py` | implemented | Runtime first-call shape binding remains planned per spec. |
| Language/API | Deterministic contract and effect inference | Effect lattice and deterministic checks | `python/tessera/compiler/effects.py`; `src/transforms/lib/EffectAnnotationPass.cpp`; tests | implemented | Cross-language parity between Python and MLIR effect levels should be documented. |
| Python API | `domain.Rect`, `dist.Block`, `dist.Cyclic`, `dist.Replicated` | Domain/distribution classes and shard specs | `python/tessera/distributed/domain.py`; `python/tessera/distributed/shard.py`; distributed tests | implemented | `Cyclic` is current, not just planned. |
| Python API | `DistributedArray.from_domain` and `parts(axis)` | DistributedArray with block/cyclic/replicated slices | `python/tessera/distributed/array.py`; distributed unit tests | implemented | Mock data partitioning only; no native distributed storage claim. |
| Python API | Core ops `gemm`/`matmul` | Numpy reference and Graph IR catalog mapping | `python/tessera/__init__.py`; `python/tessera/compiler/op_catalog.py`; `src/compiler/ir/TesseraOps.td` | implemented | Public `gemm` aliases to Graph IR `tessera.matmul`. |
| Python API | `conv2d` listed as stub | NHWC numpy reference implementation | `python/tessera/__init__.py`; `tests/unit/test_graph_ir.py`; numerical tests | spec-stale | Update spec table from `stub` to reference implementation. |
| Python API | Collectives `all_reduce`, `reduce_scatter`, `all_gather` | Public functions plus mock/no-op behavior | `python/tessera/__init__.py`; `python/tessera/testing/mock_collective.py`; collective tests | stubbed | Call out single-rank/mock status. |
| Python API | `all_to_all` support | Public op and op catalog entry | `python/tessera/__init__.py`; `python/tessera/compiler/op_catalog.py`; `tests/unit/test_moe.py` | spec-stale | Add to public API and conformance tables if retained. |
| Python API | RNG helpers | `rng_uniform`, `rng_normal`, RNG stream planning | `python/tessera/__init__.py`; `tests/unit/test_rng_legalize.py` | spec-stale | Add to op catalog section or mark extension. |
| Python API | Spectral and RMSNorm ops | Public ops and ODS entries | `python/tessera/__init__.py`; `src/compiler/ir/TesseraOps.td`; spectral/RMS tests | spec-stale | Add to Graph/Python API specs. |
| Python API | KV cache helpers | Public `kv_cache_append`/`kv_cache_prune`; ODS cache ops | `python/tessera/__init__.py`; `src/compiler/ir/TesseraOps.td`; textual frontend tests | implemented / spec-stale | Spec should cover current source or classify extension. |
| Graph IR | Dialect registration and core attrs | ODS dialect, epilogue enum, numeric policy, module attrs | `src/compiler/ir/TesseraOps.td`; `src/compiler/ir/TesseraDialect.cpp` | implemented | Numeric policy is dictionary-backed; spec examples should match. |
| Graph IR | Core op catalog `matmul`, `conv2d_nhwc`, `flash_attn`, `fused_epilogue`, `transpose`, `cast` | ODS ops and Python Graph IR builder/catalog support | `src/compiler/ir/TesseraOps.td`; `python/tessera/compiler/graph_ir.py`; `op_catalog.py` | implemented | Keep spec and catalog synchronized. |
| Graph IR | Canonicalization patterns | Fusion/simplification pass | `src/transforms/lib/CanonicalizeTesseraIR.cpp`; `tests/tessera-ir/pipelines/cleanup_pipeline.mlir`; unit tests | implemented / lit-testable | Verify each named pattern still has a test. |
| Graph IR | Tiling interface on matmul/conv | ODS declares interface; C++ has TODO/failure paths | `src/compiler/ir/TesseraTiling.cpp`; `src/compiler/ir/TilingInterface_NOTES.md` | scaffolded | Do not claim complete TilingInterface support. |
| Shape system | Python shape objects and constraints | Shape classes, broadcasting, matmul/reshape checks | `python/tessera/shape.py`; `tests/unit/test_shape_system_foundation.py` | implemented | Python-level checks are strongest evidence. |
| Shape system | MLIR shape diagnostics | Shape inference pass and diagnostics | `src/compiler/diagnostics/ShapeInferencePass.cpp`; `tests/unit/test_shape_inference.py`; phase6 lit tests | implemented / scaffolded | Identify which op verifiers enforce shape statically. |
| Lowering pipeline | `tessera-lower-to-x86` pass order | Registered pipeline effect/canonicalize/distribution/tiling/x86 | `src/transforms/lib/Passes.cpp`; `tests/tessera-ir/phase2/` | implemented / lit-testable | Source order should be canonical. |
| Lowering pipeline | `tessera-lower-to-gpu` pass order | Registered GPU lowering pipeline with FA-4/NVIDIA passes | `src/transforms/lib/Passes.cpp`; `tests/tessera-ir/phase3/` | lit-testable | Native CUDA execution is separate from artifact lowering. |
| Lowering pipeline | `EffectAnnotationPass` | C++ pass infers `tessera.effect` | `src/transforms/lib/EffectAnnotationPass.cpp`; `tests/tessera-ir/phase2/effect_annotation.mlir` | implemented / lit-testable | Keep effect names aligned with Python lattice. |
| Lowering pipeline | `DistributionLoweringPass` | Converts shard attrs to `schedule.mesh.*` generic ops | `src/transforms/lib/DistributionLoweringPass.cpp`; phase2 lit | implemented / lit-testable | Uses generic schedule ops in this pass. |
| Lowering pipeline | `TilingPass` | Lowers static matmul to tiled loop structure | `src/transforms/lib/TilingPass.cpp`; phase2 lit | implemented / lit-testable | Separate from ODS TilingInterface TODOs. |
| Lowering pipeline | `TileToX86Pass` | Lowers matmul/fused epilogue to x86 C ABI calls | `src/transforms/lib/TileToX86Pass.cpp`; phase2 lit; x86 backend tests | implemented / lit-testable | CPU native path is real for supported ops. |
| Tile IR | `tile.async_copy`, `tile.wait_async`, `tile.reduce` | PM verifier and FA-4 lowerings/tests | `src/compiler/programming_model/ir/ScheduleOps.cpp`; `src/compiler/tile_opt_fa4/`; tests | implemented / lit-testable | Spec op names should match active names. |
| Tile IR | Shared allocation | Spec says `tshared.alloc`; active PM verifier uses `tile.alloc_shared` | `docs/spec/TILE_IR.md`; `src/compiler/programming_model/ir/ScheduleOps.cpp` | spec-stale | Normalize or document alias. |
| Tile IR | Generic barrier | Spec has `tile.barrier`; active code emphasizes mbarrier ops | `docs/spec/TILE_IR.md`; `ScheduleOps.cpp`; FA-4 docs/tests | needs-clarification | Clarify barrier family and target mapping. |
| Tile IR | FA-4 attention dialect | ODS ops and verifiers for LSE, masks, online softmax | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`; `lib/Dialect/Attn`; tests | lit-testable | Mark kernel finalization separately. |
| Tile IR | Queue dialect | ODS queue create/push/pop plus verifiers | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Queue/Queue.td`; queue tests | lit-testable | Some passes use `NoneType` placeholders for queue/token state. |
| Tile IR | TMEM tcgen05 | Test fixture and schematic PTX emitter | `src/compiler/tile_opt_fa4/test/tmem/tcgen05_ptx_body.mlir`; `LowerTileToPTX.cpp` | stubbed / lit-testable | Needs real Blackwell PTX and target attrs. |
| Target IR | NVIDIA Hopper/Blackwell contracts | NVIDIA dialect/lowering files and lit contracts | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/`; `test/nvidia/*.mlir` | lit-testable | Placeholder WGMMA kernels should not be called production runtime. |
| Target IR | ROCm target | ROCm dialect, conversion, target-contract tests | `src/compiler/codegen/Tessera_ROCM_Backend/`; ROCm lit tests | lit-testable / scaffolded | HIP loader has stubs; artifact support stronger than runtime. |
| Target IR | TPU target | StableHLO/Shardy lowering and Python target profile | `src/compiler/codegen/Tessera_TPU_Backend/`; `python/tessera/compiler/tpu_target.py`; tests | implemented / lit-testable | PJRT buffer execute remains TODO. |
| Target IR | Apple target | Apple CPU/GPU artifact lowering tests | `src/compiler/codegen/Tessera_Apple_Backend/`; phase8 lit | lit-testable | Artifact-only unless backend docs prove runtime. |
| Target IR | Metalium target | ODS and tests; matmul lowering TODO | `src/compiler/codegen/Tessera_Metalium_Backend/`; Metalium tests | scaffolded / lit-testable | Keep incomplete matmul lowering in backlog. |
| Target IR | Cerebras target | Tool/pass stubs and minimal tests | `src/compiler/codegen/Tessera_Cerebras_backend/` | stubbed / scaffolded | Decide if active spec should mention it. |
| Target IR | Rubin CPX target | ODS and lit tests for CPX-specific ops | `src/compiler/codegen/Tessera_RubinCPX_Backend/` | scaffolded / lit-testable | Add backend appendix or mark extension. |
| Memory model | Async movement | Async copy lowering and Tile IR tests | `src/compiler/tile_opt_fa4/lib/AsyncCopyLoweringPass.cpp`; phase3 tests | lit-testable | Good coverage for async movement subset. |
| Memory model | Atomics/fences/happens-before | Spec text stronger than active enforcement evidence | `docs/spec/MEMORY_MODEL_SPEC.md`; limited source evidence | missing / needs-clarification | Add verifier/lit tests or downgrade to planned. |
| Runtime ABI | Lifecycle, device, stream, event, memory ABI | C headers, implementation, Python wrapper, runtime tests | `src/runtime/include/tessera/*.h`; `src/runtime/src/tessera_runtime.cpp`; `python/tessera/runtime.py`; `tests/unit/test_runtime_abi.py` | implemented / mock-runtime | Spec phase table is stale for CPU/mock status. |
| Runtime ABI | Host tile kernel launch | CPU backend thread pool and C ABI | `src/runtime/src/backend/cpu_backend.cpp`; runtime tests | implemented | Native CPU backend exists. |
| Runtime ABI | Artifact compile/load/get-kernel/launch | C ABI has artifact functions; Python compile/launch returns artifacts and JIT CPU path | `tessera_runtime.h`; `tessera_runtime.cpp`; `python/tessera/runtime.py` | scaffolded / implemented subset | Clarify artifact-only vs executable artifact. |
| Runtime ABI | CUDA/HIP runtime | Backend files compile behind flags and require devices | `src/runtime/src/backend/cuda_backend.cpp`; `hip_backend.cpp` | needs-clarification | Status is hardware-runtime only under build flags/device availability. |
| Conformance | T0 kernel-only profile | Python frontend and CPU/mock kernel execution | Python frontend/tests; runtime CPU | implemented | Conformance should cite current tests. |
| Conformance | T1 single-node profile | Single-node compiler/runtime pieces exist but target native execution varies | `COMPILER_REFERENCE.md`; phase tests | needs-clarification | Split artifact generation from runtime execution. |
| Conformance | T2 cluster profile | Mock distributed APIs and planner exist; native cluster execution not proven | `src/collectives/`; mock collective tests | scaffolded | Keep native cluster profile planned. |
| Compiler reference | Phase/status map | More current than some specs | `docs/spec/COMPILER_REFERENCE.md`; `docs/README.md` | implemented / needs-clarification | Treat as status source of truth, then update stale specs. |
