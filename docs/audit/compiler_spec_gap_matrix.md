---
status: Informative
classification: Audit Matrix
last_updated: 2026-05-06
---

# Compiler Spec Gap Matrix

Remediation status: many original gaps in this matrix were addressed by the
spec-status alignment pass and locked by
`tests/unit/test_compiler_spec_gap_remediation.py`. This May 6 refresh adds the
new compiler frontend, profiling/autotuning, and debugging/replay surfaces.
Rows classified as `spec-stale` or `needs-clarification` indicate documentation
claims that still need another rebuild. Rows classified as `scaffolded`,
`stubbed`, or `missing` remain valid implementation backlog items and are
explicitly documented as incomplete rather than current behavior.

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
| Language/API | Textual DSL frontend | Parse module/func/kernel/mesh/schedule/control-flow constructs and lower to Graph IR | `python/tessera/compiler/frontend/parser.py`; textual frontend tests | implemented | Unsupported-construct diagnostics should remain source-span aware. |
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
| Python API | Debug graph namespace | `tessera.graph.trace`, `debug_trace`, `debug_value`, `export_graphviz`, `replay_capture` | `python/tessera/__init__.py`; `python/tessera/debug.py`; `tests/unit/test_debugging_tools_foundation.py` | implemented / spec-stale | `PYTHON_API_SPEC.md` should list the new debug/replay aliases. |
| Python API | Profiling and autotuning namespace | Public profiler sessions/reports and callable GEMM autotune facade | `python/tessera/profiler.py`; `python/tessera/autotune.py`; `tests/unit/test_profiling_autotuning_foundation.py` | implemented foundation | Device-timer backends remain future work; synthetic/on-device-unmeasured statuses are explicit. |
| Developer CLI | `tessera-mlir` static source inspection | Emit static Graph/Schedule/Tile/Target stubs plus metadata/diagnostics/trace/GraphViz/all bundles | `python/tessera/cli/mlir.py`; `tests/unit/test_cli_debug_profile_commands.py` | implemented | Static mode does not execute source. |
| Developer CLI | `tessera-mlir --mode=compile_artifact --symbol` | Import selected module and read real JIT runtime artifact/lowering trace without launching tensors | `python/tessera/cli/mlir.py`; CLI tests | implemented / opt-in | Should be documented as import-side-effecting and not the default. |
| Developer CLI | `tessera-prof` and `tessera-autotune` | Source-inspection profiling, Chrome traces, JSON telemetry, GEMM tuning artifacts | `python/tessera/cli/prof.py`; `python/tessera/cli/autotune.py`; `pyproject.toml`; CLI tests | implemented foundation | True device timers are backend backlog. |
| Graph IR | Dialect registration and core attrs | ODS dialect, epilogue enum, numeric policy, module attrs | `src/compiler/ir/TesseraOps.td`; `src/compiler/ir/TesseraDialect.cpp` | implemented | Numeric policy is dictionary-backed; spec examples should match. |
| Graph IR | Core op catalog `matmul`, `conv2d_nhwc`, `flash_attn`, `fused_epilogue`, `transpose`, `cast` | ODS ops and Python Graph IR builder/catalog support | `src/compiler/ir/TesseraOps.td`; `python/tessera/compiler/graph_ir.py`; `op_catalog.py` | implemented | Keep spec and catalog synchronized. |
| Graph IR | Source spans and stable diagnostics | Line/column source spans and stable diagnostic bridging | `python/tessera/compiler/graph_ir.py`; `python/tessera/diagnostics.py`; diagnostic tests | implemented | Keep error-code guide and frontend diagnostics in sync. |
| Graph IR | `tessera.graph.debug_value` marker | Named capture point preserved into Schedule debug artifact in object-model lowering | `python/tessera/debug.py`; `python/tessera/compiler/schedule_ir.py`; schedule tests | implemented / metadata-only | Native ODS debug op remains optional/future. |
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
| Lowering pipeline | Python object-model Graph→Schedule→Tile→Target path | Object construction/verifier checks and inspectable artifacts for CPU/NVIDIA/Apple/ROCm | `python/tessera/compiler/driver.py`; `schedule_ir.py`; `tile_ir.py`; `target_ir.py`; compiler driver tests | implemented | `LOWERING_PIPELINE_SPEC.md` should describe this path separately from C++ pass pipelines. |
| Lowering pipeline | Debug artifact dumps | `TESSERA_DEBUG_IR`, `TESSERA_DUMP_STATE`, `TESSERA_DUMP_DIR` write IR/metadata/trace bundles | `python/tessera/compiler/driver.py`; `tests/unit/test_compiler_driver_foundation.py` | implemented | Informative debugging contract; not a semantic lowering pass. |
| Tile IR | `tile.async_copy`, `tile.wait_async`, `tile.reduce` | PM verifier and FA-4 lowerings/tests | `src/compiler/programming_model/ir/ScheduleOps.cpp`; `src/compiler/tile_opt_fa4/`; tests | implemented / lit-testable | Spec op names should match active names. |
| Tile IR | Shared allocation | Spec says `tshared.alloc`; active PM verifier uses `tile.alloc_shared` | `docs/spec/TILE_IR.md`; `src/compiler/programming_model/ir/ScheduleOps.cpp` | spec-stale | Normalize or document alias. |
| Tile IR | Generic barrier | Spec has `tile.barrier`; active code emphasizes mbarrier ops | `docs/spec/TILE_IR.md`; `ScheduleOps.cpp`; FA-4 docs/tests | needs-clarification | Clarify barrier family and target mapping. |
| Tile IR | FA-4 attention dialect | ODS ops and verifiers for LSE, masks, online softmax | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`; `lib/Dialect/Attn`; tests | lit-testable | Mark kernel finalization separately. |
| Tile IR | Queue dialect | ODS queue create/push/pop plus verifiers | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Queue/Queue.td`; queue tests | lit-testable | Some passes use `NoneType` placeholders for queue/token state. |
| Tile IR | Debug artifact/barrier markers | `tile.debug_artifact` and `tile.debug_barrier` metadata markers in Python object model | `python/tessera/compiler/tile_ir.py`; `tests/unit/test_tile_ir.py` | implemented / metadata-only | Should be either documented as informative markers or excluded from normative Tile IR. |
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
| Debugging/replay | Structured numerical trace | Tensor summaries with JSON export and bounded samples | `python/tessera/debug.py`; debug tests | implemented | Full tensor capture remains opt-in/planned. |
| Debugging/replay | Replay manifests | Manifest captures artifact metadata, IR hashes, debug environment, and caller metadata | `python/tessera/debug.py`; `docs/guides/Tessera_Debugging_Tools_Guide.md`; debug tests | implemented | Keep out of C ABI unless promoted to runtime contract. |
| Profiling/autotuning | Unified telemetry schema | Profiler/autotuner/benchmarks share `tessera.telemetry.v1` events | `python/tessera/telemetry.py`; profiler/autotune tests; benchmark tests | implemented foundation | Runtime device counters are planned. |
| Profiling/autotuning | GEMM legal candidate and cache foundation | Synthetic roofline evaluator, SQLite cache, schedule artifacts, unmeasured on-device status | `python/tessera/autotune.py`; `python/tessera/compiler/autotune_v2.py`; tests | implemented foundation | Softmax/attention tuning and device timers remain backlog. |
| Conformance | T0 kernel-only profile | Python frontend and CPU/mock kernel execution | Python frontend/tests; runtime CPU | implemented | Conformance should cite current tests. |
| Conformance | T1 single-node profile | Single-node compiler/runtime pieces exist but target native execution varies | `COMPILER_REFERENCE.md`; phase tests | needs-clarification | Split artifact generation from runtime execution. |
| Conformance | T2 cluster profile | Mock distributed APIs and planner exist; native cluster execution not proven | `src/collectives/`; mock collective tests | scaffolded | Keep native cluster profile planned. |
| Conformance | Header and test command accuracy | Header matches current mixed status and defers implementation state to `COMPILER_REFERENCE.md` | `docs/spec/CONFORMANCE.md`; `docs/spec/COMPILER_REFERENCE.md` | closed 2026-05-20 | Keep validation commands synchronized through `VALIDATION_SPINE.md` rather than restating every test lane here. |
| Compiler reference | Phase/status map | More current than some specs | `docs/spec/COMPILER_REFERENCE.md`; `docs/README.md` | implemented / needs-clarification | Treat as status source of truth, then update stale specs. |
