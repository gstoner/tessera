# Tessera Compiler Unit And Performance Test Plan

Status: active test plan

This plan covers the current compiler architecture:

```text
Python @jit frontend
  -> Graph IR
  -> Schedule IR
  -> Tile IR
  -> Target IR
  -> CPU proxy execution or backend-specific native lowering
```

## Test Layout

| Suite | Directory | Default Run | Purpose |
| --- | --- | --- | --- |
| Unit | `tests/unit/` | `pytest` and `scripts/test.sh` | Fast correctness contracts for Python compiler APIs, IR emission, pass preconditions, CPU proxy execution, and compiler documentation drift. |
| Performance | `tests/performance/` | `cmake --build . --target check-tessera-performance` or `TESSERA_RUN_PERFORMANCE_TESTS=1 ./scripts/test.sh` | Deterministic roofline/proxy performance contracts for compile latency, generated-artifact size, GEMM, attention, collectives, and benchmark report schema. |
| MLIR lit | `tests/tessera-ir/` | `check-tessera-ir` | FileCheck-based C++/MLIR pass and pipeline contracts. |

## Unit Test Matrix

| Compiler Area | Required Unit Coverage | Representative Tests |
| --- | --- | --- |
| Frontend source recovery | inspect source, explicit `source=`, unavailable source diagnostics | `test_end_to_end_matmul_cpu_path.py` |
| Constraint extraction | `require(...)`, invalid bindings, symbolic skip | `test_constraints.py` |
| Effect inference | pure/random/state/collective/determinism contracts | `test_effects.py`, `test_deep_learning_semantic_core.py` |
| Graph IR emission | function args, Region effects, nested op extraction, keyword attrs | `test_graph_ir.py`, `test_lowering_chain.py` |
| CPU compiler path | supported op graph execution, artifacts for all compiler layers, eager fallback diagnostics | `test_end_to_end_matmul_cpu_path.py`, `test_transformer_compiler_example.py` |
| Target profiles | GPU/TPU capability gates and lowering config attrs | `test_gpu_target.py`, `test_tpu_lowering.py`, `test_flash_attn_lowering.py` |
| Distributed planning | DP/TP/PP plans, pipeline stages, collective insertion preconditions | `test_distributed_plan.py`, `test_pipeline_stage_insertion.py`, `test_gpu_collective_insertion.py` |
| Reliability/runtime contracts | diagnostics, shape inference, runtime ABI, replay/QA helpers | `test_error_reporter.py`, `test_shape_inference.py`, `test_runtime_abi.py`, `test_qa_reliability_foundation.py` |

Unit tests should stay deterministic, CPU-only, and cheap enough for local edit-test loops.

## Performance Test Matrix

| Compiler/Runtime Concern | Required Performance Coverage | Gate |
| --- | --- | --- |
| JIT compile latency | Decoration plus Graph/Schedule/Tile/Target artifact construction for a transformer-shaped graph | Median wall time under `0.25s` on local CI-class CPU |
| Generated artifact size | Graph/Schedule/Tile/Target text remains compact for representative op graphs | Total text under `20 KiB` for the transformer proxy |
| GEMM roofline model | Model-shape and square GEMMs produce positive TFLOPs and expected compute/memory transitions | Large square GEMM compute-bound; tiny GEMM memory-bound |
| Attention roofline model | Flash-attention proxy reports positive tokens/sec, TFLOPs, and bounded MFU | `0 < mfu <= 1.0` |
| Collective model | All-reduce/reduce-scatter/all-gather model reports monotonic bus bytes and bounded utilization | `0 < utilization <= 1.0` |
| Benchmark suite schema | Combined benchmark JSON contains summary and all result families | Required keys present with non-empty result lists |

Performance tests should avoid requiring accelerators by default. Hardware-backed benchmarks belong in specialized GPU jobs and should be marked separately from deterministic proxy tests.

## Expansion Backlog

1. Add lit coverage for every named lowering pipeline alias exposed by `tessera-opt`.
2. Add compile-time regression thresholds for larger transformer blocks once multi-op native lowering exists.
3. Add hardware-marked performance gates for SM80, SM90, ROCm MFMA, and TPU backends.
4. Track benchmark baselines in JSON and compare against tolerances rather than absolute single-machine timings.
