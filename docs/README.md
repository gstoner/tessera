---
status: Normative
classification: Normative
authority: Documentation authority tree
last_updated: 2026-07-02
---

# Tessera Documentation Map

This file defines the documentation authority tree for Tessera. If documents
conflict, resolve the conflict in this order. Use status labels below for
implementation claims; phase numbers alone are too coarse for the current tree.

## Normative Root

These documents are the source of truth for current Tessera API, compiler,
runtime, and phase/status claims:

| Topic | Document |
|-------|----------|
| Public API names and current syntax | `docs/CANONICAL_API.md` |
| Tensor attribute vocabulary, dtype canonicalization, and numeric policy | `docs/reference/tessera_tensor_attributes.md` |
| Compiler architecture, IR stack, pass registry, phase status | `docs/spec/COMPILER_REFERENCE.md` |
| Language and multi-level IR semantics | `docs/spec/LANGUAGE_AND_IR_SPEC.md` |
| Python API surface | `docs/spec/PYTHON_API_SPEC.md` |
| Graph IR semantics | `docs/spec/GRAPH_IR_SPEC.md` |
| Lowering pipeline contracts | `docs/spec/LOWERING_PIPELINE_SPEC.md` |
| Schedule, Tile, and Target IR dialect details | `docs/spec/TARGET_IR_SPEC.md` |
| Memory model | `docs/spec/MEMORY_MODEL_SPEC.md` |
| Shape, layout, shard, and schedule feasibility system | `docs/spec/SHAPE_SYSTEM.md` |
| Standard operator library | `docs/operations/Tessera_Standard_Operations.md` |
| Clifford / geometric algebra primitive surface | `docs/spec/CLIFFORD_SPEC.md` |
| Energy-based model primitive surface | `docs/spec/EBM_SPEC.md` |
| GA + EBM execution status by implementation layer | `docs/spec/GA_EBM_EXECUTION_STATUS.md` |
| GA + EBM native milestone status and health check | `docs/status/ga_ebm_milestone.md` |
| Generated 8-axis compiler support table (M0 / M0.5 — drift-gated by `tests/unit/test_compiler_audit.py`; regenerate via `python -m tessera.compiler.audit support_table`) | `docs/audit/generated/support_table.md` |
| Error handling and diagnostics | `docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md` |
| Runtime C ABI | `docs/spec/RUNTIME_ABI_SPEC.md` |

## Supporting Specs

These documents remain normative only where they do not conflict with the normative root:

| Topic | Document |
|-------|----------|
| Conformance profiles | `docs/spec/CONFORMANCE.md` |
| Language notes | `docs/spec/LANGUAGE_SPEC.md` |
| Tile IR notes | `docs/spec/TILE_IR.md` |

## Informative Guides

Architecture, programming guide, reference, and tutorial documents are explanatory. They should link back to the normative root for API names, phase status, and implementation claims.

Architecture readers should start with `docs/architecture/README.md`.

Compiler planning readers should use
`docs/audit/compiler/COMPILER_AUDIT.md` for the current
source-base review, next compiler milestones, and the Visual Complex Analysis
assessment. Older audit files remain useful historical context.

The **forward compiler direction (north star)** is the paired plan + theory set:
`docs/audit/compiler/COMPILER_THEORY_OF_OPERATION.md` (read first — the three-tier
kernel model + accuracy-budgeted measured arbiter + the three-system fleet),
`docs/audit/compiler/COMPILER_REFACTOR_PLAN.md` (workstreams + coordination), and
the reassessed `docs/audit/compiler/OPTIMIZING_COMPILER_PLAN.md` (F6 = the
backend-build seam). These are *direction*; `docs/audit/MASTER_AUDIT.md` and the
generated dashboards stay status truth.

Reliability and validation readers should start with
`docs/guides/Tessera_QA_Reliability_Guide.md`. It is the hands-on guide for
correctness, numerical stability, determinism, expected failures, performance
consistency, and distributed QA behavior.

When a test or production run fails, use
`docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md` for stable error
codes, diagnostic fields, environment switches, and debugging workflows.

For graph inspection, numerical tracing, gradient checking, determinism checks,
and external debugger integration, use
`docs/guides/Tessera_Debugging_Tools_Guide.md`.

For the first executable developer frontend path, `@jit` matmul lowering,
Graph/Schedule/Tile/Target inspection, and current frontend boundaries, use
`docs/guides/Tessera_Developer_Frontend_End_To_End.md`.

For runtime metrics, Chrome trace export, cost models, autotuning workflows,
persistent tuning caches, and on-device measurement contracts, use
`docs/guides/Tessera_Profiling_And_Autotuning_Guide.md`.

For tensor layouts, packed dtype storage, KV-cache paging, explicit prefetch,
async copies, and Hopper+ mbarrier movement patterns, use
`docs/guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md`.

For differentiable architecture search, searchable Graph IR choices, relaxed
Schedule IR knobs, hardware-cost surrogates, and freeze/specialize workflow, use
`docs/guides/Tessera_Differentiable_NAS_Guide.md`.

Production operators should also read
`docs/guides/Tessera_Production_Reliability_And_Chaos_Guide.md`, which covers
monitoring, regression detection, replay debugging, observability, production
fault tolerance, stress testing, chaos testing, node-scale QA, and rack-scale
NVL72 validation.

For fault policies, elastic membership, mesh resharding, atomic runtime
checkpoints, rendezvous integration, and preemption handling, use
`docs/guides/Tessera_Fault_Tolerance_And_Elasticity_Guide.md`.

For production inference serving, `.tspkg` packaging, continuous batching,
OpenAI-compatible APIs, paged KV cache, distributed serving meshes, and
serving SRE guidance, use `docs/guides/Tessera_Inference_Server_Guide.md`.

## Archive

Pre-canonical or superseded material lives under `archive/docs/pre_canonical/`. Archived documents are retained for design history only and must not be treated as current API or implementation guidance.

## Status Labels

Use these labels consistently in active docs:

| Label | Meaning |
|-------|---------|
| implemented | Source exists, is importable/buildable in the active tree, and has unit or lit coverage. |
| lit-testable | MLIR/dialect/pipeline behavior has lit fixtures or target-contract tests, but native hardware execution is not the claimed surface. |
| mock-runtime | Runtime API exists and has a deterministic Python or CPU/mock fallback for development and tests. |
| hardware-runtime | Native runtime execution is wired for the named backend and has a concrete build/test path. |
| scaffolded | Directory, API shape, or design skeleton exists, but behavior is incomplete or intentionally artifact-only. |
| planned | Design direction only; do not describe it as current behavior. |
| archived | Retained for history under archive paths; not active implementation guidance. |

## Current Status Summary

| Area | Status | Source of truth |
|------|--------|-----------------|
| Python frontend, textual DSL frontend, constraints, effects, Graph IR | implemented | `docs/CANONICAL_API.md`, `docs/spec/PYTHON_API_SPEC.md`, `python/tessera/` |
| Object-backed Schedule IR, Tile IR, and CPU/NVIDIA/Apple/ROCm Target IR artifact lowering | implemented / lit-testable | `python/tessera/compiler/schedule_ir.py`, `tile_ir.py`, `target_ir.py`, `tests/unit/test_*_ir.py` |
| x86 AMX / AVX512 lowering and execution | implemented / hardware-runtime | `docs/spec/COMPILER_REFERENCE.md`, `python/tessera/compiler/matmul_pipeline.py`, `src/transforms/`, `src/compiler/codegen/tessera_x86_backend/` |
| CPU `tessera_jit` MLIR→LLVM JIT — the executed `@jit(target="cpu")` path for the covered op set (f32/f16/bf16/f64); numpy fallback otherwise | implemented / hardware-runtime | `docs/spec/COMPILER_REFERENCE.md` §2.3, `docs/audit/compiler/COMPILER_AUDIT.md`, `tools/tessera-jit/tessera_jit.cpp`, `tests/unit/test_native_cpu_jit.py` |
| NVIDIA SM90+ WGMMA/TMA, Blackwell TCGEN05/TMEM, and FA-4 target artifacts | implemented / lit-testable | `docs/spec/COMPILER_REFERENCE.md`, `src/compiler/tile_opt_fa4/`, `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` |
| Distributed collectives and planner foundation | implemented / scaffolded | `src/collectives/`, `python/tessera/testing/mock_collective.py`, `tests/unit/test_nccl_adapter.py` |
| Solver, RNG, sparse, linalg, and resilience passes | implemented / lit-testable | `src/solvers/`, `tests/unit/test_*solver*.py`, `tests/tessera-ir/phase5/` |
| Clifford / geometric algebra surface | implemented / lit-testable / hardware-runtime for 17/17 registered Apple GPU fused GA kernels, benchmarked by `benchmark_ga_ebm.py --ci` | `docs/spec/CLIFFORD_SPEC.md`, `docs/spec/GA_EBM_EXECUTION_STATUS.md`, `python/tessera/ga/`, `python/tessera/autodiff/geometric/`, `src/solvers/clifford/` |
| Energy-based model surface | implemented / lit-testable / hardware-runtime for **9/9 native Apple GPU EBM rows** (incl. `ebm_partition_exact` via stable-logsumexp MSL kernel, 2026-05-17) | `docs/spec/EBM_SPEC.md`, `docs/spec/GA_EBM_EXECUTION_STATUS.md`, `docs/status/ga_ebm_milestone.md`, `python/tessera/ebm/`, `src/solvers/ebm/`, `benchmarks/apple_gpu/benchmark_ga_ebm.py` |
| Agent-native MoE training stack (`tessera.train`) — MoE router/FFN, sparse dispatch, Qwen3-MoE model, GRPO loop; lazily bound at top level | implemented (Python reference) / hardware-runtime on Apple GPU single-node; multi-node EP/PP collectives hardware-gated (Phase G/H) | `docs/spec/PYTHON_API_SPEC.md` §20, `python/tessera/train/`, `tests/unit/test_train_*.py` |
| Runtime C ABI and Python wrapper | mock-runtime / hardware-runtime where C backend is built and device-present | `docs/spec/RUNTIME_ABI_SPEC.md`, `python/tessera/runtime.py`, `src/runtime/` |
| Apple CPU backend | implemented / lit-testable / hardware-runtime via Accelerate + BNNS | `python/tessera/compiler/target_ir.py`, `src/compiler/codegen/Tessera_Apple_Backend/`, `python/tessera/runtime.py`, Apple target-contract tests |
| Apple GPU backend | implemented / lit-testable / hardware-runtime on Darwin via MPS, MPSGraph, custom MSL, Metal 4 lanes, and packaged `.mtlpackage` ABI validation | `python/tessera/compiler/target_ir.py`, `python/tessera/apple_mlpkg.py`, `python/tessera/compiler/apple_packaged_manifest.py`, `src/compiler/codegen/Tessera_Apple_Backend/`, Apple target-contract tests |
| ROCm backend | implemented / lit-testable / hardware-runtime on gfx1151 (Strix Halo, RDNA 3.5) via compiler-generated HIP `runtime.launch()` lanes; CDNA/MI300-class remains hardware-gated | `python/tessera/compiler/target_ir.py`, `src/compiler/codegen/Tessera_ROCM_Backend/`, `docs/audit/backend/rocm/ROCM_AUDIT.md`, ROCm target-contract tests |
