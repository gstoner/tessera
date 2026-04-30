---
status: Normative
classification: Normative
authority: Documentation authority tree
last_updated: 2026-04-26
---

# Tessera Documentation Map

This file defines the documentation authority tree for Tessera. If documents conflict, resolve the conflict in this order.

## Normative Root

These documents are the source of truth for current Tessera Phases 1-3 behavior and Phase 4-6 planning status:

| Topic | Document |
|-------|----------|
| Public API names and current syntax | `docs/CANONICAL_API.md` |
| Compiler architecture, IR stack, pass registry, phase status | `docs/spec/COMPILER_REFERENCE.md` |
| Language and multi-level IR semantics | `docs/spec/LANGUAGE_AND_IR_SPEC.md` |
| Python API surface | `docs/spec/PYTHON_API_SPEC.md` |
| Graph IR semantics | `docs/spec/GRAPH_IR_SPEC.md` |
| Lowering pipeline contracts | `docs/spec/LOWERING_PIPELINE_SPEC.md` |
| Schedule, Tile, and Target IR dialect details | `docs/spec/TARGET_IR_SPEC.md` |
| Memory model | `docs/spec/MEMORY_MODEL_SPEC.md` |
| Shape, layout, shard, and schedule feasibility system | `docs/spec/SHAPE_SYSTEM.md` |
| Standard operator library | `docs/operations/Tessera_Standard_Operations.md` |
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

Pre-canonical or superseded material lives under `docs/archive/pre_canonical/`. Archived documents are retained for design history only and must not be treated as current API or implementation guidance.

## Phase Labels

Use these labels consistently:

| Label | Meaning |
|-------|---------|
| Phase 1-3 implemented | Current implemented behavior covered by the normative root |
| Phase 4 planned | Distributed training, NCCL/RCCL collectives, TPU StableHLO, Cyclic distribution, pipeline parallelism |
| Phase 5 planned | Autodiff expansion, activation checkpointing, ZeRO sharding, Bayesian autotuning |
| Phase 6 planned | Runtime ABI production wiring, runtime Python wrapper, full ROCm MFMA coverage, benchmark suite |
