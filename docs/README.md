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
| Python API surface | `docs/spec/PYTHON_API_SPEC.md` |
| Graph IR semantics | `docs/spec/GRAPH_IR_SPEC.md` |
| Lowering pipeline contracts | `docs/spec/LOWERING_PIPELINE_SPEC.md` |
| Schedule, Tile, and Target IR dialect details | `docs/spec/TARGET_IR_SPEC.md` |
| Runtime C ABI | `docs/spec/RUNTIME_ABI_SPEC.md` |

## Supporting Specs

These documents remain normative only where they do not conflict with the normative root:

| Topic | Document |
|-------|----------|
| Conformance profiles | `docs/spec/01_conformance.md` |
| Language notes | `docs/spec/02_language_spec.md` |
| Tile IR notes | `docs/spec/04_tile_ir.md` |
| Shape-system notes | `docs/spec/shape-system.md` |

## Informative Guides

Architecture, programming guide, operations, reference, and tutorial documents are explanatory. They should link back to the normative root for API names, phase status, and implementation claims.

Architecture readers should start with `docs/architecture/README.md`.

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
