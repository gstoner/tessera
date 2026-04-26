# Tessera Documentation

Welcome to the Tessera documentation. Start with **Architecture** to understand the system, then use **Spec** documents as authoritative references when implementing or extending the compiler.

> **Phase status:** Phases 1 (Python frontend), 2 (x86 lowering), and 3 (NVIDIA GPU / FlashAttention) are complete. Phase 4 (Distributed Training — NCCL/TPU) is next. See [System Overview](architecture/system_overview.md) for the full status table.
>
> **Session 5–6 doc update complete.** All `[planned]` items in the Spec section are now written. Chapter 5 (Kernel Programming) is written. All programming guide chapters have been audited for wrong API names and corrected.

---

## Architecture

- [System Overview](architecture/system_overview.md) — High-level architecture, phase completion status, component map
- [System Architecture (detailed)](architecture/tessera_system_architecture.md) — Full pipeline, runtime, distributed services, tooling
- [Compiler Architecture Overview](architecture/Compiler/Tessera_Compiler_Architecture_Overview.md) — End-to-end pipeline from Python surface to binary
- [Frontend & Graph IR Design](architecture/Compiler/Tessera_Compiler_Frontend_Design_GraphIR.md) — Python → Graph IR lowering
- [Schedule IR Design](architecture/Compiler/Tessera_Compiler_ScheduleIR_Design.md) — Fusion, tiling, pipelining
- [Tile IR Design](architecture/Compiler/Tessera_Compiler_TileIR_Design.md) — Explicit tile memory ops, MMA, barriers
- [Target IR Design](architecture/Compiler/Tessera_Compiler_TargetIR_Design.md) — Backend lowering (NVIDIA PTX, ROCm, oneAPI)
- [Target IR Usage Guide](architecture/tessera_target_ir_usage_guide.md) — Practical guide to Target IR lowering
- [IR Layer Documentation](architecture/Compiler/tessera_ir_layers.md) — All four IR layers in depth

---

## Spec (Normative References)

Authoritative documents for the compiler. Ground truth for Claude Code sessions and compiler engineers.

> **Note:** Entries marked `[planned]` are being written as part of the April 2026 doc update. Until complete, use `CLAUDE.md` at the repo root as the canonical reference.

- [Canonical API Quick Reference](CANONICAL_API.md) — **Single-page naming authority** for all decorator names, module paths, and syntax
- [Compiler Reference](spec/COMPILER_REFERENCE.md) — IR stack, both named pass pipelines, phase status, 9 locked architecture decisions
- [Python API Spec](spec/PYTHON_API_SPEC.md) — All public symbols, parameter signatures, error types, Phases 1–3
- [Graph IR Spec](spec/GRAPH_IR_SPEC.md) — All 6 ops, 4 canonicalization patterns, verifier rules, MLIR text examples
- [Lowering Pipeline Spec](spec/LOWERING_PIPELINE_SPEC.md) — Every pass: input/output IR contracts, options, invariants
- [Target IR Spec](spec/TARGET_IR_SPEC.md) — FA-4 Attn dialect, TMA ops, WGMMA, Schedule Mesh, Queue dialects
- [Runtime ABI Spec](spec/RUNTIME_ABI_SPEC.md) — C ABI functions, all types, error model, backend architecture (Phase 6)
- [Conformance](spec/01_conformance.md) — Conformance profiles T0/T1/T2 (skeleton)
- [Language Spec](spec/02_language_spec.md) — Language surface (skeleton)
- [Runtime ABI](spec/03_runtime_abi.md) — Runtime ABI (skeleton)
- [Tile IR](spec/04_tile_ir.md) — Tile IR (skeleton)
- [Shape System](spec/shape-system.md) — Shape constraint system (skeleton)

---

## Programming Guide

- [Chapter 1: Introduction & Overview](programming_guide/Tessera_Programming_Guide_Chapter1_Introduction_Overview.md)
- [Chapter 2: Programming Model](programming_guide/Tessera_Programming_Guide_Chapter2_Programming_Model.md)
- [Chapter 3: Memory Model](programming_guide/Tessera_Programming_Guide_Chapter3_Memory_Model.md)
- [Chapter 4: Execution Model](programming_guide/Tessera_Programming_Guide_Chapter4_Execution_Model.md)
- [Chapter 5: Kernel Programming](programming_guide/Tessera_Programming_Guide_Chapter5_Kernel_Programming.md) — `@tessera.kernel`, dtype annotations, `index_launch`, Tile IR, FA-4, `MockRankGroup`
- [Chapter 6: Numerics Model](programming_guide/Tessera_Programming_Guide_Chapter6_Numerics_Model.md)
- [Chapter 7: Autodiff](programming_guide/Tessera_Programming_Guide_Chapter7_Autodiff.md)
- [Chapter 8: Layouts & Data Movement](programming_guide/Tessera_Programming_Guide_Chapter8_Layouts_Data_Movement.md)
- [Chapter 9: Libraries & Primitives](programming_guide/Tessera_Programming_Guide_Chapter9_Libraries_Primitives.md)
- [Chapter 10: Portability](programming_guide/Tessera_Programming_Guide_Chapter10_Portability.md)
- [Chapter 11: Conclusion](programming_guide/Tessera_Programming_Guide_Chapter11_Conclusion.md)
- [Appendix: NVL72 Guide](programming_guide/Tessera_Programming_Guide_Appendix_NVL72.md)
- [Goals](programming_guide/Tessera_Goals.md)

---

## API Reference

> **For all new work use `docs/spec/PYTHON_API_SPEC.md` and `docs/CANONICAL_API.md`.**
> The volume files below are pre-canonical and contain outdated API names — each carries a
> correction banner.

- [API Reference Index](api/API_Reference_Index.md) — **Start here** — quick-lookup table by symbol → canonical spec section
- [Vol 1: Frontend & Type System](api/Tessera_API_Vol1_Frontend_and_TypeSystem.md) *(pre-canonical)*
- [Vol 2: Operations](api/Tessera_API_Vol2_Operations.md) *(pre-canonical)*
- [Vol 3: IR & Target](api/Tessera_API_Vol3_IR_and_Target.md) *(pre-canonical)*
- [Vol 4: Runtime & Deployment](api/Tessera_API_Vol4_Runtime_and_Deployment.md) *(pre-canonical — see [Runtime ABI Spec](spec/RUNTIME_ABI_SPEC.md))*
- [Python API](api/python.md)

---

## Operations Reference

- [Standard Operations](operations/Tessera_Standard_Operations.md) — RMSNorm, SwiGLU, FlashAttention, MLA, and more

---

## Reference

- [Overview & Quick Start](reference/tessera-overview.md)
- [Programming Model Guide](reference/tessera-programming-model.md)
- [IR Pipeline Reference](reference/tessera-ir-pipeline.md)
- [API Reference](reference/tessera-api-reference.md)
- [Flash Attention](reference/tessera-flash-attention.md)
- [NVL72 Guide](reference/tessera-nvl72-guide.md)
- [Deployment](reference/tessera-deployment.md)
- [Performance Tuning Guide](reference/tessera_performance_tuning_guide.md)
- [Tessera vs JAX](reference/Tessera_vs_Jax.md)
- [ML Engineer Overview](reference/ml_engineer_guide_overview.md)
- [ML Engineer Models Guide](reference/ml_engineer_models_guide.md)
- [ML Engineer Training Guide](reference/ml_engineer_training_guide.md)
- [Migration Guide Part 1](reference/tessera_migration_guide_part1.md)
- [Migration Guide Part 2](reference/tessera_migration_guide_part2.md)

---

## Tutorials

- [Flash Attention in Tessera](tutorials/Flash_Attention_in_Tessera.md)
- [Performance Tuning](tutorials/performance_tuning.md)

---

## Benchmarks

- [TesseraBench](TesseraBench/) — Benchmark suite documentation (Phase 6)

---

## Build Integration

- [CMake Integration](build/README_SRC_INTEGRATION.md)

---

## Archived / Old Concepts

- [old_concepts/](old_concepts/README.md) — Experimental designs no longer part of the active architecture: Rust frontend research, Tracing JIT research, pre-Phase-1 programming model, C++ Target IR artifacts
