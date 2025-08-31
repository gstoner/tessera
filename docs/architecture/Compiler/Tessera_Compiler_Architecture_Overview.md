# Tessera Compiler Architecture Overview
*Scope:* Full pipeline from surface language to backend Target IR  
*Status:* Draft v0.2 (with programmer context)

---

## 0. Goals
- Provide a top-level view of the **Tessera compiler pipeline**.  
- Show how frontend → IR lowering → backend connects.  
- Ensure alignment with Programming Guide (Ch.1–11 + NVL72 Appendix).  
- Highlight responsibilities, contracts, and hand-offs at each stage.  
- Add programmer context: when and why you should inspect each IR.

---

## 1. Pipeline Overview

```text
Source (Pythonic Tessera)
    │
    ▼
Frontend (Lexer → Preprocessor → Parser → Semantic Analyzer)
    │
    ▼
Graph IR (typed, effect-aware, privileges, distributions)
    │
    ▼
Schedule IR (fusion, tiling, pipelining, collective overlap, autotuning)
    │
    ▼
Tile IR (explicit tiles, memory ops, mma, barriers, async pipelines)
    │
    ▼
Target IR (PTX, CUDA Tile IR, LLVM/ROCm, oneAPI)
    │
    ▼
Binary + Runtime (NCCL, RCCL, oneCCL integration, CUDA Graphs, AOT bundles)
```

---

## 2–10. Core Sections
Frontend, Graph IR, Schedule IR, Tile IR, Target IR, Runtime Integration, Example GEMM, Diagnostics, Milestones.  
*(unchanged from v0.1 — see earlier draft for details)*

---

## 11. Programmer’s Guide Through the Pipeline

Tessera exposes **inspection hooks** at every stage of compilation. As a programmer, you don’t need to know all compiler internals — but knowing **when to look at which IR** can save you time.

### Frontend / Graph IR
- **When to care**: shape mismatches, type errors, privilege conflicts.  
- **How to inspect**:  
  ```python
  my_fn.inspect_ir("graph")
  ```
- **You’ll see**: ops like `gir.gemm`, types with dtype/policies, ShardSpec annotations.  
- **Typical fix**: adjust tensor dims, add `Region[reduce_sum]`, correct ShardSpec.  

---

### Schedule IR
- **When to care**: performance autotuning, tiling, fusion issues.  
- **How to inspect**:  
  ```python
  my_fn.inspect_ir("sched")
  ```
- **You’ll see**: `sched.tile`, `sched.autotune`, `sched.pipeline` with block/warp sizes.  
- **Typical fix**: narrow autotune search space, override with `@kernel.schedule`.  

---

### Tile IR
- **When to care**: low-level perf debugging, bank conflicts, memory stalls.  
- **How to inspect**:  
  ```python
  my_fn.inspect_ir("tile")
  ```
- **You’ll see**: `tile.load`, `tile.cp_async`, `tile.mma`, `tile.barrier`.  
- **Typical fix**: swizzle shared memory layout, reduce vector width, adjust pipeline stages.  

---

### Target IR
- **When to care**: verifying backend correctness, advanced perf tuning, portability.  
- **How to inspect**:  
  ```python
  my_fn.inspect_ir("target")
  ```
- **You’ll see**: PTX, CUDA Tile IR, LLVM/DPAS intrinsics.  
- **Typical fix**: add explicit numerics policies (`@accum(fp32)`), verify backend lowering.  

---

## 12. End-to-End Workflow Example

1. Write kernel with FP8 GEMM.  
2. Run `inspect_ir("graph")` → check shapes and FP8 policy.  
3. Run `inspect_ir("sched")` → see autotune search space.  
4. Run `inspect_ir("tile")` → confirm cp.async pipeline is generated.  
5. Run `inspect_ir("target")` → confirm FP8 ops accumulate in FP32 PTX.  

This pipeline ensures correctness, performance, and portability.  

---

## 13. Programmer Summary Table

| Compiler Stage | Inspect Command        | What You Debug              | Example Fix |
|----------------|-----------------------|-----------------------------|-------------|
| Graph IR       | `.inspect_ir("graph")` | Shapes, dtypes, privileges  | Add `Region[reduce_sum]`, fix ShardSpec |
| Schedule IR    | `.inspect_ir("sched")` | Tiling, autotuning, fusion  | Override schedule, adjust block size |
| Tile IR        | `.inspect_ir("tile")`  | Memory, cp.async, barriers  | Swizzle layout, reduce vector width |
| Target IR      | `.inspect_ir("target")`| Backend intrinsics, numerics| Add `@accum(fp32)`, confirm WGMMA ops |

---

## 14. Final Thoughts for Programmers

- You don’t need to learn compiler internals, but you **do** need to know where to look.  
- Use **Graph IR** for correctness, **Schedule IR** for performance planning, **Tile IR** for deep perf debugging, **Target IR** for backend correctness.  
- Tessera makes all stages **inspectable and debuggable** with `inspect_ir(stage)`.  

---
