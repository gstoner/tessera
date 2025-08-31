# Tessera Compiler Frontend — Design for Graph IR Ingestion
*Scope:* Lexer → Preprocessor → Parser → Semantic Analyzer → Graph IR Generator  
*Status:* Draft v0.2 (with programmer context)

---

## 0. Goals & Non‑Goals
**Goals**
- Parse the Pythonic Tessera surface into a **typed, effect-aware, distribution-aware Graph IR**.
- Make **tiles, domains/distributions, region privileges, numerics policies, collectives, and transforms (jit, vmap, pmap, scan, checkpoint)** first-class in the IR.
- Provide **precise diagnostics** (source ranges, quick-fix hints) and **shape/type inference** with symbolic dims.
- Emit IR that is stable and deterministic, ready for **Schedule IR** lowering (fusion/tiling/pipelining).

**Non‑Goals**
- Target lowering (PTX/Tile IR) and runtime execution (NCCL, CUDA Graphs) — handled later.
- Optimizations beyond light canonicalization; heavy rewrites live below the frontend.

---

## 1. Pipeline Overview

```text
Source (.tss, .py) 
  └─► Lexer (tokens)
        └─► Preprocessor (decorators, macros, attrs, policy sugar)
              └─► Parser (AST)
                    └─► Semantic Analyzer (symbols, types, shapes, effects, privileges)
                          └─► Graph IR Builder (GIR module, funcs, regions, ops, attrs)
                                └─► [handoff] Schedule IR (fusion/tiling) ► Tile IR ► Target IR
```

Artifacts kept for tooling:
- **SourceMap** (byte offsets → line/col), **TokenStream**, **AST**, **Typed AST (TAST)**, **GIR**.
- All diagnostics carry source ranges; TAST nodes link back to AST and tokens.

---

## 2–11. [Core Sections]
- Language surface, Lexer, Preprocessor, Parser, Semantic Analyzer, Graph IR generator.  
- Types, effects, region privileges, diagnostics, implementation notes, milestones.  
- *(unchanged from v0.1 — see earlier draft for details)*

---

## 12. For Programmers: Why Graph IR Matters

Graph IR (GIR) is the **first IR stage that programmers can inspect**. It tells you:  
- How your high-level Tessera function was understood by the compiler.  
- What **tensor shapes, dtypes, and policies** were inferred.  
- Whether your **ShardSpec, domains, and distributions** were legal.  
- How **region privileges** (`read`, `write`, `reduce`) were applied.  

Programmers don’t write GIR directly — but they **inspect it** to debug shape mismatches, type issues, and distribution errors.

---

## 13. Inspecting Graph IR

You can view Graph IR with:

```python
@jit @autodiff
def f(x, W): return gemm(x, W)

f.inspect_ir("graph")
```

Example output:
```mlir
gir.func @f(%x: gir.tensor<["B","D"], bf16, {accum=f32}>,
            %W: gir.tensor<["D","K"], bf16, {accum=f32}>)
        attributes {jit=true, autodiff=true} {
  %y = gir.gemm %x, %W
  gir.return %y
}
```

This shows you the **typed ops** the compiler will schedule, fuse, and lower.

---

## 14. Common Errors at Graph IR Stage

- **Shape mismatch**:  
  *“Expected rank-3 tensor, got rank-2 at param `x`.”*  

- **Privilege conflicts**:  
  *“Conflicting write privileges on `Y` — add `Region[reduce_sum]` if accumulation intended.”*  

- **Distribution mismatch**:  
  *“ShardSpec across axis ‘tp’ expects col partition; found row.”*  

- **Numerics policy violation**:  
  *“fp8_e4m3 matmul requires accum ≥ fp16; found accum=bf16.”*  

---

## 15. Programmer Workflow Example

1. You write:
   ```python
   @jit
   def bad_step(X: Region[write], Y: Region[write]):
       X[:] = Y
       Y[:] = X
   ```

2. You run:
   ```python
   bad_step.inspect_ir("graph")
   ```
   → Compiler error: *“Conflicting write privileges on overlapping regions.”*

3. You fix it:
   ```python
   @jit
   def good_step(X: Region[write], Y: Region[reduce_sum]):
       Y[:] += X
   ```

4. Re-run `inspect_ir("graph")` → now valid.

This workflow shows how **Graph IR gives early, precise feedback** before kernels are scheduled or launched.

---

## 16. Summary for Programmers

- **Graph IR** is where you check that Tessera understood your program correctly.  
- It validates shapes, types, privileges, distributions, and numerics.  
- Use **`inspect_ir("graph")`** whenever you hit shape/type errors or distributed mismatches.  
- Think of GIR as the **single source of truth** for how Tessera interprets your model before optimization.
