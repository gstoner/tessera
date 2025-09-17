<!-- === MERGE_START: Tessera Programming Model v1.1 Task Checklists (Document 2/2) === -->

# Tessera Programming Model — v1.1 **Task Checklists & Specs**

This file is a “do‑the‑work” ledger: concrete tasks, checklists, lit patterns, and stubs.

---

## A) Frontend (Rust core + Python)

### A.1 Constraint Solver
- [ ] Add predicates: `Divisible(dim, k)`, `Range(dim, lo, hi)`, `Equal(dimA, dimB)`
- [ ] Broadcast unification; emit minimal counterexample in error
- [ ] Cache specializations by static anchors (e.g., head_dim)

**BNF sketch (additions)**
```
constraint   ::= arith_cmp | divisibility | range | equal
divisibility ::= ident "%" NUMBER "==" 0
range        ::= ident ( "<=" | "<" | ">=" | ">" ) NUMBER
equal        ::= ident "==" ident
```

### A.2 Effects
- [ ] Effect lattice + inference
- [ ] `@deterministic(seed=…)` scope guard; forbid nondet ops unless wrapped
- [ ] Python `with effects.suspend("io")` plumbs into IR attr

**Signature example**
```python
@tessera.effects(["random","memory"])
def dropout(x: T) -> T: ...
```

---

## B) Numerical Policies

- [ ] Policy structs + registry; predefined Training/Inference/NVFP4
- [ ] Cast/round ops in Tile IR; explicit accum dtype on reductions
- [ ] Stable softmax primitive; causal mask path

**FileCheck (policy‑downcast rejection)**
```
// RUN: tessera-opt %s -pm-v1_1-verify -split-input-file | FileCheck %s
// CHECK: error: illegal implicit downcast from f32 to bf16
```

---

## C) Memory & Determinism

- [ ] Tile IR: `async_copy(src, dst, stage)`, `wait_async(stage)`; verifier for stage DAG
- [ ] Target IR: cp.async/TMA (NV), LDS ops (AMD), TMEM (SM_100) gates
- [ ] KV‑cache: ring buffer + page table IR ops; eviction attr

**FileCheck (cp.async sequence)**
```
// CHECK: tessera_target.ptx_instr "cp.async"{{.*}}
// CHECK: tessera_target.ptx_instr "cp.async.commit_group"
// CHECK: tessera_target.ptx_instr "cp.async.wait_group"{{.*0}}
```

---

## D) Parallelism Constructs

- [ ] Schedule IR: `mesh.region{axis="data"}`; `pipeline.region(schedule="1f1b")`
- [ ] MoE planner attrs: bucket size, pack/cast fuse, token limiter id
- [ ] Collectives legality: verify shard maps vs mesh axes

**FileCheck (pipeline)**
```
// CHECK: schedule.pipeline_region { schedule = "1f1b" }
```

---

## E) Target IR Hardening

- [ ] ODS verifiers for `kernel`, `launch`, `memcpy`, `allocate`, `ptx_instr`, `hip_instr`
- [ ] Feature flags: `wgmma`, `tma`, `tmem`, `wmma_rnda3`
- [ ] Conversion: Tile→NVVM (WGMMA/WMMA), Tile→ROCDL (WMMA RDNA3)

**Verifier cases**
- Missing `grid/block` ranks, negative sizes
- Shared‑mem bytes exceed device cap (allow configurable caps)
- Operand count/type mismatch for inline instr ops
- Async copy without `wait` in same region

---

## F) Tooling & Pipelines

- [ ] `-pm-v1_1-legalize` pipeline alias (graph→schedule→tile→target baseline)
- [ ] `-pm-v1_1-verify` pipeline alias (run all new verifiers)
- [ ] `tests/pm_v1_1/` with 20–30 focused FileCheck cases
- [ ] Three runnable examples (CMake + scripts): matmul, softmax, flash‑attn

**Example lit skeleton**
```
// RUN: tessera-opt %s -pm-v1_1-verify | FileCheck %s
tessera_target.kernel @k() config = #tessera_target.kernel_config<
  grid=[128,64], block=[256,1], shared=49152, regs=128> { 
  tessera_target.return
}
// CHECK: tessera_target.kernel @k
// CHECK-SAME: shared = 49152
```

---

## G) Documentation Tasks

- [ ] Update **Programming Model** guide with v1.1 deltas and examples
- [ ] Add **IR Usage Guides** for new ops/attrs (Graph/Schedule/Tile/Target)
- [ ] “Determinism & Effects” mini‑guide with do/don’t table
- [ ] “Numerical Policies” mini‑guide with recipes (training, inference, fp8/fp4)

---

## H) Milestone Exit Criteria

**M1 Foundation**
- [ ] Constraints + effects verified on 10+ lit cases
- [ ] Target‑IR kernel/launch/memcpy verifiers merged
- [ ] One end‑to‑end example runs

**M2 Numerics**
- [ ] Policies available + enforced; stable softmax green
- [ ] MP casts inserted by lowering; tests cover rounding modes

**M3 Memory/Parallel**
- [ ] Async copy + pipeline region; cp.async/TMA/TMEM gates
- [ ] Mesh legality & a tiny MoE A2A plan validated

**M4 Backends**
- [ ] WGMMA/WMMA/WMMA‑RDNA3 paths land; GEMM smoke tests pass

**M5 Docs**
- [ ] Guides updated; examples documented; CI matrix green

---

## I) Skeleton stubs to add (paths)

```
frontend/rust/core/constraints.rs
frontend/rust/core/effects.rs
python/tessera/effects.py
ir/graph/GraphOps.td
ir/schedule/ScheduleOps.td
ir/tile/TileOps.td
ir/target/TesseraTargetOps.td (verifiers++)
lib/Conversion/TileToNVVM/
lib/Conversion/TileToROCDL/
tools/tessera-opt/PassPipelinesPM11.cpp
tests/pm_v1_1/*.mlir
docs/programming_model_v1_1.md
docs/numerics_policies.md
docs/determinism_and_effects.md
```

---

## J) Risk & Mitigation

- **Risk**: verifier strictness breaks existing samples  
  **Mitigation**: staged “warn‑only” mode behind `-pm-strict=0/1`
- **Risk**: backend feature gaps (TMEM/WMMA‑RDNA3) delay milestones  
  **Mitigation**: guard with feature flags; CI skips when unavailable
- **Risk**: complexity creep in policies  
  **Mitigation**: registry with a few blessed presets + extension hook

---

*Use this file as the live checklist during implementation.*

<!-- === MERGE_END: Tessera Programming Model v1.1 Task Checklists (Document 2/2) === -->
