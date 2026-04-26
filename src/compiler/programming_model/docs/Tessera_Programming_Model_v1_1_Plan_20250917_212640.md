<!-- === MERGE_START: Tessera Programming Model v1.1 Update Plan (Document 1/2) === -->

# Tessera Programming Model — v1.1 Update Plan

> Scope: sharpen the **programming model contracts** (types, effects, numerics, memory, parallelism), align **IR layers** with those contracts, and harden **Target‑IR** with verifiers + tests. This plan is designed to be executed incrementally over ~3–5 short milestones.

---

## 0) Guiding Principles

- **Contracts before codegen**: Every user‑visible promise (shape, effects, numerics, determinism) must have a verifier at the earliest IR where it becomes checkable.
- **One concept, one place**: Keep semantics in the **programming model** docs; put mechanical details (dialect ops, attrs, passes) in IR docs.
- **Continuity**: Changes are staged so existing examples run; new features are opt‑in via flags or contexts.
- **Proof via tests**: Each new construct lands with FileCheck tests and 1 runnable example.

---

## 1) Frontend Contracts (Rust core + Python API)

### 1.1 Shape‑aware Types (compile‑time constraints)
- Add constraint syntax and propagation:
  - `D % 8 == 0`, `S <= 8192`, `B in 1..=4096`
  - Broadcast rules validated at parse/type‑check time
- Symbolic/partially‑dynamic shapes with **static anchors** for tile specialization.
- Error messages standardized: show the **offending dimension path** and the **expected rule**.

**Artifacts**
- Rust: `ConstraintSolver` extensions (divisibility, ranges, equalities).
- Python: type hints preserved and shown in `__repr__` of tensors/functions.
- Tests: invalid broadcast, invalid constraints, specialization cache hit.

### 1.2 Effect Types
- First‑class effects on function signatures: `["random", "io", "memory"]`.
- Effect inference: composition rules and suppression (`with effects.suspend("io")`).
- Determinism contract: a function marked `@deterministic(seed=…)` forbids nondet ops unless wrapped.

**Artifacts**
- Rust: effect lattice + propagation over AST → Graph IR attrs.
- Tests: pipeline rejects nondet op in deterministic scope; suspension works.

### 1.3 Python ergonomics
- `@function` decorator: capture source, compile options; expose `.profile()`, `.autotune()`.
- Error surface: normalized `CompilationError` with span + fix‑it where possible.

---

## 2) Numerical Policy System

### 2.1 Policy Objects
- Separate **storage / compute / accumulate / rounding / saturation / denorm**.
- Predefined policies: `Training(bf16→f32)`, `Inference(fp8→bf16)`, `NVFP4(accum=f16/f32)`.
- **AdaptivePrecision** hook: policy switch based on gradient scale or tensor range.

### 2.2 Contracts & Verifiers
- Verifier: ops that reduce/accumulate must state accumulation dtype or inherit policy.
- Mixed‑precision lowering must insert `quant/dequant` with explicit rounding.

**Artifacts**
- Graph IR attrs: `tessera.numerics.policy = {…}`
- Lowering: explicit cast/round/saturate ops in Tile/Target IR.
- Tests: policy passthrough, illegal implicit downcast, stable softmax numerics.

---

## 3) Memory & Execution Model

### 3.1 Memory Spaces
- Canonical spaces: `register`, `shared/LDS`, `global`, `managed`, `host`, `tmem (SM_100+)`.
- Declarative placement & lifetimes; async copies with stage indices for double/triple buffering.
- KV‑cache contract (paged, ring‑buffer, eviction policy).

### 3.2 Determinism & Ordering
- Ordered reductions when `@deterministic` is set (tree reduction recipe).
- RNG streams keyed by (function, mesh‑coords, step).

**Artifacts**
- Tile IR: `async_copy`, `wait_async(stage)`, `alloc_shared(swizzle=…)` become verified ops.
- Target IR: fast‑paths (cp.async/TMA, WMMA/WGMMA, AMD WMMA, Blackwell TMEM).
- Tests: cp.async sequence correctness; ring‑buffer page table smoke test.

---

## 4) Parallelism Constructs

### 4.1 Mesh / Data / Model / Pipeline
- Mesh axis contexts (`with mesh.axis("data"):`) map to Schedule IR regions.
- Pipeline schedules (`1F1B`, interleaved) as attrs + verifier of stage DAG.
- MoE expert parallel: A2A planner contract + load‑balancing hook (token throttling).

**Artifacts**
- Schedule IR ops: `mesh.region`, `pipeline.region(schedule="1f1b")`.
- Tests: shape of collectives, overlap with compute, A2A bucketization plan.

---

## 5) IR Layer Tightening

### 5.1 Graph IR
- Custom VJP/JVP hooks; autodiff safety (side‑effect barriers).
- Numerics & effect attributes propagated down.

### 5.2 Schedule IR
- Halo‑inference pass (for stencils) + pipeline‑overlap modeling.
- Mesh axis legality checks (collectives, shard maps).

### 5.3 Tile IR
- Warp/block primitives (shuffle, butterfly, vote) with types.
- Explicit register/shared usage caps for occupancy analysis.

### 5.4 Target IR
- **Add verifiers** for:
  - `kernel(config=…)` ranges (regs, smem), `launch` arity/types.
  - `ptx_instr/hip_instr` operand/constraint shape.
  - Memory ops (address spaces, align, async usage).
- Platform feature gates (WGMMA, TMEM, WMMA RDNA3).

**Artifacts**
- ODS updates in `tessera_target` + C++ verifiers.
- Conversion patterns: Tile→NVVM/ROCDL for GEMM, copy, barrier.

---

## 6) Tooling & CI

- **FileCheck** tests per dialect + end‑to‑end mini examples.
- `tessera-opt` pipelines: `-pm-v1_1-legalize`, `-pm-v1_1-verify` aliases.
- GitHub Actions: build + lit + example smoke on CUDA/ROCm matrix when available.
- Perf tooling: roofline CSV + Perfetto JSON emitted from example runs.

---

## 7) Milestones (suggested)

- **M1 (Foundation)**: Frontend constraints/effects; Target‑IR verifiers; FileCheck scaffolding.
- **M2 (Numerics)**: Policies + stable softmax; MP cast/round; tests.
- **M3 (Memory/Parallel)**: Async copies + pipeline region + mesh checks.
- **M4 (Backends)**: WGMMA/TMA/TMEM + AMD WMMA routes; minimal kernels.
- **M5 (Docs & Examples)**: Guide updates + 3 runnable examples (matmul, softmax, flash‑attn).

---

## 8) Cross‑refs into current repo (where to patch)

- `frontend/` (Rust core): constraint solver, effect lattice, Python bindings.
- `ir/graph`, `ir/schedule`, `ir/tile`, `ir/target`: ODS + verifiers + passes.
- `tools/tessera-opt`: pass registrations & pipeline aliases.
- `tests/`: FileCheck + examples.
- `docs/`: Programming Model v1.1 deltas, IR guides, examples.

---

## 9) Acceptance Criteria

- All new contracts have verifiers and at least one failing test.
- At least one end‑to‑end example per milestone.
- CI is green across CPU‑only and CUDA builds; ROCm behind flag if needed.

---

*Prepared for integration into `gstoner/tessera` as v1.1 delta plan.*

<!-- === MERGE_END: Tessera Programming Model v1.1 Update Plan (Document 1/2) === -->
