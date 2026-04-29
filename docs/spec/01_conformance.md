---
status: Normative
classification: Normative
authority: Conformance profiles; defers API and compiler disputes to docs/README.md
last_updated: 2026-04-26
---

# Tessera Conformance Specification (Normative)

**Version:** 0.3.0  
**Status:** Active — Phases 1–3 complete; Phases 4–6 planned  
**Authority:** This document defines conformance profiles. API, compiler, IR, and runtime disputes are resolved by the authority tree in `docs/README.md`.

---

## 1. Scope

This document defines what it means for a Tessera implementation to be
**conformant**. It specifies:

- The three conformance profiles (T0, T1, T2) and their mandatory feature sets
- Phase-to-profile mapping (what each development phase contributes)
- Testability criteria — how conformance is demonstrated
- Required error behavior for non-conformant inputs

Implementations that satisfy a profile's requirements **shall** be documented as
"Tessera T0/T1/T2 conformant" respectively. Partial compliance is not
conformance.

---

## 2. Normative References

| Document | Role |
|----------|------|
| `docs/spec/PYTHON_API_SPEC.md` | Python surface API — all public symbols |
| `docs/spec/COMPILER_REFERENCE.md` | IR stack, named pass pipelines |
| `docs/spec/GRAPH_IR_SPEC.md` | Graph IR op semantics and verifier rules |
| `docs/spec/LOWERING_PIPELINE_SPEC.md` | Pass input/output contracts |
| `docs/spec/TARGET_IR_SPEC.md` | Target IR dialects |
| `docs/spec/MEMORY_MODEL_SPEC.md` | Memory ordering, visibility, synchronization, atomics, and mbarriers |
| `docs/spec/RUNTIME_ABI_SPEC.md` | C ABI functions and types |
| `docs/spec/02_language_spec.md` | Supporting language notes |
| `docs/spec/04_tile_ir.md` | Supporting Tile IR notes |
| `docs/spec/shape-system.md` | Supporting shape-system notes |

---

## 3. Definitions

**Conformant Implementation** — a Tessera compiler and runtime that, for a
given profile, correctly processes all valid programs in that profile's subset
and correctly rejects all invalid programs with the required error codes.

**Valid Program** — a Python program using the Tessera API that satisfies all
type annotations, structural constraints, and effect contracts defined in the
relevant spec sections.

**Invalid Program** — a program that violates at least one of the above. A
conformant implementation **shall** reject it at decoration time (for `@jit`
constraints), compile time (for IR verifier violations), or before kernel launch
(for launch-parameter violations).

**Decoration Time** — when a `@tessera.jit` or `@tessera.kernel` decorator
executes. This is the earliest error-detection point and is where constraint
checking runs.

**Compile Time** — when the compiler pipeline (one of the two named pipelines)
processes the Graph IR. Shape inference and verifier passes run here.

---

## 4. Conformance Profiles

There are three conformance profiles. They are cumulative: T1 requires all of
T0, and T2 requires all of T1.

### 4.1 Profile T0 — Kernel-Only

**Purpose:** Minimal viable Tessera. A developer can write and compile
single-GPU kernels with type-checked annotations, shape constraints, and Graph
IR emission. No multi-rank support.

**Required capabilities:**

| Capability | Spec reference |
|-----------|----------------|
| `@tessera.jit` decorator (bare and with kwargs) | `PYTHON_API_SPEC §2` |
| `@tessera.kernel` decorator | `PYTHON_API_SPEC §3` |
| `tessera.Region["read"/"write"/"reduce_sum"/"reduce_max"/"reduce_min"]` | `PYTHON_API_SPEC §4` |
| `tessera.Tensor["dim", ...]` subscript annotation | `PYTHON_API_SPEC §16` |
| `tessera.f16[..., ...]`, `tessera.mut_f32[..., ...]` dtype annotations | `PYTHON_API_SPEC §16` |
| `tessera.require()` + `ConstraintSolver` predicates (`Divisible`, `Range`, `Equal`) | `PYTHON_API_SPEC §11`, `shape-system.md` |
| `EffectLattice` inference (pure → random → memory → io → top) | `PYTHON_API_SPEC §12`, `02_language_spec.md §5` |
| `@tessera.jit(deterministic=True, seed=N)` contract | `PYTHON_API_SPEC §2` |
| All 6 Graph IR ops (`tessera.matmul`, `conv2d`, `flash_attn`, `fused_epilogue`, `cast`, `transpose`) | `GRAPH_IR_SPEC §3` |
| 4 Graph IR canonicalization patterns | `GRAPH_IR_SPEC §5` |
| Graph IR verifier: module version attr, effect attrs, shape consistency | `GRAPH_IR_SPEC §6` |
| `fn.graph_ir.to_mlir()` — MLIR text emission | `PYTHON_API_SPEC §2` |
| `tessera.ops.*` — 15 public ops (gemm through cross_entropy) | `PYTHON_API_SPEC §15` |
| 7 error types (`TesseraConstraintError` through `TesseraRuntimeError`) | `PYTHON_API_SPEC §17` |
| x86 AMX/AVX512 GEMM execution path (`tessera-lower-to-x86` pipeline) | `COMPILER_REFERENCE §3.1` |

**Required error behavior under T0:**

- `TesseraConstraintError` raised at decoration time when a concrete binding violates `Divisible`, `Range`, or `Equal`
- `TesseraEffectError` raised at decoration time when `deterministic=True` and the function body contains unseeded random ops
- Graph IR verifier raises `TesseraCompileError` (or equivalent) for missing `tessera.version` attribute

**Test suite:** `tests/unit/` (all tests) + `tests/unit/test_lowering_chain.py`

---

### 4.2 Profile T1 — Single-Node

**Purpose:** Full single-machine GPU execution. Adds device memory management,
streams, events, and GPU kernel launch. Includes FlashAttention on SM_90.

**Phase status:** The Phase 3 compiler subset of T1 exists today: GPU target
configuration, FA-4 Tile IR, and the `tessera-lower-to-gpu` pipeline. Full T1
runtime conformance requires the Phase 6 runtime C ABI execution path.

**Requires:** All of T0, plus:

| Capability | Spec reference |
|-----------|----------------|
| `tessera.domain.Rect` | `PYTHON_API_SPEC §5` |
| `tessera.dist.Block`, `.Cyclic`, `.Replicated` | `PYTHON_API_SPEC §6` |
| `tessera.array.from_domain` | `PYTHON_API_SPEC §7` |
| `DistributedArray.parts(axis)` | `PYTHON_API_SPEC §8` |
| `ShardSpec` with integer `partition` tuples | `PYTHON_API_SPEC §9`, `shape-system.md §4` |
| `tessera.index_launch` | `PYTHON_API_SPEC §10` |
| `GPUTargetProfile`, `ISA` enum (SM_80–SM_120 placeholder) | `PYTHON_API_SPEC §13` |
| `FlashAttnLoweringConfig` | `PYTHON_API_SPEC §14` |
| `@tessera.jit(target=GPUTargetProfile(...))` routing | `COMPILER_REFERENCE §3.2` |
| `tessera-lower-to-gpu` pipeline (all 9 passes) | `COMPILER_REFERENCE §3.2` |
| FA-4 Attn dialect: `ScaledDotProduct`, `OnlineSoftmax`, `LseAccumulate`, `DropoutMask`, `CausalMask` | `TARGET_IR_SPEC §3`, `04_tile_ir.md §4` |
| `tile.async_copy` / `tile.wait_async` with `stage=` attribute | `04_tile_ir.md §3.2` |
| Hopper+ `tile.mbarrier.*` transaction barriers | `LANGUAGE_AND_IR_SPEC §9`, `MEMORY_MODEL_SPEC §4` |
| `tessera.tma.*` + `tessera.nvgpu.wgmma.*` ops (SM_90 path) | `TARGET_IR_SPEC §4–5` |
| `WarpSpecializationPass` producer/consumer role assignment | `LOWERING_PIPELINE_SPEC §2.2` |
| Runtime C ABI lifecycle: `tsrInit` through `tsrShutdown` | `RUNTIME_ABI_SPEC §5.1` (Phase 6 runtime requirement) |
| `tsrMalloc`, `tsrFree`, `tsrMemcpy`, `tsrMemset` | `RUNTIME_ABI_SPEC §5.5` (Phase 6 runtime requirement) |
| `tsrCreateStream`, `tsrStreamSynchronize` | `RUNTIME_ABI_SPEC §5.3` (Phase 6 runtime requirement) |
| `tsrLaunchHostTileKernel` / `tsrLaunchHostTileKernelSync` | `RUNTIME_ABI_SPEC §5.6` (Phase 6 runtime requirement) |
| `MockRankGroup` for multi-rank testing without NCCL | `PYTHON_API_SPEC §18` |

**Required error behavior under T1 (in addition to T0):**

- `TesseraPrivilegeError` raised when a `write`-annotated region arg is passed to a function that already holds a `write` region on overlapping data
- `TsrStatus::TSR_ERROR_INVALID_STREAM` returned (not a crash) when stream arg is null
- `TsrStatus::TSR_ERROR_OUT_OF_MEMORY` returned when device memory is exhausted

**Test suite:** T0 suite + `tests/unit/` (all tests) + `tests/tessera-ir/phase3/`

---

### 4.3 Profile T2 — Cluster

**Purpose:** Multi-node distributed training. Adds NCCL/RCCL collectives, TPU
backend, pipeline parallelism, and Cyclic MoE distribution.

**Requires:** All of T1, plus:

| Capability | Spec reference |
|-----------|----------------|
| `tessera.dist.Cyclic` full implementation (no `NotImplementedError`) | `PYTHON_API_SPEC §6` |
| `NCCLAdapter` / `RCCLAdapter` with `all_reduce`, `reduce_scatter`, `all_gather` | `RUNTIME_ABI_SPEC §7` (Phase 6) |
| `GPUCollectiveInsertionPass` at DP mesh boundaries | `LOWERING_PIPELINE_SPEC` (Phase 4) |
| `PipelineStageInsertionPass` — 1F1B schedule | `LOWERING_PIPELINE_SPEC` (Phase 4) |
| TPU StableHLO backend (`tessera-lower-to-tpu` pipeline) | `TARGET_IR_SPEC` (Phase 4) |
| `collective.reduce_scatter` / `collective.all_gather` IR ops | `TARGET_IR_SPEC §6` |
| Shardy mesh export | `TARGET_IR_SPEC §7` |

> **Phase status:** T2 capabilities are **Phase 4 planned**. No implementations
> exist yet. A conformant T2 implementation is not achievable before Phase 4 is
> complete.

---

## 5. Phase-to-Profile Mapping

| Phase | Profile contribution | Status |
|-------|---------------------|--------|
| Phase 1 | T0 Python frontend (decorators, Region, domain, dist, constraints, effects, Graph IR) | ✅ Complete |
| Phase 2 | T0 x86 lowering chain (all 4 passes, `tessera-lower-to-x86`) | ✅ Complete |
| Phase 3 | T1 compiler subset (GPUTargetProfile, FA-4 Tile IR, 9-pass GPU pipeline) | ✅ Complete |
| Phase 4 | T2 distributed (NCCL, TPU, Cyclic, pipeline parallelism) | 🔲 Next |
| Phase 5 | T1 hardening (checkpointing, ZeRO sharding, Bayesian autotuner) | 🔲 Future |
| Phase 6 | T1/T2 ROCm full MFMA, runtime C ABI wired, benchmarks | 🔲 Future |

A **T0-conformant** implementation exists today (Phases 1–2).  
A **T1 compiler-subset implementation** exists today (Phase 3). Full T1
conformance is pending Phase 6 runtime ABI wiring.  
A **T2-conformant** implementation requires Phase 4 completion.

---

## 6. Compliance Testing

### 6.1 Test Suite Structure

Conformance is demonstrated by running the official test suites and achieving
100% pass rate with no skipped tests (unless the test requires unavailable
hardware).

| Profile | Required test suites | How to run |
|---------|---------------------|------------|
| T0 | `tests/unit/`, `tests/unit/` | `pytest tests/unit/ tests/unit/ -v` |
| T1 | T0 + `tests/unit/`, `tests/tessera-ir/phase3/` | `pytest tests/unit -v && python -m lit tests/tessera-ir/phase3/ -v` |
| T2 | T1 + `tests/unit/`, `tests/tessera-ir/phase4/` | `pytest tests/unit -v && python -m lit tests/tessera-ir/ -v` |

GPU tests that require SM_90+ hardware may be skipped with `@pytest.mark.skipif`
when no GPU is present. Skipping these does **not** disqualify T1 conformance
provided the software path (mock/CPU fallback) tests pass.

### 6.2 Required Test Outcomes

A conformant implementation **shall**:

1. Pass all tests without modification to the test files
2. Produce the correct Graph IR text for each canonicalization pattern
   (checked via `assert "tessera.matmul" in ir` style assertions in phase1 tests)
3. Correctly raise each error type with the expected message format
4. Correctly reject invalid programs (negative tests marked with `pytest.raises`)

### 6.3 Static Conformance Checks

In addition to runtime tests, conformance requires passing type checking:

```bash
mypy python/tessera/distributed/ python/tessera/compiler/ --strict
```

Zero errors are required. Warnings are permitted.

### 6.4 Lit Test Conformance (T1+)

The MLIR lit tests verify IR structural properties that cannot be tested at
the Python layer:

```bash
python -m lit tests/tessera-ir/ -v
```

Each lit test uses `// CHECK:` directives. A conformant implementation shall
produce output matching every `CHECK` line in every lit test file for the
relevant profile's test directory.

---

## 7. Non-Conformant Behavior Requirements

An implementation that does not conform to a profile **shall not** claim
conformance with that profile. It **may** claim partial feature support by
listing the specific spec sections it satisfies.

### 7.1 Error Reporting

A conformant implementation **shall** report violations with:

- The correct exception type (one of the 7 types in `PYTHON_API_SPEC §17`)
- A human-readable message identifying the offending dimension path, op, or
  constraint
- Source file and line number when the MLIR `loc` attribute is available

A conformant implementation **shall not**:

- Silently produce wrong results for invalid inputs
- Crash (segfault, OOM kill, unhandled exception in C++) on any input that
  should be rejected at decoration or compile time
- Produce IR that fails its own verifier

### 7.2 Undefined Behavior

The following situations are **undefined behavior** and are explicitly outside
the conformance contract:

- Calling any `tsr*` C ABI function with a null pointer argument (unless the
  function's spec explicitly permits null)
- Calling `tsrLaunchHostTileKernel` with a `tile_count` of zero
- Modifying a tensor that is annotated `Region["read"]` from within the same
  `@jit` scope
- Using `@tessera.kernel` with a function that performs host I/O (print,
  file write, etc.)

---

## 8. Versioning and Forward Compatibility

Tessera conformance profiles are versioned together with the project.

**Current conformance version:** 0.3.0

Profile requirements **shall not** be removed in a minor version bump
(0.x.0). They **may** be added in a minor version. They **may** be added,
removed, or changed in a major version.

The `tessera.version` Graph IR module attribute **shall** match the compiler
version. The verifier rejects modules with a version mismatch.

---

## Appendix A — Conformance Checklist (Informative)

For implementers, a quick checklist to verify T0 conformance:

- [ ] `pytest tests/unit/ -v` — all green
- [ ] `pytest tests/unit/ -v` — all green
- [ ] `mypy python/tessera/ --strict` — no errors
- [ ] `@tessera.jit` with no args decorates successfully
- [ ] `@tessera.jit(deterministic=True)` rejects an unseeded dropout function
- [ ] `tessera.require(Divisible("K", 100))` with `bindings={"K": 100}` passes
- [ ] `tessera.require(Divisible("K", 100))` with `bindings={"K": 99}` raises `TesseraConstraintError`
- [ ] `fn.graph_ir.to_mlir()` returns a string containing `tessera.matmul`
- [ ] The `tessera-lower-to-x86` pipeline runs without error on a 2-op GEMM module
- [ ] All 4 canonicalization patterns fire in the Graph IR test suite

For T1, additionally:

- [ ] `GPUTargetProfile(isa=ISA.SM_90)` accepted without error
- [ ] `pytest tests/unit/ -v` — all green
- [ ] `python -m lit tests/tessera-ir/phase3/ -v` — all pass
- [ ] `tessera.flash_attn` in Graph IR lowers to `tessera.attn.scaled_dot_product`
- [ ] `WarpSpecializationPass` assigns producer/consumer roles
- [ ] WGMMA fallback path fires for `isa < ISA.SM_90`
