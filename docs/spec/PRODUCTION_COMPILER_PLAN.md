---
status: Ratified
classification: Design / Roadmap
authority: Production MLIR/LLVM compiler
last_updated: 2026-06-05
---

# Tessera Production Compiler Plan (MLIR/LLVM)

> **Status:** Ratified architecture (decisions D1–D5 locked).
>
> * **Phase 0 landed 2026-06-05** — boundary proof on `tessera.add`.
> * **Phase 1 Sprint 1.1 landed 2026-06-05** — JIT harness generalized
>   (`tessera_jit_invoke(handle, name, void** descs, int n)` via direct c-iface
>   dispatch; arity 1–8); generic rank-N f32 descriptor packing in Python;
>   `tessera.matmul` → `linalg.fill(0) → linalg.matmul` (first non-elementwise
>   op); binary elementwise family expanded to `add/sub/mul`.
> * **Phase 1 Sprint 1.2 landed 2026-06-05** — `tessera.reduce` (one
>   parameterized op, `kind` ∈ {sum,max,min,mean} × `axis`) →
>   `linalg.fill(identity) → linalg.reduce`, mean = sum + 1/N scale. **First op
>   whose result rank differs from its input rank.**
> * **Phase 1 Sprint 1.3 landed 2026-06-05** — `tessera.softmax` →
>   numerically-stable `max → (x−m) → exp → sum → (e/d)` decomposition. **First
>   broadcast** (reduced `(…)` tensor applied against full `(…,N)` input via an
>   affine map dropping the reduced axis — `emitBroadcastBinary`, reused by
>   normalization next) and **first use of the `math` dialect** (`math.exp`,
>   wired through `convert-math-to-llvm` in the JIT). Elementwise family
>   completed with `tessera.div`. **55/55 production-lane tests green**
>   (adds `tests/unit/test_production_jit_phase1_softmax.py`), incl. a
>   large-value (1000–1002) stability test that a naive `exp` would overflow.
>
> Phase 1 *DoD* (the ~15 structural patterns covering the bulk of the op
> surface, plus bf16 boundary) is *in progress*. Next slices: normalization
> (layer_norm/rmsnorm — reuses `emitBroadcastBinary`), bf16 boundary.
> **Scope:** Evolve Tessera from a Python-interpreted prototype into a production
> MLIR/LLVM-IR compiler, while retaining the Python compiler as the
> experimentation lane. This document is the committed decision record; it gates
> all sprint work below.

---

## 0. Two lanes

- **Python lane = experimentation.** Fast prototyping of new ops and
  programming-model ideas. Eager numpy/Accelerate/ctypes execution. The registries
  and the eager interpreter live here. Allowed to be loose — it is a lab.
- **MLIR/LLVM lane = production.** Real codegen, real execution on real silicon.
  This is what ships.

The lanes are connected by **oracle testing** (D4): the Python lane's numpy
reference is the production lane's test oracle. Nothing is promoted to production
without an oracle test that matches within tolerance.

Apple macOS (CPU + Metal 4 GPU) is the **production-grade end-to-end proving
ground** — the silicon on hand, used to expose ABI/dtype/shape/fallback/runtime
mistakes under real GPU conditions before NVIDIA/AMD add their own complexity.

---

## 1. Ratified decisions

### D1 — Keep `tessera` Graph IR as the stable apex; do not TOSA-ize the spine
`tessera` ops are `[Pure]` on `AnyRankedTensor` (`src/compiler/ir/TesseraOps.td`),
i.e. the value-semantic subset that lowers cleanly. The internal spine is
**`linalg` (on tensors) + `math` + `arith` + `tensor` + `scf`/`cf`**. TOSA is an
*ingestion-only* dialect (opinionated quant semantics, rank ceilings, fixed op
menu) — acceptable for *importing* external models, never a lowering target for
Tessera's own ops.

### D2 — Dialect-target map is per op-category, not uniform

| Category | Examples | Target | Notes |
|---|---|---|---|
| Pure tensor algebra | matmul, conv, norms, gelu/silu, reductions, reshape/transpose | `linalg` named + `linalg.generic` + `math`/`arith` | value-semantic; upstream → vector → llvm/gpu |
| Control flow | scan, while, fori, cond, cf_while | `scf` / `cf` | — |
| Stateful / effectful | KV-cache append/read, memory_read/write/evict, RNG state | **stay `tessera` ops w/ `MemoryEffects`**, lower late | → `memref` + `tsr*` runtime calls. **Never become `linalg.generic`.** |
| Scheduling / distribution | mesh, pipeline stages, sharding, collectives | `schedule` dialect (above linalg) | → collective runtime calls |
| Custom attention family | flash_attn, MLA, NSA, lightning, delta | high-level `tessera.attn` op | **structured op + generic fallback + target override** (correctness via tiled linalg/scf; performance via FA-4/MPS/MSL) |

### D3 — The compiled-function ABI is a first-class artifact, designed before nontrivial lowering
- **Calling convention:** MLIR C-ABI wrappers (`-llvm-request-c-wrappers` →
  `_mlir_ciface_<fn>`) taking packed **memref descriptors**
  `{alloc_ptr, aligned_ptr, offset, sizes[], strides[]}`. Descriptor carries
  shape/stride, so dynamic shapes are additive later, not a rewrite.
- **Ownership = Destination-Passing Style (DPS), caller-allocated.** Outputs are
  passed as `outs` memrefs. Aligns with linalg bufferization; avoids callee
  allocation becoming a permanent ABI wart. Callee-allocation exceptions are
  made **explicit later**, never baked into v1.
- **Layout/dtype contract:** boundary memrefs are identity-layout, C-contiguous,
  or the boundary inserts a copy.
- **bf16/dtype policy (ABI rule, not implementation accident):** `ml_dtypes` on
  the Python side, **raw 16-bit at the MLIR/runtime boundary**, copy/convert on
  mismatch.
- **Integration:** a *new* compiled-codegen ABI alongside the existing
  `tsrLaunchHostTileKernel` shim. `canonical_compile(target="cpu")` returns a
  callable bound through `mlir::ExecutionEngine`; the `tsr*` malloc/stream/event
  surface remains the device-memory/async layer underneath.

### D4 — Two lanes connected by oracle testing
Production lane is **green iff its codegen output matches the Python lane within
tolerance.** This makes experimentation *feed* production rather than fork from it.

### D5 — Production apex is a verified *subset* of the Tessera dialect; promotion is explicit
The Python Graph IR may emit a **superset** of what production accepts. The
production-accepted set is partitioned by op category (D2). **Promotion across
the boundary is explicit** and recorded — each promoted op has an oracle test
that admitted it. The coverage registry's job becomes the **promotion ledger**
(is op X in the production subset; what test admitted it), replacing aspirational
status-tracking.

---

## 2. Honest hard-problems register

Upstream MLIR provides correctness primitives and host-side plumbing; it does
**not** provide a competitive kernel. Each hard problem is assigned a phase so it
is confronted, not hidden inside "swap the back-half."

| Hard problem | Reality | Phase |
|---|---|---|
| Bufferization + ownership | Tractable on CPU with DPS; discipline starts at the ABI | 0/1 |
| Memory spaces (shared/global/register) | linalg bufferization is space-agnostic; needs promotion passes | 4 |
| Async copy / `cp.async` / TMA | `nvgpu` ops exist; double-buffer pipelining is hand-assembled | 5 |
| mbarrier / barrier sequencing | `nvgpu.mbarrier` exists; correct sequencing is manual | 5 |
| Target matmul forms (WGMMA/MMA/MFMA) | `nvgpu.warpgroup.mma` / `amdgpu.mfma` exist; shape selection + fragment layouts are real work (current `NVWGMMALoweringPass` stops at `func.call`) | 5 |
| Performance legality (occupancy, bank conflict, swizzle, reg pressure) | Entirely ours | 5 |
| Dynamic shapes | Descriptor carries it; lowering + guards are work | post-1 |
| Apple has no upstream Metal/AIR target | Metal back-half is permanently bespoke | 3 |

---

## 3. Phased roadmap

**Phase 0 — The Boundary.** Op: elementwise `add` (deliberately trivial so the
sprint is ONLY the ABI). DoD: ABI spec written; `tessera.add` → `linalg`/`arith`
→ llvm → ExecutionEngine; `canonical_compile(target="cpu")` returns a callable;
DPS round-trips; oracle test vs Python lane passes. NOT in scope: matmul, tiling,
GPU, dynamic shapes, state.

**Phase 1 — CPU coverage via linalg.** matmul, reductions, norms, elementwise,
softmax → linalg → vectorize → llvm. DoD: ~15 structural patterns covering the
bulk of the ~100 Python ops; all oracle-tested; bf16 boundary works. Performance
reasonable, not tuned. NOT: GPU, attention fusion, state, perf tuning.

**Phase 2 — State & control flow, honestly.** KV-cache/memory/RNG as effectful
ops → `memref` + `tsr*` calls; scan/while/cond → `scf`. DoD: a stateful decode
step runs end-to-end on CPU through the production lane, oracle-matched. NOT:
perf, GPU.

**Phase 3 — Apple GPU end-to-end (production milestone on real silicon).** linalg
front-half + bespoke Metal back-half (MTL4 GEMM / MPS / MPSGraph / MSL) as target
override; attention via D2 override. DoD: a full transformer block runs
production-grade on this Mac's GPU, oracle-matched, hand-tuned Metal kernels as
fast-path. **First point at which "functional AND production-grade end-to-end"
is true.** NOT: NVIDIA/AMD.

**Phase 4 — NVIDIA correctness-first.** linalg → `gpu` → `nvgpu`/`nvvm` → PTX via
the generic pipeline; confront memory spaces + bufferization-to-GPU. DoD: kernels
run correctly on NVIDIA, oracle-matched, even if slow.

**Phase 5 — Performance legality + target matmul forms + AMD.** WGMMA/MFMA,
async-copy/TMA double-buffering, mbarrier, occupancy/swizzle tuning; AMD
ROCDL/MFMA back-half. DoD: competitive MFU on headline kernels.

**Critical path:** D3 (ABI) gates everything → Phase 0 → Phase 1. Phases 3 (Apple)
and 4 (NVIDIA) depend only on Phase 1's front-half, so after CPU they are
sequenced by **priority, not dependency**. Ratified priority: **Apple (3) before
NVIDIA (4)** — Apple is the silicon that can be continuously proven.

---

## 4. Relationship to existing code

- `python/tessera/compiler/graph_ir.py` — Python-lane IR producer (superset, D5).
  Stays. Must emit canonical MLIR syntax so the production lane can parse without
  the current `driver.py` regex patch.
- `python/tessera/runtime.py` — Python-lane executor (numpy/ctypes). Stays as the
  experimentation executor + oracle. Does **not** grow into the production path.
- `canonical_compile.py` — becomes the single front door whose `executable`
  answer is a real JIT'd function once Phase 0 lands.
- C++ `src/transforms/lib/*` — fusion/verifier passes are real and reused; the
  hand-written per-op `*ToAppleGPU.cpp` lowerings become **target overrides**
  (D2), not the foundation.
- The coverage registry (`primitive_coverage.py`) — repurposed as the **promotion
  ledger** (D5).
