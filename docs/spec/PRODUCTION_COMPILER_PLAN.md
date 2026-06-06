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
>   affine map dropping the reduced axis — `emitBroadcastBinary`) and **first use
>   of the `math` dialect** (`math.exp`, via `convert-math-to-llvm`). Elementwise
>   family completed with `tessera.div`.
> * **Phase 1 Sprint 1.4 landed 2026-06-05** — `tessera.rmsnorm` /
>   `tessera.layer_norm` (unweighted, innermost axis). Pure composition over
>   Sprint 1.3 (`emitMean` + `emitBroadcastBinary` + precise `math.sqrt`).
> * **Phase 1 Sprint 1.5 landed 2026-06-05** — **bf16 boundary (ABI §12.5).**
>   `ml_dtypes.bfloat16` Python side / raw-i16 at the memref boundary; matmul
>   accumulates in f32 then `truncf` to bf16 storage. Descriptor packing went
>   dtype-generic (`_resolve_elem`); mixed-dtype rejected, not promoted. A test
>   proves the f32-accumulate policy actually engaged (beats naive bf16-accumulate
>   on K=512).
> * **Phase 1 Sprint 1.6 landed 2026-06-05** — activations
>   `relu/sigmoid/tanh/silu/gelu` (unary `math` family; gelu = tanh approximation
>   to avoid the unlowerable `math.erf`).
> * **Phase 1 Sprint 1.7 landed 2026-06-05** — `tessera.transpose` (rank-2, via
>   `linalg.transpose`) and `tessera.matmul` `transposeA/transposeB` (operand
>   transposed before a plain matmul — the `Q @ Kᵀ` shape). **A full single-head
>   attention block — `softmax(Q Kᵀ / √d) V` — now composes from production-lane
>   primitives and matches the numpy oracle.** **109/109 production-lane tests
>   green** across `tests/unit/test_production_jit_*.py`.
>
> * **Phase 1 Sprint 1.8 landed 2026-06-05** — **multi-op graph compilation.**
>   `GraphFn` (Python) builds a whole multi-op `tessera` function compiled as ONE
>   JIT'd unit — intermediates never cross the boundary, the lowering can fuse.
>   The invocation counter proves it (N-op graph ⇒ +1). **A LLaMA-style
>   single-head transformer decoder layer (rmsnorm + attention + SwiGLU MLP +
>   residuals) compiles as one function and matches numpy** — the "model layer
>   end-to-end" milestone. C ABI invoke went arity-unlimited via libffi.
> * **Phase 1 Sprint 1.9 landed 2026-06-05** — `tessera.batched_gemm` (rank-3,
>   `C[i]=A[i]@B[i]`) → `linalg.batch_matmul`, f32-accumulate for bf16. Unblocks
>   the batch/head dimension; batched per-head attention composes in one graph.
> * **Phase 1 Sprint 1.10 landed 2026-06-05** — **compilation cache** (S14
>   direction). Compiled handles cached on MLIR text; repeated same-(op,shape)
>   calls skip parse→lower→JIT. Transparent (`compile_module` cache-backed,
>   `destroy` no-op for cached, freed at exit). A C++ compile-counter proves cache
>   hits don't recompile while each invoke still runs. **129/129 production-lane
>   tests green.**
>
> **Phase 1 op coverage:** `add/sub/mul/div, matmul (±transpose, ±bf16/f32-acc),
> batched_gemm, reduce(sum/max/min/mean), softmax, rmsnorm, layer_norm,
> relu/sigmoid/tanh/silu/gelu, transpose`, plus **multi-op graph compilation** and
> a **compilation cache** — all oracle-tested through real codegen; f32 + bf16.
> Capstone proof: a transformer decoder layer compiles+runs as one function.
> Remaining Phase-1 polish (deferred, optional): dynamic shapes, reconcile
> `tessera_jit` ↔ `tsrCompileArtifact` (§12.7).
>
> ### Phase 2 — control flow (state is next)
> * **Sprint 2.1 landed 2026-06-05** — data-parallel conditional: `tessera.select`
>   (`cond!=0 ? a : b`) and `tessera.masked_fill` (`mask!=0 ? x : value`). The
>   masked_fill path is the **causal-attention masking primitive**; a causal
>   attention block (`softmax(masked_fill(Q Kᵀ, mask, -1e9)) V`) composes in one
>   compiled function.
> * **Sprint 2.2 landed 2026-06-05** — **`scf.for` control flow.** A bounded loop
>   with a tensor carry, compiled as one function through
>   tessera→linalg→scf→cf→llvm. `GraphFn.for_loop(count, init, body)`; proven with
>   power iteration and **N iterated transformer FFN blocks (shared weights)**.
>   Foundation work: registered scf bufferization, `allowReturnAllocsFromLoops`,
>   and switched the DPS rewrite to **`memref.copy`** (the redirect-and-erase trick
>   silently killed control flow — `memref.copy` is correct for any producer and
>   lowers to `memcpy` for the identity-layout boundary).
> * **Sprint 2.3 landed 2026-06-05** — **`scf.if` conditional control flow.** A
>   shape-(1,) runtime flag drives an `scf.if` (only the taken branch executes,
>   vs select). `GraphFn.cond(flag, then, else)`; **nests with `for_loop`**.
> * **Sprint 2.4 landed 2026-06-05 — state; Phase 2 COMPLETE.**
>   `tessera.write_row` (functional KV-cache update → `tensor.insert_slice`) +
>   **multi-result functions** (DPS out-param per result, so a decode step returns
>   `out` *and* the updated caches as ONE compiled function). **Capstone (the
>   Phase-2 DoD): a stateful incremental-decode loop** threads the KV cache across
>   T steps through the production lane and matches a full causal-attention numpy
>   oracle (and the accumulated cache equals K/V). ABI hardening: function inputs
>   are marked `bufferization.writable = false` so bufferization can never write
>   in-place into a caller's input buffer (write_row stays value-semantic).
>   **154/154 production-lane tests green.**
>
> **Phase 2 COMPLETE** — control flow (select/masked_fill, scf.for, scf.if) +
> state (write_row, multi-result, stateful decode). Control flow lives at the
> builder + bufferization level (no bespoke region-carrying dialect ops).
>
> ### Phase 3 — Apple GPU end-to-end (production on real silicon)
> There is **no upstream MLIR Metal/AIR backend**, so Apple GPU does NOT use the
> CPU lane's `linalg→LLVM→ORC` path — it's a **bespoke Metal back-half** (D2). The
> shared part is the `tessera` graph structure + the CPU lane as oracle; execution
> routes to hand-tuned MPS/MSL kernels.
> * **Sprint 3.1 landed 2026-06-05** — Apple GPU back-half + **cross-target
>   oracle**. `python/tessera/_apple_gpu_backend.py` reuses the existing
>   `tessera_apple_gpu_*` *kernel* C ABI (not `runtime.py`'s dispatch) via the
>   on-the-fly runtime loader. matmul / softmax / **fused matmul→softmax** / gelu
>   run on the real Apple GPU and match the **compiled CPU production lane**
>   (which matches numpy). The fused `matmul→softmax` (one Metal kernel) equals the
>   un-fused CPU composition — the **D2 fused-chain target override**, proven.
>   **12/12 GPU tests green** (`tests/unit/test_production_jit_phase3_apple_gpu.py`).
>
> * **Sprint 3.2 landed 2026-06-06** — **Apple GPU kernel coverage toward the
>   full transformer block.** Wired into the production back-half
>   (`_apple_gpu_backend.py`), each cross-target oracle-matched vs the compiled
>   CPU lane: `gpu_rmsnorm` / `gpu_layer_norm` (unweighted — GPU kernel called
>   with γ=1, β=0 to match the CPU lane's unweighted norms), `gpu_silu`
>   (MPSGraph unary opcode 4), the **fused single-head attention block**
>   `gpu_attention` (`softmax(A@B)@C` in ONE Metal kernel — the D2 fused-chain
>   override, == CPU's un-fused matmul→softmax→matmul), and the fused MLP chains
>   `gpu_matmul_gelu` / `gpu_matmul_rmsnorm`. Capstone: a **pre-norm
>   self-attention sub-block** (rmsnorm → QKV proj → softmax(QKᵀ/√d)V → residual)
>   composes entirely from production GPU kernels and stays oracle-clean against
>   the same composition on the CPU lane. **189/189 production-lane tests green**
>   (+23; `tests/unit/test_production_jit_phase3_kernels.py`).
>
> Phase 3 remaining toward the DoD (a full transformer block production-grade on
> Apple GPU, oracle-matched): a `GraphFn`-level `target="apple_gpu"` dispatch
> (Sprint 3.3 — route a whole multi-op graph to the GPU back-half as one unit,
> instead of today's per-op `agb.gpu_*` calls), then bf16 (Sprint 3.4). The
> SwiGLU-chain and attention fused kernels exist in the runtime
> (`swiglu_f32`, `matmul_softmax_matmul_f32`) and `gpu_attention` now wires the
> latter; the SwiGLU fused kernel wiring is deferred to the GraphFn-dispatch
> sprint where the whole MLP routes together.
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
