---
title: Compiler Code Audit — 2026-06-10 (refactoring · per-IR optimization · glass jaws)
last_updated: 2026-06-10
scope: C++ pass quality (Graph/Schedule/Tile/Target IR) · Python compiler core · runtime dispatch
status: every HIGH finding verified against source per Decision #27; agent-only leads are marked
---

# Compiler Code Audit — 2026-06-10

Companion to [DEEP_COMPILER_AUDIT_2026_06_10.md](DEEP_COMPILER_AUDIT_2026_06_10.md)
(which covered manifest/benchmark/dashboard semantics). This pass covers
**code-level** findings: refactoring, per-IR-level optimization correctness,
and glass jaws. Every finding marked **verified** was confirmed by reading the
owning source in this session (Decision #27); findings marked *(lead)* came
from a sub-agent sweep with plausible file:line evidence but were not
independently re-read — spot-check before acting on them.

Two agent claims were **refuted** during verification and are recorded at the
bottom so they don't resurface.

## P0 — correctness

| # | Finding | Where | Status |
|---|---------|-------|--------|
| 1 | **verified — miscompile.** `TransposeIntoMatmul` combined a folded transpose with an existing `transposeA/B` flag via logical OR; transpose∘transpose must compose by XOR. A matmul already carrying `transposeA=true` fed by `tessera.transpose` double-transposed. | `src/transforms/lib/CanonicalizeTesseraIR.cpp` | **FIXED 2026-06-10** (commit `acb5c6f`: XOR composition; lit fixture `tests/tessera-ir/phase2/canonicalize_fusion_guards.mlir`) |
| 2 | **verified.** `FuseMatmulBiasGELU` / `FuseConvRelu` fused without `hasOneUse()` guards on the intermediate add/conv → duplicate work when the intermediate has another consumer. (The newer SwiGLU/MLA fusion passes already had the guards.) Greedy-driver non-convergence was also silently discarded. | same file | **FIXED 2026-06-10** (commit `acb5c6f`: guards + convergence warning) |
| 3 | **verified — semantic hazard.** NSA fusion (`FuseNSABranches`) replaces all three branch results (`sliding_window`/`compressed_blocks`/`top_k_blocks`) with the *same* fused value while the downstream gating multiply-add remains, so the program computes `(g_w+g_c+g_s)·fused` instead of the per-branch gated sum. No result-type-equality or single-use guards before the triple `replaceOp`. | `src/transforms/lib/NativeSparseAttnFusionPass.cpp:113-121` | **GUARDED 2026-06-10** (single-result + identical-type + single-use guards; the gating-subsumption contract is now explicit). Full fix = fuse the gating chain into the op. |
| 4 | **verified — silent wrong gradients.** `AutodiffPass` reverse walk skips any op without `AdjointInterface` (`if (!adjointOp) continue;`) even when its results carry cotangents — i.e. the op is on the gradient path and the chain silently breaks. The pass header documents the Decision-#21 diagnostic; the body never implemented it. | `src/transforms/lib/AutodiffPass.cpp:154-158` | **FIXED 2026-06-10** (`AUTODIFF_OP_NOT_DIFFERENTIABLE` error when a multi-operand op on the gradient path lacks the interface) |
| 5 | **verified — systemic glass jaw.** `runtime.py` has ~33 broad `except Exception` clauses and ~70 void-return C ABI bindings; several apple_gpu dispatchers fall back to numpy on *failure* (not just envelope misses). A regressed Metal kernel silently degrades to the numpy oracle and **every numerical test still passes**. | `python/tessera/runtime.py` (e.g. grouped_gemm per-group fallback :3963-3978) | **MITIGATED 2026-06-10**: `TESSERA_STRICT_DISPATCH=1` + bounded fallback funnel (`_note_dispatch_fallback`, `dispatch_fallback_log()` / `reset_dispatch_fallback_log()`); failure-class fallbacks (unary lane symbol-missing, bmm lane unavailable, grouped_gemm kernel exceptions) raise `TesseraStrictDispatchError` in strict mode; envelope-class fallbacks stay silent by design. Locked by `tests/unit/test_strict_dispatch.py` (16 tests). Full site coverage rides on the dispatch-table refactor (P2 below). |

## P1 — per-IR-level optimization correctness

| # | Finding | Where | Status |
|---|---------|-------|--------|
| 6 | **verified.** No named pipeline schedules upstream `canonicalizer`/`CSE`; ~102 ops carry `Pure` so the passes would genuinely act. Duplicate computation survives the whole stack (Python emission has no CSE/DCE either). The Graph-IR fusion block is copy-pasted 3× (`tessera-lower-to-x86`, `tessera-lower-to-gpu`, `buildCUDA13Pipeline`). | `src/transforms/lib/Passes.cpp` | **FIXED 2026-06-10** (shared `addGraphIRPreLoweringPasses()` + canonicalizer+CSE after fusion in all three pipelines; lit suite re-validated) |
| 7 | **verified.** `tessera-verify` checks only the module version attribute; pipeline-level completeness ("no `tessera.*` compute op survived lowering") has no home, and Apple/x86 lowering passes leave unmatched ops silently (pattern `notifyMatchFailure` only — partial lowering still "succeeds"). | `src/transforms/lib/VerifyTesseraIR.cpp`, `MatmulToAppleGPU.cpp:175-180` | **TOOL ADDED 2026-06-10** (`tessera-verify{forbid-ops=...}` emits `TESSERA_VFY_FORBIDDEN_OP`; lit fixture `verify_forbid_ops.mlir`); wiring into runtime pipelines is a follow-on because Python-side envelope gating legitimately leaves ops for the numpy spine in artifact lanes. |
| 8 | **verified.** Runtime re-matches fused chains per invoke (`_apple_gpu_metadata_is_*` at `runtime.py:2263-2370`) even though `CompileResult.fusion_groups` (known_chain entries) has been in artifact metadata since 2026-06-07 — the documented "fusion intent too late" item, now actually unblocked. | `python/tessera/runtime.py`, `canonical_compile.py:661` | **FIXED 2026-06-10** (executor consults whole-program `fusion_groups` known_chain entries and short-circuits the re-matchers; re-matching kept as fallback for legacy artifacts; locked by `tests/unit/test_strict_dispatch.py`). Follow-ons: SwiGLU is not derived by `_derive_fusion_groups` (DAG, not linear chain), and the C++ Target IR fusion passes still re-discover the same chains. |
| 9 | **verified.** Schedule IR is the thinnest level: generic matmul tile sizes are pass-constructor defaults (16×16), not Schedule-IR attributes the autotuner can sweep (the FA-4 `tile_q/tile_kv` contract is honored for attention only). No LICM/double-buffering/prefetch decision pass exists at Tile IR. Moot until non-Apple hardware lights up, but should be *chosen*, not forgotten. | `src/transforms/lib/TilingPass.cpp:127` | open (deliberately deferred to Phase G/H) |
| 10 | **verified.** `SwigluFusionPass`/`MLAFusionPass` build fused ops without propagating `numeric_policy` from the constituent matmuls/flash_attn (grep: zero hits in either pass). | `src/transforms/lib/{SwigluFusionPass,MLAFusionPass}.cpp` | open |

## P2 — refactoring / performance (verified unless marked)

- **Per-dtype triplication in `runtime.py`** — ~50 `_apple_gpu_dispatch_*`
  functions with near-identical f32/f16/bf16 branches and per-call ctypes
  `argtypes` rebinding (symbol getters re-bind on every fetch; the
  `_apple_gpu_dispatch.py` `_symbol_cache` is unused by these). One
  parameterized `(op, dtype) → (symbol, ctype)` table collapses hundreds of
  lines and is the prerequisite for full strict-dispatch coverage (finding #5).
- **`extractPtr` / `ensureExternalDecl` copy-pasted** across the Apple lowering
  passes and x86 backend *(lead — ~16 occurrences)*. Move to a shared
  `LoweringUtils.h`.
- **Graph IR cache deepcopies on every hit and store**
  (`graph_ir_cache.py:152,177`) — verified, documented design choice; a
  frozen/COW module representation would remove the dominant cost on
  high-hit-rate workloads.
- **MPSGraph cache keyed by exact shape** — verified by code comment; LRU
  (cap 1024, env-tunable) bounds the damage. Shape-class bucketing remains a
  perf follow-on, not a leak.
- **Hard-coded C ABI signatures in C++ lowering** (`MatmulToAppleCPU.cpp:131`,
  `MatmulToAppleGPU.cpp:141`) — verified; a signature registry keyed by
  (target, op, dtype) prevents arity drift when the shim evolves.
- **`jit.py` god functions** — verified: `jit()` decorator ~400 lines,
  `_build_runtime_artifact()` ~200; plus apple_gpu auto_batch wrapping inlined
  at decoration time. Extract per-stage helpers.
- **Per-shape constraint cache** (`jit.py:706`) — exact-shape keys are fine for
  training, thrashy for serving with varied batch dims.
- *(leads, not re-verified)*: `LayoutLegalityPass` walks only the immediate
  producer; `MigrateTesseraIR` accepts any version string;
  `EffectAnnotationPass` assumes callees pre-annotated; NVIDIA
  WGMMA/TMA/async-copy passes don't validate SM-level stage/dim ceilings
  (hardware-gated anyway); `hybridVariant()` returns unvalidated pattern
  strings.

## Refuted during verification

- **"MPSGraph cache first-init race"** — wrong: `mpsg_graph_cache()` /
  `mpsg_lru_order()` / `mpsg_cache_capacity()` all use `std::call_once`, and
  mutation is under `g_mpsg_graph_mu` (`apple_gpu_runtime.mm`).
- **"`_native_cpu_fast_call` rebuilds the artifact per call"** — wrong:
  `runtime_artifact()` returns the cached `_cached_artifact` after first build
  (`jit.py:962-965`). The fast path also already records
  `last_fallback_reason` and honors `native_required=True`.

## Method note

HIGH findings were confirmed by reading the owning source in-session; the
sub-agent sweep that produced the initial candidate list also produced the two
refuted claims above — which is why *(lead)* items must be source-verified
before they migrate into status claims (Decision #27).
