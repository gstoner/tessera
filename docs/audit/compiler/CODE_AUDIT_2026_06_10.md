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
| 5 | **verified — systemic glass jaw.** `runtime.py` has ~33 broad `except Exception` clauses and ~70 void-return C ABI bindings; several apple_gpu dispatchers fall back to numpy on *failure* (not just envelope misses). A regressed Metal kernel silently degrades to the numpy oracle and **every numerical test still passes**. | `python/tessera/runtime.py` (e.g. grouped_gemm per-group fallback :3963-3978) | **MITIGATED 2026-06-10**: `TESSERA_STRICT_DISPATCH=1` + bounded fallback funnel (`_note_dispatch_fallback`, `dispatch_fallback_log()` / `reset_dispatch_fallback_log()`); failure-class fallbacks (matmul/unary symbol-missing, bmm lane unavailable, grouped_gemm kernel exceptions) raise `TesseraStrictDispatchError` in strict mode; envelope-class fallbacks stay silent by design. Locked by `tests/unit/test_strict_dispatch.py`. **CI lanes wired 2026-06-10**: `tests/unit/conftest.py` `_STRICT_DISPATCH_LANES` forces strict mode (Darwin-only) on the two differential-generator suites + the 10 manifest-declared Metal `execute_compare_fixture` modules — verified green on a live Metal host (zero failure-class fallbacks). |

## P1 — per-IR-level optimization correctness

| # | Finding | Where | Status |
|---|---------|-------|--------|
| 6 | **verified.** No named pipeline schedules upstream `canonicalizer`/`CSE`; ~102 ops carry `Pure` so the passes would genuinely act. Duplicate computation survives the whole stack (Python emission has no CSE/DCE either). The Graph-IR fusion block is copy-pasted 3× (`tessera-lower-to-x86`, `tessera-lower-to-gpu`, `buildCUDA13Pipeline`). | `src/transforms/lib/Passes.cpp` | **FIXED 2026-06-10** (shared `addGraphIRPreLoweringPasses()` + canonicalizer+CSE after fusion in all three pipelines; lit suite re-validated) |
| 7 | **verified.** `tessera-verify` checks only the module version attribute; pipeline-level completeness ("no `tessera.*` compute op survived lowering") has no home, and Apple/x86 lowering passes leave unmatched ops silently (pattern `notifyMatchFailure` only — partial lowering still "succeeds"). | `src/transforms/lib/VerifyTesseraIR.cpp`, `MatmulToAppleGPU.cpp:175-180` | **TOOL ADDED 2026-06-10** (`tessera-verify{forbid-ops=...}` emits `TESSERA_VFY_FORBIDDEN_OP`; lit fixture `verify_forbid_ops.mlir`); wiring into runtime pipelines is a follow-on because Python-side envelope gating legitimately leaves ops for the numpy spine in artifact lanes. |
| 8 | **verified.** Runtime re-matches fused chains per invoke (`_apple_gpu_metadata_is_*` at `runtime.py:2263-2370`) even though `CompileResult.fusion_groups` (known_chain entries) has been in artifact metadata since 2026-06-07 — the documented "fusion intent too late" item, now actually unblocked. | `python/tessera/runtime.py`, `canonical_compile.py:661` | **FIXED 2026-06-10** (executor consults whole-program `fusion_groups` known_chain entries and short-circuits the re-matchers; re-matching kept as fallback for legacy artifacts; locked by `tests/unit/test_strict_dispatch.py`). **SwiGLU derivation closed same day**: `_match_swiglu_at` in `canonical_compile.py` matches the DAG (gate/up share %x) inside the known-chain scan, and the executor consumes `fused_kernel == "swiglu"` (locked by `test_fusion_groups_swiglu` + `test_executor_dispatches_swiglu_from_fusion_groups`). Remaining follow-on: the C++ Target IR fusion passes still re-discover the same chains. |
| 9 | **verified.** Schedule IR is the thinnest level: generic matmul tile sizes are pass-constructor defaults (16×16), not Schedule-IR attributes the autotuner can sweep (the FA-4 `tile_q/tile_kv` contract is honored for attention only). No LICM/double-buffering/prefetch decision pass exists at Tile IR. Moot until non-Apple hardware lights up, but should be *chosen*, not forgotten. | `src/transforms/lib/TilingPass.cpp:127` | open (deliberately deferred to Phase G/H) |
| 10 | **verified.** `SwigluFusionPass`/`MLAFusionPass` build fused ops without propagating `numeric_policy` from the constituent matmuls/flash_attn (grep: zero hits in either pass). | `src/transforms/lib/{SwigluFusionPass,MLAFusionPass}.cpp` | **FIXED 2026-06-10** — `Tessera_SwigluFusedOp`/`Tessera_MLADecodeFusedOp` ODS gained `OptionalAttr<Tessera_NumericPolicyAttr>`; SwiGLU propagates when the three matmuls agree and **declines to fuse on conflict** (per-stage policies are inexpressible on one fused op); MLA carries the flash_attn's policy (attention dominates the fused numerics). Lit: `swiglu_fusion.mlir` + `mla_decode_fusion.mlir` propagation + conflict cases. |

## P2 — refactoring / performance (verified unless marked)

- **Per-dtype triplication in `runtime.py`** — ~50 `_apple_gpu_dispatch_*`
  functions with near-identical f32/f16/bf16 branches and per-call ctypes
  `argtypes` rebinding (symbol getters re-bind on every fetch; the
  `_apple_gpu_dispatch.py` `_symbol_cache` is unused by these). One
  parameterized `(op, dtype) → (symbol, ctype)` table collapses hundreds of
  lines and is the prerequisite for full strict-dispatch coverage (finding #5).
  **PARTIALLY CLOSED 2026-06-10**: `_apple_gpu_dispatch_matmul` and
  `_apple_gpu_dispatch_unary` are now dtype-lane tables sharing
  `_apple_gpu_gemm2d_call` / `_apple_gpu_mpsgraph_unary_call` (per-dtype
  routing decisions — MTL4 routers, bf16 capability gate — stay visible in
  the tables); missing-symbol paths route through the strict-dispatch funnel.
  **Consciously skipped**: memoizing the symbol getters — the runtime dylib is
  *lazily built* by per-test build helpers, so caching a pre-build `None`
  would wedge them; binding-once needs a "runtime generation" token first.
  Remaining: the long tail of per-dtype dispatchers (rowop, softmax, gelu,
  rope, flash_attn, fused chains) adopt the same lane-table shape.
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

## Closeout status (2026-06-10, final pass)

A dedicated closeout pass drove every remaining item to one of: **done**,
**refuted**, **accepted tradeoff**, or **tracked-deferred (with rationale)**.
No item is left as a silent open gap.

### Closed this pass

- **1e — zero-`TRACE_DEFERRED` corpus guard** — DONE.
  `tests/unit/test_audit_closeout_guards.py::test_known_good_apple_gpu_corpus_emits_no_trace_deferred`
  decorates a 7-function emittable corpus and asserts none defer to the tracer
  (closes the "AST-emission regression passes via tracer" hole). Platform-
  agnostic (decoration-time).
- **§3 `_APPLE_GPU_*` table-creep enforcer** — DONE.
  `test_envelope_tables_are_single_source` asserts `driver.py`/`runtime.py`
  never redefine an envelope-owned op/opcode/lane table (runtime-local handler
  maps + symbol-availability memo caches are correctly excluded).
- **1d — failure-class fallback coverage** — EXTENDED. The binary and rowop
  MPSGraph dispatchers now route their symbol-missing fallbacks through
  `_note_dispatch_fallback` (strict mode raises; non-strict logs); empty-array
  and unsupported-dtype paths are correctly classified as envelope misses, not
  failures. New strict-mode tests in `test_strict_dispatch.py`.
- **§4 `extractPtr`/`ensureExternalDecl` dedup** — DONE.
  `include/Tessera/Target/Apple/LoweringUtils.h` holds both as `inline`
  `tessera::apple` functions; **18** Apple lowering passes migrated (byte-
  identical bodies, brace-counted removal — zero divergent). `TesseraApple` +
  `tessera-opt` rebuild clean; phase8 lit + pass-status enforcer green.
- **§5 bf16 probe** — DONE (was already cached): `_apple_gpu_supports_native_bf16`
  memoizes via `_NATIVE_BF16_SUPPORTED`; only the first call probes.

### Refuted (no action — would be wrong)

- **§5 MPSGraph "shape-class" cache keys** — a compiled MPSGraph is
  shape-specialized; bucketing distinct shapes onto one cached graph is a
  miscompile. The LRU (cap 1024, env-tunable) is the correct mitigation and is
  already in place. (Refutes the perf suggestion as literally stated.)

### Accepted tradeoffs (correctness-first, costs bounded)

- **1f graph_ir_cache deepcopy** — intentional immutability boundary; the cost
  is one CPU-bound deepcopy outside the lock. A frozen/COW `GraphIRModule` is a
  larger change with mutation-safety risk for no correctness gain.
- **§5 per-call constraint re-check** — exact-shape memo set; a `Divisible(M,128)`
  predicate genuinely depends on the value, so per-distinct-shape checks are
  unavoidable. The residual cost is a set lookup.

### Closed in the follow-on focused passes (2026-06-10, later same day)

- **`jit.py` god-function extraction — DONE.** The `_decorate` closure dropped
  from ~325 to ~145 lines: Steps 1-4 (constraint solve + effect inference)
  extracted to `_jit_analyze_frontend`, and Step 6 (the ~140-line Graph-IR
  emission try/except, incl. the auto_batch-skip and apple_gpu trace-defer
  paths) to `_jit_emit_graph_ir`, returning `_FrontendAnalysis` /
  `_GraphIREmission` dataclasses. Faithful relocation — same control flow,
  diagnostics, and order; gated by the full `@jit` test surface (85 jit-focused
  tests + full sweep). The closure now reads as an orchestration of named
  stages. (A dead no-op `except TesseraEffectError: raise` was dropped along
  the way; the exception still propagates naturally.)
- **1d #3 — GPU dispatch error channel — DONE (errno-style, better than the
  literal "int return on 70 symbols").** Changing ~70 `void` symbol signatures
  was rejected as a massive ABI break; instead the `.mm` runtime now sets a
  **thread-local last-error at the shared command-buffer choke point**
  (`commit_and_wait_with_timeout`, ~72 callers) on its two failure branches
  (timeout/hang = kind 1, `cb.error` = kind 2), generalizing the existing
  `g_mlpkg_last_error_kind` pattern. New C ABI:
  `tessera_apple_gpu_last_error_kind` / `_message` / `_clear_last_error`
  (+ non-Darwin stub parity). The Python matmul lane arms (clears) the channel
  before each GEMM and consumes (reads+clears) it after — a silent internal
  GPU failure (which left the output buffer untouched) now funnels through
  `_note_dispatch_fallback` (strict raises) + recomputes on host instead of
  returning garbage. Validated on a Metal host: real matmul succeeds with no
  false-positive; new symbols present in both the test-helper dylib and the
  canonical `libTesseraAppleRuntime.a`. Tests in `test_strict_dispatch.py`
  (simulated GPU error funnels + raises; no-channel build is a safe no-op).
  *Remaining follow-on:* other lanes (unary/binary/rowop/bmm) adopting the
  same 3-line arm/consume pattern — mechanical, doesn't change the
  architectural close.

### Tracked-deferred (explicit, with rationale)

- **1c / partial-lowering completeness wiring** — the `tessera-verify
  forbid-ops` tool exists; auto-wiring it into the runtime pipelines needs
  per-pipeline "guaranteed-consumed op" analysis (not every `tessera.*` op is
  lowered by every pipeline; e.g. `TilingPass` only tiles static f32/bf16
  rank-2 matmul, so even `tessera.matmul` legitimately survives). A blanket
  forbid would false-positive. Deferred pending that per-pipeline analysis.
- **Target IR C++ fusion descriptor consumption** — the runtime half is closed
  (consumes `fusion_groups`, incl. SwiGLU). Having the C++ backend passes read
  a descriptor instead of re-recognizing is a larger architectural item
  (Decision #19 "emit backend descriptors").
- **Schedule IR autotuner-visible tile sizes / Tile IR LICM + double-buffering**
  — hardware-gated (no NVIDIA here); speculative without a validation target.
