---
last_updated: 2026-10-07
audit_role: plan
plan_state: open
---

# Integrated Compiler Plan

Start at [`README.md`](README.md). This document alone owns cross-domain compiler
sequencing. [MASTER_AUDIT](../MASTER_AUDIT.md#proof-vocabulary) and generated
inventories supply evidence; scoped plans own detailed acceptance contracts;
backend queues own target-specific proof. Reconcile contradictory evidence with
source and tests rather than inventing a second status registry here.

`last_updated` means last substantive review of current-priority content, not a
formatting change. The [engineering log](INTEGRATED_COMPILER_LOG.md) preserves
revision-bound outcomes. The [archived queues and waves](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md)
are provenance, including superseded priority statements and estimates.

## Foundation program

One MLIR-owned semantic program must descend through verified native compiler
passes. Python remains the public API, capture/orchestration layer and oracle.
After canonical IR exists, code generators must not reconstruct semantics from
GraphIRModule, operation names or Python descriptor/fusion objects. Shapes,
layouts, effects, derivative products, numeric policy and schedule choices must
survive in verified IR. Deployment facts and measurements remain identity-bound
inputs; runtime loads images and executes their ABI.

For new enhancements, require the final-state route: typed Graph IR, verified
MLIR differentiation/optimization, Schedule and Tile IR, backend Target IR,
native lowering, then an image with a checked runtime ABI. A Python backend,
Graph-derived package constructor or source-emitter fast path is not an
acceptable new semantic implementation. Existing such routes are migration
inventory under E2E-REAL-6; keep them bounded until the compiled replacement
has differential and owning-target evidence. A library call is legitimate when
the call and ABI are explicit in IR.

| Cut | Architectural purpose | Sequencing rule |
|---|---|---|
| F0 | Route census, proof integrity and usable compiler tooling | Cross-cutting; no hardware prerequisite for inventory or contract fixes. |
| F1 | First canonical-artifact package migration | Historical NVIDIA scheduled-matmul slice; remaining families belong to F2. |
| F2 | Migrate surviving Graph-owned packaging and callers | Select a family from the census; retain its ABI, dtype and policy envelope. |
| F3 | Native recipe instantiation, fusion, ANN and measured scheduling | Requires the selected family's IR boundary, not completion of every F2 route. |
| F4 | Structured AD/effects/memory, numerical and distributed semantics | Extend the same compiler foundation through verified carriers. |
| F5 | Retire redundant generators and broaden mathematical workloads | Delete only after differential and owning-target proof covers surviving callers. |

The cuts are an architecture map, not a waterfall. Priority is local to each
cut below. Dependencies identify the specific prerequisite portion; they do not
require the entire referenced program to close. Tuned/library candidates remain
legitimate when their selection and ABI are explicit in IR.

Native destinations: x86 uses MLIR/linalg/vector/SCF to LLVM/object or ORC;
NVIDIA uses Tile/Target to NVVM/PTX/image; ROCm uses GPU/ROCDL/LLVM to HSACO;
Apple GPU uses compiler-owned MSL and the supported Metal compiler/metallib path.
Do not imply a general LLVM-IR-to-Metal backend or transfer schedules/proof
between targets. The [foundation survey](MLIR_NATIVE_FOUNDATION_SURVEY.md) owns
migration seams; the [backend map](../backend/README.md) owns device queues.

## Start here

This is a derived navigation view, not another queue. The first `host-free`
action in each cut is the entry point below; each linked record names its gate
and prerequisite portion. Host-free means its next engineering step can start
without device access, not that its eventual promotion needs no device proof.
`device` actions require an owning-host availability check at execution time.
Dependencies and required evidence must be checked before starting dependent work.

<!-- ready-view:start -->
- F0: [E2E-REAL-6F](#e2e-real-6f).
- F2: [E2E-REAL-6](#e2e-real-6).
- F3: [FRONTEND-IR-MEDIUM-1](#frontend-ir-medium-1).
- F4: [W4-PRODUCT-1](#w4-product-1).
- F5: [W5.2f](#w52f).
<!-- ready-view:end -->

## Live queue

Records below are remaining obligations, not support/promotion statuses.
The four closed or refuted ROCm measurement records are preserved in the
[archive](archive/ROCM_MEASURED_QUEUE_2026-09-28.md); their routing rows below
remain for ID lookup.
`Gate` is the next missing deliverable and acceptance boundary. `Latest` points
to historical context, which must be read at its recorded scope. `Start` names
the next action's host requirement; it is not a live fleet-availability claim.

## F0

### E2E-REAL-6F

**Route census and exact-device certificates**

- Owner: [MLIR_NATIVE_FOUNDATION_SURVEY.md](MLIR_NATIVE_FOUNDATION_SURVEY.md)
- Gate: The expanded census includes x86 breadth packaging: 46 Graph inputs, 14 scheduled inputs and 14 raw/unclassified entries. Import-resolved caller candidates and local helper-to-emitter paths expose reconstruction paths; scope-aware import candidates now reject parameter/rebinding shadowing and sibling-scope leakage; indirect dispatch and per-envelope certificate joins still require review. Counts do not authorize constructor deletion.
- Depends on: —
- Start: host-free
- Latest: [2026-09-10 — Route census and device profile attribution](INTEGRATED_COMPILER_LOG.md#2026-09-10--route-census-and-device-profile-attribution)

### COMPILER-DEVEX-1

**Assertions-enabled validation and usable tools**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: Tajasarus now has an assertions-enabled LLVM/MLIR 23.1.1 ROCm build; RX 9070 XT gfx1201 correctness commissioning is tracked under ROCM-2 in the backend queue. The unified scheduled closure accounts for the 90 rows skipped by ordinary non-ROCm/compiler-free CI and passes 95/95 with zero skips on Tajasarus; its recorder refuses a stale compiler, inventory drift, or any non-owning target. Assertions-enabled LLVM/MLIR 23.1.1 and a hardware-free all-target Tessera compiler now pass all 475 active lit fixtures on Super-Bear. Backend-owned fixtures declare their feature requirements, the data-only x86 execution input is outside lit discovery, and the union gate requires every active fixture to pass in at least one lane. The opt-in CI lit lane requests the same full portable target matrix. The 2026-09-29 `CI-LLVM-EXACT-2026-09-29` follow-up replaces hosted patch tolerance with the SHA256-checked official LLVM/MLIR/lld 23.1.1 bundle; CI now rejects a patch mismatch and matches the release's no-RTTI ABI. From 2026-09-27 until this correction (sync `FOUNDATION-BATCH-2-2026-09-27`), the hosted MLIR lanes (`lit`, `rocm-serialize`, `sanitizer`) accepted any recorded 23.1.x and failed rather than skipped when none was present; the fleet pin stayed exact. Installed drivers now pass relocated-prefix smoke on Super-Bear with loader overrides removed; the CI lane runs this check after installation. Preserve these regression gates; owning-device correctness and performance remain separate backend gates. A full unit sweep on Princess-Luna (2026-09-17) found 31 failures present on main and invisible to CI, whose unit lane has neither device toolchain: 28 are now fixed — a legality gate that asked the capability registry about the generic `rocm` name for a request the rest of the stack compiles for gfx1151, a link requirement the runtime archive never published to out-of-CMake consumers, one Python-driven data file failing `check-tessera-rocm` for the whole repository, `arith.select` with no transpose, and index arithmetic refused as non-differentiable. Owed: `AUTODIFF-SHAPE-WHILE-FORWARD-2026-09-17`, a pre-existing crash inside JIT-compiled code that the AD gate had been masking, and the fact that the x86 JIT AD lane those tests exercise has no host in any automated check.
- Depends on: —
- Start: host-free
- Latest: [2026-09-29 — GitHub LLVM/MLIR patch pin corrected](INTEGRATED_COMPILER_LOG.md#2026-09-29--github-llvmmlir-patch-pin-corrected)

### DIAG-PY-BACKLOG-1

**Python-raised diagnostics that were never registered**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: **CLOSED 2026-09-20; owner decision recorded 2026-09-26.** The ratchet (`_UNREGISTERED_ON_2026_09_19`) is empty: all five Apple codes are registered, the accumulator hint settled from the macOS 27 SDK (`MPPTensorOpsMatMul2d.h`). The owner has since decided that Apple accumulation is **not fp32-only by policy**, so `APPLE_FRAGMENT_UNSUPPORTED_ACCUMULATOR` now reads as the simdgroup lane's implementation limit and routes reduced-precision accumulation to `matmul2d`. The history below is kept as the record. `tests/unit/test_diagnostic_code_registry.py` scanned Python source with a **prefix allowlist** (`E_*`, `JIT_*`, `TS_ERR_*`, `GRAPH_IR_*`), so every domain-prefixed code a Python module raises was invisible to it — the failure the scanner's own `GRAPH_IR_` comment records happening once already for a whole family, while the gate reported green. The scan now matches the code *shape*, as the C++ scan always has, which surfaced 15 unregistered tokens on 2026-09-19. **Ten closed the same day.** The nine `ROCM_FRAGMENT_*` codes from `rocm_fragment.select_fragment_layout` are registered (one file, one exception type, one fail-closed story: no generic gfx-prefix fallback, which is the miscompile ROCM-5 removed). The fifteenth, `TIMING_PROOF_INCOMPLETE`, was a **misclassification** — it is one of eleven x86 promotion-ineligibility reason tags, visible to the scan only because it concatenates a colon, and registering it would have put Decision #29's unconsumed declaration inside the registry; it moved to `_NOT_DIAGNOSTICS` and its vocabulary opened [X86-EVIDENCE-VOCAB-1](#x86-evidence-vocab-1). The gate as it stood 2026-09-19 (history, now met): Apple's five (`APPLE_FRAGMENT_*` ×4, `APPLE_COUNTER_EVIDENCE_UNSUPPORTED`) gain entries and the ratchet empties. They are held rather than guessed because `APPLE_FRAGMENT_UNSUPPORTED_ACCUMULATOR`'s fix hint must state whether fp32-only accumulation is permanent or pending the Metal 4 cooperative-tensor lane, and an invented answer is worse than a missing entry. Adjacent, closed here: the registry's "each prefix maps to exactly one language" rule forbade a domain prefix spanning both emitters, which `ROCM_FRAGMENT_*` does by design (C++ `LowerTileToROCMPass` and Python fragment legality); it is narrowed to the five locked sentinels, with the general case covered — and falsified as covered — by the two checks that name the offending code directly.
- Depends on: —
- Start: host-free
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### X86-EVIDENCE-VOCAB-1

**The x86 promotion-ineligibility vocabulary has no registry and no gate**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: **CLOSED 2026-09-27 (sync `EVIDENCE-GOVERNANCE-GATES-2026-09-27`).** The eleven tags were declared with a meaning each on 2026-09-20 (`X86_INELIGIBILITY_REASONS`, still eleven: `test_x86_vocabulary_is_still_eleven` re-counts them from the producer) and the validator refuses an unknown tag. The closing slice routes producer and consumers through one `evidence_reasons.ReasonVocabulary` (the route split's `_ENVIRONMENT_TAGS` is checked against it; a declared tag nothing produces must be `reserved` with a reason), and applies the same rule to the four siblings that had the same shape: the ROCm profiler packet, the NVIDIA device-clock packet, the x86 PMU event map (its validator only type-checked its reasons and now re-derives them) and the calibration corpus (two legacy tags reserved; see [EVIDENCE-PACKET-1](#evidence-packet-1)). Tags that are registered diagnostics (the shared device-clock witness/window refusals, `DEVICE_CLOCK_PART_UNVALIDATED`) are borrowed by name, never redeclared, and none of the vocabulary tags was registered in `diagnostic_codes.py`. `profiler_cuda_window` carries prose reasons, not tags -- a different pattern, listed rather than changed. Drift gate: `tests/unit/test_x86_evidence_vocabulary.py` (producer vs declaration per family, read by AST over every append/extend/`+=`/literal style; fail-closed on committed packets doctored with an undeclared tag; every committed packet still validates except two that already failed the window rule, pinned to that reason). The gate as it stood (history): `profiler_x86_evidence.py` appends **eleven** reason tags to a `reasons` list that decides whether an x86 performance measurement is promotable — `CPU_NOT_EXACT_ZEN5`, `VIRTUALIZED_HOST`, `WSL_CLOCK_DOMAIN`, `SOURCE_WORKTREE_DIRTY`, `TIMING_PROOF_INCOMPLETE`, `SYMBOL_SAMPLING_MISSING`, `SYMBOL_SAMPLING_INVALID`, `IMAGE_BUILD_ID_MISSING`, `EVENT_MAP_MISSING`, `EVENT_MAP_NOT_PROMOTABLE`, `SAMPLING_AFFINITY_NOT_PINNED`. Nothing enumerates them, nothing drift-gates them, and no consumer can know the vocabulary is eleven items or that a twelfth was added — so a reader handling a subset silently treats an unknown reason as no reason, which under Decision #21a is a semantic key failing open. Found 2026-09-19 while classifying the diagnostic backlog: the shape scan saw exactly one of the eleven, by the accident of a concatenated colon. Gate: the vocabulary is declared in one place with a meaning per tag, its producer and every consumer read that declaration, and a drift test fails when a tag is added without it. These are **not** diagnostic codes and must not be answered by registering them in `diagnostic_codes.py`.
- Depends on: —
- Start: host-free
- Latest: [2026-09-27 — Evidence governance gates: reason vocabularies, ODS consumers, corpus eligibility](INTEGRATED_COMPILER_LOG.md#2026-09-27--evidence-governance-gates-reason-vocabularies-ods-consumers-corpus-eligibility)

### ROCM-NVFP4-INGEST-1

**NVFP4 reaches the fp8 WMMA on gfx1201; the one lossy step must be declared**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: An NVFP4 checkpoint (e2m1 elements, **e4m3 scale per 16**, fp32 scale per tensor) reaches `v_wmma_f32_16x16x16_fp8_fp8` through MXFP4 (e2m1, **e8m0 scale per 32**), and the chain has exactly one lossy step: the scale requantization. Measured relRMS against the bf16 original — NVFP4 as shipped 0.113 (19 dB), bf16→MXFP4 **direct** 0.112, NVFP4→MXFP4 requantized 0.158 (16 dB) — so **the ~3 dB is double rounding, not the format**, and where a bf16 original exists, quantizing once to MXFP4 beats ingesting NVFP4. That is a `numeric_policy` preference (Decision #15a) and the requantization is a declared information-loss point (Decision #32) with a measured cost, not an implicit load-time transform. Notably this is **not** a workaround for a weaker part: AMD's MI355 path dequantizes NVFP4 to BF16 because CDNA4 has no native NVFP4 execution either, so the gfx1201 route lands on the fp8 ceiling (383 TFLOP/s) rather than bf16's 191. Gate: `nvfp4` and `mxfp4` distinguishable below Graph IR (they are already distinct *names* — `dtype.py` says "do not alias" — but nothing below Graph IR can tell them apart until block-scale metadata exists), the requantization expressed as a policy-gated conversion carrying its SQNR, and two traps covered: block-exponent selection by squared error between the no-clip rule and one binade finer (not truncation), and merged linears (`gate_up_proj`) honouring **both** global scales rather than collapsing them, which fails silently. Depends on the scale contract from [ROCM-FP8-BLOCKSCALE-1](#rocm-fp8-blockscale-1) and the fold from [ROCM-MXFP4-W4A8-1](#rocm-mxfp4-w4a8-1).
- Depends on: [ROCM-MXFP4-W4A8-1](#rocm-mxfp4-w4a8-1)
- Start: device
- Current increment: Tajasaurus gfx1201 passed the 17x19x64 and ragged
  200x2048x1536 synthetic gate/up packages after vectorized ingest; CPU ingest
  fell from 6.70 s to 487.31 ms on the larger case, with unchanged codes,
  exponents, Target/Tile digests, and HSACO. The latest 12 host-side ingest
  properties pass on Tajasaurus WSL2. A real Qwen3-8B q_proj now also traverses
  NVFP4-to-MXFP4 ingest and the native Graph/Schedule/Tile/Target package.
  Relative RMS against pinned BF16 is 9.50% for shipped NVFP4, 11.22% for
  direct BF16-to-MXFP4, and 14.99% after ingest. Ingested MXFP4 is 11.29% from
  shipped NVFP4 and 18.03% from direct BF16-to-MXFP4. Both the ingested and
  direct MXFP4 values execute through the same native package and match their
  decoded-weight references exactly. This one projection with synthetic FP8
  activations exposes material added conversion error; broader quality and
  selector promotion remain open.
- Latest: [native gfx1201 bounded NVFP4 row replay](INTEGRATED_COMPILER_LOG.md#2026-10-09--native-gfx1201-bounded-nvfp4-row-replay)

### ROCM-SPLIT-K-1

**Split-K on the typed ROCm route: landed on gfx1201 f16/bf16 (2026-09-26); slice rule measured on 16 shapes on the device clock (2026-09-27)**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: **Current gate (2026-09-27):** the three original conditions are met on gfx1201 (occupancy key, production consumer, Tajasarus router-gate proof -- see the 2026-09-26 update below), and the per-shape slice rule and the device-clock timing witness closed on 2026-09-27 (update at the end of this record). Still open: fp8/int split, gfx1151 (unmeasured, never split), and M > 64 / dynamic shapes. **Gate history (as written 2026-09-19, superseded):** `rocm_tiling.rank_candidates` computes `split_k_required` and **no production path reads it** — `scheduled_matmul.py` never imports the module, so the emitted gfx1201 kernel has no split-K whatever is ranked. Retained under Decision #29a rather than deleted, because the gfx1201 MoE router gate (M≤16, K=2048, N=256) shows split-K is genuinely needed: 16 output tiles leave half the machine idle (the part measures 64 CUs = **32 WGPs**, and a workgroup dispatches to a WGP), and the weight read dominates A by 16×, so the MMA unit is not the scarce resource. **The model is also wrong for that shape**: `split_k_required = k > 4096` answers `False` at K=2048, because the real trigger is occupancy (tiles < WGPs), not K magnitude. Gate: re-key the predicate on occupancy, give it a consumer on the typed route, and prove the router-gate shape on Tajasarus against a measured baseline — with the reduction's determinism carried as a semantic key (Decision #21a), since a split-K reduction is exactly where a reproducible router→top-k is lost. Until all three land, the declaration stays marked unwired at its site per #29a condition 1. **Updated 2026-09-25 (code read): the first of the three landed** — `rocm_tiling._split_k_required` has been keyed on occupancy (output tiles against measured WGPs via `rocm_target.dispatch_slots`, LDS overflow still forcing it) since `d4297b36` (2026-09-20). Still open: a production consumer (only `tests/unit/test_macro_tile_selection.py` reads it) and the Tajasarus router-gate proof.
**Updated 2026-09-26 — the consumer landed and the router gate is device-proven on gfx1201 (Tajasarus).** Cross-workgroup split-K, one decider: `selectGfx1201SplitK` (PMPasses.cpp) is the production authority and `rocm_tiling.select_split_k` (built on `split_k_required`) is its declared oracle, recomputed by `scheduled_matmul.verify_matmul_projection` on every package and refusing any disagreement (Decision #31). Rule: gfx1201 f16/bf16, static, K-blocked (`block_k=32`); split when output tiles < 32 WGPs, `S = ceil(32/tiles)` rounded down to a power of two, only as far as each slice stays whole macro K blocks of >= 256 (an **unmeasured** guard, stated as such). `split_k` + `split_k_reduction="ordered"` are a semantic pair (Decision #21a) carried through `schedule.matmul` (+ the digest, appended only when S>1 so every unsplit digest is unchanged), `tile.matmul_kernel` and `tessera_rocm.wmma_gemm`; each verifier fails closed (`SCHEDULE_/TILE_/ROCM_WMMA_GEMM_SPLIT_K_BAD_CONTRACT`), and there is no atomic mode. The typed generator emits a **partial** (grid.z = slice, fp32 `[S, M, N]` workspace, no epilogue) and an **ordered reduce** (sums slices 0..S-1 in fixed order, then applies bias/activation once); the runtime launches both. A K with no aligned split stays unsplit with a `ROCM_SPLIT_K_NOT_APPLIED` warning (only when K is large enough to split and misaligned; below two 256-wide slices is outside the rule and silent); the LDS body, the directive adapter and non-f32 contracts refuse a split (`ROCM_SPLIT_K_UNSUPPORTED`). Lit: `tests/tessera-ir/phase2/e2e_rocm_split_k_schedule.mlir` (incl. the negative `@wide`/`@ragged_k`), `rocm_split_k_contract_invalid.mlir`, ROCm `typed_matmul_split_k.mlir` (with an `@unsplit` CHECK-NOT control).
**Measured, Tajasarus RX 9070 XT, 16x256x2048** (`benchmarks/baselines/rocm_split_k_20260926/gfx1201.json`; paired, interleaved, 3 fresh-process runs x 15 rounds x 200 iterations; host wall clock over synchronized batches, WSL2, no counters, `performance_eligible=False`). Split time includes BOTH launches. Selected `split:2` vs the unsplit kernel of the same Tile IR: **fp16 2.05x, bf16 2.01x** (7.84 vs 16.06 us, 8.02 vs 16.04 us/iter; split faster in 45/45 rounds each). Correctness: relative error vs f64 1.6e-6 (fp16) / 5e-7 (bf16); split vs unsplit max relative difference 1.7e-6 / 7.4e-7; the device tests (`tests/unit/test_rocm_split_k.py`, ragged 15x200x2048 edge store, bias+gelu/relu epilogues) also check that two launches are bit-identical. **Finding, not acted on:** the measurement-only slice sweep says the rule is conservative on this shape -- S=4 is 2.52-2.58x and S=8 2.79-2.80x -- one untested hypothesis is that one wave per WGP leaves three of its four SIMDs idle (no counters exist on this WSL2 host to confirm it); the selection was not retuned from one shape. **Not proven:** gfx1151 (the rule never splits there; no Princess-Luna run), fp8/int storages (never split), dynamic shapes, the LDS body, and a device-clock timing witness.
**Updated 2026-09-26 (pre-PR review fixes):** the partials' scratch is declared in the descriptor's typed `workspace` (S*M*N*4 B, 256-aligned, launch lifetime, uninitialized) and the launcher allocates from it and cross-checks provenance; the artifact's split now comes from the C++ Schedule IR and the oracle is compared in the projection; a derived `k_unroll` that does not divide the slice falls back to 1 (recorded as `k_unroll_split_fallback_from`), a pinned one is refused; `ROCM_SPLIT_K_NOT_APPLIED` is a warning emitted only for K >= 512 that does not divide (K below two 256-wide slices is outside the rule); the reduce workgroup is validated. The reduce kernel stays a descriptor-carried second entry rather than a `ROCMNativeProgram` plan -- see the log entry for why.
**Updated 2026-09-27 -- the slice rule is measured, on the admitted device clock (sync `GFX1201-LANES-2026-09-27`).** `benchmarks/rocm/record_split_k_sweep.py` times every image between two compiler-built `--tessera-device-clock-span` markers. Each variant's windows become a `profiler_rocm_packet` whose `device_clock_witness` admission `build_rocm_profiler_packet` derives. The sweep covered 16 skinny shapes (router gates, MoE expert/decode GEMMs, M<=64) x {f16, bf16} x S in {1..32}. The 2026-09-26 rule (tiles < 32 WGPs, `S = ceil(32/tiles)`) was conservative everywhere it split, and it never split shapes that gain: 16x768x2048 (48 tiles) gains 2.43x at S=4, 64x512x2048 (128 tiles) 1.82x at S=2. **New rule, in `selectGfx1201SplitK` and the oracle together:** split when `2 x tiles <= 256`, and take the largest power-of-two `S <= min(32, 256/tiles)` whose slices are whole K blocks of >= 256. It was chosen because every selection it makes was measured positive in both storages, not because it hits each shape's peak. Router 16x256x2048 moves from S=2 to S=8: 2.18x -> 3.40x fp16 and 2.17x -> 3.37x bf16, 27/27 rounds. The negative side is measured. 32x1536x4096 (192 tiles) is neutral at S=2 and loses from S=8. tiles x S of 1024 / 4096 loses. The 256-per-slice guard is now measured at its boundary: K=256 into 128-wide slices loses at every S, and K=512 into 256-wide slices gains 1.16-1.18x. Caveats: 16x2048x768 S=2 (1.12-1.24x) is admitted in only 1-2 of 3 runs, because the marker-overhead gate is sporadic. Absolute times depend on window length, so ratios are compared within one sweep. Why splitting pays past one workgroup per WGP is unmeasured (no counters). Packet: `benchmarks/baselines/rocm_split_k_20260927/`.
- Depends on: —
- Start: device
- Latest: [2026-09-27 — ROCM-SPLIT-K-1: device-clock slice sweep; measured 256-workgroup target](INTEGRATED_COMPILER_LOG.md#2026-09-27--rocm-split-k-1-device-clock-slice-sweep-measured-256-workgroup-target)

### GOV-ODS-CONSUMER-1

**Decision #29's op-level clause has no gate**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: **CLOSED 2026-09-27 (gate); the waiver it seeds is the measured #29 debt (sync `EVIDENCE-GOVERNANCE-GATES-2026-09-27`).** A first gate landed 2026-09-20 (`tests/unit/test_ods_op_has_consumer.py`) but was hollow three ways: its regex matches 286 of the 623 declared op records on the current tree (it matched only `def X : Op<Dialect, "name">` and skipped the dominant `def X : Dialect_Op<"name">` class-template form), it counted a lit fixture as a consumer, and it matched bare substrings. Rebuilt on `compiler/ods_consumer_audit.py`: a balanced TableGen reader that resolves class templates and refuses what it cannot read (`defm`, an unknown op-like base, a duplicate class), pinned to the exact record count (`_DECLARED_OP_RECORDS`); cross-checked once, by hand, against `llvm-tblgen --dump-json` (every op record tblgen could dump agreed on name and mnemonic; three files it cannot dump are read here) -- no gate re-runs that check; references are tiered *compiler* (C++/TableGen under `src/`/`tools/`, Python under `python/tessera/`) / *fixture-only* / *unreferenced*, and comments, docstrings, an dialect's own implementation files, name registries (`op_catalog` and friends), prose in strings, C++ member access and foreign or `using`-imported class names (`scf::YieldOp`, `using mlir::func::FuncOp`) do not count. **A fixture alone does not satisfy #29**; fixture-only ops are waived and listed. The shrink-only waiver (`_WAIVER_CEILING`; 84 at landing, 38 fixture-only and 46 unreferenced) gives each entry a reason; an adversarial review before the PR found three fail-open holes that had passed seven ops, each now pinned by a synthetic test. **None meets #29a** (no op is marked at its site with an owning item), so every entry is a plain #29 violation for their dialect owners to consume or delete; none was deleted. Found alongside: seven `tessera.neighbors.*` names are declared by two ODS records (`TesseraOps.td` and the unbuilt `tessera_neighbors.td`, which also shadows a hand-written C++ dialect), gated by a separate shrink-only ratchet. **Resolved 2026-09-27 (sync `SMALL-CORRECTNESS-GAPS-2026-09-27`):** `TesseraOps.td` is the one authority (the parser resolves `tessera.neighbors.x` to the `tessera` dialect, so it was the only live one); the unbuilt `.td`, the hand-written dialect and its registration in `tessera-opt`/`tessera-translate` are deleted, the semantics only the dead copies stated (stencil.define tap/coefficient well-formedness, neighbor.read's required `delta`) are now the core verifier's, the duplicate ratchet is absolute (baseline empty), and `test_no_hand_written_cpp_op_shadows_an_ods_op` gates the C++ form the ODS scan could not see. The gate as it stood (history): #29 says a declared ODS op must have a named consumer or be deleted, and cites `tests/unit/test_governance_declarations.py` as its drift gate. Measured 2026-09-19: that file checks **coverage axes** (`test_every_contract_axis_names_a_consumer`) and **duplicate dialect names** across ODS files (the Queue.td/Attn.td trap), and **nothing maps an ODS op to a consumer**. Confirmed the hard way — `tessera.scaled_matmul` was committed with no consumer and all 14 governance tests passed. That also explains #29's own history: every instance it cites (`manifold` reaching no backend, `MultivectorSpec.grades`, nine `!tile.*` types, `numeric_policy` with no carrier, `TilingInterface`) was found **by hand**, which is what an ungated rule produces. Gate: a check that each op in the Tessera ODS files is referenced by at least one pass, lowering or verifier beyond its own declaration, with a shrink-only waiver list for declarations held under Decision #29a — the same ratchet shape as `DIAG-PY-BACKLOG-1`, and it should seed from whatever the first scan finds rather than assuming the count is small.
Gate: a check that each op in the Tessera ODS files is referenced by at least one pass, lowering or verifier beyond its own declaration, with a shrink-only waiver list for declarations held under Decision #29a — the same ratchet shape as `DIAG-PY-BACKLOG-1`, and it should seed from whatever the first scan finds rather than assuming the count is small.
  Connection triage 2026-09-27 ([ODS_OP_CONNECTION_TRIAGE.md](ODS_OP_CONNECTION_TRIAGE.md), reference): every waived op has a row -- 6 WIRE (each naming producer, consumer and proof after the same-day review found 19 first-draft WIRE rows naming only one end), 30 #29a debt candidates (25 under named items, 5 DNAS ops awaiting an owner item), 25 merge/supersede, 17 delete candidates for the owner, 5 AMX directed-closed, 1 consumed op the scan missed (`tile.tmem.store`). It also records scan blind spots proposed as follow-up here (not changed): catalog-driven frontend producers (`_try_map_call`), interface-method producers (`istft_jvp`), and prefix/default-branch consumers.
  2026-09-28 (sync `ODS-WIRE-B-2026-09-28`): triage slices 2 and 3 landed -- `tessera.istft_jvp` is consumed by `GraphToSchedulePass` and the ISTFT JVP package builds from its hashed contract (the kwargs derivation is now a declared oracle); `tessera.cache.commit/rollback` lower through an x86 handle ABI. Three ops left the waiver (ceiling 76). sm_120 ISTFT proof owed.
- Depends on: —
- Start: host-free
- Latest: [2026-09-28 — ODS wiring slices 2 and 3: `tessera.istft_jvp` gets its Schedule consumer; `cache.commit/rollback` lower through an x86 handle ABI](INTEGRATED_COMPILER_LOG.md#2026-09-28--ods-wiring-slices-2-and-3-tesseraistftjvp-gets-its-schedule-consumer-cachecommitrollback-lower-through-an-x86-handle-abi)

### ROCM-FP8-BLOCKSCALE-1

**Block-scaled FP8 is a different contract from dequant, and that is the gap**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: **Design established 2026-09-19; two earlier readings of this item were wrong and are corrected here.** (a) It said Tessera had "no FP8 scale concept at all" — false, and false by searching in vLLM's vocabulary: `microscaling.ScaleLayout` models block_size/axis/scale_dtype with `e8m0` (MX), `fp8_e4m3` (**named for NVFP4**) and `fp32`, `Tessera_ScaleLayoutAttr` carries it in IR, and four ops already hold it. (b) It then said the work was "extend the contract to the plain matmul" — also not quite right, because **`tessera.dequant_matmul` already is a plain scaled matmul**: `w_codes` packed int4/int8/**fp8_e4m3/fp8_e5m2**, optional `w_scale`, `scale_layout`, `quant_group_size` along the contraction axis, with a working ROCm generator (`GenerateROCMDequantGemmKernel`). **The real gap is that dequant is the wrong shape.** `dequant_matmul` computes `y = x @ dequantize(w, s)` — scales applied *before* the multiply, operands widened, and the ROCm kernel is a scalar f32 loop (`O = sum_k X * code * scale`). That is the MI355 strategy, and it is why MI355 lands on bf16. Block-scaled **W8A8** is the opposite: both operands **stay fp8**, the MMA is `v_wmma_f32_16x16x16_fp8_fp8`, and the scale is applied to the fp32 **accumulator** per block. Only the second reaches the 383 TFLOP/s ceiling; the first is capped by the widened GEMM. Gate: a block-scale contract on the *scheduled* matmul path that keeps operands in their storage and rides the scale into the accumulator/epilogue — **not** an extension of `dequant_matmul`, which would force widening by construction. Implementation constraint, measured: gfx1201 has **no scale-carrying matrix instruction** (zero `scale` hits in the calculator's `-L`), so the scale is software, and with `block_shape=[128,128]` against our K tile of 16 a K-block spans eight slabs — accumulate 128 K in fp32, scale, add to a running total. That is a schedule change, not an epilogue tweak. Measured target unchanged: the 20 tuned R9700 configs at 64/128/128, against which our route still picks 16×16 for all 60 shape/M combinations.
**Vendor precedent for the shape, read 2026-09-19 from rocMLIR** (`rocmlirTriton/mlir/lib/Dialect/Rock/Transforms/RockToTTIR.cpp`, local on Tajasarus): AMD's own MLIR generator puts the scales **on the dot op**, not as a dequantize-then-multiply — `triton::DotScaledOp(cType, a, b, c, scaledA, scaledB, aElemTy, bElemTy, fastMath, aKPack, bKPack)`, carrying per-operand `ScaleDotElemType` and K-pack flags beside the accumulator. Its comment names the anti-pattern in as many words: *"This ensures f8 (or packed f4) uses float MFMA (via dot_scaled) instead of integer MFMA (via dot with i8 operands)"* — i.e. the construct exists precisely to keep low-precision operands on the **right instruction class** rather than widening them. That is the same conclusion reached above from the MI355 bf16 fallback, now with a vendor implementation behind it, and it argues for a scaled form at Tessera's `tile.mma` / scheduled-matmul level rather than a Graph-level dequant. **Caveat on reach:** that lowering carries no WMMA handling (rocMLIR's WMMA support sits in separate tuning files), and Triton on gfx1201 is independently reported to lower `tl.dot_scaled` by upconverting e2m1 to bf16 for the 16-bit WMMA — so **block-scaled low precision on the RDNA4 WMMA appears unserved by the MLIR/Triton stack today**, which is why the hand-written gfx1201 kernel in §10b exists. Stated as what was read, not as a survey. Incidental corroboration from the same file: `InputPrecision::BF16x3` for f32 dots is the three-bf16 TF32 emulation recorded in `rocm_target.py`.
**Corrected again 2026-09-19 — there IS a working reference on this chip.** An earlier draft of this entry said block-scaled low precision on the RDNA4 WMMA was "unserved across AMD's own stack", from hipBLASLt shipping zero MX libraries for gfx1201 and rocMLIR's scaled path being MFMA-oriented. Both facts hold; the conclusion did not, because it conflated two Triton paths. **FP8 W8A8 block-scaled reaches the native fp8 WMMA on gfx1201 today** via AITER's `gemm_a8w8_blockscale` Triton kernel at `block_shape=[128,128]` — which is precisely what the 20 tuned R9700 configs are — with reported 25% (Qwen3-0.6B) and 63% (Qwen3-30B) decode gains on an R9700, AITER's C++/ASM kernels disabled because they do not run on RDNA4, **M ≥ 16 required**, 11 shapes tuned, and no upstream merge. What is *not* served is the MXFP4/`tl.dot_scaled` path, which upconverts e2m1 to bf16 — that is [ROCM-MXFP4-W4A8-1](#rocm-mxfp4-w4a8-1)'s territory. **This changes how to judge the item**: it is not building something nobody has, it is reaching through our own contract what a Triton kernel already reaches, with a working implementation to measure against rather than only a config file. The `M ≥ 16` floor and the M-bucketed config keys are the same skinny-M axis [ROCM-SPLIT-K-1](#rocm-split-k-1) already owns.
**The reference implementation, read 2026-09-19** (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py`, local on Tajasarus; AITER now carries gfx1201 support). The whole scale contract is one line in the K loop:

```python
accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
```

Operands stay **fp8** into the dot, the dot accumulates in **fp32**, and the fp32 result of *this K block* is multiplied by the **outer product of the two scale vectors** (`a_scale` per row, `b_scale` per column) before being added to the running accumulator. Scales touch neither the operands nor a final epilogue — they are applied per K block, between the MMA and the accumulate. The wrapper asserts **`GROUP_K == BLOCK_SIZE_K`** twice: the scale group along K *is* the tile K. That confirms the schedule derived above from first principles — with `block_shape=[128,128]` and a WMMA K of 16, a macro K tile of 128 means eight WMMA steps inside one scale block, then one scaled accumulate — and confirms it is a schedule change rather than an epilogue tweak. A second variant collapses to `(a_scale * b_scale)[:, None]` where one side is not per-column, and a preshuffled-weight variant advances `b_ptrs` by `BLOCK_SIZE_K * 16`. So the gate is no longer a design question: it is expressing this contract on the typed route and measuring against a kernel that already runs on this chip.
**Implementation path corrected 2026-09-21 — reuse the loop body, not the `kUnroll` semantic.** The typed K loop is an `scf::ForOp` carrying the accumulators as iter_args, with `mainPanel(...)` threading them through `tile.mma` (`C += A*B`, in place). Block scaling needs *this scale group's* product isolated before it is scaled and added: start a local accumulator from zero, issue `scale_k / instruction_k` panels, scale by the outer product, then add into the carried accumulator. The existing panel-unroll machinery can generate those panels, but the authored contract remains `scale_k`; `kUnroll` stays an independent latency/code-size knob and `k_blocks` stays the macro K tile. AITER's W8A8 route happens to assert `GROUP_K == BLOCK_SIZE_K == 128`. That equality is route-specific, not generic: OCP MXFP4 fixes `scale_k = 32` while the reviewed gfx1201 kernels use 64- or 128-wide slabs containing two or four scale groups. The generator must therefore permit `instruction_k | scale_k | macro_k` and must not derive any of those three from `kUnroll`.
**Host-free carrier landed 2026-09-21.** `python/tessera/compiler/rocm_mxfp4.py` keeps instruction K 16, MX scale group K 32, and macro K/unroll/split-K independent. `tessera.scaled_matmul` now reaches a first-class Schedule/Tile scaled-partial carrier: every scale group starts at zero, accumulates whole instruction panels, is scaled, then joins the running accumulator. A named `rocm_mxfp4_w4a8_exact_v1` physical-container form binds raw E4M3 A, packed E2M1 B, fp32 token scales, E8M0 K32 scales, and BF16 output to the proved gfx1201 WMMA ABI without promoting MXFP4 to a public Graph storage dtype.
**Binary integration landed 2026-09-21.** The packed Target directive now carries static M/N/K and the full physical/numerical contract. `package_scaled_wmma_target_ir` admits exactly one named packed directive, validates instruction K16 / scale K32 / macro K32, BF16 output, isolated scale-then-add partials, exact-per-block policy, and the proved package ABI, then materializes the existing gfx1201 WMMA generator. Logical W8A8 and approximate-fold directives remain unbound. Launch provenance retains Tile/Target digests and the schedule hash.
**Logical W8A8 bound on the typed gfx1201 route 2026-09-27 (sync `GFX1201-LANES-2026-09-27`).** An fp32-scale `tessera.scaled_matmul` over e4m3 A/B now reaches a gfx1201 HSACO through Graph → Schedule → Tile → Target with no Python code emission (Decision #31). Graph→Schedule **derives** (never accepts) `rocm_fp8_w8a8_blockscale_v1`, or `_nk_v1` for a weight stated `[N, K]` via `transposeB`, from the op's types; `scale_layout.block = [scale_n, scale_k]` is the weight block, `lhs_scale` is fp32 `[M, K/scale_k]` and `rhs_scale` fp32 `[K/scale_k, ceil(N/scale_n)]`, every extent checked while the tensor types exist, and a nonconforming fp32-scale form is refused by name (`ROCM_FP8_BLOCKSCALE_CONTRACT`) rather than left unbound -- layout, dtype and group are semantic keys (#21a). `scale_n` joins `schedule.matmul` and its digest only when set. The C++ WMMA generator emits the isolated scale-group body: each group starts a **zero** partial, walks its `scale_k/16` instruction panels (`scale-group-panels`, a performance key, default 2), and joins the accumulator through a new Tile op `tile.fragment_scaled_accumulate` whose ROCm consumer shares one accumulator element map with the fragment store. `instruction_k | scale_k | macro_k` stay independent; `kUnroll` counts whole groups and never sizes one. The Target directive now carries the bound `package_abi`, `scale_n` and the register panel; `rocm_fp8_blockscale.py` checks every field before binding a launch descriptor. The W8A8 panel is its own measured rule (32x32 at ≥ 256 tiles, else 16x32/16x16), because the doubled accumulator live set makes the unscaled 64x64 panel spill.
  **Device (Tajasarus):** 23 rows bit-equal (exact inputs) / within fp32 accumulation error of an fp64 block-scale oracle on ragged and multi-group shapes, both weight layouts, scale N blocks 128/16/1, group steps 0/1/2/4 and a two-group iteration with its remainder; each row asserts `v_wmma_f32_16x16x16_fp8_fp8` in the ISA and the generator's per-group zero/join/MMA counts; one row checks the result is far from a single-rescale kernel. Same file passes under the assertions-ON `tessera-opt`; lit 453/66-unsupported and `check-tessera-rocm` 82/82 on both trees.
  **Against AITER `gemm_a8w8_blockscale` (local checkout, unmodified, AOT-compiled by Triton 3.8 with its tuned gfx1201 configs), device-clock timing, paired and interleaved, 51 measured rows:** Tessera `[N, K]` time / AITER time geomean **0.65 at M ≤ 64** (22/24 faster), **1.09 at M = 256**, **1.31 at M ≥ 1024** (up to 1.94x slower at K = 7168). We win decode-sized M and lose prefill-sized M; AITER's large-M configs are LDS-staged multi-warp tiles and ours is the one-wave register panel. The `[K, N]` layout loses almost everywhere (strided B gather). Not a matched-output comparison: we store f32, AITER bf16. Three AITER split-K buckets were not measured. [Evidence packet](../../../benchmarks/baselines/gfx1201_fp8_blockscale_20260927/README.md).
  **Open (superseded by the next paragraph for large M and the store):** the large-M gap (an LDS-staged or multi-wave W8A8 body; the 64x32 panel's 13% at 4096^3 is one shape), a bf16/f16 store epilogue, AITER's split-K buckets, and the `[K, N]` B gather.
  **Large-M LDS body and a bf16 store, 2026-09-27 (sync `GFX1201-PERF-2026-09-27`).** A second physical body on the typed route: eight waves of 32 rows share one LDS-staged K slab (A `[wgM][128]`, the `[N, K]` weight `[wgN][128]`, 16-byte padded rows, 16-byte copies, LDS-only barriers), with each wave's scale-group semantics exactly the register body's -- bit-identical to it on device. Graph→Schedule selects it for the `[N, K]` weight at M >= 128: 128x128 when that tiling gives >= 64 workgroups (the RX 9070 XT's CU count) on a whole-128 M, else 128x64 when that covers the CUs (every ragged M: the 128x128 body's masked edge costs 13 VGPRs), else the register panel; `staging` is stated on `schedule.matmul` (digested when set), the Tile carrier and the Target directive, from which the package binds its 256-thread workgroup. The contract also admits a bf16 Graph result, rounded once (RNE) by the typed store under distinct package ABIs. Against unmodified AITER, device clock, paired: `[N, K]` / AITER geomean **0.91 at M >= 1024** (was 1.29; 12/18 faster), **0.90 at M = 256** (was 1.08), 0.65 at M <= 64; the bf16 store is timing-neutral (0.998). Every production kernel of the comparison is byte-identical at the final compiler. Measured negative: double buffering, a register-staged next slab, a 64-byte slab, 0/32-byte padding, 4-/16-wave grids; grouped raster and zero-filling the out-of-range rows were neutral. [Evidence packet](../../../benchmarks/baselines/gfx1201_fp8_blockscale_lds_20260927/README.md).
  **Open:** 6 of 18 M >= 1024 shapes remain 1.04-1.08x behind AITER (K <= 2048 or N = 1024 -- not the output bytes, which bf16 halved without effect); ragged M is 1.075x AITER geomean over 20 points and 1.37-1.56x at N = 24576, K = 1536 (a 64-row LDS tile and a cheaper masked edge are the candidates); the `[K, N]` weight stays on the register panel; AITER's split-K buckets remain unmeasured.
  **CU count from one authority; ragged M at the whole-M tile, 2026-09-27 (sync `FOUNDATION-BATCH-2-2026-09-27`).** The rule's CU count is `measuredComputeUnits` (PMPasses.cpp), a mirror of the new `rocm_target.compute_units` (2 x the measured WGP count) that a unit test compares entry for entry; `lower_blockscale` refuses a Schedule whose panel its Python oracle does not reproduce, and an unmeasured arch keeps the register panel with a registered `ROCM_FP8_BLOCKSCALE_LDS_NOT_APPLIED` warning. The ragged-M gap was the bounded store, not the tile height: the block-scale join's per-element rows, hoisted above the K loop, were reused by a store written against absolute rows, so a ragged M = 1000 took 251 VGPRs at 128x128 (238 whole) and ran 1.25x slower than M = 1024 on the same grid. The typed store (`materializeFragmentStore`) now tests each element's row as a constant against the lane's room and addresses it from the lane's row base (same elements and addresses; column unchanged), giving 240 VGPRs with no spills for ragged M, N or both; ragged M then takes the whole-M rule. Against unmodified AITER, device clock, paired: ragged-M geomean **0.965** (was 1.086, 26 points), 0.90x of the old tile where the selection changed (0.83-1.08; 200x8192x1024 loses 8%), whole-M kernels byte-identical and the 54-row comparison unchanged. Short K is still open: grouped raster (~1-3%), 16-wave grids, a register-staged next slab and double-buffered LDS at stage K 64 all measured negative. [Evidence packet](../../../benchmarks/baselines/gfx1201_fp8_blockscale_ragged_20260927/README.md).
  **2026-09-28 follow-up, sync `GFX1201-W8A8-M200-SHORTK-2026-09-28`.** A bounded gfx1201 rule now selects the 128x64 LDS panel for M=192..255, whole-128 N=8192..10240, K=1024 after paired candidate/incumbent sweeps showed wins at the measured corners and interior points; N=4096/6144, K=1536/2048 and M=300 are excluded by measured neutral or losing results. The rebuilt compiler's production image passed the fp64 oracle and selected that panel; at 200x8192x1024 it measured 44.16 µs against AITER 44.42 µs in one post-rule run. The pre-rule interleaved sweep, not absolute cross-run timing, supports the choice. The other 200x2048x2048 case remains unchanged. [Packet](../../../benchmarks/baselines/gfx1201_w8a8_followup_20260928/README.md). Princess-Luna has now timed the gfx1151 typed fragment-store route at two ragged shapes, but only in synchronized WSL2 host-wall time without a pre-change image; [packet](../../../benchmarks/baselines/gfx1151_edge_row_store_20260928/README.md). Neither speedup nor regression of that store is established.
  **Open:** the remaining short-K / N = 1024 whole-M shapes and K = 1536 ragged rows; the `[K, N]` weight on the register panel; AITER's split-K buckets unmeasured; gfx1151 edge-row store before/after device-clock timing remains unavailable.
- Depends on: —
- Start: device
- Latest: [2026-10-06 — gfx1201 native MXFP8 long-K selector trial](INTEGRATED_COMPILER_LOG.md#2026-10-06--gfx1201-native-mxfp8-long-k-selector-trial)

### ROCM-MXFP4-W4A8-1

**MXFP4 on gfx1201 is reachable through the fp8 WMMA, and we refuse it instead**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: RDNA4 has no FP4 WMMA form, and `select_fragment_layout` therefore refuses `fp4_e2m1` outright. That is correct about the hardware and wrong about the opportunity: the ecosystem answer on this exact chip (vllm-radiance `radiance_mxfp4_fp8.hip`, canonical `StillDeadcode/libr4d` `r4d_gemm_mxfp4a8_nt_m64.hip`) reaches `v_wmma_f32_16x16x16_fp8_fp8` — **the instruction ROCM-MIXED-FP8-1 proved** — through W4A8. Two routes must not be conflated. **Exact per-block:** E2M1 converts exactly to E4M3, two 16-wide WMMAs form one 32-element MX group, and its E8M0 power-of-two scale is applied exactly to that FP32 partial before it joins the running accumulator. **Folded row reference:** choose the maximum E8M0 exponent per output row, shift each block's E2M1 values into E4M3, and restore the row factor once in the epilogue. That fast fold is exact only while every non-zero shifted magnitude remains representable: gfx12 E4M3 subnormals extend the guaranteed range through exponent delta 8; the reviewed checkpoint reaches delta 10, where the source reports rounding and a non-zero residual error rather than exactness. Therefore the folded route is an explicit approximate `numeric_policy`, never the proof oracle or an unconditional replacement for the exact route. W4A8 itself remains opt-in because its activation precision differs from the checkpoint's declared W4A4 calibration.
  **Exact native vertical slice landed 2026-09-21.** `rocm_mxfp4.py` owns packed E2M1 (`low nibble = even k`), E8M0 `[K/32,N]`, per-token FP32 activation scale, fragment-order permutation/inverse, exact decoding, folded-row-reference payloads, and separate policy metadata. `rocm_mxfp4_native.py` adds an executable scalar specification and an exact WMMA route: all independent fragment loads are issued before decode/packing, two `v_wmma_f32_16x16x16_fp8_fp8` operations form one local K32 FP32 partial, and only that partial is scaled and added to the running accumulator. Tajasarus proves both packages bit-exact after BF16 rounding at ragged `17x19x64` and `32x32x128` (**4 device rows**); the recorded WMMA HSACO is wave32, 32 SGPR / 94 VGPR, zero LDS/scratch, and selects no other matrix instruction. Both exact ABIs are in the gfx1201 owning-device registry. [Evidence packet](../../../benchmarks/baselines/gfx1201_mxfp4_w4a8_20260921/README.md).

  **Production K-step/prefill slice landed 2026-09-22.** The versioned fragment ABI now serves decode and prefill. A portable Schedule/Tile isolated-scale-group contract lowers on AMD to a selective VMEM/WMMA scheduling boundary; alternating exact-device timing shows +3.3% and +0.6% on the two decode shapes and neutral prefill. The padded fragment-word LDS producer must drain its writes before `s_barrier`; the wide-N oracle caught the missing drain, then passed 10/10 repeats and the full file passed 16/16 after the fix. A true two-stage producer/consumer pipeline is correct but 13–15% slower and remains unselected. Group-M, waves-per-EU, cache, and LDS-padding sweeps found no stable promotion; streaming cache loses. The selected exact route reaches 0.0490/0.0885 ms decode and 0.5174/4.3722 ms prefill with two FP8 WMMAs and zero scratch/spills. Interleaved independent timing leaves decode 1.33x/1.02x from Radiance (the second shape beats libr4d), but prefill 3.44x/4.81x behind. The remaining prefill gap is architectural and numerical-policy-bound: Radiance uses BM256/TM4 multi-output-wave reuse and the explicitly approximate row-reference fold, whereas Tessera still applies exact K32 scales. Next is a separately opted-in folded ABI plus the taller tile; never promote it as exact-route tuning. No source was copied from the reviewed projects. [Evidence packet](../../../benchmarks/baselines/gfx1201_mxfp4_kstep_prefill_20260922/README.md).
  **Opt-in folded prefill landed 2026-09-22.** A separate `approx_bm256_tm4.v1` gfx1201 ABI consumes E4M3 `[N,K]` weights and E8M0 `[N]` row exponents prepared once with explicit approximate-policy consent. Payload hashes and quantified fold loss travel in the descriptor; the runtime checks identity, selected architecture, shape, and policy before launching. A 256×64×64 tile uses four M fragments and two N fragments per wave, padded LDS, and 32 FP8 WMMAs with no spills. Tajasarus tests deliberately lossy underflow and ragged N, then a matched lossless-fold benchmark proves full BF16 agreement against the exact K32 oracle and pinned Radiance. Tessera's folded route takes 0.1775/1.2296 ms versus exact 0.5064/4.3589 ms, leaving 1.25x/1.38x to Radiance. The exact package stays the default and correctness reference; the expanded E4M3 weight traffic and A/B staging cost remain open performance work. [Evidence packet](../../../benchmarks/baselines/gfx1201_mxfp4_folded_prefill_20260922/README.md).
  **Scale-cancellation repair and refused phase probe, 2026-09-22.** The folded epilogue combines its row and activation scales before multiplying the FP32 partial, with a rare FP64 fallback for overflowing or underflowing scale products; six Tajasarus device tests include zero-accumulator and finite-after-overflow cases. Matched timing is 0.1812/1.2391 ms versus Radiance 0.1428/0.8872 ms. A synchronized same-CTA diagnostic probe preserved output but emitted 64 versus 32 WMMAs and 24 versus four barriers, so its phase fractions are refused as production attribution and cannot train a selector. IKF-P0 and ISA-preserving stage measurement remain open. [Refusal packet](../../../benchmarks/baselines/gfx1201_folded_phase_diagnostic_20260922/README.md).
  **External-probe preflight and folded IR carrier, 2026-09-22.** Tajasarus has `rocprofv3` but no `/dev/kfd`; PC sampling cannot run there, and a kernel-trace attempt yielded no artifact. The exact-device preflight records this refusal and leaves cross-CU clocks and phase attribution unvalidated. A distinct folded `tessera.scaled_matmul` physical contract now carries E4M3 B `[N,K]`, E8M0 Ref `[N]`, explicit approximate policy, a full-K FP32 partial, BM256/TM4/K64 physical schedule and separate pointer/package ABI through Graph→Schedule→Tile→Target. The materializer validates that carrier before compiling the existing load-time-folded HIP package and binding payload/loss metadata. The `65x48x64` generic route passed one Tajasarus launch and the folded file passed 7/7 in an isolated trial build; exact K32 remains the oracle and default. Broader shapes and frontend integration remain open. [Preflight packet](../../../benchmarks/baselines/gfx1201_phase_profiler_preflight_20260922/README.md).
  **Typed folded frontend and corrected matched comparison, 2026-09-22.** A bounded gfx1201 author now constructs the Graph op from physical A/token-scale/folded-weight shapes, lowers it through Schedule/Tile/Target, and returns a receipt binding schedule, ABI, IR hashes, fold-loss metadata, and HSACO identity. Tajasarus passes 11/11 folded device tests, including both measured prefill shapes and a deliberately lossy approximate oracle; this supersedes the preceding paragraph's broader-shape/frontend-open statement. The earlier Radiance comparison did not bind its WPERM layout mode and is historical, not selector-admissible. The v2 matched benchmark requires fragment-order `RADIANCE_MXFP4_WPERM=1` and measures the frontend package at 0.182356/1.242242 ms versus Radiance 0.142856/0.888937 ms. A selected-symbol ISA census and source-derived load-request model show doubled B requests but still larger A restaging requests; they do not measure DRAM traffic or phase time. One isolated B non-temporal-load ablation emitted exactly one changed cache-hint site, preserved BF16 output, and lost 20%/104%; it remains unselected. Next: A-staging reuse/load scheduling on matched inputs, broader nonuniform numerical envelopes, and a profiler-capable gfx1201 host for IKF-P0. Exact K32 stays default; no folded selector promotion. [Packet and limits](../../../benchmarks/baselines/gfx1201_mxfp4_folded_frontend_20260922/README.md).
  Receipt and coverage follow-on, 2026-09-22: the receipt's `hsaco_sha256` now hashes the actual HSACO bytes; `artifact_image_digest` separately retains the composite target/toolchain/image identity. Two additional exact-device frontend rows cover ragged `65×48×64` and multi-K-step `257×80×192` lowering and launch, bringing the folded device file to 13/13. The generic exact selector is regression-tested to refuse the folded layout without explicit approximate authoring. Clean, uncontended matched timing remains 1.28×/1.40× behind Radiance; the older losing B-cache packet's receipt metadata was corrected from its retained payload hash without changing timing. Independently, the 95-case scheduled package gate ran with zero skips on Tajasarus against freshly rebuilt current `main` (`89b2f2fd`), not as a substitute for the follow-on branch's MXFP4 proof. Neither result promotes automatic folded selection. [Refreshed packet](../../../benchmarks/baselines/gfx1201_mxfp4_folded_frontend_20260922/README.md).
  **Folded load schedule carried in Target IR, 2026-09-27 (sync `GFX1201-LANES-2026-09-27`).** The 2026-09-23 packed/A-address/TN4/graph follow-ons are recorded in the ROCm queue. This slice attacks the named A-staging/load-scheduling step on matched inputs. Hot-operand probes showed the one-row-block gap was not operand traffic: the 256×5120 shape improved only 4.7% with A and B both cache-hot. A `FoldedPrefillSchedule` of four performance keys is now emitted by Tile→ROCm on `tessera_rocm.scaled_wmma_gemm` and consumed fail-closed by the folded materializer. The keys are grouped M-major raster (4), register-staged next-K64-slab prefetch, a complete-tile vector-scale epilogue, and CU workgroup mode once M spans at least two BM256 row blocks. The tile, WMMA order and epilogue arithmetic are unchanged: every engine is bitwise equal to exact K32. Direct packaging keeps the original schedule, whose emitted source is byte-identical. Device-clock marker timing is witnessed within 2.3% by HIP events across three processes on Tajasarus. It gives selected/original 0.93–0.94× and 0.79–0.80× on the two production shapes, now 1.06–1.07× and 0.98–1.01× Radiance. It is faster than the original on all ten shapes and at or ahead of Radiance from M=1024; M≤256 stays 1.06–1.27× behind. WMMA 32 and barriers 2+2 are unchanged; VGPRs go 109→123 with no spills. Unconditional K16 steps and LDS fragment double-buffering are recorded measured-negative, and LDS-only barriers neutral. Exact K32 stays default, folded stays opt-in, no automatic folded selection. [Packet](../../../benchmarks/baselines/gfx1201_mxfp4_prefill_20260927/README.md).
  **Per-wave M guard for partial row blocks, 2026-09-27 (sync `GFX1201-PERF-2026-09-27`).** The one-row-block gap at M <= 192 was the BM256 tile multiplying rows that are never stored: at M = 128 half of every workgroup's waves compute clamped copies of the last row. Diagnostic probes first ruled out the per-lane 64-bit staging address arithmetic (a wave-uniform base: 0.99-1.00x) and the K16 scheduling barrier (1.00-1.01x). A fifth Target-IR performance key, `row_guard` (`cta` | `wave`), now skips the WMMAs and epilogue of a wave with no row below M (it still stages and meets every barrier) and takes the vector epilogue's completeness test per wave; Tile→ROCm selects `wave` only for a partial row block, so whole-row-block kernels are byte-identical, and output stays bitwise equal to exact K32. Device clock witnessed by HIP events, three processes: M = 128 runs 0.62-0.66x of the original schedule, **0.74-0.77x Radiance at N = 5120** (was 1.10-1.11x) and **1.04-1.12x at N = 17408** (was 1.24-1.27x). M = 256 (one full row block, no idle wave) stays 1.05-1.23x behind and is unattributed; exact K32 stays default, folded opt-in. [Evidence packet](../../../benchmarks/baselines/gfx1201_mxfp4_small_m_20260927/README.md).
  **The one-row-block gap is not the weight bytes, 2026-09-27 (sync `FOUNDATION-BATCH-2-2026-09-27`).** Tested the hypothesis that Radiance's packed E2M1 weights (half the bytes) explain M = 256. An N scan at M = 256, K = 5120 with three rotating input copies and with one (weights cache-resident where they fit), device clock witnessed by HIP events, three processes, every engine bitwise equal to exact K32: residency closes the gap at N = 4096 (1.03-1.07x -> 0.98-1.01x) but not at N = 8192-12288, where both weights are resident and 1.11-1.15x remains; the marginal cost per output column is 16.0-16.7 ns against Radiance's 12.7-12.8 in both regimes. The opt-in packed-E2M1 candidates (half the bytes, bitwise exact) are 1.20-1.48x Radiance, slower than the expanded schedule at every N. Not the bytes; the per-column cost is unattributed (no counters). Exact K32 stays default, folded opt-in. [Evidence packet](../../../benchmarks/baselines/gfx1201_mxfp4_one_row_block_20260927/README.md).
  **2026-09-28 diagnostic, sync `GFX1201-MXFP4-M256-SLOPE-2026-09-28`.** A fixed-K one-copy N scan at 8192/12288/17408 ablated selected schedule keys in two independent processes, with bitwise exact K32 agreement and device-clock/HIP-event windows. The selected marginal cost was 17.99-18.09 ns per added column against Radiance 13.43-13.60; removing the vector epilogue left 17.69-18.10, removing prefetch 18.62-18.82, and removing raster 17.83-17.93. These keys help absolute latency but do not explain the slope. The remaining mechanism is in the core loop or launch geometry; A restaging, LDS fragment traffic and issue remain unseparated without counters or a validated phase ablation. The source tree was dirty, so the packet is diagnostic and makes no promotion claim. [Packet](../../../benchmarks/baselines/gfx1201_mxfp4_m256_decomposition_20260928/README.md).
- Depends on: [ROCM-FP8-BLOCKSCALE-1](#rocm-fp8-blockscale-1)
- Start: device
- Latest: [2026-10-06 — gfx1201 packed vector-scale epilogue](INTEGRATED_COMPILER_LOG.md#2026-10-06--gfx1201-packed-vector-scale-epilogue)

### EVIDENCE-PACKET-1

- Current slice (2026-10-06, sync `ROCM-MATH-WIDENING-2026-10-06`): gfx1151 physical math reloads 21 serialized native Graph/Schedule/Tile/Target packages across f32/f16/bf16 inputs, with f32 output and exact launch identity. Zero metadata probes remain in this recorder. Ordinary JIT and portable cast-to-math products have independent gfx1151/gfx1201 numerical proof and 72-row packets per architecture. No selector promotion. [Evidence](../../../benchmarks/baselines/rocm_native_math_20261006/README.md).

**Evidence packet consumers**

- Owner: [EVALUATOR_PLAN.md](EVALUATOR_PLAN.md)
- Gate: Unify missing packet consumers around artifact/image identity, clock validity and promotion eligibility; malformed or incomplete evidence must refuse. The [benchmark alignment review](../../../benchmarks/COMPILER_ALIGNMENT.md) maps six suites to actual callers: at that review point math was metadata-driven, GA/EBM composition lacked route receipts, and direct-IR AD probes needed paired public-frontend coverage. New diagnostic math/AD runs no longer inherit performance eligibility from target names or historical packets. The extended suite review retires synthetic SuperBench timing, labels DLOP dispatch counts as estimates and lattice timings as host-wall, and adds per-submission native guards for Apple policy-loss timing. Bounded CUDA/HIP ANN adapters now record driver-launch receipts; public SSD VJP has independent numerical comparisons. Dedicated CUDA SSD kernel/API attribution and serial/cooperative SSD adapters are validated; sequential synchronous mixed-artifact CUDA attribution and CUDA GEMM/attention adapters now pass. The ROCm matrix/unary generators now use LLVM 23 inherent kernel properties; assertions-enabled native compilation and bounded gfx1151/gfx1201 execution pass. Extend asynchronous attribution and clean performance admission; the slower gfx1201 LDS/pipelined diagnostic variants remain unpromoted. **2026-09-27 slice (sync `EVIDENCE-GOVERNANCE-GATES-2026-09-27`): calibration-corpus eligibility.** `target_perf.apply_corpus`, the consumer that turns a calibration corpus into selector authority, read `corpus.get("selector_eligible", True)` and never read `ineligibility_reasons`; `load_pruning_corpus` read neither. A corpus that omitted the field was trusted, and one stating eligibility beside its own reasons was believed. Both loaders now go through `corpus_selector_eligibility`, which refuses a missing or non-bool eligibility or malformed reasons (`CALIBRATION_CORPUS_ELIGIBILITY_INCOMPLETE`), eligibility the reasons contradict (`CALIBRATION_CORPUS_ELIGIBILITY_CONTRADICTED`) and an undeclared tag (`CALIBRATION_CORPUS_REASON_UNKNOWN`; `CALIBRATION_CORPUS_REASONS`, which the gfx1151 recorder checks before writing). The committed 2026-08-15 gfx1151 corpus still reads and still cannot promote. **2026-09-27 slice (sync `EVIDENCE-PACKET-1-2026-09-27`): shared envelope + GA/EBM route receipts.** `evidence_envelope.read_evidence_packet` is the one reader for the three measurement-packet families (x86 Zen 5 profiler v1/v2, ROCm gfx1151/gfx1201 profiler, NVIDIA sm_120 device clock): it runs the family validator unchanged, projects artifact/compiler identity, timing domain, clock validity, environment, source revision/worktree, sample ids, route and eligibility onto one envelope, refuses a missing or malformed field (`EVIDENCE_ENVELOPE_INCOMPLETE`) or an unregistered schema (`EVIDENCE_ENVELOPE_SCHEMA_UNKNOWN`), and enforces what no family may waive (`EVIDENCE_ENVELOPE_CONTRADICTED`): eligible exactly when no refusal cause is named, and an eligible packet has a valid clock, a clean tree, a sample id and its measured image named by its timing sample (on every route; the family checked binding only on the device-clock route). The ROCm and x86 derivations read `worktree_dirty`/`virtualized`/`wsl` by truthiness, so an omitted field derived no blocker; both now require bools. SSD admission (ROCm and NVIDIA device-clock routes) reads its calibrations through the envelope, and a drift test refuses a production module that calls a family validator directly. 186 of 188 committed packets read (149 promotable, unchanged); the 2 refusals are the pre-existing window-rule pair below. `tessera/_route_receipts.py` gives GA/EBM per-call route receipts: all 32 `_try_<target>_*` native-lane helpers are `@native_attempt`, the 29 public callers `@public_route`, orphan native dispatch or an empty capture leaves a span `unattributed`, and `clifford_core`/`energy_core`/`visual_complex_core` rows carry `route` + `route_receipts` and derive `device`. Receipts recorded on Mac, Princess-Luna, Tajasarus and Super-Bear (`benchmarks/baselines/ga_ebm_route_receipts_20260927/`): the Zen 5 hosts run EBM energy/partition on x86 AVX-512 (the jit_bridge trace, Apple-only, would have called that no native dispatch), the Mac reaches the Apple GPU runtime, Super-Bear runs the reference, and no composition reaches a ROCm or CUDA GPU lane. **2026-09-28:** all seven x86 physical-math rows now reload serialized native packages and bind launch receipts to exact artifact/image/descriptor identities (16 focused tests and 31 samples per row on Princess-Luna). At that point ROCm math remained metadata-driven. **2026-10-06:** the 21 gfx1151 math package consumers are migrated; explicit narrow Graph casts feed native f32 math. **Still open:** paired public-frontend coverage for the direct-IR AD probes; asynchronous/multi-thread attribution; DLOP profiler receipts; clean performance admission; compiler build identity in the ROCm/NVIDIA packets (the envelope records it where x86 has it and does not require it); the CUDA activity-window calibration (`profiler_cuda_window`), the calibration corpus and E2E-spine packets as envelope families; and chip attribution for a `rocm` receipt (the lane is named, the chip is only the recording host's). A future bare-metal x86 profiler-route packet needs per-row timing witnesses naming its images before the envelope admits it as eligible (none exists; every fleet host is WSL2). Two committed diagnostic/superseded ROCm packets (`gfx1151_ssd_calibrated_pairs_interleaved_20260926/diagnostics/launches_probe/cooperative-100-calibration.json`, `gfx1201_ssd_calibrated_pairs_20260926/superseded/second_6e6904dc_short_window/`) do not validate under the current window rule (they derive `DEVICE_CLOCK_WINDOW_TOO_SHORT`); that predates this slice and they are retained history, not admitted evidence.
- Depends on: —
- Start: host-free
- Latest: [2026-09-29 — evidence consumers, NVIDIA fragments, and public residuals](INTEGRATED_COMPILER_LOG.md#2026-09-29--evidence-consumers-nvidia-fragments-and-public-residuals)

### TPROF-NATIVE-1

**Native clocks and counter attribution**

- Owner: [INTRA_KERNEL_FEEDBACK_PLAN.md](INTRA_KERNEL_FEEDBACK_PLAN.md)
- Gate: The CUDA adapter now calibrates seven launch-inclusive event windows against exact Nsight product-kernel spans, checks captured device identity and rejects excessive profiler overhead. After removing SSD's unrelated 1024-element tape-parser cap, a 512x2x32x8 cooperative workload agrees within 0.51% with 0.24% profiler overhead. The source remains dirty and the host WSL. A nine-pair collector now binds fresh process nonces and Nsight PIDs and refuses unqualified hosts before production collection. Validate per-process clean-image calibration and counter attribution on each owning host; WSL regression evidence is not clean timing.
- Depends on: —
- Start: device
- Latest: [2026-09-05 — Real asynchronous GEMM comparison](INTEGRATED_COMPILER_LOG.md#2026-09-05--real-asynchronous-gemm-comparison)

### DISPATCH-BREAKER

**Safe waits and checked-output ownership**

- Owner: [../backend/README.md](../backend/README.md)
- Gate: Audit remaining bridge waits and checked-output readers; bounded failure must retain live resources and poison unsafe ownership before wider adoption.
- Next architecture gate: [gated owner and isolated recovery](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#gated-metadata-owner-and-isolated-recovery-2026-09-11): every admitted metadata operation in the opt-in gated owner has replay/device proof; live metadata and legacy import/snapshot paths refuse. Process-owned heap recovery requires confirmed death. Since 2026-09-15 ([probed admission and replacement](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#probed-admission-and-health-checked-replacement-2026-09-15)) a worker is admitted only after its in-process device probe verifies the admitted producers, and a replacement is admitted only after confirmed predecessor death plus its own probe (RTX 5070 and gfx1151). Next: migrate legacy snapshot/import callers to a gated producer and validate actual driver-failure behavior (the recorded fault is an injected stall). Epoch ordering remains; no measured overlap or promotion.
- Depends on: —
- Start: host-free
- Latest: [Pinned gfx1201 checkpoint format decision gate](INTEGRATED_COMPILER_LOG.md#2026-10-03--pinned-gfx1201-checkpoint-format-decision-gate)

## F2

### E2E-REAL-6

- Current increment (2026-10-05, sync ROCM-MOVEMENT-SPINE-2026-10-05):
  canonical static paged reads on gfx1151/gfx1201 and explicit MoE token gather
  on gfx1151 retain adjacent native Graph/Schedule/Tile/Target/backend ancestry.
  Exact capabilities, numerical fixtures and checked execution rows agree.
  Nine retained-route host comparisons pass bit-exactly and clear the 10%
  non-regression gate. General public tensor semantics/layouts/asynchronous
  movement and resident-kernel admission remain open.
  [Packet](../../../benchmarks/baselines/rocm_movement_admission_20261005/README.md).

- Current increment (2026-10-05, sync ROCM-NATIVE-MOVEMENT-2026-10-05):
  native HIP host orchestration and context-owned staging execute unchanged
  compiled paged-KV images on gfx1151/gfx1201 and MoE images on gfx1151.
  Nine bit-exact rows retain separate host/event timings and zero warm
  allocations. General layouts, asynchronous movement and retained-route
  performance admission remain open.
  [Packet](../../../benchmarks/baselines/rocm_native_movement_20261005/README.md).

- Current increment (2026-10-03, sync NVIDIA-BROADCAST-CHECKPOINT-CORE-2026-10-03):
  rank-four broadcast checkpoint native arithmetic passes 12 RTX5070 rows with
  physical-shaped deterministic dBias and output capacity guards. Public paired
  AD export/replay retains the physical shape. Checked broadcast package/tape
  ABI remains landing; dense-copy descriptor admission is guarded until host
  copy extents, saved-state capture and backward allocations are integrated.
  [Core packet](../../../benchmarks/baselines/nvidia_broadcast_checkpoint_core_20261003/README.md).

**Remaining Graph-owned packaging and frontend retirement**

- Current slice (2026-10-01, sync COMPILER-NEXT-FIVE-2026-10-01): the
  public SM120 RMSNorm -> matmul edge again passed with a four-iteration typed
  accumulator; 21-sample timings remain noisy. Saved-LSE attention passed
  eight full/causal, fp16/fp32 regular/ragged forward rows and the checkpoint
  suite; backward device-event median was 1.0871 ms with stable event timing,
  while host E2E was noisy. On gfx1201, synthetic NVFP4 ingest passed ten
  focused tests and reused the same native image; three static matmul shapes
  reused one shape-free image with max absolute error below 3.6e-7 and 31
  focused cache tests passed. These are correctness/attribution results only.
  Open: generic tensor-valued W1.1 producers on direct legacy Tile input; the
  canonical SM120 Graph pipeline lowers before those patterns. NVIDIA saved-output
  row-delta backward now passes independent-oracle checks at [1,4,2,16,16,32,32]
  and [1,4,2,128,128,64,64], with a 2.378 ms device median at shape 128 versus
  183.179 ms for the prior saved-LSE path; default selection is unchanged. Wider
  dynamic/dtype/checkpoint envelopes, model-scale/source-BF16 NVFP4 comparison,
  and broader ROCm cache envelopes remain open.
  [Evidence log](INTEGRATED_COMPILER_LOG.md#2026-10-01--five-compiler-slice-rechecks).


- Current slice (2026-09-29, sync E2E-REAL-6-ALIBI-SHAPE-2026-09-29): the public explicit-slopes ALiBi trace now infers f32 [H,S,S] from num_heads/seq_len rather than inheriting slopes[H]. Princess-Luna traces, packages, launches and matches NumPy through the checked AVX-512 ABI. Apple, ROCm and NVIDIA follow-ups remain target-owned.

- Current slice (2026-09-29, sync E2E-REAL-6-ROCM-MATMUL-IMAGE-2026-09-29): static unfused unsplit gfx1151 f16/bf16 register matmul now compiles a shape-independent Target directive image after Schedule/Tile replay. Exact gfx1151 numerical proof and a three-shape compile control are recorded; gfx1201, fused/split/dynamic/LDS and host replay cost remain open. [Packet](../../../benchmarks/baselines/gfx1151_matmul_shape_key_20260929/README.md).

- Current slice (2026-09-29, sync E2E-REAL-6-ALIBI-2026-09-29): x86
  explicit-slopes ALiBi now enters native Graph, Schedule and Tile, and the
  narrowed Python constructor is retired. Princess-Luna numerical and
  differential proof covers the existing AVX-512 ABI. Apple JIT packages,
  ROCm paged layouts/matmul identity, NVIDIA LSE/backward/quantized routes and
  the route census remain open. Diagnostic WSL host-wall timing is not
  performance admission. [Evidence](../../../benchmarks/baselines/x86_alibi_native_20260929/README.md).

- Current slice (2026-09-28, sync `COMPILER-NEXT-SLICES-2026-09-28`): x86 rank-3 Cholesky/triangular solve now use native Graph→Schedule→Tile and their retained Python constructor is retired. The derived breadth census recognizes the compiled route. ROCm forward-attention images now reuse across runtime batch/head/sequence extents, with exact gfx1151/gfx1201 numerical proof; per-shape guards and Schedule ancestry remain distinct. Native MoE/paged-KV execution was revalidated on gfx1151 and paged KV on sm_120. Movement non-regression and most sm_120 timing-stability gates remain unsatisfied. The Mac passed direct scaled-RoPE/Philox Metal ABI tests, but static `target_verify`/`ntk_rope` @jit remains artifact-only. Remaining: Apple native package execution; x86 ALiBi explicit slopes; general paged layouts and launch overhead; NVIDIA LSE/backward/quantized routes; ROCm matmul image identity and physical W8A8/MXFP4 work. See the [measured packet](../../../benchmarks/baselines/compiler_next_slices_20260928/README.md).

- Owner: [MLIR_NATIVE_FOUNDATION_SURVEY.md](MLIR_NATIVE_FOUNDATION_SURVEY.md)
- Gate: Static gfx1201 f16 matmul and f16/bf16 forward/backward attention packages retain exact driver/Schedule/Tile/image ancestry. Automatic public GQA Q/K/V AD now selects the exact live ROCm architecture and executes a recompute-only backward package, with artifact-bound physical certificates. Reusable HIP recompute workspace now owns queued host submissions/retirement; dynamic f16 matmul runs several shapes through one image. Private HIP streams and same-device read-only output leases now gate reuse and retirement; failed releases retain retryable ownership. Two independent programs have intersecting event windows in four of five uncalibrated gfx1201 trials. Bounded fp16/bf16 sparse packing now enters a registered Target op and production lowering, with six exact native comparisons. Explicit saved-LSE now reuses immutable resident O/LSE across repeated cotangents; auto remains recompute. Internal packed sparse Schedule/Tile producers now lower to SWMMAC with exact device comparisons. A logical row-major producer now performs packing/index selection in compiled GPU IR, accumulates multiple K tiles, and emits validity words; fp16/bf16 gfx1201 checks cover six cases through 48x32x128. The explicit compiler API now binds logical matrices through a validity-consuming isolated runtime package; explicit JIT tracing now specializes a single matmul under a checked 2:4 precondition, preserving output storage; declared checked-2:4 and auto_2to4 Graph matmul lower their physical recipe in C++. The auto policy selects sparse/dense per K tile on device; default-dispatch promotion remains open. Reader release can now enqueue asynchronously with non-cancellable completion and retryable failure; four gfx1201 checks verify external copies across saved/recompute and synchronous/asynchronous release. Isolated ANN replacements preserve an explicit device ordinal after confirmed worker death and repeat the numerical health probe; in-process attention teardown remains separate. Explicit sparse binding now verifies validity before exposing outputs and confirms process teardown; matching f16/bf16 accumulation has bounded gfx1201 instruction/numerical proof. Resident attention accepts exact fp32/fp64 cotangent conversions and pending host futures. Isolated attention confirms worker death before replacement and reruns a workload-scoped zero-VJP probe. A dropped saved-LSE forward carrier was corrected with reciprocal companion verification. Signed i8 and same-format E4M3FN/E5M2 sparse packing now have emitted-instruction and device comparisons. Public attention VJP accepts exact wider cotangents at capture and launch. Isolated admission checks both zero and nonzero VJP against the oracle. Mixed FP8 operand orders and independent integer signedness are now carried through the sparse package and all three IR levels. K=32 INT4 now uses byte-addressable logical inputs, explicit native nibble packing and host/device range checks. Automatic sparse selection, broader INT4 envelopes, arbitrary public AD, pointer-sharing isolation and actual driver-hang recovery remain open. F32, BF16-to-f32, FP64 and mixed uint8/int8-to-int32 x86 matmul now enter Graph-to-Schedule-to-Tile and project native ABI from replay-verified artifacts. The mixed recipe carries ui8/i8 signedness, rejects wider-than-i32 runtime extents, and has owning-CPU ragged/overflow differential proof. At this historical point, ALiBi and batched rank-3 cholesky/tri_solve were the remaining x86 constructors; the later 2026-09-28 slice above retired batched linalg, leaving ALiBi. **Apple now has a first F2 family:** the canonical GEMM reduction lowers to verified `tessera_apple.gpu.tensor_view` + `gpu.matmul2d` Target IR (storage pair incl. f16 × FP8/FP4, fp32 accumulator, Apple's 128-byte MTLTensor layout quantum) and from there to the runtime's Metal 4 matmul2d symbols with the view ABI projected from IR; seven operand pairs execute on the M1 Max against an exact-bytes oracle; packed operands off the quantum are refused, not re-formed. Standalone passes only — the incumbent Apple route is unchanged and the `apple_native` GEMM packager is not deleted. Require per-target differential proof and preserve policy before deleting each route.
- Current increment: x86 elementwise / cohort-2 / breadth (2026-09-28, sync `E2E-REAL-6-x86-kernel-2026-09-28`): `package_elementwise`, `package_cohort2` and `package_graph_breadth` admit through the scheduled contract and hand `lower_scheduled_kernel` to `package_scheduled_kernel`; the native owner is `NativeX86Kernel.h` (plus `NativeAbsolute.h`, and `schedule.norm` for x86 norms), after 46 catalog spellings were registered as Graph ODS ops so the native compiler could parse them. The retired Graph constructors are a declared oracle under `tests/_support/`; 501 envelope points agree on descriptor, Tile op and Target IR host-free and bit-for-bit on device on both Zen 5 hosts. `elementwise` is generic on the prune dashboard; `cohort2` (ALiBi: its Graph operand list is not decodable and ODS has no slopes operand) and `breadth` (rank-3 cholesky/tri_solve: the ODS ops are rank-2) keep one narrowed retained constructor each and stay gap. An x86 compile cache keyed on the compiler digest, pass and exact source text makes a repeat package call ~0.8 ms (cold 64-81 ms; compile-cost, Princess-Luna). Historical next steps: batched linalg and the bounded ROCm paged-KV route landed in the later 2026-09-28 slice above; ALiBi and broader paged layouts remain.
Previous increment: x86 unary family (2026-09-28, sync `E2E-REAL-6-x86-unary-2026-09-28`): x86 `supports_softmax` / `supports_reduction` admit through the scheduled contract and `package_softmax` / `package_reduction` hand `lower_scheduled_kernel` straight to `package_scheduled_kernel` for both the AVX-512 and `x86_64_base` images; the Graph-owned `_softmax_contract` / `_reduction_contract` and `emit_softmax_tile_ir` / `emit_reduce_tile_ir` are a declared differential oracle under `tests/_support/`. The contract now carries the one envelope point it had lost (`softmax_safe`), the descriptor carries the replayed `nan_mode`, and 288 bitwise retired-vs-compiled device rows pass on both Zen 5 hosts. The runtime now loads one copy of an x86 payload instead of one per shape-bound image digest. Historical next families; the bounded x86 and ROCm migrations are recorded above.
Previous increment: ROCm unary family (2026-09-27, sync `E2E-REAL-6-rocm-unary-2026-09-27`): gfx1151 `package_softmax` / `package_reduction` no longer author Tile IR from the Graph object; they lower through the native Schedule contract, which now carries the envelope those constructors served (f16/f32 softmax incl. `softmax_safe`, f16/bf16/f32 reductions with f32 output, keepdims), and the ROCm consumer projects the storage-keyed ABI and `nan_mode` from replayed IR. The retired constructors are a declared differential oracle under `tests/_support/`; 179 gfx1151 device rows agree bit-for-bit with them, gfx1201 keeps its proved f32 envelope, and the old name-derived combiner (`tessera.reduce {kind = "max"}` executed as a sum) is gone. Historical next families; the bounded ROCm Schedule producer and x86 migrations are recorded above.
Previous increment: APPLE-MATMUL2D-1 follow-through (2026-09-15): ragged M/N bind at their true extents (no host zero-padding), sub-block origins and padded strides are views into the parent and reach the dispatcher through one strided-view runtime entry for every pair, the fused bias/activation epilogue is `gpu.matmul2d_epilogue`, and a two-process paired corpus admitted the family into the default pipeline for the 8/4-bit storage pairs only (no executable incumbent existed); f16/bf16 keep the incumbent. The `@jit` front door for 8/4-bit storage tensors is closed: the tracer, the Graph IR spellings, the matmul result rule and the MPS dispatcher were each silently treating FP8/FP4 as f32 (a host numpy product reported as native_gpu); they now name, spell, type and dispatch the low-precision pairs on the Metal matmul2d lane. The shared TilingPass now preserves the matmul bias/residual epilogue (re-applied as broadcast + add after the nest), so the bias-operand form reaches every backend and Apple's fused op. The bf16 arbiter bucket is measured and retained (the corpus's 5–12% was wrapper overhead in the incumbent path; on device time the entries are the same kernel). Nothing on this route is open beyond the recorded follow-ups (view-wrapper host zero-fill, default-mode migration). Previous x86 increment: the selected x86 static f32 `absolute`, `floor`, `ceil` and trailing-axis `cumsum` slices now enter a native durable Schedule contract and Tile producer. Packaging replays both boundaries and projects bindings, shape, layout and numeric behavior from serialized IR; the old Python Tile constructor is bypassed for these slices. Owning Zen 5 checks cover three ragged/ranked shapes and bitwise signed-zero/subnormal/infinity/NaN magnitude for absolute, and signed-zero/subnormal/infinity/fractional values plus NaN classification for floor/ceil. The F0 lexical census still finds Graph-to-emitter paths in `package_cohort2`, remaining `package_elementwise` branches and `package_graph_breadth`; annotation counts do not establish per-envelope closure. Cumsum now bypasses the cohort emitter with replay-bound inclusive-scan policy and owning-CPU proof. Other cohort, elementwise and breadth constructors remain open.
Compiler-example increment (2026-09-20): the maintained Qwen3-MoE example
exposed that Graph artifacts retained an optional route tensor without its
`route` identity. Both frontends now preserve named MoE tensor tails, the CPU
runtime decodes that metadata rather than forwarding it as a public-op kwarg,
and every Apple CPU example launch is oracle-checked. Accelerator examples
remain artifact claims; exact-device execution stays backend-owned.
- Depends on: [E2E-REAL-6F](#e2e-real-6f): census and proof requirements for the selected route, not all certificates.
- Start: host-free
- Previous slice (2026-09-30, sync `E2E-REAL-6-RESIDENT-DYNAMIC-K-2026-09-30`): the paired resident RMSNorm → matmul packages now reuse one Graph → Schedule → Tile image for bounded K prefixes on gfx1201 and sm_120. Public `from_text` traces, fp16/bf16 storage, numerical parity and producer/consumer residency are proved on each owning GPU. Event variance remains diagnostic; dynamic K with M/N and broader layout coverage remain open. [Evidence](../../../benchmarks/baselines/gfx1201_resident_dynamic_k_20260930/README.md) and [SM120 packet](../../../benchmarks/baselines/sm120_rmsnorm_matmul_edge_20260930/dynamic_k_sm120.json).
- Previous slice (2026-09-30, sync `E2E-REAL-6-RESIDENT-STRIDED-INGRESS-2026-09-30`): paired resident packages accept padded/sliced host views and normalize them to the compact backend ABI before upload. CUDA upload staging is held until successful stream synchronization. gfx1201 and sm_120 exact-device suites prove fp16/bf16 bounded-K parity and allocation/image reuse; timings exclude host packing and remain diagnostic. Broader device layouts and K combined with M/N remain open. [padded-ingress benchmark packets](../../../benchmarks/baselines/resident_strided_ingress_20260930/README.md).

- Current slice (2026-09-30, sync E2E-REAL-6-RESIDENT-DYNAMIC-MNK-2026-09-30): The Graph-to-package contract composes bounded dynamic M/N/K on gfx1201 and sm_120. Exact-device fp16/bf16 parity, padded host ingress, allocation residency, and image reuse pass. Separate fp16 and bf16 event packets support stage attribution only; gfx1201 variation is high. Apple and x86 need independent resident consumers; wider physical layouts and remaining W1.1 producers stay open. [Evidence](../../../benchmarks/baselines/resident_dynamic_mnk_20260930/README.md).

- Latest: [2026-10-07 — native paged-KV flat-token indexing](INTEGRATED_COMPILER_LOG.md#2026-10-07--native-paged-kv-flat-token-indexing)

Current increment: checked rank-four broadcast saved-LSE packages and private tape execute on SM120; physical bias extents, deterministic reduction and saved-state pairing are bound through native Graph/Schedule/Tile and checked CUDA descriptors. [Evidence](../../../benchmarks/baselines/nvidia_broadcast_checkpoint_package_20261003/README.md). Broader W1.1, frontend/AD and sibling routes remain open.

### W1.1

- Historical slice (2026-09-29): sm_120 refused the two legacy tensor-valued TileIRLoweringPass MMA forms at NVIDIA lowering before an async token could become an invalid MMA data operand. That stop-sign behavior is superseded for registered static matmuls by the 2026-10-01 named PM pipeline integration recorded below. Generic tensor-to-fragment producer migration remains open. [Evidence](../../../benchmarks/baselines/compiler_evidence_fragment_residual_20260929/README.md).

**NVIDIA typed-fragment closure**

- Owner: [../backend/nvidia/todo.md](../backend/nvidia/todo.md)
- Gate: Reconcile remaining NVIDIA fragment producers and Target proof against current typed routes; retain uncovered dtype/architecture obligations and exact-device gates.
- Depends on: [E2E-REAL-6F](#e2e-real-6f): census and proof requirements for the selected route, not all certificates.
- Start: host-free
- Current slice (2026-10-01): public affine-free fp16/bf16 LayerNorm now
  feeds a resident scheduled SM120 matmul through Graph -> Schedule -> Tile.
  Exact RTX 5070 tests passed both dtypes; the fp16 256x256x256 packet records
  independent parity, allocation ownership, zero spills, and separate
  producer/consumer event times. Generic legacy tensor-valued Tile producers
  remain open.
- Latest: [2026-10-07 — native compiler regression repair](INTEGRATED_COMPILER_LOG.md#2026-10-07--native-compiler-regression-repair)
- Integration follow-through: zero-error Python type gate restored; bounded frontend lifetime, strict metadata and native stage lineage guards have 125 focused WSL regressions. General producer composition and wider dtype/layout/AD envelopes remain open.
- Current RHS increment: static fp16/BF16 RMSNorm KxN output feeds typed matmul B through checked row-major packages. Exact RTX 5070 complete/ragged cases validate private resident lifetime and separate producer/consumer dispatch windows. FP8, MXFP8 and MXFP4 remain mandatory before strategy/default selection.
- Census recheck (2026-10-02): both historical tensor/async-copy constructors
  remain registered only for sm<120. Explicit SM120 static canonical tensor
  M/N/K loop functions now recover their semantic Graph contraction after
  whole-function tiling replay equality, then use native Schedule/Tile storage
  and typed fragments. Exact RTX 5070 fp16/BF16 plain and bias/ReLU/residual
  packages pass. Arbitrary producer graphs, noncanonical accumulator semantics,
  dynamic generic reconstruction and older-target device proof remain open.

- Current physical increment: native Schedule/Tile row-major RHS views gather into typed B fragments on SM120. Named LHS normalization/softmax producers preserve C/F RHS storage through ordinary JIT, resident execution and portable replay with FP16/BF16 exact-device proof. General producer/layout integration remains open. [Evidence](../../../benchmarks/baselines/nvidia_lhs_rhs_layout_20261006/README.md).

### W3.3

**Tile dialect ownership review**

- Owner: [IR_STACK_INTEGRATION_REVIEW.md](IR_STACK_INTEGRATION_REVIEW.md)
- Gate: Reassess surviving Tile primitive/kernel/domain/solver ownership before any dialect split; move only consumed semantics with parse/lower/execute regression gates.
- Depends on: [E2E-REAL-6F](#e2e-real-6f): census and proof requirements for the selected route, not all certificates.
- Start: host-free
- Latest: [2026-09-05 — Deleted functionality reassessment](INTEGRATED_COMPILER_LOG.md#2026-09-05--deleted-functionality-reassessment)

## F3

### FRONTEND-IR-MEDIUM-1

- Current slice (2026-09-29, sync `COMPILER-EVIDENCE-FRAGMENT-RESIDUAL-2026-09-29`): a second public tracer residual `(x*x-theta)*(x+theta)` preserves canonical Graph operations and passes exact-device native tape checks on gfx1151 and sm_120. General solver source integration, masks and sibling backends remain open. [Evidence](../../../benchmarks/baselines/compiler_evidence_fragment_residual_20260929/README.md).

**Native recipes and broader raising**

- Owner: [FRONT_END_LOWERING_ASSESSMENT.md](FRONT_END_LOWERING_ASSESSMENT.md)
- Gate: Exact f32 attention loops, including a positive exactly representable f32 literal post-dot scale, explicit grouped-head indexing and end-aligned causal masking, now raise to a symbolic recipe and instantiate two native buckets through Schedule/Tile. The opt-in NVIDIA binding now projects native Schedule fields, replays the recipe and executes both buckets without Graph reconstruction. Apple and x86 now project target-specific parents; the x86 package executes on Zen 5. CUDA now proves ragged Q/K causal GQA buckets, with native rejection of nondivisible head counts. Bounded asymmetric window masks now execute on CUDA; mandatory serialized Presburger constraints reject fully masked rows, including when callers supply additional constraints. Finite full-shape additive bias now crosses native instantiation and Schedule projection and passes CUDA differential execution. NVIDIA raised binding now accepts irregular full-shape additive negative-infinity masks and rejects empty rows after causal/window composition; CUDA differential execution passes, including irregular masks combined with ragged GQA, causal and left-window masking for Q>K and K>Q. Extend Boolean/padding/broadcast masks, fully masked-row semantics and recognition, Apple device proof, ROCm-compatible storage and measured candidate admission; preserve complete witnesses and IR/image-bound admission.
- Current increment: Broadcast additive masks now pass on every axis: batch/head, key-padding (`[1,1,1,K]`) and per-query (`[B,Hq,Q,1]`) forms pass source recognition, symbolic bucket instantiation, native Schedule/Tile replay (the physical block survives Tile lowering) and SM120 indexing. The v3 f32 runtime ABI carries all four physical bias extents, copies only that storage and retains the seven logical kernel extents; empty-row refusal judges the logical broadcast view. Six B=2 ragged causal/GQA/window cases pass device comparison and empty-row refusal on the RTX 5070; ROCm refuses a broadcast score-bias block. Boolean/padding masks as an operand, sibling-target consumers, f16/bf16 broadcast storage and performance admission remain open.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Current slice (2026-10-01): public affine-free fp16/bf16 LayerNorm now
  feeds a resident scheduled SM120 matmul through Graph -> Schedule -> Tile.
  Exact RTX 5070 tests passed both dtypes; the fp16 256x256x256 packet records
  independent parity, allocation ownership, zero spills, and separate
  producer/consumer event times. Generic legacy tensor-valued Tile producers
  remain open.
- Current increment (2026-10-05, sync NVIDIA-JVP-NATIVE-SCHEDULE-2026-10-05): bounded static f32 SM120 saved-LSE JVP now lowers the actual native AD Graph through replay-sealed C++ Schedule/Tile, native arena and NVVM/LLVM. Python GPU arithmetic construction and inactive-load string rewriting are retired. Ten ordinary JIT oracle cases, direct/automatic saved-state products and separate dispatch/wall measurements are retained; general dynamic/composed/bias/dropout/value-only AD and sibling routes remain open. [Evidence](../../../benchmarks/baselines/nvidia_jvp_native_schedule_20261005/README.md).
- Current compiler overhead increment (2026-10-05, sync NATIVE-COMPILE-ORCHESTRATION-2026-10-05): native JVP passes retain SSA in one MLIR pass manager; GPU packaging reuses exact tool SHA-256 for stable file identities and rejects read-time rebuilds. Image/arena bytes remain identical in balanced A/B packages; cold and warm identities are measured separately. Native persistent compiler sessions and image-validator overhead remain open. [Evidence](../../../benchmarks/baselines/native_compile_orchestration_20261005/README.md).
- Current increment (2026-10-08, sync SCALED-MAP-AXIS-INTEGRATION-20261008): non-leading static typed FP8/MXFP8 input maps preserve alias views, scalar bounds, native scale JVP seeds and original VJP axis order. Checked C++ host packing retains compact compiler ABI admission. Owning gfx1201 execution and separate completed-public/native-event timings are recorded. Generic closure, dynamic/nonzero-output maps, wider storage AD and sibling packed/native routes remain open. [Evidence](../../../benchmarks/baselines/scaled_map_axes_20261008/README.md).
- Latest: [mapped result-axis semantic foundation](INTEGRATED_COMPILER_LOG.md#2026-10-08--mapped-result-axis-semantic-foundation)

### MSW-9

**Broader ANN admission and tuned candidates**

- Owner: [ANN_CALCULUS_DESIGN_SPIKE.md](ANN_CALCULUS_DESIGN_SPIKE.md)
- Gate: Terminal square now has explicit analytic error amplification and intermediate-overflow refusal alongside ReLU/absolute-value consumers. Proved row-private mutable temporaries now admit the 64x8 square workload within the unchanged 4096-byte limit; constants and nested generations stay fully allocated. Extend measured nonlinear families and physical schedules. Admit only exact-artifact scoped candidates passing the measured lower-bound gate; retain incumbents when evidence is insufficient.
- Depends on: [FRONTEND-IR-MEDIUM-1](#frontend-ir-medium-1): recipe/native identity for the candidate; broader raising is independent.
- Start: host-free
- Latest: [2026-09-08 — Broader ANN workload evidence](INTEGRATED_COMPILER_LOG.md#2026-09-08--broader-ann-workload-evidence)

### W5.2

**Measured schedule selection**

- Owner: [OPTIMIZING_COMPILER_PLAN.md](OPTIMIZING_COMPILER_PLAN.md)
- Gate: Connect remaining producers and target calibration to measured schedule selection; preserve inferred dependencies and select only with eligible exact-device evidence. Every arbiter candidate now carries a code identity (sync `AUTOTUNE-EMITTED-IDENTITY-2026-09-27`); the sm_120 registry rows were re-recorded on Super-Bear. Follow-ups (sync `SM120-AUTOTUNE-FOLLOWUPS-2026-09-27`): the stale-error rule now holds in emitted CUDA, the scalar lanes have device timers (no registry row races an untimed candidate), and `_infer_dims` has a gated rule; after the re-record 13 rows are served. Launch integrity (sync `AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`): every emitted CUDA/HIP launch is checked through the slot, the missing route resources were captured (91 registry rows selector-eligible), the shipped GEMM is byte-reproducible, and the non-registry rows carry route identities; after the re-record 20 sm_120 registry rows are served. The remaining serving blocker is unseparated device-event verdicts.
- Depends on: [EVIDENCE-PACKET-1](#evidence-packet-1)
- Start: host-free
- Latest: [2026-09-27 — Autotune launch integrity](INTEGRATED_COMPILER_LOG.md#2026-09-27--autotune-launch-integrity)

### W5.5

**Consumer-driven canonicalization**

- Owner: [OPTIMIZING_COMPILER_PLAN.md](OPTIMIZING_COMPILER_PLAN.md)
- Gate: Checked permutation composition, matrix-transpose flag folding, attribute-bearing cast retention and fusion guards now share legality across generic/custom canonicalization and direct transpose lowering. Extend live consumer coverage and measure candidate effects before promotion; equality saturation still requires a demonstrated ordering problem.
- Depends on: —
- Start: host-free
- Latest: [2026-09-09 — Four canonicalization legality improvements](INTEGRATED_COMPILER_LOG.md#2026-09-09--four-canonicalization-legality-improvements)

## F4

### W4-PRODUCT-1

**Source CFG and effect-aware recovery**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: A bounded native C++ host allocator/collector now exposes generation-checked handles, roots, cause/context cycles and reusable payload holes through a copy-out C ABI. Serialized static source exception tables now compile through MLIR/LLVM to native allocation/root/edge calls and can be enabled on the native CPU source-state exception path. Fresh-heap construction cleans up on failure. Runtime numeric tensor payloads now enter native allocation through pointer/size operands, with a one-MiB bound, owned copies, runtime shape changes and cleanup after exhaustion. The host allocation is at synchronous completion. A bounded GPU producer now transactionally copies runtime f32 site payloads into preallocated global frames with caller-supplied generation/offset/length records and exhaustion status; CUDA/HIP execution and copy-back decoding are proven. A separate bounded nonmoving GPU slot pool now generates allocation and stop-the-world mark/sweep kernels, reuses payload slots, follows up to 32 generation-checked edges per live node and collects unrooted cycles on CUDA/HIP. Opaque byte records and a resident owner now order root/edge updates, allocation and collection after closed reader leases across streams. Completed event records are reaped; failed event recording retains retryable ownership until explicit completion. Private snapshot marking now runs independently of active-graph mutations; final seeded remark/sweep remains exclusive. Bounded sweep batches now logically retire the entire unreachable cohort before range reclamation, preventing cross-batch dead-cycle edges from becoming invalid. Allocation reuses only reclaimed slots; new roots cannot resurrect retired generations. CUDA/HIP validate mutation between batches. Owned immutable pool snapshots now permit snapshot reader scopes during live-pool sweeping without racing current-storage reads; copies are writer-ordered and remain owned through reader completion. This is snapshot isolation, not unrestricted same-storage concurrent sweeping. Quiescent exact builtin containers and plain instance dictionaries are discovered without user hooks, preserving cycles and aliases. General-key dictionaries now retain key/value references; exact sets, frozensets and bytearrays are supported. Builtin subclasses with native payload/descriptors require an explicit extractor instead of silently dropping their hidden state. Opt-in fully declared slot hierarchies now use native member descriptors without attribute/property hooks; mixed instance dictionaries and undeclared opaque extension storage refuse. Explicit exact-type ExtensionLayout extractors now copy opaque payload bytes and strong referents under a caller-enforced quiescence contract, with schema/type identity and existing budgets; no global registry or automatic extension traversal is implied. Before concurrent sweeping, implement and verify a mutation barrier, generation-aware retirement epoch and reader-protected reclamation; snapshot marking alone does not authorize freeing against live mutation. Automatic throw-site fusion, concurrent sweeping, variable-size payload storage and arbitrary extension/object semantics remain open. The existing arena also supports explicit live-node rooting and cause/context edge updates. Completion publication bypasses custom exception field hooks; source records remain diagnostic, not CPython frames. Exception arena ABI leases now exclude allocation/collection until readers complete; partial collections reuse payload holes without moving live roots. Opt-in source recovery records native single-carry while and bounded multi-variable break/continue expansion, merging each iteration before the next. Assertions remain ordered native effects; unmodelled calls refuse even on dead paths. Bounded nested/tuple loop returns and statically handled builtin exceptions now execute natively; explicit CPU state slots preserve exact input aliases and non-overlapping strided views through serialized SSA state and post-completion copyback. An opt-in native CPU JIT owns four shape/alias specializations; explicit result specs transport builtin exception classes with preceding writes. Declared plain-instance/dict/SimpleNamespace tensor fields and read-only overlapping snapshots now execute, including disjoint mutable state in the same invocation; static exception args, inherited/tuple handlers and bare re-raise are transported. Source JIT exposes explicit native paired VJP of functional results and declared next-state outputs, including projected object fields, without copyback; exact aliases accumulate their adjoint at the canonical root; native slice adjoints now accumulate mapped and overlapping reads at the containing root and mask overwritten destination gradients; exception objects remain nondifferentiable; checked source VJP differentiates successful numeric paths. Rank-one positive strides and bounded injective negative/multidimensional views share an explicit contiguous containing-input SSA root; local rank-preserving slices are captured, with general mapped code generation bounded to 256 elements; positive rectangular maps use compact native slices. Native runtime-shaped slice products reuse artifacts across shapes with guarded cotangent dimensions. Source slicing captures signed int64 bounds/steps on static ranked roots, including negative and nested runtime views, clipping and empty results. Python integer arguments reuse the same CPU JIT specialization; compiler-owned capacity/shape sidecars allocate multiple multidimensional outputs up to the 1024-element host capacity. Index-only runtime gather adjoints now use shape-guarded accumulating scatters; Python index protocols and runtime-shaped original roots remain open. Single-element f32 exception-value payloads cross loop/finally completion as typed outputs; per-site/generation SSA slots retain distinct dynamic cause/context values in bounded expanded loops, and CPU/GPU VJP gates backward on successful forward completion; bounded loop-carried caught references retain their original generation payload; static caught identities, named re-raise, explicit causes and implicit contexts are reconstructed at the host boundary with native source-location notes. CUDA SM120 and ROCm gfx1151 execute mapped forward/backward and checked synchronous/asynchronous exception completion; failed frames expose no result. Extend custom object access, ownerless/noncontiguous writable views, dynamic strings, unbounded or object-carried loop exception identity and real CPython frame/traceback semantics, changing loop state and automatic effectful AD; arbitrary CFG is not closed.
- Next architecture gate: [gated owner and isolated recovery](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#gated-metadata-owner-and-isolated-recovery-2026-09-11): every admitted metadata operation in the opt-in gated owner has replay/device proof; live metadata and legacy import/snapshot paths refuse. Process-owned heap recovery requires confirmed death. Since 2026-09-15 ([probed admission and replacement](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#probed-admission-and-health-checked-replacement-2026-09-15)) a worker is admitted only after its in-process device probe verifies the admitted producers, and a replacement is admitted only after confirmed predecessor death plus its own probe (RTX 5070 and gfx1151). Next: migrate legacy snapshot/import callers to a gated producer and validate actual driver-failure behavior (the recorded fault is an injected stall). Epoch ordering remains; no measured overlap or promotion.
- Depends on: —
- Start: host-free
- AD gate: [native composed HVP execution](AUTODIFF_EXECUTION_PLAN.md#native-composed-hvp-execution-2026-09-14) adds static CPU saved-product tangents and bounded gfx1201 HVP execution; arbitrary CFG/effects, dynamic results, higher orders and general GPU binding remain open. The energy-as-typed-program acceptance's first clause is met on the CPU lane (2026-09-16, `EBM-NATIVE-QUADRATIC-2026-09-16`): the quadratic energy is a Graph IR function, its gradient is the compiler's, the K-step Langevin loop with on-device Philox noise compiles as one function and is bit-exact with the declared policy on the M1 Max, Zen 5 and Zen 2; the device clause followed the same day (`EBM-NATIVE-GPU-2026-09-16`): the row-program emitter lowers the compiler-derived gradient inside one cooperative kernel (one block per row, lanes per feature, the K-step loop and Philox in registers, ordered reductions), bit-exact on gfx1151, gfx1201 and sm_120 with one launch per loop and one `tessera-opt` invocation for the whole chain; the nonlinear energies and the sphere integrator followed the same day (`EBM-NONLINEAR-MANIFOLD-2026-09-16`): three energies run through the one integrator with no host VJP left (softplus gained its native adjoint) and `manifold = "sphere"` lowers natively with a per-row status word for its two singularities, on the CPU lane and all three GPUs; the bivector integrator and the overhead measurement closed the same day (`EBM-BIVECTOR-OVERHEAD-2026-09-16`): the grade projection rides the Clifford dialect's own op, and the measurement shows the native route flat in K (one launch per loop) against about 1.8 ms per step for the Python-emitted lane — a dispatch result that promotes nothing, since kernel-time attribution needs a box no WSL2 ROCm host can provide — [the EBM native loop architecture](../domain/EBM_NATIVE_LOOP_ARCHITECTURE.md) records what shipped and why the scoped Tile contract was not needed; four of that stream's remaining gaps closed the same day (`EBM-GA-GAPCLOSE-2026-09-16`): Clifford `exp`/`log` and `rotor_from_axis` lower to their closed forms on Cl(3, 0) so rotor sampling happens on the group, ragged batches take their loop bound from `tensor.dim` with the operands' agreement asserted rather than assumed, an annealing schedule runs as one cooperative kernel with the temperature carried in registers (a ratio of 1.0 reproducing the constant chain bit for bit), and an opaque energy adjoint is refused by name. The row-program emitter now admits only `math.*` ops whose accuracy was measured on the owning device — the sweep that established those bounds is what found `math.tanh` shipping a kernel with no body at all on gfx1151, so the packager refuses an image whose kernel stores nothing. Still open: the Clifford field ops, the 1024-feature ceiling, Apple, and promotion.
- Latest: [2026-09-16 — the math a kernel is allowed to contain, and four closed domain gaps](INTEGRATED_COMPILER_LOG.md#2026-09-16--the-math-a-kernel-is-allowed-to-contain-and-four-closed-domain-gaps)

### AD-RESIDUAL-EVAL-1

- Current slice (2026-09-29, sync `COMPILER-EVIDENCE-FRAGMENT-RESIDUAL-2026-09-29`): the coupled public residual has a nonzero primal, saved-input mutation isolation and repeated analytic VJP proof on gfx1151 and sm_120. Dynamic layouts, aliases and asynchronous ownership remain open. [Evidence](../../../benchmarks/baselines/compiler_evidence_fragment_residual_20260929/README.md).

**Logical-shape ABI and persistent checkpoint execution**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Paired source VJP retirement resumes only children that have not submitted frees after a dependency failure, on the original stream. Native AD products now bind bounded dynamic GPU inputs and scalar through rank-four floating/i8/i64 results using checked capacity/shape sidecars. A synchronous checked source VJP retains device snapshots and the matching forward residuals; exception completion is checked before backward and completion metadata receives zero seeds. Same-stream asynchronous snapshots and forward-gated backward submission now expose derivatives only after successful completion. Scoped source VJP now supports reader-aware retire/poll without a context wait on the healthy path. Explicit close can still synchronize; failed-free quarantine, deferred module unload, cross-queue writer ownership and broader product bindings remain open. Exact dominating SSA product guards now tighten joint temporary capacities. Extend relational/aliased volume proofs, saved heterogeneous products, general layout envelopes and automatic frontend wiring; exported shapes do not establish arbitrary Python CFG capture.
- Depends on: [W4-PRODUCT-1](#w4-product-1): existing bounded product carrier; arbitrary source CFG closure is not a prerequisite.
- Start: host-free
- Current slice (2026-10-01): public affine-free fp16/bf16 LayerNorm now
  feeds a resident scheduled SM120 matmul through Graph -> Schedule -> Tile.
  Exact RTX 5070 tests passed both dtypes; the fp16 256x256x256 packet records
  independent parity, allocation ownership, zero spills, and separate
  producer/consumer event times. Generic legacy tensor-valued Tile producers
  remain open.
- Current increment (2026-10-05, sync NVIDIA-VALUE-JVP-2026-10-05): isolated native AD export now carries value-only attention through the paired saved-LSE Schedule/Tile package. The linear materializer removes irrelevant V/O reads and uses one 512-byte shared reduction. Public V-only JVP passes RTX 5070 finite-difference and finite/Inf/NaN primal-V linearity checks. General composed/dynamic/bias/dropout/higher AD and sibling physical consumers remain open. [Evidence](../../../benchmarks/baselines/nvidia_value_only_jvp_20261005/README.md).
- Current increment (2026-10-05, sync NVIDIA-VJP-ACTIVITY-2026-10-05): public isolated reverse attention now derives requested gradient activity in native paired AD, seals it through Schedule and removes inactive Target arithmetic while zero-filling complete ABI outputs. RTX 5070 numerical/performance and matching gfx1201 shared regression evidence are retained; compact output allocation and general AD remain open. [Evidence](../../../benchmarks/baselines/nvidia_vjp_activity_20261005/README.md).
- Current increment (2026-10-06, sync NVIDIA-COMPACT-GRADIENTS-2026-10-06): native paired AD/Schedule/Tile now preserves complete logical results while exporting only requested physical gradient buffers. Checked static SM120 host/resident packages, private capture and repeated gradients are proved on RTX 5070. Preserved logical ranges with explicit 64/128-thread geometry address the measured packed-range loss; no universal policy/default promotion follows. Matching gfx1201 shared compiler/norm regressions pass, without HIP compact-gradient execution claims. General composed/dynamic/higher AD and sibling physical consumers remain open. [Evidence](../../../benchmarks/baselines/nvidia_compact_gradients_20261005/README.md).
- Current increment (2026-10-06, sync NVIDIA-JVP-ARGUMENT-ORDER-2026-10-06): native forward export now verifies distinct frontend Q/K/V permutations; automatic private capture and tangent requests follow verified physical roles and activity. All six orders have owning RTX 5070 finite-difference/mutation/repeated-direction proof, with unchanged native images within each semantic envelope. Shared gfx1201 compiler/norm regressions pass without HIP tangent claims. General composed/dynamic/bias/higher AD remains open. [Evidence](../../../benchmarks/baselines/nvidia_jvp_argument_order_20261006/README.md).
- Current increment (2026-10-06, sync NVIDIA-JVP-PORTABLE-2026-10-06): pinned canonical program JSON retains native images, sizing library, roles and manifests; capture validates them before CUDA allocation. Seventy-two restored-program RTX 5070 cases and three compiler-forbidden fresh-process replays pass. Public native-JVP family/runtime integration, unused reverse-image retirement, general AD and sibling physical consumers remain open. [Evidence](../../../benchmarks/baselines/nvidia_jvp_portable_20261006/README.md).
- Latest: [2026-10-06 — Native score-bias attention JVP](INTEGRATED_COMPILER_LOG.md#2026-10-06--native-score-bias-attention-jvp)

### W2.4a

**Generation ownership and scoped readers**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Late worker death can now reconcile a failed recovery ticket without repeating termination, releasing its admission slot and owner exactly once; no device-health inference follows. Isolated ANN admission now executes independent numerical probes in the fresh worker before readiness; replacement requires confirmed predecessor death. Multi-stream external readers now include checked dynamic public frames and paired source-VJP products, and record all declared completion edges, and eventless dependencies refuse implicit host synchronization. Actual wedged-driver recovery remains unproven. Bounded asynchronous isolation teardown now retains failed owners and slots, and ANN/module owners finalize only after confirmed process death. CUDA/HIP stopped-worker replacement is independently remeasured; actual driver-hang recovery and device-wide health admission remain open; bounded workload probes are implemented. Opt-in native ANN workers now own their CUDA/HIP context and return checked private host outputs; uncertain requests quarantine the worker until confirmed process death. Stopped-worker teardown/replacement is measured separately from actual driver-hang recovery. Scoped runtime-shaped public frames and asynchronous static capture now order frees after declared readers. Module unload can run off-thread with bounded admission and non-waiting polls; a stalled/failed driver retains its owner. One through eight serialized incoming statuses now support bounded fan-in with scoped readers for each prerequisite; all 256 eight-status combinations have independent SM120/gfx1151 truth-table proof. Source state can produce immutable next-state GPU generations; exclusive synchronous owned-state copyback now reuses a private allocation with independent SM120/gfx1151 proof and blocks active scoped readers. Single-stream submit/poll now gates asynchronous copyback and excludes readers until completion; failure poisons the owner and retains pending storage. External borrowed-pointer mutation and concurrent multi-writer updates remain open. Extend heterogeneous dynamic persistent capture, unbounded/heterogeneous effect joins and external-reader adoption. Driver unload itself is not cancellable or latency-bounded, and unrestricted views retain synchronous close.
- Next architecture gate: [gated owner and isolated recovery](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#gated-metadata-owner-and-isolated-recovery-2026-09-11): every admitted metadata operation in the opt-in gated owner has replay/device proof; live metadata and legacy import/snapshot paths refuse. Process-owned heap recovery requires confirmed death. Since 2026-09-15 ([probed admission and replacement](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#probed-admission-and-health-checked-replacement-2026-09-15)) a worker is admitted only after its in-process device probe verifies the admitted producers, and a replacement is admitted only after confirmed predecessor death plus its own probe (RTX 5070 and gfx1151). Next: migrate legacy snapshot/import callers to a gated producer and validate actual driver-failure behavior (the recorded fault is an injected stall). Epoch ordering remains; no measured overlap or promotion.
- Current increment: Snapshot retirement now closes reader admission, including previously created leases, and polls every completion edge without a wait or free. Eventless readers retain storage for explicit recovery. Allocation teardown remains synchronous; next extend allocator/module retirement only with separate completion and uncertain-driver ownership proof.
- Depends on: [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1): the selected product's residual and ownership ABI; independent static slices may proceed.
- Start: host-free
- Latest: [2026-09-10 — Dynamic reader fanout and row-private ANN](INTEGRATED_COMPILER_LOG.md#2026-09-10--dynamic-reader-fanout-and-row-private-ann)

### NUMPOL-CARRIER-1

**Numerical policy and analytic budget consumers**

- Owner: [FUNCTIONAL_ANALYSIS_TSOL_PLAN.md](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md)
- Gate: gfx1201 dense WMMA has operand/accumulator-complete fragment proof and exact-device scheduled matmul packages for `fp16`, `bf16`, E4M3, E5M2, signed `int8`, and packed signed `int4`. The exact `GFX_1201` ISA rows are now `ready` for those dense inputs while the otherwise identical `GFX_1200` rows remain `artifact_only`; no family-level RDNA4 proof transfer is allowed. The public execution rows use gfx1201-only executors that check both selected HIP-device architecture and compiler chip before dispatch. Native BF16 rounding still differs from f32-rounded-once and requires explicit error-budget integration. Public sparse SWMMAC admission, scaled MX/FP4 schedules, general numerical-policy consumption, and exact gfx1200 device proof remain open. `scripts/record_dtype_codegen_inventory.py` now joins each proved gfx1201 input to the machine-readable operation proof rather than presenting ISA declarations as execution evidence; use it alongside the operator dtype-flow report. The 2026-09-11 packet verifies 24 scalar/vector basic-arithmetic rows independently on CUDA and HIP, including signedness-sensitive division; the four previously failing FP8 rows per device have byte-conversion legalization and exhaustive input-pair proof. Bool logic and bounded complex component probes pass on both devices; ten NVIDIA packed/scaled layout probes pass. Apple native f32 multiply/divide flush three expected subnormal results, so strict gradual-underflow admission remains open. Extend remaining packed/scaled, Apple storage, general complex, numerical-policy and matrix consumers before claiming closure; TF32 remains a math mode. See [dtype follow-through](../../../benchmarks/baselines/dtype_followthrough_20260911/README.md) and the [gfx1201 application checklist](AMD_KERNEL_COMPILER_SURVEY.md#gfx1201-datatype-application-checklist).
- Current increment: The Apple arena now consumes explicit gradual/FTZ policy. Integer-significand f32 add/sub/mul/div with one ties-to-even rounding step passes 133,376 input pairs per operation on M1 Max; explicit input/output FTZ around the same arithmetic also passes. Legacy unspecified arithmetic retains its historical boundary. Other floating operations, vectors/dtypes and optimized performance remain open. CUDA/HIP generic storage refuses this unconsumed policy. See the [policy contract](../../spec/APPLE_ARENA_NUMERICAL_POLICY.md).
- Depends on: —
- Start: host-free
- Latest: [2026-09-13 — RDNA4 WMMA operand and accumulator audit](INTEGRATED_COMPILER_LOG.md#2026-09-13--rdna4-wmma-operand-and-accumulator-audit)

### LAYOUT-ALG-1

**Remaining physical layout envelopes**

- Owner: [CORE_SUBSTRATE_VIEW.md](CORE_SUBSTRATE_VIEW.md)
- Gate: Preserve proved static/dynamic layout consumers; extend only unresolved nonseparable tuple/layout envelopes with capacity, alias and lifetime proof. Matrix acceleration additionally needs operand packing, signedness, accumulator type, fragment shape, scale layout and target instruction witnesses; Zen 5 VNNI/vector GEMM does not establish AMX support.
- Depends on: —
- Start: host-free
- Latest: [2026-09-06 — Descriptor projection and seven-program continuation](INTEGRATED_COMPILER_LOG.md#2026-09-06--descriptor-projection-and-seven-program-continuation)

### AD-SOLVER-IFT-1

**Native implicit-solver consumers**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Extend residual/predicate/solver envelopes and Apple/NVIDIA consumers through compiler-owned children; require convergence/conditioning certificates and owning-device proof.
- Current increment: The first EBM energy differentiates through the compiler (2026-09-16, `EBM-NATIVE-QUADRATIC-2026-09-16`): `tessera.sub` gained its adjoint and the sum-reduce adjoint's `unsqueeze`/`broadcast` gained linalg lowerings, so the paired autodiff pass now carries a quadratic energy to a native gradient the EBM Langevin lowering consumes; implicit differentiation and OT primitives are untouched.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Latest: [2026-09-06 — Functional-analysis contracts — consolidated ownership](INTEGRATED_COMPILER_LOG.md#2026-09-06--functional-analysis-contracts--consolidated-ownership)

### AD-HIGHER-1

**Broader AD, batching, sparsity and jets**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Extend composed attention and broader native AD families, higher-order products, real batching, structural sparse derivatives and native jets through the existing AD owners; retain oracle, structural-zero and conditioning gates.
- Depends on: [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1): the selected product's residual and ownership ABI; independent static slices may proceed.
- Start: host-free
- Latest: [2026-09-06 — domain and autodiff documentation consolidation](INTEGRATED_COMPILER_LOG.md#2026-09-06--domain-and-autodiff-documentation-consolidation)

### DIST-NATIVE-1

**Real multi-rank transport**

- Owner: [SCHEDULE_OBJECT_DESIGN.md](SCHEDULE_OBJECT_DESIGN.md)
- Gate: Extend bounded MPI proof beyond two ranks and proper-subgroup participants; add native NCCL/RCCL and other transports with process ownership and real multi-rank packets.
- Depends on: [EVIDENCE-PACKET-1](#evidence-packet-1)
- Start: device
- Latest: [2026-09-07 — Status, native GELU and recipe instantiation](INTEGRATED_COMPILER_LOG.md#2026-09-07--status-native-gelu-and-recipe-instantiation)

## F5

### W5.2f

**Shared tiled-SSD compiler family**

- Owner: [SEQUENCE_MIXER_ENGINEERING_PLAN.md](SEQUENCE_MIXER_ENGINEERING_PLAN.md)
- Gate: The registered internal `schedule.ssd` now owns a static f32 scalar-decay recurrence, immutable initial/final carry and chunk-end checkpoints. Schedule-to-Tile lowers it to structured tensor loops, numerically executed through the native CPU JIT including partial chunks. Replay-bound serial CUDA SM120 and ROCm gfx1151 packages now execute three chunk sizes with output/carry/checkpoint and immutable-input checks. Opt-in cooperative CUDA/HIP kernels now assign each head/value column to a block, keep one state element per active lane, and reduce in state-index order through shared memory and barriers. Both owning devices pass correctness. Nine independent paired process runs per backend now retain exact artifact identities and resident event-window confidence bounds; native clock calibration still prevents promotion; explicit measured binding is now wired. Checkpoint VJP differentiates all five inputs and all three result cotangents, recomputes within chunks, and now executes on CUDA/HIP using checkpoints from the cooperative forward. A scoped native CPU owner integrates Y with automatic host-tape grad and private checkpoints. Exact-artifact measured binding now recomputes policy and retains incumbents without native clock calibration. ResidentSSDProgram now automatically pairs GPU forward/VJP packages, snapshots inputs device-to-device, owns checkpoints and all five gradients, and exposes a synchronous first-order value_and_grad API with scoped lifetime. Asynchronous SSD gradients now compose through projected scoped generations on distinct CUDA/HIP streams and retire after external readers. Public vjp now dispatches explicitly owned resident SSD programs. Capture, VJP and whole-frame retirement use stream events and async allocation/free; scoped forward and derivative readers delay reclamation. Explicit close remains synchronous; program retire_async/poll_close now orders all frame/reader retirements before bounded-admission off-thread module unload. Partial queueing is retryable and new capture stays closed. CUDA/HIP both validate two public-VJP frames without context synchronization; driver unload itself is not cancellable or latency-bounded. SSD packages now use a separate checked 64 MiB per-buffer bound; both GPUs validate a larger cooperative workload. ResidentSSDTrace now traces connected acyclic SSD compositions before submitting work, automatically captures forward/checkpoint frames and runs reverse calls with scoped readers. Fan-out and shared public inputs accumulate cotangents through replay-validated native f32 addition kernels. A three-call DAG and all five shared public input gradients pass an independent float64 finite-difference oracle on CUDA/HIP; addition allocations and modules participate in frame/program retirement. This is host orchestration of replay-bound packages, not a fused or canonical whole-program IR. Serializing composition into the compiler foundation, unused-input zeros, same-call input alias admission, effectful/data-dependent traces and additional operation families remain open. Next: arbitrary traced public tape integration, broader owner adoption and uncertain-unload isolation recovery, broader tiling/reduction tuning, public mixer/frontend integration, broader mutation/alias lineage, ReplaySSM comparison and selector-grade promotion.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Latest: [2026-09-11 — Resident DAG accumulation and snapshot readers](INTEGRATED_COMPILER_LOG.md#2026-09-11--resident-dag-accumulation-and-snapshot-readers)

### TSOL-POLICY-PHYS-1

**Spectral policy breadth**

- Owner: [TARGET_IR_REVIEW.md](TARGET_IR_REVIEW.md)
- Gate: Reconcile current spectral policy envelopes, then close remaining target-specific strides/full-spectrum/window, broadcasting and streaming execution gaps with independent adjoint proof.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: device
- Latest: [2026-09-27 — Spectral image survives a stale HIP error; streaming STFT names the chip that ran](INTEGRATED_COMPILER_LOG.md#2026-09-27--spectral-image-survives-a-stale-hip-error-streaming-stft-names-the-chip-that-ran)

### TSOL-SCALE-1

**ND and large-transform execution**

- Owner: [TARGET_IR_REVIEW.md](TARGET_IR_REVIEW.md)
- Gate: Add missing ND/large/prime transforms with explicit algorithm, plan/twiddle/workspace identity and per-target retain/promote/reject comparisons.
- Depends on: [TSOL-POLICY-PHYS-1](#tsol-policy-phys-1)
- Start: device
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--math-audit--foundation-reconciliation)

### TSOL-SHARD-1

**TSOL placement and sharding**

- Owner: [SCHEDULE_OBJECT_DESIGN.md](SCHEDULE_OBJECT_DESIGN.md)
- Gate: Reconcile spectral/solver/sparse placement rules and reshard consumers; separate mock-mesh correctness from actual multi-rank execution.
- Depends on: [DIST-NATIVE-1](#dist-native-1)
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--math-audit--foundation-reconciliation)

### TSOL-PHYS-TAIL-1

**High-value physical math families**

- Owner: [PDE_STENCIL_CAPABILITY_PLAN.md](PDE_STENCIL_CAPABILITY_PLAN.md)
- Gate: Advance high-use solver, sparse/segment and layout families from target-specific gaps; preserve PDE discretization/halo and coalition-lattice obligations.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--math-audit--foundation-reconciliation)

### W6.4

**Native batched finite algebra**

- Owner: [../domain/GA_EBM_ARCHITECTURE_REVIEW.md](../domain/GA_EBM_ARCHITECTURE_REVIEW.md)
- Gate: Carry W3.6 batching and grade-aware algebra through native consumers; packed grades/PGA/CGA/exp/log require independent semantic, AD and device proof.
- Current increment: The batched geometric product is native (2026-09-16): `ExpandProductTable` lowers any static `[..., dim]` rank to an scf.for nest over the compile-time table with grade pruning, and `libtessera_jit` runs GradeFusion + ExpandProductTable so a `tessera_clifford.geo_product` executes through MLIR/LLVM (execution-matrix row `cpu` / `cpu_clifford_llvm_jit`; parity on M1 Max, Zen 5 and Zen 2). Same day, the whole product family (wedge, left contraction, inner, norm, reverse/involution/conjugate, Hodge star, grade projection, rotor sandwich) lowers through the same table and executes behind the JIT (`cpu` / `cpu_clifford_llvm_jit`, nine ops × three shapes against the GA reference on M1 Max, Zen 5, Zen 2). The GPU package route is closed for the family the same day: the kernel skeleton carries a rank-1 Clifford op, `ts-clifford-opt` expands it through the same lowering, the arena pipeline folds it to scalar device code (64 → 24 products under a grade-2 restriction, visible in the IR) and the native storage package launches it — ten ops × three shapes match the reference on gfx1151, gfx1201 and sm_120 (`rocm` / `rocm_clifford_native_compiled`, `nvidia_sm120` / `nvidia_clifford_native_compiled`). Open: exp/log and the field ops, ragged batches, and the acceptance's separate overhead/traffic/kernel-time measurements (the device lanes are still the Python-emitted kernels until measured against this route). **Corrected 2026-09-25 against [W4-PRODUCT-1](#w4-product-1)'s record of the same day (`EBM-GA-GAPCLOSE-2026-09-16`, `EBM-BIVECTOR-OVERHEAD-2026-09-16`):** Clifford `exp`/`log` (and `rotor_from_axis`) now lower to closed forms **on Cl(3, 0) only** — other signatures still need the gate's independent semantic/AD/device proof; ragged batches are closed (loop bound from `tensor.dim`); the dispatch-overhead measurement is recorded and promotes nothing, while kernel-time attribution stays blocked on a non-WSL2 host. Still open: the field ops (`ext_deriv`, `codiff`, `vec_deriv`, `integral`), exp/log beyond Cl(3, 0), the 1024-feature row-program ceiling, Apple, and measuring the native route against the Python-emitted `x86_`/`rocm_clifford` kernels.
- Depends on: [AD-HIGHER-1](#ad-higher-1)
- Start: host-free
- Latest: [2026-09-16 — the Clifford family reaches ROCm and sm_120 through the arena pipeline](INTEGRATED_COMPILER_LOG.md#2026-09-16--the-clifford-family-reaches-rocm-and-sm120-through-the-arena-pipeline)

### RIEMANNIAN-OT

**Geometric primitives and OT validation**

- Owner: [RIEMANNIAN_OT_PLAN.md](RIEMANNIAN_OT_PLAN.md)
- Gate: Retain missing geometric primitives and constrained/KKT hypotheses under shared solver/AD owners; validate the OT workload without creating another solver stack.
- Depends on: [AD-SOLVER-IFT-1](#ad-solver-ift-1)
- Start: host-free
- Latest: [2026-09-06 — Functional-analysis contracts — consolidated ownership](INTEGRATED_COMPILER_LOG.md#2026-09-06--functional-analysis-contracts--consolidated-ownership)

## Reconciliation before archival

The move preserves every old queue/wave paragraph and historical “Next” list.
These are the disposition rules applied to the review inventory; an uncertain
remainder stays with its owner and is not silently declared complete.

| Earlier obligation | Current destination and reconciliation |
|---|---|
| W1.1; W2.4; W2.4a NVIDIA producers/Target/barriers | W1.1 and W2.4a retain architecture closure and wrapper/ownership review. Shared legality and recent bounded reader proofs are not all-route closure. |
| W3.1, W3.2, W3.4, W3.7 | E2E-REAL-6 owns remaining frontend, JitFn and forward/direct constructor work. W3.2's old estimate is retired; estimate each census-selected route. |
| W3.3 | Retains a consumed-semantics inventory gate before a dialect split. |
| W3.5; OT geometric primitives | AD-SOLVER-IFT-1 and RIEMANNIAN-OT retain broader consumers/hypotheses; shared solver slices are not universal device support. |
| W3.6; W6.4 | W6.4 retains batched grade-aware algebra and actual fold-marker consumers. |
| W4.1–W4.3 | W4-PRODUCT-1 and AD-RESIDUAL-EVAL-1 retain frontend CFG/effects, exported logical shapes and checked asynchronous products. Recent native function CFG/status work is bounded. |
| W5.1 | AD-RESIDUAL-EVAL-1 retains selected production checkpoint policies and complete exact-device comparisons; bounded native SAVE/HYBRID/recompute execution already exists. |
| W5.2; W5.2e; W5.2f; W5.5 | W5.2 retains producer/calibration/selector work; W5.2f owns tiled SSD; W5.5 owns consumer-justified canonicalization. Existing inferred DAGs and prune-only search are not promotion proof. |
| W5.4 | DIST-NATIVE-1 retains transport beyond the bounded MPI slice; TSOL-SHARD-1 retains spectral/solver placement. |
| W6.1–W6.3 | AD-HIGHER-1 routes to the active AD plan's higher-order, batching, sparse and native-jet obligations; do not rebuild landed reference algebra. |
| Ordered queue 8–11 | TSOL-POLICY-PHYS-1, TSOL-SCALE-1, TSOL-SHARD-1 and TSOL-PHYS-TAIL-1 preserve policy, scale, placement and physical-tail gates. |
| Ordered queue 12–14 | TPROF-NATIVE-1, EVIDENCE-PACKET-1 and COMPILER-DEVEX-1 retain clocks, packet consumers and installed/assertions-enabled tooling. |
| Six non-wave scoped notes | PDE, math-source, AttnRes, reference-tier, layout and frontend retain their scoped owners in the routing index. They are workload contracts, not new waves. |
| Later numerical and ANN “Next” lists | NUMPOL-CARRIER-1 and MSW-9 retain broader analytic consumers and tuned native admission; recent bounded admission/row schedules are not performance promotion. |

## Maintenance and proof rules

- A substantive delivery PR updates its active task record and appends a log
  entry naming that ID. Completion removes the record only after the routing
  index points to a scoped owner, successor or archived disposition.
- CI checks structured IDs, destinations, dependencies, latest-log links and
  the derived start view. A merge-base check requires an owner-record change
  when a new log entry is added; it does not infer ownership from arbitrary code.
- Keep five proof questions independent: parse/verify; lowering consumes the
  incoming IR; runtime binds that artifact; owning device executes correctly;
  performance evidence is eligible. Use existing proof vocabulary and F0-derived
  inventories, not hand-maintained five-column status copies.
- Preserve ABI/layout/policy/provenance through serialized IR and replay;
  migrated routes need differential tests and exact-target execution before
  retiring their old constructor. Presence of bytes or hashes is insufficient.
- Shared changes assess all four [backend queues](../backend/README.md). Runtime
  timeouts must retain in-flight resources and reject unsafe reuse. Reference
  execution and WSL regression timing never silently become native promotion.
- Follow [AGENTS.md](../../../AGENTS.md), [governance and fleet facts](../../../CLAUDE.md),
  and the [compiler test plan](../../../tests/COMPILER_TEST_PLAN.md). Do not copy
  OS versions, command pins, budgets or governance decisions into this queue.
- Scoped acceptance remains in its owner document; the global queue chooses
  sequencing, not duplicate domain designs. The log has no current next-order.
- Theme audits route to this queue and are gated on it: the
  [domain audit](../domain/DOMAIN_AUDIT.md) may cite only IDs in the routing
  index, must name the destination beside any `successor`/`archive` ID, and
  reads status from the generated
  [domain proof ladder](../generated/domain_proof_ladder.md)
  (`tests/unit/test_domain_audit_routing.py`, `check_generated_docs.sh`).

## Routing index

Fixed columns are a navigation contract. `owner`, `successor` and `archive`
describe routing, not readiness. Historical mentions need not be active tasks.

| ID | Canonical destination | Relationship |
|---|---|---|
| AD-HIGHER-1 | [AD-HIGHER-1](#ad-higher-1) | owner |
| AD-RESIDUAL-EVAL-1 | [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1) | owner |
| AD-SOLVER-IFT-1 | [AD-SOLVER-IFT-1](#ad-solver-ift-1) | owner |
| COMPILER-DEVEX-1 | [COMPILER-DEVEX-1](#compiler-devex-1) | owner |
| DISPATCH-BREAKER | [DISPATCH-BREAKER](#dispatch-breaker) | owner |
| DIAG-PY-BACKLOG-1 | [DIAG-PY-BACKLOG-1](#diag-py-backlog-1) | owner |
| DIST-NATIVE-1 | [DIST-NATIVE-1](#dist-native-1) | owner |
| ROCM-MACRO-K-TILE-1 | [ROCM-MACRO-K-TILE-1](archive/ROCM_MEASURED_QUEUE_2026-09-28.md#rocm-macro-k-tile-1) | archive |
| GOV-ODS-CONSUMER-1 | [GOV-ODS-CONSUMER-1](#gov-ods-consumer-1) | owner |
| ROCM-FP8-BLOCKSCALE-1 | [ROCM-FP8-BLOCKSCALE-1](#rocm-fp8-blockscale-1) | owner |
| ROCM-MXFP4-W4A8-1 | [ROCM-MXFP4-W4A8-1](#rocm-mxfp4-w4a8-1) | owner |
| ROCM-NVFP4-INGEST-1 | [ROCM-NVFP4-INGEST-1](#rocm-nvfp4-ingest-1) | owner |
| ROCM-LDS-BANKPAD-1 | [ROCM-LDS-BANKPAD-1](archive/ROCM_MEASURED_QUEUE_2026-09-28.md#rocm-lds-bankpad-1) | archive |
| ROCM-LDS-STAGE-VECTOR-1 | [ROCM-LDS-STAGE-VECTOR-1](archive/ROCM_MEASURED_QUEUE_2026-09-28.md#rocm-lds-stage-vector-1) | archive |
| ROCM-SCHED-GROUP-1 | [ROCM-SCHED-GROUP-1](archive/ROCM_MEASURED_QUEUE_2026-09-28.md#rocm-sched-group-1) | archive |
| ROCM-SPLIT-K-1 | [ROCM-SPLIT-K-1](#rocm-split-k-1) | owner |
| E2E-REAL-6 | [E2E-REAL-6](#e2e-real-6) | owner |
| E2E-REAL-6F | [E2E-REAL-6F](#e2e-real-6f) | owner |
| EVIDENCE-PACKET-1 | [EVIDENCE-PACKET-1](#evidence-packet-1) | owner |
| FRONTEND-IR-MEDIUM-1 | [FRONTEND-IR-MEDIUM-1](#frontend-ir-medium-1) | owner |
| LAYOUT-ALG-1 | [LAYOUT-ALG-1](#layout-alg-1) | owner |
| MSW-9 | [MSW-9](#msw-9) | owner |
| NUMPOL-CARRIER-1 | [NUMPOL-CARRIER-1](#numpol-carrier-1) | owner |
| RIEMANNIAN-OT | [RIEMANNIAN-OT](#riemannian-ot) | owner |
| TPROF-NATIVE-1 | [TPROF-NATIVE-1](#tprof-native-1) | owner |
| TSOL-PHYS-TAIL-1 | [TSOL-PHYS-TAIL-1](#tsol-phys-tail-1) | owner |
| TSOL-POLICY-PHYS-1 | [TSOL-POLICY-PHYS-1](#tsol-policy-phys-1) | owner |
| TSOL-SCALE-1 | [TSOL-SCALE-1](#tsol-scale-1) | owner |
| TSOL-SHARD-1 | [TSOL-SHARD-1](#tsol-shard-1) | owner |
| W0.1 | [W0.1](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w01) | archive |
| W0.10 | [W0.10](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w010) | archive |
| W0.2 | [W0.2](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w02) | archive |
| W0.3 | [W0.3](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w03) | archive |
| W0.4 | [W0.4](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w04) | archive |
| W0.5 | [W0.5](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w05) | archive |
| W0.6 | [W0.6](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w06) | archive |
| W0.7 | [W0.7](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w07) | archive |
| W0.8 | [W0.8](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w08) | archive |
| W0.9 | [W0.9](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w09) | archive |
| W1.1 | [W1.1](#w11) | owner |
| W1.1b | [W1.1b](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w11b) | archive |
| W1.2 | [W1.2](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w12) | archive |
| W1.3 | [W1.3](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w13) | archive |
| W1.4 | [W1.4](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w14) | archive |
| W2.1 | [W2.1](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w21) | archive |
| W2.2 | [W2.2](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w22) | archive |
| W2.3 | [W2.3](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w23) | archive |
| W2.4 | [W2.4a](#w24a) | successor |
| W2.4a | [W2.4a](#w24a) | owner |
| W3.1 | [E2E-REAL-6](#e2e-real-6) | successor |
| W3.2 | [E2E-REAL-6](#e2e-real-6) | successor |
| W3.3 | [W3.3](#w33) | owner |
| W3.4 | [E2E-REAL-6](#e2e-real-6) | successor |
| W3.5 | [AD-SOLVER-IFT-1](#ad-solver-ift-1) | successor |
| W3.6 | [W6.4](#w64) | successor |
| W3.7 | [E2E-REAL-6](#e2e-real-6) | successor |
| W4-PRODUCT-1 | [W4-PRODUCT-1](#w4-product-1) | owner |
| W4.1 | [W4-PRODUCT-1](#w4-product-1) | successor |
| W4.2 | [FRONTEND-IR-MEDIUM-1](#frontend-ir-medium-1) | successor |
| W4.3 | [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1) | successor |
| W5.1 | [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1) | successor |
| W5.2 | [W5.2](#w52) | owner |
| W5.2a | [W5.2a](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w52a) | archive |
| W5.2b | [W5.2b](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w52b) | archive |
| W5.2c | [W5.2c](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w52c) | archive |
| W5.2d | [W5.2d](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w52d) | archive |
| W5.2e | [W5.2](#w52) | successor |
| W5.2e-PRODUCER-1 | [W5.2](#w52) | successor |
| W5.2f | [W5.2f](#w52f) | owner |
| W5.2g | [W5.2g](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w52g) | archive |
| W5.3 | [W5.2](#w52) | successor |
| W5.4 | [DIST-NATIVE-1](#dist-native-1) | successor |
| W5.5 | [W5.5](#w55) | owner |
| W6.1 | [AD-HIGHER-1](#ad-higher-1) | successor |
| W6.2 | [AD-HIGHER-1](#ad-higher-1) | successor |
| W6.3 | [AD-HIGHER-1](#ad-higher-1) | successor |
| W6.4 | [W6.4](#w64) | owner |
| PDE-STENCIL-FOUNDATION-1 | [PDE_STENCIL_CAPABILITY_PLAN.md](PDE_STENCIL_CAPABILITY_PLAN.md) | owner |
| MATH-SOURCE-WORKSTREAM-1 | [MATH_SOURCE_WORKSTREAM.md](MATH_SOURCE_WORKSTREAM.md) | owner |
| BLOCK-ATTNRES-1 | [BLOCK_ATTNRES_ROCM_PLAN.md](BLOCK_ATTNRES_ROCM_PLAN.md) | owner |
| REF-TIER-PHYS-1 | [PDE_STENCIL_CAPABILITY_PLAN.md](PDE_STENCIL_CAPABILITY_PLAN.md) | owner |
| FA-1 | [FUNCTIONAL_ANALYSIS_TSOL_PLAN.md](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| FA-2 | [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md) | owner |
| AD-BATCH-1 | [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md) | owner |
| AD-SPARSE-1 | [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md) | owner |
| AD-JET-IR-1 | [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md) | owner |
| IKF-1 | [INTRA_KERNEL_FEEDBACK_PLAN.md](INTRA_KERNEL_FEEDBACK_PLAN.md) | owner |
| F0 | [F0](#f0) | owner |
| F2 | [F2](#f2) | owner |
| F3 | [F3](#f3) | owner |
| F4 | [F4](#f4) | owner |
| F5 | [F5](#f5) | owner |
| F1 | [first migration](INTEGRATED_COMPILER_LOG.md#2026-09-04--foundation-f1-implementation) | archive |
| E2E-REAL-0 | [E2E-REAL-6](#e2e-real-6) | successor |
| E2E-REAL-1 | [E2E-REAL-6](#e2e-real-6) | successor |
| E2E-REAL-2 | [E2E-REAL-6](#e2e-real-6) | successor |
| E2E-REAL-3 | [E2E-REAL-6](#e2e-real-6) | successor |
| E2E-REAL-4 | [E2E-REAL-6](#e2e-real-6) | successor |
| E2E-REAL-5 | [E2E-REAL-6](#e2e-real-6) | successor |
| MSW-1 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-2 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-3 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-4 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-5 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-6 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-7 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-8 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| FA-3 | [functional-analysis owner](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| FA-4 | [functional-analysis owner](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| FA-5 | [functional-analysis owner](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| FA-6 | [functional-analysis owner](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| FA-7 | [functional-analysis owner](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| AD-FWD-NATIVE-1 | [active AD owner](AUTODIFF_EXECUTION_PLAN.md) | owner |
| X86-EVIDENCE-VOCAB-1 | [X86-EVIDENCE-VOCAB-1](#x86-evidence-vocab-1) | owner |

## NVIDIA-NVFP4-SCHEDULE-2026-09 - land with exact-device packet

Owner E2E-REAL-6; synchronization key NVIDIA-NVFP4-SCHEDULE-2026-09.
Implement the named static SM120 K16 NVFP4 block-scale contract in Graph to
Schedule/Tile and require compiler-owned target lowering. Acceptance evidence
is three exact-device numerical cases (including ragged M/N/K), replayed
Schedule-to-Tile identity, zero spill resource record, and separate CUDA-event
and end-to-end timings. No performance promotion follows the small-shape
measurements. Sibling-backend outcomes and evidence are recorded in each
backend todo. Packet:
benchmarks/baselines/nvidia_sm120_nvfp4_scheduled_20260930/.



Current slice (2026-10-01, sync
`W1.1-SM120-GRAPH-SCHEDULE-TILE-PIPELINE-2026-10-01`): the registered named
SM120 compiler pipeline now invokes PM verification, Graph-to-Schedule, and
Schedule-to-Tile before residual lowering. A lit fixture asserts typed
fragment producers and compatibility aliases; exact RTX 5070 fp16
RMSNorm-to-matmul execution proves one resident package envelope and its
NVVM/PTX MMA output. This closes the formerly skipped PM route for registered
static SM120 matmuls. Generic tensor-valued Tile constructors and the broader
producer census remain open, and timing CV does not support promotion.


- Current slice (2026-10-02, sync `E2E-REAL-6-GFX1201-MATMUL-CACHE-WARMED-2026-10-02`): static gfx1201 f16 scheduled matmul now has warmed exact-device shape-reuse evidence across 16^3 through 512^3. One HSACO/entry serves six shapes with per-shape guards; the full focused cache suite passed on Tajasaurus and the independent fp32 oracle error stayed below 1.10e-5. Kernel-event CV was 0.74–1.45%. This closes only the measured static register route on gfx1201. Princess-Luna/gfx1151, dynamic K, split-K, fused epilogues, LDS staging, and other dtype/layout keys remain open. [Evidence](../../../benchmarks/baselines/rocm_gfx1201_matmul_shape_key_20261002/README.md).


- Current slice (2026-10-02, sync `E2E-REAL-6-ROCM-CACHE-GFX1151-PARITY-2026-10-02`): Princess-Luna gfx1151 now has exact-device shape-free cache parity for softmax, reduction, and scheduled attention. All three tests passed with one native image/entry reused across runtime shapes, distinct package guards, and reference-checked outputs. This closes those tested gfx1151 envelopes alongside the static register gfx1201 matmul evidence. Other ROCm families/layouts and stable separate timings remain open. [Evidence](../../../benchmarks/baselines/rocm_gfx1151_cache_20261002/README.md).
