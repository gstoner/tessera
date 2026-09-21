---
last_updated: 2026-09-21
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
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-10--expanded-route-callers-and-f64-ownership)

### COMPILER-DEVEX-1

**Assertions-enabled validation and usable tools**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: Tajasarus now has an assertions-enabled LLVM/MLIR 23.1.1 ROCm build; RX 9070 XT gfx1201 correctness commissioning is tracked under ROCM-2 in the backend queue. Bounded gfx1201 scheduled f32 unary and standalone backward now have owning-device proof; general matrix/attention packaging, paired AD and native-Linux profiler evidence remain separate. Assertions-enabled LLVM/MLIR 23.1.1 and a hardware-free all-target Tessera compiler now pass all 475 active lit fixtures on Super-Bear. Backend-owned fixtures declare their feature requirements, the data-only x86 execution input is outside lit discovery, and the union gate requires every active fixture to pass in at least one lane. The opt-in CI lit lane requests the same full portable target matrix. Installed drivers now pass relocated-prefix smoke on Super-Bear with loader overrides removed; the CI lane runs this check after installation. Preserve these regression gates; owning-device correctness and performance remain separate backend gates. A full unit sweep on Princess-Luna (2026-09-17) found 31 failures present on main and invisible to CI, whose unit lane has neither device toolchain: 28 are now fixed — a legality gate that asked the capability registry about the generic `rocm` name for a request the rest of the stack compiles for gfx1151, a link requirement the runtime archive never published to out-of-CMake consumers, one Python-driven data file failing `check-tessera-rocm` for the whole repository, `arith.select` with no transpose, and index arithmetic refused as non-differentiable. Owed: `AUTODIFF-SHAPE-WHILE-FORWARD-2026-09-17`, a pre-existing crash inside JIT-compiled code that the AD gate had been masking, and the fact that the x86 JIT AD lane those tests exercise has no host in any automated check.
- Depends on: —
- Start: host-free
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### DIAG-PY-BACKLOG-1

**Python-raised diagnostics that were never registered**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: `tests/unit/test_diagnostic_code_registry.py` scanned Python source with a **prefix allowlist** (`E_*`, `JIT_*`, `TS_ERR_*`, `GRAPH_IR_*`), so every domain-prefixed code a Python module raises was invisible to it — the failure the scanner's own `GRAPH_IR_` comment records happening once already for a whole family, while the gate reported green. The scan now matches the code *shape*, as the C++ scan always has, which surfaced 15 unregistered tokens on 2026-09-19. **Ten closed the same day.** The nine `ROCM_FRAGMENT_*` codes from `rocm_fragment.select_fragment_layout` are registered (one file, one exception type, one fail-closed story: no generic gfx-prefix fallback, which is the miscompile ROCM-5 removed). The fifteenth, `TIMING_PROOF_INCOMPLETE`, was a **misclassification** — it is one of eleven x86 promotion-ineligibility reason tags, visible to the scan only because it concatenates a colon, and registering it would have put Decision #29's unconsumed declaration inside the registry; it moved to `_NOT_DIAGNOSTICS` and its vocabulary opened [X86-EVIDENCE-VOCAB-1](#x86-evidence-vocab-1). Remaining gate: Apple's five (`APPLE_FRAGMENT_*` ×4, `APPLE_COUNTER_EVIDENCE_UNSUPPORTED`) gain entries and the ratchet empties. They are held rather than guessed because `APPLE_FRAGMENT_UNSUPPORTED_ACCUMULATOR`'s fix hint must state whether fp32-only accumulation is permanent or pending the Metal 4 cooperative-tensor lane, and an invented answer is worse than a missing entry. Adjacent, closed here: the registry's "each prefix maps to exactly one language" rule forbade a domain prefix spanning both emitters, which `ROCM_FRAGMENT_*` does by design (C++ `LowerTileToROCMPass` and Python fragment legality); it is narrowed to the five locked sentinels, with the general case covered — and falsified as covered — by the two checks that name the offending code directly.
- Depends on: —
- Start: host-free
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### X86-EVIDENCE-VOCAB-1

**The x86 promotion-ineligibility vocabulary has no registry and no gate**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: `profiler_x86_evidence.py` appends **eleven** reason tags to a `reasons` list that decides whether an x86 performance measurement is promotable — `CPU_NOT_EXACT_ZEN5`, `VIRTUALIZED_HOST`, `WSL_CLOCK_DOMAIN`, `SOURCE_WORKTREE_DIRTY`, `TIMING_PROOF_INCOMPLETE`, `SYMBOL_SAMPLING_MISSING`, `SYMBOL_SAMPLING_INVALID`, `IMAGE_BUILD_ID_MISSING`, `EVENT_MAP_MISSING`, `EVENT_MAP_NOT_PROMOTABLE`, `SAMPLING_AFFINITY_NOT_PINNED`. Nothing enumerates them, nothing drift-gates them, and no consumer can know the vocabulary is eleven items or that a twelfth was added — so a reader handling a subset silently treats an unknown reason as no reason, which under Decision #21a is a semantic key failing open. Found 2026-09-19 while classifying the diagnostic backlog: the shape scan saw exactly one of the eleven, by the accident of a concatenated colon. Gate: the vocabulary is declared in one place with a meaning per tag, its producer and every consumer read that declaration, and a drift test fails when a tag is added without it. These are **not** diagnostic codes and must not be answered by registering them in `diagnostic_codes.py`.
- Depends on: —
- Start: host-free
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### ROCM-NVFP4-INGEST-1

**NVFP4 reaches the fp8 WMMA on gfx1201; the one lossy step must be declared**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: An NVFP4 checkpoint (e2m1 elements, **e4m3 scale per 16**, fp32 scale per tensor) reaches `v_wmma_f32_16x16x16_fp8_fp8` through MXFP4 (e2m1, **e8m0 scale per 32**), and the chain has exactly one lossy step: the scale requantization. Measured relRMS against the bf16 original — NVFP4 as shipped 0.113 (19 dB), bf16→MXFP4 **direct** 0.112, NVFP4→MXFP4 requantized 0.158 (16 dB) — so **the ~3 dB is double rounding, not the format**, and where a bf16 original exists, quantizing once to MXFP4 beats ingesting NVFP4. That is a `numeric_policy` preference (Decision #15a) and the requantization is a declared information-loss point (Decision #32) with a measured cost, not an implicit load-time transform. Notably this is **not** a workaround for a weaker part: AMD's MI355 path dequantizes NVFP4 to BF16 because CDNA4 has no native NVFP4 execution either, so the gfx1201 route lands on the fp8 ceiling (383 TFLOP/s) rather than bf16's 191. Gate: `nvfp4` and `mxfp4` distinguishable below Graph IR (they are already distinct *names* — `dtype.py` says "do not alias" — but nothing below Graph IR can tell them apart until block-scale metadata exists), the requantization expressed as a policy-gated conversion carrying its SQNR, and two traps covered: block-exponent selection by squared error between the no-clip rule and one binade finer (not truncation), and merged linears (`gate_up_proj`) honouring **both** global scales rather than collapsing them, which fails silently. Depends on the scale contract from [ROCM-FP8-BLOCKSCALE-1](#rocm-fp8-blockscale-1) and the fold from [ROCM-MXFP4-W4A8-1](#rocm-mxfp4-w4a8-1).
- Depends on: [ROCM-MXFP4-W4A8-1](#rocm-mxfp4-w4a8-1)
- Start: device
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### ROCM-SPLIT-K-1

**Split-K on the typed ROCm route: an unwired model, mis-keyed**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: `rocm_tiling.rank_candidates` computes `split_k_required` and **no production path reads it** — `scheduled_matmul.py` never imports the module, so the emitted gfx1201 kernel has no split-K whatever is ranked. Retained under Decision #29a rather than deleted, because the gfx1201 MoE router gate (M≤16, K=2048, N=256) shows split-K is genuinely needed: 16 output tiles leave half the machine idle (the part measures 64 CUs = **32 WGPs**, and a workgroup dispatches to a WGP), and the weight read dominates A by 16×, so the MMA unit is not the scarce resource. **The model is also wrong for that shape**: `split_k_required = k > 4096` answers `False` at K=2048, because the real trigger is occupancy (tiles < WGPs), not K magnitude. Gate: re-key the predicate on occupancy, give it a consumer on the typed route, and prove the router-gate shape on Tajasarus against a measured baseline — with the reduction's determinism carried as a semantic key (Decision #21a), since a split-K reduction is exactly where a reproducible router→top-k is lost. Until all three land, the declaration stays marked unwired at its site per #29a condition 1.
- Depends on: —
- Start: device
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### ROCM-SCHED-GROUP-1

**Measured and rejected: describing the panel to the scheduler loses**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: **Closed by measurement 2026-09-19, negative.** `sched-groups=N` now emits N alternating `rocdl.sched.group.barrier vmem_read`/`mfma_wmma` groups over the panel, and on gfx1201 at 1024³ f16 every setting is slower than LLVM's default: register body 70.5 → 53.8/46.6/46.0/46.0 TFLOP/s at N=1/2/4/8, LDS body 7.8 → 7.8/5.5/5.8/5.5, all arms agreeing numerically at 1.95e-06. The barriers demonstrably took effect (instruction order, count 7017/6897/6984 and `s_wait*` 1360/1283/1261 all move), so this is a real verdict on the description and not a no-op. Consequences: the 4×4 panel's spill is **not** a scheduling artifact — barriers move it 133→123 at best while costing 34% throughput — so §9's per-tile-addressing diagnosis stands and §11's doubt on that row is withdrawn; and the K-unroll and double-K comparisons were not measured against a handicapped baseline. The knob is retained at default 0 as a recorded negative. **Not settled:** one pattern was tested; `ds_read` groups, `sched.barrier`, `iglp.opt` and placement outside the K loop are untried, so this is evidence against this description rather than against the lever. Remaining gate: none — superseded by [ROCM-LDS-BANKPAD-1](#rocm-lds-bankpad-1), which the measurement points at instead.
- Depends on: —
- Start: device
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### ROCM-LDS-BANKPAD-1

**Measured: padding is a real +10-12% and it is not the 9x**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: **Closed by measurement 2026-09-19.** The LDS tiles use a 16-element row (8 dwords for f16, `gcd(8,32)=8`), so sixteen lanes hit four banks — a 4-way conflict on every fragment read. `lds-pad-dwords` makes the stride an odd dword count (8→9 f16, 4→5 fp8/int8, 2→3 int4) and is now **default 1**. Measured 1024³ f16: 7.8→8.8 and 7.9→8.7 TFLOP/s on two runs, exact at 1.95e-06 throughout. **But the register body is 70.7**, so this moves the gap from 9.1x to 8.0x and the bank-conflict hypothesis for that gap is refuted — a prediction recorded before the run said a small movement would refute it. The ISA census found the real shape: the staging copy is **scalar in both directions** (`global_load_d16_b16` in, `ds_store_b16` out) where the register body gets `global_load_b128`/`global_load_tr_b128`. Superseded by [ROCM-LDS-STAGE-VECTOR-1](#rocm-lds-stage-vector-1). Also recorded: pad=1 *narrows* the read (`ds_load_b128` → `ds_load_2addr_b32`) because an 18-element row is not 128-bit aligned, so the win is net of a regression and alignment interacts with width in a way the gcd argument alone does not predict.
**SETTLED 2026-09-19 (third pass): default is now 4, and the conflict model does not predict it.** Behind the corrected grid and the scalar copy, 2048³ f16 gives 40.3 / 39.7 / 39.5 / 35.0 for pad 4/2/1/0 with median equal to best — monotonic, tight, and a **+13% step from unpadded to any padding** with only 2% above that. 1024³ separates nothing: pad=0 interleaved against pad=4 over 21 trials **in one process** remeasured itself 16% apart (17.6 then 15.2) at an IQR of 0.3, so a tight IQR around a drifting median is a run-to-run shift the harness never spans. The uncomfortable part: pad=4 is a 12-dword stride, `gcd(12,32)=4`, which **reinstates** the 4-way conflict the padding exists to remove, and still wins — recorded as unexplained rather than explained away. See §10k. The withdrawal that produced this:  The LDS body's workgroup tile is `wavesM * macroTileM` (128x128 at 2x2 waves and a 4x4 panel) and it computes its own grid; the harness launched the artifact's 64x64 macro-tile grid, so it ran **4x too many workgroups**, each computing a full tile. Correctness was unaffected (every arm 1.95e-06, since the redundant groups recompute the same values), which is why it went unnoticed. Corrected, the 1024³ spread across pad 0/1/2/4 is 16.4–17.7 TFLOP/s with the ordering changing between runs — **within noise** — so the "+10–12% for pad=1" is withdrawn and the shipped `lds_pad_dwords=1` default is unsupported. 2048³ gives the only clean signal: mild monotonic preference for pad=4 (+10.7% over pad=0). See `docs/backends/rocm/wmma-fragment-layout.md` §10i.
- Depends on: —
- Start: device
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### ROCM-LDS-STAGE-VECTOR-1

**The LDS staging copy moves 16 bits per thread; that is the 8x**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: `emitTypedLdsBody`'s global→LDS copy loops walk one element per thread per iteration, and the ISA shows it: `global_load_d16_b16` on the way in, `ds_store_b16` / `ds_store_b16_d16_hi` on the way out, against the register body's `global_load_b128` x12 and `global_load_tr_b128` x12. Measured 8.8 against 70.7 TFLOP/s at 1024³ f16 after [ROCM-LDS-BANKPAD-1](#rocm-lds-bankpad-1) removed the bank conflict, which accounted for ~10% of the gap and not the rest. Gate: each thread stages 8 contiguous elements (`global_load_b128` → `ds_store_b128`), which also restores the wide `ds_load_b128` the padding currently costs — the two interact, so re-measure the padding sweep after vectorizing rather than assuming pad=1 stays the right default. Only then is "the LDS body loses at every shape" a statement about LDS staging rather than about this copy loop.
**Third independent line, 2026-09-19:** AMD's shipping gfx1201 FP8 GEMM (hipBLASLt Tensile library, unbundled from CCOB and disassembled) is LDS-heavy — 55k `ds_load_b32`, 22k `ds_load_b128`, 7224 barrier pairs — and its LDS reads are **wide**. The vendor kernel is not avoiding LDS; it is staging it properly, while ours moves sixteen bits per thread per iteration. See `docs/backends/rocm/wmma-fragment-layout.md` §10h.
**Fourth line, and it is a specification rather than a hint (2026-09-19).** Composable Kernel targets gfx1200/1201/1250 and makes the copy vector width a **first-class tunable** — `SrcScalarPerVector` alone appears ~2500 times across its instances. A tuned WMMA f16 instance (`device_gemm_wmma_f16_f16_f16_mk_nk_mn_instance.cpp`) sets, for **both** A and B block transfers: `SrcVectorDim=2`, `SrcScalarPerVector=8`, `DstScalarPerVector_K1=8`, with `K1=8` and `AddExtraM/AddExtraN = true`. That is **eight elements per vector on the global read and eight on the LDS write** — 128 bits each way, for both operands — alongside LDS padding. Our staging moves one element each way and, until [ROCM-LDS-BANKPAD-1](#rocm-lds-bankpad-1) today, padded nothing. So CK's tuned instance is precisely this item plus the padding already landed, which makes the target concrete: `global_load_b128` → `ds_store_b128`.
**The "8x" in this entry is withdrawn 2026-09-19.** It came from the same broken harness; corrected, the LDS body is **2.2–4.0x** off the register body, not 8x. Three of the four lines pointing at the staging copy are external and stand (the radiance kernel's 9.5 TFLOP/s ungated-global result, AMD's LDS-heavy Tensile library, CK's `ScalarPerVector=8`); the fourth was ours and is gone.

**The vectorised copy was measured 2026-09-19 and it is a 13–49% REGRESSION — and it had never compiled.** Scalar vs vectorised at 1024³ f16 behind the corrected grid: pad0 **17.6**/13.5, pad1 **14.4**/7.4, pad2 **14.6**/11.4, pad4 **15.7**/13.7. The ISA says why and it is not a tuning matter: **neither arm emits a wide load.** Both emit only `global_load_d16_b16`, and the vectorised one adds branches (146 `s_cbranch_execz` against 128), because `llvm.intr.masked.load` expands on AMDGCN into a per-element branch plus a narrow load. A runtime-masked load **cannot** become `global_load_b128`, so CK's `SrcScalarPerVector=8` is not reachable through `vector.maskedload` at all.

Two defects produced this, and the second hid the first. (a) The comment in the code — *"the ragged tail is a masked load rather than a scalar fallback, which keeps one path"* — was the defect: keeping one path is exactly what prevents the wide load. (b) `TESSERA_OPT` on Tajasarus resolves to `build-assertions` while the rebuilds went to `build`, so **every "vectorised" figure ever recorded, including the corrected §10i table, came from a binary with no vector ops in it.** It surfaced only because the next change added a pass *option*, which fails at the option parser; a changed pass *body* has no tripwire. `rocm_native.warn_if_generator_is_stale` now provides one.

`lds_copy_width` defaults to **1** (scalar, the measured-faster arm); the vector path stays reachable behind the knob as the measured-negative arm.

**The split path was then built the same day, and it loses too — for a different reason, which re-specifies this item.** It *does* emit the wide load the masked version could not (`wide=2`), and it is still slower: 2048³ f16 gives 40.4 scalar against 34.5 / 34.1 / 30.7 / 18.8 for split-vec at pad 4/0/2/1. Neither arm spills (`scratch_ops=0`; the 126-spill figure on record is the **register** body's). The census names the cause: **`ds_store` goes 6 → 17.** The global read widened and the LDS write got three times worse, because **B's write cannot widen** — the global side is contiguous in N and the LDS side is contiguous in K per column, so the fast path loads 8 elements in one instruction and stores them with eight.

**The target was mis-stated.** CK's tuned instance sets `SrcScalarPerVector=8` **and** `DstScalarPerVector_K1=8` — both sides — and can only set the second because its LDS layout is **K1-blocked** (`K1=8`), not a plain transpose. A wider copy over our layout is the half of CK's configuration that does not work alone.

**Re-specified gate: the deliverable is the K1-blocked LDS layout**, after which both sides widen and the copy width becomes the tunable CK treats it as. A wider copy over the current layout is measured-negative twice; do not attempt it a third time. See `docs/backends/rocm/wmma-fragment-layout.md` §10j / §10j.1.
- Depends on: —
- Start: device
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### ROCM-MACRO-K-TILE-1

**There is no macro K tile, and four other items are downstream of that**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: Measured 2026-09-19 on gfx1201 — our tile geometry has **M and N but not K**. The macro tile is 16×16 at M≤64 and 64×64 at large shapes, and the K tile is **16 at every shape**, because it is the WMMA's own K rather than a blocking parameter; the K loop walks 16 (or 32 under `kUnroll`). The tuned reference is **64×128 with BLOCK_SIZE_K=128**. This was found by stepping back from threading a scale field through the descriptor and asking what the whole path needs, which is the right order and was not the order I was working in.
  **Four separately-tracked items are downstream of this one gap:**
  (1) [ROCM-FP8-BLOCKSCALE-1](#rocm-fp8-blockscale-1) — the reference asserts `GROUP_K == BLOCK_SIZE_K`, so a 128-element scale block needs a 128-wide K block to hang on. Expressing it via `kUnroll` instead would make one knob serve two jobs (latency hiding *and* blocking), which is how a tuning parameter becomes a semantic one.
  (2) [ROCM-SPLIT-K-1](#rocm-split-k-1) — a split partitions a K block; with no K block there is nothing to partition, which is a better explanation of why `split_k_required` was keyed on raw `k` than "someone chose a bad threshold".
  (3) [ROCM-LDS-STAGE-VECTOR-1](#rocm-lds-stage-vector-1) — LDS stages *a K block*; the copy width and the block extent are chosen together (CK sets `BLOCK_SIZE_K` and `SrcScalarPerVector` in the same instance).
  (4) The M-bucketed configs tune `BLOCK_SIZE_M/N/K`; we model two of three, so a per-shape selection built on M and N alone cannot reach those rows however well it is tuned.
  Gate: a macro K tile in the schedule and the descriptor, distinct from the instruction K and from `kUnroll`, with the K loop blocked on it; then the scale contract, split-K and staging width all have a defined extent to attach to. **Not claimed:** that the 4×4 panel's 133-VGPR spill is downstream of this. §9 attributes that to per-tile addressing state tracking `mt*nt`, and K blocking does not obviously change it — that stays as recorded until measured.
**CLOSED by measurement 2026-09-19 — the K tile exists, is selected, and is device-proven.** `k_blocks` had carried the macro K tile in the descriptor all along: the verifier admitted `>= 1` and all three consumers gated on `== 1` (`NVIDIALowering` and `resolveFragmentLayout` returning nullopt, the generator dropping it out of `common`), so it was expressible and unreachable — the third instance that day of a representation existing with no consumer, after `split_k_required` and `ScaleLayout`. Now: the fragment-ABI gate is relaxed (a fragment ABI describes ONE instruction's register layout, identical whether the block holds one or eight — that gate conflated "which fragment" with "how many"), the generator reads it through `WmmaGemmRequest`, and the K loop issues `blocks * unroll` panels at a stride of `fragK * blocks * unroll`.
  **Selected from measurement, not from the reference.** `blockK = 32` on gfx1201 for static K >= 64, because it is the only value positive at every measured point: fp16 1024³ 1.21x, fp16 2048³ 1.65x, fp16 256x4096x256 1.53x, int8 1024³ 1.17x, int8 2048³ 1.14x, int8 256x4096x256 1.79x. **64 has the better peaks** (fp16 2048³ **1.81x**, int8 2048³ 1.54x, int8 skinny 2.56x) and loses at int8 1024³ (0.85x); **128 is a reproduced cliff for int8 at the skinny shape** (0.37x, 2.0 TFLOP/s on both runs). Every tuned vLLM/AITER config uses `BLOCK_SIZE_K = 128` uniformly and copying that constant would have cost a measured cliff — their LDS-staged Triton geometry does not transfer to our register body. Measured headroom at 64 is real and left deliberately: three shapes and two storages is not a basis for a per-shape rule.
  **Two measurement facts kept because they would mislead.** A 0.60x int8 1024³ reading at `blockK=32` did NOT reproduce (1.17x over nine trials) and is withdrawn — writing the rule from one sweep would have special-cased int8 around a regression that does not exist. And the 256x4096x256 baseline itself moved 3.3 -> 5.4 TFLOP/s between runs, because ~0.5 GFLOP is launch-overhead dominated: the absolutes at 64 and 128 are stable, the RATIOS at that shape are not, so the honest figure for int8 skinny is 2.56x rather than the first sweep's 4.12x.
  Device: **99 rows pass on gfx1201** with the blocked loop active by default, plus a row that asserts the emitted *structure* (`count(N) - count(1) == N`, measured 2→+2, 4→+4, 8→+8) because exactness alone cannot distinguish blocked-correctly from ignored. Tajasarus: both trees build, lit 437/499, `check-tessera-rocm` 75/75.
  Remaining gate: none for the tile itself. The per-shape rule that would claim the 64 headroom belongs with the M-bucketed selection work.
- Depends on: —
- Start: device
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### GOV-ODS-CONSUMER-1

**Decision #29's op-level clause has no gate**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: #29 says a declared ODS op must have a named consumer or be deleted, and cites `tests/unit/test_governance_declarations.py` as its drift gate. Measured 2026-09-19: that file checks **coverage axes** (`test_every_contract_axis_names_a_consumer`) and **duplicate dialect names** across ODS files (the Queue.td/Attn.td trap), and **nothing maps an ODS op to a consumer**. Confirmed the hard way — `tessera.scaled_matmul` was committed with no consumer and all 14 governance tests passed. That also explains #29's own history: every instance it cites (`manifold` reaching no backend, `MultivectorSpec.grades`, nine `!tile.*` types, `numeric_policy` with no carrier, `TilingInterface`) was found **by hand**, which is what an ungated rule produces. Gate: a check that each op in the Tessera ODS files is referenced by at least one pass, lowering or verifier beyond its own declaration, with a shrink-only waiver list for declarations held under Decision #29a — the same ratchet shape as `DIAG-PY-BACKLOG-1`, and it should seed from whatever the first scan finds rather than assuming the count is small.
- Depends on: —
- Start: host-free
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

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
**Host-free foundation landed 2026-09-21.** `python/tessera/compiler/rocm_mxfp4.py` now makes the distinction executable as a contract: instruction K 16, MX scale group K 32, and macro K/unroll/split-K explicitly outside the format. `tests/unit/test_rocm_mxfp4_contract.py` gates the physical scale order and refuses to let the 32-element format group drift into a schedule knob. This is contract/reference evidence only; `tessera.scaled_matmul` still has no scheduled ROCm consumer and no execution promotion follows.
- Depends on: —
- Start: device
- Latest: [ROCM-MIXED-FP8-1: the mixed OCP FP8 pairs execute, and two gates that were not checking what they claimed](INTEGRATED_COMPILER_LOG.md#2026-09-19--rocm-mixed-fp8-1-the-mixed-ocp-fp8-pairs-execute-and-two-gates-that-were-not-checking-what-they-claimed)

### ROCM-MXFP4-W4A8-1

**MXFP4 on gfx1201 is reachable through the fp8 WMMA, and we refuse it instead**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: RDNA4 has no FP4 WMMA form, and `select_fragment_layout` therefore refuses `fp4_e2m1` outright. That is correct about the hardware and wrong about the opportunity: the ecosystem answer on this exact chip (vllm-radiance `radiance_mxfp4_fp8.hip`, canonical `StillDeadcode/libr4d` `r4d_gemm_mxfp4a8_nt_m64.hip`) reaches `v_wmma_f32_16x16x16_fp8_fp8` — **the instruction ROCM-MIXED-FP8-1 proved** — through W4A8. Two routes must not be conflated. **Exact per-block:** E2M1 converts exactly to E4M3, two 16-wide WMMAs form one 32-element MX group, and its E8M0 power-of-two scale is applied exactly to that FP32 partial before it joins the running accumulator. **Folded row reference:** choose the maximum E8M0 exponent per output row, shift each block's E2M1 values into E4M3, and restore the row factor once in the epilogue. That fast fold is exact only while every non-zero shifted magnitude remains representable: gfx12 E4M3 subnormals extend the guaranteed range through exponent delta 8; the reviewed checkpoint reaches delta 10, where the source reports rounding and a non-zero residual error rather than exactness. Therefore the folded route is an explicit approximate `numeric_policy`, never the proof oracle or an unconditional replacement for the exact route. W4A8 itself remains opt-in because its activation precision differs from the checkpoint's declared W4A4 calibration.
  **Exact native vertical slice landed 2026-09-21.** `rocm_mxfp4.py` owns packed E2M1 (`low nibble = even k`), E8M0 `[K/32,N]`, per-token FP32 activation scale, fragment-order permutation/inverse, exact decoding, folded-row-reference payloads, and separate policy metadata. `rocm_mxfp4_native.py` adds an executable scalar specification and an exact WMMA route: all independent fragment loads are issued before decode/packing, two `v_wmma_f32_16x16x16_fp8_fp8` operations form one local K32 FP32 partial, and only that partial is scaled and added to the running accumulator. Tajasarus proves both packages bit-exact after BF16 rounding at ragged `17x19x64` and `32x32x128` (**4 device rows**); the recorded WMMA HSACO is wave32, 32 SGPR / 94 VGPR, zero LDS/scratch, and selects no other matrix instruction. Both exact ABIs are in the gfx1201 owning-device registry. [Evidence packet](../../../benchmarks/baselines/gfx1201_mxfp4_w4a8_20260921/README.md).

  **Remaining gate is integration and policy, not instruction reachability.** The generic `tessera.scaled_matmul` Schedule carrier still fails closed before Tile rather than dropping its scale planes; replace that boundary with a first-class scaled partial-accumulator consumer. Add the folded route only behind its approximate numeric-policy gate, split prefill from `M <= 64` decode schedules, and compare throughput against independent libr4d/Radiance measurements on a counter-capable host. No code was copied from the reviewed projects; both inspected trees lacked visible license/SPDX declarations. The M-bucketed configs still require [ROCM-SPLIT-K-1](#rocm-split-k-1); our tuning surface still lacks SPLIT_K, GROUP_SIZE_M, num_stages, waves_per_eu and cache_modifier.
- Depends on: [ROCM-FP8-BLOCKSCALE-1](#rocm-fp8-blockscale-1)
- Start: device
- Latest: [Exact gfx1201 MXFP4 W4A8 reaches FP8 WMMA](INTEGRATED_COMPILER_LOG.md#2026-09-21--exact-gfx1201-mxfp4-w4a8-reaches-fp8-wmma)

### EVIDENCE-PACKET-1

**Evidence packet consumers**

- Owner: [EVALUATOR_PLAN.md](EVALUATOR_PLAN.md)
- Gate: Unify missing packet consumers around artifact/image identity, clock validity and promotion eligibility; malformed or incomplete evidence must refuse. The [benchmark alignment review](../../../benchmarks/COMPILER_ALIGNMENT.md) maps six suites to actual callers: math remains metadata-driven, GA/EBM composition lacks route receipts, and direct-IR AD probes need paired public-frontend coverage. New diagnostic math/AD runs no longer inherit performance eligibility from target names or historical packets. The extended suite review retires synthetic SuperBench timing, labels DLOP dispatch counts as estimates and lattice timings as host-wall, and adds per-submission native guards for Apple policy-loss timing. Bounded CUDA/HIP ANN adapters now record driver-launch receipts; public SSD VJP has independent numerical comparisons. Dedicated CUDA SSD kernel/API attribution and serial/cooperative SSD adapters are validated; sequential synchronous mixed-artifact CUDA attribution and CUDA GEMM/attention adapters now pass. The ROCm matrix/unary generators now use LLVM 23 inherent kernel properties; assertions-enabled native compilation and bounded gfx1151/gfx1201 execution pass. Extend asynchronous attribution and clean performance admission; the slower gfx1201 LDS/pipelined diagnostic variants remain unpromoted.
- Depends on: —
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-10--matrix-adapters-and-mixed-attribution)

### TPROF-NATIVE-1

**Native clocks and counter attribution**

- Owner: [INTRA_KERNEL_FEEDBACK_PLAN.md](INTRA_KERNEL_FEEDBACK_PLAN.md)
- Gate: The CUDA adapter now calibrates seven launch-inclusive event windows against exact Nsight product-kernel spans, checks captured device identity and rejects excessive profiler overhead. After removing SSD's unrelated 1024-element tape-parser cap, a 512x2x32x8 cooperative workload agrees within 0.51% with 0.24% profiler overhead. The source remains dirty and the host WSL. A nine-pair collector now binds fresh process nonces and Nsight PIDs and refuses unqualified hosts before production collection. Validate per-process clean-image calibration and counter attribution on each owning host; WSL regression evidence is not clean timing.
- Depends on: —
- Start: device
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-11--program-retirement-and-declared-slot-discovery)

### DISPATCH-BREAKER

**Safe waits and checked-output ownership**

- Owner: [../backend/README.md](../backend/README.md)
- Gate: Audit remaining bridge waits and checked-output readers; bounded failure must retain live resources and poison unsafe ownership before wider adoption.
- Next architecture gate: [gated owner and isolated recovery](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#gated-metadata-owner-and-isolated-recovery-2026-09-11): every admitted metadata operation in the opt-in gated owner has replay/device proof; live metadata and legacy import/snapshot paths refuse. Process-owned heap recovery requires confirmed death. Since 2026-09-15 ([probed admission and replacement](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#probed-admission-and-health-checked-replacement-2026-09-15)) a worker is admitted only after its in-process device probe verifies the admitted producers, and a replacement is admitted only after confirmed predecessor death plus its own probe (RTX 5070 and gfx1151). Next: migrate legacy snapshot/import callers to a gated producer and validate actual driver-failure behavior (the recorded fault is an injected stall). Epoch ordering remains; no measured overlap or promotion.
- Depends on: —
- Start: host-free
- Latest: [recorded exploration](INTEGRATED_COMPILER_LOG.md#2026-09-11--gated-metadata-owner-and-isolated-recovery)

## F2

### E2E-REAL-6

**Remaining Graph-owned packaging and frontend retirement**

- Owner: [MLIR_NATIVE_FOUNDATION_SURVEY.md](MLIR_NATIVE_FOUNDATION_SURVEY.md)
- Gate: Static gfx1201 f16 matmul and f16/bf16 forward/backward attention packages retain exact driver/Schedule/Tile/image ancestry. Automatic public GQA Q/K/V AD now selects the exact live ROCm architecture and executes a recompute-only backward package, with artifact-bound physical certificates. Reusable HIP recompute workspace now owns queued host submissions/retirement; dynamic f16 matmul runs several shapes through one image. Private HIP streams and same-device read-only output leases now gate reuse and retirement; failed releases retain retryable ownership. Two independent programs have intersecting event windows in four of five uncalibrated gfx1201 trials. Bounded fp16/bf16 sparse packing now enters a registered Target op and production lowering, with six exact native comparisons. Explicit saved-LSE now reuses immutable resident O/LSE across repeated cotangents; auto remains recompute. Internal packed sparse Schedule/Tile producers now lower to SWMMAC with exact device comparisons. A logical row-major producer now performs packing/index selection in compiled GPU IR, accumulates multiple K tiles, and emits validity words; fp16/bf16 gfx1201 checks cover six cases through 48x32x128. The explicit compiler API now binds logical matrices through a validity-consuming isolated runtime package; explicit JIT tracing now specializes a single matmul under a checked 2:4 precondition, preserving output storage; declared checked-2:4 and auto_2to4 Graph matmul lower their physical recipe in C++. The auto policy selects sparse/dense per K tile on device; default-dispatch promotion remains open. Reader release can now enqueue asynchronously with non-cancellable completion and retryable failure; four gfx1201 checks verify external copies across saved/recompute and synchronous/asynchronous release. Isolated ANN replacements preserve an explicit device ordinal after confirmed worker death and repeat the numerical health probe; in-process attention teardown remains separate. Explicit sparse binding now verifies validity before exposing outputs and confirms process teardown; matching f16/bf16 accumulation has bounded gfx1201 instruction/numerical proof. Resident attention accepts exact fp32/fp64 cotangent conversions and pending host futures. Isolated attention confirms worker death before replacement and reruns a workload-scoped zero-VJP probe. A dropped saved-LSE forward carrier was corrected with reciprocal companion verification. Signed i8 and same-format E4M3FN/E5M2 sparse packing now have emitted-instruction and device comparisons. Public attention VJP accepts exact wider cotangents at capture and launch. Isolated admission checks both zero and nonzero VJP against the oracle. Mixed FP8 operand orders and independent integer signedness are now carried through the sparse package and all three IR levels. K=32 INT4 now uses byte-addressable logical inputs, explicit native nibble packing and host/device range checks. Automatic sparse selection, broader INT4 envelopes, arbitrary public AD, pointer-sharing isolation and actual driver-hang recovery remain open. F32, BF16-to-f32, FP64 and mixed uint8/int8-to-int32 x86 matmul now enter Graph-to-Schedule-to-Tile and project native ABI from replay-verified artifacts. The mixed recipe carries ui8/i8 signedness, rejects wider-than-i32 runtime extents, and has owning-CPU ragged/overflow differential proof. Cohort, elementwise and breadth Graph constructors remain. **Apple now has a first F2 family:** the canonical GEMM reduction lowers to verified `tessera_apple.gpu.tensor_view` + `gpu.matmul2d` Target IR (storage pair incl. f16 × FP8/FP4, fp32 accumulator, Apple's 128-byte MTLTensor layout quantum) and from there to the runtime's Metal 4 matmul2d symbols with the view ABI projected from IR; seven operand pairs execute on the M1 Max against an exact-bytes oracle; packed operands off the quantum are refused, not re-formed. Standalone passes only — the incumbent Apple route is unchanged and the `apple_native` GEMM packager is not deleted. Require per-target differential proof and preserve policy before deleting each route.
- Current increment: APPLE-MATMUL2D-1 follow-through (2026-09-15): ragged M/N bind at their true extents (no host zero-padding), sub-block origins and padded strides are views into the parent and reach the dispatcher through one strided-view runtime entry for every pair, the fused bias/activation epilogue is `gpu.matmul2d_epilogue`, and a two-process paired corpus admitted the family into the default pipeline for the 8/4-bit storage pairs only (no executable incumbent existed); f16/bf16 keep the incumbent. The `@jit` front door for 8/4-bit storage tensors is closed: the tracer, the Graph IR spellings, the matmul result rule and the MPS dispatcher were each silently treating FP8/FP4 as f32 (a host numpy product reported as native_gpu); they now name, spell, type and dispatch the low-precision pairs on the Metal matmul2d lane. The shared TilingPass now preserves the matmul bias/residual epilogue (re-applied as broadcast + add after the nest), so the bias-operand form reaches every backend and Apple's fused op. The bf16 arbiter bucket is measured and retained (the corpus's 5–12% was wrapper overhead in the incumbent path; on device time the entries are the same kernel). Nothing on this route is open beyond the recorded follow-ups (view-wrapper host zero-fill, default-mode migration). Previous x86 increment: the selected x86 static f32 `absolute`, `floor`, `ceil` and trailing-axis `cumsum` slices now enter a native durable Schedule contract and Tile producer. Packaging replays both boundaries and projects bindings, shape, layout and numeric behavior from serialized IR; the old Python Tile constructor is bypassed for these slices. Owning Zen 5 checks cover three ragged/ranked shapes and bitwise signed-zero/subnormal/infinity/NaN magnitude for absolute, and signed-zero/subnormal/infinity/fractional values plus NaN classification for floor/ceil. The F0 lexical census still finds Graph-to-emitter paths in `package_cohort2`, remaining `package_elementwise` branches and `package_graph_breadth`; annotation counts do not establish per-envelope closure. Cumsum now bypasses the cohort emitter with replay-bound inclusive-scan policy and owning-CPU proof. Other cohort, elementwise and breadth constructors remain open.
Compiler-example increment (2026-09-20): the maintained Qwen3-MoE example
exposed that Graph artifacts retained an optional route tensor without its
`route` identity. Both frontends now preserve named MoE tensor tails, the CPU
runtime decodes that metadata rather than forwarding it as a public-op kwarg,
and every Apple CPU example launch is oracle-checked. Accelerator examples
remain artifact claims; exact-device execution stays backend-owned.
- Depends on: [E2E-REAL-6F](#e2e-real-6f): census and proof requirements for the selected route, not all certificates.
- Start: host-free
- Latest: [compiler-example MoE optional binding](INTEGRATED_COMPILER_LOG.md#2026-09-20--examples-expose-the-lost-moe-route-name)

### W1.1

**NVIDIA typed-fragment closure**

- Owner: [../backend/nvidia/todo.md](../backend/nvidia/todo.md)
- Gate: Reconcile remaining NVIDIA fragment producers and Target proof against current typed routes; retain uncovered dtype/architecture obligations and exact-device gates.
- Depends on: [E2E-REAL-6F](#e2e-real-6f): census and proof requirements for the selected route, not all certificates.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-05--deleted-functionality-reassessment)

### W3.3

**Tile dialect ownership review**

- Owner: [IR_STACK_INTEGRATION_REVIEW.md](IR_STACK_INTEGRATION_REVIEW.md)
- Gate: Reassess surviving Tile primitive/kernel/domain/solver ownership before any dialect split; move only consumed semantics with parse/lower/execute regression gates.
- Depends on: [E2E-REAL-6F](#e2e-real-6f): census and proof requirements for the selected route, not all certificates.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-05--deleted-functionality-reassessment)

## F3

### FRONTEND-IR-MEDIUM-1

**Native recipes and broader raising**

- Owner: [FRONT_END_LOWERING_ASSESSMENT.md](FRONT_END_LOWERING_ASSESSMENT.md)
- Gate: Exact f32 attention loops, including a positive exactly representable f32 literal post-dot scale, explicit grouped-head indexing and end-aligned causal masking, now raise to a symbolic recipe and instantiate two native buckets through Schedule/Tile. The opt-in NVIDIA binding now projects native Schedule fields, replays the recipe and executes both buckets without Graph reconstruction. Apple and x86 now project target-specific parents; the x86 package executes on Zen 5. CUDA now proves ragged Q/K causal GQA buckets, with native rejection of nondivisible head counts. Bounded asymmetric window masks now execute on CUDA; mandatory serialized Presburger constraints reject fully masked rows, including when callers supply additional constraints. Finite full-shape additive bias now crosses native instantiation and Schedule projection and passes CUDA differential execution. NVIDIA raised binding now accepts irregular full-shape additive negative-infinity masks and rejects empty rows after causal/window composition; CUDA differential execution passes, including irregular masks combined with ragged GQA, causal and left-window masking for Q>K and K>Q. Extend Boolean/padding/broadcast masks, fully masked-row semantics and recognition, Apple device proof, ROCm-compatible storage and measured candidate admission; preserve complete witnesses and IR/image-bound admission.
- Current increment: Broadcast additive masks now pass on every axis: batch/head, key-padding (`[1,1,1,K]`) and per-query (`[B,Hq,Q,1]`) forms pass source recognition, symbolic bucket instantiation, native Schedule/Tile replay (the physical block survives Tile lowering) and SM120 indexing. The v3 f32 runtime ABI carries all four physical bias extents, copies only that storage and retains the seven logical kernel extents; empty-row refusal judges the logical broadcast view. Six B=2 ragged causal/GQA/window cases pass device comparison and empty-row refusal on the RTX 5070; ROCm refuses a broadcast score-bias block. Boolean/padding masks as an operand, sibling-target consumers, f16/bf16 broadcast storage and performance admission remain open.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Latest: [query/key-axis broadcast increment](INTEGRATED_COMPILER_LOG.md#2026-09-15--querykey-axis-broadcast-attention-masks-reach-native-sm120-indexing)

### MSW-9

**Broader ANN admission and tuned candidates**

- Owner: [ANN_CALCULUS_DESIGN_SPIKE.md](ANN_CALCULUS_DESIGN_SPIKE.md)
- Gate: Terminal square now has explicit analytic error amplification and intermediate-overflow refusal alongside ReLU/absolute-value consumers. Proved row-private mutable temporaries now admit the 64x8 square workload within the unchanged 4096-byte limit; constants and nested generations stay fully allocated. Extend measured nonlinear families and physical schedules. Admit only exact-artifact scoped candidates passing the measured lower-bound gate; retain incumbents when evidence is insufficient.
- Depends on: [FRONTEND-IR-MEDIUM-1](#frontend-ir-medium-1): recipe/native identity for the candidate; broader raising is independent.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-10--dynamic-reader-fanout-and-row-private-ann)

### W5.2

**Measured schedule selection**

- Owner: [OPTIMIZING_COMPILER_PLAN.md](OPTIMIZING_COMPILER_PLAN.md)
- Gate: Connect remaining producers and target calibration to measured schedule selection; preserve inferred dependencies and select only with eligible exact-device evidence.
- Depends on: [EVIDENCE-PACKET-1](#evidence-packet-1)
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--bounded-while-recovery-and-row-parallel-ann)

### W5.5

**Consumer-driven canonicalization**

- Owner: [OPTIMIZING_COMPILER_PLAN.md](OPTIMIZING_COMPILER_PLAN.md)
- Gate: Checked permutation composition, matrix-transpose flag folding, attribute-bearing cast retention and fusion guards now share legality across generic/custom canonicalization and direct transpose lowering. Extend live consumer coverage and measure candidate effects before promotion; equality saturation still requires a demonstrated ordering problem.
- Depends on: —
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-09--four-canonicalization-legality-improvements)

## F4

### W4-PRODUCT-1

**Source CFG and effect-aware recovery**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: A bounded native C++ host allocator/collector now exposes generation-checked handles, roots, cause/context cycles and reusable payload holes through a copy-out C ABI. Serialized static source exception tables now compile through MLIR/LLVM to native allocation/root/edge calls and can be enabled on the native CPU source-state exception path. Fresh-heap construction cleans up on failure. Runtime numeric tensor payloads now enter native allocation through pointer/size operands, with a one-MiB bound, owned copies, runtime shape changes and cleanup after exhaustion. The host allocation is at synchronous completion. A bounded GPU producer now transactionally copies runtime f32 site payloads into preallocated global frames with caller-supplied generation/offset/length records and exhaustion status; CUDA/HIP execution and copy-back decoding are proven. A separate bounded nonmoving GPU slot pool now generates allocation and stop-the-world mark/sweep kernels, reuses payload slots, follows up to 32 generation-checked edges per live node and collects unrooted cycles on CUDA/HIP. Opaque byte records and a resident owner now order root/edge updates, allocation and collection after closed reader leases across streams. Completed event records are reaped; failed event recording retains retryable ownership until explicit completion. Private snapshot marking now runs independently of active-graph mutations; final seeded remark/sweep remains exclusive. Bounded sweep batches now logically retire the entire unreachable cohort before range reclamation, preventing cross-batch dead-cycle edges from becoming invalid. Allocation reuses only reclaimed slots; new roots cannot resurrect retired generations. CUDA/HIP validate mutation between batches. Owned immutable pool snapshots now permit snapshot reader scopes during live-pool sweeping without racing current-storage reads; copies are writer-ordered and remain owned through reader completion. This is snapshot isolation, not unrestricted same-storage concurrent sweeping. Quiescent exact builtin containers and plain instance dictionaries are discovered without user hooks, preserving cycles and aliases. General-key dictionaries now retain key/value references; exact sets, frozensets and bytearrays are supported. Builtin subclasses with native payload/descriptors require an explicit extractor instead of silently dropping their hidden state. Opt-in fully declared slot hierarchies now use native member descriptors without attribute/property hooks; mixed instance dictionaries and undeclared opaque extension storage refuse. Explicit exact-type ExtensionLayout extractors now copy opaque payload bytes and strong referents under a caller-enforced quiescence contract, with schema/type identity and existing budgets; no global registry or automatic extension traversal is implied. Before concurrent sweeping, implement and verify a mutation barrier, generation-aware retirement epoch and reader-protected reclamation; snapshot marking alone does not authorize freeing against live mutation. Automatic throw-site fusion, concurrent sweeping, variable-size payload storage and arbitrary extension/object semantics remain open. The existing arena also supports explicit live-node rooting and cause/context edge updates. Completion publication bypasses custom exception field hooks; source records remain diagnostic, not CPython frames. Exception arena ABI leases now exclude allocation/collection until readers complete; partial collections reuse payload holes without moving live roots. Opt-in source recovery records native single-carry while and bounded multi-variable break/continue expansion, merging each iteration before the next. Assertions remain ordered native effects; unmodelled calls refuse even on dead paths. Bounded nested/tuple loop returns and statically handled builtin exceptions now execute natively; explicit CPU state slots preserve exact input aliases and non-overlapping strided views through serialized SSA state and post-completion copyback. An opt-in native CPU JIT owns four shape/alias specializations; explicit result specs transport builtin exception classes with preceding writes. Declared plain-instance/dict/SimpleNamespace tensor fields and read-only overlapping snapshots now execute, including disjoint mutable state in the same invocation; static exception args, inherited/tuple handlers and bare re-raise are transported. Source JIT exposes explicit native paired VJP of functional results and declared next-state outputs, including projected object fields, without copyback; exact aliases accumulate their adjoint at the canonical root; native slice adjoints now accumulate mapped and overlapping reads at the containing root and mask overwritten destination gradients; exception objects remain nondifferentiable; checked source VJP differentiates successful numeric paths. Rank-one positive strides and bounded injective negative/multidimensional views share an explicit contiguous containing-input SSA root; local rank-preserving slices are captured, with general mapped code generation bounded to 256 elements; positive rectangular maps use compact native slices. Native runtime-shaped slice products reuse artifacts across shapes with guarded cotangent dimensions. Source slicing captures signed int64 bounds/steps on static ranked roots, including negative and nested runtime views, clipping and empty results. Python integer arguments reuse the same CPU JIT specialization; compiler-owned capacity/shape sidecars allocate multiple multidimensional outputs up to the 1024-element host capacity. Index-only runtime gather adjoints now use shape-guarded accumulating scatters; Python index protocols and runtime-shaped original roots remain open. Single-element f32 exception-value payloads cross loop/finally completion as typed outputs; per-site/generation SSA slots retain distinct dynamic cause/context values in bounded expanded loops, and CPU/GPU VJP gates backward on successful forward completion; bounded loop-carried caught references retain their original generation payload; static caught identities, named re-raise, explicit causes and implicit contexts are reconstructed at the host boundary with native source-location notes. CUDA SM120 and ROCm gfx1151 execute mapped forward/backward and checked synchronous/asynchronous exception completion; failed frames expose no result. Extend custom object access, ownerless/noncontiguous writable views, dynamic strings, unbounded or object-carried loop exception identity and real CPython frame/traceback semantics, changing loop state and automatic effectful AD; arbitrary CFG is not closed.
- Next architecture gate: [gated owner and isolated recovery](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#gated-metadata-owner-and-isolated-recovery-2026-09-11): every admitted metadata operation in the opt-in gated owner has replay/device proof; live metadata and legacy import/snapshot paths refuse. Process-owned heap recovery requires confirmed death. Since 2026-09-15 ([probed admission and replacement](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#probed-admission-and-health-checked-replacement-2026-09-15)) a worker is admitted only after its in-process device probe verifies the admitted producers, and a replacement is admitted only after confirmed predecessor death plus its own probe (RTX 5070 and gfx1151). Next: migrate legacy snapshot/import callers to a gated producer and validate actual driver-failure behavior (the recorded fault is an injected stall). Epoch ordering remains; no measured overlap or promotion.
- Depends on: —
- Start: host-free
- AD gate: [native composed HVP execution](AUTODIFF_EXECUTION_PLAN.md#native-composed-hvp-execution-2026-09-14) adds static CPU saved-product tangents and bounded gfx1201 HVP execution; arbitrary CFG/effects, dynamic results, higher orders and general GPU binding remain open. The energy-as-typed-program acceptance's first clause is met on the CPU lane (2026-09-16, `EBM-NATIVE-QUADRATIC-2026-09-16`): the quadratic energy is a Graph IR function, its gradient is the compiler's, the K-step Langevin loop with on-device Philox noise compiles as one function and is bit-exact with the declared policy on the M1 Max, Zen 5 and Zen 2; the device clause followed the same day (`EBM-NATIVE-GPU-2026-09-16`): the row-program emitter lowers the compiler-derived gradient inside one cooperative kernel (one block per row, lanes per feature, the K-step loop and Philox in registers, ordered reductions), bit-exact on gfx1151, gfx1201 and sm_120 with one launch per loop and one `tessera-opt` invocation for the whole chain; the nonlinear energies and the sphere integrator followed the same day (`EBM-NONLINEAR-MANIFOLD-2026-09-16`): three energies run through the one integrator with no host VJP left (softplus gained its native adjoint) and `manifold = "sphere"` lowers natively with a per-row status word for its two singularities, on the CPU lane and all three GPUs; the bivector integrator and the overhead measurement closed the same day (`EBM-BIVECTOR-OVERHEAD-2026-09-16`): the grade projection rides the Clifford dialect's own op, and the measurement shows the native route flat in K (one launch per loop) against about 1.8 ms per step for the Python-emitted lane — a dispatch result that promotes nothing, since kernel-time attribution needs a box no WSL2 ROCm host can provide — [the EBM native loop architecture](../domain/EBM_NATIVE_LOOP_ARCHITECTURE.md) records what shipped and why the scoped Tile contract was not needed; four of that stream's remaining gaps closed the same day (`EBM-GA-GAPCLOSE-2026-09-16`): Clifford `exp`/`log` and `rotor_from_axis` lower to their closed forms on Cl(3, 0) so rotor sampling happens on the group, ragged batches take their loop bound from `tensor.dim` with the operands' agreement asserted rather than assumed, an annealing schedule runs as one cooperative kernel with the temperature carried in registers (a ratio of 1.0 reproducing the constant chain bit for bit), and an opaque energy adjoint is refused by name. The row-program emitter now admits only `math.*` ops whose accuracy was measured on the owning device — the sweep that established those bounds is what found `math.tanh` shipping a kernel with no body at all on gfx1151, so the packager refuses an image whose kernel stores nothing. Still open: the Clifford field ops, the 1024-feature ceiling, Apple, and promotion.
- Latest: [the math a kernel is allowed to contain, and four closed domain gaps](INTEGRATED_COMPILER_LOG.md#2026-09-16--the-math-a-kernel-is-allowed-to-contain-and-four-closed-domain-gaps)

### AD-RESIDUAL-EVAL-1

**Logical-shape ABI and persistent checkpoint execution**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Paired source VJP retirement resumes only children that have not submitted frees after a dependency failure, on the original stream. Native AD products now bind bounded dynamic GPU inputs and scalar through rank-four floating/i8/i64 results using checked capacity/shape sidecars. A synchronous checked source VJP retains device snapshots and the matching forward residuals; exception completion is checked before backward and completion metadata receives zero seeds. Same-stream asynchronous snapshots and forward-gated backward submission now expose derivatives only after successful completion. Scoped source VJP now supports reader-aware retire/poll without a context wait on the healthy path. Explicit close can still synchronize; failed-free quarantine, deferred module unload, cross-queue writer ownership and broader product bindings remain open. Exact dominating SSA product guards now tighten joint temporary capacities. Extend relational/aliased volume proofs, saved heterogeneous products, general layout envelopes and automatic frontend wiring; exported shapes do not establish arbitrary Python CFG capture.
- Depends on: [W4-PRODUCT-1](#w4-product-1): existing bounded product carrier; arbitrary source CFG closure is not a prerequisite.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-10--ssd-recurrence-and-retryable-completion)

### W2.4a

**Generation ownership and scoped readers**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Late worker death can now reconcile a failed recovery ticket without repeating termination, releasing its admission slot and owner exactly once; no device-health inference follows. Isolated ANN admission now executes independent numerical probes in the fresh worker before readiness; replacement requires confirmed predecessor death. Multi-stream external readers now include checked dynamic public frames and paired source-VJP products, and record all declared completion edges, and eventless dependencies refuse implicit host synchronization. Actual wedged-driver recovery remains unproven. Bounded asynchronous isolation teardown now retains failed owners and slots, and ANN/module owners finalize only after confirmed process death. CUDA/HIP stopped-worker replacement is independently remeasured; actual driver-hang recovery and device-wide health admission remain open; bounded workload probes are implemented. Opt-in native ANN workers now own their CUDA/HIP context and return checked private host outputs; uncertain requests quarantine the worker until confirmed process death. Stopped-worker teardown/replacement is measured separately from actual driver-hang recovery. Scoped runtime-shaped public frames and asynchronous static capture now order frees after declared readers. Module unload can run off-thread with bounded admission and non-waiting polls; a stalled/failed driver retains its owner. One through eight serialized incoming statuses now support bounded fan-in with scoped readers for each prerequisite; all 256 eight-status combinations have independent SM120/gfx1151 truth-table proof. Source state can produce immutable next-state GPU generations; exclusive synchronous owned-state copyback now reuses a private allocation with independent SM120/gfx1151 proof and blocks active scoped readers. Single-stream submit/poll now gates asynchronous copyback and excludes readers until completion; failure poisons the owner and retains pending storage. External borrowed-pointer mutation and concurrent multi-writer updates remain open. Extend heterogeneous dynamic persistent capture, unbounded/heterogeneous effect joins and external-reader adoption. Driver unload itself is not cancellable or latency-bounded, and unrestricted views retain synchronous close.
- Next architecture gate: [gated owner and isolated recovery](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#gated-metadata-owner-and-isolated-recovery-2026-09-11): every admitted metadata operation in the opt-in gated owner has replay/device proof; live metadata and legacy import/snapshot paths refuse. Process-owned heap recovery requires confirmed death. Since 2026-09-15 ([probed admission and replacement](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#probed-admission-and-health-checked-replacement-2026-09-15)) a worker is admitted only after its in-process device probe verifies the admitted producers, and a replacement is admitted only after confirmed predecessor death plus its own probe (RTX 5070 and gfx1151). Next: migrate legacy snapshot/import callers to a gated producer and validate actual driver-failure behavior (the recorded fault is an injected stall). Epoch ordering remains; no measured overlap or promotion.
- Current increment: Snapshot retirement now closes reader admission, including previously created leases, and polls every completion edge without a wait or free. Eventless readers retain storage for explicit recovery. Allocation teardown remains synchronous; next extend allocator/module retirement only with separate completion and uncertain-driver ownership proof.
- Depends on: [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1): the selected product's residual and ownership ABI; independent static slices may proceed.
- Start: host-free
- Latest: [reconciliation and wave start](INTEGRATED_COMPILER_LOG.md#2026-09-12--reconciliation-and-wave-start)

### NUMPOL-CARRIER-1

**Numerical policy and analytic budget consumers**

- Owner: [FUNCTIONAL_ANALYSIS_TSOL_PLAN.md](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md)
- Gate: gfx1201 dense WMMA has operand/accumulator-complete fragment proof and exact-device scheduled matmul packages for `fp16`, `bf16`, E4M3, E5M2, signed `int8`, and packed signed `int4`. The exact `GFX_1201` ISA rows are now `ready` for those dense inputs while the otherwise identical `GFX_1200` rows remain `artifact_only`; no family-level RDNA4 proof transfer is allowed. The public execution rows use gfx1201-only executors that check both selected HIP-device architecture and compiler chip before dispatch. Native BF16 rounding still differs from f32-rounded-once and requires explicit error-budget integration. Public sparse SWMMAC admission, scaled MX/FP4 schedules, general numerical-policy consumption, and exact gfx1200 device proof remain open. `scripts/record_dtype_codegen_inventory.py` now joins each proved gfx1201 input to the machine-readable operation proof rather than presenting ISA declarations as execution evidence; use it alongside the operator dtype-flow report. The 2026-09-11 packet verifies 24 scalar/vector basic-arithmetic rows independently on CUDA and HIP, including signedness-sensitive division; the four previously failing FP8 rows per device have byte-conversion legalization and exhaustive input-pair proof. Bool logic and bounded complex component probes pass on both devices; ten NVIDIA packed/scaled layout probes pass. Apple native f32 multiply/divide flush three expected subnormal results, so strict gradual-underflow admission remains open. Extend remaining packed/scaled, Apple storage, general complex, numerical-policy and matrix consumers before claiming closure; TF32 remains a math mode. See [dtype follow-through](../../../benchmarks/baselines/dtype_followthrough_20260911/README.md) and the [gfx1201 application checklist](AMD_KERNEL_COMPILER_SURVEY.md#gfx1201-datatype-application-checklist).
- Current increment: The Apple arena now consumes explicit gradual/FTZ policy. Integer-significand f32 add/sub/mul/div with one ties-to-even rounding step passes 133,376 input pairs per operation on M1 Max; explicit input/output FTZ around the same arithmetic also passes. Legacy unspecified arithmetic retains its historical boundary. Other floating operations, vectors/dtypes and optimized performance remain open. CUDA/HIP generic storage refuses this unconsumed policy. See the [policy contract](../../spec/APPLE_ARENA_NUMERICAL_POLICY.md).
- Depends on: —
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-13--rdna4-wmma-operand-and-accumulator-audit)

### LAYOUT-ALG-1

**Remaining physical layout envelopes**

- Owner: [CORE_SUBSTRATE_VIEW.md](CORE_SUBSTRATE_VIEW.md)
- Gate: Preserve proved static/dynamic layout consumers; extend only unresolved nonseparable tuple/layout envelopes with capacity, alias and lifetime proof. Matrix acceleration additionally needs operand packing, signedness, accumulator type, fragment shape, scale layout and target instruction witnesses; Zen 5 VNNI/vector GEMM does not establish AMX support.
- Depends on: —
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-06--descriptor-projection-and-seven-program-continuation)

### AD-SOLVER-IFT-1

**Native implicit-solver consumers**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Extend residual/predicate/solver envelopes and Apple/NVIDIA consumers through compiler-owned children; require convergence/conditioning certificates and owning-device proof.
- Current increment: The first EBM energy differentiates through the compiler (2026-09-16, `EBM-NATIVE-QUADRATIC-2026-09-16`): `tessera.sub` gained its adjoint and the sum-reduce adjoint's `unsqueeze`/`broadcast` gained linalg lowerings, so the paired autodiff pass now carries a quadratic energy to a native gradient the EBM Langevin lowering consumes; implicit differentiation and OT primitives are untouched.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-06--functional-analysis-contracts--consolidated-ownership)

### AD-HIGHER-1

**Broader AD, batching, sparsity and jets**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Extend composed attention and broader native AD families, higher-order products, real batching, structural sparse derivatives and native jets through the existing AD owners; retain oracle, structural-zero and conditioning gates.
- Depends on: [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1): the selected product's residual and ownership ABI; independent static slices may proceed.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-06--domain-and-autodiff-documentation-consolidation)

### DIST-NATIVE-1

**Real multi-rank transport**

- Owner: [SCHEDULE_OBJECT_DESIGN.md](SCHEDULE_OBJECT_DESIGN.md)
- Gate: Extend bounded MPI proof beyond two ranks and proper-subgroup participants; add native NCCL/RCCL and other transports with process ownership and real multi-rank packets.
- Depends on: [EVIDENCE-PACKET-1](#evidence-packet-1)
- Start: device
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-09--architecture-sweep-and-failure-boundary-reconciliation)

## F5

### W5.2f

**Shared tiled-SSD compiler family**

- Owner: [SEQUENCE_MIXER_ENGINEERING_PLAN.md](SEQUENCE_MIXER_ENGINEERING_PLAN.md)
- Gate: The registered internal `schedule.ssd` now owns a static f32 scalar-decay recurrence, immutable initial/final carry and chunk-end checkpoints. Schedule-to-Tile lowers it to structured tensor loops, numerically executed through the native CPU JIT including partial chunks. Replay-bound serial CUDA SM120 and ROCm gfx1151 packages now execute three chunk sizes with output/carry/checkpoint and immutable-input checks. Opt-in cooperative CUDA/HIP kernels now assign each head/value column to a block, keep one state element per active lane, and reduce in state-index order through shared memory and barriers. Both owning devices pass correctness. Nine independent paired process runs per backend now retain exact artifact identities and resident event-window confidence bounds; native clock calibration still prevents promotion; explicit measured binding is now wired. Checkpoint VJP differentiates all five inputs and all three result cotangents, recomputes within chunks, and now executes on CUDA/HIP using checkpoints from the cooperative forward. A scoped native CPU owner integrates Y with automatic host-tape grad and private checkpoints. Exact-artifact measured binding now recomputes policy and retains incumbents without native clock calibration. ResidentSSDProgram now automatically pairs GPU forward/VJP packages, snapshots inputs device-to-device, owns checkpoints and all five gradients, and exposes a synchronous first-order value_and_grad API with scoped lifetime. Asynchronous SSD gradients now compose through projected scoped generations on distinct CUDA/HIP streams and retire after external readers. Public vjp now dispatches explicitly owned resident SSD programs. Capture, VJP and whole-frame retirement use stream events and async allocation/free; scoped forward and derivative readers delay reclamation. Explicit close remains synchronous; program retire_async/poll_close now orders all frame/reader retirements before bounded-admission off-thread module unload. Partial queueing is retryable and new capture stays closed. CUDA/HIP both validate two public-VJP frames without context synchronization; driver unload itself is not cancellable or latency-bounded. SSD packages now use a separate checked 64 MiB per-buffer bound; both GPUs validate a larger cooperative workload. ResidentSSDTrace now traces connected acyclic SSD compositions before submitting work, automatically captures forward/checkpoint frames and runs reverse calls with scoped readers. Fan-out and shared public inputs accumulate cotangents through replay-validated native f32 addition kernels. A three-call DAG and all five shared public input gradients pass an independent float64 finite-difference oracle on CUDA/HIP; addition allocations and modules participate in frame/program retirement. This is host orchestration of replay-bound packages, not a fused or canonical whole-program IR. Serializing composition into the compiler foundation, unused-input zeros, same-call input alias admission, effectful/data-dependent traces and additional operation families remain open. Next: arbitrary traced public tape integration, broader owner adoption and uncertain-unload isolation recovery, broader tiling/reduction tuning, public mixer/frontend integration, broader mutation/alias lineage, ReplaySSM comparison and selector-grade promotion.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-11--resident-dag-accumulation-and-snapshot-readers)

### TSOL-POLICY-PHYS-1

**Spectral policy breadth**

- Owner: [TARGET_IR_REVIEW.md](TARGET_IR_REVIEW.md)
- Gate: Reconcile current spectral policy envelopes, then close remaining target-specific strides/full-spectrum/window, broadcasting and streaming execution gaps with independent adjoint proof.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: device
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--math-audit--foundation-reconciliation)

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
- Current increment: The batched geometric product is native (2026-09-16): `ExpandProductTable` lowers any static `[..., dim]` rank to an scf.for nest over the compile-time table with grade pruning, and `libtessera_jit` runs GradeFusion + ExpandProductTable so a `tessera_clifford.geo_product` executes through MLIR/LLVM (execution-matrix row `cpu` / `cpu_clifford_llvm_jit`; parity on M1 Max, Zen 5 and Zen 2). Same day, the whole product family (wedge, left contraction, inner, norm, reverse/involution/conjugate, Hodge star, grade projection, rotor sandwich) lowers through the same table and executes behind the JIT (`cpu` / `cpu_clifford_llvm_jit`, nine ops × three shapes against the GA reference on M1 Max, Zen 5, Zen 2). The GPU package route is closed for the family the same day: the kernel skeleton carries a rank-1 Clifford op, `ts-clifford-opt` expands it through the same lowering, the arena pipeline folds it to scalar device code (64 → 24 products under a grade-2 restriction, visible in the IR) and the native storage package launches it — ten ops × three shapes match the reference on gfx1151, gfx1201 and sm_120 (`rocm` / `rocm_clifford_native_compiled`, `nvidia_sm120` / `nvidia_clifford_native_compiled`). Open: exp/log and the field ops, ragged batches, and the acceptance's separate overhead/traffic/kernel-time measurements (the device lanes are still the Python-emitted kernels until measured against this route).
- Depends on: [AD-HIGHER-1](#ad-higher-1)
- Start: host-free
- Latest: [GPU package route](INTEGRATED_COMPILER_LOG.md#2026-09-16--the-clifford-family-reaches-rocm-and-sm120-through-the-arena-pipeline)

### RIEMANNIAN-OT

**Geometric primitives and OT validation**

- Owner: [RIEMANNIAN_OT_PLAN.md](RIEMANNIAN_OT_PLAN.md)
- Gate: Retain missing geometric primitives and constrained/KKT hypotheses under shared solver/AD owners; validate the OT workload without creating another solver stack.
- Depends on: [AD-SOLVER-IFT-1](#ad-solver-ift-1)
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-06--functional-analysis-contracts--consolidated-ownership)

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
| ROCM-MACRO-K-TILE-1 | [ROCM-MACRO-K-TILE-1](#rocm-macro-k-tile-1) | owner |
| GOV-ODS-CONSUMER-1 | [GOV-ODS-CONSUMER-1](#gov-ods-consumer-1) | owner |
| ROCM-FP8-BLOCKSCALE-1 | [ROCM-FP8-BLOCKSCALE-1](#rocm-fp8-blockscale-1) | owner |
| ROCM-MXFP4-W4A8-1 | [ROCM-MXFP4-W4A8-1](#rocm-mxfp4-w4a8-1) | owner |
| ROCM-NVFP4-INGEST-1 | [ROCM-NVFP4-INGEST-1](#rocm-nvfp4-ingest-1) | owner |
| ROCM-LDS-BANKPAD-1 | [ROCM-LDS-BANKPAD-1](#rocm-lds-bankpad-1) | owner |
| ROCM-LDS-STAGE-VECTOR-1 | [ROCM-LDS-STAGE-VECTOR-1](#rocm-lds-stage-vector-1) | owner |
| ROCM-SCHED-GROUP-1 | [ROCM-SCHED-GROUP-1](#rocm-sched-group-1) | owner |
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
