---
last_updated: 2026-08-04
audit_role: plan
plan_state: open
supersedes_queues_in:
  - COMPILER_ARCHITECTURE_SWEEP.md §4
  - FRONTEND_GRAPH_SCHEDULE_REVIEW.md §5
  - IR_STACK_INTEGRATION_REVIEW.md §5
  - AUTODIFF_ARCHITECTURE_REVIEW.md §5
  - TARGET_IR_REVIEW.md §5
  - ../domain/GA_EBM_ARCHITECTURE_REVIEW.md §4
  - RIEMANNIAN_OT_PLAN.md §4
---

# Integrated Compiler Plan

One plan across seven reviews. Each review's own ranked queue stays as *evidence
and rationale*; **this document owns sequencing and de-duplication.** Where a
review's queue and this document disagree on ordering or cost, this document
wins — the reviews were written independently and double-counted overlapping
work.

Status truth remains `MASTER_AUDIT.md` and `docs/audit/generated/`
(Decision #26). Nothing here reclassifies a row. Effort figures are engineering
estimates for a single track with no hardware gates, not commitments.

**Source reviews:**
[GA/EBM](../domain/GA_EBM_ARCHITECTURE_REVIEW.md) ·
[Autodiff](AUTODIFF_ARCHITECTURE_REVIEW.md) ·
[Sweep](COMPILER_ARCHITECTURE_SWEEP.md) ·
[Frontend→Graph→Schedule](FRONTEND_GRAPH_SCHEDULE_REVIEW.md) ·
[IR Stack Integration](IR_STACK_INTEGRATION_REVIEW.md) ·
[Target IR](TARGET_IR_REVIEW.md) ·
[Riemannian OT](RIEMANNIAN_OT_PLAN.md)

---

## 1. The thesis

Across seven independent reviews, roughly forty findings reduce to **two root
causes and one consequence.**

### Root cause A — Declared but not consumed

The compiler computes, validates, and attaches the information a pass needs, and
then no pass reads it.

| Declaration | Consumer that ignores it | Review |
|---|---|---|
| `manifold` attribute on `ebm.langevin_step` | every backend codegen (6 grep hits, all comments) | GA/EBM §1.1 |
| `MultivectorSpec.grades`, `IsRotor`, `Even`/`Odd` | `geometric_product` — iterates all `dim²` pairs | GA/EBM §2.1 |
| `batching_rule` axis, closed across 480 primitives | `vmap` — a Python `for` loop | Autodiff §B3 |
| `shape_rule` axis, reported closed | `_infer_result_type` — a five-case if-chain | Frontend §G2 |
| `!tile.fragment`, `!tile.buffer`, `!tile.tmem`, … (9 types) | partially consumed; core `tile.mma`/`tile.async_copy` and compatibility envelopes remain open | IR Stack §T1 |
| `numeric_policy` (Decision #15a) | no carrier below Graph IR at all | IR Stack §T6 |
| `TilingInterface` on Matmul/Conv/FlashAttn | `fusion_core.py` — 7 hand-enumerated regions | Sweep §F3 |

### Root cause B — Told, not derived

Passes are *given* facts syntactically instead of *computing* them, so they are
wrong at the edges and fail open.

| Missing analysis | What happens instead | Review |
|---|---|---|
| Effect/purity on the IR | `ast.NodeVisitor` name-matching; aliased RNG ⇒ inferred pure ⇒ `deterministic=True` passes | Sweep §F1 |
| Differentiation activity | `AutodiffPass` builds adjoints for everything | Autodiff §A5 |
| Gradient demand / trajectory liveness | `CheckpointInnerLoop` marks every EBM step in a containing loop, but no downstream pass consumes those marks | GA/EBM §1.5 |
| Symbolic shape constraints | `dims_compatible` is `str(lhs) == str(rhs)` | Sweep §F2 |
| Fusion legality | region shapes enumerated by hand | Sweep §F3 |
| Sharding propagation | every layer annotated by hand; `validate()` contains `pass` | Sweep §F4 |
| Tile-level invariants | six `*LegalityPass` re-deriving what types would give free | IR Stack §T2 |

### The consequence — C: duplication

**A + B ⇒ C.** When the declared contract doesn't carry the information, and no
analysis derives it, the only way to ship is to write a second implementation
that does. Every duplication in the tree traces to this:

| Duplication | Why it exists |
|---|---|
| Two frontends (AST `_OpExtractor` / tracer), opposite failure policies | the AST path can't produce SSA through control flow |
| Two Graph→Schedule, two Schedule→Tile lowerings; Python canonical — **but see the correction below** | the MLIR types don't carry what codegen needs |
| Python AD tape + `AutodiffPass` | the tape can't be a transform (global monkey-patching) |
| GA Python fast paths + `RotorSandwichFold` marker | `ExpandProductTable` rejects batched operands |
| Two `Queue.td`, two `Attn.td` defining the same dialect | accretion, uncaught |
| Two remat passes with opposite rigor | no shared liveness analysis |

**This is why the plan is ordered A → B → C.** Collapsing a duplication before
the surviving path can carry the information just deletes a working system. Every
attempt to start at C fails.

> **Correction (2026-08-02, from W0.6 execution) — the Graph→Schedule and
> Schedule→Tile row above overstates the C++ side, and this makes W3.2 *larger*,
> not smaller.** There are not two competing lowerings at those boundaries. On
> the C++ side there is one **annotation-only skeleton**: `GraphToSchedulePass`
> stamps `schedule.artifact_hash = "__pending__"` on three op-name prefixes and
> `ScheduleToTilePass` stamps `tile.staged` on `schedule.async_copy`. Neither
> matches, replaces, or rewrites any op — the original source comment says so
> outright ("a real pass would pattern-match and replace ops"). Worse, the
> library holding them (`TesseraPM`) is linked **only into the test binary**;
> `tessera-opt --help` in the production driver does not list
> `-tessera-graph-to-schedule` at all.
>
> So W3.2 is not "converge two implementations onto MLIR" — the MLIR
> implementation does not exist yet, and the Python spine is not a duplicate of
> it but the only implementation. Re-scope W3.2 accordingly before funding it.
> The passes now carry `[annotation-only skeleton]` in their registered
> descriptions and a maturity contract in `PMPasses.h`, so nothing can cite them
> as evidence of a working boundary again.

---

## 2. Governance — the rules that stop regrowth

A cleanup without rules regrows. Six standing decisions, each derived from a
specific finding, each drift-gateable:

| # | Rule | Prevents | From |
|---|---|---|---|
| **#21a** | **Semantic keys never default.** An attribute selecting *semantics* fails closed on absence; one selecting *performance* may fall back with a diagnostic. Semantic: `manifold`, `algebra`, `math_mode`, `rounding_mode`, `distribution`, `dtype`. Performance: tile sizes, stage depth, `auto_batch`, checkpoint budget. | silent Euclidean fallback; `operand_types[0]`; unvalidated `StrAttr` | OT §H1 |
| **#10a** | **An eligibility-marking pass ships a negative fixture.** Any pass annotating work as rematerializable / fusable / pipelineable gates on demand analysis and ships ≥1 fixture where the correct output is *no annotation*. | 2500 dead steps marked rematerializable | OT §H2 |
| **#29** | **A declaration must have a consumer.** If the compiler declares metadata — an ODS type, a coverage axis, an attribute — a named pass must consume it, or the declaration is deleted. Drift-gated: a test asserting every `primitive_coverage` axis names its consumer. | root cause A, all seven instances | this doc |
| **#30** | **Derive, don't ask.** A pass needing a program fact queries the analysis layer. New bespoke walkers are rejected in review. | root cause B; the eighth hand-rolled analysis | Sweep §5 |
| **#31** | **One implementation per boundary.** Each level boundary has exactly one production lowering. A second implementation is either a declared oracle with a differential test, or deleted. | root cause C | IR Stack §T3 |
| **#32** | **Information loss across a level boundary must be declared.** A lowering carries each Decision #15a attribute forward or records a named reason it dropped it; a boundary verifier fails on silent loss. | `numeric_policy` vanishing above the MMA | IR Stack §U5 |

Adopt these **before** Wave 1, not after. #29 and #31 in particular change what
gets accepted in review, and they are what make Waves 1–3 stick.

---

## 3. De-duplication ledger

The seven reviews costed overlapping work independently. Corrections applied here:

| Double-counted work | Costed as | Merged into | Saved |
|---|---|---|---|
| Symbolic shapes (Sweep #10, 3w) + control-flow adjoints (AD D4, 6w) + frontend regions (E7) | 3 separate items | **W4** — one program, one gate | ~4w and 2 items that would each have "landed" with zero capability |
| Differential harness: trace-vs-AST (E2) + Python-spine-vs-MLIR (I5) | 2 harnesses | **W3.1** — one harness, ~~two uses~~ **one use today** (see below) | ~2w *(saving no longer holds as stated)* |
| Schedule-decision work (Frontend U3/E6) + (IR Stack U6/I6) | 2 items | **W5.2** | ~5w |
| Effects re-homing (Sweep #9) vs derive-from-traced-IR (Frontend E3) | 2 approaches | **W2.2** — E3 strictly supersedes | ~2w |
| Implicit diff: OT R2 `custom_root` + AD "finish NewtonAutodiff" | 2 items | **W3.5** — same pass | ~2w |
| Legality collapse (IR Stack I2) as independent work | standalone | **W2.4** — client of the dataflow layer | ~1w |
| Remat unification: ~~delete `EBMCheckpointInnerLoop`~~ **(done in W0.2)** + AD D5 | 2 items | **W5.1** — now AD D5 only | ~1w *(already banked)* |

**Two rows corrected after W0 execution (2026-08-02).** The ledger is accounting,
not a work queue — every row's work already lives in a wave item — but two rows
no longer describe reality:

- **Remat unification.** Deleting `EBMCheckpointInnerLoop` is **done**: W0.2
  removed it from the default EBM pipeline (its three attributes had zero
  consumers tree-wide), kept it as an explicitly experimental standalone pass,
  and shipped the Decision #10a `CHECK-NOT` fixture proving the default pipeline
  emits no checkpoint annotations. W5.1 therefore owns **only** AD D5 — the
  demand-aware residual policy as an arbiter axis. Do not re-scope W5.1 as if
  the deletion were still ahead of it.

- **Differential harness — this merge's saving does not hold.** It assumed two
  implementations to compare at the Graph→Schedule / Schedule→Tile boundaries.
  W0.6 established there are not: the C++ side is an **annotation-only skeleton
  in a test-only library** (`GraphToSchedulePass` stamps
  `schedule.artifact_hash = "__pending__"` and returns; `TesseraPM` is linked
  only into the test binary, so production `tessera-opt` never exposed
  `-tessera-graph-to-schedule`). The trace-vs-AST use is real and unaffected;
  the Python-spine-vs-MLIR use has no MLIR side to differ against until W3.2
  builds one. **Net effect: W3.1 keeps one use, and W3.2 grows** — see the
  correction under §1.

**~17 weeks of double-counting identified**, minus the corrections above.
Queue estimates are directional:
the source documents used different scopes, and the Target-IR corrections below
replace a blanket ROCm migration with an ownership-and-evidence gate.

---

## 4. The waves

Each wave has **one observable exit criterion**. A wave is not done when its
items are merged; it is done when the criterion holds.

### W0 — Stop the bleeding *(4 weeks · no dependencies · start immediately)*

Live defects, fail-open paths, inert machinery, and false documentation. Every item is
independent; run them in parallel.

| # | Item | Source | Effort |
|---|---|---|---|
| W0.1 | **Landed 2026-08-02.** `manifold` is now `EBM_ManifoldAttr` (a `StringBasedAttr` pinning `euclidean`/`sphere`/`bivector`), and `Canonicalize`'s Euclidean fallback is replaced with `emitError`+interrupt+`signalPassFailure`. Verified: unknown value and missing value are both rejected before any pass runs; negative fixture `canonicalize_rejects_bad_manifold.mlir`. The typed-`EnumAttr` upgrade stays with W1.1b. Two side findings, both fixed: the `.td` comment claiming ODS "doesn't support" a constrained string alias was false (`StringBasedAttr` is already used by the ROCm dialect in this tree), and `ts-ebm-opt` never registered `arith`, so **6 of its 12 lit fixtures could not parse** — invisible because `TESSERA_BUILD_EBM_BACKEND` is OFF by default. EBM lit is now 12/12. | GA/EBM §1.1 | 3d |
| W0.2 | **Landed 2026-08-02.** `checkpoint-inner-loop` is out of `tessera-ebm-pipeline`; verified that `tessera.ebm.{checkpoint_loop,checkpoint_budget,recompute_step}` have **zero consumers tree-wide**, which is why an unconsumed declaration must not ship in a default path (#29) and why an eligibility pass must gate on demand analysis (#10a). The pass remains registered and explicitly labelled experimental so its own fixtures keep running. The Decision #10a negative fixture lives in `full_pipeline_chain.mlir` (`CHECK-NOT` on all three attributes). Demand-aware loop rematerialization stays W5.1 — and per the de-duplication ledger correction, W5.1 now owns *only* that, not the deletion. | GA/EBM §1.5 | 1d |
| W0.3 | **Landed 2026-08-02.** A traceable EBM energy is now defined as one whose scalar result flows through `tessera.ops.*` on the state it was handed; `_tape_grad`/`_tape_grad_mv` return a reverse-mode gradient when a cotangent path is actually recorded and `None` otherwise, so raw-NumPy callbacks keep the central-difference path. Both `bivector_langevin_step` and `sphere_langevin_step` try the tape first. Measured on Cl(3,0) (D = 2³ = 8): **1 energy evaluation instead of 16**, and exact instead of first-order. **Root cause was tape identity, not the samplers** — `Multivector.coefficients` returns a fresh read-only *whole-array view* on every access, so the tape's `id()`-keyed identity could never match the state buffer and manifold energies were untraceable in principle. **Recorded negative result:** fixing this inside `Tape._describe` (resolving a whole view to its base) is the obvious move and is **wrong** — `Tape.record` keys an op's *output* on `id(output)`, so rewriting only the *input* side severs producer→consumer links and silently drops gradients. Measured: 12 Clifford/MoE autodiff failures, all numerically silent. Identity is therefore recovered *after* `backward` in `_cotangent_for_buffer`, local to the EBM helper, leaving global tape semantics untouched; a comment in `_describe` records why the tempting fix is rejected. Six regression tests cover both paths plus their agreement on a full Langevin step. | GA/EBM §2.6 | 1w |
| W0.4 | **Landed 2026-08-02.** `jacrev` records the forward pass once and re-runs `backward` with `retain_graph=True` per output element — measured 1 evaluation instead of 4 on a 4-element output, and the `retain_graph` machinery whose own docstring named `jacrev` as its motivating caller is finally wired to it. **`jacfwd` was evaluated and needs no fix:** one `jvp` per *input* dim is the definition of forward mode, not a forward-pass-per-element defect, and its docstring already says exactly that. Reviewed again in PR #490 — the single-tape rewrite removed an accidental shield (the old code wrapped `fn` in `sum(out*cotangent)` through `ops.*`, making the target tape-produced regardless), so `jacrev` of an identity or constant now resolves structurally. | Autodiff §B1–B2 | 3d |
| W0.5 | **Completed 2026-08-02:** Decision #5 in `CLAUDE.md` now states that the effect lattice walks the AST, not the IR | Sweep §F1 | done |
| W0.6 | **Landed 2026-08-02.** Deleted the dead `dialects/tessera_{queue,attn}/*.td` (no CMake referenced them, yet **six docs cited them as the authoritative source** — all repointed). The new Decision #31 drift gate then found a **third duplicate the reviews missed**: `src/compiler/programming_model/ir/tile/TileMemoryOps.td` declared the same `tile` dialect name as the production `Tessera/Dialect/Tile/TileOps.td`, with *contradictory* mnemonics (`mma.tcgen05` vs the live `tcgen05.mma`); it was tablegen'd but never `#include`d by any source and never registered. Deleted, and `CLAUDE.md`'s GPU-only tier corrected to the real mnemonic. PM passes moved to `lib/PMPasses.cpp` + `include/tessera/ProgrammingModel/PMPasses.h`. | IR Stack §T5, §T3 | 3d |
| W0.7 | **Landed 2026-08-02.** All three GA8 pass summaries now read `[annotation-only]` with an explicit "rewrites no IR" rather than the ambiguous `[GA8 stub]`, which conflated "does nothing" with "does something partial". The false claim that GA8 passes "gate on `canonical` and refuse to proceed on out-of-allow-list signatures" is removed from **both** `CliffordPasses.td` and `AnnotateAlgebra.cpp` — verified the GA8 passes reference `canonical` nowhere, making it a live #29 declaration-without-consumer, now recorded as such. (The one remaining "GA8 stubs" mention is a descriptive file-header line about where the passes live, not a capability claim.) | GA/EBM §1.4 | 1d |
| W0.8 | **Landed 2026-08-02.** All six decisions are in `CLAUDE.md`'s do-not-revisit list with their originating defect. #29 and #31 are drift-gated by `tests/unit/test_governance_declarations.py`: every `primitive_coverage` axis must name an existing consumer file, and no two ODS files may declare the same dialect name. The two genuinely-unconsumed axes (`batching_rule`, `shape_rule`) are explicit ratchet waivers naming their owning wave item, so they read as open rather than closed. The #31 half found a duplicate dialect on its first run (see W0.6). | §2 | 1d |
| W0.9 | **Landed 2026-08-02, and it found more than expected.** Substring assertions retained as smoke; a real parse + dialect-load + verifier harness now runs each emitter's text through `tessera-opt`. **Result: every Python-emitted "Target IR" fails a real MLIR parse.** Two stacked defects, both invisible to `in`-assertions: (1) module attributes are not dialect-prefixed (`arch`, `target`, `target_features`), which `builtin.module` rejects outright; (2) underneath that, the ops violate their own ODS — `tessera_rocm.mfma` is emitted as `() -> ()` carrying its result as a **string attribute** (`result = "v0"`) while the dialect requires one SSA result. So the Python lane emits text that *resembles* the dialect without being it, and Decision #19's contract was validated by a test that could never have caught this. NVIDIA targets **skip** rather than fail — `tessera_nvidia` is not compiled into the default build, and failing them would measure the build config rather than the emitter.

**The ratchet is now EMPTY — all of it was fixed, not just recorded.** Four distinct defects, each invisible to `in`-assertions: (1) module attributes are dialect-prefixed at MLIR-render time (`_mlir_module_attrs`), keeping the short Python-facing keys callers index; (2) the function container is `func.func`, replacing a hardcoded map of `tessera_apple.cpu.func` / `tessera_rocm.func` / `tessera_nvidia.func` / `tessera_x86.func` — **none of which any dialect defined**; (3) `mfma` / `async_copy` / `wait` emit their real ODS signatures with the async-copy token threaded into the wait; (4) five emitted-but-undeclared ops (`tessera_rocm.{elementwise,kv_cache_read,msa_block_sparse}`, `tessera_apple.cpu.{kv_cache_read,moe_solver}`) were added to their dialects. A second, duplicate emitter family in `matmul_pipeline.py` had the same defects and was fixed with it. The gate now also parses **every committed golden**, which is what caught defect (4): the single-matmul test passed while the multi-op `matmul_softmax` goldens did not.

**The `cpu` reference lane is now closed too (2026-08-02) — no exclusions remain.** It emitted `tessera.cpu.<source-op>`, one op name per Graph IR op, so its vocabulary grew with the op set and could never be enumerated in ODS. That name was pure redundancy: the CPU verifier already *requires* a `source` attribute naming the originating op. It now emits the single declared `tessera.cpu.reference` node (plus `cpu.profiler_probe` and `cpu.msa_block_sparse`, kept separate because they carry distinct contracts), and parses and verifies like every other lane. Every target the build compiles a dialect for — `cpu`, `x86`, `rocm`, `apple_cpu`, `apple_gpu` — now passes a real parse + dialect-load + verifier run; only NVIDIA skips, and only because its dialect is off in the default build. | Target §X4 | 1w |
| W0.10 | **Decided 2026-08-02: build `tessera_x86`.** No carve-out — Decision #19 stays universal. Evidence that settled it: `TileToX86Pass` lowers Tile IR to **21 `func::CallOp`s** into a hand-written C shim plus arith/memref glue, using neither a `tessera_x86` dialect nor MLIR's upstream `amx`/`x86vector` dialects — structurally the same `func.call`-to-a-C-symbol shape `CLAUDE.md` already flags for Apple GPU. The build is cheaper than the other backends' equivalents because the abstract ops largely exist upstream: the hardware-free layer (`tessera_x86.amx_tile_load`, `.amx_dpbf16ps`, `.avx512_gemm_microkernel`, pack/unpack) can lower into `amx.*`/`x86vector.*` rather than terminating in `func.call`. **Built 2026-08-02.** `tessera_x86` is defined, tablegen'd, linked into `tessera-opt`, and registered (`--show-dialects` lists it). It separates **value-carrying** ops — `amx_tile_load` / `amx_tile_zero` / `amx_dpbf16ps` / `amx_dpbusd` / `amx_tile_store` over a real `!tessera_x86.tile` type — from **directives** (`avx512_gemm_microkernel`, `pack_b_panel`, `elementwise`, plus the emitter's `kernel` / `kv_cache_read` / `unsupported`). `abi_call` models the C-shim boundary rather than hiding it, so Decision #28's arbiter can distinguish compiler-generated from delegated work. Positive **and negative** lit fixtures ship (`x86_target_ir{,_invalid}.mlir`); the negative one proves the typed layer rejects an AMX dot-product whose operands never came from a tile — exactly the property a substring test cannot check. The Python x86 emitter's output now parses, loads the dialect, and verifies. **Remaining, and re-scoped 2026-08-02:** lowering into upstream `x86vector.*` (AVX-512) instead of terminating in `func.call` is the live follow-on — it changes generated code and needs AVX-512 execute-and-compare on this box. **The AMX half is deprioritized to optional:** per the project owner, AMX is expected to be superseded by the ACE matrix instructions jointly agreed by Intel and AMD for future CPUs, so an AMX → `amx.*` lowering is not worth building now. (Recorded as owner direction; ACE specifics are not independently verified in this plan.) The AMX ops stay in the ODS as the IR-level contract — they cost nothing, they pin the tile/accumulator shape, and they give the eventual ACE ops a structure to follow. This also removes the fleet's only hardware blocker here: AVX-512 execution is available on this box, whereas no machine in the fleet has AMX. | Target §X1 | 1h to decide (done); dialect + fixtures done |

**Exit:** the open-string manifold key is verified, the EBM default pipeline
emits no unconsumed checkpoint policy, `CLAUDE.md` Decision #5 is accurate, and
every dialect has exactly one ODS.

### W1 — Make declarations binding *(8 weeks · depends on W0.8)*

Root cause A. Most items enforce existing declarations; fragment/buffer
parameterization and target matrix contracts require bounded, variant-aware type
design before migration.

> **W1 status — 2026-08-04. NOT closed.** Verified against `main`, not recalled.
>
> | item | state |
> |---|---|
> | W1.2 shape-rule registry | ✅ complete |
> | W1.3 metadata boundary verifier | ✅ complete |
> | W1.4 GA grade threading | ✅ complete |
> | W1.1b semantic `$kind` | 🟡 partial — the 3 fail-OPEN ops closed; 14 already fail closed, not yet hoisted to ODS |
> | W1.1 Tile IR typing | 🔴 **open** — **2 of 6 numbered steps** landed (1, 2) |
>
> **Landed but NOT numbered steps** — real work, and deliberately not counted
> as step completions: the 2b **guard** (NVWGMMA fails closed on an accumulator
> it would drop, #506) and **3a** (ragged masking in `materializeFragmentPack`
> plus the shared bounded-`tile.view` arity contract, #510). 3a is a
> prerequisite invented for option (a), not an entry in the design doc's step
> table. Counting either as a numbered step overstates progress — an earlier
> version of this block said "4 of 6" by doing exactly that.
>
> **What is really open, in dependency order:**
>
> 0. ✅ **LANDED 2026-08-04 — the typed lowering COMPOSES (§4.6.1).**
>    `convertTypedFragments()` in `TileToROCM.cpp`: `TileFragmentTypeConverter`
>    + four conversion patterns + `applyPartialConversion`, running ahead of the
>    legacy walk, which still owns the bare `!tile.fragment` spelling. The
>    K-loop, an mma feeding an mma, and a non-`fragment_zero` accumulator all
>    lower — `rocm_typed_fragment_composition.mlir`, verified to fail when the
>    synthesized-zero defect is re-injected. `scf.for` was one library call, as
>    predicted. **Not yet on an executing lane: no producer emits typed
>    fragments until step 3, so this is proven by fixture only.** Two defects it
>    exposed, both with green positive tests, are in §4.6.1 and the paired
>    `_invalid` fixture. Original scoping below.
>
>    <details><summary>Original scoping (2026-08-04)</summary>
>
>    `TileToROCM`'s typed path is
>    a single-shot whole-chain pattern match (`view → pack → zero → mma → unpack
>    → store`, then erase), so an accumulator that is not a `fragment_zero`, an
>    mma feeding another mma, and a chain crossing a loop boundary are all
>    inexpressible *by construction*. Replace it with a `TypeConverter`
>    (`!tile.fragment` → `vector<N × T>`) + conversion patterns. `scf.for`
>    iter_args come free from
>    `populateSCFStructuralTypeConversionsAndLegality`, which this LLVM 23 ships
>    — the hand-rolled region conversion §4.2 sized as the largest step is a
>    library call. Cost is that **no pass in this tree uses a `TypeConverter`
>    yet**; this is the first.
>
>    </details>
>
> 1. **step 3 — restructure producers onto `tile.view` + `fragment_pack`.**
>    `fragment_pack` requires a `!tile.tile`; zero producers supply one
>    (`TileIRLoweringPass` passes tensors, the three `GenerateWMMA*Kernel` passes
>    pass lane-level vectors). Option (a) chosen 2026-08-04; 3a landed. **Made
>    materially smaller by (0)** — the producer then emits well-typed ops rather
>    than a pattern one matcher must recognise whole.
> 2. ✅ **step 2b — CLOSED by (0) as a capability.** A non-zero accumulator is a
>    converted operand, and "synthesise a zero" is now the lowering of
>    `fragment_zero`. Closed on the **ROCm** side only, and closed as a
>    capability rather than as shipped codegen — nothing emits it until step 3.
>    The NVIDIA fail-closed guard (#506) stays: `NVWGMMALoweringPass` has had no
>    equivalent conversion built, and step 6's NVIDIA half remains unverifiable
>    on this box (needs `-DTESSERA_ENABLE_CUDA=ON`).
> 3. **step 4** — the five Python text emitters.
> 4. **step 5** — delete `MMAOp::verify`'s permissive branch. Unreachable until
>    (1) and (3) complete; deleting it earlier breaks every producer.
> 5. **step 6 — Target IR dialects.** Remove unexplained `AnyType` from
>    `tessera_nvidia` (3/3) and `tessera_apple` (12/12), with `tessera_x86`
>    (0/0) as the reference shape. Independent of the producer chain, and it
>    was omitted from an earlier version of this list — which is how still-
>    required Target IR work disappears from the owning queue.
> 6. **W1.1b** — hoist the already-fail-closed `$kind` sets into ODS.
>    **Re-measured 2026-08-05: 11 `StrAttr:$kind` sites across FIVE
>    dialects** — Graph IR `TesseraOps.td` (6), Apple (2), NVIDIA (1),
>    ROCm (1), Neighbors (1) — not one backend's ODS. That makes it a
>    shared-contract change under AGENTS.md (same PR assesses every
>    backend), and each site needs its legal set derived from its actual
>    consumer dispatch. Deriving such a set from a partial read is how
>    #499 shipped an optimizer enum missing `adafactor` and broke six
>    tests including one that executes on gfx1151, so budget per-site
>    derivation plus a run of the existing tests, not a bulk edit. A
>    layering improvement, independent of everything above.
>
> Items 1–4 are one chain. Items 5 and 6 can proceed in parallel with it.



| # | Item | Source | Effort |
|---|---|---|---|
| W1.1 | **2 of 6 numbered steps landed (1, 2), plus the unnumbered step 0 that unblocks the rest; steps 3–6 open.** Step 2b is closed on ROCm by step 0.  Design + inventory: [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md), [`W1_1_TYPING_INVENTORY.md`](W1_1_TYPING_INVENTORY.md). **Landed:** (1) `!tile.fragment` parameterized on `m/n/k, elem, acc, role, layout, family` — `family` is in the TYPE because it selects a physical register ABI (wave 32 RDNA/WMMA vs 64 CDNA/MFMA), which an earlier draft got backwards (#502). (2) `MMAOp::verify` reads the contract from the operand types, and `fragment_pack`/`fragment_zero` do the same for their result, so **the canonical K-loop verifies** — its accumulator is an `scf.for` iter-arg with no defining op, which is why producer-chasing made the typed form unusable by every real GEMM (#503). (2b-guard) `NVWGMMALoweringPass` now REFUSES an mma carrying an accumulator instead of lowering it to a two-operand call that silently dropped it — a pre-existing wrong-answer bug, not a regression (#506). (3a) `materializeFragmentPack` can mask a ragged edge, and the bounded `tile.view` arity is defined in the SHARED verifier rather than per backend (#510). (0) `TileToROCM`'s typed path is now a **dialect conversion** (`TypeConverter`: `!tile.fragment` → `vector<N × T>`), the first in the tree, so a K-loop / chained / non-zero accumulator all lower by composition — which is what closed 2b on ROCm. The 2b guard and 3a are landed work but are NOT numbered steps — counting them was how an earlier version of this row reached "4 of 6". **Open:** step 3 producer restructure; step 4 Python emitters; step 5 delete the permissive branch; **step 6 Target IR dialects** (`tessera_nvidia` 3/3, `tessera_apple` 12/12 unexplained `AnyType`, with `tessera_x86` 0/0 as the reference), which is independent of the producer chain. **The blocker (§4.5):** `fragment_pack` requires a `!tile.tile`, and **zero producers supply one** — `TileIRLoweringPass` passes tensors, the three `GenerateWMMA*Kernel` passes pass lane-level vectors whose lane math they did themselves. That is a division-of-labour mismatch, not a syntax gap, so step 3 is a rewrite of working numerically-verified generators and step 5 is unreachable until it completes. Option **(a)** (restructure producers) was chosen 2026-08-04; 3a was its prerequisite. | IR Stack §U1 + Target §X2 | 5w |
| W1.1b | **Partially landed; the row's premise did not survive measurement.** It said "62 × `$name`, 4 × `$kind`, 1 × `$mode`". Measured: **17** ops carry `$kind`, **3 of them are `I64Attr`** rather than strings, and **14 of 17 already fail closed** in their generators. `$name` is the emitted kernel SYMBOL (`flash`, `fc1`, `bwd`, …), an open set chosen by the caller — enumerating it would reject valid programs, so it is deliberately left a free string and gated as such. **Landed:** `$dtype` split into three per-op-family constraints (#499, after review showed one shared union let `softmax` accept `int8`); `reduction` / `mode` closed sets (#499); and the **three `$kind` ops that failed OPEN** — `predicate`, `optimizer`, `clifford` — closed (#505). Those three each had a trailing `else` doubling as an unnamed semantic default, so a typo silently computed `isfinite`, trained with Adam, or evaluated the **geometric product** instead of the requested Clifford operation. **Open:** hoisting the other 14 already-fail-closed `$kind` sets from their generators into ODS — a layering improvement (reject at verification, not in the generator), not a correctness fix. | Target §X3 | 1w |
| W1.2 | **Landed 2026-08-03.** Both halves now hold. (a) Unknown op ⇒ diagnostic: `_infer_result_type` raises when a catalog-declared rule has no implementation, instead of the old five-case if-chain ending in `return operand_types[0]` — correct for the ~60 elementwise ops and silently wrong for everything else. (b) **Auto-flip wired** — `primitive_coverage.shape_rule` is derived from `op_catalog` via `_catalog_shape_rule_status`, the mechanism `op_catalog`'s own source predicted and nobody had connected. It found a live defect: the dashboard promoted `shape_rule` off the LOWERING KIND and never consulted the catalog, so **all six ops whose rule the catalog had explicitly withdrawn reported `complete`** — the same bug `shape_rule_for` had already fixed one layer down. 456 complete → 450 + 6 partial, with no other entry moving, which is the proof the derivation agrees with the rest. 16 now-inert override lines deleted (Decision #29); the surviving 39 are gated by `test_shape_rule_autoflip.py` so a contradicting override fails the build rather than quietly winning. Ops the catalog does not own (~169 Python-reference/host-API) are deliberately untouched. | Frontend §U2 | 2w |
| W1.3 | **Landed 2026-08-03.** `--tessera-record-metadata` + `--tessera-verify-metadata-obligation`: the snapshot rides in the IR as a module attribute, so record → lower → verify is ONE `tessera-opt` invocation and is lit-testable (a `PassInstrumentation`, the more obvious idiom, is registered in the driver and could not be fixtured — an unfixturable verifier is what Decision #29 rejects). Comparison is per function and normalized to the attribute's last dot-component, so `tessera.layout` → `tile.layout` is not a drop; `shape`/`dtype` are untracked because they live in types. **Found a live bug on its first real program:** `TileIRLoweringPass` has two `tile.mma` producers and only the fused K-step forwarded `numeric_policy`, so the main matmul path stated the accumulator contract at Graph IR and lost it one level down — fixed. Five fail-closed refusals incl. STALE_DECLARATION (a declared drop that did not happen) and NO_SNAPSHOT (an unrun check must not look like a passed one); `not_yet_carried:<item>` keeps declared debt attributable. | IR Stack §U5 | 2w |
| W1.4 | **Landed (PB stack, 2026-08-03) — row was stale.** `python/tessera/ga/ops.py` derives grades before backend dispatch via `_blade_mask_for_grades` + `_product_grade_contract`, closing the Decision #29 violation the plan itself cites (`MultivectorSpec.grades` reaching no consumer). `GradeFusion.cpp` gained `InputGradeFusionPattern`, attaching `tessera.clifford.input_grades_lhs/rhs` — the MLIR half: `output_grades` prunes the Cayley table by which RESULTS are wanted, `input_grades` by which INPUTS can be non-zero. Fixtures `input_grade_fusion{,_prunes}.mlir`, including the Decision #10a negative case where the correct output is NO annotation. | GA/EBM §2.1 | 1w |

**Exit:** no true Tile primitive has an unexplained `AnyType`; a mismatched
`tile.mma` fails verification; every compatibility exception names its owning
level-migration item; no op reaches the `operand_types[0]` fallback; no Decision
#15a attribute drops across a boundary without a recorded reason.

> **W1.1 is a high-leverage contract project, not a mechanical ODS edit.** It is
> the precondition for W2.4, W3.2, W3.3, and every backend having a real contract
> to lower against, but it must land incrementally with producer/consumer and
> per-architecture variant coverage.

### W2 — Build the analysis layer *(8 weeks · depends on W0)*

Root cause B. One framework, then each analysis is a transfer function rather
than a subsystem.

| # | Item | Source | Effort |
|---|---|---|---|
| W2.1 | **Graph IR dataflow framework** on MLIR `DataFlowSolver`. Three required properties: **fail closed** (unprovable ⇒ ⊤, consumers treat ⊤ as unsafe), **recomputable/invalidated** (queries against current IR, not decoration-time facts), **queryable from C++ and Python** (or duplication regrows) | Sweep §3 | 3w |
| W2.2 | Effects derived from **traced IR**; retire `_EffectVisitor`; reconcile with `EffectAnnotationPass` | Frontend §U7 | 2w |
| W2.3 | Differentiation activity analysis (`ActivityInterface`) as a client | Autodiff D3 | 2w |
| W2.4 | Collapse six `*LegalityPass` → ODS constraints (post-W1.1) + one `TileDataflowLegalityPass` | IR Stack §U2 | 1w |

**Exit:** an aliased/indirect RNG call is detected by effect inference; an
inactive branch's adjoint is provably **not emitted** (`CHECK-NOT` fixture per
#10a); six legality passes are one.

> W2.2 depends on W3.1's tracer promotion for full value, but can land against
> the existing Apple-GPU tracer first and widen with W3.1.

### W3 — Collapse the duplications *(10 weeks · depends on W1, W2)*

Root cause C. Now possible, because the surviving path can carry what the deleted
path was carrying.

| # | Item | Source | Effort |
|---|---|---|---|
| W3.1 | **One differential harness**, then promote the tracer to the only frontend; delete `_OpExtractor` | Frontend §U1 + IR Stack §U3 | 4w |
| W3.2 | One lowering per boundary: converge Graph→Schedule→Tile on MLIR; Python spine demoted to oracle | IR Stack §U3 | 3w |
| W3.3 | Split the Tile dialect by level: primitives stay `tile.*`; whole-kernel ops → Graph IR / `tessera.kernel.*`; domain ops → `tessera_ebm`; `svd`/`qr`/`cholesky`/`lu` → linalg solver | IR Stack §U4 | 2w |
| W3.4 | Decompose `JitFn` (11 `_native_*_backward` → `emit/candidate.py` candidates behind `@f__bwd`); split `__init__.py`'s 315 nested defs into `tessera/ops/` | Frontend §U5–U6 | 3w |
| W3.5 | Finish `NewtonAutodiff`'s IFT body (`dF/dx = -(dR/dx)⁻¹dR/du`) — emits real `residual` + `linear_solve` ops | Autodiff §B8 + OT R2 | 2w |
| W3.6 | Batched operands in `ExpandProductTable`; connect `RotorSandwichFold`'s marker to a consumer | GA/EBM §1.3 | 2w |
| W3.7 | **Define ROCm producer ownership per package family** across registered C++ generators, compatibility Target-IR text, and `emit/rocm_hip.py` candidates. Add differential gates and retire only producers proven duplicate; preserve C++ MLIR→ROCDL/HSACO as the canonical native spine | Target §X6 | 2w initial inventory/gate |

**Exit:** one frontend, one lowering per boundary, no target string in `jit.py`,
every `tile.*` op is a tile primitive.

### W4 — The control-flow program *(10 weeks · depends on W1.2, W2.1, W3.1)*

**This is the plan's most important structural correction.** Dynamic control flow
is blocked at three independent layers — frontend (no CFG ⇒ no φ), shape system
(no symbolic dims), autodiff (`AutodiffPass` hard-errors on nested regions).
Fixing any one alone produces **zero observable capability**. They must ship as
one program with one gate.

| # | Item | Effort |
|---|---|---|
| W4.1 | Frontend emits structured regions (`scf.*`, `tessera.control_*`) via the tracer + `control.py` hooks | 2w |
| W4.2 | Symbolic dims: `Dim`/`DimProduct` → `AffineExpr` + `presburger`; converge the Python shape system onto the `tessera.dim_bindings` carrier `SymbolicDimEqualityPass` already checks | 3w |
| W4.3 | `RegionAdjointInterface` + reverse mode over `scf.for`/`if`/`while` and `tessera.control_*` | 5w |

**Exit (single gate):** *a `@jit` function containing a data-dependent loop
compiles, differentiates, and executes, with numerical agreement against the
Python oracle.*

### W5 — Decisions become measured *(9 weeks · depends on W1, W3)*

Root cause: L5, "a constant where a measured decision belongs." Every item routes
an existing hardcoded choice through the arbiter Decision #28 already built.

| # | Item | Source | Effort |
|---|---|---|---|
| W5.1 | Residual policy as an arbiter axis (SAVE/RECOMPUTE/HYBRID per `(op, bucket, dtype, target)`); Revolve/treeverse for counted loops; delete `EBMCheckpointInnerLoop` | Autodiff D5 + GA/EBM §1.5 | 4w |
| W5.2 | Scheduling decisions at Schedule IR — tile sizes, stage counts, raster order, warp roles chosen from `fusion_core` cost models via the measured arbiter, not from `--tile-q=64` | Frontend §U3 + IR Stack §U6 | 5w |
| W5.3 | Generic fusion region discovery over a legality oracle (a W2.1 client); keep the measured cost models | Sweep §F3 | *(folded into W5.2)* |
| W5.4 | Sharding **propagation** (GSPMD/Shardy-style) — annotate a few tensors, infer the rest | Sweep §F4 | 4w |
| W5.5 | Rule-table-driven canonicalization (PDL/PDLL). **Defer equality saturation** until the rule table is large enough that ordering demonstrably costs something | Sweep §F5 | 3w |

**Exit:** no tile size, residual policy, or fusion boundary is chosen by a
constant; sharding a model requires O(few) annotations, not O(layers).

### W6 — Exceed the state of the art *(14 weeks · depends on W2, W4)*

| # | Item | Source | Effort |
|---|---|---|---|
| W6.1 | Forward mode in the compiler (`TangentInterface`) — cheapest large capability; unlocks exact HVP, `jacfwd`, and W6.3 | Autodiff D2 | 3w |
| W6.2 | Sparse AD — sparsity detection + coloring (client of W2.1/W4.2). PyTorch, TF, and JAX all lack this | Autodiff D7 | 5w |
| W6.3 | Taylor/jet mode over Weil algebras on a **new generic finite-multiplication-table substrate** potentially shared with GA. The current `ga/signature.py` is Clifford-specific (blade XOR, metric signs, anti-commutation) and cannot represent arbitrary commutative nilpotent Weil algebras | Autodiff D6 | research estimate required |
| W6.4 | Table-driven GA kernel synthesis via `emit/`; then PGA `Cl(3,0,1)` | GA/EBM §2.3–2.4 | 5w |

**Exit:** a defensible "exceeds SOTA" claim with a benchmark behind it — sparse
Jacobian scaling `O(colors)` not `O(rows)`; order-`k` derivatives sharing the
tuned GA kernels.

> W6.4 can supply useful table-lowering machinery, but W6.3 still requires a
> generic algebra representation and AD semantics. Treat reuse as a design
> hypothesis to prove, not as a sequencing-based cost reduction.

### Riemannian OT — re-scoped as validation, not a track

The [OT plan](RIEMANNIAN_OT_PLAN.md) proposed R0–R5 at ~14 weeks. Integrated, most
of it is **already funded by W1–W5**:

| OT need | Provided by |
|---|---|
| manifold as a hard dispatch key (H1) | W0.1 |
| remove inert EBM checkpoint policy from the default pipeline (H2) | W0.2; demand-aware loop rematerialization remains W5.1 |
| geometric primitive layer (R1) | new — 1.5w, first consumers are `ebm/geo_sampling.py` and `hyperbolic.py` |
| `stop_gradient` + implicit diff (R2) | W3.5 + W2.3 |
| `c_transform` fused loop (R3) | W4 (control flow) + W5.1 (residual policy) |
| backend lanes (R4) | W5.2 (schedule decisions) + existing EBM Langevin kernel precedent |
| oracles (R5) | W2.1 (fail-closed analyses) + the existing Evaluator |

**Net new OT scope: ~4 weeks** (the geometric primitive layer plus the RNOT
composite ops), down from 14. And RNOT becomes the plan's **acceptance test**: a
2500-step manifold inner loop with a stop-gradient boundary exercises W1 (typed
tiles), W2 (activity + liveness), W4 (control flow), and W5 (residual policy) in
one workload. If RNOT runs at wall-clock parity with RCPM, the plan worked.

---

## 5. Dependency graph

```
W0 ─────────────────────────────────────────────────────────────►  (start now)
 │
 ├─► W1 (declarations binding) ─────────┬──────────────────────────┐
 │      W1.1 type Tile IR ──────────────┤                          │
 │      W1.2 shape-rule registry ───────┼──► W4 (control flow) ────┤
 │                                      │      W4.2 needs W1.2     │
 └─► W2 (analysis layer) ───────────────┤                          │
        W2.1 dataflow framework ────────┼──► W4.3 needs W2.1       │
        W2.2 effects ───────────────────┤                          ├─► W6
        W2.3 activity ──────────────────┘                          │   (SOTA)
                │                                                  │
                └─► W3 (collapse duplication) ──► W5 (measured) ───┘
                       W3.1 one frontend ──► W4.1
```

Critical dependency chain: **W0 → W1.1 → W2.1 → W3.1 → W4 → W5**. The earlier
40-week/63-week totals are not retained as commitments: W1.1 is new type-system
design and W6.3 needs research scoping, while the rejected blanket ROCm
migration removed six weeks of unjustified work.

---

## 6. Three budget levels

**Minimum — W0 only.** Closes verified fail-open and invalid-value paths,
preserves the correct numerical fallback for untraceable EBM callbacks, removes
inert checkpoint policy from the default pipeline, corrects architecture
documentation, removes a duplicate-dialect trap, and upgrades the generic
Target-IR contract test. Adopts the six governance rules so nothing regrows.

**Recommended scope — W0 + W1 + W2 + W3.1; re-estimate after the W1.1 design
spike.** Root causes A and B are fixed,
the frontend duplication is gone, and every subsequent piece of work becomes
cheaper rather than adding to the pile. This is the point at which the compiler
stops accumulating parallel systems. If one number is chosen, choose this.

**Full — W0…W6, re-estimate after W1.1 and W6.3 design spikes.** ROCm ownership/inventory work is host-free;
subsequent kernel-producing migrations are hardware-routed individually (§6a).
Adds the control-flow capability, measured decisions, and two defensible
exceeds-SOTA claims.

## 6a. Fleet routing — what must run on which box

**Core compiler work is driven on the Strix Halo box (decided 2026-08-02).**
`AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`, Ubuntu 24.04 under WSL2, 32 threads,
62 GB RAM, LLVM/MLIR 23 at `/usr/lib/llvm-23`, `gfx1151` visible to `rocminfo`.
It is both faster and larger-memory than the Mac M1 Max for `tessera-opt`
rebuilds, and it is the only box in the fleet with an executing GPU lane — so
compile-time contract work and its hardware gate live on the same machine. The
Mac is retained for Apple-backend work, which cannot move.

Most of this plan is compile-time contract work and runs anywhere. Two items are
hardware-bound, and one of them is the highest-risk item in the plan.

| Work | Box | Why |
|---|---|---|
| W0, W1 (typing, enums, shape rules), W2 (analyses), W3.1–W3.4 | **Strix Halo (primary)** | ODS, `tessera-opt`, lit, unit tests. No device needed; tightening a type is a compile-time change. Runs on the Mac too, but the primary is where the gates are expected to be green. |
| Any item touching the Apple backend or Apple lit fixtures | **Mac M1 Max** | Metal/Accelerate toolchain is Mac-only; this is the one thing that cannot be retargeted. |
| W0.10 build branch — `tessera_x86` dialect | **Strix Halo** for ODS/lit; **AVX-512 execution proof also here**; AMX execution proof has **no box in the fleet** | Zen 5 has AVX-512 (`avx512f` confirmed on this host) but **AMX is Intel-only** — the earlier routing to a "Zen5 box for AMX" was wrong. The NR2 Pro's Core Ultra 7 has neither. Anything AMX-gated is currently unexercised (see the note below). |
| **W3.7 — ROCm producer ownership + differential gate** | **Strix Halo** — inventory/IR equivalence is host-free, and gfx1151 for each later producer change is on the same box | The initial slice does not change kernels. Any retirement or migration that changes generated code requires execute-and-compare on the owning device. |
| W4 (control flow) end-to-end gate | **Strix Halo / gfx1151** | It has the broadest compiler-generated + hardware-verified op coverage. |
| ROCm arch breadth (gfx950 / gfx1201 / gfx1250) | deferred — no silicon | MASTER_AUDIT P2, unchanged. |

**AMX has no hardware behind it.** `tests/device/x86/test_amx_int8_gemm.py` and
`scripts/run_x86_amx_release_gate.sh` (merged in #489) gate on AMX capability,
which no current fleet box reports. The gating is honest, but the lane is
unexercised; do not let W0.10 lean on an AMX execution proof it cannot obtain.
Native x86 proof on this box means AVX-512.

**Two sequencing consequences.** First, W3.7 reuses the differential-harness
design from W3.1/W3.2 and records one owner per package family before anything
is deleted. Second, W1.1's ROCm typing may expose invalid assumptions in
registered generators; fix those compile-time contracts host-free, and require
gfx1151 evidence only when a generated kernel or selected producer changes —
both halves now on the same machine.

**What to cut first if squeezed:** W6.4 (GA synthesis / PGA) and W5.5
(canonicalization rule tables) are the most deferrable — genuine value, no
downstream dependents. **What never to cut:** W0.5 and W0.8 (an hour and a day)
— a wrong architecture decision and absent governance are what let all of this
accumulate.

---

## 7. Risks

| Risk | Wave | Mitigation |
|---|---|---|
| **W3.1 is a broad behavior change** — the AST frontend is the default on every non-Apple target | W3 | The differential harness ships *before* the switch, not after; promote per-target with the harness green |
| **W4 will exceed its estimate.** Region adjoints are the hardest item here and structured reverse mode is genuinely difficult | W4 | Land W4.1+W4.2 with a forward-only gate first, so partial progress is observable before W4.3 |
| **W1.1 touches every Tile-consuming backend** | W1 | Parameterized types and verifier contracts can invalidate producers and consumers; land per primitive/variant with parser, verifier, lowering, and backend fixtures |
| **Waves 1–3 produce no user-visible feature.** Fifteen weeks of "the compiler now enforces what it already said" is hard to fund | all | RNOT (§4) is the visible acceptance workload; state the intermediate gates as capability claims, not cleanup |
| **The governance rules are ignored under delivery pressure** | all | #29 and #31 are drift-gateable; make them tests, not conventions |
| **Someone starts at W3** (deleting duplications first, because they are the most visible waste) | — | It fails: the surviving path cannot yet carry what the deleted one carried. Ordering A→B→C is the plan's core claim |

---

## 8. What this plan does not cover

Not examined across the seven reviews, and therefore not planned:

- ~~Target IR dialects~~ — **reviewed 2026-08-02**
  ([TARGET_IR_REVIEW.md](TARGET_IR_REVIEW.md)). The `AnyType` finding **does**
  repeat, in `ROCM_{MFMA,WMMA}` and the NVIDIA mma ops; W1.1 grew 3w → 5w
  accordingly, and four new items landed (W0.9, W0.10, W1.1b, W3.7).
- The bodies of the 67 `GenerateROCM*Kernel` passes — these need review on the
  host for ownership classification, with gfx1151 required before any
  kernel-producing migration is accepted.
- `emit/nvidia_cuda.py` (4722 lines) internals. `emit/rocm_hip.py` was inspected
  far enough to establish that it is an arbiter candidate/runner surface, not a
  drop-in replacement for the canonical MLIR→ROCDL package spine.
- Spectral and TPP solver families; the collectives and neighbors dialects;
  the RubinCPX backend.
- Quantization numerics; the KV-cache and memory model.
- The Evaluator program (`EVALUATOR_PLAN.md` §9.5) — it is a consumer of this
  work, not a subject of it.
- `WarpSpecializationPass` and `AsyncCopyLoweringPass` bodies.

Absence from this plan is not a clean bill of health.
