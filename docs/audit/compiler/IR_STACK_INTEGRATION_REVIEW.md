---
last_updated: 2026-08-02
audit_role: reference
scope: TesseraOps.td, ScheduleMeshPipelineOps.td, Tile/TileOps.td, tile_opt_fa4 (Attn/Queue), TileIRLoweringPass, GraphToSchedulePass, the *LegalityPass family, and the Python schedule_ir/tile_ir/target_ir spine
companions: FRONTEND_GRAPH_SCHEDULE_REVIEW.md · COMPILER_ARCHITECTURE_SWEEP.md · AUTODIFF_ARCHITECTURE_REVIEW.md
---

# Graph IR → Schedule IR → Tile IR Integration Review

The seams between the three IR levels: what each level owns, what survives the
transition, and what has to be re-derived because it didn't.

The headline is short enough to state up front: **the Tile dialect defines nine
precise semantic types and then declares every operation as
`Variadic<AnyType> → Variadic<AnyType>`.** Everything else in this review is
either a consequence of that or a duplication problem.

Status truth stays with the generated dashboards (Decision #26).

---

## 0. What the stack actually is

Documented in `CLAUDE.md`:

```
Graph IR ──► Schedule IR ──► Tile IR ──► Target IR
```

What the source says, in
[`LowerScheduleToTarget.cpp`](../../../src/compiler/tile_opt_fa4/lib/Conversion/TesseraScheduleToTarget/LowerScheduleToTarget.cpp):

> **HONEST SCAFFOLD — NOT a working lowering.**
> The real Schedule -> Tile -> Target lowering in Tessera is performed by the
> **Python compiler spine** (`tessera.compiler.schedule_ir`,
> `tessera.compiler.tile_ir`, `tessera.compiler.target_ir`), which is what
> `@tessera.jit` uses…

So the canonical lowering path below Graph IR is Python. The MLIR dialects and
passes are real and lit-tested, but they are a *parallel* implementation of the
same boundaries, not the production one. Concretely, at each boundary:

| Boundary | Python (canonical per `@jit`) | C++/MLIR | Status |
|---|---|---|---|
| Graph → Schedule | `schedule_ir.lower_graph_to_schedule_ir` | `GraphToSchedulePass` — defined inside `PassPipelinesPM11.cpp` but compiled into `TesseraPM` | two linkable impls; no differential coverage |
| Schedule → Tile | `tile_ir.lower_schedule_to_tile_ir` | `TileIRLoweringPass` (real, pattern-matches `flash_attn`/`matmul`) | two impls |
| Schedule/Tile → Target | `target_ir` (2004 lines) | `LowerScheduleToTarget` — scaffold that fails loudly | Python only |

This is the [frontend review](FRONTEND_GRAPH_SCHEDULE_REVIEW.md)'s G7 finding,
extended: it is not one duplicated boundary, it is **all three**.

---

## 1. Architectural findings

### T1 — The Tile dialect has begun adopting nine types, but its core compute and copy envelopes remain open

[`TileOps.td`](../../../src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td)
declares a complete and well-chosen type vocabulary:

```
!tile.async_token   !tile.tile        !tile.fragment    !tile.buffer
!tile.pipeline_state  !tile.tma_descriptor  !tile.mbarrier
!tile.mbarrier_token  !tile.tmem        #tile.barrier
```

That is exactly the right vocabulary for a tile-level IR — fragments, buffers,
TMEM, TMA descriptors, mbarriers, pipeline state, async tokens.

The statement that the operations use none of these types is stale.  The current
ODS uses typed results/operands for `tile.alloc`/`dealloc`, pipeline state, TMA
descriptors, mbarriers, TMEM handles, tile values, and fragment pack/zero/unpack.
That is meaningful progress and must be preserved.

The most important portable compute and copy envelopes are still open:

```tablegen
def Tile_MMAOp : Tile_Op<"mma"> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results   = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
}

def Tile_AsyncCopyOp : Tile_Op<"async_copy"> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results   = (outs Variadic<AnyType>:$outputs);
}
```

`AnyType` still appears 71 times (66 as `Variadic<AnyType>`), concentrated in
value-lane, whole-kernel, domain, and compatibility operations.  Some of those
uses are temporary consequences of the level-mixing in T4, not all the same
typing defect.

`tile.mma` is the single most type-sensitive operation in the compiler —
fragment layout, accumulator precision, MMA shape, and register-file assignment
all ride on its operand types — and it is `Variadic<AnyType> → Variadic<AnyType>`.
`tile.async_copy` is the operation whose entire purpose is moving between memory
spaces, and it names no memory space, no shape, and no dtype.

This remains a high-value instance of the recurring pattern "metadata is
defined and only partially consumed."  The fix is not only changing operand
declarations: `!tile.fragment` and `!tile.buffer` are currently opaque, so
carrying element type, shape, layout, memory space, and numeric policy requires
a variant-aware type design and migration plan.  Whole-kernel/domain ops should
not be forced into a false primitive signature merely to drive `AnyType` to
zero; T4 must move them to their owning level.

Two consequences follow, and they are §T2 and §T6.

### T2 — The legality-pass archipelago is the cost of T1

Because the operations are untyped, every invariant that the type system would
enforce for free has to be re-derived by a pass. There are now six:

`TilePipelineLegalityPass` · `WarpSpecLegalityPass` ·
`TileBarrierReuseLegalityPass` · `LayoutLegalityPass` ·
`IRContractLegalityPass` · `PipelineScheduleLegalityPass`

They saw it happening. `LayoutLegalityPass`'s own header:

> The skeleton is named so future rules … extend a single pass body instead of a
> **one-rule-per-pass archipelago**.

And that pass is currently *"a SKELETON with **one first rule**"* — a
`tessera.cast` layout-name accept-set check. So the pass created to prevent an
archipelago is itself a one-rule pass in an archipelago of six.

The distinction that matters: a **type** rejects a malformed program at
construction, everywhere, with no pass ordering. A **legality pass** rejects it
only where the pass runs, only if the pass runs, and only for the rules someone
remembered to write. Six passes' worth of rules would mostly be one line of ODS
each.

### T3 — Every level boundary is implemented twice, and the canonical one is Python

Per §0. The specific costs:

- **Divergence is undetectable.** There is no differential test asserting that
  `lower_schedule_to_tile_ir` and `TileIRLoweringPass` produce equivalent IR from
  the same input. They can drift silently and almost certainly have.
- **`GraphToSchedulePass` has muddled ownership.** CMake already compiles
  `PassPipelinesPM11.cpp` into the linkable `TesseraPM` library, and tests link
  that target. The defect is that pass implementation, factory, registration,
  and driver-pipeline assembly share one source file with no focused fixtures or
  differential gate against the Python implementation.
- **The Python spine is untyped too.** `tile_ir.py` is 482 lines and mentions
  `dtype`/`shape`/`layout` seven times. So the canonical lowering carries even
  less type information than the ODS it mirrors.

### T4 — The Tile dialect mixes three abstraction levels

Its 58 operations fall into three clearly different tiers:

| Tier | Examples | Belongs at Tile level? |
|---|---|---|
| **True tile primitives** | `tile.async_copy`, `tile.wait_async`, `tile.mma`, `tile.alloc`, `tile.view`, `tile.fragment_pack`, `tile.mbarrier_init`, `tile.tmem_load`, `tile.pipeline_advance`, `tile.tma_descriptor` | ✅ yes — this is the dialect |
| **Whole-kernel opaque ops** | `tile.attention_kernel`, `tile.attention_backward_kernel`, `tile.softmax_kernel`, `tile.moe_dispatch_kernel`, `tile.grouped_gemm_kernel`, `tile.paged_attention_kernel`, `tile.replay_ssm_decode_kernel`, `tile.norm_kernel`, `tile.rope_kernel` | ❌ these are Graph-IR-level ops wearing a Tile prefix |
| **Domain ops** | `tile.ebm_langevin_step`, `tile.ebm_energy_quadratic`, `tile.ebm_refinement`, `tile.ebm_partition_exact`, `tile.ppo_policy_loss`, `tile.svd`, `tile.qr`, `tile.cholesky`, `tile.lu` | ❌ domain leaking down three levels |

`tile.svd` next to `tile.mbarrier_init` is a level confusion: SVD is not a tile
operation, and whatever `tile.svd` means, it cannot be lowered by the same
machinery that lowers an mbarrier init.

The practical harm is that "Tile IR" no longer has a definition. A pass author
cannot answer "what may I assume about an op in this dialect?" — the answer
ranges from *a single tensor-core instruction* to *a whole MoE dispatch*. That
uncertainty is why passes pattern-match on op *names* rather than on interfaces
(T7), which is why they don't generalize.

### T5 — Duplicate dialect ODS, both declaring the same dialect

Two files define `def Tessera_Queue_Dialect` with `let name = "tessera.queue"`
and `let cppNamespace = "::tessera::queue"`:

- `tile_opt_fa4/dialects/tessera_queue/Queue.td` (39 lines) — types
  `TileQueueType`/`TokenType`, **no mnemonics**, includes
  `mlir/IR/DialectSpecification.td`
- `tile_opt_fa4/include/tessera/Dialect/Queue/Queue.td` (56 lines) — types
  `TileQueue`/`Token` **with** mnemonics

`CMakeLists.txt:49` builds only the `include/` one. The `dialects/` copy is dead,
looks authoritative, has a different type vocabulary, and would trigger a
duplicate-dialect-registration failure if ever added to the build.

Same shape for Attn: `dialects/tessera_attn/Attn.td` (28 lines) vs
`include/tessera/Dialect/Attn/Attn.td` (258 lines).

This is a trap for the next person, not a live defect. It is also an hour's work.

### T6 — There is no metadata contract across level boundaries

Decision #15a makes six attributes normative — `shape`, `dtype`, `layout`,
`device`/`target`, `distribution`, `numeric_policy`. Graph IR carries all six
(`IRType`, `NumericPolicy`, `GraphIRMesh`, layout on `tensor_ir_type`).

Below Graph IR there is **no carrier for any of them**. Schedule IR uses
`AnyType` 10 times in a 153-line dialect and passes attributes through
`_base_attrs`/`_copy_attrs` dictionaries. Tile IR uses `AnyType` 70 times. So:

- `numeric_policy` — the accumulator precision, the `tf32` math mode — has no
  representation at the level where the MMA instruction is selected. The one
  place it decides something is the one place it isn't.
- `layout` survives only as a string attribute checked by a one-rule skeleton pass.
- `distribution` survives as mesh-region *structure* but not as a type.

The question "is this fragment in the layout `tile.mma` expects?" is not
answerable at Tile IR. That is the question Tile IR exists to answer.

### T7 — Schedule → Tile is op-pattern-matching, not scheduling

`TileIRLoweringPass` matches `tessera.flash_attn` inside `schedule.mesh.region`
and expands it to a fixed FA-4 skeleton, plus a `tessera.matmul` case. Tile sizes
come from pass options (`--tile-q`, default 64; `--tile-kv`, default 64) and the
SM version from `--sm`.

The Python side is the same shape — `_flash_attention_pipeline`,
`_sequence_mixer_pipeline`, `_msa_kv_outer_sparse`, `_media_op`, `_jepa_op`.

So the Schedule→Tile transition makes no scheduling *decision*. It is a macro
expansion keyed on op name, with the schedule parameters supplied from outside.
This is the [frontend review](FRONTEND_GRAPH_SCHEDULE_REVIEW.md)'s G8 seen from
below: Schedule IR doesn't choose, and Tile IR doesn't either — the choice
happens in `@jit` kwargs, `autotune_v2`, or a pass-option default, and both IR
levels transcribe it.

---

## 2. What is right, and should be the model

Stated because three of these are the discipline the other reviews were asking
for, already present here.

- **The honest scaffold.** `LowerScheduleToTarget` was an empty no-op that
  silently succeeded; it now `emitError`s and `signalPassFailure()`s, with a
  header explaining exactly why. *"An unimplemented lowering must never report
  success."* That is Decision #21 done right, and it is the model for every
  annotation-only pass flagged in the [GA/EBM review](../domain/GA_EBM_ARCHITECTURE_REVIEW.md).
- **Typed synchronization state.** `!tile.pipeline_state` and `!tile.mbarrier`
  as SSA values, with `#tile.barrier<kind=tma,expect=…>` — barriers as a
  correctness property carried in the IR rather than a scheduling artifact.
  `TileBarrierReuseLegalityPass`'s header says it outright: *"Barriers are a
  layout-reuse correctness property, not a scheduling artifact."* This matches
  where the tile-compiler literature has converged and is ahead of most
  contemporaries. The irony of T1 is that the *control* state is typed and the
  *data* is not.
- **`SymbolicDimEqualityPass` — and a correction to an earlier review.**
  [Sweep F2](COMPILER_ARCHITECTURE_SWEEP.md) said the shape system is name
  equality. That is accurate about `python/tessera/shape.py`, and **incomplete
  about the compiler**: MLIR carries `tessera.dim_bindings` (equations like
  `"D = H * Dh"`) and `tessera.dim_sizes`, and this pass evaluates the bindings
  and checks transpose-permutation, reshape-product, and matmul-K contracts with
  named diagnostics. The symbolic-dim *carrier* exists at the IR level; the
  Python shape system doesn't use it. That strengthens sweep item 10 rather than
  weakening it — the target to converge on is already in the tree.

---

## 3. Algorithmic and architectural updates

### U1 — Finish typing the true Tile primitives

Replace `Variadic<AnyType>` with the types the dialect already defines:

```tablegen
def Tile_AsyncCopyOp : Tile_Op<"async_copy"> {
  let arguments = (ins Tile_BufferType:$src, Tile_BufferType:$dst,
                       Optional<Tile_MBarrierType>:$barrier);
  let results   = (outs Tile_AsyncTokenType:$token);
}

def Tile_MMAOp : Tile_Op<"mma"> {
  let arguments = (ins Tile_FragmentType:$a, Tile_FragmentType:$b,
                       Tile_FragmentType:$acc);
  let results   = (outs Tile_FragmentType:$out);
}
```

Preserve the typed alloc/pipeline/TMA/mbarrier/TMEM vocabulary already landed.
Then parameterize `!tile.fragment` and `!tile.buffer` on the attributes that
actually decide codegen — element type, tile shape, layout, memory space, and
(for the accumulator) `numeric_policy`. That single step:

- makes malformed MMA operand pairings unrepresentable rather than
  pass-detectable,
- gives `numeric_policy` a carrier at the level where it decides the instruction
  (T6),
- removes the need for most of the six legality passes (T2),
- and gives every backend a typed contract to lower against instead of a
  positional variadic.

This is a bounded type-system extension, not a mechanical substitution.  Before
tightening an op, inventory every producer/consumer on ROCm, NVIDIA, Apple, and
x86; define variant-aware fragment roles and dtypes; and retain an explicit,
reviewed compatibility exception only where the op is scheduled to move out of
Tile IR under U4.

### U2 — Collapse the legality archipelago

After U1, most rules become ODS constraints or per-op `verify()`. What genuinely
remains is *cross-op dataflow* — barrier reuse, pipeline-state ordering,
warp-role consistency — and that is one analysis, not four passes. Land it as a
client of the [sweep's](COMPILER_ARCHITECTURE_SWEEP.md) proposed dataflow layer
(item 8), which is exactly the lattice-plus-fixpoint shape these checks need.

Target: six legality passes → ODS constraints + one `TileDataflowLegalityPass`.

### U3 — One lowering per boundary

Pick the surviving implementation per boundary and delete the other. Given the
Python spine is what `@jit` uses and what the suites exercise, the honest
sequence is:

1. Build the **differential harness first** — same input, both paths, compare
   emitted IR. This is the only way to know what the C++ passes actually do
   differently, and it is the same harness the
   [frontend review](FRONTEND_GRAPH_SCHEDULE_REVIEW.md) E2 needs for trace-vs-AST.
2. Move `GraphToSchedulePass` out of `PassPipelinesPM11.cpp` into a dedicated
   library-owned source/header so its API and fixtures are independently owned;
   it is already linkable through `TesseraPM`.
3. Converge on MLIR as the surviving path for Graph→Schedule→Tile (it is where
   the types from U1 live), keeping the Python spine as the reference/oracle —
   the same demotion the [autodiff review](AUTODIFF_ARCHITECTURE_REVIEW.md) M1
   prescribes for the tape.

### U4 — Split the Tile dialect by level

- Keep `tile.*` for true tile primitives (tier 1 in T4).
- Move whole-kernel ops (`tile.attention_kernel`, `tile.moe_dispatch_kernel`, …)
  to Graph IR or a `tessera.kernel.*` dialect — they are op-level, not tile-level.
- Move domain ops (`tile.ebm_*`, `tile.ppo_policy_loss`) to their own dialects,
  which already exist (`tessera_ebm`).
- Move `tile.svd`/`tile.qr`/`tile.cholesky`/`tile.lu` to the linalg solver
  dialect.

Then "Tile IR" has a definition again, and passes can dispatch on op
*interfaces* rather than op *names*.

### U5 — A metadata contract that survives lowering

Make Decision #15a's six attributes a **lowering obligation**: each boundary pass
must either carry each attribute forward in the target level's type/attribute
vocabulary, or record an explicit, named reason it is dropped. Add a boundary
verifier that fails when an attribute vanishes without a recorded reason.

This is the same fail-closed rule proposed as Decision #21a in the
[OT plan](RIEMANNIAN_OT_PLAN.md), applied to lowering rather than dispatch:
information loss across a level boundary must be *declared*, not silent.

### U6 — Make Schedule IR the level where scheduling happens

Per [frontend review](FRONTEND_GRAPH_SCHEDULE_REVIEW.md) U3, and it lands here:
Schedule→Tile should consume a cost model and *choose* tile sizes, stage counts,
raster order, and warp roles, using `fusion_core`'s cost models and
`emit/candidate.py`'s measured arbiter — instead of reading `--tile-q=64` from a
pass option. `TileIRLoweringPass` becomes the *expander* of a chosen schedule
rather than the place a default schedule is invented.

### U7 — Delete the duplicate `dialects/tessera_{queue,attn}/*.td`

One hour. Removes a trap.

---

## 4. Phasing

| Phase | Contents | Effort | Gate |
|---|---|---|---|
| **I0** | U7 duplicate ODS deletion; split `GraphToSchedulePass` into a dedicated library-owned source/header with focused fixtures | 3d | one ODS per dialect; pass ownership explicit + lit-tested |
| **I1** | **U1 — finish typing true Tile primitives**; parameterize `!tile.fragment`/`!tile.buffer`; inventory compatibility exceptions | 3w design/migration estimate | no unexplained `AnyType` on true primitives; a mismatched `tile.mma` fails verification |
| **I2** | U2 — legality passes → ODS constraints + one dataflow pass | 2w | six passes → one; no rule lost |
| **I3** | U4 — split the Tile dialect by level | 3w | every `tile.*` op is a tile primitive |
| **I4** | U5 — metadata lowering obligation + boundary verifier | 2w | no attribute drops without a recorded reason |
| **I5** | U3 — differential harness, then one lowering per boundary | 5w | one implementation per boundary |
| **I6** | U6 — scheduling decisions at Schedule IR | (= frontend E6) | tile sizes chosen by measurement |

**I0 + I1 is about four weeks and is the highest-leverage block in this review.**
Typing the Tile dialect is prerequisite to I2, makes I3 and I4 tractable, and
gives every backend a real contract to lower against.

---

## 5. How this changes the consolidated queue

Revisions to [`COMPILER_ARCHITECTURE_SWEEP.md §4`](COMPILER_ARCHITECTURE_SWEEP.md),
as amended by the [frontend review §5](FRONTEND_GRAPH_SCHEDULE_REVIEW.md):

1. **I0 joins Tier 0.** Three days, removes a duplicate-dialect trap and gives
   the already-linkable `GraphToSchedulePass` explicit source/header ownership
   plus focused lit coverage.
2. **I1 joins Tier 1, near the top.** It is a bounded type-system design and
   migration, and it is the
   precondition for the Tile-level half of nearly everything else. I would rank
   it immediately after the dataflow framework (sweep item 8) and alongside the
   shape-rule registry (frontend E1) — all three are "make the type/contract
   system actually enforce what it already declares."
3. **I2 becomes a client of sweep item 8**, not independent work.
4. **I5 shares the differential harness with frontend E2.** Build it once, use it
   for trace-vs-AST *and* Python-spine-vs-MLIR-pass. Budget once.
5. **I6 is frontend E6.** Same work, seen from the other side. Not additive.

Net new to the queue after de-duplication: **I0 (3d) + I1 (3w) + I2 (2w) +
I3 (3w) + I4 (2w) ≈ 11 weeks**, of which the first four weeks carry most of the
value.

---

## 6. Scope and limits

Examined: the three dialect ODS files, `TileIRLoweringPass`,
`GraphToSchedulePass`, `LowerScheduleToTarget`, the six `*LegalityPass` headers,
`SymbolicDimEqualityPass`, and the Python `schedule_ir`/`tile_ir`/`target_ir`
spine.

Not examined, and therefore not assessed: the Target IR dialects
(`TesseraROCMOps.td` at 1311 lines, `TesseraAppleOps.td` at 371,
`TesseraNVIDIADialect.td` at 172), the collectives dialect, the neighbors
dialect, `WarpSpecializationPass` and `AsyncCopyLoweringPass` bodies, and the
`tessera_attn` op semantics. The Target IR dialects in particular are larger than
Tile IR and would be the natural next review — especially to check whether the
`AnyType` pattern in T1 repeats there.
