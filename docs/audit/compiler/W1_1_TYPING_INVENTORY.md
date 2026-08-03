---
last_updated: 2026-08-02
audit_role: reference
owning_plan_item: W1.1
---

# W1.1 — Tile / Target IR typing inventory

W1.1 says, in its own text: **"Inventory every backend producer/consumer before
tightening each op."** This is that inventory. It is a precondition, not the
migration; nothing here changes an ODS file.

Status truth stays `MASTER_AUDIT.md` + `docs/audit/generated/` (Decision #26).
Every count below was measured on 2026-08-02 against the working tree and is
reproducible with the commands in §5.

---

## 1. The headline correction — W1.1 is a migration already in progress

The integrated plan's thesis table lists the Tile dialect as *"9 types declared
… `Variadic<AnyType>` 70 times"*, which reads as **untyped**. That is not what
the code does, and the difference changes how W1.1 should be scheduled.

`Tile_MMAOp::verify()` already implements a **dual-form migration** with the
intent stated in its own comment:

```
// Preserve the legacy permissive form during migration. Only the typed
// fragment form is eligible for physical cooperative-matrix lowering.
```

* **Typed form** — if any input is a `!tile.fragment`, the verifier enforces the
  full contract: a `#tile.mma_desc` attribute must be present, arity must be 3
  (or 5 for NVFP4: A, B, acc, scale_a, scale_b), the result must be
  `!tile.fragment`, and each operand must come from a legitimate fragment
  producer.
* **Legacy form** — otherwise `return success()`. Anything goes.

So the typed contract exists and bites. W1.1 is therefore **not** "design a type
system"; it is **"migrate producers onto the form that already verifies, then
delete the permissive branch."** That is a different, better-understood job, and
it is producer-ordered rather than design-ordered.

---

## 2. Who actually produces `tile.mma`

Ten C++ sites construct `tile.mma`. The measurement that matters is whether they
emit the **typed** form — which requires both a `FragmentType` operand and a
`#tile.mma_desc` attribute.

**Producer vs consumer (corrected 2026-08-02).** The first version of this
section called all ten "producers". They are not. Only **five sites in four
files** actually construct `tile.mma`, all via `OperationState` with the op name
as a string (which is why a `create<MMAOp>` search finds nothing):

| Construction site | |
|---|---|
| `TileIRLoweringPass.cpp:922`, `:984` | the generic Tile lowering — the main producer |
| `GenerateWMMAGemmKernel.cpp:465` | `viaTile ? "tile.mma" : "tessera_rocm.wmma"` |
| `GenerateWMMALinearAttnKernel.cpp:143` | same shape |
| `GenerateWMMAFlashAttnKernel.cpp:230` | same shape |

The rest are **consumers** (lowerings and legality passes) that match on
`tile.mma`. Two of them — `TileToROCM.cpp` and `NVIDIALowering.cpp` — already
handle the typed fragment form. So the migration surface is five construction
sites, not ten files.

| File | mentions `FragmentType` | mentions `mma_desc` | Form |
|---|---:|---:|---|
| `NVWGMMALoweringPass.cpp` | 0 | 0 | legacy |
| `NVIDIALowering.cpp` | 1 | 0 | legacy |
| `GenerateWMMAGemmKernel.cpp` | 0 | 0 | legacy |
| `GenerateWMMALinearAttnKernel.cpp` | 0 | 0 | legacy |
| `ROCMWaveLdsPipeline.cpp` | 0 | 0 | legacy |
| `TileToROCM.cpp` | 1 | 0 | legacy |
| `GenerateWMMAFlashAttnKernel.cpp` | 0 | 0 | legacy |
| `WarpSpecializationPass.cpp` | 0 | 0 | legacy |
| `WarpSpecLegalityPass.cpp` | 0 | 0 | legacy |
| `TileIRLoweringPass.cpp` | 0 | 0 | legacy |

**No C++ pass emits the typed form.** Zero producers set `#tile.mma_desc`.

The typed form's only producers are **Python text emitters** —
`python/tessera/compiler/nvidia_native.py` (four sites: `mma_sync`, NVFP4,
int4, and the attention lane) and `python/tessera/runtime.py` (one). They write
`#tile.mma_desc<family = "mma_sync", m = …, n = …, k = …, a = …, b = …, acc = …,
a_layout = …, b_layout = …, k_blocks = …>` into MLIR **source text** that is then
parsed.

Two consequences to design around:

1. **The typed Tile contract has no C++ producer at all.** The verifier's
   strongest branch is exercised only by text the Python side writes. This is
   the same seam `CLAUDE.md` already describes for Apple GPU — the Python and
   C++ sides are two compilers — showing up again one level down.
2. **No lit fixture pairs `!tile.fragment` with `tile.mma`.** Searching
   `tests/tessera-ir/` for files containing both returns nothing. The typed
   branch's coverage is Python-side (`test_rocm_wmma_gemm_generated.py`,
   `test_apple_threadgroup_pipeline.py`, `test_tile_fragment_compiler_path.py`)
   plus a few phase2 fixtures that use `mma_desc` without `tile.mma`.

### 2.1 The blocker — the typed form and warp-spec token sync are mutually exclusive

**Step 3b cannot start until this is resolved, and it explains why no C++
producer emits the typed form: today it is structurally impossible, not
neglected.**

Two contracts on the same op contradict each other:

* `MMAOp::verify()`'s typed branch requires **exactly 3 raw operands**
  (A, B, acc; 5 for NVFP4) — it counts `getInputs().size()` directly.
* `WarpSpecLegalityPass` requires a consumer `tile.mma` reading an async-staged
  tile to **also read a `!tile.async_token`** from that producer
  (`WARPSPEC_MMA_NOT_TOKEN_SYNCED`) — the SSA edge that gates the matrix op on
  copy completion instead of program order. `TileIRLoweringPass.cpp:922` duly
  emits four operands: A, B, and two tokens.

A producer therefore cannot satisfy both. Verified empirically — typed fragments
plus the required token edge is rejected:

```
error: 'tile.mma' op typed fragment form expects A, B, accumulator -> !tile.fragment
  %res = "tile.mma"(%fa, %fb, %acc, %tok)
       : (!tile.fragment, !tile.fragment, !tile.fragment, !tile.async_token) -> !tile.fragment
```

**The consumer side already solved this and the verifier did not learn.**
`TileToROCM.cpp:312` defines a file-local `static dataOperands()` that filters
out tile *control* types before counting, and uses it at the `tile.mma` site.
The verifier counts raw operands instead.

**Proposed resolution (not yet applied):** have `MMAOp::verify()` count *data*
operands the same way the consumer does, and promote `dataOperands()` from a
file-local static in one backend to a shared Tile helper — a private copy of the
rule in one of several consumers is itself the Decision #31 pattern. Only then
can the five construction sites migrate. This is a small change with a real
blast radius (it also relaxes the NVFP4 arity check), so it deserves its own
verification pass rather than being folded into a producer migration.

---

## 3. The untyped surface, measured

### Tile IR (`src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td`)

* **53 ops**, **9 declared types**, **66 `Variadic<AnyType>`** operand/result
  slots (70 `AnyType` occurrences in total).

Per-type reference count — occurrences of the ODS symbol in `TileOps.td`,
including its own `def` line. A count of **1 means the type is never named in
any op's ODS signature**. That is *not* the same as unused: a type can still
flow through a `Variadic<AnyType>` slot in real IR, which is exactly what
`Tile_AsyncTokenType` does (§3.1). Read this column as "how typed is the
declared surface", not "is this type dead".

| Type | refs | Note |
|---|---:|---|
| `Tile_AsyncTokenType` | **1** | Never named in an ODS signature — but see the correction in §3.1. |
| `Tile_TileValueType` | 4 | |
| `Tile_FragmentType` | 3 | the type the typed `mma` form turns on |
| `Tile_BufferType` | 3 | |
| `Tile_PipelineStateType` | 3 | |
| `Tile_TMADescriptorType` | 3 | |
| `Tile_MBarrierType` | 6 | |
| `Tile_MBarrierTokenType` | 3 | |
| `Tile_TMEMType` | 6 | |

The plan's corrected wording ("partially consumed; core `tile.mma` /
`tile.async_copy` and compatibility envelopes remain open") is accurate. The
two named core ops are both bare:

```tablegen
def Tile_MMAOp : Tile_Op<"mma"> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results   = (outs Variadic<AnyType>:$outputs);
}
def Tile_AsyncCopyOp : Tile_Op<"async_copy"> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results   = (outs Variadic<AnyType>:$outputs);
}
```

`tile.async_copy` had **no verifier at all**, so unlike `mma` it had no typed
form to migrate *onto*. `!tile.async_token` is the obvious result type, so the
two findings close each other — done in W1.1 steps 1-2, see §3.1.

### 3.1 Correction — `!tile.async_token` was *not* unused (recorded 2026-08-02)

The first version of this inventory read the reference count of **1** as
"declared and never used — a Decision #29 violation." **That was wrong, and
implementing the fix is what exposed it.**

The count of 1 measures occurrences of the *ODS symbol* `Tile_AsyncTokenType`
in `TileOps.td`, and that is accurate: no op declares the token in its ODS
signature. But the token is used **in IR**, flowing through the
`Variadic<AnyType>` slots — `tests/tessera-ir/phase2/tile_async_token_roundtrip.mlir`
and `warpspec_token_sync_legality.mlir` both produce and consume it, and
`tile.mma` takes it as a third operand for warp-spec ordering.

So the real finding is weaker and more precise: **the type is used but never
typed** — it has consumers, they just are not expressed in ODS, so nothing
verified them. Decision #29 was not being violated; Decision #30 ("derive, don't
ask") was, in the sense that the copy→wait dependency was carried by convention
rather than by a checked contract.

A second, sharper lesson for the rest of W1.1: the first verifier written here
required `async_copy` to take a destination *and* a source, modelled on
`tessera_rocm.async_copy(dst, src, bytes)`. The Tile-level convention is
different — one operand (the source), with the copied tile returned alongside
the token:

```mlir
%tile, %tok = tile.async_copy %src : (tensor<..>) -> (tensor<..>, !tile.async_token)
```

Six existing fixtures rejected the wrong rule immediately. **This is the
concrete argument for W1.1's own precondition**: a plausible cross-level analogy
is not a substitute for reading the producers. Expect the same on `tile.mma`'s
ten producers, and land each behind its fixture.

### Target IR dialects

| Dialect | `AnyType` | `Variadic<AnyType>` | declared types |
|---|---:|---:|---:|
| `tessera_rocm` | 10 | 0 | 1 (`!tessera_rocm.token`) |
| `tessera_nvidia` | 3 | 3 | 0 |
| `tessera_apple` | 12 | 12 | 0 |
| `tessera_x86` | **0** | **0** | 1 (`!tessera_x86.tile`) |

`tessera_x86` is 0/0 because W0.10 built it typed from the start — its
value-carrying ops take and return `!tessera_x86.tile`, and its negative fixture
proves the verifier rejects a dot-product whose operands never came from a tile
load. It is a useful reference for what the others should converge on, and
evidence the shape is achievable without a large redesign.

`tessera_apple` is the largest untyped Target-IR surface (12/12) and
`tessera_nvidia` the smallest (3/3).

---

## 4. Recommended migration order

Ordered by *risk removed per unit of work*, not by size:

1. ~~**`Tile_AsyncTokenType`**~~ — **done (W1.1 step 1).** Resolved by giving it
   a verified contract rather than deleting it; see §3.1 for why the original
   framing of this item was wrong.
2. ~~**`tile.async_copy`**~~ — **done (W1.1 step 2).** `AsyncCopyOp::verify()`
   and `WaitAsyncOp::verify()` now enforce the typed form when a token is
   present (exactly one token, last result, non-empty source; a waited token
   must come from a `tile.async_copy`, not a block argument or another op) and
   accept the legacy form unchanged. Fixtures:
   `tile_async_token{,_invalid}.mlir`, the latter covering wrong-producer,
   block-argument, and operand-less cases.
3. **`tile.mma` producer migration** — ten C++ producers onto the form the
   verifier already enforces, one at a time, each with a lit fixture. The
   missing regression net is **now in place** (W1.1 step 3):
   `tile_mma_typed_fragment.mlir` walks the full typed chain
   (`view` → `fragment_pack` a/b → `fragment_zero` acc → `mma` → `fragment_unpack`)
   and `tile_mma_typed_fragment_invalid.mlir` proves the contract bites on
   swapped operand roles, a mismatched instruction descriptor, a missing
   accumulator, and a missing descriptor. **The producer migration itself is
   not started.**
4. **Delete `MMAOp::verify()`'s permissive branch** — only once (3) is complete.
   This is the step that makes the contract binding; doing it earlier breaks
   every backend.
5. **Target IR dialects** — `tessera_nvidia` (3/3) first as the smallest, then
   `tessera_apple` (12/12). `tessera_rocm`'s 10 bare `AnyType` are on ops that
   already carry a real token type, so they are a smaller job than the count
   suggests.

**Do not start at (4).** It is the visible win and it is the one step that
requires all the others first — the same failure mode the plan's risk table
names for W3.

---

## 5. Reproducing these counts

```bash
TILE=src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td
grep -c 'Tile_Op<"' $TILE                  # ops
grep -c 'def Tile_\w*Type : Tile_Type' $TILE   # declared types
grep -c 'Variadic<AnyType>' $TILE          # untyped slots
grep -c 'Tile_AsyncTokenType' $TILE        # 1 ⇒ declared, never used

# tile.mma producers, and whether they emit the typed form
for f in $(grep -rl 'tile::MMAOp\|"tile.mma"' --include=*.cpp src/); do
  echo "$(basename $f) frag=$(grep -c FragmentType $f) desc=$(grep -c mma_desc $f)"
done

# is there a lit fixture with both? (currently: none)
grep -rl "tile.mma" tests/tessera-ir/ | xargs grep -l "!tile.fragment"
```

---

## 6. What this inventory does not cover

- The bodies of the 67 `GenerateROCM*Kernel` passes (excluded from the plan for
  the same reason: they need review on the ROCm box before consolidation).
- Whether each producer's *current* untyped operands are semantically
  fragment-shaped. This inventory establishes which form is emitted, not
  whether the migration is mechanical for a given pass. Expect per-producer
  surprises in step (3).
- `numeric_policy` propagation (W1.3 / Decision #32), which interacts with the
  accumulator half of the fragment type but is costed separately.
