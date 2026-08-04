---
last_updated: 2026-08-03
audit_role: reference
owning_plan_item: W1.1
---

# W1.1 — Tile IR typing design

W1.1 is the 5-week long pole on the critical chain **W0 → W1.1 → W2.1 → W3.1**,
and the plan asks for written scoping before code. This is that scoping.
[`W1_1_TYPING_INVENTORY.md`](W1_1_TYPING_INVENTORY.md) is the precondition —
*what exists*; this is *what to build and in what order*.

Status truth stays `MASTER_AUDIT.md` + `docs/audit/generated/` (Decision #26).
Every claim below was measured on 2026-08-03 against the working tree, and the
two decisive ones are reproducible experiments (§2), not readings of the source.

---

## 1. What changed since the inventory

Two of the inventory's blockers are now closed, which is why this document can
be a design rather than a second survey:

| Inventory item | State on 2026-08-03 |
|---|---|
| §2.1 typed form vs. warp-spec token sync are mutually exclusive | **Resolved** (#491). `MMAOp::verify()` counts `tessera::tile::dataOperands()`, shared with the ROCm consumer that already had the rule. |
| §4 step 3's missing regression net | **Landed.** `tile_mma_typed_fragment{,_invalid}.mlir`. |
| §4 steps 1–2 (`!tile.async_token`, `tile.async_copy`) | **Done.** |
| §6 `numeric_policy` propagation, costed separately | **W1.3 landed** — and it changes this design; see §5. |

So the remaining W1.1 work is inventory step 3 (migrate producers), step 4
(delete the permissive branch), step 5 (Target IR dialects). Before starting
step 3, one experiment is worth running, and it changes the plan.

---

## 2. The decisive finding — the typed form cannot cross a block-argument edge

The inventory explains the absence of C++ producers via the token conflict, and
that explanation is now spent: the conflict is fixed and still **no producer
emits the typed form**. So it is worth asking directly whether the typed form
can express a real GEMM. It cannot.

`MMAOp::verify()` recovers the contract by **chasing the producing op**:

```cpp
Operation *accProducer = accumulator.getDefiningOp();
if (!accProducer || !mmaDescAttr(accProducer) || mmaDescAttr(accProducer) != desc)
  return emitOpError("accumulator descriptor must match tile.mma");
```

A block argument has no defining op. Two experiments, both run against
`build/tools/tessera-opt`:

**(a) A fragment arriving as a function argument** — rejected:

```
error: 'tile.mma' op fragment operand must have a Tile producer
```

**(b) The canonical K-loop**, accumulator carried as an `scf.for` `iter_args` —
rejected:

```
error: 'tile.mma' op accumulator descriptor must match tile.mma
```

Case (b) is the one that matters. **A K-loop accumulator is a block argument by
construction**, and a K-loop around an MMA is what a GEMM *is*. The typed
fragment form is therefore unusable by the main matmul path — not awkward,
structurally impossible — and no amount of producer migration fixes it, because
the descriptor genuinely is not recoverable from a block argument.

This is a better explanation of the zero-producer count than the token conflict
was: the token issue blocked the straight-line path, and this blocks the loop
path, which is every real GEMM.

> Note the shape of the mistake this avoids. Inventory step 3 says "migrate the
> five construction sites onto the form the verifier already enforces." Starting
> there would have produced four working migrations and then stalled on the
> fifth — the K-loop — with the type system already half-changed. The inventory
> anticipated this class of surprise in its §3.1 lesson ("a plausible
> cross-level analogy is not a substitute for reading the producers"); the
> generalisation is to run the hardest case *first*.

---

## 3. The design decision

The contract currently lives in an **attribute on the operation**
(`#tile.mma_desc`) while the **type is bare**:

```tablegen
def Tile_FragmentType : Tile_Type<"Fragment", "fragment"> {
  let summary = "Opaque cooperative-matrix fragment";
}
```

Consequences, all measured:

* `!tile.fragment == !tile.fragment` is **always true**, so MLIR's own type
  equality checks nothing. Every compatibility rule is hand-written.
* The verifier spends 5 `requireFragmentProducer` / descriptor-equality sites
  re-deriving what a parameterized type would state — an instance of Decision
  #30 ("derive, don't ask") pointed at the type system itself.
* The contract cannot survive an SSA edge that is not a direct op result (§2).

**Recommendation: parameterize the type, and split it from the descriptor on a
principled line.**

> **The type carries what makes two fragments interchangeable.
> The attribute carries how the operation executes.**

```tablegen
// Proposed. Field names match #tile.mma_desc so the migration is a move, not a
// redesign.
!tile.fragment<m = 16, n = 16, k = 16,
               elem = bf16, acc = f32,
               role = "a", layout = "row_major">
```

| Field | Home | Why |
|---|---|---|
| `m`/`n`/`k`, `elem`, `acc`, `role`, `layout` | **type** | Determine whether a value may be used where another is expected. Needed at every use site, including block arguments. |
| `family` (`auto`/`mma_sync`/`wgmma`/`tcgen05`/`wmma`/`mfma`) | **type** | **Corrected after PR #501 review — an earlier draft of this document put it in the attribute, and that was wrong.** See §3.1. |
| `k_blocks` | attribute | Operation-level unrolling. It changes how many steps the op runs, not whether two fragments are interchangeable. |

What this buys, concretely:

1. Case (b) becomes legal — the `iter_args` accumulator carries its own
   contract, and `scf.for`'s existing "iter_arg type == yield type" rule
   enforces loop-carried consistency **for free**.
2. `requireFragmentProducer` and the accumulator descriptor-equality check
   collapse into ODS type equality; the 5 producer-chasing sites go away.
3. `elem`/`acc` in the type is exactly Decision #15a's storage/accumulator split
   (`storage=bf16, accum=fp32`) expressed where codegen reads it.

### 3.1 Why `family` is in the type — a corrected claim

The first draft of this document argued: *"two fragments do not become
incompatible because one op chose WGMMA."* **That is false**, and the reason is
worth stating because it is the one place where the "type = interchangeability"
rule is easy to apply backwards.

`family` does not merely name an instruction; it selects a **physical register
ABI**, and the backends read it as such:

* `ROCMFragmentLayout.h` resolves a descriptor to a `FragmentLayoutDescriptor`
  whose **wave size differs by family** — 32 for RDNA3/RDNA4/gfx125x WMMA, 64
  for CDNA MFMA — along with the input element count and format.
* `TileToROCM.cpp` emits, in as many words:
  `"RDNA3, RDNA4, gfx125x WMMA-v2, and CDNA MFMA descriptors are intentionally
  non-interchangeable"`.
* `NVIDIALowering.cpp` gates on `family` before matching (m, n, k, dtype) to an
  `mma.sync` variant.

Today the full descriptor-equality check in `MMAOp::verify()` prevents mixing
them. Removing producer-chasing without moving `family` into the type would
**weaken an existing contract precisely at the edge this design exists to
open**: a loop whose `iter_args` accumulator was packed for `mma_sync` could
feed a body `tile.mma` that selects `wgmma`, and the iter-arg/yield types would
compare equal.

So `family` is a type parameter. `family = "auto"` is the legal pre-resolution
value and compares equal only to itself, which reproduces today's
descriptor-equality semantics exactly rather than approximating them.

The general lesson for the rest of this migration: **"the op chooses it" does
not imply "the value does not carry it."** An operation-local *decision* can
still determine a value-level *representation*, and it is the representation
that decides interchangeability.

---

**Do not** delete `#tile.mma_desc`. It keeps a real job (`family`, `k_blocks`),
and deleting it would be the Decision #31 ordering error the plan's own risk
table names: collapsing a duplication before the surviving path can carry what
the deleted one carried.

---

## 4. Migration order

Revised from inventory §4, because §2 moves the type change **before** the
producer migration rather than after it.

| Step | Work | Gate |
|---|---|---|
| **1** | **Landed 2026-08-03.** `!tile.fragment<m, n, k, elem, acc, role, layout, family>` with a custom parser/printer; the bare `!tile.fragment` still parses and prints as all-unknown. Cheaper than the 5w estimate implied: all 7 C++ `FragmentType` uses are `isa<>` checks, so there were **no construction sites to migrate** — the only producers were fixtures and Python text emitters, and both keep working. Differing `family` / `acc` / `role` now fail on MLIR's own type equality with **zero verifier code**, and the bare form is deliberately NOT a wildcard (else the legacy spelling would be a hole through the contract). Fixtures: `tile_fragment_type{,_invalid}.mlir` — the positive one pipes through `tessera-opt` twice so it asserts a real round-trip, not just that the printer emits something FileCheck likes. | round-trip fixture; existing fixture files still pass |
| **2** | **Landed 2026-08-03.** `MMAOp::verify()` reads the contract from the operand types when any data operand is parameterized, and `FragmentPackOp` / `FragmentZeroOp` do the same for their result — without that producer half, no typed fragment could be produced at all. `#tile.mma_desc` became OPTIONAL on the typed path (it still carries `k_blocks`) and is cross-checked when present. **The canonical K-loop now verifies** (`tile_mma_typed_kloop.mlir`). | both forms verify; the §2 K-loop **verifies** — necessary, and per §4.1 not sufficient |
| **2b** | **Bigger than "accept a block argument" — see §4.2.** Both backends **materialize a zero constant** as the MMA's C operand and never read the accumulator value, so relaxing the `FragmentZeroOp` check alone would emit a silently WRONG GEMM. The real work is threading the loop-carried accumulator through the lowering, which means type-converting the `scf.for` region signature | the §2 K-loop **lowers** AND is numerically verified on gfx1151 (this box executes) — a lowering fixture alone cannot catch the wrong-answer failure mode |
| **3** | Migrate the 5 construction sites (`TileIRLoweringPass.cpp` ×2, `GenerateWMMA{Gemm,LinearAttn,FlashAttn}Kernel.cpp`), one per PR, each with a lit fixture | per-producer fixture pairs `!tile.fragment<…>` with `tile.mma`, **and a backend lowering fixture** for the producers that feed one |
| **4** | Migrate the 5 Python text emitters (`nvidia_native.py` ×4, `runtime.py`) | the Python-side tests named in inventory §2 |
| **5** | Delete `MMAOp::verify()`'s permissive branch and the bare-type fallback | inventory step 4's gate: the contract becomes binding |
| **6** | Target IR dialects — `tessera_nvidia` (3/3) then `tessera_apple` (12/12), with `tessera_x86` (0/0) as the reference shape | per inventory §4 step 5 |

**Steps 1–2b are the design risk, and 2b is now the largest of them (§4.2).
Steps 3–5 are *routine*, not mechanical** —
see §4.1; each producer still needs its own fixture and at least one needs a
backend lowering fixture. Do not start at 5 (inventory: "Do not start at (4)" —
same rule, renumbered).

### 4.3 Measured 2026-08-04 — 2b for ROCm is NOT a region-signature conversion

§4.2 concluded that threading the accumulator means converting the `scf.for`
region signature, and sized 2b as the largest remaining W1.1 step. **Measured on
gfx1151, that is wrong for ROCm.**

`GenerateWMMAGemmKernel`'s `via-tile` option emits `tile.mma %a, %b, %acc`
instead of `tessera_rocm.wmma`. Routed through
`lower-tile-to-rocm{arch=gfx1151}` it compiles, serializes an hsaco, executes,
and is **bit-identical** to the production lane:

| shape | \|base − via-tile\| | \|via-tile − numpy\| |
|---|---|---|
| 64×64×64 | **0** | 2.4e-06 |
| 256×256×256 | **0** | 1.4e-05 |
| 128×96×64 | **0** | 2.4e-06 |

So the accumulator already survives the `tile.mma` round trip on the untyped
path. No region-signature conversion is required to carry it.

**Two real gaps remain, both smaller than the conversion:**

1. The runtime's compiled-matmul pipeline is
   `generate-wmma-gemm-kernel → lower-tessera-target-to-rocdl → …` and **omits
   `lower-tile-to-rocm` entirely**, so `via-tile` is unreachable in production —
   `tile.mma` survives to LLVM translation and fails with *"missing
   LLVMTranslationDialectInterface registration … for op: tile.mma"*.
2. The TYPED fragment branch of `TileToROCM` still requires a `FragmentZeroOp`
   accumulator (§4.2), so the typed form cannot yet do what the untyped one
   demonstrably does.

**How nearly this was recorded backwards.** The first run of this experiment
reported bit-identical output too — and was meaningless: the injection never
applied (wrong match string), so the production lane ran twice. The compiled
lane also swallowed a hard `tessera-opt` failure and returned `ok=True` with
`compiler_path="rocm_compiled"`, so even a correct injection could not have been
distinguished from a fallback. Both had to be fixed before the number meant
anything, and the control — inject a bogus pass option, require `ok=False` —
is what separates the two runs. Any future 2b or step-3 measurement must carry
that control.

---

### 4.2 Accepting is not lowering correctly — 2b is bigger than it looks

Found while implementing step 2 (2026-08-03), and it changes 2b's design.

§4.1 established that both backends require the accumulator's defining op to be
a `FragmentZeroOp`. The natural reading — the one this document previously
implied — is that the check is structural pattern-matching, so teaching it to
accept a block argument is the fix. **That reading is wrong and the fix would
have been a correctness bug.**

Measured: neither typed lowering path ever reads the accumulator as a *value*.
Both synthesize a zero and pass it as the MMA's C operand.

```cpp
// TileToROCM.cpp
Value zero = arith::ConstantOp::create(builder, loc, accTy,
                                       builder.getZeroAttr(accTy));
state.addOperands({*a, *b, zero});

// NVIDIALowering.cpp — same shape, per accumulator dtype
Value zero = arith::ConstantFloatOp::create(...);
operands.append(4, zero);
```

`cZero` is used for exactly two things: the null check, and a dead-op erase at
the end. So `FragmentZeroOp` is not a pattern the lowering matches — it is a
**precondition the lowering relies on**. The generated code is correct only
because the accumulator really is zero.

Accept a block-argument accumulator without changing that, and every K-loop
iteration recomputes A×B from zero: the loop-carried value is discarded and the
GEMM returns the last K-step's partial product. No diagnostic, no crash, a
wrong number.

**So 2b is: thread the accumulator SSA value into the MMA, replacing the
synthesized zero.** That is not a local patch. At Tile level the accumulator is
a `!tile.fragment`; the lowered MMA consumes a `vector<N x f32>`. Threading it
across a loop means the `scf.for`'s iter-arg must itself be converted, i.e. a
**region-signature type conversion**, not an operand swap.

Consequences for the plan:

* 2b needs a dialect-conversion type converter over loop regions on both
  backends. Re-estimate it as the largest remaining W1.1 step, not a follow-on
  to step 2.
* Its gate must include **numerics**, not just a lowering fixture. This box
  executes gfx1151, so the ROCm half is verifiable here: a K-loop GEMM whose
  result is compared against a reference. A fixture that only checks the emitted
  ops would pass while the kernel returned the wrong answer — which is precisely
  the failure mode.
* Steps 3-5 are unblocked for the STRAIGHT-LINE producers, which do start from a
  real `fragment_zero`. Only the K-reduction producer waits on 2b.

The general form, now twice in this document: §4.1 was "verifying is not
lowering"; this is **"accepting is not lowering correctly"**. Both come from
reading a consumer's precondition as if it were a pattern match.

### 4.1 Verifying is not lowering — a corrected gate

An earlier draft made step 2's gate "the K-loop case now passes" and called
steps 3–5 mechanical. **PR #501 review showed that gate is insufficient, and it
is right.**

Making `MMAOp::verify()` accept a block-argument accumulator does not make that
GEMM *lowerable*. Both backends independently require the accumulator's direct
defining op to be a `FragmentZeroOp`:

```cpp
// NVIDIALowering.cpp — and the same shape in TileToROCM.cpp
auto cZero = mmaData[2].getDefiningOp<tessera::tile::FragmentZeroOp>();
...
if (!aPack || !bPack || !cZero || !physical) { op->emitError(...); }
```

A region iter-arg has no defining op, so `cZero` is null and both backends
error out — *after* the verifier has accepted the program. So the typed K-loop
would verify and still fail to compile, which is the worst of the two states:
the gate would be green while the motivating main-matmul path remained broken.

Hence step 2b, and hence the softened claim about 3–5. The underlying pattern is
the same one §2 found one level up — **a contract recovered by chasing the
producing op cannot survive a block-argument edge** — and it recurs in every
consumer that does the chase, not only in the verifier.

Scoped, so step 2b is not open-ended. Every site that chases a producer off a
fragment operand (measured 2026-08-03):

```
$ grep -rn "getDefiningOp<tessera::tile::Fragment" --include=*.cpp --include=*.h src/
  3  NVIDIALowering.cpp
  3  TileToROCM.cpp
```

**Two files, three sites each, and they are exactly the two the review named.**
`WarpSpecializationPass` and the epilogue materializers do *not* do this — an
earlier draft of this section guessed that they did, which would have widened
step 2b on an assumption instead of a count. Re-run the grep before declaring
step 2b done rather than trusting this number, but the shape of the work is
bounded.

### Blast radius (measured 2026-08-03)

| Surface | Count |
|---|---|
| `!tile.fragment` in fixtures | 3 files / 66 occurrences |
| `FragmentType` in C++ | 3 files / 7 occurrences |
| `mma_desc` in fixtures | 11 files |
| `mma_desc` in Python emitters | 5 files |

This is **much smaller than the 5-week estimate implies**, and the reason is not
comfort: the typed form has almost no adoption precisely *because* §2 made it
unusable. The estimate should be re-cut after step 2 lands, when the first real
producer migration reveals whether the per-producer surprises inventory §6 warns
about materialise. Treat 5w as an upper bound carried forward, not as evidence
of five weeks of known work.

---

## 5. Interaction with W1.3 (Decision #32)

W1.3 landed the boundary verifier and, in doing so, fixed
`LowerMatmulToTileMMA` to forward `numeric_policy` onto `tile.mma`. That is the
**interim** home for the accumulator contract: an attribute copied across the
boundary.

The W1.1 endpoint is better and makes the two items compose exactly:

* Today: `numeric_policy = {storage="bf16", accum="fp32"}` rides as an attribute
  on `tile.mma`, carried forward to satisfy #32.
* After step 1: the same fact is `!tile.fragment<elem = bf16, acc = f32>` — in
  the type, where codegen selects the instruction.
* At that point `TileIRLoweringPass` may legitimately declare
  `tessera.lowering.dropped = {numeric_policy = "represented_in_type"}`, which
  is a reason the W1.3 verifier already accepts and has a fixture for
  (`metadata_obligation_declared.mlir`).

So W1.1 step 1 is what converts a *carried attribute* into a *typed fact*, and
the W1.3 verifier is the thing that will notice if that conversion silently
loses the information instead. Sequence them in that order; do not add the
`represented_in_type` declaration before the type actually carries it, or the
verifier will correctly reject it as
`METADATA_OBLIGATION_STALE_DECLARATION`.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| The parameterized type changes assembly format and breaks fixtures | Step 1 keeps the bare spelling parseable; the 3 fixture files migrate in step 1's own PR |
| Per-producer surprises in step 3 (inventory §6 explicitly does not cover whether each producer's operands are semantically fragment-shaped) | One producer per PR, each with a fixture; the K-loop case in step 2 is the hardest and is deliberately first |
| `layout` in the type duplicates `#tile.layout` on `fragment_pack` | Decide in step 1: the fragment's `layout` field is the **operand layout role** (`row_major`/`col_major`) the instruction requires, NOT the full `#tile.layout` shard map. Different facts, similar names — write this into the ODS summary |
| Target dialects (step 6) invalidate assumptions, per the plan's risk table | Unchanged from the plan: land per primitive/variant with parser, verifier, lowering, and backend fixtures |

---

## 7. Done

W1.1 is complete when:

1. A `tile.mma` whose operand types disagree fails **type** verification, with no
   producer chase.
2. The §2 K-loop fixture — accumulator as `scf.for` `iter_args` — both
   **verifies and lowers**, with a per-backend lowering fixture on NVIDIA and
   ROCm (§4.1: verifying alone leaves the motivating GEMM uncompilable).
3. `MMAOp::verify()` has no permissive branch.
4. Every `tile.mma` producer in `src/` and `python/` emits the typed form.
5. `tessera_nvidia` and `tessera_apple` have no unexplained `AnyType` on true
   primitives (`tessera_x86`'s 0/0 is the reference).
