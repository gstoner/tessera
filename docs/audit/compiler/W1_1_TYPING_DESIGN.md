---
last_updated: 2026-08-18
audit_role: reference
owning_plan_item: W1.1
---

# W1.1 — Tile IR typing design

W1.1 is the 5-week long pole on the critical chain **W0 → W1.1 → W2.1 → W3.1**,
and the plan asks for written scoping before code. This is that scoping.
[`W1_1_TYPING_INVENTORY.md`](W1_1_TYPING_INVENTORY.md) is the precondition —
*what exists*; this is *what to build and in what order*.

Status truth stays `INTEGRATED_COMPILER_PLAN.md` + `docs/audit/generated/`
(Decision #26). Sections 1–4 retain the measurements that selected the design;
the 2026-08-18 resolution in §4 and the completion ledger in §7 supersede their
historical open-state wording.

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
| **3** | **Landing: 3 of 5 sites migrated.** ROCm GEMM uses the full view/pack/mma/unpack/store chain; ROCm flash/linear attention use exact-ABI typed bridges for computed register fragments. The two tensor-valued `TileIRLoweringPass` sites remain NVIDIA-owned. | per-producer typed structure plus backend lowering; exact-device where the host exists |
| **4** | **Landed by convergence, 2026-08-04.** The five old text sites emit `tile.matmul_kernel`, not `tile.mma`. The structured Python Tile builder now emits logical `tile.matmul`; target lowering owns physical fragment construction. Compatibility consumers may still read historical `tile.mma`, but no Python lowering constructs one. | Tile→Target spine tests across ROCm/NVIDIA/Apple/x86 |
| **5** | **Landed 2026-08-18.** `MMAOp::verify()` no longer producer-chases or admits bare `!tile.fragment`. The spelling remains parseable only so the verifier can emit `TILE_MMA_BARE_FRAGMENT_REMOVED`. The retained tensor-value lane is rank/arity/accumulator checked and is not a fragment compatibility path. | positive typed K-loop plus negative bare-fragment fixture |
| **6** | **Apple half landed 2026-08-04; NVIDIA half is NOT verifiable on the primary box.** `tessera_apple` had 12 `Variadic<AnyType>`: 6 are the runtime-call ops (`cpu.call`, `gpu.kernel_call`, `gpu.package_call`) and are now `Variadic<AppleTarget_Buffer>` (ranked tensor or memref) — those lower to an Accelerate/LAPACK entry point or a compiled MSL kernel, which take buffers; scalar parameters ride as attributes, so nothing is lost. The other 6 are `gpu.control_{if,loop,while}` `iter_args`/`results`, where `AnyType` is **explained and correct** (loop-carried values are polymorphic, exactly as `scf.for`'s are) and is now documented as such so a later audit does not read them as leftovers. All 303 pre-existing fixtures passed unchanged, so no producer relied on the wider set; a negative fixture proves the constraint bites on a scalar operand and a scalar result. **NVIDIA half deferred with a reason:** the `tessera_nvidia` dialect is **not registered in the default build** — `tessera-opt` rejects its ops as an unregistered dialect and its lit fixtures run under `-allow-unregistered-dialect`. Tightening its ODS here would be unverifiable, and its 3 occurrences are 2 on a shared `TesseraNVIDIA_Op` base class (deliberately generic to tolerate per-SM attribute combinations, per its own header comment) plus `cuda_math_kernel`. Needs `-DTESSERA_ENABLE_CUDA=ON`. | per inventory §4 step 5 |

**Steps 1–2b are the design risk, and 2b is now the largest of them (§4.2).
Steps 3–5 are *routine*, not mechanical** —
see §4.1; each producer still needs its own fixture and at least one needs a
backend lowering fixture. Do not start at 5 (inventory: "Do not start at (4)" —
same rule, renumbered).

### 4.5 Step 3 is not a migration — no producer is fragment-shaped

Measured 2026-08-04, on starting step 3. The inventory (§6) warned to "expect
per-producer surprises"; the surprise is not per-producer, it is all of them.

`FragmentPackOp::verify` requires **exactly one `!tile.tile` input**. What the
five construction sites actually pass to `tile.mma`:

| site | operands |
|---|---|
| `TileIRLoweringPass` ×2 | `tile.async_copy` results — **tensors** (`st.addTypes({src.getType(), …})`) |
| `GenerateWMMAGemmKernel` | lane-level **vectors** (`toFrag` → `vector::BitCastOp`) |
| `GenerateWMMALinearAttnKernel` | same shape |
| `GenerateWMMAFlashAttnKernel` | same shape |

**Zero producers pass a `!tile.tile` or a `tile.view` result.** No operand can be
wrapped in `fragment_pack`: the typed contract expects
`tile.view → fragment_pack → tile.mma`, and every producer supplies either a
tensor or a vector whose lane math it has already done.

That is a **division-of-labour mismatch**, not a syntax gap. The typed form
assumes the COMPILER performs the lane mapping (`materializeFragmentPack`); the
hand-written generators perform it themselves and hand over finished vectors.
Both are coherent; they are different models.

**Consequences the plan's wording does not survive:**

* Step 3 ("migrate the 5 construction sites, one per PR") is not a migration. It
  is a rewrite of working, numerically-verified generators — including the
  production ROCm GEMM lane — to emit logical tile views and surrender their
  lane math.
* Step 5 ("delete `MMAOp::verify()`'s permissive branch") is **unreachable as
  written**: deleting it breaks every existing producer.

**2026-08-18 resolution.** The historical option analysis below is superseded.
The ROCm producers were restructured onto the typed view/pack/mma/unpack/store
boundary, and the remaining tensor-valued producers are NVIDIA-owned migration
work. The shared verifier no longer preserves a bare-fragment compatibility
envelope. This keeps architecture closure open without leaving a hole in the
shared type contract.

**Historical options considered:**

  a. **Restructure the producers** to `tile.view → fragment_pack`. Rewrites
     proven kernels for no measured performance benefit.
  b. **Widen `fragment_pack`** to accept tensors/vectors. Cheap, and it discards
     most of what the typed contract buys — §3's whole point was that the type
     states what makes two fragments interchangeable.
  c. **Scope the typed form to synthesized kernels** (the Decision #28 lane) and
     treat the permissive branch as the documented boundary between two
     legitimate models rather than debt awaiting deletion.

**Historical recommendation: (c).** The typed contract earns its keep where the compiler
owns the lane mapping — exactly the synthesizer's job. The hand-written
generators are a separate working lane whose operands are physical by design.
Under (c) step 5 becomes "the permissive branch is a declared compatibility
envelope, and any path where the compiler owns lane mapping must use the typed
form" — a Decision #32-style declared boundary rather than an open TODO.

This also closes W1.1 honestly: steps 1, 2 and 2b's guard are real contract
improvements that landed and are gated; steps 3–5 as written were premised on a
producer shape that does not exist.

---

### 4.4 Gap 2 is not independently actionable — it is coupled to step 3

§4.3 listed two remaining ROCm gaps. The first (the pipeline could not lower
`tile.mma`) is closed. The second — `TileToROCM`'s TYPED branch requiring a
`FragmentZeroOp` accumulator — **should not be fixed yet**, and the reason is
worth stating so it is not picked up as ready work.

**The typed branch has no producer and no test.** Measured 2026-08-04:

| question | answer |
|---|---|
| C++ passes emitting `fragment_pack` / `fragment_zero` | **none** (only the two consumers and the verifier mention them) |
| Python emitters producing them for ROCm | **none** — `runtime.py` has 0 occurrences |
| lit fixtures pairing a typed fragment with a ROCm lowering | **none** |
| tests asserting its `"typed ROCm lowering requires ..."` diagnostic | **none** |
| other users of `materializeFragmentPack` | none — it is local to `TileToROCM.cpp` |

**And the fix is not the obvious one.** Relaxing the `FragmentZeroOp` check
alone would reproduce the NVIDIA defect exactly: the typed branch synthesises
its own accumulator —

```cpp
Value zero = arith::ConstantOp::create(builder, loc, accTy,
                                       builder.getZeroAttr(accTy));
state.addOperands({*a, *b, zero});      // typed branch  — accumulator DISCARDED
state.addOperands({mmaData[0], mmaData[1], acc});  // untyped branch — threaded
```

— so accepting a non-zero accumulator without threading it would silently drop
it. Threading requires materialising the incoming accumulator fragment into the
physical `accTy` vector, the accumulator-side counterpart of
`materializeFragmentPack`, which A and B already have and the accumulator does
not.

That is real work, and it is **unverifiable today**: with no producer, there is
no program to run and therefore no numeric gate. Building it now would repeat
the mistake §4.3 corrected — infrastructure for a path nothing executes, sized
by reasoning rather than measurement.

**So gap 2 is a step-3 obligation, not a predecessor of it.** When a producer
migrates to emit typed fragments for ROCm, that migration must carry the
accumulator materialisation with it, and its numeric gate covers both. Until
then the branch is unexercised code whose contract cannot be checked.

The untyped path — which production uses via `via-tile` — already threads the
accumulator correctly and is proven bit-identical on gfx1151 (§4.3).

---

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

This table is the 2026-08-03 snapshot, not current producer truth. The current
Python occurrences belong to `tile.matmul_kernel` launch envelopes; there are
zero Python `tile.mma` construction sites (see §4.8).

This is **much smaller than the 5-week estimate implies**, and the reason is not
comfort: the typed form has almost no adoption precisely *because* §2 made it
unusable. The estimate should be re-cut after step 2 lands, when the first real
producer migration reveals whether the per-producer surprises inventory §6 warns
about materialise. Treat 5w as an upper bound carried forward, not as evidence
of five weeks of known work.

---

### 4.6 The shape steps 2b/3/4/5 actually need — a dialect conversion

Measured 2026-08-04, and it reframes the chain. The blocker is not only that
producers are not fragment-shaped (§4.5); it is that **the typed lowering cannot
compose**, so there is nothing for a migrated producer to emit *into*.

**What `TileToROCM`'s typed path actually is.** Not an op-by-op lowering — a
**single-shot whole-chain pattern match**:

1. from `tile.mma`, chase operands to find `aPack`, `bPack`, `cZero`;
2. require a `fragment_unpack → tile.store` consumer (`hasOutputStore`);
3. materialise A and B from their `tile.view`s, **synthesise a zero accumulator**;
4. emit one `tessera_rocm.wmma` + the store;
5. **erase** store, unpack, mma, aPack, bPack.

The whole `view → pack → zero → mma → unpack → store` chain collapses into one
physical op. Three things are therefore inexpressible by construction:

* an accumulator that is not a `fragment_zero` (**step 2b**),
* an mma whose result feeds another mma rather than an unpack/store,
* a chain crossing a loop-body boundary (**the K-loop**).

That is why "thread the accumulator" looked small and kept growing: there is no
seam to thread it through.

**The architecture.** Convert the typed path from pattern-match-and-erase into a
**dialect conversion with a `TypeConverter`**:

| element | conversion |
|---|---|
| `!tile.fragment<…>` | → `vector<N × elem>` (input) / `vector<N × acc>` (accumulator) |
| `tile.fragment_pack` | → the existing `materializeFragmentPack` (already bounds-masked, §3a) |
| `tile.fragment_zero` | → `arith.constant` zero vector |
| `tile.mma` | → `tessera_rocm.wmma` on **converted operands** — no producer chasing |
| `fragment_unpack` + `tile.store` | → the existing store materialisation |
| `scf.for` / `scf.if` iter_args | → **free**, via `populateSCFStructuralTypeConversionsAndLegality` |

Each op lowers independently against converted operand types, so chains and
loops compose because SSA composes.

**Why this is smaller than §4.2 sized it.** That section sized 2b as a
hand-rolled `scf.for` region-signature conversion and called it the largest
remaining step. **MLIR already ships that**:
`populateSCFStructuralTypeConversionsAndLegality` is present in this LLVM 23
build (`mlir/Dialect/SCF/Transforms/Patterns.h:52`). The loop-carried case is a
library call, not a rewrite.

The cost concentrates in one place instead: **no pass in this tree uses a
`TypeConverter` today** (measured: zero hits across `src/**/*.cpp`). This would
be the first — which is precisely why every lowering here is a whole-chain
rewrite; the idiom was never available to copy.

**What falls out, per step.**

* **2b** — no longer a discrete task. A non-zero accumulator is just a converted
  operand, and "synthesise a zero" becomes the lowering of `fragment_zero`,
  which is where it belonged.
* **3** — the producer no longer emits a recognisable *pattern*, only well-typed
  ops. It must still supply `tile.view` + `fragment_pack` (§4.5), but need not
  arrange them so one matcher finds them all — a material reduction in the
  `GenerateWMMAGemmKernel` rewrite.
* **4** — the Python text emitters gain the same freedom.
* **5** — unchanged, and still last.

**Order, and the gate.**

1. Introduce the `TypeConverter` + conversion patterns behind the existing entry
   point, leaving the single-shot path handling what it already handles.
2. Prove composition with a fixture the old path **cannot** express: a K-loop
   whose accumulator is an `scf.for` iter-arg — the case
   `tile_mma_typed_kloop.mlir` verifies but nothing can lower.
3. Then migrate producers (step 3) and retire the single-shot path.

Gate stays numeric (§5 of `GEMM_PERF_LADDER.md`): a structural fixture passes
while the kernel returns a partial product. The differential oracles
(`tessera_rocm_wmma_gemm_f16_bench_{lds,pipe}`) and the γ_K bound both apply.

### 4.6.1 Built 2026-08-04 — what shipped, and the two defects it exposed

`convertTypedFragments()` in `TileToROCM.cpp`: a `TileFragmentTypeConverter`
mapping `!tile.fragment<...>` → `vector<N × T>`, four `OpConversionPattern`s
(`fragment_zero`, `fragment_pack`, `mma`, `fragment_unpack`+`store`), and
`applyPartialConversion`. It runs **before** the legacy walk; the bare
`!tile.fragment` spelling converts to itself, stays legal, and falls through
unchanged, so the two forms coexist until step 5.

`scf.for` cost one line, as predicted:
`populateSCFStructuralTypeConversionsAndLegality`. No Tile-specific loop
reasoning exists anywhere in the result.

Proven by `rocm_typed_fragment_composition.mlir` — one function per shape the
single-shot path cannot express: a K-loop accumulator arriving as an `scf.for`
iter-arg, an `mma` feeding an `mma`, and an accumulator that is not a
`fragment_zero`. **The fixture was verified to fail**: re-injecting the
synthesized-zero defect makes it red, so it is checking the accumulator operand
and not merely that a `wmma` was emitted.

Two defects surfaced only because the negative case was run. Both had green
positive tests.

1. **A blanket identity type conversion silently disabled the whole thing.**
   `TypeConverter::convertType` reads a `std::nullopt` callback result as *"not
   applicable, try the next callback"*, so an **unresolvable** fragment fell
   through to the identity conversion, was declared legal, and passed through
   untouched — emitting a `tessera_rocm.wmma` whose operands were still
   `!tile.fragment`, and **exiting 0**. The identity conversion must exclude
   fragments so that no callback applies and `convertType` fails.

2. **Accumulator convertibility does not imply input convertibility.** An `acc`
   fragment names no input dtype, so §4.6's representative-dtype device
   resolves its physical layout — correct for the *width*, which is
   `256 / waveSize` regardless of input dtype, but it means an acc-based check
   cannot police the inputs. On gfx1151 an e4m3 A/B pair is unsupported while
   its f32 accumulator resolves happily. `ConvertMMA` now checks all three
   operands converted.

Both are recorded in `rocm_typed_fragment_composition_invalid.mlir`, whose
correct output is a diagnostic (Decision #10a).

Review of the landed change (#517) found three more, all of the same shape —
**a fact stated in two places that drifted**:

3. **`role` was read from the op attribute, not the type.**
   `FragmentPackOp::verify` returns success *without ever requiring a `role`
   attribute* once the result is typed, so `tile.fragment_pack %v : (!tile.tile)
   -> !fa` is valid IR — and it is exactly what a migrated producer emits, since
   removing the redundant attribute is the point of typing. The lowering
   rejected it with `ROCM_FRAGMENT_MISSING_CONTRACT`. `role` is now a parameter,
   sourced from the type on the typed path and from the attribute on the legacy
   one.

4. **A fixed representative accumulator dtype was wrong on every architecture
   with a dtype-dependent `k`.** `resolveFragmentLayout` derives the legal `k`
   *from* the input dtype — RDNA4 int4 takes k=32 where int8 takes k=16;
   gfx125x fp8 takes k=64 where f16 takes k=32; CDNA spans k=8 to k=64 — so
   pinning "int8"/"f16" made a *supported* MMA's accumulator unresolvable. Now
   probed over a candidate list. Sound because the accumulator width is
   `256 / waveSize` in every branch and `waveSize` is fixed per architecture, so
   any candidate that resolves gives the same answer.

5. **The converted type and the materialized value disagreed for packed
   inputs.** `materializeFragmentPack` packs sub-16-bit inputs into i32
   registers (bitcast for int8/SOA-int, nibble compaction for int4), so an RDNA4
   int4 fragment leaves as `vector<2xi32>` — while the converter promised
   `vector<16xi8>`. The conversion then stranded an unresolved materialization.
   **The gfx1151 f16 path could never expose this**, because there the packing
   is the identity. Both now derive from one `packedFragmentType()`, with a
   `ROCM_FRAGMENT_TYPE_DISAGREES` guard so a future divergence is named rather
   than surfacing as an unresolved cast.

Defect 5 is the general lesson: a type converter restates, in a second place,
what the materializer decides. That is a Decision #31 duplication in miniature,
and it failed exactly the way #31 predicts — silently, on the path the default
test configuration does not cover.

**A local gate gap, found by CI, not by me.** `MLIRSCFTransforms` was missing
from `TesseraROCMConversion`'s link libraries. `ninja -C build tessera-opt`
links `MLIROptLib`'s broader set and hid it; the standalone `tessera-rocm-opt`
does not. Build **all** targets (`ninja -C build`) before pushing a change that
adds an upstream MLIR dependency — a single-target build is not a link check.

**What is NOT yet true.** No producer emits typed fragments (step 3 is
unstarted), so this path is proven by fixture and is not yet on any executing
lane. The numeric gate above therefore has nothing to run against yet — it
applies at step 3, not here. 2b is closed as a *capability*, not as shipped
codegen.

---

### 4.7 Step 3 scoped by measurement — bit-identity is reachable; the cost is NOT quantified

Measured 2026-08-05 against `GenerateWMMAGemmKernel`'s own emitted IR, before
writing any of the migration.

**Corrected after #520 review — the first version of this section carried two
wrong numbers, and they are worth naming because both looked measured.**

1. It described the measurement as taken at *"the generator's default
   `mt = nt = 16`"*. The generator defaults to `mt = nt = 1`
   (`GenerateWMMAGemmKernel.cpp:99`, `:976`) and overrides only from explicit
   attributes; 16×16 came from an int4 storage-pack fixture I had copied the
   directive from. Production selects small macro tiles (the ladder's naive row
   is MT=2, NT=4), not 16×16.
2. It reported *"32 `arith.muli` in the K-loop body"* for the producer. That was
   an `awk` range that stopped at the first inner `scf.yield` — a truncated
   prefix, not the loop body. Whole-module `arith.muli` does scale with the
   tiling: **84** at 1×1, **294** at 2×4, **4674** at 16×16.

The "~288 for per-fragment packs" figure alongside them was an *estimate*
presented in a table of apparently-measured values. **No multiply-count ratio
from that version should be relied on.** The corrected position is below.

**What is verified, and it is the load-bearing result: bit-identity is
reachable.** For B element *j* the producer computes

$$(k_0 + j)\cdot N + \mathrm{col}$$

and `materializeFragmentPack`'s strided gather computes

$$\underbrace{(k_0 \cdot N + \mathrm{col})}_{\text{linear}} + j \cdot N$$

which is the same integer. Same addresses in the same order ⇒ same loads ⇒ same
`wmma` inputs ⇒ bit-identical output. This does not depend on any count. The
producer's B assembly is already `memref.load` + `vector.insert`, exactly the
shape the gather emits, so it matches in structure as well as in value. **That
was the main risk in step 3 and it is retired.**

**What is also verified is a structural difference in address FORM.** Read
directly off the emitted IR: the producer computes the B row offset
`(k0 + j) * N` **once per *j* and reuses that one SSA value across the `nt` B
fragments** — one `arith.muli` feeding several `arith.addi`, one per column
base. A's base `arK[mi] = arM[mi] * K` is hoisted out of the K loop entirely,
leaving a single `addi` inside.

`fragment_pack` cannot do either. It derives its address from
`(base, rowOrigin, colOrigin)` operands **in isolation**: it cannot see its
sibling fragments, and it has no operand for an already-computed base. The
per-fragment granularity that makes the contract composable (§4.6) is the same
granularity that discards a cross-fragment CSE the hand-written producer gets by
construction.

**How much that costs is unknown and is deliberately not estimated here.** Two
reasons it may cost nothing: most of the affected multiplies are
loop-invariant (`j * ld` has both operands invariant; A's `row * ld` is
invariant in the K loop), so LICM/CSE may recover them; and the sharing that
does not survive scales with `nt`, which is small in production (4), not 16.
Quantifying it requires emitting the migrated form and counting — i.e. it is a
step-3 output, not a step-3 input.

So step 3 carries a design question, recorded rather than silently decided:

> Should `tile.view` be able to carry a **precomputed linear base**, so the
> hoisting the producer does by hand is expressible at Tile level — or should
> the migration rely on LICM/CSE to recover it?

Answer it with a measurement taken *after* the simplest configuration migrates,
not with a preference now.

**Sibling-backend assessment** (AGENTS.md — shared Tile IR contract). Measured
by which backends consume the ops at all:

| backend | consumes `tile.view` / `fragment_pack` | outcome |
|---|---|---|
| ROCm | yes — this work | owning backend |
| NVIDIA | yes, 8 files each | **follow-up required** — the same isolation applies to its fragment path, and a `tile.view` base operand would change its lowering too. Tracked under the sync key below. |
| x86 | no (0 files) | **not applicable** — AMX operands come from `amx_tile_load` over the `!tessera_x86.tile` type (Decision #19); there is no `tile.view`-backed fragment path to hoist for. |
| Apple | no (0 files) | **not applicable** — the MLIR lane lowers to `func.call` on runtime symbols and the MSL synthesizer is a separate Python path; neither consumes Tile fragment ops. |

Cross-backend synchronization key: `TILE-VIEW-LINEAR-BASE-2026-08-05`.

**The first slice cannot be a 1x1 toy** (measured 2026-08-05).
`select_rocm_gemm_schedule` returns macro tile **(2, 4)** for every shape tried
— 64^3, 256^3, 2048^3 — so the hardware oracle below exercises `mt=2, nt=4`.
A migration slice restricted to `mt = nt = 1` would compile and be gated by
nothing. Either the first slice handles general `mt x nt`, or the test forces a
1x1 schedule (`_build_compiled_gemm_hsaco` takes `mt`/`nt` explicitly, subject
to its `(mt, nt) != schedule.macro_tile` consistency check). Decide that before
writing code, not after.

**The numeric gate already exists and is the right one.**
`test_via_tile_matches_the_production_lane_on_hardware` routes the GEMM through
`tile.mma` and demands bit-identical output against the production lane, with a
bogus-option control that must fail and a fallback-fatal guard so the comparison
cannot be satisfied by running the hand-written oracle twice. Step 3 extends
what `via-tile` emits — from "same operands, different op name" to the full
`tile.view` → `fragment_pack` → `tile.mma` → `fragment_unpack` → `tile.store`
chain — and inherits that gate unchanged. Throughput is then read against
`GEMM_PERF_LADDER.md`'s 8.02 TFLOP/s naive-register row, at the same
`timer_source`.

### 4.8 First step-3 producer landed — dynamic addresses, full 2x4 chain

`GenerateWMMAGemmKernel{via-tile=true}` now emits the complete typed chain for
the production `mt=2, nt=4` schedule. The producer states logical origins,
bounds, and runtime leading dimensions; `fragment_pack` owns wave32 lane
mapping, ragged masking, strided-B gathering, and register packing. The
unpacked accumulators store through the same typed address contract.

This required one shared contract closure that §4.7 did not inventory:
problem-size-generic N/K cannot inhabit the static `leading_dim` field.
`#tile.memory_layout<leading_dim = 0>` therefore denotes an SSA leading
dimension in the final `tile.view` / `tile.store` operand. Hard-coding 64 would
have passed the only hardware gate while making the experimental lane wrong for
other shapes.

Gates: exact structural counts (24 views/packs, 8 zeros, 32 MMAs, 16
unpack/stores), no Tile/cast leaks after ROCDL lowering, and zero-difference
output against the direct lane on gfx1151. Exact-device coverage now includes
aligned `64x64x64` and ragged `65x67x31`, so M/N edges and the K tail execute.

The 2048^3 throughput gate is intentionally two-sided. The typed lane reaches
**12.53 TFLOP/s**, clearing the committed **8.02 TFLOP/s** row by **1.562x**,
but reaches only **0.685x** of the same-run direct compiler lane
(18.28 TFLOP/s). Therefore the producer is retained behind `via-tile` and is
not promoted. This measurement answers §4.7's open question: relying on the
backend to recover all producer-level address sharing is not sufficient on the
current pipeline; `TILE-VIEW-LINEAR-BASE-2026-08-05` remains required.

Two ROCm attention construction sites are now typed as well. Flash attention
and linear attention cannot truthfully use pointer-backed `tile.view` for every
operand: softmax probabilities and feature-mapped Q/K are computed values whose
register vectors already carry the architecture lane map. They attach the
parameterized fragment types at that register-owned boundary, and
`ConvertFragmentBridge` erases the vector-identical bridges only when the
stated type resolves to the exact gfx architecture ABI. Both routed lanes lower
to the same ROCDL operation multiset as their direct generators, including the
ReLU feature-map case. This is a typed boundary, not a permissive vector
`tile.mma`.

The remaining **two** C++ construction sites are the tensor-valued producers in
`TileIRLoweringPass`; they feed the NVIDIA warp-specialization/WGMMA lane and
require that backend's typed tensor-to-fragment materializer. Step 3 is landing,
not complete. Step 4's old “five Python text emitters” count is no longer a live
migration surface: current `nvidia_native.py` and `runtime.py` emit the typed
launch-level `tile.matmul_kernel` envelope and no Python file constructs
`tile.mma`. That outcome is recorded by re-inventory rather than manufacturing
fragment ops in Python merely to satisfy a stale count.

`via-tile` is also the right seam for *staging*: the default path stays
untouched while the typed chain is built behind the option, so a partial
migration is never a partial production lane. A partial migration **within one
`tile.mma`** is not expressible — `MMAOp::verify` enforces all-or-nothing typing
(`TILE_MMA_MIXED_FRAGMENT_FORMS`) — so the unit of migration is a whole MMA, not
an operand.

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

W1.1 shared-contract completion ledger:

1. A `tile.mma` whose operand types disagree fails **type** verification, with no
   producer chase.
2. The §2 K-loop fixture — accumulator as `scf.for` `iter_args` — verifies and
   lowers on the typed ROCm path. NVIDIA lowering evidence remains open.
3. **Closed.** `MMAOp::verify()` has no permissive bare-fragment branch.
4. ROCm and Python producers use the typed/logical boundary. NVIDIA's final two
   tensor producers remain the architecture-owned open item.
5. Apple has no unexplained `AnyType` on true primitives. NVIDIA's generic
   Target base and CUDA-enabled proof remain independently gated; x86 is the
   reference shape.

The shared contract is therefore closed; W1.1 remains `landing` only for the
NVIDIA producer/Target proof and eventual removal of the separately checked
tensor-value migration lane.
