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
| `family` (`auto`/`mma_sync`/`wgmma`/`tcgen05`/`wmma`/`mfma`) | attribute | Instruction *selection* — a property of the op, and resolved per target by the lowering. Two fragments do not become incompatible because one op chose WGMMA. |
| `k_blocks` | attribute | Operation-level unrolling. |

What this buys, concretely:

1. Case (b) becomes legal — the `iter_args` accumulator carries its own
   contract, and `scf.for`'s existing "iter_arg type == yield type" rule
   enforces loop-carried consistency **for free**.
2. `requireFragmentProducer` and the accumulator descriptor-equality check
   collapse into ODS type equality; the 5 producer-chasing sites go away.
3. `elem`/`acc` in the type is exactly Decision #15a's storage/accumulator split
   (`storage=bf16, accum=fp32`) expressed where codegen reads it.

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
| **1** | Parameterize `Tile_FragmentType` per §3; keep the bare form parseable as `!tile.fragment` = all-unknown, so nothing breaks on day one | round-trip fixture; existing 3 fixture files still pass |
| **2** | Teach `MMAOp::verify()` to prefer the type when parameterized and fall back to producer-chasing when bare | both forms verify; the §2 K-loop case now **passes** — this is the step's real gate and it must be the fixture |
| **3** | Migrate the 5 construction sites (`TileIRLoweringPass.cpp` ×2, `GenerateWMMA{Gemm,LinearAttn,FlashAttn}Kernel.cpp`), one per PR, each with a lit fixture | per-producer fixture pairs `!tile.fragment<…>` with `tile.mma` |
| **4** | Migrate the 5 Python text emitters (`nvidia_native.py` ×4, `runtime.py`) | the Python-side tests named in inventory §2 |
| **5** | Delete `MMAOp::verify()`'s permissive branch and the bare-type fallback | inventory step 4's gate: the contract becomes binding |
| **6** | Target IR dialects — `tessera_nvidia` (3/3) then `tessera_apple` (12/12), with `tessera_x86` (0/0) as the reference shape | per inventory §4 step 5 |

**Steps 1–2 are the design risk; 3–5 are mechanical.** Do not start at 5
(inventory: "Do not start at (4)" — same rule, renumbered).

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
2. The §2 K-loop fixture — accumulator as `scf.for` `iter_args` — verifies.
3. `MMAOp::verify()` has no permissive branch.
4. Every `tile.mma` producer in `src/` and `python/` emits the typed form.
5. `tessera_nvidia` and `tessera_apple` have no unexplained `AnyType` on true
   primitives (`tessera_x86`'s 0/0 is the reference).
