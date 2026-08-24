---
last_updated: 2026-08-24
audit_role: plan
plan_state: open
---

# CuTe IR assessment — layout algebra as the missing shared substrate

> **Routing:** start at [`README.md`](README.md). This document owns the CuTe IR
> source review, its mathematical verification, and the layout-algebra
> extraction decisions that follow. Global ordering lives only in
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md); `MASTER_AUDIT.md`
> + generated dashboards stay status truth (Decision #26). This is a provenance
> and build-sequence document, not a status claim.

**Date:** 2026-08-16 · **Subject:** [NVIDIA/cutlass#3426](https://github.com/NVIDIA/cutlass/pull/3426),
"Cutlass compiler: Introducing CuTe IR MLIR dialect", merged 2026-08-03 at
`3664fe2`. 475 files, ~125k lines added.

**Sources read:** the ODS (`CuteTypes.td`, `CuteOps.td`, `CuteAttrs.td`,
`CuteTypeInterfaces.td`, `CuteDialect.td`), the pass definitions
(`Cute/Transforms/Passes.td`, `Conversion/CuteToBase/Passes.td`,
`base/Conversion/BaseToTargets/Passes.td`), `CuteTypeConverter.h`, the project
README, and the layout-algebra tutorial chapters under
`media/docs/cutlass_compiler/cute_concepts/`. **Not read:** the vendored
`cutegen` C++ algebra and the conversion `.cpp` bodies — so every statement
below about *behaviour* is read off a declared contract, not observed execution.

**Verification:** every worked example in the tutorial is machine-checked by
[`tests/unit/test_layout_algebra_contracts.py`](../../../tests/unit/test_layout_algebra_contracts.py)
— independent integer evaluation, no cutegen dependency, no numpy. 32 tests.
Fixtures are the **documented layout strings verbatim** (`"((3,4),(2,5)):((4,1),(12,24))"`),
parsed into nested `(shape, stride)` trees, so mode nesting is preserved and
transcription is removed as a source of error. Each operation is checked on both
axes — its **function** (image over `[0, size)`) and its exact **result
structure** — because the two are independent and neither implies the other.

**Bottom line:** the algebra is mathematically sound and its documentation has
three defects. The surface is 63 ops but the mathematics is **four primitives
plus regrouping**. Tessera does not need CuTe IR, but it does need layout
algebra — because six independent in-tree consumers each ask for a different,
weaker mechanism for the same question, and none of them names it (the sixth,
added 2026-08-18, is the only one identified *before* it was built).

---

## 0. What the PR actually is

Not what the title suggests. There are **no math ops in it**: zero files match
tensor / MMA / copy / atom. 374 of the 475 files are tests.

| Component | Content |
|---|---|
| `cute` dialect | 8 attribute-backed types, 63 ops in 8 spec groups: constructors, accessors, layout algebra, size/index, arithmetic, 11 tiling/partitioning products |
| `Base` pseudo-dialect | Not a dialect — a *named lowering boundary* made of `arith`/`scf`/`cf`/`ub`/`func`/`math`/`gpu`/`LLVM`/`NVVM` |
| Passes | `cute-fold-static`, `cute-expand-ops`, `cute-to-base`, plus `base-prepare` / `one-shot-convert-to-llvm` / `attach-nvvm-target` / `emit-gpu-binary` |
| Drivers | `cute-opt`, `base-opt`, `cutlass-compiler` |

It is **indexing math, not mathematical ops**. That distinction governs
everything that follows.

---

## 1. Mathematical verification

Every worked example re-derived independently and machine-checked. Harness:
[`tests/unit/test_layout_algebra_contracts.py`](../../../tests/unit/test_layout_algebra_contracts.py).

### 1.1 What holds

| Claim | Result |
|---|---|
| `coalesce` preserves the function; fusion rule `d₁ == s₀·d₀` | **Exact**, both plain and by-mode |
| `composition` `R(c) = A(B(c))`, incl. the mode-splitting `((2,2),3):((24,2),8)` | **Exact** over the full domain |
| Composition with a **tile** is mode-wise, not functional | **Confirmed** — the flat-layout reading gives a different answer |
| `complement(A, M)` with cotarget: `A ⊕ A*` tiles `[0,24)` | **Bijective**, 24 distinct, no gaps |
| `right_inverse` / `left_inverse` identities | **Exact** on both documented layouts |
| `recast_layout`, incl. the `gcd`-split general case (4-bit from 6-bit) | **Bit-extent conserved** at every step |
| Six product variants = one leaf multiset, regrouped | **Confirmed** — identical `(extent, stride)` sets |
| Four divide variants = one leaf multiset, and preserve the input image | **Confirmed** |
| `tile_to_shape` block (2,2) over (8,8) | Covers `[0,64)` **bijectively** |
| `logical_divide` by a **non-identity layout tiler** → `((3,(2,2)),4):((8,(24,1)),2)` | **Exact** |

That last row is the strongest evidence the algebra is real rather than
decorative. With `complement((3,4):(1,3), 48) = 4:12`, the inner map is the
identity on `[0,48)` — yet the result is *not* the source regrouped, because
`A(j) = 8l₀ + 24(l₁ mod 2) + (l₁ div 2) + 2l₂` is genuinely non-affine in the
`l₁` mode. The nested `(2,2)` is that split, represented exactly rather than
approximated or rejected. A layout system that only carries flat
extent/stride arrays cannot express this result at all.

### 1.2 Three documentation defects

All three are in prose. None is an algebra defect. Each is carried as a
negative fixture in the harness (Decision #10a) so a future port cannot inherit
it; if a later CuTe release fixes the prose, those three tests are the ones to
revisit — they are not regressions in our code.

1. **Composition-with-shape prose contradicts its own example.**
   `core_ops.rst` says shape-composition is "equivalent to composing with the
   column-major layout built from that shape (`A ∘ make_layout(shape)`)". For
   its own example `A = (4,8):(1,4)`, that reading yields `[0,1,2,3,4,5,6,7]`;
   the stated result `(2,4):(1,4)` yields `[0,1,4,5,8,9,12,13]`. The correct
   reading — and the one `CuteOps.td` gives — is **mode-wise truncation**: mode
   *i* of `A` restricted to `shape[i]`. **The ODS is right; the tutorial is
   wrong.**

2. **The product table's `logical_product` row is wrong.** It reads
   `((M,TilerM),(N,TilerN),…)`, which is verbatim its own `blocked_product`
   row. The worked example gives `((3,4),(2,5))` — input-grouped, tiler-grouped
   — matching the standard definition `(A, complement(A,·) ∘ B)`. The example
   is right; the table row is not.

3. **`complement`'s stated guarantee fails for the no-cotarget form.** The lead
   paragraph promises coverage of "exactly `[0,M)` with no overlap". In the
   single-operand example `M` defaults to `cosize(A) = 6`, and the documented
   answer `(2,1):(2,8)` yields a bijection onto **`[0,8)`** — a strict superset.
   It does reach the codomain holes `{2,3}`, so the form is usable; the
   guarantee as written is false. Traces to `shape_div` saturating `6/8` to 1
   rather than 0. **A port that reads the prose as the contract would build a
   wrong verifier.**

### 1.3 One semantic caveat worth carrying forward

The composed-layout example evaluates `A = (6,2):(1,3)` at `n₀ = 6`, past its
own extent, and the composed image reaches 12 against `cosize(A) = 9`. This is
legal — a CuTe layout is an affine function defined beyond its shape — but any
Tessera bounds verifier must model it **deliberately** rather than assume
containment.

---

## 2. The scoping result: 63 ops, four primitives

Reading the dialect with the arithmetic lens changes the cost estimate sharply:

- **`composition` is the only real primitive.** Slice, dice, the products, the
  divides, and the tiling utilities are all defined through it.
- **`complement`** is the second — it manufactures the "where does each copy
  live" layout. Products are `(A, A* ∘ B)`; divides are
  `composition(A, (B, complement(B, size(A))))`.
- **`coalesce`** is the canonical form, i.e. what makes layout equality
  decidable.
- **`right_inverse`** converts between coordinate systems.

Everything else — 11 tiling products, 4 divides, 14 accessors, the casts — is
**mode regrouping and plumbing**. §1.1 confirms this directly: all six product
variants and all four divide variants carry the same leaf multiset and address
the same offsets in different bracketings.

> **The regrouping claim is structurally proven, not assumed** (strengthened
> 2026-08-16, PR #573 review). A leaf-multiset comparison is grouping-*invariant*
> by construction, so on its own it cannot distinguish the variants — it would be
> vacuous if the fixtures had collapsed them. The harness therefore pins each
> variant's exact nested structure first (`logical_product` → `((3,4),(2,5))`,
> `tiled_product` → `((3,4),2,5)`, `flat_product` → `(3,4,2,5)`,
> `blocked_product` → `((3,2),(4,5))`, `raked_product` → `((2,3),(5,4))`), and
> pins the equivalence classes: the six are **five** distinct structures, since
> `logical_product` and `zipped_product` coincide for a plain-layout tiler.
> Only against that does the shared multiset mean anything.
>
> An earlier revision encoded every layout flat, which collapsed four of the six
> products onto one identical tuple — so that version of the test asserted what
> had been typed rather than what the algebra does. Verified by mutation: the
> flat harness passes with `zipped_divide` deliberately returning
> `logical_divide`'s grouping; the current one fails three tests on that
> mutation, three on a wrong `blocked_product` grouping, three on a
> flatten-everything implementation, and one on flattening the depth-2 divide
> split.

So the honest scope for Tessera is **four primitives plus regrouping**, not 63
ops.

### 2.1 Mechanisms worth taking regardless

| Mechanism | What it is | Tessera relevance |
|---|---|---|
| **Value-in-type, partially static** | `!cute.layout<"(?,3):(1,2)">` — the type carries the value with dynamic leaves marked `?`. 54 of 63 ops declare `InferTypeOpInterface`, so **the type checker is the algebra evaluator** and partial evaluation is a side effect of type inference | Generalizes W1.1's fragment argument to every `!tile.*` type; Decision #30 pointed at the type system |
| **`MaybeStaticTypeInterface` + `cute.static` + `cute-fold-static`** | `isStatic()` / `getValueAttr()`; a `ConstantLike` op whose entire content is its result type; one pass folds any pure op with a static result | Tessera declares 10 `!tile.*` types with **no notion of staticness at all**. Cheapest high-value import, and orthogonal to the algebra |
| **Lowering erases the static part** | `cute-to-base` maps each type to an LLVM struct of **dynamic leaves only**; `!cute.swizzle` → an *empty* struct | The sharpest available answer to Decision #32: information is not "preserved or declared dropped", the boundary type **is** the unresolved residue |
| **One dynamic-residue shape** | `cute-expand-ops` rewrites each algebra op to either `cute.static` or `get_scalars(only_dynamic)` → `arith` → `make_*`, leaving `cute-to-base` only 10 ops | The algebra never acquires a second implementation for the dynamic case |
| **`base-opt`, a driver that makes leakage a parse error** | Registers the target boundary and **not** `cute`, so `!cute.layout<...>` fails with `unregistered dialect` | `test_target_ir_contract.py` parses and verifies every emitter and golden — real coverage — but registers everything, so it cannot falsify a *leakage* claim. Same species as Decision #19's standing lesson: a host that has the ISA cannot falsify a host-portability claim |

### 2.2 What not to take

- **The string-typed assembly** (`!cute.layout<"(2,3):(1,2)">`) parses a DSL
  inside a type parameter. It is free for NVIDIA because `cutegen` already owns
  that printer/parser; for Tessera it is pure cost. Use structured ODS
  parameters.
- **Don't push this into Graph IR.** `tessera.matmul` should not grow
  `logical_divide`.
- **The dependency shape is the real warning.** CuTe's dialect delegates all
  inference and verification to one vendored C++ algebra — their Decision #31
  answer. Tessera's corresponding question is *which* compiler owns it, given
  that the Python synthesizer and the C++ MLIR pipeline both need it. Building
  it in one and re-deriving it in the other would widen the seam this work
  exists to close. That is L0 below, and it is a decision, not a detail.

---

## 3. How this fits the existing assessments

This is the finding that matters. **Four assessments and two plans
independently ask for a piece of layout reasoning, and each proposes a
different, weaker mechanism.** None names layout algebra, because none was
looking at the others through that lens.

The sixth row is different in kind and is the argument's strongest form: it was
found **during design, before any code existed** — the autodiff plan's
coefficient axis was checked against this document and turned out to be six
layout-algebra operations wearing a new name. The first five are archaeology;
this one is the mechanism working prospectively.

| Source | What it asks for | What the question actually is |
|---|---|---|
| [`FORGE_ASSESSMENT.md`](FORGE_ASSESSMENT.md) W1 | read-locality lattice `coordinate ⊏ row ⊏ block ⊏ tensor ⊏ layer ⊏ global`; fusion legal iff `consumer.read_locality ⊑ producer.tile_partition` | Does the consumer's read layout **factor through** the producer's tile partition — `∃X. read = partition ∘ X`? A six-point chain approximating a question `composition` + divisibility **decides exactly** |
| [`FORGE_ASSESSMENT.md`](FORGE_ASSESSMENT.md) W2 | `tessera.residency ∈ {tile, layer, full}`; boundary verifier fails if the lowering materializes above it | Materialization extent is **`cosize`** of the layout the consumer sees. One op |
| [`SPARDA_REVIEW.md`](SPARDA_REVIEW.md) §III.3 item 3 | "GQA-fold-to-rows layout transform… expressible with the Decision #15a `layout` attribute" | Folding a head group into the sequence axis is `group_modes` / `logical_divide`. It is **not** expressible today — see §3.1 |
| [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) §3.2 | block-rasterization knob; "the cheapest large lever in GEMM codegen and we are not pulling it" | A rasterization **is** a layout: `composition(grid_identity, raster_layout)`. Written once, not once per emitter |
| [`GAME_THEORY_PLAN.md`](GAME_THEORY_PLAN.md) G1b | generic `butterfly_transform` + `coalition` layout value + **one** shared butterfly/FFT tiling pass (the sanctioned Decision #31 consolidation) | A butterfly's exchange pattern is a stride permutation — a layout composition. A shared pass needs a shared representation to be shared *in* |
| [`AUTODIFF_NEXTGEN_PLAN.md`](AUTODIFF_NEXTGEN_PLAN.md) §6 (added 2026-08-18) | a length-`(k+1)` **derivative-coefficient axis** carried on tiles, with planar-vs-interleaved storage, a footprint ceiling, jet-epilogue fusion legality, and one triangular Cauchy-convolution index expression per backend | Every clause is this algebra: attach the axis = `logical_product`; storage form = **mode regrouping** (§2's product-variant result); footprint = `cosize`; fusion legality = the same `⊑` factorization FORGE asks for; digest equality = `coalesce`; the emitted index math = `crd2idx`. Written against today's substrate (a two-valued `LayoutOrder` enum + ~25 string-template index sites, §3.1) a jet emitter would hand-roll coefficient indexing **per backend** — reproducing the duplication L4 exists to consolidate |

Against [`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md): layout algebra is
**not a tenth investment — it is the missing implementation under S9**, and the
enabling mechanism for S6 and two unowned rows. S9 is already flagged "**entire
pair — no rows**"; its `⊑` operator is precisely the piece with no
implementation strategy, and layout algebra is the only candidate that serves
S9, SparDA item 3, TileSight §3.2, and G1b with one mechanism instead of four.

### 3.1 The measured evidence that the gap is real

Not theoretical. [`python/tessera/__init__.py:2914`](../../../python/tessera/__init__.py#L2914)
— `rearrange`, the op that would carry SparDA's fold — **fails closed on exactly
the spec SparDA needs**, and says so:

> `an einops-style spec is not interpreted and must not be silently ignored`

Its whole vocabulary is a top-level axis permutation or
`_IDENTITY_LAYOUTS = {row_major, identity, c, contiguous}`. It cannot express a
mode split `(h d) -> h d` or a fold `b h s d -> b s (h d)`. That fail-closed
behaviour is *correct* (Decision #21a; it replaced a silent wrong answer) — but
it is a placeholder where an algebra belongs.

Three more measurements from the same pass over the tree:

- **No layout-algebra vocabulary exists anywhere.** A search for
  `composition|logical_divide|zipped_product|complement|right_inverse|crd2idx|idx2crd`
  across `src/` and `python/` returns nothing.
- **`#tile.layout`** ([`TileOps.td:1140`](../../../src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td#L1140))
  is a flat `ArrayRefParameter<int64_t>` shard/replica/offset **attribute** —
  no nesting, no dynamic leaf, no algebra. Its ~10 consumer sites only read it
  to compute a footprint or match a special case.
- **The synthesizer, which actually generates kernels, has layout as a
  two-valued enum** (`LayoutOrder.ROW_MAJOR | COLUMN_MAJOR`,
  [`executable_layout.py:19`](../../../python/tessera/compiler/emit/executable_layout.py#L19))
  and index arithmetic as string templates — `A[row * K + k]`, `B[k * N + n]`,
  repeated across ~25 sites in
  [`apple_msl.py`](../../../python/tessera/compiler/emit/apple_msl.py). The
  seam named in `CLAUDE.md` appears here in measurable form: **Tile IR has a
  structured placement object; the compiler that emits code has an enum.**

### 3.2 The rasterization knob — mechanism closed, lever still unpulled

[`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md) §1 and
[`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) §3.2 both stated the
block-rasterization knob has **no emitter consumer**. That half is stale, and
both documents are corrected in this PR.

**Verified by execution, not inspection.** Driving
[`emit/nvidia_cuda.py`](../../../python/tessera/compiler/emit/nvidia_cuda.py)'s
`_raster_launch` produces materially different block-index code per order:

```text
row_major     int mt=blockIdx.x*16, nt=blockIdx.y*8,
column_major  _tsr_mt = blockIdx.x % ((M+15)/16); _tsr_nt = blockIdx.x / ((M+15)/16)
grouped_m     _tsr_raster_pp = 4 * ((N+7)/8);  … panel walk …
```

All four emitters consume `tile_rasterization.py`
(`emit/nvidia_cuda.py`, `msl_gemm_emit.py`, `apple_gemm_schedules.py`,
`rocm_schedule.py`), and the MLX-inherited `swizzle_log` heuristic was retired
rather than silently promoted.

**But TileSight's substantive conclusion survives intact, and it is worth being
precise about why.** `raster_order` is **carried, not swept** — it defaults to
`row_major` in [`autotune_v2.py`](../../../python/tessera/compiler/autotune_v2.py)
and in every emitter, and automatic enumeration is deliberately withheld pending
an architecture-owned correlation/retain verdict, because ROCM-CALIB-1
established that an unvalidated locality metric must not change a production
raster choice. So the lever is now **expressible on every backend and still not
pulled**; what moved is the blocker — from codegen to measured device evidence.

Two consequences for this plan: L4 shrinks (the emitter plumbing exists; L4 is a
consolidation onto shared algebra, not new capability), and the measured half
stays architecture-owned and out of L4's scope entirely.

---

## 4. Do we need it?

**Yes — a strict subset, consumer-driven.** The counter-arguments are real and
are recorded rather than dismissed:

**Against.** (a) Decision #29 — on day 1 it has zero consumers, and building an
algebra before a consumer is the declaration-without-a-consumer failure the
governance rule exists to stop. (b) The C++ MLIR lane lowers to `func.call`
while the Python synthesizer generates the kernels, so building this in C++
first hands a capability to the compiler that **does not emit code**, widening
the seam. (c) NVIDIA's is backed by a mature vendored library; Tessera builds
from zero.

**For.** (a) Six independent asks already written down, with six different
proposed mechanisms — the Decision #30 "eighth bespoke walker" anti-pattern,
before any of them is built. The sixth (autodiff's coefficient axis) arrived
after this assessment and was caught by it, which is the counter-argument to
(a) below: the algebra is no longer a day-1-zero-consumer investment. (b) It is **entirely host-free**: pure integer
functions, exhaustively checkable over small domains. NVIDIA's whole algebra was
validated here in ~40 lines with no hardware, which is Decision #19's discipline
applied to index arithmetic. (c) The scope is four primitives, not 63 ops.

---

## 5. Build sequence — LAYOUT-ALG-1 (L0…L5)

**Bound to [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) as
`LAYOUT-ALG-1` on 2026-08-16.** That plan owns global order and promotion; this
document owns the mathematical verification and the acceptance criteria below.
Sizing is an estimate, not a measurement.

**Implementation checkpoint (2026-08-16).** The first native L1 ABI slice and
the required L2 consumer are executable: the ordinary build produces
`libtessera_layout_algebra`, Python loads its versioned C ABI with no semantic
fallback, compact `size`/`cosize`/`crd2idx`/`idx2crd` are exhaustively checked
through size 64 on both sides of the ABI, and `rearrange` now executes the GQA
fold `b h s d -> b s (h d)` plus its explicitly factored inverse. The native
planner parses nested mode groups and unresolved dynamic leaves fail closed.
This is **not the whole L1 surface yet**: general nested
`composition`/`complement`/`coalesce`/`right_inverse`, product/divide variants,
and general slice remain open and L1 therefore remains landing. L3 and L4 must
not treat this first consumer as the completed factorization algebra.

**Structured-L1 checkpoint (2026-08-23).** The C ABI now transports a nested
layout as paired preorder node trees (`value`, `child_count`), rather than a
layout string or a Python shadow implementation.  Native `coalesce` handles
the documented `(2,(1,6)):(1,(6,2)) → 12:1` and by-mode
`(3,(4,5)):(8,(1,4)) → (3,20):(8,1)` forms; bounded static `composition`
materializes the documented boundary-crossing result
`((2,2),3):((24,2),8)` and exhaustively rechecks its function before returning
it.  Python is ctypes transport only.  `coalesce` now carries an explicit
dynamic `-1` residue without merging through it; the compact bijective subset
has native `right_inverse`/`left_inverse`, and static affine layouts have the
documented cotarget and cosize-derived `complement` forms.  Dynamic
residues now materialize their statically knowable radix prefix and retain one
explicit dynamic tail.  General scalar composition is no longer capped by an
enumeration table and uses affine outer evaluation beyond its declared domain.
`slice` returns a residual layout plus an explicit offset carrier, and logical
divide accepts non-compact tilers through
`composition(source, (tiler, complement(tiler, size(source))))`.  The static
rectangular product family has all six structural variants and its divide
counterpart has logical/zipped/tiled/flat variants; each is constructed by the
C++ authority and preserves the exact documented nesting.  Tile IR now carries
the remaining non-affine boundary as `#tile.composed_layout`: recursive
`ArrayAttr` outer shape/stride trees, a tuple of paired recursive basis
shape/stride trees, and one explicit offset per outer coordinate. `-1` remains
a dynamic residue, while the verifier rejects invalid leaves, malformed trees,
profile mismatches, and basis/offset rank disagreement. The shared
`tile.materialize_composed_layout` proof boundary now serializes every outer
and tuple-basis shape/stride tree into the L1 C++ ABI and requires
`tessera_layout_coalesce_v1` to accept each canonical component. The first
addressable subset binds one explicit i64 coordinate per flattened outer mode,
followed by dynamic leaves in canonical outer-shape, outer-stride, basis-shape,
basis-stride order, and returns the scalar-affine offset
`Σ outer_stride[i]·(offset[i] + basis_stride[i]·coord[i])`. Nested outer trees
and dynamic shape/stride leaves are therefore materializable after runtime
substitution. NVIDIA SM120 and the ROCm Tile lowering re-run the C++ proof,
lower that offset to arithmetic, and feed the existing
`tile.view{tile.linear_base}` address contract. Neither chooses a new
fragment/register/shared-memory layout. Mixed-radix tuple basis maps now lower
as exact `remui`/`divui` digit extraction. A tuple-valued codomain is represented
by `tile.materialize_composed_layout_tuple` as a product of independently
proof-bearing scalar components over one coordinate domain; NVIDIA and ROCm
expand only that product seam and reuse the scalar materializer. Dynamic tuple
codomains and genuinely non-separable regroupings remain carrier-only and fail
closed; Apple and x86 cannot mistake the carrier for `#tile.layout` or silently
lower it.
The static f16 m16n8k16 row-major-A/column-major-B subset has exact-device
evidence on RTX 5070: nonzero A-row and B-column origins flowed through
`tile.view → fragment_pack → mma.sync` and matched the selected NumPy panels.
The corresponding ROCm f16 m16n16k16 static subset has exact gfx1151 evidence:
the shared operation produced nonzero per-lane A-row/B-column bases, which
flowed through `tile.view → fragment_pack → rocdl.wmma` and matched NumPy after
HSACO serialization and HIP launch. The follow-on bounded-dynamic gfx1151
package and device proof are recorded below; neither backend inherits the
other's physical schedule.
The canonical gfx1151 scheduled-matmul consumer now reaches that same proof
boundary without a hand-authored fixture: only when its Schedule-owned
`M/N/K` operands remain positive `arith.constant` values does the typed WMMA
generator emit row-major A `[M,K]:[K,1]` and B `[K,N]:[N,1]`
`tile.materialize_composed_layout` bases before `tile.view → fragment_pack →
tile.mma`. Dynamic dimensions use runtime `tensor.dim` extents and retain the
same target-owned WMMA materializer. Dynamic outer materialization and
separable tuple-valued materialization are closed for the target-proven subset
below. Non-compact regroupings beyond logical divide and dynamic/non-separable
tuple codomains remain open.
The parallel NVIDIA
`tessera-target-opt` registers target/upstream dialects but not Tile IR, so a
leak is now a parser error rather than an assertion about an all-dialects host.

**SM120 block-coordinate closure (2026-08-24).** Canonical scheduled f16
matmul now emits the registered, pure, two-i64-result
`tessera_nvidia.block_coordinate` boundary instead of reading NVVM CTA state in
the shared producer. Its verifier admits exactly `sm_120`, physical tile
`16×8`, and `column_major_xy`; NVIDIA lowering alone maps the results to
`ctaid.y*16` and `ctaid.x*8`. Those SSA bases feed both composed-layout
materializations, fragment packing, accumulator-carrying K loop, and final
store. RTX 5070 numerical proof passed for `16×32 @ 32×8`, `32×32 @ 32×16`,
and `48×64 @ 64×24` (maximum absolute error at most `5.97e-7`). A seven-sample,
500-repetition CUDA-event comparison found the scheduled/direct ratios
`1.003`, `1.025`, and `1.005`, respectively—within launch-scale noise, so the
macro selector remains disabled. One separate Nsight capture at `48×64×24`
reported scheduled/direct resources of 44/36 registers per thread, 1024/1024
allocated shared-memory bytes, and equal 2.08% active-warp occupancy. The
profiled durations (2.528/4.640 us) are resource-run observations, not selector
timings; selector authority remains the clean CUDA-event matrix.

**SM120 loop-invariant and reuse analysis (2026-08-24).** The native Tile-to-PTX
pipeline now runs loop-invariant-code motion while the scheduled reduction is
still structured SCF. This moves lane decomposition, block bases, and static
composed-layout address terms out of the K loop without changing its typed
accumulator recurrence; the pipeline ordering is unit-tested and versioned in
the native cache contract. All five RTX 5070 scheduled numerical cases remain
green. Larger clean CUDA-event cases separate useful work from launch
quantization: scheduled/direct ratios were `1.002` for `128x128 @ 128x128`,
`0.929` for `128x256 @ 256x64`, and `0.901` for `256x256 @ 256x256`. The first
case is tied and the rectangular scheduled sample contains an outlier, so this
single-session sweep is not enough to define a selector boundary; the selector
remains fail-closed. A separate Nsight Compute capture for `256x256 @ 256x256`
reported scheduled/direct duration `9.088/10.528 us`, DRAM bytes
`303104/634368`, executed instructions `187392/510976`, and registers per
thread `40/35`, with equal 1024-byte allocated shared memory and 21.63% active
warps. These profiler values diagnose the route; they are not selector timing.
A repeated largest-case CUDA-event run produced ratio `0.914`, but its
scheduled sample coefficient of variation was 3.89% versus 1.76% for direct,
which independently fails the low-variance promotion requirement.

The mathematical limit in that packet was one independent warp per `16x8`
output tile: at `M=N=K=256`, it logically reloaded 6,291,456 input bytes versus
262,144 unique input bytes, a 24x reuse gap before cache effects.

**SM120 macro-CTA async shared-panel closure (2026-08-24).** The follow-on
implements that physical boundary as the registered
`tessera_nvidia.macro_cta_matmul` Target IR operation; it does not reinterpret
the existing `16x8/column_major_xy` block-coordinate operation. Its verifier
admits exactly `sm_120`, a `32x32` CTA tile, four warps,
`quadrant_2x2_two_n_tiles` ownership, f16 or bf16 input storage, f32
accumulation, `m16n8k16` MMA, two async shared stages, and M/N/K zero-fill tails.
NVIDIA lowering maps the warps onto eight `16x8` results. The 128 threads each
issue one aligned 16-byte A or B `cp.async`; commit/wait plus CTA barriers
establish visibility and prevent slot reuse while a consumer is live. An
out-of-range M/N vector uses source-size zero and a safe source pointer, so the
hardware zero-fills its shared destination. A partial final K panel preserves
the fixed `k16` instruction contract: row-aligned packed inputs use the
`cp.async` source-size operand, while an arbitrary row stride that cannot prove
16-byte alignment uses the masked scalar shared-panel materializer.

Aligned and ragged FP16 cases plus ragged BF16
`257x512 @ 512x257` pass the independent FP32 oracle on RTX 5070. The retained
nine-sample, 1000-repetition WSL event packet uses a conservative 67,108,864
FLOP threshold: its three eligible scheduled/direct ratios are `0.721`,
`0.453`, and `0.429`, with every eligible sample CoV below 3%. Repeated lower
bands exposed launch-scale variance, so they retain the typed fallback even
when their median favored the macro route. This is a route-local pruning
decision only; WSL is not `target_perf` selector authority.

After numerical proof, Nsight Compute at `256^3` records scheduled/direct
duration `8.22/9.57 us`, registers `48/35`, static shared memory `4096/0`
bytes, no spills, achieved occupancy `11.1%/21.6%`, and L1/TEX throughput
pressure about `29%/80%`. The result is mathematically consistent with panel
reuse: the macro route trades occupancy and synchronization for fewer repeated
global/L1 requests. Profiler duration is diagnostic evidence; the clean event
matrix owns the pruning decision.

The static Graph package supplies runtime M/N/K scalars and now proves both an
aligned partial K panel (`K=520`) and an arbitrary-alignment masked panel
(`K=513`) on RTX 5070. It still assumes the canonical contiguous A row-major /
B column-major physical ABI. Those runtime-bound proof operands are closed by
the bounded-dynamic checkpoint below. This paragraph records the earlier static
checkpoint; the alignment-safe dynamic macro-CTA specialization is closed in
that subsequent checkpoint.

**SM120 bounded-dynamic scalar-affine closure (2026-08-24).** Rank-2 dynamic
f16/bf16 Graph matmul is admitted only with a positive
`shape_bounds=[M,N,K]` envelope. The canonical scheduled producer keeps runtime
`M/N/K`, emits bounded dynamic-leading-dimension views, and supplies
`M/N/K/LDA/LDB` as dynamic leaves of nested scalar-affine A/B composed layouts.
The versioned CUDA package exports `M/N/K/LDA/LDB/LDD`, validates element
strides A `(LDA,1)`, B `(1,LDB)`, D `(LDD,1)`, and copies overflow-checked
physical spans rather than assuming compact arrays. NVIDIA lowering re-runs
the shared C++ proof before substituting those runtime strides.

On RTX 5070, the canonical package executed `17x19 @ 19x13` with `LDA=29`,
`LDB=31`, and `LDD=23` and matched an independent FP32 NumPy oracle, including
M/N edge masks and the final K tail. An identity probe exposed a real target
bug during closure: bounded column-major B packing advanced its second vector
lane by `LDB` rather than contiguous K. The corrected packer constructs
role-specific relative row/column lane coordinates before applying the proven
affine stride. A second bug was exposed when mixed-radix materialization became
the authority: the dynamic identity basis had been encoded with extent one,
which correctly evaluates to `coord % 1 = 0` and repeated every eight-column
tile. The producer now supplies the runtime logical extent to both the outer
and basis leaves. This proof covers nested outer trees and dynamic
scalar-affine leaves.

The target-owned macro-CTA route now also accepts the bounded dynamic ABI when
its bound satisfies the retained work threshold. Because arbitrary runtime row
strides do not prove 16-byte alignment, this specialization uses one-stage
masked scalar shared A/B panels plus CTA barriers rather than `cp.async`.
`257x127 @ 127x259` with odd `LDA/LDB/LDD=139/137/269` passes the FP32 oracle
on RTX 5070. A single post-correctness NCU launch recorded 17.536 us, 40
registers/thread, 82.85% L2 sector hit rate, and 13.89% active-warps; this is
diagnostic evidence, not selector authority.

gfx1151 now admits the same bounded Graph envelope while retaining its own
compact row-major launch ABI and multi-wave WMMA schedule. The canonical
`37x35 @ 35x29` package, bounded by `[64,64,48]`, compiled through
Graph→Schedule→Tile→ROCDL→HSACO and matched the independent FP32 oracle on the
Radeon 8060S. No CUDA stride, warp, or staging decision transfers to that path.

**L3/SO-4 closure (2026-08-24).** The versioned C ABI now decides layout
factorization and residency directly. Compact bijective partitions are proven
symbolically, including FORGE-scale images; bounded non-compact layouts use an
exact finite-image proof and larger unresolved cases fail closed. Residency is
`cosize(layout) * element_bytes <= capacity_bytes`, so padding and holes cannot
be undercounted as logical `size`. `ScheduleObject` schema v2 attaches immutable
materialization proofs containing layout digest, factorization decision,
materialized elements/bytes, capacity, residency tier, alias set, and lifetime.
Positive and negative factorization/capacity fixtures pass without a device.

**L4 consolidation checkpoint (2026-08-24).** Raster emission was already one
shared `tile_rasterization.py` authority with exhaustive bijection and compiled
C-equivalence tests. Rank-2 physical `crd2idx` is now one C++
`materializeLinearIndex` authority consumed by the CUDA fragment/load/store
path and the gfx1151 generated WMMA A/B/output path; rebuilding both backends
preserves their established arithmetic and both exact-device dynamic rows stay
green. The x86 f32/f64/bf16/u8s8 core GEMMs now consume the same header-only
rank-2 authority, with odd-shape/vector-tail numerical proof on an AVX-512
Ryzen AI Max+ 395 host. AVX2 hosts fail closed before loading that image. L4
remains landing only for the Apple text-emitter migration and device proof.

**L5 x86-consumer checkpoint (2026-08-24).** The x86 Target pass now re-runs
the shared native proof and materializes the complete scalar-output set admitted
by the carrier: static and bounded-dynamic outer maps, nested mixed-radix basis
maps, and static tuple codomains expressed as products of scalar maps. Its exact i64
arithmetic executes through LLVM on both AVX2 and AVX-512 hosts. Runtime guards
reject negative or out-of-range coordinates, nonpositive dynamic extents, and negative dynamic
strides. Non-affine and non-separable tuple maps remain deliberately
unmaterializable rather than acquiring weaker CPU semantics.

The scheduled contract also carries real Graph value edges for optional fp32
`bias[N]` and fp32 `residual[M,N]`, the ordered
matmul→bias→activation→residual epilogue, and f16 reduced output. The CUDA
descriptor expands from A/B/D to A/B/bias/residual/D only when those operands
are present, and the exact RTX test proves bias+ReLU+residual with f32
accumulation followed by an f16 store. ReLU, GELU, and SiLU are verifier-owned
activation values; the exact test uses ReLU so its independent oracle does not
depend on matching two different transcendental approximations.

A post-correctness Nsight Compute capture for the aligned `257x520x257` tail
records scheduled/direct duration about `15.3/23.2 us`, registers `56/35`,
static shared memory `4.10 KiB/0`, and achieved occupancy `14.0%/23.3%`.
Logical input redundancy drops from `26.19x` to `10.09x`. The reports are
`/tmp/tessera-sm120-k520-{scheduled,direct}.ncu-rep`; profiler duration remains
diagnostic and does not satisfy the missing bare-metal event-packet gate.

**Seven-action closure ledger (2026-08-24).** This keeps implementation proof,
performance policy, and future ABI work separate:

1. **Selector boundary — route-local closed, global open.** The scheduled
   package uses the conservative 67,108,864-FLOP pruning boundary above. WSL
   packets cannot update global `target_perf`; repeat the same packet bare
   metal before a global selector row changes.
2. **Async shared staging — closed for the exact contract.** Two slots,
   `cp.async` commit/wait, CTA lifetime barriers, and independent NCU evidence
   are implemented for f16/bf16 m16n8k16.
3. **Shape/stride breadth — bounded dynamic closed.** Arbitrary M/N/K tails are
   zero-filled or masked and exact-device proven. Bounded dynamic Graph extents
   and arbitrary `lda`/`ldb`/`ldd` are now proven through the narrow typed
   route. The alignment-safe dynamic macro-CTA specialization is exact-device
   proven; explicit pointer-offset alignment metadata remains open.
4. **Dtype/epilogue breadth — bounded closure.** BF16 travels through the same
   Graph→Schedule→Target package and is device-proven. Scheduled fp32 bias,
   ReLU/GELU/SiLU activation, fp32 residual, and f16 reduced output now have a
   widened Graph/Schedule/Tile/CUDA descriptor ABI; the combined ReLU case is
   exact-device proven. Other output dtypes and epilogue families remain open.
5. **Composed-layout breadth — separable target subset closed.** Nested
   outer trees and static/dynamic scalar-affine leaves lower only through the
   shared proof plus a target consumer. Mixed-radix basis maps and static tuple
   codomains materialize as exact component products. Dynamic tuple codomains
   and non-separable regroupings remain representable and fail closed.
6. **ROCm counterpart — static and bounded-dynamic numerical proof closed;
   selector open.** The canonical gfx1151 scheduled package selects the existing
   target-owned multi-wave LDS pipeline and records
   `gfx1151_multiwave_lds_wmma_2x4`. Radeon 8060S exact-device cases
   `32x32x32`, ragged `17x19x23`, and bounded-dynamic `37x35x29` match the FP32
   oracle. Its WSL host-wall
   timing remains selector-ineligible, so no global ROCm threshold changes.
7. **Release/document closure — landing.** The Target IR spec, all four backend
   plans, retained benchmark packet, tests, and top-level README describe the
   same boundary. This item closes only after focused drift gates, the broader
   unit lane, claim lint, and graph refresh pass.

Dynamic/non-separable tuple codomains remain carrier-only until a target
materializer proves them. Dynamic scalar-affine outer/stride leaves and static
separable tuple codomains are no longer part of that open set on SM120/gfx1151.

### L0 — the home: C++ first, Python binds to the dylib (decided 2026-08-16)

**Decided by the repo owner.** The algebra is **one C++ implementation** in the
Tessera runtime/support library; `python/tessera/compiler/` reaches it through
the existing ctypes ABI. There is no second implementation and therefore no
differential-oracle pair to maintain — this is Decision #31 satisfied by
construction (one implementation per boundary) rather than by declared-oracle
exemption.

An earlier draft of this document recommended Python-first with C++ deferred to
L5. That recommendation is **withdrawn**; it optimized for the synthesizer's
immediate convenience and accepted a duplication that Decision #31 exists to
prevent. The decision taken is the stronger one against the governance rules —
it makes MLIR passes (L3, L5) first-class consumers rather than deferred ones,
and it means the FORGE `⊑` verifier and the emitter index math query the *same*
code, which is the whole point of building a shared substrate.

**The cost is real and is not being wished away.** Coupling `emit/` to a build
artifact is the failure mode the Apple dylib already demonstrates: a stale or
absent dylib currently fails 32 Apple tests as "requires a fresh dylib" rather
than skipping. Layout algebra sits under *every* emitter, so the same failure
would be broader. Two acceptance criteria fold that risk into L1 rather than
deferring it:

- **A1 — diagnose, never silently degrade.** The Python binding raises one
  named, actionable diagnostic when the symbol is missing or stale (Decision
  #21a: this is a semantic dependency, so it fails closed). No NumPy fallback
  path, because a fallback is a second implementation wearing a disguise.
- **A2 — the build dependency is declared and gated.** `ninja -C build` must
  produce the symbol as part of the ordinary target set, and a unit test asserts
  the binding loads, so the failure surfaces at test time with a fix instruction
  rather than inside an emitter.

This reorders the build sequence below: the C++ kernel and its binding are now
both inside L1, and L5 shrinks to the MLIR *carrier* (types, interface,
fold-static pass) since the algebra it would have introduced already exists.

### L1 — C++ algebra kernel + ctypes binding + exhaustive proof (~2 weeks)

Eleven operations: `composition`, `complement`, `coalesce`, `right_inverse`,
`logical_divide`, `zipped_divide`, `logical_product`, `size`, `cosize`,
`crd2idx`/`idx2crd`, `group_modes`/`flatten`, `slice`. Nested `(shape, stride)`
with **static-or-dynamic leaves from day one** — partial staticness is what
makes fold-static free later, and retrofitting it is a type change.

Four base types (`Shape`, `Stride`, `Layout`, `Coord`) plus a narrow C ABI
carrier for residual layout + offset and Tile IR's structured tuple-valued
`#tile.composed_layout` attribute. It is sufficient to preserve scalar and
tuple composition without pretending every basis map is affine. `IntTuple`,
`Tile`, and a value-producing full `ComposedLayout` type remain deferred; the
existing `#tile.swizzle` split remains intact.

Per L0 this is C++ (`src/` support library, no MLIR dependency — the MLIR
*carrier* is L5, and the algebra must be usable without loading a dialect), plus
the ctypes binding under `python/tessera/compiler/`. Sizing is ~2 weeks rather
than the ~1 week an in-language implementation would take: the extra week is the
ABI surface, the build wiring, and A1/A2.

*Acceptance:*

- every op exhaustively verified against brute-force evaluation over all layouts
  up to size 64, **on both sides of the ABI** — the C++ unit test and the Python
  binding test run the same corpus, so a marshalling bug cannot hide;
- §1.2's three defects carried as negative fixtures
  ([`test_layout_algebra_contracts.py`](../../../tests/unit/test_layout_algebra_contracts.py)
  is the existing corpus and moves to driving the real implementation);
- A1 (named fail-closed diagnostic, no fallback path) and A2 (declared build
  dependency + binding-loads test) from L0;
- host-free — no device, no MLIR, no dialect load.

**Do not land L1 without L2 committed** — otherwise it is a Decision #29
violation by construction.

### L2 — first consumer: `rearrange` / GQA-fold (~2 days)

Replace the fail-closed einops rejection in `rearrange` with real mode
regrouping. Smallest possible consumer, immediate user-visible win, and it
converts a documented `ValueError` into a working op.

*Acceptance:* `b h s d -> b s (h d)` and its inverse execute and round-trip; the
existing fail-closed behaviour survives for genuinely malformed specs (the
Decision #21a property must not regress). Consumer: `SPARDA_REVIEW.md` §III.3
item 3.

### L3 — the `⊑` decision procedure (~1–2 weeks)

Implement the FORGE locality lattice's `⊑` as layout factorization, and
`residency` as `cosize`. Keep the lattice as the **declared interface** — it is
the right user-facing vocabulary; layout algebra makes `block` precise instead
of a promise.

*Acceptance:* FORGE's four regimes reproduce as lattice positions; negative
fixtures for tied weights, `state_locality = row` (Adafactor), MoE-routed
weight. Host-free — which is the point, since no fleet machine can hold FORGE's
62 GB peak. Consumers: FORGE W1/W2, and S9 in `CORE_SUBSTRATE_VIEW.md`.

### L4 — emitter index math (~1–2 weeks, smaller than first scoped)

Route the emitters' hardcoded `A[row*K+k]` / `B[k*N+n]` templates through
`crd2idx`, and re-express the rasterization orders as `composition(grid,
raster_layout)` instead of four hand-written block-index emitters. This is where
the payoff for math ops lands: the epilogue-fusion track, the `simdgroup_matrix`
path, and the ROCm typed lane stop re-deriving index arithmetic per emitter.

**Rescoped down after §3.2's correction.** The original estimate assumed the
emitters had no way to express a rasterization. They do —
`tile_rasterization.py` is consumed by all four. So L4 is a *consolidation* of
existing working code onto the shared algebra, not new capability, and it
inherits `tile_rasterization.py`'s own tests as the regression net.

*Acceptance:* generated kernels **bit-identical** to today's output for every
currently-reachable `(raster_order, raster_group)` combination — a pure-refactor
proof, host-free, and a strictly stronger gate than the row-major-only check
first proposed. Any measured non-default raster choice remains **out of scope
for L4**: per §3.2 that is blocked on an architecture-owned correlation/retain
verdict (ROCM-CALIB-1), not on codegen, and it does not transfer between
architectures.

### L5 — MLIR carrier (~1–2 weeks, smaller under L0's C++-first decision)

Extend `#tile.layout` from flat `ArrayRefParameter<int64_t>` to nested with
dynamic leaves; add `MaybeStaticTypeInterface` + a fold-static pass, with the
attribute delegating to L1's algebra rather than reimplementing it.

**Rescoped down by L0.** With the algebra already in C++, L5 is the *carrier*
only — types, interface, and the fold pass — and there is no second
implementation to differential-test, which was the bulk of the original
estimate. This is the concrete dividend of the C++-first decision.

CUDA, ROCm, and x86 now consume the shared materializable set through
architecture-owned lowering, including static tuple products. Apple remains the physical-consumer tail; no
other backend's exact-device or host evidence transfers to Metal.

*Sequencing:* this follows W1.1 step 4, it does not precede it — the same
Decision #31 ordering argument [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md)
already makes for `#tile.mma_desc`.

### Independent of the above

**`tessera-target-opt`, a negative-scoped driver** (~hours). Register only the
Target IR dialects plus upstream, without `tessera` / `tile`, so Tile IR leaking
into a Target IR fixture is a parse error rather than something a `CHECK-NOT`
has to anticipate. Strengthens Decision #19 by the same argument its own
standing lesson makes. Does not depend on any L-item.
