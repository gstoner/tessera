---
last_updated: 2026-08-16
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
— independent integer evaluation, no cutegen dependency, no numpy. 25 tests.

**Bottom line:** the algebra is mathematically sound and its documentation has
three defects. The surface is 63 ops but the mathematics is **four primitives
plus regrouping**. Tessera does not need CuTe IR, but it does need layout
algebra — because five independent in-tree assessments each ask for a different,
weaker mechanism for the same question, and none of them names it.

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
variants and all four divide variants are the same leaf multiset in different
bracketings.

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

This is the finding that matters. **Four assessments and one plan independently
ask for a piece of layout reasoning, and each proposes a different, weaker
mechanism.** None names layout algebra, because none was looking at the others
through that lens.

| Source | What it asks for | What the question actually is |
|---|---|---|
| [`FORGE_ASSESSMENT.md`](FORGE_ASSESSMENT.md) W1 | read-locality lattice `coordinate ⊏ row ⊏ block ⊏ tensor ⊏ layer ⊏ global`; fusion legal iff `consumer.read_locality ⊑ producer.tile_partition` | Does the consumer's read layout **factor through** the producer's tile partition — `∃X. read = partition ∘ X`? A six-point chain approximating a question `composition` + divisibility **decides exactly** |
| [`FORGE_ASSESSMENT.md`](FORGE_ASSESSMENT.md) W2 | `tessera.residency ∈ {tile, layer, full}`; boundary verifier fails if the lowering materializes above it | Materialization extent is **`cosize`** of the layout the consumer sees. One op |
| [`SPARDA_REVIEW.md`](SPARDA_REVIEW.md) §III.3 item 3 | "GQA-fold-to-rows layout transform… expressible with the Decision #15a `layout` attribute" | Folding a head group into the sequence axis is `group_modes` / `logical_divide`. It is **not** expressible today — see §3.1 |
| [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) §3.2 | block-rasterization knob; "the cheapest large lever in GEMM codegen and we are not pulling it" | A rasterization **is** a layout: `composition(grid_identity, raster_layout)`. Written once, not once per emitter |
| [`GAME_THEORY_PLAN.md`](GAME_THEORY_PLAN.md) G1b | generic `butterfly_transform` + `coalition` layout value + **one** shared butterfly/FFT tiling pass (the sanctioned Decision #31 consolidation) | A butterfly's exchange pattern is a stride permutation — a layout composition. A shared pass needs a shared representation to be shared *in* |

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

**For.** (a) Five independent asks already written down, with five different
proposed mechanisms — the Decision #30 "eighth bespoke walker" anti-pattern,
before any of them is built. (b) It is **entirely host-free**: pure integer
functions, exhaustively checkable over small domains. NVIDIA's whole algebra was
validated here in ~40 lines with no hardware, which is Decision #19's discipline
applied to index arithmetic. (c) The scope is four primitives, not 63 ops.

---

## 5. Build sequence — LAYOUT-ALG-1 (L0…L5)

**Bound to [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) as
`LAYOUT-ALG-1` on 2026-08-16.** That plan owns global order and promotion; this
document owns the mathematical verification and the acceptance criteria below.
Sizing is an estimate, not a measurement.

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

Four types (`Shape`, `Stride`, `Layout`, `Coord`), not CuTe's eight — skip
`IntTuple` / `Tile` / `ComposedLayout`, and keep the existing `#tile.swizzle`
split rather than adopting `composed_layout` yet.

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

*Sequencing:* this follows W1.1 step 4, it does not precede it — the same
Decision #31 ordering argument [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md)
already makes for `#tile.mma_desc`.

### Independent of the above

**`tessera-target-opt`, a negative-scoped driver** (~hours). Register only the
Target IR dialects plus upstream, without `tessera` / `tile`, so Tile IR leaking
into a Target IR fixture is a parse error rather than something a `CHECK-NOT`
has to anticipate. Strengthens Decision #19 by the same argument its own
standing lesson makes. Does not depend on any L-item.
