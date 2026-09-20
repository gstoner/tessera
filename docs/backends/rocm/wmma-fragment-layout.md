# RDNA3/RDNA4 WMMA fragment layout — the normative contract

Every mapping below is stated once, here, because getting one of them subtly
wrong does not fail loudly: the kernel runs, most elements land, and a few
tiles are silently wrong somewhere else in the program. This page is the
statement the Tessera ROCm generators are written against, with the evidence
for each claim.

## 1. `lane` means the intra-wave lane, never a flat thread index

**Every formula on this page uses `lane = threadIdx.x % wavefront_size`** (32 on
RDNA3/RDNA3.5/RDNA4 in wave32). It is not `threadIdx.x`.

This is the contract that breaks quietly. In a single-wave block the two are
equal, so a kernel written with a flat index is correct and stays correct until
someone gives the kernel a multi-wave workgroup. Then `threadIdx.x / 16` ranges
`0..(threads/16 - 1)` instead of `0..1`, and every wave above the first
reads and scatters outside its own 16×16 tile. The corruption surfaces far from
the WMMA — in one reported case as NaN several kernel phases downstream, landing
first in uninitialized LDS padding. Abstractions layered on these formulas
inherit the same implicit contract.

**In Tessera** the intra-wave lane is computed in exactly one place,
`computeLaneCoordsFromThread` in `TileToROCM.cpp`, as `threadIdx.x %
physical.waveSize`, and every typed fragment pack, MMA and store goes through
it. That is why the LDS-staged typed body may use 128- and 256-thread
workgroups safely.

*Evidence*: the LDS-staged multi-wave body (2×2 and 4×2 waves, i.e. 128 and 256
threads per block) matches the numpy product at 256³, 1024³ and 511×513×509 on
both gfx1151 and gfx1201 —
`tests/unit/test_scheduled_matmul_consumers.py::test_rocm_lds_staged_package_executes`.
A kernel that used a flat index would fail those rows for every wave but the
first.

Kernels that compute lane coordinates themselves (the untyped directive body's
store path uses `threadIdx.x & 15` and `threadIdx.x >> 4`) are therefore
**single-wave only**, and are launched with 32 threads.

## 2. Accumulator layout, gfx12 wave32 — column-distributed

For `v_wmma_f32_16x16x16_f16` and the other 16×16 shapes, each lane holds 8
accumulator elements:

```
VGPR[lane][j] = C[(lane / 16) * 8 + j][lane % 16]        // j = 0..7
```

That is: **`lane % 16` selects the column (N)**, and `(lane / 16) * 8 + j`
selects the row (M). The fast-varying dimension across lanes is the column, not
the row. Reading it row-major instead matches only the 16 diagonal elements —
which is why a row-major misreading looks "nearly right" on a symmetric probe
and is wrong everywhere else.

RDNA3/RDNA3.5 wave32 distributes the same accumulator differently (row
`2j + lane/16`), which is why the two families need separate row formulas and
why gfx1151 evidence never transfers to gfx1201.

**In Tessera**: `materializeFragmentStore` in `TileToROCM.cpp` resolves each
element's row and column from the resolved fragment family, so the typed route
has one epilogue implementation across both layouts. The hand-written
generators that still compute rows themselves use
`rdna4 ? e + 8*half : 2*e + half` — the same formula (`GenerateWMMAFlashAttnKernel.cpp`,
`GenerateWMMALinearAttnKernel.cpp`).

*Evidence*: the typed matmul, attention and linear-attention device rows on
gfx1201. Independently confirmed twice outside this repo. A probe on a second
stack (Radeon AI PRO R9700, gfx1201, ROCm 7.14 nightly) against a CPU
reference of `A·Bᵀ` matched the column-distributed mapping on 256/256
elements across three cases, with the row-major reading matching 16/256 — the
diagonal, as expected. And a public RDNA4 WMMA guide states the same mapping
in the same form, `VGPR[lane][j] = matrix[(lane / 16) * 8 + j][lane % 16]`,
together with the same warning this page gives: a symmetric test matrix cannot
distinguish the two readings, so a probe must use asymmetric data. Three
independent statements of a contract whose failure mode is silent is the
reason to trust it.

## 3. A/B operand layout — K-major per lane

**Corrected 2026-09-19, then verified against AMD's own tool.** This section
previously stated the mapping below as the architecture's fragment layout. It is
that for some storage widths and not for others, and the difference matters.
Printed by `amd_matrix_instruction_calculator` (`-a gfx1201 -R -A -w 32`), the
**K distribution depends on the storage width**:

| storage | per-lane mapping | runs |
|---|---|---|
| 16-bit (f16, bf16) | `k = 8*(e>>2) + 4*(lane>>4) + (e&3)` | **four** |
| 8-bit (fp8, bf8, int8) | `k = 8*(lane>>4) + e` | **eight** |
| 4-bit (int4) | `k = 8*(lane>>4) + e` | **eight** |

For f16, lane 0 holds `k = 0,1,2,3,8,9,10,11` across four VGPRs at two elements
each, and lane 16 holds `4,5,6,7,12,13,14,15`. For fp8/int8 lane 0 holds
`k = 0..7` packed in two VGPRs and lane 16 holds `8..15`; for int4 the same eight
k values fit one VGPR as eight nibbles.

**So contiguous-eight — what `materializeFragmentPack` emits — IS the machine's
mapping at 8 and 4 bits, and is a relabeling only at 16 bits.** The
permutation-cancellation argument below is therefore load-bearing for exactly
one case, f16/bf16, and is not needed for the low-precision storages at all.

What `materializeFragmentPack` emits, at every width, is contiguous-eight:

```
a[h] = A[lane % 16][8 * (lane / 16) + h]        // A row-major [M][K]: row = lane % 16
b[h] = B[lane % 16][8 * (lane / 16) + h]        // B K-major   [N][K]: col = lane % 16
```

**This is legal, and deliberately better, for one reason that must not be
forgotten: the WMMA sums over K.** `C[i][j] = Σ_k A[i][k]·B[k][j]` is invariant
under any permutation of the K axis applied to *both* fragments, and the
instruction pairs slot with slot regardless of which k each slot is believed to
hold. Our loader applies one convention to A and B alike, so the permutation
cancels exactly. Contiguous-eight is then strictly cheaper to load: one 128-bit
`vector.load` per lane, where the machine's runs-of-four needs two loads or a
cross-lane shuffle.

**The hazard this creates, and it is live — at 16 bits.** A third fragment
source must adopt *our* convention where it differs from the machine's, and
nothing in the type system says so. The
one place they already collide is `GLOBAL_LOAD_TR_B128`, which delivers the
hardware's native permutation: the per-lane address derived in §8 is what
reconciles the two, and it was solved empirically against a measured mapping
rather than derived from this contract. That reconciliation is load-bearing.
Anything that changes either side — a new staging path, an LDS-resident
fragment, a pre-packed weight format — must be checked against both, because a
mismatch is numerically silent in exactly the cases where K happens to be
symmetric.

**The free transpose.** Because our A and B fragments share one layout,
`wmma(B_frag, A_frag)` computes `Cᵀ` at no instruction cost. That is an
unexploited alternative to the entire B-gather problem in §7: rather than
transposing B on the way in, swap the operands and transpose in the epilogue,
where the accumulator is already in registers and the store address math exists.
Not yet measured against §8's load-transpose.

Both operands want **K contiguous per lane**. A row of A in a row-major
`A[M][K]` already is; a column of B in a row-major `B[K][N]` is not — it is
strided by `N`, which is why `materializeFragmentPack` scalarizes the B
fragment into 16 guarded loads while A takes a single `vector.load`.

That asymmetry is the standing optimization target for the ROCm GEMM: the
remedies are a B staged K-major (once, amortized across launches — the natural
form for inference weights), or RDNA4's `GLOBAL_LOAD_TR_B128` load-transpose
(§11.6.2), which reads a 16×16 block of 16-bit data from the global aperture
directly into the transposed fragment layout and requires `EXEC` all ones.
Staging B transposed through LDS also removes the stride, and *was* measured:
at the same panel it is 0.23–0.83× the register body on gfx1201 and up to
1.26× on gfx1151, yet on neither chip does it beat the best register
configuration at any shape (see the typed-route gap packets). The barriers
cost more than the stride does wherever the register body is already
competent.

## 4. Integer shapes — nibble order for the packed int4 operands

The layout is unified across types; only the packing differs.

| Instruction | K | VGPRs per lane (A/B) | Packing |
|---|---:|---:|---|
| `v_wmma_i32_16x16x16_iu8` | 16 | 2 | 4 int8 per VGPR, byte `i` = element `i` |
| `v_wmma_i32_16x16x16_iu4` | 16 | 1 | 8 int4 per VGPR, nibble `i` = element `i` |
| `v_wmma_i32_16x16x32_iu4` | 32 | 2 | 8 int4 per VGPR, **element `i` → nibble `i % 8` of word `i / 8`** |

The double-K int4 shape packs K=32 nibbles into the same two-VGPR footprint as
`iu8` at K=16, and which raw nibble position corresponds to which logical `k`
is not derivable from the intrinsic signature. It is: **logical element `i`
(ascending `k`) occupies nibble `i % 8` — bits `4*(i%8) .. 4*(i%8)+3` — of word
`i / 8`, low nibble first.** Signedness rides the instruction's `NEG` field:
`NEG[0]` for A, `NEG[1]` for B, `0 = unsigned, 1 = signed`; the destination is
always signed, and Tessera pins `clamp` false (wrap), matching a numpy int32
accumulate. The builtin's signature carries both explicitly —
`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(int neg_a, int a, int neg_b,
int b, v8i acc, int clamp)` — so neither rides a default.

**In Tessera**: `materializeFragmentPack` compacts nibbles with exactly that
rule (`shift = 4 * (i % 8)`, word `i / 8`), and int4 values ride int8
containers at the ABI boundary, one logical value per byte, range `[-8, 7]`.

*Evidence*: the typed int4 and int8 matmul device rows are **exact** against
the int32 numpy product on both chips at 16³, 17×19×23 and 65×48×37 —
`test_gfx1201_scheduled_matmul_package_executes_integer_storage` and its
gfx1151 twin. An off-by-one nibble order would not be exact anywhere.

## 5. What each form is worth, and the one that does not exist

A fragment layout tells you how to be correct. This table tells you when to
stop: it is the denominator every ROCm GEMM number on gfx1201 should be quoted
against, and each row is the peak of an instruction `wmma_dtype_forms` already
enumerates.

| Storage | Instruction family | Dense | 2:4 structured |
|---|---|---:|---:|
| fp16 | `V_WMMA_F32_16X16X16_F16` | 191 TFLOP/s | 383 TFLOP/s |
| bf16 | `V_WMMA_F32_16X16X16_BF16` | 191 TFLOP/s | 383 TFLOP/s |
| fp8 e4m3 / e5m2 | `V_WMMA_F32_16X16X16_{FP8,BF8}_{FP8,BF8}` | 383 TFLOP/s | 766 TFLOP/s |
| int8 | `V_WMMA_I32_16X16X16_IU8` | 383 TOP/s | 766 TOP/s |
| int4 | `V_WMMA_I32_16X16X{16,32}_IU4` | 766 TOP/s | 1531 TOP/s |
| fp4 | **no form exists** | — | — |

Two things follow that a relative speedup hides.

**The typed f16 GEMM is about halfway.** Its best measured gfx1201 row is
90.3 TFLOP/s (4096³, 4×4 panel, K unroll 2 — `benchmarks/baselines/typed_route_gap_20260918/gfx1201.json`),
which is **47% of the 191 dense fp16 peak**. The K unroll that got it there was
worth 1.4–1.7×; the remaining 2× is still on the table, and the next lever is
not another macro tile.

**Low precision is a bigger lever than scheduling, and int4 is the biggest.**
Each step down the table doubles the ceiling: int4 dense is 4× fp16, and 2:4
structured int4 is 8×. Tessera already emits `V_SWMMAC_*` for the 2:4 stack on
gfx1201, so the packed-int4 *input* route on the typed path composes with work
that exists rather than starting a new lane.

**gfx1201 has no FP4 matrix instruction, and Tessera refuses rather than
pretending.** There is no `v_wmma_*_fp4` in any form, dense or sparse. The
common workaround in other stacks is to dequantize an FP4-stored weight to fp16
before the MMA, which silently returns the *fp16* ceiling — an FP4 request that
reports FP4 throughput while running at 191. `_ROCM_DTYPES[GFX_1201]` therefore
omits `fp4_e2m1` and `wmma_dtype_forms` enumerates no FP4 row, so the storage
dtype is a Decision #21a semantic key that fails closed on this arch. If an FP4
lane is ever wanted here it must be a *declared* dequantize-and-fp16-MMA route
that says so in its provenance, never an FP4 claim. (CDNA 4 / gfx950 is the
arch that does have native FP4/FP6, which is why that row carries them and this
one does not — evidence never transfers.)

*Source*: AMD's published RDNA4 WMMA throughput for the Radeon AI PRO R9700
(gfx1201), cross-checked against the instruction set this repo extracts into
`docs/reference/isa/rdna/rdna4/`. The Tessera-side table is
`rocm_target.wmma_dtype_forms`, which lists exactly these families and no
others.

## 6. Load width is the reason low precision wants a deeper K unroll

A fragment load is 8 elements per lane whatever the storage, so the *bits* it
moves shrink as the storage does. AMD's RDNA4 WMMA guide states it directly:
fp16 saturates the 128-bit interface (8 x 16 bits), fp8 and int8 use only
64-bit loads (8 x 8 bits), and int4 drops to 32-bit loads (8 x 4 bits).

That is the mechanism behind a rule Tessera arrived at by measurement alone.
The typed body's K unroll is worth the most exactly where the storage is
narrowest — f16 wants 2, fp8 wants 4 from 2048 up, and int4 on gfx1151 wants 4
everywhere — because issuing more slabs is how a narrow operand keeps the
128-bit path busy. Two ways to read that matter:

* **Deeper unroll is a workaround, not the instrument.** It reaches the same
  bandwidth by issuing *more* loads. AMD's technique instead fuses two WMMAs
  over an extended K so one load fetches 16 elements and fills 128 bits, and
  notes the result is bit-identical because matrix multiplication is
  associative — a reordering of the FMAs, not a different program.
* **For int4 the hardware already has it.** `V_WMMA_I32_16X16X32_IU4` is the
  native double-K form, and `V_SWMMAC_I32_16X16X64_IU4` its sparse twin. Both
  are in the table above. **Neither is reachable from the typed route today**:
  `materializeMma` in `TileToROCM.cpp` pins `kBlocks = 1`, so every emitted
  fragment is the K=16 shape. That is a Decision #29 gap — the registry
  declares a form nothing can select — and it is the principled version of
  int4's measured k=4 win.

## 7. Transposing B: what RDNA4 does and does not have

RDNA4 has **no shared-memory transpose load** (`ds_read_tr` is gfx950/CDNA4)
and no in-register transpose instruction. It does have **`global_load_tr`**,
wrapped upstream as `amdgpu.global_transpose_load` on gfx1200+.

**Its shapes are exactly our B fragment's, which is the reason to use it.**
The ISA defines two forms, each loading a 16x16 matrix and transposing between
row- and column-major on the way into the VGPRs:

| Instruction | Element | wave32 destination | bits per lane |
|---|---|---:|---:|
| `GLOBAL_LOAD_TR_B128` | 16-bit | 4 consecutive VGPRs | 128 |
| `GLOBAL_LOAD_TR_B64` | 8-bit | 2 consecutive VGPRs | 64 |

Our B fragment holds `16 * K / 32` elements per lane, which at K=16 is 8:

| storage | elements/lane | bits/lane | instruction |
|---|---:|---:|---|
| f16, bf16 | 8 | **128** | `GLOBAL_LOAD_TR_B128` |
| fp8 e4m3/e5m2, int8 | 8 | **64** | `GLOBAL_LOAD_TR_B64` |
| int4 (K=16) | 8 | 32 | none — `tr4` is gfx1250+ |
| int4 (K=32) | 16 | 64 | none — TR_B64 transposes 8-bit elements, wrong granularity |

So one transpose load replaces the whole strided gather for four of the six
storages, exactly, with no leftover. The ISA's selection table is by *memory*
order: a column-major read with a row-major VGPR layout takes the TR form,
and the note that the table reverses when the VGPR layout is column-major is
what makes it apply here — B is stored row-major `[K][N]` and each lane wants
a column, which is the reversed reading.

AMD's guide gives a second remedy that needs no special load at all: build an
identity matrix in the B fragment, put the source in A, and issue one WMMA —
the instruction's own layout does the transpose in registers. It is the RDNA4
answer to CUDA's `ldmatrix.trans`, and it is reported in production use for
flash attention.

Both are candidates for the standing asymmetry in §3, where A takes a single
`vector.load` and B scalarizes into 16 guarded loads. Neither is implemented
here yet.

## 8. `GLOBAL_LOAD_TR_B128`: the measured lane mapping

The instruction's shapes match our B fragment exactly (§7), so the remaining
question is the *addressing*: which address each lane supplies, and which
elements it gets back. That is not in the ISA text, and guessing it is the
silently-wrong-tiles failure this page exists to prevent. Measured on gfx1201
(2026-09-19) with `B[r][c] = r*16 + c`, every element distinct, each lane
dumping all eight values it received.

With lane `L` supplying the address of element `L*8` — so lane `L` reads row
`L/2`, columns `(L%2)*8 .. +7` — each lane receives:

```
B[(L / 8) * 4 + j / 2][(L % 8) + 8 * (j % 2)]        // j = 0..7
```

Spot checks from the run: lane 0 gets columns {0, 8} of rows 0-3; lane 15 gets
columns {7, 15} of rows 4-7; lane 31 gets columns {7, 15} of rows 12-15.

That addressing was arbitrary, and what it returns is not the fragment. But it
is enough to recover the permutation, which is the part the ISA does not
state. Writing lane `L`'s eight contiguous reads as `R(L)[0..7]`, the measured
result is exactly

```
received(L, j) = R(8 * (L / 8) + j)[L % 8]
```

an **8x8 transpose inside each group of 8 lanes**: the group collectively
reads eight runs of eight elements and transposes that tile. Every row of the
probe follows from it, including the ones that look least like a transpose.

### The address each lane must supply

§3 says the fragment wants `b(L)[h] = B_logical[L % 16][8 * (L / 16) + h]`,
one lane holding one `n` and eight consecutive `k`. With B stored row-major
`[K][N]` so that `B_logical[n][k] = mem[k * ldb + n]`, substituting the
permutation above and solving for the address gives

```
A(L) = (8 * (L / 16) + (L % 8)) * ldb + ((L / 8) % 2) * 8
```

*Evidence*: on gfx1201 with `B[k][n] = k * 16 + n` — asymmetric, every element
distinct — this reproduces the fragment on **256/256 elements**. The
arbitrary `A(L) = 8L` addressing does not, which is the control.

In the materializer the wave lane is `lane + 2*kBase` (kBase is 0 or 8), under
which `A` collapses to the form the code emits:

```
A = (kBase + lane % 8) * ldb + (lane / 8) * 8      // plus the tile origin
```

### What it is worth, and the one shape where it is not

Shipped for f16 and bf16. TFLOP/s at the 4x4 panel with K unroll 2, against
the scalar gather it replaces:

| shape | transpose load | gather | |
|---|---:|---:|---|
| 1024³ | **65.9** | 54.2 | +21.6% |
| 1536³ | **79.2** | 74.3 | +6.6% |
| 2048³ | 75.4 | **80.5** | **-6.3%** |
| 2560³ | **94.3** | 89.5 | +5.4% |
| 3072³ | **92.8** | 89.4 | +3.9% |
| 4096³ | **93.9** | 89.4 | +5.0% |

Five of six shapes win, and 2048³ is the sole reversal. It reproduces at five
runs (75.8 against 81.3), so it is not sampling noise. A leading dimension of
exactly 2048 is the obvious suspect -- that stride is where channel or
partition aliasing usually shows -- but that is a **hypothesis, not a
measurement**: no counters exist on either WSL2 ROCm box to confirm it. The
instruction stays on for every shape rather than being special-cased around
one anomaly from one point.

**8-bit storages are excluded and that is a measured decision.** `TR_B64` is a
different permutation and the derivation above does not carry to it. Enabling
it on the strength of the matching width produced wrong results on device for
every 8-bit storage -- 16 failing rows across fp8 and both integer widths,
while f16 and bf16 were untouched. It stays out until its own mapping is
measured the same way.

Everything else is in place: the `amdgpu` dialect is registered in both
drivers, `amdgpu.global_transpose_load` parses over a memref,
`convert-gpu-to-rocdl` lowers it to `rocdl.global.load.tr.b128` with no
additional pass, and the kernel builds and runs on gfx1201.

## 9. Where the 4x4 panel's registers actually go

The panel spills against a ceiling the ISA fixes at 256 (§5), so the only
lever is needing fewer live values. That was recorded as unscoped; this is the
measurement. Compile-only on gfx1201, f16, `vgpr_count` and `vgpr_spill_count`
from the kernel metadata:

| panel | K unroll | allocated | spilled | accumulators | fragments | everything else |
|---|---:|---:|---:|---:|---:|---:|
| 1x1 | 1 | 59 | 0 | 8 | 8 | ~43 |
| 2x4 | 1 | 198 | 0 | 64 | 24 | ~110 |
| 2x4 | 2 | 198 | 0 | 64 | 48 | ~86 |
| 4x4 | 1 | 256 | **133** | 128 | 32 | **~229** |
| 4x4 | 2 | 256 | **133** | 128 | 64 | **~197** |

Accumulators are `mt * nt * 8` VGPRs; fragments are `(mt + nt) * 4` per K slab
in flight. Three things fall out, and they redirect the work.

**About half the demand is neither.** At the 4x4 panel roughly 200 VGPRs are
something other than the accumulator tile and the operands feeding it — more
than the accumulators themselves. The tile is not too big for its data; the
surrounding state is too big.

**That overhead scales with the panel, not with K.** It roughly doubles from
the 2x4 panel to the 4x4 (110 to 229) while the accumulators also double, so
it tracks `mt * nt`: per-tile addressing and index state held live across the
loop, sixteen tiles' worth at 4x4.

**The K unroll is not the cause.** Spill is identical at k=1 and k=2 (133
either way), so issuing a second slab costs fragments and nothing structural.
That also means the unroll and the spill are independent problems, and fixing
one will not move the other.

So the target is the ~200 VGPRs of per-tile addressing, and halving it would
fit the 4x4 panel with no spill at all. Candidates, in the order they look
worth trying: recompute tile origins from the loop index instead of holding
sixteen of them; share the row and column origin arithmetic across a panel row
or column rather than per tile; and sink the fragment address computation into
the loop body so it does not stay live across the MMAs.

*Caveat on the numbers*: `vgpr_count` is the allocation and `vgpr_spill_count`
counts spill slots, so "allocated + spilled" is an estimate of demand rather
than an exact live-range count. The ratios are what the argument rests on, and
they are stable across the panels above.

## 10. Three gfx1201 constraints that shape every schedule here

Recorded 2026-09-19 (owner-supplied, and they explain results elsewhere in this
file rather than merely adding to them).

**No `v_cvt_pk_bf16_f32`.** Packing bf16 is software on gfx1201. The cheapest
form is two adds — the round-to-nearest-even bias on each half — followed by one
`v_perm_b32` to gather the high halves. Any bf16 epilogue that reaches for a
pack instruction is reaching for something the ISA does not have.

**No direct-to-LDS, and no `ds_read_b64_tr_b16`.** Global memory cannot be
staged into LDS without a round trip through VGPRs, and LDS has no
transposing read. **This is the mechanism behind the LDS result in §3**, which
until now was recorded as a bare measurement: staging costs a register round
trip that the register body never pays, and the LDS-side transpose that would
justify the trip does not exist. `GLOBAL_LOAD_TR_B128` is a *global* transpose
for exactly this reason — it is the only transposing load the chip has.

**FP8 conversion IS hardware, unlike bf16 packing.** `__builtin_amdgcn_cvt_pk_fp8_f32`
and `__builtin_amdgcn_cvt_f32_fp8` compile and run on gfx1201 (owner-confirmed
2026-09-19), and both are reachable from MLIR without a builtin escape:
`rocdl.cvt.pk.fp8.f32` packs two f32 into fp8 with a word select, and
`rocdl.cvt.f32.fp8` unpacks with a byte select. Note the asymmetry with the
bf16 note above — bf16 packing costs two adds and a `v_perm_b32`, while the fp8
round trip is one instruction each way. That matters wherever a scale has to be
applied: the block-scaled FP8 path (`ROCM-FP8-BLOCKSCALE-1`) and the MXFP4
W4A8 fold (§10b) both need fp8 to f32 and back, and neither has to pay for it
in software. The wider `rocdl.cvt.scalef32.pk8.*` family exists in the dialect
but is a CDNA4/gfx950 form — do not reach for it here without checking §5.

**LLVM single-buffers LDS and drains before every WMMA unless told otherwise.**
The fix is `__builtin_amdgcn_sched_group_barrier`, and the *granularity* of the
groups matters far more than their contents: coarser is better, up to the point
where the pattern asks for more outstanding loads than the hardware can hold.

That last one is not a tuning note. **This backend emits no scheduling
intrinsic at all**, so every gfx1201 measurement recorded in this file and in
the ROCm queue was taken under the default schedule — drained, single-buffered.
See §11 for which results that qualifies.

## 10a. WGP, not CU, is the occupancy denominator

Each **WGP maps to two Compute Units**, which share execution resources and
execute the scheduled waves; GL0 and GL1 implement the per-SE vector cache
hierarchy feeding them. A workgroup dispatches to a WGP, so the count that
decides whether a launch fills the machine is the WGP count, not the CU count.

Measured on Tajasarus: `rocminfo` reports **64 Compute Units** for gfx1201
(RX 9070 XT) — i.e. **32 WGPs**. Reading that 64 as the occupancy denominator
is a 2x error in exactly the direction that hides an under-filled launch, and
`ROCmTargetProfile` currently carries **neither** number.

Worked consequence: the MoE router gate (M<=16, K=2048, N=256) at our 16x16
macro tile is 16 output tiles, so **16 of 32 WGPs** have work. The weight read
dominates A by 16x (1.05 MB against 65.5 KB), so the MMA unit is not the scarce
resource there and the fix is occupancy, not a bigger tile.

## 10b. MXFP4 on a chip with no FP4: the W4A8 fold, and what it says about our results

RDNA4 has **no FP4 WMMA form** (§5), and the ecosystem's answer on gfx1201 is
not to give up but to **fold MXFP4 into the fp8 WMMA**. Read from an
independent hand-written gfx1201 kernel (vllm-radiance `radiance_mxfp4_fp8.hip`,
2026-09-19) and its tuned configs. Four things there bear directly on results
recorded in this file.

**The fold is exact, which is why it is not a compromise.** E2M1's sixteen
values are all exactly representable in e4m3, so the weight upconvert is a
lossless table lookup, and the MX block scale is E8M0 — a power of two — so
applying it to the fp32 accumulator is exact. The block exponent is folded into
the *weight* through a per-binade magnitude table, which removes the per-32-block
rescale from the inner loop and leaves one per-row factor for the epilogue. The
result is W4A8 rather than the checkpoint's declared W4A4: strictly *more*
precise than calibration, but no longer bit-identical to the emulated path, so
it is opt-in rather than default. That is a numeric-policy decision of exactly
the kind Decision #15a says belongs in the contract, not in a kernel flag.

**gfx12's fp8 WMMA honours e4m3 subnormals rather than flushing them**, verified
on hardware there. The fold depends on it: stopping at the smallest e4m3 normal
(2⁻⁶) is exact only to d ≤ 5, and a real checkpoint reached d = 10.

**Our contiguous-eight fragment convention is independently corroborated — and
so is §3's warning about how it was checked.** That kernel records its layout
as `A: lane l holds A[l%16][(l/16)*8+j]`, `B: B[(l/16)*8+j][l%16]`,
`C: C[(l/16)*8+j][l%16]` — identical to ours, "confirmed empirically (0/256
elements wrong on a 16x16x16 tile)". Note *what that check can see*: comparing
the computed C cannot distinguish our relabeling from the machine's runs-of-four
mapping, because the permutation cancels across both operands (§3). Two
independent implementations agreeing on contiguous-eight is evidence the
convention is sound; it is not evidence about which mapping the hardware uses.

**The LDS result in §3 and §11 is contradicted, not merely qualified.** That
kernel reports reading fragments straight from global at **9.5 TFLOP/s**, and
states that staging through LDS "is what makes this fast at all" — against 325.2
TFLOP/s register-resident for the fp8 WMMA on the same card. The stated cause is
the one §3 names from the other side: lanes sixteen rows apart touch sixteen
cache lines per operand. Their LDS rows are **padded 8 bytes**, without which
sixteen lanes reading rows 64 B apart collide eight ways on the 32 × 4 B banks —
and `rocm_tiling` already models exactly that as `bank_padding_required`, also
unwired (Decision #29a). Our "the LDS body loses at every shape" was measured
without padding-aware staging and under the drained default schedule (§10).
**Treat it as unsafe to cite until re-measured**; `ROCM-SCHED-GROUP-1` and this
are the same experiment.

**Achieved throughput corroborates §5's ceilings.** 325.2 TFLOP/s fp8 against
our 383 ceiling (85%) and 160.2 f16 against 191 (84%), register-resident on the
same part.

## 10c. ROCM-SCHED-GROUP-1, measured: the barriers work and they lose

The lever is now implemented (`sched-groups=N` emits N alternating
`rocdl.sched.group.barrier vmem_read` / `mfma_wmma` groups describing the
panel) and measured on Tajasarus, f16, 1024³, repeated-median over 7 trials of
50 launches. **Every setting is slower than LLVM's default.**

| sched_groups | register body | LDS body (2x2 waves) | VGPR spill |
|---|---|---|---|
| 0 (default) | **70.5 TFLOP/s** | **7.8 TFLOP/s** | 133 |
| 1 | 53.8 (0.76x) | 7.8 (1.00x) | 133 |
| 2 | 46.6 (0.66x) | 5.5 (0.70x) | **123** |
| 4 | 46.0 (0.65x) | 5.8 (0.74x) | 127 |
| 8 | 46.0 (0.65x) | 5.5 (0.70x) | **163** |

All arms agree numerically at 1.95e-06, so this is a schedule difference and
nothing else. **The barriers demonstrably took effect** — between `sg=0`, `2`
and `8` the emitted instruction order differs, the instruction count moves
(7017 / 6897 / 6984) and `s_wait*` falls monotonically (1360 / 1283 / 1261).
The scheduler did what it was told; being told was the problem.

**Three things this settles, and one it does not.**

*The spill is not a scheduling artifact.* §11 put §9's per-tile-addressing
diagnosis in doubt because outstanding-load state also scales with `mt*nt`.
It does move the spill — 133 to 123 at `sg=2` — but only by 7%, and that arm is
34% *slower*. Register pressure at the 4x4 panel is not what the scheduler is
holding. **§9's diagnosis stands; withdraw §11's doubt on that row.**

*The other three qualified rows are not overturned.* For the register body the
default schedule beats all four described patterns, so the LDS/K-unroll/double-K
comparisons were not measured against a handicapped baseline.

*The knob stays at 0.* It is retained rather than deleted because a measured
negative is worth more than the absence of one — but read the table before
re-running this experiment, not after.

*What it does not settle:* one pattern was tested. `vmem_read`/`mfma_wmma`
alternation at panel granularity is not the space — `ds_read` groups,
`sched.barrier`, `iglp.opt`, and placement outside the K loop are all untried.
This is evidence against *this* description, and weak evidence about the lever.

**The redirect is the real result.** The LDS body runs at 7.8 TFLOP/s against
the register body's 70.5 — **9x slower**, which is close to what an 8-way LDS
bank conflict costs. §10b records that the vllm-radiance gfx1201 kernel pads
its LDS rows 8 bytes precisely because sixteen lanes reading rows 64 B apart
collide eight ways on the 32 x 4 B banks, and that staging is what makes *their*
kernel fast. Ours pads nothing. `rocm_tiling` already models this as
`bank_padding_required` and does not wire it (Decision #29a). **Bank padding,
not scheduling, is the candidate that fits the number** — and it is what
ROCM-LDS-BANKPAD-1 should test before the LDS body is judged again.

## 10d. NVFP4 on gfx1201: one lossy step, and it is the scale format

An NVFP4 checkpoint can reach the fp8 WMMA on gfx1201, and the whole chain has
**exactly one lossy step**. Recorded 2026-09-19 from the vLLM MXFP4 work on
R9700 (`GGZ14/vllm-mxfp4`, `ggz14/radiance-vllm-mxfp4`).

```
NVFP4      e2m1 elements + e4m3 scale per 16 + fp32 scale per tensor
   |  LOSSY  requantize: e4m3/16 scales -> e8m0/32 scales
   v         (elements are already e2m1; only the scale contract changes)
MXFP4      e2m1 elements + e8m0 scale per 32
   |  EXACT  lossless e2m1->e4m3 lookup, power-of-two block exponent folded
   v         into the weight (section 10b)
fp8 e4m3 -> v_wmma_f32_16x16x16_fp8_fp8
```

**The loss is double rounding, not the format** — which is the part worth
keeping, because it tells you where the fix is. Measured against the bf16
original as relative RMS:

| path | relRMS | SQNR |
|---|---|---|
| NVFP4 (as shipped) | 0.113 | 19 dB |
| bf16 → MXFP4 **direct** | 0.112 | ~19 dB |
| NVFP4 → MXFP4 (requantized) | 0.158 | 16 dB |

MXFP4 is as faithful as NVFP4 when quantized *once* from bf16; going through
NVFP4 first costs ~3 dB. So where the bf16 original is available, quantizing
directly to MXFP4 is strictly better than ingesting NVFP4 and requantizing, and
that is a `numeric_policy` preference rather than a kernel detail. Reported
task accuracy is unaffected at this size (GSM8K 97.40%), on 2x R9700 with
decode 22.2–24.7 ms/step and prefill 3900–5072 tok/s.

**Two traps in the conversion.** Block-exponent selection is not truncation —
it picks by squared error between the no-clip rule and one binade finer. And a
merged linear (`gate_up_proj`) carries *two* global scales that must be honoured
separately rather than collapsed; collapsing them is silent.

**This is not RDNA4 catching up.** AMD's own MI355 path dequantizes NVFP4 to
**BF16** at GEMM time, because CDNA4 has no native NVFP4 execution path either
(`rocm.blogs.amd.com/.../nvfp4-mi355`). The gfx1201 route lands on the fp8 WMMA
instead, whose ceiling is 383 TFLOP/s against bf16's 191 (section 5) — so on
this axis the RDNA4 part has the better landing spot, not a worse one.

**What Tessera already has, and what it does not — corrected 2026-09-19.**
An earlier draft of this section said the block-scale metadata was missing
outright. It is not. `microscaling.ScaleLayout` models exactly this contract —
`block_size` elements along an axis sharing one scale of dtype `e8m0` (MX),
`fp8_e4m3` (**its docstring names NVFP4**) or `fp32`, with `block_size == 0`
meaning a per-tensor scale — and `Tessera_ScaleLayoutAttr` carries it in IR on
four grouped-GEMM ops beside optional `x_scale`/`w_scale` operands, with
consumers on the Apple path, the runtime, the manifest and the capability
tables. `dtype.py` separately names `nvfp4`, `fp4_e2m1` and `mxfp4` as distinct
with an explicit "do not alias".

What is actually missing is narrower: the **plain `tessera.matmul` carries no
scale operands at all**, and the **ROCm typed route never consults the model**.
So the requantization above has a vocabulary to be declared in — it simply has
no path from a Graph matmul to a gfx1201 kernel that reads it. That is
`ROCM-FP8-BLOCKSCALE-1`, and it is an extension of an existing contract rather
than a new one.

## 10e. ROCM-LDS-BANKPAD-1, measured: padding is real and it is not the 9x

The LDS tiles store a 16-element row — A by row, B transposed by column — which
for f16 is 32 B = **8 dwords**, and `gcd(8, 32) = 8`, so sixteen lanes reading
one element per row land on four banks: a **4-way conflict** on every fragment
read. Making the stride an odd dword count fixes that by construction, and one
dword of padding does it at every storage width (8→9 f16, 4→5 fp8/int8, 2→3
int4), which is why the knob counts dwords and the element count follows
(`32/bitwidth`).

Measured on Tajasarus, 1024³ f16, 2x2 waves, all arms exact at 1.95e-06:

| | TFLOP/s | vs pad=0 |
|---|---|---|
| pad=0 | 7.8 / 7.9 | — |
| **pad=1** | **8.8 / 8.7** | **1.12x / 1.10x** |
| pad=2 | 8.1 | 1.04x |
| pad=3 | 8.9 | 1.13x |
| register body | **70.7** | **8.0x the padded LDS** |

**The padding is a real win and the hypothesis behind it was still wrong.** It
takes the gap from 9.1x to 8.0x. A prediction recorded before the run said that
if padding barely moved the number the conflict theory was wrong; it barely
moved, so it is wrong. Bank conflicts are a ~10% tax here, not the story.

Two things the ISA census showed, once counted correctly. (**gfx12 renamed the
LDS mnemonics**: `ds_read_*`/`ds_write_*` became `ds_load_*`/`ds_store_*` in
RDNA3, and a census regex carrying the old names reports *zero* LDS traffic in a
kernel full of it — which reads exactly like a finding and is not one.)

*The staging copy is scalar in both directions.* The LDS body emits
`global_load_d16_b16` in and `ds_store_b16` / `ds_store_b16_d16_hi` out — sixteen
bits per thread per iteration — where the register body gets
`global_load_b128` x12 and `global_load_tr_b128` x12. The copy loop walks one
element per thread. **That is the shape that fits an 8x gap**, and it is
`ROCM-LDS-STAGE-VECTOR-1`.

*Padding wins despite narrowing the read.* At pad=1 the fragment read degrades
from `ds_load_b128` to `ds_load_2addr_b32`, because an 18-element row is not
128-bit aligned. So the +10-12% is **net of a regression** — the conflict cost
more than the headline — and it explains why pad=2 (+4%) trails pad=1 and pad=3
(+10-13%), which the dword-gcd argument alone does not predict. Alignment and
load width interact; the simple model is necessary and not sufficient.

Default is now `pad=1`: free, numerically identical, and it does not touch
production selection because the register body still wins by 8x.

## 10f. rocWMMA as an independent check on these tables

Read from the raw headers 2026-09-19 (`rocwmma/internal/wmma_impl.hpp`, 3310
lines, plus `types.hpp` / `vector.hpp` / `vector_util.hpp`). rocWMMA is AMD's
own WMMA wrapper, so where it agrees with a table of ours that nobody had
checked against a vendor source, that is real corroboration.

**Architecture grouping, and where we deliberately differ.** rocWMMA has
`enable_gfx11_t` = gfx1100–1103 / 1150–1153 and `enable_gfx12_t` = **gfx1200,
gfx1201 and gfx1250 together**. `rocm_fragment.py` instead splits `_RDNA4`
(gfx120x) from `_GFX125X`. Keep the split: the two share the *builtin family*
and nothing else — gfx120x is K=16 with a `_w32_gfx12` suffix, gfx1250 is
K=32/64/128 with neither `_w32` nor `_gfx12` — and a fragment-ABI table keyed on
shape has to separate them. Do not "correct" our sets to match rocWMMA's.

**Our gfx125x shape table is corroborated six rows out of seven.** Every one of
fp32 K=4, fp16 K=32, bf16 K=32, fp8 K=64 *and* K=128, int8 K=64 appears as a
rocWMMA builtin. The seventh — `fp4_e2m1` at (16,16,128) and (32,16,128) — has
**no rocWMMA builtin at all**, and neither does any other fp4 form. That is not
a refutation (rocWMMA need not wrap everything) but it is the one row of that
table with no independent support, and it is load-bearing for any MXFP4 story
on gfx125x. Treat it as unverified until an ISA source confirms it.

**The mixed FP8 pairs are the vendor position, not ours alone.** rocWMMA
exposes `f32_16x16x16_fp8_bf8_w32_gfx12` and its mirror on gfx12, and all four
pairings at K=64/128 on gfx1250 (including f16-output variants). So
`ROCM-MIXED-FP8-1` lands on the same capability rocWMMA already wraps. **This
is recorded because a summary of that header omitted the mixed builtins and
nearly produced the opposite claim** — the third instance this session of
reading silence as absence, and the second caught before it was written down.
The raw file is the source; a summary of it is not.

**int4 is genuinely ours.** There are **zero** `iu4` references in the entire
header — not gfx11, not gfx12, not gfx1250. We emit
`v_wmma_i32_16x16x16_iu4` and the double-K `v_wmma_i32_16x16x32_iu4` on
gfx1201, both proven exact on device (§4, and the double-K row in
`test_rocm_gfx1201_scheduled.py`). The hardware has it and rocWMMA does not
wrap it, so this is a vendor-library coverage gap rather than a hardware
question — and it means there is no rocWMMA source to crib for int4.

**The fragment layout is not in the headers, which is the point.** `wmma_impl`
names register aliases without saying how K distributes across the wave;
`types.hpp` is a 127-line scalar typedef facade and `vector.hpp` a HIP vector
wrapper. Neither documents the per-lane mapping. That is why §3's contract had
to be measured, why the independent gfx1201 kernel in §10b measured it too, and
why both converged on contiguous-eight without either being the machine's own
mapping — the permutation cancels, so nothing forces the issue.

## 10g. AMD's matrix instruction calculator: what it settles and what it cannot

`ROCm/amd_matrix_instruction_calculator` prints the authoritative per-lane
register layout for VOP3P matrix instructions. Run 2026-09-19 against gfx1201
(the tool reports it as RDNA4). It replaces several things here that were
measured or taken on report with a vendor source.

**Settled — the A/B mapping.** `-R -A -w 32` prints the table in §3, and it
confirms the 16-bit runs-of-four mapping exactly, element for element. It also
shows the 8-bit and 4-bit storages using contiguous-eight, which is our own
convention — so §3's permutation argument is needed for f16/bf16 and nowhere
else.

**Settled — the accumulator.** `-R -D -w 32` prints `D[7][n] = v7{n}` and
`D[8][n] = v0{n+16}`, i.e. VGPR `j` at lane `L` holds `D[(L/16)*8 + j][L%16]`.
That is §2's recorded layout, now vendor-confirmed rather than derived.

**Settled — the int4 nibble order (§4).** `A[0][k]` for k = 0..7 is
`v0{0}.[3:0]`, `[7:4]`, `[11:8]` … `[31:28]`: the low nibble is the lowest k and
k ascends with nibble position, eight nibbles per VGPR, with k = 8 starting at
lane 16.

**Settled — what gfx1201 actually has.** `-L` lists
`v_wmma_i32_16x16x16_iu4` **and** `v_wmma_i32_16x16x32_iu4`, all four fp8
pairings (`fp8_fp8`, `fp8_bf8`, `bf8_fp8`, `bf8_bf8`), and the sparse
`v_swmmac_i32_16x16x32_iu4` / `v_swmmac_i32_16x16x64_iu4`. So the int4 forms we
emit and proved exact are vendor-listed hardware, and §10f's reading — that
rocWMMA's omission of int4 is a wrapper gap rather than a hardware fact — is
confirmed. **No fp4 form appears on RDNA4**, again.

**Settled — gfx11 replicates where gfx12 splits.** `TileToROCM.cpp` carries the
comment "GFX11 replicates operands (kBase=0); GFX12's upper half-wave must
advance by eight K elements", which was an assertion until now. On gfx1151 the
tool reports **8 GPRs for A** and prints lane 0 holding *all sixteen* k values
across v0–v7; on gfx1201 it reports **4 GPRs for A** with lane 0 holding eight
and lane 16 the other eight. The register count is the tell, and the comment is
correct.

**Where it lives on the fleet.** Absolute paths, because the two boxes have
different users and a non-interactive `ssh` expands `~` to whichever account it
logged in as:

| host | path |
|---|---|
| Tajasarus (gfx1201) | `/home/angstorms/programming/amd_matrix_instruction_calculator` |
| Princess-Luna (gfx1151) | `/home/gstoner/programming/amd_matrix_instruction_calculator` |

It needs `tabulate`, installed into each box's tessera venv.

Running `-L` on each arch corroborates the RDNA3.5-vs-RDNA4 split this repo
asserts in several places. gfx1151 lists **exactly six** matrix instructions —
`v_wmma_{f32,f16}_16x16x16_f16`, `v_wmma_{f32,bf16}_16x16x16_bf16`,
`v_wmma_i32_16x16x16_{iu8,iu4}` — with **no FP8/BF8 form and no SWMMAC at all**,
while gfx1201 adds the four fp8 pairings, the double-K int4 and the sparse
family. "FP8 and sparse are RDNA4-only" is therefore a vendor-checkable fact,
not an inference from our own tables.
It is pure Python and needs no GPU, so a layout question does not need device
time — but note the two boxes answer for different chips by argument, not by
which box you are on (`-a gfx1151` works on Tajasarus and vice versa).

**NOT settled, and the silence must not be read as refutation.** The tool
supports **CDNA1, CDNA2, CDNA3, RDNA3 and RDNA4 only** — `gfx950` and `gfx1250`
are both rejected outright. It therefore says nothing about the `fp4_e2m1`
gfx125x row flagged UNCORROBORATED in §10f, which stays open pending a CDNA5
ISA source. Absence from a tool that does not model the chip is not evidence
about the chip.

## 10h. What AMD's own shipping gfx1201 FP8 GEMM actually does

hipBLASLt installs 144 Tensile code objects for gfx1201 at
`$ROCM_PATH/lib/hipblaslt/library/gfx1201`. They are **CCOB compressed offload
bundles**, not bare ELFs — `clang-offload-bundler --type=o --unbundle
--targets=hipv4-amdgcn-amd-amdhsa--gfx1201` extracts the ELF, which then
disassembles normally. Counts below are **static, across every kernel variant
inside one library file** (5.2M lines), not one kernel's mix.

From `TensileLibrary_B8F8_SB8F8_..._Ailk_Bjlk_gfx1201`:

| instruction | count | what it says |
|---|---|---|
| `v_wmma_f32_16x16x16_bf8_fp8` | 23854 | the **mixed pairs**, in shipping vendor code |
| `v_wmma_f32_16x16x16_fp8_bf8` | 13596 | |
| `global_load_tr_b64` | 14780 | **the 8-bit transpose load we excluded** |
| `buffer_load_b128` | — | the other operand, wide, untransposed |
| `ds_load_b32` / `ds_load_b128` | 55408 / 21656 | LDS staging is central, and the reads are wide |
| `ds_store_b32` | 6708 | stores are far fewer than loads |
| `s_barrier_signal`/`_wait` | 7224 each | |

**Three of our open items move on this.**

*`ROCM-GLOBAL-LOAD-TR-1`'s owed 8-bit extension is viable, and no longer needs a
blind measurement.* §8 records that enabling `TR_B128` for 8-bit storages on the
matching per-lane width alone produced wrong results on every one of them, so
the item was parked pending a measured `TR_B64` mapping. **AMD uses
`global_load_tr_b64` for the 8-bit B operand**, 14780 times in this one library,
alongside `buffer_load_b128` for the untransposed side — the same A/B asymmetry
our register body has. The addressing is a scalar 64-bit base (`s[52:53]`) with
per-lane VGPR offsets, the same shape as the address derived in §8. So the
mapping can be *read* out of this disassembly rather than measured from scratch.

*`ROCM-LDS-STAGE-VECTOR-1` is confirmed from the other side.* AMD's kernel is
LDS-heavy — tens of thousands of `ds_load_b32` and `ds_load_b128`, 7224 barrier
pairs — and its LDS reads are **wide**. Our typed LDS body stages with
`ds_store_b16` and `global_load_d16_b16`, sixteen bits per thread per iteration
(§10e). The vendor kernel is not avoiding LDS; it is staging it properly. That
is the third independent line pointing at the copy loop rather than at LDS
staging as a strategy.

*`ROCM-MIXED-FP8-1` is vendor-confirmed in shipping binaries*, not merely in the
rocWMMA headers (§10f).

**On the block-scale gap this file previously overstated — corrected
2026-09-19.** Of those 144 gfx1201 libraries, **zero** carry MX or block-scale
in their type tags (they are B8/F8/H/S/D/BB with scalar and vector scales:
`SAB`, `SAV`, `SCD`), and rocMLIR's scaled path is MFMA-oriented. From those two
I wrote that block-scaled low precision on the RDNA4 WMMA is "unserved across
AMD's own stack". **That is wrong, and it conflated two different Triton
paths.** Split them:

| path | on gfx1201 | served? |
|---|---|---|
| **FP8 W8A8, block-scaled** | Triton emits the **native fp8 WMMA** — AITER's `gemm_a8w8_blockscale` at `block_shape=[128,128]`, which is what the 20 tuned configs in `ROCM-FP8-BLOCKSCALE-1` actually are | **yes** |
| **MXFP4 / e2m1** (`tl.dot_scaled`) | upconverts e2m1 to bf16 and uses the 16-bit WMMA | **no** |

So the FP8 blockscale case *is* served on this chip, by Triton rather than by a
library: reported 25% faster decode on Qwen3-0.6B and 63% on Qwen3-30B on an
R9700, with AITER's C++/ASM kernels disabled because they do not run on RDNA4,
**M ≥ 16 required**, 11 shapes tuned, and no upstream merge. The `M ≥ 16`
constraint and the M-bucketed config keys are the same skinny-M axis §10b and
the split-K item keep running into. Only the **MXFP4 fold** (§10b) is genuinely
unserved, which is exactly why the hand-written kernel exists — and why
`ROCM-MXFP4-W4A8-1` is the item with no prior art to lean on while
`ROCM-FP8-BLOCKSCALE-1` has a working reference to measure against.

**The hardware reason for the split, and it is verified.** AMD's MXFP4/MXFP6
guidance describes native FP4/FP6 on MI355 "through **Matrix Fused Multiply Add
(MFMA) scale instructions**" — CDNA4 has scale-*carrying* matrix instructions.
gfx1201 has **none** (§10c's method: zero `scale` hits in the calculator's
instruction list), so on RDNA4 a block scale is necessarily software, whoever
writes it. That is the same constraint recorded for our own schedule.

**And it sharpens the MI355 NVFP4 story in §10d.** MI355 natively supports
**MXFP4** — block 32, E8M0 — while having *no* native **NVFP4** path (block 16,
E4M3) and dequantizing it to BF16. The two are not interchangeable: same e2m1
elements, different scale contract, and only one of them has silicon behind it
on that part.

## 10i. WITHDRAWN: every LDS throughput figure in §10e, and the padding default

**A harness bug inflated every LDS number recorded on 2026-09-19, including the
ones §10e used to pick a default.** The LDS body computes
`gridM = ceil(M / wgM)` where `wgM = wavesM * macroTileM` — a **128x128**
workgroup tile at 2x2 waves and a 4x4 panel — and indexes with `bidY`/`bidX`
directly. The measurement harness launched a grid sized for the artifact's
**64x64** macro tile, so it ran **four times too many workgroups**, each
computing a full tile. Results stayed correct (every arm at 1.95e-06, because
the redundant groups recompute the same values), which is exactly why it went
unnoticed: the numbers were wrong and nothing was.

Corrected grid, same kernel, 1024³ f16, two runs:

| pad | vector width | §10e claimed | corrected run 1 | run 2 | 2048³ |
|---|---|---|---|---|---|
| 0 | 8 | 7.9 | 17.7 | 17.7 | 34.8 |
| **1 (the shipped default)** | 2 | **8.8 "best"** | 14.5 | 16.4 | 37.6 |
| 2 | 4 | 8.1 | 16.6 | 16.6 | 37.9 |
| 4 | 8 | — | 18.5 | 17.4 | **38.5** |

**What this withdraws.** §10e's "+10–12% for pad=1" is withdrawn: corrected, the
whole 1024³ spread is 16.4–17.7 and the ordering changed between runs, so the
padding choice is **within noise at that shape**. `lds_pad_dwords=1` was
selected from the broken measurement and is not supported by the corrected one.
2048³ shows a mild monotonic preference for more padding (pad=4, +10.7% over
pad=0), which is the only clean signal in the set.

**What this withdraws about the gap.** The LDS body is **2.2–4.0x** off the
register body (71.4 at 1024³, ~83.8 at 2048³), not the 8–9x recorded in §10e and
cited in `ROCM-LDS-STAGE-VECTOR-1` as "the 8x". Three of the four independent
lines that pointed at the staging copy — the radiance kernel's ungated-global
result, AMD's LDS-heavy Tensile library, CK's `ScalarPerVector=8` — are external
and unaffected; the fourth, our own 8x, was this harness.

**And what was UNMEASURED — now closed by §10j, and worse than this section
assumed.** The vectorised copy's benefit had no valid baseline here. It now
does, and it is **negative**: 0.51-0.87x of the scalar copy. Worse, the
"corrected" table above is itself **mislabeled** — it was produced by a stale
`tessera-opt` that contained no vector ops, so it is the **scalar** pad sweep,
not the vectorised one. Its numbers stand; its column heading did not. See
§10j.

**The lesson is not "check the grid".** It is that a redundant-work bug is
invisible to a correctness check by construction — every arm agreed to
1.95e-06 — so a throughput harness needs its own proof that it computed each
output once. A cheap one: assert `gridM * wgM >= M` and `gridM * wgM < M + wgM`,
which the broken harness would have failed on its first run.

## 10j. The vectorised staging copy is a regression, and it never compiled

**Closes the "UNMEASURED" clause above, in the opposite direction from the one
it was written expecting.** Measured 2026-09-19 on `Tajasarus` (gfx1201, ROCm
10.0), 1024³ f16, LDS body, 2x2 waves, 4x4 panel, behind the corrected grid,
median of 7 trials:

| pad | derived width | **scalar copy** | vectorised copy | ratio |
|---|---|---|---|---|
| 0 | 8 | **17.6** | 13.5 | 0.77x |
| 1 | 2 | **14.4** | 7.4 | **0.51x** |
| 2 | 4 | **14.6** | 11.4 | 0.78x |
| 4 | 8 | **15.7** | 13.7 | 0.87x |

The scalar column reproduces §10i's corrected table (17.7 / 14.5 / 16.6 / 18.5)
to within its noise — which is the tell for the second defect below.

**The ISA says why, and it is not subtle: neither arm emits a wide load.**
Disassembled through `tests/_support/rocm_isa.py`:

| | `global_load_d16_b16` | `global_load_d16_hi_b16` | `global_load_b128` | `s_cbranch_execz` |
|---|---|---|---|---|
| scalar | 18 | 14 | **0** | 128 |
| vectorised | 8 | 8 | **0** | **146** |

`llvm.intr.masked.load` expands on AMDGCN into a per-element branch plus a
narrow load. A runtime-masked load therefore **cannot** become
`global_load_b128` — so the vector width bought no bandwidth and paid 18 extra
branches. That is the whole of the 0.77x.

**The defect was a comment, and it was load-bearing.** From the code it
describes: *"the ragged tail is a masked load rather than a scalar fallback,
which keeps one path."* Keeping one path is exactly what prevents the wide
load. `global_load_b128` needs an **unmasked** `vector.load` on the in-bounds
common path with the ragged tail split off into its own branch — the two-path
shape that sentence rejected on tidiness grounds. Tidiness was the wrong axis.

**A second defect hid the first, and it is the more dangerous one.** Every
"vectorised" figure ever recorded — including §10i's corrected table — came
from a `tessera-opt` that had no vector ops in it. `TESSERA_OPT` on Tajasarus
resolves to `build-assertions`, and the rebuilds went to `build`. The run
succeeded and measured the old kernel.

It surfaced only because the next change added a pass **option**
(`lds-copy-width`), which fails loudly at the option parser. A change to a pass
**body** has no such tripwire: it compiles, runs, and silently measures code
that is no longer in the tree. `rocm_native.warn_if_generator_is_stale` now
compares the resolved binary against the generator sources and says so.

**Two defects, one measurement, opposite signs.** The harness bug inflated the
LDS numbers ~4x; the stale binary meant the arm under test was never running.
Either alone would have produced a plausible-looking table. The reason to
record both is that they compose: the 2026-09-19 vectorisation was
"measured" four times, and all four measurements were of the code it replaced.

**Status.** `lds_copy_width` defaults to `1` (scalar) — the measured-faster
arm. The vector path stays reachable behind the knob as the measured-negative
arm, the same disposition `ROCM-SCHED-GROUP-1` got.

### 10j.1 The split path was built, and it loses too — for a different reason

The unmasked fast path plus a split ragged tail was implemented the same day.
It **does** produce the wide load the masked version could not (`wide=2`, one
per copy loop) — and it is still slower. 2048³ f16, corrected grid, median of
9, with a static ISA census beside each number:

| arm | TFLOP/s | wide loads | `global_load_d16` | `ds_store` | max VGPR |
|---|---|---|---|---|---|
| **scalar, pad 4** | **40.4** | 0 | 20 | **6** | 184 |
| split-vec, pad 0 | 34.1 | 2 | 16 | 17 | 193 |
| split-vec, pad 1 | 18.8 | 0 | 4 | 5 | 175 |
| split-vec, pad 2 | 30.7 | 2 | 8 | 9 | 183 |
| split-vec, pad 4 | 34.5 | 2 | 16 | 17 | 189 |

Neither arm spills (`scratch_ops = 0` both ways), so register pressure is not
it — the 126-spill figure on record belongs to the **register** body, not this
one.

**`ds_store` 6 → 17 is the finding.** The global read widened and the LDS write
got three times worse, because **B's write cannot widen at all**: the global
side is contiguous in N while the LDS side is contiguous in K per column, so
the fast path loads 8 elements in one instruction and then stores them with
eight separate `ds_store`s. The unrolled tail adds its own. The load saves less
than the store side costs.

**So the target was mis-stated, and CK says so explicitly.** The tuned CK
instance this item cites sets `SrcScalarPerVector=8` **and**
`DstScalarPerVector_K1=8` — both sides, 128 bits each way — and it can only
set the second because its LDS layout is **K1-blocked** (`K1=8`), not a plain
transpose. Widening the copy into our layout is therefore not a smaller version
of CK's configuration; it is the half of it that does not work alone.

**Re-specified gate:** the deliverable is the **K1-blocked LDS layout**, after
which both sides widen and the copy width becomes the tunable CK treats it as.
A wider copy over the current layout is measured-negative twice and should not
be attempted a third time.

### 10j.2 The ceiling probe, and the speculation it refutes

§10j.1 closed with an explicitly-flagged guess: that two failed copy attempts
were weak evidence the copy is *not* where the time goes, and that the
K1-blocked layout might therefore disappoint the same way. It said to measure
the split before building the layout, and to treat the paragraph as reasoning
rather than a result.

**Measured 2026-09-20 on gfx1201, and the guess was wrong.** The probe
(`lds-copy-elide`) emits a deliberately WRONG kernel whose staging copy writes
a constant instead of reading global — every output is zero, which the harness
asserts — while barriers, loop structure, LDS traffic and the MMA chain stay
identical. So the delta is the copy's entire cost:

| shape | real | copy elided | ceiling |
|---|---|---|---|
| 1024³ | 19.0 | 31.3 | **1.65x** |
| 2048³ | 40.4 | **120.1** | **2.97x** |

At 2048³ a free copy would reach **120 TFLOP/s, above the register body's
~84** — so the LDS body's compute structure is not the problem, and the copy
is worth up to 3x. The K1-blocked layout is worth building.

**Why the guess failed is worth more than the guess.** Two copy optimisations
had failed to move the body, and I read that as evidence about where the time
goes. It was not: both failures were about *how* the copy was expressed — a
masked load that cannot widen (§10j), then a widened load whose transposed
store cost more than it saved (§10j.1). A mechanism that fails twice for
reasons specific to the mechanism says nothing about the size of the prize.
Measuring the prize directly took one option and one afternoon; inferring it
from failures would have cancelled a 3x.

**Swept across five shapes, and a second hypothesis died.** Given the RX 9070
XT's hierarchy (L2 8 MB, Infinity Cache 64 MB), the two original shapes straddle
the L2 line exactly — 4 MiB of f16 operands at 1024³, 16 MiB at 2048³ — so the
obvious reading was that the copy is cheap while the operands fit L2 and dear
once they spill. That predicts a STEP between 1280 and 1536. Measured:

| N | A+B | resident | real | elided | ceiling |
|---|---|---|---|---|---|
| 1024 | 4.0 MiB | L2 | 18.9 | 31.4 | 1.66x |
| 1280 | 6.2 MiB | **L2** | 25.4 | 61.3 | **2.41x** |
| 1536 | 9.0 MiB | L3 | 32.5 | 97.4 | 3.00x |
| 1792 | 12.2 MiB | L3 | 32.5 | 127.3 | 3.91x |
| 2048 | 16.0 MiB | L3 | 40.4 | 126.2 | 3.13x |

No step. A smooth rise, and the largest single jump (1.66 → 2.41) happens
entirely **inside** L2. Cache residency is not the driver.

**What is:** the elided arm converts problem size into throughput (31 → 127,
plateauing near its roofline) while the real arm barely moves (18.9 → 40.4).
The ratio grows because the copy-free kernel uses the extra parallelism and the
copy-bound one cannot. The copy is the bottleneck at *every* shape here.

**So the 1024³ figure understates the prize, for a reason worth naming.** At
1024³ there are 64 work-groups over 32 WGPs — two each — so even the copy-free
kernel is occupancy-starved. 1.66x is not "the copy is cheap here"; it is
"nothing is fast here". An earlier draft blamed §10k's non-convergence at that
shape, which is a different effect and was the wrong attribution.

**Consequence for the layout work:** judge it across the occupancy range, not
at one shape. 1792³ carries the most headroom (3.91x) and 1024³ the least, and
a single-shape benchmark will over- or under-state the result by roughly 2.4x
depending only on which one is chosen.

**What the ceiling still does not say.** It is an upper bound on a copy that
cannot actually be free, and it does not rank K1-blocking against other ways
to spend the headroom.

### 10j.3 Where the 3.91x actually is: the staging loop does not pipeline

The ceiling says the copy is worth up to 3.91x. The ISA says what a copy is
supposed to look like, and our two bodies disagree completely. Counting memory
waits in the emitted gfx1201 ISA at 2048³:

| body | demand-zero waits | partial waits | distribution |
|---|---|---|---|
| register | 57 | **137** | `loadcnt<=1` x50, `<=2` x10, `<=4` x8, `<=6` x8 |
| **lds** | **134** | 15 | `loadcnt<=0` x2, `dscnt<=0` x2 |

RDNA4 5.7 is explicit that `S_WAIT_LOADCNT` takes a bound, not just zero, so a
shader can "schedule long-latency instructions, execute unrelated work and
specify when results are needed". The **register body does exactly that** --
six global loads in flight, waits on partial counts. The **LDS staging loop
does not**: it issues a load and waits for `LOADcnt<=0` before storing to LDS.

**Barriers are not the cost, and the elided arm proves it.** The copy-elided
kernel keeps every barrier and removes only the global reads; it runs at 127
TFLOP/s, near roofline. If the two barriers per K step were dominant it could
not. So the 3.91x is unhidden **load latency**, not synchronisation.

**Barriers are, however, the likely reason it cannot be hidden** -- and this
part is inference, flagged as such. The loop is `barrier -> load -> wait(0) ->
ds_store -> barrier -> MMA`, and a load for the next K step cannot be hoisted
above the trailing barrier without changing what the barrier means. The
register body has **no barriers at all**, which is precisely why its scheduler
is free to keep six loads in flight. Testing this needs a double-buffered
variant, not more counting.

**Both levers the ISA offers are unused.** The barrier is split into
`S_BARRIER_SIGNAL` (arrive) and `S_BARRIER_WAIT` (5.6), which is what lets a
wave signal after its stores and wait only before reading someone else's slab.
We emit **2 signal and 2 wait, both pairs adjacent** -- every barrier a full
stop. `gpu.barrier` lowers that way and nothing asks for more.

**So the next move is probably not the K1-blocked layout.** K1-blocking widens
each transfer; double-buffering hides its latency. The measurement says latency
is what is exposed, and the register body -- same chip, same loads, no barriers
-- already demonstrates the hiding. A double-buffered LDS slab with a split
barrier is a smaller change than a layout rewrite and attacks what was actually
measured. The layout may still be wanted afterwards for the store-side width
(§10j.1), but it should be judged after the latency is hidden, not before.

### 10j.4 Four levers for the staging copy, and which the compiler can reach

The ceiling (§10j.2) says up to 3.91x is available; the wait census
(§10j.3) says it is sitting in unhidden load latency. The RDNA4 ISA offers
four ways at it, and they are not equally reachable from our pipeline:

| lever | fixes | reachable from MLIR? |
|---|---|---|
| **double-buffer + split barrier** | exposed load latency | partly — `gpu.barrier` emits `signal`+`wait` adjacent; the split needs a producer |
| **`global_load_tr_b64/b128`** | B's transpose at load time | not checked yet |
| **cross-lane transpose** | B's store width | **yes**, via `permlane16.var` |
| K1-blocked LDS layout | both sides widen | yes (it is our own layout) |

**On the cross-lane option, a correction worth recording.** The transpose B
needs is 8 lanes x 8 elements — each lane reads 8 contiguous N and must end
holding 8 contiguous K — which is exactly DPP8's granularity (7.9: "arbitrary
cross-lane swizzling within groups of 8 lanes"). But **ROCDL does not expose
DPP8**: `ROCDL_DPPUpdateOp` is `update.dpp`, the DPP16 intrinsic with its
predefined menu, and there is no `mov.dpp8`. Checked against the LLVM 23.1.1
`ROCDLOps.td` on 2026-09-20.

What is exposed, and is more general for this: `permlane16.var` and
`permlanex16.var` — arbitrary gather with a **per-lane** select within (or
across) 16-lane groups. An 8x8 transpose fits inside a 16-lane group. Also
`ds_swizzle` (fixed menu, 32 lanes) and `permlane32.swap`.

So "use DPP8" would be an ISA-correct plan the compiler cannot express, which
is the Decision #19 question in miniature: the architecture having an
instruction is not the same as our pipeline being able to emit it.

**One correctness note if this is built.** DPP8 has two forms and the
distinction survives into any lane-shuffle design: normal reads **zero** from
EXEC-masked lanes, `DPP8FI` fetches the inactive lanes' actual data. For the
ragged K tail the zero behaviour is the one that is *wanted*; fetching
inactive lanes would carry stale values into the fragment silently. Whichever
primitive is used, check which of the two semantics it has before trusting the
tail.

**WMMA itself takes no DPP** (Table 38), so all of this has to happen before
the matrix op — there is no swizzle-on-the-way-in.

### 10j.5 A constraint the layout work must not break: sparse needs column-major B

Verified against the RDNA4 ISA text 2026-09-20, independently of the matrix
calculator this file's tables came from. Two results and one warning.

**Our fragment formulas are confirmed.** Deriving from 7.12's own tables --
A 16-bit wave32 is `lane = {col[2], row[3:0]}`, `vgpr = {col[3], col[1]}`,
`startPosn = col[0]`, which gives `k = 8*(e>>2) + 4*(lane>>4) + (e&3)`; A 8-bit
is `lane = {col[3], row[3:0]}`, `vgpr = col[2]`, `startPosn = col[1:0]`, giving
`k = 8*(lane>>4) + e`. Both match §3 exactly. The C/D map
`VGPR j at lane L = D[(L/16)*8 + j][L%16]` matches too. Two independent sources
now agree on all three.

**Our departure at 16 bits is the known one.** `materializeFragmentPack` uses
`kBase = laneGroup * inputElementsPerLane`, i.e. contiguous-eight at every
width. That IS the machine layout at 8/4 bits and a permutation of it at 16,
legal only because K reduction is permutation-invariant when the same
permutation reaches BOTH operands. Recorded here because the ISA text makes the
departure visible for the first time.

**The warning, and it lands on the K1-blocked layout.** 7.12's sparse section
says: "When the A-matrix is a 4:2 sparse matrix, the corresponding B-matrix
must be (K x N), and loaded in **column-major** order." Nothing in
`rocm_sparse_{logical,packing,runtime}.py` or the ROCm conversion states that
constraint. It is satisfied today only **by accident of the staging layout** --
§10j.1 established that our LDS B is written contiguous in K per column, which
is column-major.

So the 2:4 sparse stack depends on a property of the dense staging layout that
nobody wrote down, and the K1-blocked layout under consideration changes
exactly that property. Before any B-layout change: state the constraint at the
sparse site, and give it a test that fails when B stops being column-major.
Otherwise the failure mode is wrong sparse results from a change made for dense
performance, with nothing connecting the two.

(Also confirmed while reading: our packer satisfies the `idx0 < idx1` rule,
though via a `sorted()` call rather than a stated invariant.)

## 10k. The padding default, settled on the shape where it converges

§10e chose `lds_pad_dwords=1` from the broken harness; §10i withdrew that
without replacing it. Measured 2026-09-19 on `Tajasarus`, corrected grid,
scalar copy (the §10j default), f16, median of 9 trials:

| pad | LDS row stride (f16) | 1024³ | 2048³ |
|---|---|---|---|
| 0 | 16 | 17.4 | 35.0 |
| 1 (old default) | 18 | 15.7 | 39.5 |
| 2 | 20 | 14.3 | 39.7 |
| **4 (new default)** | 24 | 15.0 | **40.3** |

**The two shapes disagree, and only one of them is measuring anything.** At
2048³ padding helps monotonically, pad=4 wins by **15%**, and median equals
best to 0.1 — the dispersion is nil. At 1024³ the ordering inverts, and it is
noise: interleaving pad 0 and pad 4 over 21 trials **in one process** gave

```
pad 0 -> 17.6   (IQR 0.3)
pad 4 -> 17.7   (IQR 1.5)
pad 0 -> 15.2   (IQR 0.3)      <- same configuration, 16% lower
pad 4 -> 17.6   (IQR 1.7)
```

so pad=0 remeasures 16% apart against itself while its own IQR says 0.3. A
tight IQR around a drifting median is not precision — it is a run-to-run shift
inside a run-length the harness never spans. At 1024³ the LDS body is ~120 us
over 64 workgroups on 32 WGPs, short enough for clock behaviour to dominate;
2048³ is 8x the work over 256 workgroups and is stable.

**So the default comes from 2048³ and the 1024³ column decides nothing.** The
cost is 1.5x the LDS footprint (8 KiB → 12 KiB per workgroup at the 2x2-wave
128x128 tile), which is far inside gfx1201's 128 KiB per WGP and does not
change occupancy at this tile.

**And §10e's mechanism does not predict the winner.** Bank behaviour depends on
`gcd(stride_in_dwords, 32)`:

| pad | stride (dwords) | `gcd(.,32)` | model says | 2048³ measured |
|---|---|---|---|---|
| 0 | 8 | 8 | worst | 35.0 (worst ✓) |
| 1 | 9 | **1** | **best** | 39.5 |
| 2 | 10 | 2 | middling | 39.7 |
| 4 | 12 | **4** | bad | **40.3 (best ✗)** |

pad=4 reinstates a 4-way conflict and still wins. What the data actually shows
is a **+13% step from pad=0 to any padding at all**, and then a 2% monotonic
drift with stride that the conflict model has no term for. So the conflict
model explains the step and nothing else; the residual is probably `ds_load`
width or address alignment, and it is unexplained rather than explained-away.

The default is **4** because it is the pooled winner — best at 2048³, and at
1024³ it spans 15.0-17.8 against pad=1's 14.4-16.5 — not because the model
endorses it. pad=1 is within 2% at 2048³ on 25% less LDS and is a defensible
alternative the moment the residual is explained.

**What is still open** is that a padding default exists at all: the right
answer is shape-dependent and the mechanism (bank conflicts on the fragment
read vs. the wider `ds_load` a small stride allows) is shape-independent, which
means the disagreement is really the *measurement* failing at small shapes.
The per-shape selection item downstream of `ROCM-MACRO-K-TILE-1` owns it, and
its first job is a 1024³ measurement that converges, not a better default.

## 11. Which recorded results the default schedule qualifies

Every gfx1201 number in this file and in the ROCm queue was measured with **no
scheduling intrinsic emitted anywhere in this backend**, i.e. under the drained,
single-buffered default §10 describes. That does not invalidate a comparison
between two configurations measured the same way, but it does bound what any of
them can claim about the *approach*:

| Result | Status under §10 |
|---|---|
| The LDS-staged body loses at every shape | **Still qualified, but scheduling is ruled out as the explanation (§10c).** Barriers make it *slower*, not faster. Measured 7.8 against the register body's 70.5 — 9x, which is what an 8-way LDS bank conflict costs, and our rows are unpadded while the kernel that reports LDS staging as essential pads 8 bytes. Re-judge after ROCM-LDS-BANKPAD-1, not after a scheduling change. |
| The 4×4 panel spills 133 VGPRs, ~200 of them neither accumulator nor fragment | **Doubt withdrawn 2026-09-19, measured (§10c).** Group barriers move the spill by 7% at best (133→123) and cost 34% throughput doing it. Outstanding-load state is not what the 4×4 panel is holding; §9's per-tile-addressing diagnosis stands. |
| K unroll 2 beats 4 | **Qualified.** Unroll depth changes how many loads are in flight, which §10 says the scheduler owns. The k=4 rule was already withdrawn once for an unreproduced row; this supplies a mechanism for why it was unstable. |
| The double-K int4 instruction loses to the K unroll | **Qualified.** The recorded explanation was that the unroll "keeps two independent MMAs in flight" — in-flight-ness is precisely what §10 says is not ours today. |
| `GLOBAL_LOAD_TR_B128` is worth +21.6% at 1024³ | **Stands as a comparison.** Both arms are register-path kernels measured identically, so the relative figure holds. The absolute ceiling may move. |
| The 2048³ transpose-load reversal | **Stands as an anomaly, unexplained.** Still needs counters neither WSL2 ROCm box can produce. |

Closing this is `ROCM-SCHED-GROUP-1`: emit `sched_group_barrier`, then re-measure
the four qualified rows before any of them is treated as settled.

## Where the machine truth lives

Opcode tables, pseudocode and the VGPR-usage tables come from
`docs/reference/isa/rdna/` (a regenerable extraction of AMD's ISA guides — JSON
is truth, markdown is a mirror; do not hand-edit it). This page is the
*consumer-facing contract* Tessera's generators are written against, with the
device evidence attached; when the two disagree, the archive wins on what the
hardware does and this page is the bug.

## 10l. Chapters 9, 11 and 12 of the ISA, checked against what we emit

A sweep of the addressing, alignment and LDS chapters against our two gfx1201
bodies. Most of it does not reach us, but the reasons are worth recording
because three of them were things I had *asserted* rather than checked, and one
of those assertions was about to be used to dismiss a section that does apply.

### What we emit, measured

Disassembling both bodies and counting instruction families:

```
register   global_*=472, scratch_*=145,  ds_*=0     buffer/tbuffer: NONE
lds        global_*=150, ds_*=14                    buffer/tbuffer: NONE
```

Zero `BUFFER_*`/`TBUFFER_*`. ROCDL lowers everything to flat/global addressing,
so §9.2's buffer-VGPR layout rules and §9.3's `dfmt`/`nfmt` mismatch table --
which describe what happens when a typed buffer op's format disagrees with the
resource descriptor -- are unreachable from this pipeline. That was already my
belief; it is now a measurement, and the distinction mattered: I dismissed §9.3
from belief first, and the same reasoning would have dismissed §9.5, which does
apply.

### §9.5 / §11.3: two silent-wrong-address modes, neither of which we control

`SH_MEM_CONFIG.alignment_mode` governs **non-formatted** ops -- that is, the
`global_load_*` our staging copy issues:

| mode | misaligned DWORD+ access |
|---|---|
| 0 DWORD | **the two LSBs are ignored** -- reads a different address, no fault |
| 1 DWORD_STRICT | must be aligned |
| 2 STRICT | must be aligned to the data size |
| 3 UNALIGNED | any alignment |

This is the global-memory twin of the LDS hazard in §3.3.5.1, where a B128
access below 16-byte alignment has its low address bits zeroed. Mode 0 turns a
misaligned wide load into a wrong-address read that neither faults nor differs
in timing. It is a config register set by the driver, so we cannot select it and
must not depend on it.

§11.3 adds a third: LDS address arithmetic is **truncated and may wrap without
being detected**. The only range check is `LDS_ADDR.U17 < LDS_SIZE`, zero-extended.
Inside the allocation, a wrapped address is simply a different valid address.
Out-of-range is caught -- reads return zero, stores are dropped, MEMVIOL traps --
but wrap-around inside the wave's own LDS is not.

All three are silent-wrong-answer modes reachable by an address-arithmetic bug,
which is precisely what the K1-blocked layout work will be writing. Our current
vector-width derivation (`ldsStride % v == 0` in elements) prevents the LDS one
by construction, because element count and byte count scale together. It does
not prevent the other two, and neither has a test.

### §11.5 `GLOBAL_LOAD_BLOCK`: a fifth lever, and ROCDL cannot reach it

§10j.4 listed four levers for the staging copy. There is a fifth, and on paper
it is aimed exactly at §10j.3's finding that the cost is unhidden load latency
rather than transfer width:

> The entire block load/store is tracked with LOADcnt: increments 1 for the
> entire block transfer, and decrements when the block transfer has completed.

Up to 32 consecutive VGPRs per thread, one counter. §10j.3 measured 137 partial
waits in the register body because every narrow load carries its own counter;
this instruction collapses that to one.

ROCDL has no `load_block` intrinsic -- grepping `ROCDLOps.td` finds only
`workgroup.id.*` under that name. Same class as DPP8 in §10j.4: an ISA lever the
compiler cannot reach through this lowering. Reaching it would mean inline asm,
which is a Decision #31 question, not a scheduling one.

### There is no LDS-side transpose read on gfx1201

ROCDL declares two transposing-read families, and neither is ours:

```
// LDS transpose intrinsics (available in GFX950)
def ROCDL_ds_read_tr16_b64 ...
// Glb/DS load-transpose intrinsics (available in GFX1250+)
def ... ds.load.tr16.b128 ...
```

`ds.read.tr*` is gfx950 (CDNA); `ds.load.tr*` is gfx1250+. gfx1201 has only the
**global**-side transpose, which we already emit through the `amdgpu` dialect
(§8) because ROCDL's form takes `!llvm.ptr<1>` and the fragment materializer
works in memrefs.

This closes a door rather than opening one. §10j.5 recorded that sparse needs
column-major B and currently gets it only by accident of the dense staging
layout; a transpose on the read *out* of LDS would have been the clean escape
hatch, and this chip does not have one. The K1-blocked layout must therefore
satisfy sparse's major order directly.

### §12.1: "64 banks" does not falsify our 32-bank model

The ISA says 128 kB per WGP in **64** banks, which read naively makes every
bank-conflict figure in §10e and §10k wrong by 2x. It does not:

> These 64 banks are further sub-divided into two sets of 32-banks each where 32
> of the banks are affiliated with a pair of SIMD32's, and the other 32 with the
> other pair.

A wave sees 32. The `32 x 4 B` model in `GenerateWMMAGemmKernel.cpp` is the
per-wave view and is correct; the 64 is the per-WGP total. Recorded because the
two numbers are one sentence apart and only one of them is the one a conflict
calculation wants.

### CU-mode-only instructions are unreachable, and we measured the mode

`DS_DIRECT_LOAD` (§12.1.2) and `DS_PARAM_LOAD` (§12.2) are both documented
"available only in CU mode, not WGP mode". Both our bodies dispatch in WGP mode
(`rsrc1` WGP_MODE=1, measured -- the same reading §10a used to settle the
occupancy denominator), so neither is reachable without changing the dispatch
mode. `DS_PARAM_LOAD` would not be wanted anyway: it reads pixel-attribute
triples for interpolation. `DS_DIRECT_LOAD`'s broadcast-a-DWORD-to-all-lanes
behaviour has a compute use, but `s_load` into an SGPR already covers it without
giving up the WGP-mode occupancy.

### §12.5: three more levers, and the same ROCDL wall -- except for one

The LDS indexed-access chapter offers three instructions that look aimed at the
staging copy. Checked against `ROCDLOps.td` the same way `GLOBAL_LOAD_BLOCK`
was:

| ISA instruction | what it would buy | ROCDL |
|---|---|---|
| `DS_STORE_2ADDR_{B32,B64}` | two stores at unique addresses, **one DScnt** | absent |
| `DS_STORE_ADDTID_B32` | address from thread-ID, **no ADDR VGPR** | absent |
| `DS_PERMUTE/BPERMUTE_B32`, `DS_SWIZZLE_B32` | cross-lane with no LDS storage | **present** |

The first two are the ones that fit our problem best and neither is reachable.
`2ADDR` is a particularly close fit: it takes one ADDR VGPR plus two immediate
offsets, and the two elements a lane stores in our staging layout are separated
by exactly `ldsStride`, a compile-time constant -- the precise shape the
instruction wants, and §10j.1's "B's store cannot widen" is exactly the problem
it would sidestep, since 2ADDR needs no contiguity. `ADDTID` would free the
address VGPR in a kernel that already spills 126. Both would need inline asm.

The third is a correction to §10j.4, which recorded that DPP8 is not in ROCDL
and left the impression that cross-lane movement is unreachable. It is not:
`rocdl.ds_bpermute` and `rocdl.ds_swizzle` are both declared. But they are not
DPP8 by another name -- DPP is a VALU operand modifier, while these run through
the LDS hardware (using no LDS storage) and are tracked with DScnt. So the
capability exists at a cost, rather than being absent. Anything built on it must
be measured against that cost, not assumed free because the crossbar is
arbitrary.

Two semantics worth carrying if we ever use them: index values are **bytes**
(multiply the lane by 4) with `offset0` added before use, out-of-range indices
**wrap** rather than fault (wave32 uses only index bits [6:2]), and reading a
disabled lane returns zero. The wrap is the same silent-wrong-answer shape as
the LDS address truncation above.

### 10j.6 The vectorised path's penalty is not the global load

Fixing the ceiling probe to write every destination at every width (it had been
storing one scalar per `vecW` group, so above width 1 it read stale LDS and
under-counted its own LDS traffic -- review finding, 2026-09-20) made it usable
at vector widths for the first time. Running it there answers a question §10j
left open.

gfx1201, 1024^3 f16, pad=4, **elided arm only** -- no global reads at all:

| `lds-copy-width` | TFLOP/s | all-zero |
|---|---|---|
| 1 | 30.4 | True |
| 2 | 12.5 | True |
| 4 | 11.9 | True |
| 8 | 12.0 | True |

The elided arm has no global load, no `scf.if`, and no ragged tail -- the elide
branch sits above that split. Total LDS elements written is identical at every
width (A's one vector store replaces `vecW` scalars; B stays scalar either way),
and the loop runs `vecW` times fewer iterations. The vector arm should therefore
be **faster**, and it is 2.5x slower.

So the regression §10j measured is not, or not mainly, about the load side.
Something in the vectorised staging path costs ~2.5x with the global read
already removed. The mechanism is not established -- candidates are the LDS
store pattern under the padded stride, VGPR pressure in a body that already
spills 126 (§10j / the RDNA4 VGPR ceiling), or the wider loop body's scheduling
-- and this is one shape on one arm, so it is a signal, not a conclusion.

**CORRECTED 2026-09-20, same day, on review: the last sentence of this section
originally read "argues against reviving vectorisation". That is backwards, and
it is the wrong conclusion from the right measurement.**

What the measurement shows is that vectorisation *as implemented here* loses.
The staging loop is `load; store; load; store` -- one dependency chain, nothing
in flight. Widening from 1 to 8 does not add overlap, it **removes** it: the
loop now has 8x fewer independent iterations, and iteration count was the only
source of memory-level parallelism in that body. Eight independent narrow
operations became one wide one with nothing to fill the gap.

That also re-reads §10j.3. Its census found the register body at `loadcnt<=6`
and the staging loop at `loadcnt<=0`, which was recorded as "unhidden load
latency" and sent us looking for a *faster instruction* -- the search in §10l
that found four unreachable ones. The correct reading is that the loop has no
issue depth, and no instruction fixes that. A static census of the two real arms
is consistent: the scalar path reaches 20 memory ops between waits, the vector
path 10 (suggestive only -- the count spans the whole kernel, not just the
staging loop).

**A wide operation only pays if enough of them are in flight.** Three sources of
overlap exist here and the current loop uses none:

1. **Issue depth inside the copy.** Issue every load for a tile before waiting
   on any (`for i: v[i]=load(i)` then `for i: store(v[i])`), not load-wait-
   store-repeat. This is what moves the staging loop off `loadcnt<=0`. It
   competes for registers: at width 8 each in-flight value is 4 VGPRs, and
   RDNA4 caps a wave at 256 architecturally, in a body already spilling 126 --
   so wide x deep may lose to narrow x deep, which is untested.
2. **Double-buffering across K-tiles.** Stage into LDS[1] while the MMA chain
   consumes LDS[0], hiding global latency behind compute. Needs 2x LDS (there is
   room under the 64 KB workgroup cap) and the split `S_BARRIER_SIGNAL` /
   `S_BARRIER_WAIT` pair.
3. **Prefetch depth > 2** if the MMA window proves shorter than memory latency.

(1) fills the pipe within the copy; (2) hides the copy behind compute. They are
orthogonal and vectorisation multiplies both, so it is worth nothing without
either. The right next experiment is (1) measured against the current loop at
each width -- it is the smaller change and it directly tests the diagnosis. Until
that runs, **nothing here licenses a claim about vectorisation on this chip**,
only about this loop structure.

It also would not have been visible before the probe was corrected: at widths
2-8 the old probe returned `nan` and a non-zero output, so any timing from it
was measuring a kernel whose LDS was partly uninitialised.

## 10m. Issue depth, measured: the copy was latency-bound on request COUNT

§10j.6 argued the staging loop's defect is structural -- `load; store; load;
store` keeps exactly one global load in flight -- and named issue-then-wait as
the experiment. `lds-copy-depth=N` splits each batch into an issue phase and a
drain phase so N loads are outstanding. Measured on gfx1201, f16, 4x4 panel,
2x2 waves, pad=4. Every cell below is numerically exact; a wrong result fails
the harness rather than appearing as a number.

|  | 1024^3 | 2048^3 |
|---|---|---|
| w=1 (trip 16) | d1 **18.3** / d2 7.8 / d4 11.6 / d8 11.6 / d16 14.9 | d1 **40.3** / d2 26.9 / d4 32.3 / d8 32.6 / d16 34.5 |
| w=2 (trip 8) | d1 7.6 / d2 11.8 / d4 15.1 / d8 **21.0** | d1 18.3 / d2 43.4 / d4 48.4 / d8 **56.1** |
| w=4 (trip 4) | d1 12.4 / d2 16.1 / d4 **19.4** | d1 25.7 / d2 32.4 / d4 **40.4** |
| w=8 (trip 2) | d1 16.3 / d2 **20.1** | d1 34.2 / d2 **39.2** |

TFLOP/s. `w=1 d=1` is the shipped default. Best is **w=2 d=8**: **+15%** at
1024^3 and **+39%** at 2048^3.

### The prediction this refutes

It was recorded before the run: *depth is capped by trip count (16/width), so
w=1/d=16 can hold 16 requests where w=8/d=2 holds 2; therefore w=1/d=16 wins
outright.* **It does not.** `w=1 d=16` is worse than `w=1 d=1` at both shapes,
and width 1 is the only width that does not improve with depth.

The diagnosis was right and the prediction from it was wrong, because it counted
requests and ignored their size. **A width-1 f16 load is a 2-byte request** --
sub-dword, so half of every transaction is discarded. Sixteen of those in flight
is sixteen wasted half-transactions. `w=2` is exactly one dword, the natural
granularity, and it is the narrowest request that wastes nothing.

### Two costs, and why the optimum is interior

The three configurations that consume the whole trip in one batch move
identical bytes and hold identical value registers (16 f16 = 8 dwords):

| config | 1024^3 | 2048^3 |
|---|---|---|
| w=1 d=16 | 14.9 | 34.5 |
| **w=2 d=8** | **21.0** | **56.1** |
| w=8 d=2 | 20.1 | 39.2 |

What separates them is that **depth costs address and predicate registers while
width does not**: d=16 keeps sixteen address computations live in a body already
spilling 126 against RDNA4's 256-per-wave architectural ceiling. So the curve
has a genuine interior optimum -- enough outstanding requests, each at least a
dword, without paying for more live addresses than the register file has room
for. Neither axis alone finds it, which is why every previous single-axis
attempt (§10j, §10j.1) was a wash or a regression.

### Dispersion

Five independent remeasurements per cell:

| shape | w1/d1 | w2/d8 |
|---|---|---|
| 1024^3 | 9.1 - 18.2, spread **56.5%** | 20.7 - 22.6, spread 8.7% |
| 2048^3 | 40.1 - 40.3, spread 0.4% | 55.8 - 56.0, spread 0.4% |

The ranges do not overlap at either shape: at 1024^3 the new configuration's
*minimum* exceeds the old one's *maximum*. This matters because 1024^3 is the
shape §10k recorded as remeasuring 16% apart and which swings 56% here.

That swing is itself corroboration rather than noise to be averaged away. A
latency-bound body with **one** request in flight is at the mercy of memory
timing; with eight it is not, and the spread falls from 56.5% to 8.7%. The
mechanism predicts the variance reduction as well as the mean, which a
bank-conflict or instruction-count explanation would not.

### What this does and does not license

Measured on **gfx1201 only**, **f16 only**, one panel and one wave shape. It
does not transfer to gfx1151 (`ROCM_AUDIT.md`), to other dtypes -- where the
dword argument shifts, since w=1 at fp32 is already a full dword and w=4 at fp8
is -- or to other panels, whose trip counts differ and therefore whose available
depths do too. The default is unchanged pending that coverage.

Against the ceiling: §10j.2 put the 2048^3 copy at 3.13x (40.4 -> 126.2 if
free). Going 40.3 -> 56.1 captures about **18%** of that headroom, so most of it
remains, and double-buffering across K-tiles -- which hides the copy behind the
MMA chain rather than making it cheaper -- is still the larger unclaimed lever.

## 10n. Double-buffering across K-tiles: correct, ordered as intended, and worth 4%

§10j.2's ceiling probe put the 2048^3 staging copy at 3.13x, and §10m captured
about 18% of that with issue depth. Double-buffering was the named next lever:
stage slab k+1 into a second LDS buffer while the MMA chain consumes slab k, so
the load latency hides behind compute instead of sitting in front of it.

`lds-double-buffer=true`. Default **off**.

### The ordering constraint, which is the whole design

Emitting `load; store; mma` leaves the loads' waitcnt ahead of the MMA and
overlaps nothing, however many buffers exist. The loop must ISSUE the loads, run
the MMA chain, and DRAIN to LDS afterwards.

That forces a structural requirement that is easy to miss: while the staging is
an `scf.for`, the loaded values and their destination addresses are SSA values
inside that region, and **the store cannot leave the loop**. Double-buffering
therefore requires `depth == trip` -- one straight-line batch covering the tile
-- which `flatStage` emits without the loop. A first implementation that kept
the loop compiled, ran, and would have measured nothing.

One barrier per iteration instead of two: a buffer's reads in step k-1 precede
the barrier and its writes in step k follow it, so the two never alias.

### It works, and the ISA confirms the intent reached the machine

2048^3, static counts:

| | total ops | spill | wmma | global_load | **loads before 1st wmma** |
|---|---|---|---|---|---|
| w2/d8 | 3695 | 0 | 16 | 48 | **0** |
| w2/d8 + dbuf | 4439 | 0 | 16 | 96 | **48** |

48 loads now issue ahead of the MMA chain where none did. **Zero spills in
both**, which refutes the register-pressure concern raised in §10m -- the staged
values live across the MMA chain without forcing a spill. (The doubled static
counts are the prologue, emitted once outside the loop.)

### The result

Correctness first: exact (max|err| = 0) at all five shapes tested, including
three ragged ones, where the tail arm now assembles vectors it previously stored
element by element.

Performance, **paired and interleaved in one process** -- the first two attempts
disagreed on the *sign* because each measured one arm at a time and the 2048^3
baseline moves ~13% between processes:

| shape | no-dbuf | dbuf | verdict |
|---|---|---|---|
| 2048^3 | med 48.8 | med 50.8 | **1.040x, wins 8/8** |
| 1024^3 | med 22.6, range 7.8-23.0 | med 15.9, range 8.3-24.2 | **both arms bimodal -- no conclusion** |
| 768x1536x520 (ragged) | 14.0 | 7.1 | **0.51x regression** |

`k-unroll=2`, which lengthens the compute per staging, did not improve it
either (54.0 vs 56.9 without dbuf), which argues against "the MMA chain is too
short to cover the latency".

### What this says

The intended overlap was achieved and verified in the ISA, there are no spills,
and it is worth **4%**. The honest reading is that **once issue depth is fixed,
the remaining staging cost is not mostly hideable load latency.** §10m's
16-requests-in-flight already captured the accessible part; a second buffer adds
LDS pressure, code size and a prologue for very little.

So the 3.13x ceiling's remaining headroom is probably not load latency at all.
The elide probe removes the loads *and their address arithmetic*; what is left
in the real copy is that arithmetic and the LDS stores themselves, and neither
is addressed by overlapping with compute. That is the next thing to measure, and
it is a different question from the one §10j.3 set up.

### Two cautions this run produced

**The 1024^3 figures in §10m are weaker than they looked.** That shape is
bimodal here (7.8-23.0 in one arm), where the §10m dispersion check -- five
back-to-back reps of one payload -- reported 8.7%. Alternating between two
payloads exposes a mode the repeated measurement did not. Treat §10m's +15% at
1024^3 as provisional; its +39% at 2048^3 is unaffected, that shape stayed tight
under both protocols.

**Measure paired arms interleaved, in one process.** Two sequential runs here
produced opposite signs (50.2 vs 51.6, then 56.9 vs 52.3) purely from
process-to-process drift landing on whichever arm ran second.
