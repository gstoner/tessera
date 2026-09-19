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

```
a[h] = A[lane % 16][8 * (lane / 16) + h]        // A row-major [M][K]: row = lane % 16
b[h] = B[lane % 16][8 * (lane / 16) + h]        // B K-major   [N][K]: col = lane % 16
```

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

## Where the machine truth lives

Opcode tables, pseudocode and the VGPR-usage tables come from
`docs/reference/isa/rdna/` (a regenerable extraction of AMD's ISA guides — JSON
is truth, markdown is a mirror; do not hand-edit it). This page is the
*consumer-facing contract* Tessera's generators are written against, with the
device evidence attached; when the two disagree, the archive wins on what the
hardware does and this page is the bug.
