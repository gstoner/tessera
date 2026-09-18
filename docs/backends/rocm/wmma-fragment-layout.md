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
gfx1201. Independently confirmed by an external probe on a second stack
(Radeon AI PRO R9700, gfx1201, ROCm 7.14 nightly) against a CPU reference of
`A·Bᵀ`: 256/256 elements match the column-distributed mapping across three
cases, and the row-major reading matches 16/256, as expected.

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
it loses on gfx1201 and wins only narrowly on gfx1151 (see the typed-route gap
packets), because the barriers cost more than the stride does.

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
accumulate.

**In Tessera**: `materializeFragmentPack` compacts nibbles with exactly that
rule (`shift = 4 * (i % 8)`, word `i / 8`), and int4 values ride int8
containers at the ABI boundary, one logical value per byte, range `[-8, 7]`.

*Evidence*: the typed int4 and int8 matmul device rows are **exact** against
the int32 numpy product on both chips at 16³, 17×19×23 and 65×48×37 —
`test_gfx1201_scheduled_matmul_package_executes_integer_storage` and its
gfx1151 twin. An off-by-one nibble order would not be exact anywhere.

## Where the machine truth lives

Opcode tables, pseudocode and the VGPR-usage tables come from
`docs/reference/isa/rdna/` (a regenerable extraction of AMD's ISA guides — JSON
is truth, markdown is a mirror; do not hand-edit it). This page is the
*consumer-facing contract* Tessera's generators are written against, with the
device evidence attached; when the two disagree, the archive wins on what the
hardware does and this page is the bug.
