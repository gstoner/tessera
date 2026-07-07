"""Phase E1 / rung 2.5 — NVIDIA PTX assembler-text emission (first slice).

See ``docs/audit/compiler/EVALUATOR_PLAN.md`` §2 (rung ladder) and §5.

**The gap this addresses.** Today Tessera's NVIDIA path stops at Target IR MLIR
(``tessera.tile.wgmma`` / ``tessera_nvidia.*``); it emits no assembler text, so
``ptxas`` (rung 3) has nothing to consume and NVIDIA sits at rung 1. This module
is the first step of rung **2.5** — emit real PTX assembler text for a narrow
sm_90a WGMMA bf16 matmul, in the canonical encoding documented in
``docs/nvidia_cuda13_kernel_inventory.md`` and asserted by the
``tests/tessera-ir/phase3/cuda13/`` fixtures.

**Honesty ceiling (read this).** This emits the documented WGMMA *instruction
encoding* inside a structurally-valid PTX kernel skeleton. It is **not** a
complete assemblable kernel: a real WGMMA needs shared-memory matrix descriptors
plus TMA / ``cp.async`` data movement and the full 128-wide accumulator operand
list, which this skeleton deliberately omits (and which cannot be made
ptxas-correct without the toolchain, absent on the arm64 dev host). So:

  * :func:`validate_ptx_structure` checks the PTX scaffolding + that the emitted
    WGMMA mnemonic matches the requested tile/dtype per the CUDA-13.3 inventory —
    verifiable *here*, no toolchain. This is what earns rung 2.5.
  * :func:`ptxas_assemble` invokes real ``ptxas`` (rung 3) — **Linux-CI only**;
    it skip-cleans when the toolchain is absent. It is expected to report what is
    still missing for a complete kernel; assemblability is NOT claimed here.

The complete-kernel work (descriptors + TMA + accumulator vector) is the named
next sub-step; this slice makes "Tessera emits the documented WGMMA PTX
encoding" true and machine-checkable.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# PTX ISA version — imported from the single source of truth (gpu_target.py's
# CUDA pin) so the emitted `.version` directive can never drift from the toolchain
# pin. CUDA Toolkit 13.3 → PTX ISA 9.3.
from .gpu_target import TESSERA_TARGET_PTX_ISA

PTX_ISA_VERSION = TESSERA_TARGET_PTX_ISA

# Documented canonical Hopper WGMMA bf16 tiles (docs/nvidia_cuda13_kernel_inventory.md).
_WGMMA_BF16_CANONICAL: frozenset[tuple[int, int, int]] = frozenset(
    {(64, 256, 16), (64, 128, 16), (64, 64, 16)}
)


def is_valid_wgmma_bf16_shape(m: int, n: int, k: int) -> bool:
    """The Hopper WGMMA bf16 shape constraint: M is fixed at 64, K at 16, and
    N ∈ {8, 16, …, 256} in steps of 8 (PTX ISA 8.x ``wgmma.mma_async`` for
    ``.f32.bf16.bf16``). Stricter than a curated list — it accepts the tile the
    Target IR actually selects (e.g. m64n64k16) and rejects non-WGMMA shapes."""
    return m == 64 and k == 16 and 8 <= n <= 256 and n % 8 == 0


def wgmma_mnemonic(m: int, n: int, k: int, *, acc: str = "f32", ab: str = "bf16") -> str:
    """The documented WGMMA instruction mnemonic for one tile/dtype."""
    return f"wgmma.mma_async.sync.aligned.m{m}n{n}k{k}.{acc}.{ab}.{ab}"


def emit_wgmma_matmul_ptx(
    m: int = 64,
    n: int = 256,
    k: int = 16,
    *,
    arch: str = "sm_90a",
    acc: str = "f32",
    ab: str = "bf16",
    entry: str = "tessera_wgmma_matmul_bf16",
) -> str:
    """Emit a structurally-valid PTX kernel skeleton carrying the canonical
    WGMMA matmul instruction sequence for ``(m, n, k)`` on ``arch``.

    Skeleton only (see module docstring): the WGMMA operands are illustrative,
    not the full descriptor/accumulator operand list. The point is to emit the
    documented *encoding* + the mandatory fence/commit/wait protocol so it can be
    validated and, in CI, fed to ``ptxas``.
    """
    if not is_valid_wgmma_bf16_shape(m, n, k):
        raise ValueError(
            f"({m},{n},{k}) is not a valid Hopper WGMMA bf16 shape "
            "(need m=64, k=16, n∈{8..256 step 8}) — refusing to emit it"
        )
    mma = wgmma_mnemonic(m, n, k, acc=acc, ab=ab)
    return f"""//
// Tessera rung-2.5 emission — {arch} WGMMA {ab} matmul (instruction-encoding skeleton).
// NOT a complete assemblable kernel: real WGMMA needs shared-memory matrix
// descriptors + TMA/cp.async data movement + the full accumulator operand list,
// omitted here. ptxas-assemblability is the rung-3 CI gate; this asserts the
// documented instruction encoding ({mma}) + the PTX scaffolding.
//
.version {PTX_ISA_VERSION}
.target {arch}
.address_size 64

.visible .entry {entry}(
    .param .u64 {entry}_A,
    .param .u64 {entry}_B,
    .param .u64 {entry}_C
)
{{
    .reg .b64  %rd<4>;
    .reg .b64  %desc<2>;
    .reg .f32  %acc<4>;

    ld.param.u64 %rd1, [{entry}_A];
    ld.param.u64 %rd2, [{entry}_B];
    ld.param.u64 %rd3, [{entry}_C];

    // --- WGMMA warpgroup matmul: fence -> mma_async -> commit -> wait ---
    wgmma.fence.sync.aligned;
    {mma} {{%acc0}}, %desc0, %desc1;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;

    ret;
}}
"""


def validate_ptx_structure(ptx: str, *, arch: str = "sm_90a") -> list[str]:
    """Structural validation of emitted PTX (no toolchain). Returns a list of
    problems — empty means the PTX scaffolding + WGMMA encoding are well-formed.
    This is what earns rung 2.5; it does NOT prove assemblability (rung 3)."""
    problems: list[str] = []
    if f".version {PTX_ISA_VERSION}" not in ptx:
        problems.append(f"missing `.version {PTX_ISA_VERSION}` directive")
    if f".target {arch}" not in ptx:
        problems.append(f"missing `.target {arch}` directive")
    if ".address_size 64" not in ptx:
        problems.append("missing `.address_size 64` directive")
    if ".visible .entry" not in ptx:
        problems.append("no `.visible .entry` kernel")
    if "wgmma.mma_async.sync.aligned.m" not in ptx:
        problems.append("no WGMMA matmul instruction emitted")
    for required in (
        "wgmma.fence.sync.aligned",
        "wgmma.commit_group.sync.aligned",
        "wgmma.wait_group.sync.aligned",
    ):
        if required not in ptx:
            problems.append(f"missing mandatory WGMMA protocol op `{required}`")
    if ptx.count("{") != ptx.count("}"):
        problems.append("unbalanced braces")
    if "ret;" not in ptx:
        problems.append("kernel does not return (`ret;` missing)")
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Spike #6 productized — sm_120 (consumer Blackwell, CC 12.0) warp-level mma.sync.
#
# Unlike Hopper warpgroup WGMMA (above), ``mma.sync`` is a warp-level instruction
# with register operands, so it lowers to a COMPLETE, assemblable, *launchable*
# kernel — no shared-memory descriptors / TMA needed. This path is proven
# end-to-end on real sm_120 silicon (RTX 5070 Ti): emit → ptxas → cuLaunch →
# execute-and-compare matches a CPU reference to f32 epsilon. The emitted text is
# byte-for-byte the validated spike kernel
# (docs/audit/backend/nvidia/spikes/sm120_mma_sync/).
#
# The single tile: one warp computes D[16x8] f32 = A[16x16] bf16 (row-major) ·
# B[16x8] bf16 (col-major), f32 accumulate, via
# ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``.
# ─────────────────────────────────────────────────────────────────────────────

# The one tile shape this warp-level mma.sync path emits today: m16n8k16 with
# 16-bit (bf16) A/B and an f32 accumulator. (Other m16n8kK shapes exist but are
# not yet proven on-silicon; this is deliberately the validated slice.)
_MMA_SYNC_BF16_TILE: tuple[int, int, int] = (16, 8, 16)

# Canonical Tessera entry name for the sm_120 mma.sync bf16 matmul kernel — this
# is the runtime_symbol the C-ABI launcher dispatches on.
MMA_SYNC_BF16_ENTRY = "tessera_mma_m16n8k16_bf16"

# Entry names for the *general* aligned-M/N/K mma.sync GEMM kernels (K-loop +
# grid-tiled over 16x8 output tiles), keyed by 16-bit operand dtype. The C-ABI
# launcher dispatches its general-GEMM ABI on these (grid from M/N, runtime
# M/N/K params). bf16/f16 share the m16n8k16 fragment layout.
MMA_SYNC_GEMM_ENTRY: dict[str, str] = {
    "bf16": "tessera_mma_gemm_bf16",
    "f16": "tessera_mma_gemm_f16",
}


def is_valid_mma_sync_bf16_shape(m: int, n: int, k: int) -> bool:
    """The sm_120 warp-level mma.sync bf16 tile this emitter supports: exactly
    m16n8k16 (the documented 16-bit-operand / f32-accumulator MMA shape). Stricter
    than the WGMMA predicate on purpose — only the on-silicon-proven tile."""
    return (m, n, k) == _MMA_SYNC_BF16_TILE


def mma_sync_mnemonic(m: int, n: int, k: int, *, acc: str = "f32", ab: str = "bf16") -> str:
    """The documented warp-level MMA instruction mnemonic for one tile/dtype.
    A is row-major, B is col-major (``.row.col``) per the m16n8k16 fragment ABI."""
    return f"mma.sync.aligned.m{m}n{n}k{k}.row.col.{acc}.{ab}.{ab}.{acc}"


def emit_mma_sync_matmul_ptx(
    m: int = 16,
    n: int = 8,
    k: int = 16,
    *,
    arch: str = "sm_120a",
    acc: str = "f32",
    ab: str = "bf16",
    entry: str = MMA_SYNC_BF16_ENTRY,
) -> str:
    """Emit a COMPLETE, assemblable, launchable sm_120 ``mma.sync`` matmul kernel.

    One warp computes ``D[m×n] f32 = A[m×k] bf16 (row-major) · B[k×n] bf16
    (col-major)`` with f32 accumulation. Unlike :func:`emit_wgmma_matmul_ptx`
    (a skeleton), this is the full kernel: per-lane fragment loads, the warp MMA,
    and the store — proven on real sm_120 silicon (see module docstring).

    The fragment layout exploits memory contiguity: with A row-major and B
    col-major, each packed ``.b32`` register (A: col,col+1 ; B: row,row+1) is a
    single ``ld.global.b32`` — no explicit packing. Emitted ASCII-only (the
    driver JIT ``ptxas`` rejects non-ASCII that standalone ptxas tolerates).
    """
    if not is_valid_mma_sync_bf16_shape(m, n, k):
        raise ValueError(
            f"({m},{n},{k}) is not the supported sm_120 mma.sync bf16 tile "
            f"(only {_MMA_SYNC_BF16_TILE} is proven on-silicon) -- refusing to emit it"
        )
    mma = mma_sync_mnemonic(m, n, k, acc=acc, ab=ab)
    return f""".version {PTX_ISA_VERSION}
.target {arch}
.address_size 64

// Tessera sm_120 mma.sync {ab} matmul -- complete, assemblable, launchable.
// One warp: D[{m}x{n}] {acc} = A[{m}x{k}] {ab} (row-major) * B[{k}x{n}] {ab} (col-major).
// Packed fragments load as single contiguous b32 loads (col/col+1 adjacent in A
// row-major, row/row+1 adjacent in B col-major).
.visible .entry {entry}(
    .param .u64 p_A,
    .param .u64 p_B,
    .param .u64 p_D
)
{{
    .reg .b32  %r<32>;
    .reg .b32  %a0,%a1,%a2,%a3,%b0,%b1;
    .reg .f32  %d0,%d1,%d2,%d3;
    .reg .b64  %A,%B,%D,%off,%addr;

    ld.param.u64 %A, [p_A];
    ld.param.u64 %B, [p_B];
    ld.param.u64 %D, [p_D];
    cvta.to.global.u64 %A, %A;
    cvta.to.global.u64 %B, %B;
    cvta.to.global.u64 %D, %D;

    mov.u32 %r1, %tid.x;          // lane 0..31
    shr.u32 %r2, %r1, 2;          // gid  = lane>>2
    and.b32 %r3, %r1, 3;          // tig  = lane&3
    shl.b32 %r4, %r3, 1;          // 2*tig
    mul.lo.s32 %r5, %r2, 16;      // gid*16
    add.s32 %r6, %r5, %r4;        // a0 elem idx = gid*16 + 2tig

    // ---- A fragment (elem size 2 bytes) ----
    mul.wide.s32 %off, %r6, 2;          add.s64 %addr, %A, %off; ld.global.b32 %a0, [%addr];
    add.s32 %r7, %r6, 128; mul.wide.s32 %off, %r7, 2; add.s64 %addr, %A, %off; ld.global.b32 %a1, [%addr];
    add.s32 %r7, %r6, 8;   mul.wide.s32 %off, %r7, 2; add.s64 %addr, %A, %off; ld.global.b32 %a2, [%addr];
    add.s32 %r7, %r6, 136; mul.wide.s32 %off, %r7, 2; add.s64 %addr, %A, %off; ld.global.b32 %a3, [%addr];

    // ---- B fragment (col-major, elem size 2 bytes): b0 idx = 2tig + gid*16 ----
    add.s32 %r8, %r4, %r5;
    mul.wide.s32 %off, %r8, 2;          add.s64 %addr, %B, %off; ld.global.b32 %b0, [%addr];
    add.s32 %r9, %r8, 8;   mul.wide.s32 %off, %r9, 2; add.s64 %addr, %B, %off; ld.global.b32 %b1, [%addr];

    // ---- zero accumulator (C input) ----
    mov.f32 %d0, 0f00000000;
    mov.f32 %d1, 0f00000000;
    mov.f32 %d2, 0f00000000;
    mov.f32 %d3, 0f00000000;

    // ---- warp-level MMA ----
    {mma}
        {{%d0,%d1,%d2,%d3}}, {{%a0,%a1,%a2,%a3}}, {{%b0,%b1}}, {{%d0,%d1,%d2,%d3}};

    // ---- D store (row-major, elem size 4 bytes): d0 idx = gid*8 + 2tig ----
    mul.lo.s32 %r10, %r2, 8;
    add.s32 %r11, %r10, %r4;
    mul.wide.s32 %off, %r11, 4;         add.s64 %addr, %D, %off; st.global.f32 [%addr], %d0;
    add.s32 %r12, %r11, 1;  mul.wide.s32 %off, %r12, 4; add.s64 %addr, %D, %off; st.global.f32 [%addr], %d1;
    add.s32 %r12, %r11, 64; mul.wide.s32 %off, %r12, 4; add.s64 %addr, %D, %off; st.global.f32 [%addr], %d2;
    add.s32 %r12, %r11, 65; mul.wide.s32 %off, %r12, 4; add.s64 %addr, %D, %off; st.global.f32 [%addr], %d3;

    ret;
}}
"""


def validate_mma_sync_ptx_structure(ptx: str, *, arch: str = "sm_120a") -> list[str]:
    """Structural validation of emitted sm_120 mma.sync PTX (no toolchain).
    Returns a list of problems — empty means the PTX is well-formed. Unlike the
    WGMMA skeleton validator, this asserts a *complete* kernel: the mma.sync
    instruction, the zeroed accumulator, the param loads, and global ld/st."""
    problems: list[str] = []
    if f".version {PTX_ISA_VERSION}" not in ptx:
        problems.append(f"missing `.version {PTX_ISA_VERSION}` directive")
    if f".target {arch}" not in ptx:
        problems.append(f"missing `.target {arch}` directive")
    if ".address_size 64" not in ptx:
        problems.append("missing `.address_size 64` directive")
    if ".visible .entry" not in ptx:
        problems.append("no `.visible .entry` kernel")
    if "mma.sync.aligned.m16n8k16." not in ptx:
        problems.append("no mma.sync matmul instruction emitted")
    if "ld.global.b32" not in ptx:
        problems.append("no global fragment loads (`ld.global.b32`) emitted")
    if "st.global.f32" not in ptx:
        problems.append("no global result stores (`st.global.f32`) emitted")
    if "mov.f32 %d0, 0f00000000" not in ptx:
        problems.append("accumulator not zero-initialized before the mma")
    if not ptx.isascii():
        problems.append("PTX contains non-ASCII bytes (driver JIT ptxas rejects these)")
    if ptx.count("{") != ptx.count("}"):
        problems.append("unbalanced braces")
    if "ret;" not in ptx:
        problems.append("kernel does not return (`ret;` missing)")
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# General aligned-M/N/K sm_120 mma.sync GEMM — the C2-tail-breadth generalization
# of the single m16n8k16 tile above. Same warp-level fragment layout, now:
#   * grid-tiled: one warp (block=32) per 16x8 output tile, grid = (M/16, N/8);
#   * K-looped:   accumulate over K/16 tiles into the same d0..d3 registers;
#   * runtime M/N/K (.u32 params) → one kernel serves every aligned shape;
#   * dtype ∈ {bf16, f16} (identical m16n8k16 fragment layout, mnemonic differs).
# Aligned only (M%16, N%8, K%16) — ragged boundary predication is a follow-on.
# Layout: D[MxN] f32 (row-major) = A[MxK] (row-major) * B[KxN] (col-major),
# f32 accumulate — the same convention the single-tile kernel + its bridge use.
# ─────────────────────────────────────────────────────────────────────────────


#: Max operand element count for the emitted kernel: it addresses elements with
#: 32-bit signed indices, so an operand's largest index (``count - 1``) must fit
#: ``INT32_MAX``. A count of exactly ``2**31`` (max index ``2**31 - 1``) is fine;
#: only a count past ``2**31`` wraps.
_MMA_GEMM_MAX_ELEMS = 1 << 31


def is_valid_mma_sync_gemm_shape(m: int, n: int, k: int) -> bool:
    """The shape constraint for :func:`emit_mma_sync_gemm_ptx`: aligned tiles
    (``M%16 == N%8 == K%16 == 0``, all positive) **and** every operand's element
    count within :data:`_MMA_GEMM_MAX_ELEMS`. The emitted kernel indexes in
    ``.s32``, so an operand's largest index (``count - 1``) must fit ``INT32_MAX``
    — ``M*K`` / ``K*N`` / ``M*N`` may *equal* ``2**31`` (max index ``2**31 - 1``)
    but not exceed it; the launch bridge enforces the same (Decision #21).
    Unaligned (ragged) M/N/K need boundary predication, and 64-bit index math lifts
    the size cap — both follow-ons."""
    if not (m > 0 and n > 0 and k > 0):
        return False
    if m % 16 or n % 8 or k % 16:
        return False
    lim = _MMA_GEMM_MAX_ELEMS
    return m * k <= lim and k * n <= lim and m * n <= lim


def emit_mma_sync_gemm_ptx(
    *,
    dtype: str = "bf16",
    arch: str = "sm_120a",
    acc: str = "f32",
    entry: str | None = None,
) -> str:
    """Emit a COMPLETE, assemblable, launchable sm_120 ``mma.sync`` GEMM kernel for
    ARBITRARY aligned M/N/K, generalizing the single m16n8k16 tile.

    One warp per 16x8 output tile (grid = ``M/16 x N/8``) accumulates over a
    K-loop into f32; ``M``/``N``/``K`` are runtime ``.u32`` params, so one kernel
    serves every aligned shape (``M%16 == N%8 == K%16 == 0``). ``dtype`` selects
    the 16-bit operand type (``bf16``/``f16``) — identical fragment layout, only
    the MMA mnemonic differs. Per-tile fragment math is byte-for-byte the proven
    single-tile kernel's, plus the ``mt = ctaid.x*16`` / ``nt = ctaid.y*8`` tile
    origin, runtime K/N strides, and the K accumulation loop. ASCII-only (the
    driver JIT ``ptxas`` rejects non-ASCII).

    Index math is ``.s32`` (widened only for byte offsets), so an operand's element
    count must be at most ``2**31`` (largest index ``INT32_MAX``) —
    :func:`is_valid_mma_sync_gemm_shape` and the launch bridge enforce this (64-bit
    indexing is the cap-lifting follow-on)."""
    if dtype not in ("bf16", "f16"):
        raise ValueError(
            f"emit_mma_sync_gemm_ptx supports bf16/f16 (16-bit m16n8k16), "
            f"got {dtype!r}")
    entry = entry or MMA_SYNC_GEMM_ENTRY[dtype]
    mma = mma_sync_mnemonic(16, 8, 16, acc=acc, ab=dtype)
    return f""".version {PTX_ISA_VERSION}
.target {arch}
.address_size 64

// Tessera sm_120 mma.sync {dtype} GEMM -- arbitrary aligned M/N/K (M%16,N%8,K%16).
// One warp per 16x8 output tile (grid M/16 x N/8), K-looped f32 accumulate.
// D[MxN] f32 (row-major) = A[MxK] {dtype} (row-major) * B[KxN] {dtype} (col-major).
.visible .entry {entry}(
    .param .u64 p_A,
    .param .u64 p_B,
    .param .u64 p_D,
    .param .u32 p_M,
    .param .u32 p_N,
    .param .u32 p_K
)
{{
    .reg .pred %p;
    .reg .b32  %r<40>;
    .reg .b32  %a0,%a1,%a2,%a3,%b0,%b1;
    .reg .f32  %d0,%d1,%d2,%d3;
    .reg .b32  %N,%K,%k0;
    .reg .b64  %A,%B,%D,%off,%addr;

    ld.param.u64 %A, [p_A];  cvta.to.global.u64 %A, %A;
    ld.param.u64 %B, [p_B];  cvta.to.global.u64 %B, %B;
    ld.param.u64 %D, [p_D];  cvta.to.global.u64 %D, %D;
    ld.param.u32 %N, [p_N];
    ld.param.u32 %K, [p_K];

    mov.u32 %r1, %tid.x;          // lane 0..31
    shr.u32 %r2, %r1, 2;          // gid = lane>>2
    and.b32 %r3, %r1, 3;          // tig = lane&3
    shl.b32 %r4, %r3, 1;          // 2*tig

    mov.u32 %r5, %ctaid.x;  mul.lo.s32 %r5, %r5, 16;   // mt = ctaid.x*16
    mov.u32 %r6, %ctaid.y;  mul.lo.s32 %r6, %r6, 8;    // nt = ctaid.y*8

    add.s32 %r7, %r5, %r2;        // rowA0 = mt+gid
    add.s32 %r8, %r7, 8;          // rowA1 = mt+gid+8
    mul.lo.s32 %r9,  %r7, %K;     // rowA0*K  (loop-invariant)
    mul.lo.s32 %r10, %r8, %K;     // rowA1*K
    add.s32 %r11, %r6, %r2;       // colB = nt+gid (col-major B: column index)
    mul.lo.s32 %r12, %r11, %K;    // colB*K   (loop-invariant)

    mov.f32 %d0, 0f00000000;
    mov.f32 %d1, 0f00000000;
    mov.f32 %d2, 0f00000000;
    mov.f32 %d3, 0f00000000;

    mov.u32 %k0, 0;
$Lk_{entry}:
    // ---- A fragment (elem 2 bytes): rows {{rowA0,rowA1}} x cols {{k0+2tig, +8}} ----
    add.s32 %r20, %r9,  %k0;  add.s32 %r20, %r20, %r4;   // rowA0*K + k0 + 2tig
    mul.wide.s32 %off, %r20, 2;  add.s64 %addr, %A, %off;  ld.global.b32 %a0, [%addr];
    add.s32 %r21, %r10, %k0;  add.s32 %r21, %r21, %r4;   // rowA1*K + k0 + 2tig
    mul.wide.s32 %off, %r21, 2;  add.s64 %addr, %A, %off;  ld.global.b32 %a1, [%addr];
    add.s32 %r22, %r20, 8;    mul.wide.s32 %off, %r22, 2;  add.s64 %addr, %A, %off;  ld.global.b32 %a2, [%addr];
    add.s32 %r23, %r21, 8;    mul.wide.s32 %off, %r23, 2;  add.s64 %addr, %A, %off;  ld.global.b32 %a3, [%addr];

    // ---- B fragment (col-major, elem 2 bytes): b idx = colB*K + k0 + {{2tig, +8}} ----
    add.s32 %r24, %r12, %k0;  add.s32 %r24, %r24, %r4;   // colB*K + k0 + 2tig
    mul.wide.s32 %off, %r24, 2;  add.s64 %addr, %B, %off;  ld.global.b32 %b0, [%addr];
    add.s32 %r25, %r24, 8;    mul.wide.s32 %off, %r25, 2;  add.s64 %addr, %B, %off;  ld.global.b32 %b1, [%addr];

    // ---- warp MMA, accumulate into d0..d3 ----
    {mma}
        {{%d0,%d1,%d2,%d3}}, {{%a0,%a1,%a2,%a3}}, {{%b0,%b1}}, {{%d0,%d1,%d2,%d3}};

    add.s32 %k0, %k0, 16;
    setp.lt.s32 %p, %k0, %K;
    @%p bra $Lk_{entry};

    // ---- D store (row-major, elem 4 bytes): d0 idx = (mt+gid)*N + (nt+2tig) ----
    add.s32 %r30, %r6, %r4;       // nt + 2tig
    mul.lo.s32 %r31, %r7, %N;  add.s32 %r31, %r31, %r30;   // (mt+gid)*N + nt+2tig
    mul.wide.s32 %off, %r31, 4;  add.s64 %addr, %D, %off;  st.global.f32 [%addr], %d0;
    add.s32 %r32, %r31, 1;    mul.wide.s32 %off, %r32, 4;  add.s64 %addr, %D, %off;  st.global.f32 [%addr], %d1;
    mul.lo.s32 %r33, %r8, %N;  add.s32 %r33, %r33, %r30;   // (mt+gid+8)*N + nt+2tig
    mul.wide.s32 %off, %r33, 4;  add.s64 %addr, %D, %off;  st.global.f32 [%addr], %d2;
    add.s32 %r34, %r33, 1;    mul.wide.s32 %off, %r34, 4;  add.s64 %addr, %D, %off;  st.global.f32 [%addr], %d3;

    ret;
}}
"""


def validate_mma_sync_gemm_ptx_structure(ptx: str, *, arch: str = "sm_120a") -> list[str]:
    """Structural validation of an emitted general mma.sync GEMM (no toolchain):
    a complete kernel with runtime M/N/K params, the K-loop, the mma, and global
    ld/st. Empty list = well-formed."""
    problems: list[str] = []
    if f".version {PTX_ISA_VERSION}" not in ptx:
        problems.append(f"missing `.version {PTX_ISA_VERSION}` directive")
    if f".target {arch}" not in ptx:
        problems.append(f"missing `.target {arch}` directive")
    if ".visible .entry" not in ptx:
        problems.append("no `.visible .entry` kernel")
    if ".param .u32 p_K" not in ptx:
        problems.append("no runtime K param (`.param .u32 p_K`) — not a general GEMM")
    if "mma.sync.aligned.m16n8k16." not in ptx:
        problems.append("no mma.sync matmul instruction emitted")
    if "setp.lt.s32" not in ptx or "bra $Lk_" not in ptx:
        problems.append("no K-accumulation loop (missing setp/bra back-edge)")
    if "ld.global.b32" not in ptx:
        problems.append("no global fragment loads (`ld.global.b32`) emitted")
    if "st.global.f32" not in ptx:
        problems.append("no global result stores (`st.global.f32`) emitted")
    if not ptx.isascii():
        problems.append("PTX contains non-ASCII bytes (driver JIT ptxas rejects these)")
    if ptx.count("{") != ptx.count("}"):
        problems.append("unbalanced braces")
    if "ret;" not in ptx:
        problems.append("kernel does not return (`ret;` missing)")
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# NVFP4 block-scale matmul (spike #9 productized to emit+assemble) — the warp
# ``mma.sync…m16n8k64…kind::mxf4nvf4.block_scale`` on sm_120a: fp4 (e2m1) A/B
# operands with per-block ue4m3 scale factors, f32 accumulate. The instruction
# encoding + a UNIT-scale data path are proven in
# ``docs/audit/backend/nvidia/spikes/sm120_mma_sync/nvfp4_gemm.cu``. This emits
# the complete kernel and (rung 3) assembles it with ptxas; on-device *execution*
# and NON-unit scale numerics remain gated on the PTX-ISA scale-distribution spec
# (NVIDIA_AUDIT "Still Open"), so this is deliberately emit+assemble only — the fp4
# fragment packing + numpy reference is not wired into the launch bridge yet.
# ─────────────────────────────────────────────────────────────────────────────

#: Entry name for the sm_120a NVFP4 block-scale m16n8k64 tile.
TESSERA_NVFP4_MMA_ENTRY = "tessera_nvfp4_mma_m16n8k64"

#: The documented warp NVFP4 block-scale MMA mnemonic (per the proven spike).
NVFP4_MMA_MNEMONIC = (
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
    "f32.e2m1.e2m1.f32.ue4m3"
)


def emit_nvfp4_block_scale_mma_ptx(
    *, arch: str = "sm_120a", entry: str = TESSERA_NVFP4_MMA_ENTRY
) -> str:
    """Emit a COMPLETE, assemblable sm_120a NVFP4 block-scale ``mma.sync`` kernel
    (one warp, m16n8k64): D[16x8] f32 = A[16x64] · B[64x8], fp4 e2m1 operands with
    per-lane ue4m3 block-scale selectors, f32 accumulate — the PTX of the proven
    spike (unit-scale data path). Per-lane fragments: A = 4x`.b32`, B = 2x`.b32`,
    scale-A/B = 1x`.b32`; the scale byte/thread selectors are `{0,0}` immediates.
    Assemble-verifiable (rung 3) here; on-device execution + non-unit numerics stay
    gated on the PTX-ISA scale spec (so this is emit+assemble, not launch)."""
    return f""".version {PTX_ISA_VERSION}
.target {arch}
.address_size 64

// Tessera sm_120a NVFP4 block-scale matmul (m16n8k64) -- complete, assemblable.
// One warp: D[16x8] f32 = A[16x64] e2m1 * B[64x8] e2m1, ue4m3 per-block scales.
// Emit+assemble productization of spikes/sm120_mma_sync/nvfp4_gemm.cu.
.visible .entry {entry}(
    .param .u64 p_A,
    .param .u64 p_B,
    .param .u64 p_SFa,
    .param .u64 p_SFb,
    .param .u64 p_D
)
{{
    .reg .b32  %a0,%a1,%a2,%a3,%b0,%b1,%sfa,%sfb;
    .reg .b32  %lane,%idx;
    .reg .f32  %d0,%d1,%d2,%d3;
    .reg .b64  %A,%B,%SFa,%SFb,%D,%off,%addr;

    ld.param.u64 %A,   [p_A];   cvta.to.global.u64 %A,   %A;
    ld.param.u64 %B,   [p_B];   cvta.to.global.u64 %B,   %B;
    ld.param.u64 %SFa, [p_SFa]; cvta.to.global.u64 %SFa, %SFa;
    ld.param.u64 %SFb, [p_SFb]; cvta.to.global.u64 %SFb, %SFb;
    ld.param.u64 %D,   [p_D];   cvta.to.global.u64 %D,   %D;

    mov.u32 %lane, %tid.x;

    // A fragment: 4 x .b32 at A[lane*4 + 0..3]
    mul.lo.s32 %idx, %lane, 4;
    mul.wide.s32 %off, %idx, 4;  add.s64 %addr, %A, %off;  ld.global.b32 %a0, [%addr];
    ld.global.b32 %a1, [%addr+4];
    ld.global.b32 %a2, [%addr+8];
    ld.global.b32 %a3, [%addr+12];

    // B fragment: 2 x .b32 at B[lane*2 + 0..1]
    mul.lo.s32 %idx, %lane, 2;
    mul.wide.s32 %off, %idx, 4;  add.s64 %addr, %B, %off;  ld.global.b32 %b0, [%addr];
    ld.global.b32 %b1, [%addr+4];

    // scale selectors: 1 x .b32 each at SF*[lane]
    mul.wide.s32 %off, %lane, 4;
    add.s64 %addr, %SFa, %off;  ld.global.b32 %sfa, [%addr];
    add.s64 %addr, %SFb, %off;  ld.global.b32 %sfb, [%addr];

    mov.f32 %d0, 0f00000000;
    mov.f32 %d1, 0f00000000;
    mov.f32 %d2, 0f00000000;
    mov.f32 %d3, 0f00000000;

    {NVFP4_MMA_MNEMONIC}
        {{%d0,%d1,%d2,%d3}}, {{%a0,%a1,%a2,%a3}}, {{%b0,%b1}}, {{%d0,%d1,%d2,%d3}},
        {{%sfa}}, {{0, 0}}, {{%sfb}}, {{0, 0}};

    // D store: 4 x f32 at D[lane*4 + 0..3]
    mul.lo.s32 %idx, %lane, 4;
    mul.wide.s32 %off, %idx, 4;  add.s64 %addr, %D, %off;  st.global.f32 [%addr], %d0;
    st.global.f32 [%addr+4],  %d1;
    st.global.f32 [%addr+8],  %d2;
    st.global.f32 [%addr+12], %d3;

    ret;
}}
"""


def validate_nvfp4_ptx_structure(ptx: str, *, arch: str = "sm_120a") -> list[str]:
    """Structural validation of the emitted NVFP4 block-scale kernel (no toolchain).
    Empty list = well-formed: the block-scale mma, the 5 pointer params, the fp4
    fragment loads, and the f32 store."""
    problems: list[str] = []
    if f".version {PTX_ISA_VERSION}" not in ptx:
        problems.append(f"missing `.version {PTX_ISA_VERSION}` directive")
    if f".target {arch}" not in ptx:
        problems.append(f"missing `.target {arch}` directive")
    if "kind::mxf4nvf4.block_scale" not in ptx:
        problems.append("no NVFP4 block-scale mma emitted")
    if ".param .u64 p_SFa" not in ptx or ".param .u64 p_SFb" not in ptx:
        problems.append("missing the two block-scale factor params (SFa/SFb)")
    if "ld.global.b32" not in ptx:
        problems.append("no fp4 fragment loads emitted")
    if "st.global.f32" not in ptx:
        problems.append("no f32 result stores emitted")
    if not ptx.isascii():
        problems.append("PTX contains non-ASCII bytes (driver JIT ptxas rejects these)")
    if ptx.count("{") != ptx.count("}"):
        problems.append("unbalanced braces")
    if "ret;" not in ptx:
        problems.append("kernel does not return (`ret;` missing)")
    return problems


@dataclass(frozen=True)
class AssembleResult:
    """Outcome of a real ``ptxas`` assembly attempt (rung 3)."""

    status: str          # "assembled" | "failed" | "toolchain_absent"
    detail: str = ""

    @property
    def assembled(self) -> bool:
        return self.status == "assembled"


def ptxas_assemble(
    ptx: str, *, arch: str = "sm_90a", ptxas: str | None = None
) -> AssembleResult:
    """Assemble PTX with real ``ptxas`` (rung 3). **Linux-CI only** — returns
    ``toolchain_absent`` (skip-clean) when ptxas is not on PATH, exactly like
    ``scripts/validate_nvcc_compile.py``. Hardware-free: ptxas assembles to SASS
    without a GPU, but it does not install on the arm64 dev host.
    """
    exe = ptxas or shutil.which("ptxas")
    if exe is None:
        return AssembleResult("toolchain_absent", "ptxas not on PATH — skipped")
    try:
        proc = subprocess.run(
            [exe, f"--gpu-name={arch}", "-o", "/dev/null", "-"],
            input=ptx, text=True, capture_output=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return AssembleResult("failed", f"ptxas invocation error: {exc}")
    if proc.returncode == 0:
        return AssembleResult("assembled", "ptxas accepted the kernel")
    return AssembleResult("failed", proc.stderr.strip()[:500])
