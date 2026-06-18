"""AMD rung-2.5 + rung-3 — ``llvm.amdgcn.wmma`` LLVM-IR emission for RDNA 3 / 3.5 (gfx11).

The AMD analog of :mod:`tessera.compiler.ptx_emit` (NVIDIA WGMMA PTX) and
:mod:`tessera.compiler.msl_gemm_emit` (Apple simdgroup_matrix), completing the
host-free Stage-A emit set across all three vendors. See
``docs/audit/compiler/STAGE_A_EMIT_PLAN.md`` and
``docs/audit/backend/rocm/STRIX_HALO_EXECUTION_PLAN.md`` (the ``amdgpu``→``rocdl``
emit path this realizes at the LLVM-IR level).

**The advantage AMD has.** Unlike ``ptxas`` (NVIDIA) and ``metal`` (Apple) — both
absent on the arm64 dev Mac — the **LLVM AMDGPU backend ships with Homebrew LLVM
22** here, and ``llc -mcpu=gfx1151`` lowers ``llvm.amdgcn.wmma.*`` to a real
``v_wmma_*`` instruction *on this host*. So this module reaches a **real rung 3**
(emit → ``llc`` → AMDGCN with the documented instruction), not a skip-clean stub.

**Grounded** in the RDNA3.5 ISA §7.9 + Table 33: WMMA is VOP3P, tile **16×16×16**,
combos F16/BF16/IU8/IU4 (**no FP8** — that is gfx1200/RDNA 4); A=SRC0, B=SRC1,
C=SRC2, D=VDST; A/B lanes 0-15 are replicated into 16-31 (wave32); RNE rounding;
inline constants are C-matrix only. Verified intrinsic signature (compiled here):
``<8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16(<16 x half>, <16 x half>,
<8 x float>)`` → ``v_wmma_f32_16x16x16_f16``.

**Honesty ceiling.** This emits one ``wmma`` intrinsic inside a minimal
``amdgpu_kernel`` — it proves Tessera emits the documented RDNA WMMA intrinsic and
that it lowers to the real instruction, but it is *not* a complete tiled GEMM
(operand lane-replication / VGPR layout / the §7.9.1 V_NOP scheduling hazard /
threadgroup tiling are the named next sub-steps). The structural validator earns
rung 2.5; ``llc_assemble`` is the (here-runnable) rung 3.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# RDNA 3.5 (gfx1151) WMMA combos with an fp32 accumulator — the GEMM/attention
# float paths (ISA Table 33). dtype -> (intrinsic suffix, LLVM A/B element type).
# NOTE (verified by llc on this host): the RDNA bf16 intrinsic takes the bf16
# bit-pattern as <16 x i16>, NOT <16 x bfloat> — f16 uses native `half`. The
# real rung-3 (llc) caught this; the host-free validator could not.
_WMMA_F32_INPUT: dict[str, tuple[str, str]] = {
    "f16": ("f16", "half"),
    "fp16": ("f16", "half"),
    "bf16": ("bf16", "i16"),
}

# RDNA 3.5 supports ONLY 16×16×16 (no FP8 / larger-K — those are gfx1200/RDNA 4).
_RDNA35_SHAPE = (16, 16, 16)
# This emitter uses the RDNA3-class wmma intrinsic (gfx11). Verified on this host:
# it lowers on gfx1100/gfx1151 but **"Cannot select" on gfx1200** — RDNA 4 has a
# different gfx12 "v2" wmma intrinsic ABI (extra format/reuse operands, different
# operand packing), grounded separately as a follow-on. So scope to RDNA 3/3.5.
_RDNA3_CLASS_ARCHES: frozenset[str] = frozenset({"gfx1100", "gfx1151"})
# Per-lane operand widths for wave32 16×16×16: A/B = 16 elements, C/D = 8 fp32.
_A_LEN = _B_LEN = 16
_ACC_LEN = 8


def wmma_intrinsic(dtype: str, *, acc: str = "f32") -> str:
    """The documented ``llvm.amdgcn.wmma`` intrinsic for one input dtype +
    accumulator on RDNA 3.5 (16×16×16)."""
    if acc != "f32":
        raise ValueError("this slice emits the fp32-accumulator WMMA paths only")
    if dtype not in _WMMA_F32_INPUT:
        raise ValueError(
            f"unsupported RDNA3.5 WMMA input dtype {dtype!r} "
            f"(fp32-acc paths: {sorted(set(_WMMA_F32_INPUT) - {'fp16'})}); "
            "note RDNA 3.5 has NO FP8 WMMA (that is gfx1200/RDNA 4)")
    suffix, _ = _WMMA_F32_INPUT[dtype]
    return f"llvm.amdgcn.wmma.f32.16x16x16.{suffix}"


def emit_wmma_llvmir(
    dtype: str = "bf16",
    *,
    acc: str = "f32",
    arch: str = "gfx1151",
    entry: str = "tessera_wmma",
) -> str:
    """Emit an LLVM-IR ``amdgpu_kernel`` that issues the documented RDNA WMMA
    intrinsic ``D = wmma(A, B, C)`` for ``dtype`` inputs and an fp32 accumulator.

    Skeleton (see module docstring): a single intrinsic call, operands passed in
    as kernel args. Lowers via ``llc -mcpu=<arch>`` to a real ``v_wmma_*``. Scoped
    to the **RDNA3-class** targets (gfx1100 / gfx1151) — this is the gfx11 wmma
    intrinsic; gfx1200/RDNA 4 needs the gfx12 v2 ABI (a documented follow-on)."""
    if arch not in _RDNA3_CLASS_ARCHES:
        raise ValueError(
            f"arch {arch!r} not supported by this emitter — it uses the RDNA3-class "
            f"gfx11 wmma intrinsic ({sorted(_RDNA3_CLASS_ARCHES)}). RDNA 4 (gfx1200) "
            "uses a different gfx12 'v2' wmma intrinsic ABI; that is a follow-on.")
    intr = wmma_intrinsic(dtype, acc=acc)
    _, elem = _WMMA_F32_INPUT[dtype]
    a_ty = f"<{_A_LEN} x {elem}>"
    b_ty = f"<{_B_LEN} x {elem}>"
    acc_ty = f"<{_ACC_LEN} x float>"
    return f"""; Tessera rung-2.5/3 emission — RDNA WMMA {dtype} (16x16x16, fp32 acc) for {arch}.
; Grounded: RDNA3.5 ISA Table 33 (A=SRC0,B=SRC1,C=SRC2,D=VDST; A/B lanes 0-15
; replicated to 16-31 wave32; RNE; no FP8 on RDNA 3.5). NOT a complete tiled GEMM:
; operand VGPR layout / lane replication / the 7.9.1 V_NOP hazard / tiling are the
; next sub-steps. `llc -mcpu={arch}` lowers the {intr} call to a real v_wmma_*.
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @{entry}(
    ptr addrspace(1) %d, {a_ty} %a, {b_ty} %b, {acc_ty} %c) {{
  %r = call {acc_ty} @{intr}({a_ty} %a, {b_ty} %b, {acc_ty} %c)
  store {acc_ty} %r, ptr addrspace(1) %d
  ret void
}}

declare {acc_ty} @{intr}({a_ty}, {b_ty}, {acc_ty})
"""


def emit_wmma_gemm_llvmir(
    dtype: str = "bf16",
    *,
    acc: str = "f32",
    arch: str = "gfx1151",
    entry: str | None = None,
) -> str:
    """Emit a **K-reduction WMMA GEMM tile** — a step up from the single-intrinsic
    :func:`emit_wmma_llvmir`. ``D[16×16] = Σ_k A[16×K]·B[K×16]`` accumulated across
    ``K``-tiles: an ``<8 x float>`` accumulator carried in a PHI over the K-loop,
    ``<16 x {elem}>`` A/B fragments loaded from global memory addressed by the lane
    id (``llvm.amdgcn.workitem.id.x``), the ``wmma`` intrinsic in the loop body, the
    result stored at the end. Verified to lower via ``llc -mcpu=gfx1151`` to a real
    ``v_wmma_*`` inside an AMDGCN loop.

    **Honesty ceiling.** This grows the *structure* toward a real GEMM (K-loop +
    global I/O + the intrinsic in a loop) and is llc-compilable, but the per-lane
    addressing here is a **simplified documented placeholder**, NOT the exact WMMA
    operand VGPR layout (the RDNA3.5 ISA §7.9 "Matrix Element Storage in VGPRs"
    mapping + A/B lane-0-15→16-31 replication) required for *numerical* correctness
    — that, and threadgroup tiling, are the named next sub-steps and need real
    silicon to verify numerically (rungs 6-7)."""
    if arch not in _RDNA3_CLASS_ARCHES:
        raise ValueError(
            f"arch {arch!r} not supported — RDNA3-class only "
            f"({sorted(_RDNA3_CLASS_ARCHES)}); gfx1200/RDNA 4 is the gfx12 v2 follow-on.")
    intr = wmma_intrinsic(dtype, acc=acc)
    _, elem = _WMMA_F32_INPUT[dtype]
    a_ty = f"<{_A_LEN} x {elem}>"
    acc_ty = f"<{_ACC_LEN} x float>"
    name = entry or f"tessera_wmma_gemm_{dtype}"
    return f"""; Tessera rung-2.5/3 emission — RDNA WMMA {dtype} GEMM tile (16x16x16, fp32 acc) for {arch}.
; D = sum_k A·B over K-tiles. K-loop accumulator PHI; A/B {a_ty} global loads by lane id;
; {intr} in the loop body; result stored at the end. Lowers via `llc -mcpu={arch}` to a real
; v_wmma_* inside an AMDGCN loop. Honesty: the per-lane addressing is a documented placeholder,
; NOT the exact WMMA VGPR layout / lane replication (ISA 7.9) needed for numerical correctness —
; that + tiling are the next sub-steps, verified numerically only on real silicon (rungs 6-7).
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @{name}(
    ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %D, i32 %K) {{
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %lane64 = zext i32 %lane to i64
  br label %kloop
kloop:
  %k    = phi i32 [ 0, %entry ], [ %knext, %kbody ]
  %acc  = phi {acc_ty} [ zeroinitializer, %entry ], [ %accn, %kbody ]
  %done = icmp sge i32 %k, %K
  br i1 %done, label %store, label %kbody
kbody:
  %k64  = sext i32 %k to i64
  %off  = add i64 %k64, %lane64
  %aptr = getelementptr {elem}, ptr addrspace(1) %A, i64 %off
  %a    = load {a_ty}, ptr addrspace(1) %aptr
  %bptr = getelementptr {elem}, ptr addrspace(1) %B, i64 %off
  %b    = load {a_ty}, ptr addrspace(1) %bptr
  %accn = call {acc_ty} @{intr}({a_ty} %a, {a_ty} %b, {acc_ty} %acc)
  %knext = add i32 %k, 16
  br label %kloop
store:
  %dptr = getelementptr float, ptr addrspace(1) %D, i64 %lane64
  store {acc_ty} %acc, ptr addrspace(1) %dptr
  ret void
}}

declare {acc_ty} @{intr}({a_ty}, {a_ty}, {acc_ty})
declare i32 @llvm.amdgcn.workitem.id.x()
"""


def emit_wmma_gemm_layout_llvmir(
    dtype: str = "bf16",
    *,
    acc: str = "f32",
    arch: str = "gfx1151",
    entry: str | None = None,
) -> str:
    """Emit the K-reduction WMMA GEMM **with the ISA §7.9 operand layout** — a step
    up from :func:`emit_wmma_gemm_llvmir`'s placeholder addressing.

    Implements two grounded RDNA3.5 WMMA operand-layout rules:
      * **Lane replication** — wave32 requires the A/B data in lanes 0-15 to be
        replicated into lanes 16-31. Addressing by ``lane & 15`` (= ``lane % 16``)
        gives that for free: lane 16 reads the same fragment as lane 0. Verified —
        the emitted AMDGCN carries ``v_and_b32 v?, 15, v?``.
      * **nt layout** — A row-major (M×K), B **transposed** (N×K), so both the A row
        and the B "row" (= the logical B column) are **contiguous** ``<16 × {elem}>``
        loads (this is MLX's fast ``nt`` path / the standard attention layout).

    **Honesty ceiling.** The lane replication + nt contiguous operand load are real
    and llc-verified; what remains is the exact **D→C output element mapping** (the
    8 fp32 per lane go to specific (m,n) per the ISA D layout — this skeleton stores
    them lane-contiguously) and threadgroup tiling. Numerical correctness needs real
    silicon (rungs 6-7)."""
    if arch not in _RDNA3_CLASS_ARCHES:
        raise ValueError(
            f"arch {arch!r} not supported — RDNA3-class only "
            f"({sorted(_RDNA3_CLASS_ARCHES)}); gfx1200/RDNA 4 is the gfx12 v2 follow-on.")
    intr = wmma_intrinsic(dtype, acc=acc)
    _, elem = _WMMA_F32_INPUT[dtype]
    a_ty = f"<{_A_LEN} x {elem}>"
    acc_ty = f"<{_ACC_LEN} x float>"
    name = entry or f"tessera_wmma_gemm_layout_{dtype}"
    return f"""; Tessera rung-2.5/3 — RDNA WMMA {dtype} GEMM with the ISA 7.9 operand layout ({arch}).
; Lane replication: lanes 0-15 -> 16-31 via (lane & 15), wave32. nt layout: A row-major MxK,
; Bt transposed NxK -> both contiguous {a_ty} loads. Verified: llc emits v_and_b32 _,15,_ + a real
; v_wmma_* in the loop. Remaining: the exact D->C output element mapping (stored lane-contiguous
; here) + tiling; numerical correctness needs silicon (rungs 6-7).
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @{name}(
    ptr addrspace(1) %A, ptr addrspace(1) %Bt, ptr addrspace(1) %D, i32 %K, i32 %N) {{
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %lane16 = and i32 %lane, 15                 ; lanes 16-31 alias 0-15 (replication)
  %row = sext i32 %lane16 to i64
  %Kx = sext i32 %K to i64
  %rowK = mul i64 %row, %Kx
  br label %kloop
kloop:
  %k   = phi i32 [ 0, %entry ], [ %knext, %kbody ]
  %acc = phi {acc_ty} [ zeroinitializer, %entry ], [ %accn, %kbody ]
  %done = icmp sge i32 %k, %K
  br i1 %done, label %store, label %kbody
kbody:
  %k64  = sext i32 %k to i64
  %aidx = add i64 %rowK, %k64                 ; A[row][k0..k0+15] contiguous (row-major)
  %aptr = getelementptr {elem}, ptr addrspace(1) %A, i64 %aidx
  %a    = load {a_ty}, ptr addrspace(1) %aptr
  %bidx = add i64 %rowK, %k64                 ; Bt[col][k0..k0+15] contiguous (B transposed)
  %bptr = getelementptr {elem}, ptr addrspace(1) %Bt, i64 %bidx
  %b    = load {a_ty}, ptr addrspace(1) %bptr
  %accn = call {acc_ty} @{intr}({a_ty} %a, {a_ty} %b, {acc_ty} %acc)
  %knext = add i32 %k, 16
  br label %kloop
store:
  %lo = sext i32 %lane to i64
  %dptr = getelementptr {acc_ty}, ptr addrspace(1) %D, i64 %lo
  store {acc_ty} %acc, ptr addrspace(1) %dptr
  ret void
}}

declare {acc_ty} @{intr}({a_ty}, {a_ty}, {acc_ty})
declare i32 @llvm.amdgcn.workitem.id.x()
"""


def _grounded_d_store_block() -> str:
    """The 8 strided scalar stores of the D→C output element mapping. Grounded from
    the GPUOpen RDNA3 WMMA blog (the RDNA3.5 ISA's referenced source — the spec §7.9
    does NOT tabulate the layout): for wave32 fp32 output, lane ``L`` register
    ``ele`` (0-7) holds ``D[2*ele + L/16][L%16]`` (the blog's ``r = ele*2 + lIdx/16``
    with the fp16-OPSEL packing dropped). ``col = L & 15``, ``row_base = L >> 4``."""
    out = []
    for ele in range(_ACC_LEN):
        out.append(
            f"  %e{ele} = extractelement <{_ACC_LEN} x float> %acc, i32 {ele}\n"
            f"  %row{ele} = add i64 {2 * ele}, %rowbase64\n"
            f"  %ro{ele} = mul i64 %row{ele}, %Nx\n"
            f"  %ra{ele} = add i64 %ro{ele}, %col64\n"
            f"  %p{ele} = getelementptr float, ptr addrspace(1) %D, i64 %ra{ele}\n"
            f"  store float %e{ele}, ptr addrspace(1) %p{ele}")
    return "\n".join(out)


def emit_wmma_gemm_store_llvmir(
    dtype: str = "bf16",
    *,
    acc: str = "f32",
    arch: str = "gfx1151",
    entry: str | None = None,
) -> str:
    """Emit the WMMA GEMM with the operand layout **and** the grounded **D→C output
    element store** — the last piece of the WMMA operand layout.

    Builds on :func:`emit_wmma_gemm_layout_llvmir` (lane replication + nt loads) and
    replaces the lane-contiguous store with the grounded mapping: for wave32 fp32
    output, lane ``L`` register ``ele`` (0-7) → ``D[2*ele + L/16][L % 16]``
    (8 strided scalar stores). Source: the **GPUOpen RDNA3 WMMA blog** — the
    RDNA3.5 ISA §7.9 does *not* tabulate this layout (it points to that blog + the
    AMD Matrix Instruction Calculator; the tabulated "Matrix Element Storage in
    VGPRs" is an RDNA *4* addition, §7.12.2).

    **Honesty ceiling.** The store layout is grounded from the blog and `llc`-
    compiles, but two caveats stand: (1) per the blog, the WMMA **A matrix is
    column-major** (B/C/D row-major) — the A *load* here is still the row-major nt
    convention, a load-side correction to make; (2) `llc` verifies the instruction
    sequence, not the math — *numerical* correctness needs the AMD Matrix
    Instruction Calculator cross-check or real silicon (rungs 6-7)."""
    if arch not in _RDNA3_CLASS_ARCHES:
        raise ValueError(
            f"arch {arch!r} not supported — RDNA3-class only "
            f"({sorted(_RDNA3_CLASS_ARCHES)}); gfx1200/RDNA 4 is the gfx12 v2 follow-on.")
    intr = wmma_intrinsic(dtype, acc=acc)
    _, elem = _WMMA_F32_INPUT[dtype]
    a_ty = f"<{_A_LEN} x {elem}>"
    acc_ty = f"<{_ACC_LEN} x float>"
    name = entry or f"tessera_wmma_gemm_store_{dtype}"
    return f"""; Tessera rung-2.5/3 — RDNA WMMA {dtype} GEMM, operand layout + grounded D->C store ({arch}).
; Lane replication (lane & 15) + nt contiguous loads + the D->C output mapping:
; wave32 fp32 lane L reg ele(0-7) -> D[2*ele + L/16][L%16] (GPUOpen RDNA3 WMMA blog;
; the RDNA3.5 ISA 7.9 does NOT tabulate this). Caveats: A is column-major per the blog
; (load still nt row-major = load-side TODO); numerical correctness needs silicon (rungs 6-7).
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @{name}(
    ptr addrspace(1) %A, ptr addrspace(1) %Bt, ptr addrspace(1) %D, i32 %K, i32 %N) {{
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %lane16 = and i32 %lane, 15                 ; lanes 16-31 alias 0-15 (replication)
  %row = sext i32 %lane16 to i64
  %Kx = sext i32 %K to i64
  %rowK = mul i64 %row, %Kx
  br label %kloop
kloop:
  %k   = phi i32 [ 0, %entry ], [ %knext, %kbody ]
  %acc = phi {acc_ty} [ zeroinitializer, %entry ], [ %accn, %kbody ]
  %done = icmp sge i32 %k, %K
  br i1 %done, label %store, label %kbody
kbody:
  %k64  = sext i32 %k to i64
  %aidx = add i64 %rowK, %k64
  %aptr = getelementptr {elem}, ptr addrspace(1) %A, i64 %aidx
  %a    = load {a_ty}, ptr addrspace(1) %aptr
  %bptr = getelementptr {elem}, ptr addrspace(1) %Bt, i64 %aidx
  %b    = load {a_ty}, ptr addrspace(1) %bptr
  %accn = call {acc_ty} @{intr}({a_ty} %a, {a_ty} %b, {acc_ty} %acc)
  %knext = add i32 %k, 16
  br label %kloop
store:
  %col = and i32 %lane, 15                     ; output column = lane % 16
  %col64 = zext i32 %col to i64
  %rowbase = lshr i32 %lane, 4                 ; lane / 16  (lower/upper row half)
  %rowbase64 = zext i32 %rowbase to i64
  %Nx = sext i32 %N to i64
{_grounded_d_store_block()}
  ret void
}}

declare {acc_ty} @{intr}({a_ty}, {a_ty}, {acc_ty})
declare i32 @llvm.amdgcn.workitem.id.x()
"""


@dataclass(frozen=True)
class RocdlValidation:
    ok: bool
    reasons: tuple[str, ...]


def validate_wmma_llvmir_structure(
    ir: str, *, dtype: str = "bf16", arch: str = "gfx1151",
) -> RocdlValidation:
    """Host-free rung-2.5 check: the emitted LLVM IR carries the documented RDNA
    WMMA intrinsic with the correct operand vector types, inside an
    ``amdgpu_kernel`` — and (for ``gfx1151``) is FP8-free (RDNA 3.5 has no FP8)."""
    reasons: list[str] = []
    intr = wmma_intrinsic(dtype)
    _, elem = _WMMA_F32_INPUT[dtype]
    if f"@{intr}(" not in ir:
        reasons.append(f"missing the {intr} intrinsic call")
    if "amdgpu_kernel" not in ir:
        reasons.append("missing amdgpu_kernel definition")
    if f"<{_A_LEN} x {elem}>" not in ir:
        reasons.append(f"missing the A/B operand type <{_A_LEN} x {elem}>")
    if f"<{_ACC_LEN} x float>" not in ir:
        reasons.append(f"missing the fp32 accumulator type <{_ACC_LEN} x float>")
    if "amdgcn-amd-amdhsa" not in ir:
        reasons.append("missing the amdgcn target triple")
    # RDNA 3.5 has no FP8 WMMA — guard against emitting a gfx1200-only combo for it.
    if arch == "gfx1151" and (".fp8" in ir or ".bf8" in ir or "16x16x128" in ir):
        reasons.append("emitted an FP8/large-K WMMA for gfx1151 (RDNA 3.5 has none)")
    return RocdlValidation(ok=not reasons, reasons=tuple(reasons))


def validate_wmma_gemm_structure(
    ir: str, *, dtype: str = "bf16", arch: str = "gfx1151",
    expect_vector_store: bool = True,
) -> RocdlValidation:
    """Host-free rung-2.5 check for the K-reduction GEMM emit: on top of the base
    intrinsic/type/kernel checks, assert the GEMM structure — a ``<8 x float>``
    K-loop accumulator PHI, two global operand loads, lane-id addressing, the
    intrinsic in the loop, the bound check, and the result store.

    ``expect_vector_store=False`` skips the lane-contiguous ``store <8 x float>``
    assertion — used by the grounded-D-store variant, which replaces that whole-
    vector store with strided per-register scalar stores."""
    base = validate_wmma_llvmir_structure(ir, dtype=dtype, arch=arch)
    reasons = list(base.reasons)
    if f"phi <{_ACC_LEN} x float>" not in ir:
        reasons.append(f"missing the <{_ACC_LEN} x float> K-loop accumulator phi")
    _, elem = _WMMA_F32_INPUT[dtype]
    if ir.count(f"load <{_A_LEN} x {elem}>") < 2:
        reasons.append(f"expected 2 global <{_A_LEN} x {elem}> operand loads (A, B)")
    if "llvm.amdgcn.workitem.id.x" not in ir:
        reasons.append("missing workitem.id.x lane addressing")
    if "icmp sge i32" not in ir:
        reasons.append("missing the K-loop bound check")
    if expect_vector_store and f"store <{_ACC_LEN} x float>" not in ir:
        reasons.append("missing the D-matrix store")
    return RocdlValidation(ok=not reasons, reasons=tuple(reasons))


def validate_wmma_gemm_layout_structure(
    ir: str, *, dtype: str = "bf16", arch: str = "gfx1151",
) -> RocdlValidation:
    """Host-free rung-2.5 check for the operand-layout GEMM: the GEMM checks plus
    the ISA §7.9 operand-layout markers — **lane replication** (``and i32 %lane,
    15`` → lanes 0-15 aliased into 16-31) and the row-major/transposed (nt)
    contiguous operand addressing."""
    base = validate_wmma_gemm_structure(ir, dtype=dtype, arch=arch)
    reasons = list(base.reasons)
    if "and i32 %lane, 15" not in ir:
        reasons.append("missing the lane-replication mask (and i32 %lane, 15)")
    if "%rowK = mul i64 %row" not in ir:
        reasons.append("missing the per-lane row base (nt contiguous operand layout)")
    return RocdlValidation(ok=not reasons, reasons=tuple(reasons))


def validate_wmma_gemm_store_structure(
    ir: str, *, dtype: str = "bf16", arch: str = "gfx1151",
) -> RocdlValidation:
    """Host-free rung-2.5 check for the operand-layout GEMM **with the grounded
    D→C store**: the layout checks (minus the old lane-contiguous ``store <8 x
    float>``) plus the grounded output-element markers — the ``col = lane & 15`` /
    ``row_base = lane >> 4`` decomposition and 8 strided scalar ``store float``
    s (one per accumulator register, the ``D[2*ele + L/16][L%16]`` mapping)."""
    # Reuse the layout markers (lane replication + nt loads) but NOT the whole-
    # vector store, which the grounded mapping replaces with strided scalar stores.
    base = validate_wmma_gemm_structure(ir, dtype=dtype, arch=arch, expect_vector_store=False)
    reasons = list(base.reasons)
    if "and i32 %lane, 15" not in ir:
        reasons.append("missing the lane-replication mask (and i32 %lane, 15)")
    if "lshr i32 %lane, 4" not in ir:
        reasons.append("missing the output row-base decomposition (lane >> 4)")
    n_stores = ir.count("store float %e")
    if n_stores != _ACC_LEN:
        reasons.append(
            f"expected {_ACC_LEN} strided scalar D stores (one per acc reg), found {n_stores}")
    if f"extractelement <{_ACC_LEN} x float>" not in ir:
        reasons.append("missing per-register extractelement from the accumulator")
    return RocdlValidation(ok=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True)
class LlcResult:
    status: str            # "ok" | "failed" | "skipped"
    detail: str
    asm: str = ""
    wmma_instruction: str = ""


def _find_llc() -> str | None:
    for cand in ("/opt/homebrew/opt/llvm/bin/llc", "llc"):
        p = shutil.which(cand) if cand == "llc" else (cand if Path(cand).exists() else None)
        if p:
            return p
    return None


def llc_assemble(ir: str, *, arch: str = "gfx1151") -> LlcResult:
    """Rung 3: lower the emitted LLVM IR to AMDGCN with ``llc -mcpu=<arch>`` and
    confirm a real ``v_wmma_*`` instruction appears.

    **Runs on this host** (Homebrew LLVM 22 has the AMDGPU backend); skip-cleans
    (``status="skipped"``) only if no ``llc`` is found."""
    llc = _find_llc()
    if not llc:
        return LlcResult("skipped", "llc (LLVM AMDGPU backend) not available")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "wmma.ll"
        out = Path(td) / "wmma.s"
        src.write_text(ir)
        try:
            proc = subprocess.run(
                [llc, "-mtriple=amdgcn-amd-amdhsa", f"-mcpu={arch}",
                 "-filetype=asm", str(src), "-o", str(out)],
                capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            return LlcResult("failed", f"llc invocation error: {e}")
        if proc.returncode != 0 or not out.exists():
            return LlcResult("failed", proc.stderr.strip() or "llc returned nonzero")
        asm = out.read_text()
        wmma = next((ln.strip() for ln in asm.splitlines()
                     if "v_wmma" in ln.lower()), "")
        if not wmma:
            return LlcResult("failed", "llc produced no v_wmma_* instruction", asm=asm)
        return LlcResult("ok", f"lowered to AMDGCN for {arch}", asm=asm,
                         wmma_instruction=wmma)


__all__ = [
    "wmma_intrinsic",
    "emit_wmma_llvmir",
    "emit_wmma_gemm_llvmir",
    "emit_wmma_gemm_layout_llvmir",
    "emit_wmma_gemm_store_llvmir",
    "validate_wmma_llvmir_structure",
    "validate_wmma_gemm_structure",
    "validate_wmma_gemm_layout_structure",
    "validate_wmma_gemm_store_structure",
    "llc_assemble",
    "RocdlValidation",
    "LlcResult",
]
