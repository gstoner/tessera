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

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# A legal LLVM identifier for a kernel `entry` name / a gfx target string. Used to
# reject malformed user input before it is interpolated into emitted IR or an `llc`
# argv (argv already blocks shell injection; these catch typos + invalid-IR names).
_ENTRY_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.$]*$")
_LLC_ARCH_RE = re.compile(r"^gfx[0-9a-f]+$")
# Single-wave fragment-grid cap for the threadgroup emitter: 16 fragments × 8 fp32
# accumulator VGPRs = 128 VGPRs, at the wave32 budget. Beyond this the one-wave
# design is nonsensical (the cooperative multi-wave split is the perf follow-on).
_TG_MAX_FRAGMENTS = 16


def _entry_name(entry: str | None, default: str) -> str:
    """Return ``default`` when ``entry`` is None, else ``entry`` after validating it
    is a legal LLVM identifier (so a bad name can't silently emit invalid IR)."""
    if entry is None:
        return default
    if not _ENTRY_NAME_RE.match(entry):
        raise ValueError(
            f"invalid kernel entry name {entry!r} — must match {_ENTRY_NAME_RE.pattern}")
    return entry

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
# it lowers on gfx1100/gfx1151 but **"Cannot select" on gfx1200** — RDNA 4 keeps the
# plain 3-arg (A,B,C) ABI but uses DENSER per-lane fragments (<8 x elem>, no lane
# 0-15 → 16-31 replication) and adds FP8/BF8; see `emit_wmma_rdna4_llvmir`. (The
# *mods/reuse* "v2" ABI with extra immediate operands — `i1 A_mod`/`i16 C_mod`/reuse
# flags — is gfx1250/gfx1251, a LATER arch, NOT RDNA 4. Grounded by `llc` on this
# host: gfx1250/1251 select `wmma.f32.16x16x32.f16`; gfx1200/1201 do not.)
_RDNA3_CLASS_ARCHES: frozenset[str] = frozenset({"gfx1100", "gfx1151"})
# Per-lane operand widths for wave32 16×16×16: A/B = 16 elements, C/D = 8 fp32.
_A_LEN = _B_LEN = 16
_ACC_LEN = 8

# RDNA 4 (gfx1200/gfx1201) WMMA — grounded by `llc` on this host (LLVM 22 AMDGPU):
# the plain (A,B,C) 3-arg ABI is preserved, but A/B fragments are DENSER (<8 x elem>
# — half the width of gfx11's <16 x elem> — because RDNA 4 drops the wave32 lane
# 0-15 → 16-31 duplication). New: native FP8/BF8 (16×16×16). dtype → (intrinsic
# suffix, A/B LLVM vector type, v_wmma_* mnemonic infix).
_RDNA4_CLASS_ARCHES: frozenset[str] = frozenset({"gfx1200", "gfx1201"})
_RDNA4_A_LEN = 8
_RDNA4_INPUT: dict[str, tuple[str, str, str]] = {
    "f16":      ("f16",     f"<{_RDNA4_A_LEN} x half>", "f16"),
    "fp16":     ("f16",     f"<{_RDNA4_A_LEN} x half>", "f16"),
    "bf16":     ("bf16",    f"<{_RDNA4_A_LEN} x i16>",  "bf16"),
    # FP8 (the RDNA 4 unlock): AMD fp8 = e4m3, bf8 = e5m2; A/B byte-packed <2 x i32>.
    "fp8_e4m3": ("fp8.fp8", "<2 x i32>",                "fp8_fp8"),
    "fp8_e5m2": ("bf8.bf8", "<2 x i32>",                "bf8_bf8"),
}

# gfx1250/gfx1251 WMMA — the "v2" mods/reuse ABI (grounded by `llc` on this host,
# LLVM 22 AMDGPU; distinct from RDNA 4). Two differences from every prior AMD WMMA:
#   1. K is DOUBLED — the f16/bf16 tile is 16×16×**32** (A/B = <16 x elem>).
#   2. The intrinsic takes 5 extra IMMEDIATE operands: per-operand negate modifiers
#      (i1 A_mod / i1 B_mod / i16 C_mod) + two operand-reuse flags (i1 a_reuse /
#      i1 b_reuse) — `wmma(i1 A_mod, A, i1 B_mod, B, i16 C_mod, C, i1 a_reuse, i1 b_reuse)`.
# Also bf16 is NATIVE `<16 x bfloat>` here (gfx11/RDNA 4 use the <_ x i16> bit-pattern).
# FP8 on gfx1250 uses a *different* class again (16×16×64/128, ModsC ABI) — scoped
# out of this slice like iu4/iu8 were for RDNA 4; the float paths are grounded here.
_GFX1250_CLASS_ARCHES: frozenset[str] = frozenset({"gfx1250", "gfx1251"})
_GFX1250_A_LEN = 16
_GFX1250_SHAPE = (16, 16, 32)
_GFX1250_INPUT: dict[str, tuple[str, str, str]] = {
    "f16":  ("f16",  f"<{_GFX1250_A_LEN} x half>",   "f16"),
    "fp16": ("f16",  f"<{_GFX1250_A_LEN} x half>",   "f16"),
    "bf16": ("bf16", f"<{_GFX1250_A_LEN} x bfloat>", "bf16"),  # NATIVE bfloat (not i16)
}


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


def wmma_intrinsic_rdna4(dtype: str) -> str:
    """The documented ``llvm.amdgcn.wmma`` intrinsic for one input dtype on **RDNA 4**
    (gfx1200/gfx1201), fp32 accumulator, 16×16×16. Same intrinsic *names* as gfx11
    for f16/bf16, plus the new fp8/bf8 forms; the difference is the denser operand
    vector width (see :data:`_RDNA4_INPUT`), not the intrinsic name."""
    if dtype not in _RDNA4_INPUT:
        raise ValueError(
            f"unsupported RDNA 4 WMMA input dtype {dtype!r} "
            f"(supported: {sorted(_RDNA4_INPUT)})")
    suffix, _, _ = _RDNA4_INPUT[dtype]
    return f"llvm.amdgcn.wmma.f32.16x16x16.{suffix}"


def emit_wmma_rdna4_llvmir(
    dtype: str = "f16", *, arch: str = "gfx1200", entry: str | None = None,
) -> str:
    """Emit the **RDNA 4** (gfx1200/gfx1201) WMMA intrinsic — the gfx12 path, grounded
    by `llc` on this host (LLVM 22 AMDGPU backend).

    Establishes the RDNA 4 ABI the way :func:`emit_wmma_llvmir` did for gfx11: a
    single ``wmma`` in a minimal ``amdgpu_kernel``. The RDNA 4 differences from gfx11,
    all `llc`-verified:
      * **plain 3-arg ABI preserved** — ``wmma(A, B, C)``; RDNA 4 does *not* take the
        mods/reuse immediate operands (those are gfx1250/1251).
      * **denser fragments** — A/B are ``<8 x elem>`` (gfx11 is ``<16 x elem>``):
        RDNA 4 drops the wave32 lane 0-15 → 16-31 duplication.
      * **FP8/BF8** — ``fp8_e4m3``→``fp8.fp8``, ``fp8_e5m2``→``bf8.bf8`` (the RDNA 4
        low-precision unlock; gfx11/RDNA 3.5 have none).

    **Honesty ceiling.** This is the single-intrinsic ABI proof (the gfx11 path's
    starting point). The GEMM / operand-layout / threadgroup generalizations
    (column-major A, D→C store, LDS tiling) are the follow-ons — and RDNA 4's denser
    VGPR layout means its D→C mapping is its *own* grounding job, not a reuse of the
    gfx11 one. `llc` verifies the instruction; numbers wait for silicon (rungs 6-7)."""
    if arch not in _RDNA4_CLASS_ARCHES:
        raise ValueError(
            f"arch {arch!r} not supported by the RDNA 4 emitter "
            f"({sorted(_RDNA4_CLASS_ARCHES)}); gfx1100/gfx1151 use emit_wmma_llvmir "
            "(RDNA 3-class), gfx1250/1251 use the later mods/reuse ABI.")
    intr = wmma_intrinsic_rdna4(dtype)
    _, abty, _ = _RDNA4_INPUT[dtype]
    acc_ty = f"<{_ACC_LEN} x float>"
    name = _entry_name(entry, f"tessera_wmma_rdna4_{dtype}")
    return f"""; Tessera rung-2.5/3 — RDNA 4 WMMA {dtype} ({arch}). Plain (A,B,C) ABI, DENSER
; {abty} fragments (no wave32 lane replication), fp32 accumulator. fp8/bf8 = RDNA 4 unlock.
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @{name}(ptr addrspace(1) %A, ptr addrspace(1) %D) {{
entry:
  %a = load {abty}, ptr addrspace(1) %A
  %c = call {acc_ty} @{intr}({abty} %a, {abty} %a, {acc_ty} zeroinitializer)
  store {acc_ty} %c, ptr addrspace(1) %D
  ret void
}}

declare {acc_ty} @{intr}({abty}, {abty}, {acc_ty})
"""


def wmma_intrinsic_gfx1250(dtype: str) -> str:
    """The documented ``llvm.amdgcn.wmma`` intrinsic for one input dtype on the
    **gfx1250/gfx1251** v2 ABI (fp32 accumulator, 16×16×**32**)."""
    if dtype not in _GFX1250_INPUT:
        raise ValueError(
            f"unsupported gfx1250 WMMA input dtype {dtype!r} "
            f"(float paths: {sorted(_GFX1250_INPUT)}); FP8 (16×16×64/128, ModsC ABI) "
            "is a documented follow-on")
    suffix, _, _ = _GFX1250_INPUT[dtype]
    return f"llvm.amdgcn.wmma.f32.16x16x32.{suffix}"


def emit_wmma_gfx1250_llvmir(
    dtype: str = "bf16", *, arch: str = "gfx1250", entry: str | None = None,
) -> str:
    """Emit the **gfx1250/gfx1251** WMMA intrinsic — the "v2" mods/reuse ABI,
    grounded by `llc` on this host (LLVM 22 AMDGPU).

    Differs from every prior AMD WMMA slice (gfx11 / RDNA 4), all `llc`-verified:
      * **K is doubled** — the f16/bf16 tile is 16×16×**32** (A/B = ``<16 x elem>``).
      * **5 extra immediate operands** — the call is ``wmma(i1 A_mod, A, i1 B_mod, B,
        i16 C_mod, C, i1 a_reuse, i1 b_reuse)``: per-operand negate modifiers + two
        operand-reuse flags. (All passed as ``0`` here — the plain product. They are
        ``ImmArg`` in the intrinsic; non-constant values would fail to select.)
      * **native ``bfloat``** — bf16 is ``<16 x bfloat>``, not the ``<_ x i16>``
        bit-pattern gfx11/RDNA 4 require.

    **Honesty ceiling.** Single-intrinsic ABI proof (the gfx11 path's starting
    point); the GEMM/operand-layout/threadgroup generalizations are follow-ons, and
    gfx1250's own VGPR layout means its D→C mapping is its own grounding job. FP8
    (16×16×64/128, the ModsC ABI) is scoped out of this slice. `llc` verifies the
    instruction; numbers wait for silicon (rungs 6-7)."""
    if arch not in _GFX1250_CLASS_ARCHES:
        raise ValueError(
            f"arch {arch!r} not supported by the gfx1250 emitter "
            f"({sorted(_GFX1250_CLASS_ARCHES)}); gfx1100/gfx1151 use emit_wmma_llvmir, "
            "gfx1200/gfx1201 use emit_wmma_rdna4_llvmir.")
    intr = wmma_intrinsic_gfx1250(dtype)
    _, abty, _ = _GFX1250_INPUT[dtype]
    acc_ty = f"<{_ACC_LEN} x float>"
    name = _entry_name(entry, f"tessera_wmma_gfx1250_{dtype}")
    return f"""; Tessera rung-2.5/3 — gfx1250/1251 WMMA {dtype} ({arch}). v2 mods/reuse ABI:
; wmma(i1 A_mod, A, i1 B_mod, B, i16 C_mod, C, i1 a_reuse, i1 b_reuse); 16x16x32 (K doubled);
; native bfloat; fp32 accumulator. Modifiers/reuse flags all 0 = the plain product.
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @{name}(ptr addrspace(1) %A, ptr addrspace(1) %D) {{
entry:
  %a = load {abty}, ptr addrspace(1) %A
  %c = call {acc_ty} @{intr}(
      i1 0, {abty} %a, i1 0, {abty} %a, i16 0, {acc_ty} zeroinitializer, i1 0, i1 0)
  store {acc_ty} %c, ptr addrspace(1) %D
  ret void
}}

declare {acc_ty} @{intr}(i1, {abty}, i1, {abty}, i16, {acc_ty}, i1, i1)
"""


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
    name = _entry_name(entry, "tessera_wmma")
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

define amdgpu_kernel void @{name}(
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
    name = _entry_name(entry, f"tessera_wmma_gemm_{dtype}")
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
    name = _entry_name(entry, f"tessera_wmma_gemm_layout_{dtype}")
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
    """Emit the WMMA GEMM with the **complete operand layout** — lane replication,
    the **column-major A load**, the nt B load, **and** the grounded **D→C output
    element store**. The most-complete RDNA3-class WMMA GEMM emit.

    Three grounded layout rules, all from the **GPUOpen RDNA3 WMMA blog** (the
    RDNA3.5 ISA §7.9 references the blog + the AMD Matrix Instruction Calculator
    rather than tabulating the layout; the tabulated "Matrix Element Storage in
    VGPRs" is an RDNA *4* §7.12.2 addition):

      * **A is column-major** — ``a_frag[ele] = a[16*lane + ele]``: the lane selects
        the K-column, and the contiguous 16-element run walks the A-tile rows
        (``ele`` = output row m). Generalized across K-tiles, column ``= k0 + lane``
        and the column-major base ``= (k0+lane)*16`` (leading dim = the 16 A-tile
        rows). This *corrects* the earlier row-major A load.
      * **B is row-major** — the blog form is ``b[16*ele + lane]`` (strided). Here B
        is supplied pre-transposed (``Bt``, N×K) so the per-lane load is the
        equivalent **contiguous** ``nt`` form (lane = N column, contiguous over K) —
        a perf layout, same operand values.
      * **D→C output store** — wave32 fp32 lane ``L`` register ``ele`` (0-7) →
        ``D[2*ele + L/16][L % 16]`` (8 strided scalar stores).

    **Honesty ceiling.** The full layout is grounded from the blog and `llc`-
    compiles to a real ``v_wmma_*`` + ``v_and_b32`` + strided ``global_store``, but
    `llc` verifies the instruction sequence, not the math — *numerical* correctness
    needs the AMD Matrix Instruction Calculator cross-check or real silicon
    (rungs 6-7)."""
    if arch not in _RDNA3_CLASS_ARCHES:
        raise ValueError(
            f"arch {arch!r} not supported — RDNA3-class only "
            f"({sorted(_RDNA3_CLASS_ARCHES)}); gfx1200/RDNA 4 is the gfx12 v2 follow-on.")
    intr = wmma_intrinsic(dtype, acc=acc)
    _, elem = _WMMA_F32_INPUT[dtype]
    a_ty = f"<{_A_LEN} x {elem}>"
    acc_ty = f"<{_ACC_LEN} x float>"
    name = _entry_name(entry, f"tessera_wmma_gemm_store_{dtype}")
    return f"""; Tessera rung-2.5/3 — RDNA WMMA {dtype} GEMM, complete operand layout ({arch}).
; Lane replication (lane & 15) + COLUMN-MAJOR A load (a[16*lane+ele]) + nt B load +
; the D->C output mapping (wave32 fp32 lane L reg ele(0-7) -> D[2*ele + L/16][L%16]).
; All three grounded in the GPUOpen RDNA3 WMMA blog (the RDNA3.5 ISA 7.9 references it
; rather than tabulating). numerical correctness needs silicon (rungs 6-7).
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
  ; A: COLUMN-MAJOR (blog: a_frag[ele] = a[16*lane + ele]). lane selects the K-column;
  ; the contiguous 16-run walks the A-tile rows (ele = m). col = k0+lane, base = col*16.
  %kcol = add i32 %k, %lane16                 ; A column index = k0 + lane
  %kcol64 = sext i32 %kcol to i64
  %acolbase = mul i64 %kcol64, 16             ; column-major leading dim = 16 A-tile rows
  %aptr = getelementptr {elem}, ptr addrspace(1) %A, i64 %acolbase
  %a    = load {a_ty}, ptr addrspace(1) %aptr ; A[0..15][kcol] contiguous (column-major)
  ; B: nt pre-transposed (N x K), contiguous over K — perf form of the blog's row-major B.
  %bidx = add i64 %rowK, %k64                 ; Bt[lane][k0..k0+15] contiguous
  %bptr = getelementptr {elem}, ptr addrspace(1) %Bt, i64 %bidx
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


def _tg_fragment_store(i: int, j: int, acc_ssa: str) -> str:
    """The grounded D→C store for output fragment ``(i, j)`` of a threadgroup tile.
    Reuses the per-lane mapping ``D[2*ele + L/16][L%16]`` (GPUOpen RDNA3 blog) with
    the fragment's row/col block offset (``16*i`` rows, ``16*j`` cols) folded in."""
    tag = f"{i}_{j}"
    lines = [
        f"  ; --- store fragment ({i},{j}): D rows [16*{i}..], cols [16*{j}..] ---",
        f"  %fcol{tag} = add i64 %col64, {16 * j}",
        f"  %frow0{tag} = add i64 %rowbase64, {16 * i}",
    ]
    for ele in range(_ACC_LEN):
        lines.append(
            f"  %fe{tag}_{ele} = extractelement <{_ACC_LEN} x float> {acc_ssa}, i32 {ele}\n"
            f"  %frow{tag}_{ele} = add i64 {2 * ele}, %frow0{tag}\n"
            f"  %fro{tag}_{ele} = mul i64 %frow{tag}_{ele}, %Nx\n"
            f"  %fra{tag}_{ele} = add i64 %fro{tag}_{ele}, %fcol{tag}\n"
            f"  %fp{tag}_{ele} = getelementptr float, ptr addrspace(1) %D, i64 %fra{tag}_{ele}\n"
            f"  store float %fe{tag}_{ele}, ptr addrspace(1) %fp{tag}_{ele}")
    return "\n".join(lines)


def emit_wmma_gemm_threadgroup_llvmir(
    dtype: str = "bf16",
    *,
    mf: int = 2,
    nf: int = 2,
    acc: str = "f32",
    arch: str = "gfx1151",
    entry: str | None = None,
) -> str:
    """Emit a **threadgroup-tiled** WMMA GEMM — the AMD analog of the Apple "steel"
    structure. One workgroup computes a ``BM×BN`` output tile (an ``mf×nf`` grid of
    16×16 WMMA fragments) over a ``BK``-deep (16-wide) K-loop, staging the A and B
    tiles through **LDS** (``addrspace(3)``) with a workgroup ``s_barrier`` between
    the cooperative load and the matrix multiply.

    Structure (all llc-verifiable on gfx1151):
      * **LDS staging** — module-level ``addrspace(3)`` globals hold the A tile
        (``BM×16``) and B tile (``16×BN``) for the current K-step.
      * **cooperative load → barrier** — each K-step loads the column-major A
        fragments + nt B fragments from global into LDS, then ``llvm.amdgcn.s.barrier``
        before the fragments are read back (the threadgroup-synchronization point).
      * **mf×nf fragment grid** — ``mf*nf`` ``<8 x float>`` accumulator PHIs, one
        per output fragment, each updated by a ``wmma`` reading its A/B fragment
        from LDS; a second barrier guards LDS reuse before the next K-step.
      * **grounded fragment stores** — each fragment writes back with the
        ``D[2*ele + L/16][L%16]`` mapping plus its ``(16*i, 16*j)`` block offset.

    **Honesty ceiling.** This is the threadgroup *structure* — LDS staging, the
    double barrier, the fragment grid, the K-loop — with **one wave owning the whole
    ``mf×nf`` grid** (sequential WMMAs). Cooperative multi-wave fragment distribution
    + coalesced/vectorized LDS loads + double-buffering are the perf follow-ons. The
    operand addressing is the A1 column-major-A / nt-B grounding; `llc` verifies the
    instruction sequence, not the math (rungs 6-7 for numbers)."""
    if arch not in _RDNA3_CLASS_ARCHES:
        raise ValueError(
            f"arch {arch!r} not supported — RDNA3-class only "
            f"({sorted(_RDNA3_CLASS_ARCHES)}); gfx1200/RDNA 4 is the gfx12 v2 follow-on.")
    if mf < 1 or nf < 1:
        raise ValueError(f"mf/nf must be >= 1 (got mf={mf}, nf={nf})")
    if mf * nf > _TG_MAX_FRAGMENTS:
        raise ValueError(
            f"mf*nf = {mf * nf} exceeds the single-wave cap {_TG_MAX_FRAGMENTS} "
            f"({_TG_MAX_FRAGMENTS} frags × 8 fp32 acc VGPRs = 128, the wave32 budget); "
            "larger tiles need the cooperative multi-wave split (a perf follow-on)")
    intr = wmma_intrinsic(dtype, acc=acc)
    _, elem = _WMMA_F32_INPUT[dtype]
    a_ty = f"<{_A_LEN} x {elem}>"
    acc_ty = f"<{_ACC_LEN} x float>"
    bm, bn = 16 * mf, 16 * nf
    name = _entry_name(entry, f"tessera_wmma_gemm_tg_{dtype}_{mf}x{nf}")
    suff = f"{dtype}_{mf}x{nf}"
    nfrag = mf * nf

    # LDS staging: A tile BM×16 (column-major blocks), B tile 16×BN (nt blocks).
    lds = (f"@lds.a.{suff} = internal addrspace(3) global [{bm * 16} x {elem}] undef, align 16\n"
           f"@lds.b.{suff} = internal addrspace(3) global [{16 * bn} x {elem}] undef, align 16")

    # Accumulator PHIs — one per output fragment.
    phis = "\n".join(
        f"  %acc{f} = phi {acc_ty} [ zeroinitializer, %entry ], [ %accn{f}, %kbody ]"
        for f in range(nfrag))

    # LDS slot layout: each (block, lane) owns a NON-OVERLAPPING 16-element fragment.
    # Block b is a 16×16 sub-tile = 256 elements at [256*b, 256*b+256); lane L's
    # 16-element fragment sits at offset 256*b + 16*L. (A flat `getelementptr elem,
    # …, i32 <off>` — NOT a 2-index `<16 x elem>` GEP, which would stride blocks by
    # only 1 element and overlap them.) `%lane_x16` = 16*lane is computed in entry.
    coop = []
    for i in range(mf):
        coop.append(
            f"  ; stage A block {i} (column-major: col=k0+lane, base=col*16, rows 16*{i}..)\n"
            f"  %ablk{i} = add i64 %acolbase, {i * 16}\n"
            f"  %ag{i} = getelementptr {elem}, ptr addrspace(1) %A, i64 %ablk{i}\n"
            f"  %av{i} = load {a_ty}, ptr addrspace(1) %ag{i}\n"
            f"  %aoff{i} = add i32 %lane_x16, {256 * i}\n"
            f"  %as{i} = getelementptr {elem}, ptr addrspace(3) @lds.a.{suff}, i32 %aoff{i}\n"
            f"  store {a_ty} %av{i}, ptr addrspace(3) %as{i}")
    for j in range(nf):
        coop.append(
            f"  ; stage B block {j} (nt: Bt[16*{j}+lane][k0..], contiguous over K)\n"
            f"  %bblk{j} = add i64 %bidx, {j * 16} \n"
            f"  %bg{j} = getelementptr {elem}, ptr addrspace(1) %Bt, i64 %bblk{j}\n"
            f"  %bv{j} = load {a_ty}, ptr addrspace(1) %bg{j}\n"
            f"  %boff{j} = add i32 %lane_x16, {256 * j}\n"
            f"  %bs{j} = getelementptr {elem}, ptr addrspace(3) @lds.b.{suff}, i32 %boff{j}\n"
            f"  store {a_ty} %bv{j}, ptr addrspace(3) %bs{j}")
    coop_block = "\n".join(coop)

    # Fragment multiply: read A block i + B block j back from LDS (reusing the same
    # %aoff/%boff slot offsets from the stage above), wmma into acc.
    mma = []
    for i in range(mf):
        mma.append(
            f"  %la{i} = getelementptr {elem}, ptr addrspace(3) @lds.a.{suff}, i32 %aoff{i}\n"
            f"  %fa{i} = load {a_ty}, ptr addrspace(3) %la{i}")
    for j in range(nf):
        mma.append(
            f"  %lb{j} = getelementptr {elem}, ptr addrspace(3) @lds.b.{suff}, i32 %boff{j}\n"
            f"  %fb{j} = load {a_ty}, ptr addrspace(3) %lb{j}")
    for i in range(mf):
        for j in range(nf):
            f = i * nf + j
            mma.append(
                f"  %accn{f} = call {acc_ty} @{intr}({a_ty} %fa{i}, {a_ty} %fb{j}, {acc_ty} %acc{f})")
    mma_block = "\n".join(mma)

    stores = "\n".join(_tg_fragment_store(i, j, f"%acc{i * nf + j}")
                       for i in range(mf) for j in range(nf))

    return f"""; Tessera rung-2.5/3 — RDNA WMMA {dtype} threadgroup-tiled GEMM, {mf}x{nf} fragment grid ({arch}).
; BM={bm} x BN={bn} output tile via LDS-staged A/B + s_barrier + an {mf}x{nf} WMMA fragment grid over a
; BK-deep (16) K-loop. Column-major A / nt B (A1 grounding). One wave owns the grid (cooperative
; multi-wave distribution is the perf follow-on). numerical correctness needs silicon (rungs 6-7).
target triple = "amdgcn-amd-amdhsa"
{lds}

define amdgpu_kernel void @{name}(
    ptr addrspace(1) %A, ptr addrspace(1) %Bt, ptr addrspace(1) %D, i32 %K, i32 %N) {{
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %lane16 = and i32 %lane, 15                 ; lanes 16-31 alias 0-15 (replication)
  %lane_x16 = mul i32 %lane16, 16             ; per-lane 16-elem fragment offset within an LDS block
  %row = sext i32 %lane16 to i64
  %Kx = sext i32 %K to i64
  %rowK = mul i64 %row, %Kx                   ; B (nt) base: lane * K
  %col64 = zext i32 %lane16 to i64            ; output column = lane % 16
  %rowbase = lshr i32 %lane, 4
  %rowbase64 = zext i32 %rowbase to i64       ; output row half (lane / 16)
  %Nx = sext i32 %N to i64
  br label %kloop
kloop:
  %k = phi i32 [ 0, %entry ], [ %knext, %kbody ]
{phis}
  %done = icmp sge i32 %k, %K
  br i1 %done, label %store, label %kbody
kbody:
  ; --- column-major A column base (col = k0 + lane, base = col*16) ---
  %kcol = add i32 %k, %lane16
  %kcol64 = sext i32 %kcol to i64
  %acolbase = mul i64 %kcol64, 16
  ; --- nt B base (Bt[lane][k0..], contiguous over K) ---
  %k64 = sext i32 %k to i64
  %bidx = add i64 %rowK, %k64
  ; --- cooperative global -> LDS stage ---
{coop_block}
  call void @llvm.amdgcn.s.barrier()          ; threadgroup sync: stage complete
  ; --- {mf}x{nf} fragment grid: read from LDS, wmma-accumulate ---
{mma_block}
  call void @llvm.amdgcn.s.barrier()          ; guard LDS before next K-step overwrites it
  %knext = add i32 %k, 16
  br label %kloop
store:
{stores}
  ret void
}}

declare {acc_ty} @{intr}({a_ty}, {a_ty}, {acc_ty})
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
"""


def emit_dependent_wmma_chain_llvmir(
    dtype: str = "f16",
    *,
    hazard: bool = False,
    depth: int = 3,
    arch: str = "gfx1151",
    entry: str | None = None,
) -> str:
    """Emit a chain of ``depth`` dependent WMMAs to exercise the **RDNA 3.5 §7.9.1
    WMMA scheduling hazard** — a documented, `llc`-reproducible artifact.

    Two modes, both ``depth`` WMMAs deep:
      * ``hazard=False`` (the **GEMM accumulation pattern**): each WMMA feeds its
        result forward only as the **C/D accumulator** (in-place), with independent
        ``SrcA``/``SrcB``. Per §7.9.1 this is **hazard-free** — `llc` schedules it
        with ``s_delay_alu`` and **no** ``v_nop``. This is what every Tessera WMMA
        GEMM emits, so the GEMMs need no manual scheduling nop.
      * ``hazard=True``: each WMMA reads the **previous WMMA's destination** as its
        ``SrcA`` (a read-after-write on a matrix *source*). This is the §7.9.1
        hazard — `llc`'s ``GCNHazardRecognizer`` inserts a mandatory ``v_nop``
        between the dependent WMMAs.

    Used by :func:`wmma_scheduling` + the rung-3 tests to **lock both behaviors**:
    the GEMM pattern is hazard-free by construction, and the hazard — when present —
    is handled. The IR-level emit gets the (rare) mandatory nop from the backend for
    free; there is nothing for the emitter to insert by hand."""
    if arch not in _RDNA3_CLASS_ARCHES:
        raise ValueError(
            f"arch {arch!r} not supported — RDNA3-class only "
            f"({sorted(_RDNA3_CLASS_ARCHES)}); gfx1200/RDNA 4 is the gfx12 v2 follow-on.")
    if depth < 2:
        raise ValueError(f"depth must be >= 2 to form a dependent chain (got {depth})")
    intr = wmma_intrinsic(dtype)
    _, elem = _WMMA_F32_INPUT[dtype]
    a_ty = f"<{_A_LEN} x {elem}>"
    acc_ty = f"<{_ACC_LEN} x float>"
    name = _entry_name(entry, f"tessera_wmma_chain_{'hazard' if hazard else 'accum'}_{dtype}")

    body = [f"  %a0 = load {a_ty}, ptr addrspace(1) %A"]
    prev_acc = "zeroinitializer"
    prev_dst: str | None = None
    for n in range(depth):
        if hazard and prev_dst is not None:
            # SrcA reads the PRIOR WMMA's destination — the §7.9.1 RAW-on-source hazard.
            body.append(f"  %a{n} = bitcast {acc_ty} {prev_dst} to {a_ty}")
            src_a, c_in = f"%a{n}", "zeroinitializer"
        else:
            # hazard-free: independent SrcA/SrcB, accumulator (C/D) feedback only.
            src_a, c_in = "%a0", prev_acc
        body.append(
            f"  %c{n} = call {acc_ty} @{intr}({a_ty} {src_a}, {a_ty} %a0, {acc_ty} {c_in})")
        prev_acc, prev_dst = f"%c{n}", f"%c{n}"
    body.append(f"  store {acc_ty} {prev_dst}, ptr addrspace(1) %D")

    mode = ("hazard: SrcA reads the prior WMMA destination (§7.9.1 -> llc inserts v_nop)"
            if hazard else
            "accumulation: C/D feedback only, independent SrcA/B (§7.9.1 hazard-free)")
    return f"""; Tessera rung-3 — RDNA WMMA dependent chain ({arch}), depth {depth}.
; {mode}
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @{name}(ptr addrspace(1) %A, ptr addrspace(1) %D) {{
entry:
{chr(10).join(body)}
  ret void
}}

declare {acc_ty} @{intr}({a_ty}, {a_ty}, {acc_ty})
"""


@dataclass(frozen=True)
class WmmaScheduling:
    """Summary of WMMA scheduling in lowered AMDGCN — the §7.9.1 hazard lens."""
    n_wmma: int
    n_vnop: int
    n_sdelay: int

    @property
    def hazard_free(self) -> bool:
        """True when no ``v_nop`` was needed between the WMMAs (the in-place
        accumulation pattern). A mandatory hazard nop ⇒ ``False``."""
        return self.n_vnop == 0


def wmma_scheduling(asm: str) -> WmmaScheduling:
    """Analyze lowered AMDGCN for the §7.9.1 WMMA scheduling shape: count the
    ``v_wmma_*`` instructions, the ``v_nop`` hazard nops the backend inserted, and
    the ``s_delay_alu`` scheduling hints. ``hazard_free`` ⇔ no ``v_nop`` was
    needed."""
    n_wmma = sum(1 for ln in asm.splitlines() if "v_wmma" in ln.lower())
    n_vnop = sum(1 for ln in asm.splitlines() if ln.strip().startswith("v_nop"))
    n_sdelay = sum(1 for ln in asm.splitlines() if "s_delay_alu" in ln)
    return WmmaScheduling(n_wmma=n_wmma, n_vnop=n_vnop, n_sdelay=n_sdelay)


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


def validate_wmma_rdna4_structure(
    ir: str, *, dtype: str = "f16", arch: str = "gfx1200",
) -> RocdlValidation:
    """Host-free rung-2.5 check for the RDNA 4 emit: the intrinsic call, the
    ``amdgpu_kernel`` def, the fp32 accumulator, the target triple, and the two
    RDNA 4 hallmarks — the **denser** A/B fragment type (``<8 x ...>``, not gfx11's
    ``<16 x ...>``) and the **plain 3-arg ABI** (no gfx1250 ``i16`` C-mod operand)."""
    reasons: list[str] = []
    intr = wmma_intrinsic_rdna4(dtype)
    _, abty, _ = _RDNA4_INPUT[dtype]
    if f"@{intr}(" not in ir:
        reasons.append(f"missing the {intr} intrinsic call")
    if "amdgpu_kernel" not in ir:
        reasons.append("missing amdgpu_kernel definition")
    if abty not in ir:
        reasons.append(f"missing the denser RDNA 4 A/B operand type {abty}")
    if f"<{_A_LEN} x " in ir and dtype in ("f16", "fp16", "bf16"):
        reasons.append(f"emitted the gfx11 <{_A_LEN} x ...> fragment (RDNA 4 is denser <8 x ...>)")
    if f"<{_ACC_LEN} x float>" not in ir:
        reasons.append(f"missing the fp32 accumulator type <{_ACC_LEN} x float>")
    if "amdgcn-amd-amdhsa" not in ir:
        reasons.append("missing the amdgcn target triple")
    # The gfx1250 v2 ABI carries an `i16 0,` C-mod immediate + trailing `i1 0, i1 0)`
    # reuse flags — neither may appear in the RDNA 4 plain 3-arg ABI. (The earlier
    # `"i16," in ir.replace("x i16","")` heuristic was a false-negative: the real
    # operand text is `i16 0,` with a space, so it never matched.)
    if "i16 0," in ir or "i1 0, i1 0)" in ir:
        reasons.append("emitted a gfx1250 mods/reuse operand (RDNA 4 uses the plain 3-arg ABI)")
    return RocdlValidation(ok=not reasons, reasons=tuple(reasons))


def validate_wmma_gfx1250_structure(
    ir: str, *, dtype: str = "f16", arch: str = "gfx1250",
) -> RocdlValidation:
    """Host-free rung-2.5 check for the gfx1250/1251 v2 emit: the intrinsic call, the
    ``amdgpu_kernel`` def, the fp32 accumulator, the target triple, and the v2-ABI
    hallmarks — the **16×16×32** shape (``<16 x elem>`` A/B), the **mods/reuse**
    immediate operands (``i16 0,`` C-mod + the two trailing ``i1 0`` reuse flags),
    and **native ``bfloat``** for bf16."""
    reasons: list[str] = []
    intr = wmma_intrinsic_gfx1250(dtype)
    _, abty, _ = _GFX1250_INPUT[dtype]
    if f"@{intr}(" not in ir:
        reasons.append(f"missing the {intr} intrinsic call")
    if "16x16x32" not in ir:
        reasons.append("missing the gfx1250 16x16x32 (K-doubled) shape")
    if "amdgpu_kernel" not in ir:
        reasons.append("missing amdgpu_kernel definition")
    if abty not in ir:
        reasons.append(f"missing the A/B operand type {abty}")
    if f"<{_ACC_LEN} x float>" not in ir:
        reasons.append(f"missing the fp32 accumulator type <{_ACC_LEN} x float>")
    if "amdgcn-amd-amdhsa" not in ir:
        reasons.append("missing the amdgcn target triple")
    # v2 ABI: the i16 C-mod immediate + the two trailing i1 operand-reuse flags.
    if "i16 0," not in ir:
        reasons.append("missing the v2 i16 C-mod immediate operand")
    if "i1 0, i1 0)" not in ir:
        reasons.append("missing the v2 operand-reuse flags (i1 a_reuse, i1 b_reuse)")
    if dtype == "bf16" and "x i16>" in ir:
        reasons.append("bf16 emitted as <_ x i16> (gfx1250 uses native <16 x bfloat>)")
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
    """Host-free rung-2.5 check for the **complete-layout** GEMM (column-major A +
    nt B + grounded D→C store): the layout checks (minus the old lane-contiguous
    ``store <8 x float>``) plus the column-major A markers (``%kcol = add i32 %k,
    %lane16`` and the column-major ``mul i64 %kcol64, 16``) and the grounded
    output-element markers — the ``col = lane & 15`` / ``row_base = lane >> 4``
    decomposition and 8 strided scalar ``store float``s (the ``D[2*ele +
    L/16][L%16]`` mapping)."""
    # Reuse the layout markers (lane replication + nt loads) but NOT the whole-
    # vector store, which the grounded mapping replaces with strided scalar stores.
    base = validate_wmma_gemm_structure(ir, dtype=dtype, arch=arch, expect_vector_store=False)
    reasons = list(base.reasons)
    if "and i32 %lane, 15" not in ir:
        reasons.append("missing the lane-replication mask (and i32 %lane, 15)")
    if "%kcol = add i32 %k, %lane16" not in ir:
        reasons.append("missing the column-major A column index (k0 + lane)")
    if "mul i64 %kcol64, 16" not in ir:
        reasons.append("missing the column-major A base (col * 16 leading dim)")
    if "lshr i32 %lane, 4" not in ir:
        reasons.append("missing the output row-base decomposition (lane >> 4)")
    n_stores = ir.count("store float %e")
    if n_stores != _ACC_LEN:
        reasons.append(
            f"expected {_ACC_LEN} strided scalar D stores (one per acc reg), found {n_stores}")
    if f"extractelement <{_ACC_LEN} x float>" not in ir:
        reasons.append("missing per-register extractelement from the accumulator")
    return RocdlValidation(ok=not reasons, reasons=tuple(reasons))


def validate_wmma_gemm_threadgroup_structure(
    ir: str, *, dtype: str = "bf16", mf: int = 2, nf: int = 2, arch: str = "gfx1151",
) -> RocdlValidation:
    """Host-free rung-2.5 check for the **threadgroup-tiled** GEMM: the base
    intrinsic/type checks plus the threadgroup structure — **LDS** ``addrspace(3)``
    A/B staging globals, a **double** ``llvm.amdgcn.s.barrier`` (stage-complete +
    LDS-reuse guard), the column-major A base, ``mf*nf`` accumulator PHIs and
    ``mf*nf`` ``wmma`` calls (the fragment grid), and ``mf*nf*8`` strided scalar D
    stores (one grounded mapping per output fragment)."""
    base = validate_wmma_llvmir_structure(ir, dtype=dtype, arch=arch)
    reasons = list(base.reasons)
    nfrag = mf * nf
    intr = wmma_intrinsic(dtype)
    if "addrspace(3) global" not in ir:
        reasons.append("missing the LDS (addrspace(3)) staging globals")
    n_bar = ir.count("@llvm.amdgcn.s.barrier()")
    # 2 barriers in the K-loop body + 1 in the declare line = 3 textual occurrences.
    if n_bar < 3:
        reasons.append(f"expected a double s.barrier (+declare), found {n_bar} occurrences")
    if "mul i64 %kcol64, 16" not in ir:
        reasons.append("missing the column-major A base (col * 16 leading dim)")
    n_phi = ir.count(f"phi <{_ACC_LEN} x float>")
    if n_phi != nfrag:
        reasons.append(f"expected {nfrag} accumulator PHIs (mf*nf), found {n_phi}")
    n_wmma = ir.count(f"call <{_ACC_LEN} x float> @{intr}(")
    if n_wmma != nfrag:
        reasons.append(f"expected {nfrag} wmma calls (mf*nf fragment grid), found {n_wmma}")
    n_stores = ir.count("store float %fe")
    if n_stores != nfrag * _ACC_LEN:
        reasons.append(
            f"expected {nfrag * _ACC_LEN} fragment D stores (mf*nf*8), found {n_stores}")
    return RocdlValidation(ok=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True)
class LlcResult:
    status: str            # "ok" | "failed" | "skipped"
    detail: str
    asm: str = ""
    wmma_instruction: str = ""


def _find_llc() -> str | None:
    # macOS Homebrew keg, the apt.llvm.org keg on Ubuntu (newest first),
    # the versioned console script, then a bare ``llc`` on PATH.
    fixed = (
        "/opt/homebrew/opt/llvm/bin/llc",
        "/usr/lib/llvm-24/bin/llc",
        "/usr/lib/llvm-23/bin/llc",
    )
    for cand in fixed:
        if Path(cand).exists():
            return cand
    for cand in ("llc-24", "llc-23", "llc"):
        p = shutil.which(cand)
        if p:
            return p
    return None


def llc_assemble(ir: str, *, arch: str = "gfx1151") -> LlcResult:
    """Rung 3: lower the emitted LLVM IR to AMDGCN with ``llc -mcpu=<arch>`` and
    confirm a real ``v_wmma_*`` instruction appears.

    **Runs on this host** (Homebrew LLVM 22 has the AMDGPU backend); skip-cleans
    (``status="skipped"``) only if no ``llc`` is found."""
    if not _LLC_ARCH_RE.match(arch):
        raise ValueError(
            f"invalid arch {arch!r} — must match {_LLC_ARCH_RE.pattern} (e.g. gfx1151)")
    llc = _find_llc()
    if not llc:
        return LlcResult("skipped", "llc (LLVM AMDGPU backend) not available")
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


@dataclass(frozen=True)
class LlcObjectResult:
    status: str            # "ok" | "failed" | "skipped"
    detail: str
    n_bytes: int = 0
    is_amdgpu_elf: bool = False


# ELF e_machine for AMD GPU (EM_AMDGPU = 224 = 0xE0), little-endian at offset 18.
_EM_AMDGPU = 0xE0


def llc_object(ir: str, *, arch: str = "gfx1100") -> LlcObjectResult:
    """Rung 3 (object form): lower the emitted LLVM IR to a real **relocatable
    object** with ``llc -filetype=obj -mcpu=<arch>`` and confirm it is an AMD GPU
    ELF (``EM_AMDGPU``) — the plan's "compiles A to a real object" gate.

    Runs on this host (LLVM 22 AMDGPU backend); skip-cleans only if no ``llc``."""
    if not _LLC_ARCH_RE.match(arch):
        raise ValueError(
            f"invalid arch {arch!r} — must match {_LLC_ARCH_RE.pattern} (e.g. gfx1100)")
    llc = _find_llc()
    if not llc:
        return LlcObjectResult("skipped", "llc (LLVM AMDGPU backend) not available")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "wmma.ll"
        out = Path(td) / "wmma.o"
        src.write_text(ir)
        try:
            proc = subprocess.run(
                [llc, "-mtriple=amdgcn-amd-amdhsa", f"-mcpu={arch}",
                 "-filetype=obj", str(src), "-o", str(out)],
                capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            return LlcObjectResult("failed", f"llc invocation error: {e}")
        if proc.returncode != 0 or not out.exists():
            return LlcObjectResult("failed", proc.stderr.strip() or "llc returned nonzero")
        data = out.read_bytes()
        is_elf = data[:4] == b"\x7fELF"
        e_machine = int.from_bytes(data[18:20], "little") if len(data) >= 20 else 0
        is_amdgpu = is_elf and e_machine == _EM_AMDGPU
        if not is_amdgpu:
            return LlcObjectResult(
                "failed",
                f"object is not an AMDGPU ELF (elf={is_elf}, e_machine=0x{e_machine:x})",
                n_bytes=len(data))
        return LlcObjectResult("ok", f"emitted AMDGPU ELF object for {arch}",
                               n_bytes=len(data), is_amdgpu_elf=True)


__all__ = [
    "wmma_intrinsic",
    "wmma_intrinsic_rdna4",
    "wmma_intrinsic_gfx1250",
    "emit_wmma_llvmir",
    "emit_wmma_rdna4_llvmir",
    "emit_wmma_gfx1250_llvmir",
    "validate_wmma_rdna4_structure",
    "validate_wmma_gfx1250_structure",
    "emit_wmma_gemm_llvmir",
    "emit_wmma_gemm_layout_llvmir",
    "emit_wmma_gemm_store_llvmir",
    "emit_wmma_gemm_threadgroup_llvmir",
    "emit_dependent_wmma_chain_llvmir",
    "wmma_scheduling",
    "WmmaScheduling",
    "validate_wmma_llvmir_structure",
    "validate_wmma_gemm_structure",
    "validate_wmma_gemm_layout_structure",
    "validate_wmma_gemm_store_structure",
    "validate_wmma_gemm_threadgroup_structure",
    "llc_assemble",
    "llc_object",
    "RocdlValidation",
    "LlcResult",
    "LlcObjectResult",
]
