"""tessera.compiler.rocm_mma — unified cooperative-matrix MMA descriptor (A1).

AMD's rocWMMA (``fragment<MatrixT, M,N,K, DataT, Layout>``) and AMD Gluon
(``AMDMFMALayout`` anchoring ``DotOperandLayout(parent, operand_index, k_width)``)
teach the same lesson: **the MMA instruction shape is the anchor, and the A/B
operand layouts + the K-packing width are *derived* from it** — they are not
selected independently.

Today Tessera's ``rocm_target`` exposes MFMA shapes (``_MFMA_VARIANTS``) and WMMA
shapes (``_WMMA_VARIANTS``) as bare tuples, with no object tying a chosen shape to
the operand layouts/dtype/packing it implies, and the two matrix paths
(CDNA MFMA, RDNA WMMA) are modelled as parallel-but-separate tables.  This module
unifies them behind one descriptor — ``MmaDescriptor`` — produced by a single
selector that dispatches MFMA vs WMMA by arch.  This is Architecture Decision #19
("hardware-free Target IR before hardware-specific lowering") applied to the
matrix-core op: a backend-shaped, hardware-free object that lit/unit tests can
reason over before any HIP emission.

Invariant (Gluon/rocWMMA both): **register element ordering is backend-owned and
not exposed.**  ``k_width`` here is an honest *packing hint* (how many K-elements
a lane naturally co-loads, ``32 // dtype_bits``), NOT a claim about which lane
holds which element.  Exact VGPR/operand layout is grounded only by the emitter
(``rocdl_emit.py``) + real silicon (see ``STRIX_HALO_EXECUTION_PLAN.md``).

Leaf-ish: depends only on ``rocm_target`` (the per-arch shape/dtype/feature
tables) and the stdlib, so the audit registry / backend manifest can import it
without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rocm_target import (
    AMDArch,
    ROCmTargetProfile,
    TesseraROCmTargetError,
    mfma_variants,
    rocm_feature_status,
    wmma_variants,
)
from .rocm_fragment import FragmentLayoutDescriptor, select_fragment_layout

# ── dtype bit widths (MMA-relevant canonical dtypes) ────────────────────────
# Storage bit width per canonical dtype; used to derive the K-packing hint and
# to reject dtypes with no matrix-core path.  fp6 is byte-addressed in practice
# (no MMA path is modelled here), so it is intentionally absent.
_DTYPE_BITS: dict[str, int] = {
    "fp64": 64,
    "fp32": 32,
    "bf16": 16,
    "fp16": 16,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "fp4_e2m1": 4,
    "int8": 8,
    "int4": 4,
}

# Canonical MMA input dtypes (a/b operand storage).  fp32 routes through the
# xf32 ("tf32-equivalent on AMD") MFMA path, so it is allowed on CDNA only.
_MMA_INPUT_DTYPES: frozenset[str] = frozenset(_DTYPE_BITS)

# Per-dtype K for the canonical 16x16 MFMA family (CDNA).  Grounded in the
# `_MFMA_VARIANTS` comments: bf16/fp16 → K=16, fp8 → K=32, fp4 → K=64, and
# fp32 uses the xf32 lane at K=8.  int8 shares the 16-bit-class K=16 tile.
_MFMA_K_BY_DTYPE: dict[str, int] = {
    "bf16": 16,
    "fp16": 16,
    "int8": 16,
    "fp8_e4m3": 32,
    "fp8_e5m2": 32,
    "fp4_e2m1": 64,
    "fp32": 8,   # xf32 path (requires math_mode='tf32')
}

# Per-dtype K for the canonical 16x16 WMMA family, keyed by arch *class* because
# RDNA generations doubled K.  RDNA 3/3.5 (gfx1100/1151): all combos at K=16.
# RDNA 4 (gfx1200): f16/bf16/fp8 at K=16, int4 at K=32.  gfx125x ("v2" ABI):
# K is doubled — f16/bf16 at K=32, fp8 at K=64.  (Grounded in `_WMMA_VARIANTS`.)
_WMMA_K_RDNA3: dict[str, int] = {
    "fp16": 16, "bf16": 16, "int8": 16, "int4": 16,
}
_WMMA_K_RDNA4: dict[str, int] = {
    "fp16": 16, "bf16": 16, "int8": 16, "int4": 32,
    "fp8_e4m3": 16, "fp8_e5m2": 16,
}
_WMMA_K_GFX125X: dict[str, int] = {
    "fp16": 32, "bf16": 32, "fp8_e4m3": 64, "fp8_e5m2": 64,
}

_RDNA3_CLASS = frozenset({AMDArch.GFX_1100, AMDArch.GFX_1151})
_RDNA4_CLASS = frozenset({AMDArch.GFX_1200, AMDArch.GFX_1201})
_GFX125X_CLASS = frozenset({AMDArch.GFX_1250, AMDArch.GFX_1251})


def _k_packing_hint(dtype: str) -> int:
    """Honest K-packing hint: how many K-elements a 32-bit lane slot co-loads.

    This is the Gluon ``k_width`` *intent* (consecutive elements a thread pulls
    along K), expressed as ``max(1, 32 // dtype_bits)`` — f16/bf16 → 2, fp8 → 4,
    int8 → 4, fp4 → 8, fp32 → 1, fp64 → 1.  It is a packing hint only; the exact
    per-lane register layout is backend-owned (see module docstring).
    """
    return max(1, 32 // _DTYPE_BITS[dtype])


def _accum_dtype(in_dtype: str) -> str:
    """Accumulator dtype per the rocWMMA rule: i32 for int8, f64 for f64, else
    f32 (every low-precision float path accumulates in fp32)."""
    if in_dtype in ("int8", "int4"):
        return "int32"
    if in_dtype == "fp64":
        return "fp64"
    return "fp32"


@dataclass(frozen=True)
class MmaOperand:
    """One operand of an MMA instruction — role, dtype, layout, packing.

    ``role`` ∈ {matrix_a, matrix_b, accumulator}.  ``layout`` is the *logical*
    operand layout (``row_major``/``col_major``), not a register map — the
    default ``nt`` formulation has A row-major (K-major) and B col-major, which
    is what the WMMA emitter uses.  ``k_width`` is the packing hint (1 for the
    accumulator, which is not loaded along K).
    """

    role: str
    dtype: str
    layout: str
    k_width: int

    _ROLES = ("matrix_a", "matrix_b", "accumulator")
    _LAYOUTS = ("row_major", "col_major")

    def __post_init__(self) -> None:
        if self.role not in self._ROLES:
            raise TesseraROCmTargetError(
                f"MmaOperand role must be one of {self._ROLES}; got {self.role!r}")
        if self.layout not in self._LAYOUTS:
            raise TesseraROCmTargetError(
                f"MmaOperand layout must be one of {self._LAYOUTS}; got {self.layout!r}")
        if self.k_width < 1:
            raise TesseraROCmTargetError(
                f"MmaOperand k_width must be >= 1; got {self.k_width}")

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "dtype": self.dtype,
            "layout": self.layout,
            "k_width": self.k_width,
        }


@dataclass(frozen=True)
class MmaDescriptor:
    """A unified MFMA(CDNA)/WMMA(RDNA) cooperative-matrix instruction descriptor.

    The chosen ``shape`` is the anchor; ``operand_a``/``operand_b``/``accumulator``
    are *derived* from it (dtype + layout + k_width), so a consumer never picks an
    operand layout inconsistent with the instruction.  ``intrinsic`` is a
    documented mnemonic family for readability/lit-matching — NOT a claim that a
    complete, numerically-correct kernel has been assembled.
    """

    arch: AMDArch
    kind: str                 # "mfma" | "wmma"
    shape: tuple[int, int, int]   # (M, N, K)
    k_blocks: int             # MFMA K_blocks; 1 for WMMA
    in_dtype: str
    acc_dtype: str
    operand_a: MmaOperand
    operand_b: MmaOperand
    accumulator: MmaOperand
    transposed: bool          # the C^T = B^T A^T formulation flag

    @property
    def m(self) -> int:
        return self.shape[0]

    @property
    def n(self) -> int:
        return self.shape[1]

    @property
    def k(self) -> int:
        return self.shape[2]

    @property
    def operands(self) -> tuple[MmaOperand, MmaOperand, MmaOperand]:
        return (self.operand_a, self.operand_b, self.accumulator)

    @property
    def fragment_layout(self) -> FragmentLayoutDescriptor:
        return select_fragment_layout(self.arch, self.in_dtype, self.shape)

    @property
    def intrinsic(self) -> str:
        """Documented mnemonic family, e.g. ``v_wmma_f32_16x16x16_f16`` or
        ``v_mfma_f32_16x16x16_bf16``.  Readability/lit aid, not an ABI claim."""
        m, n, k = self.shape
        acc = "f32" if self.acc_dtype == "fp32" else (
            "i32" if self.acc_dtype == "int32" else "f64")
        din = _MNEMONIC_DTYPE.get(self.in_dtype, self.in_dtype)
        return f"v_{self.kind}_{acc}_{m}x{n}x{k}_{din}"

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "arch": self.arch.name,
            "kind": self.kind,
            "shape": list(self.shape),
            "k_blocks": self.k_blocks,
            "in_dtype": self.in_dtype,
            "acc_dtype": self.acc_dtype,
            "transposed": self.transposed,
            "intrinsic": self.intrinsic,
            "operands": [op.as_metadata_dict() for op in self.operands],
            "physical_fragment": self.fragment_layout.as_metadata_dict(),
        }


# Mnemonic dtype spellings for the documented intrinsic family.
_MNEMONIC_DTYPE: dict[str, str] = {
    "fp16": "f16", "bf16": "bf16", "fp32": "xf32",
    "fp8_e4m3": "fp8", "fp8_e5m2": "bf8", "fp4_e2m1": "fp4", "int8": "i8",
}


def _wmma_k_table(arch: AMDArch) -> dict[str, int]:
    if arch in _RDNA3_CLASS:
        return _WMMA_K_RDNA3
    if arch in _RDNA4_CLASS:
        return _WMMA_K_RDNA4
    if arch in _GFX125X_CLASS:
        return _WMMA_K_GFX125X
    raise TesseraROCmTargetError(  # pragma: no cover — guarded by caller
        f"{arch.name} is not a WMMA (RDNA-class) arch")


def _fp8_unsupported_on_rdna35(arch: AMDArch, dtype: str) -> bool:
    """RDNA 3.5 (gfx1151) has NO FP8 WMMA instruction (the load-bearing
    distinction from gfx1200).  Cross-checks ``wmma_f8`` feature status."""
    return (
        dtype in ("fp8_e4m3", "fp8_e5m2")
        and rocm_feature_status(arch, "wmma_f8") != "ready"
    )


def select_mma(
    arch: AMDArch,
    dtype: str,
    *,
    out_dtype: str | None = None,
    prefer_shape: tuple[int, int, int] | None = None,
    transposed: bool = False,
) -> MmaDescriptor:
    """Select the cooperative-matrix instruction for ``dtype`` on ``arch`` and
    derive its operand descriptors.

    Dispatches MFMA (CDNA) vs WMMA (RDNA/wave32) by arch.  Raises
    ``TesseraROCmTargetError`` (never silently falls back) when the dtype has no
    matrix-core path on the arch — e.g. FP8 on gfx1151, or any dtype absent from
    the arch's variant table.
    """
    if dtype not in _MMA_INPUT_DTYPES:
        raise TesseraROCmTargetError(
            f"dtype {dtype!r} has no cooperative-matrix path; MMA input dtypes "
            f"are {sorted(_MMA_INPUT_DTYPES)}")

    is_wave32 = arch in (
        AMDArch.GFX_1100, AMDArch.GFX_1151, AMDArch.GFX_1200,
        AMDArch.GFX_1201,
        AMDArch.GFX_1250, AMDArch.GFX_1251,
    )

    if is_wave32:
        kind = "wmma"
        variants = wmma_variants(arch)
        if not variants:
            raise TesseraROCmTargetError(
                f"{arch.name} exposes no WMMA shapes")
        if _fp8_unsupported_on_rdna35(arch, dtype):
            raise TesseraROCmTargetError(
                f"{arch.name} has no FP8 WMMA instruction (wmma_f8="
                f"{rocm_feature_status(arch, 'wmma_f8')!r}); FP8 matmul must "
                f"decompose. This is the load-bearing distinction from gfx1200.")
        ktab = _wmma_k_table(arch)
        if dtype not in ktab:
            raise TesseraROCmTargetError(
                f"dtype {dtype!r} has no WMMA K mapping on {arch.name} "
                f"(supported: {sorted(ktab)})")
        k = ktab[dtype]
        shape3 = (16, 16, k)
        if shape3 not in variants:
            raise TesseraROCmTargetError(
                f"{arch.name} WMMA K table derived {shape3} for {dtype!r} but "
                f"that shape is absent from _WMMA_VARIANTS ({sorted(variants)}); "
                f"the K table and the variant table have drifted.")
        k_blocks = 1
    else:
        kind = "mfma"
        variants4 = mfma_variants(arch)
        if not variants4:
            raise TesseraROCmTargetError(
                f"{arch.name} exposes no MFMA shapes")
        if dtype not in _MFMA_K_BY_DTYPE:
            raise TesseraROCmTargetError(
                f"dtype {dtype!r} has no MFMA K mapping (supported: "
                f"{sorted(_MFMA_K_BY_DTYPE)})")
        # fp8/fp4 require feature support on the arch.
        if dtype in ("fp8_e4m3", "fp8_e5m2") and \
                rocm_feature_status(arch, "mfma_f8") != "ready":
            raise TesseraROCmTargetError(
                f"{arch.name} has no FP8 MFMA (mfma_f8="
                f"{rocm_feature_status(arch, 'mfma_f8')!r})")
        if dtype == "fp4_e2m1" and rocm_feature_status(arch, "mfma_f4") != "ready":
            raise TesseraROCmTargetError(
                f"{arch.name} has no FP4 MFMA (mfma_f4="
                f"{rocm_feature_status(arch, 'mfma_f4')!r})")
        k = _MFMA_K_BY_DTYPE[dtype]
        shape3 = (16, 16, k)
        # validate against the (M,N,K,K_blocks) variant table.
        match = [v for v in variants4 if v[:3] == shape3]
        if not match:
            raise TesseraROCmTargetError(
                f"{arch.name} has no MFMA variant for {dtype!r} at shape "
                f"{shape3} (available: {sorted(variants4)})")
        k_blocks = match[0][3]

    # Honor an explicit caller preference if it is legal for the arch.
    if prefer_shape is not None:
        legal3 = {v[:3] for v in mfma_variants(arch)} if kind == "mfma" \
            else set(wmma_variants(arch))
        if prefer_shape not in legal3:
            raise TesseraROCmTargetError(
                f"prefer_shape {prefer_shape} is not a legal {kind.upper()} "
                f"shape on {arch.name} (legal: {sorted(legal3)})")
        shape3 = prefer_shape
        if kind == "mfma":
            k_blocks = next(v[3] for v in mfma_variants(arch) if v[:3] == shape3)

    acc = out_dtype or _accum_dtype(dtype)
    kw = _k_packing_hint(dtype)
    # Default nt formulation: A row-major (K-major), B col-major, C row-major.
    op_a = MmaOperand("matrix_a", dtype, "row_major", kw)
    op_b = MmaOperand("matrix_b", dtype, "col_major", kw)
    op_c = MmaOperand("accumulator", acc, "row_major", 1)

    return MmaDescriptor(
        arch=arch,
        kind=kind,
        shape=shape3,
        k_blocks=k_blocks,
        in_dtype=dtype,
        acc_dtype=acc,
        operand_a=op_a,
        operand_b=op_b,
        accumulator=op_c,
        transposed=transposed,
    )


def mma_for_matmul(
    profile: ROCmTargetProfile,
    dtype: str,
    *,
    out_dtype: str | None = None,
    transposed: bool = False,
) -> MmaDescriptor:
    """Convenience: select the MMA descriptor for a matmul on a target profile.

    ``fp32`` storage routes through the xf32 MFMA lane on CDNA — that requires
    ``numeric_policy.math_mode='tf32'``; this raises on an RDNA profile because
    RDNA has no full/xf32 fp32 WMMA path.
    """
    if dtype == "fp32" and profile.arch in _RDNA_ALL:
        raise TesseraROCmTargetError(
            "fp32 matmul has no WMMA path on RDNA; use bf16/fp16 storage or run "
            "on a CDNA arch (fp32→xf32 MFMA).")
    return select_mma(
        profile.arch, dtype, out_dtype=out_dtype, transposed=transposed)


_RDNA_ALL = frozenset(
    _RDNA3_CLASS | _RDNA4_CLASS | _GFX125X_CLASS)


__all__ = [
    "MmaOperand",
    "MmaDescriptor",
    "select_mma",
    "mma_for_matmul",
]
