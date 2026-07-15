"""Architecture-owned physical fragment descriptors for AMD matrix cores.

Tile IR owns logical matrix roles and layouts.  This module owns the physical
per-lane ABI selected after an exact AMD architecture is known.  In particular,
gfx11 WMMA duplication, gfx12 dense WMMA fragments, gfx125x WMMA-v2, and CDNA
MFMA must never share an implicit register map.

The descriptor is intentionally data-only.  C++ lowering mirrors these fields
and a consistency test prevents either side from drifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .rocm_target import AMDArch, TesseraROCmTargetError


class FragmentFamily(str, Enum):
    RDNA3_WMMA = "rdna3_wmma"
    RDNA4_WMMA = "rdna4_wmma"
    GFX125X_WMMA_V2 = "gfx125x_wmma_v2"
    CDNA2_MFMA = "cdna2_mfma"
    CDNA3_MFMA = "cdna3_mfma"
    CDNA4_MFMA = "cdna4_mfma"


class RegisterFormat(str, Enum):
    WMMA_INPUT_GFX11 = "wmma_input_gfx11"
    WMMA_ACC_GFX11 = "wmma_acc_gfx11"
    SOA = "soa"
    SOA_INT = "soa_int"


@dataclass(frozen=True)
class FragmentLayoutDescriptor:
    arch: AMDArch
    family: FragmentFamily
    matrix_op: str
    wave_size: int
    shape: tuple[int, int, int]
    dtype: str
    acc_dtype: str
    input_elements_per_lane: int
    input_registers_per_lane: int
    accumulator_elements_per_lane: int
    accumulator_registers_per_lane: int
    input_format: RegisterFormat
    accumulator_format: RegisterFormat
    input_lane_replication: int
    intrinsic_abi: str
    materialization_ready: bool = True

    def __post_init__(self) -> None:
        m, n, k = self.shape
        if self.input_lane_replication < 1:
            raise TesseraROCmTargetError("input_lane_replication must be positive")
        if (self.wave_size * self.input_elements_per_lane //
                self.input_lane_replication) != m * k:
            raise TesseraROCmTargetError(
                "fragment descriptor does not cover one A/B tile: "
                f"wave={self.wave_size}, elements/lane={self.input_elements_per_lane}, "
                f"shape={self.shape}")
        if self.wave_size * self.accumulator_elements_per_lane != m * n:
            raise TesseraROCmTargetError(
                "fragment descriptor does not cover one accumulator tile: "
                f"wave={self.wave_size}, elements/lane="
                f"{self.accumulator_elements_per_lane}, shape={self.shape}")

    @property
    def has_duplicated_inputs(self) -> bool:
        return self.input_lane_replication > 1

    @property
    def accumulator_mapping(self) -> str:
        # gfx11 pads/reorders C/D around its duplicated-input ABI.  Every other
        # family uses the normal SOA row-per-lane accumulator organization.
        return (
            "gfx11_col_per_lane"
            if self.accumulator_format is RegisterFormat.WMMA_ACC_GFX11
            else "soa_row_per_lane"
        )

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "matrix_op": self.matrix_op,
            "wave_size": self.wave_size,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "acc_dtype": self.acc_dtype,
            "input_elements_per_lane": self.input_elements_per_lane,
            "input_registers_per_lane": self.input_registers_per_lane,
            "accumulator_elements_per_lane": self.accumulator_elements_per_lane,
            "accumulator_registers_per_lane": self.accumulator_registers_per_lane,
            "input_format": self.input_format.value,
            "accumulator_format": self.accumulator_format.value,
            "input_lane_replication": self.input_lane_replication,
            "intrinsic_abi": self.intrinsic_abi,
            "accumulator_mapping": self.accumulator_mapping,
            "materialization_ready": self.materialization_ready,
        }


_RDNA3 = frozenset({AMDArch.GFX_1100, AMDArch.GFX_1151})
_RDNA4 = frozenset({AMDArch.GFX_1200, AMDArch.GFX_1201})
_GFX125X = frozenset({AMDArch.GFX_1250, AMDArch.GFX_1251})
_CDNA2 = frozenset({AMDArch.GFX_90A})
_CDNA3 = frozenset({AMDArch.GFX_940, AMDArch.GFX_942})
_CDNA4 = frozenset({AMDArch.GFX_950})

_BITS = {
    "fp16": 16,
    "bf16": 16,
    "int8": 8,
    "int4": 4,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "fp4_e2m1": 4,
    "fp32": 32,
}


def _acc_dtype(dtype: str) -> str:
    return "int32" if dtype in ("int8", "int4") else "fp32"


def select_fragment_layout(
    arch: AMDArch,
    dtype: str,
    shape: tuple[int, int, int],
) -> FragmentLayoutDescriptor:
    """Resolve the exact physical fragment ABI for ``(arch, dtype, shape)``.

    Unsupported combinations fail by name; there is deliberately no generic
    gfx-prefix fallback because that is the miscompile ROCM-5 is removing.
    """
    if dtype not in _BITS:
        raise TesseraROCmTargetError(
            f"ROCM_FRAGMENT_UNSUPPORTED_DTYPE: no physical fragment for {dtype!r}")
    m, n, k = shape
    if (m, n) != (16, 16):
        raise TesseraROCmTargetError(
            "ROCM_FRAGMENT_UNSUPPORTED_SHAPE: portable materialization currently "
            f"requires m16n16; got {shape}")

    if arch in _RDNA3:
        legal_k = 16
        legal = {"fp16", "bf16", "int8", "int4"}
        if dtype not in legal or k != legal_k:
            raise TesseraROCmTargetError(
                "ROCM_FRAGMENT_ILLEGAL_RDNA3_WMMA: gfx11 requires m16n16k16 "
                f"with {sorted(legal)}; got {dtype} {shape}")
        logical_per_lane = m * k // 16
        registers = (logical_per_lane * _BITS[dtype] + 31) // 32
        return FragmentLayoutDescriptor(
            arch, FragmentFamily.RDNA3_WMMA, "wmma", 32, shape, dtype,
            _acc_dtype(dtype), logical_per_lane, registers, 8, 8,
            RegisterFormat.WMMA_INPUT_GFX11,
            RegisterFormat.WMMA_ACC_GFX11,
            2, "abc_3arg_gfx11")

    if arch in _RDNA4:
        legal_k = 32 if dtype == "int4" else 16
        legal = {"fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8", "int4"}
        if dtype not in legal or k != legal_k:
            raise TesseraROCmTargetError(
                "ROCM_FRAGMENT_ILLEGAL_RDNA4_WMMA: gfx120x requires the RDNA4 "
                f"WMMA shape/dtype table; got {dtype} {shape}")
        elements = m * k // 32
        registers = (elements * _BITS[dtype] + 31) // 32
        return FragmentLayoutDescriptor(
            arch, FragmentFamily.RDNA4_WMMA, "wmma", 32, shape, dtype,
            _acc_dtype(dtype), elements, registers, 8, 8,
            RegisterFormat.SOA if _BITS[dtype] >= 16 else RegisterFormat.SOA_INT,
            RegisterFormat.SOA if _acc_dtype(dtype) == "fp32" else RegisterFormat.SOA_INT,
            1, "abc_3arg_gfx12")

    if arch in _GFX125X:
        legal_k = 32 if dtype in ("fp16", "bf16") else 64
        legal = {"fp16", "bf16", "fp8_e4m3", "fp8_e5m2"}
        if dtype not in legal or k != legal_k:
            raise TesseraROCmTargetError(
                "ROCM_FRAGMENT_ILLEGAL_GFX125X_WMMA_V2: gfx125x requires its "
                f"K-doubled WMMA-v2 table; got {dtype} {shape}")
        elements = m * k // 32
        registers = (elements * _BITS[dtype] + 31) // 32
        return FragmentLayoutDescriptor(
            arch, FragmentFamily.GFX125X_WMMA_V2, "wmma", 32, shape, dtype,
            "fp32", elements, registers, 8, 8, RegisterFormat.SOA,
            RegisterFormat.SOA, 1, "mods_reuse_8arg_gfx125x",
            materialization_ready=dtype in ("fp16", "bf16"))

    if arch in _CDNA2 | _CDNA3 | _CDNA4:
        family = (
            FragmentFamily.CDNA2_MFMA if arch in _CDNA2 else
            FragmentFamily.CDNA3_MFMA if arch in _CDNA3 else
            FragmentFamily.CDNA4_MFMA
        )
        legal_by_k = {
            "fp16": 16, "bf16": 16, "int8": 16, "fp32": 8,
            "fp8_e4m3": 32, "fp8_e5m2": 32, "fp4_e2m1": 64,
        }
        if dtype not in legal_by_k or k != legal_by_k[dtype]:
            raise TesseraROCmTargetError(
                "ROCM_FRAGMENT_ILLEGAL_CDNA_MFMA: CDNA requires its MFMA "
                f"shape/dtype table; got {dtype} {shape}")
        if arch in _CDNA2 and dtype in ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1"):
            raise TesseraROCmTargetError(
                "ROCM_FRAGMENT_UNSUPPORTED_CDNA2_DTYPE: gfx90a has no FP8/FP4 MFMA")
        if arch in _CDNA3 and dtype == "fp4_e2m1":
            raise TesseraROCmTargetError(
                "ROCM_FRAGMENT_UNSUPPORTED_CDNA3_DTYPE: gfx94x has no FP4 MFMA")
        elements = m * k // 64
        registers = (elements * _BITS[dtype] + 31) // 32
        return FragmentLayoutDescriptor(
            arch, family, "mfma", 64, shape, dtype, _acc_dtype(dtype),
            elements, registers, 4, 4,
            RegisterFormat.SOA if _BITS[dtype] >= 16 else RegisterFormat.SOA_INT,
            RegisterFormat.SOA if _acc_dtype(dtype) == "fp32" else RegisterFormat.SOA_INT,
            1, "mfma_abc_ctrl")

    raise TesseraROCmTargetError(
        f"ROCM_FRAGMENT_UNSUPPORTED_ARCH: no fragment family for {arch.name}")


__all__ = [
    "FragmentFamily",
    "RegisterFormat",
    "FragmentLayoutDescriptor",
    "select_fragment_layout",
]
