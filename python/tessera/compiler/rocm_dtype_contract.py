"""Layered gfx1151/RDNA3.5 datatype contract.

An ISA opcode is not automatically a Tessera tensor-storage or execution
claim.  This table keeps four separate questions explicit:

* whether RDNA3.5 has scalar/vector handling for the format;
* whether gfx1151 WMMA accepts it as an input or accumulator;
* whether Tessera registers it as executable storage on ``rocm_gfx1151``;
* which checked-in ISA opcodes support the architecture statement.

The AMD-derived JSON archive under ``docs/reference/isa/rdna/rdna35`` is the
machine-readable source.  ``isa_evidence`` names must exist in that archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ScalarVectorState = Literal[
    "native", "packed_native", "expanded_native", "logical", "unsupported"
]
MatrixState = Literal["native_input", "accumulator_only", "unsupported"]
TesseraTargetState = Literal[
    "ready", "planned_gated", "unregistered", "not_applicable"
]
ROCmToolchainState = Literal["validated", "available_unvalidated", "unsupported"]


@dataclass(frozen=True)
class GFX1151DtypeContract:
    storage: str
    scalar_vector: ScalarVectorState
    matrix: MatrixState
    matrix_format: str | None
    accumulators: tuple[str, ...]
    tessera_target_state: TesseraTargetState
    isa_evidence: tuple[str, ...]
    note: str = ""
    rocm_toolchain_state: ROCmToolchainState = "unsupported"

    def __post_init__(self) -> None:
        if not self.storage:
            raise ValueError("gfx1151 dtype rows require a storage spelling")
        if self.matrix == "native_input":
            if not self.matrix_format or not self.accumulators:
                raise ValueError("native gfx1151 matrix inputs require format and accumulators")
        elif self.matrix_format is not None:
            raise ValueError("non-input gfx1151 rows cannot name a matrix format")
        if self.matrix == "unsupported" and self.accumulators:
            raise ValueError("unsupported gfx1151 matrix rows cannot name accumulators")
        if self.scalar_vector != "unsupported" and not self.isa_evidence:
            raise ValueError("supported gfx1151 scalar/vector rows require ISA evidence")
        if (
            self.scalar_vector != "unsupported"
            and self.rocm_toolchain_state == "unsupported"
        ):
            raise ValueError("positive ISA rows require an explicit ROCm toolchain state")
        if self.tessera_target_state == "ready" and self.scalar_vector == "unsupported":
            raise ValueError("ready gfx1151 storage needs an ISA handling route")

    @property
    def matrix_selectable(self) -> bool:
        return self.matrix == "native_input" and self.tessera_target_state == "ready"


# Every canonical Tessera dtype and every planned/gated dtype has exactly one
# row.  Unsupported rows are intentional negatives, not missing information.
GFX1151_DTYPE_CONTRACTS: tuple[GFX1151DtypeContract, ...] = (
    GFX1151DtypeContract(
        "fp64", "native", "unsupported", None, (), "unregistered",
        ("V_ADD_F64", "V_MUL_F64", "V_FMA_F64"),
        "RDNA3.5 has scalar/vector FP64, but gfx1151 has no FP64 WMMA and the Tessera target row is not registered.",
        rocm_toolchain_state="available_unvalidated",
    ),
    GFX1151DtypeContract(
        "fp32", "native", "accumulator_only", None, ("fp32",), "ready",
        ("V_ADD_F32", "V_MUL_F32", "V_WMMA_F32_16X16X16_F16"),
        "FP32 is native scalar/vector and a WMMA accumulator, not a gfx1151 WMMA input format.",
        rocm_toolchain_state="validated",
    ),
    GFX1151DtypeContract(
        "fp16", "native", "native_input", "f16", ("fp16", "fp32"), "ready",
        ("V_ADD_F16", "V_DOT2_F32_F16", "V_WMMA_F32_16X16X16_F16", "V_WMMA_F16_16X16X16_F16"),
        rocm_toolchain_state="validated",
    ),
    GFX1151DtypeContract(
        "bf16", "packed_native", "native_input", "bf16", ("bf16", "fp32"), "ready",
        ("V_DOT2_F32_BF16", "V_DOT2_BF16_BF16", "V_WMMA_F32_16X16X16_BF16", "V_WMMA_BF16_16X16X16_BF16"),
        "BF16 is a packed dot/WMMA format; it must not imply the full FP16 scalar opcode surface.",
        rocm_toolchain_state="validated",
    ),
    GFX1151DtypeContract(
        "fp8_e4m3", "unsupported", "unsupported", None, (), "not_applicable", (),
        "RDNA3.5 has no FP8/BF8 scalar conversion or WMMA opcode in the archived ISA.",
    ),
    GFX1151DtypeContract(
        "fp8_e5m2", "unsupported", "unsupported", None, (), "not_applicable", (),
        "RDNA3.5 has no FP8/BF8 scalar conversion or WMMA opcode in the archived ISA.",
    ),
    GFX1151DtypeContract("fp6_e2m3", "unsupported", "unsupported", None, (), "not_applicable", ()),
    GFX1151DtypeContract("fp6_e3m2", "unsupported", "unsupported", None, (), "not_applicable", ()),
    GFX1151DtypeContract("fp4_e2m1", "unsupported", "unsupported", None, (), "not_applicable", ()),
    GFX1151DtypeContract(
        "nvfp4", "unsupported", "unsupported", None, (), "not_applicable", (),
        "NVFP4 is NVIDIA-specific and has no AMD gfx1151 representation.",
    ),
    GFX1151DtypeContract(
        "int8", "packed_native", "native_input", "iu8", ("int32",), "ready",
        ("V_DOT4_I32_IU8", "V_DOT4_U32_U8", "V_WMMA_I32_16X16X16_IU8"),
        "The physical WMMA format has signedness controls; Tessera currently exposes canonical signed int8 storage.",
        rocm_toolchain_state="validated",
    ),
    GFX1151DtypeContract(
        "int16", "packed_native", "unsupported", None, (), "unregistered",
        ("V_PK_ADD_I16", "V_MAD_I16", "GLOBAL_LOAD_I16"),
        rocm_toolchain_state="available_unvalidated",
    ),
    GFX1151DtypeContract(
        "int32", "native", "accumulator_only", None, ("int32",), "unregistered",
        ("V_ADD_NC_U32", "V_MUL_I32_I24", "V_WMMA_I32_16X16X16_IU8"),
        rocm_toolchain_state="available_unvalidated",
    ),
    GFX1151DtypeContract(
        "int64", "expanded_native", "unsupported", None, (), "unregistered",
        ("V_CMP_EQ_I64", "S_ASHR_I64"),
        "64-bit integer handling is instruction-sequence based; there is no gfx1151 integer-64 WMMA.",
        rocm_toolchain_state="available_unvalidated",
    ),
    GFX1151DtypeContract(
        "bool", "logical", "unsupported", None, (), "unregistered",
        ("V_CMP_EQ_I32",),
        "Boolean values lower through compare/mask logic rather than a numeric WMMA format.",
        rocm_toolchain_state="available_unvalidated",
    ),
    GFX1151DtypeContract(
        "uint8", "packed_native", "native_input", "iu8", ("int32",), "planned_gated",
        ("V_DOT4_U32_U8", "V_WMMA_I32_16X16X16_IU8"),
        rocm_toolchain_state="available_unvalidated",
    ),
    GFX1151DtypeContract(
        "uint16", "packed_native", "unsupported", None, (), "planned_gated",
        ("V_PK_ADD_U16", "V_MAD_U16", "GLOBAL_LOAD_U16"),
        rocm_toolchain_state="available_unvalidated",
    ),
    GFX1151DtypeContract(
        "uint32", "native", "unsupported", None, (), "planned_gated",
        ("V_ADD_NC_U32", "V_MUL_U32_U24"),
        rocm_toolchain_state="available_unvalidated",
    ),
    GFX1151DtypeContract(
        "uint64", "expanded_native", "unsupported", None, (), "planned_gated",
        ("V_CMP_EQ_U64", "S_LSHL_B64"),
        rocm_toolchain_state="available_unvalidated",
    ),
    GFX1151DtypeContract(
        "int4", "packed_native", "native_input", "iu4", ("int32",), "planned_gated",
        ("V_DOT8_I32_IU4", "V_DOT8_U32_U4", "V_WMMA_I32_16X16X16_IU4"),
        "IU4 exists in hardware and executes in the ROCm fragment lane, but first-class packed int4 storage remains gated.",
        rocm_toolchain_state="validated",
    ),
    GFX1151DtypeContract("complex32", "unsupported", "unsupported", None, (), "planned_gated", ()),
    GFX1151DtypeContract("complex64", "unsupported", "unsupported", None, (), "planned_gated", ()),
    GFX1151DtypeContract("complex128", "unsupported", "unsupported", None, (), "planned_gated", ()),
    GFX1151DtypeContract("mxfp8", "unsupported", "unsupported", None, (), "planned_gated", ()),
    GFX1151DtypeContract("mxfp6", "unsupported", "unsupported", None, (), "planned_gated", ()),
    GFX1151DtypeContract("mxfp4", "unsupported", "unsupported", None, (), "planned_gated", ()),
)


_BY_STORAGE = {row.storage: row for row in GFX1151_DTYPE_CONTRACTS}


def gfx1151_dtype_contract(storage: str) -> GFX1151DtypeContract:
    try:
        return _BY_STORAGE[storage]
    except KeyError as exc:
        raise ValueError(f"unknown gfx1151 dtype contract {storage!r}") from exc


def gfx1151_ready_storage_dtypes() -> frozenset[str]:
    return frozenset(
        row.storage
        for row in GFX1151_DTYPE_CONTRACTS
        if row.tessera_target_state == "ready"
    )


__all__ = [
    "GFX1151DtypeContract",
    "GFX1151_DTYPE_CONTRACTS",
    "gfx1151_dtype_contract",
    "gfx1151_ready_storage_dtypes",
]
