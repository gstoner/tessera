"""Layered x86 AVX-512 datatype and execution contract.

Register width is not an execution claim.  Each row separates physical
handling, the required CPUID feature set, the compute/accumulator type, and
Tessera readiness.  AMX and future AVX10/ACE targets deliberately live outside
this Zen-5 AVX-512 profile; ACE remains planned until public CPUID and compiler
contracts are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


VectorState = Literal[
    "native", "packed_native", "conversion_only", "logical", "emulated", "unsupported"
]
MatrixState = Literal["vector_gemm", "dot_product", "unsupported"]
TesseraState = Literal["ready", "planned_gated", "unregistered", "not_applicable"]


@dataclass(frozen=True)
class X86DtypeContract:
    storage: str
    vector_state: VectorState
    required_features: tuple[str, ...]
    matrix_state: MatrixState
    compute: str | None
    accumulator: str | None
    tessera_state: TesseraState
    note: str = ""

    def __post_init__(self) -> None:
        if not self.storage:
            raise ValueError("x86 dtype rows require a storage spelling")
        if self.vector_state in {"native", "packed_native", "conversion_only", "logical"} and not self.required_features:
            raise ValueError("positive x86 hardware rows require CPUID features")
        if self.matrix_state != "unsupported" and (not self.compute or not self.accumulator):
            raise ValueError("x86 compute rows require compute and accumulator types")
        if self.tessera_state == "ready" and self.vector_state in {"unsupported", "emulated"}:
            raise ValueError("ready x86 rows require a hardware handling route")

    def available_on(self, features: frozenset[str] | set[str]) -> bool:
        return set(self.required_features) <= set(features)


X86_DTYPE_CONTRACTS: tuple[X86DtypeContract, ...] = (
    X86DtypeContract("fp64", "native", ("avx512f", "fma"), "vector_gemm", "fp64", "fp64", "ready"),
    X86DtypeContract("fp32", "native", ("avx512f", "fma"), "vector_gemm", "fp32", "fp32", "ready"),
    X86DtypeContract(
        "fp16", "conversion_only", ("f16c",), "unsupported", "fp32", "fp32", "planned_gated",
        "The current WSL-visible Ryzen AI Max+ 395 does not advertise avx512_fp16; FP16 is storage/conversion only.",
    ),
    X86DtypeContract("bf16", "packed_native", ("avx512_bf16",), "dot_product", "bf16", "fp32", "ready"),
    X86DtypeContract("fp8_e4m3", "emulated", (), "unsupported", "fp32", "fp32", "planned_gated", "Packed-byte storage with software conversion; no native Zen-5 FP8 arithmetic."),
    X86DtypeContract("fp8_e5m2", "emulated", (), "unsupported", "fp32", "fp32", "planned_gated", "Packed-byte storage with software conversion; no native Zen-5 FP8 arithmetic."),
    X86DtypeContract("fp6_e2m3", "unsupported", (), "unsupported", None, None, "not_applicable"),
    X86DtypeContract("fp6_e3m2", "unsupported", (), "unsupported", None, None, "not_applicable"),
    X86DtypeContract("fp4_e2m1", "unsupported", (), "unsupported", None, None, "not_applicable"),
    X86DtypeContract("nvfp4", "unsupported", (), "unsupported", None, None, "not_applicable"),
    X86DtypeContract("int8", "packed_native", ("avx512bw", "avx512_vnni"), "dot_product", "int8", "int32", "ready"),
    X86DtypeContract("int16", "packed_native", ("avx512bw",), "unsupported", "int16", "int32", "unregistered"),
    X86DtypeContract("int32", "native", ("avx512f",), "unsupported", "int32", "int32", "ready"),
    X86DtypeContract("int64", "native", ("avx512f", "avx512dq"), "unsupported", "int64", "int64", "unregistered"),
    X86DtypeContract("bool", "logical", ("avx512bw",), "unsupported", "bool", None, "ready"),
    X86DtypeContract("uint8", "packed_native", ("avx512bw", "avx512_vnni"), "dot_product", "uint8", "int32", "planned_gated", "VNNI u8*s8 input; uint8 remains a planned-gated Graph IR storage type."),
    X86DtypeContract("uint16", "packed_native", ("avx512bw",), "unsupported", "uint16", None, "planned_gated"),
    X86DtypeContract("uint32", "native", ("avx512f",), "unsupported", "uint32", None, "planned_gated"),
    X86DtypeContract("uint64", "native", ("avx512f", "avx512dq"), "unsupported", "uint64", None, "planned_gated"),
    X86DtypeContract("int4", "emulated", (), "unsupported", "int8", "int32", "planned_gated"),
    X86DtypeContract("complex32", "unsupported", (), "unsupported", None, None, "planned_gated"),
    X86DtypeContract("complex64", "unsupported", (), "unsupported", None, None, "planned_gated"),
    X86DtypeContract("complex128", "unsupported", (), "unsupported", None, None, "planned_gated"),
    X86DtypeContract("mxfp8", "unsupported", (), "unsupported", None, None, "planned_gated", "Reserved for future AVX10/ACE profiles."),
    X86DtypeContract("mxfp6", "unsupported", (), "unsupported", None, None, "planned_gated", "Reserved for future AVX10/ACE profiles."),
    X86DtypeContract("mxfp4", "unsupported", (), "unsupported", None, None, "planned_gated", "Reserved for future AVX10/ACE profiles."),
)

_BY_STORAGE = {row.storage: row for row in X86_DTYPE_CONTRACTS}


def x86_dtype_contract(storage: str) -> X86DtypeContract:
    try:
        return _BY_STORAGE[storage]
    except KeyError as exc:
        raise ValueError(f"unknown x86 dtype contract {storage!r}") from exc


def x86_ready_storage_dtypes() -> frozenset[str]:
    return frozenset(row.storage for row in X86_DTYPE_CONTRACTS if row.tessera_state == "ready")


__all__ = [
    "X86DtypeContract", "X86_DTYPE_CONTRACTS", "x86_dtype_contract",
    "x86_ready_storage_dtypes",
]
