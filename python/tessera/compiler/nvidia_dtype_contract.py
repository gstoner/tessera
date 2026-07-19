"""Single-source dtype/compute-mode contract for consumer Blackwell.

The table intentionally separates four concepts that were previously mixed
across target capability and fragment-selection code:

* tensor storage (TF32 is never a storage dtype);
* scalar/vector CUDA handling;
* Tensor Core operand format and accumulator semantics;
* Tessera compiler/runtime readiness.

``hardware_supported`` is not an execution claim.  A row becomes selectable
only when both ``compiler_state`` and ``runtime_state`` are ``ready`` and the
operation-specific selector has exact-device evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ScalarVectorState = Literal["native", "conversion_only", "packed_conversion"]
TensorCoreState = Literal["native", "math_mode", "unsupported"]
Readiness = Literal["ready", "planned", "not_applicable"]
PtxFormatKind = Literal["fundamental", "alternate", "subbyte"]


@dataclass(frozen=True)
class SM120DtypeContract:
    key: str
    storage: str
    math_mode: str | None
    scalar_vector: ScalarVectorState
    tensor_core: TensorCoreState
    tensor_format: str | None
    fragment_shape: tuple[int, int, int] | None
    accumulators: tuple[str, ...]
    target_ir_state: Readiness
    compiler_state: Readiness
    runtime_state: Readiness
    block_scaled: bool = False
    scale_format: str | None = None
    scale_vector: str | None = None
    note: str = ""
    ptx_storage_register: str | None = None
    storage_format_kind: PtxFormatKind | None = None
    tensor_operand_register: str | None = None
    tensor_format_kind: PtxFormatKind | None = None

    def __post_init__(self) -> None:
        if self.ptx_storage_register is None or self.storage_format_kind is None:
            raise ValueError("every SM120 dtype row requires a physical PTX storage type")
        if self.storage_format_kind != "fundamental" and not (self.ptx_storage_register.startswith(".b")):
            raise ValueError("alternate/sub-byte PTX storage must use a bit-size register")
        if self.math_mode == "tf32" and self.storage != "fp32":
            raise ValueError("TF32 math mode requires fp32 storage")
        if self.tensor_core == "unsupported":
            if self.tensor_format or self.fragment_shape or self.accumulators:
                raise ValueError("unsupported Tensor Core rows cannot name a fragment")
            if self.compiler_state != "not_applicable":
                raise ValueError("unsupported Tensor Core rows are not compiler fragments")
            if self.tensor_operand_register or self.tensor_format_kind:
                raise ValueError("unsupported Tensor Core rows cannot name PTX operands")
        elif not self.tensor_format:
            raise ValueError("Tensor Core rows require a physical operand format")
        elif self.tensor_operand_register is None or self.tensor_format_kind is None:
            raise ValueError("Tensor Core rows require physical PTX operand typing")
        if self.block_scaled != bool(self.scale_format and self.scale_vector):
            raise ValueError("block-scaled rows require a scale format and vector size")

    @property
    def selectable(self) -> bool:
        return self.compiler_state == self.runtime_state == "ready"


# Hardware formats are recorded even when Tessera materialization is planned.
# Shapes are omitted for those planned forms until the exact PTX operand and
# scale-layout contract is assembled and tested; this prevents a capability
# statement from becoming an invented register ABI.
SM120_DTYPE_CONTRACTS: tuple[SM120DtypeContract, ...] = (
    SM120DtypeContract(
        "fp64",
        "fp64",
        None,
        "native",
        "native",
        "f64",
        (8, 8, 4),
        ("fp64",),
        "ready",
        "ready",
        "ready",
        note="DMMA has a compiler-owned m8n8k4 Tile/runtime descriptor path.",
        ptx_storage_register=".f64",
        storage_format_kind="fundamental",
        tensor_operand_register=".f64",
        tensor_format_kind="fundamental",
    ),
    SM120DtypeContract(
        "fp32_ieee",
        "fp32",
        "ieee",
        "native",
        "unsupported",
        None,
        None,
        (),
        "ready",
        "not_applicable",
        "ready",
        note="IEEE fp32 uses CUDA cores; it must never silently select TF32.",
        ptx_storage_register=".f32",
        storage_format_kind="fundamental",
    ),
    SM120DtypeContract(
        "fp32_tf32",
        "fp32",
        "tf32",
        "native",
        "math_mode",
        "tf32",
        (16, 8, 8),
        ("fp32",),
        "ready",
        "ready",
        "ready",
        note="TF32 is an fp32-storage math mode.",
        ptx_storage_register=".f32",
        storage_format_kind="fundamental",
        tensor_operand_register=".b32",
        tensor_format_kind="alternate",
    ),
    SM120DtypeContract(
        "fp16",
        "fp16",
        None,
        "native",
        "native",
        "f16",
        (16, 8, 16),
        ("fp16", "fp32"),
        "ready",
        "ready",
        "ready",
        note="Fundamental .f16 is limited to PTX half-precision instruction forms.",
        ptx_storage_register=".f16",
        storage_format_kind="fundamental",
        tensor_operand_register=".b32",
        tensor_format_kind="fundamental",
    ),
    SM120DtypeContract(
        "bf16",
        "bf16",
        None,
        "conversion_only",
        "native",
        "bf16",
        (16, 8, 16),
        ("fp32",),
        "ready",
        "ready",
        "ready",
        ptx_storage_register=".b16",
        storage_format_kind="alternate",
        tensor_operand_register=".b32",
        tensor_format_kind="alternate",
    ),
    SM120DtypeContract(
        "fp8_e4m3",
        "fp8_e4m3",
        None,
        "conversion_only",
        "native",
        "e4m3",
        (16, 8, 32),
        ("fp32",),
        "ready",
        "ready",
        "ready",
        ptx_storage_register=".b8",
        storage_format_kind="alternate",
        tensor_operand_register=".b32",
        tensor_format_kind="alternate",
    ),
    SM120DtypeContract(
        "fp8_e5m2",
        "fp8_e5m2",
        None,
        "conversion_only",
        "native",
        "e5m2",
        (16, 8, 32),
        ("fp32",),
        "ready",
        "ready",
        "ready",
        ptx_storage_register=".b8",
        storage_format_kind="alternate",
        tensor_operand_register=".b32",
        tensor_format_kind="alternate",
    ),
    SM120DtypeContract(
        "fp6_e2m3",
        "fp6_e2m3",
        None,
        "packed_conversion",
        "native",
        "e2m3",
        (16, 8, 32),
        ("fp32",),
        "ready",
        "ready",
        "ready",
        block_scaled=True,
        scale_format="ue8m0",
        scale_vector="1X",
        note="Compiler-owned byte storage, UE8M0 vec32 scales, and descriptor launch are proven.",
        ptx_storage_register=".b8",
        storage_format_kind="alternate",
        tensor_operand_register=".b32",
        tensor_format_kind="alternate",
    ),
    SM120DtypeContract(
        "fp6_e3m2",
        "fp6_e3m2",
        None,
        "packed_conversion",
        "native",
        "e3m2",
        (16, 8, 32),
        ("fp32",),
        "ready",
        "ready",
        "ready",
        block_scaled=True,
        scale_format="ue8m0",
        scale_vector="1X",
        note="Compiler-owned byte storage, UE8M0 vec32 scales, and descriptor launch are proven.",
        ptx_storage_register=".b8",
        storage_format_kind="alternate",
        tensor_operand_register=".b32",
        tensor_format_kind="alternate",
    ),
    SM120DtypeContract(
        "fp4_e2m1",
        "fp4_e2m1",
        None,
        "packed_conversion",
        "native",
        "e2m1",
        (16, 8, 64),
        ("fp32",),
        "ready",
        "ready",
        "ready",
        block_scaled=True,
        scale_format="ue8m0",
        scale_vector="2X",
        note=(
            "Packed E2M1 Tile/runtime descriptor execution is proven. OCP/MXFP4 is distinct from UE4M3-scaled NVFP4."
        ),
        ptx_storage_register=".b8",
        storage_format_kind="alternate",
        tensor_operand_register=".b32",
        tensor_format_kind="alternate",
    ),
    SM120DtypeContract(
        "nvfp4",
        "nvfp4",
        None,
        "packed_conversion",
        "native",
        "e2m1",
        (16, 8, 64),
        ("fp32",),
        "ready",
        "ready",
        "ready",
        block_scaled=True,
        scale_format="ue4m3",
        scale_vector="4X",
        ptx_storage_register=".b8",
        storage_format_kind="alternate",
        tensor_operand_register=".b32",
        tensor_format_kind="alternate",
    ),
    SM120DtypeContract(
        "int8",
        "int8",
        None,
        "native",
        "native",
        "s8",
        (16, 8, 32),
        ("int32",),
        "ready",
        "ready",
        "ready",
        note=("Fundamental .s8 has restricted scalar instruction use; matrix fragments are packed in .b32 registers."),
        ptx_storage_register=".s8",
        storage_format_kind="fundamental",
        tensor_operand_register=".b32",
        tensor_format_kind="fundamental",
    ),
    SM120DtypeContract(
        "int4",
        "int4",
        None,
        "packed_conversion",
        "native",
        "s4",
        None,
        ("int32",),
        "planned",
        "planned",
        "planned",
        note="Hardware format exists; int4 remains a planned-gated Tessera dtype.",
        ptx_storage_register=".b8",
        storage_format_kind="subbyte",
        tensor_operand_register=".b32",
        tensor_format_kind="subbyte",
    ),
)


_BY_KEY = {row.key: row for row in SM120_DTYPE_CONTRACTS}


def sm120_dtype_contract(
    storage: str,
    *,
    math_mode: str | None = None,
) -> SM120DtypeContract:
    """Resolve a storage dtype plus math mode to one exact SM120 contract."""
    if storage == "fp32":
        key = "fp32_tf32" if math_mode == "tf32" else "fp32_ieee"
    else:
        if math_mode not in (None, "ieee"):
            raise ValueError(f"math mode {math_mode!r} is invalid for {storage!r}")
        key = storage
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"no SM120 dtype contract for {storage!r}") from exc


def sm120_supported_storage_dtypes() -> tuple[str, ...]:
    """Canonical first-class storage types handled by SM120 CUDA code."""
    return tuple(sorted({row.storage for row in SM120_DTYPE_CONTRACTS if row.storage != "int4"}))


def sm120_tensor_core_report_names() -> frozenset[str]:
    """Storage names plus TF32 math-mode name for hardware capability reports."""
    names = {row.storage for row in SM120_DTYPE_CONTRACTS if row.tensor_core != "unsupported"}
    names.add("tf32")
    return frozenset(names)


__all__ = [
    "PtxFormatKind",
    "SM120DtypeContract",
    "SM120_DTYPE_CONTRACTS",
    "sm120_dtype_contract",
    "sm120_supported_storage_dtypes",
    "sm120_tensor_core_report_names",
]
