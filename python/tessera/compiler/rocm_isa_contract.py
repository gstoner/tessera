"""ISA-grounded AMD matrix and datatype contracts.

This module deliberately separates four facts that older ROCm target tables
conflated:

* architecture family and wave size;
* scalar/vector storage availability;
* matrix operand, accumulator, sparsity, and scaling legality;
* Tessera implementation/evidence state.

The three profiles here are the first cross-generation contract: gfx1151
(RDNA 3.5), gfx1200 (RDNA 4), and gfx1250/gfx1251 (CDNA 5).  CDNA 5 is a
wave32 WMMA/SWMMAC architecture; it must never be routed through the legacy
``CDNA => wave64 MFMA`` rule.  MI455X is gfx1250 and MI430X is gfx1251.  The
two CDNA 5 targets share the low-precision instruction ABI, while gfx1251 adds
FP64 WMMA; both retain separate scheduling and performance identities.

An ``isa_supported`` row is not an execution claim.  Low-precision and sparse
CDNA 5 rows remain ``artifact_only`` or ``planned_gated`` until their Target
IR materializers and exact-device packets exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tessera.dtype import canonical_dtypes, planned_gated_dtypes

from .rocm_target import AMDArch, TesseraROCmTargetError


ArchitectureFamily = Literal["rdna35", "rdna4", "cdna5"]
MatrixPath = Literal["wmma", "swmmac"]
ContractState = Literal[
    "ready", "artifact_only", "planned_gated", "unregistered", "unsupported"
]


@dataclass(frozen=True)
class AMDArchitectureContract:
    arch: AMDArch
    family: ArchitectureFamily
    product: str
    wave_size: int
    matrix_pipeline: str
    lds_bytes: int
    vgpr_budget: int
    dense_matrix: bool
    sparse_matrix: bool
    microscaling: bool
    cost_model_identity: str


@dataclass(frozen=True)
class AMDMatrixDtypeContract:
    storage: str
    scalar_vector: ContractState
    dense_matrix: ContractState
    sparse_matrix: ContractState
    matrix_format: str | None
    accumulators: tuple[str, ...]
    dense_shapes: tuple[tuple[int, int, int], ...]
    sparse_shapes: tuple[tuple[int, int, int], ...]
    requires_scale_operand: bool = False
    note: str = ""

    @property
    def isa_matrix_supported(self) -> bool:
        return self.dense_matrix not in {"unsupported", "unregistered"}


@dataclass(frozen=True)
class AMDMatrixInstruction:
    arch: AMDArch
    family: ArchitectureFamily
    path: MatrixPath
    mnemonic: str
    shape: tuple[int, int, int]
    a_dtype: str
    b_dtype: str
    accumulator: str
    matrix_format: str | None
    requires_scale_operand: bool
    compiler_state: ContractState


AMD_ARCHITECTURE_CONTRACTS: dict[AMDArch, AMDArchitectureContract] = {
    AMDArch.GFX_1151: AMDArchitectureContract(
        AMDArch.GFX_1151, "rdna35", "Strix Halo / Radeon 8060S", 32,
        "valu_wmma", 64 * 1024, 256, True, False, False, "gfx1151",
    ),
    AMDArch.GFX_1200: AMDArchitectureContract(
        AMDArch.GFX_1200, "rdna4",
        "Radeon RX 9060 XT / Radeon RX 9070 XT", 32,
        "valu_wmma", 64 * 1024, 256, True, True, False, "gfx1200",
    ),
    AMDArch.GFX_1250: AMDArchitectureContract(
        AMDArch.GFX_1250, "cdna5", "AMD Instinct MI455X", 32,
        "xdl_wmma", 320 * 1024, 1024, True, True, True, "gfx1250_mi455x",
    ),
    AMDArch.GFX_1251: AMDArchitectureContract(
        AMDArch.GFX_1251, "cdna5", "AMD Instinct MI430X", 32,
        "xdl_wmma", 320 * 1024, 1024, True, True, True, "gfx1251_mi430x",
    ),
}


def _empty_dtype_rows() -> dict[str, AMDMatrixDtypeContract]:
    return {
        dtype: AMDMatrixDtypeContract(
            dtype, "unregistered", "unsupported", "unsupported", None, (), (), ()
        )
        for dtype in canonical_dtypes() | planned_gated_dtypes()
    }


def _rows_gfx1151() -> dict[str, AMDMatrixDtypeContract]:
    rows = _empty_dtype_rows()
    rows.update({
        "fp64": AMDMatrixDtypeContract("fp64", "artifact_only", "unsupported", "unsupported", None, (), (), ()),
        "fp32": AMDMatrixDtypeContract("fp32", "ready", "unsupported", "unsupported", None, ("fp32",), (), (), note="Accumulator and scalar/vector format only."),
        "fp16": AMDMatrixDtypeContract("fp16", "ready", "ready", "unsupported", "f16", ("fp16", "fp32"), ((16, 16, 16),), ()),
        "bf16": AMDMatrixDtypeContract("bf16", "ready", "ready", "unsupported", "bf16", ("bf16", "fp32"), ((16, 16, 16),), ()),
        "int8": AMDMatrixDtypeContract("int8", "ready", "ready", "unsupported", "iu8", ("int32",), ((16, 16, 16),), ()),
        "int4": AMDMatrixDtypeContract("int4", "ready", "ready", "unsupported", "iu4", ("int32",), ((16, 16, 16),), ()),
        "int32": AMDMatrixDtypeContract("int32", "unregistered", "unsupported", "unsupported", None, ("int32",), (), (), note="Matrix accumulator and index ABI only."),
    })
    return rows


def _rows_gfx1200() -> dict[str, AMDMatrixDtypeContract]:
    rows = _empty_dtype_rows()
    rows.update({
        "fp64": AMDMatrixDtypeContract("fp64", "artifact_only", "unsupported", "unsupported", None, (), (), ()),
        "fp32": AMDMatrixDtypeContract("fp32", "ready", "unsupported", "unsupported", None, ("fp32",), (), ()),
        "fp16": AMDMatrixDtypeContract("fp16", "ready", "artifact_only", "artifact_only", "f16", ("fp16", "fp32"), ((16, 16, 16),), ((16, 16, 32),)),
        "bf16": AMDMatrixDtypeContract("bf16", "ready", "artifact_only", "artifact_only", "bf16", ("bf16", "fp32"), ((16, 16, 16),), ((16, 16, 32),)),
        "fp8_e4m3": AMDMatrixDtypeContract("fp8_e4m3", "artifact_only", "artifact_only", "artifact_only", "fp8", ("fp32",), ((16, 16, 16),), ((16, 16, 32),)),
        "fp8_e5m2": AMDMatrixDtypeContract("fp8_e5m2", "artifact_only", "artifact_only", "artifact_only", "bf8", ("fp32",), ((16, 16, 16),), ((16, 16, 32),)),
        "int8": AMDMatrixDtypeContract("int8", "ready", "artifact_only", "artifact_only", "iu8", ("int32",), ((16, 16, 16),), ((16, 16, 32),)),
        "int4": AMDMatrixDtypeContract("int4", "ready", "artifact_only", "artifact_only", "iu4", ("int32",), ((16, 16, 16), (16, 16, 32)), ((16, 16, 32), (16, 16, 64))),
        "int32": AMDMatrixDtypeContract("int32", "ready", "unsupported", "unsupported", None, ("int32",), (), ()),
    })
    return rows


def _rows_cdna5() -> dict[str, AMDMatrixDtypeContract]:
    rows = _empty_dtype_rows()
    rows.update({
        "fp64": AMDMatrixDtypeContract("fp64", "artifact_only", "unsupported", "unsupported", None, (), (), (), note="gfx1251 adds FP64 WMMA; it requires its own follow-up contract."),
        "fp32": AMDMatrixDtypeContract("fp32", "artifact_only", "artifact_only", "unsupported", "f32", ("fp32",), ((16, 16, 4),), ()),
        "fp16": AMDMatrixDtypeContract("fp16", "artifact_only", "artifact_only", "artifact_only", "f16", ("fp16", "fp32"), ((16, 16, 32),), ((16, 16, 64),)),
        "bf16": AMDMatrixDtypeContract("bf16", "artifact_only", "artifact_only", "artifact_only", "bf16", ("bf16", "fp32"), ((16, 16, 32),), ((16, 16, 64),)),
        "fp8_e4m3": AMDMatrixDtypeContract("fp8_e4m3", "artifact_only", "artifact_only", "artifact_only", "fp8", ("fp16", "fp32"), ((16, 16, 64), (16, 16, 128)), ((16, 16, 128),)),
        "fp8_e5m2": AMDMatrixDtypeContract("fp8_e5m2", "artifact_only", "artifact_only", "artifact_only", "bf8", ("fp16", "fp32"), ((16, 16, 64), (16, 16, 128)), ((16, 16, 128),)),
        "fp6_e2m3": AMDMatrixDtypeContract("fp6_e2m3", "planned_gated", "planned_gated", "unsupported", "f6", ("fp32",), ((16, 16, 128),), (), True),
        "fp6_e3m2": AMDMatrixDtypeContract("fp6_e3m2", "planned_gated", "planned_gated", "unsupported", "f6", ("fp32",), ((16, 16, 128),), (), True),
        "fp4_e2m1": AMDMatrixDtypeContract("fp4_e2m1", "planned_gated", "planned_gated", "unsupported", "f4", ("fp32",), ((16, 16, 128), (32, 16, 128)), (), True),
        "int8": AMDMatrixDtypeContract("int8", "artifact_only", "artifact_only", "artifact_only", "iu8", ("int32",), ((16, 16, 64),), ((16, 16, 128),)),
        "int4": AMDMatrixDtypeContract("int4", "artifact_only", "unsupported", "unsupported", None, (), (), (), note="CDNA5 has IU4 dot products but no IU4 WMMA/SWMMAC."),
        "int32": AMDMatrixDtypeContract("int32", "artifact_only", "unsupported", "unsupported", None, ("int32",), (), ()),
        "mxfp8": AMDMatrixDtypeContract("mxfp8", "planned_gated", "planned_gated", "unsupported", "f8", ("fp32",), ((16, 16, 128),), (), True),
        "mxfp6": AMDMatrixDtypeContract("mxfp6", "planned_gated", "planned_gated", "unsupported", "f6", ("fp32",), ((16, 16, 128),), (), True),
        "mxfp4": AMDMatrixDtypeContract("mxfp4", "planned_gated", "planned_gated", "unsupported", "f4", ("fp32",), ((16, 16, 128), (32, 16, 128)), (), True),
    })
    return rows


def _rows_cdna5_gfx1251() -> dict[str, AMDMatrixDtypeContract]:
    rows = _rows_cdna5()
    rows["fp64"] = AMDMatrixDtypeContract(
        "fp64", "artifact_only", "planned_gated", "unsupported", "f64",
        ("fp64",), ((16, 16, 4),), (),
        note="MI430X/gfx1251-specific FP64 WMMA; no gfx1250 inheritance.",
    )
    return rows


AMD_DTYPE_CONTRACTS: dict[AMDArch, dict[str, AMDMatrixDtypeContract]] = {
    AMDArch.GFX_1151: _rows_gfx1151(),
    AMDArch.GFX_1200: _rows_gfx1200(),
    AMDArch.GFX_1250: _rows_cdna5(),
    AMDArch.GFX_1251: _rows_cdna5_gfx1251(),
}


_FORMAT_TOKEN = {
    "fp16": "F16", "bf16": "BF16", "fp8_e4m3": "FP8",
    "fp8_e5m2": "BF8", "int8": "IU8", "int4": "IU4", "fp32": "F32",
    "fp64": "F64",
}


def amd_architecture_contract(arch: AMDArch) -> AMDArchitectureContract:
    try:
        return AMD_ARCHITECTURE_CONTRACTS[arch]
    except KeyError as exc:
        raise TesseraROCmTargetError(
            f"no cross-generation ISA contract registered for {arch.name}"
        ) from exc


def amd_dtype_contract(arch: AMDArch, storage: str) -> AMDMatrixDtypeContract:
    try:
        return AMD_DTYPE_CONTRACTS[arch][storage]
    except KeyError as exc:
        raise TesseraROCmTargetError(
            f"no datatype contract for {storage!r} on {arch.name}"
        ) from exc


def select_amd_matrix_instruction(
    arch: AMDArch,
    a_dtype: str,
    b_dtype: str | None = None,
    *,
    accumulator: str | None = None,
    sparse: bool = False,
    shape: tuple[int, int, int] | None = None,
    scaled: bool = False,
) -> AMDMatrixInstruction:
    """Select one exact ISA matrix operation, or fail closed.

    The returned mnemonic is carried as provenance into Schedule/Tile/Target
    metadata.  This selector never converts an unsupported format to another
    dtype and never transfers a sparse or scaled form between architectures.
    """
    b_dtype = b_dtype or a_dtype
    a = amd_dtype_contract(arch, a_dtype)
    b = amd_dtype_contract(arch, b_dtype)
    state_a = a.sparse_matrix if sparse else a.dense_matrix
    state_b = b.sparse_matrix if sparse else b.dense_matrix
    if state_a in {"unsupported", "unregistered"} or state_b in {"unsupported", "unregistered"}:
        raise TesseraROCmTargetError(
            f"ROCM_TILE_UNSUPPORTED_DTYPE: {arch.name} has no "
            f"{'sparse ' if sparse else ''}matrix path for {a_dtype} x {b_dtype}"
        )
    if scaled and not (a.requires_scale_operand or b.requires_scale_operand):
        raise TesseraROCmTargetError(
            f"ROCM_TILE_UNSUPPORTED_DTYPE: {arch.name} {a_dtype} x {b_dtype} "
            "does not use a matrix scale operand"
        )
    if (a.requires_scale_operand or b.requires_scale_operand) and not scaled:
        raise TesseraROCmTargetError(
            f"ROCM_TILE_UNSUPPORTED_DTYPE: {arch.name} {a_dtype} x {b_dtype} "
            "requires explicit scale lineage"
        )

    legal_shapes = set(a.sparse_shapes if sparse else a.dense_shapes)
    legal_shapes &= set(b.sparse_shapes if sparse else b.dense_shapes)
    if not legal_shapes:
        raise TesseraROCmTargetError(
            f"ROCM_TILE_UNSUPPORTED_DTYPE: no common shape for "
            f"{a_dtype} x {b_dtype} on {arch.name}"
        )
    selected_shape = shape or min(legal_shapes, key=lambda value: (value[2], value[0], value[1]))
    if selected_shape not in legal_shapes:
        raise TesseraROCmTargetError(
            f"ROCM_TILE_UNSUPPORTED_DTYPE: {selected_shape} is not legal for "
            f"{a_dtype} x {b_dtype} on {arch.name}; legal={sorted(legal_shapes)}"
        )
    common_acc = set(a.accumulators) & set(b.accumulators)
    if not common_acc:
        raise TesseraROCmTargetError(
            f"ROCM_TILE_UNSUPPORTED_DTYPE: no common accumulator for "
            f"{a_dtype} x {b_dtype} on {arch.name}"
        )
    selected_acc = accumulator or next(
        candidate for candidate in ("fp32", "int32", "fp64")
        if candidate in common_acc
    )
    if selected_acc not in common_acc:
        raise TesseraROCmTargetError(
            f"ROCM_TILE_UNSUPPORTED_DTYPE: {selected_acc} is not legal for "
            f"{a_dtype} x {b_dtype} on {arch.name}; legal={sorted(common_acc)}"
        )

    m, n, k = selected_shape
    acc_token = {
        "fp32": "F32", "fp64": "F64", "int32": "I32",
    }[selected_acc]
    path: MatrixPath = "swmmac" if sparse else "wmma"
    if scaled:
        input_token = "F8F6F4" if m == 16 else "F4"
        mnemonic = f"V_WMMA_SCALE_{acc_token}_{m}X{n}X{k}_{input_token}"
    else:
        a_token = _FORMAT_TOKEN[a_dtype]
        b_token = _FORMAT_TOKEN[b_dtype]
        suffix = a_token if a_token == b_token and a_token not in {"FP8", "BF8"} else f"{a_token}_{b_token}"
        mnemonic = f"V_{path.upper()}_{acc_token}_{m}X{n}X{k}_{suffix}"
    state: ContractState = (
        "planned_gated" if "planned_gated" in {state_a, state_b}
        else "artifact_only" if "artifact_only" in {state_a, state_b}
        else "ready"
    )
    return AMDMatrixInstruction(
        arch, amd_architecture_contract(arch).family, path, mnemonic,
        selected_shape, a_dtype, b_dtype, selected_acc, a.matrix_format,
        scaled, state,
    )


__all__ = [
    "AMDArchitectureContract", "AMDMatrixDtypeContract", "AMDMatrixInstruction",
    "AMD_ARCHITECTURE_CONTRACTS", "AMD_DTYPE_CONTRACTS",
    "amd_architecture_contract", "amd_dtype_contract",
    "select_amd_matrix_instruction",
]
