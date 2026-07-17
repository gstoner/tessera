"""Architecture-owned physical fragment descriptors for NVIDIA matrix cores.

Portable Tile IR names logical A/B/accumulator roles.  This module owns the
physical warp/register ABI selected after the exact CUDA architecture is known.
It deliberately does not reuse AMD wave/VGPR layouts or treat consumer
Blackwell as a datacenter-Blackwell/TMEM target.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NvidiaFragmentError(ValueError):
    """An unsupported exact-architecture fragment request."""


class FragmentFamily(str, Enum):
    MMA_SYNC = "mma_sync"
    MMA_SYNC_BLOCK_SCALE = "mma_sync_block_scale"


class RegisterPacking(str, Enum):
    PAIR_F16 = "pair_f16"
    SCALAR_F32 = "scalar_f32"
    PACKED_X4_I8 = "packed_x4_i8"
    PACKED_X8_E2M1 = "packed_x8_e2m1"


@dataclass(frozen=True)
class FragmentLayoutDescriptor:
    arch: str
    family: FragmentFamily
    shape: tuple[int, int, int]
    dtype: str
    acc_dtype: str
    warp_size: int
    a_elements_per_lane: int
    b_elements_per_lane: int
    a_registers_per_lane: int
    b_registers_per_lane: int
    accumulator_elements_per_lane: int
    accumulator_registers_per_lane: int
    input_packing: RegisterPacking
    accumulator_mapping: str
    instruction_family: str
    scale_dtype: str | None = None
    scale_vector: str | None = None
    typed_tile_materialization_ready: bool = True

    def __post_init__(self) -> None:
        m, n, k = self.shape
        if self.warp_size != 32:
            raise NvidiaFragmentError("sm_120a matrix fragments require one warp")
        if self.warp_size * self.a_elements_per_lane != m * k:
            raise NvidiaFragmentError("A fragment does not cover one logical tile")
        if self.warp_size * self.b_elements_per_lane != k * n:
            raise NvidiaFragmentError("B fragment does not cover one logical tile")
        if self.warp_size * self.accumulator_elements_per_lane != m * n:
            raise NvidiaFragmentError("accumulator fragment does not cover one tile")
        if self.family is FragmentFamily.MMA_SYNC_BLOCK_SCALE:
            if not self.scale_dtype or not self.scale_vector:
                raise NvidiaFragmentError("block-scaled fragments require scale ABI")
        elif self.scale_dtype or self.scale_vector:
            raise NvidiaFragmentError("ordinary mma.sync fragments cannot carry scales")

    @property
    def block_scaled(self) -> bool:
        return self.family is FragmentFamily.MMA_SYNC_BLOCK_SCALE

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "arch": self.arch,
            "family": self.family.value,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "acc_dtype": self.acc_dtype,
            "warp_size": self.warp_size,
            "a_elements_per_lane": self.a_elements_per_lane,
            "b_elements_per_lane": self.b_elements_per_lane,
            "a_registers_per_lane": self.a_registers_per_lane,
            "b_registers_per_lane": self.b_registers_per_lane,
            "accumulator_elements_per_lane": self.accumulator_elements_per_lane,
            "accumulator_registers_per_lane": self.accumulator_registers_per_lane,
            "input_packing": self.input_packing.value,
            "accumulator_mapping": self.accumulator_mapping,
            "instruction_family": self.instruction_family,
            "scale_dtype": self.scale_dtype,
            "scale_vector": self.scale_vector,
            "block_scaled": self.block_scaled,
            "typed_tile_materialization_ready": self.typed_tile_materialization_ready,
        }


_ALIASES = {
    "fp16": "f16", "f16": "f16",
    "bf16": "bf16",
    "tf32": "tf32",
    "fp8_e4m3": "e4m3", "e4m3": "e4m3",
    "fp8_e5m2": "e5m2", "e5m2": "e5m2",
    "int8": "s8", "s8": "s8",
    "nvfp4": "nvfp4", "fp4_e2m1": "nvfp4",
}


def select_sm120_fragment_layout(
    dtype: str,
    shape: tuple[int, int, int],
    acc_dtype: str | None = None,
) -> FragmentLayoutDescriptor:
    """Select the exact consumer-Blackwell warp fragment ABI.

    NVFP4 is intentionally a separate block-scaled family.  Portable Tile IR
    carries its two logical scale operands; NVIDIA owns their physical per-lane
    packing after the exact architecture is selected.
    """
    canonical = _ALIASES.get(dtype.lower())
    if canonical is None:
        raise NvidiaFragmentError(
            f"no proven sm_120a fragment exists for dtype {dtype!r}")
    m, n, k = shape
    if (m, n) != (16, 8):
        raise NvidiaFragmentError(
            "sm_120a warp fragments require "
            f"m16n8; got {shape}")

    table = {
        "f16": (16, "f32", RegisterPacking.PAIR_F16, 4, 2,
                 "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
        "bf16": (16, "f32", RegisterPacking.PAIR_F16, 4, 2,
                  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"),
        "tf32": (8, "f32", RegisterPacking.SCALAR_F32, 4, 2,
                  "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"),
        "e4m3": (32, "f32", RegisterPacking.PACKED_X4_I8, 4, 2,
                  "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"),
        "e5m2": (32, "f32", RegisterPacking.PACKED_X4_I8, 4, 2,
                  "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"),
        "s8": (32, "s32", RegisterPacking.PACKED_X4_I8, 4, 2,
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"),
    }
    if canonical == "nvfp4":
        if k != 64:
            raise NvidiaFragmentError(
                "sm_120a NVFP4 fragments require m16n8k64")
        if acc_dtype not in (None, "f32"):
            raise NvidiaFragmentError("sm_120a NVFP4 fragments require f32 accumulation")
        return FragmentLayoutDescriptor(
            "sm_120a", FragmentFamily.MMA_SYNC_BLOCK_SCALE, shape, canonical,
            "f32", 32, 32, 16, 4, 2, 4, 4,
            RegisterPacking.PACKED_X8_E2M1,
            "mma_m16n8_f32_four_registers_per_lane",
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
            scale_dtype="ue4m3", scale_vector="4X",
            typed_tile_materialization_ready=True)

    legal_k, acc, packing, a_regs, b_regs, instruction = table[canonical]
    if k != legal_k:
        raise NvidiaFragmentError(
            f"{canonical} fragments require "
            f"m16n8k{legal_k}; got {shape}")
    requested_acc = acc_dtype or acc
    if canonical == "f16" and requested_acc == "f16":
        return FragmentLayoutDescriptor(
            "sm_120a", FragmentFamily.MMA_SYNC, shape, canonical, "f16", 32,
            m * k // 32, k * n // 32, a_regs, b_regs, 4, 2, packing,
            "mma_m16n8_f16_four_elements_two_registers_per_lane",
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16")
    if requested_acc != acc:
        raise NvidiaFragmentError(
            f"{canonical} fragments do not support {requested_acc} accumulation")
    return FragmentLayoutDescriptor(
        "sm_120a", FragmentFamily.MMA_SYNC, shape, canonical, acc, 32,
        m * k // 32, k * n // 32, a_regs, b_regs, 4, 4, packing,
        "mma_m16n8_f32_four_registers_per_lane", instruction)


__all__ = [
    "FragmentFamily", "RegisterPacking", "FragmentLayoutDescriptor",
    "NvidiaFragmentError", "select_sm120_fragment_layout",
]
