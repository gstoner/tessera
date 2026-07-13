"""Unified schedule descriptor consumed by ROCm generated kernels.

This joins the previously separate macro-tile, register-budget, software
pipeline, LDS-layout, and ownership decisions.  It is a compiler contract: a
descriptor is serializable into Target-IR attributes and those attributes are
validated by the generated-kernel pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ownership_topology import OwnershipTopology
from .rocm_lds import SoftwarePipeline, select_lds_layout
from .rocm_target import AMDArch, ROCmTargetProfile
from .rocm_tiling import TileCandidate, TileShape, estimate_vgpr_usage


_ARCH_BY_NAME = {a.name.lower().replace("_", ""): a for a in AMDArch}
_DTYPE = {
    "f16": "fp16", "float16": "fp16", "bf16": "bf16",
    "int8": "int8", "i8": "int8",
    # gfx1151's int4 WMMA ABI uses int8 containers; charge the container's
    # staging registers rather than inventing an unsupported scalar model.
    "int4": "int8", "i4": "int8",
}


def parse_arch(name: str) -> AMDArch:
    key = name.lower().replace("_", "")
    try:
        return _ARCH_BY_NAME[key]
    except KeyError as exc:
        raise ValueError(f"unsupported ROCm schedule arch {name!r}") from exc


@dataclass(frozen=True)
class ROCmScheduleDescriptor:
    arch: str
    dtype: str
    instruction_tile: tuple[int, int, int]
    macro_tile: tuple[int, int]
    waves_per_cu: int
    pipeline_stages: int
    lds_layout: str
    ownership: OwnershipTopology
    vgpr_estimate: int
    source: str

    def __post_init__(self) -> None:
        if any(v < 1 for v in (*self.instruction_tile, *self.macro_tile)):
            raise ValueError("instruction and macro tile extents must be positive")
        if self.pipeline_stages < 1 or self.waves_per_cu < 1:
            raise ValueError("pipeline_stages and waves_per_cu must be positive")

    @property
    def mt(self) -> int:
        return self.macro_tile[0]

    @property
    def nt(self) -> int:
        return self.macro_tile[1]

    def target_ir_attrs(self) -> dict[str, Any]:
        return {
            "mt": self.mt,
            "nt": self.nt,
            "schedule_arch": self.arch,
            "schedule_pipeline_stages": self.pipeline_stages,
            "schedule_lds_layout": self.lds_layout,
            "schedule_ownership": self.ownership.value,
            "schedule_vgpr_estimate": self.vgpr_estimate,
            "schedule_source": self.source,
        }

    def cache_key(self) -> tuple[Any, ...]:
        return tuple(self.target_ir_attrs().values()) + (self.dtype, self.instruction_tile)


def select_rocm_gemm_schedule(
    m: int,
    n: int,
    k: int,
    *,
    dtype: str = "f16",
    arch: str = "gfx1151",
) -> ROCmScheduleDescriptor:
    """Return the measured gfx1151 production schedule plus modeled evidence."""
    if min(m, n, k) < 1:
        raise ValueError(f"GEMM dimensions must be positive, got {(m, n, k)}")
    amd_arch = parse_arch(arch)
    profile = ROCmTargetProfile(arch=amd_arch, waves_per_cu=4, pipeline_stages=2)
    mt, nt = ((3, 4) if min(m, n, k) >= 1024 else (2, 4))
    model_dtype = _DTYPE.get(dtype, dtype)
    candidate = TileCandidate(
        TileShape(16 * mt, 16 * nt, 16), model_dtype, double_buffer=True
    )
    vgprs = estimate_vgpr_usage(candidate, profile)
    lds = select_lds_layout(amd_arch, global_to_lds=False, inner_dim=16)
    lds_kind = str(lds.as_metadata_dict()["strategy"])
    pipeline = SoftwarePipeline(profile.pipeline_stages)
    return ROCmScheduleDescriptor(
        arch=arch,
        dtype=dtype,
        instruction_tile=(16, 16, 16),
        macro_tile=(mt, nt),
        waves_per_cu=profile.waves_per_cu,
        pipeline_stages=pipeline.stages,
        lds_layout=lds_kind,
        ownership=OwnershipTopology.WAVE,
        vgpr_estimate=vgprs,
        source="gfx1151 measured GEMM macro-tile + ROCm budget models",
    )


__all__ = ["ROCmScheduleDescriptor", "parse_arch", "select_rocm_gemm_schedule"]
