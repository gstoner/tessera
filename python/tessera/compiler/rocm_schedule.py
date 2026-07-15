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
    # ROCM-6 Phase-0 promotion (2026-07-14): nine interleaved HIP-event
    # trials, five candidate tiles, full square/rectangular/ragged/dtype/
    # epilogue oracles, and assembler VGPR/spill evidence.  Retain the old 3x4
    # row wherever a replacement failed the >=3% paired-median and >=75%
    # interleaved win-rate gate.
    aligned = (m % 16 == 0 and n % 16 == 0 and k % 16 == 0)
    square = m == n == k
    if square and aligned:
        if m >= 4096:
            mt, nt = 4, 4
        elif m >= 2560:
            mt, nt = 3, 4       # 3072: 4x4 gained only 2.6%; retain
        elif m >= 1280:
            mt, nt = 2, 4       # 1536/2048: spill-bounded 2x4 wins
        elif m >= 1024:
            mt, nt = ((2, 4) if dtype in ("int4", "i4") else (4, 4))
        else:
            mt, nt = 2, 4
    elif aligned and m >= 256 and k >= 2048 and n >= 4 * m:
        mt, nt = 4, 4           # wide projection and MLP rows
    elif aligned and m >= 1024 and n >= 4 * m:
        mt, nt = 4, 4           # wide rows with K=1024/2048
    elif aligned and min(m, n, k) >= 4096:
        mt, nt = 4, 4
    else:
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
        source=("gfx1151 ROCM-6 interleaved schedule matrix v1 + ROCm "
                "assembler resource and budget models"),
    )


__all__ = ["ROCmScheduleDescriptor", "parse_arch", "select_rocm_gemm_schedule"]
