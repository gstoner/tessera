"""
tessera.compiler.gpu_target — GPU target profile for @jit(target=...).

Phase 3: gates WGMMA (SM_90+) vs WMMA fallback, TMA availability,
and shared memory budget.

Usage:
    from tessera.compiler.gpu_target import GPUTargetProfile, ISA

    @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4))
    def flash_attn_fwd(Q, K, V):
        ...
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class ISA(IntEnum):
    """NVIDIA SM generation, ordered so comparisons work: SM_90 >= SM_80."""
    SM_80  = 80   # A100 — Ampere; WMMA only
    SM_86  = 86   # RTX 30xx — Ampere consumer
    SM_89  = 89   # RTX 40xx — Ada Lovelace
    SM_90  = 90   # H100 / GH200 — Hopper; WGMMA + TMA available
    SM_100 = 100  # B100 / GB200 — Blackwell
    SM_120 = 120  # Rubin placeholder until NVIDIA publishes final CC numbering


# Shared memory capacities in bytes per SM for each generation.
_SMEM_BYTES: dict[ISA, int] = {
    ISA.SM_80:  166912,   # A100: 163 KB
    ISA.SM_86:  100352,   # RTX 3090: 98 KB
    ISA.SM_89:  100352,   # RTX 4090: 98 KB
    ISA.SM_90:  233472,   # H100: 228 KB
    ISA.SM_100: 262144,   # B100: 256 KB
    ISA.SM_120: 262144,   # Rubin: preliminary placeholder
}

# Maximum warps per CTA for each generation.
_MAX_WARPS: dict[ISA, int] = {
    ISA.SM_80:  32,
    ISA.SM_86:  32,
    ISA.SM_89:  32,
    ISA.SM_90:  32,   # 128 threads / warpgroup = 4 warps; max 8 warpgroups
    ISA.SM_100: 32,
    ISA.SM_120: 32,
}


_BASE_CUDA_CORE_DTYPES = frozenset({
    "fp64",
    "fp32",
    "int32",
    "fp16",
    "bf16",
})

_TENSOR_CORE_DTYPES: dict[ISA, frozenset[str]] = {
    ISA.SM_80: frozenset({
        "fp64", "tf32", "bf16", "fp16", "int8",
    }),
    ISA.SM_86: frozenset({
        "tf32", "bf16", "fp16", "int8",
    }),
    ISA.SM_89: frozenset({
        "tf32", "bf16", "fp16", "int8",
    }),
    ISA.SM_90: frozenset({
        "fp64", "tf32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8",
    }),
    ISA.SM_100: frozenset({
        "fp64", "tf32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2",
        "fp6_e2m3", "fp6_e3m2", "fp4_e2m1", "nvfp4", "int8",
    }),
    ISA.SM_120: frozenset({
        "nvfp4", "fp4_e2m1", "fp64", "tf32", "bf16", "fp16",
        "fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2", "int8",
    }),
}


class TesseraTargetError(Exception):
    """Raised when a GPUTargetProfile has invalid or unsupported settings."""
    pass


@dataclass
class GPUTargetProfile:
    """
    Describes the GPU target for a @jit-decorated function.

    Attributes:
        isa              : SM generation enum. Governs WGMMA/TMA availability.
        warps_per_cta    : warp count per CTA (default 4; must be power of 2).
        shared_mem_bytes : override shared memory budget; None = use SM default.
        prefer_ptx       : emit raw PTX inline asm rather than NVGPU dialect ops.
        pipeline_stages  : software pipeline stages for double-buffering (≥1).

    Key capability gates (checked at lowering time):
        .supports_wgmma  → isa >= SM_90
        .supports_tma    → isa >= SM_90
        .max_smem_bytes  → generation-specific default
    """

    isa: ISA = ISA.SM_90
    warps_per_cta: int = 4
    shared_mem_bytes: Optional[int] = None
    prefer_ptx: bool = True
    pipeline_stages: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.isa, ISA):
            try:
                self.isa = ISA(int(self.isa))
            except (ValueError, KeyError):
                raise TesseraTargetError(
                    f"Unknown ISA value {self.isa!r}. "
                    f"Use an ISA enum member, e.g. ISA.SM_90."
                )
        if self.warps_per_cta < 1 or (self.warps_per_cta & (self.warps_per_cta - 1)) != 0:
            raise TesseraTargetError(
                f"warps_per_cta must be a power of 2, got {self.warps_per_cta}"
            )
        max_warps = _MAX_WARPS[self.isa]
        if self.warps_per_cta > max_warps:
            raise TesseraTargetError(
                f"warps_per_cta={self.warps_per_cta} exceeds SM {self.isa.value} "
                f"maximum of {max_warps}"
            )
        if self.pipeline_stages < 1:
            raise TesseraTargetError(
                f"pipeline_stages must be >= 1, got {self.pipeline_stages}"
            )

    # ── Capability queries ────────────────────────────────────────────────────

    @property
    def supports_wgmma(self) -> bool:
        """True for SM_90+ (Hopper). Enables wgmma.mma_async PTX path."""
        return self.isa >= ISA.SM_90

    @property
    def supports_tma(self) -> bool:
        """True for SM_90+ (Hopper). Enables cp.async.bulk.tensor TMA path."""
        return self.isa >= ISA.SM_90

    @property
    def supports_mbarrier(self) -> bool:
        """True for SM_90+ (Hopper). Enables async transaction barriers."""
        return self.isa >= ISA.SM_90

    @property
    def supports_async_transaction_barrier(self) -> bool:
        """Alias for Hopper+ mbarrier transaction-count support."""
        return self.supports_mbarrier

    @property
    def tensor_core_dtypes(self) -> frozenset[str]:
        """Tensor Core dtype names accepted by the target profile."""
        return _TENSOR_CORE_DTYPES[self.isa]

    @property
    def cuda_core_dtypes(self) -> frozenset[str]:
        """CUDA-core scalar dtype names for the target profile."""
        return _BASE_CUDA_CORE_DTYPES

    def supports_tensor_core_dtype(self, dtype: str) -> bool:
        """Return True when dtype is listed for Tensor Core lowering."""
        return dtype in self.tensor_core_dtypes

    @property
    def max_smem_bytes(self) -> int:
        """Effective shared memory limit for this target."""
        if self.shared_mem_bytes is not None:
            return self.shared_mem_bytes
        return _SMEM_BYTES[self.isa]

    @property
    def threads_per_cta(self) -> int:
        return self.warps_per_cta * 32

    @property
    def sm_version(self) -> int:
        return int(self.isa)

    # ── Serialisation for MLIR attribute emission ─────────────────────────────

    def to_mlir_attr(self) -> str:
        """Return a tessera.target MLIR attribute string for IR emission."""
        return (
            f'{{sm = {self.sm_version} : i32, '
            f'warps = {self.warps_per_cta} : i32, '
            f'smem = {self.max_smem_bytes} : i64, '
            f'pipeline_stages = {self.pipeline_stages} : i32}}'
        )

    def __repr__(self) -> str:
        return (
            f"GPUTargetProfile(isa={self.isa.name}, "
            f"warps_per_cta={self.warps_per_cta}, "
            f"smem={self.max_smem_bytes // 1024}KB, "
            f"wgmma={self.supports_wgmma}, tma={self.supports_tma}, "
            f"mbarrier={self.supports_mbarrier})"
        )
