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
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class ISA(IntEnum):
    """NVIDIA SM generation, ordered so comparisons work: SM_90 >= SM_80."""
    SM_80  = 80   # A100 — Ampere; WMMA only
    SM_86  = 86   # RTX 30xx — Ampere consumer
    SM_89  = 89   # RTX 40xx — Ada Lovelace
    SM_90  = 90   # H100 / GH200 — Hopper; WGMMA + TMA available
    SM_100 = 100  # B100 / GB200 — Blackwell datacenter (GB100)
    SM_120 = 120  # RTX 50-series — Blackwell consumer (GB20x), compute capability 12.0


# Shared memory capacities in bytes per SM for each generation.
_SMEM_BYTES: dict[ISA, int] = {
    ISA.SM_80:  166912,   # A100: 163 KB
    ISA.SM_86:  100352,   # RTX 3090: 98 KB
    ISA.SM_89:  100352,   # RTX 4090: 98 KB
    ISA.SM_90:  233472,   # H100: 228 KB
    ISA.SM_100: 262144,   # B100: 256 KB
    # CC 12.0 = 100 KB/SM (99 KB/block), per the CUDA Programming Guide
    # compute-capabilities appendix Table 31 — NOT the 256 KB datacenter sm_100 SM,
    # and NOT the 128 KB unified data cache (Table 32).  Confirmed on-silicon
    # 2026-06-25 (RTX 5070 Ti, CUDA 13.3): sharedMemPerMultiprocessor == 102400.
    # NB: per-block dynamic-smem opt-in maxes at 101376 (99 KiB), so any per-block
    # budget / cudaFuncAttributeMaxDynamicSharedMemorySize must cap below this value.
    ISA.SM_120: 102400,   # RTX 50-series: 100 KB/SM (consumer Blackwell, CC 12.0)
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

# Tensor-core operand types AND math modes per ISA.
#
# NOTE: ``tf32`` in this table is a **math mode**, not a storage dtype.
# Per ``docs/reference/tessera_tensor_attributes.md`` (normative,
# 2026-05-11), TF32 must be modelled as ``math_mode="tf32"`` on an
# ``fp32`` tensor via ``numeric_policy``.  It appears in this dict
# because the hardware tensor-core can consume fp32 operands under TF32
# math; the dict mixes storage dtypes and math modes for capability
# reporting only.  ``tessera.dtype.canonicalize_dtype("tf32")`` raises
# ``TesseraDtypeError`` so the storage-dtype path stays clean.
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
        # CC 12.0 Tensor Core input types (CUDA Programming Guide compute-
        # capabilities appendix, Table 33): FP64, TF32, BF16, FP16, FP8, FP6,
        # FP4, INT8, INT4 — all "Yes" for 12.x.  (int4 is planned_gated under
        # Tessera's dtype policy but the hardware supports it.)
        "nvfp4", "fp4_e2m1", "fp64", "tf32", "bf16", "fp16",
        "fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2", "int8", "int4",
    }),
}


# ─────────────────────────────────────────────────────────────────────────────
# Sprint G-1 — CUDA capability matrix (2026-05-11; bumped to CUDA 13.3 2026-06-18).
#
# Tessera targets CUDA 13.3 as the NVIDIA toolchain.  This block captures the
# per-SM feature flag matrix that the lowering passes gate on: WGMMA variants on
# Hopper, tcgen05 / TMEM on Blackwell, cluster launch, TMA swizzle modes, async
# mbarrier transaction counts, and the BF16/FP8/FP6/FP4 variants each generation
# accepts.  The PTX ISA version reported is aligned to the CUDA 13.3 driver/runtime
# pair.  (The 13.2→13.3 bump moved the toolkit + driver-min + PTX-ISA pins only;
# the per-SM ready/tba feature readiness was NOT re-evaluated for 13.3 — that is a
# separate grounded task.)
#
# Reference: CUDA Toolkit 13.3 release notes + PTX ISA 9.3.  The "tba" status
# (below) is reserved for a future SM whose ISA guidance is still pending — the
# matrix is currently fully resolved (every modeled SM flag is "ready" or
# "not_supported"; there are no live "tba" entries today).
# ─────────────────────────────────────────────────────────────────────────────

#: Target CUDA Toolkit release that Tessera's NVIDIA backend is built against.
TESSERA_TARGET_CUDA_TOOLKIT: str = "13.3"
TESSERA_TARGET_CUDA_DRIVER_MIN: str = "610.43.02"  # Min driver for CUDA 13.3
TESSERA_TARGET_PTX_ISA: str = "9.3"                # PTX ISA bundled with 13.3
# NCCL_MIN is a *minimum* floor, not the bundled version.  CUDA 13.3 bundles NCCL
# 2.30.7, but NCCL is backward-compatible so the required floor stays 2.22 (kept in
# sync with RCCL 2.22 on the ROCm track); raising the collective minimum is a
# separate, deliberate decision.
TESSERA_TARGET_NCCL_MIN: str = "2.22"


# Per-SM feature flag matrix.  Each flag is one of:
#   "ready"          — feature is functional under CUDA 13.3 on this ISA
#   "tba"            — present in the architecture but not enabled in 13.3
#   "not_supported"  — architecturally unavailable
_CUDA_13_3_FEATURES: dict[ISA, dict[str, str]] = {
    ISA.SM_80: {
        # Ampere — WMMA only; no WGMMA / TMA / clusters.
        "wmma":                    "ready",
        "wgmma":                   "not_supported",
        "wgmma_sparse":            "not_supported",
        "tma":                     "not_supported",
        "tma_swizzle_128b":        "not_supported",
        "cluster_launch":          "not_supported",
        "mbarrier":                "ready",
        "mbarrier_arrive_tx":      "not_supported",
        "tcgen05":                 "not_supported",
        "tcgen05_pair":            "not_supported",
        "tmem":                    "not_supported",
        "cp_async":                "ready",
        "cp_async_bulk":           "not_supported",
        "block_scaled_mma":        "not_supported",
        "async_proxy_fence":       "not_supported",
    },
    ISA.SM_86: {
        "wmma":                    "ready",
        "wgmma":                   "not_supported",
        "wgmma_sparse":            "not_supported",
        "tma":                     "not_supported",
        "tma_swizzle_128b":        "not_supported",
        "cluster_launch":          "not_supported",
        "mbarrier":                "ready",
        "mbarrier_arrive_tx":      "not_supported",
        "tcgen05":                 "not_supported",
        "tcgen05_pair":            "not_supported",
        "tmem":                    "not_supported",
        "cp_async":                "ready",
        "cp_async_bulk":           "not_supported",
        "block_scaled_mma":        "not_supported",
        "async_proxy_fence":       "not_supported",
    },
    ISA.SM_89: {
        # Ada Lovelace — WMMA + cp.async, no WGMMA/TMA.
        "wmma":                    "ready",
        "wgmma":                   "not_supported",
        "wgmma_sparse":            "not_supported",
        "tma":                     "not_supported",
        "tma_swizzle_128b":        "not_supported",
        "cluster_launch":          "not_supported",
        "mbarrier":                "ready",
        "mbarrier_arrive_tx":      "not_supported",
        "tcgen05":                 "not_supported",
        "tcgen05_pair":            "not_supported",
        "tmem":                    "not_supported",
        "cp_async":                "ready",
        "cp_async_bulk":           "not_supported",
        "block_scaled_mma":        "not_supported",
        "async_proxy_fence":       "not_supported",
    },
    ISA.SM_90: {
        # Hopper — full WGMMA + TMA + thread-block clusters + mbarrier
        # transaction-count under CUDA 13.3.
        "wmma":                    "ready",
        "wgmma":                   "ready",
        "wgmma_sparse":            "ready",
        "tma":                     "ready",
        "tma_swizzle_128b":        "ready",
        "cluster_launch":          "ready",
        "mbarrier":                "ready",
        "mbarrier_arrive_tx":      "ready",
        "tcgen05":                 "not_supported",
        "tcgen05_pair":            "not_supported",
        "tmem":                    "not_supported",
        "cp_async":                "ready",
        "cp_async_bulk":           "ready",
        "block_scaled_mma":        "not_supported",
        "async_proxy_fence":       "ready",
    },
    ISA.SM_100: {
        # Blackwell — adds tcgen05, TMEM, block-scaled MMA, CTA pairs.
        "wmma":                    "ready",
        "wgmma":                   "ready",
        "wgmma_sparse":            "ready",
        "tma":                     "ready",
        "tma_swizzle_128b":        "ready",
        "cluster_launch":          "ready",
        "mbarrier":                "ready",
        "mbarrier_arrive_tx":      "ready",
        "tcgen05":                 "ready",
        "tcgen05_pair":            "ready",
        "tmem":                    "ready",
        "cp_async":                "ready",
        "cp_async_bulk":           "ready",
        "block_scaled_mma":        "ready",
        "async_proxy_fence":       "ready",
    },
    ISA.SM_120: {
        # Blackwell CONSUMER (RTX 50-series, GB20x; compile target sm_120a).
        # CRITICAL: consumer Blackwell is NOT a superset of datacenter sm_100.
        # It does **not** have tcgen05 / TMEM (those are sm_100a only), and
        # like all Blackwell it does **not** have Hopper's wgmma.  Its FP4 /
        # block-scaled path goes through warp-level `mma.sync.aligned...
        # block_scale` (E2M1 + block scaling), not `tcgen05.mma`.  Grounded in
        # NVIDIA/cutlass#2800 (BlockScaledMmaOp restricts FP4 to sm_100a, blocks
        # sm_120) + modular#5707 ("tcgen05 not supported" on sm_120) + the
        # SM120 mma.sync FP4 fragment-layout forum thread.
        "wmma":                    "ready",
        "wgmma":                   "not_supported",   # Hopper sm_90a only
        "wgmma_sparse":            "not_supported",
        "tma":                     "ready",
        "tma_swizzle_128b":        "ready",
        "cluster_launch":          "ready",
        "mbarrier":                "ready",
        "mbarrier_arrive_tx":      "ready",
        "tcgen05":                 "not_supported",   # datacenter sm_100a only
        "tcgen05_pair":            "not_supported",   # datacenter sm_100a only
        "tmem":                    "not_supported",   # datacenter sm_100a only
        "cp_async":                "ready",
        "cp_async_bulk":           "ready",
        "block_scaled_mma":        "ready",           # FP4 via mma.sync.block_scale
        "async_proxy_fence":       "ready",
    },
}


# Per-SM nvcc / ptxas compile-target arch strings under CUDA 13.3.
# These are passed to ``nvcc -arch=...`` and ``ptxas --gpu-name=...``.
_CUDA_13_3_ARCH_STRINGS: dict[ISA, str] = {
    ISA.SM_80:  "sm_80",
    ISA.SM_86:  "sm_86",
    ISA.SM_89:  "sm_89",
    ISA.SM_90:  "sm_90a",   # Hopper architectural variant
    ISA.SM_100: "sm_100a",  # Blackwell architectural variant
    ISA.SM_120: "sm_120a",  # Blackwell consumer arch-specific (FP4 mma.sync block-scale)
}


def cuda_feature_status(isa: ISA, feature: str) -> str:
    """Return the CUDA 13.3 status for a per-SM feature.

    Values: ``"ready"`` | ``"tba"`` | ``"not_supported"``.  Unknown
    feature names raise ``KeyError``.
    """
    return _CUDA_13_3_FEATURES[isa][feature]


def cuda_arch_string(isa: ISA) -> str:
    """Return the ``nvcc -arch=...`` string for ``isa`` under CUDA 13.3."""
    return _CUDA_13_3_ARCH_STRINGS[isa]


def cuda_feature_set(isa: ISA) -> frozenset[str]:
    """Return the set of features that are ``ready`` for ``isa``."""
    return frozenset(
        name for name, status in _CUDA_13_3_FEATURES[isa].items()
        if status == "ready"
    )


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
            try:  # type: ignore[unreachable]
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

    # These queries consult the authoritative per-SM feature matrix
    # (``_CUDA_13_3_FEATURES``) — NOT a coarse ``isa >= SM_x`` ordering.  The
    # ordering is wrong for consumer Blackwell **sm_120**, which is *not* a
    # superset of datacenter sm_100: it has no Hopper ``wgmma`` and no
    # ``tcgen05``/TMEM (its matrix path is warp-level ``mma.sync``).  Sourcing
    # every capability from the matrix keeps one source of truth.

    @property
    def supports_wgmma(self) -> bool:
        """True when the ISA has Hopper ``wgmma.mma_async`` (SM_90 / SM_100 only —
        consumer Blackwell sm_120 does NOT; its matrix path is ``mma.sync``)."""
        return cuda_feature_status(self.isa, "wgmma") == "ready"

    @property
    def supports_tma(self) -> bool:
        """True when the ISA has the ``cp.async.bulk.tensor`` TMA path."""
        return cuda_feature_status(self.isa, "tma") == "ready"

    @property
    def supports_mbarrier(self) -> bool:
        """True for the Hopper+ async *transaction* barrier
        (``mbarrier.arrive.expect_tx``).  Basic named mbarrier exists earlier but
        is a different contract; this property means the transaction-count form."""
        return cuda_feature_status(self.isa, "mbarrier_arrive_tx") == "ready"

    @property
    def supports_async_transaction_barrier(self) -> bool:
        """Alias for Hopper+ mbarrier transaction-count support."""
        return self.supports_mbarrier

    @property
    def supports_tcgen05(self) -> bool:
        """True for datacenter Blackwell **sm_100** Tensor Core gen-5 (``tcgen05``).
        Consumer sm_120 is False (tcgen05 is sm_100a-only)."""
        return cuda_feature_status(self.isa, "tcgen05") == "ready"

    @property
    def supports_tmem(self) -> bool:
        """True for datacenter Blackwell **sm_100** Tensor Memory accumulators.
        Consumer sm_120 is False."""
        return cuda_feature_status(self.isa, "tmem") == "ready"

    @property
    def supports_cta_pairs(self) -> bool:
        """True when CTA-pair (``tcgen05_pair``) scheduling is available — sm_100
        only; consumer sm_120 is False."""
        return cuda_feature_status(self.isa, "tcgen05_pair") == "ready"

    @property
    def supports_block_scaled_mma(self) -> bool:
        """True for block-scaled MMA dtypes (NVFP4/FP6).  sm_100 via ``tcgen05``,
        sm_120 via ``mma.sync.block_scale`` — both report ready in the matrix."""
        return cuda_feature_status(self.isa, "block_scaled_mma") == "ready"

    @property
    def supports_wgmma_sparse(self) -> bool:
        """Sparse WGMMA variants (SM_90 / SM_100 under CUDA 13.3)."""
        return cuda_feature_status(self.isa, "wgmma_sparse") == "ready"

    @property
    def supports_tma_swizzle_128b(self) -> bool:
        """128-byte TMA swizzle modes (Hopper+ under CUDA 13.3)."""
        return cuda_feature_status(self.isa, "tma_swizzle_128b") == "ready"

    @property
    def supports_cluster_launch(self) -> bool:
        """Thread-block cluster launch (Hopper+ under CUDA 13.3)."""
        return cuda_feature_status(self.isa, "cluster_launch") == "ready"

    @property
    def supports_mbarrier_arrive_tx(self) -> bool:
        """mbarrier transaction-count arrive (Hopper+ under CUDA 13.3)."""
        return cuda_feature_status(self.isa, "mbarrier_arrive_tx") == "ready"

    @property
    def supports_tcgen05_pair(self) -> bool:
        """Paired tcgen05 MMA contracts (Blackwell+ under CUDA 13.3)."""
        return cuda_feature_status(self.isa, "tcgen05_pair") == "ready"

    @property
    def supports_cp_async_bulk(self) -> bool:
        """cp.async.bulk variants (Hopper+ under CUDA 13.3)."""
        return cuda_feature_status(self.isa, "cp_async_bulk") == "ready"

    @property
    def supports_async_proxy_fence(self) -> bool:
        """fence.proxy.async PTX (Hopper+ under CUDA 13.3)."""
        return cuda_feature_status(self.isa, "async_proxy_fence") == "ready"

    @property
    def cuda_features(self) -> frozenset[str]:
        """Set of all CUDA 13.3 features marked ``ready`` for this ISA."""
        return cuda_feature_set(self.isa)

    @property
    def nvcc_arch(self) -> str:
        """``nvcc -arch=...`` string for this profile under CUDA 13.3."""
        return cuda_arch_string(self.isa)

    @property
    def runtime_arch(self) -> str:
        """CUDA runtime architecture string for artifact metadata/builds."""
        if self.isa == ISA.SM_120:
            return "sm_120"
        if self.isa >= ISA.SM_100:
            return "sm_100a"
        if self.isa >= ISA.SM_90:
            return "sm_90a"
        return f"sm_{int(self.isa)}"

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
            f"mbarrier={self.supports_mbarrier}, "
            f"tcgen05={self.supports_tcgen05}, tmem={self.supports_tmem})"
        )
