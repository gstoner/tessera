"""tessera.compiler.rocm_target — ROCm 7.2.3 target profile.

Mirrors the structure of ``gpu_target.py`` (NVIDIA) for the AMD/ROCm
backend.  Pinned to ROCm 7.2.3 + HIP 7.2.3 as the minimum AMD toolchain
under Sprint H-1 (2026-05-11).

Per-ISA feature matrix covers:
  - gfx94x (CDNA 3, MI300A / MI300X)
  - gfx950 (CDNA 4, MI325X / future)
  - gfx1100 (RDNA 3, prosumer; kept for completeness)
  - gfx1151 (RDNA 3.5, Strix Halo APU — Radeon 8060S / Ryzen AI Max+ 395)
  - gfx1200 (RDNA 4 / GFX12 consumer class)

Each ISA exposes a feature dict + a dtype set + matrix-instruction
variants.  ``rocm_feature_status(isa, name)`` queries individual flags;
``mfma_variants(isa)`` returns the set of MFMA instruction shapes a CDNA
arch can lower to, and ``wmma_variants(isa)`` returns the WMMA shapes an
RDNA arch can lower to (the two are mutually exclusive per arch).

The gfx1151 (RDNA 3.5) entry is grounded in the "RDNA3.5" Instruction Set
Architecture Reference Guide (AMD, 23-July-2024), §7.9 Wave Matrix Multiply
Accumulate: WMMA is VOP3P, tile 16x16x16, dtype combos F32<-F16, F32<-BF16,
F16<-F16, BF16<-BF16, I32<-IU8, I32<-IU4 — **no FP8/FP4 WMMA on RDNA 3.5**
(that is CDNA 4 / RDNA 4 only).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class AMDArch(IntEnum):
    """AMD GPU architecture identifiers under ROCm 7.2.3.

    Values encode ``int(gfx_id)`` so comparisons work for CDNA-family
    ordering: gfx1200 (RDNA 4) > gfx1100 (RDNA 3) and gfx950
    (CDNA 4) > gfx942 (CDNA 3 MI300X) > gfx940 (CDNA 3 MI300A) >
    gfx90a (CDNA 2 MI250).  CDNA/RDNA comparisons are ordering
    conveniences only; feature gates below remain architecture-specific.
    """

    GFX_90A = 90       # MI250 — CDNA 2 (kept for completeness)
    GFX_940 = 940      # MI300A — CDNA 3 unified
    GFX_942 = 942      # MI300X — CDNA 3 discrete
    GFX_950 = 950      # MI325X — CDNA 4
    GFX_1100 = 1100    # RDNA 3 prosumer (RX 7900-series)
    GFX_1151 = 1151    # RDNA 3.5 APU — Strix Halo (Radeon 8060S / Ryzen AI Max+ 395)
    GFX_1200 = 1200    # RDNA 4 / GFX12 prosumer


#: RDNA-family arches (wave32, WMMA matrix path).  Everything else is CDNA
#: (wave64, MFMA).  Used by lane-width and matrix-path gates.
_RDNA_ARCHES: frozenset[AMDArch] = frozenset({
    AMDArch.GFX_1100, AMDArch.GFX_1151, AMDArch.GFX_1200,
})


#: Target ROCm release that Tessera's AMD backend is built against.
TESSERA_TARGET_ROCM: str = "7.2.3"
TESSERA_TARGET_HIP: str = "7.2.3"
TESSERA_TARGET_RCCL_MIN: str = "2.22"     # RCCL bundled with ROCm 7.2.3
TESSERA_TARGET_ROCBLAS_MIN: str = "5.0.0"
TESSERA_TARGET_MIOPEN_MIN: str = "3.5.0"


# Shared (LDS) memory budgets in bytes per CU.
_LDS_BYTES: dict[AMDArch, int] = {
    AMDArch.GFX_90A:  65536,
    AMDArch.GFX_940:  65536,
    AMDArch.GFX_942:  65536,
    AMDArch.GFX_950:  163840,   # CDNA 4 doubles LDS
    AMDArch.GFX_1100: 65536,
    AMDArch.GFX_1151: 65536,    # RDNA 3.5 — 128 KiB per WGP (2 CUs); 64 KiB/CU view
    AMDArch.GFX_1200: 65536,
}


# Maximum waves per CU for each architecture.
_MAX_WAVES: dict[AMDArch, int] = {
    AMDArch.GFX_90A:  32,
    AMDArch.GFX_940:  32,
    AMDArch.GFX_942:  32,
    AMDArch.GFX_950:  32,
    AMDArch.GFX_1100: 16,
    AMDArch.GFX_1151: 16,       # RDNA 3.5 wave32 occupancy (same as RDNA 3)
    AMDArch.GFX_1200: 16,
}


# Per-arch dtype matrix accepted by the ROCm 7.2.3 backend.  Canonical
# Tessera dtype spellings only (validated by `tessera.dtype.canonicalize_dtype`).
_ROCM_DTYPES: dict[AMDArch, frozenset[str]] = {
    AMDArch.GFX_90A: frozenset({
        "fp64", "fp32", "bf16", "fp16", "int8",
    }),
    AMDArch.GFX_940: frozenset({
        "fp64", "fp32", "bf16", "fp16",
        "fp8_e4m3", "fp8_e5m2",
        "int8",
    }),
    AMDArch.GFX_942: frozenset({
        "fp64", "fp32", "bf16", "fp16",
        "fp8_e4m3", "fp8_e5m2",
        "int8",
    }),
    AMDArch.GFX_950: frozenset({
        # CDNA 4 adds OCP FP4/FP6 + AMD MX-formats (planned/gated per
        # tessera.dtype rules — registry entries that reference these
        # must declare metadata.dtype_status='planned_gated').
        "fp64", "fp32", "bf16", "fp16",
        "fp8_e4m3", "fp8_e5m2",
        "fp6_e2m3", "fp6_e3m2",
        "fp4_e2m1",
        "int8",
    }),
    AMDArch.GFX_1100: frozenset({
        "fp32", "bf16", "fp16", "int8",
    }),
    AMDArch.GFX_1151: frozenset({
        # RDNA 3.5 (Strix Halo).  WMMA executable surface per the ISA §7.9
        # Table 33: F16, BF16, IU8 (int8).  IU4 (int4) is architecturally
        # present but stays planned-gated (no first-class packed-4 storage
        # policy yet — same stance as gfx1200), so it is omitted from the
        # executable dtype set.  Notably NO FP8 here (unlike gfx1200).
        "fp32", "bf16", "fp16", "int8",
    }),
    AMDArch.GFX_1200: frozenset({
        # GFX12 / RDNA 4 rocWMMA-class surface.  Keep this as an
        # architecture capability matrix, not a native-execution claim:
        # Tessera's ROCm backend remains artifact_only until HIP execution
        # validation lands.
        "fp32", "bf16", "fp16",
        "fp8_e4m3", "fp8_e5m2",
        "int8", "int32",
        "int4",
    }),
}


# Per-arch feature flag matrix.  Status values:
#   "ready"          — fully supported under ROCm 7.2.3
#   "tba"            — present in the architecture, not yet exposed by HIP 7.2.3
#   "not_supported"  — architecturally unavailable
_ROCM_7_2_FEATURES: dict[AMDArch, dict[str, str]] = {
    AMDArch.GFX_90A: {
        # MI250 — CDNA 2; baseline MFMA only.
        "mfma":                "ready",
        "mfma_f8":             "not_supported",
        "mfma_xf32":           "not_supported",
        "mfma_f4":             "not_supported",
        "mfma_f6":             "not_supported",
        "wmma_f16":            "not_supported",
        "wmma_bf16":           "not_supported",
        "wmma_f8":             "not_supported",
        "lds_async_copy":      "not_supported",
        "buffer_load_lds":     "ready",
        "global_load_lds":     "not_supported",
        "cluster_mode":        "not_supported",
        "xnack":               "ready",
        "sram_ecc":            "ready",
    },
    AMDArch.GFX_940: {
        # MI300A — CDNA 3 unified APU.
        "mfma":                "ready",
        "mfma_f8":             "ready",
        "mfma_xf32":           "ready",
        "mfma_f4":             "not_supported",
        "mfma_f6":             "not_supported",
        "wmma_f16":            "not_supported",
        "wmma_bf16":           "not_supported",
        "wmma_f8":             "not_supported",
        "lds_async_copy":      "ready",
        "buffer_load_lds":     "ready",
        "global_load_lds":     "ready",
        "cluster_mode":        "not_supported",
        "xnack":               "ready",
        "sram_ecc":            "ready",
    },
    AMDArch.GFX_942: {
        # MI300X — CDNA 3 discrete.
        "mfma":                "ready",
        "mfma_f8":             "ready",
        "mfma_xf32":           "ready",
        "mfma_f4":             "not_supported",
        "mfma_f6":             "not_supported",
        "wmma_f16":            "not_supported",
        "wmma_bf16":           "not_supported",
        "wmma_f8":             "not_supported",
        "lds_async_copy":      "ready",
        "buffer_load_lds":     "ready",
        "global_load_lds":     "ready",
        "cluster_mode":        "not_supported",
        "xnack":               "ready",
        "sram_ecc":            "ready",
    },
    AMDArch.GFX_950: {
        # MI325X — CDNA 4; adds MX-format MFMA + cluster mode.
        "mfma":                "ready",
        "mfma_f8":             "ready",
        "mfma_xf32":           "ready",
        "mfma_f4":             "ready",
        "mfma_f6":             "ready",
        "wmma_f16":            "not_supported",
        "wmma_bf16":           "not_supported",
        "wmma_f8":             "not_supported",
        "lds_async_copy":      "ready",
        "buffer_load_lds":     "ready",
        "global_load_lds":     "ready",
        "cluster_mode":        "ready",
        "xnack":               "ready",
        "sram_ecc":            "ready",
    },
    AMDArch.GFX_1100: {
        # RDNA 3 — WMMA only; no MFMA on the prosumer line.
        "mfma":                "not_supported",
        "mfma_f8":             "not_supported",
        "mfma_xf32":           "not_supported",
        "mfma_f4":             "not_supported",
        "mfma_f6":             "not_supported",
        "wmma_f16":            "ready",
        "wmma_bf16":           "ready",
        "wmma_f8":             "tba",
        "lds_async_copy":      "not_supported",
        "buffer_load_lds":     "ready",
        "global_load_lds":     "not_supported",
        "cluster_mode":        "not_supported",
        "xnack":               "not_supported",
        "sram_ecc":            "not_supported",
    },
    AMDArch.GFX_1151: {
        # RDNA 3.5 (Strix Halo APU).  WMMA only, no MFMA — same matrix
        # surface as RDNA 3 (gfx1100).  Grounded in the RDNA3.5 ISA §7.9
        # Table 33: WMMA combos are F16/BF16/IU8/IU4 — there is **no FP8
        # WMMA instruction** on RDNA 3.5, so wmma_f8 is not_supported
        # (this is the load-bearing distinction from gfx1200, which has it).
        "mfma":                "not_supported",
        "mfma_f8":             "not_supported",
        "mfma_xf32":           "not_supported",
        "mfma_f4":             "not_supported",
        "mfma_f6":             "not_supported",
        "wmma_f16":            "ready",
        "wmma_bf16":           "ready",
        "wmma_f8":             "not_supported",
        "lds_async_copy":      "not_supported",
        "buffer_load_lds":     "ready",
        "global_load_lds":     "not_supported",
        "cluster_mode":        "not_supported",
        # APU with truly unified LPDDR5x; xnack/managed-memory behaviour on
        # gfx1151 is left conservative (not asserted ready) until validated
        # on real silicon under a shipping ROCm.
        "xnack":               "not_supported",
        "sram_ecc":            "not_supported",
    },
    AMDArch.GFX_1200: {
        # RDNA 4 / GFX12 — WMMA/rocWMMA-class target.  Public ROCm docs
        # list GFX12 matrix datatypes/instructions including FP8/BF8 WMMA
        # variants, FP16/BF16 SWMMAC, and I32 IU8/IU4 accumulators.
        # Tessera maps FP8/BF8 to fp8_e4m3/fp8_e5m2 and keeps IU4 as
        # planned-gated int4 until a first-class unsigned packed-4 storage
        # policy exists.  Tessera models the executable
        # compiler surface conservatively: WMMA-class features are ready
        # for artifact planning, MFMA/CDNA features stay unavailable.
        "mfma":                "not_supported",
        "mfma_f8":             "not_supported",
        "mfma_xf32":           "not_supported",
        "mfma_f4":             "not_supported",
        "mfma_f6":             "not_supported",
        "wmma_f16":            "ready",
        "wmma_bf16":           "ready",
        "wmma_f8":             "ready",
        "lds_async_copy":      "not_supported",
        "buffer_load_lds":     "ready",
        "global_load_lds":     "not_supported",
        "cluster_mode":        "not_supported",
        "xnack":               "not_supported",
        "sram_ecc":            "not_supported",
    },
}


# MFMA instruction shape table per arch.  Each entry is a tuple of
# (M, N, K, K_blocks) shapes that the backend can lower to.  The full
# expanded matrix lives in `mfma_table.inc` (C++ side); this is the
# Python summary that capability queries consult.
_MFMA_VARIANTS: dict[AMDArch, frozenset[tuple[int, int, int, int]]] = {
    AMDArch.GFX_90A: frozenset({
        # Classic CDNA 2 shapes
        (32, 32, 8, 1),
        (16, 16, 16, 1),
    }),
    AMDArch.GFX_940: frozenset({
        # CDNA 3 — adds f8 + xf32 variants
        (32, 32, 8, 1),    # bf16/fp16
        (32, 32, 16, 1),   # fp8 (i_k=16)
        (16, 16, 16, 1),   # bf16/fp16
        (16, 16, 32, 1),   # fp8 (i_k=32)
        (32, 32, 4, 1),    # xf32 (tf32 equivalent on AMD)
        (16, 16, 8, 1),    # xf32
    }),
    AMDArch.GFX_942: frozenset({
        (32, 32, 8, 1),
        (32, 32, 16, 1),
        (16, 16, 16, 1),
        (16, 16, 32, 1),
        (32, 32, 4, 1),
        (16, 16, 8, 1),
    }),
    AMDArch.GFX_950: frozenset({
        # CDNA 4 — adds f4/f6 lanes
        (32, 32, 8, 1),
        (32, 32, 16, 1),
        (32, 32, 32, 1),   # fp4 (i_k=32)
        (16, 16, 16, 1),
        (16, 16, 32, 1),
        (16, 16, 64, 1),   # fp4 (i_k=64)
        (32, 32, 4, 1),
        (16, 16, 8, 1),
    }),
    AMDArch.GFX_1100: frozenset(),  # RDNA 3 has WMMA, not MFMA
    AMDArch.GFX_1151: frozenset(),  # RDNA 3.5 has WMMA, not MFMA
    AMDArch.GFX_1200: frozenset(),  # RDNA 4 has WMMA, not MFMA
}


# WMMA instruction shape table per RDNA arch.  Each entry is a tuple of
# (M, N, K) shapes the backend can lower to.  RDNA's Wave Matrix Multiply
# Accumulate (VOP3P `V_WMMA_*`) is the RDNA analogue of CDNA's MFMA; the two
# are mutually exclusive (a CDNA arch has MFMA + empty WMMA, an RDNA arch has
# WMMA + empty MFMA).  RDNA 3 / 3.5 expose a single 16x16x16 tile across all
# supported dtype combos (RDNA3.5 ISA §7.9, Table 33).
_WMMA_VARIANTS: dict[AMDArch, frozenset[tuple[int, int, int]]] = {
    AMDArch.GFX_90A:  frozenset(),  # CDNA — MFMA only
    AMDArch.GFX_940:  frozenset(),
    AMDArch.GFX_942:  frozenset(),
    AMDArch.GFX_950:  frozenset(),
    AMDArch.GFX_1100: frozenset({(16, 16, 16)}),   # RDNA 3 WMMA
    AMDArch.GFX_1151: frozenset({(16, 16, 16)}),   # RDNA 3.5 WMMA (ISA §7.9)
    # RDNA 4 dense WMMA shapes, grounded in the RDNA4 ISA §7.12 Table 41:
    # 16x16x16 (F16/BF16/FP8/BF8/IU8/IU4) + 16x16x32 (IU4 large-K).  RDNA 4 also
    # adds FP8/BF8 WMMA (the load-bearing gain over RDNA 3.5), a SWMMAC 4:2 sparse
    # family (16x16x32 / 16x16x64, A-matrix expanded — a separate sparse-op set
    # not modeled here), and WMMA load-transpose (§11.6).  No FP4 (E2M1) WMMA.
    AMDArch.GFX_1200: frozenset({(16, 16, 16), (16, 16, 32)}),
}


# Per-arch HIP/HIPCC compile-target strings under ROCm 7.2.3.
_ROCM_ARCH_STRINGS: dict[AMDArch, str] = {
    AMDArch.GFX_90A:  "gfx90a",
    AMDArch.GFX_940:  "gfx940",
    AMDArch.GFX_942:  "gfx942",
    AMDArch.GFX_950:  "gfx950",
    AMDArch.GFX_1100: "gfx1100",
    AMDArch.GFX_1151: "gfx1151",
    AMDArch.GFX_1200: "gfx1200",
}


class TesseraROCmTargetError(Exception):
    """Raised when an ROCmTargetProfile has invalid or unsupported settings."""


@dataclass
class ROCmTargetProfile:
    """Describes the ROCm/AMD target for a ``@jit(target=...)`` function.

    Attributes:
        arch              : AMDArch enum (gfx94x / gfx950 / gfx1100 / gfx1200)
        waves_per_cu      : Wave count per CU (must be in [1, _MAX_WAVES])
        lds_bytes         : Override LDS budget; None = use generation default
        pipeline_stages   : Software pipeline depth (>=1)
        prefer_inline_asm : Emit raw AMDGCN inline asm rather than hip ops

    Capability gates (checked at lowering time):
        .supports_mfma          → CDNA family (gfx90a+, excludes RDNA)
        .supports_mfma_f8       → gfx940+
        .supports_mfma_f4       → gfx950+
        .supports_wmma          → RDNA 3/4 (gfx1100 / gfx1200)
        .lds_capacity_bytes     → generation-specific default
    """

    arch: AMDArch = AMDArch.GFX_942
    waves_per_cu: int = 4
    lds_bytes: Optional[int] = None
    pipeline_stages: int = 2
    prefer_inline_asm: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.arch, AMDArch):
            try:  # type: ignore[unreachable]
                self.arch = AMDArch(int(self.arch))
            except (ValueError, KeyError):
                raise TesseraROCmTargetError(
                    f"Unknown AMD arch {self.arch!r}.  Use an AMDArch enum "
                    f"member, e.g. AMDArch.GFX_942."
                )
        if self.waves_per_cu < 1:
            raise TesseraROCmTargetError(
                f"waves_per_cu must be >= 1, got {self.waves_per_cu}"
            )
        max_waves = _MAX_WAVES[self.arch]
        if self.waves_per_cu > max_waves:
            raise TesseraROCmTargetError(
                f"waves_per_cu={self.waves_per_cu} exceeds {self.arch.name} "
                f"max of {max_waves}"
            )
        if self.pipeline_stages < 1:
            raise TesseraROCmTargetError(
                f"pipeline_stages must be >= 1, got {self.pipeline_stages}"
            )

    # ── Capability queries ──────────────────────────────────────────────

    @property
    def supports_mfma(self) -> bool:
        return rocm_feature_status(self.arch, "mfma") == "ready"

    @property
    def supports_mfma_f8(self) -> bool:
        return rocm_feature_status(self.arch, "mfma_f8") == "ready"

    @property
    def supports_mfma_xf32(self) -> bool:
        return rocm_feature_status(self.arch, "mfma_xf32") == "ready"

    @property
    def supports_mfma_f4(self) -> bool:
        return rocm_feature_status(self.arch, "mfma_f4") == "ready"

    @property
    def supports_mfma_f6(self) -> bool:
        return rocm_feature_status(self.arch, "mfma_f6") == "ready"

    @property
    def supports_wmma(self) -> bool:
        return any(
            rocm_feature_status(self.arch, name) == "ready"
            for name in ("wmma_f16", "wmma_bf16", "wmma_f8")
        )

    @property
    def supports_lds_async_copy(self) -> bool:
        return rocm_feature_status(self.arch, "lds_async_copy") == "ready"

    @property
    def supports_cluster_mode(self) -> bool:
        return rocm_feature_status(self.arch, "cluster_mode") == "ready"

    @property
    def lds_capacity_bytes(self) -> int:
        if self.lds_bytes is not None:
            return self.lds_bytes
        return _LDS_BYTES[self.arch]

    @property
    def waves_per_simd(self) -> int:
        return self.waves_per_cu

    @property
    def threads_per_wave(self) -> int:
        # AMD wavefronts are 64 lanes on CDNA, 32 on RDNA.
        return 32 if self.arch in _RDNA_ARCHES else 64

    @property
    def rocm_features(self) -> frozenset[str]:
        """Set of all ROCm 7.2.3 features marked ``ready`` for this arch."""
        return rocm_feature_set(self.arch)

    @property
    def hipcc_arch(self) -> str:
        """``hipcc --offload-arch=...`` string under ROCm 7.2.3."""
        return rocm_arch_string(self.arch)

    @property
    def dtype_set(self) -> frozenset[str]:
        return _ROCM_DTYPES[self.arch]

    @property
    def mfma_shapes(self) -> frozenset[tuple[int, int, int, int]]:
        return _MFMA_VARIANTS[self.arch]

    @property
    def wmma_shapes(self) -> frozenset[tuple[int, int, int]]:
        """WMMA (M, N, K) tiles for this arch (empty on CDNA — see mfma_shapes)."""
        return _WMMA_VARIANTS[self.arch]

    @property
    def is_rdna(self) -> bool:
        return self.arch in _RDNA_ARCHES


def rocm_feature_status(arch: AMDArch, feature: str) -> str:
    """Return ROCm 7.2.3 status for a per-arch feature."""
    return _ROCM_7_2_FEATURES[arch][feature]


def rocm_feature_set(arch: AMDArch) -> frozenset[str]:
    return frozenset(
        name for name, status in _ROCM_7_2_FEATURES[arch].items()
        if status == "ready"
    )


def rocm_arch_string(arch: AMDArch) -> str:
    return _ROCM_ARCH_STRINGS[arch]


def mfma_variants(arch: AMDArch) -> frozenset[tuple[int, int, int, int]]:
    """Return MFMA instruction shapes (M, N, K, K_blocks) for ``arch``."""
    return _MFMA_VARIANTS[arch]


def wmma_variants(arch: AMDArch) -> frozenset[tuple[int, int, int]]:
    """Return WMMA instruction shapes (M, N, K) for an RDNA ``arch`` (empty on CDNA)."""
    return _WMMA_VARIANTS[arch]


__all__ = [
    "AMDArch",
    "ROCmTargetProfile",
    "TesseraROCmTargetError",
    "TESSERA_TARGET_ROCM",
    "TESSERA_TARGET_HIP",
    "TESSERA_TARGET_RCCL_MIN",
    "TESSERA_TARGET_ROCBLAS_MIN",
    "TESSERA_TARGET_MIOPEN_MIN",
    "rocm_feature_status",
    "rocm_feature_set",
    "rocm_arch_string",
    "mfma_variants",
    "wmma_variants",
]
