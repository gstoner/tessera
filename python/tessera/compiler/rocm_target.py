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
    # gfx1250/1251 — the "v2" mods/reuse WMMA ABI (K-doubled 16x16x32, native
    # bfloat).  Family designation NOT asserted (no public ISA consulted); grounded
    # only by `llc` (LLVM 22 AMDGPU) + LLVM IntrinsicsAMDGPU.td.  See rocdl_emit.py.
    GFX_1250 = 1250
    GFX_1251 = 1251


#: wave32 / WMMA-matrix-path arches (vs CDNA wave64 / MFMA).  Used by lane-width and
#: matrix-path gates.  "RDNA" historically, now also the gfx125x WMMA targets —
#: ``is_rdna`` here means "wave32 + WMMA path", an operational gate, not a strict
#: family claim (gfx125x wave32 is `llc`-grounded; its family is not asserted).
_RDNA_ARCHES: frozenset[AMDArch] = frozenset({
    AMDArch.GFX_1100, AMDArch.GFX_1151, AMDArch.GFX_1200,
    AMDArch.GFX_1250, AMDArch.GFX_1251,
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
    # gfx1250/1251 — PROVISIONAL (no gfx1250 ISA consulted; mirrors the RDNA 3/4
    # 64 KiB/CU value).  Not used in any execution path; the grounded gfx1250 surface
    # is the WMMA emitter (rocdl_emit.py).  Replace once a gfx1250 ISA is available.
    AMDArch.GFX_1250: 65536,
    AMDArch.GFX_1251: 65536,
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
    AMDArch.GFX_1250: 16,       # PROVISIONAL (no gfx1250 ISA; mirrors RDNA wave32)
    AMDArch.GFX_1251: 16,       # PROVISIONAL
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
    AMDArch.GFX_1250: frozenset({
        # gfx1250/1251 — the v2 mods/reuse WMMA ABI.  f16/bf16 (16x16x32) are
        # `llc`-verified (rocdl_emit.py); fp8_e4m3/e5m2 (16x16x64/128, the ModsC
        # ABI) + int8 are from LLVM IntrinsicsAMDGPU.td.  int4 stays planned-gated.
        "fp32", "bf16", "fp16",
        "fp8_e4m3", "fp8_e5m2",
        "int8",
    }),
    AMDArch.GFX_1251: frozenset({
        "fp32", "bf16", "fp16",
        "fp8_e4m3", "fp8_e5m2",
        "int8",
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
    AMDArch.GFX_1250: {
        # gfx1250/1251 — WMMA-class (v2 mods/reuse ABI).  WMMA float + fp8 flags
        # are grounded (`llc` for f16/bf16 16x16x32; LLVM IntrinsicsAMDGPU.td for
        # the fp8 16x16x64/128 forms).  Everything else is "tba": NOT grounded —
        # no gfx1250 ISA was consulted, so async-copy / cluster / ECC readiness is
        # genuinely unknown rather than asserted.
        "mfma":                "not_supported",
        "mfma_f8":             "not_supported",
        "mfma_xf32":           "not_supported",
        "mfma_f4":             "not_supported",
        "mfma_f6":             "not_supported",
        "wmma_f16":            "ready",
        "wmma_bf16":           "ready",
        "wmma_f8":             "ready",
        "lds_async_copy":      "tba",
        "buffer_load_lds":     "tba",
        "global_load_lds":     "tba",
        "cluster_mode":        "tba",
        "xnack":               "tba",
        "sram_ecc":            "tba",
    },
    AMDArch.GFX_1251: {
        "mfma":                "not_supported",
        "mfma_f8":             "not_supported",
        "mfma_xf32":           "not_supported",
        "mfma_f4":             "not_supported",
        "mfma_f6":             "not_supported",
        "wmma_f16":            "ready",
        "wmma_bf16":           "ready",
        "wmma_f8":             "ready",
        "lds_async_copy":      "tba",
        "buffer_load_lds":     "tba",
        "global_load_lds":     "tba",
        "cluster_mode":        "tba",
        "xnack":               "tba",
        "sram_ecc":            "tba",
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
    AMDArch.GFX_1250: frozenset(),  # gfx1250 has WMMA (v2 ABI), not MFMA
    AMDArch.GFX_1251: frozenset(),
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
    # gfx1250/1251 — the v2 mods/reuse WMMA ABI.  The K is DOUBLED: f16/bf16 are
    # 16x16x32 (`llc`-verified, rocdl_emit.py), and the fp8/bf8 forms are 16x16x64 /
    # 16x16x128 (LLVM IntrinsicsAMDGPU.td, ModsC ABI).  bf16 is native `<16 x bfloat>`.
    AMDArch.GFX_1250: frozenset({(16, 16, 32), (16, 16, 64), (16, 16, 128)}),
    AMDArch.GFX_1251: frozenset({(16, 16, 32), (16, 16, 64), (16, 16, 128)}),
}


# ── Per-arch register budgets (per-lane) ────────────────────────────────────
# Architectural vector/accumulator register file budgets, expressed PER LANE.
# These are the registers a single work-item (lane) may hold live before the
# compiler must spill to scratch — the budget the AMD Gluon GEMM tutorial
# identified as the dominant perf lever (double-buffering regressed −73% by
# spilling past it; slicing the output tile to fit was the real fix).
#
#   CDNA (gfx90a/940/942/950): 256 VGPR + 256 AGPR per lane = 512 combined.
#     The AGPRs are the matrix-core accumulator registers; the combined 512
#     matches the Gluon "512-VGPR budget" framing on gfx950.
#   RDNA / wave32 (gfx1100/1151/1200/1250/1251): 256 VGPR + 0 AGPR.
#     RDNA has no separate AGPR file; the accumulator lives in the VGPR file.
_VGPR_BUDGET: dict[AMDArch, int] = {
    AMDArch.GFX_90A:  256,
    AMDArch.GFX_940:  256,
    AMDArch.GFX_942:  256,
    AMDArch.GFX_950:  256,
    AMDArch.GFX_1100: 256,
    AMDArch.GFX_1151: 256,
    AMDArch.GFX_1200: 256,
    AMDArch.GFX_1250: 256,
    AMDArch.GFX_1251: 256,
}

#: Accumulator (matrix-core) register budget per lane.  Non-zero only on CDNA,
#: which has a dedicated AGPR file feeding MFMA; RDNA/wave32 has none (0).
_AGPR_BUDGET: dict[AMDArch, int] = {
    AMDArch.GFX_90A:  256,
    AMDArch.GFX_940:  256,
    AMDArch.GFX_942:  256,
    AMDArch.GFX_950:  256,
    AMDArch.GFX_1100: 0,
    AMDArch.GFX_1151: 0,
    AMDArch.GFX_1200: 0,
    AMDArch.GFX_1250: 0,
    AMDArch.GFX_1251: 0,
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
    AMDArch.GFX_1250: "gfx1250",   # `llc`-accepted target string (LLVM 22 AMDGPU)
    AMDArch.GFX_1251: "gfx1251",
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
    def fp8_semantics(self) -> str:
        """FP8 bit-encoding family for this arch: ``"fnuz"`` (CDNA 3) /
        ``"ocp"`` (CDNA 4, RDNA 4, gfx125x) / ``"none"``.  The same canonical
        ``fp8_e4m3``/``fp8_e5m2`` dtype encodes different bits across these — a
        correctness-critical, arch-keyed distinction (see ``_FP8_SEMANTICS``)."""
        return _FP8_SEMANTICS[self.arch]

    @property
    def lds_capacity_bytes(self) -> int:
        if self.lds_bytes is not None:
            return self.lds_bytes
        return _LDS_BYTES[self.arch]

    @property
    def vgpr_budget(self) -> int:
        """Vector general-purpose registers available per lane on this arch.

        This is the architectural VGPR file budget a single work-item may hold
        live before the compiler must spill.  It is the dominant tiling-fit
        lever per the AMD Gluon GEMM tutorial (see ``rocm_tiling``)."""
        return _VGPR_BUDGET[self.arch]

    @property
    def agpr_budget(self) -> int:
        """Accumulator (matrix-core) registers per lane.  Non-zero only on CDNA
        (dedicated AGPR file feeding MFMA); ``0`` on RDNA/wave32 arches."""
        return _AGPR_BUDGET[self.arch]

    @property
    def total_reg_budget(self) -> int:
        """Combined per-lane register budget (``vgpr_budget + agpr_budget``).

        On CDNA this is 512 (256 VGPR + 256 AGPR — the Gluon "512-VGPR budget");
        on RDNA/wave32 it is 256 (no separate AGPR file)."""
        return self.vgpr_budget + self.agpr_budget

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
    def is_wave32(self) -> bool:
        """True when the arch uses a 32-lane wavefront + the WMMA matrix path (vs
        CDNA wave64 + MFMA). This is the precise operational predicate; ``is_rdna``
        is the historical alias (gfx125x is wave32/WMMA but not asserted-RDNA)."""
        return self.arch in _RDNA_ARCHES

    @property
    def is_rdna(self) -> bool:
        """Alias of :attr:`is_wave32` — kept for back-compat. NOTE: a literal RDNA
        family claim is *not* implied for gfx125x (see ``_RDNA_ARCHES``)."""
        return self.is_wave32


def rocm_feature_status(arch: AMDArch, feature: str) -> str:
    """Return the ROCm 7.2.3 status for a per-arch feature. Raises ``KeyError`` with
    a clear message for an unknown feature name."""
    try:
        return _ROCM_7_2_FEATURES[arch][feature]
    except KeyError as e:
        raise KeyError(
            f"unknown ROCm feature {feature!r} for {arch.name} "
            f"(known: {sorted(_ROCM_7_2_FEATURES[arch])})") from e


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


# ── FP8 numeric semantics per arch ──────────────────────────────────────────
# The SAME canonical fp8 dtype (fp8_e4m3 / fp8_e5m2) has DIFFERENT bit semantics
# across AMD generations — a correctness-critical, arch-keyed distinction that
# hipBLASLt encodes as the FNUZ-vs-OCP type split:
#   "fnuz" — CDNA 3 (gfx940/942, MI300): E4M3FNUZ / E5M2FNUZ (AMD "finite,
#            unsigned-zero, no-Inf" encoding; NaN/Inf bit patterns differ from OCP).
#   "ocp"  — CDNA 4 (gfx950, MI350), RDNA 4 (gfx1200), gfx125x: OCP standard
#            E4M3 / E5M2 (the encoding NVIDIA Blackwell + the OCP MX spec use).
#   "none" — arch has no FP8 matrix path at all.
# A registry/manifest "complete FP8 kernel" claim is silently arch-ambiguous
# without this flag — same op, different bits per target.
_FP8_SEMANTICS: dict[AMDArch, str] = {
    AMDArch.GFX_90A:  "none",
    AMDArch.GFX_940:  "fnuz",
    AMDArch.GFX_942:  "fnuz",
    AMDArch.GFX_950:  "ocp",
    AMDArch.GFX_1100: "none",
    AMDArch.GFX_1151: "none",   # RDNA 3.5 has no FP8 WMMA at all
    AMDArch.GFX_1200: "ocp",
    AMDArch.GFX_1250: "ocp",
    AMDArch.GFX_1251: "ocp",
}

#: The concrete LLVM/MLIR dtype-flavor spelling per (semantics, canonical fp8).
_FP8_FLAVOR: dict[tuple[str, str], str] = {
    ("fnuz", "fp8_e4m3"): "e4m3fnuz",
    ("fnuz", "fp8_e5m2"): "e5m2fnuz",
    ("ocp", "fp8_e4m3"):  "e4m3",
    ("ocp", "fp8_e5m2"):  "e5m2",
}


def fp8_semantics(arch: AMDArch) -> str:
    """FP8 bit-encoding family for ``arch``: ``"fnuz"`` / ``"ocp"`` / ``"none"``."""
    return _FP8_SEMANTICS[arch]


def fp8_dtype_flavor(arch: AMDArch, dtype: str) -> str:
    """Concrete fp8 flavor spelling for ``arch``.

    ``(gfx942, fp8_e4m3) -> "e4m3fnuz"`` (CDNA 3 FNUZ);
    ``(gfx950, fp8_e4m3) -> "e4m3"`` (CDNA 4 OCP).  Raises a stable diagnostic
    when the arch has no FP8 path, or ``dtype`` is not an fp8 dtype — never
    guesses an ambiguous flavor.
    """
    sem = _FP8_SEMANTICS[arch]
    if sem == "none":
        raise TesseraROCmTargetError(
            f"{arch.name} has no FP8 matrix path; there is no fp8 flavor to name "
            f"(fp8_semantics={sem!r}).")
    if dtype not in ("fp8_e4m3", "fp8_e5m2"):
        raise TesseraROCmTargetError(
            f"{dtype!r} is not an fp8 dtype (expected fp8_e4m3 / fp8_e5m2).")
    return _FP8_FLAVOR[(sem, dtype)]


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
    "fp8_semantics",
    "fp8_dtype_flavor",
]
