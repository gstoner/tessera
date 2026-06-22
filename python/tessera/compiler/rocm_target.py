"""tessera.compiler.rocm_target — ROCm 7.2.4 target profile.

Mirrors the structure of ``gpu_target.py`` (NVIDIA) for the AMD/ROCm
backend.  Pinned to ROCm 7.2.4 + HIP 7.2.4 as the minimum AMD toolchain
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
    """AMD GPU architecture identifiers under ROCm 7.2.4.

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
TESSERA_TARGET_ROCM: str = "7.2.4"
TESSERA_TARGET_HIP: str = "7.2.4"
TESSERA_TARGET_RCCL_MIN: str = "2.22"     # RCCL bundled with ROCm 7.2.4
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


# Per-arch dtype matrix accepted by the ROCm 7.2.4 backend.  Canonical
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
#   "ready"          — fully supported under ROCm 7.2.4
#   "tba"            — present in the architecture, not yet exposed by HIP 7.2.4
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


# ── Per-arch chiplet (XCD) topology ─────────────────────────────────────────
# CDNA 3/4 are multi-die: the compute is split across N Accelerator Complex Dies
# (XCDs), each with its own slice of the shared L2.  Data resident in one XCD's
# L2 slice is cheap to that XCD and a cross-die hop for the others — so a grid
# that scatters one attention head's Q-blocks across XCDs forces its K/V to be
# re-fetched into multiple L2 slices.  The moonmath CDNA3 writeup's "head-first
# swizzle" pins all of a head's blocks to a single XCD so its K/V stays resident
# in that XCD's slice (see :func:`head_first_xcd`).
#
# Counts (CDNA): MI300X/gfx942 = 8 XCDs (304 CUs), MI300A/gfx940 = 6 XCDs,
# CDNA 4/gfx950 = 8 XCDs.  MI250/gfx90a is 2 GCDs (modeled as 2).  RDNA parts
# are monolithic (1).  These are die-count topology facts, not an execution
# claim; ``gfx950`` is labeled provisional pending an MI350-class spec check.
_XCD_COUNT: dict[AMDArch, int] = {
    AMDArch.GFX_90A:  2,    # MI250 — 2 GCDs
    AMDArch.GFX_940:  6,    # MI300A
    AMDArch.GFX_942:  8,    # MI300X
    AMDArch.GFX_950:  8,    # CDNA 4 (MI350-class) — PROVISIONAL count
    AMDArch.GFX_1100: 1,    # RDNA — monolithic
    AMDArch.GFX_1151: 1,
    AMDArch.GFX_1200: 1,
    AMDArch.GFX_1250: 1,
    AMDArch.GFX_1251: 1,
}


# Per-arch HIP/HIPCC compile-target strings under ROCm 7.2.4.
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
    def num_xcds(self) -> int:
        """Number of XCDs / GCDs on this arch (1 on monolithic RDNA parts).

        Drives chiplet-aware grid mapping — see :func:`head_first_xcd`."""
        return _XCD_COUNT[self.arch]

    @property
    def is_multi_die(self) -> bool:
        """True when the arch has more than one XCD/GCD (chiplet topology)."""
        return _XCD_COUNT[self.arch] > 1

    @property
    def waves_per_simd(self) -> int:
        return self.waves_per_cu

    @property
    def threads_per_wave(self) -> int:
        # AMD wavefronts are 64 lanes on CDNA, 32 on RDNA.
        return 32 if self.arch in _RDNA_ARCHES else 64

    @property
    def rocm_features(self) -> frozenset[str]:
        """Set of all ROCm 7.2.4 features marked ``ready`` for this arch."""
        return rocm_feature_set(self.arch)

    @property
    def hipcc_arch(self) -> str:
        """``hipcc --offload-arch=...`` string under ROCm 7.2.4."""
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

    def mfma_shapes_by_footprint(
        self, *, k: Optional[int] = None
    ) -> list[tuple[tuple[int, int, int, int], int]]:
        """Legal MFMA shapes ranked cheapest-accumulator-first for this arch.

        See :func:`rank_mfma_shapes_by_footprint`.  The matrix-core selection
        lever from the moonmath CDNA3 attention writeup: prefer the shape with
        the smallest M×N accumulator so registers are free for prefetch + Q."""
        return rank_mfma_shapes_by_footprint(self.arch, k=k)

    def cheapest_mfma_shape(self, *, k: Optional[int] = None) -> tuple[int, int, int, int]:
        """The smallest-accumulator-footprint MFMA shape for this arch.

        See :func:`cheapest_mfma_shape`."""
        return cheapest_mfma_shape(self.arch, k=k)

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
    """Return the ROCm 7.2.4 status for a per-arch feature. Raises ``KeyError`` with
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


def xcd_count(arch: AMDArch) -> int:
    """Number of Accelerator Complex Dies (XCDs / GCDs) on ``arch``.

    ``1`` for monolithic (RDNA) parts.  See :data:`_XCD_COUNT`."""
    return _XCD_COUNT[arch]


def head_first_xcd(batch: int, head: int, *, num_heads: int, num_xcds: int) -> int:
    """XCD a ``(batch, head)``'s attention work is pinned to (head-first swizzle).

    All Q-blocks of a given ``(batch, head)`` map to one XCD, so that head's K/V
    stays resident in that XCD's L2 slice instead of being re-fetched into
    several slices.  The assignment is the ``(batch, head)`` pair index modulo
    the XCD count — independent of the Q-block index, which is exactly the
    L2-residency property the moonmath writeup's swizzle achieves.

    Raises a stable diagnostic on non-positive ``num_heads``/``num_xcds`` or
    out-of-range / negative ``batch``/``head``.
    """
    if num_heads <= 0 or num_xcds <= 0:
        raise TesseraROCmTargetError(
            f"head_first_xcd: num_heads ({num_heads}) and num_xcds ({num_xcds}) "
            f"must be positive")
    if batch < 0 or head < 0:
        raise TesseraROCmTargetError(
            f"head_first_xcd: batch ({batch}) and head ({head}) must be >= 0")
    if head >= num_heads:
        raise TesseraROCmTargetError(
            f"head_first_xcd: head ({head}) out of range for num_heads={num_heads}")
    return (batch * num_heads + head) % num_xcds


def naive_block_xcd(
    batch: int,
    head: int,
    q_block: int,
    *,
    num_heads: int,
    q_blocks: int,
    num_xcds: int,
) -> int:
    """XCD a Q-block lands on under the *default* round-robin block scheduler.

    The hardware's default maps the linear global block id modulo the XCD count,
    so a single head's Q-blocks scatter across XCDs (the residency problem
    head-first swizzle fixes).  Provided as the baseline :func:`head_first_xcd`
    improves on.
    """
    if num_heads <= 0 or q_blocks <= 0 or num_xcds <= 0:
        raise TesseraROCmTargetError(
            "naive_block_xcd: num_heads, q_blocks, num_xcds must be positive")
    if batch < 0 or head < 0 or q_block < 0:
        raise TesseraROCmTargetError(
            "naive_block_xcd: batch, head, q_block must be >= 0")
    if head >= num_heads or q_block >= q_blocks:
        raise TesseraROCmTargetError(
            f"naive_block_xcd: head/q_block out of range "
            f"(head={head}/{num_heads}, q_block={q_block}/{q_blocks})")
    global_block = (batch * num_heads + head) * q_blocks + q_block
    return global_block % num_xcds


def wmma_variants(arch: AMDArch) -> frozenset[tuple[int, int, int]]:
    """Return WMMA instruction shapes (M, N, K) for an RDNA ``arch`` (empty on CDNA)."""
    return _WMMA_VARIANTS[arch]


# ── MFMA accumulator-footprint cost model ───────────────────────────────────
# (moonmath CDNA3 attention writeup, §"Matrix Core Selection".)
#
# A matrix-core instruction holds its M×N fp32 accumulator tile distributed
# across the wavefront's lanes.  On CDNA wave64 each lane owns ``M*N // 64``
# fp32 accumulators (AGPRs); on RDNA/wave32 each lane owns ``M*N // 32`` VGPRs.
# The footprint depends ONLY on M and N — not on K — so two shapes that saturate
# the matrix core at the same FLOPs/cycle can have very different register cost:
#
#     16×16×16  → 16*16 // 64 =  4 fp32/lane
#     32×32×8   → 32*32 // 64 = 16 fp32/lane   (4× the accumulator footprint)
#
# Both do 512 FLOPs/cycle on gfx94x, so the article picks 16×16×16 purely to
# free registers for deeper prefetch rings + persistent Q.  This is the lever
# ``mfma_table.inc`` / ``MFMAFullCoveragePass`` cannot express: they answer
# "which shapes are *legal*", not "which legal shape is *cheapest* in registers".


def mfma_accumulator_regs(shape: tuple[int, ...], *, lanes: int = 64) -> int:
    """Per-lane fp32 accumulator registers a matrix-core ``shape`` occupies.

    ``shape`` is an ``(M, N, K[, K_blocks])`` tuple (MFMA carries a 4th
    ``K_blocks`` field, WMMA a 3-tuple).  The accumulator is the M×N output
    tile, spread one fp32 per accumulated MAC across ``lanes`` lanes, so the
    per-lane cost is ``M * N // lanes`` — independent of K.  ``lanes`` is 64
    for CDNA MFMA (the default) and 32 for RDNA/wave32 WMMA.

    Raises ``TesseraROCmTargetError`` if the M×N tile does not divide evenly
    across ``lanes`` (a malformed shape) rather than silently truncating.
    """
    if len(shape) < 2:
        raise TesseraROCmTargetError(
            f"mfma_accumulator_regs: shape must be (M, N, ...); got {shape!r}")
    m, n = shape[0], shape[1]
    if m <= 0 or n <= 0:
        raise TesseraROCmTargetError(
            f"mfma_accumulator_regs: M, N must be positive; got M={m}, N={n}")
    if lanes <= 0:
        raise TesseraROCmTargetError(
            f"mfma_accumulator_regs: lanes must be positive; got {lanes}")
    if (m * n) % lanes != 0:
        raise TesseraROCmTargetError(
            f"mfma_accumulator_regs: M*N={m * n} does not divide evenly across "
            f"{lanes} lanes (shape {shape!r})")
    return (m * n) // lanes


def rank_mfma_shapes_by_footprint(
    arch: AMDArch, *, k: Optional[int] = None
) -> list[tuple[tuple[int, int, int, int], int]]:
    """Legal MFMA shapes for ``arch``, ranked by accumulator footprint (cheapest
    first).

    Returns a list of ``(shape, accumulator_regs)`` pairs sorted ascending by
    per-lane accumulator registers, tie-broken by *descending* arithmetic
    density ``M*N*K`` (prefer the larger contraction at equal register cost).
    The order is fully deterministic so it can drive selection in a pass.

    ``k`` optionally filters to shapes with that contraction width — the way a
    caller restricts to the K a given storage dtype lowers to (bf16/fp16 → 16
    on the 16×16×16 form, fp8 → 32, fp4 → 64, xf32 → 8).
    """
    lanes = 32 if arch in _RDNA_ARCHES else 64
    shapes = [s for s in _MFMA_VARIANTS[arch] if k is None or s[2] == k]
    ranked = [(s, mfma_accumulator_regs(s, lanes=lanes)) for s in shapes]
    # Cheapest accumulator first; tie → larger M*N*K (more work per issue) first.
    ranked.sort(key=lambda sr: (sr[1], -(sr[0][0] * sr[0][1] * sr[0][2])))
    return ranked


def cheapest_mfma_shape(
    arch: AMDArch, *, k: Optional[int] = None
) -> tuple[int, int, int, int]:
    """The MFMA shape on ``arch`` with the smallest accumulator footprint.

    This is the article's "16×16×16 over 32×32×8" decision as a function:
    among the legal shapes (optionally filtered to contraction width ``k``),
    return the one that frees the most registers.  Raises a stable diagnostic
    when ``arch`` has no MFMA shapes (an RDNA/WMMA arch) or none match ``k``.
    """
    ranked = rank_mfma_shapes_by_footprint(arch, k=k)
    if not ranked:
        if not _MFMA_VARIANTS[arch]:
            raise TesseraROCmTargetError(
                f"{arch.name} has no MFMA shapes (WMMA arch — see wmma_variants).")
        raise TesseraROCmTargetError(
            f"{arch.name} has no MFMA shape with contraction width k={k}.")
    return ranked[0][0]


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
    "mfma_accumulator_regs",
    "rank_mfma_shapes_by_footprint",
    "cheapest_mfma_shape",
    "xcd_count",
    "head_first_xcd",
    "naive_block_xcd",
    "fp8_semantics",
    "fp8_dtype_flavor",
]
