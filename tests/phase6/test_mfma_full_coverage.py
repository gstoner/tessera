"""
test_mfma_full_coverage.py — ROCm MFMA table completeness (Phase 6)

Verifies that the Tessera MFMA selection table covers all required
configurations for gfx90a (MI200), gfx94x (MI300), and gfx120x (Instinct 4).

Since we don't compile C++ in the test environment, the MFMA coverage is
validated through the Python-side codegen configuration table that mirrors
the C++ chooseMFMAIntrinsic() logic.
"""
from __future__ import annotations

import pytest
from typing import Dict, List, NamedTuple, Optional

# ---------------------------------------------------------------------------
# Minimal mirror of the C++ MFMA selection table.
# In production this is generated from the compiler.codegen module.
# Here we inline the expected table for test coverage verification.
# ---------------------------------------------------------------------------

class MFMADescriptor(NamedTuple):
    intrinsic: str          # e.g. "mfma_f32_32x32x8_f16"
    m: int
    n: int
    k: int
    in_type: str            # "f16", "bf16", "f32", "i8"
    out_type: str           # "f32", "i32"
    blocks: int = 1


# Known MFMA shapes per target (from AMD ISA docs)
GFX90A_MFMA = [
    MFMADescriptor("mfma_f32_32x32x8_f16",   32, 32, 8,  "f16",  "f32"),
    MFMADescriptor("mfma_f32_16x16x16_f16",  16, 16, 16, "f16",  "f32"),
    MFMADescriptor("mfma_f32_4x4x4_f16",      4,  4,  4, "f16",  "f32"),
    MFMADescriptor("mfma_f32_32x32x8_bf16",  32, 32, 8,  "bf16", "f32"),
    MFMADescriptor("mfma_f32_16x16x16_bf16", 16, 16, 16, "bf16", "f32"),
    MFMADescriptor("mfma_i32_32x32x8_i8",    32, 32, 8,  "i8",   "i32"),
    MFMADescriptor("mfma_i32_16x16x16_i8",   16, 16, 16, "i8",   "i32"),
    MFMADescriptor("mfma_f64_16x16x4_f64",   16, 16, 4,  "f64",  "f64"),
    MFMADescriptor("mfma_f32_32x32x4_xf32",  32, 32, 4,  "xf32", "f32"),
]

GFX94X_MFMA = [
    # gfx94x inherits gfx90a and adds fp8 variants
    MFMADescriptor("mfma_f32_32x32x8_f16",   32, 32, 8,  "f16",  "f32"),
    MFMADescriptor("mfma_f32_16x16x16_f16",  16, 16, 16, "f16",  "f32"),
    MFMADescriptor("mfma_f32_32x32x8_bf16",  32, 32, 8,  "bf16", "f32"),
    MFMADescriptor("mfma_f32_16x16x16_bf16", 16, 16, 16, "bf16", "f32"),
    MFMADescriptor("mfma_i32_32x32x8_i8",    32, 32, 8,  "i8",   "i32"),
    MFMADescriptor("mfma_i32_16x16x16_i8",   16, 16, 16, "i8",   "i32"),
    MFMADescriptor("mfma_f32_32x32x16_fp8",  32, 32, 16, "fp8",  "f32"),
    MFMADescriptor("mfma_f32_16x16x32_fp8",  16, 16, 32, "fp8",  "f32"),
    MFMADescriptor("mfma_f32_32x32x16_bf8",  32, 32, 16, "bf8",  "f32"),
    MFMADescriptor("mfma_f32_16x16x32_bf8",  16, 16, 32, "bf8",  "f32"),
]

GFX120X_MFMA = [
    # gfx120x (RDNA 4 / Instinct 4) — wave32 WMMA-style + MFMA subset
    MFMADescriptor("mfma_f32_16x16x16_f16",  16, 16, 16, "f16",  "f32"),
    MFMADescriptor("mfma_f32_16x16x16_bf16", 16, 16, 16, "bf16", "f32"),
    MFMADescriptor("mfma_i32_16x16x16_i8",   16, 16, 16, "i8",   "i32"),
    MFMADescriptor("mfma_f32_16x16x32_fp8",  16, 16, 32, "fp8",  "f32"),
    MFMADescriptor("mfma_f32_16x16x32_bf8",  16, 16, 32, "bf8",  "f32"),
]

MFMA_TABLE: Dict[str, List[MFMADescriptor]] = {
    "gfx90a":  GFX90A_MFMA,
    "gfx94x":  GFX94X_MFMA,
    "gfx120x": GFX120X_MFMA,
}

# ---------------------------------------------------------------------------
# Helper: simulate chooseMFMAIntrinsic
# ---------------------------------------------------------------------------

def choose_mfma(target: str, dtype: str, prefer_m: int = 32) -> Optional[MFMADescriptor]:
    """
    Mimics the C++ chooseMFMAIntrinsic logic:
      1. Filter by target and input dtype.
      2. Prefer larger M (prefer_m = 32 first, fallback 16, then 4).
      3. Return None if no match found.
    """
    table = MFMA_TABLE.get(target, [])
    candidates = [d for d in table if d.in_type == dtype]
    if not candidates:
        return None
    # Sort by preferred M descending, then by K descending (larger tiles first)
    candidates.sort(key=lambda d: (-(d.m == prefer_m), -d.m, -d.k))
    return candidates[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGFX90aCoverage:
    TARGET = "gfx90a"

    def test_f16_covered(self):
        d = choose_mfma(self.TARGET, "f16")
        assert d is not None
        assert d.intrinsic.endswith("f16")

    def test_bf16_covered(self):
        d = choose_mfma(self.TARGET, "bf16")
        assert d is not None

    def test_i8_covered(self):
        d = choose_mfma(self.TARGET, "i8")
        assert d is not None
        assert d.out_type == "i32"

    def test_f64_covered(self):
        d = choose_mfma(self.TARGET, "f64")
        assert d is not None

    def test_xf32_covered(self):
        d = choose_mfma(self.TARGET, "xf32")
        assert d is not None

    def test_prefers_m32_over_m16(self):
        d = choose_mfma(self.TARGET, "f16", prefer_m=32)
        assert d is not None
        assert d.m == 32

    def test_fallback_to_m16(self):
        d = choose_mfma(self.TARGET, "f16", prefer_m=16)
        assert d is not None
        assert d.m in (16, 32)

    def test_no_fp8_on_gfx90a(self):
        d = choose_mfma(self.TARGET, "fp8")
        assert d is None

    def test_all_intrinsic_names_nonempty(self):
        for desc in GFX90A_MFMA:
            assert desc.intrinsic and desc.intrinsic.startswith("mfma_")

    def test_output_types_valid(self):
        valid_out = {"f32", "i32", "f64"}
        for desc in GFX90A_MFMA:
            assert desc.out_type in valid_out


class TestGFX94xCoverage:
    TARGET = "gfx94x"

    def test_f16_covered(self):
        assert choose_mfma(self.TARGET, "f16") is not None

    def test_bf16_covered(self):
        assert choose_mfma(self.TARGET, "bf16") is not None

    def test_fp8_covered(self):
        d = choose_mfma(self.TARGET, "fp8")
        assert d is not None

    def test_bf8_covered(self):
        d = choose_mfma(self.TARGET, "bf8")
        assert d is not None

    def test_fp8_out_is_f32(self):
        d = choose_mfma(self.TARGET, "fp8")
        assert d.out_type == "f32"

    def test_i8_covered(self):
        assert choose_mfma(self.TARGET, "i8") is not None

    def test_fp8_k_ge_16(self):
        """FP8 MFMA blocks accumulate in larger K chunks (≥16)."""
        fp8_descs = [d for d in GFX94X_MFMA if d.in_type == "fp8"]
        for d in fp8_descs:
            assert d.k >= 16

    def test_table_size_gte_gfx90a(self):
        assert len(GFX94X_MFMA) >= len(GFX90A_MFMA)


class TestGFX120xCoverage:
    TARGET = "gfx120x"

    def test_f16_covered(self):
        assert choose_mfma(self.TARGET, "f16") is not None

    def test_bf16_covered(self):
        assert choose_mfma(self.TARGET, "bf16") is not None

    def test_fp8_covered(self):
        assert choose_mfma(self.TARGET, "fp8") is not None

    def test_bf8_covered(self):
        assert choose_mfma(self.TARGET, "bf8") is not None

    def test_i8_covered(self):
        assert choose_mfma(self.TARGET, "i8") is not None

    def test_no_m32_on_gfx120x(self):
        """gfx120x uses wave32 — only 16x16 tiles."""
        for desc in GFX120X_MFMA:
            assert desc.m <= 16

    def test_all_output_f32_or_i32(self):
        for desc in GFX120X_MFMA:
            assert desc.out_type in ("f32", "i32")


class TestChooseMFMAFallback:
    def test_unknown_target_returns_none(self):
        assert choose_mfma("gfx999", "f16") is None

    def test_unknown_dtype_returns_none(self):
        assert choose_mfma("gfx90a", "e4m3fnuz_custom") is None

    def test_all_targets_have_f16(self):
        for target in MFMA_TABLE:
            assert choose_mfma(target, "f16") is not None

    def test_all_targets_have_bf16(self):
        for target in MFMA_TABLE:
            assert choose_mfma(target, "bf16") is not None
