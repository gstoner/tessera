"""Arch coverage for the compiled ROCm WMMA GEMM lane.

The compiler-generated GEMM emits the RDNA WMMA **16x16x16** f16/bf16 fragment
layout, which the ROCDL backend selects only on the **gfx11 family** (RDNA3 /
RDNA3.5 — hardware-verified on gfx1151). RDNA4 gfx12xx uses a 16x16x32 WMMA
layout and CDNA uses MFMA (32x32x8); neither selects this intrinsic. Per
Decision #21 the build must raise a *stable, arch-naming* diagnostic for those
targets rather than letting the backend emit a raw "Cannot select" crash.

This is the "others compile-verify" leg of the multi-arch WMMA contract: gfx11
executes (covered by test_rocm_ebm_partition_compiled / the wmma runtime-symbol
test on live silicon); gfx12xx / CDNA are honestly gated here.
"""
from __future__ import annotations

import pytest

from tessera import runtime as rt


@pytest.mark.parametrize("chip", ["gfx1200", "gfx1201", "gfx1250", "gfx942", "gfx90a"])
def test_non_gfx11_arch_raises_stable_diagnostic(monkeypatch, chip):
    monkeypatch.setattr(rt, "_rocm_chip", lambda: chip)
    with pytest.raises(rt._RocmCompiledUnavailable) as ei:
        rt._build_compiled_gemm_hsaco(64, 64, dtype="f16")
    msg = str(ei.value)
    # The diagnostic must name the target and the reason (not a generic failure).
    assert chip in msg
    assert "16x16x16" in msg
    assert "gfx11" in msg


@pytest.mark.parametrize("chip", ["gfx1100", "gfx1101", "gfx1150", "gfx1151"])
def test_gfx11_family_passes_the_arch_guard(monkeypatch, chip):
    """gfx11 targets must clear the guard (they reach the real build path).

    We force the tessera-opt lookup to miss so the build stops right *after* the
    guard with the *tessera-opt* diagnostic — never the arch-gate message. That
    proves the guard does not spuriously reject a supported RDNA3/3.5 arch,
    without paying for a full in-process kernel compile.
    """
    monkeypatch.setattr(rt, "_rocm_chip", lambda: chip)
    monkeypatch.setattr(rt, "_tessera_opt_path", lambda: None)
    try:
        rt._build_compiled_gemm_hsaco(64, 64, dtype="f16")
    except rt._RocmCompiledUnavailable as e:
        assert "16x16x16 f16/bf16 WMMA fragment layout is a gfx11" not in str(e), (
            f"guard wrongly rejected supported arch {chip}: {e}")
