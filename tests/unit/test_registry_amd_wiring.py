"""Locks the A1 (MMA descriptor) + A3 (epilogue) registry wiring into
primitive_coverage, and the A6 FP8-semantics dashboard surface.
"""

from __future__ import annotations

from tessera.compiler import backend_manifest as bm
from tessera.compiler import gpu_target_map
from tessera.compiler.primitive_coverage import _existing_coverage
from tessera.compiler.rocm_mma import MmaDescriptor


def _reg():
    return _existing_coverage()


# ── A1: rocm_mma metadata on GEMM-family ops ────────────────────────────────

def test_matmul_carries_rocm_mma():
    md = _reg()["matmul"].metadata
    assert "rocm_mma" in md
    mma = md["rocm_mma"]
    # CDNA3 (gfx942) bf16 → 16x16x16 mfma, fp32 accum, k_width 2.
    bf16 = mma["gfx942"]["bf16"]
    assert bf16["kind"] == "mfma"
    assert bf16["shape"] == [16, 16, 16]
    assert bf16["acc"] == "fp32"
    assert bf16["k_width"] == 2


def test_matmul_rocm_mma_fp8_only_where_supported():
    mma = _reg()["matmul"].metadata["rocm_mma"]
    # gfx942 has FP8 MFMA; gfx1151 (RDNA 3.5) has NO FP8 path.
    assert "fp8_e4m3" in mma["gfx942"]
    assert "fp8_e4m3" not in mma["gfx1151"]
    # gfx950 (CDNA 4) adds FP4.
    assert "fp4_e2m1" in mma["gfx950"]


def test_rdna_mma_is_wmma():
    mma = _reg()["matmul"].metadata["rocm_mma"]
    assert mma["gfx1100"]["bf16"]["kind"] == "wmma"
    assert mma["gfx1151"]["fp16"]["kind"] == "wmma"


def test_batched_and_grouped_gemm_carry_rocm_mma():
    reg = _reg()
    assert "rocm_mma" in reg["batched_gemm"].metadata
    assert "rocm_mma" in reg["grouped_gemm"].metadata


def test_non_gemm_op_has_no_rocm_mma():
    # softmax is not a GEMM-family op.
    assert "rocm_mma" not in _reg()["softmax"].metadata


# ── A3: epilogue metadata on fused_epilogue ─────────────────────────────────

def test_fused_epilogue_carries_epilogue_catalog():
    md = _reg()["fused_epilogue"].metadata
    assert "epilogue" in md
    cat = md["epilogue"]["catalog"]
    assert "matmul_gelu" in cat
    assert cat["matmul_gelu"]["activation"] == "gelu"
    # gelu has a registered backward epilogue (DGELU = 192).
    assert cat["matmul_gelu"]["backward_flags"] == 192
    assert cat["matmul_gelu"]["requires_aux"] is True


def test_epilogue_silu_has_no_backward():
    cat = _reg()["fused_epilogue"].metadata["epilogue"]["catalog"]
    # SiLU has no backward epilogue in the hipBLASLt vocabulary.
    assert cat["matmul_silu"]["backward_flags"] is None


def test_epilogue_autodiff_note_present():
    note = _reg()["fused_epilogue"].metadata["epilogue"]["autodiff_note"]
    assert "_AUX" in note


# ── A6: FP8 semantics dashboard surface ─────────────────────────────────────

def test_rocm_target_map_has_fp8_section():
    md = gpu_target_map.render_markdown("rocm")
    assert "FP8 numeric semantics" in md
    assert "e4m3fnuz" in md  # gfx942 FNUZ
    assert "| `gfx950` | ocp |" in md


def test_nvidia_map_has_no_fp8_section():
    # The FP8-semantics table is ROCm-specific.
    assert "FP8 numeric semantics" not in gpu_target_map.render_markdown("nvidia_sm90")


# ── A1: backend_manifest typed mma_descriptor field ─────────────────────────

def test_backend_manifest_rocm_gemm_carries_mma_descriptor():
    rocm = [e for e in bm.manifest_for("matmul") if e.target == "rocm"]
    assert rocm and rocm[0].mma_descriptor is not None
    assert isinstance(rocm[0].mma_descriptor, MmaDescriptor)
    # and it surfaces in the serialized manifest.
    assert "mma_descriptor" in rocm[0].as_dict()


def test_backend_manifest_non_gemm_has_no_mma_descriptor():
    rocm = [e for e in bm.manifest_for("softmax") if e.target == "rocm"]
    assert all(e.mma_descriptor is None for e in rocm)


def test_backend_manifest_mma_descriptor_rejects_non_rocm_target():
    import pytest
    from tessera.compiler.rocm_mma import select_mma
    from tessera.compiler.rocm_target import AMDArch
    desc = select_mma(AMDArch.GFX_942, "bf16")
    with pytest.raises(ValueError, match="only applies to ROCm"):
        bm.BackendKernelEntry(target="nvidia_sm90", status="planned",
                              mma_descriptor=desc)
