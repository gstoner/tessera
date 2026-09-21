"""Host-free package/ABI gates for the exact gfx1201 MXFP4 baseline."""
from __future__ import annotations

import hashlib
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime
from tessera.compiler.native_artifact import NativeEntryPoint, NativeImageArtifact
from tessera.compiler.rocm_mxfp4_native import (
    GFX_MXFP4_W4A8_EXACT_ABI,
    GFX_MXFP4_W4A8_WMMA_ABI,
    _extract_gfx1201_hsaco,
    _rocm_hipcc,
    emit_mxfp4_w4a8_exact_hip,
    emit_mxfp4_w4a8_wmma_llvmir,
    mxfp4_w4a8_descriptor,
)


def test_hipcc_selection_handles_split_rocm_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = tmp_path / "rocm" / "core"
    hipcc = tmp_path / "rocm" / "bin" / "hipcc"
    core.mkdir(parents=True)
    hipcc.parent.mkdir(parents=True, exist_ok=True)
    hipcc.write_text("#!/bin/sh\n")
    monkeypatch.delenv("TESSERA_ROCM_HIPCC", raising=False)
    monkeypatch.setenv("PATH", "")
    assert _rocm_hipcc(core) == hipcc


def test_hsaco_extractor_accepts_raw_elf_and_rejects_unknown_container(
    tmp_path: Path,
) -> None:
    compiled = tmp_path / "compiled"
    output = tmp_path / "raw.hsaco"
    compiled.write_bytes(b"\x7fELFgfx1201")
    assert _extract_gfx1201_hsaco(compiled, output, tmp_path) == b"\x7fELFgfx1201"
    compiled.write_bytes(b"not-a-device-image")
    with pytest.raises(RuntimeError, match="neither ELF nor a HIP bundle"):
        _extract_gfx1201_hsaco(compiled, output, tmp_path)


def _image() -> NativeImageArtifact:
    source = emit_mxfp4_w4a8_exact_hip()
    return NativeImageArtifact(
        target="rocm_gfx1201",
        architecture="gfx1201",
        pipeline_name="tessera-lower-to-rocm",
        compiler_fingerprint="test-compiler",
        toolchain_fingerprint="test-toolchain",
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        binary_format="hsaco",
        payload=b"\x7fELFtest",
        entry_points=(
            NativeEntryPoint("tessera_mxfp4_w4a8_exact", GFX_MXFP4_W4A8_EXACT_ABI),
        ),
        compile_state="cold",
    )


def test_source_keeps_exact_k32_group_and_physical_decode() -> None:
    source = emit_mxfp4_w4a8_exact_hip()
    assert "group < K / 32" in source
    assert "offset < 32" in source
    assert "(k & 1) ? (packed >> 4) : (packed & 15u)" in source
    assert "partial = fmaf(tessera_e4m3fn" in source
    assert "accum = fmaf(partial, scale, accum)" in source
    assert "exponent == 0u" in source
    assert "tessera_bf16_rne" in source
    assert "wmma" not in source.lower()  # this is the correctness baseline


def test_wmma_source_isolates_each_k32_partial_before_scaling() -> None:
    source = emit_mxfp4_w4a8_wmma_llvmir()
    assert source.count("call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8") == 2
    assert "<8 x float> zeroinitializer" in source
    assert "%scaled_partial = fmul <8 x float> %partial, %scale_vec" in source
    assert "%running_next = fadd <8 x float> %running, %scaled_partial" in source
    assert "%a_s0_0_k = add i64 %a_s0_0_k0, %half8" in source
    assert "%b_s1_7_k = add i64 %b_s1_7_k0, %half8" in source
    assert "@tessera_e2m1_to_e4m3" in source
    assert "@llvm.amdgcn.workgroup.id.x" in source


def test_wmma_descriptor_uses_one_wave_and_exact_policy() -> None:
    descriptor = mxfp4_w4a8_descriptor(
        _image(), m=17, n=33, k=64,
        entry="tessera_mxfp4_w4a8_wmma",
        abi_id=GFX_MXFP4_W4A8_WMMA_ABI,
        route="exact_per_block_fp8_wmma",
        workgroup=(32, 1, 1),
    )
    assert descriptor.abi_id == GFX_MXFP4_W4A8_WMMA_ABI
    assert descriptor.geometry.workgroup == (32, 1, 1)
    assert descriptor.provenance["route"] == "exact_per_block_fp8_wmma"
    assert descriptor.provenance["numeric_policy"] == "exact_per_block"


def test_descriptor_names_every_physical_plane_and_shape() -> None:
    descriptor = mxfp4_w4a8_descriptor(_image(), m=17, n=33, k=64)
    assert descriptor.abi_id == GFX_MXFP4_W4A8_EXACT_ABI
    assert [binding.name for binding in descriptor.buffers] == [
        "a", "b_packed", "a_scale", "b_scale", "output"
    ]
    assert [binding.dtype for binding in descriptor.buffers] == [
        "uint8", "uint8", "fp32", "uint8", "bf16"
    ]
    assert descriptor.geometry.grid == (3, 2, 1)
    assert descriptor.geometry.workgroup == (16, 16, 1)
    assert descriptor.provenance["scale_group_k"] == 32
    assert descriptor.provenance["numeric_policy"] == "exact_per_block"
    guards = {(guard.binding, guard.dimension): guard.value for guard in descriptor.shape_guards}
    assert guards[("b_packed", 0)] == 32
    assert guards[("b_scale", 0)] == 2


def test_descriptor_rejects_non_grouped_k_and_wrong_target() -> None:
    with pytest.raises(ValueError, match="divisible by 32"):
        mxfp4_w4a8_descriptor(_image(), m=16, n=16, k=48)
    image = _image()
    wrong = NativeImageArtifact(
        target="rocm_gfx1151", architecture="gfx1151",
        pipeline_name=image.pipeline_name,
        compiler_fingerprint=image.compiler_fingerprint,
        toolchain_fingerprint=image.toolchain_fingerprint,
        target_ir_digest=image.target_ir_digest,
        binary_format=image.binary_format, payload=image.payload,
        entry_points=image.entry_points, compile_state=image.compile_state,
    )
    with pytest.raises(ValueError, match="exact gfx1201"):
        mxfp4_w4a8_descriptor(wrong, m=16, n=16, k=32)


def test_launcher_refuses_proof_transfer_before_loading_hip(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _image()
    descriptor = mxfp4_w4a8_descriptor(image, m=1, n=1, k=32)
    buffers = {
        "a": np.zeros((1, 32), np.uint8),
        "b_packed": np.zeros((16, 1), np.uint8),
        "a_scale": np.ones((1,), np.float32),
        "b_scale": np.zeros((1, 1), np.uint8),
        "output": np.zeros((1, 1), dtype=ml_dtypes.bfloat16),
    }
    monkeypatch.setattr(runtime, "_rocm_live_arch", lambda: "gfx1151")
    monkeypatch.setattr(
        runtime, "_load_hip_for_launch",
        lambda: pytest.fail("HIP must not load after exact-target refusal"),
    )
    with pytest.raises(RuntimeError, match="selected gfx1201 device"):
        runtime._submit_rocm_mxfp4_w4a8(
            image, descriptor, buffers, {"M": 1, "N": 1, "K": 32}
        )


def test_launcher_rejects_reserved_e8m0_before_loading_hip(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _image()
    descriptor = mxfp4_w4a8_descriptor(image, m=1, n=1, k=32)
    buffers = {
        "a": np.zeros((1, 32), np.uint8),
        "b_packed": np.zeros((16, 1), np.uint8),
        "a_scale": np.ones((1,), np.float32),
        "b_scale": np.full((1, 1), 255, np.uint8),
        "output": np.zeros((1, 1), dtype=ml_dtypes.bfloat16),
    }
    monkeypatch.setattr(runtime, "_rocm_live_arch", lambda: "gfx1201")
    monkeypatch.setattr(
        runtime, "_load_hip_for_launch",
        lambda: pytest.fail("HIP must not load for a reserved E8M0 code"),
    )
    with pytest.raises(RuntimeError, match="code 255 is reserved"):
        runtime._submit_rocm_mxfp4_w4a8(
            image, descriptor, buffers, {"M": 1, "N": 1, "K": 32}
        )
