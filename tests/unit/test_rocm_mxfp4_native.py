"""Host-free package/ABI gates for the exact gfx1201 MXFP4 baseline."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.native_artifact import NativeEntryPoint, NativeImageArtifact
from tessera.compiler.rocm_mxfp4_native import (
    GFX_MXFP4_W4A8_EXACT_ABI,
    GFX_MXFP4_W4A8_WMMA_ABI,
    GFX_MXFP4_W4A8_WMMA_FRAGMENT_ABI,
    MXFP4Schedule,
    _extract_gfx1201_hsaco,
    _rocm_hipcc,
    emit_mxfp4_w4a8_exact_hip,
    emit_mxfp4_w4a8_wmma_llvmir,
    mxfp4_w4a8_descriptor,
    package_mxfp4_w4a8,
    package_scaled_wmma_target_ir,
    select_mxfp4_route,
    select_mxfp4_schedule,
)


def _packed_target_ir(*, execution_mode: str = "exact_per_block") -> str:
    return f'''module {{
  tessera_rocm.scaled_wmma_gemm {{abi = "a_b_lhs_scale_rhs_scale_d_m_n_k", instruction_k = 16 : i64, k = 64 : i64, k_step_schedule = "isolated_scale_group", m = 17 : i64, macro_k = 32 : i64, n = 19 : i64, name = "packed_w4a8", numeric_policy = {{accum = "f32", execution_mode = "{execution_mode}", storage = "e4m3_raw_u8"}}, output = "bf16", package_abi = "{GFX_MXFP4_W4A8_WMMA_ABI}", partial_combine = "scale_outer_product_then_add", physical_contract = "rocm_mxfp4_w4a8_exact_v1", scale_format = "e8m0", scale_k = 32 : i64, tessera.schedule_hash = "schedule-proof"}}
}}'''


def _packed_tile_ir(*, schedule_hash: str = "schedule-proof") -> str:
    return f'''module {{
  tile.scaled_matmul_kernel {{physical_contract = "rocm_mxfp4_w4a8_exact_v1", tessera.schedule_hash = "{schedule_hash}"}}
}}'''


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
    assert "call void @llvm.amdgcn.sched.barrier(i32 0)" in source
    assert "%a_s0_0_k = add i64 %a_s0_0_k0, %half8" in source
    assert source.count("fragment_word = load i32") == 2
    assert "%b_s1_fragment_slot = add i64" in source
    assert source.index("%b_s1_fragment_word = load i32") < source.index("%a_s0_0_byte = select")
    assert "@tessera_e2m1_to_e4m3" in source
    assert "@llvm.amdgcn.workgroup.id.x" in source


def test_prefill_source_stages_one_b_fragment_for_grouped_row_waves() -> None:
    source = emit_mxfp4_w4a8_wmma_llvmir(
        schedule=MXFP4Schedule("prefill", group_m=8),
        weight_layout=mx.MXFP4_TRANSPOSED_LAYOUT_V1,
    )
    assert "@tessera_mxfp4_b_lds" in source
    assert "%is_loader_wave = icmp eq i32 %wave32, 0" in source
    assert source.count("call void @llvm.amdgcn.s.barrier()") == 2
    assert "load <2 x i32>, ptr addrspace(3)" in source
    assert '"amdgpu-flat-work-group-size"="256,256"' in source


def test_fragment_prefill_stages_packed_lane_words_in_padded_double_buffer() -> None:
    source = emit_mxfp4_w4a8_wmma_llvmir(
        schedule=MXFP4Schedule("prefill", group_m=8, stages=2)
    )
    assert "@tessera_mxfp4_b_lds" in source
    assert "[528 x i8]" in source
    assert "%b_stage = and i32 %b_stage_group32, 1" in source
    assert "%b_stage_s1_slab = add i32 %b_stage_base, 132" in source
    assert "store i32 %b_stage_s0_word, ptr addrspace(3)" in source
    assert source.index("call void @llvm.amdgcn.s.waitcnt(i32 0)") < source.index(
        "b.wait:"
    )
    assert "%b_lds_word_0 = load i32, ptr addrspace(3)" in source
    assert "b.prefetch:" in source
    assert "%b_do_prefetch = and i1 %is_loader_wave, %b_has_next" in source
    assert "b.store.next:" in source
    assert "store i32 %b_prefetch_word_0, ptr addrspace(3)" in source
    assert "%wave_m = mul i64 %wave64, 16" in source
    assert source.count("call void @llvm.amdgcn.s.barrier()") == 2


def test_schedule_selector_splits_decode_and_prefill_and_fails_closed() -> None:
    assert select_mxfp4_schedule(8, 5120, 8704) == MXFP4Schedule(
        "decode", split_k=8
    )
    assert select_mxfp4_schedule(256, 5120, 8704) == MXFP4Schedule(
        "prefill", group_m=8, stages=2
    )
    with pytest.raises(ValueError, match="decode does not admit prefill LDS stages"):
        MXFP4Schedule("decode", stages=2)
    with pytest.raises(ValueError, match="prefill does not admit decode split-K"):
        MXFP4Schedule("prefill", split_k=2)
    with pytest.raises(ValueError, match="cache_modifier is not implemented"):
        MXFP4Schedule("prefill", cache_modifier="streaming")
    with pytest.raises(ValueError, match="k_step_schedule"):
        MXFP4Schedule("decode", k_step_schedule="amd_intrinsic")


def test_relaxed_k_step_control_omits_backend_schedule_fence() -> None:
    source = emit_mxfp4_w4a8_wmma_llvmir(
        schedule=MXFP4Schedule("decode", k_step_schedule="relaxed")
    )
    assert "llvm.amdgcn.sched.barrier" not in source


def test_route_receipts_explain_production_selection_and_refusal() -> None:
    decode = select_mxfp4_route(8, 5120, 8704)
    assert decode.accepted
    assert decode.selected_layout == mx.MXFP4_GFX12_FRAGMENT_LAYOUT_V1
    assert decode.abi_id == GFX_MXFP4_W4A8_WMMA_FRAGMENT_ABI
    assert "contiguous lane words" in decode.reason

    prefill = select_mxfp4_route(256, 5120, 8704)
    assert prefill.accepted
    assert prefill.selected_layout == mx.MXFP4_GFX12_FRAGMENT_LAYOUT_V1
    assert prefill.abi_id == GFX_MXFP4_W4A8_WMMA_FRAGMENT_ABI
    assert "multistage" in prefill.reason

    shuffled = select_mxfp4_route(
        8, 5120, 8704, requested_layout=mx.MXFP4_AITER_SHUFFLED_LAYOUT_V1
    )
    assert not shuffled.accepted
    assert shuffled.selected_layout is None
    assert "incompatible" in shuffled.reason

    ragged = select_mxfp4_route(
        5, 47, 64, requested_layout=mx.MXFP4_GFX12_FRAGMENT_LAYOUT_V1
    )
    assert not ragged.accepted
    assert "N divisible by 16" in ragged.reason


def test_decode_split_k_uses_lds_partial_reduction() -> None:
    source = emit_mxfp4_w4a8_wmma_llvmir(
        schedule=MXFP4Schedule("decode", split_k=8)
    )
    assert "@tessera_mxfp4_partial_lds" in source
    assert "%group = phi i64 [ %wave64, %entry ]" in source
    assert "%group_next = add i64 %group, 8" in source
    assert "%is_reduction_wave = icmp eq i32 %wave32, 0" in source
    assert source.count("%split_v_") == 16
    assert '"amdgpu-flat-work-group-size"="256,256"' in source


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


def test_fragment_descriptor_exposes_layout_in_distinct_launch_abi() -> None:
    descriptor = mxfp4_w4a8_descriptor(
        _image(), m=16, n=48, k=64,
        entry="tessera_mxfp4_w4a8_wmma",
        abi_id=GFX_MXFP4_W4A8_WMMA_FRAGMENT_ABI,
        route="exact_per_block_fp8_wmma",
        workgroup=(32, 1, 1),
        weight_layout="mxfp4.gfx12.n16_k16_lane_u32.v1",
    )
    packed = descriptor.buffers[1]
    assert packed.layout == "row_major"
    guards = {
        (guard.binding, guard.dimension): guard.value
        for guard in descriptor.shape_guards
    }
    assert guards[("b_packed", 0)] == 48
    assert guards[("b_packed", 1)] == 32
    assert descriptor.provenance["weight_layout"] == (
        "mxfp4.gfx12.n16_k16_lane_u32.v1"
    )


def test_fragment_descriptor_refuses_unpadded_n_boundary() -> None:
    with pytest.raises(ValueError, match="N divisible by 16"):
        mxfp4_w4a8_descriptor(
            _image(), m=5, n=47, k=64,
            weight_layout="mxfp4.gfx12.n16_k16_lane_u32.v1",
        )


def test_exact_device_proof_registry_admits_versioned_mxfp4_routes() -> None:
    proved = runtime._gfx1201_proved_scheduled_abis()
    assert GFX_MXFP4_W4A8_EXACT_ABI in proved
    assert GFX_MXFP4_W4A8_WMMA_ABI in proved
    assert GFX_MXFP4_W4A8_WMMA_FRAGMENT_ABI in proved


def test_production_selector_defaults_to_wmma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int, int, str]] = []

    def record_wmma(m: int, n: int, k: int, *, pipeline_name: str):
        calls.append(("wmma", m, n, k, pipeline_name))
        return object()

    monkeypatch.setattr(
        "tessera.compiler.rocm_mxfp4_native.package_mxfp4_w4a8_wmma",
        record_wmma,
    )
    package_mxfp4_w4a8(17, 19, 64, pipeline_name="proof-pipeline")
    assert calls == [("wmma", 17, 19, 64, "proof-pipeline")]


def test_production_selector_refuses_unknown_route() -> None:
    with pytest.raises(ValueError, match="scalar_reference"):
        package_mxfp4_w4a8(16, 16, 32, route="folded")


def test_target_ir_materializer_binds_generic_carrier_to_proved_wmma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int, str, str]] = []
    image = _image()
    descriptor = mxfp4_w4a8_descriptor(
        image,
        m=17,
        n=19,
        k=64,
        entry="tessera_mxfp4_w4a8_wmma",
        abi_id=GFX_MXFP4_W4A8_WMMA_ABI,
        route="exact_per_block_fp8_wmma",
        workgroup=(32, 1, 1),
    )

    def record_wmma(
        m: int, n: int, k: int, *, pipeline_name: str, weight_layout: str
    ):
        calls.append((m, n, k, pipeline_name, weight_layout))
        return SimpleNamespace(
            tile_ir="semantic",
            target_ir="proved llvm backend ir",
            backend_ir="compiler command",
            image=image,
            descriptor=descriptor,
        )

    monkeypatch.setattr(
        "tessera.compiler.rocm_mxfp4_native.package_mxfp4_w4a8_wmma",
        record_wmma,
    )
    target_ir = _packed_target_ir()
    package = package_scaled_wmma_target_ir(
        _packed_tile_ir(), target_ir, pipeline_name="proof-pipeline"
    )
    assert calls == [(
        17,
        19,
        64,
        "proof-pipeline",
        mx.MXFP4_TRANSPOSED_LAYOUT_V1,
    )]
    assert package.tile_ir == _packed_tile_ir()
    assert package.target_ir == target_ir
    assert package.backend_ir == "proved llvm backend ir"
    assert package.image.target_ir_digest == hashlib.sha256(
        target_ir.encode()
    ).hexdigest()
    assert package.descriptor.image_digest == package.image.image_digest
    assert package.descriptor.provenance["materializer"] == (
        "tessera_rocm.scaled_wmma_gemm"
    )
    assert package.descriptor.provenance["schedule_hash"] == "schedule-proof"
    assert package.descriptor.provenance["target_ir_sha256"] == hashlib.sha256(
        target_ir.encode()
    ).hexdigest()


def test_target_ir_materializer_keeps_logical_and_approximate_routes_closed() -> None:
    logical = _packed_target_ir().replace(GFX_MXFP4_W4A8_WMMA_ABI, "unbound")
    with pytest.raises(ValueError, match="exactly one exact packed"):
        package_scaled_wmma_target_ir("tile", logical)
    with pytest.raises(ValueError, match="numeric_policy.execution_mode"):
        package_scaled_wmma_target_ir(
            _packed_tile_ir(),
            _packed_target_ir(execution_mode="approximate_row_reference"),
        )


def test_target_ir_materializer_requires_matching_schedule_provenance() -> None:
    target_without_hash = _packed_target_ir().replace(
        ', tessera.schedule_hash = "schedule-proof"', ""
    )
    with pytest.raises(ValueError, match="Target IR directive is missing"):
        package_scaled_wmma_target_ir(_packed_tile_ir(), target_without_hash)

    with pytest.raises(ValueError, match="Tile IR carrier is missing"):
        package_scaled_wmma_target_ir(
            _packed_tile_ir().replace(
                ', tessera.schedule_hash = "schedule-proof"', ""
            ),
            _packed_target_ir(),
        )

    with pytest.raises(ValueError, match="schedule_hash mismatch"):
        package_scaled_wmma_target_ir(
            _packed_tile_ir(schedule_hash="stale-schedule"), _packed_target_ir()
        )


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
