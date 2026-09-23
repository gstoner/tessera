"""Host contracts for the opted-in folded gfx1201 prefill route."""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import (
    GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI,
    emit_mxfp4_folded_prefill_hip,
    package_mxfp4_folded_prefill,
    prepare_folded_weights,
)
from tessera.compiler.rocm_mxfp4_folded_carrier import (
    package_folded_scaled_wmma_target_ir,
)
from tessera import runtime


def test_folded_payload_requires_policy_and_carries_loss() -> None:
    codes = np.full((48, 64), 1, dtype=np.uint8)
    scales = np.full((2, 48), 128, dtype=np.uint8)
    scales[0, :] = 117
    checkpoint = mx.pack_e2m1_codes(codes)
    with pytest.raises(ValueError, match="explicit approximate"):
        prepare_folded_weights(checkpoint, scales)
    folded = prepare_folded_weights(
        checkpoint, scales, allow_approximate=True
    )
    assert folded.weight_bytes.shape == (48, 64)
    assert folded.row_reference.shape == (48,)
    assert not folded.lossless
    assert folded.inexact_value_count > 0
    assert folded.max_normalized_abs_error > 0
    assert (
        mx.mxfp4_weight_layout(mx.MXFP4_FOLDED_ROW_LAYOUT_V1).logical_shape
        == "[N,K]"
    )
    with pytest.raises(ValueError, match="explicit approximate-policy consent"):
        mx.convert_weight_layout(
            checkpoint,
            source=mx.MXFP4_CHECKPOINT_LAYOUT_V1,
            destination=mx.MXFP4_FOLDED_ROW_LAYOUT_V1,
        )
    with pytest.raises(ValueError, match="explicit approximate"):
        package_mxfp4_folded_prefill(256, 48, 64, folded)
    assert (
        "tessera.rocm.mxfp4_w4a8.a_bfold_sa_rowref_o_m_n_k."
        "e4m3_e4m3_e8m0_bf16.approx_bm256_tm4.v1"
        in runtime._gfx1201_proved_scheduled_abis()
    )


def test_folded_kernel_has_tall_tile_and_uniform_stage_barriers() -> None:
    source = emit_mxfp4_folded_prefill_hip()
    assert "wm * 64 + i * 16" in source
    assert "sA[256 * 80]" in source
    assert "sB[64 * 80]" in source
    assert source.count("__syncthreads()") == 6
    assert "__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12" in source
    assert "const float combined_scale = row_scale * activation_scale;" in source
    assert "partial * combined_scale" in source
    assert "(double)partial * (double)row_scale" in source
    assert source.index("copy_start = (unsigned long long)wall_clock64();") < source.index(
        "__syncthreads();  // no wave begins this copy phase"
    )
    assert source.index("compute_start = (unsigned long long)wall_clock64();") < source.index(
        "__syncthreads();  // no wave begins WMMA"
    )
    assert source.index("__syncthreads();  // every wave finishes WMMA") < source.index(
        "last_tick = (unsigned long long)wall_clock64();"
    )
    assert source.index("last_tick = (unsigned long long)wall_clock64();") < source.index(
        "__syncthreads();  // no wave starts the next K step"
    )
    assert "no wave begins WMMA before the copy-end stamp" in source
    assert "no wave starts the next K step before its end stamp" in source


def test_folded_k64_staging_specialization_preserves_k32_tail() -> None:
    full = emit_mxfp4_folded_prefill_hip(full_k64=True)
    tail = emit_mxfp4_folded_prefill_hip(full_k64=False)
    assert full.count("if constexpr (true)") == 2
    assert tail.count("if constexpr (false)") == 2
    assert full.count("else if (kb + off < K)") == 2
    assert tail.count("else if (kb + off < K)") == 2
    assert "__FULL_K64__" not in full + tail
    # Only the two operand copies specialize. K16 issue scheduling remains
    # guarded because fully unrolling it raised VGPR pressure on gfx1201.
    assert "step < 4 && kb + step * 16 < K" in full


def test_folded_materializer_refuses_exact_or_mismatched_carrier() -> None:
    codes = np.ones((48, 64), dtype=np.uint8)
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), np.full((2, 48), 127, dtype=np.uint8),
        allow_approximate=True,
    )
    tile = '''tile.scaled_matmul_kernel {physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1", partial_accumulator = {combine = "row_reference_after_full_k", cross_step_motion = "forbid", init = "zero", instruction_steps = 4 : i64, schedule_scope = "k_stage", scope = "full_k"}, tessera.macro_tile_m = 256 : i64, tessera.macro_tile_n = 64 : i64, tessera.problem_m = 65 : i64, tessera.problem_n = 48 : i64, tessera.problem_k = 64 : i64, tessera.schedule_hash = "schedule-a", warps = 8 : i64}'''
    target = f'''tessera_rocm.scaled_wmma_gemm {{abi = "a_bfold_sa_rowref_d_m_n_k", block_m = 256 : i64, block_n = 64 : i64, instruction_k = 16 : i64, k = 64 : i64, k_step_schedule = "isolated_k_stage", m = 65 : i64, macro_k = 64 : i64, n = 48 : i64, numeric_policy = {{accum = "f32", execution_mode = "folded_row_reference_explicit_approximate", storage = "e4m3_raw_u8"}}, output = "bf16", package_abi = "{GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI}", partial_combine = "row_reference_after_full_k", physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1", scale_format = "e8m0_row_reference", scale_k = 64 : i64, stage_k = 64 : i64, tessera.schedule_hash = "schedule-a", tile_m_per_wave = 4 : i64, tile_n_per_wave = 2 : i64}}'''
    with pytest.raises(ValueError, match="schedule hashes disagree"):
        package_folded_scaled_wmma_target_ir(
            tile, target.replace('schedule_hash = "schedule-a"',
                                 'schedule_hash = "schedule-b"'),
            folded, allow_approximate=True,
        )
    with pytest.raises(ValueError, match="physical_contract"):
        package_folded_scaled_wmma_target_ir(
            tile, target.replace("rocm_mxfp4_w4a8_folded_prefill_v1",
                                 "rocm_mxfp4_w4a8_exact_v1"),
            folded, allow_approximate=True,
        )
    with pytest.raises(ValueError, match="stage_k"):
        package_folded_scaled_wmma_target_ir(
            tile, target.replace("stage_k = 64", "stage_k = 32"),
            folded, allow_approximate=True,
        )
    with pytest.raises(ValueError, match="numeric_policy.execution_mode"):
        package_folded_scaled_wmma_target_ir(
            tile,
            target.replace("folded_row_reference_explicit_approximate",
                           "exact_per_block"),
            folded, allow_approximate=True,
        )
    with pytest.raises(ValueError, match="warps"):
        package_folded_scaled_wmma_target_ir(
            tile.replace("warps = 8", "warps = 4"), target,
            folded, allow_approximate=True,
        )
    with pytest.raises(ValueError, match="explicit approximate"):
        package_folded_scaled_wmma_target_ir(tile, target, folded)
