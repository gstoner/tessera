"""Host contracts for the opted-in folded gfx1201 prefill route."""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import (
    emit_mxfp4_folded_prefill_hip,
    package_mxfp4_folded_prefill,
    prepare_folded_weights,
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
