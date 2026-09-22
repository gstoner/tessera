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
    with pytest.raises(ValueError, match="explicit approximate"):
        package_mxfp4_folded_prefill(256, 48, 64, folded)


def test_folded_kernel_has_tall_tile_and_uniform_stage_barriers() -> None:
    source = emit_mxfp4_folded_prefill_hip()
    assert "wm * 64 + i * 16" in source
    assert "sA[256 * 80]" in source
    assert "sB[64 * 80]" in source
    assert source.count("__syncthreads()") == 2
    assert "__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12" in source
