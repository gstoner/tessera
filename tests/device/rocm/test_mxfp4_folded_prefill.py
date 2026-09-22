"""Exact-device proof for the opted-in gfx1201 folded MXFP4 ABI."""
from __future__ import annotations

import json
import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import (
    package_mxfp4_folded_prefill, prepare_folded_weights,
)
from tests._support import rocm_isa


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("shape", [(65, 48, 64), (256, 80, 128)])
def test_folded_prefill_matches_its_declared_approximate_oracle(
    shape: tuple[int, int, int],
) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = shape
    a = np.full((m, k), 0x38, dtype=np.uint8)  # E4M3 +1
    a_scale = np.ones(m, dtype=np.float32)
    codes = np.zeros((n, k), dtype=np.uint8)
    codes[:, :32] = 1
    if k > 64:
        codes[:, 32:64] = 2
    scales = np.full((k // 32, n), 127, dtype=np.uint8)
    scales[0, :] = 116  # shifted first group underflows in E4M3
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    assert not folded.lossless and folded.inexact_value_count > 0
    package = package_mxfp4_folded_prefill(
        m, n, k, folded, allow_approximate=True,
    )
    rocm_isa.assert_selected(
        package.image.payload, chip="gfx1201",
        pattern=r"v_wmma_f32_16x16x16_\w+",
        require="v_wmma_f32_16x16x16_fp8_fp8",
        what="folded MXFP4 BM256/TM4 prefill",
    )
    assert package.descriptor.provenance["fold_inexact_value_count"] > 0
    buffers = {
        "a": a, "b_folded": folded.weight_bytes,
        "a_scale": a_scale, "row_reference": folded.row_reference,
        "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
    }
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact, {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(
        result, default=str
    )
    expected = np.full((m, n), 32 if k > 64 else 0, dtype=ml_dtypes.bfloat16)
    np.testing.assert_array_equal(buffers["output"], expected)
    exact = mx.exact_weights(codes, scales)
    assert np.any(exact != mx.folded_weights(folded))


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_folded_prefill_rejects_changed_load_time_payload() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = 65, 48, 64
    codes = np.ones((n, k), dtype=np.uint8)
    scales = np.full((k // 32, n), 127, dtype=np.uint8)
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    package = package_mxfp4_folded_prefill(
        m, n, k, folded, allow_approximate=True,
    )
    changed = folded.weight_bytes.copy()
    changed[0, 0] ^= 1
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    result = rt.launch(artifact, {
        "buffers": {
            "a": np.ones((m, k), dtype=np.uint8),
            "b_folded": changed,
            "a_scale": np.ones(m, dtype=np.float32),
            "row_reference": folded.row_reference,
            "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
        },
        "scalars": {"M": m, "N": n, "K": k},
    })
    assert not result["ok"]
    assert "weight_sha256" in json.dumps(result, default=str)
