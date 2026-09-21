"""Owning-device proof for the exact gfx1201 MXFP4 W4A8 baseline."""
from __future__ import annotations

import json
import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_native import package_mxfp4_w4a8_exact


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("shape", [(17, 19, 64), (32, 32, 128)])
def test_exact_mxfp4_w4a8_package_executes_on_gfx1201(shape: tuple[int, int, int]) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = shape
    rng = np.random.default_rng(1201 + m + n + k)
    # Integer-valued E4M3 inputs and power-of-two scales make every FP32
    # product/sum exact; equality after BF16 rounding then catches layout,
    # nibble, scale-plane, and accumulation-boundary defects without a loose
    # floating tolerance hiding them.
    a_values = rng.integers(-4, 5, size=(m, k)).astype(np.float32)
    a_e4m3 = a_values.astype(ml_dtypes.float8_e4m3fn)
    a_raw = np.ascontiguousarray(a_e4m3.view(np.uint8))
    a_scale = np.exp2(rng.integers(-1, 2, size=m)).astype(np.float32)
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    packed_kn = np.ascontiguousarray(mx.pack_e2m1_codes(codes).T)
    b_scale = rng.integers(124, 131, size=(k // 32, n), dtype=np.uint8)
    b_scale[0, ::7] = 0  # reserved zero-block spelling is executable semantics
    expected = (
        (a_e4m3.astype(np.float32) * a_scale[:, None])
        @ mx.exact_weights(codes, b_scale).T
    ).astype(ml_dtypes.bfloat16)
    output = np.zeros((m, n), dtype=ml_dtypes.bfloat16)

    package = package_mxfp4_w4a8_exact(m, n, k)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact,
        {
            "buffers": {
                "a": a_raw,
                "b_packed": packed_kn,
                "a_scale": a_scale,
                "b_scale": b_scale,
                "output": output,
            },
            "scalars": {"M": m, "N": n, "K": k},
        },
    )
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(
        result, default=str
    )
    np.testing.assert_array_equal(output, expected)
