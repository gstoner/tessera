"""Exact-gfx1201 proof of the manually selected resident packed MXFP4 route."""
from __future__ import annotations

import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_packed_folded import (
    folded_oracle_from_packed,
    package_mxfp4_packed_folded_prefill,
    prepare_packed_folded_payload,
)
from tessera.compiler.rocm_mxfp4_resident import PackedFoldedResidentSession


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("m,n,k", [(65, 48, 64), (257, 80, 128)])
@pytest.mark.parametrize("lossy", [False, True])
def test_resident_packed_folded_matches_oracle(
    m: int, n: int, k: int, lossy: bool,
) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    codes = np.resize(np.arange(16, dtype=np.uint8), (n, k))
    scales = np.full((k // 32, n), 127, dtype=np.uint8)
    if lossy:
        scales[0, ::3] = 116
    payload = prepare_packed_folded_payload(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    package = package_mxfp4_packed_folded_prefill(
        m, payload, permute_decode=True, batched_loads=True,
    )
    a = np.full((m, k), 0x38, dtype=np.uint8)
    a[:, 1::2] = 0x30
    a_scale = np.ones(m, dtype=np.float32)
    a_scale[1::3] = 0.5
    folded = folded_oracle_from_packed(payload)
    activation = a.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    expected = (
        (activation @ mx.folded_weights(folded).T) * a_scale[:, None]
    ).astype(ml_dtypes.bfloat16)
    with PackedFoldedResidentSession(package, payload, m) as session:
        output = session.run_host(a, a_scale)
        np.testing.assert_array_equal(output, expected)
        output_again = session.run_host(a, a_scale)
        np.testing.assert_array_equal(output_again, expected)
        assert session.receipt()["module_loads"] == 1
        assert session.receipt()["kernel_launches"] == 2
    if not lossy:
        exact = (
            (activation @ mx.exact_weights(codes, scales).T) * a_scale[:, None]
        ).astype(ml_dtypes.bfloat16)
        np.testing.assert_array_equal(output, exact)
