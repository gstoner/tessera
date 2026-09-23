"""Exact-device BF16 proof for the manually selected packed folded ABI."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_packed_folded import (
    compile_packed_folded_scaled_matmul,
    folded_oracle_from_packed,
    package_mxfp4_packed_folded_prefill,
    prepare_packed_folded_payload,
)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("shape", [(65, 48, 64), (257, 80, 128)])
@pytest.mark.parametrize("lossy", [False, True])
@pytest.mark.parametrize(
    "integer_decode,batched_loads,batched_a_loads,reuse_pair_scales,permute_decode,vector_pair_loads",
    [(False, False, False, False, False, False),
     (True, False, False, False, False, False),
     (True, True, False, False, False, False),
     (True, False, True, False, False, False),
     (True, True, True, False, False, False),
     (True, True, False, True, False, False),
     (False, True, False, False, True, False),
     pytest.param(False, False, False, False, True, True, id="vector_pair")],
)
def test_packed_folded_prefill_matches_declared_oracle(
    shape: tuple[int, int, int], lossy: bool,
    integer_decode: bool, batched_loads: bool,
    batched_a_loads: bool, reuse_pair_scales: bool, permute_decode: bool,
    vector_pair_loads: bool,
) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = shape
    a = np.full((m, k), 0x38, dtype=np.uint8)  # E4M3 +1
    a_scale = np.ones(m, dtype=np.float32)
    codes = np.ones((n, k), dtype=np.uint8)  # E2M1 +0.5
    scales = np.full((k // 32, n), 127, dtype=np.uint8)
    if lossy:
        scales[0] = 116
    payload = prepare_packed_folded_payload(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    assert payload.lossless is not lossy
    package = package_mxfp4_packed_folded_prefill(
        m, payload, integer_decode=integer_decode, batched_loads=batched_loads,
        batched_a_loads=batched_a_loads, reuse_pair_scales=reuse_pair_scales,
        permute_decode=permute_decode, vector_pair_loads=vector_pair_loads,
    )
    assert package.descriptor.abi_id.endswith("approx_bm256_tm4.v1")
    output = np.zeros((m, n), dtype=ml_dtypes.bfloat16)
    # Probe the low-level submit path separately from the explicit launcher.
    result = rt._submit_rocm_mxfp4_w4a8(
        package.image, package.descriptor,
        {
            "a": a, "b_packed": payload.weight_bytes,
            "a_scale": a_scale, "scale_plane": payload.scale_plane,
            "output": output,
        }, {"M": m, "N": n, "K": k},
    )
    assert result is output
    folded = folded_oracle_from_packed(payload)
    approximate = mx.folded_weights(folded).sum(axis=1)
    expected = np.broadcast_to(
        approximate.astype(ml_dtypes.bfloat16), (m, n),
    )
    np.testing.assert_array_equal(output, expected)
    if not lossy:
        exact = mx.exact_weights(codes, scales).sum(axis=1)
        np.testing.assert_array_equal(
            output, np.broadcast_to(exact.astype(ml_dtypes.bfloat16), (m, n)),
        )


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_packed_graph_to_target_receipt_and_explicit_launch() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    tessera_opt = os.environ.get("TESSERA_OPT")
    assert tessera_opt, "packed frontend proof requires TESSERA_OPT"
    m, n, k = 65, 48, 64
    a = np.full((m, k), 0x38, dtype=np.uint8)
    a_scale = np.ones(m, dtype=np.float32)
    payload = prepare_packed_folded_payload(
        mx.pack_e2m1_codes(np.ones((n, k), dtype=np.uint8)),
        np.full((k // 32, n), 127, dtype=np.uint8),
        allow_approximate=True,
    )
    program = compile_packed_folded_scaled_matmul(
        a, a_scale, payload, tessera_opt=Path(tessera_opt),
    )
    package = program.package
    receipt = program.route_receipt
    assert receipt["execution_state"] == "manual_executable_candidate"
    assert (receipt["block_m"], receipt["block_n"], receipt["block_k"]) == (
        256, 64, 64,
    )
    assert receipt["schedule_hash"] in package.target_ir
    assert receipt["target_abi"] == package.descriptor.abi_id
    assert receipt["hsaco_sha256"] == hashlib.sha256(package.image.payload).hexdigest()
    output = np.zeros((m, n), dtype=ml_dtypes.bfloat16)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact,
        {"buffers": {
            "a": a, "b_packed": payload.weight_bytes,
            "a_scale": a_scale, "scale_plane": payload.scale_plane,
            "output": output,
        }, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert result["ok"] and result["execution_kind"] == "native_gpu"
    np.testing.assert_array_equal(
        output, np.full((m, n), k * 0.5, dtype=ml_dtypes.bfloat16),
    )


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize(
    "integer_decode,batched_loads,batched_a_loads,reuse_pair_scales,permute_decode,vector_pair_loads",
    [(False, False, False, False, False, False),
     (True, False, False, False, False, False),
     (True, True, False, False, False, False),
     (True, False, True, False, False, False),
     (True, True, True, False, False, False),
     (True, True, False, True, False, False),
     (False, True, False, False, True, False),
     pytest.param(False, False, False, False, True, True, id="vector_pair")],
)
def test_packed_folded_prefill_all_codes_scale_deltas_and_zero_blocks(
    integer_decode: bool, batched_loads: bool,
    batched_a_loads: bool, reuse_pair_scales: bool, permute_decode: bool,
    vector_pair_loads: bool,
) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = 65, 48, 64
    rows = np.arange(n, dtype=np.uint8)[:, None]
    cols = np.arange(k, dtype=np.uint8)[None, :]
    codes = np.ascontiguousarray((rows * 7 + cols * 3) % 16)
    scales = np.full((2, n), 127, dtype=np.uint8)
    scales[0] = 127 - (np.arange(n) % 13).astype(np.uint8)
    scales[0, ::11] = 0  # reserved zero-block semantics
    payload = prepare_packed_folded_payload(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    assert not payload.lossless
    a = np.full((m, k), 0x38, dtype=np.uint8)
    a[:, 1::2] = 0x30  # 0.5; breaks a K permutation that preserves row sums
    a[1::3, ::4] = 0xB8  # -1; exercises cancellation and signed fragments
    a_scale = np.ones(m, dtype=np.float32)
    package = package_mxfp4_packed_folded_prefill(
        m, payload, integer_decode=integer_decode, batched_loads=batched_loads,
        batched_a_loads=batched_a_loads, reuse_pair_scales=reuse_pair_scales,
        permute_decode=permute_decode, vector_pair_loads=vector_pair_loads,
    )
    output = np.zeros((m, n), dtype=ml_dtypes.bfloat16)
    rt._submit_rocm_mxfp4_w4a8(
        package.image, package.descriptor,
        {"a": a, "b_packed": payload.weight_bytes, "a_scale": a_scale,
         "scale_plane": payload.scale_plane, "output": output},
        {"M": m, "N": n, "K": k},
    )
    activation = a.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    folded = folded_oracle_from_packed(payload)
    expected = (activation @ mx.folded_weights(folded).T).astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(output, expected)
