"""Exact-gfx1201 proof for device-owned MXFP4 HIP graph capture/replay."""
from __future__ import annotations

import ctypes
import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_graph import PackedFoldedGraphSession
from tessera.compiler.rocm_mxfp4_packed_folded import (
    folded_oracle_from_packed,
    package_mxfp4_packed_folded_prefill,
    prepare_packed_folded_payload,
)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("m,n,k", [(65, 48, 64), (257, 80, 128)])
@pytest.mark.parametrize("lossy", [False, True])
def test_graph_capture_replay_and_same_stream_device_producer(
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
    folded = folded_oracle_from_packed(payload)
    weights = mx.folded_weights(folded)
    a = np.full((m, k), 0x38, dtype=np.uint8)
    a[:, 1::2] = 0x30
    a_scale = np.ones(m, dtype=np.float32)

    def oracle(activation: np.ndarray) -> np.ndarray:
        fp32 = activation.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        return ((fp32 @ weights.T) * a_scale[:, None]).astype(ml_dtypes.bfloat16)

    with PackedFoldedGraphSession(package, payload, m) as graph:
        graph.upload_inputs(a, a_scale)
        graph.capture()
        assert graph.receipt()["capture_nodes"] == (0,)
        graph.replay()
        np.testing.assert_array_equal(graph.read_output(), oracle(a))

        # An external producer uses the leased A pointer and the same stream.
        changed = np.full((m, k), 0x38, dtype=np.uint8)
        changed[:, ::3] = 0xB8
        lease = graph.buffers()["a"]
        hip = graph._hip
        assert hip.hipMemcpyAsync(
            ctypes.c_void_p(lease.pointer),
            changed.ctypes.data_as(ctypes.c_void_p), changed.nbytes, 1,
            ctypes.c_void_p(graph.stream_pointer),
        ) == 0
        graph.mark_device_inputs_ready(stream_pointer=graph.stream_pointer)
        graph.replay()
        np.testing.assert_array_equal(graph.read_output(), oracle(changed))
        receipt = graph.receipt()
        assert receipt["graph_captures"] == 1
        assert receipt["graph_replays"] == 2
        assert receipt["automatic_selection"] is False

    with pytest.raises(RuntimeError, match="closed"):
        _ = lease.pointer
