"""Opt-in exact gfx1201 BF16 proof for the distinct packed W4A4 probe."""

from __future__ import annotations

import json
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_quark import decode_quark_low_even_hypothesis
from tessera.compiler.rocm_mxfp4_quark_native import launch_quark_w4a4_probe


PACKET = Path(__file__).resolve().parents[3] / ("benchmarks/baselines/gfx1201_quark_w4a4_probe_20260923/reference.json")


def _sample_buffers(case: dict[str, object], packet: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "a_packed": np.frombuffer(bytes.fromhex(packet["activation_packed_hex"]), np.uint8).reshape(1, 16).copy(),
        "b_packed": np.stack([np.frombuffer(bytes.fromhex(row), np.uint8) for row in case["weight_packed_hex"]]).copy(),
        "a_scale": np.array([[int(packet["activation_scale_hex"], 16)]], dtype=np.uint8),
        "b_scale": np.array([[int(code, 16)] for code in case["weight_scale_hex"]], dtype=np.uint8),
        "output": np.zeros((1, 2), dtype=ml_dtypes.bfloat16),
    }


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("case_index", [0, 1], ids=["gate", "down"])
def test_pinned_independent_w4a4_projection_on_gfx1201(case_index: int) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    packet = json.loads(PACKET.read_text())
    case = packet["cases"][case_index]
    buffers = _sample_buffers(case, packet)
    result = launch_quark_w4a4_probe(buffers)
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    np.testing.assert_array_equal(buffers["output"].view(np.uint16)[0], case["bf16_bits"])
    print(f"{case['projection']}: launched_hsaco_sha256={result['hsaco_sha256']}")


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_w4a4_ragged_k64_and_scale_cancellation_on_gfx1201() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    rng = np.random.default_rng(1201)
    m, n, k = 5, 3, 64
    buffers = {
        "a_packed": rng.integers(0, 256, size=(m, k // 2), dtype=np.uint8),
        "b_packed": rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8),
        "a_scale": np.full((m, k // 32), 127, dtype=np.uint8),
        "b_scale": np.full((n, k // 32), 127, dtype=np.uint8),
        "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
    }
    buffers["a_scale"][0, 0] = 254
    buffers["b_scale"][:, 0] = 1  # Combined exponent is 1, not overflow.
    buffers["a_packed"][1, :] = 0  # A zero row stays zero at every scale.
    expected = (
        decode_quark_low_even_hypothesis(buffers["a_packed"], buffers["a_scale"])
        @ decode_quark_low_even_hypothesis(buffers["b_packed"], buffers["b_scale"]).T
    ).astype(ml_dtypes.bfloat16)
    result = launch_quark_w4a4_probe(buffers)
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    np.testing.assert_array_equal(buffers["output"], expected)
    print(f"ragged_k64: launched_hsaco_sha256={result['hsaco_sha256']}")
