"""Independent, pinned projection-slice reference and W4A4 ABI guards."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.rocm_mxfp4_quark import decode_quark_low_even_hypothesis
from tessera.compiler.rocm_mxfp4_quark_native import (
    GFX1201_QUARK_W4A4_PROBE_ABI,
    emit_quark_w4a4_probe_hip,
    validate_quark_w4a4_buffers,
)


PACKET = Path(__file__).resolve().parents[2] / ("benchmarks/baselines/gfx1201_quark_w4a4_probe_20260923/reference.json")


def reference_cases() -> list[dict[str, object]]:
    return json.loads(PACKET.read_text())["cases"]


def test_exact_device_packet_binds_current_generator_and_fixtures() -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = json.loads((PACKET.parent / "evidence.json").read_text())
    bound = {
        "generator_sha256": root / "python/tessera/compiler/rocm_mxfp4_quark_native.py",
        "device_fixture_sha256": root / "tests/device/rocm/test_quark_w4a4_probe.py",
        "reference_sha256": PACKET,
    }
    for field, path in bound.items():
        assert sha256(path.read_bytes()).hexdigest() == evidence[field]
    assert evidence["abi"] == GFX1201_QUARK_W4A4_PROBE_ABI


@pytest.mark.parametrize("case", reference_cases(), ids=lambda case: str(case["projection"]).split(".")[-1])
def test_pinned_independent_projection_slice(case: dict[str, object]) -> None:
    packet = json.loads(PACKET.read_text())
    a = np.frombuffer(bytes.fromhex(packet["activation_packed_hex"]), dtype=np.uint8).reshape(1, 16)
    sa = np.frombuffer(bytes.fromhex(packet["activation_scale_hex"]), dtype=np.uint8).reshape(1, 1)
    b = np.stack([np.frombuffer(bytes.fromhex(row), dtype=np.uint8) for row in case["weight_packed_hex"]])
    sb = np.array([[int(code, 16)] for code in case["weight_scale_hex"]], dtype=np.uint8)
    actual = decode_quark_low_even_hypothesis(a, sa) @ decode_quark_low_even_hypothesis(b, sb).T
    np.testing.assert_array_equal(actual[0], case["fp32_reference"])
    bits = actual.astype(ml_dtypes.bfloat16).view(np.uint16)[0]
    np.testing.assert_array_equal(bits, case["bf16_bits"])
    buffers = {
        "a_packed": a.copy(),
        "b_packed": b.copy(),
        "a_scale": sa.copy(),
        "b_scale": sb.copy(),
        "output": np.zeros((1, 2), dtype=ml_dtypes.bfloat16),
    }
    assert validate_quark_w4a4_buffers(buffers) == (1, 2, 32)


def test_w4a4_probe_is_distinct_from_w4a8_and_refuses_unproved_edges() -> None:
    assert "w4a4" in GFX1201_QUARK_W4A4_PROBE_ABI
    assert "e2m1_low_even_e8m0_k32" in GFX1201_QUARK_W4A4_PROBE_ABI
    source = emit_quark_w4a4_probe_hip()
    assert "a_packed[m * (K / 2)" in source
    assert "b_packed[n * (K / 2)" in source
    buffers = {
        "a_packed": np.zeros((1, 16), dtype=np.uint8),
        "b_packed": np.zeros((1, 16), dtype=np.uint8),
        "a_scale": np.array([[127]], dtype=np.uint8),
        "b_scale": np.array([[127]], dtype=np.uint8),
        "output": np.zeros((1, 1), dtype=ml_dtypes.bfloat16),
    }
    for name in ("a_scale", "b_scale"):
        for code in (0, 255):
            invalid = {**buffers, name: np.array([[code]], dtype=np.uint8)}
            with pytest.raises(ValueError, match="0/255 semantics"):
                validate_quark_w4a4_buffers(invalid)
