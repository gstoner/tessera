"""MODEL-WEIGHT-PHYS-1 persistent packed-byte operand contracts."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

from tessera import runtime as rt
from tessera.stdlib.quant import ingest_packed_weight_bytes


_EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs/audit/evidence/models/model_weight_phys/20260812-wsl-gfx1151.json"
)


class _FakeHip:
    def __init__(self) -> None:
        self.next_pointer = 0x1000
        self.allocations: list[int] = []
        self.copies: list[tuple[int, int]] = []
        self.frees: list[int] = []

    def hipInit(self, _: int) -> int:
        return 0

    def hipMalloc(self, output: object, byte_count: int) -> int:
        pointer = ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))
        pointer[0] = ctypes.c_void_p(self.next_pointer)
        self.next_pointer += 0x1000
        self.allocations.append(int(byte_count))
        return 0

    def hipMemcpy(
        self, destination: ctypes.c_void_p, _: ctypes.c_void_p,
        byte_count: int, direction: int,
    ) -> int:
        self.copies.append((int(byte_count), int(direction)))
        assert destination.value
        return 0

    def hipFree(self, pointer: ctypes.c_void_p) -> int:
        self.frees.append(int(pointer.value or 0))
        return 0


def test_gfx1151_upload_owns_two_persistent_typed_operands(monkeypatch):
    codes = np.array([[0x10, 0x32], [0x54, 0x76]], dtype=np.uint8)
    packed = ingest_packed_weight_bytes(
        codes,
        np.ones((2, 2), dtype=np.float32),
        dtype="int4",
        shape=(4, 2),
        group_size=2,
    )
    hip = _FakeHip()
    monkeypatch.setattr(rt, "_rocm_chip", lambda: "gfx1151")
    monkeypatch.setattr(rt, "_load_hip_for_launch", lambda: hip)

    operand = rt.upload_rocm_packed_weight(packed)
    assert hip.allocations == [4, 16]
    assert hip.copies == [(4, 1), (16, 1)]
    assert operand.descriptor["operand_count"] == 2
    assert operand.descriptor["residency"] == "device"
    assert operand.descriptor["full_weight_materialization"] == "prohibited"
    assert operand.content_digest == packed.content_digest

    operand.close()
    operand.close()
    assert len(hip.frees) == 2


def test_packed_weight_upload_is_exact_architecture_gated(monkeypatch):
    packed = ingest_packed_weight_bytes(
        np.zeros(4, dtype=np.uint8),
        np.ones((2, 2), dtype=np.float32),
        dtype="int4",
        shape=(4, 2),
        group_size=2,
    )
    monkeypatch.setattr(rt, "_rocm_chip", lambda: "gfx1200")
    try:
        rt.upload_rocm_packed_weight(packed)
    except ValueError as error:
        assert "proven only for gfx1151" in str(error)
    else:
        raise AssertionError("unproven ROCm architecture did not fail closed")


def test_gfx1151_packet_records_residency_lineage_and_promotion_blocker():
    packet = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    assert packet["schema"] == "tessera.rocm.packed-consumers.v2"
    assert packet["target"] == "gfx1151"
    assert packet["evidence_eligibility"] == "correctness_and_regression_only"
    row = packet["rows"][0]
    assert row["execution_kind"] == "native_gpu"
    assert row["weight_residency"] == "persistent_device"
    assert row["weight_upload_timing"] == "excluded_one_time_model_load"
    assert row["full_weight_materialization"] == "prohibited"
    assert len(row["weight_content_digest"]) == 64
    assert len(packet["kernel_image_sha256"]) == 64
