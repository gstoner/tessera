"""Pinned GPT-OSS-20B MXFP4 projection geometry and absolute-capacity logic."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.rocm import record_gpt_oss_20b_mxfp4_capacity as capacity
from tessera.compiler.rocm_mxfp4_prefill_memory import assess_prefill_weight_residency


def test_pinned_checkpoint_inventory_counts_all_expert_projections() -> None:
    source = json.loads(capacity.DEFAULT_INVENTORY.read_text())
    layers, digest = capacity._model_layers(capacity.DEFAULT_INVENTORY)
    assert len(digest) == 64
    assert source["checkpoint_revision"] == capacity.MODEL_REVISION
    assert source["checkpoint_total_tensor_bytes"] == 13_761_264_768
    assert source["config_sha256"] and source["index_sha256"]
    assert len(source["shard_headers"]) == 3
    assert sorted((layer.n, layer.k, layer.count) for layer in layers) == [
        (2880, 2880, 768), (5760, 2880, 768),
    ]
    accounted = assess_prefill_weight_residency(
        layers, available_extra_bytes=16_974_905_344,
    )
    # Tessera retains a derived row-reference byte per output row in addition
    # to the checkpoint's packed block and K32 scale bytes.
    assert accounted["packed_weight_and_scale_bytes"] == 10_158_981_120
    assert accounted["expanded_weight_and_reference_bytes"] == 19_116_933_120
    assert accounted["budget_allows_trial"] is False


def test_device_capacity_refuses_full_expansion_without_model_load() -> None:
    with (
        patch.object(capacity.rt, "_rocm_live_arch", return_value="gfx1201"),
        patch.object(capacity.rt, "_load_hip_for_launch") as load,
        patch.object(capacity.rt, "_rocm_device_memory_envelope", return_value={
            "capacity_bytes": 16_974_905_344, "free_bytes": 15_685_328_896,
        }),
        patch.object(capacity.base, "_selected_device_name", return_value="RX 9070 XT"),
        patch.object(capacity.base, "_git_revision", return_value="pinned-test"),
    ):
        load.return_value.hipInit.return_value = 0
        packet = capacity.record()
    assert packet["expert_projection_count"] == 1536
    assert packet["expanded_weights_only_bytes"] == 19_110_297_600
    assert packet["expanded_weights_only_bytes"] > packet["memory_snapshot"]["capacity_bytes"]
    assert packet["capacity_verdict"] == "refused_full_expansion_exceeds_device_capacity"
    assert packet["model_load_state"] == "not_loaded_absolute_bound_only"


def test_larger_gfx1201_device_still_refuses_without_loaded_budget() -> None:
    with (
        patch.object(capacity.rt, "_rocm_live_arch", return_value="gfx1201"),
        patch.object(capacity.rt, "_load_hip_for_launch") as load,
        patch.object(capacity.rt, "_rocm_device_memory_envelope", return_value={
            "capacity_bytes": 32_000_000_000, "free_bytes": 28_000_000_000,
        }),
        patch.object(capacity.base, "_selected_device_name", return_value="gfx1201 peer"),
        patch.object(capacity.base, "_git_revision", return_value="pinned-test"),
    ):
        load.return_value.hipInit.return_value = 0
        packet = capacity.record()
    assert packet["capacity_verdict"] == "not_ruled_out_by_absolute_capacity"
    assert packet["selection_state"] == "refused_no_loaded_model_budget"


def test_wrong_model_pin_refused_before_hip(tmp_path: Path) -> None:
    source = json.loads(capacity.DEFAULT_INVENTORY.read_text())
    source["checkpoint_revision"] = "wrong"
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(source))
    with (
        patch.object(capacity.rt, "_rocm_live_arch") as arch,
        pytest.raises(ValueError, match="pinned GPT-OSS-20B"),
    ):
        capacity.record(path)
    arch.assert_not_called()


def test_exact_device_packet_binds_inventory_recorder_and_absolute_bound() -> None:
    path = capacity.DEFAULT_INVENTORY.parent / "evidence.json"
    packet = json.loads(path.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_gpt_oss_20b_mxfp4_capacity.v1"
    assert packet["sync_key"] == "GFX1201-GPT-OSS-20B-MXFP4-CAPACITY-2026-09-23"
    assert packet["inventory_sha256"] == hashlib.sha256(
        capacity.DEFAULT_INVENTORY.read_bytes()
    ).hexdigest()
    assert packet["recorder_sha256"] == hashlib.sha256(
        Path(capacity.__file__).read_bytes()
    ).hexdigest()
    assert packet["architecture"] == "gfx1201"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["model_load_state"] == "not_loaded_absolute_bound_only"
    assert packet["capacity_verdict"] == "refused_full_expansion_exceeds_device_capacity"
    assert packet["expanded_weights_only_bytes"] > packet["memory_snapshot"]["capacity_bytes"]
    assert packet["weight_accounting"]["both_resident_bytes"] > (
        packet["memory_snapshot"]["capacity_bytes"]
    )
