#!/usr/bin/env python3
"""Exact gfx1201 capacity bound for the pinned GPT-OSS-20B MXFP4 inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_prefill_memory import assess_prefill_weight_residency
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.benchmark_gfx1201_mxfp4_prefill_sweep import _model_layers


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    ROOT / "benchmarks/baselines/gfx1201_mxfp4_gpt_oss_20b_20260923/model_inventory.json"
)
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
PINNED_INVENTORY_SHA256 = "de3e616db5f91cd896b7a3047d447297dca5af5fbdf3f73404eecb6681f78245"


def record(inventory_path: Path = DEFAULT_INVENTORY) -> dict[str, Any]:
    # Validate the pin and every shape before querying HIP.
    layers, inventory_sha256 = _model_layers(inventory_path)
    source_bytes = inventory_path.read_bytes()
    if inventory_sha256 != PINNED_INVENTORY_SHA256 or (
        hashlib.sha256(source_bytes).hexdigest() != inventory_sha256
    ):
        raise ValueError("capacity proof requires the pinned GPT-OSS-20B inventory bytes")
    source = json.loads(source_bytes)
    if source.get("model") != "openai/gpt-oss-20b" or (
        source.get("checkpoint_revision") != MODEL_REVISION
    ):
        raise ValueError("capacity proof requires the pinned GPT-OSS-20B checkpoint")
    if sorted((layer.n, layer.k, layer.count) for layer in layers) != [
        (2880, 2880, 768), (5760, 2880, 768),
    ]:
        raise ValueError("capacity proof requires all 24 x 32 expert projections")
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("capacity proof requires selected gfx1201")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("capacity proof requires usable HIP")
    device = base._selected_device_name(hip)
    memory = rt._rocm_device_memory_envelope()
    capacity = memory["capacity_bytes"]
    # Capacity itself is an optimistic upper bound on any post-load headroom.
    accounting = assess_prefill_weight_residency(
        layers, available_extra_bytes=capacity,
    )
    expanded_weights_only = sum(layer.n * layer.k * layer.count for layer in layers)
    if expanded_weights_only <= capacity:
        verdict = "not_ruled_out_by_absolute_capacity"
        selection_state = "refused_no_loaded_model_budget"
    else:
        verdict = "refused_full_expansion_exceeds_device_capacity"
        selection_state = "refused_full_expansion"
    return {
        "schema": "tessera.rocm.gfx1201_gpt_oss_20b_mxfp4_capacity.v1",
        "sync_key": "GFX1201-GPT-OSS-20B-MXFP4-CAPACITY-2026-09-23",
        "model": source["model"],
        "checkpoint_revision": MODEL_REVISION,
        "checkpoint_total_tensor_bytes": source["checkpoint_total_tensor_bytes"],
        "inventory_sha256": inventory_sha256,
        "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_revision": base._git_revision(),
        "architecture": rt._rocm_live_arch(),
        "device": device,
        "memory_snapshot": memory,
        "expert_projection_count": sum(layer.count for layer in layers),
        "expanded_weights_only_bytes": expanded_weights_only,
        "weight_accounting": accounting,
        "capacity_verdict": verdict,
        "model_load_state": "not_loaded_absolute_bound_only",
        "selection_state": selection_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    packet = record(args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
