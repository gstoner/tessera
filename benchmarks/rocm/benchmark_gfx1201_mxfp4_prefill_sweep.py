#!/usr/bin/env python3
"""Broader exact-K32-gated gfx1201 prefill sweep; never selects a route."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_prefill_memory import (
    MXFP4LayerShape, assess_prefill_weight_residency,
)
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.benchmark_gfx1201_mxfp4_safe_epilogue import benchmark


SWEEP = (
    base.Case("prefill", 128, 5120, 8704),
    base.Case("prefill", 256, 5120, 8704),
    base.Case("prefill", 1024, 5120, 8704),
    base.Case("prefill", 128, 17408, 5120),
    base.Case("prefill", 256, 17408, 5120),
    base.Case("prefill", 1024, 17408, 5120),
)


def _model_layers(path: Path) -> tuple[MXFP4LayerShape, ...]:
    """Read an explicit model inventory, not an inferred exemplar."""
    data = json.loads(path.read_text())
    if data.get("schema") != "tessera.mxfp4.model_layer_inventory.v1":
        raise ValueError("model inventory has an unsupported schema")
    rows = data.get("layers")
    if not isinstance(rows, list) or not rows:
        raise ValueError("model inventory requires nonempty layers")
    return tuple(MXFP4LayerShape(**row) for row in rows)


def _memory_snapshot() -> dict[str, int]:
    return rt._rocm_device_memory_envelope()


def sweep(
    radiance_module: Path, *, radiance_revision: str,
    cases: tuple[base.Case, ...] = SWEEP,
    model_inventory: Path | None = None,
    reserve_bytes: int = 0,
    warmup: int = 6, trials: int = 11, iterations: int = 12,
) -> dict[str, Any]:
    if reserve_bytes < 0:
        raise ValueError("reserve_bytes must be nonnegative")
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("prefill sweep requires the selected gfx1201 GPU")
    before = _memory_snapshot()
    packet = benchmark(
        cases, radiance_module, radiance_revision=radiance_revision,
        warmup=warmup, trials=trials, iterations=iterations,
    )
    after = _memory_snapshot()
    inventory: dict[str, Any] = {
        "state": "missing_model_inventory",
        "selection_state": "refused_no_model_budget",
    }
    if model_inventory is not None:
        layers = _model_layers(model_inventory)
        # The snapshot is only a valid *model* budget when this process owns
        # the loaded model. A standalone benchmark cannot make that claim.
        inventory = {
            "state": "inventory_only_no_loaded_model",
            "selection_state": "refused_no_model_budget",
            "inventory_sha256": hashlib.sha256(model_inventory.read_bytes()).hexdigest(),
            "inventory_path": str(model_inventory),
            "reserve_bytes": reserve_bytes,
            # Idle HIP free bytes do not represent headroom after model load.
            "weight_accounting": assess_prefill_weight_residency(
                layers, available_extra_bytes=0,
            ),
        }
    packet.update({
        "schema": "tessera.rocm.gfx1201_mxfp4_prefill_sweep.v1",
        "sync_key": "GFX1201-MXFP4-PREFILL-SWEEP-2026-09-23",
        "sweep_cases": [case.label for case in cases],
        "memory_before_bytes": before,
        "memory_after_bytes": after,
        "model_budget": inventory,
        "selection_state": "manual_evidence_only",
        "sweep_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiance-module", required=True, type=Path)
    parser.add_argument("--radiance-revision", required=True)
    parser.add_argument("--model-inventory", type=Path)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    packet = sweep(
        args.radiance_module, radiance_revision=args.radiance_revision,
        model_inventory=args.model_inventory, reserve_bytes=args.reserve_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "schema": packet["schema"], "sweep_cases": packet["sweep_cases"],
        "model_budget": packet["model_budget"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
