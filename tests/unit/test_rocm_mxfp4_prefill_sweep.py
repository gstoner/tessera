"""The exact-device prefill sweep cannot become a selector or invent a model budget."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.rocm import benchmark_gfx1201_mxfp4_prefill_sweep as sweep


def test_sweep_covers_both_N_families_and_three_M_regimes() -> None:
    assert len(sweep.SWEEP) == 6
    assert {case.m for case in sweep.SWEEP} == {128, 256, 1024}
    assert {case.n for case in sweep.SWEEP} == {5120, 17408}
    assert all(case.k % 64 == 0 for case in sweep.SWEEP)


def test_sweep_refuses_to_infer_model_budget_from_idle_memory() -> None:
    with (
        patch.object(sweep.rt, "_rocm_live_arch", return_value="gfx1201"),
        patch.object(sweep, "_memory_snapshot", side_effect=[
            {"capacity_bytes": 16_000, "free_bytes": 12_000},
            {"capacity_bytes": 16_000, "free_bytes": 12_000},
        ]),
        patch.object(sweep, "benchmark", return_value={"rows": []}),
    ):
        packet = sweep.sweep(Path("radiance.so"), radiance_revision="pinned", cases=())
    assert packet["model_budget"]["selection_state"] == "refused_no_model_budget"
    assert packet["memory_after_bytes"]["free_bytes"] == 12_000
    assert packet["selection_state"] == "manual_evidence_only"


def test_inventory_bytes_remain_non_authorizing_without_loaded_model(tmp_path: Path) -> None:
    inventory = tmp_path / "model.json"
    inventory.write_text(json.dumps({
        "schema": "tessera.mxfp4.model_layer_inventory.v1",
        "layers": [{"n": 17408, "k": 5120, "count": 32}],
    }))
    with (
        patch.object(sweep.rt, "_rocm_live_arch", return_value="gfx1201"),
        patch.object(sweep, "_memory_snapshot", side_effect=[
            {"capacity_bytes": 8_000_000_000, "free_bytes": 4_000_000_000},
            {"capacity_bytes": 8_000_000_000, "free_bytes": 4_000_000_000},
        ]),
        patch.object(sweep, "benchmark", return_value={"rows": []}),
    ):
        packet = sweep.sweep(
            Path("radiance.so"), radiance_revision="pinned", cases=(),
            model_inventory=inventory, reserve_bytes=1_000_000_000,
        )
    budget = packet["model_budget"]
    assert budget["weight_accounting"]["budget_allows_trial"] is False
    assert budget["selection_state"] == "refused_no_model_budget"
    with pytest.raises(ValueError, match="unsupported schema"):
        invalid = tmp_path / "invalid.json"
        invalid.write_text('{"layers": []}')
        sweep._model_layers(invalid)


@pytest.mark.parametrize("contents,expected", [
    (None, FileNotFoundError),
    ("{invalid", json.JSONDecodeError),
    ('{"schema":"unknown","layers":[]}', ValueError),
    (json.dumps({
        "schema": "tessera.mxfp4.model_layer_inventory.v1",
        "layers": [{"n": 17, "k": 64, "count": 1}],
    }), ValueError),
])
def test_invalid_inventory_fails_before_gpu_work(
    tmp_path: Path, contents: str | None, expected: type[Exception],
) -> None:
    inventory = tmp_path / "model.json"
    if contents is not None:
        inventory.write_text(contents)
    with (
        patch.object(sweep.rt, "_rocm_live_arch") as arch,
        patch.object(sweep, "_memory_snapshot") as snapshot,
        patch.object(sweep, "benchmark") as timed_benchmark,
        pytest.raises(expected),
    ):
        sweep.sweep(
            Path("radiance.so"), radiance_revision="pinned",
            model_inventory=inventory,
        )
    arch.assert_not_called()
    snapshot.assert_not_called()
    timed_benchmark.assert_not_called()
