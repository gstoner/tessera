"""The fixed-count policy experiment cannot retry or launder synthetic data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from tests.unit.test_apple_route_selector import _report, _stable_row


def _module():
    path = Path(__file__).resolve().parents[2] / "benchmarks/apple_gpu/compare_cross_run_policy.py"
    spec = importlib.util.spec_from_file_location("compare_cross_run_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reports():
    return [_report(_stable_row("mps", 1000, 1000),
                    _stable_row("simdgroup_matrix", 600, 600)) for _ in range(8)]


def test_sensitivity_keeps_raw_inputs_and_losing_run_floor():
    reports = _reports()
    before = copy.deepcopy(reports)
    result = _module().compare_reports(reports, incumbents={"matmul": "mps"})
    assert reports == before
    assert result["promotion_allowed"] is False
    assert result["default_policy_changed"] is False
    observed, stall, loss = result["scenarios"]
    assert observed["synthetic"] is False
    assert stall["synthetic"] is loss["synthetic"] is True
    bounds = [policy["decisions"][0]["route_evidence"]["simdgroup_matrix"]
              ["speedup_lower_confidence_bound"] for policy in stall["policies"].values()]
    assert bounds[0] < bounds[1]  # Strong wins can still pass both policies.
    for policy in observed["policies"].values():
        assert all(row["status"] == "promote_candidate" for row in policy["decisions"])
    for policy in loss["policies"].values():
        assert all(row["selected_route"] == "mps" for row in policy["decisions"])


@pytest.mark.parametrize("count", [0, 5, 7, 9])
def test_run_count_cannot_be_selected_after_observation(count):
    with pytest.raises(ValueError, match="exactly eight"):
        _module().compare_reports((_reports() * 2)[:count])


def test_collection_writes_plan_first_and_runs_exactly_eight_children(tmp_path, monkeypatch):
    module = _module()
    output = tmp_path / "experiment"
    calls = []
    library = tmp_path / "runtime.dylib"
    library.write_bytes(b"fixture")
    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(library))

    def child(command, *, check, timeout):
        assert json.loads((output / "plan.json").read_text()) == module.PLAN
        assert check and timeout == 300
        calls.append(command)
        Path(command[-1]).write_text(json.dumps(_reports()[0]))

    monkeypatch.setattr(module.subprocess, "run", child)
    module.collect(output)
    assert [int(command[-3]) for command in calls] == list(range(8))
    result = json.loads((output / "comparison.json").read_text())
    assert len(result["source_reports"]) == 8
    assert all(len(row["sha256"]) == 64 for row in result["source_reports"])
    with pytest.raises(FileExistsError):
        module.collect(output)
    assert len(calls) == 8


def test_failed_process_is_not_retried_or_replaced(tmp_path, monkeypatch):
    module = _module()
    output = tmp_path / "failed"
    calls = []
    library = tmp_path / "runtime.dylib"
    library.write_bytes(b"fixture")
    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(library))

    def child(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 300)

    monkeypatch.setattr(module.subprocess, "run", child)
    with pytest.raises(subprocess.TimeoutExpired):
        module.collect(output)
    assert len(calls) == 1
    assert (output / "plan.json").is_file()
    assert not (output / "comparison.json").exists()


def test_changed_runtime_invalidates_completed_collection(tmp_path, monkeypatch):
    module = _module()
    output = tmp_path / "changed"
    library = tmp_path / "runtime.dylib"
    library.write_bytes(b"before")
    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(library))

    def child(command, **kwargs):
        Path(command[-1]).write_text(json.dumps(_reports()[0]))
        library.write_bytes(b"after")

    monkeypatch.setattr(module.subprocess, "run", child)
    with pytest.raises(RuntimeError, match="changed during collection"):
        module.collect(output)
    assert len(list(output.glob("run-*.json"))) == 8
    assert not (output / "comparison.json").exists()


def test_owning_device_packet_replays_and_preserves_raw_report_hashes():
    root = Path(__file__).resolve().parents[2]
    packet = root / "benchmarks/baselines/apple7_cross_run_policy_20260904"
    summary = json.loads((packet / "summary.json").read_text())
    reports = []
    for source in summary["source_reports"]:
        raw = (packet / source["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source["sha256"]
        reports.append(json.loads(raw))
    replay = _module().compare_reports(reports)
    assert replay["plan"] == summary["plan"]
    for expected, actual in zip(summary["scenarios"], replay["scenarios"], strict=True):
        assert expected["scenario"] == actual["scenario"]
        assert expected["changed_decisions"] == actual["changed_decisions"]
        for estimator, policy in expected["policies"].items():
            for saved, row in zip(policy["decisions"], actual["policies"][estimator]["decisions"], strict=True):
                assert saved == {key: row[key] for key in saved}
    assert summary["promotion_allowed"] is False
