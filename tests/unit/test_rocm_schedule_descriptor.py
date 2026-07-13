from __future__ import annotations

import json

import pytest

from tessera.compiler.ownership_topology import (
    OwnershipProblem, OwnershipTopology, select_ownership_topology)
from tessera.compiler.rocm_profiler_experiment import collect_native_counters
from tessera.compiler.rocm_schedule import select_rocm_gemm_schedule


def test_gfx1151_schedule_preserves_measured_macro_tile_and_models_resources():
    small = select_rocm_gemm_schedule(257, 509, 127, arch="gfx1151")
    large = select_rocm_gemm_schedule(1024, 1024, 1024, arch="gfx1151")
    assert small.macro_tile == (2, 4)
    assert large.macro_tile == (3, 4)
    assert large.target_ir_attrs()["schedule_ownership"] == "wave"
    assert large.target_ir_attrs()["schedule_lds_layout"] == "swizzle"
    assert large.vgpr_estimate > small.vgpr_estimate
    assert select_rocm_gemm_schedule(
        64, 64, 64, dtype="int4", arch="gfx1151").vgpr_estimate > 0


@pytest.mark.parametrize(
    "rows,candidates,top_k,expected",
    [(64, 4096, 4, OwnershipTopology.WAVE),
     (1024, 4096, 4, OwnershipTopology.THREAD),
     (64, 512, 4, OwnershipTopology.THREAD)],
)
def test_ownership_selector_encodes_measured_sparse_crossover(
    rows, candidates, top_k, expected
):
    decision = select_ownership_topology(
        OwnershipProblem(rows, candidates, candidates, top_k),
        target="gfx1151", operation="selection")
    assert decision.topology is expected
    assert "gfx1151" in decision.calibration


def test_native_counter_runner_records_exact_ab_identity(tmp_path, monkeypatch):
    calls = []

    class Completed:
        returncode = 0

    def fake_run(cmd, check):
        calls.append((cmd, check))
        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "tessera.compiler.rocm_profiler_experiment.is_wsl", lambda: False)
    run = collect_native_counters(
        "G6-A", "candidate", ["bench", "--shape", "1024"],
        counters=["SQ_WAVES", "SQ_INSTS_VALU"], output_directory=tmp_path,
        rocprofv3="/opt/rocm/bin/rocprofv3", enabled=True)
    assert "--pmc" in calls[0][0]
    assert calls[0][0][-4:] == ("--", "bench", "--shape", "1024")
    payload = json.loads((tmp_path / "tessera_rocm6_run.json").read_text())
    assert payload["experiment"] == "G6-A"
    assert payload["variant"] == "candidate"
    assert payload["native"] is True
    assert run.returncode == 0
    assert run.status == "collected"


def test_native_counter_switch_defaults_off_without_spawning(tmp_path, monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("disabled profiler switch must not spawn rocprofv3")

    monkeypatch.setattr("subprocess.run", fail_run)
    run = collect_native_counters(
        "G6-B", "production", [], counters=[], output_directory=tmp_path)
    assert run.status == "disabled"
    assert run.returncode is None
    assert run.as_metadata_dict()["native"] is False


def test_native_counter_switch_rejects_wsl(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tessera.compiler.rocm_profiler_experiment.is_wsl", lambda: True)
    with pytest.raises(RuntimeError, match="unsupported under WSL"):
        collect_native_counters(
            "G6-C", "candidate", ["bench"], counters=["SQ_WAVES"],
            output_directory=tmp_path, enabled=True,
            rocprofv3="/opt/rocm/bin/rocprofv3")
