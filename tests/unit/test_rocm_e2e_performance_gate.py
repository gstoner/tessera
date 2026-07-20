"""Host-free contract tests for the ROCM-E2E-1 performance recorder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "benchmarks/rocm/benchmark_rocm_e2e_softmax.py"
_BASELINE = (
    _ROOT / "benchmarks/baselines/rocm_gfx1151_e2e_softmax_comparison.json"
)
_SPEC = importlib.util.spec_from_file_location("rocm_e2e_softmax_benchmark", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_benchmark_covers_aligned_ragged_multistride_and_both_dtypes() -> None:
    assert benchmark.CASES == ((32, 17), (128, 256), (64, 1024), (16, 4096))
    assert [item[0] for item in benchmark.DTYPES] == ["fp32", "fp16"]


def test_summary_retains_raw_pairs_and_applies_ten_percent_gate() -> None:
    summary = benchmark._summary([1.0, 1.1, 0.9], [1.05, 1.0, 1.0])
    assert summary["retained_samples_ms"] == [1.0, 1.1, 0.9]
    assert summary["compiler_samples_ms"] == [1.05, 1.0, 1.0]
    assert summary["non_regression_10pct"] is True
    failed = benchmark._summary([1.0, 1.0, 1.0], [1.2, 1.2, 1.2])
    assert failed["non_regression_10pct"] is False


@pytest.mark.parametrize("old,new", [([], []), ([1.0], []), ([1.0], [1.0, 2.0])])
def test_summary_rejects_unpaired_or_empty_samples(old, new) -> None:
    with pytest.raises(ValueError, match="paired timing samples"):
        benchmark._summary(old, new)


def test_recorder_source_has_separate_event_and_runtime_launch_domains() -> None:
    source = _SCRIPT.read_text()
    assert "hipEventElapsedTime" in source
    assert "rt.launch" in source
    assert "selector_changed" in source
    assert 'result["all_non_regression"]' in source
    assert "ThreadPool" not in source and "multiprocessing" not in source


def test_frozen_runtime_artifact_hash_is_cached_after_first_use() -> None:
    from tessera.runtime import RuntimeArtifact

    artifact = RuntimeArtifact(metadata={"target": "cpu", "kind": "identity-test"})
    first = artifact.artifact_hash
    assert artifact.artifact_hash == first
    assert artifact.__dict__["artifact_hash"] == first


def test_recorded_gfx1151_evidence_is_correct_and_passes_non_regression() -> None:
    data = json.loads(_BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["evidence_arch"] == "gfx1151"
    assert data["all_correct"] is True
    assert data["selector_changed"] is False
    assert len(data["rows"]) == len(benchmark.CASES) * len(benchmark.DTYPES)
    assert all(row["device"]["non_regression_10pct"] for row in data["rows"])
    assert data["all_non_regression"] is True
    assert all(row["non_regression"] for row in data["rows"])
    for dtype in ("fp32", "fp16"):
        for route in ("compiler", "retained"):
            resources = data["resources"][dtype][route]
            assert resources["vgpr_spill_count"] == 0
            assert resources["sgpr_spill_count"] == 0
