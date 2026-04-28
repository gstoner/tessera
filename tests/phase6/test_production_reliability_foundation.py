"""
Phase 6 — production reliability, stress, chaos, node/rack-scale QA contracts.
"""

from pathlib import Path

import pytest

from tessera.testing import (
    ChaosEvent,
    HealthSnapshot,
    RegressionBaseline,
    ReplayManifest,
)


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guides" / "Tessera_Production_Reliability_And_Chaos_Guide.md"


def test_production_reliability_guide_exists_and_covers_required_behaviors():
    text = GUIDE.read_text(encoding="utf-8")
    required = [
        "Monitoring And Health Checks",
        "Automated Regression Detection",
        "Replay Debugging",
        "Observability And Profiling",
        "Fault Tolerance In Production",
        "Stress Testing",
        "Chaos Testing",
        "Node-Scale QA",
        "Rack-Scale And NVL72 QA",
    ]
    for term in required:
        assert term in text


def test_docs_map_and_qa_guide_link_production_reliability_guide():
    docs_map = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    qa_guide = (
        ROOT / "docs" / "guides" / "Tessera_QA_Reliability_Guide.md"
    ).read_text(encoding="utf-8")
    guide_path = "docs/guides/Tessera_Production_Reliability_And_Chaos_Guide.md"
    assert guide_path in docs_map
    assert guide_path in qa_guide


def test_regression_baseline_accepts_within_thresholds():
    baseline = RegressionBaseline(
        name="matmul_4096_bf16_sm90",
        latency_ms=2.0,
        accuracy=0.999,
        tflops=170.0,
        max_latency_regression=0.20,
        max_accuracy_drop=0.001,
        max_tflops_regression=0.10,
    )
    baseline.validate({"latency_ms": 2.3, "accuracy": 0.9985, "tflops": 160.0})


def test_regression_baseline_rejects_drift():
    baseline = RegressionBaseline(name="collective", latency_ms=10.0)
    with pytest.raises(AssertionError, match="regression detected"):
        baseline.validate({"latency_ms": 13.0})


def test_replay_manifest_requires_reproducibility_fields():
    manifest = ReplayManifest(
        run_id="session1",
        seed=42,
        graph_hash="graph-a",
        schedule_hash="sched-b",
        target="sm90",
        backend="cuda",
        step=12,
        checkpoint="ckpt-12",
    )
    data = manifest.to_dict()
    assert data["seed"] == 42
    assert data["schedule_hash"] == "sched-b"

    bad = ReplayManifest(
        run_id="bad",
        seed=-1,
        graph_hash="",
        schedule_hash="sched",
        target="sm90",
        backend="cuda",
    )
    with pytest.raises(ValueError, match="missing invalid"):
        bad.validate()


def test_health_snapshot_requires_and_bounds_metrics():
    snapshot = HealthSnapshot(
        {
            "gpu_memory_bytes": 40_000_000_000,
            "kernel_latency_ms": 0.4,
            "all_reduce_ms": 3.8,
        }
    )
    snapshot.require("gpu_memory_bytes", "kernel_latency_ms")
    snapshot.assert_within("all_reduce_ms", max_value=5.0)
    with pytest.raises(AssertionError, match="missing metric"):
        snapshot.require("tokens_per_sec")
    with pytest.raises(AssertionError, match="expected <="):
        snapshot.assert_within("kernel_latency_ms", max_value=0.1)


def test_chaos_event_validates_recovery_contract():
    event = ChaosEvent(
        kind="kill_device",
        target="gpu:5",
        expected_recovery="checkpoint_restart",
        max_recovery_s=30.0,
    )
    event.validate()

    with pytest.raises(ValueError, match="unknown chaos event"):
        ChaosEvent(kind="meteor", target="gpu:0", expected_recovery="restart").validate()
