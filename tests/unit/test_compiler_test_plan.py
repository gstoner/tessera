from __future__ import annotations

from pathlib import Path

import numpy as np

import tessera as ts


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "tests" / "COMPILER_TEST_PLAN.md"
INTEGRATED_PLAN = ROOT / "docs" / "audit" / "compiler" / "INTEGRATED_COMPILER_PLAN.md"


def test_compiler_test_plan_documents_active_suite_layout():
    text = PLAN.read_text(encoding="utf-8")

    assert "tests/unit/" in text
    assert "tests/performance/" in text
    assert "tests/tessera-ir/" in text
    assert "Unit Test Matrix" in text
    assert "Performance Test Matrix" in text


def test_compiler_test_plan_names_current_architecture_layers():
    text = PLAN.read_text(encoding="utf-8")

    for layer in ("Python @jit frontend", "Graph IR", "Schedule IR", "Tile IR", "Target IR"):
        assert layer in text


def test_compiler_test_plan_distinguishes_presence_package_and_lineage():
    text = PLAN.read_text(encoding="utf-8")

    for contract in (
        "stage_complete",
        "package_launchable",
        "lineage_complete",
        "next.input_digest == previous.output_digest",
        "producer",
        "contract version",
    ):
        assert contract in text

    assert "GPU artifacts must return structured non-success statuses" not in text


def test_compiler_test_plan_tracks_every_e2e_real_work_item():
    test_plan = PLAN.read_text(encoding="utf-8")
    integrated = INTEGRATED_PLAN.read_text(encoding="utf-8")

    for index in range(7):
        work_item = f"E2E-REAL-{index}"
        assert work_item in integrated
        assert work_item in test_plan


def test_first_vertical_slice_names_local_dtype_and_hardware_contracts():
    text = PLAN.read_text(encoding="utf-8")

    for contract in (
        "x86-f32",
        "ROCm-f16/f32",
        "Zen 5 AVX-512",
        "gfx1151",
        "(33, 17, 31)",
        "(2048, 2048, 2048)",
        "8.02 TFLOP/s",
    ):
        assert contract in text


def test_change_routing_keeps_exact_device_evidence_architecture_owned():
    text = PLAN.read_text(encoding="utf-8")

    assert "Change-to-Gate Routing" in text
    assert "sibling evidence never transfers" in text
    assert "missing hardware is an honest access result" in text


def test_unit_plan_covers_core_compiler_modules():
    text = PLAN.read_text(encoding="utf-8")

    expected_rows = [
        "Frontend source recovery",
        "Constraint extraction",
        "Effect inference",
        "Graph IR emission",
        "CPU compiler path",
        "Target profiles",
        "Distributed planning",
        "Reliability/runtime contracts",
    ]
    for row in expected_rows:
        assert row in text


def test_compiler_test_plan_documents_project_level_evals():
    text = PLAN.read_text(encoding="utf-8")

    expected_eval_topics = [
        "Project-Level Eval Matrix",
        "Spec conformance",
        "End-to-end compiler evals",
        "Documentation evals",
        "Sample/tutorial evals",
        "Agent context graph evals",
        "context graph output freshness",
    ]
    for topic in expected_eval_topics:
        assert topic in text


def test_transformer_proxy_emits_all_python_ir_layers_without_implying_lineage():
    """Stage presence remains useful, but is not compiler-boundary E2E."""
    @ts.jit
    def block(x, wq, wk, wv, wo):
        q = ts.ops.matmul(x, wq)
        k = ts.ops.matmul(x, wk)
        v = ts.ops.matmul(x, wv)
        scores = ts.ops.matmul(q, ts.ops.transpose(k))
        probs = ts.ops.softmax(scores)
        ctx = ts.ops.matmul(probs, v)
        return ts.ops.matmul(ctx, wo)

    x = np.ones((2, 3), dtype=np.float32)
    w = np.ones((3, 3), dtype=np.float32)
    out = block(x, w, w, w, np.eye(3, dtype=np.float32))

    assert out.shape == (2, 3)
    assert block.is_executable
    artifacts = {artifact.level: artifact.text for artifact in block.lowering_artifacts()}
    assert set(artifacts) == {"graph", "schedule", "tile", "target"}
    assert "tessera.matmul" in artifacts["graph"]
    assert "schedule.tile" in artifacts["schedule"]
    assert "stable_reduction" in artifacts["tile"]
    assert 'source = "tessera.matmul"' in artifacts["target"]
