from __future__ import annotations

from pathlib import Path

import numpy as np

import tessera as ts


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "tests" / "COMPILER_TEST_PLAN.md"


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
    ]
    for topic in expected_eval_topics:
        assert topic in text


def test_transformer_proxy_exercises_all_documented_python_ir_layers():
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
    assert block.uses_compiled_path
    artifacts = {artifact.level: artifact.text for artifact in block.lowering_artifacts()}
    assert set(artifacts) == {"graph", "schedule", "tile", "target"}
    assert "tessera.matmul" in artifacts["graph"]
    assert "tessera.schedule.tile" in artifacts["schedule"]
    assert "stable_reduction" in artifacts["tile"]
    assert "tessera.cpu.matmul" in artifacts["target"]
