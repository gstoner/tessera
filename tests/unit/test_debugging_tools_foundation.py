from __future__ import annotations

import io
import pathlib

import numpy as np
import pytest

import tessera as ts
from tessera import debug


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_trace_graph_accepts_op_descriptors_and_exports_graphviz():
    ops = [
        {"op": "tensor", "output": "%0"},
        {"op": "matmul", "inputs": ["%0", "%0.T"], "output": "%1"},
        {"op": "softmax", "inputs": ["%1"], "output": "%2"},
    ]

    trace = debug.trace_graph(ops)

    text = trace.format()
    assert "%1 = matmul(%0, %0.T)" in text
    dot = trace.to_graphviz()
    assert "softmax" in dot
    assert "n0 -> n1" in dot
    assert "n1 -> n2" in dot
    payload = trace.to_dict()
    assert payload["schema"] == "tessera.debug.graph_trace.v1"
    assert payload["ops"][1]["op"] == "matmul"


def test_graph_namespace_aliases_debug_trace_helpers():
    assert ts.graph.trace is debug.trace_graph
    assert ts.graph.debug_trace is debug.debug_trace
    assert ts.graph.debug_value is debug.debug_value
    assert ts.graph.export_graphviz is debug.export_graphviz
    assert ts.graph.replay_capture is debug.replay_capture


def test_debug_trace_records_tensor_summaries_to_stream():
    stream = io.StringIO()

    with debug.debug_trace(samples=2, stream=stream) as trace:
        value = debug.trace_value("%x", np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert value.tolist() == [1.0, 2.0, 3.0]
    assert len(trace.records) == 1
    assert trace.records[0].mean == pytest.approx(2.0)
    assert "Tensor %x" in stream.getvalue()
    assert "samples=[1.0, 2.0]" in stream.getvalue()
    payload = trace.to_dict()
    assert payload["schema"] == "tessera.debug.trace.v1"
    assert payload["records"][0]["shape"] == [3]
    assert '"records"' in trace.to_json()


def test_debug_value_records_active_trace_and_returns_value():
    with debug.debug_trace(samples=1, metadata={"graph_hash": "abc"}) as trace:
        value = debug.debug_value("%scores", np.array([4.0, 5.0]))

    assert value.tolist() == [4.0, 5.0]
    assert trace.to_dict()["metadata"]["graph_hash"] == "abc"
    assert trace.records[0].name == "%scores"


def test_replay_manifest_accepts_runtime_artifact():
    artifact = ts.RuntimeArtifact(
        graph_ir="module { func.func @main() }",
        schedule_ir='"schedule.artifact"() : () -> ()',
        metadata={"target": "cpu", "graph_hash": "g"},
    )

    manifest = debug.replay_manifest(artifact, seed=123)

    assert manifest["schema"] == "tessera.debug.replay_manifest.v1"
    assert manifest["metadata"]["seed"] == 123
    assert manifest["artifact"]["metadata"]["target"] == "cpu"
    assert "graph" in manifest["artifact"]["ir_hashes"]


def test_debug_artifact_and_barrier_descriptors_are_structured():
    artifact = debug.debug_artifact("sched", metadata={"schedule_hash": "s"})
    barrier = debug.debug_barrier("q0", queue_id=0, scope="warpgroup")

    assert artifact["schema"] == "tessera.schedule.debug_artifact.v1"
    assert artifact["metadata"]["schedule_hash"] == "s"
    assert barrier["schema"] == "tessera.tile.debug_barrier.v1"
    assert barrier["queue_id"] == 0


def test_summarize_tensor_detects_non_finite_values():
    summary = debug.summarize_tensor(np.array([1.0, np.nan]))

    assert summary.finite is False
    assert summary.shape == (2,)


def test_check_grad_passes_for_quadratic_loss():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def loss(v):
        return np.sum(v * v)

    result = debug.check_grad(loss, [x], analytic_grads=[2.0 * x], eps=1e-5, atol=1e-5)

    assert result.passed
    assert "passed" in result.format()


def test_check_grad_fails_for_wrong_analytic_gradient():
    x = np.array([1.0, 2.0], dtype=np.float64)

    result = debug.check_grad(
        lambda v: np.sum(v * v),
        [x],
        analytic_grads=[np.zeros_like(x)],
        eps=1e-5,
        atol=1e-6,
    )

    assert not result.passed
    assert result.max_error > 0.0


def test_check_determinism_wraps_qa_assertion():
    result = debug.check_determinism(lambda: np.array([1, 2, 3]), runs=3)

    assert result.passed
    assert result.bitwise
    assert "All 3 runs" in result.format()


def test_debugging_tools_guide_is_registered_and_covers_core_layers():
    guide = (ROOT / "docs/guides/Tessera_Debugging_Tools_Guide.md").read_text()
    readme = (ROOT / "docs/README.md").read_text()

    for needle in [
        "Graph Inspection",
        "Numerical Tracing",
        "Gradient Checking",
        "Determinism Checks",
        "External Debugger Integration",
    ]:
        assert needle in guide
    assert "Tessera_Debugging_Tools_Guide.md" in readme
