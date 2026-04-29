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
    assert "softmax" in trace.to_graphviz()


def test_graph_namespace_aliases_debug_trace_helpers():
    assert ts.graph.trace is debug.trace_graph
    assert ts.graph.debug_trace is debug.debug_trace
    assert ts.graph.export_graphviz is debug.export_graphviz


def test_debug_trace_records_tensor_summaries_to_stream():
    stream = io.StringIO()

    with debug.debug_trace(samples=2, stream=stream) as trace:
        value = debug.trace_value("%x", np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert value.tolist() == [1.0, 2.0, 3.0]
    assert len(trace.records) == 1
    assert trace.records[0].mean == pytest.approx(2.0)
    assert "Tensor %x" in stream.getvalue()
    assert "samples=[1.0, 2.0]" in stream.getvalue()


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
