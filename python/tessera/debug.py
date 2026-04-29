"""Layered debugging helpers for Tessera programs.

The helpers in this module are intentionally runtime-light. They provide a
stable Python surface for graph inspection, numerical tracing, finite-difference
gradient checks, and determinism checks while the lower compiler layers grow
their MLIR-native debug hooks.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .testing.qa import assert_deterministic


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "numpy") and callable(value.numpy):
        return np.asarray(value.numpy())
    if hasattr(value, "_data"):
        return np.asarray(value._data)
    return np.asarray(value)


@dataclass(frozen=True)
class TensorSummary:
    """Compact numerical summary for a tensor-like value."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    mean: float
    std: float
    min: float
    max: float
    finite: bool
    samples: tuple[float, ...] = ()

    def format(self) -> str:
        finite = "finite" if self.finite else "non-finite"
        sample_text = f", samples={list(self.samples)}" if self.samples else ""
        return (
            f"Tensor {self.name}: shape={self.shape}, dtype={self.dtype}, "
            f"mean={self.mean:.6g}, std={self.std:.6g}, min={self.min:.6g}, "
            f"max={self.max:.6g}, {finite}{sample_text}"
        )


def summarize_tensor(value, *, name: str = "%tensor", samples: int = 0) -> TensorSummary:
    """Return mean/std/min/max and optional sample values for a tensor-like value."""

    arr = _as_numpy(value)
    flat = arr.reshape(-1) if arr.size else arr
    finite = bool(np.isfinite(arr).all()) if arr.size else True
    sample_values = tuple(float(x) for x in flat[:samples]) if samples > 0 else ()
    safe = arr.astype(np.float64, copy=False) if arr.size else np.asarray([0.0])
    return TensorSummary(
        name=name,
        shape=tuple(int(d) for d in arr.shape),
        dtype=str(arr.dtype),
        mean=float(np.mean(safe)),
        std=float(np.std(safe)),
        min=float(np.min(safe)),
        max=float(np.max(safe)),
        finite=finite,
        samples=sample_values,
    )


@dataclass
class DebugTrace:
    """Context manager that records numerical summaries of intermediate values."""

    samples: int = 0
    stream: Optional[io.StringIO] = None
    records: list[TensorSummary] = field(default_factory=list)

    def __enter__(self) -> "DebugTrace":
        _TRACE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if _TRACE_STACK and _TRACE_STACK[-1] is self:
            _TRACE_STACK.pop()

    def record(self, name: str, value) -> TensorSummary:
        summary = summarize_tensor(value, name=name, samples=self.samples)
        self.records.append(summary)
        if self.stream is not None:
            self.stream.write(summary.format() + "\n")
        return summary

    def format(self) -> str:
        return "\n".join(record.format() for record in self.records)


_TRACE_STACK: list[DebugTrace] = []


def debug_trace(*, samples: int = 0, stream: Optional[io.StringIO] = None) -> DebugTrace:
    """Create a numerical debug trace context."""

    return DebugTrace(samples=samples, stream=stream)


def trace_value(name: str, value):
    """Record a value in the active debug trace, if any, and return it unchanged."""

    if _TRACE_STACK:
        _TRACE_STACK[-1].record(name, value)
    return value


@dataclass(frozen=True)
class GraphTrace:
    """Printable/exportable graph inspection result."""

    lines: tuple[str, ...]
    ir_level: str = "graph"

    def format(self) -> str:
        return "\n".join(self.lines)

    def print(self, file=None) -> str:
        text = self.format()
        print(text, file=file)
        return text

    def to_mlir(self) -> str:
        return self.format()

    def to_graphviz(self) -> str:
        body = ["digraph tessera_debug {"]
        previous = None
        for idx, line in enumerate(self.lines):
            node = f"n{idx}"
            label = line.replace("\\", "\\\\").replace('"', '\\"')
            body.append(f'  {node} [label="{label}"];')
            if previous is not None:
                body.append(f"  {previous} -> {node};")
            previous = node
        body.append("}")
        return "\n".join(body)


def trace_graph(value, *, ir_level: str = "graph") -> GraphTrace:
    """Inspect a graph-like value and return a printable trace.

    Accepted inputs:
    - a ``@tessera.jit`` wrapper with ``.graph_ir.to_mlir()``
    - an object with ``to_mlir()``
    - MLIR/text strings
    - a list of op descriptor dictionaries
    """

    if hasattr(value, "graph_ir") and hasattr(value.graph_ir, "to_mlir"):
        text = value.graph_ir.to_mlir()
        return GraphTrace(_nonempty_lines(text), ir_level=ir_level)
    if hasattr(value, "to_mlir") and callable(value.to_mlir):
        return GraphTrace(_nonempty_lines(value.to_mlir()), ir_level=ir_level)
    if isinstance(value, str):
        return GraphTrace(_nonempty_lines(value), ir_level=ir_level)
    if isinstance(value, Sequence):
        return GraphTrace(tuple(_format_op_descriptor(op, i) for i, op in enumerate(value)), ir_level=ir_level)
    return GraphTrace((repr(value),), ir_level=ir_level)


def export_graphviz(value, *, ir_level: str = "graph") -> str:
    """Return GraphViz DOT for a graph-like value."""

    return trace_graph(value, ir_level=ir_level).to_graphviz()


@dataclass(frozen=True)
class GradientCheckResult:
    """Finite-difference gradient check result."""

    passed: bool
    max_error: float
    errors: Mapping[int, float]
    eps: float
    atol: float
    rtol: float

    def format(self) -> str:
        status = "passed" if self.passed else "failed"
        return f"Gradient check {status} (max error {self.max_error:.6g})"


def check_grad(
    fn: Callable[..., float],
    inputs: Sequence[object],
    *,
    analytic_grads: Optional[Sequence[object]] = None,
    eps: float = 1e-4,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> GradientCheckResult:
    """Validate gradients of a scalar function with central differences.

    If ``analytic_grads`` is omitted, the helper returns the finite-difference
    gradient magnitudes as errors. Supplying analytic gradients turns the helper
    into a pass/fail validator.
    """

    arrays = [np.asarray(_as_numpy(x), dtype=np.float64).copy() for x in inputs]
    analytic = None if analytic_grads is None else [
        np.asarray(_as_numpy(g), dtype=np.float64) for g in analytic_grads
    ]
    if analytic is not None and len(analytic) != len(arrays):
        raise ValueError("analytic_grads must match inputs length")

    per_input_errors: dict[int, float] = {}
    for input_index, arr in enumerate(arrays):
        numeric = np.zeros_like(arr, dtype=np.float64)
        for index in np.ndindex(arr.shape):
            plus = [x.copy() for x in arrays]
            minus = [x.copy() for x in arrays]
            plus[input_index][index] += eps
            minus[input_index][index] -= eps
            numeric[index] = (_scalar(fn(*plus)) - _scalar(fn(*minus))) / (2.0 * eps)

        if analytic is None:
            per_input_errors[input_index] = float(np.max(np.abs(numeric))) if numeric.size else 0.0
            continue
        if analytic[input_index].shape != numeric.shape:
            raise ValueError(
                f"analytic gradient {input_index} shape {analytic[input_index].shape} "
                f"does not match input shape {numeric.shape}"
            )
        allowed = atol + rtol * np.abs(analytic[input_index])
        diff = np.abs(numeric - analytic[input_index])
        per_input_errors[input_index] = float(np.max(diff - allowed)) if diff.size else 0.0

    max_error = max(per_input_errors.values(), default=0.0)
    return GradientCheckResult(
        passed=max_error <= 0.0,
        max_error=max_error,
        errors=per_input_errors,
        eps=eps,
        atol=atol,
        rtol=rtol,
    )


@dataclass(frozen=True)
class DeterminismCheckResult:
    """Result of running a bitwise/tolerance determinism check."""

    runs: int
    bitwise: bool
    passed: bool = True

    def format(self) -> str:
        mode = "bitwise" if self.bitwise else "within tolerance"
        return f"All {self.runs} runs produced identical results ({mode})"


def check_determinism(
    fn: Callable[[], object],
    *,
    runs: int = 5,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> DeterminismCheckResult:
    """Verify repeated executions produce the same result."""

    assert_deterministic(fn, runs=runs, rtol=rtol, atol=atol)
    return DeterminismCheckResult(runs=runs, bitwise=(rtol == 0.0 and atol == 0.0))


def _scalar(value) -> float:
    arr = _as_numpy(value)
    if arr.shape != ():
        arr = np.asarray(arr).reshape(-1)
        if arr.size != 1:
            raise ValueError("check_grad expects fn to return a scalar")
        return float(arr[0])
    return float(arr)


def _nonempty_lines(text: str) -> tuple[str, ...]:
    return tuple(line.rstrip() for line in str(text).splitlines() if line.strip())


def _format_op_descriptor(op, index: int) -> str:
    if isinstance(op, Mapping):
        name = op.get("name") or op.get("output") or f"%{index}"
        kind = op.get("op", "unknown")
        inputs = ", ".join(str(i) for i in op.get("inputs", ()))
        return f"{name} = {kind}({inputs})"
    return str(op)


__all__ = [
    "DebugTrace",
    "DeterminismCheckResult",
    "GradientCheckResult",
    "GraphTrace",
    "TensorSummary",
    "check_determinism",
    "check_grad",
    "debug_trace",
    "export_graphviz",
    "summarize_tensor",
    "trace_graph",
    "trace_value",
]
