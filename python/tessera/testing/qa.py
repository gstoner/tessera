"""
tessera.testing.qa — shared QA and reliability assertions.

These helpers implement the small, repeatable checks described in
docs/guides/Tessera_QA_Reliability_Guide.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional

import numpy as np


def _as_numpy(value):
    """Convert common tensor-like values to a NumPy array."""
    if hasattr(value, "numpy") and callable(value.numpy):
        return np.asarray(value.numpy())
    return np.asarray(value)


def assert_close_to_reference(
    actual,
    reference,
    *,
    rtol: float,
    atol: float,
    name: str = "value",
) -> None:
    """Assert that an output matches a golden reference within tolerance."""
    actual_np = _as_numpy(actual)
    reference_np = _as_numpy(reference)
    if actual_np.shape != reference_np.shape:
        raise AssertionError(
            f"{name} shape mismatch: actual {actual_np.shape}, "
            f"reference {reference_np.shape}"
        )
    np.testing.assert_allclose(actual_np, reference_np, rtol=rtol, atol=atol)


def assert_finite(value, *, name: str = "value") -> None:
    """Assert that a tensor-like value contains no NaN or Inf values."""
    arr = _as_numpy(value)
    bad = ~np.isfinite(arr)
    if bad.any():
        count = int(bad.sum())
        first = tuple(int(i) for i in np.argwhere(bad)[0])
        raise AssertionError(f"{name} contains {count} non-finite value(s); first at {first}")


def assert_deterministic(
    fn: Callable[[], object],
    *,
    runs: int = 2,
    rtol: float = 0.0,
    atol: float = 0.0,
    name: str = "deterministic result",
) -> None:
    """Run a zero-argument function repeatedly and compare all outputs."""
    if runs < 2:
        raise ValueError("runs must be >= 2")
    first = _as_numpy(fn()).copy()
    for i in range(1, runs):
        current = _as_numpy(fn())
        try:
            np.testing.assert_allclose(current, first, rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise AssertionError(f"{name} differed on run {i + 1}: {exc}") from exc


@dataclass(frozen=True)
class PerformanceExpectation:
    """Thresholds for lightweight performance regression checks."""

    name: str
    latency_ms_max: Optional[float] = None
    tflops_min: Optional[float] = None
    bandwidth_gbps_min: Optional[float] = None

    def validate(self, metrics: Mapping[str, float]) -> None:
        failures = []
        if self.latency_ms_max is not None:
            actual = metrics.get("latency_ms")
            if actual is None or actual > self.latency_ms_max:
                failures.append(
                    f"latency_ms expected <= {self.latency_ms_max}, got {actual}"
                )
        if self.tflops_min is not None:
            actual = metrics.get("tflops")
            if actual is None or actual < self.tflops_min:
                failures.append(f"tflops expected >= {self.tflops_min}, got {actual}")
        if self.bandwidth_gbps_min is not None:
            actual = metrics.get("bandwidth_gbps")
            if actual is None or actual < self.bandwidth_gbps_min:
                failures.append(
                    f"bandwidth_gbps expected >= {self.bandwidth_gbps_min}, got {actual}"
                )
        if failures:
            joined = "; ".join(failures)
            raise AssertionError(f"{self.name} performance expectation failed: {joined}")


@dataclass(frozen=True)
class RegressionBaseline:
    """Versioned production baseline for performance and accuracy drift."""

    name: str
    latency_ms: Optional[float] = None
    accuracy: Optional[float] = None
    tflops: Optional[float] = None
    max_latency_regression: float = 0.20
    max_accuracy_drop: float = 0.0
    max_tflops_regression: float = 0.20

    def validate(self, metrics: Mapping[str, float]) -> None:
        failures = []
        if self.latency_ms is not None:
            actual = metrics.get("latency_ms")
            limit = self.latency_ms * (1.0 + self.max_latency_regression)
            if actual is None or actual > limit:
                failures.append(f"latency_ms expected <= {limit}, got {actual}")
        if self.accuracy is not None:
            actual = metrics.get("accuracy")
            limit = self.accuracy - self.max_accuracy_drop
            if actual is None or actual < limit:
                failures.append(f"accuracy expected >= {limit}, got {actual}")
        if self.tflops is not None:
            actual = metrics.get("tflops")
            limit = self.tflops * (1.0 - self.max_tflops_regression)
            if actual is None or actual < limit:
                failures.append(f"tflops expected >= {limit}, got {actual}")
        if failures:
            raise AssertionError(f"{self.name} regression detected: {'; '.join(failures)}")


@dataclass(frozen=True)
class ReplayManifest:
    """Minimal production replay metadata for deterministic incident debugging."""

    run_id: str
    seed: int
    graph_hash: str
    schedule_hash: str
    target: str
    backend: str
    step: Optional[int] = None
    checkpoint: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        missing = [
            name for name in (
                "run_id",
                "graph_hash",
                "schedule_hash",
                "target",
                "backend",
            )
            if not getattr(self, name)
        ]
        if self.seed < 0:
            missing.append("seed>=0")
        if missing:
            raise ValueError(f"ReplayManifest missing invalid field(s): {missing}")

    def to_dict(self) -> Dict[str, object]:
        self.validate()
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "graph_hash": self.graph_hash,
            "schedule_hash": self.schedule_hash,
            "target": self.target,
            "backend": self.backend,
            "step": self.step,
            "checkpoint": self.checkpoint,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HealthSnapshot:
    """Health metrics captured from a production training/inference step."""

    metrics: Mapping[str, float]

    def require(self, *names: str) -> None:
        missing = [name for name in names if name not in self.metrics]
        if missing:
            raise AssertionError(f"HealthSnapshot missing metric(s): {missing}")

    def assert_within(self, name: str, *, min_value: Optional[float] = None,
                      max_value: Optional[float] = None) -> None:
        if name not in self.metrics:
            raise AssertionError(f"HealthSnapshot missing metric: {name}")
        value = self.metrics[name]
        if min_value is not None and value < min_value:
            raise AssertionError(f"{name} expected >= {min_value}, got {value}")
        if max_value is not None and value > max_value:
            raise AssertionError(f"{name} expected <= {max_value}, got {value}")


@dataclass(frozen=True)
class ChaosEvent:
    """Description of an injected failure and expected recovery behavior."""

    kind: str
    target: str
    expected_recovery: str
    max_recovery_s: Optional[float] = None

    def validate(self) -> None:
        valid_kinds = {
            "kill_device",
            "network_latency",
            "collective_timeout",
            "ecc_error",
            "drop_rank",
            "restart_worker",
        }
        if self.kind not in valid_kinds:
            raise ValueError(f"unknown chaos event kind {self.kind!r}")
        if not self.target:
            raise ValueError("chaos event target is required")
        if not self.expected_recovery:
            raise ValueError("chaos event expected_recovery is required")
        if self.max_recovery_s is not None and self.max_recovery_s <= 0:
            raise ValueError("max_recovery_s must be positive when set")
