"""Evidence-gated selection between Apple GPU execution routes.

The Apple backend has several valid execution lanes (MPSGraph, handwritten
MSL, Metal 4 cooperative tensors, and packaged MTL4 ML subgraphs).  A route is
not promoted merely because it is available: a characterization record must
show that it ran natively, matched its oracle, and beats the incumbent for the
same operation, shape, dtype, and device.

This module is deliberately runtime-free.  Benchmark drivers write its small
JSON schema; the JIT may read a selected report without importing Metal or
authoring packages during decoration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROUTE_REPORT_SCHEMA_VERSION = 1
PACKAGE_ROUTE = "package"


@dataclass(frozen=True)
class AppleRouteMeasurement:
    """One warm, numerically checked route measurement."""

    op: str
    shape: str
    dtype: str
    device: str
    route: str
    latency_ms: float
    native_dispatched: bool
    numerically_validated: bool

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "AppleRouteMeasurement | None":
        try:
            latency = float(row["latency_ms"])
            measurement = cls(
                op=str(row["op"]),
                shape=str(row["shape"]),
                dtype=str(row["dtype"]),
                device=str(row["device"]),
                route=str(row.get("route", row.get("mode", ""))),
                latency_ms=latency,
                native_dispatched=bool(row["native_dispatched"]),
                numerically_validated=bool(row["numerically_validated"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not measurement.route or measurement.latency_ms <= 0:
            return None
        return measurement


def load_route_measurements(path: str | Path) -> tuple[AppleRouteMeasurement, ...]:
    """Load only complete, current-schema measurements from ``path``.

    Old benchmark snapshots intentionally do not drive compilation: they lack
    the native-dispatch and numerical-proof fields needed to make a promotion
    decision honestly.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if payload.get("schema_version") != ROUTE_REPORT_SCHEMA_VERSION:
        return ()
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return ()
    return tuple(
        measurement
        for row in runs
        if isinstance(row, Mapping)
        for measurement in (AppleRouteMeasurement.from_mapping(row),)
        if measurement is not None
    )


def select_route(
    measurements: Iterable[AppleRouteMeasurement],
    *,
    op: str,
    shape: str,
    dtype: str,
    device: str,
    incumbent_route: str,
) -> str | None:
    """Return the fastest proven route or ``None`` when evidence is incomplete.

    Both the incumbent and winner must be native and numerically validated.
    This avoids promoting a route on a host fallback or comparing a package
    against an unrelated shape/device result.
    """
    matching = [
        row for row in measurements
        if (row.op, row.shape, row.dtype, row.device) == (op, shape, dtype, device)
        and row.native_dispatched and row.numerically_validated
    ]
    if not any(row.route == incumbent_route for row in matching):
        return None
    if not matching:
        return None
    return min(matching, key=lambda row: row.latency_ms).route


def package_route_selected(
    report_path: str | Path | None,
    *,
    op: str,
    shape: str,
    dtype: str = "f32",
    device: str = "apple_silicon_metal",
    incumbent_route: str = "live",
) -> bool:
    """Whether a report promotes a package route for this exact invocation."""
    if not report_path:
        return False
    return select_route(
        load_route_measurements(report_path), op=op, shape=shape, dtype=dtype,
        device=device, incumbent_route=incumbent_route,
    ) == PACKAGE_ROUTE


__all__ = [
    "AppleRouteMeasurement",
    "PACKAGE_ROUTE",
    "ROUTE_REPORT_SCHEMA_VERSION",
    "load_route_measurements",
    "package_route_selected",
    "select_route",
]
