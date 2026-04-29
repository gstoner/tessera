"""Runtime profiling helpers for Tessera.

This module provides a small public profiling surface that mirrors the future
runtime profiler: sessions collect per-op metrics, reports are printable, and
timeline traces can be exported in Chrome Trace Event format.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional


@dataclass(frozen=True)
class ProfileEvent:
    """One profiled operation or region."""

    name: str
    latency_ms: float
    flops_g: float = 0.0
    bandwidth_gbps: float = 0.0
    efficiency_pct: float = 0.0
    counters: Mapping[str, float] = field(default_factory=dict)
    start_us: float = 0.0
    duration_us: float = 0.0

    def to_row(self) -> tuple[str, float, float, float, float]:
        return (
            self.name,
            self.latency_ms,
            self.flops_g,
            self.bandwidth_gbps,
            self.efficiency_pct,
        )

    def to_trace_event(self, pid: int = 0, tid: int = 0) -> dict:
        return {
            "name": self.name,
            "cat": "tessera",
            "ph": "X",
            "ts": self.start_us,
            "dur": self.duration_us or self.latency_ms * 1000.0,
            "pid": pid,
            "tid": tid,
            "args": {
                "latency_ms": self.latency_ms,
                "flops_g": self.flops_g,
                "bandwidth_gbps": self.bandwidth_gbps,
                "efficiency_pct": self.efficiency_pct,
                **dict(self.counters),
            },
        }


class ProfileSession:
    """Collects runtime performance events."""

    def __init__(self) -> None:
        self.events: list[ProfileEvent] = []
        self._t0 = 0.0

    def __enter__(self) -> "ProfileSession":
        self._t0 = time.perf_counter()
        _SESSION_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if _SESSION_STACK and _SESSION_STACK[-1] is self:
            _SESSION_STACK.pop()

    def record(
        self,
        name: str,
        *,
        latency_ms: float,
        flops: float = 0.0,
        bytes_moved: float = 0.0,
        peak_tflops: Optional[float] = None,
        counters: Optional[Mapping[str, float]] = None,
    ) -> ProfileEvent:
        if latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        seconds = latency_ms * 1e-3
        flops_g = flops / 1e9 if flops else 0.0
        bandwidth_gbps = bytes_moved / seconds / 1e9 if bytes_moved and seconds > 0 else 0.0
        achieved_tflops = flops / seconds / 1e12 if flops and seconds > 0 else 0.0
        efficiency = (
            min(100.0, achieved_tflops / peak_tflops * 100.0)
            if peak_tflops and peak_tflops > 0
            else 0.0
        )
        now_us = (time.perf_counter() - self._t0) * 1e6 if self._t0 else 0.0
        event = ProfileEvent(
            name=name,
            latency_ms=latency_ms,
            flops_g=flops_g,
            bandwidth_gbps=bandwidth_gbps,
            efficiency_pct=efficiency,
            counters=dict(counters or {}),
            start_us=now_us,
            duration_us=latency_ms * 1000.0,
        )
        self.events.append(event)
        return event

    def measure(
        self,
        name: str,
        fn: Callable[[], object],
        *,
        flops: float = 0.0,
        bytes_moved: float = 0.0,
        peak_tflops: Optional[float] = None,
        counters: Optional[Mapping[str, float]] = None,
    ):
        start = time.perf_counter()
        value = fn()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        self.record(
            name,
            latency_ms=elapsed_ms,
            flops=flops,
            bytes_moved=bytes_moved,
            peak_tflops=peak_tflops,
            counters=counters,
        )
        return value

    def report(self) -> str:
        headers = ("Op", "Latency(ms)", "FLOPs(G)", "Bandwidth(GB/s)", "Efficiency(%)")
        rows = [headers] + [
            (
                e.name,
                f"{e.latency_ms:.3f}",
                f"{e.flops_g:.3f}",
                f"{e.bandwidth_gbps:.3f}",
                f"{e.efficiency_pct:.1f}",
            )
            for e in self.events
        ]
        widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
        lines = [
            "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            for row in rows
        ]
        return "\n".join(lines)

    def timeline_events(self) -> list[dict]:
        return [event.to_trace_event(tid=i) for i, event in enumerate(self.events)]

    def timeline(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"traceEvents": self.timeline_events()}, indent=2))
        return out


_SESSION_STACK: list[ProfileSession] = []


def session() -> ProfileSession:
    return ProfileSession()


def current_session() -> Optional[ProfileSession]:
    return _SESSION_STACK[-1] if _SESSION_STACK else None


def record(name: str, **kwargs) -> ProfileEvent:
    active = current_session()
    if active is None:
        raise RuntimeError("profiler.record() requires an active profiler.session()")
    return active.record(name, **kwargs)


def measure(name: str, fn: Callable[[], object], **kwargs):
    active = current_session()
    if active is None:
        with session() as active:
            return active.measure(name, fn, **kwargs)
    return active.measure(name, fn, **kwargs)


def timeline(path: str | Path, sess: Optional[ProfileSession] = None) -> Path:
    target = sess or current_session()
    if target is None:
        raise RuntimeError("profiler.timeline() requires a session")
    return target.timeline(path)


__all__ = [
    "ProfileEvent",
    "ProfileSession",
    "current_session",
    "measure",
    "record",
    "session",
    "timeline",
]
