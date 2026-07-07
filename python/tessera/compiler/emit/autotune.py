"""Workstream D2 — measured autotune loop for the D1 arbiter.

D1 (:mod:`emit.candidate`) selects by **tier priority** (crown-jewel first —
lead-safe by construction). D2 replaces that with **real on-device latency**: for
a given ``(device, target, op, shape-bucket, dtype)`` it times each F4-passing
candidate once, caches the fastest (**measure-at-first-miss**), and reuses that
verdict thereafter. Lead-safety is preserved end-to-end — only candidates that
already pass the universal F4 oracle *within their accuracy budget* are timed, so
a faster-but-wrong (or out-of-budget) kernel can never win.

This layers on the arbiter's existing ``measure`` seam
(:func:`emit.candidate.arbitrate` picks ``min(cands, key=measure)``): D2 supplies
the latency callback + the cache. The cache is process-local here; persisting it
as the committed *fleet-shared autotune corpus* (Theory §7.5 — a config proven on
one box warm-starts the others) is the follow-on that hangs off :meth:`MeasureCache.to_dict`.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from tessera.compiler.emit.candidate import (
    Candidate,
    _note_arbiter_dispatch,
    arbitrate,
    candidates_for,
)
from tessera.compiler.emit.kernel_emitter import SpecPolicy, bucket_key


@dataclass(frozen=True)
class MeasureRecord:
    """The measured verdict for one ``(device, target, op, bucket, dtype)`` key:
    the fastest candidate, its median latency (ms), and every timed candidate's
    latency (for a fallback log / the fleet corpus)."""

    winner: str
    latency_ms: float
    candidates: dict[str, float] = field(default_factory=dict)


class MeasureCache:
    """Content-keyed cache of :class:`MeasureRecord` — measure-at-first-miss. Key =
    ``(device, target, op, shape-bucket, dtype)`` so nearby shapes share a verdict
    (the bucket) while distinct devices/dtypes stay separate."""

    def __init__(self) -> None:
        self._store: dict[tuple[Any, ...], MeasureRecord] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[Any, ...]) -> MeasureRecord | None:
        rec = self._store.get(key)
        if rec is not None:
            self.hits += 1
        else:
            self.misses += 1
        return rec

    def put(self, key: tuple[Any, ...], rec: MeasureRecord) -> None:
        self._store[key] = rec

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    def to_dict(self) -> dict[str, MeasureRecord]:
        """A JSON-friendly view (string keys) — the seam the fleet-shared corpus
        persists. Follow-on; not wired to disk here."""
        return {repr(k): v for k, v in self._store.items()}


#: Process-wide default cache (the arbiter/runtime share one).
_DEFAULT_CACHE = MeasureCache()


def default_cache() -> MeasureCache:
    return _DEFAULT_CACHE


def measure_latency(run_fn: Any, *, reps: int = 20, warmup: int = 3) -> float:
    """Median wall-clock latency (ms) of ``run_fn`` over ``reps`` calls after
    ``warmup`` untimed calls. ``run_fn`` runs the candidate end-to-end (H2D /
    launch / D2H) so the comparison reflects what a caller actually pays."""
    for _ in range(warmup):
        run_fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _device_id(target: str) -> str:
    """A stable per-device tag for the cache key. Probes the live device name where
    cheap (NVIDIA), else falls back to the target id — so a config measured on one
    device is never reused on another."""
    if target == "nvidia":
        try:
            from tessera import runtime as rt
            name = rt._nvidia_device_name()
            if name:
                return f"nvidia:{name}"
        except Exception:
            pass
    return target


def measured_arbitrate(region: Any, op: str, target: str, *inputs: Any,
                       dims: tuple[int, ...] | None = None, dtype: str = "f32",
                       cache: MeasureCache | None = None, reps: int = 20,
                       warmup: int = 3, device: str | None = None) -> Candidate | None:
    """Pick the winning candidate by **measured latency** (measure-at-first-miss),
    or ``None`` if none apply/verify (caller uses the reference).

    On a cache hit for ``(device, target, op, bucket(dims), dtype)`` the recorded
    winner is returned if it is still applicable/available (no re-timing). On a
    miss, the arbiter F4-gates the candidates and times the survivors on ``inputs``
    (median of ``reps`` after ``warmup``); the fastest is cached and returned."""
    cache = cache if cache is not None else _DEFAULT_CACHE
    dev = device or _device_id(target)
    bucket = bucket_key(dims, SpecPolicy.BUCKET) if dims is not None else None
    key = (dev, target, op, bucket, dtype)

    rec = cache.get(key)
    if rec is not None:
        for c in candidates_for(target, op):
            if c.name == rec.winner and c.applies_to(region) and c.available():
                return c
        # cached winner is gone/unavailable — fall through and re-measure.

    latencies: dict[str, float] = {}

    def _measure(cand: Candidate) -> float:
        t = measure_latency(lambda: cand.run(region, *inputs), reps=reps, warmup=warmup)
        latencies[cand.name] = t
        return t

    winner = arbitrate(region, op, target, verify=True, measure=_measure)
    if winner is not None:
        cache.put(key, MeasureRecord(
            winner=winner.name,
            latency_ms=latencies.get(winner.name, float("nan")),
            candidates=dict(latencies)))
    return winner


def run_measured_arbitrated(region: Any, op: str, target: str, *inputs: Any,
                            dims: tuple[int, ...] | None = None, dtype: str = "f32",
                            cache: MeasureCache | None = None, reps: int = 20,
                            warmup: int = 3) -> tuple[Any, str]:
    """:func:`measured_arbitrate` then execute the winner on ``inputs`` →
    ``(output, tag)``. Falls back to ``region.reference(*inputs)`` tagged
    ``"reference"`` when no candidate wins (Decision #21: honest)."""
    winner = measured_arbitrate(region, op, target, *inputs, dims=dims, dtype=dtype,
                                cache=cache, reps=reps, warmup=warmup)
    if winner is None:
        _note_arbiter_dispatch(target, op, None, "reference")
        return region.reference(*inputs), "reference"
    out, tag = winner.run(region, *inputs)
    _note_arbiter_dispatch(target, op, winner.name, tag)
    return out, tag
