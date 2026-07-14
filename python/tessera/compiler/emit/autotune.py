"""Workstream D2 — measured autotune loop for the D1 arbiter.

D1 (:mod:`emit.candidate`) selects by **tier priority** (crown-jewel first —
lead-safe by construction). D2 replaces that with **real on-device latency**: for
a given ``(device, target, op, shape-bucket, dtype, timing)`` it times each
F4-passing
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

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tessera.compiler.emit.candidate import (
    Candidate,
    OP_ATTENTION,
    OP_FUSED_REGION,
    OP_MATMUL,
    _note_arbiter_dispatch,
    arbitrate,
    candidates_for,
)
from tessera.compiler.emit.kernel_emitter import SpecPolicy, bucket_key


@dataclass(frozen=True)
class MeasureRecord:
    """The measured verdict for one
    ``(device, target, op, bucket, dtype, timing)`` key:
    the fastest candidate, its median latency (ms), and every timed candidate's
    latency (for a fallback log / the fleet corpus)."""

    winner: str
    latency_ms: float
    candidates: dict[str, float] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {"winner": self.winner, "latency_ms": self.latency_ms,
                "candidates": dict(self.candidates)}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "MeasureRecord":
        return cls(winner=str(d["winner"]),
                   latency_ms=float(d["latency_ms"]),
                   candidates={str(k): float(v)
                               for k, v in dict(d.get("candidates", {})).items()})


#: Corpus JSON schema version — bump if the record/key shape changes so a stale
#: committed corpus is skipped rather than mis-read.
CORPUS_VERSION = 2


TIMING_END_TO_END = "end_to_end"
TIMING_DEVICE = "device"
_TIMING_MODES = (TIMING_END_TO_END, TIMING_DEVICE)


def _normalize_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(key) == 5:
        return (*key, TIMING_END_TO_END)
    if len(key) != 6 or key[5] not in _TIMING_MODES:
        raise ValueError(f"invalid autotune cache key {key!r}")
    return key


def _key_to_json(key: tuple[Any, ...]) -> dict[str, Any]:
    """A ``(device, target, op, bucket, dtype)`` cache key → JSON dict. ``bucket``
    is a tuple of strings (from :func:`bucket_key`) or ``None``; kept as a list so
    the record is human-diffable in the committed corpus."""
    dev, target, op, bucket, dtype, timing = _normalize_key(key)
    return {"device": dev, "target": target, "op": op,
            "bucket": list(bucket) if bucket is not None else None,
            "dtype": dtype, "timing": timing}


def _key_from_json(d: dict[str, Any]) -> tuple[Any, ...]:
    b = d.get("bucket")
    return (d["device"], d["target"], d["op"],
            tuple(b) if b is not None else None, d["dtype"],
            d.get("timing", TIMING_END_TO_END))


class MeasureCache:
    """Content-keyed cache of :class:`MeasureRecord` — measure-at-first-miss. Key =
    ``(device, target, op, shape-bucket, dtype, timing)`` so nearby shapes share a verdict
    (the bucket) while distinct devices/dtypes stay separate."""

    def __init__(self) -> None:
        self._store: dict[tuple[Any, ...], MeasureRecord] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[Any, ...]) -> MeasureRecord | None:
        rec = self._store.get(_normalize_key(key))
        if rec is not None:
            self.hits += 1
        else:
            self.misses += 1
        return rec

    def put(self, key: tuple[Any, ...], rec: MeasureRecord) -> None:
        self._store[_normalize_key(key)] = rec

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    def to_dict(self) -> dict[str, Any]:
        """A fully JSON-serializable view of the cache — the fleet-shared corpus
        (Theory §7.5): ``{"version", "records": [{**key, **record}, …]}``. Each
        record carries its own ``(device, target, op, bucket, dtype)`` key so the
        corpus is self-describing and human-diffable. Round-trips through
        :meth:`load_dict`."""
        return {
            "version": CORPUS_VERSION,
            "records": [{**_key_to_json(k), **rec.as_json()}
                        for k, rec in self._store.items()],
        }

    def load_dict(self, payload: dict[str, Any], *, overwrite: bool = False) -> int:
        """Merge a :meth:`to_dict` payload into the cache (warm-start). Returns the
        number of records loaded. A record whose key is already present is kept
        (measure-on-this-box wins) unless ``overwrite``. A version mismatch loads
        nothing (a stale corpus is skipped, not mis-read)."""
        # v1 lacked the additive ``timing`` key; its rows are unambiguously the
        # historical end-to-end metric and migrate as such.
        if int(payload.get("version", -1)) not in (1, CORPUS_VERSION):
            return 0
        loaded = 0
        for r in payload.get("records", ()):
            key = _key_from_json(r)
            if not overwrite and key in self._store:
                continue
            self._store[key] = MeasureRecord.from_json(r)
            loaded += 1
        return loaded


#: Process-wide default cache (the arbiter/runtime share one).
_DEFAULT_CACHE = MeasureCache()


def default_cache() -> MeasureCache:
    return _DEFAULT_CACHE


# --- the committed fleet-shared corpus (Theory §7.5) -------------------------
#
# The measured verdicts persist to a committed JSON file so a config proven on one
# box warm-starts the others and survives across runs (extends Decision #11's
# SQLite warm-start to the §7.3 sync contract). Because every key carries its
# device tag (``rocm:gfx1151`` / ``nvidia:sm_120``), a record only ever warm-starts
# a *matching* device — a gfx1151 verdict is inert on a CDNA/NVIDIA box.

#: Default committed corpus path (alongside the E2 ``*_hot_paths.json`` ratchets).
_CORPUS_PATH = (Path(__file__).resolve().parents[4]
                / "benchmarks/baselines/autotune_corpus.json")


def corpus_path() -> Path:
    """The corpus file location (``$TESSERA_AUTOTUNE_CORPUS`` overrides the committed
    default) — the seam the §7.3 fleet-sync contract commits back."""
    env = os.environ.get("TESSERA_AUTOTUNE_CORPUS")
    return Path(env) if env else _CORPUS_PATH


def save_corpus(path: Path | str | None = None,
                cache: MeasureCache | None = None) -> Path:
    """Write ``cache`` (default: the process cache) to ``path`` (default:
    :func:`corpus_path`) as the committed fleet corpus. Returns the path written."""
    cache = cache if cache is not None else _DEFAULT_CACHE
    p = Path(path) if path is not None else corpus_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache.to_dict(), indent=2, sort_keys=True) + "\n")
    return p


def load_corpus(path: Path | str | None = None,
                cache: MeasureCache | None = None, *,
                overwrite: bool = False) -> int:
    """Merge the committed corpus at ``path`` into ``cache`` (warm-start). Returns
    the count loaded; a missing/unreadable/version-mismatched corpus loads nothing
    (never raises) so a fresh checkout or stale file degrades to measure-on-miss."""
    cache = cache if cache is not None else _DEFAULT_CACHE
    p = Path(path) if path is not None else corpus_path()
    try:
        payload = json.loads(p.read_text())
    except (OSError, ValueError):
        return 0
    return cache.load_dict(payload, overwrite=overwrite)


_warm_started = False


def _maybe_warm_start(cache: MeasureCache) -> None:
    """Warm-start the *default* process cache from the committed corpus, once. Any
    explicit cache the caller passes is left untouched (they own its lifecycle)."""
    global _warm_started
    if _warm_started or cache is not _DEFAULT_CACHE:
        return
    _warm_started = True
    if os.environ.get("TESSERA_AUTOTUNE_NO_WARMSTART"):
        return
    load_corpus(cache=cache)


def measure_latency(run_fn: Any, *, reps: int = 20, warmup: int = 3) -> float:
    """Median **wall-clock** latency (ms) of ``run_fn`` over ``reps`` calls after
    ``warmup`` untimed calls, via ``time.perf_counter``.

    This times the candidate **end-to-end** (H2D / launch / D2H), so the arbiter
    compares what a caller actually pays — the right metric for candidate
    selection. It is deliberately *not* device-event kernel-only timing: that
    would be lower-noise for isolating GPU kernel cost but would hide transfer /
    launch overhead that differs across tiers (a fused kernel that avoids a
    round-trip should win on the metric the caller feels). A device-event timer
    (CUDA events / HIP events / Metal command-buffer timestamps) is the follow-on
    for kernel-only A/B microbenchmarks; it drops in behind this same callback
    seam (``arbitrate(measure=…)``) without an API change. Keep ``reps`` high
    enough that the median is stable under wall-clock jitter."""
    for _ in range(warmup):
        run_fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


#: target → the ``runtime`` probe returning its live device tag (``"sm_120"`` /
#: ``"gfx1151"``). A probe returns ``None`` off its silicon; then we fall back to
#: the bare target id, so nothing measured on a device is ever keyed as another.
_DEVICE_PROBES: dict[str, str] = {
    "nvidia": "_nvidia_device_name",
    "rocm": "_rocm_device_name",
}


def _device_id(target: str) -> str:
    """A stable per-device tag for the cache key. Probes the live device name where
    cheap (NVIDIA ``sm_<cc>`` / ROCm ``gfx<arch>``), else falls back to the target
    id — so a config measured on one device is never reused on another."""
    probe = _DEVICE_PROBES.get(target)
    if probe is not None:
        try:
            from tessera import runtime as rt
            name = getattr(rt, probe)()
            if name:
                return f"{target}:{name}"
        except Exception:
            pass
    return target


def _infer_dims(op: str, inputs: tuple[Any, ...]) -> tuple[int, ...] | None:
    """Infer the canonical workload dimensions used by the committed corpus.

    This keeps ordinary ``run_arbitrated`` calls evidence-backed without making
    every caller restate dimensions already present in its array operands.
    Unknown/custom op kinds stay shape-anonymous and retain tier selection.
    """
    try:
        if op in (OP_MATMUL, OP_FUSED_REGION) and len(inputs) >= 2:
            a, b = inputs[0], inputs[1]
            if len(a.shape) == 2 and len(b.shape) == 2:
                return (int(a.shape[0]), int(b.shape[1]), int(a.shape[1]))
        if op == OP_ATTENTION and len(inputs) >= 3:
            q, k, v = inputs[0], inputs[1], inputs[2]
            if len(q.shape) == len(k.shape) == len(v.shape) == 2:
                return (int(q.shape[0]), int(k.shape[0]),
                        int(q.shape[1]), int(v.shape[1]))
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return None


def corpus_winner(region: Any, op: str, target: str, *inputs: Any,
                  dims: tuple[int, ...] | None = None,
                  dtype: str | None = None,
                  cache: MeasureCache | None = None,
                  device: str | None = None,
                  timing: str = TIMING_END_TO_END) -> str | None:
    """Return the applicable winner persisted for this workload, if unambiguous.

    The recommendation is only a selection hint. ``run_arbitrated`` still runs
    the normal availability/applicability and F4 gates before execution. A stale
    row, missing candidate, ambiguous dtype, or unmatched device/bucket returns
    ``None`` and therefore falls back to lead-safe tier priority.
    """
    cache = cache if cache is not None else _DEFAULT_CACHE
    _maybe_warm_start(cache)
    dims = dims if dims is not None else _infer_dims(op, inputs)
    if dims is None:
        return None
    dev = device or _device_id(target)
    bucket = bucket_key(dims, SpecPolicy.BUCKET)
    if dtype is None:
        value = getattr(region, "dtype", None)
        dtype = str(value) if value else None
    if timing not in _TIMING_MODES:
        raise ValueError(f"unknown autotune timing mode {timing!r}")
    matches = [rec for (key_dev, key_target, key_op, key_bucket, key_dtype,
                        key_timing), rec
               in cache._store.items()
               if key_dev == dev and key_target == target and key_op == op
               and key_bucket == bucket and key_timing == timing
               and (dtype is None or key_dtype == dtype)]
    winners = {rec.winner for rec in matches}
    if len(winners) != 1:
        return None
    winner = next(iter(winners))
    for candidate in candidates_for(target, op):
        if (candidate.name == winner and candidate.applies_to(region)
                and candidate.available()):
            return winner
    return None


def measured_arbitrate(region: Any, op: str, target: str, *inputs: Any,
                       dims: tuple[int, ...] | None = None, dtype: str = "f32",
                       cache: MeasureCache | None = None, reps: int = 20,
                       warmup: int = 3, device: str | None = None,
                       timing: str = TIMING_END_TO_END) -> Candidate | None:
    """Pick the winning candidate by **measured latency** (measure-at-first-miss),
    or ``None`` if none apply/verify (caller uses the reference).

    On a cache hit for ``(device, target, op, bucket(dims), dtype)`` the recorded
    winner is returned if it is still applicable/available (no re-timing). On a
    miss, the arbiter F4-gates the candidates and times the survivors on ``inputs``
    (median of ``reps`` after ``warmup``); the fastest is cached and returned."""
    cache = cache if cache is not None else _DEFAULT_CACHE
    _maybe_warm_start(cache)
    dev = device or _device_id(target)
    bucket = bucket_key(dims, SpecPolicy.BUCKET) if dims is not None else None
    if timing not in _TIMING_MODES:
        raise ValueError(f"unknown autotune timing mode {timing!r}")
    key = (dev, target, op, bucket, dtype, timing)

    rec = cache.get(key)
    if rec is not None:
        for c in candidates_for(target, op):
            if c.name == rec.winner and c.applies_to(region) and c.available():
                return c
        # cached winner is gone/unavailable — fall through and re-measure.

    latencies: dict[str, float] = {}

    def _measure(cand: Candidate) -> float:
        if timing == TIMING_DEVICE:
            measured = cand.measure_device_latency(
                region, *inputs, reps=reps, warmup=warmup)
            if measured is None:
                return float("inf")
            t = float(measured)
        else:
            t = measure_latency(
                lambda: cand.run(region, *inputs), reps=reps, warmup=warmup)
        latencies[cand.name] = t
        return t

    winner = arbitrate(region, op, target, verify=True, measure=_measure)
    if winner is not None and winner.name in latencies:
        cache.put(key, MeasureRecord(
            winner=winner.name,
            latency_ms=latencies.get(winner.name, float("nan")),
            candidates=dict(latencies)))
        return winner
    return None


def run_measured_arbitrated(region: Any, op: str, target: str, *inputs: Any,
                            dims: tuple[int, ...] | None = None, dtype: str = "f32",
                            cache: MeasureCache | None = None, reps: int = 20,
                            warmup: int = 3,
                            timing: str = TIMING_END_TO_END) -> tuple[Any, str]:
    """:func:`measured_arbitrate` then execute the winner on ``inputs`` →
    ``(output, tag)``. Falls back to ``region.reference(*inputs)`` tagged
    ``"reference"`` when no candidate wins (Decision #21: honest)."""
    winner = measured_arbitrate(region, op, target, *inputs, dims=dims, dtype=dtype,
                                cache=cache, reps=reps, warmup=warmup,
                                timing=timing)
    if winner is None:
        _note_arbiter_dispatch(target, op, None, "reference")
        return region.reference(*inputs), "reference"
    out, tag = winner.run(region, *inputs)
    _note_arbiter_dispatch(target, op, winner.name, tag)
    return out, tag
