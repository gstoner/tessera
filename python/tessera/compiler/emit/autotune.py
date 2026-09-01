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
from typing import Any, Mapping, Sequence

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
    evidence: dict[str, Any] = field(default_factory=dict)
    #: Replayable schedule identity for every measured candidate. This is
    #: evidence, never a selector input: a missing descriptor must not be
    #: reconstructed from a winner name after the fact.
    candidate_descriptors: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Applicable candidates that were NOT raced, name -> reason.
    #:
    #: ``None`` means the record does not declare its field at all (written
    #: before this existed). ``{}`` means it declares that nothing was skipped.
    #: The distinction is the point: a winner is only meaningful against the
    #: set it actually beat, and "no entry" cannot be read as "raced everyone".
    #:
    #: This exists because the committed corpus proved it does. Every
    #: device-timed row was missing exactly the candidates that had no
    #: ``measure_device_latency``: matmul raced 2 of 4 (both GEMM lanes
    #: absent), attention 5 of 6, fused_region 6 of 10. ``_measure`` scored an
    #: unmeasurable candidate ``float("inf")``, so it lost silently, and the
    #: record stored a `winner` with nothing to say a field had been reduced.
    #: The verdicts read as "the compiled kernel is faster" when they meant
    #: "the compiled kernel was the only one that could be timed".
    unmeasured: dict[str, str] | None = None
    #: Whether the race separated its two fastest candidates, and the numbers
    #: behind that call — ``{"separated", "margin", "noise", "runner_up",
    #: "factor"}``. It describes the *measurement*, not the dispatch choice: on
    #: an unseparated race the arbiter may keep an incumbent that was not the
    #: fastest sample, precisely because "fastest sample" is not a fact there.
    #:
    #: ``None`` means the record does not state it (written before this
    #: existed, or fewer than two candidates were timed, in which case there
    #: was no margin to defend). As with ``unmeasured``, absence must not be
    #: read as the favourable answer: a record with no separation is a record
    #: that never asked, not one that passed.
    #:
    #: This exists because the arbiter published noise as a verdict. On sm_120
    #: at 256³ the two NVIDIA matmul lanes measured 0.01300 ms (sd 14.5%) and
    #: 0.01057 ms (sd 39.1%); the 18.7% gap was recorded as a clean 1.63× win.
    #: A record that stores only medians cannot tell that apart from the 1.66×
    #: at 2048³, where the spreads are 2.2% and 0.6%.
    separation: dict[str, Any] | None = None

    def declares_its_field(self) -> bool:
        """Whether this record states which applicable candidates it skipped."""
        return self.unmeasured is not None

    def is_separated(self) -> bool | None:
        """``True``/``False`` if this record judged its margin, else ``None``.

        Callers that publish a comparison ("X is 1.6× faster than Y") must
        treat ``False`` and ``None`` alike: neither establishes a ranking."""
        if self.separation is None:
            return None
        return bool(self.separation.get("separated", False))

    def as_json(self) -> dict[str, Any]:
        return {"winner": self.winner, "latency_ms": self.latency_ms,
                "candidates": dict(self.candidates),
                **({"candidate_descriptors": dict(self.candidate_descriptors)}
                   if self.candidate_descriptors else {}),
                **({} if self.unmeasured is None
                   else {"unmeasured": dict(self.unmeasured)}),
                **({} if self.separation is None
                   else {"separation": dict(self.separation)}),
                **({"evidence": dict(self.evidence)} if self.evidence else {})}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "MeasureRecord":
        raw = d.get("unmeasured")
        return cls(winner=str(d["winner"]),
                   latency_ms=float(d["latency_ms"]),
                   candidates={str(k): float(v)
                               for k, v in dict(d.get("candidates", {})).items()},
                   candidate_descriptors={
                       str(k): dict(v)
                       for k, v in dict(d.get("candidate_descriptors", {})).items()
                       if isinstance(v, dict)
                   },
                   unmeasured=(None if raw is None
                               else {str(k): str(v) for k, v in dict(raw).items()}),
                   separation=(None if d.get("separation") is None
                               else dict(d["separation"])),
                   evidence=dict(d.get("evidence", {})))


#: Corpus JSON schema version — bump if the record/key shape changes so a stale
#: committed corpus is skipped rather than mis-read.
CORPUS_VERSION = 3


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


def _evidence_matches(
    key: tuple[Any, ...], record: MeasureRecord,
    required: Mapping[str, Any] | None,
) -> bool:
    """Whether a retained row matches an explicit selector evidence policy.

    Device and timing are key fields; compiler/resource/cache fingerprints are
    evidence fields. Missing evidence fails closed whenever a requirement is
    supplied, which prevents a legacy or stale row from silently selecting a
    production route.
    """
    if not required:
        return True
    dev, target, op, _, dtype, timing = _normalize_key(key)
    key_fields = {
        "device": dev, "target": target, "op": op,
        "dtype": dtype, "timing": timing,
    }
    for name, expected in required.items():
        actual = key_fields.get(name, record.evidence.get(name))
        if name == "resource_fingerprints":
            if actual is None or sorted(actual) != sorted(expected):
                return False
        elif actual != expected:
            return False
    return True


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

    def load_dict(self, payload: dict[str, Any], *, overwrite: bool = False,
                  required_evidence: Mapping[str, Any] | None = None) -> int:
        """Merge a :meth:`to_dict` payload into the cache (warm-start). Returns the
        number of records loaded. A record whose key is already present is kept
        (measure-on-this-box wins) unless ``overwrite``. A version mismatch loads
        nothing (a stale corpus is skipped, not mis-read)."""
        # v1 lacked the additive ``timing`` key; its rows are unambiguously the
        # historical end-to-end metric and migrate as such.
        if int(payload.get("version", -1)) not in (1, 2, CORPUS_VERSION):
            return 0
        loaded = 0
        for r in payload.get("records", ()):
            key = _key_from_json(r)
            record = MeasureRecord.from_json(r)
            if not _evidence_matches(key, record, required_evidence):
                continue
            if not overwrite and key in self._store:
                continue
            self._store[key] = record
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
                overwrite: bool = False,
                required_evidence: Mapping[str, Any] | None = None) -> int:
    """Merge the committed corpus at ``path`` into ``cache`` (warm-start). Returns
    the count loaded; a missing/unreadable/version-mismatched corpus loads nothing
    (never raises) so a fresh checkout or stale file degrades to measure-on-miss."""
    cache = cache if cache is not None else _DEFAULT_CACHE
    p = Path(path) if path is not None else corpus_path()
    try:
        payload = json.loads(p.read_text())
    except (OSError, ValueError):
        return 0
    return cache.load_dict(payload, overwrite=overwrite,
                           required_evidence=required_evidence)


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
    return statistics.median(measure_latency_samples(
        run_fn, reps=reps, warmup=warmup))


def measure_latency_samples(run_fn: Any, *, reps: int = 20,
                            warmup: int = 3) -> list[float]:
    """The individual samples behind :func:`measure_latency`.

    Split out because the median alone cannot say whether a verdict is
    *separated*: a 19% gap is decisive against a 2% spread and meaningless
    against a 39% one, and the record used to keep only the medians. See
    :func:`relative_spread`."""
    for _ in range(warmup):
        run_fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def relative_spread(samples: Sequence[float]) -> float:
    """Population sd over median, as a fraction. ``0.0`` for a single sample.

    Relative rather than absolute because the arbiter compares candidates
    across four orders of magnitude of latency, and the question is always
    "is this gap bigger than the noise", never "is it bigger than N ms"."""
    usable = [s for s in samples if s == s and s > 0.0]
    if len(usable) < 2:
        return 0.0
    med = statistics.median(usable)
    return (statistics.pstdev(usable) / med) if med > 0.0 else 0.0


#: How many times the winner's margin must exceed the noise floor before the
#: arbiter calls a verdict separated. Two is the same bar used to re-derive the
#: sm_120 matmul ranking; it is a judgement call, not a derived constant, and it
#: is named here so a future change to it is visible rather than inlined.
SEPARATION_FACTOR = 2.0


def separation_verdict(latencies: Mapping[str, float],
                       spreads: Mapping[str, float],
                       winner: str) -> dict[str, Any] | None:
    """Whether ``winner``'s margin over the runner-up exceeds measurement noise.

    Returns ``None`` when the question does not arise (fewer than two timed
    candidates) -- a sole candidate is selected by applicability, not by a race,
    so there is no margin to defend.

    **This exists because the arbiter published a verdict that was noise.** On
    sm_120 at 256x256x256 the NVIDIA matmul lanes measured 0.01300 ms (delegate,
    sd 14.5%) against 0.01057 ms (emitted PTX, sd 39.1%). The 18.7% gap was
    recorded as a clean 1.63x win; against a 39.1% per-lane spread it is not a
    result at all. That shape sits at the launch-overhead floor, where run-to-run
    variation swamps the difference between kernels -- and the arbiter had no way
    to say so, because a `MeasureRecord` stored medians and nothing else.

    A tie is not an error and does not block dispatch: something still has to
    run. What it blocks is *claiming* one candidate is faster."""
    timed = {n: t for n, t in latencies.items()
             if t == t and t not in (float("inf"), float("-inf"))}
    if winner not in timed or len(timed) < 2:
        return None
    ordered = sorted(timed.items(), key=lambda kv: kv[1])
    (best_name, best), (second_name, second) = ordered[0], ordered[1]
    margin = (second - best) / second if second > 0.0 else 0.0
    noise = max(spreads.get(best_name, 0.0), spreads.get(second_name, 0.0))
    return {
        "separated": margin > SEPARATION_FACTOR * noise,
        "margin": margin,
        "noise": noise,
        "runner_up": second_name,
        "factor": SEPARATION_FACTOR,
    }


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
                  timing: str = TIMING_END_TO_END,
                  required_evidence: Mapping[str, Any] | None = None) -> str | None:
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
               and (dtype is None or key_dtype == dtype)
               and _evidence_matches(
                   (key_dev, key_target, key_op, key_bucket, key_dtype,
                    key_timing), rec, required_evidence)]
    winners = {rec.winner for rec in matches}
    if len(winners) != 1:
        return None
    winner = next(iter(winners))

    # A verdict is only usable if it beat the field that is racing *now*.
    live = {c.name: c for c in candidates_for(target, op)
            if c.applies_to(region) and c.available()}
    for rec in matches:
        if not _record_raced_the_live_field(rec, live, timing):
            return None

    # ...and only if the verdict is SUPPORTED. Without this, `separation` was
    # an unconsumed declaration (Decision #29): recorded, documented, and read
    # by nothing, while `run_arbitrated` kept dispatching on rows the corpus
    # itself marks as noise. The sm_120 corpus holds a float16 device row whose
    # 2.16% margin sits under 148.55% noise; before this check that row still
    # changed a production route.
    #
    # `separated is False` is refused outright -- a ranking the measurement
    # says is not real must never become a dispatch hint.
    #
    # `separation is None` is ALLOWED, and the asymmetry is deliberate. None
    # means the row predates the field and was never asked, which is exactly
    # the state every row was in before #663; rejecting it would silently
    # deactivate most of the committed corpus as a side effect of adding a
    # check. A row that is *known* unsupported is strictly worse than one that
    # is merely unproven, and only the first is a regression to allow.
    # Re-racing is what moves a None row to a real verdict.
    #
    # `evidence.selector_eligible` is honoured the same way: the finalizer sets
    # it False when two independent runs disagreed on the winner, which is the
    # same conclusion reached by a different mechanism.
    for rec in matches:
        if rec.is_separated() is False:
            return None
        if rec.evidence.get("selector_eligible") is False:
            return None
        # An UNPROVEN RANKING is refused too, now that every fleet row has had
        # the chance to earn a verdict (gfx1151 re-raced 2026-09-01, sm_120
        # 2026-08-31). `None` used to be allowed because rejecting it would
        # have deactivated most of the corpus before those runs existed; that
        # reason has expired.
        #
        # But `None` is NOT uniformly "unproven". `separation_verdict` returns
        # None when fewer than two candidates were timed, because a sole
        # candidate is chosen by applicability rather than by a race and has no
        # margin to defend. Refusing those would be a category error, not
        # caution -- 12 of the 23 remaining None rows are exactly that shape.
        # So the test is "ranks two or more and cannot say it separated them".
        if rec.is_separated() is None and _ranked_candidate_count(rec) >= 2:
            return None

    candidate = live.get(winner)
    return winner if candidate is not None else None


def _ranked_candidate_count(rec: MeasureRecord) -> int:
    """How many candidates this record actually timed.

    `inf` is not a latency -- it is `_measure`'s marker for "could not be timed
    in this mode" -- so it must not be counted as a competitor. Separating
    these two is what lets `corpus_winner` refuse an unproven *ranking* while
    still trusting a single-candidate row, which has no ranking to prove.
    """
    return sum(1 for value in rec.candidates.values()
               if value == value and value not in (float("inf"), float("-inf")))


def _record_raced_the_live_field(
    rec: MeasureRecord, live: Mapping[str, Any], timing: str,
) -> bool:
    """Whether ``rec``'s winner beat the candidates available here and now.

    Fails closed twice, because a partial race is indistinguishable from a
    complete one by the winner alone:

    * **Undeclared field, device timing.** A record written before
      `unmeasured` existed cannot show what it skipped, and for device timing
      we know that generation was systematically incomplete: every committed
      device-timed row was missing exactly the candidates that had no
      `measure_device_latency` (matmul raced 2 of 4, attention 5 of 6,
      fused_region 6 of 10). Serving those verdicts would select a compiled
      kernel over a hand-tuned one that was never in the race.
    * **A live candidate is missing from the timed field.** Every candidate
      racing *now* must appear in `rec.candidates` -- the set that was actually
      timed.

    The second test is a subset check rather than a scan of `unmeasured`, and
    the difference is load-bearing (review finding on PR #655). Checking only
    the declared skips treats a candidate absent from *both* maps as having
    been raced, and there are two ordinary ways to be absent from both: the
    candidate was registered after the measurement, or it was applicable then
    but failed F4 verification, so `arbitrate` filtered it out before
    `_measure` ever saw it. In both cases the recorded winner never beat it,
    yet the verdict would still be served. Requiring `live` to be a subset of
    the timed set covers the declared skips too, since a skipped candidate is
    by construction not in `rec.candidates`.

    Returning False falls back to lead-safe tier priority, never to silence.
    """
    if timing == TIMING_DEVICE and not rec.declares_its_field():
        return False
    return all(name in rec.candidates for name in live)


def measured_arbitrate(region: Any, op: str, target: str, *inputs: Any,
                       dims: tuple[int, ...] | None = None, dtype: str = "f32",
                       cache: MeasureCache | None = None, reps: int = 20,
                       warmup: int = 3, device: str | None = None,
                       timing: str = TIMING_END_TO_END,
                       device_repeats: int = 3) -> Candidate | None:
    """Pick the winning candidate by **measured latency** (measure-at-first-miss),
    or ``None`` if none apply/verify (caller uses the reference).

    On a cache hit for ``(device, target, op, bucket(dims), dtype)`` the recorded
    winner is returned if it is still applicable/available (no re-timing). On a
    miss, the arbiter F4-gates the candidates and times the survivors on ``inputs``
    (median of ``reps`` after ``warmup``); the fastest is cached and returned.

    ``device_repeats`` is how many times the whole device measurement is redone
    per candidate, and exists only to give the verdict a noise floor: a single
    ``measure_device_latency`` call returns a median with no dispersion, and a
    median alone cannot tell a real 1.66x from a tie. It does not apply to
    end-to-end timing, where the samples are already in hand."""
    cache = cache if cache is not None else _DEFAULT_CACHE
    _maybe_warm_start(cache)
    dev = device or _device_id(target)
    bucket = bucket_key(dims, SpecPolicy.BUCKET) if dims is not None else None
    if timing not in _TIMING_MODES:
        raise ValueError(f"unknown autotune timing mode {timing!r}")
    key = (dev, target, op, bucket, dtype, timing)

    live = {c.name: c for c in candidates_for(target, op)
            if c.applies_to(region) and c.available()}

    rec = cache.get(key)
    if rec is not None and _record_raced_the_live_field(rec, live, timing):
        # The exact-key hit is only usable if it beat the field racing now.
        # Validating solely that the winner is still live -- what this did
        # before -- accepted a legacy device row, or a partial one naming a
        # candidate that has since become runnable, and so preserved through
        # `measured_arbitrate`/`run_measured_arbitrated` exactly the biased
        # selection `corpus_winner` refuses (review finding on PR #655).
        winner_candidate = live.get(rec.winner)
        if winner_candidate is not None:
            return winner_candidate
        # cached winner is gone/unavailable — fall through and re-measure.

    latencies: dict[str, float] = {}
    unmeasured: dict[str, str] = {}
    spreads: dict[str, float] = {}

    def _measure(cand: Candidate) -> float:
        if timing == TIMING_DEVICE:
            # Repeat the whole device measurement so the verdict has a noise
            # floor to be judged against. One call returns a median with no
            # dispersion, and a median alone cannot distinguish a real 1.66x
            # from the 256^3 tie that was recorded as a 1.63x win.
            device_samples = [
                cand.measure_device_latency(
                    region, *inputs, reps=reps, warmup=warmup)
                for _ in range(device_repeats)
            ]
            usable = [float(s) for s in device_samples if s is not None]
            spreads[cand.name] = relative_spread(usable)
            measured = statistics.median(usable) if usable else None
            if measured is None:
                # NOT a latency. "Cannot be timed in this mode" is a statement
                # about instrumentation; `inf` is a statement about speed. The
                # committed corpus shows what conflating them costs: every
                # device-timed row silently dropped the candidates without a
                # device timer, and the survivors were recorded as winners.
                unmeasured[cand.name] = (
                    f"no measure_device_latency for timing={TIMING_DEVICE!r}")
                return float("inf")
            t = float(measured)
        else:
            samples = measure_latency_samples(
                lambda: cand.run(region, *inputs), reps=reps, warmup=warmup)
            spreads[cand.name] = relative_spread(samples)
            t = statistics.median(samples)
        latencies[cand.name] = t
        return t

    winner = arbitrate(region, op, target, verify=True, measure=_measure)
    if winner is None or winner.name not in latencies:
        # Either nothing applied/verified, or the "winner" is a candidate that
        # was never timed -- which happens when every candidate scored `inf`
        # and `min` fell back to registration order. Caching that would store
        # an arbitrary pick as a measured verdict.
        return None

    separation = separation_verdict(latencies, spreads, winner.name)
    if (separation is not None and not separation["separated"]
            and rec is not None and rec.winner in live
            and rec.winner in latencies):
        # A tie must not thrash the selection. When this race cannot tell the
        # candidates apart and a previous run already picked one of them, keep
        # that one: re-picking by noise would flip the cached winner between
        # runs, invalidating downstream keys for no measured reason. The record
        # still says the verdict was unseparated, so nothing reads it as a win.
        winner = live[rec.winner]

    cache.put(key, MeasureRecord(
        winner=winner.name,
        latency_ms=latencies.get(winner.name, float("nan")),
        candidates=dict(latencies),
        unmeasured=dict(unmeasured),
        separation=separation))
    return winner


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
