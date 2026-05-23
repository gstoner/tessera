"""G4 (2026-05-19) — process-local memoization of Graph IR builds.

The `@tessera.jit` decorator rebuilds Graph IR every time it fires.
For library code that decorates the same function repeatedly (test
suites, notebooks, hot-reload setups), that's pure overhead — the
source hasn't changed.

This module memoizes the AST → Graph IR step keyed on
``(source_text, effect_tag, target_attr, lane)``.  A cache hit
returns a **deep copy** of the cached module so the caller can
mutate freely without poisoning the cache.

Design
------

* **Process-local + bounded LRU.**  Backed by an
  ``collections.OrderedDict`` with a default capacity of 1024
  entries (tuned so the full ``tests/unit/`` sweep stays at zero
  evictions).  Capacity is configurable via the
  ``TESSERA_GRAPH_IR_CACHE_CAPACITY`` env var at module-load time,
  or by calling :func:`set_cache_capacity` at runtime.  Tests that
  don't want the cache call :func:`clear_graph_ir_cache` to flush.
  2026-05-22: pre-LRU this dict was unbounded — fine for the test
  suite but a real memory-growth risk for inference servers,
  long-running notebooks, hyperparameter sweeps that hit thousands
  of unique source strings.  LRU caps growth without changing the
  100%-hit-rate of the documented hot-loop workload.
* **Deep copy on hit + store.**  The cached module's lists are
  mutable, and downstream passes (e.g.,
  :func:`propagate_numeric_policy`) mutate ops in place.  Without
  a copy, a second consumer would see the first consumer's
  mutations.  Both the store-side deepcopy and the lookup-side
  deepcopy run *outside* the lock so concurrent JITs don't
  serialize on CPU-bound copying.
* **Thread safety via :data:`_LOCK`.**  All mutations of
  :data:`_GRAPH_IR_CACHE` and :data:`_STATS` happen under a
  module-level :class:`threading.Lock`.  This is correctness under
  the free-threaded (no-GIL) Python interpreter and harmless under
  CPython's GIL.
* **Keyed on text + lowering inputs.**  No function-identity
  caching (``id(fn)``) because re-decorating the same source
  string with a fresh function object should still hit the cache.
* **Stats accessor.**  :func:`cache_stats` returns ``hits``,
  ``misses``, ``size``, ``evictions``, and ``capacity`` — useful
  for the compile-time perf gate in
  ``tests/unit/test_static_analysis_baseline.py`` and for runtime
  telemetry that watches for thrashing.
"""

from __future__ import annotations

import copy
import hashlib
import os
import threading
from collections import OrderedDict

from .graph_ir import GraphIRModule


# Default cache capacity. Sized so the existing tests/unit/ sweep
# stays at zero evictions while still bounding long-running processes.
# Override at process start via TESSERA_GRAPH_IR_CACHE_CAPACITY=N, or
# at runtime via :func:`set_cache_capacity`.
_DEFAULT_CAPACITY = 1024


def _capacity_from_env() -> int:
    """Read TESSERA_GRAPH_IR_CACHE_CAPACITY from the environment with
    a defensive fallback. Non-positive or non-integer values fall back
    to :data:`_DEFAULT_CAPACITY` rather than corrupting startup."""
    raw = os.environ.get("TESSERA_GRAPH_IR_CACHE_CAPACITY")
    if raw is None:
        return _DEFAULT_CAPACITY
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_CAPACITY
    if n <= 0:
        return _DEFAULT_CAPACITY
    return n


# Mutable module-level state.  Mutated under _LOCK.
_LOCK = threading.Lock()
_GRAPH_IR_CACHE: "OrderedDict[str, GraphIRModule]" = OrderedDict()
_STATS: dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}
_CAPACITY: int = _capacity_from_env()


def _cache_key(
    source_text: str | None,
    *,
    effect_tag: str | None = None,
    target_attr: str | None = None,
    lane: str = "tessera_jit",
) -> str | None:
    """Build a stable cache key from the inputs that influence the
    Graph IR shape.

    Returns ``None`` when ``source_text`` is empty — without source
    there's nothing reliable to hash on, so memoization is skipped
    rather than producing false-positive hits.
    """

    if not source_text:
        return None
    hasher = hashlib.sha256()
    hasher.update(source_text.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update((effect_tag or "").encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update((target_attr or "").encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(lane.encode("utf-8"))
    return hasher.hexdigest()


def lookup(
    source_text: str | None,
    *,
    effect_tag: str | None = None,
    target_attr: str | None = None,
    lane: str = "tessera_jit",
) -> GraphIRModule | None:
    """Probe the cache.  Returns a fresh deep copy of the cached
    module on hit, ``None`` on miss.

    Increments the hit/miss counters as a side effect.  On hit the
    entry is bumped to the most-recently-used position so LRU
    eviction targets stale entries first.
    """

    key = _cache_key(
        source_text, effect_tag=effect_tag,
        target_attr=target_attr, lane=lane,
    )
    if key is None:
        return None
    with _LOCK:
        cached = _GRAPH_IR_CACHE.get(key)
        if cached is None:
            _STATS["misses"] += 1
            return None
        # LRU bump under the lock so concurrent readers see a
        # consistent recency view.
        _GRAPH_IR_CACHE.move_to_end(key)
        _STATS["hits"] += 1
    # Deepcopy runs *outside* the lock — it's pure CPU and the
    # cached module is treated as immutable from the cache's POV
    # (the next consumer gets their own copy).
    return copy.deepcopy(cached)


def store(
    source_text: str | None,
    module: GraphIRModule,
    *,
    effect_tag: str | None = None,
    target_attr: str | None = None,
    lane: str = "tessera_jit",
) -> None:
    """Stash a fresh module in the cache.  No-op when ``source_text``
    is empty (matches :func:`lookup`'s behavior).

    Evicts the least-recently-used entry when the cache is at or
    above capacity.  The store-side deepcopy runs outside the lock
    so concurrent JIT calls don't serialize on CPU-bound copying.
    """

    key = _cache_key(
        source_text, effect_tag=effect_tag,
        target_attr=target_attr, lane=lane,
    )
    if key is None:
        return
    snapshot = copy.deepcopy(module)
    with _LOCK:
        _GRAPH_IR_CACHE[key] = snapshot
        _GRAPH_IR_CACHE.move_to_end(key)  # fresh entry = most-recent
        while len(_GRAPH_IR_CACHE) > _CAPACITY:
            _GRAPH_IR_CACHE.popitem(last=False)  # evict LRU
            _STATS["evictions"] += 1


def clear_graph_ir_cache() -> None:
    """Flush the cache + stats.  Tests that touch the cache should
    call this in setup to avoid bleed-through from earlier tests."""

    with _LOCK:
        _GRAPH_IR_CACHE.clear()
        _STATS["hits"] = 0
        _STATS["misses"] = 0
        _STATS["evictions"] = 0


def set_cache_capacity(n: int) -> None:
    """Reconfigure the cache capacity at runtime.

    Immediately evicts down to ``n`` entries if the cache is
    currently larger.  Eviction proceeds in LRU order so the
    most-recently-touched entries survive.  ``n`` must be positive;
    callers that want to disable the cache should call
    :func:`clear_graph_ir_cache` and stop calling :func:`store`
    (or use the env var with a tiny value at process start).

    Raises ``ValueError`` for non-positive ``n``.
    """

    if n <= 0:
        raise ValueError(
            f"set_cache_capacity requires n > 0; got {n}.  Use "
            "clear_graph_ir_cache() to flush the cache instead."
        )
    global _CAPACITY
    with _LOCK:
        _CAPACITY = n
        while len(_GRAPH_IR_CACHE) > _CAPACITY:
            _GRAPH_IR_CACHE.popitem(last=False)
            _STATS["evictions"] += 1


def cache_stats() -> dict[str, int]:
    """Return cache observability metrics.

    Keys (2026-05-22 LRU update):
      ``hits``       — successful lookups since process start / last clear
      ``misses``     — cold-cache lookups
      ``size``       — current entry count
      ``evictions``  — LRU evictions since process start / last clear
      ``capacity``   — current capacity ceiling
    """

    with _LOCK:
        return {
            "hits": _STATS["hits"],
            "misses": _STATS["misses"],
            "size": len(_GRAPH_IR_CACHE),
            "evictions": _STATS["evictions"],
            "capacity": _CAPACITY,
        }


__all__ = [
    "cache_stats",
    "clear_graph_ir_cache",
    "lookup",
    "set_cache_capacity",
    "store",
]
