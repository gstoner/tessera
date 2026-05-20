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

* **Process-local + unbounded.**  The dict is a module-level
  ``dict`` — no LRU, no TTL.  Tests that don't want the cache call
  :func:`clear_graph_ir_cache` to flush.  Hot loops in long-running
  processes that worry about memory growth can call it
  periodically; for the common test-suite case the cache is fine
  as-is.
* **Deep copy on hit.**  The cached module's lists are mutable,
  and downstream passes (e.g., :func:`propagate_numeric_policy`)
  mutate ops in place.  Without a copy, a second consumer would
  see the first consumer's mutations.
* **Keyed on text + lowering inputs.**  No function-identity
  caching (``id(fn)``) because re-decorating the same source
  string with a fresh function object should still hit the cache.
* **Stats accessor.**  :func:`cache_stats` returns ``(hits, misses,
  size)`` for the compile-time perf gate in
  ``tests/unit/test_static_analysis_baseline.py``.
"""

from __future__ import annotations

import copy
import hashlib

from .graph_ir import GraphIRModule


# Module-level cache.  Keyed by SHA-256 of the canonical inputs.
_GRAPH_IR_CACHE: dict[str, GraphIRModule] = {}

# Cache stats — useful for tests + future telemetry.
_STATS: dict[str, int] = {"hits": 0, "misses": 0}


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

    Increments the hit/miss counters as a side effect."""

    key = _cache_key(
        source_text, effect_tag=effect_tag,
        target_attr=target_attr, lane=lane,
    )
    if key is None:
        return None
    cached = _GRAPH_IR_CACHE.get(key)
    if cached is None:
        _STATS["misses"] += 1
        return None
    _STATS["hits"] += 1
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
    is empty (matches :func:`lookup`'s behavior)."""

    key = _cache_key(
        source_text, effect_tag=effect_tag,
        target_attr=target_attr, lane=lane,
    )
    if key is None:
        return
    _GRAPH_IR_CACHE[key] = copy.deepcopy(module)


def clear_graph_ir_cache() -> None:
    """Flush the cache + stats.  Tests that touch the cache should
    call this in setup to avoid bleed-through from earlier tests."""

    _GRAPH_IR_CACHE.clear()
    _STATS["hits"] = 0
    _STATS["misses"] = 0


def cache_stats() -> dict[str, int]:
    """Return ``{"hits": int, "misses": int, "size": int}``."""

    return {
        "hits": _STATS["hits"],
        "misses": _STATS["misses"],
        "size": len(_GRAPH_IR_CACHE),
    }


__all__ = [
    "cache_stats",
    "clear_graph_ir_cache",
    "lookup",
    "store",
]
