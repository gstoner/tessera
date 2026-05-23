"""Sprint G4-LRU (2026-05-22) — bounded LRU + threading guards.

Pre-V8: `graph_ir_cache.py` was an unbounded process-local dict with
no locking — "intentional, hot loops stay 100% hit".  Real for the
test workload, a memory-growth risk for inference servers / notebooks
/ hyperparameter sweeps with many unique source strings.  Also racy
under free-threaded (no-GIL) Python 3.13+.

This file pins the new contract:
  * default capacity = 1024 (test suite stays at zero evictions)
  * LRU eviction order on overflow
  * `move_to_end` on lookup-hit keeps recently-touched entries alive
  * `set_cache_capacity(n)` shrinks immediately
  * `cache_stats()` reports `hits / misses / size / evictions / capacity`
  * threading.Lock makes concurrent lookups + stores correctness-safe
"""

from __future__ import annotations

import threading

import pytest

from tessera.compiler import graph_ir_cache as cache_mod
from tessera.compiler.graph_ir import GraphIRModule


# ─────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────


def _fresh_module(tag: str = "test") -> GraphIRModule:
    """Build a tiny GraphIRModule for cache round-trip testing.

    The cache stores deep-copies, so a fresh module per test case is
    sufficient — we don't need to make the modules byte-distinct.
    We tag via `module_attrs` so tests can verify the deepcopy round-
    trip preserves data without depending on the function-list shape.
    """
    return GraphIRModule(
        functions=[],
        module_attrs={"tessera.ir.version": '"1.0"', "test.tag": tag},
    )


@pytest.fixture(autouse=True)
def _isolated_cache():
    """Each test gets a clean cache + default capacity (1024).

    Tests that change the capacity restore the default in teardown
    via this fixture so they don't bleed into siblings.
    """
    cache_mod.clear_graph_ir_cache()
    cache_mod.set_cache_capacity(1024)
    yield
    cache_mod.clear_graph_ir_cache()
    cache_mod.set_cache_capacity(1024)


# ─────────────────────────────────────────────────────────────────────────
# Backward compatibility — existing keys + signatures
# ─────────────────────────────────────────────────────────────────────────


def test_cache_stats_returns_legacy_keys() -> None:
    """`cache_stats()` MUST keep returning the original 3 keys
    (`hits` / `misses` / `size`).  Pre-existing callers depend on
    this — the LRU upgrade is additive only."""
    stats = cache_mod.cache_stats()
    for legacy_key in ("hits", "misses", "size"):
        assert legacy_key in stats, f"legacy key missing: {legacy_key!r}"


def test_cache_stats_adds_evictions_and_capacity() -> None:
    """The LRU upgrade adds two new observability keys."""
    stats = cache_mod.cache_stats()
    assert "evictions" in stats
    assert "capacity" in stats
    assert stats["capacity"] == 1024
    assert stats["evictions"] == 0


def test_lookup_miss_returns_none_and_increments_miss_counter() -> None:
    assert cache_mod.lookup("source-a") is None
    stats = cache_mod.cache_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0
    assert stats["size"] == 0


def test_store_then_lookup_returns_deepcopy() -> None:
    """A hit returns a *deep copy* — the caller can mutate freely."""
    mod = _fresh_module("hello")
    cache_mod.store("source-a", mod)
    got = cache_mod.lookup("source-a")
    assert got is not None
    assert got is not mod, "lookup must return a deep copy, not the cached ref"
    assert got.module_attrs.get("test.tag") == "hello"


def test_clear_resets_everything() -> None:
    cache_mod.store("a", _fresh_module())
    cache_mod.lookup("a")
    cache_mod.lookup("missing")
    cache_mod.clear_graph_ir_cache()
    stats = cache_mod.cache_stats()
    assert stats == {
        "hits": 0, "misses": 0, "size": 0,
        "evictions": 0, "capacity": 1024,
    }


# ─────────────────────────────────────────────────────────────────────────
# LRU eviction order
# ─────────────────────────────────────────────────────────────────────────


def test_eviction_kicks_in_at_capacity() -> None:
    """At capacity N, the (N+1)th store evicts exactly one entry."""
    cache_mod.set_cache_capacity(3)
    for src in ("a", "b", "c"):
        cache_mod.store(src, _fresh_module(src))
    assert cache_mod.cache_stats()["size"] == 3
    assert cache_mod.cache_stats()["evictions"] == 0

    cache_mod.store("d", _fresh_module("d"))
    stats = cache_mod.cache_stats()
    assert stats["size"] == 3
    assert stats["evictions"] == 1

    # 'a' was inserted first and never re-touched → it's the LRU victim.
    assert cache_mod.lookup("a") is None
    assert cache_mod.lookup("b") is not None
    assert cache_mod.lookup("c") is not None
    assert cache_mod.lookup("d") is not None


def test_lookup_hit_bumps_to_most_recent() -> None:
    """A lookup-hit moves the entry to the most-recent slot so the
    NEXT eviction targets a different entry."""
    cache_mod.set_cache_capacity(3)
    for src in ("a", "b", "c"):
        cache_mod.store(src, _fresh_module(src))

    # Touch 'a' — it's now most-recent.  After inserting 'd', the
    # LRU victim should be 'b' (the new oldest).
    assert cache_mod.lookup("a") is not None
    cache_mod.store("d", _fresh_module("d"))

    assert cache_mod.lookup("a") is not None, "freshly-touched 'a' must survive"
    assert cache_mod.lookup("b") is None, "stale 'b' should have evicted"
    assert cache_mod.lookup("c") is not None
    assert cache_mod.lookup("d") is not None


def test_store_overwrite_does_not_evict() -> None:
    """Overwriting an existing key doesn't change cache size, so no
    eviction fires.  The entry is bumped to most-recent."""
    cache_mod.set_cache_capacity(3)
    for src in ("a", "b", "c"):
        cache_mod.store(src, _fresh_module(src))

    cache_mod.store("a", _fresh_module("a-new"))  # overwrite, not insert
    stats = cache_mod.cache_stats()
    assert stats["size"] == 3
    assert stats["evictions"] == 0

    # Verify the overwrite stuck.
    got = cache_mod.lookup("a")
    assert got is not None
    assert got.module_attrs.get("test.tag") == "a-new"


# ─────────────────────────────────────────────────────────────────────────
# set_cache_capacity
# ─────────────────────────────────────────────────────────────────────────


def test_set_cache_capacity_shrinks_immediately() -> None:
    """Calling `set_cache_capacity(n)` with n smaller than the current
    size evicts the LRU tail down to exactly `n` entries."""
    for i, src in enumerate("abcdef"):
        cache_mod.store(src, _fresh_module(src))
    assert cache_mod.cache_stats()["size"] == 6

    cache_mod.set_cache_capacity(2)
    stats = cache_mod.cache_stats()
    assert stats["size"] == 2
    assert stats["capacity"] == 2
    assert stats["evictions"] == 4

    # The 2 most-recently-stored entries should survive: 'e' and 'f'.
    assert cache_mod.lookup("a") is None
    assert cache_mod.lookup("d") is None
    assert cache_mod.lookup("e") is not None
    assert cache_mod.lookup("f") is not None


def test_set_cache_capacity_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="n > 0"):
        cache_mod.set_cache_capacity(0)
    with pytest.raises(ValueError, match="n > 0"):
        cache_mod.set_cache_capacity(-1)


def test_set_cache_capacity_growth_does_not_evict() -> None:
    """Growing the capacity beyond the current size is a no-op."""
    for src in "abc":
        cache_mod.store(src, _fresh_module(src))
    cache_mod.set_cache_capacity(100)
    stats = cache_mod.cache_stats()
    assert stats["size"] == 3
    assert stats["capacity"] == 100
    assert stats["evictions"] == 0


# ─────────────────────────────────────────────────────────────────────────
# Empty-source guard (pre-LRU contract preserved)
# ─────────────────────────────────────────────────────────────────────────


def test_empty_source_skips_cache() -> None:
    """`lookup` / `store` with empty source_text return without
    touching the cache — this is the documented no-key behavior."""
    assert cache_mod.lookup("") is None
    cache_mod.store("", _fresh_module())
    stats = cache_mod.cache_stats()
    assert stats["size"] == 0
    # The lookup with empty source should NOT increment misses
    # (matches pre-LRU behavior — early return before stats update).
    assert stats["misses"] == 0


# ─────────────────────────────────────────────────────────────────────────
# Thread safety smoke test
# ─────────────────────────────────────────────────────────────────────────


def test_concurrent_lookups_and_stores_no_corruption() -> None:
    """8 threads × 200 ops each, mix of stores + lookups.  No
    correctness assertion on individual ops (they race intentionally);
    the gate is that hits + misses == total lookups and no exception
    fires."""
    cache_mod.set_cache_capacity(64)

    NUM_THREADS = 8
    OPS_PER_THREAD = 200
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            for i in range(OPS_PER_THREAD):
                src = f"thread-{tid}-key-{i % 32}"
                if i % 3 == 0:
                    cache_mod.store(src, _fresh_module(src))
                else:
                    cache_mod.lookup(src)
        except BaseException as exc:  # pragma: no cover (debug aid)
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(t,))
        for t in range(NUM_THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread-safety regression: {errors!r}"

    stats = cache_mod.cache_stats()
    # Total lookup-ops across all threads.  Stores don't increment
    # hits/misses, so this is the number of "if i % 3 != 0" iterations.
    expected_lookups = sum(
        1 for tid in range(NUM_THREADS)
        for i in range(OPS_PER_THREAD)
        if i % 3 != 0
    )
    assert stats["hits"] + stats["misses"] == expected_lookups, (
        "lock regression: hit + miss counts don't sum to total lookups; "
        "concurrent counter increments were lost"
    )
    # Cache is bounded.
    assert stats["size"] <= stats["capacity"]


# ─────────────────────────────────────────────────────────────────────────
# Test-suite working-set sanity (no evictions at default capacity)
# ─────────────────────────────────────────────────────────────────────────


def test_default_capacity_is_large_enough_for_existing_tests() -> None:
    """The default capacity of 1024 was chosen so the existing
    tests/unit/ sweep stays at zero evictions.  Lock in the default
    so a future shrink doesn't silently regress the hot-loop case
    that the cache was originally built to serve."""
    assert cache_mod._DEFAULT_CAPACITY == 1024


def test_env_var_override_at_module_load() -> None:
    """The TESSERA_GRAPH_IR_CACHE_CAPACITY env var is honored at
    module-load time.  Mocking that without re-importing is awkward,
    so this test just verifies the helper resolves a known env value
    correctly — no module reload."""
    import os

    saved = os.environ.get("TESSERA_GRAPH_IR_CACHE_CAPACITY")
    try:
        os.environ["TESSERA_GRAPH_IR_CACHE_CAPACITY"] = "256"
        assert cache_mod._capacity_from_env() == 256
        os.environ["TESSERA_GRAPH_IR_CACHE_CAPACITY"] = "0"
        assert cache_mod._capacity_from_env() == 1024  # rejects, falls back
        os.environ["TESSERA_GRAPH_IR_CACHE_CAPACITY"] = "garbage"
        assert cache_mod._capacity_from_env() == 1024  # rejects, falls back
        os.environ.pop("TESSERA_GRAPH_IR_CACHE_CAPACITY")
        assert cache_mod._capacity_from_env() == 1024
    finally:
        if saved is None:
            os.environ.pop("TESSERA_GRAPH_IR_CACHE_CAPACITY", None)
        else:
            os.environ["TESSERA_GRAPH_IR_CACHE_CAPACITY"] = saved
