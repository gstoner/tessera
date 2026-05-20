"""Tests for G4 — Graph IR memoization by source hash.

Locks three contracts:

1. ``@tessera.jit`` / ``tessera.from_text`` populate the cache on
   first decoration and hit it on subsequent identical decorations.
2. Different source / target / effect produces different cache keys
   (no false-positive hits).
3. Cache hits return *deep copies* — mutating the returned module
   doesn't poison the cache.
"""

from __future__ import annotations

import pytest

import tessera
from tessera.compiler import (
    clear_graph_ir_cache,
    graph_ir_cache_stats,
)
from tessera.compiler.graph_ir_cache import lookup, store
from tessera.compiler.graph_ir import GraphIRModule


@pytest.fixture(autouse=True)
def _flush_cache():
    """Every test starts with a clean cache."""

    clear_graph_ir_cache()
    yield
    clear_graph_ir_cache()


class TestCacheBasics:
    def test_empty_cache_returns_none(self) -> None:
        assert lookup("def f(x): return x") is None
        stats = graph_ir_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_store_then_lookup_hits(self) -> None:
        module = GraphIRModule()
        store("def f(x): return x", module)
        retrieved = lookup("def f(x): return x")
        assert retrieved is not None
        stats = graph_ir_cache_stats()
        assert stats["hits"] == 1

    def test_lookup_with_empty_source_returns_none(self) -> None:
        """No source → no key → no cache participation."""

        assert lookup(None) is None
        assert lookup("") is None

    def test_store_with_empty_source_is_noop(self) -> None:
        module = GraphIRModule()
        store(None, module)
        store("", module)
        assert graph_ir_cache_stats()["size"] == 0


class TestKeyDiscrimination:
    def test_different_source_misses(self) -> None:
        module = GraphIRModule()
        store("def f(x): return x", module)
        assert lookup("def f(x): return x") is not None
        # Different source — must miss.
        assert lookup("def g(x): return x") is None

    def test_different_target_misses(self) -> None:
        module = GraphIRModule()
        store("source", module, target_attr="target_a")
        assert lookup("source", target_attr="target_b") is None

    def test_different_effect_tag_misses(self) -> None:
        module = GraphIRModule()
        store("source", module, effect_tag="pure")
        assert lookup("source", effect_tag="random") is None

    def test_different_lane_misses(self) -> None:
        module = GraphIRModule()
        store("source", module, lane="tessera_jit")
        assert lookup("source", lane="clifford_jit") is None


class TestDeepCopySemantics:
    def test_hit_returns_independent_module(self) -> None:
        """Mutating the returned module must not poison the cache."""

        from tessera.compiler.graph_ir import GraphIRFunction

        module = GraphIRModule()
        module.functions.append(GraphIRFunction(name="f"))
        store("source", module)

        first = lookup("source")
        assert first is not None
        # Mutate the returned module.
        first.functions.append(GraphIRFunction(name="injected"))

        # The cached module must not have the injected function.
        second = lookup("source")
        assert second is not None
        names = {f.name for f in second.functions}
        assert "injected" not in names


class TestJitIntegration:
    def test_repeated_decoration_hits_cache(self) -> None:
        source = """
            def f(x, y):
                return tessera.ops.add(x, y)
        """
        # First decoration — miss + store.
        tessera.from_text(source)
        stats_after_first = graph_ir_cache_stats()
        assert stats_after_first["misses"] >= 1
        assert stats_after_first["size"] >= 1

        # Second decoration of the same source — hit.
        tessera.from_text(source)
        stats_after_second = graph_ir_cache_stats()
        assert stats_after_second["hits"] >= 1
        # Cache size shouldn't grow.
        assert stats_after_second["size"] == stats_after_first["size"]

    def test_different_function_does_not_hit(self) -> None:
        source_a = "def f(x): return tessera.ops.relu(x)"
        source_b = "def f(x): return tessera.ops.gelu(x)"
        tessera.from_text(source_a)
        tessera.from_text(source_b)
        stats = graph_ir_cache_stats()
        # Two distinct sources → two distinct cache entries.
        assert stats["size"] >= 2


class TestPublicNamespace:
    def test_helpers_exported(self) -> None:
        import tessera.compiler as tc
        assert "clear_graph_ir_cache" in tc.__all__
        assert "graph_ir_cache_stats" in tc.__all__
        assert hasattr(tc, "clear_graph_ir_cache")
        assert hasattr(tc, "graph_ir_cache_stats")
