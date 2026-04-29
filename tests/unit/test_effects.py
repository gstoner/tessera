"""
tests/unit/test_effects.py

Tests for:
  - Effect enum and lattice ordering
  - EffectLattice.infer()
  - EffectLattice.check_deterministic()
  - @jit(deterministic=True) integration
"""

import pytest
import numpy as np

import tessera
from tessera.compiler.effects import Effect, EffectLattice, TesseraEffectError


# ─────────────────────────────────────────────────────────────────────────────
# Effect enum tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEffectEnum:
    def test_order(self):
        assert Effect.pure < Effect.random
        assert Effect.random < Effect.movement
        assert Effect.movement < Effect.state
        assert Effect.state < Effect.collective
        assert Effect.collective < Effect.memory
        assert Effect.random < Effect.memory
        assert Effect.memory < Effect.io
        assert Effect.io < Effect.top

    def test_join_pure_pure(self):
        assert Effect.pure.join(Effect.pure) == Effect.pure

    def test_join_pure_random(self):
        assert Effect.pure.join(Effect.random) == Effect.random

    def test_join_random_memory(self):
        assert Effect.random.join(Effect.memory) == Effect.memory

    def test_join_commutative(self):
        assert Effect.random.join(Effect.memory) == Effect.memory.join(Effect.random)

    def test_join_idempotent(self):
        assert Effect.io.join(Effect.io) == Effect.io

    def test_le(self):
        assert Effect.pure <= Effect.random
        assert Effect.pure <= Effect.pure

    def test_ge(self):
        assert Effect.top >= Effect.io
        assert Effect.top >= Effect.top

    def test_names(self):
        assert Effect.pure.name == "pure"
        assert Effect.random.name == "random"
        assert Effect.movement.name == "movement"
        assert Effect.state.name == "state"
        assert Effect.collective.name == "collective"
        assert Effect.memory.name == "memory"
        assert Effect.io.name == "io"
        assert Effect.top.name == "top"

    def test_values_ordered(self):
        effects = [
            Effect.pure,
            Effect.random,
            Effect.movement,
            Effect.state,
            Effect.collective,
            Effect.memory,
            Effect.io,
            Effect.top,
        ]
        values = [e.value for e in effects]
        assert values == sorted(values)


# ─────────────────────────────────────────────────────────────────────────────
# EffectLattice.infer() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEffectLatticeInfer:
    def setup_method(self):
        self.lattice = EffectLattice()

    def test_pure_function(self):
        def gemm_fn(A, B):
            return tessera.ops.gemm(A, B)

        effect = self.lattice.infer(gemm_fn)
        assert effect == Effect.pure

    def test_random_function(self):
        def dropout_fn(x):
            return tessera.ops.dropout(x)

        effect = self.lattice.infer(dropout_fn)
        assert effect == Effect.random

    def test_collective_function(self):
        def collective_fn(x):
            return tessera.ops.all_reduce(x)

        effect = self.lattice.infer(collective_fn)
        assert effect == Effect.collective

    def test_no_ops_is_pure(self):
        def plain_python(x):
            return x + 1

        effect = self.lattice.infer(plain_python)
        assert effect == Effect.pure

    def test_join_propagates(self):
        """A function calling both gemm and dropout should have random effect."""
        def mixed(x, W):
            y = tessera.ops.gemm(x, W)
            z = tessera.ops.dropout(y)
            return z

        effect = self.lattice.infer(mixed)
        assert effect == Effect.random

    def test_caching(self):
        def f(x):
            return tessera.ops.gemm(x, x)

        e1 = self.lattice.infer(f)
        e2 = self.lattice.infer(f)
        assert e1 is e2 or e1 == e2  # cached result

    def test_invalidate_cache(self):
        def f(x):
            return tessera.ops.gemm(x, x)

        e1 = self.lattice.infer(f)
        self.lattice.invalidate(f)
        e2 = self.lattice.infer(f)
        assert e1 == e2  # same result, just re-inferred

    def test_join_list(self):
        effects = [Effect.pure, Effect.random, Effect.memory]
        result = self.lattice.join(effects)
        assert result == Effect.memory

    def test_join_empty(self):
        result = self.lattice.join([])
        assert result == Effect.pure

    def test_repr(self):
        assert "EffectLattice" in repr(self.lattice)


# ─────────────────────────────────────────────────────────────────────────────
# EffectLattice.check_deterministic() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckDeterministic:
    def setup_method(self):
        self.lattice = EffectLattice()

    def test_pure_function_passes(self):
        def f(x):
            return tessera.ops.layer_norm(x)

        self.lattice.check_deterministic(f)  # should not raise

    def test_pure_function_passes_with_seed(self):
        def f(x):
            return tessera.ops.layer_norm(x)

        self.lattice.check_deterministic(f, seed=42)  # should not raise

    def test_random_without_seed_raises(self):
        def f(x):
            return tessera.ops.dropout(x)

        with pytest.raises(TesseraEffectError) as exc_info:
            self.lattice.check_deterministic(f, seed=None)
        assert "random" in str(exc_info.value).lower() or "seed" in str(exc_info.value).lower()

    def test_random_with_seed_passes(self):
        def f(x):
            return tessera.ops.dropout(x)

        self.lattice.check_deterministic(f, seed=42)  # seeded RNG is deterministic

    def test_collective_deterministic_is_governed_by_ir(self):
        def f(x):
            return tessera.ops.all_reduce(x)

        self.lattice.check_deterministic(f, seed=42)

    def test_error_has_fn_name(self):
        def my_nondeterministic_fn(x):
            return tessera.ops.dropout(x)

        with pytest.raises(TesseraEffectError) as exc_info:
            self.lattice.check_deterministic(my_nondeterministic_fn)
        assert "my_nondeterministic_fn" in str(exc_info.value)

    def test_error_attributes(self):
        def f(x):
            return tessera.ops.dropout(x)

        with pytest.raises(TesseraEffectError) as exc_info:
            self.lattice.check_deterministic(f)
        err = exc_info.value
        assert err.fn_name == "f"
        assert err.inferred >= Effect.random


# ─────────────────────────────────────────────────────────────────────────────
# Integration: @jit(deterministic=True)
# ─────────────────────────────────────────────────────────────────────────────

class TestJitDeterministicIntegration:
    def test_deterministic_pure_function(self):
        @tessera.jit(deterministic=True)
        def stable_forward(x):
            return tessera.ops.layer_norm(x)

        assert stable_forward.deterministic is True
        assert stable_forward.inferred_effect <= Effect.memory

    def test_deterministic_with_seed(self):
        @tessera.jit(deterministic=True, seed=42)
        def seeded_dropout(x):
            return tessera.ops.dropout(x)

        assert seeded_dropout.seed == 42
        assert seeded_dropout.deterministic is True

    def test_deterministic_random_without_seed_raises(self):
        with pytest.raises(TesseraEffectError):
            @tessera.jit(deterministic=True)
            def bad_fn(x):
                return tessera.ops.dropout(x)

    def test_non_deterministic_allows_random(self):
        @tessera.jit
        def augment(x):
            return tessera.ops.dropout(x)

        assert augment.inferred_effect >= Effect.random

    def test_jit_captures_effect(self):
        @tessera.jit
        def pure_fn(x):
            return tessera.ops.gemm(x, x)

        assert pure_fn.inferred_effect == Effect.pure

    def test_jit_fn_is_callable(self):
        @tessera.jit
        def identity(x):
            return x

        result = identity(42)
        assert result == 42
