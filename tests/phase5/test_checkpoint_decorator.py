"""
Phase 5 — test_checkpoint_decorator.py

Tests for @checkpoint_jit decorator, CollectiveCheckpointConfig, and
CheckpointIRAnnotator integration.
"""
import pytest
from tessera.compiler.checkpoint import (
    checkpoint_jit, CheckpointPolicy,
    CollectiveCheckpointConfig, CheckpointIRAnnotator,
)


class TestCheckpointJitDecorator:
    def test_bare_decorator_preserves_function(self):
        @checkpoint_jit
        def fn(x):
            return x + 1
        assert fn(5) == 6

    def test_bare_decorator_sets_checkpoint_flag(self):
        @checkpoint_jit
        def fn(x):
            return x
        assert fn.__tessera_checkpoint__ is True

    def test_bare_decorator_attaches_config(self):
        @checkpoint_jit
        def fn(x):
            return x
        cfg = fn.__tessera_checkpoint_config__
        assert isinstance(cfg, CollectiveCheckpointConfig)

    def test_parametrized_decorator_preserves_function(self):
        @checkpoint_jit(interval=4)
        def fn(x):
            return x * 2
        assert fn(3) == 6

    def test_parametrized_decorator_sets_interval(self):
        @checkpoint_jit(interval=4)
        def fn(x):
            return x
        assert fn.__tessera_checkpoint_config__.interval == 4

    def test_parametrized_decorator_sets_policy(self):
        @checkpoint_jit(policy=CheckpointPolicy.FULL)
        def fn(x):
            return x
        assert fn.__tessera_checkpoint_config__.policy == CheckpointPolicy.FULL

    def test_preserves_function_name(self):
        @checkpoint_jit(interval=2)
        def my_forward(x):
            return x
        assert my_forward.__name__ == "my_forward"

    def test_preserves_docstring(self):
        @checkpoint_jit(interval=2)
        def documented(x):
            """My docstring."""
            return x
        assert "My docstring" in documented.__doc__

    def test_memory_budget_propagated(self):
        @checkpoint_jit(memory_budget_gb=16.0)
        def fn(x):
            return x
        assert fn.__tessera_checkpoint_config__.memory_budget_gb == 16.0

    def test_invalid_interval_raises_at_decoration(self):
        with pytest.raises(ValueError):
            @checkpoint_jit(interval=0)
            def fn(x):
                return x

    def test_config_default_policy(self):
        @checkpoint_jit
        def fn(x):
            return x
        assert fn.__tessera_checkpoint_config__.policy == \
               CheckpointPolicy.SELECTIVE


class TestAnnotatorViaDecorator:
    """Integration: use annotator with the config attached by the decorator."""

    def _decorated_fn(self, interval=2, policy=CheckpointPolicy.SELECTIVE):
        @checkpoint_jit(interval=interval, policy=policy)
        def fn(x):
            return x
        return fn

    def _layers(self):
        return ["embed", "attn_0", "fc1_0", "fc2_0",
                "attn_1", "fc1_1", "fc2_1", "lm_head"]

    def test_annotator_from_decorator_config(self):
        fn = self._decorated_fn(interval=2)
        cfg = fn.__tessera_checkpoint_config__
        ann = CheckpointIRAnnotator(cfg)
        marks = ann.annotate(self._layers())
        # Every 2nd layer is checkpointed starting at index 0
        assert marks["embed"] is True
        assert marks["attn_0"] is False
        assert marks["fc1_0"] is True

    def test_full_policy_marks_all(self):
        fn = self._decorated_fn(policy=CheckpointPolicy.FULL)
        cfg = fn.__tessera_checkpoint_config__
        ann = CheckpointIRAnnotator(cfg)
        marks = ann.annotate(self._layers())
        assert all(marks.values())

    def test_none_policy_marks_none(self):
        fn = self._decorated_fn(policy=CheckpointPolicy.NONE)
        cfg = fn.__tessera_checkpoint_config__
        ann = CheckpointIRAnnotator(cfg)
        marks = ann.annotate(self._layers())
        assert not any(marks.values())

    def test_ir_annotations_contain_policy_string(self):
        fn = self._decorated_fn(interval=2)
        cfg = fn.__tessera_checkpoint_config__
        ann = CheckpointIRAnnotator(cfg)
        ir = ann.ir_annotations(self._layers())
        for s in ir:
            assert "selective" in s

    def test_recompute_hints_nonempty(self):
        fn = self._decorated_fn(interval=2)
        cfg = fn.__tessera_checkpoint_config__
        ann = CheckpointIRAnnotator(cfg)
        hints = ann.recompute_hints(self._layers())
        assert len(hints) > 0

    def test_checkpointed_layers_not_in_hints(self):
        fn = self._decorated_fn(interval=2)
        cfg = fn.__tessera_checkpoint_config__
        ann = CheckpointIRAnnotator(cfg)
        checkpointed = set(cfg.checkpoint_layers(self._layers()))
        hints = set(ann.recompute_hints(self._layers()))
        # Checkpoint markers and recompute hints must be disjoint.
        assert checkpointed.isdisjoint(hints)
