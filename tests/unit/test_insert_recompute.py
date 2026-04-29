"""
Phase 5 — test_insert_recompute.py

Tests for CollectiveCheckpointConfig and CheckpointIRAnnotator —
the Python-layer mirror of InsertRecomputePass.
"""
import pytest
from tessera.compiler.checkpoint import (
    CollectiveCheckpointConfig, CheckpointPolicy, CheckpointIRAnnotator,
)


class TestCollectiveCheckpointConfig:
    def test_default_interval(self):
        cfg = CollectiveCheckpointConfig()
        assert cfg.interval == 2

    def test_default_policy_selective(self):
        cfg = CollectiveCheckpointConfig()
        assert cfg.policy == CheckpointPolicy.SELECTIVE

    def test_invalid_interval_zero(self):
        with pytest.raises(ValueError):
            CollectiveCheckpointConfig(interval=0)

    def test_invalid_memory_budget(self):
        with pytest.raises(ValueError):
            CollectiveCheckpointConfig(memory_budget_gb=0.0)

    def test_repr_contains_interval(self):
        cfg = CollectiveCheckpointConfig(interval=4)
        assert "4" in repr(cfg)

    def test_to_ir_attr_contains_policy(self):
        cfg = CollectiveCheckpointConfig(policy=CheckpointPolicy.SELECTIVE)
        attr = cfg.to_ir_attr()
        assert "selective" in attr

    def test_to_mlir_attrs_is_string(self):
        cfg = CollectiveCheckpointConfig()
        assert isinstance(cfg.to_mlir_attrs(), str)

    def test_to_ir_attr_contains_interval(self):
        cfg = CollectiveCheckpointConfig(interval=3)
        assert "interval = 3" in cfg.to_ir_attr()

    def test_to_ir_attr_contains_memory_budget(self):
        cfg = CollectiveCheckpointConfig(memory_budget_gb=32.0)
        assert "32.0" in cfg.to_ir_attr()


class TestCheckpointLayers:
    """Test CollectiveCheckpointConfig.checkpoint_layers()."""

    def test_none_policy_returns_empty(self):
        cfg = CollectiveCheckpointConfig(policy=CheckpointPolicy.NONE)
        assert cfg.checkpoint_layers(["a", "b", "c"]) == []

    def test_full_policy_returns_all(self):
        cfg = CollectiveCheckpointConfig(policy=CheckpointPolicy.FULL)
        layers = ["a", "b", "c"]
        assert cfg.checkpoint_layers(layers) == layers

    def test_selective_interval_2_returns_even_indices(self):
        cfg = CollectiveCheckpointConfig(interval=2)
        layers = ["a", "b", "c", "d", "e", "f"]
        result = cfg.checkpoint_layers(layers)
        assert result == ["a", "c", "e"]

    def test_selective_interval_1_returns_all(self):
        cfg = CollectiveCheckpointConfig(interval=1)
        layers = ["a", "b", "c"]
        assert cfg.checkpoint_layers(layers) == ["a", "b", "c"]

    def test_selective_interval_3(self):
        cfg = CollectiveCheckpointConfig(interval=3)
        layers = ["l0", "l1", "l2", "l3", "l4", "l5"]
        assert cfg.checkpoint_layers(layers) == ["l0", "l3"]

    def test_disabled_returns_empty(self):
        cfg = CollectiveCheckpointConfig(enabled=False)
        assert cfg.checkpoint_layers(["a", "b"]) == []

    def test_empty_layers_returns_empty(self):
        cfg = CollectiveCheckpointConfig()
        assert cfg.checkpoint_layers([]) == []


class TestCheckpointIRAnnotator:
    def _annotator(self, interval=2, policy=CheckpointPolicy.SELECTIVE):
        return CheckpointIRAnnotator(
            CollectiveCheckpointConfig(interval=interval, policy=policy)
        )

    def test_annotate_returns_dict(self):
        ann = self._annotator().annotate(["a", "b", "c"])
        assert isinstance(ann, dict)

    def test_annotate_keys_match_layers(self):
        layers = ["a", "b", "c", "d"]
        ann = self._annotator().annotate(layers)
        assert set(ann.keys()) == set(layers)

    def test_annotate_correct_marks_interval_2(self):
        layers = ["a", "b", "c", "d"]
        ann = self._annotator(interval=2).annotate(layers)
        assert ann["a"] is True
        assert ann["b"] is False
        assert ann["c"] is True
        assert ann["d"] is False

    def test_ir_annotations_count_interval_2(self):
        layers = ["l0", "l1", "l2", "l3"]
        ann = self._annotator(interval=2)
        ir = ann.ir_annotations(layers)
        assert len(ir) == 2

    def test_ir_annotations_contain_layer_name(self):
        ann = self._annotator()
        ir = ann.ir_annotations(["layer_0.attn"])
        # layer_0.attn is index 0 → should be checkpointed
        assert any("layer_0.attn" in s for s in ir)

    def test_ir_annotations_full_policy(self):
        ann = CheckpointIRAnnotator(
            CollectiveCheckpointConfig(policy=CheckpointPolicy.FULL)
        )
        layers = ["a", "b", "c"]
        ir = ann.ir_annotations(layers)
        assert len(ir) == 3

    def test_recompute_hints_between_checkpoints(self):
        layers = ["a", "b", "c", "d", "e"]
        ann = self._annotator(interval=2)
        hints = ann.recompute_hints(layers)
        # Checkpoints at a (idx 0), c (idx 2), e (idx 4)
        # Between a and c: b  → recomputable
        # Between c and e: d  → recomputable
        assert "b" in hints
        assert "d" in hints
        assert "a" not in hints

    def test_repr_contains_config(self):
        ann = self._annotator()
        assert "CheckpointIRAnnotator" in repr(ann)
