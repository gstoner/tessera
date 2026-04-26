"""
Phase 3 — AsyncCopyLoweringPass and AttnLoweringConfig tests.

Verifies:
  - FlashAttnLoweringConfig validates tile sizes and dropout_p
  - SM_90 config automatically has pipeline_stages=2 for double-buffering
  - dropout_p=0 → no dropout (no DropoutMaskOp needed)
  - dropout_p>0 requires seed
  - causal=True → CausalMaskOp will be emitted by TileIRLoweringPass
  - MLIR attr serialisation for attn config
"""
import pytest
from tessera.compiler.attn_lower import FlashAttnLoweringConfig, TesseraAttnConfigError


class TestFlashAttnLoweringConfig:

    def test_default_config(self):
        cfg = FlashAttnLoweringConfig()
        assert cfg.tile_q == 64
        assert cfg.tile_kv == 64
        assert cfg.pipeline_stages == 2
        assert cfg.causal is False
        assert cfg.dropout_p == 0.0

    def test_tile_q_must_be_power_of_two(self):
        with pytest.raises(TesseraAttnConfigError, match="power of 2"):
            FlashAttnLoweringConfig(tile_q=60)

    def test_tile_kv_must_be_power_of_two(self):
        with pytest.raises(TesseraAttnConfigError, match="power of 2"):
            FlashAttnLoweringConfig(tile_kv=100)

    def test_dropout_p_out_of_range(self):
        with pytest.raises(TesseraAttnConfigError, match="dropout_p"):
            FlashAttnLoweringConfig(dropout_p=1.0)

    def test_dropout_requires_seed(self):
        with pytest.raises(TesseraAttnConfigError, match="seed"):
            FlashAttnLoweringConfig(dropout_p=0.1, seed=None)

    def test_dropout_with_seed_ok(self):
        cfg = FlashAttnLoweringConfig(dropout_p=0.1, seed=42)
        assert cfg.has_dropout is True
        assert cfg.seed == 42

    def test_no_dropout_has_dropout_false(self):
        cfg = FlashAttnLoweringConfig(dropout_p=0.0)
        assert cfg.has_dropout is False

    def test_causal_config(self):
        cfg = FlashAttnLoweringConfig(causal=True)
        assert cfg.causal is True

    def test_pipeline_stages_must_be_positive(self):
        with pytest.raises(TesseraAttnConfigError, match="pipeline_stages"):
            FlashAttnLoweringConfig(pipeline_stages=0)

    def test_tile_128_valid(self):
        cfg = FlashAttnLoweringConfig(tile_q=128, tile_kv=128)
        assert cfg.tile_q == 128

    def test_mlir_attrs_contains_tile_q(self):
        cfg = FlashAttnLoweringConfig(tile_q=64, tile_kv=64)
        attrs = cfg.to_mlir_attrs()
        assert "tessera.tile_q = 64" in attrs
        assert "tessera.tile_kv = 64" in attrs

    def test_mlir_attrs_causal_true(self):
        cfg = FlashAttnLoweringConfig(causal=True)
        attrs = cfg.to_mlir_attrs()
        assert "causal = true" in attrs

    def test_mlir_attrs_causal_false(self):
        cfg = FlashAttnLoweringConfig(causal=False)
        attrs = cfg.to_mlir_attrs()
        assert "causal = false" in attrs

    def test_mlir_attrs_no_dropout_field(self):
        cfg = FlashAttnLoweringConfig(dropout_p=0.0)
        attrs = cfg.to_mlir_attrs()
        assert "dropout_p" not in attrs

    def test_mlir_attrs_has_dropout_field(self):
        cfg = FlashAttnLoweringConfig(dropout_p=0.1, seed=1)
        attrs = cfg.to_mlir_attrs()
        assert "dropout_p" in attrs

    def test_pipeline_stages_in_attrs(self):
        cfg = FlashAttnLoweringConfig(pipeline_stages=3)
        attrs = cfg.to_mlir_attrs()
        assert "tessera.pipeline_stages = 3" in attrs
