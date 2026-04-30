"""Tests for GemmaConfig — no GPU or torch required."""
import pytest
from tessera_gemma.configs import GemmaConfig


class TestGemmaConfigDefaults:
    def test_default_head_dim(self):
        cfg = GemmaConfig()
        assert cfg.head_dim == 256

    def test_default_q_dim(self):
        cfg = GemmaConfig()
        assert cfg.q_dim == cfg.num_attention_heads * cfg.head_dim

    def test_default_kv_dim(self):
        cfg = GemmaConfig()
        assert cfg.kv_dim == cfg.num_kv_heads * cfg.head_dim

    def test_groups(self):
        cfg = GemmaConfig(num_attention_heads=16, num_kv_heads=8, head_dim=256)
        assert cfg.groups == 2

    def test_groups_mha(self):
        cfg = GemmaConfig(num_attention_heads=8, num_kv_heads=8, head_dim=64)
        assert cfg.groups == 1


class TestSliddingWindowPattern:
    def test_alternating_even_is_full(self):
        cfg = GemmaConfig(sliding_window_size=4096, sliding_window_pattern="alternating")
        assert cfg.layer_attention_kind(0) == "full"
        assert cfg.layer_attention_kind(2) == "full"
        assert cfg.layer_attention_kind(60) == "full"

    def test_alternating_odd_is_swa(self):
        cfg = GemmaConfig(sliding_window_size=4096, sliding_window_pattern="alternating")
        assert cfg.layer_attention_kind(1) == "sliding_window"
        assert cfg.layer_attention_kind(3) == "sliding_window"
        assert cfg.layer_attention_kind(61) == "sliding_window"

    def test_none_pattern(self):
        cfg = GemmaConfig(sliding_window_size=4096, sliding_window_pattern="none")
        for i in range(10):
            assert cfg.layer_attention_kind(i) == "full"

    def test_all_pattern(self):
        cfg = GemmaConfig(sliding_window_size=4096, sliding_window_pattern="all")
        for i in range(10):
            assert cfg.layer_attention_kind(i) == "sliding_window"

    def test_no_window_size_means_full(self):
        cfg = GemmaConfig(sliding_window_size=None, sliding_window_pattern="alternating")
        for i in range(10):
            assert cfg.layer_attention_kind(i) == "full"


class TestGemma4Factories:
    @pytest.mark.parametrize("factory,expected", [
        ("gemma4_4b",  {"hidden_size": 2560, "num_hidden_layers": 34,
                         "num_attention_heads": 16, "num_kv_heads": 8}),
        ("gemma4_12b", {"hidden_size": 3840, "num_hidden_layers": 46,
                         "num_attention_heads": 24, "num_kv_heads": 8}),
        ("gemma4_27b", {"hidden_size": 5632, "num_hidden_layers": 62,
                         "num_attention_heads": 32, "num_kv_heads": 16}),
    ])
    def test_factory_dims(self, factory, expected):
        cfg = getattr(GemmaConfig, factory)()
        for attr, val in expected.items():
            assert getattr(cfg, attr) == val, f"{factory}.{attr} mismatch"

    @pytest.mark.parametrize("factory", ["gemma4_4b", "gemma4_12b", "gemma4_27b"])
    def test_head_dim_256(self, factory):
        cfg = getattr(GemmaConfig, factory)()
        assert cfg.head_dim == 256

    @pytest.mark.parametrize("factory", ["gemma4_4b", "gemma4_12b", "gemma4_27b"])
    def test_geglu_mlp(self, factory):
        cfg = getattr(GemmaConfig, factory)()
        assert cfg.mlp_type == "geglu"

    @pytest.mark.parametrize("factory", ["gemma4_4b", "gemma4_12b", "gemma4_27b"])
    def test_alternating_swa(self, factory):
        cfg = getattr(GemmaConfig, factory)()
        assert cfg.sliding_window_size == 4096
        assert cfg.sliding_window_pattern == "alternating"

    @pytest.mark.parametrize("factory", ["gemma4_4b", "gemma4_12b", "gemma4_27b"])
    def test_long_context(self, factory):
        cfg = getattr(GemmaConfig, factory)()
        assert cfg.max_position_embeddings == 131_072

    @pytest.mark.parametrize("factory", ["gemma4_4b", "gemma4_12b", "gemma4_27b"])
    def test_q_kv_dims_divisible(self, factory):
        cfg = getattr(GemmaConfig, factory)()
        assert cfg.q_dim % cfg.num_attention_heads == 0
        assert cfg.kv_dim % cfg.num_kv_heads == 0

    def test_groups_4b(self):
        cfg = GemmaConfig.gemma4_4b()
        assert cfg.groups == 2   # 16 Q heads / 8 KV heads

    def test_groups_12b(self):
        cfg = GemmaConfig.gemma4_12b()
        assert cfg.groups == 3   # 24 Q heads / 8 KV heads

    def test_groups_27b(self):
        cfg = GemmaConfig.gemma4_27b()
        assert cfg.groups == 2   # 32 Q heads / 16 KV heads


class TestDebugTiny:
    def setup_method(self):
        self.cfg = GemmaConfig.debug_tiny()

    def test_head_dim(self):
        assert self.cfg.head_dim == 64

    def test_small_vocab(self):
        assert self.cfg.vocab_size == 32_000

    def test_swiglu(self):
        assert self.cfg.mlp_type == "swiglu"

    def test_alternating_swa(self):
        assert self.cfg.layer_attention_kind(0) == "full"
        assert self.cfg.layer_attention_kind(1) == "sliding_window"

    def test_q_dim(self):
        assert self.cfg.q_dim == 8 * 64    # 512

    def test_kv_dim(self):
        assert self.cfg.kv_dim == 2 * 64   # 128
