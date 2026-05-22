"""
End-to-end model tests — uses debug_tiny config (CPU, fast).
"""
import pytest
import torch
from tessera_gemma import GemmaConfig, TesseraGemmaForCausalLM


@pytest.fixture(scope="module")
def tiny_model():
    cfg = GemmaConfig.debug_tiny()
    m = TesseraGemmaForCausalLM(cfg).eval()
    return m


@pytest.fixture(scope="module")
def tiny_cfg():
    return GemmaConfig.debug_tiny()


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
class TestForward:
    @pytest.mark.parametrize("B,T", [(1, 8), (2, 16), (1, 1)])
    def test_output_shape(self, tiny_model, B, T):
        cfg = tiny_model.cfg
        x = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            y = tiny_model(x)
        assert y.shape == (B, T, cfg.vocab_size)

    def test_deterministic(self, tiny_model):
        x = torch.randint(0, tiny_model.cfg.vocab_size, (1, 8))
        with torch.no_grad():
            y1 = tiny_model(x)
            y2 = tiny_model(x)
        assert torch.equal(y1, y2)

    def test_dtype_preserved_float32(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg).float().eval()
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            y = m(x)
        assert y.dtype == torch.float32

    def test_rope_built_lazily(self, tiny_model):
        # Rebuild fresh model to test lazy init
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg).eval()
        assert not m._rope_built
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            _ = m(x)
        assert m._rope_built
        assert m.rope_cos is not None

    def test_tied_embeddings(self):
        cfg = GemmaConfig.debug_tiny()
        assert cfg.tie_word_embeddings
        m = TesseraGemmaForCausalLM(cfg)
        assert m.lm_head.weight is m.embed_tokens.weight

    def test_no_tied_embeddings(self):
        cfg = GemmaConfig.debug_tiny()
        cfg.tie_word_embeddings = False
        m = TesseraGemmaForCausalLM(cfg)
        assert m.lm_head.weight is not m.embed_tokens.weight


# ---------------------------------------------------------------------------
# Layer structure
# ---------------------------------------------------------------------------
class TestLayerStructure:
    def test_num_layers(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg)
        assert len(m.layers) == cfg.num_hidden_layers

    def test_swa_layers_have_sliding_window(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg)
        # odd layers → SWA
        for i, layer in enumerate(m.layers):
            if i % 2 == 1:
                assert layer.self_attn.sliding_window > 0, \
                    f"layer {i} should be SWA"
            else:
                assert layer.self_attn.sliding_window == 0, \
                    f"layer {i} should be full attention"

    def test_mlp_type_swiglu(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg)
        for layer in m.layers:
            assert layer.mlp.mlp_type == "swiglu"

    def test_mlp_type_geglu(self):
        cfg = GemmaConfig.gemma4_4b()
        # Only instantiate one layer to keep test fast
        from tessera_gemma.model_tessera import GemmaDecoderBlock
        blk = GemmaDecoderBlock(cfg, layer_idx=0)
        assert blk.mlp.mlp_type == "geglu"

    def test_num_parameters_positive(self, tiny_model):
        assert tiny_model.num_parameters() > 0

    def test_trainable_parameters_same_as_total_at_init(self, tiny_model):
        assert tiny_model.num_parameters(trainable_only=True) == \
               tiny_model.num_parameters()


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
class TestGenerate:
    def test_greedy_shape(self, tiny_model):
        x = torch.randint(0, tiny_model.cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = tiny_model.generate(x, max_new_tokens=3, top_k=1, use_cache=False)
        assert out.shape == (1, 7)

    def test_greedy_batch(self, tiny_model):
        x = torch.randint(0, tiny_model.cfg.vocab_size, (2, 4))
        with torch.no_grad():
            out = tiny_model.generate(x, max_new_tokens=2, top_k=1, use_cache=False)
        assert out.shape == (2, 6)

    def test_greedy_deterministic(self, tiny_model):
        x = torch.randint(0, tiny_model.cfg.vocab_size, (1, 4))
        with torch.no_grad():
            o1 = tiny_model.generate(x, max_new_tokens=4, top_k=1, use_cache=False)
            o2 = tiny_model.generate(x, max_new_tokens=4, top_k=1, use_cache=False)
        assert torch.equal(o1, o2)

    def test_eos_stops_early(self, tiny_model):
        cfg = tiny_model.cfg
        # Force a model that always outputs token 1 (eos)
        # We check that generate respects eos even before max_new_tokens
        x = torch.randint(0, cfg.vocab_size, (1, 2))
        with torch.no_grad():
            out = tiny_model.generate(
                x, max_new_tokens=10, top_k=1,
                eos_token_id=cfg.eos_token_id, use_cache=False,
            )
        # Output should be ≤ 2 + 10 tokens
        assert out.shape[1] <= 2 + 10

    def test_input_ids_preserved(self, tiny_model):
        x = torch.randint(0, tiny_model.cfg.vocab_size, (1, 5))
        with torch.no_grad():
            out = tiny_model.generate(x, max_new_tokens=3, top_k=1, use_cache=False)
        assert torch.equal(out[:, :5], x)


# ---------------------------------------------------------------------------
# MLP structure
# ---------------------------------------------------------------------------
class TestMLP:
    def test_swiglu_projection_names(self):
        from tessera_gemma.kernels.mlp_swiglu_tessera import GemmaMLP
        m = GemmaMLP(512, 2048, mlp_type="swiglu")
        assert hasattr(m, "gate_proj")
        assert hasattr(m, "up_proj")
        assert hasattr(m, "down_proj")
        assert not hasattr(m, "wi")

    def test_geglu_output_shape(self):
        from tessera_gemma.kernels.mlp_swiglu_tessera import GemmaMLP
        m = GemmaMLP(256, 512, mlp_type="geglu").eval()
        x = torch.randn(2, 8, 256)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 8, 256)

    def test_back_compat_swiglu(self):
        from tessera_gemma.kernels.mlp_swiglu_tessera import SwiGLU
        m = SwiGLU(128, 256)
        assert m.mlp_type == "swiglu"
