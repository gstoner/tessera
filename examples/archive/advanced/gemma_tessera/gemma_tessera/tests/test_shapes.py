import torch, pytest
from tessera_gemma.configs import GemmaConfig
from tessera_gemma.model_tessera import TesseraGemmaForCausalLM

@pytest.mark.parametrize("B,T", [(1,8),(2,16)])
def test_forward_shapes(B,T):
    cfg = GemmaConfig(vocab_size=32000, hidden_size=512, intermediate_size=1536, num_hidden_layers=2, num_attention_heads=8, num_kv_heads=2, max_position_embeddings=2048)
    m = TesseraGemmaForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        y = m(x)
    assert y.shape == (B,T,cfg.vocab_size)
