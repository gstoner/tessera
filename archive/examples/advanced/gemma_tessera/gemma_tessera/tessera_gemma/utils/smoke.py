# Simple quick test
import torch
from ..configs import GemmaConfig
from ..model_tessera import TesseraGemmaForCausalLM

def main():
    cfg = GemmaConfig(vocab_size=32000, hidden_size=512, intermediate_size=1536, num_hidden_layers=4, num_attention_heads=8, num_kv_heads=2, max_position_embeddings=2048)
    m = TesseraGemmaForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        y = m(x)
    print("OK:", tuple(y.shape))

if __name__ == "__main__":
    main()
