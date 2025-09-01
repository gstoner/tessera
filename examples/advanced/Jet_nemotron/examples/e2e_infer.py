# examples/e2e_infer.py
"""Minimal end-to-end example: tokenizer → Transformer forward pass.

This is a **scaffold** with a minimal tokenizer and a tiny model config.
Replace the tokenizer with your production one (SentencePiece/BPE) and
load real weights via `llama_convert.py` if desired.

Run (pseudo):
    python examples/e2e_infer.py
"""
from dataclasses import dataclass
from typing import List
import math

# Tessera imports (modeled)
from tessera import Tensor
from tessera_jetnemotron.transformer_block import Transformer, TransformerConfig

class MinimalTokenizer:
    def __init__(self, vocab: List[str] = None):
        self.vocab = {"<pad>":0, "<bos>":1, "<eos>":2}
        if vocab:
            for i,tok in enumerate(vocab, start=len(self.vocab)):
                self.vocab[tok] = i
        self.inv = {i:t for t,i in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        toks = text.strip().split()
        return [1] + [self.vocab.get(t, self.vocab.setdefault(t, len(self.vocab))) for t in toks] + [2]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.inv.get(i, f"<{i}>") for i in ids)

def main():
    tok = MinimalTokenizer(["hello", "world", "from", "tessera"])
    ids = tok.encode("hello world from tessera")
    # Fake embedding tensor (B=1, S=len(ids), D)
    B, S, D = 1, len(ids), 128
    import numpy as np
    x_np = np.random.randn(B, S, D).astype(np.float32)
    x = Tensor(x_np)  # stand-in

    # Mixed-attention model: every 4th layer uses full attention
    cfg = TransformerConfig(
        d_model=D, n_heads=8, head_dim=D//8, mlp_hidden=4*D,
        n_layers=6, attn_types=["full" if i % 4 == 0 else "jet" for i in range(6)],
        dropout_p=0.0, dtype="fp8_e4m3", accum="fp32"
    )
    model = Transformer(cfg)
    y, states = model(x, causal=True, streaming=False)
    print("Output shape:", y.shape, "#states:", len(states))

if __name__ == "__main__":
    main()
