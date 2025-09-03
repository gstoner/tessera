#!/usr/bin/env python3

"""
HF Gemma → Tessera tensor converter (skeleton).

Usage:
  python scripts/convert_hf_gemma_to_tessera.py --hf-id google/gemma-2-2b-it --out weights_tessera.pt

Notes:
- You must have accepted Google's Gemma terms and have access to the HF repo.
- This script maps common tensor names to the starter Tessera model.
- Adjust key renames as needed for exact checkpoint variants.
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tessera_gemma.configs import GemmaConfig
from tessera_gemma.model_tessera import TesseraGemmaForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--out", default="weights_tessera.pt")
    args = ap.parse_args()

    print("Loading HF model:", args.hf_id)
    hf = AutoModelForCausalLM.from_pretrained(args.hf_id, torch_dtype=torch.float16, device_map="cpu")
    cfg = GemmaConfig(
        vocab_size=hf.config.vocab_size,
        hidden_size=hf.config.hidden_size,
        num_hidden_layers=hf.config.num_hidden_layers,
        num_attention_heads=hf.config.num_attention_heads,
        num_kv_heads=getattr(hf.config, "num_key_value_heads", hf.config.num_attention_heads//2),
        intermediate_size=hf.config.intermediate_size,
        max_position_embeddings=getattr(hf.config, "max_position_embeddings", 8192),
    )
    model = TesseraGemmaForCausalLM(cfg)
    sd = model.state_dict()
    hfsd = hf.state_dict()

    # Minimal mapping (adjust to exact HF names if they differ)
    rename = {}
    # Embedding
    rename["embed.weight"] = "model.embed_tokens.weight" if any(k.startswith("model.embed_tokens") for k in hfsd.keys()) else "transformer.wte.weight"
    # Final LM head typically tied to embed
    # Attention and MLP per-layer naming must be adapted by inspecting keys

    # Naive copy where names align; otherwise you will extend this mapping
    copied, missing = 0, 0
    new_sd = {}
    for k in sd.keys():
        hk = None
        # direct hit
        if k in hfsd:
            hk = k
        # simple patterns
        elif k.endswith("embed.weight"):
            hk = rename.get("embed.weight")
        if hk and hk in hfsd and hfsd[hk].shape == sd[k].shape:
            new_sd[k] = hfsd[hk]
            copied += 1
        else:
            new_sd[k] = sd[k]  # keep random init
            missing += 1

    print(f"Copied {copied} tensors, left {missing} as init. Save to {args.out}")
    torch.save({"model": new_sd, "config": cfg.__dict__}, args.out)

if __name__ == "__main__":
    main()
