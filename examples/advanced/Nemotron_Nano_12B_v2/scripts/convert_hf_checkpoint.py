#!/usr/bin/env python3
"""
HF -> Tessera checkpoint exporter (skeleton).
- Downloads HF repo (default: NVIDIA-Nemotron-Nano-12B-v2-Base)
- Writes manifest.json + layer shards (.npz) under --out
"""
import argparse, json, os, pathlib, numpy as np
from dataclasses import asdict
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    cache_dir = snapshot_download(args.repo, revision=args.revision)
    cfg = AutoConfig.from_pretrained(cache_dir)
    tok = AutoTokenizer.from_pretrained(cache_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cache_dir, torch_dtype="auto", low_cpu_mem_usage=True, device_map=None)

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "repo": args.repo,
        "config": cfg.to_dict(),
        "tokenizer_files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "format": "tessera_nemotron_h_v1",
        "layers": [],
    }

    # Very light illustration: dump embedding + final lm_head only.
    # Extend to per-layer projections following modeling_nemotron_h.py.
    import torch
    emb_w = model.get_input_embeddings().weight.detach().cpu().numpy()
    np.savez_compressed(os.path.join(args.out, "embeddings.npz"), weight=emb_w)
    lm_w = model.get_output_embeddings().weight.detach().cpu().numpy()
    np.savez_compressed(os.path.join(args.out, "lm_head.npz"), weight=lm_w)
    manifest["layers"].append({"name":"embedding", "file":"embeddings.npz", "shape": list(emb_w.shape)})
    manifest["layers"].append({"name":"lm_head", "file":"lm_head.npz", "shape": list(lm_w.shape)})

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest + 2 shards to {args.out}. Extend for full parity.")

if __name__ == "__main__":
    main()
