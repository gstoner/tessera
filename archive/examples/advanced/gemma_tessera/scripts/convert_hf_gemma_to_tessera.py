#!/usr/bin/env python3
"""
HuggingFace Gemma → Tessera weight converter.

Supports Gemma 2, Gemma 3, and Gemma 4 checkpoints.
Maps HF state-dict keys to the refactored Tessera model
(separate q_proj/k_proj/v_proj/o_proj and gate_proj/up_proj/down_proj).

Usage:
    python scripts/convert_hf_gemma_to_tessera.py \\
        --hf-id google/gemma-2-2b-it \\
        --out weights_tessera.pt

Requirements (install via pyproject.toml [convert]):
    pip install "tessera_gemma[convert]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

# ---------------------------------------------------------------------------
# HF → Tessera key mapping
# ---------------------------------------------------------------------------
# Format: tessera_key_suffix → list of candidate HF key suffixes (first match wins)
#
# HF Gemma uses:
#   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
#   model.layers.{i}.mlp.{gate,up,down}_proj.weight
#   model.layers.{i}.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight
#   model.embed_tokens.weight  /  lm_head.weight
#   model.norm.weight

_LAYER_KEY_MAP: Dict[str, list[str]] = {
    # Attention
    "self_attn.q_proj.weight": ["self_attn.q_proj.weight"],
    "self_attn.k_proj.weight": ["self_attn.k_proj.weight"],
    "self_attn.v_proj.weight": ["self_attn.v_proj.weight"],
    "self_attn.o_proj.weight": ["self_attn.o_proj.weight"],
    # MLP (unified names for Gemma2/3/4)
    "mlp.gate_proj.weight": ["mlp.gate_proj.weight"],
    "mlp.up_proj.weight":   ["mlp.up_proj.weight"],
    "mlp.down_proj.weight": ["mlp.down_proj.weight"],
    # Norms — Gemma4 uses pre/post feedforward norms (4 norms per layer)
    "input_layernorm.weight":            ["input_layernorm.weight"],
    "post_attention_layernorm.weight":   [
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",   # Gemma2 alt name
    ],
    # Gemma4 extra norms (ignored if not present in Tessera model)
    "pre_feedforward_layernorm.weight":  ["pre_feedforward_layernorm.weight"],
    "post_feedforward_layernorm.weight": ["post_feedforward_layernorm.weight"],
}

_GLOBAL_KEY_MAP: Dict[str, list[str]] = {
    "embed_tokens.weight": ["model.embed_tokens.weight", "transformer.wte.weight"],
    "norm.weight":         ["model.norm.weight", "transformer.ln_f.weight"],
    "lm_head.weight":      ["lm_head.weight"],
}


def _find_hf_key(hf_sd: dict, candidates: list[str]) -> Optional[str]:
    for k in candidates:
        if k in hf_sd:
            return k
    return None


def convert(
    hf_sd: dict,
    tessera_sd: dict,
    num_layers: int,
) -> tuple[dict, int, int]:
    """Return (new_sd, n_copied, n_missing)."""
    new_sd = dict(tessera_sd)
    copied = 0
    missing = 0

    # Global keys
    for t_suffix, candidates in _GLOBAL_KEY_MAP.items():
        hk = _find_hf_key(hf_sd, candidates)
        if hk and hk in hf_sd and hf_sd[hk].shape == tessera_sd.get(t_suffix, torch.empty(0)).shape:
            new_sd[t_suffix] = hf_sd[hk].clone()
            copied += 1
        else:
            missing += 1

    # Per-layer keys
    for i in range(num_layers):
        for t_suffix, candidates in _LAYER_KEY_MAP.items():
            t_key = f"layers.{i}.{t_suffix}"
            hf_candidates = [f"model.layers.{i}.{c}" for c in candidates]
            hk = _find_hf_key(hf_sd, hf_candidates)
            t_shape = tessera_sd.get(t_key, torch.empty(0)).shape
            if hk and hf_sd[hk].shape == t_shape:
                new_sd[t_key] = hf_sd[hk].clone()
                copied += 1
            else:
                if t_key in tessera_sd:
                    missing += 1

    return new_sd, copied, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-id",  required=True,
                    help="HuggingFace model id, e.g. 'google/gemma-2-2b-it'")
    ap.add_argument("--out",    default="weights_tessera.pt",
                    help="Output .pt path (default: weights_tessera.pt)")
    ap.add_argument("--dtype",  default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="Weight dtype (default: float16)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        print("ERROR: transformers not installed. pip install 'tessera_gemma[convert]'")
        sys.exit(1)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading HF checkpoint: {args.hf_id}")
    hf_config = AutoConfig.from_pretrained(args.hf_id)
    hf_model  = AutoModelForCausalLM.from_pretrained(
        args.hf_id, torch_dtype=dtype, device_map=args.device
    )
    hf_sd = hf_model.state_dict()

    # Build matching Tessera config from HF config
    from tessera_gemma.configs import GemmaConfig
    from tessera_gemma.model_tessera import TesseraGemmaForCausalLM

    cfg = GemmaConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_kv_heads=getattr(hf_config, "num_key_value_heads",
                             hf_config.num_attention_heads),
        head_dim=getattr(hf_config, "head_dim",
                         hf_config.hidden_size // hf_config.num_attention_heads),
        rope_theta=getattr(hf_config, "rope_theta", 10_000.0),
        max_position_embeddings=getattr(hf_config, "max_position_embeddings", 8_192),
        sliding_window_size=getattr(hf_config, "sliding_window", None),
        sliding_window_pattern="alternating",
    )

    tessera_model = TesseraGemmaForCausalLM(cfg)
    tessera_sd    = tessera_model.state_dict()

    print(f"Tessera model has {sum(v.numel() for v in tessera_sd.values()):,} params")

    new_sd, n_copied, n_missing = convert(hf_sd, tessera_sd, cfg.num_hidden_layers)
    total = n_copied + n_missing
    print(f"Copied {n_copied}/{total} tensors ({n_missing} kept as random init)")

    if n_missing > 0:
        print("NOTE: missing tensors kept as random init. "
              "Inspect key mappings in this script for your checkpoint variant.")

    out_path = Path(args.out)
    torch.save({"model": new_sd, "config": cfg.__dict__}, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
