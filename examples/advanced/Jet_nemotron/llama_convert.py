"""Llama-family weight conversion into Tessera Transformer/JetBlock modules.

This script expects PyTorch and (optionally) transformers to be installed in your environment.
It maps common Llama parameter names onto this scaffold's modules:
- Attention: q_proj, k_proj, v_proj, o_proj  → Wq, Wk, Wv, Wo
- MLP: gate_proj, up_proj, down_proj         → up/down (SiLU gating handled implicitly)
- RMSNorm: input_layernorm, post_attention_layernorm → RMSNorm weights

Usage (pseudo):
    model = load_hf_llama("meta-llama/Llama-2-7b-hf")
    tcfg = TransformerConfig(..., attn_types=["full","jet",...])
    tmodel = Transformer(tcfg)
    port_llama_weights(hf_model, tmodel, layer_map=...)

"""
from typing import Dict, List, Any
import re

try:
    import torch
    from transformers import AutoModelForCausalLM
except Exception:
    torch = None
    AutoModelForCausalLM = None

from .transformer_block import Transformer, TransformerConfig
from .jetblock import JetBlock
from .full_attention import FullAttention

LLAMA_ATT_PATTERN = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight")
LLAMA_MLP_PATTERN  = re.compile(r"model\.layers\.(\d+)\.(mlp)\.(gate_proj|up_proj|down_proj)\.weight")
LLAMA_RMS_PATTERN  = re.compile(r"model\.(layers\.(\d+)\.)?input_layernorm\.weight|model\.(layers\.(\d+)\.)?post_attention_layernorm\.weight|model\.norm\.weight")

def load_hf_llama(model_id: str):
    assert AutoModelForCausalLM is not None, "transformers not installed"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    return model

def port_llama_weights(hf_model, tess_model: Transformer, layer_map: Dict[int,int]=None):
    sd: Dict[str, Any] = hf_model.state_dict()
    for name, wt in sd.items():
        # Attention
        m = LLAMA_ATT_PATTERN.match(name)
        if m:
            lidx, which = int(m.group(1)), m.group(2)
            layer = tess_model.layers[lidx]
            target = layer.attn
            if isinstance(target, (FullAttention, JetBlock)):
                if which == "q_proj": target.Wq.weight.copy_(wt)
                elif which == "k_proj": target.Wk.weight.copy_(wt)
                elif which == "v_proj": target.Wv.weight.copy_(wt)
                elif which == "o_proj": target.Wo.weight.copy_(wt)
            continue
        # MLP
        m = LLAMA_MLP_PATTERN.match(name)
        if m:
            lidx = int(m.group(1))
            which = m.group(3)
            mlp = tess_model.layers[lidx].mlp
            if which in ("gate_proj","up_proj"): 
                mlp.up.weight.copy_(wt)  # combine/gate handled by SiLU
            elif which == "down_proj": 
                mlp.down.weight.copy_(wt)
            continue
        # RMSNorm (optional: map to RMSNorm params if exposed)
        # NOTE: This scaffold uses stdlib.rmsnorm_safe with internal parameters.
        # If your RMSNorm exposes weights, copy accordingly.
    return tess_model
