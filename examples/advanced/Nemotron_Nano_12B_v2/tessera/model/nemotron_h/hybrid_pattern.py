"""
Nemotron‑H hybrid stack builder for Tessera.
Creates a list of blocks based on `hybrid_override_pattern`:
    M  -> Mamba2MixerBlock
    *  -> AttentionBlock (GQA)
    -  -> MLPBlock (ReLU^2)
"""
from dataclasses import dataclass
from typing import List

@dataclass
class NemotronHConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    hybrid_override_pattern: str
    # Mamba
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    conv_kernel: int = 4
    chunk_size: int = 128
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    mlp_bias: bool = False

class Block: ...
class Mamba2MixerBlock(Block): ...
class AttentionBlock(Block): ...
class MLPBlock(Block): ...

def build_hybrid_stack(cfg: NemotronHConfig) -> List[Block]:
    assert len(cfg.hybrid_override_pattern) == cfg.num_hidden_layers,         "pattern length must equal num_hidden_layers"
    blocks: List[Block] = []
    for i, ch in enumerate(cfg.hybrid_override_pattern):
        if ch == "M":
            blocks.append(Mamba2MixerBlock())  # wire params in your impl
        elif ch == "*":
            blocks.append(AttentionBlock())
        elif ch == "-":
            blocks.append(MLPBlock())
        else:
            raise ValueError(f"Invalid char '{ch}' at index {i}")
    return blocks
