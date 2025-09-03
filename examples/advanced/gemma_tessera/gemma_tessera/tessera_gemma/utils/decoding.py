
import torch
from typing import List, Optional

def greedy_generate_paged(model, input_ids: torch.LongTensor, max_new_tokens: int = 32):
    """
    Greedy decode using paged KV-cache. Returns new tokens (B, new_T).
    Assumes model implements forward(..., kv_caches, use_cache=True).
    """
    device = input_ids.device
    B = input_ids.size(0)
    new_tokens = []
    kv_caches = None
    x = input_ids
    for step in range(max_new_tokens):
        logits = model(x, kv_caches=kv_caches, use_cache=True, update_cache=True)
        kv_caches = kv_caches or [None] * len(model.blocks)  # caches created inside model
        last = logits[:, -1, :]  # (B,V)
        next_tok = last.argmax(dim=-1, keepdim=True)  # greedy
        new_tokens.append(next_tok)
        x = next_tok  # feed only the new token next
    return torch.cat(new_tokens, dim=1)
