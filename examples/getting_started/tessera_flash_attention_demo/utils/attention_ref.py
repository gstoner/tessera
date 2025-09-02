
# utils/attention_ref.py
import torch
import math

def sdpa_reference(q, k, v, causal=False):
    """
    q, k, v: [B, H, S, D]
    Returns: [B, H, S, D]
    """
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,S,S]
    if causal:
        # mask future positions
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)  # [B,H,S,D]
    return out
