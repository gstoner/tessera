
import torch
from torch import Tensor
from typing import Optional, Tuple

def _repeat_kv(k: Tensor, v: Tensor, num_heads: int, num_kv_heads: int) -> Tuple[Tensor, Tensor]:
    # Expand KV heads to match Q heads for GQA/MQA.
    # Shapes: k,v: (B, T, Hkv, Dh)  ->  (B, T, Hq, Dh)
    if num_kv_heads == num_heads:
        return k, v
    reps = num_heads // num_kv_heads
    return k.repeat_interleave(reps, dim=2), v.repeat_interleave(reps, dim=2)

def native_flash_attention(
    q: Tensor, k: Tensor, v: Tensor,
    *, causal: bool = True, dropout_p: float = 0.0,
    block_size: int = 128
) -> Tensor:
    """
    First Tessera-native attention kernel (PyTorch-based implementation).
    - Uses PyTorch memory-efficient SDP attention if available; otherwise a tile-blocked online-softmax fallback.
    - Expects shapes: q,k,v => (B, T, H, Dh)
    Returns: (B, T, H, Dh)
    """
    B, T, H, Dh = q.shape
    scale = 1.0 / (Dh ** 0.5)

    # Prefer PyTorch memory-efficient SDPA when present
    try:
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2),  # (B,H,T,D)
            k.transpose(1,2),
            v.transpose(1,2),
            attn_mask=None,
            dropout_p=dropout_p if q.requires_grad else 0.0,
            is_causal=causal,
            scale=None
        ).transpose(1,2)
        return out
    except Exception:
        pass

    device = q.device
    dtype = q.dtype
    out = torch.zeros((B, T, H, Dh), device=device, dtype=dtype)

    # Tile-blocked streaming softmax
    for t0 in range(0, T, block_size):
        t1 = min(t0 + block_size, T)
        q_blk = q[:, t0:t1]  # (B, tb, H, Dh)
        tb = t1 - t0
        m = torch.full((B, tb, H), float("-inf"), device=device, dtype=torch.float32)
        l = torch.zeros((B, tb, H), device=device, dtype=torch.float32)
        o = torch.zeros((B, tb, H, Dh), device=device, dtype=dtype)

        for s0 in range(0, T, block_size):
            s1 = min(s0 + block_size, T)
            k_blk = k[:, s0:s1]  # (B, sb, H, Dh)
            v_blk = v[:, s0:s1]  # (B, sb, H, Dh)

            if causal:
                scores = torch.einsum("bthd,bshd->bhts", q_blk, k_blk) * scale  # (B,H,tb,sb)
                t_idx = torch.arange(t0, t1, device=device).unsqueeze(-1)  # (tb,1)
                s_idx = torch.arange(s0, s1, device=device).unsqueeze(0)  # (1,sb)
                mask = (s_idx > t_idx).unsqueeze(0).unsqueeze(0)  # (1,1,tb,sb)
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                scores = torch.einsum("bthd,bshd->bhts", q_blk, k_blk) * scale

            scores = scores.permute(0, 2, 1, 3)  # (B,tb,H,sb)

            m_new = torch.maximum(m, scores.amax(dim=-1))
            p1 = torch.exp((m - m_new).to(torch.float32)) * l
            p2 = torch.exp((scores - m_new.unsqueeze(-1)).to(torch.float32)).sum(dim=-1)
            l_new = p1 + p2

            w_blk = torch.exp((scores - m_new.unsqueeze(-1)).to(torch.float32)).to(dtype)  # (B,tb,H,sb)
            contrib = torch.einsum("bthS,bShd->bthd", w_blk, v_blk)  # (B,tb,H,Dh)

            alpha = (p1 / torch.clamp_min(l_new, 1e-20)).to(dtype).unsqueeze(-1)
            beta  = (1.0 / torch.clamp_min(l_new, 1e-20)).to(dtype).unsqueeze(-1)
            o = o * alpha + contrib * beta

            m = m_new
            l = l_new

        out[:, t0:t1] = o

    if dropout_p > 0 and out.requires_grad:
        out = torch.nn.functional.dropout(out, p=dropout_p)

    return out
