
import torch
from torch import Tensor
from typing import Iterable, Tuple

def native_flash_attention_paged(
    q: Tensor,
    kv_pages: Iterable[Tuple[Tensor, Tensor]],
    *, causal: bool = True, dropout_p: float = 0.0,
    block_size: int = 128
) -> Tensor:
    """
    Paged variant: q is (B, Tq, H, Dh). kv_pages is iterable of (k_page, v_page),
    each (B, Tp, H, Dh). Processes K/V in pages without concatenating all K/V.
    """
    B, Tq, H, Dh = q.shape
    device = q.device
    dtype = q.dtype
    scale = 1.0 / (Dh ** 0.5)

    # If PyTorch SDPA supports key padding masks we could build a full mask, but here
    # we implement streaming online-softmax across pages.
    out = torch.zeros((B, Tq, H, Dh), device=device, dtype=dtype)
    m = torch.full((B, Tq, H), float("-inf"), device=device, dtype=torch.float32)
    l = torch.zeros((B, Tq, H), device=device, dtype=torch.float32)

    # global key index for causal masking
    k_base = 0
    for k_page, v_page in kv_pages:
        Bp, Tp, Hp, Dhp = k_page.shape
        assert (Bp, Hp, Dhp) == (B, H, Dh), "KV page shape mismatch"

        # iterate page in blocks
        for s0 in range(0, Tp, block_size):
            s1 = min(s0 + block_size, Tp)
            k_blk = k_page[:, s0:s1]   # (B, sb, H, Dh)
            v_blk = v_page[:, s0:s1]   # (B, sb, H, Dh)

            scores = torch.einsum("bthd,bshd->bhts", q, k_blk) * scale  # (B,H,Tq,sb)

            if causal:
                # mask K positions > query index
                t_idx = torch.arange(0, Tq, device=device).unsqueeze(-1)  # (Tq,1)
                s_idx = torch.arange(k_base + s0, k_base + s1, device=device).unsqueeze(0)  # (1,sb)
                mask = (s_idx > t_idx).unsqueeze(0).unsqueeze(0)  # (1,1,Tq,sb)
                scores = scores.masked_fill(mask, float("-inf"))

            scores = scores.permute(0,2,1,3)  # (B,Tq,H,sb)

            m_new = torch.maximum(m, scores.amax(dim=-1))
            p1 = torch.exp((m - m_new).to(torch.float32)) * l
            p2 = torch.exp((scores - m_new.unsqueeze(-1)).to(torch.float32)).sum(dim=-1)
            l_new = p1 + p2

            w_blk = torch.exp((scores - m_new.unsqueeze(-1)).to(torch.float32)).to(dtype)  # (B,Tq,H,sb)
            contrib = torch.einsum("bthS,bShd->bthd", w_blk, v_blk)  # (B,Tq,H,Dh)

            alpha = (p1 / torch.clamp_min(l_new, 1e-20)).to(dtype).unsqueeze(-1)
            beta  = (1.0 / torch.clamp_min(l_new, 1e-20)).to(dtype).unsqueeze(-1)
            out = out * alpha + contrib * beta

            m = m_new
            l = l_new

        k_base += Tp

    if dropout_p > 0 and out.requires_grad:
        out = torch.nn.functional.dropout(out, p=dropout_p)

    return out
