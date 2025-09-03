# Rotary position embeddings: interleave cos/sin to apply on q/k
import torch, math

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (B, T, H, Dh) or (T, B, H, Dh). We'll assume (B, T, H, Dh).
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot_x = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    rot_x = rot_x.flatten(-2)
    return rot_x

def precompute_rope_cache(seqlen: int, dim: int, theta: float, device, dtype):
    # dim must be even
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(seqlen, device=device, dtype=dtype)
    freqs = torch.einsum("t,d->t d", t, freqs)  # (T, Dh/2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # shape them to (1, T, 1, Dh/2)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return cos, sin
