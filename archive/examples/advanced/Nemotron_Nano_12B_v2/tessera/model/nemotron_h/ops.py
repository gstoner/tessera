"""
Reference‑grade (non‑optimized) Python stubs for Nemotron‑H blocks.
Replace with Tessera Graph‑IR/Tile‑IR backed kernels in your repo.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

class MLPReLU2(nn.Module):
    def __init__(self, hidden: int, ff: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(hidden, ff, bias=bias)
        self.w2 = nn.Linear(ff, hidden, bias=bias)
    def forward(self, x):
        return self.w2(F.relu(self.w1(x)) ** 2)

class AttentionGQA(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, bias: bool = False):
        super().__init__()
        self.hidden, self.heads, self.kv_heads, self.head_dim = hidden, heads, kv_heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=bias)
    def forward(self, x, attn_mask=None, kv_cache=None):
        B, T, H = x.shape
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)  # [B, h, T, d]
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        # expand K/V to full heads (GQA)
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            att = att + attn_mask  # assume mask in logit space
        p = att.softmax(-1)
        y = p @ v  # [B, h, T, d]
        y = y.transpose(1, 2).contiguous().view(B, T, self.heads * self.head_dim)
        return self.o_proj(y)

class Mamba2MixerStub(nn.Module):
    """
    Minimal, readable stub of the Nemotron‑H Mamba2 block:
      - input projection -> splits into [intermediate, conv_dim, heads]
      - depthwise causal conv (emulated) on [conv_dim]
      - selective state update (emulated chunk scan)
      - output projection
    Replace with Tessera Tile‑IR kernels.
    """
    def __init__(self, hidden: int, m_heads: int, m_head_dim: int, ssm_state: int, conv_kernel: int = 4, chunk_size: int = 128):
        super().__init__()
        self.hidden = hidden
        self.m_heads = m_heads
        self.m_head_dim = m_head_dim
        self.intermediate = m_heads * m_head_dim
        self.conv_kernel = conv_kernel
        self.chunk = chunk_size
        # simple projections (biasless per config)
        self.in_proj = nn.Linear(hidden, self.intermediate + (self.intermediate + 2 * ssm_state * 8) + m_heads, bias=False)
        self.out_proj = nn.Linear(self.intermediate, hidden, bias=False)
        self.norm = RMSNorm(hidden, 1e-5)
    def forward(self, x, attn_mask=None, state=None):
        # This is a reference path; real path should stream state and run chunked kernels
        h = self.norm(x)
        p = self.in_proj(h)
        inter = p[..., : self.intermediate]
        # fake "selective state update" and conv — replace with proper kernels
        inter = torch.nn.functional.silu(inter)
        return x + self.out_proj(inter)
