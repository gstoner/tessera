import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops.flash_attention_tessera import tessera_flash_attn, tessera_layer_norm, TileLinear


@dataclass
class TileSchedule:
    block_m: int = 64
    block_n: int = 64
    block_k: int = 64
    stages: int = 2
    num_warps: int = 4
    smem_bytes: int = 96 * 1024


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, schedule: Optional[TileSchedule]=None):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = TileLinear(dim, hidden, bias=True, activation="gelu", schedule=schedule)
        self.fc2 = TileLinear(hidden, dim, bias=True, activation=None, schedule=schedule)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, schedule: Optional[TileSchedule]=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5
        self.qkv = TileLinear(dim, dim * 3, bias=qkv_bias, activation=None, schedule=schedule)
        self.proj = TileLinear(dim, dim, bias=True, activation=None, schedule=schedule)
        self.attn_drop_p = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.schedule = schedule or TileSchedule()

        # learnable affine for LN (tessera_layer_norm accepts weight/bias)
        self.ln1_w = nn.Parameter(torch.ones(dim))
        self.ln1_b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        B, N, C = x.shape
        # LN via tessera fused
        x_norm = tessera_layer_norm(x, weight=self.ln1_w, bias=self.ln1_b)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)
        out = tessera_flash_attn(q, k, v, dropout_p=self.attn_drop_p, schedule=self.schedule)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, schedule: Optional[TileSchedule]=None):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, schedule=schedule)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop, schedule=schedule)
        self.schedule = schedule or TileSchedule()

        self.ln2_w = nn.Parameter(torch.ones(dim))
        self.ln2_b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(tessera_layer_norm(x, weight=self.ln2_w, bias=self.ln2_b))
        return x


class VisionTransformerTSR(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,  # 0 = SSL backbone
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        schedule: Optional[TileSchedule]=None,
        gram_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, schedule=schedule)
            for _ in range(depth)
        ])
        self.norm_w = nn.Parameter(torch.ones(embed_dim))
        self.norm_b = nn.Parameter(torch.zeros(embed_dim))

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.gram_layers: Set[int] = set(gram_layers or [])

    def forward_features(self, x, return_layers: bool=False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[int, torch.Tensor]]]:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        layer_tokens = {} if return_layers and len(self.gram_layers) > 0 else None
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if layer_tokens is not None and i in self.gram_layers:
                # store patch tokens only, omit cls
                layer_tokens[i] = x[:, 1:, :]
        x = tessera_layer_norm(x, weight=self.norm_w, bias=self.norm_b)
        if layer_tokens is not None and (len(self.gram_layers) == 0 or max(self.gram_layers) == 0):
            layer_tokens[0] = x[:, 1:, :]
        return x[:, 0], x[:, 1:], layer_tokens

    def forward(self, x, return_tokens=False, return_layers=False):
        cls, tokens, layer_tokens = self.forward_features(x, return_layers=return_layers)
        out = self.head(cls)
        if return_tokens or return_layers:
            return out, tokens if return_tokens else None, layer_tokens
        return out

    @torch.no_grad()
    def resize_pos_embed(self, new_hw: Tuple[int, int]):
        # interpolate patch grid PE, keep cls token PE intact.
        _, N1, C = self.pos_embed.shape
        grid = N1 - 1
        h = w = int(grid ** 0.5)
        cls_pe, patch_pe = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        patch_pe = patch_pe.reshape(1, h, w, C).permute(0, 3, 1, 2)  # (1,C,h,w)
        patch_pe = F.interpolate(patch_pe, size=new_hw, mode="bicubic", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_hw[0]*new_hw[1], C)
        self.pos_embed = nn.Parameter(torch.cat([cls_pe, patch_pe], dim=1))
