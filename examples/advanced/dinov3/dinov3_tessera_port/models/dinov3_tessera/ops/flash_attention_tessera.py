import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TESSERA_USE_CUSTOM = os.getenv("TESSERA_USE_CUSTOM_KERNELS", "0") == "1"
TESSERA_REFERENCE = os.getenv("TESSERA_REFERENCE_KERNELS", "1") == "1"
TESSERA_NAIVE_ATTENTION = os.getenv("TESSERA_NAIVE_ATTENTION", "0") == "1"
TESSERA_USE_QKV_PACK = os.getenv("TESSERA_USE_QKV_PACK", "0") == "1"

try:
    if TESSERA_USE_CUSTOM:
        from tessera_kernels import (
            flash_attn_forward as _flash_attn_forward,
            flash_attn_forward_ex as _flash_attn_forward_ex,
            flash_attn_backward as _flash_attn_backward,
            flash_attn_backward_ex as _flash_attn_backward_ex,
            layer_norm as _layer_norm_impl,
            layer_norm_bw as _layer_norm_bw_impl,
            tile_linear as _tile_linear_impl,
            tile_linear_wmma as _tile_linear_wmma_impl,
            tile_linear_wmma_bf16 as _tile_linear_wmma_bf16_impl,
            linear_bw_input as _linear_bw_input_impl,
            linear_bw_weight as _linear_bw_weight_impl,
            rowwise_softmax as _rowwise_softmax_impl,
            batched_gemm as _batched_gemm_impl,
            batched_gemm_wmma as _batched_gemm_wmma_impl,
            qkv_bias_gelu as _qkv_bias_gelu_impl,
            qkv_pack_gemm as _qkv_pack_gemm_impl,
            qkv_pack_gemm_fwd as _qkv_pack_gemm_fwd_impl,
            qkv_pack_gemm_bw as _qkv_pack_gemm_bw_impl,
            linear_bw_input_wmma_bf16 as _linear_bw_input_wmma_bf16_impl,
            linear_bw_weight_wmma_bf16 as _linear_bw_weight_wmma_bf16_impl,
        )
    else:
        _flash_attn_forward = None
        _flash_attn_forward_ex = None
        _flash_attn_backward = None
        _flash_attn_backward_ex = None
        _layer_norm_impl = None
        _layer_norm_bw_impl = None
        _tile_linear_impl = None
        _tile_linear_wmma_impl = None
        _tile_linear_wmma_bf16_impl = None
        _linear_bw_input_impl = None
        _linear_bw_weight_impl = None
        _rowwise_softmax_impl = None
        _batched_gemm_impl = None
        _batched_gemm_wmma_impl = None
        _qkv_bias_gelu_impl = None
        _qkv_pack_gemm_impl = None
        _qkv_pack_gemm_fwd_impl = None
        _qkv_pack_gemm_bw_impl = None
except Exception:
    _flash_attn_forward = None
    _flash_attn_forward_ex = None
    _flash_attn_backward = None
    _flash_attn_backward_ex = None
    _layer_norm_impl = None
    _layer_norm_bw_impl = None
    _tile_linear_impl = None
    _tile_linear_wmma_impl = None
    _tile_linear_wmma_bf16_impl = None
    _linear_bw_input_impl = None
    _linear_bw_weight_impl = None
    _rowwise_softmax_impl = None
    _batched_gemm_impl = None
    _batched_gemm_wmma_impl = None
    _qkv_bias_gelu_impl = None
    _qkv_pack_gemm_impl = None
    _qkv_pack_gemm_fwd_impl = None
    _qkv_pack_gemm_bw_impl = None


@dataclass
class DummySchedule:
    block_m: int = 128
    block_n: int = 128
    block_k: int = 64
    stages: int = 2
    num_warps: int = 8
    smem_bytes: int = 128 * 1024


def _io16_compute32(t: torch.Tensor):
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.float(), t.dtype
    return t, None

def _restore_dtype(t: torch.Tensor, dt):
    return t.to(dt) if dt is not None else t

# ---------------- TileLinear (autograd) with WMMA/WMMA-BF16 fast paths ----------------
class _TileLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, activation: Optional[str]):
        # Try WMMA BF16/FP16 first when eligible
        if _tile_linear_wmma_bf16_impl is not None and x.is_cuda and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
            if (x.size(1)%16==0) and (x.size(2)%16==0) and (weight.size(0)%16==0):
                return _tile_linear_wmma_bf16_impl(x, weight, bias, activation)
        if _tile_linear_wmma_impl is not None and x.is_cuda and x.dtype == torch.float16 and weight.dtype == torch.float16:
            if (x.size(1)%16==0) and (x.size(2)%16==0) and (weight.size(0)%16==0):
                return _tile_linear_wmma_impl(x, weight, bias, activation)

        # Fallback to FP32 custom or reference
        x_c, x_dt = _io16_compute32(x)
        w_c, _ = _io16_compute32(weight)
        if _tile_linear_impl is not None and x_c.is_cuda:
            y = _tile_linear_impl(x_c, w_c, bias, activation)
        else:
            y = torch.matmul(x_c, w_c.t())
            if bias is not None: y = y + bias
            if activation == "gelu": y = F.gelu(y)
            elif activation == "relu": y = F.relu(y)
        ctx.save_for_backward(x_c, w_c, bias if bias is not None else torch.tensor([]))
        ctx.activation = activation
        return _restore_dtype(y, x_dt)

    @staticmethod
    
def backward(ctx, grad_out):
        x_c, w_c, bias = ctx.saved_tensors
        go_c, go_dt = _io16_compute32(grad_out)
        B, N, M = go_c.shape
        go2 = go_c.reshape(B*N, M).contiguous()
        x2 = x_c.reshape(B*N, x_c.size(-1)).contiguous()
        # BF16 WMMA fast path (true tensor-core grads) when eligible
        if (_linear_bw_input_wmma_bf16_impl is not None and _linear_bw_weight_wmma_bf16_impl is not None and
            grad_out.dtype == torch.bfloat16 and w_c.dtype == torch.bfloat16 and
            (go2.size(0) % 16 == 0) and (go2.size(1) % 16 == 0) and (x2.size(1) % 16 == 0)):
            dX2 = _linear_bw_input_wmma_bf16_impl(go2.to(torch.bfloat16), w_c.to(torch.bfloat16)).to(go2.dtype)
            dW = _linear_bw_weight_wmma_bf16_impl(go2.to(torch.bfloat16), x2.to(torch.bfloat16)).to(go2.dtype)
        else:
            if _linear_bw_input_impl is not None and go2.is_cuda:
                dX2 = _linear_bw_input_impl(go2, w_c)
            else:
                dX2 = go2 @ w_c
            if _linear_bw_weight_impl is not None and go2.is_cuda:
                dW = _linear_bw_weight_impl(go2, x2)
            else:
                dW = go2.t() @ x2
        dX = dX2.reshape_as(x_c)
        dB = go2.sum(dim=0) if bias.numel() > 0 else None
        return _restore_dtype(dX, go_dt), _restore_dtype(dW, go_dt), _restore_dtype(dB, go_dt), None
class TileLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, activation: Optional[str]=None, schedule: Optional[DummySchedule]=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32)) if bias else None
        self.activation = activation
        self.schedule = schedule or DummySchedule()
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = (1 / in_features) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _TileLinearFn.apply(x, self.weight, self.bias, self.activation)


# ---------------- Batched GEMM dispatch (WMMA when possible) ----------------
def _batched_gemm_dispatch(A: torch.Tensor, B: torch.Tensor, trans_b: bool):
    if _batched_gemm_wmma_impl is not None and trans_b and A.is_cuda and B.is_cuda and \
       A.dtype in (torch.float16, torch.bfloat16) and B.dtype in (torch.float16, torch.bfloat16) and \
       (A.size(1)%16==0) and (A.size(2)%16==0) and (B.size(1)%16==0) and (B.size(2)%16==0):
        return _batched_gemm_wmma_impl(A, B)
    if _batched_gemm_impl is not None and A.is_cuda:
        return _batched_gemm_impl(A, B, trans_b)
    # reference
    return A @ (B.transpose(1,2) if trans_b else B)


# ---------------- LayerNorm ----------------
class _TesseraLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps: float, weight=None, bias=None):
        x_c, x_dt = _io16_compute32(x)
        w_c = weight.float() if weight is not None else None
        b_c = bias.float() if bias is not None else None
        if _layer_norm_impl is not None and x_c.is_cuda:
            y = _layer_norm_impl(x_c, eps, w_c, b_c)
        else:
            y = F.layer_norm(x_c, (x_c.shape[-1],), weight=w_c, bias=b_c, eps=eps)
        ctx.save_for_backward(x_c, w_c if w_c is not None else torch.tensor([]), b_c if b_c is not None else torch.tensor([]))
        ctx.eps = eps
        return _restore_dtype(y, x_dt)

    @staticmethod
    def backward(ctx, grad_out):
        x_c, w_c, b_c = ctx.saved_tensors
        go_c, go_dt = _io16_compute32(grad_out)
        if _layer_norm_bw_impl is not None and x_c.is_cuda:
            dx, _, _ = _layer_norm_bw_impl(x_c, go_c, w_c if w_c.numel() else None, ctx.eps)
        else:
            dx = torch.autograd.grad(
                outputs=F.layer_norm(x_c, (x_c.shape[-1],), weight=w_c if w_c.numel() else None, bias=b_c if b_c.numel() else None, eps=ctx.eps),
                inputs=x_c,
                grad_outputs=go_c,
                retain_graph=True,
                allow_unused=True
            )[0]
        dw = (go_c * (x_c - x_c.mean(dim=-1, keepdim=True)) * (1.0/torch.sqrt((x_c.var(dim=-1, unbiased=False, keepdim=True) + ctx.eps)))).sum(dim=tuple(range(x_c.dim()-1))) if w_c.numel() else None
        db = go_c.sum(dim=tuple(range(x_c.dim()-1))) if b_c.numel() else None
        return _restore_dtype(dx, go_dt), None, _restore_dtype(dw, go_dt) if dw is not None else None, _restore_dtype(db, go_dt) if db is not None else None

def tessera_layer_norm(x: torch.Tensor, eps: float=1e-6, weight: torch.Tensor=None, bias: torch.Tensor=None) -> torch.Tensor:
    return _TesseraLayerNormFn.apply(x, eps, weight, bias)


# ---------------- Attention with dropout/causal forward+backward ----------------
def _reference_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float, schedule: DummySchedule) -> torch.Tensor:
    B, H, N, D = q.shape
    scale = 1.0 / (D ** 0.5)
    m_i = torch.full((B, H, N), float('-inf'), device=q.device, dtype=q.dtype)
    l_i = torch.zeros((B, H, N), device=q.device, dtype=q.dtype)
    out = torch.zeros((B, H, N, D), device=q.device, dtype=q.dtype)
    BN = schedule.block_n
    for n0 in range(0, N, BN):
        n1 = min(n0 + BN, N)
        k_blk = k[:, :, n0:n1, :]
        v_blk = v[:, :, n0:n1, :]
        q_bh = q.reshape(B * H, N, D)
        k_bh = k_blk.reshape(B * H, n1 - n0, D)
        scores = torch.bmm(q_bh, k_bh.transpose(1, 2)).reshape(B, H, N, n1 - n0) * scale
        blk_max, _ = scores.max(dim=-1)
        new_m = torch.maximum(m_i, blk_max)
        exp_m_diff = torch.exp(m_i - new_m)
        l_i = l_i * exp_m_diff
        scores = scores - new_m.unsqueeze(-1)
        p = torch.exp(scores)
        l_i = l_i + p.sum(dim=-1)
        p_bh = p.reshape(B * H, N, n1 - n0)
        v_bh = v_blk.reshape(B * H, n1 - n0, D)
        contrib = torch.bmm(p_bh, v_bh).reshape(B, H, N, D)
        out = out * exp_m_diff.unsqueeze(-1) + contrib
        m_i = new_m
    out = out / l_i.unsqueeze(-1)
    return out

class _TesseraFlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p: float, schedule: DummySchedule, causal: bool=False, seed: int=0):
        q_c, q_dt = _io16_compute32(q); k_c, _ = _io16_compute32(k); v_c, _ = _io16_compute32(v)
        if _flash_attn_forward_ex is not None and q_c.is_cuda and not TESSERA_NAIVE_ATTENTION and not TESSERA_REFERENCE:
            out = _flash_attn_forward_ex(q_c, k_c, v_c, float(dropout_p), bool(causal), int(seed))
        elif TESSERA_NAIVE_ATTENTION:
            out = _naive_attention(q_c, k_c, v_c, schedule)
        else:
            out = _reference_flash_attention(q_c, k_c, v_c, dropout_p, schedule)
        ctx.save_for_backward(q_c, k_c, v_c)
        ctx.causal = causal; ctx.dropout_p = dropout_p; ctx.seed = seed
        return _restore_dtype(out, q_dt)

    @staticmethod
    def backward(ctx, grad_out):
        q_c, k_c, v_c = ctx.saved_tensors
        go_c, go_dt = _io16_compute32(grad_out)
        if _flash_attn_backward_ex is not None and q_c.is_cuda:
            dQ, dK, dV = _flash_attn_backward_ex(q_c, k_c, v_c, go_c, ctx.causal, float(ctx.dropout_p), int(ctx.seed))
        elif _flash_attn_backward is not None and q_c.is_cuda:
            dQ, dK, dV = _flash_attn_backward(q_c, k_c, v_c, go_c, ctx.causal, None)
        else:
            q_c.requires_grad_(True); k_c.requires_grad_(True); v_c.requires_grad_(True)
            out = _reference_flash_attention(q_c, k_c, v_c, 0.0, DummySchedule())
            out.backward(go_c)
            dQ, dK, dV = q_c.grad, k_c.grad, v_c.grad
        return _restore_dtype(dQ, go_dt), _restore_dtype(dK, go_dt), _restore_dtype(dV, go_dt), None, None, None, None

def tessera_flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float=0.0, schedule: Optional[DummySchedule]=None, causal: bool=False, seed: int=0) -> torch.Tensor:
    schedule = schedule or DummySchedule()
    return _TesseraFlashAttnFn.apply(q, k, v, dropout_p, schedule, causal, seed)

# ---------------- Naive attention path ----------------
def _naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, schedule: DummySchedule) -> torch.Tensor:
    B, H, N, D = q.shape
    G = B * H
    qg = q.reshape(G, N, D); kg = k.reshape(G, N, D); vg = v.reshape(G, N, D)
    scores = _batched_gemm_dispatch(qg, kg, True)
    scores = scores * (1.0 / (D ** 0.5))
    scores2d = scores.reshape(G * N, N)
    if _rowwise_softmax_impl is not None and scores2d.is_cuda:
        probs = _rowwise_softmax_impl(scores2d)
    else:
        probs = F.softmax(scores2d, dim=-1)
    probs = probs.reshape(G, N, N)
    out = _batched_gemm_dispatch(probs, vg, False)
    return out.reshape(B, H, N, D)

# ---------------- Fused QKV: autograd pack (single-GEMM) ----------------
class _FusedQKVPackFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Wcat, bcat, gelu_q: bool):
        x32, _ = _io16_compute32(x)
        if _qkv_pack_gemm_fwd_impl is not None and x32.is_cuda and Wcat.is_cuda and bcat.is_cuda:
            packed, yq_pre = _qkv_pack_gemm_fwd_impl(x32, Wcat.float(), bcat.float(), bool(gelu_q))
        else:
            # reference fallback
            M3 = bcat.numel(); D = M3 // 3
            y = x32 @ Wcat.float().t() + bcat.float()
            yq_pre = y[:, :, :D].contiguous()
            if gelu_q: y[:, :, :D] = F.gelu(yq_pre)
            packed = torch.stack([y[:, :, :D], y[:, :, D:2*D], y[:, :, 2*D:]], dim=0).reshape(3, x.size(0), x.size(1), D)
        ctx.save_for_backward(x32, Wcat.float(), yq_pre)
        ctx.gelu_q = gelu_q
        return packed

    @staticmethod
    def backward(ctx, grad_packed):
        x32, Wcat, yq_pre = ctx.saved_tensors
        gelu_q = ctx.gelu_q
        if _qkv_pack_gemm_bw_impl is not None and x32.is_cuda and Wcat.is_cuda:
            dX, dW, db = _qkv_pack_gemm_bw_impl(x32, Wcat, yq_pre, grad_packed.float(), bool(gelu_q))
        else:
            # reference
            B,N,K = x32.shape
            M3 = Wcat.size(0); D = M3//3
            dYcat = grad_packed.reshape(3, B, N, D).permute(1,2,0,3).reshape(B*N, M3).contiguous()
            if gelu_q:
                yq = yq_pre.reshape(B*N, D)
                # apply GELU' elementwise
                kAlpha = (2.0/3.141592653589793)**0.5
                t = torch.tanh(kAlpha*(yq + 0.044715*yq**3))
                dy = 0.5*(1+t) + 0.5*yq*(1-t**2)*kAlpha*(1+3*0.044715*yq**2)
                dYcat[:, :D] *= dy
            dW = dYcat.t() @ x32.reshape(B*N, K)
            dX = dYcat @ Wcat
            db = dYcat.sum(0)
            dX = dX.reshape(B, N, K)
        return dX, dW, db, None

def fused_qkv_pack(x: torch.Tensor, Wcat: torch.Tensor, bcat: torch.Tensor, gelu_q: bool=True) -> torch.Tensor:
    return _FusedQKVPackFn.apply(x, Wcat, bcat, gelu_q)
