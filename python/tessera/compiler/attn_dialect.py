"""
attn_dialect.py — Python bindings for the Tessera Attention (Attn) dialect.

Provides a high-level Python API for constructing flash-attention IR nodes
that map to the ops defined in the tile_opt_fa4 Attn dialect:

    tessera.attn.scaled_dot_product  (QK^T / sqrt(d_k), online softmax)
    tessera.attn.lse_save            (log-sum-exp accumulator save)
    tessera.attn.online_softmax      (online numerically-stable softmax)

Each builder function returns an ``AttnNode`` describing the op and its
attributes.  AttnNode instances can be serialised to MLIR text or passed to
the compilation pipeline as part of a ``FlashAttnGraph``.

Usage::

    from tessera.compiler.attn_dialect import FlashAttnBuilder

    builder = FlashAttnBuilder(causal=True, head_dim=64)
    graph = builder.build(batch=2, heads=8, seq_len=512)
    print(graph.to_mlir())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_VALID_DTYPES = {"bf16", "fp16", "fp32", "fp8", "bf8"}

_DTYPE_BITS: dict[str, int] = {
    "fp8": 8, "bf8": 8, "fp16": 16, "bf16": 16, "fp32": 32,
}


def _bpe(dtype: str) -> int:
    """Bytes per element."""
    return _DTYPE_BITS.get(dtype, 16) // 8


# ---------------------------------------------------------------------------
# Attention op descriptors
# ---------------------------------------------------------------------------

@dataclass
class ScaledDotProductNode:
    """
    tessera.attn.scaled_dot_product

    Computes Q·K^T / sqrt(d_k) (optionally causal-masked).

    Attributes
    ----------
    q_shape, k_shape : (B, H, S, D)
    causal           : apply causal mask
    scale            : softmax scale (default 1/sqrt(d_k))
    dropout_p        : dropout probability applied after softmax (0 = off)
    dtype            : compute dtype ("bf16" | "fp16" | "fp32")
    accum_dtype      : accumulator dtype (usually "fp32")
    """
    q_shape: Tuple[int, int, int, int]
    k_shape: Tuple[int, int, int, int]
    causal: bool = True
    scale: Optional[float] = None
    dropout_p: float = 0.0
    dtype: str = "bf16"
    accum_dtype: str = "fp32"

    def __post_init__(self) -> None:
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"dtype {self.dtype!r} not in {_VALID_DTYPES}")
        if not (0.0 <= self.dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0, 1); got {self.dropout_p}")
        if self.q_shape[3] != self.k_shape[3]:
            raise ValueError(
                f"Q head_dim ({self.q_shape[3]}) must equal K head_dim ({self.k_shape[3]})"
            )
        if self.scale is None:
            self.scale = 1.0 / math.sqrt(self.q_shape[3])

    @property
    def score_shape(self) -> Tuple[int, int, int, int]:
        """Shape of the attention score matrix (B, H, S_q, S_k)."""
        B, H, Sq, _ = self.q_shape
        _, _, Sk, _ = self.k_shape
        return (B, H, Sq, Sk)

    def to_mlir(self, q_val: str = "%q", k_val: str = "%k") -> str:
        B, H, Sq, D = self.q_shape
        _, _, Sk, _ = self.k_shape
        causal = "true" if self.causal else "false"
        lines = [
            f'%scores = "tessera.attn.scaled_dot_product"({q_val}, {k_val}) {{',
            f'  causal = {causal},',
            f'  scale = {self.scale:.6f} : f32,',
            f'  dropout_p = {self.dropout_p:.4f} : f32,',
            f'  d_k = {D} : i64',
            f'}} : (tensor<{B}x{H}x{Sq}x{D}x{self.dtype}>,',
            f'      tensor<{B}x{H}x{Sk}x{D}x{self.dtype}>)',
            f'  -> tensor<{B}x{H}x{Sq}x{Sk}x{self.accum_dtype}>',
        ]
        return "\n".join(lines)

    def flops(self) -> int:
        """Approximate FLOPs for QK^T."""
        B, H, Sq, D = self.q_shape
        _, _, Sk, _ = self.k_shape
        flops = 2 * B * H * Sq * Sk * D
        if self.causal:
            flops //= 2
        return flops


@dataclass
class LSESaveNode:
    """
    tessera.attn.lse_save

    Saves the log-sum-exp normaliser for the backward pass.

    Attributes
    ----------
    batch, heads, seq_len : dimensions
    dtype                 : dtype of the LSE tensor (usually fp32)
    """
    batch: int
    heads: int
    seq_len: int
    dtype: str = "fp32"

    @property
    def lse_shape(self) -> Tuple[int, int, int]:
        return (self.batch, self.heads, self.seq_len)

    def to_mlir(self, scores_val: str = "%scores") -> str:
        B, H, S = self.lse_shape
        lines = [
            f'%lse = "tessera.attn.lse_save"({scores_val}) {{',
            f'  lse_shape = [{B}, {H}, {S}]',
            f'}} : (tensor<{B}x{H}x{S}x?x{self.dtype}>) -> tensor<{B}x{H}x{S}x{self.dtype}>',
        ]
        return "\n".join(lines)


@dataclass
class OnlineSoftmaxNode:
    """
    tessera.attn.online_softmax

    Numerically-stable online softmax over the score dimension.

    Uses the FA-2 running-max recurrence:
        m_new = max(m_old, row_max(scores))
        l_new = exp(m_old - m_new) * l_old + sum(exp(scores - m_new))

    Attributes
    ----------
    score_shape : (B, H, S_q, S_k)
    dtype       : compute dtype
    save_lse    : whether to return the LSE alongside the result
    """
    score_shape: Tuple[int, int, int, int]
    dtype: str = "fp32"
    save_lse: bool = True

    def to_mlir(self, score_val: str = "%scores") -> str:
        B, H, Sq, Sk = self.score_shape
        save = "true" if self.save_lse else "false"
        lines = [
            f'%softmax, %lse = "tessera.attn.online_softmax"({score_val}) {{',
            f'  save_lse = {save},',
            f'  block_size = {min(Sk, 128)} : i64',
            f'}} : (tensor<{B}x{H}x{Sq}x{Sk}x{self.dtype}>)',
            f'  -> (tensor<{B}x{H}x{Sq}x{Sk}x{self.dtype}>,',
            f'      tensor<{B}x{H}x{Sq}x{self.dtype}>)',
        ]
        return "\n".join(lines)

    def flops(self) -> int:
        """Approximate FLOPs for softmax (exp + normalisation per row)."""
        B, H, Sq, Sk = self.score_shape
        return B * H * Sq * Sk * 3   # subtract, exp, divide


# ---------------------------------------------------------------------------
# Flash-attention graph
# ---------------------------------------------------------------------------

@dataclass
class FlashAttnGraph:
    """
    A sequence of Attn dialect ops forming a complete flash-attention kernel.

    Wraps:
      ScaledDotProductNode → OnlineSoftmaxNode → (optionally LSESaveNode)
    """
    scaled_dot: ScaledDotProductNode
    softmax: OnlineSoftmaxNode
    lse_save: Optional[LSESaveNode] = None
    v_shape: Optional[Tuple[int, int, int, int]] = None

    def to_mlir(self) -> str:
        """Emit MLIR text for the full attention graph."""
        lines = ["// tessera.attn Flash-Attention kernel"]
        lines.append(self.scaled_dot.to_mlir())
        lines.append(self.softmax.to_mlir())
        if self.lse_save:
            lines.append(self.lse_save.to_mlir())

        # Final matmul: softmax_output · V → attention output
        if self.v_shape:
            B, H, Sq, _ = self.scaled_dot.q_shape
            _, _, Sk, Dv = self.v_shape
            dt = self.scaled_dot.dtype
            lines.append(
                f'%out = "tessera.attn.attend_v"(%softmax, %v) : '
                f'(tensor<{B}x{H}x{Sq}x{Sk}xf32>, '
                f'tensor<{B}x{H}x{Sk}x{Dv}x{dt}>) '
                f'-> tensor<{B}x{H}x{Sq}x{Dv}x{dt}>'
            )
        return "\n".join(lines)

    def total_flops(self) -> int:
        return self.scaled_dot.flops() + self.softmax.flops()

    def bytes_accessed(self) -> int:
        """Approximate HBM bytes for Q, K, V, O."""
        B, H, Sq, D = self.scaled_dot.q_shape
        _, _, Sk, _ = self.scaled_dot.k_shape
        bpe = _bpe(self.scaled_dot.dtype)
        Dv = self.v_shape[3] if self.v_shape else D
        q_bytes = B * H * Sq * D * bpe
        k_bytes = B * H * Sk * D * bpe
        v_bytes = B * H * Sk * Dv * bpe
        o_bytes = B * H * Sq * Dv * bpe
        return q_bytes + k_bytes + v_bytes + o_bytes

    def roofline_bound(self, peak_tflops: float, peak_bw_gbps: float) -> str:
        compute_ms = self.total_flops() / (peak_tflops * 1e12) * 1e3
        memory_ms  = self.bytes_accessed() / (peak_bw_gbps * 1e9) * 1e3
        return "compute" if compute_ms >= memory_ms else "memory"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class FlashAttnBuilder:
    """
    High-level builder for FlashAttnGraph objects.

    Parameters
    ----------
    causal     : use causal masking
    head_dim   : model head dimension D (default 64)
    dtype      : compute dtype (default "bf16")
    save_lse   : emit tessera.attn.lse_save for backward (default True)
    dropout_p  : attention dropout (default 0.0)
    """

    def __init__(
        self,
        causal: bool = True,
        head_dim: int = 64,
        dtype: str = "bf16",
        save_lse: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        if dtype not in _VALID_DTYPES:
            raise ValueError(f"dtype {dtype!r} not in {_VALID_DTYPES}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        self.causal = causal
        self.head_dim = head_dim
        self.dtype = dtype
        self.save_lse = save_lse
        self.dropout_p = dropout_p

    def build(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        kv_seq_len: Optional[int] = None,
    ) -> FlashAttnGraph:
        """
        Construct a FlashAttnGraph for the given problem dimensions.

        Parameters
        ----------
        batch      : batch size B
        heads      : number of attention heads H
        seq_len    : query sequence length S_q
        kv_seq_len : K/V sequence length S_k (default = seq_len)
        """
        if kv_seq_len is None:
            kv_seq_len = seq_len

        q_shape = (batch, heads, seq_len, self.head_dim)
        k_shape = (batch, heads, kv_seq_len, self.head_dim)
        v_shape = (batch, heads, kv_seq_len, self.head_dim)

        scaled_dot = ScaledDotProductNode(
            q_shape=q_shape,
            k_shape=k_shape,
            causal=self.causal,
            dropout_p=self.dropout_p,
            dtype=self.dtype,
        )
        score_shape = scaled_dot.score_shape
        softmax = OnlineSoftmaxNode(
            score_shape=score_shape,
            dtype="fp32",
            save_lse=self.save_lse,
        )
        lse = LSESaveNode(
            batch=batch, heads=heads, seq_len=seq_len
        ) if self.save_lse else None

        return FlashAttnGraph(
            scaled_dot=scaled_dot,
            softmax=softmax,
            lse_save=lse,
            v_shape=v_shape,
        )

    def mfu(self, graph: FlashAttnGraph, peak_tflops: float,
            latency_ms: float) -> float:
        """Model FLOPs utilisation (0–1)."""
        achieved = graph.total_flops() / (latency_ms * 1e-3) / 1e12
        return min(achieved / peak_tflops, 1.0)
