# GPT-OSS-120B Tessera Port - Part 1: Core Model Architecture

## Overview

This document provides a complete port of OpenAI's GPT-OSS-120B model to the Tessera programming model. The implementation leverages Tessera's tile-first abstractions, distributed tensor support, and mixed-precision numerics to achieve efficient scaling on modern GPU clusters including NVL72.

## Model Configuration

```python
from dataclasses import dataclass
import tessera as ts
from tessera import dist, Tensor, Region
from tessera.numerics import fp8_e4m3, fp6, bf16, fp32
import math

@dataclass
class GPTConfig:
    """Configuration for GPT-OSS-120B model"""
    # Model dimensions
    n_layers: int = 96
    n_heads: int = 96
    n_kv_heads: int = 8  # GQA: 8 KV heads, 96 Q heads = 12x reduction
    d_model: int = 12288
    d_head: int = 128
    d_ffn: int = 49152  # 4x d_model
    
    # Sequence and batch
    max_seq_len: int = 8192
    vocab_size: int = 50257
    
    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Numerics policies
    compute_dtype: str = "bf16 @accum(fp32)"
    weight_dtype: str = "fp8_e4m3 @accum(fp32)"
    kv_cache_dtype: str = "fp8_e5m2"
    
    # Optimizations
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True
    use_parallel_attention: bool = True  # Parallel attention + FFN
    use_gated_mlp: bool = True
    
    # Distributed configuration
    tp_size: int = 9  # Tensor parallel
    pp_size: int = 2  # Pipeline parallel
    dp_size: int = 4  # Data parallel
    ep_size: int = 1  # Expert parallel (for future MoE)
    
    # Memory optimization
    gradient_checkpointing: bool = True
    activation_checkpointing_layers: list = None
    use_kv_cache_compression: bool = True
    
    @property
    def total_params(self) -> int:
        """Calculate total parameter count"""
        # Embeddings
        params = self.vocab_size * self.d_model * 2  # token + position
        
        # Attention layers
        params += self.n_layers * (
            # QKV projection (with GQA)
            self.d_model * self.d_head * (self.n_heads + 2 * self.n_kv_heads) +
            # Output projection
            self.n_heads * self.d_head * self.d_model +
            # Layer norms
            2 * self.d_model
        )
        
        # FFN layers
        if self.use_gated_mlp:
            params += self.n_layers * (
                3 * self.d_model * self.d_ffn +  # W_gate, W_up, W_down
                self.d_model  # LayerNorm
            )
        else:
            params += self.n_layers * (
                2 * self.d_model * self.d_ffn +  # W_in, W_out
                self.d_model  # LayerNorm
            )
        
        # Final layer norm
        params += self.d_model
        
        return params

# Default configuration for 120B model
CONFIG_120B = GPTConfig(
    n_layers=96,
    n_heads=96,
    n_kv_heads=8,
    d_model=12288,
    d_ffn=49152,
    max_seq_len=8192,
    vocab_size=50257
)
```

## Distributed Mesh Setup

```python
def create_model_mesh(config: GPTConfig, devices: list = None):
    """Create distributed mesh for model parallelism"""
    
    if devices is None:
        # Default to available GPUs
        import torch
        n_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(n_gpus)]
    
    total_devices = len(devices)
    expected = config.dp_size * config.tp_size * config.pp_size
    
    if total_devices != expected:
        raise ValueError(f"Device count {total_devices} != dp*tp*pp = {expected}")
    
    # Create 3D mesh for NVL72 or other configurations
    mesh = dist.mesh(
        devices=devices,
        axes=("dp", "tp", "pp"),
        shape=(config.dp_size, config.tp_size, config.pp_size)
    )
    
    return mesh

# Example: NVL72 setup (72 GPUs)
def create_nvl72_mesh():
    """Optimized mesh for NVIDIA NVL72 (72 GPU) system"""
    return dist.mesh(
        devices=[f"cuda:{i}" for i in range(72)],
        axes=("dp", "tp", "pp"),
        shape=(4, 9, 2)  # 4x data parallel, 9x tensor parallel, 2x pipeline
    )
```

## Core Attention Module

```python
@ts.jit
class MultiQueryAttention(ts.Module):
    """Multi-Query Attention with Grouped Query Attention (GQA) support"""
    
    def __init__(self, config: GPTConfig, layer_idx: int, mesh: dist.Mesh):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mesh = mesh
        
        # Dimensions
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        
        # GQA: each KV head is shared by n_heads // n_kv_heads query heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # Create sharded weight tensors
        self.W_q = self._create_weight(
            (config.d_model, config.n_heads * config.d_head),
            partition_spec=("col",),  # Shard columns across TP
            mesh_axes=("tp",)
        )
        
        self.W_k = self._create_weight(
            (config.d_model, config.n_kv_heads * config.d_head),
            partition_spec=("col",),
            mesh_axes=("tp",)
        )
        
        self.W_v = self._create_weight(
            (config.d_model, config.n_kv_heads * config.d_head),
            partition_spec=("col",),
            mesh_axes=("tp",)
        )
        
        self.W_o = self._create_weight(
            (config.n_heads * config.d_head, config.d_model),
            partition_spec=("row",),  # Shard rows for output
            mesh_axes=("tp",)
        )
        
        # Rotary embeddings
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                dim=config.d_head,
                max_seq_len=config.max_seq_len,
                base=10000.0
            )
    
    def _create_weight(self, shape, partition_spec, mesh_axes):
        """Create distributed weight tensor with proper dtype and sharding"""
        return dist.tensor(
            shape=shape,
            layout=dist.ShardSpec(
                partition=partition_spec,
                mesh_axes=mesh_axes
            ),
            mesh=self.mesh,
            dtype=self.config.weight_dtype,
            initializer=ts.nn.init.normal_(std=0.02)
        )
    
    @ts.jit
    @ts.autodiff
    def forward(
        self,
        x: Tensor["B", "S", "D", "bf16 @accum(fp32)"],
        kv_cache: Optional[ts.KVCache] = None,
        attention_mask: Optional[Tensor["B", "S", "S", "bf16"]] = None,
        position_ids: Optional[Tensor["B", "S", "i32"]] = None
    ):
        """
        Forward pass with GQA and optional KV caching
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            kv_cache: Optional KV cache for inference
            attention_mask: Optional attention mask
            position_ids: Position indices for rotary embeddings
        """
        B, S, D = x.shape
        
        # Compute Q, K, V projections with tensor parallelism
        q = ts.gemm(x, self.W_q)  # [B, S, n_heads * d_head]
        k = ts.gemm(x, self.W_k)  # [B, S, n_kv_heads * d_head]
        v = ts.gemm(x, self.W_v)  # [B, S, n_kv_heads * d_head]
        
        # Reshape for multi-head attention
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.config.use_rotary_embeddings:
            q, k = self.rotary_emb(q, k, position_ids)
        
        # KV cache handling for inference
        if kv_cache is not None:
            k, v = self._update_kv_cache(k, v, kv_cache, self.layer_idx)
        
        # Repeat KV heads for GQA
        if self.n_rep > 1:
            k = self._repeat_kv(k, self.n_rep)
            v = self._repeat_kv(v, self.n_rep)
        
        # Attention computation
        if self.config.use_flash_attention:
            # Use Flash Attention v3 implementation
            attn_output = self._flash_attention(q, k, v, attention_mask)
        else:
            # Standard attention
            attn_output = self._standard_attention(q, k, v, attention_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, S, self.n_heads * self.d_head)
        
        # Output projection with tensor parallelism
        output = ts.gemm(attn_output, self.W_o)
        
        return output
    
    @ts.jit
    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """Repeat KV heads for grouped query attention"""
        if n_rep == 1:
            return x
        B, n_kv_heads, S, d_head = x.shape
        x = x.unsqueeze(2)  # [B, n_kv_heads, 1, S, d_head]
        x = x.expand(B, n_kv_heads, n_rep, S, d_head)
        return x.reshape(B, n_kv_heads * n_rep, S, d_head)
    
    @ts.jit
    def _flash_attention(
        self,
        q: Tensor["B", "H", "S", "D_h"],
        k: Tensor["B", "H", "S", "D_h"],
        v: Tensor["B", "H", "S", "D_h"],
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Flash Attention v3 implementation in Tessera"""
        
        scale = 1.0 / math.sqrt(self.d_head)
        
        # Use Tessera's optimized Flash Attention primitive
        return ts.nn.flash_attention(
            q, k, v,
            causal=True,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            softmax_scale=scale,
            window_size=-1,  # Full attention
            alibi_slopes=None
        )
    
    @ts.jit 
    def _standard_attention(
        self,
        q: Tensor["B", "H", "S", "D_h"],
        k: Tensor["B", "H", "S", "D_h"],
        v: Tensor["B", "H", "S", "D_h"],
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Standard scaled dot-product attention"""
        
        B, H, S, D_h = q.shape
        scale = 1.0 / math.sqrt(D_h)
        
        # Compute attention scores
        scores = ts.gemm(q, k.transpose(-2, -1)) * scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax (numerically stable version)
        attn_weights = ts.nn.softmax_safe(scores, dim=-1)
        
        # Apply dropout if training
        if self.training and self.config.attention_dropout > 0:
            attn_weights = ts.nn.dropout(
                attn_weights,
                p=self.config.attention_dropout,
                training=self.training
            )
        
        # Compute attention output
        attn_output = ts.gemm(attn_weights, v)
        
        return attn_output
    
    def _update_kv_cache(self, k, v, kv_cache, layer_idx):
        """Update KV cache for inference"""
        # Implementation depends on KVCache structure
        cached_k, cached_v = kv_cache.get(layer_idx)
        k = ts.cat([cached_k, k], dim=2)  # Concatenate along sequence dimension
        v = ts.cat([cached_v, v], dim=2)
        kv_cache.set(layer_idx, (k, v))
        return k, v
```

## Rotary Position Embeddings

```python
@ts.jit
class RotaryEmbedding(ts.Module):
    """Rotary Position Embeddings (RoPE) for improved position encoding"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (ts.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache cos/sin for efficiency
        self._build_cache()
    
    def _build_cache(self):
        """Precompute cos and sin values for all positions"""
        seq_len = self.max_seq_len
        t = ts.arange(seq_len, dtype=ts.float32)
        freqs = ts.outer(t, self.inv_freq)
        
        # Create rotation matrices
        emb = ts.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    @ts.jit
    def forward(
        self,
        q: Tensor["B", "H", "S", "D"],
        k: Tensor["B", "H", "S", "D"],
        position_ids: Optional[Tensor["B", "S"]] = None
    ):
        """Apply rotary embeddings to query and key tensors"""
        
        B, H, S, D = q.shape
        
        if position_ids is None:
            position_ids = ts.arange(S, device=q.device)
        
        # Get cached cos/sin values
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [B, 1, S, D]
        sin = self.sin_cached[position_ids].unsqueeze(1)
        
        # Apply rotation
        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)
        
        return q_rot, k_rot
    
    @ts.jit
    def _apply_rotary(
        self,
        x: Tensor["B", "H", "S", "D"],
        cos: Tensor["B", "1", "S", "D"],
        sin: Tensor["B", "1", "S", "D"]
    ) -> Tensor:
        """Apply rotary transformation"""
        # Split into pairs
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotation
        x_rot = ts.cat([
            x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]],
            x2 * cos[..., :x2.shape[-1]] + x1 * sin[..., :x1.shape[-1]]
        ], dim=-1)
        
        return x_rot
```

This completes Part 1 of the GPT-OSS-120B port to Tessera, covering the core model architecture, configuration, distributed mesh setup, and the critical attention mechanism with GQA support. The implementation leverages Tessera's key features including:

- Mixed precision with FP8/FP6 weights and BF16 compute
- Tensor parallelism via ShardSpec
- Flash Attention v3 integration
- Efficient KV caching for inference
- Rotary position embeddings

Would you like me to continue with Part 2, which will cover the FFN layers, transformer blocks, and the complete model assembly?