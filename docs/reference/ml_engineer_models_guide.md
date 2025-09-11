## Vision Models

### Vision Transformer (ViT)

```python
@ts.function
def patch_embedding(
    images: ts.Tensor["B", "C", "H", "W", ts.bf16],
    patch_size: int,
    embed_dim: int,
    w_patch: ts.Tensor["C*patch_size*patch_size", "embed_dim", ts.bf16]
) -> ts.Tensor["B", "N", "embed_dim", ts.bf16]:
    """Convert images to patch embeddings."""
    
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0
    
    # Reshape to patches: [B, C, H, W] -> [B, N, C*P*P]
    patches = images.reshape(
        B, C, H // patch_size, patch_size, W // patch_size, patch_size
    ).permute(0, 2, 4, 1, 3, 5).reshape(
        B, (H // patch_size) * (W // patch_size), C * patch_size * patch_size
    )
    
    # Linear projection
    return linear(patches, w_patch)

@ts.function
def vit_model(
    images: ts.Tensor["B", "C", "H", "W", ts.bf16],
    weights: ts.Dict[str, ts.Any],
    config: ts.Dict[str, ts.Any]
) -> ts.Tensor["B", "num_classes", ts.bf16]:
    """Vision Transformer model."""
    
    B, C, H, W = images.shape
    patch_size = config["patch_size"]
    num_patches = (H // patch_size) * (W // patch_size)
    
    # Patch embeddings
    x = patch_embedding(images, patch_size, config["embed_dim"], weights["patch_emb"])
    
    # Add class token
    cls_token = weights["cls_token"].expand(B, 1, -1)
    x = ts.cat([cls_token, x], dim=1)
    
    # Add positional embeddings
    pos_emb = weights["pos_emb"][:, :num_patches + 1, :]
    x = x + pos_emb
    
    # Transformer blocks (same as BERT but without attention mask)
    for i in range(config["num_layers"]):
        block_weights = weights[f"layer_{i}"]
        # Reuse bert_block but without attention mask
        x = transformer_block(x, block_weights, config)
    
    # Classification head (use only CLS token)
    cls_output = x[:, 0]  # [B, embed_dim]
    return linear(cls_output, weights["head"])

### Convolutional Neural Networks

@ts.function
def conv2d_block(
    x: ts.Tensor["B", "C_in", "H", "W", ts.bf16],
    weight: ts.Tensor["C_out", "C_in", "K", "K", ts.bf16],
    bias: ts.Tensor["C_out", ts.bf16],
    stride: int = 1,
    padding: int = 0,
    use_bn: bool = True,
    activation: str = "relu"
) -> ts.Tensor["B", "C_out", "H_out", "W_out", ts.bf16]:
    """Convolution block with batch norm and activation."""
    
    # Convolution
    x = ts.nn.conv2d(x, weight, bias, stride=stride, padding=padding)
    
    # Batch normalization
    if use_bn:
        x = ts.nn.batch_norm2d(x)
    
    # Activation
    if activation == "relu":
        x = ts.nn.relu(x)
    elif activation == "gelu":
        x = ts.nn.gelu(x)
    elif activation == "swish":
        x = ts.nn.swish(x)
    
    return x

@ts.function
def resnet_block(
    x: ts.Tensor["B", "C", "H", "W", ts.bf16],
    weights: ts.Dict[str, ts.Tensor],
    stride: int = 1,
    downsample: bool = False
) -> ts.Tensor["B", "C_out", "H_out", "W_out", ts.bf16]:
    """ResNet residual block."""
    
    identity = x
    
    # First conv
    out = conv2d_block(
        x, weights["conv1"], weights["bn1"], 
        stride=stride, activation="relu"
    )
    
    # Second conv (no activation)
    out = ts.nn.conv2d(out, weights["conv2"], weights["conv2_bias"])
    out = ts.nn.batch_norm2d(out, weights["bn2"])
    
    # Downsample identity if needed
    if downsample:
        identity = ts.nn.conv2d(identity, weights["downsample_conv"])
        identity = ts.nn.batch_norm2d(identity, weights["downsample_bn"])
    
    # Add residual connection
    out = out + identity
    return ts.nn.relu(out)

@ts.function
def efficientnet_mbconv(
    x: ts.Tensor["B", "C_in", "H", "W", ts.bf16],
    weights: ts.Dict[str, ts.Tensor],
    expand_ratio: int,
    kernel_size: int,
    stride: int,
    se_ratio: float = 0.25
) -> ts.Tensor["B", "C_out", "H_out", "W_out", ts.bf16]:
    """EfficientNet MBConv block with squeeze-and-excitation."""
    
    C_in = x.shape[1]
    C_expanded = C_in * expand_ratio
    
    identity = x
    
    # Expansion phase (if expand_ratio > 1)
    if expand_ratio > 1:
        x = conv2d_block(x, weights["expand_conv"], activation="swish")
    
    # Depthwise convolution
    x = ts.nn.conv2d(x, weights["dw_conv"], stride=stride, 
                     padding=kernel_size//2, groups=C_expanded)
    x = ts.nn.batch_norm2d(x, weights["dw_bn"])
    x = ts.nn.swish(x)
    
    # Squeeze-and-excitation
    if se_ratio > 0:
        se_channels = max(1, int(C_in * se_ratio))
        se = ts.nn.adaptive_avg_pool2d(x, 1)  # Global average pooling
        se = conv2d_block(se, weights["se_reduce"], activation="swish")
        se = conv2d_block(se, weights["se_expand"], activation="sigmoid")
        x = x * se
    
    # Projection phase
    x = ts.nn.conv2d(x, weights["project_conv"])
    x = ts.nn.batch_norm2d(x, weights["project_bn"])
    
    # Skip connection (if same spatial size and channel count)
    if stride == 1 and identity.shape == x.shape:
        x = x + identity
    
    return x
```

## Diffusion Models

### U-Net Architecture for Diffusion

```python
@ts.function
def timestep_embedding(
    timesteps: ts.Tensor["B", ts.int32],
    embedding_dim: int,
    max_period: int = 10000
) -> ts.Tensor["B", "embedding_dim", ts.bf16]:
    """Sinusoidal timestep embeddings for diffusion models."""
    
    half_dim = embedding_dim // 2
    freqs = ts.exp(
        -math.log(max_period) * ts.arange(half_dim, dtype=ts.f32) / half_dim
    )
    
    args = timesteps[:, None].float() * freqs[None, :]
    embeddings = ts.cat([ts.cos(args), ts.sin(args)], dim=-1)
    
    if embedding_dim % 2 == 1:
        embeddings = ts.cat([embeddings, ts.zeros_like(embeddings[:, :1])], dim=-1)
    
    return embeddings.to(ts.bf16)

@ts.function
def attention_block(
    x: ts.Tensor["B", "C", "H", "W", ts.bf16],
    num_heads: int = 8
) -> ts.Tensor["B", "C", "H", "W", ts.bf16]:
    """Spatial attention block for U-Net."""
    
    B, C, H, W = x.shape
    
    # Reshape for attention: [B, C, H, W] -> [B, H*W, C]
    x_reshaped = x.reshape(B, C, H * W).transpose(1, 2)
    
    # Multi-head attention
    attn_out = multi_head_attention(
        x_reshaped, 
        num_heads=num_heads,
        causal=False  # Spatial attention is bidirectional
    )
    
    # Reshape back: [B, H*W, C] -> [B, C, H, W]
    return attn_out.transpose(1, 2).reshape(B, C, H, W) + x

@ts.function
def resnet_block_diffusion(
    x: ts.Tensor["B", "C_in", "H", "W", ts.bf16],
    timestep_emb: ts.Tensor["B", "emb_dim", ts.bf16],
    weights: ts.Dict[str, ts.Tensor],
    out_channels: int = None
) -> ts.Tensor["B", "C_out", "H", "W", ts.bf16]:
    """ResNet block with timestep conditioning for diffusion."""
    
    if out_channels is None:
        out_channels = x.shape[1]
    
    # First conv + group norm
    h = ts.nn.group_norm(x, num_groups=32)
    h = ts.nn.silu(h)
    h = ts.nn.conv2d(h, weights["conv1"])
    
    # Timestep embedding projection
    temb = ts.nn.silu(timestep_emb)
    temb = linear(temb, weights["time_emb_proj"])[:, :, None, None]
    h = h + temb
    
    # Second conv + group norm
    h = ts.nn.group_norm(h, num_groups=32)
    h = ts.nn.silu(h)
    h = ts.nn.conv2d(h, weights["conv2"])
    
    # Skip connection
    if x.shape[1] != out_channels:
        x = ts.nn.conv2d(x, weights["skip_conv"])
    
    return x + h

@ts.function
def unet_down_block(
    x: ts.Tensor["B", "C_in", "H", "W", ts.bf16],
    timestep_emb: ts.Tensor["B", "emb_dim", ts.bf16],
    weights: ts.Dict[str, ts.Tensor],
    out_channels: int,
    num_layers: int = 2,
    downsample: bool = True,
    use_attention: bool = False
) -> tuple[ts.Tensor, list[ts.Tensor]]:
    """U-Net downsampling block."""
    
    skip_connections = []
    
    # ResNet blocks
    for i in range(num_layers):
        x = resnet_block_diffusion(
            x, timestep_emb, weights[f"resnet_{i}"], 
            out_channels if i == 0 else None
        )
        skip_connections.append(x)
        
        # Optional attention
        if use_attention:
            x = attention_block(x, weights[f"attn_{i}"])
    
    # Downsampling
    if downsample:
        x = ts.nn.conv2d(x, weights["downsample"], stride=2)
    
    return x, skip_connections

@ts.function
def unet_up_block(
    x: ts.Tensor["B", "C_in", "H", "W", ts.bf16],
    skip_connections: list[ts.Tensor],
    timestep_emb: ts.Tensor["B", "emb_dim", ts.bf16],
    weights: ts.Dict[str, ts.Tensor],
    out_channels: int,
    num_layers: int = 2,
    upsample: bool = True,
    use_attention: bool = False
) -> ts.Tensor["B", "C_out", "H_out", "W_out", ts.bf16]:
    """U-Net upsampling block."""
    
    # Upsampling
    if upsample:
        x = ts.nn.interpolate(x, scale_factor=2, mode="nearest")
        x = ts.nn.conv2d(x, weights["upsample"])
    
    # ResNet blocks with skip connections
    for i in range(num_layers):
        # Concatenate with skip connection
        if i < len(skip_connections):
            skip = skip_connections[-(i+1)]  # Reverse order
            x = ts.cat([x, skip], dim=1)
        
        x = resnet_block_diffusion(
            x, timestep_emb, weights[f"resnet_{i}"], 
            out_channels if i == num_layers - 1 else None
        )
        
        # Optional attention
        if use_attention:
            x = attention_block(x, weights[f"attn_{i}"])
    
    return x

@ts.function
def unet_diffusion(
    x: ts.Tensor["B", "C", "H", "W", ts.bf16],
    timesteps: ts.Tensor["B", ts.int32],
    weights: ts.Dict[str, ts.Any],
    config: ts.Dict[str, ts.Any]
) -> ts.Tensor["B", "C", "H", "W", ts.bf16]:
    """Complete U-Net for diffusion models."""
    
    # Timestep embeddings
    time_emb = timestep_embedding(timesteps, config["time_embed_dim"])
    time_emb = linear(ts.nn.silu(time_emb), weights["time_mlp1"])
    time_emb = linear(ts.nn.silu(time_emb), weights["time_mlp2"])
    
    # Initial convolution
    h = ts.nn.conv2d(x, weights["conv_in"])
    
    # Downsampling path
    down_block_outs = [h]
    channels = config["model_channels"]
    
    for i, (mult, use_attn) in enumerate(config["down_blocks"]):
        out_ch = channels * mult
        h, skips = unet_down_block(
            h, time_emb, weights[f"down_{i}"], 
            out_ch, use_attention=use_attn,
            downsample=i < len(config["down_blocks"]) - 1
        )
        down_block_outs.extend(skips)
    
    # Middle block
    h = resnet_block_diffusion(h, time_emb, weights["mid_block1"])
    h = attention_block(h, weights["mid_attn"])
    h = resnet_block_diffusion(h, time_emb, weights["mid_block2"])
    
    # Upsampling path
    for i, (mult, use_attn) in enumerate(reversed(config["up_blocks"])):
        out_ch = channels * mult
        # Get corresponding skip connections
        num_skips = config["num_res_blocks"] + 1
        skips = down_block_outs[-num_skips:]
        down_block_outs = down_block_outs[:-num_skips]
        
        h = unet_up_block(
            h, skips, time_emb, weights[f"up_{i}"], 
            out_ch, use_attention=use_attn,
            upsample=i > 0
        )
    
    # Final convolution
    h = ts.nn.group_norm(h, num_groups=32)
    h = ts.nn.silu(h)
    return ts.nn.conv2d(h, weights["conv_out"])
```

## Custom Architectures

### Mixture of Experts (MoE)

```python
@ts.function
def top_k_gating(
    x: ts.Tensor["B", "S", "D", ts.bf16],
    gate_weights: ts.Tensor["D", "num_experts", ts.bf16],
    k: int = 2,
    noise_std: float = 1e-2
) -> tuple[ts.Tensor, ts.Tensor]:
    """Top-k gating for mixture of experts."""
    
    # Compute gate logits
    gate_logits = linear(x, gate_weights)  # [B, S, num_experts]
    
    # Add noise during training for load balancing
    if noise_std > 0:
        noise = ts.randn_like(gate_logits) * noise_std
        gate_logits = gate_logits + noise
    
    # Top-k selection
    top_k_logits, top_k_indices = ts.topk(gate_logits, k, dim=-1)
    
    # Softmax over top-k
    top_k_gates = ts.nn.softmax(top_k_logits, dim=-1)
    
    return top_k_gates, top_k_indices

@ts.function
def moe_layer(
    x: ts.Tensor["B", "S", "D", ts.bf16],
    expert_weights: list[ts.Dict[str, ts.Tensor]],
    gate_weights: ts.Tensor["D", "num_experts", ts.bf16],
    k: int = 2
) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """Mixture of Experts layer."""
    
    B, S, D = x.shape
    num_experts = len(expert_weights)
    
    # Gating
    gates, indices = top_k_gating(x, gate_weights, k)
    
    # Flatten for expert processing
    x_flat = x.reshape(-1, D)  # [B*S, D]
    gates_flat = gates.reshape(-1, k)  # [B*S, k]
    indices_flat = indices.reshape(-1, k)  # [B*S, k]
    
    # Initialize output
    output = ts.zeros_like(x_flat)
    
    # Process each expert
    for expert_id in range(num_experts):
        # Find tokens assigned to this expert
        expert_mask = (indices_flat == expert_id).any(dim=-1)
        if not expert_mask.any():
            continue
        
        # Get tokens for this expert
        expert_tokens = x_flat[expert_mask]
        
        # Expert computation (simple MLP)
        expert_out = expert_mlp(expert_tokens, expert_weights[expert_id])
        
        # Scatter back with gating weights
        for token_idx in range(expert_tokens.shape[0]):
            original_idx = expert_mask.nonzero()[token_idx]
            gate_weights = gates_flat[original_idx]
            expert_indices = indices_flat[original_idx]
            
            # Find position of current expert in top-k
            expert_pos = (expert_indices == expert_id).nonzero()[0]
            weight = gate_weights[expert_pos]
            
            output[original_idx] += weight * expert_out[token_idx]
    
    return output.reshape(B, S, D)

@ts.function
def expert_mlp(
    x: ts.Tensor["N", "D", ts.bf16],
    weights: ts.Dict[str, ts.Tensor]
) -> ts.Tensor["N", "D", ts.bf16]:
    """Individual expert MLP."""
    hidden = gelu(linear(x, weights["up"]))
    return linear(hidden, weights["down"])
```

### Memory-Efficient Attention Variants

```python
@ts.function
def linear_attention(
    q: ts.Tensor["B", "H", "S", "D", ts.bf16],
    k: ts.Tensor["B", "H", "S", "D", ts.bf16],
    v: ts.Tensor["B", "H", "S", "D", ts.bf16]
) -> ts.Tensor["B", "H", "S", "D", ts.bf16]:
    """Linear attention with O(N) complexity."""
    
    # Apply feature map (ELU + 1)
    q = ts.nn.elu(q) + 1
    k = ts.nn.elu(k) + 1
    
    # Compute KV and normalization
    kv = ts.matmul(k.transpose(-2, -1), v)  # [B, H, D, D]
    k_sum = k.sum(dim=-2, keepdim=True)     # [B, H, 1, D]
    
    # Apply to queries
    qkv = ts.matmul(q, kv)                  # [B, H, S, D]
    qk_sum = ts.matmul(q, k_sum.transpose(-2, -1))  # [B, H, S, 1]
    
    return qkv / (qk_sum + 1e-6)

@ts.function
def sliding_window_attention(
    q: ts.Tensor["B", "H", "S", "D", ts.bf16],
    k: ts.Tensor["B", "H", "S", "D", ts.bf16],
    v: ts.Tensor["B", "H", "S", "D", ts.bf16],
    window_size: int = 512
) -> ts.Tensor["B", "H", "S", "D", ts.bf16]:
    """Sliding window attention for long sequences."""
    
    B, H, S, D = q.shape
    
    # Pad sequence to window boundaries
    pad_len = (window_size - S % window_size) % window_size
    if pad_len > 0:
        q = ts.pad(q, (0, 0, 0, pad_len))
        k = ts.pad(k, (0, 0, 0, pad_len))
        v = ts.pad(v, (0, 0, 0, pad_len))
    
    S_padded = S + pad_len
    num_windows = S_padded // window_size
    
    # Reshape into windows
    q = q.reshape(B, H, num_windows, window_size, D)
    k = k.reshape(B, H, num_windows, window_size, D)
    v = v.reshape(B, H, num_windows, window_size, D)
    
    # Apply attention within each window
    output_windows = []
    for i in range(num_windows):
        window_out = ts.nn.flash_attention(
            q[:, :, i], k[:, :, i], v[:, :, i], causal=True
        )
        output_windows.append(window_out)
    
    # Concatenate windows
    output = ts.stack(output_windows, dim=2)
    output = output.reshape(B, H, S_padded, D)
    
    # Remove padding
    return output[:, :, :S, :]
```

## Model Composition

### Multi-Modal Models

```python
@ts.function
def vision_language_model(
    images: ts.Tensor["B", "C", "H", "W", ts.bf16],
    text_tokens: ts.Tensor["B", "T", ts.int32],
    weights: ts.Dict[str, ts.Any],
    config: ts.Dict[str, ts.Any]
) -> ts.Tensor["B", "T", "vocab_size", ts.bf16]:
    """Vision-language model combining ViT and language model."""
    
    # Vision encoder
    image_features = vit_model(images, weights["vision"], config["vision"])
    
    # Project to language model dimension
    image_features = linear(image_features, weights["vision_proj"])
    
    # Text embeddings
    text_emb = embedding(text_tokens, weights["text"]["token_emb"])
    
    # Combine vision and text features
    # Simple approach: prepend image features to text
    combined_features = ts.cat([image_features.unsqueeze(1), text_emb], dim=1)
    
    # Language model processing
    for i in range(config["text"]["num_layers"]):
        layer_weights = weights["text"][f"layer_{i}"]
        combined_features = gpt_block(combined_features, layer_weights, config["text"])
    
    # Only predict on text tokens (skip image feature positions)
    text_outputs = combined_features[:, image_features.shape[1]:, :]
    
    # Language modeling head
    return linear(text_outputs, weights["text"]["lm_head"])

### Hierarchical Models

@ts.function
def hierarchical_transformer(
    x: ts.Tensor["B", "S", "D", ts.bf16],
    weights: ts.Dict[str, ts.Any],
    config: ts.Dict[str, ts.Any]
) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """Hierarchical transformer with multiple resolution levels."""
    
    outputs = []
    current_x = x
    
    # Process at multiple resolutions
    for level in range(config["num_levels"]):
        level_config = config[f"level_{level}"]
        level_weights = weights[f"level_{level}"]
        
        # Downsample for higher levels
        if level > 0:
            current_x = downsample_sequence(current_x, factor=2)
        
        # Process at current resolution
        for layer_idx in range(level_config["num_layers"]):
            layer_weights = level_weights[f"layer_{layer_idx}"]
            current_x = gpt_block(current_x, layer# Tessera ML Models Guide - Building Modern ML Architectures

This guide shows how to implement complete ML models using Tessera, from simple building blocks to state-of-the-art architectures like GPT, BERT, and Diffusion models.

## Table of Contents

1. [Building Blocks](#building-blocks)
2. [Transformer Models](#transformer-models)
3. [Vision Models](#vision-models)
4. [Diffusion Models](#diffusion-models)
5. [Custom Architectures](#custom-architectures)
6. [Model Composition](#model-composition)
7. [Best Practices](#best-practices)

## Building Blocks

### Neural Network Primitives

Tessera provides optimized implementations of all standard NN components:

```python
import tessera as ts
import math

# Linear layers with automatic optimization
@ts.function
def linear(x: ts.Tensor["B", "Din", ts.bf16], 
          w: ts.Tensor["Din", "Dout", ts.bf16],
          bias: ts.Tensor["Dout", ts.bf16] = None) -> ts.Tensor["B", "Dout", ts.bf16]:
    """Optimized linear transformation."""
    out = ts.matmul(x, w)
    if bias is not None:
        out = out + bias
    return out

# Normalization with numerical stability
@ts.function  
def layer_norm(x: ts.Tensor["B", "S", "D", ts.bf16],
               eps: float = 1e-5) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """Numerically stable layer normalization."""
    return ts.nn.layer_norm_safe(x, eps=eps)

@ts.function
def rms_norm(x: ts.Tensor["B", "S", "D", ts.bf16], 
            weight: ts.Tensor["D", ts.bf16],
            eps: float = 1e-6) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """RMS normalization used in LLaMA and other modern models."""
    return ts.nn.rms_norm_safe(x, weight, eps=eps)

# Activation functions
@ts.function
def gelu(x: ts.Tensor) -> ts.Tensor:
    """GELU activation with numerical stability."""
    return ts.nn.gelu_safe(x)

@ts.function  
def swiglu(x: ts.Tensor["B", "S", "2*D", ts.bf16]) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """SwiGLU activation used in PaLM and LLaMA."""
    x1, x2 = ts.split(x, dim=-1, sections=2)
    return ts.nn.silu(x1) * x2

# Embeddings
@ts.function
def embedding(tokens: ts.Tensor["B", "S", ts.int32],
             weight: ts.Tensor["vocab_size", "D", ts.bf16]) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """Token embedding lookup."""
    return ts.nn.embedding(tokens, weight)

@ts.function
def rotary_embedding(x: ts.Tensor["B", "H", "S", "D", ts.bf16],
                    freqs: ts.Tensor["S", "D//2", ts.complex64]) -> ts.Tensor["B", "H", "S", "D", ts.bf16]:
    """Rotary Position Embedding (RoPE) from GPT-NeoX and LLaMA."""
    return ts.nn.apply_rotary_pos_emb(x, freqs)
```

### Optimized Attention Mechanisms

```python
@ts.function
def multi_head_attention(
    x: ts.Tensor["B", "S", "D", ts.bf16],
    w_qkv: ts.Tensor["D", "3*D", ts.bf16],
    w_out: ts.Tensor["D", "D", ts.bf16],
    num_heads: int,
    causal: bool = True
) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """Multi-head attention with Flash Attention optimization."""
    
    B, S, D = x.shape
    head_dim = D // num_heads
    
    # QKV projection
    qkv = linear(x, w_qkv)  # [B, S, 3*D]
    q, k, v = ts.split(qkv, dim=-1, sections=3)
    
    # Reshape for multi-head attention
    q = q.reshape(B, S, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D_h]
    k = k.reshape(B, S, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, S, num_heads, head_dim).transpose(1, 2)
    
    # Flash Attention (automatically optimized)
    out = ts.nn.flash_attention(q, k, v, causal=causal)
    
    # Reshape and project output
    out = out.transpose(1, 2).reshape(B, S, D)
    return linear(out, w_out)

@ts.function
def grouped_query_attention(
    x: ts.Tensor["B", "S", "D", ts.bf16],
    w_q: ts.Tensor["D", "H_q*D_h", ts.bf16],
    w_kv: ts.Tensor["D", "H_kv*2*D_h", ts.bf16], 
    w_out: ts.Tensor["D", "D", ts.bf16],
    num_heads_q: int,
    num_heads_kv: int
) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """Grouped Query Attention from LLaMA-2."""
    
    B, S, D = x.shape
    head_dim = D // num_heads_q
    
    # Separate Q and KV projections
    q = linear(x, w_q)  # [B, S, H_q*D_h]
    kv = linear(x, w_kv)  # [B, S, H_kv*2*D_h]
    
    # Reshape queries
    q = q.reshape(B, S, num_heads_q, head_dim).transpose(1, 2)
    
    # Reshape and split keys/values
    k, v = ts.split(kv, dim=-1, sections=2)
    k = k.reshape(B, S, num_heads_kv, head_dim).transpose(1, 2)
    v = v.reshape(B, S, num_heads_kv, head_dim).transpose(1, 2)
    
    # Repeat K,V for each Q head group
    group_size = num_heads_q // num_heads_kv
    k = ts.repeat_interleave(k, repeats=group_size, dim=1)
    v = ts.repeat_interleave(v, repeats=group_size, dim=1)
    
    # Flash Attention
    out = ts.nn.flash_attention(q, k, v, causal=True)
    out = out.transpose(1, 2).reshape(B, S, D)
    
    return linear(out, w_out)
```

## Transformer Models

### GPT-Style Decoder Model

```python
@ts.function
def gpt_block(
    x: ts.Tensor["B", "S", "D", ts.bf16],
    attn_weights: ts.Dict[str, ts.Tensor],
    mlp_weights: ts.Dict[str, ts.Tensor],
    ln1_weight: ts.Tensor["D", ts.bf16],
    ln2_weight: ts.Tensor["D", ts.bf16],
    config: ts.Dict[str, ts.Any]
) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """GPT transformer block with pre-normalization."""
    
    # Pre-norm attention
    attn_input = layer_norm(x)
    if config.get("use_rms_norm", False):
        attn_input = rms_norm(attn_input, ln1_weight)
    
    # Multi-head attention
    attn_out = multi_head_attention(
        attn_input,
        attn_weights["qkv"],
        attn_weights["out"],
        num_heads=config["num_heads"]
    )
    
    # Residual connection
    x = x + attn_out
    
    # Pre-norm MLP
    mlp_input = layer_norm(x)
    if config.get("use_rms_norm", False):
        mlp_input = rms_norm(mlp_input, ln2_weight)
    
    # MLP with configurable activation
    hidden = linear(mlp_input, mlp_weights["up"])
    
    if config.get("use_swiglu", False):
        # SwiGLU activation (LLaMA style)
        gate = linear(mlp_input, mlp_weights["gate"]) 
        hidden = ts.nn.silu(gate) * hidden
    else:
        # Standard GELU
        hidden = gelu(hidden)
    
    mlp_out = linear(hidden, mlp_weights["down"])
    
    # Final residual
    return x + mlp_out

@ts.function
def gpt_model(
    tokens: ts.Tensor["B", "S", ts.int32],
    weights: ts.Dict[str, ts.Any],
    config: ts.Dict[str, ts.Any]
) -> ts.Tensor["B", "S", "vocab_size", ts.bf16]:
    """Complete GPT model."""
    
    # Token embeddings
    x = embedding(tokens, weights["token_emb"])
    
    # Position embeddings (if not using RoPE)
    if not config.get("use_rope", False):
        pos_emb = weights["pos_emb"][:x.shape[1]]
        x = x + pos_emb
    
    # Transformer blocks
    for i in range(config["num_layers"]):
        block_weights = {
            "attn": weights[f"layer_{i}"]["attn"],
            "mlp": weights[f"layer_{i}"]["mlp"],
            "ln1": weights[f"layer_{i}"]["ln1"],
            "ln2": weights[f"layer_{i}"]["ln2"]
        }
        
        x = gpt_block(x, block_weights, config)
        
        # Optional: apply RoPE after each layer
        if config.get("use_rope", False):
            rope_freqs = ts.nn.compute_rope_freqs(
                config["head_dim"], x.shape[1], 
                theta=config.get("rope_theta", 10000)
            )
            x = rotary_embedding(x, rope_freqs)
    
    # Final layer norm
    x = layer_norm(x)
    if config.get("use_rms_norm", False):
        x = rms_norm(x, weights["final_ln"])
    
    # Language modeling head
    logits = linear(x, weights["lm_head"])
    
    return logits

# Example usage
def create_gpt_config(model_size="125M"):
    """Create configuration for different GPT model sizes."""
    configs = {
        "125M": {
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "vocab_size": 50257,
            "max_seq_len": 1024,
            "use_rope": False,
            "use_rms_norm": False,
            "use_swiglu": False
        },
        "1.3B": {
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 2048,
            "vocab_size": 50257,
            "max_seq_len": 2048,
            "use_rope": False,
            "use_rms_norm": False,
            "use_swiglu": False
        },
        "llama_7B": {
            "num_layers": 32,
            "num_heads": 32,
            "hidden_dim": 4096,
            "vocab_size": 32000,
            "max_seq_len": 2048,
            "head_dim": 128,
            "use_rope": True,
            "use_rms_norm": True,
            "use_swiglu": True,
            "rope_theta": 10000
        }
    }
    return configs[model_size]
```

### BERT-Style Encoder Model

```python
@ts.function
def bert_attention(
    x: ts.Tensor["B", "S", "D", ts.bf16],
    w_qkv: ts.Tensor["D", "3*D", ts.bf16],
    w_out: ts.Tensor["D", "D", ts.bf16],
    attention_mask: ts.Tensor["B", "S", ts.bool],
    num_heads: int
) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """BERT-style bidirectional attention."""
    
    B, S, D = x.shape
    head_dim = D // num_heads
    
    # QKV projection
    qkv = linear(x, w_qkv)
    q, k, v = ts.split(qkv, dim=-1, sections=3)
    
    # Reshape for multi