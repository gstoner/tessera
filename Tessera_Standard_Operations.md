# Tessera Standard Operations: Implementation Guide

## Chapter 1: Normalization Operations

1.1 RMSNorm (Root Mean Square Normalization)

```python
@tessera.function
def rms_norm(
    x: Tensor["B", "S", "D"],
    weight: Tensor["D"],
    eps: float = 1e-6
) -> Tensor["B", "S", "D"]:
    """
    RMSNorm: Simpler and more efficient than LayerNorm
    No mean centering, just variance normalization
    """
    # Compute RMS
    rms = tessera.sqrt(tessera.mean(x ** 2, dim=-1, keepdim=True) + eps)
    
    # Normalize and scale
    x_normed = x / rms
    return x_normed * weight

# Optimized kernel version
@tessera.kernel
def rms_norm_kernel(
    x: Tile["S", "D", bf16],
    weight: Tile["D", f32],
    output: Tile["S", "D", bf16],
    eps: float = 1e-6
):
    """Fused RMSNorm kernel for efficiency"""
    ctx = tile.context()
    
    # Load input tile
    x_tile = tile.load(x, shape=(ctx.S, ctx.D))
    w_tile = tile.load_broadcast(weight)  # Broadcast weight
    
    # Compute RMS in higher precision
    x_f32 = tile.cast(x_tile, f32)
    x_squared = tile.mul(x_f32, x_f32)
    
    # Warp-level reduction for efficiency
    rms = tile.warp.reduce_mean(x_squared, axis=-1)
    rms = tile.rsqrt(rms + eps)  # Reciprocal sqrt for efficiency
    
    # Normalize and scale
    x_normed = tile.mul(x_f32, tile.broadcast(rms, axis=-1))
    x_scaled = tile.mul(x_normed, w_tile)
    
    # Store result
    tile.store(output, tile.cast(x_scaled, bf16))

# Stable version for extreme values
@tessera.numerically_stable
def stable_rms_norm(
    x: Tensor["B", "S", "D"],
    weight: Tensor["D"],
    eps: float = 1e-6
) -> Tensor["B", "S", "D"]:
    """
    Numerically stable RMSNorm that handles:
    - Very large/small values
    - Mixed precision training
    - Gradient stability
    """
    # Use Welford's algorithm for numerical stability
    with tessera.precision(compute=f32, accumulate=f64):
        # Compute mean of squares with compensation
        mean_sq = tessera.stable.mean_of_squares(x, dim=-1, keepdim=True)
        
        # Stable RMS computation
        rms = tessera.stable.sqrt(mean_sq + eps)
        
        # Scale with gradient-stable division
        x_normed = tessera.stable.divide(x, rms)
    
    return x_normed * weight

# Distributed RMSNorm for model parallelism
@tessera.distributed
def distributed_rms_norm(
    x: MeshTensor["B", "S", "D"],
    weight: MeshTensor["D"],
    mesh: Mesh,
    eps: float = 1e-6
) -> MeshTensor["B", "S", "D"]:
    """RMSNorm with tensor parallelism"""
    with mesh.axis("model"):
        # All-reduce for computing global RMS
        local_sq = x ** 2
        global_mean_sq = tessera.mesh_reduce(
            tessera.mean(local_sq, dim=-1, keepdim=True),
            axis="model",
            op="mean"
        )
        
        # Local normalization with global statistics
        rms = tessera.sqrt(global_mean_sq + eps)
        x_normed = x / rms
        
        # Weight can be sharded or replicated
        return x_normed * weight
```
## Chapter 2: Activation Functions

### 2.1 SwiGLU (Swish-Gated Linear Unit)

```python
@tessera.function
def swiglu(
    x: Tensor["B", "S", "D"],
    W_gate: Tensor["D", "D_ff"],
    W_up: Tensor["D", "D_ff"],
    W_down: Tensor["D_ff", "D"],
    beta: float = 1.0
) -> Tensor["B", "S", "D"]:
    """
    SwiGLU: A gated activation function used in LLaMA/PaLM
    Combines Swish activation with GLU gating
    """
    # Gate path with Swish activation
    gate = tessera.nn.swish(x @ W_gate, beta=beta)
    
    # Up projection path  
    up = x @ W_up
    
    # Gated combination
    hidden = gate * up
    
    # Down projection
    return hidden @ W_down

# Optimized fused kernel
@tessera.kernel
def swiglu_fused_kernel(
    x: Tile["S", "D", bf16],
    W_gate: Tile["D", "D_ff", bf16],
    W_up: Tile["D", "D_ff", bf16],
    W_down: Tile["D_ff", "D", bf16],
    output: Tile["S", "D", bf16],
    beta: float = 1.0
):
    """Single kernel for entire SwiGLU operation"""
    ctx = tile.context()
    
    # Load input
    x_tile = tile.load(x, shape=(ctx.S, ctx.D))
    
    # Fused gate and up projection using tensor cores
    # Concatenate weights for single matmul
    W_combined = tile.concat([W_gate, W_up], axis=1)
    combined_out = tile.mma(x_tile, W_combined)  # [S, 2*D_ff]
    
    # Split results
    gate_out, up_out = tile.split(combined_out, 2, axis=-1)
    
    # Apply Swish activation to gate (with beta scaling)
    gate_sigmoid = tile.sigmoid(tile.mul(gate_out, beta))
    gate_activated = tile.mul(gate_out, gate_sigmoid)
    
    # Element-wise multiplication (gating)
    hidden = tile.mul(gate_activated, up_out)
    
    # Down projection
    output_tile = tile.mma(hidden, W_down)
    
    # Store result
    tile.store(output, output_tile)

# Memory-efficient version with activation checkpointing
@tessera.checkpoint
def swiglu_memory_efficient(
    x: Tensor["B", "S", "D"],
    W_gate: Tensor["D", "D_ff"],
    W_up: Tensor["D", "D_ff"], 
    W_down: Tensor["D_ff", "D"],
    chunk_size: int = 512
) -> Tensor["B", "S", "D"]:
    """
    Memory-efficient SwiGLU using chunking and recomputation
    Useful for very large D_ff (e.g., 4*D or 8*D)
    """
    B, S, D = x.shape
    D_ff = W_gate.shape[-1]
    
    # Process in chunks to reduce peak memory
    output = tessera.zeros_like(x)
    
    for i in range(0, D_ff, chunk_size):
        end = min(i + chunk_size, D_ff)
        
        # Checkpoint each chunk
        with tessera.checkpoint_region():
            # Chunk weights
            W_gate_chunk = W_gate[:, i:end]
            W_up_chunk = W_up[:, i:end]
            W_down_chunk = W_down[i:end, :]
            
            # Compute chunk
            gate = tessera.nn.swish(x @ W_gate_chunk)
            up = x @ W_up_chunk
            hidden = gate * up
            
            # Accumulate output
            output += hidden @ W_down_chunk
    
    return output

# Quantized SwiGLU for inference
@tessera.quantized(weights=int8, activations=fp8)
def swiglu_quantized(
    x: Tensor["B", "S", "D", fp8],
    W_gate: Tensor["D", "D_ff", int8],
    W_up: Tensor["D", "D_ff", int8],
    W_down: Tensor["D_ff", "D", int8],
    scales: Dict[str, float]
) -> Tensor["B", "S", "D", fp8]:
    """
    Quantized SwiGLU for efficient inference
    Uses INT8 weights and FP8 activations
    """
    # Dequantize for computation
    with tessera.quantization.context(scales):
        # INT8 matrix multiplication with FP8 accumulation
        gate = tessera.nn.quantized_matmul(
            x, W_gate, 
            input_scale=scales["input"],
            weight_scale=scales["gate_weight"],
            output_dtype=fp8
        )
        
        # Swish in FP8
        gate = tessera.nn.swish_fp8(gate)
        
        # Similar for up projection
        up = tessera.nn.quantized_matmul(
            x, W_up,
            input_scale=scales["input"],
            weight_scale=scales["up_weight"],
            output_dtype=fp8
        )
        
        # Gating in FP8
        hidden = tessera.mul_fp8(gate, up)
        
        # Final projection
        return tessera.nn.quantized_matmul(
            hidden, W_down,
            input_scale=scales["hidden"],
            weight_scale=scales["down_weight"],
            output_dtype=fp8
        )
```

## Chapter 3: Attention Mechanisms

### 3.1 Multi-Head Attention with Flash Attention

```python
@tessera.function
def attention(
    q: Tensor["B", "H", "S", "D"],
    k: Tensor["B", "H", "S", "D"],
    v: Tensor["B", "H", "S", "D"],
    mask: Optional[Tensor["S", "S"]] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> Tensor["B", "H", "S", "D"]:
    """
    Efficient attention using Flash Attention algorithm
    Fuses attention computation to reduce memory from O(S²) to O(S)
    """
    B, H, S, D = q.shape
    scale = scale or (1.0 / math.sqrt(D))
    
    # Use Flash Attention kernel
    return tessera.ops.flash_attention(
        q, k, v,
        mask=mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale
    )

@tessera.kernel
def flash_attention_kernel(
    Q: Tile["B*H", "S", "D", bf16],
    K: Tile["B*H", "S", "D", bf16],
    V: Tile["B*H", "S", "D", bf16],
    O: Tile["B*H", "S", "D", bf16],
    mask: Optional[Tile["S", "S"]] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 1.0
):
    """
    Flash Attention kernel implementation
    Uses tiling and online softmax for memory efficiency
    """
    ctx = tile.context()
    BH, S, D = Q.shape
    
    # Tile sizes for Q and KV
    Br = ctx.block_m  # Rows of Q
    Bc = ctx.block_n  # Cols of K/V
    
    # Allocate shared memory
    Q_block = tile.alloc_shared((Br, D), bf16)
    K_block = tile.alloc_shared((Bc, D), bf16)
    V_block = tile.alloc_shared((Bc, D), bf16)
    
    # Output accumulator in registers
    O_accum = tile.zeros((Br, D), f32)
    row_max = tile.full((Br,), -float('inf'), f32)
    row_sum = tile.zeros((Br,), f32)
    
    # Process Q blocks
    for q_idx in tile.range(0, S, Br):
        # Load Q block
        tile.load_async(Q[q_idx:q_idx+Br], Q_block)
        tile.wait()
        
        # Reset statistics for this Q block
        tile.fill(row_max, -float('inf'))
        tile.fill(row_sum, 0.0)
        tile.fill(O_accum, 0.0)
        
        # Process KV blocks
        for kv_idx in tile.range(0, S, Bc):
            # Skip if causal mask
            if is_causal and kv_idx > q_idx + Br:
                break
            
            # Load K, V blocks
            tile.load_async(K[kv_idx:kv_idx+Bc], K_block)
            tile.load_async(V[kv_idx:kv_idx+Bc], V_block)
            tile.wait()
            
            # Compute QK^T
            S_block = tile.mma(Q_block, tile.transpose(K_block))
            S_block = tile.mul(S_block, scale)
            
            # Apply mask if needed
            if is_causal:
                tile.apply_causal_mask(S_block, q_idx, kv_idx)
            if mask is not None:
                S_block = tile.add(S_block, mask[q_idx:q_idx+Br, kv_idx:kv_idx+Bc])
            
            # Online softmax computation
            block_max = tile.max(S_block, axis=1)
            
            # Update running max and adjust previous accumulator
            prev_max = row_max
            row_max = tile.maximum(row_max, block_max)
            
            # Compute exp with new max
            S_block = tile.exp(S_block - tile.broadcast(row_max, axis=1))
            
            # Adjust previous accumulator
            adjust_factor = tile.exp(prev_max - row_max)
            O_accum = tile.mul(O_accum, tile.broadcast(adjust_factor, axis=1))
            row_sum = tile.mul(row_sum, adjust_factor)
            
            # Apply dropout if needed
            if dropout_p > 0:
                S_block = tile.dropout(S_block, dropout_p)
            
            # Update accumulator
            P_V = tile.mma(S_block, V_block)
            O_accum = tile.add(O_accum, P_V)
            row_sum = tile.add(row_sum, tile.sum(S_block, axis=1))
        
        # Normalize output
        O_block = tile.div(O_accum, tile.broadcast(row_sum, axis=1))
        
        # Store output
        tile.store(O[q_idx:q_idx+Br], tile.cast(O_block, bf16))

# Grouped Query Attention (GQA)
@tessera.function
def grouped_query_attention(
    q: Tensor["B", "H_q", "S", "D"],
    k: Tensor["B", "H_kv", "S", "D"],
    v: Tensor["B", "H_kv", "S", "D"],
    is_causal: bool = True
) -> Tensor["B", "H_q", "S", "D"]:
    """
    Grouped Query Attention used in LLaMA-2
    Reduces KV cache memory by sharing KV heads across Q heads
    """
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    
    # Number of Q heads per KV head
    group_size = H_q // H_kv
    
    # Reshape Q for grouped attention
    q = q.reshape(B, H_kv, group_size, S, D)
    
    # Expand KV heads
    k = k.unsqueeze(2)  # [B, H_kv, 1, S, D]
    v = v.unsqueeze(2)  # [B, H_kv, 1, S, D]
    
    # Broadcast KV to match Q groups
    k = k.expand(B, H_kv, group_size, S, D)
    v = v.expand(B, H_kv, group_size, S, D)
    
    # Reshape back for attention
    q = q.reshape(B, H_q, S, D)
    k = k.reshape(B, H_q, S, D)
    v = v.reshape(B, H_q, S, D)
    
    return attention(q, k, v, is_causal=is_causal)

# Multi-Query Attention (MQA)
@tessera.function
def multi_query_attention(
    q: Tensor["B", "H", "S", "D"],
    k: Tensor["B", "S", "D"],  # Single KV head
    v: Tensor["B", "S", "D"],
    is_causal: bool = True
) -> Tensor["B", "H", "S", "D"]:
    """
    Multi-Query Attention - extreme KV cache reduction
    Single KV head shared across all Q heads
    """
    B, H, S, D = q.shape
    
    # Expand KV for all heads
    k = k.unsqueeze(1).expand(B, H, S, D)
    v = v.unsqueeze(1).expand(B, H, S, D)
    
    return attention(q, k, v, is_causal=is_causal)

Chapter 4: Positional Embeddings
4.1 Rotary Position Embedding (RoPE)
python@tessera.function
def rotary_embedding(
    x: Tensor["B", "S", "H", "D"],
    position_ids: Optional[Tensor["B", "S"]] = None,
    base: float = 10000.0,
    scaling_factor: float = 1.0
) -> Tensor["B", "S", "H", "D"]:
    """
    Rotary Position Embedding from RoFormer
    Encodes position information through rotation of feature dimensions
    """
    B, S, H, D = x.shape
    
    # Generate position indices if not provided
    if position_ids is None:
        position_ids = tessera.arange(S).unsqueeze(0).expand(B, S)
    
    # Apply scaling for length extrapolation
    position_ids = position_ids * scaling_factor
    
    # Compute rotation frequencies
    inv_freq = 1.0 / (base ** (tessera.arange(0, D, 2).float() / D))
    
    # Create rotation matrix
    position_ids_expanded = position_ids.unsqueeze(-1)  # [B, S, 1]
    freqs = position_ids_expanded * inv_freq  # [B, S, D/2]
    
    # Create cos and sin embeddings
    cos_emb = tessera.cos(freqs).unsqueeze(2)  # [B, S, 1, D/2]
    sin_emb = tessera.sin(freqs).unsqueeze(2)  # [B, S, 1, D/2]
    
    # Apply rotation
    return apply_rotary_pos_emb(x, cos_emb, sin_emb)

@tessera.kernel
def rotary_embedding_kernel(
    x: Tile["S", "H", "D", bf16],
    cos_cache: Tile["S", "D/2", f32],
    sin_cache: Tile["S", "D/2", f32],
    output: Tile["S", "H", "D", bf16]
):
    """
    Optimized RoPE kernel with precomputed cos/sin cache
    """
    ctx = tile.context()
    S, H, D = x.shape
    D_half = D // 2
    
    # Load input tile
    x_tile = tile.load(x)
    
    # Split into pairs for rotation
    x_r = x_tile[..., :D_half]
    x_i = x_tile[..., D_half:]
    
    # Load precomputed cos/sin values
    cos = tile.load_broadcast(cos_cache)  # Broadcast across H
    sin = tile.load_broadcast(sin_cache)
    
    # Apply rotation: (x_r + i*x_i) * (cos + i*sin)
    # Real part: x_r * cos - x_i * sin
    # Imaginary part: x_r * sin + x_i * cos
    out_r = tile.sub(tile.mul(x_r, cos), tile.mul(x_i, sin))
    out_i = tile.add(tile.mul(x_r, sin), tile.mul(x_i, cos))
    
    # Concatenate real and imaginary parts
    output_tile = tile.concat([out_r, out_i], axis=-1)
    
    # Store result
    tile.store(output, output_tile)

# Helper function for applying rotary embedding
@tessera.function
def apply_rotary_pos_emb(
    x: Tensor["B", "S", "H", "D"],
    cos: Tensor["B", "S", 1, "D/2"],
    sin: Tensor["B", "S", 1, "D/2"]
) -> Tensor["B", "S", "H", "D"]:
    """Apply rotation to input tensor"""
    # Split input into two halves
    x_r, x_i = x.chunk(2, dim=-1)
    
    # Broadcast cos/sin across heads
    cos = cos.expand(-1, -1, x.shape[2], -1)
    sin = sin.expand(-1, -1, x.shape[2], -1)
    
    # Apply rotation formula
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    
    return tessera.cat([out_r, out_i], dim=-1)

# Dynamic NTK-aware RoPE for length extrapolation
@tessera.function
def dynamic_rope(
    x: Tensor["B", "S", "H", "D"],
    position_ids: Tensor["B", "S"],
    base: float = 10000.0,
    max_position: int = 2048,
    scaling_type: str = "linear"
) -> Tensor["B", "S", "H", "D"]:
    """
    Dynamic RoPE with NTK-aware scaling for longer sequences
    """
    current_max = position_ids.max().item()
    
    if current_max <= max_position:
        # Within training length, use standard RoPE
        return rotary_embedding(x, position_ids, base)
    
    # Calculate scaling factor
    if scaling_type == "linear":
        scaling_factor = current_max / max_position
    elif scaling_type == "ntk":
        # NTK-aware scaling
        scaling_factor = (current_max / max_position) ** (x.shape[-1] / (x.shape[-1] - 2))
        base = base * scaling_factor
        scaling_factor = 1.0
    elif scaling_type == "yarn":
        # YaRN scaling
        scaling_factor = math.log(current_max / max_position) + 1
    
    return rotary_embedding(x, position_ids, base, scaling_factor)
```
### 4.2 Cosine-Sine Positional Encoding

``` python
@tessera.function
def cos_sin_embedding(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    dtype: DType = f32
) -> Tuple[Tensor["seq_len", "dim"], Tensor["seq_len", "dim"]]:
    """
    Traditional cosine-sine positional encoding from "Attention is All You Need"
    Returns both cos and sin components
    """
    position = tessera.arange(seq_len, dtype=dtype).unsqueeze(1)
    div_term = tessera.exp(
        tessera.arange(0, dim, 2, dtype=dtype) * 
        -(math.log(base) / dim)
    )
    
    # Compute cos and sin embeddings
    cos_emb = tessera.zeros(seq_len, dim, dtype=dtype)
    sin_emb = tessera.zeros(seq_len, dim, dtype=dtype)
    
    cos_emb[:, 0::2] = tessera.cos(position * div_term)
    cos_emb[:, 1::2] = tessera.cos(position * div_term)
    
    sin_emb[:, 0::2] = tessera.sin(position * div_term)
    sin_emb[:, 1::2] = tessera.sin(position * div_term)
    
    return cos_emb, sin_emb

# Learnable absolute position embedding
@tessera.module
class LearnedPositionalEmbedding:
    def __init__(self, max_seq_len: int, dim: int):
        self.embeddings = tessera.nn.Embedding(max_seq_len, dim)
    
    def forward(self, position_ids: Tensor["B", "S"]) -> Tensor["B", "S", "D"]:
        return self.embeddings(position_ids)

# ALiBi (Attention with Linear Biases)
@tessera.function
def alibi_bias(
    seq_len: int,
    num_heads: int,
    dtype: DType = f32
) -> Tensor["H", "S", "S"]:
    """
    ALiBi: Position embeddings through attention bias
    No learned parameters, just linear biases
    """
    # Create position bias matrix
    positions = tessera.arange(seq_len, dtype=dtype)
    distances = positions.unsqueeze(0) - positions.unsqueeze(1)
    
    # Slopes for each head (geometric sequence)
    slopes = 2 ** (-8 * tessera.arange(1, num_heads + 1) / num_heads)
    
    # Apply slopes to distances
    alibi = distances.unsqueeze(0) * slopes.view(-1, 1, 1)
    
    return alibi
```
## Chapter 5: Casted Operations

### 5.1 Casted Embedding

```python
@tessera.module
class CastedEmbedding:
    """
    Embedding layer with automatic type casting
    Stores embeddings in lower precision, computes in higher precision
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        storage_dtype: DType = fp8_e4m3,
        compute_dtype: DType = bf16,
        scale_factor: float = 1.0
    ):
        # Store embeddings in low precision
        self.embeddings = tessera.nn.Parameter(
            tessera.randn(vocab_size, embedding_dim, dtype=storage_dtype) * scale_factor
        )
        self.scale = scale_factor
        self.compute_dtype = compute_dtype
        
        # Quantization parameters for FP8
        if storage_dtype in [fp8_e4m3, fp8_e5m2]:
            self.quantize_scale = tessera.nn.Parameter(
                tessera.ones(1, dtype=f32)
            )
    
    @tessera.function
    def forward(self, input_ids: Tensor["B", "S"]) -> Tensor["B", "S", "D"]:
        # Cast to compute precision
        embeddings_compute = tessera.cast(self.embeddings, self.compute_dtype)
        
        # Apply quantization scale if needed
        if hasattr(self, 'quantize_scale'):
            embeddings_compute = embeddings_compute * self.quantize_scale
        
        # Gather embeddings
        output = tessera.gather(embeddings_compute, input_ids)
        
        return output

@tessera.kernel
def casted_embedding_kernel(
    indices: Tile["B*S", int32],
    embeddings: Tile["V", "D", fp8],
    output: Tile["B*S", "D", bf16],
    scale: float
):
    """Optimized kernel for casted embedding lookup"""
    ctx = tile.context()
    
    # Load indices
    idx_tile = tile.load(indices)
    
    # Gather embeddings with casting
    for i in tile.thread_range(ctx.BS):
        idx = idx_tile[i]
        
        # Load embedding vector in FP8
        emb_fp8 = tile.load_vector(embeddings[idx])
        
        # Cast to BF16 and scale
        emb_bf16 = tile.cast(emb_fp8, bf16)
        emb_scaled = tile.mul(emb_bf16, scale)
        
        # Store result
        tile.store_vector(output[i], emb_scaled)
```
### 5.2 Casted Linear

```python
@tessera.module
class CastedLinear:
    """
    Linear layer with mixed precision computation
    Weights in low precision, computation in higher precision
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_dtype: DType = int8,
        activation_dtype: DType = bf16,
        accumulate_dtype: DType = f32
    ):
        # Quantized weights
        self.weight = tessera.nn.Parameter(
            tessera.randn(out_features, in_features, dtype=weight_dtype)
        )
        
        # Quantization scales
        self.weight_scale = tessera.nn.Parameter(
            tessera.ones(out_features, 1, dtype=f32)
        )
        
        if bias:
            self.bias = tessera.nn.Parameter(
                tessera.zeros(out_features, dtype=activation_dtype)
            )
        
        self.activation_dtype = activation_dtype
        self.accumulate_dtype = accumulate_dtype
    
    @tessera.function
    def forward(self, x: Tensor["B", "S", "D_in"]) -> Tensor["B", "S", "D_out"]:
        # Cast input if needed
        x_compute = tessera.cast(x, self.activation_dtype)
        
        # Dequantize weights
        weight_deq = tessera.cast(self.weight, self.accumulate_dtype)
        weight_scaled = weight_deq * self.weight_scale
        
        # Matrix multiplication with accumulation type
        output = tessera.matmul(
            x_compute,
            weight_scaled.T,
            accumulate_dtype=self.accumulate_dtype
        )
        
        # Add bias if present
        if hasattr(self, 'bias'):
            output = output + self.bias
        
        # Cast to output type
        return tessera.cast(output, self.activation_dtype)

# INT8 optimized version
@tessera.kernel
def int8_linear_kernel(
    x: Tile["B*S", "D_in", int8],
    weight: TileRetryGContinueEditpythondef int8_linear_kernel(
   x: Tile["B*S", "D_in", int8],
   weight: Tile["D_out", "D_in", int8],
   bias: Optional[Tile["D_out", f32]],
   output: Tile["B*S", "D_out", bf16],
   x_scale: float,
   w_scale: Tensor["D_out"],
   output_scale: float
):
   """INT8 matrix multiplication with per-channel scaling"""
   ctx = tile.context()
   
   # Use INT8 tensor cores
   for batch_idx in tile.grid(ctx.BS):
       # Load input tile
       x_tile = tile.load(x[batch_idx], dtype=int8)
       
       # Compute INT8 matmul with INT32 accumulation
       acc = tile.zeros((ctx.D_out,), dtype=int32)
       
       for k in tile.range(0, ctx.D_in, 16):  # Process in chunks
           x_chunk = x_tile[k:k+16]
           w_chunk = tile.load(weight[:, k:k+16], dtype=int8)
           
           # INT8 tensor core operation
           acc = tile.dp4a(x_chunk, w_chunk, acc)  # INT8 -> INT32
       
       # Dequantize and scale
       acc_f32 = tile.cast(acc, f32)
       scaled = tile.mul(acc_f32, x_scale)
       
       # Per-channel weight scaling
       for i in tile.range(ctx.D_out):
           scaled[i] = tile.mul(scaled[i], w_scale[i])
       
       # Add bias if present
       if bias is not None:
           bias_tile = tile.load(bias)
           scaled = tile.add(scaled, bias_tile)
       
       # Quantize output
       output_bf16 = tile.cast(scaled, bf16)
       output_scaled = tile.mul(output_bf16, output_scale)
       
       # Store result
       tile.store(output[batch_idx], output_scaled)
```
### 5.3 Casted Sparse Embedding

```python
@tessera.module
class CastedSparseEmbedding:
    """
    Sparse embedding with mixed precision and efficient storage
    Useful for large vocabularies with sparsity patterns
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        sparsity: float = 0.9,
        storage_dtype: DType = fp8_e4m3,
        compute_dtype: DType = bf16,
        block_size: int = 16
    ):
        # Sparse storage format
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity = sparsity
        self.block_size = block_size
        
        # Store only non-zero blocks
        num_blocks = (num_embeddings * embedding_dim) // (block_size * block_size)
        num_dense_blocks = int(num_blocks * (1 - sparsity))
        
        # Dense blocks storage
        self.dense_blocks = tessera.nn.Parameter(
            tessera.randn(num_dense_blocks, block_size, block_size, dtype=storage_dtype)
        )
        
        # Block indices (CSR format)
        self.block_indices = tessera.nn.Buffer(
            tessera.zeros(num_dense_blocks, dtype=int32)
        )
        self.block_ptr = tessera.nn.Buffer(
            tessera.zeros(num_embeddings + 1, dtype=int32)
        )
        
        # Quantization scales per block
        self.block_scales = tessera.nn.Parameter(
            tessera.ones(num_dense_blocks, dtype=f32)
        )
        
        self.compute_dtype = compute_dtype
        
        # Initialize sparse structure
        self._init_sparse_structure()
    
    def _init_sparse_structure(self):
        """Initialize sparse block structure"""
        # Random sparsity pattern (can be learned)
        mask = tessera.rand(self.num_embeddings, self.embedding_dim // self.block_size)
        mask = mask > self.sparsity
        
        # Build CSR structure
        ptr = 0
        for i in range(self.num_embeddings):
            self.block_ptr[i] = ptr
            blocks_in_row = mask[i].sum().item()
            ptr += blocks_in_row
        self.block_ptr[-1] = ptr
    
    @tessera.function
    def forward(
        self,
        indices: Tensor["B", "S"],
        weights: Optional[Tensor["B", "S"]] = None
    ) -> Tensor["B", "S", "D"]:
        B, S = indices.shape
        output = tessera.zeros(B, S, self.embedding_dim, dtype=self.compute_dtype)
        
        # Gather sparse embeddings
        for b in range(B):
            for s in range(S):
                idx = indices[b, s]
                
                # Get blocks for this embedding
                start_ptr = self.block_ptr[idx]
                end_ptr = self.block_ptr[idx + 1]
                
                if start_ptr < end_ptr:
                    # Reconstruct embedding from blocks
                    embedding = self._reconstruct_embedding(
                        start_ptr, end_ptr
                    )
                    
                    # Apply weight if provided
                    if weights is not None:
                        embedding = embedding * weights[b, s]
                    
                    output[b, s] = embedding
        
        return output
    
    @tessera.kernel
    def sparse_embedding_kernel(
        self,
        indices: Tile["B*S", int32],
        block_data: Tile["N_blocks", "block_size", "block_size", fp8],
        block_indices: Tile["N_blocks", int32],
        block_ptr: Tile["V+1", int32],
        block_scales: Tile["N_blocks", f32],
        output: Tile["B*S", "D", bf16]
    ):
        """Optimized kernel for sparse embedding gathering"""
        ctx = tile.context()
        
        # Process each lookup
        for idx in tile.grid(ctx.BS):
            emb_idx = indices[idx]
            
            # Get block range
            start = block_ptr[emb_idx]
            end = block_ptr[emb_idx + 1]
            
            # Initialize output
            out_vec = tile.zeros(ctx.D, bf16)
            
            # Gather and decompress blocks
            for block_idx in tile.range(start, end):
                # Load compressed block
                block_fp8 = tile.load(block_data[block_idx])
                scale = block_scales[block_idx]
                
                # Decompress: cast and scale
                block_bf16 = tile.cast(block_fp8, bf16)
                block_scaled = tile.mul(block_bf16, scale)
                
                # Get position in output
                col_idx = block_indices[block_idx]
                offset = col_idx * ctx.block_size
                
                # Copy to output position
                tile.copy(
                    block_scaled.reshape(-1),
                    out_vec[offset:offset + ctx.block_size * ctx.block_size]
                )
            
            # Store result
            tile.store(output[idx], out_vec)
```
## Chapter 6: Loss Functions

### 6.1 Stable Cross-Entropy

```python
@tessera.function
def stable_cross_entropy(
    logits: Tensor["B", "S", "V"],
    targets: Tensor["B", "S"],
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean"
) -> Tensor:
    """
    Numerically stable cross-entropy loss
    Handles large vocabularies and extreme logit values
    """
    B, S, V = logits.shape
    
    # Reshape for processing
    logits_flat = logits.reshape(-1, V)  # [B*S, V]
    targets_flat = targets.reshape(-1)    # [B*S]
    
    # Mask for ignored indices
    mask = targets_flat != ignore_index
    
    # Stable log-softmax computation
    log_probs = tessera.nn.log_softmax_stable(logits_flat, dim=-1)
    
    # Label smoothing if needed
    if label_smoothing > 0:
        log_probs = apply_label_smoothing(log_probs, label_smoothing)
    
    # Gather target log probabilities
    target_log_probs = tessera.gather(
        log_probs,
        targets_flat.unsqueeze(1),
        dim=1
    ).squeeze(1)
    
    # Apply mask
    loss = -target_log_probs * mask
    
    # Reduction
    if reduction == "mean":
        return loss.sum() / mask.sum()
    elif reduction == "sum":
        return loss.sum()
    else:  # "none"
        return loss.reshape(B, S)

@tessera.numerically_stable
def log_softmax_stable(
    x: Tensor["B", "V"],
    dim: int = -1
) -> Tensor["B", "V"]:
    """
    Numerically stable log-softmax
    Avoids overflow/underflow in exp computation
    """
    # Subtract max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    
    # Compute log-sum-exp
    exp_x = tessera.exp(x_shifted)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    log_sum_exp = tessera.log(sum_exp)
    
    # Return log probabilities
    return x_shifted - log_sum_exp

@tessera.kernel
def stable_cross_entropy_kernel(
    logits: Tile["B*S", "V", bf16],
    targets: Tile["B*S", int32],
    loss: Tile["B*S", f32],
    vocab_size: int,
    ignore_index: int = -100
):
    """
    Fused kernel for stable cross-entropy computation
    Combines log-softmax and gathering in single pass
    """
    ctx = tile.context()
    
    for idx in tile.grid(ctx.BS):
        target = targets[idx]
        
        # Skip if ignore index
        if target == ignore_index:
            loss[idx] = 0.0
            continue
        
        # Load logits for this position
        logits_vec = tile.load(logits[idx], dtype=bf16)
        
        # Find max for stability (warp reduction)
        max_logit = tile.warp.reduce_max(logits_vec)
        max_logit = tile.broadcast(max_logit)
        
        # Compute exp(x - max) 
        logits_vec = tile.sub(logits_vec, max_logit)
        exp_vec = tile.exp(tile.cast(logits_vec, f32))
        
        # Sum of exponentials (warp reduction)
        sum_exp = tile.warp.reduce_sum(exp_vec)
        sum_exp = tile.broadcast(sum_exp)
        
        # Log-sum-exp
        log_sum_exp = tile.log(sum_exp) + max_logit
        
        # Get target logit
        target_logit = logits_vec[target]
        
        # Compute loss: -(target_logit - log_sum_exp)
        loss_val = -(target_logit - log_sum_exp)
        
        # Store result
        loss[idx] = loss_val

# Stable max margin loss
@tessera.function
def stable_max_margin_loss(
    scores: Tensor["B", "N"],
    targets: Tensor["B"],
    margin: float = 1.0
) -> Tensor:
    """
    Stable computation of max margin loss
    Used in ranking and metric learning
    """
    B, N = scores.shape
    
    # Get positive scores
    pos_scores = tessera.gather(scores, targets.unsqueeze(1), dim=1)
    
    # Mask out positive class
    mask = tessera.ones_like(scores)
    mask.scatter_(1, targets.unsqueeze(1), 0)
    
    # Add margin to negative scores
    scores_with_margin = scores + margin * mask
    
    # Stable max computation
    neg_scores_max = tessera.where(
        mask.bool(),
        scores_with_margin,
        tessera.tensor(float('-inf'))
    ).max(dim=1).values
    
    # Hinge loss
    loss = tessera.relu(neg_scores_max - pos_scores.squeeze(1))
    
    return loss.mean()
```
### 6.2 Softmax Cross-Entropy (Combined)

```python
@tessera.function
def softmax_cross_entropy(
    logits: Tensor["B", "C"],
    targets: Tensor["B"],
    reduction: str = "mean"
) -> Tensor:
    """
    Combined softmax and cross-entropy for efficiency
    Fuses operations to avoid storing intermediate softmax
    """
    # Use fused kernel
    return tessera.ops.fused_softmax_cross_entropy(
        logits, targets, reduction
    )

@tessera.kernel
def fused_softmax_cross_entropy_kernel(
    logits: Tile["B", "C", bf16],
    targets: Tile["B", int32],
    loss: Tile["B", f32],
    num_classes: int
):
    """
    Fused softmax + cross-entropy kernel
    More efficient than separate operations
    """
    ctx = tile.context()
    
    for b in tile.grid(ctx.B):
        # Load logits row
        logits_row = tile.load(logits[b], dtype=f32)
        target = targets[b]
        
        # Online softmax computation
        max_val = tile.reduce_max(logits_row)
        
        # Compute exp(x - max) and sum
        sum_exp = 0.0
        target_exp = 0.0
        
        for c in tile.range(num_classes):
            exp_val = tile.exp(logits_row[c] - max_val)
            sum_exp += exp_val
            
            if c == target:
                target_exp = exp_val
        
        # Cross entropy: -log(softmax[target])
        loss[b] = -tile.log(target_exp / sum_exp)

# Contrastive loss with numerical stability
@tessera.function
def stable_contrastive_loss(
    embeddings: Tensor["B", "D"],
    labels: Tensor["B"],
    temperature: float = 0.07,
    normalize: bool = True
) -> Tensor:
    """
    Stable contrastive loss (SimCLR/MoCo style)
    Handles large batch sizes and extreme similarities
    """
    B, D = embeddings.shape
    
    # Normalize embeddings if requested
    if normalize:
        embeddings = tessera.nn.l2_normalize(embeddings, dim=1)
    
    # Compute similarity matrix
    sim_matrix = embeddings @ embeddings.T / temperature
    
    # Mask out diagonal
    mask = ~tessera.eye(B, dtype=bool)
    sim_matrix = sim_matrix.masked_fill(~mask, float('-inf'))
    
    # Find positive pairs
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = labels_equal & mask
    
    # Stable log-sum-exp for denominator
    max_sim = sim_matrix.max(dim=1, keepdim=True).values
    exp_sim = tessera.exp(sim_matrix - max_sim)
    sum_exp_sim = exp_sim.sum(dim=1)
    
    # Compute loss for positive pairs
    pos_sim = sim_matrix.masked_fill(~positive_mask, float('-inf'))
    pos_exp = tessera.exp(pos_sim - max_sim)
    
    # InfoNCE loss
    loss = -tessera.log(pos_exp.sum(dim=1) / sum_exp_sim)
    
    return loss.mean()
```
## Chapter 7: Advanced Optimizations

### 7.1 Fused Operations

```python
@tessera.fused
class FusedTransformerLayer:
    """
    Fully fused transformer layer
    Combines all operations into minimal kernel launches
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = int(dim * mlp_ratio)
        
        # Fused QKV projection
        self.qkv = CastedLinear(dim, 3 * dim, weight_dtype=int8)
        
        # Output projection
        self.out_proj = CastedLinear(dim, dim, weight_dtype=int8)
        
        # Fused MLP weights
        self.mlp_gate_up = CastedLinear(
            dim, 2 * self.mlp_dim, 
            weight_dtype=int8
        )
        self.mlp_down = CastedLinear(
            self.mlp_dim, dim,
            weight_dtype=int8
        )
        
        # Norms
        self.norm1 = rms_norm
        self.norm2 = rms_norm
        
        self.dropout = dropout
    
    @tessera.function
    def forward(
        self,
        x: Tensor["B", "S", "D"],
        position_ids: Optional[Tensor["B", "S"]] = None,
        attention_mask: Optional[Tensor["B", "S", "S"]] = None
    ) -> Tensor["B", "S", "D"]:
        # Everything fused into 2 kernels
        
        # Kernel 1: Norm1 + QKV + RoPE + Attention + Proj
        hidden = self.fused_attention_block(
            x, position_ids, attention_mask
        )
        x = x + hidden
        
        # Kernel 2: Norm2 + MLP (Gate + Up + Down)
        hidden = self.fused_mlp_block(x)
        x = x + hidden
        
        return x
    
    @tessera.kernel.fused
    def fused_attention_block(
        self,
        x: Tile["B*S", "D", bf16],
        rope_cos: Tile["S", "D/2", f32],
        rope_sin: Tile["S", "D/2", f32],
        mask: Optional[Tile["S", "S"]],
        output: Tile["B*S", "D", bf16]
    ):
        """Single kernel for entire attention block"""
        ctx = tile.context()
        
        # RMSNorm (in registers)
        x_norm = tile.rms_norm(x, eps=1e-6)
        
        # QKV projection (tensor cores)
        qkv = tile.mma(x_norm, self.qkv.weight)
        q, k, v = tile.split(qkv, 3, dim=-1)
        
        # Reshape for heads
        q = tile.reshape(q, (ctx.B, ctx.S, self.num_heads, -1))
        k = tile.reshape(k, (ctx.B, ctx.S, self.num_heads, -1))
        v = tile.reshape(v, (ctx.B, ctx.S, self.num_heads, -1))
        
        # Apply RoPE (in registers)
        q = tile.apply_rope(q, rope_cos, rope_sin)
        k = tile.apply_rope(k, rope_cos, rope_sin)
        
        # Flash attention (tiled computation)
        attn_out = tile.flash_attention(
            q, k, v,
            causal=True,
            dropout=self.dropout
        )
        
        # Output projection
        attn_out = tile.reshape(attn_out, (ctx.BS, ctx.D))
        output_tile = tile.mma(attn_out, self.out_proj.weight)
        
        # Store with residual (done by caller)
        tile.store(output, output_tile)
    
    @tessera.kernel.fused
    def fused_mlp_block(
        self,
        x: Tile["B*S", "D", bf16],
        output: Tile["B*S", "D", bf16]
    ):
        """Single kernel for entire MLP block"""
        ctx = tile.context()
        
        # RMSNorm
        x_norm = tile.rms_norm(x, eps=1e-6)
        
        # Fused Gate + Up projection
        gate_up = tile.mma(x_norm, self.mlp_gate_up.weight)
        gate, up = tile.split(gate_up, 2, dim=-1)
        
        # SwiGLU activation
        gate = tile.swish(gate)
        hidden = tile.mul(gate, up)
        
        # Down projection
        output_tile = tile.mma(hidden, self.mlp_down.weight)
        
        # Store
        tile.store(output, output_tile)
```
### 7.2 Memory-Efficient Variants

```python
@tessera.memory_efficient
class MemoryEfficientAttention:
    """
    Memory-efficient attention implementations
    """
    
    @staticmethod
    @tessera.function
    def chunked_attention(
        q: Tensor["B", "H", "S", "D"],
        k: Tensor["B", "H", "S", "D"],
        v: Tensor["B", "H", "S", "D"],
        chunk_size: int = 512
    ) -> Tensor["B", "H", "S", "D"]:
        """
        Process attention in chunks to reduce memory
        """
        B, H, S, D = q.shape
        output = tessera.zeros_like(q)
        
        # Process query chunks
        for i in range(0, S, chunk_size):
            q_chunk = q[:, :, i:i+chunk_size]
            
            # Initialize accumulator for this chunk
            attn_output = tessera.zeros_like(q_chunk)
            normalizer = tessera.zeros(
                B, H, q_chunk.shape[2], 1
            )
            
            # Process key/value chunks
            for j in range(0, S, chunk_size):
                k_chunk = k[:, :, j:j+chunk_size]
                v_chunk = v[:, :, j:j+chunk_size]
                
                # Compute attention scores for this block
                scores = (q_chunk @ k_chunk.transpose(-2, -1)) / math.sqrt(D)
                
                # Causal mask if needed
                if i >= j:  # Only attend to previous positions
                    scores = apply_causal_mask(scores, i, j)
                
                # Stable softmax
                scores_max = scores.max(dim=-1, keepdim=True).values
                scores_exp = tessera.exp(scores - scores_max)
                
                # Accumulate
                attn_output += scores_exp @ v_chunk
                normalizer += scores_exp.sum(dim=-1, keepdim=True)
            
            # Normalize
            output[:, :, i:i+chunk_size] = attn_output / normalizer
        
        return output
    
    @staticmethod
    @tessera.function
    def ring_attention(
        q: Tensor["B", "H", "S", "D"],
        k: Tensor["B", "H", "S", "D"],
        v: Tensor["B", "H", "S", "D"],
        mesh: Mesh
    ) -> Tensor["B", "H", "S", "D"]:
        """
        Ring attention for extremely long sequences
        Distributes sequence across devices in a ring
        """
        # Shard sequence across devices
        local_s = S // mesh.size()
        
        # Local chunks
        q_local = q[:, :, mesh.rank*local_s:(mesh.rank+1)*local_s]
        k_local = k[:, :, mesh.rank*local_s:(mesh.rank+1)*local_s]
        v_local = v[:, :, mesh.rank*local_s:(mesh.rank+1)*local_s]
        
        output = tessera.zeros_like(q_local)
        
        # Ring communication
        for step in range(mesh.size()):
            # Compute local attention
            attn = attention(q_local, k_local, v_local)
            output += attn
            
            # Rotate KV chunks around ring
            k_local = tessera.ring_exchange(k_local, mesh)
            v_local = tessera.ring_exchange(v_local, mesh)
        
        # Gather results
        return tessera.all_gather(output, mesh, dim=2)
```
## Conclusion

This comprehensive implementation guide shows how Tessera provides:

- Numerical Stability - All operations designed to handle extreme values, mixed precision, and gradient stability
- Performance - Fused kernels, optimized memory access patterns, and hardware-specific optimizations
- Flexibility - Support for various precision levels, sparsity patterns, and distributed execution
- Correctness - Type-safe operations with compile-time shape checking

Each operation is provided in multiple variants:

- High-level functional API for ease of use
- Optimized kernels for production performance
- Memory-efficient versions for resource-constrained scenarios
- Distributed variants for large-scale training

The key innovation is that Tessera makes these advanced optimizations accessible through simple APIs while maintaining the ability to drop down to low-level control when needed.