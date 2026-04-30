"""
Vanilla Transformer Tutorial in Tessera Programming Model
=========================================================

This tutorial demonstrates how to build a complete Transformer model using Tessera,
showcasing the programming model's key features while building up complexity step by step.

What you'll learn:
1. Tessera's shape-semantic type system
2. Automatic differentiation and optimization
3. Hardware-agnostic kernel compilation
4. Multi-head attention implementation
5. Complete transformer architecture
6. Training loop with autotuning
7. Performance optimization techniques

Prerequisites:
- Basic understanding of Transformers and attention mechanisms
- Familiarity with tensor operations
- Python programming experience

Let's build a Transformer from scratch in Tessera!
"""

import tessera
from tessera import Tensor, Module
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any

# ============================================================================
# STEP 1: Basic Building Blocks
# ============================================================================

print("STEP 1: Creating basic building blocks...")

@tessera.function
def scaled_dot_product_attention(
    query: Tensor["B", "H", "S", "D"],     # Batch, Heads, Sequence, Dimension
    key: Tensor["B", "H", "S", "D"],
    value: Tensor["B", "H", "S", "D"],
    mask: Optional[Tensor["B", "S", "S"]] = None,
    dropout_rate: float = 0.1,
    is_training: bool = True
) -> Tensor["B", "H", "S", "D"]:
    """
    Step 1a: Scaled Dot-Product Attention
    
    This is the core attention mechanism. Notice how Tessera's type system
    explicitly tracks tensor dimensions - this prevents shape errors at compile time!
    """
    
    # Get dimension for scaling
    d_k = query.shape[-1]
    scale = 1.0 / math.sqrt(d_k)
    
    # Compute attention scores: Q @ K^T
    # Tessera automatically optimizes this matrix multiplication
    scores = tessera.matmul(query, key, transpose_b=True)  # [B, H, S, S]
    scores = scores * scale
    
    # Apply causal mask for decoder self-attention
    if mask is not None:
        # Tessera handles broadcasting automatically
        scores = tessera.where(mask, scores, -tessera.inf)
    
    # Softmax with numerical stability (Tessera automatically uses online softmax)
    attention_weights = tessera.softmax(scores, axis=-1)  # [B, H, S, S]
    
    # Apply dropout during training
    if is_training and dropout_rate > 0.0:
        attention_weights = tessera.dropout(attention_weights, rate=dropout_rate)
    
    # Apply attention to values: Attention @ V
    output = tessera.matmul(attention_weights, value)  # [B, H, S, D]
    
    return output

print("✓ Scaled dot-product attention implemented with shape safety!")

# ============================================================================
# STEP 2: Multi-Head Attention Layer
# ============================================================================

print("\nSTEP 2: Building Multi-Head Attention...")

@tessera.module
class MultiHeadAttention:
    """
    Step 2: Multi-Head Attention Module
    
    This demonstrates Tessera's module system and automatic parameter management.
    Notice how we don't need to manually track parameters - Tessera handles this!
    """
    
    def __init__(
        self,
        d_model: int = 512,        # Model dimension
        num_heads: int = 8,        # Number of attention heads
        dropout_rate: float = 0.1  # Dropout rate
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V (Tessera automatically optimizes these)
        self.w_q = tessera.Linear(d_model, d_model, name="query_projection")
        self.w_k = tessera.Linear(d_model, d_model, name="key_projection") 
        self.w_v = tessera.Linear(d_model, d_model, name="value_projection")
        self.w_o = tessera.Linear(d_model, d_model, name="output_projection")
    
    @tessera.function
    def __call__(
        self,
        query: Tensor["B", "S", "D"],
        key: Tensor["B", "S", "D"],
        value: Tensor["B", "S", "D"],
        mask: Optional[Tensor["B", "S", "S"]] = None,
        is_training: bool = True
    ) -> Tensor["B", "S", "D"]:
        """
        Forward pass of multi-head attention
        
        Tessera's type system tracks shapes through all transformations,
        catching errors before they cause runtime issues!
        """
        
        batch_size, seq_len, d_model = query.shape
        
        # Linear projections
        Q = self.w_q(query)  # [B, S, D]
        K = self.w_k(key)    # [B, S, D]
        V = self.w_v(value)  # [B, S, D]
        
        # Reshape for multi-head attention: [B, S, D] -> [B, H, S, D_k]
        Q = tessera.reshape(Q, (batch_size, seq_len, self.num_heads, self.d_k))
        K = tessera.reshape(K, (batch_size, seq_len, self.num_heads, self.d_k))
        V = tessera.reshape(V, (batch_size, seq_len, self.num_heads, self.d_k))
        
        # Transpose to [B, H, S, D_k] for attention computation
        Q = tessera.transpose(Q, (0, 2, 1, 3))
        K = tessera.transpose(K, (0, 2, 1, 3))
        V = tessera.transpose(V, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        # Tessera automatically selects optimal attention implementation:
        # - Flash Attention for long sequences
        # - Standard implementation for short sequences
        # - Hardware-specific optimizations (CUDA cores, Tensor cores, etc.)
        attention_output = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout_rate, is_training
        )  # [B, H, S, D_k]
        
        # Transpose back and reshape: [B, H, S, D_k] -> [B, S, D]
        attention_output = tessera.transpose(attention_output, (0, 2, 1, 3))
        attention_output = tessera.reshape(
            attention_output, 
            (batch_size, seq_len, d_model)
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output

print("✓ Multi-head attention implemented with automatic optimization!")

# ============================================================================
# STEP 3: Feed-Forward Network
# ============================================================================

print("\nSTEP 3: Creating Feed-Forward Network...")

@tessera.module
class FeedForwardNetwork:
    """
    Step 3: Position-wise Feed-Forward Network
    
    Demonstrates Tessera's automatic kernel fusion - the linear->activation->linear
    sequence will be automatically fused into a single kernel for efficiency!
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,          # Feed-forward dimension (usually 4x d_model)
        activation: str = "relu",   # Activation function
        dropout_rate: float = 0.1
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Two linear layers with tessera's optimized implementations
        self.linear1 = tessera.Linear(d_model, d_ff, name="ff_expand")
        self.linear2 = tessera.Linear(d_ff, d_model, name="ff_contract")
    
    @tessera.function
    @tessera.fused_layer  # This decorator tells Tessera to fuse operations
    def __call__(
        self,
        x: Tensor["B", "S", "D"],
        is_training: bool = True
    ) -> Tensor["B", "S", "D"]:
        """
        Forward pass with automatic fusion
        
        Tessera will automatically fuse: linear1 -> activation -> dropout -> linear2
        into optimized kernels, reducing memory bandwidth requirements!
        """
        
        # First linear transformation
        hidden = self.linear1(x)  # [B, S, D] -> [B, S, D_ff]
        
        # Activation function (Tessera supports many optimized activations)
        if self.activation == "relu":
            hidden = tessera.relu(hidden)
        elif self.activation == "gelu":
            hidden = tessera.gelu(hidden)  # More common in modern transformers
        elif self.activation == "swiglu":
            # SwiGLU activation (used in LLaMA, PaLM, etc.)
            hidden = tessera.swiglu(hidden)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        # Dropout
        if is_training and self.dropout_rate > 0.0:
            hidden = tessera.dropout(hidden, rate=self.dropout_rate)
        
        # Second linear transformation  
        output = self.linear2(hidden)  # [B, S, D_ff] -> [B, S, D]
        
        return output

print("✓ Feed-forward network with automatic kernel fusion!")

# ============================================================================
# STEP 4: Layer Normalization and Residual Connections
# ============================================================================

print("\nSTEP 4: Adding normalization and residual connections...")

@tessera.function
@tessera.numerically_stable  # Ensures stable computation even with extreme values
def layer_normalization(
    x: Tensor["B", "S", "D"],
    gamma: Tensor["D"],
    beta: Tensor["D"],
    eps: float = 1e-6
) -> Tensor["B", "S", "D"]:
    """
    Step 4a: Layer Normalization
    
    Tessera automatically uses numerically stable implementations that
    handle edge cases and extreme values gracefully.
    """
    
    # Compute mean and variance along the last dimension
    # Tessera uses Welford's algorithm for numerical stability
    mean = tessera.mean(x, axis=-1, keepdims=True)
    variance = tessera.var(x, axis=-1, keepdims=True)
    
    # Normalize
    normalized = (x - mean) / tessera.sqrt(variance + eps)
    
    # Scale and shift
    return gamma * normalized + beta

@tessera.function
def residual_connection(
    x: Tensor["B", "S", "D"],
    sublayer_output: Tensor["B", "S", "D"],
    dropout_rate: float = 0.1,
    is_training: bool = True
) -> Tensor["B", "S", "D"]:
    """
    Step 4b: Residual Connection with Dropout
    
    Implements: x + Dropout(Sublayer(LayerNorm(x)))
    """
    
    if is_training and dropout_rate > 0.0:
        sublayer_output = tessera.dropout(sublayer_output, rate=dropout_rate)
    
    return x + sublayer_output

print("✓ Normalization and residual connections with numerical stability!")

# ============================================================================
# STEP 5: Positional Encoding  
# ============================================================================

print("\nSTEP 5: Implementing positional encoding...")

@tessera.function
def sinusoidal_positional_encoding(
    max_seq_len: int,
    d_model: int,
    base: float = 10000.0
) -> Tensor["S", "D"]:
    """
    Step 5: Sinusoidal Positional Encoding
    
    Creates the classic sin/cos positional embeddings.
    Tessera pre-computes this and caches it for efficiency.
    """
    
    # Create position indices
    positions = tessera.arange(max_seq_len, dtype=tessera.float32)  # [S]
    
    # Create dimension indices  
    dimensions = tessera.arange(d_model // 2, dtype=tessera.float32)  # [D/2]
    
    # Compute frequency scaling
    frequencies = 1.0 / (base ** (2 * dimensions / d_model))  # [D/2]
    
    # Compute angles: position * frequency
    angles = tessera.outer(positions, frequencies)  # [S, D/2]
    
    # Apply sin and cos
    sin_encodings = tessera.sin(angles)  # [S, D/2]
    cos_encodings = tessera.cos(angles)  # [S, D/2]
    
    # Interleave sin and cos: [sin, cos, sin, cos, ...]
    pos_encoding = tessera.stack([sin_encodings, cos_encodings], axis=-1)  # [S, D/2, 2]
    pos_encoding = tessera.reshape(pos_encoding, (max_seq_len, d_model))  # [S, D]
    
    return pos_encoding

@tessera.module
class PositionalEncoding:
    """
    Learnable positional encoding module
    Alternative to sinusoidal encoding
    """
    
    def __init__(self, max_seq_len: int = 2048, d_model: int = 512):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Pre-compute sinusoidal encodings as default
        self.register_buffer(
            "pos_encoding",
            sinusoidal_positional_encoding(max_seq_len, d_model)
        )
    
    @tessera.function
    def __call__(self, x: Tensor["B", "S", "D"]) -> Tensor["B", "S", "D"]:
        """Add positional encoding to input embeddings"""
        
        batch_size, seq_len, d_model = x.shape
        
        # Get positional encodings for current sequence length
        pos_enc = self.pos_encoding[:seq_len, :]  # [S, D]
        
        # Add to input (broadcasting handled automatically)
        return x + pos_enc

print("✓ Positional encoding with sinusoidal patterns!")

# ============================================================================
# STEP 6: Transformer Block (Encoder Layer)
# ============================================================================

print("\nSTEP 6: Building the Transformer Block...")

@tessera.module  
class TransformerBlock:
    """
    Step 6: Complete Transformer Encoder Block
    
    This combines all our previous components into a full transformer layer.
    Tessera will automatically optimize the entire block as a unit!
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        norm_first: bool = True  # Pre-norm vs post-norm
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.norm_first = norm_first
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        # Layer normalization (Tessera automatically optimizes LayerNorm)
        self.norm1 = tessera.LayerNorm(d_model, name="attention_norm")
        self.norm2 = tessera.LayerNorm(d_model, name="ff_norm")
    
    @tessera.function
    @tessera.fused_transformer_block  # Tessera-specific optimization
    def __call__(
        self,
        x: Tensor["B", "S", "D"],
        mask: Optional[Tensor["B", "S", "S"]] = None,
        is_training: bool = True
    ) -> Tensor["B", "S", "D"]:
        """
        Transformer block forward pass
        
        Tessera automatically fuses this entire sequence into optimized kernels,
        minimizing memory transfers and maximizing hardware utilization!
        """
        
        if self.norm_first:
            # Pre-norm architecture (more stable for deep networks)
            
            # Self-attention with residual connection
            normed_x = self.norm1(x)
            attention_output = self.self_attention(
                normed_x, normed_x, normed_x, mask, is_training
            )
            x = residual_connection(x, attention_output, self.dropout_rate, is_training)
            
            # Feed-forward with residual connection
            normed_x = self.norm2(x)
            ff_output = self.feed_forward(normed_x, is_training)
            x = residual_connection(x, ff_output, self.dropout_rate, is_training)
            
        else:
            # Post-norm architecture (original Transformer paper)
            
            # Self-attention with residual and norm
            attention_output = self.self_attention(x, x, x, mask, is_training)
            x = self.norm1(residual_connection(x, attention_output, self.dropout_rate, is_training))
            
            # Feed-forward with residual and norm
            ff_output = self.feed_forward(x, is_training)
            x = self.norm2(residual_connection(x, ff_output, self.dropout_rate, is_training))
        
        return x

print("✓ Complete Transformer block with automatic fusion optimization!")

# ============================================================================
# STEP 7: Token Embeddings and Input Processing
# ============================================================================

print("\nSTEP 7: Creating token embeddings...")

@tessera.module
class TokenEmbedding:
    """
    Step 7: Token Embedding Layer
    
    Converts token IDs to dense vectors. Tessera optimizes embedding lookups
    and can automatically use different strategies based on vocabulary size.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        padding_idx: Optional[int] = 0
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # Embedding table (Tessera automatically optimizes memory layout)
        self.embedding = tessera.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
            name="token_embedding"
        )
        
        # Scaling factor (as per original Transformer paper)
        self.scale = math.sqrt(d_model)
    
    @tessera.function
    def __call__(self, token_ids: Tensor["B", "S"]) -> Tensor["B", "S", "D"]:
        """Convert token IDs to embeddings"""
        
        # Embedding lookup with automatic optimization
        embeddings = self.embedding(token_ids)  # [B, S, D]
        
        # Scale embeddings (helps with training stability)
        embeddings = embeddings * self.scale
        
        return embeddings

print("✓ Token embeddings with automatic memory optimization!")

# ============================================================================
# STEP 8: Complete Transformer Model
# ============================================================================

print("\nSTEP 8: Assembling the complete Transformer...")

@tessera.model
class VanillaTransformer:
    """
    Step 8: Complete Vanilla Transformer Model
    
    This brings together all our components into a full transformer.
    Notice the @tessera.model decorator - this enables advanced optimizations
    like automatic mixed precision, gradient checkpointing, and more!
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        max_seq_len: int = 2048,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        norm_first: bool = True,
        tie_embeddings: bool = True  # Tie input/output embeddings
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.tie_embeddings = tie_embeddings
        
        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=0
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            max_seq_len=max_seq_len,
            d_model=d_model
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                activation=activation,
                norm_first=norm_first
            )
            for layer_idx in range(num_layers)
        ]
        
        # Final layer norm (for pre-norm architecture)
        if norm_first:
            self.final_norm = tessera.LayerNorm(d_model, name="final_norm")
        
        # Output projection to vocabulary
        if tie_embeddings:
            # Share weights between input and output embeddings
            self.output_projection = tessera.TiedLinear(
                tied_to=self.token_embedding.embedding,
                bias=False,
                name="output_projection"
            )
        else:
            self.output_projection = tessera.Linear(
                d_model, vocab_size,
                bias=False,
                name="output_projection"
            )
    
    @tessera.function
    def create_causal_mask(
        self,
        seq_len: int,
        batch_size: int
    ) -> Tensor["B", "S", "S"]:
        """
        Create causal attention mask for autoregressive generation
        
        Tessera optimizes mask creation and caches common patterns
        """
        
        # Create lower triangular mask
        mask = tessera.tril(tessera.ones((seq_len, seq_len)))  # [S, S]
        
        # Expand for batch dimension
        mask = tessera.expand_dims(mask, axis=0)  # [1, S, S]
        mask = tessera.broadcast_to(mask, (batch_size, seq_len, seq_len))  # [B, S, S]
        
        return mask.astype(tessera.bool)
    
    @tessera.function
    @tessera.transformer_forward  # Enables transformer-specific optimizations
    def __call__(
        self,
        input_ids: Tensor["B", "S"],
        attention_mask: Optional[Tensor["B", "S", "S"]] = None,
        is_training: bool = True,
        use_cache: bool = False
    ) -> Tensor["B", "S", "V"]:
        """
        Complete transformer forward pass
        
        Tessera automatically applies optimizations like:
        - Attention pattern optimization
        - Memory-efficient attention for long sequences  
        - Automatic mixed precision
        - Gradient checkpointing when needed
        """
        
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings + positional encoding
        embeddings = self.token_embedding(input_ids)  # [B, S, D]
        embeddings = self.positional_encoding(embeddings)  # [B, S, D]
        
        # Input dropout
        if is_training and self.dropout_rate > 0.0:
            embeddings = tessera.dropout(embeddings, rate=self.dropout_rate)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self.create_causal_mask(seq_len, batch_size)
        
        # Pass through transformer blocks
        hidden_states = embeddings
        
        # Tessera can automatically checkpoint intermediate activations for memory efficiency
        with tessera.gradient_checkpointing(enabled=is_training and seq_len > 1024):
            for layer_idx, transformer_block in enumerate(self.transformer_blocks):
                hidden_states = transformer_block(
                    hidden_states,
                    mask=attention_mask,
                    is_training=is_training
                )
        
        # Final layer norm (for pre-norm architecture)
        if hasattr(self, 'final_norm'):
            hidden_states = self.final_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.output_projection(hidden_states)  # [B, S, V]
        
        return logits
    
    @tessera.function
    @tessera.autoregressive_generation
    def generate(
        self,
        input_ids: Tensor["B", "S"],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0
    ) -> Tensor["B", "S_new"]:
        """
        Autoregressive text generation
        
        Tessera provides optimized generation with KV-caching and
        specialized kernels for inference workloads.
        """
        
        batch_size, input_length = input_ids.shape
        
        # Initialize with input tokens
        generated_ids = input_ids
        
        # Generation loop with KV-caching for efficiency
        with tessera.inference_mode():  # Optimized for inference
            with tessera.kv_cache() as cache:  # Automatic KV caching
                
                for _ in tessera.range(max_new_tokens):
                    # Forward pass (with caching for efficiency)
                    logits = self(
                        generated_ids,
                        is_training=False,
                        use_cache=True
                    )
                    
                    # Get logits for last position
                    next_token_logits = logits[:, -1, :]  # [B, V]
                    
                    # Apply temperature scaling
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply repetition penalty
                    if repetition_penalty != 1.0:
                        next_token_logits = tessera.apply_repetition_penalty(
                            next_token_logits,
                            generated_ids,
                            penalty=repetition_penalty
                        )
                    
                    # Sample next token
                    if top_k is not None or top_p is not None:
                        # Top-k or top-p sampling
                        next_token_probs = tessera.top_k_top_p_sampling(
                            next_token_logits,
                            top_k=top_k,
                            top_p=top_p
                        )
                    else:
                        # Standard sampling
                        next_token_probs = tessera.softmax(next_token_logits, axis=-1)
                    
                    # Sample from distribution
                    next_tokens = tessera.multinomial(
                        next_token_probs,
                        num_samples=1
                    )  # [B, 1]
                    
                    # Append to generated sequence
                    generated_ids = tessera.concatenate(
                        [generated_ids, next_tokens],
                        axis=1
                    )
        
        return generated_ids

print("✓ Complete Transformer with optimized generation!")

# ============================================================================
# STEP 9: Training Infrastructure
# ============================================================================

print("\nSTEP 9: Setting up training infrastructure...")

@tessera.trainer
class TransformerTrainer:
    """
    Step 9: Training System for Transformer
    
    Tessera provides comprehensive training infrastructure with automatic
    optimizations, mixed precision, and performance monitoring.
    """
    
    def __init__(
        self,
        model: VanillaTransformer,
        learning_rate: float = 1e-4,
        warmup_steps: int = 4000,
        max_steps: int = 100000,
        gradient_clip_norm: float = 1.0
    ):
        self.model = model
        self.max_steps = max_steps
        
        # Tessera's optimized Adam with transformer-specific defaults
        self.optimizer = tessera.optimizers.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),    # Transformer paper defaults
            eps=1e-9,
            weight_decay=0.01,
            gradient_clip_norm=gradient_clip_norm
        )
        
        # Transformer learning rate schedule (warmup + decay)
        self.scheduler = tessera.schedulers.TransformerSchedule(
            optimizer=self.optimizer,
            d_model=model.d_model,
            warmup_steps=warmup_steps,
            max_steps=max_steps
        )
        
        # Automatic mixed precision for speed + stability
        self.amp_scaler = tessera.AMPScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        
        # Training metrics tracking
        self.metrics_tracker = tessera.MetricsTracker()
    
    @tessera.function
    @tessera.training_step
    def train_step(
        self,
        batch: Dict[str, Tensor],
        step: int
    ) -> Dict[str, float]:
        """
        Single training step with automatic optimization
        
        Tessera handles all the complex details:
        - Automatic differentiation
        - Mixed precision training
        - Gradient accumulation
        - Memory optimization
        """
        
        # Unpack batch
        input_ids = batch["input_ids"]      # [B, S]
        target_ids = batch["target_ids"]    # [B, S]
        attention_mask = batch.get("attention_mask", None)
        
        # Forward pass with automatic mixed precision
        with tessera.autocast():
            logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                is_training=True
            )  # [B, S, V]
            
            # Compute cross-entropy loss
            # Tessera automatically uses label smoothing and numerical stability
            loss = tessera.cross_entropy_loss(
                logits.reshape(-1, logits.shape[-1]),  # [B*S, V]
                target_ids.reshape(-1),                # [B*S]
                label_smoothing=0.1,
                ignore_index=0  # Ignore padding tokens
            )
        
        # Backward pass with gradient scaling
        scaled_loss = self.amp_scaler.scale(loss)
        scaled_loss.backward()
        
        # Gradient processing
        self.amp_scaler.unscale_(self.optimizer)
        
        # Compute gradient norm for monitoring
        grad_norm = tessera.grad_norm(self.model.parameters())
        
        # Optimizer step with automatic unscaling
        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()
        self.scheduler.step()
        
        # Calculate metrics
        with tessera.no_grad():
            # Accuracy calculation
            predictions = tessera.argmax(logits, axis=-1)
            accuracy = tessera.mean(
                (predictions == target_ids).float(),
                where=(target_ids != 0)  # Ignore padding
            )
            
            # Perplexity
            perplexity = tessera.exp(loss)
        
        return {
            "loss": loss.item(),
            "perplexity": perplexity.item(),
            "accuracy": accuracy.item(),
            "grad_norm": grad_norm.item(),
            "learning_rate": self.scheduler.get_last_lr()
        }
    
    @tessera.function
    @tessera.validation_step
    def validate(
        self,
        validation_dataloader,
        num_batches: int = 100
    ) -> Dict[str, float]:
        """
        Validation loop with automatic optimization
        """
        
        self.model.eval()
        total_metrics = tessera.defaultdict(list)
        
        with tessera.no_grad():
            for batch_idx, batch in enumerate(validation_dataloader):
                if batch_idx >= num_batches:
                    break
                
                # Forward pass
                logits = self.model(
                    batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    is_training=False
                )
                
                # Compute metrics
                loss = tessera.cross_entropy_loss(
                    logits.reshape(-1, logits.shape[-1]),
                    batch["target_ids"].reshape(-1),
                    ignore_index=0
                )
                
                predictions = tessera.argmax(logits, axis=-1)
                accuracy = tessera.mean(
                    (predictions == batch["target_ids"]).float(),
                    where=(batch["target_ids"] != 0)
                )
                
                # Track metrics
                total_metrics["loss"].append(loss.item())
                total_metrics["accuracy"].append(accuracy.item())
                total_metrics["perplexity"].append(tessera.exp(loss).item())
        
        # Average metrics
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in total_metrics.items()
        }
        
        self.model.train()
        return avg_metrics

print("✓ Complete training infrastructure with automatic optimization!")

# ============================================================================
# STEP 10: Data Loading and Preprocessing
# ============================================================================

print("\nSTEP 10: Creating data loading pipeline...")

@tessera.data_pipeline
class TransformerDataLoader:
    """
    Step 10: Efficient Data Loading for Transformer Training
    
    Tessera provides optimized data loading with automatic batching,
    padding, and preprocessing that's aware of the model architecture.
    """
    
    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        max_seq_len: int = 2048,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Tessera's optimized text dataset
        self.dataset = tessera.data.TextDataset(
            dataset_path,
            tokenizer=tokenizer,
            max_length=max_seq_len,
            preprocessing=self._preprocess_text
        )
        
        # Optimized dataloader with prefetching
        self.dataloader = tessera.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=2,
            collate_fn=self._collate_batch,
            pin_memory=True  # Faster GPU transfer
        )
    
    @tessera.function
    def _preprocess_text(self, text: str) -> Dict[str, Any]:
        """Preprocess raw text for training"""
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Create input/target pairs for language modeling
        if len(tokens) > self.max_seq_len:
            # Truncate if too long
            tokens = tokens[:self.max_seq_len]
        
        # Input is tokens[:-1], target is tokens[1:] (next token prediction)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "length": len(input_ids)
        }
    
    @tessera.function
    def _collate_batch(self, batch) -> Dict[str, Tensor]:
        """
        Collate individual examples into batches
        
        Tessera automatically optimizes padding strategies for different
        sequence lengths to minimize wasted computation.
        """
        
        # Extract sequences and lengths
        input_sequences = [item["input_ids"] for item in batch]
        target_sequences = [item["target_ids"] for item in batch]
        lengths = [item["length"] for item in batch]
        
        # Dynamic padding to longest sequence in batch (not max_seq_len)
        # This reduces wasted computation on short sequences
        max_len_in_batch = max(lengths)
        
        # Pad sequences
        padded_inputs = tessera.pad_sequence(
            input_sequences,
            batch_first=True,
            padding_value=0,  # Use 0 as padding token
            max_length=max_len_in_batch
        )
        
        padded_targets = tessera.pad_sequence(
            target_sequences,
            batch_first=True,
            padding_value=0,
            max_length=max_len_in_batch
        )
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = tessera.create_padding_mask(
            padded_inputs,
            padding_value=0
        )
        
        return {
            "input_ids": padded_inputs,
            "target_ids": padded_targets,
            "attention_mask": attention_mask,
            "lengths": tessera.tensor(lengths)
        }
    
    def __iter__(self):
        """Make dataloader iterable"""
        return iter(self.dataloader)

    # Simple tokenizer for demo purposes
    class SimpleTokenizer:
        def __init__(self, vocab_size=30000):
            self.vocab_size = vocab_size
            # Simple character-level tokenizer for demo
            self.char_to_id = {chr(i): i for i in range(32, 127)}  # Printable ASCII
            self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        def encode(self, text):
            return [self.char_to_id.get(c, 1) for c in text[:1000]]  # Limit length
        
        def decode(self, ids):
            return ''.join(self.id_to_char.get(i, '?') for i in ids)

print("✓ Data loading pipeline with dynamic batching!")

# ============================================================================
# STEP 11: Complete Training Loop
# ============================================================================

print("\nSTEP 11: Implementing the complete training loop...")

def train_vanilla_transformer(
    config: Dict[str, Any],
    train_dataset_path: str,
    val_dataset_path: str,
    output_dir: str = "./transformer_checkpoints"
):
    """
    Step 11: Complete training function
    
    This demonstrates a full training pipeline using all Tessera features:
    - Automatic optimization and tuning
    - Advanced mixed precision training
    - Comprehensive logging and monitoring
    - Automatic checkpointing and recovery
    """
    
    print(f"Starting Transformer training with config: {config}")
    
    # Enable Tessera optimizations
    tessera.config.enable_automatic_optimization()
    tessera.config.set_mixed_precision_policy("automatic")
    
    # Initialize model
    print("Initializing Transformer model...")
    model = VanillaTransformer(**config["model"])
    
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Model size: {model.memory_footprint_mb():.1f} MB")
    
    # Setup trainer
    trainer = TransformerTrainer(
        model=model,
        **config["training"]
    )
    
    # Create simple tokenizer for demonstration
    tokenizer = TransformerDataLoader.SimpleTokenizer(config["model"]["vocab_size"])
    
    # Create data loaders
    print("Setting up data loaders...")
    train_loader = TransformerDataLoader(
        dataset_path=train_dataset_path,
        tokenizer=tokenizer,
        max_seq_len=config["model"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        num_workers=4
    )
    
    val_loader = TransformerDataLoader(
        dataset_path=val_dataset_path,
        tokenizer=tokenizer,
        max_seq_len=config["model"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        num_workers=2
    )
    
    # Training loop with comprehensive monitoring
    print("Starting training loop...")
    
    best_val_loss = float('inf')
    step = 0
    
    # Tessera's training loop with automatic optimizations
    with tessera.training_context(
        auto_optimization=True,
        performance_monitoring=True,
        automatic_checkpointing=True
    ):
        
        for epoch in range(config["training"]["num_epochs"]):
            print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
            
            # Training phase
            model.train()
            epoch_metrics = []
            
            for batch_idx, batch in enumerate(train_loader):
                
                # Training step
                step_metrics = trainer.train_step(batch, step)
                epoch_metrics.append(step_metrics)
                step += 1
                
                # Logging
                if step % config["logging"]["log_interval"] == 0:
                    avg_metrics = {
                        key: sum(m[key] for m in epoch_metrics[-config["logging"]["log_interval"]:]) / 
                             min(len(epoch_metrics), config["logging"]["log_interval"])
                        for key in step_metrics.keys()
                    }
                    
                    print(f"Step {step:6d} | "
                          f"Loss: {avg_metrics['loss']:.4f} | "
                          f"PPL: {avg_metrics['perplexity']:.2f} | "
                          f"Acc: {avg_metrics['accuracy']:.3f} | "
                          f"GradNorm: {avg_metrics['grad_norm']:.3f} | "
                          f"LR: {avg_metrics['learning_rate']:.2e}")
                
                # Validation
                if step % config["training"]["eval_interval"] == 0:
                    print(f"\nRunning validation at step {step}...")
                    val_metrics = trainer.validate(val_loader, num_batches=50)
                    
                    print(f"Validation | "
                          f"Loss: {val_metrics['loss']:.4f} | "
                          f"PPL: {val_metrics['perplexity']:.2f} | "
                          f"Acc: {val_metrics['accuracy']:.3f}")
                    
                    # Save best model
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        tessera.save_checkpoint(
                            model,
                            f"{output_dir}/best_model.ckpt",
                            metadata={
                                "step": step,
                                "epoch": epoch,
                                "val_loss": best_val_loss,
                                "config": config
                            }
                        )
                        print(f"✓ New best model saved (val_loss: {best_val_loss:.4f})")
                
                # Early stopping check
                if step >= config["training"]["max_steps"]:
                    print(f"Reached maximum steps ({config['training']['max_steps']})")
                    break
            
            # End of epoch validation
            print(f"\nEnd of epoch {epoch + 1} validation...")
            epoch_val_metrics = trainer.validate(val_loader)
            print(f"Epoch {epoch + 1} validation | "
                  f"Loss: {epoch_val_metrics['loss']:.4f} | "
                  f"PPL: {epoch_val_metrics['perplexity']:.2f}")
            
            # Save epoch checkpoint
            tessera.save_checkpoint(
                model,
                f"{output_dir}/epoch_{epoch}.ckpt",
                metadata={
                    "epoch": epoch,
                    "step": step,
                    "val_metrics": epoch_val_metrics,
                    "config": config
                }
            )
    
    print(f"\n✓ Training completed! Best validation loss: {best_val_loss:.4f}")
    return model

# ============================================================================
# STEP 12: Text Generation and Inference
# ============================================================================

print("\nSTEP 12: Implementing text generation...")

@tessera.inference_engine
class TransformerInferenceEngine:
    """
    Step 12: Optimized Inference Engine
    
    Tessera provides specialized inference optimizations including:
    - KV-caching for fast autoregressive generation
    - Batched inference for throughput
    - Speculative decoding for acceleration
    """
    
    def __init__(
        self,
        model: VanillaTransformer,
        tokenizer,
        device: str = "auto"
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        # Optimize model for inference
        self.model = tessera.optimize_for_inference(
            model,
            optimization_level="aggressive",
            enable_kv_cache=True,
            enable_flash_attention=True
        )
        
        # Move to specified device
        if device == "auto":
            device = "cuda" if tessera.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.device = device
    
    @tessera.function
    @tessera.inference_optimized
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stream: bool = False
    ) -> str:
        """
        Generate text from prompt
        
        Tessera automatically optimizes generation with techniques like:
        - Parallel sampling for multiple sequences
        - Adaptive batching for variable-length generation
        - Memory-efficient attention patterns
        """
        
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = tessera.tensor([input_ids], device=self.device)  # [1, S]
        
        with tessera.inference_mode():
            # Generate with optimized autoregressive decoding
            generated_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            # Decode generated tokens
            generated_text = self.tokenizer.decode(
                generated_ids[0].tolist()  # Remove batch dimension
            )
        
        return generated_text
    
    @tessera.function
    def batch_generate(
        self,
        prompts: list[str],
        **generation_kwargs
    ) -> list[str]:
        """Generate text for multiple prompts efficiently"""
        
        # Tokenize all prompts
        input_ids_list = [self.tokenizer.encode(prompt) for prompt in prompts]
        
        # Pad to same length for batching
        max_len = max(len(ids) for ids in input_ids_list)
        padded_inputs = []
        
        for ids in input_ids_list:
            padded = ids + [0] * (max_len - len(ids))  # Pad with 0
            padded_inputs.append(padded)
        
        input_tensor = tessera.tensor(padded_inputs, device=self.device)
        
        # Batch generation
        with tessera.inference_mode():
            generated_batch = self.model.generate(input_tensor, **generation_kwargs)
        
        # Decode all generated sequences
        generated_texts = []
        for i in range(generated_batch.shape[0]):
            generated_text = self.tokenizer.decode(generated_batch[i].tolist())
            generated_texts.append(generated_text)
        
        return generated_texts

# ============================================================================
# STEP 13: Performance Optimization and Autotuning
# ============================================================================

print("\nSTEP 13: Setting up performance optimization...")

@tessera.performance_optimizer
class TransformerOptimizer:
    """
    Step 13: Automatic Performance Optimization
    
    Tessera can automatically tune hyperparameters, batch sizes, and
    low-level kernel parameters for optimal performance on your hardware.
    """
    
    def __init__(self, model: VanillaTransformer):
        self.model = model
        self.optimization_history = []
    
    @tessera.function
    @tessera.autotune_model
    def optimize_for_hardware(
        self,
        sample_batch,
        target_hardware: str = "auto"
    ) -> Dict[str, Any]:
        """
        Automatically optimize model for target hardware
        
        Tessera analyzes your model and hardware to find optimal configurations
        """
        
        if target_hardware == "auto":
            target_hardware = tessera.detect_hardware()
        
        print(f"Optimizing for hardware: {target_hardware}")
        
        # Tessera's autotuning explores multiple dimensions:
        optimization_space = {
            "batch_size": [8, 16, 32, 64, 128],
            "sequence_length": [512, 1024, 2048],
            "attention_implementation": ["standard", "flash", "memory_efficient"],
            "mixed_precision": [True, False],
            "gradient_checkpointing": [True, False],
            "fusion_level": ["conservative", "aggressive"]
        }
        
        best_config = None
        best_throughput = 0.0
        
        with tessera.autotuning_context():
            for config in tessera.grid_search(optimization_space, max_trials=20):
                
                # Apply configuration
                tessera.apply_optimization_config(self.model, config)
                
                # Benchmark with this configuration
                try:
                    throughput = self._benchmark_configuration(sample_batch, config)
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_config = config
                        
                    print(f"Config {config}: {throughput:.1f} tokens/sec")
                    
                except Exception as e:
                    print(f"Config {config} failed: {e}")
                    continue
        
        # Apply best configuration
        if best_config:
            tessera.apply_optimization_config(self.model, best_config)
            print(f"\n✓ Best configuration applied: {best_throughput:.1f} tokens/sec")
            
        return {
            "best_config": best_config,
            "best_throughput": best_throughput,
            "optimization_history": self.optimization_history
        }
    
    def _benchmark_configuration(
        self,
        sample_batch: Dict[str, Tensor],
        config: Dict[str, Any]
    ) -> float:
        """Benchmark a specific configuration"""
        
        # Warmup
        for _ in range(3):
            with tessera.no_grad():
                _ = self.model(sample_batch["input_ids"])
        
        # Actual benchmark
        with tessera.profiler() as prof:
            for _ in range(10):
                with tessera.no_grad():
                    _ = self.model(sample_batch["input_ids"])
        
        # Calculate throughput
        total_tokens = sample_batch["input_ids"].numel() * 10
        total_time = prof.total_time
        throughput = total_tokens / total_time
        
        return throughput

# ============================================================================
# STEP 14: Complete Example and Demonstration
# ============================================================================

print("\nSTEP 14: Complete working example...")

def main_example():
    """
    Step 14: Complete example that ties everything together
    
    This function demonstrates the entire workflow from model creation
    to training to inference using a small dataset.
    """
    
    print("🚀 TESSERA VANILLA TRANSFORMER TUTORIAL")
    print("=" * 60)
    
    # Configuration for our transformer
    config = {
        "model": {
            "vocab_size": 10000,
            "max_seq_len": 512,
            "d_model": 256,        # Smaller for demo
            "num_layers": 4,       # Smaller for demo
            "num_heads": 8,
            "d_ff": 1024,          # 4x d_model
            "dropout_rate": 0.1,
            "activation": "gelu",   # More modern than ReLU
            "norm_first": True,     # Pre-norm is more stable
            "tie_embeddings": True
        },
        "training": {
            "learning_rate": 1e-4,
            "warmup_steps": 1000,
            "max_steps": 10000,
            "batch_size": 16,      # Small for demo
            "gradient_clip_norm": 1.0,
            "num_epochs": 2,
            "eval_interval": 1000
        },
        "logging": {
            "log_interval": 50,
        }
    }
    
    print("Configuration:")
    for section, values in config.items():
        print(f"  {section}:")
        for key, value in values.items():
            print(f"    {key}: {value}")
    
    # Step 1: Create model
    print(f"\n📦 Creating Transformer model...")
    model = VanillaTransformer(**config["model"])
    
    print(f"✓ Model created successfully!")
    print(f"  - Parameters: {model.num_parameters():,}")
    print(f"  - Memory footprint: {model.memory_footprint_mb():.1f} MB")
    print(f"  - Model architecture: {config['model']['num_layers']} layers, {config['model']['num_heads']} heads")
    
    # Step 2: Generate some sample training data (for demonstration)
    print(f"\n📝 Generating sample training data...")
    
    def generate_sample_data(num_samples: int = 1000, seq_len: int = 128):
        """Generate simple arithmetic data for demonstration"""
        
        # Create simple arithmetic problems: "123 + 456 = 579"
        samples = []
        
        for _ in range(num_samples):
            # Random arithmetic
            a = np.random.randint(1, 1000)
            b = np.random.randint(1, 1000)
            result = a + b
            
            # Convert to text
            text = f"{a} + {b} = {result}"
            samples.append(text)
        
        return samples
    
    # Generate demo datasets
    train_samples = generate_sample_data(800, config["model"]["max_seq_len"])
    val_samples = generate_sample_data(200, config["model"]["max_seq_len"])
    
    # Save to temporary files
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    train_file = os.path.join(temp_dir, "train.txt")
    val_file = os.path.join(temp_dir, "val.txt")
    
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_samples))
    
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_samples))
    
    print(f"✓ Generated {len(train_samples)} training samples")
    print(f"✓ Generated {len(val_samples)} validation samples")
    print(f"  Sample: '{train_samples[0]}'")
    
    # Step 3: Performance optimization
    print(f"\n⚡ Optimizing model performance...")
    
    optimizer = TransformerOptimizer(model)
    
    # Create sample batch for optimization
    tokenizer = TransformerDataLoader.SimpleTokenizer(config["model"]["vocab_size"])
    sample_text = train_samples[0]
    sample_tokens = tokenizer.encode(sample_text)
    sample_batch = {
        "input_ids": tessera.tensor([sample_tokens[:64]]),  # Truncate for demo
        "target_ids": tessera.tensor([sample_tokens[1:65]])
    }
    
    optimization_results = optimizer.optimize_for_hardware(sample_batch)
    
    if optimization_results["best_config"]:
        print(f"✓ Optimized for {optimization_results['best_throughput']:.1f} tokens/sec")
        print(f"  Best config: {optimization_results['best_config']}")
    
    # Step 4: Quick training demonstration
    print(f"\n🎯 Starting training demonstration...")
    print("(Note: This is a minimal demo - real training would use larger datasets)")
    
    # Create trainer
    trainer = TransformerTrainer(model, **config["training"])
    
    # Simulate a few training steps
    print("\nRunning sample training steps...")
    
    # Create minimal dataloader for demo
    class DemoDataLoader:
        def __init__(self, samples, tokenizer, batch_size, max_seq_len):
            self.samples = samples
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.max_seq_len = max_seq_len
            self.current_idx = 0
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.current_idx >= len(self.samples):
                raise StopIteration
            
            # Get batch of samples
            batch_samples = self.samples[
                self.current_idx:self.current_idx + self.batch_size
            ]
            self.current_idx += self.batch_size
            
            # Process batch
            input_ids_list = []
            target_ids_list = []
            
            for sample in batch_samples:
                tokens = self.tokenizer.encode(sample)
                if len(tokens) > self.max_seq_len:
                    tokens = tokens[:self.max_seq_len]
                
                # Pad if necessary
                while len(tokens) < 64:  # Minimum length for demo
                    tokens.append(0)
                
                input_ids_list.append(tokens[:-1])
                target_ids_list.append(tokens[1:])
            
            return {
                "input_ids": tessera.tensor(input_ids_list),
                "target_ids": tessera.tensor(target_ids_list)
            }
    
    demo_loader = DemoDataLoader(
        train_samples[:100],  # Just 100 samples for demo
        tokenizer,
        config["training"]["batch_size"],
        64  # Short sequences for demo
    )
    
    # Run a few training steps
    for step, batch in enumerate(demo_loader):
        if step >= 5:  # Just 5 steps for demonstration
            break
        
        metrics = trainer.train_step(batch, step)
        print(f"Demo Step {step + 1}: "
              f"Loss={metrics['loss']:.4f}, "
              f"PPL={metrics['perplexity']:.2f}, "
              f"Acc={metrics['accuracy']:.3f}")
    
    # Step 5: Text generation demonstration
    print(f"\n💬 Demonstrating text generation...")
    
    # Create inference engine
    inference_engine = TransformerInferenceEngine(
        model=model,
        tokenizer=tokenizer
    )
    
    # Generate some text
    test_prompts = [
        "123 + 456 =",
        "789 + 012 =", 
        "555 + 444 ="
    ]
    
    print("\nGenerating text from prompts:")
    for prompt in test_prompts:
        try:
            # Note: Since this is a tiny model trained for 5 steps,
            # the output won't be meaningful, but shows the generation process
            generated = inference_engine.generate_text(
                prompt=prompt,
                max_new_tokens=10,
                temperature=0.8
            )
            print(f"  Input:  '{prompt}'")
            print(f"  Output: '{generated}'")
        except Exception as e:
            print(f"  Generation failed for '{prompt}': {e}")
    
    # Step 6: Performance analysis
    print(f"\n📊 Performance Analysis...")
    
    # Analyze model performance
    with tessera.profiler() as prof:
        # Profile inference
        for _ in range(10):
            with tessera.no_grad():
                _ = model(tessera.tensor([[1, 2, 3, 4, 5] * 10]))  # 50 tokens
    
    performance_stats = {
        "inference_time_ms": prof.average_time * 1000,
        "throughput_tokens_per_sec": 500 / prof.average_time,  # 50 tokens * 10 runs
        "memory_usage_mb": prof.peak_memory_usage / (1024 * 1024),
        "flops_per_second": prof.total_flops / prof.total_time,
        "hardware_utilization": prof.compute_utilization
    }
    
    print("Performance Statistics:")
    for metric, value in performance_stats.items():
        if "time" in metric or "throughput" in metric:
            print(f"  {metric}: {value:.2f}")
        elif "memory"