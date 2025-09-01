# GPT-OSS-120B Tessera Port - Part 2: FFN, Transformer Blocks & Model Assembly

## Feed-Forward Network with Gated Linear Units

```python
@ts.jit
class GatedMLP(ts.Module):
    """
    Gated MLP (SwiGLU) for improved performance
    Uses 3 matrices: gate, up, and down projections
    """
    
    def __init__(self, config: GPTConfig, mesh: dist.Mesh):
        super().__init__()
        self.config = config
        self.mesh = mesh
        
        # Gated MLP uses 3 weight matrices
        self.W_gate = self._create_weight(
            (config.d_model, config.d_ffn),
            partition_spec=("col",),
            mesh_axes=("tp",)
        )
        
        self.W_up = self._create_weight(
            (config.d_model, config.d_ffn),
            partition_spec=("col",),
            mesh_axes=("tp",)
        )
        
        self.W_down = self._create_weight(
            (config.d_ffn, config.d_model),
            partition_spec=("row",),
            mesh_axes=("tp",)
        )
    
    def _create_weight(self, shape, partition_spec, mesh_axes):
        """Create distributed weight tensor"""
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
    def forward(self, x: Tensor["B", "S", "D", "bf16 @accum(fp32)"]):
        """
        Forward pass with SwiGLU activation
        
        SwiGLU(x) = (swish(xW_gate) * xW_up) W_down
        """
        # Compute gate and up projections in parallel
        gate = ts.gemm(x, self.W_gate)  # [B, S, d_ffn]
        up = ts.gemm(x, self.W_up)      # [B, S, d_ffn]
        
        # Apply SwiGLU: swish(gate) * up
        # Using optimized fused kernel when available
        hidden = ts.nn.swiglu(gate, up)
        
        # Down projection
        output = ts.gemm(hidden, self.W_down)  # [B, S, d_model]
        
        # Apply dropout if training
        if self.training and self.config.dropout > 0:
            output = ts.nn.dropout(output, p=self.config.dropout)
        
        return output


@ts.jit
class StandardMLP(ts.Module):
    """Standard 2-layer MLP with GeLU activation"""
    
    def __init__(self, config: GPTConfig, mesh: dist.Mesh):
        super().__init__()
        self.config = config
        self.mesh = mesh
        
        self.W_in = self._create_weight(
            (config.d_model, config.d_ffn),
            partition_spec=("col",),
            mesh_axes=("tp",)
        )
        
        self.W_out = self._create_weight(
            (config.d_ffn, config.d_model),
            partition_spec=("row",),
            mesh_axes=("tp",)
        )
    
    def _create_weight(self, shape, partition_spec, mesh_axes):
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
    def forward(self, x: Tensor["B", "S", "D", "bf16 @accum(fp32)"]):
        """Forward pass with GeLU activation"""
        hidden = ts.gemm(x, self.W_in)
        hidden = ts.nn.gelu_safe(hidden)  # Safe GeLU for numerical stability
        
        if self.training and self.config.dropout > 0:
            hidden = ts.nn.dropout(hidden, p=self.config.dropout)
        
        output = ts.gemm(hidden, self.W_out)
        return output
```

## Layer Normalization

```python
@ts.jit
class RMSNorm(ts.Module):
    """
    Root Mean Square Layer Normalization
    More efficient than standard LayerNorm, especially for large models
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # Learnable scale parameter
        self.weight = ts.nn.Parameter(ts.ones(dim))
    
    @ts.jit
    @ts.autodiff
    def forward(self, x: Tensor["...", "D", "bf16 @accum(fp32)"]):
        """Apply RMS normalization"""
        # Use Tessera's safe RMSNorm implementation
        return ts.nn.rmsnorm_safe(x, self.weight, eps=self.eps)
```

## Transformer Block

```python
@ts.jit
class TransformerBlock(ts.Module):
    """
    Single transformer block with parallel attention and FFN
    Implements pre-normalization and optional parallel computation
    """
    
    def __init__(self, config: GPTConfig, layer_idx: int, mesh: dist.Mesh):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mesh = mesh
        
        # Attention components
        self.attention = MultiQueryAttention(config, layer_idx, mesh)
        self.attention_norm = RMSNorm(config.d_model)
        
        # FFN components
        if config.use_gated_mlp:
            self.ffn = GatedMLP(config, mesh)
        else:
            self.ffn = StandardMLP(config, mesh)
        self.ffn_norm = RMSNorm(config.d_model)
        
        # For parallel attention + FFN
        self.use_parallel = config.use_parallel_attention
    
    @ts.jit
    @ts.autodiff
    def forward(
        self,
        x: Tensor["B", "S", "D", "bf16 @accum(fp32)"],
        kv_cache: Optional[ts.KVCache] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None
    ):
        """
        Forward pass with optional parallel attention and FFN
        
        Standard: x = x + Attn(Norm(x)); x = x + FFN(Norm(x))
        Parallel: x = x + Attn(Norm(x)) + FFN(Norm(x))
        """
        
        if self.use_parallel:
            # Parallel computation of attention and FFN
            return self._forward_parallel(x, kv_cache, attention_mask, position_ids)
        else:
            # Sequential computation
            return self._forward_sequential(x, kv_cache, attention_mask, position_ids)
    
    @ts.jit
    def _forward_sequential(self, x, kv_cache, attention_mask, position_ids):
        """Sequential attention then FFN"""
        # Attention block with residual
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, kv_cache, attention_mask, position_ids)
        x = residual + x
        
        # FFN block with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x
    
    @ts.jit
    def _forward_parallel(self, x, kv_cache, attention_mask, position_ids):
        """Parallel attention and FFN computation"""
        residual = x
        
        # Normalize once for both paths
        x_norm = self.attention_norm(x)
        
        # Compute attention and FFN in parallel
        attn_out = self.attention(x_norm, kv_cache, attention_mask, position_ids)
        
        # For true parallel execution, use a separate norm
        # This slightly changes the computation but improves efficiency
        x_ffn_norm = self.ffn_norm(x_norm)
        ffn_out = self.ffn(x_ffn_norm)
        
        # Combine residuals
        x = residual + attn_out + ffn_out
        
        return x
```

## Complete GPT Model

```python
@ts.jit
class GPTModel(ts.Module):
    """
    Complete GPT-OSS-120B model implementation in Tessera
    """
    
    def __init__(self, config: GPTConfig, mesh: dist.Mesh):
        super().__init__()
        self.config = config
        self.mesh = mesh
        
        # Token embeddings
        self.token_embedding = self._create_embedding(
            config.vocab_size, 
            config.d_model
        )
        
        # Position embeddings (if not using RoPE)
        if not config.use_rotary_embeddings:
            self.position_embedding = self._create_embedding(
                config.max_seq_len,
                config.d_model
            )
        
        # Transformer blocks
        self.blocks = ts.nn.ModuleList([
            TransformerBlock(config, i, mesh)
            for i in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_final = RMSNorm(config.d_model)
        
        # Output projection (can be tied with token embeddings)
        self.lm_head = self._create_lm_head()
        
        # Gradient checkpointing setup
        if config.gradient_checkpointing:
            self._setup_checkpointing()
    
    def _create_embedding(self, num_embeddings: int, embedding_dim: int):
        """Create embedding layer with proper initialization"""
        return dist.tensor(
            shape=(num_embeddings, embedding_dim),
            layout=dist.ShardSpec(
                partition=("row",),  # Shard vocabulary
                mesh_axes=("tp",)
            ),
            mesh=self.mesh,
            dtype=self.config.compute_dtype,
            initializer=ts.nn.init.normal_(std=0.02)
        )
    
    def _create_lm_head(self):
        """Create language model head (output projection)"""
        if self.config.tie_word_embeddings:
            # Tie with token embeddings
            return None  # Will use token_embedding.T
        else:
            return dist.tensor(
                shape=(self.config.d_model, self.config.vocab_size),
                layout=dist.ShardSpec(
                    partition=("col",),
                    mesh_axes=("tp",)
                ),
                mesh=self.mesh,
                dtype=self.config.weight_dtype,
                initializer=ts.nn.init.normal_(std=0.02)
            )
    
    def _setup_checkpointing(self):
        """Configure gradient checkpointing for memory efficiency"""
        if self.config.activation_checkpointing_layers is None:
            # Checkpoint every 4th layer by default
            checkpoint_layers = list(range(0, self.config.n_layers, 4))
        else:
            checkpoint_layers = self.config.activation_checkpointing_layers
        
        for idx in checkpoint_layers:
            self.blocks[idx] = ts.checkpoint(self.blocks[idx])
    
    @ts.jit
    @ts.autodiff
    def forward(
        self,
        input_ids: Tensor["B", "S", "i32"],
        attention_mask: Optional[Tensor["B", "S", "S", "bf16"]] = None,
        position_ids: Optional[Tensor["B", "S", "i32"]] = None,
        kv_cache: Optional[ts.KVCache] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the GPT model
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            kv_cache: Optional KV cache for inference
            use_cache: Whether to return updated KV cache
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
        
        Returns:
            Dictionary containing:
                - logits: Output logits [batch_size, seq_len, vocab_size]
                - hidden_states: Optional list of hidden states
                - attentions: Optional list of attention weights
                - kv_cache: Optional updated KV cache
        """
        B, S = input_ids.shape
        
        # Token embeddings
        hidden_states = ts.nn.embedding(input_ids, self.token_embedding)
        
        # Add position embeddings if not using RoPE
        if not self.config.use_rotary_embeddings:
            if position_ids is None:
                position_ids = ts.arange(S, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(B, -1)
            pos_embeds = ts.nn.embedding(position_ids, self.position_embedding)
            hidden_states = hidden_states + pos_embeds
        
        # Apply dropout
        if self.training and self.config.dropout > 0:
            hidden_states = ts.nn.dropout(hidden_states, p=self.config.dropout)
        
        # Initialize cache for storing intermediate states
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Create or update KV cache
        if use_cache and kv_cache is None:
            kv_cache = self._create_kv_cache(B)
        
        # Process through transformer blocks
        for idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Apply transformer block
            hidden_states = block(
                hidden_states,
                kv_cache=kv_cache if use_cache else None,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
            if output_attentions:
                # Note: This would require modifying blocks to return attention weights
                pass
        
        # Final layer norm
        hidden_states = self.ln_final(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Compute logits
        if self.lm_head is None:
            # Tied embeddings
            logits = ts.gemm(hidden_states, self.token_embedding.T)
        else:
            logits = ts.gemm(hidden_states, self.lm_head)
        
        # Prepare output dictionary
        outputs = {"logits": logits}
        
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
        
        if output_attentions:
            outputs["attentions"] = all_attentions
        
        if use_cache:
            outputs["kv_cache"] = kv_cache
        
        return outputs
    
    def _create_kv_cache(self, batch_size: int) -> ts.KVCache:
        """Create KV cache for inference"""
        return ts.KVCache(
            batch_size=batch_size,
            max_seq_len=self.config.max_seq_len,
            n_layers=self.config.n_layers,
            n_kv_heads=self.config.n_kv_heads,
            d_head=self.config.d_head,
            dtype=self.config.kv_cache_dtype,
            device=self.mesh.devices[0]
        )