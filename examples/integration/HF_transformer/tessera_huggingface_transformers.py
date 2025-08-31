"""
Hugging Face Transformers Implementation in Tessera Programming Model

This implementation provides Hugging Face-compatible transformer models using Tessera's
advanced features including:
- Shape-polymorphic programming for flexible model architectures
- Automatic mixed precision and optimization
- Flash Attention v3 and MLA support
- Seamless integration with HF tokenizers and datasets
- Production-ready inference with quantization
"""

import tessera as ts
from tessera import Tensor, Module, function, compile, kernel
from tessera.nn import functional as F
from typing import Optional, Dict, Tuple, List, Union
import math
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# Configuration Classes (Hugging Face Compatible)
# ============================================================================

@dataclass
class PretrainedConfig:
    """Base configuration class compatible with Hugging Face."""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    classifier_dropout: Optional[float] = None
    
    # Tessera-specific optimizations
    use_flash_attention: bool = True
    use_mla: bool = False  # Multi-Latent Attention
    use_gqa: bool = False  # Grouped Query Attention
    num_kv_heads: Optional[int] = None  # For GQA/MQA
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return self.__dict__


@dataclass
class BertConfig(PretrainedConfig):
    """BERT model configuration."""
    model_type: str = "bert"
    vocab_size: int = 30522
    hidden_act: str = "gelu"
    position_embedding_type: str = "absolute"
    type_vocab_size: int = 2


@dataclass
class GPT2Config(PretrainedConfig):
    """GPT-2 model configuration."""
    model_type: str = "gpt2"
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5


@dataclass
class LlamaConfig(PretrainedConfig):
    """Llama model configuration."""
    model_type: str = "llama"
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False


# ============================================================================
# Core Attention Mechanisms
# ============================================================================

class TesseraAttention(Module):
    """
    Universal attention module supporting multiple attention variants.
    Automatically selects optimal implementation based on configuration.
    """
    
    def __init__(self, config: PretrainedConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = config.attention_probs_dropout_prob
        
        # Support for GQA/MQA
        self.num_kv_heads = config.num_kv_heads or self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # Query, Key, Value projections
        self.q_proj = ts.nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias if hasattr(config, 'attention_bias') else True
        )
        
        self.k_proj = ts.nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )
        
        self.v_proj = ts.nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )
        
        self.o_proj = ts.nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.attention_bias if hasattr(config, 'attention_bias') else True
        )
        
        # Positional embeddings
        if config.position_embedding_type == "rope":
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=config.rope_scaling
            )
        
        # MLA components if enabled
        if config.use_mla:
            self.mla_compressor = ts.nn.Linear(
                self.hidden_size,
                self.hidden_size // 8  # 8x compression
            )
            self.mla_decompressor = ts.nn.Linear(
                self.hidden_size // 8,
                self.hidden_size
            )
    
    @ts.compile(backend="flash_attention_v3")
    def forward(
        self,
        hidden_states: Tensor["B", "S", "D"],
        attention_mask: Optional[Tensor["B", 1, "S", "S"]] = None,
        position_ids: Optional[Tensor["B", "S"]] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor]]:
        """
        Forward pass with automatic optimization selection.
        """
        B, S, _ = hidden_states.shape
        
        # Compute QKV
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = query_states.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if configured
        if self.config.position_embedding_type == "rope":
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        
        # Handle past key values for caching
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = self._update_cache(
                key_states, value_states, past_key_value, cache_kwargs
            )
        
        # Apply MLA compression if enabled
        if self.config.use_mla:
            key_states, value_states = self._apply_mla_compression(
                key_states, value_states
            )
        
        # Repeat KV heads for GQA
        if self.num_kv_groups > 1:
            key_states = repeat_kv(key_states, self.num_kv_groups)
            value_states = repeat_kv(value_states, self.num_kv_groups)
        
        # Select attention implementation
        if self.config.use_flash_attention and S > 512:
            # Use Flash Attention for long sequences
            attn_output = self._flash_attention(
                query_states, key_states, value_states, attention_mask
            )
            attn_weights = None
        else:
            # Standard scaled dot-product attention
            attn_output, attn_weights = self._standard_attention(
                query_states, key_states, value_states, attention_mask
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, S, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output,)
        if use_cache:
            outputs += ((key_states, value_states),)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    @ts.kernel
    def _flash_attention(
        self,
        q: Tensor["B", "H", "S", "D"],
        k: Tensor["B", "H", "S", "D"],
        v: Tensor["B", "H", "S", "D"],
        mask: Optional[Tensor]
    ) -> Tensor["B", "H", "S", "D"]:
        """Flash Attention v3 implementation."""
        return ts.nn.flash_attention_v3(
            q, k, v,
            causal=mask is not None,
            dropout=self.dropout if self.training else 0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim)
        )
    
    def _standard_attention(
        self,
        q: Tensor["B", "H", "S", "D"],
        k: Tensor["B", "H", "S", "D"],
        v: Tensor["B", "H", "S", "D"],
        mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Standard attention implementation."""
        attn_weights = ts.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_weights = attn_weights + mask
        
        attn_weights = ts.softmax(attn_weights, dim=-1)
        attn_weights = ts.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = ts.matmul(attn_weights, v)
        
        return attn_output, attn_weights
    
    def _apply_mla_compression(
        self,
        k: Tensor["B", "H", "S", "D"],
        v: Tensor["B", "H", "S", "D"]
    ) -> Tuple[Tensor, Tensor]:
        """Apply Multi-Latent Attention compression."""
        B, H, S, D = k.shape
        
        # Compress KV to latent space
        kv = ts.cat([k, v], dim=-1)  # [B, H, S, 2*D]
        kv = kv.reshape(B, S, H * 2 * D)
        kv_compressed = self.mla_compressor(kv)  # [B, S, D/8]
        
        # Reshape back
        kv_compressed = kv_compressed.view(B, H, S, -1)
        
        return kv_compressed, kv_compressed


# ============================================================================
# Transformer Blocks
# ============================================================================

class TesseraMLP(Module):
    """
    Feed-forward network with various activation functions.
    Supports SwiGLU, GELU, and other modern activations.
    """
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        
        # Determine activation function
        self.activation_fn = self._get_activation(config.hidden_act)
        
        # MLP layers
        if config.hidden_act in ["swiglu", "silu"]:
            # GLU variant - needs 3x hidden size for gate
            self.gate_proj = ts.nn.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=False
            )
            self.up_proj = ts.nn.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=False
            )
            self.down_proj = ts.nn.Linear(
                config.intermediate_size,
                config.hidden_size,
                bias=False
            )
        else:
            # Standard MLP
            self.fc1 = ts.nn.Linear(
                config.hidden_size,
                config.intermediate_size
            )
            self.fc2 = ts.nn.Linear(
                config.intermediate_size,
                config.hidden_size
            )
        
        self.dropout = ts.nn.Dropout(config.hidden_dropout_prob)
    
    @ts.compile(mode="training")
    def forward(self, hidden_states: Tensor["B", "S", "D"]) -> Tensor["B", "S", "D"]:
        """Forward pass through MLP."""
        if hasattr(self, 'gate_proj'):
            # SwiGLU activation
            return self.down_proj(
                self.activation_fn(self.gate_proj(hidden_states)) * 
                self.up_proj(hidden_states)
            )
        else:
            # Standard MLP
            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.fc2(hidden_states)
            return hidden_states
    
    def _get_activation(self, act_name: str):
        """Get activation function by name."""
        activations = {
            "gelu": ts.nn.gelu,
            "gelu_new": ts.nn.gelu_new,
            "relu": ts.nn.relu,
            "silu": ts.nn.silu,
            "swiglu": ts.nn.silu,
            "tanh": ts.tanh
        }
        return activations.get(act_name, ts.nn.gelu)


class TransformerBlock(Module):
    """
    Universal transformer block supporting various architectures.
    """
    
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Attention layer
        self.attention = TesseraAttention(config, layer_idx)
        
        # MLP
        self.mlp = TesseraMLP(config)
        
        # Layer normalization
        if hasattr(config, 'rms_norm_eps'):
            # RMSNorm for Llama-style models
            self.input_layernorm = ts.nn.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = ts.nn.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps
            )
        else:
            # Standard LayerNorm
            self.input_layernorm = ts.nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps
            )
            self.post_attention_layernorm = ts.nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps
            )
        
        # Dropout
        self.dropout = ts.nn.Dropout(config.hidden_dropout_prob)
    
    @ts.checkpoint  # Gradient checkpointing
    def forward(
        self,
        hidden_states: Tensor["B", "S", "D"],
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[Tensor, ...]:
        """
        Forward pass through transformer block.
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        
        hidden_states = attn_outputs[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs


# ============================================================================
# Model Implementations
# ============================================================================

class TesseraPreTrainedModel(Module):
    """
    Base class for all Tessera transformer models.
    Handles initialization, loading, and common functionality.
    """
    
    config_class = PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, ts.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, ts.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (ts.nn.LayerNorm, ts.nn.RMSNorm)):
            if hasattr(module, 'bias'):
                module.bias.data.zero_()
            if hasattr(module, 'weight'):
                module.weight.data.fill_(1.0)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs
    ):
        """
        Load a pretrained model from Hugging Face hub or local path.
        Compatible with Hugging Face model loading.
        """
        # This would integrate with HF hub
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        
        # Load weights (simplified - actual implementation would handle HF formats)
        state_dict = ts.load(f"{pretrained_model_name_or_path}/model.tsr")
        model.load_state_dict(state_dict)
        
        return model
    
    def save_pretrained(self, save_directory: str):
        """Save model in Hugging Face format."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save model with Tessera optimizations
        ts.save(
            self.state_dict(),
            f"{save_directory}/model.tsr",
            optimization_level=3
        )


class BertModel(TesseraPreTrainedModel):
    """
    BERT model implementation compatible with Hugging Face.
    """
    
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.embeddings = BertEmbeddings(config)
        
        # Encoder
        self.encoder = ts.nn.ModuleList([
            TransformerBlock(config, i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Pooler for classification tasks
        self.pooler = BertPooler(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @ts.compile(mode="inference", backend="tensorrt")
    def forward(
        self,
        input_ids: Tensor["B", "S"],
        attention_mask: Optional[Tensor["B", "S"]] = None,
        token_type_ids: Optional[Tensor["B", "S"]] = None,
        position_ids: Optional[Tensor["B", "S"]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, Dict]:
        """
        Forward pass through BERT model.
        """
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask)
        
        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # Pass through encoder
        hidden_states = embedding_output
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.encoder:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Pooler
        pooled_output = self.pooler(hidden_states)
        
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "pooler_output": pooled_output,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions
            }
        
        return (hidden_states, pooled_output, all_hidden_states, all_attentions)
    
    def _prepare_attention_mask(self, attention_mask: Tensor) -> Tensor:
        """Prepare attention mask for attention layers."""
        # Convert to correct format for attention
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask


class GPT2Model(TesseraPreTrainedModel):
    """
    GPT-2 model implementation compatible with Hugging Face.
    """
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.config = config
        
        # Token and position embeddings
        self.wte = ts.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = ts.nn.Embedding(config.n_positions, config.n_embd)
        
        # Dropout
        self.drop = ts.nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.h = ts.nn.ModuleList([
            TransformerBlock(config, i)
            for i in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = ts.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @ts.compile(mode="inference")
    def forward(
        self,
        input_ids: Tensor["B", "S"],
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, Dict]:
        """
        Forward pass through GPT-2 model.
        """
        B, S = input_ids.shape
        
        # Get position ids
        if position_ids is None:
            position_ids = ts.arange(S, dtype=ts.long).unsqueeze(0).expand(B, -1)
        
        # Embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_causal_mask(attention_mask, S)
        
        # Pass through transformer blocks
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None
        
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            past_key_value = past_key_values[i] if past_key_values else None
            
            block_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions
            )
            
            hidden_states = block_outputs[0]
            
            if use_cache:
                next_cache = next_cache + (block_outputs[1],)
            
            if output_attentions:
                all_attentions = all_attentions + (block_outputs[-1],)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": next_cache,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions
            }
        
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attentions] if v is not None)
    
    def _prepare_causal_mask(self, attention_mask: Tensor, seq_length: int) -> Tensor:
        """Prepare causal attention mask."""
        # Create causal mask
        causal_mask = ts.triu(
            ts.ones((seq_length, seq_length), dtype=ts.float32),
            diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, -10000.0)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            causal_mask = causal_mask + attention_mask[:, None, None, :]
        
        return causal_mask


class LlamaModel(TesseraPreTrainedModel):
    """
    Llama model implementation with RoPE and modern optimizations.
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        
        # Token embeddings
        self.embed_tokens = ts.nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )
        
        # Transformer layers
        self.layers = ts.nn.ModuleList([
            TransformerBlock(config, i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final RMS norm
        self.norm = ts.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @ts.compile(mode="inference", backend="flash_attention_v3")
    def forward(
        self,
        input_ids: Tensor["B", "S"],
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        cache_position: Optional[Tensor] = None
    ) -> Tuple[Tensor, ...]:
        """
        Forward pass through Llama model with RoPE.
        """
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare position ids for RoPE
        if position_ids is None:
            position_ids = ts.arange(
                hidden_states.shape[1],
                dtype=ts.long,
                device=hidden_states.device
            ).unsqueeze(0)
        
        # Pass through layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache = next_cache + (layer_outputs[1],)
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-1],)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attentions] if v is not None)


# ============================================================================
# Task-Specific Model Heads
# ============================================================================

class BertForSequenceClassification(TesseraPreTrainedModel):
    """BERT model for sequence classification tasks."""
    
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout 
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = ts.nn.Dropout(classifier_dropout)
        self.classifier = ts.nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @ts.compile(mode="training")
    def forward(
        self,
        input_ids: Tensor["B", "S"],
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor["B"]] = None
    ) -> Dict[str, Tensor]:
        """Forward pass with optional loss computation."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = ts.nn.cross_entropy(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions")
        }


class GPT2ForCausalLM(TesseraPreTrainedModel):
    """GPT-2 model for causal language modeling."""
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = ts.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between input and output embeddings
        self.lm_head.weight = self.transformer.wte.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @ts.compile(mode="inference", backend="tensorrt")
    def forward(
        self,
        input_ids: Tensor["B", "S"],
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        labels: Optional[Tensor["B", "S"]] = None,
        use_cache: bool = True
    ) -> Dict[str, Tensor]:
        """Forward pass for language modeling."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        hidden_states = outputs["last_hidden_state"]
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = ts.nn.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": outputs.get("past_key_values"),
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions")
        }
    
    @ts.compile(mode="inference", backend="tensorrt")
    def generate(
        self,
        input_ids: Tensor["B", "S"],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        do_sample: bool = True,
        pad_token_id: int = 50256,
        eos_token_id: int = 50256
    ) -> Tensor["B", "S_out"]:
        """
        Generate text using various decoding strategies.
        """
        batch_size = input_ids.shape[0]
        past_key_values = None
        
        # Keep track of which sequences are done
        unfinished_sequences = ts.ones(batch_size, dtype=ts.long)
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                logits = top_k_filtering(logits, top_k)
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                logits = top_p_filtering(logits, top_p)
            
            # Sample or take argmax
            if do_sample:
                probs = ts.softmax(logits, dim=-1)
                next_tokens = ts.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = ts.argmax(logits, dim=-1)
            
            # Update unfinished sequences
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id).long()
            
            # Append to input_ids
            input_ids = ts.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Break if all sequences are done
            if unfinished_sequences.sum() == 0:
                break
        
        return input_ids


class LlamaForCausalLM(TesseraPreTrainedModel):
    """Llama model for causal language modeling."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = ts.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @ts.compile(mode="training", backend="hopper")
    def forward(
        self,
        input_ids: Tensor["B", "S"],
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = True,
        cache_position: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Forward pass for Llama language modeling."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = ts.nn.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs[1] if len(outputs) > 1 else None
        }


# ============================================================================
# Helper Components
# ============================================================================

class BertEmbeddings(Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = ts.nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id if hasattr(config, 'pad_token_id') else 0
        )
        self.position_embeddings = ts.nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.token_type_embeddings = ts.nn.Embedding(
            config.type_vocab_size,
            config.hidden_size
        )
        
        self.LayerNorm = ts.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = ts.nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        input_ids: Tensor["B", "S"],
        token_type_ids: Optional[Tensor["B", "S"]] = None,
        position_ids: Optional[Tensor["B", "S"]] = None
    ) -> Tensor["B", "S", "D"]:
        seq_length = input_ids.shape[1]
        
        if position_ids is None:
            position_ids = ts.arange(seq_length).unsqueeze(0)
        
        if token_type_ids is None:
            token_type_ids = ts.zeros_like(input_ids)
        
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertPooler(Module):
    """Pool the model by taking the hidden state of the first token."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = ts.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = ts.nn.Tanh()
    
    def forward(self, hidden_states: Tensor["B", "S", "D"]) -> Tensor["B", "D"]:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RotaryEmbedding(Module):
    """Rotary Position Embedding for models like Llama."""
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling: Optional[Dict] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling = scaling
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (ts.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache for fast inference
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = ts.arange(seq_len, dtype=self.inv_freq.dtype)
        
        # Apply scaling if configured
        if self.scaling is not None:
            t = self._apply_scaling(t)
        
        freqs = ts.outer(t, self.inv_freq)
        emb = ts.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self,
        x: Tensor,
        position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        seq_len = x.shape[1] if position_ids is None else position_ids.max() + 1
        
        # Expand cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        if position_ids is None:
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
        else:
            return self.cos_cached[position_ids], self.sin_cached[position_ids]


@ts.function
def apply_rotary_pos_emb(
    q: Tensor["B", "H", "S", "D"],
    k: Tensor["B", "H", "S", "D"],
    cos: Tensor,
    sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@ts.function
def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ts.cat((-x2, x1), dim=-1)


@ts.function
def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """
    Repeat key/value heads for grouped query attention.
    """
    if n_rep == 1:
        return hidden_states
    
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@ts.function
def top_k_filtering(logits: Tensor, top_k: int) -> Tensor:
    """Filter logits to keep only top-k tokens."""
    indices_to_remove = logits < ts.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = -float('inf')
    return logits


@ts.function
def top_p_filtering(logits: Tensor, top_p: float) -> Tensor:
    """Filter logits using nucleus (top-p) filtering."""
    sorted_logits, sorted_indices = ts.sort(logits, descending=True)
    cumulative_probs = ts.cumsum(ts.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = -float('inf')
    
    return logits


# ============================================================================
# Training and Inference Pipelines
# ============================================================================

class HuggingFaceTrainer:
    """
    Training pipeline compatible with Hugging Face Trainer API.
    """
    
    def __init__(
        self,
        model: TesseraPreTrainedModel,
        args: Dict,
        train_dataset: ts.data.Dataset,
        eval_dataset: Optional[ts.data.Dataset] = None,
        tokenizer: Optional[object] = None,
        compute_metrics: Optional[callable] = None
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        
        # Setup optimizer
        self.optimizer = ts.optim.AdamW(
            model.parameters(),
            lr=args.get("learning_rate", 5e-5),
            betas=(0.9, 0.999),
            weight_decay=args.get("weight_decay", 0.01)
        )
        
        # Setup scheduler
        self.scheduler = ts.optim.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.get("warmup_steps", 1000),
            num_training_steps=args.get("max_steps", 10000)
        )
        
        # Setup distributed training if available
        self.mesh = None
        if ts.distributed.is_initialized():
            self.mesh = ts.mesh(
                devices=ts.cuda.device_count(),
                topology="ring",
                axes={"data": ts.cuda.device_count()}
            )
            
            # Wrap model for distributed training
            self.model = ts.distributed.parallelize(
                self.model,
                mesh=self.mesh,
                strategy="ddp"
            )
    
    @ts.compile(mode="training", backend="hopper")
    def training_step(self, batch: Dict) -> Tensor:
        """Single training step."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        ts.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.args.get("max_grad_norm", 1.0)
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss
    
    def train(self):
        """Main training loop."""
        train_dataloader = ts.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.get("per_device_train_batch_size", 8),
            shuffle=True,
            num_workers=4
        )
        
        global_step = 0
        
        for epoch in range(self.args.get("num_train_epochs", 3)):
            epoch_loss = 0
            
            for batch in ts.tqdm(train_dataloader, desc=f"Epoch {epoch}"):
                loss = self.training_step(batch)
                epoch_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % self.args.get("logging_steps", 100) == 0:
                    ts.log({
                        "loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": global_step
                    })
                
                # Evaluation
                if global_step % self.args.get("eval_steps", 500) == 0:
                    self.evaluate()
                
                # Checkpointing
                if global_step % self.args.get("save_steps", 1000) == 0:
                    self.save_checkpoint(global_step)
            
            print(f"Epoch {epoch} - Average Loss: {epoch_loss / len(train_dataloader):.4f}")
    
    @ts.compile(mode="inference")
    def evaluate(self):
        """Evaluation loop."""
        if self.eval_dataset is None:
            return
        
        self.model.eval()
        eval_dataloader = ts.data.DataLoader(
            self.eval_dataset,
            batch_size=self.args.get("per_device_eval_batch_size", 8),
            shuffle=False
        )
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with ts.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(**batch)
                total_loss += outputs["loss"].item()
                
                if "logits" in outputs:
                    predictions = ts.argmax(outputs["logits"], dim=-1)
                    all_predictions.append(predictions)
                    
                    if "labels" in batch:
                        all_labels.append(batch["labels"])
        
        # Compute metrics
        avg_loss = total_loss / len(eval_dataloader)
        metrics = {"eval_loss": avg_loss}
        
        if self.compute_metrics and all_predictions:
            predictions = ts.cat(all_predictions)
            labels = ts.cat(all_labels) if all_labels else None
            custom_metrics = self.compute_metrics(predictions, labels)
            metrics.update(custom_metrics)
        
        ts.log(metrics)
        return metrics
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_dir = f"{self.args.get('output_dir', '.')}/checkpoint-{step}"
        
        # Save model with Tessera optimizations
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        ts.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step
        }, f"{checkpoint_dir}/training_state.tsr")


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """
    Example showing how to use Hugging Face compatible models with Tessera.
    """
    
    # 1. Load a BERT model for classification
    config = BertConfig(
        num_labels=2,
        use_flash_attention=True  # Enable Tessera optimizations
    )
    model = BertForSequenceClassification(config)
    
    # 2. Or load a pretrained model (would integrate with HF hub)
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    
    # 3. Prepare data (example with dummy data)
    train_dataset = ts.data.DummyDataset(
        size=1000,
        input_shape=(512,),  # sequence length
        num_labels=2
    )
    
    # 4. Setup training
    training_args = {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "num_train_epochs": 3,
        "warmup_steps": 500,
        "logging_steps": 100,
        "eval_steps": 500,
        "save_steps": 1000,
        "output_dir": "./output"
    }
    
    trainer = HuggingFaceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    # 5. Train
    trainer.train()
    
    # 6. Generate text with GPT-2
    gpt2_config = GPT2Config()
    gpt2_model = GPT2ForCausalLM(gpt2_config)
    
    # Example generation
    input_ids = ts.tensor([[50256]])  # Start token
    generated = gpt2_model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    print("Generated text:", generated)
    
    # 7. Use Llama for inference
    llama_config = LlamaConfig(
        use_flash_attention=True,
        use_gqa=True,  # Grouped Query Attention
        num_key_value_heads=8  # Reduce KV heads for efficiency
    )
    llama_model = LlamaForCausalLM(llama_config)
    
    # Compile for maximum inference performance
    optimized_model = ts.compile(
        llama_model,
        mode="inference",
        backend="tensorrt",
        optimization_level=3
    )
    
    # Deploy as service
    service = ts.serving.ModelService(
        optimized_model,
        batch_size=32,
        max_sequence_length=2048
    )
    
    # service.serve(port=8080)  # Start serving


if __name__ == "__main__":
    main()