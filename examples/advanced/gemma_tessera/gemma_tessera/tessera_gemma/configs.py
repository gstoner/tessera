from dataclasses import dataclass

@dataclass
class GemmaConfig:
    vocab_size: int = 256000
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_kv_heads: int = 8  # GQA/MQA style
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1
    tie_word_embeddings: bool = True
    dropout_p: float = 0.0
    use_flash: bool = True  # try to use Tessera/Triton-like fused attention if present
