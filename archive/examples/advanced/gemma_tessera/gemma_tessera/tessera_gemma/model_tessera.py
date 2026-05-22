import torch, torch.nn as nn
from .configs import GemmaConfig
from .kernels.rmsnorm_tessera import RMSNorm
from .kernels.mlp_swiglu_tessera import SwiGLU
from .kernels.attention_tessera import TesseraAttention
from .ops.rope import precompute_rope_cache

class DecoderBlock(nn.Module):
    def __init__(self, cfg: GemmaConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = TesseraAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.num_kv_heads, dropout_p=cfg.dropout_p)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, rope_cos, rope_sin, kv_cache=None, use_cache: bool=False, update_cache: bool=True):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin, kv_cache=kv_cache, use_cache=use_cache, update_cache=update_cache)
        x = x + self.mlp(self.norm2(x))
        return x

class TesseraGemmaForCausalLM(nn.Module):
    def __init__(self, cfg: GemmaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        # rope cache will be created on first forward
        self.register_buffer("_rope_ready", torch.tensor(0), persistent=False)
        self.rope_cos = None
        self.rope_sin = None

    @torch.no_grad()
    def _maybe_build_rope(self, device, dtype):
        if self._rope_ready.item() == 1:
            return
        Dh = self.cfg.hidden_size // self.cfg.num_attention_heads
        cos, sin = precompute_rope_cache(self.cfg.max_position_embeddings, Dh, self.cfg.rope_theta, device, dtype)
        self.rope_cos, self.rope_sin = cos, sin
        self._rope_ready.fill_(1)

    def forward(self, input_ids: torch.LongTensor, *, kv_caches=None, use_cache: bool=False, update_cache: bool=True) -> torch.Tensor:
        from .utils.shapes import ShapeSpec, check_shape
        symbols = {}
        check_shape('input_ids', tuple(input_ids.shape), ShapeSpec(['B','T']), symbols)
        B, T = input_ids.shape
        x = self.embed(input_ids)
        self._maybe_build_rope(x.device, x.dtype)
        # Optionally create caches if requested
        if use_cache and kv_caches is None:
            from .kernels.kv_cache_tessera import PagedKVCache
            Dh = self.cfg.hidden_size // self.cfg.num_attention_heads
            kv_caches = [PagedKVCache(B, self.cfg.num_kv_heads, Dh, page_size=128, device=x.device, dtype=x.dtype) for _ in range(len(self.blocks))]
        for i, blk in enumerate(self.blocks):
            cache_i = kv_caches[i] if (use_cache and kv_caches is not None) else None
            x = blk(x, self.rope_cos, self.rope_sin, kv_cache=cache_i, use_cache=use_cache, update_cache=update_cache)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
