<!-- MERGE_START:NEMOTRON_NANO_12B_V2_BASE -->
# MODEL_SPECS — Nemotron‑Nano‑12B‑v2‑Base (for Tessera)

**What it is.** Hybrid Mamba‑Transformer (Nemotron‑H): most layers are **Mamba‑2** or **MLP(ReLU²)**, with **six Attention** layers; the max context is **128K** (pretraining 20T tokens; BF16 inference).

**Key hyper‑params (from HF `config.json`):**
- `hidden_size`: **5120**
- `num_hidden_layers`: **62**
- `num_attention_heads`: **40**
- `num_key_value_heads` (GQA groups): **8**
- `head_dim`: **128**
- `intermediate_size` (MLP): **20480**
- `hybrid_override_pattern`: `M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M-`
- `mamba_num_heads`: **128**, `mamba_head_dim`: **80**, `ssm_state_size`: **128**
- `chunk_size` (Mamba processing): **128**
- `max_position_embeddings`: **131072** (≈128K)
- `vocab_size`: **131072**

**Sources:** See the Nemotron‑Nano‑2 tech report (§2) and the model card/config on Hugging Face for the above values and architectural description.

**Tokenizer.** Provided via HF (BPE JSON + special tokens). Use HF tokenizer in the converter/runtime wrapper.

**Licensing.** NVIDIA Open Model License (weights use `nvidia-open-model-license`). This starter ships **no weights**.

## Tessera mapping (Graph‑IR → Schedule‑IR → Tile‑IR)

- **Graph‑IR nodes**
  - `nemotron.mamba2_mixer(hidden_size, m_heads, m_head_dim, ssm_state, chunk=128)`
  - `nemotron.attn_gqa(num_heads=40, kv_heads=8, head_dim=128, attn_bias=false)`
  - `nemotron.mlp_relu2(hidden=5120, ff=20480, bias=false)`
  - `nemotron.rmsnorm(eps=1e-5)` (group‑gated variant for Mamba2 pre/post)
- **Schedule‑IR**
  - Stream Mamba in **chunks of 128 tokens**, checkpointing `conv_state[intermediate, d_conv]` and
    `ssm_state[intermediate, ssm_state]`. KV‑cache is standard for Attention layers.
- **Tile‑IR / Target‑IR**
  - Mamba2: tile the depthwise causal‑conv + SSM update; fuse gate/proj; surface `lds/smem` footprints;
    map to WGMMA/MFMA/AMX where profitable; export LDS size metadata on ROCm; use TMA/cp.async on NVIDIA.
  - Attention: plug **FlashAttention** kernels (GQA, 40→8 heads, head_dim=128), with paged‑KV option.

## Conversion shape contracts

- Embedding: `[vocab, hidden]`
- Attention QKV proj (GQA): `[hidden, 3*hidden]` with KV shardable into `kv_heads*head_dim`
- Mamba2 projections: see mixer stub — input proj splits into `intermediate + conv_dim + num_heads`
- MLP (ReLU²): gate/feedforward two‑mat pattern with squared‑ReLU activation

<!-- MERGE_END:NEMOTRON_NANO_12B_V2_BASE -->
