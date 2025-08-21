# Tessera Example — GPT‑OSS‑120B (Distributed Transformer)

**Goal:** Show how to express a 120B‑parameter Transformer in Tessera using distributed tensors, pipeline/tensor/data parallelism, deterministic collectives, and efficient inference.

---

## 1) Model Overview

- Architecture: Decoder‑only Transformer, ~120B params
- Hidden size: 12288  ·  Layers: 96  ·  Heads: 96  ·  KV heads: 96
- Precision: training (bf16/fp16 with fp32 accum), inference (fp8/bf16)
- Parallelism: TP × PP × DP mesh (e.g., 8 × 6 × 4 = 192 GPUs)

---

## 2) Distributed Mesh & Sharding

```python
from tessera import dist

mesh = dist.Mesh(axes=["tp", "pp", "dp"], devices=range(192))

# Shard weights on TP, pipeline layers over PP, replicate over DP
shard_wqkv = dist.ShardSpec(partition=("col",), mesh_axes=("tp",))
shard_ffn  = dist.ShardSpec(partition=("row",), mesh_axes=("tp",))

# Example: attention projection
W_qkv = dist.tensor((hidden, 3*hidden), layout=shard_wqkv, mesh=mesh, dtype="bf16")
W_out = dist.tensor((hidden, hidden),   layout=shard_ffn,  mesh=mesh, dtype="bf16")
```

---

## 3) Transformer Block (FlashAttention + Fused MLP)

```python
from tessera import op, schedule

def transformer_block(x, cache):
    # RMSNorm
    x = op.rmsnorm(x, eps=1e-5)

    # QKV proj (TP-sharded)
    qkv = op.matmul(x, W_qkv)  # sharded over columns

    # Reshape & split heads
    q, k, v = op.split_heads(qkv, num_heads=96, head_dim=head_dim)

    # FlashAttention with cp.async pipelining
    y = op.flash_attention(q, k, v, block_q=128, block_k=128, block_d=head_dim,
                           precision="bf16", accumulate="fp32", cache=cache)

    # Output proj (row-sharded, TP)
    y = op.matmul(y, W_out)

    # Residual
    x = x + y

    # Fused MLP (gate-up-proj + SiLU + down-proj)
    y = op.fused_mlp(x, up=W_up, gate=W_gate, down=W_down, activation="silu")
    return x + y
```

**Scheduling Hint**
```python
@schedule(pipeline=True, tile={"BM":128, "BN":128, "BK":32}, warps=8)
def forward_step(x, cache):
    return transformer_block(x, cache)
```

---

## 4) Pipeline & Tensor Parallel

```python
from tessera import pipeline

layers = [transformer_block for _ in range(96)]
pipe = pipeline.compose(stages=6, assign=layers)   # 6 PP stages

@pipeline.run(mesh_axes=("pp",))
def model_step(x, kv_cache):
    for blk in layers:
        x = blk(x, kv_cache)
    return x
```

- **TP (tensor parallel):** matrix multiplications sharded over columns/rows
- **PP (pipeline parallel):** layer blocks distributed into 6 stages
- **DP (data parallel):** batch shards on dp axis with reduce‑scatter/all‑reduce

---

## 5) Deterministic Collectives & Checkpointing

```python
from tessera import dist, checkpoint

with dist.deterministic(reduce_tree="fixed"):
    grads = op.reduce_scatter(grads, axis="dp")  # ZeRO‑style
    # ... optimizer step ...

checkpoint.save(tag="step_100k", tensors=model.parameters(),
                optimizer=optimizer.state_dict(),
                mesh=mesh, atomic=True)
```

---

## 6) Inference (KV‑Cache, Prefill/Decode, CUDA‑Graphs)

```python
from tessera import graph

@graph.capture(bucket_by=["seq_len"])
def decode_step(q, k, v, cache):
    return op.flash_attention(q, k, v, cache=cache)

# batched decoding with fixed shapes for graph replay
y = decode_step(q, k, v, cache)
```

---

## 7) Autotuning & Profiles

```python
from tessera import autotune, profile

autotune.enable(cache_key=("arch","dtype","shape"))
with profile.session(exporters=[profile.summary("gpt_120b.csv")]):
    run_training_epoch()
```

---

## 8) Notes
- Use fp8 for attention proj at inference (error‑bounded), accumulate fp32
- Prefer hierarchical collectives with offsets to limit power spikes
- Use async/delta checkpointing for fast recovery
