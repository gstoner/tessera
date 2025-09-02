# Tessera API Reference - Volume 2
## Operations Reference

### 1. Normalization Ops

#### RMSNorm
```python
@tessera.function
def rms_norm(x: Tensor["B", "S", "D"], weight: Tensor["D"], eps: float = 1e-6) -> Tensor["B", "S", "D"]:
    rms = tessera.sqrt(tessera.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight
```
- Efficient normalization (no centering)
- Stable and distributed versions available【19†source】

---

### 2. Activation Functions

#### SwiGLU
```python
@tessera.function
def swiglu(x: Tensor["B", "S", "D"], W_gate: Tensor["D","D_ff"],
           W_up: Tensor["D","D_ff"], W_down: Tensor["D_ff","D"]) -> Tensor["B", "S", "D"]:
    gate = tessera.nn.swish(x @ W_gate)
    up = x @ W_up
    return (gate * up) @ W_down
```
- Supports fused kernels and quantized inference【19†source】

---

### 3. Attention Mechanisms

#### Flash Attention
```python
@tessera.function
def attention(q: Tensor["B","H","S","D"], k: Tensor["B","H","S","D"], v: Tensor["B","H","S","D"],
              is_causal: bool = False) -> Tensor["B","H","S","D"]:
    return tessera.ops.flash_attention(q, k, v, is_causal=is_causal)
```
- O(N) memory scaling【19†source】

#### Multi-Latent Attention (MLA)
Implements KV compression and RoPE decomposition【17†source】.

```python
mla = MultiLatentAttention(model_dim=4096, latent_dim=512, num_q_heads=32, num_kv_heads=32)
output, cache = mla(hidden_states, kv_cache=kv_cache)
```

| Feature             | Benefit                  |
|---------------------|--------------------------|
| KV Compression      | 93% cache reduction      |
| Decoupled RoPE      | Efficient positional enc |
| Paged KV Cache      | Supports var-length seqs |

---

### 4. Positional Embeddings

#### Rotary (RoPE)
```python
@tessera.function
def rotary_embedding(x: Tensor["B","S","H","D"], position_ids: Optional[Tensor["B","S"]] = None):
    ...
```

Supports **Dynamic RoPE** (NTK-aware scaling).

---

### 5. Embeddings

#### Casted Embedding
```python
@tessera.module
class CastedEmbedding:
    def __init__(self, vocab_size: int, embedding_dim: int, storage_dtype: DType = fp8_e4m3):
        self.embeddings = tessera.nn.Parameter(
            tessera.randn(vocab_size, embedding_dim, dtype=storage_dtype)
        )
```

---

### 6. Distributed Ops

Mesh parallelism【26†source】:
```python
mesh = tessera.Mesh(devices=np.array(range(16)).reshape(2, 4, 2),
                    axis_names=["data", "model", "pipeline"])

@tessera.on_mesh(mesh)
def mesh_reduce(x: MeshTensor) -> MeshTensor:
    return tessera.mesh_reduce(x, axis="data", op="mean")
```
