# Tessera Examples & Integration Guide

This document provides **informative worked examples**, lowering flows, numerics checklists, and integration paths for Tessera.

---

## Chapter 6: Worked Operator Examples

### Example 1: FlashAttention Kernel

**Graph IR:**
```text
Q: Tensor[B, H, N, D]
K: Tensor[B, H, N, D]
V: Tensor[B, H, N, D]
S = Q @ adjoint(K)
P = softmax(S)
O = P @ V
```

**Tile IR:**
```mlir
%s = tile.matmul %Q, %K^T {bm=64, bn=64, bk=64}
%p = tile.softmax %s {axis=-1}
%o = tile.matmul %p, %V {bm=64, bn=64, bk=64}
```

---

### Example 2: Spectral Denoising

```mlir
%U, %S, %V = tile.svd %A
%S' = tile.threshold %S, %tau
%A' = tile.matmul %U, (%S' @ %V)
```

---

### Example 3: Mixture of Recursions

```mlir
%r0 = tile.init %x
%r1 = Σ wi * fi(%r0)
%r2 = Σ wi * fi(%r1)
```

---

### Example 4: Mixture of Experts

```mlir
%p = tile.gate %x → [p1, p2, ...]
%o = tile.moe_dispatch %x, %p, [E1, E2, ... En]
```

---

### Example 5: Test-Time Compute

```mlir
%y_draft = tile.invoke %small_model(%x)
%check = tile.verify %large_model(%x), %y_draft
%y = tile.select %check, %y_draft, %large_model(%x)
```

---

### Example 6: SFT + RLHF

```mlir
%y = tile.model %policy(%x)
%loss = tile.cross_entropy %y, %labels

%reward = tile.reward_model %policy(%x)
%adv = tile.gae %reward
%update = tile.ppo %policy, %adv
```

---

## Appendix E: Integration Examples

### E.1 GPT-OSS-120B in Tessera

```c
tessera_context_handle ctx;
tessera_context_create(&ctx);

// Load GPT-OSS-120B graph
tessera_operator_handle gpt = tessera_load_operator(ctx, "gpt-oss-120b.tessera.ops");

// Launch attention + MLP tiles
tessera_launch_desc attn = {128, 128, 64, 8, 2, 0};
tessera_launch_tile(gpt, &attn);
```

---

### E.2 Hugging Face Transformers Importer

```python
from tessera.importers import hf_import
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
tessera_graph = hf_import(model)
tessera_graph.save("gpt2.tessera.ops")
```

---

### E.3 Hugging Face Baseline (PyTorch)

```python
from transformers import GPT2LMHeadModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()

x = tok("Hello", return_tensors="pt").to("cuda")
out = model.generate(**x, max_new_tokens=32)
print(tok.decode(out[0]))
```

---

### E.4 SFT + RLHF Training in Tessera

```mlir
%logits = tile.model %policy(%x)
%loss = tile.cross_entropy %logits, %labels

%reward = tile.reward_model %policy(%x)
%adv = tile.gae %reward
%update = tile.ppo %policy, %adv
```

---

## Numerics & Determinism Checklist

- FlashAttention: stable softmax, fp16/bf16 accumulators
- MoE: reproducible top-k gating
- RLHF: deterministic sampling with fixed seeds
