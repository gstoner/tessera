# Advanced Examples — Honest Status

These subdirectories explore advanced ML techniques (diffusion LLMs, hybrid SSMs,
speculative decoding, Multi-Latent Attention, RLVR, KV-cache compression). **Most
were written ahead of compiler capability** — they reference APIs that Tessera does
not yet expose, or build Graph IR directly to bypass missing Pythonic layers.

This README tells you exactly what works today and what each blocked example is
waiting on. The full per-theme tracking plan lives at
[`docs/audit/advanced_examples_capability_gap.md`](../../docs/audit/advanced_examples_capability_gap.md).

If you're new to Tessera, **start with [`examples/getting_started/basic_tensor_ops.py`](../getting_started/basic_tensor_ops.py)** — it uses only the canonical surface (see [`docs/CANONICAL_API.md`](../../docs/CANONICAL_API.md)) and runs on CPU.

---

## What runs today

### Pure Python planning utilities (no Tessera ops — just plan / classify / accounting)

These are useful as design sketches and don't need any compiler features. They run on a stock Python install.

| Subdir | What it does |
|--------|--------------|
| [`kv_cache_serving/`](kv_cache_serving/) | TurboQuant/DuoAttention/Mooncake-style cache-compression planner. Estimates memory; does not execute attention. |
| [`long_context_attention/`](long_context_attention/) | Retrieval-head vs. streaming-head specialization classifier. Pure heuristics. |
| [`rlvr_reasoning_suite/`](rlvr_reasoning_suite/) | GRPO/RLVR rollout batching + reward accounting. No Tessera ops. |
| [`speculative_decoding/`](speculative_decoding/) | Yggdrasil/Medusa/EAGLE tree-decoding scheduler. Tree expansion only. |
| [`Tessera_Empirical_Software_Agent/`](Tessera_Empirical_Software_Agent/) | Tree-search agent skeleton with kernel-autotuning task. LLM hooks are stubs. |

### Compiler smoke tests (build Graph IR directly)

These bypass the Python `@tessera.jit` surface and emit Graph IR through internal compiler hooks. Useful for compile-path testing, **not** for end-to-end runs.

| Subdir | What it tests |
|--------|---------------|
| [`Fast_dLLM_v2/`](Fast_dLLM_v2/) | Diffusion-LLM Graph IR with confidence-aware parallel decoding scaffolding. |
| [`mla/`](mla/) | Multi-Latent Attention / FlashMLA Graph IR sketch. |
| [`Nemotron_Nano_12B_v2/`](Nemotron_Nano_12B_v2/) | Hybrid Mamba2/GQA/MLP Graph IR (Mamba2 mixer is a placeholder reference). |

---

## Blocked on backlog

### Scaffolds — reference phantom APIs, do not run

| Subdir | LOC | Primary blockers (audit Tier) |
|--------|-----|--------------------------------|
| [`Diffusion_LLM/`](Diffusion_LLM/) | ~5,300 | `nn.Module` / `Parameter` / `Sequential` (Tier 1), reverse-mode autodiff (Tier 2), `autocast` / `checkpoint` (Tier 3), gather/clip/einsum/masked_fill ops (Tier 3) |
| [`Jet_nemotron/`](Jet_nemotron/) | ~1,100 | `nn.Module` (Tier 1), `DynamicDepthwiseConv1d` streaming kernel (Tier 1), autodiff (Tier 2), fp8 lowering (Tier 3) |

### Stubs

| Subdir | Status |
|--------|--------|
| [`power_retention/`](power_retention/) | README + C++ kernel sketch only. Retention-attention op is unimplemented. |

---

## What "phantom API" means

Until the corresponding backlog items land, calling phantom names raises `NotImplementedError` with a pointer to the audit:

```python
>>> import tessera
>>> tessera.nn.Module()
NotImplementedError: tessera.nn.Module is on the Tier 1 backlog — see
docs/audit/advanced_examples_capability_gap.md. Until it lands, compose ops
via @tessera.jit and pass weights in explicitly.
```

The functional surface that **does** work today:

```python
import tessera
import numpy as np

x = tessera.randn((4, 16, 512)).numpy()
W = tessera.ones((512,)).numpy()
y = tessera.nn.rms_norm(x, weight=W)
W_gate = tessera.randn((512, 2048)).numpy()
W_up   = tessera.randn((512, 2048)).numpy()
W_down = tessera.randn((2048, 512)).numpy()
out = tessera.nn.swiglu(y, W_gate, W_up, W_down)
```

Or the torch-style alias:

```python
from tessera.nn import functional as F
y = F.rms_norm(x, weight=W)
```

See [`docs/CANONICAL_API.md`](../../docs/CANONICAL_API.md) for the full surface
that exists today, and the [capability-gap audit](../../docs/audit/advanced_examples_capability_gap.md)
for the prioritized backlog that will unblock the scaffolds above.

---

## Archive

Older, duplicate, or superseded drops:

- `archive/examples/advanced/`
- `archive/examples/advanced/consolidated_rl_sources/` — RL source drops folded into `rlvr_reasoning_suite/`
