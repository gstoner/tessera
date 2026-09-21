# Advanced Examples — Honest Status

These subdirectories provide maintained, manifest-audited slices of advanced ML
techniques: diffusion LLMs, hybrid SSMs, speculative decoding, Multi-Latent
Attention, RLVR, and KV-cache compression. Incomplete source drops are preserved
under `archive/examples/advanced/`, not mixed into the active examples.

This README tells you exactly what each active example proves. The full
per-theme tracking plan lives at
[`docs/audit/coverage/COVERAGE_AUDIT.md`](../../docs/audit/coverage/COVERAGE_AUDIT.md).

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
| [`gumiho/`](gumiho/) | Gumiho (ICML'25) hybrid speculative decoding — serial 2-layer Transformer + 5 parallel MLP heads + Full Tree Attention, draft+verify on the Apple GPU/CPU backend, validated vs numpy. |
| [`Tessera_Empirical_Software_Agent/`](Tessera_Empirical_Software_Agent/) | Deterministic kernel-candidate benchmark with a numerical oracle. |
| [`Diffusion_LLM/`](Diffusion_LLM/) | Torch-free masked-diffusion denoising loop with deterministic sampling and a NumPy oracle. |
| [`Jet_nemotron/`](Jet_nemotron/) | D=128 canonical linear-attention compiler smoke. |

### Compiler smoke tests (build Graph IR directly)

These bypass the Python `@tessera.jit` surface and emit Graph IR through internal compiler hooks. Useful for compile-path testing, **not** for end-to-end runs.

| Subdir | What it tests |
|--------|---------------|
| [`Fast_dLLM_v2/`](Fast_dLLM_v2/) | Diffusion-LLM NumPy reference plus current Graph IR compiler smoke. |
| [`mla/`](mla/) | Multi-Latent Attention / FlashMLA Graph IR sketch. |
| [`Nemotron_Nano_12B_v2/`](Nemotron_Nano_12B_v2/) | Hybrid Mamba2/GQA/MLP NumPy reference plus current Graph IR compiler smoke. |

---

## Archived research surfaces

Incomplete full-model packages, placeholder passes, optional framework ports,
and superseded source drops live under `archive/examples/advanced/`. A runnable
slice is evidence only for the operation and target it actually executes.

---

## What "phantom API" means

Until the corresponding backlog items land, calling phantom names raises `NotImplementedError` with a pointer to the audit:

```python
>>> import tessera
>>> tessera.nn.Module()
NotImplementedError: tessera.nn.Module is on the Tier 1 backlog — see
docs/audit/coverage/COVERAGE_AUDIT.md. Until it lands, compose ops
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
that exists today, and the [capability-gap audit](../../docs/audit/coverage/COVERAGE_AUDIT.md)
for the prioritized compiler backlog.

---

## Archive

Older, duplicate, or superseded drops:

- `archive/examples/advanced/`
- `archive/examples/advanced/consolidated_rl_sources/` — RL source drops folded into `rlvr_reasoning_suite/`
