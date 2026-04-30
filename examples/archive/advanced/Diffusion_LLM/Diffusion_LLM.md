# Diffusion LLM — Tessera Example

Three production-quality diffusion language model variants implemented in pure
PyTorch, sharing a common bidirectional transformer backbone with full
[Tessera compiler](https://tessera.ai) integration.

---

## Model Variants

| Variant | Class | Key Paper |
|---------|-------|-----------|
| Masked Discrete Diffusion | `MDLM` | Shi et al. 2024, arXiv:2406.04329 |
| Continuous Embedding Diffusion | `ContinuousDiffusionLLM` | Ho et al. 2020; Li et al. 2022 |
| Rectified Flow (Flow Matching) | `FlowMatchingLLM` | Liu et al. 2022, arXiv:2209.03003 |

---

## Architecture

### Shared Backbone — `DiffusionTransformer`

```
Token IDs (B, T)
    │
    ▼
embed_tokens  ← tied to lm_head output projection
    │
    + pos_embed (learned absolute)
    │
    ▼  x N blocks
┌─────────────────────────────────────────────────┐
│  Time embed t → sinusoidal → MLP → cond (B, H)  │
│                                                  │
│  adaLN_attn(cond) → (scale_a, shift_a)          │
│  RMSNorm(x) * scale_a + shift_a                 │
│  BidirectionalAttention (causal=False, GQA)      │  ← tessera.flash_attn
│  + residual                                      │
│                                                  │
│  adaLN_mlp(cond) → (scale_m, shift_m)           │
│  RMSNorm(x) * scale_m + shift_m                 │
│  SwiGLU / GeGLU MLP                             │  ← tessera.elementwise.mlp_gate
│  + residual                                      │
└─────────────────────────────────────────────────┘
    │
    ▼
RMSNorm
    │
    ▼
hidden states (B, T, H)
```

Key design choices:
- **Bidirectional attention** (`causal=False`) — diffusion models process all positions simultaneously
- **adaLN-Single** time conditioning — DiT-style scale+shift modulation per block
- **Tied embeddings** — `lm_head.weight = embed_tokens.weight` saves parameters and improves alignment
- **GQA** — configurable `num_kv_heads < num_attention_heads` for memory efficiency

---

## MDLM — Masked Discrete Diffusion

**Forward process** (absorbing-state diffusion):
```
q(x_t | x_0) = x_0    with prob 1 - m(t)
              [MASK]   with prob m(t)
```
where `m(t)` follows a cosine or linear masking schedule.

**Loss** (ELBO with uniform-SNR reweighting):
```
L = E_t [ 1/(m(t)+ε) · CE(f_θ(x_t, t), x_0) ]  on masked positions
```

**Generation** (confidence-ordered unmasking):
1. Start fully masked: `x_T = [MASK, MASK, ..., MASK]`
2. For each step `t → t_prev`:
   - Run denoising model to get token probabilities
   - Unmask `(m(t) - m(t_prev)) / m(t)` fraction of positions
3. Return fully unmasked sequence at `t=0`

---

## Continuous Diffusion

**Forward process** (Gaussian in embedding space):
```
x_t = √ᾱ_t · embed(x_0) + √(1-ᾱ_t) · ε,   ε ~ N(0, I)
```

**Model output**: ε-prediction or x₀-prediction in embedding space.

**Sampling**: DDPM (stochastic) or DDIM (deterministic/semi-stochastic).

**Final step**: project continuous x₀_hat → vocabulary logits via tied `lm_head`.

---

## Flow Matching

**Interpolation** (Rectified Flow):
```
x_t = (1 - t) · embed(x_0) + t · ε,   ε ~ N(0, I),  t ∈ [0, 1]
```

**Target velocity** (constant along straight paths):
```
v* = ε - embed(x_0)
```

**Loss**:
```
L = E_{t,x_0,ε} [ ||v_θ(x_t, t) - v*||² ]
```

**Sampling**: Euler or Midpoint ODE integration from `t=1` to `t=0`.

---

## Package Structure

```
tessera_diffusion_llm/
├── __init__.py              ← top-level exports
├── configs.py               ← TransformerConfig, MDLMConfig, ContinuousDiffusionConfig,
│                               FlowMatchingConfig
├── kernels/
│   ├── attention.py         ← BidirectionalAttention (causal=False, GQA)
│   ├── mlp.py               ← DiffusionMLP (SwiGLU / GeGLU)
│   ├── rmsnorm.py           ← RMSNorm, AdaLNModulation, adaLN_forward
│   └── time_embed.py        ← SinusoidalTimeEmbed, CombinedTimeEmbed
├── models/
│   ├── transformer.py       ← DiffusionTransformer backbone
│   ├── mdlm.py              ← MDLM
│   ├── continuous.py        ← ContinuousDiffusionLLM
│   └── flow_match.py        ← FlowMatchingLLM
├── schedules/
│   ├── noise.py             ← beta schedules, NoiseSchedule, MaskSchedule
│   └── sampling.py          ← ddpm_step/sample, ddim_step/sample,
│                               ode_euler_step, flow_ode_sample,
│                               mdlm_step, mdlm_sample
├── training/
│   ├── losses.py            ← mdlm_elbo_loss, continuous_diffusion_loss,
│   │                           flow_matching_loss, per_timestep_loss
│   └── trainer.py           ← DiffusionTrainer, TrainerConfig, EMAModel
├── inference/
│   └── generator.py         ← DiffusionGenerator, GeneratorConfig
└── utils/
    └── __init__.py          ← count_parameters, param_summary, top_k_filter, ...

mlir/
├── diffusion_graph_ir.mlir    ← Graph IR: bidirectional flash attn + SwiGLU MLP
└── diffusion_schedule_ir.mlir ← Schedule IR: tp=2, sm_90, tile params

tests/
├── test_configs.py            ← 30+ config tests
├── test_schedules.py          ← 35+ schedule + sampler tests
├── test_models.py             ← 40+ model forward/loss/generate tests
└── test_sampling.py           ← 25+ inference wrapper + utils tests
```

---

## Quick Start

```python
from tessera_diffusion_llm import MDLM, MDLMConfig

# Tiny model for testing
cfg   = MDLMConfig.debug_tiny()
model = MDLM(cfg)

# Training step
import torch
input_ids = torch.randint(0, cfg.transformer.vocab_size, (4, 64))
loss = model.compute_loss(input_ids)
loss.backward()

# Generation
tokens = model.generate(batch_size=4, seq_len=64, num_steps=50)
print(tokens.shape)  # (4, 64)
```

```python
from tessera_diffusion_llm import (
    ContinuousDiffusionLLM, ContinuousDiffusionConfig,
    DiffusionGenerator, GeneratorConfig,
)

cfg   = ContinuousDiffusionConfig.debug_tiny()
model = ContinuousDiffusionLLM(cfg)
gen   = DiffusionGenerator(model, GeneratorConfig(num_steps=50, sampler="ddim"))
tokens = gen.generate(batch_size=8, seq_len=128)
```

```python
from tessera_diffusion_llm import FlowMatchingLLM, FlowMatchingConfig

cfg   = FlowMatchingConfig.debug_tiny()
model = FlowMatchingLLM(cfg)
tokens = model.generate(batch_size=4, seq_len=64, num_steps=20, solver="midpoint")
```

---

## Training

```python
from tessera_diffusion_llm import MDLM, MDLMConfig, DiffusionTrainer, TrainerConfig
from torch.utils.data import DataLoader, TensorDataset
import torch

# Build model
model = MDLM(MDLMConfig())

# Fake dataset for illustration
ids     = torch.randint(0, 50_000, (256, 128))
dataset = TensorDataset(ids)
loader  = DataLoader(dataset, batch_size=8, shuffle=True)

# Train
trainer = DiffusionTrainer(model, TrainerConfig(num_epochs=3, lr=3e-4))
trainer.fit(loader)
```

---

## Tessera Compiler Integration

Every attention and MLP module carries `_tessera_op` annotations:

```python
# In kernels/attention.py
class BidirectionalAttention(nn.Module):
    _tessera_op     = "tessera.flash_attn"
    _tessera_causal = False           # bidirectional — key difference from AR models

# In kernels/mlp.py
class DiffusionMLP(nn.Module):
    _tessera_op = "tessera.elementwise.mlp_gate"
```

To compile for H100:
```bash
tessera-compile \
  --input  mlir/diffusion_graph_ir.mlir \
  --sched  mlir/diffusion_schedule_ir.mlir \
  --arch   sm_90 \
  --tp     2 \
  --output build/diffusion_sm90/
```

---

## Model Sizes (approximate)

| Config | Hidden | Layers | Heads | Params |
|--------|--------|--------|-------|--------|
| debug_tiny | 256 | 4 | 4 | ~4M |
| small | 768 | 12 | 12 | ~117M |
| medium | 1024 | 24 | 16 | ~345M |
| large | 1280 | 36 | 20 | ~762M |

---

## References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models*. arXiv:2006.11239
- Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models*. arXiv:2102.09672
- Li et al. (2022). *Diffusion-LM Improves Controllable Text Generation*. arXiv:2205.14217
- Austin et al. (2021). *Structured Denoising Diffusion for Discrete State Spaces*. arXiv:2107.03006
- Liu et al. (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. arXiv:2209.03003
- Lipman et al. (2022). *Flow Matching for Generative Modeling*. arXiv:2210.02747
- Shi et al. (2024). *Simplified and Generalized Masked Diffusion for Discrete Data*. arXiv:2406.04329
