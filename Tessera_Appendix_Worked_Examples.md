# Tessera Programming Guide
# Appendix. Worked Examples

---

## A.1 Overview

This appendix provides **complete worked examples** in Tessera’s programming model.  
The goal is to demonstrate how Tessera’s abstractions (operators, meshes, ShardSpec, adjoints) apply to real-world workloads such as:  

- Attention / FlashAttention kernels  
- Spectral denoising  
- Mixture of Recursions (MoR)  
- Mixture of Experts (MoE)  
- Test-time compute (speculative decoding)  
- SFT + RL training loop  
- Hugging Face GPT integration  
- Physics-Informed Neural Networks (PINNs)  

---

## A.2 FlashAttention Example

FlashAttention fuses `softmax ∘ matmul ∘ matmul` into one tile kernel.  

```python
from tessera import op

def flash_attention(Q, K, V):
    # Algebraic fusion rule: softmax(QK^T) @ V
    return op.flash_attention(Q, K, V)
```

Compared to CUDA, user does not write low-level tiling; Tessera rewrites into fused kernels.  

---

## A.3 Spectral Denoising Example

```python
from tessera import op

def spectral_denoise(X, threshold=0.1):
    Xf = op.fft(X)
    Xf = op.threshold(Xf, threshold)
    return op.ifft(Xf)
```

This demonstrates Tessera’s ability to move seamlessly between spatial and spectral domains.  

---

## A.4 Mixture of Recursions (MoR)

MoR combines recursive Hilbert operators with learned modules.  

```python
from tessera import op

experts = [op.mlp(in_dim=D, hidden=[2048], out_dim=D) for _ in range(8)]
MoR = op.recursive_mixture(experts, depth=4)

Y = MoR(X)
```

The recursion depth and branching are algebraically encoded, enabling efficient parallel execution.  

---

## A.5 Mixture of Experts (MoE)

```python
from tessera import op

mesh = dist.Mesh(axes=["experts","dp"], devices=range(16))

experts = [op.mlp(in_dim=D, hidden=[4096], out_dim=D) for _ in range(16)]
router = op.router(policy="top2")

Y = op.moe(X, experts=experts, router=router, mesh=mesh)
```

Unlike PyTorch/JAX, MoE routing is algebraically represented and compiler-optimized.  

---

## A.6 Test-Time Compute: Speculative Decoding

```python
from tessera import op

draft_model = op.transformer(hidden=2048, layers=12)
target_model = op.transformer(hidden=4096, layers=48)

def speculative_decode(input_tokens):
    draft = draft_model(input_tokens)
    refined = op.verify_and_correct(draft, target_model)
    return refined
```

Tessera fuses verification with sampling into one deterministic operator graph.  

---

## A.7 SFT + RL Training Loop

```python
from tessera import op, graph

policy = op.transformer(hidden=4096, layers=48)
ref_model = op.copy(policy)

@graph.training_step(module="SFT")
def sft_step(batch):
    logits = policy(batch["input"])
    loss = op.cross_entropy(logits, batch["labels"])
    grads = graph.backward(loss)
    return grads, {"loss": loss}

@graph.training_step(module="RL")
def rl_step(batch, reward_model):
    logits = policy(batch["input"])
    reward = reward_model(logits, batch["labels"])
    loss = -op.mean(reward)
    grads = graph.backward(loss)
    return grads, {"loss": loss}
```

Both SFT and RL are expressed as first-class operator graphs, preserving determinism.  

---

## A.8 Hugging Face Transformers Integration

Tessera provides shims for Hugging Face models:  

```python
from tessera import hf

# Load Hugging Face GPT-OSS-120B
model = hf.load("gpt-oss-120b", backend="tessera")

# Compile into Tessera operator graph
compiled_model = tessera.compile(model)

# Training step
@graph.training_step(module="GPT-OSS-120B")
def train_step(batch):
    logits = compiled_model(batch["input"])
    loss = op.cross_entropy(logits, batch["labels"])
    grads = graph.backward(loss)
    return grads, {"loss": loss}
```

This allows reusing existing HF checkpoints while benefiting from Tessera’s determinism.  

---

## A.9 Physics-Informed Neural Networks (PINNs)

Example: 2D Navier–Stokes PINN with stream-function formulation.  

```python
psi = op.mlp(in_dim=2, hidden=[256,256,256], out_dim=1)

# Stream-function derivatives
u = op.grad(psi, wrt="y")
v = -op.grad(psi, wrt="x")

# Incompressibility: div(u,v)=0
incomp = op.grad(u,"x") + op.grad(v,"y")

# PDE residual (Navier–Stokes)
residual = op.laplacian(psi) - forcing

# Loss: PDE residual + incompressibility
loss = op.reduce_sum(residual**2 + incomp**2, stable=True)
```

Adjoints are exact, ensuring stable gradient propagation in PINNs.  

---

## A.10 Summary

- Tessera’s operator model unifies ML, spectral, recursive, and PDE workloads.  
- FlashAttention, MoE, and speculative decoding are naturally fused.  
- SFT+RL and Hugging Face integration show training loops at scale.  
- PINNs benefit from deterministic adjoints for PDE constraints.  

