# Tessera Interop & Tooling Guide
## Chapter 3: External ML Framework Interop

---

### 3.1 Overview

Tessera is designed to **interoperate with existing ML frameworks**.  
This enables gradual adoption: models written in **PyTorch, JAX, or Hugging Face** can offload operators to Tessera while retaining their ecosystem.

---

### 3.2 PyTorch Interop

Tessera integrates with PyTorch via **custom operators**:

```python
import torch
from tessera.torch import op as t_op

class TesseraLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return t_op.matmul(x, self.W)
```
PyTorch tensors ↔ Tessera tensors converted automatically.

	•	Autograd hooks map to Tessera’s autodiff engine.

TorchScript Compatibility

```python
@torch.jit.script
def forward_fn(x, W):
    return t_op.matmul(x, W)
    ```
3.3 JAX Interop

Tessera integrates with JAX via XLA custom calls:
```python
import jax
import jax.numpy as jnp
from tessera.jax import op as t_op

@jax.jit
def step(x, W):
    return t_op.matmul(x, W)
```
JAX → XLA IR → Tessera lowering.

	•	Gradients flow through Tessera ops using JAX’s autodiff.
	•	Supports pmap and pjit with distributed Tessera tensors.

⸻

3.4 Hugging Face Transformers Interop

Tessera can accelerate large language models directly:

```python
from transformers import AutoModelForCausalLM
from tessera.hf import accelerate

model = AutoModelForCausalLM.from_pretrained("gpt2")
accelerate(model)   # Replace attention & matmul ops with Tessera kernels
```
- Replaces attention, matmul, and softmax with FlashAttention-like Tessera kernels.
- Supports training and inference.
- Transparent to the Hugging Face API.

3.5 Example: Mixed Stack

A hybrid workflow:

```python
# PyTorch dataloader → JAX preprocessing → Tessera compute
batch = torch.randn(32, 1024)

# Convert to JAX
import jax.numpy as jnp
x = jnp.array(batch.numpy())

# Run Tessera kernel
from tessera import op
y = op.softmax(op.matmul(x, x.T))
```

3.6 Advantages of Tessera Interop

	•	Seamless: drop-in acceleration.
	•	Composable: mix with PyTorch, JAX, Hugging Face layers.
	•	Future-proof: as Tessera expands, more ops offload automatically.

⸻

3.7 Summary

	•	Tessera integrates with PyTorch, JAX, and Hugging Face.
	•	Operators can be swapped transparently.
	•	Enables gradual migration and acceleration of existing models.
