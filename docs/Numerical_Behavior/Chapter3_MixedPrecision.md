# Tessera Numerical Behavior Guide
## Chapter 3: Mixed Precision Arithmetic

---

### 3.1 Motivation

Training large models requires balancing **performance, memory footprint, and numerical stability**.  
Mixed precision leverages lower-bit datatypes (FP16, BF16, FP8) while preserving accuracy via FP32 accumulation and scaling.

---

### 3.2 Precision Hierarchy

| Type   | Bits | Range Characteristics | Typical Use |
|--------|------|-----------------------|-------------|
| FP32   | 32   | Wide range, high precision | Accumulation, normalization, reference |
| BF16   | 16   | Same range as FP32, fewer mantissa bits | Training activations, weights |
| FP16   | 16   | Narrower range, faster math on GPUs | Training on memory-constrained systems |
| FP8    | 8    | Ultra-low precision, requires scaling | Frontier-scale trillion-parameter models |

---

### 3.3 Tessera Precision Policies

Tessera controls precision globally:

```python
from tessera import numerics

numerics.policy("fp32")   # High precision everywhere
numerics.policy("mixed")  # BF16/FP16 inputs, FP32 accumulation
numerics.policy("fp8")    # FP8 compute, BF16 accumulation

3.4 Automatic Casting

Tessera automatically casts inputs to the chosen mode:

```python
A = op.tensor((B, D), dtype="bf16")
B = op.tensor((D, H), dtype="bf16")
C = op.matmul(A, B)       # Internally: FP32 accumulation
```
3.5 Loss Scaling

Low precision risks underflow during gradient updates.
Tessera integrates dynamic loss scaling:

```python
from tessera import training

training.enable_loss_scaling(dynamic=True)
```
3.6 Precision in Optimizers

Best practice:
	•	Store master weights in FP32.
	•	Update with FP32 precision.
	•	Cast back to FP16/BF16 for forward compute.

Example:
```python
W = op.tensor((D, H), dtype="bf16", master_dtype="fp32")
optimizer.step(W)   # Updates master FP3
```
3.7 FP8 Considerations

FP8 is experimental but supported for ultra-large models:
	•	Tessera enforces block scaling (per-tensor or per-channel).
	•	Accumulations performed in BF16 or FP32.
	•	Autotuner selects optimal scaling policy per shape/device.

```python
numerics.policy("fp8")
op.matmul(A, B, scale="per_channel")
```

3.8 Example: Mixed Precision Training Loop

```python
from tessera import numerics, graph, op, training

numerics.policy("mixed")
training.enable_loss_scaling(dynamic=True)

@graph.training_step(module="MixedModel")
def step(batch):
    out = model(batch["input"])      # FP16/BF16 compute
    loss = op.cross_entropy(out, batch["labels"])
    grads = graph.backward(loss)     # FP32 accumulation
    return grads, {"loss": loss}
```
3.9 Best Practices

	•	Use BF16 over FP16 when available (wider dynamic range).
	•	Always keep accumulations in FP32.
	•	Enable loss scaling for FP16/FP8.
	•	Store master weights in FP32.
	•	Validate accuracy when moving to FP8.

⸻

3.10 Summary

	•	Mixed precision is critical for scaling deep learning.
	•	Tessera exposes precision via policies.
	•	Loss scaling and FP32 accumulations preserve stability.
	•	FP8 enables trillion-parameter training, with block scaling for safety.

⸻
