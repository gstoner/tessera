# Tessera Performance Best Practices Guide
## Chapter 5: Operator Fusion and Pipelines

---

### 5.1 Overview

**Operator fusion** reduces memory traffic and kernel launch overhead by merging multiple operations into a single kernel.  
**Pipelining** overlaps computation with memory transfers to hide latency.

In Tessera, fusion and pipelines are expressed at the **Graph IR** and **Schedule IR** levels, then lowered into fused Tile IR kernels.

---

### 5.2 Benefits of Fusion

- **Reduced HBM traffic**: Intermediate tensors remain in registers/shared memory.  
- **Fewer kernel launches**: Reduces scheduling overhead.  
- **Better locality**: Reuse data within a fused kernel.  

---

### 5.3 Fusion Tradeoffs

- **Pros**: Higher arithmetic intensity, lower bandwidth usage.  
- **Cons**: Increased register usage, possible spills, lower occupancy.  
- **Balance**: The Tessera autotuner evaluates fusion depth with cost models.  

---

### 5.4 Fusion Types in Tessera

1. **Elementwise Fusion**  
   - Chains of add/mul/activation → fused into a single kernel.  
   - Example: `Y = gelu(X @ W + b)` → one fused kernel.  

2. **Reduction Fusion**  
   - Combine reductions with adjacent ops (e.g., `softmax` inside attention).  

3. **Pipeline Fusion**  
   - Multiple stages (load → compute → store) overlapped with double/triple buffering.  

---

### 5.5 Fusion in Schedule IR

```python
from tessera import schedule, op

@op.kernel
def fused_attention(Q, K, V):
    schedule.fuse(["matmul", "softmax", "matmul"])   # QK^T → softmax → softmax*V
    schedule.pipeline_stages(3)                      # triple-buffered
    schedule.async_copy(True)
    return op.flash_attention(Q, K, V)
```

This schedule lowers to a single kernel with:
	•	Shared memory tiling for Q/K/V.
	•	Warp-level reductions for softmax.
	•	Triple-buffer pipeline overlapping loads and matmul compute.

⸻

5.6 Pipelines for Latency Hiding
	•	Global → Shared → Register pipeline is the standard.
	•	Tessera allows multi-stage pipelines (2, 3, or 4 stages).
	•	Async copies (cp.async in PTX) are automatically generated when schedule.async_copy(True) is enabled.

Example (pipelined matmul):

```python
schedule.tile((128, 128, 64))
schedule.pipeline_stages(3)    # triple buffering
schedule.async_copy(True)      # overlap loads with compute
```
5.7 Worked Example: Fused MLP Block

MLP = Y = activation(X @ W1 + b1) @ W2 + b2

Naive: 3 kernels (matmul → activation → matmul).
Fused: 1 kernel with staged pipelines.

```python
@op.kernel
def fused_mlp(X, W1, W2, b1, b2):
    schedule.fuse(["matmul", "gelu", "matmul"])
    schedule.pipeline_stages(2)
    return (X @ W1 + b1).gelu() @ W2 + b2
```
Performance Results:
	•	2× less HBM traffic.
	•	25–30% speedup on Hopper/GB200.
	•	Autotuner adjusts tile shapes to maintain register usage <64k per block.

⸻

5.8 Best Practices
	•	Use fusion for bandwidth-bound ops (MLPs, attention).
	•	Avoid over-fusion that causes register spills.
	•	Always enable pipelining for large-tile matmuls or convolutions.
	•	Combine fusion + autotuning for best results.

⸻

5.9 Key Takeaways
	•	Fusion eliminates unnecessary global memory traffic.
	•	Pipelines hide latency with async copies.
	•	Tessera autotuner balances fusion depth vs occupancy automatically.
	•	Most ML workloads (MLPs, attention, convs) benefit greatly from aggressive fusion + pipelines.
