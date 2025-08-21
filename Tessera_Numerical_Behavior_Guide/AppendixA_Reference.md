# Tessera Numerical Behavior Guide
## Appendix A: Reference Tables

---

### A.1 Precision Modes

| Mode     | Storage | Compute | Accumulation | Use Case |
|----------|---------|---------|--------------|----------|
| `fp64`   | FP64    | FP64    | FP64         | Scientific computing, PDE solvers |
| `fp32`   | FP32    | FP32    | FP32         | Debugging, baselines |
| `bf16`   | BF16    | BF16    | FP32         | Default for large-scale training |
| `fp16`   | FP16    | FP16    | FP32         | Vision models, inference |
| `fp8`    | FP8     | FP8     | BF16 / FP32  | Trillion-param LLMs, low memory |
| `int8`   | INT8    | INT8    | FP32         | Quantized inference |

---

### A.2 Determinism Policies

| Policy           | Description                                      | Performance |
|------------------|--------------------------------------------------|-------------|
| `fast`           | Hardware default; may reorder ops, use FTZ.      | Fastest |
| `deterministic`  | Ordered reductions, reproducible on one backend. | Medium |
| `strict`         | Bitwise reproducible across backends.            | Slowest |

---

### A.3 Reduction Strategies

| Strategy         | Description                          | Accuracy | Cost |
|------------------|--------------------------------------|----------|------|
| Naive sum        | Hardware reduction, unordered        | Low      | Low  |
| Tree reduction   | Fixed binary tree order              | Medium   | Med  |
| Chunked reduction| Blockwise FP32 reduction             | High     | Med  |
| Kahan summation  | Compensated sum with residual term   | Highest  | High |

---

### A.4 Stability Tricks per Operator

| Operator   | Stability Enhancement       | Default in Tessera |
|------------|-----------------------------|--------------------|
| Softmax    | Max-subtraction             | ✅ |
| LayerNorm  | FP32 variance accumulation  | ✅ |
| BatchNorm  | FP32 running stats          | ✅ |
| MatMul     | FP32 accumulation           | ✅ |
| FFT/IFFT   | Orthonormal normalization   | ✅ |
| PDE Laplacian | FP64 override available  | ✅ |

---

### A.5 Recommended Defaults

- **Training**: `bf16` compute, FP32 accumulations, deterministic profile.  
- **Inference**: `fp16` or `int8` with validated quantization.  
- **Scientific**: FP64 operators, strict profile.  
- **Debugging**: FP32 everywhere, strict profile.  

---

### A.6 Quick Reference Snippets

#### Set Numerical Profile
```python
from tessera import numerics
numerics.profile("deterministic")
```
Enable Compensated Reductions
```python
numerics.policy("kahan_sum")
```
Safe LayerNorm
```python
Y = op.layer_norm(X, dtype="fp32_accum")
```
A.7 Summary

These tables provide a quick reference to Tessera’s numerical controls:

	•	Precision defaults.
	•	Determinism policies.
	•	Stable operator implementations.
	•	Best practices for training, inference, and scientific workloads.