# Tessera Numerical Behavior Guide
## Chapter 7: Best Practices & Recommendations

---

### 7.1 Debugging Numerical Behavior

When debugging unstable training or PDE solvers:
- Enable **deterministic reductions** to eliminate run-to-run noise.
- Compare against **strict profile** for cross-backend consistency.
- Use **FP32 accumulation** in all reductions to isolate underflow issues.

```python
from tessera import numerics

numerics.profile("deterministic")
```
7.2 When to Use Strict Determinism
	•	Research reproducibility: publishing models, comparing baselines.
	•	Scientific computing: PINNs, CFD, quantum simulations.
	•	Debugging: ruling out hardware-specific divergence.

Strict mode guarantees bitwise identical runs, even across NVIDIA and AMD.

⸻

7.3 Mixed Precision Recipes

|Model Type |Recommended Policy | Notes|
|------------ ----------------|--------------------------|-----------------------------|
|Large Language Models (LLMs)| bf16 compute, FP32 accum| BF16 avoids FP16 overflow | issues
|Vision Transformers (ViTs) | fp16 compute, FP32 accum | Use dynamic loss scaling |
|Trillion-parameter models | fp8 compute, BF16 accum| Requires per-channel scaling |
PINNs / PDE solvers | FP32 / FP64 everywhere| Accuracy prioritized |

7.4 Stable Reductions
	•	Always prefer chunked reductions or Kahan summation when summing across very large batches.
	•	For optimizers, store master weights in FP32.
	•	For statistics (mean/variance): accumulate in FP32 even if inputs are FP16/BF16.

⸻

7.5 Validating Across Hardware

Use Tessera’s validation utilities:

```python
from tessera import numerics

numerics.validate_cross_hardware(model,
                                 backends=["ptx","rocm"],
                                 profile="strict")
```
Ensures your model behaves identically on NVIDIA and AMD.
	•	Essential when distributing models or moving clusters.

⸻

7.6 Best Practice Checklist

✅ Use FP32 accumulation for all critical reductions.
✅ Apply stable softmax by default.
✅ Enable dynamic loss scaling for FP16/FP8.
✅ Store master weights in FP32.
✅ Validate with deterministic/strict mode before production.
✅ Normalize FFTs to preserve energy.
✅ For PINNs, prefer FP64 operators.

⸻

7.7 Summary
	•	Tessera provides explicit policies to balance speed, stability, and reproducibility.
	•	BF16 + FP32 accum is the best default for modern training.
	•	Strict determinism is crucial for science and reproducibility.
	•	By combining mixed precision, stable reductions, and cross-hardware validation, Tessera ensures models train faster, safer, and more predictably.

⸻

