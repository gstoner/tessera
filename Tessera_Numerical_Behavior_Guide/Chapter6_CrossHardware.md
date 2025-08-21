# Tessera Numerical Behavior Guide
## Chapter 6: Cross-Hardware Consistency

---

### 6.1 Motivation

Deep learning workloads often span **heterogeneous hardware**:
- NVIDIA GPUs (PTX backend).
- AMD GPUs (ROCm LLVM backend).
- CPU fallback (LLVM CPU backend).

Without explicit controls, floating-point differences in **rounding, flush-to-zero, and math library implementations** can cause divergent results.  
Tessera provides tools for **cross-hardware reproducibility**.

---

### 6.2 Numerical Profiles

Tessera exposes **numerical profiles** to enforce consistent behavior:

```python
from tessera import numerics

numerics.profile("fast")          # Max performance, hardware defaults
numerics.profile("deterministic") # Ordered reductions, consistent rounding
numerics.profile("strict")        # Bitwise reproducibility across backends
```

6.3 Sources of Inconsistency
	•	Rounding modes: IEEE-754 vs hardware fast-math shortcuts.
	•	Flush-to-zero (FTZ): Handling of denormal values differs across vendors.
	•	Math library differences: exp, log, tanh approximations differ.
	•	Atomic reductions: Different orderings produce diverging results.

⸻

6.4 Tessera Enforcement Mechanisms
	•	MLIR Lowering
	•	Inserts explicit fadd.round and fmul.round ops.
	•	Forces identical rounding/FTZ behavior.
	•	Standardized Collectives
	•	Cross-GPU collectives use fixed-order trees/rings.
	•	Same reduction order across NVIDIA and AMD.
	•	Math Intrinsics Harmonization
	•	Tessera ships its own numerically stable math library for exp, log, etc.
	•	Ensures identical results across PTX and ROCm backends.

⸻

6.5 Example: Strict Mode Across NVIDIA & AMD

```python
from tessera import numerics, op

numerics.profile("strict")

A = op.tensor((B, D), dtype="bf16")
B = op.exp(A)     # Uses Tessera’s stable exp, not vendor intrinsic
```
	•	Guarantees bitwise identical results between PTX and ROCm.

⸻

6.6 Testing & Validation Tools

Tessera includes validation utilities:

```python
from tessera import numerics

numerics.validate_cross_hardware(model, backends=["ptx","rocm","cpu"])
```

	•	Runs forward/backward pass on each backend.
	•	Compares outputs within strict tolerance (or bitwise if strict).
	•	Reports discrepancies.

6.7 Performance vs. Consistency
	•	fast → fastest, hardware-specific shortcuts enabled.
	•	deterministic → stable order, consistent results on the same hardware.
	•	strict → guarantees reproducibility across vendors, ~5–20% slower.

⸻

6.8 Best Practices
	•	Use strict profile when validating scientific models or benchmarks.
	•	Use deterministic profile for reproducible ML training on one vendor.
	•	Use fast profile for production inference where small deviations are tolerable.

⸻

6.9 Summary
	•	Hardware differences (rounding, FTZ, math libs) can cause divergent results.
	•	Tessera profiles (fast, deterministic, strict) provide explicit control.
	•	Cross-hardware reproducibility is essential for portability and science.
	•	Tessera harmonizes math intrinsics, collectives, and rounding for consistency.
    