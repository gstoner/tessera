# Tessera QA & Reliability Guide  
## Chapter 1: Introduction & Device-Scale QA

---

## 1.1 Introduction  

Tessera’s design philosophy emphasizes **correctness, determinism, and stability** across every scale of deployment — from a single GPU to clusters with thousands of devices.  

This guide provides **practical best practices** for testing and validating Tessera programs. Unlike formal specs, the focus here is **hands-on reliability engineering**:  
- How to test for correctness.  
- How to ensure reproducibility.  
- How to catch and debug failures.  
- How to validate performance against expectations.  

---

## 1.2 Device-Scale QA Goals  

On a **single GPU**, QA focuses on:  
1. **Correctness** — ensure operators return expected values.  
2. **Numerical stability** — detect NaNs, overflows, and unstable mixed precision.  
3. **Determinism** — verify bitwise reproducibility when given fixed seeds.  
4. **Fault tolerance** — catch illegal memory accesses and runaway kernels.  
5. **Performance consistency** — validate that kernels achieve expected utilization.  

---

## 1.3 Correctness Testing  

Every Tessera operator should be validated against a **golden CPU reference** implementation.  

**Example: Unit Test for MatMul**
```python
import numpy as np
from tessera import op, tensor

# Reference CPU result
A = np.random.randn(128, 128).astype(np.float32)
B = np.random.randn(128, 128).astype(np.float32)
ref = A @ B

# Tessera execution
TA = tensor.from_numpy(A)
TB = tensor.from_numpy(B)
out = op.matmul(TA, TB).numpy()

# Compare with tolerance
np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
```

---

## 1.4 Numerical Stability  

Tessera supports FP32, TF32, BF16, and FP16. QA should check for:  
- NaN/Inf propagation.  
- Catastrophic cancellation (subtraction of close values).  
- Mixed-precision consistency (BF16/FP16 vs FP32 baseline).  

**Tip:** Always test critical ops (softmax, log, exp) with both random and edge-case inputs.  

**Example: Detect NaNs**
```python
Y = op.softmax(X, axis=-1)
assert not np.isnan(Y.numpy()).any(), "NaN detected in softmax output"
```

---

## 1.5 Determinism Testing  

Determinism is critical for debugging and regression testing. Tessera provides a deterministic execution context:  

```python
from tessera import dist, op

with dist.deterministic():
    y1 = op.all_reduce(x, op="sum")
    y2 = op.all_reduce(x, op="sum")
    assert (y1 == y2).all(), "Non-deterministic result detected"
```

Best practices:  
- Fix seeds for RNG.  
- Use Tessera’s deterministic mode for collectives.  
- Run regression tests multiple times to confirm reproducibility.  

---

## 1.6 Fault Tolerance  

Tessera catches illegal memory accesses and runaway kernels:  
- **Memory out-of-bounds** → raises runtime error with backtrace.  
- **Watchdog timers** → configurable timeout aborts infinite loops.  

**Example: Expected Fault**
```python
try:
    bad = op.gather(tensor.zeros((10,)), indices=[100])  # OOB index
except RuntimeError as e:
    print("Caught expected error:", e)
```

---

## 1.7 Performance Consistency  

Even at device scale, performance regressions can creep in. Tessera QA should track:  
- Kernel occupancy.  
- Global memory throughput.  
- Achieved FLOPs vs theoretical peak.  

**Example: Simple Perf Check**
```python
from tessera import profile

with profile.session() as sess:
    y = op.matmul(A, B)
    sess.report()
```

This outputs utilization metrics (SM occupancy, memory bandwidth). Store and compare across versions.  

---
