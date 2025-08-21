# Tessera Numerical Behavior Guide
## Chapter 4: Stability in Reductions

---

### 4.1 The Problem: Floating-Point Non-Associativity

Floating-point addition is **not associative**:
(a + b) + c  ≠  a + (b + c)

```
- In parallel reductions, the order of operations depends on GPU scheduling.  
- Results may differ across runs, even with identical inputs.  
- Large-scale training can accumulate small errors into instability.

---

### 4.2 Sources of Instability

- **Summation of gradients** over large batches.  
- **Dot products** in attention and MLP layers.  
- **Optimizer updates** when summing small deltas into large weights.  
- **PDE solvers** where roundoff propagates through adjoint operators.

---

### 4.3 Tessera Reduction Strategies

Tessera provides three key modes for summations:

#### (1) Deterministic Reduction Trees
- Fixed binary-tree reduction order.  
- Guarantees reproducibility across runs.  

```python
from tessera import numerics
numerics.policy("deterministic")
~~~

(2) Compensated Summation (Kahan/Babylonian)

Reduces roundoff error by tracking small residuals:

```python
numerics.policy("kahan_sum")
```
Algorithm:
```
sum = 0
c = 0
for x in data:
    y = x - c
    t = sum + y
    c = (t - sum) - y
    sum = t
```

(3) Chunked Reductions
	•	Partition input into blocks.
	•	Reduce each block in higher precision.
	•	Combine results in FP32/FP64.

⸻

4.4 Mixed Precision Reductions

Tessera automatically promotes critical reductions (like gradient accumulation, normalization statistics) to FP32:

```python
A = op.tensor((B, D), dtype="bf16")
mean = op.reduce_mean(A, dtype="fp32")   # safe accumulation
```

4.5 Example: Stable Gradient Accumulation

```python
rom tessera import op, numerics

numerics.policy("kahan_sum")

# Accumulate gradients safely
grads = [op.tensor((D,), dtype="bf16") for _ in range(N)]
total = op.reduce_sum(grads, dtype="fp32")
```

4.6 Performance vs. Stability Tradeoffs
	•	Naive reduction: fastest, lowest stability.
	•	Tree reductions: deterministic, moderate overhead.
	•	Kahan summation: most stable, but up to 2× slower.
	•	Chunked reductions: balance between accuracy and performance.

⸻

4.7 Best Practices
	•	For debugging/research: use kahan_sum or deterministic mode.
	•	For large-scale training: enable FP32 accumulation + chunked reductions.
	•	For PDE operators: always use compensated summation for adjoints.
	•	Validate training with strict numerics before switching to fast mode.

⸻

4.8 Summary
	•	Floating-point reductions are inherently unstable.
	•	Tessera exposes multiple reduction strategies:
	•	Deterministic trees for reproducibility.
	•	Compensated summation for accuracy.
	•	Chunked reductions for performance/stability balance.
	•	Choosing the right reduction mode is critical for stable deep learning.

⸻
