# Tessera Numerical Behavior Guide
## Chapter 5: Stability in Neural Operators

---

### 5.1 Why Stability Matters in Operators

Neural network operators are sensitive to floating-point errors:
- **Softmax** can overflow or underflow.  
- **LayerNorm/BatchNorm** require stable variance estimates.  
- **PDE adjoints** amplify small numerical errors in iterative solvers.  

Tessera integrates **stability-enhanced operator implementations**.

---

### 5.2 Stable Softmax

Naive softmax:
```
softmax(x)_i = exp(x_i) / Σ exp(x_j)
```

- Risk of overflow if `x_i` is large.  
- Risk of underflow if `x_i` is small.  

Stable version subtracts max:

```python
def stable_softmax(x):
    m = op.max(x, axis=-1, keepdims=True)
    exps = op.exp(x - m)
    return exps / op.sum(exps, axis=-1, keepdims=True)
```
Tessera lowers op.softmax to this stable form automatically.

⸻

5.3 Normalization Operators

LayerNorm
	•	Mean and variance reduced in FP32.
	•	Uses two-pass reduction for accuracy.

```python
Y = op.layer_norm(X, dtype="fp32_accum")
```
BatchNorm
	•	Running statistics accumulated in FP32.
	•	Deterministic updates when numerics.policy("deterministic").

⸻

5.4 Attention Mechanisms

FlashAttention-like kernels require:
	•	Stable softmax (max subtraction).
	•	FP32 accumulation in dot-products.
	•	Careful scaling with mixed precision.

```python 
QK = op.matmul(Q, K.T, dtype="fp32")   # FP32 dot-product
A  = op.softmax(QK)                    # stable softmax
O  = op.matmul(A, V)
```

5.5 PDE Operators & Adjoint Stability

Physics-Informed Neural Networks (PINNs) and spectral PDE solvers require:
	•	Stable discretizations (avoid catastrophic cancellation).
	•	FP64 where adjoints amplify error.
	•	Tessera supports precision overrides per operator:

```python
U = op.laplacian(phi, precision="fp64")    # stable PDE operator
```
Adjoint propagation also honors stability:

```python
grad_phi = graph.backward(loss, wrt=phi, precision="fp64")
```

5.6 Spectral Operators

Fourier / Wavelet transforms:
	•	Scaling normalized to preserve energy (L2 norm).
	•	Avoids bias in spectral-domain training.

```python 
Xf = op.fft(X, normalize="ortho")
```

5.7 Example: Stable Attention Block
```python
from tessera import op

def attention(Q, K, V):
    QK = op.matmul(Q, K.T, dtype="fp32")   # FP32 stability
    A  = op.softmax(QK)                    # numerically stable softmax
    O  = op.matmul(A, V, dtype="fp32")     # safe accumulation
    return O
``` 

5.8 Best Practices
	•	Always use stable softmax (default in Tessera).
	•	Accumulate LayerNorm/BatchNorm stats in FP32.
	•	For attention: FP32 dot-products, FP16/BF16 storage.
	•	PDE adjoints may require FP64 precision.
	•	Normalize FFTs for spectral stability.

⸻

5.9 Summary
	•	Neural operators are prone to instability.
	•	Tessera integrates stability-enhanced kernels:
	•	Safe softmax, FP32 accumulations, FP64 PDE adjoints.
	•	Users can override precision per operator.
	•	Stability-aware operators are essential for both ML and physics.
    