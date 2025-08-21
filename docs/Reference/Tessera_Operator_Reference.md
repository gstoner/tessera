# Tessera Standard Library Operator Reference

This document describes the built-in operators and functions available in the Tessera programming model. 
Operators are organized by category, with signatures, parameter descriptions, return values, notes, and examples.

---

## 1. Tensor Creation & Manipulation

### `op.tensor`
```python
op.tensor(shape, dtype="fp32")
```
Create a new tensor.

**Parameters**  
- `shape`: Tuple of ints, tensor dimensions.  
- `dtype`: Data type (`"fp32"`, `"bf16"`, `"int32"`, `"bool"`, etc.).  

**Returns**  
- Tensor of given shape and dtype.

---

### `op.reshape`
```python
op.reshape(tensor, new_shape)
```
Reshape a tensor without changing memory layout.

**Example**  
```python
x = op.tensor((2, 3))
y = op.reshape(x, (3, 2))
```

---

### `op.transpose`
```python
op.transpose(tensor, axes)
```

Transpose tensor dimensions.

---

### `op.concat`
```python
op.concat([tensors], axis)
```
Concatenate a list of tensors along an axis.

---

### `op.split`
```python
op.split(tensor, num_splits, axis)
```

Split tensor into multiple parts.

---

## 2. Linear Algebra

### `op.matmul`
```python
op.matmul(A, B)
```
Matrix multiplication.

**Example**  
```python
C = op.matmul(A, B)
```

---

### `op.factorized_matmul`
```python
op.factorized_matmul(A, B, rank)
```
Low-rank factorized matrix multiplication.

---

### `op.flash_attention`
```python
op.flash_attention(Q, K, V, causal=False)
```
Memory-efficient attention kernel.

---

### `op.norm`
```python
op.norm(tensor, p=2)
```

Vector or matrix norm.

---

### `op.svd`
```python
op.svd(tensor)
```
Singular Value Decomposition.

---

## 3. Spectral & Transform Operators

### `op.fft` / `op.ifft`
```python
op.fft(tensor)
op.ifft(tensor)
```

Forward and inverse Fast Fourier Transform.

---

### `op.wavelet`
```python
op.wavelet(tensor, type="haar")
```

Wavelet transform.

---

### `op.spectral_filter`
```python
op.spectral_filter(tensor, cutoff)
```

Apply frequency-domain filter.

---

## 4. Autodiff & Gradient Ops

### `op.grad`
```python
op.grad(output, wrt)
```

Compute gradient of `output` w.r.t. input(s).

---

### `op.jvp` / `op.vjp`
```python
op.jvp(fn, inputs)
op.vjp(fn, inputs)
```

Jacobian-vector product (forward mode) and vector-Jacobian product (reverse mode).

---

### `op.check_numerics`
```python
op.check_numerics(tensor)
```

Detect NaN/Inf in a tensor.

---

## 5. Neural Network Primitives

### `op.mlp`
```python
op.mlp(in_dim, hidden, out_dim, activation="gelu")
```

Create a multi-layer perceptron.

---

### `op.conv2d`
```python
op.conv2d(input, weight, stride, padding)
```

2D convolution.

---

### `op.batch_norm`
```python
op.batch_norm(input, weight, bias)
```

Batch normalization.

---

### `op.dropout`
```python
op.dropout(input, p)
```

Dropout regularization.

---

### `op.softmax`
```python
op.softmax(input, axis)
```

Softmax activation.

---

## 6. Mixture-of-Experts & Routing

### `op.router`
```python
op.router(experts, strategy="top2")
```

Route inputs to experts.

---

### `op.recursive_mixture`
```python
op.recursive_mixture(experts, depth)
```

Recursive Mixture-of-Experts.

---

### `op.moe_layer`
```python
op.moe_layer(input, experts)
```

Standard MoE layer.

---

## 7. Distributed Ops

### `dist.tensor`
```python
dist.tensor(shape, layout, mesh, dtype)
```

Distributed tensor creation.

---

### `dist.all_reduce`
```python
dist.all_reduce(tensor, op="sum")
```

Collective all-reduce.

---

### `dist.broadcast`
```python
dist.broadcast(tensor, root=0)
```

Broadcast tensor to all devices.

---

### `dist.shard`
```python
dist.shard(tensor, mesh_axes)
```

Shard tensor along mesh axes.

---

## 8. Loss Functions

### `op.cross_entropy`
```python
op.cross_entropy(pred, target)
```

Cross-entropy loss.

---

### `op.mse`
```python
op.mse(pred, target)
```

Mean Squared Error loss.

---

### `op.kl_divergence`
```python
op.kl_divergence(p, q)
```

Kullback-Leibler divergence.

---

## 9. Scheduling & Fusion Hints

### `schedule.hint`
```python
schedule.hint(fusion="aggressive", tile_sizes=[...])
```

Set scheduling hints.

---

### `schedule.pipeline`
```python
schedule.pipeline(stages, overlap=True)
```

Define execution pipeline.

---

# Summary
This operator reference serves as a **quick-access catalog** of Tessera’s built-in functions.  
Each operator is fully compatible with Tessera’s IR layers, autodiff, and distributed runtime.
