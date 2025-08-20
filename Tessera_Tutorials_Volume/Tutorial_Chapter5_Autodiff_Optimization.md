# Tessera Tutorials Volume
## Chapter 5 — Autodiff & Optimization

### 5.1 Forward vs Reverse Mode
Tessera provides both **forward-mode** and **reverse-mode** automatic differentiation (AD).  

- **Forward Mode**: Efficient for functions with few inputs and many outputs.  
- **Reverse Mode**: Efficient for deep neural networks with many parameters but a single loss scalar.  

```python
from tessera import autodiff, op

# Forward mode example
y, dy_dx = autodiff.forward(lambda x: op.sin(x) * op.exp(x), x=1.0)

# Reverse mode example
f = lambda x: op.sum(op.square(x))
df_dx = autodiff.reverse(f, inputs=[op.tensor([1.0,2.0,3.0])])
```

---

### 5.2 Gradient of Composite Operators
Tessera allows differentiating **graphs** directly, not just scalar functions.  

```python
from tessera import graph

@graph.module
def mlp(x, W1, W2):
    return op.relu(op.matmul(x, W1)) @ W2

x = op.tensor((64, 1024))
W1 = op.tensor((1024, 2048))
W2 = op.tensor((2048, 10))

loss = op.cross_entropy(mlp(x, W1, W2), labels=op.tensor((64,)))

grads = graph.backward(loss)  # reverse-mode autodiff
```

---

### 5.3 Optimizers
Tessera ships with standard optimizers (SGD, Adam, Adafactor) and supports **operator fusion** for efficiency.

```python
from tessera import optim

opt = optim.Adam(params=[W1, W2], lr=1e-4)

for step in range(1000):
    loss = training_step(batch)
    grads = graph.backward(loss)
    opt.apply(grads)
```

---

### 5.4 Custom Gradient Rules
Users can register **custom adjoints** for new operators.  

```python
@autodiff.custom_grad
def my_relu(x):
    return op.max(x, 0.0)

@my_relu.grad
def my_relu_grad(x, upstream):
    return upstream * (x > 0)
```

---

### 5.5 Higher-Order Derivatives
Tessera supports higher-order gradients by applying `autodiff.reverse` or `autodiff.forward` recursively.

```python
f = lambda x: op.sum(op.sin(x)**2)
d2f_dx2 = autodiff.reverse(lambda x: autodiff.reverse(f, x), x=op.tensor([1.0]))
```

---

### 5.6 Optimization Patterns
- **Gradient Checkpointing**: Reduce memory by recomputing intermediates.  
- **Fused Optimizers**: Single-kernel application of Adam updates.  
- **Mixed Precision AD**: Keep grads in FP32 while activations in BF16.  
