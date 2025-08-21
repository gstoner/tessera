# Tessera Autodiff Programming Guide

*(CUDA-style companion; normative unless stated otherwise)*

---

## 1. Scope

This document describes the **automatic differentiation (autodiff)** capabilities of Tessera’s programming model. Tessera provides **reverse-mode (backpropagation)** and **forward-mode (JVP)** differentiation as **first-class compiler transformations** on the Graph IR, with deterministic and numerically stable semantics.

---

## 2. Design Principles

- **First-Class Operators**: Derivatives are defined at the operator level.  
- **Determinism**: Gradients must be reproducible across executions.  
- **Numerical Stability**: Stable adjoints for reductions and nonlinear ops.  
- **Zero Overhead**: Differentiation expressed at IR level, not runtime tape.  
- **Hybrid AD**: Forward- and reverse-mode can interleave for optimal efficiency.

---

## 3. Reverse-Mode Differentiation

Reverse-mode is the **default** for training deep learning models.

### 3.1 Semantics
- Gradients are constructed by traversing the **Graph IR DAG** backwards.
- Each operator provides a **symbolic adjoint** implementation.
- Intermediate tensors required for backprop may be recomputed or checkpointed.

### 3.2 Stability Policies
- Reductions must use **pairwise or Kahan accumulation** to avoid gradient drift.  
- Log-sum-exp trick is used for softmax, log, and exp adjoints.  
- Division by zero adjoints produce NaN unless guarded.

### 3.3 MLIR Example
```mlir
tgraph.module @m {
  %x = tgraph.arg : tensor<?xf32>
  %y = tgraph.sin %x : tensor<?xf32>
  %z = tgraph.reduce %y {op = "sum"} : tensor<?xf32> -> f32
  tgraph.return %z : f32
}

// Autodiff pass (reverse mode)
tgraph.adjoint @m wrt(%x) -> @m_grad
```

### 3.4 Generated Adjoint (Sketch)
```mlir
tgraph.module @m_grad {
  %x = tgraph.arg : tensor<?xf32>
  %dy = tgraph.ones_like %x
  %cosx = tgraph.cos %x
  %dx = tgraph.mul %dy, %cosx
  tgraph.return %dx
}
```

---

## 4. Forward-Mode Differentiation

Forward-mode is designed for **Jacobian-vector products (JVPs)**, PDE linearization, and cases where input dimension is small relative to output.

### 4.1 Semantics
- Attach a **tangent value** to each primal tensor.
- Propagate tangents forward using per-operator **differentials**.
- More efficient than reverse-mode when `dim(inputs) << dim(outputs)`.

### 4.2 MLIR Example
```mlir
tgraph.module @m {
  %x = tgraph.arg : tensor<?xf32>
  %y = tgraph.exp %x : tensor<?xf32>
  tgraph.return %y : tensor<?xf32>
}

// Forward-mode transform
tgraph.jvp @m wrt(%x) -> @m_jvp
```

### 4.3 Generated JVP (Sketch)
```mlir
tgraph.module @m_jvp {
  %x, %dx = tgraph.arg, tgraph.tangent : tensor<?xf32>
  %y = tgraph.exp %x
  %dy = tgraph.mul %y, %dx
  tgraph.return %y, %dy
}
```

---

## 5. Mixed-Mode Differentiation

Tessera supports **hybrid strategies**:
- Reverse-over-forward for **Hessian-vector products**.  
- Forward-over-reverse for **higher-order gradients**.  

Example use case: PINNs (Physics-Informed Neural Nets) where forward-mode JVPs compute PDE residuals, while reverse-mode computes parameter gradients.

---

## 6. Determinism & Reproducibility

- All gradients must be **bitwise reproducible**.  
- No floating-point nondeterminism (e.g., parallel reduction order).  
- RNG ops must propagate **stateless seeds** into adjoints.  

---

## 7. Worked Example: Attention Gradient

```python
@graph.module("attention")
def attention(Q, K, V):
    S = op.matmul(Q, K.T)
    P = op.softmax(S)
    O = op.matmul(P, V)
    return O

# Reverse-mode gradient
grads = graph.backward(loss, wrt=[Q, K, V])

# Forward-mode JVP for linearization
outputs, jvp = graph.forward(attention, (Q, K, V), tangents=(dQ, dK, dV))
```

---

## 8. Validation

- Tessera’s compiler checks **adjoint correctness** by comparing symbolic vs. numeric gradients (finite differences).  
- Any operator without an adjoint or differential raises a **compile-time error**.  

---

## 9. Higher-Order Derivatives

Tessera supports higher-order derivatives through **mode composition**.

### 9.1 Hessian-Vector Products (HvP)
Computed via **reverse-over-forward**:
1. Apply forward-mode to propagate tangent directions.  
2. Apply reverse-mode to compute gradients of the JVP.  

### 9.2 Jacobian-Matrix Products (JMP)
Computed via **forward-over-reverse**:
1. Apply reverse-mode to compute vector-Jacobian product (VJP).  
2. Apply forward-mode on the VJP graph.  

### 9.3 MLIR Example (HvP)
```mlir
// Hessian-vector product: ∇²f(x) * v
tgraph.hvp @m wrt(%x) dir(%v) -> @m_hvp
```

---

## 10. Implicit Differentiation

Tessera supports differentiation through optimization problems and fixed-point solvers.

### 10.1 Fixed-Point Equations
If `x*` is defined by `F(x*) = 0`, then implicit differentiation uses:
```
dx*/dθ = -(∂F/∂x)⁻¹ (∂F/∂θ)
```

### 10.2 Example: Differentiable Optimization Layer
```python
@graph.module("opt_layer")
def opt_layer(params, input):
    # x* = argmin f(x; params, input)
    x_star = op.solve(params, input)
    return x_star

grads = graph.backward(loss, wrt=params)
```

The compiler automatically inserts Jacobian solves using adjoint operators.

---

## 11. Worked Example: PINN Hessian

In Physics-Informed Neural Nets (PINNs), second derivatives of the network output w.r.t. spatial coordinates are required.

```python
@graph.module("pinn")
def pinn(x, y, params):
    u = net(x, y, params)
    # PDE residual for Navier-Stokes
    du_dx = graph.jvp(u, wrt=x)
    du_dy = graph.jvp(u, wrt=y)
    d2u_dx2 = graph.hvp(u, wrt=x, dir=dx)
    d2u_dy2 = graph.hvp(u, wrt=y, dir=dy)
    residual = du_dx + du_dy - nu * (d2u_dx2 + d2u_dy2)
    return residual
```

---

## 12. Comparison: Tessera vs. JAX vs. PyTorch

| Feature | Tessera | JAX | PyTorch |
|---------|---------|-----|---------|
| **Reverse-mode (VJP)** | First-class IR transform, symbolic adjoints | `grad` / `vjp` | `autograd.backward` |
| **Forward-mode (JVP)** | Native, first-class | `jax.jvp` | Limited (`functorch.jvp`) |
| **Mixed-mode** | Composable reverse-over-forward and forward-over-reverse | Available but verbose | Limited, experimental (`functorch`) |
| **Higher-order** | Built-in with mode composition | Supported | Supported but slower, less stable |
| **Implicit Diff** | Compiler inserts adjoint solvers | `jaxopt`, custom | Manual, fragile |
| **Numerical Stability** | Enforced policies (pairwise reduction, log-sum-exp) | User responsibility | User responsibility |
| **Determinism** | Bitwise reproducible across devices | Not guaranteed | Not guaranteed |
| **MLIR Integration** | Native lowering with autodiff dialect | No (XLA-based) | No (ATen custom ops) |

---


---

## 13. Side‑by‑Side Code Examples (Tessera vs. JAX vs. PyTorch)

This section shows equivalent autodiff tasks across the three ecosystems.

### 13.1 Reverse‑Mode Gradient of a 2‑Layer MLP

**Tessera**
```python
from tessera import op, graph

W1 = op.tensor((D, H), dtype="fp32")
b1 = op.tensor((H,), dtype="fp32")
W2 = op.tensor((H, C), dtype="fp32")
b2 = op.tensor((C,), dtype="fp32")

@graph.module("mlp2")
def mlp2(x):
    h = op.relu(op.matmul(x, W1) + b1)
    y = op.matmul(h, W2) + b2
    return y

@graph.training_step(module="mlp2_train")
def step(x, y_true):
    y_pred = mlp2(x)
    loss = op.cross_entropy(y_pred, y_true)
    grads = graph.backward(loss, wrt=[W1, b1, W2, b2])
    return grads, {"loss": loss}
```

**JAX**
```python
import jax, jax.numpy as jnp

def mlp2(params, x):
    W1, b1, W2, b2 = params
    h = jax.nn.relu(x @ W1 + b1)
    y = h @ W2 + b2
    return y

def loss_fn(params, x, y_true):
    logits = mlp2(params, x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y_true))

grads = jax.grad(loss_fn)(params, x, y_true)
```

**PyTorch**
```python
import torch, torch.nn.functional as F

class MLP2(torch.nn.Module):
    def __init__(self, D, H, C):
        super().__init__()
        self.W1 = torch.nn.Linear(D, H)
        self.W2 = torch.nn.Linear(H, C)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))

model = MLP2(D, H, C)
loss = F.cross_entropy(model(x), y_true)
loss.backward()   # gradients in model.parameters()
```

---

### 13.2 Forward‑Mode JVP (Jacobian‑Vector Product)

**Tessera**
```python
from tessera import graph

# Compute y, dy given x and tangent dx
y, dy = graph.forward(mlp2, (x,), tangents=(dx,))
```

**JAX**
```python
y, dy = jax.jvp(lambda z: mlp2(params, z), (x,), (dx,))
```

**PyTorch (functorch)**
```python
from functorch import jvp
y, dy = jvp(model, (x,), (dx,))
```

---

### 13.3 Hessian‑Vector Product (HvP) of a Scalar Loss

**Tessera**
```python
# v is the perturbation vector
hvp = graph.hvp(loss_fn, wrt=params, dir=v, inputs=(x, y_true))
```

**JAX**
```python
hvp = jax.jvp(jax.grad(loss_fn), (params,), (v,))[1]
```

**PyTorch (functorch)**
```python
from functorch import vjp, jvp

loss = lambda p: loss_fn(p, x, y_true)
_, vjp_fn = vjp(torch.func.grad(loss), params)
hvp = vjp_fn(v)[0]
```

---

### 13.4 Deterministic Reductions (Sum)

**Tessera**
```python
s = op.reduce_sum(X, stable=True)   # pairwise/Kahan, fixed order
```

**JAX**
```python
s = jnp.sum(X)   # order may vary across devices/backends
```

**PyTorch**
```python
s = torch.sum(X) # nondeterminism depends on backend/kernel
```

---

### 13.5 Implicit Differentiation Through a Fixed‑Point Solver

**Tessera**
```python
@graph.module("fp_solve")
def fp_solve(theta):
    x_star = op.fixed_point(lambda x: F(x, theta), x0, tol=1e-6, max_iter=100)
    return x_star

# Compiler inserts (∂F/∂x)^{-1} (∂F/∂θ) in the adjoint
grads = graph.backward(loss_fn(fp_solve(theta)), wrt=[theta])
```

**JAX**
```python
# Requires custom implicit function implementation or jaxopt
from jaxopt import fixed_point
x_star = fixed_point.fixed_point(lambda x: F(x, theta), x0)[0]
grads = jax.grad(lambda th: loss_fn(f(x_star, th)))(theta)
```

**PyTorch**
```python
# Typically manual w.r.t. implicit function theorem or higher-level libs
x_star = fixed_point_solver(lambda x: F(x, theta), x0)
loss = loss_fn(x_star, theta)
loss.backward()
```

---
