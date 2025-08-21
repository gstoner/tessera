# Tessera Tutorials Volume
## Chapter 6 — Case Studies

This chapter presents **end-to-end examples** of how Tessera can be used to implement real-world workloads.

---

### 6.1 Case Study: Transformer Pretraining
Demonstrates large-scale language model pretraining with **data + tensor parallelism**.

```python
from tessera import dist, op, graph

# Define mesh
mesh = dist.Mesh(axes=["tp","dp"], devices=[[0,1,2,3],[4,5,6,7]])

# Model
@graph.module
def transformer(x, Wq, Wk, Wv, Wo):
    attn = op.flash_attention(x @ Wq, x @ Wk, x @ Wv)
    return attn @ Wo

# Loss and training loop
@graph.training_step(module="TransformerLM")
def step(batch):
    out = transformer(batch["input"])
    loss = op.cross_entropy(out, batch["labels"])
    grads = graph.backward(loss)
    return grads, {"loss": loss}
```

---

### 6.2 Case Study: Physics-Informed Neural Network (PINN)
Uses **operator adjoints** to enforce PDE constraints (Navier–Stokes 2D).  

```python
@graph.module
def ns_pinn(x, y, t):
    psi = op.mlp([x,y,t], hidden=[256,256], out_dim=1)
    u = op.grad(psi, x)
    v = op.grad(psi, y)
    p = op.grad(psi, t)
    return u, v, p

@graph.loss
def ns_loss(u, v, p):
    continuity = op.div(u, "x") + op.div(v, "y")
    momentum = op.grad(u, "t") + u*op.grad(u,"x") + v*op.grad(u,"y") + op.grad(p,"x")
    return op.mse(continuity,0) + op.mse(momentum,0)
```

---

### 6.3 Case Study: Mixture of Experts (MoE)
Leverages **expert parallelism** and Tessera’s routing operators.

```python
experts = [op.mlp(4096, [8192], 4096) for _ in range(16)]
router = op.router(experts)

@graph.module
def moe_layer(x): return router(x)
```

---

### 6.4 Case Study: Spectral Learning
Uses **operator-based transforms** (FFT, wavelets) to build compact models.

```python
Xf = op.fft(batch["input"])
Yf = op.recursive_mixture([op.mlp(4096, [4096], 4096) for _ in range(4)], depth=3)(Xf)
Y = op.ifft(Yf)
```

---

### 6.5 Takeaways
- Tessera unifies **autodiff + distributed execution + operator algebra**.  
- Case studies demonstrate its **versatility**: LLMs, PDE surrogates, MoE, and spectral learning.  
- Each workload benefits from **IR lowering and optimized collectives** transparently.  
