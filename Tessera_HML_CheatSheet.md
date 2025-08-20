# Tessera HML — Quick Reference (Cheat Sheet)
*(Keep alongside the Programming Guide and HML Spec)*

---

## 1) Core Types & Literals
- **Scalars:** `fp8_e4m3`, `fp8_e5m2`, `fp16`, `bf16`, `fp32`, `fp64`, `int{8,16,32,64}`, `uint{8,16,32,64}`, `bool`, `complex{64,128}`
- **Tensor:** `tensor[(d0, d1, ...), dtype, layout?]` (dims may be symbolic: `B, T, D, ?`)
- **Literals:** `op.tensor((2,3), dtype="fp32")` creates an uninitialized tensor of shape `(2,3)`

**Casting Policy**
```python
with policy("promote"):      # default
    Y = X_fp16 + X_fp32      # → fp32
with policy("strict"):
    Y = X_fp16 + X_fp32      # ERROR (requires explicit cast)
```

---

## 2) Modules, Functions, Sub‑functions
```python
@graph.module("block")
def block(X, W1, W2, b1, b2):
    H = op.relu(op.matmul(X, W1) + b1)
    return op.matmul(H, W2) + b2
```

**Training Step**
```python
@graph.training_step(module="train_step")
def step(batch):
    logits = model(batch["input"])
    loss = op.cross_entropy(logits, batch["labels"])
    grads = graph.backward(loss, wrt=model.parameters())
    return grads, {"loss": loss}
```

---

## 3) Mesh & Sharding (Distribution)
```python
from tessera import dist

mesh = dist.Mesh(axes=["tp","pp","dp"], devices=range(72))

W = dist.tensor(shape=(1_000_000, 1_000_000),
    layout=dist.ShardSpec(partition=("row","col"), mesh_axes=("tp","pp"), block=(128,128)),
    mesh=mesh, dtype="bf16")
```
- `tp` = tensor parallel; `pp` = pipeline parallel; `dp` = data parallel
- Every distributed tensor SHOULD declare a `ShardSpec`

**Collectives**
```python
g = op.all_reduce(g_local, op="sum", axis="dp")  # deterministic order
```

---

## 4) Shape & Indexing Ops
```python
Y = op.reshape(X, (B, T*D))
Z = op.transpose(Y, (1, 0))
S = op.slice(Z, starts=(0,0), sizes=(B//2, D))
C = op.concat([A, B], axis=-1)
P = op.pad(X, pads=((0,0),(1,1)), value=0)
```

---

## 5) Elementwise & Reductions
```python
Y = op.relu(X);  Z = op.gelu(X)
S = op.reduce_sum(X, axis=-1, stable=True)    # pairwise/Kahan, fixed order
M = op.reduce_max(X, axis=0)
```

---

## 6) Linear Algebra & Spectral
```python
C = op.matmul(A, B, tile_shape=(128,128,64), accumulate="fp32")
Y = op.einsum("bthd,bThd->btTh", Q, K)
Xf = op.fft(X);  Y  = op.ifft(Xf * Kf)        # spectral conv via rewrite
```

---

## 7) NN Primitives
```python
A = op.layernorm(X)
O = op.flash_attention(Q, K, V)      # softmax ∘ matmul ∘ matmul (fused)
Y = op.mlp(in_dim=D, hidden=[8192], out_dim=D, activation="gelu")
```

**MoE**
```python
experts = [op.mlp(D, [8192], D) for _ in range(16)]
Y = op.mixture_of_experts(X, experts=experts, router=op.router("top2"))
```

---

## 8) PDE / PINN Operators
```python
u_x = op.grad(u, axis="x");  u_y = op.grad(u, axis="y")
lap = op.laplacian(u)
res = u_x + u_y - nu * lap
loss = op.reduce_sum(res**2, stable=True)
```

---

## 9) Autodiff API
```python
grads = graph.backward(loss, wrt=[W1, b1, W2, b2])   # reverse‑mode
y, jvp = graph.forward(f, (x,), tangents=(dx,))      # forward‑mode JVP
hvp = graph.hvp(loss_fn, wrt=params, dir=v)          # Hessian‑vector product
```

---

## 10) Numerical Policies
- Reductions: accumulate in **fp32** (min) unless dtype is fp64/complex128
- `softmax`: log‑sum‑exp stabilization by default
- RNG: stateless, mesh‑consistent (`seed + indices`)

---

## 11) Scheduling & Autotuning (Hints)
```python
from tessera import autotune

sig = autotune.signature(op="flash_attention", shape=(B,H,L,D), dtype="fp8_e4m3",
                         mesh={"tp":12,"dp":1}, arch="GB200", policy="stable")
best = autotune.search(autotune.plan(
    candidates=dict(tiles=[(128,128,64),(64,128,64)], fuse_bias=[True,False]),
    objectives=["latency","bytes_moved"],
    constraints={"HBM_util":"<0.85", "deterministic": True},
    signature=sig
), mode="hybrid")
```
- Cache is persistent per (op, shape, dtype, mesh, arch)
- Autotuner never changes numeric semantics (reduction order)

---

## 12) Profiling & Diagnostics
```python
from tessera import profiler
with profiler.session("step"): step(batch)
# Outputs: per‑op tile times, BW, collective latency, fusion trace

graph.explain(model)  # why/where it fused, chosen tiles, collective plans
```

---

## 13) Common Errors (and Fixes)
- **Shape mismatch** → ensure consistent dims or use `reshape/transpose`
- **Dtype under `policy=strict`** → cast explicitly: `X.astype("fp32")`
- **Missing layout on distributed tensor** → add `ShardSpec` or replicate
- **Memory over‑subscription** → shrink batch / change shard / enable KV spill
- **Nondeterministic path** → disable `fastmath` / ensure deterministic collectives

---

## 14) One‑Page Attention Example
```python
@graph.module("attn_block")
def attn_block(X, Wq, Wk, Wv, Wo):
    Q = op.matmul(X, Wq);  K = op.matmul(X, Wk);  V = op.matmul(X, Wv)
    O = op.flash_attention(Q, K, V)
    return op.matmul(O, Wo)
```

**Distributed Setup**
```python
mesh = dist.Mesh(axes=["tp","dp"], devices=range(16))
Q = dist.tensor((B,H,L,D), layout=dist.ShardSpec(("head",),("tp",)), mesh=mesh, dtype="bf16")
K = dist.tensor_like(Q); V = dist.tensor_like(Q)
O = attn_block(Q, Wq, Wk, Wv, Wo)
```

---

## 15) Quick Operator Table (subset)

| Op | Signature | Notes |
|----|-----------|-------|
| `matmul` | `(A[m,k], B[k,n]) → C[m,n]` | `tile_shape`, `accumulate="fp32"` |
| `flash_attention` | `(Q,K,V) → O` | fused, stable softmax |
| `reduce_sum` | `X → scalar/kept` | deterministic pairwise/Kahan |
| `fft/ifft` | `X → X` | spectral pipelines |
| `grad` | `u → ∇u` | PDEs / PINNs |
| `mixture_of_experts` | `X → Y` | router=`top2|topk|switch` |

---
