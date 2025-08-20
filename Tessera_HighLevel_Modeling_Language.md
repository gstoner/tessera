# Tessera High‑Level Modeling Language (HML) Specification
*(CUDA‑style companion; normative unless otherwise noted)*

---

## 1. Scope
This document defines the **High‑Level Modeling Language (HML)** for Tessera: syntax, semantics, functions, sub‑functions, operators, and data types. HML is the user‑facing layer that compiles to Graph IR → Schedule IR → Tile IR → Target IR.

---

## 2. Design Goals
- **Algebraic**: Programs are compositions of operators over tensors.
- **Deterministic**: Execution is bitwise reproducible (stable reductions, fixed collective order).
- **Static yet ergonomic**: Shapes, dtypes, and layouts are inferred where possible, but explicit when necessary.
- **Distributed‑first**: Tensors carry layouts (`ShardSpec`) independent of device IDs.
- **Extensible**: User‑defined functions and custom operators with adjoints.

---

## 3. Lexical & Grammar (Informative)

### 3.1 Tokens
Identifiers: `[A-Za-z_][A-Za-z0-9_]*`  
Literals: integers, floats, booleans, strings.  
Specials: `@graph.module`, `@op.custom`, `with policy(...)`.

### 3.2 Mini‑EBNF
```
program   := {decl}
decl      := module_decl | func_decl | const_decl | layout_decl
module_decl := "@graph.module" "(" name ")" func_body
func_decl := "def" name "(" params? ")" "->"? type? ":" suite
params    := param {"," param}
param     := name (":" type)? ("=" default)?
type      := tensor_type | scalar_type | tuple_type
tensor_type := "tensor" "[" shape "," dtype ("," layout)? "]"
shape     := "(" dim {"," dim} ")"
dim       := int | "?" | name
dtype     := "fp8_e4m3" | "fp8_e5m2" | "fp16" | "bf16" | "fp32" | "fp64"
           | "int8" | "int16" | "int32" | "int64" | "bool"
tuple_type:= "(" type {"," type} ")"
layout    := "layout=" shard_spec
shard_spec:= "ShardSpec" "(" "partition=" axis_list "," "mesh_axes=" axis_list
             ["," "block=" block_shape] ["," "replicate=" bool] ")"
axis_list := "(" name {"," name} ")"
suite     := NEWLINE INDENT {stmt} DEDENT
stmt      := assign | call | return | control | with | comment
call      := name "(" [args] ")"
with      := "with" "policy" "(" policy_name ")" ":" suite
```

---

## 4. Core Types (Normative)

### 4.1 Scalar Types
- Integer: `int8`, `int16`, `int32`, `int64`
- Unsigned: `uint8`, `uint16`, `uint32`, `uint64`
- Float: `fp8_e4m3`, `fp8_e5m2`, `fp16`, `bf16`, `fp32`, `fp64`
- Complex: `complex64` (2×fp32), `complex128` (2×fp64)
- Boolean: `bool`

### 4.2 Tensor Types
`tensor[(d0, d1, …, dn), dtype, layout?]`  
- Dimensions MAY be symbolic (`?`, `N`, `D`).  
- Layout is optional; inferred when omitted.

### 4.3 Composite Types
- Tuples: `(T1, T2, …)`  
- Records (informative): via dict‑like literals for named outputs.  
- Enums (informative): for finite options (`policy=strict|promote|downcast`).

---

## 5. Numeric & Casting Policies (Normative)

- `policy=strict`: no implicit casts.  
- `policy=promote` (default): safe upcasts (`fp16 + fp32 → fp32`).  
- `policy=downcast`: explicit only, with warnings.  
- Reductions MUST accumulate in at least `fp32` unless `dtype ∈ {fp64, complex128}`.

---

## 6. Functions & Sub‑Functions

### 6.1 Module Functions
HML programs group user‑defined functions into **modules** (graph‑level entry points).

```python
@graph.module("encoder_block")
def encoder_block(X: tensor[(B,T,D), fp16],
                  Wq, Wk, Wv, Wo, W1, W2, b1, b2):
    Q = op.matmul(X, Wq)
    K = op.matmul(X, Wk)
    V = op.matmul(X, Wv)
    O = op.flash_attention(Q, K, V)
    H = op.relu(op.matmul(O, W1) + b1)
    Y = op.matmul(H, W2) + b2
    return Y
```

**Sub‑functions** are regular `def` and can be inlined/fused by the compiler.

### 6.2 Control Sub‑Functions
- `graph.training_step(module=...)`: captures forward+loss+backward.  
- `graph.forward(fn, inputs, tangents?)`: forward‑mode JVP.  
- `graph.backward(loss, wrt=[...])`: reverse‑mode gradients.  
- `graph.checkpoint(fn)`: enable recompute‑vs‑store trade‑off.  

---

## 7. Operator Catalog (Normative)
Each operator specifies: **signature**, **broadcast**, **adjoint**, **stability**, **fusion**.

### 7.1 Shape & Indexing
- `op.reshape(X, shape)`  
- `op.transpose(X, perm)`  
- `op.slice(X, starts, sizes)`  
- `op.concat(xs, axis)`  
- `op.pad(X, pads, value=0)`

### 7.2 Elementwise Math
- Unary: `exp, log, sin, cos, tanh, relu, gelu, sigmoid`
- Binary: `add, sub, mul, div, pow, minimum, maximum`
- Comparison: `lt, le, gt, ge, eq, ne`
- Logical: `and, or, xor, not`

**Adjoints**: elementwise chain rule; division guards.  
**Fusion**: elementwise ops eagerly fuse with producers/consumers.

### 7.3 Reductions
- `reduce_sum(X, axis=?, keepdims=False, stable=True)`  
- `reduce_mean, reduce_max, reduce_min, reduce_prod`  

**Stability**: pairwise/Kahan accumulation; deterministic order.  
**Adjoints**: broadcasted scatter of upstream gradient.

### 7.4 Linear Algebra
- `matmul(A, B, tile_shape=?, accumulate="fp32")`
- `einsum(spec, *Xs)`
- Factorized: `factorized_matmul(A, B, rank)`
- Decomp: `cholesky, qr, svd`

**Adjoints**: explicit rules; decomp backprop follows standard math.  
**Fusion**: `matmul + bias + activation` is fusible.

### 7.5 Spectral & Transforms
- `fft, ifft, dct, idct, wavelet, iwavelet`
- `conv_fft(X, K) := ifft(fft(X) ⊙ fft(K))` (rewrite)

**Adjoints**: `adj(fft) = ifft`, etc.  
**Fusion**: `fft ∘ pointwise ∘ ifft` fuses into spectral kernels.

### 7.6 Neural Network Primitives
- Normalization: `layernorm, rmsnorm, batchnorm`
- Attention: `scaled_dot_product_attention`, `flash_attention`
- MLP: `mlp(in_dim, hidden=[...], out_dim, activation="gelu")`
- Convolution: `conv1d/2d/3d`, `depthwise_conv`, `separable_conv`

**Adjoints**: provided; softmax uses log‑sum‑exp stabilization.  
**Fusion**: attention+softmax+matmul fused; norm+matmul may fuse.

### 7.7 MoE & Routing
- `mixture_of_experts(experts, router="top2", capacity=?, drop_policy=?)`
- `router(policy="top2|topk|switch", temperature=?, jitter=?)`

**Adjoints**: straight‑through or differentiable routing per policy.  
**Distribution**: experts map to `tp` axis; tokens route via A2A.

### 7.8 PDE & Differential Operators
- `grad(X, axis)`, `div(U)`, `curl(U)`, `laplacian(X)`
- `helmholtz(X, k)`, `poisson(rhs)` (solver wrapper)

**Adjoints**: exact; mixed derivatives commute under stability policy.

### 7.9 Randomness
- `random_normal(shape, seed)`, `random_uniform(shape, seed)`  
**Determinism**: stateless RNG → same seed+indices = same result.

### 7.10 Collectives (Distributed)
- `all_reduce(X, op="sum", axis="dp")`
- `reduce_scatter(X, op="sum", axis="dp")`
- `all_gather(X, axis="tp")`, `broadcast(X, root=0)`

**Determinism**: fixed tree order; quantization only within policy.

---

## 8. Layout & Distribution (Normative)

### 8.1 Mesh & ShardSpec
```python
mesh = dist.Mesh(axes=["tp","pp","dp"], devices=range(72))

W = dist.tensor(shape=(1_000_000, 1_000_000),
                layout=dist.ShardSpec(partition=("row","col"),
                                      mesh_axes=("tp","pp"),
                                      block=(128,128)),
                mesh=mesh, dtype="bf16")
```

- `partition`: tuple of tensor dims to shard.
- `mesh_axes`: mapping to mesh dimensions (`tp`, `pp`, `dp`, `ep`).
- `block`: optional tile block size hint.
- `replicate`: replicate across given axes (costly).

### 8.2 Semantics
- Every distributed tensor MUST declare a layout (explicit or inferred).
- Replication allowed but compiler warns on imbalance/overuse.
- Collectives are inserted at boundaries as needed.

---

## 9. Autodiff Interfaces (Normative)

- Reverse‑mode: `graph.backward(loss, wrt=[params])`  
- Forward‑mode: `graph.forward(fn, inputs, tangents)`  
- HvP: `graph.hvp(loss_fn, wrt=params, dir=v)`  
- JVP: `graph.jvp(fn, wrt=x, dir=dx)`  
- Checkpointing: `graph.checkpoint(fn)`

All gradients are deterministic; stable reductions enforced.

---

## 10. Error Model (Normative)
- **Static shape mismatch**: compile‑time error.
- **Dtype violation under `policy=strict`**: compile‑time error.
- **Undeclared layout on distributed tensor**: compile‑time error.
- **Nondeterministic op attempt**: compile‑time error unless `fastmath` explicitly enabled (non‑normative).

---

## 11. Diagnostics & Profiling (Informative)
- `graph.profile(fn)` → per‑operator/tile timing, BW, occupancy.
- `graph.explain(fn)` → fusion decisions, rewrites, collective plans.
- Autotuning surface: per‑op shape/arch cache keys; export/import DB.

---

## 12. Examples

### 12.1 FlashAttention Block
```python
@graph.module("attn")
def attn(Q, K, V):
    return op.flash_attention(Q, K, V)   # fused kernel via rewrite
```

### 12.2 PINN Residual (Navier–Stokes 2D)
```python
@graph.module("ns2d_residual")
def ns2d_residual(psi, Re):
    u = op.grad(psi, axis="y")
    v = -op.grad(psi, axis="x")
    omega = op.laplacian(psi)
    adv = u*op.grad(omega,"x") + v*op.grad(omega,"y")
    diff = (1/Re)*op.laplacian(omega)
    return adv - diff
```

### 12.3 SFT + RL Step
```python
@graph.training_step(module="sft_step")
def sft_step(batch):
    logits = model(batch["input"])
    loss = op.cross_entropy(logits, batch["labels"])
    grads = graph.backward(loss, wrt=model.parameters())
    return grads, {"loss": loss}
```

---

## 13. MLIR View (Informative)

### 13.1 Graph Dialect (`tgraph`)
```mlir
%S = tgraph.matmul %Q, %KT : (...) -> tensor<?x?x?xf32>
%P = tgraph.softmax %S {stable=true} : tensor<?x?x?xf32>
%O = tgraph.matmul %P, %V : (...) -> tensor<?x?x?xf16>
```

### 13.2 Schedule Dialect (`tsched`)
```mlir
%f = tsched.fuse @attn(%Q,%K,%V) { pattern="flash_attention" }
tsched.tile %f candidates = [[128,128,64],[64,128,64]]
tsched.collective %f { pattern="RS_MM_AG", axis="tp" }
```

### 13.3 Tile Dialect (`ttile`)
```mlir
%acc = ttile.mma.sync %fragA, %fragB, %acc : f32
```

---

## 14. Appendix: Operator Signatures (Subset)

| Operator | Signature | Broadcast | Adjoint | Stable |
|---|---|---|---|---|
| `matmul` | `(A[m,k], B[k,n]) → C[m,n]` | No | dA = dC·Bᵀ; dB = Aᵀ·dC | Acc=f32 |
| `softmax` | `(X[...,n]) → Y[...,n]` | Last axis | dX = (dY - sum(dY∘Y))∘Y | log‑sum‑exp |
| `fft/ifft` | `X → X` | No | adj(fft)=ifft | N/A |
| `reduce_sum` | `X → scalar/kept` | Axes | scatter | pairwise/Kahan |
| `grad` | `X → ∇X` | No | symmetric | N/A |
| `flash_attention` | `(Q,K,V) → O` | Heads | fused | stable softmax |

---

## 15. Versioning
- HML is versioned alongside Tessera runtime/ABI.
- Minor updates MAY add operators; removals require major version bump.

