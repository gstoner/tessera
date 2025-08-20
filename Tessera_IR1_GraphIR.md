# Tessera IR Layer 1 — Graph IR (Operator Algebra & Autodiff)
*(CUDA-style programming guide companion; normative unless stated otherwise)*

---

## 1. Scope
Graph IR is the **front-end functional IR** that represents computation as an algebraic graph of **operators** over **tensors**. It is the source for scheduling, fusion, autodiff, and determinism policies.

---

## 2. Design Goals
- **Algebraic**: Operators compose; rewrite rules preserve semantics.
- **Deterministic**: Same program, inputs, and seeds MUST produce bitwise-identical outputs.
- **Differentiable**: Reverse- and forward-mode AD are first-class.
- **Distribution-ready**: Graph nodes carry layout hints, not device specifics.

---

## 3. Core Concepts
### 3.1 Operators
An operator is a pure function on tensors with:
- **Signature**: ranks, dtypes, broadcasting rules.
- **Adjoint**: mathematically specified backward.
- **Rewrites**: fusible forms (e.g., softmax∘matmul).

### 3.2 Tensors
Opaque values with **shape**, **dtype**, and optional **layout_hint** (e.g., `{"prefer": "col-shard:tp"}`). **No aliasing** is permitted at the Graph IR level.

### 3.3 Graph Modules
A **module** is a closed graph (entry + exports). Modules MAY import subgraphs (libraries).

---

## 4. Autodiff
### 4.1 Reverse-Mode (Default)
- Construct tape-free adjoints by composing known operator adjoints.
- All reductions MUST use stable accumulation (pairwise/Kahan).

### 4.2 Forward-Mode (Optional)
- Used for Jacobian-vector products (JVP) and PDE linearization.

---

## 5. Determinism Policies (Normative)
- Reductions MUST have fixed order.
- Randomness MUST be stateless (seed + indices).
- No data races, in-place update is disallowed at Graph IR.

---

## 6. Examples

### 6.1 Attention in Graph IR (Pythonic Pseudo)
```python
@graph.module("attention")
def attention(Q, K, V):
    S = op.matmul(Q, op.transpose(K, (-1, -2)))
    P = op.softmax(S)        # stable version implied
    O = op.matmul(P, V)
    return O
```

### 6.2 MLIR Sketch (custom `tgraph` dialect)
```mlir
tgraph.module @m {
  %Q = tgraph.arg : tensor<?x?x?xf16>
  %K = tgraph.arg : tensor<?x?x?xf16>
  %V = tgraph.arg : tensor<?x?x?xf16>

  %KT = tgraph.transpose %K {perm = [0,1,3,2]} : tensor<?x?x?xf16> -> tensor<?x?x?xf16>
  %S  = tgraph.matmul %Q, %KT : (tensor<?x?x?xf16>, tensor<?x?x?xf16>) -> tensor<?x?x?xf32>
  %P  = tgraph.softmax %S {stable = true} : tensor<?x?x?xf32> -> tensor<?x?x?xf32>
  %O  = tgraph.matmul %P, %V : (tensor<?x?x?xf32>, tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  tgraph.return %O : tensor<?x?x?xf16>
}
```

### 6.3 PINN Residual
```mlir
%lap = tgraph.laplacian %phi : tensor<?x?xf32> -> tensor<?x?xf32>
%res = tgraph.sub %lap, %f : tensor<?x?xf32>
%L   = tgraph.reduce %res {op = "sum", stable = true} : tensor<?x?xf32> -> f32
```

---

## 7. Rewrites (Informative)
- `tgraph.softmax(tgraph.matmul(Q, KT))` → `tgraph.flash_attention(Q, K, V)` if compatible.
- `tgraph.fft -> pointwise_mul -> ifft` recognized as spectral convolution.

---

## 8. Validation
- Shape/dtype inference MUST succeed before lowering.
- Adjoint check: numeric vs symbolic grads within tolerance.
