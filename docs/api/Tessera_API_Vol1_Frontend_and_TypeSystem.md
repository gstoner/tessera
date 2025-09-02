# Tessera API Reference - Volume 1
## Frontend API and Type System

### 1. Introduction
Tessera is a next-generation deep learning programming model built on MLIR infrastructure【24†source】【26†source】.

Core principles:
- Shape polymorphism
- Memory efficiency
- Numerical stability
- Multi-level IR pipeline

---

### 2. Frontend API (Python + Rust)

#### Python Layer
```python
import tessera as ts

@ts.function
def flash_attention(q: ts.Tensor["B", "H", "S", "D"],
                    k: ts.Tensor["B", "H", "S", "D"],
                    v: ts.Tensor["B", "H", "S", "D"]) -> ts.Tensor["B", "H", "S", "D"]:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = ts.matmul(q, k.transpose(-2, -1)) * scale
    probs = ts.softmax(scores, dim=-1)
    return ts.matmul(probs, v)
```

#### Rust Core
Handles parsing, type system, and MLIR lowering【25†source】.

```rust
pub struct TensorType {
    pub element_type: ElementType,
    pub shape: Shape,
    pub constraints: Vec<Constraint>,
}
```

---

### 3. Type System & Numerical Policies

#### Shape-Aware Types
```python
def attention[B, S, D, H](
    query: Tensor[B, H, S, D],
    key: Tensor[B, H, S, D],
    value: Tensor[B, H, S, D]
) -> Tensor[B, H, S, D]:
    scores = query @ key.transpose(-2, -1)
    weights = softmax(scores / sqrt(D))
    return weights @ value
```

#### Broadcasting Rules
- NumPy semantics checked at compile time【26†source】

#### Numerical Policies
| Attribute   | Description                  | Example    |
|-------------|------------------------------|------------|
| storage     | Storage format               | `fp8_e4m3` |
| compute     | Arithmetic precision         | `bf16`     |
| accumulate  | Reduction accumulation type  | `f32`      |
| rounding    | Rounding strategy            | stochastic |

#### Example
```python
x: Tensor[B, D, Policy(storage=fp8_e4m3, compute=bf16, accumulate=f32)]
```

---

### 4. Effects & Determinism

#### Effects
```python
@tessera.effects(["random", "io"])
def dropout(x: Tensor, p: float = 0.5) -> Tensor:
    mask = tessera.random.bernoulli(p, x.shape)
    return x * mask
```

#### Deterministic Mode
```python
@tessera.deterministic(seed=42)
def reproducible_training(dataset):
    for batch in dataset:
        loss = model(batch)
    return loss
```
