# Tessera Tutorials Volume
## Chapter 2 — Tensors & Operators

### 2.1 Creating Tensors
Tensors are the core data abstraction in Tessera. They can live on **CPU** or **GPU** and may be **dense**, **distributed**, or **sparse**.

```python
import tessera as ts

# Dense tensor on GPU
A = ts.tensor([[1, 2], [3, 4]], device="cuda")

# Distributed tensor (2-way sharding along rows)
mesh = ts.dist.Mesh(axes=["dp"], devices=[0,1])
B = ts.dist.tensor((1024, 1024), layout=ts.dist.ShardSpec(("row",), ("dp",)), mesh=mesh)

# Sparse tensor
S = ts.tensor.sparse(indices=[[0,1],[2,3]], values=[10, 20], shape=(4,4), device="cuda")
```

---

### 2.2 Basic Operators
Operators are the **building blocks** of Tessera programs.  
Most standard mathematical and ML operators are available out of the box.

```python
from tessera import op

x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
y = ts.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")

z = op.matmul(x, y)
print(z)

z2 = op.fft(x)         # Spectral transform
z3 = op.relu(y)        # Nonlinearity
z4 = op.softmax(y)     # Normalization
```

---

### 2.3 Composing Operators
Operators can be chained, fused, or expressed as higher-order compositions.  

```python
@op.kernel
def attention(Q, K, V):
    scores = op.matmul(Q, K.T) / (Q.shape[-1] ** 0.5)
    weights = op.softmax(scores)
    return op.matmul(weights, V)
```

---

### 2.4 Inspecting IR for Operators
Just like in Chapter 1, you can inspect how composite operators lower through Tessera’s IRs:

```python
print(attention.inspect("graph"))     # High-level algebra
print(attention.inspect("schedule"))  # Fused loops, tiling
print(attention.inspect("tile"))      # GPU warp/block mapping
```

---

### 2.5 Visualizing IR Lowering
For teaching/debugging purposes, Tessera can emit diagrams:

```python
attention.export("attention_graph.dot", ir="graph")     # Graph IR as DOT
attention.export("attention_schedule.json", ir="schedule")  # Schedule IR as JSON
```

This allows visualization of how operators fuse and schedule down to hardware.
