# Tessera Tutorials Volume
## Chapter 1 — Getting Started

### 1.1 Installation & Environment Setup
Tessera provides both a **Python API** for rapid prototyping and a **C++ API** for production.  
Minimum setup:
```bash
pip install tessera
```

Optional GPU-enabled build (with CUDA backend):
```bash
pip install tessera[gpu]
```

Verify installation:
```bash
python -c "import tessera; print(tessera.__version__)"
```

---

### 1.2 Hello World: Vector Addition
```python
import tessera as ts

# Allocate tensors on GPU
a = ts.tensor([1, 2, 3, 4], device="cuda")
b = ts.tensor([10, 20, 30, 40], device="cuda")

# Elementwise add
c = a + b
print(c)  # [11, 22, 33, 44]
```

---

### 1.3 First Custom Operator
Using Tessera’s operator API to define a kernel:
```python
from tessera import op

@op.kernel
def scale_add(x, y, alpha: float):
    return alpha * x + y

a = ts.tensor([1, 2, 3, 4], device="cuda")
b = ts.tensor([10, 20, 30, 40], device="cuda")

out = scale_add(a, b, alpha=0.5)
print(out)  # [10.5, 21, 31.5, 42]
```

---

### 1.4 Inspecting IR
Developers can inspect Tessera IR lowering:
```python
print(scale_add.inspect("graph"))     # High-level Graph IR
print(scale_add.inspect("schedule"))  # Fusion/tiling schedule
print(scale_add.inspect("tile"))      # GPU block/warp mapping
```

---

### 1.5 Running with Runtime Backend
Switching between CPU and GPU:
```python
x = ts.tensor([1, 2, 3, 4], device="cpu")
y = ts.tensor([10, 20, 30, 40], device="cpu")
print(scale_add(x, y, alpha=2.0))

# Now run on GPU
x = ts.tensor([1, 2, 3, 4], device="cuda")
y = ts.tensor([10, 20, 30, 40], device="cuda")
print(scale_add(x, y, alpha=2.0))
```
