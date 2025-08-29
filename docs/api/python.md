# Tessera Python API Reference

## Core Module (`tessera.core`)

### Tensor Class

The `Tensor` class supports shape polymorphism and automatic differentiation.

```python
import tessera as tsr

# Create tensors with dynamic shapes
x = tsr.randn([4, "S", 512])  # Batch=4, dynamic sequence, dim=512
y = tsr.zeros([4, "S", 512])

# Basic operations
z = x + y
w = tsr.matmul(x, y.transpose(-1, -2))
```

### Module Class

Base class for all Tessera models.

```python
class MyModel(tsr.Module):
    def __init__(self):
        super().__init__()
        self.linear = tsr.nn.Linear(512, 256)
    
    def forward(self, x):
        return self.linear(x)
```

## Neural Network Module (`tessera.nn`)

### Flash Attention

Memory-efficient attention implementation.

```python
attention = tsr.nn.FlashAttention(
    dim=512,
    heads=8,
    causal=True
)
output = attention(q, k, v)
```

This API will be expanded as implementation progresses.
