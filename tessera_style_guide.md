# Tessera Programming Model Style Guide

## Version 1.0 | Last Updated: September 2025

This style guide establishes consistent conventions for writing Tessera code, documentation, and examples across the entire programming model ecosystem.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Import Conventions](#import-conventions)
3. [Naming Conventions](#naming-conventions)
4. [Type Annotations](#type-annotations)
5. [Numerical Precision](#numerical-precision)
6. [Kernel Development](#kernel-development)
7. [Distributed Programming](#distributed-programming)
8. [Documentation Standards](#documentation-standards)
9. [Code Examples](#code-examples)
10. [Common Patterns](#common-patterns)

---

## Core Principles

### 1. Explicitness Over Implicitness
- Always declare tensor shapes, dtypes, and precision policies explicitly
- Make memory spaces and data movement explicit
- Prefer explicit barriers and synchronization over implicit behavior

### 2. Consistency Across Scale
- Code should read similarly whether targeting single GPU or NVL72
- Distribution patterns should be declarative and clear
- Kernel patterns should scale from single tile to mesh-wide execution

### 3. Safety First
- Use region privileges (`read`, `write`, `reduce_sum`) consistently
- Employ safe numerical primitives (`softmax_safe`, `layernorm_safe`)
- Include error handling in production examples

---

## Import Conventions

### Standard Import Pattern
```python
import tessera as ts
from tessera import dist, tile, autodiff
from tessera.ops import gemm, flash_attention, layernorm_safe
```

### Discouraged Patterns
```python
# Avoid: Wildcard imports
from tessera import *

# Avoid: Inconsistent aliasing
import tessera
from tessera import dist as d

# Avoid: Mixed import styles in same file
import tessera as ts
from tessera import dist
```

### Specialized Imports
```python
# For kernel development
from tessera.kernel import tile, tshared, tbarrier, cp_async

# For distributed programming
from tessera.dist import mesh, ShardSpec, tensor

# For autodiff
from tessera.autodiff import grad, vjp, jvp, custom_vjp
```

---

## Naming Conventions

### Variables and Functions
```python
# Use snake_case for variables and functions
def flash_attention_v3(query, key, value):
    batch_size = query.shape[0]
    seq_len = query.shape[1]
    head_dim = query.shape[2]
    
# Use descriptive names for tensors
weight_matrix = ts.randn((8192, 8192), dtype=ts.bf16)
activation_tensor = ts.zeros((batch_size, seq_len, hidden_dim))

# Use conventional abbreviations
Q, K, V = split_qkv(qkv_projection)  # OK: Standard attention notation
W_q, W_k, W_v = query_weights, key_weights, value_weights  # OK: Clear
```

### Constants and Configuration
```python
# Use UPPER_CASE for constants
BLOCK_SIZE = 128
MAX_SEQUENCE_LENGTH = 8192
DEFAULT_PRECISION = ts.bf16

# Use descriptive names for hyperparameters
learning_rate = 1e-4
dropout_probability = 0.1
attention_heads = 16
```

### Mesh and Distribution Names
```python
# Use descriptive axis names
mesh = dist.mesh(
    devices=range(72),
    axes=("data_parallel", "tensor_parallel", "pipeline_parallel"),
    shape=(4, 9, 2)
)

# Short forms acceptable for common patterns
mesh = dist.mesh(
    devices=range(72),
    axes=("dp", "tp", "pp"),
    shape=(4, 9, 2)
)
```

---

## Type Annotations

### Tensor Type Annotations
```python
# Standard format: Tensor[shape, dtype @policies]
def transformer_layer(
    x: ts.Tensor["batch", "seq_len", "hidden_dim", ts.bf16 @ts.accum(ts.f32)],
    weights: ts.Tensor["hidden_dim", "hidden_dim", ts.bf16 @ts.accum(ts.f32)]
) -> ts.Tensor["batch", "seq_len", "hidden_dim", ts.bf16]:
    pass

# For symbolic dimensions, use descriptive names
def attention_block(
    q: ts.Tensor["B", "H", "S", "D_h", ts.bf16],
    k: ts.Tensor["B", "H", "S", "D_h", ts.bf16],
    v: ts.Tensor["B", "H", "S", "D_h", ts.bf16]
) -> ts.Tensor["B", "H", "S", "D_h", ts.bf16]:
    pass
```

### Region Privileges
```python
# Always specify region privileges for distributed functions
@ts.jit
def gradient_step(
    params: ts.Region[ts.read],
    gradients: ts.Region[ts.read], 
    optimizer_state: ts.Region[ts.write],
    loss_accumulator: ts.Region[ts.reduce_sum]
) -> None:
    pass
```

### Kernel Type Annotations
```python
# Tile kernel annotations
@ts.kernel
def matrix_multiply_tile(
    A: ts.Tile["M", "K", ts.f16],
    B: ts.Tile["K", "N", ts.f16],
    C: ts.Tile["M", "N", ts.f32]  # Accumulation type
) -> None:
    pass
```

---

## Numerical Precision

### Precision Policy Format
```python
# Standard format: dtype @policy1 @policy2
input_tensor: ts.Tensor["B", "S", "D", ts.fp8_e4m3 @ts.accum(ts.f32) @ts.stochastic_round]

# For mixed precision training
weights = ts.tensor(
    shape=(8192, 8192),
    dtype=ts.bf16 @ts.accum(ts.f32) @ts.loss_scale(2.0),
    device="cuda:0"
)
```

### Supported Precision Types
```python
# Storage types (in order of preference for documentation)
ts.f32        # Full precision
ts.bf16       # Brain float 16
ts.f16        # Half precision
ts.fp8_e4m3   # FP8 E4M3 (training)
ts.fp8_e5m2   # FP8 E5M2 (inference)
ts.fp6        # FP6 (Blackwell)
ts.fp4        # FP4 (ultra-low precision)

# Accumulation policies
@ts.accum(ts.f32)     # Preferred for stability
@ts.accum(ts.bf16)    # For memory-constrained scenarios
@ts.accum(ts.f16)     # Rare, only when necessary
```

### Safe Numerical Operations
```python
# Always use safe variants for numerically sensitive ops
y = ts.ops.softmax_safe(x)           # Not: ts.ops.softmax(x)
normalized = ts.ops.layernorm_safe(x)  # Not: ts.ops.layernorm(x)
log_sum = ts.ops.logsumexp_safe(x)   # Not: ts.ops.logsumexp(x)
```

---

## Kernel Development

### Kernel Structure
```python
@ts.kernel
@ts.autotune(
    configs=[
        {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 3},
        {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    ],
    key=["M", "N", "K"]  # Autotuning dimensions
)
def optimized_gemm(
    A: ts.Tile["M", "K", ts.f16],
    B: ts.Tile["K", "N", ts.f16], 
    C: ts.Tile["M", "N", ts.f32]
) -> None:
    """
    Optimized GEMM kernel with autotuning.
    
    Args:
        A: Left matrix tile
        B: Right matrix tile  
        C: Output accumulation tile
    """
    ctx = tile.context()
    
    # Shared memory allocation with clear naming
    smem_A = tshared.alloc[ts.f16](ctx.BLOCK_M, ctx.BLOCK_K, swizzle="xor")
    smem_B = tshared.alloc[ts.f16](ctx.BLOCK_K, ctx.BLOCK_N, swizzle="xor")
    
    # Async pipeline with explicit stages
    for k_tile in tile.range(0, ctx.K, ctx.BLOCK_K, stages=ctx.num_stages):
        cp_async.shared.global(smem_A, A[:, k_tile:k_tile+ctx.BLOCK_K])
        cp_async.shared.global(smem_B, B[k_tile:k_tile+ctx.BLOCK_K, :])
        tbarrier()
        
        # Matrix multiplication with explicit accumulation
        C += tile.mma(smem_A, smem_B, accumulator=ts.f32)
```

### Memory Management Patterns
```python
# Explicit memory space allocation
registers_acc = tile.zeros((BLOCK_M, BLOCK_N), dtype=ts.f32)
shared_buffer = tshared.alloc[ts.f16](BLOCK_SIZE, BLOCK_SIZE)

# Clear data movement patterns
tile.async_copy_global_to_shared(source=global_tensor, dest=shared_buffer)
tbarrier()  # Explicit synchronization
tile.async_copy_shared_to_global(source=shared_buffer, dest=output_tensor)
```

---

## Distributed Programming

### Mesh Creation
```python
# Standard mesh creation pattern
mesh = dist.mesh(
    devices=[f"cuda:{i}" for i in range(72)],  # Explicit device list
    axes=("dp", "tp", "pp"),                   # Clear axis names
    shape=(4, 9, 2)                           # Factorization: 4×9×2=72
)

# For smaller configurations
mesh_8gpu = dist.mesh(
    devices=range(8),
    axes=("dp", "tp"),
    shape=(2, 4)
)
```

### Tensor Sharding
```python
# Standard sharding specification
weight_matrix = dist.tensor(
    shape=(8192, 8192),
    layout=dist.ShardSpec(
        partition=("col",),           # Which dimensions to partition
        mesh_axes=("tp",),           # Which mesh axes to use
        replication=None             # Explicit replication specification
    ),
    mesh=mesh,
    dtype=ts.bf16 @ts.accum(ts.f32)
)

# Multiple partitioning
activation_tensor = dist.tensor(
    shape=(batch_size, seq_len, hidden_dim),
    layout=dist.ShardSpec(
        partition=("row", None, "col"),  # Batch and hidden_dim partitioned
        mesh_axes=("dp", None, "tp"),    # Map to dp and tp axes
    ),
    mesh=mesh,
    dtype=ts.bf16
)
```

### Index Launches
```python
# Standard index launch pattern
@ts.index_launch(axis="tp")
def distributed_attention(
    Q_shards: ts.TensorParts["tp"],
    K_shards: ts.TensorParts["tp"], 
    V_shards: ts.TensorParts["tp"],
    output_shards: ts.TensorParts["tp"]
) -> None:
    """Launch attention computation across tensor-parallel shards."""
    attention_kernel(Q_shards, K_shards, V_shards, output_shards)
```

---

## Documentation Standards

### Function Docstrings
```python
def flash_attention_v3(
    query: ts.Tensor["B", "H", "S", "D", ts.bf16],
    key: ts.Tensor["B", "H", "S", "D", ts.bf16],
    value: ts.Tensor["B", "H", "S", "D", ts.bf16],
    causal: bool = True,
    dropout_p: float = 0.0
) -> ts.Tensor["B", "H", "S", "D", ts.bf16]:
    """
    Flash Attention v3 implementation with Hopper optimizations.
    
    Implements memory-efficient attention with O(N) memory complexity
    using tiling and online softmax computation. Includes Hopper-specific
    optimizations like TMA async copies and warp specialization.
    
    Args:
        query: Query tensor [batch, heads, seq_len, head_dim]
        key: Key tensor [batch, heads, seq_len, head_dim] 
        value: Value tensor [batch, heads, seq_len, head_dim]
        causal: Apply causal masking for autoregressive models
        dropout_p: Dropout probability (0.0 = no dropout)
        
    Returns:
        Attention output tensor with same shape as query
        
    Example:
        >>> mesh = dist.mesh(devices=range(8), axes=("dp", "tp"), shape=(2, 4))
        >>> Q = dist.tensor((32, 16, 2048, 128), mesh=mesh, dtype=ts.bf16)
        >>> K = dist.tensor((32, 16, 2048, 128), mesh=mesh, dtype=ts.bf16) 
        >>> V = dist.tensor((32, 16, 2048, 128), mesh=mesh, dtype=ts.bf16)
        >>> output = flash_attention_v3(Q, K, V, causal=True)
        
    Note:
        Requires CUDA compute capability 8.0+ for optimal performance.
        On Hopper (9.0+), uses TMA and wgmma instructions automatically.
    """
    pass
```

### Code Comments
```python
def transformer_block(x, weights):
    # Stage 1: Pre-normalization (safe variant prevents overflow)
    normalized = layernorm_safe(x)
    
    # Stage 2: Multi-head attention with causal masking
    # Note: Uses memory-efficient Flash Attention v3 implementation
    attention_out = flash_attention_v3(
        query=linear(normalized, weights.q_proj),
        key=linear(normalized, weights.k_proj),
        value=linear(normalized, weights.v_proj),
        causal=True  # Autoregressive generation
    )
    
    # Stage 3: Residual connection with dropout
    x = x + dropout(attention_out, p=0.1)
    
    return x
```

---

## Code Examples

### Complete Example Structure
```python
#!/usr/bin/env python3
"""
Example: Distributed Flash Attention on NVL72

This example demonstrates Flash Attention v3 implementation
across a 72-GPU NVL72 system with tensor parallelism.
"""

import tessera as ts
from tessera import dist, autodiff
from tessera.ops import flash_attention_v3, layernorm_safe

def main():
    # Setup: 72-GPU mesh with 4×9×2 topology
    mesh = dist.mesh(
        devices=[f"cuda:{i}" for i in range(72)],
        axes=("dp", "tp", "pp"), 
        shape=(4, 9, 2)
    )
    
    # Configuration
    batch_size = 32
    seq_len = 8192
    hidden_dim = 4096
    num_heads = 32
    head_dim = hidden_dim // num_heads
    
    # Distributed tensor creation
    input_tensor = dist.tensor(
        shape=(batch_size, seq_len, hidden_dim),
        layout=dist.ShardSpec(
            partition=("row", None, "col"),
            mesh_axes=("dp", None, "tp")
        ),
        mesh=mesh,
        dtype=ts.bf16 @ts.accum(ts.f32)
    )
    
    # Model computation
    @ts.jit
    @autodiff.grad
    def attention_forward(x):
        # Safe pre-normalization
        normalized = layernorm_safe(x)
        
        # Reshape for multi-head attention
        q = normalized.reshape(batch_size, seq_len, num_heads, head_dim)
        k = normalized.reshape(batch_size, seq_len, num_heads, head_dim)
        v = normalized.reshape(batch_size, seq_len, num_heads, head_dim)
        
        # Flash attention computation
        attention_out = flash_attention_v3(
            query=q, key=k, value=v,
            causal=True,
            dropout_p=0.1
        )
        
        # Reshape back and apply output projection
        output = attention_out.reshape(batch_size, seq_len, hidden_dim)
        return output
    
    # Execute computation
    result = attention_forward(input_tensor)
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")

if __name__ == "__main__":
    main()
```

---

## Common Patterns

### Error Handling
```python
@ts.jit
def safe_computation(x):
    try:
        # Use safe numerical primitives
        normalized = layernorm_safe(x)
        
        # Validate tensor properties
        assert normalized.dtype == ts.bf16, f"Expected bf16, got {normalized.dtype}"
        assert not ts.any(ts.isnan(normalized)), "NaN detected in normalized tensor"
        
        return normalized
    except ts.NumericalError as e:
        ts.logger.warning(f"Numerical instability detected: {e}")
        # Fallback to higher precision
        return layernorm_safe(x.to(ts.f32)).to(ts.bf16)
```

### Performance Monitoring
```python
@ts.benchmark(warmup=10, iterations=100)
def benchmark_attention(batch_size, seq_len):
    """Benchmark attention performance with standard metrics."""
    
    # Setup
    q = ts.randn((batch_size, 16, seq_len, 128), dtype=ts.bf16)
    k = ts.randn((batch_size, 16, seq_len, 128), dtype=ts.bf16)
    v = ts.randn((batch_size, 16, seq_len, 128), dtype=ts.bf16)
    
    # Computation with timing
    with ts.profile("attention_forward"):
        output = flash_attention_v3(q, k, v, causal=True)
    
    # Calculate metrics
    flops = 4 * batch_size * 16 * seq_len * seq_len * 128  # Attention FLOPS
    memory_gb = (q.nbytes + k.nbytes + v.nbytes + output.nbytes) / 1e9
    
    return {
        "flops": flops,
        "memory_gb": memory_gb,
        "output_shape": output.shape
    }
```

### Configuration Management
```python
@ts.dataclass
class AttentionConfig:
    """Configuration for attention computation."""
    
    # Model dimensions
    batch_size: int = 32
    seq_len: int = 2048
    num_heads: int = 16
    head_dim: int = 128
    
    # Numerical precision
    dtype: ts.DType = ts.bf16
    accumulation_dtype: ts.DType = ts.f32
    
    # Optimization settings
    causal: bool = True
    dropout_p: float = 0.0
    
    # Hardware configuration
    mesh_shape: tuple[int, ...] = (2, 4)  # (dp, tp)
    device_count: int = 8
    
    def create_mesh(self) -> dist.Mesh:
        """Create mesh from configuration."""
        return dist.mesh(
            devices=list(range(self.device_count)),
            axes=("dp", "tp"),
            shape=self.mesh_shape
        )
```

---

## Style Checking

### Automated Checks
Consider implementing these automated style checks:

1. **Import Pattern Validation**
   - Ensure `import tessera as ts` is used consistently
   - Check for prohibited wildcard imports

2. **Naming Convention Checks** 
   - Variable names use snake_case
   - Constants use UPPER_CASE
   - Type annotations follow Tensor[shape, dtype] format

3. **Numerical Precision Validation**
   - Accumulation types are explicitly specified
   - Safe numerical primitives are used appropriately

4. **Documentation Quality**
   - All public functions have docstrings
   - Examples are complete and runnable
   - Cross-references are valid

This style guide ensures consistency across the entire Tessera ecosystem while maintaining readability and performance.
