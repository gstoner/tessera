# Tessera: Overview and Quick Start Guide

## What is Tessera?

Tessera is a next-generation GPU programming framework that unifies modeling, training, kernels, and distributed execution in a single stack. It's designed to scale seamlessly from single-GPU research code to massive NVL72 rack-scale deployments (72 GPUs).

## Key Features

### 🎯 Core Principles
- **Tile-First Programming Model**: Think in tiles and groups, not threads
- **Unified Stack**: One language for modeling, kernels, and distribution
- **Numerics as Types**: FP4/FP6/FP8/BF16/FP16/FP32 with explicit policies
- **Built-in Autodiff**: Forward and reverse AD with effect awareness
- **Production Ready**: From prototype to deployment without rewrites

### 🚀 Performance Features
- **Multi-Level IR Stack**: Graph IR → Schedule IR → Tile IR → Target IR
- **Hardware Optimization**: Automatic mapping to tensor cores (WMMA/WGMMA)
- **Memory Hierarchy**: Explicit control over registers, shared memory, HBM, NVLink
- **Distributed by Design**: Native support for DP/TP/PP parallelism
- **NVL72 Support**: Scale to 72-GPU systems with NVSwitch

## Quick Start Examples

### Example 1: Simple Kernel

```python
import tessera as ts

@ts.kernel
def saxpy(x: ts.Tensor["n", ts.f32], 
          y: ts.Tensor["n", ts.f32], 
          alpha: ts.f32) -> ts.Tensor["n", ts.f32]:
    """Simple SAXPY: y = alpha * x + y"""
    i = ts.tile.linear_id()
    if i < n:
        y[i] = alpha * x[i] + y[i]
    return y
```

### Example 2: Matrix Multiplication with Tensor Cores

```python
@ts.kernel
def gemm(A: ts.Tensor["M", "K", ts.bf16],
         B: ts.Tensor["K", "N", ts.bf16],
         C: ts.Tensor["M", "N", ts.f32]) -> ts.Tensor["M", "N", ts.f32]:
    """Matrix multiplication using tensor cores"""
    # Allocate shared memory tiles
    smem_a = ts.tile.alloc_shared((64, 32), ts.bf16, swizzle="xor")
    smem_b = ts.tile.alloc_shared((32, 64), ts.bf16, swizzle="xor")
    
    # Load tiles asynchronously
    ts.tile.cp_async(smem_a, A[tile_m:tile_m+64, tile_k:tile_k+32])
    ts.tile.cp_async(smem_b, B[tile_k:tile_k+32, tile_n:tile_n+64])
    ts.tile.barrier()
    
    # Compute using tensor cores
    acc = ts.tile.mma(smem_a, smem_b, accumulator=ts.f32)
    
    # Store result
    C[tile_m:tile_m+64, tile_n:tile_n+64] += acc
    return C
```

### Example 3: Distributed Training on Multiple GPUs

```python
# Create mesh for distributed execution
mesh = ts.dist.mesh(
    devices=[f"cuda:{i}" for i in range(8)],
    axes=("dp", "tp"),  # Data parallel and tensor parallel
    shape=(4, 2)        # 4-way DP, 2-way TP
)

# Distributed tensor with sharding
W = ts.dist.tensor(
    shape=(8192, 8192),
    layout=ts.ShardSpec(partition=("col",), mesh_axes=("tp",)),
    mesh=mesh,
    dtype="bf16 @accum(f32)"  # BF16 storage, FP32 accumulation
)

@ts.jit
@ts.autodiff
def train_step(x, y, W):
    """Training step with automatic gradient computation"""
    pred = ts.gemm(x, W)  # Automatically handles TP sharding
    loss = ts.mse(pred, y)
    return loss

# Gradients automatically handle distributed communication
grad_W = ts.grad(train_step)(x, y, W)
```

### Example 4: Flash Attention Implementation

```python
@ts.kernel.autotune(
    space=dict(
        BLOCK_M=[64, 128],
        BLOCK_N=[64, 128],
        num_warps=[4, 8],
        num_stages=[2, 3]
    )
)
def flash_attention(Q: ts.Tensor["B*H", "S", "D", ts.bf16],
                   K: ts.Tensor["B*H", "S", "D", ts.bf16],
                   V: ts.Tensor["B*H", "S", "D", ts.bf16],
                   is_causal: bool = True) -> ts.Tensor["B*H", "S", "D", ts.bf16]:
    """Flash Attention with online softmax"""
    
    # Tile configuration from autotuner
    ctx = ts.tile.context()
    BM, BN = ctx.BLOCK_M, ctx.BLOCK_N
    
    # Allocate shared memory
    smem_q = ts.tile.alloc_shared((BM, ctx.D), ts.bf16)
    smem_k = ts.tile.alloc_shared((BN, ctx.D), ts.bf16)
    smem_v = ts.tile.alloc_shared((BN, ctx.D), ts.bf16)
    
    # Initialize accumulators
    acc = ts.tile.zeros((BM, ctx.D), ts.f32)
    m_i = ts.tile.full((BM,), -float('inf'), ts.f32)
    l_i = ts.tile.zeros((BM,), ts.f32)
    
    # Main loop with online softmax
    for q_block in ts.tile.range(0, ctx.S, BM):
        # Load Q block
        ts.tile.cp_async(smem_q, Q[q_block:q_block+BM])
        
        for kv_block in ts.tile.range(0, ctx.S, BN):
            # Load K,V blocks
            ts.tile.cp_async(smem_k, K[kv_block:kv_block+BN])
            ts.tile.cp_async(smem_v, V[kv_block:kv_block+BN])
            ts.tile.barrier()
            
            # Compute attention scores
            scores = ts.tile.mma(smem_q, smem_k.T) * (1.0 / math.sqrt(ctx.D))
            
            # Apply causal mask if needed
            if is_causal:
                scores = ts.tile.causal_mask(scores, q_block, kv_block)
            
            # Online softmax update
            m_new = ts.tile.row_max(scores)
            m_i_new = ts.tile.maximum(m_i, m_new)
            
            # Compute correction factors
            alpha = ts.tile.exp(m_i - m_i_new)
            beta = ts.tile.exp(m_new - m_i_new)
            
            # Update accumulator
            p = ts.tile.exp(scores - m_i_new)
            l_i = alpha * l_i + beta * ts.tile.row_sum(p)
            acc = alpha * acc + ts.tile.mma(p, smem_v)
            
            m_i = m_i_new
    
    # Finalize
    return ts.tile.cast(acc / l_i[:, None], ts.bf16)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/gstoner/tessera.git
cd tessera

# Install dependencies
pip install -r requirements.txt

# Install Tessera
pip install -e .

# Verify installation
python -c "import tessera; print(tessera.__version__)"
```

## System Requirements

### Minimum Requirements
- NVIDIA GPU (Compute Capability 7.0+)
- CUDA 11.8+
- Python 3.8+
- 16GB GPU memory (recommended)

### Optimal Performance
- NVIDIA H100/A100 GPUs
- CUDA 12.0+
- NVLink/NVSwitch for multi-GPU
- 80GB+ GPU memory for large models

## Architecture Support

| GPU Architecture | Compute Capability | Support Level | Key Features |
|-----------------|-------------------|---------------|--------------|
| Volta (V100) | sm_70 | Full | WMMA tensor cores |
| Turing (T4) | sm_75 | Full | Enhanced WMMA |
| Ampere (A100) | sm_80/86 | Full | Async copy, sparsity |
| Hopper (H100) | sm_90 | Optimal | WGMMA, TMA, clusters |
| Ada Lovelace | sm_89 | Full | Consumer tensor cores |

## Key Concepts

### Tile-First Programming
Instead of thinking about threads and blocks, Tessera uses tiles:
- **Tile**: Unit of parallel work
- **Tile Group**: Cooperating tiles sharing memory
- **Mesh**: Collection of devices for distributed execution

### Memory Hierarchy
```
Registers (fastest, tile-local)
    ↓
Shared Memory (group-local, synchronized)
    ↓
Global Memory / HBM (device-local)
    ↓
NVLink/NVSwitch (multi-device)
```

### Numerical Policies
```python
# Explicit precision control
x: Tensor["B", "D", fp8_e4m3 @accum(fp32) @loss_scale(128.0)]
```

### Distribution Strategies
- **Data Parallel (DP)**: Replicate model, shard data
- **Tensor Parallel (TP)**: Shard model tensors
- **Pipeline Parallel (PP)**: Split model layers
- **Expert Parallel (EP)**: For mixture-of-experts

## Performance Expectations

| Operation | H100 Performance | Notes |
|-----------|-----------------|--------|
| Dense GEMM | 1200+ TFLOPS | With tensor cores |
| Flash Attention | 1000+ TFLOPS | Fused implementation |
| FFT | 90+ TFLOPS | Optimized spectral ops |
| All-Reduce | 900 GB/s | Over NVSwitch |

## Next Steps

1. **Learn the Programming Model**: See the [Programming Model Guide](./tessera-programming-model.md)
2. **Explore Examples**: Check out the examples directory
3. **Understand the IR Stack**: Read about the compilation pipeline
4. **Try Distributed Training**: Set up multi-GPU experiments
5. **Optimize Performance**: Use autotuning and profiling tools

## Community and Support

- **GitHub**: https://github.com/gstoner/tessera
- **Documentation**: See accompanying guides
- **Issues**: Report bugs on GitHub
- **Contributing**: See CONTRIBUTING.md

## License

Tessera is open-source software. See LICENSE file for details.