# Tessera Programming Model Guide

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Tiles and Execution Model](#tiles-and-execution-model)
3. [Memory Management](#memory-management)
4. [Numerics and Types](#numerics-and-types)
5. [Autodiff System](#autodiff-system)
6. [Distributed Computing](#distributed-computing)
7. [Advanced Patterns](#advanced-patterns)

## Core Concepts

### The Tile Abstraction

Tessera's fundamental innovation is replacing thread-centric programming with **tile-centric** programming:

```python
# Traditional CUDA thinking
thread_id = blockIdx.x * blockDim.x + threadIdx.x

# Tessera thinking
tile_id = ts.tile.linear_id()
```

### Hierarchical Parallelism

```
Mesh (Multi-device)
  ↓
Device (Single GPU)
  ↓
Tile Group (Cooperating tiles, maps to CUDA block)
  ↓
Tile (Work unit, maps to warp or thread group)
  ↓
Lane (Individual thread)
```

## Tiles and Execution Model

### Basic Tile Operations

```python
@ts.kernel
def tile_example(data: ts.Tensor["N", ts.f32]):
    # Get tile identification
    tile_id = ts.tile.linear_id()
    group_id = ts.tile.group_id()
    lane_id = ts.tile.lane_id()
    
    # Tile-level operations
    if tile_id < N:
        # Each tile processes one element
        data[tile_id] *= 2.0
```

### Tile Ranges and Iteration

```python
@ts.kernel
def process_chunks(input: ts.Tensor["N", ts.f32],
                  output: ts.Tensor["N", ts.f32]):
    # Process data in chunks
    CHUNK_SIZE = 128
    
    for chunk_start in ts.tile.range(0, N, CHUNK_SIZE):
        # Each tile group processes a chunk
        chunk_end = min(chunk_start + CHUNK_SIZE, N)
        
        # Cooperative processing within tile group
        for i in ts.tile.range(chunk_start, chunk_end):
            output[i] = process(input[i])
```

### Tile Groups and Synchronization

```python
@ts.kernel
def cooperative_reduction(data: ts.Tensor["N", ts.f32]) -> ts.f32:
    # Allocate shared memory for tile group
    shared_sum = ts.tile.alloc_shared((128,), ts.f32)
    
    # Each tile computes partial sum
    tile_id = ts.tile.linear_id()
    local_sum = 0.0
    
    for i in ts.tile.range(tile_id, N, ts.tile.group_size()):
        local_sum += data[i]
    
    # Store to shared memory
    shared_sum[ts.tile.local_id()] = local_sum
    
    # Synchronize tile group
    ts.tile.barrier()
    
    # Cooperative reduction
    stride = ts.tile.group_size() // 2
    while stride > 0:
        if ts.tile.local_id() < stride:
            shared_sum[ts.tile.local_id()] += shared_sum[ts.tile.local_id() + stride]
        ts.tile.barrier()
        stride //= 2
    
    return shared_sum[0]
```

## Memory Management

### Memory Hierarchy Operations

```python
@ts.kernel
def memory_hierarchy_example(A: ts.Tensor["M", "N", ts.f32]):
    # 1. Register allocation (tile-local)
    reg_value = ts.tile.alloc_register(ts.f32)
    
    # 2. Shared memory allocation (group-local)
    shared_tile = ts.tile.alloc_shared((64, 64), ts.f32, swizzle="xor")
    
    # 3. Global memory access
    global_value = A[i, j]
    
    # 4. Async memory operations
    ts.tile.cp_async(shared_tile, A[block_m:block_m+64, block_n:block_n+64])
    ts.tile.cp_commit_group()
    ts.tile.cp_wait_group(0)
```

### Memory Access Patterns

```python
@ts.kernel
def optimized_memory_access(src: ts.Tensor["M", "N", ts.bf16],
                           dst: ts.Tensor["M", "N", ts.bf16]):
    # Coalesced access pattern
    TILE_M, TILE_N = 128, 128
    
    # Allocate shared memory with bank conflict avoidance
    smem = ts.tile.alloc_shared((TILE_M, TILE_N), ts.bf16, 
                                swizzle="xor",      # XOR swizzling
                                alignment=16)        # 128-bit alignment
    
    # Vectorized loads for maximum bandwidth
    for i in ts.tile.range(0, TILE_M, 4):
        # Load 4 elements at once
        vec4 = ts.tile.load_vector(src[block_m + i], width=4)
        ts.tile.store_vector(smem[i], vec4)
```

### Double Buffering Pattern

```python
@ts.kernel
def double_buffered_gemm(A: ts.Tensor["M", "K", ts.bf16],
                        B: ts.Tensor["K", "N", ts.bf16],
                        C: ts.Tensor["M", "N", ts.f32]):
    # Allocate double buffers
    smem_a = [
        ts.tile.alloc_shared((BM, BK), ts.bf16),
        ts.tile.alloc_shared((BM, BK), ts.bf16)
    ]
    smem_b = [
        ts.tile.alloc_shared((BK, BN), ts.bf16),
        ts.tile.alloc_shared((BK, BN), ts.bf16)
    ]
    
    # Pipeline stages
    write_stage = 0
    compute_stage = 1
    
    # Prefetch first blocks
    ts.tile.cp_async(smem_a[write_stage], A[0:BM, 0:BK])
    ts.tile.cp_async(smem_b[write_stage], B[0:BK, 0:BN])
    
    for k in ts.tile.range(BK, K, BK):
        # Start loading next blocks
        write_stage, compute_stage = compute_stage, write_stage
        ts.tile.cp_async(smem_a[write_stage], A[0:BM, k:k+BK])
        ts.tile.cp_async(smem_b[write_stage], B[k:k+BK, 0:BN])
        
        # Wait for previous load and compute
        ts.tile.cp_wait_group(0)
        ts.tile.barrier()
        
        # Compute on current blocks
        C += ts.tile.mma(smem_a[compute_stage], smem_b[compute_stage])
```

## Numerics and Types

### Type System with Policies

```python
# Basic types with accumulation policies
x1: ts.Tensor["B", "D", ts.fp8_e4m3 @accum(ts.f32)]
x2: ts.Tensor["B", "D", ts.bf16 @accum(ts.f32) @stochastic_round]
x3: ts.Tensor["B", "D", ts.fp6 @accum(ts.f32) @loss_scale(128.0)]

# Mixed precision computation
@ts.jit
def mixed_precision_gemm(
    A: ts.Tensor["M", "K", ts.fp8_e4m3 @accum(ts.f32)],
    B: ts.Tensor["K", "N", ts.fp8_e4m3 @accum(ts.f32)]
) -> ts.Tensor["M", "N", ts.f32]:
    """FP8 inputs, FP32 accumulation and output"""
    return ts.gemm(A, B)  # Automatic precision handling
```

### Safe Numerical Operations

```python
@ts.jit
def numerically_stable_softmax(x: ts.Tensor["B", "S", ts.bf16]) -> ts.Tensor["B", "S", ts.bf16]:
    """Numerically stable softmax with explicit precision control"""
    # Convert to FP32 for stability
    x_f32 = ts.cast(x, ts.f32)
    
    # Safe operations
    x_max = ts.max(x_f32, axis=-1, keepdims=True)
    exp_x = ts.exp(x_f32 - x_max)  # Prevent overflow
    sum_exp = ts.sum(exp_x, axis=-1, keepdims=True)
    
    # Safe division with epsilon
    softmax = exp_x / (sum_exp + 1e-10)
    
    # Cast back to BF16
    return ts.cast(softmax, ts.bf16)
```

### Custom Numeric Policies

```python
# Define custom precision policy
@ts.precision_policy
class QuantizedPolicy:
    storage = ts.int8
    scale = ts.f32
    compute = ts.f32
    accumulate = ts.f32
    
    def quantize(self, x: ts.f32) -> ts.int8:
        scale = ts.max(ts.abs(x)) / 127.0
        return ts.round(x / scale).astype(ts.int8), scale
    
    def dequantize(self, x: ts.int8, scale: ts.f32) -> ts.f32:
        return x.astype(ts.f32) * scale

# Use custom policy
quantized_weight: ts.Tensor["D", "D", QuantizedPolicy]
```

## Autodiff System

### Basic Autodiff

```python
@ts.jit
@ts.autodiff
def loss_function(x, W, b, y_true):
    """Automatic differentiation example"""
    # Forward pass
    y_pred = ts.matmul(x, W) + b
    y_pred = ts.relu(y_pred)
    loss = ts.mse(y_pred, y_true)
    return loss

# Compute gradients
grad_fn = ts.grad(loss_function)
grad_W, grad_b = grad_fn(x, W, b, y_true)
```

### Custom VJP/JVP Rules

```python
@ts.custom_vjp
def custom_gelu(x):
    """GELU with custom backward pass"""
    return 0.5 * x * (1 + ts.tanh(0.79788456 * (x + 0.044715 * x**3)))

def custom_gelu_fwd(x):
    y = custom_gelu(x)
    return y, (x, y)  # Save for backward

def custom_gelu_bwd(saved, grad_output):
    x, y = saved
    # Custom gradient computation
    cdf = 0.5 * (1 + ts.tanh(0.79788456 * (x + 0.044715 * x**3)))
    pdf = ts.exp(-0.5 * x**2) / ts.sqrt(2 * ts.pi)
    grad_input = grad_output * (cdf + x * pdf)
    return grad_input

custom_gelu.defvjp(custom_gelu_fwd, custom_gelu_bwd)
```

### Effect-Aware Autodiff

```python
@ts.jit
@ts.autodiff
def distributed_loss(x, W):
    """Autodiff with distributed operations"""
    # Forward pass with collective operations
    y = ts.matmul(x, W)
    y = ts.all_gather(y, axis="tp")  # Tensor parallel gather
    loss = ts.sum(y**2)
    
    # Backward automatically inserts reduce_scatter
    return loss

# Gradients handle distributed communication
grad_W = ts.grad(distributed_loss)(x, W)
# Automatically includes reduce_scatter in backward
```

## Distributed Computing

### Mesh Construction

```python
# Create various mesh configurations
# Data parallel mesh
dp_mesh = ts.dist.mesh(
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    axes=("dp",),
    shape=(4,)
)

# Tensor parallel mesh
tp_mesh = ts.dist.mesh(
    devices=["cuda:0", "cuda:1"],
    axes=("tp",),
    shape=(2,)
)

# Hybrid mesh (DP + TP + PP)
hybrid_mesh = ts.dist.mesh(
    devices=[f"cuda:{i}" for i in range(16)],
    axes=("dp", "tp", "pp"),
    shape=(4, 2, 2)  # 4-way DP, 2-way TP, 2-way PP
)
```

### Tensor Sharding

```python
# Different sharding strategies
# Column sharding for tensor parallelism
W_col = ts.dist.tensor(
    shape=(8192, 8192),
    layout=ts.ShardSpec(
        partition=("col",),    # Partition along columns
        mesh_axes=("tp",)      # Map to tensor parallel axis
    ),
    mesh=tp_mesh,
    dtype="bf16"
)

# Row sharding for data parallelism
X_row = ts.dist.tensor(
    shape=(batch_size, 8192),
    layout=ts.ShardSpec(
        partition=("row",),    # Partition along rows
        mesh_axes=("dp",)      # Map to data parallel axis
    ),
    mesh=dp_mesh,
    dtype="bf16"
)

# 2D sharding
W_2d = ts.dist.tensor(
    shape=(8192, 8192),
    layout=ts.ShardSpec(
        partition=("row", "col"),  # 2D partitioning
        mesh_