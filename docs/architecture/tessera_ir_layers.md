# Tessera IR Layer Documentation

## Overview

Tessera's compilation pipeline transforms high-level Python code through multiple Intermediate Representation (IR) layers, each serving a specific purpose in the optimization and code generation process. This multi-layered approach enables aggressive optimizations while maintaining correctness and numerical stability.

```
User Code (Python) 
    ↓ [Front-end parsing, type checking]
Graph IR (Autodiff, Types)
    ↓ [Shape inference, automatic differentiation]  
Schedule IR (Fusion, Tiling)
    ↓ [Optimization passes, autotuning]
Tile IR (Blocks, Warps, DMA)
    ↓ [Hardware mapping, memory hierarchy]
Target IR (PTX, ROCm, CPU)
    ↓ [Platform-specific code generation]
Executable Binary
```

## Layer 1: User Code (Python)

**Purpose**: High-level mathematical specification with type annotations and numerical policies.

**Characteristics**:
- Pure Python syntax with Tessera decorators
- Shape-polymorphic tensor operations
- Automatic memory management
- Hardware-agnostic algorithm description

**Example**:
```python
@tessera.function
def flash_attention(
    q: Tensor["B", "H", "S", "D"],
    k: Tensor["B", "H", "S", "D"], 
    v: Tensor["B", "H", "S", "D"],
    scale: float = None
) -> Tensor["B", "H", "S", "D"]:
    """Flash Attention implementation"""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    
    # Online softmax computation
    output = tessera.zeros_like(q)
    max_val = tessera.full((B, H, S, 1), -float('inf'))
    sum_val = tessera.zeros((B, H, S, 1))
    
    for j in range(0, S, BLOCK_SIZE):
        # Load tiles
        q_tile = q[:, :, :, :]
        k_tile = k[:, :, j:j+BLOCK_SIZE, :]
        v_tile = v[:, :, j:j+BLOCK_SIZE, :]
        
        # Compute scores
        scores = q_tile @ k_tile.transpose(-2, -1) * scale
        
        # Update statistics
        new_max = tessera.maximum(max_val, scores.max(dim=-1, keepdim=True))
        scores_exp = tessera.exp(scores - new_max)
        
        # Apply mask if causal
        if tessera.is_causal():
            scores_exp = tessera.apply_causal_mask(scores_exp)
        
        # Update output
        output += scores_exp @ v_tile
    
    return output
```

**Transformations Applied**:
- Syntax validation and parsing
- Type annotation processing
- Decorator expansion
- Import resolution

## Layer 2: Graph IR

**Purpose**: Mathematical graph representation with automatic differentiation, shape inference, and high-level optimizations.

**Characteristics**:
- Computational graph with typed nodes
- Shape propagation and verification
- Automatic differentiation support
- High-level algebraic optimizations
- Device placement annotations

**Example Graph IR**:
```python
# Internal Graph IR representation
GraphIR {
    nodes: [
        Node(id=0, op="parameter", 
             name="q", shape=["B", "H", "S", "D"], 
             dtype=bf16, device="gpu"),
        Node(id=1, op="parameter",
             name="k", shape=["B", "H", "S", "D"],
             dtype=bf16, device="gpu"),
        Node(id=2, op="parameter", 
             name="v", shape=["B", "H", "S", "D"],
             dtype=bf16, device="gpu"),
        Node(id=3, op="constant",
             value=1/sqrt(D), dtype=f32),
        Node(id=4, op="matmul",
             inputs=[0, 1], transpose_b=True,
             shape=["B", "H", "S", "S"]),
        Node(id=5, op="multiply",
             inputs=[4, 3], broadcast=True),
        Node(id=6, op="softmax_online", 
             inputs=[5], axis=-1, stable=True,
             memory_efficient=True),
        Node(id=7, op="matmul",
             inputs=[6, 2],
             shape=["B", "H", "S", "D"])
    ],
    edges: [(0,4), (1,4), (4,5), (3,5), (5,6), (6,7), (2,7)],
    outputs: [7],
    
    # Gradient information
    gradients: {
        7: [∂loss/∂output],
        6: [∂loss/∂attention_weights], 
        4: [∂loss/∂scores]
    },
    
    # Shape constraints
    constraints: [
        Constraint("B == B", nodes=[0,1,2]),
        Constraint("H == H", nodes=[0,1,2]),
        Constraint("S == S", nodes=[0,1,2]),
        Constraint("D == D", nodes=[0,1,2])
    ]
}
```

**Transformations Applied**:
- Shape inference and propagation
- Type checking and coercion
- Automatic differentiation
- Common subexpression elimination
- Constant folding
- Dead code elimination
- Memory layout optimization

## Layer 3: Schedule IR

**Purpose**: Execution scheduling with fusion decisions, tiling strategies, and resource allocation.

**Characteristics**:
- Explicit fusion boundaries
- Tiling and blocking strategies
- Memory hierarchy management
- Parallelization decisions
- Autotuning parameters

**Example Schedule IR**:
```python
ScheduleIR {
    fused_kernels: [
        FusedKernel(
            name="flash_attention_fused",
            operations=[
                TiledOp(
                    op="matmul_qk",
                    tile_size=(BLOCK_M=128, BLOCK_K=64),
                    memory_layout="row_major",
                    precision_policy=NumericalPolicy(
                        compute=bf16, accumulate=f32
                    )
                ),
                TiledOp(
                    op="online_softmax",
                    tile_size=(BLOCK_M=128, BLOCK_N=128),
                    stable_algorithm=True,
                    memory_efficient=True
                ),
                TiledOp(
                    op="matmul_av", 
                    tile_size=(BLOCK_M=128, BLOCK_N=64),
                    fuse_with_previous=True
                )
            ],
            
            memory_plan: {
                shared_memory: {
                    "q_shared": (BLOCK_M, HEAD_DIM, bf16),
                    "k_shared": (BLOCK_N, HEAD_DIM, bf16),
                    "v_shared": (BLOCK_N, HEAD_DIM, bf16)
                },
                register_usage: {
                    "accumulator": (BLOCK_M, HEAD_DIM, f32),
                    "max_vals": (BLOCK_M, f32),
                    "sum_vals": (BLOCK_M, f32)
                }
            },
            
            parallelization: {
                block_dimensions: (BATCH*HEADS*SEQ_LEN//BLOCK_M, 1, 1),
                thread_dimensions: (32, 4, 1),  # warp_size, warps_per_block
                warp_specialization: {
                    0: "producer",  # Data loading
                    1: "consumer",  # Computation
                    2: "consumer", 
                    3: "epilogue"   # Output writing
                }
            },
            
            autotuning_space: {
                "BLOCK_M": [64, 128, 256],
                "BLOCK_N": [64, 128],
                "BLOCK_K": [32, 64],
                "num_stages": [2, 3, 4],
                "num_warps": [4, 8],
                "swizzle": [1, 2, 4, 8]
            }
        )
    ],
    
    # Global scheduling decisions
    device_placement: {"gpu": ["flash_attention_fused"]},
    memory_pools: {
        "workspace": 64*1024*1024,  # 64MB
        "persistent": 16*1024*1024  # 16MB
    }
}
```

**Transformations Applied**:
- Fusion analysis and optimization
- Tiling strategy generation
- Register allocation
- Memory layout optimization
- Parallelization strategy
- Autotuning configuration

## Layer 4: Tile IR

**Purpose**: Hardware-specific tiling with explicit memory hierarchy management and thread/warp coordination.

**Characteristics**:
- Explicit thread block organization
- Memory hierarchy (global → shared → registers)
- Synchronization primitives
- Hardware-specific optimizations
- Precise resource management

**Example Tile IR**:
```python
TileIR {
    kernel: FlashAttentionKernel {
        launch_config: {
            grid_dim: (batch*heads*seq_len//128, 1, 1),
            block_dim: (128, 1, 1),  # 4 warps * 32 threads
            shared_memory: 49152,    # 48KB
            registers_per_thread: 64
        },
        
        tile_operations: [
            # Stage 1: Global → Shared Memory
            TileLoad(
                source=GlobalMemory("q_global"),
                dest=SharedMemory("q_shared", 
                    layout=RowMajorSwizzled(swizzle=4),
                    size=(128, 64)),
                async_copy=True,
                vector_width=8,
                predicated=True
            ),
            
            # Stage 2: Shared → Registers
            TileLoad(
                source=SharedMemory("q_shared"),
                dest=RegisterTile("q_reg",
                    fragment_layout=MatrixFragment(16, 16)),
                ldmatrix=True,
                thread_group=Warp(32)
            ),
            
            # Stage 3: Tensor Core Computation
            TileMMA(
                a=RegisterTile("q_reg"),
                b=RegisterTile("k_reg"),
                c=RegisterTile("scores_reg"),
                instruction=MMAInstruction(
                    shape=(16, 8, 16),
                    dtype=(bf16, bf16, f32),
                    layout="tn"
                ),
                async_execution=True
            ),
            
            # Stage 4: Online Softmax
            TileReduce(
                input=RegisterTile("scores_reg"),
                output=RegisterTile("max_vals"),
                operation="max",
                axis=1,
                warp_reduction=True,
                stable_numerics=True
            ),
            
            TileElementwise(
                inputs=[RegisterTile("scores_reg"), 
                        RegisterTile("max_vals")],
                output=RegisterTile("exp_scores"),
                operation="exp(a - b)",
                stable_math=True
            ),
            
            # Stage 5: Attention × V
            TileMMA(
                a=RegisterTile("exp_scores"),
                b=RegisterTile("v_reg"),
                c=RegisterTile("output_reg"),
                accumulate=True
            ),
            
            # Stage 6: Registers → Global Memory
            TileStore(
                source=RegisterTile("output_reg"),
                dest=GlobalMemory("output_global"),
                async_store=True,
                write_back_cache=True
            )
        ],
        
        synchronization: [
            Barrier(type="shared_memory", stage=1),
            Barrier(type="warp_sync", stage=3),
            Barrier(type="shared_memory", stage=6)
        ],
        
        memory_coalescing: {
            global_loads: "vectorized_128bit",
            shared_stores: "bank_conflict_free",
            register_reuse: "aggressive"
        }
    }
}
```

**Transformations Applied**:
- Thread block decomposition
- Memory hierarchy optimization
- Synchronization insertion
- Register allocation
- Instruction scheduling
- Hardware-specific optimizations

## Layer 5: Target IR (PTX/SASS)

**Purpose**: Platform-specific assembly code with hardware instruction mapping.

**Characteristics**:
- GPU assembly instructions (PTX for NVIDIA, LLVM IR for AMD)
- Register allocation and spilling
- Instruction scheduling for latency hiding
- Platform-specific optimizations
- Binary generation

**Example PTX (NVIDIA)**:
```ptx
// Flash Attention PTX for SM_90 (Hopper)
.version 7.8
.target sm_90
.address_size 64

.visible .entry flash_attention_kernel(
    .param .u64 q_ptr,
    .param .u64 k_ptr, 
    .param .u64 v_ptr,
    .param .u64 o_ptr,
    .param .f32 scale,
    .param .u32 batch_size,
    .param .u32 seq_len,
    .param .u32 head_dim
)
{
    .reg .pred %p<16>;
    .reg .f16 %hf<64>;
    .reg .f32 %f<32>;
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;
    
    // Shared memory declarations
    .shared .align 16 .b8 shared_q[8192];
    .shared .align 16 .b8 shared_k[8192];
    .shared .align 16 .b8 shared_v[8192];
    
    // Calculate thread indices
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r4, %r1, %r3, %r2;
    
    // Load Q tile asynchronously
    cp.async.cg.shared.global [shared_q], [%rd1], 16;
    
    // Tensor core matrix multiplication (Hopper WGMMA)
    wgmma.mma_async.sync.m64n256k32.f32.f16.f16.f32
        {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7},
        %rd2,  // Q matrix descriptor
        %rd3,  // K matrix descriptor  
        {%f8, %f9, %f10, %f11};  // C accumulator
    
    // Wait for async operations
    cp.async.wait_all;
    wgmma.wait_group.sync.aligned 0;
    
    // Online softmax computation
    max.f32 %f12, %f0, %f1;
    max.f32 %f12, %f12, %f2;
    max.f32 %f12, %f12, %f3;
    
    sub.f32 %f0, %f0, %f12;
    sub.f32 %f1, %f1, %f12;
    sub.f32 %f2, %f2, %f12;
    sub.f32 %f3, %f3, %f12;
    
    ex2.approx.f32 %f0, %f0;
    ex2.approx.f32 %f1, %f1;
    ex2.approx.f32 %f2, %f2;
    ex2.approx.f32 %f3, %f3;
    
    add.f32 %f13, %f0, %f1;
    add.f32 %f13, %f13, %f2;
    add.f32 %f13, %f13, %f3;
    
    rcp.approx.f32 %f14, %f13;
    mul.f32 %f0, %f0, %f14;
    mul.f32 %f1, %f1, %f14;
    mul.f32 %f2, %f2, %f14;
    mul.f32 %f3, %f3, %f14;
    
    // Second WGMMA for attention × V
    wgmma.mma_async.sync.m64n256k32.f32.f16.f16.f32
        {%f16, %f17, %f18, %f19, %f20, %f21, %f22, %f23},
        %rd4,  // Attention weights
        %rd5,  // V matrix descriptor
        {%f16, %f17, %f18, %f19};
        
    // Store results
    st.global.f16 [%rd6], %hf0;
    st.global.f16 [%rd6+2], %hf1;
    // ... continue storing
    
    ret;
}
```

**Example PTX for Blackwell SM_100**:
```ptx
// Blackwell-specific optimizations
.version 8.7  
.target sm_100
.address_size 64

.visible .entry blackwell_flash_attention(
    .param .u64 q_ptr,
    .param .u64 k_ptr,
    .param .u64 v_ptr, 
    .param .u64 o_ptr
)
{
    .reg .pred %p<16>;
    .reg .f16 %hf<64>;
    .reg .f32 %f<32>;
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;
    
    // Tensor Memory allocation (Blackwell-specific)
    .local .tmem .align 128 .b8 tmem_accumulator[32768];
    
    // Allocate TMEM dynamically
    tcgen05.alloc %r10, 128;  // 128 columns
    
    // Load descriptors for tcgen05
    mov.u64 %rd10, shared_q_desc;
    mov.u64 %rd11, shared_k_desc;
    
    // CTA pair coordination (2-SM operation)
    tcgen05.mma.cta_group::2.async.m128n256k32.f16.f16.f32
        %r10,      // TMEM accumulator base
        %rd10,     // Q descriptor
        %rd11,     // K descriptor  
        %r12;      // Instruction descriptor
    
    // Block-scaled FP8 computation (hardware scaling)
    tcgen05.mma.mxfp8.block_scale.async.m128n256k32
        %r10,      // TMEM accumulator
        %rd12,     // FP8 Q descriptor
        %rd13,     // FP8 K descriptor
        %r13,      // Scale factors in TMEM
        %r14;      // Instruction descriptor
    
    // TMEM to register transfer for epilogue
    tcgen05.ld %f0, [%r10 + 0];
    tcgen05.ld %f1, [%r10 + 16];
    tcgen05.ld %f2, [%r10 + 32];
    tcgen05.ld %f3, [%r10 + 48];
    
    // Deallocate TMEM
    tcgen05.dealloc %r10;
    
    // Tensor Memory Accelerator for output
    tma.store.2d [%rd15], %f0;
    
    ret;
}
```

**Transformations Applied**:
- Instruction selection
- Register allocation and spilling
- Instruction scheduling
- Peephole optimizations
- Platform-specific code generation

## Cross-Layer Optimizations

### Shape Specialization
```python
# Graph IR → Schedule IR
if all_shapes_static(graph):
    apply_static_optimizations()
    generate_specialized_schedule()
else:
    apply_dynamic_shape_support()
    generate_runtime_dispatch()
```

### Precision Lowering
```python
# Schedule IR → Tile IR
precision_policy = infer_precision_requirements()
if precision_policy.allow_mixed():
    apply_mixed_precision_tiling()
if precision_policy.allow_quantization():
    insert_quantization_ops()
```

### Memory Coalescing
```python
# Tile IR → Target IR  
memory_pattern = analyze_memory_access(tile_ops)
if memory_pattern.is_coalesced():
    generate_vectorized_loads()
else:
    insert_memory_reordering()
```

## Debugging and Introspection

### IR Visualization
```python
# Dump IR at each level
@tessera.debug.dump_ir(levels=["graph", "schedule", "tile"])
def debug_attention(q, k, v):
    return flash_attention(q, k, v)

# Outputs:
# debug_attention.graph.dot  - Computational graph
# debug_attention.schedule.yaml - Execution plan
# debug_attention.tile.ptx - Generated assembly
```

### Performance Analysis
```python
# Profile compilation time per layer
with tessera.profiler.compilation_time():
    compiled_fn = tessera.compile(flash_attention)

# Results:
"""
Compilation Profile:
  Graph IR construction: 12.3ms
  Shape inference: 45.2ms  
  Schedule optimization: 234.1ms (autotuning: 89%)
  Tile generation: 67.8ms
  PTX codegen: 23.4ms
  Total: 382.8ms
"""
```

### Numerical Verification
```python
# Cross-layer numerical consistency checks  
@tessera.verify_numerics(
    reference="torch",
    tolerance=1e-5,
    layers=["graph", "schedule", "tile"]
)
def verified_attention(q, k, v):
    return flash_attention(q, k, v)
```

## Summary

Tessera's multi-layer IR design provides:

1. **Separation of Concerns**: Each layer focuses on specific optimization problems
2. **Progressive Lowering**: High-level semantics gradually transformed to hardware instructions  
3. **Composable Optimizations**: Optimizations can be applied at appropriate abstraction levels
4. **Hardware Abstraction**: Same high-level code generates optimal code for different targets
5. **Debugging Support**: Each layer can be inspected and validated independently

This architecture enables Tessera to achieve both ease of programming and optimal performance by carefully managing the complexity of modern GPU optimization through well-designed abstractions.