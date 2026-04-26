# Tessera Compiler Passes Overview

The Tessera compiler implements a multi-level IR stack that transforms high-level Python code into optimized target code for GPUs and other accelerators. This document provides an overview of the compiler architecture and the major passes that transform code through each IR level.

## Compiler Architecture

Tessera uses a four-level IR hierarchy:

```
Python/Tessera DSL
        ↓
   Graph IR (MLIR)
        ↓
  Schedule IR (MLIR)
        ↓
   Tile IR (MLIR)
        ↓
  Target IR (PTX/LLVM)
```

Each level serves specific purposes:

- **Graph IR**: High-level operations, autodiff, effects, and algebraic simplification
- **Schedule IR**: Loop tiling, fusion, memory placement, and parallelization
- **Tile IR**: Explicit tile operations, shared memory management, and hardware intrinsics
- **Target IR**: Backend-specific code generation (PTX, CUDA Tile IR, LLVM)

## Pass Organization

The compiler is organized into several major pass families:

### 1. Frontend Passes (Python → Graph IR)
- **AST Lowering**: Converts Tessera Python to Graph IR
- **Type Inference**: Infers tensor shapes and dtypes
- **Effect Analysis**: Tracks side effects (RNG, state, collectives)
- **Symbol Resolution**: Resolves symbolic dimensions and mesh layouts

### 2. Graph IR Passes
- **Autodiff Passes**: Forward/reverse mode differentiation
- **Algebraic Simplification**: Mathematical optimizations
- **Fusion Analysis**: Identifies fusable operations
- **Distribution Passes**: Handles mesh parallelism and sharding

### 3. Schedule IR Passes
- **Tiling Passes**: Loop tiling and blocking strategies
- **Memory Placement**: Assigns tensors to memory hierarchy
- **Pipeline Generation**: Creates async execution pipelines
- **Autotuning Integration**: Explores scheduling search spaces

### 4. Tile IR Passes
- **Memory Management**: Shared memory allocation and barriers
- **Intrinsic Lowering**: Maps to hardware-specific operations
- **Register Allocation**: Manages register pressure
- **Collective Insertion**: Adds distributed communication ops

### 5. Target IR Passes
- **Code Generation**: Emits PTX, CUDA Tile IR, or LLVM
- **Optimization**: Target-specific peephole optimizations
- **Runtime Integration**: Links with CUDA/HIP/NCCL runtime

## Example: Flash Attention Compilation

Let's trace how a Flash Attention kernel flows through the compiler:

### Input: Python/Tessera
```python
@tessera.kernel
def flash_attention(Q: Tensor["B","H","S","D", bf16],
                   K: Tensor["B","H","S","D", bf16],
                   V: Tensor["B","H","S","D", bf16]) -> Tensor["B","H","S","D", bf16]:
    return ops.flash_attention(Q, K, V, causal=True)
```

### Graph IR (Simplified)
```mlir
func @flash_attention(%Q: tensor<?x?x?x?xbf16>, 
                     %K: tensor<?x?x?x?xbf16>,
                     %V: tensor<?x?x?x?xbf16>) -> tensor<?x?x?x?xbf16> {
  %0 = tessera.flash_attention %Q, %K, %V {causal = true} : 
    tensor<?x?x?x?xbf16>, tensor<?x?x?x?xbf16>, tensor<?x?x?x?xbf16> -> tensor<?x?x?x?xbf16>
  return %0 : tensor<?x?x?x?xbf16>
}
```

### Schedule IR (After Tiling)
```mlir
func @flash_attention_tiled(%Q: memref<?x?x?x?xbf16>, 
                           %K: memref<?x?x?x?xbf16>,
                           %V: memref<?x?x?x?xbf16>,
                           %O: memref<?x?x?x?xbf16>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  
  scf.for %bh = %c0 to %batch_heads step %c1 {
    scf.for %m_block = %c0 to %seq_len step %c128 {
      %q_tile = memref.subview %Q[%bh, 0, %m_block, 0] [1, %heads, 128, %head_dim] [1, 1, 1, 1]
      %o_tile = memref.subview %O[%bh, 0, %m_block, 0] [1, %heads, 128, %head_dim] [1, 1, 1, 1]
      
      scf.for %n_block = %c0 to %seq_len step %c128 {
        %k_tile = memref.subview %K[%bh, 0, %n_block, 0] [1, %heads, 128, %head_dim] [1, 1, 1, 1]
        %v_tile = memref.subview %V[%bh, 0, %n_block, 0] [1, %heads, 128, %head_dim] [1, 1, 1, 1]
        
        // Online softmax computation
        tessera.online_softmax %q_tile, %k_tile, %v_tile, %o_tile : 
          memref<1x?x128x?xbf16>, memref<1x?x128x?xbf16>, memref<1x?x128x?xbf16>, memref<1x?x128x?xbf16>
      }
    }
  }
  return
}
```

### Tile IR (With Explicit Memory Management)
```mlir
func @flash_attention_tile(%Q_global: memref<?x?x?x?xbf16>, 
                          %K_global: memref<?x?x?x?xbf16>,
                          %V_global: memref<?x?x?x?xbf16>,
                          %O_global: memref<?x?x?x?xbf16>) {
  %smem_q = tessera.alloc_shared() : memref<128x128xbf16, 3>
  %smem_k = tessera.alloc_shared() : memref<128x128xbf16, 3>
  %smem_v = tessera.alloc_shared() : memref<128x128xbf16, 3>
  
  %acc = tessera.alloc_register() : memref<128x128xf32, 5>
  %m_state = tessera.alloc_register() : memref<128xf32, 5>
  %l_state = tessera.alloc_register() : memref<128xf32, 5>
  
  // Async copy Q tile to shared memory
  tessera.cp_async %Q_global, %smem_q : memref<?x?x?x?xbf16>, memref<128x128xbf16, 3>
  
  scf.for %kv_block = %c0 to %seq_len step %c128 {
    // Async copy K,V tiles
    tessera.cp_async %K_global, %smem_k : memref<?x?x?x?xbf16>, memref<128x128xbf16, 3>
    tessera.cp_async %V_global, %smem_v : memref<?x?x?x?xbf16>, memref<128x128xbf16, 3>
    tessera.cp_commit
    tessera.cp_wait
    
    // Compute attention scores
    %scores = tessera.mma %smem_q, %smem_k : memref<128x128xbf16, 3>, memref<128x128xbf16, 3> -> memref<128x128xf32, 5>
    
    // Online softmax update
    %m_new, %l_new = tessera.online_softmax_update %scores, %m_state, %l_state : 
      memref<128x128xf32, 5>, memref<128xf32, 5>, memref<128xf32, 5> -> memref<128xf32, 5>, memref<128xf32, 5>
    
    // Update accumulator
    %prob = tessera.softmax_normalize %scores, %m_new, %l_new : 
      memref<128x128xf32, 5>, memref<128xf32, 5>, memref<128xf32, 5> -> memref<128x128xbf16, 5>
    %update = tessera.mma %prob, %smem_v : memref<128x128xbf16, 5>, memref<128x128xbf16, 3> -> memref<128x128xf32, 5>
    %acc = tessera.accumulate %acc, %update, %m_state, %m_new, %l_state, %l_new : 
      memref<128x128xf32, 5>, memref<128x128xf32, 5>, memref<128xf32, 5>, memref<128xf32, 5>, memref<128xf32, 5>, memref<128xf32, 5> -> memref<128x128xf32, 5>
    
    %m_state = %m_new : memref<128xf32, 5>
    %l_state = %l_new : memref<128xf32, 5>
  }
  
  // Finalize and store result
  %result = tessera.finalize_softmax %acc, %l_state : memref<128x128xf32, 5>, memref<128xf32, 5> -> memref<128x128xbf16, 5>
  tessera.cp_async %result, %O_global : memref<128x128xbf16, 5>, memref<?x?x?x?xbf16>
  return
}
```

### Target IR (PTX/CUDA Tile IR)
```ptx
.version 8.0
.target sm_90
.address_size 64

.visible .entry flash_attention_kernel(
    .param .u64 Q_ptr,
    .param .u64 K_ptr,
    .param .u64 V_ptr,
    .param .u64 O_ptr,
    .param .u32 batch_size,
    .param .u32 num_heads,
    .param .u32 seq_len,
    .param .u32 head_dim
) {
    .reg .pred %p<16>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .reg .f16 %h<256>;
    .reg .f32 %f<128>;
    
    // Shared memory allocation
    .shared .align 16 .b8 smem_q[32768];  // 128x128 bf16
    .shared .align 16 .b8 smem_k[32768];  
    .shared .align 16 .b8 smem_v[32768];
    
    // Thread and block indexing
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    
    // Async copy Q tile using cp.async
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes 
        [smem_q], [%rd1, {%r3, %r4}], [%rd2], 16;
    
    // Main computation loop
loop_kv:
    // Load K, V tiles with cp.async
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes 
        [smem_k], [%rd3, {%r5, %r6}], [%rd2], 16;
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes 
        [smem_v], [%rd4, {%r7, %r8}], [%rd2], 16;
    
    cp.async.commit_group;
    cp.async.wait_group 0;
    
    // Matrix multiply using wgmma (Hopper instruction)
    wgmma.mma_async.sync.m64n256k32.f32.bf16.bf16 
        {%f0, %f1, %f2, %f3}, 
        smem_q+0, smem_k+0, %r9;
    
    // Online softmax computation
    // ... (detailed softmax implementation)
    
    // Accumulate with previous results
    fma.rn.f32 %f16, %f0, %f12, %f16;
    // ... (more accumulation)
    
    add.s32 %r10, %r10, 128;
    setp.lt.s32 %p1, %r10, %r11;
    @%p1 bra loop_kv;
    
    // Store final result
    cp.async.bulk.tensor.2d.global.shared.mbarrier::complete_tx::bytes 
        [%rd5, {%r12, %r13}], [smem_result], [%rd2], 16;
    
    ret;
}
```

## Pass Infrastructure

Tessera uses MLIR's pass infrastructure with custom pass managers for each IR level:

```python
# Example pass pipeline configuration
pipeline = tessera.PassPipeline([
    # Graph IR passes
    tessera.passes.TypeInferencePass(),
    tessera.passes.AutodiffPass(),
    tessera.passes.AlgebraicSimplificationPass(),
    tessera.passes.FusionAnalysisPass(),
    
    # Lower to Schedule IR
    tessera.passes.LowerToScheduleIRPass(),
    
    # Schedule IR passes
    tessera.passes.TilingPass(tile_sizes=[128, 128]),
    tessera.passes.MemoryPlacementPass(),
    tessera.passes.PipelineGenerationPass(stages=3),
    
    # Lower to Tile IR
    tessera.passes.LowerToTileIRPass(),
    
    # Tile IR passes
    tessera.passes.SharedMemoryAllocationPass(),
    tessera.passes.IntrinsicLoweringPass(),
    tessera.passes.CollectiveInsertionPass(),
    
    # Lower to Target IR
    tessera.passes.LowerToPTXPass(),
])
```

## Next Documents

This overview will be followed by detailed documentation on:

1. **Graph IR Passes** - Autodiff, algebraic simplification, and high-level optimizations
2. **Schedule IR Passes** - Tiling, memory placement, and pipeline generation
3. **Tile IR Passes** - Memory management, intrinsics, and hardware mapping
4. **Target IR Passes** - Code generation and backend-specific optimizations
5. **Autotuning Integration** - How passes interact with the autotuning system
6. **Distributed Compilation** - Handling mesh parallelism and collectives

Each document will include detailed code examples, pass implementations, and debugging techniques for that IR level.
