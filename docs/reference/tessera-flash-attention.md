# Flash Attention Implementation in Tessera

## Table of Contents
1. [Overview](#overview)
2. [Basic Implementation](#basic-implementation)
3. [Flash Attention v2](#flash-attention-v2)
4. [Flash Attention v3](#flash-attention-v3)
5. [Optimizations](#optimizations)
6. [Multi-GPU Flash Attention](#multi-gpu-flash-attention)
7. [Performance Analysis](#performance-analysis)
8. [Debugging and Testing](#debugging-and-testing)

## Overview

Flash Attention is a memory-efficient attention mechanism that computes attention without materializing the full attention matrix. Tessera provides optimized implementations that leverage modern GPU features.

### Key Concepts

1. **Online Softmax**: Compute softmax incrementally without storing full attention matrix
2. **Tiling**: Process attention in blocks to fit in shared memory
3. **Recomputation**: Trade compute for memory in backward pass
4. **IO-Awareness**: Minimize HBM accesses

## Basic Implementation

### Simple Flash Attention

```python
import tessera as ts
import math

@ts.kernel
def flash_attention_basic(
    Q: ts.Tensor["batch*heads", "seq_len", "head_dim", ts.bf16],
    K: ts.Tensor["batch*heads", "seq_len", "head_dim", ts.bf16],
    V: ts.Tensor["batch*heads", "seq_len", "head_dim", ts.bf16],
    scale: ts.f32,
    is_causal: bool = False
) -> ts.Tensor["batch*heads", "seq_len", "head_dim", ts.bf16]:
    """Basic Flash Attention implementation"""
    
    batch_heads, seq_len, head_dim = Q.shape
    
    # Tile sizes (tunable)
    BLOCK_M = 64  # Q block size
    BLOCK_N = 64  # K,V block size
    
    # Allocate output
    O = ts.zeros_like(Q)
    
    # Process each sequence position
    for q_start in range(0, seq_len, BLOCK_M):
        q_end = min(q_start + BLOCK_M, seq_len)
        
        # Initialize accumulators for this Q block
        m_i = ts.full((q_end - q_start,), -float('inf'), ts.f32)
        l_i = ts.zeros((q_end - q_start,), ts.f32)
        acc = ts.zeros((q_end - q_start, head_dim), ts.f32)
        
        # Load Q block
        Q_block = Q[:, q_start:q_end, :]
        
        # Iterate over K,V blocks
        kv_end = seq_len if not is_causal else q_end
        for kv_start in range(0, kv_end, BLOCK_N):
            kv_block_end = min(kv_start + BLOCK_N, kv_end)
            
            # Load K,V blocks
            K_block = K[:, kv_start:kv_block_end, :]
            V_block = V[:, kv_start:kv_block_end, :]
            
            # Compute attention scores
            S = ts.matmul(Q_block, K_block.transpose(-2, -1)) * scale
            
            # Apply causal mask if needed
            if is_causal:
                mask = ts.triu(ts.ones_like(S) * -float('inf'), diagonal=kv_start - q_start + 1)
                S = S + mask
            
            # Online softmax update
            m_new = ts.max(S, dim=-1)
            m_i_new = ts.maximum(m_i, m_new)
            
            # Correction factors
            alpha = ts.exp(m_i - m_i_new)
            beta = ts.exp(m_new - m_i_new)
            
            # Update statistics
            P = ts.exp(S - m_i_new.unsqueeze(-1))
            l_i = alpha * l_i + beta * ts.sum(P, dim=-1)
            
            # Update accumulator
            acc = alpha.unsqueeze(-1) * acc + ts.matmul(P, V_block)
            
            # Update m_i for next iteration
            m_i = m_i_new
        
        # Write output for this Q block
        O[:, q_start:q_end, :] = ts.cast(acc / l_i.unsqueeze(-1), ts.bf16)
    
    return O
```

## Flash Attention v2

### Optimized Implementation with Autotuning

```python
@ts.kernel.autotune(
    configs=[
        {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
        {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2},
        {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 3},
        {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4, "num_stages": 3},
    ],
    key=["seq_len", "head_dim", "is_causal"]
)
def flash_attention_v2(
    Q: ts.Tensor["B*H", "S", "D", ts.bf16],
    K: ts.Tensor["B*H", "S", "D", ts.bf16],
    V: ts.Tensor["B*H", "S", "D", ts.bf16],
    dropout_p: float = 0.0,
    is_causal: bool = True
) -> ts.Tensor["B*H", "S", "D", ts.bf16]:
    """Flash Attention v2 with optimizations"""
    
    ctx = ts.tile.context()
    B_H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Get autotuned parameters
    BLOCK_M = ctx.BLOCK_M
    BLOCK_N = ctx.BLOCK_N
    
    # Allocate shared memory with swizzling
    smem_q = ts.tile.alloc_shared((BLOCK_M, D), ts.bf16, swizzle="xor")
    smem_k = ts.tile.alloc_shared((BLOCK_N, D), ts.bf16, swizzle="xor")
    smem_v = ts.tile.alloc_shared((BLOCK_N, D), ts.bf16, swizzle="xor")
    
    # Output accumulator in registers
    O = ts.zeros((B_H, S, D), ts.bf16)
    
    # Grid-stride loop for batch*heads dimension
    for bh in ts.tile.grid_stride_range(0, B_H):
        # Process Q blocks
        for q_block_id in ts.tile.range(0, S, BLOCK_M):
            # Initialize online softmax state
            m_i = ts.tile.alloc_register((BLOCK_M,), ts.f32)
            l_i = ts.tile.alloc_register((BLOCK_M,), ts.f32)
            acc = ts.tile.alloc_register((BLOCK_M, D), ts.f32)
            
            ts.tile.fill(m_i, -float('inf'))
            ts.tile.fill(l_i, 0.0)
            ts.tile.fill(acc, 0.0)
            
            # Prefetch Q block
            q_end = min(q_block_id + BLOCK_M, S)
            ts.tile.cp_async(smem_q, Q[bh, q_block_id:q_end, :], stages=2)
            
            # Determine KV range for causal attention
            kv_max = S if not is_causal else q_end
            
            # Double-buffered KV loading
            write_stage = 0
            compute_stage = 1
            stages = [
                (ts.tile.alloc_shared((BLOCK_N, D), ts.bf16),
                 ts.tile.alloc_shared((BLOCK_N, D), ts.bf16)),
                (ts.tile.alloc_shared((BLOCK_N, D), ts.bf16),
                 ts.tile.alloc_shared((BLOCK_N, D), ts.bf16))
            ]
            
            # Prefetch first KV blocks
            if kv_max > 0:
                kv_end = min(BLOCK_N, kv_max)
                ts.tile.cp_async(stages[write_stage][0], K[bh, 0:kv_end, :])
                ts.tile.cp_async(stages[write_stage][1], V[bh, 0:kv_end, :])
            
            # Main KV loop with pipelining
            for kv_block_id in ts.tile.range(0, kv_max, BLOCK_N):
                # Swap stages
                write_stage, compute_stage = compute_stage, write_stage
                
                # Start loading next KV blocks
                next_kv = kv_block_id + BLOCK_N
                if next_kv < kv_max:
                    next_kv_end = min(next_kv + BLOCK_N, kv_max)
                    ts.tile.cp_async(stages[write_stage][0], K[bh, next_kv:next_kv_end, :])
                    ts.tile.cp_async(stages[write_stage][1], V[bh, next_kv:next_kv_end, :])
                
                # Wait for current stage data
                ts.tile.cp_wait_group(0)
                ts.tile.barrier()
                
                # Compute attention scores using tensor cores
                S = ts.tile.mma(smem_q, stages[compute_stage][0].T, precision="tf32")
                S = S * scale
                
                # Apply causal mask
                if is_causal:
                    S = ts.tile.apply_causal_mask(S, q_block_id, kv_block_id)
                
                # Online softmax with numerical stability
                m_new = ts.tile.row_max(S)
                m_i_new = ts.tile.maximum(m_i, m_new)
                
                # Compute correction factors
                alpha = ts.tile.exp(m_i - m_i_new)
                beta = ts.tile.exp(m_new - m_i_new)
                
                # Update statistics
                P = ts.tile.exp(S - ts.tile.broadcast(m_i_new, dim=1))
                l_new = ts.tile.row_sum(P)
                l_i = alpha * l_i + beta * l_new
                
                # Apply dropout if specified
                if dropout_p > 0:
                    P = ts.tile.dropout(P, p=dropout_p)
                
                # Update accumulator with tensor cores
                acc = ts.tile.scale(acc, alpha)
                acc = ts.tile.mma(P, stages[compute_stage][1], accumulator=acc)
                
                # Update m_i for next iteration
                m_i = m_i_new
            
            # Finalize and store output
            output = acc / ts.tile.broadcast(l_i, dim=1)
            output = ts.tile.cast(output, ts.bf16)
            ts.tile.store(O[bh, q_block_id:q_end, :], output)
    
    return O
```

## Flash Attention v3

### Hopper-Optimized Implementation

```python
@ts.kernel.target("hopper")  # Requires H100 or newer
def flash_attention_v3(
    Q: ts.Tensor["B*H", "S", "D", ts.bf16],
    K: ts.Tensor["B*H", "S", "D", ts.bf16],
    V: ts.Tensor["B*H", "S", "D", ts.bf16],
    window_size: int = -1,  # Local attention window
    alibi_slopes: ts.Tensor["H"] = None  # ALiBi support
) -> ts.Tensor["B*H", "S", "D", ts.bf16]:
    """Flash Attention v3 with Hopper optimizations"""
    
    ctx = ts.tile.context()
    B_H, S, D = Q.shape
    
    # Hopper-specific configurations
    BLOCK_M = 128  # Larger blocks on Hopper
    BLOCK_N = 128
    NUM_WARPS = 8
    CLUSTER_SIZE = (2, 2, 1)  # Thread block clusters
    
    # Enable persistent kernel mode
    ts.tile.set_persistent_kernel(True)
    
    # Use distributed shared memory across cluster
    smem_q = ts.tile.alloc_distributed_shared(
        (BLOCK_M, D), ts.bf16, 
        cluster_size=CLUSTER_SIZE,
        swizzle="hopper_native"
    )
    smem_k = ts.tile.alloc_distributed_shared(
        (BLOCK_N, D), ts.bf16,
        cluster_size=CLUSTER_SIZE,
        swizzle="hopper_native"
    )
    smem_v = ts.tile.alloc_distributed_shared(
        (BLOCK_N, D), ts.bf16,
        cluster_size=CLUSTER_SIZE,
        swizzle="hopper_native"
    )
    
    # TMA (Tensor Memory Accelerator) descriptors
    tma_q = ts.tile.create_tma_descriptor(Q, (BLOCK_M, D))
    tma_k = ts.tile.create_tma_descriptor(K, (BLOCK_N, D))
    tma_v = ts.tile.create_tma_descriptor(V, (BLOCK_N, D))
    
    # Asynchronous barriers for TMA
    mbar = ts.tile.create_mbarrier()
    
    O = ts.zeros_like(Q)
    
    # Grid-stride loop with cluster coordination
    for bh in ts.tile.cluster_range(0, B_H):
        for q_tile in ts.tile.range(0, S, BLOCK_M):
            # Warp specialization
            warp_id = ts.tile.warp_id()
            
            if warp_id < NUM_WARPS // 2:
                # First half of warps: handle Q and scores
                # TMA bulk load Q
                ts.tile.tma_load_async(
                    smem_q, tma_q, 
                    indices=(bh, q_tile),
                    multicast=True,
                    cache_policy="L2_priority"
                )
            else:
                # Second half: handle K,V loading
                pass  # Handled in loop below
            
            # Initialize states with WGMMA layout
            m_i = ts.tile.alloc_register((BLOCK_M,), ts.f32, layout="wgmma")
            l_i = ts.tile.alloc_register((BLOCK_M,), ts.f32, layout="wgmma")
            acc = ts.tile.alloc_register((BLOCK_M, D), ts.f32, layout="wgmma")
            
            # Determine window boundaries for local attention
            kv_start = 0 if window_size < 0 else max(0, q_tile - window_size)
            kv_end = S if window_size < 0 else min(S, q_tile + BLOCK_M + window_size)
            
            # Split-K parallelization for long sequences
            if S > 4096:
                num_splits = 4
                split_id = ts.tile.cluster_rank() % num_splits
                kv_start = kv_start + (kv_end - kv_start) * split_id // num_splits
                kv_end = kv_start + (kv_end - kv_start) * (split_id + 1) // num_splits
            
            # Main computation loop with TMA and WGMMA
            for kv_tile in ts.tile.range(kv_start, kv_end, BLOCK_N):
                # TMA bulk loads with multicast
                ts.tile.tma_load_async(
                    smem_k, tma_k,
                    indices=(bh, kv_tile),
                    multi