"""
Matrix Multiplication Example for Nvidia Blackwell B200 GPU
Using Tensor Cores via Tessera Programming Model

This example demonstrates:
- Blackwell SM_100 specific optimizations
- Tensor Memory (TMEM) utilization 
- CTA pair optimization
- Block-scaled FP8 precision
- Advanced tensor core operations (tcgen05.mma)
"""

import tessera
from tessera.blackwell import *

@tessera.blackwell_optimized
class BlackwellMatMulStack:
    """Complete Blackwell B200 optimization stack for matrix multiplication"""
    
    @tessera.component
    def memory_hierarchy(self):
        return tessera.MemoryHierarchy([
            tessera.TensorMemory(capacity="256KB", power_efficiency=2.0, bandwidth="19TB/s"),
            tessera.SharedMemory(capacity="256KB", bandwidth="19TB/s"),
            tessera.GlobalMemory(bandwidth="3.3TB/s")
        ])
    
    @tessera.component  
    def tensor_cores(self):
        return tessera.TensorCores([
            tessera.TCGEN05(
                throughput="3200 TFLOP/s",
                precisions=["fp16", "bf16", "fp8", "mxfp8", "nvfp4"],
                tile_sizes=[(128, 256, 32), (256, 128, 32), (64, 256, 32)]
            )
        ])
    
    @tessera.component
    def synchronization(self):
        return tessera.SyncPrimitives([
            tessera.TCGen05Barrier(),
            tessera.CTAGroupBarrier(), 
            tessera.TMEMFence()
        ])

@tessera.precision_policy
class BlackwellPrecisionPolicy:
    """Native block-scaled precision support on Blackwell"""
    formats = {
        "mxfp8": tessera.MXFP8BlockScaled(block_size=32),
        "nvfp4": tessera.NVFP4BlockScaled(block_size=32),
        "mxfp6": tessera.MXFP6BlockScaled(block_size=32)
    }
    
    def create_scale_layout(self, M: int, N: int, K: int):
        """Create hardware-optimized scale factor layout"""
        return tessera.BlockScaledLayout(
            basic_block=(128, 4),  # 128 M/N, 4 scale factors in K
            storage_order="k_major"
        )

@tessera.kernel.target("sm_100")  # Blackwell B200 architecture
@tessera.kernel.precision(BlackwellPrecisionPolicy)
@tessera.kernel.cta_group_size(2)  # Enable CTA pair optimization
def blackwell_matmul_fp8(
    A: tessera.Tensor["M", "K", tessera.MXFP8],
    B: tessera.Tensor["K", "N", tessera.MXFP8], 
    C: tessera.Tensor["M", "N", tessera.FP32],
    A_scales: tessera.Tensor,
    B_scales: tessera.Tensor,
    M: tessera.Dim,
    N: tessera.Dim,
    K: tessera.Dim
):
    """
    High-performance matrix multiplication using Blackwell B200 features:
    - Tensor Memory (TMEM) for accumulation
    - CTA pair optimization for 2x shared memory
    - Block-scaled FP8 precision
    - tcgen05.mma instructions
    """
    
    # Configure tile sizes for optimal B200 performance
    BM, BN, BK = 128, 256, 32
    
    # Get CTA coordinates and configure CTA pairs
    cta_m = tessera.program_id(0)
    cta_n = tessera.program_id(1)
    
    # CTA pair coordination - each pair processes double the shared memory
    cta_group = tessera.cta_group_id()
    is_leader = tessera.cta_group_rank() == 0
    
    # Allocate Tensor Memory (TMEM) - 256KB per SM, separate from shared memory
    tmem_accumulator = tessera.tmem_alloc(
        shape=(BM, BN),
        dtype=tessera.FP32,
        layout="row_major"
    )
    
    # Shared memory allocation with CTA pair optimization (2x capacity)
    shared_memory_size = 512 * 1024 if tessera.cta_group_size() == 2 else 256 * 1024
    
    A_shared = tessera.shared_alloc(
        shape=(BM, BK), 
        dtype=tessera.MXFP8,
        swizzle_pattern="cute_128B"  # Bank conflict avoidance
    )
    
    B_shared = tessera.shared_alloc(
        shape=(BK, BN),
        dtype=tessera.MXFP8, 
        swizzle_pattern="cute_128B"
    )
    
    # Load scale factors into TMEM with optimized layout
    tessera.tcgen05_cp_scales_to_tmem(
        [A_scales, B_scales],
        layout="chunk_based_4warp_duplication",
        tmem_offset=0
    )
    
    # Initialize TMEM accumulator
    tessera.tmem_fill(tmem_accumulator, 0.0)
    
    # Main computation loop with software pipelining
    num_k_tiles = tessera.ceildiv(K, BK)
    
    for k_tile in tessera.range_with_prefetch(num_k_tiles, prefetch_stages=3):
        # Asynchronous loading with Tensor Memory Accelerator (TMA)
        if k_tile < num_k_tiles:
            # Calculate global memory offsets
            A_offset = (cta_m * BM, k_tile * BK) 
            B_offset = (k_tile * BK, cta_n * BN)
            
            # TMA copy with optimal 2D patterns
            tessera.tma_copy_2d_async(
                src=A[A_offset],
                dst=A_shared,
                shape=(BM, BK),
                src_stride=(K, 1),
                dst_stride=(BK, 1)
            )
            
            tessera.tma_copy_2d_async(
                src=B[B_offset],
                dst=B_shared, 
                shape=(BK, BN),
                src_stride=(N, 1),
                dst_stride=(BN, 1)
            )
        
        # Wait for async copies to complete
        tessera.tma_wait_group(0)
        
        # CTA group barrier for coordination
        tessera.cta_group_barrier()
        
        # Block-scaled matrix multiplication using tcgen05.mma
        # This accumulates directly in TMEM, avoiding register spills
        tessera.tcgen05_mma_cta_group_async(
            tmem_acc=tmem_accumulator,
            A_shared=A_shared,
            B_shared=B_shared,
            scales=(A_scales, B_scales),
            instruction="tcgen05.mma.cta_group::2.async.m128n256k32.mxfp8.mxfp8.f32",
            tile_shape=(BM, BN, BK)
        )
        
        # CTA group barrier before next iteration
        tessera.cta_group_barrier()
    
    # Wait for all TMEM operations to complete
    tessera.tmem_fence()
    
    # Store results from TMEM to global memory
    C_offset = (cta_m * BM, cta_n * BN)
    
    # Direct TMEM to global memory transfer (bypasses registers)
    tessera.tcgen05_cp_tmem_to_global(
        tmem_src=tmem_accumulator,
        global_dst=C[C_offset],
        shape=(BM, BN),
        dst_stride=(N, 1)
    )

@tessera.kernel.target("sm_100")
@tessera.kernel.autotune({
    "BM": [64, 128, 256],
    "BN": [128, 256, 512], 
    "BK": [32, 64],
    "cta_group_size": [1, 2],
    "precision": ["fp16", "mxfp8", "nvfp4"]
})
def adaptive_blackwell_matmul(
    A: tessera.Tensor,
    B: tessera.Tensor,
    C: tessera.Tensor
):
    """Automatically optimized matrix multiplication with parameter search"""
    
    # Tessera automatically explores the configuration space
    # and selects optimal parameters for the specific problem size
    return tessera.matmul_template(A, B, C, hardware="blackwell_b200")

# High-level API usage example
def blackwell_matmul_example():
    """Complete example of using Blackwell B200 optimized matrix multiplication"""
    
    # Problem dimensions
    M, N, K = 8192, 8192, 8192
    
    # Create tensors with block-scaled FP8 format
    A = tessera.randn((M, K), dtype=tessera.MXFP8, device="cuda")
    B = tessera.randn((K, N), dtype=tessera.MXFP8, device="cuda") 
    C = tessera.zeros((M, N), dtype=tessera.FP32, device="cuda")
    
    # Create scale factors for block-scaled precision
    A_scales = tessera.create_block_scales(A, block_size=32)
    B_scales = tessera.create_block_scales(B, block_size=32)
    
    # Launch optimized kernel
    with tessera.device("cuda"):
        blackwell_matmul_fp8(
            A, B, C, A_scales, B_scales,
            M=M, N=N, K=K
        )
    
    # Performance metrics
    metrics = tessera.profile_last_kernel()
    print(f"Performance: {metrics.throughput_tflops:.1f} TFLOPS")
    print(f"Memory Bandwidth: {metrics.memory_bandwidth_gb_s:.1f} GB/s") 
    print(f"Tensor Core Utilization: {metrics.tensor_core_utilization:.1f}%")
    print(f"TMEM Utilization: {metrics.tmem_utilization:.1f}%")
    
    return C

# Generated Target IR (PTX assembly) for reference
TARGET_IR_EXAMPLE = """
// Generated PTX for Blackwell SM_100
.version 8.7
.target sm_100
.address_size 64

.visible .entry blackwell_matmul_fp8_kernel(
    .param .u64 A_ptr,
    .param .u64 B_ptr, 
    .param .u64 C_ptr,
    .param .u64 A_scales_ptr,
    .param .u64 B_scales_ptr,
    .param .u32 M,
    .param .u32 N,
    .param .u32 K
) {
    .reg .pred %p<16>;
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>; 
    .reg .f32 %f<64>;
    
    // Tensor Memory allocation (Blackwell-specific)
    .shared .tmem .align 256 .b8 tmem_accumulator[32768]; // 128x256 f32
    .shared .align 128 .b8 shared_memory[65536];          // 64KB shared
    
    // CTA group coordination
    .reg .u32 %cta_group_id, %cta_group_rank;
    mov.u32 %cta_group_id, %ctaid.x;
    mov.u32 %cta_group_rank, %tid.x;
    
    // Load scale factors into TMEM
    tcgen05.cp.async.scales.to.tmem [tmem_accumulator], [%rd4], %r10;
    
    // Initialize TMEM accumulator
    tcgen05.fill.tmem.f32 tmem_accumulator, 0.0f;
    
    // Main computation loop
    loop_k:
        // TMA async copy A tile
        cp.async.bulk.tensor.2d.shared::cta.global [%rd8], [%rd1, {%r1, %r2}];
        
        // TMA async copy B tile  
        cp.async.bulk.tensor.2d.shared::cta.global [%rd9], [%rd2, {%r3, %r4}];
        
        // Wait for copies
        cp.async.bulk.wait_group 0;
        
        // CTA group barrier
        bar.cta_group.sync;
        
        // Block-scaled matrix multiply with TMEM accumulation
        tcgen05.mma.cta_group::2.async.m128n256k32.mxfp8.mxfp8.f32
            tmem_accumulator, [%rd8], [%rd9], tmem_accumulator, %rd10;
            
        // Loop control
        add.s32 %r5, %r5, 32;
        setp.lt.s32 %p1, %r5, %K;
        @%p1 bra loop_k;
    
    // Store TMEM results to global memory
    tcgen05.cp.tmem.to.global [%rd3], tmem_accumulator;
    
    ret;
}
"""

if __name__ == "__main__":
    # Run the example
    result = blackwell_matmul_example()
    print("Blackwell B200 matrix multiplication completed successfully!")
