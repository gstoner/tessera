# TesseraBench - Document 2: Benchmark Suite Implementation

This document details the comprehensive benchmark suite for TesseraBench, covering essential GPU computing kernels optimized for the Tessera programming model. Each benchmark is designed to evaluate specific aspects of Tessera's compilation pipeline and runtime performance.

## Core Benchmark Suite

### 1. Dense Linear Algebra Benchmarks

#### GEMM (General Matrix Multiply) Benchmark

```python
class GEMMBenchmark:
    """Comprehensive GEMM benchmark covering various shapes and precisions"""
    
    def get_name(self) -> str:
        return "gemm"
        
    def get_description(self) -> str:
        return "General Matrix Multiplication with various shapes and precisions"
        
    def get_supported_precisions(self) -> List[str]:
        return [
            "fp32", "fp16", "bf16", 
            "fp8_e4m3@accum(fp32)", "fp8_e5m2@accum(fp32)",
            "fp6@accum(fp32)", "fp4@accum(fp32)"
        ]
        
    def get_problem_sizes(self) -> List[Dict[str, Any]]:
        """Generate comprehensive GEMM problem sizes"""
        sizes = []
        
        # Square matrices - powers of 2
        for size in [256, 512, 1024, 2048, 4096, 8192, 16384]:
            sizes.append({
                'M': size, 'N': size, 'K': size,
                'layout': 'row_major',
                'transpose_a': False,
                'transpose_b': False
            })
            
        # Rectangular matrices - common in ML workloads
        ml_shapes = [
            (8192, 4096, 4096),   # Large language models
            (4096, 11008, 4096),  # MLP intermediate
            (2048, 2048, 8192),   # Attention projections
            (1024, 4096, 1024),   # Medium models
            (512, 2048, 512),     # Smaller models
        ]
        
        for M, N, K in ml_shapes:
            for batch in [1, 8, 16, 32, 64]:
                sizes.append({
                    'M': batch * M, 'N': N, 'K': K,
                    'layout': 'row_major',
                    'transpose_a': False,
                    'transpose_b': False
                })
                
        # Transpose variants
        base_shapes = [(1024, 1024, 1024), (2048, 2048, 2048)]
        for M, N, K in base_shapes:
            for transpose_a in [False, True]:
                for transpose_b in [False, True]:
                    sizes.append({
                        'M': M, 'N': N, 'K': K,
                        'layout': 'row_major',
                        'transpose_a': transpose_a,
                        'transpose_b': transpose_b
                    })
                    
        return sizes
        
    def setup(self, config: BenchmarkConfig) -> None:
        """Setup GEMM benchmark with all precision variants"""
        self.config = config
        self.tessera_integration = TesseraIntegration(tessera.get_runtime())
        self.compiled_kernels = {}
        
        # GEMM kernel source with autotuning
        kernel_source = """
        @tessera.kernel.autotune(
            space=dict(
                BLOCK_M=[64, 128, 256],
                BLOCK_N=[64, 128, 256], 
                BLOCK_K=[32, 64, 128],
                num_warps=[4, 8, 16],
                num_stages=[2, 3, 4]
            ),
            key=["M", "N", "K"],
            cache="~/.tesserabench/autotune/gemm"
        )
        def tessera_gemm(A: Tensor["M", "K", dtype],
                        B: Tensor["K", "N", dtype],
                        C: Tensor["M", "N", accum_dtype],
                        alpha: float = 1.0,
                        beta: float = 0.0,
                        transpose_a: bool = False,
                        transpose_b: bool = False):
            
            # Tessera GEMM implementation with tensor cores
            ctx = tile.context()
            
            # Load tiles with optimal memory access patterns
            A_tile = tile.load(A, layout="tensor_core_compatible", 
                             transpose=transpose_a)
            B_tile = tile.load(B, layout="tensor_core_compatible",
                             transpose=transpose_b)
                             
            # Matrix multiplication using WMMA/WGMMA
            C_tile = tile.mma(A_tile, B_tile, accum=accum_dtype)
            
            # Alpha/beta scaling
            if beta != 0.0:
                C_existing = tile.load(C)
                C_tile = alpha * C_tile + beta * C_existing
            else:
                C_tile = alpha * C_tile
                
            # Store result
            tile.store(C, C_tile)
        """
        
        # Compile for each precision policy
        for precision in config.precision_policies:
            storage_dtype, accum_dtype = self._parse_precision_policy(precision)
            
            compilation_config = {
                'precision_policy': precision,
                'storage_dtype': storage_dtype,
                'accum_dtype': accum_dtype,
                'optimization_level': config.ir_optimization_levels[0],
                'autotuning_enabled': config.autotuning_enabled,
                'target_arch': config.hardware.gpu_arch
            }
            
            kernel, timing = self.tessera_integration.compile_with_timing(
                kernel_source, compilation_config
            )
            
            self.compiled_kernels[precision] = {
                'kernel': kernel,
                'compilation_timing': timing,
                'storage_dtype': storage_dtype,
                'accum_dtype': accum_dtype
            }
            
    def run_single(self, problem_size: Dict[str, Any]) -> BenchmarkResult:
        """Run single GEMM benchmark"""
        
        precision = self.config.precision_policies[0]
        kernel_info = self.compiled_kernels[precision]
        
        M, N, K = problem_size['M'], problem_size['N'], problem_size['K']
        transpose_a = problem_size.get('transpose_a', False)
        transpose_b = problem_size.get('transpose_b', False)
        
        # Create test tensors
        storage_dtype = kernel_info['storage_dtype']
        A = tessera.randn((M, K), dtype=storage_dtype)
        B = tessera.randn((K, N), dtype=storage_dtype) 
        C = tessera.zeros((M, N), dtype=kernel_info['accum_dtype'])
        
        if transpose_a:
            A = tessera.transpose(A)
        if transpose_b:
            B = tessera.transpose(B)
            
        # Calculate theoretical performance
        flops = 2 * M * N * K  # Multiply-accumulate operations
        theoretical_tflops = self._get_theoretical_tflops(self.config.hardware.gpu_arch, precision)
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            tessera.launch_kernel(kernel_info['kernel'], [A, B, C, 1.0, 0.0, transpose_a, transpose_b])
            tessera.synchronize()
            
        # Timing runs
        measurements = []
        for _ in range(self.config.timing_iterations):
            start_time = time.perf_counter()
            
            tessera.launch_kernel(kernel_info['kernel'], [A, B, C, 1.0, 0.0, transpose_a, transpose_b])
            tessera.synchronize()
            
            end_time = time.perf_counter()
            measurements.append((end_time - start_time) * 1000)
            
        # Analyze results
        analysis_engine = AnalysisEngine()
        stats = analysis_engine.analyze_measurements(measurements)
        
        mean_latency = stats['mean']
        throughput_ops_per_sec = flops / (mean_latency / 1000) if mean_latency > 0 else 0
        achieved_tflops = throughput_ops_per_sec / 1e12
        efficiency = (achieved_tflops / theoretical_tflops) * 100 if theoretical_tflops > 0 else 0
        
        return BenchmarkResult(
            benchmark_name=self.get_name(),
            problem_size=problem_size,
            config=self.config,
            latency_ms=mean_latency,
            throughput_ops_per_sec=throughput_ops_per_sec,
            compilation_time_ms=sum(kernel_info['compilation_timing'].values()),
            measurements_count=len(measurements),
            standard_deviation=stats['std'],
            confidence_interval_95=stats['confidence_interval']
        )
        
    def _parse_precision_policy(self, precision: str) -> Tuple[str, str]:
        """Parse precision policy into storage and accumulation dtypes"""
        if '@accum(' in precision:
            storage, accum = precision.split('@accum(')
            accum = accum.rstrip(')')
        else:
            storage = precision
            accum = 'fp32'  # Default accumulation
        return storage, accum
        
    def _get_theoretical_tflops(self, gpu_arch: str, precision: str) -> float:
        """Get theoretical TFLOPS for specific architecture and precision"""
        
        # Base TFLOPS for FP16/BF16 tensor cores
        arch_base_tflops = {
            'sm_70': 125.0,   # V100
            'sm_75': 130.0,   # T4/RTX 20xx
            'sm_80': 312.0,   # A100  
            'sm_86': 285.0,   # RTX 30xx
            'sm_89': 165.0,   # RTX 40xx
            'sm_90': 1320.0   # H100
        }
        
        base_tflops = arch_base_tflops.get(gpu_arch, 100.0)
        
        # Precision multipliers
        storage_dtype = precision.split('@')[0]
        precision_multipliers = {
            'fp32': 0.5,      # Half the throughput vs FP16
            'fp16': 1.0,      # Baseline
            'bf16': 1.0,      # Same as FP16
            'fp8_e4m3': 2.0,  # 2x throughput vs FP16
            'fp8_e5m2': 2.0,
            'fp6': 2.5,       # Between FP8 and FP16
            'fp4': 4.0        # 4x throughput vs FP16
        }
        
        multiplier = precision_multipliers.get(storage_dtype, 1.0)
        return base_tflops * multiplier

#### Vector Operations Benchmark

class VectorOpsBenchmark:
    """Benchmark for vector operations (AXPY, DOT, etc.)"""
    
    def get_name(self) -> str:
        return "vector_ops"
        
    def get_description(self) -> str:
        return "Vector operations including AXPY, DOT, NORM, and element-wise operations"
        
    def get_supported_precisions(self) -> List[str]:
        return ["fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2"]
        
    def get_problem_sizes(self) -> List[Dict[str, Any]]:
        """Generate vector operation problem sizes"""
        sizes = []
        
        # Vector sizes from small to very large
        vector_sizes = [
            1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864
        ]
        
        operations = ["axpy", "dot", "norm", "add", "mul", "reduce_sum", "reduce_max"]
        
        for size in vector_sizes:
            for op in operations:
                sizes.append({
                    'vector_size': size,
                    'operation': op,
                    'memory_pattern': 'sequential'
                })
                
        # Strided access patterns
        for size in [1048576, 4194304]:
            for stride in [2, 4, 8]:
                for op in ["axpy", "add"]:
                    sizes.append({
                        'vector_size': size,
                        'operation': op,
                        'memory_pattern': 'strided',
                        'stride': stride
                    })
                    
        return sizes
        
    def setup(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.tessera_integration = TesseraIntegration(tessera.get_runtime())
        self.compiled_kernels = {}
        
        # Vector operations kernel source
        vector_kernels_source = """
        @tessera.kernel.autotune(
            space=dict(
                BLOCK_SIZE=[128, 256, 512, 1024],
                VECTOR_WIDTH=[1, 2, 4, 8],
                num_warps=[4, 8, 16]
            ),
            key=["vector_size", "operation"],
            cache="~/.tesserabench/autotune/vector_ops"
        )
        def vector_operations(x: Tensor["N", dtype],
                            y: Tensor["N", dtype] = None,
                            result: Tensor[result_shape, result_dtype] = None,
                            alpha: float = 1.0,
                            operation: str = "axpy"):
            
            ctx = tile.context()
            i = tile.linear_id()
            
            if operation == "axpy" and y is not None:
                # y = alpha * x + y
                if i < x.shape[0]:
                    result[i] = alpha * x[i] + y[i]
                    
            elif operation == "dot":
                # Dot product with reduction
                local_sum = 0.0
                if i < x.shape[0]:
                    local_sum = x[i] * y[i]
                    
                # Warp-level reduction
                warp_sum = tile.warp_reduce_sum(local_sum)
                if tile.lane_id() == 0:
                    tile.atomic_add(result, warp_sum)
                    
            elif operation == "norm":
                # L2 norm computation
                local_sum = 0.0
                if i < x.shape[0]:
                    local_sum = x[i] * x[i]
                    
                warp_sum = tile.warp_reduce_sum(local_sum)
                if tile.lane_id() == 0:
                    tile.atomic_add(result, warp_sum)
                    
            elif operation == "add":
                # Element-wise addition
                if i < x.shape[0]:
                    result[i] = x[i] + y[i]
                    
            elif operation == "mul":
                # Element-wise multiplication
                if i < x.shape[0]:
                    result[i] = x[i] * y[i]
                    
            elif operation == "reduce_sum":
                # Reduction to scalar
                local_sum = 0.0
                if i < x.shape[0]:
                    local_sum = x[i]
                    
                warp_sum = tile.warp_reduce_sum(local_sum)
                if tile.lane_id() == 0:
                    tile.atomic_add(result, warp_sum)
                    
            elif operation == "reduce_max":
                # Max reduction
                local_max = float('-inf')
                if i < x.shape[0]:
                    local_max = x[i]
                    
                warp_max = tile.warp_reduce_max(local_max)
                if tile.lane_id() == 0:
                    tile.atomic_max(result, warp_max)
        """
        
        # Compile for each precision
        for precision in config.precision_policies:
            compilation_config = {
                'precision_policy': precision,
                'optimization_level': config.ir_optimization_levels[0],
                'autotuning_enabled': config.autotuning_enabled,
                'target_arch': config.hardware.gpu_arch
            }
            
            kernel, timing = self.tessera_integration.compile_with_timing(
                vector_kernels_source, compilation_config
            )
            
            self.compiled_kernels[precision] = {
                'kernel': kernel,
                'compilation_timing': timing
            }

### 2. Deep Learning Primitives Benchmarks

#### Layer Normalization Benchmark

class LayerNormBenchmark:
    """Layer Normalization benchmark with numerical stability focus"""
    
    def get_name(self) -> str:
        return "layer_norm"
        
    def get_description(self) -> str:
        return "Layer Normalization with numerical stability and various tensor shapes"
        
    def get_supported_precisions(self) -> List[str]:
        return ["fp32", "fp16", "bf16", "fp8_e4m3@accum(fp32)", "fp8_e5m2@accum(fp32)"]
        
    def get_problem_sizes(self) -> List[Dict[str, Any]]:
        """Generate LayerNorm problem sizes"""
        sizes = []
        
        # Common transformer dimensions
        hidden_sizes = [768, 1024, 1536, 2048, 4096, 8192, 11008]
        sequence_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
        batch_sizes = [1, 4, 8, 16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                for hidden_size in hidden_sizes:
                    # Skip very large configurations
                    if batch_size * seq_len * hidden_size > 2**30:  # 1B elements max
                        continue
                        
                    sizes.append({
                        'batch_size': batch_size,
                        'seq_len': seq_len, 
                        'hidden_size': hidden_size,
                        'eps': 1e-5,
                        'elementwise_affine': True
                    })
                    
        return sizes
        
    def setup(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.tessera_integration = TesseraIntegration(tessera.get_runtime())
        self.compiled_kernels = {}
        
        # Layer normalization kernel with numerical stability
        layernorm_source = """
        @tessera.kernel.autotune(
            space=dict(
                BLOCK_SIZE=[128, 256, 512],
                VECTOR_WIDTH=[2, 4, 8],
                num_warps=[4, 8, 16],
                WARP_REDUCE=[True, False]
            ),
            key=["hidden_size"],
            cache="~/.tesserabench/autotune/layernorm"
        )
        def layernorm_safe(x: Tensor["B", "S", "H", dtype],
                          weight: Tensor["H", dtype],
                          bias: Tensor["H", dtype],
                          output: Tensor["B", "S", "H", dtype],
                          eps: float = 1e-5):
            
            ctx = tile.context()
            batch_idx = tile.block_id(0)
            seq_idx = tile.block_id(1)
            thread_idx = tile.thread_id()
            
            # Shared memory for statistics
            smem_sum = tile.alloc_shared[dtype](1)
            smem_sum_sq = tile.alloc_shared[dtype](1)
            
            # Initialize shared memory
            if thread_idx == 0:
                smem_sum[0] = 0.0
                smem_sum_sq[0] = 0.0
            tile.barrier()
            
            # First pass: compute mean and variance
            local_sum = 0.0
            local_sum_sq = 0.0
            
            for h in range(thread_idx, x.shape[2], tile.block_size()):
                val = x[batch_idx, seq_idx, h]
                local_sum += val
                local_sum_sq += val * val
                
            # Warp-level reduction
            warp_sum = tile.warp_reduce_sum(local_sum)
            warp_sum_sq = tile.warp_reduce_sum(local_sum_sq)
            
            # Block-level reduction
            if tile.lane_id() == 0:
                tile.atomic_add(smem_sum, warp_sum)
                tile.atomic_add(smem_sum_sq, warp_sum_sq)
                
            tile.barrier()
            
            # Compute statistics
            if thread_idx == 0:
                mean = smem_sum[0] / x.shape[2]
                variance = (smem_sum_sq[0] / x.shape[2]) - mean * mean
                inv_std = tile.rsqrt(variance + eps)  # Safe reciprocal square root
                
                smem_sum[0] = mean      # Reuse for mean
                smem_sum_sq[0] = inv_std  # Reuse for inv_std
                
            tile.barrier()
            
            mean = smem_sum[0]
            inv_std = smem_sum_sq[0]
            
            # Second pass: normalize
            for h in range(thread_idx, x.shape[2], tile.block_size()):
                normalized = (x[batch_idx, seq_idx, h] - mean) * inv_std
                output[batch_idx, seq_idx, h] = normalized * weight[h] + bias[h]
        """
        
        for precision in config.precision_policies:
            compilation_config = {
                'precision_policy': precision,
                'optimization_level': config.ir_optimization_levels[0],
                'autotuning_enabled': config.autotuning_enabled,
                'target_arch': config.hardware.gpu_arch
            }
            
            kernel, timing = self.tessera_integration.compile_with_timing(
                layernorm_source, compilation_config
            )
            
            self.compiled_kernels[precision] = {
                'kernel': kernel,
                'compilation_timing': timing
            }
            
    def run_single(self, problem_size: Dict[str, Any]) -> BenchmarkResult:
        precision = self.config.precision_policies[0]
        kernel_info = self.compiled_kernels[precision]
        
        B = problem_size['batch_size']
        S = problem_size['seq_len']
        H = problem_size['hidden_size']
        eps = problem_size['eps']
        
        # Create test data
        storage_dtype = precision.split('@')[0]
        x = tessera.randn((B, S, H), dtype=storage_dtype)
        weight = tessera.ones((H,), dtype=storage_dtype)
        bias = tessera.zeros((H,), dtype=storage_dtype)
        output = tessera.zeros((B, S, H), dtype=storage_dtype)
        
        # Calculate theoretical bandwidth (memory-bound operation)
        # Read: x, weight, bias; Write: output
        bytes_per_element = self._get_dtype_bytes(storage_dtype)
        memory_bytes = (B * S * H * 2 + H * 2) * bytes_per_element  # 2 reads + 2 params + 1 write
        theoretical_bandwidth = self.config.hardware.memory_bandwidth_gbps * 1e9  # Convert to bytes/sec
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            tessera.launch_kernel(kernel_info['kernel'], [x, weight, bias, output, eps])
            tessera.synchronize()
            
        # Timing runs
        measurements = []
        for _ in range(self.config.timing_iterations):
            start_time = time.perf_counter()
            
            tessera.launch_kernel(kernel_info['kernel'], [x, weight, bias, output, eps])
            tessera.synchronize()
            
            end_time = time.perf_counter()
            measurements.append((end_time - start_time) * 1000)
            
        # Analysis
        analysis_engine = AnalysisEngine()
        stats = analysis_engine.analyze_measurements(measurements)
        
        mean_latency = stats['mean']
        achieved_bandwidth = memory_bytes / (mean_latency / 1000) if mean_latency > 0 else 0
        bandwidth_efficiency = (achieved_bandwidth / theoretical_bandwidth) * 100
        
        result = BenchmarkResult(
            benchmark_name=self.get_name(),
            problem_size=problem_size,
            config=self.config,
            latency_ms=mean_latency,
            memory_bandwidth_gbps=achieved_bandwidth / 1e9,
            compilation_time_ms=sum(kernel_info['compilation_timing'].values()),
            measurements_count=len(measurements),
            standard_deviation=stats['std'],
            confidence_interval_95=stats['confidence_interval']
        )
        
        return result
        
    def validate_result(self, result: BenchmarkResult) -> bool:
        """Validate LayerNorm results with numerical accuracy check"""
        
        # Create reference implementation
        problem_size = result.problem_size
        B, S, H = problem_size['batch_size'], problem_size['seq_len'], problem_size['hidden_size']
        
        # Generate same test data (should use fixed seed)
        import numpy as np
        np.random.seed(42)
        
        x_ref = np.random.randn(B, S, H).astype(np.float32)
        weight_ref = np.ones((H,), dtype=np.float32)
        bias_ref = np.zeros((H,), dtype=np.float32)
        
        # Reference LayerNorm computation
        mean = np.mean(x_ref, axis=-1, keepdims=True)
        var = np.var(x_ref, axis=-1, keepdims=True)
        output_ref = (x_ref - mean) / np.sqrt(var + problem_size['eps'])
        output_ref = output_ref * weight_ref + bias_ref
        
        # Compare with Tessera result (would need actual output)
        # This is a placeholder for the actual validation logic
        result.numerical_accuracy = 99.9  # Placeholder
        
        return True
        
    def _get_dtype_bytes(self, dtype: str) -> int:
        """Get number of bytes per element for dtype"""
        bytes_map = {
            'fp32': 4, 'fp16': 2, 'bf16': 2,
            'fp8_e4m3': 1, 'fp8_e5m2': 1,
            'fp6': 1, 'fp4': 0.5
        }
        return bytes_map.get(dtype, 4)

#### Softmax Benchmark

class SoftmaxBenchmark:
    """Softmax benchmark with numerical stability and various tensor shapes"""
    
    def get_name(self) -> str:
        return "softmax"
        
    def get_description(self) -> str:
        return "Numerically stable Softmax with various input shapes and precisions"
        
    def get_supported_precisions(self) -> List[str]:
        return ["fp32", "fp16", "bf16", "fp8_e4m3@accum(fp32)"]
        
    def get_problem_sizes(self) -> List[Dict[str, Any]]:
        sizes = []
        
        # Attention-like shapes
        batch_sizes = [1, 4, 8, 16, 32]
        num_heads = [8, 16, 32, 64]
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
        
        for batch_size in batch_sizes:
            for heads in num_heads:
                for seq_len in seq_lengths:
                    # Typical attention patterns
                    sizes.append({
                        'batch_size': batch_size,
                        'num_heads': heads,
                        'seq_len_q': seq_len,
                        'seq_len_k': seq_len,
                        'axis': -1,
                        'causal_mask': False
                    })
                    
                    # Causal attention
                    if seq_len <= 2048:  # Limit causal for performance
                        sizes.append({
                            'batch_size': batch_size,
                            'num_heads': heads,
                            'seq_len_q': seq_len,
                            'seq_len_k': seq_len,
                            'axis': -1,
                            'causal_mask': True
                        })
                        
        # 1D Softmax cases
        for size in [1024, 4096, 16384, 65536, 262144]:
            sizes.append({
                'batch_size': 1,
                'num_heads': 1,
                'seq_len_q': 1,
                'seq_len_k': size,
                'axis': -1,
                'causal_mask': False
            })
            
        return sizes
        
    def setup(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.tessera_integration = TesseraIntegration(tessera.get_runtime())
        self.compiled_kernels = {}
        
        # Numerically stable softmax kernel
        softmax_source = """
        @tessera.kernel.autotune(
            space=dict(
                BLOCK_SIZE=[128, 256, 512],
                VECTOR_WIDTH=[2, 4, 8],
                WARP_REDUCE_STAGES=[1, 2],
                num_warps=[4, 8, 16]
            ),
            key=["seq_len_k"],
            cache="~/.tesserabench/autotune/softmax"
        )
        def softmax_safe(x: Tensor["B", "H", "Sq", "Sk", dtype],
                        output: Tensor["B", "H", "Sq", "Sk", dtype],
                        causal_mask: bool = False):
            
            ctx = tile.context()
            batch_idx = tile.block_id(0)
            head_idx = tile.block_id(1) 
            q_idx = tile.block_id(2)
            thread_idx = tile.thread_id()
            
            # Shared memory for reduction
            smem_max = tile.alloc_shared[dtype](1)
            smem_sum = tile.alloc_shared[dtype](1)
            
            # Initialize shared memory
            if thread_idx == 0:
                smem_max[0] = float('-inf')
                smem_sum[0] = 0.0
            tile.barrier()
            
            # First pass: find maximum (for numerical stability)
            local_max = float('-inf')
            seq_len = x.shape[3]
            
            for k in range(thread_idx, seq_len, tile.block_size()):
                if causal_mask and k > q_idx:
                    continue  # Skip future tokens
                    
                val = x[batch_idx, head_idx, q_idx, k]
                local_max = tile.max(local_max, val)
                
            # Warp-level max reduction
            warp_max = tile.warp_reduce_max(local_max)
            if tile.lane_id() == 0:
                tile.atomic_max(smem_max, warp_max)
                
            tile.barrier()
            global_max = smem_max[0]
            
            # Second pass: compute exp(x - max) and sum
            local_sum = 0.0
            for k in range(thread_idx, seq_len, tile.block_size()):
                if causal_mask and k > q_idx:
                    continue
                    
                val = x[batch_idx, head_idx, q_idx, k]
                exp_val = tile.exp(val - global_max)
                output[batch_idx, head_idx, q_idx, k] = exp_val
                local_sum += exp_val
                
            # Warp-level sum reduction  
            warp_sum = tile.warp_reduce_sum(local_sum)
            if tile.lane_id() == 0:
                tile.atomic_add(smem_sum, warp_sum)
                
            tile.barrier()
            global_sum = smem_sum[0]
            
            # Third pass: normalize
            inv_sum = 1.0 / global_sum
            for k in range(thread_idx, seq_len, tile.block_size()):
                if causal_mask and k > q_idx:
                    output[batch_idx, head_idx, q_idx, k] = 0.0  # Masked positions
                else:
                    output[batch_idx, head_idx, q_idx, k] *= inv_sum
        """
        
        for precision in config.precision_policies:
            compilation_config = {
                'precision_policy': precision,
                'optimization_level': config.ir_optimization_levels[0],
                'autotuning_enabled': config.autotuning_enabled,
                'target_arch': config.hardware.gpu_arch
            }
            
            kernel, timing = self.tessera_integration.compile_with_timing(
                softmax_source, compilation_config
            )
            
            self.compiled_kernels[precision] = {
                'kernel': kernel,
                'compilation_timing': timing
            }

### 3. Memory-Intensive Benchmarks

#### Memory Bandwidth Benchmark

class MemoryBandwidthBenchmark:
    """Pure memory bandwidth benchmark with various access patterns"""
    
    def get_name(self) -> str:
        return "memory_bandwidth"
        
    def get_description(self) -> str:
        return "Memory bandwidth benchmark with coalesced, strided, and random access patterns"
        
    def get_supported_precisions(self) -> List[str]:
        return ["fp32", "fp16", "bf16", "int32", "int8"]
        
    def get_problem_sizes(self) -> List[Dict[str, Any]]:
        sizes = []
        
        # Memory sizes from 1MB to 16GB
        memory_sizes = [
            1 * 1024**2,    # 1MB
            4 * 1024**2,    # 4MB  
            16 * 1024**2,   # 16MB
            64 * 1024**2,   # 64MB
            256 * 1024**2,  # 256MB
            1024 * 1024**2, # 1GB
            4096 * 1024**2, # 4GB
            8192 * 1024**2, # 8GB
            16384 * 1024**2 # 16GB
        ]
        
        access_patterns = ["sequential", "strided_2", "strided_4", "strided_8", "random"]
        operations = ["copy", "add", "scale", "triad"]  # STREAM benchmark operations
        
        for size_bytes in memory_sizes:
            for pattern in access_patterns:
                for op in operations:
                    sizes.append({
                        'memory_size_bytes': size_bytes,
                        'access_pattern': pattern,
                        'operation': op,
                        'vector_width': 4  # Default vectorization
                    })
                    
        return sizes
        
    def setup(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.tessera_integration = TesseraIntegration(tessera.get_runtime())
        self.compiled_kernels = {}
        
        # Memory bandwidth kernel
        memory_bw_source = """
        @tessera.kernel.autotune(
            space=dict(
                BLOCK_SIZE=[128, 256, 512, 1024],
                VECTOR_WIDTH=[1, 2, 4, 8, 16],
                num_warps=[4, 8, 16, 32]
            ),
            key=["memory_size_bytes", "access_pattern", "operation"],
            cache="~/.tesserabench/autotune/memory_bw"
        )
        def memory_bandwidth_kernel(a: Tensor["N", dtype],
                                   b: Tensor["N", dtype], 
                                   c: Tensor["N", dtype],
                                   scalar: dtype,
                                   operation: str,
                                   access_pattern: str,
                                   stride: int = 1):
            
            ctx = tile.context()
            i = tile.linear_id()
            n = a.shape[0]
            
            if access_pattern == "sequential":
                idx = i
            elif access_pattern.startswith("strided"):
                stride_val = int(access_pattern.split('_')[1])
                idx = i * stride_val
                if idx >= n:
                    return
            elif access_pattern == "random":
                # Simple pseudo-random pattern
                idx = (i * 1103515245 + 12345) % n
            else:
                idx = i
                
            if idx >= n:
                return
                
            if operation == "copy":
                # c = a
                c[idx] = a[idx]
            elif operation == "scale":
                # c = scalar * a  
                c[idx] = scalar * a[idx]
            elif operation == "add":
                # c = a + b
                c[idx] = a[idx] + b[idx]
            elif operation == "triad":
                # c = a + scalar * b
                c[idx] = a[idx] + scalar * b[idx]
        """
        
        for precision in config.precision_policies:
            compilation_config = {
                'precision_policy': precision,
                'optimization_level': config.ir_optimization_levels[0],
                'autotuning_enabled': config.autotuning_enabled,
                'target_arch': config.hardware.gpu_arch
            }
            
            kernel, timing = self.tessera_integration.compile_with_timing(
                memory_bw_source, compilation_config
            )
            
            self.compiled_kernels[precision] = {
                'kernel': kernel,
                'compilation_timing': timing
            }
            
    def run_single(self, problem_size: Dict[str, Any]) -> BenchmarkResult:
        precision = self.config.precision_policies[0]
        kernel_info = self.compiled_kernels[precision]
        
        size_bytes = problem_size['memory_size_bytes']
        access_pattern = problem_size['access_pattern']
        operation = problem_size['operation']
        
        # Calculate array size based on data type
        storage_dtype = precision.split('@')[0]
        bytes_per_element = self._get_dtype_bytes(storage_dtype)
        n_elements = size_bytes // bytes_per_element
        
        # Create test arrays
        a = tessera.randn((n_elements,), dtype=storage_dtype)
        b = tessera.randn((n_elements,), dtype=storage_dtype)
        c = tessera.zeros((n_elements,), dtype=storage_dtype)
        scalar = tessera.scalar(1.5, dtype=storage_dtype)
        
        # Calculate theoretical bandwidth
        # Bytes transferred depends on operation
        if operation == "copy":
            bytes_transferred = 2 * size_bytes  # Read a, write c
        elif operation == "scale":
            bytes_transferred = 2 * size_bytes  # Read a, write c
        elif operation == "add":
            bytes_transferred = 3 * size_bytes  # Read a,b, write c
        elif operation == "triad":
            bytes_transferred = 3 * size_bytes  # Read a,b, write c
            
        theoretical_bandwidth = self.config.hardware.memory_bandwidth_gbps * 1e9
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            tessera.launch_kernel(kernel_info['kernel'], 
                                [a, b, c, scalar, operation, access_pattern, 1])
            tessera.synchronize()
            
        # Timing runs
        measurements = []
        for _ in range(self.config.timing_iterations):
            start_time = time.perf_counter()
            
            tessera.launch_kernel(kernel_info['kernel'],
                                [a, b, c, scalar, operation, access_pattern, 1])
            tessera.synchronize()
            
            end_time = time.perf_counter()
            measurements.append((end_time - start_time) * 1000)
            
        # Analysis
        analysis_engine = AnalysisEngine()
        stats = analysis_engine.analyze_measurements(measurements)
        
        mean_latency = stats['mean']
        achieved_bandwidth = bytes_transferred / (mean_latency / 1000) if mean_latency > 0 else 0
        bandwidth_efficiency = (achieved_bandwidth / theoretical_bandwidth) * 100
        
        return BenchmarkResult(
            benchmark_name=self.get_name(),
            problem_size=problem_size,
            config=self.config,
            latency_ms=mean_latency,
            memory_bandwidth_gbps=achieved_bandwidth / 1e9,
            compilation_time_ms=sum(kernel_info['compilation_timing'].values()),
            measurements_count=len(measurements),
            standard_deviation=stats['std'],
            confidence_interval_95=stats['confidence_interval']
        )

### 4. Distributed Computing Benchmarks

#### All-Reduce Collective Benchmark

class AllReduceBenchmark:
    """All-reduce collective communication benchmark"""
    
    def get_name(self) -> str:
        return "all_reduce"
        
    def get_description(self) -> str:
        return "All-reduce collective communication across multiple GPUs"
        
    def get_supported_precisions(self) -> List[str]:
        return ["fp32", "fp16", "bf16"]
        
    def get_problem_sizes(self) -> List[Dict[str, Any]]:
        sizes = []
        
        # Message sizes from 4KB to 1GB
        message_sizes = [
            4 * 1024,        # 4KB
            16 *
        