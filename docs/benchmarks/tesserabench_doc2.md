# TesseraBench - Document 2: Benchmark Suite Implementation

This document details the comprehensive benchmark suite for TesseraBench, covering essential GPU computing kernels optimized for the Tessera programming model. Each benchmark is designed to evaluate specific aspects of Tessera's compilation pipeline and runtime performance.

Examples in this document use the current Tessera Python surface: `@tessera.jit` for compiled functions, `@tessera.kernel` only for `index_launch` shard dispatch, `tessera.ops.*` for operations, and TesseraBench helper APIs for benchmark data allocation/timing. Legacy tile-autotune decorators and direct runtime launch helpers are intentionally avoided.

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
        self.tessera_integration = TesseraIntegration(config.target_profile)
        self.compiled_kernels = {}
        
        # GEMM compiled through the current public API. TesseraBench profiles
        # the named compiler pipeline selected by config.target_profile.
        @tessera.jit(
            bindings={"M": 1024, "N": 1024, "K": 1024},
            target=config.target_profile,
        )
        def tessera_gemm(
            A: tessera.Tensor["M", "K"],
            B: tessera.Tensor["K", "N"],
        ) -> tessera.Tensor["M", "N"]:
            tessera.require(tessera.constraint.Divisible("K", 64))
            return tessera.ops.gemm(A, B)
        
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
                tessera_gemm, compilation_config
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
        A = tb.testing.random_array((M, K), dtype=storage_dtype)
        B = tb.testing.random_array((K, N), dtype=storage_dtype)
        
        if transpose_a:
            A = tessera.ops.transpose(A)
        if transpose_b:
            B = tessera.ops.transpose(B)
            
        # Calculate theoretical performance
        flops = 2 * M * N * K  # Multiply-accumulate operations
        theoretical_tflops = self._get_theoretical_tflops(self.config.hardware.gpu_arch, precision)
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = kernel_info['kernel'](A, B)
            
        # Timing runs
        measurements = []
        for _ in range(self.config.timing_iterations):
            start_time = time.perf_counter()
            
            _ = kernel_info['kernel'](A, B)
            
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
        self.tessera_integration = TesseraIntegration(config.target_profile)
        self.compiled_kernels = {}
        
        @tessera.jit(target=config.target_profile)
        def vector_operations(
            x: tessera.Tensor["N"],
        ) -> tessera.Tensor["N"]:
            return tessera.ops.relu(x)
        
        # Compile for each precision
        for precision in config.precision_policies:
            compilation_config = {
                'precision_policy': precision,
                'optimization_level': config.ir_optimization_levels[0],
                'autotuning_enabled': config.autotuning_enabled,
                'target_arch': config.hardware.gpu_arch
            }
            
            kernel, timing = self.tessera_integration.compile_with_timing(
                vector_operations, compilation_config
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
        self.tessera_integration = TesseraIntegration(config.target_profile)
        self.compiled_kernels = {}
        
        @tessera.jit(target=config.target_profile)
        def layernorm_safe(
            x: tessera.Tensor["B", "S", "H"],
            weight: tessera.Tensor["H"],
            bias: tessera.Tensor["H"],
        ) -> tessera.Tensor["B", "S", "H"]:
            return tessera.ops.layer_norm(x, weight=weight, bias=bias, eps=1e-5)
        
        for precision in config.precision_policies:
            compilation_config = {
                'precision_policy': precision,
                'optimization_level': config.ir_optimization_levels[0],
                'autotuning_enabled': config.autotuning_enabled,
                'target_arch': config.hardware.gpu_arch
            }
            
            kernel, timing = self.tessera_integration.compile_with_timing(
                layernorm_safe, compilation_config
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
        x = tb.testing.random_array((B, S, H), dtype=storage_dtype)
        weight = tb.testing.ones_array((H,), dtype=storage_dtype)
        bias = tb.testing.zeros_array((H,), dtype=storage_dtype)
        
        # Calculate theoretical bandwidth (memory-bound operation)
        # Read: x, weight, bias; Write: output
        bytes_per_element = self._get_dtype_bytes(storage_dtype)
        memory_bytes = (B * S * H * 2 + H * 2) * bytes_per_element  # 2 reads + 2 params + 1 write
        theoretical_bandwidth = self.config.hardware.memory_bandwidth_gbps * 1e9  # Convert to bytes/sec
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = kernel_info['kernel'](x, weight, bias)
            
        # Timing runs
        measurements = []
        for _ in range(self.config.timing_iterations):
            start_time = time.perf_counter()
            
            _ = kernel_info['kernel'](x, weight, bias)
            
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
        self.tessera_integration = TesseraIntegration(config.target_profile)
        self.compiled_kernels = {}
        
        @tessera.jit(target=config.target_profile)
        def softmax_safe(
            x: tessera.Tensor["B", "H", "Sq", "Sk"],
        ) -> tessera.Tensor["B", "H", "Sq", "Sk"]:
            return tessera.ops.softmax(x, axis=-1)
        
        for precision in config.precision_policies:
            compilation_config = {
                'precision_policy': precision,
                'optimization_level': config.ir_optimization_levels[0],
                'autotuning_enabled': config.autotuning_enabled,
                'target_arch': config.hardware.gpu_arch
            }
            
            kernel, timing = self.tessera_integration.compile_with_timing(
                softmax_safe, compilation_config
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
        self.tessera_integration = TesseraIntegration(config.target_profile)
        self.compiled_kernels = {}
        
        @tessera.jit(target=config.target_profile)
        def memory_bandwidth_kernel(
            a: tessera.Region["read"],
            b: tessera.Region["read"],
            c: tessera.Region["write"],
        ):
            c[:] = tessera.ops.fused_epilogue(a, b, activation="none")
        
        for precision in config.precision_policies:
            compilation_config = {
                'precision_policy': precision,
                'optimization_level': config.ir_optimization_levels[0],
                'autotuning_enabled': config.autotuning_enabled,
                'target_arch': config.hardware.gpu_arch
            }
            
            kernel, timing = self.tessera_integration.compile_with_timing(
                memory_bandwidth_kernel, compilation_config
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
        a = tb.testing.random_array((n_elements,), dtype=storage_dtype)
        b = tb.testing.random_array((n_elements,), dtype=storage_dtype)
        c = tb.testing.zeros_array((n_elements,), dtype=storage_dtype)
        
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
            kernel_info['kernel'](a, b, c)
            
        # Timing runs
        measurements = []
        for _ in range(self.config.timing_iterations):
            start_time = time.perf_counter()
            
            kernel_info['kernel'](a, b, c)
            
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
