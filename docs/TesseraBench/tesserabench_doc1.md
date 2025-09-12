# TesseraBench - Document 1: Architecture and Design

TesseraBench is a comprehensive benchmarking framework specifically designed for the Tessera programming model. Drawing inspiration from tritonbench, it provides systematic performance evaluation capabilities for GPU kernels compiled through Tessera's multi-level IR stack, from Graph IR to Target IR.

## Overview and Motivation

### Why TesseraBench?

The Tessera programming model introduces unique challenges and opportunities for performance benchmarking:

- **Multi-Level IR Stack**: Performance characteristics vary across Graph IR → Schedule IR → Tile IR → Target IR transformations
- **Architecture-Specific Optimization**: Different backends (PTX, CUDA Tile IR) require specialized benchmarking
- **Numerical Precision Policies**: FP4/FP6/FP8/BF16/FP16/FP32 combinations need systematic evaluation
- **Distributed Execution**: Multi-GPU meshes with TP/DP/PP parallelism require distributed benchmarking
- **Autotuning Integration**: Performance-guided optimization requires robust measurement infrastructure

### Key Design Principles

1. **Comprehensive Coverage**: Benchmark across all IR levels and optimization passes
2. **Architecture Awareness**: Adapt measurements to specific GPU architectures and features
3. **Reproducible Results**: Eliminate variance through statistical rigor and environmental control
4. **Scalable Execution**: Support single-GPU to NVL72-scale distributed benchmarking
5. **Integration Ready**: Seamlessly integrate with Tessera's compilation and runtime systems

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TesseraBench Core                            │
├─────────────────┬───────────────────┬───────────────────┬───────────┤
│   Benchmark     │    Measurement    │    Analysis       │  Reporting│
│   Definition    │     Engine        │     Engine        │  System   │
│                 │                   │                   │           │
│ • Kernel Specs  │ • Timer Systems   │ • Statistical     │ • Results │
│ • Test Cases    │ • Resource Mon.   │   Analysis        │   Export  │
│ • Parameters    │ • Hardware Prof.  │ • Comparison      │ • Visual. │
│ • Validation    │ • Distributed     │ • Regression      │ • CI/CD   │
│                 │   Coordination    │   Detection       │   Integ.  │
└─────────────────┼───────────────────┼───────────────────┼───────────┘
                  │                   │                   │
                  ↓                   ↓                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     Tessera Integration Layer                       │
├─────────────────┬───────────────────┬───────────────────┬───────────┤
│   Compilation   │     Runtime       │   Multi-GPU       │ IR Insp.  │
│   Interface     │    Interface      │   Coordination    │ Tools     │
│                 │                   │                   │           │
│ • Graph IR      │ • Kernel Launch   │ • NCCL Collects   │ • IR Dump │
│ • Schedule IR   │ • Memory Mgmt     │ • Mesh Config     │ • Optim.  │
│ • Tile IR       │ • Stream Mgmt     │ • Sync Barriers   │   Trace   │
│ • Target IR     │ • Error Handle    │ • Load Balance    │ • Debug   │
└─────────────────┴───────────────────┴───────────────────┴───────────┘
```

### Core Architecture Classes

```python
from typing import Protocol, Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import enum
import asyncio

class BenchmarkMode(enum.Enum):
    """Benchmark execution modes"""
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu" 
    DISTRIBUTED = "distributed"
    REGRESSION = "regression"
    COMPARISON = "comparison"
    PROFILING = "profiling"

class MeasurementType(enum.Enum):
    """Types of measurements to collect"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    COMPUTE_UTILIZATION = "compute_utilization"
    ENERGY = "energy"
    OCCUPANCY = "occupancy"
    ACCURACY = "accuracy"

@dataclass
class HardwareConfig:
    """Hardware configuration for benchmarking"""
    gpu_arch: str  # sm_70, sm_80, sm_90, etc.
    gpu_count: int = 1
    memory_size_gb: int = 80
    memory_bandwidth_gbps: float = 3352.0  # H100 HBM3e
    compute_capability: str = "9.0"
    tensor_cores: bool = True
    nvlink_bandwidth_gbps: float = 900.0
    nvswitch_bandwidth_gbps: float = 1800.0
    
@dataclass 
class BenchmarkConfig:
    """Configuration for a benchmark run"""
    name: str
    description: str
    mode: BenchmarkMode
    measurements: List[MeasurementType]
    hardware: HardwareConfig
    
    # Execution parameters
    warmup_iterations: int = 5
    timing_iterations: int = 100
    statistical_significance: float = 0.95
    max_variance_percent: float = 5.0
    
    # Tessera-specific parameters
    precision_policies: List[str] = field(default_factory=lambda: ["bf16@accum(fp32)"])
    ir_optimization_levels: List[str] = field(default_factory=lambda: ["O3"])
    autotuning_enabled: bool = True
    
    # Multi-GPU parameters
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
class BenchmarkProtocol(Protocol):
    """Protocol that all benchmarks must implement"""
    
    def get_name(self) -> str:
        """Return benchmark name"""
        ...
    
    def get_description(self) -> str:
        """Return benchmark description"""
        ...
        
    def get_supported_precisions(self) -> List[str]:
        """Return supported precision policies"""
        ...
        
    def get_problem_sizes(self) -> List[Dict[str, Any]]:
        """Return list of problem sizes to benchmark"""
        ...
        
    def setup(self, config: BenchmarkConfig) -> None:
        """Setup benchmark with given configuration"""
        ...
        
    def run_single(self, problem_size: Dict[str, Any]) -> 'BenchmarkResult':
        """Run benchmark for single problem size"""
        ...
        
    def cleanup(self) -> None:
        """Clean up benchmark resources"""
        ...
        
    def validate_result(self, result: 'BenchmarkResult') -> bool:
        """Validate benchmark result correctness"""
        ...

class TesseraBenchCore:
    """Core benchmarking engine"""
    
    def __init__(self, tessera_runtime: 'TesseraRuntime'):
        self.runtime = tessera_runtime
        self.measurement_engine = MeasurementEngine()
        self.analysis_engine = AnalysisEngine()
        self.registered_benchmarks: Dict[str, BenchmarkProtocol] = {}
        
    def register_benchmark(self, benchmark: BenchmarkProtocol) -> None:
        """Register a new benchmark"""
        name = benchmark.get_name()
        if name in self.registered_benchmarks:
            raise ValueError(f"Benchmark {name} already registered")
        self.registered_benchmarks[name] = benchmark
        
    def run_benchmark(self, 
                     benchmark_name: str,
                     config: BenchmarkConfig) -> 'BenchmarkSuiteResult':
        """Run a specific benchmark with given configuration"""
        if benchmark_name not in self.registered_benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
            
        benchmark = self.registered_benchmarks[benchmark_name]
        return self._execute_benchmark(benchmark, config)
        
    def run_benchmark_suite(self,
                          benchmark_names: List[str],
                          config: BenchmarkConfig) -> 'BenchmarkSuiteResult':
        """Run multiple benchmarks as a suite"""
        results = {}
        
        for name in benchmark_names:
            try:
                result = self.run_benchmark(name, config)
                results[name] = result
            except Exception as e:
                results[name] = BenchmarkError(str(e))
                
        return BenchmarkSuiteResult(results)
        
    def _execute_benchmark(self, 
                          benchmark: BenchmarkProtocol,
                          config: BenchmarkConfig) -> 'BenchmarkSuiteResult':
        """Execute a single benchmark"""
        benchmark.setup(config)
        
        try:
            problem_sizes = benchmark.get_problem_sizes()
            results = []
            
            for problem_size in problem_sizes:
                # Run with statistical significance testing
                result = self._run_with_statistics(benchmark, problem_size, config)
                
                # Validate result correctness
                if not benchmark.validate_result(result):
                    result.add_warning("Result validation failed")
                    
                results.append(result)
                
            return BenchmarkSuiteResult(results)
            
        finally:
            benchmark.cleanup()
```

### Measurement Engine Design

```python
import time
import threading
from contextlib import contextmanager
from collections import defaultdict

class MeasurementEngine:
    """High-precision measurement engine for Tessera kernels"""
    
    def __init__(self):
        self.active_measurements = defaultdict(list)
        self.hardware_profiler = HardwareProfiler()
        self.distributed_coordinator = DistributedCoordinator()
        
    @contextmanager
    def measure_execution(self, 
                         measurement_types: List[MeasurementType],
                         context: Dict[str, Any]):
        """Context manager for comprehensive measurements"""
        
        measurement_id = f"measure_{int(time.time() * 1e9)}"
        measurements = {}
        
        try:
            # Start all measurements
            for mtype in measurement_types:
                if mtype == MeasurementType.LATENCY:
                    measurements[mtype] = self._start_latency_measurement()
                elif mtype == MeasurementType.MEMORY_BANDWIDTH:
                    measurements[mtype] = self._start_bandwidth_measurement()
                elif mtype == MeasurementType.COMPUTE_UTILIZATION:
                    measurements[mtype] = self._start_utilization_measurement()
                elif mtype == MeasurementType.ENERGY:
                    measurements[mtype] = self._start_energy_measurement()
                    
            yield measurements
            
        finally:
            # Stop all measurements and collect results
            results = {}
            for mtype, measurement in measurements.items():
                results[mtype] = self._stop_measurement(mtype, measurement)
                
            self.active_measurements[measurement_id] = results
            
    def _start_latency_measurement(self) -> Dict[str, Any]:
        """Start high-precision latency measurement"""
        import cuda
        
        # Create CUDA events for precise GPU timing
        start_event = cuda.Event()
        end_event = cuda.Event()
        
        # Record CPU timing as backup
        cpu_start = time.perf_counter_ns()
        
        # Synchronize and record start
        cuda.synchronize()
        start_event.record()
        
        return {
            'start_event': start_event,
            'end_event': end_event,
            'cpu_start': cpu_start,
            'method': 'cuda_events'
        }
        
    def _start_bandwidth_measurement(self) -> Dict[str, Any]:
        """Start memory bandwidth measurement"""
        
        # Get baseline memory counters
        initial_counters = self.hardware_profiler.get_memory_counters()
        
        return {
            'initial_counters': initial_counters,
            'start_time': time.perf_counter_ns()
        }
        
    def _start_utilization_measurement(self) -> Dict[str, Any]:
        """Start compute utilization measurement"""
        
        # Start hardware profiling
        profiling_handle = self.hardware_profiler.start_profiling([
            'sm_efficiency',
            'achieved_occupancy', 
            'tensor_active',
            'warp_execution_efficiency'
        ])
        
        return {
            'profiling_handle': profiling_handle,
            'start_time': time.perf_counter_ns()
        }

class HardwareProfiler:
    """Interface to hardware profiling systems"""
    
    def __init__(self):
        self.nvml_available = self._check_nvml()
        self.nsys_available = self._check_nsight_systems()
        self.ncu_available = self._check_nsight_compute()
        
    def get_memory_counters(self) -> Dict[str, int]:
        """Get current memory usage counters"""
        if not self.nvml_available:
            return {}
            
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        counters = {}
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            counters[f'gpu_{i}_memory_used'] = mem_info.used
            counters[f'gpu_{i}_memory_free'] = mem_info.free
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            counters[f'gpu_{i}_gpu_util'] = util.gpu
            counters[f'gpu_{i}_mem_util'] = util.memory
            
        return counters
        
    def start_profiling(self, metrics: List[str]) -> 'ProfilingHandle':
        """Start hardware profiling session"""
        
        handle = ProfilingHandle()
        
        if self.ncu_available and any(m in ['sm_efficiency', 'achieved_occupancy'] for m in metrics):
            handle.ncu_session = self._start_ncu_profiling(metrics)
            
        if self.nsys_available:
            handle.nsys_session = self._start_nsys_profiling()
            
        return handle
        
    def stop_profiling(self, handle: 'ProfilingHandle') -> Dict[str, float]:
        """Stop profiling and return results"""
        results = {}
        
        if handle.ncu_session:
            ncu_results = self._stop_ncu_profiling(handle.ncu_session)
            results.update(ncu_results)
            
        if handle.nsys_session:
            nsys_results = self._stop_nsys_profiling(handle.nsys_session)
            results.update(nsys_results)
            
        return results

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""
    benchmark_name: str
    problem_size: Dict[str, Any]
    config: BenchmarkConfig
    
    # Performance metrics
    latency_ms: Optional[float] = None
    throughput_ops_per_sec: Optional[float] = None
    memory_bandwidth_gbps: Optional[float] = None
    compute_utilization_percent: Optional[float] = None
    energy_joules: Optional[float] = None
    
    # Resource usage
    peak_memory_mb: Optional[int] = None
    registers_per_thread: Optional[int] = None
    shared_memory_per_block: Optional[int] = None
    occupancy_percent: Optional[float] = None
    
    # Tessera-specific metrics
    compilation_time_ms: Optional[float] = None
    ir_optimization_time_ms: Optional[float] = None
    autotuning_time_ms: Optional[float] = None
    
    # Statistical information
    measurements_count: int = 0
    standard_deviation: Optional[float] = None
    confidence_interval_95: Optional[tuple] = None
    
    # Correctness validation
    numerical_accuracy: Optional[float] = None
    reference_comparison: Optional[Dict[str, float]] = None
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
        
    def add_error(self, error: str) -> None:
        self.errors.append(error)
        
    def is_valid(self) -> bool:
        return len(self.errors) == 0
        
    def calculate_efficiency(self, theoretical_peak: float) -> float:
        """Calculate efficiency vs theoretical peak performance"""
        if self.throughput_ops_per_sec is None:
            return 0.0
        return (self.throughput_ops_per_sec / theoretical_peak) * 100.0

@dataclass
class BenchmarkSuiteResult:
    """Results from a complete benchmark suite"""
    results: Dict[str, Union[BenchmarkResult, 'BenchmarkError']]
    suite_start_time: float = field(default_factory=time.time)
    suite_end_time: float = 0.0
    
    def get_successful_results(self) -> Dict[str, BenchmarkResult]:
        """Get only successful benchmark results"""
        return {k: v for k, v in self.results.items() 
                if isinstance(v, BenchmarkResult) and v.is_valid()}
                
    def get_failed_results(self) -> Dict[str, Union[BenchmarkResult, 'BenchmarkError']]:
        """Get failed benchmark results"""  
        return {k: v for k, v in self.results.items()
                if isinstance(v, BenchmarkError) or (isinstance(v, BenchmarkResult) and not v.is_valid())}
                
    def calculate_suite_statistics(self) -> Dict[str, Any]:
        """Calculate aggregate statistics across all benchmarks"""
        successful = self.get_successful_results()
        
        if not successful:
            return {}
            
        stats = {
            'total_benchmarks': len(self.results),
            'successful_benchmarks': len(successful),
            'failed_benchmarks': len(self.results) - len(successful),
            'average_latency_ms': 0.0,
            'average_throughput': 0.0,
            'average_efficiency': 0.0
        }
        
        latencies = [r.latency_ms for r in successful.values() if r.latency_ms]
        if latencies:
            stats['average_latency_ms'] = sum(latencies) / len(latencies)
            
        throughputs = [r.throughput_ops_per_sec for r in successful.values() if r.throughput_ops_per_sec]
        if throughputs:
            stats['average_throughput'] = sum(throughputs) / len(throughputs)
            
        return stats

@dataclass
class BenchmarkError:
    """Error information for failed benchmarks"""
    error_message: str
    error_type: str = "unknown"
    stack_trace: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
```

### Statistical Analysis Framework

```python
import numpy as np
from scipy import stats
from typing import Tuple, List
import warnings

class AnalysisEngine:
    """Statistical analysis engine for benchmark results"""
    
    def __init__(self):
        self.significance_level = 0.05
        self.min_samples = 30
        self.max_cv_percent = 10.0  # Maximum coefficient of variation
        
    def analyze_measurements(self, 
                           measurements: List[float],
                           confidence_level: float = 0.95) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on measurements"""
        
        if len(measurements) < 2:
            return {'error': 'Insufficient measurements for analysis'}
            
        measurements = np.array(measurements)
        
        # Basic statistics
        mean = np.mean(measurements)
        std = np.std(measurements, ddof=1)
        median = np.median(measurements)
        cv = (std / mean) * 100.0 if mean > 0 else float('inf')
        
        # Outlier detection using IQR method
        q1 = np.percentile(measurements, 25)
        q3 = np.percentile(measurements, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = measurements[(measurements < lower_bound) | (measurements > upper_bound)]
        clean_measurements = measurements[(measurements >= lower_bound) & (measurements <= upper_bound)]
        
        # Recalculate statistics without outliers if significant outliers found
        if len(outliers) > len(measurements) * 0.05:  # More than 5% outliers
            mean_clean = np.mean(clean_measurements)
            std_clean = np.std(clean_measurements, ddof=1)
        else:
            mean_clean = mean
            std_clean = std
            clean_measurements = measurements
            
        # Confidence interval
        alpha = 1 - confidence_level
        dof = len(clean_measurements) - 1
        t_value = stats.t.ppf(1 - alpha/2, dof) if dof > 0 else 0
        margin_error = t_value * (std_clean / np.sqrt(len(clean_measurements)))
        ci_lower = mean_clean - margin_error
        ci_upper = mean_clean + margin_error
        
        # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
        if len(clean_measurements) <= 50:
            normality_stat, normality_p = stats.shapiro(clean_measurements)
            normality_test = 'shapiro'
        else:
            normality_result = stats.anderson(clean_measurements, dist='norm')
            normality_stat = normality_result.statistic
            normality_p = 1.0 if normality_stat < normality_result.critical_values[2] else 0.0
            normality_test = 'anderson'
            
        is_normal = normality_p > self.significance_level
        
        return {
            'mean': mean_clean,
            'std': std_clean,
            'median': median,
            'cv_percent': cv,
            'min': np.min(clean_measurements),
            'max': np.max(clean_measurements),
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': confidence_level,
            'outliers_count': len(outliers),
            'outliers_percent': (len(outliers) / len(measurements)) * 100.0,
            'is_normal_distribution': is_normal,
            'normality_test': normality_test,
            'normality_p_value': normality_p,
            'sample_size': len(clean_measurements),
            'is_reliable': cv < self.max_cv_percent and len(clean_measurements) >= self.min_samples
        }
        
    def compare_benchmarks(self,
                          results_a: List[float],
                          results_b: List[float],
                          test_type: str = 'auto') -> Dict[str, Any]:
        """Compare two benchmark result sets statistically"""
        
        if len(results_a) < 2 or len(results_b) < 2:
            return {'error': 'Insufficient data for comparison'}
            
        results_a = np.array(results_a)
        results_b = np.array(results_b)
        
        # Basic comparison statistics
        mean_a = np.mean(results_a)
        mean_b = np.mean(results_b)
        std_a = np.std(results_a, ddof=1)
        std_b = np.std(results_b, ddof=1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(results_a) - 1) * std_a**2 + 
                             (len(results_b) - 1) * std_b**2) / 
                            (len(results_a) + len(results_b) - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
        
        # Choose appropriate statistical test
        if test_type == 'auto':
            # Check assumptions for parametric vs non-parametric test
            
            # Normality tests
            _, p_norm_a = stats.shapiro(results_a) if len(results_a) <= 50 else stats.normaltest(results_a)
            _, p_norm_b = stats.shapiro(results_b) if len(results_b) <= 50 else stats.normaltest(results_b)
            
            both_normal = p_norm_a > 0.05 and p_norm_b > 0.05
            
            # Variance equality test
            _, p_var = stats.levene(results_a, results_b)
            equal_variance = p_var > 0.05
            
            if both_normal and equal_variance:
                test_type = 'ttest_ind'
            elif both_normal:
                test_type = 'ttest_welch'  
            else:
                test_type = 'mannwhitneyu'
                
        # Perform the chosen statistical test
        if test_type == 'ttest_ind':
            statistic, p_value = stats.ttest_ind(results_a, results_b, equal_var=True)
            test_name = "Independent t-test"
        elif test_type == 'ttest_welch':
            statistic, p_value = stats.ttest_ind(results_a, results_b, equal_var=False)
            test_name = "Welch's t-test"
        elif test_type == 'mannwhitneyu':
            statistic, p_value = stats.mannwhitneyu(results_a, results_b, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        else:
            return {'error': f'Unknown test type: {test_type}'}
            
        # Interpret results
        is_significant = p_value < self.significance_level
        
        if cohens_d < 0.2:
            effect_size = "negligible"
        elif cohens_d < 0.5:
            effect_size = "small"
        elif cohens_d < 0.8:
            effect_size = "medium" 
        else:
            effect_size = "large"
            
        # Calculate confidence interval for difference in means
        diff_mean = mean_a - mean_b
        se_diff = np.sqrt(std_a**2/len(results_a) + std_b**2/len(results_b))
        dof = len(results_a) + len(results_b) - 2
        t_crit = stats.t.ppf(0.975, dof)
        ci_lower = diff_mean - t_crit * se_diff
        ci_upper = diff_mean + t_crit * se_diff
        
        return {
            'test_name': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_significant': is_significant,
            'significance_level': self.significance_level,
            'mean_a': float(mean_a),
            'mean_b': float(mean_b),
            'mean_difference': float(diff_mean),
            'mean_difference_percent': float((diff_mean / mean_b) * 100.0) if mean_b != 0 else float('inf'),
            'cohens_d': float(cohens_d),
            'effect_size': effect_size,
            'confidence_interval_difference': (float(ci_lower), float(ci_upper)),
            'sample_size_a': len(results_a),
            'sample_size_b': len(results_b)
        }
        
    def detect_performance_regression(self,
                                    baseline_results: List[float],
                                    current_results: List[float],
                                    regression_threshold_percent: float = 5.0) -> Dict[str, Any]:
        """Detect if current results show performance regression vs baseline"""
        
        comparison = self.compare_benchmarks(current_results, baseline_results)
        
        if 'error' in comparison:
            return comparison
            
        baseline_mean = comparison['mean_b']  # baseline is 'b' in comparison
        current_mean = comparison['mean_a']   # current is 'a' in comparison
        
        # Calculate performance change
        perf_change_percent = comparison['mean_difference_percent']
        
        # Regression detection logic
        is_regression = False
        regression_severity = "none"
        
        if perf_change_percent < -regression_threshold_percent:  # Performance decreased
            is_regression = True
            if perf_change_percent < -20.0:
                regression_severity = "severe"
            elif perf_change_percent < -10.0:
                regression_severity = "moderate"
            else:
                regression_severity = "mild"
                
        elif perf_change_percent > regression_threshold_percent:  # Performance improved
            regression_severity = "improvement"
            
        return {
            **comparison,
            'is_regression': is_regression,
            'regression_severity': regression_severity,
            'regression_threshold_percent': regression_threshold_percent,
            'performance_change_percent': perf_change_percent,
            'baseline_mean_performance': float(baseline_mean),
            'current_mean_performance': float(current_mean)
        }
```

## Integration with Tessera Runtime

### Tessera-Specific Extensions

```python
import tessera
from tessera.runtime import TesseraRuntime
from tessera.ir import GraphIR, ScheduleIR, TileIR, TargetIR

class TesseraIntegration:
    """Integration layer between TesseraBench and Tessera runtime"""
    
    def __init__(self, tessera_runtime: TesseraRuntime):
        self.runtime = tessera_runtime
        self.ir_inspector = IRInspector()
        self.compilation_profiler = CompilationProfiler()
        
    def compile_with_timing(self, 
                          kernel_source: str,
                          compilation_config: Dict[str, Any]) -> Tuple['CompiledKernel', Dict[str, float]]:
        """Compile kernel and measure compilation times at each IR level"""
        
        timing_results = {}
        
        # Graph IR generation
        start_time = time.perf_counter()
        graph_ir = tessera.compile.to_graph_ir(kernel_source)
        timing_results['graph_ir_ms'] = (time.perf_counter() - start_time) * 1000
        
        # Schedule IR optimization
        start_time = time.perf_counter()
        schedule_ir = tessera.compile.to_schedule_ir(graph_ir, compilation_config.get('schedule_opts', {}))
        timing_results['schedule_ir_ms'] = (time.perf_counter() - start_time) * 1000
        
        # Tile IR lowering
        start_time = time.perf_counter()
        tile_ir = tessera.compile.to_tile_ir(schedule_ir, compilation_config.get('tile_opts', {}))
        timing_results['tile_ir_ms'] = (time.perf_counter() - start_time) * 1000
        
        # Target IR generation (PTX or CUDA Tile IR)
        start_time = time.perf_counter()
        target_ir = tessera.compile.to_target_ir(tile_ir, compilation_config.get('target_opts', {}))
        timing_results['target_ir_ms'] = (time.perf_counter() - start_time) * 1000
        
        # Autotuning if enabled
        if compilation_config.get('autotuning_enabled', False):
            start_time = time.perf_counter()
            optimized_kernel = tessera.autotune.optimize(target_ir, compilation_config.get('autotune_config', {}))
            timing_results['autotuning_ms'] = (time.perf_counter() - start_time) * 1000
        else:
            optimized_kernel = target_ir
            timing_results['autotuning_ms'] = 0.0
            
        # Final compilation to executable
        start_time = time.perf_counter()
        compiled_kernel = tessera.compile.to_executable(optimized_kernel)
        timing_results['final_compilation_ms'] = (time.perf_counter() - start_time) * 1000
        
        return compiled_kernel, timing_results
        
    def launch_kernel_with_profiling(self,
                                   kernel: 'CompiledKernel',
                                   args: List[Any],
                                   grid_dim: Tuple[int, ...],
                                   block_dim: Tuple[int, ...],
                                   profiling_config: Dict[str, Any]) -> Dict[str, Any]:
        """Launch kernel with comprehensive profiling"""
        
        profiling_results = {}
        
        # Memory usage before launch
        initial_memory = self.runtime.get_memory_usage()
        
        # Launch with timing
        with self.runtime.profiling_context(profiling_config) as prof:
            launch_result = self.runtime.launch_kernel(
                kernel, args, grid_dim, block_dim
            )
            
        # Extract profiling data
        profiling_results.update(prof.get_results())
        
        # Memory usage after launch
        final_memory = self.runtime.get_memory_usage()
        profiling_results['memory_delta_mb'] = (final_memory - initial_memory) / (1024 * 1024)
        
        return {
            'launch_result': launch_result,
            'profiling_results': profiling_results
        }

class IRInspector:
    """Inspector for Tessera IR at different levels"""
    
    def inspect_graph_ir(self, graph_ir: GraphIR) -> Dict[str, Any]:
        """Inspect Graph IR characteristics"""
        return {
            'node_count': len(graph_ir.nodes),
            'operation_types': self._count_operation_types(graph_ir),
            'memory_footprint_estimate': self._estimate_memory_footprint(graph_ir),
            'autodiff_complexity': self._analyze_autodiff_complexity(graph_ir),
            'collective_operations': self._count_collective_ops(graph_ir)
        }
        
    def inspect_schedule_ir(self, schedule_ir: ScheduleIR) -> Dict[str, Any]:
        """Inspect Schedule IR optimizations"""
        return {
            'loop_structure': self._analyze_loop_structure(schedule_ir),
            'tiling_configuration': self._extract_tiling_config(schedule_ir),
            'fusion_opportunities': self._analyze_fusion_opportunities(schedule_ir),
            'memory_layout_optimizations': self._analyze_memory_layouts(schedule_ir),
            'parallelization_strategy': self._extract_parallelization_strategy(schedule_ir)
        }
        
    def inspect_tile_ir(self, tile_ir: TileIR) -> Dict[str, Any]:
        """Inspect Tile IR hardware mapping"""
        return {
            'shared_memory_usage': self._analyze_shared_memory_usage(tile_ir),
            'register_pressure': self._estimate_register_pressure(tile_ir),
            'tensor_core_utilization': self._analyze_tensor_core_usage(tile_ir),
            'async_copy_patterns': self._analyze_async_copies(tile_ir),
            'synchronization_overhead': self._analyze_barriers(tile_ir)
        }
        
    def inspect_target_ir(self, target_ir: TargetIR) -> Dict[str, Any]:
        """Inspect Target IR code generation"""
        return {
            'instruction_count': len(target_ir.instructions),
            'instruction_mix': self._analyze_instruction_mix(target_ir),
            'occupancy_estimate': self._estimate_occupancy(target_ir),
            'memory_access_patterns': self._analyze_memory_patterns(target_ir),
            'optimization_passes_applied': target_ir.applied_optimizations
        }

class DistributedCoordinator:
    """Coordinator for multi-GPU distributed benchmarking"""
    
    def __init__(self):
        self.mesh_configs = {}
        self.synchronization_barriers = {}
        
    def setup_benchmark_mesh(self, 
                           device_ids: List[int],
                           mesh_config: Dict[str, Any]) -> str:
        """Setup distributed mesh for benchmarking"""
        
        mesh_id = f"mesh_{hash(tuple(device_ids))}"
        
        # Initialize NCCL communicators
        import tessera.distributed as dist
        mesh = dist.create_mesh(
            devices=device_ids,
            axes=mesh_config.get('axes', ['dp', 'tp']),
            shape=mesh_config.get('shape', [len(device_ids), 1])
        )
        
        self.mesh_configs[mesh_id] = {
            'mesh': mesh,
            'device_ids': device_ids,
            'config': mesh_config
        }
        
        return mesh_id
        
    def run_distributed_benchmark(self,
                                 mesh_id: str,
                                 benchmark_func: callable,
                                 args: List[Any]) -> Dict[str, Any]:
        """Run benchmark across distributed mesh"""
        
        if mesh_id not in self.mesh_configs:
            raise ValueError(f"Unknown mesh ID: {mesh_id}")
            
        mesh_config = self.mesh_configs[mesh_id]
        mesh = mesh_config['mesh']
        device_ids = mesh_config['device_ids']
        
        # Create synchronization barriers
        barrier_name = f"benchmark_barrier_{mesh_id}"
        self.synchronization_barriers[barrier_name] = threading.Barrier(len(device_ids))
        
        results = {}
        threads = []
        
        # Launch benchmark on each device
        for i, device_id in enumerate(device_ids):
            thread = threading.Thread(
                target=self._run_device_benchmark,
                args=(device_id, barrier_name, benchmark_func, args, results)
            )
            threads.append(thread)
            thread.start()
            
        # Wait for all devices to complete
        for thread in threads:
            thread.join()
            
        # Aggregate results
        aggregated = self._aggregate_distributed_results(results, mesh_config)
        
        return aggregated
        
    def _run_device_benchmark(self,
                            device_id: int,
                            barrier_name: str,
                            benchmark_func: callable,
                            args: List[Any],
                            results: Dict[str, Any]) -> None:
        """Run benchmark on a single device"""
        
        # Set device context
        import tessera.cuda as cuda
        cuda.set_device(device_id)
        
        try:
            # Synchronize all devices before starting
            self.synchronization_barriers[barrier_name].wait()
            
            # Run benchmark
            result = benchmark_func(*args)
            results[f'device_{device_id}'] = result
            
            # Synchronize all devices after completion
            self.synchronization_barriers[barrier_name].wait()
            
        except Exception as e:
            results[f'device_{device_id}'] = {'error': str(e)}
            
    def _aggregate_distributed_results(self,
                                     device_results: Dict[str, Any],
                                     mesh_config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all devices"""
        
        successful_results = {k: v for k, v in device_results.items() 
                            if 'error' not in v}
        failed_results = {k: v for k, v in device_results.items() 
                        if 'error' in v}
        
        if not successful_results:
            return {'error': 'All devices failed', 'device_errors': failed_results}
            
        # Calculate aggregate metrics
        latencies = [r.get('latency_ms', 0) for r in successful_results.values()]
        throughputs = [r.get('throughput_ops_per_sec', 0) for r in successful_results.values()]
        
        return {
            'successful_devices': len(successful_results),
            'failed_devices': len(failed_results),
            'aggregate_latency_ms': max(latencies) if latencies else 0,  # Slowest device
            'aggregate_throughput': sum(throughputs),  # Total throughput
            'average_device_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'latency_variance': np.var(latencies) if len(latencies) > 1 else 0,
            'scaling_efficiency': (sum(throughputs) / (throughputs[0] * len(throughputs))) * 100 if throughputs else 0,
            'device_results': device_results,
            'mesh_config': mesh_config['config']
        }

# Example benchmark implementation
class FlashAttentionBenchmark:
    """Flash Attention benchmark implementation"""
    
    def get_name(self) -> str:
        return "flash_attention"
        
    def get_description(self) -> str:
        return "Flash Attention with online softmax computation"
        
    def get_supported_precisions(self) -> List[str]:
        return ["fp16", "bf16", "fp8_e4m3@accum(fp32)", "fp8_e5m2@accum(fp32)"]
        
    def get_problem_sizes(self) -> List[Dict[str, Any]]:
        """Return comprehensive problem sizes for Flash Attention"""
        sizes = []
        
        # Standard sizes
        for batch_size in [1, 4, 16, 32]:
            for num_heads in [8, 16, 32]:
                for seq_len in [512, 1024, 2048, 4096, 8192]:
                    for head_dim in [64, 128]:
                        sizes.append({
                            'batch_size': batch_size,
                            'num_heads': num_heads,
                            'seq_len': seq_len,
                            'head_dim': head_dim,
                            'causal': True
                        })
                        
        # Long sequence special cases
        for seq_len in [16384, 32768, 65536]:
            sizes.append({
                'batch_size': 1,
                'num_heads': 16,
                'seq_len': seq_len,
                'head_dim': 64,
                'causal': True
            })
            
        return sizes
        
    def setup(self, config: BenchmarkConfig) -> None:
        """Setup Flash Attention benchmark"""
        self.config = config
        self.tessera_integration = TesseraIntegration(tessera.get_runtime())
        
        # Compile Flash Attention kernel for each precision
        self.compiled_kernels = {}
        
        kernel_source = """
        @tessera.kernel
        def flash_attention(Q: Tensor["B", "H", "S", "D", dtype],
                          K: Tensor["B", "H", "S", "D", dtype], 
                          V: Tensor["B", "H", "S", "D", dtype],
                          O: Tensor["B", "H", "S", "D", dtype],
                          scale: float,
                          causal: bool = True):
            # Flash Attention implementation in Tessera
            pass  # Implementation details...
        """
        
        for precision in config.precision_policies:
            compilation_config = {
                'precision_policy': precision,
                'optimization_level': config.ir_optimization_levels[0],
                'autotuning_enabled': config.autotuning_enabled,
                'target_arch': config.hardware.gpu_arch
            }
            
            kernel, timing = self.tessera_integration.compile_with_timing(
                kernel_source, compilation_config
            )
            
            self.compiled_kernels[precision] = {
                'kernel': kernel,
                'compilation_timing': timing
            }
            
    def run_single(self, problem_size: Dict[str, Any]) -> BenchmarkResult:
        """Run single Flash Attention benchmark"""
        
        # Use first precision policy for this run
        precision = self.config.precision_policies[0]
        kernel_info = self.compiled_kernels[precision]
        
        # Create input tensors
        B = problem_size['batch_size']
        H = problem_size['num_heads'] 
        S = problem_size['seq_len']
        D = problem_size['head_dim']
        
        import tessera
        Q = tessera.randn((B, H, S, D), dtype=precision.split('@')[0])
        K = tessera.randn((B, H, S, D), dtype=precision.split('@')[0])
        V = tessera.randn((B, H, S, D), dtype=precision.split('@')[0])
        O = tessera.zeros((B, H, S, D), dtype=precision.split('@')[0])
        
        scale = 1.0 / (D ** 0.5)
        
        # Calculate theoretical performance
        # Flash Attention: 4 * B * H * S^2 * D FLOPs
        flops = 4 * B * H * S * S * D
        theoretical_tflops = self._get_theoretical_tflops(self.config.hardware.gpu_arch)
        
        # Warmup runs
        for _ in range(self.config.warmup_iterations):
            tessera.launch_kernel(kernel_info['kernel'], [Q, K, V, O, scale, True])
            tessera.synchronize()
            
        # Timing runs
        measurements = []
        for _ in range(self.config.timing_iterations):
            start_time = time.perf_counter()
            
            # Launch kernel with profiling
            profiling_result = self.tessera_integration.launch_kernel_with_profiling(
                kernel_info['kernel'],
                [Q, K, V, O, scale, True],
                grid_dim=(B * H, (S + 127) // 128, 1),
                block_dim=(128, 1, 1),
                profiling_config={'measure_bandwidth': True, 'measure_occupancy': True}
            )
            
            tessera.synchronize()
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            measurements.append(latency_ms)
            
        # Statistical analysis
        analysis_engine = AnalysisEngine()
        stats = analysis_engine.analyze_measurements(measurements)
        
        # Calculate performance metrics
        mean_latency = stats['mean']
        throughput_ops_per_sec = flops / (mean_latency / 1000) if mean_latency > 0 else 0
        achieved_tflops = throughput_ops_per_sec / 1e12
        efficiency = (achieved_tflops / theoretical_tflops) * 100 if theoretical_tflops > 0 else 0
        
        # Create result
        result = BenchmarkResult(
            benchmark_name=self.get_name(),
            problem_size=problem_size,
            config=self.config,
            latency_ms=mean_latency,
            throughput_ops_per_sec=throughput_ops_per_sec,
            compilation_time_ms=kernel_info['compilation_timing']['total_ms'],
            measurements_count=len(measurements),
            standard_deviation=stats['std'],
            confidence_interval_95=(stats['confidence_interval'][0], stats['confidence_interval'][1])
        )
        
        # Add profiling information if available
        if 'profiling_results' in profiling_result:
            prof_results = profiling_result['profiling_results']
            result.occupancy_percent = prof_results.get('occupancy_percent')
            result.memory_bandwidth_gbps = prof_results.get('memory_bandwidth_gbps')
            result.compute_utilization_percent = prof_results.get('compute_utilization_percent')
            
        return result
        
    def cleanup(self) -> None:
        """Cleanup benchmark resources"""
        self.compiled_kernels.clear()
        
    def validate_result(self, result: BenchmarkResult) -> bool:
        """Validate Flash Attention result correctness"""
        
        # Basic sanity checks
        if result.latency_ms is None or result.latency_ms <= 0:
            return False
            
        if result.throughput_ops_per_sec is None or result.throughput_ops_per_sec <= 0:
            return False
            
        # Check if performance is reasonable (not too slow, not impossibly fast)
        problem_size = result.problem_size
        expected_min_latency = 0.01  # 0.01ms minimum
        expected_max_latency = 10000  # 10 seconds maximum
        
        if not (expected_min_latency <= result.latency_ms <= expected_max_latency):
            result.add_warning(f"Latency {result.latency_ms}ms outside expected range [{expected_min_latency}, {expected_max_latency}]")
            
        return True
        
    def _get_theoretical_tflops(self, gpu_arch: str) -> float:
        """Get theoretical TFLOPS for GPU architecture"""
        arch_specs = {
            'sm_70': 125.0,   # V100
            'sm_75': 130.0,   # T4/RTX 20xx  
            'sm_80': 312.0,   # A100
            'sm_86': 285.0,   # RTX 30xx
            'sm_89': 165.0,   # RTX 40xx
            'sm_90': 1320.0   # H100
        }
        return arch_specs.get(gpu_arch, 100.0)  # Default fallback