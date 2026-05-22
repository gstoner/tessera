        // Synchronize results across ranks
        tessera::distributed::all_reduce(O);
        
        if (rank == 0) {
            std::cout << "Distributed meta-tracing execution completed successfully!" << std::endl;
            
            // Print distributed performance statistics
            auto stats = runtime.getRuntimeStats();
            std::cout << "Distributed stats - Total executions: " << stats.total_executions << std::endl;
            std::cout << "Average execution time: " << stats.average_tier3_time_ms << "ms" << std::endl;
        }
        
        tessera::distributed::destroy_process_group();
    }
};

int main() {
    std::cout << "=== Tessera Meta-Tracing PoC API Examples ===" << std::endl;
    
    try {
        // Run basic usage example
        std::cout << "\n1. Running Basic Usage Example..." << std::endl;
        // (Basic usage code from A.1 would go here)
        
        // Run advanced configuration example
        std::cout << "\n2. Running Advanced Configuration Example..." << std::endl;
        CustomMetaTracingSetup advanced_setup;
        advanced_setup.demonstrateAdvancedUsage();
        
        // Run pattern analysis example
        std::cout << "\n3. Running Custom Pattern Analysis..." << std::endl;
        CustomPatternAnalyzer pattern_demo;
        pattern_demo.demonstratePatternAnalysis();
        
        // Run comprehensive benchmark
        std::cout << "\n4. Running Comprehensive Benchmark Suite..." << std::endl;
        ComprehensiveBenchmarkDemo benchmark_demo;
        benchmark_demo.runCompleteBenchmarkSuite();
        
        // Run integration examples
        std::cout << "\n5. Running Tessera Integration Examples..." << std::endl;
        TesseraMetaTracingIntegration integration_demo;
        integration_demo.demonstrateSeamlessIntegration();
        integration_demo.demonstrateAutogradIntegration();
        
        std::cout << "\n=== All Examples Completed Successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error running examples: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
```

---

## Appendix B: Performance Models and Predictions

### B.1 Theoretical Performance Model

```cpp
namespace tessera::metatracing::performance {

class FlashAttentionPerformanceModel {
public:
    struct ModelParameters {
        // Hardware characteristics
        float peak_compute_tflops;
        float memory_bandwidth_gbps;
        int num_sms;
        int max_threads_per_sm;
        
        // Kernel characteristics
        float tensor_core_utilization;
        float memory_efficiency;
        float occupancy_achieved;
        
        // Problem characteristics
        int batch_size;
        int num_heads; 
        int seq_len;
        int head_dim;
        bool is_causal;
    };
    
    struct PerformancePrediction {
        float predicted_time_ms;
        float predicted_tflops;
        float predicted_memory_bw_gbps;
        float predicted_occupancy;
        
        // Confidence intervals
        float time_confidence_interval[2];
        float tflops_confidence_interval[2];
        
        // Breakdown analysis
        float compute_bound_ratio;
        float memory_bound_ratio;
        float overhead_ratio;
    };
    
    PerformancePrediction predictPerformance(const ModelParameters& params) {
        PerformancePrediction prediction;
        
        // Calculate theoretical FLOP count
        int64_t total_flops = calculateFlashAttentionFlops(
            params.batch_size, params.num_heads, params.seq_len, params.head_dim, params.is_causal
        );
        
        // Compute-bound prediction
        float compute_bound_time = (total_flops / 1e12f) / 
                                  (params.peak_compute_tflops * params.tensor_core_utilization);
        
        // Memory-bound prediction  
        int64_t memory_bytes = calculateFlashAttentionMemoryBytes(
            params.batch_size, params.num_heads, params.seq_len, params.head_dim
        );
        float memory_bound_time = (memory_bytes / 1e9f) / 
                                 (params.memory_bandwidth_gbps * params.memory_efficiency);
        
        // Take the maximum (bottleneck)
        prediction.predicted_time_ms = std::max(compute_bound_time, memory_bound_time) * 1000.0f;
        
        // Account for occupancy effects
        prediction.predicted_time_ms /= params.occupancy_achieved;
        
        // Calculate derived metrics
        prediction.predicted_tflops = (total_flops / 1e12f) / (prediction.predicted_time_ms / 1000.0f);
        prediction.predicted_memory_bw_gbps = (memory_bytes / 1e9f) / (prediction.predicted_time_ms / 1000.0f);
        prediction.predicted_occupancy = params.occupancy_achieved;
        
        // Compute bottleneck ratios
        prediction.compute_bound_ratio = compute_bound_time / prediction.predicted_time_ms;
        prediction.memory_bound_ratio = memory_bound_time / prediction.predicted_time_ms;
        prediction.overhead_ratio = 1.0f - prediction.compute_bound_ratio - prediction.memory_bound_ratio;
        
        // Calculate confidence intervals (simplified model)
        float uncertainty_factor = 0.15f;  // 15% uncertainty
        prediction.time_confidence_interval[0] = prediction.predicted_time_ms * (1.0f - uncertainty_factor);
        prediction.time_confidence_interval[1] = prediction.predicted_time_ms * (1.0f + uncertainty_factor);
        prediction.tflops_confidence_interval[0] = prediction.predicted_tflops * (1.0f - uncertainty_factor);
        prediction.tflops_confidence_interval[1] = prediction.predicted_tflops * (1.0f + uncertainty_factor);
        
        return prediction;
    }

private:
    int64_t calculateFlashAttentionFlops(int B, int H, int S, int D, bool is_causal) {
        // Q @ K^T: B * H * S * S * D FLOPs
        int64_t qk_flops = static_cast<int64_t>(B) * H * S * S * D;
        if (is_causal) {
            qk_flops /= 2;  // Roughly half due to causal masking
        }
        
        // Softmax: approximately 5 * B * H * S * S FLOPs
        int64_t softmax_flops = static_cast<int64_t>(5) * B * H * S * S;
        if (is_causal) {
            softmax_flops /= 2;
        }
        
        // P @ V: B * H * S * S * D FLOPs
        int64_t pv_flops = static_cast<int64_t>(B) * H * S * S * D;
        if (is_causal) {
            pv_flops /= 2;
        }
        
        return qk_flops + softmax_flops + pv_flops;
    }
    
    int64_t calculateFlashAttentionMemoryBytes(int B, int H, int S, int D) {
        // Input tensors: Q, K, V (each B * H * S * D * sizeof(bf16))
        int64_t input_bytes = static_cast<int64_t>(3) * B * H * S * D * 2;
        
        // Output tensor: O (B * H * S * D * sizeof(bf16))
        int64_t output_bytes = static_cast<int64_t>(B) * H * S * D * 2;
        
        // Intermediate shared memory usage (approximation)
        int64_t intermediate_bytes = static_cast<int64_t>(B) * H * 256 * 128 * 2;  // Tile size dependent
        
        return input_bytes + output_bytes + intermediate_bytes;
    }
};

class TierPerformancePrediction {
public:
    struct TierPrediction {
        float tier1_time_ms;
        float tier2_time_ms;  
        float tier3_time_ms;
        
        float tier2_speedup;
        float tier3_speedup;
        
        float tier2_confidence;
        float tier3_confidence;
    };
    
    TierPrediction predictTierPerformance(
        const FlashAttentionPerformanceModel::ModelParameters& base_params,
        const std::vector<tessera::metatracing::AttentionTrace>& historical_traces
    ) {
        TierPrediction prediction;
        
        FlashAttentionPerformanceModel model;
        
        // Tier 1: Interpreter performance (baseline with overhead)
        auto tier1_params = base_params;
        tier1_params.tensor_core_utilization *= 0.7f;  // Lower utilization in interpreter
        tier1_params.memory_efficiency *= 0.8f;        // Some overhead
        auto tier1_pred = model.predictPerformance(tier1_params);
        prediction.tier1_time_ms = tier1_pred.predicted_time_ms;
        
        // Tier 2: Standard compilation with trace guidance
        auto tier2_params = base_params;
        tier2_params.tensor_core_utilization *= 1.1f;  // Better utilization from trace guidance
        tier2_params.memory_efficiency *= 1.05f;       // Slightly better memory usage
        tier2_params.occupancy_achieved *= 1.02f;      // Better occupancy
        auto tier2_pred = model.predictPerformance(tier2_params);
        prediction.tier2_time_ms = tier2_pred.predicted_time_ms;
        
        // Tier 3: Specialized kernels
        auto tier3_params = base_params;
        tier3_params.tensor_core_utilization *= 1.25f; // Much better utilization
        tier3_params.memory_efficiency *= 1.15f;       // Optimized memory layout
        tier3_params.occupancy_achieved *= 1.05f;      // Optimized for specific shapes
        
        // Additional specialization benefits
        if (hasConsistentShapes(historical_traces)) {
            tier3_params.tensor_core_utilization *= 1.1f;  // Shape specialization benefit
        }
        if (hasConsistentDataTypes(historical_traces)) {
            tier3_params.memory_efficiency *= 1.05f;       // Data type specialization benefit
        }
        
        auto tier3_pred = model.predictPerformance(tier3_params);
        prediction.tier3_time_ms = tier3_pred.predicted_time_ms;
        
        // Calculate speedups
        prediction.tier2_speedup = prediction.tier1_time_ms / prediction.tier2_time_ms;
        prediction.tier3_speedup = prediction.tier1_time_ms / prediction.tier3_time_ms;
        
        // Calculate confidence based on trace consistency
        prediction.tier2_confidence = calculateTraceConsistency(historical_traces) * 0.8f + 0.2f;
        prediction.tier3_confidence = calculateTraceConsistency(historical_traces);
        
        return prediction;
    }

private:
    bool hasConsistentShapes(const std::vector<tessera::metatracing::AttentionTrace>& traces) {
        if (traces.empty()) return false;
        
        auto first_shape = traces[0].input_profile;
        for (const auto& trace : traces) {
            if (trace.input_profile.seq_len != first_shape.seq_len ||
                trace.input_profile.head_dim != first_shape.head_dim) {
                return false;
            }
        }
        return true;
    }
    
    bool hasConsistentDataTypes(const std::vector<tessera::metatracing::AttentionTrace>& traces) {
        if (traces.empty()) return false;
        
        auto first_dtype = traces[0].input_profile.input_dtype;
        for (const auto& trace : traces) {
            if (trace.input_profile.input_dtype != first_dtype) {
                return false;
            }
        }
        return true;
    }
    
    float calculateTraceConsistency(const std::vector<tessera::metatracing::AttentionTrace>& traces) {
        if (traces.size() < 2) return 0.5f;
        
        // Calculate coefficient of variation for execution times
        float mean_time = 0.0f;
        for (const auto& trace : traces) {
            mean_time += trace.execution_pattern.execution_time_ms;
        }
        mean_time /= traces.size();
        
        float variance = 0.0f;
        for (const auto& trace : traces) {
            float diff = trace.execution_pattern.execution_time_ms - mean_time;
            variance += diff * diff;
        }
        variance /= traces.size();
        
        float cv = std::sqrt(variance) / mean_time;  // Coefficient of variation
        
        // Convert to consistency score (lower CV = higher consistency)
        return std::exp(-cv * 2.0f);  // Exponential decay function
    }
};

} // namespace tessera::metatracing::performance
```

### B.2 Optimization Impact Model

```cpp
namespace tessera::metatracing::optimization {

class OptimizationImpactPredictor {
public:
    struct OptimizationImpact {
        float expected_speedup;
        float confidence_score;
        int implementation_effort; // 1-10 scale
        float risk_factor;         // 0-1 scale
        std::string description;
        
        struct DetailedBreakdown {
            float compute_improvement;
            float memory_improvement;
            float occupancy_improvement;
            float overhead_reduction;
        } breakdown;
    };
    
    OptimizationImpact predictTileSizeOptimization(
        const tessera::metatracing::TileConfiguration& current_config,
        const tessera::metatracing::TileConfiguration& proposed_config,
        const std::vector<tessera::metatracing::AttentionTrace>& traces
    ) {
        OptimizationImpact impact;
        impact.implementation_effort = 3;  // Medium effort
        impact.risk_factor = 0.2f;         // Low risk
        impact.description = "Tile size optimization based on trace analysis";
        
        // Analyze current performance characteristics
        float avg_occupancy = calculateAverageOccupancy(traces);
        float avg_memory_utilization = calculateAverageMemoryUtilization(traces);
        
        // Predict impact of new tile configuration
        float occupancy_improvement = predictOccupancyImprovement(
            current_config, proposed_config, avg_occupancy
        );
        float memory_improvement = predictMemoryImpact(
            current_config, proposed_config, avg_memory_utilization
        );
        
        // Calculate overall speedup
        impact.expected_speedup = 1.0f + 
                                 occupancy_improvement * 0.4f +     // Occupancy contributes 40%
                                 memory_improvement * 0.3f;         // Memory contributes 30%
        
        // Confidence based on trace consistency
        impact.confidence_score = calculateTileOptimizationConfidence(traces);
        
        // Detailed breakdown
        impact.breakdown.occupancy_improvement = occupancy_improvement;
        impact.breakdown.memory_improvement = memory_improvement;
        impact.breakdown.compute_improvement = occupancy_improvement * 0.6f;
        impact.breakdown.overhead_reduction = 0.05f;  // Small overhead reduction
        
        return impact;
    }
    
    OptimizationImpact predictShapeSpecialization(
        const std::vector<tessera::metatracing::AttentionTrace>& traces
    ) {
        OptimizationImpact impact;
        impact.implementation_effort = 6;  // Higher effort
        impact.risk_factor = 0.3f;         // Medium risk
        impact.description = "Shape specialization for consistent input dimensions";
        
        // Analyze shape consistency
        float shape_consistency = calculateShapeConsistency(traces);
        
        if (shape_consistency > 0.8f) {
            // High consistency - significant benefits expected
            impact.expected_speedup = 1.15f + shape_consistency * 0.25f;  // 15-40% improvement
            impact.confidence_score = shape_consistency;
            
            impact.breakdown.compute_improvement = 0.2f;      // Better instruction selection
            impact.breakdown.memory_improvement = 0.15f;     // Optimized memory layout
            impact.breakdown.occupancy_improvement = 0.1f;   // Better resource utilization
            impact.breakdown.overhead_reduction = 0.05f;     // Reduced branching
            
        } else {
            // Low consistency - limited benefits
            impact.expected_speedup = 1.05f;
            impact.confidence_score = 0.3f;
            impact.breakdown.compute_improvement = 0.05f;
            impact.breakdown.memory_improvement = 0.02f;
            impact.breakdown.occupancy_improvement = 0.01f;
            impact.breakdown.overhead_reduction = 0.01f;
        }
        
        return impact;
    }
    
    OptimizationImpact predictDataTypeSpecialization(
        const std::vector<tessera::metatracing::AttentionTrace>& traces
    ) {
        OptimizationImpact impact;
        impact.implementation_effort = 4;  // Medium effort
        impact.risk_factor = 0.25f;        // Medium-low risk
        impact.description = "Data type specialization for consistent precision";
        
        // Analyze data type consistency
        auto dtype_distribution = analyzeDataTypeDistribution(traces);
        
        float max_frequency = 0.0f;
        tessera::DataType dominant_dtype;
        for (const auto& [dtype, frequency] : dtype_distribution) {
            if (frequency > max_frequency) {
                max_frequency = frequency;
                dominant_dtype = dtype;
            }
        }
        
        if (max_frequency > 0.7f) {
            // Strong data type consistency
            float dtype_benefit = getDataTypeSpecializationBenefit(dominant_dtype);
            impact.expected_speedup = 1.0f + dtype_benefit;
            impact.confidence_score = max_frequency;
            
            impact.breakdown.compute_improvement = dtype_benefit * 0.6f;
            impact.breakdown.memory_improvement = dtype_benefit * 0.4f;
            impact.breakdown.occupancy_improvement = 0.0f;
            impact.breakdown.overhead_reduction = 0.02f;
            
        } else {
            // Mixed data types - limited benefits
            impact.expected_speedup = 1.02f;
            impact.confidence_score = 0.4f;
            impact.breakdown.compute_improvement = 0.01f;
            impact.breakdown.memory_improvement = 0.01f;
            impact.breakdown.occupancy_improvement = 0.0f;
            impact.breakdown.overhead_reduction = 0.0f;
        }
        
        return impact;
    }

private:
    float calculateAverageOccupancy(const std::vector<tessera::metatracing::AttentionTrace>& traces) {
        float total = 0.0f;
        for (const auto& trace : traces) {
            total += trace.execution_pattern.occupancy_achieved;
        }
        return traces.empty() ? 0.0f : (total / traces.size());
    }
    
    float calculateAverageMemoryUtilization(const std::vector<tessera::metatracing::AttentionTrace>& traces) {
        float total = 0.0f;
        for (const auto& trace : traces) {
            total += trace.execution_pattern.memory_bandwidth_gbps / 1600.0f;  // Assume 1.6 TB/s peak
        }
        return traces.empty() ? 0.0f : (total / traces.size());
    }
    
    float predictOccupancyImprovement(
        const tessera::metatracing::TileConfiguration& current,
        const tessera::metatracing::TileConfiguration& proposed,
        float current_occupancy
    ) {
        // Simplified occupancy model
        int current_threads = current.block_m * current.block_n * current.num_warps * 32;
        int proposed_threads = proposed.block_m * proposed.block_n * proposed.num_warps * 32;
        
        if (current_occupancy < 0.7f && proposed_threads > current_threads) {
            return 0.15f;  // 15% improvement for low occupancy cases
        } else if (current_occupancy > 0.9f && proposed_threads < current_threads) {
            return 0.05f;  // 5% improvement for over-subscribed cases
        }
        
        return 0.02f;  // Minimal change
    }
    
    float predictMemoryImpact(
        const tessera::metatracing::TileConfiguration& current,
        const tessera::metatracing::TileConfiguration& proposed,
        float current_memory_utilization
    ) {
        // Analyze shared memory usage change
        int current_smem = (current.block_m * current.block_k + 
                           current.block_k * current.block_n) * 2;  // Simplified
        int proposed_smem = (proposed.block_m * proposed.block_k +
                            proposed.block_k * proposed.block_n) * 2;
        
        float smem_ratio = static_cast<float>(proposed_smem) / current_smem;
        
        if (smem_ratio > 1.2f) {
            return -0.05f;  // Higher memory usage might hurt
        } else if (smem_ratio < 0.8f) {
            return 0.08f;   // Lower memory usage helps with occupancy
        }
        
        return 0.02f;  // Minimal impact
    }
    
    float calculateShapeConsistency(const std::vector<tessera::metatracing::AttentionTrace>& traces) {
        if (traces.empty()) return 0.0f;
        
        std::map<std::pair<int, int>, int> shape_counts;
        
        for (const auto& trace : traces) {
            auto shape_key = std::make_pair(trace.input_profile.seq_len, trace.input_profile.head_dim);
            shape_counts[shape_key]++;
        }
        
        int max_count = 0;
        for (const auto& [shape, count] : shape_counts) {
            max_count = std::max(max_count, count);
        }
        
        return static_cast<float>(max_count) / traces.size();
    }
    
    std::map<tessera::DataType, float> analyzeDataTypeDistribution(
        const std::vector<tessera::metatracing::AttentionTrace>& traces
    ) {
        std::map<tessera::DataType, int> dtype_counts;
        
        for (const auto& trace : traces) {
            dtype_counts[trace.input_profile.input_dtype]++;
        }
        
        std::map<tessera::DataType, float> distribution;
        for (const auto& [dtype, count] : dtype_counts) {
            distribution[dtype] = static_cast<float>(count) / traces.size();
        }
        
        return distribution;
    }
    
    float getDataTypeSpecializationBenefit(tessera::DataType dtype) {
        switch (dtype) {
            case tessera::DataType::FP8_E4M3:
                return 0.20f;  // 20% benefit for FP8 specialization
            case tessera::DataType::BF16:
                return 0.10f;  // 10% benefit for BF16 specialization
            case tessera::DataType::FP16:
                return 0.08f;  // 8% benefit for FP16 specialization
            case tessera::DataType::FP32:
                return 0.05f;  // 5% benefit for FP32 specialization
            default:
                return 0.02f;  // Minimal benefit
        }
    }
};

} // namespace tessera::metatracing::optimization
```

This comprehensive set of API examples and performance models provides a complete picture of how the Tessera Meta-Tracing PoC would be used in practice. The examples show:

1. **Basic Integration**: How to use the system with minimal configuration
2. **Advanced Configuration**: Customization options for power users
3. **Pattern Analysis**: Deep inspection of execution patterns
4. **Validation Framework**: Comprehensive testing and validation
5. **Tessera Integration**: Seamless integration with existing Tessera workflows
6. **Performance Modeling**: Theoretical prediction of optimization benefits

These examples would serve as both documentation and validation tools during the PoC development, ensuring the system meets practical usage requirements while achieving the targeted performance improvements.# Tessera Meta-Tracing PoC - API Examples and Usage Guide

## Appendix A: Complete API Examples

### A.1 Basic Usage Example

```cpp
#include "tessera/meta_tracing/meta_tracing_runtime.h"
#include "tessera/tensor.h"

int main() {
    // Initialize the meta-tracing runtime
    tessera::metatracing::RuntimeConfig config;
    config.enable_tier1_fallback = true;
    config.enable_tier2_compilation = true;
    config.enable_tier3_specialization = true;
    
    tessera::metatracing::MetaTracingRuntime runtime(config);
    
    // Create input tensors
    auto Q = tessera::randn({8, 16, 2048, 64}, tessera::DataType::FP16, "cuda:0");
    auto K = tessera::randn({8, 16, 2048, 64}, tessera::DataType::FP16, "cuda:0");  
    auto V = tessera::randn({8, 16, 2048, 64}, tessera::DataType::FP16, "cuda:0");
    auto O = tessera::zeros({8, 16, 2048, 64}, tessera::DataType::FP16, "cuda:0");
    
    // Configure attention parameters
    tessera::AttentionParams params;
    params.is_causal = true;
    params.softmax_scale = 1.0f / std::sqrt(64.0f);
    
    // Execute Flash Attention with meta-tracing
    for (int i = 0; i < 100; ++i) {
        auto result = runtime.executeFlashAttention(Q, K, V, O, params);
        
        if (!result.success) {
            std::cerr << "Execution failed: " << result.errorMessage << std::endl;
            return -1;
        }
        
        // Print performance stats every 10 iterations
        if ((i + 1) % 10 == 0) {
            auto stats = runtime.getRuntimeStats();
            std::cout << "Iteration " << (i + 1) 
                      << ": Avg time = " << stats.average_tier3_time_ms << "ms"
                      << ", Tier = " << (stats.tier3_executions > 0 ? "3" : 
                                        stats.tier2_executions > 0 ? "2" : "1")
                      << std::endl;
        }
    }
    
    // Print final performance summary
    auto stats = runtime.getRuntimeStats();
    std::cout << "\nFinal Performance Summary:" << std::endl;
    std::cout << "Total executions: " << stats.total_executions << std::endl;
    std::cout << "Tier 1 executions: " << stats.tier1_executions << std::endl;
    std::cout << "Tier 2 executions: " << stats.tier2_executions << std::endl;
    std::cout << "Tier 3 executions: " << stats.tier3_executions << std::endl;
    std::cout << "Average Tier 3 time: " << stats.average_tier3_time_ms << "ms" << std::endl;
    std::cout << "Specialized kernels: " << stats.active_specialized_kernels << std::endl;
    
    return 0;
}
```

### A.2 Advanced Configuration Example

```cpp
#include "tessera/meta_tracing/meta_tracing_runtime.h"
#include "tessera/meta_tracing/pattern_analyzer.h"
#include "tessera/meta_tracing/specialized_kernel_generator.h"

class CustomMetaTracingSetup {
public:
    void demonstrateAdvancedUsage() {
        // Create runtime with custom configuration
        tessera::metatracing::RuntimeConfig config;
        config.tier2_compilation_threshold_ms = 5.0f;    // Lower threshold
        config.tier3_specialization_threshold_ms = 50.0f; // Lower threshold  
        config.max_specialized_kernels = 32;              // More cache
        
        tessera::metatracing::MetaTracingRuntime runtime(config);
        
        // Configure custom hotspot detection
        tessera::metatracing::HotspotDetector::HotspotCriteria criteria;
        criteria.min_execution_time_ms = 0.5f;
        criteria.min_execution_count = 3;
        criteria.performance_threshold = 0.9f;  // More aggressive
        
        // Run workload with multiple configurations
        std::vector<InputConfig> configs = generateVariableWorkload();
        
        for (const auto& config : configs) {
            executeWithConfig(runtime, config);
        }
        
        // Analyze patterns and optimization opportunities
        analyzeExecutionPatterns(runtime);
        
        // Demonstrate manual specialization
        demonstrateManualSpecialization(runtime);
    }

private:
    struct InputConfig {
        int batch_size, num_heads, seq_len, head_dim;
        tessera::DataType dtype;
        bool is_causal;
        std::string description;
    };
    
    std::vector<InputConfig> generateVariableWorkload() {
        return {
            {4, 8, 512, 64, tessera::DataType::FP16, true, "Small model"},
            {4, 8, 1024, 64, tessera::DataType::FP16, true, "Small model - longer seq"},
            {8, 16, 2048, 64, tessera::DataType::FP16, true, "Medium model"},
            {8, 16, 4096, 64, tessera::DataType::FP16, true, "Medium model - longer seq"},
            {16, 32, 2048, 128, tessera::DataType::FP16, true, "Large model"},
            {32, 40, 4096, 128, tessera::DataType::BF16, true, "XL model"},
        };
    }
    
    void executeWithConfig(tessera::metatracing::MetaTracingRuntime& runtime,
                          const InputConfig& config) {
        auto Q = tessera::randn({config.batch_size, config.num_heads, 
                               config.seq_len, config.head_dim}, 
                               config.dtype, "cuda:0");
        auto K = tessera::randn({config.batch_size, config.num_heads,
                               config.seq_len, config.head_dim},
                               config.dtype, "cuda:0");
        auto V = tessera::randn({config.batch_size, config.num_heads,
                               config.seq_len, config.head_dim},
                               config.dtype, "cuda:0");
        auto O = tessera::zeros({config.batch_size, config.num_heads,
                               config.seq_len, config.head_dim},
                               config.dtype, "cuda:0");
        
        tessera::AttentionParams params;
        params.is_causal = config.is_causal;
        params.softmax_scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
        
        std::cout << "Executing: " << config.description << std::endl;
        
        // Execute multiple times to trigger optimization
        for (int i = 0; i < 20; ++i) {
            auto result = runtime.executeFlashAttention(Q, K, V, O, params);
            if (!result.success) {
                std::cerr << "Failed: " << result.errorMessage << std::endl;
                break;
            }
        }
    }
    
    void analyzeExecutionPatterns(tessera::metatracing::MetaTracingRuntime& runtime) {
        auto performance_history = runtime.getPerformanceHistory();
        
        std::cout << "\nExecution Pattern Analysis:" << std::endl;
        
        // Group by execution tier
        std::map<tessera::metatracing::ExecutionTier, std::vector<float>> tier_times;
        
        for (const auto& sample : performance_history) {
            tier_times[sample.tier_used].push_back(sample.execution_time_ms);
        }
        
        for (const auto& [tier, times] : tier_times) {
            float avg_time = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
            float min_time = *std::min_element(times.begin(), times.end());
            float max_time = *std::max_element(times.begin(), times.end());
            
            std::string tier_name = (tier == tessera::metatracing::ExecutionTier::TIER1_INTERPRETER) ? "Tier 1" :
                                   (tier == tessera::metatracing::ExecutionTier::TIER2_STANDARD) ? "Tier 2" : "Tier 3";
            
            std::cout << tier_name << ": " << times.size() << " executions, "
                      << "avg=" << avg_time << "ms, "
                      << "min=" << min_time << "ms, "
                      << "max=" << max_time << "ms" << std::endl;
        }
    }
    
    void demonstrateManualSpecialization(tessera::metatracing::MetaTracingRuntime& runtime) {
        // Access internal tracing data (for demonstration)
        auto traces = runtime.getInternalTraces(); // Hypothetical API
        
        if (traces.empty()) {
            std::cout << "No traces available for manual specialization demo" << std::endl;
            return;
        }
        
        // Create manual specialization recommendation
        tessera::metatracing::SpecializationRecommendation recommendation;
        recommendation.recommendation_type = 
            tessera::metatracing::SpecializationRecommendation::Type::SHAPE_SPECIALIZATION;
        recommendation.description = "Manual shape specialization for 8x16x2048x64";
        recommendation.expected_improvement = 0.35f; // 35% improvement
        recommendation.params.shape_params = {2048, 64}; // Fixed seq_len, head_dim
        
        std::cout << "\nManual Specialization Demo:" << std::endl;
        std::cout << "Recommendation: " << recommendation.description << std::endl;
        std::cout << "Expected improvement: " << (recommendation.expected_improvement * 100) << "%" << std::endl;
    }
};
```

### A.3 Custom Pattern Analysis Example

```cpp
#include "tessera/meta_tracing/pattern_analyzer.h"

class CustomPatternAnalyzer {
public:
    void demonstratePatternAnalysis() {
        // Create pattern analyzer with custom configuration
        tessera::metatracing::PatternAnalyzer analyzer;
        
        // Generate sample traces for analysis
        std::vector<tessera::metatracing::AttentionTrace> traces = generateSampleTraces();
        
        // Run pattern analysis
        auto analysis_result = analyzer.analyzeTraces(traces);
        
        std::cout << "Pattern Analysis Results:" << std::endl;
        std::cout << "Pattern confidence: " << analysis_result.pattern_confidence << std::endl;
        std::cout << "Supporting traces: " << analysis_result.supporting_traces << std::endl;
        std::cout << "Common patterns found: " << analysis_result.common_patterns.size() << std::endl;
        std::cout << "Optimization candidates: " << analysis_result.optimization_candidates.size() << std::endl;
        
        // Analyze each optimization candidate
        for (const auto& candidate : analysis_result.optimization_candidates) {
            analyzeOptimizationCandidate(candidate);
        }
        
        // Demonstrate custom pattern matching
        demonstrateCustomPatternMatching(traces);
    }

private:
    std::vector<tessera::metatracing::AttentionTrace> generateSampleTraces() {
        std::vector<tessera::metatracing::AttentionTrace> traces;
        
        // Generate traces with varying characteristics
        for (int i = 0; i < 50; ++i) {
            tessera::metatracing::AttentionTrace trace;
            
            // Simulate common input patterns
            if (i % 3 == 0) {
                // Small model pattern
                trace.input_profile = {8, 16, 1024, 64, tessera::DataType::FP16, true, 0.125f};
                trace.execution_pattern.execution_time_ms = 2.5f + (i % 10) * 0.1f;
            } else if (i % 3 == 1) {
                // Medium model pattern  
                trace.input_profile = {16, 32, 2048, 64, tessera::DataType::FP16, true, 0.125f};
                trace.execution_pattern.execution_time_ms = 8.2f + (i % 10) * 0.3f;
            } else {
                // Large model pattern
                trace.input_profile = {32, 40, 4096, 128, tessera::DataType::BF16, true, 0.089f};
                trace.execution_pattern.execution_time_ms = 25.1f + (i % 10) * 0.8f;
            }
            
            // Simulate execution characteristics
            trace.execution_pattern.memory_bandwidth_gbps = 1200.0f + (i % 20) * 10.0f;
            trace.execution_pattern.compute_utilization = 0.75f + (i % 10) * 0.02f;
            trace.execution_pattern.occupancy_achieved = 0.85f + (i % 15) * 0.01f;
            
            trace.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
            trace.execution_count = 1;
            
            traces.push_back(trace);
        }
        
        return traces;
    }
    
    void analyzeOptimizationCandidate(const tessera::metatracing::OptimizationCandidate& candidate) {
        std::cout << "\nOptimization Candidate:" << std::endl;
        std::cout << "  Type: " << getOptimizationTypeName(candidate.type) << std::endl;
        std::cout << "  Expected improvement: " << (candidate.expected_improvement * 100) << "%" << std::endl;
        std::cout << "  Implementation complexity: " << candidate.implementation_complexity << std::endl;
        std::cout << "  Description: " << candidate.description << std::endl;
        
        // Analyze specific optimization parameters
        switch (candidate.type) {
            case tessera::metatracing::OptimizationCandidate::Type::TILE_SIZE_OPTIMIZATION:
                std::cout << "  Recommended tile size: " 
                          << candidate.params.tile_params.block_m << "x"
                          << candidate.params.tile_params.block_n << "x"
                          << candidate.params.tile_params.block_k << std::endl;
                break;
                
            case tessera::metatracing::OptimizationCandidate::Type::SHAPE_SPECIALIZATION:
                std::cout << "  Specialized shapes: seq_len=" 
                          << candidate.params.shape_params.fixed_seq_len
                          << ", head_dim=" << candidate.params.shape_params.fixed_head_dim << std::endl;
                break;
                
            case tessera::metatracing::OptimizationCandidate::Type::DTYPE_SPECIALIZATION:
                std::cout << "  Specialized dtype: " 
                          << getDtypeString(candidate.params.dtype_params.specialized_dtype) << std::endl;
                break;
                
            default:
                std::cout << "  Generic optimization parameters" << std::endl;
                break;
        }
    }
    
    void demonstrateCustomPatternMatching(const std::vector<tessera::metatracing::AttentionTrace>& traces) {
        std::cout << "\nCustom Pattern Matching Demo:" << std::endl;
        
        // Find sequences of similar input shapes
        auto shape_sequences = findShapeSequences(traces);
        std::cout << "Shape sequence patterns found: " << shape_sequences.size() << std::endl;
        
        // Analyze performance trends
        auto performance_trends = analyzePerformanceTrends(traces);
        std::cout << "Performance trend correlation: " << performance_trends.correlation_coefficient << std::endl;
        
        // Detect memory access patterns
        auto memory_patterns = detectMemoryPatterns(traces);
        std::cout << "Memory access patterns identified: " << memory_patterns.size() << std::endl;
    }
    
    struct ShapeSequence {
        std::vector<tessera::metatracing::AttentionTrace::InputProfile> sequence;
        int frequency;
        float average_performance;
    };
    
    std::vector<ShapeSequence> findShapeSequences(const std::vector<tessera::metatracing::AttentionTrace>& traces) {
        std::map<std::vector<int>, ShapeSequence> sequence_map;
        
        // Sliding window approach to find common sequences
        const int window_size = 3;
        for (size_t i = 0; i <= traces.size() - window_size; ++i) {
            std::vector<int> shape_signature;
            float total_performance = 0.0f;
            
            for (int j = 0; j < window_size; ++j) {
                const auto& profile = traces[i + j].input_profile;
                shape_signature.push_back(profile.seq_len);
                shape_signature.push_back(profile.head_dim);
                total_performance += traces[i + j].execution_pattern.compute_utilization;
            }
            
            if (sequence_map.find(shape_signature) == sequence_map.end()) {
                ShapeSequence seq;
                for (int j = 0; j < window_size; ++j) {
                    seq.sequence.push_back(traces[i + j].input_profile);
                }
                seq.frequency = 1;
                seq.average_performance = total_performance / window_size;
                sequence_map[shape_signature] = seq;
            } else {
                sequence_map[shape_signature].frequency++;
                sequence_map[shape_signature].average_performance = 
                    (sequence_map[shape_signature].average_performance * (sequence_map[shape_signature].frequency - 1) +
                     total_performance / window_size) / sequence_map[shape_signature].frequency;
            }
        }
        
        std::vector<ShapeSequence> result;
        for (const auto& [signature, sequence] : sequence_map) {
            if (sequence.frequency >= 2) {  // Only include repeated sequences
                result.push_back(sequence);
            }
        }
        
        return result;
    }
    
    struct PerformanceTrend {
        float correlation_coefficient;
        std::string trend_description;
        float confidence_level;
    };
    
    PerformanceTrend analyzePerformanceTrends(const std::vector<tessera::metatracing::AttentionTrace>& traces) {
        // Analyze correlation between input size and performance
        std::vector<float> input_sizes;
        std::vector<float> performance_values;
        
        for (const auto& trace : traces) {
            float input_size = static_cast<float>(
                trace.input_profile.batch_size * 
                trace.input_profile.num_heads * 
                trace.input_profile.seq_len * 
                trace.input_profile.head_dim
            );
            input_sizes.push_back(input_size);
            performance_values.push_back(trace.execution_pattern.compute_utilization);
        }
        
        float correlation = calculateCorrelation(input_sizes, performance_values);
        
        PerformanceTrend trend;
        trend.correlation_coefficient = correlation;
        trend.confidence_level = std::abs(correlation);
        
        if (correlation > 0.5f) {
            trend.trend_description = "Performance improves with larger inputs (good cache utilization)";
        } else if (correlation < -0.5f) {
            trend.trend_description = "Performance degrades with larger inputs (memory bound)";
        } else {
            trend.trend_description = "No clear correlation between input size and performance";
        }
        
        return trend;
    }
    
    std::vector<std::string> detectMemoryPatterns(const std::vector<tessera::metatracing::AttentionTrace>& traces) {
        std::vector<std::string> patterns;
        
        // Analyze memory bandwidth utilization patterns
        float avg_bandwidth = 0.0f;
        float max_bandwidth = 0.0f;
        float min_bandwidth = std::numeric_limits<float>::max();
        
        for (const auto& trace : traces) {
            float bandwidth = trace.execution_pattern.memory_bandwidth_gbps;
            avg_bandwidth += bandwidth;
            max_bandwidth = std::max(max_bandwidth, bandwidth);
            min_bandwidth = std::min(min_bandwidth, bandwidth);
        }
        avg_bandwidth /= traces.size();
        
        if ((max_bandwidth - min_bandwidth) / avg_bandwidth < 0.1f) {
            patterns.push_back("Consistent memory bandwidth utilization");
        } else {
            patterns.push_back("Variable memory bandwidth utilization");
        }
        
        // Analyze occupancy patterns
        float avg_occupancy = 0.0f;
        for (const auto& trace : traces) {
            avg_occupancy += trace.execution_pattern.occupancy_achieved;
        }
        avg_occupancy /= traces.size();
        
        if (avg_occupancy > 0.9f) {
            patterns.push_back("High occupancy achieved consistently");
        } else if (avg_occupancy > 0.7f) {
            patterns.push_back("Good occupancy with room for improvement");
        } else {
            patterns.push_back("Low occupancy - potential register or shared memory pressure");
        }
        
        return patterns;
    }
    
    float calculateCorrelation(const std::vector<float>& x, const std::vector<float>& y) {
        if (x.size() != y.size() || x.empty()) return 0.0f;
        
        float mean_x = 0.0f, mean_y = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            mean_x += x[i];
            mean_y += y[i];
        }
        mean_x /= x.size();
        mean_y /= y.size();
        
        float numerator = 0.0f, denom_x = 0.0f, denom_y = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            float dx = x[i] - mean_x;
            float dy = y[i] - mean_y;
            numerator += dx * dy;
            denom_x += dx * dx;
            denom_y += dy * dy;
        }
        
        float denominator = std::sqrt(denom_x * denom_y);
        return (denominator > 0.0f) ? (numerator / denominator) : 0.0f;
    }
    
    std::string getOptimizationTypeName(tessera::metatracing::OptimizationCandidate::Type type) {
        switch (type) {
            case tessera::metatracing::OptimizationCandidate::Type::TILE_SIZE_OPTIMIZATION:
                return "Tile Size Optimization";
            case tessera::metatracing::OptimizationCandidate::Type::SHAPE_SPECIALIZATION:
                return "Shape Specialization";
            case tessera::metatracing::OptimizationCandidate::Type::DTYPE_SPECIALIZATION:
                return "Data Type Specialization";
            case tessera::metatracing::OptimizationCandidate::Type::MEMORY_LAYOUT_OPTIMIZATION:
                return "Memory Layout Optimization";
            case tessera::metatracing::OptimizationCandidate::Type::FUSION_OPTIMIZATION:
                return "Operation Fusion";
            default:
                return "Unknown Optimization";
        }
    }
    
    std::string getDtypeString(tessera::DataType dtype) {
        switch (dtype) {
            case tessera::DataType::FP16: return "FP16";
            case tessera::DataType::BF16: return "BF16";
            case tessera::DataType::FP32: return "FP32";
            case tessera::DataType::FP8_E4M3: return "FP8_E4M3";
            default: return "Unknown";
        }
    }
};
```

### A.4 Benchmarking and Validation Example

```cpp
#include "tessera/meta_tracing/benchmark_suite.h"
#include "tessera/meta_tracing/validation_framework.h"

class ComprehensiveBenchmarkDemo {
public:
    void runCompleteBenchmarkSuite() {
        // Initialize runtime
        tessera::metatracing::MetaTracingRuntime runtime;
        
        // Configure benchmark suite
        tessera::metatracing::PoC_BenchmarkSuite::BenchmarkConfig config;
        config.warmup_iterations = 5;
        config.benchmark_iterations = 50;
        config.enable_statistical_analysis = true;
        config.confidence_interval = 0.95f;
        
        // Add diverse input configurations
        config.input_configs = createComprehensiveTestConfigs();
        
        std::cout << "Running comprehensive benchmark suite with " 
                  << config.input_configs.size() << " configurations..." << std::endl;
        
        // Run benchmarks
        tessera::metatracing::PoC_BenchmarkSuite benchmark_suite;
        auto results = benchmark_suite.runBenchmarkSuite(runtime, config);
        
        // Analyze results
        analyzeAndReportResults(results);
        
        // Run validation
        runValidationSuite(runtime);
        
        // Generate performance report
        generatePerformanceReport(results);
    }

private:
    std::vector<tessera::metatracing::PoC_BenchmarkSuite::InputConfiguration> createComprehensiveTestConfigs() {
        std::vector<tessera::metatracing::PoC_BenchmarkSuite::InputConfiguration> configs;
        
        // Small configurations
        auto small_configs = tessera::metatracing::StandardBenchmarkConfigs::getSmallConfigs();
        configs.insert(configs.end(), small_configs.begin(), small_configs.end());
        
        // Medium configurations
        auto medium_configs = tessera::metatracing::StandardBenchmarkConfigs::getMediumConfigs();
        configs.insert(configs.end(), medium_configs.begin(), medium_configs.end());
        
        // Large configurations  
        auto large_configs = tessera::metatracing::StandardBenchmarkConfigs::getLargeConfigs();
        configs.insert(configs.end(), large_configs.begin(), large_configs.end());
        
        // Variable length configurations
        auto variable_configs = tessera::metatracing::StandardBenchmarkConfigs::getVariableLengthConfigs();
        configs.insert(configs.end(), variable_configs.begin(), variable_configs.end());
        
        // Mixed precision configurations
        auto mixed_configs = tessera::metatracing::StandardBenchmarkConfigs::getMixedPrecisionConfigs();
        configs.insert(configs.end(), mixed_configs.begin(), mixed_configs.end());
        
        return configs;
    }
    
    void analyzeAndReportResults(const std::vector<tessera::metatracing::PoC_BenchmarkSuite::BenchmarkResult>& results) {
        std::cout << "\n=== Benchmark Results Analysis ===" << std::endl;
        
        float total_tier2_speedup = 0.0f;
        float total_tier3_speedup = 0.0f;
        int significant_tier2_improvements = 0;
        int significant_tier3_improvements = 0;
        
        for (const auto& result : results) {
            std::cout << "\nConfiguration: " << result.input_config.description << std::endl;
            std::cout << "Tier 1 performance: " << result.tier1_metrics.mean_time_ms << " ± " 
                      << result.tier1_metrics.std_dev_ms << " ms" << std::endl;
            std::cout << "Tier 2 performance: " << result.tier2_metrics.mean_time_ms << " ± "
                      << result.tier2_metrics.std_dev_ms << " ms" << std::endl;
            std::cout << "Tier 3 performance: " << result.tier3_metrics.mean_time_ms << " ± "
                      << result.tier3_metrics.std_dev_ms << " ms" << std::endl;
            
            std::cout << "Tier 2 speedup: " << result.tier2_vs_tier1_speedup << "x";
            if (result.tier2_improvement_significant) {
                std::cout << " (significant, p=" << result.p_value_tier2 << ")";
                significant_tier2_improvements++;
            }
            std::cout << std::endl;
            
            std::cout << "Tier 3 speedup: " << result.tier3_vs_tier1_speedup << "x";
            if (result.tier3_improvement_significant) {
                std::cout << " (significant, p=" << result.p_value_tier3 << ")";
                significant_tier3_improvements++;
            }
            std::cout << std::endl;
            
            std::cout << "TFLOPs achieved - Tier 1: " << result.tier1_metrics.tflops_achieved
                      << ", Tier 2: " << result.tier2_metrics.tflops_achieved
                      << ", Tier 3: " << result.tier3_metrics.tflops_achieved << std::endl;
            
            total_tier2_speedup += result.tier2_vs_tier1_speedup;
            total_tier3_speedup += result.tier3_vs_tier1_speedup;
        }
        
        // Summary statistics
        std::cout << "\n=== Summary Statistics ===" << std::endl;
        std::cout << "Average Tier 2 speedup: " << (total_tier2_speedup / results.size()) << "x" << std::endl;
        std::cout << "Average Tier 3 speedup: " << (total_tier3_speedup / results.size()) << "x" << std::endl;
        std::cout << "Significant Tier 2 improvements: " << significant_tier2_improvements 
                  << "/" << results.size() << " (" << (100.0f * significant_tier2_improvements / results.size()) << "%)" << std::endl;
        std::cout << "Significant Tier 3 improvements: " << significant_tier3_improvements
                  << "/" << results.size() << " (" << (100.0f * significant_tier3_improvements / results.size()) << "%)" << std::endl;
    }
    
    void runValidationSuite(tessera::metatracing::MetaTracingRuntime& runtime) {
        std::cout << "\n=== Running Validation Suite ===" << std::endl;
        
        tessera::metatracing::ValidationFramework validator;
        tessera::metatracing::ValidationFramework::ValidationConfig config;
        config.numerical_tolerance = 1e-4f;
        config.enable_cross_tier_validation = true;
        config.enable_reference_validation = true;
        config.max_acceptable_regression_percent = 5.0f;
        
        auto validation_result = validator.validateImplementation(runtime, config);
        
        std::cout << "Numerical correctness: " << (validation_result.numerical_correctness_passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Performance regression: " << (validation_result.performance_regression_passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Cross-tier consistency: " << (validation_result.cross_tier_consistency_passed ? "PASSED" : "FAILED") << std::endl;
        
        std::cout << "Max absolute error: " << validation_result.max_absolute_error << std::endl;
        std::cout << "Max relative error: " << validation_result.max_relative_error << std::endl;
        std::cout << "Performance regression: " << validation_result.performance_regression_percent << "%" << std::endl;
        
        if (!validation_result.issues.empty()) {
            std::cout << "\nValidation Issues:" << std::endl;
            for (const auto& issue : validation_result.issues) {
                std::cout << "  " << getSeverityString(issue.severity) << ": " << issue.description << std::endl;
                std::cout << "    Suggested fix: " << issue.suggested_fix << std::endl;
            }
        }
    }
    
    void generatePerformanceReport(const std::vector<tessera::metatracing::PoC_BenchmarkSuite::BenchmarkResult>& results) {
        std::cout << "\n=== Generating Performance Report ===" << std::endl;
        
        std::ofstream report("performance_report.md");
        report << "# Tessera Meta-Tracing PoC Performance Report\n\n";
        report << "## Executive Summary\n\n";
        
        float avg_tier2_speedup = 0.0f;
        float avg_tier3_speedup = 0.0f;
        float max_tier3_speedup = 0.0f;
        
        for (const auto& result : results) {
            avg_tier2_speedup += result.tier2_vs_tier1_speedup;
            avg_tier3_speedup += result.tier3_vs_tier1_speedup;
            max_tier3_speedup = std::max(max_tier3_speedup, result.tier3_vs_tier1_speedup);
        }
        
        avg_tier2_speedup /= results.size();
        avg_tier3_speedup /= results.size();
        
        report << "- Average Tier 2 speedup: **" << avg_tier2_speedup << "x**\n";
        report << "- Average Tier 3 speedup: **" << avg_tier3_speedup << "x**\n";
        report << "- Maximum Tier 3 speedup: **" << max_tier3_speedup << "x**\n";
        report << "- Configurations tested: **" << results.size() << "**\n\n";
        
        report << "## Detailed Results\n\n";
        report << "| Configuration | Tier 1 (ms) | Tier 2 (ms) | Tier 3 (ms) | T2 Speedup | T3 Speedup | T3 TFLOPs |\n";
        report << "|---------------|--------------|--------------|--------------|------------|------------|----------|\n";
        
        for (const auto& result : results) {
            report << "| " << result.input_config.description 
                   << " | " << std::fixed << std::setprecision(2) << result.tier1_metrics.mean_time_ms
                   << " | " << result.tier2_metrics.mean_time_ms
                   << " | " << result.tier3_metrics.mean_time_ms
                   << " | " << std::setprecision(2) << result.tier2_vs_tier1_speedup << "x"
                   << " | " << result.tier3_vs_tier1_speedup << "x"  
                   << " | " << std::setprecision(1) << result.tier3_metrics.tflops_achieved
                   << " |\n";
        }
        
        report.close();
        std::cout << "Performance report written to performance_report.md" << std::endl;
    }
    
    std::string getSeverityString(tessera::metatracing::ValidationFramework::ValidationIssue::Severity severity) {
        switch (severity) {
            case tessera::metatracing::ValidationFramework::ValidationIssue::WARNING: return "WARNING";
            case tessera::metatracing::ValidationFramework::ValidationIssue::ERROR: return "ERROR";
            case tessera::metatracing::ValidationFramework::ValidationIssue::CRITICAL: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }
};
```

### A.5 Integration with Existing Tessera Code

```cpp
#include "tessera/core/tensor.h"
#include "tessera/ops/attention.h"
#include "tessera/meta_tracing/meta_tracing_runtime.h"

class TesseraMetaTracingIntegration {
public:
    // Seamless integration with existing Tessera attention operations
    void demonstrateSeamlessIntegration() {
        // Standard Tessera tensor creation
        auto Q = tessera::randn({8, 16, 2048, 64}, tessera::dtype::bf16).cuda();
        auto K = tessera::randn({8, 16, 2048, 64}, tessera::dtype::bf16).cuda();
        auto V = tessera::randn({8, 16, 2048, 64}, tessera::dtype::bf16).cuda();
        
        // Option 1: Use existing Tessera ops (baseline)
        auto O1 = tessera::ops::scaled_dot_product_attention(Q, K, V, /*is_causal=*/true);
        
        // Option 2: Use meta-tracing enhanced version (new)
        tessera::metatracing::MetaTracingRuntime runtime;
        auto O2 = tessera::zeros_like(Q);
        
        tessera::AttentionParams params;
        params.is_causal = true;
        params.softmax_scale = 1.0f / std::sqrt(64.0f);
        
        auto result = runtime.executeFlashAttention(Q, K, V, O2, params);
        
        // Verify numerical equivalence
        auto diff = tessera::abs(O1 - O2);
        auto max_diff = tessera::max(diff);
        
        std::cout << "Maximum difference between standard and meta-tracing: " 
                  << max_diff.item<float>() << std::endl;
        
        assert(max_diff.item<float>() < 1e-3f);  // Should be numerically equivalent
    }
    
    // Integration with Tessera's autograd system
    void demonstrateAutogradIntegration() {
        // Enable gradients for input tensors
        auto Q = tessera::randn({4, 8, 1024, 64}, tessera::dtype::bf16).cuda().requires_grad_(true);
        auto K = tessera::randn({4, 8, 1024, 64}, tessera::dtype::bf16).cuda().requires_grad_(true);
        auto V = tessera::randn({4, 8, 1024, 64}, tessera::dtype::bf16).cuda().requires_grad_(true);
        
        tessera::metatracing::MetaTracingRuntime runtime;
        auto O = tessera::zeros_like(Q);
        
        // Forward pass with meta-tracing
        tessera::AttentionParams params;
        params.is_causal = true;
        params.softmax_scale = 1.0f / std::sqrt(64.0f);
        
        auto result = runtime.executeFlashAttention(Q, K, V, O, params);
        
        // Compute loss and backward pass
        auto target = tessera::randn_like(O);
        auto loss = tessera::mse_loss(O, target);
        
        loss.backward();
        
        // Verify gradients are computed correctly
        assert(Q.grad().defined());
        assert(K.grad().defined());
        assert(V.grad().defined());
        
        std::cout << "Autograd integration successful!" << std::endl;
        std::cout << "Q gradient norm: " << tessera::norm(Q.grad()).item<float>() << std::endl;
        std::cout << "K gradient norm: " << tessera::norm(K.grad()).item<float>() << std::endl;
        std::cout << "V gradient norm: " << tessera::norm(V.grad()).item<float>() << std::endl;
    }
    
    // Integration with Tessera's distributed training
    void demonstrateDistributedIntegration() {
        // Initialize distributed environment (conceptual)
        tessera::distributed::initialize_process_group("nccl");
        
        int world_size = tessera::distributed::get_world_size();
        int rank = tessera::distributed::get_rank();
        
        std::cout << "Running on rank " << rank << " of " << world_size << std::endl;
        
        // Create tensors distributed across ranks
        auto Q = tessera::randn({2, 8, 2048, 64}, tessera::dtype::bf16).cuda();
        auto K = tessera::randn({2, 8, 2048, 64}, tessera::dtype::bf16).cuda();
        auto V = tessera::randn({2, 8, 2048, 64}, tessera::dtype::bf16).cuda();
        
        // Meta-tracing runtime with distributed awareness
        tessera::metatracing::RuntimeConfig config;
        config.enable_distributed_tracing = true;
        config.distributed_rank = rank;
        config.distributed_world_size = world_size;
        
        tessera::metatracing::MetaTracingRuntime runtime(config);
        auto O = tessera::zeros_like(Q);
        
        tessera::AttentionParams params;
        params.is_causal = true;
        params.softmax_scale = 1.0f / std::sqrt(64.0f);
        
        // Execute with distributed meta-tracing
        auto result = runtime.executeFlashAttention(Q, K, V, O, params);
        
        