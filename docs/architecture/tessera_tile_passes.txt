//===- TesseraTilePasses.h - Tessera Tile IR Passes -*- C++ -*-===//
//
// This file defines transformation passes for Tessera's Tile IR dialect.
// These passes perform low-level optimizations, memory hierarchy management,
// and target-specific code generation preparation.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TILE_PASSES_H
#define TESSERA_TILE_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "TesseraTileTypes.h"

namespace mlir {
namespace tessera {
namespace tile {

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_DECL
#include "Tessera/Tile/TesseraTilePasses.h.inc"

//===----------------------------------------------------------------------===//
// Memory Hierarchy Optimization Passes
//===----------------------------------------------------------------------===//

/// Pass to optimize memory allocation and placement across hierarchy levels.
/// This pass analyzes memory access patterns and moves allocations to the
/// optimal memory space (global, shared, register, TMEM).
std::unique_ptr<Pass> createMemoryHierarchyOptimizationPass();

/// Pass to eliminate bank conflicts in shared memory accesses.
/// Applies swizzle patterns and layout transformations to maximize bandwidth.
std::unique_ptr<Pass> createBankConflictEliminationPass();

/// Pass to insert cooperative memory copy operations.
/// Replaces individual loads/stores with optimized cooperative copies.
std::unique_ptr<Pass> createCooperativeMemoryPass();

/// Pass to optimize memory coalescing for global memory accesses.
/// Reorders operations and adjusts thread mappings for optimal coalescing.
std::unique_ptr<Pass> createMemoryCoalescingPass();

//===----------------------------------------------------------------------===//
// Thread and Warp Optimization Passes  
//===----------------------------------------------------------------------===//

/// Pass to optimize thread mappings for compute operations.
/// Analyzes workload and generates optimal thread/warp configurations.
std::unique_ptr<Pass> createThreadMappingOptimizationPass();

/// Pass to eliminate thread divergence where possible.
/// Reorders operations to minimize divergent execution paths.
std::unique_ptr<Pass> createDivergenceMinimizationPass();

/// Pass to optimize warp-level operations.
/// Generates efficient warp shuffles, reductions, and broadcasts.
std::unique_ptr<Pass> createWarpLevelOptimizationPass();

/// Pass to insert optimal synchronization barriers.
/// Minimizes synchronization overhead while maintaining correctness.
std::unique_ptr<Pass> createSynchronizationOptimizationPass();

//===----------------------------------------------------------------------===//
// Compute Optimization Passes
//===----------------------------------------------------------------------===//

/// Pass to optimize GEMM operations for Tensor Cores.
/// Converts suitable operations to use hardware acceleration.
std::unique_ptr<Pass> createTensorCoreOptimizationPass();

/// Pass to generate optimal fragment operations.
/// Converts memory operations to register fragment patterns.
std::unique_ptr<Pass> createFragmentOptimizationPass();

/// Pass to optimize reduction operations.
/// Generates efficient tree reductions and warp shuffles.
std::unique_ptr<Pass> createReductionOptimizationPass();

/// Pass to fuse elementwise operations.
/// Combines multiple elementwise ops to reduce memory traffic.
std::unique_ptr<Pass> createElementwiseFusionPass();

//===----------------------------------------------------------------------===//
// Pipeline and Scheduling Passes
//===----------------------------------------------------------------------===//

/// Pass to create software pipelines for overlapping computation and memory.
/// Analyzes dependencies and generates multi-stage pipelines.
std::unique_ptr<Pass> createSoftwarePipeliningPass();

/// Pass to optimize instruction scheduling within tiles.
/// Reorders instructions to minimize latency and maximize throughput.
std::unique_ptr<Pass> createInstructionSchedulingPass();

/// Pass to optimize register allocation and spilling.
/// Minimizes register usage while maintaining performance.
std::unique_ptr<Pass> createRegisterOptimizationPass();

/// Pass to insert prefetch operations for predictable access patterns.
/// Overlaps memory fetches with computation to hide latency.
std::unique_ptr<Pass> createPrefetchOptimizationPass();

//===----------------------------------------------------------------------===//
// Hardware-Specific Passes
//===----------------------------------------------------------------------===//

/// Pass to optimize for NVIDIA Hopper architecture (sm_90).
/// Enables Hopper-specific features like TMA and distributed shared memory.
std::unique_ptr<Pass> createHopperOptimizationPass();

/// Pass to optimize for NVIDIA Blackwell architecture (sm_100).
/// Enables Blackwell-specific features like TMEM and CTA pairs.
std::unique_ptr<Pass> createBlackwellOptimizationPass();

/// Pass to optimize for AMD RDNA3+ architectures.
/// Generates optimized code for AMD GPU compute units.
std::unique_ptr<Pass> createAMDOptimizationPass();

/// Pass to prepare code for CuTe kernel generation.
/// Transforms Tile IR to patterns compatible with CuTe templates.
std::unique_ptr<Pass> createCuTePreparationPass();

//===----------------------------------------------------------------------===//
// Layout and Access Pattern Passes
//===----------------------------------------------------------------------===//

/// Pass to optimize tensor layouts for access patterns.
/// Chooses optimal data layouts (row/column major, tiled, etc.).
std::unique_ptr<Pass> createLayoutOptimizationPass();

/// Pass to apply swizzle patterns for bank conflict avoidance.
/// Generates hardware-specific swizzle transformations.
std::unique_ptr<Pass> createSwizzleOptimizationPass();

/// Pass to optimize vectorization opportunities.
/// Groups scalar operations into vector operations where beneficial.
std::unique_ptr<Pass> createVectorizationPass();

/// Pass to analyze and optimize for cache locality.
/// Reorders operations to improve L1/L2 cache utilization.
std::unique_ptr<Pass> createCacheLocalityOptimizationPass();

//===----------------------------------------------------------------------===//
// Code Generation Preparation Passes
//===----------------------------------------------------------------------===//

/// Pass to prepare Tile IR for CUDA PTX generation.
/// Transforms operations to PTX-compatible patterns.
std::unique_ptr<Pass> createCUDACodegenPreparationPass();

/// Pass to prepare Tile IR for HIP code generation.
/// Transforms operations for AMD ROCm compatibility.
std::unique_ptr<Pass> createHIPCodegenPreparationPass();

/// Pass to prepare Tile IR for Triton compilation.
/// Converts to Triton-compatible operation patterns.
std::unique_ptr<Pass> createTritonCodegenPreparationPass();

/// Pass to lower Tile IR to target-specific IR (GPU, LLVM).
/// Final lowering step before target code generation.
std::unique_ptr<Pass> createTileIRLoweringPass();

//===----------------------------------------------------------------------===//
// Analysis and Verification Passes
//===----------------------------------------------------------------------===//

/// Pass to analyze resource usage (registers, shared memory).
/// Provides detailed resource utilization reports.
std::unique_ptr<Pass> createResourceAnalysisPass();

/// Pass to estimate performance characteristics.
/// Generates performance models and bottleneck analysis.
std::unique_ptr<Pass> createPerformanceAnalysisPass();

/// Pass to verify hardware constraints and compatibility.
/// Ensures generated code meets target architecture requirements.
std::unique_ptr<Pass> createHardwareVerificationPass();

/// Pass to validate memory coherence and synchronization.
/// Checks for race conditions and memory ordering issues.
std::unique_ptr<Pass> createMemoryCoherenceVerificationPass();

//===----------------------------------------------------------------------===//
// Autotuning Integration Passes
//===----------------------------------------------------------------------===//

/// Pass to insert autotuning parameters for tile sizes.
/// Generates parameterized code for runtime optimization.
std::unique_ptr<Pass> createAutotuningParameterizationPass();

/// Pass to generate autotuning search space.
/// Creates configuration space for automated optimization.
std::unique_ptr<Pass> createAutotuningSearchSpaceGenerationPass();

/// Pass to apply autotuned parameters.
/// Substitutes optimal parameters discovered through autotuning.
std::unique_ptr<Pass> createAutotuningParameterApplicationPass();

//===----------------------------------------------------------------------===//
// Debugging and Profiling Passes
//===----------------------------------------------------------------------===//

/// Pass to insert profiling instrumentation.
/// Adds timing and performance counters to generated kernels.
std::unique_ptr<Pass> createProfilingInstrumentationPass();

/// Pass to insert debugging information.
/// Adds source location and variable information for debugging.
std::unique_ptr<Pass> createDebuggingInstrumentationPass();

/// Pass to generate kernel launch configuration.
/// Computes optimal grid and block dimensions.
std::unique_ptr<Pass> createLaunchConfigurationPass();

//===----------------------------------------------------------------------===//
// Pass Pipeline Construction
//===----------------------------------------------------------------------===//

/// Build the standard Tile IR optimization pipeline.
/// Includes memory optimization, compute optimization, and code generation prep.
void buildTileIROptimizationPipeline(mlir::PassManager& pm,
                                      const TileIRPipelineOptions& options);

/// Build hardware-specific optimization pipeline.
/// Tailored for specific target architectures (Hopper, Blackwell, RDNA3).
void buildHardwareSpecificPipeline(mlir::PassManager& pm,
                                   llvm::StringRef target_architecture,
                                   const TileIRPipelineOptions& options);

/// Build debugging and profiling pipeline.
/// Includes instrumentation and verification passes.
void buildDebuggingPipeline(mlir::PassManager& pm,
                            const TileIRPipelineOptions& options);

/// Build autotuning pipeline.
/// Includes parameterization and search space generation.
void buildAutotuningPipeline(mlir::PassManager& pm,
                             const TileIRPipelineOptions& options);

//===----------------------------------------------------------------------===//
// Pass Options
//===----------------------------------------------------------------------===//

struct TileIRPipelineOptions : public mlir::PassPipelineOptions<TileIRPipelineOptions> {
  // Target architecture
  Option<std::string> targetArchitecture{
    *this, "target-arch", 
    llvm::cl::desc("Target GPU architecture (sm_90, sm_100, gfx1100)"),
    llvm::cl::init("sm_90")
  };
  
  // Memory optimization options
  Option<bool> enableMemoryOptimization{
    *this, "enable-memory-opt",
    llvm::cl::desc("Enable memory hierarchy optimization"),
    llvm::cl::init(true)
  };
  
  Option<bool> enableBankConflictElimination{
    *this, "enable-bank-conflict-elimination",
    llvm::cl::desc("Enable shared memory bank conflict elimination"),
    llvm::cl::init(true)
  };
  
  Option<bool> enableCooperativeMemory{
    *this, "enable-cooperative-memory",
    llvm::cl::desc("Enable cooperative memory operations"),
    llvm::cl::init(true)
  };
  
  // Compute optimization options
  Option<bool> enableTensorCores{
    *this, "enable-tensor-cores",
    llvm::cl::desc("Enable Tensor Core optimization"),
    llvm::cl::init(true)
  };
  
  Option<bool> enableFragmentOptimization{
    *this, "enable-fragment-opt",
    llvm::cl::desc("Enable register fragment optimization"),
    llvm::cl::init(true)
  };
  
  Option<bool> enableElementwiseFusion{
    *this, "enable-elementwise-fusion",
    llvm::cl::desc("Enable elementwise operation fusion"),
    llvm::cl::init(true)
  };
  
  // Pipeline optimization options
  Option<bool> enableSoftwarePipelining{
    *this, "enable-software-pipelining",
    llvm::cl::desc("Enable software pipelining"),
    llvm::cl::init(true)
  };
  
  Option<int> maxPipelineStages{
    *this, "max-pipeline-stages",
    llvm::cl::desc("Maximum software pipeline stages"),
    llvm::cl::init(3)
  };
  
  // Thread optimization options
  Option<bool> enableThreadMappingOptimization{
    *this, "enable-thread-mapping-opt",
    llvm::cl::desc("Enable thread mapping optimization"),
    llvm::cl::init(true)
  };
  
  Option<bool> enableDivergenceMinimization{
    *this, "enable-divergence-minimization",
    llvm::cl::desc("Enable thread divergence minimization"),
    llvm::cl::init(true)
  };
  
  // Hardware-specific options
  Option<bool> enableHardwareSpecificOpt{
    *this, "enable-hardware-specific-opt",
    llvm::cl::desc("Enable hardware-specific optimizations"),
    llvm::cl::init(true)
  };
  
  Option<bool> enableTMEM{
    *this, "enable-tmem",
    llvm::cl::desc("Enable Tensor Memory (TMEM) on Blackwell"),
    llvm::cl::init(false)
  };
  
  Option<bool> enableCTAPairs{
    *this, "enable-cta-pairs",
    llvm::cl::desc("Enable CTA pair coordination on Blackwell"),
    llvm::cl::init(false)
  };
  
  // Code generation options
  Option<std::string> codegenBackend{
    *this, "codegen-backend",
    llvm::cl::desc("Code generation backend (cuda, hip, triton, cute)"),
    llvm::cl::init("cuda")
  };
  
  Option<bool> enableCuTeGeneration{
    *this, "enable-cute-generation",
    llvm::cl::desc("Enable CuTe kernel generation"),
    llvm::cl::init(false)
  };
  
  // Debugging and profiling options
  Option<bool> enableProfiling{
    *this, "enable-profiling",
    llvm::cl::desc("Enable profiling instrumentation"),
    llvm::cl::init(false)
  };
  
  Option<bool> enableDebugging{
    *this, "enable-debugging",
    llvm::cl::desc("Enable debugging instrumentation"),
    llvm::cl::init(false)
  };
  
  Option<bool> enableVerification{
    *this, "enable-verification",
    llvm::cl::desc("Enable hardware verification passes"),
    llvm::cl::init(true)
  };
  
  // Autotuning options
  Option<bool> enableAutotuning{
    *this, "enable-autotuning",
    llvm::cl::desc("Enable autotuning parameter generation"),
    llvm::cl::init(false)
  };
  
  Option<std::string> autotuningSearchSpace{
    *this, "autotuning-search-space",
    llvm::cl::desc("Autotuning search space configuration file"),
    llvm::cl::init("")
  };
  
  Option<int> autotuningBudget{
    *this, "autotuning-budget",
    llvm::cl::desc("Autotuning evaluation budget"),
    llvm::cl::init(100)
  };
  
  // Performance optimization options
  Option<int> optimizationLevel{
    *this, "opt-level",
    llvm::cl::desc("Optimization level (0-3)"),
    llvm::cl::init(2)
  };
  
  Option<bool> enableAggressiveOptimizations{
    *this, "enable-aggressive-opt",
    llvm::cl::desc("Enable aggressive optimizations that may increase compile time"),
    llvm::cl::init(false)
  };
};

//===----------------------------------------------------------------------===//
// Pass Implementation Helpers
//===----------------------------------------------------------------------===//

/// Base class for Tile IR transformation passes.
template<typename DerivedT>
class TileIRPassBase : public mlir::OperationPass<mlir::func::FuncOp> {
public:
  using Base = mlir::OperationPass<mlir::func::FuncOp>;
  
  TileIRPassBase(const ArchitectureInfo& arch_info) 
    : Base(DerivedT::getPassName()), arch_info_(arch_info) {}

protected:
  const ArchitectureInfo& getArchitectureInfo() const { return arch_info_; }
  
  /// Get performance estimator for the target architecture
  PerformanceEstimator getPerformanceEstimator() const {
    return PerformanceEstimator(arch_info_);
  }
  
  /// Get tile optimizer for the target architecture
  TileOptimizer getTileOptimizer() const {
    return TileOptimizer(arch_info_);
  }
  
  /// Get access analyzer for memory optimization
  AccessAnalyzer getAccessAnalyzer() const {
    return AccessAnalyzer();
  }

private:
  const ArchitectureInfo& arch_info_;
};

/// Utility functions for pass implementation
namespace pass_utils {

/// Find all operations of a specific type in the function
template<typename OpType>
llvm::SmallVector<OpType, 8> findOpsOfType(mlir::func::FuncOp func) {
  llvm::SmallVector<OpType, 8> ops;
  func.walk([&](OpType op) {
    ops.push_back(op);
  });
  return ops;
}

/// Check if operation can benefit from Tensor Core acceleration
bool canUseTensorCores(Operation* op, const ArchitectureInfo& arch_info);

/// Estimate register pressure for a region
int64_t estimateRegisterPressure(Region* region, const ArchitectureInfo& arch_info);

/// Check if memory accesses have bank conflicts
bool hasBankConflicts(ArrayRef<Operation*> memory_ops, int64_t bank_width = 32);

/// Generate optimal thread mapping for operation
ThreadMapping generateOptimalThreadMapping(Operation* op, 
                                           const ArchitectureInfo& arch_info);

/// Check if operations can be fused
bool canFuseOperations(Operation* op1, Operation* op2);

/// Estimate memory bandwidth utilization
double estimateMemoryBandwidthUtilization(Operation* op, 
                                           const ArchitectureInfo& arch_info);

/// Generate swizzle pattern for shared memory layout
llvm::SmallVector<int64_t, 4> generateSwizzlePattern(ArrayRef<int64_t> tensor_shape,
                                                      int64_t num_banks = 32);

} // namespace pass_utils

//===----------------------------------------------------------------------===//
// Specific Pass Implementations Preview
//===----------------------------------------------------------------------===//

/// Memory Hierarchy Optimization Pass Implementation
class MemoryHierarchyOptimizationPass 
    : public TileIRPassBase<MemoryHierarchyOptimizationPass> {
public:
  explicit MemoryHierarchyOptimizationPass(const ArchitectureInfo& arch_info)
    : TileIRPassBase(arch_info) {}

  void runOnOperation() override;
  
  static StringRef getPassName() { return "tessera-memory-hierarchy-optimization"; }
  static StringRef getDescription() { 
    return "Optimize memory allocation across hierarchy levels"; 
  }

private:
  void optimizeMemoryAllocations();
  void analyzeAccessPatterns();
  void promoteToSharedMemory();
  void promoteToRegisters();
  void insertCooperativeCopies();
};

/// Tensor Core Optimization Pass Implementation  
class TensorCoreOptimizationPass 
    : public TileIRPassBase<TensorCoreOptimizationPass> {
public:
  explicit TensorCoreOptimizationPass(const ArchitectureInfo& arch_info)
    : TileIRPassBase(arch_info) {}

  void runOnOperation() override;
  
  static StringRef getPassName() { return "tessera-tensor-core-optimization"; }
  static StringRef getDescription() { 
    return "Optimize GEMM operations for Tensor Cores"; 
  }

private:
  void identifyTensorCoreOpportunities();
  void transformToFragmentOperations();
  void optimizeFragmentLayouts();
  void insertOptimalBarriers();
  bool canUseTensorCores(Operation* gemm_op);
  FragmentInfo getOptimalFragmentLayout(Operation* op);
};

/// Software Pipelining Pass Implementation
class SoftwarePipeliningPass 
    : public TileIRPassBase<SoftwarePipeliningPass> {
public:
  explicit SoftwarePipeliningPass(const ArchitectureInfo& arch_info, int max_stages = 3)
    : TileIRPassBase(arch_info), max_stages_(max_stages) {}

  void runOnOperation() override;
  
  static StringRef getPassName() { return "tessera-software-pipelining"; }
  static StringRef getDescription() { 
    return "Create software pipelines for overlapping computation and memory"; 
  }

private:
  void analyzeDependencies();
  void identifyPipelineOpportunities();
  void createPipelineStages();
  void insertStageSynchronization();
  
  int max_stages_;
  llvm::DenseMap<Operation*, int> stage_assignment_;
  llvm::SmallVector<llvm::SmallVector<Operation*, 4>, 4> pipeline_stages_;
};

/// CuTe Preparation Pass Implementation
class CuTePreparationPass 
    : public TileIRPassBase<CuTePreparationPass> {
public:
  explicit CuTePreparationPass(const ArchitectureInfo& arch_info)
    : TileIRPassBase(arch_info) {}

  void runOnOperation() override;
  
  static StringRef getPassName() { return "tessera-cute-preparation"; }
  static StringRef getDescription() { 
    return "Prepare Tile IR for CuTe kernel generation"; 
  }

private:
  void convertToFragmentOperations();
  void optimizeLayoutsForCuTe();
  void insertCuTeCompatibleCopies();
  void generateCuTeMetadata();
  std::string generateCuTeLayoutString(ArrayRef<int64_t> shape, 
                                       const ThreadMapping& mapping);
};

//===----------------------------------------------------------------------===//
// Pass Registration and Factory Functions
//===----------------------------------------------------------------------===//

/// Register all Tile IR passes with the pass registry
void registerTileIRPasses();

/// Create pass with architecture-specific configuration
template<typename PassT>
std::unique_ptr<Pass> createArchSpecificPass(StringRef target_arch) {
  ArchitectureInfo arch_info(target_arch);
  return std::make_unique<PassT>(arch_info);
}

/// Pass factory with options
struct PassFactory {
  static std::unique_ptr<Pass> createMemoryOptPass(const TileIRPipelineOptions& options);
  static std::unique_ptr<Pass> createComputeOptPass(const TileIRPipelineOptions& options);  
  static std::unique_ptr<Pass> createPipelineOptPass(const TileIRPipelineOptions& options);
  static std::unique_ptr<Pass> createHardwareOptPass(const TileIRPipelineOptions& options);
  static std::unique_ptr<Pass> createCodegenPrepPass(const TileIRPipelineOptions& options);
};

} // namespace tile
} // namespace tessera  
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pass Pipeline Registration
//===----------------------------------------------------------------------===//

/// Register pass pipelines with MLIR
void registerTileIRPassPipelines();

/// Macro for registering individual passes
#define TESSERA_TILE_PASS_REGISTRATION \
  mlir::tessera::tile::registerTileIRPasses(); \
  mlir::tessera::tile::registerTileIRPassPipelines();

#endif // TESSERA_TILE_PASSES_H
    