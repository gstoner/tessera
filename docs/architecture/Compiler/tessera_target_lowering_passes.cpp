//===- TesseraTargetLoweringPasses.h - Target IR Lowering Passes -*- C++ -*-===//
//
// This file defines passes that lower Tessera Tile IR to Target IR and
// subsequently to LLVM GPU IR, PTX, HIP, and other target formats.
// These passes handle final code generation with platform-specific optimization.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TARGET_LOWERING_PASSES_H
#define TESSERA_TARGET_LOWERING_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

namespace mlir {
namespace tessera {
namespace target {

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering target lowering passes.
#define GEN_PASS_DECL
#include "Tessera/Target/TesseraTargetLoweringPasses.h.inc"

//===----------------------------------------------------------------------===//
// Tile IR to Target IR Lowering Passes
//===----------------------------------------------------------------------===//

/// Pass to lower Tile IR operations to Target IR operations.
/// This is the first stage of final code generation, converting high-level
/// tile operations to platform-specific target operations.
std::unique_ptr<Pass> createTileToTargetLoweringPass();

/// Pass to lower Tile IR memory operations to Target IR memory operations.
/// Handles memory hierarchy mapping and platform-specific memory management.
std::unique_ptr<Pass> createTileMemoryToTargetLoweringPass();

/// Pass to lower Tile IR compute operations to Target IR compute operations.
/// Maps high-level compute operations to hardware-specific instructions.
std::unique_ptr<Pass> createTileComputeToTargetLoweringPass();

/// Pass to lower Tile IR fragment operations to Target IR tensor core operations.
/// Converts register fragments to hardware-accelerated matrix operations.
std::unique_ptr<Pass> createTileFragmentToTargetLoweringPass();

//===----------------------------------------------------------------------===//
// Target IR to GPU IR Lowering Passes
//===----------------------------------------------------------------------===//

/// Pass to lower Target IR kernels to GPU dialect kernels.
/// Maps target-specific kernels to standard GPU kernel representation.
std::unique_ptr<Pass> createTargetToGPULoweringPass();

/// Pass to lower Target IR platform-specific operations to GPU operations.
/// Handles platform-specific instruction mapping to GPU dialect.
std::unique_ptr<Pass> createTargetInstructionToGPULoweringPass();

/// Pass to lower Target IR memory operations to GPU memory operations.
/// Maps target memory operations to GPU memory model.
std::unique_ptr<Pass> createTargetMemoryToGPULoweringPass();

//===----------------------------------------------------------------------===//
// GPU IR to LLVM IR Lowering Passes
//===----------------------------------------------------------------------===//

/// Pass to lower GPU operations to LLVM GPU IR.
/// Final stage before PTX/SASS generation for NVIDIA targets.
std::unique_ptr<Pass> createGPUToLLVMLoweringPass();

/// Pass to lower GPU kernel launches to LLVM runtime calls.
/// Generates host-side kernel launch code with proper parameter marshaling.
std::unique_ptr<Pass> createGPUKernelToLLVMRuntimeLoweringPass();

/// Pass to lower GPU memory operations to LLVM memory intrinsics.
/// Maps GPU memory operations to LLVM memory model with address spaces.
std::unique_ptr<Pass> createGPUMemoryToLLVMLoweringPass();

//===----------------------------------------------------------------------===//
// Platform-Specific Code Generation Passes
//===----------------------------------------------------------------------===//

/// Pass to generate NVIDIA PTX code from LLVM GPU IR.
/// Produces optimized PTX assembly for NVIDIA GPU targets.
std::unique_ptr<Pass> createLLVMToPTXCodeGenPass();

/// Pass to generate AMD HIP code from LLVM GPU IR.
/// Produces optimized HIP kernels for AMD GPU targets.
std::unique_ptr<Pass> createLLVMToHIPCodeGenPass();

/// Pass to generate Intel GPU code from LLVM GPU IR.
/// Produces optimized code for Intel Xe GPU architectures.
std::unique_ptr<Pass> createLLVMToIntelGPUCodeGenPass();

/// Pass to generate SPIR-V from GPU operations.
/// Produces SPIR-V for Vulkan compute or OpenCL targets.
std::unique_ptr<Pass> createGPUToSPIRVCodeGenPass();

//===----------------------------------------------------------------------===//
// Hardware-Specific Optimization Passes
//===----------------------------------------------------------------------===//

/// Pass to optimize for NVIDIA Hopper architecture during lowering.
/// Applies Hopper-specific instruction selection and optimization.
std::unique_ptr<Pass> createHopperTargetOptimizationPass();

/// Pass to optimize for NVIDIA Blackwell architecture during lowering.
/// Applies Blackwell-specific features (TMEM, CTA pairs, etc.).
std::unique_ptr<Pass> createBlackwellTargetOptimizationPass();

/// Pass to optimize for AMD RDNA3 architecture during lowering.
/// Applies RDNA3-specific instruction selection and optimization.
std::unique_ptr<Pass> createRDNA3TargetOptimizationPass();

/// Pass to optimize for AMD CDNA architecture during lowering.
/// Applies CDNA-specific compute optimization for data center workloads.
std::unique_ptr<Pass> createCDNATargetOptimizationPass();

//===----------------------------------------------------------------------===//
// Instruction Selection and Scheduling Passes
//===----------------------------------------------------------------------===//

/// Pass to perform instruction selection for target architectures.
/// Maps high-level operations to optimal instruction sequences.
std::unique_ptr<Pass> createInstructionSelectionPass();

/// Pass to schedule instructions for optimal performance.
/// Reorders instructions to minimize latency and maximize throughput.
std::unique_ptr<Pass> createTargetInstructionSchedulingPass();

/// Pass to perform register allocation for GPU targets.
/// Allocates registers while minimizing spills and maximizing occupancy.
std::unique_ptr<Pass> createTargetRegisterAllocationPass();

/// Pass to perform peephole optimizations on target code.
/// Local optimizations on instruction sequences for better performance.
std::unique_ptr<Pass> createTargetPeepholeOptimizationPass();

//===----------------------------------------------------------------------===//
// CuTe Integration Passes
//===----------------------------------------------------------------------===//

/// Pass to generate CuTe-compatible kernel code.
/// Produces kernels using CuTe template library patterns.
std::unique_ptr<Pass> createCuTeKernelGenerationPass();

/// Pass to optimize CuTe layout transformations.
/// Optimizes CuTe layout operations for maximum performance.
std::unique_ptr<Pass> createCuTeLayoutOptimizationPass();

/// Pass to generate CuTe cooperative operations.
/// Maps cooperative patterns to CuTe implementations.
std::unique_ptr<Pass> createCuTeCooperativeOperationPass();

//===----------------------------------------------------------------------===//
// Runtime Integration Passes
//===----------------------------------------------------------------------===//

/// Pass to generate CUDA runtime integration code.
/// Produces host-side code for kernel launches and memory management.
std::unique_ptr<Pass> createCUDARuntimeIntegrationPass();

/// Pass to generate HIP runtime integration code.
/// Produces host-side code for AMD GPU runtime integration.
std::unique_ptr<Pass> createHIPRuntimeIntegrationPass();

/// Pass to generate performance profiling instrumentation.
/// Inserts profiling code for performance analysis and tuning.
std::unique_ptr<Pass> createProfilingInstrumentationPass();

/// Pass to generate debugging instrumentation.
/// Inserts debugging support for GPU kernel development.
std::unique_ptr<Pass> createDebuggingInstrumentationPass();

//===----------------------------------------------------------------------===//
// Verification and Validation Passes
//===----------------------------------------------------------------------===//

/// Pass to verify target code correctness.
/// Validates generated code against hardware constraints and semantics.
std::unique_ptr<Pass> createTargetVerificationPass();

/// Pass to validate hardware resource usage.
/// Ensures resource usage is within hardware limits.
std::unique_ptr<Pass> createResourceValidationPass();

/// Pass to check memory coherence and synchronization.
/// Validates memory ordering and synchronization correctness.
std::unique_ptr<Pass> createMemoryCoherenceValidationPass();

//===----------------------------------------------------------------------===//
// Pass Pipeline Construction
//===----------------------------------------------------------------------===//

/// Build complete Tile IR to Target lowering pipeline.
/// Comprehensive pipeline from Tile IR to target-specific code.
void buildTileToTargetLoweringPipeline(mlir::PassManager& pm,
                                       const TargetLoweringOptions& options);

/// Build Target IR to LLVM lowering pipeline.
/// Converts Target IR to LLVM GPU IR ready for code generation.
void buildTargetToLLVMLoweringPipeline(mlir::PassManager& pm,
                                       const TargetLoweringOptions& options);

/// Build platform-specific code generation pipeline.
/// Final code generation for specific GPU platforms.
void buildPlatformCodeGenPipeline(mlir::PassManager& pm,
                                  StringRef target_platform,
                                  const TargetLoweringOptions& options);

/// Build complete end-to-end lowering pipeline.
/// From Tile IR to final executable GPU code.
void buildEndToEndLoweringPipeline(mlir::PassManager& pm,
                                   const TargetLoweringOptions& options);

//===----------------------------------------------------------------------===//
// Pass Options
//===----------------------------------------------------------------------===//

struct TargetLoweringOptions : public mlir::PassPipelineOptions<TargetLoweringOptions> {
  // Target platform specification
  Option<std::string> targetPlatform{
    *this, "target-platform",
    llvm::cl::desc("Target GPU platform (cuda, hip, spirv, metal)"),
    llvm::cl::init("cuda")
  };
  
  Option<std::string> targetArchitecture{
    *this, "target-arch",
    llvm::cl::desc("Target GPU architecture (sm_90, gfx1100, xe_hpg)"),
    llvm::cl::init("sm_90")
  };
  
  Option<std::string> hostTarget{
    *this, "host-target",
    llvm::cl::desc("Host target triple"),
    llvm::cl::init("x86_64-unknown-linux-gnu")
  };
  
  // Code generation options
  Option<int> optimizationLevel{
    *this, "opt-level",
    llvm::cl::desc("Optimization level (0-3)"),
    llvm::cl::init(2)
  };
  
  Option<bool> enableDebugInfo{
    *this, "enable-debug-info",
    llvm::cl::desc("Enable debug information generation"),
    llvm::cl::init(false)
  };
  
  Option<bool> enableProfiling{
    *this, "enable-profiling",
    llvm::cl::desc("Enable profiling instrumentation"),
    llvm::cl::init(false)
  };
  
  // Hardware-specific options
  Option<bool> enableTensorCores{
    *this, "enable-tensor-cores",
    llvm::cl::desc("Enable Tensor Core acceleration"),
    llvm::cl::init(true)
  };
  
  Option<bool> enableCooperativeKernels{
    *this, "enable-cooperative-kernels",
    llvm::cl::desc("Enable cooperative kernel features"),
    llvm::cl::init(true)
  };
  
  Option<bool> enableAsyncOperations{
    *this, "enable-async-ops",
    llvm::cl::desc("Enable asynchronous operations"),
    llvm::cl::init(true)
  };
  
  // Memory management options
  Option<std::string> memoryModel{
    *this, "memory-model",
    llvm::cl::desc("Memory model (unified, discrete)"),
    llvm::cl::init("discrete")
  };
  
  Option<bool> enableUnifiedMemory{
    *this, "enable-unified-memory",
    llvm::cl::desc("Enable unified memory when available"),
    llvm::cl::init(false)
  };
  
  Option<int> maxSharedMemoryBytes{
    *this, "max-shared-memory",
    llvm::cl::desc("Maximum shared memory per block in bytes"),
    llvm::cl::init(49152)  // 48KB default
  };
  
  // CuTe integration options
  Option<bool> enableCuTeGeneration{
    *this, "enable-cute-generation",
    llvm::cl::desc("Enable CuTe kernel generation"),
    llvm::cl::init(false)
  };
  
  Option<std::string> cuTeVersion{
    *this, "cute-version",
    llvm::cl::desc("CuTe library version"),
    llvm::cl::init("2.0")
  };
  
  // Runtime integration options
  Option<std::string> runtimeAPI{
    *this, "runtime-api",
    llvm::cl::desc("Runtime API (driver, runtime)"),
    llvm::cl::init("runtime")
  };
  
  Option<bool> generateHostCode{
    *this, "generate-host-code",
    llvm::cl::desc("Generate host-side runtime code"),
    llvm::cl::init(true)
  };
  
  Option<bool> generateKernelMetadata{
    *this, "generate-kernel-metadata",
    llvm::cl::desc("Generate kernel metadata for profiling"),
    llvm::cl::init(true)
  };
};

//===----------------------------------------------------------------------===//
// Lowering Patterns and Utilities
//===----------------------------------------------------------------------===//

/// Base class for Target IR lowering patterns
template<typename SourceOp, typename TargetOp>
class TargetLoweringPattern : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
    SourceOp op, 
    typename SourceOp::Adaptor adaptor,
    ConversionPatternRewriter& rewriter) const override;

protected:
  /// Get target platform information
  virtual StringRef getTargetPlatform() const = 0;
  
  /// Get target architecture information  
  virtual StringRef getTargetArchitecture() const = 0;
  
  /// Check if target supports specific feature
  virtual bool supportsFeature(StringRef feature) const = 0;
};

/// Utility functions for lowering patterns
namespace lowering_utils {

/// Convert Tile IR memory space to LLVM address space
unsigned getAddressSpaceForMemorySpace(StringRef memory_space, 
                                       StringRef target_platform);

/// Get optimal instruction sequence for operation
SmallVector<Operation*> getOptimalInstructionSequence(
  Operation* op, 
  StringRef target_arch,
  OpBuilder& builder);

/// Generate kernel launch parameters
struct KernelLaunchParams {
  SmallVector<int64_t> grid_size;
  SmallVector<int64_t> block_size;
  int64_t shared_memory_bytes;
  int64_t register_count;
  double occupancy;
};

KernelLaunchParams computeOptimalLaunchParams(
  Operation* kernel_op,
  StringRef target_arch);

/// Generate runtime API calls
SmallVector<Operation*> generateRuntimeCalls(
  Operation* op,
  StringRef runtime_api,
  OpBuilder& builder);

} // namespace lowering_utils

//===----------------------------------------------------------------------===//
// Platform-Specific Lowering Implementation
//===----------------------------------------------------------------------===//

/// CUDA/PTX lowering implementation
class CUDALoweringProvider {
public:
  explicit CUDALoweringProvider(StringRef target_arch) 
    : target_arch_(target_arch) {}
  
  /// Lower Tile IR to CUDA Target IR
  LogicalResult lowerTileToTarget(Operation* op, 
                                  ConversionPatternRewriter& rewriter) const;
  
  /// Lower Target IR to PTX
  LogicalResult lowerTargetToPTX(Operation* op,
                                 ConversionPatternRewriter& rewriter) const;
  
  /// Generate CUDA runtime calls
  SmallVector<Operation*> generateCUDARuntimeCalls(
    Operation* op, OpBuilder& builder) const;

private:
  StringRef target_arch_;
  
  /// CUDA-specific helper methods
  bool supportsTensorCores() const;
  bool supportsCooperativeGroups() const;
  bool supportsTMEM() const;  // Blackwell TMEM
  unsigned getMaxSharedMemoryBytes() const;
  unsigned getMaxRegistersPerThread() const;
};

/// HIP/ROCm lowering implementation  
class HIPLoweringProvider {
public:
  explicit HIPLoweringProvider(StringRef target_arch)
    : target_arch_(target_arch) {}
    
  /// Lower Tile IR to HIP Target IR
  LogicalResult lowerTileToTarget(Operation* op,
                                  ConversionPatternRewriter& rewriter) const;
                                  
  /// Lower Target IR to HIP
  LogicalResult lowerTargetToHIP(Operation* op,
                                 ConversionPatternRewriter& rewriter) const;
                                 
  /// Generate HIP runtime calls
  SmallVector<Operation*> generateHIPRuntimeCalls(
    Operation* op, OpBuilder& builder) const;

private:
  StringRef target_arch_;
  
  /// HIP-specific helper methods
  bool supportsMatrixCores() const;  // AMD Matrix Cores
  bool supportsWavefrontOperations() const;
  unsigned getMaxLDSBytes() const;   // Local Data Share
  unsigned getWavefrontSize() const; // 32 or 64
};

/// Intel GPU lowering implementation
class IntelGPULoweringProvider {
public:
  explicit IntelGPULoweringProvider(StringRef target_arch)
    : target_arch_(target_arch) {}
    
  /// Lower Tile IR to Intel GPU Target IR  
  LogicalResult lowerTileToTarget(Operation* op,
                                  ConversionPatternRewriter& rewriter) const;
                                  
  /// Generate Intel GPU runtime calls
  SmallVector<Operation*> generateIntelGPURuntimeCalls(
    Operation* op, OpBuilder& builder) const;

private:
  StringRef target_arch_;
  
  /// Intel GPU-specific helper methods  
  bool supportsXeMatrixExtensions() const;
  bool supportsDPAS() const;  // Dot Product Accumulate Systolic
  unsigned getMaxSLMBytes() const;  // Shared Local Memory
  unsigned getSubgroupSize() const;
};

//===----------------------------------------------------------------------===//
// Code Generation Templates
//===----------------------------------------------------------------------===//

/// Template for generating platform-specific kernels
class KernelCodeGenerator {
public:
  virtual ~KernelCodeGenerator() = default;
  
  /// Generate kernel function signature
  virtual std::string generateKernelSignature(
    Operation* kernel_op, 
    const TargetLoweringOptions& options) const = 0;
  
  /// Generate kernel body
  virtual std::string generateKernelBody(
    Operation* kernel_op,
    const TargetLoweringOptions& options) const = 0;
    
  /// Generate host launch code
  virtual std::string generateHostLaunchCode(
    Operation* launch_op,
    const TargetLoweringOptions& options) const = 0;
};

/// CUDA kernel code generator
class CUDAKernelCodeGenerator : public KernelCodeGenerator {
public:
  std::string generateKernelSignature(
    Operation* kernel_op,
    const TargetLoweringOptions& options) const override;
    
  std::string generateKernelBody(
    Operation* kernel_op, 
    const TargetLoweringOptions& options) const override;
    
  std::string generateHostLaunchCode(
    Operation* launch_op,
    const TargetLoweringOptions& options) const override;

private:
  /// Generate PTX inline assembly
  std::string generatePTXAssembly(Operation* op) const;
  
  /// Generate CuTe kernel using templates
  std::string generateCuTeKernel(Operation* op) const;
};

/// HIP kernel code generator
class HIPKernelCodeGenerator : public KernelCodeGenerator {
public:
  std::string generateKernelSignature(
    Operation* kernel_op,
    const TargetLoweringOptions& options) const override;
    
  std::string generateKernelBody(
    Operation* kernel_op,
    const TargetLoweringOptions& options) const override;
    
  std::string generateHostLaunchCode(
    Operation* launch_op, 
    const TargetLoweringOptions& options) const override;

private:
  /// Generate ROCm inline assembly
  std::string generateROCmAssembly(Operation* op) const;
};

//===----------------------------------------------------------------------===//
// Pass Implementation Helpers
//===----------------------------------------------------------------------===//

/// Helper class for managing conversion patterns
class TargetLoweringPatternManager {
public:
  explicit TargetLoweringPatternManager(
    MLIRContext* context,
    const TargetLoweringOptions& options)
    : context_(context), options_(options) {}
  
  /// Add platform-specific lowering patterns
  void addTileToTargetPatterns(RewritePatternSet& patterns);
  void addTargetToLLVMPatterns(RewritePatternSet& patterns);
  void addPlatformSpecificPatterns(RewritePatternSet& patterns);
  
  /// Get type converter for target platform
  std::unique_ptr<TypeConverter> getTargetTypeConverter();

private:
  MLIRContext* context_;
  const TargetLoweringOptions& options_;
  
  /// Platform-specific pattern addition
  void addCUDAPatterns(RewritePatternSet& patterns);
  void addHIPPatterns(RewritePatternSet& patterns);
  void addIntelGPUPatterns(RewritePatternSet& patterns);
};

/// Helper for managing target-specific attributes
class TargetAttributeManager {
public:
  /// Add platform-specific attributes to operations
  static void addTargetAttributes(Operation* op, StringRef platform);
  
  /// Get kernel launch configuration
  static LogicalResult getKernelConfig(Operation* op, 
                                       KernelLaunchParams& params);
  
  /// Set optimization hints
  static void setOptimizationHints(Operation* op,
                                   const TargetLoweringOptions& options);
};

} // namespace target
} // namespace tessera
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Register all target lowering passes
void registerTesseraTargetLoweringPasses();

/// Register target lowering pass pipelines
void registerTesseraTargetLoweringPipelines();

/// Macro for registering target passes
#define TESSERA_TARGET_LOWERING_PASS_REGISTRATION \
  mlir::tessera::target::registerTesseraTargetLoweringPasses(); \
  mlir::tessera::target::registerTesseraTargetLoweringPipelines();

#endif // TESSERA_TARGET_LOWERING_PASSES_H