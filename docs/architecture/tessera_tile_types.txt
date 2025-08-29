//===- TesseraTileTypes.h - Tessera Tile IR Types -*- C++ -*-===//
//
// This file defines the types and utilities for Tessera's Tile IR dialect.
// The Tile IR represents low-level compute and memory operations with explicit
// hardware control and optimization.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TILE_TYPES_H
#define TESSERA_TILE_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace tessera {
namespace tile {

//===----------------------------------------------------------------------===//
// Memory Hierarchy Utilities
//===----------------------------------------------------------------------===//

enum class MemorySpace {
  Global,    // Global GPU memory (HBM/GDDR)
  Shared,    // Shared memory per SM
  Register,  // Register file
  TMEM,      // Tensor Memory (Blackwell)
  Texture,   // Texture memory
  Constant   // Constant memory
};

// Convert string to memory space enum
MemorySpace getMemorySpaceFromString(llvm::StringRef space);
llvm::StringRef getStringFromMemorySpace(MemorySpace space);

// Memory space attributes and properties
struct MemorySpaceInfo {
  size_t bandwidth_gbps;      // Peak bandwidth in GB/s
  size_t capacity_bytes;      // Maximum capacity
  size_t bank_width_bytes;    // Bank width for conflict analysis
  size_t alignment_bytes;     // Required alignment
  bool supports_vectorization;
  bool supports_bank_conflicts;
};

MemorySpaceInfo getMemorySpaceInfo(MemorySpace space, 
                                   llvm::StringRef target_arch = "sm_90");

//===----------------------------------------------------------------------===//
// Thread Mapping Utilities
//===----------------------------------------------------------------------===//

struct ThreadMapping {
  llvm::SmallVector<int64_t, 4> thread_shape;  // Threads per dimension
  llvm::SmallVector<int64_t, 4> warp_shape;    // Warps per dimension
  llvm::StringRef strategy;                     // Mapping strategy
  
  // Calculate total threads
  int64_t getTotalThreads() const;
  
  // Calculate total warps
  int64_t getTotalWarps() const;
  
  // Check if mapping is valid for given tensor shape
  bool isValidForShape(llvm::ArrayRef<int64_t> tensor_shape) const;
  
  // Generate optimal thread mapping for tensor shape
  static ThreadMapping generateOptimal(llvm::ArrayRef<int64_t> tensor_shape,
                                       llvm::StringRef target_arch = "sm_90");
};

//===----------------------------------------------------------------------===//
// Fragment Layout Utilities
//===----------------------------------------------------------------------===//

enum class FragmentLayout {
  RowMajor,
  ColumnMajor,
  WMMA_MatrixA,
  WMMA_MatrixB, 
  WMMA_Accumulator,
  MMA_MatrixA,
  MMA_MatrixB,
  MMA_Accumulator,
  Custom
};

FragmentLayout getFragmentLayoutFromString(llvm::StringRef layout);
llvm::StringRef getStringFromFragmentLayout(FragmentLayout layout);

struct FragmentInfo {
  Type element_type;
  llvm::SmallVector<int64_t, 4> shape;
  FragmentLayout layout;
  bool accumulate;
  
  // Get register count for this fragment
  int64_t getRegisterCount() const;
  
  // Check compatibility with MMA operations
  bool isCompatibleWithMMA(const FragmentInfo& other) const;
  
  // Get optimal fragment size for given operation
  static FragmentInfo getOptimalForMMA(Type element_type,
                                       llvm::ArrayRef<int64_t> tile_shape,
                                       llvm::StringRef target_arch = "sm_90");
};

//===----------------------------------------------------------------------===//
// Hardware Architecture Utilities
//===----------------------------------------------------------------------===//

enum class ArchitectureFeatures {
  TensorCores,      // Tensor Core support
  WMMA,            // Warp Matrix Multiply Accumulate
  MMA,             // Matrix Multiply Accumulate PTX
  TMEM,            // Tensor Memory (Blackwell)
  CTAPairs,        // CTA pair coordination
  AsyncCopy,       // Asynchronous memory copy
  DistributedShared // Distributed shared memory
};

class ArchitectureInfo {
public:
  explicit ArchitectureInfo(llvm::StringRef arch_string);
  
  // Architecture queries
  bool hasFeature(ArchitectureFeatures feature) const;
  int64_t getComputeCapability() const;
  int64_t getMaxThreadsPerBlock() const;
  int64_t getMaxWarpsPerSM() const;
  int64_t getSharedMemorySize() const;
  int64_t getRegisterFileSize() const;
  
  // Memory hierarchy info
  MemorySpaceInfo getMemoryInfo(MemorySpace space) const;
  
  // Optimal parameters for operations
  ThreadMapping getOptimalThreadMapping(llvm::ArrayRef<int64_t> shape,
                                        llvm::StringRef operation) const;
  
  FragmentInfo getOptimalFragment(Type element_type,
                                  llvm::ArrayRef<int64_t> shape,
                                  llvm::StringRef operation) const;

private:
  llvm::StringRef arch_string_;
  int64_t compute_capability_;
  llvm::DenseSet<ArchitectureFeatures> features_;
};

//===----------------------------------------------------------------------===//
// Performance Modeling
//===----------------------------------------------------------------------===//

struct PerformanceModel {
  // Estimate execution time in cycles
  int64_t compute_cycles;
  int64_t memory_cycles;
  int64_t synchronization_cycles;
  
  // Resource utilization
  double compute_utilization;   // [0.0, 1.0]
  double memory_utilization;    // [0.0, 1.0]
  double register_pressure;     // [0.0, 1.0]
  
  // Get total estimated cycles
  int64_t getTotalCycles() const {
    return std::max({compute_cycles, memory_cycles, synchronization_cycles});
  }
  
  // Get bottleneck type
  llvm::StringRef getBottleneck() const;
};

class PerformanceEstimator {
public:
  explicit PerformanceEstimator(const ArchitectureInfo& arch_info)
    : arch_info_(arch_info) {}
  
  // Estimate performance for specific operations
  PerformanceModel estimateGEMM(llvm::ArrayRef<int64_t> lhs_shape,
                                llvm::ArrayRef<int64_t> rhs_shape,
                                Type element_type,
                                const ThreadMapping& mapping) const;
  
  PerformanceModel estimateReduce(llvm::ArrayRef<int64_t> input_shape,
                                  llvm::ArrayRef<int64_t> axes,
                                  Type element_type,
                                  const ThreadMapping& mapping) const;
  
  PerformanceModel estimateMemoryCopy(size_t bytes,
                                      MemorySpace src_space,
                                      MemorySpace dst_space,
                                      const ThreadMapping& mapping) const;

private:
  const ArchitectureInfo& arch_info_;
  
  // Helper methods for cost modeling
  int64_t estimateArithmeticCycles(int64_t operations, Type element_type) const;
  int64_t estimateMemoryAccessCycles(size_t bytes, MemorySpace space) const;
  int64_t estimateSynchronizationCycles(llvm::StringRef sync_type) const;
};

//===----------------------------------------------------------------------===//
// Layout and Access Pattern Analysis
//===----------------------------------------------------------------------===//

enum class AccessPattern {
  Sequential,
  Strided,
  Random,
  Coalesced,
  Broadcast,
  Transpose
};

struct AccessInfo {
  AccessPattern pattern;
  int64_t stride;
  int64_t vectorization_factor;
  bool has_bank_conflicts;
  double efficiency;  // [0.0, 1.0]
};

class AccessAnalyzer {
public:
  // Analyze memory access pattern
  static AccessInfo analyzeAccess(llvm::ArrayRef<int64_t> tensor_shape,
                                  llvm::ArrayRef<int64_t> access_indices,
                                  const ThreadMapping& mapping,
                                  MemorySpace memory_space);
  
  // Suggest optimal access pattern
  static ThreadMapping suggestOptimalMapping(llvm::ArrayRef<int64_t> tensor_shape,
                                             MemorySpace memory_space,
                                             const ArchitectureInfo& arch_info);
  
  // Check for bank conflicts in shared memory
  static bool hasBankConflicts(llvm::ArrayRef<int64_t> access_pattern,
                               int64_t bank_width = 32);
  
  // Generate swizzle pattern to avoid conflicts
  static llvm::SmallVector<int64_t, 4> 
    generateSwizzlePattern(llvm::ArrayRef<int64_t> tensor_shape,
                           int64_t num_banks = 32);
};

//===----------------------------------------------------------------------===//
// Optimization Utilities
//===----------------------------------------------------------------------===//

struct TileConfiguration {
  llvm::SmallVector<int64_t, 4> tile_sizes;
  ThreadMapping thread_mapping;
  FragmentInfo fragment_info;
  MemorySpace primary_memory;
  bool use_tensor_cores;
  bool enable_pipelining;
  int64_t pipeline_stages;
};

class TileOptimizer {
public:
  explicit TileOptimizer(const ArchitectureInfo& arch_info)
    : arch_info_(arch_info) {}
  
  // Find optimal tiling configuration
  TileConfiguration optimizeForGEMM(llvm::ArrayRef<int64_t> lhs_shape,
                                    llvm::ArrayRef<int64_t> rhs_shape,
                                    Type element_type) const;
  
  TileConfiguration optimizeForReduce(llvm::ArrayRef<int64_t> input_shape,
                                      llvm::ArrayRef<int64_t> reduce_axes,
                                      Type element_type) const;
  
  TileConfiguration optimizeForElementwise(llvm::ArrayRef<int64_t> tensor_shape,
                                           Type element_type,
                                           int64_t num_operands = 1) const;
  
  // Validate configuration
  bool validateConfiguration(const TileConfiguration& config) const;
  
  // Estimate performance for configuration
  PerformanceModel estimatePerformance(const TileConfiguration& config,
                                       llvm::StringRef operation,
                                       llvm::ArrayRef<int64_t> shapes) const;

private:
  const ArchitectureInfo& arch_info_;
  
  // Helper methods for optimization
  llvm::SmallVector<TileConfiguration, 8> 
    generateCandidateConfigurations(llvm::ArrayRef<int64_t> tensor_shape,
                                    llvm::StringRef operation) const;
  
  TileConfiguration selectBestConfiguration(
      llvm::ArrayRef<TileConfiguration> candidates,
      llvm::StringRef operation,
      llvm::ArrayRef<int64_t> shapes) const;
};

//===----------------------------------------------------------------------===//
// CuTe Integration Utilities
//===----------------------------------------------------------------------===//

// Interface for CuTe-style layouts and operations
class CuTeLayoutInterface {
public:
  virtual ~CuTeLayoutInterface() = default;
  
  // Convert to CuTe layout representation
  virtual std::string toCuTeLayout() const = 0;
  
  // Generate CuTe copy operations
  virtual std::string generateCuTeCopy(llvm::StringRef src_layout,
                                       llvm::StringRef dst_layout) const = 0;
  
  // Generate CuTe MMA operations
  virtual std::string generateCuTeMMA(llvm::StringRef a_layout,
                                      llvm::StringRef b_layout,
                                      llvm::StringRef c_layout) const = 0;
};

class CuTeGenerator {
public:
  // Generate CuTe kernel code from Tile IR
  static std::string generateKernel(Operation* tile_op,
                                    const TileConfiguration& config);
  
  // Generate optimal CuTe layouts
  static std::string generateOptimalLayout(llvm::ArrayRef<int64_t> tensor_shape,
                                           const ThreadMapping& mapping,
                                           FragmentLayout layout);
  
  // Generate cooperative copy patterns
  static std::string generateCooperativeCopy(MemorySpace src_space,
                                             MemorySpace dst_space,
                                             const ThreadMapping& mapping);
};

//===----------------------------------------------------------------------===//
// Verification and Validation
//===----------------------------------------------------------------------===//

class TileIRVerifier {
public:
  explicit TileIRVerifier(const ArchitectureInfo& arch_info)
    : arch_info_(arch_info) {}
  
  // Verify operation constraints
  LogicalResult verifyGEMMOp(Operation* op) const;
  LogicalResult verifyReduceOp(Operation* op) const;
  LogicalResult verifyMemoryOp(Operation* op) const;
  LogicalResult verifyFragmentOp(Operation* op) const;
  
  // Verify resource constraints
  LogicalResult verifyResourceUsage(Operation* op) const;
  LogicalResult verifyMemoryCoherence(Operation* op) const;
  LogicalResult verifyThreadMapping(Operation* op) const;
  
  // Verify hardware compatibility
  LogicalResult verifyHardwareSupport(Operation* op) const;

private:
  const ArchitectureInfo& arch_info_;
  
  // Helper verification methods
  bool checkMemoryBounds(Operation* op) const;
  bool checkRegisterPressure(Operation* op) const;
  bool checkBankConflicts(Operation* op) const;
  bool checkAlignment(Operation* op) const;
};

//===----------------------------------------------------------------------===//
// Code Generation Utilities
//===----------------------------------------------------------------------===//

enum class TargetBackend {
  CUDA_PTX,
  HIP_AMD,
  Triton,
  CuTe_CUDA,
  Metal_Shaders
};

class CodeGenerator {
public:
  explicit CodeGenerator(TargetBackend backend, 
                         const ArchitectureInfo& arch_info)
    : backend_(backend), arch_info_(arch_info) {}
  
  // Generate target-specific code
  std::string generateKernel(Operation* tile_op,
                             const TileConfiguration& config) const;
  
  std::string generateMemoryOps(Operation* op) const;
  std::string generateComputeOps(Operation* op) const;
  std::string generateSyncOps(Operation* op) const;
  
  // Generate runtime support code
  std::string generateLaunchParameters(const TileConfiguration& config) const;
  std::string generateResourceAllocation(Operation* op) const;

private:
  TargetBackend backend_;
  const ArchitectureInfo& arch_info_;
  
  // Backend-specific generators
  std::string generateCUDACode(Operation* op, const TileConfiguration& config) const;
  std::string generateHIPCode(Operation* op, const TileConfiguration& config) const;
  std::string generateTritonCode(Operation* op, const TileConfiguration& config) const;
  std::string generateCuTeCode(Operation* op, const TileConfiguration& config) const;
};

} // namespace tile
} // namespace tessera
} // namespace mlir

// Include the generated type definitions
#define GET_TYPEDEF_CLASSES
#include "Tessera/Tile/TesseraTileTypes.h.inc"

#endif // TESSERA_TILE_TYPES_H