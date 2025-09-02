# Tessera Target IR - Document 1: Introduction and Architecture

Target IR represents the final stage of Tessera's multi-level compiler pipeline, responsible for transforming portable Tile IR into optimized, backend-specific executable code. This document provides a comprehensive overview of the Target IR architecture, design principles, and compilation strategy.

## What is Target IR?

Target IR is the lowest level of Tessera's intermediate representation stack, sitting between portable Tile IR and actual hardware execution. Unlike higher-level IRs that focus on mathematical correctness and optimization opportunities, Target IR is concerned with:

- **Hardware-specific code generation**
- **Optimal instruction selection** 
- **Memory hierarchy optimization**
- **Runtime system integration**
- **Deployment artifact creation**

## Target IR Architecture Overview

### Multi-Level IR Stack Context

```
Python/Tessera DSL
        ↓
   Graph IR          High-level operations, autodiff, effects
        ↓
  Schedule IR        Loop tiling, memory placement, parallelization
        ↓
    Tile IR         Hardware-aware operations, intrinsics, barriers
        ↓
   Target IR        ← YOU ARE HERE
        ↓
  Executable Code   PTX, CUDA Tile IR, machine code
```

### Target IR Design Philosophy

Target IR embodies several key design principles:

1. **Performance First**: Every decision prioritizes execution speed and hardware utilization
2. **Architecture Awareness**: Deep integration with specific GPU capabilities
3. **Portability with Specialization**: Common abstractions with target-specific optimizations
4. **Production Ready**: Generate deployment-ready artifacts, not just prototypes

## Target Backend Architecture

### Multi-Backend Strategy

Tessera's Target IR supports multiple backend targets through a unified interface:

```
                    Target IR (Unified)
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
    NVIDIA             AMD/ROCm          Intel/oneAPI
  PTX/Tile IR        LLVM IR/HIP       LLVM IR/SYCL
        ↓                  ↓                  ↓
   CUDA Runtime       HIP Runtime      Level Zero
      NCCL              rccl            oneCCL
        ↓                  ↓                  ↓
   H100/A100           MI300X           PVC/Max
```

### Backend Capabilities Matrix

| Feature | NVIDIA | AMD | Intel | Status |
|---------|--------|-----|-------|---------|
| **PTX Code Gen** | ✅ Complete | ❌ | ❌ | Production |
| **CUDA Tile IR** | ✅ Hopper+ | ❌ | ❌ | Production |
| **LLVM Backend** | 🚧 Planned | 🚧 Planned | 🚧 Planned | Development |
| **Tensor Cores** | ✅ WMMA/WGMMA | 🚧 MFMA | 🚧 DPAS | Production/Planned |
| **Async Memory** | ✅ cp.async/TMA | 🚧 LDS | 🚧 LSC | Production/Planned |
| **Collectives** | ✅ NCCL | 🚧 rccl | 🚧 oneCCL | Production/Planned |

## NVIDIA Target Architecture (Primary)

### PTX Code Generation Pipeline

```
Tile IR Operations
        ↓
  PTX Instruction Selection
        ↓
   Register Allocation
        ↓
  Memory Address Computation
        ↓
   Barrier Optimization
        ↓
    PTX Assembly
        ↓
   NVCC Compilation
        ↓
     CUBIN Binary
```

### CUDA Tile IR Pipeline (Hopper+)

```
Tile IR Operations
        ↓
  CUDA Tile IR Conversion
        ↓
  Hardware Feature Mapping
        ↓
   TMA/WGMMA Integration
        ↓
  Cluster Optimization
        ↓
    CUDA Tile IR
        ↓
   NVCC Compilation
        ↓
     CUBIN Binary
```

## Key Architectural Components

### 1. Instruction Selection Engine

The instruction selection engine maps portable Tile IR operations to optimal hardware instructions:

```cpp
class InstructionSelector {
  // Maps tile.mma -> WMMA/WGMMA/MFMA based on target
  virtual InstructionSequence selectMMA(tile::MmaOp op) = 0;
  
  // Maps tile.cp_async -> cp.async/TMA/LDS based on target  
  virtual InstructionSequence selectAsyncCopy(tile::CpAsyncOp op) = 0;
  
  // Maps tile.barrier -> bar.sync/s_barrier based on target
  virtual InstructionSequence selectBarrier(tile::BarrierOp op) = 0;
};
```

#### Architecture-Specific Selection

- **sm_90 (Hopper)**: Prefers WGMMA, TMA, cluster barriers
- **sm_80 (Ampere)**: Uses WMMA, cp.async, standard barriers  
- **sm_75 (Turing)**: Falls back to CUDA cores, explicit copies
- **RDNA3 (AMD)**: Maps to MFMA, LDS operations
- **Xe-HPC (Intel)**: Uses DPAS, LSC load/store

### 2. Memory System Integration

Target IR deeply integrates with each architecture's memory hierarchy:

#### NVIDIA Memory Mapping
- **Registers**: Direct PTX register allocation
- **Shared Memory**: .shared directives with swizzling
- **Global Memory**: Coalesced access patterns
- **Texture Cache**: For read-only data patterns

#### Memory Address Computation
```cpp
class MemoryAddressComputer {
  // Computes optimal addressing for coalesced access
  AddressSequence computeGlobalAddress(Value memref, ArrayRef<Value> indices);
  
  // Handles shared memory bank conflict avoidance
  AddressSequence computeSharedAddress(Value memref, SwizzlePattern pattern);
  
  // Manages register spilling and reuse
  RegisterAllocation allocateRegisters(ArrayRef<Value> liveValues);
};
```

### 3. Runtime Integration Layer

Target IR includes a comprehensive runtime integration layer:

```cpp
class RuntimeIntegrator {
  // CUDA Runtime API integration
  void generateLauncherCode(FuncOp kernel);
  
  // NCCL collective operation lowering
  void lowerCollectiveOps(ModuleOp module);
  
  // Memory management code generation
  void generateMemoryManagement(ModuleOp module);
};
```

## Code Generation Strategy

### Compilation Phases

Target IR compilation proceeds through several distinct phases:

#### Phase 1: Analysis and Planning
- **Resource Requirements**: Shared memory, registers, barriers
- **Optimization Opportunities**: Vectorization, fusion, pipelining
- **Hardware Constraints**: Occupancy limits, memory bandwidth

#### Phase 2: Instruction Selection
- **Operation Mapping**: Tile IR → Target instructions
- **Optimization Patterns**: Peephole optimizations
- **Resource Allocation**: Registers, shared memory regions

#### Phase 3: Code Emission
- **Assembly Generation**: PTX, CUDA Tile IR, or LLVM IR
- **Metadata Creation**: Launch parameters, resource usage
- **Runtime Linking**: Host-side launcher generation

#### Phase 4: Binary Generation
- **Compilation**: NVCC, HIP compiler, or DPC++
- **Optimization**: Architecture-specific tuning
- **Packaging**: Deployment-ready artifacts

### Example: Matrix Multiplication Lowering

Consider how a simple matrix multiplication flows through Target IR:

```mlir
// Input: Tile IR
%result = tile.mma %A, %B, %C : 
  memref<16x16xbf16, 3>, memref<16x16xbf16, 3>, memref<16x16xf32, 5> -> 
  memref<16x16xf32, 5>
```

#### sm_90 (Hopper) Target:
```ptx
// Output: PTX with WGMMA
wgmma.mma_async.sync.m64n256k32.f32.bf16.bf16 
    {%f0, %f1, %f2, %f3}, smem_a+0, smem_b+0, 1;
```

#### sm_80 (Ampere) Target:
```ptx  
// Output: PTX with WMMA
wmma.load.a.sync.aligned.m16n16k16.global.bf16 {%r0, %r1, %r2, %r3}, [%rd0], 16;
wmma.load.b.sync.aligned.m16n16k16.global.bf16 {%r4, %r5, %r6, %r7}, [%rd1], 16;
wmma.mma.sync.aligned.m16n16k16.f32.bf16.bf16.f32 
    {%f0, %f1, %f2, %f3}, {%r0, %r1, %r2, %r3}, {%r4, %r5, %r6, %r7}, {%f0, %f1, %f2, %f3};
```

## Performance Optimization Strategies

### Hardware-Specific Optimizations

Target IR applies numerous hardware-specific optimizations:

#### Occupancy Optimization
- **Register Pressure Management**: Minimize register usage per thread
- **Shared Memory Layout**: Optimize for bank conflict avoidance
- **Thread Block Sizing**: Maximize SM utilization

#### Memory Bandwidth Optimization  
- **Coalesced Access**: Ensure optimal memory access patterns
- **Cache Utilization**: Leverage L1/L2 caches effectively
- **Async Pipelining**: Overlap memory and compute operations

#### Instruction Throughput Optimization
- **Tensor Core Utilization**: Maximize WMMA/WGMMA usage
- **Warp Scheduling**: Minimize warp divergence
- **Pipeline Balancing**: Balance memory and compute stages

### Cross-Architecture Patterns

Some optimizations apply across all target architectures:

- **Loop Unrolling**: Reduce branch overhead
- **Constant Propagation**: Eliminate redundant operations
- **Dead Code Elimination**: Remove unused computations
- **Register Coalescing**: Reduce register pressure

## Debugging and Profiling Support

### Debug Information Generation

Target IR can generate comprehensive debug information:

```cpp
class DebugInfoGenerator {
  // Maps generated code back to source
  void generateLineInfo(Operation* op, TargetInstruction& instr);
  
  // Creates NVTX markers for profiling
  void insertProfilingMarkers(FuncOp kernel);
  
  // Generates performance counter instrumentation  
  void addPerformanceCounters(ModuleOp module);
};
```

### Performance Analysis Integration

- **NVTX Ranges**: Automatic insertion of profiling markers
- **Performance Counters**: Hardware performance monitoring
- **Occupancy Analysis**: Theoretical vs. achieved occupancy
- **Roofline Modeling**: Compute vs. memory bound analysis

## Quality Assurance and Testing

### Correctness Verification

Target IR includes extensive testing infrastructure:

- **Numerical Accuracy Tests**: Verify mathematical correctness
- **Cross-Architecture Validation**: Ensure consistent results
- **Regression Testing**: Prevent performance degradation
- **Integration Testing**: End-to-end validation

### Performance Benchmarking

- **Automated Performance Testing**: CI/CD integration
- **Architecture Comparison**: Performance across GPU generations
- **Scalability Analysis**: Multi-GPU performance characteristics
- **Energy Efficiency Metrics**: Performance per watt analysis

## Future Architecture Extensions

### Planned Target Additions

#### AMD ROCm Support
- **LLVM IR Backend**: Integration with AMD's compiler stack
- **HIP Runtime Integration**: Host-side code generation
- **MFMA Instruction Support**: Matrix acceleration
- **rccl Integration**: Distributed computing support

#### Intel oneAPI Support  
- **DPC++ Integration**: SYCL-based compilation
- **Level Zero Runtime**: Low-level GPU control
- **DPAS Instruction Support**: Matrix acceleration
- **oneCCL Integration**: Collective operations

### Emerging Hardware Support

Target IR is designed to accommodate future hardware:

- **Next-Generation Tensor Units**: Flexible instruction selection
- **New Memory Hierarchies**: Extensible memory models
- **Quantum-Classical Hybrid**: Modular backend architecture
- **Neuromorphic Accelerators**: Pluggable computation models

## Summary

Target IR represents the culmination of Tessera's compiler pipeline, transforming high-level mathematical specifications into optimized, hardware-specific executable code. Its architecture balances:

- **Performance**: Aggressive hardware-specific optimization
- **Portability**: Support for multiple GPU vendors
- **Productivity**: Automated deployment and debugging
- **Extensibility**: Clean interfaces for future hardware

The following documents in this series will explore each major component of the Target IR system in detail, providing comprehensive coverage of code generation, runtime integration, deployment, and performance analysis.

---

**Next**: Document 2 covers the complete Flash Attention example, showing the full transformation from Tile IR input through final PTX assembly output.