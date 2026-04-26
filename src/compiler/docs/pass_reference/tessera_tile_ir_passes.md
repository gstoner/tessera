# Tessera Tile IR Passes

Tile IR is the lowest level of Tessera's portable IR stack, operating directly on hardware concepts like thread blocks, warps, shared memory, and tensor cores. This document details the major passes that transform Schedule IR into optimized Tile IR and prepare for target-specific code generation.

## Tile IR Structure

Tile IR operations map directly to GPU hardware concepts:

- **Thread Operations**: `tile.thread_id`, `tile.warp_id`, `tile.block_id`
- **Memory Operations**: `tile.load_shared`, `tile.store_global`, `tile.barrier`
- **Compute Operations**: `tile.mma`, `tile.reduce`, `tile.broadcast`
- **Async Operations**: `tile.cp_async`, `tile.wait_group`
- **Collective Operations**: `tile.allreduce`, `tile.allgather`

### Example Tile IR
```mlir
// Flash Attention kernel in Tile IR
func @flash_attention_tile(
  %Q_global: memref<?x?x?x?xbf16>,
  %K_global: memref<?x?x?x?xbf16>, 
  %V_global: memref<?x?x?x?xbf16>,
  %O_global: memref<?x?x?x?xbf16>) {
  
  // Thread and block identification
  %thread_id = tile.thread_id : index
  %warp_id = tile.warp_id : index
  %block_id = tile.block_id : index
  
  // Shared memory allocation with bank conflict avoidance
  %smem_q = tile.alloc_shared {swizzle = "xor"} : memref<128x128xbf16, 3>
  %smem_k = tile.alloc_shared {swizzle = "xor"} : memref<128x128xbf16, 3>
  %smem_v = tile.alloc_shared {swizzle = "xor"} : memref<128x128xbf16, 3>
  
  // Register allocation for accumulators
  %acc = tile.alloc_register : memref<8x8xf32, 5>
  %m_state = tile.alloc_register : memref<8xf32, 5>
  %l_state = tile.alloc_register : memref<8xf32, 5>
  
  // Initialize states
  %neg_inf = arith.constant -3.40282347e+38 : f32
  %zero = arith.constant 0.0 : f32
  tile.fill %m_state, %neg_inf : memref<8xf32, 5>
  tile.fill %l_state, %zero : memref<8xf32, 5>
  tile.fill %acc, %zero : memref<8x8xf32, 5>
  
  // Compute Q block coordinates
  %batch_heads = memref.dim %Q_global, %c0 : memref<?x?x?x?xbf16>
  %seq_len = memref.dim %Q_global, %c2 : memref<?x?x?x?xbf16>
  %head_dim = memref.dim %Q_global, %c3 : memref<?x?x?x?xbf16>
  
  %c128 = arith.constant 128 : index
  %bh_idx = arith.divui %block_id, %c128 : index
  %q_block_start = arith.remui %block_id, %c128 : index
  %q_block_start_scaled = arith.muli %q_block_start, %c128 : index
  
  // Load Q block asynchronously
  %q_slice = memref.subview %Q_global[%bh_idx, 0, %q_block_start_scaled, 0]
                                    [1, 1, 128, %head_dim] [1, 1, 1, 1]
                                    : memref<?x?x?x?xbf16> to memref<128x?xbf16>
  
  tile.cp_async %q_slice, %smem_q {bypass_l1 = true} 
               : memref<128x?xbf16>, memref<128x128xbf16, 3>
  tile.cp_commit_group
  
  // Loop over K/V blocks
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %kv_block = %c0 to %seq_len step %c128 {
    // Load K, V blocks with double buffering
    %k_slice = memref.subview %K_global[%bh_idx, 0, %kv_block, 0]
                                      [1, 1, 128, %head_dim] [1, 1, 1, 1]
                                      : memref<?x?x?x?xbf16> to memref<128x?xbf16>
    %v_slice = memref.subview %V_global[%bh_idx, 0, %kv_block, 0] 
                                      [1, 1, 128, %head_dim] [1, 1, 1, 1]
                                      : memref<?x?x?x?xbf16> to memref<128x?xbf16>
    
    tile.cp_async %k_slice, %smem_k {bypass_l1 = true}
                 : memref<128x?xbf16>, memref<128x128xbf16, 3>
    tile.cp_async %v_slice, %smem_v {bypass_l1 = true}
                 : memref<128x?xbf16>, memref<128x128xbf16, 3>
    tile.cp_commit_group
    tile.cp_wait_group 0
    tile.barrier
    
    // Compute attention scores using warp-level MMA
    %scores = tile.mma %smem_q, %smem_k {
      transpose_b = true,
      accumulate = true,
      layout = "row_major"
    } : memref<128x128xbf16, 3>, memref<128x128xbf16, 3> -> memref<8x8xf32, 5>
    
    // Scale scores
    %scale = arith.constant 0.125 : f32  // 1/sqrt(64)
    tile.broadcast_scale %scores, %scale : memref<8x8xf32, 5>, f32
    
    // Apply causal mask
    %q_pos = arith.addi %q_block_start_scaled, %thread_id : index
    %kv_pos = arith.addi %kv_block, %thread_id : index
    %is_causal = arith.cmpi slt, %q_pos, %kv_pos : index
    scf.if %is_causal {
      %mask_val = arith.constant -3.40282347e+38 : f32
      tile.mask_fill %scores, %mask_val : memref<8x8xf32, 5>, f32
    }
    
    // Online softmax update
    %m_new = tile.row_max %scores : memref<8x8xf32, 5> -> memref<8xf32, 5>
    %m_max = tile.element_max %m_state, %m_new : memref<8xf32, 5>, memref<8xf32, 5> -> memref<8xf32, 5>
    
    // Compute exponentials and new normalizer
    %alpha = tile.exp_diff %m_state, %m_max : memref<8xf32, 5>, memref<8xf32, 5> -> memref<8xf32, 5>
    %beta = tile.exp_diff %m_new, %m_max : memref<8xf32, 5>, memref<8xf32, 5> -> memref<8xf32, 5>
    
    %exp_scores = tile.exp_subtract %scores, %m_max : memref<8x8xf32, 5>, memref<8xf32, 5> -> memref<8x8xf32, 5>
    %row_sum = tile.row_sum %exp_scores : memref<8x8xf32, 5> -> memref<8xf32, 5>
    %l_new = tile.fma %alpha, %l_state, tile.mul(%beta, %row_sum) : 
             memref<8xf32, 5>, memref<8xf32, 5>, memref<8xf32, 5> -> memref<8xf32, 5>
    
    // Update accumulator
    tile.scale_accumulator %acc, %alpha : memref<8x8xf32, 5>, memref<8xf32, 5>
    
    // Compute P = exp(scores - m_max) / row_sum and convert to bf16
    %prob_f32 = tile.div_broadcast %exp_scores, %row_sum : 
                memref<8x8xf32, 5>, memref<8xf32, 5> -> memref<8x8xf32, 5>
    %prob = tile.cast %prob_f32 : memref<8x8xf32, 5> to memref<8x8xbf16, 5>
    
    // Accumulate: acc += P @ V
    %v_update = tile.mma %prob, %smem_v {
      accumulate = true,
      layout = "row_major"  
    } : memref<8x8xbf16, 5>, memref<128x128xbf16, 3> -> memref<8x8xf32, 5>
    
    tile.accumulate %acc, %v_update : memref<8x8xf32, 5>, memref<8x8xf32, 5>
    
    // Update states
    %m_state = %m_max : memref<8xf32, 5>
    %l_state = %l_new : memref<8xf32, 5>
  }
  
  // Finalize: acc = acc / l_state
  %final_output = tile.div_broadcast %acc, %l_state : 
                  memref<8x8xf32, 5>, memref<8xf32, 5> -> memref<8x8xf32, 5>
  %final_bf16 = tile.cast %final_output : memref<8x8xf32, 5> to memref<8x8xbf16, 5>
  
  // Store result
  %o_slice = memref.subview %O_global[%bh_idx, 0, %q_block_start_scaled, 0]
                                    [1, 1, 128, %head_dim] [1, 1, 1, 1]
                                    : memref<?x?x?x?xbf16> to memref<128x?xbf16>
  tile.store_global %final_bf16, %o_slice {coalesce = true}
                   : memref<8x8xbf16, 5>, memref<128x?xbf16>
  
  return
}
```

## Major Tile IR Passes

### 1. Memory Management Pass

This pass optimizes shared memory allocation, bank conflict avoidance, and register allocation.

```cpp
class TileMemoryManagementPass : public PassWrapper<TileMemoryManagementPass, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Analyze memory usage patterns
    MemoryUsageAnalysis analysis(func);
    
    // Optimize shared memory allocations
    optimizeSharedMemory(func, analysis);
    
    // Optimize register allocation
    optimizeRegisterAllocation(func, analysis);
    
    // Insert memory coalescing hints
    insertCoalescingHints(func);
  }

private:
  class MemoryUsageAnalysis {
  public:
    MemoryUsageAnalysis(FuncOp func) {
      analyzeMemoryAccess(func);
    }
    
    struct AccessPattern {
      SmallVector<Operation*> loads;
      SmallVector<Operation*> stores;
      bool hasStride1Access = false;
      bool hasBankConflicts = false;
      int64_t totalSize = 0;
    };
    
    const AccessPattern& getAccessPattern(Value memref) const {
      auto it = accessPatterns_.find(memref);
      assert(it != accessPatterns_.end());
      return it->second;
    }
    
    SmallVector<Value> getSharedMemoryAllocations() const {
      SmallVector<Value> allocations;
      for (const auto& [memref, pattern] : accessPatterns_) {
        if (isSharedMemory(memref)) {
          allocations.push_back(memref);
        }
      }
      return allocations;
    }
    
  private:
    DenseMap<Value, AccessPattern> accessPatterns_;
    
    void analyzeMemoryAccess(FuncOp func) {
      func.walk([&](Operation* op) {
        if (auto loadOp = dyn_cast<tile::LoadSharedOp>(op)) {
          Value memref = loadOp.getMemref();
          auto& pattern = accessPatterns_[memref];
          pattern.loads.push_back(op);
          analyzeAccessPattern(loadOp, pattern);
        } else if (auto storeOp = dyn_cast<tile::StoreSharedOp>(op)) {
          Value memref = storeOp.getMemref();
          auto& pattern = accessPatterns_[memref];
          pattern.stores.push_back(op);
          analyzeAccessPattern(storeOp, pattern);
        }
      });
    }
    
    void analyzeAccessPattern(Operation* op, AccessPattern& pattern) {
      // Check for stride-1 access patterns
      if (hasStride1Access(op)) {
        pattern.hasStride1Access = true;
      }
      
      // Check for potential bank conflicts
      if (hasPotentialBankConflicts(op)) {
        pattern.hasBankConflicts = true;
      }
    }
    
    bool hasStride1Access(Operation* op) {
      // Analyze index expressions to detect unit stride
      // This is a simplified implementation
      return true; // Placeholder
    }
    
    bool hasPotentialBankConflicts(Operation* op) {
      // Detect access patterns that could cause bank conflicts
      // This is a simplified implementation 
      return false; // Placeholder
    }
    
    bool isSharedMemory(Value memref) const {
      auto memrefType = memref.getType().dyn_cast<MemRefType>();
      if (!memrefType) return false;
      
      auto spaceAttr = memrefType.getMemorySpace();
      if (auto intAttr = spaceAttr.dyn_cast_or_null<IntegerAttr>()) {
        return intAttr.getInt() == 3; // Shared memory space
      }
      return false;
    }
  };
  
  void optimizeSharedMemory(FuncOp func, const MemoryUsageAnalysis& analysis) {
    OpBuilder builder(&getContext());
    
    for (Value allocation : analysis.getSharedMemoryAllocations()) {
      const auto& pattern = analysis.getAccessPattern(allocation);
      
      if (pattern.hasBankConflicts) {
        // Apply swizzling to avoid bank conflicts
        applySwizzling(allocation, builder);
      }
      
      // Optimize memory layout for coalesced access
      if (pattern.hasStride1Access) {
        optimizeForCoalescing(allocation, builder);
      }
    }
  }
  
  void applySwizzling(Value allocation, OpBuilder& builder) {
    auto definingOp = allocation.getDefiningOp();
    if (auto allocOp = dyn_cast<tile::AllocSharedOp>(definingOp)) {
      builder.setInsertionPoint(allocOp);
      
      // Create swizzled allocation
      auto swizzledOp = builder.create<tile::AllocSharedOp>(
        allocOp.getLoc(), allocOp.getType());
      swizzledOp->setAttr("swizzle", builder.getStringAttr("xor"));
      swizzledOp->setAttr("swizzle_size", builder.getI64IntegerAttr(128));
      
      // Replace uses
      allocOp.getResult().replaceAllUsesWith(swizzledOp.getResult());
      allocOp->erase();
    }
  }
  
  void optimizeRegisterAllocation(FuncOp func, const MemoryUsageAnalysis& analysis) {
    // Perform register pressure analysis
    RegisterPressureAnalysis regAnalysis(func);
    
    // Apply register spilling if needed
    if (regAnalysis.exceedsRegisterBudget()) {
      applyRegisterSpilling(func, regAnalysis);
    }
    
    // Optimize register reuse
    optimizeRegisterReuse(func, regAnalysis);
  }
  
  class RegisterPressureAnalysis {
  public:
    RegisterPressureAnalysis(FuncOp func) {
      computeRegisterPressure(func);
    }
    
    bool exceedsRegisterBudget() const {
      const int64_t MAX_REGISTERS_PER_THREAD = 255;
      return maxPressure_ > MAX_REGISTERS_PER_THREAD;
    }
    
    int64_t getMaxPressure() const { return maxPressure_; }
    
    SmallVector<Value> getSpillCandidates() const {
      return spillCandidates_;
    }
    
  private:
    int64_t maxPressure_ = 0;
    SmallVector<Value> spillCandidates_;
    
    void computeRegisterPressure(FuncOp func) {
      // Simplified register pressure computation
      int64_t currentPressure = 0;
      
      func.walk([&](Operation* op) {
        if (auto allocOp = dyn_cast<tile::AllocRegisterOp>(op)) {
          auto memrefType = allocOp.getType().cast<MemRefType>();
          int64_t size = computeRegisterSize(memrefType);
          currentPressure += size;
          maxPressure_ = std::max(maxPressure_, currentPressure);
          
          // Mark as spill candidate if large
          if (size > 32) {
            spillCandidates_.push_back(allocOp.getResult());
          }
        }
      });
    }
    
    int64_t computeRegisterSize(MemRefType type) {
      int64_t elements = 1;
      for (int64_t dim : type.getShape()) {
        elements *= dim;
      }
      
      int64_t elementSize = type.getElementType().getIntOrFloatBitWidth() / 8;
      return (elements * elementSize + 3) / 4; // Convert to 4-byte register units
    }
  };
  
  void applyRegisterSpilling(FuncOp func, const RegisterPressureAnalysis& analysis) {
    OpBuilder builder(&getContext());
    
    for (Value spillCandidate : analysis.getSpillCandidates()) {
      auto definingOp = spillCandidate.getDefiningOp<tile::AllocRegisterOp>();
      if (!definingOp) continue;
      
      builder.setInsertionPoint(definingOp);
      
      // Spill to shared memory instead
      auto memrefType = definingOp.getType().cast<MemRefType>();
      auto sharedType = MemRefType::get(
        memrefType.getShape(), memrefType.getElementType(),
        memrefType.getLayout(), builder.getI64IntegerAttr(3));
      
      auto sharedAlloc = builder.create<tile::AllocSharedOp>(
        definingOp.getLoc(), sharedType);
      
      definingOp.getResult().replaceAllUsesWith(sharedAlloc.getResult());
      definingOp->erase();
    }
  }
};
```

### 2. Intrinsic Lowering Pass

This pass lowers high-level Tile IR operations to hardware-specific intrinsics.

```cpp
class TileIntrinsicLoweringPass : public PassWrapper<TileIntrinsicLoweringPass, OperationPass<FuncOp>> {
public:
  TileIntrinsicLoweringPass(StringRef targetArch) : targetArch_(targetArch) {}
  
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Lower operations based on target architecture
    if (targetArch_ == "sm_90") {
      lowerForHopper(func);
    } else if (targetArch_ == "sm_80") {
      lowerForAmpere(func);
    } else {
      lowerGeneric(func);
    }
  }

private:
  StringRef targetArch_;
  
  void lowerForHopper(FuncOp func) {
    OpBuilder builder(&getContext());
    
    func.walk([&](Operation* op) {
      if (auto mmaOp = dyn_cast<tile::MmaOp>(op)) {
        lowerMmaForHopper(mmaOp, builder);
      } else if (auto copyOp = dyn_cast<tile::CpAsyncOp>(op)) {
        lowerCopyAsyncForHopper(copyOp, builder);
      } else if (auto reductionOp = dyn_cast<tile::RowMaxOp>(op)) {
        lowerReductionForHopper(reductionOp, builder);
      }
    });
  }
  
  void lowerMmaForHopper(tile::MmaOp op, OpBuilder& builder) {
    builder.setInsertionPoint(op);
    
    auto lhsType = op.getLhs().getType().cast<MemRefType>();
    auto rhsType = op.getRhs().getType().cast<MemRefType>();
    auto resultType = op.getResult().getType().cast<MemRefType>();
    
    // Determine appropriate WGMMA instruction
    std::string intrinsic;
    if (lhsType.getElementType().isBF16() && 
        rhsType.getElementType().isBF16() &&
        resultType.getElementType().isF32()) {
      // Use Hopper's WGMMA instruction
      intrinsic = "wgmma.mma_async.sync.m64n256k32.f32.bf16.bf16";
    } else if (lhsType.getElementType().isF16() &&
               rhsType.getElementType().isF16() &&
               resultType.getElementType().isF32()) {
      intrinsic = "wgmma.mma_async.sync.m64n256k32.f32.f16.f16";
    }
    
    auto wgmmaOp = builder.create<WGMMAOp>(
      op.getLoc(), resultType, op.getLhs(), op.getRhs());
    wgmmaOp->setAttr("intrinsic", builder.getStringAttr(intrinsic));
    
    if (op->hasAttr("transpose_b")) {
      wgmmaOp->setAttr("transpose_b", op->getAttr("transpose_b"));
    }
    
    op.getResult().replaceAllUsesWith(wgmmaOp.getResult());
    op->erase();
  }
  
  void lowerCopyAsyncForHopper(tile::CpAsyncOp op, OpBuilder& builder) {
    builder.setInsertionPoint(op);
    
    auto srcType = op.getSrc().getType().cast<MemRefType>();
    auto dstType = op.getDst().getType().cast<MemRefType>();
    
    // Use TMA (Tensor Memory Accelerator) for Hopper
    bool useTMA = (srcType.getMemorySpace() && 
                   dstType.getMemorySpace() &&
                   srcType.getMemorySpace().cast<IntegerAttr>().getInt() == 0 && // Global
                   dstType.getMemorySpace().cast<IntegerAttr>().getInt() == 3);  // Shared
    
    if (useTMA) {
      auto tmaOp = builder.create<TMALoadOp>(
        op.getLoc(), op.getSrc(), op.getDst());
      tmaOp->setAttr("tile_size", builder.getI64ArrayAttr({128, 128}));
      op->erase();
    } else {
      // Fallback to cp.async.bulk
      auto cpAsyncOp = builder.create<CpAsyncBulkOp>(
        op.getLoc(), op.getSrc(), op.getDst());
      cpAsyncOp->setAttr("bypass_l1", op->getAttr("bypass_l1"));
      op->erase();
    }
  }
  
  void lowerReductionForHopper(tile::RowMaxOp op, OpBuilder& builder) {
    builder.setInsertionPoint(op);
    
    // Use warp-level reduction primitives
    auto warpReduceOp = builder.create<WarpReduceMaxOp>(
      op.getLoc(), op.getResult().getType(), op.getInput());
    
    op.getResult().replaceAllUsesWith(warpReduceOp.getResult());
    op->erase();
  }
  
  void lowerForAmpere(FuncOp func) {
    OpBuilder builder(&getContext());
    
    func.walk([&](Operation* op) {
      if (auto mmaOp = dyn_cast<tile::MmaOp>(op)) {
        lowerMmaForAmpere(mmaOp, builder);
      } else if (auto copyOp = dyn_cast<tile::CpAsyncOp>(op)) {
        lowerCopyAsyncForAmpere(copyOp, builder);
      }
    });
  }
  
  void lowerMmaForAmpere(tile::MmaOp op, OpBuilder& builder) {
    builder.setInsertionPoint(op);
    
    // Use WMMA for Ampere
    auto wmmaOp = builder.create<WMMAOp>(
      op.getLoc(), op.getResult().getType(), 
      op.getLhs(), op.getRhs(), op.getAcc());
    
    wmmaOp->setAttr("shape", builder.getI64ArrayAttr({16, 16, 16}));
    wmmaOp->setAttr("layout", builder.getStringAttr("row_major"));
    
    op.getResult().replaceAllUsesWith(wmmaOp.getResult());
    op->erase();
  }
  
  void lowerCopyAsyncForAmpere(tile::CpAsyncOp op, OpBuilder& builder) {
    builder.setInsertionPoint(op);
    
    // Use cp.async.cg for Ampere
    auto cpAsyncOp = builder.create<CpAsyncCGOp>(
      op.getLoc(), op.getSrc(), op.getDst());
    
    if (op->hasAttr("bypass_l1")) {
      cpAsyncOp->setAttr("cache_level", builder.getStringAttr("L2"));
    }
    
    op->erase();
  }
};

// Hardware-specific intrinsic operations
class WGMMAOp : public Op<WGMMAOp> {
  // Hopper WGMMA instruction
  static StringRef getOperationName() { return "tile.wgmma"; }
  
  // This maps directly to PTX wgmma instruction in code generation
};

class WMMAOp : public Op<WMMAOp> {
  // Ampere WMMA instruction  
  static StringRef getOperationName() { return "tile.wmma"; }
};

class TMALoadOp : public Op<TMALoadOp> {
  // Hopper TMA (Tensor Memory Accelerator) load
  static StringRef getOperationName() { return "tile.tma_load"; }
};
```

### 3. Collective Insertion Pass

This pass inserts distributed communication operations for multi-GPU execution.

```cpp
class CollectiveInsertionPass : public PassWrapper<CollectiveInsertionPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Analyze distributed operations
    DistributedAnalysis analysis(module);
    
    // Insert required collectives
    insertCollectives(module, analysis);
  }

private:
  class DistributedAnalysis {
  public:
    DistributedAnalysis(ModuleOp module) {
      analyzeDistribution(module);
    }
    
    struct CollectiveRequirement {
      Operation* op;
      CollectiveKind kind;
      StringRef meshAxis;
      Value tensor;
    };
    
    enum class CollectiveKind {
      AllReduce,      // Sum across replicas
      AllGather,      // Gather sharded tensor
      ReduceScatter,  // Scatter reduced tensor
      AllToAll        // Transpose sharding
    };
    
    const SmallVector<CollectiveRequirement>& getRequirements() const {
      return requirements_;
    }
    
  private:
    SmallVector<CollectiveRequirement> requirements_;
    
    void analyzeDistribution(ModuleOp module) {
      module.walk([&](Operation* op) {
        if (auto matmulOp = dyn_cast<tile::MmaOp>(op)) {
          analyzeDistributedMatmul(matmulOp);
        } else if (auto attentionOp = dyn_cast<FlashAttentionOp>(op)) {
          analyzeDistributedAttention(attentionOp);
        }
      });
    }
    
    void analyzeDistributedMatmul(tile::MmaOp op) {
      // Check if this is tensor parallel matmul
      Value lhs = op.getLhs();
      Value rhs = op.getRhs();
      
      ShardingSpec lhsSharding = getShardingSpec(lhs);
      ShardingSpec rhsSharding = getShardingSpec(rhs);
      
      if (lhsSharding.isShardedOnAxis("tensor_parallel") && 
          !rhsSharding.isSharded()) {
        // A_sharded @ B -> needs AllReduce after
        requirements_.push_back({op, CollectiveKind::AllReduce, 
                                "tensor_parallel", op.getResult()});
      } else if (!lhsSharding.isSharded() && 
                 rhsSharding.isShardedOnAxis("tensor_parallel")) {
        // A @ B_sharded -> needs AllGather before
        requirements_.push_back({op, CollectiveKind::AllGather,
                                "tensor_parallel", rhs});
      }
    }
    
    void analyzeDistributedAttention(FlashAttentionOp op) {
      // Attention typically needs AllGather for full attention matrix
      // but FlashAttention can work with sharded K,V
      
      Value Q = op.getQ();
      Value K = op.getK();  
      Value V = op.getV();
      
      ShardingSpec kSharding = getShardingSpec(K);
      ShardingSpec vSharding = getShardingSpec(V);
      
      if (kSharding.isShardedOnAxis("tensor_parallel") ||
          vSharding.isShardedOnAxis("tensor_parallel")) {
        // Flash attention with sharded K,V needs reduce-scatter of output
        requirements_.push_back({op, CollectiveKind::ReduceScatter,
                                "tensor_parallel", op.getResult()});
      }
    }
    
    ShardingSpec getShardingSpec(Value tensor) {
      // Extract sharding specification from tensor attributes
      // This is a placeholder implementation
      return ShardingSpec{}; 
    }
  };
  
  void insertCollectives(ModuleOp module, const DistributedAnalysis& analysis) {
    OpBuilder builder(&getContext());
    
    for (const auto& req : analysis.getRequirements()) {
      insertCollective(req, builder);
    }
  }
  
  void insertCollective(const DistributedAnalysis::CollectiveRequirement& req, 
                       OpBuilder& builder) {
    switch (req.kind) {
      case DistributedAnalysis::CollectiveKind::AllReduce:
        insertAllReduce(req, builder);
        break;
      case DistributedAnalysis::CollectiveKind::AllGather:
        insertAllGather(req, builder);
        break;
      case DistributedAnalysis::CollectiveKind::ReduceScatter:
        insertReduceScatter(req, builder);
        break;
      case DistributedAnalysis::CollectiveKind::AllToAll:
        insertAllToAll(req, builder);
        break;
    }
  }
  
  void insertAllReduce(const DistributedAnalysis::CollectiveRequirement& req,
                      OpBuilder& builder) {
    builder.setInsertionPointAfter(req.op);
    
    auto allReduceOp = builder.create<tile::AllReduceOp>(
      req.op->getLoc(), req.tensor.getType(), req.tensor);
    
    allReduceOp->setAttr("mesh_axis", builder.getStringAttr(req.meshAxis));
    allReduceOp->setAttr("reduction_kind", builder.getStringAttr("sum"));
    allReduceOp->setAttr("implementation", builder.getStringAttr("nccl"));
    
    req.tensor.replaceAllUsesExcept(allReduceOp.getResult(), allReduceOp);
  }
  
  void insertAllGather(const DistributedAnalysis::CollectiveRequirement& req,
                      OpBuilder& builder) {
    builder.setInsertionPoint(req.op);
    
    auto gatheredType = computeGatheredType(req.tensor.getType(), req.meshAxis);
    
    auto allGatherOp = builder.create<tile::AllGatherOp>(
      req.op->getLoc(), gatheredType, req.tensor);
    
    allGatherOp->setAttr("mesh_axis", builder.getStringAttr(req.meshAxis));
    allGatherOp->setAttr("implementation", builder.getStringAttr("nccl"));
    
    // Update the operation to use gathered tensor
    for (int i = 0; i < req.op->getNumOperands(); ++i) {
      if (req.op->getOperand(i) == req.tensor) {
        req.op->setOperand(i, allGatherOp.getResult());
      }
    }
  }
  
  void insertReduceScatter(const DistributedAnalysis::CollectiveRequirement& req,
                          OpBuilder& builder) {
    builder.setInsertionPointAfter(req.op);
    
    auto scatteredType = computeScatteredType(req.tensor.getType(), req.meshAxis);
    
    auto reduceScatterOp = builder.create<tile::ReduceScatterOp>(
      req.op->getLoc(), scatteredType, req.tensor);
    
    reduceScatterOp->setAttr("mesh_axis", builder.getStringAttr(req.meshAxis));
    reduceScatterOp->setAttr("reduction_kind", builder.getStringAttr("sum"));
    reduceScatterOp->setAttr("implementation", builder.getStringAttr("nccl"));
    
    req.tensor.replaceAllUsesExcept(reduceScatterOp.getResult(), reduceScatterOp);
  }
  
  Type computeGatheredType(Type shardedType, StringRef meshAxis) {
    // Compute the type after all-gather (remove sharding)
    auto memrefType = shardedType.cast<MemRefType>();
    // This would expand the sharded dimension
    // Placeholder implementation
    return shardedType;
  }
  
  Type computeScatteredType(Type gatheredType, StringRef meshAxis) {
    // Compute the type after reduce-scatter (add sharding)
    auto memrefType = gatheredType.cast<MemRefType>();
    // This would shard one of the dimensions
    // Placeholder implementation  
    return gatheredType;
  }
};
```

### 4. Register Coalescing Pass

This pass optimizes register usage and memory coalescing patterns.

```cpp
class RegisterCoalescingPass : public PassWrapper<RegisterCoalescingPass, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Analyze register usage patterns
    RegisterUsageAnalysis analysis(func);
    
    // Perform coalescing optimizations
    performCoalescing(func, analysis);
    
    // Optimize memory access patterns
    optimizeMemoryAccesses(func);
  }

private:
  class RegisterUsageAnalysis {
  public:
    RegisterUsageAnalysis(FuncOp func) {
      analyzeRegisterUsage(func);
    }
    
    struct RegisterInfo {
      Value register_val;
      SmallVector<Operation*> uses;
      SmallVector<Operation*> defs;
      LivenessInterval liveness;
      bool canBeCoalesced = false;
    };
    
    struct LivenessInterval {
      int start = -1;
      int end = -1;
      bool overlaps(const LivenessInterval& other) const {
        return !(end < other.start || other.end < start);
      }
    };
    
    const RegisterInfo& getRegisterInfo(Value reg) const {
      auto it = registerInfo_.find(reg);
      assert(it != registerInfo_.end());
      return it->second;
    }
    
    SmallVector<std::pair<Value, Value>> getCoalescingCandidates() const {
      return coalescingCandidates_;
    }
    
  private:
    DenseMap<Value, RegisterInfo> registerInfo_;
    SmallVector<std::pair<Value, Value>> coalescingCandidates_;
    
    void analyzeRegisterUsage(FuncOp func) {
      // Number instructions for liveness analysis
      DenseMap<Operation*, int> instructionNumbers;
      int instrNum = 0;
      
      func.walk([&](Operation* op) {
        instructionNumbers[op] = instrNum++;
        
        // Analyze register allocations
        if (auto allocOp = dyn_cast<tile::AllocRegisterOp>(op)) {
          Value reg = allocOp.getResult();
          auto& info = registerInfo_[reg];
          info.register_val = reg;
          info.liveness.start = instrNum;
          
          // Find all uses
          for (auto& use : reg.getUses()) {
            Operation* user = use.getOwner();
            info.uses.push_back(user);
            int useNum = instructionNumbers[user];
            info.liveness.end = std::max(info.liveness.end, useNum);
          }
        }
      });
      
      // Find coalescing opportunities
      findCoalescingOpportunities();
    }
    
    void findCoalescingOpportunities() {
      SmallVector<Value> registers;
      for (const auto& [reg, info] : registerInfo_) {
        registers.push_back(reg);
      }
      
      // Look for copy operations that can be coalesced
      for (size_t i = 0; i < registers.size(); ++i) {
        for (size_t j = i + 1; j < registers.size(); ++j) {
          Value reg1 = registers[i];
          Value reg2 = registers[j];
          
          if (canCoalesce(reg1, reg2)) {
            coalescingCandidates_.emplace_back(reg1, reg2);
          }
        }
      }
    }
    
    bool canCoalesce(Value reg1, Value reg2) {
      const auto& info1 = registerInfo_[reg1];
      const auto& info2 = registerInfo_[reg2];
      
      // Check if liveness intervals don't overlap
      if (info1.liveness.overlaps(info2.liveness)) {
        return false;
      }
      
      // Check if there's a copy between them
      for (Operation* use : info1.uses) {
        if (auto copyOp = dyn_cast<tile::CopyOp>(use)) {
          if (copyOp.getDst() == reg2) {
            return true;
          }
        }
      }
      
      return false;
    }
  };
  
  void performCoalescing(FuncOp func, const RegisterUsageAnalysis& analysis) {
    OpBuilder builder(&getContext());
    
    for (const auto& [reg1, reg2] : analysis.getCoalescingCandidates()) {
      coalesceRegisters(reg1, reg2, builder);
    }
  }
  
  void coalesceRegisters(Value reg1, Value reg2, OpBuilder& builder) {
    // Replace all uses of reg2 with reg1
    reg2.replaceAllUsesWith(reg1);
    
    // Remove the copy operation
    for (auto& use : reg1.getUses()) {
      if (auto copyOp = dyn_cast<tile::CopyOp>(use.getOwner())) {
        if (copyOp.getDst() == reg2) {
          copyOp->erase();
          break;
        }
      }
    }
    
    // Remove reg2 allocation if it's no longer used
    if (auto allocOp = reg2.getDefiningOp<tile::AllocRegisterOp>()) {
      if (reg2.use_empty()) {
        allocOp->erase();
      }
    }
  }
  
  void optimizeMemoryAccesses(FuncOp func) {
    // Look for memory access patterns that can be vectorized
    func.walk([&](Operation* op) {
      if (auto loadOp = dyn_cast<tile::LoadGlobalOp>(op)) {
        optimizeLoad(loadOp);
      } else if (auto storeOp = dyn_cast<tile::StoreGlobalOp>(op)) {
        optimizeStore(storeOp);
      }
    });
  }
  
  void optimizeLoad(tile::LoadGlobalOp loadOp) {
    OpBuilder builder(loadOp);
    
    // Check if this load can be vectorized
    if (canVectorizeLoad(loadOp)) {
      // Replace with vectorized load
      auto vectorizedLoad = builder.create<tile::LoadGlobalVectorOp>(
        loadOp.getLoc(), loadOp.getResult().getType(), 
        loadOp.getMemref(), loadOp.getIndices());
      
      vectorizedLoad->setAttr("vector_width", builder.getI64IntegerAttr(4));
      vectorizedLoad->setAttr("coalesce", builder.getBoolAttr(true));
      
      loadOp.getResult().replaceAllUsesWith(vectorizedLoad.getResult());
      loadOp->erase();
    }
  }
  
  bool canVectorizeLoad(tile::LoadGlobalOp loadOp) {
    // Check if access pattern is suitable for vectorization
    // This is a simplified check
    return loadOp->hasAttr("stride_1") || 
           loadOp->hasAttr("coalesce_hint");
  }
};
```

### 5. Barrier Optimization Pass

This pass optimizes synchronization barriers and async operations.

```cpp
class BarrierOptimizationPass : public PassWrapper<BarrierOptimizationPass, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Analyze barrier usage
    BarrierAnalysis analysis(func);
    
    // Remove redundant barriers
    removeRedundantBarriers(func, analysis);
    
    // Optimize async wait operations
    optimizeAsyncWaits(func, analysis);
    
    // Insert missing barriers
    insertRequiredBarriers(func, analysis);
  }

private:
  class BarrierAnalysis {
  public:
    BarrierAnalysis(FuncOp func) {
      analyzeBarriers(func);
    }
    
    struct BarrierInfo {
      Operation* barrier;
      SmallVector<Operation*> synchronizedOps;
      bool isRedundant = false;
      bool isRequired = true;
    };
    
    const SmallVector<BarrierInfo>& getBarriers() const {
      return barriers_;
    }
    
    SmallVector<Operation*> getMissingBarriers() const {
      return missingBarriers_;
    }
    
  private:
    SmallVector<BarrierInfo> barriers_;
    SmallVector<Operation*> missingBarriers_;
    
    void analyzeBarriers(FuncOp func) {
      func.walk([&](Operation* op) {
        if (auto barrierOp = dyn_cast<tile::BarrierOp>(op)) {
          analyzeBarrier(barrierOp);
        }
      });
      
      // Find missing barriers
      findMissingBarriers(func);
    }
    
    void analyzeBarrier(tile::BarrierOp barrierOp) {
      BarrierInfo info;
      info.barrier = barrierOp;
      
      // Find operations synchronized by this barrier
      findSynchronizedOperations(barrierOp, info.synchronizedOps);
      
      // Check if barrier is redundant
      info.isRedundant = isBarrierRedundant(barrierOp, info.synchronizedOps);
      
      barriers_.push_back(info);
    }
    
    void findSynchronizedOperations(tile::BarrierOp barrier, 
                                  SmallVector<Operation*>& syncOps) {
      // Find shared memory operations before this barrier
      auto* currentOp = barrier->getPrevNode();
      while (currentOp) {
        if (isa<tile::StoreSharedOp, tile::LoadSharedOp, tile::CpAsyncOp>(currentOp)) {
          syncOps.push_back(currentOp);
        }
        
        // Stop at previous barrier
        if (isa<tile::BarrierOp>(currentOp)) {
          break;
        }
        
        currentOp = currentOp->getPrevNode();
      }
    }
    
    bool isBarrierRedundant(tile::BarrierOp barrier, 
                          const SmallVector<Operation*>& syncOps) {
      // Barrier is redundant if there are no shared memory writes
      // or if there's already a barrier right before
      if (syncOps.empty()) {
        return true;
      }
      
      auto* prevOp = barrier->getPrevNode();
      if (isa_and_nonnull<tile::BarrierOp>(prevOp)) {
        return true;
      }
      
      return false;
    }
    
    void findMissingBarriers(FuncOp func) {
      func.walk([&](Operation* op) {
        if (auto storeOp = dyn_cast<tile::StoreSharedOp>(op)) {
          if (needsBarrierAfter(storeOp)) {
            missingBarriers_.push_back(storeOp);
          }
        }
      });
    }
    
    bool needsBarrierAfter(tile::StoreSharedOp storeOp) {
      // Check if there are reads from the same shared memory location
      // without an intervening barrier
      Value memref = storeOp.getMemref();
      
      auto* nextOp = storeOp->getNextNode();
      while (nextOp) {
        if (isa<tile::BarrierOp>(nextOp)) {
          return false; // Found barrier, no need for another
        }
        
        if (auto loadOp = dyn_cast<tile::LoadSharedOp>(nextOp)) {
          if (loadOp.getMemref() == memref) {
            return true; // Found read without barrier
          }
        }
        
        nextOp = nextOp->getNextNode();
      }
      
      return false;
    }
  };
  
  void removeRedundantBarriers(FuncOp func, const BarrierAnalysis& analysis) {
    for (const auto& barrierInfo : analysis.getBarriers()) {
      if (barrierInfo.isRedundant) {
        barrierInfo.barrier->erase();
      }
    }
  }
  
  void optimizeAsyncWaits(FuncOp func, const BarrierAnalysis& analysis) {
    OpBuilder builder(&getContext());
    
    func.walk([&](tile::AsyncWaitOp waitOp) {
      optimizeWaitOperation(waitOp, builder);
    });
  }
  
  void optimizeWaitOperation(tile::AsyncWaitOp waitOp, OpBuilder& builder) {
    // Check if we can move the wait closer to where the data is needed
    SmallVector<Operation*> dependentOps;
    
    auto* nextOp = waitOp->getNextNode();
    while (nextOp) {
      if (usesAsyncData(nextOp, waitOp)) {
        dependentOps.push_back(nextOp);
        break; // Move wait just before first use
      }
      nextOp = nextOp->getNextNode();
    }
    
    if (!dependentOps.empty()) {
      builder.setInsertionPoint(dependentOps[0]);
      auto newWait = builder.clone(*waitOp);
      waitOp->erase();
    }
  }
  
  bool usesAsyncData(Operation* op, tile::AsyncWaitOp waitOp) {
    // Check if operation uses data from async copy
    // This would need more sophisticated analysis
    return isa<tile::LoadSharedOp, tile::MmaOp>(op);
  }
  
  void insertRequiredBarriers(FuncOp func, const BarrierAnalysis& analysis) {
    OpBuilder builder(&getContext());
    
    for (Operation* op : analysis.getMissingBarriers()) {
      builder.setInsertionPointAfter(op);
      builder.create<tile::BarrierOp>(op->getLoc());
    }
  }
};
```

### 6. Performance Annotation Pass

This pass adds performance hints and profiling annotations.

```cpp
class PerformanceAnnotationPass : public PassWrapper<PerformanceAnnotationPass, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Add performance annotations
    annotateHotPaths(func);
    annotateMemoryHints(func);
    addProfilingMarkers(func);
  }

private:
  void annotateHotPaths(FuncOp func) {
    // Identify loops and frequently executed operations
    func.walk([&](Operation* op) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        // Mark inner loops as hot
        if (isInnerLoop(forOp)) {
          forOp->setAttr("tessera.hot_path", UnitAttr::get(&getContext()));
          forOp->setAttr("tessera.unroll_hint", IntegerAttr::get(
            IntegerType::get(&getContext(), 64), 4));
        }
      } else if (isa<tile::MmaOp, tile::LoadGlobalOp>(op)) {
        // Mark compute-intensive operations
        op->setAttr("tessera.performance_critical", UnitAttr::get(&getContext()));
      }
    });
  }
  
  bool isInnerLoop(scf::ForOp loop) {
    // Check if this loop contains other loops
    bool hasNestedLoop = false;
    loop.walk([&](scf::ForOp nestedLoop) {
      if (nestedLoop != loop) {
        hasNestedLoop = true;
      }
    });
    return !hasNestedLoop;
  }
  
  void annotateMemoryHints(FuncOp func) {
    func.walk([&](Operation* op) {
      if (auto loadOp = dyn_cast<tile::LoadGlobalOp>(op)) {
        // Add cache hints
        if (isReusedData(loadOp)) {
          loadOp->setAttr("tessera.cache_hint", StringAttr::get(&getContext(), "L2"));
        } else {
          loadOp->setAttr("tessera.cache_hint", StringAttr::get(&getContext(), "streaming"));
        }
        
        // Add prefetch hints
        if (isPrefetchable(loadOp)) {
          loadOp->setAttr("tessera.prefetch", BoolAttr::get(&getContext(), true));
        }
      } else if (auto storeOp = dyn_cast<tile::StoreGlobalOp>(op)) {
        // Add write-back hints
        storeOp->setAttr("tessera.write_policy", StringAttr::get(&getContext(), "write_back"));
      }
    });
  }
  
  bool isReusedData(tile::LoadGlobalOp loadOp) {
    // Check if the same memory location is accessed multiple times
    Value memref = loadOp.getMemref();
    int useCount = 0;
    
    memref.getParentRegion()->walk([&](tile::LoadGlobalOp otherLoad) {
      if (otherLoad.getMemref() == memref) {
        useCount++;
      }
    });
    
    return useCount > 1;
  }
  
  bool isPrefetchable(tile::LoadGlobalOp loadOp) {
    // Check if load is in a predictable access pattern
    return loadOp->getParentOfType<scf::ForOp>() != nullptr;
  }
  
  void addProfilingMarkers(FuncOp func) {
    OpBuilder builder(&getContext());
    
    // Add function entry/exit markers
    builder.setInsertionPointToStart(&func.front());
    builder.create<ProfileMarkerOp>(
      func.getLoc(), 
      StringAttr::get(&getContext(), "function_entry"),
      StringAttr::get(&getContext(), func.getName()));
    
    // Add markers around expensive operations
    func.walk([&](Operation* op) {
      if (isa<tile::MmaOp, FlashAttentionOp>(op)) {
        builder.setInsertionPoint(op);
        builder.create<ProfileMarkerOp>(
          op->getLoc(),
          StringAttr::get(&getContext(), "compute_start"),
          StringAttr::get(&getContext(), op->getName().getStringRef()));
        
        builder.setInsertionPointAfter(op);
        builder.create<ProfileMarkerOp>(
          op->getLoc(),
          StringAttr::get(&getContext(), "compute_end"),
          StringAttr::get(&getContext(), op->getName().getStringRef()));
      }
    });
  }
};

class ProfileMarkerOp : public Op<ProfileMarkerOp> {
  static StringRef getOperationName() { return "tile.profile_marker"; }
  // This generates NVTX markers in the final code
};
```

## Testing Tile IR Passes

```cpp
// Unit tests for Tile IR passes
TEST(TileIRPasses, MemoryManagement) {
  MLIRContext context;
  context.loadDialect<TesseraDialect>();
  
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func @memory_test() {
      %smem = tile.alloc_shared : memref<128x128xbf16, 3>
      %reg = tile.alloc_register : memref<8x8xf32, 5>
      
      // Operations that cause bank conflicts
      %0 = tile.load_shared %smem[%thread_id, %thread_id] : memref<128x128xbf16, 3>
      
      return
    }
  )mlir", &context);
  
  PassManager pm(&context);
  pm.addPass(createTileMemoryManagementPass());
  
  ASSERT_TRUE(succeeded(pm.run(module.get())));
  
  // Verify swizzling was applied
  auto func = module->lookupSymbol<FuncOp>("memory_test");
  bool foundSwizzledAlloc = false;
  func.walk([&](tile::AllocSharedOp alloc) {
    if (alloc->hasAttr("swizzle")) {
      foundSwizzledAlloc = true;
    }
  });
  
  EXPECT_TRUE(foundSwizzledAlloc);
}

TEST(TileIRPasses, IntrinsicLowering) {
  MLIRContext context;
  context.loadDialect<TesseraDialect>();
  
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func @mma_test(%A: memref<16x16xbf16, 3>, %B: memref<16x16xbf16, 3>) -> memref<16x16xf32, 5> {
      %C = tile.alloc_register : memref<16x16xf32, 5>
      %result = tile.mma %A, %B, %C : memref<16x16xbf16, 3>, memref<16x16xbf16, 3>, memref<16x16xf32, 5> -> memref<16x16xf32, 5>
      return %result : memref<16x16xf32, 5>
    }
  )mlir", &context);
  
  PassManager pm(&context);
  pm.addPass(createTileIntrinsicLoweringPass("sm_90"));
  
  ASSERT_TRUE(succeeded(pm.run(module.get())));
  
  // Verify WGMMA instruction was generated
  auto func = module->lookupSymbol<FuncOp>("mma_test");
  bool foundWGMMA = false;
  func.walk([&](WGMMAOp wgmma) {
    foundWGMMA = true;
  });
  
  EXPECT_TRUE(foundWGMMA);
}

TEST(TileIRPasses, CollectiveInsertion) {
  MLIRContext context;
  context.loadDialect<TesseraDialect>();
  
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func @distributed_matmul(%A: memref<128x64xbf16> {sharding = "tensor_parallel"},
                            %B: memref<64x128xbf16>) -> memref<128x128xf32> {
      %C = tile.alloc_register : memref<128x128xf32, 5>
      %result = tile.mma %A, %B, %C : memref<128x64xbf16>, memref<64x128xbf16>, memref<128x128xf32, 5> -> memref<128x128xf32, 5>
      return %result : memref<128x128xf32, 5>
    }
  )mlir", &context);
  
  PassManager pm(&context());
  pm.addPass(createCollectiveInsertionPass());
  
  ASSERT_TRUE(succeeded(pm.run(module.get())));
  
  // Verify AllReduce was inserted
  auto func = module->lookupSymbol<FuncOp>("distributed_matmul");
  bool foundAllReduce = false;
  func.walk([&](tile::AllReduceOp allreduce) {
    foundAllReduce = true;
  });
  
  EXPECT_TRUE(foundAllReduce);
}
```

## Summary

Tile IR passes prepare code for efficient hardware execution:

- **Memory Management Pass** optimizes shared memory allocation and register usage
- **Intrinsic Lowering Pass** maps operations to hardware-specific instructions (WGMMA, WMMA, TMA)  
- **Collective Insertion Pass** adds distributed communication operations for multi-GPU execution
- **Register Coalescing Pass** optimizes register allocation and memory access patterns
- **Barrier Optimization Pass** removes redundant synchronization and optimizes async operations
- **Performance Annotation Pass** adds profiling markers and performance hints

These passes transform portable Tile IR into hardware-optimized code ready for target-specific code generation. The final document will cover **Target IR Passes** and code generation for specific backends like PTX and CUDA Tile IR.