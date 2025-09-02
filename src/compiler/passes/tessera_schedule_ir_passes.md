# Tessera Schedule IR Passes

Schedule IR is the middle level of Tessera's compiler stack, responsible for transforming high-level tensor operations into explicit loop nests with tiling, memory hierarchy management, and parallelization. This document details the major passes that operate on Schedule IR.

## Schedule IR Structure

Schedule IR extends standard MLIR loop constructs (`scf`, `affine`) with Tessera-specific operations:

- **Loop Operations**: `tessera.for`, `tessera.parallel_for` with tiling attributes
- **Memory Operations**: `tessera.alloc_shared`, `tessera.alloc_register`
- **Synchronization**: `tessera.barrier`, `tessera.async_wait`
- **Data Movement**: `tessera.copy_async`, `tessera.prefetch`

### Example Schedule IR
```mlir
// Flash Attention with explicit tiling and memory management
func @flash_attention_scheduled(
  %Q: memref<?x?x?x?xbf16>, 
  %K: memref<?x?x?x?xbf16>,
  %V: memref<?x?x?x?xbf16>,
  %O: memref<?x?x?x?xbf16>) {
  
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  
  %batch_heads = memref.dim %Q, %c0 : memref<?x?x?x?xbf16>
  %seq_len = memref.dim %Q, %c2 : memref<?x?x?x?xbf16>
  %head_dim = memref.dim %Q, %c3 : memref<?x?x?x?xbf16>
  
  // Parallel over batch and heads
  tessera.parallel_for %bh = %c0 to %batch_heads step %c1 {
    // Allocate shared memory with explicit sizes
    %smem_q = tessera.alloc_shared() : memref<128x128xbf16, 3>
    %smem_k = tessera.alloc_shared() : memref<128x128xbf16, 3>  
    %smem_v = tessera.alloc_shared() : memref<128x128xbf16, 3>
    
    // Tiled loop over sequence dimension (Q blocks)
    tessera.for %q_tile = %c0 to %seq_len step %c128 
                attributes {pipeline_stage = 0} {
      
      // Asynchronous copy Q tile to shared memory
      %q_view = memref.subview %Q[%bh, 0, %q_tile, 0] 
                              [1, 1, 128, %head_dim] 
                              [1, 1, 1, 1] 
                              : memref<?x?x?x?xbf16> to memref<128x?xbf16>
      tessera.copy_async %q_view, %smem_q 
                        {copy_kind = "global_to_shared", stages = 3} 
                        : memref<128x?xbf16>, memref<128x128xbf16, 3>
      
      // Initialize accumulators in registers
      %acc = tessera.alloc_register() : memref<128x128xf32, 5>
      %m_state = tessera.alloc_register() {value = -inf} : memref<128xf32, 5>
      %l_state = tessera.alloc_register() {value = 0.0} : memref<128xf32, 5>
      
      // Tiled loop over K/V sequence dimension
      tessera.for %kv_tile = %c0 to %seq_len step %c128 
                  attributes {pipeline_stage = 1} {
        
        // Async copy K, V tiles with double buffering
        %k_view = memref.subview %K[%bh, 0, %kv_tile, 0] 
                                [1, 1, 128, %head_dim] 
                                [1, 1, 1, 1]
                                : memref<?x?x?x?xbf16> to memref<128x?xbf16>
        %v_view = memref.subview %V[%bh, 0, %kv_tile, 0] 
                                [1, 1, 128, %head_dim] 
                                [1, 1, 1, 1]
                                : memref<?x?x?x?xbf16> to memref<128x?xbf16>
        
        tessera.copy_async %k_view, %smem_k 
                          {copy_kind = "global_to_shared", 
                           double_buffer = true} 
                          : memref<128x?xbf16>, memref<128x128xbf16, 3>
        tessera.copy_async %v_view, %smem_v 
                          {copy_kind = "global_to_shared",
                           double_buffer = true} 
                          : memref<128x?xbf16>, memref<128x128xbf16, 3>
        
        // Wait for async copies to complete
        tessera.async_wait {group = "copy_group"} : () -> ()
        tessera.barrier : () -> ()
        
        // Compute attention scores: Q @ K^T
        %scores = tessera.matmul_tiled %smem_q, %smem_k 
                                     {transpose_b = true,
                                      tile_m = 128, tile_n = 128, tile_k = 128}
                                     : memref<128x128xbf16, 3>, 
                                       memref<128x128xbf16, 3> -> 
                                       memref<128x128xf32, 5>
        
        // Apply scaling
        %scale = arith.constant 0.125 : f32  // 1/sqrt(64)
        tessera.scale_inplace %scores, %scale 
                             : memref<128x128xf32, 5>, f32
        
        // Apply causal mask if needed
        %causal_cond = arith.cmpi sgt, %kv_tile, %q_tile : index
        scf.if %causal_cond {
          tessera.apply_causal_mask %scores, %q_tile, %kv_tile 
                                   : memref<128x128xf32, 5>, index, index
        }
        
        // Online softmax update
        %m_new, %l_new = tessera.online_softmax_update %scores, %m_state, %l_state
                                                       : memref<128x128xf32, 5>,
                                                         memref<128xf32, 5>,
                                                         memref<128xf32, 5> ->
                                                         memref<128xf32, 5>,
                                                         memref<128xf32, 5>
        
        // Update accumulator: acc = acc * correction + P @ V
        %prob = tessera.compute_prob %scores, %m_new 
                                    : memref<128x128xf32, 5>, memref<128xf32, 5> -> 
                                      memref<128x128xbf16, 5>
        %update = tessera.matmul_tiled %prob, %smem_v
                                      {tile_m = 128, tile_n = 128, tile_k = 128}
                                      : memref<128x128xbf16, 5>,
                                        memref<128x128xbf16, 3> ->
                                        memref<128x128xf32, 5>
        
        tessera.accumulate_with_correction %acc, %update, %m_state, %m_new, %l_state, %l_new
                                          : memref<128x128xf32, 5>, memref<128x128xf32, 5>,
                                            memref<128xf32, 5>, memref<128xf32, 5>,
                                            memref<128xf32, 5>, memref<128xf32, 5>
        
        // Update states
        %m_state = %m_new : memref<128xf32, 5>
        %l_state = %l_new : memref<128xf32, 5>
      }
      
      // Finalize output: acc / l_state
      tessera.finalize_softmax %acc, %l_state 
                              : memref<128x128xf32, 5>, memref<128xf32, 5>
      
      // Cast and store result
      %result_bf16 = tessera.cast %acc : memref<128x128xf32, 5> to memref<128x128xbf16, 5>
      %o_view = memref.subview %O[%bh, 0, %q_tile, 0] 
                              [1, 1, 128, %head_dim] 
                              [1, 1, 1, 1]
                              : memref<?x?x?x?xbf16> to memref<128x?xbf16>
      tessera.copy_async %result_bf16, %o_view 
                        {copy_kind = "register_to_global"}
                        : memref<128x128xbf16, 5>, memref<128x?xbf16>
    }
  }
  return
}
```

## Major Schedule IR Passes

### 1. Tiling Pass

The tiling pass transforms loops into tiled versions with configurable tile sizes and shapes.

```cpp
class TilingPass : public PassWrapper<TilingPass, OperationPass<FuncOp>> {
public:
  TilingPass(ArrayRef<int64_t> tileSizes) : tileSizes_(tileSizes.begin(), tileSizes.end()) {}
  
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Find tileable operations
    SmallVector<Operation*> tileableOps;
    func.walk([&](Operation* op) {
      if (auto tileableOp = dyn_cast<TileableOpInterface>(op)) {
        tileableOps.push_back(op);
      }
    });
    
    // Apply tiling transformations
    for (auto* op : tileableOps) {
      applyTiling(op);
    }
  }

private:
  SmallVector<int64_t> tileSizes_;
  
  void applyTiling(Operation* op) {
    OpBuilder builder(op);
    
    if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
      tileMatmul(matmulOp, builder);
    } else if (auto attentionOp = dyn_cast<FlashAttentionOp>(op)) {
      tileAttention(attentionOp, builder);
    } else if (auto reductionOp = dyn_cast<ReductionOp>(op)) {
      tileReduction(reductionOp, builder);
    }
  }
  
  void tileMatmul(MatmulOp op, OpBuilder& builder) {
    // Extract tile sizes (default to 128x128x64)
    int64_t tileM = tileSizes_.size() > 0 ? tileSizes_[0] : 128;
    int64_t tileN = tileSizes_.size() > 1 ? tileSizes_[1] : 128;  
    int64_t tileK = tileSizes_.size() > 2 ? tileSizes_[2] : 64;
    
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value result = op.getResult();
    
    // Get tensor dimensions
    auto lhsType = lhs.getType().cast<MemRefType>();
    auto rhsType = rhs.getType().cast<MemRefType>();
    auto resultType = result.getType().cast<MemRefType>();
    
    // Create tiled loops
    SmallVector<Value> lbs, ubs, steps;
    
    // M dimension
    lbs.push_back(builder.create<arith::ConstantIndexOp>(op.getLoc(), 0));
    ubs.push_back(builder.create<memref::DimOp>(op.getLoc(), lhs, 0));
    steps.push_back(builder.create<arith::ConstantIndexOp>(op.getLoc(), tileM));
    
    // N dimension  
    lbs.push_back(builder.create<arith::ConstantIndexOp>(op.getLoc(), 0));
    ubs.push_back(builder.create<memref::DimOp>(op.getLoc(), rhs, 1));
    steps.push_back(builder.create<arith::ConstantIndexOp>(op.getLoc(), tileN));
    
    // K dimension
    lbs.push_back(builder.create<arith::ConstantIndexOp>(op.getLoc(), 0));
    ubs.push_back(builder.create<memref::DimOp>(op.getLoc(), lhs, 1));
    steps.push_back(builder.create<arith::ConstantIndexOp>(op.getLoc(), tileK));
    
    // Create nested loops: for m, for n, for k
    auto outerLoop = builder.create<scf::ForOp>(
      op.getLoc(), lbs[0], ubs[0], steps[0], ValueRange{});
    
    builder.setInsertionPointToStart(outerLoop.getBody());
    Value m_idx = outerLoop.getInductionVar();
    
    auto middleLoop = builder.create<scf::ForOp>(
      op.getLoc(), lbs[1], ubs[1], steps[1], ValueRange{});
    
    builder.setInsertionPointToStart(middleLoop.getBody());
    Value n_idx = middleLoop.getInductionVar();
    
    auto innerLoop = builder.create<scf::ForOp>(
      op.getLoc(), lbs[2], ubs[2], steps[2], ValueRange{});
    
    builder.setInsertionPointToStart(innerLoop.getBody());
    Value k_idx = innerLoop.getInductionVar();
    
    // Create tile subviews
    auto tileM_val = builder.create<arith::ConstantIndexOp>(op.getLoc(), tileM);
    auto tileN_val = builder.create<arith::ConstantIndexOp>(op.getLoc(), tileN);
    auto tileK_val = builder.create<arith::ConstantIndexOp>(op.getLoc(), tileK);
    
    auto lhsTile = builder.create<memref::SubViewOp>(
      op.getLoc(), lhs,
      ValueRange{m_idx, k_idx},           // offsets
      ValueRange{tileM_val, tileK_val},   // sizes  
      ValueRange{});                      // strides (default)
    
    auto rhsTile = builder.create<memref::SubViewOp>(
      op.getLoc(), rhs,
      ValueRange{k_idx, n_idx},
      ValueRange{tileK_val, tileN_val},
      ValueRange{});
    
    auto resultTile = builder.create<memref::SubViewOp>(
      op.getLoc(), result,
      ValueRange{m_idx, n_idx},
      ValueRange{tileM_val, tileN_val},
      ValueRange{});
    
    // Create tiled matmul operation
    builder.create<TiledMatmulOp>(
      op.getLoc(), 
      lhsTile.getResult(), 
      rhsTile.getResult(), 
      resultTile.getResult());
    
    // Remove original operation
    op->erase();
  }
  
  void tileAttention(FlashAttentionOp op, OpBuilder& builder) {
    // Attention-specific tiling with online softmax
    int64_t blockM = tileSizes_.size() > 0 ? tileSizes_[0] : 128;
    int64_t blockN = tileSizes_.size() > 1 ? tileSizes_[1] : 128;
    
    Value Q = op.getQ();
    Value K = op.getK();
    Value V = op.getV();
    Value O = op.getO();
    
    // Create tiled attention structure
    createAttentionTileStructure(builder, op.getLoc(), Q, K, V, O, blockM, blockN);
    
    op->erase();
  }
  
  void createAttentionTileStructure(OpBuilder& builder, Location loc,
                                   Value Q, Value K, Value V, Value O,
                                   int64_t blockM, int64_t blockN) {
    // Get sequence length
    Value seqLen = builder.create<memref::DimOp>(loc, Q, 2);
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value stepM = builder.create<arith::ConstantIndexOp>(loc, blockM);
    Value stepN = builder.create<arith::ConstantIndexOp>(loc, blockN);
    
    // Outer loop over Q tiles (batch-heads parallelized separately)
    auto qLoop = builder.create<scf::ForOp>(loc, c0, seqLen, stepM);
    builder.setInsertionPointToStart(qLoop.getBody());
    
    Value q_start = qLoop.getInductionVar();
    
    // Allocate accumulator and softmax state
    auto accType = MemRefType::get({blockM, -1}, builder.getF32Type(), 
                                  {}, builder.getI64IntegerAttr(5)); // register space
    auto stateType = MemRefType::get({blockM}, builder.getF32Type(),
                                    {}, builder.getI64IntegerAttr(5));
    
    auto acc = builder.create<AllocOp>(loc, accType);
    auto mState = builder.create<AllocOp>(loc, stateType);
    auto lState = builder.create<AllocOp>(loc, stateType);
    
    // Initialize states
    Value negInf = builder.create<arith::ConstantFloatOp>(
      loc, APFloat::getInf(builder.getF32Type().getFloatSemantics(), true), 
      builder.getF32Type());
    Value zero = builder.create<arith::ConstantFloatOp>(
      loc, APFloat(0.0f), builder.getF32Type());
    
    builder.create<FillOp>(loc, ValueRange{negInf}, mState);
    builder.create<FillOp>(loc, ValueRange{zero}, lState);
    builder.create<FillOp>(loc, ValueRange{zero}, acc);
    
    // Inner loop over K/V tiles
    auto kvLoop = builder.create<scf::ForOp>(loc, c0, seqLen, stepN);
    builder.setInsertionPointToStart(kvLoop.getBody());
    
    Value kv_start = kvLoop.getInductionVar();
    
    // Create online attention computation
    builder.create<OnlineAttentionOp>(
      loc, Q, K, V, acc, mState, lState, q_start, kv_start);
    
    // After KV loop, finalize and store
    builder.setInsertionPointAfter(kvLoop);
    builder.create<FinalizeAttentionOp>(loc, acc, mState, lState, O, q_start);
  }
};
```

### 2. Memory Placement Pass

This pass assigns tensors to appropriate memory levels and inserts data movement operations.

```cpp
class MemoryPlacementPass : public PassWrapper<MemoryPlacementPass, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Analyze tensor usage patterns
    TensorUsageAnalysis usageAnalysis(func);
    
    // Assign memory levels
    MemoryLevelAssignment assignment = computeMemoryAssignment(usageAnalysis);
    
    // Transform operations to use assigned memory levels
    transformMemoryAccess(func, assignment);
  }

private:
  struct TensorUsageInfo {
    Value tensor;
    SmallVector<Operation*> readers;
    SmallVector<Operation*> writers;
    int64_t size;
    int accessCount;
    bool isTemporary;
  };
  
  enum class MemoryLevel {
    Global = 0,    // HBM
    Shared = 3,    // SMEM  
    Register = 5   // Registers
  };
  
  class TensorUsageAnalysis {
  public:
    TensorUsageAnalysis(FuncOp func) {
      analyzeUsage(func);
    }
    
    const TensorUsageInfo& getUsageInfo(Value tensor) const {
      auto it = usageInfo_.find(tensor);
      assert(it != usageInfo_.end() && "Tensor not found in usage analysis");
      return it->second;
    }
    
    SmallVector<Value> getTensorsByUsage() const {
      SmallVector<std::pair<Value, int>> tensors;
      for (const auto& [tensor, info] : usageInfo_) {
        tensors.emplace_back(tensor, info.accessCount);
      }
      
      // Sort by access frequency (descending)
      llvm::sort(tensors, [](const auto& a, const auto& b) {
        return a.second > b.second;
      });
      
      SmallVector<Value> result;
      for (const auto& [tensor, _] : tensors) {
        result.push_back(tensor);
      }
      return result;
    }
    
  private:
    DenseMap<Value, TensorUsageInfo> usageInfo_;
    
    void analyzeUsage(FuncOp func) {
      func.walk([&](Operation* op) {
        for (Value operand : op->getOperands()) {
          if (operand.getType().isa<MemRefType>()) {
            auto& info = usageInfo_[operand];
            info.tensor = operand;
            info.readers.push_back(op);
            info.accessCount++;
            
            // Estimate tensor size
            if (auto memrefType = operand.getType().dyn_cast<MemRefType>()) {
              info.size = computeTensorSize(memrefType);
            }
          }
        }
        
        for (Value result : op->getResults()) {
          if (result.getType().isa<MemRefType>()) {
            auto& info = usageInfo_[result];
            info.tensor = result;
            info.writers.push_back(op);
            
            // Check if this is a temporary (single use)
            info.isTemporary = result.hasOneUse();
          }
        }
      });
    }
    
    int64_t computeTensorSize(MemRefType type) {
      int64_t size = 1;
      for (int64_t dim : type.getShape()) {
        if (dim > 0) {
          size *= dim;
        } else {
          size *= 128; // Estimate for dynamic dimensions
        }
      }
      
      // Account for element size
      if (type.getElementType().isF32()) {
        size *= 4;
      } else if (type.getElementType().isBF16() || type.getElementType().isF16()) {
        size *= 2;
      }
      
      return size;
    }
  };
  
  using MemoryLevelAssignment = DenseMap<Value, MemoryLevel>;
  
  MemoryLevelAssignment computeMemoryAssignment(const TensorUsageAnalysis& analysis) {
    MemoryLevelAssignment assignment;
    
    // Memory size constraints (typical GPU values)
    const int64_t MAX_SHARED_MEMORY = 48 * 1024;  // 48KB
    const int64_t MAX_REGISTER_MEMORY = 256 * 32; // 256 registers * 32 bytes
    
    int64_t usedSharedMemory = 0;
    int64_t usedRegisterMemory = 0;
    
    // Prioritize tensors by access frequency
    auto tensorsByUsage = analysis.getTensorsByUsage();
    
    for (Value tensor : tensorsByUsage) {
      const auto& info = analysis.getUsageInfo(tensor);
      MemoryLevel level = MemoryLevel::Global; // Default to global
      
      // Small, frequently accessed tensors -> registers
      if (info.size <= 4096 && info.accessCount >= 10 && 
          usedRegisterMemory + info.size <= MAX_REGISTER_MEMORY) {
        level = MemoryLevel::Register;
        usedRegisterMemory += info.size;
      }
      // Medium-sized, reused tensors -> shared memory  
      else if (info.size <= 16384 && info.accessCount >= 3 && 
               usedSharedMemory + info.size <= MAX_SHARED_MEMORY) {
        level = MemoryLevel::Shared;
        usedSharedMemory += info.size;
      }
      // Temporary small tensors -> registers if space available
      else if (info.isTemporary && info.size <= 1024 &&
               usedRegisterMemory + info.size <= MAX_REGISTER_MEMORY) {
        level = MemoryLevel::Register;
        usedRegisterMemory += info.size;
      }
      
      assignment[tensor] = level;
    }
    
    return assignment;
  }
  
  void transformMemoryAccess(FuncOp func, const MemoryLevelAssignment& assignment) {
    OpBuilder builder(&getContext());
    
    // Transform allocation sites
    func.walk([&](Operation* op) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        transformAllocation(allocOp, assignment, builder);
      } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
        // SubViews inherit memory space from source
        inheritMemorySpace(subviewOp, assignment);
      }
    });
    
    // Insert data movement operations
    insertDataMovementOps(func, assignment, builder);
  }
  
  void transformAllocation(memref::AllocOp allocOp, 
                          const MemoryLevelAssignment& assignment,
                          OpBuilder& builder) {
    Value result = allocOp.getResult();
    auto it = assignment.find(result);
    if (it == assignment.end()) return;
    
    MemoryLevel level = it->second;
    if (level == MemoryLevel::Global) return; // No change needed
    
    builder.setInsertionPoint(allocOp);
    
    // Get memory space attribute
    auto memorySpace = builder.getI64IntegerAttr(static_cast<int64_t>(level));
    
    // Create new type with memory space
    auto oldType = allocOp.getType().cast<MemRefType>();
    auto newType = MemRefType::get(
      oldType.getShape(), oldType.getElementType(), 
      oldType.getLayout(), memorySpace);
    
    Operation* newAlloc = nullptr;
    if (level == MemoryLevel::Shared) {
      newAlloc = builder.create<AllocSharedOp>(
        allocOp.getLoc(), newType, allocOp.getDynamicSizes());
    } else if (level == MemoryLevel::Register) {
      newAlloc = builder.create<AllocRegisterOp>(
        allocOp.getLoc(), newType);
    }
    
    if (newAlloc) {
      allocOp.getResult().replaceAllUsesWith(newAlloc->getResult(0));
      allocOp->erase();
    }
  }
  
  void insertDataMovementOps(FuncOp func, 
                            const MemoryLevelAssignment& assignment,
                            OpBuilder& builder) {
    func.walk([&](Operation* op) {
      // Look for operations that cross memory boundaries
      for (int i = 0; i < op->getNumOperands(); ++i) {
        Value operand = op->getOperand(i);
        if (!operand.getType().isa<MemRefType>()) continue;
        
        MemoryLevel srcLevel = getMemoryLevel(operand, assignment);
        MemoryLevel dstLevel = getRequiredMemoryLevel(op, i);
        
        if (srcLevel != dstLevel && dstLevel != MemoryLevel::Global) {
          // Insert copy operation
          builder.setInsertionPoint(op);
          
          auto srcType = operand.getType().cast<MemRefType>();
          auto dstType = MemRefType::get(
            srcType.getShape(), srcType.getElementType(),
            srcType.getLayout(), 
            builder.getI64IntegerAttr(static_cast<int64_t>(dstLevel)));
          
          Value dstBuffer;
          if (dstLevel == MemoryLevel::Shared) {
            dstBuffer = builder.create<AllocSharedOp>(op->getLoc(), dstType);
          } else if (dstLevel == MemoryLevel::Register) {
            dstBuffer = builder.create<AllocRegisterOp>(op->getLoc(), dstType);
          }
          
          // Create async copy if beneficial
          bool useAsyncCopy = (srcLevel == MemoryLevel::Global && 
                              dstLevel == MemoryLevel::Shared);
          
          if (useAsyncCopy) {
            builder.create<CopyAsyncOp>(
              op->getLoc(), operand, dstBuffer,
              builder.getStringAttr("global_to_shared"));
          } else {
            builder.create<CopyOp>(op->getLoc(), operand, dstBuffer);
          }
          
          // Replace operand with copied version
          op->setOperand(i, dstBuffer);
        }
      }
    });
  }
  
  MemoryLevel getMemoryLevel(Value tensor, const MemoryLevelAssignment& assignment) {
    auto it = assignment.find(tensor);
    if (it != assignment.end()) {
      return it->second;
    }
    
    // Infer from memory space attribute
    if (auto memrefType = tensor.getType().dyn_cast<MemRefType>()) {
      if (auto spaceAttr = memrefType.getMemorySpace()) {
        if (auto intAttr = spaceAttr.dyn_cast<IntegerAttr>()) {
          return static_cast<MemoryLevel>(intAttr.getInt());
        }
      }
    }
    
    return MemoryLevel::Global; // Default
  }
  
  MemoryLevel getRequiredMemoryLevel(Operation* op, int operandIndex) {
    // Operation-specific memory requirements
    if (isa<TiledMatmulOp>(op)) {
      return MemoryLevel::Shared; // Matrix tiles should be in shared memory
    } else if (isa<OnlineAttentionOp>(op)) {
      // Q, K, V tiles in shared memory; accumulators in registers
      return operandIndex < 3 ? MemoryLevel::Shared : MemoryLevel::Register;
    } else if (isa<SoftmaxOp, LayerNormOp>(op)) {
      return MemoryLevel::Shared; // Reductions benefit from shared memory
    }
    
    return MemoryLevel::Global; // Default requirement
  }
};
```

### 3. Pipeline Generation Pass

This pass creates async execution pipelines with overlapping computation and communication.

```cpp
class PipelineGenerationPass : public PassWrapper<PipelineGenerationPass, OperationPass<FuncOp>> {
public:
  PipelineGenerationPass(int numStages = 3) : numStages_(numStages) {}
  
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Find pipelineable loops
    SmallVector<scf::ForOp> pipelineableLoops;
    func.walk([&](scf::ForOp forOp) {
      if (canPipeline(forOp)) {
        pipelineableLoops.push_back(forOp);
      }
    });
    
    // Transform loops to use software pipelining
    for (auto loop : pipelineableLoops) {
      transformToPipeline(loop);
    }
  }

private:
  int numStages_;
  
  bool canPipeline(scf::ForOp loop) {
    // Check for data dependencies and pipeline opportunities
    bool hasAsyncOps = false;
    bool hasComputeOps = false;
    
    loop.walk([&](Operation* op) {
      if (isa<CopyAsyncOp>(op)) {
        hasAsyncOps = true;
      } else if (isa<TiledMatmulOp, OnlineAttentionOp>(op)) {
        hasComputeOps = true;
      }
    });
    
    return hasAsyncOps && hasComputeOps;
  }
  
  void transformToPipeline(scf::ForOp loop) {
    OpBuilder builder(loop);
    
    // Analyze loop body for pipelineable operations
    PipelineAnalysis analysis(loop, numStages_);
    
    // Create prologue, steady state, and epilogue
    createPipelinePrologue(loop, analysis, builder);
    createPipelineSteadyState(loop, analysis, builder);
    createPipelineEpilogue(loop, analysis, builder);
    
    // Remove original loop
    loop->erase();
  }
  
  struct PipelineStage {
    SmallVector<Operation*> asyncOps;    // Async copies
    SmallVector<Operation*> computeOps;  // Compute operations
    SmallVector<Operation*> syncOps;     // Barriers and waits
  };
  
  class PipelineAnalysis {
  public:
    PipelineAnalysis(scf::ForOp loop, int numStages) : numStages_(numStages) {
      analyzeLoopBody(loop);
    }
    
    const SmallVector<PipelineStage>& getStages() const { return stages_; }
    int getNumStages() const { return numStages_; }
    
    // Get values that need to be rotated between stages
    SmallVector<Value> getRotatingValues() const { return rotatingValues_; }
    
  private:
    int numStages_;
    SmallVector<PipelineStage> stages_;
    SmallVector<Value> rotatingValues_;
    
    void analyzeLoopBody(scf::ForOp loop) {
      // Stage 0: Async copies (prefetch)
      PipelineStage stage0;
      
      // Stage 1: Wait and compute  
      PipelineStage stage1;
      
      // Stage 2: Finalize and store
      PipelineStage stage2;
      
      loop.walk([&](Operation* op) {
        if (auto copyOp = dyn_cast<CopyAsyncOp>(op)) {
          stage0.asyncOps.push_back(op);
          
          // Track buffers that need rotation
          if (copyOp->hasAttr("double_buffer")) {
            rotatingValues_.push_back(copyOp->getOperand(1));
          }
        } else if (auto waitOp = dyn_cast<AsyncWaitOp>(op)) {
          stage1.syncOps.push_back(op);
        } else if (auto barrierOp = dyn_cast<BarrierOp>(op)) {
          stage1.syncOps.push_back(op);
        } else if (isa<TiledMatmulOp, OnlineAttentionOp>(op)) {
          stage1.computeOps.push_back(op);
        } else if (isa<StoreOp, CopyOp>(op)) {
          stage2.computeOps.push_back(op);
        }
      });
      
      stages_.push_back(stage0);
      stages_.push_back(stage1);  
      stages_.push_back(stage2);
    }
  };
  
  void createPipelinePrologue(scf::ForOp originalLoop, 
                             const PipelineAnalysis& analysis,
                             OpBuilder& builder) {
    Location loc = originalLoop.getLoc();
    
    // Create buffer rotation infrastructure
    auto rotatingBuffers = createRotatingBuffers(analysis, builder, loc);
    
    // Prologue: Fill the pipeline
    for (int stage = 0; stage < analysis.getNumStages() - 1; ++stage) {
      // Compute iteration for this prologue stage
      Value iterationVar = computePrologueIteration(
        originalLoop, stage, builder, loc);
      
      // Execute operations for this stage
      if (stage < analysis.getStages().size()) {
        executeStageOperations(analysis.getStages()[stage], iterationVar, 
                              rotatingBuffers, builder, loc);
      }
    }
  }
  
  void createPipelineSteadyState(scf::ForOp originalLoop,
                                const PipelineAnalysis& analysis, 
                                OpBuilder& builder) {
    Location loc = originalLoop.getLoc();
    
    // Compute steady state bounds
    Value lowerBound = originalLoop.getLowerBound();
    Value upperBound = originalLoop.getUpperBound();
    Value step = originalLoop.getStep();
    
    Value pipelineDepth = builder.create<arith::ConstantIndexOp>(
      loc, analysis.getNumStages() - 1);
    Value adjustedLower = builder.create<arith::AddIOp>(
      loc, lowerBound, 
      builder.create<arith::MulIOp>(loc, step, pipelineDepth));
    
    // Create steady state loop
    auto steadyLoop = builder.create<scf::ForOp>(
      loc, adjustedLower, upperBound, step);
    
    builder.setInsertionPointToStart(steadyLoop.getBody());
    Value steadyIter = steadyLoop.getInductionVar();
    
    // Execute all pipeline stages in steady state
    auto rotatingBuffers = getRotatingBuffers(analysis);
    
    for (int stageIdx = 0; stageIdx < analysis.getStages().size(); ++stageIdx) {
      const auto& stage = analysis.getStages()[stageIdx];
      
      // Compute which iteration this stage is processing
      Value stageOffset = builder.create<arith::ConstantIndexOp>(loc, stageIdx);
      Value stageStep = builder.create<arith::MulIOp>(loc, step, stageOffset);
      Value stageIter = builder.create<arith::SubIOp>(loc, steadyIter, stageStep);
      
      executeStageOperations(stage, stageIter, rotatingBuffers, builder, loc);
    }
    
    // Rotate buffers at end of iteration
    rotateBuffers(rotatingBuffers, builder, loc);
  }
  
  void createPipelineEpilogue(scf::ForOp originalLoop,
                             const PipelineAnalysis& analysis,
                             OpBuilder& builder) {
    Location loc = originalLoop.getLoc();
    
    // Epilogue: Drain the pipeline
    for (int stage = 1; stage < analysis.getNumStages(); ++stage) {
      // Execute remaining compute stages
      if (stage < analysis.getStages().size()) {
        executeStageOperations(analysis.getStages()[stage], 
                              /*iterationVar=*/Value{},
                              getRotatingBuffers(analysis), 
                              builder, loc);
      }
    }
  }
  
  DenseMap<Value, SmallVector<Value>> createRotatingBuffers(
      const PipelineAnalysis& analysis, OpBuilder& builder, Location loc) {
    DenseMap<Value, SmallVector<Value>> rotatingBuffers;
    
    for (Value originalBuffer : analysis.getRotatingValues()) {
      auto bufferType = originalBuffer.getType().cast<MemRefType>();
      
      SmallVector<Value> rotatedBuffers;
      for (int i = 0; i < analysis.getNumStages(); ++i) {
        auto rotatedBuffer = builder.create<AllocSharedOp>(loc, bufferType);
        rotatedBuffers.push_back(rotatedBuffer);
      }
      
      rotatingBuffers[originalBuffer] = std::move(rotatedBuffers);
    }
    
    return rotatingBuffers;
  }
  
  void executeStageOperations(const PipelineStage& stage, Value iterationVar,
                             const DenseMap<Value, SmallVector<Value>>& rotatingBuffers,
                             OpBuilder& builder, Location loc) {
    // Execute async operations
    for (Operation* op : stage.asyncOps) {
      cloneOperationWithRotatedBuffers(op, rotatingBuffers, builder);
    }
    
    // Execute sync operations
    for (Operation* op : stage.syncOps) {
      builder.clone(*op);
    }
    
    // Execute compute operations
    for (Operation* op : stage.computeOps) {
      cloneOperationWithRotatedBuffers(op, rotatingBuffers, builder);
    }
  }
  
  void cloneOperationWithRotatedBuffers(
      Operation* op, 
      const DenseMap<Value, SmallVector<Value>>& rotatingBuffers,
      OpBuilder& builder) {
    
    IRMapping mapping;
    
    // Map operands to rotated versions
    for (Value operand : op->getOperands()) {
      auto it = rotatingBuffers.find(operand);
      if (it != rotatingBuffers.end()) {
        // Use current rotation of this buffer
        Value rotatedBuffer = getCurrentRotation(it->second);
        mapping.map(operand, rotatedBuffer);
      }
    }
    
    builder.clone(*op, mapping);
  }
  
  Value getCurrentRotation(const SmallVector<Value>& rotations) {
    // Simple rotation scheme - could be more sophisticated
    static int currentIndex = 0;
    Value result = rotations[currentIndex % rotations.size()];
    return result;
  }
  
  void rotateBuffers(const DenseMap<Value, SmallVector<Value>>& rotatingBuffers,
                    OpBuilder& builder, Location loc) {
    // Rotate buffer indices - this is conceptual, actual implementation
    // would track rotation state in the pass
  }
};
```

### 4. Loop Fusion Pass

This pass fuses compatible loops to reduce memory traffic and improve data locality.

```cpp
class LoopFusionPass : public PassWrapper<LoopFusionPass, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Find fusion candidates
    SmallVector<LoopFusionCandidate> candidates = findFusionCandidates(func);
    
    // Apply profitable fusions
    for (auto& candidate : candidates) {
      if (isProfitableFusion(candidate)) {
        performFusion(candidate);
      }
    }
  }

private:
  struct LoopFusionCandidate {
    scf::ForOp producerLoop;
    scf::ForOp consumerLoop;
    Value sharedTensor;
    FusionType type;
  };
  
  enum class FusionType {
    ProducerConsumer,  // Standard producer-consumer fusion
    Sibling,          // Sibling loops with compatible iteration spaces
    Distribution      // Loop distribution for better parallelism
  };
  
  SmallVector<LoopFusionCandidate> findFusionCandidates(FuncOp func) {
    SmallVector<LoopFusionCandidate> candidates;
    
    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp loop) {
      loops.push_back(loop);
    });
    
    // Look for producer-consumer relationships
    for (size_t i = 0; i < loops.size(); ++i) {
      for (size_t j = i + 1; j < loops.size(); ++j) {
        auto producer = loops[i];
        auto consumer = loops[j];
        
        // Check if producer output is consumed by consumer
        for (Value result : producer->getResults()) {
          if (isConsumedBy(result, consumer)) {
            candidates.push_back({producer, consumer, result, 
                                 FusionType::ProducerConsumer});
          }
        }
        
        // Check for sibling fusion opportunities
        if (haveSiblingRelation(producer, consumer)) {
          candidates.push_back({producer, consumer, Value{}, 
                               FusionType::Sibling});
        }
      }
    }
    
    return candidates;
  }
  
  bool isProfitableFusion(const LoopFusionCandidate& candidate) {
    // Cost model for fusion profitability
    switch (candidate.type) {
      case FusionType::ProducerConsumer:
        return isProfitableProducerConsumerFusion(candidate);
      case FusionType::Sibling:
        return isProfitableSiblingFusion(candidate);
      default:
        return false;
    }
  }
  
  bool isProfitableProducerConsumerFusion(const LoopFusionCandidate& candidate) {
    // Estimate memory traffic reduction
    int64_t tensorSize = estimateTensorSize(candidate.sharedTensor);
    
    // Fusion is profitable if shared tensor is large enough to matter
    // but small enough to fit in cache/shared memory
    const int64_t MIN_FUSION_SIZE = 1024;   // 1KB
    const int64_t MAX_FUSION_SIZE = 32768;  // 32KB
    
    return tensorSize >= MIN_FUSION_SIZE && tensorSize <= MAX_FUSION_SIZE;
  }
  
  void performFusion(const LoopFusionCandidate& candidate) {
    OpBuilder builder(candidate.consumerLoop);
    
    switch (candidate.type) {
      case FusionType::ProducerConsumer:
        performProducerConsumerFusion(candidate, builder);
        break;
      case FusionType::Sibling:
        performSiblingFusion(candidate, builder);
        break;
    }
  }
  
  void performProducerConsumerFusion(const LoopFusionCandidate& candidate,
                                    OpBuilder& builder) {
    auto producerLoop = candidate.producerLoop;
    auto consumerLoop = candidate.consumerLoop;
    
    // Create fused loop with combined iteration space
    Value fusedLower = producerLoop.getLowerBound();
    Value fusedUpper = producerLoop.getUpperBound();
    Value fusedStep = producerLoop.getStep();
    
    auto fusedLoop = builder.create<scf::ForOp>(
      consumerLoop.getLoc(), fusedLower, fusedUpper, fusedStep);
    
    builder.setInsertionPointToStart(fusedLoop.getBody());
    
    // Create shared temporary in faster memory
    Value sharedTensor = candidate.sharedTensor;
    auto tensorType = sharedTensor.getType().cast<MemRefType>();
    
    // Allocate in shared memory for better locality
    auto sharedType = MemRefType::get(
      tensorType.getShape(), tensorType.getElementType(),
      tensorType.getLayout(), builder.getI64IntegerAttr(3)); // Shared memory
    
    auto localBuffer = builder.create<AllocSharedOp>(
      fusedLoop.getLoc(), sharedType);
    
    // Clone producer body
    IRMapping producerMapping;
    producerMapping.map(producerLoop.getInductionVar(), fusedLoop.getInductionVar());
    producerMapping.map(sharedTensor, localBuffer);
    
    for (Operation& op : producerLoop.getBody()->without_terminator()) {
      builder.clone(op, producerMapping);
    }
    
    // Insert barrier to ensure producer writes are visible
    builder.create<BarrierOp>(fusedLoop.getLoc());
    
    // Clone consumer body  
    IRMapping consumerMapping;
    consumerMapping.map(consumerLoop.getInductionVar(), fusedLoop.getInductionVar());
    consumerMapping.map(sharedTensor, localBuffer);
    
    for (Operation& op : consumerLoop.getBody()->without_terminator()) {
      builder.clone(op, consumerMapping);
    }
    
    // Update uses of loop results
    for (auto [oldResult, newResult] : llvm::zip(
          consumerLoop.getResults(), fusedLoop.getResults())) {
      oldResult.replaceAllUsesWith(newResult);
    }
    
    // Remove original loops
    consumerLoop->erase();
    producerLoop->erase();
  }
};
```

### 5. Autotuning Integration Pass

This pass integrates with Tessera's autotuning system to explore optimization spaces.

```cpp
class AutotuningIntegrationPass : public PassWrapper<AutotuningIntegrationPass, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Find operations with autotuning annotations
    SmallVector<Operation*> autotunableOps;
    func.walk([&](Operation* op) {
      if (op->hasAttr("tessera.autotune")) {
        autotunableOps.push_back(op);
      }
    });
    
    // Generate tuning variants for each operation
    for (Operation* op : autotunableOps) {
      generateTuningVariants(op);
    }
  }

private:
  void generateTuningVariants(Operation* op) {
    auto tuningAttr = op->getAttrOfType<DictionaryAttr>("tessera.autotune");
    if (!tuningAttr) return;
    
    OpBuilder builder(op);
    
    if (auto matmulOp = dyn_cast<TiledMatmulOp>(op)) {
      generateMatmulVariants(matmulOp, tuningAttr, builder);
    } else if (auto attentionOp = dyn_cast<FlashAttentionOp>(op)) {
      generateAttentionVariants(attentionOp, tuningAttr, builder);
    }
  }
  
  void generateMatmulVariants(TiledMatmulOp op, DictionaryAttr tuningAttr, 
                             OpBuilder& builder) {
    // Extract tuning parameters
    auto tileSizesAttr = tuningAttr.get("tile_sizes").dyn_cast<ArrayAttr>();
    auto numWarpsAttr = tuningAttr.get("num_warps").dyn_cast<ArrayAttr>();
    auto stagesAttr = tuningAttr.get("stages").dyn_cast<ArrayAttr>();
    
    if (!tileSizesAttr || !numWarpsAttr || !stagesAttr) return;
    
    SmallVector<TuningConfiguration> configs;
    
    // Generate all combinations
    for (auto tileSizeAttr : tileSizesAttr) {
      for (auto numWarpsValue : numWarpsAttr) {
        for (auto stagesValue : stagesAttr) {
          TuningConfiguration config;
          config.tileSizes = parseTileSizes(tileSizeAttr);
          config.numWarps = numWarpsValue.cast<IntegerAttr>().getInt();
          config.numStages = stagesValue.cast<IntegerAttr>().getInt();
          configs.push_back(config);
        }
      }
    }
    
    // Create variant selector
    createVariantSelector(op, configs, builder);
  }
  
  struct TuningConfiguration {
    SmallVector<int64_t> tileSizes;
    int64_t numWarps;
    int64_t numStages;
    
    std::string getKey() const {
      std::string key = "tiles_";
      for (size_t i = 0; i < tileSizes.size(); ++i) {
        if (i > 0) key += "x";
        key += std::to_string(tileSizes[i]);
      }
      key += "_warps_" + std::to_string(numWarps);
      key += "_stages_" + std::to_string(numStages);
      return key;
    }
  };
  
  void createVariantSelector(Operation* op, 
                            const SmallVector<TuningConfiguration>& configs,
                            OpBuilder& builder) {
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);
    
    // Create runtime parameter for variant selection
    auto indexType = builder.getIndexType();
    auto variantParam = builder.create<GetTuningParameterOp>(
      loc, indexType, builder.getStringAttr("variant_id"));
    
    // Create switch statement for variant selection
    SmallVector<std::pair<int64_t, Block*>> cases;
    
    auto switchOp = builder.create<SwitchOp>(
      loc, op->getResultTypes(), variantParam, /*defaultBlock=*/nullptr);
    
    // Generate a block for each configuration
    for (size_t i = 0; i < configs.size(); ++i) {
      const auto& config = configs[i];
      
      auto* caseBlock = switchOp.addCase(i);
      builder.setInsertionPointToStart(caseBlock);
      
      // Clone operation with specific configuration
      Operation* variantOp = builder.clone(*op);
      
      // Set configuration attributes
      variantOp->setAttr("tile_sizes", 
        builder.getI64ArrayAttr(config.tileSizes));
      variantOp->setAttr("num_warps", 
        builder.getI64IntegerAttr(config.numWarps));
      variantOp->setAttr("num_stages", 
        builder.getI64IntegerAttr(config.numStages));
      
      builder.create<YieldOp>(loc, variantOp->getResults());
    }
    
    // Replace original operation
    op->replaceAllUsesWith(switchOp.getResults());
    op->erase();
  }
};

// Runtime support for autotuning
class GetTuningParameterOp : public Op<GetTuningParameterOp> {
  // Implementation of runtime parameter lookup
  // This connects to the autotuning system to get the best variant
  // for the current input shapes and hardware configuration
};
```

## Testing Schedule IR Passes

```cpp
// Unit tests for Schedule IR passes
TEST(ScheduleIRPasses, TilingTransformation) {
  MLIRContext context;
  context.loadDialect<TesseraDialect>();
  
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func @matmul_test(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>) {
      tessera.matmul %A, %B, %C : memref<1024x512xf32>, memref<512x1024xf32>, memref<1024x1024xf32>
      return
    }
  )mlir", &context);
  
  // Apply tiling pass
  PassManager pm(&context);
  pm.addPass(createTilingPass({128, 128, 64}));
  
  ASSERT_TRUE(succeeded(pm.run(module.get())));
  
  // Verify tiled structure was created
  auto func = module->lookupSymbol<FuncOp>("matmul_test");
  bool foundTiledLoop = false;
  func.walk([&](scf::ForOp loop) {
    foundTiledLoop = true;
  });
  
  EXPECT_TRUE(foundTiledLoop);
}

TEST(ScheduleIRPasses, MemoryPlacement) {
  MLIRContext context;
  context.loadDialect<TesseraDialect>();
  
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func @memory_test() {
      %0 = memref.alloc() : memref<128x128xf32>
      %1 = memref.alloc() : memref<64x64xf32>
      // Use tensors multiple times
      tessera.use %0 : memref<128x128xf32>
      tessera.use %0 : memref<128x128xf32>
      tessera.use %1 : memref<64x64xf32>
      return
    }
  )mlir", &context);
  
  PassManager pm(&context);
  pm.addPass(createMemoryPlacementPass());
  
  ASSERT_TRUE(succeeded(pm.run(module.get())));
  
  // Verify memory spaces were assigned
  auto func = module->lookupSymbol<FuncOp>("memory_test");
  bool foundSharedAlloc = false;
  func.walk([&](AllocSharedOp alloc) {
    foundSharedAlloc = true;
  });
  
  EXPECT_TRUE(foundSharedAlloc);
}
```

## Summary

Schedule IR passes transform high-level operations into explicit, optimized loop nests:

- **Tiling Pass** creates blocked loop structures for cache efficiency
- **Memory Placement Pass** assigns tensors to appropriate memory levels
- **Pipeline Generation Pass** creates async execution pipelines
- **Loop Fusion Pass** combines compatible loops for better locality
- **Autotuning Integration** generates variants for performance optimization

These passes work together to create high-performance schedule representations that can be lowered to efficient tile-level code. The next document will cover **Tile IR Passes**, which handle the final hardware-specific optimizations.
    
      