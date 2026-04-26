# Tessera Graph IR Passes

Graph IR is the highest level of abstraction in Tessera's compiler stack. It represents computations as a graph of high-level operations with explicit handling of autodiff, effects, and distributed semantics. This document details the major passes that operate on Graph IR.

## Graph IR Structure

Graph IR uses MLIR dialects to represent:
- **Tensor Operations**: `tessera.add`, `tessera.matmul`, `tessera.softmax`
- **Control Flow**: Standard MLIR `scf` dialect operations
- **Effects**: `tessera.random`, `tessera.state_update`, `tessera.collective`
- **Autodiff**: `tessera.forward`, `tessera.reverse`, `tessera.vjp`
- **Distribution**: `tessera.shard`, `tessera.replicate`, `tessera.allreduce`

### Example Graph IR
```mlir
module @transformer_layer {
  func @forward(%x: tensor<32x512x1024xbf16>, 
                %w_q: tensor<1024x1024xbf16>,
                %w_k: tensor<1024x1024xbf16>, 
                %w_v: tensor<1024x1024xbf16>) -> tensor<32x512x1024xbf16> {
    // Layer normalization
    %ln_out = tessera.layer_norm %x {eps = 1.0e-5} : tensor<32x512x1024xbf16> -> tensor<32x512x1024xbf16>
    
    // QKV projections
    %q = tessera.matmul %ln_out, %w_q : tensor<32x512x1024xbf16>, tensor<1024x1024xbf16> -> tensor<32x512x1024xbf16>
    %k = tessera.matmul %ln_out, %w_k : tensor<32x512x1024xbf16>, tensor<1024x1024xbf16> -> tensor<32x512x1024xbf16>
    %v = tessera.matmul %ln_out, %w_v : tensor<32x512x1024xbf16>, tensor<1024x1024xbf16> -> tensor<32x512x1024xbf16>
    
    // Reshape for multi-head attention
    %q_heads = tessera.reshape %q : tensor<32x512x1024xbf16> -> tensor<32x16x512x64xbf16>
    %k_heads = tessera.reshape %k : tensor<32x512x1024xbf16> -> tensor<32x16x512x64xbf16>
    %v_heads = tessera.reshape %v : tensor<32x512x1024xbf16> -> tensor<32x16x512x64xbf16>
    
    // Attention computation
    %attn_out = tessera.flash_attention %q_heads, %k_heads, %v_heads {causal = true} :
      tensor<32x16x512x64xbf16>, tensor<32x16x512x64xbf16>, tensor<32x16x512x64xbf16> -> tensor<32x16x512x64xbf16>
    
    // Reshape back and add residual
    %attn_flat = tessera.reshape %attn_out : tensor<32x16x512x64xbf16> -> tensor<32x512x1024xbf16>
    %result = tessera.add %x, %attn_flat : tensor<32x512x1024xbf16>, tensor<32x512x1024xbf16> -> tensor<32x512x1024xbf16>
    
    return %result : tensor<32x512x1024xbf16>
  }
}
```

## Major Graph IR Passes

### 1. Type Inference Pass

The type inference pass resolves symbolic dimensions and infers precise tensor types throughout the computation graph.

```cpp
class TypeInferencePass : public mlir::PassWrapper<TypeInferencePass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Walk all functions and infer types
    module.walk([&](FuncOp func) {
      inferFunctionTypes(func);
    });
  }

private:
  void inferFunctionTypes(FuncOp func) {
    // Create type inference context
    TypeInferenceContext ctx;
    
    // Initialize with function arguments
    for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
      if (auto tensorType = arg.getType().dyn_cast<TensorType>()) {
        ctx.setType(arg, tensorType);
      }
    }
    
    // Forward propagation through operations
    func.walk([&](Operation* op) {
      if (auto inferrable = dyn_cast<TypeInferrableOpInterface>(op)) {
        inferrable.inferTypes(ctx);
      }
    });
  }
};

// Registration
std::unique_ptr<Pass> createTypeInferencePass() {
  return std::make_unique<TypeInferencePass>();
}
```

#### Example: Matmul Type Inference
```cpp
LogicalResult MatmulOp::inferReturnTypes(
    MLIRContext* context, 
    Optional<Location> location,
    ValueRange operands, 
    DictionaryAttr attributes,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  
  auto lhsType = operands[0].getType().cast<TensorType>();
  auto rhsType = operands[1].getType().cast<TensorType>();
  
  // Get shapes
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  
  // Validate dimensions for matrix multiply
  if (lhsShape.size() < 2 || rhsShape.size() < 2) {
    return failure();
  }
  
  // Infer result shape: [..., M, K] @ [..., K, N] -> [..., M, N]
  SmallVector<int64_t> resultShape;
  
  // Batch dimensions (broadcast semantics)
  auto lhsBatch = lhsShape.drop_back(2);
  auto rhsBatch = rhsShape.drop_back(2);
  
  if (lhsBatch.size() != rhsBatch.size()) {
    return failure();
  }
  
  for (auto [lhs, rhs] : llvm::zip(lhsBatch, rhsBatch)) {
    if (lhs != rhs && lhs != 1 && rhs != 1) {
      return failure();
    }
    resultShape.push_back(std::max(lhs, rhs));
  }
  
  // Matrix dimensions
  int64_t M = lhsShape[lhsShape.size()-2];
  int64_t K_lhs = lhsShape[lhsShape.size()-1];
  int64_t K_rhs = rhsShape[rhsShape.size()-2];
  int64_t N = rhsShape[rhsShape.size()-1];
  
  if (K_lhs != K_rhs) {
    return failure();
  }
  
  resultShape.push_back(M);
  resultShape.push_back(N);
  
  // Infer element type (promote precision if needed)
  Type elementType = promoteTypes(lhsType.getElementType(), rhsType.getElementType());
  
  inferredReturnTypes.push_back(RankedTensorType::get(resultShape, elementType));
  return success();
}
```

### 2. Autodiff Pass

The autodiff pass transforms functions to compute gradients using either forward or reverse mode automatic differentiation.

```cpp
class AutodiffPass : public PassWrapper<AutodiffPass, OperationPass<ModuleOp>> {
public:
  AutodiffPass(AutodiffMode mode) : mode_(mode) {}
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Find functions marked for autodiff
    SmallVector<FuncOp> autodiffFuncs;
    module.walk([&](FuncOp func) {
      if (func->hasAttr("tessera.autodiff")) {
        autodiffFuncs.push_back(func);
      }
    });
    
    // Generate gradient functions
    for (auto func : autodiffFuncs) {
      if (mode_ == AutodiffMode::Reverse) {
        generateReverseMode(func);
      } else {
        generateForwardMode(func);
      }
    }
  }

private:
  AutodiffMode mode_;
  
  void generateReverseMode(FuncOp originalFunc) {
    OpBuilder builder(originalFunc.getContext());
    
    // Create gradient function signature
    std::string gradFuncName = originalFunc.getName().str() + "_grad";
    
    // Gradient function takes original inputs + output gradients
    SmallVector<Type> gradInputTypes;
    gradInputTypes.append(originalFunc.getArgumentTypes().begin(), 
                         originalFunc.getArgumentTypes().end());
    gradInputTypes.append(originalFunc.getResultTypes().begin(),
                         originalFunc.getResultTypes().end());
    
    // Returns gradients for all inputs
    auto gradFuncType = builder.getFunctionType(gradInputTypes, originalFunc.getArgumentTypes());
    
    // Create gradient function
    FuncOp gradFunc = builder.create<FuncOp>(
      originalFunc.getLoc(), gradFuncName, gradFuncType);
    
    // Generate backward pass
    Block* gradBlock = gradFunc.addEntryBlock();
    builder.setInsertionPointToStart(gradBlock);
    
    // Run reverse-mode AD algorithm
    ReverseADContext adCtx(builder, originalFunc, gradFunc);
    adCtx.generateBackwardPass();
    
    // Insert the gradient function into the module
    originalFunc->getParentOp()->getRegion(0).push_back(gradFunc);
  }
};

// Reverse-mode AD implementation
class ReverseADContext {
public:
  ReverseADContext(OpBuilder& builder, FuncOp forward, FuncOp backward)
    : builder_(builder), forwardFunc_(forward), backwardFunc_(backward) {}
  
  void generateBackwardPass() {
    // Step 1: Replay forward pass and build tape
    replayForwardPass();
    
    // Step 2: Initialize output gradients
    initializeOutputGradients();
    
    // Step 3: Propagate gradients backwards
    propagateGradientsBackward();
    
    // Step 4: Return input gradients
    returnInputGradients();
  }

private:
  OpBuilder& builder_;
  FuncOp forwardFunc_;
  FuncOp backwardFunc_;
  
  // Maps forward values to their gradients
  DenseMap<Value, Value> gradients_;
  
  // Tape of operations in reverse order
  SmallVector<Operation*> tape_;
  
  void replayForwardPass() {
    // Clone forward operations to build tape
    IRMapping mapping;
    
    // Map forward arguments to backward arguments
    for (auto [fwdArg, bwdArg] : llvm::zip(
        forwardFunc_.getArguments(),
        backwardFunc_.getArguments().take_front(forwardFunc_.getNumArguments()))) {
      mapping.map(fwdArg, bwdArg);
    }
    
    // Clone each operation
    forwardFunc_.walk([&](Operation* op) {
      if (isa<ReturnOp>(op)) return;
      
      Operation* cloned = builder_.clone(*op, mapping);
      tape_.push_back(cloned);
    });
  }
  
  void initializeOutputGradients() {
    // Output gradients are the last arguments to backward function
    auto outputGrads = backwardFunc_.getArguments().take_back(
      forwardFunc_.getNumResults());
    
    auto forwardResults = forwardFunc_.front().getTerminator()->getOperands();
    
    for (auto [result, grad] : llvm::zip(forwardResults, outputGrads)) {
      // Map cloned result to its gradient
      Value clonedResult = /* find cloned version */;
      gradients_[clonedResult] = grad;
    }
  }
  
  void propagateGradientsBackward() {
    // Process tape in reverse order
    for (auto it = tape_.rbegin(); it != tape_.rend(); ++it) {
      Operation* op = *it;
      
      // Get VJP rule for this operation
      if (auto vjpOp = dyn_cast<VJPOpInterface>(op)) {
        vjpOp.applyVJP(*this);
      } else {
        // Default numerical differentiation
        applyNumericalVJP(op);
      }
    }
  }
  
  void returnInputGradients() {
    SmallVector<Value> inputGrads;
    for (auto arg : backwardFunc_.getArguments().take_front(forwardFunc_.getNumArguments())) {
      inputGrads.push_back(gradients_[arg]);
    }
    
    builder_.create<ReturnOp>(backwardFunc_.getLoc(), inputGrads);
  }
};
```

#### Example: Matmul VJP Implementation
```cpp
void MatmulOp::applyVJP(ReverseADContext& ctx) {
  // For C = A @ B, given grad_C:
  // grad_A = grad_C @ B^T
  // grad_B = A^T @ grad_C
  
  Value A = getOperand(0);
  Value B = getOperand(1);
  Value C = getResult();
  Value grad_C = ctx.getGradient(C);
  
  OpBuilder& builder = ctx.getBuilder();
  
  // grad_A = grad_C @ B^T
  Value B_T = builder.create<TransposeOp>(getLoc(), B, ArrayRef<int64_t>{-1, -2});
  Value grad_A = builder.create<MatmulOp>(getLoc(), grad_C, B_T);
  
  // grad_B = A^T @ grad_C  
  Value A_T = builder.create<TransposeOp>(getLoc(), A, ArrayRef<int64_t>{-1, -2});
  Value grad_B = builder.create<MatmulOp>(getLoc(), A_T, grad_C);
  
  ctx.setGradient(A, grad_A);
  ctx.setGradient(B, grad_B);
}
```

### 3. Algebraic Simplification Pass

This pass applies mathematical identities and simplifications to reduce computation.

```cpp
class AlgebraicSimplificationPass : public PassWrapper<AlgebraicSimplificationPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Apply simplification patterns
    RewritePatternSet patterns(&getContext());
    populateAlgebraicSimplificationPatterns(patterns);
    
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateAlgebraicSimplificationPatterns(RewritePatternSet& patterns) {
  patterns.add<
    AddZeroElimination,
    MulOneElimination,
    MulZeroElimination,
    TransposeTransposeCancellation,
    MatmulTransposeSimplification,
    ReshapeReshapeComposition,
    BroadcastElimination
  >(patterns.getContext());
}

// Example simplification patterns

// Eliminate addition with zero: x + 0 = x
struct AddZeroElimination : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(AddOp op, PatternRewriter& rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    // Check if RHS is zero constant
    if (auto constantOp = rhs.getDefiningOp<ConstantOp>()) {
      if (auto denseAttr = constantOp.getValue().dyn_cast<DenseElementsAttr>()) {
        if (denseAttr.isSplat() && 
            denseAttr.getSplatValue<APFloat>().isZero()) {
          rewriter.replaceOp(op, lhs);
          return success();
        }
      }
    }
    
    // Check if LHS is zero constant
    if (auto constantOp = lhs.getDefiningOp<ConstantOp>()) {
      if (auto denseAttr = constantOp.getValue().dyn_cast<DenseElementsAttr>()) {
        if (denseAttr.isSplat() && 
            denseAttr.getSplatValue<APFloat>().isZero()) {
          rewriter.replaceOp(op, rhs);
          return success();
        }
      }
    }
    
    return failure();
  }
};

// Eliminate multiplication by one: x * 1 = x
struct MulOneElimination : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(MulOp op, PatternRewriter& rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    // Check if RHS is one constant
    if (auto constantOp = rhs.getDefiningOp<ConstantOp>()) {
      if (auto denseAttr = constantOp.getValue().dyn_cast<DenseElementsAttr>()) {
        if (denseAttr.isSplat()) {
          APFloat value = denseAttr.getSplatValue<APFloat>();
          if (value.isExactlyValue(1.0)) {
            rewriter.replaceOp(op, lhs);
            return success();
          }
        }
      }
    }
    
    // Check if LHS is one constant
    if (auto constantOp = lhs.getDefiningOp<ConstantOp>()) {
      if (auto denseAttr = constantOp.getValue().dyn_cast<DenseElementsAttr>()) {
        if (denseAttr.isSplat()) {
          APFloat value = denseAttr.getSplatValue<APFloat>();
          if (value.isExactlyValue(1.0)) {
            rewriter.replaceOp(op, rhs);
            return success();
          }
        }
      }
    }
    
    return failure();
  }
};

// Cancel double transpose: transpose(transpose(x)) = x
struct TransposeTransposeCancellation : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(TransposeOp op, PatternRewriter& rewriter) const override {
    // Check if input is also a transpose
    if (auto innerTranspose = op.getInput().getDefiningOp<TransposeOp>()) {
      auto outerPermutation = op.getPermutation();
      auto innerPermutation = innerTranspose.getPermutation();
      
      // Check if permutations cancel out
      bool isIdentity = true;
      for (size_t i = 0; i < outerPermutation.size(); ++i) {
        int64_t composed = innerPermutation[outerPermutation[i]];
        if (composed != static_cast<int64_t>(i)) {
          isIdentity = false;
          break;
        }
      }
      
      if (isIdentity) {
        rewriter.replaceOp(op, innerTranspose.getInput());
        return success();
      }
    }
    
    return failure();
  }
};

// Optimize matmul with transpose: A @ B^T -> A @ B with different layout
struct MatmulTransposeSimplification : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter& rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    // Check for A @ B^T pattern
    if (auto rhsTranspose = rhs.getDefiningOp<TransposeOp>()) {
      auto permutation = rhsTranspose.getPermutation();
      
      // Check if this is a matrix transpose (swap last two dims)
      bool isMatrixTranspose = true;
      auto tensorType = rhsTranspose.getInput().getType().cast<TensorType>();
      int64_t rank = tensorType.getRank();
      
      for (int64_t i = 0; i < rank - 2; ++i) {
        if (permutation[i] != i) {
          isMatrixTranspose = false;
          break;
        }
      }
      
      if (isMatrixTranspose && 
          permutation[rank-2] == rank-1 && 
          permutation[rank-1] == rank-2) {
        
        // Replace with optimized matmul variant
        auto newOp = rewriter.create<MatmulOp>(
          op.getLoc(), op.getType(), lhs, rhsTranspose.getInput());
        newOp->setAttr("rhs_transpose", rewriter.getBoolAttr(true));
        
        rewriter.replaceOp(op, newOp.getResult());
        return success();
      }
    }
    
    return failure();
  }
};
```

### 4. Fusion Analysis Pass

This pass identifies opportunities to fuse operations for better performance and memory efficiency.

```cpp
class FusionAnalysisPass : public PassWrapper<FusionAnalysisPass, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Build dependency graph
    OperationDependencyGraph depGraph(func);
    
    // Find fusion opportunities
    SmallVector<FusionCluster> clusters = findFusionClusters(depGraph);
    
    // Apply fusions
    for (auto& cluster : clusters) {
      if (cluster.size() > 1) {
        fuseClusters(cluster);
      }
    }
  }

private:
  struct FusionCluster {
    SmallVector<Operation*> ops;
    FusionType type;
    
    size_t size() const { return ops.size(); }
  };
  
  enum class FusionType {
    Elementwise,    // Element-wise operations
    Reduction,      // Reductions with compatible shapes
    MatmulBias,     // Matmul + bias + activation
    Attention       // Flash attention pattern
  };
  
  SmallVector<FusionCluster> findFusionClusters(const OperationDependencyGraph& graph) {
    SmallVector<FusionCluster> clusters;
    
    // Find elementwise fusion opportunities
    findElementwiseFusions(graph, clusters);
    
    // Find matmul + bias + activation patterns
    findMatmulBiasActivationFusions(graph, clusters);
    
    // Find attention pattern fusions
    findAttentionFusions(graph, clusters);
    
    return clusters;
  }
  
  void findElementwiseFusions(const OperationDependencyGraph& graph,
                             SmallVector<FusionCluster>& clusters) {
    // Find chains of elementwise operations
    for (auto* op : graph.getOperations()) {
      if (!isElementwise(op)) continue;
      
      FusionCluster cluster;
      cluster.type = FusionType::Elementwise;
      
      // Collect fusable chain starting from this op
      collectElementwiseChain(op, graph, cluster.ops);
      
      if (cluster.size() > 1) {
        clusters.push_back(std::move(cluster));
      }
    }
  }
  
  void findMatmulBiasActivationFusions(const OperationDependencyGraph& graph,
                                      SmallVector<FusionCluster>& clusters) {
    for (auto* op : graph.getOperations()) {
      if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
        FusionCluster cluster;
        cluster.type = FusionType::MatmulBias;
        cluster.ops.push_back(matmulOp);
        
        // Look for bias addition
        for (auto* user : matmulOp->getUsers()) {
          if (auto addOp = dyn_cast<AddOp>(user)) {
            // Check if this is a bias add (broadcasting)
            if (isBiasAdd(addOp, matmulOp->getResult(0))) {
              cluster.ops.push_back(addOp);
              
              // Look for activation
              for (auto* biasUser : addOp->getUsers()) {
                if (isActivationOp(biasUser)) {
                  cluster.ops.push_back(biasUser);
                  break;
                }
              }
              break;
            }
          }
        }
        
        if (cluster.size() > 1) {
          clusters.push_back(std::move(cluster));
        }
      }
    }
  }
  
  void findAttentionFusions(const OperationDependencyGraph& graph,
                           SmallVector<FusionCluster>& clusters) {
    // Pattern: Q @ K^T -> softmax -> @ V
    for (auto* op : graph.getOperations()) {
      if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
        // Check if RHS is transposed (attention scores)
        if (matmulOp->hasAttr("rhs_transpose")) {
          FusionCluster cluster;
          cluster.type = FusionType::Attention;
          cluster.ops.push_back(matmulOp);
          
          // Look for scale -> softmax -> matmul pattern
          Operation* currentOp = matmulOp;
          
          // Optional scaling
          if (auto scaleUser = getSingleUserOfType<MulOp>(currentOp)) {
            cluster.ops.push_back(scaleUser);
            currentOp = scaleUser;
          }
          
          // Softmax
          if (auto softmaxUser = getSingleUserOfType<SoftmaxOp>(currentOp)) {
            cluster.ops.push_back(softmaxUser);
            currentOp = softmaxUser;
            
            // Value matmul
            if (auto valueMatmul = getSingleUserOfType<MatmulOp>(currentOp)) {
              cluster.ops.push_back(valueMatmul);
              
              clusters.push_back(std::move(cluster));
            }
          }
        }
      }
    }
  }
  
  void fuseClusters(const FusionCluster& cluster) {
    OpBuilder builder(cluster.ops[0]);
    
    switch (cluster.type) {
      case FusionType::Elementwise:
        createElementwiseFusion(builder, cluster.ops);
        break;
      case FusionType::MatmulBias:
        createMatmulBiasFusion(builder, cluster.ops);
        break;
      case FusionType::Attention:
        createAttentionFusion(builder, cluster.ops);
        break;
    }
  }
  
  void createAttentionFusion(OpBuilder& builder, const SmallVector<Operation*>& ops) {
    // Extract Q, K, V from the pattern
    auto firstMatmul = cast<MatmulOp>(ops[0]);
    Value Q = firstMatmul.getLhs();
    Value K_T = firstMatmul.getRhs();
    
    // Find the value matmul to get V
    MatmulOp valueMatmul = nullptr;
    for (auto* op : ops) {
      if (auto matmul = dyn_cast<MatmulOp>(op)) {
        if (matmul != firstMatmul) {
          valueMatmul = matmul;
          break;
        }
      }
    }
    
    if (!valueMatmul) return;
    
    Value V = valueMatmul.getRhs();
    
    // Create fused flash attention op
    auto fusedOp = builder.create<FlashAttentionOp>(
      firstMatmul.getLoc(),
      valueMatmul.getType(),
      Q, K_T, V);
    
    // Set attributes
    fusedOp->setAttr("causal", builder.getBoolAttr(false));
    fusedOp->setAttr("scale", builder.getF32FloatAttr(1.0f / std::sqrt(64.0f))); // head_dim = 64
    
    // Replace the last operation's uses
    valueMatmul.getResult().replaceAllUsesWith(fusedOp.getResult());
    
    // Remove original operations
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      (*it)->erase();
    }
  }
};
```

### 5. Distribution Pass

This pass handles mesh parallelism and distributed tensor operations.

```cpp
class DistributionPass : public PassWrapper<DistributionPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Collect mesh and sharding specifications
    MeshAnalysis meshAnalysis(module);
    
    // Transform distributed operations
    module.walk([&](Operation* op) {
      if (auto distributedOp = dyn_cast<DistributedOpInterface>(op)) {
        lowerDistributedOperation(distributedOp, meshAnalysis);
      }
    });
  }

private:
  void lowerDistributedOperation(Operation* op, const MeshAnalysis& meshAnalysis) {
    OpBuilder builder(op);
    
    if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
      lowerDistributedMatmul(matmulOp, meshAnalysis, builder);
    } else if (auto allreduceOp = dyn_cast<AllreduceOp>(op)) {
      lowerAllreduce(allreduceOp, meshAnalysis, builder);
    }
  }
  
  void lowerDistributedMatmul(MatmulOp op, const MeshAnalysis& meshAnalysis, OpBuilder& builder) {
    // Get sharding specifications
    auto lhsSharding = getShardingSpec(op.getLhs());
    auto rhsSharding = getShardingSpec(op.getRhs());
    auto resultSharding = getShardingSpec(op.getResult());
    
    // Determine if collectives are needed
    bool needsAllGather = false;
    bool needsReduceScatter = false;
    
    // Analyze sharding pattern
    // For tensor parallel matmul: A @ B_sharded -> needs all_gather
    if (rhsSharding.isSharded() && !lhsSharding.isSharded()) {
      needsAllGather = true;
    }
    
    // For data parallel: sharded_A @ B -> needs reduce_scatter
    if (lhsSharding.isSharded() && !rhsSharding.isSharded()) {
      needsReduceScatter = true;
    }
    
    if (needsAllGather) {
      // Insert all_gather before matmul
      auto allGatherOp = builder.create<AllGatherOp>(
        op.getLoc(),
        op.getRhs().getType(), // unsharded type
        op.getRhs(),
        meshAnalysis.getMeshAttr(),
        builder.getStringAttr("tensor_parallel"));
      
      // Update matmul to use gathered tensor
      op->setOperand(1, allGatherOp.getResult());
    }
    
    if (needsReduceScatter) {
      // Insert reduce_scatter after matmul
      builder.setInsertionPointAfter(op);
      
      auto reduceScatterOp = builder.create<ReduceScatterOp>(
        op.getLoc(),
        resultSharding.getShardedType(),
        op.getResult(),
        meshAnalysis.getMeshAttr(),
        builder.getStringAttr("data_parallel"),
        builder.getStringAttr("sum"));
      
      // Replace uses of matmul result
      op.getResult().replaceAllUsesExcept(reduceScatterOp.getResult(), reduceScatterOp);
    }
  }
  
  void lowerAllreduce(AllreduceOp op, const MeshAnalysis& meshAnalysis, OpBuilder& builder) {
    // Lower to NCCL collective
    auto ncclOp = builder.create<NCCLAllreduceOp>(
      op.getLoc(),
      op.getType(),
      op.getInput(),
      op.getReductionKind(),
      meshAnalysis.getDeviceListAttr());
    
    op.getResult().replaceAllUsesWith(ncclOp.getResult());
    op->erase();
  }
};

// Helper class for mesh analysis
class MeshAnalysis {
public:
  MeshAnalysis(ModuleOp module) {
    // Extract mesh specifications from module
    module.walk([&](MeshOp meshOp) {
      meshes_[meshOp.getName()] = meshOp;
    });
  }
  
  MeshOp getMesh(StringRef name) const {
    auto it = meshes_.find(name);
    return it != meshes_.end() ? it->second : nullptr;
  }
  
  Attribute getMeshAttr() const {
    // Return default mesh attribute
    return /* construct mesh attribute */;
  }
  
  Attribute getDeviceListAttr() const {
    // Return device list for collectives
    return /* construct device list */;
  }

private:
  DenseMap<StringRef, MeshOp> meshes_;
};
```

### 6. Effect Analysis Pass

This pass analyzes and orders operations with side effects (RNG, state updates, collectives).

```cpp
class EffectAnalysisPass : public PassWrapper<EffectAnalysisPass, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    
    // Build effect dependency graph
    EffectDependencyGraph effectGraph;
    analyzeEffects(func, effectGraph);
    
    // Insert explicit dependencies
    insertEffectDependencies(func, effectGraph);
  }

private:
  struct EffectInfo {
    Operation* op;
    EffectKind kind;
    Value resource;  // RNG state, collective group, etc.
  };
  
  enum class EffectKind {
    RandomRead,    // Read from RNG state
    RandomWrite,   // Update RNG state
    StateRead,     // Read from mutable state
    StateWrite,    // Write to mutable state
    CollectiveSync // Collective synchronization
  };
  
  class EffectDependencyGraph {
  public:
    void addEffect(Operation* op, EffectKind kind, Value resource) {
      effects_.emplace_back(EffectInfo{op, kind, resource});
    }
    
    SmallVector<std::pair<Operation*, Operation*>> getDependencies() const {
      SmallVector<std::pair<Operation*, Operation*>> deps;
      
      // RAW, WAR, WAW dependencies
      for (size_t i = 0; i < effects_.size(); ++i) {
        for (size_t j = i + 1; j < effects_.size(); ++j) {
          const auto& earlier = effects_[i];
          const auto& later = effects_[j];
          
          if (earlier.resource == later.resource && hasDependency(earlier.kind, later.kind)) {
            deps.emplace_back(earlier.op, later.op);
          }
        }
      }
      
      return deps;
    }
    
  private:
    SmallVector<EffectInfo> effects_;
    
    bool hasDependency(EffectKind earlier, EffectKind later) const {
      // RAW: earlier write, later read
      if ((earlier == EffectKind::RandomWrite && later == EffectKind::RandomRead) ||
          (earlier == EffectKind::StateWrite && later == EffectKind::StateRead)) {
        return true;
      }
      
      // WAR: earlier read, later write  
      if ((earlier == EffectKind::RandomRead && later == EffectKind::RandomWrite) ||
          (earlier == EffectKind::StateRead && later == EffectKind::StateWrite)) {
        return true;
      }
      
      // WAW: earlier write, later write
      if ((earlier == EffectKind::RandomWrite && later == EffectKind::RandomWrite) ||
          (earlier == EffectKind::StateWrite && later == EffectKind::StateWrite)) {
        return true;
      }
      
      // All collectives have dependencies with each other
      if (earlier == EffectKind::CollectiveSync && later == EffectKind::CollectiveSync) {
        return true;
      }
      
      return false;
    }
  };
  
  void analyzeEffects(FuncOp func, EffectDependencyGraph& graph) {
    func.walk([&](Operation* op) {
      // Analyze random number generation
      if (auto randOp = dyn_cast<RandomOp>(op)) {
        Value rngState = randOp.getRngState();
        graph.addEffect(op, EffectKind::RandomRead, rngState);
        
        // If RNG state is updated
        if (randOp.getUpdatedState()) {
          graph.addEffect(op, EffectKind::RandomWrite, rngState);
        }
      }
      
      // Analyze state operations
      if (auto stateOp = dyn_cast<StateOpInterface>(op)) {
        for (auto resource : stateOp.getReadResources()) {
          graph.addEffect(op, EffectKind::StateRead, resource);
        }
        for (auto resource : stateOp.getWriteResources()) {
          graph.addEffect(op, EffectKind::StateWrite, resource);
        }
      }
      
      // Analyze collective operations
      if (isa<CollectiveOpInterface>(op)) {
        graph.addEffect(op, EffectKind::CollectiveSync, Value{});
      }
    });
  }
  
  void insertEffectDependencies(FuncOp func, const EffectDependencyGraph& graph) {
    auto dependencies = graph.getDependencies();
    
    for (auto [earlier, later] : dependencies) {
      // Insert explicit dependency token
      OpBuilder builder(later);
      
      auto tokenType = builder.getType<EffectTokenType>();
      auto dependencyOp = builder.create<EffectDependencyOp>(
        earlier->getLoc(), tokenType, earlier->getResults());
      
      // Add dependency to later operation
      SmallVector<Value> operands(later->getOperands());
      operands.push_back(dependencyOp.getResult());
      later->setOperands(operands);
    }
  }
};
```

## Testing Graph IR Passes

```cpp
// Unit test example
TEST(GraphIRPasses, TypeInference) {
  MLIRContext context;
  context.loadDialect<TesseraDialect>();
  
  // Create test module
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func @test(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<*xf32> {
      %0 = tessera.matmul %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<*xf32>
      return %0 : tensor<*xf32>
    }
  )mlir", &context);
  
  // Apply type inference pass
  PassManager pm(&context);
  pm.addPass(createTypeInferencePass());
  
  ASSERT_TRUE(succeeded(pm.run(module.get())));
  
  // Verify result type is inferred correctly
  auto func = module->lookupSymbol<FuncOp>("test");
  auto matmulOp = func.front().front();
  auto resultType = matmulOp->getResult(0).getType().cast<TensorType>();
  
  EXPECT_EQ(resultType.getRank(), 2);
  EXPECT_TRUE(resultType.isDynamicDim(0));
  EXPECT_TRUE(resultType.isDynamicDim(1));
}

TEST(GraphIRPasses, AutodiffGeneration) {
  MLIRContext context;
  context.loadDialect<TesseraDialect>();
  
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func @forward(%x: tensor<32x128xf32>, %w: tensor<128x64xf32>) -> tensor<32x64xf32> attributes {tessera.autodiff} {
      %0 = tessera.matmul %x, %w : tensor<32x128xf32>, tensor<128x64xf32> -> tensor<32x64xf32>
      return %0 : tensor<32x64xf32>
    }
  )mlir", &context);
  
  // Apply autodiff pass
  PassManager pm(&context);
  pm.addPass(createAutodiffPass(AutodiffMode::Reverse));
  
  ASSERT_TRUE(succeeded(pm.run(module.get())));
  
  // Verify gradient function was generated
  auto gradFunc = module->lookupSymbol<FuncOp>("forward_grad");
  EXPECT_TRUE(gradFunc);
  
  // Verify function signature: (x, w, grad_output) -> (grad_x, grad_w)
  auto funcType = gradFunc.getFunctionType();
  EXPECT_EQ(funcType.getNumInputs(), 3);
  EXPECT_EQ(funcType.getNumResults(), 2);
}
```

## Summary

Graph IR passes form the foundation of Tessera's compiler, handling:

- **Type inference** for shape and dtype propagation
- **Automatic differentiation** for gradient computation
- **Algebraic simplification** for mathematical optimizations
- **Fusion analysis** for performance optimization
- **Distribution lowering** for mesh parallelism
- **Effect analysis** for correct ordering of side effects

These passes work together to transform high-level Tessera code into an optimized intermediate representation ready for scheduling and code generation in the lower IR levels.

The next document will cover **Schedule IR Passes**, which handle tiling, memory placement, and pipeline generation.