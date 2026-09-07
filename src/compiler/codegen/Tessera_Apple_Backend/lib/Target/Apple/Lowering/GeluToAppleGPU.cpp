//===- GeluToAppleGPU.cpp - Lower tessera.gelu to MSL kernel -------------===//
//
// Phase 8.4.2 — Apple GPU custom MSL gelu kernel.
//
// Replaces static/dynamic rank-2 f32/f16/bf16 GELU with the corresponding
// native-only status ABI, after checking extents and element-count capacity:
//
//   tessera_apple_gpu_gelu_{f32,f16,bf16}_status(X, Out, N) -> i32
//
// The kernel is rank-agnostic at the runtime layer (one thread per element);
// the Phase 8.4.2 lowering pass restricts to rank-2 to keep the memref layout
// straightforward. Higher-rank tensors fall through to the artifact-only path.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/LoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

using namespace ::mlir;

namespace tessera {
namespace apple {

namespace {

constexpr llvm::StringLiteral kGeluF32Symbol = "tessera_apple_gpu_gelu_f32_status";
constexpr llvm::StringLiteral kGeluF16Symbol = "tessera_apple_gpu_gelu_f16_status";
constexpr llvm::StringLiteral kGeluBF16Symbol = "tessera_apple_gpu_gelu_bf16_status";



struct LowerGeluToAppleGPU : public RewritePattern {
  LowerGeluToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.gelu", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1)
      return failure();
    Value x = op->getOperand(0);
    auto xTy = dyn_cast<RankedTensorType>(x.getType());
    if (!xTy || xTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "AppleGPU gelu MSL path is rank-2 only in Phase 8.4.2");
    Type xElem = xTy.getElementType();
    StringRef symbol;
    if (xElem.isF32()) {
      symbol = kGeluF32Symbol;
    } else if (xElem.isF16()) {
      symbol = kGeluF16Symbol;
    } else if (xElem.isBF16()) {
      symbol = kGeluBF16Symbol;
    } else {
      return rewriter.notifyMatchFailure(
          op, "AppleGPU gelu MSL path supports f32, f16, and bf16 in Phase 8.4.4.1");
    }
    if (op->getNumResults() != 1 || op->getResult(0).getType() != xTy)
      return rewriter.notifyMatchFailure(op, "GELU input and output types must agree");
    int64_t M = xTy.getDimSize(0), K = xTy.getDimSize(1);
    for (int64_t extent : xTy.getShape())
      if (!ShapedType::isDynamic(extent) && (extent <= 0 || extent > INT32_MAX))
        return rewriter.notifyMatchFailure(op, "Apple runtime shape exceeds its positive i32 ABI");

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();

    SmallVector<Value> extents, dynamicSizes;
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value max = rewriter.create<arith::ConstantIndexOp>(loc, INT32_MAX);
    for (int64_t axis = 0; axis < 2; ++axis) {
      Value extent = rewriter.create<tensor::DimOp>(loc, x, axis);
      Value positive = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, extent, zero);
      Value bounded = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, extent, max);
      Value valid = rewriter.create<arith::AndIOp>(loc, positive, bounded);
      rewriter.create<cf::AssertOp>(loc, valid, "Apple GELU extent exceeds positive i32 ABI");
      extents.push_back(extent);
      if (xTy.isDynamicDim(axis)) dynamicSizes.push_back(extent);
    }
    // Each factor is at most INT32_MAX, so their index-width product fits i64.
    Value count = rewriter.create<arith::MulIOp>(loc, extents[0], extents[1]);
    Value fits = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, count, max);
    rewriter.create<cf::AssertOp>(loc, fits, "Apple GELU element count exceeds i32 ABI");
    Value Nv = rewriter.create<arith::IndexCastOp>(loc, i32Ty, count);
    auto memTy = MemRefType::get({M, K}, xElem);
    Value xPtr = extractPtr(rewriter, loc, x, memTy);
    auto outAlloc = rewriter.create<memref::AllocOp>(loc, memTy, dynamicSizes);
    auto pi = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outAlloc);
    Value outPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);

    FunctionType fnTy =
        FunctionType::get(ctx, {i64Ty, i64Ty, i32Ty}, {i32Ty});
    ensureExternalDecl(mod, symbol, fnTy);

    auto status = rewriter.create<func::CallOp>(
        loc, symbol, TypeRange{i32Ty}, ValueRange{xPtr, outPtr, Nv});

    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    Value succeeded = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, status.getResult(0), one);
    rewriter.create<cf::AssertOp>(loc, succeeded,
                                "Apple gelu did not execute on Metal");

    auto outTensorTy = RankedTensorType::get({M, K}, xElem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, outAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerGeluToAppleGPUPass
    : public PassWrapper<LowerGeluToAppleGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGeluToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-gelu-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.gelu (rank-2, f32/f16/bf16) to Apple GPU runtime "
           "calls (custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect, tensor::TensorDialect, cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerGeluToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerGeluToAppleGPUPass() {
  return std::make_unique<LowerGeluToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
