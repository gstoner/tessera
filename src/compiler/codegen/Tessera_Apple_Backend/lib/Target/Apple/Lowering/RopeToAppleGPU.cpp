//===- RopeToAppleGPU.cpp - Lower tessera.rope to a custom MSL kernel ----===//
//
// Phase 8.4 — Apple GPU custom MSL kernel path.
//
// Replaces tessera.rope ops (rank-2, f32, x.shape == theta.shape) with calls
// to the Apple-GPU runtime shim:
//
//   tessera_apple_gpu_rope_f32_status
//       (X: f32, Theta: f32 -> Out: f32, row-major; 1 = Metal executed)
//
// Runtime side: the shim carries an embedded MSL source for the rope kernel,
// compiles it via [device newLibraryWithSource:options:error:] on first call,
// caches the resulting MTLComputePipelineState by sha256 of the source, and
// dispatches via MTLComputeCommandEncoder.
//
// Same plumbing shape as MatmulToAppleGPU.cpp — three i64 pointers + two i32
// dim sizes + a row-major output memref allocated on the host.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/LoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <limits>

using namespace ::mlir;

namespace tessera {
namespace apple {

namespace {

constexpr llvm::StringLiteral kRopeF32Symbol = "tessera_apple_gpu_rope_f32_status";
constexpr llvm::StringLiteral kRopeF16Symbol = "tessera_apple_gpu_rope_f16";
constexpr llvm::StringLiteral kRopeBF16Symbol = "tessera_apple_gpu_rope_bf16";

// The composite rewrite emits exactly this tensor division for scaled
// ntk_rope. Keep its physical execution in the Apple compiler pipeline: a
// status-bearing Metal call, with no host fallback hidden behind a void ABI.
struct LowerGraphDivToAppleGPU : public RewritePattern {
  LowerGraphDivToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.div", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    auto ty = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!ty || ty.getRank() != 2 || !ty.hasStaticShape() ||
        !ty.getElementType().isF32() || op->getOperand(1).getType() != ty ||
        op->getResult(0).getType() != ty)
      return rewriter.notifyMatchFailure(op,
          "Apple Graph division requires equal static rank-2 f32 tensors");
    int64_t n = 1;
    for (int64_t dim : ty.getShape()) {
      if (dim <= 0 || n > std::numeric_limits<int64_t>::max() / dim)
        return rewriter.notifyMatchFailure(op, "Apple Graph division exceeds its ABI");
      n *= dim;
    }
    Location loc = op->getLoc();
    auto memTy = MemRefType::get(ty.getShape(), rewriter.getF32Type());
    auto ptrTy = rewriter.getI64Type();
    Value lhs = extractPtr(rewriter, loc, op->getOperand(0), memTy);
    Value rhs = extractPtr(rewriter, loc, op->getOperand(1), memTy);
    auto out = rewriter.create<memref::AllocOp>(loc, memTy);
    Value outIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, out);
    Value outPtr = rewriter.create<arith::IndexCastOp>(loc, ptrTy, outIndex);
    Value opcode = rewriter.create<arith::ConstantIntOp>(loc, 3, 32);
    Value count = rewriter.create<arith::ConstantIntOp>(loc, n, 64);
    constexpr llvm::StringLiteral symbol =
        "tessera_apple_gpu_mpsgraph_binary_f32_status";
    auto i32Ty = rewriter.getI32Type();
    auto fnTy = FunctionType::get(op->getContext(),
        {i32Ty, ptrTy, ptrTy, ptrTy, ptrTy}, {i32Ty});
    ensureExternalDecl(op->getParentOfType<ModuleOp>(), symbol, fnTy);
    auto status = rewriter.create<func::CallOp>(loc, symbol, TypeRange{i32Ty},
        ValueRange{opcode, lhs, rhs, outPtr, count});
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    Value succeeded = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
        status.getResult(0), one);
    rewriter.create<cf::AssertOp>(loc, succeeded,
        "Apple Graph division did not execute on Metal");
    Value result = rewriter.create<bufferization::ToTensorOp>(loc, ty, out);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerGraphDivToAppleGPUPass
    : public PassWrapper<LowerGraphDivToAppleGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphDivToAppleGPUPass)
  StringRef getArgument() const override { return "tessera-graph-div-to-apple_gpu"; }
  StringRef getDescription() const override {
    return "Lower same-shape static rank-2 f32 Graph division to the checked Apple Metal ABI";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    cf::ControlFlowDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerGraphDivToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};



struct LowerRopeToAppleGPU : public RewritePattern {
  LowerRopeToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.rope", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2)
      return failure();
    Value x = op->getOperand(0);
    Value theta = op->getOperand(1);

    auto xTy = dyn_cast<RankedTensorType>(x.getType());
    auto thetaTy = dyn_cast<RankedTensorType>(theta.getType());
    if (!xTy || !thetaTy || xTy.getRank() != 2 || thetaTy.getRank() != 2)
      return failure();

    Type xElem = xTy.getElementType();
    Type thetaElem = thetaTy.getElementType();
    if (xElem != thetaElem)
      return rewriter.notifyMatchFailure(
          op, "AppleGPU rope MSL path requires matching x/theta dtypes");

    // Phase 8.4.4.1 — pick the runtime symbol by dtype. Same i64×3 + i32×2
    // ABI shape across all three; the element type is encoded in the
    // symbol name, not the signature.
    StringRef symbol;
    if (xElem.isF32()) {
      symbol = kRopeF32Symbol;
    } else if (xElem.isF16()) {
      symbol = kRopeF16Symbol;
    } else if (xElem.isBF16()) {
      symbol = kRopeBF16Symbol;
    } else {
      return rewriter.notifyMatchFailure(
          op, "AppleGPU rope MSL path supports f32, f16, and bf16 in Phase 8.4.4.1");
    }

    if (xTy.isDynamicDim(0) || xTy.isDynamicDim(1) ||
        thetaTy.isDynamicDim(0) || thetaTy.isDynamicDim(1))
      return rewriter.notifyMatchFailure(
          op, "AppleGPU rope MSL path requires static shapes");

    int64_t M = xTy.getDimSize(0);
    int64_t K = xTy.getDimSize(1);
    if (K % 2 != 0)
      return rewriter.notifyMatchFailure(
          op, "rope requires an even innermost dimension");
    if (thetaTy.getDimSize(0) != M || thetaTy.getDimSize(1) != K)
      return rewriter.notifyMatchFailure(
          op, "AppleGPU rope MSL path requires x.shape == theta.shape "
              "in Phase 8.4");

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();

    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();

    auto memTy = MemRefType::get({M, K}, xElem);
    Value xPtr = extractPtr(rewriter, loc, x, memTy);
    Value thetaPtr = extractPtr(rewriter, loc, theta, memTy);
    auto outAlloc = rewriter.create<memref::AllocOp>(loc, memTy);
    Value outPtr;
    {
      auto pi =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outAlloc);
      outPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pi);
    }

    Value Mv = rewriter.create<arith::ConstantIntOp>(loc, M, 32);
    Value Kv = rewriter.create<arith::ConstantIntOp>(loc, K, 32);

    bool checkedF32 = xElem.isF32();
    FunctionType ropeFnTy = FunctionType::get(ctx,
        {i64Ty, i64Ty, i64Ty, i32Ty, i32Ty},
        checkedF32 ? TypeRange{i32Ty} : TypeRange{});
    ensureExternalDecl(mod, symbol, ropeFnTy);

    auto call = rewriter.create<func::CallOp>(loc, symbol,
        checkedF32 ? TypeRange{i32Ty} : TypeRange{},
        ValueRange{xPtr, thetaPtr, outPtr, Mv, Kv});
    if (checkedF32) {
      Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
      Value succeeded = rewriter.create<arith::CmpIOp>(loc,
          arith::CmpIPredicate::eq, call.getResult(0), one);
      rewriter.create<cf::AssertOp>(loc, succeeded,
          "Apple rope did not execute on Metal");
    }

    auto outTensorTy = RankedTensorType::get({M, K}, xElem);
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, outAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerRopeToAppleGPUPass
    : public PassWrapper<LowerRopeToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRopeToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-rope-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.rope (rank-2, f32/f16/bf16) to Apple GPU runtime "
           "calls (custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    cf::ControlFlowDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerRopeToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerRopeToAppleGPUPass() {
  return std::make_unique<LowerRopeToAppleGPUPass>();
}

std::unique_ptr<Pass> createLowerGraphDivToAppleGPUPass() {
  return std::make_unique<LowerGraphDivToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
