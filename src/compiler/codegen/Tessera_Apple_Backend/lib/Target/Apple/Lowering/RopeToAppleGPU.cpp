//===- RopeToAppleGPU.cpp - Lower tessera.rope to a custom MSL kernel ----===//
//
// Phase 8.4 — Apple GPU custom MSL kernel path.
//
// Replaces tessera.rope ops (rank-2, f32, x.shape == theta.shape) with calls
// to the Apple-GPU runtime shim:
//
//   tessera_apple_gpu_rope_f32  (X: f32, Theta: f32 -> Out: f32, row-major)
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace ::mlir;

namespace tessera {
namespace apple {

namespace {

constexpr llvm::StringLiteral kRopeF32Symbol = "tessera_apple_gpu_rope_f32";
constexpr llvm::StringLiteral kRopeF16Symbol = "tessera_apple_gpu_rope_f16";
constexpr llvm::StringLiteral kRopeBF16Symbol = "tessera_apple_gpu_rope_bf16";



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

    FunctionType ropeFnTy =
        FunctionType::get(ctx, {i64Ty, i64Ty, i64Ty, i32Ty, i32Ty}, {});
    ensureExternalDecl(mod, symbol, ropeFnTy);

    rewriter.create<func::CallOp>(
        loc, symbol, TypeRange{},
        ValueRange{xPtr, thetaPtr, outPtr, Mv, Kv});

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

} // namespace apple
} // namespace tessera
