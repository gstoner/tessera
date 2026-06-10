//===- UnaryToAppleGPU.cpp - Lower Tier-1 unary ops to MPSGraph lane -----===//
//
// 2026-05-29 — Apple GPU MetalPerformanceShadersGraph (MPSGraph) lane.
//
// Lowers the Tier-1 elementwise unary activations to a single op-coded
// runtime call:
//
//   tessera_apple_gpu_mpsgraph_unary_f32(int32 op, x, out, int64 n)
//   tessera_apple_gpu_mpsgraph_unary_f16(...)
//
// Op codes match apple_gpu_runtime.mm::mpsg_unary_node and
// runtime.py::_APPLE_GPU_UNARY_OPCODES. The kernel is elementwise, so the
// pass is rank-agnostic (folds to the total element count). bf16 falls
// through to the artifact-only path (the runtime upcasts bf16 host-side).
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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

using namespace ::mlir;

namespace tessera {
namespace apple {

namespace {

// op name -> opcode (must match mpsg_unary_node in apple_gpu_runtime.mm).
static int unaryOpcode(StringRef name) {
  return llvm::StringSwitch<int>(name)
      .Case("tessera.relu", 0)
      .Case("tessera.sigmoid", 1)
      .Case("tessera.sigmoid_safe", 1)
      .Case("tessera.tanh", 2)
      .Case("tessera.softplus", 3)
      .Case("tessera.silu", 4)
      .Case("tessera.exp", 6)
      .Case("tessera.log", 7)
      .Case("tessera.sqrt", 8)
      .Case("tessera.rsqrt", 9)
      .Case("tessera.neg", 10)
      .Case("tessera.negative", 10)
      .Case("tessera.abs", 11)
      .Case("tessera.absolute", 11)
      .Default(-1);
}



struct LowerUnaryToAppleGPU : public RewritePattern {
  LowerUnaryToAppleGPU(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    int opcode = unaryOpcode(op->getName().getStringRef());
    if (opcode < 0)
      return failure();
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return failure();
    Value x = op->getOperand(0);
    auto xTy = dyn_cast<RankedTensorType>(x.getType());
    if (!xTy || !xTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "AppleGPU unary lane needs a static-shape tensor");
    Type elem = xTy.getElementType();
    StringRef symbol;
    if (elem.isF32())
      symbol = "tessera_apple_gpu_mpsgraph_unary_f32";
    else if (elem.isF16())
      symbol = "tessera_apple_gpu_mpsgraph_unary_f16";
    else
      return rewriter.notifyMatchFailure(op, "AppleGPU unary lane supports f32/f16 (bf16 upcasts at runtime)");

    int64_t n = 1;
    for (int64_t d : xTy.getShape())
      n *= d;

    Location loc = op->getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = op->getContext();
    Type i64Ty = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();

    auto memTy = MemRefType::get(xTy.getShape(), elem);
    Value xPtr = extractPtr(rewriter, loc, x, memTy);
    auto outAlloc = rewriter.create<memref::AllocOp>(loc, memTy);
    auto outIdx =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outAlloc);
    Value outPtr = rewriter.create<arith::IndexCastOp>(loc, i64Ty, outIdx);

    Value opc = rewriter.create<arith::ConstantIntOp>(loc, opcode, 32);
    Value nv = rewriter.create<arith::ConstantIntOp>(loc, n, 64);

    FunctionType fnTy =
        FunctionType::get(ctx, {i32Ty, i64Ty, i64Ty, i64Ty}, {});
    ensureExternalDecl(mod, symbol, fnTy);
    rewriter.create<func::CallOp>(loc, symbol, TypeRange{},
                                  ValueRange{opc, xPtr, outPtr, nv});

    Value result = rewriter.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get(xTy.getShape(), elem), outAlloc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerUnaryToAppleGPUPass
    : public PassWrapper<LowerUnaryToAppleGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerUnaryToAppleGPUPass)

  StringRef getArgument() const override { return "tessera-unary-to-apple_gpu"; }
  StringRef getDescription() const override {
    return "Lower Tier-1 unary activations (silu/relu/sigmoid/tanh/softplus/"
           "exp/log/sqrt/rsqrt/neg/abs) to Apple GPU MPSGraph runtime calls";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerUnaryToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerUnaryToAppleGPUPass() {
  return std::make_unique<LowerUnaryToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
