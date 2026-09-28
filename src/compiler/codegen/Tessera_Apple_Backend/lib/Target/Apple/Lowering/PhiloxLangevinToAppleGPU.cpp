// Native Graph consumer for the bounded f32 Philox Langevin step.
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

#include <cmath>
#include <cstdint>
#include <limits>

using namespace ::mlir;

namespace tessera::apple {
namespace {

struct LowerPhiloxLangevin final : RewritePattern {
  explicit LowerPhiloxLangevin(MLIRContext *ctx)
      : RewritePattern("tessera.ebm.langevin_step_philox", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 4 || op->getNumResults() != 1)
      return failure();
    auto y = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto grad = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto seed = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
    auto counter = dyn_cast<RankedTensorType>(op->getOperand(3).getType());
    if (!y || !y.hasStaticShape() || !y.getElementType().isF32() ||
        y.getRank() < 1 || y != grad || y != op->getResult(0).getType() ||
        !seed || !counter || seed.getRank() != 1 || seed.getDimSize(0) != 1 ||
        counter.getRank() != 1 || counter.getDimSize(0) != 4 ||
        !seed.getElementType().isInteger(64) ||
        !counter.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op, "Apple Philox requires static f32 result and i64 seed[1]/counter[4]");
    int64_t count = 1;
    for (int64_t dim : y.getShape()) {
      if (dim <= 0 || count > std::numeric_limits<int32_t>::max() / dim)
        return rewriter.notifyMatchFailure(op, "Apple Philox exceeds the i32 element-count ABI");
      count *= dim;
    }
    auto eta = op->getAttrOfType<FloatAttr>("eta");
    auto temperature = op->getAttrOfType<FloatAttr>("temperature");
    auto explicitNoise = op->getAttrOfType<FloatAttr>("noise_scale");
    if (!eta || !temperature)
      return rewriter.notifyMatchFailure(op, "Apple Philox needs eta and temperature");
    double etaValue = eta.getValueAsDouble();
    double temperatureValue = temperature.getValueAsDouble();
    double noiseValue = explicitNoise ? explicitNoise.getValueAsDouble()
                                      : std::sqrt(2.0 * etaValue * temperatureValue);
    if (!std::isfinite(etaValue) || etaValue <= 0 ||
        !std::isfinite(temperatureValue) || temperatureValue <= 0 ||
        !std::isfinite(noiseValue) || noiseValue < 0 ||
        etaValue > std::numeric_limits<float>::max() ||
        noiseValue > std::numeric_limits<float>::max() ||
        static_cast<float>(etaValue) <= 0 ||
        (!explicitNoise && noiseValue > 0 &&
         static_cast<float>(noiseValue) <= 0))
      return rewriter.notifyMatchFailure(op, "Apple Philox has an unrepresentable numeric policy");

    Location loc = op->getLoc();
    auto i64 = rewriter.getI64Type();
    auto i32 = rewriter.getI32Type();
    auto f32 = rewriter.getF32Type();
    auto yMem = MemRefType::get(y.getShape(), f32);
    auto seedMem = MemRefType::get({1}, i64);
    auto counterMem = MemRefType::get({4}, i64);
    Value yPtr = extractPtr(rewriter, loc, op->getOperand(0), yMem);
    Value gradPtr = extractPtr(rewriter, loc, op->getOperand(1), yMem);
    Value seedPtr = extractPtr(rewriter, loc, op->getOperand(2), seedMem);
    Value counterPtr = extractPtr(rewriter, loc, op->getOperand(3), counterMem);
    auto out = rewriter.create<memref::AllocOp>(loc, yMem);
    Value outIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, out);
    Value outPtr = rewriter.create<arith::IndexCastOp>(loc, i64, outIndex);
    Value etaArg = rewriter.create<arith::ConstantFloatOp>(loc, f32,
        APFloat(static_cast<float>(etaValue)));
    Value noiseArg = rewriter.create<arith::ConstantFloatOp>(loc, f32,
        APFloat(static_cast<float>(noiseValue)));
    Value nArg = rewriter.create<arith::ConstantIntOp>(loc, count, 32);
    constexpr llvm::StringLiteral symbol =
        "tessera_apple_gpu_ebm_langevin_step_philox_graph_f32_status";
    auto fnType = FunctionType::get(op->getContext(),
        {i64, i64, i64, i64, f32, f32, i64, i32}, {i32});
    ensureExternalDecl(op->getParentOfType<ModuleOp>(), symbol, fnType);
    auto status = rewriter.create<func::CallOp>(loc, symbol, TypeRange{i32},
        ValueRange{yPtr, gradPtr, seedPtr, counterPtr, etaArg, noiseArg, outPtr, nArg});
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    Value succeeded = rewriter.create<arith::CmpIOp>(loc,
        arith::CmpIPredicate::eq, status.getResult(0), one);
    rewriter.create<cf::AssertOp>(loc, succeeded,
        "Apple Philox Langevin did not execute on Metal");
    Value result = rewriter.create<bufferization::ToTensorOp>(loc, y, out);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerPhiloxLangevinToAppleGPUPass final
    : PassWrapper<LowerPhiloxLangevinToAppleGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPhiloxLangevinToAppleGPUPass)
  StringRef getArgument() const override {
    return "tessera-philox-langevin-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower bounded f32 Philox Langevin Graph op to checked Metal ABI";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    cf::ControlFlowDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerPhiloxLangevin>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    bool unmatched = false;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() ==
          "tessera.ebm.langevin_step_philox") {
        op->emitError("Apple Philox Langevin is outside the bounded Metal ABI");
        unmatched = true;
      }
    });
    if (unmatched) signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createLowerPhiloxLangevinToAppleGPUPass() {
  return std::make_unique<LowerPhiloxLangevinToAppleGPUPass>();
}
} // namespace tessera::apple
