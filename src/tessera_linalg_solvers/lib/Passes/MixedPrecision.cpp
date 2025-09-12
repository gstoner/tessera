//===- MixedPrecision.cpp - Insert quant/dequant & set policies ---------*- C++ -*-===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <optional>

namespace tessera { namespace solver {

struct MixedPrecisionPass : public mlir::PassWrapper<MixedPrecisionPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MixedPrecisionPass)
  void runOnOperation() override {
    // Walk solver ops, attach default precision policies where missing.
    // Insert quantize/dequantize stubs (tessera.quantize/dequantize) around factor/solve boundaries.
    getOperation().walk([&](mlir::Operation *op){
      (void)op; // TODO: implement
    });
  }
};

std::unique_ptr<mlir::Pass> createMixedPrecisionPass() {
  return std::make_unique<MixedPrecisionPass>();
}

}} // namespace tessera::solver
