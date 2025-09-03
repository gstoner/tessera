//===- LowerKANToTessera.cpp - rewrite kan.* to base ops ---------------------------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

namespace tessera { namespace kan {

struct LowerKANPass : public mlir::PassWrapper<LowerKANPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKANPass)
  void runOnOperation() override {
    // TODO: register patterns:
    //  - kan.bspline_eval -> affine/loop nest or vector dialect
    //  - kan.linear_mix  -> reshape + linalg.matmul (+ add)
  }
};

std::unique_ptr<mlir::Pass> createLowerKANToTesseraPass() {
  return std::make_unique<LowerKANPass>();
}

}} // namespace