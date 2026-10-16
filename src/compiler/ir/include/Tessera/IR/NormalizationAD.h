#pragma once
#include "Tessera/IR/TesseraOps.h"
#include "Tessera/IR/NumericPolicyAccumulator.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
namespace tessera::normalization_ad {
inline mlir::RankedTensorType normalizationWorkType(
    mlir::Operation *op, mlir::RankedTensorType storage) {
  mlir::FloatType accum;
  switch (numeric_policy::readAccumulator(op, storage.getElementType(), accum)) {
  case numeric_policy::AccumRealizability::NoWidening: return storage;
  case numeric_policy::AccumRealizability::Widen:
    return mlir::cast<mlir::RankedTensorType>(storage.clone(accum));
  case numeric_policy::AccumRealizability::Unrealizable: return {};
  }
  llvm_unreachable("unknown accumulator realizability");
}
inline mlir::Value normalizationCast(mlir::OpBuilder &builder,
    mlir::Location loc, mlir::Value value, mlir::Type element) {
  auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
  if (type.getElementType() == element) return value;
  auto target = mlir::cast<mlir::RankedTensorType>(type.clone(element));
  return builder.create<tessera::CastOp>(
      loc, target, value, mlir::DictionaryAttr{}).getY();
}
inline void normalizationGuard(mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value primal, mlir::Value seed) {
  auto type = mlir::cast<mlir::RankedTensorType>(primal.getType());
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    if (!type.isDynamicDim(axis)) continue;
    auto p = builder.create<mlir::tensor::DimOp>(loc, primal, axis);
    auto d = builder.create<mlir::tensor::DimOp>(loc, seed, axis);
    auto equal = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, p, d);
    builder.create<mlir::cf::AssertOp>(
        loc, equal, "normalization derivative seed shape must match primal");
  }
}

}
