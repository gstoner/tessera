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

// Algebraically identical RMS projection with an explicit epsilon term.
// Scaling x by inverse^2 before both reductions avoids x*dx overflow and
// makes the two projections bitwise identical for an exactly radial seed.
// Keep the existing unspecified/storage accumulator formula in its callers.
inline mlir::Value stableRmsProjection(mlir::OpBuilder &builder,
    mlir::Location loc, mlir::RankedTensorType type, mlir::Value x,
    mlir::Value seed, mlir::Value inverse, mlir::FloatAttr epsilon) {
  auto mul=[&](mlir::Value a,mlir::Value b) {
    return builder.create<tessera::MulOp>(loc,type,a,b).getResult();
  };
  llvm::SmallVector<int64_t> statShape(type.getShape());
  statShape.pop_back();
  auto statType=mlir::RankedTensorType::get(
      statShape,type.getElementType(),type.getEncoding());
  llvm::SmallVector<mlir::Attribute> dimensions;
  for (int64_t axis=0;axis+1<type.getRank();++axis)
    dimensions.push_back(builder.getI64IntegerAttr(axis));
  auto mean=[&](mlir::Value value) {
    auto reduced=builder.create<tessera::ReduceOp>(loc,statType,value,
        builder.getStringAttr("mean"),builder.getI64IntegerAttr(type.getRank()-1)).getResult();
    return builder.create<tessera::BroadcastInDimOp>(
        loc,type,reduced,x,builder.getArrayAttr(dimensions)).getY();
  };
  auto inverseSquared=mul(inverse,inverse);
  auto weightedX=mul(x,inverseSquared);
  auto varianceScaled=mean(mul(x,weightedX));
  auto directionScaled=mean(mul(seed,weightedX));
  auto residual=builder.create<tessera::SubOp>(
      loc,type,mul(seed,varianceScaled),mul(x,directionScaled)).getResult();
  auto scalarType=mlir::RankedTensorType::get({},type.getElementType());
  auto scalar=builder.create<mlir::arith::ConstantOp>(loc,
      mlir::DenseElementsAttr::get(scalarType,builder.getFloatAttr(
          type.getElementType(),epsilon ? epsilon.getValueAsDouble() : 1.0e-5)));
  auto epsilonLike=builder.create<tessera::BroadcastInDimOp>(
      loc,type,scalar.getResult(),x,builder.getArrayAttr({})).getY();
  auto epsilonDirection=mul(seed,mul(epsilonLike,inverseSquared));
  auto corrected=builder.create<tessera::AddOp>(
      loc,type,residual,epsilonDirection).getResult();
  return mul(corrected,inverse);
}

}
