//===- TileEpilogue.h - shared portable Tile epilogue emission -*- C++ -*-===//

#ifndef TESSERA_DIALECT_TILE_EPILOGUE_H
#define TESSERA_DIALECT_TILE_EPILOGUE_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"

namespace tessera::tile {

inline bool isSupportedActivation(llvm::StringRef activation) {
  return activation == "none" || activation == "relu" ||
         activation == "gelu" || activation == "silu";
}

inline bool isSupportedOutputType(llvm::StringRef outputType) {
  return outputType == "f32" || outputType == "f16" || outputType == "i32";
}

/// Emit a portable scalar floating-point activation directly in the caller's
/// IR. This is a compile-time helper, not a runtime call. Bias loading and final
/// stores remain backend-owned because their address spaces differ.
inline mlir::Value emitScalarFloatActivation(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value value,
                                              llvm::StringRef activation) {
  mlir::Type type = value.getType();
  auto constant = [&](double number) {
    return mlir::arith::ConstantOp::create(
        builder, loc, type, builder.getFloatAttr(type, number));
  };
  if (activation == "none")
    return value;
  if (activation == "relu")
    return mlir::arith::MaximumFOp::create(builder, loc, value, constant(0.0));
  if (activation == "silu") {
    mlir::Value exponent = mlir::math::ExpOp::create(
        builder, loc, mlir::arith::NegFOp::create(builder, loc, value));
    mlir::Value denominator = mlir::arith::AddFOp::create(
        builder, loc, constant(1.0), exponent);
    return mlir::arith::DivFOp::create(builder, loc, value, denominator);
  }

  // GELU tanh approximation, matching gelu(approximate="tanh"):
  // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). The Tile attribute verifier
  // guarantees that the only remaining activation is "gelu".
  mlir::Value x2 = mlir::arith::MulFOp::create(builder, loc, value, value);
  mlir::Value x3 = mlir::arith::MulFOp::create(builder, loc, x2, value);
  mlir::Value inner = mlir::arith::AddFOp::create(
      builder, loc, value,
      mlir::arith::MulFOp::create(builder, loc, constant(0.044715), x3));
  inner = mlir::arith::MulFOp::create(builder, loc, constant(0.7978845608028654),
                                      inner);
  mlir::Value onePlusTanh = mlir::arith::AddFOp::create(
      builder, loc, constant(1.0),
      mlir::math::TanhOp::create(builder, loc, inner));
  return mlir::arith::MulFOp::create(
      builder, loc,
      mlir::arith::MulFOp::create(builder, loc, constant(0.5), value),
      onePlusTanh);
}

/// Convert an f32 accumulator scalar to the requested portable float output.
inline mlir::Value emitFloatOutputConversion(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value value,
                                              mlir::Type outputType) {
  if (value.getType() == outputType)
    return value;
  return mlir::arith::TruncFOp::create(builder, loc, outputType, value);
}

} // namespace tessera::tile

#endif // TESSERA_DIALECT_TILE_EPILOGUE_H
