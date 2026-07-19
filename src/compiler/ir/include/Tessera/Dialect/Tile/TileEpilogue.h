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
  return outputType == "f64" || outputType == "f32" ||
         outputType == "f16" || outputType == "i32";
}

/// Portable bounded tanh approximation used where a target has no selectable
/// LLVM `ftanh` lowering. Keeping it shared prevents attention softcap and
/// fused epilogues from silently using different arithmetic definitions.
inline mlir::Value emitBoundedTanhApprox(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value input) {
  mlir::Type type = input.getType();
  auto constant = [&](double number) {
    return mlir::arith::ConstantOp::create(
        builder, loc, type, builder.getFloatAttr(type, number));
  };
  mlir::Value bounded = mlir::arith::MaximumFOp::create(
      builder, loc, input, constant(-5.0));
  bounded = mlir::arith::MinimumFOp::create(builder, loc, bounded,
                                             constant(5.0));
  mlir::Value z2 = mlir::arith::MulFOp::create(builder, loc, bounded, bounded);
  auto horner = [&](double a, double b, mlir::Value x) {
    return mlir::arith::AddFOp::create(
        builder, loc, constant(a),
        mlir::arith::MulFOp::create(builder, loc, constant(b), x));
  };
  mlir::Value numeratorPoly = horner(135135.0, 17325.0, z2);
  numeratorPoly = mlir::arith::AddFOp::create(
      builder, loc, numeratorPoly,
      mlir::arith::MulFOp::create(builder, loc, horner(0.0, 378.0, z2), z2));
  numeratorPoly = mlir::arith::AddFOp::create(
      builder, loc, numeratorPoly,
      mlir::arith::MulFOp::create(
          builder, loc, mlir::arith::MulFOp::create(builder, loc, z2, z2), z2));
  mlir::Value denominatorPoly = horner(135135.0, 62370.0, z2);
  denominatorPoly = mlir::arith::AddFOp::create(
      builder, loc, denominatorPoly,
      mlir::arith::MulFOp::create(builder, loc, horner(0.0, 3150.0, z2), z2));
  denominatorPoly = mlir::arith::AddFOp::create(
      builder, loc, denominatorPoly,
      mlir::arith::MulFOp::create(
          builder, loc,
          mlir::arith::MulFOp::create(builder, loc, constant(28.0), z2),
          mlir::arith::MulFOp::create(builder, loc, z2, z2)));
  return mlir::arith::DivFOp::create(
      builder, loc,
      mlir::arith::MulFOp::create(builder, loc, bounded, numeratorPoly),
      denominatorPoly);
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
    // sigmoid(x) = 0.5 * (1 + tanh(x / 2)); use the arithmetic form above
    // so the CUDA path does not require an unavailable fexp libcall either.
    mlir::Value sigmoid = mlir::arith::MulFOp::create(
        builder, loc, constant(0.5),
        mlir::arith::AddFOp::create(
            builder, loc, constant(1.0),
            emitBoundedTanhApprox(builder, loc, mlir::arith::MulFOp::create(
                builder, loc, constant(0.5), value))));
    return mlir::arith::MulFOp::create(builder, loc, value, sigmoid);
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
      builder, loc, constant(1.0), emitBoundedTanhApprox(builder, loc, inner));
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
