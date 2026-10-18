// Ordered FP32 contraction shared by native target materializers.
#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <array>

namespace tessera::scalar_contraction {
// Callers verify the six-operand contract and supply target-owned block/thread
// indices in the same shape type as M/N/K. Buffer operands are either raw
// device pointers or flat typed memrefs; this does not construct host tensors.
template <typename BuildIds>
inline mlir::LogicalResult emitBody(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange inputs,
    mlir::Type aType, mlir::Type bType, mlir::Type outputType,
    bool transposeA, bool transposeB, BuildIds buildIds) {
  using namespace mlir;
  Type shapeType = inputs[3].getType();
  auto shapeConstant = [&](OpBuilder &b, Location at, int64_t value) -> Value {
    if (shapeType.isIndex())
      return arith::ConstantIndexOp::create(b, at, value);
    return arith::ConstantIntOp::create(b, at, value, 64);
  };
  auto addShape = [](OpBuilder &b, Location at, Value lhs, Value rhs) -> Value {
    return arith::AddIOp::create(b, at, lhs, rhs);
  };
  auto mulShape = [](OpBuilder &b, Location at, Value lhs, Value rhs) -> Value {
    return arith::MulIOp::create(b, at, lhs, rhs);
  };
  auto lessShape = [](OpBuilder &b, Location at, Value lhs, Value rhs) -> Value {
    return arith::CmpIOp::create(b, at, arith::CmpIPredicate::ult, lhs, rhs);
  };
  Value m = inputs[3], n = inputs[4], k = inputs[5];
  auto f32 = builder.getF32Type();
  Value zero = shapeConstant(builder, loc, 0);
  Value one = shapeConstant(builder, loc, 1);
  auto [blockX, blockY, tid] = buildIds();
  Value inWarp = lessShape(builder, loc, tid, shapeConstant(builder, loc, 32));
  Value positive = arith::AndIOp::create(
      builder, loc, arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt, m, zero),
      arith::AndIOp::create(
          builder, loc, arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt, n, zero),
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt, k, zero)));
  for (int64_t item = 0; item < 4; ++item) {
    Value linear = addShape(builder, loc, tid, shapeConstant(builder, loc, item * 32));
    Value row = addShape(builder, loc, mulShape(builder, loc, blockY, shapeConstant(builder, loc, 16)),
        arith::DivUIOp::create(builder, loc, linear, shapeConstant(builder, loc, 8)));
    Value col = addShape(builder, loc, mulShape(builder, loc, blockX, shapeConstant(builder, loc, 8)),
        arith::RemUIOp::create(builder, loc, linear, shapeConstant(builder, loc, 8)));
    Value valid = arith::AndIOp::create(
        builder, loc, arith::AndIOp::create(builder, loc, inWarp, positive),
        arith::AndIOp::create(builder, loc, lessShape(builder, loc, row, m), lessShape(builder, loc, col, n)));
    auto guarded = scf::IfOp::create(builder, loc, valid, false);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&guarded.getThenRegion().front());
      Value initial = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
      auto loop = scf::ForOp::create(builder, loc, zero, k, one, ValueRange{initial});
      {
        OpBuilder::InsertionGuard loopGuard(builder);
        builder.setInsertionPointToStart(loop.getBody());
        Value index = loop.getInductionVar();
        Value aIndex = transposeA ? addShape(builder, loc, mulShape(builder, loc, index, m), row)
                                 : addShape(builder, loc, mulShape(builder, loc, row, k), index);
        Value bIndex = transposeB ? addShape(builder, loc, mulShape(builder, loc, col, k), index)
                                 : addShape(builder, loc, mulShape(builder, loc, index, n), col);
        auto load = [&](Value base, Value offset, Type type) -> Value {
          Value value;
          if (isa<MemRefType>(base.getType()))
            value = memref::LoadOp::create(builder, loc, base, ValueRange{offset});
          else
            value = LLVM::LoadOp::create(
                builder, loc, type,
                LLVM::GEPOp::create(builder, loc, base.getType(), type, base, ValueRange{offset}),
                type.isF32() ? 4 : 2);
          return type.isF32() ? value : Value(arith::ExtFOp::create(builder, loc, f32, value));
        };
        Value result = math::FmaOp::create(builder, loc,
            load(inputs[0], aIndex, aType), load(inputs[1], bIndex, bType),
            loop.getRegionIterArgs().front());
        scf::YieldOp::create(builder, loc, result);
      }
      builder.setInsertionPointAfter(loop);
      Value result = loop.getResult(0);
      if (!outputType.isF32())
        result = arith::TruncFOp::create(builder, loc, outputType, result);
      Value outputIndex = addShape(builder, loc, mulShape(builder, loc, row, n), col);
      if (isa<MemRefType>(inputs[2].getType()))
        memref::StoreOp::create(builder, loc, result, inputs[2], ValueRange{outputIndex});
      else
        LLVM::StoreOp::create(builder, loc, result,
            LLVM::GEPOp::create(builder, loc, inputs[2].getType(), outputType, inputs[2],
                ValueRange{outputIndex}), outputType.isF32() ? 4 : 2);
    }
    builder.setInsertionPointAfter(guarded);
  }
  return success();
}
} // namespace tessera::scalar_contraction
