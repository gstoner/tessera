#pragma once
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include <climits>
#include <cstdint>

namespace tessera {
struct BoundedQueryNormShape {
  bool bounded = false;
  int64_t sequenceBound = 0;
  int64_t rowCapacity = 0;
  int64_t fixedColumns = 0;
  int64_t rowMultiple = 0;
};

// An all-or-none shape contract shared by Schedule, Tile and native lowering.
// Static operations carry none of these fields and retain their old recipes.
inline mlir::FailureOr<BoundedQueryNormShape> readBoundedQueryNormShape(
    mlir::Operation *op, mlir::Type logicalType = {}) {
  using namespace mlir;
  BoundedQueryNormShape shape;
  bool requested = false;
  for (StringRef key : {"shape_policy", "sequence_bound", "row_capacity",
                        "fixed_columns", "row_multiple"})
    requested |= op->hasAttr(key);
  if (!requested) return shape;
  auto reject = [&]() -> FailureOr<BoundedQueryNormShape> {
    return op->emitOpError("requires the complete SM120 half bounded-query RMSNorm capacity contract"), failure();
  };
  auto policy = op->getAttrOfType<StringAttr>("shape_policy");
  auto integer = [&](StringRef key) -> int64_t {
    auto value = op->getAttrOfType<IntegerAttr>(key);
    return value && value.getType().isInteger(64) ? value.getInt() : -1;
  };
  shape.bounded = true;
  shape.sequenceBound = integer("sequence_bound");
  shape.rowCapacity = integer("row_capacity");
  shape.fixedColumns = integer("fixed_columns");
  shape.rowMultiple = integer("row_multiple");
  auto module = op->getParentOfType<ModuleOp>();
  auto arch = module ? module->getAttrOfType<StringAttr>("tessera.arch") : StringAttr{};
  auto target = module ? module->getAttrOfType<StringAttr>("tessera.target") : StringAttr{};
  auto ownArch = op->getAttrOfType<StringAttr>("arch");
  auto storage = op->getAttrOfType<StringAttr>("storage");
  auto kind = op->getAttrOfType<StringAttr>("kind");
  auto accum = op->getAttrOfType<StringAttr>("accum");
  if (!policy || policy.getValue() != "bounded_query_v1" || !arch ||
      arch.getValue() != "sm_120" || (target && target.getValue() != "nvidia_sm120") ||
      (ownArch && ownArch.getValue() != "sm_120") || !storage ||
      (storage.getValue() != "f16" && storage.getValue() != "bf16") ||
      !kind || kind.getValue() != "rmsnorm" || !accum || accum.getValue() != "f32" ||
      shape.sequenceBound <= 0 || shape.sequenceBound > INT32_MAX ||
      shape.rowCapacity <= 0 || shape.rowCapacity > INT32_MAX ||
      shape.fixedColumns <= 0 || shape.fixedColumns > INT32_MAX ||
      shape.rowMultiple <= 0 || shape.rowMultiple > shape.rowCapacity ||
      shape.rowCapacity % shape.rowMultiple != 0 ||
      shape.rowCapacity / shape.rowMultiple != shape.sequenceBound ||
      shape.rowCapacity > INT64_MAX / 2 / shape.fixedColumns)
    return reject();
  if (logicalType) {
    auto type = dyn_cast<RankedTensorType>(logicalType);
    if (!type || type.getEncoding() || type.getRank() != 4 ||
        !type.isDynamicDim(2) || type.getDimSize(0) <= 0 ||
        type.getDimSize(0) > INT32_MAX || type.getDimSize(1) <= 0 ||
        type.getDimSize(1) > INT32_MAX / type.getDimSize(0) ||
        type.getDimSize(0) * type.getDimSize(1) != shape.rowMultiple ||
        type.getDimSize(3) != shape.fixedColumns ||
        (storage.getValue() == "f16" ? !type.getElementType().isF16()
                                      : !type.getElementType().isBF16()))
      return reject();
  }
  return shape;
}
inline void addBoundedQueryNormShape(mlir::OperationState &state, mlir::Builder &b,
                                    const BoundedQueryNormShape &shape) {
  if (!shape.bounded) return;
  state.addAttribute("shape_policy", b.getStringAttr("bounded_query_v1"));
  state.addAttribute("sequence_bound", b.getI64IntegerAttr(shape.sequenceBound));
  state.addAttribute("row_capacity", b.getI64IntegerAttr(shape.rowCapacity));
  state.addAttribute("fixed_columns", b.getI64IntegerAttr(shape.fixedColumns));
  state.addAttribute("row_multiple", b.getI64IntegerAttr(shape.rowMultiple));
}
} // namespace tessera
