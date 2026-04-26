
#include "Tessera/IR/TesseraOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace tessera;

// Define simple, conservative TilingInterface models.
// Guard with a macro so you can enable/iterate without breaking builds.
#ifndef TESSERA_ENABLE_TILING_INTERFACE
#define TESSERA_ENABLE_TILING_INTERFACE 0
#endif

namespace {

static SmallVector<Range> get2DItrDomainFromResult(RankedTensorType resTy, Location loc, OpBuilder &b) {
  auto c0 = b.getIndexAttr(0);
  auto s0 = b.getIndexAttr(resTy.getDimSize(0));
  auto s1 = b.getIndexAttr(resTy.getDimSize(1));
  return {
    Range{b.create<arith::ConstantIndexOp>(loc, 0), b.create<arith::ConstantIndexOp>(loc, resTy.getDimSize(0)), b.create<arith::ConstantIndexOp>(loc, 1)},
    Range{b.create<arith::ConstantIndexOp>(loc, 0), b.create<arith::ConstantIndexOp>(loc, resTy.getDimSize(1)), b.create<arith::ConstantIndexOp>(loc, 1)},
  };
}

} // namespace

#if TESSERA_ENABLE_TILING_INTERFACE
// NOTE: The following are placeholders; fill out for true tiling.
::mlir::FailureOr<SmallVector<OpFoldResult>> tessera::MatmulOp::getMixedSizes(OpBuilder &b) {
  (void)b;
  return failure();
}
::mlir::FailureOr<SmallVector<Range>> tessera::MatmulOp::getIterationDomain(OpBuilder &b) {
  auto resTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resTy || resTy.getRank() < 2) return failure();
  return get2DItrDomainFromResult(resTy, getLoc(), b);
}
::mlir::FailureOr<SmallVector<Operation *>> tessera::MatmulOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  return failure(); // TODO: implement with tensor.extract_slice on lhs/rhs and rebuild op
}
::mlir::FailureOr<SmallVector<OpFoldResult>> tessera::MatmulOp::getResultTilePosition(
    OpBuilder &b, unsigned, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  return SmallVector<OpFoldResult>(offsets.begin(), offsets.end());
}

::mlir::FailureOr<SmallVector<Range>> tessera::Conv2DNHWCOp::getIterationDomain(OpBuilder &b) {
  auto resTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resTy || resTy.getRank() < 4) return failure();
  // Tile H,W as iter domain; keep N,C untiled in this simple model.
  Location loc = getLoc();
  SmallVector<Range> dom;
  dom.push_back(Range{b.create<arith::ConstantIndexOp>(loc, 0), b.create<arith::ConstantIndexOp>(loc, resTy.getDimSize(1)), b.create<arith::ConstantIndexOp>(loc, 1)}); // H
  dom.push_back(Range{b.create<arith::ConstantIndexOp>(loc, 0), b.create<arith::ConstantIndexOp>(loc, resTy.getDimSize(2)), b.create<arith::ConstantIndexOp>(loc, 1)}); // W
  return dom;
}
::mlir::FailureOr<SmallVector<Operation *>> tessera::Conv2DNHWCOp::getTiledImplementation(
    OpBuilder &, ArrayRef<OpFoldResult>, ArrayRef<OpFoldResult>) {
  return failure(); // TODO
}
::mlir::FailureOr<SmallVector<OpFoldResult>> tessera::Conv2DNHWCOp::getResultTilePosition(
    OpBuilder &, unsigned, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  return SmallVector<OpFoldResult>(offsets.begin(), offsets.end());
}
#endif
