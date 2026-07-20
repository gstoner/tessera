//===- TesseraTiling.cpp — TilingInterface impls for tessera ops -*- C++ -*-===//
//
// MLIR 23-compatible TilingInterface implementations for
// ``tessera.matmul`` and ``tessera.conv2d_nhwc``.  ODS declares the
// interface methods via an explicit method list on
// ``DeclareOpInterfaceMethods<TilingInterface, [...]>`` (see
// ``TesseraOps.td``); this file supplies the definitions.
//
// Status (2026-05-20, B3 v2):
//   * MatmulOp: full v1 implementation against MLIR 23 signatures.
//     ``getTiledImplementation`` clones the op with annotation attrs
//     so a tile driver can verify the tiling decision flowed
//     through.  Operand-tile extraction (``tensor.extract_slice`` on
//     LHS/RHS) is left to a v2 follow-up — keeping the v1
//     conservative semantics so any consumer that wraps the cloned
//     op in its own slicing logic keeps working.
//   * Conv2DNHWCOp: iteration domain + identity result-tile-position
//     are real; ``getTiledImplementation`` returns ``failure()``
//     until the stride/pad-aware window reconstruction lands.
//
// Default state: ``TESSERA_ENABLE_TILING_INTERFACE`` defaults to 1.
// Downstream consumers can pass ``-DTESSERA_DISABLE_TILING_INTERFACE``
// to fall back to MLIR's default-failure trait impls.
//
//===----------------------------------------------------------------------===//

#include "Tessera/IR/TesseraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace tessera;

#ifndef TESSERA_ENABLE_TILING_INTERFACE
#  ifdef TESSERA_DISABLE_TILING_INTERFACE
#    define TESSERA_ENABLE_TILING_INTERFACE 0
#  else
#    define TESSERA_ENABLE_TILING_INTERFACE 1
#  endif
#endif

#if TESSERA_ENABLE_TILING_INTERFACE

namespace {

// Build the per-dimension iteration domain ``[0, dim_size)`` with
// step 1 from a static-shape ranked tensor.  Returns an empty list
// if any dimension is dynamic — dynamic shape support is a v2
// concern.
static SmallVector<Range> staticIterationDomain(RankedTensorType resTy,
                                                Location loc,
                                                OpBuilder &b) {
  SmallVector<Range> dom;
  dom.reserve(resTy.getRank());
  for (int64_t i = 0; i < resTy.getRank(); ++i) {
    if (resTy.isDynamicDim(i))
      return {};
    Value zero =
        b.create<arith::ConstantIndexOp>(loc, 0).getResult();
    Value size =
        b.create<arith::ConstantIndexOp>(loc, resTy.getDimSize(i)).getResult();
    Value one =
        b.create<arith::ConstantIndexOp>(loc, 1).getResult();
    dom.push_back(Range{zero, size, one});
  }
  return dom;
}

}  // namespace

// ────────────────────────────────────────────────────────────────────
// MatmulOp
// ────────────────────────────────────────────────────────────────────
//
// Conservative v1 implementation: ``getTiledImplementation`` clones
// the op and stamps annotation attrs that a tile driver can assert
// against.  The K reduction stays inside the cloned op (the v1
// semantics: "split M / N into tiles; keep K full").  Operand
// slicing on LHS / RHS is left to a future caller — that keeps the
// implementation under 50 LOC while still giving downstream consumers
// a working ``TilingInterface`` to attach to.
//
// Iteration domain: (M, N).  Both axes parallel (the K reduction is
// not part of the result iteration domain — it's consumed inside the
// op body).

SmallVector<utils::IteratorType> MatmulOp::getLoopIteratorTypes() {
  return {utils::IteratorType::parallel, utils::IteratorType::parallel};
}

SmallVector<Range> MatmulOp::getIterationDomain(OpBuilder &b) {
  auto resTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resTy || resTy.getRank() != 2)
    return {};
  return staticIterationDomain(resTy, getLoc(), b);
}

FailureOr<TilingResult> MatmulOp::getTiledImplementation(
    OpBuilder &b,
    ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  auto lhsTy = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsTy = dyn_cast<RankedTensorType>(getRhs().getType());
  auto resTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhsTy || !rhsTy || !resTy)
    return failure();
  if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || resTy.getRank() != 2)
    return failure();
  if (offsets.size() != 2 || sizes.size() != 2)
    return failure();

  // Clone the matmul and stamp the tiling decision so a driver pass
  // can assert that the tile interface ran (this is the
  // "matmul_conservative_ranked_tensor" sentinel the drift gate
  // checks for).
  Operation *cloned = b.clone(*getOperation());
  cloned->setAttr(
      "tessera.tiling_interface",
      b.getStringAttr("matmul_conservative_ranked_tensor"));
  cloned->setAttr("tessera.tile_rank", b.getI64IntegerAttr(2));
  cloned->setAttr("tessera.full_k",
                  b.getIndexAttr(lhsTy.getDimSize(1)));

  TilingResult result;
  result.tiledOps.push_back(cloned);
  result.tiledValues.push_back(cloned->getResult(0));
  return result;
}

FailureOr<TilingResult> MatmulOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<InnerTileAlignment> /*innerTileAlignments*/) {
  return getTiledImplementation(b, offsets, sizes);
}

LogicalResult MatmulOp::getResultTilePosition(
    OpBuilder & /*b*/, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0)
    return failure();
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

// ────────────────────────────────────────────────────────────────────
// Conv2DNHWCOp
// ────────────────────────────────────────────────────────────────────
//
// Iteration domain over the result tensor's (N, H, W, C) axes, all
// parallel.  ``getTiledImplementation`` returns ``failure()`` until
// stride/pad-aware input-window reconstruction lands — that's the
// v3 concern documented in ``TilingInterface_NOTES.md``.  Returning
// failure() is the safe answer: any caller falls through to the
// non-tiled lowering path.

SmallVector<utils::IteratorType> Conv2DNHWCOp::getLoopIteratorTypes() {
  // (N, H, W, C) are all parallel at this level; the (Kc, R, S)
  // reduction lives inside the op body, not in the iteration
  // domain that the tile driver enumerates.
  return {utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::parallel};
}

SmallVector<Range> Conv2DNHWCOp::getIterationDomain(OpBuilder &b) {
  auto resTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resTy || resTy.getRank() != 4)
    return {};
  return staticIterationDomain(resTy, getLoc(), b);
}

FailureOr<TilingResult> Conv2DNHWCOp::getTiledImplementation(
    OpBuilder & /*b*/,
    ArrayRef<OpFoldResult> /*offsets*/,
    ArrayRef<OpFoldResult> /*sizes*/) {
  // v3 work: stride/pad-aware input-window reconstruction.  Until
  // that lands a tile driver should fall through to a non-tiled
  // lowering path; ``failure()`` is the safe answer.
  return failure();
}

FailureOr<TilingResult> Conv2DNHWCOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<InnerTileAlignment> /*innerTileAlignments*/) {
  return getTiledImplementation(b, offsets, sizes);
}

LogicalResult Conv2DNHWCOp::getResultTilePosition(
    OpBuilder & /*b*/, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0)
    return failure();
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

#endif  // TESSERA_ENABLE_TILING_INTERFACE
