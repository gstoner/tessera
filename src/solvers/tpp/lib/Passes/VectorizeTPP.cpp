//===- VectorizeTPP.cpp - choose tile + vector params for stencil ops -----===//
//
// Previously a no-op.  Now derives a concrete vectorization/tiling plan for
// each stencil-like op from the *shape and element type* of its field, and
// records it as attributes a backend lowering consumes:
//
//   tpp.vector_width : i64        SIMD lanes along the innermost (unit-stride)
//                                 dimension: the largest power of two that is
//                                 <= a target lane budget (256-bit SIMD => 8
//                                 f32 / 4 f64 lanes) and <= the innermost
//                                 extent, so it never over-vectorises a short
//                                 row.
//   tpp.tile_shape   : i64 array  per-dim tile: the innermost dim is tiled to a
//                                 multiple of vector_width (<= kRowTile), outer
//                                 dims to <= kOuterTile — the loop-nest blocking
//                                 the codegen emits.
//   tpp.vectorized   : unit       marker.
//
// The op must have a ranked result (operand-0 field); anything else is left
// untouched.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// Lane budget for a 256-bit SIMD register, in elements of `bitwidth`.
static int64_t laneBudget(unsigned bitwidth) {
  if (bitwidth == 0)
    return 4;
  return std::max<int64_t>(1, 256 / bitwidth);
}

// Largest power of two <= n (n >= 1).
static int64_t floorPow2(int64_t n) {
  int64_t p = 1;
  while (p * 2 <= n)
    p *= 2;
  return p;
}

struct VectorizeTPP : public PassWrapper<VectorizeTPP, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeTPP)

  static constexpr int64_t kRowTile = 64;   // innermost blocking cap
  static constexpr int64_t kOuterTile = 8;  // outer-dim blocking cap

  StringRef getArgument() const final { return "tpp-vectorize"; }
  StringRef getDescription() const final {
    return "Derive per-op vector width + tile shape from stencil field shape "
           "and element type";
  }

  void runOnOperation() final {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    m.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      bool isStencilLike = name.ends_with("tpp.grad") ||
                           name.ends_with("tpp.stencil.apply") ||
                           name.ends_with("tpp.halo.exchange");
      if (!isStencilLike || op->hasAttr("tpp.vectorized"))
        return;
      if (op->getNumResults() == 0)
        return;
      auto shaped = dyn_cast<ShapedType>(op->getResult(0).getType());
      if (!shaped || !shaped.hasRank() || shaped.getRank() == 0)
        return;

      int64_t rank = shaped.getRank();
      unsigned bw = shaped.getElementType().getIntOrFloatBitWidth();
      int64_t budget = laneBudget(bw);

      int64_t inner = shaped.getDimSize(rank - 1);
      int64_t vw = ShapedType::isDynamic(inner)
                       ? budget
                       : std::max<int64_t>(1, floorPow2(std::min(inner, budget)));

      SmallVector<int64_t, 4> tiles(rank);
      for (int64_t d = 0; d < rank; ++d) {
        int64_t dim = shaped.getDimSize(d);
        if (d == rank - 1) {
          // Innermost: multiple of vw, capped at kRowTile (or the dim).
          int64_t cap = ShapedType::isDynamic(dim) ? kRowTile
                                                    : std::min(dim, kRowTile);
          tiles[d] = std::max(vw, (cap / vw) * vw);
        } else {
          int64_t cap = ShapedType::isDynamic(dim) ? kOuterTile
                                                   : std::min(dim, kOuterTile);
          tiles[d] = std::max<int64_t>(1, cap);
        }
      }

      op->setAttr("tpp.vector_width", b.getI64IntegerAttr(vw));
      op->setAttr("tpp.tile_shape", b.getI64ArrayAttr(tiles));
      op->setAttr("tpp.vectorized", b.getUnitAttr());
    });
  }
};

} // namespace

std::unique_ptr<Pass> createVectorizeTPPPass() {
  return std::make_unique<VectorizeTPP>();
}
