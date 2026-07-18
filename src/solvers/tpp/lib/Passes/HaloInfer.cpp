//===- HaloInfer.cpp - infer halo widths from stencil access patterns -----===//
//
// Derives the ghost-cell (halo) width each stencil-like op needs and records
// it as a structured attribute the distribution pass consumes.  This replaces
// the v0.2 placeholder (a hardcoded "1,1,0" string) with a real inference
// from the op's finite-difference / stencil access pattern:
//
//   tpp.grad           A finite-difference gradient.  A central scheme of
//                      accuracy `order` (default 2) reaches +/- order/2 cells,
//                      so the halo radius = order/2.  If an integer `axis`
//                      attribute is present the derivative is directional and
//                      only that dimension gets a halo; otherwise every
//                      spatial dimension does.
//
//   tpp.stencil.apply  Halo = stencil radius.  Taken from an explicit integer
//                      `radius` attribute if present; otherwise read directly
//                      from the *kernel* operand's shape — a kernel of extent
//                      E along a dim reaches (E-1)/2 cells, which is exactly
//                      the halo that dimension needs.
//
// Emitted per op:
//   tpp.halo         : i64 array, per-spatial-dim halo width (machine form the
//                      distribute pass reads).
//   tpp.halo.width   : i64, max over dims (convenience for uniform exchanges).
//   tpp.halo.inferred: unit marker so downstream passes / the audit can tell
//                      an inferred halo from a user-supplied one.
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

// Spatial rank of an op's primary (operand-0) field.  -1 if not ranked.
static int64_t spatialRank(Operation *op) {
  if (op->getNumOperands() == 0)
    return -1;
  auto shaped = dyn_cast<ShapedType>(op->getOperand(0).getType());
  if (!shaped || !shaped.hasRank())
    return -1;
  return shaped.getRank();
}

struct HaloInfer : public PassWrapper<HaloInfer, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HaloInfer)

  StringRef getArgument() const final { return "tpp-halo-infer"; }
  StringRef getDescription() const final {
    return "Infer per-dimension halo widths from tpp.grad / tpp.stencil.apply "
           "access patterns";
  }

  // Fill `halo` (length = rank) for a gradient op.
  static void gradHalo(Operation *op, int64_t rank,
                       SmallVectorImpl<int64_t> &halo) {
    int64_t order = 2;
    if (auto o = op->getAttrOfType<IntegerAttr>("order"))
      order = o.getInt();
    int64_t radius = std::max<int64_t>(1, order / 2);
    halo.assign(rank, 0);
    if (auto axis = op->getAttrOfType<IntegerAttr>("axis")) {
      int64_t a = axis.getInt();
      if (a >= 0 && a < rank)
        halo[a] = radius; // directional derivative: one axis only
    } else {
      halo.assign(rank, radius); // gradient in every spatial direction
    }
  }

  // Fill `halo` for a stencil.apply op, preferring an explicit radius, then
  // the kernel operand's shape.
  static void stencilHalo(Operation *op, int64_t rank,
                          SmallVectorImpl<int64_t> &halo) {
    halo.assign(rank, 0);
    if (auto r = op->getAttrOfType<IntegerAttr>("radius")) {
      halo.assign(rank, std::max<int64_t>(0, r.getInt()));
      return;
    }
    // Derive per-dim radius from the kernel operand's static shape.
    if (op->getNumOperands() >= 2) {
      if (auto k = dyn_cast<ShapedType>(op->getOperand(1).getType())) {
        if (k.hasRank()) {
          for (int64_t d = 0; d < rank; ++d) {
            int64_t kd = (d < k.getRank()) ? k.getDimSize(d) : 1;
            halo[d] = (!ShapedType::isDynamic(kd) && kd > 0) ? (kd - 1) / 2 : 1;
          }
          return;
        }
      }
    }
    halo.assign(rank, 1); // conservative default
  }

  void runOnOperation() final {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    m.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      bool isGrad = name.ends_with("tpp.grad");
      bool isStencil = name.ends_with("tpp.stencil.apply");
      if (!isGrad && !isStencil)
        return;
      if (op->hasAttr("tpp.halo")) // respect a user-supplied halo
        return;
      int64_t rank = spatialRank(op);
      if (rank <= 0)
        return; // cannot infer without a ranked field

      SmallVector<int64_t, 4> halo;
      if (isGrad)
        gradHalo(op, rank, halo);
      else
        stencilHalo(op, rank, halo);

      int64_t maxW = 0;
      for (int64_t h : halo)
        maxW = std::max(maxW, h);

      op->setAttr("tpp.halo", b.getI64ArrayAttr(halo));
      op->setAttr("tpp.halo.width", b.getI64IntegerAttr(maxW));
      op->setAttr("tpp.halo.inferred", b.getUnitAttr());
    });
  }
};

} // namespace

std::unique_ptr<Pass> createHaloInferPass() {
  return std::make_unique<HaloInfer>();
}
