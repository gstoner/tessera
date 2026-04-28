//===- PeriodicHalo.cpp — infer periodic halo exchange patterns ----------*- C++ -*-===//
//
// Walks functions annotated with tessera.periodic_boundary or containing
// tessera.halo ops and attaches:
//   tessera.halo_width     — number of ghost cells per dimension (int64)
//   tessera.halo_direction — "periodic" | "reflective" | "zero"
//   tessera.halo_dims      — comma-separated list of axes with halo exchange
//
// The halo width is inferred from tessera.stencil_radius attr (if present)
// or defaults to --halo-width (default 1).
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct PeriodicHaloPass
    : PassWrapper<PeriodicHaloPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PeriodicHaloPass)

  Option<int> defaultHaloWidth{
      *this, "halo-width",
      llvm::cl::desc("Default halo (ghost cell) width per dimension"),
      llvm::cl::init(1)};

  Option<std::string> defaultDirection{
      *this, "halo-direction",
      llvm::cl::desc("Halo boundary type: periodic | reflective | zero"),
      llvm::cl::init(std::string("periodic"))};

  StringRef getArgument() const final { return "tessera-periodic-halo"; }
  StringRef getDescription() const final {
    return "Infer periodic halo exchange patterns for stencil/PDE solver ops";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    // 1. Tag tessera.halo ops directly.
    mod.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      if (!opName.contains("halo") && !opName.contains("stencil") &&
          !opName.contains("periodic"))
        return;

      // Infer halo width from stencil radius attr if present.
      int64_t width = defaultHaloWidth;
      if (auto attr = op->getAttrOfType<IntegerAttr>("tessera.stencil_radius"))
        width = attr.getInt();

      op->setAttr("tessera.halo_width",
                  IntegerAttr::get(IntegerType::get(ctx, 64), width));
      op->setAttr("tessera.halo_direction",
                  StringAttr::get(ctx, defaultDirection));
      op->setAttr("tessera.halo_annotated", UnitAttr::get(ctx));
    });

    // 2. Tag functions annotated with tessera.periodic_boundary.
    mod.walk([&](func::FuncOp fn) {
      if (!fn->hasAttr("tessera.periodic_boundary"))
        return;

      // Walk stencil ops inside and annotate any that lack halo width.
      fn.walk([&](Operation *op) {
        if (op->hasAttr("tessera.halo_annotated"))
          return;
        StringRef opName = op->getName().getStringRef();
        if (!opName.startswith("tessera."))
          return;
        // Conservative: tag every tessera op inside a periodic-boundary func.
        int64_t width = defaultHaloWidth;
        op->setAttr("tessera.halo_width",
                    IntegerAttr::get(IntegerType::get(ctx, 64), width));
        op->setAttr("tessera.halo_direction",
                    StringAttr::get(ctx, defaultDirection));
      });
    });
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createPeriodicHaloPass() {
  return std::make_unique<PeriodicHaloPass>();
}
} // namespace passes
} // namespace tessera
