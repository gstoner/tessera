//===- LegalizeSpaceTime.cpp - normalize + validate space-time constructs -===//
//
// First pass of the `-tpp-space-time` pipeline.  Previously a no-op; now it
// normalizes the metadata every later pass relies on and enforces the
// normative verifiers from the TPP syntax spec (section 6):
//
//   * Differential/stencil ops (tpp.grad, tpp.stencil.apply, tpp.div):
//     - default a missing `scheme` to "central" and a missing `order` to 2,
//     - validate `scheme` in {central, upwind, weno, eno},
//     - tag `tpp.scheme.normalized` so halo-infer/codegen read consistent
//       metadata (halo-infer derives the radius from `order`).
//
//   * Temporal ops (tpp.time.step):
//     - validate `scheme` in {RK2, RK4, semi_lagrangian, BDF2},
//     - derive `tpp.time.stages` (RK2->2, RK4->4, semi_lagrangian->1,
//       BDF2->2) and `tpp.time.order`, and carry `dt` onto `tpp.time.dt`,
//       so fuse-stencil-time / codegen know the temporal structure.
//
// Unknown schemes are a hard error (the pass fails), matching the spec's
// "incompatible ... are illegal" intent.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {

// Spatial scheme -> valid?  central|upwind|weno|eno.
static bool validSpatialScheme(StringRef s) {
  return s == "central" || s == "upwind" || s == "weno" || s == "eno";
}

// Temporal scheme -> (stages, order), or {-1,-1} if unknown.
static std::pair<int64_t, int64_t> timeSchemeInfo(StringRef s) {
  return llvm::StringSwitch<std::pair<int64_t, int64_t>>(s)
      .Case("RK2", {2, 2})
      .Case("RK4", {4, 4})
      .Case("semi_lagrangian", {1, 1})
      .Case("BDF2", {2, 2})
      .Default({-1, -1});
}

struct LegalizeSpaceTime
    : public PassWrapper<LegalizeSpaceTime, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-legalize-space-time"; }
  StringRef getDescription() const final {
    return "Normalize + validate TPP space-time metadata (stencil scheme/order,"
           " time-step scheme/stages) before the rest of the pipeline";
  }

  void runOnOperation() final {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());
    bool failed = false;

    m.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      bool isStencil = name.ends_with("tpp.grad") ||
                       name.ends_with("tpp.stencil.apply") ||
                       name.ends_with("tpp.div");
      bool isTimeStep = name.ends_with("tpp.time.step");

      if (isStencil) {
        // Default + validate the spatial scheme.
        StringRef scheme = "central";
        if (auto s = op->getAttrOfType<StringAttr>("scheme"))
          scheme = s.getValue();
        else
          op->setAttr("scheme", b.getStringAttr(scheme));
        if (!validSpatialScheme(scheme)) {
          op->emitError("tpp: unknown spatial scheme '")
              << scheme << "' (expected central|upwind|weno|eno)";
          failed = true;
          return;
        }
        // Default the accuracy order (halo-infer reads this).
        if (!op->getAttrOfType<IntegerAttr>("order"))
          op->setAttr("order", b.getI64IntegerAttr(2));
        op->setAttr("tpp.scheme.normalized", b.getUnitAttr());
      } else if (isTimeStep) {
        StringRef scheme;
        if (auto s = op->getAttrOfType<StringAttr>("scheme"))
          scheme = s.getValue();
        auto info = timeSchemeInfo(scheme);
        if (info.first < 0) {
          op->emitError("tpp: unknown time scheme '")
              << scheme << "' (expected RK2|RK4|semi_lagrangian|BDF2)";
          failed = true;
          return;
        }
        op->setAttr("tpp.time.scheme", b.getStringAttr(scheme));
        op->setAttr("tpp.time.stages", b.getI64IntegerAttr(info.first));
        op->setAttr("tpp.time.order", b.getI64IntegerAttr(info.second));
        if (auto dt = op->getAttrOfType<FloatAttr>("dt"))
          op->setAttr("tpp.time.dt", dt);
      }
    });

    if (failed) {
      signalPassFailure();
      return;
    }
    m->setAttr("tpp.legalized", b.getUnitAttr());
  }
};

} // namespace

std::unique_ptr<Pass> createLegalizeSpaceTimePass() {
  return std::make_unique<LegalizeSpaceTime>();
}
