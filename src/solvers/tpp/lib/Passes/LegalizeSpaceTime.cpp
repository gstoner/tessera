//===- LegalizeSpaceTime.cpp - normalize + validate space-time constructs -===//
//
// First pass of the `-tpp-space-time` pipeline.  Previously a no-op; now it
// normalizes the metadata every later pass relies on and enforces the
// normative verifiers from the TPP syntax spec (section 6):
//
//   * Differential/stencil ops (tpp.grad, tpp.stencil.apply, tpp.div):
//     - require explicit `scheme`, `order`, and per-axis positive `spacing`,
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

#include <cmath>

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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeSpaceTime)

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
        // These attributes change the mathematical operator. Missing values
        // are never defaults: inventing a centered unit-grid scheme can turn
        // a valid PDE into an unstable or dimensionally incorrect program.
        auto schemeAttr = op->getAttrOfType<StringAttr>("scheme");
        if (!schemeAttr) {
          op->emitError("tpp: stencil operation requires explicit 'scheme'");
          failed = true;
          return;
        }
        StringRef scheme = schemeAttr.getValue();
        if (!validSpatialScheme(scheme)) {
          op->emitError("tpp: unknown spatial scheme '")
              << scheme << "' (expected central|upwind|weno|eno)";
          failed = true;
          return;
        }
        auto order = op->getAttrOfType<IntegerAttr>("order");
        if (!order || order.getInt() <= 0 || (order.getInt() & 1)) {
          op->emitError("tpp: stencil operation requires a positive even "
                        "'order'");
          failed = true;
          return;
        }

        auto spacing = op->getAttrOfType<ArrayAttr>("spacing");
        if (!spacing) {
          op->emitError("tpp: stencil operation requires explicit per-axis "
                        "'spacing'");
          failed = true;
          return;
        }
        int64_t rank = -1;
        ShapedType fieldType;
        if (op->getNumOperands() > 0)
          if (auto shaped = dyn_cast<ShapedType>(op->getOperand(0).getType())) {
            fieldType = shaped;
            if (shaped.hasRank())
              rank = shaped.getRank();
          }
        if (!fieldType || !isa<FloatType>(fieldType.getElementType())) {
          op->emitError("tpp: stencil field must have a floating-point "
                        "element dtype");
          failed = true;
          return;
        }
        if (rank <= 0 || static_cast<int64_t>(spacing.size()) != rank) {
          op->emitError("tpp: 'spacing' length must match the ranked field; got ")
              << spacing.size() << " values for rank " << rank;
          failed = true;
          return;
        }
        for (Attribute raw : spacing) {
          auto value = dyn_cast<FloatAttr>(raw);
          if (!value) {
            op->emitError("tpp: every spacing entry must be an f64 value");
            failed = true;
            return;
          }
          double h = value.getValueAsDouble();
          if (!std::isfinite(h) || h <= 0.0) {
            op->emitError("tpp: every spacing value must be finite and > 0");
            failed = true;
            return;
          }
        }
        if (name.ends_with("tpp.stencil.apply")) {
          if (op->getNumOperands() < 2) {
            op->emitError("tpp: stencil.apply requires a coefficient tensor");
            failed = true;
            return;
          }
          auto kernel = dyn_cast<ShapedType>(op->getOperand(1).getType());
          if (!kernel || !kernel.hasRank() ||
              !isa<FloatType>(kernel.getElementType())) {
            op->emitError("tpp: stencil coefficient tensor must be a ranked "
                          "floating-point tensor");
            failed = true;
            return;
          }
          if (kernel.getRank() != rank) {
            op->emitError("tpp: stencil coefficient tensor rank must match "
                          "the field rank");
            failed = true;
            return;
          }
          if (kernel.getElementType() != fieldType.getElementType()) {
            op->emitError("tpp: stencil coefficient dtype must match the "
                          "field element dtype");
            failed = true;
            return;
          }
        }
        if (auto axis = op->getAttrOfType<IntegerAttr>("axis")) {
          if (axis.getInt() < 0 || axis.getInt() >= rank) {
            op->emitError("tpp: gradient 'axis' is outside the field rank");
            failed = true;
            return;
          }
        }
        op->setAttr("tpp.scheme.normalized", b.getUnitAttr());
        op->setAttr("tpp.spacing.validated", b.getUnitAttr());
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
