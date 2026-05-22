// LayoutLegalityPass.cpp — Sprint V2 (2026-05-22)
//
// Closes the "no `LayoutLegalityPass`" item in SHAPE_SYSTEM.md §11.2.
//
// Today this is a SKELETON with **one first rule**: a `tessera.cast`
// op whose `tessera.layout` attribute names a layout not in the
// canonical accept-set emits a diagnostic.  The skeleton is named so
// future rules — producer/consumer accept-set mismatches across the
// dialect — extend a single pass body instead of a one-rule-per-pass
// archipelago.
//
// Canonical layout accept-set (matches SHAPE_SYSTEM.md §2.1):
//
//   row_major        — default for dense tensors
//   col_major        — Fortran ordering / Apple Metal column-major
//   nhwc             — Conv2D default (4D)
//   nchw             — Conv2D Caffe-style (4D)
//   bhsd             — Attention (B, H, S, D) (4D)
//   tile             — generic tile layout (parameters carried in
//                      the attribute payload, not the name)
//   bsr              — block-sparse-row
//   packed           — vector-packed (parameters in payload)
//
// Future rules (placeholders in the source — not implemented):
//
//   LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH
//     Detect a `tessera.matmul` whose lhs/rhs operand layout attr
//     names a layout NOT in matmul's accept-set without an
//     intervening `tessera.cast` or `tessera.layout_cast` op.
//
//   LAYOUT_LEGALITY_FUSION_CASCADE
//     Cascade of layout casts within a single function whose net
//     effect is identity should fold; the pass currently does not
//     fold but flags the cascade.
//
// Pipeline placement: the pass is registered standalone for now via
// `--tessera-layout-legality`.  It will be inserted into the x86 /
// GPU named pipelines in a follow-up sprint once the rule set is
// large enough to justify the per-target ordering question.
//
// Diagnostic codes (stable for SHAPE_SYSTEM.md §11 cross-linking):
//
//   LAYOUT_LEGALITY_UNKNOWN_LAYOUT
//     The `tessera.layout` attribute names a layout not in the
//     canonical accept-set listed above.
//

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

// Canonical layout names accepted today.  Order is informative; the
// set is what matters.  Sprint V2 closes Gap #3 from SHAPE_SYSTEM
// §11.2 by giving these names a single home.
static const llvm::StringSet<> &canonicalLayouts() {
  static const llvm::StringSet<> kSet = {
      "row_major", "col_major",
      "nhwc",      "nchw",
      "bhsd",
      "tile",
      "bsr",
      "packed",
  };
  return kSet;
}

struct LayoutLegality
    : public PassWrapper<LayoutLegality, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutLegality)

  StringRef getArgument() const override {
    return "tessera-layout-legality";
  }
  StringRef getDescription() const override {
    return "Sprint V2 — layout legality pass (skeleton + first rule: "
           "tessera.cast layout attribute must be in the canonical "
           "accept-set).";
  }

  // First rule: tessera.cast with a `tessera.layout` attribute whose
  // string value isn't in the canonical accept-set above.
  static LogicalResult checkCastLayout(Operation *op) {
    if (op->getName().getStringRef() != "tessera.cast")
      return success();
    auto layoutAttr = op->getAttrOfType<StringAttr>("tessera.layout");
    if (!layoutAttr) return success();
    StringRef name = layoutAttr.getValue();
    if (canonicalLayouts().contains(name))
      return success();
    return op->emitOpError(
        "LAYOUT_LEGALITY_UNKNOWN_LAYOUT: tessera.layout=\"")
        << name
        << "\" is not in the canonical accept-set "
           "{row_major, col_major, nhwc, nchw, bhsd, tile, bsr, "
           "packed}.  See SHAPE_SYSTEM.md §2.1.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool anyError = false;
    module.walk([&](Operation *op) {
      if (failed(checkCastLayout(op)))
        anyError = true;
    });
    if (anyError)
      signalPassFailure();
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createLayoutLegalityPass() {
  return std::make_unique<LayoutLegality>();
}
}  // namespace tessera
