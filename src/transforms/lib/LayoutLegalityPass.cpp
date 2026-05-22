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

  // Sprint V4a (2026-05-22) — producer/consumer accept-set rule.
  //
  // For canonical compute ops we know which layouts the kernel
  // actually consumes.  When a consumer's operand carries a
  // `tessera.layout` attribute that isn't in the consumer's
  // accept-set AND no `tessera.cast` op intervenes to convert it,
  // emit `LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH`.
  //
  // V4a covers `tessera.matmul` whose lhs/rhs must each be
  // row_major or col_major (the FA-4 kernels and x86 AMX both
  // accept these).  Future rules can extend per consumer.
  static const llvm::StringSet<> &matmulAcceptSet() {
    static const llvm::StringSet<> kSet = {"row_major", "col_major"};
    return kSet;
  }

  static LogicalResult checkMatmulOperandLayouts(Operation *op) {
    if (op->getName().getStringRef() != "tessera.matmul") return success();
    if (op->getNumOperands() < 2) return success();
    // Reach back through the def-use chain to find a `tessera.layout`
    // attribute on the producer.  V4a only walks 1 step (the
    // immediate producer); future rules can widen.
    auto operandLayout = [](Value v) -> StringRef {
      Operation *prod = v.getDefiningOp();
      if (!prod) return StringRef();
      auto attr = prod->getAttrOfType<StringAttr>("tessera.layout");
      return attr ? attr.getValue() : StringRef();
    };
    bool failed = false;
    const char *names[] = {"lhs", "rhs"};
    for (int i = 0; i < 2; ++i) {
      StringRef layout = operandLayout(op->getOperand(i));
      if (layout.empty()) continue;  // no layout attr → no enforcement
      if (matmulAcceptSet().contains(layout)) continue;
      // The producer's layout attribute is the layout that reaches
      // matmul — regardless of whether the producer is a cast op or
      // any other op.  matmul rejects.  The user must insert another
      // cast that converts to row_major or col_major.
      op->emitOpError(
          "LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH: tessera.matmul "
          "operand '")
          << names[i] << "' has layout \"" << layout
          << "\" but matmul's accept-set is {row_major, col_major}. "
          << "Insert a `tessera.cast` to convert.";
      failed = true;
    }
    return failed ? failure() : success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool anyError = false;
    module.walk([&](Operation *op) {
      if (failed(checkCastLayout(op))) anyError = true;
      // Sprint V4a: producer/consumer accept-set rule for matmul.
      if (failed(checkMatmulOperandLayouts(op))) anyError = true;
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
