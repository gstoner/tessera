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
//   LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH
//     A consumer op (matmul / conv2d_nhwc / flash_attn) receives an
//     operand whose producer layout attr is not in the consumer's
//     accept-set, with no intervening cast.
//
//   LAYOUT_LEGALITY_SCALE_WITHOUT_LAYOUT
//     A tessera.grouped_gemm / tessera.moe_swiglu_block carries a
//     low-precision scale *operand* but no `scale_layout` attribute —
//     an untyped scale tensor has no layout contract (DeepGEMM
//     keystone, 2026-06).
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
    return "Layout legality pass — cast-layout accept-set + producer/consumer "
           "accept-set rule for tessera.matmul/batched_gemm "
           "(row_major/col_major), "
           "tessera.conv2d_nhwc (nhwc, data operand), tessera.flash_attn "
           "(bhsd, Q/K/V), last-axis tessera.reduce (row_major), and "
           "scale-layout legality for tessera.grouped_gemm / "
           "tessera.moe_swiglu_block (scale operand requires scale_layout).";
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
    StringRef opName = op->getName().getStringRef();
    if (opName != "tessera.matmul" && opName != "tessera.batched_gemm")
      return success();
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
    StringRef acceptSetOwner =
        opName == "tessera.matmul" ? "matmul's" : "batched_gemm's";
    for (int i = 0; i < 2; ++i) {
      StringRef layout = operandLayout(op->getOperand(i));
      if (layout.empty()) continue;  // no layout attr → no enforcement
      if (matmulAcceptSet().contains(layout)) continue;
      // The producer's layout attribute is the layout that reaches
      // matmul — regardless of whether the producer is a cast op or
      // any other op.  matmul rejects.  The user must insert another
      // cast that converts to row_major or col_major.
      op->emitOpError(
          "LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH: ")
          << opName << " operand '"
          << names[i] << "' has layout \"" << layout
          << "\" but " << acceptSetOwner
          << " accept-set is {row_major, col_major}. "
          << "Insert a `tessera.cast` to convert.";
      failed = true;
    }
    return failed ? failure() : success();
  }

  // Extended producer/consumer accept-set rule (2026-06-11) — the same rule
  // matmul gets above, generalized to more consumer ops with per-op accept-sets.
  // Unlike matmul (whose data/weight operands share {row_major, col_major}),
  // these ops carry the layout contract on *specific* operands:
  //   tessera.conv2d_nhwc — only the data operand (#0) is NHWC (the filter is a
  //                         separate weight layout, not checked here);
  //   tessera.flash_attn  — Q/K/V (operands #0..2) are all (B,H,S,D) = bhsd.
  // Adding a consumer is a small edit here; the rule is otherwise identical.
  static LogicalResult checkOneOperand(Operation *op, unsigned idx,
                                       const llvm::StringSet<> &accept,
                                       StringRef human) {
    if (idx >= op->getNumOperands()) return success();
    Operation *prod = op->getOperand(idx).getDefiningOp();
    if (!prod) return success();
    auto attr = prod->getAttrOfType<StringAttr>("tessera.layout");
    if (!attr) return success();  // no layout attr → no enforcement
    StringRef layout = attr.getValue();
    if (accept.contains(layout)) return success();
    return op->emitOpError(
        "LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH: ")
        << op->getName().getStringRef() << " operand #" << idx
        << " has layout \"" << layout << "\" but its accept-set is " << human
        << ".  Insert a `tessera.cast` to convert.";
  }

  static LogicalResult checkTensorOpLayouts(Operation *op) {
    StringRef opName = op->getName().getStringRef();
    static const llvm::StringSet<> kConv = {"nhwc"};
    static const llvm::StringSet<> kAttn = {"bhsd"};
    static const llvm::StringSet<> kRowMajor = {"row_major"};
    bool anyFail = false;
    if (opName == "tessera.conv2d_nhwc") {
      if (failed(checkOneOperand(op, 0, kConv, "{nhwc}"))) anyFail = true;
    } else if (opName == "tessera.flash_attn") {
      for (unsigned i = 0; i < 3; ++i)
        if (failed(checkOneOperand(op, i, kAttn, "{bhsd}"))) anyFail = true;
    } else if (opName == "tessera.reduce" && op->getNumOperands() == 1) {
      auto inputTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      if (inputTy && axis && inputTy.getRank() > 0) {
        int64_t normalized = axis.getInt();
        if (normalized < 0)
          normalized += inputTy.getRank();
        if (normalized == inputTy.getRank() - 1 &&
            failed(checkOneOperand(op, 0, kRowMajor, "{row_major}")))
          anyFail = true;
      }
    }
    return anyFail ? failure() : success();
  }

  // Scale-layout legality for grouped GEMM / MoE (2026-06 — DeepGEMM keystone).
  //
  // A low-precision scale *operand* (the optional x_scale/w_scale on
  // tessera.grouped_gemm, or the four optional scales on
  // tessera.moe_swiglu_block) is only legal when the op also declares a
  // `scale_layout` attribute describing that operand's packed layout
  // (granularity / block / packing / transposed).  An untyped scale tensor has
  // no compiler-visible layout contract, so the target-lowering layer would have
  // nothing to legalize against.
  //
  // NOTE on scope: this pass *enforces the invariant*; it does not insert the
  // repack/transpose/align op that converts a declared scale layout into a
  // target's wanted layout.  That rewrite belongs in the Tile->Target lowering
  // (which carries the per-target wanted-scale-layout signal this legality pass
  // deliberately does not have) — see TileToApple / NVIDIA scale lowering.
  static LogicalResult checkGroupedScaleLayout(Operation *op) {
    StringRef name = op->getName().getStringRef();
    unsigned base;
    if (name == "tessera.grouped_gemm")
      base = 3;  // x, weights, group_sizes
    else if (name == "tessera.moe_swiglu_block")
      base = 5;  // x, w_gate, w_up, w_down, group_sizes
    else
      return success();
    bool hasScaleOperands = op->getNumOperands() > base;
    if (!hasScaleOperands) return success();  // bare (unscaled) form — legal
    if (op->getAttr("scale_layout")) return success();
    return op->emitOpError(
               "LAYOUT_LEGALITY_SCALE_WITHOUT_LAYOUT: ")
           << name
           << " has scale operand(s) but no `scale_layout` attribute; an "
              "untyped scale tensor has no layout contract.  Declare "
              "scale_layout (granularity / block / packing / transposed).";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool anyError = false;
    module.walk([&](Operation *op) {
      if (failed(checkCastLayout(op))) anyError = true;
      // Sprint V4a: producer/consumer accept-set rule for matmul.
      if (failed(checkMatmulOperandLayouts(op))) anyError = true;
      // 2026-06-11: same rule for conv2d_nhwc (nhwc) + flash_attn (bhsd).
      if (failed(checkTensorOpLayouts(op))) anyError = true;
      // 2026-06 (DeepGEMM keystone): scale operand requires a scale_layout.
      if (failed(checkGroupedScaleLayout(op))) anyError = true;
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
