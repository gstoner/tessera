//===- RotorSandwichFold.cpp -----------------------------------*- C++ -*-===//
//
// CliffordRotorSandwichFoldPass: recognizes the pattern
//
//     %t = clifford.geo_product %R, %x           : ...
//     %R_dag = clifford.reverse %R                : ...
//     %y = clifford.geo_product %t, %R_dag        : ...
//
// (i.e. ``R · x · R†`` written as three primitive ops) and rewrites
// it into a single high-level ``clifford.rotor_sandwich(R, x)``. The
// rotor-sandwich op survives as a high-level marker that GA9 backends
// can pick up to dispatch a fused kernel; if no fused backend kernel
// exists, a later expansion pass can lower it back to the three
// primitives.
//
// Why fuse at all? The fused form preserves the equivariance proof
// obligation (Decision GA-L4) explicitly. A downstream backend can
// also exploit shared subexpressions: every coefficient of the
// intermediate ``R · x`` appears in two of the four blade products
// expanded by ExpandProductTable.
//
// This pass MUST run before GradeFusion (which attaches output_grades
// to geo_products and would obscure the sandwich structure).
//
//===----------------------------------------------------------------------===//

#include "tessera/Clifford/CliffordPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace tessera {
namespace {

constexpr StringRef kGeoProductOpName = "tessera_clifford.geo_product";
constexpr StringRef kReverseOpName = "tessera_clifford.reverse";
constexpr StringRef kRotorSandwichOpName = "tessera_clifford.rotor_sandwich";

struct RotorSandwichFoldPattern : public RewritePattern {
  RotorSandwichFoldPattern(MLIRContext *ctx)
      : RewritePattern(kGeoProductOpName, /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Outer pattern: clifford.geo_product(%inner, %R_dag)
    //   where %inner = clifford.geo_product(%R, %x)
    //   and   %R_dag = clifford.reverse(%R).
    if (op->getNumOperands() != 2) return failure();
    Value innerVal = op->getOperand(0);
    Value rDagVal = op->getOperand(1);

    Operation *innerOp = innerVal.getDefiningOp();
    Operation *rDagOp = rDagVal.getDefiningOp();
    if (!innerOp || !rDagOp) return failure();
    if (innerOp->getName().getStringRef() != kGeoProductOpName) return failure();
    if (rDagOp->getName().getStringRef() != kReverseOpName) return failure();

    // Verify the reverse's source matches the inner product's left operand.
    Value rFromReverse = rDagOp->getOperand(0);
    Value rFromInner = innerOp->getOperand(0);
    if (rFromReverse != rFromInner) return failure();

    Value x = innerOp->getOperand(1);
    Value R = rFromInner;

    // Verify all three ops share the same algebra + dtype attributes.
    auto algebraOuter = op->getAttrOfType<ArrayAttr>("algebra");
    auto algebraInner = innerOp->getAttrOfType<ArrayAttr>("algebra");
    auto algebraReverse = rDagOp->getAttrOfType<ArrayAttr>("algebra");
    if (!algebraOuter || !algebraInner || !algebraReverse) return failure();
    if (algebraOuter != algebraInner || algebraOuter != algebraReverse) {
      return failure();
    }
    auto dtypeOuter = op->getAttrOfType<StringAttr>("dtype");
    auto dtypeInner = innerOp->getAttrOfType<StringAttr>("dtype");
    auto dtypeReverse = rDagOp->getAttrOfType<StringAttr>("dtype");
    if (!dtypeOuter || dtypeOuter != dtypeInner || dtypeOuter != dtypeReverse) {
      return failure();
    }

    // Build the fused rotor_sandwich op.
    OperationState newState(op->getLoc(), kRotorSandwichOpName);
    newState.addOperands({R, x});
    newState.addTypes({op->getResult(0).getType()});
    newState.addAttribute("algebra", algebraOuter);
    newState.addAttribute("dtype", dtypeOuter);
    newState.addAttribute("tessera.clifford.from_chain_fold",
                          rewriter.getUnitAttr());
    Operation *fused = rewriter.create(newState);
    rewriter.replaceOp(op, fused->getResults());

    // The inner geo_product and reverse may still have other uses;
    // we don't erase them here.  DCE handles the dead-code case
    // automatically if their only user was the outer geo_product we
    // just replaced.
    return success();
  }
};

struct CliffordRotorSandwichFoldPass
    : public PassWrapper<CliffordRotorSandwichFoldPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CliffordRotorSandwichFoldPass)

  StringRef getArgument() const final {
    return "tessera-clifford-rotor-sandwich-fold";
  }
  StringRef getDescription() const final {
    return "Recognize gp(gp(R, x), reverse(R)) chains and fuse into "
           "clifford.rotor_sandwich; tagged with "
           "`tessera.clifford.from_chain_fold` for diagnostic traceability.";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<RotorSandwichFoldPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCliffordRotorSandwichFoldPass() {
  return std::make_unique<CliffordRotorSandwichFoldPass>();
}

}  // namespace tessera
