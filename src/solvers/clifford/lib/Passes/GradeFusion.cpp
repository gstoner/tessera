//===- GradeFusion.cpp -----------------------------------------*- C++ -*-===//
//
// CliffordGradeFusionPass: walks `tessera_clifford.grade` ops whose
// source is a `tessera_clifford.geo_product`, attaches the
// `tessera.clifford.output_grades` attribute on the geo_product, and
// replaces the grade op with the (now-grade-restricted) geo_product
// result.
//
// The downstream `tessera-clifford-expand-product-table` pass reads
// the attribute and only emits the table slice that contributes to
// the requested grades — a compile-time-known sparsity saving on top
// of the already-sparse Cayley contraction.
//
// Worked example: `grade(2, geo_product(a, b))` in Cl(3,0):
//   - Without fusion: 64 mul-adds across 8 output coefficients.
//   - With fusion:    only the 6 (i, j) table entries whose result
//                     mask has popcount 2 contribute.
//
//===----------------------------------------------------------------------===//

#include "tessera/Clifford/CliffordPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <set>

using namespace mlir;

namespace tessera {
namespace {

constexpr StringRef kGradeOpName = "tessera_clifford.grade";
constexpr StringRef kGeoProductOpName = "tessera_clifford.geo_product";
constexpr StringRef kOutputGradesAttr = "tessera.clifford.output_grades";

struct GradeFusionPattern : public RewritePattern {
  GradeFusionPattern(MLIRContext *ctx)
      : RewritePattern(kGradeOpName, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // The grade op's source must be a geo_product.
    if (op->getNumOperands() != 1) return failure();
    Operation *src = op->getOperand(0).getDefiningOp();
    if (!src || src->getName().getStringRef() != kGeoProductOpName) {
      return failure();
    }
    // Read the grade restriction from the grade op.
    auto gradesAttr = op->getAttrOfType<ArrayAttr>("grades");
    if (!gradesAttr) return failure();

    // Merge with any existing output_grades on the geo_product (set by a
    // previous GradeFusion application — the same geo_product may be
    // consumed by multiple grade ops, in which case we need the union).
    std::set<int64_t> gradeSet;
    for (Attribute g : gradesAttr) {
      if (auto gi = dyn_cast<IntegerAttr>(g)) {
        gradeSet.insert(gi.getInt());
      }
    }
    if (auto existing = src->getAttrOfType<ArrayAttr>(kOutputGradesAttr)) {
      for (Attribute g : existing) {
        if (auto gi = dyn_cast<IntegerAttr>(g)) {
          gradeSet.insert(gi.getInt());
        }
      }
    }
    SmallVector<Attribute, 4> mergedGrades;
    for (int64_t g : gradeSet) {
      mergedGrades.push_back(rewriter.getI64IntegerAttr(g));
    }
    src->setAttr(kOutputGradesAttr,
                 ArrayAttr::get(rewriter.getContext(), mergedGrades));

    // Replace the grade op with the (annotated) geo_product result.
    // The geo_product still produces a full-dim tensor; the
    // ExpandProductTable pass will emit zero for the non-requested-grade
    // coefficients. That preserves type compatibility for downstream uses.
    rewriter.replaceOp(op, src->getResult(0));
    return success();
  }
};

struct CliffordGradeFusionPass
    : public PassWrapper<CliffordGradeFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CliffordGradeFusionPass)

  StringRef getArgument() const final { return "tessera-clifford-grade-fusion"; }
  StringRef getDescription() const final {
    return "Fuse grade(k, geo_product(a, b)) chains: attach output_grades "
           "attribute on the geo_product and erase the grade op.";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<GradeFusionPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCliffordGradeFusionPass() {
  return std::make_unique<CliffordGradeFusionPass>();
}

}  // namespace tessera
