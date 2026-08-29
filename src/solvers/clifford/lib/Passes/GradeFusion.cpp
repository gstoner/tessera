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
constexpr StringRef kLhsGradesAttr = "tessera.clifford.input_grades_lhs";
constexpr StringRef kRhsGradesAttr = "tessera.clifford.input_grades_rhs";

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

    std::set<int64_t> gradeSet;
    for (Attribute g : gradesAttr) {
      if (auto gi = dyn_cast<IntegerAttr>(g)) {
        gradeSet.insert(gi.getInt());
      }
    }

    // `output_grades` restricts the PRODUCT, not this one consumer:
    // ExpandProductTable drops every Cayley entry whose result grade is
    // outside the set. So the projection may be folded into the product only
    // when it speaks for every consumer of that product. A second grade op
    // asking for different grades, or any non-grade consumer of the raw
    // product, would otherwise silently receive this projection's restriction
    // — a union across differing consumers hands each of them the others'
    // grades, and a lone consumer's set strips the raw user's.
    auto gradesOf = [](Operation *o, std::set<int64_t> &out) -> bool {
      auto attr = o->getAttrOfType<ArrayAttr>("grades");
      if (!attr) return false;
      for (Attribute g : attr)
        if (auto gi = dyn_cast<IntegerAttr>(g)) out.insert(gi.getInt());
      return true;
    };
    for (Operation *user : src->getResult(0).getUsers()) {
      if (user->getName().getStringRef() != kGradeOpName) return failure();
      std::set<int64_t> userGrades;
      if (!gradesOf(user, userGrades)) return failure();
      if (userGrades != gradeSet) return failure();
    }
    // A previous fold may already have restricted the product; it must agree.
    if (auto existing = src->getAttrOfType<ArrayAttr>(kOutputGradesAttr)) {
      std::set<int64_t> existingSet;
      for (Attribute g : existing)
        if (auto gi = dyn_cast<IntegerAttr>(g)) existingSet.insert(gi.getInt());
      if (existingSet != gradeSet) return failure();
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

// W1.4 — the mirror of the pattern above.
//
// `output_grades` prunes the Cayley table by WHICH RESULTS ARE WANTED.
// `input_grades` prunes it by WHICH INPUTS CAN BE NON-ZERO, which is the other
// half of the same compile-time sparsity and was declared nowhere: the plan
// cites `MultivectorSpec.grades` reaching `geometric_product` as a live
// Decision #29 violation, and this is its MLIR-side counterpart.
//
// Worked example in Cl(3,0), `geo_product(grade(1,a), grade(1,b))`:
//   - Unrestricted:   64 (i, j) table entries.
//   - output_grades:  prunes by the 8 result masks.
//   - input_grades:   only 3 lhs blades x 3 rhs blades can contribute, so 9
//                     entries survive -- and the result is grades {0, 2},
//                     which no output restriction had to be written to learn.
//
// The two compose: an input restriction narrows which products exist, an
// output restriction narrows which are kept.
struct InputGradeFusionPattern : public RewritePattern {
  InputGradeFusionPattern(MLIRContext *ctx)
      : RewritePattern(kGeoProductOpName, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2) return failure();

    bool changed = false;
    const StringRef attrNames[2] = {kLhsGradesAttr, kRhsGradesAttr};
    for (unsigned side = 0; side < 2; ++side) {
      // Already annotated: leave it. Re-deriving would be harmless but the
      // pattern would never converge, since the greedy driver re-runs while
      // anything changes.
      if (op->hasAttr(attrNames[side])) continue;

      Operation *def = op->getOperand(side).getDefiningOp();
      if (!def || def->getName().getStringRef() != kGradeOpName) continue;
      auto gradesAttr = def->getAttrOfType<ArrayAttr>("grades");
      if (!gradesAttr) continue;

      std::set<int64_t> gradeSet;
      for (Attribute g : gradesAttr)
        if (auto gi = dyn_cast<IntegerAttr>(g)) gradeSet.insert(gi.getInt());
      if (gradeSet.empty()) continue;

      SmallVector<Attribute, 4> grades;
      for (int64_t g : gradeSet)
        grades.push_back(rewriter.getI64IntegerAttr(g));
      op->setAttr(attrNames[side],
                  ArrayAttr::get(rewriter.getContext(), grades));
      changed = true;
    }
    return changed ? success() : failure();
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
    patterns.add<InputGradeFusionPattern>(ctx);
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
