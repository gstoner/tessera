//===- ExpandProductTable.cpp ----------------------------------*- C++ -*-===//
//
// CliffordExpandProductTablePass: lowers `tessera_clifford.geo_product`
// from an opaque high-level op into explicit `arith.mulf` / `arith.addf`
// sequences indexed by the algebra's compile-time-known Cayley table.
//
// For Cl(3, 0) (dim = 8) the unrolled contraction has up to 64
// mul-adds per output coefficient (8x8 table). The table is dense
// enough that emitting unrolled IR is reasonable; the alternative
// (linalg.generic + sparse-tensor encoding) would add MLIR-pipeline
// complexity for marginal benefit at these algebra sizes — Q1 locks
// v1 to dim ≤ 16.
//
// Restrictions (v1):
//   - Operands must be rank-1 RankedTensorType<dim x dtype> for now.
//     Higher-rank (batched) operands raise a diagnostic and skip; a
//     follow-on sprint can wrap the emission in an scf.for over the
//     leading axes.
//   - dtype must be float (f32 / f64 / f16 / bf16).
//
// Optimisations (v1):
//   - If the geo_product carries a `tessera.clifford.output_grades`
//     attribute (set by GradeFusionPass when fusing a downstream
//     `clifford.grade` consumer), only emit the slice of the table that
//     contributes to those grades — same compile-time-known sparsity
//     pattern, but smaller. This is the GA8 grade-fusion savings.
//
//===----------------------------------------------------------------------===//

#include "tessera/Clifford/CliffordPasses.h"
#include "CayleyTable.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdint>
#include <vector>

using namespace mlir;

namespace tessera {
namespace {

constexpr StringRef kGeoProductOpName = "tessera_clifford.geo_product";
constexpr StringRef kOutputGradesAttr = "tessera.clifford.output_grades";
constexpr StringRef kExpandedMarker = "tessera.clifford.expanded";

struct ExpandProductTablePattern : public RewritePattern {
  ExpandProductTablePattern(MLIRContext *ctx)
      : RewritePattern(kGeoProductOpName, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Skip ops already expanded.
    if (op->hasAttr(kExpandedMarker)) return failure();

    // Validate algebra attribute.
    auto algebra = op->getAttrOfType<ArrayAttr>("algebra");
    if (!algebra || algebra.size() != 3) return failure();
    int64_t p = cast<IntegerAttr>(algebra[0]).getInt();
    int64_t q = cast<IntegerAttr>(algebra[1]).getInt();
    int64_t r = cast<IntegerAttr>(algebra[2]).getInt();
    int64_t n = p + q + r;
    int64_t dim = int64_t(1) << n;

    // Restrict v1 to rank-1 static tensors.
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsTy || !rhsTy) return failure();
    if (lhsTy.getRank() != 1 || rhsTy.getRank() != 1) {
      op->emitWarning(
          "ExpandProductTable: batched (rank > 1) operands are pending a "
          "follow-on sprint; skipping");
      return failure();
    }
    if (lhsTy.getShape()[0] != dim || rhsTy.getShape()[0] != dim) {
      op->emitError("ExpandProductTable: operand last-dim must equal ")
          << dim << " for Cl(" << p << ", " << q << ", " << r << ")";
      return failure();
    }
    Type elemTy = lhsTy.getElementType();
    if (!elemTy.isF32() && !elemTy.isF64() && !elemTy.isF16() &&
        !elemTy.isBF16()) {
      op->emitError("ExpandProductTable: unsupported element type ") << elemTy;
      return failure();
    }

    // Compute the Cayley table at pass time.
    auto table = tessera::clifford::buildCayleyTable(p, q, r);

    // Read optional output_grades restriction (set by GradeFusion).
    std::vector<bool> wantGrade(n + 1, true);
    if (auto gradesAttr = op->getAttrOfType<ArrayAttr>(kOutputGradesAttr)) {
      std::fill(wantGrade.begin(), wantGrade.end(), false);
      for (Attribute g : gradesAttr) {
        if (auto gi = dyn_cast<IntegerAttr>(g)) {
          int64_t k = gi.getInt();
          if (k >= 0 && k <= n) wantGrade[k] = true;
        }
      }
    }

    Location loc = op->getLoc();

    // Pre-extract all lhs and rhs coefficients as scalars.
    // tensor.extract %lhs[%i_idx] : tensor<dim x elemTy>
    std::vector<Value> lhsCoeffs(dim), rhsCoeffs(dim);
    for (int64_t i = 0; i < dim; ++i) {
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      lhsCoeffs[i] = rewriter.create<tensor::ExtractOp>(loc, lhs, ValueRange{idx});
      rhsCoeffs[i] = rewriter.create<tensor::ExtractOp>(loc, rhs, ValueRange{idx});
    }

    // For each output coefficient k, accumulate the relevant (i, j) terms.
    auto zeroAttr = rewriter.getZeroAttr(elemTy);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elemTy, cast<TypedAttr>(zeroAttr));

    std::vector<Value> outCoeffs(dim, zero);
    for (int64_t i = 0; i < dim; ++i) {
      for (int64_t j = 0; j < dim; ++j) {
        auto entry = table[i][j];
        if (entry.sign == 0) continue;
        int outGrade = tessera::clifford::gradeOfMask(entry.result_mask);
        if (!wantGrade[outGrade]) continue;
        // term = lhs[i] * rhs[j]
        Value prod = rewriter.create<arith::MulFOp>(loc, lhsCoeffs[i], rhsCoeffs[j]);
        Value updated;
        if (entry.sign == 1) {
          updated = rewriter.create<arith::AddFOp>(loc, outCoeffs[entry.result_mask], prod);
        } else {  // -1
          updated = rewriter.create<arith::SubFOp>(loc, outCoeffs[entry.result_mask], prod);
        }
        outCoeffs[entry.result_mask] = updated;
      }
    }

    // Build the result tensor: tensor.from_elements %c0, %c1, ..., %c{dim-1}.
    Value resultTensor =
        rewriter.create<tensor::FromElementsOp>(loc, lhsTy, outCoeffs);

    rewriter.replaceOp(op, resultTensor);
    return success();
  }
};

struct CliffordExpandProductTablePass
    : public PassWrapper<CliffordExpandProductTablePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CliffordExpandProductTablePass)

  StringRef getArgument() const final {
    return "tessera-clifford-expand-product-table";
  }
  StringRef getDescription() const final {
    return "Lower clifford.geo_product to an unrolled arith.mulf/addf "
           "sequence driven by the algebra's compile-time-known Cayley "
           "table.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ExpandProductTablePattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCliffordExpandProductTablePass() {
  return std::make_unique<CliffordExpandProductTablePass>();
}

}  // namespace tessera
