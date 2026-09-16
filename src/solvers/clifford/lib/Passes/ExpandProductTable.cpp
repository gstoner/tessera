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
// Restrictions:
//   - Operands are static RankedTensorType<[..., dim] x dtype> of equal
//     shape. Rank 1 is the single multivector (v1). Since 2026-09-16 any
//     higher rank is the W6.4 batched form: the same compile-time-known
//     table is emitted once inside an `scf.for` nest over the leading axes
//     (tensor iter_arg, `tensor.extract` / `tensor.insert` per coefficient),
//     so the batch loops and the sparse table reach native IR together and
//     bufferize in place. Dynamic extents fail closed with a diagnostic.
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdint>
#include <functional>
#include <set>
#include <vector>

using namespace mlir;

namespace tessera {
namespace {

constexpr StringRef kGeoProductOpName = "tessera_clifford.geo_product";
constexpr StringRef kOutputGradesAttr = "tessera.clifford.output_grades";
constexpr StringRef kLhsGradesAttr = "tessera.clifford.input_grades_lhs";
constexpr StringRef kRhsGradesAttr = "tessera.clifford.input_grades_rhs";

// Per-blade "can be non-zero" mask from a declared grade set.  Blade index is
// the basis-blade bitmask, so its grade is popcount(index).  Returns all-true
// when the attribute is absent -- an undeclared operand is unrestricted, not
// empty, and getting that backwards would silently emit a zero product.
static std::vector<bool> bladeMaskFor(Operation *op, StringRef attrName,
                                      int64_t dim) {
  std::vector<bool> mask(dim, true);
  auto gradesAttr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!gradesAttr) return mask;
  std::set<int64_t> wanted;
  for (Attribute g : gradesAttr)
    if (auto gi = dyn_cast<IntegerAttr>(g)) wanted.insert(gi.getInt());
  if (wanted.empty()) return mask;
  for (int64_t i = 0; i < dim; ++i)
    mask[i] = wanted.count(tessera::clifford::gradeOfMask(i)) > 0;
  return mask;
}
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

    // Static `[..., dim]` operands of one shape; rank 1 is a single
    // multivector, higher ranks are batched over the leading axes.
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsTy || !rhsTy) return failure();
    if (lhsTy.getRank() < 1 || !lhsTy.hasStaticShape() || !rhsTy.hasStaticShape()) {
      op->emitError("ExpandProductTable: operands must be static ranked "
                    "tensors of shape [..., ")
          << dim << "]; dynamic or unranked operands are not lowered";
      return failure();
    }
    if (lhsTy.getShape() != rhsTy.getShape()) {
      op->emitError("ExpandProductTable: operand shapes must agree (got ")
          << lhsTy << " and " << rhsTy << ")";
      return failure();
    }
    if (lhsTy.getShape().back() != dim) {
      op->emitError("ExpandProductTable: operand last-dim must equal ")
          << dim << " for Cl(" << p << ", " << q << ", " << r << ")";
      return failure();
    }
    const int64_t rank = lhsTy.getRank();
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

    // W1.4 -- operand grade restrictions (set by GradeFusion's input pattern).
    // `output_grades` prunes by which results are wanted; these prune by which
    // inputs can be non-zero.  The two compose: an input restriction narrows
    // which products exist at all, an output restriction narrows which of them
    // are kept.
    const std::vector<bool> lhsMask = bladeMaskFor(op, kLhsGradesAttr, dim);
    const std::vector<bool> rhsMask = bladeMaskFor(op, kRhsGradesAttr, dim);

    Location loc = op->getLoc();
    auto zeroAttr = rewriter.getZeroAttr(elemTy);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elemTy, cast<TypedAttr>(zeroAttr));

    // One multivector product at coordinate `prefix` (the leading indices;
    // empty for rank 1): extract both coefficient rows, accumulate the
    // table's surviving (i, j) terms, return the `dim` output coefficients.
    auto productAt = [&](ArrayRef<Value> prefix) -> std::vector<Value> {
      std::vector<Value> lhsCoeffs(dim), rhsCoeffs(dim);
      for (int64_t i = 0; i < dim; ++i) {
        SmallVector<Value> indices(prefix.begin(), prefix.end());
        indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
        lhsCoeffs[i] = rewriter.create<tensor::ExtractOp>(loc, lhs, indices);
        rhsCoeffs[i] = rewriter.create<tensor::ExtractOp>(loc, rhs, indices);
      }
      std::vector<Value> outCoeffs(dim, zero);
      for (int64_t i = 0; i < dim; ++i) {
        if (!lhsMask[i]) continue;
        for (int64_t j = 0; j < dim; ++j) {
          if (!rhsMask[j]) continue;
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
      return outCoeffs;
    };

    Value resultTensor;
    if (rank == 1) {
      // Single multivector: tensor.from_elements %c0, ..., %c{dim-1}.
      std::vector<Value> outCoeffs = productAt({});
      resultTensor = rewriter.create<tensor::FromElementsOp>(loc, lhsTy, outCoeffs);
    } else {
      // Batched: an scf.for nest over the leading axes carrying the result
      // tensor; every coefficient (pruned grades included, as zero) is
      // written, so the result is fully defined.
      Value init = rewriter.create<tensor::EmptyOp>(loc, lhsTy.getShape(), elemTy);
      Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      std::function<Value(int64_t, SmallVector<Value> &, Value)> buildLoops =
          [&](int64_t axis, SmallVector<Value> &ivs, Value carried) -> Value {
        if (axis == rank - 1) {
          std::vector<Value> outCoeffs = productAt(ivs);
          Value updated = carried;
          for (int64_t k = 0; k < dim; ++k) {
            SmallVector<Value> indices(ivs.begin(), ivs.end());
            indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, k));
            updated = rewriter.create<tensor::InsertOp>(loc, outCoeffs[k], updated, indices);
          }
          return updated;
        }
        Value ub = rewriter.create<arith::ConstantIndexOp>(loc, lhsTy.getShape()[axis]);
        auto loop = rewriter.create<scf::ForOp>(loc, c0, ub, c1, ValueRange{carried});
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());
        ivs.push_back(loop.getInductionVar());
        Value inner = buildLoops(axis + 1, ivs, loop.getRegionIterArgs()[0]);
        ivs.pop_back();
        rewriter.create<scf::YieldOp>(loc, ValueRange{inner});
        return loop.getResult(0);
      };
      SmallVector<Value> ivs;
      resultTensor = buildLoops(0, ivs, init);
    }

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
    registry.insert<arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect>();
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
