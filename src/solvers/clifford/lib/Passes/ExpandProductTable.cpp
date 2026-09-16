//===- ExpandProductTable.cpp ----------------------------------*- C++ -*-===//
//
// CliffordExpandProductTablePass: lowers the Clifford product family from
// opaque high-level ops into explicit `arith.mulf` / `arith.addf` sequences
// indexed by the algebra's compile-time-known Cayley table.
//
// Since 2026-09-16 (W6.4 second slice) the same table and the same batched
// loop nest carry every linear/bilinear op the standalone GA reference
// defines, so one lowering path -- not a per-op kernel -- reaches the JIT:
//   bilinear:  geo_product (all terms), wedge (disjoint blades only),
//              left_contract (result grade == grade(b) - grade(a) >= 0)
//   scalar:    inner  = <a . reverse(b)>_0, norm = sqrt(max(<a . a~>_0, 0))
//   unary:     reverse / grade_involute / conjugate (per-grade signs),
//              hodge_star (reverse(a) . I, a blade permutation with signs),
//              grade (standalone projection: keep the listed grades)
//   rewrite:   rotor_sandwich(R, x) -> geo_product(geo_product(R, x), reverse(R))
// Inner/norm results are `[..., 1]` (one scalar per multivector).
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
#include "mlir/Dialect/Math/IR/Math.h"
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


// ---------------------------------------------------------------------------
// Shared machinery for the rest of the family.
// ---------------------------------------------------------------------------

// Reads `algebra = [p, q, r]`; returns false when absent or malformed.
static bool readAlgebra(Operation *op, int64_t &p, int64_t &q, int64_t &r) {
  auto algebra = op->getAttrOfType<ArrayAttr>("algebra");
  if (!algebra || algebra.size() != 3) return false;
  p = cast<IntegerAttr>(algebra[0]).getInt();
  q = cast<IntegerAttr>(algebra[1]).getInt();
  r = cast<IntegerAttr>(algebra[2]).getInt();
  return p >= 0 && q >= 0 && r >= 0 && p + q + r <= 4;
}

// Validates a `[..., dim]` static float operand and returns its type.
static RankedTensorType multivectorType(Operation *op, Value v, int64_t dim,
                                        StringRef what) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty || ty.getRank() < 1 || !ty.hasStaticShape()) {
    op->emitError("Clifford lowering: ") << what << " must be a static ranked tensor of shape [..., " << dim << "]";
    return nullptr;
  }
  if (ty.getShape().back() != dim) {
    op->emitError("Clifford lowering: ") << what << " last-dim must equal " << dim;
    return nullptr;
  }
  Type e = ty.getElementType();
  if (!e.isF32() && !e.isF64() && !e.isF16() && !e.isBF16()) {
    op->emitError("Clifford lowering: unsupported element type ") << e;
    return nullptr;
  }
  return ty;
}

// Emits `resultTy` by running `body(prefix)` once per leading coordinate;
// `body` returns the trailing coefficients (resultTy's last dim of them).
// Rank 1 uses tensor.from_elements; higher ranks an scf.for nest carrying the
// result tensor as iter_arg, every coefficient written.
static Value emitPerMultivector(
    PatternRewriter &rewriter, Location loc, RankedTensorType resultTy,
    llvm::function_ref<std::vector<Value>(ArrayRef<Value>)> body) {
  const int64_t rank = resultTy.getRank();
  if (rank == 1)
    return rewriter.create<tensor::FromElementsOp>(loc, resultTy, body({}));
  Value init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType());
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  const int64_t trailing = resultTy.getShape().back();
  std::function<Value(int64_t, SmallVector<Value> &, Value)> build =
      [&](int64_t axis, SmallVector<Value> &ivs, Value carried) -> Value {
    if (axis == rank - 1) {
      std::vector<Value> coeffs = body(ivs);
      Value updated = carried;
      for (int64_t k = 0; k < trailing; ++k) {
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, k));
        updated = rewriter.create<tensor::InsertOp>(loc, coeffs[k], updated, indices);
      }
      return updated;
    }
    Value ub = rewriter.create<arith::ConstantIndexOp>(loc, resultTy.getShape()[axis]);
    auto loop = rewriter.create<scf::ForOp>(loc, c0, ub, c1, ValueRange{carried});
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    ivs.push_back(loop.getInductionVar());
    Value inner = build(axis + 1, ivs, loop.getRegionIterArgs()[0]);
    ivs.pop_back();
    rewriter.create<scf::YieldOp>(loc, ValueRange{inner});
    return loop.getResult(0);
  };
  SmallVector<Value> ivs;
  return build(0, ivs, init);
}

// Extracts the `dim` coefficients of `v` at leading coordinate `prefix`.
static std::vector<Value> coefficientsAt(PatternRewriter &rewriter, Location loc,
                                         Value v, int64_t dim, ArrayRef<Value> prefix) {
  std::vector<Value> out(dim);
  for (int64_t i = 0; i < dim; ++i) {
    SmallVector<Value> indices(prefix.begin(), prefix.end());
    indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
    out[i] = rewriter.create<tensor::ExtractOp>(loc, v, indices);
  }
  return out;
}

static int reverseSign(int k) { return ((k * (k - 1)) / 2) % 2 ? -1 : 1; }
static int involuteSign(int k) { return k % 2 ? -1 : 1; }
static int conjugateSign(int k) { return ((k * (k + 1)) / 2) % 2 ? -1 : 1; }

// Accumulate `acc (+|-)= a * b` per the sign.
static Value fma(PatternRewriter &rewriter, Location loc, Value acc, Value a, Value b, int sign) {
  Value prod = rewriter.create<arith::MulFOp>(loc, a, b);
  return sign > 0 ? rewriter.create<arith::AddFOp>(loc, acc, prod).getResult()
                  : rewriter.create<arith::SubFOp>(loc, acc, prod).getResult();
}

enum class Bilinear { Wedge, LeftContract };

// wedge / left_contract: the geometric-product table gated per term.
struct BilinearTablePattern : public RewritePattern {
  Bilinear kind;
  BilinearTablePattern(MLIRContext *ctx, StringRef name, Bilinear kind)
      : RewritePattern(name, /*benefit=*/1, ctx), kind(kind) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    int64_t p, q, r;
    if (!readAlgebra(op, p, q, r)) return failure();
    const int64_t dim = int64_t(1) << (p + q + r);
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    auto lhsTy = multivectorType(op, lhs, dim, "lhs");
    auto rhsTy = multivectorType(op, rhs, dim, "rhs");
    if (!lhsTy || !rhsTy) return failure();
    if (lhsTy.getShape() != rhsTy.getShape()) {
      op->emitError("Clifford lowering: operand shapes must agree");
      return failure();
    }
    auto table = tessera::clifford::buildCayleyTable(p, q, r);
    Location loc = op->getLoc();
    Type elemTy = lhsTy.getElementType();
    Value zero = rewriter.create<arith::ConstantOp>(loc, elemTy, cast<TypedAttr>(rewriter.getZeroAttr(elemTy)));
    Value result = emitPerMultivector(rewriter, loc, lhsTy, [&](ArrayRef<Value> prefix) {
      auto a = coefficientsAt(rewriter, loc, lhs, dim, prefix);
      auto b = coefficientsAt(rewriter, loc, rhs, dim, prefix);
      std::vector<Value> out(dim, zero);
      for (int64_t i = 0; i < dim; ++i)
        for (int64_t j = 0; j < dim; ++j) {
          auto entry = table[i][j];
          if (entry.sign == 0) continue;
          if (kind == Bilinear::Wedge && (i & j) != 0) continue;  // shared generator
          if (kind == Bilinear::LeftContract) {
            int target = tessera::clifford::gradeOfMask(j) - tessera::clifford::gradeOfMask(i);
            if (target < 0 || tessera::clifford::gradeOfMask(entry.result_mask) != target) continue;
          }
          out[entry.result_mask] = fma(rewriter, loc, out[entry.result_mask], a[i], b[j], entry.sign);
        }
      return out;
    });
    rewriter.replaceOp(op, result);
    return success();
  }
};

// inner: <a . reverse(b)>_0 ; norm: sqrt(max(<a . reverse(a)>_0, 0)). Both
// yield one scalar per multivector, typed `[..., 1]`.
struct ScalarFormPattern : public RewritePattern {
  bool isNorm;
  ScalarFormPattern(MLIRContext *ctx, StringRef name, bool isNorm)
      : RewritePattern(name, /*benefit=*/1, ctx), isNorm(isNorm) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    int64_t p, q, r;
    if (!readAlgebra(op, p, q, r)) return failure();
    const int64_t dim = int64_t(1) << (p + q + r);
    Value lhs = op->getOperand(0);
    Value rhs = isNorm ? lhs : op->getOperand(1);
    auto lhsTy = multivectorType(op, lhs, dim, "lhs");
    auto rhsTy = multivectorType(op, rhs, dim, "rhs");
    if (!lhsTy || !rhsTy) return failure();
    if (lhsTy.getShape() != rhsTy.getShape()) {
      op->emitError("Clifford lowering: operand shapes must agree");
      return failure();
    }
    SmallVector<int64_t> resultShape(lhsTy.getShape().begin(), lhsTy.getShape().end());
    resultShape.back() = 1;
    auto resultTy = RankedTensorType::get(resultShape, lhsTy.getElementType());
    if (op->getResult(0).getType() != resultTy) {
      op->emitError("Clifford lowering: scalar-form result must be typed ") << resultTy;
      return failure();
    }
    auto table = tessera::clifford::buildCayleyTable(p, q, r);
    Location loc = op->getLoc();
    Type elemTy = lhsTy.getElementType();
    Value zero = rewriter.create<arith::ConstantOp>(loc, elemTy, cast<TypedAttr>(rewriter.getZeroAttr(elemTy)));
    Value result = emitPerMultivector(rewriter, loc, resultTy, [&](ArrayRef<Value> prefix) {
      auto a = coefficientsAt(rewriter, loc, lhs, dim, prefix);
      auto b = isNorm ? a : coefficientsAt(rewriter, loc, rhs, dim, prefix);
      Value acc = zero;
      // Only i == j reaches the scalar blade; reverse(b) flips grade signs.
      for (int64_t i = 0; i < dim; ++i) {
        auto entry = table[i][i];
        if (entry.sign == 0 || entry.result_mask != 0) continue;
        int sign = entry.sign * reverseSign(tessera::clifford::gradeOfMask(i));
        acc = fma(rewriter, loc, acc, a[i], b[i], sign);
      }
      if (isNorm) {
        Value clipped = rewriter.create<arith::MaximumFOp>(loc, acc, zero);
        acc = rewriter.create<math::SqrtOp>(loc, clipped);
      }
      return std::vector<Value>{acc};
    });
    rewriter.replaceOp(op, result);
    return success();
  }
};

enum class Unary { Reverse, Involute, Conjugate, HodgeStar, Grade };

// Linear maps: each output coefficient is +-1 times one input coefficient
// (or dropped), all decided at pass time.
struct UnaryMapPattern : public RewritePattern {
  Unary kind;
  UnaryMapPattern(MLIRContext *ctx, StringRef name, Unary kind)
      : RewritePattern(name, /*benefit=*/1, ctx), kind(kind) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    int64_t p, q, r;
    if (!readAlgebra(op, p, q, r)) return failure();
    const int64_t n = p + q + r, dim = int64_t(1) << n;
    Value in = op->getOperand(0);
    auto ty = multivectorType(op, in, dim, "operand");
    if (!ty) return failure();
    // out[target[i]] = sign[i] * in[i]; sign 0 drops the blade.
    std::vector<int64_t> target(dim);
    std::vector<int> sign(dim, 1);
    std::set<int64_t> keep;
    if (kind == Unary::Grade) {
      auto grades = op->getAttrOfType<ArrayAttr>("grades");
      if (!grades) {
        op->emitError("Clifford lowering: grade projection requires a `grades` attribute");
        return failure();
      }
      for (Attribute g : grades)
        if (auto gi = dyn_cast<IntegerAttr>(g)) keep.insert(gi.getInt());
    }
    auto table = tessera::clifford::buildCayleyTable(p, q, r);
    for (int64_t i = 0; i < dim; ++i) {
      int k = tessera::clifford::gradeOfMask(i);
      target[i] = i;
      switch (kind) {
      case Unary::Reverse:   sign[i] = reverseSign(k); break;
      case Unary::Involute:  sign[i] = involuteSign(k); break;
      case Unary::Conjugate: sign[i] = conjugateSign(k); break;
      case Unary::Grade:     sign[i] = keep.count(k) ? 1 : 0; break;
      case Unary::HodgeStar: {
        // reverse(a) . I with I the pseudoscalar blade (mask dim-1).
        auto entry = table[i][dim - 1];
        target[i] = entry.result_mask;
        sign[i] = entry.sign * reverseSign(k);
        break;
      }
      }
    }
    Location loc = op->getLoc();
    Type elemTy = ty.getElementType();
    Value zero = rewriter.create<arith::ConstantOp>(loc, elemTy, cast<TypedAttr>(rewriter.getZeroAttr(elemTy)));
    Value result = emitPerMultivector(rewriter, loc, ty, [&](ArrayRef<Value> prefix) {
      auto a = coefficientsAt(rewriter, loc, in, dim, prefix);
      std::vector<Value> out(dim, zero);
      for (int64_t i = 0; i < dim; ++i) {
        if (sign[i] == 0) continue;
        out[target[i]] = sign[i] > 0 ? a[i] : rewriter.create<arith::NegFOp>(loc, a[i]).getResult();
      }
      return out;
    });
    rewriter.replaceOp(op, result);
    return success();
  }
};

// rotor_sandwich(R, x) -> geo_product(geo_product(R, x), reverse(R)); the
// three ops then lower through the patterns above in the same driver run.
struct RotorSandwichExpandPattern : public RewritePattern {
  RotorSandwichExpandPattern(MLIRContext *ctx)
      : RewritePattern("tessera_clifford.rotor_sandwich", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    int64_t p, q, r;
    if (!readAlgebra(op, p, q, r)) return failure();
    Value rotor = op->getOperand(0), x = op->getOperand(1);
    auto ty = dyn_cast<RankedTensorType>(x.getType());
    if (!ty || rotor.getType() != x.getType()) {
      op->emitError("Clifford lowering: rotor_sandwich operands must share one static type");
      return failure();
    }
    Location loc = op->getLoc();
    auto make = [&](StringRef name, ValueRange operands) -> Value {
      OperationState state(loc, name);
      state.addOperands(operands);
      state.addTypes(ty);
      state.addAttribute("algebra", op->getAttr("algebra"));
      if (Attribute dtype = op->getAttr("dtype")) state.addAttribute("dtype", dtype);
      return rewriter.create(state)->getResult(0);
    };
    Value rx = make("tessera_clifford.geo_product", {rotor, x});
    Value rdag = make("tessera_clifford.reverse", {rotor});
    Value y = make("tessera_clifford.geo_product", {rx, rdag});
    rewriter.replaceOp(op, y);
    return success();
  }
};

struct CliffordExpandProductTablePass
    : public PassWrapper<CliffordExpandProductTablePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CliffordExpandProductTablePass)

  CliffordExpandProductTablePass() = default;
  CliffordExpandProductTablePass(const CliffordExpandProductTablePass &other)
      : PassWrapper(other) {}
  explicit CliffordExpandProductTablePass(bool expandSandwich) {
    expandRotorSandwich = expandSandwich;
  }

  // `rotor_sandwich` is a fused marker for backends that ship a sandwich
  // kernel (RotorSandwichFold forms it); by default it survives this pass.
  // A consumer with no such kernel -- the MLIR/LLVM JIT lane -- asks for the
  // expansion into gp(gp(R, x), reverse(R)), which then lowers here too.
  Option<bool> expandRotorSandwich{
      *this, "expand-rotor-sandwich",
      llvm::cl::desc("Lower rotor_sandwich to its geo_product/reverse chain "
                     "instead of keeping it as a fused-kernel marker"),
      llvm::cl::init(false)};

  StringRef getArgument() const final {
    return "tessera-clifford-expand-product-table";
  }
  StringRef getDescription() const final {
    return "Lower the Clifford product family (geo_product, wedge, "
           "left_contract, inner, norm, reverse, grade_involute, conjugate, "
           "hodge_star, grade, rotor_sandwich) to arith/scf/tensor driven by "
           "the algebra's compile-time-known Cayley table, batched over "
           "leading axes.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ExpandProductTablePattern>(ctx);
    if (expandRotorSandwich) patterns.add<RotorSandwichExpandPattern>(ctx);
    patterns.add<BilinearTablePattern>(ctx, "tessera_clifford.wedge", Bilinear::Wedge);
    patterns.add<BilinearTablePattern>(ctx, "tessera_clifford.left_contract", Bilinear::LeftContract);
    patterns.add<ScalarFormPattern>(ctx, "tessera_clifford.inner", /*isNorm=*/false);
    patterns.add<ScalarFormPattern>(ctx, "tessera_clifford.norm", /*isNorm=*/true);
    patterns.add<UnaryMapPattern>(ctx, "tessera_clifford.reverse", Unary::Reverse);
    patterns.add<UnaryMapPattern>(ctx, "tessera_clifford.grade_involute", Unary::Involute);
    patterns.add<UnaryMapPattern>(ctx, "tessera_clifford.conjugate", Unary::Conjugate);
    patterns.add<UnaryMapPattern>(ctx, "tessera_clifford.hodge_star", Unary::HodgeStar);
    patterns.add<UnaryMapPattern>(ctx, "tessera_clifford.grade", Unary::Grade);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCliffordExpandProductTablePass() {
  return std::make_unique<CliffordExpandProductTablePass>();
}

std::unique_ptr<mlir::Pass> createCliffordExpandProductTablePass(bool expandRotorSandwich) {
  return std::make_unique<CliffordExpandProductTablePass>(expandRotorSandwich);
}

}  // namespace tessera
