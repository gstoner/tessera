//===- TesseraToLinalgPass.cpp --------------------------------------------===//
// Phase 0–1 of the production MLIR/LLVM compiler
// (docs/spec/PRODUCTION_COMPILER_PLAN.md).
//
// Lowers Tessera Graph IR onto upstream `linalg` on tensors — the shared
// front-half of the production spine. Every target (CPU LLVM/ORC now; NVVM,
// ROCDL later) inherits this lowering.
//
// Coverage:
//   Phase 0: tessera.add (elementwise, total)
//   Phase 1: tessera.{sub,mul,div} (elementwise via shared table)
//            tessera.matmul (linalg.fill + linalg.matmul; rank-2, no transpose)
//            tessera.reduce (linalg.reduce; sum/max/min/mean over one axis —
//                            first op whose result rank != input rank)
//            tessera.softmax (stable max→sub→exp→sum→div over one axis; first
//                             use of broadcast affine maps + the math dialect)
//            tessera.{rmsnorm,layer_norm} (mean-reduce + broadcast + math.sqrt;
//                             unweighted, innermost axis, eps default 1e-5)
//            tessera.{relu,sigmoid,tanh,silu,gelu} (unary math family; gelu is
//                             the tanh approximation; bf16/f16-aware)
//   bf16: matmul accumulates in f32 then truncf to storage dtype (ABI §12.5)
//
// Decisions:
//   - Match by op-name string (RewritePattern). Avoids leaking generated op
//     class includes through this pass; the ODS still verifies at parse time.
//   - All ops emit DPS form (`outs` is a fresh tensor.empty / linalg.fill);
//     bufferization + the harness's DPS rewrite turn the result into a
//     caller-allocated out-param (RUNTIME_ABI_SPEC §12.3).
//   - tessera.matmul attributes (transposeA/B, tile_k, numeric_policy) cause
//     match failure today; later sprints add transpose / accum-policy support.
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// Forward declaration: defined with the reduction/broadcast helpers below, but
// used earlier by MatmulLowering (bf16 truncf epilogue).
static Value emitUnaryElementwise(
    PatternRewriter &rewriter, Location loc, RankedTensorType ty, Value input,
    function_ref<Value(OpBuilder &, Location, Value)> fn);

// ── Binary elementwise ─────────────────────────────────────────────────────

enum class BinaryKind { Add, Sub, Mul, Div };

static Value emitBinaryScalar(OpBuilder &b, Location loc, BinaryKind kind,
                              Value a, Value c, Type elem) {
  bool isFloat = isa<FloatType>(elem);
  switch (kind) {
  case BinaryKind::Add:
    return isFloat ? arith::AddFOp::create(b, loc, a, c).getResult()
                   : arith::AddIOp::create(b, loc, a, c).getResult();
  case BinaryKind::Sub:
    return isFloat ? arith::SubFOp::create(b, loc, a, c).getResult()
                   : arith::SubIOp::create(b, loc, a, c).getResult();
  case BinaryKind::Mul:
    return isFloat ? arith::MulFOp::create(b, loc, a, c).getResult()
                   : arith::MulIOp::create(b, loc, a, c).getResult();
  case BinaryKind::Div:
    return isFloat ? arith::DivFOp::create(b, loc, a, c).getResult()
                   : arith::DivSIOp::create(b, loc, a, c).getResult();
  }
  llvm_unreachable("unhandled BinaryKind");
}

// Build a fully-parallel elementwise `linalg.generic` applying `combine` over
// `lhs`/`rhs` into a fresh `tensor.empty` of `resultType` (DPS — init is `outs`,
// matching RUNTIME_ABI_SPEC §12.3).
static Value buildElementwiseGeneric(
    PatternRewriter &rewriter, Location loc, RankedTensorType resultType,
    Value lhs, Value rhs,
    function_ref<Value(OpBuilder &, Location, Value, Value)> combine) {
  int64_t rank = resultType.getRank();
  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType());
  AffineMap id = rewriter.getMultiDimIdentityMap(rank);
  SmallVector<AffineMap, 3> maps = {id, id, id};
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      rewriter, loc, TypeRange{resultType}, ValueRange{lhs, rhs},
      ValueRange{init}, maps, iters,
      [&](OpBuilder &b, Location l, ValueRange args) {
        Value r = combine(b, l, args[0], args[1]);
        linalg::YieldOp::create(b, l, r);
      });
  return generic.getResult(0);
}

struct BinaryEltwiseLowering : public RewritePattern {
  BinaryKind kind;
  BinaryEltwiseLowering(MLIRContext *ctx, StringRef opName, BinaryKind k)
      : RewritePattern(opName, /*benefit=*/1, ctx), kind(k) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "static-shape ranked-tensor result required");
    Type elem = resultType.getElementType();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Location loc = op->getLoc();
    BinaryKind k = kind;
    Value out = buildElementwiseGeneric(
        rewriter, loc, resultType, lhs, rhs,
        [&](OpBuilder &b, Location l, Value a, Value c) -> Value {
          return emitBinaryScalar(b, l, k, a, c, elem);
        });
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ScoreCombineLowering : public RewritePattern {
  ScoreCombineLowering(MLIRContext *ctx)
      : RewritePattern("tessera.score_combine", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "static-shape ranked-tensor result required");
    Type elem = resultType.getElementType();
    auto floatElem = dyn_cast<FloatType>(elem);
    if (!floatElem)
      return rewriter.notifyMatchFailure(op, "floating tensor required");
    auto gammaAttr = op->getAttrOfType<FloatAttr>("gamma");
    double gamma = gammaAttr ? gammaAttr.getValueAsDouble() : 1.0;
    Value out = buildElementwiseGeneric(
        rewriter, op->getLoc(), resultType, op->getOperand(0),
        op->getOperand(1),
        [&](OpBuilder &b, Location l, Value base, Value delta) -> Value {
          Value g = arith::ConstantOp::create(b, l, elem,
                                              b.getFloatAttr(elem, gamma));
          Value scaled = arith::MulFOp::create(b, l, delta, g).getResult();
          return arith::AddFOp::create(b, l, base, scaled).getResult();
        });
    rewriter.replaceOp(op, out);
    return success();
  }
};

// N-ary elementwise `linalg.generic` over `inputs` (all same shape as
// `resultType`) into a fresh DPS init. Generalizes buildElementwiseGeneric.
static Value buildElementwiseNary(
    PatternRewriter &rewriter, Location loc, RankedTensorType resultType,
    ValueRange inputs,
    function_ref<Value(OpBuilder &, Location, ValueRange)> combine) {
  int64_t rank = resultType.getRank();
  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType());
  AffineMap id = rewriter.getMultiDimIdentityMap(rank);
  SmallVector<AffineMap> maps(inputs.size() + 1, id);
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      rewriter, loc, TypeRange{resultType}, inputs, ValueRange{init}, maps,
      iters, [&](OpBuilder &b, Location l, ValueRange args) {
        linalg::YieldOp::create(b, l, combine(b, l, args));
      });
  return generic.getResult(0);
}

// tessera.select(cond, a, b): elementwise `cond != 0 ? a : b`.
struct SelectLowering : public RewritePattern {
  SelectLowering(MLIRContext *ctx)
      : RewritePattern("tessera.select", /*benefit=*/1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return failure();
    auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!ty || !ty.hasStaticShape() || !isa<FloatType>(ty.getElementType()))
      return rewriter.notifyMatchFailure(op, "static-shape float tensor");
    Type elem = ty.getElementType();
    Value out = buildElementwiseNary(
        rewriter, op->getLoc(), ty,
        {op->getOperand(0), op->getOperand(1), op->getOperand(2)},
        [&](OpBuilder &b, Location l, ValueRange a) -> Value {
          Value zero = arith::ConstantOp::create(b, l, elem,
                                                 b.getFloatAttr(elem, 0.0));
          Value p = arith::CmpFOp::create(b, l, arith::CmpFPredicate::ONE,
                                          a[0], zero);
          return arith::SelectOp::create(b, l, p, a[1], a[2]).getResult();
        });
    rewriter.replaceOp(op, out);
    return success();
  }
};

// tessera.masked_fill(x, mask, {value}): elementwise `mask != 0 ? x : value`.
struct MaskedFillLowering : public RewritePattern {
  MaskedFillLowering(MLIRContext *ctx)
      : RewritePattern("tessera.masked_fill", /*benefit=*/1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!ty || !ty.hasStaticShape() || !isa<FloatType>(ty.getElementType()))
      return rewriter.notifyMatchFailure(op, "static-shape float tensor");
    auto valAttr = op->getAttrOfType<FloatAttr>("value");
    if (!valAttr)
      return rewriter.notifyMatchFailure(op, "missing value attr");
    Type elem = ty.getElementType();
    double value = valAttr.getValueAsDouble();
    Value out = buildElementwiseNary(
        rewriter, op->getLoc(), ty, {op->getOperand(0), op->getOperand(1)},
        [&](OpBuilder &b, Location l, ValueRange a) -> Value {
          Value zero = arith::ConstantOp::create(b, l, elem,
                                                 b.getFloatAttr(elem, 0.0));
          Value v = arith::ConstantOp::create(b, l, elem,
                                              b.getFloatAttr(elem, value));
          Value p = arith::CmpFOp::create(b, l, arith::CmpFPredicate::ONE,
                                          a[1], zero);
          return arith::SelectOp::create(b, l, p, a[0], v).getResult();
        });
    rewriter.replaceOp(op, out);
    return success();
  }
};

// tessera.write_row(buffer (T,D), value (1,D), {row}) -> buffer with row `row`
// set to value. Lowers to tensor.insert_slice (value-semantic; bufferization may
// place it in-place). This is the functional KV-cache update primitive.
struct WriteRowLowering : public RewritePattern {
  WriteRowLowering(MLIRContext *ctx)
      : RewritePattern("tessera.write_row", /*benefit=*/1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    auto bufTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto valTy = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!bufTy || !valTy || !outTy || bufTy != outTy ||
        !bufTy.hasStaticShape() || !valTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "static-shape, result==buffer");
    if (bufTy.getRank() != 2 || valTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "write_row is rank-2 (T,D)<-(1,D)");
    int64_t T = bufTy.getDimSize(0), D = bufTy.getDimSize(1);
    if (valTy.getDimSize(0) != 1 || valTy.getDimSize(1) != D)
      return rewriter.notifyMatchFailure(op, "value must be (1, D)");
    auto rowAttr = op->getAttrOfType<IntegerAttr>("row");
    if (!rowAttr)
      return rewriter.notifyMatchFailure(op, "missing row attr");
    int64_t row = rowAttr.getInt();
    if (row < 0 || row >= T)
      return rewriter.notifyMatchFailure(op, "row out of range");

    Location loc = op->getLoc();
    SmallVector<OpFoldResult> offsets = {rewriter.getIndexAttr(row),
                                         rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1),
                                       rewriter.getIndexAttr(D)};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};
    Value out = tensor::InsertSliceOp::create(rewriter, loc, op->getOperand(1),
                                              op->getOperand(0), offsets, sizes,
                                              strides);
    rewriter.replaceOp(op, out);
    return success();
  }
};

// ── transpose / matmul ─────────────────────────────────────────────────────

// Rank-2 transpose via linalg.transpose (DPS). Used both standalone
// (tessera.transpose) and by matmul to materialize transposed operands.
static Value emitTranspose2d(PatternRewriter &rewriter, Location loc, Value in) {
  auto ty = cast<RankedTensorType>(in.getType());
  ArrayRef<int64_t> sh = ty.getShape();
  SmallVector<int64_t> outShape = {sh[1], sh[0]};
  Value init =
      tensor::EmptyOp::create(rewriter, loc, outShape, ty.getElementType());
  return linalg::TransposeOp::create(rewriter, loc, in, init,
                                     ArrayRef<int64_t>{1, 0})
      .getResults()[0];
}

// Rank-2; transposeA/transposeB supported by transposing the operand first;
// bf16/f16 accumulate in f32 then truncf. linalg.matmul takes (lhs: MxK,
// rhs: KxN) → out: MxN with accumulation into a zero-filled DPS init.
struct MatmulLowering : public RewritePattern {
  MatmulLowering(MLIRContext *ctx)
      : RewritePattern("tessera.matmul", /*benefit=*/1, ctx) {}

  static bool attrIsSetTrue(Operation *op, StringRef name) {
    if (auto a = op->getAttrOfType<BoolAttr>(name))
      return a.getValue();
    return false;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();

    auto lhsTy0 = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhsTy0 = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!lhsTy0 || !rhsTy0 || !outTy)
      return failure();
    if (lhsTy0.getRank() != 2 || rhsTy0.getRank() != 2 || outTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "Phase 1 matmul is rank-2 only");
    if (!lhsTy0.hasStaticShape() || !rhsTy0.hasStaticShape() ||
        !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "static shapes required");

    Location loc = op->getLoc();
    // transposeA: A stored KxM; transposeB: B stored NxK. Materialize the
    // transpose, then a plain matmul. (The op verifier already checked the
    // post-transpose contracting dims agree.)
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    if (attrIsSetTrue(op, "transposeA"))
      lhs = emitTranspose2d(rewriter, loc, lhs);
    if (attrIsSetTrue(op, "transposeB"))
      rhs = emitTranspose2d(rewriter, loc, rhs);
    auto lhsTy = cast<RankedTensorType>(lhs.getType());
    auto rhsTy = cast<RankedTensorType>(rhs.getType());

    int64_t M = lhsTy.getDimSize(0), K = lhsTy.getDimSize(1);
    int64_t K2 = rhsTy.getDimSize(0), N = rhsTy.getDimSize(1);
    if (K != K2 || outTy.getDimSize(0) != M || outTy.getDimSize(1) != N)
      return rewriter.notifyMatchFailure(op, "matmul shape mismatch");

    Type elem = outTy.getElementType();
    if (!isa<FloatType>(elem))
      return rewriter.notifyMatchFailure(op, "Phase 1 matmul is float-only");

    // ABI §12.5 / numeric policy: low-precision storage matmul accumulates in
    // f32. For bf16/f16 we run linalg.matmul into an f32 init (the named op
    // casts the low-precision inputs up), then truncate the result back to the
    // storage dtype. f32 storage accumulates in f32 directly.
    bool lowPrecision = elem.isBF16() || elem.isF16();
    Type accElem = lowPrecision ? rewriter.getF32Type() : elem;
    auto accTy = RankedTensorType::get(outTy.getShape(), accElem);

    Value empty =
        tensor::EmptyOp::create(rewriter, loc, outTy.getShape(), accElem);
    Value zero = arith::ConstantOp::create(rewriter, loc, accElem,
                                           rewriter.getFloatAttr(accElem, 0.0));
    Value filled = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                          ValueRange{empty})
                       .getResult(0);
    Value acc = linalg::MatmulOp::create(rewriter, loc, TypeRange{accTy},
                                         ValueRange{lhs, rhs},
                                         ValueRange{filled})
                    .getResult(0);

    Value out = acc;
    if (lowPrecision) {
      // truncf f32 accumulator -> storage dtype (bf16/f16).
      out = emitUnaryElementwise(
          rewriter, loc, outTy, acc,
          [&](OpBuilder &b, Location l, Value a) -> Value {
            return arith::TruncFOp::create(b, l, elem, a).getResult();
          });
    }
    rewriter.replaceOp(op, out);
    return success();
  }
};

// tessera.batched_gemm: C[b] = A[b] @ B[b], rank-3 → linalg.batch_matmul.
// Same f32-accumulate-for-low-precision policy as matmul. No transposes.
struct BatchedGemmLowering : public RewritePattern {
  BatchedGemmLowering(MLIRContext *ctx)
      : RewritePattern("tessera.batched_gemm", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    auto lhsTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhsTy = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!lhsTy || !rhsTy || !outTy)
      return failure();
    if (lhsTy.getRank() != 3 || rhsTy.getRank() != 3 || outTy.getRank() != 3)
      return rewriter.notifyMatchFailure(op, "batched_gemm is rank-3");
    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() ||
        !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "static shapes required");

    int64_t Bb = lhsTy.getDimSize(0), M = lhsTy.getDimSize(1),
            K = lhsTy.getDimSize(2);
    int64_t Bb2 = rhsTy.getDimSize(0), K2 = rhsTy.getDimSize(1),
            N = rhsTy.getDimSize(2);
    if (Bb != Bb2 || K != K2 || outTy.getDimSize(0) != Bb ||
        outTy.getDimSize(1) != M || outTy.getDimSize(2) != N)
      return rewriter.notifyMatchFailure(op, "batched_gemm shape mismatch");

    Type elem = outTy.getElementType();
    if (!isa<FloatType>(elem))
      return rewriter.notifyMatchFailure(op, "float-only");

    Location loc = op->getLoc();
    bool lowPrecision = elem.isBF16() || elem.isF16();
    Type accElem = lowPrecision ? rewriter.getF32Type() : elem;
    auto accTy = RankedTensorType::get(outTy.getShape(), accElem);

    Value empty =
        tensor::EmptyOp::create(rewriter, loc, outTy.getShape(), accElem);
    Value zero = arith::ConstantOp::create(rewriter, loc, accElem,
                                           rewriter.getFloatAttr(accElem, 0.0));
    Value filled = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                          ValueRange{empty})
                       .getResult(0);
    Value acc = linalg::BatchMatmulOp::create(
                    rewriter, loc, TypeRange{accTy},
                    ValueRange{op->getOperand(0), op->getOperand(1)},
                    ValueRange{filled})
                    .getResult(0);

    Value out = acc;
    if (lowPrecision)
      out = emitUnaryElementwise(
          rewriter, loc, outTy, acc,
          [&](OpBuilder &b, Location l, Value a) -> Value {
            return arith::TruncFOp::create(b, l, elem, a).getResult();
          });
    rewriter.replaceOp(op, out);
    return success();
  }
};

// ── tessera.reduce ─────────────────────────────────────────────────────────

// Elementwise scale `input * scalar` into a fresh DPS init of `ty`. Used by the
// `mean` reduction (sum then multiply by 1/N). The scalar is captured into the
// linalg.generic body directly (allowed — it's an outer SSA value).
static Value buildUnaryScale(PatternRewriter &rewriter, Location loc,
                             RankedTensorType ty, Value input, Value scalar) {
  int64_t rank = ty.getRank();
  Value init = tensor::EmptyOp::create(rewriter, loc, ty.getShape(),
                                       ty.getElementType());
  AffineMap id = rewriter.getMultiDimIdentityMap(rank);
  SmallVector<AffineMap, 2> maps = {id, id};
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      rewriter, loc, TypeRange{ty}, ValueRange{input}, ValueRange{init}, maps,
      iters, [&](OpBuilder &b, Location l, ValueRange args) {
        Value m = arith::MulFOp::create(b, l, args[0], scalar);
        linalg::YieldOp::create(b, l, m);
      });
  return generic.getResult(0);
}

// Unary elementwise `linalg.generic` applying `fn` over `input` into a fresh DPS
// init of `ty`.
static Value emitUnaryElementwise(
    PatternRewriter &rewriter, Location loc, RankedTensorType ty, Value input,
    function_ref<Value(OpBuilder &, Location, Value)> fn) {
  int64_t rank = ty.getRank();
  Value init = tensor::EmptyOp::create(rewriter, loc, ty.getShape(),
                                       ty.getElementType());
  AffineMap id = rewriter.getMultiDimIdentityMap(rank);
  SmallVector<AffineMap, 2> maps = {id, id};
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      rewriter, loc, TypeRange{ty}, ValueRange{input}, ValueRange{init}, maps,
      iters, [&](OpBuilder &b, Location l, ValueRange args) {
        linalg::YieldOp::create(b, l, fn(b, l, args[0]));
      });
  return generic.getResult(0);
}

// Reduce `input` (a full-rank tensor) over a single `axis` with kind
// sum/max/min, returning the rank-(R-1) reduced tensor. Identity-fill +
// linalg.reduce. (mean is handled by the caller as sum + scale.)
static Value emitReduceCore(PatternRewriter &rewriter, Location loc,
                            RankedTensorType inTy, Value input, int64_t axis,
                            StringRef kind) {
  Type elem = inTy.getElementType();
  auto fty = cast<FloatType>(elem);
  int64_t rank = inTy.getRank();
  SmallVector<int64_t> outShape;
  for (int64_t i = 0; i < rank; ++i)
    if (i != axis)
      outShape.push_back(inTy.getDimSize(i));

  const llvm::fltSemantics &sem = fty.getFloatSemantics();
  APFloat ident = APFloat::getZero(sem);
  if (kind == "max")
    ident = APFloat::getInf(sem, /*Negative=*/true);
  else if (kind == "min")
    ident = APFloat::getInf(sem, /*Negative=*/false);

  Value empty = tensor::EmptyOp::create(rewriter, loc, outShape, elem);
  Value identC = arith::ConstantOp::create(rewriter, loc, elem,
                                           rewriter.getFloatAttr(elem, ident));
  Value filled = linalg::FillOp::create(rewriter, loc, ValueRange{identC},
                                        ValueRange{empty})
                     .getResult(0);
  auto reduceOp = linalg::ReduceOp::create(
      rewriter, loc, ValueRange{input}, ValueRange{filled},
      ArrayRef<int64_t>{axis}, [&](OpBuilder &b, Location l, ValueRange args) {
        Value in = args[0], acc = args[1], r;
        if (kind == "sum")
          r = arith::AddFOp::create(b, l, in, acc);
        else if (kind == "max")
          r = arith::MaximumFOp::create(b, l, in, acc);
        else
          r = arith::MinimumFOp::create(b, l, in, acc);
        linalg::YieldOp::create(b, l, r);
      });
  return reduceOp.getResults()[0];
}

// Broadcast-binary: `fn(full[i...], reduced[i... without axis])` over the full
// iteration space, into a fresh DPS init of `fullTy`. The reduced operand uses
// an affine map that drops `axis`, so it broadcasts along that dim. This is the
// load-bearing new capability of Sprint 1.3 — softmax's `x - max` and `e / sum`,
// and (later) normalization's centering/scaling, all route through it.
static Value emitBroadcastBinary(
    PatternRewriter &rewriter, Location loc, RankedTensorType fullTy, Value full,
    Value reduced, int64_t axis,
    function_ref<Value(OpBuilder &, Location, Value, Value)> fn) {
  MLIRContext *ctx = fullTy.getContext();
  int64_t rank = fullTy.getRank();
  Value init = tensor::EmptyOp::create(rewriter, loc, fullTy.getShape(),
                                       fullTy.getElementType());
  AffineMap idFull = rewriter.getMultiDimIdentityMap(rank);
  SmallVector<AffineExpr> exprs;
  for (int64_t d = 0; d < rank; ++d)
    if (d != axis)
      exprs.push_back(getAffineDimExpr(d, ctx));
  AffineMap redMap = AffineMap::get(rank, /*symbolCount=*/0, exprs, ctx);
  SmallVector<AffineMap, 3> maps = {idFull, redMap, idFull};
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      rewriter, loc, TypeRange{fullTy}, ValueRange{full, reduced},
      ValueRange{init}, maps, iters,
      [&](OpBuilder &b, Location l, ValueRange args) {
        linalg::YieldOp::create(b, l, fn(b, l, args[0], args[1]));
      });
  return generic.getResult(0);
}

// Mean of `input` over `axis`: sum reduction scaled by 1/N. Returns the
// rank-(R-1) reduced tensor. Used by both norms (rmsnorm: mean(x²);
// layer_norm: mean(x) and mean((x-mu)²)).
static Value emitMean(PatternRewriter &rewriter, Location loc,
                      RankedTensorType inTy, Value input, int64_t axis) {
  Value summed = emitReduceCore(rewriter, loc, inTy, input, axis, "sum");
  auto redTy = cast<RankedTensorType>(summed.getType());
  Type elem = inTy.getElementType();
  double n = static_cast<double>(inTy.getDimSize(axis));
  Value recip = arith::ConstantOp::create(rewriter, loc, elem,
                                          rewriter.getFloatAttr(elem, 1.0 / n));
  return buildUnaryScale(rewriter, loc, redTy, summed, recip);
}

// Single-axis reduction (sum/max/min/mean) → linalg.reduce over `axis`, with a
// fresh init pre-filled with the reduction identity (0 / -inf / +inf). `mean` is
// `sum` followed by a 1/N scale. Result rank = input rank - 1.
struct ReduceLowering : public RewritePattern {
  ReduceLowering(MLIRContext *ctx)
      : RewritePattern("tessera.reduce", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    auto inTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "static-shape tensors required");
    Type elem = inTy.getElementType();
    auto fty = dyn_cast<FloatType>(elem);
    if (!fty || elem != outTy.getElementType())
      return rewriter.notifyMatchFailure(op, "float-only, matching elem types");

    auto kindAttr = op->getAttrOfType<StringAttr>("kind");
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (!kindAttr || !axisAttr)
      return rewriter.notifyMatchFailure(op, "missing kind/axis attrs");
    int64_t rank = inTy.getRank();
    int64_t axis = axisAttr.getInt();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range");

    // Result shape must be the input shape with `axis` removed.
    SmallVector<int64_t> expected;
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis)
        expected.push_back(inTy.getDimSize(i));
    if (outTy.getShape() != ArrayRef<int64_t>(expected))
      return rewriter.notifyMatchFailure(op, "result shape != input minus axis");

    StringRef kind = kindAttr.getValue();
    bool isMean = kind == "mean";
    StringRef redKind = isMean ? StringRef("sum") : kind;
    if (redKind != "sum" && redKind != "max" && redKind != "min")
      return rewriter.notifyMatchFailure(op, "unknown reduce kind");

    Location loc = op->getLoc();
    Value reduced =
        emitReduceCore(rewriter, loc, inTy, op->getOperand(0), axis, redKind);

    if (isMean) {
      double n = static_cast<double>(inTy.getDimSize(axis));
      Value recip = arith::ConstantOp::create(
          rewriter, loc, elem, rewriter.getFloatAttr(elem, 1.0 / n));
      reduced = buildUnaryScale(rewriter, loc, outTy, reduced, recip);
    }
    rewriter.replaceOp(op, reduced);
    return success();
  }
};

// tessera.softmax (over `axis`, default innermost) → numerically-stable
// decomposition:  m = max(x); e = exp(x - m); y = e / sum(e).  Composes the
// reduction machinery (emitReduceCore) with broadcast-binary (emitBroadcastBinary)
// and a math.exp unary. Result shape == input shape.
struct SoftmaxLowering : public RewritePattern {
  SoftmaxLowering(MLIRContext *ctx)
      : RewritePattern("tessera.softmax", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    auto ty = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!ty || !outTy || ty != outTy || !ty.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "static same-shape tensor required");
    if (!isa<FloatType>(ty.getElementType()))
      return rewriter.notifyMatchFailure(op, "float-only");

    int64_t rank = ty.getRank();
    int64_t axis = rank - 1; // default: innermost
    if (auto a = op->getAttrOfType<IntegerAttr>("axis")) {
      axis = a.getInt();
      if (axis < 0)
        axis += rank;
    }
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range");

    Location loc = op->getLoc();
    Value x = op->getOperand(0);
    Value m = emitReduceCore(rewriter, loc, ty, x, axis, "max");
    Value shifted = emitBroadcastBinary(
        rewriter, loc, ty, x, m, axis,
        [](OpBuilder &b, Location l, Value a, Value c) -> Value {
          return arith::SubFOp::create(b, l, a, c).getResult();
        });
    Value e = emitUnaryElementwise(
        rewriter, loc, ty, shifted,
        [](OpBuilder &b, Location l, Value a) -> Value {
          return math::ExpOp::create(b, l, a).getResult();
        });
    Value denom = emitReduceCore(rewriter, loc, ty, e, axis, "sum");
    Value y = emitBroadcastBinary(
        rewriter, loc, ty, e, denom, axis,
        [](OpBuilder &b, Location l, Value a, Value c) -> Value {
          return arith::DivFOp::create(b, l, a, c).getResult();
        });
    rewriter.replaceOp(op, y);
    return success();
  }
};

// ── Normalization ──────────────────────────────────────────────────────────

// Shared helpers for the norms: read `eps` (default 1e-5), square, add-eps,
// sqrt — all reductions/broadcasts over the innermost axis.
static double readEps(Operation *op) {
  if (auto e = op->getAttrOfType<FloatAttr>("eps"))
    return e.getValueAsDouble();
  return 1e-5;
}
static Value emitSquare(PatternRewriter &r, Location loc, RankedTensorType ty,
                        Value v) {
  return emitUnaryElementwise(
      r, loc, ty, v, [](OpBuilder &b, Location l, Value a) -> Value {
        return arith::MulFOp::create(b, l, a, a).getResult();
      });
}
static Value emitAddEpsThenSqrt(PatternRewriter &r, Location loc,
                                RankedTensorType redTy, Value reduced,
                                Type elem, double eps) {
  Value epsC = arith::ConstantOp::create(r, loc, elem,
                                         r.getFloatAttr(elem, eps));
  Value withEps = emitUnaryElementwise(
      r, loc, redTy, reduced, [&](OpBuilder &b, Location l, Value a) -> Value {
        return arith::AddFOp::create(b, l, a, epsC).getResult();
      });
  // Precise IEEE sqrt (not rsqrt) so `x / sqrt(...)` matches the numpy oracle.
  return emitUnaryElementwise(
      r, loc, redTy, withEps, [](OpBuilder &b, Location l, Value a) -> Value {
        return math::SqrtOp::create(b, l, a).getResult();
      });
}
static Value emitDivBroadcast(PatternRewriter &r, Location loc,
                              RankedTensorType ty, Value full, Value reducedDenom,
                              int64_t axis) {
  return emitBroadcastBinary(
      r, loc, ty, full, reducedDenom, axis,
      [](OpBuilder &b, Location l, Value a, Value c) -> Value {
        return arith::DivFOp::create(b, l, a, c).getResult();
      });
}

// rmsnorm(x) = x / sqrt(mean(x²) + eps), over the innermost axis (unweighted).
struct RmsNormLowering : public RewritePattern {
  RmsNormLowering(MLIRContext *ctx)
      : RewritePattern("tessera.rmsnorm", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    auto ty = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!ty || !outTy || ty != outTy || !ty.hasStaticShape() || ty.getRank() < 1)
      return rewriter.notifyMatchFailure(op, "static same-shape rank>=1 tensor");
    if (!isa<FloatType>(ty.getElementType()))
      return rewriter.notifyMatchFailure(op, "float-only");

    int64_t axis = ty.getRank() - 1;
    double eps = readEps(op);
    Location loc = op->getLoc();
    Type elem = ty.getElementType();
    Value x = op->getOperand(0);

    Value ms = emitMean(rewriter, loc, ty, emitSquare(rewriter, loc, ty, x), axis);
    auto redTy = cast<RankedTensorType>(ms.getType());
    Value denom = emitAddEpsThenSqrt(rewriter, loc, redTy, ms, elem, eps);
    Value y = emitDivBroadcast(rewriter, loc, ty, x, denom, axis);
    rewriter.replaceOp(op, y);
    return success();
  }
};

// layer_norm(x) = (x - mean) / sqrt(var + eps), over the innermost axis
// (unweighted; no gamma/beta).
struct LayerNormLowering : public RewritePattern {
  LayerNormLowering(MLIRContext *ctx)
      : RewritePattern("tessera.layer_norm", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    auto ty = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!ty || !outTy || ty != outTy || !ty.hasStaticShape() || ty.getRank() < 1)
      return rewriter.notifyMatchFailure(op, "static same-shape rank>=1 tensor");
    if (!isa<FloatType>(ty.getElementType()))
      return rewriter.notifyMatchFailure(op, "float-only");

    int64_t axis = ty.getRank() - 1;
    double eps = readEps(op);
    Location loc = op->getLoc();
    Type elem = ty.getElementType();
    Value x = op->getOperand(0);

    Value mu = emitMean(rewriter, loc, ty, x, axis);
    Value centered = emitBroadcastBinary(
        rewriter, loc, ty, x, mu, axis,
        [](OpBuilder &b, Location l, Value a, Value c) -> Value {
          return arith::SubFOp::create(b, l, a, c).getResult();
        });
    Value var =
        emitMean(rewriter, loc, ty, emitSquare(rewriter, loc, ty, centered), axis);
    auto redTy = cast<RankedTensorType>(var.getType());
    Value denom = emitAddEpsThenSqrt(rewriter, loc, redTy, var, elem, eps);
    Value y = emitDivBroadcast(rewriter, loc, ty, centered, denom, axis);
    rewriter.replaceOp(op, y);
    return success();
  }
};

// ── Activations (unary math family) ────────────────────────────────────────

enum class ActKind { Relu, Sigmoid, Tanh, Silu, Gelu };

// Per-scalar activation body. All float; uses math.{exp,tanh} (lowered via
// convert-math-to-llvm) + arith. gelu is the tanh approximation (GPT-2/BERT
// form) — avoids math.erf, which has no standard LLVM-intrinsic lowering.
static Value emitActScalar(OpBuilder &b, Location loc, ActKind kind, Value x,
                           Type elem) {
  auto cst = [&](double v) -> Value {
    return arith::ConstantOp::create(b, loc, elem, b.getFloatAttr(elem, v))
        .getResult();
  };
  switch (kind) {
  case ActKind::Relu:
    return arith::MaximumFOp::create(b, loc, x, cst(0.0)).getResult();
  case ActKind::Tanh:
    return math::TanhOp::create(b, loc, x).getResult();
  case ActKind::Sigmoid: {
    Value negx = arith::NegFOp::create(b, loc, x).getResult();
    Value e = math::ExpOp::create(b, loc, negx).getResult();
    Value one = cst(1.0);
    Value denom = arith::AddFOp::create(b, loc, one, e).getResult();
    return arith::DivFOp::create(b, loc, one, denom).getResult();
  }
  case ActKind::Silu: {
    Value sig = emitActScalar(b, loc, ActKind::Sigmoid, x, elem);
    return arith::MulFOp::create(b, loc, x, sig).getResult();
  }
  case ActKind::Gelu: {
    // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x³) ))
    Value x2 = arith::MulFOp::create(b, loc, x, x).getResult();
    Value x3 = arith::MulFOp::create(b, loc, x2, x).getResult();
    Value inner = arith::AddFOp::create(
                      b, loc, x,
                      arith::MulFOp::create(b, loc, cst(0.044715), x3)
                          .getResult())
                      .getResult();
    Value scaled =
        arith::MulFOp::create(b, loc, cst(0.7978845608028654), inner)
            .getResult();
    Value t = math::TanhOp::create(b, loc, scaled).getResult();
    Value onePlus = arith::AddFOp::create(b, loc, cst(1.0), t).getResult();
    Value hx = arith::MulFOp::create(b, loc, cst(0.5), x).getResult();
    return arith::MulFOp::create(b, loc, hx, onePlus).getResult();
  }
  }
  llvm_unreachable("unhandled ActKind");
}

struct UnaryActLowering : public RewritePattern {
  ActKind kind;
  UnaryActLowering(MLIRContext *ctx, StringRef opName, ActKind k)
      : RewritePattern(opName, /*benefit=*/1, ctx), kind(k) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!ty || !ty.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "static-shape tensor required");
    Type elem = ty.getElementType();
    if (!isa<FloatType>(elem))
      return rewriter.notifyMatchFailure(op, "float-only");
    ActKind k = kind;
    Value out = emitUnaryElementwise(
        rewriter, op->getLoc(), ty, op->getOperand(0),
        [&](OpBuilder &b, Location l, Value a) -> Value {
          return emitActScalar(b, l, k, a, elem);
        });
    rewriter.replaceOp(op, out);
    return success();
  }
};

// tessera.transpose (rank-2; the result type fixes the permutation as [1,0]).
struct TransposeLowering : public RewritePattern {
  TransposeLowering(MLIRContext *ctx)
      : RewritePattern("tessera.transpose", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    auto inTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "static-shape tensors required");
    if (inTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "Phase 1 transpose is rank-2 only (op has no permutation attr)");
    if (outTy.getDimSize(0) != inTy.getDimSize(1) ||
        outTy.getDimSize(1) != inTy.getDimSize(0))
      return rewriter.notifyMatchFailure(op, "result must be the [1,0] transpose");
    rewriter.replaceOp(op,
                       emitTranspose2d(rewriter, op->getLoc(), op->getOperand(0)));
    return success();
  }
};

// ── Pass ───────────────────────────────────────────────────────────────────

class TesseraToLinalgPass
    : public PassWrapper<TesseraToLinalgPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TesseraToLinalgPass)

  StringRef getArgument() const override { return "tessera-to-linalg"; }
  StringRef getDescription() const override {
    return "Lower total elementwise + matmul Tessera Graph IR ops to upstream "
           "linalg on tensors (production spine, Phases 0-1)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    tensor::TensorDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<BinaryEltwiseLowering>(ctx, "tessera.add", BinaryKind::Add);
    patterns.add<BinaryEltwiseLowering>(ctx, "tessera.sub", BinaryKind::Sub);
    patterns.add<BinaryEltwiseLowering>(ctx, "tessera.mul", BinaryKind::Mul);
    patterns.add<BinaryEltwiseLowering>(ctx, "tessera.div", BinaryKind::Div);
    patterns.add<ScoreCombineLowering>(ctx);
    patterns.add<SelectLowering>(ctx);
    patterns.add<MaskedFillLowering>(ctx);
    patterns.add<WriteRowLowering>(ctx);
    patterns.add<MatmulLowering>(ctx);
    patterns.add<BatchedGemmLowering>(ctx);
    patterns.add<TransposeLowering>(ctx);
    patterns.add<ReduceLowering>(ctx);
    patterns.add<SoftmaxLowering>(ctx);
    patterns.add<RmsNormLowering>(ctx);
    patterns.add<LayerNormLowering>(ctx);
    patterns.add<UnaryActLowering>(ctx, "tessera.relu", ActKind::Relu);
    patterns.add<UnaryActLowering>(ctx, "tessera.sigmoid", ActKind::Sigmoid);
    patterns.add<UnaryActLowering>(ctx, "tessera.tanh", ActKind::Tanh);
    patterns.add<UnaryActLowering>(ctx, "tessera.silu", ActKind::Silu);
    patterns.add<UnaryActLowering>(ctx, "tessera.gelu", ActKind::Gelu);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createTesseraToLinalgPass() {
  return std::make_unique<TesseraToLinalgPass>();
}
} // namespace tessera
