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
//   Phase 1: tessera.{sub,mul} (elementwise via shared table)
//            tessera.matmul (linalg.fill + linalg.matmul; rank-2, no transpose)
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// ── Binary elementwise ─────────────────────────────────────────────────────

enum class BinaryKind { Add, Sub, Mul };

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

// ── tessera.matmul ─────────────────────────────────────────────────────────

// Rank-2 only; no transposes; accumulator policy ignored for now (Phase 1
// scope). linalg.matmul takes (lhs: MxK, rhs: KxN) → out: MxN with
// accumulation, so we zero-fill the DPS init first.
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
    if (attrIsSetTrue(op, "transposeA") || attrIsSetTrue(op, "transposeB"))
      return rewriter.notifyMatchFailure(
          op, "Phase 1 does not yet support transposeA/transposeB");

    auto lhsTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhsTy = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!lhsTy || !rhsTy || !outTy)
      return failure();
    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || outTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "Phase 1 matmul is rank-2 only");
    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() ||
        !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "static shapes required");

    int64_t M = lhsTy.getDimSize(0), K = lhsTy.getDimSize(1);
    int64_t K2 = rhsTy.getDimSize(0), N = rhsTy.getDimSize(1);
    if (K != K2 || outTy.getDimSize(0) != M || outTy.getDimSize(1) != N)
      return rewriter.notifyMatchFailure(op, "matmul shape mismatch");

    Type elem = outTy.getElementType();
    if (!isa<FloatType>(elem))
      return rewriter.notifyMatchFailure(
          op, "Phase 1 matmul is float-only (i.e. f32)");

    Location loc = op->getLoc();
    Value empty = tensor::EmptyOp::create(rewriter, loc, outTy.getShape(), elem);
    auto zeroAttr = rewriter.getFloatAttr(elem, 0.0);
    Value zero = arith::ConstantOp::create(rewriter, loc, elem, zeroAttr);
    Value filled = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                          ValueRange{empty})
                       .getResult(0);
    Value out = linalg::MatmulOp::create(
                    rewriter, loc, TypeRange{outTy},
                    ValueRange{op->getOperand(0), op->getOperand(1)},
                    ValueRange{filled})
                    .getResult(0);
    rewriter.replaceOp(op, out);
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
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<BinaryEltwiseLowering>(ctx, "tessera.add", BinaryKind::Add);
    patterns.add<BinaryEltwiseLowering>(ctx, "tessera.sub", BinaryKind::Sub);
    patterns.add<BinaryEltwiseLowering>(ctx, "tessera.mul", BinaryKind::Mul);
    patterns.add<MatmulLowering>(ctx);
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
