//===- MatmulToAppleSimdgroup.cpp - Apple GPU machine-primitive matmul ---===//
//
// Lowers `tessera.matmul` to the Apple GPU **machine primitives** rather than
// to a call into the hand-written runtime shim.
//
// Why this pass exists
// --------------------
// `MatmulToAppleGPU` -- the existing lowering -- emits a `func.call` to
// `tessera_apple_gpu_mps_matmul_*`. The MLIR pipeline therefore never says
// what an Apple matmul *is*; it names a symbol and the kernel lives in
// `apple_gpu_runtime.mm`. That is the seam CLAUDE.md records for Apple: "the
// Python synthesizer and the C++ MLIR pipeline are two disconnected
// compilers."
//
// This pass emits `simdgroup_fill` / `simdgroup_load` / `simdgroup_matmul` /
// `simdgroup_store` inside an `scf.for` nest, so the accumulation is
// expressed in IR and is visible to every downstream analysis. It is also
// what gives those ops a *producer*: Decision #29 requires a declaration to
// have a consumer, and its sequencing corollary requires an op to land with
// the pass that emits it. A verifier alone is not that.
//
// Scope, stated rather than implied
// ---------------------------------
// This is the register-level core, not the whole coopmat kernel. It emits the
// tile nest and the MMA chain; it does NOT yet emit threadgroup staging or
// the cooperative K-slab copy that `emit/apple_msl.py` performs, so it is not
// a performance replacement for the MPS lane and does not remove it. What it
// establishes is that the pipeline can express the computation.
//
// The index arithmetic is the part worth checking, because it is where a
// tiled lowering silently computes a different matrix:
//
//   A[m, k] -> offset m*K + k, row stride K
//   B[k, n] -> offset k*N + n, row stride N
//   C[m, n] -> offset m*N + n, row stride N
//
// `tests/unit/test_apple_simdgroup_contract.py` checks that decomposition
// against a reference matmul numerically rather than by inspection.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/LoweringUtils.h"
#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace tessera {
namespace apple {
namespace {

/// Apple7's only simdgroup-matrix extent. Every dimension must be a multiple
/// of this or the tail would need masking the primitives do not model yet.
constexpr int64_t kExtent = 8;

struct LowerMatmulToAppleSimdgroup : public RewritePattern {
  LowerMatmulToAppleSimdgroup(MLIRContext *ctx)
      : RewritePattern("tessera.matmul", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumResults() != 1)
      return failure();
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsTy || !rhsTy || lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
      return failure();
    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "requires static shapes");

    Type elem = lhsTy.getElementType();
    if (elem != rhsTy.getElementType())
      return rewriter.notifyMatchFailure(op, "mixed input element types");
    // The MMA storage set: an f64 or integer matmul has no simdgroup form.
    if (!elem.isF16() && !elem.isF32())
      return rewriter.notifyMatchFailure(op, "simdgroup MMA needs f16 or f32");

    const int64_t M = lhsTy.getDimSize(0), K = lhsTy.getDimSize(1);
    const int64_t N = rhsTy.getDimSize(1);
    if (rhsTy.getDimSize(0) != K)
      return rewriter.notifyMatchFailure(op, "matmul shape mismatch");
    // Ragged tails would need predicated loads. Declining is correct: the MPS
    // lane still serves those shapes, and emitting an unmasked nest for them
    // would read out of bounds.
    if (M % kExtent || N % kExtent || K % kExtent)
      return rewriter.notifyMatchFailure(
          op, "every extent must be a multiple of 8 until tail masking lands");

    // The accumulator is f32, and simdgroup_store does not convert -- storing
    // it into an f16 buffer would reinterpret bits, not round values. The MSL
    // kernel handles this by storing to a `threadgroup float` tile and
    // converting in the epilogue. Until this pass emits that epilogue it
    // declines an f16 result rather than emitting the reinterpretation; the
    // MPS lane still serves those. Found by building the pass and reading what
    // it produced, not by a failing test.
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resTy || !resTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          op, "f32 result required: the fp32 accumulator has no convert "
              "epilogue in this pass yet");

    Location loc = op->getLoc();
    auto lhsMem = MemRefType::get({M, K}, elem);
    auto rhsMem = MemRefType::get({K, N}, elem);
    auto outMem = MemRefType::get({M, N}, rewriter.getF32Type());

    Value aBuf = rewriter.create<bufferization::ToBufferOp>(loc, lhsMem, lhs);
    Value bBuf = rewriter.create<bufferization::ToBufferOp>(loc, rhsMem, rhs);
    Value cBuf = rewriter.create<memref::AllocOp>(loc, outMem);

    Type f32 = rewriter.getF32Type();
    auto matTy = [&](Type t) { return SimdgroupMatrixType::get(getContext(), t); };
    // The accumulator is f32 whatever the inputs are -- the simdgroup MMA's
    // fixed numerical contract, enforced by SimdgroupMatmulOp::verify.
    Type accTy = matTy(f32);
    Type inTy = matTy(elem);
    StringRef storage = elem.isF16() ? "f16" : "f32";

    auto idx = [&](int64_t v) {
      return rewriter.create<arith::ConstantIndexOp>(loc, v).getResult();
    };
    Value zero = idx(0), step = idx(kExtent);
    Value mBound = idx(M), nBound = idx(N), kBound = idx(K);
    Value kStride = idx(K), nStride = idx(N);

    // for m in 0..M step 8 { for n in 0..N step 8 { ... } }
    auto mLoop = rewriter.create<scf::ForOp>(loc, zero, mBound, step);
    rewriter.setInsertionPointToStart(mLoop.getBody());
    auto nLoop = rewriter.create<scf::ForOp>(loc, zero, nBound, step);
    rewriter.setInsertionPointToStart(nLoop.getBody());

    Value m = mLoop.getInductionVar(), n = nLoop.getInductionVar();
    Value acc0 = rewriter.create<SimdgroupFillOp>(
        loc, accTy, rewriter.getF32FloatAttr(0.0f));

    // acc = for k in 0..K step 8 iter_args(acc) { acc = a*b + acc }
    //
    // The accumulator is an iteration argument rather than a memory cell: the
    // dependence between K steps is then explicit in the IR, which is what
    // lets a later pass reorder or split K without having to prove anything
    // about aliasing.
    auto kLoop = rewriter.create<scf::ForOp>(loc, zero, kBound, step,
                                             ValueRange{acc0});
    rewriter.setInsertionPointToStart(kLoop.getBody());
    Value k = kLoop.getInductionVar();
    Value accIn = kLoop.getRegionIterArgs()[0];

    // A[m, k] at m*K + k; B[k, n] at k*N + n.
    Value aOff = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, m, kStride), k);
    Value bOff = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, k, nStride), n);

    auto aMat = rewriter.create<SimdgroupLoadOp>(
        loc, inTy, aBuf, aOff, rewriter.getI64IntegerAttr(K),
        rewriter.getStringAttr("threadgroup"));
    auto bMat = rewriter.create<SimdgroupLoadOp>(
        loc, inTy, bBuf, bOff, rewriter.getI64IntegerAttr(N),
        rewriter.getStringAttr("threadgroup"));
    auto mma = rewriter.create<SimdgroupMatmulOp>(
        loc, accTy, aMat, bMat, accIn, rewriter.getStringAttr(storage),
        rewriter.getI64IntegerAttr(kExtent), rewriter.getI64IntegerAttr(kExtent),
        rewriter.getI64IntegerAttr(kExtent));
    rewriter.create<scf::YieldOp>(loc, ValueRange{mma.getResult()});

    // C[m, n] at m*N + n.
    rewriter.setInsertionPointAfter(kLoop);
    Value cOff = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, m, nStride), n);
    rewriter.create<SimdgroupStoreOp>(
        loc, kLoop.getResult(0), cBuf, cOff, rewriter.getI64IntegerAttr(N),
        rewriter.getStringAttr("threadgroup"));

    rewriter.setInsertionPointAfter(mLoop);
    auto outTensorTy = RankedTensorType::get({M, N}, rewriter.getF32Type());
    Value result =
        rewriter.create<bufferization::ToTensorOp>(loc, outTensorTy, cBuf);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerMatmulToAppleSimdgroupPass
    : public PassWrapper<LowerMatmulToAppleSimdgroupPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmulToAppleSimdgroupPass)

  StringRef getArgument() const override {
    return "tessera-matmul-to-apple-simdgroup";
  }
  StringRef getDescription() const override {
    return "Lower tessera.matmul to Apple GPU simdgroup machine primitives";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect, arith::ArithDialect, scf::SCFDialect,
                    memref::MemRefDialect, bufferization::BufferizationDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMatmulToAppleSimdgroup>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createLowerMatmulToAppleSimdgroupPass() {
  return std::make_unique<LowerMatmulToAppleSimdgroupPass>();
}

} // namespace apple
} // namespace tessera
