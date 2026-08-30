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

    // The accumulator is f32 and `simdgroup_store` does not convert -- it
    // moves raw elements. An f16 result therefore needs an explicit rounding
    // epilogue, which is exactly what the MSL kernel does: store the tile to
    // `threadgroup float`, then convert per element on the way out. Emitting
    // that is what keeps Decision #15a's split real here -- storage f16,
    // accumulator f32 -- and it is why the rounding happens ONCE at the end
    // rather than at every K step.
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resTy || !resTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "requires a static result type");
    Type resElem = resTy.getElementType();
    if (!resElem.isF16() && !resElem.isF32())
      return rewriter.notifyMatchFailure(
          op, "result must be f16 or f32; the accumulator is f32 and only "
              "those have a defined rounding epilogue");

    Location loc = op->getLoc();
    Type f32Ty = rewriter.getF32Type();
    const int64_t MP = ((M + kExtent - 1) / kExtent) * kExtent;
    const int64_t NP = ((N + kExtent - 1) / kExtent) * kExtent;

    // Flat buffers: the primitives take a base plus a LINEAR element offset and
    // an explicit row stride, mirroring Metal's pointer arithmetic.
    SmallVector<ReassociationIndices, 1> collapse{{0, 1}};
    auto flat = [&](Value tensor, int64_t rows, int64_t cols, Type et) {
      Value buf = rewriter.create<bufferization::ToBufferOp>(
          loc, MemRefType::get({rows, cols}, et), tensor);
      return rewriter.create<memref::CollapseShapeOp>(loc, buf, collapse)
          .getResult();
    };
    Value aBuf = flat(lhs, M, K, elem);
    Value bBuf = flat(rhs, K, N, elem);
    // Padded to whole tiles so a ragged edge writes into the pad rather than
    // out of bounds; the epilogue copies back only the valid region.
    Value accBuf =
        rewriter.create<memref::AllocOp>(loc, MemRefType::get({MP * NP}, f32Ty));

    auto idx = [&](int64_t v) {
      return rewriter.create<arith::ConstantIndexOp>(loc, v).getResult();
    };
    Value zero = idx(0), one = idx(1), step = idx(kExtent);
    Value mBound = idx(MP), nBound = idx(NP), kBound = idx(K);
    Value mLimit = idx(M), nLimit = idx(N), kLimit = idx(K);
    Value kStride = idx(K), nStride = idx(N), npStride = idx(NP);

    // Staging tiles. Metal's simdgroup_load has NO bounds predicate, so an
    // out-of-range element cannot be masked at the load; it is masked when the
    // tile is copied in. That is why ragged shapes need threadgroup memory at
    // all, and why this pass stages unconditionally rather than only on the
    // ragged path -- one code path is easier to trust than two.
    const int64_t tileElems = kExtent * kExtent;
    const int64_t budget = 32768;  // [MTLDevice maxThreadgroupMemoryLength], Apple7
    auto tileTy = MemRefType::get({tileElems}, elem);
    Value aTile = rewriter.create<ThreadgroupAllocOp>(
        loc, tileTy, rewriter.getI64IntegerAttr(tileElems),
        rewriter.getI64IntegerAttr(budget));
    Value bTile = rewriter.create<ThreadgroupAllocOp>(
        loc, tileTy, rewriter.getI64IntegerAttr(tileElems),
        rewriter.getI64IntegerAttr(budget));

    Value zeroElem = rewriter.create<arith::ConstantOp>(
        loc, elem, rewriter.getFloatAttr(elem, 0.0));

    Type accTy = SimdgroupMatrixType::get(getContext(), f32Ty);
    Type inTy = SimdgroupMatrixType::get(getContext(), elem);
    StringRef storage = elem.isF16() ? "f16" : "f32";
    auto scope = rewriter.getStringAttr("threadgroup");
    auto tileStride = rewriter.getI64IntegerAttr(kExtent);

    auto mLoop = rewriter.create<scf::ForOp>(loc, zero, mBound, step);
    rewriter.setInsertionPointToStart(mLoop.getBody());
    auto nLoop = rewriter.create<scf::ForOp>(loc, zero, nBound, step);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value m = mLoop.getInductionVar(), n = nLoop.getInductionVar();

    Value acc0 = rewriter.create<SimdgroupFillOp>(
        loc, accTy, rewriter.getF32FloatAttr(0.0f));
    auto kLoop = rewriter.create<scf::ForOp>(loc, zero, kBound, step,
                                             ValueRange{acc0});
    rewriter.setInsertionPointToStart(kLoop.getBody());
    Value k = kLoop.getInductionVar();
    Value accIn = kLoop.getRegionIterArgs()[0];

    // Copy one 8x8 tile in, substituting zero out of range. The load itself is
    // inside the guard: computing the address and selecting afterwards would
    // still have read out of bounds.
    auto stageTile = [&](Value dst, Value src, Value rowBase, Value colBase,
                         Value rowLimit, Value colLimit, Value srcStride) {
      auto iLoop = rewriter.create<scf::ForOp>(loc, zero, step, one);
      rewriter.setInsertionPointToStart(iLoop.getBody());
      Value i = iLoop.getInductionVar();
      auto jLoop = rewriter.create<scf::ForOp>(loc, zero, step, one);
      rewriter.setInsertionPointToStart(jLoop.getBody());
      Value j = jLoop.getInductionVar();

      Value r = rewriter.create<arith::AddIOp>(loc, rowBase, i);
      Value c = rewriter.create<arith::AddIOp>(loc, colBase, j);
      Value rOk = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 r, rowLimit);
      Value cOk = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 c, colLimit);
      Value inRange = rewriter.create<arith::AndIOp>(loc, rOk, cOk);
      auto guard = rewriter.create<scf::IfOp>(loc, TypeRange{elem}, inRange,
                                              /*withElseRegion=*/true);
      rewriter.setInsertionPointToStart(guard.thenBlock());
      Value off = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<arith::MulIOp>(loc, r, srcStride), c);
      Value v = rewriter.create<memref::LoadOp>(loc, src, ValueRange{off});
      rewriter.create<scf::YieldOp>(loc, ValueRange{v});
      rewriter.setInsertionPointToStart(guard.elseBlock());
      rewriter.create<scf::YieldOp>(loc, ValueRange{zeroElem});
      rewriter.setInsertionPointAfter(guard);

      Value dstOff = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<arith::MulIOp>(loc, i, step), j);
      rewriter.create<memref::StoreOp>(loc, guard.getResult(0), dst,
                                       ValueRange{dstOff});
      rewriter.setInsertionPointAfter(iLoop);
    };

    rewriter.create<ThreadgroupBarrierOp>(loc, scope);
    stageTile(aTile, aBuf, m, k, mLimit, kLimit, kStride);
    stageTile(bTile, bBuf, k, n, kLimit, nLimit, nStride);
    // Orders the staging writes against the simdgroup reads below. `mem_none`
    // here would compile and race.
    rewriter.create<ThreadgroupBarrierOp>(loc, scope);

    auto aMat = rewriter.create<SimdgroupLoadOp>(loc, inTy, aTile, zero,
                                                 tileStride, scope);
    auto bMat = rewriter.create<SimdgroupLoadOp>(loc, inTy, bTile, zero,
                                                 tileStride, scope);
    auto mma = rewriter.create<SimdgroupMatmulOp>(
        loc, accTy, aMat, bMat, accIn, rewriter.getStringAttr(storage),
        rewriter.getI64IntegerAttr(kExtent), rewriter.getI64IntegerAttr(kExtent),
        rewriter.getI64IntegerAttr(kExtent));
    rewriter.create<scf::YieldOp>(loc, ValueRange{mma.getResult()});

    rewriter.setInsertionPointAfter(kLoop);
    Value cOff = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, m, npStride), n);
    rewriter.create<SimdgroupStoreOp>(loc, kLoop.getResult(0), accBuf, cOff,
                                      rewriter.getI64IntegerAttr(NP), scope);

    rewriter.setInsertionPointAfter(mLoop);

    // Epilogue: copy the valid region out of the padded accumulator, rounding
    // ONCE when the result is narrower. Measured over K = 4096 that is 1.7e-04
    // relative error against 5.8e-03 for accumulating in f16 -- which is what
    // the fp32 tile buys (tests/unit/test_apple_simdgroup_contract.py).
    Value outFlat = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get({M * N}, resElem));
    auto rLoop = rewriter.create<scf::ForOp>(loc, zero, mLimit, one);
    rewriter.setInsertionPointToStart(rLoop.getBody());
    Value r = rLoop.getInductionVar();
    auto cLoop = rewriter.create<scf::ForOp>(loc, zero, nLimit, one);
    rewriter.setInsertionPointToStart(cLoop.getBody());
    Value c = cLoop.getInductionVar();
    Value srcOff = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, r, npStride), c);
    Value v = rewriter.create<memref::LoadOp>(loc, accBuf, ValueRange{srcOff});
    // arith.truncf is round-to-nearest-even despite the name.
    Value outV = resElem.isF32()
                     ? v
                     : rewriter.create<arith::TruncFOp>(loc, resElem, v).getResult();
    Value dstOff = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, r, nStride), c);
    rewriter.create<memref::StoreOp>(loc, outV, outFlat, ValueRange{dstOff});
    rewriter.setInsertionPointAfter(rLoop);

    SmallVector<ReassociationIndices, 1> expand{{0, 1}};
    Value out2d = rewriter.create<memref::ExpandShapeOp>(
        loc, MemRefType::get({M, N}, resElem), outFlat, expand);
    Value result = rewriter.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get({M, N}, resElem), out2d);
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
