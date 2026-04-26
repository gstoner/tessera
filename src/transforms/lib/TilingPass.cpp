
// TilingPass.cpp
//
// Tiles tessera.matmul ops into scf.for loop nests over the M and N output
// dimensions, using tensor.extract_slice / tensor.insert_slice to carve and
// re-assemble tiles.
//
// Only ops with statically-shaped ranked tensor operands are tiled; any op
// whose shape is dynamic or unranked is left unchanged.
//
// Transformation for C = A @ B  (A: MxK, B: KxN → C: MxN):
//
//   %init = tensor.empty() : tensor<MxNxeT>
//   %C = scf.for %i = 0 to M step tile_m
//            iter_args(%acc0 = %init) -> tensor<MxNxeT> {
//     %C1 = scf.for %j = 0 to N step tile_n
//               iter_args(%acc1 = %acc0) -> tensor<MxNxeT> {
//       %a_sl = tensor.extract_slice %A[%i, 0][tile_m, K][1, 1]
//       %b_sl = tensor.extract_slice %B[0, %j][K, tile_n][1, 1]
//       %c_sl = tessera.matmul %a_sl, %b_sl : (...) -> tensor<tile_mxtile_nxeT>
//       %acc2 = tensor.insert_slice %c_sl into %acc1[%i, %j][tile_m, tile_n][1,1]
//       scf.yield %acc2
//     }
//     scf.yield %C1
//   }
//
// Pass options
//   --tile-m  M-dimension tile size (default 16)
//   --tile-n  N-dimension tile size (default 16)

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// Helper: create an arith.constant of index type.
static Value idx(OpBuilder &b, Location loc, int64_t v) {
  return b.create<arith::ConstantIndexOp>(loc, v);
}

// ─────────────────────────────────────────────────────────────────────────────
// Rewrite pattern: TileMatmul
// ─────────────────────────────────────────────────────────────────────────────

struct TileMatmul : public RewritePattern {
  TileMatmul(MLIRContext *ctx, int64_t tileM, int64_t tileN)
      : RewritePattern("tessera.matmul", /*benefit=*/1, ctx),
        tileM_(tileM), tileN_(tileN) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumResults() != 1)
      return failure();

    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);

    // Require statically-shaped 2-D ranked tensors.
    auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsTy || !rhsTy) return failure();
    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2) return failure();
    if (lhsTy.isDynamicDim(0) || lhsTy.isDynamicDim(1) ||
        rhsTy.isDynamicDim(0) || rhsTy.isDynamicDim(1))
      return failure();

    int64_t M = lhsTy.getDimSize(0);
    int64_t K = lhsTy.getDimSize(1);
    int64_t N = rhsTy.getDimSize(1);

    // Skip if the matmul is already tile-sized (avoid infinite loop).
    if (M <= tileM_ && N <= tileN_) return failure();

    // Check that K matches.
    if (rhsTy.getDimSize(0) != K) return failure();

    auto resultTy = op->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!resultTy) return failure();

    Location loc = op->getLoc();
    Type elemTy  = resultTy.getElementType();

    // Clamp tile sizes to actual dims.
    int64_t tm = std::min(tileM_, M);
    int64_t tn = std::min(tileN_, N);

    // Carry forward transposeA / transposeB attributes.
    SmallVector<NamedAttribute> innerAttrs;
    for (auto &na : op->getAttrs())
      innerAttrs.push_back(na);

    // ── Emit the tiled loop nest ───────────────────────────────────────────

    // Accumulator init: zero-filled tensor<MxNxelemTy>
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{M, N}, elemTy);

    Value zero = idx(rewriter, loc, 0);
    Value one  = idx(rewriter, loc, 1);
    Value Mval = idx(rewriter, loc, M);
    Value Nval = idx(rewriter, loc, N);
    Value tmVal = idx(rewriter, loc, tm);
    Value tnVal = idx(rewriter, loc, tn);
    Value Kval = idx(rewriter, loc, K);

    // Outer loop over M tiles.
    auto outerFor = rewriter.create<scf::ForOp>(
        loc, zero, Mval, tmVal, ValueRange{initTensor});
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(outerFor.getBody());
      Value i    = outerFor.getInductionVar();
      Value acc0 = outerFor.getRegionIterArg(0);

      // Inner loop over N tiles.
      auto innerFor = rewriter.create<scf::ForOp>(
          loc, zero, Nval, tnVal, ValueRange{acc0});
      {
        OpBuilder::InsertionGuard g2(rewriter);
        rewriter.setInsertionPointToStart(innerFor.getBody());
        Value j    = innerFor.getInductionVar();
        Value acc1 = innerFor.getRegionIterArg(0);

        // Extract A tile: A[i:i+tm, 0:K]
        auto aTileType =
            RankedTensorType::get({tm, K}, lhsTy.getElementType());
        Value aSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, aTileType, lhs,
            ValueRange{i, zero},          // offsets
            ValueRange{tmVal, Kval},       // sizes
            ValueRange{one, one});         // strides

        // Extract B tile: B[0:K, j:j+tn]
        auto bTileType =
            RankedTensorType::get({K, tn}, rhsTy.getElementType());
        Value bSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, bTileType, rhs,
            ValueRange{zero, j},
            ValueRange{Kval, tnVal},
            ValueRange{one, one});

        // Inner tessera.matmul on the tile.
        auto cTileType = RankedTensorType::get({tm, tn}, elemTy);
        OperationState innerSt(loc, "tessera.matmul");
        innerSt.addOperands({aSlice, bSlice});
        innerSt.addTypes(cTileType);
        // Carry attributes (tile_k, transposeA, transposeB …)
        for (auto &na : innerAttrs)
          innerSt.addAttribute(na.getName(), na.getValue());
        Operation *innerMM = rewriter.create(innerSt);
        Value cTile = innerMM->getResult(0);

        // Insert tile result back into accumulator.
        Value acc2 = rewriter.create<tensor::InsertSliceOp>(
            loc, cTile, acc1,
            ValueRange{i, j},             // offsets
            ValueRange{tmVal, tnVal},      // sizes
            ValueRange{one, one});         // strides

        rewriter.create<scf::YieldOp>(loc, ValueRange{acc2});
      }

      // Outer yield with inner result.
      rewriter.create<scf::YieldOp>(loc, innerFor.getResults());
    }

    rewriter.replaceOp(op, outerFor.getResults());
    return success();
  }

private:
  int64_t tileM_;
  int64_t tileN_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass
// ─────────────────────────────────────────────────────────────────────────────

struct TilingPassImpl
    : public PassWrapper<TilingPassImpl, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilingPassImpl)

  Option<int> tileMOpt{*this, "tile-m",
                       llvm::cl::desc("M-dimension tile size"), llvm::cl::init(16)};
  Option<int> tileNOpt{*this, "tile-n",
                       llvm::cl::desc("N-dimension tile size"), llvm::cl::init(16)};

  StringRef getArgument()    const override { return "tessera-tiling"; }
  StringRef getDescription() const override {
    return "Tile tessera.matmul into scf.for M×N loop nests";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TileMatmul>(&getContext(),
                             static_cast<int64_t>(tileMOpt),
                             static_cast<int64_t>(tileNOpt));
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                           std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createTilingPass() {
  return std::make_unique<TilingPassImpl>();
}
} // namespace tessera
