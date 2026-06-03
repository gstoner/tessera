
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
#include "llvm/Support/Casting.h"

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
    auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());
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

    auto resultTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
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
// Rewrite pattern: TileLinalg  (L-series linalg family, 2026-06-02)
// ─────────────────────────────────────────────────────────────────────────────
//
// The linalg primitives (cholesky / tri_solve / cholesky_solve / lu / qr / svd)
// are sequential blocked factorizations and solves, not embarrassingly-parallel
// loop nests like matmul, so the pilot does not tile them into scf.for.  Each
// lowers 1:1 to an opaque Tile-IR op `tile.<suffix>` (e.g. tessera.tri_solve →
// tile.tri_solve), giving the Tile layer an explicit, distinct representation
// that the Tile→Apple pass consumes.  This single pattern is the table-driven
// generalization of the original cholesky one-off: it copies all operands and
// all result types verbatim, so it handles multi-operand (tri_solve,
// *_solve: A,B→X) and multi-result (lu, qr, svd: A→U,S,V) ops uniformly.
// All attributes are carried forward and the Graph-IR op spelling is preserved
// as `source` so the Tile→Apple pass (which matches by `source`) recognizes the
// opaque tile op.  Matching `tessera.<op>` → `tile.<op>` (a different name)
// means the greedy driver applies it exactly once.
//
// The op set IS the table; adding a linalg op is a one-line edit here plus a
// matching spec in TileToApple.cpp's kLinalgSpecs and a runtime symbol.
static constexpr llvm::StringLiteral kLinalgGraphOps[] = {
    "tessera.cholesky",      "tessera.tri_solve", "tessera.cholesky_solve",
    "tessera.lu",            "tessera.qr",        "tessera.svd"};

struct TileLinalg : public RewritePattern {
  TileLinalg(MLIRContext *ctx, llvm::StringRef opName)
      : RewritePattern(opName, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    llvm::StringRef graphName = op->getName().getStringRef();
    llvm::StringRef suffix = graphName;
    suffix.consume_front("tessera.");
    std::string tileName = ("tile." + suffix).str();

    OperationState st(op->getLoc(), tileName);
    st.addOperands(op->getOperands());
    st.addTypes(op->getResultTypes());
    for (auto &na : op->getAttrs())
      st.addAttribute(na.getName(), na.getValue());
    st.addAttribute("source", rewriter.getStringAttr(graphName));
    Operation *tiled = rewriter.create(st);
    rewriter.replaceOp(op, tiled->getResults());
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Rewrite pattern: TileMatmulValue  (Apple Value Target IR sprint 5, 2026-06-03)
// ─────────────────────────────────────────────────────────────────────────────
//
// The artifact/default path tiles matmul into scf.for loop nests (TileMatmul).
// The *value* path needs the dense contraction to survive as a single Tile-IR
// op so the Tile→Apple value-mode lowering can hand it to a single
// `tessera_apple.cpu.call` GEMM (Accelerate). This pattern lowers
// `tessera.matmul`/`tessera.gemm` 1:1 to `tile.matmul`/`tile.gemm`, but **only**
// for the executable envelope: static rank-2, f32, K-consistent. Anything else
// (dynamic, batched/rank≠2, non-f32) is left untouched — it then reaches the
// value lowering with no value op and is honestly gated with a named
// diagnostic, never silently tiled or fabricated. Enabled only when the pass
// runs in valueMode (the apple_cpu `-full` pipeline).
struct TileMatmulValue : public RewritePattern {
  TileMatmulValue(MLIRContext *ctx, llvm::StringRef opName)
      : RewritePattern(opName, /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumResults() != 1)
      return failure();
    auto lhsTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhsTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto resTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!lhsTy || !rhsTy || !resTy)
      return failure();
    // Executable envelope: static rank-2, f32, K-consistent.
    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || resTy.getRank() != 2)
      return failure();
    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() ||
        !resTy.hasStaticShape())
      return failure();
    // Sprint 7: rank-2 value matmul executes for f32 (cblas_sgemm) + f16/bf16
    // (BNNS). Require a single shared float element type across lhs/rhs/result;
    // mixed-precision and integer matmul stay out of the value envelope.
    Type elemTy = resTy.getElementType();
    auto isValueMatmulElem = [](Type t) {
      return t.isF32() || t.isF16() || t.isBF16();
    };
    if (!isValueMatmulElem(elemTy) ||
        lhsTy.getElementType() != elemTy || rhsTy.getElementType() != elemTy)
      return failure();
    // Transpose is gated: the value ABI / runtime dispatch only honors the
    // physical (M,K)@(K,N) layout (cblas CblasNoTrans). A transposeA/transposeB
    // matmul must NOT become an executable value call that silently computes
    // the non-transposed product. Leave it untouched → gated downstream with a
    // named diagnostic until the value ABI carries transpose attrs and the
    // runtime honors them.
    if (auto a = op->getAttrOfType<BoolAttr>("transposeA"); a && a.getValue())
      return failure();
    if (auto b = op->getAttrOfType<BoolAttr>("transposeB"); b && b.getValue())
      return failure();
    // K-consistency AND result shape (M,N). The result check is essential: a
    // malformed (4x8)@(8x16)->(5x5) passes rank+static+f32+K but must NOT
    // become an executable value call producing a wrong-shaped output. The
    // registered MatmulOp verifier also enforces this; we re-check so the value
    // tile op is never created for a shape-inconsistent matmul.
    if (lhsTy.getDimSize(1) != rhsTy.getDimSize(0))
      return failure();
    if (resTy.getDimSize(0) != lhsTy.getDimSize(0) ||
        resTy.getDimSize(1) != rhsTy.getDimSize(1))
      return failure();

    llvm::StringRef graphName = op->getName().getStringRef();
    llvm::StringRef suffix = graphName;
    suffix.consume_front("tessera.");
    std::string tileName = ("tile." + suffix).str();
    OperationState st(op->getLoc(), tileName);
    st.addOperands(op->getOperands());
    st.addTypes(op->getResultTypes());
    for (auto &na : op->getAttrs())
      st.addAttribute(na.getName(), na.getValue());
    st.addAttribute("source", rewriter.getStringAttr(graphName));
    Operation *tiled = rewriter.create(st);
    rewriter.replaceOp(op, tiled->getResults());
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Rewrite pattern: TileBatchedMatmulValue (Apple Value Target IR sprint 6)
// ─────────────────────────────────────────────────────────────────────────────
//
// Value path only: preserve a static rank-3 f32 `tessera.batched_gemm` as a
// single `tile.batched_gemm` for the Accelerate batched-GEMM value call. Strict
// envelope — static rank-3, f32, batch + K + M + N consistent. No broadcasting,
// no transpose, no rank-2/rank-4. Out-of-envelope ops are left untouched and
// gated downstream (the registered BatchedGemmOp verifier rejects most of them
// before we even get here; this is the defense-in-depth gate at the tile seam).
struct TileBatchedMatmulValue : public RewritePattern {
  TileBatchedMatmulValue(MLIRContext *ctx, llvm::StringRef opName)
      : RewritePattern(opName, /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumResults() != 1)
      return failure();
    auto lhsTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhsTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto resTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!lhsTy || !rhsTy || !resTy)
      return failure();
    if (lhsTy.getRank() != 3 || rhsTy.getRank() != 3 || resTy.getRank() != 3)
      return failure();
    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() ||
        !resTy.hasStaticShape())
      return failure();
    if (!lhsTy.getElementType().isF32() || !rhsTy.getElementType().isF32() ||
        !resTy.getElementType().isF32())
      return failure();
    // batch, K, M, N consistency — no broadcasting (batch must match exactly).
    if (lhsTy.getDimSize(0) != rhsTy.getDimSize(0) ||
        lhsTy.getDimSize(0) != resTy.getDimSize(0))
      return failure();
    if (lhsTy.getDimSize(2) != rhsTy.getDimSize(1))
      return failure();
    if (resTy.getDimSize(1) != lhsTy.getDimSize(1) ||
        resTy.getDimSize(2) != rhsTy.getDimSize(2))
      return failure();

    llvm::StringRef graphName = op->getName().getStringRef();
    llvm::StringRef suffix = graphName;
    suffix.consume_front("tessera.");
    std::string tileName = ("tile." + suffix).str();
    OperationState st(op->getLoc(), tileName);
    st.addOperands(op->getOperands());
    st.addTypes(op->getResultTypes());
    for (auto &na : op->getAttrs())
      st.addAttribute(na.getName(), na.getValue());
    st.addAttribute("source", rewriter.getStringAttr(graphName));
    Operation *tiled = rewriter.create(st);
    rewriter.replaceOp(op, tiled->getResults());
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass
// ─────────────────────────────────────────────────────────────────────────────

struct TilingPassImpl
    : public PassWrapper<TilingPassImpl, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilingPassImpl)

  TilingPassImpl() = default;
  explicit TilingPassImpl(bool valueMode) : valueMode(valueMode) {}
  TilingPassImpl(const TilingPassImpl &other)
      : PassWrapper(other), valueMode(other.valueMode) {}

  // Apple Value Target IR sprint 5: in value mode, preserve static rank-2 f32
  // matmul/gemm as a single tile op (TileMatmulValue) for the Accelerate GEMM
  // value call, instead of tiling to scf.for. Default (artifact) tiling is
  // unchanged. Set by the apple_cpu `-full` pipeline only.
  bool valueMode = false;

  Option<int> tileMOpt{*this, "tile-m",
                       llvm::cl::desc("M-dimension tile size"), llvm::cl::init(16)};
  Option<int> tileNOpt{*this, "tile-n",
                       llvm::cl::desc("N-dimension tile size"), llvm::cl::init(16)};

  StringRef getArgument()    const override { return "tessera-tiling"; }
  StringRef getDescription() const override {
    return "Tile tessera.matmul into scf.for M×N loop nests; lower the linalg "
           "family (cholesky/tri_solve/cholesky_solve/lu/qr/svd) to opaque "
           "tile.<op> Tile-IR ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    if (valueMode) {
      // Value path (apple_cpu `-full`): preserve static rank-2 f32 matmul as a
      // single tile op for the Accelerate GEMM value call; do NOT tile to
      // scf.for (that would dissolve the dense contraction the value lane wants
      // to hand to one cblas_sgemm). Out-of-envelope matmuls are left untouched
      // and gated downstream.
      //
      // NOTE: only `tessera.matmul` is registered in the Graph IR dialect today.
      // `tessera.gemm` is a vocabulary alias, not a distinct registered op — a
      // pattern keyed on it would be dead (and tessera-opt rejects the unknown
      // op at parse time). The Tile→Apple value lowering still emits op_kind
      // "gemm" if a `tile.gemm` ever arrives, but the executable Graph IR
      // spelling is `tessera.matmul`.
      patterns.add<TileMatmulValue>(&getContext(), "tessera.matmul");
      // Sprint 6: static rank-3 f32 batched matmul → tile.batched_gemm.
      patterns.add<TileBatchedMatmulValue>(&getContext(), "tessera.batched_gemm");
    } else {
      patterns.add<TileMatmul>(&getContext(),
                               static_cast<int64_t>(tileMOpt),
                               static_cast<int64_t>(tileNOpt));
    }
    for (llvm::StringRef opName : kLinalgGraphOps)
      patterns.add<TileLinalg>(&getContext(), opName);
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
std::unique_ptr<Pass> createTilingPass(bool valueMode) {
  return std::make_unique<TilingPassImpl>(valueMode);
}
} // namespace tessera
