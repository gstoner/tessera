
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
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
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
  return arith::ConstantIndexOp::create(b, loc, v);
}

static bool hasOperandSegment(Operation *op, unsigned index) {
  auto segments = op->getAttrOfType<DenseI32ArrayAttr>("operand_segment_sizes");
  if (!segments)
    segments = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segments || index >= segments.asArrayRef().size())
    return false;
  return segments.asArrayRef()[index] != 0;
}

static bool hasOptionalOperand(Operation *op, unsigned index) {
  return index < op->getNumOperands() &&
         (hasOperandSegment(op, index) || op->getNumOperands() > index);
}

static bool sameStaticShape(RankedTensorType a, RankedTensorType b) {
  if (a.getRank() != b.getRank() || !a.hasStaticShape() || !b.hasStaticShape())
    return false;
  for (int64_t i = 0, e = a.getRank(); i < e; ++i)
    if (a.getDimSize(i) != b.getDimSize(i))
      return false;
  return true;
}

static int64_t i64AttrOr(Operation *op, llvm::StringRef name,
                         int64_t fallback) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

static bool isStaticF32Tensor(Type type, RankedTensorType &out) {
  out = llvm::dyn_cast<RankedTensorType>(type);
  return out && out.hasStaticShape() && out.getElementType().isF32();
}

static bool isStrictCl30Tensor(RankedTensorType ty) {
  return ty && ty.hasStaticShape() && ty.getElementType().isF32() &&
         ty.getRank() >= 1 && ty.getDimSize(ty.getRank() - 1) == 8;
}

static bool isStrictCl30Op(Operation *op) {
  return i64AttrOr(op, "p", -1) == 3 && i64AttrOr(op, "q", -1) == 0;
}

static void copyAttrs(Operation *op, OperationState &st) {
  for (auto &na : op->getAttrs())
    st.addAttribute(na.getName(), na.getValue());
}

static Operation *createPreservedTileOp(PatternRewriter &rewriter,
                                        Operation *op,
                                        llvm::StringRef tileName,
                                        llvm::StringRef source,
                                        ArrayRef<NamedAttribute> extra = {}) {
  OperationState st(op->getLoc(), tileName);
  st.addOperands(op->getOperands());
  st.addTypes(op->getResultTypes());
  copyAttrs(op, st);
  st.addAttribute("source", rewriter.getStringAttr(source));
  for (auto &na : extra)
    st.addAttribute(na.getName(), na.getValue());
  Operation *tiled = rewriter.create(st);
  rewriter.replaceOp(op, tiled->getResults());
  return tiled;
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
        tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{M, N}, elemTy);

    Value zero = idx(rewriter, loc, 0);
    Value one  = idx(rewriter, loc, 1);
    Value Mval = idx(rewriter, loc, M);
    Value Nval = idx(rewriter, loc, N);
    Value tmVal = idx(rewriter, loc, tm);
    Value tnVal = idx(rewriter, loc, tn);
    Value Kval = idx(rewriter, loc, K);

    // Outer loop over M tiles.
    auto outerFor = scf::ForOp::create(
        rewriter,
        loc, zero, Mval, tmVal, ValueRange{initTensor});
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(outerFor.getBody());
      Value i    = outerFor.getInductionVar();
      Value acc0 = outerFor.getRegionIterArg(0);

      // Inner loop over N tiles.
      auto innerFor = scf::ForOp::create(
          rewriter,
          loc, zero, Nval, tnVal, ValueRange{acc0});
      {
        OpBuilder::InsertionGuard g2(rewriter);
        rewriter.setInsertionPointToStart(innerFor.getBody());
        Value j    = innerFor.getInductionVar();
        Value acc1 = innerFor.getRegionIterArg(0);

        // Extract A tile: A[i:i+tm, 0:K]
        auto aTileType =
            RankedTensorType::get({tm, K}, lhsTy.getElementType());
        Value aSlice = tensor::ExtractSliceOp::create(
            rewriter,
            loc, aTileType, lhs,
            ValueRange{i, zero},          // offsets
            ValueRange{tmVal, Kval},       // sizes
            ValueRange{one, one});         // strides

        // Extract B tile: B[0:K, j:j+tn]
        auto bTileType =
            RankedTensorType::get({K, tn}, rhsTy.getElementType());
        Value bSlice = tensor::ExtractSliceOp::create(
            rewriter,
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
        Value acc2 = tensor::InsertSliceOp::create(
            rewriter,
            loc, cTile, acc1,
            ValueRange{i, j},             // offsets
            ValueRange{tmVal, tnVal},      // sizes
            ValueRange{one, one});         // strides

        scf::YieldOp::create(rewriter, loc, ValueRange{acc2});
      }

      // Outer yield with inner result.
      scf::YieldOp::create(rewriter, loc, innerFor.getResults());
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
    // Sprint 8: batched value matmul covers f32 (CPU) + f16/bf16 (GPU). Require
    // a single shared float element type; the CPU vs GPU TileToApple value
    // blocks decide which dtypes are executable on each backend (CPU gates
    // non-f32 batched; GPU accepts f32/f16/bf16).
    Type belemTy = resTy.getElementType();
    auto isBatchedElem = [](Type t) {
      return t.isF32() || t.isF16() || t.isBF16();
    };
    if (!isBatchedElem(belemTy) ||
        lhsTy.getElementType() != belemTy || rhsTy.getElementType() != belemTy)
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
// Rewrite pattern: TilePPOPolicyLossValue (Stage 13)
// ─────────────────────────────────────────────────────────────────────────────
//
// Preserve the narrow PPO loss executable envelope as a single registered Tile
// op. The Apple GPU value lane consumes `tile.ppo_policy_loss` and emits a
// MPSGraph-backed `tessera_apple.gpu.kernel_call`. Everything outside this
// strict shape/attr envelope is left as Graph IR and gated downstream.
struct TilePPOPolicyLossValue : public RewritePattern {
  TilePPOPolicyLossValue(MLIRContext *ctx)
      : RewritePattern("tessera.rl.ppo_policy_loss", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 3 || op->getNumOperands() > 6 ||
        op->getNumResults() != 1)
      return failure();
    auto nextTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto oldTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto advTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(2).getType());
    auto resTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!nextTy || !oldTy || !advTy || !resTy)
      return failure();
    if (!nextTy.hasStaticShape() || !oldTy.hasStaticShape() ||
        !advTy.hasStaticShape() || !resTy.hasStaticShape())
      return failure();
    if (!nextTy.getElementType().isF32() || !oldTy.getElementType().isF32() ||
        !advTy.getElementType().isF32() || !resTy.getElementType().isF32())
      return failure();
    auto sameShape = [](RankedTensorType a, RankedTensorType b) {
      if (a.getRank() != b.getRank())
        return false;
      for (int64_t i = 0, e = a.getRank(); i < e; ++i)
        if (a.getDimSize(i) != b.getDimSize(i))
          return false;
      return true;
    };
    if (!sameShape(nextTy, oldTy) || !sameShape(nextTy, advTy))
      return failure();
    for (Value side : op->getOperands().drop_front(3)) {
      auto sideTy = llvm::dyn_cast<RankedTensorType>(side.getType());
      if (!sideTy || !sideTy.hasStaticShape() ||
          !sideTy.getElementType().isF32() || !sameShape(nextTy, sideTy))
        return failure();
    }
    if (resTy.getRank() != 0)
      return failure();
    if (auto r = op->getAttrOfType<StringAttr>("reduction");
        r && r.getValue() != "mean")
      return failure();
    if (auto c = op->getAttrOfType<FloatAttr>("clip_epsilon");
        c && c.getValueAsDouble() <= 0.0)
      return failure();
    if (auto k = op->getAttrOfType<FloatAttr>("kl_coef");
        k && k.getValueAsDouble() < 0.0)
      return failure();
    if (auto e = op->getAttrOfType<FloatAttr>("entropy_coef");
        e && e.getValueAsDouble() < 0.0)
      return failure();

    OperationState st(op->getLoc(), "tile.ppo_policy_loss");
    st.addOperands(op->getOperands());
    st.addTypes(op->getResultTypes());
    for (auto &na : op->getAttrs())
      st.addAttribute(na.getName(), na.getValue());
    st.addAttribute("has_mask",
                    rewriter.getBoolAttr(hasOperandSegment(op, 3)));
    st.addAttribute("has_ref_kl",
                    rewriter.getBoolAttr(hasOperandSegment(op, 4)));
    st.addAttribute("has_entropy",
                    rewriter.getBoolAttr(hasOperandSegment(op, 5)));
    st.addAttribute("source",
                    rewriter.getStringAttr("tessera.rl.ppo_policy_loss"));
    Operation *tiled = rewriter.create(st);
    rewriter.replaceOp(op, tiled->getResults());
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Rewrite patterns: EBM value kernels
// ─────────────────────────────────────────────────────────────────────────────
//
// Preserve the first EBM executable envelopes as registered Tile ops for the
// Apple GPU value lane. The runtime symbols report success only when the real
// Metal dispatch path ran; CPU/reference fallback never receives a GPU label.
struct TileEBMEnergyQuadraticValue : public RewritePattern {
  TileEBMEnergyQuadraticValue(MLIRContext *ctx)
      : RewritePattern("tessera.ebm.energy_quadratic", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    auto xTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto yTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto eTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!xTy || !yTy || !eTy)
      return failure();
    if (!xTy.hasStaticShape() || !yTy.hasStaticShape() ||
        !eTy.hasStaticShape())
      return failure();
    if (!xTy.getElementType().isF32() || !yTy.getElementType().isF32() ||
        !eTy.getElementType().isF32())
      return failure();
    if (xTy.getRank() != 2 || yTy.getRank() != 2 || eTy.getRank() != 1)
      return failure();
    if (xTy.getShape() != yTy.getShape() ||
        eTy.getDimSize(0) != xTy.getDimSize(0))
      return failure();

    createPreservedTileOp(rewriter, op, "tile.ebm_energy_quadratic",
                          "tessera.ebm.energy_quadratic");
    return success();
  }
};

struct TileEBMLangevinStepValue : public RewritePattern {
  TileEBMLangevinStepValue(MLIRContext *ctx)
      : RewritePattern("tessera.ebm.langevin_step", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return failure();
    if (!hasOptionalOperand(op, 2))
      return failure();
    auto yTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto gTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto nTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(2).getType());
    auto oTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!yTy || !gTy || !nTy || !oTy)
      return failure();
    if (!yTy.hasStaticShape() || !gTy.hasStaticShape() ||
        !nTy.hasStaticShape() || !oTy.hasStaticShape())
      return failure();
    if (!yTy.getElementType().isF32() || !gTy.getElementType().isF32() ||
        !nTy.getElementType().isF32() || !oTy.getElementType().isF32())
      return failure();
    if (yTy.getShape() != gTy.getShape() || yTy.getShape() != nTy.getShape() ||
        yTy.getShape() != oTy.getShape())
      return failure();
    auto eta = op->getAttrOfType<FloatAttr>("eta");
    if (!eta || eta.getValueAsDouble() <= 0.0)
      return failure();
    if (auto scale = op->getAttrOfType<FloatAttr>("noise_scale");
        scale && scale.getValueAsDouble() < 0.0)
      return failure();

    SmallVector<NamedAttribute, 1> extra;
    extra.emplace_back(rewriter.getStringAttr("has_noise"),
                       rewriter.getBoolAttr(true));
    createPreservedTileOp(rewriter, op, "tile.ebm_langevin_step",
                          "tessera.ebm.langevin_step", extra);
    return success();
  }
};

struct TileEBMRefinementValue : public RewritePattern {
  TileEBMRefinementValue(MLIRContext *ctx)
      : RewritePattern("tessera.ebm.refinement", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Stage 16E executable envelope: deterministic refinement only.
    // The Metal symbol computes y - steps*eta*grad and has no noise or
    // temperature semantics, so noisy/temperature-controlled variants remain
    // gated until a matching runtime ABI exists.
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    if (op->hasAttr("temperature") || op->hasAttr("noise_scale"))
      return failure();
    auto yTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto gTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto oTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!yTy || !gTy || !oTy)
      return failure();
    if (!yTy.hasStaticShape() || !gTy.hasStaticShape() || !oTy.hasStaticShape())
      return failure();
    if (!yTy.getElementType().isF32() || !gTy.getElementType().isF32() ||
        !oTy.getElementType().isF32())
      return failure();
    if (!sameStaticShape(yTy, gTy) || !sameStaticShape(yTy, oTy))
      return failure();
    auto eta = op->getAttrOfType<FloatAttr>("eta");
    auto steps = op->getAttrOfType<IntegerAttr>("steps");
    if (!eta || eta.getValueAsDouble() <= 0.0 || !steps ||
        steps.getInt() <= 0)
      return failure();

    createPreservedTileOp(rewriter, op, "tile.ebm_refinement",
                          "tessera.ebm.refinement");
    return success();
  }
};

struct TileEBMPartitionExactValue : public RewritePattern {
  TileEBMPartitionExactValue(MLIRContext *ctx)
      : RewritePattern("tessera.ebm.partition_exact", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    auto eTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto oTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!eTy || !oTy)
      return failure();
    if (!eTy.hasStaticShape() || !oTy.hasStaticShape())
      return failure();
    if (!eTy.getElementType().isF32() || !oTy.getElementType().isF32())
      return failure();
    if (oTy.getRank() != 0)
      return failure();
    if (auto temperature = op->getAttrOfType<FloatAttr>("temperature");
        temperature && temperature.getValueAsDouble() <= 0.0)
      return failure();
    if (auto reduction = op->getAttrOfType<StringAttr>("reduction");
        reduction && reduction.getValue() != "logsumexp")
      return failure();

    createPreservedTileOp(rewriter, op, "tile.ebm_partition_exact",
                          "tessera.ebm.partition_exact");
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Rewrite patterns: GA / Clifford value seam (Stage 16C)
// ─────────────────────────────────────────────────────────────────────────────
//
// Preserve strict cl30 fp32 Clifford ops as registered Tile IR carriers. This is
// not an Apple execution claim: TileToApple still emits a named value-lowering
// diagnostic until a GA value executor is promoted. The point of this stage is
// that in-envelope GA ops cross Graph→Tile as typed, registered IR instead of an
// opaque tile.* husk.
struct TileCliffordBinaryValue : public RewritePattern {
  TileCliffordBinaryValue(MLIRContext *ctx, llvm::StringRef opName,
                          llvm::StringRef tileName)
      : RewritePattern(opName, /*benefit=*/2, ctx), tileName(tileName) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    if (!isStrictCl30Op(op))
      return failure();
    RankedTensorType lhsTy, rhsTy, resTy;
    if (!isStaticF32Tensor(op->getOperand(0).getType(), lhsTy) ||
        !isStaticF32Tensor(op->getOperand(1).getType(), rhsTy) ||
        !isStaticF32Tensor(op->getResult(0).getType(), resTy))
      return failure();
    if (!isStrictCl30Tensor(lhsTy) || !isStrictCl30Tensor(rhsTy) ||
        !isStrictCl30Tensor(resTy))
      return failure();
    if (!sameStaticShape(lhsTy, rhsTy) || !sameStaticShape(lhsTy, resTy))
      return failure();

    llvm::StringRef source = op->getName().getStringRef();
    SmallVector<NamedAttribute, 2> extra;
    extra.emplace_back(rewriter.getStringAttr("has_signature"),
                       rewriter.getBoolAttr(op->hasAttr("signature")));
    extra.emplace_back(rewriter.getStringAttr("has_grade_mask"),
                       rewriter.getBoolAttr(op->hasAttr("grade_mask")));
    createPreservedTileOp(rewriter, op, tileName, source, extra);
    return success();
  }

  llvm::StringRef tileName;
};

struct TileCliffordUnaryValue : public RewritePattern {
  TileCliffordUnaryValue(MLIRContext *ctx, llvm::StringRef opName,
                         llvm::StringRef tileName, bool allowNormDrop = false)
      : RewritePattern(opName, /*benefit=*/2, ctx), tileName(tileName),
        allowNormDrop(allowNormDrop) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    if (!isStrictCl30Op(op))
      return failure();
    RankedTensorType inputTy, resTy;
    if (!isStaticF32Tensor(op->getOperand(0).getType(), inputTy) ||
        !isStaticF32Tensor(op->getResult(0).getType(), resTy))
      return failure();
    if (!isStrictCl30Tensor(inputTy))
      return failure();
    if (allowNormDrop && resTy.getRank() == inputTy.getRank() - 1) {
      for (int64_t i = 0, e = resTy.getRank(); i < e; ++i)
        if (resTy.getDimSize(i) != inputTy.getDimSize(i))
          return failure();
    } else {
      if (!isStrictCl30Tensor(resTy) || !sameStaticShape(inputTy, resTy))
        return failure();
    }

    llvm::StringRef source = op->getName().getStringRef();
    SmallVector<NamedAttribute, 2> extra;
    extra.emplace_back(rewriter.getStringAttr("has_signature"),
                       rewriter.getBoolAttr(op->hasAttr("signature")));
    extra.emplace_back(rewriter.getStringAttr("has_grade_mask"),
                       rewriter.getBoolAttr(op->hasAttr("grade_mask")));
    createPreservedTileOp(rewriter, op, tileName, source, extra);
    return success();
  }

  llvm::StringRef tileName;
  bool allowNormDrop;
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
  Option<bool> valueModeOpt{
      *this, "value-mode",
      llvm::cl::desc("Preserve strict value-executable envelopes as registered "
                     "Tile IR instead of expanding to artifact tiling"),
      llvm::cl::init(false)};

  StringRef getArgument()    const override { return "tessera-tiling"; }
  StringRef getDescription() const override {
    return "Tile tessera.matmul into scf.for M×N loop nests; lower the linalg "
           "family (cholesky/tri_solve/cholesky_solve/lu/qr/svd) to opaque "
           "tile.<op> Tile-IR ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect,
                    tensor::TensorDialect,
                    // Sprint 9: the value/linalg tiling patterns create
                    // registered tile.* ops — load the Tile dialect so they are
                    // verified, not unregistered (no --allow-unregistered-dialect).
                    ::tessera::tile::TesseraTileDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    const bool effectiveValueMode = valueMode || valueModeOpt;
    if (effectiveValueMode) {
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
      // Stage 13: static fp32 PPO mean loss → tile.ppo_policy_loss for the
      // Apple GPU value lane. CPU TileToApple gates it explicitly.
      patterns.add<TilePPOPolicyLossValue>(&getContext());
      // EBM value lane: strict static fp32 tensor-shaped kernels.
      patterns.add<TileEBMEnergyQuadraticValue, TileEBMLangevinStepValue,
                   TileEBMRefinementValue, TileEBMPartitionExactValue>(
          &getContext());
      // Stage 16C: strict cl30 fp32 GA ops cross Graph→Tile as registered Tile
      // IR carriers. They remain target-gated in TileToApple until a GA value
      // executor is promoted.
      patterns.add<TileCliffordBinaryValue>(
          &getContext(), "tessera.clifford.geometric_product",
          "tile.clifford_geometric_product");
      patterns.add<TileCliffordBinaryValue>(
          &getContext(), "tessera.clifford.outer_product",
          "tile.clifford_outer_product");
      patterns.add<TileCliffordBinaryValue>(
          &getContext(), "tessera.clifford.inner_product",
          "tile.clifford_inner_product");
      patterns.add<TileCliffordUnaryValue>(
          &getContext(), "tessera.clifford.reverse", "tile.clifford_reverse");
      patterns.add<TileCliffordUnaryValue>(
          &getContext(), "tessera.clifford.grade_project",
          "tile.clifford_grade_project");
      patterns.add<TileCliffordUnaryValue>(
          &getContext(), "tessera.clifford.norm", "tile.clifford_norm",
          /*allowNormDrop=*/true);
      patterns.add<TileCliffordBinaryValue>(
          &getContext(), "tessera.clifford.rotor_sandwich",
          "tile.clifford_rotor_sandwich");
    } else {
      patterns.add<TileMatmul>(&getContext(),
                               static_cast<int64_t>(tileMOpt),
                               static_cast<int64_t>(tileNOpt));
    }
    for (llvm::StringRef opName : kLinalgGraphOps)
      patterns.add<TileLinalg>(&getContext(), opName);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
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
