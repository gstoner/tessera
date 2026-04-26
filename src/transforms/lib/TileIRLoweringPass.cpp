//===- TileIRLoweringPass.cpp — Phase 3 ──────────────────────────────────===//
//
// Lowers schedule.mesh.region bodies containing tessera.flash_attn into
// FA-4 Tile IR ops:
//
//   tessera.flash_attn(Q, K, V) {causal, tile_q, tile_kv}
//   →
//   tile.async_copy(Q_tile) + tessera.attn.scaled_dot_product
//   + tessera.attn.causal_mask? + tessera.attn.online_softmax
//   + tessera.attn.lse_accumulate + tile.wait_async
//
// The pass also handles tessera.matmul inside mesh.region bodies by emitting
// tile.async_copy + tile.mma + tile.wait_async for the GPU tiling path.
//
// Registration: --tessera-tile-ir-lowering
//   Options:
//     --tile-q   Q tile rows (default 64, must match GPU WGMMA tile)
//     --tile-kv  KV tile cols (default 64)
//     --sm       target SM version (int, e.g. 90 for SM_90)
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tessera {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Emit a tile.async_copy op (string-based to avoid depending on tile_opt_fa4
// dialect headers in the transforms library).
static Operation *emitAsyncCopy(OpBuilder &b, Location loc, Value src,
                                 int64_t tileRows, int64_t tileCols) {
  OperationState st(loc, "tile.async_copy");
  st.addOperands({src});
  st.addAttribute("tile_rows",
                  b.getI64IntegerAttr(tileRows));
  st.addAttribute("tile_cols",
                  b.getI64IntegerAttr(tileCols));
  // Result type: same tensor type as src but tiled dimensions.
  st.addTypes(src.getType());
  return b.create(st);
}

// Emit a tile.wait_async op — drains all in-flight async copies before use.
static Operation *emitWaitAsync(OpBuilder &b, Location loc) {
  OperationState st(loc, "tile.wait_async");
  return b.create(st);
}

// Emit a tessera.attn op by name with given operands and result types.
static Operation *emitAttnOp(OpBuilder &b, Location loc,
                               StringRef opName,
                               ValueRange operands,
                               TypeRange resultTypes,
                               ArrayRef<NamedAttribute> attrs = {}) {
  OperationState st(loc, opName);
  st.addOperands(operands);
  st.addTypes(resultTypes);
  st.addAttributes(attrs);
  return b.create(st);
}

// ─────────────────────────────────────────────────────────────────────────────
// FlashAttn lowering pattern
// ─────────────────────────────────────────────────────────────────────────────

struct LowerFlashAttnToTileIR : public RewritePattern {
  int64_t tileQ;
  int64_t tileKV;
  int      smVersion;

  LowerFlashAttnToTileIR(MLIRContext *ctx, int64_t tq, int64_t tkv, int sm)
      : RewritePattern("tessera.flash_attn", /*benefit=*/2, ctx),
        tileQ(tq), tileKV(tkv), smVersion(sm) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // flash_attn takes Q, K, V operands.
    if (op->getNumOperands() < 3)
      return failure();

    Value Q = op->getOperand(0);
    Value K = op->getOperand(1);
    Value V = op->getOperand(2);
    Location loc = op->getLoc();

    bool causal = false;
    if (auto causalAttr = op->getAttrOfType<BoolAttr>("causal"))
      causal = causalAttr.getValue();

    // Use tile_q / tile_kv from op attrs if present (autotuner may have set them).
    int64_t tq  = tileQ;
    int64_t tkv = tileKV;
    if (auto a = op->getAttrOfType<IntegerAttr>("tessera.tile_q"))
      tq = a.getInt();
    if (auto a = op->getAttrOfType<IntegerAttr>("tessera.tile_kv"))
      tkv = a.getInt();

    // ── Emit tile.async_copy for Q, K, V tiles ──────────────────────────────
    Operation *cpQ = emitAsyncCopy(rewriter, loc, Q, tq, /*d_k inferred*/ -1);
    Operation *cpK = emitAsyncCopy(rewriter, loc, K, tkv, -1);
    Operation *cpV = emitAsyncCopy(rewriter, loc, V, tkv, -1);

    // Barrier: wait for all async copies to complete before compute.
    emitWaitAsync(rewriter, loc);

    // ── Emit online softmax initialiser constants ────────────────────────────
    // running_m init: -inf  (f32 per row)
    // running_l init:  0.0  (f32 per row)
    Value negInf = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), /*neg=*/true),
        rewriter.getF32Type());
    Value zero = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat(0.0f), rewriter.getF32Type());

    // Accumulator output init: zero tensor same shape as Q result.
    Type outType = op->getResultTypes().empty()
                       ? Q.getType()
                       : op->getResult(0).getType();

    // ── Emit scaled dot product ──────────────────────────────────────────────
    // scale = 1/sqrt(d_k) — emitted as F32Attr; d_k unknown at this level so
    // we emit -1.0 as sentinel and expect NVFlashAttnKernelEmitter to fill it.
    SmallVector<NamedAttribute> sdpAttrs = {
        rewriter.getNamedAttr("scale",
                              rewriter.getF32FloatAttr(-1.0f))};
    Operation *sdp = emitAttnOp(
        rewriter, loc, "tessera.attn.scaled_dot_product",
        {cpQ->getResult(0), cpK->getResult(0)}, {outType}, sdpAttrs);

    Value scores = sdp->getResult(0);

    // ── Optional causal mask ─────────────────────────────────────────────────
    if (causal) {
      SmallVector<NamedAttribute> cmAttrs = {
          rewriter.getNamedAttr("q_offset", rewriter.getI64IntegerAttr(0)),
          rewriter.getNamedAttr("kv_offset", rewriter.getI64IntegerAttr(0))};
      Operation *cm = emitAttnOp(rewriter, loc, "tessera.attn.causal_mask",
                                  {scores}, {outType}, cmAttrs);
      scores = cm->getResult(0);
    }

    // ── Online softmax ───────────────────────────────────────────────────────
    // Emit with sentinel init values; the actual init is in the loop preamble.
    Operation *osm = emitAttnOp(
        rewriter, loc, "tessera.attn.online_softmax",
        {scores, negInf, zero, zero /* acc placeholder */},
        {outType, rewriter.getF32Type(), rewriter.getF32Type()});

    // ── LSE accumulate (finalisation) ────────────────────────────────────────
    Operation *lseAcc = emitAttnOp(
        rewriter, loc, "tessera.attn.lse_accumulate",
        {osm->getResult(0), osm->getResult(1), osm->getResult(2)},
        {outType, rewriter.getF32Type()});

    // Store LSE for backward pass.
    emitAttnOp(rewriter, loc, "tessera.attn.lse.save",
               {lseAcc->getResult(1)}, {rewriter.getF32Type()});

    // Replace flash_attn result with normalised output.
    if (!op->getResults().empty())
      rewriter.replaceOp(op, lseAcc->getResult(0));
    else
      rewriter.eraseOp(op);

    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Matmul → tile.mma pattern (GPU path)
// ─────────────────────────────────────────────────────────────────────────────

struct LowerMatmulToTileMMA : public RewritePattern {
  int64_t tileM, tileN;
  int     smVersion;

  LowerMatmulToTileMMA(MLIRContext *ctx, int64_t tm, int64_t tn, int sm)
      : RewritePattern("tessera.matmul", /*benefit=*/1, ctx),
        tileM(tm), tileN(tn), smVersion(sm) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2)
      return failure();

    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Location loc = op->getLoc();
    Type resType = op->getResults().empty() ? A.getType()
                                             : op->getResult(0).getType();

    // Async copies for A and B tiles.
    Operation *cpA = emitAsyncCopy(rewriter, loc, A, tileM, -1);
    Operation *cpB = emitAsyncCopy(rewriter, loc, B, -1, tileN);
    emitWaitAsync(rewriter, loc);

    // tile.mma — the WGMMA/WMMA selector is resolved by NVWGMMALoweringPass.
    OperationState mmaState(loc, "tile.mma");
    mmaState.addOperands({cpA->getResult(0), cpB->getResult(0)});
    mmaState.addTypes(resType);
    mmaState.addAttribute("sm", rewriter.getI32IntegerAttr(smVersion));
    Operation *mma = rewriter.create(mmaState);

    if (!op->getResults().empty())
      rewriter.replaceOp(op, mma->getResult(0));
    else
      rewriter.eraseOp(op);

    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass definition
// ─────────────────────────────────────────────────────────────────────────────

struct TileIRLoweringPass
    : public PassWrapper<TileIRLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileIRLoweringPass)

  Option<int64_t> tileQ{*this, "tile-q",
                        llvm::cl::desc("Q tile rows for flash attention"),
                        llvm::cl::init(64)};
  Option<int64_t> tileKV{*this, "tile-kv",
                         llvm::cl::desc("KV tile cols for flash attention"),
                         llvm::cl::init(64)};
  Option<int>     smVersion{*this, "sm",
                            llvm::cl::desc("Target SM version (e.g. 90)"),
                            llvm::cl::init(90)};

  StringRef getArgument() const override { return "tessera-tile-ir-lowering"; }
  StringRef getDescription() const override {
    return "Lower schedule.mesh.region { tessera.flash_attn / tessera.matmul }"
           " to FA-4 Tile IR ops (tile.async_copy, tile.mma, tessera.attn.*)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerFlashAttnToTileIR>(ctx, tileQ, tileKV, smVersion);
    patterns.add<LowerMatmulToTileMMA>(ctx, tileQ, tileKV, smVersion);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTileIRLoweringPass() {
  return std::make_unique<TileIRLoweringPass>();
}

} // namespace tessera
