//===- TileIRLoweringPass.cpp — Phase 3 ──────────────────────────────────===//
//
// Lowers schedule.mesh.region bodies containing tessera.flash_attn into
// FA-4 Tile IR ops:
//
//   tessera.flash_attn(Q, K, V) {causal, tile_q, tile_kv}
//   tessera.flash_attn(Q, KVCache) {causal, tile_q, tile_kv}
//   →
//   tile.async_copy(Q_tile) + tessera_attn.scaled_dot_product
//   + tessera_attn.causal_mask? + tessera_attn.online_softmax
//   + tessera_attn.lse_accumulate + tile.wait_async
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
#include "Tessera/Dialect/Tile/TileDialect.h"
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
                                int64_t tileRows, int64_t tileCols,
                                StringRef nvidiaLayout = {}) {
  OperationState st(loc, "tile.async_copy");
  st.addOperands({src});
  st.addAttribute("tile_rows",
                  b.getI64IntegerAttr(tileRows));
  st.addAttribute("tile_cols",
                  b.getI64IntegerAttr(tileCols));
  if (!nvidiaLayout.empty())
    st.addAttribute("tessera.nvidia.layout",
                    b.getStringAttr(nvidiaLayout));
  // The staged tile and its completion token are one contract.  Emitting the
  // token here (rather than waiting for warp specialization) keeps straight-
  // line Graph -> Tile lowering legal too: the wait retires this exact copy
  // and the consumer carries the dependency as SSA.
  st.addTypes({src.getType(), tile::AsyncTokenType::get(b.getContext())});
  return b.create(st);
}

static StringRef nvidiaOperandLayout(Operation *op, unsigned operand) {
  std::string name = "tessera.nvidia.operand_layout_" +
                     std::to_string(operand);
  if (auto attr = op->getAttrOfType<StringAttr>(name))
    return attr.getValue();
  return {};
}

// Emit a tile.wait_async op that retires the named in-flight copies.
static Operation *emitWaitAsync(OpBuilder &b, Location loc,
                                ValueRange tokens) {
  OperationState st(loc, "tile.wait_async");
  st.addOperands(tokens);
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
    // flash_attn takes either Q,K,V or Q,KVCache.
    if (op->getNumOperands() < 2)
      return failure();

    Value Q = op->getOperand(0);
    Value K = op->getOperand(1);
    Value V = (op->getNumOperands() >= 3) ? op->getOperand(2) : op->getOperand(1);
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

    // ── Emit tile.async_copy for Q and either K/V tensors or a staged cache ──
    Operation *cpQ = emitAsyncCopy(rewriter, loc, Q, tq,
                                   /*d_k inferred*/ -1,
                                   nvidiaOperandLayout(op, 0));
    Operation *cpK = emitAsyncCopy(rewriter, loc, K, tkv, -1,
                                   nvidiaOperandLayout(op, 1));
    Operation *cpV = emitAsyncCopy(rewriter, loc, V, tkv, -1,
                                   nvidiaOperandLayout(op, 2));

    // Barrier: wait for all async copies to complete before compute.
    emitWaitAsync(rewriter, loc,
                  {cpQ->getResult(1), cpK->getResult(1), cpV->getResult(1)});

    // ── Emit online softmax initialiser constants ────────────────────────────
    // running_m init: -inf  (f32 per row)
    // running_l init:  0.0  (f32 per row)
    Value negInf = rewriter.create<arith::ConstantFloatOp>(
        loc, rewriter.getF32Type(),
        llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), /*neg=*/true));
    Value zero = rewriter.create<arith::ConstantFloatOp>(
        loc, rewriter.getF32Type(), llvm::APFloat(0.0f));

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
        rewriter, loc, "tessera_attn.scaled_dot_product",
        {cpQ->getResult(0), cpK->getResult(0)}, {outType}, sdpAttrs);

    Value scores = sdp->getResult(0);

    // ── Optional causal mask ─────────────────────────────────────────────────
    if (causal) {
      SmallVector<NamedAttribute> cmAttrs = {
          rewriter.getNamedAttr("q_offset", rewriter.getI64IntegerAttr(0)),
          rewriter.getNamedAttr("kv_offset", rewriter.getI64IntegerAttr(0))};
      Operation *cm = emitAttnOp(rewriter, loc, "tessera_attn.causal_mask",
                                  {scores}, {outType}, cmAttrs);
      scores = cm->getResult(0);
    }

    // ── Online softmax ───────────────────────────────────────────────────────
    // Emit with sentinel init values; the actual init is in the loop preamble.
    Operation *osm = emitAttnOp(
        rewriter, loc, "tessera_attn.online_softmax",
        {scores, negInf, zero, zero /* acc placeholder */},
        {outType, rewriter.getF32Type(), rewriter.getF32Type()});

    // ── LSE accumulate (finalisation) ────────────────────────────────────────
    Operation *lseAcc = emitAttnOp(
        rewriter, loc, "tessera_attn.lse_accumulate",
        {osm->getResult(0), osm->getResult(1), osm->getResult(2)},
        {outType, rewriter.getF32Type()});

    // Store LSE for backward pass.
    emitAttnOp(rewriter, loc, "tessera_attn.lse.save",
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
    Operation *cpA = emitAsyncCopy(rewriter, loc, A, tileM, -1,
                                   nvidiaOperandLayout(op, 0));
    Operation *cpB = emitAsyncCopy(rewriter, loc, B, -1, tileN,
                                   nvidiaOperandLayout(op, 1));
    emitWaitAsync(rewriter, loc, {cpA->getResult(1), cpB->getResult(1)});

    // tile.mma — the WGMMA/WMMA selector is resolved by NVWGMMALoweringPass.
    OperationState mmaState(loc, "tile.mma");
    mmaState.addOperands({cpA->getResult(0), cpB->getResult(0),
                          cpA->getResult(1), cpB->getResult(1)});
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

struct LowerSchedulePrefetchToTileCopy : public RewritePattern {
  LowerSchedulePrefetchToTileCopy(MLIRContext *ctx)
      : RewritePattern("schedule.prefetch", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();

    Operation *copy = emitAsyncCopy(rewriter, op->getLoc(), op->getOperand(0),
                                    /*tileRows=*/-1, /*tileCols=*/-1);
    rewriter.replaceOp(op, copy->getResult(0));
    return success();
  }
};

// Preserve the verified Graph control ABI at the Tile boundary.  Payload attrs
// are copied verbatim: CUDA codegen consumes them inside one kernel launch.
struct LowerControlToTileIR : public RewritePattern {
  std::string tileName;

  LowerControlToTileIR(MLIRContext *ctx, StringRef graphName,
                       StringRef tileName)
      : RewritePattern(graphName, /*benefit=*/3, ctx),
        tileName(tileName.str()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OperationState state(op->getLoc(), tileName);
    state.addOperands(op->getOperands());
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());
    state.addAttribute("source", rewriter.getStringAttr(
                                     op->getName().getStringRef()));
    Operation *tile = rewriter.create(state);
    rewriter.replaceOp(op, tile->getResults());
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass definition
// ─────────────────────────────────────────────────────────────────────────────

struct TileIRLoweringPass
    : public PassWrapper<TileIRLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileIRLoweringPass)

  TileIRLoweringPass() = default;
  explicit TileIRLoweringPass(int sm) { smVersion = sm; }
  TileIRLoweringPass(const TileIRLoweringPass &other)
      : PassWrapper(other) {}

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
    return "Lower Graph/Schedule attention, matmul, and executable bounded "
           "control-flow contracts to typed Tile IR";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<tessera::tile::TesseraTileDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerFlashAttnToTileIR>(ctx, tileQ, tileKV, smVersion);
    patterns.add<LowerMatmulToTileMMA>(ctx, tileQ, tileKV, smVersion);
    patterns.add<LowerSchedulePrefetchToTileCopy>(ctx);
    patterns.add<LowerControlToTileIR>(
        ctx, "tessera.control_for", "tile.control_for");
    patterns.add<LowerControlToTileIR>(
        ctx, "tessera.control_if", "tile.control_if");
    patterns.add<LowerControlToTileIR>(
        ctx, "tessera.control_while", "tile.control_while");
    patterns.add<LowerControlToTileIR>(
        ctx, "tessera.control_scan", "tile.control_scan");

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
      signalPassFailure();
      return;
    }

    // Decision #21: applyPatternsGreedily returns success even when it
    // matched nothing, so a supported source op that failed a
    // pattern guard (e.g. unsupported operand count / shape) would silently
    // survive and the module would be reported as "GPU-lowered". Refuse that:
    // any surviving target op is a hard lowering failure with a named diagnostic.
    WalkResult residual = getOperation()->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.flash_attn" || name == "tessera.matmul" ||
          name == "tessera.control_for" || name == "tessera.control_if" ||
          name == "tessera.control_while" || name == "tessera.control_scan") {
        op->emitError() << "[TILE_IR_LOWERING] '" << name
                        << "' was not lowered to FA-4 Tile IR for sm_"
                        << smVersion
                        << " (unsupported operands/shape); refusing to report a "
                           "partially-lowered module as success";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (residual.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTileIRLoweringPass(int sm) {
  return std::make_unique<TileIRLoweringPass>(sm);
}

} // namespace tessera
