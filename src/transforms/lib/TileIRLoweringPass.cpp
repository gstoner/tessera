//===- TileIRLoweringPass.cpp — Phase 3 ──────────────────────────────────===//
//
// Lowers schedule.mesh.region bodies containing tessera.flash_attn into
// FA-4 Tile IR ops:
//
//   tessera.flash_attn(Q, K, V) {causal, tile_q, tile_kv}
//   tessera.flash_attn(Q, KVCache) {causal, tile_q, tile_kv}
//   →
//   tile.async_copy(Q_tile)
//   + scf.for %kv iter_args(%acc, %m, %l, %producer, %consumer, %boundary)
//       tile.async_copy(KV_tile) + typed wait/pipeline dependencies
//       + tessera_attn.scaled_dot_product
//       + tessera_attn.boundary_mask? + tessera_attn.block_dropout?
//       + tessera_attn.streaming_update
//   + tessera_attn.lse_accumulate
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

#include <algorithm>

using namespace mlir;

namespace tessera {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Emit a tile.async_copy op (string-based to avoid depending on tile_opt_fa4
// dialect headers in the transforms library).
static tile::TileLayoutAttr rowMajorStorageLayout(OpBuilder &b,
                                                  RankedTensorType tensorTy) {
  SmallVector<int64_t> extents(tensorTy.getShape());
  SmallVector<int64_t> strides(extents.size(), 1);
  for (int i = static_cast<int>(extents.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * extents[i + 1];
  SmallVector<StringAttr> axes(extents.size(), b.getStringAttr("m"));
  return tile::TileLayoutAttr::get(
      b.getContext(), extents, strides, axes,
      /*replicaCounts=*/{}, /*replicaStrides=*/{}, /*replicaAxes=*/{},
      /*offset=*/0, /*swizzle=*/tile::TileSwizzleAttr());
}

static Operation *emitAsyncCopy(OpBuilder &b, Location loc, Value src,
                                int64_t tileRows, int64_t tileCols,
                                tile::TileLayoutAttr layout = {}) {
  OperationState st(loc, "tile.async_copy");
  st.addOperands({src});
  st.addAttribute("tile_rows",
                  b.getI64IntegerAttr(tileRows));
  st.addAttribute("tile_cols",
                  b.getI64IntegerAttr(tileCols));
  if (layout)
    st.addAttribute("tile.layout", layout);
  if (auto tensorTy = dyn_cast<RankedTensorType>(src.getType());
      tensorTy && tensorTy.hasStaticShape() && tensorTy.getRank() == 2) {
    if (!layout)
      st.addAttribute("tile.layout", rowMajorStorageLayout(b, tensorTy));
  }
  // The staged tile and its completion token are one contract.  Emitting the
  // token here (rather than waiting for warp specialization) keeps straight-
  // line Graph -> Tile lowering legal too: the wait retires this exact copy
  // and the consumer carries the dependency as SSA.
  st.addTypes({src.getType(), tile::AsyncTokenType::get(b.getContext())});
  return b.create(st);
}

static tile::TileLayoutAttr operandLayout(Operation *op, unsigned operand) {
  std::string name = "tile.operand_layout_" + std::to_string(operand);
  return op->getAttrOfType<tile::TileLayoutAttr>(name);
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
    // The canonical streaming slice owns static rank-2 Q/K/V. Higher-rank
    // batch/head distribution and KV-cache handles remain explicit residuals
    // instead of being silently treated as whole tensors.
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return failure();

    Value Q = op->getOperand(0);
    Value K = op->getOperand(1);
    Value V = op->getOperand(2);
    Location loc = op->getLoc();
    auto qType = dyn_cast<RankedTensorType>(Q.getType());
    auto kType = dyn_cast<RankedTensorType>(K.getType());
    auto vType = dyn_cast<RankedTensorType>(V.getType());
    auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!qType || !kType || !vType || !outType ||
        !qType.hasStaticShape() || !kType.hasStaticShape() ||
        !vType.hasStaticShape() || !outType.hasStaticShape() ||
        qType.getRank() != 2 || kType.getRank() != 2 ||
        vType.getRank() != 2 || outType.getRank() != 2)
      return failure();
    int64_t qRows = qType.getDimSize(0);
    int64_t sk = kType.getDimSize(0);
    int64_t d = qType.getDimSize(1);
    int64_t dv = vType.getDimSize(1);
    if (qRows <= 0 || sk <= 0 || d <= 0 || dv <= 0 ||
        kType.getDimSize(1) != d || vType.getDimSize(0) != sk ||
        outType.getDimSize(0) != qRows || outType.getDimSize(1) != dv ||
        !outType.getElementType().isF32())
      return failure();

    bool causal = false;
    if (auto causalAttr = op->getAttrOfType<BoolAttr>("causal"))
      causal = causalAttr.getValue();
    int64_t windowLeft = -1;
    int64_t windowRight = -1;
    if (auto attr = op->getAttrOfType<IntegerAttr>("window_left"))
      windowLeft = attr.getInt();
    if (auto attr = op->getAttrOfType<IntegerAttr>("window_right"))
      windowRight = attr.getInt();
    if (windowLeft < -1 || windowRight < -1)
      return failure();
    auto dropout = op->getAttrOfType<FloatAttr>("dropout_p");
    auto dropoutSeed = op->getAttrOfType<IntegerAttr>("dropout_seed");
    if (dropout && dropout.getValueAsDouble() > 0.0 && !dropoutSeed)
      return failure();

    // Use tile_kv from op attrs if present (autotuner may have set it). The
    // canonical slice streams KV while retaining one bounded Q tile.
    int64_t tq = qRows;
    int64_t tkv = tileKV;
    if (auto a = op->getAttrOfType<IntegerAttr>("tessera.tile_q"))
      tq = a.getInt();
    if (auto a = op->getAttrOfType<IntegerAttr>("tessera.tile_kv"))
      tkv = a.getInt();
    if (tq != qRows || tkv <= 0)
      return failure();
    tkv = std::min<int64_t>(tkv, sk);
    int64_t paddedSk = ((sk + tkv - 1) / tkv) * tkv;

    // Zero-pad only the KV sequence axis. Padded keys/values are coupled with
    // the explicit logical_sk attribute on the boundary op, so physical tail
    // lanes are never admitted as valid attention positions.
    Value paddedK = K;
    Value paddedV = V;
    if (paddedSk != sk) {
      auto paddedKType =
          RankedTensorType::get({paddedSk, d}, kType.getElementType());
      auto paddedVType =
          RankedTensorType::get({paddedSk, dv}, vType.getElementType());
      Value kZero = arith::ConstantOp::create(
          rewriter, loc, paddedKType, rewriter.getZeroAttr(paddedKType));
      Value vZero = arith::ConstantOp::create(
          rewriter, loc, paddedVType, rewriter.getZeroAttr(paddedVType));
      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
      paddedK = tensor::InsertSliceOp::create(
          rewriter, loc, K, kZero, offsets,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(sk),
                                    rewriter.getIndexAttr(d)},
          strides);
      paddedV = tensor::InsertSliceOp::create(
          rewriter, loc, V, vZero, offsets,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(sk),
                                    rewriter.getIndexAttr(dv)},
          strides);
    }

    // Q is invariant across the KV loop and is staged once.
    Operation *cpQ = emitAsyncCopy(rewriter, loc, Q, tq,
                                   /*d_k=*/d, operandLayout(op, 0));
    emitWaitAsync(rewriter, loc, {cpQ->getResult(1)});

    auto statsType =
        RankedTensorType::get({qRows}, rewriter.getF32Type());
    Value negInf = arith::ConstantOp::create(
        rewriter, loc, statsType,
        DenseElementsAttr::get(
            statsType,
            llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(),
                                  /*negative=*/true)));
    Value zero = arith::ConstantOp::create(
        rewriter, loc, statsType, rewriter.getZeroAttr(statsType));
    Value accInit = arith::ConstantOp::create(
        rewriter, loc, outType, rewriter.getZeroAttr(outType));

    auto makePipelineInit = [&](StringRef role, int64_t phase) -> Value {
      OperationState state(loc, "tile.pipeline_init");
      state.addAttribute("depth", rewriter.getI64IntegerAttr(3));
      state.addAttribute("stage", rewriter.getI64IntegerAttr(0));
      state.addAttribute("phase", rewriter.getI64IntegerAttr(phase));
      state.addAttribute("role", rewriter.getStringAttr(role));
      state.addTypes(tile::PipelineStateType::get(rewriter.getContext()));
      return rewriter.create(state)->getResult(0);
    };
    Value producerInit = makePipelineInit("producer", 1);
    Value consumerInit = makePipelineInit("consumer", 0);
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upper = arith::ConstantIndexOp::create(rewriter, loc, paddedSk);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, tkv);

    auto kvLoop = scf::ForOp::create(
        rewriter, loc, zeroIndex, upper, step,
        ValueRange{accInit, negInf, zero, producerInit, consumerInit,
                   zeroIndex});
    kvLoop->setAttr("tessera.streaming_attention", rewriter.getUnitAttr());
    kvLoop->setAttr("tessera.logical_sk", rewriter.getI64IntegerAttr(sk));
    kvLoop->setAttr("tessera.kv_block", rewriter.getI64IntegerAttr(tkv));
    kvLoop->setAttr(
        "tile.pipeline_depths",
        tile::TilePipelineDepthsAttr::get(rewriter.getContext(), 2, 3, 2));
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(kvLoop.getBody());
      Value kv = kvLoop.getInductionVar();
      Value acc = kvLoop.getRegionIterArg(0);
      Value runningM = kvLoop.getRegionIterArg(1);
      Value runningL = kvLoop.getRegionIterArg(2);
      Value producerState = kvLoop.getRegionIterArg(3);
      Value consumerState = kvLoop.getRegionIterArg(4);
      Value boundary = kvLoop.getRegionIterArg(5);

      auto kTileType =
          RankedTensorType::get({tkv, d}, kType.getElementType());
      auto vTileType =
          RankedTensorType::get({tkv, dv}, vType.getElementType());
      SmallVector<OpFoldResult> offsets{kv, rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
      Value kSlice = tensor::ExtractSliceOp::create(
          rewriter, loc, kTileType, paddedK, offsets,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(tkv),
                                    rewriter.getIndexAttr(d)},
          strides);
      Value vSlice = tensor::ExtractSliceOp::create(
          rewriter, loc, vTileType, paddedV, offsets,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(tkv),
                                    rewriter.getIndexAttr(dv)},
          strides);
      Operation *cpK = emitAsyncCopy(rewriter, loc, kSlice, tkv, d,
                                     operandLayout(op, 1));
      Operation *cpV = emitAsyncCopy(rewriter, loc, vSlice, tkv, dv,
                                     operandLayout(op, 2));
      emitWaitAsync(rewriter, loc,
                    {cpK->getResult(1), cpV->getResult(1)});

      OperationState producerAdvance(loc, "tile.pipeline_advance");
      producerAdvance.addOperands(
          {producerState, cpK->getResult(1), cpV->getResult(1)});
      producerAdvance.addTypes(
          tile::PipelineStateType::get(rewriter.getContext()));
      Value nextProducer =
          rewriter.create(producerAdvance)->getResult(0);

      auto scoresType =
          RankedTensorType::get({qRows, tkv}, rewriter.getF32Type());
      Operation *sdp = emitAttnOp(
          rewriter, loc, "tessera_attn.scaled_dot_product",
          {cpQ->getResult(0), cpK->getResult(0)}, {scoresType},
          {rewriter.getNamedAttr("scale",
                                 rewriter.getF32FloatAttr(-1.0f))});
      Value scores = sdp->getResult(0);

      if (causal || windowLeft >= 0 || windowRight >= 0 || paddedSk != sk) {
        SmallVector<NamedAttribute> boundaryAttrs = {
            rewriter.getNamedAttr("causal", rewriter.getBoolAttr(causal)),
            rewriter.getNamedAttr("window_left",
                                  rewriter.getI64IntegerAttr(windowLeft)),
            rewriter.getNamedAttr("window_right",
                                  rewriter.getI64IntegerAttr(windowRight)),
            rewriter.getNamedAttr("logical_sk",
                                  rewriter.getI64IntegerAttr(sk))};
        Operation *mask = emitAttnOp(
            rewriter, loc, "tessera_attn.boundary_mask",
            {scores, zeroIndex, boundary}, {scoresType}, boundaryAttrs);
        scores = mask->getResult(0);
      }

      if (dropout && dropout.getValueAsDouble() > 0.0) {
        Operation *drop = emitAttnOp(
            rewriter, loc, "tessera_attn.block_dropout",
            {scores, boundary}, {scoresType},
            {rewriter.getNamedAttr(
                 "dropout_p",
                 rewriter.getF32FloatAttr(
                     static_cast<float>(dropout.getValueAsDouble()))),
             rewriter.getNamedAttr(
                 "seed",
                 rewriter.getI64IntegerAttr(dropoutSeed.getInt()))});
        scores = drop->getResult(0);
      }

      Operation *update = emitAttnOp(
          rewriter, loc, "tessera_attn.streaming_update",
          {scores, cpV->getResult(0), runningM, runningL, acc},
          {outType, statsType, statsType});

      OperationState consumerAdvance(loc, "tile.pipeline_advance");
      consumerAdvance.addOperands(
          {consumerState, cpK->getResult(1), cpV->getResult(1),
           update->getResult(0)});
      consumerAdvance.addTypes(
          tile::PipelineStateType::get(rewriter.getContext()));
      Value nextConsumer =
          rewriter.create(consumerAdvance)->getResult(0);
      Value nextBoundary =
          arith::AddIOp::create(rewriter, loc, boundary, step);
      scf::YieldOp::create(
          rewriter, loc,
          ValueRange{update->getResult(0), update->getResult(1),
                     update->getResult(2), nextProducer, nextConsumer,
                     nextBoundary});
    }

    Operation *lseAcc = emitAttnOp(
        rewriter, loc, "tessera_attn.lse_accumulate",
        {kvLoop.getResult(0), kvLoop.getResult(1), kvLoop.getResult(2)},
        {outType, statsType});

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
                                   operandLayout(op, 0));
    Operation *cpB = emitAsyncCopy(rewriter, loc, B, -1, tileN,
                                   operandLayout(op, 1));
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

// Fuse the canonical K-step's explicit tensor accumulation into the Tile MMA.
// The surrounding scf.for continues to own the reduction and pipeline state;
// this rewrite only maps one target-neutral step onto the Tile async/MMA
// dependency contract.
struct LowerKReductionAddToTileMMA : public RewritePattern {
  int smVersion;

  LowerKReductionAddToTileMMA(MLIRContext *ctx, int sm)
      : RewritePattern("tessera.add", /*benefit=*/3, ctx), smVersion(sm) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("tessera.k_reduction_accumulate") ||
        op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();

    Operation *matmul = nullptr;
    Value accumulator;
    for (unsigned index = 0; index < 2; ++index) {
      Operation *candidate = op->getOperand(index).getDefiningOp();
      if (candidate &&
          candidate->getName().getStringRef() == "tessera.matmul" &&
          candidate->hasAttr("tessera.canonical_k_step")) {
        matmul = candidate;
        accumulator = op->getOperand(1 - index);
        break;
      }
    }
    if (!matmul || !matmul->hasOneUse() || matmul->getNumOperands() < 2)
      return failure();

    Location loc = op->getLoc();
    int64_t tm = 16, tn = 16, tk = 16;
    if (auto attr = matmul->getAttrOfType<IntegerAttr>("tessera.tile_m"))
      tm = attr.getInt();
    if (auto attr = matmul->getAttrOfType<IntegerAttr>("tessera.tile_n"))
      tn = attr.getInt();
    if (auto attr = matmul->getAttrOfType<IntegerAttr>("tessera.tile_k"))
      tk = attr.getInt();

    Operation *cpA = emitAsyncCopy(rewriter, loc, matmul->getOperand(0), tm, tk,
                                   operandLayout(matmul, 0));
    Operation *cpB = emitAsyncCopy(rewriter, loc, matmul->getOperand(1), tk, tn,
                                   operandLayout(matmul, 1));
    emitWaitAsync(rewriter, loc, {cpA->getResult(1), cpB->getResult(1)});

    OperationState mmaState(loc, "tile.mma");
    mmaState.addOperands({cpA->getResult(0), cpB->getResult(0), accumulator,
                          cpA->getResult(1), cpB->getResult(1)});
    mmaState.addTypes(op->getResult(0).getType());
    mmaState.addAttribute("sm", rewriter.getI32IntegerAttr(smVersion));
    mmaState.addAttribute("tessera.canonical_k_step",
                          rewriter.getUnitAttr());
    mmaState.addAttribute("tessera.tile_m", rewriter.getI64IntegerAttr(tm));
    mmaState.addAttribute("tessera.tile_n", rewriter.getI64IntegerAttr(tn));
    mmaState.addAttribute("tessera.tile_k", rewriter.getI64IntegerAttr(tk));
    if (auto policy = matmul->getAttr("numeric_policy"))
      mmaState.addAttribute("numeric_policy", policy);
    Operation *mma = rewriter.create(mmaState);

    rewriter.replaceOp(op, mma->getResult(0));
    rewriter.eraseOp(matmul);
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
    patterns.add<LowerKReductionAddToTileMMA>(ctx, smVersion);
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
                        << static_cast<int>(smVersion)
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
