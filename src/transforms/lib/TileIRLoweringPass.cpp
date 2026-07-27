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
#include "tessera/Dialect/Attn/AttnDialect.h"
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

// Distribute a static rank-4 [B,H,S,D] attention operation into explicit
// batch/query-head loops whose body is the canonical rank-2 streaming
// operation.  The rank-2 rewrite below remains the single owner of online
// softmax, boundary/dropout counters, ragged zero-fill, async tokens, and
// pipeline state.  Backends therefore see distribution wrapped around the
// same recurrence instead of a second rank-4 attention implementation.
struct DistributeRank4FlashAttn : public RewritePattern {
  DistributeRank4FlashAttn(MLIRContext *ctx)
      : RewritePattern("tessera.flash_attn", /*benefit=*/3, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if ((op->getNumOperands() != 3 && op->getNumOperands() != 4) ||
        op->getNumResults() != 1)
      return failure();

    Value q = op->getOperand(0);
    Value k = op->getOperand(1);
    Value v = op->getOperand(2);
    Value bias = op->getNumOperands() == 4 ? op->getOperand(3) : Value();
    auto qType = dyn_cast<RankedTensorType>(q.getType());
    auto kType = dyn_cast<RankedTensorType>(k.getType());
    auto vType = dyn_cast<RankedTensorType>(v.getType());
    auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto biasType =
        bias ? dyn_cast<RankedTensorType>(bias.getType()) : RankedTensorType();
    if (!qType || !kType || !vType || !outType ||
        !qType.hasStaticShape() || !kType.hasStaticShape() ||
        !vType.hasStaticShape() || !outType.hasStaticShape() ||
        qType.getRank() != 4 || kType.getRank() != 4 ||
        vType.getRank() != 4 || outType.getRank() != 4)
      return failure();
    if (bias &&
        (!biasType || !biasType.hasStaticShape() ||
         (biasType.getRank() != 3 && biasType.getRank() != 4) ||
         (biasType.getDimSize(0) != 1 &&
          biasType.getDimSize(0) != qType.getDimSize(0)) ||
         (biasType.getRank() == 4 &&
          biasType.getDimSize(1) != 1 &&
          biasType.getDimSize(1) != qType.getDimSize(1)) ||
         biasType.getDimSize(biasType.getRank() - 2) !=
             qType.getDimSize(2) ||
         biasType.getDimSize(biasType.getRank() - 1) !=
             kType.getDimSize(2) ||
         !biasType.getElementType().isF32()))
      return failure();

    int64_t batch = qType.getDimSize(0);
    int64_t queryHeads = qType.getDimSize(1);
    int64_t kvHeads = kType.getDimSize(1);
    int64_t queryRows = qType.getDimSize(2);
    int64_t keyRows = kType.getDimSize(2);
    int64_t headDim = qType.getDimSize(3);
    int64_t valueDim = vType.getDimSize(3);
    if (batch <= 0 || queryHeads <= 0 || kvHeads <= 0 || queryRows <= 0 ||
        keyRows <= 0 || headDim <= 0 || valueDim <= 0 ||
        queryHeads % kvHeads != 0 ||
        kType.getDimSize(0) != batch || vType.getDimSize(0) != batch ||
        vType.getDimSize(1) != kvHeads ||
        kType.getDimSize(3) != headDim ||
        vType.getDimSize(2) != keyRows ||
        outType.getDimSize(0) != batch ||
        outType.getDimSize(1) != queryHeads ||
        outType.getDimSize(2) != queryRows ||
        outType.getDimSize(3) != valueDim ||
        !outType.getElementType().isF32())
      return failure();

    Location loc = op->getLoc();
    Value outputInit = arith::ConstantOp::create(
        rewriter, loc, outType, rewriter.getZeroAttr(outType));
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value batchUpper =
        arith::ConstantIndexOp::create(rewriter, loc, batch);
    Value headUpper =
        arith::ConstantIndexOp::create(rewriter, loc, queryHeads);
    Value groupSize = arith::ConstantIndexOp::create(
        rewriter, loc, queryHeads / kvHeads);

    auto batchLoop = scf::ForOp::create(
        rewriter, loc, zero, batchUpper, one, ValueRange{outputInit});
    batchLoop->setAttr("tessera.attention_distribution",
                       rewriter.getStringAttr("batch"));
    batchLoop->setAttr("tessera.batch_count",
                       rewriter.getI64IntegerAttr(batch));
    {
      OpBuilder::InsertionGuard batchGuard(rewriter);
      rewriter.setInsertionPointToStart(batchLoop.getBody());
      Value batchIndex = batchLoop.getInductionVar();
      Value batchOutput = batchLoop.getRegionIterArg(0);
      auto headLoop = scf::ForOp::create(
          rewriter, loc, zero, headUpper, one, ValueRange{batchOutput});
      headLoop->setAttr("tessera.attention_distribution",
                        rewriter.getStringAttr("query_head"));
      headLoop->setAttr("tessera.query_head_count",
                        rewriter.getI64IntegerAttr(queryHeads));
      headLoop->setAttr("tessera.kv_head_count",
                        rewriter.getI64IntegerAttr(kvHeads));
      headLoop->setAttr("tessera.gqa_group_size",
                        rewriter.getI64IntegerAttr(queryHeads / kvHeads));
      {
        OpBuilder::InsertionGuard headGuard(rewriter);
        rewriter.setInsertionPointToStart(headLoop.getBody());
        Value queryHead = headLoop.getInductionVar();
        Value headOutput = headLoop.getRegionIterArg(0);
        Value kvHead =
            arith::DivUIOp::create(rewriter, loc, queryHead, groupSize);

        auto qTileType = RankedTensorType::get(
            {queryRows, headDim}, qType.getElementType());
        auto kTileType = RankedTensorType::get(
            {keyRows, headDim}, kType.getElementType());
        auto vTileType = RankedTensorType::get(
            {keyRows, valueDim}, vType.getElementType());
        auto outTileType = RankedTensorType::get(
            {queryRows, valueDim}, outType.getElementType());
        SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
        SmallVector<OpFoldResult> qOffsets{
            batchIndex, queryHead, rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult> kvOffsets{
            batchIndex, kvHead, rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult> qSizes{
            rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(queryRows),
            rewriter.getIndexAttr(headDim)};
        SmallVector<OpFoldResult> kSizes{
            rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(keyRows),
            rewriter.getIndexAttr(headDim)};
        SmallVector<OpFoldResult> vSizes{
            rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(keyRows),
            rewriter.getIndexAttr(valueDim)};
        Value qTile = tensor::ExtractSliceOp::create(
            rewriter, loc, qTileType, q, qOffsets, qSizes, strides);
        Value kTile = tensor::ExtractSliceOp::create(
            rewriter, loc, kTileType, k, kvOffsets, kSizes, strides);
        Value vTile = tensor::ExtractSliceOp::create(
            rewriter, loc, vTileType, v, kvOffsets, vSizes, strides);
        Value biasTile;
        if (bias) {
          auto biasTileType = RankedTensorType::get(
              {queryRows, keyRows}, biasType.getElementType());
          OpFoldResult biasBatch =
              biasType.getDimSize(0) == 1 ? OpFoldResult(rewriter.getIndexAttr(0))
                                          : OpFoldResult(batchIndex);
          if (biasType.getRank() == 4) {
            OpFoldResult biasHead =
                biasType.getDimSize(1) == 1
                    ? OpFoldResult(rewriter.getIndexAttr(0))
                    : OpFoldResult(queryHead);
            biasTile = tensor::ExtractSliceOp::create(
                rewriter, loc, biasTileType, bias,
                SmallVector<OpFoldResult>{
                    biasBatch, biasHead, rewriter.getIndexAttr(0),
                    rewriter.getIndexAttr(0)},
                SmallVector<OpFoldResult>{
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                    rewriter.getIndexAttr(queryRows),
                    rewriter.getIndexAttr(keyRows)},
                SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(1)));
          } else {
            biasTile = tensor::ExtractSliceOp::create(
                rewriter, loc, biasTileType, bias,
                SmallVector<OpFoldResult>{biasBatch, rewriter.getIndexAttr(0),
                                          rewriter.getIndexAttr(0)},
                SmallVector<OpFoldResult>{
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(queryRows),
                    rewriter.getIndexAttr(keyRows)},
                SmallVector<OpFoldResult>(3, rewriter.getIndexAttr(1)));
          }
        }

        OperationState rank2State(loc, "tessera.flash_attn");
        rank2State.addOperands({qTile, kTile, vTile});
        if (biasTile)
          rank2State.addOperands(biasTile);
        rank2State.addTypes(outTileType);
        rank2State.addAttributes(op->getAttrs());
        rank2State.addAttribute("tessera.rank4_distributed",
                                rewriter.getUnitAttr());
        rank2State.addAttribute(
            "tessera.gqa_group_size",
            rewriter.getI64IntegerAttr(queryHeads / kvHeads));
        Operation *rank2 = rewriter.create(rank2State);

        Value inserted = tensor::InsertSliceOp::create(
            rewriter, loc, rank2->getResult(0), headOutput, qOffsets,
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(queryRows),
                rewriter.getIndexAttr(valueDim)},
            strides);
        scf::YieldOp::create(rewriter, loc, inserted);
      }
      rewriter.setInsertionPointAfter(headLoop);
      scf::YieldOp::create(rewriter, loc, headLoop.getResult(0));
    }

    rewriter.replaceOp(op, batchLoop.getResult(0));
    return success();
  }
};

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
    if ((op->getNumOperands() != 3 && op->getNumOperands() != 4) ||
        op->getNumResults() != 1)
      return failure();

    Value Q = op->getOperand(0);
    Value K = op->getOperand(1);
    Value V = op->getOperand(2);
    Value bias = op->getNumOperands() == 4 ? op->getOperand(3) : Value();
    Location loc = op->getLoc();
    auto qType = dyn_cast<RankedTensorType>(Q.getType());
    auto kType = dyn_cast<RankedTensorType>(K.getType());
    auto vType = dyn_cast<RankedTensorType>(V.getType());
    auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto biasType =
        bias ? dyn_cast<RankedTensorType>(bias.getType()) : RankedTensorType();
    if (!qType || !kType || !vType || !outType ||
        !qType.hasStaticShape() || !kType.hasStaticShape() ||
        !vType.hasStaticShape() || !outType.hasStaticShape() ||
        qType.getRank() != 2 || kType.getRank() != 2 ||
        vType.getRank() != 2 || outType.getRank() != 2)
      return failure();
    if (bias &&
        (!biasType || !biasType.hasStaticShape() || biasType.getRank() != 2 ||
         biasType.getDimSize(0) != qType.getDimSize(0) ||
         biasType.getDimSize(1) != kType.getDimSize(0) ||
         !biasType.getElementType().isF32()))
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
    Value paddedBias = bias;
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
      if (bias) {
        auto paddedBiasType =
            RankedTensorType::get({qRows, paddedSk}, biasType.getElementType());
        Value biasZero = arith::ConstantOp::create(
            rewriter, loc, paddedBiasType,
            rewriter.getZeroAttr(paddedBiasType));
        paddedBias = tensor::InsertSliceOp::create(
            rewriter, loc, bias, biasZero, offsets,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(qRows),
                                      rewriter.getIndexAttr(sk)},
            strides);
      }
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
      float scale = -1.0f;
      if (auto scaleAttr = op->getAttrOfType<FloatAttr>("scale"))
        scale = static_cast<float>(scaleAttr.getValueAsDouble());
      Operation *sdp = emitAttnOp(
          rewriter, loc, "tessera_attn.scaled_dot_product",
          {cpQ->getResult(0), cpK->getResult(0)}, {scoresType},
          {rewriter.getNamedAttr("scale",
                                 rewriter.getF32FloatAttr(scale))});
      Value scores = sdp->getResult(0);

      if (bias) {
        auto biasBlockType =
            RankedTensorType::get({qRows, tkv}, biasType.getElementType());
        Value biasSlice = tensor::ExtractSliceOp::create(
            rewriter, loc, biasBlockType, paddedBias,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(0), kv},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(qRows),
                                      rewriter.getIndexAttr(tkv)},
            strides);
        Operation *addBias = emitAttnOp(
            rewriter, loc, "tessera_attn.score_bias", {scores, biasSlice},
            {scoresType});
        scores = addBias->getResult(0);
      }
      if (auto softcap = op->getAttrOfType<FloatAttr>("softcap");
          softcap && softcap.getValueAsDouble() > 0.0) {
        Operation *cap = emitAttnOp(
            rewriter, loc, "tessera_attn.softcap", {scores}, {scoresType},
            {rewriter.getNamedAttr(
                "cap", rewriter.getF32FloatAttr(
                           static_cast<float>(softcap.getValueAsDouble())))});
        scores = cap->getResult(0);
      }

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

// Materialize the target-neutral attention VJP contract as tensor-valued loop
// bodies.  The phase operations own the mathematical block update; scf.for
// owns distribution, split workspace, and deterministic reduction order.
struct LowerAttentionBackwardToLoops : public RewritePattern {
  LowerAttentionBackwardToLoops(MLIRContext *ctx)
      : RewritePattern("tessera_attn.backward", /*benefit=*/4, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 5 || op->getNumResults() != 3)
      return failure();
    auto qType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto kType = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
    auto dQType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto dKType = dyn_cast<RankedTensorType>(op->getResult(1).getType());
    auto dVType = dyn_cast<RankedTensorType>(op->getResult(2).getType());
    auto splitAttr = op->getAttrOfType<IntegerAttr>("split_count");
    auto queryBlockAttr = op->getAttrOfType<IntegerAttr>("query_block");
    auto keyBlockAttr = op->getAttrOfType<IntegerAttr>("key_block");
    if (!qType || !kType || !dQType || !dKType || !dVType ||
        !qType.hasStaticShape() || !kType.hasStaticShape() ||
        !dQType.hasStaticShape() || !dKType.hasStaticShape() ||
        !dVType.hasStaticShape() || !splitAttr || !queryBlockAttr ||
        !keyBlockAttr)
      return failure();

    int64_t batch = qType.getDimSize(0);
    int64_t queryHeads = qType.getDimSize(1);
    int64_t queryRows = qType.getDimSize(2);
    int64_t kvHeads = kType.getDimSize(1);
    int64_t keyRows = kType.getDimSize(2);
    int64_t splitCount = splitAttr.getInt();
    int64_t queryBlock = queryBlockAttr.getInt();
    int64_t keyBlock = keyBlockAttr.getInt();
    if (batch <= 0 || queryHeads <= 0 || queryRows <= 0 || kvHeads <= 0 ||
        keyRows <= 0 || splitCount < 2 || queryBlock <= 0 || keyBlock <= 0)
      return failure();

    Location loc = op->getLoc();
    auto zeroTensor = [&](RankedTensorType type) {
      return arith::ConstantOp::create(rewriter, loc, type,
                                       rewriter.getZeroAttr(type))
          .getResult();
    };
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value batchUpper =
        arith::ConstantIndexOp::create(rewriter, loc, batch);
    Value queryHeadUpper =
        arith::ConstantIndexOp::create(rewriter, loc, queryHeads);
    Value kvHeadUpper =
        arith::ConstantIndexOp::create(rewriter, loc, kvHeads);
    Value queryUpper =
        arith::ConstantIndexOp::create(rewriter, loc, queryRows);
    Value keyUpper =
        arith::ConstantIndexOp::create(rewriter, loc, keyRows);
    Value splitUpper =
        arith::ConstantIndexOp::create(rewriter, loc, splitCount);
    Value queryStep =
        arith::ConstantIndexOp::create(rewriter, loc, queryBlock);
    Value keyStep =
        arith::ConstantIndexOp::create(rewriter, loc, keyBlock);

    auto tagLoop = [&](scf::ForOp loop, StringRef phase) {
      loop->setAttr("tessera.attention_backward_phase",
                    rewriter.getStringAttr(phase));
      loop->setAttr("tessera.tensor_iter_args", rewriter.getUnitAttr());
    };

    // dQ is query-head owned. Each [query_block,key_block] update carries the
    // full tensor in SSA, leaving physical subview/register allocation to the
    // target schedule.
    Value dQInit = zeroTensor(dQType);
    auto dqBatch = scf::ForOp::create(
        rewriter, loc, zero, batchUpper, one, ValueRange{dQInit});
    tagLoop(dqBatch, "dq.batch");
    {
      OpBuilder::InsertionGuard g0(rewriter);
      rewriter.setInsertionPointToStart(dqBatch.getBody());
      auto dqHead = scf::ForOp::create(
          rewriter, loc, zero, queryHeadUpper, one,
          ValueRange{dqBatch.getRegionIterArg(0)});
      tagLoop(dqHead, "dq.query_head");
      {
        OpBuilder::InsertionGuard g1(rewriter);
        rewriter.setInsertionPointToStart(dqHead.getBody());
        auto dqQuery = scf::ForOp::create(
            rewriter, loc, zero, queryUpper, queryStep,
            ValueRange{dqHead.getRegionIterArg(0)});
        tagLoop(dqQuery, "dq.query_block");
        {
          OpBuilder::InsertionGuard g2(rewriter);
          rewriter.setInsertionPointToStart(dqQuery.getBody());
          auto dqKey = scf::ForOp::create(
              rewriter, loc, zero, keyUpper, keyStep,
              ValueRange{dqQuery.getRegionIterArg(0)});
          tagLoop(dqKey, "dq.key_block");
          {
            OpBuilder::InsertionGuard g3(rewriter);
            rewriter.setInsertionPointToStart(dqKey.getBody());
            Operation *update = emitAttnOp(
                rewriter, loc, "tessera_attn.backward_dq_block",
                {op->getOperand(0), op->getOperand(1), op->getOperand(2),
                 op->getOperand(3), op->getOperand(4),
                 dqBatch.getInductionVar(), dqHead.getInductionVar(),
                 dqQuery.getInductionVar(), dqKey.getInductionVar(),
                 dqKey.getRegionIterArg(0)},
                {dQType}, op->getAttrs());
            scf::YieldOp::create(rewriter, loc, update->getResult(0));
          }
          rewriter.setInsertionPointAfter(dqKey);
          scf::YieldOp::create(rewriter, loc, dqKey.getResult(0));
        }
        rewriter.setInsertionPointAfter(dqQuery);
        scf::YieldOp::create(rewriter, loc, dqQuery.getResult(0));
      }
      rewriter.setInsertionPointAfter(dqHead);
      scf::YieldOp::create(rewriter, loc, dqHead.getResult(0));
    }

    SmallVector<int64_t> partialKShape{splitCount};
    partialKShape.append(dKType.getShape().begin(), dKType.getShape().end());
    SmallVector<int64_t> partialVShape{splitCount};
    partialVShape.append(dVType.getShape().begin(), dVType.getShape().end());
    auto partialKType =
        RankedTensorType::get(partialKShape, rewriter.getF32Type());
    auto partialVType =
        RankedTensorType::get(partialVShape, rewriter.getF32Type());
    Value partialKInit = zeroTensor(partialKType);
    Value partialVInit = zeroTensor(partialVType);

    // The canonical ownership loop is B/Hkv/split/Q-block/KV-block. No phase
    // can alias another split's partial, and both partial tensors are explicit
    // iter_args at every level.
    auto partialBatch = scf::ForOp::create(
        rewriter, loc, zero, batchUpper, one,
        ValueRange{partialKInit, partialVInit});
    tagLoop(partialBatch, "dkdv.batch");
    {
      OpBuilder::InsertionGuard g0(rewriter);
      rewriter.setInsertionPointToStart(partialBatch.getBody());
      auto partialHead = scf::ForOp::create(
          rewriter, loc, zero, kvHeadUpper, one,
          partialBatch.getRegionIterArgs());
      tagLoop(partialHead, "dkdv.kv_head");
      {
        OpBuilder::InsertionGuard g1(rewriter);
        rewriter.setInsertionPointToStart(partialHead.getBody());
        auto partialSplit = scf::ForOp::create(
            rewriter, loc, zero, splitUpper, one,
            partialHead.getRegionIterArgs());
        tagLoop(partialSplit, "dkdv.split");
        partialSplit->setAttr("tessera.workspace_owner",
                              rewriter.getStringAttr("program_launch"));
        {
          OpBuilder::InsertionGuard g2(rewriter);
          rewriter.setInsertionPointToStart(partialSplit.getBody());
          auto partialQuery = scf::ForOp::create(
              rewriter, loc, zero, queryUpper, queryStep,
              partialSplit.getRegionIterArgs());
          tagLoop(partialQuery, "dkdv.query_block");
          {
            OpBuilder::InsertionGuard g3(rewriter);
            rewriter.setInsertionPointToStart(partialQuery.getBody());
            auto partialKey = scf::ForOp::create(
                rewriter, loc, zero, keyUpper, keyStep,
                partialQuery.getRegionIterArgs());
            tagLoop(partialKey, "dkdv.key_block");
            {
              OpBuilder::InsertionGuard g4(rewriter);
              rewriter.setInsertionPointToStart(partialKey.getBody());
              Operation *update = emitAttnOp(
                  rewriter, loc, "tessera_attn.backward_dkdv_block",
                  {op->getOperand(0), op->getOperand(1), op->getOperand(2),
                   op->getOperand(3), op->getOperand(4),
                   partialBatch.getInductionVar(),
                   partialHead.getInductionVar(),
                   partialSplit.getInductionVar(),
                   partialQuery.getInductionVar(),
                   partialKey.getInductionVar(),
                   partialKey.getRegionIterArg(0),
                   partialKey.getRegionIterArg(1)},
                  {partialKType, partialVType}, op->getAttrs());
              scf::YieldOp::create(
                  rewriter, loc,
                  ValueRange{update->getResult(0), update->getResult(1)});
            }
            rewriter.setInsertionPointAfter(partialKey);
            scf::YieldOp::create(rewriter, loc, partialKey.getResults());
          }
          rewriter.setInsertionPointAfter(partialQuery);
          scf::YieldOp::create(rewriter, loc, partialQuery.getResults());
        }
        rewriter.setInsertionPointAfter(partialSplit);
        scf::YieldOp::create(rewriter, loc, partialSplit.getResults());
      }
      rewriter.setInsertionPointAfter(partialHead);
      scf::YieldOp::create(rewriter, loc, partialHead.getResults());
    }

    Value dKInit = zeroTensor(dKType);
    Value dVInit = zeroTensor(dVType);
    auto reduceBatch = scf::ForOp::create(
        rewriter, loc, zero, batchUpper, one, ValueRange{dKInit, dVInit});
    tagLoop(reduceBatch, "reduce.batch");
    {
      OpBuilder::InsertionGuard g0(rewriter);
      rewriter.setInsertionPointToStart(reduceBatch.getBody());
      auto reduceHead = scf::ForOp::create(
          rewriter, loc, zero, kvHeadUpper, one,
          reduceBatch.getRegionIterArgs());
      tagLoop(reduceHead, "reduce.kv_head");
      {
        OpBuilder::InsertionGuard g1(rewriter);
        rewriter.setInsertionPointToStart(reduceHead.getBody());
        auto reduceSplit = scf::ForOp::create(
            rewriter, loc, zero, splitUpper, one,
            reduceHead.getRegionIterArgs());
        tagLoop(reduceSplit, "reduce.fixed_order_split");
        reduceSplit->setAttr("tessera.reduction_order",
                             rewriter.getStringAttr("ascending_split"));
        {
          OpBuilder::InsertionGuard g2(rewriter);
          rewriter.setInsertionPointToStart(reduceSplit.getBody());
          Operation *reduce = emitAttnOp(
              rewriter, loc, "tessera_attn.backward_reduce_split",
              {partialBatch.getResult(0), partialBatch.getResult(1),
               reduceBatch.getInductionVar(), reduceHead.getInductionVar(),
               reduceSplit.getInductionVar(),
               reduceSplit.getRegionIterArg(0),
               reduceSplit.getRegionIterArg(1)},
              {dKType, dVType}, op->getAttrs());
          scf::YieldOp::create(
              rewriter, loc,
              ValueRange{reduce->getResult(0), reduce->getResult(1)});
        }
        rewriter.setInsertionPointAfter(reduceSplit);
        scf::YieldOp::create(rewriter, loc, reduceSplit.getResults());
      }
      rewriter.setInsertionPointAfter(reduceHead);
      scf::YieldOp::create(rewriter, loc, reduceHead.getResults());
    }

    rewriter.replaceOp(op, ValueRange{dqBatch.getResult(0),
                                      reduceBatch.getResult(0),
                                      reduceBatch.getResult(1)});
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
    if (matmul->hasAttr("tessera.ragged_zero_pad"))
      mmaState.addAttribute("tessera.ragged_zero_pad",
                            rewriter.getUnitAttr());
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
    registry.insert<tessera::attn::TesseraAttnDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerAttentionBackwardToLoops>(ctx);
    patterns.add<DistributeRank4FlashAttn>(ctx);
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
