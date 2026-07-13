//===- GenerateROCMBlockSparseTopKKernel.cpp - sparse block top-k select ---===//
//
// Expands `tessera_rocm.block_sparse_topk_select` into a GPU-resident selector
// for DK2 MSA/NSA score rows. One workgroup owns one (B,Hkv,Sq) row; lanes scan
// block candidates in parallel and reduce deterministic (score, block-id)
// winners through LDS. Ties choose the lowest block id. This keeps the
// selected-block ABI stable while scaling selection to larger block counts.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <limits>

using namespace mlir;

namespace {

// Two Wave32s amortize row setup while keeping the repeated top-k barriers and
// LDS footprint bounded.  A 256-lane group was measured to lose badly once
// enough independent score rows were available to fill the machine.
static constexpr int64_t SERIAL_BD = 256;
static constexpr int64_t COOP_BD = 64;

void emitTopKBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  auto ne = arith::CmpIPredicate::ne;
  auto sle = arith::CmpIPredicate::sle;

  b.setInsertionPointToStart(&f.getBody().front());
  Value Scores = f.getArgument(0);
  Value Qpos = f.getArgument(1);
  Value Out = f.getArgument(2);
  Value B = f.getArgument(3), Hkv = f.getArgument(4), Sq = f.getArgument(5);
  Value Nb = f.getArgument(6), Ksel = f.getArgument(7);
  Value Block = f.getArgument(8), Causal = f.getArgument(9);
  Value ForceLocal = f.getArgument(10);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), cBD = ci(SERIAL_BD);
  Value negInf = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));

  auto flat4 = [&](Value a, Value bb, Value c, Value d, Value BB, Value C,
                   Value Dd) {
    Value ab = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, a, BB), bb);
    Value abc = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, ab, C), c);
    return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, abc, Dd), d);
  };

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value row = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, cBD), tid);
  Value rows = b.create<arith::MulIOp>(loc, b.create<arith::MulIOp>(loc, B, Hkv), Sq);
  Value inb = b.create<arith::CmpIOp>(loc, slt, row, rows);
  auto rowIf = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());

  Value sq = b.create<arith::RemUIOp>(loc, row, Sq);
  Value tmp = b.create<arith::DivUIOp>(loc, row, Sq);
  Value h = b.create<arith::RemUIOp>(loc, tmp, Hkv);
  Value batch = b.create<arith::DivUIOp>(loc, tmp, Hkv);
  Value qpos = b.create<memref::LoadOp>(loc, Qpos, ValueRange{sq});
  Value localRaw = b.create<arith::DivUIOp>(loc, qpos, Block);
  Value maxBlock = b.create<arith::SubIOp>(loc, Nb, c1);
  Value localGt = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                          localRaw, maxBlock);
  Value local = b.create<arith::SelectOp>(loc, localGt, maxBlock, localRaw);
  Value causalOn = b.create<arith::CmpIOp>(loc, ne, Causal, c0);
  Value forceOn = b.create<arith::CmpIOp>(loc, ne, ForceLocal, c0);

  auto slotLoop = b.create<scf::ForOp>(loc, c0, Ksel, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(slotLoop.getBody());
    Value slot = slotLoop.getInductionVar();
    Value isFirst = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, slot, c0);
    Value emitLocal = b.create<arith::AndIOp>(loc, forceOn, isFirst);
    auto choose = b.create<scf::IfOp>(loc, b.getIndexType(), emitLocal,
                                      /*withElse=*/true);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(choose.thenBlock());
      b.create<scf::YieldOp>(loc, ValueRange{local});

      b.setInsertionPointToStart(choose.elseBlock());
      Value falseVal = b.create<arith::ConstantIntOp>(loc, false, 1);
      auto scan = b.create<scf::ForOp>(loc, c0, Nb, c1,
                                       ValueRange{negInf, c0, falseVal});
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(scan.getBody());
        Value blk = scan.getInductionVar();
        Value bestScore = scan.getRegionIterArgs()[0];
        Value bestBlk = scan.getRegionIterArgs()[1];
        Value found = scan.getRegionIterArgs()[2];
        Value future = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               blk, local);
        Value pastOk = b.create<arith::OrIOp>(
            loc, b.create<arith::XOrIOp>(
                     loc, causalOn, b.create<arith::ConstantIntOp>(loc, true, 1)),
            b.create<arith::CmpIOp>(loc, sle, blk, local));
        Value notForcedLocal = b.create<arith::OrIOp>(
            loc, b.create<arith::XOrIOp>(
                     loc, forceOn, b.create<arith::ConstantIntOp>(loc, true, 1)),
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, blk, local));
        Value notDup = b.create<arith::ConstantIntOp>(loc, true, 1);
        auto prevLoop = b.create<scf::ForOp>(loc, c0, slot, c1, ValueRange{notDup});
        {
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(prevLoop.getBody());
          Value prev = prevLoop.getInductionVar();
          Value ok = prevLoop.getRegionIterArgs()[0];
          Value prevIdx = flat4(batch, h, sq, prev, Hkv, Sq, Ksel);
          Value old = b.create<memref::LoadOp>(loc, Out, ValueRange{prevIdx});
          Value neq = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, old, blk);
          b.create<scf::YieldOp>(loc, ValueRange{b.create<arith::AndIOp>(loc, ok, neq)});
        }
        Value valid = b.create<arith::AndIOp>(
            loc, b.create<arith::AndIOp>(loc, pastOk, notForcedLocal),
            prevLoop.getResult(0));
        Value scoreIdx = flat4(batch, h, sq, blk, Hkv, Sq, Nb);
        Value raw = b.create<memref::LoadOp>(loc, Scores, ValueRange{scoreIdx});
        Value score = b.create<arith::SelectOp>(loc, valid, raw, negInf);
        Value gt = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                           score, bestScore);
        Value foundNext = b.create<arith::OrIOp>(loc, found, valid);
        b.create<scf::YieldOp>(
            loc, ValueRange{b.create<arith::SelectOp>(loc, gt, score, bestScore),
                            b.create<arith::SelectOp>(loc, gt, blk, bestBlk),
                            foundNext});
        (void)future;
      }
      // When causal masking leaves fewer score-valid blocks than Ksel, keep
      // filler ids unique. The attention kernel later masks future filler
      // tokens, while duplicate past/local ids would incorrectly double-count.
      Value trueVal = b.create<arith::ConstantIntOp>(loc, true, 1);
      auto fallback = b.create<scf::ForOp>(loc, c0, Nb, c1,
                                           ValueRange{c0, falseVal});
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(fallback.getBody());
        Value blk = fallback.getInductionVar();
        Value chosen = fallback.getRegionIterArgs()[0];
        Value have = fallback.getRegionIterArgs()[1];
        Value notDup = b.create<arith::ConstantIntOp>(loc, true, 1);
        auto prevLoop = b.create<scf::ForOp>(loc, c0, slot, c1,
                                             ValueRange{notDup});
        {
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(prevLoop.getBody());
          Value prev = prevLoop.getInductionVar();
          Value ok = prevLoop.getRegionIterArgs()[0];
          Value prevIdx = flat4(batch, h, sq, prev, Hkv, Sq, Ksel);
          Value old = b.create<memref::LoadOp>(loc, Out, ValueRange{prevIdx});
          Value neq = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, old, blk);
          b.create<scf::YieldOp>(loc, ValueRange{b.create<arith::AndIOp>(loc, ok, neq)});
        }
        Value take = b.create<arith::AndIOp>(
            loc, b.create<arith::XOrIOp>(loc, have, trueVal),
            prevLoop.getResult(0));
        b.create<scf::YieldOp>(
            loc, ValueRange{b.create<arith::SelectOp>(loc, take, blk, chosen),
                            b.create<arith::OrIOp>(loc, have, take)});
      }
      b.create<scf::YieldOp>(
          loc, ValueRange{b.create<arith::SelectOp>(
              loc, scan.getResult(2), scan.getResult(1), fallback.getResult(0))});
    }
    Value outIdx = flat4(batch, h, sq, slot, Hkv, Sq, Ksel);
    b.create<memref::StoreOp>(loc, choose.getResult(0), Out, ValueRange{outIdx});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitTopKCooperativeBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  auto ne = arith::CmpIPredicate::ne;
  auto sle = arith::CmpIPredicate::sle;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  Value laneScores = f.addWorkgroupAttribution(
      MemRefType::get({COOP_BD}, f32, MemRefLayoutAttrInterface(), ws), loc);
  Value laneIds = f.addWorkgroupAttribution(
      MemRefType::get({COOP_BD}, b.getIndexType(), MemRefLayoutAttrInterface(), ws),
      loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value Scores = f.getArgument(0);
  Value Qpos = f.getArgument(1);
  Value Out = f.getArgument(2);
  Value Hkv = f.getArgument(4), Sq = f.getArgument(5);
  Value Nb = f.getArgument(6), Ksel = f.getArgument(7);
  Value Block = f.getArgument(8), Causal = f.getArgument(9);
  Value ForceLocal = f.getArgument(10);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), cBD = ci(COOP_BD);
  Value negInf = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
  Value trueVal = b.create<arith::ConstantIntOp>(loc, true, 1);

  auto flat4 = [&](Value a, Value bb, Value c, Value d, Value BB, Value C,
                   Value Dd) {
    Value ab = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, a, BB), bb);
    Value abc = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, ab, C), c);
    return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, abc, Dd), d);
  };

  Value row = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value sq = b.create<arith::RemUIOp>(loc, row, Sq);
  Value tmp = b.create<arith::DivUIOp>(loc, row, Sq);
  Value h = b.create<arith::RemUIOp>(loc, tmp, Hkv);
  Value batch = b.create<arith::DivUIOp>(loc, tmp, Hkv);
  Value qpos = b.create<memref::LoadOp>(loc, Qpos, ValueRange{sq});
  Value localRaw = b.create<arith::DivUIOp>(loc, qpos, Block);
  Value maxBlock = b.create<arith::SubIOp>(loc, Nb, c1);
  Value localGt = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sgt, localRaw, maxBlock);
  Value local = b.create<arith::SelectOp>(loc, localGt, maxBlock, localRaw);
  Value causalOn = b.create<arith::CmpIOp>(loc, ne, Causal, c0);
  Value forceOn = b.create<arith::CmpIOp>(loc, ne, ForceLocal, c0);
  Value isLeader = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, tid, c0);

  auto notPreviouslySelected = [&](Value blk, Value slot) {
    auto prev = b.create<scf::ForOp>(loc, c0, slot, c1, ValueRange{trueVal});
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(prev.getBody());
      Value p = prev.getInductionVar();
      Value ok = prev.getRegionIterArgs()[0];
      Value oldIdx = flat4(batch, h, sq, p, Hkv, Sq, Ksel);
      Value old = b.create<memref::LoadOp>(loc, Out, ValueRange{oldIdx});
      Value neqOld = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, old, blk);
      b.create<scf::YieldOp>(
          loc, ValueRange{b.create<arith::AndIOp>(loc, ok, neqOld)});
    }
    return prev.getResult(0);
  };

  auto slots = b.create<scf::ForOp>(loc, c0, Ksel, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(slots.getBody());
    Value slot = slots.getInductionVar();
    Value first = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, slot, c0);
    Value emitLocal = b.create<arith::AndIOp>(loc, forceOn, first);
    auto choose = b.create<scf::IfOp>(loc, emitLocal, /*withElse=*/true);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(choose.thenBlock());
      auto lead = b.create<scf::IfOp>(loc, isLeader, /*withElse=*/false);
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(lead.thenBlock());
        Value outIdx = flat4(batch, h, sq, slot, Hkv, Sq, Ksel);
        b.create<memref::StoreOp>(loc, local, Out, ValueRange{outIdx});
      }

      b.setInsertionPointToStart(choose.elseBlock());
      // Each lane scans a strided subset of candidate blocks.
      auto scan = b.create<scf::ForOp>(loc, tid, Nb, cBD,
                                       ValueRange{negInf, Nb});
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(scan.getBody());
        Value blk = scan.getInductionVar();
        Value bestScore = scan.getRegionIterArgs()[0];
        Value bestId = scan.getRegionIterArgs()[1];
        Value pastOk = b.create<arith::OrIOp>(
            loc, b.create<arith::XOrIOp>(loc, causalOn, trueVal),
            b.create<arith::CmpIOp>(loc, sle, blk, local));
        Value notLocal = b.create<arith::OrIOp>(
            loc, b.create<arith::XOrIOp>(loc, forceOn, trueVal),
            b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::ne, blk, local));
        Value valid = b.create<arith::AndIOp>(
            loc, b.create<arith::AndIOp>(loc, pastOk, notLocal),
            notPreviouslySelected(blk, slot));
        Value scoreIdx = flat4(batch, h, sq, blk, Hkv, Sq, Nb);
        Value raw = b.create<memref::LoadOp>(loc, Scores, ValueRange{scoreIdx});
        Value gt = b.create<arith::CmpFOp>(
            loc, arith::CmpFPredicate::OGT, raw, bestScore);
        Value eq = b.create<arith::CmpFOp>(
            loc, arith::CmpFPredicate::OEQ, raw, bestScore);
        Value lowerId = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, blk, bestId);
        Value better = b.create<arith::AndIOp>(
            loc, valid,
            b.create<arith::OrIOp>(
                loc, gt, b.create<arith::AndIOp>(loc, eq, lowerId)));
        b.create<scf::YieldOp>(
            loc, ValueRange{b.create<arith::SelectOp>(loc, better, raw, bestScore),
                            b.create<arith::SelectOp>(loc, better, blk, bestId)});
      }
      b.create<memref::StoreOp>(loc, scan.getResult(0), laneScores,
                                ValueRange{tid});
      b.create<memref::StoreOp>(loc, scan.getResult(1), laneIds,
                                ValueRange{tid});
      b.create<gpu::BarrierOp>(loc);

      // Reduce the 256 lane-local winners cooperatively.  Keeping this as an
      // unrolled tree avoids replacing the old serial Nb scan with a new
      // 256-entry lane-0 scan for every selected slot.
      for (int64_t stride = COOP_BD / 2; stride >= 1; stride /= 2) {
        Value cStride = ci(stride);
        Value active = b.create<arith::CmpIOp>(loc, slt, tid, cStride);
        auto reduceStep =
            b.create<scf::IfOp>(loc, active, /*withElse=*/false);
        {
          OpBuilder::InsertionGuard g3(b);
          b.setInsertionPointToStart(reduceStep.thenBlock());
          Value otherLane = b.create<arith::AddIOp>(loc, tid, cStride);
          Value score =
              b.create<memref::LoadOp>(loc, laneScores, ValueRange{tid});
          Value id = b.create<memref::LoadOp>(loc, laneIds, ValueRange{tid});
          Value otherScore = b.create<memref::LoadOp>(
              loc, laneScores, ValueRange{otherLane});
          Value otherId = b.create<memref::LoadOp>(
              loc, laneIds, ValueRange{otherLane});
          Value otherValid = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, otherId, Nb);
          Value gt = b.create<arith::CmpFOp>(
              loc, arith::CmpFPredicate::OGT, otherScore, score);
          Value eq = b.create<arith::CmpFOp>(
              loc, arith::CmpFPredicate::OEQ, otherScore, score);
          Value lowerId = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, otherId, id);
          Value takeOther = b.create<arith::AndIOp>(
              loc, otherValid,
              b.create<arith::OrIOp>(
                  loc, gt, b.create<arith::AndIOp>(loc, eq, lowerId)));
          b.create<memref::StoreOp>(
              loc, b.create<arith::SelectOp>(loc, takeOther, otherScore, score),
              laneScores, ValueRange{tid});
          b.create<memref::StoreOp>(
              loc, b.create<arith::SelectOp>(loc, takeOther, otherId, id),
              laneIds, ValueRange{tid});
        }
        b.create<gpu::BarrierOp>(loc);
      }

      auto reduceLead = b.create<scf::IfOp>(loc, isLeader, /*withElse=*/false);
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(reduceLead.thenBlock());
        Value winnerId =
            b.create<memref::LoadOp>(loc, laneIds, ValueRange{c0});
        Value have = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, winnerId, Nb);
        // If causal masking leaves no score-valid candidate, preserve the old
        // unique-filler contract by choosing the lowest unselected block.
        auto fallback = b.create<scf::ForOp>(loc, c0, Nb, c1,
                                             ValueRange{Nb});
        {
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(fallback.getBody());
          Value blk = fallback.getInductionVar();
          Value chosen = fallback.getRegionIterArgs()[0];
          Value unset = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, chosen, Nb);
          Value take = b.create<arith::AndIOp>(
              loc, unset, notPreviouslySelected(blk, slot));
          b.create<scf::YieldOp>(
              loc, ValueRange{b.create<arith::SelectOp>(loc, take, blk, chosen)});
        }
        Value winner = b.create<arith::SelectOp>(
            loc, have, winnerId, fallback.getResult(0));
        Value outIdx = flat4(batch, h, sq, slot, Hkv, Sq, Ksel);
        b.create<memref::StoreOp>(loc, winner, Out, ValueRange{outIdx});
      }
    }
    // The next slot reads all prior winners, so bracket every iteration.
    b.create<gpu::BarrierOp>(loc);
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMBlockSparseTopKKernelPass
    : PassWrapper<GenerateROCMBlockSparseTopKKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMBlockSparseTopKKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-block-sparse-topk-kernel";
  }
  StringRef getDescription() const final {
    return "Expand tessera_rocm.block_sparse_topk_select into a GPU-resident "
           "selected-block top-k kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera_rocm.block_sparse_topk_select")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.block_sparse_topk_select missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto fmem = MemRefType::get({ShapedType::kDynamic}, b.getF32Type());
      auto imem = MemRefType::get({ShapedType::kDynamic}, idxTy);
      auto fnTy = b.getFunctionType(
          {fmem, imem, imem, idxTy, idxTy, idxTy, idxTy, idxTy, idxTy, idxTy, idxTy},
          {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      if (auto strategy = op->getAttrOfType<StringAttr>("strategy");
          strategy && strategy.getValue() == "serial")
        emitTopKBody(body, loc, gpuFunc);
      else
        emitTopKCooperativeBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMBlockSparseTopKKernelPass() {
  return std::make_unique<GenerateROCMBlockSparseTopKKernelPass>();
}
