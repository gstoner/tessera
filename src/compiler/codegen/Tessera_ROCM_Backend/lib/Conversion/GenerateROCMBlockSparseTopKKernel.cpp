//===- GenerateROCMBlockSparseTopKKernel.cpp - sparse block top-k select ---===//
//
// Expands `tessera_rocm.block_sparse_topk_select` into a GPU-resident selector
// for DK2 MSA/NSA score rows. One thread owns one (B,Hkv,Sq) row and writes
// deterministic selected block ids `[B,Hkv,Sq,Ksel]`; ties choose the lowest
// block id. This keeps the selected-block ABI stable while moving top-k off the
// host for large sparse-attention batches.
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

static constexpr int64_t BD = 256;

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
  Value c0 = ci(0), c1 = ci(1), cBD = ci(BD);
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
      if (op->getName().getStringRef() == "tessera_rocm.block_sparse_topk_select")
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
      emitTopKBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMBlockSparseTopKKernelPass() {
  return std::make_unique<GenerateROCMBlockSparseTopKKernelPass>();
}
