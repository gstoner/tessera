//===- GenerateROCMBlockSparseAttnKernel.cpp - selected-block attention ---===//
//
// Expands `tessera_rocm.block_sparse_attention{,_tiled}` into the DK2 ROCm
// selected-block decode kernels. The cooperative variant maps one workgroup to
// one (B,Hq,Sq) query row: lanes compute selected-token QK scores once, stage
// normalized weights in LDS, then reuse them across value lanes. This removes
// the old row-tiled rung's Dv-fold score recomputation.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
static constexpr int64_t MAX_SELECTED_TOKENS = 256;

void emitBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, bool rowTiled) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  auto sle = arith::CmpIPredicate::sle;
  auto ult = arith::CmpIPredicate::ult;

  b.setInsertionPointToStart(&f.getBody().front());
  Value Q = f.getArgument(0), K = f.getArgument(1), V = f.getArgument(2);
  Value Sel = f.getArgument(3), Qpos = f.getArgument(4), O = f.getArgument(5);
  Value B = f.getArgument(6), Hq = f.getArgument(7), Hkv = f.getArgument(8);
  Value Sq = f.getArgument(9), Sk = f.getArgument(10), D = f.getArgument(11);
  Value Dv = f.getArgument(12), Block = f.getArgument(13);
  Value Ksel = f.getArgument(14), Causal = f.getArgument(15);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), cBD = ci(BD);
  Value f0 = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
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
  Value rows = b.create<arith::MulIOp>(
      loc, b.create<arith::MulIOp>(loc, B, Hq), Sq);
  Value total = b.create<arith::MulIOp>(loc, rows, Dv);
  Value gid;
  if (rowTiled) {
    gid = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, bid, Dv), tid);
  } else {
    gid = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, bid, cBD), tid);
  }
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto rowIf = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());

  Value dv = b.create<arith::RemUIOp>(loc, gid, Dv);
  Value row = b.create<arith::DivUIOp>(loc, gid, Dv);
  Value sq = b.create<arith::RemUIOp>(loc, row, Sq);
  Value tmp1 = b.create<arith::DivUIOp>(loc, row, Sq);
  Value hq = b.create<arith::RemUIOp>(loc, tmp1, Hq);
  Value batch = b.create<arith::DivUIOp>(loc, tmp1, Hq);
  Value group = b.create<arith::DivUIOp>(loc, Hq, Hkv);
  Value hkv = b.create<arith::DivUIOp>(loc, hq, group);
  Value qpos = b.create<memref::LoadOp>(loc, Qpos, ValueRange{sq});

  auto tokenFor = [&](Value ks, Value off) {
    Value selIdx = flat4(batch, hkv, sq, ks, Hkv, Sq, Ksel);
    Value blk = b.create<memref::LoadOp>(loc, Sel, ValueRange{selIdx});
    return b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, blk, Block), off);
  };

  auto scoreAt = [&](Value ks, Value off) -> Value {
    Value tok = tokenFor(ks, off);
    Value validTok = b.create<arith::CmpIOp>(loc, ult, tok, Sk);
    Value causalOk = b.create<arith::CmpIOp>(loc, sle, tok, qpos);
    Value causalEnabled = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                 Causal, c0);
    Value useTok = b.create<arith::OrIOp>(
        loc, b.create<arith::XOrIOp>(loc, causalEnabled,
                                     b.create<arith::ConstantIntOp>(loc, true, 1)),
        causalOk);
    Value valid = b.create<arith::AndIOp>(loc, validTok, useTok);
    auto ifop = b.create<scf::IfOp>(loc, f32, valid, /*withElse=*/true);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(ifop.thenBlock());
      Value acc0 = f0;
      auto loop = b.create<scf::ForOp>(loc, c0, D, c1, ValueRange{acc0});
      {
        OpBuilder::InsertionGuard g2(b);
        b.setInsertionPointToStart(loop.getBody());
        Value d = loop.getInductionVar();
        Value acc = loop.getRegionIterArgs()[0];
        Value q = b.create<memref::LoadOp>(loc, Q,
                                           ValueRange{flat4(batch, hq, sq, d, Hq, Sq, D)});
        Value k = b.create<memref::LoadOp>(loc, K,
                                           ValueRange{flat4(batch, hkv, tok, d, Hkv, Sk, D)});
        b.create<scf::YieldOp>(
            loc, ValueRange{b.create<arith::AddFOp>(
                loc, acc, b.create<arith::MulFOp>(loc, q, k))});
      }
      b.create<scf::YieldOp>(loc, loop.getResults());
      b.setInsertionPointToStart(ifop.elseBlock());
      b.create<scf::YieldOp>(loc, ValueRange{negInf});
    }
    return ifop.getResult(0);
  };

  auto maxOuter = b.create<scf::ForOp>(loc, c0, Ksel, c1, ValueRange{negInf});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(maxOuter.getBody());
    Value ks = maxOuter.getInductionVar();
    Value best0 = maxOuter.getRegionIterArgs()[0];
    auto maxInner = b.create<scf::ForOp>(loc, c0, Block, c1, ValueRange{best0});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(maxInner.getBody());
      Value off = maxInner.getInductionVar();
      Value best = maxInner.getRegionIterArgs()[0];
      Value s = scoreAt(ks, off);
      Value gt = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, s, best);
      b.create<scf::YieldOp>(
          loc, ValueRange{b.create<arith::SelectOp>(loc, gt, s, best)});
    }
    b.create<scf::YieldOp>(loc, maxInner.getResults());
  }
  Value maxScore = maxOuter.getResult(0);

  auto sumOuter = b.create<scf::ForOp>(loc, c0, Ksel, c1, ValueRange{f0, f0});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(sumOuter.getBody());
    Value ks = sumOuter.getInductionVar();
    Value den0 = sumOuter.getRegionIterArgs()[0];
    Value num0 = sumOuter.getRegionIterArgs()[1];
    auto sumInner = b.create<scf::ForOp>(loc, c0, Block, c1, ValueRange{den0, num0});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(sumInner.getBody());
      Value off = sumInner.getInductionVar();
      Value den = sumInner.getRegionIterArgs()[0];
      Value num = sumInner.getRegionIterArgs()[1];
      Value s = scoreAt(ks, off);
      Value finite = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, s, negInf);
      auto ifop = b.create<scf::IfOp>(loc, TypeRange{f32, f32}, finite,
                                      /*withElse=*/true);
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(ifop.thenBlock());
        Value e = b.create<math::ExpOp>(loc, b.create<arith::SubFOp>(loc, s, maxScore));
        Value tok = tokenFor(ks, off);
        Value vv = b.create<memref::LoadOp>(loc, V,
                                            ValueRange{flat4(batch, hkv, tok, dv, Hkv, Sk, Dv)});
        b.create<scf::YieldOp>(
            loc, ValueRange{b.create<arith::AddFOp>(loc, den, e),
                            b.create<arith::AddFOp>(
                                loc, num, b.create<arith::MulFOp>(loc, e, vv))});
        b.setInsertionPointToStart(ifop.elseBlock());
        b.create<scf::YieldOp>(loc, ValueRange{den, num});
      }
      b.create<scf::YieldOp>(loc, ifop.getResults());
    }
    b.create<scf::YieldOp>(loc, sumInner.getResults());
  }
  Value out = b.create<arith::DivFOp>(loc, sumOuter.getResult(1), sumOuter.getResult(0));
  b.create<memref::StoreOp>(loc, out, O, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitCooperativeBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  auto sle = arith::CmpIPredicate::sle;
  auto ult = arith::CmpIPredicate::ult;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  Value weights = f.addWorkgroupAttribution(
      MemRefType::get({MAX_SELECTED_TOKENS}, f32,
                      MemRefLayoutAttrInterface(), ws),
      loc);
  Value stats = f.addWorkgroupAttribution(
      MemRefType::get({2}, f32, MemRefLayoutAttrInterface(), ws), loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value Q = f.getArgument(0), K = f.getArgument(1), V = f.getArgument(2);
  Value Sel = f.getArgument(3), Qpos = f.getArgument(4), O = f.getArgument(5);
  Value Hq = f.getArgument(7), Hkv = f.getArgument(8);
  Value Sq = f.getArgument(9), Sk = f.getArgument(10), D = f.getArgument(11);
  Value Dv = f.getArgument(12), Block = f.getArgument(13);
  Value Ksel = f.getArgument(14), Causal = f.getArgument(15);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1);
  Value f0 = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value negInf = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));

  auto flat4 = [&](Value a, Value bb, Value c, Value d, Value BB, Value C,
                   Value Dd) {
    Value ab = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, a, BB), bb);
    Value abc = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, ab, C), c);
    return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, abc, Dd), d);
  };

  Value row = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value sq = b.create<arith::RemUIOp>(loc, row, Sq);
  Value tmp1 = b.create<arith::DivUIOp>(loc, row, Sq);
  Value hq = b.create<arith::RemUIOp>(loc, tmp1, Hq);
  Value batch = b.create<arith::DivUIOp>(loc, tmp1, Hq);
  Value group = b.create<arith::DivUIOp>(loc, Hq, Hkv);
  Value hkv = b.create<arith::DivUIOp>(loc, hq, group);
  Value qpos = b.create<memref::LoadOp>(loc, Qpos, ValueRange{sq});
  Value ntokens = b.create<arith::MulIOp>(loc, Ksel, Block);

  auto tokenFor = [&](Value t) {
    Value ks = b.create<arith::DivUIOp>(loc, t, Block);
    Value off = b.create<arith::RemUIOp>(loc, t, Block);
    Value selIdx = flat4(batch, hkv, sq, ks, Hkv, Sq, Ksel);
    Value blk = b.create<memref::LoadOp>(loc, Sel, ValueRange{selIdx});
    return b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, blk, Block), off);
  };

  // One lane computes each selected token's QK score exactly once.
  Value activeToken = b.create<arith::CmpIOp>(loc, slt, tid, ntokens);
  auto scoreIf = b.create<scf::IfOp>(loc, f32, activeToken, /*withElse=*/true);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(scoreIf.thenBlock());
    Value tok = tokenFor(tid);
    Value validTok = b.create<arith::CmpIOp>(loc, ult, tok, Sk);
    Value causalOk = b.create<arith::CmpIOp>(loc, sle, tok, qpos);
    Value causalEnabled = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, Causal, c0);
    Value causalPass = b.create<arith::OrIOp>(
        loc, b.create<arith::XOrIOp>(
                 loc, causalEnabled,
                 b.create<arith::ConstantIntOp>(loc, true, 1)),
        causalOk);
    Value valid = b.create<arith::AndIOp>(loc, validTok, causalPass);
    auto validIf = b.create<scf::IfOp>(loc, f32, valid, /*withElse=*/true);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(validIf.thenBlock());
      auto dot = b.create<scf::ForOp>(loc, c0, D, c1, ValueRange{f0});
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(dot.getBody());
        Value d = dot.getInductionVar();
        Value acc = dot.getRegionIterArgs()[0];
        Value qv = b.create<memref::LoadOp>(
            loc, Q, ValueRange{flat4(batch, hq, sq, d, Hq, Sq, D)});
        Value kv = b.create<memref::LoadOp>(
            loc, K, ValueRange{flat4(batch, hkv, tok, d, Hkv, Sk, D)});
        b.create<scf::YieldOp>(
            loc, ValueRange{b.create<arith::AddFOp>(
                     loc, acc, b.create<arith::MulFOp>(loc, qv, kv))});
      }
      b.create<scf::YieldOp>(loc, dot.getResults());
      b.setInsertionPointToStart(validIf.elseBlock());
      b.create<scf::YieldOp>(loc, ValueRange{negInf});
    }
    b.create<scf::YieldOp>(loc, validIf.getResults());
    b.setInsertionPointToStart(scoreIf.elseBlock());
    b.create<scf::YieldOp>(loc, ValueRange{negInf});
  }
  b.create<memref::StoreOp>(loc, scoreIf.getResult(0), weights,
                            ValueRange{tid});
  b.create<gpu::BarrierOp>(loc);

  // Lane 0 normalizes the at-most-256 selected scores in LDS. This serial
  // reduction is tiny relative to the eliminated Dv-fold QK recomputation.
  Value isLeader = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, tid, c0);
  auto leaderIf = b.create<scf::IfOp>(loc, isLeader, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(leaderIf.thenBlock());
    auto maxLoop = b.create<scf::ForOp>(loc, c0, ntokens, c1,
                                        ValueRange{negInf});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(maxLoop.getBody());
      Value t = maxLoop.getInductionVar();
      Value best = maxLoop.getRegionIterArgs()[0];
      Value s = b.create<memref::LoadOp>(loc, weights, ValueRange{t});
      b.create<scf::YieldOp>(
          loc, ValueRange{b.create<arith::MaximumFOp>(loc, best, s)});
    }
    auto sumLoop = b.create<scf::ForOp>(loc, c0, ntokens, c1,
                                        ValueRange{f0});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(sumLoop.getBody());
      Value t = sumLoop.getInductionVar();
      Value den = sumLoop.getRegionIterArgs()[0];
      Value s = b.create<memref::LoadOp>(loc, weights, ValueRange{t});
      Value finite = b.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OGT, s, negInf);
      Value eRaw = b.create<math::ExpOp>(
          loc, b.create<arith::SubFOp>(loc, s, maxLoop.getResult(0)));
      Value e = b.create<arith::SelectOp>(loc, finite, eRaw, f0);
      b.create<memref::StoreOp>(loc, e, weights, ValueRange{t});
      b.create<scf::YieldOp>(
          loc, ValueRange{b.create<arith::AddFOp>(loc, den, e)});
    }
    b.create<memref::StoreOp>(loc, sumLoop.getResult(0), stats, ValueRange{c0});
  }
  b.create<gpu::BarrierOp>(loc);

  // Value lanes reuse the shared normalized weights.
  Value activeDv = b.create<arith::CmpIOp>(loc, slt, tid, Dv);
  auto outIf = b.create<scf::IfOp>(loc, activeDv, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(outIf.thenBlock());
    auto accLoop = b.create<scf::ForOp>(loc, c0, ntokens, c1, ValueRange{f0});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(accLoop.getBody());
      Value t = accLoop.getInductionVar();
      Value acc = accLoop.getRegionIterArgs()[0];
      Value tok = tokenFor(t);
      Value validTok = b.create<arith::CmpIOp>(loc, ult, tok, Sk);
      auto valIf = b.create<scf::IfOp>(loc, f32, validTok, /*withElse=*/true);
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(valIf.thenBlock());
        Value w = b.create<memref::LoadOp>(loc, weights, ValueRange{t});
        Value vv = b.create<memref::LoadOp>(
            loc, V, ValueRange{flat4(batch, hkv, tok, tid, Hkv, Sk, Dv)});
        b.create<scf::YieldOp>(loc, ValueRange{b.create<arith::MulFOp>(loc, w, vv)});
        b.setInsertionPointToStart(valIf.elseBlock());
        b.create<scf::YieldOp>(loc, ValueRange{f0});
      }
      b.create<scf::YieldOp>(
          loc, ValueRange{b.create<arith::AddFOp>(loc, acc, valIf.getResult(0))});
    }
    Value den = b.create<memref::LoadOp>(loc, stats, ValueRange{c0});
    Value out = b.create<arith::DivFOp>(loc, accLoop.getResult(0), den);
    Value outIdx = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, row, Dv), tid);
    b.create<memref::StoreOp>(loc, out, O, ValueRange{outIdx});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMBlockSparseAttnKernelPass
    : PassWrapper<GenerateROCMBlockSparseAttnKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMBlockSparseAttnKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-block-sparse-attn-kernel";
  }
  StringRef getDescription() const final {
    return "Expand tessera_rocm.block_sparse_attention into a selected-block "
           "ROCm sparse attention kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "tessera_rocm.block_sparse_attention" ||
          name == "tessera_rocm.block_sparse_attention_tiled")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      bool rowTiled =
          op->getName().getStringRef() == "tessera_rocm.block_sparse_attention_tiled";
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.block_sparse_attention missing name");
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
      auto imem = MemRefType::get({ShapedType::kDynamic}, b.getIndexType());
      auto fnTy = b.getFunctionType(
          {fmem, fmem, fmem, imem, imem, fmem, idxTy, idxTy, idxTy, idxTy,
           idxTy, idxTy, idxTy, idxTy, idxTy, idxTy},
          {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      if (rowTiled)
        emitCooperativeBody(body, loc, gpuFunc);
      else
        emitBody(body, loc, gpuFunc, /*rowTiled=*/false);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMBlockSparseAttnKernelPass() {
  return std::make_unique<GenerateROCMBlockSparseAttnKernelPass>();
}
