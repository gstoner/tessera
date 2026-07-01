//===- GenerateROCMDSparkDraftBlockKernel.cpp - DSpark draft block kernel --===//
//
// Expands a `tessera_rocm.dspark_draft_block` directive into a fused f32 GPU
// kernel for the DS2 vanilla draft-block contract:
//
//   state_d = tanh(state_{d-1}@hidden_proj
//                  + embedding(prev_token)@token_proj
//                  + optional markov(prev_token))
//   logits_d = state_d@out_proj
//   confidence_d = state_d@confidence_proj
//   prev_token = argmax(logits_d)
//
// One GPU thread owns one (batch, anchor) chain and runs the block-size
// recurrence serially.  This is the first native ABI proof for the fused DS2
// surface; later tuning can split the H/V reductions cooperatively without
// changing the runtime contract.
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

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitDSparkBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  Type i64 = b.getI64Type();
  auto slt = arith::CmpIPredicate::slt;
  auto eq = arith::CmpIPredicate::eq;
  auto ogt = arith::CmpFPredicate::OGT;

  b.setInsertionPointToStart(&f.getBody().front());
  Value targetHidden = f.getArgument(0);
  Value prevTokens = f.getArgument(1);
  Value anchors = f.getArgument(2);
  Value embedding = f.getArgument(3);
  Value hiddenProj = f.getArgument(4);
  Value tokenProj = f.getArgument(5);
  Value outProj = f.getArgument(6);
  Value confidenceProj = f.getArgument(7);
  Value markov = f.getArgument(8);
  Value logits = f.getArgument(9);
  Value confidence = f.getArgument(10);
  Value tokens = f.getArgument(11);
  Value hidden = f.getArgument(12);
  Value B = f.getArgument(13);
  Value S = f.getArgument(14);
  Value H = f.getArgument(15);
  Value A = f.getArgument(16);
  Value D = f.getArgument(17);
  Value V = f.getArgument(18);
  Value hasMarkovFlag = f.getArgument(19);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), cBD = ci(BD);
  Value f0 = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value negInf =
      b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(-3.4028235e38f));
  Value hasMarkov = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                            hasMarkovFlag, c0);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value totalRows = b.create<arith::MulIOp>(loc, B, A);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, totalRows);
  auto rowIf = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());

  Value bidx = b.create<arith::DivUIOp>(loc, gid, A);
  Value aidx = b.create<arith::RemUIOp>(loc, gid, A);
  Value anchorI64 = b.create<memref::LoadOp>(loc, anchors, ValueRange{aidx});
  Value anchor = b.create<arith::IndexCastOp>(loc, b.getIndexType(), anchorI64);
  Value prev0I64 = b.create<memref::LoadOp>(loc, prevTokens, ValueRange{bidx});
  Value prev0 = b.create<arith::IndexCastOp>(loc, b.getIndexType(), prev0I64);

  auto flat3 = [&](Value i, Value j, Value k, Value J, Value K) -> Value {
    return b.create<arith::AddIOp>(
        loc, b.create<arith::AddIOp>(
                 loc, b.create<arith::MulIOp>(
                          loc, b.create<arith::MulIOp>(loc, i, J), K),
                 b.create<arith::MulIOp>(loc, j, K)),
        k);
  };
  auto flat4 = [&](Value i, Value j, Value k, Value l, Value J, Value K,
                   Value L) -> Value {
    return b.create<arith::AddIOp>(
        loc, b.create<arith::AddIOp>(
                 loc, b.create<arith::AddIOp>(
                          loc, b.create<arith::MulIOp>(
                                   loc, b.create<arith::MulIOp>(
                                            loc, b.create<arith::MulIOp>(loc, i, J),
                                            K),
                                   L),
                          b.create<arith::MulIOp>(
                              loc, b.create<arith::MulIOp>(loc, j, K), L)),
                 b.create<arith::MulIOp>(loc, k, L)),
        l);
  };
  auto flat2 = [&](Value i, Value j, Value J) -> Value {
    return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, i, J), j);
  };

  auto dLoop = b.create<scf::ForOp>(loc, c0, D, c1, ValueRange{prev0});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(dLoop.getBody());
    Value d = dLoop.getInductionVar();
    Value prev = dLoop.getRegionIterArgs()[0];
    Value isFirst = b.create<arith::CmpIOp>(loc, eq, d, c0);

    auto loadPrevState = [&](Value h) -> Value {
      auto ifop = b.create<scf::IfOp>(loc, f32, isFirst, /*withElse=*/true);
      {
        OpBuilder::InsertionGuard gi(b);
        b.setInsertionPointToStart(ifop.thenBlock());
        Value thenIdx = flat3(bidx, anchor, h, S, H);
        Value thenVal =
            b.create<memref::LoadOp>(loc, targetHidden, ValueRange{thenIdx});
        b.create<scf::YieldOp>(loc, ValueRange{thenVal});
        b.setInsertionPointToStart(ifop.elseBlock());
        Value dm1 = b.create<arith::SubIOp>(loc, d, c1);
        Value elseIdx = flat4(bidx, aidx, dm1, h, A, D, H);
        Value elseVal =
            b.create<memref::LoadOp>(loc, hidden, ValueRange{elseIdx});
        b.create<scf::YieldOp>(loc, ValueRange{elseVal});
      }
      return ifop.getResult(0);
    };

    auto hLoop = b.create<scf::ForOp>(loc, c0, H, c1);
    {
      OpBuilder::InsertionGuard gh(b);
      b.setInsertionPointToStart(hLoop.getBody());
      Value h = hLoop.getInductionVar();

      Value stateAcc;
      {
        auto lp = b.create<scf::ForOp>(loc, c0, H, c1, ValueRange{f0});
        OpBuilder::InsertionGuard gl(b);
        b.setInsertionPointToStart(lp.getBody());
        Value j = lp.getInductionVar();
        Value acc = lp.getRegionIterArgs()[0];
        Value s = loadPrevState(j);
        Value w = b.create<memref::LoadOp>(
            loc, hiddenProj, ValueRange{flat2(j, h, H)});
        Value prod = b.create<arith::MulFOp>(loc, s, w);
        Value next = b.create<arith::AddFOp>(loc, acc, prod);
        b.create<scf::YieldOp>(loc, ValueRange{next});
        stateAcc = lp.getResult(0);
      }

      Value tokenAcc;
      {
        auto lp = b.create<scf::ForOp>(loc, c0, H, c1, ValueRange{stateAcc});
        OpBuilder::InsertionGuard gl(b);
        b.setInsertionPointToStart(lp.getBody());
        Value j = lp.getInductionVar();
        Value acc = lp.getRegionIterArgs()[0];
        Value e = b.create<memref::LoadOp>(
            loc, embedding, ValueRange{flat2(prev, j, H)});
        Value w =
            b.create<memref::LoadOp>(loc, tokenProj, ValueRange{flat2(j, h, H)});
        Value prod = b.create<arith::MulFOp>(loc, e, w);
        Value next = b.create<arith::AddFOp>(loc, acc, prod);
        b.create<scf::YieldOp>(loc, ValueRange{next});
        tokenAcc = lp.getResult(0);
      }

      auto markIf = b.create<scf::IfOp>(loc, f32, hasMarkov, /*withElse=*/true);
      {
        OpBuilder::InsertionGuard gm(b);
        b.setInsertionPointToStart(markIf.thenBlock());
        Value m =
            b.create<memref::LoadOp>(loc, markov, ValueRange{flat2(prev, h, H)});
        Value next = b.create<arith::AddFOp>(loc, tokenAcc, m);
        b.create<scf::YieldOp>(loc, ValueRange{next});
        b.setInsertionPointToStart(markIf.elseBlock());
        b.create<scf::YieldOp>(loc, ValueRange{tokenAcc});
      }
      Value st = b.create<math::TanhOp>(loc, markIf.getResult(0));
      Value outIdx = flat4(bidx, aidx, d, h, A, D, H);
      b.create<memref::StoreOp>(loc, st, hidden, ValueRange{outIdx});
    }

    Value confAcc;
    {
      auto lp = b.create<scf::ForOp>(loc, c0, H, c1, ValueRange{f0});
      OpBuilder::InsertionGuard gl(b);
      b.setInsertionPointToStart(lp.getBody());
      Value h = lp.getInductionVar();
      Value acc = lp.getRegionIterArgs()[0];
      Value st =
          b.create<memref::LoadOp>(loc, hidden, ValueRange{flat4(bidx, aidx, d, h, A, D, H)});
      Value w = b.create<memref::LoadOp>(loc, confidenceProj, ValueRange{h});
      Value prod = b.create<arith::MulFOp>(loc, st, w);
      Value next = b.create<arith::AddFOp>(loc, acc, prod);
      b.create<scf::YieldOp>(loc, ValueRange{next});
      confAcc = lp.getResult(0);
    }
    b.create<memref::StoreOp>(
        loc, confAcc, confidence,
        ValueRange{b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(
                     loc, b.create<arith::MulIOp>(loc, bidx, A), D),
            b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, aidx, D),
                                    d))});

    auto vLoop = b.create<scf::ForOp>(loc, c0, V, c1, ValueRange{negInf, c0});
    {
      OpBuilder::InsertionGuard gv(b);
      b.setInsertionPointToStart(vLoop.getBody());
      Value v = vLoop.getInductionVar();
      Value bestVal = vLoop.getRegionIterArgs()[0];
      Value bestIdx = vLoop.getRegionIterArgs()[1];
      Value logitAcc;
      {
        auto lp = b.create<scf::ForOp>(loc, c0, H, c1, ValueRange{f0});
        OpBuilder::InsertionGuard gl(b);
        b.setInsertionPointToStart(lp.getBody());
        Value h = lp.getInductionVar();
        Value acc = lp.getRegionIterArgs()[0];
        Value st = b.create<memref::LoadOp>(
            loc, hidden, ValueRange{flat4(bidx, aidx, d, h, A, D, H)});
        Value w = b.create<memref::LoadOp>(loc, outProj, ValueRange{flat2(h, v, V)});
        Value prod = b.create<arith::MulFOp>(loc, st, w);
        Value next = b.create<arith::AddFOp>(loc, acc, prod);
        b.create<scf::YieldOp>(loc, ValueRange{next});
        logitAcc = lp.getResult(0);
      }
      b.create<memref::StoreOp>(
          loc, logitAcc, logits, ValueRange{flat4(bidx, aidx, d, v, A, D, V)});
      Value better = b.create<arith::CmpFOp>(loc, ogt, logitAcc, bestVal);
      auto ifop = b.create<scf::IfOp>(loc, TypeRange{f32, b.getIndexType()},
                                      better, /*withElse=*/true);
      {
        OpBuilder::InsertionGuard gi(b);
        b.setInsertionPointToStart(ifop.thenBlock());
        b.create<scf::YieldOp>(loc, ValueRange{logitAcc, v});
        b.setInsertionPointToStart(ifop.elseBlock());
        b.create<scf::YieldOp>(loc, ValueRange{bestVal, bestIdx});
      }
      b.create<scf::YieldOp>(loc, ifop.getResults());
    }
    Value tok = vLoop.getResult(1);
    Value tok64 = b.create<arith::IndexCastOp>(loc, i64, tok);
    Value tokIdx = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(
                 loc, b.create<arith::MulIOp>(loc, bidx, A), D),
        b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, aidx, D), d));
    b.create<memref::StoreOp>(loc, tok64, tokens, ValueRange{tokIdx});
    b.create<scf::YieldOp>(loc, ValueRange{tok});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMDSparkDraftBlockKernelPass
    : PassWrapper<GenerateROCMDSparkDraftBlockKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMDSparkDraftBlockKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-dspark-draft-block-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.dspark_draft_block directive into a fused "
           "DSpark draft-block gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.dspark_draft_block")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.dspark_draft_block missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      Type i64 = b.getI64Type();
      auto fmem = MemRefType::get({ShapedType::kDynamic}, b.getF32Type());
      auto imem = MemRefType::get({ShapedType::kDynamic}, i64);
      auto fnTy = b.getFunctionType(
          {fmem, imem, imem, fmem, fmem, fmem, fmem, fmem, fmem,
           fmem, fmem, imem, fmem,
           idxTy, idxTy, idxTy, idxTy, idxTy, idxTy, idxTy},
          {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitDSparkBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMDSparkDraftBlockKernelPass() {
  return std::make_unique<GenerateROCMDSparkDraftBlockKernelPass>();
}
