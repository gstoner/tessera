//===- GenerateROCMSpecAcceptSampleKernel.cpp — Leviathan spec accept -----===//
//
// SD1-2 of CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN (ROCm-led, gfx1151). The
// distribution-preserving (Leviathan) rejection-sampling form of spec_accept for
// a LINEAR draft chain, run as ONE device kernel. Per drafted position i the
// draft token d_i is accepted iff `accept_u[i] * p_draft <= p_target` (the
// division-free form of `accept_u[i] <= min(1, p_target/p_draft)`, p_draft > 0);
// on the first rejection a corrected token is drawn from the residual
// normalize(relu(p_target[i] - p_draft[i])); a fully-accepted chain draws a bonus
// from target_probs's extra row. The single categorical draw is CDF inversion of
// the explicit `resid_u` uniform.
//
// RNG is explicit (CF0 contract): accept_u (one per position) + resid_u (one) are
// operands, so the kernel is a deterministic, device-bit-exact function of its
// inputs. The verify is inherently serial (accept left-to-right, stop at first
// reject), so ONE thread runs it.
//
//   gpu.func @spec_accept_sample(%DRAFT,%TP,%DP,%AU,%RU,%OUT) kernel {
//     if tid==0 {
//       zero OUT
//       (accepted, done) = scf.for i=0..D iter(acc=0, done=false) {
//         if !done {
//           tok=DRAFT[i] ; pd=DP[i*V+tok] ; pt=TP[i*V+tok]
//           accept = pd>0 && AU[i]*pd <= pt
//           if accept { OUT[1+i]=tok ; acc++ }
//           else { OUT[1+i]=categorical(RU, residual@i) ; done=true }
//         }
//       }
//       if !done { OUT[1+D]=categorical(RU, target_row@D) }   // bonus
//       OUT[0]=accepted
//     }
//   }
//
// categorical(u, w): smallest k with cumsum(w)[k] > u·Σw ; argmax(fallback) if
// Σw<=0. Single-tile (one workgroup; D, V small). The tree (multi-path)
// rejection form is a later follow-up.

#include "TesseraROCM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// draft D i32 ; target_probs (D+1)×V f32 ; draft_probs D×V f32 ; accept_u D f32 ;
// resid_u 1 f32 ; result (D+2) i32. (The op verifier already guarantees this.)
static bool validateSpecAcceptSample(Operation *op, int64_t &D, int64_t &V) {
  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return false;
  auto draftT = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto tpT = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto dpT = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto auT = dyn_cast<RankedTensorType>(op->getOperand(3).getType());
  auto ruT = dyn_cast<RankedTensorType>(op->getOperand(4).getType());
  auto resT = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!draftT || !tpT || !dpT || !auT || !ruT || !resT)
    return false;
  if (draftT.getRank() != 1 || !draftT.getElementType().isInteger(32))
    return false;
  D = draftT.getDimSize(0);
  if (D <= 0 || dpT.getRank() != 2 || !dpT.getElementType().isF32() ||
      dpT.getDimSize(0) != D)
    return false;
  V = dpT.getDimSize(1);
  if (V <= 0 || tpT.getRank() != 2 || !tpT.getElementType().isF32() ||
      tpT.getDimSize(0) != D + 1 || tpT.getDimSize(1) != V ||
      auT.getRank() != 1 || !auT.getElementType().isF32() ||
      auT.getDimSize(0) != D || ruT.getRank() != 1 ||
      !ruT.getElementType().isF32() || ruT.getDimSize(0) != 1 ||
      resT.getRank() != 1 || !resT.getElementType().isInteger(32) ||
      resT.getDimSize(0) != D + 2)
    return false;
  return true;
}

struct GenerateROCMSpecAcceptSampleKernelPass
    : public PassWrapper<GenerateROCMSpecAcceptSampleKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMSpecAcceptSampleKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-spec-accept-sample-kernel";
  }
  StringRef getDescription() const final {
    return "SD1-2: lower a tessera.spec_accept_sample (Leviathan rejection "
           "sampling with explicit uniforms + CDF-inversion categorical) to one "
           "single-thread gpu.func for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void emitKernel(Operation *op, int64_t D, int64_t V, ModuleOp module,
                  unsigned idx) {
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_spec_accept_sample_" + Twine(idx)).str();

    Type i32 = b.getI32Type(), f32 = b.getF32Type(), idxTy = b.getIndexType();
    Type i1 = b.getI1Type();
    auto iMem = MemRefType::get({ShapedType::kDynamic}, i32);
    auto fMem = MemRefType::get({ShapedType::kDynamic}, f32);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    // (DRAFT i32, TARGET_PROBS f32, DRAFT_PROBS f32, ACCEPT_U f32, RESID_U f32,
    //  OUT i32)
    auto fnTy = b.getFunctionType({iMem, fMem, fMem, fMem, fMem, iMem}, {});
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value DRAFT = f.getArgument(0), TP = f.getArgument(1), DP = f.getArgument(2),
          AU = f.getArgument(3), RU = f.getArgument(4), OUT = f.getArgument(5);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    auto cf = [&](float v) {
      return arith::ConstantOp::create(kb, loc, f32, kb.getF32FloatAttr(v));
    };
    auto cI = [&](int32_t v) {
      return arith::ConstantOp::create(kb, loc, i32, kb.getI32IntegerAttr(v));
    };
    Value c0 = ci(0), c1 = ci(1), cV = ci(V), cD = ci(D);
    Value zerof = cf(0.0f), zeroI = cI(0);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value isT0 = arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::eq, tid, c0);

    // categorical sample by CDF inversion of uniform u over weight w(v) ≥ 0:
    // smallest k with cumsum(w)[k] > u·Σw ; argmax(fb(v)) if Σw ≤ 0. Returns i32.
    auto categorical = [&](Value u, llvm::function_ref<Value(Value)> w,
                           llvm::function_ref<Value(Value)> fb) -> Value {
      auto sumL = scf::ForOp::create(kb, loc, c0, cV, c1, ValueRange{zerof});
      {
        OpBuilder::InsertionGuard ig(kb);
        kb.setInsertionPointToStart(sumL.getBody());
        Value v = sumL.getInductionVar(), acc = sumL.getRegionIterArg(0);
        scf::YieldOp::create(kb, loc,
                             ValueRange{arith::AddFOp::create(kb, loc, acc, w(v))});
      }
      Value s = sumL.getResult(0);
      Value sPos =
          arith::CmpFOp::create(kb, loc, arith::CmpFPredicate::OGT, s, zerof);
      auto pick = scf::IfOp::create(kb, loc, TypeRange{i32}, sPos,
                                    /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard ig(kb);
        kb.setInsertionPointToStart(pick.thenBlock());
        Value tgt = arith::MulFOp::create(kb, loc, u, s);
        auto fF = arith::ConstantOp::create(kb, loc, i1, kb.getBoolAttr(false));
        auto scan = scf::ForOp::create(kb, loc, c0, cV, c1,
                                       ValueRange{zerof, zeroI, fF});
        {
          OpBuilder::InsertionGuard ig2(kb);
          kb.setInsertionPointToStart(scan.getBody());
          Value v = scan.getInductionVar();
          Value cum = scan.getRegionIterArg(0), k = scan.getRegionIterArg(1),
                found = scan.getRegionIterArg(2);
          Value cum2 = arith::AddFOp::create(kb, loc, cum, w(v));
          Value over = arith::CmpFOp::create(kb, loc, arith::CmpFPredicate::OGT,
                                             cum2, tgt);
          Value notFound = arith::XOrIOp::create(
              kb, loc, found,
              arith::ConstantOp::create(kb, loc, i1, kb.getBoolAttr(true)));
          Value hit = arith::AndIOp::create(kb, loc, notFound, over);
          Value vI = arith::IndexCastOp::create(kb, loc, i32, v);
          Value k2 = arith::SelectOp::create(kb, loc, hit, vI, k);
          Value f2 = arith::OrIOp::create(kb, loc, found, hit);
          scf::YieldOp::create(kb, loc, ValueRange{cum2, k2, f2});
        }
        scf::YieldOp::create(kb, loc, ValueRange{scan.getResult(1)});
      }
      {
        OpBuilder::InsertionGuard ig(kb);
        kb.setInsertionPointToStart(pick.elseBlock());
        // argmax over fb(v).
        Value negInf = cf(-3.0e38f);
        auto am = scf::ForOp::create(kb, loc, c0, cV, c1,
                                     ValueRange{negInf, zeroI});
        {
          OpBuilder::InsertionGuard ig2(kb);
          kb.setInsertionPointToStart(am.getBody());
          Value v = am.getInductionVar();
          Value best = am.getRegionIterArg(0), k = am.getRegionIterArg(1);
          Value wv = fb(v);
          Value gt = arith::CmpFOp::create(kb, loc, arith::CmpFPredicate::OGT, wv,
                                           best);
          Value vI = arith::IndexCastOp::create(kb, loc, i32, v);
          Value k2 = arith::SelectOp::create(kb, loc, gt, vI, k);
          Value b2 = arith::SelectOp::create(kb, loc, gt, wv, best);
          scf::YieldOp::create(kb, loc, ValueRange{b2, k2});
        }
        scf::YieldOp::create(kb, loc, ValueRange{am.getResult(1)});
      }
      return pick.getResult(0);
    };

    auto g = scf::IfOp::create(kb, loc, isT0, /*withElse=*/false);
    OpBuilder::InsertionGuard ig(kb);
    kb.setInsertionPointToStart(g.thenBlock());

    // zero OUT[0 .. D+1].
    {
      auto z = scf::ForOp::create(kb, loc, c0, ci(D + 2), c1);
      OpBuilder::InsertionGuard ig2(kb);
      kb.setInsertionPointToStart(z.getBody());
      memref::StoreOp::create(kb, loc, zeroI, OUT,
                              ValueRange{z.getInductionVar()});
    }

    Value ru = memref::LoadOp::create(kb, loc, RU, ValueRange{c0});

    // accept loop: (accepted, done).
    auto fF = arith::ConstantOp::create(kb, loc, i1, kb.getBoolAttr(false));
    auto loop = scf::ForOp::create(kb, loc, c0, cD, c1, ValueRange{zeroI, fF});
    {
      OpBuilder::InsertionGuard ig2(kb);
      kb.setInsertionPointToStart(loop.getBody());
      Value i = loop.getInductionVar();
      Value accepted = loop.getRegionIterArg(0), done = loop.getRegionIterArg(1);
      Value notDone = arith::XOrIOp::create(
          kb, loc, done,
          arith::ConstantOp::create(kb, loc, i1, kb.getBoolAttr(true)));
      auto step = scf::IfOp::create(kb, loc, TypeRange{i32, i1}, notDone,
                                    /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard ig3(kb);
        kb.setInsertionPointToStart(step.thenBlock());
        Value rowDP = arith::MulIOp::create(kb, loc, i, cV);  // i*V
        Value rowTP = rowDP;                                   // tp row i = i*V
        Value tok = memref::LoadOp::create(kb, loc, DRAFT, ValueRange{i});
        Value tokIdx = arith::IndexCastOp::create(kb, loc, idxTy, tok);
        Value pd = memref::LoadOp::create(
            kb, loc, DP, ValueRange{arith::AddIOp::create(kb, loc, rowDP, tokIdx)});
        Value pt = memref::LoadOp::create(
            kb, loc, TP, ValueRange{arith::AddIOp::create(kb, loc, rowTP, tokIdx)});
        Value au = memref::LoadOp::create(kb, loc, AU, ValueRange{i});
        // accept = pd>0 && au*pd <= pt  (== au <= min(1, pt/pd)).
        Value pdPos =
            arith::CmpFOp::create(kb, loc, arith::CmpFPredicate::OGT, pd, zerof);
        Value aupd = arith::MulFOp::create(kb, loc, au, pd);
        Value le =
            arith::CmpFOp::create(kb, loc, arith::CmpFPredicate::OLE, aupd, pt);
        Value accept = arith::AndIOp::create(kb, loc, pdPos, le);
        Value out1i = arith::AddIOp::create(kb, loc, c1, i);  // OUT[1+i]
        auto br = scf::IfOp::create(kb, loc, TypeRange{i32, i1}, accept,
                                    /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard ig4(kb);
          kb.setInsertionPointToStart(br.thenBlock());
          memref::StoreOp::create(kb, loc, tok, OUT, ValueRange{out1i});
          Value acc2 = arith::AddIOp::create(kb, loc, accepted, cI(1));
          scf::YieldOp::create(kb, loc, ValueRange{acc2, done});
        }
        {
          OpBuilder::InsertionGuard ig4(kb);
          kb.setInsertionPointToStart(br.elseBlock());
          // corrected = categorical(ru, residual@i): w=relu(tp-dp), fb=tp[i].
          Value corrected = categorical(
              ru,
              [&](Value v) -> Value {
                Value tpv = memref::LoadOp::create(
                    kb, loc, TP, ValueRange{arith::AddIOp::create(kb, loc, rowTP, v)});
                Value dpv = memref::LoadOp::create(
                    kb, loc, DP, ValueRange{arith::AddIOp::create(kb, loc, rowDP, v)});
                Value d = arith::SubFOp::create(kb, loc, tpv, dpv);
                return arith::MaximumFOp::create(kb, loc, d, zerof);
              },
              [&](Value v) -> Value {
                return memref::LoadOp::create(
                    kb, loc, TP, ValueRange{arith::AddIOp::create(kb, loc, rowTP, v)});
              });
          memref::StoreOp::create(kb, loc, corrected, OUT, ValueRange{out1i});
          Value trueV =
              arith::ConstantOp::create(kb, loc, i1, kb.getBoolAttr(true));
          scf::YieldOp::create(kb, loc, ValueRange{accepted, trueV});
        }
        scf::YieldOp::create(kb, loc,
                             ValueRange{br.getResult(0), br.getResult(1)});
      }
      {
        OpBuilder::InsertionGuard ig3(kb);
        kb.setInsertionPointToStart(step.elseBlock());
        scf::YieldOp::create(kb, loc, ValueRange{accepted, done});
      }
      scf::YieldOp::create(kb, loc,
                           ValueRange{step.getResult(0), step.getResult(1)});
    }
    Value accepted = loop.getResult(0), done = loop.getResult(1);

    // full accept → bonus from target row D ; OUT[1+D].
    Value notDone = arith::XOrIOp::create(
        kb, loc, done,
        arith::ConstantOp::create(kb, loc, i1, kb.getBoolAttr(true)));
    {
      auto fa = scf::IfOp::create(kb, loc, notDone, /*withElse=*/false);
      OpBuilder::InsertionGuard ig2(kb);
      kb.setInsertionPointToStart(fa.thenBlock());
      Value rowD = arith::MulIOp::create(kb, loc, cD, cV);  // D*V
      Value bonus = categorical(
          ru,
          [&](Value v) -> Value {
            return memref::LoadOp::create(
                kb, loc, TP, ValueRange{arith::AddIOp::create(kb, loc, rowD, v)});
          },
          [&](Value v) -> Value {
            return memref::LoadOp::create(
                kb, loc, TP, ValueRange{arith::AddIOp::create(kb, loc, rowD, v)});
          });
      memref::StoreOp::create(kb, loc, bonus, OUT,
                              ValueRange{arith::AddIOp::create(kb, loc, c1, cD)});
    }
    memref::StoreOp::create(kb, loc, accepted, OUT, ValueRange{c0});

    kb.setInsertionPointToEnd(&f.getBody().front());
    gpu::ReturnOp::create(kb, loc);
    op->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> ops;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.spec_accept_sample")
        ops.push_back(op);
    });
    unsigned idx = 0;
    for (Operation *op : ops) {
      int64_t D = 0, V = 0;
      if (validateSpecAcceptSample(op, D, V))
        emitKernel(op, D, V, module, idx++);
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMSpecAcceptSampleKernelPass() {
  return std::make_unique<GenerateROCMSpecAcceptSampleKernelPass>();
}
