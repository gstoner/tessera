//===- GenerateROCMMLAAbsorbDecodeKernel.cpp - absorbed MLA decode kernel -===//
//
// Expands a `tessera_rocm.mla_absorb_decode` directive into the DK1 native
// ROCm proof kernel for Multi-Latent Attention decode:
//
//   scores[s,h,t] = (q_nope[s,h] @ Wuk[:,h,:]^T) · c[t]
//                 + q_rope[s,h] · k_rope[t]
//   O[s,h,dv]     = softmax_causal(scores)[t] · (c[t] @ Wuv[:,h,dv])
//
// Q projection, RoPE on q_rope, scaling, and cache append are handled by the
// runtime wrapper to match stdlib.attention.mla_decode_step exactly. This
// kernel proves the ROCm absorbed-latent decode slot without materializing the
// per-head K/V cache; later tuning can replace the scalar-per-output kernel.
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

void emitMLAAbsorbDecodeBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  auto sle = arith::CmpIPredicate::sle;

  b.setInsertionPointToStart(&f.getBody().front());
  Value Qn = f.getArgument(0);
  Value Qr = f.getArgument(1);
  Value C = f.getArgument(2);
  Value Kr = f.getArgument(3);
  Value Wuk = f.getArgument(4);
  Value Wuv = f.getArgument(5);
  Value O = f.getArgument(6);
  Value Sq = f.getArgument(7);
  Value T = f.getArgument(8);
  Value Hh = f.getArgument(9);
  Value Dn = f.getArgument(10);
  Value Dr = f.getArgument(11);
  Value Dc = f.getArgument(12);
  Value Dv = f.getArgument(13);
  Value KvStart = f.getArgument(14);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), cBD = ci(BD);
  Value f0 = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value negInf = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, cBD), tid);
  Value total = b.create<arith::MulIOp>(
      loc, b.create<arith::MulIOp>(loc, Sq, Hh), Dv);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto rowIf = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());

  auto flat2 = [&](Value i, Value j, Value J) -> Value {
    return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, i, J), j);
  };
  auto flat3 = [&](Value i, Value j, Value k, Value J, Value K) -> Value {
    Value ij = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, i, J), j);
    return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, ij, K), k);
  };

  Value dv = b.create<arith::RemUIOp>(loc, gid, Dv);
  Value sh = b.create<arith::DivUIOp>(loc, gid, Dv);
  Value h = b.create<arith::RemUIOp>(loc, sh, Hh);
  Value s = b.create<arith::DivUIOp>(loc, sh, Hh);
  Value qpos = b.create<arith::AddIOp>(loc, KvStart, s);

  auto scoreAt = [&](Value t) -> Value {
    Value qAbsDotC = f0;
    auto dcLoop = b.create<scf::ForOp>(loc, c0, Dc, c1, ValueRange{qAbsDotC});
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(dcLoop.getBody());
      Value dc = dcLoop.getInductionVar();
      Value accDc = dcLoop.getRegionIterArgs()[0];
      Value qabs = f0;
      auto dnLoop = b.create<scf::ForOp>(loc, c0, Dn, c1, ValueRange{qabs});
      {
        OpBuilder::InsertionGuard g2(b);
        b.setInsertionPointToStart(dnLoop.getBody());
        Value dn = dnLoop.getInductionVar();
        Value accDn = dnLoop.getRegionIterArgs()[0];
        Value q = b.create<memref::LoadOp>(loc, Qn, ValueRange{flat3(s, h, dn, Hh, Dn)});
        Value w = b.create<memref::LoadOp>(loc, Wuk, ValueRange{flat3(dc, h, dn, Hh, Dn)});
        b.create<scf::YieldOp>(loc, ValueRange{
            b.create<arith::AddFOp>(loc, accDn, b.create<arith::MulFOp>(loc, q, w))});
      }
      Value c = b.create<memref::LoadOp>(loc, C, ValueRange{flat2(t, dc, Dc)});
      Value prod = b.create<arith::MulFOp>(loc, dnLoop.getResult(0), c);
      b.create<scf::YieldOp>(
          loc, ValueRange{b.create<arith::AddFOp>(loc, accDc, prod)});
    }

    Value rope = f0;
    auto drLoop = b.create<scf::ForOp>(loc, c0, Dr, c1, ValueRange{rope});
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(drLoop.getBody());
      Value dr = drLoop.getInductionVar();
      Value acc = drLoop.getRegionIterArgs()[0];
      Value q = b.create<memref::LoadOp>(loc, Qr, ValueRange{flat3(s, h, dr, Hh, Dr)});
      Value k = b.create<memref::LoadOp>(loc, Kr, ValueRange{flat2(t, dr, Dr)});
      b.create<scf::YieldOp>(
          loc, ValueRange{b.create<arith::AddFOp>(
              loc, acc, b.create<arith::MulFOp>(loc, q, k))});
    }
    return b.create<arith::AddFOp>(loc, dcLoop.getResult(0), drLoop.getResult(0));
  };

  auto maxLoop = b.create<scf::ForOp>(loc, c0, T, c1, ValueRange{negInf});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(maxLoop.getBody());
    Value t = maxLoop.getInductionVar();
    Value best = maxLoop.getRegionIterArgs()[0];
    Value valid = b.create<arith::CmpIOp>(loc, sle, t, qpos);
    auto ifop = b.create<scf::IfOp>(loc, f32, valid, /*withElse=*/true);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(ifop.thenBlock());
      Value score = scoreAt(t);
      Value isGt = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, score, best);
      b.create<scf::YieldOp>(loc, ValueRange{b.create<arith::SelectOp>(loc, isGt, score, best)});
      b.setInsertionPointToStart(ifop.elseBlock());
      b.create<scf::YieldOp>(loc, ValueRange{best});
    }
    b.create<scf::YieldOp>(loc, ifop.getResults());
  }
  Value maxScore = maxLoop.getResult(0);

  auto sumLoop = b.create<scf::ForOp>(loc, c0, T, c1, ValueRange{f0, f0});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(sumLoop.getBody());
    Value t = sumLoop.getInductionVar();
    Value denom = sumLoop.getRegionIterArgs()[0];
    Value numer = sumLoop.getRegionIterArgs()[1];
    Value valid = b.create<arith::CmpIOp>(loc, sle, t, qpos);
    auto ifop = b.create<scf::IfOp>(loc, TypeRange{f32, f32}, valid,
                                    /*withElse=*/true);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(ifop.thenBlock());
      Value score = scoreAt(t);
      Value e = b.create<math::ExpOp>(
          loc, b.create<arith::SubFOp>(loc, score, maxScore));
      Value vt = f0;
      auto dcLoop = b.create<scf::ForOp>(loc, c0, Dc, c1, ValueRange{vt});
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(dcLoop.getBody());
        Value dc = dcLoop.getInductionVar();
        Value acc = dcLoop.getRegionIterArgs()[0];
        Value c = b.create<memref::LoadOp>(loc, C, ValueRange{flat2(t, dc, Dc)});
        Value w = b.create<memref::LoadOp>(loc, Wuv, ValueRange{flat3(dc, h, dv, Hh, Dv)});
        b.create<scf::YieldOp>(
            loc, ValueRange{b.create<arith::AddFOp>(
                loc, acc, b.create<arith::MulFOp>(loc, c, w))});
      }
      b.create<scf::YieldOp>(
          loc, ValueRange{b.create<arith::AddFOp>(loc, denom, e),
                          b.create<arith::AddFOp>(
                              loc, numer, b.create<arith::MulFOp>(
                                               loc, e, dcLoop.getResult(0)))});
      b.setInsertionPointToStart(ifop.elseBlock());
      b.create<scf::YieldOp>(loc, ValueRange{denom, numer});
    }
    b.create<scf::YieldOp>(loc, ifop.getResults());
  }

  Value out = b.create<arith::DivFOp>(loc, sumLoop.getResult(1), sumLoop.getResult(0));
  b.create<memref::StoreOp>(loc, out, O, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMMLAAbsorbDecodeKernelPass
    : PassWrapper<GenerateROCMMLAAbsorbDecodeKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMMLAAbsorbDecodeKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-mla-absorb-decode-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.mla_absorb_decode directive into a native "
           "absorbed-latent MLA decode gpu kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.mla_absorb_decode")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.mla_absorb_decode missing name");
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
      auto fnTy = b.getFunctionType(
          {fmem, fmem, fmem, fmem, fmem, fmem, fmem, idxTy, idxTy, idxTy,
           idxTy, idxTy, idxTy, idxTy, idxTy},
          {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitMLAAbsorbDecodeBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMMLAAbsorbDecodeKernelPass() {
  return std::make_unique<GenerateROCMMLAAbsorbDecodeKernelPass>();
}
