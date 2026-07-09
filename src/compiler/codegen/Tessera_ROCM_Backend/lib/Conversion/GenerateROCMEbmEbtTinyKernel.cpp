//===- GenerateROCMEbmEbtTinyKernel.cpp - EBT-tiny fused pipeline --------===//
//
// Expands a `tessera_rocm.ebm_ebt_tiny` directive into a fused
// energy-based-transformer inference kernel. For B batches of K candidate
// trajectories (y0, grad row-major [B*K, D], K ≤ 256) it fuses, per batch b:
//
//   refinement:  y_T[k,:] = y0[k,:] - (T·eta)·grad[k,:]   (closed form)
//   energy:      e[k]     = Σ_d y_T[k,d]^2                 (squared norm)
//   argmin:      k*       = argmin_k e[k]                  (first-min tie-break)
//   gather:      out[b,:] = y_T[k*,:]
//
// ONE workgroup per batch (grid = B), blockDim 256. Thread k (k < K) computes
// candidate k's closed-form energy over D elements; lanes k ≥ K seed +inf. A
// shared-memory tree argmin over the 256 lanes — tie-broken toward the lower
// index to match numpy argmin — picks k*; then the lanes stride over D to write
// y_T[k*,:]. Matches `tessera.ebm.ebt_tiny`. CPU analog:
// tessera_x86_ebm_ebt_tiny_f32. Args: (y0 : memref<?xf32>, grad : memref<?xf32>,
// eta : f32, T : i32, B : index, K : index, D : index, out : memref<?xf32>).
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

using namespace mlir;

namespace {

static constexpr int64_t BD = 256; // blockDim == max candidates K

void emitEbtTinyBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  Type i32 = b.getI32Type();

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  Value shE = f.addWorkgroupAttribution(
      MemRefType::get({BD}, f32, MemRefLayoutAttrInterface(), ws), loc);
  Value shI = f.addWorkgroupAttribution(
      MemRefType::get({BD}, i32, MemRefLayoutAttrInterface(), ws), loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value Y0 = f.getArgument(0), G = f.getArgument(1);
  Value eta = f.getArgument(2), Ti = f.getArgument(3);
  Value B = f.getArgument(4), K = f.getArgument(5), D = f.getArgument(6);
  Value OUT = f.getArgument(7);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto slt = arith::CmpIPredicate::slt;
  Value c0 = ci(0), cBD = ci(BD);
  Value bigE =
      b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(3.4028235e38f));
  Value zerof = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value halff = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.5f));
  (void)halff;
  Value maxI = b.create<arith::ConstantIntOp>(loc, i32, 2147483647);

  // scale = eta · T  (the T inner refinement steps collapse to one affine).
  Value tF = b.create<arith::SIToFPOp>(loc, f32, Ti);
  Value scale = b.create<arith::MulFOp>(loc, eta, tF);

  // One workgroup per batch: b = blockIdx.x. Guard b < B.
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bInb = b.create<arith::CmpIOp>(loc, slt, bid, B);
  auto bGuard = b.create<scf::IfOp>(loc, bInb, /*withElse=*/false);
  b.setInsertionPointToStart(bGuard.thenBlock());

  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value tidI = b.create<arith::IndexCastOp>(loc, i32, tid);
  Value rowBase = b.create<arith::MulIOp>(
      loc, b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, K),
                                   tid),
      D); // (b*K + tid) * D

  // Phase 1 — candidate energy for lanes tid < K; +inf / MAXINT otherwise.
  Value tidLtK = b.create<arith::CmpIOp>(loc, slt, tid, K);
  auto eGuard = b.create<scf::IfOp>(loc, tidLtK, /*withElse=*/true);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(eGuard.thenBlock());
    auto lp = b.create<scf::ForOp>(loc, c0, D, ci(1), ValueRange{zerof});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(lp.getBody());
      Value d = lp.getInductionVar();
      Value acc = lp.getRegionIterArgs()[0];
      Value idx = b.create<arith::AddIOp>(loc, rowBase, d);
      Value yv = b.create<memref::LoadOp>(loc, Y0, ValueRange{idx});
      Value gv = b.create<memref::LoadOp>(loc, G, ValueRange{idx});
      Value v =
          b.create<arith::SubFOp>(loc, yv, b.create<arith::MulFOp>(loc, scale, gv));
      Value sq = b.create<arith::MulFOp>(loc, v, v);
      b.create<scf::YieldOp>(loc,
                             ValueRange{b.create<arith::AddFOp>(loc, acc, sq)});
    }
    b.create<memref::StoreOp>(loc, lp.getResult(0), shE, ValueRange{tid});
    b.create<memref::StoreOp>(loc, tidI, shI, ValueRange{tid});
    b.setInsertionPointToStart(eGuard.elseBlock());
    b.create<memref::StoreOp>(loc, bigE, shE, ValueRange{tid});
    b.create<memref::StoreOp>(loc, maxI, shI, ValueRange{tid});
  }
  b.create<gpu::BarrierOp>(loc);

  // Phase 2 — tree argmin over the 256 lanes, first-min tie-break:
  //   replace when e_other < e_cur, or (e_other == e_cur and idx_other < idx_cur).
  for (int64_t stride = BD / 2; stride > 0; stride >>= 1) {
    Value strideC = ci(stride);
    Value active = b.create<arith::CmpIOp>(loc, slt, tid, strideC);
    auto sGuard = b.create<scf::IfOp>(loc, active, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(sGuard.thenBlock());
      Value other = b.create<arith::AddIOp>(loc, tid, strideC);
      Value eCur = b.create<memref::LoadOp>(loc, shE, ValueRange{tid});
      Value eOth = b.create<memref::LoadOp>(loc, shE, ValueRange{other});
      Value iCur = b.create<memref::LoadOp>(loc, shI, ValueRange{tid});
      Value iOth = b.create<memref::LoadOp>(loc, shI, ValueRange{other});
      Value lt = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, eOth,
                                         eCur);
      Value eq = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, eOth,
                                         eCur);
      Value iLt =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iOth, iCur);
      Value take = b.create<arith::OrIOp>(
          loc, lt, b.create<arith::AndIOp>(loc, eq, iLt));
      b.create<memref::StoreOp>(
          loc, b.create<arith::SelectOp>(loc, take, eOth, eCur), shE,
          ValueRange{tid});
      b.create<memref::StoreOp>(
          loc, b.create<arith::SelectOp>(loc, take, iOth, iCur), shI,
          ValueRange{tid});
    }
    b.create<gpu::BarrierOp>(loc);
  }

  // Phase 3 — gather: out[b, :] = y_T[k*, :], lanes stride over D.
  Value kstarI = b.create<memref::LoadOp>(loc, shI, ValueRange{c0});
  Value kstar = b.create<arith::IndexCastOp>(loc, b.getIndexType(), kstarI);
  Value srcBase = b.create<arith::MulIOp>(
      loc, b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, K),
                                   kstar),
      D);
  Value dstBase = b.create<arith::MulIOp>(loc, bid, D);
  {
    auto lp = b.create<scf::ForOp>(loc, tid, D, cBD);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value d = lp.getInductionVar();
    Value si = b.create<arith::AddIOp>(loc, srcBase, d);
    Value yv = b.create<memref::LoadOp>(loc, Y0, ValueRange{si});
    Value gv = b.create<memref::LoadOp>(loc, G, ValueRange{si});
    Value v =
        b.create<arith::SubFOp>(loc, yv, b.create<arith::MulFOp>(loc, scale, gv));
    Value di = b.create<arith::AddIOp>(loc, dstBase, d);
    b.create<memref::StoreOp>(loc, v, OUT, ValueRange{di});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMEbmEbtTinyKernelPass
    : PassWrapper<GenerateROCMEbmEbtTinyKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMEbmEbtTinyKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-ebm-ebt-tiny-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.ebm_ebt_tiny directive into a fused "
           "refine→energy→argmin→gather EBT-tiny gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.ebm_ebt_tiny")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.ebm_ebt_tiny missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type i32 = b.getI32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      // (y0, grad : memref<?xf32>, eta : f32, T : i32, B, K, D : index,
      //  out : memref<?xf32>)
      auto fnTy = b.getFunctionType(
          {memF32, memF32, f32, i32, idxTy, idxTy, idxTy, memF32}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitEbtTinyBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMEbmEbtTinyKernelPass() {
  return std::make_unique<GenerateROCMEbmEbtTinyKernelPass>();
}
