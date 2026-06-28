//===- GenerateROCMMoeKernel.cpp - mixture-of-experts compute gpu kernel --===//
//
// Expands `tessera_rocm.moe` into the MoE compute part — the routed per-token
// expert matmuls (top-1), one thread per (token, out-column):
//
//   out[t, o] = Σ_i x[t, i] · experts[route[t], i, o]
//
// experts is [num_experts, in_dim, out_dim]; route[t] (i32) is the resolved
// top-1 expert (routing — argmax/round-robin — is done on the host). This runs
// the FLOP-heavy expert GEMVs. All f32 (route i32). CPU analog: avx512_moe_f32.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitMoeBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  Type idx = b.getIndexType();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), EXP = f.getArgument(1), ROUTE = f.getArgument(2);
  Value OUT = f.getArgument(3);
  Value tokens = f.getArgument(4), IN = f.getArgument(5), OUTD = f.getArgument(6);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value total = b.create<arith::MulIOp>(loc, tokens, OUTD);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value t = b.create<arith::DivUIOp>(loc, gid, OUTD);
  Value o = b.create<arith::RemUIOp>(loc, gid, OUTD);
  // e = route[t]  (i32 → index)
  Value e = b.create<arith::IndexCastOp>(
      loc, idx, b.create<memref::LoadOp>(loc, ROUTE, ValueRange{t}));
  Value xbase = b.create<arith::MulIOp>(loc, t, IN);              // t*in_dim
  Value ebase = b.create<arith::MulIOp>(loc, e,
      b.create<arith::MulIOp>(loc, IN, OUTD));                    // e*in*out
  // acc = Σ_i x[t,i] · experts[e, i, o]
  auto kl = b.create<scf::ForOp>(loc, c0, IN, c1, ValueRange{zero});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(kl.getBody());
    Value i = kl.getInductionVar();
    Value xv = b.create<memref::LoadOp>(
        loc, X, ValueRange{b.create<arith::AddIOp>(loc, xbase, i)});
    // experts[ebase + i*out_dim + o]
    Value woff = b.create<arith::AddIOp>(
        loc, ebase, b.create<arith::AddIOp>(
                        loc, b.create<arith::MulIOp>(loc, i, OUTD), o));
    Value wv = b.create<memref::LoadOp>(loc, EXP, ValueRange{woff});
    Value acc = b.create<arith::AddFOp>(loc, kl.getRegionIterArgs()[0],
                                        b.create<arith::MulFOp>(loc, xv, wv));
    b.create<scf::YieldOp>(loc, ValueRange{acc});
  }
  b.create<memref::StoreOp>(loc, kl.getResult(0), OUT, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMMoeKernelPass
    : PassWrapper<GenerateROCMMoeKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMMoeKernelPass)

  StringRef getArgument() const final { return "generate-rocm-moe-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.moe directive into the MoE compute kernel "
           "(routed per-token expert GEMVs, one thread per (token, out-column))";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.moe")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.moe missing name");
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
      auto memI32 = MemRefType::get({ShapedType::kDynamic}, i32);
      auto fnTy = b.getFunctionType(
          {memF32, memF32, memI32, memF32, idxTy, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitMoeBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMMoeKernelPass() {
  return std::make_unique<GenerateROCMMoeKernelPass>();
}
