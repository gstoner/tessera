//===- GenerateROCMSelectiveSsmKernel.cpp - Mamba2 selective scan kernel --===//
//
// Expands `tessera_rocm.selective_ssm` into a Mamba2 selective state-space scan,
// one thread per (b,d) channel — sequential over time, parallel over channels:
//
//   A_bar = exp(delta[b,t,d]·A[d,n]) ; B_bar = delta[b,t,d]·B[b,t,n]
//   h[b,d,n] = A_bar·h[b,d,n] + B_bar·x[b,t,d]
//   y[b,t,d] = Σ_n C[b,t,n]·h[b,d,n]
//
// Each thread owns the N-length state slice h[b,d,:] (a global scratch buffer,
// init to the caller's state or zeros), an outer scf.for over t and an inner
// scf.for over n (accumulating y, updating h in place). exp via math→ROCDL. All
// f32. CPU analog: avx512_ssm_f32. Validated vs numpy on gfx1151.
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

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitSsmBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy) {
  Type f32 = b.getF32Type();
  bool half = storeTy != f32;
  // load an element of a storage-typed memref, extended to f32 for compute.
  auto loadF32 = [&](Value mem, ValueRange idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, mem, idx);
    return half ? b.create<arith::ExtFOp>(loc, f32, v).getResult() : v;
  };
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), A = f.getArgument(1), B = f.getArgument(2);
  Value C = f.getArgument(3), DELTA = f.getArgument(4), H = f.getArgument(5);
  Value Y = f.getArgument(6);
  Value Bsz = f.getArgument(7), S = f.getArgument(8), D = f.getArgument(9);
  Value N = f.getArgument(10);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value total = b.create<arith::MulIOp>(loc, Bsz, D);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value bb = b.create<arith::DivUIOp>(loc, gid, D);   // batch
  Value dd = b.create<arith::RemUIOp>(loc, gid, D);   // channel
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value hbase = b.create<arith::MulIOp>(loc, b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bb, D), dd), N);  // (b*D+d)*N
  Value abase = b.create<arith::MulIOp>(loc, dd, N);      // d*N

  // outer loop over time t
  auto tloop = b.create<scf::ForOp>(loc, c0, S, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(tloop.getBody());
    Value t = tloop.getInductionVar();
    // row = b*S + t
    Value row = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bb, S),
                                        t);
    Value xoff = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, row, D),
                                         dd);            // (b*S+t)*D + d
    Value dt = loadF32(DELTA, ValueRange{xoff});
    Value xt = loadF32(X, ValueRange{xoff});
    Value bcbase = b.create<arith::MulIOp>(loc, row, N);  // (b*S+t)*N

    auto nloop = b.create<scf::ForOp>(loc, c0, N, c1, ValueRange{zero});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(nloop.getBody());
      Value n = nloop.getInductionVar();
      Value acc = nloop.getRegionIterArgs()[0];
      Value hoff = b.create<arith::AddIOp>(loc, hbase, n);
      Value aoff = b.create<arith::AddIOp>(loc, abase, n);
      Value bcoff = b.create<arith::AddIOp>(loc, bcbase, n);
      Value av = loadF32(A, ValueRange{aoff});
      Value bv = loadF32(B, ValueRange{bcoff});
      Value cv = loadF32(C, ValueRange{bcoff});
      Value hprev = b.create<memref::LoadOp>(loc, H, ValueRange{hoff});
      Value ab = b.create<math::ExpOp>(loc, b.create<arith::MulFOp>(loc, dt, av));
      Value bbar = b.create<arith::MulFOp>(loc, dt, bv);
      Value hv = b.create<arith::AddFOp>(
          loc, b.create<arith::MulFOp>(loc, ab, hprev),
          b.create<arith::MulFOp>(loc, bbar, xt));
      b.create<memref::StoreOp>(loc, hv, H, ValueRange{hoff});
      Value nacc = b.create<arith::AddFOp>(loc, acc,
                                           b.create<arith::MulFOp>(loc, cv, hv));
      b.create<scf::YieldOp>(loc, ValueRange{nacc});
    }
    Value yv = nloop.getResult(0);
    if (half)
      yv = b.create<arith::TruncFOp>(loc, storeTy, yv);
    b.create<memref::StoreOp>(loc, yv, Y, ValueRange{xoff});
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSelectiveSsmKernelPass
    : PassWrapper<GenerateROCMSelectiveSsmKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMSelectiveSsmKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-selective-ssm-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.selective_ssm directive into a Mamba2 "
           "selective state-space scan gpu kernel (one thread per (b,d) channel)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.selective_ssm")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.selective_ssm missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      // Storage dtype for x/A/B/C/delta + y; h + compute stay f32.
      Type storeTy = f32;
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16")
          storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16")
          storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-selective-ssm-kernel: dtype must be "
                        "f32/f16/bf16 (got '") << dt << "')";
          return signalPassFailure();
        }
      }
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto memSt = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // (x, A, B, C, delta : store), (h : f32 scratch), (y : store), 4 index.
      auto fnTy = b.getFunctionType(
          {memSt, memSt, memSt, memSt, memSt, memF32, memSt,
           idxTy, idxTy, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitSsmBody(body, loc, gpuFunc, storeTy);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSelectiveSsmKernelPass() {
  return std::make_unique<GenerateROCMSelectiveSsmKernelPass>();
}
