//===- GenerateROCMTriSolveKernel.cpp - batched triangular solve kernel ---===//
//
// Expands `tessera_rocm.tri_solve` into a batched triangular solve A·X = B,
// one thread per (matrix, RHS column) — A[b,n,n] (triangular part used),
// B/X[b,n,m]:
//
//   lower (forward): X[i] = (B[i] - Σ_{k<i} A[i,k]·X[k]) / A[i,i]
//   upper (back):    X[i] = (B[i] - Σ_{k>i} A[i,k]·X[k]) / A[i,i]   (i = n-1..0)
//
// `lower` (BoolAttr) selects the substitution direction at codegen (two cached
// kernels). One thread per (b,c) solves one RHS column. All f32. CPU analog:
// avx512_linalg_f32 tri_solve.
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

void emitTriBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, bool lower) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), B = f.getArgument(1), X = f.getArgument(2);
  Value batch = f.getArgument(3), N = f.getArgument(4), M = f.getArgument(5);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value total = b.create<arith::MulIOp>(loc, batch, M);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value bi = b.create<arith::DivUIOp>(loc, gid, M);   // matrix index
  Value ci = b.create<arith::RemUIOp>(loc, gid, M);   // RHS column index
  Value abase = b.create<arith::MulIOp>(loc, bi, b.create<arith::MulIOp>(loc, N, N));
  Value xbase = b.create<arith::MulIOp>(loc, bi, b.create<arith::MulIOp>(loc, N, M));
  auto aoff = [&](Value r, Value c) {
    return b.create<arith::AddIOp>(
        loc, abase, b.create<arith::AddIOp>(
                        loc, b.create<arith::MulIOp>(loc, r, N), c)).getResult();
  };
  auto xoff = [&](Value r) {
    return b.create<arith::AddIOp>(
        loc, xbase, b.create<arith::AddIOp>(
                        loc, b.create<arith::MulIOp>(loc, r, M), ci)).getResult();
  };

  auto solveRow = [&](Value i) {
    // acc = Σ_{k in range} A[i,k]·X[k]
    Value lo = lower ? c0 : b.create<arith::AddIOp>(loc, i, c1).getResult();
    Value hi = lower ? i : N;
    auto kl = b.create<scf::ForOp>(loc, lo, hi, c1, ValueRange{zero});
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(kl.getBody());
      Value k = kl.getInductionVar();
      Value aik = b.create<memref::LoadOp>(loc, A, ValueRange{aoff(i, k)});
      Value xk = b.create<memref::LoadOp>(loc, X, ValueRange{xoff(k)});
      Value acc = b.create<arith::AddFOp>(
          loc, kl.getRegionIterArgs()[0],
          b.create<arith::MulFOp>(loc, aik, xk));
      b.create<scf::YieldOp>(loc, ValueRange{acc});
    }
    Value bv = b.create<memref::LoadOp>(loc, B, ValueRange{xoff(i)});
    Value diag = b.create<memref::LoadOp>(loc, A, ValueRange{aoff(i, i)});
    Value xi = b.create<arith::DivFOp>(
        loc, b.create<arith::SubFOp>(loc, bv, kl.getResult(0)), diag);
    b.create<memref::StoreOp>(loc, xi, X, ValueRange{xoff(i)});
  };

  auto rl = b.create<scf::ForOp>(loc, c0, N, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(rl.getBody());
    Value it = rl.getInductionVar();
    // forward: i = it ; back: i = n-1-it
    Value i = lower ? it
                    : b.create<arith::SubIOp>(
                          loc, b.create<arith::SubIOp>(loc, N, c1), it)
                          .getResult();
    solveRow(i);
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMTriSolveKernelPass
    : PassWrapper<GenerateROCMTriSolveKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMTriSolveKernelPass)

  StringRef getArgument() const final { return "generate-rocm-tri-solve-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.tri_solve directive into a batched triangular "
           "solve gpu kernel (one thread per matrix/RHS-column)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.tri_solve")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.tri_solve missing name");
        return signalPassFailure();
      }
      bool lower = true;
      if (auto a = op->getAttrOfType<BoolAttr>("lower"))
        lower = a.getValue();
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto fnTy = b.getFunctionType(
          {memF32, memF32, memF32, idxTy, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitTriBody(body, loc, gpuFunc, lower);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMTriSolveKernelPass() {
  return std::make_unique<GenerateROCMTriSolveKernelPass>();
}
