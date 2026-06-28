//===- GenerateROCMSddmmKernel.cpp - sampled dense-dense matmul kernel ----===//
//
// Expands `tessera_rocm.sddmm` into a SAMPLED dense-dense matmul kernel —
// OUT[M,N] = (A[M,K] · Bt[N,K]^row) ⊙ mask[M,N], one thread per (i,j):
//
//   m = mask[i·N+j];  OUT[i·N+j] = (m≠0) ? m·Σ_k A[i·K+k]·Bt[j·K+k] : 0
//
// The mask≠0 guard makes it genuinely sampled — threads at masked-zero entries
// skip the length-K dot entirely (an scf.if around the scf.for). B is passed
// ROW-MAJOR-TRANSPOSED as Bt[N,K] so both dot operands are contiguous. All f32.
// Validated vs numpy on gfx1151.
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

void emitSddmmBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), Bt = f.getArgument(1), mask = f.getArgument(2);
  Value OUT = f.getArgument(3);
  Value M = f.getArgument(4), N = f.getArgument(5), K = f.getArgument(6);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value total = b.create<arith::MulIOp>(loc, M, N);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value i = b.create<arith::DivUIOp>(loc, gid, N);
  Value j = b.create<arith::RemUIOp>(loc, gid, N);
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value m = b.create<memref::LoadOp>(loc, mask, ValueRange{gid});
  Value nz = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, m, zero);

  auto sel = b.create<scf::IfOp>(loc, TypeRange{f32}, nz, /*withElse=*/true);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(sel.thenBlock());
    Value aoff = b.create<arith::MulIOp>(loc, i, K);
    Value boff = b.create<arith::MulIOp>(loc, j, K);
    auto lp = b.create<scf::ForOp>(loc, c0, K, c1, ValueRange{zero});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(lp.getBody());
      Value k = lp.getInductionVar();
      Value acc = lp.getRegionIterArgs()[0];
      Value av = b.create<memref::LoadOp>(
          loc, A, ValueRange{b.create<arith::AddIOp>(loc, aoff, k)});
      Value bv = b.create<memref::LoadOp>(
          loc, Bt, ValueRange{b.create<arith::AddIOp>(loc, boff, k)});
      Value nacc = b.create<arith::AddFOp>(loc, acc,
                                           b.create<arith::MulFOp>(loc, av, bv));
      b.create<scf::YieldOp>(loc, ValueRange{nacc});
    }
    Value scaled = b.create<arith::MulFOp>(loc, lp.getResult(0), m);
    b.create<scf::YieldOp>(loc, ValueRange{scaled});
  }
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(sel.elseBlock());
    b.create<scf::YieldOp>(loc, ValueRange{zero});
  }
  b.create<memref::StoreOp>(loc, sel.getResult(0), OUT, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSddmmKernelPass
    : PassWrapper<GenerateROCMSddmmKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMSddmmKernelPass)

  StringRef getArgument() const final { return "generate-rocm-sddmm-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.sddmm directive into a sampled dense-dense "
           "matmul gpu kernel (one thread per masked entry)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.sddmm")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.sddmm missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto fnTy = b.getFunctionType(
          {memF32, memF32, memF32, memF32, idxTy, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitSddmmBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSddmmKernelPass() {
  return std::make_unique<GenerateROCMSddmmKernelPass>();
}
