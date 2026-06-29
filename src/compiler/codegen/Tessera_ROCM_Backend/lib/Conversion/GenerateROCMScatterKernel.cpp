//===- GenerateROCMScatterKernel.cpp - indexed store / scatter ----------===//
//
// Expands a `tessera_rocm.scatter` directive into an indexed-store gpu kernel —
// the 0-reduce lane (P8 of S_SERIES_GAP_CLOSURE_PLAN), companion to the P4
// gather kernel. Where gather is an indexed LOAD, scatter is an indexed STORE
// into `out` indexed by `idx`, reducing duplicate targets:
//
//   mode "set":  out[idx[i]*row_len + c]  = src[i*row_len + c]
//   mode "add":  out[idx[i]*row_len + c] += src[...]   (scatter_add)
//   mode "min":  atomic min                            (scatter_reduce amin)
//   mode "max":  atomic max                            (scatter_reduce amax)
//
// One thread per source element (i, c): thread t -> i = t / row_len,
// c = t % row_len. The add/min/max modes use `memref.atomic_rmw` so concurrent
// updates to the same output row reduce correctly; `set` is a plain store (its
// duplicate-index order is unspecified, matching numpy fancy-index assignment).
// The host moves the scatter axis to 0, flattens the trailing dims into a
// `row_len`-wide row, and preloads `out` with the base tensor. CPU analog:
// avx512_scatter_f32. Args: (out : memref<?xf32>, out_rows : index,
// src : memref<?xf32>, idx : memref<?xi64>, n_idx : index, row_len : index).
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

static constexpr int64_t BD = 256;

void emitScatterBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                     StringRef mode) {
  Type i64 = b.getIntegerType(64);
  Type idxTy = b.getIndexType();
  auto slt = arith::CmpIPredicate::slt;
  auto sge = arith::CmpIPredicate::sge;

  b.setInsertionPointToStart(&f.getBody().front());
  Value OUT = f.getArgument(0), OUTROWS = f.getArgument(1);
  Value SRC = f.getArgument(2), IDX = f.getArgument(3);
  Value NIDX = f.getArgument(4), ROWLEN = f.getArgument(5);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, cBD), tid);
  // total = n_idx * row_len source elements
  Value total = b.create<arith::MulIOp>(loc, NIDX, ROWLEN);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value i = b.create<arith::DivUIOp>(loc, gid, ROWLEN);    // source row
  Value c = b.create<arith::RemUIOp>(loc, gid, ROWLEN);    // column
  Value rI = b.create<memref::LoadOp>(loc, IDX, ValueRange{i});  // i64 target
  Value zero64 = b.create<arith::ConstantOp>(loc, i64, b.getI64IntegerAttr(0));
  Value outRows64 = b.create<arith::IndexCastOp>(loc, i64, OUTROWS);
  Value ge0 = b.create<arith::CmpIOp>(loc, sge, rI, zero64);
  Value ltR = b.create<arith::CmpIOp>(loc, slt, rI, outRows64);
  Value ok = b.create<arith::AndIOp>(loc, ge0, ltR);
  auto wif = b.create<scf::IfOp>(loc, ok, /*withElse=*/false);
  {
    OpBuilder wb = OpBuilder::atBlockBegin(wif.thenBlock());
    Value r = wb.create<arith::IndexCastOp>(loc, idxTy, rI);
    Value dst = wb.create<arith::AddIOp>(
        loc, wb.create<arith::MulIOp>(loc, r, ROWLEN), c);
    Value v = wb.create<memref::LoadOp>(loc, SRC, ValueRange{gid});
    if (mode == "add") {
      wb.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addf, v, OUT,
                                     ValueRange{dst});
    } else if (mode == "min") {
      wb.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::minimumf, v, OUT,
                                     ValueRange{dst});
    } else if (mode == "max") {
      wb.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::maximumf, v, OUT,
                                     ValueRange{dst});
    } else {  // set
      wb.create<memref::StoreOp>(loc, v, OUT, ValueRange{dst});
    }
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMScatterKernelPass
    : PassWrapper<GenerateROCMScatterKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMScatterKernelPass)

  StringRef getArgument() const final { return "generate-rocm-scatter-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.scatter directive into an indexed-store gpu "
           "kernel with atomic reduce (scatter/scatter_add/scatter_reduce)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.scatter")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      auto modeAttr = op->getAttrOfType<StringAttr>("mode");
      if (!nameAttr || !modeAttr) {
        op->emitError("tessera_rocm.scatter missing name/mode");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type i64 = b.getIntegerType(64);
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto memI64 = MemRefType::get({ShapedType::kDynamic}, i64);
      auto fnTy = b.getFunctionType(
          {memF32, idxTy, memF32, memI64, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitScatterBody(body, loc, gpuFunc, modeAttr.getValue());
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMScatterKernelPass() {
  return std::make_unique<GenerateROCMScatterKernelPass>();
}
