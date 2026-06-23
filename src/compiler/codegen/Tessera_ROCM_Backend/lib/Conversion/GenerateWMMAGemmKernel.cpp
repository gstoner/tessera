//===- GenerateWMMAGemmKernel.cpp - Stage K: compiler-generated WMMA GEMM -===//
//
// Expands a `tessera_rocm.wmma_gemm` matmul directive into a real, fragment-
// materialized RDNA WMMA GEMM kernel: a `gpu.module` + `gpu.func` whose body
// loads the A/B tiles into WMMA fragment vectors, calls `tessera_rocm.wmma`
// (Stage J lowers that to the real `rocdl.wmma` intrinsic), and stores the f32
// accumulator with the wave32 lane/element layout. The GEMM is therefore
// compiler-*generated*, not authored MLIR — the Stage K milestone.
//
// First realization: the single 16x16x16 tile (the shape proven bit-identical
// to the hardware_verified hand-written oracle). The body is emitted fully
// unrolled (no scf loops) — 16 fragment loads, one wmma, 8 accumulator stores.
//
// Layout (RDNA wave32, identical to tessera_rocm_gemm.cpp + rocdl_emit.py):
//   lane L -> l15 = L & 15, lhi = L >> 4
//   A frag a[i] = A[l15*16 + i]   (contiguous row -> one vector.load)
//   B frag b[i] = B[i*16 + l15]   (strided -> 16 loads + inserts)
//   D[(2e+lhi)*16 + l15] = acc[e]  for e in 0..7
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// Emit the unrolled 16x16x16 WMMA GEMM body into `gpuFunc` (args: A, B, D).
void emitTileBody(OpBuilder &b, Location loc, gpu::GPUFuncOp gpuFunc) {
  b.setInsertionPointToStart(&gpuFunc.getBody().front());
  Value A = gpuFunc.getArgument(0);
  Value B = gpuFunc.getArgument(1);
  Value D = gpuFunc.getArgument(2);

  Type idxTy = b.getIndexType();
  Type i32Ty = b.getIntegerType(32);
  Type f16Ty = b.getF16Type();
  Type f32Ty = b.getF32Type();
  auto fragTy = VectorType::get({16}, f16Ty);
  auto accTy = VectorType::get({8}, f32Ty);

  Value c16 = b.create<arith::ConstantIndexOp>(loc, 16);
  Value c15i = b.create<arith::ConstantIntOp>(loc, 15, 32);
  Value c4i = b.create<arith::ConstantIntOp>(loc, 4, 32);

  // lane = threadIdx.x; l15 = lane & 15; lhi = lane >> 4.
  Value tx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value txi = b.create<arith::IndexCastOp>(loc, i32Ty, tx);
  Value l15i = b.create<arith::AndIOp>(loc, txi, c15i);
  Value lhii = b.create<arith::ShRUIOp>(loc, txi, c4i);
  Value l15 = b.create<arith::IndexCastOp>(loc, idxTy, l15i);
  Value lhi = b.create<arith::IndexCastOp>(loc, idxTy, lhii);

  // A frag: contiguous row l15 -> a[i] = A[l15*16 + i].
  Value arow = b.create<arith::MulIOp>(loc, l15, c16);
  Value aFrag = b.create<vector::LoadOp>(loc, fragTy, A, ValueRange{arow});

  // B frag: b[i] = B[i*16 + l15], strided -> unrolled loads + inserts.
  Value zeroFrag = b.create<arith::ConstantOp>(
      loc, fragTy, DenseElementsAttr::get(fragTy, APFloat(APFloat::IEEEhalf(), 0)));
  Value bFrag = zeroFrag;
  for (int64_t i = 0; i < 16; ++i) {
    Value off = b.create<arith::ConstantIndexOp>(loc, i * 16);
    Value idx = b.create<arith::AddIOp>(loc, off, l15);
    Value bv = b.create<memref::LoadOp>(loc, B, ValueRange{idx});
    bFrag = b.create<vector::InsertOp>(loc, bv, bFrag, ArrayRef<int64_t>{i});
  }

  // acc = 0; d = wmma(a, b, acc)  (tessera_rocm.wmma -> Stage J real rocdl.wmma).
  Value accZero = b.create<arith::ConstantOp>(
      loc, accTy, DenseElementsAttr::get(accTy, APFloat(0.0f)));
  OperationState wmma(loc, "tessera_rocm.wmma");
  wmma.addOperands({aFrag, bFrag, accZero});
  wmma.addTypes({accTy});
  Value d = b.create(wmma)->getResult(0);

  // Store: D[(2e+lhi)*16 + l15] = acc[e].
  for (int64_t e = 0; e < 8; ++e) {
    Value dv = b.create<vector::ExtractOp>(loc, d, ArrayRef<int64_t>{e});
    Value twoE = b.create<arith::ConstantIndexOp>(loc, e * 2);
    Value row = b.create<arith::AddIOp>(loc, twoE, lhi);
    Value rowOff = b.create<arith::MulIOp>(loc, row, c16);
    Value didx = b.create<arith::AddIOp>(loc, rowOff, l15);
    b.create<memref::StoreOp>(loc, dv, D, ValueRange{didx});
  }
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateWMMAGemmKernelPass
    : PassWrapper<GenerateWMMAGemmKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateWMMAGemmKernelPass)

  StringRef getArgument() const final { return "generate-wmma-gemm-kernel"; }
  StringRef getDescription() const final {
    return "Stage K: expand a tessera_rocm.wmma_gemm directive into a fragment-"
           "materialized RDNA WMMA GEMM gpu kernel (compiler-generated)";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, vector::VectorDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.wmma_gemm")
        directives.push_back(op);
    });

    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      auto mAttr = op->getAttrOfType<IntegerAttr>("m");
      auto nAttr = op->getAttrOfType<IntegerAttr>("n");
      auto kAttr = op->getAttrOfType<IntegerAttr>("k");
      if (!nameAttr || !mAttr || !nAttr || !kAttr) {
        op->emitError("tessera_rocm.wmma_gemm missing name/m/n/k");
        return signalPassFailure();
      }
      if (mAttr.getInt() != 16 || nAttr.getInt() != 16 || kAttr.getInt() != 16) {
        op->emitError("generate-wmma-gemm-kernel: only the 16x16x16 tile is "
                      "implemented (got ")
            << mAttr.getInt() << "x" << nAttr.getInt() << "x" << kAttr.getInt()
            << "); the multi-tile loop nest is future work";
        return signalPassFailure();
      }

      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      // gpu.module @<name>_mod { gpu.func @<name>(A, B, D) kernel { ... } }
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());

      Type f16Ty = b.getF16Type();
      Type f32Ty = b.getF32Type();
      auto abTy = MemRefType::get({256}, f16Ty);
      auto dTy = MemRefType::get({256}, f32Ty);
      auto fnTy = b.getFunctionType({abTy, abTy, dTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());

      OpBuilder body(gpuFunc.getContext());
      emitTileBody(body, loc, gpuFunc);

      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateWMMAGemmKernelPass() {
  return std::make_unique<GenerateWMMAGemmKernelPass>();
}
