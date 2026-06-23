//===- GenerateWMMAGemmKernel.cpp - compiler-generated WMMA GEMM ----------===//
//
// Expands a `tessera_rocm.wmma_gemm` matmul directive into a real, fragment-
// materialized RDNA WMMA GEMM kernel: a `gpu.module` + `gpu.func` whose body
// loads the A/B tiles into WMMA fragment vectors, calls `tessera_rocm.wmma`
// (Stage J lowers that to the real `rocdl.wmma` intrinsic), and stores the f32
// accumulator with the wave32 lane/element layout. The GEMM is therefore
// compiler-*generated*, not authored MLIR — the Stage K milestone.
//
// Stage L1 — *problem-size-generic* codegen. The directive's `m`/`n`/`k` are the
// WMMA instruction tile (16x16x16 — the only tile RDNA's V_WMMA exposes; other
// extents are a named error). The emitted kernel itself is generic over the
// runtime problem size: it takes `(A, B, D : memref<?>, M, N, K : index)`, a
// 2-D grid of one wave per 16x16 output tile (blockIdx.y -> M tiles,
// blockIdx.x -> N tiles), an `scf.for` K-loop that accumulates across 16-wide
// K panels, and ragged-edge masking so non-multiple-of-16 shapes are correct.
// One launch with M=N=K=16 / grid (1,1,1) reduces to the single tile that is
// bit-identical to the hardware_verified hand-written oracle.
//
// MT=NT=1 (one 16x16 output tile per wave). Register-blocked macro-tiling
// (the measured-best 3x4) is Stage L2 — carried as a directive/Tile-IR attr.
//
// Layout (RDNA wave32, identical to tessera_rocm_gemm.cpp + rocdl_emit.py):
//   lane L -> lane = L & 15, lhi = L >> 4
//   baseRow = blockIdx.y*16, baseCol = blockIdx.x*16
//   A frag a[i] = A[(baseRow+lane)*K + (k0+i)]  (masked: row<M && k0+i<K)
//   B frag b[i] = B[(k0+i)*N + (baseCol+lane)]  (masked: k0+i<K && col<N)
//   D[(baseRow+2e+lhi)*N + (baseCol+lane)] = acc[e]  for e in 0..7 (masked)
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// Emit the problem-size-generic MT=NT=1 WMMA GEMM body into `gpuFunc`
// (args: A, B, D : memref<?>, M, N, K : index).
void emitGeneralBody(OpBuilder &b, Location loc, gpu::GPUFuncOp gpuFunc) {
  b.setInsertionPointToStart(&gpuFunc.getBody().front());
  Value A = gpuFunc.getArgument(0);
  Value B = gpuFunc.getArgument(1);
  Value D = gpuFunc.getArgument(2);
  Value M = gpuFunc.getArgument(3);
  Value N = gpuFunc.getArgument(4);
  Value K = gpuFunc.getArgument(5);

  Type f16Ty = b.getF16Type();
  Type f32Ty = b.getF32Type();
  auto fragTy = VectorType::get({16}, f16Ty);
  auto accTy = VectorType::get({8}, f32Ty);
  auto slt = arith::CmpIPredicate::slt;

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c4 = b.create<arith::ConstantIndexOp>(loc, 4);
  Value c15 = b.create<arith::ConstantIndexOp>(loc, 15);
  Value c16 = b.create<arith::ConstantIndexOp>(loc, 16);
  Value f16Zero =
      b.create<arith::ConstantOp>(loc, f16Ty, b.getFloatAttr(f16Ty, 0.0));
  Value fragZero = b.create<arith::ConstantOp>(
      loc, fragTy,
      DenseElementsAttr::get(fragTy, APFloat(APFloat::IEEEhalf(), 0)));
  Value accZero = b.create<arith::ConstantOp>(
      loc, accTy, DenseElementsAttr::get(accTy, APFloat(0.0f)));

  // lane = threadIdx.x & 15; lhi = threadIdx.x >> 4.
  Value tx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value lane = b.create<arith::AndIOp>(loc, tx, c15);
  Value lhi = b.create<arith::ShRUIOp>(loc, tx, c4);

  // This wave's 16x16 output tile origin.
  Value bidX = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bidY = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
  Value baseRow = b.create<arith::MulIOp>(loc, bidY, c16);
  Value baseCol = b.create<arith::MulIOp>(loc, bidX, c16);

  // ar = A row this lane loads; col = B/D column this lane handles.
  Value ar = b.create<arith::AddIOp>(loc, baseRow, lane);
  Value col = b.create<arith::AddIOp>(loc, baseCol, lane);
  Value arInb = b.create<arith::CmpIOp>(loc, slt, ar, M);
  Value colInb = b.create<arith::CmpIOp>(loc, slt, col, N);

  // K-loop: acc = sum over 16-wide K panels of wmma(aFrag, bFrag, acc).
  auto kLoop = b.create<scf::ForOp>(
      loc, c0, K, c16, ValueRange{accZero},
      [&](OpBuilder &bb, Location l, Value k0, ValueRange iter) {
        Value acc = iter[0];
        Value aFrag = fragZero;
        Value bFrag = fragZero;
        for (int64_t i = 0; i < 16; ++i) {
          Value ci = bb.create<arith::ConstantIndexOp>(l, i);
          Value ak = bb.create<arith::AddIOp>(l, k0, ci);
          Value akInb = bb.create<arith::CmpIOp>(l, slt, ak, K);
          // A frag: a[i] = A[ar*K + ak], clamp index to 0 when OOB then mask.
          Value aInb = bb.create<arith::AndIOp>(l, arInb, akInb);
          Value aLin = bb.create<arith::AddIOp>(
              l, bb.create<arith::MulIOp>(l, ar, K), ak);
          Value aSafe = bb.create<arith::SelectOp>(l, aInb, aLin, c0);
          Value av = bb.create<memref::LoadOp>(l, A, ValueRange{aSafe});
          Value avM = bb.create<arith::SelectOp>(l, aInb, av, f16Zero);
          aFrag = bb.create<vector::InsertOp>(l, avM, aFrag,
                                              ArrayRef<int64_t>{i});
          // B frag: b[i] = B[ak*N + col].
          Value bInb = bb.create<arith::AndIOp>(l, akInb, colInb);
          Value bLin = bb.create<arith::AddIOp>(
              l, bb.create<arith::MulIOp>(l, ak, N), col);
          Value bSafe = bb.create<arith::SelectOp>(l, bInb, bLin, c0);
          Value bv = bb.create<memref::LoadOp>(l, B, ValueRange{bSafe});
          Value bvM = bb.create<arith::SelectOp>(l, bInb, bv, f16Zero);
          bFrag = bb.create<vector::InsertOp>(l, bvM, bFrag,
                                              ArrayRef<int64_t>{i});
        }
        // d = wmma(a, b, acc)  (tessera_rocm.wmma -> Stage J real rocdl.wmma).
        OperationState wmma(l, "tessera_rocm.wmma");
        wmma.addOperands({aFrag, bFrag, acc});
        wmma.addTypes({accTy});
        Value d = bb.create(wmma)->getResult(0);
        bb.create<scf::YieldOp>(l, ValueRange{d});
      });
  Value accF = kLoop.getResult(0);

  // Store: D[(baseRow+2e+lhi)*N + col] = acc[e], masked to the ragged edge.
  for (int64_t e = 0; e < 8; ++e) {
    Value twoE = b.create<arith::ConstantIndexOp>(loc, e * 2);
    Value rowOff = b.create<arith::AddIOp>(loc, twoE, lhi);
    Value r = b.create<arith::AddIOp>(loc, baseRow, rowOff);
    Value rInb = b.create<arith::CmpIOp>(loc, slt, r, M);
    Value inb = b.create<arith::AndIOp>(loc, rInb, colInb);
    auto ifOp = b.create<scf::IfOp>(loc, inb, /*withElseRegion=*/false);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(ifOp.thenBlock());
    Value dv = b.create<vector::ExtractOp>(loc, accF, ArrayRef<int64_t>{e});
    Value didx = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, r, N), col);
    b.create<memref::StoreOp>(loc, dv, D, ValueRange{didx});
  }
  b.setInsertionPointToEnd(&gpuFunc.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateWMMAGemmKernelPass
    : PassWrapper<GenerateWMMAGemmKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateWMMAGemmKernelPass)

  StringRef getArgument() const final { return "generate-wmma-gemm-kernel"; }
  StringRef getDescription() const final {
    return "Stage K/L1: expand a tessera_rocm.wmma_gemm directive into a "
           "problem-size-generic, fragment-materialized RDNA WMMA GEMM gpu "
           "kernel (compiler-generated)";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, vector::VectorDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
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
        op->emitError("generate-wmma-gemm-kernel: the WMMA instruction tile "
                      "must be 16x16x16 (got ")
            << mAttr.getInt() << "x" << nAttr.getInt() << "x" << kAttr.getInt()
            << "); RDNA V_WMMA exposes no other tile. The problem size is a "
               "runtime (M,N,K) kernel argument, not the tile";
        return signalPassFailure();
      }

      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      // gpu.module @<name>_mod { gpu.func @<name>(A,B,D,M,N,K) kernel { ... } }
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());

      Type f16Ty = b.getF16Type();
      Type f32Ty = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto abTy = MemRefType::get({ShapedType::kDynamic}, f16Ty);
      auto dTy = MemRefType::get({ShapedType::kDynamic}, f32Ty);
      auto fnTy = b.getFunctionType({abTy, abTy, dTy, idxTy, idxTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());

      OpBuilder body(gpuFunc.getContext());
      emitGeneralBody(body, loc, gpuFunc);

      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateWMMAGemmKernelPass() {
  return std::make_unique<GenerateWMMAGemmKernelPass>();
}
