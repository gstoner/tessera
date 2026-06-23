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
// 2-D grid of one wave per macro-tile, an `scf.for` K-loop that accumulates
// across 16-wide K panels, and ragged-edge masking so non-multiple-of-16 shapes
// are correct.
//
// Stage L2 — *register-blocked macro-tiling*. `mt`/`nt` (default 1) make each
// wave compute an `mt`x`nt` grid of 16x16 output tiles: a loaded A fragment is
// reused across the `nt` B-tiles and a loaded B fragment across the `mt`
// A-tiles, so global-load traffic per output element drops ~the reuse factor.
// This is the occupancy/VGPR-budgeted knob the hand-written oracle sweeps
// (measured-best 3x4 for large problems). mt=nt=1 is the L1 single-tile body;
// the M=N=K=16, mt=nt=1 launch is still bit-identical to the oracle.
//
// Layout (RDNA wave32, identical to tessera_rocm_gemm.cpp + rocdl_emit.py):
//   lane L -> lane = L & 15, lhi = L >> 4
//   baseRow = blockIdx.y*16*mt, baseCol = blockIdx.x*16*nt
//   A frag a[mi][i] = A[(baseRow+mi*16+lane)*K + (k0+i)]   (masked)
//   B frag b[ni][i] = B[(k0+i)*N + (baseCol+ni*16+lane)]   (masked)
//   c[mi][ni] = wmma(a[mi], b[ni], c[mi][ni])  over the K-loop
//   D[(baseRow+mi*16+2e+lhi)*N + (baseCol+ni*16+lane)] = c[mi][ni][e]  (masked)
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

// Emit the problem-size-generic, register-blocked (mt x nt) WMMA GEMM body into
// `gpuFunc` (args: A, B, D : memref<?>, M, N, K : index).
void emitGeneralBody(OpBuilder &b, Location loc, gpu::GPUFuncOp gpuFunc,
                     int64_t mt, int64_t nt) {
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

  // This wave's mt x nt macro-tile origin.
  Value bidX = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bidY = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
  Value c16mt = b.create<arith::ConstantIndexOp>(loc, 16 * mt);
  Value c16nt = b.create<arith::ConstantIndexOp>(loc, 16 * nt);
  Value baseRow = b.create<arith::MulIOp>(loc, bidY, c16mt);
  Value baseCol = b.create<arith::MulIOp>(loc, bidX, c16nt);

  // Per-tile (loop-invariant) row/col this lane handles + in-bounds masks.
  // arK[mi] = (row)*K is the A row base offset; reused as the vector.load base.
  SmallVector<Value> arM(mt), arK(mt), arInb(mt), colN(nt), colInb(nt);
  for (int64_t mi = 0; mi < mt; ++mi) {
    Value off = b.create<arith::ConstantIndexOp>(loc, mi * 16);
    Value rowBase = b.create<arith::AddIOp>(loc, baseRow, off);
    arM[mi] = b.create<arith::AddIOp>(loc, rowBase, lane);
    arK[mi] = b.create<arith::MulIOp>(loc, arM[mi], K);
    arInb[mi] = b.create<arith::CmpIOp>(loc, slt, arM[mi], M);
  }
  for (int64_t ni = 0; ni < nt; ++ni) {
    Value off = b.create<arith::ConstantIndexOp>(loc, ni * 16);
    Value colBase = b.create<arith::AddIOp>(loc, baseCol, off);
    colN[ni] = b.create<arith::AddIOp>(loc, colBase, lane);
    colInb[ni] = b.create<arith::CmpIOp>(loc, slt, colN[ni], N);
  }
  SmallVector<Value> rowOrigin(mt);
  for (int64_t mi = 0; mi < mt; ++mi)
    rowOrigin[mi] = b.create<arith::AddIOp>(
        loc, baseRow, b.create<arith::ConstantIndexOp>(loc, mi * 16));

  // WMMA accumulation over mt*nt fragments, reusing each loaded fragment.
  auto wmmaAll = [&](OpBuilder &bb, Location l, ValueRange aFrag,
                     ValueRange bFrag, ValueRange acc) {
    SmallVector<Value> next(mt * nt);
    for (int64_t mi = 0; mi < mt; ++mi)
      for (int64_t ni = 0; ni < nt; ++ni) {
        OperationState wmma(l, "tessera_rocm.wmma");
        wmma.addOperands({aFrag[mi], bFrag[ni], acc[mi * nt + ni]});
        wmma.addTypes({accTy});
        next[mi * nt + ni] = bb.create(wmma)->getResult(0);
      }
    return next;
  };

  SmallVector<Value> initAccs(mt * nt, accZero);

  // --- Interior fast path: whole macro-tile in-bounds AND K % 16 == 0, so no
  //     element masking is needed. A fragments are a single contiguous
  //     vector.load (coalesced); B fragments are 16 unmasked strided loads.
  auto emitFast = [&](OpBuilder &fb) {
    auto kLoop = fb.create<scf::ForOp>(
        loc, c0, K, c16, initAccs,
        [&](OpBuilder &bb, Location l, Value k0, ValueRange iter) {
          SmallVector<Value> aFrag(mt), bFrag(nt, fragZero);
          for (int64_t mi = 0; mi < mt; ++mi) {
            Value base = bb.create<arith::AddIOp>(l, arK[mi], k0);
            aFrag[mi] =
                bb.create<vector::LoadOp>(l, fragTy, A, ValueRange{base});
          }
          for (int64_t i = 0; i < 16; ++i) {
            Value ci = bb.create<arith::ConstantIndexOp>(l, i);
            Value ak = bb.create<arith::AddIOp>(l, k0, ci);
            Value akN = bb.create<arith::MulIOp>(l, ak, N);
            for (int64_t ni = 0; ni < nt; ++ni) {
              Value lin = bb.create<arith::AddIOp>(l, akN, colN[ni]);
              Value v = bb.create<memref::LoadOp>(l, B, ValueRange{lin});
              bFrag[ni] = bb.create<vector::InsertOp>(l, v, bFrag[ni],
                                                      ArrayRef<int64_t>{i});
            }
          }
          bb.create<scf::YieldOp>(l, wmmaAll(bb, l, aFrag, bFrag, iter));
        });
    // Unmasked store (the whole macro-tile is in-bounds).
    for (int64_t mi = 0; mi < mt; ++mi)
      for (int64_t ni = 0; ni < nt; ++ni) {
        Value accF = kLoop.getResult(mi * nt + ni);
        for (int64_t e = 0; e < 8; ++e) {
          Value twoE = fb.create<arith::ConstantIndexOp>(loc, e * 2);
          Value rowOff = fb.create<arith::AddIOp>(loc, twoE, lhi);
          Value r = fb.create<arith::AddIOp>(loc, rowOrigin[mi], rowOff);
          Value dv = fb.create<vector::ExtractOp>(loc, accF, ArrayRef<int64_t>{e});
          Value didx = fb.create<arith::AddIOp>(
              loc, fb.create<arith::MulIOp>(loc, r, N), colN[ni]);
          fb.create<memref::StoreOp>(loc, dv, D, ValueRange{didx});
        }
      }
  };

  // --- Masked edge path: ragged rows/cols or K not a multiple of 16. Every
  //     element is clamp-and-select masked; stores are scf.if-guarded.
  auto emitMasked = [&](OpBuilder &mb) {
    auto kLoop = mb.create<scf::ForOp>(
        loc, c0, K, c16, initAccs,
        [&](OpBuilder &bb, Location l, Value k0, ValueRange iter) {
          SmallVector<Value> aFrag(mt, fragZero), bFrag(nt, fragZero);
          for (int64_t i = 0; i < 16; ++i) {
            Value ci = bb.create<arith::ConstantIndexOp>(l, i);
            Value ak = bb.create<arith::AddIOp>(l, k0, ci);
            Value akInb = bb.create<arith::CmpIOp>(l, slt, ak, K);
            for (int64_t mi = 0; mi < mt; ++mi) {
              Value inb = bb.create<arith::AndIOp>(l, arInb[mi], akInb);
              Value lin = bb.create<arith::AddIOp>(l, arK[mi], ak);
              Value safe = bb.create<arith::SelectOp>(l, inb, lin, c0);
              Value v = bb.create<memref::LoadOp>(l, A, ValueRange{safe});
              Value vm = bb.create<arith::SelectOp>(l, inb, v, f16Zero);
              aFrag[mi] = bb.create<vector::InsertOp>(l, vm, aFrag[mi],
                                                      ArrayRef<int64_t>{i});
            }
            for (int64_t ni = 0; ni < nt; ++ni) {
              Value inb = bb.create<arith::AndIOp>(l, akInb, colInb[ni]);
              Value lin = bb.create<arith::AddIOp>(
                  l, bb.create<arith::MulIOp>(l, ak, N), colN[ni]);
              Value safe = bb.create<arith::SelectOp>(l, inb, lin, c0);
              Value v = bb.create<memref::LoadOp>(l, B, ValueRange{safe});
              Value vm = bb.create<arith::SelectOp>(l, inb, v, f16Zero);
              bFrag[ni] = bb.create<vector::InsertOp>(l, vm, bFrag[ni],
                                                      ArrayRef<int64_t>{i});
            }
          }
          bb.create<scf::YieldOp>(l, wmmaAll(bb, l, aFrag, bFrag, iter));
        });
    for (int64_t mi = 0; mi < mt; ++mi)
      for (int64_t ni = 0; ni < nt; ++ni) {
        Value accF = kLoop.getResult(mi * nt + ni);
        for (int64_t e = 0; e < 8; ++e) {
          Value twoE = mb.create<arith::ConstantIndexOp>(loc, e * 2);
          Value rowOff = mb.create<arith::AddIOp>(loc, twoE, lhi);
          Value r = mb.create<arith::AddIOp>(loc, rowOrigin[mi], rowOff);
          Value rInb = mb.create<arith::CmpIOp>(loc, slt, r, M);
          Value inb = mb.create<arith::AndIOp>(loc, rInb, colInb[ni]);
          auto ifOp = mb.create<scf::IfOp>(loc, inb, /*withElseRegion=*/false);
          OpBuilder::InsertionGuard g(mb);
          mb.setInsertionPointToStart(ifOp.thenBlock());
          Value dv = mb.create<vector::ExtractOp>(loc, accF, ArrayRef<int64_t>{e});
          Value didx = mb.create<arith::AddIOp>(
              loc, mb.create<arith::MulIOp>(loc, r, N), colN[ni]);
          mb.create<memref::StoreOp>(loc, dv, D, ValueRange{didx});
        }
      }
  };

  // fastCond = (baseRow+16*mt <= M) && (baseCol+16*nt <= N) && (K % 16 == 0).
  auto sle = arith::CmpIPredicate::sle;
  Value rowEnd = b.create<arith::AddIOp>(loc, baseRow, c16mt);
  Value colEnd = b.create<arith::AddIOp>(loc, baseCol, c16nt);
  Value rowFull = b.create<arith::CmpIOp>(loc, sle, rowEnd, M);
  Value colFull = b.create<arith::CmpIOp>(loc, sle, colEnd, N);
  Value kRem = b.create<arith::RemUIOp>(loc, K, c16);
  Value kAligned = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, kRem, c0);
  Value fastCond = b.create<arith::AndIOp>(
      loc, b.create<arith::AndIOp>(loc, rowFull, colFull), kAligned);

  auto ifOp = b.create<scf::IfOp>(loc, fastCond, /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(ifOp.thenBlock());
    emitFast(b);
    b.setInsertionPointToStart(ifOp.elseBlock());
    emitMasked(b);
  }
  b.setInsertionPointToEnd(&gpuFunc.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateWMMAGemmKernelPass
    : PassWrapper<GenerateWMMAGemmKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateWMMAGemmKernelPass)

  StringRef getArgument() const final { return "generate-wmma-gemm-kernel"; }
  StringRef getDescription() const final {
    return "Stage K/L: expand a tessera_rocm.wmma_gemm directive into a "
           "problem-size-generic, register-blocked (mt x nt), fragment-"
           "materialized RDNA WMMA GEMM gpu kernel (compiler-generated)";
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
      // mt/nt default to 1 (DefaultValuedAttr); the register-blocked macro-tile.
      int64_t mt = 1, nt = 1;
      if (auto a = op->getAttrOfType<IntegerAttr>("mt"))
        mt = a.getInt();
      if (auto a = op->getAttrOfType<IntegerAttr>("nt"))
        nt = a.getInt();
      if (mt < 1 || nt < 1) {
        op->emitError("generate-wmma-gemm-kernel: mt/nt (macro-tile in WMMA "
                      "tiles) must be >= 1 (got ")
            << mt << "x" << nt << ")";
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
      emitGeneralBody(body, loc, gpuFunc, mt, nt);

      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateWMMAGemmKernelPass() {
  return std::make_unique<GenerateWMMAGemmKernelPass>();
}
