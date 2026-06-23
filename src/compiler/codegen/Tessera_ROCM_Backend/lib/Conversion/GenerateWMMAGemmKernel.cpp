//===- GenerateWMMAGemmKernel.cpp - compiler-generated WMMA GEMM ----------===//
//
// Expands a `tessera_rocm.wmma_gemm` matmul directive into a real, fragment-
// materialized RDNA WMMA GEMM kernel: a `gpu.module` + `gpu.func` whose body
// loads the A/B tiles into WMMA fragment vectors, calls `tessera_rocm.wmma`
// (Stage J lowers that to the real `rocdl.wmma` intrinsic), and stores the f32
// accumulator with the wave32 lane/element layout. The GEMM is therefore
// compiler-*generated*, not authored MLIR — the Stage K milestone.
//
// Stage L1 — problem-size-generic: `m`/`n`/`k` are the WMMA instruction tile
// (16x16x16 — the only tile RDNA's V_WMMA exposes). The kernel takes the runtime
// `(A,B,D : memref<?>, M,N,K : index)`, a 2-D grid of one wave per macro-tile.
//
// Stage L2 — register-blocked: `mt`/`nt` (default 1) make each wave compute an
// `mt`x`nt` grid of 16x16 output tiles with fragment reuse.
//
// dtype: f16 (default) or bf16 storage; f32 accumulate. The fragment element
// type / memref element type follow `dtype`; Stage J emits the matching
// rocdl.wmma.*.{f16,bf16} intrinsic.
//
// Performance structure — the K-loop is split so masking never sits on the hot
// path. Per wave:
//   * main loop over the aligned K range [0, kMain) (kMain = K rounded down to a
//     multiple of 16), then a single masked tail panel for [kMain, K) when K is
//     ragged. So ragged K costs one extra masked panel, not a masked K-loop.
//   * the main panel is chosen by whether the wave's macro-tile is interior:
//       - fast  (tile fully in-bounds): contiguous vector.load A, unmasked B.
//       - edge  (tile straddles the M/N edge): load A/B at a row/col clamped
//         into range, then zero an OOB fragment with ONE loop-invariant vector
//         select — keeps coalesced loads, so ragged M/N stays vector-load speed.
//   * stores are scf.if-masked only when the tile is ragged.
//
// Layout (RDNA wave32, identical to tessera_rocm_gemm.cpp + rocdl_emit.py):
//   lane L -> lane = L & 15, lhi = L >> 4
//   baseRow = blockIdx.y*16*mt, baseCol = blockIdx.x*16*nt
//   A frag a[mi][i] = A[(baseRow+mi*16+lane)*K + (k0+i)]   (masked)
//   B frag b[ni][i] = B[(k0+i)*N + (baseCol+ni*16+lane)]   (masked)
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
// `gpuFunc` (args: A, B, D : memref<?>, M, N, K : index). `storeTy` is the A/B
// storage element type (f16 or bf16); the accumulator is always f32.
void emitGeneralBody(OpBuilder &b, Location loc, gpu::GPUFuncOp gpuFunc,
                     int64_t mt, int64_t nt, Type storeTy) {
  b.setInsertionPointToStart(&gpuFunc.getBody().front());
  Value A = gpuFunc.getArgument(0);
  Value B = gpuFunc.getArgument(1);
  Value D = gpuFunc.getArgument(2);
  Value M = gpuFunc.getArgument(3);
  Value N = gpuFunc.getArgument(4);
  Value K = gpuFunc.getArgument(5);

  Type f32Ty = b.getF32Type();
  auto fragTy = VectorType::get({16}, storeTy);
  auto accTy = VectorType::get({8}, f32Ty);
  auto slt = arith::CmpIPredicate::slt;

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c4 = b.create<arith::ConstantIndexOp>(loc, 4);
  Value c15 = b.create<arith::ConstantIndexOp>(loc, 15);
  Value c16 = b.create<arith::ConstantIndexOp>(loc, 16);
  Value storeZero =
      b.create<arith::ConstantOp>(loc, storeTy, b.getFloatAttr(storeTy, 0.0));
  APFloat zeroAP = cast<FloatAttr>(b.getFloatAttr(storeTy, 0.0)).getValue();
  Value fragZero = b.create<arith::ConstantOp>(
      loc, fragTy, DenseElementsAttr::get(fragTy, zeroAP));
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

  // Per-tile (loop-invariant) values.
  //   arK[mi]     = row*K              — A-fragment base offset (fast path).
  //   arKsafe[mi] = clamp(row,0)*K     — same, OOB row clamped to 0 so the edge
  //                                      path can still issue a vector.load.
  //   colSafe[ni] = clamp(col,0)       — likewise for the B column.
  SmallVector<Value> arM(mt), arK(mt), arKsafe(mt), arInb(mt), rowOrigin(mt);
  SmallVector<Value> colN(nt), colSafe(nt), colInb(nt);
  for (int64_t mi = 0; mi < mt; ++mi) {
    Value off = b.create<arith::ConstantIndexOp>(loc, mi * 16);
    Value rowBase = b.create<arith::AddIOp>(loc, baseRow, off);
    rowOrigin[mi] = rowBase;
    arM[mi] = b.create<arith::AddIOp>(loc, rowBase, lane);
    arK[mi] = b.create<arith::MulIOp>(loc, arM[mi], K);
    arInb[mi] = b.create<arith::CmpIOp>(loc, slt, arM[mi], M);
    Value rowSafe = b.create<arith::SelectOp>(loc, arInb[mi], arM[mi], c0);
    arKsafe[mi] = b.create<arith::MulIOp>(loc, rowSafe, K);
  }
  for (int64_t ni = 0; ni < nt; ++ni) {
    Value off = b.create<arith::ConstantIndexOp>(loc, ni * 16);
    Value colBase = b.create<arith::AddIOp>(loc, baseCol, off);
    colN[ni] = b.create<arith::AddIOp>(loc, colBase, lane);
    colInb[ni] = b.create<arith::CmpIOp>(loc, slt, colN[ni], N);
    colSafe[ni] = b.create<arith::SelectOp>(loc, colInb[ni], colN[ni], c0);
  }

  // WMMA accumulation over mt*nt fragments, reusing each loaded fragment.
  auto wmmaAll = [&](OpBuilder &bb, Location l, ArrayRef<Value> aFrag,
                     ArrayRef<Value> bFrag, ValueRange acc) {
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

  // --- fast panel: interior tile, full K panel — no masking. ---
  auto fastPanel = [&](OpBuilder &bb, Location l, Value k0, ValueRange acc) {
    SmallVector<Value> aFrag(mt), bFrag(nt, fragZero);
    for (int64_t mi = 0; mi < mt; ++mi) {
      Value base = bb.create<arith::AddIOp>(l, arK[mi], k0);
      aFrag[mi] = bb.create<vector::LoadOp>(l, fragTy, A, ValueRange{base});
    }
    for (int64_t i = 0; i < 16; ++i) {
      Value ci = bb.create<arith::ConstantIndexOp>(l, i);
      Value ak = bb.create<arith::AddIOp>(l, k0, ci);
      Value akN = bb.create<arith::MulIOp>(l, ak, N);
      for (int64_t ni = 0; ni < nt; ++ni) {
        Value lin = bb.create<arith::AddIOp>(l, akN, colN[ni]);
        Value v = bb.create<memref::LoadOp>(l, B, ValueRange{lin});
        bFrag[ni] =
            bb.create<vector::InsertOp>(l, v, bFrag[ni], ArrayRef<int64_t>{i});
      }
    }
    return wmmaAll(bb, l, aFrag, bFrag, acc);
  };

  // --- edge panel: full K panel, ragged M/N — coalesced loads at a clamped
  //     row/col, then one vector select zeroes an OOB fragment. ---
  auto edgePanel = [&](OpBuilder &bb, Location l, Value k0, ValueRange acc) {
    SmallVector<Value> aFrag(mt), bFrag(nt, fragZero);
    for (int64_t mi = 0; mi < mt; ++mi) {
      Value base = bb.create<arith::AddIOp>(l, arKsafe[mi], k0);
      Value v = bb.create<vector::LoadOp>(l, fragTy, A, ValueRange{base});
      aFrag[mi] = bb.create<arith::SelectOp>(l, arInb[mi], v, fragZero);
    }
    for (int64_t i = 0; i < 16; ++i) {
      Value ci = bb.create<arith::ConstantIndexOp>(l, i);
      Value ak = bb.create<arith::AddIOp>(l, k0, ci);
      Value akN = bb.create<arith::MulIOp>(l, ak, N);
      for (int64_t ni = 0; ni < nt; ++ni) {
        Value lin = bb.create<arith::AddIOp>(l, akN, colSafe[ni]);
        Value v = bb.create<memref::LoadOp>(l, B, ValueRange{lin});
        bFrag[ni] =
            bb.create<vector::InsertOp>(l, v, bFrag[ni], ArrayRef<int64_t>{i});
      }
    }
    for (int64_t ni = 0; ni < nt; ++ni)
      bFrag[ni] =
          bb.create<arith::SelectOp>(l, colInb[ni], bFrag[ni], fragZero);
    return wmmaAll(bb, l, aFrag, bFrag, acc);
  };

  // --- masked panel: ragged K tail — per-element clamp-and-select on both K and
  //     M/N. Runs once (the [kMain,K) remainder), so the cost is off the hot
  //     path. Correct for full or ragged M/N (masks are no-ops when in-bounds).
  auto maskedPanel = [&](OpBuilder &bb, Location l, Value k0, ValueRange acc) {
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
        Value vm = bb.create<arith::SelectOp>(l, inb, v, storeZero);
        aFrag[mi] =
            bb.create<vector::InsertOp>(l, vm, aFrag[mi], ArrayRef<int64_t>{i});
      }
      for (int64_t ni = 0; ni < nt; ++ni) {
        Value inb = bb.create<arith::AndIOp>(l, akInb, colInb[ni]);
        Value lin = bb.create<arith::AddIOp>(
            l, bb.create<arith::MulIOp>(l, ak, N), colN[ni]);
        Value safe = bb.create<arith::SelectOp>(l, inb, lin, c0);
        Value v = bb.create<memref::LoadOp>(l, B, ValueRange{safe});
        Value vm = bb.create<arith::SelectOp>(l, inb, v, storeZero);
        bFrag[ni] =
            bb.create<vector::InsertOp>(l, vm, bFrag[ni], ArrayRef<int64_t>{i});
      }
    }
    return wmmaAll(bb, l, aFrag, bFrag, acc);
  };

  // Shared store. When `masked`, each store is scf.if-guarded against the ragged
  // M/N edge (stores run once, so the guard cost is negligible).
  auto emitStore = [&](OpBuilder &sb, ValueRange accs, bool masked) {
    for (int64_t mi = 0; mi < mt; ++mi)
      for (int64_t ni = 0; ni < nt; ++ni) {
        Value accF = accs[mi * nt + ni];
        for (int64_t e = 0; e < 8; ++e) {
          Value twoE = sb.create<arith::ConstantIndexOp>(loc, e * 2);
          Value rowOff = sb.create<arith::AddIOp>(loc, twoE, lhi);
          Value r = sb.create<arith::AddIOp>(loc, rowOrigin[mi], rowOff);
          Value dv =
              sb.create<vector::ExtractOp>(loc, accF, ArrayRef<int64_t>{e});
          Value didx = sb.create<arith::AddIOp>(
              loc, sb.create<arith::MulIOp>(loc, r, N), colN[ni]);
          if (!masked) {
            sb.create<memref::StoreOp>(loc, dv, D, ValueRange{didx});
            continue;
          }
          Value rInb = sb.create<arith::CmpIOp>(loc, slt, r, M);
          Value inb = sb.create<arith::AndIOp>(loc, rInb, colInb[ni]);
          auto ifOp = sb.create<scf::IfOp>(loc, inb, /*withElseRegion=*/false);
          OpBuilder::InsertionGuard g(sb);
          sb.setInsertionPointToStart(ifOp.thenBlock());
          sb.create<memref::StoreOp>(loc, dv, D, ValueRange{didx});
        }
      }
  };

  SmallVector<Value> initAccs(mt * nt, accZero);

  // kMain = largest multiple of 16 <= K; the tail panel covers [kMain, K).
  Value kRem = b.create<arith::RemUIOp>(loc, K, c16);
  Value kMain = b.create<arith::SubIOp>(loc, K, kRem);
  Value needTail =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, kRem, c0);

  // Run the aligned main K-loop with `mainPanel`, fold the ragged-K tail (when
  // present) through `maskedPanel`, and store (masked iff ragged M/N).
  auto runPath = [&](OpBuilder &rb, function_ref<SmallVector<Value>(
                                        OpBuilder &, Location, Value, ValueRange)>
                                        mainPanel,
                     bool masked) {
    auto kLoop = rb.create<scf::ForOp>(
        loc, c0, kMain, c16, initAccs,
        [&](OpBuilder &bb, Location l, Value k0, ValueRange iter) {
          bb.create<scf::YieldOp>(l, mainPanel(bb, l, k0, iter));
        });
    auto tail = rb.create<scf::IfOp>(
        loc, needTail,
        [&](OpBuilder &tb, Location l) {
          tb.create<scf::YieldOp>(
              l, maskedPanel(tb, l, kMain, kLoop.getResults()));
        },
        [&](OpBuilder &eb, Location l) {
          eb.create<scf::YieldOp>(l, ValueRange(kLoop.getResults()));
        });
    emitStore(rb, tail.getResults(), masked);
  };

  // Dispatch on whether the macro-tile is interior (fast) or straddles the
  // M/N edge (edge); ragged K is handled by the tail in both.
  Value rowEnd = b.create<arith::AddIOp>(loc, baseRow, c16mt);
  Value colEnd = b.create<arith::AddIOp>(loc, baseCol, c16nt);
  auto sle = arith::CmpIPredicate::sle;
  Value rowFull = b.create<arith::CmpIOp>(loc, sle, rowEnd, M);
  Value colFull = b.create<arith::CmpIOp>(loc, sle, colEnd, N);
  Value tileFull = b.create<arith::AndIOp>(loc, rowFull, colFull);

  auto outer = b.create<scf::IfOp>(loc, tileFull, /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(outer.thenBlock());
    runPath(b, fastPanel, /*masked=*/false);
    b.setInsertionPointToStart(outer.elseBlock());
    runPath(b, edgePanel, /*masked=*/true);
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

      // dtype: f16 (default) or bf16 storage; f32 accumulate. RDNA exposes
      // both f16 and bf16 WMMA at 16x16x16.
      Type storeTy = b.getF16Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16")
          storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16")
          storeTy = b.getBF16Type();
        else {
          op->emitError("generate-wmma-gemm-kernel: dtype must be f16 or bf16 "
                        "(got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      // gpu.module @<name>_mod { gpu.func @<name>(A,B,D,M,N,K) kernel { ... } }
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());

      Type f32Ty = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto abTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto dTy = MemRefType::get({ShapedType::kDynamic}, f32Ty);
      auto fnTy = b.getFunctionType({abTy, abTy, dTy, idxTy, idxTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());

      OpBuilder bodyB(gpuFunc.getContext());
      emitGeneralBody(bodyB, loc, gpuFunc, mt, nt, storeTy);

      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateWMMAGemmKernelPass() {
  return std::make_unique<GenerateWMMAGemmKernelPass>();
}
