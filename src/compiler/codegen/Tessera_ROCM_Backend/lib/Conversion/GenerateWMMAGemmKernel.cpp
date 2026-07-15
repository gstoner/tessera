//===- GenerateWMMAGemmKernel.cpp - compiler-generated WMMA GEMM ----------===//
//
// Expands a portable `tile.matmul_kernel` or legacy
// `tessera_rocm.wmma_gemm` directive into a real, fragment-materialized RDNA
// WMMA GEMM kernel. Both front doors populate one in-memory request consumed by
// the same generator; the portable path creates no temporary target op. The
// generated `gpu.module` + `gpu.func`
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
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Dialect/Tile/TileEpilogue.h"
#include "TesseraROCMDialect.h.inc"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

// The WMMA element/fragment/accumulator types for one dtype.
//   store   : A/B memref element (f16 / bf16 / i8).
//   load    : vector<16 x store> — the fragment as loaded/built per lane.
//   frag    : the `tessera_rocm.wmma` operand type. == load for f16/bf16; for
//             int8 the 16 i8 are bitcast to vector<4xi32> (the iu8 ABI).
//   acc     : vector<8 x accElem> accumulator/result.
//   accElem : f32 / i32 — the D memref element.
//   pack    : how a loaded `load` vector becomes a `frag` operand —
//             0 = identity (f16/bf16), 1 = bitcast vector<16xi8>->vector<4xi32>
//             (int8/iu8), 2 = nibble-pack 16 i8 (each an int4 value in [-8,7])
//             into vector<2xi32> (int4/iu4).
struct WmmaTypes {
  Type store, load, frag, acc, accElem;
  bool isInt;
  // `pack` is the in-kernel codegen ABI mode (0 = fp passthrough, 1 = int8
  // bitcast->v4i32, 2 = int4 nibble-pack->v2i32) — an emission detail, NOT the
  // storage-pack contract. `packFactor` is the contract: logical values per
  // byte container (f16/bf16/int8 -> 1, int4 -> 2), exactly the `factor`
  // StoragePackConsume computes (container_bits / storage_bits). Keep them
  // separate so a future int ABI mode (a new `pack` at the same logical factor)
  // cannot silently break the contract check — verify against `packFactor`.
  int pack;
  int packFactor;
};

/// Backend-neutral input to the one gfx11 WMMA kernel generator. Portable
/// tile.matmul_kernel and the legacy tessera_rocm.wmma_gemm directive are only
/// adapters that populate this request; neither creates an intermediate IR op.
struct WmmaGemmRequest {
  Operation *anchor = nullptr;
  Operation *eraseOwner = nullptr;
  std::string name;
  int64_t m = 16, n = 16, k = 16;
  int64_t mt = 1, nt = 1;
  std::string dtype = "f16";
  std::string activation = "none";
  std::string output;
  bool bias = false;
  bool portableABI = false;
  DictionaryAttr storagePack;
};

// Emit the problem-size-generic, register-blocked (mt x nt) WMMA GEMM body into
// `gpuFunc` (args: A, B, D : memref<?>, M, N, K : index), for the dtype in `T`.
void emitGeneralBody(OpBuilder &b, Location loc, gpu::GPUFuncOp gpuFunc,
                     int64_t mt, int64_t nt, const WmmaTypes &T,
                     Type outputType, bool portableABI = false,
                     bool viaTile = false, bool hasBias = false,
                     StringRef activation = "none") {
  b.setInsertionPointToStart(&gpuFunc.getBody().front());
  Value A = gpuFunc.getArgument(0);
  Value B = gpuFunc.getArgument(1);
  // Preserve the portable Tile ABI (A, B, bias, D, M, N, K). The legacy
  // backend directive retains its historical (A, B, D, M, N, K, bias) ABI.
  unsigned dIndex = portableABI && hasBias ? 3 : 2;
  Value D = gpuFunc.getArgument(dIndex);
  Value M = gpuFunc.getArgument(dIndex + 1);
  Value N = gpuFunc.getArgument(dIndex + 2);
  Value K = gpuFunc.getArgument(dIndex + 3);
  // Fused-epilogue bias is the trailing memref arg (length N), present only when
  // `hasBias`. Only float dtypes reach the epilogue (gated at the pass level).
  Value bias = hasBias
      ? gpuFunc.getArgument(portableABI ? 2 : 6)
      : Value();

  Type loadTy = T.load;
  Type fragTy = T.frag;
  Type accTy = T.acc;
  auto slt = arith::CmpIPredicate::slt;

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c4 = b.create<arith::ConstantIndexOp>(loc, 4);
  Value c15 = b.create<arith::ConstantIndexOp>(loc, 15);
  Value c16 = b.create<arith::ConstantIndexOp>(loc, 16);

  // Zero constants: a scalar store-element zero (per-element masking), the
  // loaded-fragment zero (edge select / masked-build init), and the accumulator
  // zero — built from APInt for integer dtypes, APFloat otherwise.
  Value storeZero, loadZero, accZero;
  if (T.isInt) {
    unsigned sw = cast<IntegerType>(T.store).getWidth();
    storeZero = b.create<arith::ConstantOp>(loc, T.store,
                                            b.getIntegerAttr(T.store, 0));
    loadZero = b.create<arith::ConstantOp>(
        loc, loadTy, DenseElementsAttr::get(cast<ShapedType>(loadTy),
                                            APInt(sw, 0)));
    accZero = b.create<arith::ConstantOp>(
        loc, accTy, DenseElementsAttr::get(cast<ShapedType>(accTy),
                                           APInt(32, 0)));
  } else {
    storeZero = b.create<arith::ConstantOp>(loc, T.store,
                                            b.getFloatAttr(T.store, 0.0));
    APFloat zAP = cast<FloatAttr>(b.getFloatAttr(T.store, 0.0)).getValue();
    loadZero = b.create<arith::ConstantOp>(
        loc, loadTy, DenseElementsAttr::get(cast<ShapedType>(loadTy), zAP));
    accZero = b.create<arith::ConstantOp>(
        loc, accTy, DenseElementsAttr::get(cast<ShapedType>(accTy),
                                           APFloat(0.0f)));
  }

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

  // Reinterpret a loaded fragment (vector<16 x store>) as the wmma operand type.
  //   pack 0 (f16/bf16): identity.
  //   pack 1 (int8/iu8): bitcast vector<16xi8> -> vector<4xi32> (byte k -> word
  //                      k/4, byte k%4 — the layout iu8 expects).
  //   pack 2 (int4/iu4): nibble-pack 16 int4 values (held in int8, low nibble)
  //                      into vector<2xi32> (value k -> word k/8, nibble k%8).
  Type i32Ty = b.getIntegerType(32);
  auto toFrag = [&](OpBuilder &bb, Location l, Value v) -> Value {
    if (T.pack == 0)
      return v;
    if (T.pack == 1)
      return bb.create<vector::BitCastOp>(l, fragTy, v);
    // pack == 2: int4 nibble compaction.
    Value cF = bb.create<arith::ConstantIntOp>(l, 0xF, 32);
    Value words[2] = {bb.create<arith::ConstantIntOp>(l, 0, 32),
                      bb.create<arith::ConstantIntOp>(l, 0, 32)};
    for (int64_t k = 0; k < 16; ++k) {
      Value e = bb.create<vector::ExtractOp>(l, v, ArrayRef<int64_t>{k}); // i8
      Value ei = bb.create<arith::ExtUIOp>(l, i32Ty, e);
      Value nib = bb.create<arith::AndIOp>(l, ei, cF);
      Value sh = bb.create<arith::ConstantIntOp>(l, 4 * (k % 8), 32);
      Value shf = bb.create<arith::ShLIOp>(l, nib, sh);
      words[k / 8] = bb.create<arith::OrIOp>(l, words[k / 8], shf);
    }
    Value frag = bb.create<arith::ConstantOp>(
        l, fragTy, DenseElementsAttr::get(cast<ShapedType>(fragTy), APInt(32, 0)));
    frag = bb.create<vector::InsertOp>(l, words[0], frag, ArrayRef<int64_t>{0});
    frag = bb.create<vector::InsertOp>(l, words[1], frag, ArrayRef<int64_t>{1});
    return frag;
  };

  // WMMA accumulation over mt*nt fragments, reusing each loaded fragment.
  auto wmmaAll = [&](OpBuilder &bb, Location l, ArrayRef<Value> aFrag,
                     ArrayRef<Value> bFrag, ValueRange acc) {
    SmallVector<Value> af(mt), bf(nt);
    for (int64_t mi = 0; mi < mt; ++mi)
      af[mi] = toFrag(bb, l, aFrag[mi]);
    for (int64_t ni = 0; ni < nt; ++ni)
      bf[ni] = toFrag(bb, l, bFrag[ni]);
    SmallVector<Value> next(mt * nt);
    for (int64_t mi = 0; mi < mt; ++mi)
      for (int64_t ni = 0; ni < nt; ++ni) {
        // Fork A (via-tile): emit the matrix op at the Tile-IR seam
        // (tile.mma %a, %b, %acc) so it flows through rocm-wave-lds-pipeline +
        // lower-tile-to-rocm, which lowers it back to tessera_rocm.wmma with the
        // SAME (a, b, acc) operands. Default path emits tessera_rocm.wmma
        // directly (the established executable lane). Same operands/types either
        // way — only the op name differs, so the lowered kernel is identical.
        OperationState wmma(l, viaTile ? "tile.mma" : "tessera_rocm.wmma");
        wmma.addOperands({af[mi], bf[ni], acc[mi * nt + ni]});
        wmma.addTypes({accTy});
        next[mi * nt + ni] = bb.create(wmma)->getResult(0);
      }
    return next;
  };

  // --- fast panel: interior tile, full K panel — no masking. ---
  auto fastPanel = [&](OpBuilder &bb, Location l, Value k0, ValueRange acc) {
    SmallVector<Value> aFrag(mt), bFrag(nt, loadZero);
    for (int64_t mi = 0; mi < mt; ++mi) {
      Value base = bb.create<arith::AddIOp>(l, arK[mi], k0);
      aFrag[mi] = bb.create<vector::LoadOp>(l, loadTy, A, ValueRange{base});
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
    SmallVector<Value> aFrag(mt), bFrag(nt, loadZero);
    for (int64_t mi = 0; mi < mt; ++mi) {
      Value base = bb.create<arith::AddIOp>(l, arKsafe[mi], k0);
      Value v = bb.create<vector::LoadOp>(l, loadTy, A, ValueRange{base});
      aFrag[mi] = bb.create<arith::SelectOp>(l, arInb[mi], v, loadZero);
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
          bb.create<arith::SelectOp>(l, colInb[ni], bFrag[ni], loadZero);
    return wmmaAll(bb, l, aFrag, bFrag, acc);
  };

  // --- masked panel: ragged K tail — per-element clamp-and-select on both K and
  //     M/N. Runs once (the [kMain,K) remainder), so the cost is off the hot
  //     path. Correct for full or ragged M/N (masks are no-ops when in-bounds).
  auto maskedPanel = [&](OpBuilder &bb, Location l, Value k0, ValueRange acc) {
    SmallVector<Value> aFrag(mt, loadZero), bFrag(nt, loadZero);
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
  // M/N edge (stores run once, so the guard cost is negligible). Bias is
  // invariant across all eight accumulator elements and every M tile for one
  // output column, so load it once per N tile and reuse it.
  auto emitStore = [&](OpBuilder &sb, ValueRange accs, bool masked) {
    for (int64_t ni = 0; ni < nt; ++ni) {
      Value biasValue = bias
          ? Value(sb.create<memref::LoadOp>(loc, bias,
                                            ValueRange{colSafe[ni]}))
          : Value();
      for (int64_t mi = 0; mi < mt; ++mi) {
        Value accF = accs[mi * nt + ni];
        for (int64_t e = 0; e < 8; ++e) {
          Value twoE = sb.create<arith::ConstantIndexOp>(loc, e * 2);
          Value rowOff = sb.create<arith::AddIOp>(loc, twoE, lhi);
          Value r = sb.create<arith::AddIOp>(loc, rowOrigin[mi], rowOff);
          Value dv =
              sb.create<vector::ExtractOp>(loc, accF, ArrayRef<int64_t>{e});
          if (biasValue)
            dv = sb.create<arith::AddFOp>(loc, dv, biasValue);
          if (!T.isInt)
            dv = tessera::tile::emitScalarFloatActivation(
                sb, loc, dv, activation);
          if (!T.isInt)
            dv = tessera::tile::emitFloatOutputConversion(
                sb, loc, dv, outputType);
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

  // Explicit ctors: the Option<bool> member is non-copyable, so the compiler-
  // deleted copy ctor would break PassWrapper's clonePass(). Re-base on the
  // PassWrapper copy ctor (MLIR copies option VALUES separately).
  GenerateWMMAGemmKernelPass() = default;
  GenerateWMMAGemmKernelPass(const GenerateWMMAGemmKernelPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "generate-wmma-gemm-kernel"; }
  StringRef getDescription() const final {
    return "Stage K/L: expand a tessera_rocm.wmma_gemm directive into a "
           "problem-size-generic, register-blocked (mt x nt), fragment-"
           "materialized RDNA WMMA GEMM gpu kernel (compiler-generated)";
  }

  // Fork A (pilot): emit the matrix op as tile.mma so the generated GEMM flows
  // through rocm-wave-lds-pipeline + lower-tile-to-rocm instead of emitting
  // tessera_rocm.wmma directly. Default false keeps the established direct lane.
  Option<bool> viaTile{*this, "via-tile",
                       llvm::cl::desc("emit tile.mma (route through the wave/LDS "
                                      "pipeline) instead of tessera_rocm.wmma"),
                       llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, vector::VectorDialect,
                    arith::ArithDialect, math::MathDialect,
                    memref::MemRefDialect,
                    tessera::tile::TesseraTileDialect,
                    mlir::tessera_rocm::TesseraROCMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<WmmaGemmRequest> requests;

    // Portable launch-level adapter. It validates the target-neutral contract
    // and directly populates the in-memory request consumed by the production
    // generator below. No temporary backend directive or marker is introduced.
    SmallVector<tessera::tile::MatmulKernelOp> portableKernels;
    module.walk([&](tessera::tile::MatmulKernelOp op) {
      portableKernels.push_back(op);
    });
    for (tessera::tile::MatmulKernelOp kernel : portableKernels) {
      Operation *op = kernel.getOperation();
      auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
      auto epilogue =
          op->getAttrOfType<tessera::tile::TileEpilogueAttr>("epilogue");
      bool common = desc && epilogue &&
          (desc.getFamily() == "auto" || desc.getFamily() == "wmma") &&
          desc.getM() == 16 && desc.getN() == 16 && desc.getK() == 16 &&
          desc.getAType() == desc.getBType() && desc.getALayout() == "row_major" &&
          desc.getBLayout() == "col_major" && desc.getKBlocks() == 1;
      bool floatContract = common &&
          (desc.getAType() == "f16" || desc.getAType() == "bf16") &&
          desc.getAccType() == "f32";
      bool integerContract = common &&
          (desc.getAType() == "int8" || desc.getAType() == "int4") &&
          (desc.getAccType() == "i32" || desc.getAccType() == "int32");
      bool canonical = floatContract || integerContract;
      if (!canonical) {
        op->emitError("ROCm tile.matmul_kernel requires an m16n16k16 row/col "
                      "WMMA descriptor: f16/bf16 with f32 accumulation or "
                      "int8/int4 with i32 accumulation");
        return signalPassFailure();
      }
      if (auto staging = op->getAttrOfType<StringAttr>("staging");
          staging && staging.getValue() != "global") {
        op->emitError("ROCm tile.matmul_kernel portable-fragment slice currently "
                      "requires global staging");
        return signalPassFailure();
      }
      auto parent = op->getParentOfType<func::FuncOp>();
      if (!parent) {
        op->emitError("ROCm tile.matmul_kernel must be nested in func.func");
        return signalPassFailure();
      }
      int64_t mt = 1, nt = 1;
      if (auto warps = op->getAttrOfType<IntegerAttr>("warps");
          warps && warps.getInt() == 4) {
        mt = 2;
        nt = 2;
      }
      WmmaGemmRequest request;
      request.anchor = op;
      request.eraseOwner = parent;
      request.name = parent.getSymName().str();
      request.mt = mt;
      request.nt = nt;
      request.dtype = desc.getAType().str();
      request.bias = epilogue.getBias();
      request.activation = epilogue.getActivation().str();
      request.output = epilogue.getOutputType().str();
      request.portableABI = true;
      requests.push_back(std::move(request));
    }

    // Backward-compatible legacy directive adapter.
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
      WmmaGemmRequest request;
      request.anchor = op;
      request.eraseOwner = op;
      request.name = nameAttr.getValue().str();
      request.m = mAttr.getInt();
      request.n = nAttr.getInt();
      request.k = kAttr.getInt();
      if (auto a = op->getAttrOfType<IntegerAttr>("mt"))
        request.mt = a.getInt();
      if (auto a = op->getAttrOfType<IntegerAttr>("nt"))
        request.nt = a.getInt();
      if (auto a = op->getAttrOfType<StringAttr>("dtype"))
        request.dtype = a.getValue().str();
      if (auto a = op->getAttrOfType<BoolAttr>("bias"))
        request.bias = a.getValue();
      if (auto a = op->getAttrOfType<StringAttr>("activation"))
        request.activation = a.getValue().str();
      if (auto a = op->getAttrOfType<StringAttr>("output"))
        request.output = a.getValue().str();
      request.storagePack =
          op->getAttrOfType<DictionaryAttr>("tessera.storage_pack");
      requests.push_back(std::move(request));
    }

    SmallVector<Operation *> generatedOwners;
    for (WmmaGemmRequest &request : requests) {
      Operation *op = request.anchor;
      if (request.m != 16 || request.n != 16 || request.k != 16) {
        op->emitError("generate-wmma-gemm-kernel: the WMMA instruction tile "
                      "must be 16x16x16 (got ")
            << request.m << "x" << request.n << "x" << request.k
            << "); RDNA V_WMMA exposes no other tile. The problem size is a "
               "runtime (M,N,K) kernel argument, not the tile";
        return signalPassFailure();
      }
      // mt/nt default to 1 (DefaultValuedAttr); the register-blocked macro-tile.
      int64_t mt = request.mt, nt = request.nt;
      if (mt < 1 || nt < 1) {
        op->emitError("generate-wmma-gemm-kernel: mt/nt (macro-tile in WMMA "
                      "tiles) must be >= 1 (got ")
            << mt << "x" << nt << ")";
        return signalPassFailure();
      }
      // The runtime and Target-IR now hand this pass one unified schedule
      // descriptor.  mt/nt are executable knobs; the remaining attributes are
      // validated evidence carried onto the kernel so profiler A/B output can
      // be joined back to the exact schedule that ran.
      if (auto arch = op->getAttrOfType<StringAttr>("schedule_arch")) {
        if (!arch.getValue().starts_with("gfx11")) {
          op->emitError("generate-wmma-gemm-kernel: schedule_arch must select "
                        "the gfx11 16x16x16 WMMA ABI; got ")
              << arch.getValue();
          return signalPassFailure();
        }
      }
      if (auto stages =
              op->getAttrOfType<IntegerAttr>("schedule_pipeline_stages")) {
        if (stages.getInt() < 1 || stages.getInt() > 4) {
          op->emitError("generate-wmma-gemm-kernel: schedule pipeline stages "
                        "must be in [1,4]");
          return signalPassFailure();
        }
      }
      if (auto layout =
              op->getAttrOfType<StringAttr>("schedule_lds_layout")) {
        if (layout.getValue() != "swizzle" && layout.getValue() != "padding") {
          op->emitError("generate-wmma-gemm-kernel: schedule LDS layout must "
                        "be swizzle or padding");
          return signalPassFailure();
        }
      }
      if (auto owner =
              op->getAttrOfType<StringAttr>("schedule_ownership")) {
        if (owner.getValue() != "wave") {
          op->emitError("generate-wmma-gemm-kernel: WMMA macro-tiles require "
                        "wave ownership");
          return signalPassFailure();
        }
      }

      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = request.name;

      // dtype: f16 (default) / bf16 (f32 accumulate), or int8 (i32 accumulate).
      // All confirmed on gfx1151. The fragment/accumulator types follow dtype:
      //   f16/bf16 : A/B vector<16x{f16,bf16}>, acc vector<8xf32>, D = f32.
      //   int8     : A/B 16 i8 loaded as vector<16xi8> then bitcast to the iu8
      //              ABI vector<4xi32>; acc vector<8xi32>, D = i32 (signed).
      Type f16Ty = b.getF16Type();
      Type bf16Ty = b.getBF16Type();
      Type i8Ty = b.getIntegerType(8);
      Type i32Ty = b.getIntegerType(32);
      Type f32Ty = b.getF32Type();
      WmmaTypes T;
      StringRef dt = request.dtype;
      bool portableContract = request.portableABI;
      // C4 reconciliation (2026-06-23): if the directive carries the
      // backend-neutral `tessera.storage_pack = {logical, container, factor}`
      // descriptor (from StoragePackConsume), its `logical` selects the dtype —
      // one packing contract feeds both AMD (here) and NVIDIA. Fall back to the
      // legacy `dtype` attr when no descriptor is present (non-breaking).
      DictionaryAttr packDesc = request.storagePack;
      if (packDesc) {
        if (auto logical = packDesc.getAs<StringAttr>("logical"))
          dt = logical.getValue();
      }
      auto v8i32 = VectorType::get({8}, i32Ty);
      auto v8f32 = VectorType::get({8}, f32Ty);
      auto v16i8 = VectorType::get({16}, i8Ty);
      if (dt == "f16" || dt == "float16") {
        T = {f16Ty, VectorType::get({16}, f16Ty), VectorType::get({16}, f16Ty),
             v8f32, f32Ty, /*isInt=*/false, /*pack=*/0, /*packFactor=*/1};
      } else if (dt == "bf16" || dt == "bfloat16") {
        T = {bf16Ty, VectorType::get({16}, bf16Ty),
             VectorType::get({16}, bf16Ty), v8f32, f32Ty, /*isInt=*/false,
             /*pack=*/0, /*packFactor=*/1};
      } else if (dt == "int8" || dt == "i8") {
        T = {i8Ty, v16i8, VectorType::get({4}, i32Ty), v8i32, i32Ty,
             /*isInt=*/true, /*pack=*/1, /*packFactor=*/1};
      } else if (dt == "int4" || dt == "i4") {
        // int4 values supplied in int8 containers (range [-8,7]); the low nibble
        // is the int4 two's-complement. Nibble-packed in-kernel to the iu4 ABI
        // vector<2xi32>; i32 accumulate. (correctness-first — no coalesced load.)
        T = {i8Ty, v16i8, VectorType::get({2}, i32Ty), v8i32, i32Ty,
             /*isInt=*/true, /*pack=*/2, /*packFactor=*/2};
      } else {
        op->emitError("generate-wmma-gemm-kernel: dtype must be f16, bf16, "
                      "int8, or int4 (got '")
            << dt << "')";
        return signalPassFailure();
      }

      // C4 reconciliation: the storage-pack `factor` (logical values per byte
      // container) must equal this dtype's `packFactor` — the single packing
      // contract. (Verify against `packFactor`, the logical contract, NOT
      // `pack`, the codegen ABI mode: today int8/int4 happen to share the value,
      // but a new int ABI mode at the same logical factor must still pass.)
      if (packDesc && T.isInt) {
        if (auto fAttr = packDesc.getAs<IntegerAttr>("factor")) {
          int64_t factor = fAttr.getInt();
          if (factor != T.packFactor) {
            op->emitError("DTYPE_PACK_FACTOR_MISMATCH: tessera.storage_pack "
                          "factor ")
                << factor << " disagrees with the dtype packing factor "
                << T.packFactor << " for dtype '" << dt << "'.";
            return signalPassFailure();
          }
        }
      }

      // Fused epilogue: optional per-column bias add + pointwise activation,
      // applied on the in-register f32 accumulator before the store. The
      // epilogue is float-only (gelu/silu are transcendentals; bias is an fadd):
      // an int8/int4 directive carrying it is a named error, not a silent no-op.
      bool hasBias = request.bias;
      StringRef activation = request.activation;
      if (!tessera::tile::isSupportedActivation(activation)) {
        op->emitError("generate-wmma-gemm-kernel: activation must be one of "
                      "none/relu/gelu/silu (got '")
            << activation << "')";
        return signalPassFailure();
      }
      if (T.isInt && (hasBias || activation != "none")) {
        op->emitError("generate-wmma-gemm-kernel: the fused epilogue "
                      "(bias/activation) is float-only; dtype '")
            << dt << "' is integer";
        return signalPassFailure();
      }
      Type outputTy = T.accElem;
      if (!request.output.empty()) {
        StringRef output = request.output;
        if (!T.isInt && output == "f16")
          outputTy = f16Ty;
        else if ((!T.isInt && output != "f32") ||
                 (T.isInt && output != "i32" && output != "int32")) {
          op->emitError("generate-wmma-gemm-kernel: output type is incompatible "
                        "with dtype '")
              << dt << "'";
          return signalPassFailure();
        }
      }

      // gpu.module @<name>_mod { gpu.func @<name>(A,B,D,M,N,K[,bias]) kernel }
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());

      Type idxTy = b.getIndexType();
      auto abTy = MemRefType::get({ShapedType::kDynamic}, T.store);
      auto dTy = MemRefType::get({ShapedType::kDynamic}, outputTy);
      auto biasTy = MemRefType::get({ShapedType::kDynamic}, T.accElem);
      SmallVector<Type> argTys{abTy, abTy};
      if (hasBias && portableContract)
        argTys.push_back(biasTy);
      argTys.append({dTy, idxTy, idxTy, idxTy});
      if (hasBias && !portableContract)
        argTys.push_back(biasTy); // legacy directive ABI: trailing bias
      auto fnTy = b.getFunctionType(argTys, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      for (StringRef attrName : {"schedule_arch", "schedule_pipeline_stages",
                                 "schedule_lds_layout", "schedule_ownership",
                                 "schedule_vgpr_estimate", "schedule_source"})
        if (Attribute attr = op->getAttr(attrName))
          gpuFunc->setAttr((Twine("tessera.rocm.") + attrName).str(), attr);

      OpBuilder bodyB(gpuFunc.getContext());
      emitGeneralBody(bodyB, loc, gpuFunc, mt, nt, T, outputTy,
                      portableContract, viaTile, hasBias, activation);

      if (!llvm::is_contained(generatedOwners, request.eraseOwner))
        generatedOwners.push_back(request.eraseOwner);
    }
    for (Operation *owner : generatedOwners)
      owner->erase();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateWMMAGemmKernelPass() {
  return std::make_unique<GenerateWMMAGemmKernelPass>();
}
