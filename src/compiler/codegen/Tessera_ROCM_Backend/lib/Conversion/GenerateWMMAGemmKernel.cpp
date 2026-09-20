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

#include <algorithm>
#include "TesseraROCM/Passes.h"
#include "ROCMPhysicalWMMAPanel.h"
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Dialect/Tile/TileEpilogue.h"
#include "TesseraROCMDialect.h.inc"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

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
  // gfx11 reduced-precision WMMA returns vector<16xf16>, with the selected
  // 16-bit half of each VGPR holding the eight logical accumulator elements.
  // `opsel=false` selects vector indices 0,2,...,14.
  bool halfAccumulator = false;
  // `pack` is the in-kernel codegen ABI mode (0 = fp passthrough, 1 = int8
  // bitcast->v4i32, 2 = int4 nibble-pack->v2i32) — an emission detail, NOT the
  // storage-pack contract. `packFactor` is the contract: logical values per
  // byte container (f16/bf16/int8 -> 1, int4 -> 2), exactly the `factor`
  // StoragePackConsume computes (container_bits / storage_bits). Keep them
  // separate so a future int ABI mode (a new `pack` at the same logical factor)
  // cannot silently break the contract check — verify against `packFactor`.
  int pack;
  int packFactor;
  // K elements ONE fragment consumes. 16 everywhere except RDNA4 int4, which
  // has a native double-K form (`V_WMMA_I32_16X16X32_IU4`). That matters
  // because a fragment load is 8 elements per lane whatever the storage, so
  // int4 moves only 32 of the 128 available bits at K=16; the double-K shape
  // fetches 16 elements and fills the interface (AMD's RDNA4 WMMA guide,
  // part 2). It is the instrument the K-unroll workaround stands in for.
  //
  // Only the K axis scales. The 16s that build the M/N macro tile are fragment
  // geometry and are untouched.
  int64_t fragK = 16;
  // ROCM-MACRO-K-TILE-1. The descriptor's `k_blocks`: how many fragment-K
  // steps one K BLOCK spans, so the block's contraction extent is
  // `fragK * kBlocks`. Distinct from `kUnroll`, which is a measured latency
  // knob: `kBlocks` is a contract the Schedule stated and a block scale must
  // align to exactly (the descriptor verifier enforces
  // `scale_k == k * k_blocks`), while `kUnroll` may be retuned freely without
  // changing what the program computes. They shape the loop the same way and
  // must not be conflated -- that is how a tuning parameter becomes a
  // semantic one.
  int64_t kBlocks = 1;
  // B's storage name when it differs from A's. RDNA4 has the mixed OCP FP8
  // pairs (`V_WMMA_F32_16X16X16_FP8_BF8` and its mirror), and empty means
  // "same as A", which is every other case.
  std::string bElem;
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
  bool canonicalKLoop = false;
  bool ssaOwnershipProof = false;
  bool raggedZeroPad = false;
  // Populated only by the canonical Schedule -> Tile consumer.  These are
  // static problem extents, never a substitute for dynamic leading dimensions.
  int64_t staticM = 0, staticN = 0, staticK = 0;
  int64_t logicalTileM = 0, logicalTileN = 0, logicalTileK = 0;
  std::string accumulate;
  std::string rasterOrder = "row_major";
  int64_t rasterGroup = 1;
  // ROCM-MACRO-K-TILE-1: the descriptor's `k_blocks`. Arrives here rather than
  // being read from `desc` at the emission site, because the descriptor is only
  // in scope in the adapters that populate this request.
  int64_t kBlocks = 1;
  tessera::tile::TilePackedFormatAttr storagePack;
};

static bool isForOp(Operation *op) {
  return op && op->getName().getStringRef() == "scf.for";
}

/// Recognize the shared CORE-GEMM-KLOOP semantic contract. This deliberately
/// validates the loop instead of merely looking for the inner marker: the K
/// loop must carry real pipeline state, the contraction must slice two values
/// defined outside the nest, and the outer result must be the function result.
/// ROCm then re-forms the semantic loop into its existing problem-size-generic
/// WMMA schedule; it does not build a second GEMM implementation.
static FailureOr<WmmaGemmRequest>
matchCanonicalGemmLoop(Operation *matmul) {
  StringRef opName = matmul->getName().getStringRef();
  bool sharedMatmul = opName == "tessera.matmul";
  bool loweredTileMma = opName == "tile.mma";
  if (!matmul->hasAttr("tessera.canonical_k_step") ||
      (!sharedMatmul && !loweredTileMma) || matmul->getNumResults() != 1 ||
      (sharedMatmul && matmul->getNumOperands() != 2) ||
      (loweredTileMma && matmul->getNumOperands() < 3))
    return failure();

  Operation *kLoop = matmul->getParentOp();
  Operation *nLoop = kLoop ? kLoop->getParentOp() : nullptr;
  Operation *mLoop = nLoop ? nLoop->getParentOp() : nullptr;
  if (!isForOp(kLoop) || !isForOp(nLoop) || !isForOp(mLoop))
    return failure();
  if (!llvm::any_of(kLoop->getResultTypes(), [](Type type) {
        return llvm::isa<tessera::tile::PipelineStateType>(type);
      }) ||
      mLoop->getNumResults() != 1)
    return failure();

  bool hasAsyncToken = false;
  bool hasBuffer = false;
  bool hasPipelineState = false;
  for (Value operand : matmul->getOperands()) {
    hasAsyncToken |=
        llvm::isa<tessera::tile::AsyncTokenType>(operand.getType());
    hasBuffer |= llvm::isa<tessera::tile::BufferType>(operand.getType());
    hasPipelineState |=
        llvm::isa<tessera::tile::PipelineStateType>(operand.getType());
  }
  if (loweredTileMma &&
      (!hasAsyncToken || !hasBuffer || !hasPipelineState))
    return failure();

  SmallVector<Value, 2> sources;
  for (Value operand : matmul->getOperands().take_front(2)) {
    if (loweredTileMma) {
      Operation *copy = operand.getDefiningOp();
      if (!copy || copy->getName().getStringRef() != "tile.async_copy" ||
          copy->getNumOperands() == 0 ||
          !llvm::any_of(copy->getOperands(), [](Value value) {
            return llvm::isa<tessera::tile::BufferType>(value.getType());
          }) ||
          !llvm::any_of(copy->getOperands(), [](Value value) {
            return llvm::isa<tessera::tile::PipelineStateType>(
                value.getType());
          }) ||
          !llvm::any_of(copy->getResults(), [](Value value) {
            return llvm::isa<tessera::tile::AsyncTokenType>(value.getType());
          }))
        return failure();
      operand = copy->getOperand(0);
    }
    Operation *slice = operand.getDefiningOp();
    if (!slice || slice->getName().getStringRef() != "tensor.extract_slice")
      return failure();
    Value source = slice->getOperand(0);
    Operation *sourceDef = source.getDefiningOp();
    if (sourceDef && mLoop->isProperAncestor(sourceDef))
      return failure();
    sources.push_back(source);
  }

  auto parent = matmul->getParentOfType<func::FuncOp>();
  if (!parent || parent.getNumArguments() != 2 ||
      parent.getFunctionType().getNumResults() != 1)
    return failure();
  Operation *terminator = parent.getBody().front().getTerminator();
  if (!terminator || terminator->getNumOperands() != 1)
    return failure();
  Value returned = terminator->getOperand(0);
  if (Operation *slice = returned.getDefiningOp();
      slice && slice->getName().getStringRef() == "tensor.extract_slice")
    returned = slice->getOperand(0);
  if (returned != mLoop->getResult(0))
    return failure();

  auto aType = llvm::dyn_cast<RankedTensorType>(sources[0].getType());
  auto bType = llvm::dyn_cast<RankedTensorType>(sources[1].getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(mLoop->getResult(0).getType());
  if (!aType || !bType || !resultType || aType.getRank() != 2 ||
      bType.getRank() != 2 || resultType.getRank() != 2 ||
      !aType.hasStaticShape() || !bType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      aType.getElementType() != bType.getElementType())
    return failure();

  WmmaGemmRequest request;
  request.anchor = matmul;
  request.eraseOwner = parent;
  request.name = parent.getSymName().str();
  request.portableABI = true;
  request.canonicalKLoop = true;
  request.ssaOwnershipProof = loweredTileMma;
  request.raggedZeroPad = matmul->hasAttr("tessera.ragged_zero_pad");
  auto tileM = matmul->getAttrOfType<IntegerAttr>("tessera.tile_m");
  auto tileN = matmul->getAttrOfType<IntegerAttr>("tessera.tile_n");
  auto tileK = matmul->getAttrOfType<IntegerAttr>("tessera.tile_k");
  if (!tileM || !tileN || !tileK || tileM.getInt() <= 0 ||
      tileN.getInt() <= 0 || tileK.getInt() <= 0 ||
      !request.raggedZeroPad)
    return failure();
  request.logicalTileM = tileM.getInt();
  request.logicalTileN = tileN.getInt();
  request.logicalTileK = tileK.getInt();
  Type storage = aType.getElementType();
  Type accum = resultType.getElementType();
  if (storage.isF16() && accum.isF32()) {
    request.dtype = "f16";
    request.output = "f32";
    request.accumulate = "f32";
  } else if (storage.isBF16() && accum.isF32()) {
    request.dtype = "bf16";
    request.output = "f32";
    request.accumulate = "f32";
  } else if (storage.isInteger(8) && accum.isInteger(32)) {
    request.dtype = "int8";
    request.output = "i32";
    request.accumulate = "i32";
  } else {
    return failure();
  }
  return request;
}

// Emit the problem-size-generic, register-blocked (mt x nt) WMMA GEMM body into
// `gpuFunc` (args: A, B, D : memref<?>, M, N, K : index), for the dtype in `T`.
void emitGeneralBody(OpBuilder &b, Location loc, gpu::GPUFuncOp gpuFunc,
                     int64_t mt, int64_t nt, const WmmaTypes &T,
                     Type outputType, bool portableABI = false,
                     bool viaTile = false, bool hasBias = false,
                     StringRef activation = "none",
                     bool packedInt4Memory = false,
                     StringRef rasterOrder = "row_major",
                     int64_t rasterGroup = 1, int64_t staticM = 0,
                     int64_t staticN = 0, int64_t staticK = 0,
                     int64_t kUnroll = 1, int64_t schedGroups = 0) {
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
  Value c2 = b.create<arith::ConstantIndexOp>(loc, 2);
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
    APFloat accZ =
        cast<FloatAttr>(b.getFloatAttr(T.accElem, 0.0)).getValue();
    accZero = b.create<arith::ConstantOp>(
        loc, accTy,
        DenseElementsAttr::get(cast<ShapedType>(accTy), accZ));
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
  Value tileM = bidY, tileN = bidX;
  if (rasterOrder != "row_major") {
    Value gridM = b.create<arith::DivUIOp>(
        loc, b.create<arith::AddIOp>(
                 loc, M, b.create<arith::ConstantIndexOp>(loc, 16 * mt - 1)),
        c16mt);
    Value gridN = b.create<arith::DivUIOp>(
        loc, b.create<arith::AddIOp>(
                 loc, N, b.create<arith::ConstantIndexOp>(loc, 16 * nt - 1)),
        c16nt);
    Value flat = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, bidY, gridN), bidX);
    if (rasterOrder == "column_major") {
      tileM = b.create<arith::RemUIOp>(loc, flat, gridM);
      tileN = b.create<arith::DivUIOp>(loc, flat, gridM);
    } else {
      bool groupM = rasterOrder == "grouped_m";
      Value gridMajor = groupM ? gridM : gridN;
      Value gridMinor = groupM ? gridN : gridM;
      Value group = b.create<arith::ConstantIndexOp>(loc, rasterGroup);
      Value perPanel = b.create<arith::MulIOp>(loc, group, gridMinor);
      Value panel = b.create<arith::DivUIOp>(loc, flat, perPanel);
      Value firstMajor = b.create<arith::MulIOp>(loc, panel, group);
      Value remaining = b.create<arith::SubIOp>(loc, gridMajor, firstMajor);
      Value shortPanel = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, remaining, group);
      Value panelRows =
          b.create<arith::SelectOp>(loc, shortPanel, remaining, group);
      Value within = b.create<arith::RemUIOp>(loc, flat, perPanel);
      Value major = b.create<arith::AddIOp>(
          loc, firstMajor,
          b.create<arith::RemUIOp>(loc, within, panelRows));
      Value minor = b.create<arith::DivUIOp>(loc, within, panelRows);
      tileM = groupM ? major : minor;
      tileN = groupM ? minor : major;
    }
  }
  Value baseRow = b.create<arith::MulIOp>(loc, tileM, c16mt);
  Value baseCol = b.create<arith::MulIOp>(loc, tileN, c16nt);

  // W1.1 step 3 pilot. The typed Tile chain must retain the runtime leading
  // dimensions of this problem-size-generic kernel, so leading_dim=0 means the
  // final tile.view/tile.store operand supplies K or N as SSA.
  MLIRContext *ctx = b.getContext();
  SmallVector<StringAttr> tileAxes{b.getStringAttr("tlane"),
                                   b.getStringAttr("reg")};
  // M and N are fragment geometry and stay 16. K follows the storage: RDNA4
  // int4 has a native double-K form, and the fragment type and the operand
  // view layouts are what carry that request to `resolveFragmentLayout`.
  const int64_t fragK = T.fragK;
  // Three DIFFERENT tiles shared one literal here, and they coincide only at
  // K = 16: the A view is {M, K}, the B view is {K, N}, and the accumulator
  // the epilogue unpacks and stores is {M, N}. `materializeFragmentPack`
  // checks the view's shard extents against {M, K} / {K, N} derived from the
  // descriptor, so at RDNA4's double-K int4 (K = 32) a shared {16, 16} is
  // wrong for both operands and right for the accumulator. Each is now built
  // from what it actually describes; every one of them is exactly {16, 16}
  // with stride {16, 1} again when fragK is 16, so the K = 16 route emits
  // byte-identical IR.
  auto layoutFor = [&](int64_t rows, int64_t cols) {
    return tessera::tile::TileLayoutAttr::get(
        ctx, {rows, cols}, {cols, 1}, tileAxes, {}, {}, {}, 0,
        tessera::tile::TileSwizzleAttr());
  };
  auto aTileLayout = layoutFor(16, fragK);    // {M, K}
  auto bTileLayout = layoutFor(fragK, 16);    // {K, N}
  auto accTileLayout = layoutFor(16, 16);     // {M, N}, independent of K
  auto dynamicRowMajor = tessera::tile::TileMemoryLayoutAttr::get(
      ctx, "gmem", "row_major", 0);
  auto tileValueTy = tessera::tile::TileValueType::get(ctx);
  StringRef fragmentElem = T.store.isF16()                 ? "f16"
                           : T.store.isBF16()                ? "bf16"
                           : isa<Float8E4M3FNType>(T.store)  ? "e4m3"
                           : isa<Float8E5M2Type>(T.store)    ? "e5m2"
                           : T.pack == 1                     ? "int8"
                                                             : "int4";
  StringRef fragmentAcc = T.isInt ? "i32" : "f32";
  // B may name a DIFFERENT storage from A. RDNA4 has the mixed OCP FP8 pairs
  // (`V_WMMA_F32_16X16X16_FP8_BF8` and its mirror) and
  // `resolveFragmentLayout` already selects them from the descriptor's two
  // types -- the fragment types are what carry the request down to it, so B
  // takes the descriptor's own `b` rather than inheriting A's. Both are 8-bit,
  // so nothing about the register format or the packing changes; only the
  // instruction the pair selects does.
  StringRef bFragmentElem = T.bElem.empty() ? fragmentElem : StringRef(T.bElem);
  auto aFragmentTy = tessera::tile::FragmentType::get(
      ctx, 16, 16, fragK, fragmentElem, fragmentAcc, "a", "row_major", "wmma");
  auto bFragmentTy = tessera::tile::FragmentType::get(
      ctx, 16, 16, fragK, bFragmentElem, fragmentAcc, "b", "col_major", "wmma");
  auto accFragmentTy = tessera::tile::FragmentType::get(
      ctx, 16, 16, fragK, fragmentAcc, fragmentAcc, "acc", "row_major", "wmma");

  // `layout` is the operand's own shard shape; passing it in is what keeps the
  // A and B views from silently sharing one.
  auto makeTileView = [&](OpBuilder &bb, Location l, Value base, Value row,
                          Value col, Value linearBase, Value rowBound,
                          Value colBound, Value leadingDim, bool bounded,
                          tessera::tile::TileLayoutAttr layout) -> Value {
    OperationState state(l, "tile.view");
    if (bounded)
      state.addOperands(
          {base, linearBase, row, col, rowBound, colBound, leadingDim});
    else
      state.addOperands({base, linearBase, row, col, leadingDim});
    state.addTypes(tileValueTy);
    state.addAttribute("tile.layout", layout);
    state.addAttribute("tile.memory", dynamicRowMajor);
    state.addAttribute("tile.linear_base", bb.getUnitAttr());
    return bb.create(state)->getResult(0);
  };
  // Materialize only the canonical static affine subset.  General tuple
  // composition, dynamic outer shapes, and dynamic leading dimensions remain
  // carrier-only until their target materializers own a proof boundary.
  auto makeStaticRowMajorLayout = [&](int64_t rows, int64_t cols) {
    auto i64 = [&](int64_t value) { return b.getI64IntegerAttr(value); };
    auto leaf = [&](int64_t value) -> Attribute { return i64(value); };
    auto singleton = [&](int64_t value) { return b.getArrayAttr({leaf(value)}); };
    auto basis = b.getArrayAttr({
        b.getArrayAttr({singleton(rows), singleton(1)}),
        b.getArrayAttr({singleton(cols), singleton(1)}),
    });
    return tessera::tile::TileComposedLayoutAttr::get(
        ctx, b.getArrayAttr({leaf(rows), leaf(cols)}),
        b.getArrayAttr({leaf(cols), leaf(1)}), basis, {0, 0});
  };
  const bool materializeComposedBases =
      viaTile && staticM > 0 && staticN > 0 && staticK > 0;
  auto aLayout = materializeComposedBases
                     ? makeStaticRowMajorLayout(staticM, staticK)
                     : tessera::tile::TileComposedLayoutAttr();
  auto bLayout = materializeComposedBases
                     ? makeStaticRowMajorLayout(staticK, staticN)
                     : tessera::tile::TileComposedLayoutAttr();
  auto materializeBase = [&](OpBuilder &bb, Location l,
                             tessera::tile::TileComposedLayoutAttr layout,
                             Value row, Value col) -> Value {
    Value row64 = bb.create<arith::IndexCastOp>(l, bb.getI64Type(), row);
    Value col64 = bb.create<arith::IndexCastOp>(l, bb.getI64Type(), col);
    OperationState state(l, "tile.materialize_composed_layout");
    state.addOperands({row64, col64});
    state.addTypes(bb.getI64Type());
    state.addAttribute("layout", layout);
    return bb.create(state)->getResult(0);
  };
  auto packFragment = [&](OpBuilder &bb, Location l, Value tile,
                          Type type) -> Value {
    OperationState state(l, "tile.fragment_pack");
    state.addOperands(tile);
    state.addTypes(type);
    return bb.create(state)->getResult(0);
  };

  // Per-tile (loop-invariant) values.
  //   arK[mi]     = row*K              — A-fragment base offset (fast path).
  //   arKsafe[mi] = clamp(row,0)*K     — same, OOB row clamped to 0 so the edge
  //                                      path can still issue a vector.load.
  //   colSafe[ni] = clamp(col,0)       — likewise for the B column.
  SmallVector<Value> arM(mt), arK(mt), arKsafe(mt), arInb(mt), rowOrigin(mt);
  SmallVector<Value> colN(nt), colSafe(nt), colInb(nt), colOrigin(nt);
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
    colOrigin[ni] = colBase;
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

  // First-class signed int4 memory is physically nibble packed: logical index
  // 2q occupies the low nibble and 2q+1 the high nibble of one i8 container.
  // Decode to the legacy vector<16xi8> staging form; `toFrag` then reconstructs
  // the exact IU4 register payload consumed by WMMA.
  auto loadLogical = [&](OpBuilder &bb, Location l, Value memref,
                         Value logicalIndex) -> Value {
    if (!packedInt4Memory)
      return bb.create<memref::LoadOp>(l, memref, ValueRange{logicalIndex});
    Value byteIndex = bb.create<arith::DivUIOp>(l, logicalIndex, c2);
    Value parity = bb.create<arith::RemUIOp>(l, logicalIndex, c2);
    Value raw8 = bb.create<memref::LoadOp>(l, memref, ValueRange{byteIndex});
    Value raw = bb.create<arith::ExtUIOp>(l, i32Ty, raw8);
    Value c4i = bb.create<arith::ConstantIntOp>(l, 4, 32);
    Value high = bb.create<arith::ShRUIOp>(l, raw, c4i);
    Value lowHalf = bb.create<arith::CmpIOp>(
        l, arith::CmpIPredicate::eq, parity, c0);
    Value selected = bb.create<arith::SelectOp>(l, lowHalf, raw, high);
    Value cF = bb.create<arith::ConstantIntOp>(l, 0xF, 32);
    Value nibble = bb.create<arith::AndIOp>(l, selected, cF);
    Value c8 = bb.create<arith::ConstantIntOp>(l, 8, 32);
    Value c16i = bb.create<arith::ConstantIntOp>(l, 16, 32);
    Value negative = bb.create<arith::CmpIOp>(
        l, arith::CmpIPredicate::sge, nibble, c8);
    Value signedValue = bb.create<arith::SelectOp>(
        l, negative, bb.create<arith::SubIOp>(l, nibble, c16i), nibble);
    return bb.create<arith::TruncIOp>(l, T.store, signedValue);
  };

  // WMMA accumulation over mt*nt fragments, reusing each loaded fragment.
  // ROCM-SCHED-GROUP-1. LLVM single-buffers LDS and drains before every WMMA
  // unless the pipeline is described to it, and until 2026-09-19 this backend
  // emitted no scheduling intrinsic at all -- so every recorded gfx1201 number
  // was taken against that drained default.
  //
  // `sched.group.barrier` describes the body as an alternating sequence: take
  // `size` instructions of `mask`, then `size` of the next, repeating. The
  // knob is GRANULARITY, not contents: coarser groups give the scheduler more
  // room to overlap, up to the point where the pattern asks for more
  // outstanding loads than the hardware can hold, past which it spills. That
  // is why this is a measured axis and not a fixed pattern -- and why 0
  // (emit nothing, keep the default schedule) stays the default until the
  // measurement says otherwise.
  auto emitSchedGroups = [&](OpBuilder &bb, Location l, int64_t loads,
                             int64_t mmas) {
    if (schedGroups <= 0 || loads <= 0 || mmas <= 0)
      return;
    const int64_t groups = std::min<int64_t>(schedGroups, std::min(loads, mmas));
    const int64_t perLoad = (loads + groups - 1) / groups;
    const int64_t perMma = (mmas + groups - 1) / groups;
    auto i32 = bb.getI32Type();
    for (int64_t g = 0; g < groups; ++g) {
      bb.create<ROCDL::SchedGroupBarrier>(
          l, ROCDL::SchedGroupMaskAttr::get(bb.getContext(),
                                            ROCDL::SchedGroupMask::vmem_read),
          IntegerAttr::get(i32, perLoad), IntegerAttr::get(i32, 0));
      bb.create<ROCDL::SchedGroupBarrier>(
          l, ROCDL::SchedGroupMaskAttr::get(bb.getContext(),
                                            ROCDL::SchedGroupMask::mfma_wmma),
          IntegerAttr::get(i32, perMma), IntegerAttr::get(i32, 0));
    }
  };

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
        // The typed path returns from each panel before reaching this helper.
        // This is the established direct, generator-owned vector lane.
        OperationState wmma(l, "tessera_rocm.wmma");
        wmma.addOperands({af[mi], bf[ni], acc[mi * nt + ni]});
        wmma.addTypes({accTy});
        next[mi * nt + ni] = bb.create(wmma)->getResult(0);
      }
    emitSchedGroups(bb, l, mt + nt, mt * nt);
    return next;
  };

  // The typed producer path owns logical origins and bounds only. ROCm's
  // fragment_pack lowering owns lane mapping, masking, strided-B gathering,
  // register packing, and the physical WMMA ABI.
  auto typedWmmaAll = [&](OpBuilder &bb, Location l, Value k0,
                          ValueRange acc, bool bounded) {
    SmallVector<Value> af(mt), bf(nt);
    for (int64_t mi = 0; mi < mt; ++mi) {
      Value linearBase = materializeComposedBases
                             ? materializeBase(bb, l, aLayout, arM[mi], k0)
                             : bb.create<arith::AddIOp>(l, arK[mi], k0);
      Value view =
          makeTileView(bb, l, A, rowOrigin[mi], k0, linearBase, M, K, K,
                       bounded, aTileLayout);
      af[mi] = packFragment(bb, l, view, aFragmentTy);
    }
    for (int64_t ni = 0; ni < nt; ++ni) {
      Value linearBase = materializeComposedBases
                             ? materializeBase(bb, l, bLayout, k0, colN[ni])
                             : *tessera::tile::materializeLinearIndex(
                                   bb, l, k0, colN[ni], N, "row_major");
      Value view =
          makeTileView(bb, l, B, k0, colOrigin[ni], linearBase, K, N, N,
                       bounded, bTileLayout);
      bf[ni] = packFragment(bb, l, view, bFragmentTy);
    }
    SmallVector<Value> next(mt * nt);
    for (int64_t mi = 0; mi < mt; ++mi)
      for (int64_t ni = 0; ni < nt; ++ni) {
        OperationState mma(l, "tile.mma");
        mma.addOperands({af[mi], bf[ni], acc[mi * nt + ni]});
        mma.addTypes(accFragmentTy);
        next[mi * nt + ni] = bb.create(mma)->getResult(0);
      }
    // The typed route is the canonical gfx1201 selection, so it is the one
    // that matters here; `tile.mma` still lowers to the WMMA the mask names.
    emitSchedGroups(bb, l, mt + nt, mt * nt);
    return next;
  };

  // --- fast panel: interior tile, full K panel — no masking. ---
  auto fastPanel = [&](OpBuilder &bb, Location l, Value k0, ValueRange acc) {
    if (viaTile)
      return typedWmmaAll(bb, l, k0, acc, /*bounded=*/false);
    if (T.pack == 0)
      return tessera_rocm::emitGfx11WmmaPhysicalPanel(
          bb, l, A, B, N, k0, arK, colN, acc, loadZero,
          cast<VectorType>(loadTy), cast<VectorType>(accTy));
    SmallVector<Value> aFrag(mt), bFrag(nt, loadZero);
    for (int64_t mi = 0; mi < mt; ++mi) {
      Value base = bb.create<arith::AddIOp>(l, arK[mi], k0);
      if (!packedInt4Memory) {
        aFrag[mi] = bb.create<vector::LoadOp>(l, loadTy, A, ValueRange{base});
      } else {
        aFrag[mi] = loadZero;
        for (int64_t i = 0; i < 16; ++i) {
          Value ci = bb.create<arith::ConstantIndexOp>(l, i);
          Value logical = bb.create<arith::AddIOp>(l, base, ci);
          Value v = loadLogical(bb, l, A, logical);
          aFrag[mi] = bb.create<vector::InsertOp>(
              l, v, aFrag[mi], ArrayRef<int64_t>{i});
        }
      }
    }
    for (int64_t i = 0; i < 16; ++i) {
      Value ci = bb.create<arith::ConstantIndexOp>(l, i);
      Value ak = bb.create<arith::AddIOp>(l, k0, ci);
      for (int64_t ni = 0; ni < nt; ++ni) {
        Value lin = *tessera::tile::materializeLinearIndex(
            bb, l, ak, colN[ni], N, "row_major");
        Value v = loadLogical(bb, l, B, lin);
        bFrag[ni] =
            bb.create<vector::InsertOp>(l, v, bFrag[ni], ArrayRef<int64_t>{i});
      }
    }
    return wmmaAll(bb, l, aFrag, bFrag, acc);
  };

  // --- edge panel: full K panel, ragged M/N — coalesced loads at a clamped
  //     row/col, then one vector select zeroes an OOB fragment. ---
  auto edgePanel = [&](OpBuilder &bb, Location l, Value k0, ValueRange acc) {
    if (viaTile)
      return typedWmmaAll(bb, l, k0, acc, /*bounded=*/true);
    SmallVector<Value> aFrag(mt), bFrag(nt, loadZero);
    for (int64_t mi = 0; mi < mt; ++mi) {
      Value base = bb.create<arith::AddIOp>(l, arKsafe[mi], k0);
      if (!packedInt4Memory) {
        Value v = bb.create<vector::LoadOp>(l, loadTy, A, ValueRange{base});
        aFrag[mi] = bb.create<arith::SelectOp>(l, arInb[mi], v, loadZero);
      } else {
        aFrag[mi] = loadZero;
        for (int64_t i = 0; i < 16; ++i) {
          Value ci = bb.create<arith::ConstantIndexOp>(l, i);
          Value logical = bb.create<arith::AddIOp>(l, base, ci);
          Value v = loadLogical(bb, l, A, logical);
          Value vm = bb.create<arith::SelectOp>(l, arInb[mi], v, storeZero);
          aFrag[mi] = bb.create<vector::InsertOp>(
              l, vm, aFrag[mi], ArrayRef<int64_t>{i});
        }
      }
    }
    for (int64_t i = 0; i < 16; ++i) {
      Value ci = bb.create<arith::ConstantIndexOp>(l, i);
      Value ak = bb.create<arith::AddIOp>(l, k0, ci);
      for (int64_t ni = 0; ni < nt; ++ni) {
        Value lin = *tessera::tile::materializeLinearIndex(
            bb, l, ak, colSafe[ni], N, "row_major");
        Value v = loadLogical(bb, l, B, lin);
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
    if (viaTile)
      return typedWmmaAll(bb, l, k0, acc, /*bounded=*/true);
    SmallVector<Value> aFrag(mt, loadZero), bFrag(nt, loadZero);
    for (int64_t i = 0; i < 16; ++i) {
      Value ci = bb.create<arith::ConstantIndexOp>(l, i);
      Value ak = bb.create<arith::AddIOp>(l, k0, ci);
      Value akInb = bb.create<arith::CmpIOp>(l, slt, ak, K);
      for (int64_t mi = 0; mi < mt; ++mi) {
        Value inb = bb.create<arith::AndIOp>(l, arInb[mi], akInb);
        Value lin = *tessera::tile::materializeLinearIndex(
            bb, l, arM[mi], ak, K, "row_major");
        Value safe = bb.create<arith::SelectOp>(l, inb, lin, c0);
        Value v = loadLogical(bb, l, A, safe);
        Value vm = bb.create<arith::SelectOp>(l, inb, v, storeZero);
        aFrag[mi] =
            bb.create<vector::InsertOp>(l, vm, aFrag[mi], ArrayRef<int64_t>{i});
      }
      for (int64_t ni = 0; ni < nt; ++ni) {
        Value inb = bb.create<arith::AndIOp>(l, akInb, colInb[ni]);
        Value lin = *tessera::tile::materializeLinearIndex(
            bb, l, ak, colN[ni], N, "row_major");
        Value safe = bb.create<arith::SelectOp>(l, inb, lin, c0);
        Value v = loadLogical(bb, l, B, safe);
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
  // The typed path hands the fused epilogue to the architecture consumer on
  // the store (`tile.epilogue` + trailing bias operand): TileToROCM resolves
  // each element's row/column per fragment family (gfx11, RDNA4, CDNA) and
  // applies the bias add and activation there, so one epilogue implementation
  // is correct on every layout. The untyped body below keeps its own gfx11
  // element loop.
  const bool typedEpilogue = viaTile && (hasBias || activation != "none");
  Attribute typedEpilogueAttr;
  if (typedEpilogue) {
    StringRef outputName = outputType.isF16()   ? "f16"
                           : outputType.isBF16() ? "bf16"
                           : outputType.isF32()  ? "f32"
                                                 : "i32";
    typedEpilogueAttr = tessera::tile::TileEpilogueAttr::get(
        b.getContext(), hasBias, activation, outputName);
  }
  auto emitStore = [&](OpBuilder &sb, ValueRange accs, bool masked) {
    if (viaTile) {
      for (int64_t ni = 0; ni < nt; ++ni)
        for (int64_t mi = 0; mi < mt; ++mi) {
          OperationState unpack(loc, "tile.fragment_unpack");
          unpack.addOperands(accs[mi * nt + ni]);
          unpack.addTypes(tileValueTy);
          unpack.addAttribute("tile.layout", accTileLayout);
          Value tile = sb.create(unpack)->getResult(0);
          OperationState store(loc, "tile.store");
          if (masked)
            store.addOperands(
                {tile, D, rowOrigin[mi], colOrigin[ni], M, N, N});
          else
            store.addOperands({tile, D, rowOrigin[mi], colOrigin[ni], N});
          if (typedEpilogue) {
            if (hasBias)
              store.addOperands({bias});
            store.addAttribute("tile.epilogue", typedEpilogueAttr);
          }
          store.addAttribute("tile.layout", accTileLayout);
          store.addAttribute("tile.memory", dynamicRowMajor);
          sb.create(store);
        }
      return;
    }
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
          int64_t accumulatorIndex = T.halfAccumulator ? 2 * e : e;
          Value dv = sb.create<vector::ExtractOp>(
              loc, accF, ArrayRef<int64_t>{accumulatorIndex});
          if (biasValue)
            dv = sb.create<arith::AddFOp>(loc, dv, biasValue);
          if (!T.isInt)
            dv = tessera::tile::emitScalarFloatActivation(
                sb, loc, dv, activation);
          if (!T.isInt)
            dv = tessera::tile::emitFloatOutputConversion(
                sb, loc, dv, outputType);
          Value didx = *tessera::tile::materializeLinearIndex(
              sb, loc, r, colN[ni], N, "row_major");
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

  SmallVector<Value> initAccs;
  if (viaTile) {
    initAccs.reserve(mt * nt);
    for (int64_t i = 0; i < mt * nt; ++i) {
      OperationState zero(loc, "tile.fragment_zero");
      zero.addTypes(accFragmentTy);
      initAccs.push_back(b.create(zero)->getResult(0));
    }
  } else {
    initAccs.assign(mt * nt, accZero);
  }

  // kMain = largest multiple of 16 <= K; the tail panel covers [kMain, K).
  // K-width constants follow the fragment, not the 16 that builds the M/N
  // macro tile. RDNA4 int4 consumes 32 K elements per fragment.
  Value cFragK = b.create<arith::ConstantIndexOp>(loc, T.fragK);
  Value kRem = b.create<arith::RemUIOp>(loc, K, cFragK);
  Value kMain = b.create<arith::SubIOp>(loc, K, kRem);
  Value needTail =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, kRem, c0);

  // Run the aligned main K-loop with `mainPanel`, fold the ragged-K tail (when
  // present) through `maskedPanel`, and store (masked iff ragged M/N).
  auto runPath = [&](OpBuilder &rb, function_ref<SmallVector<Value>(
                                        OpBuilder &, Location, Value, ValueRange)>
                                        mainPanel,
                     bool masked) {
    // K unrolling (2026-09-18): issue `kUnroll` full slabs per iteration, so
    // this iteration's fragment loads for slab j+1 are in flight while the
    // MMAs of slab j retire. It costs no extra registers beyond the extra
    // fragments and needs no barriers -- on gfx1201 the register body is
    // memory-latency bound (LDS staging measured 0.39-0.65x, and the panel
    // axis tops out at 4x4 fragments before the VGPR cliff), so this is the
    // remaining lever. `kUnroll = 1` is exactly the established loop.
    // Panels issued per loop iteration = the descriptor's K block times the
    // unroll. With `kBlocks == 1` this is exactly the established loop, so the
    // generalisation is inert until a schedule states a wider block.
    const int64_t blocks = std::max<int64_t>(T.kBlocks, 1);
    const int64_t unroll =
        masked ? 1 : std::max<int64_t>(kUnroll, 1) * blocks;
    Value kStep = rb.create<arith::ConstantIndexOp>(loc, T.fragK * unroll);
    Value kMainU = kMain;
    if (unroll > 1) {
      Value remU = rb.create<arith::RemUIOp>(loc, kMain, kStep);
      kMainU = rb.create<arith::SubIOp>(loc, kMain, remU);
    }
    auto kLoop = rb.create<scf::ForOp>(
        loc, c0, kMainU, kStep, initAccs,
        [&](OpBuilder &bb, Location l, Value k0, ValueRange iter) {
          SmallVector<Value> accs(iter.begin(), iter.end());
          for (int64_t u = 0; u < unroll; ++u) {
            Value ku = u == 0 ? k0
                              : bb.create<arith::AddIOp>(
                                    l, k0,
                                    bb.create<arith::ConstantIndexOp>(l, T.fragK * u));
            accs = mainPanel(bb, l, ku, accs);
          }
          bb.create<scf::YieldOp>(l, accs);
        });
    // The 1..unroll-1 full slabs the unrolled loop could not take.
    scf::ForOp remainder;
    if (unroll > 1) {
      remainder = rb.create<scf::ForOp>(
          loc, kMainU, kMain, cFragK, kLoop.getResults(),
          [&](OpBuilder &bb, Location l, Value k0, ValueRange iter) {
            bb.create<scf::YieldOp>(l, mainPanel(bb, l, k0, iter));
          });
    }
    ValueRange mainResults =
        unroll > 1 ? ValueRange(remainder.getResults()) : ValueRange(kLoop.getResults());
    auto tail = rb.create<scf::IfOp>(
        loc, needTail,
        [&](OpBuilder &tb, Location l) {
          tb.create<scf::YieldOp>(l, maskedPanel(tb, l, kMain, mainResults));
        },
        [&](OpBuilder &eb, Location l) {
          eb.create<scf::YieldOp>(l, mainResults);
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

// The LDS-staged TYPED body (typed-route gap, 2026-09-18). WM x WN waves per
// workgroup, each owning an mt x nt panel of 16x16 fragments, so the block
// tile is (WM*mt*16) x (WN*nt*16). Every 16-wide K slab is staged
// cooperatively: A as [WG_M][16] (K contiguous per row) and B TRANSPOSED as
// [WG_N][16] (K contiguous per column), so both fragment packs are contiguous
// vector loads from LDS rather than the hand-written kernels' 16 strided
// scalar gathers per B fragment (the reason those lose to the register panel
// in the gap packets). Zero-fill past every edge makes the K tail and the
// ragged M/N edges free at the loads; the stores are masked. The typed
// fragment ops are the same ones the register body emits; only the source
// views name the `lds` space, which the Tile->ROCm materializer admits.
// A measured variant: selected nowhere until the recorder says where it wins.
void emitTypedLdsBody(OpBuilder &b, Location loc, gpu::GPUFuncOp gpuFunc,
                      int64_t mt, int64_t nt, int64_t wavesM, int64_t wavesN,
                      const WmmaTypes &T, Type outputType, bool hasBias,
                      StringRef activation, StringRef rasterOrder,
                      int64_t rasterGroup, int64_t ldsPadDwords,
                      int64_t ldsCopyWidth, bool ldsCopyElide,
                      int64_t ldsCopyDepth, bool ldsDoubleBuffer,
                      int64_t ldsSchedValuPerMma, bool ldsBRowMajor) {
  MLIRContext *ctx = b.getContext();
  const int64_t wgM = wavesM * mt * 16, wgN = wavesN * nt * 16;
  const int64_t nthreads = wavesM * wavesN * 32;
  // ROCM-LDS-BANKPAD-1. Both tiles are stored with a 16-element row (A by row,
  // B transposed by column), and LDS banks are 32 x 4 B. An f16 row is 32 B =
  // 8 dwords, so sixteen lanes reading one element per row land on banks
  // {0,8,16,24}: four banks, a 4-WAY CONFLICT on every fragment read. The fix
  // is to make the row stride an ODD number of dwords, after which (stride*L)
  // mod 32 is a bijection over the lanes because gcd(odd, 32) = 1.
  //
  // One dword of padding does it for every storage width: 8->9 dwords for f16,
  // 4->5 for fp8/int8, 2->3 for int4. Expressed in ELEMENTS that is
  // 32/bitwidth -- 2 f16, 4 fp8, 8 int4 -- so the knob counts dwords and the
  // element count follows the storage.
  const int64_t bits = T.store.getIntOrFloatBitWidth();
  const int64_t padElems = ldsPadDwords * 32 / bits;
  const int64_t ldsStride = 16 + padElems;
  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  // Double-buffering needs two slabs live at once: one being read by the MMA
  // chain, one being written by the staging copy for the next K step.
  const int64_t nbuf = ldsDoubleBuffer ? 2 : 1;
  Value ldsA = gpuFunc.addWorkgroupAttribution(
      MemRefType::get({nbuf * wgM * ldsStride}, T.store,
                      MemRefLayoutAttrInterface(), ws),
      loc);
  // Row-major B is [K=16][wgN] and needs no padding: the strided column read
  // that the padding existed for (10e) is exactly what it removes.
  const int64_t ldsBElems = ldsBRowMajor ? 16 * wgN : wgN * ldsStride;
  Value ldsB = gpuFunc.addWorkgroupAttribution(
      MemRefType::get({nbuf * ldsBElems}, T.store,
                      MemRefLayoutAttrInterface(), ws),
      loc);
  // `known_block_size` is an INHERENT property of gpu.func: set it through the
  // accessor, never by raw name. A raw `setAttr` leaves the op holding the
  // attribute twice, which the NDEBUG build prints and the assertions build
  // aborts on ("DictionaryAttr element names must be unique") -- the exact
  // defect 72 ROCm generators carried for `gpu.kernel` until 2026-09-17.
  gpuFunc.setKnownBlockSize(
      ArrayRef<int32_t>{int32_t(nthreads), 1, 1});
  gpuFunc->setAttr("tessera.rocm.lds_bytes",
                   b.getI64IntegerAttr((wgM + wgN) * ldsStride * bits / 8));
  gpuFunc->setAttr("tessera.rocm.lds_pad_dwords",
                   b.getI64IntegerAttr(ldsPadDwords));
  gpuFunc->setAttr("tessera.rocm.lds_waves",
                   b.getDenseI64ArrayAttr({wavesM, wavesN}));

  b.setInsertionPointToStart(&gpuFunc.getBody().front());
  Value A = gpuFunc.getArgument(0);
  Value B = gpuFunc.getArgument(1);
  unsigned dIndex = hasBias ? 3 : 2;
  Value D = gpuFunc.getArgument(dIndex);
  Value M = gpuFunc.getArgument(dIndex + 1);
  Value N = gpuFunc.getArgument(dIndex + 2);
  Value K = gpuFunc.getArgument(dIndex + 3);
  Value bias = hasBias ? gpuFunc.getArgument(2) : Value();
  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto slt = arith::CmpIPredicate::slt;
  Value c0 = ci(0), c16 = ci(16), c15 = ci(15), c32 = ci(32);
  Value cWavesN = ci(wavesN), cThreads = ci(nthreads);
  Value cWgM = ci(wgM), cWgN = ci(wgN), cWgM16 = ci(wgM * 16),
        cWgN16 = ci(wgN * 16);
  // The LDS row stride, padded to an odd dword count (see above). The copy
  // loops still walk wgM*16 / wgN*16 logical elements; only the destination
  // and the fragment view's leading dimension carry the padding.
  Value cLdsStride = ci(ldsStride);

  // ROCM-LDS-STAGE-VECTOR-1. The staging copy moved SIXTEEN BITS per thread per
  // iteration -- `global_load_d16_b16` in, `ds_store_b16` out -- against the
  // register body's `global_load_b128`. Measured 8.8 against 70.7 TFLOP/s at
  // 1024^3 f16, and that copy is the 8x, not the bank conflict the padding
  // already fixed (which was worth ~10%).
  //
  // The width is DERIVED from the padded stride, not chosen: a V-wide vector
  // needs every LDS row start V-aligned, so V must divide `ldsStride`. That
  // couples this to ROCM-LDS-BANKPAD-1 exactly as predicted there --
  //
  //   pad=0 stride 16: V=8 but FOUR banks (the conflict the padding removed)
  //   pad=1 stride 18: conflict-free, V=2   <- optimal when copies were scalar
  //   pad=2 stride 20: conflict-free, V=4   <- the only wide conflict-free pair
  //   pad=4 stride 24: V=8 but eight banks
  //
  // so the padding that was right for a scalar copy is nearly the worst for a
  // vectorised one. The default moves with the measurement, not with this
  // comment.
  // `ldsCopyWidth` 0 derives the width; 1 forces the historical scalar copy.
  // The forced value exists so the vectorisation can be MEASURED against a
  // scalar baseline at the same grid -- without it the only scalar figures
  // available came from a harness that launched 4x too many workgroups, and
  // the vectorisation's benefit was unmeasurable (section 10i).
  int64_t vecW = 1;
  if (ldsCopyWidth <= 0) {
    for (int64_t v : {8, 4, 2})
      if (ldsStride % v == 0 && 16 % v == 0) { vecW = v; break; }
  } else {
    vecW = ldsCopyWidth;
    if (ldsStride % vecW != 0 || 16 % vecW != 0)
      vecW = 1;  // an unaligned forced width would scatter across rows
  }
  auto vecTy = VectorType::get({vecW}, T.store);

  Value scalarZero =
      T.isInt ? b.create<arith::ConstantOp>(loc, T.store,
                                            b.getIntegerAttr(T.store, 0))
              : b.create<arith::ConstantOp>(loc, T.store,
                                            b.getFloatAttr(T.store, 0.0));

  Value tx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value waveId = b.create<arith::DivUIOp>(loc, tx, c32);
  Value waveRow = b.create<arith::DivUIOp>(loc, waveId, cWavesN);
  Value waveCol = b.create<arith::RemUIOp>(loc, waveId, cWavesN);

  // Block-tile origin with the shared raster contract (row-major identity,
  // or the column-major / grouped remaps the register body also honours).
  Value bidX = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bidY = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
  Value tileM = bidY, tileN = bidX;
  if (rasterOrder != "row_major") {
    Value gridM = b.create<arith::DivUIOp>(
        loc, b.create<arith::AddIOp>(loc, M, ci(wgM - 1)), cWgM);
    Value gridN = b.create<arith::DivUIOp>(
        loc, b.create<arith::AddIOp>(loc, N, ci(wgN - 1)), cWgN);
    Value flat = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, bidY, gridN), bidX);
    if (rasterOrder == "column_major") {
      tileM = b.create<arith::RemUIOp>(loc, flat, gridM);
      tileN = b.create<arith::DivUIOp>(loc, flat, gridM);
    } else {
      bool groupM = rasterOrder == "grouped_m";
      Value gridMajor = groupM ? gridM : gridN;
      Value gridMinor = groupM ? gridN : gridM;
      Value group = ci(rasterGroup);
      Value perPanel = b.create<arith::MulIOp>(loc, group, gridMinor);
      Value panel = b.create<arith::DivUIOp>(loc, flat, perPanel);
      Value firstMajor = b.create<arith::MulIOp>(loc, panel, group);
      Value remaining = b.create<arith::SubIOp>(loc, gridMajor, firstMajor);
      Value shortPanel = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, remaining, group);
      Value panelRows =
          b.create<arith::SelectOp>(loc, shortPanel, remaining, group);
      Value within = b.create<arith::RemUIOp>(loc, flat, perPanel);
      Value major = b.create<arith::AddIOp>(
          loc, firstMajor, b.create<arith::RemUIOp>(loc, within, panelRows));
      Value minor = b.create<arith::DivUIOp>(loc, within, panelRows);
      tileM = groupM ? major : minor;
      tileN = groupM ? minor : major;
    }
  }
  Value baseRow = b.create<arith::MulIOp>(loc, tileM, cWgM);
  Value baseCol = b.create<arith::MulIOp>(loc, tileN, cWgN);
  Value waveRowOff = b.create<arith::MulIOp>(loc, waveRow, ci(mt * 16));
  Value waveColOff = b.create<arith::MulIOp>(loc, waveCol, ci(nt * 16));

  // Per-fragment origins: global (for the store) and LDS-local (for the packs).
  SmallVector<Value> rowOrigin(mt), lrow(mt), colOrigin(nt), lcol(nt);
  for (int64_t mi = 0; mi < mt; ++mi) {
    lrow[mi] = b.create<arith::AddIOp>(loc, waveRowOff, ci(mi * 16));
    rowOrigin[mi] = b.create<arith::AddIOp>(loc, baseRow, lrow[mi]);
  }
  for (int64_t ni = 0; ni < nt; ++ni) {
    lcol[ni] = b.create<arith::AddIOp>(loc, waveColOff, ci(ni * 16));
    colOrigin[ni] = b.create<arith::AddIOp>(loc, baseCol, lcol[ni]);
  }

  SmallVector<StringAttr> tileAxes{b.getStringAttr("tlane"),
                                   b.getStringAttr("reg")};
  auto tileLayout = tessera::tile::TileLayoutAttr::get(
      ctx, {16, 16}, {16, 1}, tileAxes, {}, {}, {}, 0,
      tessera::tile::TileSwizzleAttr());
  auto gmemRowMajor =
      tessera::tile::TileMemoryLayoutAttr::get(ctx, "gmem", "row_major", 0);
  auto ldsRowMajor =
      tessera::tile::TileMemoryLayoutAttr::get(ctx, "lds", "row_major", 0);
  auto ldsColMajor =
      tessera::tile::TileMemoryLayoutAttr::get(ctx, "lds", "col_major", 0);
  auto tileValueTy = tessera::tile::TileValueType::get(ctx);
  StringRef fragmentElem = T.store.isF16()                 ? "f16"
                           : T.store.isBF16()                ? "bf16"
                           : isa<Float8E4M3FNType>(T.store)  ? "e4m3"
                           : isa<Float8E5M2Type>(T.store)    ? "e5m2"
                           : T.pack == 1                     ? "int8"
                                                             : "int4";
  StringRef fragmentAcc = T.isInt ? "i32" : "f32";
  auto aFragmentTy = tessera::tile::FragmentType::get(
      ctx, 16, 16, 16, fragmentElem, fragmentAcc, "a", "row_major", "wmma");
  auto bFragmentTy = tessera::tile::FragmentType::get(
      ctx, 16, 16, 16, fragmentElem, fragmentAcc, "b", "col_major", "wmma");
  auto accFragmentTy = tessera::tile::FragmentType::get(
      ctx, 16, 16, 16, fragmentAcc, fragmentAcc, "acc", "row_major", "wmma");
  auto ldsViewStrided = [&](OpBuilder &bb, Location l, Value base, Value row,
                            Value col, Value stride,
                            Attribute memory) -> Value {
    OperationState state(l, "tile.view");
    state.addOperands({base, row, col, stride});
    state.addTypes(tileValueTy);
    state.addAttribute("tile.layout", tileLayout);
    state.addAttribute("tile.memory", memory);
    return bb.create(state)->getResult(0);
  };
  auto ldsView = [&](OpBuilder &bb, Location l, Value base, Value row,
                     Value col, Attribute memory) -> Value {
    return ldsViewStrided(bb, l, base, row, col, cLdsStride, memory);
  };
  auto packFragment = [&](OpBuilder &bb, Location l, Value view, Type type,
                          bool transpose = false) {
    OperationState state(l, "tile.fragment_pack");
    state.addOperands(view);
    state.addTypes(type);
    // Semantic key (Decision #21a): the source arrives in the opposite major
    // order and the lowering owes the transpose. Never a hint.
    if (transpose)
      state.addAttribute("transpose", bb.getUnitAttr());
    return bb.create(state)->getResult(0);
  };

  SmallVector<Value> initAccs;
  for (int64_t i = 0; i < mt * nt; ++i) {
    OperationState zero(loc, "tile.fragment_zero");
    zero.addTypes(accFragmentTy);
    initAccs.push_back(b.create(zero)->getResult(0));
  }

  // kEnd = K rounded up to a multiple of 16: the last slab zero-fills past K.
  Value kEnd = b.create<arith::MulIOp>(
      loc,
      b.create<arith::DivUIOp>(loc, b.create<arith::AddIOp>(loc, K, c15), c16),
      c16);
  // ------------------------------------------------------------------
  // Staging, factored so it can be called three ways: as a whole copy (the
  // single-buffered loop and the double-buffered prologue), or split into an
  // ISSUE phase and a DRAIN phase with the MMA chain between them.
  //
  // The split is the entire point of double-buffering. Emitting
  // `load; store; mma` puts the waitcnt for the load in FRONT of the MMA and
  // overlaps nothing, however many buffers there are. `load; mma; store` puts
  // it behind, so the MMA chain runs while the loads are outstanding.
  // ------------------------------------------------------------------
  struct Staged {
    SmallVector<Value> aVals, aDsts;   // aVals are vecTy when vecW > 1
    SmallVector<Value> bVals, bDsts;   // bDsts: depth * vecW scalar addresses
    SmallVector<Value> bVecVals, bVecDsts;  // row-major B: one store per group
  };

  Value cVec = ci(vecW);
  Value txV = b.create<arith::MulIOp>(loc, tx, cVec);
  const int64_t tripA = (wgM * 16) / (nthreads * vecW);
  const int64_t tripB = (wgN * 16) / (nthreads * vecW);
  auto clampDepth = [&](int64_t trip) {
    int64_t d = std::max<int64_t>(1, std::min<int64_t>(ldsCopyDepth, trip));
    while (d > 1 && trip % d) --d;
    return d;
  };
  // Double-buffering REQUIRES a single flat batch (depth == trip), because the
  // drain has to be movable past the MMA chain. While the staging is an
  // `scf.for`, the loaded values and their destination addresses are SSA
  // values inside that loop's region and the store cannot leave it -- which
  // would put the load's waitcnt back in front of the MMA and overlap nothing.
  // At depth == trip the batch covers the whole tile, so it is emitted as
  // straight-line code in the enclosing block and the drain can be placed
  // wherever the caller wants it.
  const bool flatStage = ldsDoubleBuffer;
  const int64_t depthA = flatStage ? tripA : clampDepth(tripA);
  const int64_t depthB = flatStage ? tripB : clampDepth(tripB);

  // `aOff`/`bOff` are element offsets selecting the LDS buffer; zero when
  // single-buffered. `out` non-null means ISSUE only.
  auto emitStage = [&](OpBuilder &kb, Location l, Value k0, Value aOff,
                       Value bOff, Staged *out) {
    auto withOff = [&](Value idx, Value off) {
      return off ? Value(kb.create<arith::AddIOp>(l, idx, off)) : idx;
    };
    // ---------------- A ----------------
    {
      OpBuilder::InsertionGuard g(kb);
      Value e0 = txV;
      if (!flatStage) {
        auto copyA = kb.create<scf::ForOp>(
            l, txV, cWgM16, ci(nthreads * vecW * depthA));
        kb.setInsertionPointToStart(copyA.getBody());
        e0 = copyA.getInductionVar();
      }
      struct GrpA { Value e, gr, gk, rowIn, dst, logical, whole; };
      SmallVector<GrpA> ga(depthA);
      for (int64_t i = 0; i < depthA; ++i) {
        GrpA &q = ga[i];
        q.e = i == 0 ? e0
                     : Value(kb.create<arith::AddIOp>(
                           l, e0, ci(i * nthreads * vecW)));
        Value row = kb.create<arith::DivUIOp>(l, q.e, c16);
        Value kk = kb.create<arith::RemUIOp>(l, q.e, c16);
        q.gr = kb.create<arith::AddIOp>(l, baseRow, row);
        q.gk = kb.create<arith::AddIOp>(l, k0, kk);
        q.rowIn = kb.create<arith::CmpIOp>(l, slt, q.gr, M);
        Value flat = padElems == 0
                         ? q.e
                         : Value(kb.create<arith::AddIOp>(
                               l, kb.create<arith::MulIOp>(l, row, cLdsStride),
                               kk));
        q.dst = withOff(flat, aOff);
        q.logical = kb.create<arith::AddIOp>(
            l, kb.create<arith::MulIOp>(l, q.gr, K), q.gk);
        q.whole = kb.create<arith::AndIOp>(
            l, q.rowIn,
            kb.create<arith::CmpIOp>(l, arith::CmpIPredicate::sle,
                                     kb.create<arith::AddIOp>(l, q.gk, cVec),
                                     K));
      }
      SmallVector<Value> vals(depthA);
      if (ldsCopyElide) {
        // CEILING PROBE -- writes a constant instead of reading global, so the
        // kernel is DELIBERATELY WRONG. It must still write exactly the
        // destinations the real copy writes, at every width and depth.
        for (int64_t i = 0; i < depthA; ++i) {
          if (vecW == 1)
            kb.create<memref::StoreOp>(l, scalarZero, ldsA,
                                       ValueRange{ga[i].dst});
          else
            kb.create<vector::StoreOp>(
                l, kb.create<vector::BroadcastOp>(l, vecTy, scalarZero), ldsA,
                ValueRange{ga[i].dst});
        }
        return;
      }
      if (vecW == 1) {
        // Branchless: masks with `select` on a clamped address, not control
        // flow, so nothing separates the loads in the issue phase.
        SmallVector<Value> in(depthA), raw(depthA);
        for (int64_t i = 0; i < depthA; ++i) {   // ISSUE
          in[i] = kb.create<arith::AndIOp>(
              l, ga[i].rowIn, kb.create<arith::CmpIOp>(l, slt, ga[i].gk, K));
          raw[i] = kb.create<memref::LoadOp>(
              l, A,
              ValueRange{kb.create<arith::SelectOp>(l, in[i], ga[i].logical,
                                                    c0)});
        }
        for (int64_t i = 0; i < depthA; ++i)
          vals[i] =
              kb.create<arith::SelectOp>(l, in[i], raw[i], scalarZero);
      } else {
        // TWO PATHS. A `vector.maskedload` reaches AMDGCN as a per-element
        // branch plus a NARROW load, so the in-bounds case must be an UNMASKED
        // `vector.load`. The guard covers the WHOLE batch: a per-group
        // `scf.if` would put a block boundary between two loads and cap the
        // depth at one. Both arms YIELD the staged vectors so the store can be
        // moved past the MMA; the ragged arm assembles its vector from the
        // per-element masked loads it already had to do.
        Value allWhole = ga[0].whole;
        for (int64_t i = 1; i < depthA; ++i)
          allWhole = kb.create<arith::AndIOp>(l, allWhole, ga[i].whole);
        // scf::IfOp has no result-typed builder that takes body callbacks
        // (only bool add*Block forms), so the regions are filled by hand.
        SmallVector<Type> vtys(depthA, vecTy);
        auto ifOp = kb.create<scf::IfOp>(l, TypeRange(vtys), allWhole,
                                         /*addThenBlock=*/true,
                                         /*addElseBlock=*/true);
        {
          OpBuilder::InsertionGuard ig(kb);
          kb.setInsertionPointToStart(ifOp.thenBlock());
          SmallVector<Value> v(depthA);
          for (int64_t i = 0; i < depthA; ++i)
            v[i] = kb.create<vector::LoadOp>(l, vecTy, A,
                                             ValueRange{ga[i].logical});
          kb.create<scf::YieldOp>(l, v);
        }
        {
          OpBuilder::InsertionGuard ig(kb);
          kb.setInsertionPointToStart(ifOp.elseBlock());
          SmallVector<Value> v(depthA);
          for (int64_t q = 0; q < depthA; ++q) {
            Value acc = kb.create<vector::BroadcastOp>(l, vecTy, scalarZero);
            for (int64_t i = 0; i < vecW; ++i) {
              Value off = ci(i);
              Value in = kb.create<arith::AndIOp>(
                  l, ga[q].rowIn,
                  kb.create<arith::CmpIOp>(
                      l, slt, kb.create<arith::AddIOp>(l, ga[q].gk, off), K));
              Value e1 = kb.create<memref::LoadOp>(
                  l, A,
                  ValueRange{kb.create<arith::SelectOp>(
                      l, in, kb.create<arith::AddIOp>(l, ga[q].logical, off),
                      c0)});
              e1 = kb.create<arith::SelectOp>(l, in, e1, scalarZero);
              acc = kb.create<vector::InsertOp>(l, e1, acc,
                                                ArrayRef<int64_t>{i});
            }
            v[q] = acc;
          }
          kb.create<scf::YieldOp>(l, v);
        }
        for (int64_t i = 0; i < depthA; ++i) vals[i] = ifOp.getResult(i);
      }
      auto drainA = [&](OpBuilder &ob, Location ol) {
        for (int64_t i = 0; i < depthA; ++i) {
          if (vecW == 1)
            ob.create<memref::StoreOp>(ol, vals[i], ldsA,
                                       ValueRange{ga[i].dst});
          else
            ob.create<vector::StoreOp>(ol, vals[i], ldsA,
                                       ValueRange{ga[i].dst});
        }
      };
      if (!out) {
        drainA(kb, l);
      } else {
        // ISSUE only. Straight-line (flatStage), so these values outlive this
        // scope and the caller drains them after the MMA chain.
        out->aVals.assign(vals.begin(), vals.end());
        for (int64_t i = 0; i < depthA; ++i) out->aDsts.push_back(ga[i].dst);
      }
    }
    // ---------------- B ----------------
    {
      OpBuilder::InsertionGuard g(kb);
      Value e0 = txV;
      if (!flatStage) {
        auto copyB = kb.create<scf::ForOp>(
            l, txV, cWgN16, ci(nthreads * vecW * depthB));
        kb.setInsertionPointToStart(copyB.getBody());
        e0 = copyB.getInductionVar();
      }
      struct GrpB { Value e, col, kk, gk, gc, kIn, logical, whole; };
      SmallVector<GrpB> gb(depthB);
      for (int64_t i = 0; i < depthB; ++i) {
        GrpB &q = gb[i];
        q.e = i == 0 ? e0
                     : Value(kb.create<arith::AddIOp>(
                           l, e0, ci(i * nthreads * vecW)));
        q.kk = kb.create<arith::DivUIOp>(l, q.e, cWgN);
        q.col = kb.create<arith::RemUIOp>(l, q.e, cWgN);
        q.gk = kb.create<arith::AddIOp>(l, k0, q.kk);
        q.gc = kb.create<arith::AddIOp>(l, baseCol, q.col);
        q.kIn = kb.create<arith::CmpIOp>(l, slt, q.gk, K);
        q.logical = kb.create<arith::AddIOp>(
            l, kb.create<arith::MulIOp>(l, q.gk, N), q.gc);
        q.whole = kb.create<arith::AndIOp>(
            l, q.kIn,
            kb.create<arith::CmpIOp>(l, arith::CmpIPredicate::sle,
                                     kb.create<arith::AddIOp>(l, q.gc, cVec),
                                     N));
      }
      // B's LDS side is scalar at EVERY width: consecutive `e` are consecutive
      // COLUMNS, one full LDS row apart in the transposed layout (10j.1).
      auto dstOf = [&](OpBuilder &ob, Location ol, const GrpB &q,
                       int64_t lane) {
        Value colI = lane == 0 ? q.col
                               : Value(ob.create<arith::AddIOp>(ol, q.col,
                                                                ci(lane)));
        // Row-major [K][N]: consecutive `e` are consecutive COLUMNS at one k,
        // so consecutive destinations are adjacent and the store widens --
        // which is the entire point. Column-major puts a full padded row
        // between them, which is why it cannot.
        Value flat =
            ldsBRowMajor
                ? Value(ob.create<arith::AddIOp>(
                      ol, ob.create<arith::MulIOp>(ol, q.kk, cWgN), colI))
                : Value(ob.create<arith::AddIOp>(
                      ol, ob.create<arith::MulIOp>(ol, colI, cLdsStride),
                      q.kk));
        return bOff ? Value(ob.create<arith::AddIOp>(ol, flat, bOff)) : flat;
      };
      if (ldsCopyElide) {
        for (int64_t q = 0; q < depthB; ++q)
          for (int64_t i = 0; i < vecW; ++i)
            kb.create<memref::StoreOp>(l, scalarZero, ldsB,
                                       ValueRange{dstOf(kb, l, gb[q], i)});
        return;
      }
      if (vecW == 1) {
        SmallVector<Value> in(depthB), raw(depthB);
        for (int64_t i = 0; i < depthB; ++i) {   // ISSUE
          in[i] = kb.create<arith::AndIOp>(
              l, gb[i].kIn, kb.create<arith::CmpIOp>(l, slt, gb[i].gc, N));
          raw[i] = kb.create<memref::LoadOp>(
              l, B,
              ValueRange{kb.create<arith::SelectOp>(l, in[i], gb[i].logical,
                                                    c0)});
        }
        for (int64_t i = 0; i < depthB; ++i) {
          Value v =
              kb.create<arith::SelectOp>(l, in[i], raw[i], scalarZero);
          Value d = dstOf(kb, l, gb[i], 0);
          if (out) { out->bVals.push_back(v); out->bDsts.push_back(d); }
          else kb.create<memref::StoreOp>(l, v, ldsB, ValueRange{d});
        }
      } else {
        Value allWhole = gb[0].whole;
        for (int64_t i = 1; i < depthB; ++i)
          allWhole = kb.create<arith::AndIOp>(l, allWhole, gb[i].whole);
        SmallVector<Type> vtys(depthB, vecTy);
        auto ifOp = kb.create<scf::IfOp>(l, TypeRange(vtys), allWhole,
                                         /*addThenBlock=*/true,
                                         /*addElseBlock=*/true);
        {
          OpBuilder::InsertionGuard ig(kb);
          kb.setInsertionPointToStart(ifOp.thenBlock());
          SmallVector<Value> v(depthB);
          for (int64_t i = 0; i < depthB; ++i)
            v[i] = kb.create<vector::LoadOp>(l, vecTy, B,
                                             ValueRange{gb[i].logical});
          kb.create<scf::YieldOp>(l, v);
        }
        {
          OpBuilder::InsertionGuard ig(kb);
          kb.setInsertionPointToStart(ifOp.elseBlock());
          SmallVector<Value> v(depthB);
          for (int64_t q = 0; q < depthB; ++q) {
            Value acc = kb.create<vector::BroadcastOp>(l, vecTy, scalarZero);
            for (int64_t i = 0; i < vecW; ++i) {
              Value off = ci(i);
              Value in = kb.create<arith::AndIOp>(
                  l, gb[q].kIn,
                  kb.create<arith::CmpIOp>(
                      l, slt, kb.create<arith::AddIOp>(l, gb[q].gc, off), N));
              Value e1 = kb.create<memref::LoadOp>(
                  l, B,
                  ValueRange{kb.create<arith::SelectOp>(
                      l, in, kb.create<arith::AddIOp>(l, gb[q].logical, off),
                      c0)});
              e1 = kb.create<arith::SelectOp>(l, in, e1, scalarZero);
              acc = kb.create<vector::InsertOp>(l, e1, acc,
                                                ArrayRef<int64_t>{i});
            }
            v[q] = acc;
          }
          kb.create<scf::YieldOp>(l, v);
        }
        for (int64_t q = 0; q < depthB; ++q) {
          if (ldsBRowMajor) {
            // The destinations are adjacent, so the whole group is one store.
            Value d = dstOf(kb, l, gb[q], 0);
            if (out) {
              out->bVecVals.push_back(ifOp.getResult(q));
              out->bVecDsts.push_back(d);
            } else {
              kb.create<vector::StoreOp>(l, ifOp.getResult(q), ldsB,
                                         ValueRange{d});
            }
            continue;
          }
          for (int64_t i = 0; i < vecW; ++i) {
            Value v = kb.create<vector::ExtractOp>(l, ifOp.getResult(q),
                                                   ArrayRef<int64_t>{i});
            Value d = dstOf(kb, l, gb[q], i);
            if (out) { out->bVals.push_back(v); out->bDsts.push_back(d); }
            else kb.create<memref::StoreOp>(l, v, ldsB, ValueRange{d});
          }
        }
      }
    }
  };

  // The DRAIN. Placed after the MMA chain by the double-buffered loop, which
  // is what puts the loads' waitcnt behind the compute instead of in front.
  auto emitDrain = [&](OpBuilder &kb, Location l, const Staged &st) {
    for (size_t i = 0; i < st.aVals.size(); ++i) {
      if (vecW == 1)
        kb.create<memref::StoreOp>(l, st.aVals[i], ldsA,
                                   ValueRange{st.aDsts[i]});
      else
        kb.create<vector::StoreOp>(l, st.aVals[i], ldsA,
                                   ValueRange{st.aDsts[i]});
    }
    for (size_t i = 0; i < st.bVals.size(); ++i)
      kb.create<memref::StoreOp>(l, st.bVals[i], ldsB,
                                 ValueRange{st.bDsts[i]});
    for (size_t i = 0; i < st.bVecVals.size(); ++i)
      kb.create<vector::StoreOp>(l, st.bVecVals[i], ldsB,
                                 ValueRange{st.bVecDsts[i]});
  };

  auto emitCompute = [&](OpBuilder &kb, Location l, ValueRange accs,
                         Value aRow, Value bCol, SmallVectorImpl<Value> &next) {
    SmallVector<Value> af(mt), bf(nt);
    for (int64_t mi = 0; mi < mt; ++mi) {
      Value r = aRow ? Value(kb.create<arith::AddIOp>(l, lrow[mi], aRow))
                     : lrow[mi];
      af[mi] = packFragment(kb, l, ldsView(kb, l, ldsA, r, c0, ldsRowMajor),
                            aFragmentTy);
    }
    for (int64_t ni = 0; ni < nt; ++ni) {
      if (ldsBRowMajor) {
        // [K=16][wgN], so the sub-tile origin is (k=0, n=lcol[ni]) and the
        // buffer selector moves whole K-blocks. The A-role read pattern is
        // contiguous here; `transpose` owes the major-order fix.
        Value br = bCol ? bCol : c0;
        bf[ni] = packFragment(
            kb, l,
            ldsViewStrided(kb, l, ldsB, br, lcol[ni], cWgN, ldsRowMajor),
            bFragmentTy, /*transpose=*/true);
        continue;
      }
      Value cc = bCol ? Value(kb.create<arith::AddIOp>(l, lcol[ni], bCol))
                      : lcol[ni];
      bf[ni] = packFragment(kb, l, ldsView(kb, l, ldsB, c0, cc, ldsColMajor),
                            bFragmentTy);
    }
    next.assign(mt * nt, Value());
    for (int64_t mi = 0; mi < mt; ++mi)
      for (int64_t ni = 0; ni < nt; ++ni) {
        OperationState mma(l, "tile.mma");
        mma.addOperands({af[mi], bf[ni], accs[mi * nt + ni]});
        mma.addTypes(accFragmentTy);
        next[mi * nt + ni] = kb.create(mma)->getResult(0);
      }
  };

  scf::ForOp kLoop;
  if (!ldsDoubleBuffer) {
    kLoop = b.create<scf::ForOp>(
        loc, c0, kEnd, c16, initAccs,
        [&](OpBuilder &kb, Location l, Value k0, ValueRange accs) {
          // Every wave finished reading the previous slab before it is
          // overwritten.
          kb.create<gpu::BarrierOp>(l);
          emitStage(kb, l, k0, Value(), Value(), nullptr);
          kb.create<gpu::BarrierOp>(l);
          SmallVector<Value> next;
          emitCompute(kb, l, accs, Value(), Value(), next);
          kb.create<scf::YieldOp>(l, next);
        });
  } else {
    // PROLOGUE: slab 0 into buffer 0, so the loop always has a full buffer to
    // compute on and only ever writes the other one.
    emitStage(b, loc, c0, Value(), Value(), nullptr);
    b.create<gpu::BarrierOp>(loc);
    Value cAOff = ci(wgM * ldsStride), cBOff = ci(ldsBElems);
    Value cWgMv = ci(wgM);
    // A's fragment view is indexed by ROW, so its buffer selector counts rows.
    // Row-major B is indexed by k-row too, so its selector counts the 16 rows
    // of a K-slab; column-major B is indexed by column and counts wgN.
    Value cWgNv = ci(ldsBRowMajor ? 16 : wgN);
    kLoop = b.create<scf::ForOp>(
        loc, c0, kEnd, c16, initAccs,
        [&](OpBuilder &kb, Location l, Value k0, ValueRange accs) {
          // cur = (k0/16) & 1. The staging writes the OTHER buffer, so the two
          // never alias within a step and ONE barrier per iteration suffices:
          // reads of a buffer in step k-1 precede that barrier, writes of the
          // same buffer in step k follow it.
          Value step = kb.create<arith::DivUIOp>(l, k0, c16);
          Value cur = kb.create<arith::RemUIOp>(l, step, ci(2));
          Value nxt = kb.create<arith::SubIOp>(l, ci(1), cur);
          auto sel = [&](Value which, Value unit) {
            return Value(kb.create<arith::MulIOp>(l, which, unit));
          };
          // ISSUE: slab k0+16 into `nxt`. Past the end its loads mask to zero
          // and nothing ever reads the result, so no guard is needed.
          Staged st;
          emitStage(kb, l, kb.create<arith::AddIOp>(l, k0, c16),
                    sel(nxt, cAOff), sel(nxt, cBOff), &st);
          // COMPUTE on `cur` while those loads are outstanding.
          SmallVector<Value> next;
          emitCompute(kb, l, accs, sel(cur, cWgMv), sel(cur, cWgNv), next);
          // DRAIN: only now does anything wait on the loads.
          emitDrain(kb, l, st);
          // Describe the pipeline: issue the loads, then alternate staging
          // VALU with single matrix ops so the U pipe is busy through the
          // chain, then drain to LDS. The independence this relies on is
          // exactly what double-buffering bought.
          if (ldsSchedValuPerMma != 0) {
            auto i32 = kb.getI32Type();
            auto grp = [&](ROCDL::SchedGroupMask m, int64_t n) {
              kb.create<ROCDL::SchedGroupBarrier>(
                  l, ROCDL::SchedGroupMaskAttr::get(kb.getContext(), m),
                  IntegerAttr::get(i32, n), IntegerAttr::get(i32, 0));
            };
            grp(ROCDL::SchedGroupMask::vmem_read, depthA + depthB);
            // N < 0 is the CONTROL arm: memory grouping only, no (VALU, wmma)
            // alternation. It exists because the alternation measurably did not
            // happen -- the ISA shows zero VALU between the first and last wmma
            // at every positive N -- while the option still helped. Without
            // this arm the gain would be credited to a mechanism that is not
            // running.
            // N == -2 is the MECHANISM arm: the matrix groups WITHOUT any
            // valu groups. The schedule diff says the alternation's VALU half
            // places nothing (zero VALU in the chain at every N) while its
            // mfma half clusters the matrix ops ahead of the ds_write group,
            // which is what lifts the global-load waits out of the chain. If
            // that reading is right this arm reproduces the whole gain.
            if (ldsSchedValuPerMma > 0)
              for (int64_t g = 0; g < mt * nt; ++g) {
                grp(ROCDL::SchedGroupMask::valu, ldsSchedValuPerMma);
                grp(ROCDL::SchedGroupMask::mfma_wmma, 1);
              }
            else if (ldsSchedValuPerMma == -2)
              for (int64_t g = 0; g < mt * nt; ++g)
                grp(ROCDL::SchedGroupMask::mfma_wmma, 1);
            // N == -3 discriminates the remaining reading. Measured: the mfma
            // groups ALONE are worse than no description at all (0.973x), so
            // the valu groups are load-bearing even though they place no VALU
            // in the chain -- they act as a RESERVATION that keeps the
            // ds_st/WAIT_load pairs out of it. If that is right, any spacer the
            // scheduler cannot fill should do, and an salu spacer must behave
            // like the valu one. If instead it reverts, something about VALU
            // specifically matters and the reading is still incomplete.
            else if (ldsSchedValuPerMma == -3)
              for (int64_t g = 0; g < mt * nt; ++g) {
                grp(ROCDL::SchedGroupMask::salu, 32);
                grp(ROCDL::SchedGroupMask::mfma_wmma, 1);
              }
            grp(ROCDL::SchedGroupMask::ds_write, depthA + depthB * vecW);
          }
          kb.create<gpu::BarrierOp>(l);
          kb.create<scf::YieldOp>(l, next);
        });
  }

  // Masked typed stores with the fused epilogue handed to the consumer.
  Attribute epilogueAttr;
  if (hasBias || activation != "none") {
    StringRef outputName = outputType.isF16()   ? "f16"
                           : outputType.isBF16() ? "bf16"
                           : outputType.isF32()  ? "f32"
                                                 : "i32";
    epilogueAttr = tessera::tile::TileEpilogueAttr::get(ctx, hasBias,
                                                        activation, outputName);
  }
  ValueRange accs = kLoop.getResults();
  for (int64_t ni = 0; ni < nt; ++ni)
    for (int64_t mi = 0; mi < mt; ++mi) {
      OperationState unpack(loc, "tile.fragment_unpack");
      unpack.addOperands(accs[mi * nt + ni]);
      unpack.addTypes(tileValueTy);
      unpack.addAttribute("tile.layout", tileLayout);
      Value tile = b.create(unpack)->getResult(0);
      OperationState store(loc, "tile.store");
      store.addOperands({tile, D, rowOrigin[mi], colOrigin[ni], M, N, N});
      if (epilogueAttr) {
        if (hasBias)
          store.addOperands({bias});
        store.addAttribute("tile.epilogue", epilogueAttr);
      }
      store.addAttribute("tile.layout", tileLayout);
      store.addAttribute("tile.memory", gmemRowMajor);
      b.create(store);
    }
  b.create<gpu::ReturnOp>(loc);
}

// Materialize the canonical K loop as a one-wave LDS-staged schedule. The
// shared Tile loop has already proven allocation, completion, and phase
// ownership before it reaches this re-former; this body is the AMD physical
// answer: cooperative global loads, address-space-3 storage, s_barrier on both
// sides of the WMMA consumer, ragged zero fill, and a loop-carried accumulator.
//
// CORRECTED 2026-08-04 -- the note here previously read "the measured gfx1151
// incumbent remains the register schedule ... an LDS schedule that is slower on
// unified-memory Strix Halo". The measurement was right; the conclusion
// generalized from a configuration in which LDS CANNOT help.
//
// This body is one-wave, MT=NT=1, so its block tile is 16x16 and its arithmetic
// intensity is BM*BN/(BM+BN) = 8 FLOP/byte -- IDENTICAL to the naive register
// schedule. Staging through LDS at equal AI buys no reuse and costs barriers
// plus a round trip, so it must be slower, and it is. Measured at 2048^3 f16 on
// gfx1151:
//
//   naive          MT=NT=1                16x16 tile   AI  8.0    3.62 TFLOP/s
//   LDS  1x1 waves MT=NT=1                16x16 tile   AI  8.0    2.90
//   LDS  2x2 waves MT=NT=2                64x64 tile   AI 32.0    8.47
//   LDS  4x2 waves MT=NT=2               128x64 tile   AI 42.7    9.78
//   LDS+pipelined 4x2 waves MT=NT=2      128x64 tile   AI 42.7   10.33
//
// LDS staging pays exactly when it enables reuse across a MULTI-WAVE,
// MULTI-TILE block; at 1x1 there is no reuse to capture. The hand-written
// `tessera_rocm_wmma_gemm_f16_bench_{lds,pipe}` kernels reach 2.7-3.5x the
// production register schedule, so the incumbent is not the ceiling -- it is
// the configuration this generator happens to support.
//
// Do not re-derive "LDS is slower here" from a 1x1 experiment.
void emitCanonicalLdsBody(OpBuilder &b, Location loc, gpu::GPUFuncOp gpuFunc,
                          const WmmaTypes &T, Type outputType) {
  MLIRContext *ctx = b.getContext();
  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto ldsTy =
      MemRefType::get({256}, T.store, MemRefLayoutAttrInterface(), ws);
  Value ldsA = gpuFunc.addWorkgroupAttribution(ldsTy, loc);
  Value ldsB = gpuFunc.addWorkgroupAttribution(ldsTy, loc);

  b.setInsertionPointToStart(&gpuFunc.getBody().front());
  Value A = gpuFunc.getArgument(0);
  Value B = gpuFunc.getArgument(1);
  Value D = gpuFunc.getArgument(2);
  Value M = gpuFunc.getArgument(3);
  Value N = gpuFunc.getArgument(4);
  Value K = gpuFunc.getArgument(5);
  auto ci = [&](int64_t value) {
    return b.create<arith::ConstantIndexOp>(loc, value);
  };
  Value c0 = ci(0), c4 = ci(4), c15 = ci(15), c16 = ci(16);
  Value c32 = ci(32), c256 = ci(256);
  Value tx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value lane = b.create<arith::AndIOp>(loc, tx, c15);
  Value lhi = b.create<arith::ShRUIOp>(loc, tx, c4);
  Value baseRow = b.create<arith::MulIOp>(
      loc, b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y), c16);
  Value baseCol = b.create<arith::MulIOp>(
      loc, b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x), c16);

  Value scalarZero;
  Value fragmentZero;
  Value accumulatorZero;
  if (T.isInt) {
    scalarZero = b.create<arith::ConstantOp>(
        loc, T.store, b.getIntegerAttr(T.store, 0));
    fragmentZero = b.create<arith::ConstantOp>(
        loc, T.load,
        DenseElementsAttr::get(cast<ShapedType>(T.load), APInt(8, 0)));
    accumulatorZero = b.create<arith::ConstantOp>(
        loc, T.acc,
        DenseElementsAttr::get(cast<ShapedType>(T.acc), APInt(32, 0)));
  } else {
    scalarZero =
        b.create<arith::ConstantOp>(loc, T.store, b.getFloatAttr(T.store, 0.0));
    APFloat zero =
        cast<FloatAttr>(b.getFloatAttr(T.store, 0.0)).getValue();
    fragmentZero = b.create<arith::ConstantOp>(
        loc, T.load,
        DenseElementsAttr::get(cast<ShapedType>(T.load), zero));
    APFloat accZero =
        cast<FloatAttr>(b.getFloatAttr(T.accElem, 0.0)).getValue();
    accumulatorZero = b.create<arith::ConstantOp>(
        loc, T.acc,
        DenseElementsAttr::get(cast<ShapedType>(T.acc), accZero));
  }

  auto loadSafe = [&](OpBuilder &ib, Value memref, Value logical,
                      Value inBounds) {
    Value safe = ib.create<arith::SelectOp>(loc, inBounds, logical, c0);
    Value loaded = ib.create<memref::LoadOp>(loc, memref, ValueRange{safe});
    return Value(
        ib.create<arith::SelectOp>(loc, inBounds, loaded, scalarZero));
  };

  auto kLoop = b.create<scf::ForOp>(
      loc, c0, K, c16, ValueRange{accumulatorZero},
      [&](OpBuilder &kb, Location kloc, Value k0, ValueRange iter) {
        auto copyA = kb.create<scf::ForOp>(kloc, tx, c256, c32);
        {
          OpBuilder::InsertionGuard guard(kb);
          kb.setInsertionPointToStart(copyA.getBody());
          Value e = copyA.getInductionVar();
          Value row = kb.create<arith::DivUIOp>(kloc, e, c16);
          Value kk = kb.create<arith::RemUIOp>(kloc, e, c16);
          Value gr = kb.create<arith::AddIOp>(kloc, baseRow, row);
          Value gk = kb.create<arith::AddIOp>(kloc, k0, kk);
          Value rowIn = kb.create<arith::CmpIOp>(
              kloc, arith::CmpIPredicate::slt, gr, M);
          Value kIn = kb.create<arith::CmpIOp>(
              kloc, arith::CmpIPredicate::slt, gk, K);
          Value in = kb.create<arith::AndIOp>(kloc, rowIn, kIn);
          Value logical = kb.create<arith::AddIOp>(
              kloc, kb.create<arith::MulIOp>(kloc, gr, K), gk);
          kb.create<memref::StoreOp>(kloc, loadSafe(kb, A, logical, in), ldsA,
                                    ValueRange{e});
        }
        auto copyB = kb.create<scf::ForOp>(kloc, tx, c256, c32);
        {
          OpBuilder::InsertionGuard guard(kb);
          kb.setInsertionPointToStart(copyB.getBody());
          Value e = copyB.getInductionVar();
          Value kk = kb.create<arith::DivUIOp>(kloc, e, c16);
          Value col = kb.create<arith::RemUIOp>(kloc, e, c16);
          Value gk = kb.create<arith::AddIOp>(kloc, k0, kk);
          Value gc = kb.create<arith::AddIOp>(kloc, baseCol, col);
          Value kIn = kb.create<arith::CmpIOp>(
              kloc, arith::CmpIPredicate::slt, gk, K);
          Value colIn = kb.create<arith::CmpIOp>(
              kloc, arith::CmpIPredicate::slt, gc, N);
          Value in = kb.create<arith::AndIOp>(kloc, kIn, colIn);
          Value logical = kb.create<arith::AddIOp>(
              kloc, kb.create<arith::MulIOp>(kloc, gk, N), gc);
          kb.create<memref::StoreOp>(kloc, loadSafe(kb, B, logical, in), ldsB,
                                    ValueRange{e});
        }
        kb.create<gpu::BarrierOp>(kloc);

        Value aBase = kb.create<arith::MulIOp>(kloc, lane, c16);
        Value aFragment =
            kb.create<vector::LoadOp>(kloc, T.load, ldsA, ValueRange{aBase});
        Value bFragment = fragmentZero;
        for (int64_t i = 0; i < 16; ++i) {
          Value index = kb.create<arith::AddIOp>(
              kloc,
              kb.create<arith::MulIOp>(
                  kloc, kb.create<arith::ConstantIndexOp>(kloc, i), c16),
              lane);
          Value element =
              kb.create<memref::LoadOp>(kloc, ldsB, ValueRange{index});
          bFragment = kb.create<vector::InsertOp>(
              kloc, element, bFragment, ArrayRef<int64_t>{i});
        }
        Value aOperand = aFragment;
        Value bOperand = bFragment;
        if (T.pack == 1) {
          aOperand = kb.create<vector::BitCastOp>(kloc, T.frag, aFragment);
          bOperand = kb.create<vector::BitCastOp>(kloc, T.frag, bFragment);
        }
        OperationState wmma(kloc, "tessera_rocm.wmma");
        wmma.addOperands({aOperand, bOperand, iter.front()});
        wmma.addTypes(T.acc);
        Value next = kb.create(wmma)->getResult(0);
        kb.create<gpu::BarrierOp>(kloc);
        kb.create<scf::YieldOp>(kloc, next);
      });

  Value acc = kLoop.getResult(0);
  for (int64_t e = 0; e < 8; ++e) {
    Value row = b.create<arith::AddIOp>(
        loc, baseRow,
        b.create<arith::AddIOp>(loc, ci(2 * e), lhi));
    Value col = b.create<arith::AddIOp>(loc, baseCol, lane);
    Value rowIn =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, row, M);
    Value colIn =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, col, N);
    auto storeIf = b.create<scf::IfOp>(
        loc, b.create<arith::AndIOp>(loc, rowIn, colIn),
        /*withElseRegion=*/false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(storeIf.thenBlock());
    int64_t accumulatorIndex = T.halfAccumulator ? 2 * e : e;
    Value value = b.create<vector::ExtractOp>(
        loc, acc, ArrayRef<int64_t>{accumulatorIndex});
    if (value.getType() != outputType)
      value = b.create<arith::TruncFOp>(loc, outputType, value);
    Value index = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, row, N), col);
    b.create<memref::StoreOp>(loc, value, D, ValueRange{index});
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
  Option<std::string> canonicalStaging{
      *this, "canonical-staging",
      llvm::cl::desc("physical schedule for a canonical M/N/K loop: "
                     "register (gfx1151 incumbent) or lds (comparison lane); "
                     "with via-tile, lds selects the multi-wave LDS-staged "
                     "typed body"),
      llvm::cl::init("register")};
  Option<int> kUnroll{*this, "k-unroll",
                      llvm::cl::desc("typed body: full 16-wide K slabs issued "
                                     "per loop iteration (latency hiding; 1 = "
                                     "the established one-slab loop)"),
                      llvm::cl::init(1)};
  Option<int> schedGroups{
      *this, "sched-groups",
      llvm::cl::desc("rocdl.sched.group.barrier granularity for the panel: "
                     "how many (vmem_read, mfma_wmma) groups the body is "
                     "described as. 0 emits nothing and keeps LLVM's default "
                     "drained schedule (ROCM-SCHED-GROUP-1)"),
      llvm::cl::init(0)};
  Option<int> ldsPadDwords{
      *this, "lds-pad-dwords",
      llvm::cl::desc("LDS-staged typed body: dwords of padding added to each "
                     "tile row so the fragment read stops colliding on the "
                     "32 x 4 B banks. 0 is the unpadded historical layout. "
                     "Default 4, from the only shape where the measurement "
                     "converges: at 2048 cubed f16 padding helps monotonically "
                     "and 4 wins by 15% with tight dispersion, while at 1024 "
                     "cubed the same configuration remeasures 16% apart in one "
                     "process and separates nothing. The earlier default of 1 "
                     "came from a harness that launched 4x the workgroups and "
                     "is withdrawn (ROCM-LDS-BANKPAD-1, "
                     "docs/backends/rocm/wmma-fragment-layout.md 10i/10j). "
                     "DEFAULT MOVED 4 -> 1 on 2026-09-20: measured paired and "
                     "interleaved on BOTH ROCm parts, 1 beats 4 by 1.10-1.19x "
                     "on gfx1201 and 1.21-1.29x on gfx1151, winning 6/6 at "
                     "2048 and 4096 cubed with double-buffering on and off. "
                     "4 was set from one shape in a configuration predating "
                     "issue depth; the value it replaces was right for the "
                     "wrong reason and is now right for a measured one (10u, "
                     "10v)"),
      llvm::cl::init(1)};
  Option<bool> ldsBRowMajor{
      *this, "lds-b-row-major",
      llvm::cl::desc(
          "LDS-staged typed body: stage B ROW-major [K, N] and transpose the "
          "fragment in-register instead of transposing during the LDS write. "
          "B's global read is contiguous in N while a b-role fragment needs 8 "
          "contiguous K, so exactly one of the three sides is always strided "
          "-- today it is the LDS write, which stays scalar at every copy "
          "width (10j.1). Row-major makes the global read, the LDS write AND "
          "the fragment read contiguous, and pays one WMMA per B fragment for "
          "the transpose. Also drops B's padding, since the strided column "
          "read that needed it is gone. See "
          "docs/backends/rocm/wmma-fragment-layout.md 10s"),
      llvm::cl::init(false)};
  Option<int> ldsSchedValuPerMma{
      *this, "lds-sched-valu-per-mma",
      llvm::cl::desc(
          "LDS-staged typed body: describe the loop to the scheduler as "
          "[all loads][N VALU, 1 wmma] x mmas [all LDS stores] via "
          "rocdl.sched.group.barrier, so the staging address arithmetic runs "
          "on the U pipe while the matrix core occupies V. Measured 2026-09-20: "
          "the body emits ZERO VALU between the first and last wmma, i.e. the U "
          "pipe is idle for the whole matrix chain. Only meaningful with "
          "lds-double-buffer, which is what makes the staging VALU independent "
          "of the current step's MMA -- without it the staging must complete "
          "before the barrier the MMA reads through. 0 = emit nothing. "
          "See docs/backends/rocm/wmma-fragment-layout.md 10o"),
      llvm::cl::init(0)};
  Option<bool> ldsDoubleBuffer{
      *this, "lds-double-buffer",
      llvm::cl::desc(
          "LDS-staged typed body: stage slab k+1 into a second LDS buffer while "
          "the MMA chain consumes slab k, so the global load latency hides "
          "behind compute instead of in front of it. Issues the loads BEFORE "
          "the MMA and drains them to LDS AFTER, which is the whole point -- "
          "emitting load;store;mma leaves the waitcnt in front of the MMA and "
          "overlaps nothing. Costs 2x LDS and saves one barrier per iteration "
          "(the two buffers are never read and written in the same step). "
          "See docs/backends/rocm/wmma-fragment-layout.md 10n"),
      llvm::cl::init(false)};
  Option<int> ldsCopyDepth{
      *this, "lds-copy-depth",
      llvm::cl::desc(
          "LDS-staged typed body: how many staging groups are ISSUED before "
          "any is consumed. 1 is the historical load-store-load-store loop, "
          "which keeps exactly one global load in flight and is why the copy "
          "sits at loadcnt<=0 while the register body reaches loadcnt<=6. "
          "Depth N splits the batch into an issue phase and a drain phase, so "
          "N loads are outstanding. Clamped to a divisor of the per-thread "
          "trip count (16/width here), which is compile-time, so no remainder "
          "loop is needed. See docs/backends/rocm/wmma-fragment-layout.md "
          "10j.6"),
      llvm::cl::init(1)};
  Option<bool> ldsCopyElide{
      *this, "lds-copy-elide",
      llvm::cl::desc("CEILING PROBE ONLY -- emits a DELIBERATELY WRONG kernel. "
                     "The LDS staging copy writes a constant instead of reading "
                     "global, so every output is zero. Keeps barriers, loop "
                     "structure and the MMA chain identical, so (real - elided) "
                     "bounds the staging copy's entire cost and therefore what "
                     "any copy optimisation can buy. Never a production path: "
                     "its own test asserts the result is WRONG "
                     "(ROCM-LDS-STAGE-VECTOR-1, wmma-fragment-layout.md 10j.1)"),
      llvm::cl::init(false)};
  Option<int> ldsCopyWidth{
      *this, "lds-copy-width",
      llvm::cl::desc("LDS staging copy: elements per thread per step. 1 is "
                     "the default and the MEASURED FASTER arm; 0 derives the "
                     "widest the padded stride allows, which is currently a "
                     "13-49% REGRESSION because vector.maskedload expands to "
                     "per-element branches rather than a wide load "
                     "(ROCM-LDS-STAGE-VECTOR-1)"),
      llvm::cl::init(1)};
  Option<int> ldsWavesM{*this, "lds-waves-m",
                        llvm::cl::desc("LDS-staged typed body: waves along M "
                                       "per workgroup"),
                        llvm::cl::init(2)};
  Option<int> ldsWavesN{*this, "lds-waves-n",
                        llvm::cl::desc("LDS-staged typed body: waves along N "
                                       "per workgroup"),
                        llvm::cl::init(2)};

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, vector::VectorDialect,
                    arith::ArithDialect, math::MathDialect,
                    memref::MemRefDialect,
                    tessera::tile::TesseraTileDialect,
                    // ROCDL because the panel emits rocdl.sched.group.barrier
                    // directly. An undeclared dependent dialect is silent on
                    // an NDEBUG driver and a hard error under assertions.
                    ROCDL::ROCDLDialect,
                    mlir::tessera_rocm::TesseraROCMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (canonicalStaging != "register" && canonicalStaging != "lds") {
      getOperation()->emitError(
          "generate-wmma-gemm-kernel: canonical-staging must be register or "
          "lds");
      return signalPassFailure();
    }

    SmallVector<WmmaGemmRequest, 2> requests;

    // Direct shared-contract adapter. Prefer the Tile form after the shared
    // async seam and ROCm ownership planner have materialized !tile.buffer,
    // !tile.async_token, and !tile.pipeline_state edges. The pre-Tile form is
    // retained for narrow structural compatibility tests.
    SmallVector<Operation *> canonicalSteps;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if ((name == "tessera.matmul" || name == "tile.mma") &&
          op->hasAttr("tessera.canonical_k_step"))
        canonicalSteps.push_back(op);
    });
    for (Operation *step : canonicalSteps) {
      FailureOr<WmmaGemmRequest> request = matchCanonicalGemmLoop(step);
      if (failed(request)) {
        step->emitError(
            "ROCm canonical GEMM requires the verified three-level M/N/K "
            "scf.for contract with loop-carried pipeline state, external "
            "rank-2 slices, ragged zero-fill, and f16/bf16->f32 or i8->i32 "
            "accumulation");
        return signalPassFailure();
      }
      requests.push_back(std::move(*request));
    }

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
      // K is 16 for every form except the native double-K int4
      // (`V_WMMA_I32_16X16X32_IU4`). `resolveFragmentLayout` already selects
      // that shape and the nibble packer in TileToROCM already emits it -- it
      // loops over `inputElementsPerLane` with `shift = 4 * (i % 8)` into word
      // `i / 8`, so at K=32 it produces two words in the documented order with
      // no change. This gate was the only thing refusing it.
      //
      // The ARCH is deliberately not checked here. This pass does not know it:
      // the target arrives as a `lower-tile-to-rocm` option one pass later,
      // and the IR need not carry it. Duplicating the check from a default
      // would refuse the shape on a request that never named an arch. The
      // lowering is where it belongs, and it already fails closed -- the
      // gfx1100/gfx1151 branch of `resolveFragmentLayout` admits `k == 16`
      // only, so a K=32 int4 fragment aimed at RDNA3 resolves to nothing.
      // RDNA4 has the mixed OCP FP8 pairs, so A and B may name different
      // storages -- but only that pairing, and only fp8 with fp8.
      auto isFp8 = [](llvm::StringRef t) { return t == "e4m3" || t == "e5m2"; };
      const bool mixedFp8Pair = desc && isFp8(desc.getAType()) &&
                                isFp8(desc.getBType()) &&
                                desc.getAType() != desc.getBType();
      const int64_t expectedK =
          (desc && desc.getAType() == "int4" && desc.getK() == 32) ? 32 : 16;
      auto epilogue =
          op->getAttrOfType<tessera::tile::TileEpilogueAttr>("epilogue");
      bool common = desc && epilogue &&
          (desc.getFamily() == "auto" || desc.getFamily() == "wmma") &&
          desc.getM() == 16 && desc.getN() == 16 && desc.getK() == expectedK &&
          (desc.getAType() == desc.getBType() || mixedFp8Pair) &&
          desc.getALayout() == "row_major" &&
          desc.getBLayout() == "col_major" && desc.getKBlocks() >= 1;
      bool floatContract = common &&
          (desc.getAType() == "f16" || desc.getAType() == "bf16") &&
          desc.getAccType() == "f32";
      bool integerContract = common &&
          (desc.getAType() == "int8" || desc.getAType() == "int4") &&
          (desc.getAccType() == "i32" || desc.getAccType() == "int32");
      // OCP FP8 storage (RDNA4 WMMA, k=16 per fragment) accumulates in f32;
      // the typed route packs it per chip (GFX1201-PARITY slice 5).
      bool fp8Contract = common && isFp8(desc.getAType()) &&
          isFp8(desc.getBType()) && desc.getAccType() == "f32";
      bool canonical = floatContract || integerContract || fp8Contract;
      if (!canonical) {
        op->emitError("ROCm tile.matmul_kernel requires an m16n16k16 row/col "
                      "(m16n16k32 for int4 on gfx12) "
                      "WMMA descriptor: f16/bf16/e4m3/e5m2 with f32 "
                      "accumulation or int8/int4 with i32 accumulation");
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
      auto logicalM = op->getAttrOfType<IntegerAttr>("tessera.macro_tile_m");
      auto logicalN = op->getAttrOfType<IntegerAttr>("tessera.macro_tile_n");
      if (logicalM || logicalN) {
        if (!logicalM || !logicalN || logicalM.getInt() < 16 ||
            logicalN.getInt() < 16 || logicalM.getInt() % 16 != 0 ||
            logicalN.getInt() % 16 != 0) {
          op->emitError("ROCm tile.matmul_kernel macro_tile_m/macro_tile_n must "
                        "both be positive multiples of the 16x16 WMMA tile");
          return signalPassFailure();
        }
        mt = logicalM.getInt() / 16;
        nt = logicalN.getInt() / 16;
      } else if (auto warps = op->getAttrOfType<IntegerAttr>("warps");
                 warps && warps.getInt() == 4) {
        // Compatibility for portable launch carriers authored before logical
        // macro-tile extents became the schedule-owned source of truth.
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
      // ROCM-MACRO-K-TILE-1: the descriptor's K block reaches the generator.
      // Until 2026-09-19 `k_blocks` was stated by the Schedule, verified >= 1,
      // and read by nobody except three gates that refused anything but 1 -- so
      // a macro K tile was expressible and unreachable. This is the consumer.
      request.kBlocks = std::max<int64_t>(desc.getKBlocks(), 1);
      request.bias = epilogue.getBias();
      request.activation = epilogue.getActivation().str();
      request.output = epilogue.getOutputType().str();
      request.portableABI = true;
      // Schedule -> Tile binds M/N/K as i64 constants.  Materialization is
      // allowed only when those exact operands remain static; arbitrary Tile
      // producers and dynamic leading dimensions keep the established path.
      auto staticExtent = [&](unsigned operand) -> std::optional<int64_t> {
        if (auto constant =
                op->getOperand(operand).getDefiningOp<arith::ConstantIntOp>())
          return constant.value();
        return std::nullopt;
      };
      unsigned dimStart = op->getNumOperands() - 3;
      auto staticM = staticExtent(dimStart);
      auto staticN = staticExtent(dimStart + 1);
      auto staticK = staticExtent(dimStart + 2);
      if (staticM && staticN && staticK && *staticM > 0 && *staticN > 0 &&
          *staticK > 0) {
        request.staticM = *staticM;
        request.staticN = *staticN;
        request.staticK = *staticK;
      }
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
      // OCP FP8 never rides `$dtype` (the dtype contract keeps fp8 and
      // microscaling forms on `numeric_policy.storage`); the RDNA4 FP8 GEMM
      // leaves `dtype` at its default and names its storage here.
      if (auto policy = op->getAttrOfType<DictionaryAttr>("numeric_policy"))
        if (auto storage = policy.getAs<StringAttr>("storage")) {
          StringRef s = storage.getValue();
          if (s == "e4m3" || s == "e5m2" || s == "fp8" || s == "bf8" ||
              s == "fp8_e4m3" || s == "fp8_e5m2")
            request.dtype = s.str();
        }
      if (auto a = op->getAttrOfType<BoolAttr>("bias"))
        request.bias = a.getValue();
      if (auto a = op->getAttrOfType<StringAttr>("activation"))
        request.activation = a.getValue().str();
      if (auto a = op->getAttrOfType<StringAttr>("output"))
        request.output = a.getValue().str();
      // The directive lane spells the raster contract `schedule_raster_*`;
      // the typed `tile.matmul_kernel` carries the Schedule's decision as
      // `tessera.raster_*`. Read both (ROCM-RASTER-1 on the typed route,
      // 2026-09-18); the selection is still row-major until measured.
      if (auto a = op->getAttrOfType<StringAttr>("schedule_raster_order"))
        request.rasterOrder = a.getValue().str();
      else if (auto a = op->getAttrOfType<StringAttr>("tessera.raster_order"))
        request.rasterOrder = a.getValue().str();
      if (auto a = op->getAttrOfType<IntegerAttr>("schedule_raster_group"))
        request.rasterGroup = a.getInt();
      else if (auto a = op->getAttrOfType<IntegerAttr>("tessera.raster_group"))
        request.rasterGroup = a.getInt();
      request.storagePack =
          op->getAttrOfType<tessera::tile::TilePackedFormatAttr>(
              "tessera.storage_pack");
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
      if (request.rasterOrder != "row_major" &&
          request.rasterOrder != "column_major" &&
          request.rasterOrder != "grouped_m" &&
          request.rasterOrder != "grouped_n") {
        op->emitError("generate-wmma-gemm-kernel: schedule raster order must "
                      "be row_major, column_major, grouped_m, or grouped_n");
        return signalPassFailure();
      }
      if (request.rasterGroup < 1) {
        op->emitError("generate-wmma-gemm-kernel: schedule raster group must "
                      "be >= 1");
        return signalPassFailure();
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
      // backend-neutral structured `#tile.packed_format`
      // descriptor (from StoragePackConsume), its `logical` selects the dtype —
      // one packing contract feeds both AMD (here) and NVIDIA. Fall back to the
      // legacy `dtype` attr when no descriptor is present (non-breaking).
      tessera::tile::TilePackedFormatAttr packDesc = request.storagePack;
      if (packDesc) {
        dt = packDesc.getLogicalType();
      }
      auto v8i32 = VectorType::get({8}, i32Ty);
      auto v8f32 = VectorType::get({8}, f32Ty);
      auto v16i8 = VectorType::get({16}, i8Ty);
      StringRef declaredAccum;
      if (auto policy = op->getAttrOfType<DictionaryAttr>("numeric_policy"))
        if (auto accumAttr = policy.getAs<StringAttr>("accum"))
          declaredAccum = accumAttr.getValue();
      const bool requestF16Accumulator =
          (dt == "f16" || dt == "float16") &&
          (declaredAccum == "f16" || declaredAccum == "fp16");
      if (requestF16Accumulator) {
        auto ack = op->getAttrOfType<StringAttr>(
            "tessera.rocm.reduced_precision_accumulation");
        if (!ack || ack.getValue() != "f16_wmma_accuracy_cost_ack_v1") {
          op->emitError(
              "ROCM_WMMA_ACCUM_UNSUPPORTED: numeric_policy accum=\"fp16\" "
              "is an opt-in accuracy class; set "
              "tessera.rocm.reduced_precision_accumulation"
              "=\"f16_wmma_accuracy_cost_ack_v1\" to acknowledge the measured "
              "5212x (K=64) to 7856x (K=4096) relative-error cost");
          return signalPassFailure();
        }
        T = {f16Ty, VectorType::get({16}, f16Ty),
             VectorType::get({16}, f16Ty), VectorType::get({16}, f16Ty),
             f16Ty, /*isInt=*/false, /*halfAccumulator=*/true,
             /*pack=*/0, /*packFactor=*/1};
      } else if (dt == "f16" || dt == "float16") {
        T = {f16Ty, VectorType::get({16}, f16Ty), VectorType::get({16}, f16Ty),
             v8f32, f32Ty, /*isInt=*/false, /*halfAccumulator=*/false,
             /*pack=*/0, /*packFactor=*/1};
      } else if (dt == "bf16" || dt == "bfloat16") {
        T = {bf16Ty, VectorType::get({16}, bf16Ty),
             VectorType::get({16}, bf16Ty), v8f32, f32Ty, /*isInt=*/false,
             /*halfAccumulator=*/false, /*pack=*/0, /*packFactor=*/1};
      } else if (dt == "int8" || dt == "i8") {
        T = {i8Ty, v16i8, VectorType::get({4}, i32Ty), v8i32, i32Ty,
             /*isInt=*/true, /*halfAccumulator=*/false, /*pack=*/1,
             /*packFactor=*/1};
      } else if (dt == "int4" || dt == "i4") {
        // int4 values supplied in int8 containers (range [-8,7]); the low nibble
        // is the int4 two's-complement. Nibble-packed in-kernel to the iu4 ABI
        // vector<2xi32>; i32 accumulate. (correctness-first — no coalesced load.)
        T = {i8Ty, v16i8, VectorType::get({2}, i32Ty), v8i32, i32Ty,
             /*isInt=*/true, /*halfAccumulator=*/false, /*pack=*/2,
             /*packFactor=*/2};
        // RDNA4's native double-K int4 consumes 32 K elements per fragment.
        // Only the K loop's stride changes here; the fragment types above are
        // the gfx11 shapes for the directive path, and the typed route
        // re-derives its own per arch from the descriptor.
        if (auto mma = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma"))
          if (mma.getK() == 32)
            T.fragK = 32;
      } else if (dt == "e4m3" || dt == "fp8" || dt == "fp8_e4m3" ||
                 dt == "e5m2" || dt == "bf8" || dt == "fp8_e5m2") {
        // OCP FP8 storage, f32 accumulate. Only the typed route carries it:
        // TileToROCM packs the 8-bit fragment per chip (RDNA4 `rdna4_wmma`,
        // vector<2xi32> per lane, k=16) and gfx11 has no FP8 WMMA at all, so
        // the direct gfx11 body below refuses it by name.
        const bool e4m3 = (dt == "e4m3" || dt == "fp8" || dt == "fp8_e4m3");
        Type f8Ty = e4m3 ? static_cast<Type>(Float8E4M3FNType::get(b.getContext()))
                         : static_cast<Type>(Float8E5M2Type::get(b.getContext()));
        T = {f8Ty, VectorType::get({16}, f8Ty), VectorType::get({2}, i32Ty),
             v8f32, f32Ty, /*isInt=*/false, /*halfAccumulator=*/false,
             /*pack=*/0, /*packFactor=*/1};
        // The mixed pair: A and B may name different FP8 storages, and the
        // fragment types are what carry that to `resolveFragmentLayout`, which
        // already selects FP8_BF8 / BF8_FP8 from them. Both are 8 bits, so the
        // register format and the packing are unchanged -- only the selected
        // instruction differs.
        if (auto mma = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma"))
          if (mma.getAType() != mma.getBType() &&
              (mma.getBType() == "e4m3" || mma.getBType() == "e5m2"))
            T.bElem = mma.getBType().str();
        if (!viaTile) {
          op->emitError("generate-wmma-gemm-kernel: FP8 storage ('")
              << dt << "') is a typed-route contract (via-tile=true); the "
                        "direct gfx11 body has no FP8 WMMA";
          return signalPassFailure();
        }
      } else {
        op->emitError("generate-wmma-gemm-kernel: dtype must be f16, bf16, "
                      "int8, int4, e4m3, or e5m2 (got '")
            << dt << "')";
        return signalPassFailure();
      }

      // ROCM-MACRO-K-TILE-1: the descriptor's K block reaches the loop. Until
      // 2026-09-19 `k_blocks` was stated by the Schedule, verified >= 1, and
      // read by nobody except three gates that refused anything but 1 -- so a
      // macro K tile was expressible and unreachable. This is the consumer.
      T.kBlocks = std::max<int64_t>(request.kBlocks, 1);

      // ── NUMPOL-CARRIER-1: the declared accumulator gets a CONSUMER ──
      //
      // Measured 2026-08-25: `numeric_policy` was carried faithfully all the
      // way here — TileIRLoweringPass puts it on `tile.mma`, TileToROCM copies
      // it onto `tessera_rocm.wmma_gemm` — and then nothing read it. The
      // accumulator that reached the hardware was inferred a few lines above
      // from the STORAGE dtype alone. Two sources of truth for one fact, and
      // the declared one lost; they agreed only because every real program
      // asks for fp32.
      //
      // That is Decision #29's case (a declaration whose consumer does not
      // exist reads as a closed contract while carrying nothing) sitting on
      // top of #21a's (a semantic key silently defaulting). `accum` is a
      // semantic key: it decides what the program COMPUTES, so an accumulator
      // this path cannot provide must fail closed here rather than be quietly
      // replaced by f32 and reported as success.
      //
      // gfx1151 really does have the alternative — the ISA archive records
      // `V_WMMA_F16_16X16X16_F16` on RDNA 3.5 — but its ROCDL form is
      // `(v16f16, v16f16, v16f16) -> v16f16` with an `opsel` bit selecting a
      // half, a different accumulator ABI from the v8f32 path emitted below.
      // Wiring that is its own slice with its own device proof. Until then the
      // honest answer to "accumulate in f16" is a diagnostic, not an f32
      // kernel.
      if (auto policy = op->getAttrOfType<DictionaryAttr>("numeric_policy")) {
        if (auto accumAttr = policy.getAs<StringAttr>("accum")) {
          StringRef declared = accumAttr.getValue();
          StringRef provided =
              T.isInt ? "int32" : T.halfAccumulator ? "fp16" : "fp32";
          bool matches = declared == provided ||
                         (T.isInt && (declared == "i32" || declared == "int32")) ||
                         (!T.isInt && !T.halfAccumulator &&
                          (declared == "f32" || declared == "fp32")) ||
                         (T.halfAccumulator &&
                          (declared == "f16" || declared == "fp16"));
          if (!matches) {
            op->emitError(
                  "ROCM_WMMA_ACCUM_UNSUPPORTED: numeric_policy declares "
                  "accum=\"")
                << declared << "\" for storage '" << dt
                << "', but the gfx1151 WMMA path emitted here accumulates in "
                << provided
                << ". Refusing rather than substituting: `accum` selects what "
                   "the program computes, so silently widening or narrowing it "
                   "would report success for a different computation "
                   "(Decisions #21a/#29).";
            return signalPassFailure();
          }
        }
      }

      // C4 reconciliation: the storage-pack `factor` (logical values per byte
      // container) must equal this dtype's `packFactor` — the single packing
      // contract. (Verify against `packFactor`, the logical contract, NOT
      // `pack`, the codegen ABI mode: today int8/int4 happen to share the value,
      // but a new int ABI mode at the same logical factor must still pass.)
      if (packDesc && T.isInt) {
        int64_t factor = packDesc.getElementsPerContainer();
        if (factor != T.packFactor) {
          op->emitError("DTYPE_PACK_FACTOR_MISMATCH: tessera.storage_pack "
                        "factor ")
              << factor << " disagrees with the dtype packing factor "
              << T.packFactor << " for dtype '" << dt << "'.";
          return signalPassFailure();
        }
        if (dt == "int4") {
          if (packDesc.getSignedness() != "signed_twos_complement") {
            op->emitError("DTYPE_PACK_SIGNEDNESS_MISMATCH: gfx1151 int4 WMMA "
                          "requires signed_twos_complement packed storage");
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
      if (T.halfAccumulator && (hasBias || activation != "none")) {
        op->emitError(
            "ROCM_WMMA_ACCUM_UNSUPPORTED: the initial f16-accumulate WMMA "
            "envelope excludes fused bias and activation epilogues");
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
      // B's buffer carries B's own element type. It is the same as A's in every
      // case but RDNA4's mixed OCP FP8 pairs, and giving them one type made the
      // fragment materializer reject the pair: it derives the expected source
      // element from the descriptor's `b`, which then disagreed with a memref
      // typed from A.
      Type bStoreTy = T.store;
      if (!T.bElem.empty())
        bStoreTy = T.bElem == "e4m3"
                       ? static_cast<Type>(Float8E4M3FNType::get(b.getContext()))
                       : static_cast<Type>(Float8E5M2Type::get(b.getContext()));
      auto bAbTy = MemRefType::get({ShapedType::kDynamic}, bStoreTy);
      auto dTy = MemRefType::get({ShapedType::kDynamic}, outputTy);
      auto biasTy = MemRefType::get({ShapedType::kDynamic}, T.accElem);
      SmallVector<Type> argTys{abTy, bAbTy};
      if (hasBias && portableContract)
        argTys.push_back(biasTy);
      argTys.append({dTy, idxTy, idxTy, idxTy});
      if (hasBias && !portableContract)
        argTys.push_back(biasTy); // legacy directive ABI: trailing bias
      auto fnTy = b.getFunctionType(argTys, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc.setKernel(true);
      // The 256-VGPR ceiling this body hits at the 4x4 panel is ARCHITECTURAL,
      // not an occupancy default: RDNA4 ISA 3.3.2.1 -- "VGPRs are allocated in
      // blocks of 16 for wave32 or 8 for wave64, and a shader may have up to
      // 256 VGPRs" -- and dynamic VGPR mode (3.3.3) caps at the same 256 with
      // a 32-VGPR block size, 128 with 16. The 768 KiB file is per CU and
      // shared across wave slots; no occupancy request hands one wave more
      // than 256. A `waves-per-eu` knob briefly lived here to test the
      // opposite hypothesis (that constraining to one wave per SIMD would
      // lift the cap, as it does on CDNA); the attribute reached the llvm.func
      // and changed nothing, and the ISA says why. It is deleted rather than
      // left as an unconsumed declaration (Decision #29). Reducing the panel's
      // live-register footprint is the lever; raising the ceiling is not
      // available.
      // The typed 2x4 f16/bf16 body carries gfx1151's performance-closure
      // digest (TileToROCM refuses it on any other arch). Stamp it only when
      // the request is gfx11's: the op's `arch`/`schedule_arch`, else the
      // module's `tessera.arch`, else the historical gfx11 default. A gfx12
      // 2x4 panel is an ordinary typed body, measured on its own chip
      // (GFX1201-PARITY, typed-route gap, 2026-09-18).
      StringRef requestArch = "gfx1151";
      if (auto a = op->getAttrOfType<StringAttr>("arch"))
        requestArch = a.getValue();
      else if (auto a = op->getAttrOfType<StringAttr>("schedule_arch"))
        requestArch = a.getValue();
      else if (auto moduleOp = op->getParentOfType<ModuleOp>())
        if (auto a = moduleOp->getAttrOfType<StringAttr>("tessera.arch"))
          requestArch = a.getValue();
      const bool gfx11Request = requestArch.starts_with("gfx11");
      // The stamp claims THIS body is gfx1151's measured 2x4 register panel,
      // and TileToROCM pins anything carrying it to that chip AND checks its
      // exact view/pack/mma/unpack/store topology. An LDS-staged body, or a
      // K-unrolled one (two slabs of packs and MMAs per iteration), is a
      // different physical body: it must not claim the contract, or the
      // topology check refuses it with a message about a contract it never
      // meant to make (2026-09-18).
      if (viaTile && gfx11Request && canonicalStaging != "lds" && kUnroll <= 1 &&
          mt == 2 && nt == 4 && T.pack == 0 && !hasBias &&
          activation == "none" && outputTy == T.accElem) {
        gpuFunc->setAttr("tessera.rocm.typed_gfx11_gemm_contract",
                         b.getUnitAttr());
        gpuFunc->setAttr("tessera.rocm.physical_panel_mt",
                         b.getI64IntegerAttr(mt));
        gpuFunc->setAttr("tessera.rocm.physical_panel_nt",
                         b.getI64IntegerAttr(nt));
      }
      for (StringRef attrName : {"schedule_arch", "schedule_pipeline_stages",
                                 "schedule_lds_layout", "schedule_ownership",
                                 "schedule_vgpr_estimate", "schedule_source",
                                 "schedule_raster_order",
                                 "schedule_raster_group"})
        if (Attribute attr = op->getAttr(attrName))
          gpuFunc->setAttr((Twine("tessera.rocm.") + attrName).str(), attr);
      // The typed spelling lands under the same kernel attribute names so a
      // reader of the generated kernel sees one raster contract.
      if (!op->hasAttr("schedule_raster_order"))
        for (auto [typed, kernel] :
             {std::pair{"tessera.raster_order", "tessera.rocm.schedule_raster_order"},
              std::pair{"tessera.raster_group", "tessera.rocm.schedule_raster_group"}})
          if (Attribute attr = op->getAttr(typed))
            gpuFunc->setAttr(kernel, attr);
      if (request.canonicalKLoop) {
        gpuFunc->setAttr("tessera.rocm.source",
                         b.getStringAttr("canonical_mnk_scf_for"));
        gpuFunc->setAttr("tessera.rocm.canonical_k_loop",
                         b.getBoolAttr(true));
        gpuFunc->setAttr("tessera.rocm.ssa_ownership_proof",
                         b.getBoolAttr(request.ssaOwnershipProof));
        gpuFunc->setAttr("tessera.rocm.ragged_zero_pad",
                         b.getBoolAttr(request.raggedZeroPad));
        gpuFunc->setAttr("tessera.rocm.accumulate",
                         b.getStringAttr(request.accumulate));
        gpuFunc->setAttr("tessera.rocm.tile_m",
                         b.getI64IntegerAttr(request.logicalTileM));
        gpuFunc->setAttr("tessera.rocm.tile_n",
                         b.getI64IntegerAttr(request.logicalTileN));
        gpuFunc->setAttr("tessera.rocm.tile_k",
                         b.getI64IntegerAttr(request.logicalTileK));
        gpuFunc->setAttr("tessera.rocm.physical_staging",
                         b.getStringAttr(canonicalStaging));
      }

      OpBuilder bodyB(gpuFunc.getContext());
      // The typed via-tile path carries the fused bias/activation epilogue
      // since 2026-09-17 (applied by the architecture consumer at the store)
      // and, since 2026-09-18, int4: the typed producer hands TileToROCM an
      // i8 tile view and its fragment materializer compacts the nibbles per
      // chip (slice 1b). A reduced output type remains untyped-only.
      if (viaTile && outputTy != T.accElem) {
        op->emitError(
            "generate-wmma-gemm-kernel: typed via-tile pilot requires a GEMM "
            "stored in its accumulator type");
        return signalPassFailure();
      }
      if (viaTile && canonicalStaging == "lds") {
        if (T.halfAccumulator || !portableContract) {
          op->emitError("generate-wmma-gemm-kernel: the LDS-staged typed body "
                        "takes the portable Tile ABI with a full-width "
                        "accumulator");
          return signalPassFailure();
        }
        if (ldsWavesM <= 0 || ldsWavesN <= 0 || ldsWavesM * ldsWavesN > 16) {
          op->emitError("generate-wmma-gemm-kernel: lds-waves-m/n must be "
                        "positive with at most 16 waves per workgroup");
          return signalPassFailure();
        }
        emitTypedLdsBody(bodyB, loc, gpuFunc, mt, nt, ldsWavesM, ldsWavesN, T,
                         outputTy, hasBias, activation, request.rasterOrder,
                         request.rasterGroup, ldsPadDwords,
                         ldsCopyWidth, ldsCopyElide, ldsCopyDepth,
                         ldsDoubleBuffer, ldsSchedValuPerMma,
                         ldsBRowMajor);
      } else if (request.canonicalKLoop && canonicalStaging == "lds") {
        // This body writes its accumulator back at row `2*e + lhi`, which is
        // RDNA3's wave32 distribution. gfx12 distributes the same accumulator
        // by COLUMN (`(lane/16)*8 + j`, see
        // docs/backends/rocm/wmma-fragment-layout.md section 2), and the
        // `tessera_rocm.wmma` this body emits is the arch-resolving Target IR
        // op -- so on gfx12 it would lower to the RDNA4 instruction and then
        // scatter the result to RDNA3 rows. That is the silent-wrong-tiles
        // failure the layout contract exists to prevent: no verifier catches
        // it and the kernel runs to completion.
        //
        // Unlike the control-for-WMMA generators, which emit the gfx11 rocdl
        // intrinsic directly and therefore die at instruction selection on
        // gfx12, this one has nothing to fail on. It is a gfx11-only
        // comparison lane whose evidence is gfx1151-only, so it refuses by
        // name instead (Decision #21). Found 2026-09-19 by a structural
        // search for accumulator index math with no arch branch; a substring
        // audit the same week had cleared this file.
        if (!gfx11Request) {
          op->emitError(
              "ROCM_CANONICAL_LDS_ARCH_UNSUPPORTED: the canonical LDS "
              "comparison body stores its accumulator in the RDNA3 row "
              "distribution and is admitted on gfx11 only; this request is ")
              << requestArch;
          return signalPassFailure();
        }
        if (T.halfAccumulator) {
          op->emitError(
              "ROCM_WMMA_ACCUM_UNSUPPORTED: f16 accumulation is admitted "
              "only on the measured register-staged gfx1151 path");
          return signalPassFailure();
        }
        if (hasBias || activation != "none" || T.pack == 2 || mt != 1 ||
            nt != 1) {
          op->emitError("generate-wmma-gemm-kernel: canonical LDS comparison "
                        "supports one-wave f16/bf16/int8 GEMM without a fused "
                        "epilogue");
          return signalPassFailure();
        }
        gpuFunc->setAttr("tessera.rocm.lds_bytes",
                         b.getI64IntegerAttr(
                             512 * T.store.getIntOrFloatBitWidth() / 8));
        gpuFunc->setAttr("tessera.rocm.pipeline_stages",
                         b.getI64IntegerAttr(1));
        emitCanonicalLdsBody(bodyB, loc, gpuFunc, T, outputTy);
      } else {
        emitGeneralBody(bodyB, loc, gpuFunc, mt, nt, T, outputTy,
                        portableContract, viaTile, hasBias, activation,
                        packDesc && dt == "int4", request.rasterOrder,
                        request.rasterGroup, request.staticM, request.staticN,
                        request.staticK, kUnroll, schedGroups);
      }
      if (gpuFunc->hasAttr("tessera.rocm.typed_gfx11_gemm_contract"))
        gpuFunc->setAttr(
            "tessera.rocm.typed_contract_digest",
            b.getStringAttr(
                tessera_rocm::gfx11WmmaGemmTileBodyDigest(gpuFunc)));

      if (!llvm::is_contained(generatedOwners, request.eraseOwner))
        generatedOwners.push_back(request.eraseOwner);
    }
    for (Operation *owner : generatedOwners)
      owner->erase();
  }
};

} // namespace

LogicalResult mlir::tessera_rocm::materializeGfx11WmmaGemmPhysicalBody(
    gpu::GPUFuncOp function, int64_t mt, int64_t nt) {
  if (mt != 2 || nt != 4 || function.getNumArguments() != 6)
    return failure();
  auto inputMemref = dyn_cast<MemRefType>(function.getArgument(0).getType());
  auto outputMemref = dyn_cast<MemRefType>(function.getArgument(2).getType());
  if (!inputMemref || !outputMemref ||
      (!inputMemref.getElementType().isF16() &&
       !inputMemref.getElementType().isBF16()) ||
      !outputMemref.getElementType().isF32())
    return failure();

  MLIRContext *context = function.getContext();
  WmmaTypes types;
  types.store = inputMemref.getElementType();
  types.load = VectorType::get({16}, types.store);
  types.frag = types.load;
  types.accElem = Float32Type::get(context);
  types.acc = VectorType::get({8}, types.accElem);
  types.isInt = false;
  types.halfAccumulator = false;
  types.pack = 0;
  types.packFactor = 1;

  StringRef rasterOrder = "row_major";
  int64_t rasterGroup = 1;
  if (auto attr = function->getAttrOfType<StringAttr>(
          "tessera.rocm.schedule_raster_order"))
    rasterOrder = attr.getValue();
  if (auto attr = function->getAttrOfType<IntegerAttr>(
          "tessera.rocm.schedule_raster_group"))
    rasterGroup = attr.getInt();

  Block &body = function.getBody().front();
  while (!body.empty())
    body.back().erase();
  OpBuilder builder(context);
  emitGeneralBody(builder, function.getLoc(), function, mt, nt, types,
                  outputMemref.getElementType(), /*portableABI=*/false,
                  /*viaTile=*/false, /*hasBias=*/false, "none",
                  /*packedInt4Memory=*/false, rasterOrder, rasterGroup);
  function->removeAttr("tessera.rocm.typed_gfx11_gemm_contract");
  function->removeAttr("tessera.rocm.typed_contract_digest");
  function->removeAttr("tessera.rocm.physical_panel_mt");
  function->removeAttr("tessera.rocm.physical_panel_nt");
  return success();
}

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateWMMAGemmKernelPass() {
  return std::make_unique<GenerateWMMAGemmKernelPass>();
}
