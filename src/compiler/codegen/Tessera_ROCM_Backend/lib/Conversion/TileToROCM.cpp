#include "TesseraROCM/Passes.h"
#include "ROCMDirectAttention.h"
#include "ROCMFragmentLayout.h"

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "TesseraROCMDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "TesseraROCMTypes.h.inc"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

// ── FP8 flavor derivation (B4 — arch-keyed FNUZ vs OCP) ─────────────────────
// The SAME canonical fp8 dtype encodes different bits across AMD generations.
// The *base* (e4m3 / e5m2) comes from the operand element type; the *suffix*
// (fnuz vs OCP-plain) comes from the target arch.  This C++ table is the
// emission-side mirror of the Python source of truth
// `tessera.compiler.rocm_target._FP8_SEMANTICS` and is held in sync by
// tests/unit/test_rocm_fp8_cpp_python_consistency.py.

static bool isFP8Element(Type t) {
  if (auto sh = dyn_cast<ShapedType>(t))
    t = sh.getElementType();
  return isa<Float8E4M3FNType, Float8E5M2Type, Float8E4M3FNUZType,
             Float8E5M2FNUZType>(t);
}

static std::string fp8Base(Type t) {
  if (auto sh = dyn_cast<ShapedType>(t))
    t = sh.getElementType();
  if (isa<Float8E4M3FNType, Float8E4M3FNUZType>(t))
    return "e4m3";
  if (isa<Float8E5M2Type, Float8E5M2FNUZType>(t))
    return "e5m2";
  return "";
}

// RDNA arches use the WMMA matrix instruction; CDNA arches use MFMA. The
// matmul tile lowering must pick the right matrix op per arch — emitting MFMA
// on RDNA (which has no matrix-fused-multiply-add core) is a silent miscompile.
// RDNA = gfx11xx (RDNA 3 / 3.5) and gfx12xx (RDNA 4); CDNA = gfx9xx.
static bool isWmmaArch(llvm::StringRef arch) {
  return arch.starts_with("gfx11") || arch.starts_with("gfx12");
}

static Value toIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(), value);
}

// `desc` is passed rather than read off the op: on the typed path there is no
// `mma` attribute at all -- the fragment TYPE carries the contract, and the
// descriptor is synthesized from it (see `descriptorFromFragment`). Reading it
// off the op here is what made this helper usable only by the attribute form.
// `role` is passed rather than read off the op for the same reason as `desc`:
// on the typed path the RESULT TYPE carries the role, and
// `FragmentPackOp::verify` returns success without ever requiring the `role`
// attribute once the result is typed. So `tile.fragment_pack %v : (!tile.tile)
// -> !fa` is valid IR -- and it is exactly the shape a migrated producer emits.
// Reading the attribute here rejected it with ROCM_FRAGMENT_MISSING_CONTRACT.
static FailureOr<Value> materializeFragmentPack(
    tessera::tile::FragmentPackOp pack, StringRef role,
    tessera::tile::TileMmaDescAttr desc, OpBuilder &builder, Value lane,
    Value laneGroup,
    const tessera_rocm::FragmentLayoutDescriptor &physical) {
  Operation *op = pack.getOperation();
  auto view = pack.getInputs().front().getDefiningOp<tessera::tile::ViewOp>();
  if (role.empty() || !view || !desc) {
    op->emitError("ROCM_FRAGMENT_MISSING_CONTRACT: fragment materialization "
                  "requires tile.view-backed A/B fragments and a resolved "
                  "architecture descriptor");
    return failure();
  }
  auto memory = view->getAttrOfType<tessera::tile::TileMemoryLayoutAttr>(
      "tile.memory");
  auto layout = view->getAttrOfType<tessera::tile::TileLayoutAttr>(
      "tile.layout");
  std::array<int64_t, 2> expectedShape =
      role == "a"
          ? std::array<int64_t, 2>{desc.getM(), desc.getK()}
          : std::array<int64_t, 2>{desc.getK(), desc.getN()};
  // Both memory orders are addressable. The fragment ALWAYS walks K; what the
  // order decides is whether K is contiguous in memory or strided by the
  // leading dimension:
  //
  //   role  order        K axis   linear                  K stride
  //   a     row_major    col      row * ld + col          1
  //   a     col_major    col      col * ld + row          ld
  //   b     row_major    row      row * ld + col          ld
  //   b     col_major    row      col * ld + row          1
  //
  // Only the two contiguous cases used to be accepted, and the strided ones
  // were rejected as an "unsupported source layout". That is why
  // `GenerateWMMAGemmKernel` could not migrate: it stores B row-major
  // (`k * N + col`), so its B fragment is a stride-N gather, which it does by
  // hand with 16 scalar loads.
  const bool kIsContiguous = (role == "a") == (memory.getOrder() == "row_major");
  // 3 inputs = (base, rowOrigin, colOrigin); 5 adds (rowBound, colBound) for a
  // ragged problem. Anything else is malformed rather than merely unsupported.
  // This guard used to require exactly 3, which made the bounded form
  // unreachable -- and silently so, since it fell into the generic
  // "unsupported source layout" diagnostic rather than naming the arity.
  const size_t viewInputs = view.getInputs().size();
  if ((role != "a" && role != "b") || !memory ||
      !layout || (viewInputs != 3 && viewInputs != 5) ||
      memory.getSpace() != "gmem" ||
      (memory.getOrder() != "row_major" && memory.getOrder() != "col_major") ||
      layout.getShardExtents() != ArrayRef<int64_t>(expectedShape) ||
      layout.getSwizzle()) {
    op->emitError("ROCM_FRAGMENT_UNSUPPORTED_SOURCE_LAYOUT: unsupported ")
        << physical.familyName << " fragment source layout for role " << role;
    return failure();
  }

  Value base = view.getInputs()[0];
  auto memrefTy = dyn_cast<MemRefType>(base.getType());
  if (!memrefTy || memrefTy.getRank() != 1) {
    op->emitError("ROCM_FRAGMENT_SOURCE_RANK: fragment_pack currently requires "
                  "a rank-1 memref source");
    return failure();
  }
  bool integer = desc.getAType() == "int8" || desc.getAType() == "int4";
  Type elementTy;
  if (integer)
    elementTy = builder.getIntegerType(8);
  else if (desc.getAType() == "bf16")
    elementTy = builder.getBF16Type();
  else if (desc.getAType() == "e4m3" || desc.getAType() == "fp8")
    elementTy = Float8E4M3FNType::get(builder.getContext());
  else if (desc.getAType() == "e5m2" || desc.getAType() == "bf8")
    elementTy = Float8E5M2Type::get(builder.getContext());
  else
    elementTy = builder.getF16Type();
  if (memrefTy.getElementType() != elementTy) {
    op->emitError("ROCM_FRAGMENT_SOURCE_TYPE: source element type does not "
                  "match the architecture MMA descriptor");
    return failure();
  }

  Location loc = op->getLoc();
  Value rowOrigin = toIndex(builder, loc, view.getInputs()[1]);
  Value colOrigin = toIndex(builder, loc, view.getInputs()[2]);
  Value leadingDim = arith::ConstantIndexOp::create(
      builder, loc, memory.getLeadingDim());
  auto vectorTy =
      VectorType::get({physical.inputElementsPerLane}, elementTy);
  Value zero = arith::ConstantOp::create(builder, loc, vectorTy,
                                         builder.getZeroAttr(vectorTy));
  Value fragment = zero;
  Value kBase = arith::MulIOp::create(
      builder, loc, laneGroup,
      arith::ConstantIndexOp::create(builder, loc,
                                     physical.inputElementsPerLane));
  // W1.1 step 3 — ragged-edge masking.
  //
  // `tile.view` is pointer-backed as (base, rowOrigin, colOrigin). A view may
  // additionally carry (rowBound, colBound): the LOGICAL extents of the matrix,
  // which are runtime values on a ragged problem.
  //
  // Without them this loaded `inputElementsPerLane` contiguous elements
  // unguarded, so a fragment straddling the edge of a ragged matrix read past
  // it. `GenerateWMMAGemmKernel` avoids that by assembling fragments element by
  // element with an `inb ? value : zero` select -- lane math the typed contract
  // is supposed to own, which is exactly why the producer could not migrate
  // (design doc §4.5).
  //
  // The out-of-bounds elements are always a contiguous TAIL: the load walks the
  // fast axis, so element j is in bounds iff `fast + j < fastBound`. That makes
  // `vector.create_mask` of the clamped remaining length exactly right, and
  // `maskedload` with a zero passthru reproduces the generator's `storeZero`
  // element for element -- which is what lets the migrated producer stay
  // bit-identical rather than merely close.
  const bool haveBounds = view.getInputs().size() >= 5;
  Value fastBound, slowIndex, slowBound;

  Value linear, fastIndex;
  Value row, col;
  if (role == "a") {
    // K is the column; the lane selects the row.
    row = arith::AddIOp::create(builder, loc, rowOrigin, lane);
    col = arith::AddIOp::create(builder, loc, colOrigin, kBase);
    fastIndex = col;
    slowIndex = row;
    if (haveBounds) {
      slowBound = toIndex(builder, loc, view.getInputs()[3]);   // rowBound
      fastBound = toIndex(builder, loc, view.getInputs()[4]);   // colBound
    }
  } else {
    // K is the row; the lane selects the column.
    row = arith::AddIOp::create(builder, loc, rowOrigin, kBase);
    col = arith::AddIOp::create(builder, loc, colOrigin, lane);
    fastIndex = row;
    slowIndex = col;
    if (haveBounds) {
      fastBound = toIndex(builder, loc, view.getInputs()[3]);   // rowBound
      slowBound = toIndex(builder, loc, view.getInputs()[4]);   // colBound
    }
  }
  // The fast axis is K for BOTH roles, so the bound mapping above does not
  // depend on the memory order -- only the linearization below does. Getting
  // the fast/slow roles backwards would mask the wrong axis and silently zero
  // live data on ragged shapes.
  linear = memory.getOrder() == "row_major"
               ? arith::AddIOp::create(
                     builder, loc,
                     arith::MulIOp::create(builder, loc, row, leadingDim), col)
               : arith::AddIOp::create(
                     builder, loc,
                     arith::MulIOp::create(builder, loc, col, leadingDim), row);

  if (!kIsContiguous) {
    // Strided K: element j lives `j * leadingDim` away, so neither `vector.load`
    // nor `vector.maskedload` applies. Gather element by element with the same
    // `inb ? value : zero` shape `GenerateWMMAGemmKernel` uses by hand, which is
    // what lets that producer migrate bit-identically rather than merely close.
    Value zeroScalar = arith::ConstantOp::create(
        builder, loc, elementTy, builder.getZeroAttr(elementTy));
    Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
    Value slowOk;
    if (haveBounds)
      slowOk = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                     slowIndex, slowBound);
    for (int64_t j = 0; j < physical.inputElementsPerLane; ++j) {
      Value cj = arith::ConstantIndexOp::create(builder, loc, j);
      Value index = arith::AddIOp::create(
          builder, loc, linear,
          arith::MulIOp::create(builder, loc, cj, leadingDim));
      Value element;
      if (!haveBounds) {
        element = memref::LoadOp::create(builder, loc, base, ValueRange{index});
      } else {
        Value fastOk = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::slt,
            arith::AddIOp::create(builder, loc, fastIndex, cj), fastBound);
        Value inBounds = arith::AndIOp::create(builder, loc, fastOk, slowOk);
        // Clamp the ADDRESS too: an out-of-bounds lane must not fault, and a
        // select on the loaded value alone still performs the load.
        Value safe =
            arith::SelectOp::create(builder, loc, inBounds, index, zeroIdx);
        element = memref::LoadOp::create(builder, loc, base, ValueRange{safe});
        element = arith::SelectOp::create(builder, loc, inBounds, element,
                                          zeroScalar);
      }
      fragment = vector::InsertOp::create(builder, loc, element, fragment,
                                          ArrayRef<int64_t>{j});
    }
  } else if (!haveBounds) {
    fragment = vector::LoadOp::create(builder, loc, vectorTy, base,
                                      ValueRange{linear});
  } else {
    Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
    Value lanes = arith::ConstantIndexOp::create(
        builder, loc, physical.inputElementsPerLane);
    // remaining = clamp(fastBound - fastIndex, 0, lanes)
    Value remaining =
        arith::SubIOp::create(builder, loc, fastBound, fastIndex);
    remaining = arith::MaxSIOp::create(builder, loc, remaining, zeroIdx);
    remaining = arith::MinSIOp::create(builder, loc, remaining, lanes);
    // The slow axis is a single scalar guard: out of range zeroes the fragment.
    Value slowOk = arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::slt,
                                         slowIndex, slowBound);
    Value length =
        arith::SelectOp::create(builder, loc, slowOk, remaining, zeroIdx);
    Value mask = vector::CreateMaskOp::create(
        builder, loc, VectorType::get({physical.inputElementsPerLane},
                                      builder.getI1Type()),
        ValueRange{length});
    fragment = vector::MaskedLoadOp::create(builder, loc, vectorTy, base,
                                            ValueRange{linear}, mask, zero);
  }
  bool packToI32 = physical.inputFormat ==
                       tessera_rocm::FragmentRegisterFormat::SOAInt ||
                   desc.getAType() == "int8";
  if (packToI32 && desc.getAType() != "int4")
    return Value(vector::BitCastOp::create(
        builder, loc,
        VectorType::get({physical.inputRegistersPerLane},
                        builder.getIntegerType(32)),
        fragment));
  if (desc.getAType() != "int4")
    return fragment;

  // int4 uses i8 logical containers. Compact each low two's-complement nibble
  // into the gfx11 iu4 operand ABI: 16 values -> two i32 words per lane.
  Type i32Ty = builder.getIntegerType(32);
  Value mask = arith::ConstantIntOp::create(builder, loc, 0xf, 32);
  SmallVector<Value> words(
      physical.inputRegistersPerLane,
      arith::ConstantIntOp::create(builder, loc, 0, 32));
  for (int64_t i = 0; i < physical.inputElementsPerLane; ++i) {
    Value scalar = vector::ExtractOp::create(builder, loc, fragment,
                                             ArrayRef<int64_t>{i});
    Value extended = arith::ExtUIOp::create(builder, loc, i32Ty, scalar);
    Value nibble = arith::AndIOp::create(builder, loc, extended, mask);
    Value shift = arith::ConstantIntOp::create(builder, loc, 4 * (i % 8), 32);
    Value shifted = arith::ShLIOp::create(builder, loc, nibble, shift);
    words[i / 8] =
        arith::OrIOp::create(builder, loc, words[i / 8], shifted);
  }
  auto packedTy = VectorType::get({physical.inputRegistersPerLane}, i32Ty);
  Value packed = arith::ConstantOp::create(builder, loc, packedTy,
                                           builder.getZeroAttr(packedTy));
  for (int64_t i = 0; i < physical.inputRegistersPerLane; ++i)
    packed = vector::InsertOp::create(builder, loc, words[i], packed,
                                      ArrayRef<int64_t>{i});
  return packed;
}

// Takes the accumulator as a VALUE, not as the producing operation. The typed
// path's accumulator may be an `scf.for` iter-arg -- a block argument with no
// defining op -- which is precisely the case W1.1 exists to make expressible.
static LogicalResult materializeFragmentStore(
    Value accumulator, tessera::tile::TileMmaDescAttr desc,
    tessera::tile::FragmentUnpackOp unpack,
    tessera::tile::StoreOp store, OpBuilder &builder, Value lane,
    Value laneGroup,
    const tessera_rocm::FragmentLayoutDescriptor &physical) {
  Operation *op = store.getOperation();
  auto unpackLayout =
      unpack->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
  auto storeLayout =
      store->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
  auto memory =
      store->getAttrOfType<tessera::tile::TileMemoryLayoutAttr>("tile.memory");
  std::array<int64_t, 2> outputShape{16, 16};
  if (!desc || !unpackLayout || !storeLayout || !memory ||
      memory.getSpace() != "gmem" ||
      memory.getOrder() != "row_major" || unpackLayout.getSwizzle() ||
      storeLayout.getSwizzle() ||
      unpackLayout.getShardExtents() != ArrayRef<int64_t>(outputShape) ||
      storeLayout.getShardExtents() != ArrayRef<int64_t>(outputShape)) {
    op->emitError("ROCM_FRAGMENT_STORE_LAYOUT: architecture accumulator store "
                  "requires unswizzled 16x16 row-major gmem output with "
                  "f32/i32 accumulation");
    return failure();
  }
  Value base = store.getInputs()[1];
  auto memrefTy = dyn_cast<MemRefType>(base.getType());
  bool integer = desc.getAType() == "int8" || desc.getAType() == "int4";
  Type outputTy = integer ? Type(builder.getIntegerType(32))
                          : Type(builder.getF32Type());
  if (!memrefTy || memrefTy.getRank() != 1 ||
      memrefTy.getElementType() != outputTy) {
    op->emitError("ROCM_FRAGMENT_STORE_TYPE: accumulator store requires a "
                  "rank-1 memref whose "
                  "element type matches the f32/i32 accumulator");
    return failure();
  }
  Location loc = op->getLoc();
  Value rowOrigin = toIndex(builder, loc, store.getInputs()[2]);
  Value colOrigin = toIndex(builder, loc, store.getInputs()[3]);
  Value leadingDim = arith::ConstantIndexOp::create(
      builder, loc, memory.getLeadingDim());
  Value groupStride = arith::ConstantIndexOp::create(
      builder, loc, physical.accumulatorElementsPerLane);
  for (int64_t i = 0; i < physical.accumulatorElementsPerLane; ++i) {
    Value ci = arith::ConstantIndexOp::create(builder, loc, i);
    Value row;
    Value col;
    if (physical.usesGfx11AccumulatorMap()) {
      Value two = arith::ConstantIndexOp::create(builder, loc, 2);
      Value rowOffset = arith::AddIOp::create(
          builder, loc, arith::MulIOp::create(builder, loc, ci, two),
          laneGroup);
      row = arith::AddIOp::create(builder, loc, rowOrigin, rowOffset);
      col = arith::AddIOp::create(builder, loc, colOrigin, lane);
    } else {
      row = arith::AddIOp::create(builder, loc, rowOrigin, lane);
      Value colOffset = arith::AddIOp::create(
          builder, loc,
          arith::MulIOp::create(builder, loc, laneGroup, groupStride), ci);
      col = arith::AddIOp::create(builder, loc, colOrigin, colOffset);
    }
    Value linear = arith::AddIOp::create(
        builder, loc,
        arith::MulIOp::create(builder, loc, row, leadingDim), col);
    Value scalar = vector::ExtractOp::create(builder, loc, accumulator,
                                             ArrayRef<int64_t>{i});
    memref::StoreOp::create(builder, loc, scalar, base, ValueRange{linear});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// W1.1 step 0 — the typed fragment path as a dialect conversion.
//
// The legacy typed path (in `runOnOperation`, below) is a single-shot
// WHOLE-CHAIN pattern match: starting from `tile.mma` it chases operands for
// `fragment_pack` A/B and a `fragment_zero` accumulator, requires a
// `fragment_unpack -> tile.store` consumer, emits one physical op, then erases
// the entire chain. Three things are therefore inexpressible *by construction*:
//
//   * an accumulator that is not a literal `fragment_zero`  (plan step 2b),
//   * an `tile.mma` whose result feeds another `tile.mma`,
//   * a chain crossing a loop boundary — i.e. a K-loop, which is what a GEMM
//     *is* (`tile_mma_typed_kloop.mlir` verifies but nothing could lower it).
//
// This replaces chasing with composition. `!tile.fragment<...>` converts to the
// physical per-lane `vector<N x T>`; each op lowers against its ALREADY
// CONVERTED operands and never looks at its producer; SSA does the rest.
// `scf.for` iter-args need no Tile-specific reasoning at all — they come from
// upstream's `populateSCFStructuralTypeConversionsAndLegality`.
//
// The bare `!tile.fragment` spelling converts to itself, so it stays legal and
// the legacy walk keeps handling it. The two forms coexist until plan step 5
// deletes the permissive path.
//===----------------------------------------------------------------------===//

// Defined below. NOT `tessera::tile::isTileControlType`: the local overload is
// a superset that also admits `tessera_rocm::TokenType`, and narrowing to the
// upstream one here would misclassify a ROCm token as a data operand.
static bool isTileControlType(Type type);

namespace {

static Type accumulatorElementType(MLIRContext *ctx, StringRef acc) {
  if (acc == "i32" || acc == "int32")
    return IntegerType::get(ctx, 32);
  if (acc == "f32")
    return Float32Type::get(ctx);
  return {};
}

// Mirrors `materializeFragmentPack`'s own element-type choice, including int4
// riding in i8 storage. Divergence here would silently mistype the loads.
static Type inputElementType(MLIRContext *ctx, StringRef elem) {
  if (elem == "int8" || elem == "int4")
    return IntegerType::get(ctx, 8);
  if (elem == "bf16")
    return BFloat16Type::get(ctx);
  if (elem == "f16")
    return Float16Type::get(ctx);
  if (elem == "e4m3" || elem == "fp8")
    return Float8E4M3FNType::get(ctx);
  if (elem == "e5m2" || elem == "bf8")
    return Float8E5M2Type::get(ctx);
  return {};
}

/// The physical value type `materializeFragmentPack` actually produces for an
/// A/B fragment.
///
/// This is NOT simply `inputElementsPerLane x elem`: the materializer packs
/// sub-16-bit inputs into i32 registers (bitcast for int8/SOA-int, explicit
/// nibble compaction for int4), so an RDNA4 int4 fragment leaves as
/// `vector<2xi32>`, not `vector<16xi8>`. The type converter and the
/// materializer MUST agree or the conversion strands an unresolved
/// materialization; stating the rule in one place is what keeps them agreeing.
static Type packedFragmentType(
    MLIRContext *ctx, StringRef elem,
    const tessera_rocm::FragmentLayoutDescriptor &physical) {
  if (physical.inputFormat == tessera_rocm::FragmentRegisterFormat::SOAInt ||
      elem == "int8" || elem == "int4")
    return VectorType::get({physical.inputRegistersPerLane},
                           IntegerType::get(ctx, 32));
  Type element = inputElementType(ctx, elem);
  return element ? VectorType::get({physical.inputElementsPerLane}, element)
                 : Type();
}

/// The architecture descriptor a fragment TYPE implies, for the typed path
/// where no `#tile.mma_desc` attribute exists.
///
/// The layout parameter is placed on the side the role names and the opposite
/// side is given its canonical orientation, so a fragment declaring the wrong
/// orientation fails to resolve rather than being silently corrected.
///
/// An `acc` fragment names no input dtype, so one must be supplied by the
/// caller; see `descriptorFor`.
static tessera::tile::TileMmaDescAttr
descriptorFromFragment(tessera::tile::FragmentType f, StringRef input) {
  MLIRContext *ctx = f.getContext();
  if (f.isUnknown() || f.getM() <= 0 || f.getN() <= 0 || f.getK() <= 0 ||
      f.getRole().empty() || f.getLayout().empty() || f.getFamily().empty() ||
      input.empty())
    return {};
  StringRef aLayout = f.getRole() == "a" ? f.getLayout() : StringRef("row_major");
  StringRef bLayout = f.getRole() == "b" ? f.getLayout() : StringRef("col_major");
  return tessera::tile::TileMmaDescAttr::get(ctx, f.getFamily(), f.getM(),
                                             f.getN(), f.getK(), input, input,
                                             f.getAcc(), aLayout, bLayout,
                                             /*kBlocks=*/1);
}

/// Input dtypes to try when resolving an `acc` fragment, which names none.
///
/// A single fixed representative is WRONG: `resolveFragmentLayout` derives the
/// legal `k` FROM the input dtype, and that mapping is not constant. RDNA4 int4
/// takes k=32 while int8 takes k=16; gfx125x fp8 takes k=64 while f16 takes
/// k=32; CDNA spans k=8 (f32) through k=64 (fp4). Pinning "int8"/"f16" made an
/// accumulator fail to resolve on every one of those, even though the MMA
/// itself was supported.
///
/// Searching is sound because the accumulator's width does not depend on which
/// candidate matches: every branch of `resolveFragmentLayout` sets both
/// accumulator fields to `256 / waveSize`, and `waveSize` is fixed per
/// architecture. So any candidate that resolves yields the same answer; the
/// search only has to find one. Candidates preserve integer-ness, which is what
/// selects the i32-vs-f32 accumulator element type downstream.
static SmallVector<StringRef> accumulatorProbeDtypes(StringRef acc) {
  if (acc == "i32" || acc == "int32")
    return {"int8", "int4"};
  if (acc == "f32")
    return {"f16", "bf16", "e4m3", "e5m2", "f32", "fp4"};
  return {};
}

class TileFragmentTypeConverter : public TypeConverter {
public:
  explicit TileFragmentTypeConverter(StringRef arch) : arch(arch.str()) {
    // The identity conversion deliberately EXCLUDES fragments. A blanket
    // identity is a trap here: `convertType` reads `std::nullopt` from a
    // callback as "not applicable, try the next one", so an unresolvable
    // fragment fell through to identity, was declared legal, and sailed
    // through unconverted -- yielding a `tessera_rocm.wmma` with
    // `!tile.fragment` operands and a SUCCESSFUL exit code. With no callback
    // applicable, `convertType` fails, the op stays illegal, and the
    // conversion reports it.
    addConversion([](Type type) -> std::optional<Type> {
      if (isa<tessera::tile::FragmentType>(type))
        return std::nullopt;
      return type;
    });
    addConversion([this](tessera::tile::FragmentType f) -> std::optional<Type> {
      // The legacy bare spelling is not ours: converting it to itself keeps
      // every op that uses it LEGAL, so the single-shot path still sees it.
      if (f.isUnknown())
        return Type(f);
      std::optional<tessera_rocm::FragmentLayoutDescriptor> physical =
          layoutFor(f);
      if (!physical || !physical->materializationReady)
        return std::nullopt;
      MLIRContext *ctx = f.getContext();
      if (f.getRole() == "acc") {
        Type elem = accumulatorElementType(ctx, f.getAcc());
        return elem ? std::optional<Type>(VectorType::get(
                          {physical->accumulatorElementsPerLane}, elem))
                    : std::nullopt;
      }
      Type packed = packedFragmentType(ctx, f.getElem(), *physical);
      return packed ? std::optional<Type>(packed) : std::nullopt;
    });
  }

  /// The descriptor this fragment resolves under on this architecture, or null.
  ///
  /// For A/B the input dtype is stated by the type. For `acc` it is not, so the
  /// candidate list is probed — see `accumulatorProbeDtypes` for why searching
  /// is sound and why a single fixed representative is not.
  tessera::tile::TileMmaDescAttr
  descriptorFor(tessera::tile::FragmentType f) const {
    if (f.isUnknown())
      return {};
    if (f.getRole() != "acc")
      return descriptorFromFragment(f, f.getElem());
    for (StringRef candidate : accumulatorProbeDtypes(f.getAcc())) {
      tessera::tile::TileMmaDescAttr desc = descriptorFromFragment(f, candidate);
      if (desc && tessera_rocm::resolveFragmentLayout(desc, arch))
        return desc;
    }
    return {};
  }

  std::optional<tessera_rocm::FragmentLayoutDescriptor>
  layoutFor(tessera::tile::FragmentType f) const {
    tessera::tile::TileMmaDescAttr desc = descriptorFor(f);
    if (!desc)
      return std::nullopt;
    return tessera_rocm::resolveFragmentLayout(desc, arch);
  }

  StringRef getArch() const { return arch; }

private:
  std::string arch;
};

/// Wave-relative lane coordinates, identical to the legacy path's.
struct FragmentLaneCoords {
  Value lane;       // column within the 16-wide matrix row
  Value packGroup;  // input half-wave selector, forced to 0 when replicated
  Value storeGroup; // accumulator half-wave selector, never forced
};

static FragmentLaneCoords
computeLaneCoords(OpBuilder &builder, Location loc,
                  const tessera_rocm::FragmentLayoutDescriptor &physical) {
  Value tx = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value waveSize =
      arith::ConstantIndexOp::create(builder, loc, physical.waveSize);
  Value waveLane = arith::RemUIOp::create(builder, loc, tx, waveSize);
  Value c16 = arith::ConstantIndexOp::create(builder, loc, 16);
  FragmentLaneCoords coords;
  coords.lane = arith::RemUIOp::create(builder, loc, waveLane, c16);
  coords.storeGroup = arith::DivUIOp::create(builder, loc, waveLane, c16);
  coords.packGroup =
      physical.inputLaneReplication > 1
          ? Value(arith::ConstantIndexOp::create(builder, loc, 0))
          : coords.storeGroup;
  return coords;
}

/// Shared failure path: a stated fragment contract this target cannot realize
/// is an ERROR, not a quiet non-match. Returning `notifyMatchFailure` here
/// would surface as an unexplained "failed to legalize operation".
static LogicalResult emitUnresolvableFragment(Operation *op,
                                              tessera::tile::FragmentType f,
                                              StringRef arch) {
  return op->emitError(
             "ROCM_FRAGMENT_ILLEGAL_ARCH_DESCRIPTOR: no exact ")
         << arch << " fragment layout accepts the typed fragment " << f;
}

struct ConvertFragmentZero
    : public OpConversionPattern<tessera::tile::FragmentZeroOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tessera::tile::FragmentZeroOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto f = dyn_cast<tessera::tile::FragmentType>(op.getResult().getType());
    if (!f || f.isUnknown())
      return rewriter.notifyMatchFailure(op, "legacy bare fragment");
    auto vecTy = dyn_cast_or_null<VectorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!vecTy)
      return emitUnresolvableFragment(
          op, f,
          static_cast<const TileFragmentTypeConverter *>(getTypeConverter())
              ->getArch());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, vecTy, rewriter.getZeroAttr(vecTy));
    return success();
  }
};

struct ConvertFragmentPack
    : public OpConversionPattern<tessera::tile::FragmentPackOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tessera::tile::FragmentPackOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto f = dyn_cast<tessera::tile::FragmentType>(op.getResult().getType());
    if (!f || f.isUnknown())
      return rewriter.notifyMatchFailure(op, "legacy bare fragment");
    const auto *converter =
        static_cast<const TileFragmentTypeConverter *>(getTypeConverter());
    std::optional<tessera_rocm::FragmentLayoutDescriptor> physical =
        converter->layoutFor(f);
    if (!physical || !physical->materializationReady)
      return emitUnresolvableFragment(op, f, converter->getArch());
    FragmentLaneCoords coords =
        computeLaneCoords(rewriter, op.getLoc(), *physical);
    // The source `tile.view` is untouched by this conversion (`!tile.tile`
    // converts to itself), so the original operand still reaches it.
    // Role comes from the TYPE. `FragmentPackOp::verify` returns success
    // without a `role` attribute once the result is typed, so requiring the
    // attribute here rejected valid IR -- and it is precisely the shape a
    // migrated producer emits.
    FailureOr<Value> packed = materializeFragmentPack(
        op, f.getRole(), converter->descriptorFor(f), rewriter, coords.lane,
        coords.packGroup, *physical);
    if (failed(packed))
      return failure();
    Type expected = getTypeConverter()->convertType(f);
    if (packed->getType() != expected)
      return op->emitError(
                 "ROCM_FRAGMENT_TYPE_DISAGREES: the materialized fragment is ")
             << packed->getType() << " but the type converter promised "
             << expected
             << "; the pack rule and the converted type must be derived from "
                "the same place (packedFragmentType)";
    rewriter.replaceOp(op, *packed);
    return success();
  }
};

struct ConvertMMA : public OpConversionPattern<tessera::tile::MMAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tessera::tile::MMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Pair originals with converted values: the filter is on the ORIGINAL
    // type, since control tokens convert to themselves.
    SmallVector<Value> data;
    SmallVector<Type> dataTypes;
    for (auto [original, converted] :
         llvm::zip_equal(op.getInputs(), adaptor.getInputs())) {
      if (isTileControlType(original.getType()))
        continue;
      data.push_back(converted);
      dataTypes.push_back(original.getType());
    }
    if (data.size() != 3 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "not the A,B,acc -> acc form");
    auto accTy = dyn_cast<tessera::tile::FragmentType>(dataTypes[2]);
    if (!accTy || accTy.isUnknown())
      return rewriter.notifyMatchFailure(op, "legacy bare fragment");
    const auto *converter =
        static_cast<const TileFragmentTypeConverter *>(getTypeConverter());
    std::optional<tessera_rocm::FragmentLayoutDescriptor> physical =
        converter->layoutFor(accTy);
    Type resultTy =
        getTypeConverter()->convertType(op->getResult(0).getType());
    if (!physical || !physical->materializationReady || !resultTy)
      return emitUnresolvableFragment(op, accTy, converter->getArch());
    auto aTy = dyn_cast<tessera::tile::FragmentType>(dataTypes[0]);
    auto bTy = dyn_cast<tessera::tile::FragmentType>(dataTypes[1]);
    if (!aTy || !bTy)
      return rewriter.notifyMatchFailure(op, "A/B operand is not a fragment");
    // `MMAOp::verify` deliberately does NOT require A and B to agree on `elem`
    // -- `#tile.mma_desc` has always carried aType/bType separately, so a mixed
    // pair is well-formed Tile IR. ROCm cannot execute one:
    // `resolveFragmentLayout` only admits descriptors with aType == bType. Each
    // fragment resolving INDIVIDUALLY (its own descriptor uses its own dtype on
    // both sides) hid that, and the emitted wmma recorded A's dtype alone.
    if (aTy.getElem() != bTy.getElem())
      return op->emitError("ROCM_FRAGMENT_ILLEGAL_ARCH_DESCRIPTOR: ")
             << converter->getArch() << " has no mixed-input matrix form; A "
             << "states elem \"" << aTy.getElem() << "\" and B states \""
             << bTy.getElem() << "\"";
    // The accumulator resolving does NOT imply the inputs did: an `acc`
    // fragment names no input dtype, so it resolves via a representative one
    // (see `descriptorFromFragment`). On gfx1151 an e4m3 A/B pair is
    // unsupported while its f32 accumulator resolves happily -- which emitted a
    // `tessera_rocm.wmma` still holding `!tile.fragment` operands.
    for (auto [value, type] : llvm::zip_equal(data, dataTypes)) {
      if (isa<VectorType>(value.getType()))
        continue;
      auto f = dyn_cast<tessera::tile::FragmentType>(type);
      return f ? emitUnresolvableFragment(op, f, converter->getArch())
               : rewriter.notifyMatchFailure(op, "operand did not convert");
    }

    bool integer = aTy.getElem() == "int8" || aTy.getElem() == "int4";
    OperationState state(op.getLoc(), physical->matrixOp == "wmma"
                                          ? "tessera_rocm.wmma"
                                          : "tessera_rocm.mfma");
    // The accumulator is the CONVERTED third operand. The legacy path
    // synthesized a zero here and discarded whatever was passed, which is the
    // defect plan step 2b names.
    state.addOperands({data[0], data[1], data[2]});
    state.addTypes({resultTy});
    state.addAttribute("arch", rewriter.getStringAttr(converter->getArch()));
    state.addAttribute(
        "shape", rewriter.getStringAttr(("m16n16k" + Twine(aTy.getK())).str()));
    state.addAttribute("accum",
                       rewriter.getStringAttr(integer ? "i32" : "f32"));
    state.addAttribute("input_dtype", rewriter.getStringAttr(aTy.getElem()));
    state.addAttribute("source", rewriter.getStringAttr("tile.fragment_pack"));
    state.addAttribute("fragment_family",
                       rewriter.getStringAttr(physical->familyName));
    state.addAttribute(
        "fragment_input_format",
        rewriter.getStringAttr(
            tessera_rocm::registerFormatName(physical->inputFormat)));
    state.addAttribute("fragment_wave_size",
                       rewriter.getI64IntegerAttr(physical->waveSize));
    state.addAttribute("fragment_intrinsic_abi",
                       rewriter.getStringAttr(physical->intrinsicABI));
    state.addAttribute("ordinal", rewriter.getI64IntegerAttr(0));
    Operation *lowered = rewriter.create(state);
    rewriter.replaceOp(op, lowered->getResults());
    return success();
  }
};

/// `fragment_unpack` produces a `!tile.tile`, which has no physical form of its
/// own — the value only becomes real at the `tile.store` that consumes it. So
/// the pair lowers together. This is still a LOCAL two-op match, not a chain
/// walk: it looks only at its own single use and never at its producer, which
/// is what lets the accumulator arrive from a loop.
struct ConvertFragmentUnpackStore
    : public OpConversionPattern<tessera::tile::FragmentUnpackOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tessera::tile::FragmentUnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto f =
        dyn_cast<tessera::tile::FragmentType>(op.getInputs().front().getType());
    if (!f || f.isUnknown())
      return rewriter.notifyMatchFailure(op, "legacy bare fragment");
    if (!op.getResult().hasOneUse())
      return op->emitError(
          "ROCM_FRAGMENT_UNPACK_UNCONSUMED: a typed fragment_unpack must have "
          "exactly one tile.store consumer on this target");
    auto store = dyn_cast<tessera::tile::StoreOp>(
        *op.getResult().getUsers().begin());
    if (!store)
      return op->emitError(
          "ROCM_FRAGMENT_UNPACK_UNCONSUMED: a typed fragment_unpack must be "
          "consumed by tile.store on this target");
    const auto *converter =
        static_cast<const TileFragmentTypeConverter *>(getTypeConverter());
    std::optional<tessera_rocm::FragmentLayoutDescriptor> physical =
        converter->layoutFor(f);
    if (!physical || !physical->materializationReady)
      return emitUnresolvableFragment(op, f, converter->getArch());

    rewriter.setInsertionPoint(store);
    FragmentLaneCoords coords =
        computeLaneCoords(rewriter, store.getLoc(), *physical);
    if (failed(materializeFragmentStore(
            adaptor.getInputs().front(), converter->descriptorFor(f), op, store,
            rewriter, coords.lane, coords.storeGroup, *physical)))
      return failure();
    rewriter.eraseOp(store);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

/// Legalize every op that names a STATED `!tile.fragment` contract. Returns
/// failure only when a stated contract could not be realized.
static LogicalResult convertTypedFragments(Operation *root, StringRef arch) {
  MLIRContext *ctx = root->getContext();
  TileFragmentTypeConverter converter(arch);

  // Nothing to do unless a stated fragment contract is actually present. This
  // keeps the whole conversion off the legacy attribute-form path rather than
  // relying on every pattern to decline it.
  bool hasTypedFragment = false;
  root->walk([&](Operation *op) {
    auto typed = [](Type t) {
      auto f = dyn_cast<tessera::tile::FragmentType>(t);
      return f && !f.isUnknown();
    };
    if (llvm::any_of(op->getOperandTypes(), typed) ||
        llvm::any_of(op->getResultTypes(), typed)) {
      hasTypedFragment = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!hasTypedFragment)
    return success();

  RewritePatternSet patterns(ctx);
  patterns.add<ConvertFragmentZero, ConvertFragmentPack, ConvertMMA,
               ConvertFragmentUnpackStore>(converter, ctx);

  ConversionTarget target(*ctx);
  target.markUnknownOpDynamicallyLegal(
      [&converter](Operation *op) { return converter.isLegal(op); });
  // `scf.for` iter-args, `scf.yield`, and friends — the case that motivated
  // this whole item is handled entirely by upstream.
  scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                       target);
  if (failed(applyPartialConversion(root, target, std::move(patterns))))
    return failure();

  // The pointer-backed source views are dead once the fragments materialize:
  // the physical loads address the base memref directly. `tile.view` is `Pure`,
  // but `applyPartialConversion` does not DCE, and a Tile-dialect op surviving
  // into Target IR is a level-boundary leak (Decision #19) -- the legacy path
  // erased them explicitly and so must this one.
  SmallVector<tessera::tile::ViewOp> deadViews;
  root->walk([&](tessera::tile::ViewOp view) {
    if (view.getResult().use_empty())
      deadViews.push_back(view);
  });
  for (tessera::tile::ViewOp view : deadViews)
    view->erase();
  return success();
}

// "fnuz" (CDNA 3) | "ocp" (CDNA 4 / RDNA 4 / gfx125x) | "none" (no FP8 path).
static llvm::StringRef fp8SemanticsForArch(llvm::StringRef arch) {
  if (arch == "gfx940" || arch == "gfx942")
    return "fnuz"; // CDNA 3 — E4M3FNUZ / E5M2FNUZ
  if (arch == "gfx950" || arch == "gfx1200" || arch == "gfx1201" ||
      arch == "gfx1250" || arch == "gfx1251")
    return "ocp"; // CDNA 4 / RDNA 4 / gfx125x — OCP E4M3 / E5M2
  return "none";  // gfx90a, gfx1100, gfx1151 — no FP8 matrix path
}

static LogicalResult rejectUnconsumedStoragePack(Operation *op) {
  if (!op->hasAttr("tessera.storage_packed"))
    return success();
  if (op->hasAttr("tessera.storage_pack"))
    return success();
  op->emitOpError(
      "ROCM_LOWERING_UNCONSUMED_STORAGE_PACK: packed low-precision storage "
      "reached ROCm lowering without a tessera.storage_pack consumer "
      "descriptor; run tessera-storage-pack-consume or add an explicit ROCm "
      "consumer before lowering.");
  return failure();
}

static void copyAttrIfPresent(OperationState &state, Operation *op,
                              StringRef name) {
  if (Attribute attr = op->getAttr(name))
    state.addAttribute(name, attr);
}

static bool layoutHasLdsAxis(tessera::tile::TileLayoutAttr layout) {
  for (StringAttr axis : layout.getShardAxes())
    if (axis.getValue() == "lds")
      return true;
  return false;
}

// COMPOSE the shared Tile rule with this backend's own control type rather
// than keeping a private copy. The Tile half now lives in
// `tessera::tile::isTileControlType` -- it used to exist only here, which is
// why the ODS verifier counted raw operands and the typed `tile.mma` form was
// unreachable for any producer that also carried a warp-spec token.
// Tile IR cannot see `tessera_rocm::TokenType`, so the backend adds it here.
static bool isTileControlType(Type type) {
  return tessera::tile::isTileControlType(type) ||
         isa<mlir::tessera_rocm::TokenType>(type);
}

static SmallVector<Value> dataOperands(Operation *op) {
  SmallVector<Value> values;
  for (Value operand : op->getOperands())
    if (!isTileControlType(operand.getType()))
      values.push_back(operand);
  return values;
}

static Value tileBufferRoot(Operation *op) {
  Value buffer;
  for (Value operand : op->getOperands())
    if (isa<tessera::tile::BufferType>(operand.getType())) {
      buffer = operand;
      break;
    }
  while (buffer) {
    Operation *def = buffer.getDefiningOp();
    if (!def)
      break;
    Value parent;
    for (Value operand : def->getOperands())
      if (isa<tessera::tile::BufferType>(operand.getType())) {
        parent = operand;
        break;
      }
    if (!parent)
      break;
    buffer = parent;
  }
  return buffer;
}

// Validate and consume the shared rank-4 batch/head distribution plus its
// canonical KV-block recurrence as one gfx1151 launch directive.  The shared
// loop owns attention semantics and SSA dependencies; this adapter selects only
// the AMD physical schedule, packaging, and launch ABI.
static LogicalResult materializeCanonicalStreamingAttention(
    ModuleOp module, func::FuncOp function, StringRef arch) {
  if (arch != "gfx1151") {
    function.emitError(
        "ROCm canonical streaming-attention consumption is currently "
        "exact-device gated to gfx1151");
    return failure();
  }

  SmallVector<scf::ForOp> streamingLoops;
  function.walk([&](scf::ForOp loop) {
    if (loop->hasAttr("tessera.streaming_attention"))
      streamingLoops.push_back(loop);
  });
  if (streamingLoops.size() != 1) {
    function.emitError(
        "ROCm canonical attention requires exactly one KV-block scf.for per "
        "rank-4 launch function");
    return failure();
  }
  scf::ForOp kvLoop = streamingLoops.front();
  auto headLoop = kvLoop->getParentOfType<scf::ForOp>();
  auto batchLoop =
      headLoop ? headLoop->getParentOfType<scf::ForOp>() : scf::ForOp();
  auto headDistribution =
      headLoop ? headLoop->getAttrOfType<StringAttr>(
                     "tessera.attention_distribution")
               : StringAttr();
  auto batchDistribution =
      batchLoop ? batchLoop->getAttrOfType<StringAttr>(
                      "tessera.attention_distribution")
                : StringAttr();
  if (!headDistribution || headDistribution.getValue() != "query_head" ||
      !batchDistribution || batchDistribution.getValue() != "batch") {
    kvLoop.emitError(
        "ROCm canonical attention requires explicit rank-4 batch and "
        "query-head scf.for distribution");
    return failure();
  }

  if ((function.getNumArguments() != 3 &&
       function.getNumArguments() != 4) ||
      function.getNumResults() != 1) {
    function.emitError(
        "ROCm canonical streaming attention requires Q, K, V, optional bias, "
        "and one output");
    return failure();
  }
  auto qType = dyn_cast<RankedTensorType>(function.getArgument(0).getType());
  auto kType = dyn_cast<RankedTensorType>(function.getArgument(1).getType());
  auto vType = dyn_cast<RankedTensorType>(function.getArgument(2).getType());
  auto biasType =
      function.getNumArguments() == 4
          ? dyn_cast<RankedTensorType>(function.getArgument(3).getType())
          : RankedTensorType();
  auto outType = dyn_cast<RankedTensorType>(function.getResultTypes().front());
  if (!qType || !kType || !vType || !outType || !qType.hasStaticShape() ||
      !kType.hasStaticShape() || !vType.hasStaticShape() ||
      !outType.hasStaticShape() || qType.getRank() != 4 ||
      kType.getRank() != 4 || vType.getRank() != 4 ||
      outType.getRank() != 4) {
    function.emitError(
        "ROCm canonical streaming attention currently requires static rank-4 "
        "Q/K/V/output tensors");
    return failure();
  }

  int64_t batch = qType.getDimSize(0);
  int64_t queryHeads = qType.getDimSize(1);
  int64_t kvHeads = kType.getDimSize(1);
  int64_t queryRows = qType.getDimSize(2);
  int64_t keyRows = kType.getDimSize(2);
  int64_t headDim = qType.getDimSize(3);
  int64_t valueDim = vType.getDimSize(3);
  if (batch <= 0 || queryHeads <= 0 || kvHeads <= 0 ||
      queryHeads % kvHeads != 0 || headDim <= 0 || headDim % 16 != 0 ||
      headDim != valueDim || kType.getDimSize(0) != batch ||
      vType.getDimSize(0) != batch || vType.getDimSize(1) != kvHeads ||
      kType.getDimSize(2) != keyRows || vType.getDimSize(2) != keyRows ||
      kType.getDimSize(3) != headDim ||
      outType.getDimSize(0) != batch ||
      outType.getDimSize(1) != queryHeads ||
      outType.getDimSize(2) != queryRows ||
      outType.getDimSize(3) != valueDim ||
      !outType.getElementType().isF32()) {
    function.emitError(
        "ROCm canonical streaming attention requires valid static GQA shapes, "
        "f32 output, and equal WMMA-compatible head/value dimensions");
    return failure();
  }

  StringRef storage;
  if (qType.getElementType().isF16())
    storage = "f16";
  else if (qType.getElementType().isBF16())
    storage = "bf16";
  else {
    function.emitError(
        "gfx1151 canonical streaming attention requires f16 or bf16 storage");
    return failure();
  }
  if (kType.getElementType() != qType.getElementType() ||
      vType.getElementType() != qType.getElementType()) {
    function.emitError("ROCm canonical attention Q/K/V storage must match");
    return failure();
  }
  if (biasType &&
      (!biasType.hasStaticShape() ||
       (biasType.getRank() != 3 && biasType.getRank() != 4) ||
       !biasType.getElementType().isF32() ||
       (biasType.getDimSize(0) != 1 &&
        biasType.getDimSize(0) != batch) ||
       (biasType.getRank() == 4 && biasType.getDimSize(1) != 1 &&
        biasType.getDimSize(1) != queryHeads) ||
       biasType.getDimSize(biasType.getRank() - 2) != queryRows ||
       biasType.getDimSize(biasType.getRank() - 1) != keyRows)) {
    function.emitError(
        "ROCm canonical attention bias must be f32 [B|1,Sq,Sk] or "
        "[B|1,Hq|1,Sq,Sk]");
    return failure();
  }

  if (kvLoop.getNumRegionIterArgs() != 6 ||
      !isa<tessera::tile::PipelineStateType>(
          kvLoop.getRegionIterArg(3).getType()) ||
      !isa<tessera::tile::PipelineStateType>(
          kvLoop.getRegionIterArg(4).getType()) ||
      !kvLoop.getRegionIterArg(5).getType().isIndex()) {
    kvLoop.emitError(
        "ROCm canonical attention requires producer/consumer "
        "!tile.pipeline_state and an absolute boundary offset");
    return failure();
  }

  unsigned asyncCopies = 0;
  unsigned tokenCopies = 0;
  unsigned waits = 0;
  unsigned semanticAdvances = 0;
  bool hasScaledDotProduct = false;
  bool hasStreamingUpdate = false;
  Operation *boundaryMask = nullptr;
  Operation *blockDropout = nullptr;
  Operation *scoreBias = nullptr;
  Operation *softcap = nullptr;
  kvLoop.walk([&](Operation *nested) {
    StringRef name = nested->getName().getStringRef();
    if (name == "tile.async_copy") {
      ++asyncCopies;
      if (llvm::any_of(nested->getResults(), [](Value value) {
            return isa<tessera::tile::AsyncTokenType>(value.getType());
          }))
        ++tokenCopies;
    } else if (name == "tile.wait_async") {
      ++waits;
    } else if (name == "tile.pipeline_advance") {
      // The shared recurrence owns exactly one producer and one consumer
      // advance. ROCMWaveLdsPipeline may insert additional target-stamped
      // physical advances while threading allocated buffers through async
      // copies; those refine rather than replace the semantic state machine.
      if (!nested->hasAttr("target"))
        ++semanticAdvances;
    } else if (name == "tessera_attn.scaled_dot_product") {
      hasScaledDotProduct = true;
    } else if (name == "tessera_attn.streaming_update") {
      hasStreamingUpdate = true;
    } else if (name == "tessera_attn.boundary_mask") {
      boundaryMask = nested;
    } else if (name == "tessera_attn.block_dropout") {
      blockDropout = nested;
    } else if (name == "tessera_attn.score_bias") {
      scoreBias = nested;
    } else if (name == "tessera_attn.softcap") {
      softcap = nested;
    }
  });
  if (asyncCopies != 2 || tokenCopies != 2 || waits != 1 ||
      semanticAdvances != 2 || !hasScaledDotProduct ||
      !hasStreamingUpdate) {
    kvLoop.emitError(
        "ROCm canonical attention requires two token-producing K/V copies, "
        "one wait, two pipeline advances, scaled-dot-product, and streaming "
        "update");
    return failure();
  }
  if ((biasType && !scoreBias) || (!biasType && scoreBias)) {
    kvLoop.emitError(
        "ROCm canonical attention bias argument and score_bias recurrence "
        "must be present together");
    return failure();
  }
  if (softcap) {
    auto cap = softcap->getAttrOfType<FloatAttr>("cap");
    if (!cap || !cap.getValue().isFinite() ||
        cap.getValueAsDouble() <= 0.0) {
      softcap->emitError(
          "ROCm canonical attention softcap requires a finite positive cap");
      return failure();
    }
  }

  auto kvBlock = kvLoop->getAttrOfType<IntegerAttr>("tessera.kv_block");
  auto logicalSk = kvLoop->getAttrOfType<IntegerAttr>("tessera.logical_sk");
  if (!kvBlock || kvBlock.getInt() <= 0 || !logicalSk ||
      logicalSk.getInt() != keyRows) {
    kvLoop.emitError(
        "ROCm canonical attention requires explicit KV block and logical Sk");
    return failure();
  }

  bool causal = false;
  int64_t windowLeft = -1;
  int64_t windowRight = -1;
  if (boundaryMask) {
    auto causalAttr = boundaryMask->getAttrOfType<BoolAttr>("causal");
    auto leftAttr = boundaryMask->getAttrOfType<IntegerAttr>("window_left");
    auto rightAttr = boundaryMask->getAttrOfType<IntegerAttr>("window_right");
    if (!causalAttr || !leftAttr || !rightAttr) {
      boundaryMask->emitError(
          "ROCm canonical attention boundary mask is missing policy fields");
      return failure();
    }
    causal = causalAttr.getValue();
    windowLeft = leftAttr.getInt();
    windowRight = rightAttr.getInt();
  }
  bool slidingWindow = windowLeft >= 0 || windowRight >= 0;
  if (slidingWindow &&
      (!causal || windowLeft < 0 || windowRight != 0)) {
    kvLoop.emitError(
        "gfx1151 WMMA streaming attention currently supports only causal "
        "left-window schedules");
    return failure();
  }

  bool hasCall = false;
  module.walk([&](func::CallOp call) {
    if (call.getCallee() == function.getSymName())
      hasCall = true;
  });
  if (hasCall) {
    function.emitError(
        "ROCm canonical streaming launch function must not have internal "
        "callers before Target-IR packaging");
    return failure();
  }

  OpBuilder builder(function);
  OperationState state(function.getLoc(), "tessera_rocm.flash_attn");
  state.addAttribute("name", builder.getStringAttr(function.getSymName()));
  state.addAttribute("head_dim", builder.getI64IntegerAttr(headDim));
  state.addAttribute("dtype", builder.getStringAttr(storage));
  state.addAttribute("gqa", builder.getBoolAttr(queryHeads != kvHeads));
  state.addAttribute("sliding_window",
                     builder.getBoolAttr(slidingWindow));
  state.addAttribute("logit_softcap",
                     builder.getBoolAttr(softcap != nullptr));
  state.addAttribute("attn_bias",
                     builder.getBoolAttr(scoreBias != nullptr));
  state.addAttribute("dropout", builder.getBoolAttr(blockDropout != nullptr));
  state.addAttribute("causal", builder.getBoolAttr(causal));
  state.addAttribute("arch", builder.getStringAttr(arch));
  state.addAttribute("source",
                     builder.getStringAttr("canonical_rank4_kv_scf_for"));
  state.addAttribute(
      "schedule",
      builder.getStringAttr("gfx1151_wmma_streaming_attention"));
  state.addAttribute("canonical_kv_loop", builder.getBoolAttr(true));
  // A training package explicitly selects saved-LSE or recompute. Inference
  // forward has no checkpoint attribute and retains the existing ABI.
  auto lseCheckpoint =
      function->getAttrOfType<StringAttr>("tessera.lse_checkpoint");
  bool saveLse = lseCheckpoint && lseCheckpoint.getValue() == "saved";
  state.addAttribute("save_lse", builder.getBoolAttr(saveLse));
  state.addAttribute("rank4_distribution", builder.getBoolAttr(true));
  state.addAttribute("ssa_pipeline", builder.getBoolAttr(true));
  state.addAttribute("kv_block", kvBlock);
  state.addAttribute("logical_sk", logicalSk);
  state.addAttribute("batch", builder.getI64IntegerAttr(batch));
  state.addAttribute("query_heads",
                     builder.getI64IntegerAttr(queryHeads));
  state.addAttribute("kv_heads", builder.getI64IntegerAttr(kvHeads));
  state.addAttribute("query_rows", builder.getI64IntegerAttr(queryRows));
  state.addAttribute("key_rows", builder.getI64IntegerAttr(keyRows));
  builder.create(state);
  function.erase();
  return success();
}

static StringRef attentionBackwardPhase(scf::ForOp loop) {
  auto attr =
      loop->getAttrOfType<StringAttr>("tessera.attention_backward_phase");
  return attr ? attr.getValue() : StringRef();
}

static bool hasAttentionBackwardParentPhases(
    Operation *op, ArrayRef<StringRef> expected) {
  Operation *parent = op->getParentOp();
  for (StringRef phase : expected) {
    auto loop = dyn_cast_or_null<scf::ForOp>(parent);
    if (!loop || attentionBackwardPhase(loop) != phase)
      return false;
    parent = parent->getParentOp();
  }
  return true;
}

// Consume the target-neutral tensor-valued VJP loops as one gfx1151 backward
// directive. The phase ops own the mathematical recurrence and scf.for owns
// distribution/workspace/reduction ordering; this adapter validates those
// invariants before selecting the AMD five-entry physical program.
static LogicalResult materializeCanonicalAttentionBackward(
    ModuleOp module, func::FuncOp function, StringRef arch) {
  if (arch != "gfx1151") {
    function.emitError(
        "ROCm canonical attention-backward consumption is currently "
        "exact-device gated to gfx1151");
    return failure();
  }
  if ((function.getNumArguments() != 4 &&
       function.getNumArguments() != 5) ||
      function.getNumResults() != 3) {
    function.emitError(
        "ROCm canonical attention backward requires dO, Q, K, V, optional "
        "bias, and dQ/dK/dV results");
    return failure();
  }

  SmallVector<Operation *> dqBlocks;
  SmallVector<Operation *> dkdvBlocks;
  SmallVector<Operation *> reduceBlocks;
  function.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "tessera_attn.backward_dq_block")
      dqBlocks.push_back(op);
    else if (name == "tessera_attn.backward_dkdv_block")
      dkdvBlocks.push_back(op);
    else if (name == "tessera_attn.backward_reduce_split")
      reduceBlocks.push_back(op);
  });
  if (dqBlocks.size() != 1 || dkdvBlocks.size() != 1 ||
      reduceBlocks.size() != 1) {
    function.emitError(
        "ROCm canonical attention backward requires exactly one dQ block, "
        "one split dK/dV block, and one fixed-order reduction body");
    return failure();
  }
  Operation *dqBlock = dqBlocks.front();
  Operation *dkdvBlock = dkdvBlocks.front();
  Operation *reduceBlock = reduceBlocks.front();
  if (!hasAttentionBackwardParentPhases(
          dqBlock, {"dq.key_block", "dq.query_block", "dq.query_head",
                    "dq.batch"}) ||
      !hasAttentionBackwardParentPhases(
          dkdvBlock,
          {"dkdv.key_block", "dkdv.query_block", "dkdv.split",
           "dkdv.kv_head", "dkdv.batch"}) ||
      !hasAttentionBackwardParentPhases(
          reduceBlock,
          {"reduce.fixed_order_split", "reduce.kv_head", "reduce.batch"})) {
    function.emitError(
        "ROCm canonical attention backward phase ops are not nested in the "
        "verified shared loop order");
    return failure();
  }

  auto splitLoop = dkdvBlock->getParentOfType<scf::ForOp>();
  splitLoop = splitLoop ? splitLoop->getParentOfType<scf::ForOp>()
                        : scf::ForOp();
  splitLoop = splitLoop ? splitLoop->getParentOfType<scf::ForOp>()
                        : scf::ForOp();
  auto reduceLoop = reduceBlock->getParentOfType<scf::ForOp>();
  auto workspaceOwner =
      splitLoop
          ? splitLoop->getAttrOfType<StringAttr>("tessera.workspace_owner")
          : StringAttr();
  auto reductionOrder =
      reduceLoop
          ? reduceLoop->getAttrOfType<StringAttr>("tessera.reduction_order")
          : StringAttr();
  if (!workspaceOwner || workspaceOwner.getValue() != "program_launch" ||
      !reductionOrder ||
      reductionOrder.getValue() != "ascending_split") {
    function.emitError(
        "ROCm canonical attention backward requires launch-owned split "
        "workspace and ascending split reduction");
    return failure();
  }
  if (dqBlock->getNumOperands() != 10 ||
      dkdvBlock->getNumOperands() != 12 ||
      reduceBlock->getNumOperands() != 7 ||
      dqBlock->getNumResults() != 1 ||
      dkdvBlock->getNumResults() != 2 ||
      reduceBlock->getNumResults() != 2) {
    function.emitError(
        "ROCm canonical attention backward phase signatures disagree with "
        "the shared tensor contract");
    return failure();
  }
  for (unsigned i = 0; i < 5; ++i) {
    if (dqBlock->getOperand(i) != dkdvBlock->getOperand(i)) {
      function.emitError(
          "ROCm canonical attention backward phase bodies must share the "
          "same dO/Q/K/V/bias SSA values");
      return failure();
    }
  }
  if (dqBlock->getAttrDictionary() != dkdvBlock->getAttrDictionary() ||
      dqBlock->getAttrDictionary() != reduceBlock->getAttrDictionary()) {
    function.emitError(
        "ROCm canonical attention backward phase bodies must carry one "
        "identical shared semantic policy");
    return failure();
  }
  auto partialOwner =
      reduceBlock->getOperand(0).getDefiningOp<scf::ForOp>();
  if (!partialOwner ||
      reduceBlock->getOperand(1).getDefiningOp() !=
          partialOwner.getOperation() ||
      attentionBackwardPhase(partialOwner) != "dkdv.batch" ||
      !partialOwner->isProperAncestor(dkdvBlock)) {
    function.emitError(
        "ROCm canonical attention backward reduction must consume the "
        "SSA results of the split dK/dV batch loop");
    return failure();
  }
  auto returnOp =
      dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
  auto dqOwner =
      returnOp && returnOp.getNumOperands() == 3
          ? returnOp.getOperand(0).getDefiningOp<scf::ForOp>()
          : scf::ForOp();
  auto reduceOwner =
      returnOp && returnOp.getNumOperands() == 3
          ? returnOp.getOperand(1).getDefiningOp<scf::ForOp>()
          : scf::ForOp();
  if (!dqOwner || attentionBackwardPhase(dqOwner) != "dq.batch" ||
      !reduceOwner ||
      attentionBackwardPhase(reduceOwner) != "reduce.batch" ||
      returnOp.getOperand(2).getDefiningOp() != reduceOwner.getOperation()) {
    function.emitError(
        "ROCm canonical attention backward function results must be the "
        "shared dQ and fixed-reduction loop results");
    return failure();
  }

  auto dOutType =
      dyn_cast<RankedTensorType>(function.getArgument(0).getType());
  auto qType = dyn_cast<RankedTensorType>(function.getArgument(1).getType());
  auto kType = dyn_cast<RankedTensorType>(function.getArgument(2).getType());
  auto vType = dyn_cast<RankedTensorType>(function.getArgument(3).getType());
  auto dQType = dyn_cast<RankedTensorType>(function.getResultTypes()[0]);
  auto dKType = dyn_cast<RankedTensorType>(function.getResultTypes()[1]);
  auto dVType = dyn_cast<RankedTensorType>(function.getResultTypes()[2]);
  auto validStaticRank4 = [](RankedTensorType type) {
    return type && type.hasStaticShape() && type.getRank() == 4;
  };
  if (!validStaticRank4(dOutType) || !validStaticRank4(qType) ||
      !validStaticRank4(kType) || !validStaticRank4(vType) ||
      !validStaticRank4(dQType) || !validStaticRank4(dKType) ||
      !validStaticRank4(dVType)) {
    function.emitError(
        "ROCm canonical attention backward currently requires static rank-4 "
        "dO/Q/K/V/dQ/dK/dV tensors");
    return failure();
  }

  int64_t batch = qType.getDimSize(0);
  int64_t queryHeads = qType.getDimSize(1);
  int64_t queryRows = qType.getDimSize(2);
  int64_t headDim = qType.getDimSize(3);
  int64_t kvHeads = kType.getDimSize(1);
  int64_t keyRows = kType.getDimSize(2);
  int64_t valueDim = vType.getDimSize(3);
  SmallVector<int64_t> expectedDOut{
      batch, queryHeads, queryRows, valueDim};
  if (batch <= 0 || queryHeads <= 0 || kvHeads <= 0 ||
      queryHeads % kvHeads != 0 || queryRows <= 0 || keyRows <= 0 ||
      headDim <= 0 || headDim % 16 != 0 || headDim != valueDim ||
      kType.getDimSize(0) != batch || vType.getDimSize(0) != batch ||
      vType.getDimSize(1) != kvHeads ||
      kType.getDimSize(2) != keyRows ||
      vType.getDimSize(2) != keyRows ||
      kType.getDimSize(3) != headDim ||
      dOutType.getShape() != ArrayRef<int64_t>(expectedDOut) ||
      dQType.getShape() != qType.getShape() ||
      dKType.getShape() != kType.getShape() ||
      dVType.getShape() != vType.getShape() ||
      !dQType.getElementType().isF32() ||
      !dKType.getElementType().isF32() ||
      !dVType.getElementType().isF32()) {
    function.emitError(
        "ROCm canonical attention backward requires valid static GQA shapes, "
        "f32 gradients, and equal WMMA-compatible head/value dimensions");
    return failure();
  }
  StringRef storage;
  if (qType.getElementType().isF16())
    storage = "f16";
  else if (qType.getElementType().isBF16())
    storage = "bf16";
  else {
    function.emitError(
        "gfx1151 canonical attention backward requires f16 or bf16 storage");
    return failure();
  }
  if (dOutType.getElementType() != qType.getElementType() ||
      kType.getElementType() != qType.getElementType() ||
      vType.getElementType() != qType.getElementType()) {
    function.emitError(
        "ROCm canonical attention backward dO/Q/K/V storage must match");
    return failure();
  }

  Value biasValue = dqBlock->getOperand(4);
  bool hasBias = function.getNumArguments() == 5;
  auto biasType = dyn_cast<RankedTensorType>(biasValue.getType());
  if (!biasType || !biasType.hasStaticShape() ||
      (biasType.getRank() != 3 && biasType.getRank() != 4) ||
      !biasType.getElementType().isF32()) {
    function.emitError(
        "ROCm canonical attention backward bias must be a static rank-3/4 "
        "f32 tensor");
    return failure();
  }
  int64_t biasSequenceDim = biasType.getRank() - 2;
  if ((biasType.getDimSize(0) != 1 &&
       biasType.getDimSize(0) != batch) ||
      (biasType.getRank() == 4 && biasType.getDimSize(1) != 1 &&
       biasType.getDimSize(1) != queryHeads) ||
      biasType.getDimSize(biasSequenceDim) != queryRows ||
      biasType.getDimSize(biasSequenceDim + 1) != keyRows) {
    function.emitError(
        "ROCm canonical attention backward bias must be "
        "[B|1,Sq,Sk] or [B|1,Hq|1,Sq,Sk]");
    return failure();
  }
  if (hasBias) {
    if (biasValue != function.getArgument(4)) {
      function.emitError(
          "ROCm canonical attention backward bias phase operand must be the "
          "launch bias argument");
      return failure();
    }
  } else {
    auto constant = biasValue.getDefiningOp<arith::ConstantOp>();
    auto values =
        constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                 : DenseElementsAttr();
    if (!values || !values.isSplat() ||
        !values.getSplatValue<APFloat>().isZero()) {
      function.emitError(
          "ROCm bias-free canonical attention backward requires the shared "
          "zero-bias tensor operand");
      return failure();
    }
  }

  auto splitCount = dqBlock->getAttrOfType<IntegerAttr>("split_count");
  auto queryBlock = dqBlock->getAttrOfType<IntegerAttr>("query_block");
  auto keyBlock = dqBlock->getAttrOfType<IntegerAttr>("key_block");
  auto scale = dqBlock->getAttrOfType<FloatAttr>("scale");
  auto causal = dqBlock->getAttrOfType<BoolAttr>("causal");
  auto windowLeft = dqBlock->getAttrOfType<IntegerAttr>("window_left");
  auto windowRight = dqBlock->getAttrOfType<IntegerAttr>("window_right");
  auto softcap = dqBlock->getAttrOfType<FloatAttr>("softcap");
  auto dropout = dqBlock->getAttrOfType<FloatAttr>("dropout_p");
  auto dropoutSeed =
      dqBlock->getAttrOfType<IntegerAttr>("dropout_seed");
  bool windowCompatible =
      windowLeft && windowRight &&
      ((windowLeft.getInt() == -1 && windowRight.getInt() == -1) ||
       (causal && causal.getValue() && windowLeft.getInt() >= 0 &&
        windowRight.getInt() == 0));
  if (!splitCount || splitCount.getInt() != 2 || !queryBlock ||
      queryBlock.getInt() != 16 || !keyBlock || keyBlock.getInt() != 16 ||
      !scale || !scale.getValue().isFinite() ||
      scale.getValueAsDouble() <= 0.0 || !causal || !softcap ||
      !softcap.getValue().isFinite() || softcap.getValueAsDouble() < 0.0 ||
      !dropout || !dropout.getValue().isFinite() ||
      dropout.getValueAsDouble() < 0.0 ||
      dropout.getValueAsDouble() >= 1.0 || !dropoutSeed ||
      dropoutSeed.getInt() < 0 || !windowCompatible) {
    function.emitError(
        "ROCm canonical attention backward requires the verified two-split "
        "16x16 block contract and supported scale/mask/dropout policy");
    return failure();
  }

  auto partialK =
      dyn_cast<RankedTensorType>(dkdvBlock->getResult(0).getType());
  auto partialV =
      dyn_cast<RankedTensorType>(dkdvBlock->getResult(1).getType());
  SmallVector<int64_t> expectedPartialK{2, batch, kvHeads, keyRows, headDim};
  SmallVector<int64_t> expectedPartialV{2, batch, kvHeads, keyRows, valueDim};
  if (!partialK || !partialV ||
      partialK.getShape() != ArrayRef<int64_t>(expectedPartialK) ||
      partialV.getShape() != ArrayRef<int64_t>(expectedPartialV) ||
      !partialK.getElementType().isF32() ||
      !partialV.getElementType().isF32()) {
    function.emitError(
        "ROCm canonical attention backward split partials must be explicit "
        "[2,B,Hkv,Sk,D] f32 tensors");
    return failure();
  }

  bool hasCall = false;
  module.walk([&](func::CallOp call) {
    if (call.getCallee() == function.getSymName())
      hasCall = true;
  });
  if (hasCall) {
    function.emitError(
        "ROCm canonical attention-backward launch function must not have "
        "internal callers before Target-IR packaging");
    return failure();
  }

  OpBuilder builder(function);
  OperationState state(function.getLoc(), "tessera_rocm.flash_attn_bwd");
  state.addAttribute("name", builder.getStringAttr(function.getSymName()));
  state.addAttribute("head_dim", builder.getI64IntegerAttr(headDim));
  state.addAttribute("dtype", builder.getStringAttr(storage));
  state.addAttribute("gqa", builder.getBoolAttr(queryHeads != kvHeads));
  state.addAttribute("sliding_window",
                     builder.getBoolAttr(windowLeft.getInt() >= 0));
  state.addAttribute("logit_softcap",
                     builder.getBoolAttr(softcap.getValueAsDouble() > 0.0));
  state.addAttribute("dropout",
                     builder.getBoolAttr(dropout.getValueAsDouble() > 0.0));
  state.addAttribute("attn_bias", builder.getBoolAttr(hasBias));
  state.addAttribute("split_reduced", builder.getBoolAttr(true));
  state.addAttribute("arch", builder.getStringAttr(arch));
  state.addAttribute(
      "source", builder.getStringAttr("canonical_tensor_backward_scf_for"));
  state.addAttribute(
      "schedule",
      builder.getStringAttr("gfx1151_wmma_backward_split_reduced"));
  state.addAttribute("canonical_phase_loops", builder.getBoolAttr(true));
  auto lseCheckpoint =
      function->getAttrOfType<StringAttr>("tessera.lse_checkpoint");
  state.addAttribute(
      "saved_lse",
      builder.getBoolAttr(lseCheckpoint &&
                          lseCheckpoint.getValue() == "saved"));
  state.addAttribute("workspace_owner",
                     builder.getStringAttr("program_launch"));
  state.addAttribute("reduction_order",
                     builder.getStringAttr("ascending_split"));
  state.addAttribute("split_count", splitCount);
  state.addAttribute("query_block", queryBlock);
  state.addAttribute("key_block", keyBlock);
  state.addAttribute("batch", builder.getI64IntegerAttr(batch));
  state.addAttribute("query_heads",
                     builder.getI64IntegerAttr(queryHeads));
  state.addAttribute("kv_heads", builder.getI64IntegerAttr(kvHeads));
  state.addAttribute("query_rows", builder.getI64IntegerAttr(queryRows));
  state.addAttribute("key_rows", builder.getI64IntegerAttr(keyRows));
  builder.create(state);
  function.erase();
  return success();
}

struct LowerTileToROCMPass
    : PassWrapper<LowerTileToROCMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToROCMPass)

  // A pass carrying ``Option`` members is not implicitly copyable, but the pass
  // manager clones passes — provide the canonical copy ctor that re-inits the
  // options via their in-class initializers.
  LowerTileToROCMPass() = default;
  LowerTileToROCMPass(const LowerTileToROCMPass &other)
      : PassWrapper<LowerTileToROCMPass, OperationPass<ModuleOp>>(other) {}

  StringRef getArgument() const final { return "lower-tile-to-rocm"; }
  StringRef getDescription() const final {
    return "Lower Tessera Tile IR matmul movement contracts to ROCm Target IR";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect,
                    LLVM::LLVMDialect, math::MathDialect, scf::SCFDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    tessera::tile::TesseraTileDialect,
                    mlir::tessera_rocm::TesseraROCMDialect>();
  }

  // Target gfx arch — drives the emitted `arch` attribute and the arch-keyed
  // FP8 flavor (FNUZ vs OCP).  Defaults to gfx90a for backward compatibility
  // with existing fixtures.
  Option<std::string> archOpt{
      *this, "arch",
      llvm::cl::desc("target gfx arch (gfx90a/gfx942/gfx950/gfx1200/...)"),
      llvm::cl::init("gfx90a")};

  void runOnOperation() override {
    StringRef arch = archOpt;
    // W1.1 step 0 — legalize the typed fragment form first, by composition.
    // Anything still carrying the legacy bare `!tile.fragment` falls through to
    // the single-shot walk below, unchanged.
    if (failed(convertTypedFragments(getOperation(), arch))) {
      signalPassFailure();
      return;
    }
    SmallVector<func::FuncOp> canonicalAttentionFunctions;
    llvm::SmallPtrSet<Operation *, 4> seenFunctions;
    getOperation().walk([&](scf::ForOp loop) {
      if (!loop->hasAttr("tessera.streaming_attention"))
        return;
      auto function = loop->getParentOfType<func::FuncOp>();
      if (function && seenFunctions.insert(function).second)
        canonicalAttentionFunctions.push_back(function);
    });
    for (func::FuncOp function : canonicalAttentionFunctions) {
      if (failed(materializeCanonicalStreamingAttention(
              getOperation(), function, arch))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<func::FuncOp> canonicalBackwardFunctions;
    llvm::SmallPtrSet<Operation *, 4> seenBackwardFunctions;
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "tessera_attn.backward_dq_block" &&
          name != "tessera_attn.backward_dkdv_block" &&
          name != "tessera_attn.backward_reduce_split")
        return;
      auto function = op->getParentOfType<func::FuncOp>();
      if (function && seenBackwardFunctions.insert(function).second)
        canonicalBackwardFunctions.push_back(function);
    });
    for (func::FuncOp function : canonicalBackwardFunctions) {
      if (failed(materializeCanonicalAttentionBackward(
              getOperation(), function, arch))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<Operation *> worklist;
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.mma" || name == "tile.matmul_kernel" ||
          name == "tile.softmax_kernel" || name == "tile.reduce_kernel" ||
          name == "tile.attention_kernel" ||
          name == "tile.attention_backward_kernel" ||
          name == "tile.paged_kv_read_kernel" ||
          name == "tile.moe_dispatch_kernel" ||
          name == "tile.async_copy" ||
          name == "tile.wait_async" || name == "tile.kv_cache" ||
          name.starts_with("tile.tmem."))
        worklist.push_back(op);
    });

    // FIFO of outstanding async copies (oldest first), keyed by the stamped
    // tile.barrier_id from rocm-wave-lds-pipeline. A wait retires the id it
    // names (or the oldest if idless) — NOT "the last token", so each wait
    // gates the right copy and double-buffering is correct.
    SmallVector<std::pair<std::string, Value>> outstanding;
    for (Operation *op : worklist) {
      OpBuilder builder(op);
      StringRef name = op->getName().getStringRef();

      if (name == "tile.attention_kernel") {
        if (failed(tessera_rocm::materializeROCMDirectAttention(
                cast<tessera::tile::AttentionKernelOp>(op), builder))) {
          signalPassFailure();
          return;
        }
        continue;
      }

      if (name == "tile.attention_backward_kernel") {
        if (failed(tessera_rocm::materializeROCMDirectAttentionBackward(
                cast<tessera::tile::AttentionBackwardKernelOp>(op),
                builder))) {
          signalPassFailure();
          return;
        }
        continue;
      }

      if (name == "tile.softmax_kernel") {
        auto storage = op->getAttrOfType<StringAttr>("storage");
        auto accum = op->getAttrOfType<StringAttr>("accum");
        auto axis = op->getAttrOfType<IntegerAttr>("axis");
        auto expMode = op->getAttrOfType<StringAttr>("exp_mode");
        auto ftz = op->getAttrOfType<BoolAttr>("ftz");
        if (!storage || !accum || !axis || !expMode || !ftz) {
          op->emitError("ROCm softmax lowering requires explicit storage, "
                        "accum, axis, exp_mode, and ftz fields");
          signalPassFailure();
          return;
        }
        if ((storage.getValue() != "f16" && storage.getValue() != "f32") ||
            accum.getValue() != "f32" || axis.getInt() != -1) {
          op->emitError("ROCm softmax pilot requires f16/f32 storage, f32 "
                        "accumulation, and axis=-1");
          signalPassFailure();
          return;
        }
        Operation *symbolOwner = op->getParentOp();
        while (symbolOwner &&
               !symbolOwner->hasAttr(SymbolTable::getSymbolAttrName()))
          symbolOwner = symbolOwner->getParentOp();
        auto symbol = symbolOwner
                          ? symbolOwner->getAttrOfType<StringAttr>(
                                SymbolTable::getSymbolAttrName())
                          : StringAttr();
        if (!symbol) {
          op->emitError("ROCm softmax lowering requires a symbol-owned "
                        "launch envelope");
          signalPassFailure();
          return;
        }

        OperationState state(op->getLoc(), "tessera_rocm.softmax");
        state.addAttribute("name", symbol);
        state.addAttribute("dtype", storage);
        state.addAttribute("accum", accum);
        state.addAttribute("axis", axis);
        state.addAttribute("exp_mode", expMode);
        state.addAttribute("ftz", ftz);
        state.addAttribute("arch", builder.getStringAttr(arch));
        state.addAttribute("source",
                           builder.getStringAttr("tile.softmax_kernel"));
        builder.create(state);
        op->erase();
        continue;
      }

      if (name == "tile.reduce_kernel") {
        auto storage = op->getAttrOfType<StringAttr>("storage");
        auto accum = op->getAttrOfType<StringAttr>("accum");
        auto kind = op->getAttrOfType<StringAttr>("kind");
        auto axis = op->getAttrOfType<IntegerAttr>("axis");
        auto keepdims = op->getAttrOfType<BoolAttr>("keepdims");
        auto schedule = op->getAttrOfType<StringAttr>("schedule");
        auto nanMode = op->getAttrOfType<StringAttr>("nan_mode");
        auto innerIsOne = op->getAttrOfType<BoolAttr>("inner_is_one");
        if (!storage || !accum || !kind || !axis || !keepdims || !schedule ||
            !nanMode) {
          op->emitError("ROCm reduction lowering requires explicit storage, "
                        "accum, kind, axis, keepdims, schedule, and nan_mode");
          signalPassFailure();
          return;
        }
        if ((storage.getValue() != "f16" && storage.getValue() != "bf16" &&
             storage.getValue() != "f32") ||
            accum.getValue() != "f32" ||
            (kind.getValue() != "sum" && kind.getValue() != "mean" &&
             kind.getValue() != "max") ||
            axis.getInt() < 0 || schedule.getValue() != "serial" ||
            nanMode.getValue() != "propagate") {
          op->emitError("ROCM-E2E-2 reduction slice requires f16/bf16/f32 "
                        "storage, f32 accumulation/output, sum|mean|max, a normalized axis, "
                        "the serial semantic schedule, and nan_mode=propagate");
          signalPassFailure();
          return;
        }
        Operation *symbolOwner = op->getParentOp();
        while (symbolOwner &&
               !symbolOwner->hasAttr(SymbolTable::getSymbolAttrName()))
          symbolOwner = symbolOwner->getParentOp();
        auto symbol = symbolOwner
                          ? symbolOwner->getAttrOfType<StringAttr>(
                                SymbolTable::getSymbolAttrName())
                          : StringAttr();
        if (!symbol) {
          op->emitError("ROCm reduction lowering requires a symbol-owned "
                        "launch envelope");
          signalPassFailure();
          return;
        }
        OperationState state(op->getLoc(), "tessera_rocm.reduce");
        state.addAttribute("name", symbol);
        state.addAttribute("dtype", storage);
        state.addAttribute("output_dtype", builder.getStringAttr("f32"));
        state.addAttribute("accum", accum);
        state.addAttribute("kind", kind);
        state.addAttribute("axis", axis);
        state.addAttribute("keepdims", keepdims);
        state.addAttribute("schedule",
                           builder.getStringAttr("workgroup_256"));
        state.addAttribute("nan_mode", nanMode);
        state.addAttribute("layout",
                           builder.getStringAttr("outer_axis_inner"));
        if (innerIsOne && innerIsOne.getValue())
          state.addAttribute("inner_is_one", innerIsOne);
        state.addAttribute("arch", builder.getStringAttr(arch));
        state.addAttribute("source",
                           builder.getStringAttr("tile.reduce_kernel"));
        builder.create(state);
        op->erase();
        continue;
      }

      if (name == "tile.paged_kv_read_kernel") {
        auto storage = op->getAttrOfType<StringAttr>("storage");
        auto tableStorage = op->getAttrOfType<StringAttr>("table_storage");
        auto route = op->getAttrOfType<StringAttr>("route");
        if (op->getNumOperands() != 10 || !storage ||
            storage.getValue() != "f32" || !tableStorage ||
            tableStorage.getValue() != "i32" || !route ||
            route.getValue() != "direct") {
          op->emitError("ROCm paged_kv_read_kernel requires f32 pages, i32 "
                        "table, route=direct, and the canonical ten-operand ABI");
          signalPassFailure();
          return;
        }
        Operation *symbolOwner = op->getParentOp();
        while (symbolOwner &&
               !symbolOwner->hasAttr(SymbolTable::getSymbolAttrName()))
          symbolOwner = symbolOwner->getParentOp();
        auto symbol = symbolOwner
                          ? symbolOwner->getAttrOfType<StringAttr>(
                                SymbolTable::getSymbolAttrName())
                          : StringAttr();
        if (!symbol) {
          op->emitError("ROCm paged-KV lowering requires a symbol-owned launch envelope");
          signalPassFailure();
          return;
        }
        OperationState state(op->getLoc(), "tessera_rocm.paged_kv_read");
        state.addAttribute("name", symbol);
        state.addAttribute("storage", storage);
        state.addAttribute("table_storage", tableStorage);
        state.addAttribute("route", route);
        state.addAttribute("arch", builder.getStringAttr(arch));
        state.addAttribute("source",
                           builder.getStringAttr("tile.paged_kv_read_kernel"));
        builder.create(state);
        op->erase();
        continue;
      }

      if (name == "tile.moe_dispatch_kernel") {
        auto storage = op->getAttrOfType<StringAttr>("storage");
        auto indexStorage = op->getAttrOfType<StringAttr>("index_storage");
        if (op->getNumOperands() != 6 || !storage ||
            storage.getValue() != "f32" || !indexStorage ||
            indexStorage.getValue() != "i32") {
          op->emitError("ROCm moe_dispatch_kernel requires f32 payloads, i32 "
                        "indices, and the canonical six-operand ABI");
          signalPassFailure();
          return;
        }
        Operation *symbolOwner = op->getParentOp();
        while (symbolOwner &&
               !symbolOwner->hasAttr(SymbolTable::getSymbolAttrName()))
          symbolOwner = symbolOwner->getParentOp();
        auto symbol = symbolOwner
                          ? symbolOwner->getAttrOfType<StringAttr>(
                                SymbolTable::getSymbolAttrName())
                          : StringAttr();
        if (!symbol) {
          op->emitError("ROCm MoE dispatch lowering requires a symbol-owned launch envelope");
          signalPassFailure();
          return;
        }
        OperationState state(op->getLoc(), "tessera_rocm.moe_dispatch");
        state.addAttribute("name", symbol);
        state.addAttribute("storage", storage);
        state.addAttribute("index_storage", indexStorage);
        state.addAttribute("route", builder.getStringAttr("direct_gather"));
        state.addAttribute("arch", builder.getStringAttr(arch));
        state.addAttribute("source", builder.getStringAttr("tile.moe_dispatch_kernel"));
        builder.create(state);
        op->erase();
        continue;
      }

      if (name == "tile.matmul_kernel") {
        op->emitError("ROCm tile.matmul_kernel pack/loop/epilogue materializer "
                      "is not implemented for this target");
        signalPassFailure();
        return;
      }

      if (name == "tile.mma") {
        SmallVector<Value> mmaData = dataOperands(op);
        bool typedFragmentForm =
            llvm::any_of(mmaData, [](Value value) {
              return isa<tessera::tile::FragmentType>(value.getType());
            });
        if (typedFragmentForm) {
          if (mmaData.size() != 3 || op->getNumResults() != 1) {
            op->emitError("typed ROCm tile.mma requires exactly A, B, "
                          "accumulator -> one fragment");
            signalPassFailure();
            return;
          }
          tessera::tile::FragmentUnpackOp unpack;
          tessera::tile::StoreOp store;
          bool hasOutputStore = false;
          if (op->getNumResults() == 1 && op->getResult(0).hasOneUse()) {
            unpack = dyn_cast<tessera::tile::FragmentUnpackOp>(
                *op->getResult(0).getUsers().begin());
            if (unpack && unpack.getResult().hasOneUse()) {
              store = dyn_cast<tessera::tile::StoreOp>(
                  *unpack.getResult().getUsers().begin());
              hasOutputStore = static_cast<bool>(store);
            }
          }
          auto aPack = mmaData[0]
                           .getDefiningOp<tessera::tile::FragmentPackOp>();
          auto bPack = mmaData[1]
                           .getDefiningOp<tessera::tile::FragmentPackOp>();
          auto cZero = mmaData[2]
                           .getDefiningOp<tessera::tile::FragmentZeroOp>();
          auto desc =
              op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
          std::optional<tessera_rocm::FragmentLayoutDescriptor> physical =
              tessera_rocm::resolveFragmentLayout(desc, arch);
          if (!physical) {
            if ((arch == "gfx1100" || arch == "gfx1151") && desc &&
                tessera_rocm::isAnyOf(
                    desc.getAType(), {"e4m3", "e5m2", "fp8", "bf8"})) {
              op->emitError("ROCM_TILE_UNSUPPORTED_DTYPE: gfx1151 RDNA 3.5 "
                            "WMMA has no FP8/BF8 matrix form");
              signalPassFailure();
              return;
            }
            op->emitError("ROCM_FRAGMENT_ILLEGAL_ARCH_DESCRIPTOR: no exact ")
                << arch
                << " fragment layout accepts the requested family/dtype/shape; "
                   "RDNA3, RDNA4, gfx125x WMMA-v2, and CDNA MFMA descriptors "
                   "are intentionally non-interchangeable";
            signalPassFailure();
            return;
          }
          if (!physical->materializationReady) {
            op->emitError("ROCM_FRAGMENT_MATERIALIZATION_GATED: ")
                << physical->familyName
                << " recognizes this instruction ABI but its physical "
                   "pack/unpack map is not enabled";
            signalPassFailure();
            return;
          }
          if (!aPack || !bPack || !cZero || !hasOutputStore) {
            op->emitError(
                "typed ROCm lowering requires fragment_pack A/B, "
                "fragment_zero accumulator, fragment_unpack -> tile.store, "
                "and an exact architecture fragment descriptor");
            signalPassFailure();
            return;
          }

          Location loc = op->getLoc();
          Value tx = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
          Value waveSize = arith::ConstantIndexOp::create(
              builder, loc, physical->waveSize);
          Value waveLane = arith::RemUIOp::create(builder, loc, tx, waveSize);
          Value c16 = arith::ConstantIndexOp::create(builder, loc, 16);
          Value lane = arith::RemUIOp::create(builder, loc, waveLane, c16);
          Value laneGroup = arith::DivUIOp::create(builder, loc, waveLane, c16);
          if (physical->inputLaneReplication > 1)
            laneGroup = arith::ConstantIndexOp::create(builder, loc, 0);
          auto packRole = [](Operation *p) -> StringRef {
            auto attr = p->getAttrOfType<StringAttr>("role");
            return attr ? attr.getValue() : StringRef();
          };
          FailureOr<Value> a = materializeFragmentPack(
              aPack, packRole(aPack),
              aPack->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma"),
              builder, lane, laneGroup, *physical);
          FailureOr<Value> b = materializeFragmentPack(
              bPack, packRole(bPack),
              bPack->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma"),
              builder, lane, laneGroup, *physical);
          if (failed(a) || failed(b)) {
            signalPassFailure();
            return;
          }
          bool integer =
              desc.getAType() == "int8" || desc.getAType() == "int4";
          Type accElem = integer ? Type(builder.getIntegerType(32))
                                 : Type(builder.getF32Type());
          auto accTy =
              VectorType::get({physical->accumulatorElementsPerLane}, accElem);
          Value zero = arith::ConstantOp::create(
              builder, loc, accTy, builder.getZeroAttr(accTy));
          OperationState state(
              loc, physical->matrixOp == "wmma" ? "tessera_rocm.wmma"
                                                : "tessera_rocm.mfma");
          state.addOperands({*a, *b, zero});
          state.addTypes({accTy});
          state.addAttribute("arch", builder.getStringAttr(arch));
          state.addAttribute(
              "shape",
              builder.getStringAttr(("m16n16k" + Twine(desc.getK())).str()));
          state.addAttribute("accum",
                             builder.getStringAttr(integer ? "i32" : "f32"));
          state.addAttribute("input_dtype",
                             builder.getStringAttr(desc.getAType()));
          state.addAttribute("source",
                             builder.getStringAttr("tile.fragment_pack"));
          state.addAttribute("fragment_family",
                             builder.getStringAttr(physical->familyName));
          state.addAttribute(
              "fragment_input_format",
              builder.getStringAttr(
                  tessera_rocm::registerFormatName(physical->inputFormat)));
          state.addAttribute("fragment_wave_size",
                             builder.getI64IntegerAttr(physical->waveSize));
          state.addAttribute(
              "fragment_intrinsic_abi",
              builder.getStringAttr(physical->intrinsicABI));
          state.addAttribute("ordinal", builder.getI64IntegerAttr(0));
          Operation *target = builder.create(state);
          Value storeGroup = arith::DivUIOp::create(builder, loc, waveLane, c16);
          if (failed(materializeFragmentStore(
                  target->getResult(0),
                  unpack->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma"),
                  unpack, store, builder, lane, storeGroup, *physical))) {
            signalPassFailure();
            return;
          }

          Operation *aView = aPack.getInputs().front().getDefiningOp();
          Operation *bView = bPack.getInputs().front().getDefiningOp();
          store->erase();
          unpack->erase();
          op->erase();
          if (aPack.getResult().use_empty())
            aPack->erase();
          if (bPack.getResult().use_empty())
            bPack->erase();
          if (cZero.getResult().use_empty())
            cZero->erase();
          if (aView && aView->use_empty())
            aView->erase();
          if (bView && bView != aView && bView->use_empty())
            bView->erase();
          continue;
        }
        if (failed(rejectUnconsumedStoragePack(op))) {
          signalPassFailure();
          return;
        }
        if (mmaData.size() < 2 || op->getNumResults() != 1) {
          op->emitError("ROCm lowering requires tile.mma(lhs, rhs) -> result");
          signalPassFailure();
          return;
        }

        // Derive the arch-keyed FP8 flavor when the operands are FP8.  An FP8
        // matmul on an arch with no FP8 matrix path is a hard, named error
        // (Decision #21) — never a silent flavor guess.
        std::string fp8Flavor;
        if (isFP8Element(mmaData[0].getType())) {
          llvm::StringRef sem = fp8SemanticsForArch(arch);
          if (sem == "none") {
            op->emitError("ROCm lowering: FP8 matmul requested on arch '")
                << arch
                << "' which has no FP8 matrix path (see "
                   "tessera.compiler.rocm_target._FP8_SEMANTICS)";
            signalPassFailure();
            return;
          }
          std::string base = fp8Base(mmaData[0].getType());
          fp8Flavor = (sem == "fnuz") ? base + "fnuz" : base;
        }

        // RDNA -> WMMA, CDNA -> MFMA. Same 16x16x16 v1 artifact contract; only
        // the target matrix op (and its eventual ROCDL marker) differs.
        StringRef matrixOp =
            isWmmaArch(arch) ? "tessera_rocm.wmma" : "tessera_rocm.mfma";
        OperationState state(op->getLoc(), matrixOp);
        // Accumulator: a 3-operand tile.mma(lhs, rhs, acc) carries the real
        // accumulator SSA value (the executable Fork-A form the GEMM generator
        // emits in --via-tile mode) — thread it straight through so the lowered
        // tessera_rocm.wmma is bit-identical to the direct generator's. The
        // legacy 2-operand artifact form has no accumulator SSA yet, so it falls
        // back to lhs as a typed placeholder (IR-contract lowering, not run).
        Value acc = mmaData.size() >= 3 ? mmaData[2] : mmaData[0];
        state.addOperands({mmaData[0], mmaData[1], acc});
        state.addTypes(op->getResultTypes());
        state.addAttribute("arch", builder.getStringAttr(arch));
        state.addAttribute("shape", builder.getStringAttr("m16n16k16"));
        state.addAttribute("accum", builder.getStringAttr("f32"));
        state.addAttribute("source", builder.getStringAttr("tessera.matmul"));
        state.addAttribute("ordinal", builder.getI64IntegerAttr(0));
        if (!fp8Flavor.empty())
          state.addAttribute("fp8_flavor", builder.getStringAttr(fp8Flavor));
        copyAttrIfPresent(state, op, "numeric_policy");
        copyAttrIfPresent(state, op, "tessera.storage_pack");
        copyAttrIfPresent(state, op, "tile.pipeline_depths");
        copyAttrIfPresent(state, op, "tile.rocm_matrix_path");
        Operation *rocmOp = builder.create(state);
        op->replaceAllUsesWith(rocmOp->getResults());
        op->erase();
        continue;
      }

      if (name == "tile.async_copy") {
        if (failed(rejectUnconsumedStoragePack(op))) {
          signalPassFailure();
          return;
        }
        // The planner may append a !tile.async_token result (the SSA completion
        // edge). It is the trailing result; the leading result is the staged
        // tile the rocm op produces. Require exactly one data result.
        unsigned numResults = op->getNumResults();
        bool hasTileToken =
            numResults >= 1 &&
            isa<tessera::tile::AsyncTokenType>(
                op->getResult(numResults - 1).getType());
        unsigned numData = hasTileToken ? numResults - 1 : numResults;
        if (op->getNumOperands() < 3 || numData != 1) {
          op->emitError("ROCm lowering requires tile.async_copy(dst, src, bytes) -> token");
          signalPassFailure();
          return;
        }

        Value byteCount = op->getOperand(2);
        if (byteCount.getType().isIndex())
          byteCount = arith::IndexCastOp::create(
              builder, op->getLoc(), builder.getI64Type(), byteCount);
        OperationState state(op->getLoc(), "tessera_rocm.async_copy");
        state.addOperands({op->getOperand(0), op->getOperand(1),
                           byteCount});
        state.addTypes(
            mlir::tessera_rocm::TokenType::get(builder.getContext()));
        state.addAttribute("src_space", builder.getStringAttr("global"));
        state.addAttribute("dst_space", builder.getStringAttr("lds"));
        state.addAttribute("arch", builder.getStringAttr(arch));
        Value buffer = tileBufferRoot(op);
        auto layout =
            op->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
        bool hasPipelineState = llvm::any_of(
            op->getOperands(), [](Value value) {
              return isa<tessera::tile::PipelineStateType>(value.getType());
            });
        if (layout && (!buffer || !hasTileToken || !hasPipelineState)) {
          op->emitOpError(
              "structured ROCm LDS copies require !tile.buffer allocation "
              "identity, a !tile.async_token result, and a "
              "!tile.pipeline_state operand; run rocm-wave-lds-pipeline");
          signalPassFailure();
          return;
        }
        if (buffer) {
          auto alloc = buffer.getDefiningOp<tessera::tile::AllocOp>();
          if (!alloc || alloc.getSpace() != "smem") {
            op->emitOpError(
                "ROCm async copy requires its !tile.buffer root to be a "
                "tile.alloc in shared-memory space");
            signalPassFailure();
            return;
          }
          state.addAttribute("buffer_identity",
                             builder.getStringAttr("ssa_allocation"));
          state.addAttribute(
              "buffer_bytes",
              builder.getI64IntegerAttr(static_cast<int64_t>(alloc.getBytes())));
          state.addAttribute("buffer_space", builder.getStringAttr("lds"));
          state.addAttribute("buffer_layout", alloc.getLayout());
        } else {
          // Unshaped pointer copies carry a dynamic byte count and externally
          // owned LDS destination. Keep that distinct from compiler-owned
          // tile.alloc identity instead of fabricating a static allocation.
          state.addAttribute("buffer_identity",
                             builder.getStringAttr("external_pointer"));
        }
        if (layout) {
          if (!layoutHasLdsAxis(layout)) {
            op->emitOpError(
                "ROCM_LOWERING_LAYOUT_NOT_LDS: ROCm async copy can only "
                "consume #tile.layout placements that include the lds axis.");
            signalPassFailure();
            return;
          }
          state.addAttribute("uses_tile_layout", builder.getBoolAttr(true));
          state.addAttribute("layout_storage", builder.getStringAttr("lds"));
          copyAttrIfPresent(state, op, "tile.layout");
        }
        copyAttrIfPresent(state, op, "numeric_policy");
        copyAttrIfPresent(state, op, "tessera.storage_pack");
        copyAttrIfPresent(state, op, "tile.pipeline_depths");
        Operation *rocmOp = builder.create(state);
        // Record this copy in the FIFO keyed by its stamped barrier id and its
        // SSA token (the rocm result). A wait retires it by SSA value when its
        // operand names the token, else by id/order.
        std::string id;
        if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id"))
          id = a.getValue().str();
        outstanding.push_back({id, rocmOp->getResult(0)});
        // Redirect the data result and, if present, the planner's
        // !tile.async_token result to the rocm token — so a consuming wait/mma's
        // token operand resolves to this copy's SSA value after lowering.
        op->getResult(0).replaceAllUsesWith(rocmOp->getResult(0));
        if (hasTileToken)
          op->getResult(numResults - 1).replaceAllUsesWith(rocmOp->getResult(0));
        op->erase();
        continue;
      }

      if (name == "tile.wait_async") {
        if (outstanding.empty()) {
          op->emitError("ROCm lowering requires tile.wait_async after tile.async_copy");
          signalPassFailure();
          return;
        }

        // Retire the copy this wait consumes. Prefer the SSA token operand (the
        // planner threaded it, post-lowering it resolves to the copy's rocm
        // token) — a precise def-use retirement. Else fall back to the stamped
        // tile.barrier_id, else the oldest outstanding — never "the last token".
        Value token;
        for (Value operand : op->getOperands()) {
          auto it = llvm::find_if(outstanding, [&](const auto &e) {
            return e.second == operand;
          });
          if (it != outstanding.end()) {
            token = it->second;
            outstanding.erase(it);
            break;
          }
        }
        if (!token) {
          if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id")) {
            StringRef want = a.getValue();
            auto it = llvm::find_if(outstanding, [&](const auto &e) {
              return e.first == want;
            });
            if (it != outstanding.end()) {
              token = it->second;
              outstanding.erase(it);
            }
          }
        }
        if (!token) {
          token = outstanding.front().second; // oldest
          outstanding.erase(outstanding.begin());
        }

        OperationState state(op->getLoc(), "tessera_rocm.wait");
        state.addOperands(token);
        // The async copy is global→LDS, which retires on the vector-memory
        // counter — gate on vmcnt rather than draining the wavefront with a full
        // barrier (the decoupled-wait lever: the matrix core keeps issuing while
        // the copy is still in flight).
        StringRef counter = "vmcnt";
        if (auto counterAttr =
                op->getAttrOfType<StringAttr>("tile.wait_counter"))
          counter = counterAttr.getValue();
        state.addAttribute("counter", builder.getStringAttr(counter));
        // Preserve the barrier-id + waitcnt threshold for ROCDL contract
        // lowering (vmcnt(threshold) metadata), so targeted waits stay targeted.
        if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id"))
          state.addAttribute("barrier_id", a);
        if (auto a = op->getAttrOfType<IntegerAttr>("tile.waitcnt_threshold"))
          state.addAttribute("threshold", a);
        builder.create(state);
        op->erase();
        continue;
      }

      if (name == "tile.kv_cache") {
        op->emitError("ROCm lowering does not implement KV-cache artifacts in this phase");
        signalPassFailure();
        return;
      }

      if (name.starts_with("tile.tmem.")) {
        op->emitError("ROCm lowering does not support TMEM operations");
        signalPassFailure();
        return;
      }
    }

    // The portable SSA state/allocation ops are compile-time ownership proof.
    // Once every ROCm target op has consumed their token, buffer, and phase
    // dependencies, remove dead proof chains at this target boundary. Keep any
    // still-live chain intact so a later pass cannot silently lose ownership.
    SmallVector<Operation *> advances;
    SmallVector<Operation *> inits;
    SmallVector<tessera::tile::AllocOp> allocs;
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.pipeline_advance")
        advances.push_back(op);
      else if (name == "tile.pipeline_init")
        inits.push_back(op);
      else if (auto alloc = dyn_cast<tessera::tile::AllocOp>(op))
        allocs.push_back(alloc);
    });
    for (Operation *advance : llvm::reverse(advances))
      if (llvm::all_of(advance->getResults(),
                       [](Value result) { return result.use_empty(); }))
        advance->erase();
    for (Operation *init : llvm::reverse(inits))
      if (llvm::all_of(init->getResults(),
                       [](Value result) { return result.use_empty(); }))
        init->erase();
    for (tessera::tile::AllocOp alloc : allocs) {
      SmallVector<Operation *> deallocs;
      bool onlyDeallocUsers = true;
      for (Operation *user : alloc.getBuffer().getUsers()) {
        if (isa<tessera::tile::DeallocOp>(user))
          deallocs.push_back(user);
        else
          onlyDeallocUsers = false;
      }
      if (!onlyDeallocUsers)
        continue;
      for (Operation *dealloc : deallocs)
        dealloc->erase();
      if (alloc.getBuffer().use_empty())
        alloc->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::tessera_rocm::createLowerTileToROCMImpl() {
  return std::make_unique<LowerTileToROCMPass>();
}
