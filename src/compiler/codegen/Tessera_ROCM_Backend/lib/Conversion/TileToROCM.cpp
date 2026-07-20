#include "TesseraROCM/Passes.h"
#include "ROCMFragmentLayout.h"

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "TesseraROCMDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "TesseraROCMTypes.h.inc"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Twine.h"

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

static FailureOr<Value> materializeFragmentPack(
    tessera::tile::FragmentPackOp pack, OpBuilder &builder, Value lane,
    Value laneGroup,
    const tessera_rocm::FragmentLayoutDescriptor &physical) {
  Operation *op = pack.getOperation();
  auto role = op->getAttrOfType<StringAttr>("role");
  auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
  auto view = pack.getInputs().front().getDefiningOp<tessera::tile::ViewOp>();
  if (!role || !view || !desc) {
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
      role.getValue() == "a"
          ? std::array<int64_t, 2>{desc.getM(), desc.getK()}
          : std::array<int64_t, 2>{desc.getK(), desc.getN()};
  StringRef expectedOrder = role.getValue() == "a" ? "row_major" : "col_major";
  if ((role.getValue() != "a" && role.getValue() != "b") || !memory ||
      !layout || view.getInputs().size() != 3 ||
      memory.getSpace() != "gmem" || memory.getOrder() != expectedOrder ||
      layout.getShardExtents() != ArrayRef<int64_t>(expectedShape) ||
      layout.getSwizzle()) {
    op->emitError("ROCM_FRAGMENT_UNSUPPORTED_SOURCE_LAYOUT: unsupported ")
        << physical.familyName << " fragment source layout for role "
        << (role ? role.getValue() : StringRef("<missing>"));
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
  if (role.getValue() == "a") {
    Value row = arith::AddIOp::create(builder, loc, rowOrigin, lane);
    Value col = arith::AddIOp::create(builder, loc, colOrigin, kBase);
    Value linear = arith::AddIOp::create(
        builder, loc,
        arith::MulIOp::create(builder, loc, row, leadingDim), col);
    fragment = vector::LoadOp::create(builder, loc, vectorTy, base,
                                      ValueRange{linear});
  } else {
    Value row = arith::AddIOp::create(builder, loc, rowOrigin, kBase);
    Value col = arith::AddIOp::create(builder, loc, colOrigin, lane);
    Value linear = arith::AddIOp::create(
        builder, loc,
        arith::MulIOp::create(builder, loc, col, leadingDim), row);
    fragment = vector::LoadOp::create(builder, loc, vectorTy, base,
                                      ValueRange{linear});
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

static LogicalResult materializeFragmentStore(
    Operation *target, tessera::tile::FragmentUnpackOp unpack,
    tessera::tile::StoreOp store, OpBuilder &builder, Value lane,
    Value laneGroup,
    const tessera_rocm::FragmentLayoutDescriptor &physical) {
  Operation *op = store.getOperation();
  auto desc = unpack->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
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
    Value scalar = vector::ExtractOp::create(builder, loc, target->getResult(0),
                                             ArrayRef<int64_t>{i});
    memref::StoreOp::create(builder, loc, scalar, base, ValueRange{linear});
  }
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
    SmallVector<Operation *> worklist;
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.mma" || name == "tile.matmul_kernel" ||
          name == "tile.softmax_kernel" || name == "tile.reduce_kernel" ||
          name == "tile.paged_kv_read_kernel" ||
          name == "tile.moe_dispatch_kernel" ||
          name == "tile.async_copy" ||
          name == "tile.wait_async" || name == "tile.kv_cache" ||
          name.starts_with("tile.tmem."))
        worklist.push_back(op);
    });

    StringRef arch = archOpt;
    // FIFO of outstanding async copies (oldest first), keyed by the stamped
    // tile.barrier_id from rocm-wave-lds-pipeline. A wait retires the id it
    // names (or the oldest if idless) — NOT "the last token", so each wait
    // gates the right copy and double-buffering is correct.
    SmallVector<std::pair<std::string, Value>> outstanding;
    for (Operation *op : worklist) {
      OpBuilder builder(op);
      StringRef name = op->getName().getStringRef();

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
        bool typedFragmentForm =
            llvm::any_of(op->getOperandTypes(), [](Type type) {
              return isa<tessera::tile::FragmentType>(type);
            });
        if (typedFragmentForm) {
          if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
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
          auto aPack = op->getOperand(0)
                           .getDefiningOp<tessera::tile::FragmentPackOp>();
          auto bPack = op->getOperand(1)
                           .getDefiningOp<tessera::tile::FragmentPackOp>();
          auto cZero = op->getOperand(2)
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
          FailureOr<Value> a =
              materializeFragmentPack(aPack, builder, lane, laneGroup, *physical);
          FailureOr<Value> b =
              materializeFragmentPack(bPack, builder, lane, laneGroup, *physical);
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
          if (failed(materializeFragmentStore(target, unpack, store, builder,
                                              lane, storeGroup, *physical))) {
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
        if (op->getNumOperands() < 2 || op->getNumResults() != 1) {
          op->emitError("ROCm lowering requires tile.mma(lhs, rhs) -> result");
          signalPassFailure();
          return;
        }

        // Derive the arch-keyed FP8 flavor when the operands are FP8.  An FP8
        // matmul on an arch with no FP8 matrix path is a hard, named error
        // (Decision #21) — never a silent flavor guess.
        std::string fp8Flavor;
        if (isFP8Element(op->getOperand(0).getType())) {
          llvm::StringRef sem = fp8SemanticsForArch(arch);
          if (sem == "none") {
            op->emitError("ROCm lowering: FP8 matmul requested on arch '")
                << arch
                << "' which has no FP8 matrix path (see "
                   "tessera.compiler.rocm_target._FP8_SEMANTICS)";
            signalPassFailure();
            return;
          }
          std::string base = fp8Base(op->getOperand(0).getType());
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
        Value acc = op->getNumOperands() >= 3 ? op->getOperand(2)
                                              : op->getOperand(0);
        state.addOperands({op->getOperand(0), op->getOperand(1), acc});
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
        if (auto buf = op->getAttrOfType<tessera::tile::TileBufferRefAttr>(
                "tile.buf")) {
          if (buf.getSpace() != "lds") {
            op->emitOpError(
                "ROCM_LOWERING_NON_LDS_BUFFER: tile.async_copy expected "
                "#tile.buffer_ref<space = \"lds\"> for ROCm global-to-LDS "
                "movement.");
            signalPassFailure();
            return;
          }
          if (buf.getAccess() != "write") {
            op->emitOpError(
                "ROCM_LOWERING_NON_WRITE_BUFFER: tile.async_copy expected "
                "#tile.buffer_ref access = \"write\" for the LDS destination.");
            signalPassFailure();
            return;
          }
          state.addAttribute("buffer", builder.getStringAttr(buf.getName()));
        }
        if (auto layout =
                op->getAttrOfType<tessera::tile::TileLayoutAttr>(
                    "tile.layout")) {
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
        if (auto a = op->getAttrOfType<tessera::tile::TileBufferRefAttr>(
                "tile.buf"))
          id = a.getName().str();
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
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::tessera_rocm::createLowerTileToROCMImpl() {
  return std::make_unique<LowerTileToROCMPass>();
}
