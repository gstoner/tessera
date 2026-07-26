//===- ROCMDirectAttention.cpp - dynamic Tile attention on ROCm -----------===//
//
// Correctness-first consumers for the canonical launch-level attention
// carriers. Unlike the optimized tessera_rocm.flash_attn directives, these
// materializers consume runtime B/H/S/D operands directly and therefore do not
// invent a compile-time head dimension. One workitem owns one output (forward)
// or gradient (backward) element. The implementation is deliberately scalar
// but preserves the complete portable recurrence and provides the executable
// baseline against which an LDS/WMMA schedule can be selected.

#include "ROCMDirectAttention.h"

#include "Tessera/Dialect/Tile/TileEpilogue.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include <optional>

using namespace mlir;

namespace mlir::tessera_rocm {
namespace {

static Value i64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(builder, loc, value, 64);
}

static Value addI64(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  return arith::AddIOp::create(builder, loc, lhs, rhs);
}

static Value mulI64(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  return arith::MulIOp::create(builder, loc, lhs, rhs);
}

static Value lessI64(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult, lhs,
                              rhs);
}

static Value workitemLinear(OpBuilder &builder, Location loc) {
  Value block =
      gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value thread =
      gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Type i64 = builder.getI64Type();
  block = arith::IndexCastUIOp::create(builder, loc, i64, block);
  thread = arith::IndexCastUIOp::create(builder, loc, i64, thread);
  return addI64(builder, loc,
                mulI64(builder, loc, block,
                       i64Constant(builder, loc, 128)),
                thread);
}

// The generic scf.if builder overload intentionally permits an empty region.
// These carrier materializers build guarded bodies incrementally, so create a
// structurally complete single-block region up front and keep its yield as the
// insertion anchor.
static scf::IfOp createGuardedRegion(OpBuilder &builder, Location loc,
                                     Value condition) {
  auto guard =
      scf::IfOp::create(builder, loc, TypeRange{}, condition,
                        /*addThenBlock=*/false, /*addElseBlock=*/false);
  Block *thenBlock = new Block();
  guard.getThenRegion().push_back(thenBlock);
  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToEnd(thenBlock);
  scf::YieldOp::create(builder, loc);
  return guard;
}

struct AttentionContract {
  Operation *op;
  OpBuilder &builder;
  Location loc;
  ValueRange inputs;
  FloatType f32;
  Type storageType;
  unsigned storageAlignment;
  bool f16Storage;
  bool bf16Storage;
  bool hasBias;
  FloatAttr scaleAttr;
  BoolAttr causalAttr;
  IntegerAttr windowLeftAttr;
  IntegerAttr windowRightAttr;
  FloatAttr softcapAttr;
  FloatAttr dropoutAttr;
  IntegerAttr dropoutSeedAttr;
  Value B;
  Value Hq;
  Value Hkv;
  Value Sq;
  Value Sk;
  Value D;
  Value Dv;

  Value zeroI64() { return i64Constant(builder, loc, 0); }
  Value oneI64() { return i64Constant(builder, loc, 1); }
  Value zeroF32() {
    return arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
  }
  Value oneF32() {
    return arith::ConstantFloatOp::create(builder, loc, f32, APFloat(1.0f));
  }

  Value load(unsigned pointer, Value index, bool f32Pointer = false) {
    Type type = f32Pointer ? f32 : storageType;
    unsigned alignment = f32Pointer ? 4 : storageAlignment;
    Value ptr = LLVM::GEPOp::create(builder, loc, inputs[pointer].getType(),
                                    type, inputs[pointer], ValueRange{index});
    Value value = LLVM::LoadOp::create(builder, loc, type, ptr, alignment);
    if (!f32Pointer && (f16Storage || bf16Storage))
      return LLVM::FPExtOp::create(builder, loc, f32, value);
    return value;
  }

  void store(unsigned pointer, Value index, Value value,
             bool f32Pointer = false) {
    Type type = f32Pointer ? f32 : storageType;
    unsigned alignment = f32Pointer ? 4 : storageAlignment;
    if (!f32Pointer && (f16Storage || bf16Storage))
      value = LLVM::FPTruncOp::create(builder, loc, storageType, value);
    Value ptr = LLVM::GEPOp::create(builder, loc, inputs[pointer].getType(),
                                    type, inputs[pointer], ValueRange{index});
    LLVM::StoreOp::create(builder, loc, value, ptr, alignment);
  }

  Value qIndex(Value b, Value h, Value q, Value d) {
    return addI64(
        builder, loc,
        mulI64(
            builder, loc,
            addI64(builder, loc,
                   mulI64(builder, loc,
                          addI64(builder, loc,
                                 mulI64(builder, loc, b, Hq), h),
                          Sq),
                   q),
            D),
        d);
  }

  Value kvIndex(Value b, Value h, Value key, Value d, Value width) {
    return addI64(
        builder, loc,
        mulI64(
            builder, loc,
            addI64(builder, loc,
                   mulI64(builder, loc,
                          addI64(builder, loc,
                                 mulI64(builder, loc, b, Hkv), h),
                          Sk),
                   key),
            width),
        d);
  }

  Value outputIndex(Value b, Value h, Value q, Value d) {
    return addI64(
        builder, loc,
        mulI64(
            builder, loc,
            addI64(builder, loc,
                   mulI64(builder, loc,
                          addI64(builder, loc,
                                 mulI64(builder, loc, b, Hq), h),
                          Sq),
                   q),
            Dv),
        d);
  }

  Value biasIndex(Value b, Value h, Value q, Value key) {
    return addI64(
        builder, loc,
        mulI64(
            builder, loc,
            addI64(builder, loc,
                   mulI64(builder, loc,
                          addI64(builder, loc,
                                 mulI64(builder, loc, b, Hq), h),
                          Sq),
                   q),
            Sk),
        key);
  }

  Value legalKey(Value q, Value key) {
    Value legal = arith::ConstantIntOp::create(builder, loc, 1, 1);
    if (causalAttr.getValue())
      legal = arith::AndIOp::create(
          builder, loc, legal,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ule, key,
                                q));
    if (windowLeftAttr.getInt() >= 0) {
      Value lower = arith::SubIOp::create(
          builder, loc, q,
          i64Constant(builder, loc, windowLeftAttr.getInt()));
      legal = arith::AndIOp::create(
          builder, loc, legal,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge, key,
                                lower));
    }
    if (windowRightAttr.getInt() >= 0) {
      Value upper =
          addI64(builder, loc, q,
                 i64Constant(builder, loc, windowRightAttr.getInt()));
      legal = arith::AndIOp::create(
          builder, loc, legal,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle, key,
                                upper));
    }
    return legal;
  }

  Value rawScore(Value b, Value hq, Value hkv, Value q, Value key,
                 unsigned qPointer, unsigned kPointer,
                 std::optional<unsigned> biasPointer = std::nullopt) {
    auto dot = scf::ForOp::create(builder, loc, zeroI64(), D, oneI64(),
                                  ValueRange{zeroF32()});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(dot.getBody());
      Value d = dot.getInductionVar();
      Value product = arith::MulFOp::create(
          builder, loc, load(qPointer, qIndex(b, hq, q, d)),
          load(kPointer, kvIndex(b, hkv, key, d, D)));
      Value next = arith::AddFOp::create(
          builder, loc, dot.getRegionIterArgs()[0], product);
      scf::YieldOp::create(builder, loc, ValueRange{next});
    }
    builder.setInsertionPointAfter(dot);
    Value scale =
        arith::ConstantFloatOp::create(builder, loc, f32, scaleAttr.getValue());
    Value value =
        arith::MulFOp::create(builder, loc, dot.getResult(0), scale);
    if (biasPointer)
      value = arith::AddFOp::create(
          builder, loc, value,
          load(*biasPointer, biasIndex(b, hq, q, key),
               /*f32Pointer=*/true));
    return value;
  }

  Value cappedScore(Value raw) {
    if (softcapAttr.getValueAsDouble() <= 0.0)
      return raw;
    Value cap = arith::ConstantFloatOp::create(builder, loc, f32,
                                               softcapAttr.getValue());
    return arith::MulFOp::create(
        builder, loc, cap,
        tessera::tile::emitBoundedTanhApprox(
            builder, loc,
            arith::DivFOp::create(builder, loc, raw, cap)));
  }

  Value softcapDerivative(Value raw) {
    if (softcapAttr.getValueAsDouble() <= 0.0)
      return oneF32();
    Value cap = arith::ConstantFloatOp::create(builder, loc, f32,
                                               softcapAttr.getValue());
    Value t = tessera::tile::emitBoundedTanhApprox(
        builder, loc, arith::DivFOp::create(builder, loc, raw, cap));
    return arith::SubFOp::create(
        builder, loc, oneF32(),
        arith::MulFOp::create(builder, loc, t, t));
  }

  Value maskedScore(Value b, Value hq, Value hkv, Value q, Value key,
                    unsigned qPointer, unsigned kPointer,
                    std::optional<unsigned> biasPointer = std::nullopt) {
    Value value =
        cappedScore(rawScore(b, hq, hkv, q, key, qPointer, kPointer,
                             biasPointer));
    Value negInf = arith::ConstantFloatOp::create(
        builder, loc, f32,
        APFloat::getInf(APFloat::IEEEsingle(), /*negative=*/true));
    return arith::SelectOp::create(builder, loc, legalKey(q, key), value,
                                   negInf);
  }

  Value expShifted(Value value, Value maxValue) {
    return math::ExpOp::create(
        builder, loc,
        arith::SubFOp::create(builder, loc, value, maxValue));
  }

  Value dropoutScale(Value b, Value hq, Value q, Value key) {
    if (dropoutAttr.getValueAsDouble() <= 0.0)
      return oneF32();
    Value counter = addI64(
        builder, loc,
        mulI64(
            builder, loc,
            addI64(builder, loc,
                   mulI64(builder, loc,
                          addI64(builder, loc,
                                 mulI64(builder, loc, b, Hq), hq),
                          Sq),
                   q),
            Sk),
        key);
    Value hash = addI64(
        builder, loc,
        mulI64(builder, loc, counter,
               i64Constant(builder, loc, 1664525)),
        i64Constant(builder, loc,
                    dropoutSeedAttr.getInt() + 1013904223));
    hash = arith::AndIOp::create(
        builder, loc, hash,
        i64Constant(builder, loc, 0xffffffffULL));
    uint64_t threshold = static_cast<uint64_t>(
        dropoutAttr.getValueAsDouble() * 4294967296.0);
    Value keep = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::uge, hash,
        i64Constant(builder, loc, threshold));
    Value invKeep = arith::ConstantFloatOp::create(
        builder, loc, f32,
        APFloat(static_cast<float>(
            1.0 / (1.0 - dropoutAttr.getValueAsDouble()))));
    return arith::SelectOp::create(builder, loc, keep, invKeep, zeroF32());
  }
};

static FailureOr<AttentionContract>
parseContract(Operation *op, OpBuilder &builder, unsigned dimStart,
              bool hasBias) {
  auto storage = op->getAttrOfType<StringAttr>("storage");
  auto accum = op->getAttrOfType<StringAttr>("accum");
  auto scale = op->getAttrOfType<FloatAttr>("scale");
  auto causal = op->getAttrOfType<BoolAttr>("causal");
  auto windowLeft = op->getAttrOfType<IntegerAttr>("window_left");
  auto windowRight = op->getAttrOfType<IntegerAttr>("window_right");
  auto softcap = op->getAttrOfType<FloatAttr>("softcap");
  auto dropout = op->getAttrOfType<FloatAttr>("dropout_p");
  auto dropoutSeed = op->getAttrOfType<IntegerAttr>("dropout_seed");
  bool f16 = storage && storage.getValue() == "f16";
  bool bf16 = storage && storage.getValue() == "bf16";
  if (!storage || (!f16 && !bf16 && storage.getValue() != "f32") ||
      !accum || accum.getValue() != "f32" || !scale || !causal ||
      !windowLeft || !windowRight || !softcap || !dropout || !dropoutSeed) {
    op->emitError(
        "ROCm attention carrier requires f16/bf16/f32 storage, f32 "
        "accumulation, and complete scale/mask/dropout attributes");
    return failure();
  }
  FloatType f32 = builder.getF32Type();
  Type storageType =
      f16 ? Type(builder.getF16Type())
          : bf16 ? Type(builder.getBF16Type()) : f32;
  ValueRange inputs = op->getOperands();
  return AttentionContract{
      op,
      builder,
      op->getLoc(),
      inputs,
      f32,
      storageType,
      f16 || bf16 ? 2u : 4u,
      f16,
      bf16,
      hasBias,
      scale,
      causal,
      windowLeft,
      windowRight,
      softcap,
      dropout,
      dropoutSeed,
      inputs[dimStart],
      inputs[dimStart + 1],
      inputs[dimStart + 2],
      inputs[dimStart + 3],
      inputs[dimStart + 4],
      inputs[dimStart + 5],
      inputs[dimStart + 6]};
}

} // namespace

LogicalResult
materializeROCMDirectAttention(tessera::tile::AttentionKernelOp kernel,
                               OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  auto biasAttr = op->getAttrOfType<BoolAttr>("bias");
  bool hasBias = biasAttr && biasAttr.getValue();
  unsigned outputPointer = 3 + unsigned(hasBias);
  unsigned dimStart = outputPointer + 1;
  if (!biasAttr || kernel.getInputs().size() != 11 + unsigned(hasBias)) {
    op->emitError(
        "ROCm attention_kernel requires the canonical launch-level ABI");
    return failure();
  }

  // A statically bucketed head/value dimension can use the existing physical
  // gfx1151 WMMA schedule. Keep the launch dimensions themselves dynamic: the
  // generated kernel accepts Sq/Sk (and GQA metadata) at launch time. The
  // scalar direct recurrence below remains the total fallback for carriers
  // whose semantics the optimized schedule does not yet implement.
  auto headDim = op->getAttrOfType<IntegerAttr>("head_dim");
  auto valueDim = op->getAttrOfType<IntegerAttr>("value_dim");
  auto storage = op->getAttrOfType<StringAttr>("storage");
  auto dropout = op->getAttrOfType<FloatAttr>("dropout_p");
  auto causal = op->getAttrOfType<BoolAttr>("causal");
  auto windowLeft = op->getAttrOfType<IntegerAttr>("window_left");
  auto windowRight = op->getAttrOfType<IntegerAttr>("window_right");
  auto softcap = op->getAttrOfType<FloatAttr>("softcap");
  bool windowCompatible =
      windowLeft && windowRight &&
      ((windowLeft.getInt() == -1 && windowRight.getInt() == -1) ||
       (causal && causal.getValue() && windowLeft.getInt() >= 0 &&
        windowRight.getInt() == 0));
  bool optimized =
      headDim && valueDim && storage && dropout && causal && softcap &&
      headDim.getInt() == valueDim.getInt() && headDim.getInt() > 0 &&
      headDim.getInt() % 16 == 0 &&
      (storage.getValue() == "f16" || storage.getValue() == "bf16") &&
      dropout.getValueAsDouble() >= 0.0 &&
      dropout.getValueAsDouble() < 1.0 && windowCompatible;
  if (optimized) {
    Operation *symbolOwner = op->getParentOp();
    while (symbolOwner &&
           !symbolOwner->hasAttr(SymbolTable::getSymbolAttrName()))
      symbolOwner = symbolOwner->getParentOp();
    auto symbol =
        symbolOwner
            ? symbolOwner->getAttrOfType<StringAttr>(
                  SymbolTable::getSymbolAttrName())
            : StringAttr();
    if (!symbol) {
      op->emitError("ROCm optimized attention requires a symbol-owned launch "
                    "envelope");
      return failure();
    }
    OperationState state(op->getLoc(), "tessera_rocm.flash_attn");
    state.addAttribute("name", symbol);
    state.addAttribute("head_dim", headDim);
    state.addAttribute("dtype", storage);
    state.addAttribute("gqa", builder.getBoolAttr(
                                  op->getAttrOfType<BoolAttr>("gqa")
                                      ? op->getAttrOfType<BoolAttr>("gqa")
                                            .getValue()
                                      : false));
    state.addAttribute(
        "sliding_window",
        builder.getBoolAttr(windowLeft.getInt() >= 0));
    state.addAttribute("logit_softcap",
                       builder.getBoolAttr(
                           softcap.getValueAsDouble() > 0.0));
    state.addAttribute("attn_bias", builder.getBoolAttr(hasBias));
    state.addAttribute(
        "dropout",
        builder.getBoolAttr(dropout.getValueAsDouble() > 0.0));
    state.addAttribute("source",
                       builder.getStringAttr("tile.attention_kernel"));
    state.addAttribute("schedule",
                       builder.getStringAttr("gfx1151_wmma_streaming"));
    builder.create(state);
    op->erase();
    return success();
  }

  FailureOr<AttentionContract> parsed =
      parseContract(op, builder, dimStart, hasBias);
  if (failed(parsed))
    return failure();
  AttentionContract &c = *parsed;
  std::optional<unsigned> biasPointer =
      hasBias ? std::optional<unsigned>(3) : std::nullopt;

  Value linear = workitemLinear(builder, c.loc);
  Value total = mulI64(
      builder, c.loc,
      mulI64(builder, c.loc,
             mulI64(builder, c.loc, c.B, c.Hq), c.Sq),
      c.Dv);
  auto active = createGuardedRegion(
      builder, c.loc, lessI64(builder, c.loc, linear, total));
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&active.getThenRegion().front());
    Value dv = arith::RemUIOp::create(builder, c.loc, linear, c.Dv);
    Value t0 = arith::DivUIOp::create(builder, c.loc, linear, c.Dv);
    Value q = arith::RemUIOp::create(builder, c.loc, t0, c.Sq);
    Value t1 = arith::DivUIOp::create(builder, c.loc, t0, c.Sq);
    Value hq = arith::RemUIOp::create(builder, c.loc, t1, c.Hq);
    Value b = arith::DivUIOp::create(builder, c.loc, t1, c.Hq);
    Value ratio = arith::DivUIOp::create(builder, c.loc, c.Hq, c.Hkv);
    Value hkv = arith::DivUIOp::create(builder, c.loc, hq, ratio);

    Value negInf = arith::ConstantFloatOp::create(
        builder, c.loc, c.f32,
        APFloat::getInf(APFloat::IEEEsingle(), /*negative=*/true));
    auto maxLoop = scf::ForOp::create(builder, c.loc, c.zeroI64(), c.Sk,
                                      c.oneI64(), ValueRange{negInf});
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(maxLoop.getBody());
      Value score = c.maskedScore(b, hq, hkv, q,
                                  maxLoop.getInductionVar(), 0, 1,
                                  biasPointer);
      Value next = arith::MaximumFOp::create(
          builder, c.loc, maxLoop.getRegionIterArgs()[0], score);
      scf::YieldOp::create(builder, c.loc, ValueRange{next});
    }
    builder.setInsertionPointAfter(maxLoop);

    auto accumLoop =
        scf::ForOp::create(builder, c.loc, c.zeroI64(), c.Sk,
                           c.oneI64(),
                           ValueRange{c.zeroF32(), c.zeroF32()});
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(accumLoop.getBody());
      Value key = accumLoop.getInductionVar();
      Value weight = c.expShifted(
          c.maskedScore(b, hq, hkv, q, key, 0, 1, biasPointer),
          maxLoop.getResult(0));
      Value v = c.load(2, c.kvIndex(b, hkv, key, dv, c.Dv));
      Value outputWeight = arith::MulFOp::create(
          builder, c.loc, weight, c.dropoutScale(b, hq, q, key));
      Value denom = arith::AddFOp::create(
          builder, c.loc, accumLoop.getRegionIterArgs()[0], weight);
      Value numer = arith::AddFOp::create(
          builder, c.loc, accumLoop.getRegionIterArgs()[1],
          arith::MulFOp::create(builder, c.loc, outputWeight, v));
      scf::YieldOp::create(builder, c.loc, ValueRange{denom, numer});
    }
    builder.setInsertionPointAfter(accumLoop);
    Value result = arith::DivFOp::create(
        builder, c.loc, accumLoop.getResult(1), accumLoop.getResult(0));
    c.store(outputPointer, linear, result, /*f32Pointer=*/true);
  }
  builder.setInsertionPointAfter(active);
  op->erase();
  return success();
}

LogicalResult materializeROCMDirectAttentionBackward(
    tessera::tile::AttentionBackwardKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  auto biasAttr = op->getAttrOfType<BoolAttr>("bias");
  bool hasBias = biasAttr && biasAttr.getValue();
  unsigned biasPointer = 4;
  unsigned dqPointer = 4 + unsigned(hasBias);
  unsigned dkPointer = dqPointer + 1;
  unsigned dvPointer = dqPointer + 2;
  unsigned dimStart = dqPointer + 3;
  auto route = op->getAttrOfType<StringAttr>("route");
  auto deterministic = op->getAttrOfType<BoolAttr>("deterministic");
  auto workspace = op->getAttrOfType<IntegerAttr>("workspace_bytes");
  auto workspaceOwner =
      op->getAttrOfType<StringAttr>("workspace_owner");
  if (!biasAttr ||
      kernel.getInputs().size() != 14 + unsigned(hasBias) || !route ||
      route.getValue() != "deterministic_direct" || !deterministic ||
      !deterministic.getValue() || !workspace || workspace.getInt() != 0 ||
      !workspaceOwner ||
      workspaceOwner.getValue() != "output_element") {
    op->emitError(
        "ROCm attention_backward_kernel requires the canonical deterministic "
        "direct ABI with output-element workspace ownership");
    return failure();
  }

  // Keep the canonical carrier's deterministic-direct route as the semantic
  // reference, but select the compiler-generated split/reduced physical
  // schedule when the static bucket is representable by gfx1151 WMMA.  The
  // native package owns the launch workspace; no workspace pointer is added to
  // the portable carrier ABI.
  auto headDim = op->getAttrOfType<IntegerAttr>("head_dim");
  auto valueDim = op->getAttrOfType<IntegerAttr>("value_dim");
  auto storage = op->getAttrOfType<StringAttr>("storage");
  auto dropout = op->getAttrOfType<FloatAttr>("dropout_p");
  auto causal = op->getAttrOfType<BoolAttr>("causal");
  auto windowLeft = op->getAttrOfType<IntegerAttr>("window_left");
  auto windowRight = op->getAttrOfType<IntegerAttr>("window_right");
  auto softcap = op->getAttrOfType<FloatAttr>("softcap");
  bool windowCompatible =
      windowLeft && windowRight &&
      ((windowLeft.getInt() == -1 && windowRight.getInt() == -1) ||
       (causal && causal.getValue() && windowLeft.getInt() >= 0 &&
        windowRight.getInt() == 0));
  bool optimized =
      headDim && valueDim && storage && dropout && causal && softcap &&
      headDim.getInt() == valueDim.getInt() && headDim.getInt() > 0 &&
      headDim.getInt() % 16 == 0 &&
      (storage.getValue() == "f16" || storage.getValue() == "bf16") &&
      dropout.getValueAsDouble() == 0.0 && windowCompatible;
  if (optimized) {
    Operation *symbolOwner = op->getParentOp();
    while (symbolOwner &&
           !symbolOwner->hasAttr(SymbolTable::getSymbolAttrName()))
      symbolOwner = symbolOwner->getParentOp();
    auto symbol =
        symbolOwner
            ? symbolOwner->getAttrOfType<StringAttr>(
                  SymbolTable::getSymbolAttrName())
            : StringAttr();
    if (!symbol) {
      op->emitError("ROCm optimized attention backward requires a "
                    "symbol-owned launch envelope");
      return failure();
    }
    OperationState state(op->getLoc(), "tessera_rocm.flash_attn_bwd");
    state.addAttribute("name", symbol);
    state.addAttribute("head_dim", headDim);
    state.addAttribute("dtype", storage);
    state.addAttribute(
        "gqa", builder.getBoolAttr(
                   op->getAttrOfType<BoolAttr>("gqa")
                       ? op->getAttrOfType<BoolAttr>("gqa").getValue()
                       : false));
    state.addAttribute("sliding_window",
                       builder.getBoolAttr(windowLeft.getInt() >= 0));
    state.addAttribute(
        "logit_softcap",
        builder.getBoolAttr(softcap.getValueAsDouble() > 0.0));
    state.addAttribute("attn_bias", builder.getBoolAttr(hasBias));
    state.addAttribute("split_reduced", builder.getBoolAttr(true));
    state.addAttribute(
        "source", builder.getStringAttr("tile.attention_backward_kernel"));
    state.addAttribute(
        "schedule",
        builder.getStringAttr("gfx1151_wmma_backward_split_reduced"));
    builder.create(state);
    op->erase();
    return success();
  }

  FailureOr<AttentionContract> parsed =
      parseContract(op, builder, dimStart, hasBias);
  if (failed(parsed))
    return failure();
  AttentionContract &c = *parsed;
  std::optional<unsigned> bias =
      hasBias ? std::optional<unsigned>(biasPointer) : std::nullopt;
  Value ratio = arith::DivUIOp::create(builder, c.loc, c.Hq, c.Hkv);
  Value scale = arith::ConstantFloatOp::create(
      builder, c.loc, c.f32, c.scaleAttr.getValue());

  auto doDotV = [&](Value b, Value hq, Value hkv, Value q,
                    Value key) -> Value {
    auto dot = scf::ForOp::create(builder, c.loc, c.zeroI64(), c.Dv,
                                  c.oneI64(), ValueRange{c.zeroF32()});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(dot.getBody());
      Value d = dot.getInductionVar();
      Value product = arith::MulFOp::create(
          builder, c.loc,
          c.load(0, c.outputIndex(b, hq, q, d)),
          c.load(3, c.kvIndex(b, hkv, key, d, c.Dv)));
      Value next = arith::AddFOp::create(
          builder, c.loc, dot.getRegionIterArgs()[0], product);
      scf::YieldOp::create(builder, c.loc, ValueRange{next});
    }
    builder.setInsertionPointAfter(dot);
    return dot.getResult(0);
  };

  auto rowStats = [&](Value b, Value hq, Value hkv,
                      Value q) -> SmallVector<Value, 3> {
    Value negInf = arith::ConstantFloatOp::create(
        builder, c.loc, c.f32,
        APFloat::getInf(APFloat::IEEEsingle(), /*negative=*/true));
    auto maxLoop = scf::ForOp::create(builder, c.loc, c.zeroI64(), c.Sk,
                                      c.oneI64(), ValueRange{negInf});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(maxLoop.getBody());
      Value score = c.maskedScore(b, hq, hkv, q,
                                  maxLoop.getInductionVar(), 1, 2, bias);
      Value next = arith::MaximumFOp::create(
          builder, c.loc, maxLoop.getRegionIterArgs()[0], score);
      scf::YieldOp::create(builder, c.loc, ValueRange{next});
    }
    builder.setInsertionPointAfter(maxLoop);
    auto sums = scf::ForOp::create(
        builder, c.loc, c.zeroI64(), c.Sk, c.oneI64(),
        ValueRange{c.zeroF32(), c.zeroF32()});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(sums.getBody());
      Value key = sums.getInductionVar();
      Value weight = c.expShifted(
          c.maskedScore(b, hq, hkv, q, key, 1, 2, bias),
          maxLoop.getResult(0));
      Value denom = arith::AddFOp::create(
          builder, c.loc, sums.getRegionIterArgs()[0], weight);
      Value dP = doDotV(b, hq, hkv, q, key);
      Value numer = arith::AddFOp::create(
          builder, c.loc, sums.getRegionIterArgs()[1],
          arith::MulFOp::create(
              builder, c.loc, weight,
              arith::MulFOp::create(
                  builder, c.loc, c.dropoutScale(b, hq, q, key), dP)));
      scf::YieldOp::create(builder, c.loc, ValueRange{denom, numer});
    }
    builder.setInsertionPointAfter(sums);
    Value delta = arith::DivFOp::create(
        builder, c.loc, sums.getResult(1), sums.getResult(0));
    return {maxLoop.getResult(0), sums.getResult(0), delta};
  };

  auto probability = [&](Value b, Value hq, Value hkv, Value q, Value key,
                         Value maxValue, Value denom) -> Value {
    Value weight = c.expShifted(
        c.maskedScore(b, hq, hkv, q, key, 1, 2, bias), maxValue);
    return arith::DivFOp::create(builder, c.loc, weight, denom);
  };

  auto dScore = [&](Value b, Value hq, Value hkv, Value q, Value key,
                    ArrayRef<Value> stats) -> Value {
    Value raw = c.rawScore(b, hq, hkv, q, key, 1, 2, bias);
    Value p =
        probability(b, hq, hkv, q, key, stats[0], stats[1]);
    Value dP = arith::MulFOp::create(
        builder, c.loc, c.dropoutScale(b, hq, q, key),
        doDotV(b, hq, hkv, q, key));
    Value ds = arith::MulFOp::create(
        builder, c.loc, p,
        arith::SubFOp::create(builder, c.loc, dP, stats[2]));
    return arith::MulFOp::create(builder, c.loc, ds,
                                 c.softcapDerivative(raw));
  };

  Value linear = workitemLinear(builder, c.loc);
  Value dqCount = mulI64(
      builder, c.loc,
      mulI64(builder, c.loc,
             mulI64(builder, c.loc, c.B, c.Hq), c.Sq),
      c.D);
  Value dkCount = mulI64(
      builder, c.loc,
      mulI64(builder, c.loc,
             mulI64(builder, c.loc, c.B, c.Hkv), c.Sk),
      c.D);
  Value dvCount = mulI64(
      builder, c.loc,
      mulI64(builder, c.loc,
             mulI64(builder, c.loc, c.B, c.Hkv), c.Sk),
      c.Dv);

  auto dqGuard = createGuardedRegion(
      builder, c.loc, lessI64(builder, c.loc, linear, dqCount));
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&dqGuard.getThenRegion().front());
    Value d = arith::RemUIOp::create(builder, c.loc, linear, c.D);
    Value t0 = arith::DivUIOp::create(builder, c.loc, linear, c.D);
    Value q = arith::RemUIOp::create(builder, c.loc, t0, c.Sq);
    Value t1 = arith::DivUIOp::create(builder, c.loc, t0, c.Sq);
    Value hq = arith::RemUIOp::create(builder, c.loc, t1, c.Hq);
    Value b = arith::DivUIOp::create(builder, c.loc, t1, c.Hq);
    Value hkv = arith::DivUIOp::create(builder, c.loc, hq, ratio);
    SmallVector<Value, 3> stats = rowStats(b, hq, hkv, q);
    auto sum = scf::ForOp::create(builder, c.loc, c.zeroI64(), c.Sk,
                                  c.oneI64(), ValueRange{c.zeroF32()});
    {
      OpBuilder::InsertionGuard sumGuard(builder);
      builder.setInsertionPointToStart(sum.getBody());
      Value key = sum.getInductionVar();
      Value term = arith::MulFOp::create(
          builder, c.loc,
          arith::MulFOp::create(
              builder, c.loc, dScore(b, hq, hkv, q, key, stats), scale),
          c.load(2, c.kvIndex(b, hkv, key, d, c.D)));
      Value next = arith::AddFOp::create(
          builder, c.loc, sum.getRegionIterArgs()[0], term);
      scf::YieldOp::create(builder, c.loc, ValueRange{next});
    }
    builder.setInsertionPointAfter(sum);
    c.store(dqPointer, linear, sum.getResult(0));
  }
  builder.setInsertionPointAfter(dqGuard);

  Value dkLinear =
      arith::SubIOp::create(builder, c.loc, linear, dqCount);
  auto dkGuard = createGuardedRegion(
      builder, c.loc,
      arith::AndIOp::create(
          builder, c.loc,
          arith::CmpIOp::create(builder, c.loc,
                                arith::CmpIPredicate::uge, linear, dqCount),
          lessI64(builder, c.loc, dkLinear, dkCount)));
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&dkGuard.getThenRegion().front());
    Value d = arith::RemUIOp::create(builder, c.loc, dkLinear, c.D);
    Value t0 = arith::DivUIOp::create(builder, c.loc, dkLinear, c.D);
    Value key = arith::RemUIOp::create(builder, c.loc, t0, c.Sk);
    Value t1 = arith::DivUIOp::create(builder, c.loc, t0, c.Sk);
    Value hkv = arith::RemUIOp::create(builder, c.loc, t1, c.Hkv);
    Value b = arith::DivUIOp::create(builder, c.loc, t1, c.Hkv);
    Value hqBegin = mulI64(builder, c.loc, hkv, ratio);
    Value hqEnd = addI64(builder, c.loc, hqBegin, ratio);
    auto headLoop = scf::ForOp::create(
        builder, c.loc, hqBegin, hqEnd, c.oneI64(),
        ValueRange{c.zeroF32()});
    {
      OpBuilder::InsertionGuard headGuard(builder);
      builder.setInsertionPointToStart(headLoop.getBody());
      Value hq = headLoop.getInductionVar();
      auto qLoop = scf::ForOp::create(
          builder, c.loc, c.zeroI64(), c.Sq, c.oneI64(),
          ValueRange{headLoop.getRegionIterArgs()[0]});
      {
        OpBuilder::InsertionGuard qGuard(builder);
        builder.setInsertionPointToStart(qLoop.getBody());
        Value q = qLoop.getInductionVar();
        SmallVector<Value, 3> stats = rowStats(b, hq, hkv, q);
        Value term = arith::MulFOp::create(
            builder, c.loc,
            arith::MulFOp::create(
                builder, c.loc,
                dScore(b, hq, hkv, q, key, stats), scale),
            c.load(1, c.qIndex(b, hq, q, d)));
        Value next = arith::AddFOp::create(
            builder, c.loc, qLoop.getRegionIterArgs()[0], term);
        scf::YieldOp::create(builder, c.loc, ValueRange{next});
      }
      builder.setInsertionPointAfter(qLoop);
      scf::YieldOp::create(builder, c.loc,
                           ValueRange{qLoop.getResult(0)});
    }
    builder.setInsertionPointAfter(headLoop);
    c.store(dkPointer, dkLinear, headLoop.getResult(0));
  }
  builder.setInsertionPointAfter(dkGuard);

  Value dvLinear =
      arith::SubIOp::create(builder, c.loc, dkLinear, dkCount);
  Value dqDkCount = addI64(builder, c.loc, dqCount, dkCount);
  auto dvGuard = createGuardedRegion(
      builder, c.loc,
      arith::AndIOp::create(
          builder, c.loc,
          arith::CmpIOp::create(builder, c.loc,
                                arith::CmpIPredicate::uge, linear,
                                dqDkCount),
          lessI64(builder, c.loc, dvLinear, dvCount)));
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&dvGuard.getThenRegion().front());
    Value d = arith::RemUIOp::create(builder, c.loc, dvLinear, c.Dv);
    Value t0 = arith::DivUIOp::create(builder, c.loc, dvLinear, c.Dv);
    Value key = arith::RemUIOp::create(builder, c.loc, t0, c.Sk);
    Value t1 = arith::DivUIOp::create(builder, c.loc, t0, c.Sk);
    Value hkv = arith::RemUIOp::create(builder, c.loc, t1, c.Hkv);
    Value b = arith::DivUIOp::create(builder, c.loc, t1, c.Hkv);
    Value hqBegin = mulI64(builder, c.loc, hkv, ratio);
    Value hqEnd = addI64(builder, c.loc, hqBegin, ratio);
    auto headLoop = scf::ForOp::create(
        builder, c.loc, hqBegin, hqEnd, c.oneI64(),
        ValueRange{c.zeroF32()});
    {
      OpBuilder::InsertionGuard headGuard(builder);
      builder.setInsertionPointToStart(headLoop.getBody());
      Value hq = headLoop.getInductionVar();
      auto qLoop = scf::ForOp::create(
          builder, c.loc, c.zeroI64(), c.Sq, c.oneI64(),
          ValueRange{headLoop.getRegionIterArgs()[0]});
      {
        OpBuilder::InsertionGuard qGuard(builder);
        builder.setInsertionPointToStart(qLoop.getBody());
        Value q = qLoop.getInductionVar();
        SmallVector<Value, 3> stats = rowStats(b, hq, hkv, q);
        Value p =
            probability(b, hq, hkv, q, key, stats[0], stats[1]);
        Value term = arith::MulFOp::create(
            builder, c.loc,
            arith::MulFOp::create(
                builder, c.loc, p,
                c.dropoutScale(b, hq, q, key)),
            c.load(0, c.outputIndex(b, hq, q, d)));
        Value next = arith::AddFOp::create(
            builder, c.loc, qLoop.getRegionIterArgs()[0], term);
        scf::YieldOp::create(builder, c.loc, ValueRange{next});
      }
      builder.setInsertionPointAfter(qLoop);
      scf::YieldOp::create(builder, c.loc,
                           ValueRange{qLoop.getResult(0)});
    }
    builder.setInsertionPointAfter(headLoop);
    c.store(dvPointer, dvLinear, headLoop.getResult(0));
  }
  builder.setInsertionPointAfter(dvGuard);
  op->erase();
  return success();
}

} // namespace mlir::tessera_rocm
