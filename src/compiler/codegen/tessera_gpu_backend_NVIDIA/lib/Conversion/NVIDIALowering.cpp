#include "tessera/gpu/BackendRegistration.h"
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Dialect/Tile/TileEpilogue.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <optional>

#include "TesseraNVIDIADialect.h.inc"

using namespace mlir;

namespace tessera {
namespace {

static constexpr int kHopperSM = 90;
static constexpr int kBlackwellSM = 100;          // datacenter Blackwell (sm_100a)
static constexpr int kConsumerBlackwellSM = 120;  // consumer Blackwell (sm_120)

static StringRef archStringForSM(int sm) {
  if (sm >= 120)
    return "sm_120";
  if (sm >= 100)
    return "sm_100a";
  if (sm >= 90)
    return "sm_90a";
  if (sm >= 80)
    return "sm_80";
  return "sm_unknown";
}

static bool isTileOp(Operation *op, StringRef suffix) {
  return op->getName().getStringRef() == suffix;
}

static Operation *createContractOp(OpBuilder &builder, Location loc,
                                   StringRef name, ValueRange operands,
                                   TypeRange results,
                                   ArrayRef<NamedAttribute> attrs = {}) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttributes(attrs);
  return builder.create(state);
}

enum class Sm120InputPacking {
  PairF16,
  ScalarF32,
  PackedX4I8,
  PackedX8E2M1,
};

struct Sm120FragmentDescriptor {
  StringRef dtype;
  int64_t k;
  StringRef accumulator;
  Sm120InputPacking packing;
  unsigned aRegistersPerLane;
  unsigned bRegistersPerLane;
  unsigned accumulatorRegistersPerLane;
  StringRef instructionFamily;
};

static std::optional<Sm120FragmentDescriptor>
selectSm120Fragment(tessera::tile::TileMmaDescAttr desc) {
  if (!desc || (desc.getFamily() != "auto" && desc.getFamily() != "mma_sync" &&
                desc.getFamily() != "mma_sync_block_scale") ||
      desc.getM() != 16 || desc.getN() != 8 ||
      desc.getAType() != desc.getBType() ||
      desc.getALayout() != "row_major" || desc.getBLayout() != "col_major" ||
      desc.getKBlocks() != 1)
    return std::nullopt;
  StringRef dtype = desc.getAType();
  if ((dtype == "nvfp4" || dtype == "fp4_e2m1") && desc.getK() == 64 &&
      desc.getAccType() == "f32")
    return Sm120FragmentDescriptor{
        dtype, 64, "f32", Sm120InputPacking::PackedX8E2M1, 4, 2, 4,
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
        ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3"};
  if (dtype == "f16" && desc.getK() == 16 && desc.getAccType() == "f16")
    return Sm120FragmentDescriptor{
        dtype, 16, "f16", Sm120InputPacking::PairF16, 4, 2, 2,
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"};
  if ((dtype == "f16" || dtype == "bf16") && desc.getK() == 16 &&
      desc.getAccType() == "f32")
    return Sm120FragmentDescriptor{
        dtype, 16, "f32", Sm120InputPacking::PairF16, 4, 2, 4,
        dtype == "f16"
            ? "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            : "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"};
  if (dtype == "tf32" && desc.getK() == 8)
    return desc.getAccType() == "f32"
        ? std::optional<Sm120FragmentDescriptor>(Sm120FragmentDescriptor{
              dtype, 8, "f32", Sm120InputPacking::ScalarF32, 4, 2, 4,
              "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"})
        : std::nullopt;
  if ((dtype == "e4m3" || dtype == "e5m2") && desc.getK() == 32)
    return desc.getAccType() == "f32"
        ? std::optional<Sm120FragmentDescriptor>(Sm120FragmentDescriptor{
              dtype, 32, "f32", Sm120InputPacking::PackedX4I8, 4, 2, 4,
              dtype == "e4m3"
                  ? "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
                  : "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"})
        : std::nullopt;
  if ((dtype == "s8" || dtype == "int8") && desc.getK() == 32)
    return (desc.getAccType() == "s32" || desc.getAccType() == "int32")
        ? std::optional<Sm120FragmentDescriptor>(Sm120FragmentDescriptor{
              dtype, 32, "s32", Sm120InputPacking::PackedX4I8, 4, 2, 4,
              "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"})
        : std::nullopt;
  return std::nullopt;
}

static bool isCanonicalSm120Mma16(tessera::tile::TileMmaDescAttr desc) {
  return selectSm120Fragment(desc).has_value();
}

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
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                               lhs, rhs);
}

static Value mask2(OpBuilder &builder, Location loc, Value first, Value second) {
  auto maskTy = VectorType::get({2}, builder.getI1Type());
  Value mask = arith::ConstantOp::create(builder, loc, maskTy,
                                         builder.getZeroAttr(maskTy));
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 64);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 64);
  mask = LLVM::InsertElementOp::create(builder, loc, maskTy, mask, first, zero);
  return LLVM::InsertElementOp::create(builder, loc, maskTy, mask, second, one);
}

static Value maskedLoadPair(OpBuilder &builder, Location loc, Value base,
                            Type elementType, Value linear, Value valid0,
                            Value valid1, unsigned alignment) {
  auto pairTy = VectorType::get({2}, elementType);
  Value ptr = LLVM::GEPOp::create(builder, loc, base.getType(), elementType,
                                  base, ValueRange{linear});
  Value passthru = arith::ConstantOp::create(builder, loc, pairTy,
                                              builder.getZeroAttr(pairTy));
  return LLVM::MaskedLoadOp::create(builder, loc, pairTy, ptr,
                                    mask2(builder, loc, valid0, valid1),
                                    passthru, alignment, false);
}

static Value pairFromScalars(OpBuilder &builder, Location loc, Type elementType,
                             Value first, Value second) {
  auto pairTy = VectorType::get({2}, elementType);
  Value pair = arith::ConstantOp::create(builder, loc, pairTy,
                                         builder.getZeroAttr(pairTy));
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 64);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 64);
  pair = LLVM::InsertElementOp::create(builder, loc, pairTy, pair, first, zero);
  return LLVM::InsertElementOp::create(builder, loc, pairTy, pair, second, one);
}

static Value maskedLoadScalar(OpBuilder &builder, Location loc, Value base,
                              Type elementType, Value linear, Value valid,
                              unsigned alignment) {
  auto vectorTy = VectorType::get({1}, elementType);
  auto maskTy = VectorType::get({1}, builder.getI1Type());
  Value ptr = LLVM::GEPOp::create(builder, loc, base.getType(), elementType,
                                  base, ValueRange{linear});
  Value passthru = arith::ConstantOp::create(builder, loc, vectorTy,
                                              builder.getZeroAttr(vectorTy));
  Value mask = arith::ConstantOp::create(builder, loc, maskTy,
                                          builder.getZeroAttr(maskTy));
  Value zero = i64Constant(builder, loc, 0);
  mask = LLVM::InsertElementOp::create(builder, loc, maskTy, mask, valid, zero);
  Value loaded = LLVM::MaskedLoadOp::create(builder, loc, vectorTy, ptr, mask,
                                             passthru, alignment, false);
  return LLVM::ExtractElementOp::create(builder, loc, elementType, loaded, zero);
}

static FailureOr<Value> sm120SharedBuffer(Operation *anchor, OpBuilder &builder,
                                          Type elementType, StringRef dtype) {
  ModuleOp module = anchor->getParentOfType<ModuleOp>();
  if (!module)
    return failure();
  std::string symbol = ("__tessera_sm120_ab_stage_" + dtype).str();
  auto global = module.lookupSymbol<LLVM::GlobalOp>(symbol);
  // Four warps own a 32x32 CTA macro-tile.  One K panel stages A[32,16]
  // followed by B[32,16] in its column-major physical representation.
  auto arrayTy = LLVM::LLVMArrayType::get(elementType, 1024);
  if (!global) {
    OpBuilder globalBuilder(module.getBodyRegion());
    globalBuilder.setInsertionPointToStart(module.getBody());
    global = LLVM::GlobalOp::create(
        globalBuilder, anchor->getLoc(), arrayTy, /*isConstant=*/false,
        LLVM::Linkage::Internal, symbol, Attribute(), /*alignment=*/16,
        /*addrSpace=*/3, /*dsoLocal=*/false, /*threadLocal=*/false,
        SymbolRefAttr(), ArrayRef<NamedAttribute>{}, ArrayRef<Attribute>{});
  }
  return Value(LLVM::AddressOfOp::create(builder, anchor->getLoc(), global));
}

static LogicalResult materializeSm120MatmulKernel(
    tessera::tile::MatmulKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
  auto epilogue = op->getAttrOfType<tessera::tile::TileEpilogueAttr>("epilogue");
  if (!isCanonicalSm120Mma16(desc) || desc.getAccType() != "f32" || !epilogue) {
    op->emitError("sm_120 matmul_kernel requires m16n8k16 f16 or bf16 inputs "
                  "with f32 accumulation");
    return failure();
  }
  ValueRange inputs = kernel.getInputs();
  bool hasBias = epilogue.getBias();
  Value aBase = inputs[0], bBase = inputs[1];
  Value biasBase = hasBias ? inputs[2] : Value();
  unsigned dIndex = hasBias ? 3 : 2;
  Value dBase = inputs[dIndex];
  Value m = inputs[dIndex + 1], n = inputs[dIndex + 2], k = inputs[dIndex + 3];
  Location loc = op->getLoc();
  Type inputType = desc.getAType() == "bf16"
      ? Type(builder.getBF16Type()) : Type(builder.getF16Type());
  Type f16 = builder.getF16Type();
  Type f32 = builder.getF32Type();
  int64_t warps = 1;
  if (auto attr = op->getAttrOfType<IntegerAttr>("warps"))
    warps = attr.getInt();
  bool sharedStaging = false;
  if (auto attr = op->getAttrOfType<StringAttr>("staging"))
    sharedStaging = attr.getValue() == "shared";
  FailureOr<Value> sharedBase = failure();
  if (sharedStaging) {
    sharedBase = sm120SharedBuffer(op, builder, inputType, desc.getAType());
    if (failed(sharedBase)) {
      op->emitError("failed to materialize sm_120 shared staging buffer");
      return failure();
    }
  }

  Value blockX32 = NVVM::BlockIdXOp::create(builder, loc, builder.getI32Type());
  Value blockY32 = NVVM::BlockIdYOp::create(builder, loc, builder.getI32Type());
  Value blockX = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), blockX32);
  Value blockY = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), blockY32);
  Value tid = NVVM::ThreadIdXOp::create(builder, loc, builder.getI32Type());
  Value warp32 = arith::ShRUIOp::create(
      builder, loc, tid, arith::ConstantIntOp::create(builder, loc, 5, 32));
  Value warpRow32 = warps == 4
      ? Value(arith::ShRUIOp::create(
            builder, loc, warp32,
            arith::ConstantIntOp::create(builder, loc, 1, 32)))
      : Value(arith::ConstantIntOp::create(builder, loc, 0, 32));
  Value warpCol32 = warps == 4
      ? Value(arith::AndIOp::create(
            builder, loc, warp32,
            arith::ConstantIntOp::create(builder, loc, 1, 32)))
      : Value(arith::ConstantIntOp::create(builder, loc, 0, 32));
  Value warpRow = arith::ExtUIOp::create(
      builder, loc, builder.getI64Type(), warpRow32);
  Value warpCol = arith::ExtUIOp::create(
      builder, loc, builder.getI64Type(), warpCol32);
  Value macroRowOrigin = mulI64(
      builder, loc, blockY,
      i64Constant(builder, loc, warps == 4 ? 32 : 16));
  Value macroColOrigin = mulI64(
      builder, loc, blockX,
      i64Constant(builder, loc, warps == 4 ? 32 : 8));
  Value rowOrigin = addI64(
      builder, loc, macroRowOrigin,
      mulI64(builder, loc, warpRow, i64Constant(builder, loc, 16)));
  Value colOrigin = addI64(
      builder, loc, macroColOrigin,
      mulI64(builder, loc, warpCol,
             i64Constant(builder, loc, warps == 4 ? 16 : 8)));

  Value lane = arith::AndIOp::create(
      builder, loc, tid, arith::ConstantIntOp::create(builder, loc, 31, 32));
  Value gid32 = arith::ShRUIOp::create(
      builder, loc, lane, arith::ConstantIntOp::create(builder, loc, 2, 32));
  Value tig32 = arith::AndIOp::create(
      builder, loc, lane, arith::ConstantIntOp::create(builder, loc, 3, 32));
  Value twoTig32 = arith::MulIOp::create(
      builder, loc, tig32, arith::ConstantIntOp::create(builder, loc, 2, 32));
  Value gid = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), gid32);
  Value twoTig = arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                        twoTig32);

  Value zeroI64 = i64Constant(builder, loc, 0);
  Value sixteen = i64Constant(builder, loc, 16);
  Value zeroF32 = arith::ConstantFloatOp::create(
      builder, loc, builder.getF32Type(), APFloat(0.0f));
  // A shared-staged warp computes two adjacent m16n8 fragments so each pair of
  // CTA barriers is amortized over eight MMA instructions rather than four.
  SmallVector<Value> init(sharedStaging ? 8 : 4, zeroF32);
  auto loop = scf::ForOp::create(builder, loc, zeroI64, k, sixteen, init);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value kOrigin = loop.getInductionVar();
    Value eight = i64Constant(builder, loc, 8);
    Value one = i64Constant(builder, loc, 1);
    Value row0 = addI64(builder, loc, rowOrigin, gid);
    Value row1 = addI64(builder, loc, row0, eight);
    Value k0 = addI64(builder, loc, kOrigin, twoTig);
    Value k1 = addI64(builder, loc, k0, eight);

    if (sharedStaging) {
      Value tid64 = arith::ExtUIOp::create(
          builder, loc, builder.getI64Type(), tid);
      auto stageLoop = scf::ForOp::create(
          builder, loc, tid64, i64Constant(builder, loc, 1024),
          i64Constant(builder, loc, warps * 32));
      {
        OpBuilder::InsertionGuard stageGuard(builder);
        builder.setInsertionPointToStart(stageLoop.getBody());
        Value index = stageLoop.getInductionVar();
        Value isA = lessI64(builder, loc, index,
                            i64Constant(builder, loc, 512));
        auto branch = scf::IfOp::create(builder, loc, isA,
                                        /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard branchGuard(builder);
          {
            builder.setInsertionPointToStart(branch.thenBlock());
            Value localRow = arith::DivUIOp::create(
                builder, loc, index, sixteen);
            Value localK = arith::RemUIOp::create(
                builder, loc, index, sixteen);
            Value globalRow = addI64(builder, loc, macroRowOrigin, localRow);
            Value globalK = addI64(builder, loc, kOrigin, localK);
            Value valid = arith::AndIOp::create(
                builder, loc, lessI64(builder, loc, globalRow, m),
                lessI64(builder, loc, globalK, k));
            Value linear = addI64(
                builder, loc, mulI64(builder, loc, globalRow, k), globalK);
            Value scalar = maskedLoadScalar(
                builder, loc, aBase, inputType, linear, valid, 2);
            Value ptr = LLVM::GEPOp::create(
                builder, loc, (*sharedBase).getType(), inputType, *sharedBase,
                ValueRange{index});
            LLVM::StoreOp::create(builder, loc, scalar, ptr, /*alignment=*/2);
          }

          {
            builder.setInsertionPointToStart(branch.elseBlock());
            Value bIndex = arith::SubIOp::create(
                builder, loc, index, i64Constant(builder, loc, 512));
            Value localCol = arith::DivUIOp::create(
                builder, loc, bIndex, sixteen);
            Value localK = arith::RemUIOp::create(
                builder, loc, bIndex, sixteen);
            Value globalCol = addI64(builder, loc, macroColOrigin, localCol);
            Value globalK = addI64(builder, loc, kOrigin, localK);
            Value valid = arith::AndIOp::create(
                builder, loc, lessI64(builder, loc, globalCol, n),
                lessI64(builder, loc, globalK, k));
            Value linear = addI64(
                builder, loc, mulI64(builder, loc, globalCol, k), globalK);
            Value scalar = maskedLoadScalar(
                builder, loc, bBase, inputType, linear, valid, 2);
            Value ptr = LLVM::GEPOp::create(
                builder, loc, (*sharedBase).getType(), inputType, *sharedBase,
                ValueRange{index});
            LLVM::StoreOp::create(builder, loc, scalar, ptr, /*alignment=*/2);
          }
        }
      }
      builder.setInsertionPointAfter(stageLoop);
      NVVM::BarrierOp::create(builder, loc);
    }

    auto loadA = [&](Value row, Value col) {
      if (sharedStaging) {
        Value localRow = arith::SubIOp::create(
            builder, loc, row, macroRowOrigin);
        Value localCol = arith::SubIOp::create(builder, loc, col, kOrigin);
        Value linear = addI64(
            builder, loc, mulI64(builder, loc, localRow, sixteen), localCol);
        Value ptr = LLVM::GEPOp::create(
            builder, loc, (*sharedBase).getType(), inputType, *sharedBase,
            ValueRange{linear});
        return Value(LLVM::LoadOp::create(
            builder, loc, VectorType::get({2}, inputType), ptr, /*alignment=*/2));
      }
      Value linear = addI64(builder, loc, mulI64(builder, loc, row, k), col);
      Value rowValid = lessI64(builder, loc, row, m);
      Value valid0 = arith::AndIOp::create(
          builder, loc, rowValid, lessI64(builder, loc, col, k));
      Value valid1 = arith::AndIOp::create(
          builder, loc, rowValid,
          lessI64(builder, loc, addI64(builder, loc, col, one), k));
      return maskedLoadPair(builder, loc, aBase, inputType, linear,
                            valid0, valid1, 2);
    };
    auto loadB = [&](Value row, Value tileColOrigin) {
      Value fragmentCol = addI64(builder, loc, tileColOrigin, gid);
      if (sharedStaging) {
        Value localRow = arith::SubIOp::create(builder, loc, row, kOrigin);
        Value localCol = arith::SubIOp::create(
            builder, loc, fragmentCol, macroColOrigin);
        Value linear = addI64(
            builder, loc, i64Constant(builder, loc, 512),
            addI64(builder, loc,
                   mulI64(builder, loc, localCol, sixteen), localRow));
        Value ptr = LLVM::GEPOp::create(
            builder, loc, (*sharedBase).getType(), inputType, *sharedBase,
            ValueRange{linear});
        return Value(LLVM::LoadOp::create(
            builder, loc, VectorType::get({2}, inputType), ptr, /*alignment=*/2));
      }
      Value linear = addI64(
          builder, loc, mulI64(builder, loc, fragmentCol, k), row);
      Value colValid = lessI64(builder, loc, fragmentCol, n);
      Value valid0 = arith::AndIOp::create(
          builder, loc, colValid, lessI64(builder, loc, row, k));
      Value valid1 = arith::AndIOp::create(
          builder, loc, colValid,
          lessI64(builder, loc, addI64(builder, loc, row, one), k));
      return maskedLoadPair(builder, loc, bBase, inputType, linear,
                            valid0, valid1, 2);
    };

    Type resultTy = LLVM::LLVMStructType::getLiteral(
        builder.getContext(), {f32, f32, f32, f32});
    SmallVector<NamedAttribute> attrs = {
        builder.getNamedAttr("arch", builder.getStringAttr("sm_120")),
        builder.getNamedAttr("shape", builder.getStringAttr("m16n8k16")),
        builder.getNamedAttr("dtype_ab", builder.getStringAttr(desc.getAType())),
        builder.getNamedAttr("dtype_c", builder.getStringAttr("f32")),
        builder.getNamedAttr("block_scaled", builder.getBoolAttr(false))};
    SmallVector<Value> next;
    SmallVector<Value> aOperands = {loadA(row0, k0), loadA(row1, k0),
                                    loadA(row0, k1), loadA(row1, k1)};
    auto emitMma = [&](Value mmaCol, unsigned accumulatorBase) {
      SmallVector<Value> operands(aOperands.begin(), aOperands.end());
      operands.push_back(loadB(k0, mmaCol));
      operands.push_back(loadB(k1, mmaCol));
      if (desc.getAType() == "bf16")
        for (Value &operand : operands)
          operand = LLVM::BitcastOp::create(
              builder, loc, builder.getI32Type(), operand);
      for (unsigned i = 0; i < 4; ++i)
        operands.push_back(loop.getRegionIterArgs()[accumulatorBase + i]);
      Operation *mma = createContractOp(
          builder, loc, "tessera_nvidia.mma_sync", operands,
          TypeRange{resultTy}, attrs);
      for (int64_t index = 0; index < 4; ++index)
        next.push_back(LLVM::ExtractValueOp::create(
            builder, loc, f32, mma->getResult(0), ArrayRef<int64_t>{index}));
    };
    emitMma(colOrigin, 0);
    if (sharedStaging)
      emitMma(addI64(builder, loc, colOrigin, eight), 4);
    if (sharedStaging)
      NVVM::BarrierOp::create(builder, loc);
    scf::YieldOp::create(builder, loc, next);
  }

  builder.setInsertionPointAfter(loop);
  SmallVector<Value> values(loop.getResults().begin(), loop.getResults().end());
  Value one = i64Constant(builder, loc, 1);
  Value eight = i64Constant(builder, loc, 8);
  Type outputType = epilogue.getOutputType() == "f16" ? f16 : f32;
  unsigned nTiles = sharedStaging ? 2 : 1;
  for (unsigned nTile = 0; nTile < nTiles; ++nTile) {
    unsigned base = nTile * 4;
    Value tileCol = nTile == 0
        ? colOrigin : addI64(builder, loc, colOrigin, eight);
    Value outCol = addI64(builder, loc, tileCol, twoTig);
    Value colValid0 = lessI64(builder, loc, outCol, n);
    Value colValid1 = lessI64(
        builder, loc, addI64(builder, loc, outCol, one), n);

    if (hasBias) {
      Value biasPair = maskedLoadPair(builder, loc, biasBase, f32, outCol,
                                      colValid0, colValid1, 4);
      Value bias0 = LLVM::ExtractElementOp::create(
          builder, loc, f32, biasPair, i64Constant(builder, loc, 0));
      Value bias1 = LLVM::ExtractElementOp::create(
          builder, loc, f32, biasPair, i64Constant(builder, loc, 1));
      values[base] = arith::AddFOp::create(
          builder, loc, values[base], bias0);
      values[base + 1] = arith::AddFOp::create(
          builder, loc, values[base + 1], bias1);
      values[base + 2] = arith::AddFOp::create(
          builder, loc, values[base + 2], bias0);
      values[base + 3] = arith::AddFOp::create(
          builder, loc, values[base + 3], bias1);
    }
    if (epilogue.getActivation() != "none")
      for (unsigned i = 0; i < 4; ++i)
        values[base + i] = tessera::tile::emitScalarFloatActivation(
            builder, loc, values[base + i], epilogue.getActivation());
    if (outputType == f16)
      for (unsigned i = 0; i < 4; ++i)
        values[base + i] = tessera::tile::emitFloatOutputConversion(
            builder, loc, values[base + i], outputType);

    for (unsigned pair = 0; pair < 2; ++pair) {
      Value rowOffset = pair == 0 ? gid : addI64(builder, loc, gid, eight);
      Value row = addI64(builder, loc, rowOrigin, rowOffset);
      Value rowValid = lessI64(builder, loc, row, m);
      Value valid0 = arith::AndIOp::create(builder, loc, rowValid, colValid0);
      Value valid1 = arith::AndIOp::create(builder, loc, rowValid, colValid1);
      Value linear = addI64(builder, loc, mulI64(builder, loc, row, n), outCol);
      Value ptr = LLVM::GEPOp::create(builder, loc, dBase.getType(), outputType,
                                      dBase, ValueRange{linear});
      Value outPair = pairFromScalars(
          builder, loc, outputType, values[base + pair * 2],
          values[base + pair * 2 + 1]);
      LLVM::MaskedStoreOp::create(builder, loc, outPair, ptr,
                                  mask2(builder, loc, valid0, valid1),
                                  outputType == f16 ? 2 : 4);
    }
  }
  op->erase();
  return success();
}

static FailureOr<SmallVector<Value>> materializeSm120Mma16Pack(
    tessera::tile::FragmentPackOp pack, OpBuilder &builder, Value gid,
    Value tig) {
  Operation *op = pack.getOperation();
  auto role = op->getAttrOfType<StringAttr>("role");
  auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
  std::optional<Sm120FragmentDescriptor> physical =
      selectSm120Fragment(desc);
  if (!role || !physical) {
    op->emitError("sm_120 fragment materialization requires a supported "
                  "m16n8k{8,16,32} row/col descriptor");
    return failure();
  }

  auto view = pack.getInputs().front().getDefiningOp<tessera::tile::ViewOp>();
  if (!view) {
    op->emitError("fragment_pack source must be produced by tile.view");
    return failure();
  }
  auto memory = view->getAttrOfType<tessera::tile::TileMemoryLayoutAttr>(
      "tile.memory");
  auto layout = view->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
  if (!memory || !layout || view.getInputs().size() != 3) {
    op->emitError("fragment_pack requires pointer-backed tile.view with "
                  "base, row origin, and column origin");
    return failure();
  }

  StringRef expectedOrder = role.getValue() == "a" ? "row_major" : "col_major";
  SmallVector<int64_t, 2> expectedShape = role.getValue() == "a"
      ? SmallVector<int64_t, 2>{16, desc.getK()}
      : SmallVector<int64_t, 2>{desc.getK(), 8};
  if ((role.getValue() != "a" && role.getValue() != "b") ||
      memory.getOrder() != expectedOrder || memory.getSpace() == "lds" ||
      layout.getShardExtents() != ArrayRef<int64_t>(expectedShape) ||
      layout.getSwizzle()) {
    op->emitError("unsupported sm_120 fragment source layout for role ")
        << role.getValue();
    return failure();
  }

  Value base = view.getInputs()[0];
  Value rowOrigin = view.getInputs()[1];
  Value colOrigin = view.getInputs()[2];
  auto pointerType = dyn_cast<LLVM::LLVMPointerType>(base.getType());
  if (!pointerType ||
      !rowOrigin.getType().isInteger(64) || !colOrigin.getType().isInteger(64)) {
    op->emitError("pointer-backed tile.view requires !llvm.ptr base and "
                  "i64 row/column origins");
    return failure();
  }
  unsigned addressSpace = pointerType.getAddressSpace();
  bool validAddressSpace = memory.getSpace() == "gmem"
      ? (addressSpace == 0 || addressSpace == 1)
      : (memory.getSpace() == "smem" && addressSpace == 3);
  if (!validAddressSpace) {
    op->emitError("tile.memory space does not match the LLVM pointer address space");
    return failure();
  }

  Location loc = op->getLoc();
  MLIRContext *ctx = builder.getContext();
  StringRef dtype = physical->dtype;
  Type inputTy;
  Type loadTy;
  unsigned alignment;
  int64_t columnsPerRegister;
  switch (physical->packing) {
  case Sm120InputPacking::PairF16:
    inputTy = dtype == "bf16" ? Type(BFloat16Type::get(ctx))
                               : Type(Float16Type::get(ctx));
    loadTy = VectorType::get({2}, inputTy);
    alignment = 2;
    columnsPerRegister = 2;
    break;
  case Sm120InputPacking::ScalarF32:
    inputTy = builder.getF32Type();
    loadTy = inputTy;
    alignment = 4;
    columnsPerRegister = 1;
    break;
  case Sm120InputPacking::PackedX4I8:
    inputTy = builder.getI8Type();
    loadTy = VectorType::get({4}, inputTy);
    alignment = 1;
    columnsPerRegister = 4;
    break;
  case Sm120InputPacking::PackedX8E2M1:
    // Canonical NVFP4 storage is nibble-packed: two logical E2M1 codes per
    // byte.  Each OMMA input register covers eight logical elements, so the
    // physical materializer loads four bytes while Tile IR retains logical
    // [16x64]/[64x8] coordinates.
    inputTy = builder.getI8Type();
    loadTy = builder.getI32Type();
    alignment = 4;
    columnsPerRegister = 8;
    break;
  }
  Value leadingDim = i64Constant(builder, loc, memory.getLeadingDim());
  Value eight = i64Constant(builder, loc, 8);
  Value registerColumn = mulI64(
      builder, loc, tig, i64Constant(builder, loc, columnsPerRegister));
  Value halfK = i64Constant(builder, loc, desc.getK() / 2);

  SmallVector<std::pair<Value, Value>> coords;
  if (role.getValue() == "a") {
    coords = {{gid, registerColumn},
              {addI64(builder, loc, gid, eight), registerColumn},
              {gid, addI64(builder, loc, registerColumn, halfK)},
              {addI64(builder, loc, gid, eight),
               addI64(builder, loc, registerColumn, halfK)}};
  } else {
    coords = {{registerColumn, gid},
              {addI64(builder, loc, registerColumn, halfK), gid}};
  }

  SmallVector<Value> fragments;
  for (auto [row, col] : coords) {
    row = addI64(builder, loc, rowOrigin, row);
    col = addI64(builder, loc, colOrigin, col);
    if (physical->packing == Sm120InputPacking::PackedX8E2M1) {
      Value two = i64Constant(builder, loc, 2);
      if (memory.getLeadingDim() != 32) {
        op->emitError("NVFP4 fragment storage requires a nibble-packed 32-byte leading dimension");
        return failure();
      }
      if (role.getValue() == "a")
        col = arith::DivUIOp::create(builder, loc, col, two);
      else
        row = arith::DivUIOp::create(builder, loc, row, two);
    }
    Value linear = memory.getOrder() == "row_major"
        ? addI64(builder, loc, mulI64(builder, loc, row, leadingDim), col)
        : addI64(builder, loc, mulI64(builder, loc, col, leadingDim), row);
    Value ptr = LLVM::GEPOp::create(builder, loc, base.getType(), inputTy, base,
                                    ValueRange{linear});
    Value fragment = LLVM::LoadOp::create(builder, loc, loadTy, ptr, alignment);
    if (dtype != "f16")
      fragment = LLVM::BitcastOp::create(
          builder, loc, builder.getI32Type(), fragment);
    fragments.push_back(fragment);
  }
  unsigned expectedRegisters = role.getValue() == "a"
      ? physical->aRegistersPerLane : physical->bRegistersPerLane;
  if (fragments.size() != expectedRegisters) {
    op->emitError("selected sm_120 fragment packing produced an invalid register count");
    return failure();
  }
  return fragments;
}

static FailureOr<Value> materializeSm120NVFP4ScalePack(
    tessera::tile::FragmentPackOp pack, OpBuilder &builder, Value gid,
    Value tig) {
  Operation *op = pack.getOperation();
  auto role = op->getAttrOfType<StringAttr>("role");
  auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
  auto physical = selectSm120Fragment(desc);
  if (!role || !physical ||
      physical->packing != Sm120InputPacking::PackedX8E2M1 ||
      (role.getValue() != "scale_a" && role.getValue() != "scale_b")) {
    op->emitError("NVFP4 scale materialization requires scale_a/scale_b fragments");
    return failure();
  }
  auto view = pack.getInputs().front().getDefiningOp<tessera::tile::ViewOp>();
  auto memory = view ? view->getAttrOfType<tessera::tile::TileMemoryLayoutAttr>(
                           "tile.memory") : nullptr;
  auto layout = view ? view->getAttrOfType<tessera::tile::TileLayoutAttr>(
                           "tile.layout") : nullptr;
  SmallVector<int64_t, 2> expectedShape = role.getValue() == "scale_a"
      ? SmallVector<int64_t, 2>{16, 4}
      : SmallVector<int64_t, 2>{4, 8};
  StringRef expectedOrder = role.getValue() == "scale_a" ? "row_major"
                                                          : "col_major";
  if (!view || !memory || !layout || view.getInputs().size() != 3 ||
      memory.getSpace() != "gmem" || memory.getOrder() != expectedOrder ||
      memory.getLeadingDim() != 4 ||
      layout.getShardExtents() != ArrayRef<int64_t>(expectedShape) ||
      layout.getSwizzle()) {
    op->emitError("NVFP4 scale fragment requires logical 16x4 row-major A or 4x8 col-major B view");
    return failure();
  }
  Value base = view.getInputs()[0];
  auto pointerType = dyn_cast<LLVM::LLVMPointerType>(base.getType());
  if (!pointerType ||
      (pointerType.getAddressSpace() != 0 && pointerType.getAddressSpace() != 1)) {
    op->emitError("NVFP4 scale view requires a global-memory LLVM pointer");
    return failure();
  }

  Location loc = op->getLoc();
  Value zero32 = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value zero64 = i64Constant(builder, loc, 0);
  Value one = i64Constant(builder, loc, 1);
  Value activeTig = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, tig, zero64);
  Value scaleRow = gid;
  if (role.getValue() == "scale_a") {
    Value isUpper = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, tig, one);
    activeTig = arith::OrIOp::create(builder, loc, activeTig, isUpper);
    scaleRow = arith::SelectOp::create(
        builder, loc, isUpper, addI64(builder, loc, gid,
                                     i64Constant(builder, loc, 8)), gid);
  }
  Value linear = role.getValue() == "scale_a"
      ? mulI64(builder, loc, scaleRow, i64Constant(builder, loc, 4))
      : mulI64(builder, loc, gid, i64Constant(builder, loc, 4));
  Value ptr = LLVM::GEPOp::create(builder, loc, base.getType(),
                                  builder.getI8Type(), base,
                                  ValueRange{linear});
  Value loaded = LLVM::LoadOp::create(builder, loc, builder.getI32Type(), ptr,
                                      /*alignment=*/4);
  return arith::SelectOp::create(builder, loc, activeTig, loaded, zero32)
      .getResult();
}

static LogicalResult materializeSm120AccumulatorStore(
    Operation *mmaTarget, tessera::tile::FragmentUnpackOp unpack,
    tessera::tile::StoreOp store, OpBuilder &builder, Value gid,
    Value twoTig) {
  Operation *op = store.getOperation();
  auto desc = unpack->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
  auto unpackLayout =
      unpack->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
  auto storeLayout =
      store->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
  auto memory =
      store->getAttrOfType<tessera::tile::TileMemoryLayoutAttr>("tile.memory");
  std::array<int64_t, 2> outputShape{16, 8};
  bool isF32 = desc && desc.getAccType() == "f32";
  bool isS32 = desc && (desc.getAccType() == "s32" ||
                        desc.getAccType() == "int32");
  if (!isCanonicalSm120Mma16(desc) || (!isF32 && !isS32) ||
      !unpackLayout || !storeLayout || !memory ||
      unpackLayout.getShardExtents() != ArrayRef<int64_t>(outputShape) ||
      storeLayout.getShardExtents() != ArrayRef<int64_t>(outputShape) ||
      unpackLayout.getSwizzle() ||
      storeLayout.getSwizzle() || memory.getSpace() != "gmem" ||
      memory.getOrder() != "row_major") {
    op->emitError("sm_120 accumulator store requires unswizzled 16x8 row-major "
                  "gmem output and f32 or s32 accumulator");
    return failure();
  }

  Value base = store.getInputs()[1];
  Value rowOrigin = store.getInputs()[2];
  Value colOrigin = store.getInputs()[3];
  auto pointerType = dyn_cast<LLVM::LLVMPointerType>(base.getType());
  if (!pointerType ||
      (pointerType.getAddressSpace() != 0 && pointerType.getAddressSpace() != 1) ||
      !rowOrigin.getType().isInteger(64) || !colOrigin.getType().isInteger(64)) {
    op->emitError("sm_120 accumulator store requires gmem !llvm.ptr and i64 origins");
    return failure();
  }

  Location loc = op->getLoc();
  Type accumulatorTy = isF32 ? Type(builder.getF32Type())
                             : Type(builder.getI32Type());
  Value one = i64Constant(builder, loc, 1);
  Value eight = i64Constant(builder, loc, 8);
  SmallVector<std::pair<Value, Value>> coords = {
      {gid, twoTig},
      {gid, addI64(builder, loc, twoTig, one)},
      {addI64(builder, loc, gid, eight), twoTig},
      {addI64(builder, loc, gid, eight),
       addI64(builder, loc, twoTig, one)}};
  Value leadingDim = i64Constant(builder, loc, memory.getLeadingDim());
  Value result = mmaTarget->getResult(0);
  for (auto [index, coord] : llvm::enumerate(coords)) {
    Value row = addI64(builder, loc, rowOrigin, coord.first);
    Value col = addI64(builder, loc, colOrigin, coord.second);
    Value linear = addI64(builder, loc,
                          mulI64(builder, loc, row, leadingDim), col);
    Value ptr = LLVM::GEPOp::create(builder, loc, base.getType(), accumulatorTy, base,
                                    ValueRange{linear});
    Value scalar = LLVM::ExtractValueOp::create(
        builder, loc, accumulatorTy, result,
        ArrayRef<int64_t>{static_cast<int64_t>(index)});
    LLVM::StoreOp::create(builder, loc, scalar, ptr, /*alignment=*/4);
  }
  return success();
}

struct LowerTileToNVIDIAPass
    : PassWrapper<LowerTileToNVIDIAPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToNVIDIAPass)

  LowerTileToNVIDIAPass() = default;
  explicit LowerTileToNVIDIAPass(int sm) { smVersion = sm; }
  LowerTileToNVIDIAPass(const LowerTileToNVIDIAPass &other)
      : PassWrapper(other) {
    smVersion = other.smVersion;
  }

  Option<int> smVersion{*this, "sm",
                        llvm::cl::desc("Target NVIDIA SM version"),
                        llvm::cl::init(kHopperSM)};

  StringRef getArgument() const final { return "lower-tile-to-nvidia"; }

  StringRef getDescription() const final {
    return "Lower Tessera Tile IR to NVIDIA Hopper/Blackwell Target IR contracts";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    LLVM::LLVMDialect, math::MathDialect, NVVM::NVVMDialect,
                    scf::SCFDialect,
                    tessera::nvidia::TesseraNVIDIADialect>();
  }

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    ctx->loadDialect<tessera::nvidia::TesseraNVIDIADialect>();

    ModuleOp module = getOperation();
    OpBuilder moduleBuilder(ctx);
    module->setAttr("tessera.nvidia.arch",
                    moduleBuilder.getStringAttr(archStringForSM(smVersion)));

    SmallVector<Operation *> worklist;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.mma" || name == "tile.async_copy" ||
          name == "tile.matmul_kernel" ||
          name == "tile.wait_async" || name == "tile.conv2d" ||
          name == "tile.kv_cache" || name.starts_with("tile.control_") ||
          name.starts_with("tile.tmem."))
        worklist.push_back(op);
    });

    Value lastAsyncToken;
    for (Operation *op : worklist) {
      OpBuilder builder(op);
      Location loc = op->getLoc();
      StringRef name = op->getName().getStringRef();
      SmallVector<NamedAttribute> attrs;
      attrs.push_back(builder.getNamedAttr("arch",
                                           builder.getStringAttr(archStringForSM(smVersion))));

      if (name.starts_with("tile.tmem.") &&
          (smVersion < kBlackwellSM || smVersion >= kConsumerBlackwellSM)) {
        // TMEM is datacenter-Blackwell-only (sm_100a). Consumer Blackwell
        // sm_120 has no TMEM — it is NOT a superset of sm_100.
        op->emitError("NVIDIA TMEM lowering requires datacenter Blackwell "
                      "SM100 (consumer sm_120 has no TMEM)");
        signalPassFailure();
        return;
      }

      if (isTileOp(op, "tile.matmul_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120MatmulKernel(
                cast<tessera::tile::MatmulKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError("tile.matmul_kernel currently requires sm_120");
          signalPassFailure();
          return;
        }
        continue;
      }

      if (isTileOp(op, "tile.mma")) {
        bool typedFragmentForm = llvm::any_of(op->getOperandTypes(), [](Type type) {
              return isa<tessera::tile::FragmentType>(type);
            });
        if (typedFragmentForm) {
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
          auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
          std::optional<Sm120FragmentDescriptor> physical =
              selectSm120Fragment(desc);
          bool isNVFP4 = physical &&
              physical->packing == Sm120InputPacking::PackedX8E2M1;
          unsigned expectedOperands = isNVFP4 ? 5 : 3;
          if (smVersion < kConsumerBlackwellSM ||
              op->getNumOperands() != expectedOperands ||
              op->getNumResults() != 1 ||
              (!op->getResult(0).use_empty() && !hasOutputStore)) {
            op->emitError("initial typed fragment lowering requires sm_120, "
                          "logical A/B/zero-C operands (plus NVFP4 scales), and either an unused result or "
                          "fragment_unpack -> tile.store");
            signalPassFailure();
            return;
          }
          auto aPack = op->getOperand(0).getDefiningOp<tessera::tile::FragmentPackOp>();
          auto bPack = op->getOperand(1).getDefiningOp<tessera::tile::FragmentPackOp>();
          auto cZero = op->getOperand(2).getDefiningOp<tessera::tile::FragmentZeroOp>();
          tessera::tile::FragmentPackOp scaleAPack;
          tessera::tile::FragmentPackOp scaleBPack;
          if (isNVFP4) {
            scaleAPack = op->getOperand(3).getDefiningOp<
                tessera::tile::FragmentPackOp>();
            scaleBPack = op->getOperand(4).getDefiningOp<
                tessera::tile::FragmentPackOp>();
          }
          if (!aPack || !bPack || !cZero || !physical) {
            op->emitError("typed sm_120 lowering requires fragment_pack A/B "
                          "and fragment_zero accumulator with one descriptor");
            signalPassFailure();
            return;
          }
          if (isNVFP4 && (!scaleAPack || !scaleBPack)) {
            op->emitError("typed NVFP4 lowering requires logical scale_a and scale_b fragment packs");
            signalPassFailure();
            return;
          }
          if (hasOutputStore && desc.getAccType() != "f32" &&
              desc.getAccType() != "s32" && desc.getAccType() != "int32") {
            op->emitError("sm_120 fragment unpack/store requires f32 or s32 accumulator");
            signalPassFailure();
            return;
          }

          Value tid = NVVM::ThreadIdXOp::create(builder, loc, builder.getI32Type());
          Value lane = arith::AndIOp::create(
              builder, loc, tid,
              arith::ConstantIntOp::create(builder, loc, 31, 32));
          Value gid32 = arith::ShRUIOp::create(
              builder, loc, lane,
              arith::ConstantIntOp::create(builder, loc, 2, 32));
          Value tig32 = arith::AndIOp::create(
              builder, loc, lane,
              arith::ConstantIntOp::create(builder, loc, 3, 32));
          Value twoTig32 = arith::MulIOp::create(
              builder, loc, tig32,
              arith::ConstantIntOp::create(builder, loc, 2, 32));
          Value gid = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), gid32);
          Value tig = arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                             tig32);
          Value twoTig = arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                                twoTig32);
          FailureOr<SmallVector<Value>> a =
              materializeSm120Mma16Pack(aPack, builder, gid, tig);
          FailureOr<SmallVector<Value>> b =
              materializeSm120Mma16Pack(bPack, builder, gid, tig);
          if (failed(a) || failed(b)) {
            signalPassFailure();
            return;
          }
          FailureOr<Value> scaleA = failure();
          FailureOr<Value> scaleB = failure();
          if (isNVFP4) {
            scaleA = materializeSm120NVFP4ScalePack(
                scaleAPack, builder, gid, tig);
            scaleB = materializeSm120NVFP4ScalePack(
                scaleBPack, builder, gid, tig);
            if (failed(scaleA) || failed(scaleB)) {
              signalPassFailure();
              return;
            }
          }

          Type fragTy = desc.getAType() == "f16"
              ? Type(VectorType::get({2}, Float16Type::get(ctx)))
              : Type(builder.getI32Type());
          SmallVector<Value> operands(*a);
          operands.append(b->begin(), b->end());
          Type resultTy;
          if (desc.getAccType() == "f32") {
            Value zero = arith::ConstantFloatOp::create(
                builder, loc, builder.getF32Type(), APFloat(0.0f));
            operands.append(4, zero);
            resultTy = LLVM::LLVMStructType::getLiteral(
                ctx, {builder.getF32Type(), builder.getF32Type(),
                      builder.getF32Type(), builder.getF32Type()});
          } else if (desc.getAccType() == "f16") {
            Value zero = arith::ConstantOp::create(
                builder, loc, fragTy, builder.getZeroAttr(fragTy));
            operands.push_back(zero);
            operands.push_back(zero);
            resultTy = LLVM::LLVMStructType::getLiteral(ctx, {fragTy, fragTy});
          } else {
            Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
            operands.append(4, zero);
            resultTy = LLVM::LLVMStructType::getLiteral(
                ctx, {builder.getI32Type(), builder.getI32Type(),
                      builder.getI32Type(), builder.getI32Type()});
          }
          if (isNVFP4) {
            operands.push_back(*scaleA);
            operands.push_back(*scaleB);
          }
          std::string shape = "m16n8k" + std::to_string(desc.getK());
          SmallVector<NamedAttribute> mmaAttrs = {
              builder.getNamedAttr("arch", builder.getStringAttr("sm_120")),
              builder.getNamedAttr("shape", builder.getStringAttr(shape)),
              builder.getNamedAttr("dtype_ab",
                                   builder.getStringAttr(desc.getAType())),
              builder.getNamedAttr("dtype_c",
                                   builder.getStringAttr(desc.getAccType())),
              builder.getNamedAttr(
                  "instruction_family",
                  builder.getStringAttr(physical->instructionFamily)),
              builder.getNamedAttr(
                  "a_registers_per_lane",
                  builder.getI32IntegerAttr(physical->aRegistersPerLane)),
              builder.getNamedAttr(
                  "b_registers_per_lane",
                  builder.getI32IntegerAttr(physical->bRegistersPerLane)),
              builder.getNamedAttr(
                  "accumulator_registers_per_lane",
                  builder.getI32IntegerAttr(
                      physical->accumulatorRegistersPerLane)),
              builder.getNamedAttr("block_scaled", builder.getBoolAttr(isNVFP4))};
          if (isNVFP4) {
            mmaAttrs.push_back(builder.getNamedAttr(
                "scale_dtype", builder.getStringAttr("ue4m3")));
            mmaAttrs.push_back(builder.getNamedAttr(
                "scale_vector", builder.getStringAttr("4X")));
          }
          Operation *target = createContractOp(
              builder, loc,
              isNVFP4 ? "tessera_nvidia.nvfp4_block_scale_mma"
                      : "tessera_nvidia.mma_sync",
              operands,
              TypeRange{resultTy}, mmaAttrs);
          if (hasOutputStore &&
              failed(materializeSm120AccumulatorStore(
                  target, unpack, store, builder, gid, twoTig))) {
            signalPassFailure();
            return;
          }

          Operation *aView = aPack.getInputs().front().getDefiningOp();
          Operation *bView = bPack.getInputs().front().getDefiningOp();
          Operation *scaleAView = isNVFP4
              ? scaleAPack.getInputs().front().getDefiningOp() : nullptr;
          Operation *scaleBView = isNVFP4
              ? scaleBPack.getInputs().front().getDefiningOp() : nullptr;
          if (hasOutputStore) {
            store->erase();
            unpack->erase();
          }
          op->erase();
          if (aPack.getResult().use_empty())
            aPack->erase();
          if (bPack.getResult().use_empty())
            bPack->erase();
          if (cZero.getResult().use_empty())
            cZero->erase();
          if (scaleAPack && scaleAPack.getResult().use_empty())
            scaleAPack->erase();
          if (scaleBPack && scaleBPack.getResult().use_empty())
            scaleBPack->erase();
          if (aView && aView->use_empty())
            aView->erase();
          if (bView && bView != aView && bView->use_empty())
            bView->erase();
          if (scaleAView && scaleAView->use_empty())
            scaleAView->erase();
          if (scaleBView && scaleBView != scaleAView && scaleBView->use_empty())
            scaleBView->erase();
          continue;
        }
        if (op->getNumOperands() < 2 || op->getNumResults() > 1) {
          op->emitError("NVIDIA lowering requires tile.mma(lhs, rhs) -> optional result");
          signalPassFailure();
          return;
        }

        TypeRange resultTypes = op->getResultTypes();
        ValueRange operands = op->getOperands();
        if (smVersion >= kConsumerBlackwellSM) {
          // Consumer Blackwell (RTX 50-series, sm_120): warp-level `mma.sync`.
          // NOT a superset of datacenter sm_100 — no tcgen05/TMEM, no Hopper
          // wgmma (FP4 rides `mma.sync.aligned...block_scale`). Grounded in
          // gpu_target._CUDA_13_3_FEATURES[SM_120]. Mirrors the Python emitter
          // target_ir.py::_lower_nvidia_op; the tma_async_copy + mbarrier
          // companions come from the separate tile.async_copy / tile.wait_async
          // ops in the worklist (as with the wgmma path below).
          // The canonical fragment form is deliberately explicit at Tile IR:
          // A[4], B[2], C[2] all use vector<2xf16>, with the struct result
          // required by NVVM's m16n8k16 f16 MMA builder.  Do not synthesize
          // fragments from scalar/tensor values here: Tile IR must first carry
          // a real lane/layout pack, otherwise that would invent semantics.
          bool fragmentForm = operands.size() == 8 && resultTypes.size() == 1;
          auto fragTy = VectorType::get({2}, Float16Type::get(ctx));
          if (fragmentForm) {
            for (Value operand : operands)
              fragmentForm &= operand.getType() == fragTy;
            auto resultTy = dyn_cast<LLVM::LLVMStructType>(resultTypes.front());
            fragmentForm &= resultTy && resultTy.getBody().size() == 2 &&
                resultTy.getBody()[0] == fragTy && resultTy.getBody()[1] == fragTy;
          }
          attrs.push_back(builder.getNamedAttr("shape",
                                               builder.getStringAttr("m16n8k16")));
          attrs.push_back(builder.getNamedAttr("dtype_ab",
                                               builder.getStringAttr(fragmentForm ? "f16" : "bf16")));
          attrs.push_back(builder.getNamedAttr("dtype_c",
                                               builder.getStringAttr(fragmentForm ? "f16" : "f32")));
          attrs.push_back(builder.getNamedAttr("block_scaled",
                                               builder.getBoolAttr(false)));
          Operation *target = createContractOp(builder, loc,
                                               "tessera_nvidia.mma_sync",
                                               operands, resultTypes, attrs);
          op->replaceAllUsesWith(target->getResults());
          op->erase();
          continue;
        }
        if (smVersion >= kBlackwellSM) {
          attrs.push_back(builder.getNamedAttr("shape",
                                               builder.getStringAttr("m128n128k32")));
          attrs.push_back(builder.getNamedAttr("accum",
                                               builder.getStringAttr("tmem_f32")));
          attrs.push_back(builder.getNamedAttr("cta_group",
                                               builder.getI64IntegerAttr(2)));
          attrs.push_back(builder.getNamedAttr("source",
                                               builder.getStringAttr("tessera.matmul")));
          auto alloc = createContractOp(builder, loc, "tessera_nvidia.tmem_alloc",
                                        ValueRange{}, TypeRange{},
                                        {builder.getNamedAttr("columns",
                                                              builder.getI64IntegerAttr(128)),
                                         builder.getNamedAttr("arch",
                                                              builder.getStringAttr(archStringForSM(smVersion)))});
          (void)alloc;
          Operation *target = createContractOp(builder, loc,
                                               "tessera_nvidia.tcgen05_mma",
                                               operands, resultTypes, attrs);
          op->replaceAllUsesWith(target->getResults());
          op->erase();
          continue;
        }

        if (smVersion >= kHopperSM) {
          attrs.push_back(builder.getNamedAttr("shape",
                                               builder.getStringAttr("m64n64k16")));
          attrs.push_back(builder.getNamedAttr("dtype_ab",
                                               builder.getStringAttr("bf16")));
          attrs.push_back(builder.getNamedAttr("dtype_c",
                                               builder.getStringAttr("f32")));
          attrs.push_back(builder.getNamedAttr("warpgroup",
                                               builder.getI64IntegerAttr(4)));
          Operation *target = createContractOp(builder, loc,
                                               "tessera_nvidia.wgmma",
                                               operands, resultTypes, attrs);
          op->replaceAllUsesWith(target->getResults());
          op->erase();
          continue;
        }

        attrs.push_back(builder.getNamedAttr("shape",
                                             builder.getStringAttr("m16n16k16")));
        Operation *target = createContractOp(builder, loc, "tessera_nvidia.wmma",
                                             operands, resultTypes, attrs);
        op->replaceAllUsesWith(target->getResults());
        op->erase();
        continue;
      }

      if (isTileOp(op, "tile.async_copy")) {
        if (smVersion < kHopperSM) {
          op->emitError("NVIDIA TMA lowering requires Hopper SM90+");
          signalPassFailure();
          return;
        }
        attrs.push_back(builder.getNamedAttr("src_space",
                                             builder.getStringAttr("global")));
        attrs.push_back(builder.getNamedAttr("dst_space",
                                             builder.getStringAttr("shared")));
        attrs.push_back(builder.getNamedAttr("bytes",
                                             builder.getI64IntegerAttr(16)));
        Operation *target = createContractOp(builder, loc,
                                             "tessera_nvidia.tma_async_copy",
                                             op->getOperands(), op->getResultTypes(),
                                             attrs);
        if (target->getNumResults() > 0)
          lastAsyncToken = target->getResult(0);
        op->replaceAllUsesWith(target->getResults());
        op->erase();
        continue;
      }

      if (isTileOp(op, "tile.wait_async")) {
        if (smVersion < kHopperSM) {
          op->emitError("NVIDIA mbarrier lowering requires Hopper SM90+");
          signalPassFailure();
          return;
        }
        if (lastAsyncToken)
          createContractOp(builder, loc, "tessera_nvidia.mbarrier",
                           ValueRange{lastAsyncToken}, TypeRange{}, attrs);
        else
          createContractOp(builder, loc, "tessera_nvidia.mbarrier",
                           ValueRange{}, TypeRange{}, attrs);
        op->erase();
        continue;
      }

      // Structured kernels that do not map to a single NVIDIA matrix or
      // movement instruction still need a typed Target-IR contract.  Keep
      // their SSA signature intact and identify the canonical source op so a
      // later executable CUDA lowering can select the kernel implementation.
      // This is deliberately hardware-free Target-IR codegen evidence, not an
      // execution or linked-PTX claim.
      if (isTileOp(op, "tile.conv2d") || isTileOp(op, "tile.kv_cache")) {
        StringRef source = "tessera.conv2d_nhwc";
        if (isTileOp(op, "tile.kv_cache")) {
          source = "tessera.kv_cache.read";
          if (auto sourceAttr = op->getAttrOfType<StringAttr>("source"))
            source = sourceAttr.getValue();
        }
        attrs.push_back(builder.getNamedAttr("source",
                                             builder.getStringAttr(source)));
        Operation *target = createContractOp(
            builder, loc, "tessera_nvidia.cuda_kernel", op->getOperands(),
            op->getResultTypes(), attrs);
        op->replaceAllUsesWith(target->getResults());
        op->erase();
        continue;
      }

      if (name.starts_with("tile.control_")) {
        for (NamedAttribute attr : op->getAttrs())
          if (attr.getName() != "arch")
            attrs.push_back(attr);
        StringRef targetName = "tessera_nvidia.control_for";
        if (name == "tile.control_if")
          targetName = "tessera_nvidia.control_if";
        else if (name == "tile.control_while")
          targetName = "tessera_nvidia.control_while";
        else if (name == "tile.control_scan")
          targetName = "tessera_nvidia.control_scan";
        Operation *target = createContractOp(
            builder, loc, targetName, op->getOperands(), op->getResultTypes(),
            attrs);
        op->replaceAllUsesWith(target->getResults());
        op->erase();
        continue;
      }

      if (name.starts_with("tile.tmem.")) {
        StringRef contractName = "tessera_nvidia.tmem_store";
        if (name == "tile.tmem.alloc")
          contractName = "tessera_nvidia.tmem_alloc";
        else if (name == "tile.tmem.load")
          contractName = "tessera_nvidia.tmem_load";
        createContractOp(builder, loc, contractName, op->getOperands(),
                         op->getResultTypes(), attrs);
        op->erase();
      }
    }

    Operation *unloweredFragment = nullptr;
    module.walk([&](Operation *candidate) {
      StringRef candidateName = candidate->getName().getStringRef();
      if (!unloweredFragment &&
          (candidateName == "tile.fragment_pack" ||
           candidateName == "tile.fragment_zero" ||
           candidateName == "tile.fragment_unpack" ||
           candidateName == "tile.store"))
        unloweredFragment = candidate;
    });
    if (unloweredFragment) {
      unloweredFragment->emitError(
          "NVIDIA lowering left an unsupported Tile fragment operation");
      signalPassFailure();
    }
  }
};

static LLVM::LLVMFuncOp declareVoidMarker(ModuleOp module, StringRef name) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto fnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(module.getContext()), {}, false);
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

static StringRef markerForTargetOp(StringRef opName) {
  if (opName == "tessera_nvidia.wgmma")
    return "llvm.nvvm.wgmma.contract";
  if (opName == "tessera_nvidia.tma_async_copy")
    return "llvm.nvvm.cp.async.bulk.tensor.contract";
  if (opName == "tessera_nvidia.mbarrier")
    return "llvm.nvvm.mbarrier.contract";
  if (opName == "tessera_nvidia.wmma")
    return "llvm.nvvm.mma.sync.contract";
  if (opName == "tessera_nvidia.mma_sync")  // consumer Blackwell sm_120 warp-level MMA
    return "llvm.nvvm.mma.sync.contract";
  if (opName == "tessera_nvidia.nvfp4_block_scale_mma")
    return "llvm.nvvm.mma.sync.block.scale.contract";
  if (opName == "tessera_nvidia.tcgen05_mma")
    return "llvm.nvvm.tcgen05.mma.contract";
  if (opName == "tessera_nvidia.tmem_alloc")
    return "llvm.nvvm.tmem.alloc.contract";
  if (opName == "tessera_nvidia.tmem_load")
    return "llvm.nvvm.tmem.load.contract";
  if (opName == "tessera_nvidia.tmem_store")
    return "llvm.nvvm.tmem.store.contract";
  if (opName == "tessera_nvidia.cuda_kernel")
    return "llvm.nvvm.cuda.kernel.contract";
  if (opName.starts_with("tessera_nvidia.control_"))
    return "llvm.nvvm.cuda.control.contract";
  return "llvm.nvvm.tessera.nvidia.diagnostic.contract";
}

// Increment 2 of the Tile IR / native Target IR tail: emit a *real*
// `nvvm.mma.sync` for the canonical m16n8k16 f16 fragment contract instead of a
// void marker. This is the codegen-contract end of the NVIDIA Target IR path —
// the emitted op is validated by the NVVM verifier (fragment counts / types /
// result struct), a real structural correctness signal even without a device.
//
// It fires only when `mma_sync` already carries the fragment operands. Both the
// f16 accumulator ABI (C:2 vector<2xf16>) and the execution-proven f32 ABI
// (C:4 f32, four-f32 struct result) are accepted. The abstract tile->target form
// (scalar operands, dtype_ab="bf16") does NOT match and falls through to the
// honest marker (Decision #21: never silently emit a different / wrong kernel).
// f16/bf16, tf32, fp8, and int8 all use four A and two B registers for the
// canonical sm_120 shapes. Their physical register types and K extents differ.
static bool tryLowerMmaSyncToNVVM(Operation *op, OpBuilder &builder) {
  if (op->getName().getStringRef() ==
      "tessera_nvidia.nvfp4_block_scale_mma") {
    auto shape = op->getAttrOfType<StringAttr>("shape");
    auto dtypeC = op->getAttrOfType<StringAttr>("dtype_c");
    auto blockScaled = op->getAttrOfType<BoolAttr>("block_scaled");
    if (!shape || shape.getValue() != "m16n8k64" || !dtypeC ||
        dtypeC.getValue() != "f32" || !blockScaled ||
        !blockScaled.getValue() || op->getNumOperands() != 12 ||
        op->getNumResults() != 1)
      return false;
    auto structTy = dyn_cast<LLVM::LLVMStructType>(op->getResult(0).getType());
    if (!structTy || structTy.getBody().size() != 4 ||
        llvm::any_of(structTy.getBody(), [](Type type) { return !type.isF32(); }))
      return false;
    ValueRange operands = op->getOperands();
    for (Value value : operands.take_front(6))
      if (!value.getType().isInteger(32))
        return false;
    for (Value value : operands.slice(6, 4))
      if (!value.getType().isF32())
        return false;
    for (Value value : operands.take_back(2))
      if (!value.getType().isInteger(32))
        return false;
    builder.setInsertionPoint(op);
    std::string assembly =
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale."
        "scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, "
        "{$14}, {0, 0}, {$15}, {0, 0};";
    auto inlineMma = LLVM::InlineAsmOp::create(
        builder, op->getLoc(), structTy, operands, assembly,
        "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3,r,r",
        /*has_side_effects=*/false, /*is_align_stack=*/false,
        LLVM::tailcallkind::TailCallKind::None, /*asm_dialect=*/nullptr,
        /*operand_attrs=*/nullptr);
    op->replaceAllUsesWith(inlineMma.getOperation()->getResults());
    op->erase();
    return true;
  }
  if (op->getName().getStringRef() != "tessera_nvidia.mma_sync")
    return false;
  auto shape = op->getAttrOfType<StringAttr>("shape");
  auto dtypeAB = op->getAttrOfType<StringAttr>("dtype_ab");
  auto dtypeC = op->getAttrOfType<StringAttr>("dtype_c");
  if (!shape || !dtypeAB)
    return false;
  StringRef dtype = dtypeAB.getValue();
  bool isF16 = dtype == "f16";
  bool isBF16 = dtype == "bf16";
  bool isTF32 = dtype == "tf32";
  bool isFP8 = dtype == "e4m3" || dtype == "e5m2";
  bool isS8 = dtype == "s8" || dtype == "int8";
  if (!((shape.getValue() == "m16n8k16" && (isF16 || isBF16)) ||
        (shape.getValue() == "m16n8k8" && isTF32) ||
        (shape.getValue() == "m16n8k32" && (isFP8 || isS8))))
    return false;
  if (op->getNumResults() != 1)
    return false;
  auto structTy = dyn_cast<LLVM::LLVMStructType>(op->getResult(0).getType());
  if (!structTy)
    return false;
  ValueRange operands = op->getOperands();
  Type fragTy = isF16
      ? Type(VectorType::get({2}, Float16Type::get(op->getContext())))
      : Type(IntegerType::get(op->getContext(), 32));
  bool isF32Accumulator = dtypeC && dtypeC.getValue() == "f32";
  bool isF16Accumulator = dtypeC && dtypeC.getValue() == "f16";
  bool isS32Accumulator = dtypeC &&
      (dtypeC.getValue() == "s32" || dtypeC.getValue() == "int32");
  if ((!isS8 && !isF32Accumulator && !(isF16 && isF16Accumulator)) ||
      (isS8 && !isS32Accumulator))
    return false;
  unsigned cCount = isF16Accumulator ? 2 : 4;
  if (operands.size() != 6 + cCount)
    return false;
  for (Value v : operands.take_front(6))
    if (v.getType() != fragTy)
      return false;
  Type cType = isF32Accumulator
      ? Type(Float32Type::get(op->getContext()))
      : (isS32Accumulator ? Type(IntegerType::get(op->getContext(), 32))
                          : Type(fragTy));
  for (Value v : operands.drop_front(6))
    if (v.getType() != cType)
      return false;
  ArrayRef<Type> resultBody = structTy.getBody();
  if (resultBody.size() != cCount ||
      llvm::any_of(resultBody, [&](Type type) { return type != cType; }))
    return false;

  SmallVector<Value> a(operands.begin(), operands.begin() + 4);
  SmallVector<Value> b(operands.begin() + 4, operands.begin() + 6);
  SmallVector<Value> c(operands.begin() + 6, operands.end());
  builder.setInsertionPoint(op);
  // LLVM 22 exposes FP8 MMA enums but its NVVM MmaOp verifier does not yet
  // admit the m16n8k32 FP8 shapes accepted by CUDA 13.3 and sm_120. Keep the
  // same typed i32/f32 contract and legalize only that gap through inline PTX.
  if (isFP8) {
    SmallVector<Value> asmOperands(a);
    asmOperands.append(b.begin(), b.end());
    asmOperands.append(c.begin(), c.end());
    std::string assembly =
        "mma.sync.aligned.m16n8k32.row.col.f32." + dtype.str() + "." +
        dtype.str() +
        ".f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, "
        "{$10,$11,$12,$13};";
    auto inlineMma = LLVM::InlineAsmOp::create(
        builder, op->getLoc(), structTy, ValueRange(asmOperands), assembly,
        "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3",
        /*has_side_effects=*/false, /*is_align_stack=*/false,
        LLVM::tailcallkind::TailCallKind::None, /*asm_dialect=*/nullptr,
        /*operand_attrs=*/nullptr);
    op->replaceAllUsesWith(inlineMma.getOperation()->getResults());
    op->erase();
    return true;
  }
  NVVM::MMATypes inputPtxType = NVVM::MMATypes::f16;
  if (isBF16)
    inputPtxType = NVVM::MMATypes::bf16;
  else if (isTF32)
    inputPtxType = NVVM::MMATypes::tf32;
  else if (dtype == "e4m3")
    inputPtxType = NVVM::MMATypes::e4m3;
  else if (dtype == "e5m2")
    inputPtxType = NVVM::MMATypes::e5m2;
  else if (isS8)
    inputPtxType = NVVM::MMATypes::s8;
  int64_t k = shape.getValue() == "m16n8k8" ? 8 :
              (shape.getValue() == "m16n8k32" ? 32 : 16);
  std::optional<NVVM::MMAIntOverflow> intOverflow = std::nullopt;
  if (isS8)
    intOverflow = NVVM::MMAIntOverflow::wrapped;
  auto mma = builder.create<NVVM::MmaOp>(
      op->getLoc(), structTy, ValueRange(a), ValueRange(b), ValueRange(c),
      ArrayRef<int64_t>{16, 8, k}, /*b1Op=*/std::nullopt,
      /*intOverflow=*/intOverflow,
      /*multiplicandPtxTypes=*/
      std::array<NVVM::MMATypes, 2>{inputPtxType, inputPtxType},
      /*multiplicandLayouts=*/
      std::array<NVVM::MMALayout, 2>{NVVM::MMALayout::row, NVVM::MMALayout::col});
  op->replaceAllUsesWith(mma.getOperation()->getResults());
  op->erase();
  return true;
}

struct LowerNVIDIAToNVVMPass
    : PassWrapper<LowerNVIDIAToNVVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerNVIDIAToNVVMPass)

  StringRef getArgument() const final { return "lower-tessera-nvidia-to-nvvm"; }

  StringRef getDescription() const final {
    return "Lower Tessera NVIDIA Target IR contracts to LLVM/NVVM artifact markers";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }

  void runOnOperation() final {
    getContext().loadDialect<LLVM::LLVMDialect, NVVM::NVVMDialect>();
    ModuleOp module = getOperation();
    SmallVector<Operation *> targetOps;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("tessera_nvidia."))
        targetOps.push_back(op);
    });

    for (Operation *op : targetOps) {
      OpBuilder builder(op);
      // Real NVVM emission for the fragment-typed mma_sync contract; the abstract
      // marker form falls through to the void-marker path below.
      if (tryLowerMmaSyncToNVVM(op, builder))
        continue;
      auto marker = declareVoidMarker(module, markerForTargetOp(op->getName().getStringRef()));
      builder.create<LLVM::CallOp>(op->getLoc(), TypeRange{},
                                   SymbolRefAttr::get(marker), ValueRange{});
      if (!op->use_empty())
        op->dropAllUses();
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerTileToNVIDIAPass(int sm) {
  return std::make_unique<LowerTileToNVIDIAPass>(sm);
}

std::unique_ptr<Pass> createLowerNVIDIAToNVVMPass() {
  return std::make_unique<LowerNVIDIAToNVVMPass>();
}

void buildTesseraNVIDIABackendPipeline(OpPassManager &pm) {
  pm.addPass(createLowerTileToNVIDIAPass(kHopperSM));
  pm.addPass(createLowerNVIDIAToNVVMPass());
}

void buildTesseraHopperBackendPipeline(OpPassManager &pm) {
  pm.addPass(createLowerTileToNVIDIAPass(kHopperSM));
  pm.addPass(createLowerNVIDIAToNVVMPass());
}

void buildTesseraBlackwellBackendPipeline(OpPassManager &pm) {
  pm.addPass(createLowerTileToNVIDIAPass(kBlackwellSM));
  pm.addPass(createLowerNVIDIAToNVVMPass());
}

void registerTesseraNVIDIABackendPasses() {
  registerPass([]() { return createLowerTileToNVIDIAPass(kHopperSM); });
  registerPass([]() { return createLowerNVIDIAToNVVMPass(); });

  PassPipelineRegistration<> nvidiaPipeline(
      "tessera-lower-to-nvidia",
      "Lower Tessera Tile IR to NVIDIA NVVM/PTX artifact contracts",
      [](OpPassManager &pm) { buildTesseraNVIDIABackendPipeline(pm); });
  PassPipelineRegistration<> hopperPipeline(
      "tessera-lower-to-hopper",
      "Lower Tessera Tile IR to Hopper WGMMA/TMA artifact contracts",
      [](OpPassManager &pm) { buildTesseraHopperBackendPipeline(pm); });
  PassPipelineRegistration<> blackwellPipeline(
      "tessera-lower-to-blackwell",
      "Lower Tessera Tile IR to Blackwell TCGEN05/TMEM artifact contracts",
      [](OpPassManager &pm) { buildTesseraBlackwellBackendPipeline(pm); });
}

void registerTesseraNVIDIABackendDialects(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                  NVVM::NVVMDialect, tessera::nvidia::TesseraNVIDIADialect,
                  tessera::tile::TesseraTileDialect>();
}

} // namespace tessera
