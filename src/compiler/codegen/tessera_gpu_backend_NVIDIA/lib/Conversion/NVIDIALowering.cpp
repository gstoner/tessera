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
  ScalarF64,
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
      desc.getAType() != desc.getBType() ||
      desc.getALayout() != "row_major" || desc.getBLayout() != "col_major" ||
      desc.getKBlocks() != 1)
    return std::nullopt;
  StringRef dtype = desc.getAType();
  if (dtype == "f64" && desc.getM() == 8 && desc.getN() == 8 &&
      desc.getK() == 4 && desc.getAccType() == "f64")
    return Sm120FragmentDescriptor{
        dtype, 4, "f64", Sm120InputPacking::ScalarF64, 1, 1, 2,
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"};
  if (desc.getM() != 16 || desc.getN() != 8)
    return std::nullopt;
  if (dtype == "nvfp4" && desc.getK() == 64 &&
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
  if ((dtype == "e2m3" || dtype == "e3m2") && desc.getK() == 32 &&
      desc.getAccType() == "f32")
    return Sm120FragmentDescriptor{
        dtype, 32, "f32", Sm120InputPacking::PackedX4I8, 4, 2, 4,
        dtype == "e2m3"
            ? "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale"
              ".scale_vec::1X.f32.e2m3.e2m3.f32.ue8m0"
            : "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale"
              ".scale_vec::1X.f32.e3m2.e3m2.f32.ue8m0"};
  if (dtype == "fp4_e2m1" && desc.getK() == 64 &&
      desc.getAccType() == "f32")
    return Sm120FragmentDescriptor{
        dtype, 64, "f32", Sm120InputPacking::PackedX8E2M1, 4, 2, 4,
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale"
        ".scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0"};
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

static FailureOr<Value> sm120ReductionScratch(Operation *anchor,
                                              OpBuilder &builder) {
  ModuleOp module = anchor->getParentOfType<ModuleOp>();
  if (!module)
    return failure();
  constexpr StringLiteral symbol = "__tessera_sm120_reduce_scratch_f32";
  auto global = module.lookupSymbol<LLVM::GlobalOp>(symbol);
  auto arrayTy = LLVM::LLVMArrayType::get(builder.getF32Type(), 128);
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

// Launch-level general-shape NVFP4 materialization.  The portable ABI carries
// packed E2M1 A/B, logical UE4M3 scale-A/scale-B views, D, and runtime M/N/K.
// One warp owns each 16x8 output tile and accumulates K64 fragments.  Ragged
// matrix/scale reads zero-fill and output stores are masked.
static LogicalResult materializeSm120Nvfp4MatmulKernel(
    tessera::tile::MatmulKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
  auto epilogue = op->getAttrOfType<tessera::tile::TileEpilogueAttr>("epilogue");
  std::optional<Sm120FragmentDescriptor> physical = selectSm120Fragment(desc);
  if (!physical || physical->packing != Sm120InputPacking::PackedX8E2M1 ||
      !epilogue || epilogue.getBias() || epilogue.getActivation() != "none" ||
      epilogue.getOutputType() != "f32" || kernel.getInputs().size() != 8) {
    op->emitError("sm_120 NVFP4 matmul_kernel requires packed A/B, scale A/B, "
                  "D, M/N/K, f32 output, and no fused epilogue");
    return failure();
  }
  ValueRange inputs = kernel.getInputs();
  Value aBase = inputs[0], bBase = inputs[1];
  Value scaleABase = inputs[2], scaleBBase = inputs[3];
  Value dBase = inputs[4];
  Value m = inputs[5], n = inputs[6], k = inputs[7];
  Location loc = op->getLoc();
  Type i8 = builder.getI8Type();
  Type i32 = builder.getI32Type();
  auto f32 = builder.getF32Type();
  Value zero64 = i64Constant(builder, loc, 0);
  Value one64 = i64Constant(builder, loc, 1);
  Value two64 = i64Constant(builder, loc, 2);
  Value four64 = i64Constant(builder, loc, 4);
  Value sixteen64 = i64Constant(builder, loc, 16);
  Value thirtyTwo64 = i64Constant(builder, loc, 32);
  Value sixtyFour64 = i64Constant(builder, loc, 64);
  Value zero32 = arith::ConstantIntOp::create(builder, loc, 0, 32);

  Value blockX32 = NVVM::BlockIdXOp::create(builder, loc, i32);
  Value blockY32 = NVVM::BlockIdYOp::create(builder, loc, i32);
  Value blockX = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), blockX32);
  Value blockY = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), blockY32);
  Value mt = mulI64(builder, loc, blockY, sixteen64);
  Value nt = mulI64(builder, loc, blockX, i64Constant(builder, loc, 8));
  Value tid = NVVM::ThreadIdXOp::create(builder, loc, i32);
  Value lane = arith::AndIOp::create(
      builder, loc, tid, arith::ConstantIntOp::create(builder, loc, 31, 32));
  Value gid32 = arith::ShRUIOp::create(
      builder, loc, lane, arith::ConstantIntOp::create(builder, loc, 2, 32));
  Value tig32 = arith::AndIOp::create(
      builder, loc, lane, arith::ConstantIntOp::create(builder, loc, 3, 32));
  Value gid = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), gid32);
  Value tig = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), tig32);
  Value eightTig = mulI64(builder, loc, tig, i64Constant(builder, loc, 8));
  Value packedK = arith::DivUIOp::create(
      builder, loc, addI64(builder, loc, k, one64), two64);
  Value scaleK = arith::DivUIOp::create(
      builder, loc, addI64(builder, loc, k, i64Constant(builder, loc, 15)),
      sixteen64);

  auto packCodes = [&](Value base, Value row, Value col, bool operandA) {
    Value word = zero32;
    for (int64_t j = 0; j < 8; ++j) {
      Value logicalRow = operandA ? row : addI64(
          builder, loc, row, i64Constant(builder, loc, j));
      Value logicalCol = operandA ? addI64(
          builder, loc, col, i64Constant(builder, loc, j)) : col;
      Value valid = arith::AndIOp::create(
          builder, loc, lessI64(builder, loc, logicalRow, operandA ? m : k),
          lessI64(builder, loc, logicalCol, operandA ? k : n));
      Value contraction = operandA ? logicalCol : logicalRow;
      Value packedIndex = arith::DivUIOp::create(
          builder, loc, contraction, two64);
      Value linear = operandA
          ? addI64(builder, loc, mulI64(builder, loc, logicalRow, packedK),
                   packedIndex)
          : addI64(builder, loc, mulI64(builder, loc, packedIndex, n),
                   logicalCol);
      Value byte = maskedLoadScalar(builder, loc, base, i8, linear, valid, 1);
      Value extended = arith::ExtUIOp::create(builder, loc, i32, byte);
      Value parity64 = arith::RemUIOp::create(
          builder, loc, contraction, two64);
      Value shift64 = mulI64(builder, loc, parity64, four64);
      Value shift = arith::TruncIOp::create(builder, loc, i32, shift64);
      Value code = arith::ShRUIOp::create(builder, loc, extended, shift);
      code = arith::AndIOp::create(
          builder, loc, code, arith::ConstantIntOp::create(builder, loc, 15, 32));
      if (j != 0)
        code = arith::ShLIOp::create(
            builder, loc, code,
            arith::ConstantIntOp::create(builder, loc, 4 * j, 32));
      word = arith::OrIOp::create(builder, loc, word, code);
    }
    return word;
  };

  auto packScales = [&](Value base, Value row, Value col, Value kOrigin,
                        Value laneValid, bool operandA) {
    Value word = zero32;
    Value kBlockOrigin = arith::DivUIOp::create(
        builder, loc, kOrigin, sixteen64);
    for (int64_t j = 0; j < 4; ++j) {
      Value kBlock = addI64(
          builder, loc, kBlockOrigin, i64Constant(builder, loc, j));
      Value bound = operandA ? m : n;
      Value outer = operandA ? row : col;
      Value valid = arith::AndIOp::create(
          builder, loc, laneValid,
          arith::AndIOp::create(
              builder, loc, lessI64(builder, loc, outer, bound),
              lessI64(builder, loc, kBlock, scaleK)));
      Value linear = operandA
          ? addI64(builder, loc, mulI64(builder, loc, row, scaleK), kBlock)
          : addI64(builder, loc, mulI64(builder, loc, kBlock, n), col);
      Value byte = maskedLoadScalar(builder, loc, base, i8, linear, valid, 1);
      Value extended = arith::ExtUIOp::create(builder, loc, i32, byte);
      if (j != 0)
        extended = arith::ShLIOp::create(
            builder, loc, extended,
            arith::ConstantIntOp::create(builder, loc, 8 * j, 32));
      word = arith::OrIOp::create(builder, loc, word, extended);
    }
    return word;
  };

  Value zeroF32 = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
  SmallVector<Value> init(4, zeroF32);
  auto loop = scf::ForOp::create(builder, loc, zero64, k, sixtyFour64, init);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value kOrigin = loop.getInductionVar();
    Value row0 = addI64(builder, loc, mt, gid);
    Value row1 = addI64(builder, loc, row0, i64Constant(builder, loc, 8));
    Value aCol0 = addI64(builder, loc, kOrigin, eightTig);
    Value aCol1 = addI64(builder, loc, aCol0, thirtyTwo64);
    Value bRow0 = addI64(builder, loc, kOrigin, eightTig);
    Value bRow1 = addI64(builder, loc, bRow0, thirtyTwo64);
    Value bCol = addI64(builder, loc, nt, gid);
    SmallVector<Value> operands = {
        packCodes(aBase, row0, aCol0, true),
        packCodes(aBase, row1, aCol0, true),
        packCodes(aBase, row0, aCol1, true),
        packCodes(aBase, row1, aCol1, true),
        packCodes(bBase, bRow0, bCol, false),
        packCodes(bBase, bRow1, bCol, false),
    };
    operands.append(loop.getRegionIterArgs().begin(),
                    loop.getRegionIterArgs().end());
    Value isTig0 = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, tig32, zero32);
    Value isTig1 = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, tig32,
        arith::ConstantIntOp::create(builder, loc, 1, 32));
    Value scaleARow = arith::SelectOp::create(builder, loc, isTig1, row1, row0);
    Value scaleALane = arith::OrIOp::create(builder, loc, isTig0, isTig1);
    operands.push_back(packScales(
        scaleABase, scaleARow, zero64, kOrigin, scaleALane, true));
    operands.push_back(packScales(
        scaleBBase, zero64, bCol, kOrigin, isTig0, false));
    Type resultTy = LLVM::LLVMStructType::getLiteral(
        builder.getContext(), {f32, f32, f32, f32});
    SmallVector<NamedAttribute> attrs = {
        builder.getNamedAttr("arch", builder.getStringAttr("sm_120a")),
        builder.getNamedAttr("shape", builder.getStringAttr("m16n8k64")),
        builder.getNamedAttr("dtype_ab", builder.getStringAttr("nvfp4")),
        builder.getNamedAttr("dtype_c", builder.getStringAttr("f32")),
        builder.getNamedAttr("block_scaled", builder.getBoolAttr(true))};
    Operation *mma = createContractOp(
        builder, loc, "tessera_nvidia.nvfp4_block_scale_mma", operands,
        TypeRange{resultTy}, attrs);
    SmallVector<Value> next;
    for (int64_t index = 0; index < 4; ++index)
      next.push_back(LLVM::ExtractValueOp::create(
          builder, loc, f32, mma->getResult(0), ArrayRef<int64_t>{index}));
    scf::YieldOp::create(builder, loc, next);
  }

  builder.setInsertionPointAfter(loop);
  Value outCol = addI64(
      builder, loc, nt,
      arith::ExtUIOp::create(
          builder, loc, builder.getI64Type(),
          arith::MulIOp::create(
              builder, loc, tig32,
              arith::ConstantIntOp::create(builder, loc, 2, 32))));
  Value colValid0 = lessI64(builder, loc, outCol, n);
  Value colValid1 = lessI64(builder, loc, addI64(builder, loc, outCol, one64), n);
  for (unsigned pair = 0; pair < 2; ++pair) {
    Value row = addI64(
        builder, loc, mt,
        pair == 0 ? gid : addI64(builder, loc, gid, i64Constant(builder, loc, 8)));
    Value rowValid = lessI64(builder, loc, row, m);
    Value valid0 = arith::AndIOp::create(builder, loc, rowValid, colValid0);
    Value valid1 = arith::AndIOp::create(builder, loc, rowValid, colValid1);
    Value linear = addI64(builder, loc, mulI64(builder, loc, row, n), outCol);
    Value ptr = LLVM::GEPOp::create(
        builder, loc, dBase.getType(), f32, dBase, ValueRange{linear});
    Value outPair = pairFromScalars(
        builder, loc, f32, loop.getResult(pair * 2),
        loop.getResult(pair * 2 + 1));
    LLVM::MaskedStoreOp::create(
        builder, loc, outPair, ptr, mask2(builder, loc, valid0, valid1), 4);
  }
  op->erase();
  return success();
}

// General-shape OCP MX materialization. FP6 values occupy one byte each (the
// low six bits carry E2M3/E3M2); MXFP4 values are nibble packed. Both use one
// UE8M0 scale per logical 32-value block, unlike NVFP4's UE4M3/16 contract.
static LogicalResult materializeSm120MxMatmulKernel(
    tessera::tile::MatmulKernelOp kernel, OpBuilder &builder,
    const Sm120FragmentDescriptor &physical) {
  Operation *op = kernel.getOperation();
  auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
  auto epilogue = op->getAttrOfType<tessera::tile::TileEpilogueAttr>("epilogue");
  bool isFP6 = desc && (desc.getAType() == "e2m3" || desc.getAType() == "e3m2");
  bool isMXFP4 = desc && desc.getAType() == "fp4_e2m1";
  if ((!isFP6 && !isMXFP4) || !epilogue || epilogue.getBias() ||
      epilogue.getActivation() != "none" || epilogue.getOutputType() != "f32" ||
      kernel.getInputs().size() != 8) {
    op->emitError("sm_120 FP6/MXFP4 matmul_kernel requires packed A/B, "
                  "UE8M0 scale A/B, D, M/N/K, and f32 output");
    return failure();
  }
  if (auto attr = op->getAttrOfType<StringAttr>("staging");
      !attr || attr.getValue() != "global") {
    op->emitError("sm_120 FP6/MXFP4 materializer requires direct global staging");
    return failure();
  }
  ValueRange inputs = kernel.getInputs();
  Value aBase = inputs[0], bBase = inputs[1];
  Value scaleABase = inputs[2], scaleBBase = inputs[3], dBase = inputs[4];
  Value m = inputs[5], n = inputs[6], k = inputs[7];
  Location loc = op->getLoc();
  Type i8 = builder.getI8Type();
  Type i32 = builder.getI32Type();
  auto f32 = builder.getF32Type();
  Value one = i64Constant(builder, loc, 1);
  Value two = i64Constant(builder, loc, 2);
  Value eight = i64Constant(builder, loc, 8);
  Value thirtyTwo = i64Constant(builder, loc, 32);
  Value fragmentK = i64Constant(builder, loc, physical.k);
  Value halfK = i64Constant(builder, loc, physical.k / 2);
  int64_t valuesPerRegister = isFP6 ? 4 : 8;
  int64_t scaleBytesPerFragment = isFP6 ? 1 : 2;

  Value blockX32 = NVVM::BlockIdXOp::create(builder, loc, i32);
  Value blockY32 = NVVM::BlockIdYOp::create(builder, loc, i32);
  Value blockX = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), blockX32);
  Value blockY = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), blockY32);
  Value rowOrigin = mulI64(builder, loc, blockY, i64Constant(builder, loc, 16));
  Value colOrigin = mulI64(builder, loc, blockX, eight);
  Value tid = NVVM::ThreadIdXOp::create(builder, loc, i32);
  Value lane = arith::AndIOp::create(
      builder, loc, tid, arith::ConstantIntOp::create(builder, loc, 31, 32));
  Value gid32 = arith::ShRUIOp::create(
      builder, loc, lane, arith::ConstantIntOp::create(builder, loc, 2, 32));
  Value tig32 = arith::AndIOp::create(
      builder, loc, lane, arith::ConstantIntOp::create(builder, loc, 3, 32));
  Value gid = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), gid32);
  Value tig = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), tig32);
  Value registerOffset = mulI64(
      builder, loc, tig, i64Constant(builder, loc, valuesPerRegister));
  Value packedK = arith::DivUIOp::create(
      builder, loc, addI64(builder, loc, k, one), two);
  Value scaleK = arith::DivUIOp::create(
      builder, loc, addI64(builder, loc, k, i64Constant(builder, loc, 31)),
      thirtyTwo);

  auto packData = [&](Value base, Value row, Value col, bool operandA) {
    Value word = arith::ConstantIntOp::create(builder, loc, 0, 32);
    for (int64_t index = 0; index < valuesPerRegister; ++index) {
      Value logicalRow = operandA ? row : addI64(
          builder, loc, row, i64Constant(builder, loc, index));
      Value logicalCol = operandA ? addI64(
          builder, loc, col, i64Constant(builder, loc, index)) : col;
      Value valid = arith::AndIOp::create(
          builder, loc, lessI64(builder, loc, logicalRow, operandA ? m : k),
          lessI64(builder, loc, logicalCol, operandA ? k : n));
      Value contraction = operandA ? logicalCol : logicalRow;
      Value physicalContraction = isFP6
          ? contraction
          : Value(arith::DivUIOp::create(builder, loc, contraction, two));
      Value leading = isFP6 ? k : packedK;
      Value linear = operandA
          ? addI64(builder, loc, mulI64(builder, loc, logicalRow, leading),
                   physicalContraction)
          : addI64(builder, loc, mulI64(builder, loc, physicalContraction, n),
                   logicalCol);
      Value byte = maskedLoadScalar(builder, loc, base, i8, linear, valid, 1);
      Value code = arith::ExtUIOp::create(builder, loc, i32, byte);
      if (!isFP6) {
        Value parity = arith::RemUIOp::create(builder, loc, contraction, two);
        Value shift64 = mulI64(builder, loc, parity, i64Constant(builder, loc, 4));
        Value shift = arith::TruncIOp::create(builder, loc, i32, shift64);
        code = arith::ShRUIOp::create(builder, loc, code, shift);
        code = arith::AndIOp::create(
            builder, loc, code, arith::ConstantIntOp::create(builder, loc, 15, 32));
      }
      int64_t bits = isFP6 ? 8 : 4;
      if (index != 0)
        code = arith::ShLIOp::create(
            builder, loc, code,
            arith::ConstantIntOp::create(builder, loc, bits * index, 32));
      word = arith::OrIOp::create(builder, loc, word, code);
    }
    return word;
  };

  auto packScale = [&](Value base, Value outer, Value kOrigin,
                       Value laneValid, bool operandA) {
    Value word = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value blockOrigin = arith::DivUIOp::create(builder, loc, kOrigin, thirtyTwo);
    for (int64_t index = 0; index < scaleBytesPerFragment; ++index) {
      Value block = addI64(
          builder, loc, blockOrigin, i64Constant(builder, loc, index));
      Value valid = arith::AndIOp::create(
          builder, loc, laneValid,
          arith::AndIOp::create(
              builder, loc, lessI64(builder, loc, outer, operandA ? m : n),
              lessI64(builder, loc, block, scaleK)));
      Value linear = operandA
          ? addI64(builder, loc, mulI64(builder, loc, outer, scaleK), block)
          : addI64(builder, loc, mulI64(builder, loc, block, n), outer);
      Value byte = maskedLoadScalar(builder, loc, base, i8, linear, valid, 1);
      Value widened = arith::ExtUIOp::create(builder, loc, i32, byte);
      if (index != 0)
        widened = arith::ShLIOp::create(
            builder, loc, widened,
            arith::ConstantIntOp::create(builder, loc, 8 * index, 32));
      word = arith::OrIOp::create(builder, loc, word, widened);
    }
    return word;
  };

  Value zeroF32 = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
  auto loop = scf::ForOp::create(
      builder, loc, i64Constant(builder, loc, 0), k, fragmentK,
      ValueRange{zeroF32, zeroF32, zeroF32, zeroF32});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value kOrigin = loop.getInductionVar();
    Value row0 = addI64(builder, loc, rowOrigin, gid);
    Value row1 = addI64(builder, loc, row0, eight);
    Value k0 = addI64(builder, loc, kOrigin, registerOffset);
    Value k1 = addI64(builder, loc, k0, halfK);
    Value bCol = addI64(builder, loc, colOrigin, gid);
    SmallVector<Value> operands = {
        packData(aBase, row0, k0, true), packData(aBase, row1, k0, true),
        packData(aBase, row0, k1, true), packData(aBase, row1, k1, true),
        packData(bBase, k0, bCol, false), packData(bBase, k1, bCol, false)};
    operands.append(loop.getRegionIterArgs().begin(),
                    loop.getRegionIterArgs().end());
    Value isTig0 = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, tig32,
        arith::ConstantIntOp::create(builder, loc, 0, 32));
    Value isTig1 = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, tig32,
        arith::ConstantIntOp::create(builder, loc, 1, 32));
    Value scaleARow = arith::SelectOp::create(builder, loc, isTig1, row1, row0);
    operands.push_back(packScale(
        scaleABase, scaleARow, kOrigin,
        arith::OrIOp::create(builder, loc, isTig0, isTig1), true));
    operands.push_back(packScale(scaleBBase, bCol, kOrigin, isTig0, false));
    Type resultTy = LLVM::LLVMStructType::getLiteral(
        builder.getContext(), {f32, f32, f32, f32});
    SmallVector<NamedAttribute> attrs = {
        builder.getNamedAttr("arch", builder.getStringAttr("sm_120a")),
        builder.getNamedAttr("shape", builder.getStringAttr(
            isFP6 ? "m16n8k32" : "m16n8k64")),
        builder.getNamedAttr("dtype_ab", builder.getStringAttr(
            isFP6 ? desc.getAType() : StringRef("e2m1"))),
        builder.getNamedAttr("dtype_c", builder.getStringAttr("f32")),
        builder.getNamedAttr("scale_dtype", builder.getStringAttr("ue8m0")),
        builder.getNamedAttr("scale_vector", builder.getStringAttr(
            isFP6 ? "1X" : "2X")),
        builder.getNamedAttr("block_scaled", builder.getBoolAttr(true))};
    Operation *mma = createContractOp(
        builder, loc, "tessera_nvidia.mx_block_scale_mma", operands,
        TypeRange{resultTy}, attrs);
    SmallVector<Value> next;
    for (int64_t index = 0; index < 4; ++index)
      next.push_back(LLVM::ExtractValueOp::create(
          builder, loc, f32, mma->getResult(0), ArrayRef<int64_t>{index}));
    scf::YieldOp::create(builder, loc, next);
  }

  builder.setInsertionPointAfter(loop);
  Value outCol = addI64(
      builder, loc, colOrigin,
      mulI64(builder, loc, tig, i64Constant(builder, loc, 2)));
  Value colValid0 = lessI64(builder, loc, outCol, n);
  Value colValid1 = lessI64(builder, loc, addI64(builder, loc, outCol, one), n);
  for (unsigned pair = 0; pair < 2; ++pair) {
    Value row = addI64(
        builder, loc, rowOrigin,
        pair == 0 ? gid : addI64(builder, loc, gid, eight));
    Value rowValid = lessI64(builder, loc, row, m);
    Value valid0 = arith::AndIOp::create(builder, loc, rowValid, colValid0);
    Value valid1 = arith::AndIOp::create(builder, loc, rowValid, colValid1);
    Value linear = addI64(builder, loc, mulI64(builder, loc, row, n), outCol);
    Value ptr = LLVM::GEPOp::create(
        builder, loc, dBase.getType(), f32, dBase, ValueRange{linear});
    Value outPair = pairFromScalars(
        builder, loc, f32, loop.getResult(pair * 2),
        loop.getResult(pair * 2 + 1));
    LLVM::MaskedStoreOp::create(
        builder, loc, outPair, ptr, mask2(builder, loc, valid0, valid1), 4);
  }
  op->erase();
  return success();
}

// General-shape launch materialization for the f32-accumulating scalar-f32
// (TF32 math mode) and packed-four-byte (FP8) MMA families.  Storage remains
// explicit: TF32 consumes ordinary f32 memory while FP8 consumes one byte per
// logical element.  One warp owns each m16n8 output tile; ragged reads zero-fill
// before register packing and stores are masked.
static LogicalResult materializeSm120F32PackedMatmulKernel(
    tessera::tile::MatmulKernelOp kernel, OpBuilder &builder,
    const Sm120FragmentDescriptor &physical) {
  Operation *op = kernel.getOperation();
  auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
  auto epilogue = op->getAttrOfType<tessera::tile::TileEpilogueAttr>("epilogue");
  bool isInt8 = physical.accumulator == "s32";
  bool hasBias = epilogue && epilogue.getBias();
  bool hasResidual = false;
  if (auto attr = op->getAttrOfType<BoolAttr>("residual"))
    hasResidual = attr.getValue();
  StringRef expectedOutput = isInt8 ? "i32" : "f32";
  unsigned expectedInputs = 6 + unsigned(hasBias) + unsigned(hasResidual);
  if (!desc || !epilogue ||
      epilogue.getOutputType() != expectedOutput ||
      kernel.getInputs().size() != expectedInputs ||
      (isInt8 && (hasBias || hasResidual || epilogue.getActivation() != "none"))) {
    op->emitError("sm_120 TF32/FP8/INT8 matmul_kernel has an invalid fused "
                  "epilogue ABI (INT8 requires the unfused i32 route)");
    return failure();
  }
  if (auto attr = op->getAttrOfType<StringAttr>("staging");
      !attr || attr.getValue() != "global") {
    op->emitError("sm_120 TF32/FP8 canonical materializer currently requires "
                  "direct global staging");
    return failure();
  }
  if (auto attr = op->getAttrOfType<IntegerAttr>("warps");
      !attr || attr.getInt() != 1) {
    op->emitError("sm_120 TF32/FP8 canonical materializer requires one warp");
    return failure();
  }

  ValueRange inputs = kernel.getInputs();
  Value aBase = inputs[0], bBase = inputs[1];
  Value biasBase = hasBias ? inputs[2] : Value();
  unsigned residualIndex = 2 + unsigned(hasBias);
  Value residualBase = hasResidual ? inputs[residualIndex] : Value();
  unsigned dIndex = 2 + unsigned(hasBias) + unsigned(hasResidual);
  Value dBase = inputs[dIndex];
  Value m = inputs[dIndex + 1], n = inputs[dIndex + 2], k = inputs[dIndex + 3];
  Location loc = op->getLoc();
  Type i8 = builder.getI8Type();
  Type i32 = builder.getI32Type();
  auto f32 = builder.getF32Type();
  bool isTF32 = physical.packing == Sm120InputPacking::ScalarF32;
  int64_t packWidth = isTF32 ? 1 : 4;
  Value one = i64Constant(builder, loc, 1);
  Value eight = i64Constant(builder, loc, 8);
  Value fragmentK = i64Constant(builder, loc, physical.k);
  Value halfK = i64Constant(builder, loc, physical.k / 2);

  Value blockX32 = NVVM::BlockIdXOp::create(builder, loc, i32);
  Value blockY32 = NVVM::BlockIdYOp::create(builder, loc, i32);
  Value blockX = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), blockX32);
  Value blockY = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), blockY32);
  Value rowOrigin = mulI64(builder, loc, blockY, i64Constant(builder, loc, 16));
  Value colOrigin = mulI64(builder, loc, blockX, eight);
  Value tid = NVVM::ThreadIdXOp::create(builder, loc, i32);
  Value lane = arith::AndIOp::create(
      builder, loc, tid, arith::ConstantIntOp::create(builder, loc, 31, 32));
  Value gid32 = arith::ShRUIOp::create(
      builder, loc, lane, arith::ConstantIntOp::create(builder, loc, 2, 32));
  Value tig32 = arith::AndIOp::create(
      builder, loc, lane, arith::ConstantIntOp::create(builder, loc, 3, 32));
  Value gid = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), gid32);
  Value tig = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), tig32);
  Value registerColumn = mulI64(
      builder, loc, tig, i64Constant(builder, loc, packWidth));

  auto loadRegister = [&](Value base, Value row, Value col, bool operandA) {
    if (isTF32) {
      Value valid = arith::AndIOp::create(
          builder, loc, lessI64(builder, loc, row, operandA ? m : k),
          lessI64(builder, loc, col, operandA ? k : n));
      Value linear = operandA
          ? addI64(builder, loc, mulI64(builder, loc, row, k), col)
          : addI64(builder, loc, mulI64(builder, loc, col, k), row);
      Value scalar = maskedLoadScalar(builder, loc, base, f32, linear, valid, 4);
      return Value(LLVM::BitcastOp::create(builder, loc, i32, scalar));
    }
    Value word = arith::ConstantIntOp::create(builder, loc, 0, 32);
    for (int64_t index = 0; index < 4; ++index) {
      Value logicalRow = operandA ? row : addI64(
          builder, loc, row, i64Constant(builder, loc, index));
      Value logicalCol = operandA ? addI64(
          builder, loc, col, i64Constant(builder, loc, index)) : col;
      Value valid = arith::AndIOp::create(
          builder, loc, lessI64(builder, loc, logicalRow, operandA ? m : k),
          lessI64(builder, loc, logicalCol, operandA ? k : n));
      Value linear = operandA
          ? addI64(builder, loc, mulI64(builder, loc, logicalRow, k), logicalCol)
          : addI64(builder, loc, mulI64(builder, loc, logicalCol, k), logicalRow);
      Value byte = maskedLoadScalar(builder, loc, base, i8, linear, valid, 1);
      Value widened = arith::ExtUIOp::create(builder, loc, i32, byte);
      if (index != 0)
        widened = arith::ShLIOp::create(
            builder, loc, widened,
            arith::ConstantIntOp::create(builder, loc, index * 8, 32));
      word = arith::OrIOp::create(builder, loc, word, widened);
    }
    return word;
  };

  Type accumulatorType = isInt8 ? Type(i32) : Type(f32);
  Value zeroAccumulator = isInt8
      ? Value(arith::ConstantIntOp::create(builder, loc, 0, 32))
      : Value(arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f)));
  SmallVector<Value> init(4, zeroAccumulator);
  auto loop = scf::ForOp::create(
      builder, loc, i64Constant(builder, loc, 0), k, fragmentK, init);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value kOrigin = loop.getInductionVar();
    Value row0 = addI64(builder, loc, rowOrigin, gid);
    Value row1 = addI64(builder, loc, row0, eight);
    Value k0 = addI64(builder, loc, kOrigin, registerColumn);
    Value k1 = addI64(builder, loc, k0, halfK);
    Value bCol = addI64(builder, loc, colOrigin, gid);
    SmallVector<Value> operands = {
        loadRegister(aBase, row0, k0, true),
        loadRegister(aBase, row1, k0, true),
        loadRegister(aBase, row0, k1, true),
        loadRegister(aBase, row1, k1, true),
        loadRegister(bBase, k0, bCol, false),
        loadRegister(bBase, k1, bCol, false),
    };
    operands.append(loop.getRegionIterArgs().begin(),
                    loop.getRegionIterArgs().end());
    Type resultTy = LLVM::LLVMStructType::getLiteral(
        builder.getContext(), {accumulatorType, accumulatorType,
                               accumulatorType, accumulatorType});
    SmallVector<NamedAttribute> attrs = {
        builder.getNamedAttr("arch", builder.getStringAttr("sm_120")),
        builder.getNamedAttr(
            "shape", builder.getStringAttr(
                isTF32 ? "m16n8k8" : "m16n8k32")),
        builder.getNamedAttr("dtype_ab", builder.getStringAttr(desc.getAType())),
        builder.getNamedAttr(
            "dtype_c", builder.getStringAttr(isInt8 ? "s32" : "f32")),
        builder.getNamedAttr("block_scaled", builder.getBoolAttr(false))};
    Operation *mma = createContractOp(
        builder, loc, "tessera_nvidia.mma_sync", operands,
        TypeRange{resultTy}, attrs);
    SmallVector<Value> next;
    for (int64_t index = 0; index < 4; ++index)
      next.push_back(LLVM::ExtractValueOp::create(
          builder, loc, accumulatorType, mma->getResult(0),
          ArrayRef<int64_t>{index}));
    scf::YieldOp::create(builder, loc, next);
  }

  builder.setInsertionPointAfter(loop);
  Value outCol = addI64(
      builder, loc, colOrigin,
      mulI64(builder, loc, tig, i64Constant(builder, loc, 2)));
  Value colValid0 = lessI64(builder, loc, outCol, n);
  Value colValid1 = lessI64(builder, loc, addI64(builder, loc, outCol, one), n);
  for (unsigned pair = 0; pair < 2; ++pair) {
    Value row = addI64(
        builder, loc, rowOrigin,
        pair == 0 ? gid : addI64(builder, loc, gid, eight));
    Value rowValid = lessI64(builder, loc, row, m);
    Value valid0 = arith::AndIOp::create(builder, loc, rowValid, colValid0);
    Value valid1 = arith::AndIOp::create(builder, loc, rowValid, colValid1);
    Value linear = addI64(builder, loc, mulI64(builder, loc, row, n), outCol);
    Value value0 = loop.getResult(pair * 2);
    Value value1 = loop.getResult(pair * 2 + 1);
    if (hasBias) {
      Value biasPair = maskedLoadPair(
          builder, loc, biasBase, f32, outCol, colValid0, colValid1, 4);
      value0 = arith::AddFOp::create(
          builder, loc, value0, LLVM::ExtractElementOp::create(
              builder, loc, f32, biasPair, i64Constant(builder, loc, 0)));
      value1 = arith::AddFOp::create(
          builder, loc, value1, LLVM::ExtractElementOp::create(
              builder, loc, f32, biasPair, i64Constant(builder, loc, 1)));
    }
    if (epilogue.getActivation() != "none") {
      value0 = tessera::tile::emitScalarFloatActivation(
          builder, loc, value0, epilogue.getActivation());
      value1 = tessera::tile::emitScalarFloatActivation(
          builder, loc, value1, epilogue.getActivation());
    }
    if (hasResidual) {
      Value residualPair = maskedLoadPair(
          builder, loc, residualBase, f32, linear, valid0, valid1, 4);
      value0 = arith::AddFOp::create(
          builder, loc, value0, LLVM::ExtractElementOp::create(
              builder, loc, f32, residualPair, i64Constant(builder, loc, 0)));
      value1 = arith::AddFOp::create(
          builder, loc, value1, LLVM::ExtractElementOp::create(
              builder, loc, f32, residualPair, i64Constant(builder, loc, 1)));
    }
    Value ptr = LLVM::GEPOp::create(
        builder, loc, dBase.getType(), accumulatorType, dBase,
        ValueRange{linear});
    Value outPair = pairFromScalars(
        builder, loc, accumulatorType, value0, value1);
    LLVM::MaskedStoreOp::create(
        builder, loc, outPair, ptr, mask2(builder, loc, valid0, valid1), 4);
  }
  op->erase();
  return success();
}

// General-shape FP64 DMMA materialization.  For m8n8k4 each lane owns A(gid,
// tig), B(tig,gid), and two adjacent f64 outputs at (gid,2*tig+[0,1]).
static LogicalResult materializeSm120F64MatmulKernel(
    tessera::tile::MatmulKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
  auto epilogue = op->getAttrOfType<tessera::tile::TileEpilogueAttr>("epilogue");
  if (!desc || !epilogue || epilogue.getBias() ||
      epilogue.getActivation() != "none" ||
      epilogue.getOutputType() != "f64" || kernel.getInputs().size() != 6) {
    op->emitError("sm_120 FP64 matmul_kernel requires A/B/D/M/N/K, f64 "
                  "output, and no fused epilogue");
    return failure();
  }
  if (auto attr = op->getAttrOfType<StringAttr>("staging");
      !attr || attr.getValue() != "global") {
    op->emitError("sm_120 FP64 canonical materializer requires direct global staging");
    return failure();
  }
  ValueRange inputs = kernel.getInputs();
  Value aBase = inputs[0], bBase = inputs[1], dBase = inputs[2];
  Value m = inputs[3], n = inputs[4], k = inputs[5];
  Location loc = op->getLoc();
  Type i32 = builder.getI32Type();
  auto f64 = builder.getF64Type();
  Value one = i64Constant(builder, loc, 1);
  Value eight = i64Constant(builder, loc, 8);
  Value blockX32 = NVVM::BlockIdXOp::create(builder, loc, i32);
  Value blockY32 = NVVM::BlockIdYOp::create(builder, loc, i32);
  Value blockX = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), blockX32);
  Value blockY = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), blockY32);
  Value rowOrigin = mulI64(builder, loc, blockY, eight);
  Value colOrigin = mulI64(builder, loc, blockX, eight);
  Value tid = NVVM::ThreadIdXOp::create(builder, loc, i32);
  Value lane = arith::AndIOp::create(
      builder, loc, tid, arith::ConstantIntOp::create(builder, loc, 31, 32));
  Value gid32 = arith::ShRUIOp::create(
      builder, loc, lane, arith::ConstantIntOp::create(builder, loc, 2, 32));
  Value tig32 = arith::AndIOp::create(
      builder, loc, lane, arith::ConstantIntOp::create(builder, loc, 3, 32));
  Value gid = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), gid32);
  Value tig = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), tig32);
  Value zero = arith::ConstantFloatOp::create(
      builder, loc, f64, APFloat(0.0));
  auto loop = scf::ForOp::create(
      builder, loc, i64Constant(builder, loc, 0), k,
      i64Constant(builder, loc, 4), ValueRange{zero, zero});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value kOrigin = loop.getInductionVar();
    Value row = addI64(builder, loc, rowOrigin, gid);
    Value contraction = addI64(builder, loc, kOrigin, tig);
    Value col = addI64(builder, loc, colOrigin, gid);
    Value aValid = arith::AndIOp::create(
        builder, loc, lessI64(builder, loc, row, m),
        lessI64(builder, loc, contraction, k));
    Value bValid = arith::AndIOp::create(
        builder, loc, lessI64(builder, loc, contraction, k),
        lessI64(builder, loc, col, n));
    Value aLinear = addI64(
        builder, loc, mulI64(builder, loc, row, k), contraction);
    Value bLinear = addI64(
        builder, loc, mulI64(builder, loc, col, k), contraction);
    Value a = maskedLoadScalar(builder, loc, aBase, f64, aLinear, aValid, 8);
    Value b = maskedLoadScalar(builder, loc, bBase, f64, bLinear, bValid, 8);
    SmallVector<Value> operands = {a, b, loop.getRegionIterArgs()[0],
                                   loop.getRegionIterArgs()[1]};
    Type resultTy = LLVM::LLVMStructType::getLiteral(
        builder.getContext(), {f64, f64});
    SmallVector<NamedAttribute> attrs = {
        builder.getNamedAttr("arch", builder.getStringAttr("sm_120")),
        builder.getNamedAttr("shape", builder.getStringAttr("m8n8k4")),
        builder.getNamedAttr("dtype_ab", builder.getStringAttr("f64")),
        builder.getNamedAttr("dtype_c", builder.getStringAttr("f64")),
        builder.getNamedAttr("block_scaled", builder.getBoolAttr(false))};
    Operation *mma = createContractOp(
        builder, loc, "tessera_nvidia.mma_sync", operands,
        TypeRange{resultTy}, attrs);
    scf::YieldOp::create(builder, loc, ValueRange{
        LLVM::ExtractValueOp::create(
            builder, loc, f64, mma->getResult(0), ArrayRef<int64_t>{0}),
        LLVM::ExtractValueOp::create(
            builder, loc, f64, mma->getResult(0), ArrayRef<int64_t>{1})});
  }
  builder.setInsertionPointAfter(loop);
  Value row = addI64(builder, loc, rowOrigin, gid);
  Value outCol = addI64(
      builder, loc, colOrigin,
      mulI64(builder, loc, tig, i64Constant(builder, loc, 2)));
  Value rowValid = lessI64(builder, loc, row, m);
  Value valid0 = arith::AndIOp::create(
      builder, loc, rowValid, lessI64(builder, loc, outCol, n));
  Value valid1 = arith::AndIOp::create(
      builder, loc, rowValid,
      lessI64(builder, loc, addI64(builder, loc, outCol, one), n));
  Value linear = addI64(builder, loc, mulI64(builder, loc, row, n), outCol);
  Value ptr = LLVM::GEPOp::create(
      builder, loc, dBase.getType(), f64, dBase, ValueRange{linear});
  Value outPair = pairFromScalars(
      builder, loc, f64, loop.getResult(0), loop.getResult(1));
  LLVM::MaskedStoreOp::create(
      builder, loc, outPair, ptr, mask2(builder, loc, valid0, valid1), 8);
  op->erase();
  return success();
}

static LogicalResult materializeSm120MatmulKernel(
    tessera::tile::MatmulKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  auto desc = op->getAttrOfType<tessera::tile::TileMmaDescAttr>("mma");
  auto epilogue = op->getAttrOfType<tessera::tile::TileEpilogueAttr>("epilogue");
  std::optional<Sm120FragmentDescriptor> physical = selectSm120Fragment(desc);
  if (physical && desc.getAType() == "nvfp4")
    return materializeSm120Nvfp4MatmulKernel(kernel, builder);
  if (physical && (desc.getAType() == "e2m3" || desc.getAType() == "e3m2" ||
                   desc.getAType() == "fp4_e2m1"))
    return materializeSm120MxMatmulKernel(kernel, builder, *physical);
  if (physical && physical->packing == Sm120InputPacking::ScalarF64)
    return materializeSm120F64MatmulKernel(kernel, builder);
  if (physical && (physical->packing == Sm120InputPacking::ScalarF32 ||
                   physical->packing == Sm120InputPacking::PackedX4I8) &&
      (physical->accumulator == "f32" || physical->accumulator == "s32"))
    return materializeSm120F32PackedMatmulKernel(kernel, builder, *physical);
  if (!physical || physical->packing != Sm120InputPacking::PairF16 ||
      desc.getAccType() != "f32" || !epilogue) {
    op->emitError("sm_120 matmul_kernel requires m16n8k16 f16 or bf16 inputs "
                  "with f32 accumulation");
    return failure();
  }
  ValueRange inputs = kernel.getInputs();
  bool hasBias = epilogue.getBias();
  bool hasResidual = false;
  if (auto attr = op->getAttrOfType<BoolAttr>("residual"))
    hasResidual = attr.getValue();
  Value aBase = inputs[0], bBase = inputs[1];
  Value biasBase = hasBias ? inputs[2] : Value();
  unsigned residualIndex = 2 + unsigned(hasBias);
  Value residualBase = hasResidual ? inputs[residualIndex] : Value();
  unsigned dIndex = 2 + unsigned(hasBias) + unsigned(hasResidual);
  Value dBase = inputs[dIndex];
  Value m = inputs[dIndex + 1], n = inputs[dIndex + 2], k = inputs[dIndex + 3];
  Location loc = op->getLoc();
  Type inputType = desc.getAType() == "bf16"
      ? Type(builder.getBF16Type()) : Type(builder.getF16Type());
  Type f16 = builder.getF16Type();
  auto f32 = builder.getF32Type();
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
    if (hasResidual) {
      for (unsigned pair = 0; pair < 2; ++pair) {
        Value rowOffset = pair == 0 ? gid : addI64(builder, loc, gid, eight);
        Value row = addI64(builder, loc, rowOrigin, rowOffset);
        Value rowValid = lessI64(builder, loc, row, m);
        Value valid0 = arith::AndIOp::create(builder, loc, rowValid, colValid0);
        Value valid1 = arith::AndIOp::create(builder, loc, rowValid, colValid1);
        Value linear = addI64(builder, loc, mulI64(builder, loc, row, n), outCol);
        Value residualPair = maskedLoadPair(
            builder, loc, residualBase, f32, linear, valid0, valid1, 4);
        for (unsigned element = 0; element < 2; ++element) {
          Value residualValue = LLVM::ExtractElementOp::create(
              builder, loc, f32, residualPair,
              i64Constant(builder, loc, element));
          unsigned valueIndex = base + pair * 2 + element;
          values[valueIndex] = arith::AddFOp::create(
              builder, loc, values[valueIndex], residualValue);
        }
      }
    }
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

// NVIDIA-E2E-2 row-softmax pilot. One CUDA thread owns one flattened row and
// performs a stable max-shifted f32 reduction over the last axis. This is a
// correctness-first compiler-native candidate: it deliberately does not claim
// the cooperative block reduction schedule used by the existing CUDA-C lane.
static LogicalResult materializeSm120SoftmaxKernel(
    tessera::tile::SoftmaxKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  ValueRange inputs = kernel.getInputs();
  if (inputs.size() != 4) {
    op->emitError("sm_120 softmax_kernel requires X, O, rows, and columns");
    return failure();
  }
  auto storage = op->getAttrOfType<StringAttr>("storage");
  auto accum = op->getAttrOfType<StringAttr>("accum");
  auto axis = op->getAttrOfType<IntegerAttr>("axis");
  auto expMode = op->getAttrOfType<StringAttr>("exp_mode");
  auto ftz = op->getAttrOfType<BoolAttr>("ftz");
  bool f16Storage = storage && storage.getValue() == "f16";
  if (!storage || (!f16Storage && storage.getValue() != "f32") || !accum ||
      accum.getValue() != "f32" || !axis || axis.getInt() != -1 ||
      !expMode || expMode.getValue() != "approx_exp2" || !ftz ||
      ftz.getValue()) {
    op->emitError("sm_120 softmax_kernel requires f16/f32 storage, f32 accum, "
                  "axis=-1, exp_mode=approx_exp2, and ftz=false");
    return failure();
  }

  Location loc = op->getLoc();
  Value xBase = inputs[0], outBase = inputs[1];
  Value rows = inputs[2], columns = inputs[3];
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  auto f32 = builder.getF32Type();
  Type storageType = f16Storage ? Type(builder.getF16Type()) : Type(f32);
  unsigned storageAlignment = f16Storage ? 2 : 4;
  Value block = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::BlockIdXOp::create(builder, loc, i32));
  Value thread = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::ThreadIdXOp::create(builder, loc, i32));
  Value row = addI64(
      builder, loc, mulI64(builder, loc, block, i64Constant(builder, loc, 128)),
      thread);
  Value active = lessI64(builder, loc, row, rows);
  auto expF32 = [&](Value value) -> Value {
    Value log2e = arith::ConstantFloatOp::create(
        builder, loc, f32, APFloat(1.4426950408889634f));
    Value scaled = arith::MulFOp::create(builder, loc, value, log2e);
    return NVVM::Ex2Op::create(builder, loc, scaled, /*ftz=*/false);
  };
  auto loadF32 = [&](Value linear) -> Value {
    Value ptr = LLVM::GEPOp::create(
        builder, loc, xBase.getType(), storageType, xBase,
        ValueRange{linear});
    Value value = LLVM::LoadOp::create(
        builder, loc, storageType, ptr, storageAlignment);
    return f16Storage
        ? Value(LLVM::FPExtOp::create(builder, loc, f32, value))
        : value;
  };
  auto storeF32 = [&](Value linear, Value value) {
    Value stored = f16Storage
        ? Value(LLVM::FPTruncOp::create(builder, loc, storageType, value))
        : value;
    Value ptr = LLVM::GEPOp::create(
        builder, loc, outBase.getType(), storageType, outBase,
        ValueRange{linear});
    LLVM::StoreOp::create(builder, loc, stored, ptr, storageAlignment);
  };
  auto guarded = scf::IfOp::create(builder, loc, active, /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(guarded.thenBlock());
    Value zero = i64Constant(builder, loc, 0);
    Value one = i64Constant(builder, loc, 1);
    Value rowBase = mulI64(builder, loc, row, columns);
    Value negInf = arith::ConstantFloatOp::create(
        builder, loc, f32,
        APFloat::getInf(APFloat::IEEEsingle(), /*negative=*/true));
    auto maxLoop = scf::ForOp::create(
        builder, loc, zero, columns, one, ValueRange{negInf});
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(maxLoop.getBody());
      Value linear = addI64(
          builder, loc, rowBase, maxLoop.getInductionVar());
      Value value = loadF32(linear);
      Value next = arith::MaximumFOp::create(
          builder, loc, maxLoop.getRegionIterArgs()[0], value);
      scf::YieldOp::create(builder, loc, ValueRange{next});
    }
    builder.setInsertionPointAfter(maxLoop);
    Value maximum = maxLoop.getResult(0);
    Value zeroF32 = arith::ConstantFloatOp::create(
        builder, loc, f32, APFloat(0.0f));
    auto sumLoop = scf::ForOp::create(
        builder, loc, zero, columns, one, ValueRange{zeroF32});
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(sumLoop.getBody());
      Value linear = addI64(
          builder, loc, rowBase, sumLoop.getInductionVar());
      Value value = loadF32(linear);
      Value shifted = arith::SubFOp::create(builder, loc, value, maximum);
      Value exponent = expF32(shifted);
      Value next = arith::AddFOp::create(
          builder, loc, sumLoop.getRegionIterArgs()[0], exponent);
      scf::YieldOp::create(builder, loc, ValueRange{next});
    }
    builder.setInsertionPointAfter(sumLoop);
    Value denominator = sumLoop.getResult(0);
    auto storeLoop = scf::ForOp::create(builder, loc, zero, columns, one);
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(storeLoop.getBody());
      Value linear = addI64(
          builder, loc, rowBase, storeLoop.getInductionVar());
      Value value = loadF32(linear);
      Value shifted = arith::SubFOp::create(builder, loc, value, maximum);
      Value exponent = expF32(shifted);
      Value normalized = arith::DivFOp::create(
          builder, loc, exponent, denominator);
      storeF32(linear, normalized);
    }
  }
  builder.setInsertionPointAfter(guarded);
  op->erase();
  return success();
}

// Canonical arbitrary-axis reduction over contiguous [outer, axis, inner].
// The serial schedule assigns one output to a thread; cooperative_128 assigns
// one output to a CTA and reduces thread partials through shared memory.
static LogicalResult materializeSm120ReduceKernel(
    tessera::tile::ReduceKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  ValueRange inputs = kernel.getInputs();
  auto storage = op->getAttrOfType<StringAttr>("storage");
  auto accum = op->getAttrOfType<StringAttr>("accum");
  auto kind = op->getAttrOfType<StringAttr>("kind");
  auto axis = op->getAttrOfType<IntegerAttr>("axis");
  auto keepdims = op->getAttrOfType<BoolAttr>("keepdims");
  auto schedule = op->getAttrOfType<StringAttr>("schedule");
  auto nanMode = op->getAttrOfType<StringAttr>("nan_mode");
  bool f16Storage = storage && storage.getValue() == "f16";
  if (inputs.size() != 5 || !storage ||
      (!f16Storage && storage.getValue() != "f32") || !accum ||
      accum.getValue() != "f32" || !kind ||
      (kind.getValue() != "sum" && kind.getValue() != "mean" &&
       kind.getValue() != "max") || !axis || axis.getInt() < 0 || !keepdims ||
      !schedule || (schedule.getValue() != "serial" &&
                    schedule.getValue() != "cooperative_128") ||
      !nanMode || nanMode.getValue() != "propagate") {
    op->emitError("sm_120 reduce_kernel requires f16/f32 storage, f32 accum, "
                  "normalized axis, keepdims, serial|cooperative_128 schedule, "
                  "and nan_mode=propagate");
    return failure();
  }
  Location loc = op->getLoc();
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  auto f32 = builder.getF32Type();
  Type storageType = f16Storage ? Type(builder.getF16Type()) : f32;
  unsigned alignment = f16Storage ? 2 : 4;
  Value block = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::BlockIdXOp::create(builder, loc, i32));
  Value thread = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::ThreadIdXOp::create(builder, loc, i32));
  Value outer = inputs[2], axisExtent = inputs[3], inner = inputs[4];
  bool cooperative = schedule.getValue() == "cooperative_128";
  Value output = cooperative ? block : addI64(
      builder, loc, mulI64(builder, loc, block, i64Constant(builder, loc, 128)),
      thread);
  Value outputs = mulI64(builder, loc, outer, inner);
  Value active = lessI64(builder, loc, output, outputs);
  auto guarded = scf::IfOp::create(builder, loc, active, false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(guarded.thenBlock());
    Value initial = kind.getValue() == "max"
        ? Value(arith::ConstantFloatOp::create(
              builder, loc, f32,
              APFloat::getInf(APFloat::IEEEsingle(), /*negative=*/true)))
        : Value(arith::ConstantFloatOp::create(
              builder, loc, f32, APFloat(0.0f)));
    Value outerIndex = arith::DivUIOp::create(builder, loc, output, inner);
    Value innerIndex = arith::RemUIOp::create(builder, loc, output, inner);
    Value first = cooperative ? thread : i64Constant(builder, loc, 0);
    Value step = i64Constant(builder, loc, cooperative ? 128 : 1);
    auto loop = scf::ForOp::create(
        builder, loc, first, axisExtent, step, ValueRange{initial});
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      Value linear = addI64(builder, loc, mulI64(builder, loc,
          addI64(builder, loc, mulI64(builder, loc, outerIndex, axisExtent),
                 loop.getInductionVar()), inner), innerIndex);
      Value ptr = LLVM::GEPOp::create(
          builder, loc, inputs[0].getType(), storageType, inputs[0],
          ValueRange{linear});
      Value loaded = LLVM::LoadOp::create(
          builder, loc, storageType, ptr, alignment);
      Value value = f16Storage
          ? Value(LLVM::FPExtOp::create(builder, loc, f32, loaded))
          : loaded;
      Value next = kind.getValue() == "max"
          ? Value(arith::MaximumFOp::create(
                builder, loc, loop.getRegionIterArgs()[0], value))
          : Value(arith::AddFOp::create(
                builder, loc, loop.getRegionIterArgs()[0], value));
      scf::YieldOp::create(builder, loc, ValueRange{next});
    }
    builder.setInsertionPointAfter(loop);
    Value result = loop.getResult(0);
    if (cooperative) {
      FailureOr<Value> scratch = sm120ReductionScratch(op, builder);
      if (failed(scratch)) {
        op->emitError("failed to materialize cooperative reduction scratch");
        return failure();
      }
      Value scratchPtr = LLVM::GEPOp::create(
          builder, loc, scratch->getType(), f32, *scratch, ValueRange{thread});
      LLVM::StoreOp::create(builder, loc, result, scratchPtr, 4);
      NVVM::BarrierOp::create(builder, loc);
      for (int64_t stride = 64; stride >= 1; stride >>= 1) {
        Value participates = lessI64(
            builder, loc, thread, i64Constant(builder, loc, stride));
        auto combine = scf::IfOp::create(builder, loc, participates, false);
        {
          OpBuilder::InsertionGuard combineGuard(builder);
          builder.setInsertionPointToStart(combine.thenBlock());
          Value lhsPtr = LLVM::GEPOp::create(
              builder, loc, scratch->getType(), f32, *scratch,
              ValueRange{thread});
          Value rhsPtr = LLVM::GEPOp::create(
              builder, loc, scratch->getType(), f32, *scratch,
              ValueRange{addI64(builder, loc, thread,
                                i64Constant(builder, loc, stride))});
          Value lhs = LLVM::LoadOp::create(builder, loc, f32, lhsPtr, 4);
          Value rhs = LLVM::LoadOp::create(builder, loc, f32, rhsPtr, 4);
          Value combined = kind.getValue() == "max"
              ? Value(arith::MaximumFOp::create(builder, loc, lhs, rhs))
              : Value(arith::AddFOp::create(builder, loc, lhs, rhs));
          LLVM::StoreOp::create(builder, loc, combined, lhsPtr, 4);
        }
        builder.setInsertionPointAfter(combine);
        NVVM::BarrierOp::create(builder, loc);
      }
      Value leader = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, thread,
          i64Constant(builder, loc, 0));
      auto store = scf::IfOp::create(builder, loc, leader, false);
      {
        OpBuilder::InsertionGuard storeGuard(builder);
        builder.setInsertionPointToStart(store.thenBlock());
        Value ptr = LLVM::GEPOp::create(
            builder, loc, scratch->getType(), f32, *scratch,
            ValueRange{i64Constant(builder, loc, 0)});
        Value reduced = LLVM::LoadOp::create(builder, loc, f32, ptr, 4);
        if (kind.getValue() == "mean")
          reduced = arith::DivFOp::create(
              builder, loc, reduced,
              arith::UIToFPOp::create(builder, loc, f32, axisExtent));
        Value outPtr = LLVM::GEPOp::create(
            builder, loc, inputs[1].getType(), f32, inputs[1],
            ValueRange{output});
        LLVM::StoreOp::create(builder, loc, reduced, outPtr, 4);
      }
      builder.setInsertionPointAfter(store);
    } else {
      if (kind.getValue() == "mean")
        result = arith::DivFOp::create(
            builder, loc, result,
            arith::UIToFPOp::create(builder, loc, f32, axisExtent));
      Value outPtr = LLVM::GEPOp::create(
          builder, loc, inputs[1].getType(), f32, inputs[1], ValueRange{output});
      LLVM::StoreOp::create(builder, loc, result, outPtr, 4);
    }
  }
  builder.setInsertionPointAfter(guarded);
  op->erase();
  return success();
}

// Correctness-first SDPA materializer. One thread owns one O[b,hq,q,dv]
// element and recomputes the score row. This deliberately favors a small,
// auditable ABI/semantic proof over the existing optimized CUDA-C candidate.
static LogicalResult materializeSm120AttentionKernel(
    tessera::tile::AttentionKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  ValueRange in = kernel.getInputs();
  auto storage = op->getAttrOfType<StringAttr>("storage");
  auto accum = op->getAttrOfType<StringAttr>("accum");
  auto scaleAttr = op->getAttrOfType<FloatAttr>("scale");
  auto causalAttr = op->getAttrOfType<BoolAttr>("causal");
  auto biasAttr = op->getAttrOfType<BoolAttr>("bias");
  auto windowLeftAttr = op->getAttrOfType<IntegerAttr>("window_left");
  auto windowRightAttr = op->getAttrOfType<IntegerAttr>("window_right");
  auto softcapAttr = op->getAttrOfType<FloatAttr>("softcap");
  auto dropoutAttr = op->getAttrOfType<FloatAttr>("dropout_p");
  auto dropoutSeedAttr = op->getAttrOfType<IntegerAttr>("dropout_seed");
  bool hasBias = biasAttr && biasAttr.getValue();
  unsigned outputIndex = 3 + unsigned(hasBias);
  unsigned dimStart = outputIndex + 1;
  bool f16Storage = storage && storage.getValue() == "f16";
  if (in.size() != 11 + unsigned(hasBias) || !storage ||
      (!f16Storage && storage.getValue() != "f32") || !accum ||
      accum.getValue() != "f32" || !scaleAttr || !causalAttr || !biasAttr ||
      !windowLeftAttr || !windowRightAttr || !softcapAttr || !dropoutAttr ||
      !dropoutSeedAttr) {
    op->emitError("sm_120 attention_kernel requires f16/f32 storage, f32 "
                  "accum, complete mask/dropout attrs, and the canonical ABI");
    return failure();
  }
  Location loc = op->getLoc();
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  auto f32 = builder.getF32Type();
  Type storageType = f16Storage ? Type(builder.getF16Type()) : Type(f32);
  unsigned alignment = f16Storage ? 2 : 4;
  Value B = in[dimStart], Hq = in[dimStart + 1], Hkv = in[dimStart + 2];
  Value Sq = in[dimStart + 3], Sk = in[dimStart + 4];
  Value D = in[dimStart + 5], Dv = in[dimStart + 6];
  Value block = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::BlockIdXOp::create(builder, loc, i32));
  Value thread = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::ThreadIdXOp::create(builder, loc, i32));
  Value linearOut = addI64(
      builder, loc, mulI64(builder, loc, block, i64Constant(builder, loc, 128)),
      thread);
  Value total = mulI64(builder, loc, mulI64(builder, loc,
      mulI64(builder, loc, B, Hq), Sq), Dv);
  Value active = lessI64(builder, loc, linearOut, total);
  auto guarded = scf::IfOp::create(builder, loc, active, false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(guarded.thenBlock());
    Value dv = arith::RemUIOp::create(builder, loc, linearOut, Dv);
    Value t0 = arith::DivUIOp::create(builder, loc, linearOut, Dv);
    Value q = arith::RemUIOp::create(builder, loc, t0, Sq);
    Value t1 = arith::DivUIOp::create(builder, loc, t0, Sq);
    Value hq = arith::RemUIOp::create(builder, loc, t1, Hq);
    Value b = arith::DivUIOp::create(builder, loc, t1, Hq);
    Value ratio = arith::DivUIOp::create(builder, loc, Hq, Hkv);
    Value hkv = arith::DivUIOp::create(builder, loc, hq, ratio);
    Value zero = i64Constant(builder, loc, 0);
    Value one = i64Constant(builder, loc, 1);
    Value scale = arith::ConstantFloatOp::create(
        builder, loc, f32, scaleAttr.getValue());
    Value log2e = arith::ConstantFloatOp::create(
        builder, loc, f32, APFloat(1.4426950408889634f));
    auto load = [&](Value base, Value index) -> Value {
      Value ptr = LLVM::GEPOp::create(builder, loc, base.getType(), storageType,
                                      base, ValueRange{index});
      Value value = LLVM::LoadOp::create(builder, loc, storageType, ptr, alignment);
      return f16Storage ? Value(LLVM::FPExtOp::create(builder, loc, f32, value))
                        : value;
    };
    auto score = [&](Value key) -> Value {
      Value qBase = mulI64(builder, loc,
          addI64(builder, loc, mulI64(builder, loc,
              addI64(builder, loc, mulI64(builder, loc, b, Hq), hq), Sq), q), D);
      Value kBase = mulI64(builder, loc,
          addI64(builder, loc, mulI64(builder, loc,
              addI64(builder, loc, mulI64(builder, loc, b, Hkv), hkv), Sk), key), D);
      Value z = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
      auto dot = scf::ForOp::create(builder, loc, zero, D, one, ValueRange{z});
      {
        OpBuilder::InsertionGuard dotGuard(builder);
        builder.setInsertionPointToStart(dot.getBody());
        Value qv = load(in[0], addI64(builder, loc, qBase, dot.getInductionVar()));
        Value kv = load(in[1], addI64(builder, loc, kBase, dot.getInductionVar()));
        Value product = arith::MulFOp::create(builder, loc, qv, kv);
        Value next = arith::AddFOp::create(
            builder, loc, dot.getRegionIterArgs()[0], product);
        scf::YieldOp::create(builder, loc, ValueRange{next});
      }
      builder.setInsertionPointAfter(dot);
      Value value = arith::MulFOp::create(builder, loc, dot.getResult(0), scale);
      if (hasBias) {
        Value biasIndex = addI64(builder, loc,
            mulI64(builder, loc,
                addI64(builder, loc, mulI64(builder, loc,
                    addI64(builder, loc, mulI64(builder, loc, b, Hq), hq), Sq), q), Sk),
            key);
        Value biasPtr = LLVM::GEPOp::create(builder, loc, in[3].getType(), f32,
                                            in[3], ValueRange{biasIndex});
        Value bias = LLVM::LoadOp::create(builder, loc, f32, biasPtr, 4);
        value = arith::AddFOp::create(builder, loc, value, bias);
      }
      if (softcapAttr.getValueAsDouble() > 0.0) {
        Value cap = arith::ConstantFloatOp::create(
            builder, loc, f32, softcapAttr.getValue());
        Value normalized = arith::DivFOp::create(builder, loc, value, cap);
        value = arith::MulFOp::create(
            builder, loc, cap,
            tessera::tile::emitBoundedTanhApprox(builder, loc, normalized));
      }
      Value legal = arith::ConstantIntOp::create(builder, loc, 1, 1);
      if (causalAttr.getValue())
        legal = arith::AndIOp::create(
            builder, loc, legal,
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ule, key, q));
      if (windowLeftAttr.getInt() >= 0) {
        Value lower = arith::SubIOp::create(
            builder, loc, q, i64Constant(builder, loc, windowLeftAttr.getInt()));
        legal = arith::AndIOp::create(
            builder, loc, legal,
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge, key, lower));
      }
      if (windowRightAttr.getInt() >= 0) {
        Value upper = addI64(builder, loc, q,
                             i64Constant(builder, loc, windowRightAttr.getInt()));
        legal = arith::AndIOp::create(
            builder, loc, legal,
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle, key, upper));
      }
      Value negInf = arith::ConstantFloatOp::create(
          builder, loc, f32,
          APFloat::getInf(APFloat::IEEEsingle(), /*negative=*/true));
      value = arith::SelectOp::create(builder, loc, legal, value, negInf);
      return value;
    };
    Value negInf = arith::ConstantFloatOp::create(
        builder, loc, f32,
        APFloat::getInf(APFloat::IEEEsingle(), /*negative=*/true));
    auto maxLoop = scf::ForOp::create(builder, loc, zero, Sk, one,
                                      ValueRange{negInf});
    {
      OpBuilder::InsertionGuard maxGuard(builder);
      builder.setInsertionPointToStart(maxLoop.getBody());
      Value s = score(maxLoop.getInductionVar());
      Value next = arith::MaximumFOp::create(
          builder, loc, maxLoop.getRegionIterArgs()[0], s);
      scf::YieldOp::create(builder, loc, ValueRange{next});
    }
    builder.setInsertionPointAfter(maxLoop);
    Value z = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
    auto accumLoop = scf::ForOp::create(builder, loc, zero, Sk, one,
                                        ValueRange{z, z});
    {
      OpBuilder::InsertionGuard accumGuard(builder);
      builder.setInsertionPointToStart(accumLoop.getBody());
      Value key = accumLoop.getInductionVar();
      Value shifted = arith::SubFOp::create(builder, loc, score(key),
                                            maxLoop.getResult(0));
      Value weight = NVVM::Ex2Op::create(
          builder, loc,
          arith::MulFOp::create(builder, loc, shifted, log2e), false);
      Value outputWeight = weight;
      if (dropoutAttr.getValueAsDouble() > 0.0) {
        Value counter = addI64(builder, loc,
            mulI64(builder, loc,
                addI64(builder, loc, mulI64(builder, loc,
                    addI64(builder, loc, mulI64(builder, loc, b, Hq), hq), Sq), q), Sk),
            key);
        Value hash = addI64(builder, loc,
            mulI64(builder, loc, counter, i64Constant(builder, loc, 1664525)),
            i64Constant(builder, loc, dropoutSeedAttr.getInt() + 1013904223));
        hash = arith::AndIOp::create(
            builder, loc, hash, i64Constant(builder, loc, 0xffffffffULL));
        uint64_t threshold = static_cast<uint64_t>(
            dropoutAttr.getValueAsDouble() * 4294967296.0);
        Value keep = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::uge, hash,
            i64Constant(builder, loc, threshold));
        Value invKeep = arith::ConstantFloatOp::create(
            builder, loc, f32,
            APFloat(static_cast<float>(1.0 / (1.0 - dropoutAttr.getValueAsDouble()))));
        Value scaledWeight = arith::MulFOp::create(builder, loc, weight, invKeep);
        Value zeroWeight = arith::ConstantFloatOp::create(
            builder, loc, f32, APFloat(0.0f));
        outputWeight = arith::SelectOp::create(
            builder, loc, keep, scaledWeight, zeroWeight);
      }
      Value vIndex = addI64(builder, loc,
          mulI64(builder, loc,
              addI64(builder, loc, mulI64(builder, loc,
                  addI64(builder, loc, mulI64(builder, loc, b, Hkv), hkv), Sk), key), Dv),
          dv);
      Value vv = load(in[2], vIndex);
      Value denom = arith::AddFOp::create(
          builder, loc, accumLoop.getRegionIterArgs()[0], weight);
      Value numer = arith::AddFOp::create(
          builder, loc, accumLoop.getRegionIterArgs()[1],
          arith::MulFOp::create(builder, loc, outputWeight, vv));
      scf::YieldOp::create(builder, loc, ValueRange{denom, numer});
    }
    builder.setInsertionPointAfter(accumLoop);
    Value result = arith::DivFOp::create(builder, loc, accumLoop.getResult(1),
                                         accumLoop.getResult(0));
    Value outPtr = LLVM::GEPOp::create(builder, loc, in[outputIndex].getType(), f32,
                                       in[outputIndex], ValueRange{linearOut});
    LLVM::StoreOp::create(builder, loc, result, outPtr, 4);
  }
  builder.setInsertionPointAfter(guarded);
  op->erase();
  return success();
}

static LogicalResult materializeSm120AttentionBackwardKernel(
    tessera::tile::AttentionBackwardKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  ValueRange in = kernel.getInputs();
  auto biasAttr = op->getAttrOfType<BoolAttr>("bias");
  auto storage = op->getAttrOfType<StringAttr>("storage");
  auto accum = op->getAttrOfType<StringAttr>("accum");
  auto scaleAttr = op->getAttrOfType<FloatAttr>("scale");
  auto causalAttr = op->getAttrOfType<BoolAttr>("causal");
  auto windowLeftAttr = op->getAttrOfType<IntegerAttr>("window_left");
  auto windowRightAttr = op->getAttrOfType<IntegerAttr>("window_right");
  auto softcapAttr = op->getAttrOfType<FloatAttr>("softcap");
  auto route = op->getAttrOfType<StringAttr>("route");
  auto deterministic = op->getAttrOfType<BoolAttr>("deterministic");
  auto workspace = op->getAttrOfType<IntegerAttr>("workspace_bytes");
  const bool hasBias = biasAttr && biasAttr.getValue();
  if (in.size() != 14 + unsigned(hasBias) || !storage ||
      storage.getValue() != "f32" || !accum || accum.getValue() != "f32" ||
      !scaleAttr || !causalAttr || !windowLeftAttr || !windowRightAttr ||
      !softcapAttr || !route || route.getValue() != "deterministic_direct" ||
      !deterministic || !deterministic.getValue() || !workspace ||
      workspace.getInt() != 0) {
    op->emitError("sm_120 attention_backward_kernel requires the canonical "
                  "deterministic-direct f32 ABI");
    return failure();
  }

  Location loc = op->getLoc();
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  auto f32 = builder.getF32Type();
  const unsigned biasIndex = 4;
  const unsigned dqIndex = 4 + unsigned(hasBias);
  const unsigned dkIndex = dqIndex + 1;
  const unsigned dvIndex = dqIndex + 2;
  const unsigned dimIndex = dqIndex + 3;
  Value B = in[dimIndex], Hq = in[dimIndex + 1], Hkv = in[dimIndex + 2];
  Value Sq = in[dimIndex + 3], Sk = in[dimIndex + 4];
  Value D = in[dimIndex + 5], Dv = in[dimIndex + 6];
  Value zero = i64Constant(builder, loc, 0);
  Value one = i64Constant(builder, loc, 1);
  Value scale = arith::ConstantFloatOp::create(
      builder, loc, f32, scaleAttr.getValue());
  Value log2e = arith::ConstantFloatOp::create(
      builder, loc, f32, APFloat(1.4426950408889634f));
  Value ratio = arith::DivUIOp::create(builder, loc, Hq, Hkv);

  auto load = [&](unsigned pointer, Value index) -> Value {
    Value ptr = LLVM::GEPOp::create(builder, loc, in[pointer].getType(), f32,
                                    in[pointer], ValueRange{index});
    return LLVM::LoadOp::create(builder, loc, f32, ptr, 4);
  };
  auto qIndex = [&](Value b, Value h, Value q, Value d) -> Value {
    return addI64(builder, loc,
        mulI64(builder, loc,
            addI64(builder, loc,
                mulI64(builder, loc,
                    addI64(builder, loc, mulI64(builder, loc, b, Hq), h), Sq), q), D), d);
  };
  auto kvIndex = [&](Value b, Value h, Value key, Value d, Value width) -> Value {
    return addI64(builder, loc,
        mulI64(builder, loc,
            addI64(builder, loc,
                mulI64(builder, loc,
                    addI64(builder, loc, mulI64(builder, loc, b, Hkv), h), Sk), key), width), d);
  };
  auto outputIndex = [&](Value b, Value h, Value q, Value d) -> Value {
    return addI64(builder, loc,
        mulI64(builder, loc,
            addI64(builder, loc,
                mulI64(builder, loc,
                    addI64(builder, loc, mulI64(builder, loc, b, Hq), h), Sq), q), Dv), d);
  };
  auto legalKey = [&](Value q, Value key) -> Value {
    Value legal = arith::ConstantIntOp::create(builder, loc, 1, 1);
    if (causalAttr.getValue())
      legal = arith::AndIOp::create(
          builder, loc, legal,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ule, key, q));
    if (windowLeftAttr.getInt() >= 0) {
      Value lower = arith::SubIOp::create(
          builder, loc, q, i64Constant(builder, loc, windowLeftAttr.getInt()));
      legal = arith::AndIOp::create(
          builder, loc, legal,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge, key, lower));
    }
    if (windowRightAttr.getInt() >= 0) {
      Value upper = addI64(builder, loc, q,
                           i64Constant(builder, loc, windowRightAttr.getInt()));
      legal = arith::AndIOp::create(
          builder, loc, legal,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle, key, upper));
    }
    return legal;
  };
  auto rawScore = [&](Value b, Value hq, Value hkv, Value q, Value key) -> Value {
    Value zf = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
    auto dot = scf::ForOp::create(builder, loc, zero, D, one, ValueRange{zf});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(dot.getBody());
      Value d = dot.getInductionVar();
      Value product = arith::MulFOp::create(
          builder, loc, load(1, qIndex(b, hq, q, d)),
          load(2, kvIndex(b, hkv, key, d, D)));
      scf::YieldOp::create(builder, loc, ValueRange{
          arith::AddFOp::create(builder, loc, dot.getRegionIterArgs()[0], product)});
    }
    builder.setInsertionPointAfter(dot);
    Value value = arith::MulFOp::create(builder, loc, dot.getResult(0), scale);
    if (hasBias) {
      Value index = addI64(builder, loc,
          mulI64(builder, loc,
              addI64(builder, loc,
                  mulI64(builder, loc,
                      addI64(builder, loc, mulI64(builder, loc, b, Hq), hq), Sq), q), Sk), key);
      value = arith::AddFOp::create(builder, loc, value, load(biasIndex, index));
    }
    return value;
  };
  auto cappedScore = [&](Value raw) -> Value {
    if (softcapAttr.getValueAsDouble() <= 0.0)
      return raw;
    Value cap = arith::ConstantFloatOp::create(
        builder, loc, f32, softcapAttr.getValue());
    return arith::MulFOp::create(
        builder, loc, cap,
        tessera::tile::emitBoundedTanhApprox(
            builder, loc, arith::DivFOp::create(builder, loc, raw, cap)));
  };
  auto softcapDerivative = [&](Value raw) -> Value {
    Value onef = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(1.0f));
    if (softcapAttr.getValueAsDouble() <= 0.0)
      return onef;
    Value cap = arith::ConstantFloatOp::create(
        builder, loc, f32, softcapAttr.getValue());
    Value t = tessera::tile::emitBoundedTanhApprox(
        builder, loc, arith::DivFOp::create(builder, loc, raw, cap));
    return arith::SubFOp::create(
        builder, loc, onef, arith::MulFOp::create(builder, loc, t, t));
  };
  auto maskedScore = [&](Value b, Value hq, Value hkv, Value q, Value key) -> Value {
    Value value = cappedScore(rawScore(b, hq, hkv, q, key));
    Value negInf = arith::ConstantFloatOp::create(
        builder, loc, f32, APFloat::getInf(APFloat::IEEEsingle(), true));
    return arith::SelectOp::create(builder, loc, legalKey(q, key), value, negInf);
  };
  auto doDotV = [&](Value b, Value hq, Value hkv, Value q, Value key) -> Value {
    Value zf = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
    auto dot = scf::ForOp::create(builder, loc, zero, Dv, one, ValueRange{zf});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(dot.getBody());
      Value d = dot.getInductionVar();
      Value product = arith::MulFOp::create(
          builder, loc, load(0, outputIndex(b, hq, q, d)),
          load(3, kvIndex(b, hkv, key, d, Dv)));
      scf::YieldOp::create(builder, loc, ValueRange{
          arith::AddFOp::create(builder, loc, dot.getRegionIterArgs()[0], product)});
    }
    builder.setInsertionPointAfter(dot);
    return dot.getResult(0);
  };
  // Return the row maximum, softmax denominator, and dO dot O delta.
  auto rowStats = [&](Value b, Value hq, Value hkv, Value q) -> SmallVector<Value, 3> {
    Value negInf = arith::ConstantFloatOp::create(
        builder, loc, f32, APFloat::getInf(APFloat::IEEEsingle(), true));
    auto maxLoop = scf::ForOp::create(builder, loc, zero, Sk, one, ValueRange{negInf});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(maxLoop.getBody());
      Value s = maskedScore(b, hq, hkv, q, maxLoop.getInductionVar());
      scf::YieldOp::create(builder, loc, ValueRange{
          arith::MaximumFOp::create(builder, loc, maxLoop.getRegionIterArgs()[0], s)});
    }
    builder.setInsertionPointAfter(maxLoop);
    Value zf = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
    auto sums = scf::ForOp::create(builder, loc, zero, Sk, one, ValueRange{zf, zf});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(sums.getBody());
      Value key = sums.getInductionVar();
      Value shifted = arith::SubFOp::create(
          builder, loc, maskedScore(b, hq, hkv, q, key), maxLoop.getResult(0));
      Value weight = NVVM::Ex2Op::create(
          builder, loc, arith::MulFOp::create(builder, loc, shifted, log2e), false);
      Value denom = arith::AddFOp::create(
          builder, loc, sums.getRegionIterArgs()[0], weight);
      Value numer = arith::AddFOp::create(
          builder, loc, sums.getRegionIterArgs()[1],
          arith::MulFOp::create(builder, loc, weight, doDotV(b, hq, hkv, q, key)));
      scf::YieldOp::create(builder, loc, ValueRange{denom, numer});
    }
    builder.setInsertionPointAfter(sums);
    Value delta = arith::DivFOp::create(
        builder, loc, sums.getResult(1), sums.getResult(0));
    return {maxLoop.getResult(0), sums.getResult(0), delta};
  };
  auto probability = [&](Value b, Value hq, Value hkv, Value q, Value key,
                         Value maxValue, Value denom) -> Value {
    Value shifted = arith::SubFOp::create(
        builder, loc, maskedScore(b, hq, hkv, q, key), maxValue);
    Value weight = NVVM::Ex2Op::create(
        builder, loc, arith::MulFOp::create(builder, loc, shifted, log2e), false);
    return arith::DivFOp::create(builder, loc, weight, denom);
  };

  Value block = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::BlockIdXOp::create(builder, loc, i32));
  Value thread = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::ThreadIdXOp::create(builder, loc, i32));
  Value linear = addI64(builder, loc,
      mulI64(builder, loc, block, i64Constant(builder, loc, 128)), thread);
  Value dqCount = mulI64(builder, loc, mulI64(builder, loc,
      mulI64(builder, loc, B, Hq), Sq), D);
  Value dkCount = mulI64(builder, loc, mulI64(builder, loc,
      mulI64(builder, loc, B, Hkv), Sk), D);
  Value dvCount = mulI64(builder, loc, mulI64(builder, loc,
      mulI64(builder, loc, B, Hkv), Sk), Dv);

  auto dqGuard = scf::IfOp::create(builder, loc,
      lessI64(builder, loc, linear, dqCount), false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(dqGuard.thenBlock());
    Value d = arith::RemUIOp::create(builder, loc, linear, D);
    Value t0 = arith::DivUIOp::create(builder, loc, linear, D);
    Value q = arith::RemUIOp::create(builder, loc, t0, Sq);
    Value t1 = arith::DivUIOp::create(builder, loc, t0, Sq);
    Value hq = arith::RemUIOp::create(builder, loc, t1, Hq);
    Value b = arith::DivUIOp::create(builder, loc, t1, Hq);
    Value hkv = arith::DivUIOp::create(builder, loc, hq, ratio);
    SmallVector<Value, 3> stats = rowStats(b, hq, hkv, q);
    Value zf = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
    auto sum = scf::ForOp::create(builder, loc, zero, Sk, one, ValueRange{zf});
    {
      OpBuilder::InsertionGuard sumGuard(builder);
      builder.setInsertionPointToStart(sum.getBody());
      Value key = sum.getInductionVar();
      Value raw = rawScore(b, hq, hkv, q, key);
      Value p = probability(b, hq, hkv, q, key, stats[0], stats[1]);
      Value ds = arith::MulFOp::create(builder, loc, p,
          arith::SubFOp::create(builder, loc, doDotV(b, hq, hkv, q, key), stats[2]));
      ds = arith::MulFOp::create(builder, loc, ds, softcapDerivative(raw));
      Value term = arith::MulFOp::create(builder, loc,
          arith::MulFOp::create(builder, loc, ds, scale),
          load(2, kvIndex(b, hkv, key, d, D)));
      scf::YieldOp::create(builder, loc, ValueRange{
          arith::AddFOp::create(builder, loc, sum.getRegionIterArgs()[0], term)});
    }
    builder.setInsertionPointAfter(sum);
    Value ptr = LLVM::GEPOp::create(builder, loc, in[dqIndex].getType(), f32,
                                    in[dqIndex], ValueRange{linear});
    LLVM::StoreOp::create(builder, loc, sum.getResult(0), ptr, 4);
  }
  builder.setInsertionPointAfter(dqGuard);

  Value dkLinear = arith::SubIOp::create(builder, loc, linear, dqCount);
  auto dkGuard = scf::IfOp::create(builder, loc,
      arith::AndIOp::create(builder, loc,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::uge, linear, dqCount),
          lessI64(builder, loc, dkLinear, dkCount)), false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(dkGuard.thenBlock());
    Value d = arith::RemUIOp::create(builder, loc, dkLinear, D);
    Value t0 = arith::DivUIOp::create(builder, loc, dkLinear, D);
    Value key = arith::RemUIOp::create(builder, loc, t0, Sk);
    Value t1 = arith::DivUIOp::create(builder, loc, t0, Sk);
    Value hkv = arith::RemUIOp::create(builder, loc, t1, Hkv);
    Value b = arith::DivUIOp::create(builder, loc, t1, Hkv);
    Value hqBegin = mulI64(builder, loc, hkv, ratio);
    Value hqEnd = addI64(builder, loc, hqBegin, ratio);
    Value zf = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
    auto headLoop = scf::ForOp::create(builder, loc, hqBegin, hqEnd, one, ValueRange{zf});
    {
      OpBuilder::InsertionGuard headGuard(builder);
      builder.setInsertionPointToStart(headLoop.getBody());
      Value hq = headLoop.getInductionVar();
      auto qLoop = scf::ForOp::create(builder, loc, zero, Sq, one,
                                      ValueRange{headLoop.getRegionIterArgs()[0]});
      {
        OpBuilder::InsertionGuard qGuard(builder);
        builder.setInsertionPointToStart(qLoop.getBody());
        Value q = qLoop.getInductionVar();
        SmallVector<Value, 3> stats = rowStats(b, hq, hkv, q);
        Value raw = rawScore(b, hq, hkv, q, key);
        Value p = probability(b, hq, hkv, q, key, stats[0], stats[1]);
        Value ds = arith::MulFOp::create(builder, loc, p,
            arith::SubFOp::create(builder, loc, doDotV(b, hq, hkv, q, key), stats[2]));
        ds = arith::MulFOp::create(builder, loc, ds, softcapDerivative(raw));
        Value term = arith::MulFOp::create(builder, loc,
            arith::MulFOp::create(builder, loc, ds, scale),
            load(1, qIndex(b, hq, q, d)));
        scf::YieldOp::create(builder, loc, ValueRange{
            arith::AddFOp::create(builder, loc, qLoop.getRegionIterArgs()[0], term)});
      }
      builder.setInsertionPointAfter(qLoop);
      scf::YieldOp::create(builder, loc, ValueRange{qLoop.getResult(0)});
    }
    builder.setInsertionPointAfter(headLoop);
    Value ptr = LLVM::GEPOp::create(builder, loc, in[dkIndex].getType(), f32,
                                    in[dkIndex], ValueRange{dkLinear});
    LLVM::StoreOp::create(builder, loc, headLoop.getResult(0), ptr, 4);
  }
  builder.setInsertionPointAfter(dkGuard);

  Value dvLinear = arith::SubIOp::create(builder, loc, dkLinear, dkCount);
  Value dqDkCount = addI64(builder, loc, dqCount, dkCount);
  auto dvGuard = scf::IfOp::create(builder, loc,
      arith::AndIOp::create(builder, loc,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::uge, linear, dqDkCount),
          lessI64(builder, loc, dvLinear, dvCount)), false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(dvGuard.thenBlock());
    Value d = arith::RemUIOp::create(builder, loc, dvLinear, Dv);
    Value t0 = arith::DivUIOp::create(builder, loc, dvLinear, Dv);
    Value key = arith::RemUIOp::create(builder, loc, t0, Sk);
    Value t1 = arith::DivUIOp::create(builder, loc, t0, Sk);
    Value hkv = arith::RemUIOp::create(builder, loc, t1, Hkv);
    Value b = arith::DivUIOp::create(builder, loc, t1, Hkv);
    Value hqBegin = mulI64(builder, loc, hkv, ratio);
    Value hqEnd = addI64(builder, loc, hqBegin, ratio);
    Value zf = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
    auto headLoop = scf::ForOp::create(builder, loc, hqBegin, hqEnd, one, ValueRange{zf});
    {
      OpBuilder::InsertionGuard headGuard(builder);
      builder.setInsertionPointToStart(headLoop.getBody());
      Value hq = headLoop.getInductionVar();
      auto qLoop = scf::ForOp::create(builder, loc, zero, Sq, one,
                                      ValueRange{headLoop.getRegionIterArgs()[0]});
      {
        OpBuilder::InsertionGuard qGuard(builder);
        builder.setInsertionPointToStart(qLoop.getBody());
        Value q = qLoop.getInductionVar();
        SmallVector<Value, 3> stats = rowStats(b, hq, hkv, q);
        Value p = probability(b, hq, hkv, q, key, stats[0], stats[1]);
        Value term = arith::MulFOp::create(
            builder, loc, p, load(0, outputIndex(b, hq, q, d)));
        scf::YieldOp::create(builder, loc, ValueRange{
            arith::AddFOp::create(builder, loc, qLoop.getRegionIterArgs()[0], term)});
      }
      builder.setInsertionPointAfter(qLoop);
      scf::YieldOp::create(builder, loc, ValueRange{qLoop.getResult(0)});
    }
    builder.setInsertionPointAfter(headLoop);
    Value ptr = LLVM::GEPOp::create(builder, loc, in[dvIndex].getType(), f32,
                                    in[dvIndex], ValueRange{dvLinear});
    LLVM::StoreOp::create(builder, loc, headLoop.getResult(0), ptr, 4);
  }
  builder.setInsertionPointAfter(dvGuard);
  op->erase();
  return success();
}

static LogicalResult materializeSm120PagedKVReadKernel(
    tessera::tile::PagedKVReadKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  ValueRange in = kernel.getInputs();
  auto storage = op->getAttrOfType<StringAttr>("storage");
  auto tableStorage = op->getAttrOfType<StringAttr>("table_storage");
  auto route = op->getAttrOfType<StringAttr>("route");
  if (in.size() != 10 || !storage || storage.getValue() != "f32" ||
      !tableStorage || tableStorage.getValue() != "i32" || !route ||
      route.getValue() != "direct") {
    op->emitError("sm_120 paged_kv_read_kernel requires f32 pages, i32 table, "
                  "route=direct, and the canonical ten-operand ABI");
    return failure();
  }
  Location loc = op->getLoc();
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  Type f32 = builder.getF32Type();
  Value pageSize = in[5], heads = in[6], dim = in[7];
  Value start = in[8], tokens = in[9];
  Value block = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::BlockIdXOp::create(builder, loc, i32));
  Value thread = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::ThreadIdXOp::create(builder, loc, i32));
  Value z = addI64(builder, loc,
      mulI64(builder, loc, block, i64Constant(builder, loc, 256)), thread);
  Value total = mulI64(builder, loc, mulI64(builder, loc, tokens, heads), dim);
  auto guarded = scf::IfOp::create(builder, loc, lessI64(builder, loc, z, total), false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(guarded.thenBlock());
    Value d = arith::RemUIOp::create(builder, loc, z, dim);
    Value t0 = arith::DivUIOp::create(builder, loc, z, dim);
    Value h = arith::RemUIOp::create(builder, loc, t0, heads);
    Value token = arith::DivUIOp::create(builder, loc, t0, heads);
    Value logical = addI64(builder, loc, start, token);
    Value logicalPage = arith::DivUIOp::create(builder, loc, logical, pageSize);
    Value pageOffset = arith::RemUIOp::create(builder, loc, logical, pageSize);
    Value tablePtr = LLVM::GEPOp::create(builder, loc, in[1].getType(), i32,
                                         in[1], ValueRange{logicalPage});
    Value physical32 = LLVM::LoadOp::create(builder, loc, i32, tablePtr, 4);
    Value physical = arith::ExtUIOp::create(builder, loc, i64, physical32);
    Value pageIndex = addI64(builder, loc,
        mulI64(builder, loc,
            addI64(builder, loc,
                mulI64(builder, loc,
                    addI64(builder, loc, mulI64(builder, loc, physical, pageSize), pageOffset),
                    heads), h),
            dim), d);
    Value pagePtr = LLVM::GEPOp::create(builder, loc, in[0].getType(), f32,
                                        in[0], ValueRange{pageIndex});
    Value value = LLVM::LoadOp::create(builder, loc, f32, pagePtr, 4);
    Value outPtr = LLVM::GEPOp::create(builder, loc, in[2].getType(), f32,
                                       in[2], ValueRange{z});
    LLVM::StoreOp::create(builder, loc, value, outPtr, 4);
  }
  builder.setInsertionPointAfter(guarded);
  op->erase();
  return success();
}

static Value sm120LinearThread(OpBuilder &builder, Location loc,
                               int64_t threads) {
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  Value block = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::BlockIdXOp::create(builder, loc, i32));
  Value thread = arith::ExtUIOp::create(
      builder, loc, i64, NVVM::ThreadIdXOp::create(builder, loc, i32));
  return addI64(builder, loc,
                mulI64(builder, loc, block, i64Constant(builder, loc, threads)),
                thread);
}

static Value sm120LoadF32(OpBuilder &builder, Location loc, Value base,
                          Value index) {
  Type f32 = builder.getF32Type();
  Value ptr = LLVM::GEPOp::create(builder, loc, base.getType(), f32, base,
                                  ValueRange{index});
  return LLVM::LoadOp::create(builder, loc, f32, ptr, 4);
}

static Value sm120LoadI32(OpBuilder &builder, Location loc, Value base,
                          Value index) {
  Type i32 = builder.getI32Type();
  Value ptr = LLVM::GEPOp::create(builder, loc, base.getType(), i32, base,
                                  ValueRange{index});
  return LLVM::LoadOp::create(builder, loc, i32, ptr, 4);
}

static void sm120StoreF32(OpBuilder &builder, Location loc, Value base,
                          Value index, Value value) {
  Type f32 = builder.getF32Type();
  Value ptr = LLVM::GEPOp::create(builder, loc, base.getType(), f32, base,
                                  ValueRange{index});
  LLVM::StoreOp::create(builder, loc, value, ptr, 4);
}

static Value sm120ExpF32(OpBuilder &builder, Location loc, Value value) {
  Value log2e = arith::ConstantFloatOp::create(
      builder, loc, builder.getF32Type(), APFloat(1.4426950408889634f));
  return NVVM::Ex2Op::create(
      builder, loc, arith::MulFOp::create(builder, loc, value, log2e), false);
}

static LogicalResult materializeSm120ReplaySSMDecodeKernel(
    tessera::tile::ReplaySSMDecodeKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  ValueRange in = kernel.getInputs();
  auto storage = op->getAttrOfType<StringAttr>("storage");
  auto route = op->getAttrOfType<StringAttr>("route");
  if (in.size() != 11 || !storage || storage.getValue() != "f32" || !route ||
      route.getValue() != "output_only") {
    op->emitError("sm_120 ReplaySSM decode requires f32 output_only and the canonical ABI");
    return failure();
  }
  Location loc = op->getLoc();
  FloatType f32 = builder.getF32Type();
  Value B = in[7], D = in[8], N = in[9], M = in[10];
  Value z = sm120LinearThread(builder, loc, 128);
  Value total = mulI64(builder, loc, B, D);
  auto guarded = scf::IfOp::create(builder, loc, lessI64(builder, loc, z, total), false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(guarded.thenBlock());
    Value bi = arith::DivUIOp::create(builder, loc, z, D);
    Value di = arith::RemUIOp::create(builder, loc, z, D);
    Value zero = i64Constant(builder, loc, 0), one = i64Constant(builder, loc, 1);
    Value zf = arith::ConstantFloatOp::create(builder, loc, f32, APFloat(0.0f));
    Value a = sm120LoadF32(builder, loc, in[5], di);
    auto decayLoop = scf::ForOp::create(builder, loc, zero, M, one, ValueRange{zf});
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(decayLoop.getBody());
      Value i = decayLoop.getInductionVar();
      Value q = addI64(builder, loc,
          mulI64(builder, loc, addI64(builder, loc, mulI64(builder, loc, i, B), bi), D), di);
      Value term = arith::MulFOp::create(builder, loc,
          sm120LoadF32(builder, loc, in[0], q), a);
      scf::YieldOp::create(builder, loc, ValueRange{arith::AddFOp::create(
          builder, loc, decayLoop.getRegionIterArgs()[0], term)});
    }
    builder.setInsertionPointAfter(decayLoop);
    auto baseLoop = scf::ForOp::create(builder, loc, zero, N, one, ValueRange{zf});
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(baseLoop.getBody());
      Value n = baseLoop.getInductionVar();
      Value cIdx = addI64(builder, loc, mulI64(builder, loc, bi, N), n);
      Value sIdx = addI64(builder, loc, mulI64(builder, loc, z, N), n);
      Value term = arith::MulFOp::create(builder, loc,
          sm120LoadF32(builder, loc, in[4], cIdx),
          sm120LoadF32(builder, loc, in[3], sIdx));
      scf::YieldOp::create(builder, loc, ValueRange{arith::AddFOp::create(
          builder, loc, baseLoop.getRegionIterArgs()[0], term)});
    }
    builder.setInsertionPointAfter(baseLoop);
    Value initial = arith::MulFOp::create(
        builder, loc, baseLoop.getResult(0), sm120ExpF32(builder, loc, decayLoop.getResult(0)));
    auto replayLoop = scf::ForOp::create(builder, loc, zero, M, one,
                                         ValueRange{zf, initial});
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(replayLoop.getBody());
      Value i = replayLoop.getInductionVar();
      Value q = addI64(builder, loc,
          mulI64(builder, loc, addI64(builder, loc, mulI64(builder, loc, i, B), bi), D), di);
      Value delta = sm120LoadF32(builder, loc, in[0], q);
      Value prefix = arith::AddFOp::create(builder, loc,
          replayLoop.getRegionIterArgs()[0], arith::MulFOp::create(builder, loc, delta, a));
      auto gramLoop = scf::ForOp::create(builder, loc, zero, N, one, ValueRange{zf});
      {
        OpBuilder::InsertionGuard gramGuard(builder);
        builder.setInsertionPointToStart(gramLoop.getBody());
        Value n = gramLoop.getInductionVar();
        Value cIdx = addI64(builder, loc, mulI64(builder, loc, bi, N), n);
        Value bIdx = addI64(builder, loc,
            mulI64(builder, loc, addI64(builder, loc, mulI64(builder, loc, i, B), bi), N), n);
        Value term = arith::MulFOp::create(builder, loc,
            sm120LoadF32(builder, loc, in[4], cIdx),
            sm120LoadF32(builder, loc, in[2], bIdx));
        scf::YieldOp::create(builder, loc, ValueRange{arith::AddFOp::create(
            builder, loc, gramLoop.getRegionIterArgs()[0], term)});
      }
      builder.setInsertionPointAfter(gramLoop);
      Value exponent = arith::SubFOp::create(builder, loc, decayLoop.getResult(0), prefix);
      Value term = arith::MulFOp::create(builder, loc,
          arith::MulFOp::create(builder, loc, sm120ExpF32(builder, loc, exponent), delta),
          arith::MulFOp::create(builder, loc,
              sm120LoadF32(builder, loc, in[1], q), gramLoop.getResult(0)));
      Value output = arith::AddFOp::create(builder, loc,
          replayLoop.getRegionIterArgs()[1], term);
      scf::YieldOp::create(builder, loc, ValueRange{prefix, output});
    }
    builder.setInsertionPointAfter(replayLoop);
    sm120StoreF32(builder, loc, in[6], z, replayLoop.getResult(1));
  }
  builder.setInsertionPointAfter(guarded);
  op->erase();
  return success();
}

static LogicalResult materializeSm120ReplaySSMFlushKernel(
    tessera::tile::ReplaySSMFlushKernelOp kernel, OpBuilder &builder) {
  Operation *op = kernel.getOperation();
  ValueRange in = kernel.getInputs();
  if (in.size() != 9) return failure();
  Location loc = op->getLoc();
  Value B=in[5], D=in[6], N=in[7], M=in[8];
  Value z=sm120LinearThread(builder,loc,128);
  Value total=mulI64(builder,loc,mulI64(builder,loc,B,D),N);
  auto guarded=scf::IfOp::create(builder,loc,lessI64(builder,loc,z,total),false);
  {
    OpBuilder::InsertionGuard guard(builder); builder.setInsertionPointToStart(guarded.thenBlock());
    Value n=arith::RemUIOp::create(builder,loc,z,N);
    Value t=arith::DivUIOp::create(builder,loc,z,N);
    Value di=arith::RemUIOp::create(builder,loc,t,D);
    Value bi=arith::DivUIOp::create(builder,loc,t,D);
    Value a=sm120LoadF32(builder,loc,in[4],di);
    Value zero=i64Constant(builder,loc,0), one=i64Constant(builder,loc,1);
    auto loop=scf::ForOp::create(builder,loc,zero,M,one,
                                 ValueRange{sm120LoadF32(builder,loc,in[3],z)});
    {
      OpBuilder::InsertionGuard loopGuard(builder); builder.setInsertionPointToStart(loop.getBody());
      Value i=loop.getInductionVar();
      Value q=addI64(builder,loc,mulI64(builder,loc,addI64(builder,loc,mulI64(builder,loc,i,B),bi),D),di);
      Value bIdx=addI64(builder,loc,mulI64(builder,loc,addI64(builder,loc,mulI64(builder,loc,i,B),bi),N),n);
      Value decay=sm120ExpF32(builder,loc,arith::MulFOp::create(builder,loc,
          sm120LoadF32(builder,loc,in[0],q),a));
      Value next=arith::AddFOp::create(builder,loc,
          arith::MulFOp::create(builder,loc,decay,loop.getRegionIterArgs()[0]),
          arith::MulFOp::create(builder,loc,
              arith::MulFOp::create(builder,loc,sm120LoadF32(builder,loc,in[0],q),
                                    sm120LoadF32(builder,loc,in[1],q)),
              sm120LoadF32(builder,loc,in[2],bIdx)));
      scf::YieldOp::create(builder,loc,ValueRange{next});
    }
    builder.setInsertionPointAfter(loop); sm120StoreF32(builder,loc,in[3],z,loop.getResult(0));
  }
  builder.setInsertionPointAfter(guarded); op->erase(); return success();
}

static LogicalResult materializeSm120MoEDispatchKernel(
    tessera::tile::MoEDispatchKernelOp kernel, OpBuilder &builder) {
  Operation *op=kernel.getOperation(); ValueRange in=kernel.getInputs(); Location loc=op->getLoc();
  Value T=in[3], S=in[4], H=in[5], z=sm120LinearThread(builder,loc,256);
  auto guarded=scf::IfOp::create(builder,loc,lessI64(builder,loc,z,mulI64(builder,loc,S,H)),false);
  { OpBuilder::InsertionGuard guard(builder); builder.setInsertionPointToStart(guarded.thenBlock());
    Value slot=arith::DivUIOp::create(builder,loc,z,H), h=arith::RemUIOp::create(builder,loc,z,H);
    Value token=arith::ExtUIOp::create(builder,loc,builder.getI64Type(),sm120LoadI32(builder,loc,in[1],slot));
    Value idx=addI64(builder,loc,mulI64(builder,loc,token,H),h);
    sm120StoreF32(builder,loc,in[2],z,sm120LoadF32(builder,loc,in[0],idx)); }
  builder.setInsertionPointAfter(guarded); (void)T; op->erase(); return success();
}

static LogicalResult materializeSm120MoECombineKernel(
    tessera::tile::MoECombineKernelOp kernel, OpBuilder &builder) {
  Operation *op=kernel.getOperation(); ValueRange in=kernel.getInputs(); Location loc=op->getLoc();
  FloatType f32=builder.getF32Type(); Value T=in[4],S=in[5],H=in[6],z=sm120LinearThread(builder,loc,256);
  auto guarded=scf::IfOp::create(builder,loc,lessI64(builder,loc,z,mulI64(builder,loc,T,H)),false);
  { OpBuilder::InsertionGuard guard(builder); builder.setInsertionPointToStart(guarded.thenBlock());
    Value tokenOut=arith::DivUIOp::create(builder,loc,z,H), h=arith::RemUIOp::create(builder,loc,z,H);
    Value zero=i64Constant(builder,loc,0),one=i64Constant(builder,loc,1);
    Value zf=arith::ConstantFloatOp::create(builder,loc,f32,APFloat(0.0f));
    auto loop=scf::ForOp::create(builder,loc,zero,S,one,ValueRange{zf});
    { OpBuilder::InsertionGuard loopGuard(builder); builder.setInsertionPointToStart(loop.getBody());
      Value slot=loop.getInductionVar();
      Value token=arith::ExtUIOp::create(builder,loc,builder.getI64Type(),sm120LoadI32(builder,loc,in[1],slot));
      Value match=arith::CmpIOp::create(builder,loc,arith::CmpIPredicate::eq,token,tokenOut);
      Value idx=addI64(builder,loc,mulI64(builder,loc,slot,H),h);
      Value term=arith::MulFOp::create(builder,loc,sm120LoadF32(builder,loc,in[0],idx),
                                      sm120LoadF32(builder,loc,in[2],slot));
      Value selected=arith::SelectOp::create(builder,loc,match,term,zf);
      scf::YieldOp::create(builder,loc,ValueRange{arith::AddFOp::create(builder,loc,loop.getRegionIterArgs()[0],selected)}); }
    builder.setInsertionPointAfter(loop); sm120StoreF32(builder,loc,in[3],z,loop.getResult(0)); }
  builder.setInsertionPointAfter(guarded); op->erase(); return success();
}

static LogicalResult materializeSm120GroupedGemmKernel(
    tessera::tile::GroupedGemmKernelOp kernel, OpBuilder &builder) {
  Operation *op=kernel.getOperation(); ValueRange in=kernel.getInputs(); Location loc=op->getLoc();
  FloatType f32=builder.getF32Type(); Value T=in[4],K=in[5],N=in[6],E=in[7];
  Value z=sm120LinearThread(builder,loc,256);
  auto guarded=scf::IfOp::create(builder,loc,lessI64(builder,loc,z,mulI64(builder,loc,T,N)),false);
  { OpBuilder::InsertionGuard guard(builder); builder.setInsertionPointToStart(guarded.thenBlock());
    Value row=arith::DivUIOp::create(builder,loc,z,N), col=arith::RemUIOp::create(builder,loc,z,N);
    Value zero=i64Constant(builder,loc,0),one=i64Constant(builder,loc,1);
    auto expertLoop=scf::ForOp::create(builder,loc,zero,E,one,ValueRange{zero});
    { OpBuilder::InsertionGuard expertGuard(builder); builder.setInsertionPointToStart(expertLoop.getBody());
      Value e=expertLoop.getInductionVar();
      Value begin=arith::ExtUIOp::create(builder,loc,builder.getI64Type(),sm120LoadI32(builder,loc,in[2],e));
      Value end=arith::ExtUIOp::create(builder,loc,builder.getI64Type(),sm120LoadI32(builder,loc,in[2],addI64(builder,loc,e,one)));
      Value ge=arith::CmpIOp::create(builder,loc,arith::CmpIPredicate::uge,row,begin);
      Value lt=lessI64(builder,loc,row,end);
      Value match=arith::AndIOp::create(builder,loc,ge,lt);
      scf::YieldOp::create(builder,loc,ValueRange{arith::SelectOp::create(builder,loc,match,e,expertLoop.getRegionIterArgs()[0])}); }
    builder.setInsertionPointAfter(expertLoop);
    Value zf=arith::ConstantFloatOp::create(builder,loc,f32,APFloat(0.0f));
    auto kLoop=scf::ForOp::create(builder,loc,zero,K,one,ValueRange{zf});
    { OpBuilder::InsertionGuard kGuard(builder); builder.setInsertionPointToStart(kLoop.getBody());
      Value k=kLoop.getInductionVar();
      Value xIdx=addI64(builder,loc,mulI64(builder,loc,row,K),k);
      Value wIdx=addI64(builder,loc,mulI64(builder,loc,
          addI64(builder,loc,mulI64(builder,loc,expertLoop.getResult(0),K),k),N),col);
      Value term=arith::MulFOp::create(builder,loc,sm120LoadF32(builder,loc,in[0],xIdx),
                                      sm120LoadF32(builder,loc,in[1],wIdx));
      scf::YieldOp::create(builder,loc,ValueRange{arith::AddFOp::create(builder,loc,kLoop.getRegionIterArgs()[0],term)}); }
    builder.setInsertionPointAfter(kLoop); sm120StoreF32(builder,loc,in[3],z,kLoop.getResult(0)); }
  builder.setInsertionPointAfter(guarded); op->erase(); return success();
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
  case Sm120InputPacking::ScalarF64:
    inputTy = builder.getF64Type();
    loadTy = inputTy;
    alignment = 8;
    columnsPerRegister = 1;
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
  if (physical->packing == Sm120InputPacking::ScalarF64) {
    coords = role.getValue() == "a"
        ? SmallVector<std::pair<Value, Value>>{{gid, tig}}
        : SmallVector<std::pair<Value, Value>>{{tig, gid}};
  } else if (role.getValue() == "a") {
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
    if (dtype != "f16" && dtype != "f64")
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
  Value rowOrigin = view.getInputs()[1];
  Value colOrigin = view.getInputs()[2];
  auto pointerType = dyn_cast<LLVM::LLVMPointerType>(base.getType());
  if (!pointerType ||
      (pointerType.getAddressSpace() != 0 && pointerType.getAddressSpace() != 1) ||
      !rowOrigin.getType().isInteger(64) ||
      !colOrigin.getType().isInteger(64)) {
    op->emitError("NVFP4 scale view requires a global-memory LLVM pointer and "
                  "i64 origins");
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
  Value row = role.getValue() == "scale_a"
      ? addI64(builder, loc, rowOrigin, scaleRow)
      : rowOrigin;
  Value col = role.getValue() == "scale_a"
      ? colOrigin
      : addI64(builder, loc, colOrigin, gid);
  Value leadingDim = i64Constant(builder, loc, memory.getLeadingDim());
  Value linear = memory.getOrder() == "row_major"
      ? addI64(builder, loc, mulI64(builder, loc, row, leadingDim), col)
      : addI64(builder, loc, mulI64(builder, loc, col, leadingDim), row);
  Value ptr = LLVM::GEPOp::create(builder, loc, base.getType(),
                                  builder.getI8Type(), base,
                                  ValueRange{linear});
  Value loaded = LLVM::LoadOp::create(builder, loc, builder.getI32Type(), ptr,
                                      /*alignment=*/1);
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
          name == "tile.matmul_kernel" || name == "tile.softmax_kernel" ||
          name == "tile.reduce_kernel" || name == "tile.attention_kernel" ||
          name == "tile.attention_backward_kernel" ||
          name == "tile.paged_kv_read_kernel" ||
          name == "tile.replay_ssm_decode_kernel" ||
          name == "tile.replay_ssm_flush_kernel" ||
          name == "tile.moe_dispatch_kernel" ||
          name == "tile.moe_combine_kernel" ||
          name == "tile.grouped_gemm_kernel" ||
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

      if (isTileOp(op, "tile.softmax_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120SoftmaxKernel(
                cast<tessera::tile::SoftmaxKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError("tile.softmax_kernel currently requires sm_120");
          signalPassFailure();
          return;
        }
        continue;
      }

      if (isTileOp(op, "tile.reduce_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120ReduceKernel(
                cast<tessera::tile::ReduceKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError("tile.reduce_kernel currently requires sm_120");
          signalPassFailure();
          return;
        }
        continue;
      }

      if (isTileOp(op, "tile.attention_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120AttentionKernel(
                cast<tessera::tile::AttentionKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError("tile.attention_kernel currently requires sm_120");
          signalPassFailure();
          return;
        }
        continue;
      }

      if (isTileOp(op, "tile.attention_backward_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120AttentionBackwardKernel(
                cast<tessera::tile::AttentionBackwardKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError(
                "tile.attention_backward_kernel currently requires sm_120");
          signalPassFailure();
          return;
        }
        continue;
      }

      if (isTileOp(op, "tile.paged_kv_read_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120PagedKVReadKernel(
                cast<tessera::tile::PagedKVReadKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError("tile.paged_kv_read_kernel currently requires sm_120");
          signalPassFailure();
          return;
        }
        continue;
      }

      if (isTileOp(op, "tile.replay_ssm_decode_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120ReplaySSMDecodeKernel(
                cast<tessera::tile::ReplaySSMDecodeKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError("tile.replay_ssm_decode_kernel currently requires sm_120");
          signalPassFailure(); return;
        }
        continue;
      }
      if (isTileOp(op, "tile.replay_ssm_flush_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120ReplaySSMFlushKernel(
                cast<tessera::tile::ReplaySSMFlushKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError("tile.replay_ssm_flush_kernel currently requires sm_120");
          signalPassFailure(); return;
        }
        continue;
      }
      if (isTileOp(op, "tile.moe_dispatch_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120MoEDispatchKernel(
                cast<tessera::tile::MoEDispatchKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError("tile.moe_dispatch_kernel currently requires sm_120");
          signalPassFailure(); return;
        }
        continue;
      }
      if (isTileOp(op, "tile.moe_combine_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120MoECombineKernel(
                cast<tessera::tile::MoECombineKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError("tile.moe_combine_kernel currently requires sm_120");
          signalPassFailure(); return;
        }
        continue;
      }
      if (isTileOp(op, "tile.grouped_gemm_kernel")) {
        if (smVersion < kConsumerBlackwellSM ||
            failed(materializeSm120GroupedGemmKernel(
                cast<tessera::tile::GroupedGemmKernelOp>(op), builder))) {
          if (smVersion < kConsumerBlackwellSM)
            op->emitError("tile.grouped_gemm_kernel currently requires sm_120");
          signalPassFailure(); return;
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
  if (opName == "tessera_nvidia.mx_block_scale_mma")
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
// f16/bf16, tf32, fp8, and int8 use four A and two B registers for the
// canonical m16n8 shapes. FP64 DMMA uses the distinct m8n8k4 contract with one
// f64 A, one f64 B, and two f64 accumulator/result elements.
static bool tryLowerMmaSyncToNVVM(Operation *op, OpBuilder &builder) {
  if (op->getName().getStringRef() ==
      "tessera_nvidia.mx_block_scale_mma") {
    auto shape = op->getAttrOfType<StringAttr>("shape");
    auto dtypeAB = op->getAttrOfType<StringAttr>("dtype_ab");
    auto dtypeC = op->getAttrOfType<StringAttr>("dtype_c");
    auto scaleDtype = op->getAttrOfType<StringAttr>("scale_dtype");
    auto scaleVector = op->getAttrOfType<StringAttr>("scale_vector");
    auto blockScaled = op->getAttrOfType<BoolAttr>("block_scaled");
    if (!shape || !dtypeAB || !dtypeC || dtypeC.getValue() != "f32" ||
        !scaleDtype || scaleDtype.getValue() != "ue8m0" || !scaleVector ||
        !blockScaled || !blockScaled.getValue() ||
        op->getNumOperands() != 12 || op->getNumResults() != 1)
      return false;
    StringRef dtype = dtypeAB.getValue();
    bool isFP6 = dtype == "e2m3" || dtype == "e3m2";
    bool isMXFP4 = dtype == "e2m1";
    if (!((isFP6 && shape.getValue() == "m16n8k32" &&
           scaleVector.getValue() == "1X") ||
          (isMXFP4 && shape.getValue() == "m16n8k64" &&
           scaleVector.getValue() == "2X")))
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
    std::string assembly = "mma.sync.aligned." + shape.getValue().str() +
        ".row.col.kind::" + (isMXFP4 ? std::string("mxf4")
                                     : std::string("mxf8f6f4")) +
        ".block_scale.scale_vec::" + scaleVector.getValue().str() +
        ".f32." + dtype.str() + "." + dtype.str() +
        ".f32.ue8m0 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, "
        "{$10,$11,$12,$13}, {$14}, {0, 0}, {$15}, {0, 0};";
    builder.setInsertionPoint(op);
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
  bool isF64 = dtype == "f64";
  bool isTF32 = dtype == "tf32";
  bool isFP8 = dtype == "e4m3" || dtype == "e5m2";
  bool isS8 = dtype == "s8" || dtype == "int8";
  if (!((shape.getValue() == "m8n8k4" && isF64) ||
        (shape.getValue() == "m16n8k16" && (isF16 || isBF16)) ||
        (shape.getValue() == "m16n8k8" && isTF32) ||
        (shape.getValue() == "m16n8k32" && (isFP8 || isS8))))
    return false;
  if (op->getNumResults() != 1)
    return false;
  auto structTy = dyn_cast<LLVM::LLVMStructType>(op->getResult(0).getType());
  if (!structTy)
    return false;
  ValueRange operands = op->getOperands();
  Type fragTy = isF64
      ? Type(Float64Type::get(op->getContext()))
      : isF16
      ? Type(VectorType::get({2}, Float16Type::get(op->getContext())))
      : Type(IntegerType::get(op->getContext(), 32));
  unsigned aCount = isF64 ? 1 : 4;
  unsigned bCount = isF64 ? 1 : 2;
  bool isF32Accumulator = dtypeC && dtypeC.getValue() == "f32";
  bool isF16Accumulator = dtypeC && dtypeC.getValue() == "f16";
  bool isS32Accumulator = dtypeC &&
      (dtypeC.getValue() == "s32" || dtypeC.getValue() == "int32");
  bool isF64Accumulator = dtypeC && dtypeC.getValue() == "f64";
  if ((isF64 && !isF64Accumulator) ||
      (!isF64 && !isS8 && !isF32Accumulator &&
       !(isF16 && isF16Accumulator)) ||
      (isS8 && !isS32Accumulator))
    return false;
  unsigned cCount = (isF16Accumulator || isF64Accumulator) ? 2 : 4;
  if (operands.size() != aCount + bCount + cCount)
    return false;
  for (Value v : operands.take_front(aCount + bCount))
    if (v.getType() != fragTy)
      return false;
  Type cType = isF32Accumulator
      ? Type(Float32Type::get(op->getContext()))
      : (isS32Accumulator ? Type(IntegerType::get(op->getContext(), 32))
         : (isF64Accumulator ? Type(Float64Type::get(op->getContext()))
                             : Type(fragTy)));
  for (Value v : operands.drop_front(aCount + bCount))
    if (v.getType() != cType)
      return false;
  ArrayRef<Type> resultBody = structTy.getBody();
  if (resultBody.size() != cCount ||
      llvm::any_of(resultBody, [&](Type type) { return type != cType; }))
    return false;

  SmallVector<Value> a(operands.begin(), operands.begin() + aCount);
  SmallVector<Value> b(operands.begin() + aCount,
                       operands.begin() + aCount + bCount);
  SmallVector<Value> c(operands.begin() + aCount + bCount, operands.end());
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
  if (isF64)
    inputPtxType = NVVM::MMATypes::f64;
  else if (isBF16)
    inputPtxType = NVVM::MMATypes::bf16;
  else if (isTF32)
    inputPtxType = NVVM::MMATypes::tf32;
  else if (dtype == "e4m3")
    inputPtxType = NVVM::MMATypes::e4m3;
  else if (dtype == "e5m2")
    inputPtxType = NVVM::MMATypes::e5m2;
  else if (isS8)
    inputPtxType = NVVM::MMATypes::s8;
  int64_t m = isF64 ? 8 : 16;
  int64_t n = 8;
  int64_t k = isF64 ? 4 : shape.getValue() == "m16n8k8" ? 8 :
              (shape.getValue() == "m16n8k32" ? 32 : 16);
  std::optional<NVVM::MMAIntOverflow> intOverflow = std::nullopt;
  if (isS8)
    intOverflow = NVVM::MMAIntOverflow::wrapped;
  auto mma = builder.create<NVVM::MmaOp>(
      op->getLoc(), structTy, ValueRange(a), ValueRange(b), ValueRange(c),
      ArrayRef<int64_t>{m, n, k}, /*b1Op=*/std::nullopt,
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

void buildTesseraConsumerBlackwellBackendPipeline(OpPassManager &pm) {
  pm.addPass(createLowerTileToNVIDIAPass(kConsumerBlackwellSM));
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
  PassPipelineRegistration<> sm90Pipeline(
      "tessera-lower-to-nvidia-sm90",
      "Lower Tessera Tile IR to exact SM90 WGMMA/TMA NVVM contracts",
      [](OpPassManager &pm) { buildTesseraHopperBackendPipeline(pm); });
  PassPipelineRegistration<> sm100Pipeline(
      "tessera-lower-to-nvidia-sm100",
      "Lower Tessera Tile IR to exact SM100 TCGEN05/TMEM NVVM contracts",
      [](OpPassManager &pm) { buildTesseraBlackwellBackendPipeline(pm); });
  PassPipelineRegistration<> sm120Pipeline(
      "tessera-lower-to-nvidia-sm120",
      "Lower Tessera Tile IR to exact SM120 warp-level MMA NVVM contracts",
      [](OpPassManager &pm) {
        buildTesseraConsumerBlackwellBackendPipeline(pm);
      });
}

void registerTesseraNVIDIABackendDialects(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                  NVVM::NVVMDialect, tessera::nvidia::TesseraNVIDIADialect,
                  tessera::tile::TesseraTileDialect>();
}

} // namespace tessera
