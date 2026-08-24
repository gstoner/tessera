#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringSet.h"

#include "TesseraNVIDIADialect.h.inc"

#define GET_OP_CLASSES
#include "TesseraNVIDIAOps.h.inc"

using namespace mlir;
using namespace tessera::nvidia;

#include "TesseraNVIDIADialect.cpp.inc"

#define GET_OP_CLASSES
#include "TesseraNVIDIAOps.cpp.inc"

void TesseraNVIDIADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TesseraNVIDIAOps.cpp.inc"
      >();
}

LogicalResult BlockCoordinateOp::verify() {
  auto arch = getOperation()->getAttrOfType<StringAttr>("arch");
  auto tileM = getOperation()->getAttrOfType<IntegerAttr>("tile_m");
  auto tileN = getOperation()->getAttrOfType<IntegerAttr>("tile_n");
  auto gridOrder = getOperation()->getAttrOfType<StringAttr>("grid_order");
  if (!arch || arch.getValue() != "sm_120")
    return emitOpError("requires arch=sm_120");
  if (!tileM || tileM.getInt() != 16)
    return emitOpError("requires tile_m=16");
  if (!tileN || tileN.getInt() != 8)
    return emitOpError("requires tile_n=8");
  if (!gridOrder || gridOrder.getValue() != "column_major_xy")
    return emitOpError("requires grid_order=column_major_xy");
  return success();
}

LogicalResult MacroCTAMatmulOp::verify() {
  auto exactString = [&](StringRef name, StringRef expected) {
    auto attr = getOperation()->getAttrOfType<StringAttr>(name);
    return attr && attr.getValue() == expected;
  };
  auto exactInteger = [&](StringRef name, int64_t expected) {
    auto attr = getOperation()->getAttrOfType<IntegerAttr>(name);
    return attr && attr.getInt() == expected;
  };
  if (!exactString("arch", "sm_120"))
    return emitOpError("requires arch=sm_120");
  if (!exactInteger("cta_m", 32) || !exactInteger("cta_n", 32))
    return emitOpError("requires a 32x32 CTA tile");
  if (!exactInteger("tile_m", 16) || !exactInteger("tile_n", 8) ||
      !exactInteger("tile_k", 16))
    return emitOpError("requires the m16n8k16 instruction tile");
  if (getLeadingDimensions().size() != 0 &&
      getLeadingDimensions().size() != 3)
    return emitOpError("requires either no physical strides or exactly lda, ldb, ldd");
  bool dynamicStrides = getLeadingDimensions().size() == 3;
  StringRef expectedStaging = dynamicStrides
                                  ? "masked_scalar_shared_ab_16bit"
                                  : "cp_async_shared_ab_16bit";
  int64_t expectedStages = dynamicStrides ? 1 : 2;
  StringRef expectedCompletion = dynamicStrides
                                     ? "cta_barrier"
                                     : "wait_group_0_cta_barrier";
  if (!exactInteger("warps", 4) ||
      !exactString("warp_ownership", "quadrant_2x2_two_n_tiles") ||
      !(exactString("storage", "f16") || exactString("storage", "bf16")) ||
      !exactString("accum", "f32") ||
      !exactString("staging", expectedStaging) ||
      !exactInteger("stages", expectedStages) ||
      !exactString("completion", expectedCompletion) ||
      !exactString("bounds", "zero_fill_mnk_tail") ||
      !exactString("grid_order", "column_major_xy"))
    return emitOpError("requires the canonical f16/bf16-to-f32 four-warp shared-panel ownership contract");
  auto digest = getOperation()->getAttrOfType<StringAttr>("tessera.schedule_hash");
  if (!digest || digest.getValue().size() != 64)
    return emitOpError("requires a 64-character tessera.schedule_hash");
  return success();
}

LogicalResult CudaMathKernelOp::verify() {
  if (getInputs().size() != 5)
    return emitOpError("expects A, B, C, destination, and N operands");
  for (Value pointer : getInputs().take_front(4))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("A/B/C/destination operands must be !llvm.ptr");
  if (!getInputs().back().getType().isInteger(64))
    return emitOpError("N must be i64");

  StringRef kind = getKind();
  static const llvm::StringSet<> kScalarI32 = {
      "abs_i32", "min_i32", "max_i32", "brev_u32", "byte_perm_u32",
      "clz_u32", "ffs_u32", "popc_u32", "funnelshift_l_wrap_u32",
      "dp2a_lo_s32", "dp4a_s32"};
  static const llvm::StringSet<> kPacked = {
      "vadd2_u16x2", "vadd4_u8x4", "vadd4_s8x4",
      "vaddss4_s8x4", "vsub4_u8x4", "vsub4_s8x4",
      "vsubss4_s8x4", "vabsdiff4_u8x4",
      "vcmpeq4_u8x4_mask", "vcmpeq4_u8x4_pred"};
  bool numericCast = kind.starts_with("cvt_f32_i32_");
  bool bitcastF32I32 = kind == "bitcast_f32_i32";
  bool bitcastI32F32 = kind == "bitcast_i32_f32";
  if (!kScalarI32.contains(kind) && !kPacked.contains(kind) &&
      !numericCast && !bitcastF32I32 && !bitcastI32F32)
    return emitOpError("unsupported typed CUDA Math kind");

  if (numericCast) {
    StringRef mode = kind.drop_front(StringRef("cvt_f32_i32_").size());
    if (getInputStorage() != "f32" || getOutputStorage() != "i32" ||
        getRounding() != mode ||
        (mode != "rn" && mode != "rd" && mode != "ru" && mode != "rz"))
      return emitOpError(
          "numeric casts require f32->i32 and matching RN/RD/RU/RZ");
  } else if (bitcastF32I32) {
    if (getInputStorage() != "f32" || getOutputStorage() != "i32")
      return emitOpError("float_as_int requires equal-width f32->i32 storage");
  } else if (bitcastI32F32) {
    if (getInputStorage() != "i32" || getOutputStorage() != "f32")
      return emitOpError("int_as_float requires equal-width i32->f32 storage");
  } else if (getInputStorage() != "i32" ||
             getOutputStorage() != "i32" || getRounding() != "none") {
    return emitOpError(
        "integer and packed CUDA Math requires i32 bit containers and "
        "rounding=none");
  }

  if (kPacked.contains(kind)) {
    uint64_t expectedLane = kind.contains("2_") ? 16 : 8;
    if (getLaneWidth() != expectedLane)
      return emitOpError("packed CUDA Math lane_width disagrees with kind");
    bool signedKind = kind.contains("_s8x4");
    if (getSignedness() != (signedKind ? "signed" : "unsigned"))
      return emitOpError("packed CUDA Math signedness disagrees with kind");
    bool expectedSaturation = kind.contains("ss4");
    if (getSaturation() != expectedSaturation)
      return emitOpError("packed CUDA Math saturation disagrees with kind");
    StringRef expectedPredicate =
        kind.ends_with("_mask") ? "lane_mask"
        : kind.ends_with("_pred") ? "bitset"
                                  : "none";
    if (getPredicateForm() != expectedPredicate)
      return emitOpError("packed comparison predicate_form disagrees with kind");
  } else if (getLaneWidth() != 0 || getSignedness() != "scalar" ||
             getSaturation() || getPredicateForm() != "none") {
    return emitOpError(
        "scalar CUDA Math requires lane_width=0, signedness=scalar, "
        "saturation=false, and predicate_form=none");
  }
  return success();
}
