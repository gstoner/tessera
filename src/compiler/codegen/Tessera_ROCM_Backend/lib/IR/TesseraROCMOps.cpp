#include "Tessera/IR/NVFP4IngestContract.h"
#include <limits>
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TesseraROCMDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "TesseraROCMTypes.h.inc"
#define GET_OP_CLASSES
#include "TesseraROCMOps.h.inc"

using namespace mlir::tessera_rocm;

#include "TesseraROCMDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TesseraROCMTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TesseraROCMOps.cpp.inc"

void TesseraROCMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TesseraROCMTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "TesseraROCMOps.cpp.inc"
  >();
}

mlir::LogicalResult SWMMACOp::verify() {
  auto a = mlir::dyn_cast<mlir::VectorType>(getA().getType());
  auto b = mlir::dyn_cast<mlir::VectorType>(getB().getType());
  auto c = mlir::dyn_cast<mlir::VectorType>(getAcc().getType());
  if (getArch() != "gfx1201" || !a || !b || !c ||
      a.getRank() != 1 || b.getRank() != 1 || c.getRank() != 1 ||
      a.getNumElements() != 8 || b.getNumElements() != 16 || c.getNumElements() != 8 ||
      !(a.getElementType().isF16() || a.getElementType().isBF16() ||
        a.getElementType().isInteger(8) || mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(a.getElementType())) ||
      (b.getElementType() != a.getElementType() &&
       !(mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(a.getElementType()) &&
         mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(b.getElementType()))) ||
      !(a.getElementType().isInteger(8) ? c.getElementType().isInteger(32) :
        (c.getElementType().isF32() || ((a.getElementType().isF16() || a.getElementType().isBF16()) && c.getElementType() == a.getElementType()))) ||
      getRes().getType() != c)
    return emitOpError("requires gfx1201 wave32 A[8] B[16] matching f16/bf16/FP8 or signed i8, C/result[8] matching supported accumulation and OPSEL=0 indices");
  if (!a.getElementType().isInteger(8) && (!getASigned() || !getBSigned()))
    return emitOpError("integer signedness flags require i8 operands");
  if ((getIntegerBits() != 4 && getIntegerBits() != 8) ||
      (getIntegerBits() != 8 && !a.getElementType().isInteger(8)))
    return emitOpError("sparse integer width requires i8 operands and 4 or 8 bits");
  return mlir::success();
}

// ROCM-SPLIT-K-1: the slice count and the reduction order are semantic keys
// (Decision #21a) and travel as a pair.
mlir::LogicalResult WMMAGemmOp::verify() {
  if (getKBlocks() < 1)
    return emitOpError("ROCM_WMMA_GEMM_SPLIT_K_BAD_CONTRACT: k_blocks must be positive");
  if (getSplitK() < 1)
    return emitOpError("ROCM_WMMA_GEMM_SPLIT_K_BAD_CONTRACT: split_k must be >= 1");
  if (getSplitK() == 1 && !getSplitKReduction().empty())
    return emitOpError("ROCM_WMMA_GEMM_SPLIT_K_BAD_CONTRACT: split_k_reduction "
                       "requires split_k > 1");
  if (getSplitK() > 1 && getSplitKReduction() != "ordered")
    return emitOpError("ROCM_WMMA_GEMM_SPLIT_K_BAD_CONTRACT: split_k > 1 "
                       "requires split_k_reduction = \"ordered\"");
  if (getSplitK() > 1 && (!getProblemK() || *getProblemK() <= 0))
    return emitOpError("ROCM_WMMA_GEMM_SPLIT_K_BAD_CONTRACT: split-K requires positive problem_k");
  return mlir::success();
}


mlir::LogicalResult NVFP4RequantizeOp::verify() {
  if (auto policy = getOperation()->getAttrOfType<mlir::DictionaryAttr>("numeric_policy"))
    if (failed(mlir::tessera_contract::verifyNVFP4Policy(getOperation(),policy)))
      return mlir::failure();
  if (getArch() != "gfx1201" || getN() <= 0 || getK() <= 0 || getK() % 32 ||
      getExecutionMode() != "explicit_scale_requantization" ||
      getSourceLayout() != "e2m1_row_k_e4m3_k16_projection_global" ||
      getDestinationLayout() != "e2m1_row_k_e8m0_k32_group_n")
    return emitOpError("requires gfx1201 positive N/K32 and explicit named NVFP4-to-MXFP4 scale requantization layouts");
  if (getN() > std::numeric_limits<int64_t>::max() / (getK() / 2))
    return emitOpError("requires representable packed buffer extents");
  auto offsets = getRowOffsets();
  if (offsets.size() < 2 || mlir::cast<mlir::IntegerAttr>(offsets[0]).getInt() != 0 ||
      mlir::cast<mlir::IntegerAttr>(offsets[offsets.size() - 1]).getInt() != getN())
    return emitOpError("requires complete projection row offsets from zero through N");
  int64_t previous = -1;
  for (auto offset : offsets) {
    int64_t current = mlir::cast<mlir::IntegerAttr>(offset).getInt();
    if (current <= previous)
      return emitOpError("requires strictly increasing projection row offsets");
    previous = current;
  }
  return mlir::success();
}

mlir::LogicalResult MXFP4FoldedStorageOp::verify() {
  if (getArch() != "gfx1201")
    return emitOpError("MXFP4 storage bridge requires gfx1201");
  if (getScheduleHash().size() != 64 ||
      !llvm::all_of(getScheduleHash(), [](char c) { return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')); }))
    return emitOpError("MXFP4 storage bridge requires a Schedule digest");
  return mlir::tessera_contract::verifyMXFP4StorageExtents(
      getOperation(), getN(), getK(), getStorageContractAttr());
}

#include "Tessera/IR/StructuredReductionContract.h"
mlir::LogicalResult StructuredReductionOp::verify() {
  return tessera::verifyStructuredReductionCarrier(getOperation());
}
