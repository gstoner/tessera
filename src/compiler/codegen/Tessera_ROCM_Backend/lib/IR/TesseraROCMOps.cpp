#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
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
