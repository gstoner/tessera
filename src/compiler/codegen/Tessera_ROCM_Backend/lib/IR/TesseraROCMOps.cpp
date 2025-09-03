#include "TesseraROCM/IR/TesseraROCMDialect.h.inc"
#include "TesseraROCM/IR/TesseraROCMOps.h.inc"
using namespace mlir;
void TesseraROCMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TesseraROCM/IR/TesseraROCMOps.cpp.inc"
  >();
}
