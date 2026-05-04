#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"

#include "TesseraNVIDIADialect.h.inc"

using namespace mlir;

#include "TesseraNVIDIADialect.cpp.inc"

void tessera::nvidia::TesseraNVIDIADialect::initialize() {
  allowUnknownOperations();
}
