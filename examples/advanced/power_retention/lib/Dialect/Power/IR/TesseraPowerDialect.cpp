// The dialect class is generated (`-gen-dialect-decls`); this file only supplies
// `initialize()`. It used to hand-write a second `PowerDialect` beside the
// generated declaration and could not compile in any clean tree
// (2026-09-17).
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "tessera/power/TesseraPowerDialect.h.inc"
#define GET_OP_CLASSES
#include "tessera/power/TesseraPowerOps.h.inc"

#include "tessera/power/TesseraPowerDialect.cpp.inc"

void tessera::power::PowerDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tessera/power/TesseraPowerOps.cpp.inc"
      >();
}
