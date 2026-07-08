//===- SpectralOps.cpp ----------------------------------------*- C++ -*-===//
//
// Op method definitions for the tessera_spectral dialect.  The generated
// definitions are fully qualified (::tessera::spectral::FFTOp::...), so they
// live at global scope, not inside a namespace block.
//
//===----------------------------------------------------------------------===//
#include "tessera/Spectral/SpectralDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "SpectralOps.h.inc"

#define GET_OP_CLASSES
#include "SpectralOps.cpp.inc"
