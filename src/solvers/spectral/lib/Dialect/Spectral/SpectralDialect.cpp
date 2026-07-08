//===- SpectralDialect.cpp ------------------------------------*- C++ -*-===//
//
// Registration glue for the tessera_spectral dialect: its `plan` type and
// its fft/ifft/plan/twiddle/conv_fft ops.  The generated class definitions
// are fully qualified, so they are included at global scope; only
// initialize() lives inside the dialect namespace.
//
//===----------------------------------------------------------------------===//
#include "tessera/Spectral/SpectralDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

// Type definitions (parser/printer/storage for !tessera_spectral.plan).
#define GET_TYPEDEF_CLASSES
#include "SpectralTypes.cpp.inc"

// Op class declarations — needed by addOperations<> below.
#define GET_OP_CLASSES
#include "SpectralOps.h.inc"

namespace tessera {
namespace spectral {

void SpectralDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "SpectralTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "SpectralOps.cpp.inc"
      >();
}

} // namespace spectral
} // namespace tessera

// Dialect definition (ctor/dtor + default type printer/parser dispatch).
#include "SpectralDialect.cpp.inc"
