
//===- SpectralOps.cpp ----------------------------------------*- C++ -*-===//
#include "tessera/Spectral/SpectralDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
namespace tessera { namespace spectral {

#define GET_OP_CLASSES
#include "SpectralOps.cpp.inc"

}} // namespace tessera::spectral
