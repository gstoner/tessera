
//===- SpectralDialect.cpp ------------------------------------*- C++ -*-===//
#include "tessera/Spectral/SpectralDialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

namespace tessera { namespace spectral {
#include "SpectralDialect.cpp.inc"
}} // namespace tessera::spectral
