
//===- CPXTargetIRTypes.cpp - NVFP4/NVFP6 type storage and printing --------===//
//
// The bulk of NVFP4Type and NVFP6Type is ODS-generated (see NVRubinCPXTypes.cpp.inc
// pulled in from NVRubinCPX.cpp).  This translation unit adds any hand-written
// type printing / parsing overrides and storage-key helpers that the tablegen
// output can't generate.
//
// Currently: NVFP4 and NVFP6 are parameter-less singleton types, so no extra
// storage is needed.  The file is kept as a dedicated home for future per-type
// custom logic (e.g., NVFP8 variants with exponent-bias parameters).
//
//===-----------------------------------------------------------------------===//

#include "tessera/Target/NVRubinCPX/NVRubinCPX.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace tessera::target;

//===----------------------------------------------------------------------===//
// NVFP4Type — custom printing (supplement to ODS-generated)
//===----------------------------------------------------------------------===//

// ODS emits the default `!tessera.target.cpx.nvfp4` mnemonic printer.
// Override only if a different textual form is needed.
//
// void NVFP4Type::print(AsmPrinter &printer) const { printer << "nvfp4"; }
// Type NVFP4Type::parse(AsmParser &parser) {
//   return NVFP4Type::get(parser.getContext());
// }

//===----------------------------------------------------------------------===//
// NVFP6Type — custom printing (supplement to ODS-generated)
//===----------------------------------------------------------------------===//

// void NVFP6Type::print(AsmPrinter &printer) const { printer << "nvfp6"; }
// Type NVFP6Type::parse(AsmParser &parser) {
//   return NVFP6Type::get(parser.getContext());
// }

//===----------------------------------------------------------------------===//
// Type-system helpers
//===----------------------------------------------------------------------===//

/// Returns true if \p type is one of the CPX low-precision scalar types
/// (NVFP4 or NVFP6) that the CPX tensor units natively accelerate.
bool isCPXAcceleratedType(Type type) {
  return type.isa<NVFP4Type, NVFP6Type>();
}

/// Returns the bit-width of a CPX low-precision type.
unsigned getCPXTypeBitWidth(Type type) {
  if (type.isa<NVFP4Type>()) return 4;
  if (type.isa<NVFP6Type>()) return 6;
  return 0; // not a CPX accelerated type
}
