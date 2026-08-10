//===- TesseraX86Dialect.h — public handle for the tessera_x86 dialect ----===//
//
// One include for anything that only needs to REGISTER the dialect (drivers,
// pass pipelines). Ops and types come from the generated headers next to it.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_X86_IR_TESSERAX86DIALECT_H
#define TESSERA_X86_IR_TESSERAX86DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"

#include "TesseraX86Dialect.h.inc"

namespace mlir {
namespace tessera_x86 {

/// Register the hardware-free x86 Target IR dialect (Decision #19 / W0.10).
void registerTesseraX86Dialect(::mlir::DialectRegistry &registry);

}  // namespace tessera_x86
}  // namespace mlir

#endif  // TESSERA_X86_IR_TESSERAX86DIALECT_H
