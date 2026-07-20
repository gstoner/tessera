//===- AttnDialect.h - FA-4 attention Tile IR dialect ----------*- C++ -*-===//
//
// Sprint V7 (2026-05-22): public C++ header for the FA-4 attention
// dialect.  Mirrors the canonical pattern used by
// `Tessera/Target/Apple/TesseraAppleDialect.h` (see Apple backend).
//
// The dialect itself is defined in `Attn.td` (tablegen) and the op
// verifiers live in `lib/Dialect/Attn/AttnOps.cpp`.  This header
// exposes the generated class declarations + a public
// `registerAttnDialect(DialectRegistry&)` so external tools
// (`tessera-opt`, future translation drivers) can insert the dialect
// into their `mlir::DialectRegistry`.
//
// Prior to V7 the dialect class was only reachable from inside
// `AttnOps.cpp` — `tessera-opt` could not load it, which is why the
// two `tessera_attn.scaled_dot_product` lit fixtures
// (`flash_attn_full.mlir`, `tile_ir_lowering.mlir`) carried
// `// XFAIL: *` and the V6c verifier could not be exercised
// end-to-end.  V7 closes that gap.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_DIALECT_ATTN_DIALECT_H
#define TESSERA_DIALECT_ATTN_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated dialect declarations (cppNamespace = ::tessera::attn).
#include "AttnDialect.h.inc"

#define GET_OP_CLASSES
#include "AttnOps.h.inc"

namespace tessera {
namespace attn {

/// Insert the FA-4 attention dialect into a DialectRegistry. Call from
/// `tessera-opt` and any other tool that needs to parse/verify the
/// `tessera_attn.*` ops.
///
/// Sprint V7 (2026-05-22) — public registration entry; mirrors
/// `tessera::apple::registerAppleDialect()`.
void registerAttnDialect(::mlir::DialectRegistry &registry);

} // namespace attn
} // namespace tessera

#endif // TESSERA_DIALECT_ATTN_DIALECT_H
