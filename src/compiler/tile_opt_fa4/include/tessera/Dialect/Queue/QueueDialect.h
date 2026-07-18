//===- QueueDialect.h - FA-4 tile-queue Tile IR dialect --------*- C++ -*-===//
//
// Sprint V8 (2026-05-22): public C++ header for the FA-4 tile-queue
// dialect.  Mirrors the V7 pattern used for the Attn dialect — without
// public registration, `tessera-opt` can parse the queue op names but
// not the queue type names, because the longest-prefix dialect lookup
// can't reach `tessera.queue` if no codepath has loaded it into the
// context.
//
// The dialect is defined in `Queue.td` (tablegen) and the op verifiers
// live in `lib/Dialect/Queue/QueueVerifiers.cpp`.  This header exposes
// the generated class declarations + a public
// `registerQueueDialect(DialectRegistry&)` for `tessera-opt` and any
// future translation driver.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_DIALECT_QUEUE_DIALECT_H
#define TESSERA_DIALECT_QUEUE_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// Generated dialect declarations (cppNamespace = ::tessera::queue).
#include "QueueDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "QueueTypes.h.inc"
#define GET_OP_CLASSES
#include "QueueOps.h.inc"

namespace tessera {
namespace queue {

/// Insert the FA-4 tile-queue dialect into a DialectRegistry. Call
/// from `tessera-opt` and any other tool that needs to parse/verify
/// the `tessera.queue.*` ops + types.
///
/// Sprint V8 (2026-05-22) — public registration entry; mirrors
/// `tessera::attn::registerAttnDialect()`. MLIR 23 rejects eager
/// construction of the legacy dotted `tessera.queue` namespace, so callers
/// that need queue IR must load the registered dialect explicitly.
void registerQueueDialect(::mlir::DialectRegistry &registry);

} // namespace queue
} // namespace tessera

#endif // TESSERA_DIALECT_QUEUE_DIALECT_H
