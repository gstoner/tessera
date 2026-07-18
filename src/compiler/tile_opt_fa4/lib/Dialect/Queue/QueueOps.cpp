#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/TypeSwitch.h"

#include "QueueDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "QueueTypes.h.inc"
#define GET_OP_CLASSES
#include "QueueOps.h.inc"

namespace tessera {
namespace queue {

void TesseraQueueDialect::initialize() {
  addTypes<TileQueueType, TokenType>();
  addOperations<
#define GET_OP_LIST
#include "QueueOps.cpp.inc"
      >();
}

// Sprint V8 (2026-05-22) — public registration entry; mirrors
// tessera::attn::registerAttnDialect() from V7+V7b.  Without this,
// tessera-opt cannot parse the `tessera.queue.tile_queue` /
// `tessera.queue.token` types in standalone IR fixtures, because the
// longest-prefix dialect lookup needs `tessera.queue` to be
// pre-loaded in the context.
//
// MLIR 23 rejects the legacy dotted namespace at dialect construction time.
// Register it without eagerly loading it alongside the parent `tessera`
// dialect, so ordinary Graph IR parsing does not abort.
void registerQueueDialect(::mlir::DialectRegistry &registry) {
  registry.insert<TesseraQueueDialect>();
}

} // namespace queue
} // namespace tessera

#define GET_OP_CLASSES
#include "QueueOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "QueueTypes.cpp.inc"
#include "QueueDialect.cpp.inc"
