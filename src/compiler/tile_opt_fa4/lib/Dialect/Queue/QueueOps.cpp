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

// Sprint V8 (2026-05-22): the Queue dialect anchors its eager-load
// extension on the parent `tessera` (Graph IR) dialect — same pattern
// V7b used for Attn.  Including TesseraOps.h pulls in the
// TesseraDialect class declaration that the extension lambda needs.
#include "Tessera/IR/TesseraOps.h"

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
// The DialectExtension fires once per MLIRContext, immediately after
// the parent `tessera` Graph IR dialect attaches — making the
// dotted-prefix parse path work for `tessera.queue.*` ops and types.
void registerQueueDialect(::mlir::DialectRegistry &registry) {
  registry.insert<TesseraQueueDialect>();
  registry.addExtension(
      +[](::mlir::MLIRContext *ctx, ::tessera::TesseraDialect *) {
        ctx->getOrLoadDialect<TesseraQueueDialect>();
      });
}

} // namespace queue
} // namespace tessera

#define GET_OP_CLASSES
#include "QueueOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "QueueTypes.cpp.inc"
#include "QueueDialect.cpp.inc"
