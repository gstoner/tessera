#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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

} // namespace queue
} // namespace tessera

#define GET_OP_CLASSES
#include "QueueOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "QueueTypes.cpp.inc"
#include "QueueDialect.cpp.inc"
