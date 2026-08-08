//===- ScheduleDialect.h — production Schedule IR dialect ---------------===//

#ifndef TESSERA_PROGRAMMING_MODEL_SCHEDULEDIALECT_H
#define TESSERA_PROGRAMMING_MODEL_SCHEDULEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ScheduleDialect.h.inc"

#define GET_OP_CLASSES
#include "ScheduleMeshPipelineOps.h.inc"

namespace tessera {
namespace schedule {

inline void registerScheduleDialect(::mlir::DialectRegistry &registry) {
  registry.insert<ScheduleDialect>();
}

} // namespace schedule
} // namespace tessera

#endif // TESSERA_PROGRAMMING_MODEL_SCHEDULEDIALECT_H
