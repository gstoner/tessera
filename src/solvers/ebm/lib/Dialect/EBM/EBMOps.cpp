//===- EBMOps.cpp --------------------------------------------*- C++ -*-===//
#include "tessera/EBM/EBMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace tessera::ebm;

#define GET_OP_CLASSES
#include "EBMOps.cpp.inc"

// The temperature is a semantic key (Decision #21a): it selects which
// distribution the chain samples, so it may be neither defaulted nor given
// twice. Exactly one of the constant attribute and the runtime operand.
LogicalResult LangevinStepOp::verify() {
  const bool hasAttr = getTemperature().has_value();
  const bool hasValue = getTemperatureValue() != nullptr;
  if (hasAttr == hasValue)
    return emitOpError(hasAttr
        ? "temperature is given twice: as the `temperature` attribute and as the "
          "`temperature_value` operand. One of them, not both"
        : "requires a temperature: either the `temperature` attribute (a constant "
          "chain) or the `temperature_value` operand (an annealing schedule). It "
          "selects which distribution is sampled and has no default");
  if (hasAttr && getTemperature()->convertToDouble() < 0.0)
    return emitOpError("temperature must be >= 0");
  if (hasValue && !getTemperatureValue().getType().isIntOrFloat())
    return emitOpError("temperature_value must be a scalar float, got ")
           << getTemperatureValue().getType();
  return success();
}
