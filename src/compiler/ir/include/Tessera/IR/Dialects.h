#pragma once

#include "mlir/IR/DialectRegistry.h"

namespace tessera {

void registerTesseraDialects(mlir::DialectRegistry &registry);

} // namespace tessera
