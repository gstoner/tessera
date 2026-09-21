#pragma once
#include "mlir/Pass/Pass.h"

namespace tessera {
std::unique_ptr<mlir::Pass> createEmpiricalSearchPass();
} // namespace tessera
