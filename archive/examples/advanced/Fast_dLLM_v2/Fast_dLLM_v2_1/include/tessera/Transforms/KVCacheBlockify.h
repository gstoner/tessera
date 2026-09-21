#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace tessera {
std::unique_ptr<mlir::Pass> createKVCacheBlockifyPass();
void registerKVCacheBlockifyPass();
} // namespace tessera
