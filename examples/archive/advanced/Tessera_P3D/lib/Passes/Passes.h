#pragma once
#include <memory>
namespace mlir { class Pass; }
std::unique_ptr<mlir::Pass> createLowerP3DPass();
std::unique_ptr<mlir::Pass> createAutotuneP3DPass();
