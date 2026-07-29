//===- TesseraAppleDialect.cpp - Apple Silicon Target IR ------*- C++ -*-===//
//
// Dialect / op registration for the hardware-free Apple Silicon Target IR.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/TesseraAppleDialect.h"

#include "mlir/IR/DialectImplementation.h"

#include <optional>

#include "Tessera/Target/Apple/TesseraAppleDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Tessera/Target/Apple/TesseraAppleOps.cpp.inc"

namespace tessera {
namespace apple {

void TesseraAppleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tessera/Target/Apple/TesseraAppleOps.cpp.inc"
      >();
}

void registerAppleDialect(::mlir::DialectRegistry &registry) {
  registry.insert<TesseraAppleDialect>();
}

//===----------------------------------------------------------------------===//
// Value-call verification
//
// These ops name a C ABI symbol the runtime is expected to dispatch. Nothing
// checked that: a call naming a symbol that exists nowhere, with an invented
// op_kind and a negative dimension, round-tripped clean and only failed (or
// silently fell back) at launch. The checks below are the ones a wrong emitter
// actually trips, kept structural so they do not need a live runtime.
//===----------------------------------------------------------------------===//

namespace {

/// Shared shape checks for cpu.call / gpu.kernel_call / gpu.package_call.
::mlir::LogicalResult verifyValueCall(::mlir::Operation *op,
                                      ::llvm::StringRef opKind,
                                      ::llvm::StringRef symbol,
                                      ::std::optional<::llvm::StringRef> status) {
  if (symbol.empty())
    return op->emitOpError("`symbol` must name the C ABI entry the runtime "
                           "dispatches to; it is empty");
  if (symbol.find_first_of(" \t\n") != ::llvm::StringRef::npos)
    return op->emitOpError("`symbol` must be a C identifier, got '")
           << symbol << "'";
  if (opKind.empty())
    return op->emitOpError("`op_kind` must name the op this call stands for; "
                           "it is empty");
  if (opKind.find_first_of(" \t\n") != ::llvm::StringRef::npos)
    return op->emitOpError("`op_kind` must be a bare name, got '")
           << opKind << "'";
  if (status && *status != "executable" && *status != "artifact")
    return op->emitOpError("`status` must be \"executable\" (a runtime "
                           "dispatcher exists) or \"artifact\" (values flow, "
                           "execution is pending), got '")
           << *status << "'";
  if (op->getNumResults() == 0)
    return op->emitOpError("a value call must produce at least one result; "
                           "use the attribute-only artifact ops for metadata");
  return ::mlir::success();
}

} // namespace

::mlir::LogicalResult CallOp::verify() {
  return verifyValueCall(*this, getOpKind(), getSymbol(), getStatus());
}

::mlir::LogicalResult KernelCallOp::verify() {
  // No op_kind registry check here, deliberately. `status = "executable"` means
  // the named C ABI symbol exists — `tessera_apple_gpu_cholesky_f32` does, and
  // the compiler emits it. Whether `runtime.py`'s *value-lane launcher* accepts
  // that op_kind is a narrower, separate question: its allowlist is shorter
  // than the emitter's, so enforcing it here would reject IR that is correct by
  // this op's own contract. That gap is real and worth watching, so it is
  // tracked as a ratchet in `test_apple_value_op_kind_registry.py` instead of
  // being asserted as a compile-time invariant it was never defined to be.
  return verifyValueCall(*this, getOpKind(), getSymbol(), getStatus());
}

::mlir::LogicalResult PackageCallOp::verify() {
  return verifyValueCall(*this, getOpKind(), getSymbol(), getStatus());
}

} // namespace apple
} // namespace tessera
