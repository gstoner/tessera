//===- TesseraAppleDialect.cpp - Apple Silicon Target IR ------*- C++ -*-===//
//
// Dialect / op registration for the hardware-free Apple Silicon Target IR.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/TesseraAppleDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

#include "Tessera/Target/Apple/TesseraAppleDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Tessera/Target/Apple/TesseraAppleTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Tessera/Target/Apple/TesseraAppleOps.cpp.inc"

namespace tessera {
namespace apple {

void TesseraAppleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tessera/Target/Apple/TesseraAppleOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Tessera/Target/Apple/TesseraAppleTypes.cpp.inc"
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

//===----------------------------------------------------------------------===//
// GPU machine primitives — the numerics are contract, so they are verified.
//===----------------------------------------------------------------------===//

// Apple7 supports exactly one simdgroup-matrix shape. Stated once so a future
// part fails closed at the op rather than lowering to a shape the hardware
// does not have.
static constexpr int64_t kSimdgroupMatrixExtent = 8;

// `simdgroup_load`/`store` address row *r* at `base + r * leading_dim`. A
// stride below the matrix width makes consecutive rows overlap: the load reads
// elements belonging to the previous row and the store overwrites them. Metal
// does not fault -- the kernel runs and the numbers are wrong -- so this is the
// one bound worth rejecting rather than trusting.
static ::mlir::LogicalResult verifyRowStride(::mlir::Operation *op,
                                             int64_t leadingDim) {
  if (leadingDim < kSimdgroupMatrixExtent)
    return op->emitOpError()
           << "`leading_dim` must be at least " << kSimdgroupMatrixExtent
           << " (the matrix width); a smaller row stride makes consecutive "
              "rows overlap in memory, which Metal executes silently and "
              "computes wrongly";
  return ::mlir::success();
}

::mlir::LogicalResult SimdgroupLoadOp::verify() {
  return verifyRowStride(getOperation(), getLeadingDim());
}

::mlir::LogicalResult SimdgroupStoreOp::verify() {
  return verifyRowStride(getOperation(), getLeadingDim());
}

::mlir::LogicalResult SimdgroupMatmulOp::verify() {
  // d = a*b + c over 8x8 operands: a full 8x8x8 MMA with K = 8.
  if (getM() != kSimdgroupMatrixExtent || getN() != kSimdgroupMatrixExtent ||
      getK() != kSimdgroupMatrixExtent)
    return emitOpError()
           << "requires an " << kSimdgroupMatrixExtent << "x"
           << kSimdgroupMatrixExtent << "x" << kSimdgroupMatrixExtent
           << " shape; Apple7 has exactly one simdgroup-matrix shape and a "
              "different one has no instruction to lower to";

  auto elementOf = [](::mlir::Type t) {
    return ::llvm::cast<SimdgroupMatrixType>(t).getElementType();
  };
  ::mlir::Type aElem = elementOf(getA().getType());
  ::mlir::Type bElem = elementOf(getB().getType());
  ::mlir::Type cElem = elementOf(getC().getType());
  ::mlir::Type dElem = elementOf(getD().getType());

  // Both inputs carry the declared storage type. A mixed-precision pair is not
  // a Metal simdgroup MMA -- it would be a convert plus an MMA, and accepting
  // it here would hide that conversion from the numerics the epilogue reasons
  // about.
  const bool wantF16 = getStorage() == "f16";
  auto matchesStorage = [&](::mlir::Type t) {
    return wantF16 ? t.isF16() : t.isF32();
  };
  if (!matchesStorage(aElem) || !matchesStorage(bElem))
    return emitOpError() << "operands `a` and `b` must both have element type "
                         << getStorage()
                         << " to match `storage`; a mixed-precision pair is a "
                            "convert plus an MMA, not a simdgroup MMA";

  // The accumulator is ALWAYS fp32, whatever the inputs are. Metal's
  // accumulate form takes simdgroup_float8x8, and the MSL synthesizer relies on
  // it so the fused epilogue sees full-precision matmul results. An f16
  // accumulator would silently change the numerics of every kernel built on
  // this op.
  if (!cElem.isF32() || !dElem.isF32())
    return emitOpError()
           << "accumulator `c` and result `d` must be f32; the simdgroup MMA "
              "accumulates in fp32 regardless of input precision, and an f16 "
              "accumulator would re-round every partial sum";
  return ::mlir::success();
}

::mlir::LogicalResult ThreadgroupBarrierOp::verify() {
  // The enum attribute already constrains the legal set; what is worth saying
  // here is why `none` is legal at all: it is Metal's execution-only barrier,
  // meaningful for lane reconvergence with no memory to order. It is legal and
  // it is almost never what a staging loop wants.
  return ::mlir::success();
}

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
