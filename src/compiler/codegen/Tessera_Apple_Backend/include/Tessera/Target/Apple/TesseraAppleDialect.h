//===- TesseraAppleDialect.h - Apple Silicon Target IR --------*- C++ -*-===//
//
// Hardware-free Target IR dialect for Apple Silicon CPU (Accelerate / vecLib /
// BNNS) and GPU (Metal / MPS) artifacts. Mirrors the retained backend
// pattern (Architecture Decision #19).
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TARGET_APPLE_DIALECT_H
#define TESSERA_TARGET_APPLE_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated dialect declarations (cppNamespace = ::tessera::apple).
#include "Tessera/Target/Apple/TesseraAppleDialect.h.inc"

// Types before operations: the GPU machine primitives take a
// `!tessera_apple.simdgroup_matrix<T>` operand, so the generated op classes
// reference SimdgroupMatrixType and will not compile without it in scope.
#define GET_TYPEDEF_CLASSES
#include "Tessera/Target/Apple/TesseraAppleTypes.h.inc"

#define GET_OP_CLASSES
#include "Tessera/Target/Apple/TesseraAppleOps.h.inc"

namespace tessera {
namespace apple {

/// Insert the Apple Target IR dialect into a DialectRegistry. Call from
/// tessera-opt and any other tool that needs to parse / verify Apple IR.
void registerAppleDialect(::mlir::DialectRegistry &registry);

} // namespace apple
} // namespace tessera


namespace tessera::apple {
/// Bits per element of a Metal 4 tensor storage type (f16/bf16 16, f32 32,
/// f8E4M3FN/f8E5M2 8, f4E2M1FN 4), or 0 for a type MTLTensor cannot hold.
int64_t appleTensorStorageBits(::mlir::Type elementType);
/// Why a rank-2 view violates Apple's MTLTensor layout contract (empty string
/// when it is legal). Shared by the tensor_view verifier and the lowering that
/// decides whether a canonical GEMM can be re-formed on this lane.
std::string appleTensorViewLayoutReason(::mlir::Type elementType,
                                        ::llvm::ArrayRef<int64_t> extents,
                                        ::llvm::ArrayRef<int64_t> strides,
                                        int64_t byteOffset);
/// The MPP matmul2d operand-pair table. Returns the runtime format code for
/// low-precision pairs (0 e4m3/e4m3, 1 e5m2/e5m2, 2 e2m1/e2m1, 3 f16/e4m3,
/// 4 f16/e5m2, 5 f16/e2m1), 10 for f16/f16, 11 for bf16/bf16, or -1.
int appleMatmul2dPairCode(::mlir::Type a, ::mlir::Type b);

/// Runtime activation code for `gpu.matmul2d_epilogue`'s `act`: 0 none,
/// 1 relu, 2 gelu (tanh form), 3 silu; -1 for anything else. The codes are
/// the MSL kernel's `ts_epi` contract.
int appleMatmul2dActCode(::llvm::StringRef act);
} // namespace tessera::apple

#endif // TESSERA_TARGET_APPLE_DIALECT_H
