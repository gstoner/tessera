//===- Lowering.h - Shared Tile->Target lowering helpers -------*- C++ -*-===//
//
// Workstream A1 (COMPILER_REFACTOR_PLAN §3, Workstream A) — the
// bufferize->extract-ptr->func.call C-ABI helpers were duplicated
// byte-for-byte between the x86 backend
// (src/transforms/lib/TileToX86Pass.cpp) and the Apple backend (~18 Tile->Target
// passes via Tessera/Target/Apple/LoweringUtils.h). Both lower a tile op into a
// call to a runtime C-ABI symbol that takes raw data pointers as i64. Hoisted
// here as one definition in `tessera::common`; the backend-local names forward
// to it with `using` declarations, so no call site changes.
//
// NOTE: ROCm's TileToROCM does NOT use this pattern — it rewrites tile.mma into
// `tessera_rocm.mfma`/`wmma` ops directly (op-rewriting, not a C-ABI call), so
// it is intentionally out of scope for this helper.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_COMMON_LOWERING_H
#define TESSERA_COMMON_LOWERING_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace tessera {
namespace common {

/// Bufferize `tensor` to `memTy` and return its aligned base pointer as an
/// i64 (the runtime C ABI takes raw pointers as i64).
inline ::mlir::Value extractPtr(::mlir::OpBuilder &b, ::mlir::Location loc,
                                ::mlir::Value tensor,
                                ::mlir::MemRefType memTy) {
  auto buf = b.create<::mlir::bufferization::ToBufferOp>(loc, memTy, tensor);
  auto ptrIdx =
      b.create<::mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, buf);
  return b.create<::mlir::arith::IndexCastOp>(loc, b.getI64Type(), ptrIdx);
}

/// Look up (or insert, private) an external `func.func` declaration `name`
/// with signature `fnTy` at the top of `mod`.
inline ::mlir::func::FuncOp ensureExternalDecl(::mlir::ModuleOp mod,
                                               ::mlir::StringRef name,
                                               ::mlir::FunctionType fnTy) {
  if (auto fn = mod.lookupSymbol<::mlir::func::FuncOp>(name))
    return fn;
  ::mlir::OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  auto fn = b.create<::mlir::func::FuncOp>(mod.getLoc(), name, fnTy);
  fn.setPrivate();
  return fn;
}

} // namespace common
} // namespace tessera

#endif // TESSERA_COMMON_LOWERING_H
