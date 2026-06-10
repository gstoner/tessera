//===- LoweringUtils.h - Shared Apple lowering helpers ---------*- C++ -*-===//
//
// Audit 2026-06-10 (CODE_AUDIT_2026_06_10 §4) — these two helpers were
// copy-pasted byte-for-byte as anonymous-namespace `static` functions across
// ~18 Apple Tile→Target lowering passes. Hoisted here as `inline` functions
// in `tessera::apple` so every pass shares one definition; the unqualified
// call sites inside `tessera::apple::{anonymous}` resolve via enclosing-
// namespace lookup with no change.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TARGET_APPLE_LOWERINGUTILS_H
#define TESSERA_TARGET_APPLE_LOWERINGUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace tessera {
namespace apple {

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

} // namespace apple
} // namespace tessera

#endif // TESSERA_TARGET_APPLE_LOWERINGUTILS_H
