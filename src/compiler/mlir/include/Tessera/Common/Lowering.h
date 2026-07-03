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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

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

/// One input operand of a fusion runtime call: the tensor value and the
/// `MemRefType` it bufferizes to (its shape may differ from the tensor's, e.g.
/// a gate operand packed to `{B,H,S,numBlocks}`).
struct FusionCallInput {
  ::mlir::Value tensor;
  ::mlir::MemRefType memTy;
};

/// Emit the shared Apple-GPU fusion-lowering skeleton and return the output
/// tensor value.  Bufferizes each input to a raw i64 pointer, allocates the
/// output and takes its pointer, materializes the i32 dimension args, declares
/// + calls the runtime C-ABI `symbol`, stamps the `descriptorAttrs` on the call,
/// and rounds the output memref back to a tensor.  The caller does the match and
/// the `replaceOp`/`eraseOp`, and must have set the builder insertion point.
///
/// The runtime C ABI is uniform across every fusion kernel:
///   (input ptrs… i64, output ptr i64, dimension args… i32) -> ()
/// Op-creation order (input ptrs → output alloc+ptr → int consts → decl → call
/// → to_tensor) is fixed so the emitted IR is byte-identical to the previously
/// hand-written passes.
inline ::mlir::Value emitFusionCall(
    ::mlir::OpBuilder &b, ::mlir::Location loc, ::mlir::ModuleOp mod,
    ::mlir::StringRef symbol, ::llvm::ArrayRef<FusionCallInput> inputs,
    ::mlir::MemRefType outMemTy, ::mlir::RankedTensorType outTensorTy,
    ::llvm::ArrayRef<int64_t> intArgs,
    ::llvm::ArrayRef<::mlir::NamedAttribute> descriptorAttrs) {
  ::mlir::Type i64Ty = b.getI64Type();
  ::mlir::Type i32Ty = b.getI32Type();

  ::llvm::SmallVector<::mlir::Value> args;
  ::llvm::SmallVector<::mlir::Type> argTypes;
  for (const FusionCallInput &in : inputs) {
    args.push_back(extractPtr(b, loc, in.tensor, in.memTy));
    argTypes.push_back(i64Ty);
  }
  auto oAlloc = b.create<::mlir::memref::AllocOp>(loc, outMemTy);
  {
    auto pi =
        b.create<::mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, oAlloc);
    args.push_back(b.create<::mlir::arith::IndexCastOp>(loc, i64Ty, pi));
    argTypes.push_back(i64Ty);
  }
  for (int64_t v : intArgs) {
    args.push_back(b.create<::mlir::arith::ConstantIntOp>(loc, v, 32));
    argTypes.push_back(i32Ty);
  }

  auto fnTy = ::mlir::FunctionType::get(b.getContext(), argTypes, {});
  ensureExternalDecl(mod, symbol, fnTy);

  auto callOp =
      b.create<::mlir::func::CallOp>(loc, symbol, ::mlir::TypeRange{}, args);
  for (const ::mlir::NamedAttribute &attr : descriptorAttrs)
    callOp->setAttr(attr.getName(), attr.getValue());

  return b.create<::mlir::bufferization::ToTensorOp>(loc, outTensorTy, oAlloc);
}

} // namespace common
} // namespace tessera

#endif // TESSERA_COMMON_LOWERING_H
