#ifndef TESSERA_ROCM_PHYSICAL_WMMA_PANEL_H
#define TESSERA_ROCM_PHYSICAL_WMMA_PANEL_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SHA256.h"

#include <cassert>
#include <string>

namespace mlir::tessera_rocm {

LogicalResult materializeGfx11WmmaGemmPhysicalBody(gpu::GPUFuncOp function,
                                                   int64_t mt, int64_t nt);

inline std::string gfx11WmmaGemmTileBodyDigest(gpu::GPUFuncOp function) {
  auto isContractOp = [](Operation *op) {
    StringRef name = op->getName().getStringRef();
    return name == "tile.view" || name == "tile.fragment_zero" ||
           name == "tile.fragment_pack" || name == "tile.mma" ||
           name == "tile.fragment_unpack" || name == "tile.store";
  };
  DenseMap<Operation *, unsigned> ordinal;
  function.walk([&](Operation *op) {
    if (isContractOp(op))
      ordinal.try_emplace(op, ordinal.size());
  });
  std::string text;
  llvm::raw_string_ostream stream(text);
  function.walk([&](Operation *op) {
    auto found = ordinal.find(op);
    if (found == ordinal.end())
      return;
    stream << found->second << ':' << op->getName() << ':';
    for (Type type : op->getResultTypes()) {
      type.print(stream);
      stream << ';';
    }
    stream << "<-";
    for (Value operand : op->getOperands()) {
      if (Operation *definition = operand.getDefiningOp()) {
        auto producer = ordinal.find(definition);
        if (producer != ordinal.end())
          stream << 't' << producer->second;
        else
          stream << 'o' << definition->getName() << '#'
                 << cast<OpResult>(operand).getResultNumber();
      } else {
        auto argument = cast<BlockArgument>(operand);
        stream << 'b' << argument.getArgNumber();
      }
      stream << ':';
      operand.getType().print(stream);
      stream << ';';
    }
    for (StringRef name : {"tile.layout", "tile.memory", "tile.linear_base"})
      if (Attribute attr = op->getAttr(name)) {
        stream << name << '=';
        attr.print(stream);
        stream << ';';
      }
    stream << '\n';
  });
  stream.flush();
  return llvm::toHex(
      llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}

/// Emit one register-owned gfx11 WMMA panel.
///
/// This is deliberately below Tile semantics and above ROCDL instruction
/// selection. The direct GEMM generator and typed Tile consumer call the same
/// routine, giving address evolution, fragment construction, WMMA issue order,
/// and the resulting register-pressure graph one target-owned implementation.
inline SmallVector<Value> emitGfx11WmmaPhysicalPanel(
    OpBuilder &builder, Location loc, Value a, Value b, Value n, Value k0,
    ArrayRef<Value> aRowOffsets, ArrayRef<Value> bColumns, ValueRange acc,
    Value inputZero, VectorType inputType, VectorType accumulatorType) {
  const int64_t mt = static_cast<int64_t>(aRowOffsets.size());
  const int64_t nt = static_cast<int64_t>(bColumns.size());
  assert(mt > 0 && nt > 0 && acc.size() == size_t(mt * nt));

  SmallVector<Value> aFragments(mt), bFragments(nt, inputZero);
  for (int64_t mi = 0; mi < mt; ++mi) {
    Value base = arith::AddIOp::create(builder, loc, aRowOffsets[mi], k0);
    aFragments[mi] =
        vector::LoadOp::create(builder, loc, inputType, a, ValueRange{base});
  }
  for (int64_t i = 0; i < 16; ++i) {
    Value ci = arith::ConstantIndexOp::create(builder, loc, i);
    Value row = arith::AddIOp::create(builder, loc, k0, ci);
    Value rowBase = arith::MulIOp::create(builder, loc, row, n);
    for (int64_t ni = 0; ni < nt; ++ni) {
      Value address =
          arith::AddIOp::create(builder, loc, rowBase, bColumns[ni]);
      Value loaded =
          memref::LoadOp::create(builder, loc, b, ValueRange{address});
      bFragments[ni] = vector::InsertOp::create(
          builder, loc, loaded, bFragments[ni], ArrayRef<int64_t>{i});
    }
  }

  SmallVector<Value> next(mt * nt);
  for (int64_t mi = 0; mi < mt; ++mi)
    for (int64_t ni = 0; ni < nt; ++ni) {
      OperationState state(loc, "tessera_rocm.wmma");
      state.addOperands(
          {aFragments[mi], bFragments[ni], acc[mi * nt + ni]});
      state.addTypes(accumulatorType);
      next[mi * nt + ni] = builder.create(state)->getResult(0);
    }
  return next;
}

} // namespace mlir::tessera_rocm

#endif // TESSERA_ROCM_PHYSICAL_WMMA_PANEL_H
