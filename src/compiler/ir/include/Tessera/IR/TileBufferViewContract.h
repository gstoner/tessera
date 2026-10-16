#pragma once
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <optional>

namespace tessera::tile {
inline std::optional<int64_t> scalarViewConstant(mlir::Value value) {
  auto op = value.getDefiningOp<mlir::arith::ConstantOp>();
  auto attr = op ? mlir::dyn_cast<mlir::IntegerAttr>(op.getValue()) : mlir::IntegerAttr{};
  return attr ? std::optional<int64_t>(attr.getInt()) : std::nullopt;
}
inline bool scalarViewSameBound(mlir::Value lhs, mlir::Value rhs) {
  if (lhs == rhs) return true;
  auto a = scalarViewConstant(lhs), b = scalarViewConstant(rhs);
  return a && b && *a == *b;
}
// A proof about the actual enclosing SSA regions, not a trusted metadata flag.
inline bool scalarViewCoordinateInBounds(mlir::Operation *access,
                                        mlir::Value index, mlir::Value bound) {
  using namespace mlir;
  auto extent = scalarViewConstant(bound);
  if (!extent || *extent <= 0) return false;
  if (auto constant = scalarViewConstant(index)) return *constant >= 0 && *constant < *extent;
  if (auto argument = dyn_cast<BlockArgument>(index)) {
    auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (loop && index == loop.getInductionVar() &&
        scalarViewConstant(loop.getLowerBound()) == std::optional<int64_t>(0) &&
        scalarViewConstant(loop.getStep()) == std::optional<int64_t>(1) &&
        scalarViewSameBound(loop.getUpperBound(),bound)) return true;
  }
  for (Operation *child = access, *parent = child->getParentOp(); parent;
       child = parent, parent = parent->getParentOp()) {
    auto guard = dyn_cast<scf::IfOp>(parent);
    if (!guard || child->getParentRegion() != &guard.getThenRegion()) continue;
    auto compare = guard.getCondition().getDefiningOp<arith::CmpIOp>();
    if (compare && compare.getPredicate() == arith::CmpIPredicate::ult &&
        compare.getLhs() == index && scalarViewSameBound(compare.getRhs(),bound)) return true;
  }
  return false;
}
inline mlir::LogicalResult verifyScalarBufferView(mlir::Operation *op) {
  using namespace mlir;
  if (auto view = dyn_cast<BufferViewOp>(op)) {
    auto ptr = dyn_cast<LLVM::LLVMPointerType>(view.getBase().getType());
    if (!ptr || (ptr.getAddressSpace() != 0 && ptr.getAddressSpace() != 1 && ptr.getAddressSpace() != 3))
      return op->emitOpError("requires a native pointer in generic, global or shared memory");
    auto type = cast<BufferViewType>(view.getResult().getType());
    auto constant = [](Value value) -> IntegerAttr {
      auto c = value.getDefiningOp<arith::ConstantOp>();
      return c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr{};
    };
    auto rows = constant(view.getRows()), cols = constant(view.getColumns()), stride = constant(view.getLeadingDim());
    if ((rows && rows.getInt() <= 0) || (cols && cols.getInt() <= 0) || (stride && stride.getInt() <= 0))
      return op->emitOpError("constant bounds and physical leading dimension must be positive");
    auto minor = type.getOrder() == "row_major" ? cols : rows;
    if (stride && minor && stride.getInt() < minor.getInt())
      return op->emitOpError("physical leading dimension is smaller than the minor extent");
    return success();
  }
  Value viewValue, element;
  bool read = false;
  if (auto load = dyn_cast<LoadElementOp>(op)) {
    viewValue = load.getView(); element = load.getResult(); read = true;
  } else if (auto store = dyn_cast<StoreElementOp>(op)) {
    viewValue = store.getView(); element = store.getValue();
  } else return op->emitOpError("expected a scalar buffer access");
  auto view = viewValue.getDefiningOp<BufferViewOp>();
  auto type = dyn_cast<BufferViewType>(viewValue.getType());
  if (!view || !type || element.getType() != type.getElementType())
    return op->emitOpError("requires a direct typed buffer view and an exactly matching storage element");
  if ((read && type.getAccess() == "write") || (!read && type.getAccess() == "read"))
    return op->emitOpError("scalar access violates the view's read/write permission");
  return success();
}

// Shared native memory materializer: consumes the typed addressing contract,
// never an operation-name arithmetic recipe. Call only after target admission.
inline mlir::LogicalResult lowerScalarBufferAccesses(mlir::Operation *root) {
  using namespace mlir;
  SmallVector<Operation *> accesses;
  root->walk([&](Operation *op) {
    if (isa<LoadElementOp, StoreElementOp>(op)) accesses.push_back(op);
  });
  for (Operation *op : accesses) {
    if (failed(verifyScalarBufferView(op))) return failure();
    OpBuilder b(op);
    Location loc = op->getLoc();
    auto load = dyn_cast<LoadElementOp>(op);
    auto store = dyn_cast<StoreElementOp>(op);
    auto view = (load ? load.getView() : store.getView()).getDefiningOp<BufferViewOp>();
    auto type = cast<BufferViewType>(view.getResult().getType());
    Value row = load ? load.getRow() : store.getRow();
    Value col = load ? load.getColumn() : store.getColumn();
    auto rows = scalarViewConstant(view.getRows()), cols = scalarViewConstant(view.getColumns());
    auto stride = scalarViewConstant(view.getLeadingDim());
    bool rowMajor = type.getOrder() == "row_major";
    bool dimensions = rows && cols && stride && *rows > 0 && *cols > 0 &&
        *stride >= (rowMajor ? *cols : *rows) &&
        (rowMajor ? *rows : *cols) <= INT64_MAX / *stride;
    unsigned alignment = cast<FloatType>(type.getElementType()).getWidth() / 8;
    if (dimensions) dimensions = (rowMajor ? *rows : *cols) * *stride <= INT64_MAX / alignment;
    auto accessElement = [&]() -> Value {
      Value major = rowMajor ? row : col;
      Value minorIndex = rowMajor ? col : row;
      Value origin = arith::MulIOp::create(b, loc, major, view.getLeadingDim());
      Value index = arith::AddIOp::create(b, loc, origin, minorIndex);
      Value address = LLVM::GEPOp::create(b, loc, view.getBase().getType(), type.getElementType(), view.getBase(), ValueRange{index});
      if (load) return LLVM::LoadOp::create(b, loc, type.getElementType(), address, alignment);
      LLVM::StoreOp::create(b, loc, store.getValue(), address, alignment);
      return {};
    };
    if (dimensions && scalarViewCoordinateInBounds(op,row,view.getRows()) &&
        scalarViewCoordinateInBounds(op,col,view.getColumns())) {
      Value result = accessElement();
      if (load) load.getResult().replaceAllUsesWith(result);
      op->erase();
      continue;
    }
    Value zero = arith::ConstantIntOp::create(b, loc, 0, 64);
    Value positiveRows = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, view.getRows(), zero);
    Value positiveCols = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, view.getColumns(), zero);
    Value valid = arith::AndIOp::create(b, loc, positiveRows, positiveCols);
    Value rowBound = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, row, view.getRows());
    Value colBound = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, col, view.getColumns());
    valid = arith::AndIOp::create(b, loc, valid, arith::AndIOp::create(b, loc, rowBound, colBound));
    Value minor = type.getOrder() == "row_major" ? view.getColumns() : view.getRows();
    Value strideValid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, view.getLeadingDim(), minor);
    valid = arith::AndIOp::create(b, loc, valid, strideValid);
    auto guarded = scf::IfOp::create(b, loc, load ? TypeRange{type.getElementType()} : TypeRange{}, valid, bool(load));
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&guarded.getThenRegion().front());
      Value result = accessElement();
      if (load) {
        scf::YieldOp::create(b, loc, result);
      }
    }
    if (load) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&guarded.getElseRegion().front());
      Value empty = arith::ConstantOp::create(b, loc, type.getElementType(), b.getFloatAttr(type.getElementType(), 0));
      scf::YieldOp::create(b, loc, empty);
      load.getResult().replaceAllUsesWith(guarded.getResult(0));
    }
    op->erase();
  }
  SmallVector<BufferViewOp> views;
  root->walk([&](BufferViewOp view) { views.push_back(view); });
  for (auto view : views) {
    if (!view.getResult().use_empty()) return view.emitError("typed native buffer view has an unsupported escaping consumer");
    view.erase();
  }
  return success();
}
} // namespace tessera::tile
