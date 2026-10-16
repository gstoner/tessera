#include "Tessera/Dialect/Tile/TileDialect.h"
// Isolated native scalarization component. No Schedule/Tile execution claim.
// The caller supplies verified static row Graph SSA; no derivative is rebuilt.
#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include <cmath>
#include <functional>

namespace tessera::row_region {
using namespace mlir;

// Keep numerical validation independent of the row ownership ABI. This scalar
// function is subsequently embedded in a checked Schedule/Tile kernel region.
class RowScalarizer {
  func::FuncOp graph;
  OpBuilder b;
  Location loc;
  RankedTensorType tensor;
  Value row, width;
  SmallVector<Value> pointers;
  SmallVector<Value> views;
  llvm::DenseMap<Value, Value> rowValues;
  llvm::DenseMap<Operation *, Value> normDenominators;
  llvm::DenseMap<Value, Value> secondMoments;
  using Cache = llvm::DenseMap<Value, Value>;

  Value constant(Type type, double value) {
    return arith::ConstantOp::create(b, loc, type, b.getFloatAttr(type, value));
  }
  Value convert(Value value, Type type) {
    if (value.getType() == type) return value;
    auto source = cast<FloatType>(value.getType());
    auto target = cast<FloatType>(type);
    if (source.getWidth() < target.getWidth())
      return arith::ExtFOp::create(b, loc, type, value);
    return arith::TruncFOp::create(b, loc, type, value);
  }
  Value address(Value pointer, Value column) {
    Value base = arith::MulIOp::create(b, loc, row, width);
    Value offset = arith::AddIOp::create(b, loc, base, column);
    return LLVM::GEPOp::create(b, loc, pointer.getType(),
                              tensor.getElementType(), pointer, ValueRange{offset});
  }
  Value reduction(const std::function<Value(Value, Cache &)> &expression) {
    auto zero = arith::ConstantIntOp::create(b, loc, 0, 64);
    auto one = arith::ConstantIntOp::create(b, loc, 1, 64);
    Value initial = constant(b.getF32Type(), 0);
    auto loop = scf::ForOp::create(b, loc, zero, width, one, ValueRange{initial});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Cache cache;
      Value value = expression(loop.getInductionVar(), cache);
      Value sum = arith::AddFOp::create(b, loc, loop.getRegionIterArg(0), value);
      scf::YieldOp::create(b, loc, sum);
    }
    return arith::DivFOp::create(b, loc, loop.getResult(0),
                                constant(b.getF32Type(), tensor.getDimSize(1)));
  }
  Value epsilon(Operation *op) {
    auto eps = op->getAttrOfType<FloatAttr>("eps");
    return constant(b.getF32Type(), eps ? eps.getValueAsDouble() : 1e-5);
  }
  Value secondMoment(Value source) {
    // An exact storage->FP32 extension does not change the value entering
    // the FP32 square. Key both spellings by the same actual SSA source.
    // Never cross a truncation, arithmetic expression, or policy boundary.
    if (Operation *castOp = source.getDefiningOp(); castOp &&
        castOp->getName().getStringRef() == "tessera.cast" &&
        cast<RankedTensorType>(source.getType()).getElementType().isF32() &&
        cast<RankedTensorType>(castOp->getOperand(0).getType()).getElementType() == tensor.getElementType())
      source = castOp->getOperand(0);
    if (auto found = secondMoments.find(source); found != secondMoments.end()) return found->second;
    Value moment = reduction([&](Value col, Cache &cache) -> Value {
      Value x = convert(scalar(source, col, cache), b.getF32Type());
      return arith::MulFOp::create(b, loc, x, x);
    });
    secondMoments[source] = moment;
    return moment;
  }
  void prepare(Operation *op) {
    auto name = op->getName().getStringRef();
    if (name == "tessera.rmsnorm") {
      Value meanSquare = secondMoment(op->getOperand(0));
      Value moment = arith::AddFOp::create(b, loc, meanSquare, epsilon(op));
      normDenominators[op] = math::SqrtOp::create(b, loc, moment);
    } else if (name == "tessera.normalization_stats") {
      // Center is a genuine Graph result, but can be dead in an uncentered
      // derivative. Emit it only if SSA contains an actual use.
      if (!op->getResult(0).use_empty())
        rowValues[op->getResult(0)] = reduction([&](Value col, Cache &cache) {
          return scalar(op->getOperand(0), col, cache);
        });
      Value moment = secondMoment(op->getOperand(0));
      Value adjusted = arith::AddFOp::create(b, loc, moment, epsilon(op));
      Value denom = math::SqrtOp::create(b, loc, adjusted);
      rowValues[op->getResult(1)] = arith::DivFOp::create(
          b, loc, constant(b.getF32Type(), 1), denom);
    } else if (name == "tessera.reduce") {
      rowValues[op->getResult(0)] = reduction([&](Value col, Cache &cache) {
        return scalar(op->getOperand(0), col, cache);
      });
    }
  }
  Value scalar(Value value, Value column, Cache &cache) {
    if (auto found = rowValues.find(value); found != rowValues.end())
      return found->second;
    if (auto found = cache.find(value); found != cache.end()) return found->second;
    Value result;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      result = tile::LoadElementOp::create(b, loc, tensor.getElementType(), views[arg.getArgNumber()], row, column);
    } else {
      Operation *op = value.getDefiningOp();
      auto name = op->getName().getStringRef();
      auto elem = cast<RankedTensorType>(value.getType()).getElementType();
      if (name == "arith.constant") {
        auto dense = op->getAttrOfType<DenseElementsAttr>("value");
        result = constant(elem, dense.getSplatValue<APFloat>().convertToDouble());
        cache[value] = result;
        return result;
      }
      Value lhs = scalar(op->getOperand(0), column, cache);
      if (name == "tessera.cast") result = convert(lhs, elem);
      else if (name == "tessera.broadcast_in_dim") result = lhs;
      else if (name == "tessera.rmsnorm") {
        Value widened = convert(lhs, b.getF32Type());
        Value normalized = arith::DivFOp::create(b, loc, widened, normDenominators.lookup(op));
        result = convert(normalized, elem);
      } else if (name == "tessera.relu") {
        // Preserve the original storage payload for unordered comparisons
        // and BF16 negative zero, matching the canonical ReLU contract.
        Value zero = constant(elem, 0);
        Value compared = convert(lhs, b.getF32Type());
        auto predicate = elem.isBF16() ? arith::CmpFPredicate::OLT
                                      : arith::CmpFPredicate::OLE;
        Value replace = arith::CmpFOp::create(
            b, loc, predicate, compared, constant(b.getF32Type(), 0));
        result = arith::SelectOp::create(b, loc, replace, zero, lhs);
      } else if (name == "tessera.compare_scalar") {
        auto rhs = op->getAttrOfType<FloatAttr>("rhs");
        result = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT,
            lhs, constant(lhs.getType(), rhs.getValueAsDouble()));
      } else {
        Value rhs = scalar(op->getOperand(1), column, cache);
        if (name == "tessera.mul") result = arith::MulFOp::create(b, loc, lhs, rhs);
        else if (name == "tessera.sub") result = arith::SubFOp::create(b, loc, lhs, rhs);
        else if (name == "tessera.add") result = arith::AddFOp::create(b, loc, lhs, rhs);
        else if (name == "tessera.masked_fill") {
          auto fill = op->getAttrOfType<FloatAttr>("value");
          result = arith::SelectOp::create(b, loc, rhs, lhs, constant(elem, fill.getValueAsDouble()));
        }
      }
    }
    assert(result && "validated row operation must have a scalar result");
    cache[value] = result;
    return result;
  }

public:
  explicit RowScalarizer(func::FuncOp fn)
      : graph(fn), b(fn.getContext()), loc(fn.getLoc()) {}
  LogicalResult validate() {
    auto reject = [&]() { return graph.emitError("native row scalarization requires a verified static two-root/two-result explicit-fp32 row graph"); };
    if (graph.getNumArguments() != 2 || graph.getNumResults() != 2 ||
        !graph.getBody().hasOneBlock()) return reject();
    tensor = dyn_cast<RankedTensorType>(graph.getArgumentTypes()[0]);
    if (!tensor || tensor.getRank() != 2 || !tensor.hasStaticShape() || tensor.getEncoding() ||
        tensor.getDimSize(0) <= 0 || tensor.getDimSize(1) <= 0 ||
        tensor.getDimSize(0) > INT32_MAX || tensor.getDimSize(1) > INT32_MAX ||
        (!tensor.getElementType().isF16() && !tensor.getElementType().isBF16()) ||
        graph.getArgumentTypes()[1] != tensor || graph.getResultTypes()[0] != tensor ||
        graph.getResultTypes()[1] != tensor) return reject();
    for (Operation &op : graph.getBody().front().without_terminator()) {
      StringRef name = op.getName().getStringRef();
      if (op.getNumRegions()) return reject();
      bool stats = name == "tessera.normalization_stats", reduce = name == "tessera.reduce";
      bool scalarConstant = name == "arith.constant";
      if (op.getNumResults() != (stats ? 2 : 1)) return reject();
      for (Value result : op.getResults()) {
        auto t = dyn_cast<RankedTensorType>(result.getType());
        if (!t || t.getEncoding() || !t.hasStaticShape()) return reject();
        if (scalarConstant) {
          if (t.getRank() != 0 || !t.getElementType().isF32()) return reject();
        } else if (stats || reduce) {
          if (t.getShape() != ArrayRef<int64_t>{tensor.getDimSize(0)} || !t.getElementType().isF32()) return reject();
        } else if (t.getShape() != tensor.getShape() ||
                   (t.getElementType() != tensor.getElementType() && !t.getElementType().isF32() && !t.getElementType().isInteger(1))) return reject();
      }
      auto only = [&](std::initializer_list<StringRef> fields) {
        for (auto attr : op.getAttrs()) {
          StringRef key = attr.getName().getValue();
          if (key == "tessera.effect_kind" && attr.getValue() == b.getStringAttr("pure")) continue;
          if (!llvm::is_contained(fields, key)) return false;
        }
        return true;
      };
      if (scalarConstant) {
        auto dense = op.getAttrOfType<DenseElementsAttr>("value");
        if (!only({"value"}) || op.getNumOperands() || !dense || !dense.isSplat() ||
            !dense.getSplatValue<APFloat>().isFinite()) return reject();
      } else if (name == "tessera.rmsnorm") {
        auto policy = op.getAttrOfType<DictionaryAttr>("numeric_policy");
        auto accum = policy ? policy.getAs<StringAttr>("accum") : StringAttr{};
        if (!only({"eps", "axis", "numeric_policy"}) || op.getNumOperands() != 1 ||
            op.getOperand(0).getType() != tensor || op.getResult(0).getType() != tensor ||
            !accum || accum.getValue() != "fp32") return reject();
        for (auto a : policy) {
          if (a.getName() == "accum") continue;
          auto storage = dyn_cast<StringAttr>(a.getValue());
          if (a.getName() != "storage" || !storage || storage.getValue() !=
              (tensor.getElementType().isF16() ? "fp16" : "bf16")) return reject();
        }
      } else if (stats) {
        auto centered = op.getAttrOfType<BoolAttr>("centered");
        if (!only({"eps", "axis", "centered"}) || !centered || centered.getValue() ||
            op.getNumOperands() != 1 || !cast<RankedTensorType>(op.getOperand(0).getType()).getElementType().isF32()) return reject();
      } else if (reduce) {
        auto kind = op.getAttrOfType<StringAttr>("kind");
        if (!only({"axis", "kind", "keepdims"}) || !kind || kind.getValue() != "mean" || op.getNumOperands() != 1) return reject();
        if (auto keep = op.getAttrOfType<BoolAttr>("keepdims"); keep && keep.getValue()) return reject();
      } else if (name == "tessera.broadcast_in_dim") {
        auto dims = op.getAttrOfType<ArrayAttr>("broadcast_dimensions");
        auto axis = dims && dims.size() == 1 ? dyn_cast<IntegerAttr>(dims[0]) : IntegerAttr{};
        if (!only({"broadcast_dimensions"}) || op.getNumOperands() != 2 || !dims)
          return reject();
        auto sourceType = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
        bool scalar = sourceType && sourceType.getRank() == 0 && dims.empty();
        bool rowStat = sourceType && sourceType.getRank() == 1 && axis && axis.getInt() == 0;
        if (!scalar && !rowStat) return reject();
      } else if (name == "tessera.cast") {
        if (!only({}) || op.getNumOperands() != 1) return reject();
        auto from = cast<RankedTensorType>(op.getOperand(0).getType()).getElementType();
        auto to = cast<RankedTensorType>(op.getResult(0).getType()).getElementType();
        if (!((from == tensor.getElementType() && to.isF32()) || (from.isF32() && to == tensor.getElementType()))) return reject();
      } else if (name == "tessera.compare_scalar") {
        auto pred = op.getAttrOfType<StringAttr>("predicate");
        if (!only({"predicate", "rhs"}) || op.getNumOperands() != 1 || !pred || pred.getValue() != "gt" ||
            !op.getAttrOfType<FloatAttr>("rhs")) return reject();
      } else if (name == "tessera.masked_fill") {
        if (!only({"value"}) || op.getNumOperands() != 2 || !op.getAttrOfType<FloatAttr>("value")) return reject();
      } else if (name == "tessera.relu") {
        if (!only({}) || op.getNumOperands() != 1) return reject();
      } else if (name == "tessera.mul" || name == "tessera.sub" || name == "tessera.add") {
        if (!only({}) || op.getNumOperands() != 2) return reject();
      } else return reject();
      if (auto axis = op.getAttrOfType<IntegerAttr>("axis"); axis && axis.getInt() != 1 && axis.getInt() != -1) return reject();
      if (auto eps = op.getAttrOfType<FloatAttr>("eps"); eps && (!std::isfinite(float(eps.getValueAsDouble())) || float(eps.getValueAsDouble()) <= 0)) return reject();
    }
    return success();
  }
  FailureOr<func::FuncOp> lower(ModuleOp module) {
    if (failed(validate())) return failure();
    b.setInsertionPointToEnd(module.getBody());
    auto ptr = LLVM::LLVMPointerType::get(b.getContext());
    auto fn = func::FuncOp::create(b, loc, "native_row", b.getFunctionType(
        {ptr, ptr, ptr, ptr, b.getI64Type()}, {}));
    Block *body = fn.addEntryBlock();
    b.setInsertionPointToStart(body);
    pointers.assign(body->getArguments().begin(), body->getArguments().begin() + 4);
    row = body->getArgument(4);
    width = arith::ConstantIntOp::create(b, loc, tensor.getDimSize(1), 64);
    Value rows = arith::ConstantIntOp::create(b, loc, tensor.getDimSize(0), 64);
    // Unsigned test also excludes negative rows before any buffer access.
    for (unsigned i=0;i<4;++i) {
      auto viewType = tile::BufferViewType::get(b.getContext(),tensor.getElementType(),i<2?"read":"write","row_major");
      views.push_back(tile::BufferViewOp::create(b,loc,viewType,pointers[i],rows,width,width));
    }
    Value active = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, row, rows);
    auto guarded = scf::IfOp::create(b, loc, active, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&guarded.getThenRegion().front());
      for (Operation &op : graph.getBody().front().without_terminator()) prepare(&op);
      auto zero = arith::ConstantIntOp::create(b, loc, 0, 64);
      auto one = arith::ConstantIntOp::create(b, loc, 1, 64);
      auto loop = scf::ForOp::create(b, loc, zero, width, one);
      b.setInsertionPointToStart(loop.getBody());
      Cache cache;
      for (auto [index, output] : llvm::enumerate(graph.getBody().front().getTerminator()->getOperands())) {
        Value value = scalar(output, loop.getInductionVar(), cache);
        tile::StoreElementOp::create(b,loc,views[index+2],row,loop.getInductionVar(),value);
      }
    }
    b.setInsertionPointToEnd(body);
    func::ReturnOp::create(b, loc);
    return fn;
  }
};

} // namespace tessera::row_region
