//===- RowProgramToGPUPass.cpp ---------------------------------*- C++ -*-===//
//
// RowProgramToGPUPass (2026-09-16, EBM_NATIVE_LOOP_ARCHITECTURE.md slice G2):
// a *row program* -- a tensor-level function over `[rows, features]` tensors
// built from parallel linalg.generic bodies, single-axis linalg.reduce over
// the feature axis, small uniform integer vectors (RNG keys) and scf.for
// loops carrying those tensors -- becomes one cooperative GPU kernel:
//
//   * one block per row, one lane per feature (block = next power of two
//     >= features, inactive lanes masked);
//   * every `[rows, features]` value is one f32 register per lane, every
//     `[rows]` / `[rows, 1]` value is a block-uniform scalar, every
//     `[n] xi64` value (n <= 8) is n uniform scalars, every non-tensor value
//     is cloned as is;
//   * a linalg.generic body is cloned once per lane (or once per block for
//     row results) with `linalg.index 0/1` mapped to (row, lane);
//   * a linalg.reduce over the feature axis is an *ordered* reduction:
//     lanes write to workgroup memory, a barrier, the block leader folds the
//     combiner body over features 0..F-1 in order, a barrier, all lanes read
//     the result -- the same order the CPU lane's sequential loop uses, so
//     reduced quantities agree to the same rounding on both;
//   * scf.for loops carry their tensors as per-lane / uniform registers, so a
//     K-step sampling loop never leaves the kernel.
//
// The kernel uses the native GPU storage ABI (`!llvm.ptr<1>` per tensor plus
// one index scratch, `tile.alloc_shared` marker for the arena sizer), so
// `build_native_gpu_storage` packages it for gfx1151/gfx1201/sm_120 and
// `replay_arena_ir` validates it. The input module is retained verbatim in
// `tessera.row_program.source` for that replay.
//
// This is the device route for the EBM Langevin loop (the compiler-derived
// gradient is exactly such a row program after tessera-to-linalg) and for any
// future row-wise program; nothing in it knows about energies.
//
// Fails closed on anything outside the contract (dynamic shapes, calls,
// tensors of other ranks, non-parallel generics, reductions over other axes,
// unsupported ops) with a diagnostic naming the op.
//
//===----------------------------------------------------------------------===//
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/IR/TesseraOps.h"
#include "Tessera/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace tessera {
namespace {

enum class Kind { Lane, Row, Uniform, Scalar, None };

struct Slot {
  Kind kind = Kind::None;
  SmallVector<Value, 4> values;  // 1 for Lane/Row/Scalar, n for Uniform
};

struct Emitter {
  ModuleOp module;
  func::FuncOp entry;
  OpBuilder b;
  Location loc;
  int64_t rows = 0, feats = 0, lanes = 0;
  Value row, lane, active, leader, shared, fzero;
  SmallVector<Value> pointers;  // one per entry argument then per result
  DenseMap<Value, Slot> slots;
  LLVM::LLVMArrayType sharedTy;
  bool broken = false;

  Emitter(ModuleOp m, func::FuncOp f) : module(m), entry(f), b(m.getContext()), loc(f.getLoc()) {}

  InFlightDiagnostic fail(Operation *op, const Twine &msg) {
    broken = true;
    return op->emitError("row program: ") << msg;
  }

  // --- classification ----------------------------------------------------
  Kind classify(Type type, Operation *at) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor) return Kind::Scalar;
    if (!tensor.hasStaticShape()) { fail(at, "dynamic tensor shapes are not admitted"); return Kind::None; }
    Type elem = tensor.getElementType();
    if (tensor.getRank() == 0) return Kind::Scalar;
    if (tensor.getRank() == 2 && elem.isF32()) {
      if (tensor.getDimSize(0) == rows && tensor.getDimSize(1) == feats) return Kind::Lane;
      if (tensor.getDimSize(0) == rows && tensor.getDimSize(1) == 1) return Kind::Row;
      fail(at, "rank-2 tensors must be [rows, features] or [rows, 1]");
      return Kind::None;
    }
    if (tensor.getRank() == 1) {
      if (elem.isF32() && tensor.getDimSize(0) == rows) return Kind::Row;
      if (elem.isInteger() && tensor.getDimSize(0) >= 1 && tensor.getDimSize(0) <= 8) return Kind::Uniform;
      fail(at, "rank-1 tensors must be f32 [rows] or an integer vector of at most 8 elements");
      return Kind::None;
    }
    fail(at, "unsupported tensor type");
    return Kind::None;
  }

  Slot &slot(Value v, Operation *at) {
    auto it = slots.find(v);
    if (it == slots.end()) {
      fail(at, "value has no device mapping (produced by an unsupported op?)");
      static Slot dead;
      return dead;
    }
    return it->second;
  }

  // Scalar broadcast of a slot for use inside a body: Lane/Row/Scalar -> the one value.
  Value scalarOf(Value v, Operation *at) {
    Slot &s = slot(v, at);
    if (s.values.size() != 1) { fail(at, "expected a single-valued operand"); return {}; }
    return s.values[0];
  }

  // --- device helpers ----------------------------------------------------
  Value idx(int64_t v) { return arith::ConstantIndexOp::create(b, loc, v); }
  Value i64(Value index) { return arith::IndexCastOp::create(b, loc, b.getI64Type(), index); }
  Value gep(Value base, Type elem, Value index64) {
    return LLVM::GEPOp::create(b, loc, base.getType(), elem, base, ValueRange{index64});
  }
  Value load(Value base, Type elem, Value index) { return LLVM::LoadOp::create(b, loc, elem, gep(base, elem, i64(index))); }
  void store(Value value, Value base, Value index) {
    LLVM::StoreOp::create(b, loc, value, gep(base, value.getType(), i64(index)));
  }
  Value laneIndex() {  // row * feats + lane
    return arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, row, idx(feats)), lane);
  }
  Value sharedPtr(Value index) {
    Value zero = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(0));
    return LLVM::GEPOp::create(b, loc, shared.getType(), sharedTy, shared, ValueRange{zero, i64(index)});
  }
  Value zeroOf(Type t) {
    if (auto ft = dyn_cast<FloatType>(t)) return arith::ConstantOp::create(b, loc, b.getFloatAttr(ft, 0.0));
    return arith::ConstantOp::create(b, loc, b.getIntegerAttr(t, 0));
  }

  // Load an argument tensor into a slot.
  Slot loadArgument(Value pointer, Type type, Operation *at) {
    Slot s;
    s.kind = classify(type, at);
    auto tensor = dyn_cast<RankedTensorType>(type);
    Type elem = tensor ? tensor.getElementType() : type;
    switch (s.kind) {
    case Kind::Lane: {
      auto guarded = scf::IfOp::create(b, loc, TypeRange{elem}, active, /*withElse=*/true);
      {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(&guarded.getThenRegion().front());
        scf::YieldOp::create(b, loc, load(pointer, elem, laneIndex()));
        b.setInsertionPointToStart(&guarded.getElseRegion().front());
        scf::YieldOp::create(b, loc, zeroOf(elem));
      }
      s.values.push_back(guarded.getResult(0));
      break;
    }
    case Kind::Row:
      s.values.push_back(load(pointer, elem, row));
      break;
    case Kind::Uniform:
      for (int64_t k = 0; k < tensor.getDimSize(0); ++k) s.values.push_back(load(pointer, elem, idx(k)));
      break;
    case Kind::Scalar:
      if (tensor) s.values.push_back(load(pointer, elem, idx(0)));
      else { fail(at, "non-tensor arguments are not admitted"); }
      break;
    case Kind::None:
      break;
    }
    return s;
  }

  void storeResult(Slot &s, Value pointer, Operation *at) {
    switch (s.kind) {
    case Kind::Lane: {
      auto guarded = scf::IfOp::create(b, loc, active, /*withElse=*/false);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(&guarded.getThenRegion().front());
      store(s.values[0], pointer, laneIndex());
      break;
    }
    case Kind::Row:
    case Kind::Uniform:
    case Kind::Scalar: {
      auto guarded = scf::IfOp::create(b, loc, leader, /*withElse=*/false);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(&guarded.getThenRegion().front());
      if (s.kind == Kind::Row || s.kind == Kind::Scalar) store(s.values[0], pointer, s.kind == Kind::Row ? row : idx(0));
      else for (auto [k, v] : llvm::enumerate(s.values)) store(v, pointer, idx(k));
      break;
    }
    case Kind::None:
      fail(at, "result has no device mapping");
    }
  }

  // --- ops ----------------------------------------------------------------
  // Clone a scalar body region once with `args` bound to its block arguments.
  Value cloneBody(Region &region, ArrayRef<Value> args, Operation *at, bool inLane) {
    Block &body = region.front();
    if (body.getNumArguments() != args.size()) { fail(at, "body arity disagrees"); return {}; }
    IRMapping map;
    for (auto [arg, value] : llvm::zip(body.getArguments(), args)) map.map(arg, value);
    Value result;
    for (Operation &op : body) {
      if (auto index = dyn_cast<linalg::IndexOp>(op)) {
        if (index.getDim() == 0) map.map(index.getResult(), row);
        else if (index.getDim() == 1 && inLane) map.map(index.getResult(), lane);
        else { fail(at, "linalg.index outside the (row, lane) mapping"); return {}; }
        continue;
      }
      if (auto yield = dyn_cast<linalg::YieldOp>(op)) {
        if (yield.getNumOperands() != 1) { fail(at, "single-result bodies only"); return {}; }
        result = map.lookupOrDefault(yield.getOperand(0));
        continue;
      }
      if (op.getNumRegions() != 0) { fail(&op, "nested regions inside bodies are not admitted"); return {}; }
      for (Type t : op.getResultTypes())
        if (isa<TensorType>(t)) { fail(&op, "tensor-typed ops inside bodies are not admitted"); return {}; }
      // Operands defined outside the body (hoisted constants, loop scalars)
      // resolve through their device slots, never through the source module.
      for (Value operand : op.getOperands()) {
        if (map.contains(operand)) continue;
        if (isa<TensorType>(operand.getType())) { fail(&op, "tensor operand captured inside a body"); return {}; }
        Value v = scalarOf(operand, &op);
        if (!v) return {};
        map.map(operand, v);
      }
      b.clone(op, map);
    }
    return result;
  }

  bool isIdentityMap(AffineMap m, int64_t rank) { return m.isIdentity() && m.getNumDims() == rank; }
  // (d0, d1) -> (d0, 0) or (d0, d1) -> (d0): a row-uniform read.
  bool isRowMap(AffineMap m) {
    if (m.getNumDims() != 2 || m.getNumResults() < 1 || m.getNumResults() > 2) return false;
    if (m.getResult(0) != getAffineDimExpr(0, m.getContext())) return false;
    if (m.getNumResults() == 2) {
      auto c = dyn_cast<AffineConstantExpr>(m.getResult(1));
      return c && c.getValue() == 0;
    }
    return true;
  }

  LogicalResult emitGeneric(linalg::GenericOp op) {
    if (op.getNumResults() != 1) return fail(op, "one-result generics only");
    Kind out = classify(op.getResult(0).getType(), op);
    if (out == Kind::None) return failure();
    for (auto it : op.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel) return fail(op, "reductions must be linalg.reduce, not a reduction generic");
    auto maps = op.getIndexingMapsArray();
    const int64_t rank = cast<RankedTensorType>(op.getResult(0).getType()).getRank();
    if (!isIdentityMap(maps.back(), rank)) return fail(op, "the output map must be the identity");
    const bool laneBody = out == Kind::Lane;
    auto perElement = [&](int64_t element) -> LogicalResult {
      SmallVector<Value> args;
      for (auto [i, input] : llvm::enumerate(op.getInputs())) {
        Slot &s = slot(input, op);
        AffineMap m = maps[i];
        if (s.kind == Kind::Lane) {
          if (!laneBody || !isIdentityMap(m, 2)) return fail(op, "a [rows, features] input needs a lane body and an identity map");
          args.push_back(s.values[0]);
        } else if (s.kind == Kind::Row) {
          if (!(isRowMap(m) || (rank == 1 && isIdentityMap(m, 1)) || (rank == 2 && isIdentityMap(m, 2) && !laneBody)))
            return fail(op, "a row input must be read with a row map");
          args.push_back(s.values[0]);
        } else if (s.kind == Kind::Scalar) {
          args.push_back(s.values[0]);
        } else if (s.kind == Kind::Uniform) {
          if (out != Kind::Uniform || !isIdentityMap(m, 1)) return fail(op, "uniform vectors combine only elementwise with uniform vectors");
          args.push_back(s.values[element]);
        } else return failure();
      }
      // outs block arguments: the init value if it has one, else zero.
      for (Value init : op.getOutputs()) {
        auto it = slots.find(init);
        if (it != slots.end() && !it->second.values.empty())
          args.push_back(it->second.kind == Kind::Uniform ? it->second.values[element] : it->second.values[0]);
        else args.push_back(zeroOf(cast<RankedTensorType>(init.getType()).getElementType()));
      }
      Value r = cloneBody(op.getRegion(), args, op, laneBody);
      if (!r) return failure();
      slots[op.getResult(0)].values.push_back(r);
      return success();
    };
    slots[op.getResult(0)].kind = out;
    if (out == Kind::Uniform) {
      const int64_t n = cast<RankedTensorType>(op.getResult(0).getType()).getDimSize(0);
      for (int64_t k = 0; k < n; ++k) if (failed(perElement(k))) return failure();
      return success();
    }
    return perElement(0);
  }

  LogicalResult emitReduce(linalg::ReduceOp op) {
    if (op.getNumResults() != 1 || op.getInputs().size() != 1) return fail(op, "single-input reductions only");
    auto dims = op.getDimensions();
    if (dims.size() != 1 || dims[0] != 1) return fail(op, "reductions must be over the feature axis (dimension 1)");
    Slot &in = slot(op.getInputs()[0], op);
    if (in.kind != Kind::Lane) return fail(op, "the reduced operand must be [rows, features]");
    Kind out = classify(op.getResult(0).getType(), op);
    if (out != Kind::Row) return fail(op, "the reduction result must be [rows]");
    Value init = scalarOf(op.getInits()[0], op);
    if (!init) return failure();
    Type elem = init.getType();
    // lanes -> shared memory, in order.
    {
      auto guarded = scf::IfOp::create(b, loc, active, /*withElse=*/false);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(&guarded.getThenRegion().front());
      LLVM::StoreOp::create(b, loc, in.values[0], sharedPtr(lane));
    }
    gpu::BarrierOp::create(b, loc);
    auto folded = scf::IfOp::create(b, loc, TypeRange{elem}, leader, /*withElse=*/true);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(&folded.getThenRegion().front());
      auto loop = scf::ForOp::create(b, loc, idx(0), idx(feats), idx(1), ValueRange{init});
      {
        OpBuilder::InsertionGuard g2(b);
        b.setInsertionPointToStart(loop.getBody());
        Value element = LLVM::LoadOp::create(b, loc, elem, sharedPtr(loop.getInductionVar()));
        Value acc = cloneBody(op.getRegion(), {element, loop.getRegionIterArgs()[0]}, op, /*inLane=*/false);
        if (!acc) return failure();
        scf::YieldOp::create(b, loc, acc);
      }
      LLVM::StoreOp::create(b, loc, loop.getResult(0), sharedPtr(idx(0)));
      scf::YieldOp::create(b, loc, loop.getResult(0));
      b.setInsertionPointToStart(&folded.getElseRegion().front());
      scf::YieldOp::create(b, loc, init);
    }
    gpu::BarrierOp::create(b, loc);
    Value result = LLVM::LoadOp::create(b, loc, elem, sharedPtr(idx(0)));
    gpu::BarrierOp::create(b, loc);  // shared memory is free for the next reduction
    Slot s; s.kind = Kind::Row; s.values.push_back(result);
    slots[op.getResult(0)] = s;
    return success();
  }

  LogicalResult emitFor(scf::ForOp op) {
    for (Value v : {op.getLowerBound(), op.getUpperBound(), op.getStep()})
      if (isa<TensorType>(v.getType())) return fail(op, "loop bounds must be scalars");
    SmallVector<Value> flat;
    SmallVector<Slot> layout;
    for (Value init : op.getInitArgs()) {
      Slot &s = slot(init, op);
      if (s.kind == Kind::None) return failure();
      layout.push_back(s);
      flat.append(s.values.begin(), s.values.end());
    }
    auto loop = scf::ForOp::create(b, loc, scalarOf(op.getLowerBound(), op), scalarOf(op.getUpperBound(), op),
                                   scalarOf(op.getStep(), op), flat);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(loop.getBody());
      Slot iv; iv.kind = Kind::Scalar; iv.values.push_back(loop.getInductionVar());
      slots[op.getInductionVar()] = iv;
      unsigned cursor = 0;
      for (auto [arg, shape] : llvm::zip(op.getRegionIterArgs(), layout)) {
        Slot s; s.kind = shape.kind;
        for (unsigned k = 0; k < shape.values.size(); ++k) s.values.push_back(loop.getRegionIterArgs()[cursor++]);
        slots[arg] = s;
      }
      auto yield = cast<scf::YieldOp>(op.getBody()->getTerminator());
      if (failed(emitBlock(*op.getBody(), /*terminator=*/yield))) return failure();
      SmallVector<Value> yielded;
      for (auto [v, shape] : llvm::zip(yield.getOperands(), layout)) {
        Slot &s = slot(v, op);
        if (s.kind != shape.kind || s.values.size() != shape.values.size()) return fail(op, "loop-carried value changed its mapping");
        yielded.append(s.values.begin(), s.values.end());
      }
      scf::YieldOp::create(b, loc, yielded);
    }
    unsigned cursor = 0;
    for (auto [result, shape] : llvm::zip(op.getResults(), layout)) {
      Slot s; s.kind = shape.kind;
      for (unsigned k = 0; k < shape.values.size(); ++k) s.values.push_back(loop.getResult(cursor++));
      slots[result] = s;
    }
    return success();
  }

  LogicalResult emitConstant(arith::ConstantOp op) {
    Slot s;
    if (auto dense = dyn_cast<DenseElementsAttr>(op.getValue())) {
      s.kind = classify(op.getType(), op);
      if (s.kind == Kind::None) return failure();
      Type elem = dense.getElementType();
      if (dense.isSplat()) {
        Attribute v = dense.getSplatValue<Attribute>();
        Value c = arith::ConstantOp::create(b, loc, elem, cast<TypedAttr>(v));
        const int64_t n = s.kind == Kind::Uniform ? cast<RankedTensorType>(op.getType()).getDimSize(0) : 1;
        for (int64_t k = 0; k < n; ++k) s.values.push_back(c);
      } else {
        if (s.kind != Kind::Uniform) return fail(op, "non-splat constants must be small uniform integer vectors");
        for (Attribute v : dense.getValues<Attribute>())
          s.values.push_back(arith::ConstantOp::create(b, loc, elem, cast<TypedAttr>(v)));
      }
    } else {
      s.kind = Kind::Scalar;
      s.values.push_back(b.clone(*op.getOperation())->getResult(0));
    }
    slots[op.getResult()] = s;
    return success();
  }

  LogicalResult emitOp(Operation &op) {
    if (auto c = dyn_cast<arith::ConstantOp>(op)) return emitConstant(c);
    if (isa<tensor::EmptyOp>(op)) { slots[op.getResult(0)] = Slot{classify(op.getResult(0).getType(), &op), {}}; return success(); }
    if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      Slot s; s.kind = classify(fill.getResult(0).getType(), &op);
      Value v = scalarOf(fill.getInputs()[0], &op);
      if (!v) return failure();
      const int64_t n = s.kind == Kind::Uniform ? cast<RankedTensorType>(fill.getResult(0).getType()).getDimSize(0) : 1;
      for (int64_t k = 0; k < n; ++k) s.values.push_back(v);
      slots[fill.getResult(0)] = s;
      return success();
    }
    if (auto g = dyn_cast<linalg::GenericOp>(op)) return emitGeneric(g);
    if (auto r = dyn_cast<linalg::ReduceOp>(op)) return emitReduce(r);
    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op)) {
      Slot &in = slot(op.getOperand(0), &op);
      Kind out = classify(op.getResult(0).getType(), &op);
      if (in.kind != Kind::Row || out != Kind::Row) return fail(&op, "reshapes are admitted only between [rows] and [rows, 1]");
      slots[op.getResult(0)] = in;
      return success();
    }
    if (auto extract = dyn_cast<tensor::ExtractOp>(op)) {
      Slot &in = slot(extract.getTensor(), &op);
      if (in.kind != Kind::Uniform || extract.getIndices().size() != 1) return fail(&op, "tensor.extract is admitted on uniform vectors only");
      auto c = extract.getIndices()[0].getDefiningOp<arith::ConstantIndexOp>();
      if (!c || c.value() < 0 || c.value() >= (int64_t)in.values.size()) return fail(&op, "uniform vector extracts need a constant in-range index");
      Slot s; s.kind = Kind::Scalar; s.values.push_back(in.values[c.value()]);
      slots[extract.getResult()] = s;
      return success();
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) return emitFor(loop);
    if (isa<func::CallOp>(op)) return fail(&op, "calls must be inlined before the row-program lowering");
    if (op.getNumRegions() != 0) return fail(&op, "unsupported region-carrying op");
    // Scalar op: clone with its operands' single values.
    IRMapping map;
    for (Value operand : op.getOperands()) {
      if (isa<TensorType>(operand.getType())) return fail(&op, "tensor operand on an unsupported op");
      Value v = scalarOf(operand, &op);
      if (!v) return failure();
      map.map(operand, v);
    }
    for (Type t : op.getResultTypes())
      if (isa<TensorType>(t)) return fail(&op, "tensor result on an unsupported op");
    Operation *cloned = b.clone(op, map);
    for (auto [r, c] : llvm::zip(op.getResults(), cloned->getResults())) {
      Slot s; s.kind = Kind::Scalar; s.values.push_back(c);
      slots[r] = s;
    }
    return success();
  }

  LogicalResult emitBlock(Block &block, Operation *terminator) {
    for (Operation &op : block) {
      if (&op == terminator) break;
      if (failed(emitOp(op)) || broken) return failure();
    }
    return success();
  }

  LogicalResult run(StringRef backend) {
    // Shape from the first [rows, features] argument.
    for (Type t : entry.getArgumentTypes())
      if (auto tt = dyn_cast<RankedTensorType>(t))
        if (tt.getRank() == 2 && tt.hasStaticShape() && tt.getElementType().isF32()) { rows = tt.getDimSize(0); feats = tt.getDimSize(1); break; }
    if (rows <= 0 || feats <= 0) return entry.emitError("row program: no [rows, features] f32 argument");
    if (feats > 1024) return entry.emitError("row program: at most 1024 features per row (one lane each)");
    lanes = 1; while (lanes < feats) lanes *= 2;
    if (!entry.getBody().hasOneBlock()) return entry.emitError("row program: the entry must have one block");
    // Calls are inlined upstream: `--inline` after `--tessera-to-linalg` (the
    // Tessera dialect has no inliner interface, so inlining before the linalg
    // lowering is refused by the inliner; measured 2026-09-16).
    bool calls = false;
    entry.walk([&](func::CallOp) { calls = true; });
    if (calls) return entry.emitError("row program: inline every call before lowering (run --inline after --tessera-to-linalg)");

    // Retain the source for replay, then build the kernel skeleton textually
    // (the same shape the SSD and native-tape routes use) and fill its body.
    std::string source; { llvm::raw_string_ostream os(source); module.print(os); }
    const unsigned count = entry.getNumArguments() + entry.getNumResults();
    std::string text; llvm::raw_string_ostream s(text);
    s << "module attributes {tessera.row_program.source = ";
    StringAttr::get(module.getContext(), source).print(s);
    s << ", tessera.row_program.entry = \"" << entry.getName() << "\", tessera.row_program.rows = " << rows
      << " : i64, tessera.row_program.features = " << feats << " : i64, tessera.row_program.backend = \"" << backend
      << "\", tessera.autodiff.temporary_bytes = " << lanes * 8 << " : i64} {\n gpu.module @native_row {\n";
    s << "  llvm.mlir.global private @row_reduction() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<" << lanes << " x f32>\n";
    s << "  gpu.func @row_program(";
    for (unsigned i = 0; i < count; ++i) s << "%p" << i << ": !llvm.ptr<1>, ";
    s << "%scratch: index) kernel attributes {known_block_size = array<i32: " << lanes << ", 1, 1>} {\n";
    s << "   %marker = memref.alloca(%scratch) : memref<?xf32>\n   \"tile.alloc_shared\"(%marker) : (memref<?xf32>) -> ()\n";
    s << "   gpu.return\n  }\n }\n}\n";
    s.flush();
    OwningOpRef<ModuleOp> lowered = parseSourceString<ModuleOp>(text, module.getContext());
    if (!lowered) return entry.emitError("row program: kernel skeleton failed to parse");
    gpu::GPUFuncOp kernel;
    lowered->walk([&](gpu::GPUFuncOp f) { kernel = f; });
    Operation *ret = kernel.getBody().front().getTerminator();
    b.setInsertionPoint(ret);
    for (unsigned i = 0; i < count; ++i) pointers.push_back(kernel.getArgument(i));
    sharedTy = LLVM::LLVMArrayType::get(b.getF32Type(), lanes);
    row = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
    lane = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
    active = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, lane, idx(feats));
    leader = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, idx(0));
    LLVM::GlobalOp global;
    lowered->walk([&](LLVM::GlobalOp g) { global = g; });
    shared = LLVM::AddressOfOp::create(b, loc, global);
    fzero = zeroOf(b.getF32Type());
    for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
      Slot s = loadArgument(pointers[i], arg.getType(), entry);
      if (broken || s.kind == Kind::None) return failure();
      slots[arg] = s;
    }
    auto terminator = cast<func::ReturnOp>(entry.getBody().front().getTerminator());
    if (failed(emitBlock(entry.getBody().front(), terminator))) return failure();
    for (auto [r, result] : llvm::enumerate(terminator.getOperands())) {
      Slot &s = slot(result, terminator);
      storeResult(s, pointers[entry.getNumArguments() + r], terminator);
      if (broken) return failure();
    }
    // Replace the module contents with the kernel module.
    module->setAttrs((*lowered)->getAttrs());
    module.getBody()->clear();
    module.getBody()->getOperations().splice(module.getBody()->end(), lowered->getBody()->getOperations());
    return success();
  }
};

struct RowProgramToGPUPass : public PassWrapper<RowProgramToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RowProgramToGPUPass)
  RowProgramToGPUPass() = default;
  RowProgramToGPUPass(const RowProgramToGPUPass &other) : PassWrapper(other) {}
  Option<std::string> backend{*this, "backend", llvm::cl::desc("nvidia or rocm"), llvm::cl::init("nvidia")};
  Option<std::string> entryName{*this, "entry", llvm::cl::desc("The row-program entry function (default: the only function)"), llvm::cl::init("")};
  StringRef getArgument() const final { return "tessera-row-program-to-gpu"; }
  StringRef getDescription() const final {
    return "Lower a [rows, features] tensor row program (parallel linalg bodies, feature-axis "
           "reductions, uniform integer vectors, scf.for) to one cooperative GPU kernel in the "
           "native storage ABI: one block per row, one lane per feature, ordered reductions.";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    // The emitted skeleton names `tile.alloc_shared` (the arena sizer marker):
    // the tile dialect must be declared here or an assertions-enabled MLIR
    // refuses to load it inside the pass manager (Tajasarus, 2026-09-16).
    // The provenance attributes are tessera.*-prefixed: parsing them loads the
    // Tessera dialect, which an input without Tessera ops never loaded.
    r.insert<tessera::TesseraDialect, tessera::tile::TesseraTileDialect, arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
             LLVM::LLVMDialect, linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
             tensor::TensorDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (backend != "nvidia" && backend != "rocm") { module.emitError("row program: backend must be nvidia or rocm"); return signalPassFailure(); }
    func::FuncOp entry;
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    if (entryName.empty()) {
      if (functions.size() != 1) { module.emitError("row program: name the entry when the module has several functions"); return signalPassFailure(); }
      entry = functions[0];
    } else {
      for (auto f : functions) if (f.getName() == entryName) entry = f;
      if (!entry) { module.emitError("row program: entry function not found: ") << entryName; return signalPassFailure(); }
    }
    Emitter emitter(module, entry);
    if (failed(emitter.run(backend))) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createRowProgramToGPUPass() { return std::make_unique<RowProgramToGPUPass>(); }

}  // namespace tessera
