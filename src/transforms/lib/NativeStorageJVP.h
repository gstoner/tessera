// Physical child generation from the actual C++ forward-product SSA graph.
#ifndef TESSERA_NATIVE_STORAGE_JVP_H
#define TESSERA_NATIVE_STORAGE_JVP_H
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

namespace tessera {
inline mlir::LogicalResult emitNativeStorageJVP(mlir::ModuleOp module, bool reverse = false) {
  if (reverse) {
    llvm::SmallVector<mlir::func::FuncOp> backwards;
    for (auto fn : module.getOps<mlir::func::FuncOp>())
      if (auto role = fn->getAttrOfType<mlir::StringAttr>("tessera.autodiff.role"); role && role.getValue() == "backward") backwards.push_back(fn);
    if (backwards.size() != 1) return module.emitError("native VJP requires one compiler backward function");
    auto bwd = backwards.front();
    auto ref = bwd->getAttrOfType<mlir::FlatSymbolRefAttr>("tessera.autodiff.forward");
    auto fwd = ref ? module.lookupSymbol<mlir::func::FuncOp>(ref.getValue()) : mlir::func::FuncOp();
    if (!fwd || fwd.getNumArguments() != 1 || fwd.getNumResults() < 1 || bwd.getNumArguments() != fwd.getNumResults() + 1 ||
        bwd.getNumResults() != 1 || !fwd.getBody().hasOneBlock() || !bwd.getBody().hasOneBlock())
      return module.emitError("native VJP requires one input/primal and an explicit straight-line residual ABI");
    auto residuals = fwd->getAttrOfType<mlir::ArrayAttr>("tessera.autodiff.residual_sources");
    if (fwd.getNumResults() != 1 + (residuals ? residuals.size() : 0))
      return module.emitError("native VJP requires exactly one primal result before saved residuals");
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    auto pairType = builder.getFunctionType({bwd.getArgumentTypes()[0], bwd.getArgumentTypes()[1]}, {fwd.getResultTypes()[0], bwd.getResultTypes()[0]});
    auto combined = builder.create<mlir::func::FuncOp>(fwd.getLoc(), "__native_vjp_pair", pairType);
    combined->setAttr("tessera.autodiff.role", builder.getStringAttr("vjp"));
    combined->setAttr("tessera.autodiff.forward", ref);
    auto *body = combined.addEntryBlock();
    builder.setInsertionPointToStart(body);
    mlir::IRMapping fm, bm;
    fm.map(fwd.getArgument(0), body->getArgument(0));
    bm.map(bwd.getArgument(0), body->getArgument(0));
    bm.map(bwd.getArgument(1), body->getArgument(1));
    for (auto &op : fwd.getBody().front().without_terminator()) builder.clone(op, fm);
    for (unsigned i = 1; i < fwd.getNumResults(); ++i) {
      auto saved = fm.lookup(fwd.getBody().front().getTerminator()->getOperand(i));
      if (saved.getType() != bwd.getArgument(i + 1).getType())
        return module.emitError("native reverse residual type disagrees");
      bm.map(bwd.getArgument(i + 1), saved);
    }
    for (auto &op : bwd.getBody().front().without_terminator()) builder.clone(op, bm);
    auto primal = fm.lookup(fwd.getBody().front().getTerminator()->getOperand(0));
    auto cotangent = bm.lookup(bwd.getBody().front().getTerminator()->getOperand(0));
    builder.create<mlir::func::ReturnOp>(fwd.getLoc(), mlir::ValueRange{primal, cotangent});
  }
  if (reverse) for (auto fn : module.getOps<mlir::func::FuncOp>())
    if (fn.getName() == "__native_vjp_pair") fn.walk([](mlir::Operation *op) {
      for (auto name : {"tessera.autodiff.checkpoint_policy", "tessera.autodiff.residual_materialized",
                        "tessera.autodiff.residual_result_indices", "tessera.autodiff.residual_owner"})
        op->removeAttr(name);
    });
  llvm::SmallVector<mlir::func::FuncOp> pairs;
  for (auto fn : module.getOps<mlir::func::FuncOp>())
    if (auto role = fn->getAttrOfType<mlir::StringAttr>("tessera.autodiff.role");
        role && role.getValue() == (reverse ? "vjp" : "jvp")) pairs.push_back(fn);
  auto reject = [&]() {
    return module.emitError("native storage JVP requires one straight-line, equal-shape rank-one f32 arithmetic/sigmoid/tanh/stop-gradient pair (1..1024 elements)");
  };
  if (pairs.size() != 1) return reject();
  auto pair = pairs.front();
  if (!pair.getBody().hasOneBlock() || pair.getNumResults() != 2 || !pair.getNumArguments()) return reject();
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(pair.getArgument(0).getType());
  if (!type || type.getRank() != 1 || !type.hasStaticShape() || !type.getElementType().isF32() ||
      type.getDimSize(0) < 1 || type.getDimSize(0) > 1024) return reject();
  auto widthOf = [&](mlir::Type t) -> int64_t {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(t);
    if (!ranked || !ranked.hasStaticShape() || !ranked.getElementType().isF32()) return 0;
    if (t == type) return type.getDimSize(0);
    return ranked.getNumElements() == 1 ? 1 : 0;
  };
  for (auto t : pair.getArgumentTypes()) if (!widthOf(t)) return reject();
  for (auto t : pair.getResultTypes()) if (!widthOf(t)) return reject();
  auto forwardRef = pair->getAttrOfType<mlir::FlatSymbolRefAttr>("tessera.autodiff.forward");
  auto forward = forwardRef ? module.lookupSymbol<mlir::func::FuncOp>(forwardRef.getValue()) : mlir::func::FuncOp();
  if (!forward) return reject();
  std::string pairedText;
  llvm::raw_string_ostream pairedStream(pairedText);
  module.print(pairedStream);
  llvm::DenseMap<mlir::Value, std::string> values;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "module { gpu.module @native_jvp { gpu.func @paired_child(";
  unsigned count = pair.getNumArguments();
  for (unsigned i = 0; i < count + 2; ++i) os << "%p" << i << ": !llvm.ptr<1>, ";
  os << "%n: index) kernel {\n"
        "%tid = gpu.thread_id x\n%i = arith.index_cast %tid : index to i64\n";
  for (unsigned i = 0; i < count; ++i) {
    os << "%ptr" << i << " = llvm.getelementptr %p" << i << "[" << (widthOf(pair.getArgument(i).getType()) == 1 ? "0" : "%i") << "] : (!llvm.ptr<1>" << (widthOf(pair.getArgument(i).getType()) == 1 ? "" : ", i64") << ") -> !llvm.ptr<1>, f32\n"
       << "%v" << i << " = llvm.load %ptr" << i << " : !llvm.ptr<1> -> f32\n";
    values[pair.getArgument(i)] = "%v" + std::to_string(i);
  }
  unsigned next = count;
  for (auto &op : pair.getBody().front().without_terminator()) {
    auto name = op.getName().getStringRef();
    if (op.getNumResults() != 1 || !widthOf(op.getResult(0).getType()) ||
        op.getNumRegions()) return reject();
    std::string value = "%v" + std::to_string(next++);
    if (name == "tessera.unsqueeze" || name == "tessera.broadcast") {
      if (op.getNumOperands() != 1 || widthOf(op.getOperand(0).getType()) != 1 ||
          !values.count(op.getOperand(0))) return reject();
      if (name == "tessera.unsqueeze" && widthOf(op.getResult(0).getType()) != 1) return reject();
      values[op.getResult(0)] = values.lookup(op.getOperand(0));
      continue;
    }
    if (auto add = mlir::dyn_cast<mlir::arith::AddFOp>(op)) {
      if (add.getFastmath() != mlir::arith::FastMathFlags::none || !values.count(add.getLhs()) || !values.count(add.getRhs())) return reject();
      os << value << " = arith.addf " << values.lookup(add.getLhs()) << ", " << values.lookup(add.getRhs()) << " : f32\n";
      values[op.getResult(0)] = value;
      continue;
    }
    if (name == "tessera.reduce") {
      auto kind = op.getAttrOfType<mlir::StringAttr>("kind");
      auto axis = op.getAttrOfType<mlir::IntegerAttr>("axis");
      auto rt = mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
      int64_t width = type.getDimSize(0);
      if (!kind || (kind.getValue() != "sum" && kind.getValue() != "mean") || !axis ||
          (axis.getInt() != 0 && axis.getInt() != -1) || !rt || rt.getNumElements() != 1 ||
          op.getNumOperands() != 1 || op.getOperand(0).getType() != type ||
          !values.count(op.getOperand(0)) || width < 2 || (width & (width - 1))) return reject();
      for (auto attr : op.getAttrs())
        if (attr.getName() != "kind" && attr.getName() != "axis" && attr.getName() != "keepdims" &&
            attr.getName() != "tessera.autodiff.role" && attr.getName() != "tessera.autodiff.activity" &&
            attr.getName() != "tessera.effect_kind") return reject();
      auto p = "%reduce" + std::to_string(next++);
      os << p << " = memref.alloca(%n) : memref<?xf32>\n"
         << "\"tile.alloc_shared\"(" << p << ") : (memref<?xf32>) -> ()\n"
         << "memref.store " << values.lookup(op.getOperand(0)) << ", " << p << "[%tid] : memref<?xf32>\n"
         << "gpu.barrier\n" << p << "zero = arith.constant 0 : index\n";
      for (int64_t stride = width / 2; stride; stride /= 2) {
        auto q = p + "step" + std::to_string(stride);
        os << q << " = arith.constant " << stride << " : index\n"
           << q << "active = arith.cmpi ult, %tid, " << q << " : index\n"
           << "scf.if " << q << "active {\n"
           << q << "other = arith.addi %tid, " << q << " : index\n"
           << q << "a = memref.load " << p << "[%tid] : memref<?xf32>\n"
           << q << "b = memref.load " << p << "[" << q << "other] : memref<?xf32>\n"
           << q << "sum = arith.addf " << q << "a, " << q << "b : f32\n"
           << "memref.store " << q << "sum, " << p << "[%tid] : memref<?xf32>\n}\ngpu.barrier\n";
      }
      os << p << "sum = memref.load " << p << "[" << p << "zero] : memref<?xf32>\ngpu.barrier\n";
      if (kind.getValue() == "mean") {
        os << p << "width = arith.constant " << width << ".0 : f32\n"
           << value << " = arith.divf " << p << "sum, " << p << "width : f32\n";
      } else value = p + "sum";
      values[op.getResult(0)] = value;
      continue;
    }
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      auto dense = mlir::dyn_cast<mlir::DenseFPElementsAttr>(constant.getValue());
      if (!dense || !dense.isSplat()) return reject();
      // Preserve the exact f32 bit pattern, including signed zero. Constants
      // belong to the compiler's paired graph (e.g. inactive tangents).
      os << value << " = arith.constant ";
      mlir::FloatAttr::get(type.getElementType(), dense.getSplatValue<llvm::APFloat>()).print(os);
      os << "\n";
    } else {
      for (auto attr : op.getAttrs())
        if (attr.getName() != "tessera.autodiff.role" && attr.getName() != "tessera.autodiff.activity" &&
            !(attr.getName() == "tessera.effect_kind" && attr.getValue() == mlir::StringAttr::get(module.getContext(), "pure"))) return reject();
      for (auto operand : op.getOperands())
        if (!values.count(operand)) return reject();
      if (name == "tessera.stop_gradient" && op.getNumOperands() == 1) {
        // The primal is identity; the forward pass already produced its zero
        // tangent. Never reconstruct a derivative here.
        values[op.getResult(0)] = values.lookup(op.getOperand(0));
        continue;
      }
      if ((name == "tessera.sigmoid" || name == "tessera.tanh") && op.getNumOperands() == 1) {
        auto input = values.lookup(op.getOperand(0));
        // Evaluate exp only on a nonpositive argument. This avoids overflow
        // in either saturation tail; the paired SSA still owns derivatives.
        auto suffix = std::to_string(next++);
        auto prefix = "%nl" + suffix;
        os << prefix << "a = math.absf " << input << " : f32\n"
           << prefix << "k = arith.constant " << (name == "tessera.tanh" ? "-2.88539004" : "-1.44269502") << " : f32\n"
           << prefix << "x = arith.mulf " << prefix << "a, " << prefix << "k : f32\n"
           << prefix << "e = math.exp2 " << prefix << "x : f32\n"
           << prefix << "one = arith.constant 1.0 : f32\n"
           << prefix << "den = arith.addf " << prefix << "one, " << prefix << "e : f32\n";
        if (name == "tessera.tanh") {
          os << prefix << "num = arith.subf " << prefix << "one, " << prefix << "e : f32\n"
             << prefix << "r = arith.divf " << prefix << "num, " << prefix << "den : f32\n"
             // The exp ratio cancels near zero. Use the odd Taylor polynomial
             // only for |x| < 1/8, where the omitted term is below f32 error.
             << prefix << "x2 = arith.mulf " << input << ", " << input << " : f32\n"
             << prefix << "c7 = arith.constant -0.0539682540 : f32\n"
             << prefix << "c5 = arith.constant 0.133333340 : f32\n"
             << prefix << "c3 = arith.constant -0.333333343 : f32\n"
             << prefix << "p7 = arith.mulf " << prefix << "x2, " << prefix << "c7 : f32\n"
             << prefix << "p5 = arith.addf " << prefix << "p7, " << prefix << "c5 : f32\n"
             << prefix << "q5 = arith.mulf " << prefix << "x2, " << prefix << "p5 : f32\n"
             << prefix << "p3 = arith.addf " << prefix << "q5, " << prefix << "c3 : f32\n"
             << prefix << "q3 = arith.mulf " << prefix << "x2, " << prefix << "p3 : f32\n"
             << prefix << "scale = arith.addf " << prefix << "one, " << prefix << "q3 : f32\n"
             << prefix << "poly = arith.mulf " << input << ", " << prefix << "scale : f32\n"
             << prefix << "cutoff = arith.constant 0.125 : f32\n"
             << prefix << "small = arith.cmpf olt, " << prefix << "a, " << prefix << "cutoff : f32\n"
             << prefix << "selected = arith.select " << prefix << "small, " << prefix << "poly, " << prefix << "r : f32\n"
             << value << " = math.copysign " << prefix << "selected, " << input << " : f32\n";
        } else {
          os << prefix << "pos = arith.divf " << prefix << "one, " << prefix << "den : f32\n"
             << prefix << "neg = arith.divf " << prefix << "e, " << prefix << "den : f32\n"
             << prefix << "zero = arith.constant 0.0 : f32\n"
             << prefix << "sign = arith.cmpf olt, " << input << ", " << prefix << "zero : f32\n"
             << value << " = arith.select " << prefix << "sign, " << prefix << "neg, " << prefix << "pos : f32\n";
        }
        values[op.getResult(0)] = value;
        continue;
      }
      if ((name != "tessera.add" && name != "tessera.sub" && name != "tessera.mul") ||
          op.getNumOperands() != 2) return reject();
      os << value << " = arith." << (name == "tessera.add" ? "addf " : name == "tessera.sub" ? "subf " : "mulf ")
         << values.lookup(op.getOperand(0)) << ", " << values.lookup(op.getOperand(1)) << " : f32\n";
    }
    values[op.getResult(0)] = value;
  }
  auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(pair.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands() != 2) return reject();
  for (unsigned i = 0; i < 2; ++i) {
    if (!values.count(ret.getOperand(i))) return reject();
    if (widthOf(ret.getOperand(i).getType()) == 1) {
      os << "%outzero" << i << " = arith.constant 0 : index\n"
         << "%first" << i << " = arith.cmpi eq, %tid, %outzero" << i << " : index\n"
         << "scf.if %first" << i << " {\nllvm.store " << values.lookup(ret.getOperand(i))
         << ", %p" << count+i << " : f32, !llvm.ptr<1>\n}\n";
      continue;
    }
    os << "%s" << i << " = memref.alloca(%n) : memref<?xf32>\n"
       << "\"tile.alloc_shared\"(%s" << i << ") : (memref<?xf32>) -> ()\n"
       << "memref.store " << values.lookup(ret.getOperand(i)) << ", %s" << i << "[%tid] : memref<?xf32>\n"
       << "gpu.barrier\n%r" << i << " = memref.load %s" << i << "[%tid] : memref<?xf32>\n"
       << "gpu.barrier\n%o" << i << " = llvm.getelementptr %p" << count+i << "[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32\n"
       << "llvm.store %r" << i << ", %o" << i << " : f32, !llvm.ptr<1>\n";
  }
  os << "gpu.return\n} } }\n";
  auto generated = mlir::parseSourceString<mlir::ModuleOp>(text, module.getContext());
  if (!generated) return mlir::failure();
  mlir::Builder b(module.getContext());
  module->setAttr("tessera.native_jvp_pair", b.getStringAttr(pairedText));
  module->setAttr("tessera.native_jvp_width", b.getI64IntegerAttr(type.getDimSize(0)));
  module->setAttr("tessera.native_jvp_output_width", b.getI64IntegerAttr(widthOf(pair.getResultTypes()[0])));
  module->setAttr("tessera.native_jvp_inputs", b.getI64IntegerAttr(count));
  llvm::SmallVector<mlir::Attribute> inputWidths, outputWidths;
  for (auto t : pair.getArgumentTypes()) inputWidths.push_back(b.getI64IntegerAttr(widthOf(t)));
  for (auto t : pair.getResultTypes()) outputWidths.push_back(b.getI64IntegerAttr(widthOf(t)));
  module->setAttr("tessera.native_jvp_input_widths", b.getArrayAttr(inputWidths));
  module->setAttr("tessera.native_jvp_output_widths", b.getArrayAttr(outputWidths));
  llvm::SmallVector<mlir::Attribute> wrt;
  if (auto indices = forward->getAttrOfType<mlir::ArrayAttr>("tessera.autodiff.wrt_indices"))
    wrt.append(indices.begin(), indices.end());
  else for (unsigned i = 0; i < forward.getNumArguments(); ++i) wrt.push_back(b.getI64IntegerAttr(i));
  module->setAttr("tessera.native_jvp_wrt", b.getArrayAttr(wrt));
  if (reverse) {
    for (auto field : {"pair", "width", "output_width", "input_widths", "output_widths", "inputs", "wrt"}) {
      auto oldName = std::string("tessera.native_jvp_") + field;
      module->setAttr(std::string("tessera.native_vjp_") + field, module->getAttr(oldName));
      module->removeAttr(oldName);
    }
  }
  module.getBody()->clear();
  module.getBody()->getOperations().splice(module.getBody()->end(), generated->getBody()->getOperations());
  return mlir::success();
}
}
#endif
