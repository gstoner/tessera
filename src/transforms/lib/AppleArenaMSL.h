// Bounded, typed GPU arena -> MSL materialization. No Graph reconstruction.
#ifndef TESSERA_APPLE_ARENA_MSL_H
#define TESSERA_APPLE_ARENA_MSL_H
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>

namespace tessera {
class AppleArenaMSL {
  llvm::DenseMap<mlir::Value, std::string> names;
  unsigned serial = 0;
  std::string text;
  llvm::raw_string_ostream os{text};
  std::string name(mlir::Value v) { return names.lookup(v); }
  std::string fresh(mlir::Value v) {
    return names[v] = "v" + std::to_string(serial++);
  }
  std::string type(mlir::Type t) {
    if (t.isF32()) return "float";
    if (t.isIndex() || t.isInteger(64)) return "long";
    if (t.isInteger(32)) return "int";
    if (t.isInteger(1)) return "bool";
    return "";
  }
  bool emit(mlir::Operation &op) {
    using namespace mlir;
    auto opname = op.getName().getStringRef();
    // Dead descriptor allocations left behind by arena replacement carry no
    // payload and must not become private/thread-local scratch in MSL.
    if (isa<memref::AllocaOp>(op) && op.use_empty()) return true;
    for (Value v : op.getOperands()) if (!names.count(v)) return false;
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      auto t = type(c.getType());
      if (t.empty()) return false;
      os << t << " " << fresh(c.getResult()) << " = ";
      if (auto i = dyn_cast<IntegerAttr>(c.getValue())) os << i.getInt();
      else if (auto f = dyn_cast<FloatAttr>(c.getValue())) {
        // Hex float literals preserve the exact f32 constant.
        char buffer[128];
        snprintf(buffer, sizeof(buffer), "%a", f.getValueAsDouble());
        os << buffer << "f";
      } else return false;
      os << ";\n"; return true;
    }
    if (isa<gpu::ThreadIdOp, gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp>(op)) {
      auto dim = op.getAttrOfType<gpu::DimensionAttr>("dimension");
      if (!dim) return false;
      auto builtin = isa<gpu::ThreadIdOp>(op) ? "tid" : isa<gpu::BlockIdOp>(op) ? "bid" : isa<gpu::BlockDimOp>(op) ? "bdim" : "gdim";
      os << "long " << fresh(op.getResult(0)) << " = " << builtin << "." << "xyz"[static_cast<unsigned>(dim.getValue())] << ";\n";
      return true;
    }
    if (auto shared = dyn_cast<gpu::DynamicSharedMemoryOp>(op)) {
      names[shared.getResult()] = "arena"; return true;
    }
    if (auto view = dyn_cast<memref::ViewOp>(op)) {
      auto t = view.getType();
      if (t.getRank() != 1 || !t.getElementType().isF32() || !t.getLayout().isIdentity() || name(view.getSource()) != "arena") return false;
      os << "threadgroup float* " << fresh(view.getResult()) << " = reinterpret_cast<threadgroup float*>(arena + " << name(view.getByteShift()) << ");\n";
      return true;
    }
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (load.getIndices().size() != 1 || !load.getType().isF32()) return false;
      os << "float " << fresh(load.getResult()) << " = " << name(load.getMemRef()) << "[" << name(load.getIndices()[0]) << "];\n"; return true;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (store.getIndices().size() != 1 || !store.getValue().getType().isF32()) return false;
      os << name(store.getMemRef()) << "[" << name(store.getIndices()[0]) << "] = " << name(store.getValue()) << ";\n"; return true;
    }
    if (auto gep = dyn_cast<LLVM::GEPOp>(op)) {
      if (!gep.getElemType().isF32() || gep.getIndices().size() != 1 || gep.getDynamicIndices().size() > 1) return false;
      auto offset = gep.getDynamicIndices().empty() ? std::to_string(gep.getRawConstantIndices()[0]) : name(gep.getDynamicIndices()[0]);
      os << "device float* " << fresh(gep.getResult()) << " = reinterpret_cast<device float*>(" << name(gep.getBase()) << ") + " << offset << ";\n"; return true;
    }
    if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
      if (!load.getType().isF32() || load.getVolatile_() || load.getOrdering() != LLVM::AtomicOrdering::not_atomic) return false;
      os << "float " << fresh(load.getResult()) << " = *reinterpret_cast<device float*>(" << name(load.getAddr()) << ");\n"; return true;
    }
    if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
      if (!store.getValue().getType().isF32() || store.getVolatile_() || store.getOrdering() != LLVM::AtomicOrdering::not_atomic) return false;
      os << "*reinterpret_cast<device float*>(" << name(store.getAddr()) << ") = " << name(store.getValue()) << ";\n"; return true;
    }
    if (isa<gpu::BarrierOp>(op)) { os << "threadgroup_barrier(mem_flags::mem_threadgroup);\n"; return true; }
    if (isa<gpu::ReturnOp>(op)) { os << "return;\n"; return true; }
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (branch.getNumResults()) return false;
      bool collective = false;
      branch.walk([&](gpu::BarrierOp) { collective = true; });
      if (collective) return false; // Divergent regions may not hide a barrier.
      os << "if (" << name(branch.getCondition()) << ") {\n";
      for (auto &nested : branch.thenBlock()->without_terminator()) if (!emit(nested)) return false;
      os << "}\n";
      if (!branch.getElseRegion().empty()) {
        os << "else {\n";
        for (auto &nested : branch.elseBlock()->without_terminator()) if (!emit(nested)) return false;
        os << "}\n";
      }
      return true;
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (loop.getUnsignedCmp() || !memory::workgroupUniform(loop.getLowerBound()) || !memory::workgroupUniform(loop.getUpperBound()) || !memory::workgroupUniform(loop.getStep())) return false;
      for (auto [arg, init] : llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs())) {
        auto t = type(arg.getType()); if (t.empty()) return false;
        os << t << " " << fresh(arg) << " = " << name(init) << ";\n";
      }
      auto iv = fresh(loop.getInductionVar());
      os << "for (long " << iv << " = " << name(loop.getLowerBound()) << "; " << iv << " < " << name(loop.getUpperBound()) << "; " << iv << " += " << name(loop.getStep()) << ") {\n";
      for (auto &nested : loop.getBody()->without_terminator()) if (!emit(nested)) return false;
      auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      // Parallel assignment is essential for swaps and multiple recurrences.
      llvm::SmallVector<std::string> temporaries;
      for (Value v : yield.getOperands()) {
        auto temp = "next" + std::to_string(serial++); temporaries.push_back(temp);
        os << type(v.getType()) << " " << temp << " = " << name(v) << ";\n";
      }
      for (auto [arg, temp] : llvm::zip(loop.getRegionIterArgs(), temporaries)) os << name(arg) << " = " << temp << ";\n";
      os << "}\n";
      for (auto [result, arg] : llvm::zip(loop.getResults(), loop.getRegionIterArgs())) names[result] = name(arg);
      return true;
    }
    if (op.getNumResults() != 1 || type(op.getResult(0).getType()).empty()) return false;
    std::string expr;
    if (opname == "math.absf" || opname == "math.exp2")
      expr = (opname == "math.absf" ? "abs(" : "exp2(") + name(op.getOperand(0)) + ")";
    else if (opname == "math.copysign") expr = "copysign(" + name(op.getOperand(0)) + ", " + name(op.getOperand(1)) + ")";
    else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      if (cmp.getPredicate() == arith::CmpIPredicate::eq) expr = name(cmp.getLhs()) + " == " + name(cmp.getRhs());
      else if (cmp.getPredicate() == arith::CmpIPredicate::ult) expr = "(ulong)" + name(cmp.getLhs()) + " < (ulong)" + name(cmp.getRhs());
      else return false;
    } else if (auto cmp = dyn_cast<arith::CmpFOp>(op)) {
      if (cmp.getPredicate() != arith::CmpFPredicate::OLT) return false;
      expr = name(cmp.getLhs()) + " < " + name(cmp.getRhs());
    } else if (auto select = dyn_cast<arith::SelectOp>(op))
      expr = name(select.getCondition()) + " ? " + name(select.getTrueValue()) + " : " + name(select.getFalseValue());
    else if (isa<arith::IndexCastOp, arith::SIToFPOp>(op)) expr = "(" + type(op.getResult(0).getType()) + ")(" + name(op.getOperand(0)) + ")";
    else if (op.getNumOperands() == 2) {
      auto a = name(op.getOperand(0)), b = name(op.getOperand(1));
      auto token = opname == "arith.addi" || opname == "arith.addf" ? "+" : opname == "arith.muli" || opname == "arith.mulf" ? "*" : opname == "arith.subi" || opname == "arith.subf" ? "-" : opname == "arith.divf" ? "/" : "";
      if (*token) expr = a + " " + token + " " + b;
      else if (opname == "arith.remui" || opname == "arith.divui") expr = "(long)((ulong)" + a + (opname == "arith.remui" ? " % " : " / ") + "(ulong)" + b + ")";
      else if (opname == "arith.maxsi") expr = "max(" + a + ", " + b + ")";
      else if (opname == "arith.maxui") expr = "(long)max((ulong)" + a + ", (ulong)" + b + ")";
    }
    if (expr.empty()) return false;
    os << type(op.getResult(0).getType()) << " " << fresh(op.getResult(0)) << " = " << expr << ";\n";
    return true;
  }
public:
  mlir::LogicalResult materialize(mlir::ModuleOp module) {
    using namespace mlir;
    llvm::SmallVector<gpu::GPUFuncOp> kernels;
    module.walk([&](gpu::GPUFuncOp fn) { kernels.push_back(fn); });
    if (kernels.size() != 1 || !kernels[0].isKernel()) return module.emitError("Apple arena MSL requires exactly one GPU kernel");
    auto fn = kernels[0];
    if (fn.getNumArguments() > 31) return fn.emitError("Apple arena MSL exceeds Metal buffer slots");
    auto sizer = fn->getAttrOfType<FlatSymbolRefAttr>("tile.dynamic_shared_size");
    if (!sizer || !module.lookupSymbol<func::FuncOp>(sizer.getValue()) || !fn.getBody().hasOneBlock()) return fn.emitError("Apple arena MSL requires the native dynamic sizing companion");
    os << "#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\nkernel void " << fn.getName() << "(\n";
    unsigned i = 0;
    for (Value arg : fn.getArguments()) {
      auto ptr = dyn_cast<LLVM::LLVMPointerType>(arg.getType());
      if (!(ptr && ptr.getAddressSpace() == 1) && !arg.getType().isIndex()) return fn.emitError("Apple arena MSL admits device pointers and index launch scalars only");
      os << (ptr ? "device uchar* " : "constant long& ") << fresh(arg) << " [[buffer(" << i++ << ")]],\n";
    }
    os << "threadgroup uchar* arena [[threadgroup(0)]],\nuint3 tid [[thread_position_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],\nuint3 bdim [[threads_per_threadgroup]], uint3 gdim [[threadgroups_per_grid]]) {\n";
    unsigned arenas = 0;
    fn.walk([&](gpu::DynamicSharedMemoryOp) { ++arenas; });
    if (arenas != 1) return fn.emitError("Apple arena MSL requires one dynamic storage slot");
    for (auto &op : fn.getBody().front()) if (!emit(op)) return op.emitError("unsupported operation in Apple arena MSL materialization");
    os << "}\n";
    module->setAttr("tessera.apple.arena_msl", StringAttr::get(module.getContext(), text));
    module->setAttr("tessera.apple.arena_sizer", sizer);
    module->setAttr("tessera.apple.arena_slot", IntegerAttr::get(IntegerType::get(module.getContext(), 64), 0));
    return success();
  }
};
}
#endif
