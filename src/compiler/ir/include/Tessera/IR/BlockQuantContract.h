// Shared semantic admission for standalone and composed K16 quantization.
#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include <array>
#include <climits>

namespace tessera::block_quant {
struct Profile {
  mlir::func::FuncOp graph;
  int64_t rows, columns, groups;
};

inline mlir::FailureOr<Profile> inspect(mlir::ModuleOp module) {
  using namespace mlir;
  auto refuse = [&]() -> FailureOr<Profile> {
    module.emitError("native K16 quantization requires its explicit BF16/max/scale/normalize/E4M3 Graph contract");
    return failure();
  };
  auto target = module->getAttrOfType<StringAttr>("tessera.target");
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  if (!isa<UnitAttr>(module->getAttr("tessera.native.block_quant")) ||
      !target || target.getValue() != "rocm_gfx1201" ||
      !arch || arch.getValue() != "gfx1201" || failed(verify(module)))
    return refuse();
  for (auto attr : module->getAttrs()) {
    StringRef key = attr.getName().getValue();
    if (key != "tessera.target" && key != "tessera.arch" &&
        key != "tessera.native.block_quant" && key != "tessera.ir.version" &&
        key != "tessera.frontend.authority" && key != "tessera.launch_bindings") return refuse();
  }
  if (auto raw=module->getAttr("tessera.launch_bindings")) {
    auto names=dyn_cast<ArrayAttr>(raw);
    if (!names || names.size()!=3) return refuse();
    for (unsigned i=0;i<3;++i) {
      auto name=dyn_cast<StringAttr>(names[i]);
      if (!name || name.getValue().empty()) return refuse();
      for (unsigned j=0;j<i;++j) if (names[i]==names[j]) return refuse();
    }
  }
  auto roots = module.getOps<func::FuncOp>();
  if (llvm::range_size(roots) != 1) return refuse();
  auto fn = *roots.begin();
  if (fn.isDeclaration() || !fn.getBody().hasOneBlock() ||
      fn.getNumArguments() != 1 || fn.getNumResults() != 2) return refuse();
  for (auto attr : fn->getAttrs()) {
    StringRef key = attr.getName().getValue();
    if (key != "sym_name" && key != "function_type" && key != "sym_visibility" &&
        key != "tessera.frontend.authority" &&
        key != "tessera.structured_cfg.schema" && key != "tessera.structured_cfg.digest" &&
        key != "tessera.structured_cfg.blocks") return refuse();
  }
  auto input = dyn_cast<RankedTensorType>(fn.getArgumentTypes()[0]);
  if (!input || input.getRank() != 2 || !input.hasStaticShape() ||
      input.getEncoding() || !input.getElementType().isBF16()) return refuse();
  int64_t m = input.getDimSize(0), k = input.getDimSize(1);
  if (m <= 0 || m > INT32_MAX || k <= 0 || k > INT32_MAX || k % 16 ||
      m > INT32_MAX / (k / 16)) return refuse();
  int64_t groups = m * (k / 16);
  SmallVector<Operation *> ops;
  for (Operation &op : fn.getBody().front().without_terminator()) ops.push_back(&op);
  const char *names[] = {
    "tessera.reshape", "tessera.cast", "tessera.absolute", "tessera.reduce",
    "arith.constant", "tessera.div", "arith.constant", "tessera.maximum",
    "tessera.expand", "tessera.div", "tessera.cast", "tessera.reshape", "tessera.reshape"};
  if (ops.size() != std::size(names)) return refuse();
  for (unsigned i=0;i<ops.size();++i)
    if (ops[i]->getName().getStringRef() != names[i] ||
        ops[i]->getNumResults() != 1 || ops[i]->getNumRegions()) return refuse();
  auto f32 = Float32Type::get(module.getContext());
  auto f8 = Float8E4M3FNType::get(module.getContext());
  auto blocks = RankedTensorType::get({groups,16}, input.getElementType());
  auto wide = RankedTensorType::get({groups,16}, f32);
  auto scale = RankedTensorType::get({groups,1}, f32);
  auto encoded = RankedTensorType::get({groups,16}, f8);
  auto codes = RankedTensorType::get({m,k}, f8);
  auto scales = RankedTensorType::get({m,k/16}, f32);
  Type expected[] = {blocks,wide,wide,scale,scale,scale,scale,scale,wide,wide,encoded,codes,scales};
  for (unsigned i=0;i<ops.size();++i)
    if (ops[i]->getResult(0).getType() != expected[i]) return refuse();
  std::array<SmallVector<Value>,13> operands = {
    SmallVector<Value>{fn.getArgument(0)}, {ops[0]->getResult(0)},
    {ops[1]->getResult(0)}, {ops[2]->getResult(0)}, {},
    {ops[3]->getResult(0),ops[4]->getResult(0)}, {},
    {ops[5]->getResult(0),ops[6]->getResult(0)}, {ops[7]->getResult(0)},
    {ops[1]->getResult(0),ops[8]->getResult(0)}, {ops[9]->getResult(0)},
    {ops[10]->getResult(0)}, {ops[7]->getResult(0)}};
  for (unsigned i=0;i<ops.size();++i) {
    if (!llvm::equal(ops[i]->getOperands(),operands[i])) return refuse();
    for (auto attr : ops[i]->getAttrs()) {
      StringRef key = attr.getName().getValue();
      if (key == "tessera.effect_kind" &&
          attr.getValue() == StringAttr::get(module.getContext(),"pure")) continue;
      if ((i==0 || i==8 || i==11 || i==12) && key=="tessera.aliasing" &&
          attr.getValue()==StringAttr::get(module.getContext(),"operand_0")) continue;
      if ((i==0 || i==8 || i==11 || i==12) && key=="shape") {
        auto shape = dyn_cast<ArrayAttr>(attr.getValue());
        auto result = cast<RankedTensorType>(expected[i]);
        if (!shape || shape.size() != 2) return refuse();
        for (unsigned axis=0;axis<2;++axis) {
          auto dim = dyn_cast<IntegerAttr>(shape[axis]);
          if (!dim || dim.getInt() != result.getDimSize(axis)) return refuse();
        }
        continue;
      }
      if ((i==1 || i==10) && key=="dtype" &&
          attr.getValue()==StringAttr::get(module.getContext(),i==1?"fp32":"fp8_e4m3")) continue;
      if (i==3 && (key=="kind" || key=="axis" || key=="keepdims")) continue;
      if ((i==4 || i==6) && key=="value") continue;
      return refuse();
    }
  }
  auto kind = ops[3]->getAttrOfType<StringAttr>("kind");
  auto axis = ops[3]->getAttrOfType<IntegerAttr>("axis");
  auto keep = ops[3]->getAttrOfType<BoolAttr>("keepdims");
  if (!kind || kind.getValue()!="max" || !axis || (axis.getInt()!=1 && axis.getInt()!=-1) ||
      !keep || !keep.getValue()) return refuse();
  for (unsigned index : {4u,6u}) {
    auto value = ops[index]->getAttrOfType<DenseElementsAttr>("value");
    if (!value || !value.isSplat() || value.getElementType()!=f32 ||
        value.getSplatValue<APFloat>().bitcastToAPInt().getZExtValue() !=
          (index==4 ? uint64_t(0x43e00000) : uint64_t(0x00800000))) return refuse();
  }
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
  if (!ret || ret.getNumOperands()!=2 ||
      ret.getOperand(0)!=ops[11]->getResult(0) || ret.getOperand(1)!=ops[12]->getResult(0))
    return refuse();
  return Profile{fn,m,k,groups};
}

}
