// Native bounded sequence projection shared by forward and reverse Graph AD.
#ifndef TESSERA_NATIVE_ATTENTION_PROJECTION_H
#define TESSERA_NATIVE_ATTENTION_PROJECTION_H
#include "../../compiler/ir/AttentionADContract.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
namespace tessera {
inline mlir::LogicalResult projectBoundedAttentionEntry(mlir::ModuleOp module,mlir::func::FuncOp fn) {
  auto request=module->getAttrOfType<mlir::DenseI64ArrayAttr>("tessera.attention_sequence_bounds");
  auto target=module->getAttrOfType<mlir::StringAttr>("tessera.target");
  auto arch=module->getAttrOfType<mlir::StringAttr>("tessera.arch");
  if (!request || request.size()!=2 || request[0]<=0 || request[1]<=0 ||
      !target || target.getValue()!="nvidia_sm120" || !arch || arch.getValue()!="sm_120" ||
      module->hasAttr("tessera.attention_shape_bounds") || !fn.getBody().hasOneBlock() ||
      fn.getBody().front().getOperations().size()!=2)
    return fn.emitError("bounded attention projection requires one isolated SM120 entry and positive Sq/Sk capacities");
  auto *op=&fn.getBody().front().front();
  if (op->getName().getStringRef()!="tessera.flash_attn" || !denseAttentionAD(op,true,true))
    return fn.emitError("bounded attention projection requires a supported static attention Graph");
  auto ret=mlir::dyn_cast<mlir::func::ReturnOp>(fn.getBody().front().back());
  if (!ret || ret.getOperands()!=op->getResults() || fn.getNumArguments()!=op->getNumOperands())
    return fn.emitError("bounded attention projection must retain every result and input role");
  llvm::SmallDenseSet<unsigned> inputs;
  for (auto value:op->getOperands()) {
    auto argument=mlir::dyn_cast<mlir::BlockArgument>(value);
    if (!argument || argument.getOwner()!=&fn.getBody().front() ||
        !inputs.insert(argument.getArgNumber()).second ||
        fn.getArgAttr(argument.getArgNumber(),"tessera.layout"))
      return fn.emitError("bounded attention projection requires distinct direct input roles without layout overrides");
  }
  auto q=mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto k=mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  auto v=mlir::cast<mlir::RankedTensorType>(op->getOperand(2).getType());
  if (q.getDimSize(2)>request[0] || k.getDimSize(2)>request[1])
    return fn.emitError("attention trace exceeds requested sequence capacity");
  auto project=[&](mlir::Type type,llvm::ArrayRef<unsigned> axes) {
    auto tensor=mlir::cast<mlir::RankedTensorType>(type);
    llvm::SmallVector<int64_t> shape(tensor.getShape());
    for (unsigned axis:axes) shape[axis]=mlir::ShapedType::kDynamic;
    return mlir::RankedTensorType::get(shape,tensor.getElementType(),tensor.getEncoding());
  };
  for (unsigned role=0;role<3;++role)
    op->getOperand(role).setType(project(op->getOperand(role).getType(),{2}));
  if (op->getNumOperands()==4) {
    auto bias=mlir::cast<mlir::RankedTensorType>(op->getOperand(3).getType());
    llvm::SmallVector<unsigned> axes;
    bool logical=bias.getShape()==llvm::ArrayRef<int64_t>({q.getDimSize(0),q.getDimSize(1),q.getDimSize(2),k.getDimSize(2)});
    if (logical || bias.getDimSize(2)!=1) axes.push_back(2);
    if (logical || bias.getDimSize(3)!=1) axes.push_back(3);
    op->getOperand(3).setType(project(bias,axes));
  }
  for (auto result:op->getResults()) result.setType(project(result.getType(),{2}));
  mlir::OpBuilder builder(module.getContext());
  fn.setFunctionType(builder.getFunctionType(fn.getBody().front().getArgumentTypes(),op->getResultTypes()));
  module->setAttr("tessera.attention_shape_bounds",builder.getDenseI64ArrayAttr(
      {q.getDimSize(0),q.getDimSize(1),k.getDimSize(1),request[0],request[1],q.getDimSize(3),v.getDimSize(3)}));
  module->removeAttr("tessera.attention_sequence_bounds");
  return mlir::success();
}

} // namespace tessera
#endif
